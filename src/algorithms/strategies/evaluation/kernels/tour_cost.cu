/**
 * GPU Batch Hamiltonian Tour Cost Kernel (`DP11-15`, `DP10-19`, `DP17-7`)
 *
 * Computes tour costs for multiple TSP/ATSP tours in parallel.
 * Supports both closed tours (`return_to_depot != 0`, summing `n` edges including
 * `tour[n - 1] -> tour[0]`) and open tours (`return_to_depot == 0`, summing `n - 1`
 * edges and omitting the closing arc).
 *
 * Algorithm:
 *   1. Each block computes cost for one tour.
 *   2. Threads cooperatively sum edge costs via coalesced block-stride loop.
 *   3. Two-level warp-shuffle + shared-memory block reduction (`block_reductions.cuh`).
 *
 * On-the-fly coordinate mode (`DP10-19`): when `distances == nullptr` the kernel
 * computes each edge with `gpu_pairwise_distance` from `coords` + `edge_type_code`.
 */

#include "block_reductions.cuh"
#include "pairwise_device.cuh"

extern "C" __global__ void tour_cost_kernel(
    const int *tours,     // Flattened: num_tours * n (WITHOUT duplicate depot)
    const int *distances, // n * n distance matrix (int32), or nullptr for coordinate mode
    const double *coords, // Row-major coordinates, 2n or 3n doubles (ignored if distances != nullptr)
    long long *costs,     // Output: num_tours costs (int64 for overflow safety)
    int n,                // Number of nodes per tour (without duplicate depot)
    int num_tours,        // Number of tours in batch
    int edge_type_code,   // `GpuEdgeTypeCode` for the coordinate mode
    int return_to_depot   // 1 for closed tour, 0 for open tour
)
{
    extern __shared__ long long s_partials[];

    int tour_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (tour_idx >= num_tours)
    {
        return;
    }

    const int *tour = tours + (tour_idx * n);
    int num_edges = (return_to_depot != 0) ? n : ((n > 0) ? (n - 1) : 0);

    long long partial_cost = 0;

    // Coalesced block-stride loop (DP11-15, EVAL-A): threads read adjacent tour elements.
    for (int i = tid; i < num_edges; i += block_size)
    {
        int from_node = tour[i];
        int to_node = (i + 1 < n) ? tour[i + 1] : tour[0];
        partial_cost += (long long)gpu_pairwise_distance(
            from_node, to_node, distances, coords, n, edge_type_code);
    }

    // Two-level warp-shuffle + shared-memory block reduction (`Task 17.21`).
    long long total_cost = block_reduce_sum<long long>(partial_cost, s_partials);
    if (tid == 0)
    {
        costs[tour_idx] = total_cost;
    }
}
