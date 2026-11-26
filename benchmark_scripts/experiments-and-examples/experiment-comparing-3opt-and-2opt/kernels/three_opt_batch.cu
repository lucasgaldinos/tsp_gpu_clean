/**
 * GPU 3-Opt Batch Processing Kernel (Single Steepest Move per Pass)
 * =================================================================
 *
 * Mirrors `two_opt_batch.cu` (`two_opt_batch_kernel`):
 *   1. One CUDA block processes one tour independently.
 *   2. Parallel scan of the 3-opt neighborhood (candidate-list pruned or exhaustive)
 *      via `three_opt_common.cuh`.
 *   3. Parallel block reduction finds the single steepest improving move (min delta).
 *   4. Cooperative in-place 3-opt reconnection applies that single move.
 *   5. Writes back the improved tour when cached in shared memory.
 */

#include "three_opt_common.cuh"

extern "C" __global__ void three_opt_batch_kernel(
    int *__restrict__ population,       // [population_size, n]
    const int *__restrict__ distances,  // [dist_stride, dist_stride], or nullptr for coordinate mode
    const double *__restrict__ coords,  // Row-major coordinates (ignored if distances != nullptr)
    const int n,                        // Route length (number of stops per row)
    const int dist_stride,              // Distance matrix row stride / problem dimension
    const int population_size,          // Number of tours in the batch
    const int edge_type_code,           // GpuEdgeTypeCode for coordinate mode
    const int is_closed,                // 0 = open cyclic TSP, 1 = depot-bounded CVRP
    const int use_shared_tour,          // 1 = cache tour & edge costs in shared memory
    const int *__restrict__ candidates, // [dist_stride, cand_k] sorted k-NN table, or nullptr
    const int cand_k,                   // 0 = exhaustive scan, else neighbors per node
    const int inner_k,                  // Inner loop bound in [1, cand_k] for 3-opt probes
    int *__restrict__ positions,        // [population_size, n] inverse-map scratch (candidate mode)
    int *__restrict__ edge_costs_global // [population_size, n] edge-cost scratch (when !use_shared_tour)
)
{
    const int route_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (route_id >= population_size)
        return;

    extern __shared__ int shared_mem[];
    const bool use_matrix = distances != nullptr;
    const bool shmem_tour = use_shared_tour != 0;
    const bool closed = is_closed != 0;
    const bool use_candidates = !closed && candidates != nullptr && cand_k > 0 && positions != nullptr;

    int *s_deltas = shared_mem;
    int *s_i = &s_deltas[block_size];
    int *s_j = &s_i[block_size];
    int *s_k = &s_j[block_size];
    int *s_case = &s_k[block_size];

    int *tour_global = &population[(size_t)route_id * (size_t)n];
    int *tour_buf = shmem_tour ? &s_case[block_size] : tour_global;
    int *edge_buf = shmem_tour
                        ? &tour_buf[n]
                        : (edge_costs_global + (size_t)route_id * (size_t)n);
    int *pos = use_candidates ? (positions + (size_t)route_id * (size_t)n) : nullptr;

    if (shmem_tour)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            tour_buf[idx] = tour_global[idx];
        }
        __syncthreads();
    }

    three_opt_scan_neighborhood(
        tour_buf, edge_buf, distances, coords,
        n, dist_stride, edge_type_code, use_matrix, closed,
        use_candidates, candidates, cand_k, inner_k, pos,
        s_deltas, s_i, s_j, s_k, s_case, tid, block_size);

    three_opt_reduce_block_min(s_deltas, s_i, s_j, s_k, s_case, tid, block_size);

    if (s_deltas[0] < 0)
    {
        three_opt_apply_reconnection(tour_buf, s_i[0], s_j[0], s_k[0], s_case[0], tid, block_size);
    }

    if (shmem_tour)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            tour_global[idx] = tour_buf[idx];
        }
    }
}
