/**
 * GPU 3-Opt Single Tour Kernel (`naive` per-route transfer baseline)
 * ==================================================================
 *
 * Mirrors `two_opt_single.cu` (`two_opt_kernel`):
 *   1. Processes a single tour on a 1-block grid `(1,)`.
 *   2. Shares the exact neighborhood scan, block reduction, and cooperative
 *      reconnection from `three_opt_common.cuh`.
 *   3. Kept as the per-route `CPU -> GPU -> CPU` baseline parallel to
 *      `LocalSearchGPUTwoOpt(mode="naive")`.
 */

#include "three_opt_common.cuh"

extern "C" __global__ void three_opt_kernel(
    int *__restrict__ tour,             // [n] single tour
    const int *__restrict__ distances,  // [dist_stride, dist_stride]
    const int n,                        // Route length
    const int dist_stride,              // Distance matrix row stride / problem dimension
    const int is_closed,                // 0 = open cyclic TSP, 1 = depot-bounded CVRP
    const int use_shared_tour,          // 1 = cache tour & edge costs in shared memory
    const int *__restrict__ candidates, // [dist_stride, cand_k] sorted k-NN table, or nullptr
    const int cand_k,                   // 0 = exhaustive scan, else neighbors per node
    const int inner_k,                  // Inner loop bound in [1, cand_k]
    int *__restrict__ positions,        // [n] inverse-map scratch (candidate mode)
    int *__restrict__ edge_costs_global // [n] edge-cost scratch (when !use_shared_tour)
)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ int shared_mem[];
    const bool shmem_tour = use_shared_tour != 0;
    const bool closed = is_closed != 0;
    const bool use_candidates = !closed && candidates != nullptr && cand_k > 0 && positions != nullptr;

    int *s_deltas = shared_mem;
    int *s_i = &s_deltas[block_size];
    int *s_j = &s_i[block_size];
    int *s_k = &s_j[block_size];
    int *s_case = &s_k[block_size];

    int *tour_buf = shmem_tour ? &s_case[block_size] : tour;
    int *edge_buf = shmem_tour ? &tour_buf[n] : edge_costs_global;

    if (shmem_tour)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            tour_buf[idx] = tour[idx];
        }
        __syncthreads();
    }

    three_opt_scan_neighborhood(
        tour_buf, edge_buf, distances, nullptr,
        n, dist_stride, 0, true, closed,
        use_candidates, candidates, cand_k, inner_k, positions,
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
            tour[idx] = tour_buf[idx];
        }
    }
}
