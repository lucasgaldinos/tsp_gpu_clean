/**
 * Multi-Move Disjoint-Segment Batch 3-Opt CUDA Kernel
 * ===================================================
 *
 * Mirrors `two_opt_multi_move.cu` (`two_opt_multi_move_batch_kernel`):
 *   1. One CUDA block processes one tour independently.
 *   2. Parallel scan of the 3-opt neighborhood via `three_opt_common.cuh`, where
 *      each thread retains its own steepest improving candidate.
 *   3. Thread 0 greedily selects non-conflicting improving candidates (Rocki &
 *      Suda, 2013), publishing each selected move through shared memory while the
 *      whole block applies the in-place 3-opt reconnection cooperatively.
 *   4. Any surviving candidate whose cut span `[i_t, k_t]` overlaps `[mi, mk]`
 *      (i.e. `!(k_t < mi || mk < i_t)`) is invalidated in parallel.
 */

#include "three_opt_common.cuh"

extern "C" __global__ void three_opt_multi_move_batch_kernel(
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

    __shared__ int s_sel_i;
    __shared__ int s_sel_j;
    __shared__ int s_sel_k;
    __shared__ int s_sel_case;

    const int max_moves = block_size < (n / 2) ? block_size : (n / 2);

    for (int applied = 0; applied < max_moves; ++applied)
    {
        if (tid == 0)
        {
            int best_t = -1;
            int best_d = 0;
            for (int t = 0; t < block_size; ++t)
            {
                if (s_deltas[t] < best_d)
                {
                    best_d = s_deltas[t];
                    best_t = t;
                }
            }
            if (best_t < 0)
            {
                s_sel_case = 0;
            }
            else
            {
                s_sel_i = s_i[best_t];
                s_sel_j = s_j[best_t];
                s_sel_k = s_k[best_t];
                s_sel_case = s_case[best_t];
                s_deltas[best_t] = 0;
            }
        }
        __syncthreads();

        const int move_case = s_sel_case;
        if (move_case == 0)
            break;

        const int mi = s_sel_i;
        const int mj = s_sel_j;
        const int mk = s_sel_k;

        three_opt_apply_reconnection(tour_buf, mi, mj, mk, move_case, tid, block_size);

        // Two moves are strictly independent iff (k_t < mi || mk < i_t) because tour[0]
        // is invariant (seg_start >= 1) and neither move's modified range [i+1, k]
        // touches the other's 6 cut endpoints {i, i+1, j, j+1, k, (k+1)%n}.
        if (s_deltas[tid] < 0)
        {
            const int ti = s_i[tid];
            const int tk = s_k[tid];
            if (!(tk < mi || mk < ti))
            {
                s_deltas[tid] = 0;
            }
        }
        __syncthreads();
    }

    if (shmem_tour)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            tour_global[idx] = tour_buf[idx];
        }
    }
}
