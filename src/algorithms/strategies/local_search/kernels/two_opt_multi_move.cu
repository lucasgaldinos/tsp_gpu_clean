/**
 * Multi-Move Disjoint-Segment Batch 2-Opt CUDA Kernel
 * ====================================================
 *
 * Evaluates all O(n^2) 2-opt moves across threads in the block, then greedily
 * applies multiple non-overlapping (disjoint) improving segment reversals
 * in the same iteration pass.
 *
 * Memory layout (`Task 10.18`):
 *   - `tour_tile_size == 0`: the whole tour lives in `s_tour[n]`, loaded once.
 *   - `tour_tile_size > 0`: only one `j`-window of `tour_tile_size` positions plus one
 *     extra slot (`s_tour[tour_tile_size + 1]`) is cached. The block owns rows
 *     `[i_base, i_base + block_size)` and walks the windows from `i_base + 2`, so 100%
 *     of the inner-loop `j` reads hit shared memory instead of falling back to global.
 *
 * References:
 *   - Rocki, K., & Suda, R. (2013). "High Performance GPU Accelerated Local
 *     Optimization in TSP." IEEE IPDPSW, pp. 1788-1796. (Applying multiple
 *     non-conflicting 2-opt moves from a single parallel neighborhood scan).
 *   - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
 *     Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
 *   - O'Neil, M. A., & Burtscher, M. (2015). "Rethinking the Parallelization of
 *     Random-Restart Hill Climbing: A Case Study in Optimizing a 2-Opt TSP Solver
 *     for GPU Execution." GPGPU-8, pp. 99-108.
 */

#include "two_opt_common.cuh"

extern "C" __global__ void two_opt_multi_move_batch_kernel(
    int *__restrict__ population,      // [population_size, n]
    const int *__restrict__ distances, // [n, n], or nullptr for coordinate mode
    const double *__restrict__ coords, // Row-major coordinates (ignored if distances != nullptr)
    const int n,
    const int population_size,
    const int edge_type_code,           // `GpuEdgeTypeCode` for the coordinate mode
    const int tour_tile_size,           // 0 = full-tour shared memory, else tile width (2048)
    const int *__restrict__ candidates, // [n, cand_k] nearest-neighbour table, or nullptr
    const int cand_k,                   // 0 = exhaustive scan, else neighbours per node
    int *__restrict__ positions         // [population_size, n] inverse-map scratch
)
{
    int route_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (route_id >= population_size)
        return;

    extern __shared__ int shared_mem[];
    const bool tiled = tour_tile_size > 0;
    const bool use_matrix = distances != nullptr;
    // Candidate mode never caches the tour: the `j` scan becomes `Theta(n * k)` over
    // `pos[candidates[tour[i] * cand_k + r]]`, which needs global-memory tour reads
    // (`DP10-20`). It is only enabled on the untiled path.
    const bool use_candidates = !tiled && candidates != nullptr && cand_k > 0 && positions != nullptr;

    int *s_tour = shared_mem;                                         // [n] or [tour_tile_size + 1]
    int *s_deltas = tiled ? &s_tour[tour_tile_size + 1] : &s_tour[n]; // [block_size]
    int *s_swap_i = &s_deltas[block_size];                            // [block_size]
    int *s_swap_j = &s_swap_i[block_size];                            // [block_size]

    int *tour = &population[route_id * n];
    if (!tiled && !use_candidates)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            s_tour[idx] = tour[idx];
        }
        __syncthreads();
    }

    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;

    int tile_start = 0;
    int tile_end = -1;

    if (use_candidates)
    {
        // Inverse tour map: `pos[tour[idx]] = idx` in one O(n) pass (`DP10-20`).
        int *pos = positions + (route_id * n);
        for (int idx = tid; idx < n; idx += block_size)
        {
            pos[tour[idx]] = idx;
        }
        __syncthreads();

        for (int i = tid; i < n - 2; i += block_size)
        {
            // Hoisted row endpoints and current edge cost (`Task 10.16`).
            int a = tour[i];
            int b = tour[i + 1];
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)n : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)n : nullptr;
            int d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, n, edge_type_code);

            for (int r = 0; r < cand_k; ++r)
            {
                int j = pos[candidates[(size_t)a * (size_t)cand_k + (size_t)r]];
                if (j <= i + 1 || (i == 0 && j == n - 1))
                    continue;
                int c = tour[j];
                int d = tour[(j + 1) % n];
                two_opt_consider(
                    i, j,
                    two_opt_delta(a, b, d_ab, c, d, row_a, row_b, distances, coords, n, edge_type_code,
                                  use_matrix),
                    s_deltas, s_swap_i, s_swap_j, tid);
            }
        }
    }
    else if (!tiled)
    {
        // Full-tour scan: upper triangle only, hoisted row endpoints + carry-over.
        for (int i = tid; i < n - 2; i += block_size)
        {
            int a = s_tour[i];
            int b = s_tour[i + 1];
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)n : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)n : nullptr;
            int d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, n, edge_type_code);
            // Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`
            int c = s_tour[i + 2];

            for (int j = i + 2; j < n; ++j)
            {
                int d = s_tour[(j + 1) % n];
                if (!(i == 0 && j == n - 1))
                {
                    two_opt_consider(
                        i, j,
                        two_opt_delta(a, b, d_ab, c, d, row_a, row_b, distances, coords, n, edge_type_code,
                                      use_matrix),
                        s_deltas, s_swap_i, s_swap_j, tid);
                }
                c = d;
            }
        }
    }
    else
    {
        // Windowed scan (`Task 10.18` refinement): the block owns the rows
        // `[i_base, i_base + block_size)` and walks the `j` windows from `i_base + 2`,
        // so every inner-loop `j` read hits shared memory `s_tour[j - window_start]`.
        const int stride = tour_tile_size;
        for (int i_base = 0; i_base < n - 2; i_base += block_size)
        {
            const int row = i_base + tid;
            const bool active = row < n - 2;
            // Hoisted row endpoints stay in registers across every `j` window.
            int a = 0;
            int b = 0;
            int d_ab = 0;
            const int *row_a = nullptr;
            const int *row_b = nullptr;
            if (active)
            {
                a = tour[row];
                b = tour[row + 1];
                row_a = use_matrix ? distances + (size_t)a * (size_t)n : nullptr;
                row_b = use_matrix ? distances + (size_t)b * (size_t)n : nullptr;
                d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, n, edge_type_code);
            }

            for (int window_start = i_base + 2; window_start < n; window_start += stride)
            {
                int window_end = window_start + stride;
                if (window_end > n)
                    window_end = n;
                const int window_len = window_end - window_start;

                __syncthreads();
                for (int idx = window_start + tid; idx < window_end; idx += block_size)
                {
                    s_tour[idx - window_start] = tour[idx];
                }
                if (tid == 0)
                {
                    // One extra slot holds the `j + 1` endpoint; it wraps to `tour[0]`
                    // on the last window (`tour[(idx) % n]`).
                    s_tour[window_len] = tour[window_end % n];
                }
                __syncthreads();

                int j_begin = row + 2 > window_start ? row + 2 : window_start;
                if (active && j_begin < window_end)
                {
                    int c = s_tour[j_begin - window_start];
                    for (int j = j_begin; j < window_end; ++j)
                    {
                        int d = s_tour[j - window_start + 1];
                        if (!(row == 0 && j == n - 1))
                        {
                            two_opt_consider(
                                row, j,
                                two_opt_delta(a, b, d_ab, c, d, row_a, row_b, distances, coords, n, edge_type_code,
                                              use_matrix),
                                s_deltas, s_swap_i, s_swap_j, tid);
                        }
                        c = d;
                    }
                }
                __syncthreads();
            }
        }
        // The window is stale once every block row has been scanned: the reversal below
        // reads the global tour (an empty window makes `TOUR_AT` do exactly that).
        tile_start = 1;
        tile_end = 0;
    }
    __syncthreads();

    // Thread 0 greedily selects and applies all mutually disjoint improving moves
    // collected across the threads of the block in steepest-descent order.
    //
    // The bound is `min(block_size, n / 2)` -- at most one disjoint reversal per two
    // tour positions and at most one candidate per thread -- instead of a hardcoded
    // 16 (`BUG-20` / `DP11-2`). No bookkeeping arrays are needed: an applied move has
    // its own delta zeroed, and every remaining candidate whose reversed segment
    // touches the applied one is zeroed in place as soon as the move is applied.
    if (tid == 0)
    {
        // Every thread contributes at most one candidate, so `block_size` bounds the
        // number of selectable moves; a reversal always covers at least two positions,
        // so `n / 2` bounds the number of disjoint reversals a tour can absorb.
        const int max_moves = block_size < (n / 2) ? block_size : (n / 2);

        for (int applied = 0; applied < max_moves; applied++)
        {
            int best_t = -1;
            int best_d = 0;
            for (int t = 0; t < block_size; t++)
            {
                if (s_deltas[t] < best_d)
                {
                    best_d = s_deltas[t];
                    best_t = t;
                }
            }

            if (best_t < 0)
                break;

            const int swap_i = s_swap_i[best_t];
            const int swap_j = s_swap_j[best_t];
            s_deltas[best_t] = 0; // Mark consumed

            int left = swap_i + 1;
            int right = swap_j;
            if (use_candidates)
            {
                // Candidate mode never cached the tour: edit global memory directly.
                while (left < right)
                {
                    int tmp = tour[left];
                    tour[left] = tour[right];
                    tour[right] = tmp;
                    left++;
                    right--;
                }
            }
            else
            {
                while (left < right)
                {
                    int tmp = TOUR_AT(left);
                    TOUR_SET(left, TOUR_AT(right));
                    TOUR_SET(right, tmp);
                    left++;
                    right--;
                }
            }

            // Reversed segment of the move just applied: tour positions
            // [seg_start, seg_end] = [swap_i + 1, swap_j]. A reversal only permutes
            // positions inside that segment, so a candidate whose four break positions
            // stay outside it keeps a valid delta; every other candidate is dropped.
            const int seg_start = swap_i + 1;
            const int seg_end = swap_j;
            for (int t = 0; t < block_size; t++)
            {
                if (s_deltas[t] >= 0)
                    continue;
                int ms = s_swap_i[t] + 1;
                int me = s_swap_j[t];
                if (ms > me)
                {
                    const int tmp = ms;
                    ms = me;
                    me = tmp;
                }
                if (!(me + 1 < seg_start || seg_end + 1 < ms))
                {
                    s_deltas[t] = 0; // Conflicting candidate eliminated on the fly
                }
            }
        }
    }
    __syncthreads();

    // Write back to global memory (the tiled and candidate paths already wrote through)
    if (!tiled && !use_candidates)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            tour[idx] = s_tour[idx];
        }
    }
}
