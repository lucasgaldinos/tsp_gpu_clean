/**
 * Multi-Move Disjoint-Segment Batch 2-Opt CUDA Kernel
 * ====================================================
 *
 * Evaluates all O(n^2) 2-opt moves across threads in the block, then greedily
 * applies multiple non-overlapping (disjoint) improving segment reversals
 * in the same iteration pass.
 *
 * Candidate bound (`DP11-18` refinement, `BUG-29`): each thread keeps only its own
 * steepest candidate (`two_opt_consider`), so a pass offers at most
 * `min(block_size, n / 2)` moves to the greedy disjoint selection (Rocki & Suda, 2013).
 * The CPU `multi_move` reference scans every pair and therefore exposes all improving
 * moves of the pass; this kernel trades that breadth for O(block_size) bookkeeping.
 *
 * Reversal ownership (`DP11-18`, `2OPT-A`): only the selection is serial. Thread 0
 * publishes the winning segment through shared memory and the whole block reverses it
 * cooperatively (`for (k = tid; k < num_swaps; k += block_size)`), so the segment
 * reversal never runs on a single thread.
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
    const int *__restrict__ distances, // [dist_stride, dist_stride], or nullptr for coordinate mode
    const double *__restrict__ coords, // Row-major coordinates (ignored if distances != nullptr)
    const int n,
    const int dist_stride,             // Distance matrix row stride / problem dimension
    const int population_size,
    const int edge_type_code,           // `GpuEdgeTypeCode` for the coordinate mode
    const int is_closed,                // 1 = closed depot-bounded route [0, ..., 0] (j <= n - 2)
    const int is_cyclic,                // 1 = cyclic wrap-around edge (tour[n - 1], tour[0])
    const int fixed_start,              // 1 = tour[0] is fixed (no prefix reversal i == -1)
    const int tour_tile_size,           // 0 = full-tour shared memory, else tile width (2048)
    const int *__restrict__ candidates, // [dist_stride, cand_k] nearest-neighbour table, or nullptr
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
    const bool closed = is_closed != 0;
    const bool cyclic = !closed && (is_cyclic != 0);
    const bool allow_prefix = !closed && !cyclic && (fixed_start == 0);
    // Candidate mode never caches the tour: the `j` scan becomes `Theta(n * k)` over
    // `pos[candidates[tour[i] * cand_k + r]]`, which needs global-memory tour reads
    // (`DP10-20`). Disabled when tiled, closed, or non-cyclic (`BUG-107`).
    const bool use_candidates = !tiled && !closed && cyclic && candidates != nullptr && cand_k > 0 && positions != nullptr;

    const int i_start = allow_prefix ? -1 : 0;
    const int i_end = closed ? (n - 3) : (n - 2);
    const int j_end = closed ? (n - 1) : n;

    int *s_tour = shared_mem;                                         // [n] or [tour_tile_size + 1]
    // Candidate mode never caches the tour, so its counters start at the first slot and
    // the launch does not reserve `n` unused ints (`BUG-27`).
    int *s_deltas = use_candidates
                        ? shared_mem
                        : (tiled ? &s_tour[tour_tile_size + 1] : &s_tour[n]); // [block_size]
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
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)dist_stride : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)dist_stride : nullptr;
            int d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, dist_stride, edge_type_code);

            for (int r = 0; r < cand_k; ++r)
            {
                int j = pos[candidates[(size_t)a * (size_t)cand_k + (size_t)r]];
                if (j <= i + 1 || (i == 0 && j == n - 1))
                    continue;
                int c = tour[j];
                int d = tour[(j + 1) % n];
                two_opt_consider(
                    i, j,
                    two_opt_delta(a, b, d_ab, c, d, row_a, row_b, distances, coords, dist_stride, edge_type_code,
                                  use_matrix),
                    s_deltas, s_swap_i, s_swap_j, tid);
            }
        }
    }
    else if (!tiled)
    {
        // Full-tour scan: upper triangle only, hoisted row endpoints + carry-over.
        for (int i = i_start + tid; i < i_end; i += block_size)
        {
            int a = (i >= 0) ? s_tour[i] : -1;
            int b = s_tour[i + 1];
            const int *row_a = (use_matrix && i >= 0) ? distances + (size_t)a * (size_t)dist_stride : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)dist_stride : nullptr;
            int d_ab = (i < 0)
                           ? 0
                           : (use_matrix
                                  ? row_a[b]
                                  : gpu_pairwise_distance(a, b, nullptr, coords, dist_stride, edge_type_code));
            // Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`
            int c = s_tour[i + 2];

            for (int j = i + 2; j < j_end; ++j)
            {
                int d = (j + 1 < n) ? s_tour[j + 1] : s_tour[0];
                if (!((cyclic && i == 0 && j == n - 1) || (i == -1 && j == n - 1)))
                {
                    two_opt_consider(
                        i, j,
                        two_opt_delta_topology(
                            i, j, n, closed, cyclic,
                            a, b, d_ab, c, d, row_a, row_b, distances, coords, dist_stride, edge_type_code,
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
        for (int i_base = i_start; i_base < i_end; i_base += block_size)
        {
            const int row = i_base + tid;
            const bool active = row < i_end;
            // Hoisted row endpoints stay in registers across every `j` window.
            int a = -1;
            int b = 0;
            int d_ab = 0;
            const int *row_a = nullptr;
            const int *row_b = nullptr;
            if (active)
            {
                a = (row >= 0) ? tour[row] : -1;
                b = tour[row + 1];
                row_a = (use_matrix && row >= 0) ? distances + (size_t)a * (size_t)dist_stride : nullptr;
                row_b = use_matrix ? distances + (size_t)b * (size_t)dist_stride : nullptr;
                d_ab = (row < 0)
                           ? 0
                           : (use_matrix
                                  ? row_a[b]
                                  : gpu_pairwise_distance(a, b, nullptr, coords, dist_stride, edge_type_code));
            }

            for (int window_start = i_base + 2; window_start < j_end; window_start += stride)
            {
                int window_end = window_start + stride;
                if (window_end > j_end)
                    window_end = j_end;
                const int window_len = window_end - window_start;

                __syncthreads();
                for (int idx = window_start + tid; idx < window_end; idx += block_size)
                {
                    s_tour[idx - window_start] = tour[idx];
                }
                if (tid == 0)
                {
                    // One extra slot holds the `j + 1` endpoint; it wraps to `tour[0]`
                    // only when `window_end == n` (safe fallback even on open paths).
                    s_tour[window_len] = tour[window_end < n ? window_end : 0];
                }
                __syncthreads();

                int j_begin = row + 2 > window_start ? row + 2 : window_start;
                if (active && j_begin < window_end)
                {
                    int c = s_tour[j_begin - window_start];
                    for (int j = j_begin; j < window_end; ++j)
                    {
                        int d = s_tour[j - window_start + 1];
                        if (!((cyclic && row == 0 && j == n - 1) || (row == -1 && j == n - 1)))
                        {
                            two_opt_consider(
                                row, j,
                                two_opt_delta_topology(
                                    row, j, n, closed, cyclic,
                                    a, b, d_ab, c, d, row_a, row_b, distances, coords, dist_stride, edge_type_code,
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

    // Greedy disjoint-move selection, with the reversal applied by the whole block.
    //
    // Each thread contributed at most one steepest candidate (`two_opt_consider`), so
    // one pass offers at most `min(block_size, n / 2)` moves to the selection step
    // (Rocki & Suda, 2013). That deliberate bound keeps the per-pass bookkeeping
    // O(block_size) instead of O(n^2), which is what distinguishes this kernel from the
    // exhaustive CPU `multi_move` reference.
    //
    // Only the *selection* is serial: thread 0 picks the steepest surviving candidate
    // and publishes the segment through shared memory, then every thread of the block
    // reverses it in parallel (`DP11-18`, `2OPT-A`). Reversing with thread 0 alone left
    // up to n/2 sequential swaps while the rest of the block waited at the barrier.
    __shared__ int s_move_start;
    __shared__ int s_move_end;

    // Every thread contributes at most one candidate, so `block_size` bounds the
    // number of selectable moves; a reversal always covers at least two positions,
    // so `n / 2` bounds the number of disjoint reversals a tour can absorb (`BUG-20`).
    const int max_moves = block_size < (n / 2) ? block_size : (n / 2);

    for (int applied = 0; applied < max_moves; applied++)
    {
        if (tid == 0)
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
            {
                s_move_start = -1; // No improving candidate left for this pass
            }
            else
            {
                s_move_start = s_swap_i[best_t] + 1;
                s_move_end = s_swap_j[best_t];
                s_deltas[best_t] = 0; // Mark consumed
            }
        }
        __syncthreads();

        if (s_move_start < 0)
            break; // Uniform across the block: `s_move_start` is shared

        // Reversed segment of the move just applied: tour positions
        // [seg_start, seg_end] = [swap_i + 1, swap_j].
        const int seg_start = s_move_start;
        const int seg_end = s_move_end;
        const int num_swaps = (seg_end - seg_start + 1) / 2;

        // Cooperative reversal: pair `k` swaps (seg_start + k, seg_end - k), and the
        // pairs are mutually disjoint, so every thread writes a distinct slot.
        if (use_candidates)
        {
            // Candidate mode never cached the tour: edit global memory directly.
            for (int k = tid; k < num_swaps; k += block_size)
            {
                const int tmp = tour[seg_start + k];
                tour[seg_start + k] = tour[seg_end - k];
                tour[seg_end - k] = tmp;
            }
        }
        else
        {
            for (int k = tid; k < num_swaps; k += block_size)
            {
                const int tmp = TOUR_AT(seg_start + k);
                TOUR_SET(seg_start + k, TOUR_AT(seg_end - k));
                TOUR_SET(seg_end - k, tmp);
            }
        }
        __syncthreads();

        // A reversal only permutes positions inside [seg_start, seg_end], so a candidate
        // whose break positions stay outside it keeps a valid delta; every other
        // candidate is dropped. Each thread owns its own slot, so no barrier is needed.
        if (s_deltas[tid] < 0)
        {
            int ms = s_swap_i[tid] + 1;
            int me = s_swap_j[tid];
            if (ms > me)
            {
                const int tmp = ms;
                ms = me;
                me = tmp;
            }
            if (!(me + 1 < seg_start || seg_end + 1 < ms))
            {
                s_deltas[tid] = 0; // Conflicting candidate eliminated on the fly
            }
        }
        __syncthreads();
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
