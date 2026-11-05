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

#include "pairwise_device.cuh"

//: Tour positions cached per shared-memory tile when `n` exceeds the full-tour budget.
#define TWO_OPT_TILE_SIZE 2048

// Accessor macros used by the tiled scan (`Task 10.18`). When `tiled` is false the
// whole tour lives in `s_tour[n]`; otherwise only `s_tour[TWO_OPT_TILE_SIZE + 1]`
// holds the current window and out-of-window reads fall back to global memory.
#define TOUR_AT(idx) \
    (tiled ? (((idx) >= tile_start && (idx) <= tile_end) ? s_tour[(idx) - tile_start] : tour[(idx)]) : s_tour[(idx)])

#define TOUR_SET(idx, value) \
    do { \
        int _pos = (idx); \
        int _val = (value); \
        if (!tiled) { \
            s_tour[_pos] = _val; \
        } else { \
            tour[_pos] = _val; \
            if (_pos >= tile_start && _pos <= tile_end) { \
                s_tour[_pos - tile_start] = _val; \
            } \
        } \
    } while (0)

// Exact 2-opt delta for breaking edges (a, b) and (c, d).
// `row_a`/`row_b` alias `distances` rows in dense-matrix mode; the coordinate mode
// evaluates `gpu_pairwise_distance` on the fly (`DP10-19`).
static __device__ __forceinline__ int two_opt_delta(
    int a, int b, int d_ab, int c, int d,
    const int *row_a, const int *row_b,
    const int *distances, const double *coords, int n, int edge_type_code, bool use_matrix) {
    if (use_matrix) {
        int old_dist = d_ab + distances[(size_t)c * (size_t)n + (size_t)d];
        int new_dist = row_a[c] + row_b[d];
        return new_dist - old_dist;
    }
    int old_dist = d_ab + gpu_pairwise_distance(c, d, nullptr, coords, n, edge_type_code);
    int new_dist = gpu_pairwise_distance(a, c, nullptr, coords, n, edge_type_code)
                 + gpu_pairwise_distance(b, d, nullptr, coords, n, edge_type_code);
    return new_dist - old_dist;
}

// Keep the steepest (most negative) delta of this thread.
static __device__ __forceinline__ void two_opt_consider(
    int i, int j, int delta, int *s_deltas, int *s_swap_i, int *s_swap_j, int tid) {
    if (delta < s_deltas[tid]) {
        s_deltas[tid] = delta;
        s_swap_i[tid] = i;
        s_swap_j[tid] = j;
    }
}

extern "C" __global__
void two_opt_multi_move_batch_kernel(
    int* __restrict__ population,             // [population_size, n]
    const int* __restrict__ distances,        // [n, n], or nullptr for coordinate mode
    const double* __restrict__ coords,        // Row-major coordinates (ignored if distances != nullptr)
    const int n,
    const int population_size,
    const int edge_type_code,                 // `GpuEdgeTypeCode` for the coordinate mode
    const int tour_tile_size                  // 0 = full-tour shared memory, else tile width (2048)
) {
    int route_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (route_id >= population_size) return;

    extern __shared__ int shared_mem[];
    const bool tiled = tour_tile_size > 0;
    const bool use_matrix = distances != nullptr;

    int* s_tour = shared_mem;                              // [n] or [tour_tile_size + 1]
    int* s_deltas = tiled ? &s_tour[tour_tile_size + 1] : &s_tour[n];  // [block_size]
    int* s_swap_i = &s_deltas[block_size];                 // [block_size]
    int* s_swap_j = &s_swap_i[block_size];                 // [block_size]

    int* tour = &population[route_id * n];
    if (!tiled) {
        for (int idx = tid; idx < n; idx += block_size) {
            s_tour[idx] = tour[idx];
        }
        __syncthreads();
    }

    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;

    int tile_start = 0;
    int tile_end = -1;

    if (!tiled) {
        // Full-tour scan: upper triangle only, hoisted row endpoints + carry-over.
        for (int i = tid; i < n - 2; i += block_size) {
            int a = s_tour[i];
            int b = s_tour[i + 1];
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)n : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)n : nullptr;
            int d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, n, edge_type_code);
            // Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`
            int c = s_tour[i + 2];

            for (int j = i + 2; j < n; ++j) {
                int d = s_tour[(j + 1) % n];
                if (!(i == 0 && j == n - 1)) {
                    two_opt_consider(
                        i, j,
                        two_opt_delta(a, b, d_ab, c, d, row_a, row_b, distances, coords, n, edge_type_code,
                                      use_matrix),
                        s_deltas, s_swap_i, s_swap_j, tid);
                }
                c = d;
            }
        }
    } else {
        // Windowed scan (`Task 10.18` refinement): the block owns the rows
        // `[i_base, i_base + block_size)` and walks the `j` windows from `i_base + 2`,
        // so every inner-loop `j` read hits shared memory `s_tour[j - window_start]`.
        const int stride = tour_tile_size;
        for (int i_base = 0; i_base < n - 2; i_base += block_size) {
            const int row = i_base + tid;
            const bool active = row < n - 2;
            // Hoisted row endpoints stay in registers across every `j` window.
            int a = 0;
            int b = 0;
            int d_ab = 0;
            const int *row_a = nullptr;
            const int *row_b = nullptr;
            if (active) {
                a = tour[row];
                b = tour[row + 1];
                row_a = use_matrix ? distances + (size_t)a * (size_t)n : nullptr;
                row_b = use_matrix ? distances + (size_t)b * (size_t)n : nullptr;
                d_ab = use_matrix
                           ? row_a[b]
                           : gpu_pairwise_distance(a, b, nullptr, coords, n, edge_type_code);
            }

            for (int window_start = i_base + 2; window_start < n; window_start += stride) {
                int window_end = window_start + stride;
                if (window_end > n) window_end = n;
                const int window_len = window_end - window_start;

                __syncthreads();
                for (int idx = window_start + tid; idx < window_end; idx += block_size) {
                    s_tour[idx - window_start] = tour[idx];
                }
                if (tid == 0) {
                    // One extra slot holds the `j + 1` endpoint; it wraps to `tour[0]`
                    // on the last window (`tour[(idx) % n]`).
                    s_tour[window_len] = tour[window_end % n];
                }
                __syncthreads();

                int j_begin = row + 2 > window_start ? row + 2 : window_start;
                if (active && j_begin < window_end) {
                    int c = s_tour[j_begin - window_start];
                    for (int j = j_begin; j < window_end; ++j) {
                        int d = s_tour[j - window_start + 1];
                        if (!(row == 0 && j == n - 1)) {
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
    if (tid == 0) {
        int applied_start[16];
        int applied_end[16];
        int num_applied = 0;

        for (int pass = 0; pass < 16; pass++) {
            int best_t = -1;
            int best_d = 0;
            for (int t = 0; t < block_size; t++) {
                if (s_deltas[t] < best_d) {
                    int cand_i = s_swap_i[t];
                    int cand_j = s_swap_j[t];
                    bool conflict = false;
                    for (int k = 0; k < num_applied; k++) {
                        // Two moves (i1, j1) and (i2, j2) conflict if [i1, j1] and [i2, j2] overlap
                        if (!(cand_j < applied_start[k] || cand_i > applied_end[k])) {
                            conflict = true;
                            break;
                        }
                    }
                    if (!conflict) {
                        best_d = s_deltas[t];
                        best_t = t;
                    }
                }
            }

            if (best_t < 0) break;

            int swap_i = s_swap_i[best_t];
            int swap_j = s_swap_j[best_t];
            s_deltas[best_t] = 0;  // Mark consumed

            int left = swap_i + 1;
            int right = swap_j;
            while (left < right) {
                int tmp = TOUR_AT(left);
                TOUR_SET(left, TOUR_AT(right));
                TOUR_SET(right, tmp);
                left++;
                right--;
            }

            applied_start[num_applied] = swap_i;
            applied_end[num_applied] = swap_j;
            num_applied++;
        }
    }
    __syncthreads();

    // Write back to global memory (the tiled path already wrote through TOUR_SET)
    if (!tiled) {
        for (int idx = tid; idx < n; idx += block_size) {
            tour[idx] = s_tour[idx];
        }
    }
}
