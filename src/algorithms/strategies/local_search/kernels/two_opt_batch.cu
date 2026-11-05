/**
 * GPU 2-opt Batch Processing Kernel
 * 
 * Performs one pass of 2-opt improvement for multiple TSP tours in parallel.
 * Each CUDA block processes one tour independently.
 * 
 * Algorithm:
 *   1. Each block loads its tour into shared memory
 *   2. Parallel evaluation of all (i,j) swap pairs within block
 *   3. Block-level reduction finds best improvement
 *   4. Apply best swap in-place
 *   5. Write back improved tour
 * 
 * Performance Notes:
 *   - No inter-block synchronization (tours are independent)
 *   - Shared memory per block reduces global memory traffic
 *   - Batch processing eliminates per-tour kernel launch overhead
 *   - Expected 5-10x speedup for population-based algorithms
 *
 * Memory layout (`Task 10.18`):
 *   - `tour_tile_size == 0`: the whole tour lives in `s_tour[n]`, loaded once.
 *   - `tour_tile_size > 0`: only one `j`-window of `tour_tile_size` positions plus one
 *     extra slot (`s_tour[tour_tile_size + 1]`) is cached. The block owns rows
 *     `[i_base, i_base + block_size)` and walks the windows from `i_base + 2`, so 100%
 *     of the inner-loop `j` reads hit shared memory instead of falling back to global.
 *
 * Candidate lists (`DP10-20`):
 *   - With `candidate_k > 0` each block builds the inverse tour map
 *     `pos[tour[idx]] = idx` in `O(n)` and replaces the `j` scan with
 *     `j = pos[candidates[tour[i] * candidate_k + r]]`, i.e. `Theta(n * k)` per pass.
 * 
 * Usage Pattern (Phase 3 Batch API):
 *   - Genetic Algorithm: Improve entire population in one kernel launch
 *   - Multistart: Process multiple independent runs in parallel
 *   - Grid: num_tours blocks, each with up to 256 threads
 * 
 * References:
 *   - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a
 *     GPU Computing Platform." ICNAAM, Vol. 1389, pp. 264-267. (One CUDA block
 *     per tour with shared-memory tour storage `s_tour`).
 *   - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
 *     Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
 *     (Thread-cooperative O(n^2) move evaluation and shared-memory reduction).
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

// Exact 2-opt delta for breaking edges (node_i, node_i1) and (node_j, node_j1).
// `row_i`/`row_i1` alias `dist` rows in dense-matrix mode; the coordinate mode
// evaluates `gpu_pairwise_distance` on the fly (`DP10-19`).
static __device__ __forceinline__ int two_opt_delta(
    int node_i, int node_i1, int d_ab, int node_j, int node_j1,
    const int *row_i, const int *row_i1,
    const int *dist, const double *coords, int n, int edge_type_code, bool use_matrix) {
    if (use_matrix) {
        int old_dist = d_ab + dist[(size_t)node_j * (size_t)n + (size_t)node_j1];
        int new_dist = row_i[node_j] + row_i1[node_j1];
        return new_dist - old_dist;
    }
    int old_dist = d_ab + gpu_pairwise_distance(node_j, node_j1, nullptr, coords, n, edge_type_code);
    int new_dist = gpu_pairwise_distance(node_i, node_j, nullptr, coords, n, edge_type_code)
                 + gpu_pairwise_distance(node_i1, node_j1, nullptr, coords, n, edge_type_code);
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
void two_opt_batch_kernel(
    int *tours,             // Flattened: num_tours * n
    const int *dist,        // n * n distance matrix (int32), or nullptr for coordinate mode
    const double *coords,   // Row-major coordinates (ignored if dist != nullptr)
    int n,                  // Number of nodes per tour
    int num_tours,          // Number of tours
    int edge_type_code,     // `GpuEdgeTypeCode` for the coordinate mode
    int tour_tile_size,     // 0 = full-tour shared memory, else tile width (2048)
    const int *candidates,  // n * candidate_k nearest-neighbour table, or nullptr
    int candidate_k,        // 0 = exhaustive scan, else neighbours per node (k_eff)
    int *positions          // num_tours * n inverse map scratch (candidate mode only)
) {
    extern __shared__ int shared_mem[];

    const bool tiled = tour_tile_size > 0;
    const bool use_matrix = dist != nullptr;
    const bool use_candidates = candidate_k > 0 && candidates != nullptr && positions != nullptr;

    // Partition shared memory (per-block): the tour buffer comes first so the
    // tiled and untiled layouts share one allocation.
    int *s_tour = shared_mem;
    int *s_deltas = tiled ? (int*)&s_tour[tour_tile_size + 1] : (int*)&s_tour[n];
    int *s_swap_i = (int*)&s_deltas[blockDim.x];
    int *s_swap_j = (int*)&s_swap_i[blockDim.x];

    int tour_idx = blockIdx.x;  // Which tour this block processes
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Offset to this tour's data
    int *tour = tours + (tour_idx * n);

    // Untiled path: load the whole tour once (candidate mode reads global memory instead).
    if (!tiled && !use_candidates) {
        for (int i = tid; i < n; i += block_size) {
            s_tour[i] = tour[i];
        }
        __syncthreads();
    }

    // Initialize shared memory for this thread
    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;

    int tile_start = 0;
    int tile_end = -1;

    if (use_candidates) {
        // Inverse tour map: `pos[tour[idx]] = idx` in one O(n) pass (`DP10-20`).
        int *pos = positions + (tour_idx * n);
        for (int idx = tid; idx < n; idx += block_size) {
            pos[tour[idx]] = idx;
        }
        __syncthreads();

        for (int i = tid; i < n - 2; i += block_size) {
            // Hoisted row endpoints and current edge cost (`Task 10.16`).
            int node_i = tour[i];
            int node_i1 = tour[i + 1];
            const int *row_i = use_matrix ? dist + (size_t)node_i * (size_t)n : nullptr;
            const int *row_i1 = use_matrix ? dist + (size_t)node_i1 * (size_t)n : nullptr;
            int d_ab = use_matrix
                           ? row_i[node_i1]
                           : gpu_pairwise_distance(node_i, node_i1, nullptr, coords, n, edge_type_code);

            for (int r = 0; r < candidate_k; ++r) {
                int j = pos[candidates[(size_t)node_i * (size_t)candidate_k + (size_t)r]];
                if (j <= i + 1 || (i == 0 && j == n - 1)) continue;
                int node_j = tour[j];
                int node_j1 = tour[(j + 1) % n];
                two_opt_consider(
                    i, j,
                    two_opt_delta(node_i, node_i1, d_ab, node_j, node_j1, row_i, row_i1, dist, coords, n,
                                  edge_type_code, use_matrix),
                    s_deltas, s_swap_i, s_swap_j, tid);
            }
        }
    } else if (!tiled) {
        // Full-tour scan: upper triangle only, hoisted row endpoints + carry-over.
        for (int i = tid; i < n - 2; i += block_size) {
            int node_i = s_tour[i];
            int node_i1 = s_tour[i + 1];
            const int *row_i = use_matrix ? dist + (size_t)node_i * (size_t)n : nullptr;
            const int *row_i1 = use_matrix ? dist + (size_t)node_i1 * (size_t)n : nullptr;
            int d_ab = use_matrix
                           ? row_i[node_i1]
                           : gpu_pairwise_distance(node_i, node_i1, nullptr, coords, n, edge_type_code);
            // Carry-over: `node_j` of iteration `j + 1` is `node_j1` of iteration `j`
            int node_j = s_tour[i + 2];

            for (int j = i + 2; j < n; ++j) {
                int node_j1 = s_tour[(j + 1) % n];
                if (!(i == 0 && j == n - 1)) {
                    two_opt_consider(
                        i, j,
                        two_opt_delta(node_i, node_i1, d_ab, node_j, node_j1, row_i, row_i1, dist, coords, n,
                                      edge_type_code, use_matrix),
                        s_deltas, s_swap_i, s_swap_j, tid);
                }
                node_j = node_j1;
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
            int node_i = 0;
            int node_i1 = 0;
            int d_ab = 0;
            const int *row_i = nullptr;
            const int *row_i1 = nullptr;
            if (active) {
                node_i = tour[row];
                node_i1 = tour[row + 1];
                row_i = use_matrix ? dist + (size_t)node_i * (size_t)n : nullptr;
                row_i1 = use_matrix ? dist + (size_t)node_i1 * (size_t)n : nullptr;
                d_ab = use_matrix
                           ? row_i[node_i1]
                           : gpu_pairwise_distance(node_i, node_i1, nullptr, coords, n, edge_type_code);
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
                    int node_j = s_tour[j_begin - window_start];
                    for (int j = j_begin; j < window_end; ++j) {
                        int node_j1 = s_tour[j - window_start + 1];
                        if (!(row == 0 && j == n - 1)) {
                            two_opt_consider(
                                row, j,
                                two_opt_delta(node_i, node_i1, d_ab, node_j, node_j1, row_i, row_i1, dist, coords,
                                              n, edge_type_code, use_matrix),
                                s_deltas, s_swap_i, s_swap_j, tid);
                        }
                        node_j = node_j1;
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
    
    // Parallel reduction to find minimum delta across all threads in this block
    // For non-power-of-2 block sizes, we need special handling
    // Strategy: Parallel reduce as much as possible, then serial reduction for remainder
    
    // Parallel reduction for power-of-2 portion
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (tid + s < block_size && s_deltas[tid + s] < s_deltas[tid]) {
                s_deltas[tid] = s_deltas[tid + s];
                s_swap_i[tid] = s_swap_i[tid + s];
                s_swap_j[tid] = s_swap_j[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Final reduction: use thread 0 to serially reduce remaining values
    // This guarantees correctness for any block_size (power-of-2 or not)
    if (tid == 0) {
        for (int i = 1; i < block_size; i++) {
            if (s_deltas[i] < s_deltas[0]) {
                s_deltas[0] = s_deltas[i];
                s_swap_i[0] = s_swap_i[i];
                s_swap_j[0] = s_swap_j[i];
            }
        }
    }
    __syncthreads();
    
    // Apply best swap (Thread 0 only)
    if (tid == 0 && s_deltas[0] < 0) {
        int swap_i = s_swap_i[0];
        int swap_j = s_swap_j[0];
        
        // Reverse segment [i+1, j]
        int left = swap_i + 1;
        int right = swap_j;
        if (use_candidates) {
            // Candidate mode never cached the tour: edit global memory directly.
            while (left < right) {
                int temp = tour[left];
                tour[left] = tour[right];
                tour[right] = temp;
                left++;
                right--;
            }
        } else {
            while (left < right) {
                int temp = TOUR_AT(left);
                TOUR_SET(left, TOUR_AT(right));
                TOUR_SET(right, temp);
                left++;
                right--;
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory (the tiled and candidate paths already wrote through)
    if (!tiled && !use_candidates) {
        for (int i = tid; i < n; i += block_size) {
            tour[i] = s_tour[i];
        }
    }
}
