/**
 * GPU 2-opt Single Tour Kernel
 *
 * Performs one pass of 2-opt improvement for a single TSP tour.
 * Uses shared memory for efficient parallel evaluation of all possible swaps.
 *
 * Algorithm:
 *   1. Load tour into shared memory
 *   2. Each thread evaluates subset of (i,j) swap pairs
 *   3. Parallel reduction finds best improvement (minimum delta)
 *   4. Apply best swap in-place
 *   5. Write back improved tour
 *
 * Candidate lists (`DP10-20`, `BUG-57`):
 *   - `candidate_k <= 0` scans the whole upper triangle, i.e. every `j in [i + 2, n)`.
 *   - `candidate_k > 0` builds the inverse tour map `s_pos[tour[idx]] = idx` in shared
 *     memory (`n * 4` extra bytes) and replaces the `j` scan with
 *     `j = s_pos[candidates[tour[i] * candidate_k + r]]`, so this kernel and
 *     `two_opt_batch.cu` search the same `Theta(n * k)` neighbourhood.
 *
 * Performance Notes:
 *   - Shared memory reduces global memory accesses
 *   - Parallel reduction minimizes synchronization
 *   - In-place reversal avoids temporary allocations
 *
 * References:
 *   - Croes, G. A. (1958). "A Method for Solving Traveling-Salesman Problems."
 *     Operations Research, 6(6), 791-812.
 *   - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a
 *     GPU Computing Platform." ICNAAM, Vol. 1389, pp. 264-267. (Shared-memory
 *     tour representation per CUDA thread block).
 *   - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
 *     Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
 *     (Cooperative parallel evaluation of (i, j) pairs and block reduction).
 */

// Delta arithmetic and per-thread best bookkeeping come from the shared header, so a
// fix to either cannot diverge between this kernel and `two_opt_batch.cu` (`DP10-19`,
// `BUG-57`).
#include "two_opt_common.cuh"

extern "C" __global__ void two_opt_kernel(
    int *tour,
    const int *dist,
    int n,
    int dist_stride,
    int is_closed,
    int is_cyclic,
    int fixed_start,
    int tour_idx,
    const int *candidates, // dist_stride * candidate_k nearest-neighbour table, or nullptr
    int candidate_k,       // 0 = exhaustive scan, else neighbours per node (k_eff)
    int *positions         // n inverse-map scratch in global memory (candidate mode only)
)
{
    extern __shared__ int shared_mem[];

    // Partition shared memory
    int *s_deltas = shared_mem;
    int *s_swap_i = (int *)&s_deltas[blockDim.x];
    int *s_swap_j = (int *)&s_swap_i[blockDim.x];
    int *s_tour = (int *)&s_swap_j[blockDim.x];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const bool closed = is_closed != 0;
    const bool cyclic = !closed && (is_cyclic != 0);
    const bool allow_prefix = !closed && !cyclic && (fixed_start == 0);
    const bool use_candidates = !closed && cyclic && candidate_k > 0 && candidates != nullptr && positions != nullptr;

    const int i_start = allow_prefix ? -1 : 0;
    const int i_end = closed ? (n - 3) : (n - 2);
    const int j_end = closed ? (n - 1) : n;

    // FIX: Load tour using loop to handle n > block_size
    for (int i = tid; i < n; i += block_size)
    {
        s_tour[i] = tour[i];
    }
    // The inverse map lives in global memory, like the batched kernel's `positions`
    // scratch: keeping it in shared memory would cut this kernel's ceiling in half
    // (`n` ints for the tour plus `n` for the map against the 48 KB per block).
    if (use_candidates)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            positions[tour[idx]] = idx;
        }
    }
    __syncthreads();

    // Each thread's best move lives in the shared reduction slots from the start, so
    // `two_opt_consider` is the only place that records a candidate (`DP10-19`).
    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;

    // Each thread evaluates a subset of i positions
    if (use_candidates)
    {
        for (int i = tid; i < n - 2; i += block_size)
        {
            int node_i = s_tour[i];
            int node_i1 = s_tour[i + 1];
            const int *row_i = dist + (size_t)node_i * (size_t)dist_stride;
            const int *row_i1 = dist + (size_t)node_i1 * (size_t)dist_stride;
            int d_ab = row_i[node_i1];

            // `Theta(n * k)` scan: walk the `k` nearest neighbours of `node_i` and
            // translate each node id into a tour position through `positions`.
            const int *row_c = candidates + (size_t)node_i * (size_t)candidate_k;
            for (int r = 0; r < candidate_k; ++r)
            {
                int j = positions[row_c[r]];
                // Upper triangle only; the wrap-around pair of `i == 0` would reverse
                // the whole tour for a zero gain, exactly as in `two_opt_batch.cu`.
                if (j <= i + 1 || (i == 0 && j == n - 1))
                    continue;

                int node_j = s_tour[j];
                int node_j1 = s_tour[(j + 1) % n];
                two_opt_consider(
                    i, j,
                    two_opt_delta(node_i, node_i1, d_ab, node_j, node_j1, row_i, row_i1, dist, nullptr, dist_stride,
                                  0, true),
                    s_deltas, s_swap_i, s_swap_j, tid);
            }
        }
    }
    else
    {
        for (int i = i_start + tid; i < i_end; i += block_size)
        {
            // Hoisted row endpoints and current edge cost (`Task 10.16`)
            int node_i = (i >= 0) ? s_tour[i] : -1;
            int node_i1 = s_tour[i + 1];
            const int *row_i = (i >= 0) ? (dist + (size_t)node_i * (size_t)dist_stride) : nullptr;
            const int *row_i1 = dist + (size_t)node_i1 * (size_t)dist_stride;
            int d_ab = (i >= 0) ? row_i[node_i1] : 0;

            // Carry-over: `node_j` of iteration `j + 1` is `node_j1` of iteration `j`
            int node_j = s_tour[i + 2];

            // Serial inner loop over j (upper triangle only)
            for (int j = i + 2; j < j_end; ++j)
            {
                int node_j1 = (j + 1 < n) ? s_tour[j + 1] : s_tour[0];

                if (!((cyclic && i == 0 && j == n - 1) || (i == -1 && j == n - 1)))
                {
                    two_opt_consider(
                        i, j,
                        two_opt_delta_topology(
                            i, j, n, closed, cyclic,
                            node_i, node_i1, d_ab, node_j, node_j1, row_i, row_i1, dist, nullptr, dist_stride,
                            0, true),
                        s_deltas, s_swap_i, s_swap_j, tid);
                }
                node_j = node_j1;
            }
        }
    }

    __syncthreads();

    // Block reduction to the exact minimum delta.
    //
    // Phase 1 folds thread partials in parallel while the stride still spans more than
    // one warp. Phase 2 lets thread 0 fold only the surviving `min(block_size, 64)`
    // partials -- the pre-fix code rescanned all `block_size` of them after the tree had
    // already merged the upper half (`BUG-29`).
    for (unsigned int s = block_size / 2; s > 32; s >>= 1)
    {
        if (tid < s && tid + s < block_size && s_deltas[tid + s] < s_deltas[tid])
        {
            s_deltas[tid] = s_deltas[tid + s];
            s_swap_i[tid] = s_swap_i[tid + s];
            s_swap_j[tid] = s_swap_j[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        const int remaining = block_size < 64 ? block_size : 64;
        for (int i = 1; i < remaining; i++)
        {
            if (s_deltas[i] < s_deltas[0])
            {
                s_deltas[0] = s_deltas[i];
                s_swap_i[0] = s_swap_i[i];
                s_swap_j[0] = s_swap_j[i];
            }
        }
    }
    __syncthreads();

    // `s_deltas[0]` is now the exact minimum, so a pass with no improving move skips
    // the reversal branch instead of synchronizing for nothing (`BUG-29`).
    if (s_deltas[0] < 0)
    {
        int swap_i = s_swap_i[0];
        int swap_j = s_swap_j[0];

        // Reverse segment [swap_i + 1, swap_j]
        int left = swap_i + 1;
        int right = swap_j;
        int num_swaps = (right - left + 1) / 2;
        for (int k = tid; k < num_swaps; k += block_size)
        {
            int temp = s_tour[left + k];
            s_tour[left + k] = s_tour[right - k];
            s_tour[right - k] = temp;
        }
    }
    __syncthreads();

    // FIX: Write back using loop to handle n > block_size
    for (int i = tid; i < n; i += block_size)
    {
        tour[i] = s_tour[i];
    }
}
