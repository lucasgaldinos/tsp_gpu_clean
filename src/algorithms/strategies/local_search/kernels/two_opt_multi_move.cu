/**
 * Multi-Move Disjoint-Segment Batch 2-Opt CUDA Kernel
 * ====================================================
 *
 * Evaluates all O(n^2) 2-opt moves across threads in the block, then greedily
 * applies multiple non-overlapping (disjoint) improving segment reversals
 * in the same iteration pass.
 */

extern "C" __global__
void two_opt_multi_move_batch_kernel(
    int* __restrict__ population,       // [population_size, n]
    const int* __restrict__ distances,  // [n, n]
    const int n,
    const int population_size
) {
    int route_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (route_id >= population_size) return;

    extern __shared__ int shared_mem[];
    int* s_tour = shared_mem;                              // [n]
    int* s_deltas = &shared_mem[n];                        // [block_size]
    int* s_swap_i = &shared_mem[n + block_size];           // [block_size]
    int* s_swap_j = &shared_mem[n + 2 * block_size];       // [block_size]

    int* tour = &population[route_id * n];
    for (int idx = tid; idx < n; idx += block_size) {
        s_tour[idx] = tour[idx];
    }
    __syncthreads();

    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;

    int total_pairs = (n * (n - 3)) / 2;
    for (int pair_id = tid; pair_id < total_pairs; pair_id += block_size) {
        int i = 0;
        int remaining = pair_id;
        for (int row = 0; row < n - 2; row++) {
            int row_size = (n - 2) - row - (row == 0 ? 1 : 0);
            if (remaining < row_size) {
                i = row;
                break;
            }
            remaining -= row_size;
        }
        int j = i + 2 + remaining;

        if (i < n - 2 && j < n && !(i == 0 && j == n - 1)) {
            int a = s_tour[i];
            int b = s_tour[i + 1];
            int c = s_tour[j];
            int d = s_tour[(j + 1) % n];

            int delta = (distances[a * n + c] + distances[b * n + d])
                      - (distances[a * n + b] + distances[c * n + d]);

            if (delta < s_deltas[tid]) {
                s_deltas[tid] = delta;
                s_swap_i[tid] = i;
                s_swap_j[tid] = j;
            }
        }
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
                int tmp = s_tour[left];
                s_tour[left] = s_tour[right];
                s_tour[right] = tmp;
                left++;
                right--;
            }

            applied_start[num_applied] = swap_i;
            applied_end[num_applied] = swap_j;
            num_applied++;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n; idx += block_size) {
        tour[idx] = s_tour[idx];
    }
}
