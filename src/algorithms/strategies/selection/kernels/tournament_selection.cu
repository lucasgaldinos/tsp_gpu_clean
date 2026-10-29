/**
 * Modular GPU Tournament Selection Kernel
 * =======================================
 *
 * Selects `num_parents` parent routes from `population` in parallel on GPU.
 * Each thread `idx` (or block) selects one winner from `tournament_size`
 * randomly sampled candidates and copies the winning route to `parents_out`.
 */

extern "C" __global__
void tournament_selection_kernel(
    const int* __restrict__ population,      // [pop_size * n]
    const double* __restrict__ fitness,      // [pop_size]
    const int* __restrict__ candidates,      // [num_parents * tournament_size]
    int* __restrict__ parents_out,           // [num_parents * n]
    const int pop_size,
    const int num_parents,
    const int n,
    const int tournament_size
) {
    int parent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent_idx >= num_parents) return;

    const int* tour_candidates = &candidates[parent_idx * tournament_size];
    int best_idx = tour_candidates[0];
    double best_fit = fitness[best_idx];

    for (int k = 1; k < tournament_size; k++) {
        int cand = tour_candidates[k];
        double cand_fit = fitness[cand];
        if (cand_fit < best_fit) {
            best_fit = cand_fit;
            best_idx = cand;
        }
    }

    const int* src_route = &population[best_idx * n];
    int* dst_route = &parents_out[parent_idx * n];
    for (int c = 0; c < n; c++) {
        dst_route[c] = src_route[c];
    }
}
