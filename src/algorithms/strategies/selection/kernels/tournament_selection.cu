/**
 * Modular GPU Tournament Selection Kernel
 * =======================================
 *
 * Selects `num_parents` parent indices from `population` in parallel on GPU (DP11-16, SEL-A).
 * Each thread `idx` selects one winner index from `tournament_size` randomly sampled
 * candidates and stores the winner in `winner_indices[idx]`.
 * Route gathering is delegated to CuPy's vectorized 2D indexer (NVIDIA CUB DeviceGather).
 */

extern "C" __global__
void tournament_selection_kernel(
    const double* __restrict__ fitness,      // [pop_size]
    const int* __restrict__ candidates,      // [num_parents * tournament_size]
    int* __restrict__ winner_indices,        // [num_parents]
    const int pop_size,
    const int num_parents,
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

    winner_indices[parent_idx] = best_idx;
}
