/**
 * Modular GPU Inversion Mutation Kernel
 * =====================================
 *
 * Applies segment inversion mutation in-place to routes `(N, n)` where
 * `probs[idx] < mutation_rate`. Each thread owns one route row `routes[idx * n]`
 * and reverses the contiguous slice `[left, right]`.
 */

extern "C" __global__
void inversion_mutation_kernel(
    int* __restrict__ routes,            // [N * n]
    const double* __restrict__ probs,    // [N]
    const int* __restrict__ pos1_arr,    // [N]
    const int* __restrict__ pos2_arr,    // [N]
    const double mutation_rate,
    const int num_routes,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_routes || n < 2) return;

    if (probs[idx] < mutation_rate) {
        int* route = &routes[idx * n];
        int p1 = pos1_arr[idx];
        int p2 = pos2_arr[idx];

        int left = (p1 < p2) ? p1 : p2;
        int right = (p1 < p2) ? p2 : p1;

        while (left < right) {
            int tmp = route[left];
            route[left] = route[right];
            route[right] = tmp;
            left++;
            right--;
        }
    }
}

