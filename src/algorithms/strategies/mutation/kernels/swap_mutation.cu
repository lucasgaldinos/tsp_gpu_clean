/**
 * Modular GPU Swap Mutation Kernel
 * ================================
 *
 * Applies swap mutation in-place to routes `(N, n)` where `rand_probs[idx] < mutation_rate`.
 */

extern "C" __global__
void swap_mutation_kernel(
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
        int tmp = route[p1];
        route[p1] = route[p2];
        route[p2] = tmp;
    }
}
