/**
 * Modular GPU Inversion Mutation Kernel
 * =====================================
 *
 * Applies segment inversion mutation in-place to routes `(N, n)` where
 * `probs[route_idx] < mutation_rate` (DP11-17, INV-A).
 * Assigns 1 warp (32 threads) per route (`route_idx = tid >> 5`, `lane_id = tid & 31`).
 * Reverses the contiguous slice `[left, right]` cooperatively in parallel across the 32 lanes.
 */

extern "C" __global__ void inversion_mutation_kernel(
    int *__restrict__ routes,         // [N * n]
    const double *__restrict__ probs, // [N]
    const int *__restrict__ pos1_arr, // [N]
    const int *__restrict__ pos2_arr, // [N]
    const double mutation_rate,
    const int num_routes,
    const int n)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int route_idx = global_tid >> 5; // 1 warp per route
    int lane_id = threadIdx.x & 31;

    if (route_idx >= num_routes || n < 2)
        return;

    if (probs[route_idx] < mutation_rate)
    {
        int *route = &routes[route_idx * n];
        int p1 = pos1_arr[route_idx];
        int p2 = pos2_arr[route_idx];

        int left = (p1 < p2) ? p1 : p2;
        int right = (p1 < p2) ? p2 : p1;
        int num_swaps = (right - left + 1) / 2;

        for (int k = lane_id; k < num_swaps; k += 32)
        {
            int tmp = route[left + k];
            route[left + k] = route[right - k];
            route[right - k] = tmp;
        }
    }
}
