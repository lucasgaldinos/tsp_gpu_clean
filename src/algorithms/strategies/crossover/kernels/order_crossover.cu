/**
 * Modular GPU Order Crossover (OX) Kernel
 * =======================================
 *
 * Recombines pairs of parent routes `(N, n)` into offspring routes `(N, n)` in parallel.
 * Each thread `idx` constructs offspring `idx` from `parents[idx]` and `parents[(idx + 1) % N]`
 * using cut points `cut1[idx]` and `cut2[idx]`.
 */

extern "C" __global__
void order_crossover_kernel(
    const int* __restrict__ parents,     // [N * n]
    const int* __restrict__ cut1_arr,    // [N]
    const int* __restrict__ cut2_arr,    // [N]
    int* __restrict__ offspring,         // [N * n]
    const int num_routes,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_routes) return;

    int p2_idx = (idx + 1) % num_routes;
    const int* parent1 = &parents[idx * n];
    const int* parent2 = &parents[p2_idx * n];
    int* child = &offspring[idx * n];

    int c1 = cut1_arr[idx];
    int c2 = cut2_arr[idx];
    if (c1 > c2) {
        int tmp = c1;
        c1 = c2;
        c2 = tmp;
    }

    for (int i = 0; i < n; i++) {
        child[i] = -1;
    }

    for (int i = c1; i <= c2; i++) {
        child[i] = parent1[i];
    }

    int child_pos = (c2 + 1) % n;
    for (int i = 0; i < n; i++) {
        int p2_pos = (c2 + 1 + i) % n;
        int city = parent2[p2_pos];

        bool in_child = false;
        for (int j = c1; j <= c2; j++) {
            if (child[j] == city) {
                in_child = true;
                break;
            }
        }

        if (!in_child) {
            child[child_pos] = city;
            child_pos = (child_pos + 1) % n;
        }
    }
}
