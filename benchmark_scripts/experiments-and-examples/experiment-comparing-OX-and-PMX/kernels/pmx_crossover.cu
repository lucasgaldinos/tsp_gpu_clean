/**
 * Experiment-Local GPU Partially Mapped Crossover (PMX) Kernel (1 Warp per Route)
 * ===============================================================================
 *
 * Implements Partially Mapped Crossover (PMX) for TSP permutations and closed
 * depot-bounded tours with:
 *   1. 1 warp (32 threads) per route (`route_idx = global_tid >> 5`, `lane_id = tid & 31`).
 *   2. Warp-private shared memory holding:
 *        - `inv_p2[n]` (`int`, 4 bytes/city): inverse lookup `city -> position in parent2`
 *        - `in_seg[n]` (`unsigned char`, 1 byte/city): O(1) membership of `parent1[c1..c2]`
 *      Total shared memory = `warps_per_block * 5 * n` bytes (aligned).
 *   3. Conflict-free parallel bijection-chain traversal with zero atomics:
 *      distinct unmapped entries in `parent2[c1..c2]` trace disjoint chains to
 *      strictly distinct target positions outside `[c1, c2]`.
 *   4. Disjoint parent pairing `(2k, 2k+1)` sharing cut points `(c1, c2)`, odd
 *      trailing route passthrough, and fixed depot endpoints when `is_closed != 0`.
 */

extern "C" __global__
void pmx_crossover_kernel(
    const int* __restrict__ parents,     // [N * n]
    const int* __restrict__ cut1_arr,    // [N]
    const int* __restrict__ cut2_arr,    // [N]
    int* __restrict__ offspring,         // [N * n]
    const int num_routes,
    const int n,
    const int is_closed
) {
    extern __shared__ int s_mem[];

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int route_idx = global_tid >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    if (route_idx >= num_routes || n < 1) return;

    int* child = &offspring[route_idx * n];

    // Odd-sized batch: copy trailing route unchanged
    if ((num_routes & 1) && (route_idx == num_routes - 1)) {
        const int* parent = &parents[route_idx * n];
        for (int i = lane_id; i < n; i += 32) {
            child[i] = parent[i];
        }
        return;
    }

    // Disjoint parent pairing: (2k, 2k+1)
    int p1_idx = route_idx;
    int p2_idx = ((route_idx & 1) == 0) ? (route_idx + 1) : (route_idx - 1);
    const int* parent1 = &parents[p1_idx * n];
    const int* parent2 = &parents[p2_idx * n];

    int c1 = cut1_arr[route_idx];
    int c2 = cut2_arr[route_idx];
    if (c1 > c2) {
        int tmp = c1;
        c1 = c2;
        c2 = tmp;
    }

    int stream_start = is_closed ? 1 : 0;
    int stream_len = is_closed ? (n - 2) : n;

    if (stream_len <= 1) {
        for (int i = lane_id; i < n; i += 32) {
            child[i] = parent1[i];
        }
        return;
    }

    // Warp-private shared memory: `inv_p2` (int) followed by `in_seg` (byte)
    int* my_inv_p2 = &s_mem[warp_in_block * n];
    unsigned char* s_in_seg_base = reinterpret_cast<unsigned char*>(&s_mem[warps_per_block * n]);
    unsigned char* my_in_seg = &s_in_seg_base[warp_in_block * n];

    // Step 1: Parallel clear of child, inverse map, and segment membership
    for (int i = lane_id; i < n; i += 32) {
        child[i] = (is_closed && (i == 0 || i == n - 1)) ? parent1[i] : -1;
        my_inv_p2[i] = -1;
        my_in_seg[i] = 0;
    }
    __syncwarp();

    // Step 2: Parallel inverse lookup of parent2 & copy of segment [c1, c2] from parent1
    for (int k = lane_id; k < stream_len; k += 32) {
        int pos = stream_start + k;
        int city = parent2[pos];
        if (city >= 0 && city < n) {
            my_inv_p2[city] = pos;
        }
    }

    int seg_len = c2 - c1 + 1;
    for (int k = lane_id; k < seg_len; k += 32) {
        int pos = c1 + k;
        int city = parent1[pos];
        child[pos] = city;
        if (city >= 0 && city < n) {
            my_in_seg[city] = 1;
        }
    }
    __syncwarp();

    // Step 3: Parallel PMX mapping chain resolution for parent2[c1..c2] \ parent1[c1..c2]
    for (int k = lane_id; k < seg_len; k += 32) {
        int idx = c1 + k;
        int val = parent2[idx];
        if (val >= 0 && val < n && my_in_seg[val] == 0) {
            int pos = idx;
            int steps = 0;
            while (pos >= c1 && pos <= c2 && steps <= seg_len) {
                int mapped_val = parent1[pos];
                if (mapped_val < 0 || mapped_val >= n) break;
                int next_pos = my_inv_p2[mapped_val];
                if (next_pos < 0) break;
                pos = next_pos;
                steps++;
            }
            if (pos >= stream_start && pos < stream_start + stream_len && (pos < c1 || pos > c2)) {
                child[pos] = val;
            }
        }
    }
    __syncwarp();

    // Step 4: Parallel fill of remaining unmapped positions from parent2
    for (int k = lane_id; k < stream_len; k += 32) {
        int pos = stream_start + k;
        if (child[pos] == -1) {
            child[pos] = parent2[pos];
        }
    }
}
