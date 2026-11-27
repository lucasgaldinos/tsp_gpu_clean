/**
 * Modular GPU Order Crossover (OX) Kernel (Format A, 1 Warp per Route)
 * ====================================================================
 *
 * Implements Order Crossover with:
 *   1. Format A direct-address membership in shared memory: 1 byte per city per warp,
 *      cleared in parallel, marked in parallel O(1) without atomics (DP11-11, BUG-22).
 *   2. 1 warp (32 threads) per route (route_idx = global_tid >> 5, lane_id = tid & 31).
 *   3. Cyclic streaming from parent 2 starting at (c2 + 1) % n compacted deterministically
 *      via __ballot_sync and __popc.
 *   4. Disjoint parent pairs (2k, 2k+1) sharing identical cut points (c1, c2) (OX-A).
 *   5. Preserving fixed depot endpoints when is_closed is non-zero (OX-A).
 */

extern "C" __global__ void order_crossover_kernel(
    const int *__restrict__ parents,  // [N * n]
    const int *__restrict__ cut1_arr, // [N]
    const int *__restrict__ cut2_arr, // [N]
    int *__restrict__ offspring,      // [N * n]
    const int num_routes,
    const int n,
    const int is_closed)
{
    extern __shared__ unsigned char s_in_seg[];

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int route_idx = global_tid >> 5; // 1 warp (32 threads) per route
    int lane_id = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;

    if (route_idx >= num_routes || n < 1)
        return;

    int *child = &offspring[route_idx * n];

    // Odd route handling: if odd-sized batch, last route is copied as-is
    if ((num_routes & 1) && (route_idx == num_routes - 1))
    {
        const int *parent = &parents[route_idx * n];
        for (int i = lane_id; i < n; i += 32)
        {
            child[i] = parent[i];
        }
        return;
    }

    // Disjoint parent pairing: (2k, 2k+1)
    int p1_idx = route_idx;
    int p2_idx = ((route_idx & 1) == 0) ? (route_idx + 1) : (route_idx - 1);
    const int *parent1 = &parents[p1_idx * n];
    const int *parent2 = &parents[p2_idx * n];

    // Warp-private shared memory table: 1 byte per city for this warp
    unsigned char *my_in_seg = &s_in_seg[warp_in_block * n];

    // Step 1: Parallel clear of membership table
    for (int c = lane_id; c < n; c += 32)
    {
        my_in_seg[c] = 0;
    }
    __syncwarp();

    int c1 = cut1_arr[route_idx];
    int c2 = cut2_arr[route_idx];
    if (c1 > c2)
    {
        int tmp = c1;
        c1 = c2;
        c2 = tmp;
    }

    int stream_start = is_closed ? 1 : 0;
    int stream_len = is_closed ? (n - 2) : n;
    int c2_rel = is_closed ? (c2 - 1) : c2;

    if (stream_len <= 1)
    {
        for (int i = lane_id; i < n; i += 32)
        {
            child[i] = parent1[i];
        }
        return;
    }

    if (is_closed && lane_id == 0)
    {
        child[0] = parent1[0];
        child[n - 1] = parent1[n - 1];
    }

    // Step 2: Parallel copy of segment [c1, c2] from parent1 and mark in shared memory
    int seg_len = c2 - c1 + 1;
    for (int k = lane_id; k < seg_len; k += 32)
    {
        int city = parent1[c1 + k];
        child[c1 + k] = city;
        if (city >= 0 && city < n)
        {
            my_in_seg[city] = 1;
        }
    }
    __syncwarp();

    // Step 3: Cyclic streaming from parent2 and compaction via __ballot_sync and __popc
    int child_pos = (c2_rel + 1) % stream_len;
    int dest_offset = 0;

    for (int step = 0; step < stream_len; step += 32)
    {
        int idx = step + lane_id;
        int retain = 0;
        int city = -1;
        if (idx < stream_len)
        {
            int p2_pos = stream_start + ((c2_rel + 1 + idx) % stream_len);
            city = parent2[p2_pos];
            if (city >= 0 && city < n)
            {
                if (my_in_seg[city] == 0)
                {
                    retain = 1;
                }
            }
            else
            {
                retain = 1;
            }
        }

        unsigned int mask = __ballot_sync(0xFFFFFFFFu, retain);
        if (retain)
        {
            unsigned int lower_mask = (1u << lane_id) - 1u;
            int rank = __popc(mask & lower_mask);
            int write_idx = stream_start + ((child_pos + dest_offset + rank) % stream_len);
            child[write_idx] = city;
        }
        dest_offset += __popc(mask);
    }
}
