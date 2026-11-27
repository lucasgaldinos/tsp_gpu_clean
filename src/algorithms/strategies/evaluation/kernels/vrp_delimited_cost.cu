/**
 * GPU Direct 0-Delimited VRP Route Cost & Constraint Violation Kernel (`DP17-1`, `DP17-3`, `DP17-6`, `DP17-7`)
 *
 * Evaluates a batch of `(batch_size, width)` `int32` 0-delimited VRP routes in parallel
 * using **one 32-thread warp per individual** (4 warps per 128-thread block).
 * Adjacent lanes read `row[tile + lane]` coalesced in 128-byte transactions, compute
 * `gpu_edge_distance(prev_node, node, ...)` and `demands[node]` across all 32 lanes,
 * detect route starts/ends via `__ballot_sync`, and segment-reduce per-route loads and
 * distances across route boundaries and 32-column tiles via warp shuffles + `warp_reduce_sum`.
 *
 * Computes for each individual:
 *   - `unpenalized_costs[idx]`: exact travel cost c(R) under TSPLIB95 int32 distances
 *     (closing each non-empty route back to depot 0 iff `return_to_depot != 0`).
 *   - `violations[idx]`: sum of capacity excess, route-distance excess, and fleet excess.
 *   - `penalized_costs[idx]`: IEEE-754-safe `c(R) + cap_pen + dist_pen + veh_pen`
 *     (never computing `0.0 * inf`).
 *   - `num_vehicles[idx]`: count of active non-empty routes.
 *   - `raw_components[idx, 0..2]`: `[viol_q, viol_l, viol_k]` when non-null.
 *
 * Supports both dense `int32` distance matrices (`dist_stride = context.dimension`) and
 * on-the-fly coordinate distances (`distances == nullptr`) via `pairwise_device.cuh`.
 */

#include "block_reductions.cuh"
#include "pairwise_device.cuh"

extern "C" __global__ void vrp_delimited_cost_kernel(
    const int *__restrict__ routes,    // [batch_size, width]
    const int *__restrict__ demands,   // [dist_stride]
    const int *__restrict__ distances, // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords, // [dist_stride, 2|3] or nullptr
    const int batch_size,
    const int width,
    const int dist_stride, // context.dimension
    const int edge_type_code,
    const int return_to_depot,
    const int has_capacity,
    const long long capacity,
    const int has_max_distance,
    const double max_route_distance,
    const int has_max_vehicles,
    const int max_vehicles,
    const double omega_q,
    const double omega_l,
    const double omega_k,
    double *__restrict__ penalized_costs,   // [batch_size]
    double *__restrict__ unpenalized_costs, // [batch_size]
    double *__restrict__ violations,        // [batch_size]
    int *__restrict__ num_vehicles,         // [batch_size]
    double *__restrict__ raw_components     // [batch_size * 3] [viol_q, viol_l, viol_k] or nullptr
)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int idx = blockIdx.x * warps_per_block + warp_id;
    if (idx >= batch_size)
    {
        return;
    }

    const bool use_matrix = (distances != nullptr);
    const int *row = routes + (size_t)idx * (size_t)width;

    long long lane_total_dist = 0LL;
    long long lane_cap_viol = 0LL;
    double lane_dist_viol = 0.0;
    int lane_vehicles = 0;

    int prev_tile_last_node = 0;
    long long carry_load = 0LL;
    long long carry_dist = 0LL;

    for (int tile = 0; tile < width; tile += 32)
    {
        const int col = tile + lane;
        const int node = (col < width) ? row[col] : 0;

        // Uniform fast-skip across all-zero padded tiles when no open route carried in.
        if (prev_tile_last_node == 0 && __all_sync(0xFFFFFFFFu, node == 0))
        {
            continue;
        }

        const int shifted_prev = __shfl_up_sync(0xFFFFFFFFu, node, 1);
        const int prev_node = (lane == 0) ? prev_tile_last_node : shifted_prev;

        const bool is_route_end = (node == 0 && prev_node != 0);
        const unsigned int end_mask = __ballot_sync(0xFFFFFFFFu, is_route_end);

        long long seg_load = 0LL;
        long long seg_dist = 0LL;
        if (node != 0)
        {
            seg_load = (long long)demands[node];
            seg_dist = (long long)gpu_edge_distance(
                prev_node, node, distances, coords, dist_stride, edge_type_code, use_matrix);
        }
        else if (is_route_end && return_to_depot != 0)
        {
            seg_dist = (long long)gpu_edge_distance(
                prev_node, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        }

        if (lane == 0 && prev_tile_last_node != 0)
        {
            seg_load += carry_load;
            seg_dist += carry_dist;
        }

        // Segmented inclusive warp scan: reset at tile boundary (lane 0) or after any depot 0.
        int seg_head = (lane == 0 || prev_node == 0) ? 1 : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            const long long left_load = __shfl_up_sync(0xFFFFFFFFu, seg_load, offset);
            const long long left_dist = __shfl_up_sync(0xFFFFFFFFu, seg_dist, offset);
            const int left_head = __shfl_up_sync(0xFFFFFFFFu, seg_head, offset);
            if (lane >= offset && seg_head == 0)
            {
                seg_load += left_load;
                seg_dist += left_dist;
                seg_head = left_head;
            }
        }

        if (end_mask != 0u && is_route_end)
        {
            lane_total_dist += seg_dist;
            if (has_capacity != 0 && seg_load > capacity)
            {
                lane_cap_viol += (seg_load - capacity);
            }
            if (has_max_distance != 0 && (double)seg_dist > max_route_distance)
            {
                lane_dist_viol += ((double)seg_dist - max_route_distance);
            }
            lane_vehicles += 1;
        }

        prev_tile_last_node = __shfl_sync(0xFFFFFFFFu, node, 31);
        carry_load = (prev_tile_last_node != 0) ? __shfl_sync(0xFFFFFFFFu, seg_load, 31) : 0LL;
        carry_dist = (prev_tile_last_node != 0) ? __shfl_sync(0xFFFFFFFFu, seg_dist, 31) : 0LL;
    }

    // Close trailing open route when `width` is a multiple of 32 and `row[width - 1] != 0`.
    if (prev_tile_last_node != 0 && lane == 0)
    {
        long long tail_dist = carry_dist;
        if (return_to_depot != 0)
        {
            tail_dist += (long long)gpu_edge_distance(
                prev_tile_last_node, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        }
        lane_total_dist += tail_dist;
        if (has_capacity != 0 && carry_load > capacity)
        {
            lane_cap_viol += (carry_load - capacity);
        }
        if (has_max_distance != 0 && (double)tail_dist > max_route_distance)
        {
            lane_dist_viol += ((double)tail_dist - max_route_distance);
        }
        lane_vehicles += 1;
    }

    const long long total_dist = warp_reduce_sum<long long>(lane_total_dist);
    const long long total_cap_viol = warp_reduce_sum<long long>(lane_cap_viol);
    const double total_dist_viol = warp_reduce_sum<double>(lane_dist_viol);
    const int active_vehicles = warp_reduce_sum<int>(lane_vehicles);

    if (lane == 0)
    {
        const long long veh_viol =
            (has_max_vehicles != 0 && active_vehicles > max_vehicles)
                ? (long long)(active_vehicles - max_vehicles)
                : 0LL;
        const double total_viol = (double)total_cap_viol + total_dist_viol + (double)veh_viol;
        const double cap_pen = (total_cap_viol > 0LL) ? (omega_q * (double)total_cap_viol) : 0.0;
        const double dist_pen = (total_dist_viol > 0.0) ? (omega_l * total_dist_viol) : 0.0;
        const double veh_pen = (veh_viol > 0LL) ? (omega_k * (double)veh_viol) : 0.0;
        const double pen = (double)total_dist + cap_pen + dist_pen + veh_pen;

        penalized_costs[idx] = pen;
        unpenalized_costs[idx] = (double)total_dist;
        violations[idx] = total_viol;
        num_vehicles[idx] = active_vehicles;
        if (raw_components != nullptr)
        {
            const size_t base = (size_t)idx * 3;
            raw_components[base + 0] = (double)total_cap_viol;
            raw_components[base + 1] = total_dist_viol;
            raw_components[base + 2] = (double)veh_viol;
        }
    }
}
