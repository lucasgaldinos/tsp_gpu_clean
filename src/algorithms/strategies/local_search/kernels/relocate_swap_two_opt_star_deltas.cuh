/**
 * O(1) penalized inter-route delta evaluators for `Relocate(1, 0)`, `Swap(1, 1)`, and `2-opt*` (`Task 17.29`).
 */

#ifndef GPU_RELOCATE_SWAP_TWO_OPT_STAR_DELTAS_CUH
#define GPU_RELOCATE_SWAP_TWO_OPT_STAR_DELTAS_CUH

#include "vrp_route_prefix_buffers.cuh"

static __device__ __forceinline__ double vrp_ls_eval_relocate(
    int r_a,
    int pos_v,
    int r_b,
    int ins_pos,
    int m,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len,
    const long long *r_load,
    const long long *r_dist,
    const int *__restrict__ demands,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    bool has_cap,
    long long capacity,
    bool has_dist,
    double max_route_distance,
    bool has_max_veh,
    int max_vehicles,
    double omega_q,
    double omega_l,
    double omega_k)
{
    const int off_a = r_offset[r_a];
    const int s_a = r_len[r_a];
    const int off_b = r_offset[r_b];
    const int s_b = r_len[r_b];
    const int v = cust_seq[off_a + pos_v];
    const long long q_v = (long long)demands[v];

    const long long old_load_a = r_load[r_a];
    const long long old_dist_a = r_dist[r_a];
    const long long old_load_b = r_load[r_b];
    const long long old_dist_b = r_dist[r_b];

    const long long new_load_a = old_load_a - q_v;
    long long new_dist_a = 0;
    int delta_m = 0;
    if (s_a == 1)
    {
        delta_m = -1;
    }
    else
    {
        const int prev_v = (pos_v == 0) ? 0 : cust_seq[off_a + pos_v - 1];
        long long rem_a = (long long)gpu_edge_distance(
            prev_v, v, distances, coords, dist_stride, edge_type_code, use_matrix);
        long long add_a = 0;
        if (pos_v < s_a - 1)
        {
            const int next_v = cust_seq[off_a + pos_v + 1];
            rem_a += (long long)gpu_edge_distance(
                v, next_v, distances, coords, dist_stride, edge_type_code, use_matrix);
            add_a = (long long)gpu_edge_distance(
                prev_v, next_v, distances, coords, dist_stride, edge_type_code, use_matrix);
        }
        else if (closed)
        {
            rem_a += (long long)gpu_edge_distance(
                v, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
            add_a = (long long)gpu_edge_distance(
                prev_v, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        }
        new_dist_a = old_dist_a - rem_a + add_a;
    }

    const long long new_load_b = old_load_b + q_v;
    const int prev_b = (ins_pos == 0) ? 0 : cust_seq[off_b + ins_pos - 1];
    long long new_dist_b = old_dist_b;
    if (ins_pos < s_b)
    {
        const int next_b = cust_seq[off_b + ins_pos];
        new_dist_b = old_dist_b - (long long)gpu_edge_distance(prev_b, next_b, distances, coords, dist_stride, edge_type_code, use_matrix) + (long long)gpu_edge_distance(prev_b, v, distances, coords, dist_stride, edge_type_code, use_matrix) + (long long)gpu_edge_distance(v, next_b, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    else if (closed)
    {
        new_dist_b = old_dist_b - (long long)gpu_edge_distance(prev_b, 0, distances, coords, dist_stride, edge_type_code, use_matrix) + (long long)gpu_edge_distance(prev_b, v, distances, coords, dist_stride, edge_type_code, use_matrix) + (long long)gpu_edge_distance(v, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    else
    {
        new_dist_b = old_dist_b + (long long)gpu_edge_distance(prev_b, v, distances, coords, dist_stride, edge_type_code, use_matrix);
    }

    double old_pen =
        vrp_ls_route_penalty(old_load_a, old_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(old_load_b, old_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    double new_pen =
        vrp_ls_route_penalty(new_load_a, new_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(new_load_b, new_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    if (has_max_veh && delta_m != 0)
    {
        const int old_ex = (m > max_vehicles) ? (m - max_vehicles) : 0;
        const int new_m = m + delta_m;
        const int new_ex = (new_m > max_vehicles) ? (new_m - max_vehicles) : 0;
        if (old_ex > 0)
        {
            old_pen += omega_k * (double)old_ex;
        }
        if (new_ex > 0)
        {
            new_pen += omega_k * (double)new_ex;
        }
    }
    return new_pen - old_pen;
}

static __device__ __forceinline__ double vrp_ls_eval_swap(
    int r_a,
    int pos_v,
    int r_b,
    int pos_u,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len,
    const long long *r_load,
    const long long *r_dist,
    const int *__restrict__ demands,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    bool has_cap,
    long long capacity,
    bool has_dist,
    double max_route_distance,
    double omega_q,
    double omega_l)
{
    const int off_a = r_offset[r_a];
    const int s_a = r_len[r_a];
    const int off_b = r_offset[r_b];
    const int s_b = r_len[r_b];
    const int v = cust_seq[off_a + pos_v];
    const int u = cust_seq[off_b + pos_u];

    const long long old_load_a = r_load[r_a];
    const long long old_dist_a = r_dist[r_a];
    const long long old_load_b = r_load[r_b];
    const long long old_dist_b = r_dist[r_b];

    const long long new_load_a = old_load_a - (long long)demands[v] + (long long)demands[u];
    const long long new_load_b = old_load_b - (long long)demands[u] + (long long)demands[v];

    const int prev_v = (pos_v == 0) ? 0 : cust_seq[off_a + pos_v - 1];
    long long rem_a = (long long)gpu_edge_distance(prev_v, v, distances, coords, dist_stride, edge_type_code, use_matrix);
    long long add_a = (long long)gpu_edge_distance(prev_v, u, distances, coords, dist_stride, edge_type_code, use_matrix);
    if (pos_v < s_a - 1)
    {
        const int next_v = cust_seq[off_a + pos_v + 1];
        rem_a += (long long)gpu_edge_distance(v, next_v, distances, coords, dist_stride, edge_type_code, use_matrix);
        add_a += (long long)gpu_edge_distance(u, next_v, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    else if (closed)
    {
        rem_a += (long long)gpu_edge_distance(v, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        add_a += (long long)gpu_edge_distance(u, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    const long long new_dist_a = old_dist_a - rem_a + add_a;

    const int prev_u = (pos_u == 0) ? 0 : cust_seq[off_b + pos_u - 1];
    long long rem_b = (long long)gpu_edge_distance(prev_u, u, distances, coords, dist_stride, edge_type_code, use_matrix);
    long long add_b = (long long)gpu_edge_distance(prev_u, v, distances, coords, dist_stride, edge_type_code, use_matrix);
    if (pos_u < s_b - 1)
    {
        const int next_u = cust_seq[off_b + pos_u + 1];
        rem_b += (long long)gpu_edge_distance(u, next_u, distances, coords, dist_stride, edge_type_code, use_matrix);
        add_b += (long long)gpu_edge_distance(v, next_u, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    else if (closed)
    {
        rem_b += (long long)gpu_edge_distance(u, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        add_b += (long long)gpu_edge_distance(v, 0, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
    const long long new_dist_b = old_dist_b - rem_b + add_b;

    const double old_pen =
        vrp_ls_route_penalty(old_load_a, old_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(old_load_b, old_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    const double new_pen =
        vrp_ls_route_penalty(new_load_a, new_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(new_load_b, new_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    return new_pen - old_pen;
}

static __device__ __forceinline__ double vrp_ls_eval_two_opt_star(
    int r_a,
    int cut_a,
    int r_b,
    int cut_b,
    int m,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len,
    const long long *r_load,
    const long long *r_dist,
    const long long *prefix_load,
    const long long *prefix_dist,
    const long long *suffix_dist,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    bool has_cap,
    long long capacity,
    bool has_dist,
    double max_route_distance,
    bool has_max_veh,
    int max_vehicles,
    double omega_q,
    double omega_l,
    double omega_k)
{
    const int off_a = r_offset[r_a];
    const int s_a = r_len[r_a];
    const int off_b = r_offset[r_b];
    const int s_b = r_len[r_b];
    if ((cut_a == -1 && cut_b == -1) || (cut_a == s_a - 1 && cut_b == s_b - 1))
    {
        return 0.0;
    }

    const long long old_load_a = r_load[r_a];
    const long long old_dist_a = r_dist[r_a];
    const long long old_load_b = r_load[r_b];
    const long long old_dist_b = r_dist[r_b];

    const long long head_load_a = (cut_a < 0) ? 0LL : prefix_load[off_a + cut_a];
    const long long head_open_a = (cut_a < 0) ? 0LL : prefix_dist[off_a + cut_a];
    const long long tail_load_a = old_load_a - head_load_a;
    const long long tail_close_a = (cut_a >= s_a - 1) ? 0LL : suffix_dist[off_a + cut_a + 1];

    const long long head_load_b = (cut_b < 0) ? 0LL : prefix_load[off_b + cut_b];
    const long long head_open_b = (cut_b < 0) ? 0LL : prefix_dist[off_b + cut_b];
    const long long tail_load_b = old_load_b - head_load_b;
    const long long tail_close_b = (cut_b >= s_b - 1) ? 0LL : suffix_dist[off_b + cut_b + 1];

    const long long new_load_a = head_load_a + tail_load_b;
    const long long new_load_b = head_load_b + tail_load_a;

    int delta_m = 0;
    long long new_dist_a = 0;
    if (cut_a < 0 && cut_b >= s_b - 1)
    {
        delta_m -= 1;
    }
    else if (cut_a < 0)
    {
        new_dist_a = (long long)gpu_edge_distance(
                         0, cust_seq[off_b + cut_b + 1], distances, coords, dist_stride, edge_type_code, use_matrix) +
                     tail_close_b;
    }
    else if (cut_b >= s_b - 1)
    {
        new_dist_a = head_open_a + (closed ? (long long)gpu_edge_distance(
                                                 cust_seq[off_a + cut_a], 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                                           : 0LL);
    }
    else
    {
        new_dist_a = head_open_a + (long long)gpu_edge_distance(cust_seq[off_a + cut_a], cust_seq[off_b + cut_b + 1], distances, coords, dist_stride, edge_type_code, use_matrix) + tail_close_b;
    }

    long long new_dist_b = 0;
    if (cut_b < 0 && cut_a >= s_a - 1)
    {
        delta_m -= 1;
    }
    else if (cut_b < 0)
    {
        new_dist_b = (long long)gpu_edge_distance(
                         0, cust_seq[off_a + cut_a + 1], distances, coords, dist_stride, edge_type_code, use_matrix) +
                     tail_close_a;
    }
    else if (cut_a >= s_a - 1)
    {
        new_dist_b = head_open_b + (closed ? (long long)gpu_edge_distance(
                                                 cust_seq[off_b + cut_b], 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                                           : 0LL);
    }
    else
    {
        new_dist_b = head_open_b + (long long)gpu_edge_distance(cust_seq[off_b + cut_b], cust_seq[off_a + cut_a + 1], distances, coords, dist_stride, edge_type_code, use_matrix) + tail_close_a;
    }

    double old_pen =
        vrp_ls_route_penalty(old_load_a, old_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(old_load_b, old_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    double new_pen =
        vrp_ls_route_penalty(new_load_a, new_dist_a, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l) + vrp_ls_route_penalty(new_load_b, new_dist_b, has_cap, capacity, has_dist, max_route_distance, omega_q, omega_l);
    if (has_max_veh && delta_m != 0)
    {
        const int old_ex = (m > max_vehicles) ? (m - max_vehicles) : 0;
        const int new_m = m + delta_m;
        const int new_ex = (new_m > max_vehicles) ? (new_m - max_vehicles) : 0;
        if (old_ex > 0)
        {
            old_pen += omega_k * (double)old_ex;
        }
        if (new_ex > 0)
        {
            new_pen += omega_k * (double)new_ex;
        }
    }
    return new_pen - old_pen;
}

#endif
