/**
 * Inter-route candidate-list and full-matrix neighborhood scan (`cvrp_ls_eval_customer`)
 * for VRP 2-opt* local search CUDA kernels (`Task 17.29`).
 */

#ifndef GPU_VRP_CANDIDATE_NEIGHBORHOOD_SCAN_CUH
#define GPU_VRP_CANDIDATE_NEIGHBORHOOD_SCAN_CUH

#include "relocate_swap_two_opt_star_deltas.cuh"

static __device__ __forceinline__ void cvrp_ls_eval_customer(
    int v,
    int m,
    int dim,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len,
    const int *cust_route,
    const int *cust_pos,
    const long long *r_load,
    const long long *r_dist,
    const long long *prefix_load,
    const long long *prefix_dist,
    const long long *suffix_dist,
    const int *__restrict__ demands,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    const int *__restrict__ candidates,
    int cand_k,
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
    double omega_k,
    double &best_delta,
    int &best_type,
    int &best_arg1,
    int &best_arg2,
    int &best_ra,
    int &best_rb)
{
    const int r_a = cust_route[v];
    const int pos_v = cust_pos[v];
    if (r_a < 0)
    {
        return;
    }

    if (candidates != nullptr && cand_k > 0)
    {
        const int *row_c = candidates + (size_t)v * (size_t)cand_k;
        for (int idx = 0; idx < cand_k; ++idx)
        {
            const int u = row_c[idx];
            if (u == 0)
            {
                for (int r_b = 0; r_b < m; ++r_b)
                {
                    if (r_b == r_a)
                    {
                        continue;
                    }
                    cvrp_ls_consider(
                        cvrp_ls_eval_relocate(
                            r_a, pos_v, r_b, 0, m, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances,
                            coords, dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist,
                            max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                        CVRP_MOVE_RELOCATE, pos_v, 0, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                        best_rb);
                    cvrp_ls_consider(
                        cvrp_ls_eval_two_opt_star(
                            r_a, pos_v, r_b, -1, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                            suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap,
                            capacity, has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l,
                            omega_k),
                        CVRP_MOVE_TWO_OPT_STAR, pos_v, -1, r_a, r_b, best_delta, best_type, best_arg1, best_arg2,
                        best_ra, best_rb);
                }
                continue;
            }
            if (u < 0 || u >= dim)
            {
                continue;
            }
            const int r_b = cust_route[u];
            if (r_b < 0 || r_b == r_a)
            {
                continue;
            }
            const int pos_u = cust_pos[u];

            cvrp_ls_consider(
                cvrp_ls_eval_relocate(
                    r_a, pos_v, r_b, pos_u + 1, m, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances,
                    coords, dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist,
                    max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                CVRP_MOVE_RELOCATE, pos_v, pos_u + 1, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                best_rb);

            if (pos_u == 0)
            {
                cvrp_ls_consider(
                    cvrp_ls_eval_relocate(
                        r_a, pos_v, r_b, 0, m, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances, coords,
                        dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist,
                        max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                    CVRP_MOVE_RELOCATE, pos_v, 0, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                    best_rb);
                cvrp_ls_consider(
                    cvrp_ls_eval_two_opt_star(
                        r_a, pos_v, r_b, -1, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                        suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap,
                        capacity, has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                    CVRP_MOVE_TWO_OPT_STAR, pos_v, -1, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                    best_rb);
            }

            cvrp_ls_consider(
                cvrp_ls_eval_swap(
                    r_a, pos_v, r_b, pos_u, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances, coords,
                    dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist, max_route_distance,
                    omega_q, omega_l),
                CVRP_MOVE_SWAP, pos_v, pos_u, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra, best_rb);

            cvrp_ls_consider(
                cvrp_ls_eval_two_opt_star(
                    r_a, pos_v, r_b, pos_u, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                    suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity,
                    has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                CVRP_MOVE_TWO_OPT_STAR, pos_v, pos_u, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                best_rb);

            if (pos_v == 0)
            {
                cvrp_ls_consider(
                    cvrp_ls_eval_two_opt_star(
                        r_a, -1, r_b, pos_u, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                        suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap,
                        capacity, has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                    CVRP_MOVE_TWO_OPT_STAR, -1, pos_u, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                    best_rb);
            }
        }
        return;
    }

    for (int r_b = 0; r_b < m; ++r_b)
    {
        if (r_b == r_a)
        {
            continue;
        }
        const int s_b = r_len[r_b];
        cvrp_ls_consider(
            cvrp_ls_eval_relocate(
                r_a, pos_v, r_b, 0, m, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances, coords,
                dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist, max_route_distance,
                has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
            CVRP_MOVE_RELOCATE, pos_v, 0, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra, best_rb);
        cvrp_ls_consider(
            cvrp_ls_eval_two_opt_star(
                r_a, pos_v, r_b, -1, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity,
                has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
            CVRP_MOVE_TWO_OPT_STAR, pos_v, -1, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra, best_rb);

        for (int pos_u = 0; pos_u < s_b; ++pos_u)
        {
            cvrp_ls_consider(
                cvrp_ls_eval_relocate(
                    r_a, pos_v, r_b, pos_u + 1, m, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances,
                    coords, dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist,
                    max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                CVRP_MOVE_RELOCATE, pos_v, pos_u + 1, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                best_rb);
            if (r_a < r_b)
            {
                cvrp_ls_consider(
                    cvrp_ls_eval_swap(
                        r_a, pos_v, r_b, pos_u, cust_seq, r_offset, r_len, r_load, r_dist, demands, distances, coords,
                        dist_stride, edge_type_code, use_matrix, closed, has_cap, capacity, has_dist,
                        max_route_distance, omega_q, omega_l),
                    CVRP_MOVE_SWAP, pos_v, pos_u, r_a, r_b, best_delta, best_type, best_arg1, best_arg2, best_ra,
                    best_rb);
                cvrp_ls_consider(
                    cvrp_ls_eval_two_opt_star(
                        r_a, pos_v, r_b, pos_u, m, cust_seq, r_offset, r_len, r_load, r_dist, prefix_load, prefix_dist,
                        suffix_dist, distances, coords, dist_stride, edge_type_code, use_matrix, closed, has_cap,
                        capacity, has_dist, max_route_distance, has_max_veh, max_vehicles, omega_q, omega_l, omega_k),
                    CVRP_MOVE_TWO_OPT_STAR, pos_v, pos_u, r_a, r_b, best_delta, best_type, best_arg1, best_arg2,
                    best_ra, best_rb);
            }
        }
    }
}

#endif
