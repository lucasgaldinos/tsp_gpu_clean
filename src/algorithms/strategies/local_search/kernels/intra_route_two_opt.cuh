/**
 * Intra-route `2-opt` refinement pass (reusing `two_opt_delta_topology` from `two_opt_common.cuh`)
 * separated cleanly from inter-route neighborhood scans (`Task 17.29`).
 */

#ifndef GPU_INTRA_ROUTE_TWO_OPT_CUH
#define GPU_INTRA_ROUTE_TWO_OPT_CUH

#include "two_opt_common.cuh"
#include "vrp_route_prefix_buffers.cuh"

static __device__ __forceinline__ void vrp_ls_intra_two_opt_route(
    int *seq,
    int s,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    bool has_dist,
    double max_route_distance,
    double omega_l,
    int max_iterations)
{
    if (s < 2 || max_iterations <= 0)
    {
        return;
    }
    const int n_full = closed ? (s + 2) : (s + 1);
    const int j_limit = closed ? (n_full - 1) : n_full;
    const int i_limit = closed ? (n_full - 3) : (n_full - 2);

    for (int it = 0; it < max_iterations; ++it)
    {
        long long cur_dist = (long long)gpu_edge_distance(
            0, seq[0], distances, coords, dist_stride, edge_type_code, use_matrix);
        for (int k = 0; k < s - 1; ++k)
        {
            cur_dist += (long long)gpu_edge_distance(
                seq[k], seq[k + 1], distances, coords, dist_stride, edge_type_code, use_matrix);
        }
        if (closed)
        {
            cur_dist += (long long)gpu_edge_distance(
                seq[s - 1], 0, distances, coords, dist_stride, edge_type_code, use_matrix);
        }

        double best_pen_delta = CVRP_DELTA_EPS;
        int best_i = -1;
        int best_j = -1;

        for (int i = 0; i < i_limit; ++i)
        {
            const int a = (i == 0) ? 0 : seq[i - 1];
            const int b = seq[i];
            const int d_ab = gpu_edge_distance(a, b, distances, coords, dist_stride, edge_type_code, use_matrix);
            const int *row_a = use_matrix ? (distances + (size_t)a * (size_t)dist_stride) : nullptr;
            const int *row_b = use_matrix ? (distances + (size_t)b * (size_t)dist_stride) : nullptr;
            for (int j = i + 2; j < j_limit; ++j)
            {
                const int c = seq[j - 1];
                const int d = (j >= s) ? 0 : seq[j];
                const int delta_d = two_opt_delta_topology(
                    i, j, n_full, closed, false, a, b, d_ab, c, d,
                    row_a, row_b, distances, coords, dist_stride, edge_type_code, use_matrix);
                if (delta_d < 0)
                {
                    double pen_delta = (double)delta_d;
                    if (has_dist)
                    {
                        const long long new_dist = cur_dist + (long long)delta_d;
                        const double old_v = ((double)cur_dist > max_route_distance)
                                                 ? ((double)cur_dist - max_route_distance)
                                                 : 0.0;
                        const double new_v = ((double)new_dist > max_route_distance)
                                                 ? ((double)new_dist - max_route_distance)
                                                 : 0.0;
                        const double old_p = (old_v > 0.0) ? (omega_l * old_v) : 0.0;
                        const double new_p = (new_v > 0.0) ? (omega_l * new_v) : 0.0;
                        pen_delta += (new_p - old_p);
                    }
                    if (pen_delta < best_pen_delta)
                    {
                        best_pen_delta = pen_delta;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }
        if (best_i < 0)
        {
            break;
        }
        int left = best_i;
        int right = best_j - 1;
        while (left < right)
        {
            const int tmp = seq[left];
            seq[left] = seq[right];
            seq[right] = tmp;
            left += 1;
            right -= 1;
        }
    }
}

#endif
