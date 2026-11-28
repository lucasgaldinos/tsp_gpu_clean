/**
 * GPU VRP Steepest-Descent 2-opt* Inter-Route & Intra-Route Local Search Kernel (`Task 17.9`, `Task 17.24`, `Task 17.29`).
 *
 * One CUDA block owns one individual in `(batch_size, width)`. Evaluates
 * `Relocate(1, 0)`, `Swap(1, 1)`, and `2-opt*` across all threads in parallel,
 * selects the steepest improving move via `block_reduce_argmin_double_int` (`block_reductions.cuh`),
 * copies untouched routes in parallel across threads, and finishes with intra-route `2-opt`.
 */

#include "block_reductions.cuh"
#include "intra_route_two_opt.cuh"
#include "vrp_candidate_neighborhood_scan.cuh"

extern "C" __global__ void two_opt_star_steepest_kernel(
    int *__restrict__ routes,           // [batch_size, width] 0-delimited routes
    const int *__restrict__ demands,    // [dist_stride] int32
    const int *__restrict__ distances,  // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords,  // [dist_stride, 2|3] or nullptr
    const int *__restrict__ candidates, // [dist_stride, cand_k] or nullptr
    const int width,
    const int n_cust,
    const int dist_stride,
    const int batch_size,
    const int edge_type_code,
    const int cand_k,
    const int max_iterations,
    const int intra_two_opt,
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
    const int use_shared,
    int *__restrict__ ws_int,      // [batch_size, 7 * n_cust + 2 * dist_stride]
    long long *__restrict__ ws_i64 // [batch_size, 5 * n_cust]
)
{
    const int ind_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    if (ind_idx >= batch_size)
    {
        return;
    }

    extern __shared__ unsigned char smem[];
    __shared__ int s_m;
    __shared__ int s_used_len;
    __shared__ int s_continue;

    double *s_delta = reinterpret_cast<double *>(smem);
    unsigned char *cursor = smem + (size_t)block_size * sizeof(double);

    long long *s_i64_base = nullptr;
    if (use_shared != 0)
    {
        s_i64_base = reinterpret_cast<long long *>(cursor);
        cursor += (size_t)(5 * n_cust) * sizeof(long long);
    }

    int *s_type = reinterpret_cast<int *>(cursor);
    int *s_arg1 = s_type + block_size;
    int *s_arg2 = s_arg1 + block_size;
    int *s_ra = s_arg2 + block_size;
    int *s_rb = s_ra + block_size;
    int *s_g = s_rb + block_size;
    int *s_int_base = s_g + block_size;

    const int int_stride = 7 * n_cust + 2 * dist_stride;
    int *int_buf = (use_shared != 0) ? s_int_base : (ws_int + (size_t)ind_idx * (size_t)int_stride);
    long long *i64_buf = (use_shared != 0) ? s_i64_base : (ws_i64 + (size_t)ind_idx * (size_t)(5 * n_cust));

    int *cust_seq = int_buf;
    int *cust_seq_next = cust_seq + n_cust;
    int *r_offset = cust_seq_next + n_cust;
    int *r_offset_next = r_offset + n_cust;
    int *r_len = r_offset_next + n_cust;
    int *r_len_next = r_len + n_cust;
    int *route_move_slot = r_len_next + n_cust;
    int *cust_route = route_move_slot + n_cust;
    int *cust_pos = cust_route + dist_stride;

    long long *r_load = i64_buf;
    long long *r_dist = r_load + n_cust;
    long long *prefix_load = r_dist + n_cust;
    long long *prefix_dist = prefix_load + n_cust;
    long long *suffix_dist = prefix_dist + n_cust;

    int *row_routes = routes + (size_t)ind_idx * (size_t)width;
    const bool use_matrix = distances != nullptr;
    const bool closed = return_to_depot != 0;
    const bool has_cap = has_capacity != 0;
    const bool has_dist = has_max_distance != 0;
    const bool has_max_veh = has_max_vehicles != 0;

    if (tid == 0)
    {
        s_m = vrp_ls_parse_delimited(row_routes, width, n_cust, cust_seq, r_offset, r_len);
    }
    __syncthreads();

    if (s_m <= 0)
    {
        return;
    }

    vrp_ls_rebuild_state(
        s_m, dist_stride, cust_seq, r_offset, r_len, cust_route, cust_pos, r_load, r_dist, prefix_load, prefix_dist,
        suffix_dist, demands, distances, coords, dist_stride, edge_type_code, use_matrix, closed, tid, block_size);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        if (s_m < 2)
        {
            break;
        }

        double best_delta = CVRP_DELTA_EPS;
        int best_type = CVRP_MOVE_NONE;
        int best_arg1 = -1;
        int best_arg2 = -1;
        int best_ra = -1;
        int best_rb = -1;
        int best_g = -1;

        for (int g = tid; g < n_cust; g += block_size)
        {
            const int v = cust_seq[g];
            double g_delta = best_delta;
            int g_type = CVRP_MOVE_NONE;
            int g_arg1 = -1;
            int g_arg2 = -1;
            int g_ra = -1;
            int g_rb = -1;
            vrp_ls_eval_customer(
                v, s_m, dist_stride, cust_seq, r_offset, r_len, cust_route, cust_pos, r_load, r_dist, prefix_load,
                prefix_dist, suffix_dist, demands, distances, coords, candidates, cand_k, dist_stride, edge_type_code,
                use_matrix, closed, has_cap, capacity, has_dist, max_route_distance, has_max_veh, max_vehicles,
                omega_q, omega_l, omega_k, g_delta, g_type, g_arg1, g_arg2, g_ra, g_rb);
            if (g_type != CVRP_MOVE_NONE && g_delta < best_delta)
            {
                best_delta = g_delta;
                best_type = g_type;
                best_arg1 = g_arg1;
                best_arg2 = g_arg2;
                best_ra = g_ra;
                best_rb = g_rb;
                best_g = g;
            }
        }

        s_type[tid] = best_type;
        s_arg1[tid] = best_arg1;
        s_arg2[tid] = best_arg2;
        s_ra[tid] = best_ra;
        s_rb[tid] = best_rb;
        __syncthreads();

        double win_delta = (best_type != CVRP_MOVE_NONE) ? best_delta : 0.0;
        int win_g = (best_type != CVRP_MOVE_NONE) ? best_g : -1;
        block_reduce_argmin_double_int(win_delta, win_g, s_delta, s_g);

        if (tid == 0)
        {
            if (win_g < 0 || win_delta >= CVRP_DELTA_EPS)
            {
                s_continue = 0;
            }
            else
            {
                s_continue = 1;
                const int win_tid = win_g % block_size;
                const int m = s_m;
                const int win_ra = s_ra[win_tid];
                const int win_rb = s_rb[win_tid];
                int write_pos = 0;
                for (int r = 0; r < m; ++r)
                {
                    const int slot = (r == win_ra || r == win_rb) ? win_tid : -1;
                    route_move_slot[r] = slot;
                    const int emitted = vrp_ls_emitted_len(r, slot, s_type, s_arg1, s_arg2, s_ra, s_rb, r_len);
                    r_offset_next[r] = write_pos;
                    r_len_next[r] = emitted;
                    write_pos += emitted;
                }
            }
        }
        __syncthreads();

        if (s_continue == 0)
        {
            break;
        }

        for (int g = tid; g < n_cust; g += block_size)
        {
            const int v = cust_seq[g];
            const int r = cust_route[v];
            if (route_move_slot[r] < 0)
            {
                cust_seq_next[r_offset_next[r] + cust_pos[v]] = v;
            }
        }
        for (int r = tid; r < s_m; r += block_size)
        {
            if (route_move_slot[r] >= 0)
            {
                vrp_ls_emit_route(
                    r, route_move_slot[r], s_type, s_arg1, s_arg2, s_ra, s_rb, cust_seq, r_offset, r_len,
                    cust_seq_next, r_offset_next[r]);
            }
        }
        __syncthreads();

        for (int k = tid; k < n_cust; k += block_size)
        {
            cust_seq[k] = cust_seq_next[k];
        }
        if (tid == 0)
        {
            const int m = s_m;
            int new_m = 0;
            for (int r = 0; r < m; ++r)
            {
                if (r_len_next[r] > 0)
                {
                    r_offset[new_m] = r_offset_next[r];
                    r_len[new_m] = r_len_next[r];
                    new_m += 1;
                }
            }
            s_m = new_m;
        }
        __syncthreads();

        vrp_ls_rebuild_state(
            s_m, dist_stride, cust_seq, r_offset, r_len, cust_route, cust_pos, r_load, r_dist, prefix_load,
            prefix_dist, suffix_dist, demands, distances, coords, dist_stride, edge_type_code, use_matrix, closed, tid,
            block_size);
    }

    if (intra_two_opt != 0 && max_iterations > 0)
    {
        for (int r = tid; r < s_m; r += block_size)
        {
            vrp_ls_intra_two_opt_route(
                cust_seq + r_offset[r], r_len[r], distances, coords, dist_stride, edge_type_code, use_matrix, closed,
                has_dist, max_route_distance, omega_l, max_iterations);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        s_used_len = vrp_ls_write_delimited(row_routes, width, s_m, cust_seq, r_offset, r_len);
    }
    __syncthreads();

    for (int p = s_used_len + tid; p < width; p += block_size)
    {
        row_routes[p] = 0;
    }
}
