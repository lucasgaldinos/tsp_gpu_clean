/**
 * GPU VRP Route-Disjoint Multi-Move 2-opt* Inter-Route & Intra-Route Local Search Kernel (`Task 17.9`, `Task 17.24`, `Task 17.29`).
 *
 * Compacts active improving thread candidates (`s_type[t] > 0`) before greedy route-disjoint
 * selection, emits untouched routes in parallel across all threads (`N2`), and finishes with
 * intra-route `2-opt` refinement.
 */

#include "intra_route_two_opt.cuh"
#include "vrp_candidate_neighborhood_scan.cuh"

extern "C" __global__ void two_opt_star_multi_move_kernel(
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
    int *__restrict__ ws_int,           // [batch_size, 7 * n_cust + 2 * dist_stride]
    long long *__restrict__ ws_i64,     // [batch_size, 5 * n_cust]
    unsigned char *__restrict__ ws_used // [batch_size, n_cust] fallback when m > 1024
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
    __shared__ unsigned int route_used_mask[32];
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
    int *s_active = s_rb + block_size;
    int *s_int_base = s_active + block_size;

    const int int_stride = 7 * n_cust + 2 * dist_stride;
    int *int_buf = (use_shared != 0) ? s_int_base : (ws_int + (size_t)ind_idx * (size_t)int_stride);
    long long *i64_buf = (use_shared != 0) ? s_i64_base : (ws_i64 + (size_t)ind_idx * (size_t)(5 * n_cust));
    unsigned char *used_global = (ws_used != nullptr) ? (ws_used + (size_t)ind_idx * (size_t)n_cust) : nullptr;

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
        s_m = cvrp_ls_parse_delimited(row_routes, width, n_cust, cust_seq, r_offset, r_len);
    }
    __syncthreads();

    if (s_m <= 0)
    {
        return;
    }

    cvrp_ls_rebuild_state(
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

        for (int g = tid; g < n_cust; g += block_size)
        {
            const int v = cust_seq[g];
            cvrp_ls_eval_customer(
                v, s_m, dist_stride, cust_seq, r_offset, r_len, cust_route, cust_pos, r_load, r_dist, prefix_load,
                prefix_dist, suffix_dist, demands, distances, coords, candidates, cand_k, dist_stride, edge_type_code,
                use_matrix, closed, has_cap, capacity, has_dist, max_route_distance, has_max_veh, max_vehicles,
                omega_q, omega_l, omega_k, best_delta, best_type, best_arg1, best_arg2, best_ra, best_rb);
        }

        s_delta[tid] = best_delta;
        s_type[tid] = best_type;
        s_arg1[tid] = best_arg1;
        s_arg2[tid] = best_arg2;
        s_ra[tid] = best_ra;
        s_rb[tid] = best_rb;
        __syncthreads();

        if (tid == 0)
        {
            int n_active = 0;
            for (int t = 0; t < block_size; ++t)
            {
                if (s_type[t] > 0 && s_delta[t] < CVRP_DELTA_EPS)
                {
                    s_active[n_active++] = t;
                }
            }

            if (n_active == 0)
            {
                s_continue = 0;
            }
            else
            {
                const int m = s_m;
                const bool use_bitmask = (m <= 1024);
                if (use_bitmask)
                {
                    const int words = (m + 31) / 32;
                    for (int w = 0; w < words; ++w)
                    {
                        route_used_mask[w] = 0U;
                    }
                }
                else if (used_global != nullptr)
                {
                    for (int r = 0; r < m; ++r)
                    {
                        used_global[r] = 0;
                    }
                }

                for (int r = 0; r < m; ++r)
                {
                    route_move_slot[r] = -1;
                }

                int selected = 0;
                const int max_select = n_active < (m / 2) ? n_active : (m / 2);
                for (int step = 0; step < max_select && n_active > 0; ++step)
                {
                    int best_k = -1;
                    int best_t = -1;
                    double best_d = CVRP_DELTA_EPS;
                    int k = 0;
                    while (k < n_active)
                    {
                        const int t = s_active[k];
                        const int ra = s_ra[t];
                        const int rb = s_rb[t];
                        bool used_a = false;
                        bool used_b = false;
                        if (use_bitmask)
                        {
                            used_a = (route_used_mask[ra >> 5] & (1U << (ra & 31))) != 0U;
                            used_b = (route_used_mask[rb >> 5] & (1U << (rb & 31))) != 0U;
                        }
                        else if (used_global != nullptr)
                        {
                            used_a = used_global[ra] != 0;
                            used_b = used_global[rb] != 0;
                        }
                        if (used_a || used_b)
                        {
                            s_active[k] = s_active[--n_active];
                            continue;
                        }
                        if (s_delta[t] < best_d)
                        {
                            best_d = s_delta[t];
                            best_t = t;
                            best_k = k;
                        }
                        ++k;
                    }
                    if (best_t < 0)
                    {
                        break;
                    }
                    const int ra = s_ra[best_t];
                    const int rb = s_rb[best_t];
                    if (use_bitmask)
                    {
                        route_used_mask[ra >> 5] |= (1U << (ra & 31));
                        route_used_mask[rb >> 5] |= (1U << (rb & 31));
                    }
                    else if (used_global != nullptr)
                    {
                        used_global[ra] = 1;
                        used_global[rb] = 1;
                    }
                    route_move_slot[ra] = best_t;
                    route_move_slot[rb] = best_t;
                    s_active[best_k] = s_active[--n_active];
                    selected += 1;
                }

                if (selected == 0)
                {
                    s_continue = 0;
                }
                else
                {
                    s_continue = 1;
                    int write_pos = 0;
                    for (int r = 0; r < m; ++r)
                    {
                        const int emitted = cvrp_ls_emitted_len(
                            r, route_move_slot[r], s_type, s_arg1, s_arg2, s_ra, s_rb, r_len);
                        r_offset_next[r] = write_pos;
                        r_len_next[r] = emitted;
                        write_pos += emitted;
                    }
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
                cvrp_ls_emit_route(
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

        cvrp_ls_rebuild_state(
            s_m, dist_stride, cust_seq, r_offset, r_len, cust_route, cust_pos, r_load, r_dist, prefix_load,
            prefix_dist, suffix_dist, demands, distances, coords, dist_stride, edge_type_code, use_matrix, closed, tid,
            block_size);
    }

    if (intra_two_opt != 0 && max_iterations > 0)
    {
        for (int r = tid; r < s_m; r += block_size)
        {
            cvrp_ls_intra_two_opt_route(
                cust_seq + r_offset[r], r_len[r], distances, coords, dist_stride, edge_type_code, use_matrix, closed,
                has_dist, max_route_distance, omega_l, max_iterations);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        s_used_len = cvrp_ls_write_delimited(row_routes, width, s_m, cust_seq, r_offset, r_len);
    }
    __syncthreads();

    for (int p = s_used_len + tid; p < width; p += block_size)
    {
        row_routes[p] = 0;
    }
}
