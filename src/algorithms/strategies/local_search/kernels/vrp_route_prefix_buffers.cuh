/**
 * Route-state representation, 0-delimited parser/writer, prefix/suffix load & distance
 * tables, and move emission helpers for VRP 2-opt* inter-route CUDA kernels (`Task 17.29`).
 *
 * Uses `gpu_edge_distance` from `pairwise_device.cuh` (`DP10-19`, `Task 17.24`) and
 * IEEE-754-safe penalty multiplication (avoiding `0.0 * inf = NaN`).
 */

#ifndef GPU_VRP_ROUTE_PREFIX_BUFFERS_CUH
#define GPU_VRP_ROUTE_PREFIX_BUFFERS_CUH

#include "pairwise_device.cuh"

#define CVRP_MOVE_NONE 0
#define CVRP_MOVE_RELOCATE 1
#define CVRP_MOVE_SWAP 2
#define CVRP_MOVE_TWO_OPT_STAR 3
#define CVRP_DELTA_EPS (-1e-9)

static __device__ __forceinline__ double cvrp_ls_route_penalty(
    long long load,
    long long dist,
    bool has_cap,
    long long capacity,
    bool has_dist,
    double max_route_distance,
    double omega_q,
    double omega_l)
{
    if (load == 0 && dist == 0)
    {
        return 0.0;
    }
    const double cap_pen = (has_cap && load > capacity) ? (omega_q * (double)(load - capacity)) : 0.0;
    const double dist_pen = (has_dist && (double)dist > max_route_distance)
                                ? (omega_l * ((double)dist - max_route_distance))
                                : 0.0;
    return (double)dist + cap_pen + dist_pen;
}

static __device__ __forceinline__ void cvrp_ls_consider(
    double delta,
    int move_type,
    int arg1,
    int arg2,
    int r_a,
    int r_b,
    double &best_delta,
    int &best_type,
    int &best_arg1,
    int &best_arg2,
    int &best_ra,
    int &best_rb)
{
    if (delta < best_delta)
    {
        best_delta = delta;
        best_type = move_type;
        best_arg1 = arg1;
        best_arg2 = arg2;
        best_ra = r_a;
        best_rb = r_b;
    }
}

static __device__ __forceinline__ void cvrp_ls_rebuild_state(
    int m,
    int dim,
    const int *__restrict__ cust_seq,
    const int *__restrict__ r_offset,
    const int *__restrict__ r_len,
    int *cust_route,
    int *cust_pos,
    long long *r_load,
    long long *r_dist,
    long long *prefix_load,
    long long *prefix_dist,
    long long *suffix_dist,
    const int *__restrict__ demands,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    int tid,
    int block_size)
{
    for (int k = tid; k < dim; k += block_size)
    {
        cust_route[k] = -1;
        cust_pos[k] = -1;
    }
    __syncthreads();

    for (int r = tid; r < m; r += block_size)
    {
        const int off = r_offset[r];
        const int s = r_len[r];
        long long acc_load = 0;
        long long acc_dist = 0;
        int prev_node = 0;
        for (int p = 0; p < s; ++p)
        {
            const int node = cust_seq[off + p];
            cust_route[node] = r;
            cust_pos[node] = p;
            acc_load += (long long)demands[node];
            acc_dist += (long long)gpu_edge_distance(
                prev_node, node, distances, coords, dist_stride, edge_type_code, use_matrix);
            prefix_load[off + p] = acc_load;
            prefix_dist[off + p] = acc_dist;
            prev_node = node;
        }
        const long long close_edge =
            closed ? (long long)gpu_edge_distance(
                         prev_node, 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                   : 0LL;
        r_load[r] = acc_load;
        r_dist[r] = acc_dist + close_edge;

        long long tail_dist = close_edge;
        suffix_dist[off + s - 1] = tail_dist;
        for (int p = s - 2; p >= 0; --p)
        {
            tail_dist += (long long)gpu_edge_distance(
                cust_seq[off + p], cust_seq[off + p + 1], distances, coords, dist_stride, edge_type_code, use_matrix);
            suffix_dist[off + p] = tail_dist;
        }
    }
    __syncthreads();
}

static __device__ __forceinline__ int cvrp_ls_emitted_len(
    int r,
    int slot,
    const int *s_type,
    const int *s_arg1,
    const int *s_arg2,
    const int *s_ra,
    const int *s_rb,
    const int *r_len)
{
    if (slot < 0)
    {
        return r_len[r];
    }
    const int mtype = s_type[slot];
    const int r_a = s_ra[slot];
    const int s_a = r_len[r_a];
    const int s_b = r_len[s_rb[slot]];
    if (mtype == CVRP_MOVE_RELOCATE)
    {
        return (r == r_a) ? (s_a - 1) : (s_b + 1);
    }
    if (mtype == CVRP_MOVE_SWAP)
    {
        return (r == r_a) ? s_a : s_b;
    }
    if (mtype == CVRP_MOVE_TWO_OPT_STAR)
    {
        const int arg1 = s_arg1[slot];
        const int arg2 = s_arg2[slot];
        return (r == r_a) ? ((arg1 + 1) + (s_b - arg2 - 1)) : ((arg2 + 1) + (s_a - arg1 - 1));
    }
    return r_len[r];
}

static __device__ __forceinline__ int cvrp_ls_emit_route(
    int r,
    int slot,
    const int *s_type,
    const int *s_arg1,
    const int *s_arg2,
    const int *s_ra,
    const int *s_rb,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len,
    int *cust_seq_next,
    int write_pos)
{
    if (slot < 0)
    {
        const int off = r_offset[r];
        const int s = r_len[r];
        for (int p = 0; p < s; ++p)
        {
            cust_seq_next[write_pos + p] = cust_seq[off + p];
        }
        return s;
    }

    const int mtype = s_type[slot];
    const int r_a = s_ra[slot];
    const int r_b = s_rb[slot];
    const int arg1 = s_arg1[slot];
    const int arg2 = s_arg2[slot];
    const int off_a = r_offset[r_a];
    const int s_a = r_len[r_a];
    const int off_b = r_offset[r_b];
    const int s_b = r_len[r_b];

    int count = 0;
    if (mtype == CVRP_MOVE_RELOCATE)
    {
        if (r == r_a)
        {
            for (int p = 0; p < s_a; ++p)
            {
                if (p != arg1)
                {
                    cust_seq_next[write_pos + count++] = cust_seq[off_a + p];
                }
            }
        }
        else
        {
            const int v = cust_seq[off_a + arg1];
            for (int p = 0; p < arg2; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_b + p];
            }
            cust_seq_next[write_pos + count++] = v;
            for (int p = arg2; p < s_b; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_b + p];
            }
        }
    }
    else if (mtype == CVRP_MOVE_SWAP)
    {
        if (r == r_a)
        {
            const int u = cust_seq[off_b + arg2];
            for (int p = 0; p < s_a; ++p)
            {
                cust_seq_next[write_pos + count++] = (p == arg1) ? u : cust_seq[off_a + p];
            }
        }
        else
        {
            const int v = cust_seq[off_a + arg1];
            for (int p = 0; p < s_b; ++p)
            {
                cust_seq_next[write_pos + count++] = (p == arg2) ? v : cust_seq[off_b + p];
            }
        }
    }
    else if (mtype == CVRP_MOVE_TWO_OPT_STAR)
    {
        if (r == r_a)
        {
            for (int p = 0; p <= arg1; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_a + p];
            }
            for (int p = arg2 + 1; p < s_b; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_b + p];
            }
        }
        else
        {
            for (int p = 0; p <= arg2; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_b + p];
            }
            for (int p = arg1 + 1; p < s_a; ++p)
            {
                cust_seq_next[write_pos + count++] = cust_seq[off_a + p];
            }
        }
    }
    return count;
}

static __device__ __forceinline__ int cvrp_ls_parse_delimited(
    const int *__restrict__ row_in,
    int width,
    int n_cust,
    int *cust_seq,
    int *r_offset,
    int *r_len)
{
    int m = 0;
    int cust_write = 0;
    int cur_start = 0;
    int cur_len = 0;
    for (int p = 0; p < width; ++p)
    {
        const int tok = row_in[p];
        if (tok == 0)
        {
            if (cur_len > 0)
            {
                r_offset[m] = cur_start;
                r_len[m] = cur_len;
                m += 1;
                cur_start = cust_write;
                cur_len = 0;
            }
        }
        else if (cust_write < n_cust)
        {
            cust_seq[cust_write++] = tok;
            cur_len += 1;
        }
    }
    if (cur_len > 0)
    {
        r_offset[m] = cur_start;
        r_len[m] = cur_len;
        m += 1;
    }
    return m;
}

static __device__ __forceinline__ int cvrp_ls_write_delimited(
    int *__restrict__ row_out,
    int width,
    int m,
    const int *cust_seq,
    const int *r_offset,
    const int *r_len)
{
    if (m <= 0)
    {
        return 0;
    }
    int pos = 0;
    row_out[pos++] = 0;
    for (int r = 0; r < m; ++r)
    {
        const int off = r_offset[r];
        const int s = r_len[r];
        for (int p = 0; p < s; ++p)
        {
            if (pos < width - 1)
            {
                row_out[pos++] = cust_seq[off + p];
            }
        }
        if (pos < width)
        {
            row_out[pos++] = 0;
        }
    }
    return pos;
}

#endif
