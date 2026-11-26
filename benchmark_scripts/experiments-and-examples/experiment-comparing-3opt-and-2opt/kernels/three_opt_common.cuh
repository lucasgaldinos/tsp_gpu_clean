/**
 * Shared device primitives for the 3-opt CUDA kernel family.
 *
 * `three_opt_batch.cu`, `three_opt_multi_move.cu`, and `three_opt_single.cu`
 * evaluate the exact same 3-opt neighborhood (3 2-opt + 4 pure 3-opt = 7
 * non-identity reconnections): they differ only in acceptance rule and batching
 * (one steepest move per pass vs. greedy disjoint multi-move vs. per-tour launch).
 * Keeping the distance lookup, cyclic-to-linear C_3 rotation mapping, candidate-list
 * / exhaustive neighborhood scan, and 3-step cooperative in-place reconnection here
 * mirrors `two_opt_common.cuh` so a fix can never diverge across variants.
 *
 * Reconnection decomposition (0 <= i0 < j0 < k0 <= k_max, or i0 < j0 == k0 for
 * pure 2-opt sentinel case 1):
 *   - Case 1 (2-opt on S1):       a -> S1' -> S2  -> f   [Step 1]
 *   - Case 2 (2-opt on S2):       a -> S1  -> S2' -> f   [Step 2]
 *   - Case 3 (2-opt on S1+S2):    a -> S2' -> S1' -> f   [Step 3]
 *   - Case 4 (pure 3-opt):        a -> S1' -> S2' -> f   [Step 1, Step 2]
 *   - Case 5 (pure 3-opt, swap):  a -> S2  -> S1  -> f   [Step 1, Step 2, Step 3]
 *   - Case 6 (pure 3-opt):        a -> S2  -> S1' -> f   [Step 2, Step 3]
 *   - Case 7 (pure 3-opt):        a -> S2' -> S1  -> f   [Step 1, Step 3]
 *
 * Where Steps 1 (`reverse [i0+1, j0]`) and 2 (`reverse [j0+1, k0]`) are strictly
 * disjoint and Step 3 (`reverse [i0+1, k0]`) executes after a single `__syncthreads()`.
 *
 * References:
 *   - Lin, S. (1965). "Computer Solutions of the Traveling Salesman Problem."
 *     Bell System Technical Journal, 44(10), 2245-2269.
 *   - Bentley, J. L. (1992). "Fast Algorithms for Geometric Traveling Salesman
 *     Problems." ORSA Journal on Computing, 4(4), 387-411.
 *   - Johnson, D. S., & McGeoch, L. A. (1997). "The Traveling Salesman Problem:
 *     A Case Study in Local Optimization." Local Search in Combinatorial Optimization.
 *   - Rocki, K., & Suda, R. (2012, 2013). GPU-accelerated 2-opt and 3-opt TSP local search.
 *   - O'Neil, M. A., & Burtscher, M. (2015). GPGPU-8, pp. 99-108.
 */

#ifndef GPU_THREE_OPT_COMMON_CUH
#define GPU_THREE_OPT_COMMON_CUH

#include "pairwise_device.cuh"

static __device__ __forceinline__ int three_opt_dist(
    int u,
    int v,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool use_matrix)
{
    if (use_matrix)
    {
        return distances[(size_t)u * (size_t)dist_stride + (size_t)v];
    }
    return gpu_pairwise_distance(u, v, nullptr, coords, dist_stride, edge_type_code);
}

/** Keep the steepest (most negative) move found by this thread. */
static __device__ __forceinline__ void three_opt_consider(
    int i0,
    int j0,
    int k0,
    int case_id,
    int delta,
    int *s_deltas,
    int *s_i,
    int *s_j,
    int *s_k,
    int *s_case,
    int tid)
{
    if (delta < s_deltas[tid])
    {
        s_deltas[tid] = delta;
        s_i[tid] = i0;
        s_j[tid] = j0;
        s_k[tid] = k0;
        s_case[tid] = case_id;
    }
}

/**
 * Map a pure 3-opt move specified in cyclic cut order (i -> j_cyc -> k_cyc -> i)
 * onto canonical sorted linear cuts `0 <= i0 < j0 < k0 < n` via the exact C_3
 * rotation isomorphism on `{Case 4, Case 5, Case 6, Case 7}`.
 */
static __device__ __forceinline__ void three_opt_consider_cyclic(
    int i,
    int j_cyc,
    int k_cyc,
    int cyc_case,
    int delta,
    int *s_deltas,
    int *s_i,
    int *s_j,
    int *s_k,
    int *s_case,
    int tid)
{
    if (delta >= s_deltas[tid])
    {
        return;
    }

    int i0, j0, k0, lin_case;
    if (i < j_cyc && j_cyc < k_cyc)
    {
        i0 = i;
        j0 = j_cyc;
        k0 = k_cyc;
        lin_case = cyc_case;
    }
    else if (k_cyc < i && i < j_cyc)
    {
        // Rotation 1: C_3 cycle 4 -> 6 -> 7 -> 4, Case 5 fixed
        i0 = k_cyc;
        j0 = i;
        k0 = j_cyc;
        if (cyc_case == 4)
            lin_case = 6;
        else if (cyc_case == 6)
            lin_case = 7;
        else if (cyc_case == 7)
            lin_case = 4;
        else
            lin_case = 5;
    }
    else
    {
        // Rotation 2: C_3 cycle 4 -> 7 -> 6 -> 4, Case 5 fixed
        i0 = j_cyc;
        j0 = k_cyc;
        k0 = i;
        if (cyc_case == 4)
            lin_case = 7;
        else if (cyc_case == 7)
            lin_case = 6;
        else if (cyc_case == 6)
            lin_case = 4;
        else
            lin_case = 5;
    }

    s_deltas[tid] = delta;
    s_i[tid] = i0;
    s_j[tid] = j0;
    s_k[tid] = k0;
    s_case[tid] = lin_case;
}

/**
 * Evaluate the valid pure 3-opt cyclic reconnections anchored at cut `i` using
 * new edges `(a, v1)` and `(b, v2)`, guarded by an exact non-negative lower bound.
 */
static __device__ __forceinline__ void three_opt_eval_cyclic_pair(
    int i,
    int n,
    int dist_stride,
    int d_ab,
    int p1,
    int d_av1,
    int p2,
    int d_bv2,
    const int *__restrict__ tour_buf,
    const int *__restrict__ edge_buf,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int edge_type_code,
    bool use_matrix,
    int *s_deltas,
    int *s_i,
    int *s_j,
    int *s_k,
    int *s_case,
    int tid)
{
    int rel1 = p1 - i;
    if (rel1 < 0)
        rel1 += n;
    int rel2 = p2 - i;
    if (rel2 < 0)
        rel2 += n;

    const int base_new = d_av1 + d_bv2 - d_ab;

    if (rel1 >= 2 && rel1 < rel2)
    {
        // Case 4: j_cyc = p1 (c = v1), k_cyc = p2 (e = v2); third edge (d, f)
        {
            const int j_cyc = p1;
            const int k_cyc = p2;
            const int lb = base_new - edge_buf[j_cyc] - edge_buf[k_cyc];
            if (lb < s_deltas[tid])
            {
                const int d_node = tour_buf[(j_cyc + 1 < n) ? (j_cyc + 1) : 0];
                const int f_node = tour_buf[(k_cyc + 1 < n) ? (k_cyc + 1) : 0];
                const int d_df = three_opt_dist(d_node, f_node, distances, coords, dist_stride, edge_type_code, use_matrix);
                three_opt_consider_cyclic(i, j_cyc, k_cyc, 4, lb + d_df, s_deltas, s_i, s_j, s_k, s_case, tid);
            }
        }

        // Case 5: j_cyc = p1 - 1 (d = v1), k_cyc = p2 (e = v2); third edge (c, f)
        {
            const int j_cyc = (p1 > 0) ? (p1 - 1) : (n - 1);
            const int k_cyc = p2;
            const int lb = base_new - edge_buf[j_cyc] - edge_buf[k_cyc];
            if (lb < s_deltas[tid])
            {
                const int c_node = tour_buf[j_cyc];
                const int f_node = tour_buf[(k_cyc + 1 < n) ? (k_cyc + 1) : 0];
                const int d_cf = three_opt_dist(c_node, f_node, distances, coords, dist_stride, edge_type_code, use_matrix);
                three_opt_consider_cyclic(i, j_cyc, k_cyc, 5, lb + d_cf, s_deltas, s_i, s_j, s_k, s_case, tid);
            }
        }

        // Case 6: j_cyc = p1 - 1 (d = v1), k_cyc = p2 - 1 (f = v2); third edge (c, e)
        {
            const int j_cyc = (p1 > 0) ? (p1 - 1) : (n - 1);
            const int k_cyc = (p2 > 0) ? (p2 - 1) : (n - 1);
            const int lb = base_new - edge_buf[j_cyc] - edge_buf[k_cyc];
            if (lb < s_deltas[tid])
            {
                const int c_node = tour_buf[j_cyc];
                const int e_node = tour_buf[k_cyc];
                const int d_ce = three_opt_dist(c_node, e_node, distances, coords, dist_stride, edge_type_code, use_matrix);
                three_opt_consider_cyclic(i, j_cyc, k_cyc, 6, lb + d_ce, s_deltas, s_i, s_j, s_k, s_case, tid);
            }
        }
    }
    else if (rel1 >= 2 && rel1 == rel2)
    {
        // Degenerate Or-opt-1 case (v1 == v2): relocate single city v1 between a and b (Case 5)
        const int j_cyc = (p1 > 0) ? (p1 - 1) : (n - 1);
        const int k_cyc = p1;
        const int lb = base_new - edge_buf[j_cyc] - edge_buf[k_cyc];
        if (lb < s_deltas[tid])
        {
            const int c_node = tour_buf[j_cyc];
            const int f_node = tour_buf[(k_cyc + 1 < n) ? (k_cyc + 1) : 0];
            const int d_cf = three_opt_dist(c_node, f_node, distances, coords, dist_stride, edge_type_code, use_matrix);
            three_opt_consider_cyclic(i, j_cyc, k_cyc, 5, lb + d_cf, s_deltas, s_i, s_j, s_k, s_case, tid);
        }
    }
    else if (rel2 >= 2 && rel2 < rel1)
    {
        // Case 7: j_cyc = p2 - 1 (d = v2), k_cyc = p1 (e = v1); third edge (c, f)
        const int j_cyc = (p2 > 0) ? (p2 - 1) : (n - 1);
        const int k_cyc = p1;
        const int lb = base_new - edge_buf[j_cyc] - edge_buf[k_cyc];
        if (lb < s_deltas[tid])
        {
            const int c_node = tour_buf[j_cyc];
            const int f_node = tour_buf[(k_cyc + 1 < n) ? (k_cyc + 1) : 0];
            const int d_cf = three_opt_dist(c_node, f_node, distances, coords, dist_stride, edge_type_code, use_matrix);
            three_opt_consider_cyclic(i, j_cyc, k_cyc, 7, lb + d_cf, s_deltas, s_i, s_j, s_k, s_case, tid);
        }
    }
}

/**
 * Shared neighborhood scan populating each thread's steepest candidate in
 * `(s_deltas[tid], s_i[tid], s_j[tid], s_k[tid], s_case[tid])`.
 */
static __device__ __forceinline__ void three_opt_scan_neighborhood(
    const int *__restrict__ tour_buf,
    int *__restrict__ edge_buf,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int n,
    int dist_stride,
    int edge_type_code,
    bool use_matrix,
    bool closed,
    bool use_candidates,
    const int *__restrict__ candidates,
    int cand_k,
    int inner_k,
    int *__restrict__ pos,
    int *s_deltas,
    int *s_i,
    int *s_j,
    int *s_k,
    int *s_case,
    int tid,
    int block_size)
{
    // Precompute consecutive edge costs: edge_buf[idx] = d(tour[idx], tour[(idx + 1) % n])
    const int edge_limit = closed ? (n - 1) : n;
    for (int idx = tid; idx < edge_limit; idx += block_size)
    {
        const int u = tour_buf[idx];
        const int v = tour_buf[(idx + 1 < n) ? (idx + 1) : 0];
        edge_buf[idx] = three_opt_dist(u, v, distances, coords, dist_stride, edge_type_code, use_matrix);
    }

    s_deltas[tid] = 0;
    s_i[tid] = -1;
    s_j[tid] = -1;
    s_k[tid] = -1;
    s_case[tid] = 0;

    if (use_candidates)
    {
        for (int idx = tid; idx < n; idx += block_size)
        {
            pos[tour_buf[idx]] = idx;
        }
        __syncthreads();

        const int eff_inner = (inner_k > 0 && inner_k < cand_k) ? inner_k : cand_k;

        for (int i = tid; i < n; i += block_size)
        {
            const int a = tour_buf[i];
            const int b = tour_buf[(i + 1 < n) ? (i + 1) : 0];
            const int d_ab = edge_buf[i];
            const int *cand_a = candidates + (size_t)a * (size_t)cand_k;
            const int *cand_b = candidates + (size_t)b * (size_t)cand_k;
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)dist_stride : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)dist_stride : nullptr;

            // 1. Seed s_deltas[tid] with all candidate 2-opt moves (strictly subsumes 2-opt)
            for (int r1 = 0; r1 < cand_k; ++r1)
            {
                const int v1 = cand_a[r1];
                const int p1 = pos[v1];
                if (p1 > i + 1 && !(i == 0 && p1 == n - 1))
                {
                    const int d_av1 = use_matrix ? row_a[v1] : gpu_pairwise_distance(a, v1, nullptr, coords, dist_stride, edge_type_code);
                    const int lb2 = d_av1 - d_ab - edge_buf[p1];
                    if (lb2 < s_deltas[tid])
                    {
                        const int d_node = tour_buf[(p1 + 1 < n) ? (p1 + 1) : 0];
                        const int d_bd = use_matrix ? row_b[d_node] : gpu_pairwise_distance(b, d_node, nullptr, coords, dist_stride, edge_type_code);
                        three_opt_consider(i, p1, p1, 1, lb2 + d_bd, s_deltas, s_i, s_j, s_k, s_case, tid);
                    }
                }
            }

            // 2. Pass A pure 3-opt: d(a, v1) < d(a, b) with sorted-list early exit
            for (int r1 = 0; r1 < cand_k; ++r1)
            {
                const int v1 = cand_a[r1];
                const int d_av1 = use_matrix ? row_a[v1] : gpu_pairwise_distance(a, v1, nullptr, coords, dist_stride, edge_type_code);
                if (d_av1 >= d_ab)
                    break;
                const int p1 = pos[v1];

                for (int r2 = 0; r2 < eff_inner; ++r2)
                {
                    const int v2 = cand_b[r2];
                    const int p2 = pos[v2];
                    const int d_bv2 = use_matrix ? row_b[v2] : gpu_pairwise_distance(b, v2, nullptr, coords, dist_stride, edge_type_code);
                    three_opt_eval_cyclic_pair(
                        i, n, dist_stride, d_ab,
                        p1, d_av1, p2, d_bv2,
                        tour_buf, edge_buf, distances, coords, edge_type_code, use_matrix,
                        s_deltas, s_i, s_j, s_k, s_case, tid);
                }
            }

            // 3. Pass B pure 3-opt: d(b, v2) < d(a, b) with d(a, v1) >= d(a, b)
            for (int r2 = 0; r2 < cand_k; ++r2)
            {
                const int v2 = cand_b[r2];
                const int d_bv2 = use_matrix ? row_b[v2] : gpu_pairwise_distance(b, v2, nullptr, coords, dist_stride, edge_type_code);
                if (d_bv2 >= d_ab)
                    break;
                const int p2 = pos[v2];

                for (int r1 = 0; r1 < eff_inner; ++r1)
                {
                    const int v1 = cand_a[r1];
                    const int d_av1 = use_matrix ? row_a[v1] : gpu_pairwise_distance(a, v1, nullptr, coords, dist_stride, edge_type_code);
                    if (d_av1 < d_ab)
                        continue;
                    const int p1 = pos[v1];
                    three_opt_eval_cyclic_pair(
                        i, n, dist_stride, d_ab,
                        p1, d_av1, p2, d_bv2,
                        tour_buf, edge_buf, distances, coords, edge_type_code, use_matrix,
                        s_deltas, s_i, s_j, s_k, s_case, tid);
                }
            }
        }
    }
    else
    {
        // Exhaustive scan over all 0 <= i < j < k <= k_max (supports both open and closed tours).
        __syncthreads();
        const int k_max = closed ? (n - 2) : (n - 1);

        for (int i = tid; i <= k_max - 2; i += block_size)
        {
            const int a = tour_buf[i];
            const int b = tour_buf[i + 1];
            const int d_ab = edge_buf[i];
            const int *row_a = use_matrix ? distances + (size_t)a * (size_t)dist_stride : nullptr;
            const int *row_b = use_matrix ? distances + (size_t)b * (size_t)dist_stride : nullptr;

            for (int j = i + 1; j <= k_max - 1; ++j)
            {
                const int c = tour_buf[j];
                const int d = tour_buf[j + 1];
                const int d_cd = edge_buf[j];
                const int d_ac = use_matrix ? row_a[c] : gpu_pairwise_distance(a, c, nullptr, coords, dist_stride, edge_type_code);
                const int d_ad = use_matrix ? row_a[d] : gpu_pairwise_distance(a, d, nullptr, coords, dist_stride, edge_type_code);
                const int d_bd = use_matrix ? row_b[d] : gpu_pairwise_distance(b, d, nullptr, coords, dist_stride, edge_type_code);
                const int *row_c = use_matrix ? distances + (size_t)c * (size_t)dist_stride : nullptr;
                const int *row_d = use_matrix ? distances + (size_t)d * (size_t)dist_stride : nullptr;

                if (j > i + 1)
                {
                    const int delta1 = d_ac + d_bd - d_ab - d_cd;
                    three_opt_consider(i, j, j, 1, delta1, s_deltas, s_i, s_j, s_k, s_case, tid);
                }

                const int old_ab_cd = d_ab + d_cd;
                for (int k = j + 1; k <= k_max; ++k)
                {
                    const int e = tour_buf[k];
                    const int f = tour_buf[(k + 1 < n) ? (k + 1) : 0];
                    const int d_ef = edge_buf[k];
                    const int old3 = old_ab_cd + d_ef;

                    const int d_ae = use_matrix ? row_a[e] : gpu_pairwise_distance(a, e, nullptr, coords, dist_stride, edge_type_code);
                    const int d_be = use_matrix ? row_b[e] : gpu_pairwise_distance(b, e, nullptr, coords, dist_stride, edge_type_code);
                    const int d_ce = use_matrix ? row_c[e] : gpu_pairwise_distance(c, e, nullptr, coords, dist_stride, edge_type_code);
                    const int d_bf = use_matrix ? row_b[f] : gpu_pairwise_distance(b, f, nullptr, coords, dist_stride, edge_type_code);
                    const int d_cf = use_matrix ? row_c[f] : gpu_pairwise_distance(c, f, nullptr, coords, dist_stride, edge_type_code);
                    const int d_df = use_matrix ? row_d[f] : gpu_pairwise_distance(d, f, nullptr, coords, dist_stride, edge_type_code);

                    if (k > j + 1)
                    {
                        three_opt_consider(i, j, k, 2, d_ce + d_df - d_cd - d_ef, s_deltas, s_i, s_j, s_k, s_case, tid);
                    }
                    if (closed || !(i == 0 && k == n - 1))
                    {
                        three_opt_consider(i, j, k, 3, d_ae + d_bf - d_ab - d_ef, s_deltas, s_i, s_j, s_k, s_case, tid);
                    }
                    three_opt_consider(i, j, k, 4, d_ac + d_be + d_df - old3, s_deltas, s_i, s_j, s_k, s_case, tid);
                    three_opt_consider(i, j, k, 5, d_ad + d_be + d_cf - old3, s_deltas, s_i, s_j, s_k, s_case, tid);
                    three_opt_consider(i, j, k, 6, d_ad + d_ce + d_bf - old3, s_deltas, s_i, s_j, s_k, s_case, tid);
                    three_opt_consider(i, j, k, 7, d_ae + d_bd + d_cf - old3, s_deltas, s_i, s_j, s_k, s_case, tid);
                }
            }
        }
    }
    __syncthreads();
}

static __device__ __forceinline__ void three_opt_reverse_segment(
    int *buf,
    int seg_start,
    int seg_end,
    int tid,
    int block_size)
{
    const int num_swaps = (seg_end - seg_start + 1) / 2;
    for (int s = tid; s < num_swaps; s += block_size)
    {
        const int tmp = buf[seg_start + s];
        buf[seg_start + s] = buf[seg_end - s];
        buf[seg_end - s] = tmp;
    }
}

/**
 * Cooperatively apply 3-opt reconnection `move_case` (`1..7`) at cuts `(mi, mj, mk)`
 * in-place across the whole thread block.
 */
static __device__ __forceinline__ void three_opt_apply_reconnection(
    int *tour_buf,
    int mi,
    int mj,
    int mk,
    int move_case,
    int tid,
    int block_size)
{
    const int seg_start = mi + 1;
    // Steps 1 & 2 operate on strictly disjoint index ranges [mi+1, mj] and [mj+1, mk]
    if (move_case == 1 || move_case == 4 || move_case == 5 || move_case == 7)
    {
        three_opt_reverse_segment(tour_buf, seg_start, mj, tid, block_size);
    }
    if (move_case == 2 || move_case == 4 || move_case == 5 || move_case == 6)
    {
        three_opt_reverse_segment(tour_buf, mj + 1, mk, tid, block_size);
    }
    __syncthreads();

    // Step 3 completes block-swap / mixed orientations on [mi+1, mk]
    if (move_case == 3 || move_case == 5 || move_case == 6 || move_case == 7)
    {
        three_opt_reverse_segment(tour_buf, seg_start, mk, tid, block_size);
        __syncthreads();
    }
}

/**
 * Parallel block reduction folding `s_deltas[0..block_size-1]` into slot `0`
 * (matching `two_opt_batch.cu` / `two_opt_single.cu`).
 */
static __device__ __forceinline__ void three_opt_reduce_block_min(
    int *s_deltas,
    int *s_i,
    int *s_j,
    int *s_k,
    int *s_case,
    int tid,
    int block_size)
{
    for (unsigned int s = block_size / 2; s > 32; s >>= 1)
    {
        if (tid < s && tid + s < block_size && s_deltas[tid + s] < s_deltas[tid])
        {
            s_deltas[tid] = s_deltas[tid + s];
            s_i[tid] = s_i[tid + s];
            s_j[tid] = s_j[tid + s];
            s_k[tid] = s_k[tid + s];
            s_case[tid] = s_case[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        const int remaining = block_size < 64 ? block_size : 64;
        for (int t = 1; t < remaining; ++t)
        {
            if (s_deltas[t] < s_deltas[0])
            {
                s_deltas[0] = s_deltas[t];
                s_i[0] = s_i[t];
                s_j[0] = s_j[t];
                s_k[0] = s_k[t];
                s_case[0] = s_case[t];
            }
        }
    }
    __syncthreads();
}

#endif // GPU_THREE_OPT_COMMON_CUH
