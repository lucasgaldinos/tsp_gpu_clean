/**
 * Warp-Parallel VRAM-Native Bin-Packing (FFD / BFD) Fleet Repair Kernel (`Task 17.30`, `BUG-117`).
 *
 * One warp (32 lanes) processes one `0`-delimited VRP row:
 *   - Step A: 32-lane coalesced VRAM feasibility screen (`__ballot_sync` + `__shfl_up_sync`).
 *     Returns immediately with zero global writes when `m <= K_max` and all route loads <= Q.
 *   - Step B: 32-lane cooperative repack for violating rows using either dynamic shared memory
 *     (`use_shared == 1`, 1 warp/block) or monotonic global workspace (`use_shared == 0`, 4 warps/block).
 *     Supports FFD (`mode == 0`) and BFD (`mode == 1`) with 32-lane parallel position detour scans
 *     and both dense `distances` matrices and on-the-fly coordinate distances (`distances == nullptr`).
 */

#include "bin_packing_helpers.cuh"

extern "C" __global__ void bin_packing_pack_kernel(
    const int *__restrict__ routes_in, // [batch_size, in_width]
    const int *__restrict__ demands,   // [dist_stride]
    const int *__restrict__ distances, // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords, // [dist_stride, 2|3] or nullptr
    const int batch_size,
    const int in_width,
    const int out_width,
    const int dist_stride,
    const int edge_type_code,
    const int mode, // 0 = FFD, 1 = BFD
    const int return_to_depot,
    const int has_capacity,
    const long long capacity,
    const int has_max_distance,
    const double max_route_distance,
    const int has_max_vehicles,
    const int max_vehicles_constraint,
    const int cap_c,
    const int use_shared,
    double *__restrict__ ws_f64,    // [batch_size * cap_c] when !use_shared
    long long *__restrict__ ws_i64, // [batch_size * 2 * cap_c] when !use_shared
    int *__restrict__ ws_i32,       // [batch_size * (8 * cap_c + 1)] when !use_shared
    int *__restrict__ routes_out    // [batch_size, out_width]
)
{
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row_idx = blockIdx.x * warps_per_block + warp_in_block;
  const int lane = threadIdx.x & 31;
  if (row_idx >= batch_size)
  {
    return;
  }

  const int *row_in = routes_in + (size_t)row_idx * (size_t)in_width;

  // Step A: 32-lane coalesced VRAM feasibility screen.
  int prev_nz = 0;
  long long carry_load = 0LL;
  int m = 0;
  int total_nz = 0;
  bool any_cap_viol = false;
  bool any_zero = false;
  const unsigned int lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
  const unsigned int upto_mask = (lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u);

  for (int base = 0; base < in_width; base += 32)
  {
    const int idx = base + lane;
    const bool in_bounds = idx < in_width;
    const int tok = in_bounds ? row_in[idx] : 0;
    const int is_nz = (tok != 0) ? 1 : 0;
    if (__any_sync(0xFFFFFFFFu, in_bounds && tok == 0) != 0)
    {
      any_zero = true;
    }

    const unsigned int nz_mask = __ballot_sync(0xFFFFFFFFu, is_nz != 0);
    total_nz += __popc(nz_mask);
    const unsigned int start_mask = nz_mask & ~((nz_mask << 1) | (unsigned int)prev_nz);
    m += __popc(start_mask);

    const long long seg_load = bp_warp_tile_seg_load(
        is_nz, tok, demands, nz_mask, upto_mask, carry_load, lane);
    if (__any_sync(0xFFFFFFFFu, has_capacity != 0 && is_nz != 0 && seg_load > capacity) != 0)
    {
      any_cap_viol = true;
    }

    carry_load = __shfl_sync(0xFFFFFFFFu, seg_load, 31);
    prev_nz = (int)((nz_mask >> 31) & 1u);
  }

  const bool exceeds_fleet = (has_max_vehicles != 0) && (m > max_vehicles_constraint);
  if (!any_zero || m == 0 || (!exceeds_fleet && !any_cap_viol) || total_nz > cap_c)
  {
    return;
  }

  // Step B: Bind shared or global workspace buffers.
  extern __shared__ unsigned char smem[];
  const size_t c_sz = (size_t)cap_c;
  double *route_dist = (use_shared != 0)
                           ? reinterpret_cast<double *>(smem)
                           : (ws_f64 + (size_t)row_idx * c_sz);
  long long *i64_base = (use_shared != 0)
                            ? reinterpret_cast<long long *>(smem + c_sz * sizeof(double))
                            : (ws_i64 + (size_t)row_idx * (2 * c_sz));
  long long *route_load = i64_base;
  long long *tmp_load = i64_base + c_sz;

  int *i32_base = (use_shared != 0)
                      ? reinterpret_cast<int *>(tmp_load + c_sz)
                      : (ws_i32 + (size_t)row_idx * (8 * c_sz + 1));
  int *flat_nodes = i32_base;
  int *tmp_nodes = flat_nodes + c_sz;
  int *ejected = tmp_nodes + c_sz;
  int *route_offset = ejected + c_sz;
  int *route_len = route_offset + (c_sz + 1);
  int *route_order = route_len + c_sz;
  int *tmp_off = route_order + c_sz;
  int *tmp_len = tmp_off + c_sz;

  // Step B1: Cooperative 32-lane parse of initial sub-routes.
  prev_nz = 0;
  carry_load = 0LL;
  int nz_cursor = 0;
  int route_cursor = 0;

  for (int base = 0; base < in_width; base += 32)
  {
    const int idx = base + lane;
    const int tok = (idx < in_width) ? row_in[idx] : 0;
    const int is_nz = (tok != 0) ? 1 : 0;
    const unsigned int nz_mask = __ballot_sync(0xFFFFFFFFu, is_nz != 0);
    const int my_nz_pos = nz_cursor + __popc(nz_mask & lower_mask);
    if (is_nz != 0)
    {
      flat_nodes[my_nz_pos] = tok;
    }

    const unsigned int start_mask = nz_mask & ~((nz_mask << 1) | (unsigned int)prev_nz);
    const int my_start_id = route_cursor + __popc(start_mask & lower_mask);
    if ((start_mask & (1u << lane)) != 0u)
    {
      route_offset[my_start_id] = my_nz_pos;
    }

    const long long seg_load = bp_warp_tile_seg_load(
        is_nz, tok, demands, nz_mask, upto_mask, carry_load, lane);

    bool next_is_zero = false;
    if (is_nz != 0)
    {
      next_is_zero = (lane < 31) ? ((nz_mask & (1u << (lane + 1))) == 0u)
                                 : ((idx + 1 >= in_width) || (row_in[idx + 1] == 0));
    }
    if (next_is_zero)
    {
      const int end_route_id = route_cursor + __popc(start_mask & upto_mask) - 1;
      route_load[end_route_id] = seg_load;
    }

    nz_cursor += __popc(nz_mask);
    route_cursor += __popc(start_mask);
    carry_load = __shfl_sync(0xFFFFFFFFu, seg_load, 31);
    prev_nz = (int)((nz_mask >> 31) & 1u);
  }
  __syncwarp();

  if (lane == 0)
  {
    route_offset[m] = total_nz;
  }
  for (int i = lane; i < total_nz; i += 32)
  {
    tmp_nodes[i] = flat_nodes[i];
  }
  __syncwarp();

  for (int r = lane; r < m; r += 32)
  {
    const int len_r = route_offset[r + 1] - route_offset[r];
    route_len[r] = len_r;
    route_order[r] = r;
    tmp_off[r] = route_offset[r];
    tmp_len[r] = len_r;
    tmp_load[r] = route_load[r];
  }
  __syncwarp();

  // Step B2: Sort routes, retain <= K_max, truncate tails, and sort ejected customers.
  int num_kept = 0;
  int total_kept = 0;
  int n_eject = 0;

  if (lane == 0)
  {
    if (m > 1)
    {
      bp_heapsort_routes(route_order, m, tmp_load, has_capacity, capacity);
    }
    const int k_limit = (has_max_vehicles != 0) ? max_vehicles_constraint : m;
    for (int rank = 0; rank < m; ++rank)
    {
      const int orig_r = route_order[rank];
      const int off = tmp_off[orig_r];
      int rlen = tmp_len[orig_r];
      long long rload = tmp_load[orig_r];
      if (rank < k_limit)
      {
        if (has_capacity != 0 && rload > capacity)
        {
          while (rlen > 0 && rload > capacity)
          {
            --rlen;
            const int rem = tmp_nodes[off + rlen];
            rload -= (long long)demands[rem];
            ejected[n_eject++] = rem;
          }
        }
        if (rlen > 0)
        {
          route_offset[num_kept] = total_kept;
          route_len[num_kept] = rlen;
          route_load[num_kept] = rload;
          for (int k = 0; k < rlen; ++k)
          {
            flat_nodes[total_kept + k] = tmp_nodes[off + k];
          }
          total_kept += rlen;
          num_kept += 1;
        }
      }
      else
      {
        for (int k = 0; k < rlen; ++k)
        {
          ejected[n_eject++] = tmp_nodes[off + k];
        }
      }
    }
    if (n_eject > 1)
    {
      bp_heapsort_ejected(ejected, n_eject, demands);
    }
  }
  num_kept = __shfl_sync(0xFFFFFFFFu, num_kept, 0);
  total_kept = __shfl_sync(0xFFFFFFFFu, total_kept, 0);
  n_eject = __shfl_sync(0xFFFFFFFFu, n_eject, 0);
  __syncwarp();

  // Step B3: Cooperative 32-lane initial distance evaluation for kept routes.
  const bool closed = return_to_depot != 0;
  for (int r = 0; r < num_kept; ++r)
  {
    const int off = route_offset[r];
    const int rlen = route_len[r];
    double local_d = 0.0;
    for (int k = lane; k < rlen; k += 32)
    {
      const int prev_c = (k == 0) ? 0 : flat_nodes[off + k - 1];
      const int cur_c = flat_nodes[off + k];
      local_d += (double)gpu_pairwise_distance(
          prev_c, cur_c, distances, coords, dist_stride, edge_type_code);
    }
    if (closed && lane == 0 && rlen > 0)
    {
      local_d += (double)gpu_pairwise_distance(
          flat_nodes[off + rlen - 1], 0, distances, coords, dist_stride, edge_type_code);
    }
    const double r_d = warp_reduce_sum<double>(local_d);
    if (lane == 0)
    {
      route_dist[r] = r_d;
    }
  }
  __syncwarp();

  // Step B4: Insert each ejected customer using FFD (mode == 0) or BFD (mode == 1).
  const double kInf = __longlong_as_double(0x7ff0000000000000ULL);
  for (int e_idx = 0; e_idx < n_eject; ++e_idx)
  {
    const int u = ejected[e_idx];
    const long long q_u = (long long)demands[u];
    int best_r = -1;
    int best_pos = -1;
    double best_detour = kInf;
    long long best_res = 0LL;

    for (int r = 0; r < num_kept; ++r)
    {
      if (has_capacity != 0 && route_load[r] + q_u > capacity)
      {
        continue;
      }
      double cand_detour = kInf;
      int cand_pos = -1;
      bp_best_route_detour(
          flat_nodes, route_offset[r], route_len[r], u, distances, coords,
          dist_stride, edge_type_code, closed, has_max_distance,
          route_dist[r], max_route_distance, lane, cand_detour, cand_pos);
      if (cand_pos < 0)
      {
        continue;
      }
      if (mode == 0)
      {
        best_r = r;
        best_pos = cand_pos;
        best_detour = cand_detour;
        break;
      }
      const long long res = (has_capacity != 0) ? (capacity - (route_load[r] + q_u)) : 0LL;
      if (best_r < 0 || res < best_res || (res == best_res && cand_detour < best_detour))
      {
        best_r = r;
        best_pos = cand_pos;
        best_detour = cand_detour;
        best_res = res;
      }
    }

    if (best_r < 0 && (has_max_vehicles == 0 || num_kept < max_vehicles_constraint))
    {
      if (lane == 0)
      {
        flat_nodes[total_kept] = u;
        route_offset[num_kept] = total_kept;
        route_len[num_kept] = 1;
        route_load[num_kept] = q_u;
        double single_d = (double)gpu_pairwise_distance(
            0, u, distances, coords, dist_stride, edge_type_code);
        if (closed)
        {
          single_d += (double)gpu_pairwise_distance(
              u, 0, distances, coords, dist_stride, edge_type_code);
        }
        route_dist[num_kept] = single_d;
      }
      total_kept += 1;
      num_kept += 1;
      __syncwarp();
      continue;
    }

    if (best_r < 0)
    {
      long long fb_cap_delta = 0LL;
      double fb_dist_delta = kInf;
      for (int r = 0; r < num_kept; ++r)
      {
        double cand_detour = kInf;
        int cand_pos = -1;
        bp_best_route_detour(
            flat_nodes, route_offset[r], route_len[r], u, distances, coords,
            dist_stride, edge_type_code, closed, 0, route_dist[r], 0.0,
            lane, cand_detour, cand_pos);
        const long long cap_delta =
            ((has_capacity != 0 && route_load[r] + q_u > capacity)
                 ? (route_load[r] + q_u - capacity)
                 : 0LL) -
            ((has_capacity != 0 && route_load[r] > capacity) ? (route_load[r] - capacity) : 0LL);
        const double dist_delta =
            ((has_max_distance != 0 && route_dist[r] + cand_detour > max_route_distance)
                 ? (route_dist[r] + cand_detour - max_route_distance)
                 : 0.0) -
            ((has_max_distance != 0 && route_dist[r] > max_route_distance)
                 ? (route_dist[r] - max_route_distance)
                 : 0.0);
        if (best_r < 0 || cap_delta < fb_cap_delta ||
            (cap_delta == fb_cap_delta &&
             (dist_delta < fb_dist_delta ||
              (dist_delta == fb_dist_delta && cand_detour < best_detour))))
        {
          best_r = r;
          best_pos = cand_pos;
          best_detour = cand_detour;
          fb_cap_delta = cap_delta;
          fb_dist_delta = dist_delta;
        }
      }
    }

    if (best_r >= 0)
    {
      bp_shift_insert_and_update(
          flat_nodes, route_offset, route_len, route_load, route_dist,
          num_kept, total_kept, best_r, best_pos, u, q_u, best_detour, lane);
    }
  }

  // Step B5: Cooperative 32-lane canonical write-back [0, r_0..., 0, r_1..., 0, 0...].
  int *row_out = routes_out + (size_t)row_idx * (size_t)out_width;
  if (num_kept == 0)
  {
    for (int idx = lane; idx < out_width; idx += 32)
    {
      row_out[idx] = 0;
    }
    return;
  }

  for (int r = 0; r < num_kept; ++r)
  {
    const int off = route_offset[r];
    const int rlen = route_len[r];
    const int out_base = off + r;
    if (lane == 0 && out_base < out_width)
    {
      row_out[out_base] = 0;
    }
    for (int k = lane; k < rlen; k += 32)
    {
      if (out_base + 1 + k < out_width)
      {
        row_out[out_base + 1 + k] = flat_nodes[off + k];
      }
    }
  }
  for (int idx = total_kept + num_kept + lane; idx < out_width; idx += 32)
  {
    row_out[idx] = 0;
  }
}
