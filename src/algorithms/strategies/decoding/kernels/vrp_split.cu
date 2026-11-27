/**
 * Warp-Parallel GPU VRP Split Decoder & Depot-Delimiter Stripper Kernels
 * (`Task 17.21`, `DP17-2`, `DP17-3`, `DP17-6`, `BUG-113`).
 *
 * Implements:
 *   1. `vrp_split_kernel`: 1 warp (32 threads) per Giant-Tour block:
 *      - All 32 lanes cooperatively load `tour[0..n_cust-1]`, evaluate `d(0, tour[k])`
 *        and consecutive `d(tour[k-1], tour[k])` via `gpu_edge_distance`, and build
 *        prefix sums `W[0..n_cust]` and `P[0..n_cust]` with tiled `__shfl_up_sync`.
 *      - Prins mode (`mode_prins == 1`): warp-parallel Bellman DAG relaxation where
 *        lane `t` strides predecessors `i = j - 1 - t, j - 1 - t - 32, ... >= 0`,
 *        reducing `(best_cost, best_i)` via deterministic `warp_reduce_argmin_double_int`.
 *      - Greedy mode (`mode_prins == 0`): O(n_cust) prefix-assisted linear scan sharing
 *        the 32-lane prefix build, cooperative scoring, and cooperative write-back.
 *      - Cooperative 32-lane route scoring (`warp_reduce_sum`) and route write-back.
 *      - Shared memory caches `V, W, P, pred, d0` (`32 * n_cust + 28` bytes, ~32 KB at
 *        `n_cust = 1024` < 48 KB Pascal `sm_61` cap) with global workspace fallback.
 *   2. `strip_depot_delimiters_kernel`: Zero-D2H `(N, W) -> (N, n_cust)` extractor.
 */

#include "block_reductions.cuh"
#include "pairwise_device.cuh"

extern "C" __global__ void vrp_split_kernel(
    const int *__restrict__ giant_tours, // [batch_size, n_cust]
    const int *__restrict__ demands,     // [dist_stride] int32 (demands[0] == 0)
    const int *__restrict__ distances,   // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords,   // [dist_stride, 2|3] or nullptr
    const int n_cust,                    // context.dimension - 1
    const int dist_stride,               // context.dimension
    const int batch_size,
    const int edge_type_code,             // GpuEdgeTypeCode for coordinate mode
    const int mode_prins,                 // 1 = Prins warp-parallel DP, 0 = Greedy scan
    const int return_to_depot,            // 1 = closed route, 0 = open route (OVRP)
    const int has_capacity,               // 1 if Q is finite
    const long long capacity,             // Q
    const int has_max_distance,           // 1 if L_max is finite
    const double max_route_distance,      // L_max
    const int has_max_vehicles,           // 1 if K_max constraint is active
    const int max_vehicles_constraint,    // K_max
    const int buffer_k,                   // W = dist_stride + buffer_k
    const double omega_q,                 // Penalty weight for capacity excess
    const double omega_l,                 // Penalty weight for route distance excess
    const double omega_k,                 // Penalty weight for fleet size excess
    const int use_shared,                 // 1 = shared-memory buffers, 0 = global workspace
    double *__restrict__ workspace_V,     // [batch_size, n_cust + 1] when !use_shared
    long long *__restrict__ workspace_W,  // [batch_size, n_cust + 1] when !use_shared
    long long *__restrict__ workspace_P,  // [batch_size, n_cust + 1] when !use_shared
    int *__restrict__ workspace_pred,     // [batch_size, n_cust + 1] when !use_shared
    int *__restrict__ workspace_d0,       // [batch_size, max(1, n_cust)] when !use_shared
    int *__restrict__ routes_out,         // [batch_size, dist_stride + buffer_k] or nullptr
    double *__restrict__ penalized_out,   // [batch_size]
    double *__restrict__ unpenalized_out, // [batch_size]
    double *__restrict__ violations_out,  // [batch_size]
    int *__restrict__ num_vehicles_out    // [batch_size]
)
{
  const int ind_idx = blockIdx.x;
  const int lane = threadIdx.x;
  if (ind_idx >= batch_size)
  {
    return;
  }

  const bool use_matrix = distances != nullptr;
  const bool closed = return_to_depot != 0;
  const bool has_cap = has_capacity != 0;
  const bool has_dist = has_max_distance != 0;
  const int row_width = dist_stride + buffer_k;

  if (n_cust <= 0)
  {
    if (routes_out != nullptr)
    {
      int *row_out = routes_out + (size_t)ind_idx * (size_t)row_width;
      for (int idx = lane; idx < row_width; idx += 32)
      {
        row_out[idx] = 0;
      }
    }
    if (lane == 0)
    {
      penalized_out[ind_idx] = 0.0;
      unpenalized_out[ind_idx] = 0.0;
      violations_out[ind_idx] = 0.0;
      num_vehicles_out[ind_idx] = 0;
    }
    return;
  }

  extern __shared__ unsigned char smem[];
  const size_t state_len = (size_t)(n_cust + 1);
  const size_t cust_len = (size_t)n_cust;

  double *V = nullptr;
  long long *W = nullptr;
  long long *P = nullptr;
  int *pred = nullptr;
  int *d0 = nullptr;

  if (use_shared != 0)
  {
    V = reinterpret_cast<double *>(smem);
    W = reinterpret_cast<long long *>(smem + state_len * sizeof(double));
    P = reinterpret_cast<long long *>(smem + state_len * (sizeof(double) + sizeof(long long)));
    pred = reinterpret_cast<int *>(smem + state_len * (sizeof(double) + 2 * sizeof(long long)));
    d0 = reinterpret_cast<int *>(
        smem + state_len * (sizeof(double) + 2 * sizeof(long long) + sizeof(int)));
  }
  else
  {
    V = workspace_V + (size_t)ind_idx * state_len;
    W = workspace_W + (size_t)ind_idx * state_len;
    P = workspace_P + (size_t)ind_idx * state_len;
    pred = workspace_pred + (size_t)ind_idx * state_len;
    d0 = workspace_d0 + (size_t)ind_idx * cust_len;
  }

  const int *g_tour = giant_tours + (size_t)ind_idx * cust_len;

  // Phase 1: Cooperative 32-lane prefix sums W[0..n_cust], P[0..n_cust], and d0[0..n_cust-1].
  if (lane == 0)
  {
    V[0] = 0.0;
    W[0] = 0LL;
    P[0] = 0LL;
    pred[0] = -1;
  }

  long long carry_w = 0LL;
  long long carry_p = 0LL;
  for (int base = 0; base < n_cust; base += 32)
  {
    const int k = base + lane;
    long long w_val = 0LL;
    long long p_val = 0LL;
    if (k < n_cust)
    {
      const int cust = g_tour[k];
      w_val = (long long)demands[cust];
      d0[k] = gpu_edge_distance(0, cust, distances, coords, dist_stride, edge_type_code, use_matrix);
      if (k > 0)
      {
        const int prev_cust = g_tour[k - 1];
        p_val = (long long)gpu_edge_distance(
            prev_cust, cust, distances, coords, dist_stride, edge_type_code, use_matrix);
      }
    }

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const long long up_w = __shfl_up_sync(0xFFFFFFFFu, w_val, offset);
      const long long up_p = __shfl_up_sync(0xFFFFFFFFu, p_val, offset);
      if (lane >= offset)
      {
        w_val += up_w;
        p_val += up_p;
      }
    }
    w_val += carry_w;
    p_val += carry_p;

    if (k < n_cust)
    {
      W[k + 1] = w_val;
      P[k + 1] = p_val;
    }
    carry_w = __shfl_sync(0xFFFFFFFFu, w_val, 31);
    carry_p = __shfl_sync(0xFFFFFFFFu, p_val, 31);
  }
  if (use_shared == 0)
  {
    __threadfence_block();
  }
  __syncwarp();

  // Phase 2: Partitioning (Prins warp-parallel Bellman relaxation OR Greedy linear scan).
  const double kInf = __longlong_as_double(0x7ff0000000000000ULL);

  if (mode_prins != 0)
  {
    for (int j = 1; j <= n_cust; ++j)
    {
      const long long w_j = W[j];
      const long long p_j = P[j];
      const long long lane0_ret = (lane == 0 && closed)
                                      ? (long long)gpu_edge_distance(
                                            g_tour[j - 1], 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                                      : 0LL;
      const long long return_j = __shfl_sync(0xFFFFFFFFu, lane0_ret, 0);

      double best_val = kInf;
      int best_i = -1;

      for (int i = j - 1 - lane; i >= 0; i -= 32)
      {
        const long long load = w_j - W[i];
        const long long suffix_chain = p_j - P[i + 1];
        const long long open_dist = (long long)d0[i] + suffix_chain;
        const long long arc_dist = open_dist + return_j;

        if (i == j - 1)
        {
          // Single-customer arc always admissible (penalized when q_u > Q or single-route > L_max).
          const long long cap_ex = (has_cap && load > capacity) ? (load - capacity) : 0LL;
          const double dist_ex =
              (has_dist && (double)arc_dist > max_route_distance)
                  ? ((double)arc_dist - max_route_distance)
                  : 0.0;
          const double cap_pen = (cap_ex > 0LL) ? (omega_q * (double)cap_ex) : 0.0;
          const double dist_pen = (dist_ex > 0.0) ? (omega_l * dist_ex) : 0.0;
          const double cand = V[i] + (double)arc_dist + cap_pen + dist_pen;
          if (argmin_pair_beats(cand, i, best_val, best_i))
          {
            best_val = cand;
            best_i = i;
          }
        }
        else
        {
          if (has_cap && load > capacity)
          {
            break;
          }
          if (has_dist && (double)(suffix_chain + return_j) > max_route_distance)
          {
            break;
          }
          if (has_dist && (double)arc_dist > max_route_distance)
          {
            continue;
          }
          const double cand = V[i] + (double)arc_dist;
          if (argmin_pair_beats(cand, i, best_val, best_i))
          {
            best_val = cand;
            best_i = i;
          }
        }
      }

      warp_reduce_argmin_double_int(best_val, best_i);
      if (lane == 0)
      {
        V[j] = best_val;
        pred[j] = (best_i >= 0) ? best_i : (j - 1);
        if (use_shared == 0)
        {
          __threadfence_block();
        }
      }
      __syncwarp();
    }
  }
  else
  {
    if (lane == 0)
    {
      int route_start = 0;
      for (int j = 1; j <= n_cust; ++j)
      {
        if (j == route_start + 1)
        {
          pred[j] = route_start;
        }
        else
        {
          const long long new_load = W[j] - W[route_start];
          const long long new_open = (long long)d0[route_start] + (P[j] - P[route_start + 1]);
          const long long ret_edge =
              closed
                  ? (long long)gpu_edge_distance(
                        g_tour[j - 1], 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                  : 0LL;
          const long long new_arc = new_open + ret_edge;
          const bool cap_ok = (!has_cap) || (new_load <= capacity);
          const bool dist_ok = (!has_dist) || ((double)new_arc <= max_route_distance);
          if (!cap_ok || !dist_ok)
          {
            route_start = j - 1;
          }
          pred[j] = route_start;
        }
      }
      if (use_shared == 0)
      {
        __threadfence_block();
      }
    }
    __syncwarp();
  }

  // Phase 3: Extract materialized route boundaries into dead `V` buffer (cuts[0..eff_m]).
  int *cuts = reinterpret_cast<int *>(V);
  int eff_m = 0;
  if (lane == 0)
  {
    int m = 0;
    int cur = n_cust;
    while (cur > 0)
    {
      cur = pred[cur];
      m += 1;
    }
    eff_m = (m <= buffer_k) ? m : buffer_k;
    int rem_merges = m - eff_m;

    cuts[eff_m] = n_cust;
    int cut_pos = eff_m - 1;
    cur = n_cust;
    while (cur > 0)
    {
      const int prev = pred[cur];
      if (rem_merges > 0)
      {
        rem_merges -= 1;
      }
      else
      {
        cuts[cut_pos--] = prev;
      }
      cur = prev;
    }
    if (use_shared == 0)
    {
      __threadfence_block();
    }
  }
  eff_m = __shfl_sync(0xFFFFFFFFu, eff_m, 0);
  __syncwarp();

  // Phase 4: Cooperative 32-lane route scoring and reduction.
  long long local_unpen = 0LL;
  long long local_cap_viol = 0LL;
  double local_dist_viol = 0.0;

  for (int r = lane; r < eff_m; r += 32)
  {
    const int a = cuts[r];
    const int b = cuts[r + 1];
    const long long r_load = W[b] - W[a];
    const long long r_ret =
        closed
            ? (long long)gpu_edge_distance(
                  g_tour[b - 1], 0, distances, coords, dist_stride, edge_type_code, use_matrix)
            : 0LL;
    const long long r_dist = (long long)d0[a] + (P[b] - P[a + 1]) + r_ret;
    local_unpen += r_dist;
    if (has_cap && r_load > capacity)
    {
      local_cap_viol += (r_load - capacity);
    }
    if (has_dist && (double)r_dist > max_route_distance)
    {
      local_dist_viol += ((double)r_dist - max_route_distance);
    }
  }

  const long long total_unpen = warp_reduce_sum<long long>(local_unpen);
  const long long total_cap_viol = warp_reduce_sum<long long>(local_cap_viol);
  const double total_dist_viol = warp_reduce_sum<double>(local_dist_viol);

  if (lane == 0)
  {
    const long long veh_viol =
        (has_max_vehicles != 0 && eff_m > max_vehicles_constraint)
            ? (long long)(eff_m - max_vehicles_constraint)
            : 0LL;
    const double total_viol = (double)total_cap_viol + total_dist_viol + (double)veh_viol;
    const double cap_pen = (total_cap_viol > 0LL) ? (omega_q * (double)total_cap_viol) : 0.0;
    const double dist_pen = (total_dist_viol > 0.0) ? (omega_l * total_dist_viol) : 0.0;
    const double veh_pen = (veh_viol > 0LL) ? (omega_k * (double)veh_viol) : 0.0;

    penalized_out[ind_idx] = (double)total_unpen + cap_pen + dist_pen + veh_pen;
    unpenalized_out[ind_idx] = (double)total_unpen;
    violations_out[ind_idx] = total_viol;
    num_vehicles_out[ind_idx] = eff_m;
  }

  // Phase 5: Cooperative 32-lane route write-back and trailing zero padding.
  if (routes_out != nullptr)
  {
    int *row_out = routes_out + (size_t)ind_idx * (size_t)row_width;
    for (int r = 0; r < eff_m; ++r)
    {
      const int a = cuts[r];
      const int b = cuts[r + 1];
      if (lane == 0)
      {
        row_out[a + r] = 0;
      }
      for (int k = a + lane; k < b; k += 32)
      {
        row_out[k + r + 1] = g_tour[k];
      }
    }
    for (int idx = n_cust + eff_m + lane; idx < row_width; idx += 32)
    {
      row_out[idx] = 0;
    }
  }
}

extern "C" __global__ void strip_depot_delimiters_kernel(
    const int *__restrict__ delimited_routes, // [batch_size, width]
    int *__restrict__ giant_tours_out,        // [batch_size, n_cust]
    const int batch_size,
    const int width,
    const int n_cust)
{
  const int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind_idx >= batch_size)
  {
    return;
  }
  const int *row_in = delimited_routes + (size_t)ind_idx * (size_t)width;
  int *row_out = giant_tours_out + (size_t)ind_idx * (size_t)n_cust;
  int pos = 0;
  for (int c = 0; c < width && pos < n_cust; ++c)
  {
    const int gene = row_in[c];
    if (gene != 0)
    {
      row_out[pos++] = gene;
    }
  }
}
