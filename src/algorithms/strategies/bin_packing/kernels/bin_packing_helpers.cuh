/**
 * Device helper primitives for VRAM-native FFD/BFD bin-packing (`Task 17.30`, `BUG-117`).
 *
 * Included by `kernels/bin_packing.cu` (`-I<kernel_dir>` via `utils/gpu_helpers.py`).
 */

#ifndef GPU_BIN_PACKING_HELPERS_CUH
#define GPU_BIN_PACKING_HELPERS_CUH

#include "block_reductions.cuh"
#include "pairwise_device.cuh"

__device__ __forceinline__ bool bp_route_beats(
    int a,
    int b,
    const long long *route_load,
    int has_capacity,
    long long capacity)
{
  const bool inf_a = (has_capacity != 0) && (route_load[a] > capacity);
  const bool inf_b = (has_capacity != 0) && (route_load[b] > capacity);
  if (inf_a != inf_b)
  {
    return !inf_a;
  }
  if (route_load[a] != route_load[b])
  {
    return route_load[a] > route_load[b];
  }
  return a < b;
}

__device__ __forceinline__ void bp_sift_down_routes(
    int *order,
    int start,
    int count,
    const long long *route_load,
    int has_capacity,
    long long capacity)
{
  int root = start;
  while (2 * root + 1 < count)
  {
    int child = 2 * root + 1;
    if (child + 1 < count &&
        bp_route_beats(order[child], order[child + 1], route_load, has_capacity, capacity))
    {
      child += 1;
    }
    if (bp_route_beats(order[root], order[child], route_load, has_capacity, capacity))
    {
      const int tmp = order[root];
      order[root] = order[child];
      order[child] = tmp;
      root = child;
    }
    else
    {
      break;
    }
  }
}

__device__ __forceinline__ void bp_heapsort_routes(
    int *order,
    int count,
    const long long *route_load,
    int has_capacity,
    long long capacity)
{
  for (int start = (count - 2) / 2; start >= 0; --start)
  {
    bp_sift_down_routes(order, start, count, route_load, has_capacity, capacity);
  }
  for (int end = count - 1; end > 0; --end)
  {
    const int tmp = order[0];
    order[0] = order[end];
    order[end] = tmp;
    bp_sift_down_routes(order, 0, end, route_load, has_capacity, capacity);
  }
}

__device__ __forceinline__ bool bp_eject_beats(int u, int v, const int *__restrict__ demands)
{
  const long long qu = (long long)demands[u];
  const long long qv = (long long)demands[v];
  if (qu != qv)
  {
    return qu > qv;
  }
  return u < v;
}

__device__ __forceinline__ void bp_sift_down_ejected(
    int *arr,
    int start,
    int count,
    const int *__restrict__ demands)
{
  int root = start;
  while (2 * root + 1 < count)
  {
    int child = 2 * root + 1;
    if (child + 1 < count && bp_eject_beats(arr[child], arr[child + 1], demands))
    {
      child += 1;
    }
    if (bp_eject_beats(arr[root], arr[child], demands))
    {
      const int tmp = arr[root];
      arr[root] = arr[child];
      arr[child] = tmp;
      root = child;
    }
    else
    {
      break;
    }
  }
}

__device__ __forceinline__ void bp_heapsort_ejected(
    int *arr,
    int count,
    const int *__restrict__ demands)
{
  for (int start = (count - 2) / 2; start >= 0; --start)
  {
    bp_sift_down_ejected(arr, start, count, demands);
  }
  for (int end = count - 1; end > 0; --end)
  {
    const int tmp = arr[0];
    arr[0] = arr[end];
    arr[end] = tmp;
    bp_sift_down_ejected(arr, 0, end, demands);
  }
}

__device__ __forceinline__ long long bp_warp_tile_seg_load(
    int is_nz,
    int tok,
    const int *__restrict__ demands,
    unsigned int nz_mask,
    unsigned int upto_mask,
    long long carry_load,
    int lane)
{
  long long pref = (is_nz != 0) ? (long long)demands[tok] : 0LL;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
  {
    const long long up_q = __shfl_up_sync(0xFFFFFFFFu, pref, offset);
    if (lane >= offset)
    {
      pref += up_q;
    }
  }
  const unsigned int zeros_upto_me = (~nz_mask) & upto_mask;
  if (zeros_upto_me == 0u)
  {
    return carry_load + pref;
  }
  const int last_zero = 31 - __clz(zeros_upto_me);
  return pref - __shfl_sync(0xFFFFFFFFu, pref, last_zero);
}

__device__ __forceinline__ void bp_best_route_detour(
    const int *flat_nodes,
    int off,
    int rlen,
    int node,
    const int *__restrict__ distances,
    const double *__restrict__ coords,
    int dist_stride,
    int edge_type_code,
    bool closed,
    int has_max_distance,
    double route_dist,
    double max_route_distance,
    int lane,
    double &out_detour,
    int &out_pos)
{
  const double kInf = __longlong_as_double(0x7ff0000000000000ULL);
  double best_detour = kInf;
  int best_pos = -1;

  for (int pos = lane; pos <= rlen; pos += 32)
  {
    const int prev_node = (pos == 0) ? 0 : flat_nodes[off + pos - 1];
    const int next_node = (pos == rlen) ? 0 : flat_nodes[off + pos];
    const double d_uv = (double)gpu_pairwise_distance(
        prev_node, node, distances, coords, dist_stride, edge_type_code);
    double d_un = 0.0;
    double d_pn = 0.0;
    if (pos < rlen || closed)
    {
      d_un = (double)gpu_pairwise_distance(
          node, next_node, distances, coords, dist_stride, edge_type_code);
      if (rlen > 0)
      {
        d_pn = (double)gpu_pairwise_distance(
            prev_node, next_node, distances, coords, dist_stride, edge_type_code);
      }
    }
    const double detour = d_uv + d_un - d_pn;
    if (has_max_distance != 0 && (route_dist + detour > max_route_distance))
    {
      continue;
    }
    if (argmin_pair_beats(detour, pos, best_detour, best_pos))
    {
      best_detour = detour;
      best_pos = pos;
    }
  }
  warp_reduce_argmin_double_int(best_detour, best_pos);
  out_detour = best_detour;
  out_pos = best_pos;
}

__device__ __forceinline__ void bp_shift_insert_and_update(
    int *flat_nodes,
    int *route_offset,
    int *route_len,
    long long *route_load,
    double *route_dist,
    int num_kept,
    int &total_kept,
    int best_r,
    int best_pos,
    int u,
    long long q_u,
    double best_detour,
    int lane)
{
  const int ins_idx = route_offset[best_r] + best_pos;
  for (int tile_end = total_kept; tile_end > ins_idx; tile_end -= 32)
  {
    const int src = tile_end - 1 - lane;
    const int val = (src >= ins_idx) ? flat_nodes[src] : 0;
    __syncwarp();
    if (src >= ins_idx)
    {
      flat_nodes[src + 1] = val;
    }
    __syncwarp();
  }
  if (lane == 0)
  {
    flat_nodes[ins_idx] = u;
    route_len[best_r] += 1;
    route_load[best_r] += q_u;
    route_dist[best_r] += best_detour;
  }
  for (int r = best_r + 1 + lane; r < num_kept; r += 32)
  {
    route_offset[r] += 1;
  }
  total_kept += 1;
  __syncwarp();
}

#endif // GPU_BIN_PACKING_HELPERS_CUH
