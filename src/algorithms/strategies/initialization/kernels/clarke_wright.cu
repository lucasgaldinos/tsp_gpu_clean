/**
 * Parallel Clarke & Wright (1964) savings initialization kernel (`Task 17.31`).
 *
 * One thread per population row; row `0` executes deterministic greedy CW (`rcl_width = 1`),
 * while rows `1..N-1` use a register-resident 32-bit sliding-window RCL driven by `SplitMix64`.
 * Bit-identical to `InitializationCPURouteClarkeWright`.
 */

#include "pairwise_device.cuh"
#include "splitmix64.cuh"

__device__ __forceinline__ void cw_reverse_chain(
    int start_node,
    int *prev_node,
    int *next_node) {
  int cur = start_node;
  while (cur != -1) {
    const int nxt = next_node[cur];
    next_node[cur] = prev_node[cur];
    prev_node[cur] = nxt;
    cur = nxt;
  }
}

extern "C" __global__ void clarke_wright_init_kernel(
    const int *__restrict__ customers,            // [n_cust]
    const int *__restrict__ demands,              // [dist_stride]
    const int *__restrict__ distances,            // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords,            // [dist_stride, 2|3] or nullptr
    const int *__restrict__ pairs_u,              // [n_pairs]
    const int *__restrict__ pairs_v,              // [n_pairs]
    const unsigned long long *__restrict__ seeds, // [population_size]
    int *__restrict__ ws_prev,                    // [population_size, dist_stride]
    int *__restrict__ ws_next,                    // [population_size, dist_stride]
    int *__restrict__ ws_other,                   // [population_size, dist_stride]
    long long *__restrict__ ws_load,              // [population_size, dist_stride]
    long long *__restrict__ ws_open_dist,         // [population_size, dist_stride]
    int *__restrict__ population,                 // [population_size, row_width]
    const int population_size,
    const int n_cust,
    const int n_pairs,
    const int row_width,
    const int dist_stride,
    const int edge_type_code,
    const int k_max,
    const int rcl_width_nonzero,
    const int is_symmetric,
    const int is_delimited,
    const int return_to_depot,
    const int has_capacity,
    const long long capacity,
    const int has_max_distance,
    const double max_route_distance) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= population_size) {
    return;
  }

  int *out_row = population + (size_t)row * (size_t)row_width;
  if (is_delimited != 0) {
    for (int i = 0; i < row_width; ++i) {
      out_row[i] = 0;
    }
  }
  if (n_cust <= 0) {
    return;
  }

  int *prev_node = ws_prev + (size_t)row * (size_t)dist_stride;
  int *next_node = ws_next + (size_t)row * (size_t)dist_stride;
  int *other_end = ws_other + (size_t)row * (size_t)dist_stride;
  long long *route_load = ws_load + (size_t)row * (size_t)dist_stride;
  long long *route_open_dist = ws_open_dist + (size_t)row * (size_t)dist_stride;

  for (int i = 0; i < n_cust; ++i) {
    const int c = customers[i];
    prev_node[c] = -1;
    next_node[c] = -1;
    other_end[c] = c;
    route_load[c] = (long long)demands[c];
    route_open_dist[c] = 0LL;
  }

  unsigned long long state = seeds[row];
  const int rcl_width = (row == 0) ? 1 : rcl_width_nonzero;
  const bool use_matrix = distances != nullptr;
  const bool closed = return_to_depot != 0;
  const bool sym = is_symmetric != 0;

  int win_base = 0;
  unsigned long long used_mask = 0ULL;

  while (win_base < n_pairs) {
    while (win_base < n_pairs && (used_mask & 1ULL) != 0ULL) {
      win_base += 1;
      used_mask >>= 1;
    }
    if (win_base >= n_pairs) {
      break;
    }
    int avail = n_pairs - win_base;
    if (avail > rcl_width) {
      avail = rcl_width;
    }

    int k = 0;
    if (rcl_width > 1 && avail > 1) {
      const int r = splitmix64_bounded(state, avail);
      for (int step = 0; step < avail; ++step) {
        const int cand = (r + step) % avail;
        if ((used_mask & (1ULL << cand)) == 0ULL) {
          k = cand;
          break;
        }
      }
    }
    used_mask |= (1ULL << k);

    const int pair_idx = win_base + k;
    const int u = pairs_u[pair_idx];
    const int v = pairs_v[pair_idx];
    if (u == v || other_end[u] == v) {
      continue;
    }

    const bool u_at_end = (next_node[u] == -1);
    const bool u_at_start = (prev_node[u] == -1);
    const bool v_at_start = (prev_node[v] == -1);
    const bool v_at_end = (next_node[v] == -1);

    if (sym) {
      if ((!u_at_end && !u_at_start) || (!v_at_start && !v_at_end)) {
        continue;
      }
    } else {
      if (!u_at_end || !v_at_start) {
        continue;
      }
    }

    const long long new_load = route_load[u] + route_load[v];
    if (has_capacity != 0 && new_load > capacity) {
      continue;
    }

    const int a = other_end[u];
    const int b = other_end[v];
    const long long d_uv = (long long)gpu_edge_distance(
        u, v, distances, coords, dist_stride, edge_type_code, use_matrix);
    const long long new_open = route_open_dist[u] + d_uv + route_open_dist[v];

    if (has_max_distance != 0) {
      const long long d_0a = (long long)gpu_edge_distance(
          0, a, distances, coords, dist_stride, edge_type_code, use_matrix);
      const long long d_b0 = closed
          ? (long long)gpu_edge_distance(
                b, 0, distances, coords, dist_stride, edge_type_code, use_matrix)
          : 0LL;
      if ((double)(d_0a + new_open + d_b0) > max_route_distance) {
        continue;
      }
    }

    if (sym) {
      if (!u_at_end) {
        cw_reverse_chain(u, prev_node, next_node);
      }
      if (!v_at_start) {
        cw_reverse_chain(b, prev_node, next_node);
      }
    }

    next_node[u] = v;
    prev_node[v] = u;
    other_end[a] = b;
    other_end[b] = a;
    route_load[a] = new_load;
    route_load[b] = new_load;
    route_open_dist[a] = new_open;
    route_open_dist[b] = new_open;
  }

  // Materialize routes by scanning `customers[0..n_cust-1]` for route heads (`prev_node[c] == -1`).
  int write_pos = 0;
  int routes_emitted = 0;
  for (int i = 0; i < n_cust; ++i) {
    const int head = customers[i];
    if (prev_node[head] != -1) {
      continue;
    }
    if (is_delimited != 0) {
      if (routes_emitted == 0 || routes_emitted < k_max) {
        if (write_pos < row_width) {
          out_row[write_pos++] = 0;
        }
      }
      routes_emitted += 1;
    }
    int cur = head;
    while (cur != -1) {
      if (write_pos < row_width) {
        out_row[write_pos++] = cur;
      }
      cur = next_node[cur];
    }
  }
}
