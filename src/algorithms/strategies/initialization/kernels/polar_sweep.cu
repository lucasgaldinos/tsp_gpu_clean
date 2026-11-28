/**
 * Polar Sweep (Gillett & Miller, 1974) initialization kernel (`Task 17.31`).
 *
 * One thread per population row; reads pre-ordered `sorted_customers`, draws
 * a cyclic `SplitMix64` offset for rows `1..N-1` (`offset = 0` for row `0`),
 * and greedily packs routes honoring `capacity`, `max_route_distance`, and
 * `return_to_depot` via `gpu_edge_distance`. Bit-identical to `InitializationCPURouteSweep`.
 */

#include "pairwise_device.cuh"
#include "splitmix64.cuh"

extern "C" __global__ void polar_sweep_kernel(
    const int *__restrict__ sorted_customers,     // [n_cust]
    const int *__restrict__ demands,              // [dist_stride]
    const int *__restrict__ distances,            // [dist_stride, dist_stride] or nullptr
    const double *__restrict__ coords,            // [dist_stride, 2|3] or nullptr
    const unsigned long long *__restrict__ seeds, // [population_size]
    int *__restrict__ population,                 // [population_size, row_width]
    const int population_size,
    const int n_cust,
    const int row_width,
    const int dist_stride,
    const int edge_type_code,
    const int k_max,
    const int is_delimited,
    const int return_to_depot,
    const int has_capacity,
    const long long capacity,
    const int has_max_distance,
    const double max_route_distance)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= population_size)
  {
    return;
  }

  int *out_row = population + (size_t)row * (size_t)row_width;
  if (is_delimited != 0)
  {
    for (int i = 0; i < row_width; ++i)
    {
      out_row[i] = 0;
    }
  }
  if (n_cust <= 0)
  {
    return;
  }

  unsigned long long state = seeds[row];
  const int offset = (row == 0 || n_cust <= 1) ? 0 : splitmix64_bounded(state, n_cust);
  const bool use_matrix = distances != nullptr;
  const bool closed = return_to_depot != 0;

  if (is_delimited == 0)
  {
    for (int i = 0; i < n_cust; ++i)
    {
      const int idx = (offset + i) % n_cust;
      out_row[i] = sorted_customers[idx];
    }
    return;
  }

  int last_cust = -1;
  long long cur_load = 0LL;
  long long cur_open_dist = 0LL;
  int routes_opened = 0;
  int write_pos = 0;

  for (int i = 0; i < n_cust; ++i)
  {
    const int cust = sorted_customers[(offset + i) % n_cust];
    const long long q_cust = (long long)demands[cust];
    if (last_cust < 0)
    {
      out_row[0] = 0;
      out_row[1] = cust;
      write_pos = 2;
      routes_opened = 1;
      last_cust = cust;
      cur_load = q_cust;
      cur_open_dist = (long long)gpu_edge_distance(
          0, cust, distances, coords, dist_stride, edge_type_code, use_matrix);
      continue;
    }

    const long long new_load = cur_load + q_cust;
    const long long new_open = cur_open_dist + (long long)gpu_edge_distance(
                                                   last_cust, cust, distances, coords, dist_stride, edge_type_code, use_matrix);
    const long long ret_edge = closed
                                   ? (long long)gpu_edge_distance(
                                         cust, 0, distances, coords, dist_stride, edge_type_code, use_matrix)
                                   : 0LL;
    const bool cap_ok = (has_capacity == 0) || (new_load <= capacity);
    const bool dist_ok = (has_max_distance == 0) || ((double)(new_open + ret_edge) <= max_route_distance);

    if (cap_ok && dist_ok)
    {
      if (write_pos < row_width)
      {
        out_row[write_pos++] = cust;
      }
      last_cust = cust;
      cur_load = new_load;
      cur_open_dist = new_open;
    }
    else
    {
      if (routes_opened < k_max)
      {
        if (write_pos < row_width)
        {
          out_row[write_pos++] = 0;
        }
        routes_opened += 1;
      }
      if (write_pos < row_width)
      {
        out_row[write_pos++] = cust;
      }
      last_cust = cust;
      cur_load = q_cust;
      cur_open_dist = (long long)gpu_edge_distance(
          0, cust, distances, coords, dist_stride, edge_type_code, use_matrix);
    }
  }
}
