/**
 * Candidate-guided nearest-neighbour (GRASP RCL) initialization kernel (`Task 17.31`).
 *
 * One thread per population row using deterministic `SplitMix64` and `O(1)` unvisited
 * swap-and-pop tracking; bit-identical to `InitializationCPUTourNearestNeighbor`.
 */

#include "splitmix64.cuh"

extern "C" __global__ void nearest_neighbor_init_kernel(
    const int *__restrict__ customers,            // [n_cities]
    const int *__restrict__ candidates,           // [cand_rows, k_eff]
    const unsigned long long *__restrict__ seeds, // [population_size]
    int *__restrict__ ws_unvisited,               // [population_size, n_cities]
    int *__restrict__ ws_pos,                     // [population_size, max_node + 1]
    int *__restrict__ ws_cand_order,              // [population_size, k_eff]
    int *__restrict__ population,                 // [population_size, n_cities]
    const int population_size,
    const int n_cities,
    const int max_node,
    const int cand_rows,
    const int k_eff)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= population_size || n_cities <= 0)
  {
    return;
  }

  int *unvisited = ws_unvisited + (size_t)row * (size_t)n_cities;
  int *pos_map = ws_pos + (size_t)row * (size_t)(max_node + 1);
  int *cand_order = ws_cand_order + (size_t)row * (size_t)k_eff;
  int *out_row = population + (size_t)row * (size_t)n_cities;

  for (int i = 0; i <= max_node; ++i)
  {
    pos_map[i] = -1;
  }
  for (int i = 0; i < n_cities; ++i)
  {
    const int c = customers[i];
    unvisited[i] = c;
    pos_map[c] = i;
  }

  unsigned long long state = seeds[row];
  int rem_count = n_cities;

  const int start_idx = splitmix64_bounded(state, rem_count);
  int current = unvisited[start_idx];
  out_row[0] = current;

  // Swap-and-pop `current` from `unvisited[0..rem_count-1]`.
  {
    const int last_idx = rem_count - 1;
    const int last_node = unvisited[last_idx];
    unvisited[start_idx] = last_node;
    unvisited[last_idx] = current;
    pos_map[last_node] = start_idx;
    pos_map[current] = last_idx;
    rem_count = last_idx;
  }

  for (int step = 1; step < n_cities; ++step)
  {
    int chosen = -1;
    if (current >= 0 && current < cand_rows && k_eff > 0)
    {
      for (int i = 0; i < k_eff; ++i)
      {
        cand_order[i] = i;
      }
      const int *cand_row = candidates + (size_t)current * (size_t)k_eff;
      for (int i = 0; i < k_eff; ++i)
      {
        const int j = i + splitmix64_bounded(state, k_eff - i);
        const int tmp = cand_order[i];
        cand_order[i] = cand_order[j];
        cand_order[j] = tmp;

        const int nb = cand_row[cand_order[i]];
        if (nb >= 0 && nb <= max_node)
        {
          const int p = pos_map[nb];
          if (p >= 0 && p < rem_count)
          {
            chosen = nb;
            break;
          }
        }
      }
    }

    if (chosen < 0)
    {
      const int u_idx = splitmix64_bounded(state, rem_count);
      chosen = unvisited[u_idx];
    }

    out_row[step] = chosen;
    const int p = pos_map[chosen];
    const int last_idx = rem_count - 1;
    const int last_node = unvisited[last_idx];
    unvisited[p] = last_node;
    unvisited[last_idx] = chosen;
    pos_map[last_node] = p;
    pos_map[chosen] = last_idx;
    rem_count = last_idx;
    current = chosen;
  }
}
