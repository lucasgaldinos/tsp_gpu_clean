/**
 * Deterministic SplitMix64 Fisher-Yates permutation initialization kernel (`Task 17.31`).
 *
 * One thread per population row; bit-identical to `InitializationCPURandom`.
 */

#include "splitmix64.cuh"

extern "C" __global__ void random_init_kernel(
    const int *__restrict__ customers,            // [n_cities]
    const unsigned long long *__restrict__ seeds, // [population_size]
    int *__restrict__ population,                 // [population_size, n_cities]
    const int population_size,
    const int n_cities)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= population_size)
  {
    return;
  }

  int *out_row = population + (size_t)row * (size_t)n_cities;
  for (int i = 0; i < n_cities; ++i)
  {
    out_row[i] = customers[i];
  }

  unsigned long long state = seeds[row];
  for (int i = n_cities - 1; i > 0; --i)
  {
    const int j = splitmix64_bounded(state, i + 1);
    const int tmp = out_row[i];
    out_row[i] = out_row[j];
    out_row[j] = tmp;
  }
}
