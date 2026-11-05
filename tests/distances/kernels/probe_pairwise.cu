/**
 * Test-only probe kernel for `gpu_pairwise_distance` (`Task 10.15`).
 *
 * Evaluates every ordered `(u, v)` pair through the shared device dispatcher,
 * either reading the dense matrix (`use_matrix != 0`, `dist != nullptr`) or
 * computing on the fly from coordinates (`use_matrix == 0`, `dist == nullptr`).
 *
 * The `#include "pairwise_device.cuh"` works because `utils.gpu_helpers` adds
 * `-I<src/distances/kernels>` to every JIT compilation.
 */

#include "pairwise_device.cuh"

extern "C" __global__
void probe_pairwise_kernel(
    const int *__restrict__ dist,
    const double *__restrict__ coords,
    int n,
    int edge_type_code,
    int use_matrix,
    int *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * n;
  if (idx >= total) {
    return;
  }
  int u = idx / n;
  int v = idx - u * n;
  const int *dist_ptr = use_matrix ? dist : nullptr;
  out[idx] = gpu_pairwise_distance(u, v, dist_ptr, coords, n, edge_type_code);
}
