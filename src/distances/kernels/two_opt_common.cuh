/**
 * Shared device primitives for the batched 2-opt CUDA kernels (`DP10-19`).
 *
 * `two_opt_batch.cu` and `two_opt_multi_move.cu` evaluate the exact same 2-opt
 * neighborhood: they differ only in how many disjoint improving moves they apply per
 * pass (one steepest move versus every non-overlapping move). The arithmetic, the
 * shared-memory tiling rule and the per-thread best-move bookkeeping are therefore one
 * single source of truth, kept here so a fix to the delta cannot diverge between the
 * two kernels.
 *
 * Contract of `TOUR_AT`/`TOUR_SET`: they are macros, so they read the caller's locals.
 * Every including kernel must define, before use:
 *
 *   - `bool tiled`, `int tile_start`, `int tile_end`;
 *   - `int *s_tour` (shared memory: `n` slots when untiled, `tile_size + 1` when tiled);
 *   - `int *tour` (this block's route in global memory).
 *
 * Usage:
 *
 *   #include "two_opt_common.cuh"
 *   extern "C" __global__ void my_kernel(...) { ... TOUR_AT(left) ... }
 */

#ifndef GPU_TWO_OPT_COMMON_CUH
#define GPU_TWO_OPT_COMMON_CUH

#include "pairwise_device.cuh"

//: Tour positions cached per shared-memory tile when `n` exceeds the full-tour budget.
#define TWO_OPT_TILE_SIZE 2048

// Accessor macros used by the tiled scan (`Task 10.18`). When `tiled` is false the
// whole tour lives in `s_tour[n]`; otherwise only `s_tour[TWO_OPT_TILE_SIZE + 1]`
// holds the current window and out-of-window reads fall back to global memory.
#define TOUR_AT(idx) \
  (tiled ? (((idx) >= tile_start && (idx) <= tile_end) ? s_tour[(idx) - tile_start] : tour[(idx)]) : s_tour[(idx)])

#define TOUR_SET(idx, value)                      \
  do                                              \
  {                                               \
    int _pos = (idx);                             \
    int _val = (value);                           \
    if (!tiled)                                   \
    {                                             \
      s_tour[_pos] = _val;                        \
    }                                             \
    else                                          \
    {                                             \
      tour[_pos] = _val;                          \
      if (_pos >= tile_start && _pos <= tile_end) \
      {                                           \
        s_tour[_pos - tile_start] = _val;         \
      }                                           \
    }                                             \
  } while (0)

// Exact 2-opt delta for breaking edges (a, b) and (c, d), where `a`/`b` are
// `tour[i]`/`tour[i + 1]` and `c`/`d` are `tour[j]`/`tour[j + 1]`.
// `row_a`/`row_b` alias `dist` rows in dense-matrix mode; the coordinate mode evaluates
// `gpu_pairwise_distance` on the fly (`DP10-19`).
static __device__ __forceinline__ int two_opt_delta(
    int a, int b, int d_ab, int c, int d,
    const int *row_a, const int *row_b,
    const int *dist, const double *coords, int n, int edge_type_code, bool use_matrix)
{
  if (use_matrix)
  {
    int old_dist = d_ab + dist[(size_t)c * (size_t)n + (size_t)d];
    int new_dist = row_a[c] + row_b[d];
    return new_dist - old_dist;
  }
  int old_dist = d_ab + gpu_pairwise_distance(c, d, nullptr, coords, n, edge_type_code);
  int new_dist = gpu_pairwise_distance(a, c, nullptr, coords, n, edge_type_code) + gpu_pairwise_distance(b, d, nullptr, coords, n, edge_type_code);
  return new_dist - old_dist;
}

// Keep the steepest (most negative) delta of this thread.
static __device__ __forceinline__ void two_opt_consider(
    int i, int j, int delta, int *s_deltas, int *s_swap_i, int *s_swap_j, int tid)
{
  if (delta < s_deltas[tid])
  {
    s_deltas[tid] = delta;
    s_swap_i[tid] = i;
    s_swap_j[tid] = j;
  }
}

#endif
