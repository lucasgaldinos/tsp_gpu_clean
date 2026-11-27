/**
 * Shared CUDA warp and block reduction primitives (`Task 17.21`, `DP17-7`).
 *
 * NVRTC-safe header (no system includes) providing:
 *   - `warp_reduce_sum<T>(val, mask, active_width)`: shuffle-down sum over an active warp.
 *   - `block_reduce_sum<T>(val, s_partials)`: two-level warp-shuffle + shared-memory sum
 *     valid for any `blockDim.x` in `[1, 1024]` (requires at least `ceil(blockDim.x / 32)`
 *     elements in `s_partials` when `blockDim.x > 32`).
 *   - `warp_reduce_argmin_double_int(val, idx, mask)`: butterfly `__shfl_xor_sync` argmin
 *     with deterministic tie-breaking favoring the smallest non-negative `idx`.
 *   - `block_reduce_argmin_double_int(val, idx, s_vals, s_idxs)`: two-level block argmin
 *     preserving deterministic smallest-index tie-breaking.
 */

#ifndef GPU_BLOCK_REDUCTIONS_CUH
#define GPU_BLOCK_REDUCTIONS_CUH

/**
 * Reduce `val` across active lanes in one warp using `__shfl_down_sync`.
 *
 * Result is valid on lane 0 of the warp. Guarding `(lane + offset) < active_width`
 * prevents self-doubling when a trailing partial warp has fewer than 32 active threads.
 */
template <typename T>
static __device__ __forceinline__ T warp_reduce_sum(
    T val,
    unsigned int mask = 0xFFFFFFFFu,
    int active_width = 32) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const T other = __shfl_down_sync(mask, val, offset);
    if (lane + offset < active_width) {
      val += other;
    }
  }
  return val;
}

/**
 * Reduce `val` across all threads in a 1D block; result is valid on `threadIdx.x == 0`.
 *
 * Caller must supply `s_partials` with at least `(blockDim.x + 31) / 32` slots
 * (up to 32 slots for a 1024-thread block).
 */
template <typename T>
static __device__ __forceinline__ T block_reduce_sum(T val, T *s_partials) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  const int warp_start = warp_id << 5;
  const int warp_rem = blockDim.x - warp_start;
  const int warp_active = (warp_rem < 32) ? warp_rem : 32;
  const unsigned int warp_mask =
      (warp_active == 32) ? 0xFFFFFFFFu : ((1u << warp_active) - 1u);

  val = warp_reduce_sum<T>(val, warp_mask, warp_active);
  if (num_warps == 1) {
    return val;
  }

  if (lane == 0) {
    s_partials[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    const T warp_val = (lane < num_warps) ? s_partials[lane] : (T)0;
    const unsigned int mask0 =
        (num_warps == 32) ? 0xFFFFFFFFu : ((1u << num_warps) - 1u);
    if (lane < num_warps) {
      val = warp_reduce_sum<T>(warp_val, mask0, num_warps);
    }
  }
  return val;
}

/** Strict total-order comparator favoring smaller `val`, then smaller non-negative `idx`. */
static __device__ __forceinline__ bool argmin_pair_beats(
    double cand_val,
    int cand_idx,
    double cur_val,
    int cur_idx) {
  return (cand_val < cur_val) ||
         (cand_val == cur_val && cand_idx >= 0 && (cur_idx < 0 || cand_idx < cur_idx));
}

/**
 * Full-warp butterfly argmin over `(double val, int idx)` via `__shfl_xor_sync`.
 *
 * After reduction, all 32 lanes hold the exact minimum `(val, idx)` with deterministic
 * smallest-index tie-breaking matching CPU `np.argmin`.
 */
static __device__ __forceinline__ void warp_reduce_argmin_double_int(
    double &val,
    int &idx,
    unsigned int mask = 0xFFFFFFFFu) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const double other_val = __shfl_xor_sync(mask, val, offset);
    const int other_idx = __shfl_xor_sync(mask, idx, offset);
    if (argmin_pair_beats(other_val, other_idx, val, idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

/**
 * Block-wide argmin over `(double val, int idx)` across a 1D block.
 *
 * Requires `s_vals` and `s_idxs` with at least `(blockDim.x + 31) / 32` slots.
 * Result is valid on `threadIdx.x == 0`.
 */
static __device__ __forceinline__ void block_reduce_argmin_double_int(
    double &val,
    int &idx,
    double *s_vals,
    int *s_idxs) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  const int warp_start = warp_id << 5;
  const int warp_rem = blockDim.x - warp_start;
  const int warp_active = (warp_rem < 32) ? warp_rem : 32;
  const unsigned int warp_mask =
      (warp_active == 32) ? 0xFFFFFFFFu : ((1u << warp_active) - 1u);

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const double other_val = __shfl_down_sync(warp_mask, val, offset);
    const int other_idx = __shfl_down_sync(warp_mask, idx, offset);
    if (lane + offset < warp_active && argmin_pair_beats(other_val, other_idx, val, idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
  if (num_warps == 1) {
    return;
  }

  if (lane == 0) {
    s_vals[warp_id] = val;
    s_idxs[warp_id] = idx;
  }
  __syncthreads();

  if (warp_id == 0 && lane < num_warps) {
    val = s_vals[lane];
    idx = s_idxs[lane];
    const unsigned int mask0 =
        (num_warps == 32) ? 0xFFFFFFFFu : ((1u << num_warps) - 1u);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const double other_val = __shfl_down_sync(mask0, val, offset);
      const int other_idx = __shfl_down_sync(mask0, idx, offset);
      if (lane + offset < num_warps && argmin_pair_beats(other_val, other_idx, val, idx)) {
        val = other_val;
        idx = other_idx;
      }
    }
  }
}

#endif  // GPU_BLOCK_REDUCTIONS_CUH
