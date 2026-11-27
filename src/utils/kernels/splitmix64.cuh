/**
 * Deterministic 64-bit SplitMix64 PRNG primitives for CUDA kernels (`Task 17.31`).
 *
 * Matches `src/utils/splitmix64.py` bit-for-bit using unsigned 64-bit wraparound
 * arithmetic and Lemire-style high-32 multiply bounded sampling.
 */

#ifndef GPU_SPLITMIX64_CUH
#define GPU_SPLITMIX64_CUH

/** Advance SplitMix64 state and return the mixed 64-bit output. */
__device__ __forceinline__ unsigned long long splitmix64_next(unsigned long long &state)
{
  state += 0x9E3779B97F4A7C15ULL;
  unsigned long long z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/** Return a deterministic integer in `[0, bound)` for `bound > 0` using high-32 multiply. */
__device__ __forceinline__ int splitmix64_bounded(unsigned long long &state, int bound)
{
  const unsigned long long z = splitmix64_next(state);
  return (int)(((z >> 32) * (unsigned long long)(unsigned int)bound) >> 32);
}

#endif // GPU_SPLITMIX64_CUH
