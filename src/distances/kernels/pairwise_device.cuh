/**
 * TSPLIB95 pairwise distance primitives for CUDA device code (`DP10-19`).
 *
 * Mirrors, bit for bit, the scalar arithmetic of `src/distances/pairwise.py`:
 *
 *   - `nint(x) = (int)floor(x + 0.5)` (TSPLIB95 C `nint`), never `round()`;
 *   - `CEIL_2D` uses `ceil(sqrt(...))`, not `nint`;
 *   - `ATT` keeps the pseudo-Euclidean conditional `+1`;
 *   - `GEO` truncates the degree part (`(int)x`), not `nint`, because rounding
 *     there changes canonical lengths (burma14 d(1,3) = 510, not 560).
 *
 * The dispatcher `gpu_pairwise_distance` is the single entry point kernels call:
 * it reads the dense matrix when `dist != nullptr` and computes on the fly from
 * `coords` otherwise. `edge_type_code` values come from
 * `distances.pairwise.EDGE_TYPE_TO_CODE` and are verified against this enum by
 * `tests/distances/test_unit_gpu_pairwise_device.py`.
 *
 * Coordinates are `double` (TSPLIB files carry real-valued coordinates); the
 * distance matrix is `int32`, matching `compute_distance_matrix`.
 *
 * Usage in a `.cu` kernel:
 *
 *   #include "pairwise_device.cuh"
 *   int d = gpu_pairwise_distance(u, v, dist, coords, n, GPU_EDGE_EUC_2D);
 */

#ifndef GPU_PAIRWISE_DEVICE_CUH
#define GPU_PAIRWISE_DEVICE_CUH

// No <math.h>: NVRTC (CuPy's JIT) ships no system headers but predeclares the
// device math functions (floor, sqrt, fabs, ceil, cos, acos) this header uses.

//: Integer codes shared with Python (`distances.pairwise.EDGE_TYPE_TO_CODE`).
enum GpuEdgeTypeCode {
  GPU_EDGE_EUC_2D = 0,
  GPU_EDGE_EUC_3D = 1,
  GPU_EDGE_MAN_2D = 2,
  GPU_EDGE_MAN_3D = 3,
  GPU_EDGE_MAX_2D = 4,
  GPU_EDGE_MAX_3D = 5,
  GPU_EDGE_CEIL_2D = 6,
  GPU_EDGE_ATT = 7,
  GPU_EDGE_GEO = 8
};

//: TSPLIB95 approximation of pi for DDD.MM coordinate conversion.
#define GPU_PI 3.141592

//: Earth radius in km used by the TSPLIB95 geographical distance.
#define GPU_EARTH_RADIUS_KM 6378.388

/** TSPLIB95 `nint`: round half up, i.e. `(int)floor(x + 0.5)`. */
__device__ __forceinline__ int gpu_nint(double x) {
  return (int)floor(x + 0.5);
}

/** `EUC_2D`: `nint(sqrt(dx^2 + dy^2))`. */
__device__ __forceinline__ int gpu_euc_2d(double xd, double yd) {
  return gpu_nint(sqrt(xd * xd + yd * yd));
}

/** `EUC_3D`: `nint(sqrt(dx^2 + dy^2 + dz^2))`. */
__device__ __forceinline__ int gpu_euc_3d(double xd, double yd, double zd) {
  return gpu_nint(sqrt(xd * xd + yd * yd + zd * zd));
}

/** `MAN_2D`: `nint(|dx| + |dy|)`. */
__device__ __forceinline__ int gpu_man_2d(double xd, double yd) {
  return gpu_nint(fabs(xd) + fabs(yd));
}

/** `MAN_3D`: `nint(|dx| + |dy| + |dz|)`. */
__device__ __forceinline__ int gpu_man_3d(double xd, double yd, double zd) {
  return gpu_nint(fabs(xd) + fabs(yd) + fabs(zd));
}

/** `MAX_2D`: `max(nint(|dx|), nint(|dy|))`. */
__device__ __forceinline__ int gpu_max_2d(double xd, double yd) {
  int dx = gpu_nint(fabs(xd));
  int dy = gpu_nint(fabs(yd));
  return dx > dy ? dx : dy;
}

/** `MAX_3D`: `max(nint(|dx|), nint(|dy|), nint(|dz|))`. */
__device__ __forceinline__ int gpu_max_3d(double xd, double yd, double zd) {
  int dx = gpu_nint(fabs(xd));
  int dy = gpu_nint(fabs(yd));
  int dz = gpu_nint(fabs(zd));
  int best = dx > dy ? dx : dy;
  return best > dz ? best : dz;
}

/** `CEIL_2D`: `ceil(sqrt(dx^2 + dy^2))` (no `nint`). */
__device__ __forceinline__ int gpu_ceil_2d(double xd, double yd) {
  return (int)ceil(sqrt(xd * xd + yd * yd));
}

/** `ATT`: `tij = nint(rij)` with `rij = sqrt((dx^2 + dy^2) / 10)`, plus the conditional `+1`. */
__device__ __forceinline__ int gpu_att(double xd, double yd) {
  double rij = sqrt((xd * xd + yd * yd) / 10.0);
  int tij = gpu_nint(rij);
  return tij < rij ? tij + 1 : tij;
}

/** TSPLIB95 DDD.MM -> radians, truncating the degree part (matches `pairwise._to_radians_geo`). */
__device__ __forceinline__ double gpu_geo_radians(double coord_val) {
  int deg = (int)coord_val;
  double minutes = coord_val - (double)deg;
  return GPU_PI * (deg + 5.0 * minutes / 3.0) / 180.0;
}

/** `GEO`: spherical law of cosines, `int(R * acos(arg) + 1.0)`. */
__device__ __forceinline__ int gpu_geo(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
  double lat1 = gpu_geo_radians(lat1_deg);
  double lon1 = gpu_geo_radians(lon1_deg);
  double lat2 = gpu_geo_radians(lat2_deg);
  double lon2 = gpu_geo_radians(lon2_deg);

  double q1 = cos(lon1 - lon2);
  double q2 = cos(lat1 - lat2);
  double q3 = cos(lat1 + lat2);
  double acos_arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3);
  acos_arg = acos_arg < -1.0 ? -1.0 : (acos_arg > 1.0 ? 1.0 : acos_arg);
  return (int)(GPU_EARTH_RADIUS_KM * acos(acos_arg) + 1.0);
}

/**
 * TSPLIB95 pairwise distance dispatcher.
 *
 * @param u,v            Node ids in `[0, n)`; `u == v` always returns 0.
 * @param dist           Dense `int32` `(n, n)` matrix, or `nullptr` for the
 *                       on-the-fly coordinate path.
 * @param coords         Row-major coordinates, `2 * n` (2D) or `3 * n` (3D)
 *                       doubles. Ignored when `dist != nullptr`.
 * @param n              Number of nodes (matrix row stride and coordinate count).
 * @param edge_type_code One of `GpuEdgeTypeCode`.
 *
 * @return Integer TSPLIB95 distance. An unknown code returns 0; the Python
 *         mapping is closed over the enum, so that branch is unreachable in
 *         supported pipelines.
 */
__device__ __forceinline__ int gpu_pairwise_distance(
    int u,
    int v,
    const int *__restrict__ dist,
    const double *__restrict__ coords,
    int n,
    int edge_type_code) {
  if (u == v) {
    return 0;
  }
  if (dist != nullptr) {
    return dist[(size_t)u * (size_t)n + (size_t)v];
  }

  switch (edge_type_code) {
    case GPU_EDGE_EUC_2D: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_euc_2d(cu[0] - cv[0], cu[1] - cv[1]);
    }
    case GPU_EDGE_EUC_3D: {
      const double *cu = coords + (size_t)u * 3;
      const double *cv = coords + (size_t)v * 3;
      return gpu_euc_3d(cu[0] - cv[0], cu[1] - cv[1], cu[2] - cv[2]);
    }
    case GPU_EDGE_MAN_2D: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_man_2d(cu[0] - cv[0], cu[1] - cv[1]);
    }
    case GPU_EDGE_MAN_3D: {
      const double *cu = coords + (size_t)u * 3;
      const double *cv = coords + (size_t)v * 3;
      return gpu_man_3d(cu[0] - cv[0], cu[1] - cv[1], cu[2] - cv[2]);
    }
    case GPU_EDGE_MAX_2D: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_max_2d(cu[0] - cv[0], cu[1] - cv[1]);
    }
    case GPU_EDGE_MAX_3D: {
      const double *cu = coords + (size_t)u * 3;
      const double *cv = coords + (size_t)v * 3;
      return gpu_max_3d(cu[0] - cv[0], cu[1] - cv[1], cu[2] - cv[2]);
    }
    case GPU_EDGE_CEIL_2D: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_ceil_2d(cu[0] - cv[0], cu[1] - cv[1]);
    }
    case GPU_EDGE_ATT: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_att(cu[0] - cv[0], cu[1] - cv[1]);
    }
    case GPU_EDGE_GEO: {
      const double *cu = coords + (size_t)u * 2;
      const double *cv = coords + (size_t)v * 2;
      return gpu_geo(cu[0], cu[1], cv[0], cv[1]);
    }
    default:
      return 0;
  }
}

#endif  // GPU_PAIRWISE_DEVICE_CUH
