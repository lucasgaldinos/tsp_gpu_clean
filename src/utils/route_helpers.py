"""
Route batch conversion and device residency helpers.

Enables route-transforming modules (crossover, mutation, local search, fitness evaluation)
to transparently handle either single 1D tours `(n_cities,)` or 2D batches `(N, n_cities)`
and normalize CPU/GPU array residency without boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from protocols.types import Route, RouteOrRoutes, Routes


def is_gpu_array(arr: Any) -> bool:
  """Return True if `arr` is a CuPy device array (`cp.ndarray`)."""
  return bool(CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray))


def ensure_routes_batch(routes: RouteOrRoutes) -> tuple[Routes, bool]:
  """
  Normalize a single 1D route `(n_cities,)` or 2D batch `(N, n_cities)` into a 2D batch.

  Enables every route-transforming module (crossover, mutation, local search,
  fitness evaluation) to ingest either a single route or a population batch using
  the exact same parameter contract.

  Args:
      routes: 1D `(n_cities,)` or 2D `(N, n_cities)` NumPy or CuPy array (or sequence).

  Returns:
      Tuple `(routes_2d, was_1d)` where `routes_2d` has shape `(N, n_cities)` and
      `was_1d` indicates whether the input was originally 1D.

  Raises:
      ValueError: If `routes` is not 1D or 2D.
  """
  if not hasattr(routes, "ndim"):
    routes = np.asarray(routes, dtype=np.int32)
  if routes.ndim == 1:
    return routes.reshape(1, -1), True
  if routes.ndim == 2:
    return routes, False
  raise ValueError(f"Expected 1D route or 2D routes batch, got array with ndim={routes.ndim}")


def to_cpu_batch(routes: RouteOrRoutes, *, copy: bool = False) -> tuple[np.ndarray, bool, bool]:
  """
  Normalize `routes` into a 2D CPU `np.ndarray` of `int32` and report its original shape/device.

  Args:
      routes: 1D `(n_cities,)` or 2D `(N, n_cities)` NumPy or CuPy array.
      copy: Whether to force a defensive host memory copy when already on CPU.

  Returns:
      Tuple `(batch_np, was_1d, was_gpu)` where `batch_np` is a 2D host `np.ndarray`
      with `dtype=np.int32`.
  """
  batch, was_1d = ensure_routes_batch(routes)
  was_gpu: bool = is_gpu_array(batch)
  if was_gpu and cp is not None:
    batch_np: np.ndarray = cp.asnumpy(batch).astype(np.int32, copy=False)
  else:
    batch_np = np.asarray(batch, dtype=np.int32)
    if copy:
      batch_np = batch_np.copy()
  return batch_np, was_1d, was_gpu


def restore_routes_shape(
  routes_batch: Routes,
  was_1d: bool,
  *,
  to_gpu: bool = False,
) -> RouteOrRoutes:
  """
  Restore a `(N, n_cities)` batch to its original device residency and 1D/2D shape.

  Args:
      routes_batch: 2D array of shape `(N, n_cities)`.
      was_1d: Whether the original caller passed a 1D route.
      to_gpu: If True and CuPy is available, convert `routes_batch` back to a `cp.ndarray`.

  Returns:
      1D route `(n_cities,)` if `was_1d` is True, otherwise the 2D `routes_batch`.
  """
  out: Routes = routes_batch
  if to_gpu and CUPY_AVAILABLE and cp is not None and not isinstance(out, cp.ndarray):
    out = cast("Routes", cp.asarray(out, dtype=cp.int32))
  if was_1d:
    return cast("Route", out[0])
  return out


__all__: list[str] = [
  "ensure_routes_batch",
  "is_gpu_array",
  "restore_routes_shape",
  "to_cpu_batch",
]
