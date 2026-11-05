"""
Route batch conversion and device residency helpers.

Enables route-transforming modules (crossover, mutation, local search, fitness evaluation)
to transparently handle either single 1D tours `(n_cities,)` or 2D batches `(N, n_cities)`
and normalize CPU/GPU array residency without boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from protocols.backend import cupy_backend, require_gpu

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.types import Route, RouteOrRoutes, Routes


def is_gpu_array(arr: Any) -> bool:
  """Return True if `arr` is a CuPy device array (`cp.ndarray`)."""
  cp = cupy_backend()
  return bool(cp is not None and isinstance(arr, cp.ndarray))


def _mutation_bounds(n: int) -> tuple[int, int]:
  """Return the mutable index slice `[low, high)` for a route of length `n`.

  Endpoints (index `0` and `n - 1`) are never mutated, so depot delimiters stay
  fixed. For `n <= 3` the slice is empty (`high - low < 2`), making mutation a
  safe no-op when there is no interior position to swap or invert.
  """
  return 1, n - 1


def inverse_positions(route: np.ndarray) -> np.ndarray:
  """Return the inverse map `pos` with `pos[route[idx]] = idx` in `O(n)`.

  The array is sized by the largest node id, not by the route length, because a
  sparse CVRP sub-route such as `[0, 5, 12, 19, 0]` carries ids above
  `len(route) - 1`. Slots for nodes absent from the route stay `-1`, so callers
  can detect "node not in this route" instead of raising `IndexError` (`BUG-19`).

  Args:
      route: 1D route `(n_stops,)` of node ids.

  Returns:
      `int32` array of size `max(route.size, route.max() + 1)` (empty for an
      empty route), filled with `-1` outside the route positions.
  """
  if route.size == 0:
    return np.full(0, -1, dtype=np.int32)
  size = max(int(route.size), int(route.max()) + 1)
  positions = np.full(size, -1, dtype=np.int32)
  positions[route] = np.arange(route.size, dtype=np.int32)
  return positions


def select_proximity_index(
  route: np.ndarray,
  anchor_idx: int,
  low: int,
  high: int,
  candidates: np.ndarray | None,
  rng: Generator,
) -> int:
  """Return the tour position of a candidate neighbour of `route[anchor_idx]`.

  Candidates are visited in a random order so repeated calls explore different
  neighbours. Positions outside the mutable interior `[low, high)` and the anchor
  itself are skipped. When no candidate is present in the sub-route (a sparse
  CVRP segment whose node ids exceed its length, `BUG-19`), a uniform interior
  position is drawn so the caller still performs a real swap.

  Args:
      route: 1D route `(n_stops,)` of node ids.
      anchor_idx: Index of the city whose neighbourhood is consulted.
      low: First mutable index (inclusive).
      high: Last mutable index (exclusive).
      candidates: `(n_nodes, k_eff)` `k`-NN table, or `None` to draw uniformly.
      rng: Random generator used for the candidate order and the fallback draw.

  Returns:
      An index in `[low, high)` different from `anchor_idx` whenever the slice
      holds at least two positions.
  """
  if candidates is not None and candidates.ndim == 2 and candidates.shape[0] > 0:
    node = int(route[anchor_idx])
    if 0 <= node < candidates.shape[0]:
      positions = inverse_positions(route)
      for order in rng.permutation(int(candidates.shape[1])):
        neighbor = int(candidates[node, int(order)])
        if neighbor < 0 or neighbor >= positions.size:
          continue
        index = int(positions[neighbor])
        if index != anchor_idx and low <= index < high:
          return index
  pool = np.array([index for index in range(low, high) if index != anchor_idx], dtype=np.int64)
  return int(rng.choice(pool))


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
  if was_gpu:
    cp = require_gpu()
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
  if to_gpu:
    cp = require_gpu()
    if not isinstance(out, cp.ndarray):
      out = cast("Routes", cp.asarray(out, dtype=cp.int32))
  if was_1d:
    return cast("Route", out[0])
  return out


__all__: list[str] = [
  "_mutation_bounds",
  "ensure_routes_batch",
  "inverse_positions",
  "is_gpu_array",
  "restore_routes_shape",
  "select_proximity_index",
  "to_cpu_batch",
]
