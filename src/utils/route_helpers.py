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

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, Routes


def is_gpu_array(arr: Any) -> bool:
  """Return True if `arr` is a CuPy device array (`cp.ndarray`)."""
  cp = cupy_backend()
  return bool(cp is not None and isinstance(arr, cp.ndarray))


def resolve_route_bounds(
  n: int,
  context: ProblemContext | None = None,
  route_sample: Any | None = None,
) -> tuple[int, int]:
  """Return the mutable index slice `[low, high)` for a route of length `n`.

  Depot-bounded routes `[0, ..., 0]` must keep both endpoints, while open TSP
  permutations are free to mutate every position (`DP11-19`, `BUG-28`). The
  topology is read from the two endpoints when the sample lives on the host
  (`O(1)`, no device transfer) and from `context.is_closed` when it lives on the
  device, so no generation pays a D2H sync or trusts a per-instance cache that a
  zero-filled warm-up batch can poison (`BUG-24`).

  Args:
      n: Route length.
      context: Optional `ProblemContext`; its `is_closed` decides device samples.
      route_sample: Optional route used to detect depot-bounded endpoints.

  Returns:
      `(0, n)` for open permutations or `(1, n - 1)` for depot-bounded routes;
      with no information at all the conservative depot-preserving slice wins.
  """
  if route_sample is not None and n > 2 and not is_gpu_array(route_sample):
    sample = np.asarray(route_sample)
    if sample.size == n and int(sample[0]) == int(sample[-1]):
      return 1, n - 1
  if context is not None:
    return (1, n - 1) if bool(getattr(context, "is_closed", False)) else (0, n)
  if route_sample is not None:
    if is_gpu_array(route_sample):
      cp = require_gpu()
      closed = n > 2 and int(cp.asnumpy(route_sample[0])) == int(cp.asnumpy(route_sample[-1]))
      return (1, n - 1) if closed else (0, n)
    return 0, n
  return 1, n - 1


def sample_distinct_pairs(
  rng: Generator,
  num_pairs: int,
  low: int,
  high: int,
) -> np.ndarray:
  """Draw `num_pairs` ordered pairs of distinct integers from `[low, high)`.

  Uses the rejection-free bijection `v = v' + [v' >= u]`: `u` is uniform on
  `[low, high)` and `v'` uniform on `[low, high - 1)`, so `v` lands uniformly on
  `[low, high) \\ {u}`. That replaces `choice(..., replace=False)` with two flat
  integer draws, cutting the host sampling loop from ~1.22 ms to ~0.02 ms for a
  256-route batch (`DP11-20`, `OX-B`) while keeping CPU and GPU on one stream.

  Args:
      rng: Generator shared by the CPU and GPU callers.
      num_pairs: Number of pairs to draw.
      low: First admissible integer (inclusive).
      high: Upper bound (exclusive).

  Returns:
      `int32` array of shape `(num_pairs, 2)`; columns are unordered, so callers
      that need cuts order the pair themselves.

  Raises:
      ValueError: When fewer than two integers exist in `[low, high)`.
  """
  if high - low < 2:
    raise ValueError(f"Need at least two distinct integers in [{low}, {high})")
  if num_pairs <= 0:
    return np.empty((0, 2), dtype=np.int32)
  first = rng.integers(low, high, size=num_pairs, dtype=np.int32)
  shifted = rng.integers(low, high - 1, size=num_pairs, dtype=np.int32)
  second = shifted + (shifted >= first).astype(np.int32)
  return np.stack((first, second), axis=1).astype(np.int32, copy=False)


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
  if pool.size == 0:
    if low >= high:
      raise ValueError(f"Invalid mutable range [low, high): [{low}, {high}) is empty or degenerate")
    if low <= anchor_idx < high:
      return anchor_idx
    return low
  return int(rng.choice(pool))


def sample_mutation_positions(
  rng: Generator,
  rows: Any,
  low: int,
  high: int,
  *,
  candidate_k: int | None = None,
  context: ProblemContext | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Draw the two mutable positions of every row in a route batch.

  Both the CPU and the GPU mutation strategies call this single host oracle, so
  they issue identical draws and produce bit-identical offspring (`SWAP-A`,
  `INV-A`). With `candidate_k` the second position is the tour position of a
  candidate neighbour of the first city (`DP10-20`); otherwise the pair comes from
  the rejection-free bijection of `sample_distinct_pairs` (`DP11-20`).

  Args:
      rng: Host generator shared with the sibling CPU/GPU strategy.
      rows: 2D batch `(N, n_cities)` on host or device.
      low: First mutable index (inclusive).
      high: Last mutable index (exclusive).
      candidate_k: Optional `k`-NN neighbourhood size for the second position.
      context: Optional `ProblemContext` providing the candidate list.

  Returns:
      `(pos1, pos2, d2h_bytes)`: `int32` arrays of length `N` plus the bytes the
      candidate mode had to copy back from the device (zero otherwise).
  """
  num_rows = int(rows.shape[0])
  if candidate_k is None or candidate_k <= 0 or context is None:
    pairs = sample_distinct_pairs(rng, num_rows, low, high)
    return pairs[:, 0].copy(), pairs[:, 1].copy(), 0
  pos1 = np.zeros(num_rows, dtype=np.int32)
  pos2 = np.zeros(num_rows, dtype=np.int32)
  d2h_bytes = 0
  device_rows = is_gpu_array(rows)
  cp = require_gpu() if device_rows else None
  candidates = context.get_cpu_candidate_list(candidate_k)
  for i in range(num_rows):
    if cp is not None:
      row = cp.asnumpy(rows[i])
      d2h_bytes += int(row.nbytes)
    else:
      row = np.asarray(rows[i], dtype=np.int32)
    first = int(rng.integers(low, high))
    pos1[i] = first
    pos2[i] = select_proximity_index(row, first, low, high, candidates, rng)
  return pos1, pos2, d2h_bytes


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
  "ensure_routes_batch",
  "inverse_positions",
  "is_gpu_array",
  "resolve_route_bounds",
  "restore_routes_shape",
  "sample_distinct_pairs",
  "sample_mutation_positions",
  "select_proximity_index",
  "to_cpu_batch",
]
