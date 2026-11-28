"""Route-Based Crossover (RBX, Potvin & Bengio 1996) for `0`-delimited VRP batches (`DP17-2`, `DP17-7`, `BUG-114`).

Separates CPU (`CrossoverCPURouteRBX`) and GPU (`CrossoverGPURouteRBX`) strategies, honoring capacity ($Q$),
max route distance ($L_{\\max}$), and `return_to_depot` (OVRP vs. CVRP/DCVRP) when packing
Parent-2 residual customers, using identical `int32` RNG draws across single/batch calls and
reusing a monotonic `uint8` `d_visited` GPU workspace (`BUG-113`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.gpu_base import DistanceOperandCache, ensure_gpu_workspace
from protocols.algorithm_strategies import CrossoverStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape, to_cpu_batch

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike

_BLOCK_SIZE: int = 128


def _rbx_pair_cpu(
  p1: np.ndarray,
  p2: np.ndarray,
  demands: np.ndarray,
  distances: np.ndarray | None,
  *,
  dimension: int,
  capacity: int | None,
  max_route_distance: float | None,
  return_to_depot: bool,
  route_pick: int,
) -> np.ndarray:
  """Apply Route-Based Crossover (RBX) to two 1D `0`-delimited `int32` parent routes."""
  width = int(p1.size)
  child = np.zeros(width, dtype=np.int32)
  visited = np.zeros(dimension, dtype=bool)
  check_dist = max_route_distance is not None and distances is not None

  p1_routes: list[list[int]] = []
  current: list[int] = []
  for node in p1.astype(np.int64, copy=False):
    val = int(node)
    if val > 0:
      current.append(val)
    elif current:
      p1_routes.append(current)
      current = []
  if current:
    p1_routes.append(current)

  write_pos = 0
  copied_count = 0
  used_routes = 0
  k_max = max(1, width - dimension)
  p1_load = 0
  p1_dist = 0
  p1_last = 0

  if p1_routes:
    chosen = p1_routes[route_pick % len(p1_routes)]
    if write_pos < width:
      child[write_pos] = 0
      write_pos += 1
    for node in chosen:
      if 0 < node < dimension and not visited[node]:
        if write_pos < width:
          child[write_pos] = node
          write_pos += 1
        visited[node] = True
        copied_count += 1
        p1_load += int(demands[node])
        if check_dist and distances is not None:
          p1_dist += int(distances[p1_last, node])
        p1_last = node
    if copied_count > 0:
      used_routes = 1
      if write_pos < width:
        child[write_pos] = 0
        write_pos += 1
    else:
      write_pos = 0

  rem_unvisited = (dimension - 1) - copied_count
  cur_open = False
  cur_load = 0
  cur_dist = 0
  last_node = 0

  for raw_node in p2.astype(np.int64, copy=False):
    if rem_unvisited <= 0:
      break
    node = int(raw_node)
    if node <= 0 or node >= dimension or visited[node]:
      continue

    q = int(demands[node])
    if not cur_open:
      if used_routes == 0:
        if write_pos + 1 + rem_unvisited <= width:
          child[write_pos] = 0
          write_pos += 1
        used_routes = 1
        cur_open = True
        cur_load = q
        cur_dist = int(distances[0, node]) if check_dist and distances is not None else 0
        last_node = node
      elif used_routes < k_max and (write_pos + rem_unvisited < width):
        used_routes += 1
        cur_open = True
        cur_load = q
        cur_dist = int(distances[0, node]) if check_dist and distances is not None else 0
        last_node = node
      else:
        if write_pos > 1 and child[write_pos - 1] == 0:
          write_pos -= 1
        cur_open = True
        cur_load = p1_load + q
        cur_dist = (p1_dist + int(distances[p1_last, node])) if check_dist and distances is not None else 0
        last_node = node
    else:
      step_dist = int(distances[last_node, node]) if check_dist and distances is not None else 0
      ret_dist = int(distances[node, 0]) if (check_dist and return_to_depot and distances is not None) else 0
      cap_viol = capacity is not None and (cur_load + q > capacity)
      dist_viol = (
        check_dist and max_route_distance is not None and float(cur_dist + step_dist + ret_dist) > max_route_distance
      )
      if (cap_viol or dist_viol) and (used_routes < k_max) and (write_pos + 1 + rem_unvisited < width):
        child[write_pos] = 0
        write_pos += 1
        used_routes += 1
        cur_load = q
        cur_dist = int(distances[0, node]) if check_dist and distances is not None else 0
        last_node = node
      else:
        cur_load += q
        cur_dist += step_dist
        last_node = node

    if write_pos < width:
      child[write_pos] = node
      write_pos += 1
    visited[node] = True
    rem_unvisited -= 1

  return child


class CrossoverCPURouteRBX(StrategyBase, CrossoverStrategy):
  """CPU Route-Based Crossover (`RBX`) for `0`-delimited `(N, n + K_max)` VRP batches."""

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize CPU Route-Based Crossover with optional RNG seed."""
    super().__init__(seed)

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D `0`-delimited parent routes into a single child route on CPU."""
    p1_np = np.asarray(require_gpu().asnumpy(parent1) if is_gpu_array(parent1) else parent1, dtype=np.int32)
    p2_np = np.asarray(require_gpu().asnumpy(parent2) if is_gpu_array(parent2) else parent2, dtype=np.int32)
    dimension = context.dimension if context is not None else int(max(int(p1_np.max()), int(p2_np.max()))) + 1
    demands = context.get_cpu_demands() if context is not None else None
    if demands is None:
      demands = np.zeros(dimension, dtype=np.int32)
    max_dist = context.constraints.max_route_distance if context is not None else None
    distances = context.get_cpu_distances() if (context is not None and max_dist is not None) else None
    capacity = context.constraints.capacity if context is not None else None
    return_to_depot = context.constraints.return_to_depot if context is not None else True
    rng: Generator = resolve_rng(context, self.rng)
    pick = int(rng.integers(0, 1_000_000, dtype=np.int32))
    return _rbx_pair_cpu(
      p1_np,
      p2_np,
      demands,
      distances,
      dimension=dimension,
      capacity=capacity,
      max_route_distance=max_dist,
      return_to_depot=return_to_depot,
      route_pick=pick,
    )

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Recombine a `(N, W)` `0`-delimited batch on CPU."""
    batch_np, was_1d, _ = to_cpu_batch(routes, copy=False)
    batch_size = int(batch_np.shape[0])
    rng: Generator = resolve_rng(context, self.rng)
    if batch_size <= 1:
      return restore_routes_shape(batch_np.copy(), was_1d)

    route_picks = np.asarray(rng.integers(0, 1_000_000, size=batch_size, dtype=np.int32), dtype=np.int32)
    dimension = context.dimension if context is not None else int(batch_np.max()) + 1
    demands = context.get_cpu_demands() if context is not None else None
    if demands is None:
      demands = np.zeros(dimension, dtype=np.int32)
    max_dist = context.constraints.max_route_distance if context is not None else None
    distances = context.get_cpu_distances() if (context is not None and max_dist is not None) else None
    capacity = context.constraints.capacity if context is not None else None
    return_to_depot = context.constraints.return_to_depot if context is not None else True

    offspring = np.empty_like(batch_np)
    for idx in range(batch_size):
      p2_idx = idx + 1 if idx + 1 < batch_size else 0
      offspring[idx] = _rbx_pair_cpu(
        batch_np[idx],
        batch_np[p2_idx],
        demands,
        distances,
        dimension=dimension,
        capacity=capacity,
        max_route_distance=max_dist,
        return_to_depot=return_to_depot,
        route_pick=int(route_picks[idx]),
      )
    return restore_routes_shape(offspring, was_1d)


class CrossoverGPURouteRBX(StrategyBase, CrossoverStrategy):
  """GPU CUDA Route-Based Crossover (`RBX`) for `0`-delimited `(N, n + K_max)` VRP batches."""

  def __init__(
    self,
    return_gpu: bool = False,
    max_matrix_dimension: int | None = None,
    seed: SeedLike = None,
  ) -> None:
    """Initialize GPU Route-Based Crossover."""
    super().__init__(seed)
    self.return_gpu: bool = return_gpu
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self._kernel: Any = None
    self._operands: DistanceOperandCache = DistanceOperandCache()
    self._ws_visited: Any = None

  def reset_metrics(self) -> None:
    """Reset counters, mirrored distance operands, and cached GPU visited buffer."""
    super().reset_metrics()
    self._operands.reset()
    self._ws_visited = None

  def _get_kernel(self) -> Any:
    """Lazily compile and cache `rbx_crossover_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel("rbx_crossover.cu", "rbx_crossover_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch `rbx_crossover_kernel` once on a valid `(2, n + 1)` GPU batch."""
    require_gpu()
    super()._warmup_impl(context)
    cp = require_gpu()
    row = np.zeros(context.dimension + 1, dtype=np.int32)
    row[1 : context.dimension] = np.arange(1, context.dimension, dtype=np.int32)
    batch = cp.asarray(np.vstack([row, row]), dtype=cp.int32)
    self.crossover_batch(batch, context=context)

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D `0`-delimited parent routes (requires GPU backend per `DP9-2`)."""
    require_gpu()
    return CrossoverCPURouteRBX(seed=self.seed).crossover(parent1, parent2, context=context)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Recombine a `(N, W)` `0`-delimited batch on CUDA with cached `d_visited` workspace."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    batch_size, width = batch.shape
    input_on_gpu = is_gpu_array(batch)
    keep_on_gpu = input_on_gpu or self.return_gpu
    rng: Generator = resolve_rng(context, self.rng)

    if batch_size <= 1:
      if keep_on_gpu:
        d_copy = cp.asarray(batch, dtype=cp.int32).copy()
        if not input_on_gpu:
          self.h2d_bytes += int(d_copy.nbytes)
        return restore_routes_shape(d_copy, was_1d)
      return restore_routes_shape(np.asarray(batch, dtype=np.int32).copy(), was_1d)

    if context is None:
      raise ValueError("CrossoverGPURouteRBX requires a ProblemContext for GPU batches")

    if input_on_gpu:
      d_parents = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      parents_np = np.ascontiguousarray(batch, dtype=np.int32)
      d_parents = cp.asarray(parents_np, dtype=cp.int32)
      self.h2d_bytes += int(d_parents.nbytes)

    d_offspring = cp.empty_like(d_parents)
    route_picks = np.asarray(rng.integers(0, 1_000_000, size=batch_size, dtype=np.int32), dtype=np.int32)
    d_picks = cp.asarray(route_picks, dtype=cp.int32)
    self.h2d_bytes += int(route_picks.nbytes)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    constraints = context.constraints
    if constraints.max_route_distance is not None:
      (d_distances, d_coords, edge_type_code), op_bytes = self._operands.resolve(
        context, context.dimension, self.max_matrix_dimension
      )
      self.h2d_bytes += op_bytes
    else:
      d_distances, d_coords, edge_type_code = 0, 0, 0

    self._ws_visited = ensure_gpu_workspace(self._ws_visited, (batch_size, context.dimension), cp.uint8)
    d_visited = self._ws_visited[:batch_size, : context.dimension]

    blocks = (batch_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    kernel = self._get_kernel()
    kernel(
      (blocks,),
      (_BLOCK_SIZE,),
      (
        d_parents,
        d_offspring,
        d_demands,
        d_distances,
        d_coords,
        d_picks,
        d_visited,
        np.int32(batch_size),
        np.int32(width),
        np.int32(context.dimension),
        np.int32(edge_type_code),
        np.int32(1 if constraints.return_to_depot else 0),
        np.int32(1 if constraints.capacity is not None else 0),
        np.int64(constraints.capacity if constraints.capacity is not None else 0),
        np.int32(1 if constraints.max_route_distance is not None else 0),
        np.float64(constraints.max_route_distance if constraints.max_route_distance is not None else 0.0),
      ),
    )
    self.kernel_launches += 1

    if keep_on_gpu:
      return restore_routes_shape(d_offspring, was_1d)
    host_out = d_offspring.get()
    self.d2h_bytes += int(host_out.nbytes)
    return restore_routes_shape(host_out, was_1d)


__all__: list[str] = ["CrossoverCPURouteRBX", "CrossoverGPURouteRBX"]
