"""
Partially Mapped Crossover (PMX) strategy for routing permutations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils.phase_timer import time_strategy
from utils.rng import make_rng, resolve_rng
from utils.route_helpers import restore_routes_shape, to_cpu_batch

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


def _has_depot_endpoints(route: np.ndarray) -> bool:
  """Check if a 1D route has matching start/end depot nodes `[depot, ..., depot]`."""
  return len(route) > 2 and int(route[0]) == int(route[-1])


class PartiallyMappedCrossover:
  """
  Partially Mapped Crossover (PMX) strategy for TSP permutations and depot-bounded tours.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize PMX crossover with optional RNG seed."""
    self.seed: SeedLike = seed
    self.rng: Generator = make_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _pmx_core(
    self,
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
  ) -> np.ndarray:
    """Perform PMX on two 1D pure permutation arrays of length `n`."""
    size: int = len(p1)
    if size <= 1:
      return p1.copy()

    cuts = rng.choice(size, size=2, replace=False)
    start, end = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))

    child: np.ndarray = np.full(size, -1, dtype=np.int32)
    child[start : end + 1] = p1[start : end + 1]

    p2_pos: dict[int, int] = {int(val): idx for idx, val in enumerate(p2)}
    in_segment = set(p1[start : end + 1].tolist())

    for idx in range(start, end + 1):
      val = int(p2[idx])
      if val not in in_segment:
        pos: int = idx
        while start <= pos <= end:
          mapped_val = int(p1[pos])
          pos = p2_pos[mapped_val]
        child[pos] = val

    for idx in range(size):
      if child[idx] == -1:
        child[idx] = p2[idx]

    return child

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D parent routes into a single 1D child route via PMX."""
    p1 = np.asarray(parent1, dtype=np.int32)
    p2 = np.asarray(parent2, dtype=np.int32)

    rng: Generator = resolve_rng(context, self.rng)

    if _has_depot_endpoints(p1) and _has_depot_endpoints(p2):
      depot = int(p1[0])
      inner_child = self._pmx_core(p1[1:-1], p2[1:-1], rng)
      return np.concatenate(([depot], inner_child, [depot])).astype(np.int32)

    return self._pmx_core(p1, p2, rng)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Recombine a 2D batch of routes `(N, n_cities)` via PMX."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    if was_1d or len(batch_np) < 2:
      return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

    num_routes = len(batch_np)
    offspring = np.empty_like(batch_np, dtype=np.int32)

    for i in range(0, num_routes, 2):
      p1 = batch_np[i]
      p2 = batch_np[(i + 1) % num_routes]
      offspring[i] = self.crossover(p1, p2, context=context)
      if i + 1 < num_routes:
        offspring[i + 1] = self.crossover(p2, p1, context=context)

    return restore_routes_shape(offspring, was_1d, to_gpu=was_gpu)


__all__: list[str] = ["PartiallyMappedCrossover"]
