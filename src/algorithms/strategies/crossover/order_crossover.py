"""
Order Crossover (OX) strategies for routing permutations.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Order Crossover (OX).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import make_rng, resolve_rng
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  to_cpu_batch,
)

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


def _has_depot_endpoints(route: np.ndarray) -> bool:
  """Check if a 1D route has matching start/end depot nodes `[depot, ..., depot]`."""
  return len(route) > 2 and int(route[0]) == int(route[-1])


class OrderCrossover:
  """
  Order Crossover (OX) strategy for TSP permutations and depot-bounded tours.

  Preserves a contiguous sub-segment from `parent1` and fills the remaining
  positions in the relative order they appear in `parent2`.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize OX crossover with optional RNG seed."""
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

  def _ox_core(
    self,
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
  ) -> np.ndarray:
    """Perform OX on two 1D pure permutation arrays of length `n`."""
    size: int = len(p1)
    if size <= 1:
      return p1.copy()

    cuts = rng.choice(size, size=2, replace=False)
    start, end = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))

    child: np.ndarray = np.full(size, -1, dtype=np.int32)
    child[start : end + 1] = p1[start : end + 1]
    in_segment = set(p1[start : end + 1].tolist())

    child_idx: int = (end + 1) % size
    for i in range(size):
      p2_idx: int = (end + 1 + i) % size
      gene = int(p2[p2_idx])
      if gene not in in_segment:
        child[child_idx] = gene
        child_idx = (child_idx + 1) % size

    return child

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """
    Recombine two 1D parent routes into a single 1D child route via Order Crossover.
    """
    p1 = np.asarray(parent1, dtype=np.int32)
    p2 = np.asarray(parent2, dtype=np.int32)

    rng: Generator = resolve_rng(context, self.rng)

    if _has_depot_endpoints(p1) and _has_depot_endpoints(p2):
      depot = int(p1[0])
      inner_child = self._ox_core(p1[1:-1], p2[1:-1], rng)
      return np.concatenate(([depot], inner_child, [depot])).astype(np.int32)

    return self._ox_core(p1, p2, rng)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """
    Recombine a batch of routes `(N, n_cities)` into `(N, n_cities)` offspring routes.

    If a single 1D route `(n_cities,)` is passed, returns a copy of that 1D route.
    """
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    if was_1d or len(batch_np) < 2:
      return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

    num_routes: int = len(batch_np)
    offspring: np.ndarray = np.empty_like(batch_np, dtype=np.int32)

    for i in range(0, num_routes, 2):
      p1 = batch_np[i]
      p2 = batch_np[(i + 1) % num_routes]
      offspring[i] = self.crossover(p1, p2, context=context)
      if i + 1 < num_routes:
        offspring[i + 1] = self.crossover(p2, p1, context=context)

    return restore_routes_shape(offspring, was_1d, to_gpu=was_gpu)


class GPUOrderCrossover:
  """
  Modular GPU CUDA kernel Order Crossover strategy (`order_crossover.cu`).

  Recombines an entire batch of parent routes `(N, n_cities)` in parallel on GPU.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize GPU Order Crossover strategy."""
    self.seed: SeedLike = seed
    self.rng: Generator = make_rng(seed)
    self._kernel: Any = None
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `order_crossover_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "order_crossover.cu",
        "order_crossover_kernel",
        __file__,
      )
    return self._kernel

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D parent routes."""
    cpu_ox = OrderCrossover(seed=self.seed)
    return cpu_ox.crossover(parent1, parent2, context=context)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """
    Recombine a 2D batch of routes `(N, n_cities)` in parallel on GPU.
    """
    if not CUPY_AVAILABLE or cp is None:
      cpu_ox = OrderCrossover(seed=self.seed)
      return cpu_ox.crossover_batch(routes, context=context)

    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n_cities = batch.shape
    if was_1d or num_routes < 2:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    if input_on_gpu:
      d_parents = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      d_parents = cp.asarray(batch, dtype=cp.int32)
      self.h2d_bytes += int(d_parents.nbytes)

    rng: Generator = resolve_rng(context, self.rng)

    cut1_np = rng.integers(0, n_cities, size=num_routes, dtype=np.int32)
    cut2_np = rng.integers(0, n_cities, size=num_routes, dtype=np.int32)
    d_cut1 = cp.asarray(cut1_np, dtype=cp.int32)
    d_cut2 = cp.asarray(cut2_np, dtype=cp.int32)
    self.h2d_bytes += int(d_cut1.nbytes + d_cut2.nbytes)

    d_offspring = cp.empty((num_routes, n_cities), dtype=cp.int32)

    threads_per_block = 128
    blocks = (num_routes + threads_per_block - 1) // threads_per_block
    kernel = self._get_kernel()
    kernel(
      (blocks,),
      (threads_per_block,),
      (
        d_parents,
        d_cut1,
        d_cut2,
        d_offspring,
        np.int32(num_routes),
        np.int32(n_cities),
      ),
    )
    cp.cuda.Stream.null.synchronize()
    self.kernel_launches += 1

    if input_on_gpu:
      return restore_routes_shape(d_offspring, was_1d)
    self.d2h_bytes += int(d_offspring.nbytes)
    return restore_routes_shape(cp.asnumpy(d_offspring), was_1d)


__all__: list[str] = ["GPUOrderCrossover", "OrderCrossover"]

