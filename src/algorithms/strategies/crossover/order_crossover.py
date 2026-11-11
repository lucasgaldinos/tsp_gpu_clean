"""
Order Crossover (OX) strategies for routing permutations.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Order Crossover (OX).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from protocols.algorithm_strategies import CrossoverStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  resolve_route_bounds,
  restore_routes_shape,
  sample_distinct_pairs,
  to_cpu_batch,
)

from ..base import StrategyBase

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


class CrossoverOrder(StrategyBase, CrossoverStrategy):
  """
  Order Crossover (OX) strategy for TSP permutations and depot-bounded tours.

  Preserves a contiguous sub-segment from `parent1` and fills the remaining
  positions in the relative order they appear in `parent2`.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize OX crossover with optional RNG seed."""
    super().__init__(seed)

  @staticmethod
  def _apply_ox(
    p1: np.ndarray,
    p2: np.ndarray,
    c1: int,
    c2: int,
    is_closed: bool,
  ) -> np.ndarray:
    """Apply Order Crossover to a single pair of parent routes with fixed cut points."""
    n = len(p1)
    child = np.full(n, -1, dtype=np.int32)
    stream_start = 1 if is_closed else 0
    stream_len = (n - 2) if is_closed else n
    c2_rel = (c2 - 1) if is_closed else c2

    if is_closed:
      child[0] = p1[0]
      child[-1] = p1[-1]

    child[c1 : c2 + 1] = p1[c1 : c2 + 1]
    in_seg = set(p1[c1 : c2 + 1].tolist())

    child_pos = (c2_rel + 1) % stream_len
    for i in range(stream_len):
      p2_pos = stream_start + ((c2_rel + 1 + i) % stream_len)
      gene = int(p2[p2_pos])
      if gene not in in_seg:
        child[stream_start + child_pos] = gene
        child_pos = (child_pos + 1) % stream_len

    return child

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

    cuts = sample_distinct_pairs(rng, 1, 0, size)[0]
    start, end = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))
    return self._apply_ox(p1, p2, start, end, is_closed=False)

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
    n = len(p1)
    if n <= 1:
      return p1.copy()

    rng: Generator = resolve_rng(context, self.rng)
    low, high = resolve_route_bounds(n, context=context, route_sample=p1)
    if high - low < 2:
      return p1.copy()
    cuts = sample_distinct_pairs(rng, 1, low, high)[0]
    c1, c2 = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))
    return self._apply_ox(p1, p2, c1, c2, is_closed=(low, high) != (0, n))

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """
    Recombine a batch of routes `(N, n_cities)` into `(N, n_cities)` offspring routes.

    Pairs parents in disjoint pairs (2k, 2k+1) sharing identical cut points (OX-A).
    """
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    if was_1d or len(batch_np) < 2:
      return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

    num_routes: int = len(batch_np)
    n_cities: int = batch_np.shape[1]
    low, high = resolve_route_bounds(n_cities, context=context, route_sample=batch_np[0])
    if n_cities <= 1 or high - low < 2:
      return restore_routes_shape(batch_np.copy(), was_1d, to_gpu=was_gpu)
    is_closed: bool = (low, high) != (0, n_cities)
    offspring: np.ndarray = np.empty_like(batch_np, dtype=np.int32)

    rng: Generator = resolve_rng(context, self.rng)
    num_pairs: int = num_routes // 2
    pairs = sample_distinct_pairs(rng, num_pairs, low, high)

    for k in range(num_pairs):
      i = 2 * k
      c1 = int(min(pairs[k, 0], pairs[k, 1]))
      c2 = int(max(pairs[k, 0], pairs[k, 1]))
      offspring[i] = self._apply_ox(batch_np[i], batch_np[i + 1], c1, c2, is_closed)
      offspring[i + 1] = self._apply_ox(batch_np[i + 1], batch_np[i], c1, c2, is_closed)

    if num_routes % 2 == 1:
      offspring[-1] = batch_np[-1]

    return restore_routes_shape(offspring, was_1d, to_gpu=was_gpu)


class CrossoverGPUOrder(StrategyBase, CrossoverStrategy):
  """
  Modular GPU CUDA kernel Order Crossover strategy (`order_crossover.cu`).

  Recombines an entire batch of parent routes `(N, n_cities)` in parallel on GPU
  using 1 warp per route and Format A shared-memory direct-address membership (DP11-11, OX-A).
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize GPU Order Crossover strategy."""
    super().__init__(seed)
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `order_crossover_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "order_crossover.cu",
        "order_crossover_kernel",
        __file__,
      )
    return self._kernel

  def warmup(self, context: ProblemContext) -> None:
    """Compile the kernel and launch it once on two valid parent permutations.

    `crossover_batch` needs at least two routes to launch, so the warm-up batch is
    `(2, n)`; both rows are `arange`, never `cp.zeros` (`BUG-24`).
    """
    super().warmup(context)
    self.crossover_batch(self._warmup_batch(context, rows=2), context=context)

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D parent routes."""
    cpu_ox = CrossoverOrder(seed=self.seed)
    return cpu_ox.crossover(parent1, parent2, context=context)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """
    Recombine a 2D batch of routes `(N, n_cities)` in parallel on GPU.

    Raises:
        GPUUnavailableError: When CuPy is missing or no CUDA device is reachable.
    """
    cp = require_gpu()

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

    low, high = resolve_route_bounds(n_cities, context=context, route_sample=batch[0])
    is_closed = (low, high) != (0, n_cities)

    if n_cities <= 1 or high - low < 2:
      return restore_routes_shape(d_parents.copy() if input_on_gpu else batch.copy(), was_1d)

    rng: Generator = resolve_rng(context, self.rng)

    num_pairs: int = num_routes // 2
    pairs = sample_distinct_pairs(rng, num_pairs, low, high)
    ordered_low = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int32, copy=False)
    ordered_high = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int32, copy=False)
    cut1_np = np.zeros(num_routes, dtype=np.int32)
    cut2_np = np.zeros(num_routes, dtype=np.int32)
    cut1_np[: 2 * num_pairs] = np.repeat(ordered_low, 2)
    cut2_np[: 2 * num_pairs] = np.repeat(ordered_high, 2)

    d_cut1 = cp.asarray(cut1_np, dtype=cp.int32)
    d_cut2 = cp.asarray(cut2_np, dtype=cp.int32)
    self.h2d_bytes += int(d_cut1.nbytes + d_cut2.nbytes)

    d_offspring = cp.empty((num_routes, n_cities), dtype=cp.int32)

    threads_per_block = 256
    dev_shared_mem = getattr(cp.cuda.Device(), "attributes", {}).get(
      "MaxSharedMemoryPerBlock", 48 * 1024
    )
    while threads_per_block > 32 and (threads_per_block // 32) * n_cities > dev_shared_mem:
      threads_per_block //= 2
    warps_per_block = threads_per_block // 32
    blocks = (num_routes + warps_per_block - 1) // warps_per_block
    shared_mem_bytes = warps_per_block * n_cities * 1  # Format A: 1 byte per city per warp

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
        np.int32(1 if is_closed else 0),
      ),
      shared_mem=shared_mem_bytes,
    )
    self.kernel_launches += 1

    if input_on_gpu:
      return restore_routes_shape(d_offspring, was_1d)
    self.d2h_bytes += int(d_offspring.nbytes)
    return restore_routes_shape(cp.asnumpy(d_offspring), was_1d)


__all__: list[str] = ["CrossoverGPUOrder", "CrossoverOrder"]
