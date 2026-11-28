"""Experiment-local GPU Partially Mapped Crossover (PMX) strategy.

Implements `CrossoverGPUPermPartiallyMapped` outside `src/` using the modular
`CrossoverStrategy` protocol and a 1-warp-per-route CUDA kernel stored in
`kernels/pmx_crossover.cu`, mirroring `algorithms.strategies.crossover.CrossoverGPUPermOrder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.crossover.pmx_crossover import CrossoverCPUPermPartiallyMapped
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
)

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


class CrossoverGPUPermPartiallyMapped(StrategyBase, CrossoverStrategy):
  """Modular GPU CUDA kernel Partially Mapped Crossover (PMX) strategy.

  Recombines a 2D batch of parent routes `(N, n_cities)` in parallel on GPU
  using 1 warp per route and warp-private shared memory (`inv_p2` + `in_seg`),
  preserving CuPy device residency for zero-copy full-GPU GA pipelines.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize GPU PMX crossover strategy with an optional RNG seed."""
    super().__init__(seed)
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `pmx_crossover_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "pmx_crossover.cu",
        "pmx_crossover_kernel",
        __file__,
      )
    return self._kernel

  def warmup(self, context: ProblemContext) -> None:
    """Compile the kernel and launch it once on two valid parent permutations."""
    super().warmup(context)
    self.crossover_batch(self._warmup_batch(context, rows=2), context=context)

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D parent routes via CPU PMX."""
    cpu_pmx = CrossoverCPUPermPartiallyMapped(seed=self.seed)
    return cpu_pmx.crossover(parent1, parent2, context=context)

  @time_strategy
  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Recombine a 2D batch of routes `(N, n_cities)` in parallel on GPU.

    Pairs parents in disjoint pairs `(2k, 2k+1)` sharing cut points `(c1, c2)`.
    When `routes` is already a CuPy device array, returns a device array without
    a device-to-host copy.
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

    dev_shared_mem = int(getattr(cp.cuda.Device(), "attributes", {}).get("MaxSharedMemoryPerBlock", 48 * 1024))
    bytes_per_warp = 5 * n_cities  # 4 bytes/city for inv_p2 + 1 byte/city for in_seg
    if bytes_per_warp > dev_shared_mem:
      raise ValueError(
        f"CrossoverGPUPermPartiallyMapped shared-memory requirement ({bytes_per_warp} B/warp for n={n_cities}) "
        f"exceeds device MaxSharedMemoryPerBlock ({dev_shared_mem} B)."
      )

    threads_per_block = 256
    while threads_per_block > 32 and (threads_per_block // 32) * bytes_per_warp > dev_shared_mem:
      threads_per_block //= 2
    warps_per_block = threads_per_block // 32
    blocks = (num_routes + warps_per_block - 1) // warps_per_block
    shared_mem_bytes = warps_per_block * bytes_per_warp

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


__all__: list[str] = ["CrossoverGPUPermPartiallyMapped"]
