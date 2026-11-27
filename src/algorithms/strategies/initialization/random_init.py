"""Uniform random permutation initialization strategies (`InitializationCPURandom`, `InitializationGPURandom`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from algorithms.strategies.base import StrategyBase
from protocols.algorithm_strategies import InitializationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.splitmix64 import draw_row_seeds, splitmix64_bounded, splitmix64_step

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


class InitializationCPURandom(StrategyBase, InitializationStrategy):
  """Vectorized CPU uniform permutation initialization via ``SplitMix64`` Fisher-Yates."""

  def __init__(self, seed: SeedLike = None) -> None:
    super().__init__(seed)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` uniform random permutations of ``customers`` on CPU."""
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")
    customers_arr = np.asarray(customers, dtype=np.int32)
    n_cities = int(customers_arr.size)
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)
    if population_size == 0 or n_cities == 0:
      return np.empty((population_size, n_cities), dtype=np.int32)

    pop = np.tile(customers_arr, (population_size, 1)).astype(np.int32, copy=False)
    states = row_seeds.copy()
    row_idx = np.arange(population_size, dtype=np.intp)
    for i in range(n_cities - 1, 0, -1):
      states, z = splitmix64_step(states)
      j = splitmix64_bounded(z, i + 1).astype(np.intp, copy=False)
      tmp = pop[:, i].copy()
      pop[:, i] = pop[row_idx, j]
      pop[row_idx, j] = tmp
    return pop


class InitializationGPURandom(StrategyBase, InitializationStrategy):
  """VRAM-native CUDA uniform permutation initialization bit-identical to ``InitializationCPURandom``."""

  def __init__(self, seed: SeedLike = None, return_gpu: bool = True) -> None:
    super().__init__(seed)
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    """Lazily compile and cache ``random_init_kernel``."""
    if self._kernel is None:
      self._kernel = load_kernel("random_init.cu", "random_init_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch ``random_init_kernel`` ahead of measured runs without mutating state."""
    require_gpu()
    super()._warmup_impl(context)
    self._get_kernel()
    n_cities = max(1, context.dimension)
    self.initialize(list(range(n_cities)), 2, context)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` uniform random permutations in VRAM."""
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")
    cp = require_gpu()
    customers_arr = np.ascontiguousarray(customers, dtype=np.int32)
    n_cities = int(customers_arr.size)
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0 or n_cities == 0:
      empty_gpu = cp.empty((population_size, n_cities), dtype=cp.int32)
      return cast("Routes", empty_gpu) if self.return_gpu else np.empty((population_size, n_cities), dtype=np.int32)

    d_customers = cp.asarray(customers_arr, dtype=cp.int32)
    d_seeds = cp.asarray(row_seeds, dtype=cp.uint64)
    self.h2d_bytes += int(customers_arr.nbytes + row_seeds.nbytes)

    d_pop = cp.empty((population_size, n_cities), dtype=cp.int32)
    block = 128
    grid = (population_size + block - 1) // block
    kernel = self._get_kernel()
    kernel(
      (grid,),
      (block,),
      (d_customers, d_seeds, d_pop, np.int32(population_size), np.int32(n_cities)),
    )
    self.kernel_launches += 1

    if self.return_gpu:
      return cast("Routes", d_pop)
    host_pop = np.asarray(d_pop.get(), dtype=np.int32)
    self.d2h_bytes += int(host_pop.nbytes)
    return host_pop


__all__: list[str] = ["InitializationCPURandom", "InitializationGPURandom"]
