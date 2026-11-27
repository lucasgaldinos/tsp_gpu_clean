"""Candidate-guided nearest-neighbour (GRASP RCL) CPU and GPU initializers (`Task 17.31`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.gpu_base import ensure_gpu_workspace
from protocols.algorithm_strategies import InitializationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.splitmix64 import draw_row_seeds, splitmix64_bounded_scalar

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


def _validate_k(k: int) -> int:
  if isinstance(k, bool) or not isinstance(k, int) or k < 1:
    raise ValueError(f"Candidate-list width k must be >= 1, got {k!r}")
  return k


def _nearest_neighbor_row_cpu(
  customers: np.ndarray,
  candidates: np.ndarray,
  seed_u64: int,
) -> np.ndarray:
  """Build one GRASP nearest-neighbour tour matching ``nearest_neighbor_init.cu`` bit-for-bit."""
  n_cities = int(customers.size)
  if n_cities == 0:
    return np.empty(0, dtype=np.int32)

  max_node = int(customers.max())
  cand_rows = int(candidates.shape[0])
  k_eff = int(candidates.shape[1]) if candidates.ndim == 2 else 0

  unvisited = customers.copy()
  pos_map = np.full(max_node + 1, -1, dtype=np.int32)
  pos_map[unvisited] = np.arange(n_cities, dtype=np.int32)

  state = int(seed_u64)
  rem_count = n_cities
  out_row = np.empty(n_cities, dtype=np.int32)

  state, start_idx = splitmix64_bounded_scalar(state, rem_count)
  current = int(unvisited[start_idx])
  out_row[0] = current

  last_idx = rem_count - 1
  last_node = int(unvisited[last_idx])
  unvisited[start_idx] = last_node
  unvisited[last_idx] = current
  pos_map[last_node] = start_idx
  pos_map[current] = last_idx
  rem_count = last_idx

  cand_order = np.arange(k_eff, dtype=np.int32)
  for step in range(1, n_cities):
    chosen = -1
    if 0 <= current < cand_rows and k_eff > 0:
      cand_order[:] = np.arange(k_eff, dtype=np.int32)
      cand_row = candidates[current]
      for i in range(k_eff):
        state, offset = splitmix64_bounded_scalar(state, k_eff - i)
        j = i + offset
        tmp = int(cand_order[i])
        cand_order[i] = cand_order[j]
        cand_order[j] = tmp
        nb = int(cand_row[int(cand_order[i])])
        if 0 <= nb <= max_node:
          p = int(pos_map[nb])
          if 0 <= p < rem_count:
            chosen = nb
            break

    if chosen < 0:
      state, u_idx = splitmix64_bounded_scalar(state, rem_count)
      chosen = int(unvisited[u_idx])

    out_row[step] = chosen
    p = int(pos_map[chosen])
    last_idx = rem_count - 1
    last_node = int(unvisited[last_idx])
    unvisited[p] = last_node
    unvisited[last_idx] = chosen
    pos_map[last_node] = p
    pos_map[chosen] = last_idx
    rem_count = last_idx
    current = chosen

  return out_row


class InitializationCPUNearestNeighbor(StrategyBase, InitializationStrategy):
  """CPU candidate-guided nearest-neighbour (GRASP RCL) initialization over ``N_k``."""

  def __init__(self, k: int = 20, seed: SeedLike = None) -> None:
    super().__init__(seed)
    self.k: int = _validate_k(k)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` nearest-neighbour tours on CPU."""
    if context is None:
      raise ValueError("InitializationCPUNearestNeighbor requires a ProblemContext to read the k-NN candidate list")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    customers_arr = np.asarray(customers, dtype=np.int32)
    n_cities = int(customers_arr.size)
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)
    if population_size == 0 or n_cities == 0:
      return np.empty((population_size, n_cities), dtype=np.int32)

    candidates = context.get_cpu_candidate_list(self.k)
    pop = np.empty((population_size, n_cities), dtype=np.int32)
    for idx in range(population_size):
      pop[idx] = _nearest_neighbor_row_cpu(customers_arr, candidates, int(row_seeds[idx]))
    return pop


class InitializationGPUNearestNeighbor(StrategyBase, InitializationStrategy):
  """VRAM-native CUDA nearest-neighbour initialization bit-identical to ``InitializationCPUNearestNeighbor``."""

  def __init__(self, k: int = 20, seed: SeedLike = None, return_gpu: bool = True) -> None:
    super().__init__(seed)
    self.k: int = _validate_k(k)
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None
    self._ws_unvisited: Any = None
    self._ws_pos: Any = None
    self._ws_cand: Any = None

  def reset_metrics(self) -> None:
    """Reset transfer/launch counters and cached GPU workspaces."""
    super().reset_metrics()
    self._ws_unvisited = None
    self._ws_pos = None
    self._ws_cand = None

  def _get_kernel(self) -> Any:
    """Lazily compile and cache ``nearest_neighbor_init_kernel``."""
    if self._kernel is None:
      self._kernel = load_kernel("nearest_neighbor_init.cu", "nearest_neighbor_init_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch ``nearest_neighbor_init_kernel`` ahead of measured runs without mutating state."""
    require_gpu()
    super()._warmup_impl(context)
    self._get_kernel()
    if context.dimension <= 1:
      return
    self.initialize(list(range(context.dimension)), 2, context)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` nearest-neighbour tours directly in VRAM."""
    if context is None:
      raise ValueError("InitializationGPUNearestNeighbor requires a ProblemContext to read the k-NN candidate list")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    cp = require_gpu()
    customers_arr = np.ascontiguousarray(customers, dtype=np.int32)
    n_cities = int(customers_arr.size)
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0 or n_cities == 0:
      empty_gpu = cp.empty((population_size, n_cities), dtype=cp.int32)
      return cast("Routes", empty_gpu) if self.return_gpu else np.empty((population_size, n_cities), dtype=np.int32)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_candidates = context.get_gpu_candidate_list(self.k)
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)

    d_customers = cp.asarray(customers_arr, dtype=cp.int32)
    d_seeds = cp.asarray(row_seeds, dtype=cp.uint64)
    self.h2d_bytes += int(customers_arr.nbytes + row_seeds.nbytes)

    max_node = int(customers_arr.max())
    cand_rows = int(d_candidates.shape[0])
    k_eff = int(d_candidates.shape[1]) if d_candidates.ndim == 2 else 0

    need_unvisited = population_size * n_cities
    need_pos = population_size * (max_node + 1)
    need_cand = max(1, population_size * max(1, k_eff))
    self._ws_unvisited = ensure_gpu_workspace(self._ws_unvisited, (need_unvisited,), cp.int32)
    self._ws_pos = ensure_gpu_workspace(self._ws_pos, (need_pos,), cp.int32)
    self._ws_cand = ensure_gpu_workspace(self._ws_cand, (need_cand,), cp.int32)

    d_pop = cp.empty((population_size, n_cities), dtype=cp.int32)
    block = 128
    grid = (population_size + block - 1) // block
    kernel = self._get_kernel()
    kernel(
      (grid,),
      (block,),
      (
        d_customers,
        d_candidates,
        d_seeds,
        self._ws_unvisited[:need_unvisited],
        self._ws_pos[:need_pos],
        self._ws_cand[:need_cand],
        d_pop,
        np.int32(population_size),
        np.int32(n_cities),
        np.int32(max_node),
        np.int32(cand_rows),
        np.int32(k_eff),
      ),
    )
    self.kernel_launches += 1

    if self.return_gpu:
      return cast("Routes", d_pop)
    host_pop = np.asarray(d_pop.get(), dtype=np.int32)
    self.d2h_bytes += int(host_pop.nbytes)
    return host_pop


__all__: list[str] = ["InitializationCPUNearestNeighbor", "InitializationGPUNearestNeighbor"]
