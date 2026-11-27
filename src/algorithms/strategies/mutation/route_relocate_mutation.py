"""CPU and GPU Route Relocate mutation for `0`-delimited VRP batches (`DP17-7`, `BUG-114`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.base import StrategyBase
from protocols.algorithm_strategies import MutationStrategy
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


def _resolve_relocate_indices(
  rng: Generator,
  batch_size: int,
  width: int,
  n_cust: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
  """Sample unbiased customer ordinal `k1` (or `[0, 1)` float) and target slot `s2` in shared order."""
  if n_cust is not None and n_cust >= 1:
    k1_arr: np.ndarray | None = np.asarray(rng.integers(0, n_cust, size=batch_size, dtype=np.int32), dtype=np.int32)
    u1_arr: np.ndarray | None = None
  else:
    k1_arr = None
    u1_arr = np.asarray(rng.random(batch_size), dtype=np.float64)
  s2_arr = np.asarray(rng.integers(0, width - 1, size=batch_size, dtype=np.int32), dtype=np.int32)
  return k1_arr, u1_arr, s2_arr


class MutationRouteRelocate(StrategyBase, MutationStrategy):
  """CPU intra/inter-route customer Relocate mutation for `0`-delimited VRP batches."""

  def __init__(self, seed: SeedLike = None) -> None:
    super().__init__(seed)

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D `0`-delimited route with probability `mutation_rate`."""
    return self.mutate_batch(individual, mutation_rate, context=context)

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Relocate a random customer to any valid slot `[0, width) \\ {pos1}` on CPU."""
    batch_np, was_1d, _ = to_cpu_batch(routes, copy=True)
    batch_size, width = batch_np.shape
    rng: Generator = resolve_rng(context, self.rng)
    if batch_size == 0 or width < 2:
      return restore_routes_shape(batch_np, was_1d)

    active = np.asarray(rng.random(batch_size) < mutation_rate, dtype=np.uint8)
    n_cust_ctx = (context.dimension - 1) if context is not None else int(np.count_nonzero(batch_np[0] > 0))
    if n_cust_ctx < 1:
      return restore_routes_shape(batch_np, was_1d)

    k1_arr, _, s2_arr = _resolve_relocate_indices(rng, batch_size, width, n_cust_ctx)
    if k1_arr is None:
      return restore_routes_shape(batch_np, was_1d)

    for idx in range(batch_size):
      if not active[idx]:
        continue
      row = batch_np[idx]
      cust_positions = np.flatnonzero(row > 0)
      n_cust = int(cust_positions.size)
      if n_cust < 1:
        continue
      k1 = int(k1_arr[idx])
      if k1 >= n_cust:
        continue
      pos1 = int(cust_positions[k1])
      s2 = int(s2_arr[idx])
      pos2 = s2 + (1 if s2 >= pos1 else 0)
      cust = int(row[pos1])
      if pos1 < pos2:
        row[pos1:pos2] = row[pos1 + 1 : pos2 + 1]
      else:
        row[pos2 + 1 : pos1 + 1] = row[pos2:pos1]
      row[pos2] = cust

    return restore_routes_shape(batch_np, was_1d)


class MutationGPURouteRelocate(StrategyBase, MutationStrategy):
  """GPU CUDA intra/inter-route customer Relocate mutation for `0`-delimited VRP batches."""

  def __init__(self, seed: SeedLike = None, return_gpu: bool = False) -> None:
    super().__init__(seed)
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    if self._kernel is None:
      self._kernel = load_kernel("route_relocate_mutation.cu", "route_relocate_mutation_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    require_gpu()
    super()._warmup_impl(context)
    cp = require_gpu()
    row = np.zeros((1, context.dimension + 1), dtype=np.int32)
    row[0, 1 : context.dimension] = np.arange(1, context.dimension, dtype=np.int32)
    self.mutate_batch(cp.asarray(row, dtype=cp.int32), mutation_rate=1.0, context=context)

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D `0`-delimited route on GPU with probability `mutation_rate`."""
    return self.mutate_batch(individual, mutation_rate, context=context)

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Relocate a random customer to any valid slot `[0, width) \\ {pos1}` on CUDA."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    batch_size, width = batch.shape
    input_on_gpu = is_gpu_array(batch)
    keep_on_gpu = input_on_gpu or self.return_gpu
    rng: Generator = resolve_rng(context, self.rng)

    if batch_size == 0 or width < 2:
      if keep_on_gpu:
        d_copy = cp.asarray(batch, dtype=cp.int32).copy()
        if not input_on_gpu:
          self.h2d_bytes += int(d_copy.nbytes)
        return restore_routes_shape(d_copy, was_1d)
      return restore_routes_shape(np.asarray(batch, dtype=np.int32).copy(), was_1d)

    active = np.asarray(rng.random(batch_size) < mutation_rate, dtype=np.uint8)
    n_cust_ctx = (context.dimension - 1) if context is not None else None
    if n_cust_ctx is not None and n_cust_ctx < 1:
      if keep_on_gpu:
        d_copy = cp.asarray(batch, dtype=cp.int32).copy()
        if not input_on_gpu:
          self.h2d_bytes += int(d_copy.nbytes)
        return restore_routes_shape(d_copy, was_1d)
      return restore_routes_shape(np.asarray(batch, dtype=np.int32).copy(), was_1d)

    if input_on_gpu:
      d_in = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      d_in = cp.asarray(np.ascontiguousarray(batch, dtype=np.int32), dtype=cp.int32)
      self.h2d_bytes += int(d_in.nbytes)
    k1_arr, u1_arr, s2_arr = _resolve_relocate_indices(rng, batch_size, width, n_cust_ctx)
    d_out = cp.empty_like(d_in)
    d_active = cp.asarray(active, dtype=cp.uint8)
    d_s2 = cp.asarray(s2_arr, dtype=cp.int32)
    self.h2d_bytes += int(active.nbytes + s2_arr.nbytes)

    if k1_arr is not None:
      d_k1: Any = cp.asarray(k1_arr, dtype=cp.int32)
      d_u1: Any = 0
      self.h2d_bytes += int(k1_arr.nbytes)
    elif u1_arr is not None:
      d_k1 = 0
      d_u1 = cp.asarray(u1_arr, dtype=cp.float64)
      self.h2d_bytes += int(u1_arr.nbytes)
    else:
      raise RuntimeError("Unreachable relocate index sampling state")

    blocks = (batch_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    self._get_kernel()(
      (blocks,),
      (_BLOCK_SIZE,),
      (d_in, d_out, d_active, d_k1, d_u1, d_s2, np.int32(batch_size), np.int32(width)),
    )
    self.kernel_launches += 1

    if keep_on_gpu:
      return restore_routes_shape(d_out, was_1d)
    host_out = d_out.get()
    self.d2h_bytes += int(host_out.nbytes)
    return restore_routes_shape(host_out, was_1d)


__all__: list[str] = ["MutationGPURouteRelocate", "MutationRouteRelocate"]
