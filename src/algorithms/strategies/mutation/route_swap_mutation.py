"""CPU and GPU Route Swap mutation for `0`-delimited VRP batches (`DP17-7`, `BUG-114`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.base import StrategyBase
from protocols.algorithm_strategies import MutationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  sample_distinct_pairs,
  to_cpu_batch,
)

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike

_BLOCK_SIZE: int = 128


def _resolve_swap_indices(
  rng: Generator,
  batch_size: int,
  n_cust: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
  """Sample unbiased distinct customer ordinal pairs `(k1, k2)` or `[0, 1)` floats when `n_cust` is unknown."""
  if n_cust is not None and n_cust >= 2:
    pairs = sample_distinct_pairs(rng, batch_size, 0, n_cust)
    return pairs[:, 0].copy(), pairs[:, 1].copy(), None, None
  u1 = np.asarray(rng.random(batch_size), dtype=np.float64)
  u2 = np.asarray(rng.random(batch_size), dtype=np.float64)
  return None, None, u1, u2


class MutationCPURouteSwap(StrategyBase, MutationStrategy):
  """CPU intra/inter-route customer Swap mutation for `0`-delimited VRP batches."""

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
    """Apply unbiased distinct customer Swap on CPU."""
    batch_np, was_1d, _ = to_cpu_batch(routes, copy=True)
    batch_size, width = batch_np.shape
    rng: Generator = resolve_rng(context, self.rng)
    if batch_size == 0 or width < 2:
      return restore_routes_shape(batch_np, was_1d)

    active = np.asarray(rng.random(batch_size) < mutation_rate, dtype=np.uint8)
    n_cust_ctx = (context.dimension - 1) if context is not None else int(np.count_nonzero(batch_np[0] > 0))
    if n_cust_ctx < 2:
      return restore_routes_shape(batch_np, was_1d)

    k1_arr, k2_arr, _, _ = _resolve_swap_indices(rng, batch_size, n_cust_ctx)
    if k1_arr is None or k2_arr is None:
      return restore_routes_shape(batch_np, was_1d)
    for idx in range(batch_size):
      if not active[idx]:
        continue
      row = batch_np[idx]
      cust_positions = np.flatnonzero(row > 0)
      n_cust = int(cust_positions.size)
      if n_cust < 2:
        continue
      k1 = int(k1_arr[idx])
      k2 = int(k2_arr[idx])
      if k1 >= n_cust or k2 >= n_cust or k1 == k2:
        continue
      pos1 = int(cust_positions[k1])
      pos2 = int(cust_positions[k2])
      row[pos1], row[pos2] = row[pos2], row[pos1]

    return restore_routes_shape(batch_np, was_1d)


class MutationGPURouteSwap(StrategyBase, MutationStrategy):
  """GPU CUDA intra/inter-route customer Swap mutation for `0`-delimited VRP batches."""

  def __init__(self, seed: SeedLike = None, return_gpu: bool = False) -> None:
    super().__init__(seed)
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    if self._kernel is None:
      self._kernel = load_kernel("route_swap_mutation.cu", "route_swap_mutation_kernel", __file__)
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
    """Apply unbiased distinct customer Swap on CUDA (`d2h_bytes == 0` when kept on GPU)."""
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
    if n_cust_ctx is not None and n_cust_ctx < 2:
      if keep_on_gpu:
        d_copy = cp.asarray(batch, dtype=cp.int32).copy()
        if not input_on_gpu:
          self.h2d_bytes += int(d_copy.nbytes)
        return restore_routes_shape(d_copy, was_1d)
      return restore_routes_shape(np.asarray(batch, dtype=np.int32).copy(), was_1d)

    k1_arr, k2_arr, u1_arr, u2_arr = _resolve_swap_indices(rng, batch_size, n_cust_ctx)
    if input_on_gpu:
      d_in = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      d_in = cp.asarray(np.ascontiguousarray(batch, dtype=np.int32), dtype=cp.int32)
      self.h2d_bytes += int(d_in.nbytes)
    d_out = cp.empty_like(d_in)
    d_active = cp.asarray(active, dtype=cp.uint8)
    self.h2d_bytes += int(active.nbytes)

    if k1_arr is not None and k2_arr is not None:
      d_k1: Any = cp.asarray(k1_arr, dtype=cp.int32)
      d_k2: Any = cp.asarray(k2_arr, dtype=cp.int32)
      d_u1: Any = 0
      d_u2: Any = 0
      self.h2d_bytes += int(k1_arr.nbytes + k2_arr.nbytes)
    elif u1_arr is not None and u2_arr is not None:
      d_k1 = 0
      d_k2 = 0
      d_u1 = cp.asarray(u1_arr, dtype=cp.float64)
      d_u2 = cp.asarray(u2_arr, dtype=cp.float64)
      self.h2d_bytes += int(u1_arr.nbytes + u2_arr.nbytes)
    else:
      raise RuntimeError("Unreachable swap index sampling state")

    blocks = (batch_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    self._get_kernel()(
      (blocks,),
      (_BLOCK_SIZE,),
      (d_in, d_out, d_active, d_k1, d_k2, d_u1, d_u2, np.int32(batch_size), np.int32(width)),
    )
    self.kernel_launches += 1

    if keep_on_gpu:
      return restore_routes_shape(d_out, was_1d)
    host_out = d_out.get()
    self.d2h_bytes += int(host_out.nbytes)
    return restore_routes_shape(host_out, was_1d)


__all__: list[str] = ["MutationCPURouteSwap", "MutationGPURouteSwap"]
