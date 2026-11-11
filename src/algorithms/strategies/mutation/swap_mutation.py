"""
Swap Mutation strategies for routing permutations.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Swap Mutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from protocols.algorithm_strategies import MutationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  resolve_route_bounds,
  restore_routes_shape,
  sample_mutation_positions,
  to_cpu_batch,
)

from ..base import StrategyBase

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


class MutationSwap(StrategyBase, MutationStrategy):
  """
  Swap Mutation strategy for single routes and 2D route batches.

  Swaps two randomly chosen positions in a route with probability `mutation_rate`.
  """

  def __init__(self, seed: SeedLike = None, candidate_k: int | None = None) -> None:
    """Initialize MutationSwap with optional RNG seed.

    Args:
        seed: Optional random seed override.
        candidate_k: When set, the second swap position is taken from the tour
            position of a random candidate neighbour of the first city instead of a
            uniform draw (`DP10-20`). `None` keeps the uniform behaviour.
    """
    super().__init__(seed)
    self.candidate_k: int | None = candidate_k

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route `(n_cities,)` with probability `mutation_rate`."""
    return self.mutate_batch(individual, mutation_rate, context=context)

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Mutate a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` of routes.

    Positions come from the single shared host oracle, so this method and
    `MutationGPUSwap.mutate_batch` issue identical draws and stay bit-identical
    for the same seed (`SWAP-A`, `DP11-14`).
    """
    batch_np, was_1d, was_gpu = to_cpu_batch(routes, copy=True)
    low, high = resolve_route_bounds(batch_np.shape[1], context=context, route_sample=batch_np[0])
    if high - low < 2:
      return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

    rng: Generator = resolve_rng(context, self.rng)
    probs = rng.random(len(batch_np))
    pos1, pos2, _ = sample_mutation_positions(
      rng,
      batch_np,
      low,
      high,
      candidate_k=self.candidate_k,
      context=context,
    )

    out = batch_np.copy()
    active = np.flatnonzero(probs < mutation_rate)
    out[active, pos1[active]], out[active, pos2[active]] = (
      out[active, pos2[active]],
      out[active, pos1[active]],
    )
    return restore_routes_shape(out, was_1d, to_gpu=was_gpu)


class MutationGPUSwap(StrategyBase, MutationStrategy):
  """
  Modular GPU CUDA kernel Swap Mutation strategy (`swap_mutation.cu`).

  Applies swap mutation across a batch of routes `(N, n_cities)` in parallel on GPU.
  """

  def __init__(self, seed: SeedLike = None, candidate_k: int | None = None) -> None:
    """Initialize GPU Swap Mutation strategy.

    Args:
        seed: Optional random seed override.
        candidate_k: Accepted for LSP parity with `MutationSwap` (`DP11-20`); when set,
            the second position comes from the candidate neighbourhood (`BUG-28`).
    """
    super().__init__(seed)
    self.candidate_k: int | None = candidate_k
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `swap_mutation_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "swap_mutation.cu",
        "swap_mutation_kernel",
        __file__,
      )
    return self._kernel

  def warmup(self, context: ProblemContext) -> None:
    """Compile the kernel and launch it once on a valid permutation (`BUG-24`, `BUG-25`)."""
    super().warmup(context)
    self.mutate_batch(self._warmup_batch(context, rows=1), 1.0, context=context)

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route `(n_cities,)`."""
    return self.mutate_batch(individual, mutation_rate, context=context)

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Mutate a 1D route or 2D batch `(N, n_cities)` on GPU via `swap_mutation.cu`.

    Raises:
        GPUUnavailableError: When CuPy is missing or no CUDA device is reachable. The
            strategy never delegates to `MutationSwap`, so a CPU batch is never
            reported as the GPU kernel's offspring (`DP9-2`).
    """
    cp = require_gpu()

    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n_cities = batch.shape
    low, high = resolve_route_bounds(n_cities, context=context, route_sample=batch[0])
    if high - low < 2:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    if input_on_gpu:
      d_routes = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      d_routes = cp.asarray(batch, dtype=cp.int32)
      self.h2d_bytes += int(d_routes.nbytes)

    rng: Generator = resolve_rng(context, self.rng)
    probs_np = rng.random(num_routes)
    pos1_np, pos2_np, d2h_positions = sample_mutation_positions(
      rng,
      batch,
      low,
      high,
      candidate_k=self.candidate_k,
      context=context,
    )
    self.d2h_bytes += d2h_positions

    d_probs = cp.asarray(probs_np, dtype=cp.float64)
    d_pos1 = cp.asarray(pos1_np, dtype=cp.int32)
    d_pos2 = cp.asarray(pos2_np, dtype=cp.int32)
    self.h2d_bytes += int(d_probs.nbytes + d_pos1.nbytes + d_pos2.nbytes)

    threads_per_block = 128
    blocks = (num_routes + threads_per_block - 1) // threads_per_block
    kernel = self._get_kernel()
    kernel(
      (blocks,),
      (threads_per_block,),
      (
        d_routes,
        d_probs,
        d_pos1,
        d_pos2,
        np.float64(mutation_rate),
        np.int32(num_routes),
        np.int32(n_cities),
      ),
    )
    self.kernel_launches += 1

    if input_on_gpu:
      return restore_routes_shape(d_routes, was_1d)
    self.d2h_bytes += int(d_routes.nbytes)
    return restore_routes_shape(cp.asnumpy(d_routes), was_1d)


__all__: list[str] = ["MutationGPUSwap", "MutationSwap"]
