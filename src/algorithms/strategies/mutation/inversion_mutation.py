"""
Inversion Mutation strategy for routing permutations.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Inversion Mutation.
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
  _mutation_bounds,
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  select_proximity_index,
  to_cpu_batch,
)

from ..base import StrategyBase

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


class MutationInversion(StrategyBase, MutationStrategy):
  """
  Inversion Mutation strategy for single routes and 2D route batches.

  Reverses a randomly chosen contiguous segment of the route with probability
  `mutation_rate`.
  """

  def __init__(self, seed: SeedLike = None, candidate_k: int | None = None) -> None:
    """Initialize MutationInversion with optional RNG seed.

    Args:
        seed: Optional random seed override.
        candidate_k: When set, the second cut is taken from the tour position of a
            random candidate neighbour of the first cut's city instead of a uniform
            draw (`DP10-20`). `None` keeps the uniform behaviour.
    """
    super().__init__(seed)
    self.candidate_k: int | None = candidate_k

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route `(n_cities,)` via segment inversion."""
    mutated = np.asarray(individual, dtype=np.int32).copy()
    low, high = _mutation_bounds(len(mutated))
    if high - low < 2:
      return mutated

    rng: Generator = resolve_rng(context, self.rng)

    if float(rng.random()) < mutation_rate:
      if self.candidate_k is not None and self.candidate_k > 0 and context is not None:
        first_cut = int(rng.integers(low, high))
        candidates = context.get_cpu_candidate_list(self.candidate_k)
        second_cut = select_proximity_index(mutated, first_cut, low, high, candidates, rng)
      else:
        drawn = rng.choice(np.arange(low, high), size=2, replace=False)
        first_cut, second_cut = int(drawn[0]), int(drawn[1])
      start, end = int(min(first_cut, second_cut)), int(max(first_cut, second_cut))
      mutated[start : end + 1] = mutated[start : end + 1][::-1]

    return mutated

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Mutate a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` via segment inversion."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes, copy=True)

    out = np.empty_like(batch_np, dtype=np.int32)
    for i in range(len(batch_np)):
      out[i] = self.mutate(batch_np[i], mutation_rate, context=context)

    return restore_routes_shape(out, was_1d, to_gpu=was_gpu)


class MutationGPUInversion(StrategyBase, MutationStrategy):
  """
  Modular GPU CUDA kernel Inversion Mutation strategy (`inversion_mutation.cu`).

  Reverses a randomly chosen contiguous segment across a batch of routes `(N, n_cities)` in parallel on GPU.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize GPU Inversion Mutation strategy."""
    super().__init__(seed)
    self._kernel: Any = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `inversion_mutation_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "inversion_mutation.cu",
        "inversion_mutation_kernel",
        __file__,
      )
    return self._kernel

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
    """Mutate a 1D route or 2D batch `(N, n_cities)` on GPU via `inversion_mutation.cu`.

    Raises:
        GPUUnavailableError: When CuPy is missing or no CUDA device is reachable. The
            strategy never delegates to `MutationInversion`, so a CPU batch is never
            reported as the GPU kernel's offspring (`DP9-2`).
    """
    cp = require_gpu()

    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n_cities = batch.shape
    if n_cities < 2:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    if input_on_gpu:
      d_routes = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      d_routes = cp.asarray(batch, dtype=cp.int32)
      self.h2d_bytes += int(d_routes.nbytes)

    first_0 = int(d_routes[0, 0].get()) if input_on_gpu else int(batch[0, 0])
    first_end = int(d_routes[0, -1].get()) if input_on_gpu else int(batch[0, -1])
    low, high = (1, n_cities - 1) if (n_cities > 3 and first_0 == first_end) else (0, n_cities)
    if high - low < 2:
      if input_on_gpu:
        return restore_routes_shape(d_routes, was_1d)
      return restore_routes_shape(cp.asnumpy(d_routes), was_1d)

    rng: Generator = resolve_rng(context, self.rng)
    probs_np = rng.random(num_routes, dtype=np.float64)
    pos1_np = rng.integers(low, high, size=num_routes, dtype=np.int32)
    pos2_np = rng.integers(low, high, size=num_routes, dtype=np.int32)

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
    cp.cuda.Stream.null.synchronize()
    self.kernel_launches += 1

    if input_on_gpu:
      return restore_routes_shape(d_routes, was_1d)
    self.d2h_bytes += int(d_routes.nbytes)
    return restore_routes_shape(cp.asnumpy(d_routes), was_1d)


__all__: list[str] = ["MutationGPUInversion", "MutationInversion"]
