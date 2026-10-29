"""
Mutation strategies for evolutionary routing algorithms.

All mutation strategies support both:
- `mutate(individual, mutation_rate, context=None) -> Route` (1D route `(n_cities,)`)
- `mutate_batch(routes, mutation_rate, context=None) -> RouteOrRoutes` (1D route or 2D batch `(N, n_cities)`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from protocols.algorithm_strategies import (
  Route,
  RouteOrRoutes,
  ensure_routes_batch,
  restore_routes_shape,
)
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext


def _mutation_bounds(route: np.ndarray) -> tuple[int, int]:
  """
  Determine valid mutable index slice `[low, high)` for a 1D route.

  If the route is depot-bounded (`route[0] == route[-1]` and `len > 3`),
  preserves the depot at index `0` and `len - 1`. Otherwise, allows all
  indices `[0, len(route))`.
  """
  n = len(route)
  if n > 3 and int(route[0]) == int(route[-1]):
    return 1, n - 1
  return 0, n


class SwapMutation:
  """
  Swap Mutation strategy for single routes and 2D route batches.

  Swaps two randomly chosen positions in a route with probability `mutation_rate`.
  """

  def __init__(self, seed: int | None = None) -> None:
    """Initialize SwapMutation with optional RNG seed."""
    self.rng = np.random.default_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route `(n_cities,)` with probability `mutation_rate`."""
    mutated = np.asarray(individual, dtype=np.int32).copy()
    low, high = _mutation_bounds(mutated)
    if high - low < 2:
      return mutated

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    if float(rng.random()) < mutation_rate:
      idx1, idx2 = rng.choice(np.arange(low, high), size=2, replace=False)
      mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

    return mutated

  @time_strategy
  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Mutate a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` of routes."""
    batch, was_1d = ensure_routes_batch(routes)
    is_gpu = CUPY_AVAILABLE and cp is not None and isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if is_gpu else np.asarray(batch, dtype=np.int32).copy()

    out = np.empty_like(batch_np, dtype=np.int32)
    for i in range(len(batch_np)):
      out[i] = self.mutate(batch_np[i], mutation_rate, context=context)

    if is_gpu and cp is not None:
      return restore_routes_shape(cp.asarray(out, dtype=cp.int32), was_1d)
    return restore_routes_shape(out, was_1d)


class InversionMutation:
  """
  Inversion Mutation strategy for single routes and 2D route batches.

  Reverses a randomly chosen contiguous segment of the route with probability
  `mutation_rate`.
  """

  def __init__(self, seed: int | None = None) -> None:
    """Initialize InversionMutation with optional RNG seed."""
    self.rng = np.random.default_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route `(n_cities,)` via segment inversion."""
    mutated = np.asarray(individual, dtype=np.int32).copy()
    low, high = _mutation_bounds(mutated)
    if high - low < 2:
      return mutated

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    if float(rng.random()) < mutation_rate:
      cuts = rng.choice(np.arange(low, high), size=2, replace=False)
      start, end = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))
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
    batch, was_1d = ensure_routes_batch(routes)
    is_gpu = CUPY_AVAILABLE and cp is not None and isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if is_gpu else np.asarray(batch, dtype=np.int32).copy()

    out = np.empty_like(batch_np, dtype=np.int32)
    for i in range(len(batch_np)):
      out[i] = self.mutate(batch_np[i], mutation_rate, context=context)

    if is_gpu and cp is not None:
      return restore_routes_shape(cp.asarray(out, dtype=cp.int32), was_1d)
    return restore_routes_shape(out, was_1d)


class GPUSwapMutation:
  """
  Modular GPU CUDA kernel Swap Mutation strategy (`swap_mutation.cu`).

  Applies swap mutation across a batch of routes `(N, n_cities)` in parallel on GPU.
  """

  def __init__(self, seed: int | None = None) -> None:
    """Initialize GPU Swap Mutation strategy."""
    self.seed = seed
    self.rng = np.random.default_rng(seed)
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
    """Lazily compile and return the `swap_mutation_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "swap_mutation.cu",
        "swap_mutation_kernel",
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
    """Mutate a 1D route or 2D batch `(N, n_cities)` on GPU via `swap_mutation.cu`."""
    if not CUPY_AVAILABLE or cp is None:
      cpu_mut = SwapMutation(seed=self.seed)
      return cpu_mut.mutate_batch(routes, mutation_rate, context=context)

    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n_cities = batch.shape
    if n_cities < 2:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu = isinstance(batch, cp.ndarray)
    if input_on_gpu:
      d_routes = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      d_routes = cp.asarray(batch, dtype=cp.int32)
      self.h2d_bytes += int(d_routes.nbytes)

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    probs_np = rng.random(num_routes, dtype=np.float64)
    pos1_np = rng.integers(0, n_cities, size=num_routes, dtype=np.int32)
    pos2_np = rng.integers(0, n_cities, size=num_routes, dtype=np.int32)

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
