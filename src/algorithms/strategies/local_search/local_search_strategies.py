"""
Local search (2-opt) strategies for route and batch refinement.

Every strategy in this module accepts `RouteOrRoutes` (`(n_cities,)` or `(N, n_cities)`),
allowing 2-opt to be composed at any point in the evolutionary pipeline:
- After crossover (`post_crossover`)
- After mutation (`post_mutation`)
- After fitness evaluation (`post_evaluation`)
- Only once at the end of the full genetic algorithm (`post_evolution`)
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


def _next_power_of_2(value: int) -> int:
  """Round `value` down to a power of 2 in `[1, 256]` for CUDA shared-memory reduction."""
  power = 1
  while power * 2 <= value:
    power *= 2
  return max(1, min(power, 256))


class NoOpLocalSearch:
  """Identity local search strategy that returns routes unchanged."""

  def __init__(self, max_iterations: int = 0) -> None:
    """Initialize no-op local search."""
    self.max_iterations = max_iterations
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Return single 1D route unchanged."""
    return route.copy()

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Return 1D route or 2D batch of routes unchanged."""
    batch, was_1d = ensure_routes_batch(routes)
    return restore_routes_shape(batch.copy(), was_1d)


class CPUTwoOpt:
  """
  Pure NumPy CPU 2-opt local search strategy.

  Evaluates all O(n^2) 2-opt edge swaps sequentially per route and applies
  the best improving reversal per iteration for up to `max_iterations`.
  """

  def __init__(self, max_iterations: int = 10) -> None:
    """
    Initialize CPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt refinement iterations per route.
    """
    self.max_iterations = max_iterations
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _two_opt_single(self, tour: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Apply 2-opt local search to a single 1D route on CPU."""
    n = len(tour)
    improved = tour.copy()
    if n < 4 or self.max_iterations <= 0:
      return improved

    for _ in range(self.max_iterations):
      best_delta = 0.0
      best_i = -1
      best_j = -1

      for i in range(n - 2):
        for j in range(i + 2, n):
          if i == 0 and j == n - 1:
            continue

          a, b = int(improved[i]), int(improved[i + 1])
          c, d = int(improved[j]), int(improved[(j + 1) % n])

          old_cost = float(distances[a, b] + distances[c, d])
          new_cost = float(distances[a, c] + distances[b, d])
          delta = new_cost - old_cost

          if delta < best_delta - 1e-6:
            best_delta = delta
            best_i = i
            best_j = j

      if best_delta < -1e-6:
        improved[best_i + 1 : best_j + 1] = improved[best_i + 1 : best_j + 1][::-1]
      else:
        break

    return improved

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using CPU 2-opt."""
    distances = context.get_cpu_distances()
    route_np = np.asarray(route, dtype=np.int32)
    return self._two_opt_single(route_np, distances)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` using CPU 2-opt."""
    batch, was_1d = ensure_routes_batch(routes)
    is_gpu = CUPY_AVAILABLE and cp is not None and isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if is_gpu else np.asarray(batch, dtype=np.int32)

    distances = context.get_cpu_distances()
    improved = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._two_opt_single(batch_np[idx], distances)

    if is_gpu and cp is not None:
      return restore_routes_shape(cp.asarray(improved, dtype=cp.int32), was_1d)
    return restore_routes_shape(improved, was_1d)


class NaiveGPUTwoOpt:
  """
  Hybrid-Naive GPU 2-opt strategy (`two_opt_single.cu`).

  Deliberately preserves the naive memory transfer pattern studied in Chapter 4:
  transfers each route individually `CPU -> GPU -> CPU` inside a Python loop
  and launches `two_opt_kernel` per route and per iteration.
  """

  def __init__(self, max_iterations: int = 10) -> None:
    """
    Initialize Hybrid-Naive GPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route.
    """
    self.max_iterations = max_iterations
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
    """Lazily compile and return `two_opt_kernel` from `two_opt_single.cu`."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "two_opt_single.cu",
        "two_opt_kernel",
        __file__,
      )
    return self._kernel

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using `two_opt_single.cu`."""
    return self.improve_batch(route, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """
    Improve a 1D route or 2D batch `(N, n_cities)` using per-route GPU transfers (`two_opt_single.cu`).
    """
    if not CUPY_AVAILABLE or cp is None:
      cpu_fallback = CPUTwoOpt(max_iterations=self.max_iterations)
      return cpu_fallback.improve_batch(routes, context)

    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu = isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if input_on_gpu else np.asarray(batch, dtype=np.int32)

    distances_cpu = context.get_cpu_distances()
    distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
    self.h2d_bytes += int(distances_cpu.nbytes)

    threads_per_block = _next_power_of_2(min(256, max(1, n - 2)))
    shared_mem = (
      threads_per_block * 4  # s_deltas (int32)
      + threads_per_block * 4  # s_swap_i (int32)
      + threads_per_block * 4  # s_swap_j (int32)
      + n * 4  # s_tour (int32)
    )
    kernel = self._get_kernel()

    improved = np.empty_like(batch_np, dtype=np.int32)
    for i in range(num_routes):
      tour = batch_np[i]
      d_tour = cp.asarray(tour, dtype=cp.int32)
      self.h2d_bytes += int(tour.nbytes)

      for _ in range(self.max_iterations):
        kernel(
          (1,),
          (threads_per_block,),
          (d_tour, distances_gpu, np.int32(n), np.int32(0)),
          shared_mem=shared_mem,
        )
        self.kernel_launches += 1

      improved[i] = d_tour.get()
      self.d2h_bytes += int(tour.nbytes)

    if input_on_gpu:
      return restore_routes_shape(cp.asarray(improved, dtype=cp.int32), was_1d)
    return restore_routes_shape(improved, was_1d)


class BatchGPUTwoOpt:
  """
  Batch GPU 2-opt strategy (`two_opt_batch.cu`).

  Processes an entire batch of `N` routes in parallel using a `N`-block CUDA grid
  (one CUDA block per route, threads cooperating on the `O(n^2)` 2-opt neighborhood).
  If `routes` is already a CuPy array in VRAM, executes in-place in VRAM with zero
  PCIe transfers; if `routes` is a NumPy array, performs a single batched H2D and
  D2H transfer.
  """

  def __init__(self, max_iterations: int = 10, threads_per_block: int = 256) -> None:
    """
    Initialize Batch GPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route.
        threads_per_block: Maximum CUDA threads per block.
    """
    self.max_iterations = max_iterations
    self.threads_per_block = threads_per_block
    self._kernel: Any = None
    self._cached_distances_gpu: Any = None
    self._cached_problem_name: str | None = None
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0
    self._cached_distances_gpu = None
    self._cached_problem_name = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return `two_opt_batch_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "two_opt_batch.cu",
        "two_opt_batch_kernel",
        __file__,
      )
    return self._kernel

  def _get_distances_int32_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `int32` distance matrix on GPU for `two_opt_batch_kernel`."""
    prob_name = getattr(getattr(context, "problem", None), "name", None)
    if self._cached_distances_gpu is None or self._cached_problem_name != prob_name:
      distances_cpu = context.get_cpu_distances()
      self._cached_distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
      self._cached_problem_name = prob_name
      self.h2d_bytes += int(distances_cpu.nbytes)
    return self._cached_distances_gpu

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` via `two_opt_batch.cu`."""
    return self.improve_batch(route, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """
    Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` in parallel on GPU.
    """
    if not CUPY_AVAILABLE or cp is None:
      cpu_fallback = CPUTwoOpt(max_iterations=self.max_iterations)
      return cpu_fallback.improve_batch(routes, context)

    batch, was_1d = ensure_routes_batch(routes)
    batch_size, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu = isinstance(batch, cp.ndarray)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      batch_np = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)
      self.h2d_bytes += int(batch_np.nbytes)

    d_distances = self._get_distances_int32_gpu(context)
    block_size = _next_power_of_2(min(self.threads_per_block, max(1, n - 2)))
    shared_mem = (
      block_size * 4  # s_deltas (int32)
      + block_size * 4  # s_swap_i (int32)
      + block_size * 4  # s_swap_j (int32)
      + n * 4  # s_tour (int32)
    )

    kernel = self._get_kernel()
    for _ in range(self.max_iterations):
      kernel(
        (batch_size,),
        (block_size,),
        (
          d_population,
          d_distances,
          np.int32(n),
          np.int32(batch_size),
        ),
        shared_mem=shared_mem,
      )
      self.kernel_launches += 1

    cp.cuda.Stream.null.synchronize()

    if input_on_gpu:
      return restore_routes_shape(d_population, was_1d)

    improved = d_population.get()
    self.d2h_bytes += int(improved.nbytes)
    return restore_routes_shape(improved, was_1d)
