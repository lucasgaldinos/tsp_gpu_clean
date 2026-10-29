"""
Fitness / cost evaluation strategies for routes and population batches.

Every strategy in this module accepts `RouteOrRoutes` (`(n_cities,)` or `(N, n_cities)`),
returning a scalar float via `evaluate(route, context)` or a 1D cost array `(N,)` via
`evaluate_batch(routes, context)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from protocols.algorithm_strategies import (
  Fitness,
  Route,
  RouteOrRoutes,
  ensure_routes_batch,
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


class CPUCostEvaluation:
  """
  CPU tour cost evaluation strategy using `context.get_cpu_distances()`.

  Computes closed-loop TSP tour costs `sum_{i=0}^{n-1} d(tour[i], tour[(i+1) % n])`
  using vectorized NumPy indexing across the batch of routes.
  """

  def __init__(self) -> None:
    """Initialize CPU cost evaluation strategy."""
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def evaluate(self, route: Route, context: ProblemContext) -> float:
    """Compute total closed-loop tour cost for a single 1D route `(n_cities,)`."""
    costs = self.evaluate_batch(route, context)
    return float(costs[0])

  @time_strategy
  def evaluate_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> Fitness:
    """
    Compute 1D cost array `(N,)` for a 1D route `(n_cities,)` or 2D batch `(N, n_cities)`.
    """
    batch, _ = ensure_routes_batch(routes)
    is_gpu = CUPY_AVAILABLE and cp is not None and isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if is_gpu else np.asarray(batch, dtype=np.int32)

    distances = context.get_cpu_distances()
    next_cities = np.roll(batch_np, -1, axis=1)
    edge_costs = distances[batch_np, next_cities]
    costs = np.sum(edge_costs, axis=1, dtype=np.float64)
    return costs


class GPUCostEvaluation:
  """
  GPU batch tour cost evaluation strategy (`cost_calculator.cu`).

  Computes closed-loop TSP tour costs for all `N` routes in parallel on GPU
  using `cost_calculator_kernel` (one CUDA block per tour, parallel edge reduction).
  """

  def __init__(self, return_gpu: bool = False, threads_per_block: int = 256) -> None:
    """
    Initialize GPU cost evaluation strategy.

    Args:
        return_gpu: If True and input `routes` is on GPU, return the `Fitness`
            array as a GPU `cp.ndarray` without D2H transfer.
        threads_per_block: Threads per block for parallel reduction (power of 2).
    """
    self.return_gpu = return_gpu
    self.threads_per_block = _next_power_of_2(threads_per_block)
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
    """Lazily compile and return `cost_calculator_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "cost_calculator.cu",
        "cost_calculator_kernel",
        __file__,
      )
    return self._kernel

  def _get_distances_int32_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `int32` distance matrix on GPU for `cost_calculator_kernel`."""
    prob_name = getattr(getattr(context, "problem", None), "name", None)
    if self._cached_distances_gpu is None or self._cached_problem_name != prob_name:
      distances_cpu = context.get_cpu_distances()
      self._cached_distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
      self._cached_problem_name = prob_name
    return self._cached_distances_gpu

  def evaluate(self, route: Route, context: ProblemContext) -> float:
    """Compute total closed-loop tour cost for a single 1D route `(n_cities,)`."""
    costs = self.evaluate_batch(route, context)
    if CUPY_AVAILABLE and cp is not None and isinstance(costs, cp.ndarray):
      return float(cp.asnumpy(costs)[0])
    return float(costs[0])

  @time_strategy
  def evaluate_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> Fitness:
    """
    Compute 1D cost array `(N,)` on GPU for a 1D route or 2D batch `(N, n_cities)`.
    """
    if not CUPY_AVAILABLE or cp is None:
      cpu_eval = CPUCostEvaluation()
      return cpu_eval.evaluate_batch(routes, context)

    batch, _ = ensure_routes_batch(routes)
    batch_size, n = batch.shape

    input_on_gpu = isinstance(batch, cp.ndarray)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      batch_np = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)

    d_distances = self._get_distances_int32_gpu(context)
    d_costs = cp.zeros(batch_size, dtype=cp.int64)
    cost_shared = self.threads_per_block * 8  # s_partials (long long)

    kernel = self._get_kernel()
    kernel(
      (batch_size,),
      (self.threads_per_block,),
      (
        d_population,
        d_distances,
        d_costs,
        np.int32(n),
        np.int32(batch_size),
      ),
      shared_mem=cost_shared,
    )
    cp.cuda.Stream.null.synchronize()
    self.kernel_launches += 1

    if input_on_gpu and self.return_gpu:
      return cast("Fitness", d_costs.astype(cp.float64))

    costs_cpu = d_costs.get().astype(np.float64)
    self.d2h_bytes += int(d_costs.nbytes)
    return cast("Fitness", costs_cpu)
