"""
Tour cost evaluation strategies for routing algorithms.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of TSP tour cost evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from distances.pairwise import get_edge_type_code
from protocols.types import Fitness
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.route_helpers import ensure_routes_batch, to_cpu_batch

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from data_models.problem import Problem
  from protocols.problem_context import ProblemContext
  from protocols.types import Fitness, Route, RouteOrRoutes

#: Default `n` above which coordinate-based problems skip the dense `(n, n)` matrix.
_DEFAULT_MAX_MATRIX_DIMENSION: int = 10_000


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
    costs: Fitness = self.evaluate_batch(route, context)
    return float(costs[0])

  @time_strategy
  def evaluate_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> Fitness:
    """
    Compute 1D cost array `(N,)` for a 1D route `(n_cities,)` or 2D batch `(N, n_cities)`.
    """
    batch_np, _, _ = to_cpu_batch(routes)
    distances: np.ndarray = context.get_cpu_distances()
    next_cities: np.ndarray = np.roll(batch_np, -1, axis=1)
    edge_costs: np.ndarray = distances[batch_np, next_cities]
    costs: Fitness = np.sum(edge_costs, axis=1, dtype=np.float64)
    return costs


class GPUCostEvaluation:
  """
  GPU batch tour cost evaluation strategy (`cost_calculator.cu`).

  Computes closed-loop TSP tour costs for all `N` routes in parallel on GPU
  using `cost_calculator_kernel` (one CUDA block per tour, parallel edge reduction).

  When `n` exceeds `max_matrix_dimension` and the problem is coordinate-based, the
  strategy passes `distances = nullptr` plus `d_coords`, so the kernel computes each
  edge on the fly with `gpu_pairwise_distance` (`DP10-19`).
  """

  def __init__(
    self,
    return_gpu: bool = False,
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
  ) -> None:
    """
    Initialize GPU cost evaluation strategy.

    Args:
        return_gpu: If True and input `routes` is on GPU, return the `Fitness`
            array as a GPU `cp.ndarray` without D2H transfer.
        threads_per_block: Threads per block for parallel reduction (power of 2).
        max_matrix_dimension: Threshold above which coordinate-based problems use
            on-the-fly distances. `None` defers to the context's value.
    """
    self.return_gpu: bool = return_gpu
    self.threads_per_block: int = _next_power_of_2(threads_per_block)
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self._kernel: Any = None
    self._cached_distances_gpu: Any = None
    self._cached_problem_key: tuple[str, int, str, int] | None = None
    self._cached_coords_gpu: Any = None
    self._cached_coords_key: tuple[str, int, str, int] | None = None
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0
    self._cached_distances_gpu = None
    self._cached_problem_key = None
    self._cached_coords_gpu = None
    self._cached_coords_key = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return `cost_calculator_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "cost_calculator.cu",
        "cost_calculator_kernel",
        __file__,
      )
    return self._kernel

  def _get_distances_int32_gpu(self, context: ProblemContext | None = None, problem: Problem | None = None) -> Any:
    """Get the cached ``int32`` distance matrix on GPU for ``cost_calculator_kernel``.

    When ``context`` is supplied the matrix is taken from
    ``context.get_gpu_distances()``, which caches one matrix per problem instance
    (``BUG-14``). Without a context, the local cache is keyed by the full problem
    identity so two same-named instances with different data cannot alias.
    """
    if context is not None:
      return context.get_gpu_distances()
    if problem is None:
      if self._cached_distances_gpu is None:
        raise RuntimeError("A ProblemContext or Problem is required to obtain GPU distance matrices")
      return self._cached_distances_gpu

    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    if self._cached_distances_gpu is None or self._cached_problem_key != problem_key:
      if problem.distances is not None:
        source: np.ndarray = problem.distances
      elif problem.coordinates is not None:
        from distances.matrix import compute_distance_matrix

        source = compute_distance_matrix(problem.coordinates, problem.edge_type)
      else:
        raise ValueError(f"Problem '{problem.name}' has no distances or coordinates")
      self._cached_distances_gpu = cp.asarray(source, dtype=cp.int32)
      self._cached_problem_key = problem_key
    return self._cached_distances_gpu

  def _get_coords_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `float64` contiguous coordinates on GPU for the coordinate mode."""
    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    if self._cached_coords_gpu is None or self._cached_coords_key != problem_key:
      coords = context.get_gpu_coordinates()
      if coords is None:
        raise ValueError(f"Problem '{problem.name}' has no coordinates for the on-the-fly distance mode")
      self._cached_coords_gpu = cp.ascontiguousarray(coords, dtype=cp.float64)
      self._cached_coords_key = problem_key
      self.h2d_bytes += int(self._cached_coords_gpu.nbytes)
    return self._cached_coords_gpu

  def _distance_operands(self, context: ProblemContext, n: int) -> tuple[Any, Any, int]:
    """Return `(d_distances, d_coords, edge_type_code)` for one kernel launch.

    Coordinate mode passes `0` for `d_distances` (a CUDA `nullptr`) and never
    materializes the dense matrix, so memory stays `O(n)` instead of `O(n^2)`.
    """
    threshold = self.max_matrix_dimension
    if threshold is None:
      threshold = getattr(context, "max_matrix_dimension", _DEFAULT_MAX_MATRIX_DIMENSION)
    if context.uses_on_the_fly_distances(n, threshold):
      return 0, self._get_coords_gpu(context), get_edge_type_code(context.problem.edge_type)
    return self._get_distances_int32_gpu(context), 0, 0

  def evaluate(self, route: Route, context: ProblemContext) -> float:
    """Compute total closed-loop tour cost for a single 1D route `(n_cities,)`."""
    costs: Fitness = self.evaluate_batch(route, context)
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

    input_on_gpu: bool = isinstance(batch, cp.ndarray)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      batch_np: np.ndarray = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)

    d_distances, d_coords, edge_type_code = self._distance_operands(context, n)
    d_costs = cp.zeros(batch_size, dtype=cp.int64)
    cost_shared: int = self.threads_per_block * 8  # s_partials (long long)

    kernel = self._get_kernel()
    kernel(
      (batch_size,),
      (self.threads_per_block,),
      (
        d_population,
        d_distances,
        d_coords,
        d_costs,
        np.int32(n),
        np.int32(batch_size),
        np.int32(edge_type_code),
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


__all__: list[str] = ["CPUCostEvaluation", "GPUCostEvaluation"]
