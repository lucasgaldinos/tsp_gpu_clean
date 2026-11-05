"""
2-opt local search strategies for tour refinement.

Provides:
- `CPUTwoOpt`: Best-improvement (steepest-descent) CPU 2-opt (Croes, 1958; Flood, 1956).
- `CPUFirstImprovementTwoOpt`: First-improvement CPU 2-opt (Croes, 1958; Johnson & McGeoch, 1997).
- `CPUMultiMoveTwoOpt`: CPU reference for multi-move disjoint 2-opt (Rocki & Suda, 2013).
- `NaiveGPUTwoOpt`: Per-route CPU->GPU->CPU transfers via `two_opt_single.cu`
  (Fujimoto & Tsutsui, 2011; Rocki & Suda, 2012).
- `BatchGPUTwoOpt`: Grid-level parallel batch 2-opt (one CUDA block per tour,
  single best move per pass) via `two_opt_batch.cu` (Fujimoto & Tsutsui, 2011;
  Rocki & Suda, 2012; O'Neil & Burtscher, 2015).
- `BatchGPUMultiMoveTwoOpt`: Grid-level parallel batch 2-opt applying multiple
  mutually disjoint (non-overlapping) improving moves per pass via
  `two_opt_multi_move.cu` (Rocki & Suda, 2013; O'Neil, Burtscher, & Coates, 2011).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from distances.pairwise import get_edge_type_code
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  to_cpu_batch,
)

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes

#: Default `n` above which coordinate-based problems skip the dense `(n, n)` matrix.
_DEFAULT_MAX_MATRIX_DIMENSION: int = 10_000
#: Default `n` above which the full-tour shared buffer is replaced by a tile.
_DEFAULT_MAX_SHARED_TOUR_CITIES: int = 11_520
#: Tour positions loaded per shared-memory tile (must match `TWO_OPT_TILE_SIZE`).
_TILE_SIZE: int = 2048


def _next_power_of_2(value: int) -> int:
  """Round `value` down to a power of 2 in `[1, 256]` for CUDA shared-memory reduction."""
  power = 1
  while power * 2 <= value:
    power *= 2
  return max(1, min(power, 256))


def _uses_on_the_fly_distances(
  context: ProblemContext,
  n: int,
  max_matrix_dimension: int | None,
) -> bool:
  """Return True when the kernel should compute distances from coordinates.

  Keeping `n` at or below the threshold preserves the dense-matrix fast path;
  above it a coordinate-based problem is evaluated with `gpu_pairwise_distance`
  and no `(n, n)` allocation (`DP10-19`).
  """
  threshold = max_matrix_dimension
  if threshold is None:
    threshold = getattr(context, "max_matrix_dimension", _DEFAULT_MAX_MATRIX_DIMENSION)
  problem = context.problem
  return n > threshold and problem.edge_type.upper() != "EXPLICIT" and problem.coordinates is not None


def _shared_mem_bytes(block_size: int, n: int, tile_size: int) -> int:
  """Return dynamic shared-memory bytes for the per-block buffers.

  Layout: `s_deltas`, `s_swap_i`, `s_swap_j` (three `block_size` ints) plus either
  the full tour (`n` ints) or one tile (`tile_size + 1` ints, `Task 10.18`).
  """
  tour_ints = tile_size + 1 if tile_size > 0 else n
  return (block_size * 3 + tour_ints) * 4


def _inverse_positions(tour: np.ndarray) -> np.ndarray:
  """Return the inverse tour permutation `pos` with `pos[tour[idx]] = idx` in `O(n)`.

  `pos` maps a node to its current position, which is what turns a candidate-list
  lookup `v = N_k[u, r]` into a tour index `j = pos[v]` (`DP10-20`).
  """
  positions = np.empty(len(tour), dtype=np.int32)
  positions[tour] = np.arange(len(tour), dtype=np.int32)
  return positions


def _resolve_candidates(
  context: ProblemContext,
  tour: np.ndarray,
  candidate_k: int | None,
) -> np.ndarray | None:
  """Return the `(n, k_eff)` candidate table for `candidate_k`, or `None` to disable.

  `None` (the default) keeps the classic exhaustive scan. A non-positive `k` or a
  route holding node ids outside the context candidate table also fall back to the
  exhaustive scan, so a mismatched route can never index out of bounds.
  """
  if candidate_k is None or candidate_k <= 0:
    return None
  candidates = context.get_cpu_candidate_list(candidate_k)
  largest_node = int(tour.max()) if tour.size else -1
  if largest_node >= candidates.shape[0]:
    return None
  return candidates


class CPUTwoOpt:
  """
  Best-improvement (steepest-descent) CPU 2-opt local search strategy.

  Evaluates all O(n^2) 2-opt edge swaps sequentially per route and applies
  the single best improving segment reversal per iteration for up to `max_iterations`.

  References:
      - Croes, G. A. (1958). "A Method for Solving Traveling-Salesman Problems."
        Operations Research, 6(6), 791-812.
      - Flood, M. M. (1956). "The Traveling-Salesman Problem." Operations Research,
        4(1), 61-75.
  """

  def __init__(self, max_iterations: int = 10, candidate_k: int | None = None) -> None:
    """
    Initialize CPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt refinement iterations per route.
        candidate_k: When set, evaluate only the `k` nearest neighbours of each node
            as the second break point (`DP10-20`). `None` keeps the exhaustive scan.
    """
    self.max_iterations: int = max_iterations
    self.candidate_k: int | None = candidate_k
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _two_opt_single(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None = None,
  ) -> np.ndarray:
    """Apply 2-opt local search to a single 1D route on CPU.

    With a candidate table `N_k`, each row `i` scans only `N_k[tour[i], :]` and maps
    each neighbour `v` to its tour position `j = pos[v]` (`Theta(n * k)` per pass).
    """
    n: int = len(tour)
    improved = tour.copy()
    if n < 4 or self.max_iterations <= 0:
      return improved

    for _ in range(self.max_iterations):
      best_delta = 0.0
      best_i = -1
      best_j = -1

      if candidates is None:
        for i in range(n - 2):
          # Hoisted row endpoints and current edge cost (`Task 10.16`).
          a: int = int(improved[i])
          b: int = int(improved[i + 1])
          d_ab: int = int(distances[a, b])
          # Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`.
          c: int = int(improved[i + 2])

          for j in range(i + 2, n):
            d: int = int(improved[(j + 1) % n])
            if i == 0 and j == n - 1:
              c = d
              continue

            old_cost = float(d_ab + distances[c, d])
            new_cost = float(distances[a, c] + distances[b, d])
            delta: float = new_cost - old_cost

            if delta < best_delta - 1e-6:
              best_delta = delta
              best_i = i
              best_j = j
            c = d
      else:
        pos = _inverse_positions(improved)
        neighbors: int = int(candidates.shape[1])
        for i in range(n - 2):
          a = int(improved[i])
          b = int(improved[i + 1])
          d_ab = int(distances[a, b])

          for r in range(neighbors):
            j = int(pos[candidates[a, r]])
            if j <= i + 1 or (i == 0 and j == n - 1):
              continue
            c = int(improved[j])
            d = int(improved[(j + 1) % n])
            old_cost = float(d_ab + distances[c, d])
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
    return self._two_opt_single(route_np, distances, _resolve_candidates(context, route_np, self.candidate_k))

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` using CPU 2-opt."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()
    candidates = _resolve_candidates(context, batch_np, self.candidate_k)
    improved: np.ndarray = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._two_opt_single(batch_np[idx], distances, candidates)

    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)


class NaiveGPUTwoOpt:
  """
  Hybrid-Naive GPU 2-opt strategy (`two_opt_single.cu`).

  Deliberately preserves the naive memory transfer pattern studied in Chapter 4:
  transfers each route individually `CPU -> GPU -> CPU` inside a Python loop
  and launches `two_opt_kernel` per route and per iteration.

  References:
      - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a
        GPU Computing Platform." ICNAAM, Vol. 1389, pp. 264-267.
      - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
        Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
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

    input_on_gpu: bool = isinstance(batch, cp.ndarray)
    batch_np = cp.asnumpy(batch) if input_on_gpu else np.asarray(batch, dtype=np.int32)

    distances_cpu = context.get_cpu_distances()
    distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
    self.h2d_bytes += int(distances_cpu.nbytes)

    threads_per_block: int = _next_power_of_2(min(256, max(1, n - 2)))
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

  References:
      - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a
        GPU Computing Platform." ICNAAM, Vol. 1389, pp. 264-267. (Block-per-tour
        shared-memory GA + 2-opt mapping).
      - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
        Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
        (Cooperative thread evaluation of 2-opt pairs and parallel reduction).
      - O'Neil, M. A., & Burtscher, M. (2015). "Rethinking the Parallelization of
        Random-Restart Hill Climbing: A Case Study in Optimizing a 2-Opt TSP Solver
        for GPU Execution." GPGPU-8, pp. 99-108.
  """

  def __init__(
    self,
    max_iterations: int = 10,
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    max_shared_tour_cities: int = _DEFAULT_MAX_SHARED_TOUR_CITIES,
    candidate_k: int | None = None,
  ) -> None:
    """
    Initialize Batch GPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route.
        threads_per_block: Maximum CUDA threads per block.
        max_matrix_dimension: `n` above which coordinate-based problems use
            on-the-fly distances (`None` defers to the context value).
        max_shared_tour_cities: `n` above which the tour is scanned tile by tile
            instead of being cached whole in shared memory (`Task 10.18`).
        candidate_k: When set, the kernel evaluates only the `k` nearest neighbours
            of each node as the second break point (`DP10-20`). `None` keeps the
            exhaustive scan.
    """
    self.max_iterations: int = max_iterations
    self.threads_per_block: int = threads_per_block
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.max_shared_tour_cities: int = max_shared_tour_cities
    self.candidate_k: int | None = candidate_k
    self._kernel: Any = None
    self._cached_distances_gpu: Any = None
    self._cached_problem_name: str | None = None
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
    self._cached_problem_name = None
    self._cached_coords_gpu = None
    self._cached_coords_key = None

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
    prob_name: Any | None = getattr(getattr(context, "problem", None), "name", None)
    if self._cached_distances_gpu is None or self._cached_problem_name != prob_name:
      distances_cpu = context.get_cpu_distances()
      self._cached_distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
      self._cached_problem_name = prob_name
      self.h2d_bytes += int(distances_cpu.nbytes)
    return self._cached_distances_gpu

  def _uses_coords(self, context: ProblemContext, n: int) -> bool:
    """Return True when this launch must use the on-the-fly coordinate path."""
    return _uses_on_the_fly_distances(context, n, self.max_matrix_dimension)

  def _get_coords_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `float64` contiguous coordinates on GPU."""
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
    """Return `(d_distances, d_coords, edge_type_code)` for one kernel launch."""
    if self._uses_coords(context, n):
      return 0, self._get_coords_gpu(context), get_edge_type_code(context.problem.edge_type)
    return self._get_distances_int32_gpu(context), 0, 0

  def _tile_size(self, n: int) -> int:
    """Return 0 for the full-tour path, else the shared-memory tile width."""
    return 0 if n <= self.max_shared_tour_cities else _TILE_SIZE

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

    input_on_gpu: bool = isinstance(batch, cp.ndarray)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      batch_np = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)
      self.h2d_bytes += int(batch_np.nbytes)

    d_distances, d_coords, edge_type_code = self._distance_operands(context, n)
    block_size: int = _next_power_of_2(min(self.threads_per_block, max(1, n - 2)))
    tile_size: int = self._tile_size(n)
    shared_mem: int = _shared_mem_bytes(block_size, n, tile_size)

    # Candidate-list mode (`DP10-20`): upload `N_k` once and let each block build its
    # inverse tour map in its own `positions` slice.
    candidate_table = _resolve_candidates(context, batch, self.candidate_k)
    candidate_count: int = int(candidate_table.shape[1]) if candidate_table is not None else 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if candidate_count > 0 and candidate_table is not None:
      d_candidates = cp.ascontiguousarray(cp.asarray(candidate_table, dtype=cp.int32))
      d_positions = cp.empty((batch_size, n), dtype=cp.int32)
      self.h2d_bytes += int(candidate_table.nbytes)

    kernel = self._get_kernel()
    for _ in range(self.max_iterations):
      kernel(
        (batch_size,),
        (block_size,),
        (
          d_population,
          d_distances,
          d_coords,
          np.int32(n),
          np.int32(batch_size),
          np.int32(edge_type_code),
          np.int32(tile_size),
          d_candidates,
          np.int32(candidate_count),
          d_positions,
        ),
        shared_mem=shared_mem,
      )
      self.kernel_launches += 1

    # No explicit host-device barrier (``BUG-15`` / ``DP10-18``): a GPU-resident result
    # stays stream-ordered for the caller's next CuPy operation, and the D2H branch
    # below synchronizes implicitly through ``.get()``.
    if input_on_gpu:
      return restore_routes_shape(d_population, was_1d)

    improved = d_population.get()
    self.d2h_bytes += int(improved.nbytes)
    return restore_routes_shape(improved, was_1d)


class CPUFirstImprovementTwoOpt:
  """
  First-Improvement CPU 2-opt local search strategy.

  Applies the first improving edge swap encountered immediately during the scan,
  restarting passes until no improvement is found or max_iterations is reached.

  References:
      - Croes, G. A. (1958). "A Method for Solving Traveling-Salesman Problems."
        Operations Research, 6(6), 791-812.
      - Johnson, D. S., & McGeoch, L. A. (1997). "The Traveling Salesman Problem:
        A Case Study in Local Optimization." In Local Search in Combinatorial
        Optimization (pp. 215-310), Wiley.
  """

  def __init__(self, max_iterations: int = 10, candidate_k: int | None = None) -> None:
    """Initialize First-Improvement CPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route.
        candidate_k: When set, scan only the `k` nearest neighbours of each node
            (`DP10-20`); `None` keeps the exhaustive scan.
    """
    self.max_iterations: int = max_iterations
    self.candidate_k: int | None = candidate_k
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _first_improvement_single(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None = None,
  ) -> np.ndarray:
    """Apply First-Improvement 2-opt to a single 1D route."""
    n: int = len(tour)
    improved = tour.copy()
    if n < 4 or self.max_iterations <= 0:
      return improved

    for _ in range(self.max_iterations):
      applied = False
      pos = _inverse_positions(improved) if candidates is not None else None
      neighbors: int = int(candidates.shape[1]) if candidates is not None else 0
      for i in range(n - 2):
        # Hoisted row endpoints and current edge cost (`Task 10.16`).
        a: int = int(improved[i])
        b: int = int(improved[i + 1])
        d_ab: int = int(distances[a, b])

        if candidates is None or pos is None:
          # Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`.
          c: int = int(improved[i + 2])
          for j in range(i + 2, n):
            d: int = int(improved[(j + 1) % n])
            if i == 0 and j == n - 1:
              c = d
              continue

            old_cost = float(d_ab + distances[c, d])
            new_cost = float(distances[a, c] + distances[b, d])
            delta: float = new_cost - old_cost

            if delta < -1e-6:
              improved[i + 1 : j + 1] = improved[i + 1 : j + 1][::-1]
              applied = True
              break
            c = d
        else:
          for r in range(neighbors):
            j = int(pos[candidates[a, r]])
            if j <= i + 1 or (i == 0 and j == n - 1):
              continue
            c = int(improved[j])
            d = int(improved[(j + 1) % n])
            old_cost = float(d_ab + distances[c, d])
            new_cost = float(distances[a, c] + distances[b, d])
            delta = new_cost - old_cost

            if delta < -1e-6:
              improved[i + 1 : j + 1] = improved[i + 1 : j + 1][::-1]
              applied = True
              break
        if applied:
          break
      if not applied:
        break

    return improved

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using First-Improvement CPU 2-opt."""
    distances = context.get_cpu_distances()
    route_np = np.asarray(route, dtype=np.int32)
    return self._first_improvement_single(
      route_np, distances, _resolve_candidates(context, route_np, self.candidate_k)
    )

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route or 2D batch of routes using First-Improvement CPU 2-opt."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()
    candidates = _resolve_candidates(context, batch_np, self.candidate_k)
    improved: np.ndarray = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._first_improvement_single(batch_np[idx], distances, candidates)
    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)


class CPUMultiMoveTwoOpt:
  """
  Multi-Move Disjoint-Segment CPU 2-opt local search strategy.

  Evaluates all O(n^2) edge pairs in a single pass and greedily applies all
  mutually disjoint (non-overlapping) improving segment reversals in the same
  iteration. Serves as the deterministic CPU reference counterpart for
  `BatchGPUMultiMoveTwoOpt`.

  References:
      - Rocki, K., & Suda, R. (2013). "High Performance GPU Accelerated Local
        Optimization in TSP." IEEE IPDPSW, pp. 1788-1796. (Multiple non-conflicting
        2-opt moves applied per O(n^2) pass).
  """

  def __init__(self, max_iterations: int = 10, candidate_k: int | None = None) -> None:
    """Initialize Multi-Move CPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route.
        candidate_k: When set, collect candidate moves only from the `k` nearest
            neighbours of each node (`DP10-20`); `None` keeps the exhaustive scan.
    """
    self.max_iterations: int = max_iterations
    self.candidate_k: int | None = candidate_k
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _multi_move_single(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None = None,
  ) -> np.ndarray:
    """Apply Multi-Move Disjoint 2-opt to a single 1D route."""
    n: int = len(tour)
    improved = tour.copy()
    if n < 4 or self.max_iterations <= 0:
      return improved

    for _ in range(self.max_iterations):
      moves: list[tuple[float, int, int]] = []
      pos = _inverse_positions(improved) if candidates is not None else None
      neighbors: int = int(candidates.shape[1]) if candidates is not None else 0

      for i in range(n - 2):
        # Hoisted row endpoints and current edge cost (`Task 10.16`).
        a: int = int(improved[i])
        b: int = int(improved[i + 1])
        d_ab: int = int(distances[a, b])

        if candidates is None or pos is None:
          # Carry-over: `c` of iteration `j + 1` is `d` of iteration `j`.
          c: int = int(improved[i + 2])
          for j in range(i + 2, n):
            d: int = int(improved[(j + 1) % n])
            if i == 0 and j == n - 1:
              c = d
              continue

            old_cost = float(d_ab + distances[c, d])
            new_cost = float(distances[a, c] + distances[b, d])
            delta: float = new_cost - old_cost

            if delta < -1e-6:
              moves.append((delta, i, j))
            c = d
        else:
          for r in range(neighbors):
            j = int(pos[candidates[a, r]])
            if j <= i + 1 or (i == 0 and j == n - 1):
              continue
            c = int(improved[j])
            d = int(improved[(j + 1) % n])
            old_cost = float(d_ab + distances[c, d])
            new_cost = float(distances[a, c] + distances[b, d])
            delta = new_cost - old_cost

            if delta < -1e-6:
              moves.append((delta, i, j))

      if not moves:
        break

      moves.sort(key=lambda x: x[0])
      applied_intervals: list[tuple[int, int]] = []
      for _delta, i, j in moves:
        conflict = any(not (j < start or i > end) for start, end in applied_intervals)
        if not conflict:
          improved[i + 1 : j + 1] = improved[i + 1 : j + 1][::-1]
          applied_intervals.append((i, j))

      if not applied_intervals:
        break

    return improved

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using Multi-Move CPU 2-opt."""
    distances = context.get_cpu_distances()
    route_np = np.asarray(route, dtype=np.int32)
    return self._multi_move_single(route_np, distances, _resolve_candidates(context, route_np, self.candidate_k))

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route or 2D batch of routes using Multi-Move CPU 2-opt."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()
    candidates = _resolve_candidates(context, batch_np, self.candidate_k)
    improved: np.ndarray = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._multi_move_single(batch_np[idx], distances, candidates)
    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)


class BatchGPUMultiMoveTwoOpt:
  """
  Multi-Move Disjoint-Segment Batch GPU 2-Opt strategy (`two_opt_multi_move.cu`).

  Evaluates all O(n^2) 2-opt edge pairs in parallel across threads in a CUDA block,
  then greedily applies up to `max_moves_per_pass` non-overlapping (mutually disjoint)
  improving segment reversals in the same iteration pass, amortizing the O(n^2)
  neighborhood scan across multiple swaps.

  References:
      - Rocki, K., & Suda, R. (2013). "High Performance GPU Accelerated Local
        Optimization in TSP." IEEE International Symposium on Parallel & Distributed
        Processing, Workshops and Phd Forum (IPDPSW), pp. 1788-1796.
      - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
        Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
      - O'Neil, M. A., & Burtscher, M. (2015). "Rethinking the Parallelization of
        Random-Restart Hill Climbing: A Case Study in Optimizing a 2-Opt TSP Solver
        for GPU Execution." GPGPU-8, pp. 99-108.
  """

  def __init__(
    self,
    max_iterations: int = 10,
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    max_shared_tour_cities: int = _DEFAULT_MAX_SHARED_TOUR_CITIES,
  ) -> None:
    """Initialize Multi-Move Batch GPU 2-opt strategy."""
    self.max_iterations = max_iterations
    self.threads_per_block = _next_power_of_2(threads_per_block)
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.max_shared_tour_cities: int = max_shared_tour_cities
    self._kernel: Any = None
    self._cached_distances_gpu: Any = None
    self._cached_problem_name: str | None = None
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
    self._cached_problem_name = None
    self._cached_coords_gpu = None
    self._cached_coords_key = None

  def _get_kernel(self) -> Any:
    """Lazily compile and return `two_opt_multi_move_batch_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "two_opt_multi_move.cu",
        "two_opt_multi_move_batch_kernel",
        __file__,
      )
    return self._kernel

  def _get_distances_int32_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `int32` distance matrix on GPU."""
    prob_name: Any | None = getattr(getattr(context, "problem", None), "name", None)
    if self._cached_distances_gpu is None or self._cached_problem_name != prob_name:
      distances_cpu = context.get_cpu_distances()
      self._cached_distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
      self._cached_problem_name = prob_name
      self.h2d_bytes += int(distances_cpu.nbytes)
    return self._cached_distances_gpu

  def _uses_coords(self, context: ProblemContext, n: int) -> bool:
    """Return True when this launch must use the on-the-fly coordinate path."""
    return _uses_on_the_fly_distances(context, n, self.max_matrix_dimension)

  def _get_coords_gpu(self, context: ProblemContext) -> Any:
    """Get or cache `float64` contiguous coordinates on GPU."""
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
    """Return `(d_distances, d_coords, edge_type_code)` for one kernel launch."""
    if self._uses_coords(context, n):
      return 0, self._get_coords_gpu(context), get_edge_type_code(context.problem.edge_type)
    return self._get_distances_int32_gpu(context), 0, 0

  def _tile_size(self, n: int) -> int:
    """Return 0 for the full-tour path, else the shared-memory tile width."""
    return 0 if n <= self.max_shared_tour_cities else _TILE_SIZE

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` via Multi-Move GPU 2-opt."""
    return self.improve_batch(route, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route or 2D batch of routes in parallel on GPU."""
    if not CUPY_AVAILABLE or cp is None:
      cpu_fallback = CPUMultiMoveTwoOpt(max_iterations=self.max_iterations)
      return cpu_fallback.improve_batch(routes, context)

    batch, was_1d = ensure_routes_batch(routes)
    batch_size, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      batch_np = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)
      self.h2d_bytes += int(batch_np.nbytes)

    d_distances, d_coords, edge_type_code = self._distance_operands(context, n)
    block_size: int = _next_power_of_2(min(self.threads_per_block, max(1, n - 2)))
    tile_size: int = self._tile_size(n)
    shared_mem: int = _shared_mem_bytes(block_size, n, tile_size)

    kernel = self._get_kernel()
    for _ in range(self.max_iterations):
      kernel(
        (batch_size,),
        (block_size,),
        (
          d_population,
          d_distances,
          d_coords,
          np.int32(n),
          np.int32(batch_size),
          np.int32(edge_type_code),
          np.int32(tile_size),
        ),
        shared_mem=shared_mem,
      )
      self.kernel_launches += 1

    # No explicit host-device barrier (``BUG-15`` / ``DP10-18``): a GPU-resident result
    # stays stream-ordered for the caller's next CuPy operation, and the D2H branch
    # below synchronizes implicitly through ``.get()``.
    if input_on_gpu:
      return restore_routes_shape(d_population, was_1d)

    improved = d_population.get()
    self.d2h_bytes += int(improved.nbytes)
    return restore_routes_shape(improved, was_1d)


__all__: list[str] = [
  "BatchGPUMultiMoveTwoOpt",
  "BatchGPUTwoOpt",
  "CPUFirstImprovementTwoOpt",
  "CPUMultiMoveTwoOpt",
  "CPUTwoOpt",
  "NaiveGPUTwoOpt",
]

