"""
2-opt local search strategies for tour refinement.

Two classes cover every 2-opt variant instead of six near-duplicates (`DP11-1`):

- `LocalSearchCPUTwoOpt`: one NumPy neighbourhood scan shared by three acceptance
  rules, selected with `mode="best" | "first" | "multi_move"`.
- `LocalSearchGPUTwoOpt`: one CUDA operand pipeline shared by three kernels, selected
  with `mode="batch" | "multi_move" | "naive"`.

Sharing the scan means CPU and GPU expose the exact same constructor shape (Liskov
substitution), and the multi-move acceptance rule has one implementation per backend,
so a CPU/GPU parity bug can only come from the kernel, not from the driver code.

References:
    - Croes, G. A. (1958). "A Method for Solving Traveling-Salesman Problems."
      Operations Research, 6(6), 791-812.
    - Flood, M. M. (1956). "The Traveling-Salesman Problem." Operations Research, 4(1).
    - Johnson, D. S., & McGeoch, L. A. (1997). "The Traveling Salesman Problem:
      A Case Study in Local Optimization." Local Search in Combinatorial Optimization.
    - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search Using
      GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
    - Rocki, K., & Suda, R. (2013). "High Performance GPU Accelerated Local
      Optimization in TSP." IEEE IPDPSW, pp. 1788-1796.
    - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a GPU
      Computing Platform." ICNAAM, Vol. 1389, pp. 264-267.
    - O'Neil, M. A., & Burtscher, M. (2015). "Rethinking the Parallelization of
      Random-Restart Hill Climbing." GPGPU-8, pp. 99-108.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.route_helpers import (
  ensure_routes_batch,
  inverse_positions,
  is_gpu_array,
  restore_routes_shape,
  to_cpu_batch,
)

from ..base import StrategyBase

if TYPE_CHECKING:
  from collections.abc import Iterator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes

type TwoOptCPUMode = Literal["best", "first", "multi_move"]
"""Acceptance rule shared by the three CPU 2-opt variants."""

type TwoOptGPUMode = Literal["batch", "multi_move", "naive"]
"""Kernel family selected by `LocalSearchGPUTwoOpt`."""

_CPU_MODES: frozenset[str] = frozenset({"best", "first", "multi_move"})
_GPU_MODES: frozenset[str] = frozenset({"batch", "multi_move", "naive"})

#: Kernel source and entry point per GPU mode.
_GPU_KERNELS: dict[str, tuple[str, str]] = {
  "batch": ("two_opt_batch.cu", "two_opt_batch_kernel"),
  "multi_move": ("two_opt_multi_move.cu", "two_opt_multi_move_batch_kernel"),
  "naive": ("two_opt_single.cu", "two_opt_kernel"),
}

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


class _DistanceOperandCache:
  """Strategy-local mirror of the GPU buffers owned by `ProblemContext`.

  The context is the single owner of the `int32` distance matrix and of the
  coordinate buffer (`DP10-17`); this cache only records which buffers the strategy
  bound last and how many bytes the first upload cost, so two GPU strategies running
  in the same pipeline never allocate a second copy in VRAM (`BUG-14`).
  """

  def __init__(self) -> None:
    self.distances: Any = None
    self.distance_key: tuple[str, int, str, int] | None = None
    self.coords: Any = None
    self.coords_key: tuple[str, int, str, int] | None = None

  def reset(self) -> None:
    """Forget the mirrored buffers after a metrics reset."""
    self.distances = None
    self.distance_key = None
    self.coords = None
    self.coords_key = None

  def resolve(
    self,
    context: ProblemContext,
    n: int,
    max_matrix_dimension: int | None,
  ) -> tuple[tuple[Any, Any, int], int]:
    """Return `((d_distances, d_coords, edge_type_code), uploaded_bytes)`.

    `ProblemContext.get_gpu_distance_operands` decides between the dense matrix and
    on-the-fly coordinates and hands back the context-owned buffers, so
    `EvaluationGPUCost` and `LocalSearchGPUTwoOpt` bind the same VRAM objects
    (`DP10-17`).
    """
    d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(n, max_matrix_dimension)
    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    uploaded_bytes = 0
    if isinstance(d_distances, int):
      if self.coords is not d_coords:
        uploaded_bytes = int(d_coords.nbytes)
      self.coords = d_coords
      self.coords_key = problem_key
      self.distances = None
      self.distance_key = None
    else:
      if self.distances is not d_distances:
        uploaded_bytes = int(d_distances.nbytes)
      self.distances = d_distances
      self.distance_key = problem_key
      self.coords = None
      self.coords_key = None
    return (d_distances, d_coords, edge_type_code), uploaded_bytes


def _shared_mem_bytes(block_size: int, n: int, tile_size: int) -> int:
  """Return dynamic shared-memory bytes for the per-block buffers.

  Layout: `s_deltas`, `s_swap_i`, `s_swap_j` (three `block_size` ints) plus either
  the full tour (`n` ints) or one tile (`tile_size + 1` ints, `Task 10.18`).
  """
  tour_ints = tile_size + 1 if tile_size > 0 else n
  return (block_size * 3 + tour_ints) * 4


def _resolve_candidates(
  context: ProblemContext,
  tour: np.ndarray,
  candidate_k: int | None,
  *,
  require_dense_ids: bool = False,
) -> np.ndarray | None:
  """Return the `(n, k_eff)` candidate table for `candidate_k`, or `None` to disable.

  `None` (the default) keeps the classic exhaustive scan. A non-positive `k` or a
  route holding node ids outside the context candidate table also fall back to the
  exhaustive scan, so a mismatched route can never index out of bounds.

  Args:
      context: Problem context owning the cached `N_k` table.
      tour: 1D route or 2D batch whose largest node id bounds the lookup.
      candidate_k: Requested neighbours per node; `None` disables candidate mode.
      require_dense_ids: Also reject routes whose node ids exceed the route width.
          The GPU kernels index a `(batch, n)` `positions` buffer by node id, so a
          sparse CVRP sub-route (`BUG-19`) cannot use the kernel candidate path and
          stays on the exhaustive scan.

  Returns:
      The candidate table, or `None` when the exhaustive scan must be used.
  """
  if candidate_k is None or candidate_k <= 0:
    return None
  candidates = context.get_cpu_candidate_list(candidate_k)
  largest_node = int(tour.max()) if tour.size else -1
  if largest_node >= candidates.shape[0]:
    return None
  if require_dense_ids and largest_node >= int(tour.shape[-1]):
    return None
  return candidates


class LocalSearchCPUTwoOpt(StrategyBase, LocalSearchStrategy):
  """
  CPU 2-opt local search with a selectable acceptance rule.

  All three modes walk the same neighbourhood (`_iter_moves`): the upper triangle of
  edge pairs, or the `k` nearest neighbours of each node when `candidate_k` is set.
  They differ only in which improving move is applied:

  - `"best"`: steepest descent, one move per pass (Croes, 1958; Flood, 1956).
  - `"first"`: first improvement, restarting the scan after each move
    (Johnson & McGeoch, 1997).
  - `"multi_move"`: every mutually disjoint improving move per pass
    (Rocki & Suda, 2013), the deterministic CPU reference for the GPU multi-move kernel.
  """

  def __init__(
    self,
    max_iterations: int = 10,
    mode: TwoOptCPUMode = "best",
    candidate_k: int | None = None,
  ) -> None:
    """
    Initialize the CPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt passes per route; `0` disables improvement.
        mode: Acceptance rule, one of `"best"`, `"first"`, `"multi_move"`.
        candidate_k: When set, evaluate only the `k` nearest neighbours of each node
            as the second break point (`DP10-20`). `None` keeps the exhaustive scan.
    """
    super().__init__()
    if mode not in _CPU_MODES:
      raise ValueError(f"Invalid 2-opt CPU mode {mode!r}; expected one of {sorted(_CPU_MODES)}")
    if max_iterations < 0:
      raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    self.mode: TwoOptCPUMode = mode
    self.max_iterations: int = max_iterations
    self.candidate_k: int | None = candidate_k

  def _iter_moves(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
  ) -> Iterator[tuple[int, int, float]]:
    """Yield `(i, j, delta)` for every admissible 2-opt edge pair of `tour`.

    The exhaustive branch walks the upper triangle `i + 2 <= j <= n - 1` and skips the
    wrap-around pair `(0, n - 1)`; it carries `c` over from the previous iteration
    because `tour[j + 1]` of one step is `tour[j]` of the next (`Task 10.16`).

    With a candidate table `N_k`, each row `i` scans only `N_k[tour[i], :]` and maps
    each neighbour `v` to its tour position `j = pos[v]` (`Theta(n * k)` per pass).
    """
    n: int = len(tour)
    if candidates is None:
      for i in range(n - 2):
        a: int = int(tour[i])
        b: int = int(tour[i + 1])
        d_ab: int = int(distances[a, b])
        c: int = int(tour[i + 2])
        for j in range(i + 2, n):
          d: int = int(tour[(j + 1) % n])
          if i == 0 and j == n - 1:
            c = d
            continue
          delta: float = float(distances[a, c] + distances[b, d]) - float(d_ab + distances[c, d])
          yield i, j, delta
          c = d
      return

    pos: np.ndarray = inverse_positions(tour)
    neighbors: int = int(candidates.shape[1])
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      for r in range(neighbors):
        neighbor = int(candidates[a, r])
        if neighbor < 0 or neighbor >= pos.size:
          continue
        j = int(pos[neighbor])
        if j < 0 or j <= i + 1 or (i == 0 and j == n - 1):
          continue
        c = int(tour[j])
        d = int(tour[(j + 1) % n])
        delta = float(distances[a, c] + distances[b, d]) - float(d_ab + distances[c, d])
        yield i, j, delta

  def _step_best(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply the single steepest improving move; return whether one was applied."""
    best_delta: float = 0.0
    best_i: int = -1
    best_j: int = -1
    for i, j, delta in self._iter_moves(tour, distances, candidates):
      if delta < best_delta - 1e-6:
        best_delta = delta
        best_i = i
        best_j = j
    if best_delta < -1e-6:
      tour[best_i + 1 : best_j + 1] = tour[best_i + 1 : best_j + 1][::-1]
      return True
    return False

  def _step_first(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply the first improving move found; return whether one was applied."""
    for i, j, delta in self._iter_moves(tour, distances, candidates):
      if delta < -1e-6:
        tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
        return True
    return False

  def _step_multi_move(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply every mutually disjoint improving move of one scan; return whether any was applied."""
    moves: list[tuple[float, int, int]] = [
      (delta, i, j) for i, j, delta in self._iter_moves(tour, distances, candidates) if delta < -1e-6
    ]
    if not moves:
      return False

    moves.sort(key=lambda move: move[0])
    applied_intervals: list[tuple[int, int]] = []
    for _delta, i, j in moves:
      conflict = any(not (j < start or i > end) for start, end in applied_intervals)
      if conflict:
        continue
      tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
      applied_intervals.append((i, j))
    return bool(applied_intervals)

  def _refine(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> np.ndarray:
    """Run up to `max_iterations` acceptance steps on a copy of `tour`."""
    improved: np.ndarray = tour.copy()
    if len(tour) < 4 or self.max_iterations <= 0:
      return improved

    step = {
      "best": self._step_best,
      "first": self._step_first,
      "multi_move": self._step_multi_move,
    }[self.mode]
    for _ in range(self.max_iterations):
      if not step(improved, distances, candidates):
        break
    return improved

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using CPU 2-opt."""
    distances = context.get_cpu_distances()
    route_np = np.asarray(route, dtype=np.int32)
    return self._refine(route_np, distances, _resolve_candidates(context, route_np, self.candidate_k))

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` using CPU 2-opt."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()
    candidates = _resolve_candidates(context, batch_np, self.candidate_k)
    improved: np.ndarray = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._refine(batch_np[idx], distances, candidates)
    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)


class LocalSearchGPUTwoOpt(StrategyBase, LocalSearchStrategy):
  """
  GPU 2-opt local search with a selectable kernel family.

  - `"batch"`: `two_opt_batch.cu`, one CUDA block per route, a single steepest move
    per pass (Fujimoto & Tsutsui, 2011; Rocki & Suda, 2012; O'Neil & Burtscher, 2015).
  - `"multi_move"`: `two_opt_multi_move.cu`, every mutually disjoint improving move
    per pass (Rocki & Suda, 2013; O'Neil, Burtscher, & Coates, 2011).
  - `"naive"`: `two_opt_single.cu`, per-route `CPU -> GPU -> CPU` transfers and one
    kernel launch per route and iteration, kept as the Chapter 4 transfer baseline.

  The grid modes perform a single batched H2D/D2H transfer; a CuPy-resident input is
  refined in place in VRAM with zero PCIe traffic (`BUG-15` / `DP10-18`).
  """

  def __init__(
    self,
    max_iterations: int = 10,
    mode: TwoOptGPUMode = "batch",
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    max_shared_tour_cities: int = _DEFAULT_MAX_SHARED_TOUR_CITIES,
    candidate_k: int | None = None,
  ) -> None:
    """
    Initialize the GPU 2-opt strategy.

    Args:
        max_iterations: Maximum 2-opt iterations per route; `0` disables improvement.
        mode: Kernel family, one of `"batch"`, `"multi_move"`, `"naive"`.
        threads_per_block: Maximum CUDA threads per block (grid modes).
        max_matrix_dimension: `n` above which coordinate-based problems use
            on-the-fly distances (`None` defers to the context value).
        max_shared_tour_cities: `n` above which the tour is scanned tile by tile
            instead of being cached whole in shared memory (`Task 10.18`).
        candidate_k: When set, the kernel evaluates only the `k` nearest neighbours
            of each node as the second break point (`DP10-20`). `None` keeps the
            exhaustive scan.
    """
    super().__init__()
    if mode not in _GPU_MODES:
      raise ValueError(f"Invalid 2-opt GPU mode {mode!r}; expected one of {sorted(_GPU_MODES)}")
    if max_iterations < 0:
      raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    self.mode: TwoOptGPUMode = mode
    self.max_iterations: int = max_iterations
    self.threads_per_block: int = threads_per_block
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.max_shared_tour_cities: int = max_shared_tour_cities
    self.candidate_k: int | None = candidate_k
    self._kernel: Any = None
    self._operands: _DistanceOperandCache = _DistanceOperandCache()

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    super().reset_metrics()
    self._operands.reset()

  def _get_kernel(self) -> Any:
    """Lazily compile and return the CUDA kernel selected by `mode`."""
    if self._kernel is None:
      source, entry_point = _GPU_KERNELS[self.mode]
      self._kernel = load_kernel(source, entry_point, __file__)
    return self._kernel

  def _tile_size(self, n: int) -> int:
    """Return 0 for the full-tour path, else the shared-memory tile width."""
    return 0 if n <= self.max_shared_tour_cities else _TILE_SIZE

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` through the selected GPU kernel family."""
    return self.improve_batch(route, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` on GPU.

    Raises:
        GPUUnavailableError: When CuPy is missing or no CUDA device is reachable.
            The strategy never degrades into `LocalSearchCPUTwoOpt`: a silent CPU run
            reported under the GPU variant's name corrupts the benchmark comparison
            the thesis depends on (`DP9-2`).
    """
    # Resolve the device before touching the batch, so a machine without CUDA fails at
    # the entry point instead of after paying the host-side marshalling cost.
    require_gpu()

    if self.mode == "naive":
      return self._improve_naive(routes, context)
    return self._improve_grid(routes, context)

  def _improve_naive(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Run the per-route `two_opt_single.cu` baseline with one transfer pair per route."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
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

  def _improve_grid(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Run the `N`-block grid kernels (`batch` or `multi_move`) with one batched transfer."""
    cp = require_gpu()
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

    (d_distances, d_coords, edge_type_code), uploaded_bytes = self._operands.resolve(
      context, n, self.max_matrix_dimension
    )
    self.h2d_bytes += uploaded_bytes
    block_size: int = _next_power_of_2(min(self.threads_per_block, max(1, n - 2)))
    tile_size: int = self._tile_size(n)
    shared_mem: int = _shared_mem_bytes(block_size, n, tile_size)

    # Candidate-list mode (`DP10-20`): upload `N_k` once and let each block build its
    # inverse tour map in its own `positions` slice. Sparse sub-routes stay exhaustive
    # because the kernel indexes `positions` by node id (`BUG-19`).
    candidate_table = _resolve_candidates(context, batch, self.candidate_k, require_dense_ids=True)
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


__all__: list[str] = [
  "LocalSearchCPUTwoOpt",
  "LocalSearchGPUTwoOpt",
  "TwoOptCPUMode",
  "TwoOptGPUMode",
]
