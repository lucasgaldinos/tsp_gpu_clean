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

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel, prev_power_of_2
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
    self.candidates: Any = None
    self.candidates_key: tuple[str, int, str, int] | None = None

  def reset(self) -> None:
    """Forget the mirrored buffers after a metrics reset."""
    self.distances = None
    self.distance_key = None
    self.coords = None
    self.coords_key = None
    self.candidates = None
    self.candidates_key = None

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

  def resolve_candidates(self, context: ProblemContext, k: int) -> tuple[Any, int]:
    """Return `(d_candidates, uploaded_bytes)` for the context-owned `N_k` table.

    `ProblemContext.get_gpu_candidate_list` caches the device table per `k`, so the
    first strategy that asks pays the single upload and every later generation, and
    every sibling strategy, binds the same VRAM object for zero bytes (`BUG-27`).
    """
    d_candidates = context.get_gpu_candidate_list(k)
    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    reused: bool = self.candidates is d_candidates and self.candidates_key == problem_key
    uploaded_bytes = 0 if reused else int(d_candidates.nbytes)
    self.candidates = d_candidates
    self.candidates_key = problem_key
    return d_candidates, uploaded_bytes


def _shared_mem_bytes(block_size: int, n: int, tile_size: int, use_candidates: bool = False) -> int:
  """Return dynamic shared-memory bytes for the per-block buffers.

  Layout: `s_deltas`, `s_swap_i`, `s_swap_j` (three `block_size` ints) plus either
  the full tour (`n` ints) or one tile (`tile_size + 1` ints, `Task 10.18`).

  Candidate mode never caches the tour (`DP10-20`): the kernel reads node ids straight
  from global memory, so reserving `n` ints of shared memory would cap `n` at the
  48 KB per-block budget for no benefit (`BUG-27`).
  """
  if use_candidates:
    return block_size * 3 * 4
  tour_ints = tile_size + 1 if tile_size > 0 else n
  return (block_size * 3 + tour_ints) * 4


def _resolve_candidates(
  context: ProblemContext,
  tour: np.ndarray,
  candidate_k: int | None,
) -> np.ndarray | None:
  """Return the `(n, k_eff)` candidate table for `candidate_k`, or `None` to disable.

  `None` (the default) keeps the classic exhaustive scan. A non-positive `k` or a
  route holding node ids outside the context candidate table also fall back to the
  exhaustive scan, so a mismatched route can never index out of bounds.

  Args:
      context: Problem context owning the cached `N_k` table.
      tour: 1D route or 2D batch whose largest node id bounds the lookup.
      candidate_k: Requested neighbours per node; `None` disables candidate mode.

  Returns:
      The candidate table, or `None` when the exhaustive scan must be used.
  """
  if candidate_k is None or candidate_k <= 0:
    return None
  candidates = context.get_cpu_candidate_list(candidate_k)
  largest_node = int(tour.max()) if tour.size else -1
  if largest_node >= candidates.shape[0]:
    return None
  return candidates


#: Largest number of `(i, j)` pair entries evaluated per NumPy block by the
#: exhaustive scan. Bounds the peak temporaries at a few megabytes for the largest
#: TSPLIB cohort (`n = 1002`), where the pair set alone is `~5e5` entries.
_PAIR_BLOCK: int = 1 << 18


@lru_cache(maxsize=2)
def _exhaustive_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Return the exhaustive 2-opt pair set `(i, j)` in ascending scan order.

  The neighbourhood is every pair with `i + 2 <= j`, minus the pair `(0, n - 1)`:
  reversing `tour[1:n]` is the identity move, so a sequential scan stops one short
  at `j = i` and at `j = n - 1` when `i == 0`. `np.triu_indices` produces exactly
  that order in one vectorised pass, which lets the three acceptance rules share a
  single enumeration instead of three copies of the same nested loop.

  Args:
      n: Tour length. Cached per `n` because the pair set depends on nothing else,
          and every route of a population shares the same length.

  Returns:
      `(i_idx, j_idx)` `int32` arrays aligned by position.
  """
  i_idx, j_idx = np.triu_indices(n, k=2)
  keep: np.ndarray = ~((i_idx == 0) & (j_idx == n - 1))
  return i_idx[keep].astype(np.int32, copy=False), j_idx[keep].astype(np.int32, copy=False)


def _pair_deltas(
  tour: np.ndarray,
  next_node: np.ndarray,
  edge_costs: np.ndarray,
  distances: np.ndarray,
  i_idx: np.ndarray,
  j_idx: np.ndarray,
) -> np.ndarray:
  """Return the `int64` gain of reversing `tour[i + 1 : j + 1]` for each pair.

  The move replaces edges `(a, b)` and `(c, d)` with `(a, c)` and `(b, d)`, so the
  gain is `d(a, c) + d(b, d) - d(a, b) - d(c, d)`; negative means the reversal
  shortens the tour.

  `next_node` is `tour` rotated left by one and `edge_costs[k]` is `d(tour[k],
  tour[k + 1])`, so the two retired edges are two lookups into a length-`n` vector
  instead of two more gathers from the dense `(n, n)` matrix. Arithmetic is widened
  to `int64` before summing: the matrix is `int32` and a tour cost near the TSPLIB
  ceiling makes `d(a, c) + d(b, d)` exceed the `int32` range even though each edge
  fits.

  Args:
      tour: Current tour, `(n,)` node ids.
      next_node: `tour` rotated left by one, `(n,)`.
      edge_costs: `int64` cost of each consecutive edge, `(n,)`.
      distances: `(n, n)` `int32` distance matrix.
      i_idx: Positions of the first break point, `(p,)`.
      j_idx: Positions of the second break point, `(p,)`.

  Returns:
      `int64` array `(p,)` of gains, aligned with `i_idx` and `j_idx`.
  """
  a: np.ndarray = tour[i_idx]
  b: np.ndarray = next_node[i_idx]
  c: np.ndarray = tour[j_idx]
  d: np.ndarray = next_node[j_idx]
  d_ac: np.ndarray = np.asarray(distances[a, c], dtype=np.int64)
  d_bd: np.ndarray = np.asarray(distances[b, d], dtype=np.int64)
  gain: np.ndarray = d_ac + d_bd - edge_costs[i_idx] - edge_costs[j_idx]
  return gain


class LocalSearchCPUTwoOpt(StrategyBase, LocalSearchStrategy):
  """
  CPU 2-opt local search with a selectable acceptance rule.

  All three modes walk the same neighbourhood: the upper triangle of
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

  def _pair_blocks(
    self,
    tour: np.ndarray,
    positions: np.ndarray | None,
    candidates: np.ndarray | None,
  ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield the `(i, j)` break-point pairs of one scan, in ascending scan order.

    Both neighbourhood flavours reduce to the same flat pair enumeration, so the
    three acceptance rules consume one generator instead of each carrying its own
    nested loop:

    - exhaustive (`candidates is None`): every pair with `i + 2 <= j`, in blocks of
      `_PAIR_BLOCK` to bound temporaries;
    - candidate mode: exactly the pairs a sequential `for r in range(k)` visit
      would reach, including duplicates when two neighbours of `a` share a
      position — the acceptance rules are order- and multiplicity-sensitive, so
      deduplicating here would silently change the tour sequence.

    Args:
        tour: Current tour, `(n,)` node ids.
        positions: `inverse_positions(tour)` in candidate mode, `None` otherwise.
        candidates: `(n, k_eff)` nearest-neighbour table, or `None`.

    Yields:
        `(i_idx, j_idx)` `int32` arrays aligned by position.
    """
    n: int = len(tour)
    if candidates is None or positions is None:
      i_all, j_all = _exhaustive_pairs(n)
      for start in range(0, i_all.size, _PAIR_BLOCK):
        yield i_all[start : start + _PAIR_BLOCK], j_all[start : start + _PAIR_BLOCK]
      return

    rows: int = n - 2
    neighbours: np.ndarray = candidates[tour[:rows]]
    in_range: np.ndarray = (neighbours >= 0) & (neighbours < positions.size)
    # `positions[neighbour]` is the position of that node in the tour, or `-1` when
    # the node is absent (sparse CVRP sub-route), which the bounds test below rejects.
    j_mat: np.ndarray
    if bool(in_range.all()):
      j_mat = positions[neighbours]
    else:
      j_mat = np.where(in_range, positions[np.clip(neighbours, 0, max(positions.size - 1, 0))], -1)
    i_mat: np.ndarray = np.arange(rows, dtype=np.int32)[:, np.newaxis]
    # `i == 0` must stop at `n - 1`: reversing `tour[1:n]` is the identity move.
    keep: np.ndarray = (j_mat > i_mat + 1) & (j_mat < n)
    if rows:
      keep[0, :] &= j_mat[0, :] < n - 1
    i_keep, _ = np.nonzero(keep)
    yield i_keep.astype(np.int32, copy=False), j_mat[keep]

  def _scan_context(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return `(next_node, edge_costs, positions)` shared by one scan pass.

    Recomputing the retired-edge vector costs `O(n)` per pass, but it replaces the
    `2 n^2` matrix gathers a sequential scan performs, which is the difference this
    vectorisation buys.
    """
    next_node: np.ndarray = np.roll(tour, -1)
    edge_costs: np.ndarray = distances[tour, next_node].astype(np.int64)
    positions: np.ndarray | None = inverse_positions(tour) if candidates is not None else None
    return next_node, edge_costs, positions

  def _step_best(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply the single steepest improving move; return whether one was applied.

    Ties resolve to the first pair in scan order, matching the strict `<` of a
    sequential scan, so the resulting tour sequence stays bit-identical to the
    scalar reference (`TASK 13.14`).
    """
    n: int = len(tour)
    if n < 4:
      return False
    next_node, edge_costs, positions = self._scan_context(tour, distances, candidates)

    best_delta: int = 0
    best_i: int = -1
    best_j: int = -1
    for i_idx, j_idx in self._pair_blocks(tour, positions, candidates):
      if i_idx.size == 0:
        continue
      deltas: np.ndarray = _pair_deltas(tour, next_node, edge_costs, distances, i_idx, j_idx)
      pos: int = int(np.argmin(deltas))
      gain: int = int(deltas[pos])
      if gain < best_delta:
        best_delta = gain
        best_i = int(i_idx[pos])
        best_j = int(j_idx[pos])

    if best_delta < 0:
      tour[best_i + 1 : best_j + 1] = tour[best_i + 1 : best_j + 1][::-1]
      return True
    return False

  def _step_first(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply the first improving move found in scan order; return whether one was applied."""
    n: int = len(tour)
    if n < 4:
      return False
    next_node, edge_costs, positions = self._scan_context(tour, distances, candidates)

    for i_idx, j_idx in self._pair_blocks(tour, positions, candidates):
      if i_idx.size == 0:
        continue
      improving: np.ndarray = _pair_deltas(tour, next_node, edge_costs, distances, i_idx, j_idx) < 0
      if not bool(improving.any()):
        continue
      # `argmax` on a boolean mask returns the first `True` in scan order.
      pos: int = int(np.argmax(improving))
      i: int = int(i_idx[pos])
      j: int = int(j_idx[pos])
      tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
      return True
    return False

  def _step_multi_move(self, tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
    """Apply every mutually disjoint improving move of one scan; return whether any was applied.

    The disjoint selection is inherently sequential — each accepted reversal removes
    its window from the candidates — so only that part stays a Python loop, over the
    improving moves of the pass rather than over all `O(n^2)` pairs. A boolean
    occupancy vector replaces the quadratic `any()` overlap check: two windows
    conflict exactly when their inclusive index ranges intersect, which is what
    marking `[i, j]` and testing the slice reports.
    """
    n: int = len(tour)
    if n < 4:
      return False
    next_node, edge_costs, positions = self._scan_context(tour, distances, candidates)

    gains: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for i_idx, j_idx in self._pair_blocks(tour, positions, candidates):
      if i_idx.size == 0:
        continue
      deltas: np.ndarray = _pair_deltas(tour, next_node, edge_costs, distances, i_idx, j_idx)
      improving: np.ndarray = deltas < 0
      if not bool(improving.any()):
        continue
      gains.append(deltas[improving])
      starts.append(i_idx[improving])
      ends.append(j_idx[improving])
    if not gains:
      return False

    # Stable sort on the gain keeps scan order among equal gains, so the greedy
    # pass applies the same windows in the same sequence as `list.sort(key=...)`.
    order: np.ndarray = np.argsort(np.concatenate(gains), kind="stable")
    from_idx: np.ndarray = np.concatenate(starts)
    to_idx: np.ndarray = np.concatenate(ends)
    occupied: np.ndarray = np.zeros(n, dtype=bool)
    applied: bool = False
    for slot in order:
      i: int = int(from_idx[slot])
      j: int = int(to_idx[slot])
      if occupied[i : j + 1].any():
        continue
      tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
      occupied[i : j + 1] = True
      applied = True
    return applied

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
  - `"multi_move"`: `two_opt_multi_move.cu`, many mutually disjoint improving moves
    per pass (Rocki & Suda, 2013; O'Neil, Burtscher, & Coates, 2011).
  - `"naive"`: `two_opt_single.cu`, per-route `CPU -> GPU -> CPU` transfers and one
    kernel launch per route and iteration, kept as the Chapter 4 transfer baseline.

  The grid modes perform a single batched H2D/D2H transfer; a CuPy-resident input is
  refined in place in VRAM with zero PCIe traffic (`BUG-15` / `DP10-18`).

  All three families honour `candidate_k`: the same `k` binds the same `N_k` table and
  restricts every `j` scan to those neighbours (`DP10-20`), so the per-route baseline
  searches the neighbourhood its configuration declares (`BUG-57`).

  `"multi_move"` is not a re-implementation of the CPU acceptance rule: each thread
  retains only its own steepest candidate, so one pass offers at most
  `min(block_size, n // 2)` moves to the greedy disjoint-selection step (Rocki & Suda,
  2013). That deliberate bound is what keeps the per-pass bookkeeping O(block_size)
  instead of O(n^2); the CPU reference scans every pair and therefore exposes all
  improving moves of the pass, so GPU and CPU `multi_move` converge in the same
  direction but not necessarily through the same intermediate tours.
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

  def warmup(self, context: ProblemContext) -> None:
    """Compile the selected kernel and launch it once on a valid permutation.

    The first launch is what makes the driver load the module; doing it here keeps
    that cost out of the measured repetitions, including for periodic pipeline steps
    that would only launch it at generation 10 (`BUG-25`). The batch is `arange`,
    never `cp.zeros` (`BUG-24`).
    """
    super().warmup(context)
    self.improve_batch(self._warmup_batch(context, rows=1), context)

  def _tile_size(self, n: int) -> int:
    """Return 0 for the full-tour path, else the shared-memory tile width."""
    return 0 if n <= self.max_shared_tour_cities else _TILE_SIZE

  def _uses_candidates(self, context: ProblemContext, n: int) -> bool:
    """Return whether the kernel candidate path is admissible for this batch.

    The kernel builds the inverse map `pos[tour[idx]] = idx` in a `(batch, n)` buffer
    and indexes it by node id, so it needs a dense `0..n-1` permutation. Sparse CVRP
    sub-routes (`problem_type == "CVRP"`) and batches narrower than the problem
    (`n < context.dimension`) therefore keep the exhaustive scan.

    The decision is taken from host metadata only — `problem_type`, `dimension` and the
    batch width — so the hot path never calls `int(tour.max())` on a GPU array, which
    used to force one D2H synchronization per generation (`BUG-27`).
    """
    if self.candidate_k is None or self.candidate_k <= 0:
      return False
    if self._tile_size(n) > 0:
      # Tiling and candidate mode are mutually exclusive: a tiled block already
      # streams the tour through shared memory, so the `positions` scratch would
      # compete with the window for the 48 KB budget for no throughput gain.
      return False
    if context.problem.problem_type == "CVRP":
      return False
    return n >= context.dimension

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

    # Candidate-list mode (`DP10-20`, `BUG-57`): bind the same context-owned `N_k` table
    # and charge the same upload as `_improve_grid`, so `naive` searches the same
    # `Theta(n * k)` neighbourhood as the grid kernels for a given `candidate_k`.
    candidate_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n):
      d_candidates, candidate_bytes = self._operands.resolve_candidates(context, int(self.candidate_k or 0))
      self.h2d_bytes += candidate_bytes
      candidate_count = int(d_candidates.shape[1])
      # Inverse map in VRAM, mirroring `_improve_grid`: keeping it in shared memory
      # would halve this kernel's capacity, which is already bounded by `s_tour`.
      d_positions = cp.empty(n, dtype=cp.int32)

    threads_per_block: int = prev_power_of_2(min(256, max(1, n - 2)), min_val=1, max_val=256)
    # Three reduction buffers plus `s_tour[n]`; the candidate path adds no shared memory,
    # so both modes share the ceiling pinned by `max_shared_tour_cities`.
    shared_mem: int = _shared_mem_bytes(threads_per_block, n, tile_size=0)
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
          (d_tour, distances_gpu, np.int32(n), np.int32(0), d_candidates, np.int32(candidate_count), d_positions),
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
    block_size: int = prev_power_of_2(min(self.threads_per_block, max(1, n - 2)), min_val=1, max_val=256)
    tile_size: int = self._tile_size(n)

    # Candidate-list mode (`DP10-20`): bind the context-owned `N_k` once and let each
    # block build its inverse tour map in its own `positions` slice. The table lives in
    # VRAM across generations; only the first bind charges H2D bytes (`BUG-27`).
    candidate_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n):
      d_candidates, candidate_bytes = self._operands.resolve_candidates(context, int(self.candidate_k or 0))
      self.h2d_bytes += candidate_bytes
      candidate_count = int(d_candidates.shape[1])
      d_positions = cp.empty((batch_size, n), dtype=cp.int32)

    shared_mem: int = _shared_mem_bytes(block_size, n, tile_size, use_candidates=candidate_count > 0)

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
