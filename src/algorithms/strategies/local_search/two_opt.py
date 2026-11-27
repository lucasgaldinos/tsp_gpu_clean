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
from .gpu_base import BaseLocalSearchGPU

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


def _resolve_topology(context: ProblemContext) -> tuple[bool, bool, bool]:
  """Return `(is_closed, is_cyclic, fixed_start)` from `context` metadata.

  - `is_closed`: `True` for closed depot-bounded VRP sub-routes `[0, ..., 0]`
    (`context.is_closed`), where both endpoints `0` and `n - 1` stay fixed.
  - `is_cyclic`: `True` when the route closes with a wrap-around edge
    `(tour[n - 1], tour[0])` (`not is_closed` and `return_to_depot`).
  - `fixed_start`: `True` when `tour[0]` is a fixed depot or cycle anchor
    (`is_closed`, `is_cyclic`, or `problem_type in {"CVRP", "OVRP", "DCVRP"}`).
  """
  is_closed: bool = bool(getattr(context, "is_closed", False))
  return_to_depot: bool = bool(getattr(context, "return_to_depot", True))
  has_depot: bool = str(context.problem.problem_type) in {"CVRP", "OVRP", "DCVRP"}
  if is_closed:
    return True, False, True
  is_cyclic: bool = return_to_depot
  fixed_start: bool = is_cyclic or has_depot
  return False, is_cyclic, fixed_start


def _resolve_candidates(
  context: ProblemContext,
  tour: np.ndarray,
  candidate_k: int | None,
) -> np.ndarray | None:
  """Return the `(n, k_eff)` candidate table for `candidate_k`, or `None` to disable.

  `None` (the default) keeps the classic exhaustive scan. Non-positive `k`, closed or
  non-cyclic topologies, sub-routes whose length differs from `context.dimension`, or
  routes holding node ids outside the candidate table fall back to the exhaustive scan
  (`BUG-107`).

  Args:
      context: Problem context owning the cached `N_k` table.
      tour: 1D route or 2D batch whose length and node ids bound the lookup.
      candidate_k: Requested neighbours per node; `None` disables candidate mode.

  Returns:
      The candidate table, or `None` when the exhaustive scan must be used.
  """
  if candidate_k is None or candidate_k <= 0:
    return None
  is_closed, is_cyclic, _ = _resolve_topology(context)
  if is_closed or not is_cyclic or str(context.problem.problem_type) in {"CVRP", "OVRP", "DCVRP"}:
    return None
  route_len = int(tour.shape[-1]) if tour.ndim >= 1 and tour.size > 0 else 0
  if route_len != context.dimension:
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


@lru_cache(maxsize=8)
def _exhaustive_pairs(
  n: int,
  is_closed: bool = False,
  is_cyclic: bool = True,
  fixed_start: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
  """Return the exhaustive 2-opt pair set `(i, j)` in ascending scan order.

  Each pair `(i, j)` reverses `tour[i + 1 : j + 1]`:

  - **Cyclic tour** (`is_closed=False, is_cyclic=True`): `0 <= i < j <= n - 1`
    with `j >= i + 2`, excluding `(0, n - 1)` whose reversal `tour[1:n]` is the
    identity move.
  - **Closed depot-bounded route** (`is_closed=True`): `0 <= i < j <= n - 2`
    with `j >= i + 2`, keeping both `tour[0]` and `tour[n - 1]` fixed (`BUG-107`).
  - **Open path** (`is_closed=False, is_cyclic=False`): allows `j == n - 1`
    including `(0, n - 1)` because there is no closing edge `(tour[n - 1], tour[0])`;
    when `fixed_start=False` (Open TSP), also prepends prefix reversals
    `i == -1` for `1 <= j <= n - 2` (`tour[0 : j + 1]`).

  Args:
      n: Route length.
      is_closed: True for depot-bounded `[0, ..., 0]` routes.
      is_cyclic: True for closed-loop tours with wrap-around edge `(n - 1, 0)`.
      fixed_start: True when `tour[0]` cannot move (depot or cycle anchor).

  Returns:
      `(i_idx, j_idx)` `int32` arrays aligned by position.
  """
  i_raw, j_raw = np.triu_indices(n, k=2)
  i_idx: np.ndarray = i_raw.astype(np.int32, copy=False)
  j_idx: np.ndarray = j_raw.astype(np.int32, copy=False)
  if is_closed:
    keep: np.ndarray = j_idx < n - 1
    return i_idx[keep], j_idx[keep]
  if is_cyclic:
    keep = ~((i_idx == 0) & (j_idx == n - 1))
    return i_idx[keep], j_idx[keep]
  if fixed_start:
    return i_idx, j_idx
  prefix_j: np.ndarray = np.arange(1, max(1, n - 1), dtype=np.int32)
  prefix_i: np.ndarray = np.full(prefix_j.shape, -1, dtype=np.int32)
  return np.concatenate((prefix_i, i_idx)), np.concatenate((prefix_j, j_idx))


def _pair_deltas(
  tour: np.ndarray,
  next_node: np.ndarray,
  edge_costs: np.ndarray,
  distances: np.ndarray,
  i_idx: np.ndarray,
  j_idx: np.ndarray,
  *,
  is_closed: bool = False,
  is_cyclic: bool = True,
) -> np.ndarray:
  """Return the `int64` gain of reversing `tour[i + 1 : j + 1]` for each pair.

  For interior and cyclic moves the reversal replaces edges `(a, b)` and `(c, d)`
  with `(a, c)` and `(b, d)`, giving `d(a, c) + d(b, d) - d(a, b) - d(c, d)`.
  On open paths (`not is_closed and not is_cyclic`), prefix moves (`i == -1`)
  replace `(c, d)` with `(tour[0], d)` and suffix moves (`j == n - 1`) replace
  `(a, b)` with `(a, c)` (`BUG-107`). Arithmetic is widened to `int64` before
  summing so edge sums near the TSPLIB ceiling cannot overflow `int32`.

  Args:
      tour: Current tour, `(n,)` node ids.
      next_node: `tour` rotated left by one, `(n,)`.
      edge_costs: `int64` cost of each consecutive edge, `(n,)`.
      distances: `(dist_stride, dist_stride)` `int32` distance matrix.
      i_idx: Positions of the first break point, `(p,)`.
      j_idx: Positions of the second break point, `(p,)`.
      is_closed: True for closed depot-bounded `[0, ..., 0]` routes.
      is_cyclic: True for cyclic tours with wrap-around edge `(n - 1, 0)`.

  Returns:
      `int64` array `(p,)` of gains, aligned with `i_idx` and `j_idx`.
  """
  if is_closed or is_cyclic:
    a: np.ndarray = tour[i_idx]
    b: np.ndarray = next_node[i_idx]
    c: np.ndarray = tour[j_idx]
    d: np.ndarray = next_node[j_idx]
    d_ac: np.ndarray = np.asarray(distances[a, c], dtype=np.int64)
    d_bd: np.ndarray = np.asarray(distances[b, d], dtype=np.int64)
    gain: np.ndarray = d_ac + d_bd - edge_costs[i_idx] - edge_costs[j_idx]
    return gain

  n: int = tour.size
  is_prefix: np.ndarray = i_idx < 0
  is_suffix: np.ndarray = j_idx == n - 1
  safe_i: np.ndarray = np.maximum(i_idx, 0)
  safe_j: np.ndarray = np.minimum(j_idx, max(0, n - 2))
  a = tour[safe_i]
  b = np.where(is_prefix, tour[0], next_node[safe_i])
  c = tour[j_idx]
  d = next_node[safe_j]
  d_ac = np.where(is_prefix, np.int64(0), np.asarray(distances[a, c], dtype=np.int64))
  d_bd = np.where(is_suffix, np.int64(0), np.asarray(distances[b, d], dtype=np.int64))
  old_ab: np.ndarray = np.where(is_prefix, np.int64(0), edge_costs[safe_i])
  old_cd: np.ndarray = np.where(is_suffix, np.int64(0), edge_costs[safe_j])
  open_gain: np.ndarray = d_ac + d_bd - old_ab - old_cd
  return open_gain


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
    *,
    is_closed: bool = False,
    is_cyclic: bool = True,
    fixed_start: bool = True,
  ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield the `(i, j)` break-point pairs of one scan, in ascending scan order.

    Both neighbourhood flavours reduce to the same flat pair enumeration, so the
    three acceptance rules consume one generator instead of each carrying its own
    nested loop:

    - exhaustive (`candidates is None`): every pair matching the route topology, in
      blocks of `_PAIR_BLOCK` to bound temporaries;
    - candidate mode: exactly the pairs a sequential `for r in range(k)` visit
      would reach, including duplicates when two neighbours of `a` share a
      position — the acceptance rules are order- and multiplicity-sensitive, so
      deduplicating here would silently change the tour sequence.

    Args:
        tour: Current tour, `(n,)` node ids.
        positions: `inverse_positions(tour)` in candidate mode, `None` otherwise.
        candidates: `(n, k_eff)` nearest-neighbour table, or `None`.
        is_closed: True for depot-bounded `[0, ..., 0]` routes.
        is_cyclic: True for closed-loop tours with wrap-around edge `(n - 1, 0)`.
        fixed_start: True when `tour[0]` is fixed.

    Yields:
        `(i_idx, j_idx)` `int32` arrays aligned by position.
    """
    n: int = len(tour)
    if candidates is None or positions is None:
      i_all, j_all = _exhaustive_pairs(n, is_closed=is_closed, is_cyclic=is_cyclic, fixed_start=fixed_start)
      for start in range(0, i_all.size, _PAIR_BLOCK):
        yield (
          i_all[start : start + _PAIR_BLOCK],
          j_all[start : start + _PAIR_BLOCK],
        )
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
      j_mat = np.where(
        in_range,
        positions[np.clip(neighbours, 0, max(positions.size - 1, 0))],
        -1,
      )
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

  def _step_best(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
    *,
    is_closed: bool = False,
    is_cyclic: bool = True,
    fixed_start: bool = True,
  ) -> bool:
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
    for i_idx, j_idx in self._pair_blocks(
      tour,
      positions,
      candidates,
      is_closed=is_closed,
      is_cyclic=is_cyclic,
      fixed_start=fixed_start,
    ):
      if i_idx.size == 0:
        continue
      deltas: np.ndarray = _pair_deltas(
        tour,
        next_node,
        edge_costs,
        distances,
        i_idx,
        j_idx,
        is_closed=is_closed,
        is_cyclic=is_cyclic,
      )
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

  def _step_first(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
    *,
    is_closed: bool = False,
    is_cyclic: bool = True,
    fixed_start: bool = True,
  ) -> bool:
    """Apply the first improving move found in scan order; return whether one was applied."""
    n: int = len(tour)
    if n < 4:
      return False
    next_node, edge_costs, positions = self._scan_context(tour, distances, candidates)

    for i_idx, j_idx in self._pair_blocks(
      tour,
      positions,
      candidates,
      is_closed=is_closed,
      is_cyclic=is_cyclic,
      fixed_start=fixed_start,
    ):
      if i_idx.size == 0:
        continue
      improving: np.ndarray = (
        _pair_deltas(
          tour,
          next_node,
          edge_costs,
          distances,
          i_idx,
          j_idx,
          is_closed=is_closed,
          is_cyclic=is_cyclic,
        )
        < 0
      )
      if not bool(improving.any()):
        continue
      # `argmax` on a boolean mask returns the first `True` in scan order.
      pos: int = int(np.argmax(improving))
      i: int = int(i_idx[pos])
      j: int = int(j_idx[pos])
      tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
      return True
    return False

  def _step_multi_move(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
    *,
    is_closed: bool = False,
    is_cyclic: bool = True,
    fixed_start: bool = True,
  ) -> bool:
    """Apply every mutually disjoint improving move of one scan; return whether any was applied.

    The disjoint selection is inherently sequential — each accepted reversal removes
    its window from the candidates — so only that part stays a Python loop, over the
    improving moves of the pass rather than over all `O(n^2)` pairs. A boolean
    occupancy vector replaces the quadratic `any()` overlap check: two windows
    conflict exactly when their inclusive index ranges intersect, which is what
    marking `[max(0, i), j]` and testing the slice reports.
    """
    n: int = len(tour)
    if n < 4:
      return False
    next_node, edge_costs, positions = self._scan_context(tour, distances, candidates)

    gains: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for i_idx, j_idx in self._pair_blocks(
      tour,
      positions,
      candidates,
      is_closed=is_closed,
      is_cyclic=is_cyclic,
      fixed_start=fixed_start,
    ):
      if i_idx.size == 0:
        continue
      deltas: np.ndarray = _pair_deltas(
        tour,
        next_node,
        edge_costs,
        distances,
        i_idx,
        j_idx,
        is_closed=is_closed,
        is_cyclic=is_cyclic,
      )
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
      occ_start: int = max(0, i)
      if occupied[occ_start : j + 1].any():
        continue
      tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
      occupied[occ_start : j + 1] = True
      applied = True
    return applied

  def _refine(
    self,
    tour: np.ndarray,
    distances: np.ndarray,
    candidates: np.ndarray | None,
    *,
    is_closed: bool = False,
    is_cyclic: bool = True,
    fixed_start: bool = True,
  ) -> np.ndarray:
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
      if not step(
        improved,
        distances,
        candidates,
        is_closed=is_closed,
        is_cyclic=is_cyclic,
        fixed_start=fixed_start,
      ):
        break
    return improved

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` using CPU 2-opt."""
    distances = context.get_cpu_distances()
    route_np = np.asarray(route, dtype=np.int32)
    is_closed, is_cyclic, fixed_start = _resolve_topology(context)
    return self._refine(
      route_np,
      distances,
      _resolve_candidates(context, route_np, self.candidate_k),
      is_closed=is_closed,
      is_cyclic=is_cyclic,
      fixed_start=fixed_start,
    )

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` using CPU 2-opt."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()
    candidates = _resolve_candidates(context, batch_np, self.candidate_k)
    is_closed, is_cyclic, fixed_start = _resolve_topology(context)
    improved: np.ndarray = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(len(batch_np)):
      improved[idx] = self._refine(
        batch_np[idx],
        distances,
        candidates,
        is_closed=is_closed,
        is_cyclic=is_cyclic,
        fixed_start=fixed_start,
      )
    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)


class LocalSearchGPUTwoOpt(BaseLocalSearchGPU):
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
    super().__init__(
      threads_per_block=threads_per_block,
      max_matrix_dimension=max_matrix_dimension,
      candidate_k=candidate_k,
    )
    if mode not in _GPU_MODES:
      raise ValueError(f"Invalid 2-opt GPU mode {mode!r}; expected one of {sorted(_GPU_MODES)}")
    if max_iterations < 0:
      raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    self.mode: TwoOptGPUMode = mode
    self.max_iterations: int = max_iterations
    self.max_shared_tour_cities: int = max_shared_tour_cities

  def _get_kernel(self) -> Any:
    """Lazily compile and return the CUDA kernel selected by `mode`."""
    if self._kernel is None:
      source, entry_point = _GPU_KERNELS[self.mode]
      self._kernel = load_kernel(source, entry_point, __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile the selected kernel and launch it once on a valid permutation (`BUG-24`, `BUG-25`)."""
    super()._warmup_impl(context)
    self.improve_batch(self._warmup_batch(context, rows=1), context)

  def _tile_size(self, n: int) -> int:
    """Return 0 for the full-tour path, else the shared-memory tile width."""
    return 0 if n <= self.max_shared_tour_cities else _TILE_SIZE

  def _uses_candidates(self, context: ProblemContext, n: int) -> bool:
    """Return whether the kernel candidate path is admissible for this batch.

    The kernel builds the inverse map `pos[tour[idx]] = idx` in a `(batch, n)` buffer
    and indexes it by node id, so it needs a dense `0..n-1` cyclic permutation.
    Closed depot-bounded routes, non-cyclic open paths, VRP variants, and batches
    whose route width differs from `context.dimension` keep the exhaustive scan
    (`BUG-107`).

    The decision is taken from host metadata only — topology, `problem_type`,
    `dimension` and the batch width — so the hot path never calls `int(tour.max())`
    on a GPU array, which used to force one D2H synchronization per generation (`BUG-27`).
    """
    if self.candidate_k is None or self.candidate_k <= 0:
      return False
    if self._tile_size(n) > 0:
      # Tiling and candidate mode are mutually exclusive: a tiled block already
      # streams the tour through shared memory, so the `positions` scratch would
      # compete with the window for the 48 KB budget for no throughput gain.
      return False
    is_closed, is_cyclic, _ = _resolve_topology(context)
    if is_closed or not is_cyclic:
      return False
    if str(context.problem.problem_type) in {"CVRP", "OVRP", "DCVRP"}:
      return False
    return n == context.dimension

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
    dist_stride: int = int(distances_cpu.shape[1])
    distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
    self.h2d_bytes += int(distances_cpu.nbytes)

    is_closed, is_cyclic, fixed_start = _resolve_topology(context)

    # Candidate-list mode (`DP10-20`, `BUG-57`): bind the same context-owned `N_k` table
    # and charge the same upload as `_improve_grid`, so `naive` searches the same
    # `Theta(n * k)` neighbourhood as the grid kernels for a given `candidate_k`.
    candidate_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n):
      d_candidates = self._resolve_candidate_table(context, int(self.candidate_k or 0))
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
          (
            d_tour,
            distances_gpu,
            np.int32(n),
            np.int32(dist_stride),
            np.int32(1 if is_closed else 0),
            np.int32(1 if is_cyclic else 0),
            np.int32(1 if fixed_start else 0),
            np.int32(0),
            d_candidates,
            np.int32(candidate_count),
            d_positions,
          ),
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

    is_closed, is_cyclic, fixed_start = _resolve_topology(context)
    d_distances, d_coords, edge_type_code = self._resolve_distance_operands(context, int(context.dimension))
    dist_stride: int = int(context.dimension if isinstance(d_distances, int) else d_distances.shape[1])
    block_size: int = prev_power_of_2(min(self.threads_per_block, max(1, n - 2)), min_val=1, max_val=256)
    tile_size: int = self._tile_size(n)

    # Candidate-list mode (`DP10-20`): bind the context-owned `N_k` once and let each
    # block build its inverse tour map in its own `positions` slice. The table lives in
    # VRAM across generations; only the first bind charges H2D bytes (`BUG-27`).
    candidate_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n):
      d_candidates = self._resolve_candidate_table(context, int(self.candidate_k or 0))
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
          np.int32(dist_stride),
          np.int32(batch_size),
          np.int32(edge_type_code),
          np.int32(1 if is_closed else 0),
          np.int32(1 if is_cyclic else 0),
          np.int32(1 if fixed_start else 0),
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
