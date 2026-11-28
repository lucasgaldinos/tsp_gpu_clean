"""Vectorised neighbourhood scan for `LocalSearchCPUTourTwoOpt` (`TASK 13.14`, `BUG-63`).

The CPU variant enumerates its 2-opt neighbourhood with NumPy and feeds the same flat
pair list to the `best`, `first` and `multi_move` acceptance rules.

WHY: the CPU variant is the baseline of every reported CPU/GPU speedup, so a vectorised
scan that resolved a tie differently would change the tour sequence of the whole GA and
silently re-baseline the dissertation's central comparison. The scalar reference below is
the pre-vectorisation nested loop; parity against it is the contract.

EXPECTED: identical final tours, identical move counts, and a pair enumeration that
reproduces the sequential visit order exactly — including the duplicated pair a repeat
neighbour produces, because the acceptance rules are multiplicity-sensitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies import LocalSearchCPUTourTwoOpt
from algorithms.strategies.local_search.two_opt import _exhaustive_pairs
from data_models.problem import Problem
from protocols import ProblemContext
from utils.route_helpers import inverse_positions

if TYPE_CHECKING:
  from algorithms.strategies.local_search import TwoOptCPUMode

CPU_TWO_OPT_MODES: list[TwoOptCPUMode] = ["best", "first", "multi_move"]

#: Tour lengths spanning the `n == 4` minimum, the odd/even wrap, and a realistic size.
PAIRITY_TOUR_LENGTHS: list[int] = [4, 5, 6, 8, 10, 17, 40]


def _scalar_step_best(tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
  """Reference `best` rule: nested loops, strict `<`, first pair wins a tie."""
  n = len(tour)
  best_delta = 0
  best_i = -1
  best_j = -1

  if candidates is None:
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      c = int(tour[i + 2])
      limit_j = n - 1 if i == 0 else n
      for j in range(i + 2, limit_j):
        d = int(tour[(j + 1) % n])
        delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta < best_delta:
          best_delta = delta
          best_i = i
          best_j = j
        c = d
  else:
    pos = inverse_positions(tour)
    neighbors = int(candidates.shape[1])
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      limit_j = n - 1 if i == 0 else n
      for r in range(neighbors):
        neighbor = int(candidates[a, r])
        if neighbor < 0 or neighbor >= pos.size:
          continue
        j = int(pos[neighbor])
        if j <= i + 1 or j >= limit_j:
          continue
        c = int(tour[j])
        d = int(tour[(j + 1) % n])
        delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta < best_delta:
          best_delta = delta
          best_i = i
          best_j = j

  if best_delta < 0:
    tour[best_i + 1 : best_j + 1] = tour[best_i + 1 : best_j + 1][::-1]
    return True
  return False


def _scalar_step_first(tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
  """Reference `first` rule: the first improving pair in scan order wins."""
  n = len(tour)
  if candidates is None:
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      c = int(tour[i + 2])
      limit_j = n - 1 if i == 0 else n
      for j in range(i + 2, limit_j):
        d = int(tour[(j + 1) % n])
        delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta < 0:
          tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
          return True
        c = d
    return False

  pos = inverse_positions(tour)
  neighbors = int(candidates.shape[1])
  for i in range(n - 2):
    a = int(tour[i])
    b = int(tour[i + 1])
    d_ab = int(distances[a, b])
    limit_j = n - 1 if i == 0 else n
    for r in range(neighbors):
      neighbor = int(candidates[a, r])
      if neighbor < 0 or neighbor >= pos.size:
        continue
      j = int(pos[neighbor])
      if j <= i + 1 or j >= limit_j:
        continue
      c = int(tour[j])
      d = int(tour[(j + 1) % n])
      delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
      if delta < 0:
        tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
        return True
  return False


def _scalar_step_multi_move(tour: np.ndarray, distances: np.ndarray, candidates: np.ndarray | None) -> bool:
  """Reference `multi_move` rule: steepest-first greedy over disjoint windows."""
  n = len(tour)
  moves: list[tuple[int, int, int]] = []

  if candidates is None:
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      c = int(tour[i + 2])
      limit_j = n - 1 if i == 0 else n
      for j in range(i + 2, limit_j):
        d = int(tour[(j + 1) % n])
        delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta < 0:
          moves.append((delta, i, j))
        c = d
  else:
    pos = inverse_positions(tour)
    neighbors = int(candidates.shape[1])
    for i in range(n - 2):
      a = int(tour[i])
      b = int(tour[i + 1])
      d_ab = int(distances[a, b])
      limit_j = n - 1 if i == 0 else n
      for r in range(neighbors):
        neighbor = int(candidates[a, r])
        if neighbor < 0 or neighbor >= pos.size:
          continue
        j = int(pos[neighbor])
        if j <= i + 1 or j >= limit_j:
          continue
        c = int(tour[j])
        d = int(tour[(j + 1) % n])
        delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta < 0:
          moves.append((delta, i, j))

  if not moves:
    return False

  moves.sort(key=lambda move: move[0])
  applied: list[tuple[int, int]] = []
  for _delta, i, j in moves:
    if any(not (j < start or i > end) for start, end in applied):
      continue
    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
    applied.append((i, j))
  return bool(applied)


_SCALAR_STEPS = {
  "best": _scalar_step_best,
  "first": _scalar_step_first,
  "multi_move": _scalar_step_multi_move,
}


def _scalar_refine(
  tour: np.ndarray,
  distances: np.ndarray,
  candidates: np.ndarray | None,
  mode: TwoOptCPUMode,
  iterations: int,
) -> np.ndarray:
  """Reference `_refine`: run the scalar acceptance rule up to `iterations` times."""
  improved = tour.copy()
  if len(tour) < 4 or iterations <= 0:
    return improved
  for _ in range(iterations):
    if not _SCALAR_STEPS[mode](improved, distances, candidates):
      break
  return improved


def _rounded_matrix(n: int, seed: int) -> np.ndarray:
  """Return a symmetric `int32` EUC_2D-shaped matrix, mirroring `compute_distance_matrix`."""
  rng = np.random.default_rng(seed)
  coords = rng.random((n, 2)) * 1_000.0
  diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
  distances = np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5).astype(np.int32)
  np.fill_diagonal(distances, 0)
  return distances


def _nearest(distances: np.ndarray, k: int) -> np.ndarray:
  """Return the `(n, k_eff)` `N_k` table, self excluded, ascending by distance."""
  n = distances.shape[0]
  k_eff = min(k, n - 1)
  order: np.ndarray = np.argsort(distances, axis=1, kind="stable")[:, 1 : k_eff + 1]
  return np.asarray(order, dtype=np.int32)


def _lattice_matrix(n: int) -> np.ndarray:
  """Return a rounded `int32` matrix over an integer lattice.

  Integer lattice coordinates collapse many intercity distances to the same value, so
  several `(i, j)` pairs share the minimum gain on the same pass. That is the only input
  class where the `best` tie-break and the `multi_move` stable sort are observable;
  floating Euclidean coordinates produced no tied minimum in 230 sampled scans.
  """
  side = int(np.ceil(np.sqrt(n)))
  coords = np.array([[x, y] for x in range(side) for y in range(side)][:n], dtype=float)
  diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
  distances: np.ndarray = np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5).astype(np.int32)
  np.fill_diagonal(distances, 0)
  return distances


def _sequential_pairs(n: int) -> list[tuple[int, int]]:
  """Return the pairs a nested sequential 2-opt scan visits, in visit order."""
  pairs: list[tuple[int, int]] = []
  for i in range(n - 2):
    limit_j = n - 1 if i == 0 else n
    for j in range(i + 2, limit_j):
      pairs.append((i, j))
  return pairs


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
@pytest.mark.parametrize("candidate_k", [None, 3, 20])
@pytest.mark.parametrize("n", PAIRITY_TOUR_LENGTHS)
def test_vectorised_refine_matches_scalar_reference(
  mode: TwoOptCPUMode, candidate_k: int | None, n: int
) -> None:
  """WHAT: `_refine` against the pre-vectorisation scan for one (mode, neighbourhood, n).

  WHY: the acceptance rules read a flat pair array now, so tie-breaking and candidate
  visit order are the two places a rewrite can silently change which move is applied.

  EXPECTED: bit-identical tours after the same number of passes.
  """
  distances = _rounded_matrix(n, seed=n)
  rng = np.random.default_rng(n * 100 + (candidate_k or 0))
  candidates = None if candidate_k is None else _nearest(distances, candidate_k)

  for trial in range(4):
    start = rng.permutation(n).astype(np.int32)
    for iterations in (1, 3, 10):
      expected = _scalar_refine(start, distances, candidates, mode, iterations)
      actual = LocalSearchCPUTourTwoOpt(
        max_iterations=iterations, mode=mode, candidate_k=candidate_k
      )._refine(start.copy(), distances, candidates)
      assert np.array_equal(actual, expected), (
        f"{mode} diverged from the scalar scan at n={n}, k={candidate_k}, "
        f"iterations={iterations}, trial={trial}"
      )


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
@pytest.mark.parametrize("candidate_k", [None, 3])
@pytest.mark.parametrize("n", [9, 16, 25])
def test_vectorised_refine_breaks_gain_ties_like_the_scalar_scan(
  mode: TwoOptCPUMode, candidate_k: int | None, n: int
) -> None:
  """WHAT: `_refine` on an integer lattice, where many pairs share the minimum gain.

  WHY: `np.argmin` keeps the first minimum and `np.argsort(..., kind="stable")` keeps scan
  order among equal gains, which is what a strict `<` scan and a stable `list.sort` do. On
  floating Euclidean coordinates no scan in 230 samples had a tied minimum, so that
  equivalence needs an input that actually produces ties to be tested at all.

  EXPECTED: bit-identical tours for every acceptance rule.
  """
  distances = _lattice_matrix(n)
  candidates = None if candidate_k is None else _nearest(distances, candidate_k)

  for trial in range(4):
    start = np.random.default_rng(n * 31 + trial).permutation(n).astype(np.int32)
    expected = _scalar_refine(start, distances, candidates, mode, 10)
    actual = LocalSearchCPUTourTwoOpt(
      max_iterations=10, mode=mode, candidate_k=candidate_k
    )._refine(start.copy(), distances, candidates)
    assert np.array_equal(actual, expected), (
      f"{mode} resolved a gain tie differently from the scalar scan at n={n}, "
      f"k={candidate_k}, trial={trial}"
    )


@pytest.mark.parametrize("n", [4, 5, 6, 7, 17, 40])
def test_exhaustive_pairs_reproduce_the_sequential_visit_order(n: int) -> None:
  """WHAT: The `(i, j)` pairs `_exhaustive_pairs` returns for a tour length.

  WHY: the three acceptance rules consume this enumeration instead of their own loop, so
  its order and its single exclusion are load-bearing: `(0, n - 1)` reverses `tour[1:n]`,
  which is the identity move.

  EXPECTED: exactly the sequential pair list, in the same order.
  """
  i_idx, j_idx = _exhaustive_pairs(n)
  assert list(zip(i_idx.tolist(), j_idx.tolist(), strict=True)) == _sequential_pairs(n)
  assert (0, n - 1) not in set(zip(i_idx.tolist(), j_idx.tolist(), strict=True)), (
    "the wrap-around pair reverses the whole tail and must stay excluded"
  )


def test_candidate_pairs_keep_duplicates_from_repeated_neighbours() -> None:
  """WHAT: `_pair_blocks` in candidate mode when two entries of a row repeat a node.

  WHY: `first` and `multi_move` consume pairs in visit order, so collapsing a duplicate
  would let a later, worse move win a slot the earlier one owns. The kernel's `N_k` table
  can repeat a node after `candidate_k` clamping, and the guards must not deduplicate.

  EXPECTED: the repeated node contributes two identical `(i, j)` entries.
  """
  n = 8
  tour = np.arange(n, dtype=np.int32)
  positions = inverse_positions(tour)
  candidates = np.tile(np.array([3, 5], dtype=np.int32), (n, 1))
  candidates[0] = np.array([3, 3], dtype=np.int32)

  solver = LocalSearchCPUTourTwoOpt(candidate_k=2)
  pairs = list(solver._pair_blocks(tour, positions, candidates))
  i_all = np.concatenate([block[0] for block in pairs])
  j_all = np.concatenate([block[1] for block in pairs])

  assert int(np.count_nonzero((i_all == 0) & (j_all == 3))) == 2, (
    "a neighbour repeated in the same row must yield the pair twice"
  )


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_improve_matches_scalar_reference_through_the_public_api(mode: TwoOptCPUMode) -> None:
  """WHAT: `improve` on an `EXPLICIT` problem against the scalar reference.

  WHY: the parity above covers `_refine` directly; this covers the wiring that resolves the
  distance matrix and the candidate table, where a transposed or clipped lookup would hide.

  EXPECTED: bit-identical tour and a cost no worse than the input.
  """
  n = 12
  distances = _rounded_matrix(n, seed=909)
  problem = Problem(
    name="explicit12",
    dimension=n,
    problem_type="TSP",
    edge_type="EXPLICIT",
    coordinates=None,
    distances=distances,
    capacity=None,
    demands=None,
  )
  context = ProblemContext(problem, xp=np, seed=42)
  start = np.random.default_rng(3).permutation(n).astype(np.int32)

  expected = _scalar_refine(start, distances, None, mode, 10)
  actual = LocalSearchCPUTourTwoOpt(max_iterations=10, mode=mode).improve(start, context=context)

  assert np.array_equal(actual, expected)
  assert sorted(actual.tolist()) == list(range(n)), "improvement must preserve the permutation"


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_sparse_route_ids_beyond_the_route_length_stay_in_bounds(mode: TwoOptCPUMode) -> None:
  """WHAT: A CVRP-shaped route whose node ids exceed `len(route)`.

  WHY: `inverse_positions` sizes the lookup by the largest node id, so `pos.size` can be
  smaller than the candidate table's node space; the vectorised gather must clamp instead
  of indexing out of bounds (`BUG-19`).

  EXPECTED: no exception, and the result is a permutation of the input route.
  """
  n = 10
  distances = _rounded_matrix(n, seed=17)
  route = np.array([0, 9, 3, 7, 1, 5], dtype=np.int32)
  positions = inverse_positions(route)
  candidates = _nearest(distances, 3)

  solver = LocalSearchCPUTourTwoOpt(max_iterations=10, mode=mode, candidate_k=3)
  improved = solver._refine(route.copy(), distances, candidates)

  assert sorted(improved.tolist()) == sorted(route.tolist())
  assert positions.size == n, "the lookup is sized by the largest node id, not the route length"
