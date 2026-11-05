"""Capacity contract for the multi-move GPU 2-opt kernel (`BUG-20`, `DP11-2`).

`two_opt_multi_move.cu` used to stop after 16 disjoint moves because thread 0 kept the
applied intervals in two fixed `int[16]` arrays. On a tour carrying more than 16
independent crossings the cap silently diverged from the CPU reference, which needs only
one pass per disjoint set.

The fixture makes the 20 planted moves provably the steepest ones. On a line metric
(`d(x, y) = |x - y|`) the identity tour is optimal; reversing a length-`L` window
`[p, p + L - 1]` of it breaks the edges `(p - 1, p)` and `(p + L - 1, p + L)` and
reconnecting them costs

$$
\\Delta = 2 - 2L ,
$$

i.e. `-6` for `L = 4`, while every incidental cyclic shortcut through the wrap-around
edge saves only `-2`. The greedy steepest-descent pass must therefore select all 20
planted moves first, so a single `max_iterations=1` pass restores all of them.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from algorithms.strategies.local_search import LocalSearchCPUTwoOpt, LocalSearchGPUTwoOpt
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext

cp = cupy_backend()

#: Number of independent crossings planted in the tour; 17 is already past the old cap.
_CROSSINGS: int = 20
#: Width of each reversed window; `L = 4` makes the undo worth `2 - 2L = -6`.
_SEGMENT: int = 4
#: Distance between window starts, leaving one free edge between consecutive windows.
_STRIDE: int = 5
#: First window start; positions `0` and `1` stay untouched.
_FIRST_START: int = 2
#: Last window is `[97, 100]`, so 105 cities leave its `j + 1` neighbour in range.
_TOUR_LENGTH: int = 105


def _line_problem(n: int) -> Problem:
  """Return an EXPLICIT TSP whose cost is the 1-D distance `|x - y|`."""
  axis = np.arange(n, dtype=np.int64)
  distances = np.abs(axis[:, None] - axis[None, :])
  return Problem(
    name=f"line{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EXPLICIT",
    coordinates=None,
    distances=distances,
    capacity=None,
    demands=None,
  )


def _window_starts() -> list[int]:
  """Return the 20 window starts, one free edge apart."""
  return [_FIRST_START + _STRIDE * k for k in range(_CROSSINGS)]


def _crossed_tour(n: int) -> np.ndarray:
  """Return the sorted tour with `_CROSSINGS` disjoint length-`_SEGMENT` windows reversed."""
  tour = np.arange(n, dtype=np.int32)
  for start in _window_starts():
    tour[start : start + _SEGMENT] = tour[start : start + _SEGMENT][::-1]
  return tour


def _cupy() -> Any:
  """Return the importable CuPy module, skipping the test when it is unavailable."""
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


def _restored_windows(tour: np.ndarray) -> int:
  """Count how many planted windows are back in ascending order."""
  return sum(
    1
    for start in _window_starts()
    if np.array_equal(tour[start : start + _SEGMENT], np.arange(start, start + _SEGMENT, dtype=np.int32))
  )


def test_planted_tour_has_twenty_independent_crossings() -> None:
  """WHAT: the fixture tour itself, checked before blaming the kernel.

  WHY: if the planted windows shared an edge the pass could legitimately apply fewer
  moves, and the kernel assertion would be untestable.

  EXPECTED: 20 descending windows, each separated from the next by at least one
  ascending position.
  """
  tour = _crossed_tour(_TOUR_LENGTH)
  windows = _window_starts()
  reversed_windows = [
    start
    for start in windows
    if np.array_equal(
      tour[start : start + _SEGMENT],
      np.arange(start + _SEGMENT - 1, start - 1, -1, dtype=np.int32),
    )
  ]

  assert reversed_windows == windows
  assert all(later - earlier > _SEGMENT for earlier, later in pairwise(windows)), "windows must not share an edge"
  assert _restored_windows(tour) == 0, "the fixture must start with every window reversed"


@pytest.mark.gpu
def test_multi_move_applies_more_than_sixteen_disjoint_moves_in_one_pass() -> None:
  """WHAT: `LocalSearchGPUTwoOpt(max_iterations=1, mode="multi_move")` on 20 crossings.

  WHY: the fixed `int applied_start[16]` / `pass < 16` cap (`BUG-20`) left the remaining
  crossings for a later pass, so a single-pass comparison against
  `LocalSearchCPUTwoOpt(mode="multi_move")` disagreed beyond 16 moves.

  EXPECTED: every planted window is restored in one pass -- 20 > 16 -- the tour becomes
  the sorted optimum, and the CPU multi-move reference agrees exactly.
  """
  # Arrange
  cupy = _cupy()
  problem = _line_problem(_TOUR_LENGTH)
  context = ProblemContext(problem, xp=cupy, seed=0)
  batch = _crossed_tour(_TOUR_LENGTH).reshape(1, _TOUR_LENGTH)
  expected = np.arange(_TOUR_LENGTH, dtype=np.int32)

  # Act
  improved = LocalSearchGPUTwoOpt(max_iterations=1, mode="multi_move").improve_batch(batch, context)
  cpu_improved = LocalSearchCPUTwoOpt(max_iterations=1, mode="multi_move").improve_batch(batch, context)

  # Assert
  gpu_tour = np.asarray(improved)[0]
  restored = _restored_windows(gpu_tour)
  assert restored > 16, f"single pass restored only {restored} of {_CROSSINGS} disjoint windows"
  assert restored == _CROSSINGS, "every disjoint window must be restored in a single pass"
  np.testing.assert_array_equal(gpu_tour, expected)
  np.testing.assert_array_equal(np.asarray(cpu_improved)[0], expected)
