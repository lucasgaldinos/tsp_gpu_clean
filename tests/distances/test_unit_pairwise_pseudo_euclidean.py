"""Unit tests for the pseudo-Euclidean pairwise distance functions: ``ATT`` and ``CEIL_2D``.

Both families scale or round a Euclidean distance, and both are easy to confuse: ``CEIL_2D``
adds no unit, while ``ATT`` adds one only when ``nint`` rounded down. Each rule is pinned
exactly, plus an invariant over the shared deterministic point grid.
"""

from __future__ import annotations

import math

import pytest

from distances.pairwise import compute_att, compute_ceiling_2d, compute_euclidean_2d

# Deterministic point grid used by the invariant tests (no randomness: runs are comparable).
PAIRWISE_GRID: tuple[tuple[float, float], ...] = (
  (0.0, 0.0),
  (1.0, 0.0),
  (1.0, 3.0),
  (3.0, 4.0),
  (3.1, 4.0),
  (12.0, 16.0),
  (-2.5, 7.25),
)


def _att_intermediates(coord1: tuple[float, float], coord2: tuple[float, float]) -> tuple[float, int]:
  """Return ``(r_ij, t_ij)`` for the ATT formula, as defined by TSPLIB95."""
  rij = math.sqrt(((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) / 10.0)
  return rij, int(rij + 0.5)


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0), (3.0, 4.0), 5),  # exact integer distance is NOT bumped to 6
    ((0.0, 0.0), (3.1, 4.0), 6),  # 5.0606 -> 6
    ((1.5, 2.5), (4.4, 6.4), 5),  # sqrt(23.62) = 4.860 -> 5
  ],
)
def test_ceiling_2d_rounds_up_without_adding_one(
  coord1: tuple[float, float], coord2: tuple[float, float], expected: int
) -> None:
  """WHAT: CEIL_2D is ``ceil(sqrt(dx^2 + dy^2))`` with no extra unit.

  WHY: confusing CEIL_2D with ATT's ``+1`` rule inflates every edge by one unit when the
  Euclidean distance is already an integer.

  EXPECTED: 5 for the exact 3-4-5 triangle and 6 for a distance of 5.0606.
  """
  assert compute_ceiling_2d(coord1, coord2) == expected


def test_ceiling_2d_is_never_below_euclidean_2d() -> None:
  """WHAT: ``ceiling_2d >= euclidean_2d`` and the gap never exceeds 1 for every grid pair.

  WHY: CEIL_2D exists to over-estimate; returning less than the nint distance would violate
  the metric's purpose and invalidate lower bounds built on it.

  EXPECTED: all 28 grid pairs (28 = 7 points including self-pairs) satisfy ``0 <= ceil - euc <= 1``.
  """
  violations = []
  for index, first in enumerate(PAIRWISE_GRID):
    for second in PAIRWISE_GRID[index:]:
      gap = compute_ceiling_2d(first, second) - compute_euclidean_2d(first, second)
      if not 0 <= gap <= 1:
        violations.append((first, second, gap))
  assert violations == [], f"ceiling below nint or gap > 1 for: {violations}"


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0), (1.0, 3.0), 1),  # r_ij = 1.0 exactly, t_ij = r_ij -> no +1
    ((0.0, 0.0), (1.0, 0.0), 1),  # r_ij = 0.3162, t_ij = 0 < r_ij -> +1
    ((0.0, 0.0), (3.0, 4.0), 2),  # r_ij = 1.5811, t_ij = 2 >= r_ij -> no +1
    ((0.0, 0.0), (10.0, 0.0), 4),  # r_ij = 3.1623, t_ij = 3 < r_ij -> +1
    ((0.0, 0.0), (12.0, 16.0), 7),  # r_ij = 6.3246, t_ij = 6 < r_ij -> +1
    ((6734.0, 1453.0), (2233.0, 10.0), 1495),  # att48 cities 1-2, r_ij = 1494.6990
  ],
)
def test_att_applies_plus_one_only_when_rounding_down(
  coord1: tuple[float, float], coord2: tuple[float, float], expected: int
) -> None:
  """WHAT: ATT returns ``t_ij + 1`` only when ``t_ij < r_ij``, where ``r_ij = sqrt(d^2 / 10)``.

  WHY: the conditional ``+1`` is what makes att48 = 10628 and att532 = 27686 reproducible;
  applying it unconditionally (or never) shifts whole-family tour lengths.

  EXPECTED: the exact tie at ``r_ij = 1.0`` stays 1 while ``r_ij = 3.1623`` becomes 4.
  """
  assert compute_att(coord1, coord2) == expected


def test_att_result_is_the_ceiling_of_the_scaled_distance() -> None:
  """WHAT: for every grid pair, ``t_ij <= d_ij <= t_ij + 1`` and ``d_ij >= r_ij``.

  WHY: expresses the TSPLIB95 rule as an invariant, so an off-by-one introduced in the
  ``where`` branch is caught even for pairs nobody hand-computed.

  EXPECTED: no violation across the 28 unordered grid pairs.
  """
  violations = []
  for index, first in enumerate(PAIRWISE_GRID):
    for second in PAIRWISE_GRID[index:]:
      rij, tij = _att_intermediates(first, second)
      dij = compute_att(first, second)
      if not (tij <= dij <= tij + 1 and dij >= rij):
        violations.append((first, second, rij, tij, dij))
  assert violations == [], f"ATT bound violations (a, b, r_ij, t_ij, d_ij): {violations}"
