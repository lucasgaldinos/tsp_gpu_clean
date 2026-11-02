"""Unit tests for the Euclidean pairwise distance functions in :mod:`distances.pairwise`.

``EUC_2D`` and ``EUC_3D`` share the TSPLIB ``nint(x) = int(x + 0.5)`` rounding rule; a
one-unit change there shifts canonical tour lengths (berlin52 = 7542) for the whole family,
so the values are pinned exactly.
"""

from __future__ import annotations

import pytest

from distances.pairwise import compute_euclidean_2d, compute_euclidean_3d


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0), (3.0, 4.0), 5),  # exact 5.0 -> 5
    ((1.5, 2.5), (4.5, 6.5), 5),  # exact 5.0 -> 5
    ((0.0, 0.0), (2.4, 0.0), 2),  # 2.4 + 0.5 = 2.9 -> 2 (rounds down)
    ((0.0, 0.0), (2.5, 0.0), 3),  # 2.5 + 0.5 = 3.0 -> 3 (tie rounds up)
    ((0.0, 0.0), (0.5, 0.0), 1),  # 0.5 + 0.5 = 1.0 -> 1 (tie rounds up)
    ((0.0, 0.0), (0.49, 0.0), 0),  # 0.49 + 0.5 = 0.99 -> 0 (just below the tie)
    ((0.0, 0.0), (3.1, 4.0), 5),  # nint(5.0606) = 5, not 6
  ],
)
def test_euclidean_2d_uses_tsplib_nint_rounding(
  coord1: tuple[float, float], coord2: tuple[float, float], expected: int
) -> None:
  """WHAT: EUC_2D rounds with TSPLIB ``nint(x) = int(x + 0.5)``.

  WHY: Python's ``round()`` uses banker's rounding (``round(2.5) == 2``) and ``ceil``
  overshoots; either would break canonical tour lengths such as berlin52 = 7542.

  EXPECTED: each pair yields the stated integer, with ties (2.5, 0.5) rounding up.
  """
  assert compute_euclidean_2d(coord1, coord2) == expected


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2),  # sqrt(3) = 1.732 -> 2
    ((1.0, 2.0, 3.0), (4.0, 6.0, 8.0), 7),  # sqrt(50) = 7.071 -> 7
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1),  # unit edge stays 1
  ],
)
def test_euclidean_3d_values(
  coord1: tuple[float, float, float], coord2: tuple[float, float, float], expected: int
) -> None:
  """WHAT: EUC_3D sums three squared deltas before the square root.

  WHY: dropping the z term (a copy-paste from EUC_2D) is invisible in 2D-only fixtures
  and silently under-prices every 3D instance.

  EXPECTED: the three hand-checked integers above.
  """
  assert compute_euclidean_3d(coord1, coord2) == expected
