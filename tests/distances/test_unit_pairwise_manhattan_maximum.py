"""Unit tests for the Manhattan and Chebyshev pairwise distance functions.

``MAN_2D``/``MAN_3D`` round once after summing the component deltas, while ``MAX_2D``/
``MAX_3D`` round each delta and then take the maximum; both orders are pinned exactly
because they are indistinguishable on integer-only fixtures.
"""

from __future__ import annotations

import pytest

from distances.pairwise import (
  compute_manhattan_2d,
  compute_manhattan_3d,
  compute_maximum_2d,
  compute_maximum_3d,
)


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0), (3.0, 4.0), 7),
    ((1.5, 2.5), (4.5, 6.5), 7),
    ((0.0, 0.0), (3.5, 4.25), 8),  # 7.75 + 0.5 = 8.25 -> 8
  ],
)
def test_manhattan_2d_values(coord1: tuple[float, float], coord2: tuple[float, float], expected: int) -> None:
  """WHAT: MAN_2D is ``nint(|dx| + |dy|)``, rounding once after the sum.

  WHY: rounding each component first changes the total (e.g. 3.5 + 4.25 -> 4 + 4 = 8 here,
  but 0.4 + 0.4 would give 0 + 0 instead of 1).

  EXPECTED: the three hand-checked integers above.
  """
  assert compute_manhattan_2d(coord1, coord2) == expected


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3),
    ((1.0, 2.0, 3.0), (4.0, 6.0, 8.0), 12),
  ],
)
def test_manhattan_3d_values(
  coord1: tuple[float, float, float], coord2: tuple[float, float, float], expected: int
) -> None:
  """WHAT: MAN_3D is ``nint(|dx| + |dy| + |dz|)``.

  WHY: the third term is easy to omit when copying the 2D implementation.

  EXPECTED: 3 for the unit cube diagonal and 12 for (1,2,3)-(4,6,8).
  """
  assert compute_manhattan_3d(coord1, coord2) == expected


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0), (3.0, 4.0), 4),
    ((1.5, 2.5), (4.5, 6.5), 4),
    ((0.0, 0.0), (3.75, 1.2), 4),
  ],
)
def test_maximum_2d_values(coord1: tuple[float, float], coord2: tuple[float, float], expected: int) -> None:
  """WHAT: MAX_2D (Chebyshev) is the larger of the two rounded component deltas.

  WHY: ``nint`` is monotonic, so ``max(nint(dx), nint(dy))`` must equal
  ``nint(max(dx, dy))`` — a mismatch would reveal a broken rounding order.

  EXPECTED: 4, 4 and 4 for the three pairs above.
  """
  assert compute_maximum_2d(coord1, coord2) == expected


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 3),
    ((1.0, 2.0, 3.0), (4.0, 6.0, 8.0), 5),
  ],
)
def test_maximum_3d_values(
  coord1: tuple[float, float, float], coord2: tuple[float, float, float], expected: int
) -> None:
  """WHAT: MAX_3D takes the maximum over all three rounded component deltas.

  WHY: an omitted z component returns 4 instead of 5 for (1,2,3)-(4,6,8).

  EXPECTED: 3 and 5.
  """
  assert compute_maximum_3d(coord1, coord2) == expected
