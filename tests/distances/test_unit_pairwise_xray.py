"""Unit tests for the ``XRAY1`` pairwise distance function.

``XRAY1`` combines a circular ``PHI`` distance with a max-over-angles and a factor-of-100
scale; the wrap-around and the scale are pinned exactly because either mistake misprices
every crystallography instance.
"""

from __future__ import annotations

import pytest

from distances.pairwise import compute_xray1


@pytest.mark.parametrize(
  ("coord1", "coord2", "expected"),
  [
    ((10.0, 20.0, 30.0), (350.0, 25.0, 40.0), 2000),  # PHI wraps: |340| -> 20
    ((0.0, 0.0, 0.0), (90.0, 45.0, 180.0), 18000),  # TWOTH dominates
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0),  # identical angles cost 0
  ],
)
def test_xray1_wraps_phi_and_scales_by_100(
  coord1: tuple[float, float, float], coord2: tuple[float, float, float], expected: int
) -> None:
  """WHAT: XRAY1 takes the circular PHI distance, the max over the three angles, and scales by 100.

  WHY: PHI is an angle on a circle, so 350 deg is 20 deg away from 10 deg, not 340; without
  the wrap and the scale factor every crystallography instance is mispriced.

  EXPECTED: 2000 for the wrapped pair, 18000 when TWOTH dominates, 0 for identical angles.
  """
  assert compute_xray1(coord1, coord2) == expected
