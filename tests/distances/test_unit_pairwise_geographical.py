"""Unit tests for the geographical (``GEO``) pairwise distance function.

``GEO`` coordinates are DDD.MM (minutes, not hundredths of a degree) and the result is
``int(d + 1.0)``; both details shift every distance of the family, so they are pinned
against hand-computed radians and kilometres plus a real TSPLIB95 instance.
"""

from __future__ import annotations

import math

import pytest

from distances.pairwise import _to_radians_geo, compute_geographical

# Berlin (52.31N, 13.23E) -> Munich (48.08N, 11.34E) in TSPLIB DDD.MM format.
GEO_BERLIN = (52.31, 13.23)
GEO_MUNICH = (48.08, 11.34)
GEO_HAND_COMPUTED_KM = 504.718377149365722


@pytest.mark.parametrize(
  ("coordinate", "expected_radians"),
  [
    (52.31, 0.916588554814815),  # 52 deg 18.6 min
    (13.23, 0.233583182962963),
    (48.08, 0.840084971851852),
    (11.34, 0.201876374814815),
  ],
)
def test_geographical_degree_minute_conversion(coordinate: float, expected_radians: float) -> None:
  """WHAT: DDD.MM values become radians via ``PI * (deg + 5 * min / 3) / 180``.

  WHY: the fractional part of a TSPLIB GEO coordinate is minutes, not hundredths of a
  degree; treating 52.31 as 52.31 degrees instead of 52 deg 18.6 min shifts every distance.

  EXPECTED: each coordinate matches the hand-computed radian value to 1e-9 relative.
  """
  assert _to_radians_geo(coordinate) == pytest.approx(expected_radians, rel=1e-9)


def test_geographical_matches_hand_computed_value() -> None:
  """WHAT: Berlin -> Munich equals the great-circle distance computed from hand-derived angles.

  WHY: exercises the full spherical law of cosines and the TSPLIB ``int(d + 1.0)`` tail in
  one check that does not reuse the module's own arithmetic.

  EXPECTED: ``d = 6378.388 * acos(0.996870897783265) = 504.718377149 km`` (rel 1e-9) and
  ``compute_geographical(...) == int(504.718377149 + 1.0) == 505``.
  """
  lat1, lon1 = 0.916588554814815, 0.233583182962963
  lat2, lon2 = 0.840084971851852, 0.201876374814815
  q1 = math.cos(lon1 - lon2)
  q2 = math.cos(lat1 - lat2)
  q3 = math.cos(lat1 + lat2)
  distance_km = 6378.388 * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3))

  assert distance_km == pytest.approx(GEO_HAND_COMPUTED_KM, rel=1e-9)
  assert compute_geographical(GEO_BERLIN, GEO_MUNICH) == int(distance_km + 1.0) == 505


def test_geographical_identical_points_cost_one() -> None:
  """WHAT: GEO returns 1 for a city paired with itself, not 0.

  WHY: TSPLIB95 truncates ``d + 1.0``, so GEO distances are never below 1; the matrix
  builder compensates by zeroing its diagonal explicitly.

  EXPECTED: ``compute_geographical(p, p) == 1``.
  """
  assert compute_geographical(GEO_BERLIN, GEO_BERLIN) == 1


def test_geographical_burma14_first_pair() -> None:
  """WHAT: burma14 cities 1-2 are 153 km apart under GEO.

  WHY: anchors the implementation to a real TSPLIB95 instance so a formula change cannot
  pass on synthetic points alone.

  EXPECTED: 153 for (16.47, 96.10) -> (16.47, 94.44).
  """
  assert compute_geographical((16.47, 96.10), (16.47, 94.44)) == 153
