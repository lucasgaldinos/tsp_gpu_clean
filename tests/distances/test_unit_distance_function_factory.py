"""Unit tests for the ``get_distance_function`` registry in :mod:`distances.pairwise`.

The factory is the only entry point that turns an ``EDGE_WEIGHT_TYPE`` header into a
callable, so its dispatch table, its case handling and its rejection paths are asserted
here, together with the family-spanning invariants that only the registry tables can
enumerate (symmetry, and a zero self-distance for the Cartesian metrics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from distances.pairwise import (
  compute_att,
  compute_ceiling_2d,
  compute_euclidean_2d,
  compute_euclidean_3d,
  compute_geographical,
  compute_manhattan_2d,
  compute_manhattan_3d,
  compute_maximum_2d,
  compute_maximum_3d,
  compute_xray1,
  get_distance_function,
)

if TYPE_CHECKING:
  from collections.abc import Callable

# Every type the factory maps to a concrete function (src/distances/pairwise.py:445-457).
SUPPORTED_TYPE_FUNCTIONS: list[tuple[str, Callable[..., int]]] = [
  ("EUC_2D", compute_euclidean_2d),
  ("EUC_3D", compute_euclidean_3d),
  ("MAN_2D", compute_manhattan_2d),
  ("MAN_3D", compute_manhattan_3d),
  ("MAX_2D", compute_maximum_2d),
  ("MAX_3D", compute_maximum_3d),
  ("GEO", compute_geographical),
  ("ATT", compute_att),
  ("CEIL_2D", compute_ceiling_2d),
  ("XRAY1", compute_xray1),
]

CARTESIAN_2D_FUNCTIONS: list[tuple[str, Callable[..., int]]] = [
  ("EUC_2D", compute_euclidean_2d),
  ("MAN_2D", compute_manhattan_2d),
  ("MAX_2D", compute_maximum_2d),
  ("CEIL_2D", compute_ceiling_2d),
  ("ATT", compute_att),
]


@pytest.mark.parametrize(("edge_weight_type", "expected_function"), SUPPORTED_TYPE_FUNCTIONS)
def test_get_distance_function_returns_the_mapped_function(
  edge_weight_type: str, expected_function: Callable[..., int]
) -> None:
  """WHAT: the factory returns the module-level function object for each of the ten types.

  WHY: identity (not equality) proves the dispatch table is wired to the implementation
  under test rather than to a wrapper or a stale copy.

  EXPECTED: ``get_distance_function("EUC_2D") is compute_euclidean_2d`` and likewise for
  EUC_3D, MAN_2D, MAN_3D, MAX_2D, MAX_3D, GEO, ATT, CEIL_2D and XRAY1.
  """
  assert get_distance_function(edge_weight_type) is expected_function


@pytest.mark.parametrize("edge_weight_type", ["euc_2d", "Euc_2d", "geo", "Att", "ceil_2d", "xray1"])
def test_get_distance_function_is_case_insensitive(edge_weight_type: str) -> None:
  """WHAT: mixed-case and lower-case type names resolve to the same callable.

  WHY: the loader upper-cases file headers, but CLI and notebook callers pass free text;
  a case-sensitive lookup would raise ``ValueError`` on ``"geo"``.

  EXPECTED: the lower-case variant returns the same object as its upper-case form.
  """
  assert get_distance_function(edge_weight_type) is get_distance_function(edge_weight_type.upper())


@pytest.mark.parametrize("edge_weight_type", ["EXPLICIT", "SPECIAL", "explicit", "special"])
def test_get_distance_function_rejects_explicit_and_special(edge_weight_type: str) -> None:
  """WHAT: EXPLICIT and SPECIAL raise ``ValueError`` instead of returning a function.

  WHY: EXPLICIT costs come from a stored matrix and SPECIAL from external documentation;
  computing a coordinate formula for them would fabricate distances.

  EXPECTED: ``ValueError`` matching ``handled via explicit data`` (case-insensitively,
  including the lower-case spellings).
  """
  with pytest.raises(ValueError, match="handled via explicit data"):
    get_distance_function(edge_weight_type)


@pytest.mark.parametrize("edge_weight_type", ["XRAY2", "BOGUS", ""])
def test_get_distance_function_rejects_unknown_types(edge_weight_type: str) -> None:
  """WHAT: unknown types raise ``ValueError`` naming the supported set.

  WHY: a typo in an instance header must fail loudly at load time, not return a fallback
  metric that silently changes the problem being solved.

  EXPECTED: ``ValueError`` matching ``Unknown or unsupported EDGE_WEIGHT_TYPE``.
  """
  with pytest.raises(ValueError, match="Unknown or unsupported EDGE_WEIGHT_TYPE"):
    get_distance_function(edge_weight_type)


def _sample_pair(edge_weight_type: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
  """Return a coordinate pair whose dimensionality matches ``edge_weight_type``."""
  if edge_weight_type == "GEO":
    return (16.47, 96.10), (20.09, 92.54)
  if edge_weight_type in {"EUC_3D", "MAN_3D", "MAX_3D", "XRAY1"}:
    return (0.0, 0.0, 0.0), (4.0, 6.0, 8.0)
  return (0.0, 0.0), (3.1, 4.0)


@pytest.mark.parametrize(("edge_weight_type", "distance_function"), SUPPORTED_TYPE_FUNCTIONS)
def test_pairwise_function_is_symmetric(edge_weight_type: str, distance_function: Callable[..., int]) -> None:
  """WHAT: ``f(a, b) == f(b, a)`` for every supported type.

  WHY: every coordinate-based TSPLIB metric is symmetric; the builder mirrors the upper
  triangle, so an asymmetric function would corrupt half of the matrix.

  EXPECTED: identical values in both argument orders for all ten types.
  """
  first, second = _sample_pair(edge_weight_type)

  assert distance_function(first, second) == distance_function(second, first)


@pytest.mark.parametrize(("edge_weight_type", "distance_function"), CARTESIAN_2D_FUNCTIONS)
def test_cartesian_function_of_a_point_with_itself_is_zero(
  edge_weight_type: str, distance_function: Callable[..., int]
) -> None:
  """WHAT: a Cartesian metric returns 0 for identical points.

  WHY: distinguishes the Cartesian families from GEO (which returns 1 by the ``+1.0`` rule)
  and guards the rounding of an all-zero vector.

  EXPECTED: 0 for every type in the Cartesian family.
  """
  assert distance_function((3.1, 4.0), (3.1, 4.0)) == 0, f"{edge_weight_type}: self-distance must be 0"
