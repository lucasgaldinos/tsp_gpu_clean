"""Construction-invariant tests for `data_models.problem.Problem`.

Covers every `ValueError` branch of `Problem.__post_init__`, the values accepted at
the edge of each invariant, and the immutable value-object contract. Instances are
built inline from synthetic arrays: no test reads `datasets/`, and none needs a GPU
or any randomness.
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from data_models.problem import Problem

DIMENSION = 3
COORDINATES: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
DISTANCES: np.ndarray = np.zeros((DIMENSION, DIMENSION), dtype=np.int32)
DEMANDS: np.ndarray = np.array([0, 1, 1], dtype=np.int32)

GEOMETRIC_EDGE_TYPES: tuple[str, ...] = ("EUC_2D", "CEIL_2D", "GEO", "ATT")

#: Repr of `set(VALID_*_TYPES)` inside an error message: element order is not stable,
#: so only the shape of the set literal can be pinned.
_SET_LITERAL_PATTERN = r"\{[^}]*\}"


def _make_problem(**overrides: Any) -> Problem:
  """Return a valid 3-node geometric TSP instance with `overrides` applied."""
  fields: dict[str, Any] = {
    "name": "tiny3",
    "dimension": DIMENSION,
    "problem_type": "TSP",
    "edge_type": "EUC_2D",
    "coordinates": COORDINATES,
    "distances": None,
  }
  fields.update(overrides)
  return Problem(**fields)


def _make_cvrp(**overrides: Any) -> Problem:
  """Return a valid 3-node CVRP instance with `overrides` applied."""
  fields: dict[str, Any] = {"problem_type": "CVRP", "capacity": 10, "demands": DEMANDS}
  fields.update(overrides)
  return _make_problem(**fields)


def _problem_of_type(problem_type: str) -> Problem:
  """Return a valid instance of `problem_type` (ATSP is explicit, CVRP carries capacity and demands)."""
  if problem_type == "ATSP":
    return _make_problem(problem_type="ATSP", edge_type="EXPLICIT", coordinates=None, distances=DISTANCES)
  if problem_type == "CVRP":
    return _make_cvrp()
  return _make_problem()


# ═══════════════════════════════════════════════════════════════════
# Value-object contract
# ═══════════════════════════════════════════════════════════════════


def test_extended_metadata_fields_default_to_none() -> None:
  """
  WHAT: defaults of the extended-metadata fields on a minimally populated instance.
  WHY: consumers test these with `is None`; a changed default silently alters checkpoints.
  EXPECTED: every optional field is None and `has_solution` is False.
  """
  problem = _make_problem()

  assert problem.capacity is None
  assert problem.demands is None
  assert problem.problem_id is None
  assert problem.comment is None
  assert problem.edge_weight_format is None
  assert problem.tsplib_name is None
  assert problem.has_solution is False
  assert problem.display_coords is None
  assert problem.depots is None
  assert problem.source_file is None
  assert problem.file_checksum is None


def test_instance_is_frozen() -> None:
  """
  WHAT: attribute assignment on a constructed instance.
  WHY: the dataclass is documented as immutable; mutation would desynchronise dependent caches.
  EXPECTED: `FrozenInstanceError` on assignment.
  """
  problem = _make_problem()

  with pytest.raises(FrozenInstanceError):
    setattr(problem, "name", "renamed")  # noqa: B010 - the point is to trigger the frozen __setattr__


def test_str_describes_the_instance() -> None:
  """
  WHAT: `__str__` output for a plain TSP instance and for a CVRP instance.
  WHY: the string is written into logs and report tables, and the CVRP branch adds capacity.
  EXPECTED: node count always, capacity only for CVRP.
  """
  assert str(_make_problem()) == "tiny3: TSP with 3 nodes"
  assert str(_make_cvrp()) == "tiny3: CVRP with 3 nodes, capacity=10"


def test_repr_reports_shapes_instead_of_array_contents() -> None:
  """
  WHAT: `__repr__` of a CVRP instance with coordinate, demand and distance-free arrays.
  WHY: a repr that dumped array data would flood logs; shapes are the useful summary.
  EXPECTED: the exact one-line representation with tuple shapes.
  """
  expected = (
    "Problem(name='tiny3', dimension=3, problem_type='CVRP', edge_type='EUC_2D', "
    "coordinates=(3, 2), distances=None, capacity=10, demands=(3,))"
  )

  assert repr(_make_cvrp()) == expected


def test_problems_with_equal_content_compare_equal() -> None:
  """
  WHAT: `==` between two independently built instances with identical content.
  WHY: the module documents value semantics, and benchmarking code compares instances by equality.
  EXPECTED: True, without raising.
  """
  first = _make_problem(coordinates=COORDINATES.copy())
  second = _make_problem(coordinates=COORDINATES.copy())

  assert first == second


# ═══════════════════════════════════════════════════════════════════
# __post_init__ rejections
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
  "dimension",
  [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(-10, id="far-negative"),
  ],
)
def test_rejects_non_positive_dimension(dimension: int) -> None:
  """
  WHAT: construction with `dimension <= 0`.
  WHY: node indices span 0..dimension-1, so a non-positive dimension addresses no node.
  EXPECTED: ValueError naming the offending dimension.
  """
  expected = f"dimension must be positive, got {dimension}"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(dimension=dimension)


@pytest.mark.parametrize(
  "problem_type",
  [
    pytest.param("TSPX", id="unknown"),
    pytest.param("tsp", id="lowercase"),
    pytest.param("", id="empty"),
  ],
)
def test_rejects_unknown_problem_type(problem_type: str) -> None:
  """
  WHAT: construction with a `problem_type` outside {TSP, ATSP, CVRP}.
  WHY: downstream dispatch and symmetry flags branch on this exact, case-sensitive value.
  EXPECTED: ValueError listing the valid types and echoing the rejected value.
  """
  pattern = "problem_type must be one of " + _SET_LITERAL_PATTERN + ", got '" + re.escape(problem_type) + "'"

  with pytest.raises(ValueError, match=pattern):
    _make_problem(problem_type=problem_type)


@pytest.mark.parametrize(
  "edge_type",
  [
    pytest.param("EUC2D", id="missing-underscore"),
    pytest.param("explicit", id="lowercase"),
    pytest.param("", id="empty"),
  ],
)
def test_rejects_unknown_edge_type(edge_type: str) -> None:
  """
  WHAT: construction with an `edge_type` outside {EUC_2D, CEIL_2D, GEO, ATT, EXPLICIT}.
  WHY: the distance factory matches on this value; an unknown key would fail later and elsewhere.
  EXPECTED: ValueError listing the valid types and echoing the rejected value.
  """
  pattern = "edge_type must be one of " + _SET_LITERAL_PATTERN + ", got '" + re.escape(edge_type) + "'"

  with pytest.raises(ValueError, match=pattern):
    _make_problem(edge_type=edge_type)


def test_explicit_edge_type_requires_distances() -> None:
  """
  WHAT: `edge_type='EXPLICIT'` with `distances=None`.
  WHY: an explicit instance has no coordinate path, so a missing matrix leaves the problem unusable.
  EXPECTED: ValueError stating the distances matrix is required.
  """
  expected = "edge_type='EXPLICIT' requires distances matrix, got None"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(problem_type="ATSP", edge_type="EXPLICIT", coordinates=None)


@pytest.mark.parametrize("edge_type", GEOMETRIC_EDGE_TYPES)
def test_geometric_edge_type_requires_coordinates(edge_type: str) -> None:
  """
  WHAT: each geometric `edge_type` with `coordinates=None`.
  WHY: every non-explicit distance formula reads node coordinates, so None must fail at construction.
  EXPECTED: ValueError naming the edge type that needs coordinates.
  """
  expected = f"edge_type='{edge_type}' requires coordinates, got None"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(edge_type=edge_type, coordinates=None)


@pytest.mark.parametrize(
  "coordinates",
  [
    pytest.param(np.zeros((2, 2)), id="fewer-rows"),
    pytest.param(np.zeros((4, 2)), id="more-rows"),
    pytest.param(np.zeros((3, 3)), id="wrong-width"),
    pytest.param(np.zeros((3,)), id="one-dimensional"),
  ],
)
def test_rejects_coordinates_with_wrong_shape(coordinates: np.ndarray) -> None:
  """
  WHAT: coordinates whose shape is not `(dimension, 2)`.
  WHY: distance kernels index `coordinates[i, 0]`/`[i, 1]` for every node; any other shape reads garbage.
  EXPECTED: ValueError stating the required and received shapes.
  """
  expected = f"coordinates.shape should be ({DIMENSION}, 2), got {coordinates.shape}"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(coordinates=coordinates)


def test_rejects_wrong_coordinates_shape_even_when_distances_are_present() -> None:
  """
  WHAT: an explicit problem that also carries wrongly shaped coordinates.
  WHY: the coordinate check is unconditional, not a branch of the EXPLICIT requirement.
  EXPECTED: ValueError on the coordinate shape, not on the matrix.
  """
  expected = f"coordinates.shape should be ({DIMENSION}, 2), got (2, 2)"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(
      problem_type="ATSP",
      edge_type="EXPLICIT",
      coordinates=np.zeros((2, 2)),
      distances=DISTANCES,
    )


@pytest.mark.parametrize(
  "distances",
  [
    pytest.param(np.zeros((2, 2), dtype=np.int32), id="fewer-rows"),
    pytest.param(np.zeros((4, 4), dtype=np.int32), id="more-rows"),
    pytest.param(np.zeros((3, 2), dtype=np.int32), id="rectangular"),
    pytest.param(np.zeros((3,), dtype=np.int32), id="one-dimensional"),
  ],
)
def test_rejects_distances_with_wrong_shape(distances: np.ndarray) -> None:
  """
  WHAT: an explicit matrix whose shape is not `(dimension, dimension)`.
  WHY: solvers index `distances[i, j]` over the full node range; a non-square matrix is out of bounds.
  EXPECTED: ValueError stating the required and received shapes.
  """
  expected = f"distances.shape should be ({DIMENSION}, {DIMENSION}), got {distances.shape}"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_problem(problem_type="ATSP", edge_type="EXPLICIT", coordinates=None, distances=distances)


def test_cvrp_requires_capacity() -> None:
  """
  WHAT: a CVRP instance with `capacity=None`.
  WHY: the capacity constraint defines CVRP; without it the instance is a plain TSP.
  EXPECTED: ValueError stating capacity is required.
  """
  with pytest.raises(ValueError, match=re.escape("CVRP instances require capacity")):
    _make_cvrp(capacity=None)


@pytest.mark.parametrize(
  "capacity",
  [
    pytest.param(0, id="zero"),
    pytest.param(-5, id="negative"),
  ],
)
def test_cvrp_rejects_non_positive_capacity(capacity: int) -> None:
  """
  WHAT: a CVRP instance with `capacity <= 0`.
  WHY: a non-positive vehicle capacity admits no non-empty route, so the instance is unsolvable.
  EXPECTED: ValueError naming the offending capacity.
  """
  expected = f"capacity must be positive, got {capacity}"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_cvrp(capacity=capacity)


def test_cvrp_requires_demands() -> None:
  """
  WHAT: a CVRP instance with `demands=None`.
  WHY: route feasibility is checked against per-node demands; None makes the check impossible.
  EXPECTED: ValueError stating demands are required.
  """
  with pytest.raises(ValueError, match=re.escape("CVRP instances require demands")):
    _make_cvrp(demands=None)


@pytest.mark.parametrize(
  "demands",
  [
    pytest.param(np.zeros((2,), dtype=np.int32), id="fewer-nodes"),
    pytest.param(np.zeros((4,), dtype=np.int32), id="more-nodes"),
    pytest.param(np.zeros((3, 1), dtype=np.int32), id="two-dimensional"),
  ],
)
def test_cvrp_rejects_wrong_demands_shape(demands: np.ndarray) -> None:
  """
  WHAT: a CVRP instance whose demands are not 1-D of length `dimension`.
  WHY: demands are consumed per node index; a transposed or truncated vector misassigns load.
  EXPECTED: ValueError stating the required and received shapes.
  """
  expected = f"demands.shape should be ({DIMENSION},), got {demands.shape}"

  with pytest.raises(ValueError, match=re.escape(expected)):
    _make_cvrp(demands=demands)


def test_cvrp_checks_capacity_before_demands() -> None:
  """
  WHAT: a CVRP instance missing both capacity and demands.
  WHY: the reported error must be stable so callers can rely on the documented validation order.
  EXPECTED: the capacity error, not the demands error.
  """
  with pytest.raises(ValueError, match=re.escape("CVRP instances require capacity")):
    _make_cvrp(capacity=None, demands=None)


# ═══════════════════════════════════════════════════════════════════
# Accepted boundaries
# ═══════════════════════════════════════════════════════════════════


def test_accepts_single_node_geometric_problem() -> None:
  """
  WHAT: the smallest legal geometric instance (`dimension=1`, one coordinate row).
  WHY: the positive-dimension and shape checks must accept the lower edge, not only reject below it.
  EXPECTED: construction succeeds and the arrays keep their edge shape.
  """
  coordinates = np.array([[0.0, 0.0]])

  problem = _make_problem(dimension=1, coordinates=coordinates)

  assert problem.dimension == 1
  assert problem.coordinates is coordinates
  assert problem.coordinates.shape == (1, 2)
  assert problem.requires_distance_computation is True


def test_accepts_cvrp_at_minimum_capacity() -> None:
  """
  WHAT: a single-node CVRP instance with `capacity=1` and a depot-only demand vector.
  WHY: capacity 1 is the smallest legal value and `[0]` the smallest legal demand vector.
  EXPECTED: construction succeeds with the edge values preserved.
  """
  demands = np.zeros((1,), dtype=np.int32)

  problem = _make_problem(
    problem_type="CVRP",
    dimension=1,
    coordinates=np.array([[0.0, 0.0]]),
    capacity=1,
    demands=demands,
  )

  assert problem.capacity == 1
  assert problem.demands is demands
  assert problem.demands.tolist() == [0]
  assert problem.has_capacity_constraints is True


def test_accepts_explicit_problem_without_coordinates() -> None:
  """
  WHAT: an ATSP instance holding only an explicit matrix.
  WHY: the documented ATSP shape is `coordinates=None`, so the coordinate requirement must not fire.
  EXPECTED: construction succeeds with distances retained and no coordinate array.
  """
  problem = _problem_of_type("ATSP")

  assert problem.coordinates is None
  assert problem.distances is DISTANCES
  assert problem.distances.shape == (DIMENSION, DIMENSION)
  assert problem.requires_distance_computation is False
  assert problem.is_symmetric is False


def test_accepts_explicit_problem_with_coordinates() -> None:
  """
  WHAT: an explicit instance carrying both a square matrix and well-shaped coordinates.
  WHY: display coordinates are legal alongside an explicit matrix; only their shape is constrained.
  EXPECTED: construction succeeds and both arrays are retained unchanged.
  """
  problem = _make_problem(
    problem_type="ATSP",
    edge_type="EXPLICIT",
    coordinates=COORDINATES,
    distances=DISTANCES,
  )

  assert problem.coordinates is COORDINATES
  assert problem.distances is DISTANCES
  assert problem.to_metadata_dict()["has_coordinates"] is True


def test_accepts_geometric_problem_carrying_a_square_matrix() -> None:
  """
  WHAT: a geometric TSP instance that also supplies a correctly shaped matrix.
  WHY: the matrix shape check applies to any present matrix; geometric instances may precompute one.
  EXPECTED: construction succeeds and the geometric edge type still drives distance computation.
  """
  problem = _make_problem(distances=DISTANCES)

  assert problem.distances is DISTANCES
  assert problem.requires_distance_computation is True
  assert problem.to_metadata_dict()["has_explicit_matrix"] is False
