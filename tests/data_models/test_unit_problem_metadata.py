"""Metadata and derived-property tests for `data_models.problem.Problem`.

Covers the payload produced by `Problem.to_metadata_dict` and the derived properties
`is_symmetric`, `has_capacity_constraints` and `requires_distance_computation` that
the payload and the loaders branch on. Instances are built inline from synthetic
arrays: no test reads `datasets/`, and none needs a GPU or any randomness.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from data_models.problem import Problem

DIMENSION = 3
COORDINATES: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
DISTANCES: np.ndarray = np.zeros((DIMENSION, DIMENSION), dtype=np.int32)
DEMANDS: np.ndarray = np.array([0, 1, 1], dtype=np.int32)
DISPLAY_COORDS: np.ndarray = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

EXPECTED_METADATA_KEYS: frozenset[str] = frozenset(
  {
    "problem_id",
    "name",
    "tsplib_name",
    "problem_type",
    "dimension",
    "edge_type",
    "edge_weight_type",
    "edge_weight_format",
    "is_symmetric",
    "comment",
    "capacity",
    "has_coordinates",
    "has_display_coords",
    "has_explicit_matrix",
    "has_solution",
    "optimal_cost",
    "source_file",
    "file_checksum",
  }
)


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
# Derived properties
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
  ("problem_type", "expected"),
  [
    pytest.param("TSP", True, id="tsp"),
    pytest.param("CVRP", True, id="cvrp"),
    pytest.param("ATSP", False, id="atsp"),
  ],
)
def test_is_symmetric_reflects_problem_type(problem_type: str, expected: bool) -> None:
  """
  WHAT: `is_symmetric` for each supported problem type.
  WHY: symmetry decides whether upper-triangle-only kernels are valid; a wrong answer corrupts costs.
  EXPECTED: True for TSP and CVRP, False for ATSP.
  """
  assert _problem_of_type(problem_type).is_symmetric is expected


@pytest.mark.parametrize(
  ("problem_type", "expected"),
  [
    pytest.param("TSP", False, id="tsp"),
    pytest.param("CVRP", True, id="cvrp"),
    pytest.param("ATSP", False, id="atsp"),
  ],
)
def test_has_capacity_constraints_reflects_problem_type(problem_type: str, expected: bool) -> None:
  """
  WHAT: `has_capacity_constraints` for each supported problem type.
  WHY: the flag gates capacity-aware selection and crossover, so it must follow CVRP only.
  EXPECTED: True for CVRP, False for TSP and ATSP.
  """
  assert _problem_of_type(problem_type).has_capacity_constraints is expected


@pytest.mark.parametrize(
  ("edge_type", "expected"),
  [
    pytest.param("EUC_2D", True, id="euc-2d"),
    pytest.param("CEIL_2D", True, id="ceil-2d"),
    pytest.param("GEO", True, id="geo"),
    pytest.param("ATT", True, id="att"),
    pytest.param("EXPLICIT", False, id="explicit"),
  ],
)
def test_requires_distance_computation_reflects_edge_type(edge_type: str, expected: bool) -> None:
  """
  WHAT: `requires_distance_computation` for every valid edge type.
  WHY: the loader uses it to decide between computing a matrix and fetching the stored one.
  EXPECTED: True for all geometric types, False only for EXPLICIT.
  """
  problem = _problem_of_type("ATSP") if edge_type == "EXPLICIT" else _make_problem(edge_type=edge_type)

  assert problem.requires_distance_computation is expected


# ═══════════════════════════════════════════════════════════════════
# to_metadata_dict
# ═══════════════════════════════════════════════════════════════════


def test_metadata_dict_has_exactly_the_documented_keys() -> None:
  """
  WHAT: key set of a default metadata payload.
  WHY: checkpoints and report writers index these keys by literal, so an added or renamed key breaks them.
  EXPECTED: exactly the 18 documented keys.
  """
  metadata = _make_problem().to_metadata_dict()

  assert set(metadata) == EXPECTED_METADATA_KEYS
  assert len(metadata) == 18


def test_metadata_dict_carries_field_values() -> None:
  """
  WHAT: every metadata value for a fully populated geometric instance.
  WHY: the payload is persisted verbatim; aliasing or dropping a field loses provenance data.
  EXPECTED: the exact dictionary below.
  """
  problem = _make_problem(
    problem_id=7,
    tsplib_name="tiny3.tsp",
    comment="three-node synthetic instance",
    edge_weight_format="FUNCTION",
    has_solution=True,
    display_coords=DISPLAY_COORDS,
    source_file="fixtures/tsp/tiny3.tsp",
    file_checksum="deadbeef",
  )

  assert problem.to_metadata_dict(optimal_cost=42.5) == {
    "problem_id": 7,
    "name": "tiny3",
    "tsplib_name": "tiny3.tsp",
    "problem_type": "TSP",
    "dimension": DIMENSION,
    "edge_type": "EUC_2D",
    "edge_weight_type": "EUC_2D",
    "edge_weight_format": "FUNCTION",
    "is_symmetric": True,
    "comment": "three-node synthetic instance",
    "capacity": None,
    "has_coordinates": True,
    "has_display_coords": True,
    "has_explicit_matrix": False,
    "has_solution": True,
    "optimal_cost": 42.5,
    "source_file": "fixtures/tsp/tiny3.tsp",
    "file_checksum": "deadbeef",
  }


@pytest.mark.parametrize(
  "tsplib_name",
  [
    pytest.param(None, id="absent"),
    pytest.param("", id="empty-string"),
  ],
)
def test_metadata_dict_falls_back_to_name_for_tsplib_name(tsplib_name: str | None) -> None:
  """
  WHAT: `tsplib_name` in the payload when the field is None or an empty string.
  WHY: report tables key on the TSPLIB label; a falsy value must degrade to the instance name, not to ''.
  EXPECTED: `name` in both falsy cases.
  """
  metadata = _make_problem(tsplib_name=tsplib_name).to_metadata_dict()

  assert metadata["tsplib_name"] == "tiny3"


def test_metadata_dict_optimal_cost_defaults_to_none_and_accepts_a_value() -> None:
  """
  WHAT: the `optimal_cost` argument of `to_metadata_dict`.
  WHY: instances without a known optimum must record JSON `null` rather than a sentinel number.
  EXPECTED: None when omitted, the given float when supplied.
  """
  problem = _make_problem()

  assert problem.to_metadata_dict()["optimal_cost"] is None
  assert problem.to_metadata_dict(optimal_cost=3323.0)["optimal_cost"] == 3323.0


def test_metadata_dict_flags_for_explicit_problem() -> None:
  """
  WHAT: the boolean flags and capacity for an explicit ATSP instance.
  WHY: `has_explicit_matrix` and `has_coordinates` are derived, not stored; a wrong derivation mislabels data.
  EXPECTED: explicit-matrix True, coordinates False, display coords False, symmetry False, capacity None.
  """
  metadata = _problem_of_type("ATSP").to_metadata_dict()

  assert metadata["has_coordinates"] is False
  assert metadata["has_display_coords"] is False
  assert metadata["has_explicit_matrix"] is True
  assert metadata["is_symmetric"] is False
  assert metadata["capacity"] is None


def test_metadata_dict_is_json_serializable() -> None:
  """
  WHAT: a JSON dump and reload of a CVRP payload with a known optimum.
  WHY: the payload feeds checkpoint JSON; a numpy scalar in any field would abort the write.
  EXPECTED: the round-trip equals the original mapping and keeps the numeric values.
  """
  metadata = _make_cvrp().to_metadata_dict(optimal_cost=123.5)

  round_tripped = json.loads(json.dumps(metadata))

  assert round_tripped == metadata
  assert round_tripped["optimal_cost"] == 123.5
  assert round_tripped["capacity"] == 10


@pytest.mark.parametrize(
  ("fixture_name", "expected_dimension", "expected_edge_type"),
  [
    pytest.param("burma14_problem", 14, "GEO", id="burma14-geo"),
    pytest.param("berlin52_problem", 52, "EUC_2D", id="berlin52-euc-2d"),
  ],
)
def test_metadata_dict_for_loaded_problem(
  request: pytest.FixtureRequest,
  fixture_name: str,
  expected_dimension: int,
  expected_edge_type: str,
) -> None:
  """
  WHAT: metadata of a Problem produced by the real loader on the hermetic fixtures.
  WHY: inline instances cannot catch a loader that populates fields differently from the contract.
  EXPECTED: documented keys, matching dimension/edge type, coordinates present, JSON round-trip.
  """
  problem: Problem = request.getfixturevalue(fixture_name)

  metadata = problem.to_metadata_dict()

  assert set(metadata) == EXPECTED_METADATA_KEYS
  assert metadata["dimension"] == expected_dimension
  assert metadata["edge_type"] == expected_edge_type
  assert metadata["edge_weight_type"] == expected_edge_type
  assert metadata["has_coordinates"] is True
  assert metadata["has_explicit_matrix"] is False
  assert metadata["is_symmetric"] is True
  assert json.loads(json.dumps(metadata)) == metadata
