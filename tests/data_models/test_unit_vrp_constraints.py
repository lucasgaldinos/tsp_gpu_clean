"""Unit tests for `VRPConstraints` and its integration into `Problem` (`DP17-1`, `DP17-3`).

Covers Tier-1 scalar validation, Tier-2 reserved array normalization/read-only guarantees,
`bool`/`NaN`/`Inf` rejection, array-safe equality & hashing, `validate_against_dimension`,
and `Problem` synchronization across `TSP`, `ATSP`, `CVRP`, `OVRP`, and `DCVRP`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from data_models import (
  VALID_PROBLEM_TYPES,
  VRP_PROBLEM_TYPES,
  InvalidProblemDataError,
  Problem,
  VRPConstraints,
)

DIMENSION = 4
COORDINATES: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
DEMANDS: np.ndarray = np.array([0, 2, 3, 4], dtype=np.int32)
TIME_WINDOWS: np.ndarray = np.array([[0.0, 100.0], [10.0, 30.0], [20.0, 50.0], [40.0, 40.0]], dtype=np.float64)
SERVICE_TIMES: np.ndarray = np.array([0.0, 5.0, 10.0, 2.5], dtype=np.float64)


def _make_problem(**overrides: Any) -> Problem:
  """Return a valid 4-node geometric TSP instance with `overrides` applied."""
  fields: dict[str, Any] = {
    "name": "tiny4",
    "dimension": DIMENSION,
    "problem_type": "TSP",
    "edge_type": "EUC_2D",
    "coordinates": COORDINATES,
    "distances": None,
  }
  fields.update(overrides)
  return Problem(**fields)


# ═══════════════════════════════════════════════════════════════════
# VRPConstraints standalone contract
# ═══════════════════════════════════════════════════════════════════


def test_default_constraints_are_unconstrained_and_closed() -> None:
  """WHAT: Default `VRPConstraints()`.
  WHY: TSP/ATSP instances inherit this default without breaking existing semantics.
  EXPECTED: all limits None, `return_to_depot=True`, `is_constrained=False`.
  """
  c = VRPConstraints()
  assert c.capacity is None
  assert c.max_vehicles is None
  assert c.max_route_distance is None
  assert c.return_to_depot is True
  assert c.time_windows is None
  assert c.service_times is None
  assert c.is_constrained is False


def test_tier1_and_tier2_valid_construction_and_normalization() -> None:
  """WHAT: Full Tier-1 + Tier-2 construction with int distance and non-float64 arrays.
  WHY: `max_route_distance` must normalize to `float`, and Tier-2 arrays must become
  read-only C-contiguous `float64` copies without mutating caller inputs.
  EXPECTED: normalized types, read-only flags, caller buffers untouched, `is_constrained=True`.
  """
  tw_in: Any = np.asfortranarray(TIME_WINDOWS.astype(np.float32))
  st_in: Any = np.array([0, 5, 10, 2], dtype=np.int32)

  c = VRPConstraints(
    capacity=50,
    max_vehicles=3,
    max_route_distance=120,
    return_to_depot=False,
    time_windows=tw_in,
    service_times=st_in,
  )

  assert c.capacity == 50
  assert c.max_vehicles == 3
  assert c.max_route_distance == 120.0
  assert isinstance(c.max_route_distance, float)
  assert c.return_to_depot is False
  assert c.is_constrained is True

  assert c.time_windows is not None
  assert c.time_windows.dtype == np.dtype(np.float64)
  assert c.time_windows.flags["C_CONTIGUOUS"]
  assert not c.time_windows.flags["WRITEABLE"]
  assert tw_in.flags["WRITEABLE"]

  assert c.service_times is not None
  assert c.service_times.dtype == np.dtype(np.float64)
  assert c.service_times.flags["C_CONTIGUOUS"]
  assert not c.service_times.flags["WRITEABLE"]
  assert st_in.flags["WRITEABLE"]

  with pytest.raises(ValueError, match="read-only"):
    c.time_windows[0, 0] = 999.0
  with pytest.raises(ValueError, match="read-only"):
    c.service_times[0] = 999.0


def test_constraints_instance_is_frozen() -> None:
  """WHAT: Attribute reassignment on `VRPConstraints`.
  WHY: Frozen contract ensures cached hash and ProblemContext consistency.
  EXPECTED: `FrozenInstanceError`.
  """
  c = VRPConstraints(capacity=10)
  with pytest.raises(FrozenInstanceError):
    setattr(c, "capacity", 20)  # noqa: B010


@pytest.mark.parametrize(
  "field_name, bad_value",
  [
    pytest.param("capacity", 0, id="cap-zero"),
    pytest.param("capacity", -10, id="cap-negative"),
    pytest.param("capacity", True, id="cap-bool-true"),
    pytest.param("capacity", False, id="cap-bool-false"),
    pytest.param("capacity", 10.0, id="cap-float"),
    pytest.param("capacity", "10", id="cap-str"),
    pytest.param("max_vehicles", 0, id="veh-zero"),
    pytest.param("max_vehicles", -1, id="veh-negative"),
    pytest.param("max_vehicles", True, id="veh-bool-true"),
    pytest.param("max_vehicles", False, id="veh-bool-false"),
    pytest.param("max_vehicles", 2.5, id="veh-float"),
    pytest.param("max_route_distance", 0.0, id="dist-zero"),
    pytest.param("max_route_distance", -5.5, id="dist-negative"),
    pytest.param("max_route_distance", float("nan"), id="dist-nan"),
    pytest.param("max_route_distance", float("inf"), id="dist-inf"),
    pytest.param("max_route_distance", float("-inf"), id="dist-neg-inf"),
    pytest.param("max_route_distance", True, id="dist-bool-true"),
    pytest.param("max_route_distance", False, id="dist-bool-false"),
    pytest.param("max_route_distance", "100.0", id="dist-str"),
    pytest.param("return_to_depot", 1, id="rtd-int-1"),
    pytest.param("return_to_depot", 0, id="rtd-int-0"),
    pytest.param("return_to_depot", "True", id="rtd-str"),
  ],
)
def test_rejects_invalid_tier1_scalars(field_name: str, bad_value: Any) -> None:
  """WHAT: Invalid Tier-1 scalar types and boundary values.
  WHY: `bool` subclasses `int` in Python and `NaN`/`Inf` silently corrupt CUDA/NumPy math.
  EXPECTED: `ValueError`.
  """
  with pytest.raises(ValueError):
    VRPConstraints(**{field_name: bad_value})


@pytest.mark.parametrize(
  "bad_tw",
  [
    pytest.param(np.zeros((4,), dtype=np.float64), id="1d"),
    pytest.param(np.zeros((4, 3), dtype=np.float64), id="3-cols"),
    pytest.param(np.zeros((0, 2), dtype=np.float64), id="empty"),
    pytest.param(np.ones((4, 2), dtype=bool), id="bool-dtype"),
    pytest.param(np.zeros((4, 2), dtype=np.complex128), id="complex-dtype"),
    pytest.param(np.array([[0.0, np.nan], [1.0, 2.0]]), id="nan"),
    pytest.param(np.array([[0.0, np.inf], [1.0, 2.0]]), id="inf"),
    pytest.param(np.array([[10.0, 5.0], [1.0, 2.0]]), id="inverted-window"),
  ],
)
def test_rejects_invalid_time_windows(bad_tw: np.ndarray) -> None:
  """WHAT: Malformed Tier-2 `time_windows` arrays.
  WHY: Must enforce `(n, 2)` finite `float64` with `earliest <= latest`.
  EXPECTED: `ValueError`.
  """
  with pytest.raises(ValueError):
    VRPConstraints(time_windows=bad_tw)


@pytest.mark.parametrize(
  "bad_st",
  [
    pytest.param(np.zeros((4, 1), dtype=np.float64), id="2d"),
    pytest.param(np.zeros((0,), dtype=np.float64), id="empty"),
    pytest.param(np.ones((4,), dtype=bool), id="bool-dtype"),
    pytest.param(np.array([0.0, np.nan, 1.0]), id="nan"),
    pytest.param(np.array([0.0, np.inf, 1.0]), id="inf"),
    pytest.param(np.array([0.0, -0.1, 1.0]), id="negative"),
  ],
)
def test_rejects_invalid_service_times(bad_st: np.ndarray) -> None:
  """WHAT: Malformed Tier-2 `service_times` arrays.
  WHY: Must enforce `(n,)` finite non-negative `float64`.
  EXPECTED: `ValueError`.
  """
  with pytest.raises(ValueError):
    VRPConstraints(service_times=bad_st)


def test_rejects_mismatched_tier2_array_lengths() -> None:
  """WHAT: `time_windows` and `service_times` with differing node counts.
  WHY: Both arrays index the same problem nodes.
  EXPECTED: `ValueError` naming both lengths.
  """
  with pytest.raises(ValueError, match="must match service_times length"):
    VRPConstraints(time_windows=TIME_WINDOWS, service_times=SERVICE_TIMES[:3])


def test_is_constrained_triggers_for_each_active_field() -> None:
  """WHAT: `is_constrained` across each single-field perturbation.
  WHY: Callers use `is_constrained` to detect whether any restriction is active.
  EXPECTED: `True` for every non-default configuration.
  """
  assert VRPConstraints(capacity=10).is_constrained is True
  assert VRPConstraints(max_vehicles=2).is_constrained is True
  assert VRPConstraints(max_route_distance=50.0).is_constrained is True
  assert VRPConstraints(return_to_depot=False).is_constrained is True
  assert VRPConstraints(time_windows=TIME_WINDOWS).is_constrained is True
  assert VRPConstraints(service_times=SERVICE_TIMES).is_constrained is True


def test_validate_against_dimension() -> None:
  """WHAT: `validate_against_dimension` with matching, mismatched, and invalid dimensions.
  WHY: Ensures Tier-2 arrays match `Problem.dimension`.
  EXPECTED: passes on matching `4`, raises `ValueError` on `3`, `0`, `-1`, `True`.
  """
  c = VRPConstraints(time_windows=TIME_WINDOWS, service_times=SERVICE_TIMES)
  c.validate_against_dimension(4)

  with pytest.raises(ValueError, match=r"time_windows\.shape should be \(3, 2\)"):
    c.validate_against_dimension(3)
  with pytest.raises(ValueError, match="positive int"):
    c.validate_against_dimension(0)
  with pytest.raises(ValueError, match="positive int"):
    c.validate_against_dimension(True)

  c_st_only = VRPConstraints(service_times=SERVICE_TIMES)
  with pytest.raises(ValueError, match=r"service_times\.shape should be \(5,\)"):
    c_st_only.validate_against_dimension(5)


def test_array_safe_equality_and_hashing() -> None:
  """WHAT: Equality and hashing of `VRPConstraints` holding NumPy arrays.
  WHY: Default dataclass `==` and `hash` crash on `np.ndarray` fields.
  EXPECTED: independent copies compare equal, share hash, and work in sets/dicts;
  differing instances compare unequal.
  """
  left = VRPConstraints(
    capacity=10,
    max_vehicles=2,
    max_route_distance=40.0,
    return_to_depot=True,
    time_windows=TIME_WINDOWS.copy(),
    service_times=SERVICE_TIMES.copy(),
  )
  right = VRPConstraints(
    capacity=10,
    max_vehicles=2,
    max_route_distance=40,
    return_to_depot=True,
    time_windows=TIME_WINDOWS.copy(),
    service_times=SERVICE_TIMES.copy(),
  )
  different = VRPConstraints(
    capacity=10,
    max_vehicles=2,
    max_route_distance=40.0,
    return_to_depot=True,
    time_windows=TIME_WINDOWS + 1.0,
    service_times=SERVICE_TIMES.copy(),
  )

  assert left == right
  assert hash(left) == hash(right)
  assert len({left, right, different}) == 2
  assert left != different
  assert (left == "not-constraints") is False


# ═══════════════════════════════════════════════════════════════════
# Problem integration (TSP, ATSP, CVRP, OVRP, DCVRP)
# ═══════════════════════════════════════════════════════════════════


def test_problem_type_constants_include_vrp_variants() -> None:
  """WHAT: Exported `VRP_PROBLEM_TYPES` and `VALID_PROBLEM_TYPES` constants.
  WHY: Downstream loaders and strategies share one canonical problem-type registry.
  EXPECTED: `VRP_PROBLEM_TYPES == {'CVRP', 'OVRP', 'DCVRP'}` and subset of `VALID_PROBLEM_TYPES`.
  """
  assert VRP_PROBLEM_TYPES == frozenset({"CVRP", "OVRP", "DCVRP"})
  assert VALID_PROBLEM_TYPES == frozenset({"TSP", "ATSP", "CVRP", "OVRP", "DCVRP"})


def test_cvrp_syncs_capacity_and_constraints_bidirectionally() -> None:
  """WHAT: Backward-compatible CVRP creation via `capacity=...` and via `constraints=...`.
  WHY: Existing callers pass `capacity` only; new callers may pass `VRPConstraints`.
  EXPECTED: both produce identical `Problem` with synced `capacity` and `constraints.capacity`.
  """
  from_cap = _make_problem(problem_type="CVRP", capacity=10, demands=DEMANDS)
  from_constraints = _make_problem(
    problem_type="CVRP",
    demands=DEMANDS,
    constraints=VRPConstraints(capacity=10),
  )

  assert from_cap.capacity == 10
  assert from_cap.constraints == VRPConstraints(capacity=10, return_to_depot=True)
  assert from_constraints.capacity == 10
  assert from_cap == from_constraints
  assert hash(from_cap) == hash(from_constraints)


def test_conflicting_capacity_and_constraints_raises_invalid_problem_data() -> None:
  """WHAT: `Problem` given both `capacity=10` and `constraints=VRPConstraints(capacity=20)`.
  WHY: Silent precedence would hide configuration bugs.
  EXPECTED: `InvalidProblemDataError`.
  """
  with pytest.raises(InvalidProblemDataError, match="Conflicting capacity"):
    _make_problem(
      problem_type="CVRP",
      capacity=10,
      demands=DEMANDS,
      constraints=VRPConstraints(capacity=20),
    )


def test_ovrp_defaults_to_open_routes_and_rejects_closed_override() -> None:
  """WHAT: `OVRP` with omitted `constraints`, explicit `return_to_depot=False`, and invalid `True`.
  WHY: Open VRP routes never return to the depot; passing `return_to_depot=True` contradicts OVRP.
  EXPECTED: omitted `constraints` derives `return_to_depot=False`; explicit `True` raises `InvalidProblemDataError`.
  """
  ovrp_auto = _make_problem(problem_type="OVRP", capacity=10, demands=DEMANDS)
  assert ovrp_auto.constraints == VRPConstraints(capacity=10, return_to_depot=False)
  assert ovrp_auto.is_symmetric is True
  assert ovrp_auto.has_capacity_constraints is True
  assert "capacity=10" in str(ovrp_auto)

  ovrp_explicit = _make_problem(
    problem_type="OVRP",
    demands=DEMANDS,
    constraints=VRPConstraints(capacity=10, return_to_depot=False),
  )
  assert ovrp_explicit == ovrp_auto

  with pytest.raises(InvalidProblemDataError, match="return_to_depot=False"):
    _make_problem(
      problem_type="OVRP",
      capacity=10,
      demands=DEMANDS,
      constraints=VRPConstraints(return_to_depot=True),
    )


def test_dcvrp_requires_max_route_distance_and_accepts_valid_limit() -> None:
  """WHAT: `DCVRP` without and with `constraints.max_route_distance`.
  WHY: Distance-Constrained CVRP is undefined without `L_max`.
  EXPECTED: missing `max_route_distance` raises `ValueError`; valid limit succeeds and syncs `capacity`.
  """
  with pytest.raises(ValueError, match="max_route_distance"):
    _make_problem(problem_type="DCVRP", capacity=10, demands=DEMANDS)

  dcvrp = _make_problem(
    problem_type="DCVRP",
    capacity=10,
    demands=DEMANDS,
    constraints=VRPConstraints(max_route_distance=75.0, max_vehicles=2),
  )
  assert dcvrp.capacity == 10
  assert dcvrp.constraints == VRPConstraints(capacity=10, max_vehicles=2, max_route_distance=75.0)
  assert dcvrp.is_symmetric is True
  assert dcvrp.has_capacity_constraints is True


def test_vrp_validates_depot_demand_zero_and_non_negative_integer_demands() -> None:
  """WHAT: VRP demand vector validation (`demands[0] == 0`, non-negative, integer dtype).
  WHY: Split and capacity kernels assume `q_0 == 0` and `q_v >= 0` in `int32` (`DP17-3`).
  EXPECTED: non-zero depot demand, negative demand, and float/bool demands raise `ValueError`;
  `int64` demands normalize to C-contiguous `int32`.
  """
  with pytest.raises(ValueError, match=r"demands\[0\] must be 0"):
    _make_problem(problem_type="CVRP", capacity=10, demands=np.array([1, 2, 3, 4], dtype=np.int32))

  with pytest.raises(ValueError, match="non-negative"):
    _make_problem(problem_type="CVRP", capacity=10, demands=np.array([0, -1, 3, 4], dtype=np.int32))

  with pytest.raises(ValueError, match="integer dtype"):
    _make_problem(problem_type="CVRP", capacity=10, demands=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))

  normalized = _make_problem(problem_type="CVRP", capacity=10, demands=np.array([0, 2, 3, 4], dtype=np.int64))
  assert normalized.demands is not None
  assert normalized.demands.dtype == np.dtype(np.int32)
  assert normalized.demands.flags["C_CONTIGUOUS"]


def test_problem_validates_tier2_constraints_dimension_and_supports_open_tsp() -> None:
  """WHAT: `Problem` with mismatched Tier-2 arrays and Open TSP (`return_to_depot=False`).
  WHY: `validate_against_dimension` must run for every problem type, and TSP supports `return_to_depot=False`.
  EXPECTED: wrong Tier-2 length raises `ValueError`; Open TSP succeeds cleanly.
  """
  with pytest.raises(ValueError, match=r"time_windows\.shape should be \(4, 2\)"):
    _make_problem(constraints=VRPConstraints(time_windows=TIME_WINDOWS[:3]))

  open_tsp = _make_problem(
    constraints=VRPConstraints(return_to_depot=False, time_windows=TIME_WINDOWS, service_times=SERVICE_TIMES)
  )
  assert open_tsp.constraints.return_to_depot is False
  assert open_tsp.has_capacity_constraints is False
  assert open_tsp == _make_problem(
    constraints=VRPConstraints(
      return_to_depot=False,
      time_windows=TIME_WINDOWS.copy(),
      service_times=SERVICE_TIMES.copy(),
    )
  )
