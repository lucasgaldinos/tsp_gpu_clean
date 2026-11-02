"""Contract tests for `data_models.records`.

Pins the runtime shape of the TypedDicts exchanged across module boundaries
(required versus optional keys of `BenchmarkRunInput` and the `JsonPayload`
union). No GPU, no randomness, no dataset access.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from data_models.records import BenchmarkRunInput, JsonPayload, NemenyiResult

#: The 27 columns `NOT NULL` without a default in `schema.sql`.
REQUIRED_FIELDS: frozenset[str] = frozenset(
  {
    "problem_name",
    "problem_size",
    "optimal_cost",
    "algorithm",
    "backend",
    "repetitions",
    "successful_runs",
    "timestamp",
    "max_generations_configured",
    "mean_time",
    "std_time",
    "min_time",
    "max_time",
    "mean_cost",
    "std_cost",
    "best_cost",
    "worst_cost",
    "mean_gap",
    "std_gap",
    "best_gap",
    "mean_generations",
    "std_generations",
    "raw_times",
    "raw_costs",
    "raw_gaps",
    "algorithm_config",
    "strategies",
  }
)

#: The nullable columns that older checkpoints may omit.
OPTIONAL_FIELDS: frozenset[str] = frozenset(
  {
    "mean_initial_cost",
    "std_initial_cost",
    "mean_improvement",
    "min_generations",
    "max_generations",
    "mean_h2d_mb",
    "mean_d2h_mb",
    "total_transfer_mb",
    "mean_kernels",
    "raw_initial_costs",
    "raw_generations",
    "raw_stop_reasons",
    "construction_heuristic",
  }
)

MINIMAL_PAYLOAD: dict[str, object] = {
  "problem_name": "burma14",
  "problem_size": 14,
  "optimal_cost": 3323.0,
  "algorithm": "GeneticAlgorithm",
  "backend": "numpy",
  "repetitions": 5,
  "successful_runs": 5,
  "timestamp": "2026-10-30T10:00:00",
  "max_generations_configured": 100,
  "mean_time": 1.5,
  "std_time": 0.2,
  "min_time": 1.2,
  "max_time": 1.9,
  "mean_cost": 3400.0,
  "std_cost": 12.5,
  "best_cost": 3350.0,
  "worst_cost": 3450.0,
  "mean_gap": 0.023,
  "std_gap": 0.004,
  "best_gap": 0.008,
  "mean_generations": 80.0,
  "std_generations": 5.0,
  "raw_times": [1.2, 1.5, 1.9],
  "raw_costs": [3350.0, 3400.0, 3450.0],
  "raw_gaps": [0.008, 0.023, 0.038],
  "algorithm_config": '{"population_size": 32}',
  "strategies": {"local_search": {"class": "CPUTwoOpt"}},
}

FULL_PAYLOAD: dict[str, object] = {
  **MINIMAL_PAYLOAD,
  "mean_initial_cost": 4000.0,
  "std_initial_cost": 20.0,
  "mean_improvement": 600.0,
  "min_generations": 70,
  "max_generations": 100,
  "mean_h2d_mb": 0.5,
  "mean_d2h_mb": 0.25,
  "total_transfer_mb": 3.75,
  "mean_kernels": 40.0,
  "raw_initial_costs": [4000.0, 4020.0, 3980.0],
  "raw_generations": [70, 80, 100],
  "raw_stop_reasons": ["max_generations", "converged", "max_generations"],
  "construction_heuristic": "nearest_neighbor",
}

# ═══════════════════════════════════════════════════════════════════
# records.py — BenchmarkRunInput
# ═══════════════════════════════════════════════════════════════════


def test_benchmark_run_input_required_keys_match_the_not_null_columns() -> None:
  """
  WHAT: `BenchmarkRunInput.__required_keys__`.
  WHY: the runtime validator derives its required-field list from this set, so an edit here changes validation.
  EXPECTED: exactly the 27 documented required columns.
  """
  assert set(BenchmarkRunInput.__required_keys__) == REQUIRED_FIELDS
  assert len(BenchmarkRunInput.__required_keys__) == 27


def test_benchmark_run_input_optional_keys_match_the_nullable_columns() -> None:
  """
  WHAT: `BenchmarkRunInput.__optional_keys__`.
  WHY: these are the keys the writer reads with `dict.get`; a missing one must stay legal for old checkpoints.
  EXPECTED: exactly the 13 documented `NotRequired` columns.
  """
  assert set(BenchmarkRunInput.__optional_keys__) == OPTIONAL_FIELDS
  assert len(BenchmarkRunInput.__optional_keys__) == 13


def test_benchmark_run_input_key_sets_partition_the_annotations() -> None:
  """
  WHAT: the relationship between annotations, required keys and optional keys.
  WHY: a key added without `NotRequired` silently becomes mandatory and rejects old checkpoints.
  EXPECTED: disjoint sets whose union is every annotation, on a total TypedDict.
  """
  required = set(BenchmarkRunInput.__required_keys__)
  optional = set(BenchmarkRunInput.__optional_keys__)

  assert required.isdisjoint(optional)
  assert required | optional == set(BenchmarkRunInput.__annotations__)
  assert BenchmarkRunInput.__total__ is True


def test_benchmark_run_input_excludes_database_assigned_run_id() -> None:
  """
  WHAT: presence of `run_id` in the payload contract.
  WHY: the database assigns it from `benchmark_runs_seq`; a caller-supplied id would collide.
  EXPECTED: absent from annotations, required keys and optional keys.
  """
  assert "run_id" not in BenchmarkRunInput.__annotations__
  assert "run_id" not in BenchmarkRunInput.__required_keys__
  assert "run_id" not in BenchmarkRunInput.__optional_keys__


def test_minimal_payload_covers_every_required_field() -> None:
  """
  WHAT: a checkpoint payload built from required fields only.
  WHY: it is the smallest payload `validate_benchmark_data` must accept; an uncovered key would fail there.
  EXPECTED: keys equal the required set and no value is None, since required columns are NOT NULL.
  """
  assert set(MINIMAL_PAYLOAD) == set(BenchmarkRunInput.__required_keys__)
  assert all(value is not None for value in MINIMAL_PAYLOAD.values())


def test_full_payload_covers_every_declared_field() -> None:
  """
  WHAT: a checkpoint payload naming all 40 fields.
  WHY: it proves no contract key is unnamed and no payload key is invented, which is the drift this test guards.
  EXPECTED: keys equal required plus optional, a subset of the annotations, and 40 fields in total.
  """
  declared = set(BenchmarkRunInput.__required_keys__) | set(BenchmarkRunInput.__optional_keys__)

  assert set(FULL_PAYLOAD) == declared
  assert set(FULL_PAYLOAD) <= set(BenchmarkRunInput.__annotations__)
  assert len(FULL_PAYLOAD) == 40


def test_benchmark_run_input_uses_both_json_payload_shapes() -> None:
  """
  WHAT: the values chosen for the two `JsonPayload` columns in the sample payload.
  WHY: the writer must handle both an already serialized string and a mapping; one branch is easy to leave unused.
  EXPECTED: one column is `str`, the other a `Mapping`.
  """
  assert isinstance(MINIMAL_PAYLOAD["algorithm_config"], str)
  assert isinstance(MINIMAL_PAYLOAD["strategies"], Mapping)


# ═══════════════════════════════════════════════════════════════════
# records.py — NemenyiResult and JsonPayload
# ═══════════════════════════════════════════════════════════════════


def test_nemenyi_result_is_a_total_typed_dict_of_four_keys() -> None:
  """
  WHAT: key sets of `NemenyiResult`.
  WHY: consumers read all four keys unconditionally, so any of them turning optional would break indexing.
  EXPECTED: four required keys, no optional keys, total TypedDict.
  """
  assert set(NemenyiResult.__required_keys__) == {
    "p_values_matrix",
    "significant_pairs",
    "algorithm_names",
    "critical_distance",
  }
  assert NemenyiResult.__optional_keys__ == frozenset()
  assert NemenyiResult.__total__ is True


def test_nemenyi_result_annotation_types() -> None:
  """
  WHAT: resolved annotation types of `NemenyiResult`.
  WHY: the payload crosses into plotting and report code that indexes rows and columns with these dtypes.
  EXPECTED: float64 matrix, `(i, j, p)` triples, name list and float critical distance.
  """
  hints = typing.get_type_hints(NemenyiResult)

  assert hints["p_values_matrix"] == NDArray[np.float64]
  assert hints["significant_pairs"] == list[tuple[int, int, float]]
  assert hints["algorithm_names"] == list[str]
  assert hints["critical_distance"] is float


def test_json_payload_is_a_two_member_union() -> None:
  """
  WHAT: members of the `JsonPayload` type alias.
  WHY: the writer branches on text versus mapping; a third member would be handled by neither branch.
  EXPECTED: exactly `str` and one `Mapping` member, in a union type.
  """
  union = JsonPayload.__value__
  members = typing.get_args(union)

  assert typing.get_origin(union) is types.UnionType
  assert len(members) == 2
  assert members[0] is str


def test_json_payload_mapping_branch_accepts_any_value_type() -> None:
  """
  WHAT: the mapping member of `JsonPayload`.
  WHY: `Mapping` is covariant, so `dict[str, int]` is accepted where a `dict` parameter would reject it.
  EXPECTED: `collections.abc.Mapping[str, object]`.
  """
  mapping_branch = typing.get_args(JsonPayload.__value__)[1]

  assert typing.get_origin(mapping_branch) is Mapping
  assert typing.get_args(mapping_branch) == (str, object)


@pytest.mark.parametrize(
  "payload",
  [
    pytest.param("", id="empty-string"),
    pytest.param('{"population_size": 32}', id="serialized-json"),
    pytest.param({}, id="empty-mapping"),
    pytest.param({"local_search": {"class": "CPUTwoOpt"}}, id="nested-mapping"),
  ],
)
def test_json_payload_members_accept_the_values_the_writer_receives(payload: object) -> None:
  """
  WHAT: runtime membership of real payload values in one union member.
  WHY: the alias cannot be used with `isinstance`, so this reproduces the check the writer performs.
  EXPECTED: each value satisfies exactly one member type.
  """
  mapping_type = typing.get_origin(typing.get_args(JsonPayload.__value__)[1])

  assert isinstance(payload, str) or isinstance(payload, mapping_type)
