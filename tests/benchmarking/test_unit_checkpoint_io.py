"""Unit tests for checkpoint predicates in `benchmarking.runner.checkpoint_io`.

WHAT: Exercises `is_valid_checkpoint`, `can_resume_checkpoint` and
`get_remaining_repetitions` against JSON payloads written under `tmp_path`,
covering valid, missing-key, wrong-type, corrupt, completed and partially
completed checkpoints, plus the `CheckpointManager` path helpers.
WHY: Resume logic decides whether a run is skipped, extended or redone; a wrong
predicate silently truncates or duplicates a 30-repetition experiment.
EXPECTED: The documented truth table (valid / resumable / remaining) for every
payload shape, with no writes outside `tmp_path`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from benchmarking.runner.checkpoint_io import (
  CheckpointManager,
  can_resume_checkpoint,
  get_remaining_repetitions,
  is_valid_checkpoint,
)

if TYPE_CHECKING:
  from pathlib import Path

EXPECTED_REPS = 30


def _payload(successful_runs: int, array_length: int | None = None) -> dict[str, Any]:
  """Build a well-formed checkpoint payload.

  `array_length` defaults to `successful_runs`, producing internally consistent
  raw arrays; passing a different value creates a deliberately inconsistent file.
  """
  length = successful_runs if array_length is None else array_length
  return {
    "algorithm": "CPU",
    "repetitions": successful_runs,
    "successful_runs": successful_runs,
    "mean_time": 1.5,
    "mean_cost": 100.0,
    "raw_times": [1.5] * length,
    "raw_costs": [100.0] * length,
    "raw_gaps": [0.0] * length,
    "raw_seeds": list(range(length)),
  }


def _checkpoint_for_case(case: str) -> dict[str, Any]:
  """Return the payload for one truth-table case."""
  match case:
    case "complete":
      return _payload(EXPECTED_REPS)
    case "partial":
      return _payload(10)
    case "zero_runs":
      return _payload(0)
    case "missing_algorithm":
      payload = _payload(10)
      del payload["algorithm"]
      return payload
    case "missing_mean_time":
      payload = _payload(10)
      del payload["mean_time"]
      return payload
    case "wrong_type":
      payload = _payload(10)
      payload["successful_runs"] = "ten"
      return payload
    case "arrays_shorter_than_successful":
      return _payload(10, array_length=9)
    case "inconsistent_array_lengths":
      payload = _payload(10)
      payload["raw_costs"] = payload["raw_costs"][:9]
      return payload
    case "over_delivered":
      return _payload(10)
    case _:
      raise AssertionError(f"unknown checkpoint case: {case}")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
  """Write `payload` as JSON under `path` and return the path."""
  path.write_text(json.dumps(payload))
  return path


# ============================================================================
# Truth table
# ============================================================================


@pytest.mark.parametrize(
  ("case", "expected_reps", "expected_valid", "expected_resumable", "expected_remaining"),
  [
    pytest.param("complete", 30, True, False, 0, id="complete_run"),
    pytest.param("partial", 30, False, True, 20, id="partially_completed_run"),
    pytest.param("zero_runs", 30, False, True, 30, id="empty_run"),
    pytest.param("missing_algorithm", 30, False, False, 30, id="missing_required_key"),
    pytest.param("missing_mean_time", 30, False, True, 20, id="missing_mean_key"),
    pytest.param("wrong_type", 30, False, False, 30, id="wrong_field_type"),
    pytest.param("arrays_shorter_than_successful", 30, False, False, 30, id="arrays_shorter_than_runs"),
    pytest.param("inconsistent_array_lengths", 30, False, False, 30, id="raw_arrays_disagree"),
    pytest.param("over_delivered", 5, True, False, 0, id="more_runs_than_requested"),
  ],
)
def test_checkpoint_predicate_truth_table(
  tmp_path: Path,
  case: str,
  expected_reps: int,
  expected_valid: bool,
  expected_resumable: bool,
  expected_remaining: int,
) -> None:
  """WHAT: The three checkpoint predicates on one payload shape per case.

  WHY: Resume decisions combine these three answers; each combination maps to a
  different action (skip, extend, rerun).
  EXPECTED: see the parametrized table; `remaining` is 0 for complete files,
  `expected - successful_runs` for resumable ones, and `expected` otherwise.
  """
  # Arrange
  path = _write_json(tmp_path / f"{case}.json", _checkpoint_for_case(case))

  # Act
  valid = is_valid_checkpoint(path, expected_reps)
  resumable = can_resume_checkpoint(path, expected_reps)
  remaining = get_remaining_repetitions(path, expected_reps)

  # Assert
  assert valid is expected_valid, f"is_valid_checkpoint mismatch for {case}"
  assert resumable is expected_resumable, f"can_resume_checkpoint mismatch for {case}"
  assert remaining == expected_remaining, f"get_remaining_repetitions mismatch for {case}"


def test_missing_file_is_neither_valid_nor_resumable(tmp_path: Path) -> None:
  """WHAT: Predicates on a checkpoint path that does not exist.

  WHY: A fresh run has no file; every predicate must degrade to "run all".
  EXPECTED: False / False / 30.
  """
  # Arrange
  path = tmp_path / "absent.json"

  # Act / Assert
  assert is_valid_checkpoint(path, EXPECTED_REPS) is False
  assert can_resume_checkpoint(path, EXPECTED_REPS) is False
  assert get_remaining_repetitions(path, EXPECTED_REPS) == EXPECTED_REPS


def test_corrupt_json_is_neither_valid_nor_resumable(tmp_path: Path) -> None:
  """WHAT: A checkpoint containing invalid JSON.

  WHY: Atomic writes protect against this, but a crashed legacy run can leave a
  truncated file; it must be treated as absent.
  EXPECTED: False / False / 30, no exception.
  """
  # Arrange
  path = tmp_path / "corrupt.json"
  path.write_text("{not valid json")

  # Act / Assert
  assert is_valid_checkpoint(path, EXPECTED_REPS) is False
  assert can_resume_checkpoint(path, EXPECTED_REPS) is False
  assert get_remaining_repetitions(path, EXPECTED_REPS) == EXPECTED_REPS


@pytest.mark.parametrize(
  ("successful_runs", "expected_remaining"),
  [
    pytest.param(1, 29, id="one_done"),
    pytest.param(7, 23, id="seven_done"),
    pytest.param(29, 1, id="one_left"),
  ],
)
def test_remaining_repetitions_arithmetic(tmp_path: Path, successful_runs: int, expected_remaining: int) -> None:
  """WHAT: `expected_reps - successful_runs` for resumable checkpoints.

  WHY: The resume loop uses this count to request exactly the missing runs.
  EXPECTED: 29, 23 and 1 respectively.
  """
  # Arrange
  path = _write_json(tmp_path / "partial.json", _payload(successful_runs))

  # Act
  remaining = get_remaining_repetitions(path, EXPECTED_REPS)

  # Assert
  assert can_resume_checkpoint(path, EXPECTED_REPS) is True
  assert remaining == expected_remaining


def test_remaining_repetitions_never_negative(tmp_path: Path) -> None:
  """WHAT: A checkpoint with more runs than requested.

  WHY: Extending a longer run must not ask for a negative number of repetitions.
  EXPECTED: 0.
  """
  # Arrange
  path = _write_json(tmp_path / "long.json", _payload(12))

  # Act / Assert
  assert get_remaining_repetitions(path, 5) == 0


# ============================================================================
# CheckpointManager
# ============================================================================


def test_checkpoint_manager_builds_paths_and_delegates(tmp_path: Path) -> None:
  """WHAT: `CheckpointManager` directory creation, path naming and delegation.

  WHY: The runner addresses checkpoints exclusively through these paths; a naming
  drift orphans every previous checkpoint.
  EXPECTED: directories exist, `<problem>_<algorithm>.json` naming, and the
  manager predicates agree with the module functions.
  """
  # Arrange
  manager = CheckpointManager(tmp_path / "checkpoints", tmp_path / "problem_stats")
  path = manager.get_checkpoint_path("berlin52", "CPU")

  # Act
  _write_json(path, _payload(EXPECTED_REPS))

  # Assert
  assert (tmp_path / "checkpoints").is_dir()
  assert (tmp_path / "problem_stats").is_dir()
  assert path.name == "berlin52_CPU.json"
  assert manager.get_problem_stats_path("berlin52").name == "berlin52_stats.json"
  assert manager.is_valid_checkpoint("berlin52", "CPU", EXPECTED_REPS) is True
  assert manager.can_resume_checkpoint("berlin52", "CPU", EXPECTED_REPS) is False
  assert manager.get_remaining_repetitions("berlin52", "CPU", EXPECTED_REPS) == 0


def test_validity_rejects_arrays_inconsistent_with_successful_runs(tmp_path: Path) -> None:
  """WHAT: `is_valid_checkpoint` on `successful_runs=30` with empty raw arrays.

  WHY: `can_resume_checkpoint` treats array-length consistency as a hard
  invariant; the completeness check must apply the same rule, otherwise data-less
  checkpoint rows are imported as complete runs.
  EXPECTED: False.
  """
  # Arrange
  path = _write_json(tmp_path / "empty_arrays.json", _payload(EXPECTED_REPS, array_length=0))

  # Act
  valid = is_valid_checkpoint(path, EXPECTED_REPS)

  # Assert
  assert valid is False


def test_predicates_use_the_checkpoint_own_repetitions_when_no_target_is_given(tmp_path: Path) -> None:
  """WHAT: a finished 3-repetition checkpoint judged without an explicit target.

  WHY: `benchmark.json` configures 15 repetitions and `selected_10` drives 3, so a
  hard-coded default of 30 (`BUG-17`) reported every real checkpoint as incomplete
  and could never mark a short run as done.
  EXPECTED: valid, not resumable, and 0 remaining against its own target.
  """
  # Arrange
  path = _write_json(tmp_path / "selected_10.json", _payload(3))

  # Act / Assert
  assert is_valid_checkpoint(path) is True
  assert can_resume_checkpoint(path) is False
  assert get_remaining_repetitions(path, 3) == 0


@pytest.mark.parametrize(
  "repetitions",
  [pytest.param(0, id="zero"), pytest.param(-1, id="negative"), pytest.param("three", id="string")],
)
def test_predicates_reject_an_unusable_repetitions_field(tmp_path: Path, repetitions: object) -> None:
  """WHAT: checkpoints whose `repetitions` field cannot express a target.

  WHY: inferring the target from the file is only sound when the field is a
  positive integer; otherwise the file must be rejected rather than silently
  accepted as complete.
  EXPECTED: both predicates answer False.
  """
  # Arrange
  payload = _payload(3)
  payload["repetitions"] = repetitions
  path = _write_json(tmp_path / "bad_reps.json", payload)

  # Act / Assert
  assert is_valid_checkpoint(path) is False
  assert can_resume_checkpoint(path) is False
