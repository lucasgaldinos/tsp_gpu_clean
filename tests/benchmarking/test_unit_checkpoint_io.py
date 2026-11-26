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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.runner.algorithm_runner import run_single_algorithm
from benchmarking.runner.checkpoint_io import (
  CheckpointManager,
  can_resume_checkpoint,
  get_remaining_repetitions,
  is_valid_checkpoint,
  load_checkpoint,
  quarantine_checkpoint,
  save_checkpoint_atomic,
  validate_existing_checkpoints,
)
from benchmarking.runner.config_loader import AlgorithmConfig, GAParams, ProblemConfig

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


def test_checkpoint_manager_count_completed_checkpoints_threshold_behavior(tmp_path: Path) -> None:
  """WHAT: count_completed_checkpoints includes CPU on all instances when cpu_size_threshold is None,
  and skips CPU for problem_size > cpu_size_threshold when integer is provided.

  WHY: Task 13.26 (BUG-81 / DP13-1) — DoD requirement for None vs integer threshold behavior.
  """
  manager = CheckpointManager(tmp_path / "checkpoints", tmp_path / "problem_stats")
  problems = [
    {"name": "small_prob", "size": 50},
    {"name": "large_prob", "size": 500},
  ]
  algorithms = ["CPU", "GPU_Hybrid"]

  # Case 1: cpu_size_threshold=None (run on all instances) -> 2 problems * 2 algorithms = 4 total
  completed, total = manager.count_completed_checkpoints(
    problems=problems,
    algorithms=algorithms,
    expected_reps=EXPECTED_REPS,
    cpu_size_threshold=None,
  )
  assert total == 4
  assert completed == 0

  # Case 2: cpu_size_threshold=100 (large_prob skips CPU) -> 2 (small) + 1 (large, GPU only) = 3 total
  completed_gated, total_gated = manager.count_completed_checkpoints(
    problems=problems,
    algorithms=algorithms,
    expected_reps=EXPECTED_REPS,
    cpu_size_threshold=100,
  )
  assert total_gated == 3
  assert completed_gated == 0

  # When small_prob CPU checkpoint exists:
  _write_json(manager.get_checkpoint_path("small_prob", "CPU"), _payload(EXPECTED_REPS))
  comp_none, _ = manager.count_completed_checkpoints(
    problems=problems,
    algorithms=algorithms,
    expected_reps=EXPECTED_REPS,
    cpu_size_threshold=None,
  )
  assert comp_none == 1

  comp_gated, _ = manager.count_completed_checkpoints(
    problems=problems,
    algorithms=algorithms,
    expected_reps=EXPECTED_REPS,
    cpu_size_threshold=100,
  )
  assert comp_gated == 1

  # Case 3: When algorithm_configs is supplied, non-GPU variants are identified by use_gpu (not hardcoded 'CPU')
  from benchmarking.runner.config_loader import AlgorithmConfig

  custom_configs = {
    "BaselineNumPy": AlgorithmConfig(
      name="BaselineNumPy",
      metaheuristic="GeneticAlgorithm",
      use_gpu=False,
      description="Non-GPU baseline",
      params={},
      strategies={},
    ),
    "AcceleratedGPU": AlgorithmConfig(
      name="AcceleratedGPU",
      metaheuristic="GeneticAlgorithm",
      use_gpu=True,
      description="GPU variant",
      params={},
      strategies={},
    ),
  }
  _, custom_total = manager.count_completed_checkpoints(
    problems=problems,
    algorithms=["BaselineNumPy", "AcceleratedGPU"],
    expected_reps=EXPECTED_REPS,
    cpu_size_threshold=100,
    algorithm_configs=custom_configs,
  )
  assert custom_total == 3


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


def test_checkpoint_valid_with_consistent_raw_phase_timings(tmp_path: Path) -> None:
  """Verify checkpoint with consistent raw_phase_timings is valid."""
  payload = _payload(3)
  payload["raw_phase_timings"] = [
    {"initialization": {"total_ms": 1.0, "calls": 1}},
    {"initialization": {"total_ms": 1.1, "calls": 1}},
    {"initialization": {"total_ms": 0.9, "calls": 1}},
  ]
  path = _write_json(tmp_path / "valid_phase_timings.json", payload)
  assert is_valid_checkpoint(path) is True


def test_checkpoint_rejects_inconsistent_raw_phase_timings(tmp_path: Path) -> None:
  """Verify checkpoint with raw_phase_timings length mismatch is rejected."""
  payload = _payload(3)
  payload["raw_phase_timings"] = [
    {"initialization": {"total_ms": 1.0, "calls": 1}},
  ]  # Length 1 != successful_runs 3
  path = _write_json(tmp_path / "bad_phase_timings.json", payload)
  assert is_valid_checkpoint(path) is False


def test_quarantine_corrupt_checkpoint(tmp_path: Path) -> None:
  """WHAT: Quarantining a corrupted or truncated checkpoint file.

  WHY: BUG-37 — Interrupted runs leave invalid JSON that crashes subsequent runs.
  EXPECTED: File renamed to <path>.corrupt.<timestamp>.bak, original removed,
  and content preserved in the quarantined file.
  """
  corrupt_file = tmp_path / "berlin52_CPU.json"
  corrupt_file.write_text('{"algorithm": "CPU", "successful_runs":', encoding="utf-8")

  fixed_ts = 1731350000
  quarantined = quarantine_checkpoint(corrupt_file, timestamp=fixed_ts)

  assert not corrupt_file.exists()
  assert quarantined.exists()
  assert quarantined.name == "berlin52_CPU.json.corrupt.1731350000.bak"
  assert quarantined.read_text(encoding="utf-8") == '{"algorithm": "CPU", "successful_runs":'


def test_incremental_mode_generates_strictly_new_seeds_without_collision(
  burma14_problem: Any,
) -> None:
  """WHAT: Resuming incremental runs generates strictly new seeds.

  WHY: BUG-37 — Offset calculation based on raw seed indices prevents
  duplicate seeds between initial and resumed benchmark runs.
  EXPECTED: Seeds in the resumed slice do not overlap with existing_checkpoint seeds.
  """

  @dataclass
  class _MockAlgo:
    dimension: int
    calls: int = field(default=0)

    def evolve(
      self,
      context: Any,
      customers: Any,
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, customers, max_generations, optimal_cost, patience
      self.calls += 1
      stats = {
        "initial_fitness": 100.0,
        "best_fitness": 90.0,
        "generations_completed": 5,
        "stop_reason": "completed",
      }
      return np.arange(self.dimension, dtype=np.int32), stats

  existing_checkpoint: dict[str, Any] = {
    "algorithm": "MockAlgo",
    "repetitions": 3,
    "successful_runs": 3,
    "mean_time": 0.05,
    "mean_cost": 90.0,
    "raw_times": [0.05, 0.05, 0.05],
    "raw_initial_costs": [100.0, 100.0, 100.0],
    "raw_costs": [90.0, 90.0, 90.0],
    "raw_gaps": [5.0, 5.0, 5.0],
    "raw_seeds": [42, 43, 44],
    "raw_generations": [5, 5, 5],
    "raw_stop_reasons": ["completed", "completed", "completed"],
    "raw_phase_timings": [{}, {}, {}],
    "raw_best_tours": [[0] * 14] * 3,
  }

  algorithm = _MockAlgo(dimension=burma14_problem.dimension)
  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="MockAlgo",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=3,
    seed_base=42,
    use_gpu=False,
    incremental_mode=True,
    existing_checkpoint=existing_checkpoint,
  )

  # Seeds should be [42, 43, 44, 45, 46, 47]
  assert summary["raw_seeds"] == [42, 43, 44, 45, 46, 47]
  assert len(summary["raw_seeds"]) == len(set(summary["raw_seeds"]))


def test_per_repetition_checkpoint_callback_and_timing_isolation(
  burma14_problem: Any,
) -> None:
  """WHAT: on_repetition_completed invoked after each repetition with timing isolation.

  WHY: Task 12.14 (DP12-7) — Checkpoint callback must receive running aggregates
  and disk I/O / callback duration must NOT pollute measured evolution time.
  EXPECTED:
  - Callback invoked 3 times with rep_num 1, 2, 3.
  - Intermediate summary in callback has exact online running aggregates.
  - Artificial 50ms callback delay does NOT inflate raw_times (< 20ms).
  """

  @dataclass
  class _FastMockAlgo:
    dimension: int

    def evolve(
      self,
      context: Any,
      customers: Any,
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, customers, max_generations, optimal_cost, patience
      stats = {
        "initial_fitness": 100.0,
        "best_fitness": 80.0,
        "generations_completed": 10,
        "stop_reason": "completed",
      }
      return np.arange(self.dimension, dtype=np.int32), stats

  invocations: list[tuple[int, dict[str, Any]]] = []

  def _callback(rep_num: int, current_summary: dict[str, Any]) -> None:
    # Deliberate 50ms latency (simulating slow disk I/O / fsync)
    time.sleep(0.05)
    invocations.append((rep_num, dict(current_summary)))

  algorithm = _FastMockAlgo(dimension=burma14_problem.dimension)
  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="FastMock",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=10,
    repetitions=3,
    seed_base=100,
    use_gpu=False,
    on_repetition_completed=_callback,
  )

  # Exactly 3 invocations
  assert len(invocations) == 3
  assert [inv[0] for inv in invocations] == [1, 2, 3]

  # Online running aggregates check
  for rep_num, intermediate in invocations:
    assert intermediate["successful_runs"] == rep_num
    assert intermediate["repetitions"] == rep_num
    assert len(intermediate["raw_times"]) == rep_num
    assert len(intermediate["raw_costs"]) == rep_num
    assert len(intermediate["raw_seeds"]) == rep_num
    if rep_num == 1:
      assert intermediate["std_time"] == 0.0
      assert intermediate["std_cost"] == 0.0
    else:
      # Bessel's correction applied
      assert "std_cost" in intermediate

  # Timing isolation check:
  # Without isolation, each repetition would take > 0.05s (50ms).
  # With strict isolation, fast mock runs in < 0.02s.
  for measured_time in summary["raw_times"]:
    assert measured_time < 0.035, f"Measured time {measured_time}s indicates callback I/O leakage!"


def test_atomic_checkpoint_persistence_on_interrupted_run(
  burma14_problem: Any,
  tmp_path: Path,
) -> None:
  """WHAT: Partial in-flight checkpoint resumes seamlessly to completion (DP12-7).

  WHY: Task 12.14 — When a run is interrupted at K < N repetitions, checkpoint
  on disk must be valid and resumable, executing only the remaining N - K runs.
  EXPECTED: Partial checkpoint has successful_runs=2; resume reaches successful_runs=5.
  """

  @dataclass
  class _MockAlgo:
    dimension: int

    def evolve(
      self,
      context: Any,
      customers: Any,
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, customers, max_generations, optimal_cost, patience
      stats = {
        "initial_fitness": 120.0,
        "best_fitness": 95.0,
        "generations_completed": 5,
        "stop_reason": "completed",
      }
      return np.arange(self.dimension, dtype=np.int32), stats

  checkpoint_file = tmp_path / "burma14_MockAlgo.json"

  def _save_callback(rep_num: int, current_summary: dict[str, Any]) -> None:
    payload = dict(current_summary)
    payload["problem_name"] = "burma14"
    payload["problem_size"] = burma14_problem.dimension
    payload["optimal_cost"] = 30.0
    payload["backend"] = "CPU"
    payload["timestamp"] = "2026-11-12T12:00:00"
    payload["strategies"] = {}
    payload["algorithm_config"] = {"strategy_params": {}}
    payload["construction_heuristic"] = "Random"
    payload["problem_type"] = "TSP"
    payload["edge_type"] = "EUC_2D"
    save_checkpoint_atomic(checkpoint_file, payload)

  algorithm = _MockAlgo(dimension=burma14_problem.dimension)

  # Phase 1: Simulate interrupted execution after 2 of 5 runs
  run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="MockAlgo",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=2,
    seed_base=1000,
    use_gpu=False,
    on_repetition_completed=_save_callback,
  )

  # Checkpoint exists on disk and is resumable
  assert checkpoint_file.exists()
  loaded_partial = load_checkpoint(checkpoint_file)
  assert loaded_partial["successful_runs"] == 2
  assert can_resume_checkpoint(checkpoint_file, expected_reps=5) is True
  assert is_valid_checkpoint(checkpoint_file, expected_reps=5) is False

  # Phase 2: Resume remaining 3 runs to reach target 5
  remaining_reps = get_remaining_repetitions(checkpoint_file, expected_reps=5)
  assert remaining_reps == 3

  completed_summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="MockAlgo",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=remaining_reps,
    seed_base=1000,
    use_gpu=False,
    incremental_mode=True,
    existing_checkpoint=loaded_partial,
    on_repetition_completed=_save_callback,
  )

  assert completed_summary["successful_runs"] == 5
  assert len(completed_summary["raw_seeds"]) == 5
  assert len(set(completed_summary["raw_seeds"])) == 5  # No seed collision

  final_loaded = load_checkpoint(checkpoint_file)
  assert final_loaded["successful_runs"] == 5
  assert is_valid_checkpoint(checkpoint_file, expected_reps=5) is True


# ============================================================================
# validate_existing_checkpoints (Task 13.25 / BUG-79)
# ============================================================================


def _modular_checkpoint_payload(
  *,
  problem_name: str = "gr17",
  algorithm: str = "CPU",
  population_size: int = 32,
  mutation_rate: float = 0.05,
  seed: int = 42,
  local_search_class: str = "LocalSearchCPUTwoOpt",
) -> dict[str, Any]:
  return {
    "problem_name": problem_name,
    "algorithm": algorithm,
    "successful_runs": 2,
    "raw_costs": [2085.0, 2090.0],
    "raw_times": [0.1, 0.12],
    "strategies": {
      "initialization": "InitializationRandom",
      "selection": "SelectionTournament",
      "crossover": "CrossoverOrder",
      "mutation": "MutationSwap",
      "local_search": local_search_class,
      "evaluation": "EvaluationCPUCost",
    },
    "algorithm_config": {
      "population_size": population_size,
      "mutation_rate": mutation_rate,
      "seed": seed,
    },
  }


def _sample_algo_configs() -> dict[str, AlgorithmConfig]:
  return {
    "CPU": AlgorithmConfig(
      name="CPU",
      use_gpu=False,
      description="CPU",
      strategies={
        "initialization": {"class": "InitializationRandom"},
        "selection": {"class": "SelectionTournament"},
        "crossover": {"class": "CrossoverOrder"},
        "mutation": {"class": "MutationSwap"},
        "local_search": {"class": "LocalSearchCPUTwoOpt"},
        "evaluation": {"class": "EvaluationCPUCost"},
      },
    ),
    "HybridOptimized": AlgorithmConfig(
      name="HybridOptimized",
      use_gpu=True,
      description="GPU",
      strategies={
        "initialization": {"class": "InitializationRandom"},
        "selection": {"class": "SelectionTournament"},
        "crossover": {"class": "CrossoverOrder"},
        "mutation": {"class": "MutationSwap"},
        "local_search": {"class": "LocalSearchGPUTwoOpt"},
        "evaluation": {"class": "EvaluationCPUCost"},
      },
    ),
  }


def test_validate_existing_checkpoints_accepts_matching_modular_payload(tmp_path: Path) -> None:
  """WHAT: validate_existing_checkpoints accepts a matching modular checkpoint.

  WHY: Task 13.25 (BUG-79) — Valid checkpoints must resume cleanly without false positives.
  """
  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  _write_json(ckpt_dir / "gr17_CPU.json", _modular_checkpoint_payload())

  errors = validate_existing_checkpoints(
    ckpt_dir,
    [ProblemConfig(name="gr17", optimal=2085.0, size=17)],
    ["CPU"],
    _sample_algo_configs(),
    GAParams(population_size=32, mutation_rate=0.05, seed=42),
  )
  assert errors == []


def test_validate_existing_checkpoints_rejects_old_schema_strategy_and_param_drift(tmp_path: Path) -> None:
  """WHAT: validate_existing_checkpoints rejects pre-modular schema, strategy drift, and GA param drift.

  WHY: Task 13.25 (BUG-79) — Stale checkpoints from an older architecture or parameter set
  must fail preflight before being resumed or reused.
  """
  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  problems = [ProblemConfig(name="gr17", optimal=2085.0, size=17)]
  algo_configs = _sample_algo_configs()
  ga_params = GAParams(population_size=32, mutation_rate=0.05, seed=42)

  # 1. Pre-modular schema missing 'strategies', 'algorithm_config', 'problem_name'
  _write_json(ckpt_dir / "gr17_CPU.json", _payload(2))
  old_errors = validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], algo_configs, ga_params)
  assert len(old_errors) == 1
  assert "missing required keys" in old_errors[0]

  # 2. Mismatched strategy class
  _write_json(
    ckpt_dir / "gr17_CPU.json",
    _modular_checkpoint_payload(local_search_class="LocalSearchGPUTwoOpt"),
  )
  strat_errors = validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], algo_configs, ga_params)
  assert any("strategies.local_search" in e for e in strat_errors)

  # 3. Mismatched algorithm_config params (population_size, mutation_rate, seed)
  _write_json(
    ckpt_dir / "gr17_CPU.json",
    _modular_checkpoint_payload(population_size=256, mutation_rate=0.02, seed=99),
  )
  param_errors = validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], algo_configs, ga_params)
  assert any("population_size" in e for e in param_errors)
  assert any("mutation_rate" in e for e in param_errors)
  assert any("seed" in e for e in param_errors)


def test_validate_existing_checkpoints_respects_cpu_size_threshold(tmp_path: Path) -> None:
  """WHAT: CPU checkpoints for instances above cpu_size_threshold are not inspected when skipped."""
  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  # Write an incompatible CPU checkpoint for a large problem
  _write_json(ckpt_dir / "rat783_CPU.json", _payload(2))

  errors = validate_existing_checkpoints(
    ckpt_dir,
    [ProblemConfig(name="rat783", optimal=8806.0, size=783)],
    ["CPU", "HybridOptimized"],
    _sample_algo_configs(),
    GAParams(population_size=32, mutation_rate=0.05, seed=42),
    cpu_size_threshold=400,
  )
  assert errors == []


def test_checkpoint_predicates_accept_null_raw_gaps_for_unknown_optimal(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Checkpoints without known optima (`raw_gaps: None`) pass validation and resume checks.

  WHY: BUG-96 — Direct `.tsp` loading emits `optimal_cost=None, raw_gaps=None`; calling
  `len(raw_gaps)` previously triggered a `TypeError` and reported valid checkpoints as corrupt.
  """
  payload = {
    "problem_name": "custom50",
    "problem_size": 50,
    "optimal_cost": None,
    "algorithm": "CPU",
    "repetitions": 5,
    "successful_runs": 5,
    "mean_time": 0.1,
    "mean_cost": 500.0,
    "mean_gap": None,
    "raw_times": [0.1] * 5,
    "raw_costs": [500.0] * 5,
    "raw_gaps": None,
    "raw_seeds": [42, 43, 44, 45, 46],
  }
  path = tmp_path / "custom50_CPU.json"
  save_checkpoint_atomic(path, payload)

  with caplog.at_level("WARNING"):
    assert is_valid_checkpoint(path, 5) is True
    assert can_resume_checkpoint(path, 10) is True
    assert get_remaining_repetitions(path, 5) == 0
    assert get_remaining_repetitions(path, 10) == 5
  assert "corrupted" not in caplog.text.lower()

  # Non-null raw_gaps with mismatched length must still be rejected
  bad_payload = dict(payload)
  bad_payload["raw_gaps"] = [1.0, 2.0]
  bad_path = _write_json(tmp_path / "custom50_bad_gaps.json", bad_payload)
  assert is_valid_checkpoint(bad_path, 5) is False
  assert can_resume_checkpoint(bad_path, 10) is False


def test_save_problem_statistics_derives_retroactive_gaps_and_handles_summary_only(tmp_path: Path) -> None:
  """WHAT: `save_problem_statistics` derives gaps from `raw_costs` + retroactive `optimal_cost`.

  WHY: BUG-96 / BUG-99 — Checkpoints originally collected without `optimal_cost` (`raw_gaps=None`)
  must populate `mean_gap`/`std_gap` when regenerated with an explicit `problems.json` optimum,
  and summary-only entries must preserve summary metrics instead of zeroing them.
  """
  from benchmarking.runner.checkpoint_io import save_problem_statistics

  cm = CheckpointManager(tmp_path / "checkpoints", tmp_path / "problem_statistics")
  save_problem_statistics(
    problem_name="custom_opt",
    problem_size=20,
    optimal_cost=400.0,
    problem_results={
      "RawNoGaps": {
        "raw_times": [0.5, 0.7],
        "raw_costs": [440.0, 440.0],
        "raw_gaps": None,
        "backend": "CPU",
      },
      "SummaryOnly": {
        "mean_time": 0.25,
        "std_time": 0.05,
        "mean_cost": 420.0,
        "std_cost": 4.0,
        "success_rate": 1.0,
        "backend": "GPU",
      },
    },
    statistical_test_results={},
    checkpoint_manager=cm,
  )
  saved = json.loads(cm.get_problem_stats_path("custom_opt").read_text(encoding="utf-8"))
  assert saved["algorithms"]["RawNoGaps"]["mean_gap"] == pytest.approx(10.0)
  assert saved["algorithms"]["RawNoGaps"]["std_gap"] == pytest.approx(0.0)
  assert saved["algorithms"]["SummaryOnly"]["mean_time"] == pytest.approx(0.25)
  assert saved["algorithms"]["SummaryOnly"]["mean_cost"] == pytest.approx(420.0)
  assert saved["algorithms"]["SummaryOnly"]["mean_gap"] == pytest.approx(5.0)
  assert saved["algorithms"]["SummaryOnly"]["success_rate"] == pytest.approx(1.0)


def test_validate_existing_checkpoints_detects_stage_strategy_params_pipeline_and_corrupt_mode(
  tmp_path: Path,
) -> None:
  """WHAT: Preflight catches `local_search_stage`, `strategy_params`, `pipeline`, and array drift.

  WHY: BUG-98 — Changing local-search stage or operator hyperparameters must fail preflight,
  while `allow_corrupt=True` lets malformed JSON proceed to execution-loop quarantine.
  """
  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  problems = [ProblemConfig(name="p1", optimal=100.0, size=50)]
  ga_params = GAParams(population_size=256, mutation_rate=0.02, seed=42)

  payload = {
    "problem_name": "p1",
    "algorithm": "CPU",
    "successful_runs": 2,
    "raw_costs": [100.0, 102.0],
    "raw_times": [1.0, 1.1],
    "strategies": {"local_search": "LocalSearchCPUTwoOpt"},
    "algorithm_config": {
      "population_size": 256,
      "mutation_rate": 0.02,
      "seed": 42,
      "local_search_stage": "post_mutation",
      "strategy_params": {
        "local_search": {"class": "LocalSearchCPUTwoOpt", "max_iterations": 10, "mode": "first"},
      },
    },
  }
  _write_json(ckpt_dir / "p1_CPU.json", payload)

  drifted_cfg = {
    "CPU": AlgorithmConfig(
      name="CPU",
      use_gpu=False,
      description="CPU",
      params={
        "local_search_stage": "post_evolution",
        "pipeline": [{"class": "ElitistLocalSearchStep", "params": {"elite_count": 4}}],
      },
      strategies={"local_search": {"class": "LocalSearchCPUTwoOpt", "params": {"max_iterations": 999, "mode": "best"}}},
    )
  }
  errors = validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], drifted_cfg, ga_params)
  assert any("local_search_stage" in e for e in errors)
  assert any("max_iterations" in e for e in errors)
  assert any("mode" in e for e in errors)
  assert any("pipeline" in e for e in errors)

  # CLI strategy_overrides drift also rejected
  matching_cfg = {
    "CPU": AlgorithmConfig(
      name="CPU",
      use_gpu=False,
      description="CPU",
      params={"local_search_stage": "post_mutation"},
      strategies={"local_search": {"class": "LocalSearchCPUTwoOpt", "params": {"max_iterations": 10, "mode": "first"}}},
    )
  }
  assert validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], matching_cfg, ga_params) == []
  override_errors = validate_existing_checkpoints(
    ckpt_dir,
    problems,
    ["CPU"],
    matching_cfg,
    ga_params,
    strategy_overrides={"local_search": {"max_iterations": 25}},
  )
  assert any("max_iterations" in e for e in override_errors)

  # Custom pipeline matching vs parameter drift inside pipeline step
  pipe_payload = json.loads((ckpt_dir / "p1_CPU.json").read_text(encoding="utf-8"))
  pipe_payload["algorithm_config"]["pipeline"] = [
    {"class": "ElitistLocalSearchStep", "params": {"elite_count": 4}},
  ]
  _write_json(ckpt_dir / "p1_CPU.json", pipe_payload)
  pipe_cfg = {
    "CPU": AlgorithmConfig(
      name="CPU",
      use_gpu=False,
      description="CPU",
      params={
        "local_search_stage": "post_mutation",
        "pipeline": [{"class": "ElitistLocalSearchStep", "params": {"elite_count": 4}}],
      },
      strategies={"local_search": {"class": "LocalSearchCPUTwoOpt", "params": {"max_iterations": 10, "mode": "first"}}},
    )
  }
  assert validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], pipe_cfg, ga_params) == []
  pipe_cfg["CPU"].params["pipeline"] = [{"class": "ElitistLocalSearchStep", "params": {"elite_count": 8}}]
  pipe_drift = validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], pipe_cfg, ga_params)
  assert any("pipeline[0].params.elite_count" in e for e in pipe_drift)

  # Corrupt JSON fails when allow_corrupt=False (dry-run) and defers when allow_corrupt=True
  (ckpt_dir / "p1_CPU.json").write_text("{truncated json", encoding="utf-8")
  corrupt_errors = validate_existing_checkpoints(
    ckpt_dir, problems, ["CPU"], matching_cfg, ga_params, allow_corrupt=False
  )
  assert len(corrupt_errors) == 1
  assert validate_existing_checkpoints(ckpt_dir, problems, ["CPU"], matching_cfg, ga_params, allow_corrupt=True) == []


def test_run_comprehensive_benchmark_quarantines_corrupt_checkpoint_and_runs_cleanly(tmp_path: Path) -> None:
  """WHAT: Truncated checkpoint JSON is quarantined on live runs instead of aborting preflight.

  WHY: BUG-98 / BUG-37 — Preflight previously raised before `quarantine_checkpoint` ran,
  the post-quarantine block fell into the resume branch instead of starting a clean run,
  and custom strategy overrides outside `_STRATEGY_PARAM_ATTRS` were omitted from metadata.
  """
  from pathlib import Path as _Path

  from algorithms import register_strategy
  from algorithms.strategies import STRATEGY_REGISTRY, LocalSearchCPUTwoOpt
  from benchmarking.runner.orchestration import run_comprehensive_benchmark

  class _CustomTwoOpt(LocalSearchCPUTwoOpt):
    def __init__(self, max_iterations: int = 10, custom_span: int = 2) -> None:
      super().__init__(max_iterations=max_iterations)
      self.custom_span = custom_span

  repo_root = _Path(__file__).resolve().parents[2]
  cfg_dir = repo_root / "tests" / "fixtures" / "configs"
  gr17_path = repo_root / "tests" / "fixtures" / "tsp" / "gr17.tsp"

  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir(parents=True)
  corrupt_ckpt = ckpt_dir / "gr17_CPU.json"
  corrupt_ckpt.write_text('{"problem_name": "gr17", "algorithm":', encoding="utf-8")

  # dry_run=True must reject corrupt JSON without touching disk
  with pytest.raises(ValueError, match="failed to read checkpoint"):
    run_comprehensive_benchmark(
      output_dir=tmp_path,
      config_dir=cfg_dir,
      tsp_files=[gr17_path],
      algorithms=["CPU"],
      repetitions=1,
      max_generations=2,
      dry_run=True,
    )
  assert corrupt_ckpt.exists()

  register_strategy("_CustomTwoOptTest", _CustomTwoOpt)
  try:
    custom_algos = {
      "CPU": AlgorithmConfig(
        name="CPU",
        use_gpu=False,
        description="Custom CPU",
        strategies={
          "initialization": {"class": "InitializationRandom"},
          "selection": {"class": "SelectionTournament", "params": {"k": 3}},
          "crossover": {"class": "CrossoverOrder"},
          "mutation": {"class": "MutationSwap"},
          "local_search": {"class": "_CustomTwoOptTest", "params": {"max_iterations": 2, "custom_span": 2}},
          "evaluation": {"class": "EvaluationCPUCost"},
        },
      )
    }
    # Live run must quarantine corrupt JSON, complete 1 clean rep, and persist custom override
    results = run_comprehensive_benchmark(
      output_dir=tmp_path,
      config_dir=cfg_dir,
      tsp_files=[gr17_path],
      algorithms=["CPU"],
      algorithm_configs_override=custom_algos,
      repetitions=1,
      max_generations=2,
      incremental_runs=False,
      strategy_overrides={"local_search": {"max_iterations": 3, "custom_span": 7}},
    )
  finally:
    STRATEGY_REGISTRY.pop("_CustomTwoOptTest", None)

  quarantined = list(ckpt_dir.glob("gr17_CPU.json.corrupt.*.bak"))
  assert len(quarantined) == 1
  assert results["gr17"]["CPU"]["successful_runs"] == 1
  saved = load_checkpoint(corrupt_ckpt)
  assert saved["successful_runs"] == 1
  assert saved["algorithm_config"]["strategy_params"]["local_search"]["max_iterations"] == 3
  assert saved["algorithm_config"]["strategy_params"]["local_search"]["custom_span"] == 7
