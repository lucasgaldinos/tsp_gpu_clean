"""
Unit tests for the phase-timing and transfer telemetry of `run_single_algorithm` (DP11-7, BUG-30).

Verifies that:
1. `phase_share_cv_pct` uses the sample standard deviation (`ddof=1`), not the
   population one, which understates dispersion on short cohorts.
2. An incremental resume carries `h2d_bytes` / `d2h_bytes` / `kernel_launches` from the
   checkpoint instead of reporting only the new slice as the whole cohort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.runner.algorithm_runner import (
  _build_summary_payload,
  _initialize_results_dict,
  _record_failure,
  compute_series_stats,
  run_single_algorithm,
)
from benchmarking.runner.checkpoint_io import CheckpointManager
from benchmarking.runner.orchestration import _save_problem_statistics_file

if TYPE_CHECKING:
  from pathlib import Path

  from data_models.problem import Problem


@dataclass
class _ScriptedAlgorithm:
  """Algorithm double returning one scripted `stats` payload per repetition."""

  dimension: int
  stats_per_run: list[dict[str, Any]]
  evolve_calls: int = field(default=0)

  def evolve(
    self,
    context: Any,
    customers: Any,
    max_generations: int,
    optimal_cost: float | None = None,
    patience: int | None = None,
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the next scripted payload and a deterministic tour."""
    del context, customers, max_generations, optimal_cost, patience
    stats = self.stats_per_run[self.evolve_calls]
    self.evolve_calls += 1
    return np.arange(self.dimension, dtype=np.int32), stats


def _stats(
  *,
  evaluation_total_ms: float,
  h2d_bytes: int = 0,
  d2h_bytes: int = 0,
  kernel_launches: int = 0,
  initial_fitness: float = 100.0,
  best_fitness: float = 50.0,
) -> dict[str, Any]:
  """Build the subset of `evolve` stats the runner aggregates."""
  return {
    "initial_fitness": initial_fitness,
    "best_fitness": best_fitness,
    "generations_completed": 5,
    "stop_reason": "completed",
    "phase_timings": {
      "initialization": {"total_ms": 1.0, "calls": 1},
      "evaluation": {"total_ms": evaluation_total_ms, "calls": 5},
    },
    "h2d_bytes": h2d_bytes,
    "d2h_bytes": d2h_bytes,
    "kernel_launches": kernel_launches,
  }


def test_phase_share_cv_uses_sample_standard_deviation(burma14_problem: Problem) -> None:
  """Verify `phase_share_cv_pct` is `std(shares, ddof=1) / mean(shares)`.

  WHY: `DP11-7` asks for the dispersion of the per-run shares; `ddof=0` shrinks it by
  `sqrt((n - 1) / n)`, which is exactly the error that matters on the 2-3 run cohorts
  used by the sub-100 ms cells of Table 4.
  """
  algorithm = _ScriptedAlgorithm(
    dimension=burma14_problem.dimension,
    stats_per_run=[_stats(evaluation_total_ms=10.0), _stats(evaluation_total_ms=30.0)],
  )

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="ScriptedCPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=2,
    seed_base=42,
    use_gpu=False,
  )

  shares = [
    float(phase_timings["evaluation"]["total_ms"]) / (float(wall) * 1000.0) * 100.0
    for phase_timings, wall in zip(summary["raw_phase_timings"], summary["raw_times"], strict=True)
  ]
  sample_cv = float(np.std(shares, ddof=1)) / float(np.mean(shares)) * 100.0
  population_cv = float(np.std(shares, ddof=0)) / float(np.mean(shares)) * 100.0

  assert sample_cv != pytest.approx(population_cv)
  assert summary["phase_share_cv_pct"]["evaluation"] == pytest.approx(sample_cv)
  assert summary["phase_share_pct"]["evaluation"] == pytest.approx(float(np.mean(shares)))


def test_incremental_mode_carries_transfer_and_launch_telemetry(burma14_problem: Problem) -> None:
  """Verify resumed runs report the whole cohort's transfers, not just the new slice.

  WHY: `BUG-30` — `incremental_mode` rebuilds `results` and the transfer/launch lists
  start empty, so a 2+2 resume published the new runs' means as if they described all
  four. The checkpoint stores per-run means, so absolute totals are reconstructed from
  `mean_h2d_mb * successful_runs` and re-averaged over old + new runs.
  """
  algorithm = _ScriptedAlgorithm(
    dimension=burma14_problem.dimension,
    stats_per_run=[
      _stats(evaluation_total_ms=10.0, h2d_bytes=3000, d2h_bytes=300, kernel_launches=30),
      _stats(evaluation_total_ms=12.0, h2d_bytes=3000, d2h_bytes=300, kernel_launches=30),
    ],
  )
  existing_checkpoint: dict[str, Any] = {
    "algorithm": "ScriptedCPUAlgorithm",
    "repetitions": 2,
    "successful_runs": 2,
    "mean_time": 0.05,
    "mean_cost": 100.0,
    "mean_h2d_mb": 1e-3,
    "mean_d2h_mb": 1e-4,
    "mean_kernels": 5.0,
    "raw_times": [0.05, 0.05],
    "raw_initial_costs": [120.0, 120.0],
    "raw_costs": [100.0, 100.0],
    "raw_gaps": [5.0, 5.0],
    "raw_seeds": [42, 43],
    "raw_generations": [5, 5],
    "raw_stop_reasons": ["completed", "completed"],
    "raw_phase_timings": [{}, {}],
    "raw_best_tours": [[0] * 14, [0] * 14],
  }

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="ScriptedCPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=2,
    seed_base=42,
    use_gpu=False,
    incremental_mode=True,
    existing_checkpoint=existing_checkpoint,
  )

  # Old runs: 1000 B each; new runs: 3000 B each -> mean over 4 runs = 2000 B = 2e-3 MB.
  assert summary["mean_h2d_mb"] == pytest.approx(2e-3)
  assert summary["mean_d2h_mb"] == pytest.approx(2e-4)
  assert summary["total_transfer_mb"] == pytest.approx(2.2e-3)
  # Old runs: 5 launches; new runs: 30 -> 17.5 mean over 4 runs.
  assert summary["mean_kernels"] == pytest.approx(17.5)
  assert summary["successful_runs"] == 4


def test_algorithm_runner_sample_std_uses_bessels_correction(burma14_problem: Problem) -> None:
  """WHAT: 3 repetitions with varied costs [100.0, 110.0, 120.0].

  WHY: BUG-38 — Standard deviation of benchmark samples must use Bessel's correction
  (ddof=1), avoiding population underestimation. For single runs, std must be 0.0.
  EXPECTED: std_cost is sqrt(200 / 2) = 10.0, not sqrt(200 / 3) ≈ 8.165.
  """
  algorithm = _ScriptedAlgorithm(
    dimension=burma14_problem.dimension,
    stats_per_run=[
      _stats(evaluation_total_ms=10.0, best_fitness=100.0),
      _stats(evaluation_total_ms=10.0, best_fitness=110.0),
      _stats(evaluation_total_ms=10.0, best_fitness=120.0),
    ],
  )

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="ScriptedCPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=3,
    seed_base=42,
    use_gpu=False,
  )

  # Sample std dev with ddof=1: sqrt(((-10)^2 + 0^2 + 10^2) / (3 - 1)) = sqrt(200 / 2) = 10.0
  assert summary["mean_cost"] == pytest.approx(110.0)
  assert summary["std_cost"] == pytest.approx(10.0)
  assert summary["std_cost"] != pytest.approx(np.sqrt(200.0 / 3.0))

  # Guard: 1 repetition must yield std=0.0 (no division by zero)
  single_algo = _ScriptedAlgorithm(
    dimension=burma14_problem.dimension,
    stats_per_run=[_stats(evaluation_total_ms=10.0, best_fitness=100.0)],
  )
  single_summary = run_single_algorithm(
    algorithm_instance=single_algo,
    algorithm_name="ScriptedCPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=1,
    seed_base=42,
    use_gpu=False,
  )
  assert single_summary["std_cost"] == 0.0
  assert single_summary["std_time"] == 0.0


def test_runner_helpers_dry_refactoring() -> None:
  """WHAT: Unit test compute_series_stats, _initialize_results_dict, and _record_failure.

  WHY: Task 12.23 (DP12-16) — Verify DRY helper functions maintain statistical rigor and schemas.
  EXPECTED: Bessel's correction applied for N>1, 0.0 for N<=1 or empty/all-NaN; proper dict keys initialized.
  """
  # 1. compute_series_stats with N=0
  assert compute_series_stats(np.array([]), 0) == (0.0, 0.0, 0.0, 0.0)

  # 2. compute_series_stats with all-NaN array
  assert compute_series_stats(np.array([np.nan, np.nan]), 2) == (0.0, 0.0, 0.0, 0.0)

  # 3. compute_series_stats with [np.nan, 42.0] edge (single valid observation)
  mean_edge, std_edge, min_edge, max_edge = compute_series_stats(np.array([np.nan, 42.0]), 2)
  assert mean_edge == 42.0
  assert std_edge == 0.0
  assert min_edge == 42.0
  assert max_edge == 42.0

  # 4. compute_series_stats with N=1
  mean, std, min_v, max_v = compute_series_stats(np.array([42.0]), 1)
  assert mean == 42.0
  assert std == 0.0
  assert min_v == 42.0
  assert max_v == 42.0

  # 5. compute_series_stats with N=3 and Bessel's correction
  arr = np.array([100.0, 110.0, 120.0])
  mean, std, min_v, max_v = compute_series_stats(arr, 3)
  assert mean == pytest.approx(110.0)
  assert std == pytest.approx(10.0)
  assert min_v == 100.0
  assert max_v == 120.0

  # 6. _initialize_results_dict
  empty_results = _initialize_results_dict()
  assert "times" in empty_results
  assert "seeds" in empty_results
  assert len(empty_results["times"]) == 0

  checkpoint = {
    "successful_runs": 2,
    "raw_times": [1.0, 2.0],
    "raw_costs": [100.0, 90.0],
    "raw_gaps": [5.0, 2.0],
    "raw_generations": [50, 60],
    "raw_stop_reasons": ["completed", "stagnation"],
    "raw_phase_timings": [{}, {}],
    "raw_best_tours": [[0, 1], [1, 0]],
    "raw_seeds": [42, 43],
  }
  resumed = _initialize_results_dict(checkpoint)
  assert resumed["times"] == [1.0, 2.0]
  assert resumed["final_costs"] == [100.0, 90.0]
  assert len(resumed["initial_costs"]) == 2
  assert all(np.isnan(c) for c in resumed["initial_costs"])

  # 7. _record_failure
  _record_failure(empty_results)
  assert np.isnan(empty_results["times"][0])
  assert empty_results["stop_reasons"] == ["error"]
  assert empty_results["generations"] == [0]
  assert empty_results["best_tours"] == [None]


def test_save_problem_statistics_file_uses_compute_series_stats(tmp_path: Path) -> None:
  """WHAT: Test _save_problem_statistics_file in orchestration.py.

  WHY: Task 12.23 (DP12-16) — Ensure orchestration reuses compute_series_stats and handles
  empty raw_initial_costs -> 0.0 without crash or NaN.
  EXPECTED: Valid JSON stats file written via CheckpointManager; mean_initial_cost and std_initial_cost are 0.0.
  """
  checkpoint_dir = tmp_path / "checkpoints"
  stats_dir = tmp_path / "problem_stats"
  cm = CheckpointManager(checkpoint_dir=checkpoint_dir, problem_stats_dir=stats_dir)

  problem_results = {
    "GA_Test": {
      "raw_times": [1.0, 2.0],
      "raw_costs": [100.0, 110.0],
      "raw_gaps": [5.0, 15.0],
      "raw_generations": [50.0, 60.0],
      # raw_initial_costs is intentionally omitted / empty
      "raw_initial_costs": [],
      "backend": "gpu",
    }
  }

  _save_problem_statistics_file(
    problem_name="test_prob",
    problem_size=50,
    optimal_cost=100.0,
    problem_results=problem_results,
    statistical_test_results={"friedman": "none"},
    checkpoint_manager=cm,
  )

  stats_path = cm.get_problem_stats_path("test_prob")
  assert stats_path.exists()

  with open(stats_path, encoding="utf-8") as f:
    data = json.load(f)

  assert data["problem_info"]["name"] == "test_prob"
  assert data["problem_info"]["size"] == 50
  assert "GA_Test" in data["algorithms"]
  alg_stats = data["algorithms"]["GA_Test"]

  # Time: mean=(1.0+2.0)/2 = 1.5, std=sqrt(((1-1.5)^2+(2-1.5)^2)/1) = sqrt(0.5) ≈ 0.7071
  assert alg_stats["mean_time"] == pytest.approx(1.5)
  assert alg_stats["std_time"] == pytest.approx(float(np.std([1.0, 2.0], ddof=1)))

  # Initial costs empty: compute_series_stats returns 0.0
  assert alg_stats["mean_initial_cost"] == 0.0
  assert alg_stats["std_initial_cost"] == 0.0

  # Costs: mean=105.0, std=sqrt(50) ≈ 7.071
  assert alg_stats["mean_cost"] == pytest.approx(105.0)
  assert alg_stats["std_cost"] == pytest.approx(float(np.std([100.0, 110.0], ddof=1)))


def test_incremental_resume_missing_raw_initial_costs_does_not_fabricate_costs(burma14_problem: Problem) -> None:
  """WHAT: Test _build_summary_payload when raw_initial_costs is missing from checkpoint.

  WHY: Task 12.32 (BUG-48 / DP12-25) — When resuming from an older checkpoint without
  raw_initial_costs, initializing initial_costs as raw_costs fabricated a 0% improvement
  and caused division warnings under -W error. Missing initial costs must be padded with NaN
  and evaluated using safe masked division without RuntimeWarning.
  EXPECTED: mean_initial_cost is 0.0, mean_improvement is 0.0, and execution raises no warning under -W error.
  """
  checkpoint = {
    "successful_runs": 2,
    "raw_times": [1.0, 2.0],
    "raw_costs": [100.0, 90.0],
    "raw_gaps": [5.0, 2.0],
    "raw_generations": [50, 60],
    "raw_stop_reasons": ["completed", "stagnation"],
    "raw_phase_timings": [{}, {}],
    "raw_best_tours": [[0, 1], [1, 0]],
    "raw_seeds": [42, 43],
  }
  results = _initialize_results_dict(checkpoint)
  assert len(results["initial_costs"]) == 2
  assert all(np.isnan(c) for c in results["initial_costs"])

  summary = _build_summary_payload(results, burma14_problem, "TestAlgo")
  assert summary["mean_initial_cost"] == 0.0
  assert summary["std_initial_cost"] == 0.0
  assert summary["mean_improvement"] == 0.0
  assert all(np.isnan(c) for c in summary["raw_initial_costs"])

  # Now simulate one subsequent successful repetition with known initial cost 200.0 and final cost 100.0
  results["times"].append(1.5)
  results["initial_costs"].append(200.0)
  results["final_costs"].append(100.0)
  results["gaps"].append(0.0)
  results["improvements"].append(50.0)
  results["generations"].append(40)
  results["stop_reasons"].append("completed")
  results["phase_timings"].append({})
  results["h2d_bytes"].append(0)
  results["d2h_bytes"].append(0)
  results["kernel_launches"].append(0)
  results["best_tours"].append([0, 1])
  results["seeds"].append(44)

  summary2 = _build_summary_payload(results, burma14_problem, "TestAlgo")
  assert summary2["mean_initial_cost"] == pytest.approx(200.0)
  assert summary2["mean_improvement"] == pytest.approx(50.0)


def test_shorter_raw_initial_costs_padding_and_zero_initial_cost(burma14_problem: Problem) -> None:
  """WHAT: Test padding when raw_initial_costs is shorter than raw_costs, and zero initial cost.

  WHY: Review Gate 22c181dd — Checkpoints with partially populated raw_initial_costs must be
  padded with NaN up to len(raw_costs). Initial cost <= 0 must not trigger ZeroDivisionError or
  RuntimeWarning under -W error.
  """
  checkpoint = {
    "successful_runs": 3,
    "raw_times": [1.0, 2.0, 3.0],
    "raw_costs": [100.0, 90.0, 80.0],
    "raw_gaps": [5.0, 2.0, 1.0],
    "raw_generations": [50, 60, 70],
    "raw_stop_reasons": ["completed", "completed", "completed"],
    "raw_phase_timings": [{}, {}, {}],
    "raw_best_tours": [[0, 1], [0, 1], [0, 1]],
    "raw_seeds": [42, 43, 44],
    "raw_initial_costs": [200.0],
  }
  results = _initialize_results_dict(checkpoint)
  assert len(results["initial_costs"]) == 3
  assert results["initial_costs"][0] == 200.0
  assert np.isnan(results["initial_costs"][1])
  assert np.isnan(results["initial_costs"][2])

  # Append run with initial_cost = 0.0 (edge case)
  results["times"].append(1.0)
  results["initial_costs"].append(0.0)
  results["final_costs"].append(50.0)
  results["gaps"].append(0.0)
  results["improvements"].append(np.nan)
  results["generations"].append(10)
  results["stop_reasons"].append("completed")
  results["phase_timings"].append({})
  results["h2d_bytes"].append(0)
  results["d2h_bytes"].append(0)
  results["kernel_launches"].append(0)
  results["best_tours"].append([0, 1])
  results["seeds"].append(45)

  summary = _build_summary_payload(results, burma14_problem, "TestAlgo")
  assert summary["mean_improvement"] == pytest.approx(50.0)
