"""
Unit tests for the phase-timing and transfer telemetry of `run_single_algorithm` (DP11-7, BUG-30).

Verifies that:
1. `phase_share_cv_pct` uses the sample standard deviation (`ddof=1`), not the
   population one, which understates dispersion on short cohorts.
2. An incremental resume carries `h2d_bytes` / `d2h_bytes` / `kernel_launches` from the
   checkpoint instead of reporting only the new slice as the whole cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.runner.algorithm_runner import run_single_algorithm

if TYPE_CHECKING:
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
) -> dict[str, Any]:
  """Build the subset of `evolve` stats the runner aggregates."""
  return {
    "initial_fitness": 100.0,
    "best_fitness": 50.0,
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
    for phase_timings, wall in zip(
      summary["raw_phase_timings"], summary["raw_times"], strict=True
    )
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
