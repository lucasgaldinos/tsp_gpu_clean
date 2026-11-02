#!/usr/bin/env python3
"""
Validation tests for incremental checkpoint mode in benchmarking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.runner.algorithm_runner import run_single_algorithm
from benchmarking.runner.config_loader import load_algorithms_config

if TYPE_CHECKING:
  from pathlib import Path

  from data_models.problem import Problem


@pytest.mark.slow
def test_fresh_and_incremental_run(burma14_problem: Problem, fixture_config_dir: Path) -> None:
  """Verify fresh run followed by incremental run accumulation."""
  problem: Problem = burma14_problem

  ga_params, alg_configs = load_algorithms_config(fixture_config_dir)
  algorithm = alg_configs["CPU"].instantiate(ga_params, seed=42)

  fresh_results: dict[str, Any] = run_single_algorithm(
    algorithm_name="CPU",
    algorithm_instance=algorithm,
    problem=problem,
    problem_name="burma14",
    problem_size=problem.dimension,
    customers=list(range(problem.dimension)),
    max_generations=20,
    optimal_cost=3323.0,
    repetitions=2,
    use_gpu=False,
    patience=8,
    existing_checkpoint=None,
    incremental_mode=False,
  )

  assert fresh_results["successful_runs"] == 2
  assert len(fresh_results["raw_costs"]) == 2
  assert len(fresh_results["raw_seeds"]) == 2

  combined_results: dict[str, Any] = run_single_algorithm(
    algorithm_name="CPU",
    algorithm_instance=algorithm,
    problem=problem,
    problem_name="burma14",
    problem_size=problem.dimension,
    customers=list(range(problem.dimension)),
    max_generations=20,
    optimal_cost=3323.0,
    repetitions=2,
    use_gpu=False,
    patience=8,
    existing_checkpoint=fresh_results,
    incremental_mode=True,
  )

  assert combined_results["successful_runs"] == 4
  assert len(combined_results["raw_costs"]) == 4
  assert len(combined_results["raw_seeds"]) == 4
  assert combined_results["raw_costs"][:2] == fresh_results["raw_costs"]
  assert np.isclose(combined_results["mean_cost"], float(np.mean(combined_results["raw_costs"])))
