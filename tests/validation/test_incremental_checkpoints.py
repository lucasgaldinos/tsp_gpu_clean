#!/usr/bin/env python3
"""
Validation tests for incremental checkpoint mode in benchmarking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms import GeneticAlgorithmHybridOptimized
from benchmarking.runner.algorithm_runner import run_single_algorithm
from data_models.problem import Problem
from loaders.database_loader import DatabaseLoader


def test_fresh_and_incremental_run() -> None:
  """Verify fresh run followed by incremental run accumulation."""
  with DatabaseLoader() as loader:
    problem: Problem = loader.load("burma14")

  algorithm = GeneticAlgorithmHybridOptimized(
    population_size=32,
    mutation_rate=0.02,
    tournament_size=5,
    two_opt_iterations=5,
    seed=42,
  )

  fresh_results: dict[str, Any] = run_single_algorithm(
    algorithm_name="HybridOptimized",
    algorithm_instance=algorithm,
    problem=problem,
    problem_name="burma14",
    problem_size=problem.dimension,
    customers=list(range(problem.dimension)),
    max_generations=20,
    optimal_cost=3323.0,
    repetitions=2,
    use_gpu=True,
    patience=8,
    existing_checkpoint=None,
    incremental_mode=False,
  )

  assert fresh_results["successful_runs"] == 2
  assert len(fresh_results["raw_costs"]) == 2
  assert len(fresh_results["raw_seeds"]) == 2

  combined_results: dict[str, Any] = run_single_algorithm(
    algorithm_name="HybridOptimized",
    algorithm_instance=algorithm,
    problem=problem,
    problem_name="burma14",
    problem_size=problem.dimension,
    customers=list(range(problem.dimension)),
    max_generations=20,
    optimal_cost=3323.0,
    repetitions=2,
    use_gpu=True,
    patience=8,
    existing_checkpoint=fresh_results,
    incremental_mode=True,
  )

  assert combined_results["successful_runs"] == 4
  assert len(combined_results["raw_costs"]) == 4
  assert len(combined_results["raw_seeds"]) == 4
  assert combined_results["raw_costs"][:2] == fresh_results["raw_costs"]
  assert np.isclose(combined_results["mean_cost"], np.mean(combined_results["raw_costs"]))
