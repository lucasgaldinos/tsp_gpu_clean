#!/usr/bin/env python3
"""
Quick test of early stopping on berlin52 (small problem) across all variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.runner.config_loader import load_algorithms_config, load_problems_config
from protocols.backend import cupy_backend, gpu_available
from protocols.problem_context import ProblemContext

GPU_AVAILABLE = gpu_available()
cp = cupy_backend()

if TYPE_CHECKING:
  from pathlib import Path

  from data_models.problem import Problem


@pytest.mark.slow
def test_early_stopping_berlin52(berlin52_problem: Problem, fixture_config_dir: Path) -> None:
  """Verify early stopping and standardized stats across all GA variants on berlin52."""
  problem: Problem = berlin52_problem

  optimal_cost = next(cfg.optimal for cfg in load_problems_config(fixture_config_dir) if cfg.name == "berlin52")
  max_generations = 80
  patience = 14
  customers: list[int] = list(range(problem.dimension))

  ga_params, alg_configs = load_algorithms_config(fixture_config_dir)

  for name, cfg in alg_configs.items():
    if cfg.use_gpu and not GPU_AVAILABLE:
      continue
    xp: Any = cp if cfg.use_gpu else np
    context = ProblemContext(problem, xp=xp, seed=42)
    ga = cfg.instantiate(ga_params, seed=42)
    best_tour, stats = ga.evolve(
      context,
      customers,
      max_generations,
      optimal_cost=optimal_cost,
      patience=patience,
    )
    assert len(best_tour) == problem.dimension, f"{name} tour length mismatch"
    assert sorted(best_tour.tolist()) == customers, f"{name} invalid tour permutation"
    assert stats["best_fitness"] <= stats["initial_fitness"] + 1e-6
    assert 1 <= stats["generations_completed"] <= max_generations
    assert stats["stop_reason"] in {"hit_optimal", "within_1pct_optimal", "no_improvements", "max_generations"}
