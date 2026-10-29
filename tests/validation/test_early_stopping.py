#!/usr/bin/env python3
"""
Quick test of early stopping on berlin52 (small problem) across all variants.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from algorithms import (
  GeneticAlgorithmCPU,
  GeneticAlgorithmFullGPUEarlyStop,
  GeneticAlgorithmFullGPUModular,
  GeneticAlgorithmHybridNaive,
  GeneticAlgorithmHybridOptimized,
)
from data_models.problem import Problem
from loaders.database_loader import DatabaseLoader
from protocols.problem_context import ProblemContext


def test_early_stopping_berlin52() -> None:
  """Verify early stopping and standardized stats across all GA variants on berlin52."""
  with DatabaseLoader("datasets/routing.duckdb") as loader:
    problem: Problem = loader.load("berlin52")

  optimal_cost = 7542.0
  max_generations = 80
  patience = 14
  customers: list[int] = list(range(problem.dimension))

  algorithms = [
    ("CPU", GeneticAlgorithmCPU, False),
    ("HybridNaive", GeneticAlgorithmHybridNaive, True),
    ("HybridOptimized", GeneticAlgorithmHybridOptimized, True),
    ("FullGPUModular", GeneticAlgorithmFullGPUModular, True),
    ("FullGPUEarlyStop", GeneticAlgorithmFullGPUEarlyStop, True),
  ]

  for name, algo_cls, use_gpu in algorithms:
    xp = cp if use_gpu else np
    context = ProblemContext(problem, xp=xp, seed=42)
    ga = algo_cls(
      population_size=64,
      mutation_rate=0.02,
      tournament_size=5,
      two_opt_iterations=5,
      seed=42,
    )
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
