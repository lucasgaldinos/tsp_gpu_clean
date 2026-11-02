"""Boundary contracts for `GeneticAlgorithm.evolve`.

Pins the degenerate generation budgets: zero generations still initializes and
evaluates the population, while zero patience stops after exactly one generation.
An empty customer sequence contradicts `Problem`'s positive-dimension invariant, so
the GA is expected to reject it instead of returning a zero-cost empty tour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms import GeneticAlgorithm
from algorithms.strategies import (
  CPUCostEvaluation,
  OrderCrossover,
  RandomPermutationInitialization,
  SwapMutation,
  TournamentSelection,
)
from protocols import ProblemContext

if TYPE_CHECKING:
  from data_models.problem import Problem


def _make_ga(population_size: int = 8) -> GeneticAlgorithm:
  """Build a deterministic CPU GA with every operator explicitly seeded."""
  return GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=0.05,
    seed=42,
    initialization=RandomPermutationInitialization(seed=42),
    selection=TournamentSelection(k=3, seed=42),
    crossover=OrderCrossover(seed=42),
    mutation=SwapMutation(seed=42),
    evaluation=CPUCostEvaluation(),
  )


def test_zero_generations_returns_best_initial_route(burma14_problem: Problem) -> None:
  """WHAT: `evolve(..., max_generations=0)` on burma14.

  WHY: Benchmarks may disable evolution for a baseline; the call must still produce a
  valid tour and report the initial evaluation instead of an empty history.

  EXPECTED: the best initial tour, zero completed generations, no early stop, and a
  single-entry convergence history equal to the initial fitness.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))

  best_tour, stats = _make_ga().evolve(context, customers, max_generations=0)

  assert best_tour.shape == (burma14_problem.dimension,)
  assert sorted(best_tour.tolist()) == customers
  assert stats["generations_completed"] == 0
  assert stats["stopped_early"] is False
  assert stats["stop_reason"] == "max_generations"
  assert stats["best_fitness"] == pytest.approx(stats["initial_fitness"])
  assert stats["convergence_history"] == pytest.approx([stats["initial_fitness"]])


def test_zero_patience_stops_after_one_generation(burma14_problem: Problem) -> None:
  """WHAT: `evolve(..., max_generations=5, patience=0)` on burma14.

  WHY: Patience is checked after the first generation; a zero window must stop there
  rather than silently falling back to the full generation budget.

  EXPECTED: one completed generation, an early stop flagged as `no_improvements`,
  and a two-entry convergence history.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))

  best_tour, stats = _make_ga().evolve(context, customers, max_generations=5, patience=0)

  assert best_tour.shape == (burma14_problem.dimension,)
  assert sorted(best_tour.tolist()) == customers
  assert stats["generations_completed"] == 1
  assert stats["stopped_early"] is True
  assert stats["stop_reason"] == "no_improvements"
  assert len(stats["convergence_history"]) == 2


def test_empty_customers_are_rejected(burma14_problem: Problem) -> None:
  """WHAT: `evolve` called with an empty customer sequence.

  WHY: An empty instance cannot be solved and would report a fictitious perfect cost,
  masking an upstream loader or filtering error.

  EXPECTED: `ValueError` naming the invalid customer sequence (currently absent;
  registered as a defect).
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)

  with pytest.raises(ValueError, match="customers"):
    _make_ga().evolve(context, [], max_generations=1)
