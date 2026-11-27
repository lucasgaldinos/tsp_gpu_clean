"""
Unit tests for the conditional best-tour transfer in `GeneticAlgorithm.evolve` (BUG-30).

Verifies that the winning route crosses device-to-host only when a generation strictly
improves the incumbent, instead of being copied on every generation. The counter-check
is a pipeline whose offspring is always discarded, so the population is frozen and the
improvement count is exactly zero by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.metaheuristics.genetic_algorithm import (
  EvaluationStep,
  GeneticAlgorithm,
  SelectionStep,
  SurvivalStep,
)
from algorithms.strategies.crossover.order_crossover import CrossoverOrder
from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost
from algorithms.strategies.initialization.random_init import InitializationCPURandom
from algorithms.strategies.mutation.swap_mutation import MutationSwap
from algorithms.strategies.selection.tournament_selection import SelectionTournament
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  import pytest

  from data_models.problem import Problem


def test_best_tour_is_transferred_only_on_strict_improvement(
  burma14_problem: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verify `_extract_route_at` runs once (generation 0) when no generation improves.

  WHY: `_extract_route_at` performs the `asnumpy` device-to-host copy of the winning route.
  Doing it every generation moves `(n,)` int32 across PCIe for a route that is then
  thrown away; the frozen-population pipeline below makes "no improvement" a property
  of the configuration, so the call count is deterministic — and the previous
  one-call-per-generation behaviour fails the assertion by `max_generations`.
  """
  extract_calls: list[int] = []
  original_extract = GeneticAlgorithm._extract_route_at

  def _spy(population: Any, idx: int) -> np.ndarray:
    extract_calls.append(1)
    return original_extract(population, idx)

  monkeypatch.setattr(GeneticAlgorithm, "_extract_route_at", staticmethod(_spy))

  evaluator = EvaluationCPUCost()
  selection = SelectionTournament(k=3, seed=42)
  ga = GeneticAlgorithm(
    population_size=8,
    mutation_rate=0.0,
    seed=42,
    initialization=InitializationCPURandom(seed=42),
    selection=selection,
    crossover=CrossoverOrder(seed=42),
    mutation=MutationSwap(seed=42),
    evaluation=evaluator,
    local_search=None,
    local_search_stage="none",
  )

  def _keep_parents(
    old_pop: Any, old_fitness: Any, new_pop: Any, new_fitness: Any
  ) -> tuple[Any, Any]:
    """Survivor selector that freezes the population: offspring never enters."""
    return old_pop, old_fitness

  ga.pipeline = [
    SelectionStep(selection, ga.population_size),
    EvaluationStep(evaluator, target="offspring"),
    SurvivalStep(_keep_parents),
  ]
  context = ProblemContext(burma14_problem, xp=np, seed=42)

  _, stats = ga.evolve(
    context,
    list(range(burma14_problem.dimension)),
    max_generations=4,
    patience=100,
  )

  assert stats["generations_completed"] == 4
  assert extract_calls == [1], "only generation 0 may materialize the winning route"
