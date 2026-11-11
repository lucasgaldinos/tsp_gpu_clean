"""
Unit tests for Generation 0 raw evaluation and stage dispatch completeness (DP11-8, DP11-9, BUG-21).

Verifies that:
1. Every stage declared in `LocalSearchStage` has exactly one `StageSpec` in `STAGE_SPECS`
   and no unknown stages are registered.
2. Generation 0 initial_fitness is strictly raw and identical across all 5 stages
   for the same seed, confirming that post_evaluation no longer refines P_0 (BUG-21).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import (
  STAGE_SPECS,
  GeneticAlgorithm,
  LocalSearchStage,
  StageSpec,
)
from algorithms.strategies.crossover.order_crossover import CrossoverOrder
from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost
from algorithms.strategies.initialization.random_initialization import InitializationRandom
from algorithms.strategies.local_search.two_opt import LocalSearchCPUTwoOpt
from algorithms.strategies.mutation.swap_mutation import MutationSwap
from algorithms.strategies.selection.tournament_selection import SelectionTournament
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from data_models.problem import Problem


def test_stage_specs_completeness() -> None:
  """Verify that STAGE_SPECS covers get_args(LocalSearchStage.__value__) bijectively."""
  valid_stages = set(get_args(LocalSearchStage.__value__))
  registered_stages = set(STAGE_SPECS.keys())

  missing = valid_stages - registered_stages
  assert not missing, f"STAGE_SPECS is missing stages: {missing}"

  extra = registered_stages - valid_stages
  assert not extra, f"STAGE_SPECS has unregistered stages: {extra}"

  for stage, spec in STAGE_SPECS.items():
    assert isinstance(spec, StageSpec), f"STAGE_SPECS[{stage!r}] is not a StageSpec"
    assert callable(spec.pipeline_steps), f"STAGE_SPECS[{stage!r}].pipeline_steps is not callable"


@pytest.mark.parametrize(
  "stage",
  [
    "none",
    "post_crossover",
    "post_mutation",
    "post_evaluation",
    "post_evolution",
  ],
)
def test_initial_fitness_strictly_raw_across_all_stages(burma14_problem: Problem, stage: LocalSearchStage) -> None:
  """
  Verify initial_fitness is identical across all 5 stages for the same seed (BUG-21).

  Prior to BUG-21 fix, post_evaluation refined Generation 0 before loop start, causing
  its initial_fitness to be lower than all other stages.
  """
  seed = 42
  context = ProblemContext(burma14_problem, xp=np, seed=seed)
  customers = list(range(burma14_problem.dimension))

  ga = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.05,
    seed=seed,
    initialization=InitializationRandom(seed=seed),
    selection=SelectionTournament(k=3, seed=seed),
    crossover=CrossoverOrder(seed=seed),
    mutation=MutationSwap(seed=seed),
    evaluation=EvaluationCPUCost(),
    local_search=LocalSearchCPUTwoOpt(max_iterations=10),
    local_search_stage=stage,
  )

  _, stats = ga.evolve(context, customers, max_generations=1, patience=5)

  # Baseline raw evaluation of random population with fresh context
  baseline_context = ProblemContext(burma14_problem, xp=np, seed=seed)
  init_strat = InitializationRandom(seed=seed)
  eval_strat = EvaluationCPUCost()
  pop_raw = init_strat.initialize(customers, 16, baseline_context)
  fit_raw = eval_strat.evaluate_batch(pop_raw, baseline_context)
  expected_initial_raw = float(np.min(fit_raw))

  assert stats["initial_fitness"] == pytest.approx(expected_initial_raw, rel=1e-6)
