"""
Unit tests for composable pipeline steps (DP11-6, DP11-10).

Covers:
- `PostInitLocalSearchStep`: Generation 1 refinement of P_0, call count, timing isolation,
  and contract comparison with non-pipeline variant.
- `EliteOnlyLocalSearchStep`: Top-k route refinement, permutation validity, non-worsening
  invariant, and call counting.
- `PeriodicLocalSearchStep`: Interval-based execution counting, target modes, and parameter
  validation.
- `PIPELINE_STEP_REGISTRY` and `AlgorithmConfig.instantiate` configuration resolution and error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import (
  GenerationState,
  GenerationStep,
  GeneticAlgorithm,
)
from algorithms.pipeline_steps import (
  PIPELINE_STEP_REGISTRY,
  EliteOnlyLocalSearchStep,
  PeriodicLocalSearchStep,
  PostInitLocalSearchStep,
  StageAnchor,
)
from algorithms.strategies.crossover.order_crossover import CrossoverOrder
from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost
from algorithms.strategies.initialization.random_initialization import InitializationRandom
from algorithms.strategies.local_search.two_opt import LocalSearchCPUTwoOpt
from algorithms.strategies.mutation.swap_mutation import MutationSwap
from algorithms.strategies.selection.tournament_selection import SelectionTournament
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  GAParams,
  load_algorithms_config,
)
from benchmarking.runner.orchestration import _serialize_custom_pipeline
from protocols.problem_context import ProblemContext
from utils.phase_timer import PhaseTimer

if TYPE_CHECKING:
  from data_models.problem import Problem
  from protocols.algorithm_strategies import LocalSearchStrategy


# =============================================================================
# Registry & Schema Tests
# =============================================================================


def test_pipeline_step_registry_contents() -> None:
  """Verify PIPELINE_STEP_REGISTRY contains the expected pipeline step classes."""
  expected = {
    "PostInitLocalSearchStep": PostInitLocalSearchStep,
    "EliteOnlyLocalSearchStep": EliteOnlyLocalSearchStep,
    "PeriodicLocalSearchStep": PeriodicLocalSearchStep,
  }
  assert PIPELINE_STEP_REGISTRY == expected


def test_instantiate_unknown_step_class_raises_value_error() -> None:
  """Verify unknown step class raises ValueError naming the unknown class."""
  cfg = AlgorithmConfig(
    name="test_unknown",
    metaheuristic="GeneticAlgorithm",
    use_gpu=False,
    description="test",
    params={
      "pipeline": [
        {"class": "NonExistentStepClass"},
      ],
    },
    strategies={
      "initialization": {"class": "InitializationRandom"},
      "selection": {"class": "SelectionTournament"},
      "crossover": {"class": "CrossoverOrder"},
      "mutation": {"class": "MutationSwap"},
      "evaluation": {"class": "EvaluationCPUCost"},
      "local_search": {"class": "LocalSearchCPUTwoOpt"},
    },
  )
  with pytest.raises(ValueError, match="Unknown pipeline step class 'NonExistentStepClass'"):
    cfg.instantiate(GAParams())


def test_instantiate_step_with_invalid_param_raises_value_error() -> None:
  """Verify unrecognized parameter for a pipeline step raises ValueError naming accepted parameters."""
  cfg = AlgorithmConfig(
    name="test_invalid_param",
    metaheuristic="GeneticAlgorithm",
    use_gpu=False,
    description="test",
    params={
      "pipeline": [
        {"class": "PostInitLocalSearchStep", "params": {"bogus_parameter": 123}},
      ],
    },
    strategies={
      "initialization": {"class": "InitializationRandom"},
      "selection": {"class": "SelectionTournament"},
      "crossover": {"class": "CrossoverOrder"},
      "mutation": {"class": "MutationSwap"},
      "evaluation": {"class": "EvaluationCPUCost"},
      "local_search": {"class": "LocalSearchCPUTwoOpt"},
    },
  )
  with pytest.raises(ValueError, match="does not accept parameter 'bogus_parameter'"):
    cfg.instantiate(GAParams())


# =============================================================================
# PostInitLocalSearchStep Tests (DP11-6)
# =============================================================================


def test_post_init_step_validation() -> None:
  """Verify PostInitLocalSearchStep rejects None for required strategies."""
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  with pytest.raises(ValueError, match="requires a local_search strategy"):
    PostInitLocalSearchStep(local_search=None, evaluation=eval_strat)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="requires an evaluation strategy"):
    PostInitLocalSearchStep(local_search=ls_strat, evaluation=None)  # type: ignore[arg-type]


def test_post_init_step_runs_only_on_generation_1(burma14_problem: Problem) -> None:
  """Verify PostInitLocalSearchStep runs improve_batch only on generation 1 and skips generation > 1."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  timer = PhaseTimer(enabled=True)
  eval_strat = EvaluationCPUCost()

  mock_ls = MagicMock()
  mock_ls.improve_batch.side_effect = lambda pop, ctx: pop.copy()

  step = PostInitLocalSearchStep(local_search=mock_ls, evaluation=eval_strat)

  pop = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (8, 1))
  fit = eval_strat.evaluate_batch(pop, context)
  state = GenerationState(population=pop, fitness=fit, context=context, generation=0)

  # Generation 0: skip
  step.run(state, timer)
  assert mock_ls.improve_batch.call_count == 0

  # Generation 1: execute exactly once
  state.generation = 1
  step.run(state, timer)
  assert mock_ls.improve_batch.call_count == 1

  # Generation 2..10: skip
  for gen in range(2, 11):
    state.generation = gen
    step.run(state, timer)
  assert mock_ls.improve_batch.call_count == 1


def test_post_init_contract_identical_initial_fitness_different_best_fitness(burma14_problem: Problem) -> None:
  """
  DoD verification: Two variants with same initialization and seed produce IDENTICAL initial_fitness,
  but DIFFERENT best_fitness, when one declares pipeline: [PostInitLocalSearchStep] and the other does not.

  Refinement timing appears in phase_timings['local_search'] of generation 1 and not in 'initialization'.
  improve_batch is called exactly once in 10 generations.
  """
  seed = 42
  customers = list(range(burma14_problem.dimension))

  # 1. Variant WITHOUT pipeline (local_search_stage="none")
  ga_plain = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.02,
    seed=seed,
    initialization=InitializationRandom(seed=seed),
    selection=SelectionTournament(k=3, seed=seed),
    crossover=CrossoverOrder(seed=seed),
    mutation=MutationSwap(seed=seed),
    evaluation=EvaluationCPUCost(),
    local_search=LocalSearchCPUTwoOpt(max_iterations=10),
    local_search_stage="none",
  )

  # 2. Variant WITH pipeline [PostInitLocalSearchStep]
  post_init_step = PostInitLocalSearchStep(
    local_search=LocalSearchCPUTwoOpt(max_iterations=10),
    evaluation=EvaluationCPUCost(),
  )
  ga_pipelined = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.02,
    seed=seed,
    initialization=InitializationRandom(seed=seed),
    selection=SelectionTournament(k=3, seed=seed),
    crossover=CrossoverOrder(seed=seed),
    mutation=MutationSwap(seed=seed),
    evaluation=EvaluationCPUCost(),
    local_search=LocalSearchCPUTwoOpt(max_iterations=10),
    local_search_stage="none",
    pipeline=[post_init_step],
  )

  ctx1 = ProblemContext(burma14_problem, xp=np, seed=seed)
  ctx2 = ProblemContext(burma14_problem, xp=np, seed=seed)

  _, stats_plain = ga_plain.evolve(ctx1, customers, max_generations=10, patience=20)
  _, stats_pipelined = ga_pipelined.evolve(ctx2, customers, max_generations=10, patience=20)

  # Identical initial fitness (raw Generation 0 population)
  assert stats_plain["initial_fitness"] == pytest.approx(stats_pipelined["initial_fitness"], rel=1e-6)

  # Different best fitness because P_0 was refined by 2-opt in generation 1
  assert stats_pipelined["best_fitness"] < stats_plain["best_fitness"] - 1e-3

  # Local search ran in pipeline: exactly 1 call recorded under local_search
  assert "local_search" in stats_pipelined["phase_timings"]
  assert stats_pipelined["phase_timings"]["local_search"]["calls"] == 1
  assert stats_pipelined["phase_timings"]["local_search"]["total_ms"] > 0.0

  # Plain variant ran 0 local search calls
  assert "local_search" not in stats_plain["phase_timings"]

  # Standard GA pipeline steps were preserved and executed every generation
  for phase in ("selection", "crossover", "mutation", "evaluation", "host_sync"):
    assert phase in stats_pipelined["phase_timings"]
    assert stats_pipelined["phase_timings"][phase]["calls"] >= 10


# =============================================================================
# EliteOnlyLocalSearchStep Tests (DP11-10)
# =============================================================================


def test_elite_only_step_validation() -> None:
  """Verify EliteOnlyLocalSearchStep constructor validations."""
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  with pytest.raises(ValueError, match="requires a local_search strategy"):
    EliteOnlyLocalSearchStep(local_search=None, evaluation=eval_strat)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="requires an evaluation strategy"):
    EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=None)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="elite_count must be > 0"):
    EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=0)

  with pytest.raises(ValueError, match="elite_count must be > 0"):
    EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=-5)


def test_elite_only_step_refines_only_elites_preserving_non_elites(burma14_problem: Problem) -> None:
  """
  Verify EliteOnlyLocalSearchStep:
  - Slices and refines only the top elite_count routes.
  - Leaves the remaining non-elite routes strictly untouched.
  - Invariant: fitness of refined routes never worsens.
  - Permutation invariant: all routes remain valid permutations.
  """
  seed = 42
  context = ProblemContext(burma14_problem, xp=np, seed=seed)
  timer = PhaseTimer(enabled=True)
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt(max_iterations=10, mode="best")

  elite_count = 3
  pop_size = 10
  step = EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=elite_count)

  customers = list(range(burma14_problem.dimension))
  init_strat = InitializationRandom(seed=seed)
  pop = init_strat.initialize(customers, pop_size, context)
  fit = eval_strat.evaluate_batch(pop, context)

  state = GenerationState(population=pop.copy(), fitness=fit.copy(), context=context, generation=1)

  elite_indices = np.argsort(state.fitness)[:elite_count]
  non_elite_indices = np.argsort(state.fitness)[elite_count:]

  orig_non_elite_routes = state.population[non_elite_indices].copy()
  orig_non_elite_fitness = state.fitness[non_elite_indices].copy()
  orig_elite_fitness = state.fitness[elite_indices].copy()

  step.run(state, timer)

  # Non-elites must be untouched
  np.testing.assert_array_equal(state.population[non_elite_indices], orig_non_elite_routes)
  np.testing.assert_array_equal(state.fitness[non_elite_indices], orig_non_elite_fitness)

  # Elites must not worsen
  assert np.all(state.fitness[elite_indices] <= orig_elite_fitness + 1e-6)

  # All routes remain valid permutations
  for route in state.population:
    assert sorted(route.tolist()) == customers


def test_elite_only_step_call_count_in_10_generations(burma14_problem: Problem) -> None:
  """Verify EliteOnlyLocalSearchStep refines elite_count routes each generation across 10 generations."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  timer = PhaseTimer(enabled=True)
  eval_strat = EvaluationCPUCost()

  mock_ls = MagicMock()
  mock_ls.improve_batch.side_effect = lambda routes, ctx: routes.copy()

  elite_count = 4
  step = EliteOnlyLocalSearchStep(local_search=mock_ls, evaluation=eval_strat, elite_count=elite_count)

  pop = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (16, 1))
  fit = eval_strat.evaluate_batch(pop, context)
  state = GenerationState(population=pop, fitness=fit, context=context)

  for gen in range(1, 11):
    state.generation = gen
    step.run(state, timer)

  assert mock_ls.improve_batch.call_count == 10
  for call_args in mock_ls.improve_batch.call_args_list:
    passed_routes = call_args[0][0]
    assert passed_routes.shape[0] == elite_count


# =============================================================================
# PeriodicLocalSearchStep Tests (DP11-10)
# =============================================================================


def test_periodic_step_validation() -> None:
  """Verify PeriodicLocalSearchStep constructor validations."""
  ls_strat = LocalSearchCPUTwoOpt()

  with pytest.raises(ValueError, match="requires a local_search strategy"):
    PeriodicLocalSearchStep(local_search=None)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="interval must be > 0"):
    PeriodicLocalSearchStep(local_search=ls_strat, interval=0)

  with pytest.raises(ValueError, match="interval must be > 0"):
    PeriodicLocalSearchStep(local_search=ls_strat, interval=-2)

  with pytest.raises(ValueError, match="re_evaluate EvaluationStrategy is required"):
    PeriodicLocalSearchStep(local_search=ls_strat, interval=5, target="population", re_evaluate=None)

  with pytest.raises(ValueError, match="target must be 'offspring' or 'population'"):
    PeriodicLocalSearchStep(local_search=ls_strat, interval=5, target="invalid")  # type: ignore[arg-type]


def test_periodic_step_execution_counting_and_interval(burma14_problem: Problem) -> None:
  """
  Verify PeriodicLocalSearchStep fires only when generation % interval == 0.

  For interval=3 across 10 generations (1..10), it fires on generations 3, 6, 9 (3 times).
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  timer = PhaseTimer(enabled=True)
  eval_strat = EvaluationCPUCost()

  mock_ls = MagicMock()
  mock_ls.improve_batch.side_effect = lambda pop, ctx: pop.copy()

  interval = 3
  step = PeriodicLocalSearchStep(
    local_search=mock_ls,
    interval=interval,
    target="population",
    re_evaluate=eval_strat,
  )

  pop = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (8, 1))
  fit = eval_strat.evaluate_batch(pop, context)
  state = GenerationState(population=pop, fitness=fit, context=context)

  executed_generations: list[int] = []
  for gen in range(1, 11):
    state.generation = gen
    call_count_before = mock_ls.improve_batch.call_count
    step.run(state, timer)
    if mock_ls.improve_batch.call_count > call_count_before:
      executed_generations.append(gen)

  assert executed_generations == [3, 6, 9]
  assert mock_ls.improve_batch.call_count == 3


def test_periodic_step_offspring_target(burma14_problem: Problem) -> None:
  """Verify PeriodicLocalSearchStep with target='offspring' refines state.offspring."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  timer = PhaseTimer(enabled=True)
  ls_strat = LocalSearchCPUTwoOpt(max_iterations=5)

  step = PeriodicLocalSearchStep(local_search=ls_strat, interval=2, target="offspring")

  pop = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (8, 1))
  fit = np.ones(8, dtype=np.float64) * 100.0

  # Calling before offspring exists raises RuntimeError
  state = GenerationState(population=pop, fitness=fit, context=context, generation=2)
  with pytest.raises(RuntimeError, match="called before offspring exists"):
    step.run(state, timer)

  # With populated offspring, it refines offspring on gen 2
  state.offspring = pop.copy()
  step.run(state, timer)
  assert state.offspring is not None
  for route in state.offspring:
    assert sorted(route.tolist()) == list(range(burma14_problem.dimension))


# =============================================================================
# Full Pipeline Composition Tests (DP11-6, DP11-10)
# =============================================================================


def test_instantiate_and_evolve_from_config_runs_all_phases(burma14_problem: Problem) -> None:
  """
  Verify that an algorithm configured via JSON with a custom pipeline step
  (ExampleCPU_PostInitMemetic) instantiates properly, composes with standard GA steps,
  and executes all evolutionary phases (initialization, selection, crossover, mutation,
  evaluation, host_sync, and local_search).
  """
  config_path = Path("configs")
  ga_params, configs = load_algorithms_config(config_path, config_name="algorithms.full.example.json")
  assert "ExampleCPU_PostInitMemetic" in configs

  cfg = configs["ExampleCPU_PostInitMemetic"]
  ga = cfg.instantiate(ga_params, seed=42)

  # Check that the composed pipeline preserves all core evolutionary steps
  step_names = [type(s).__name__ for s in ga.pipeline]
  assert step_names[0] == "PostInitLocalSearchStep"
  assert "SelectionStep" in step_names
  assert "CrossoverStep" in step_names
  assert "MutationStep" in step_names
  assert "EvaluationStep" in step_names
  assert "SurvivalStep" in step_names

  customers = list(range(burma14_problem.dimension))
  ctx = ProblemContext(burma14_problem, xp=np, seed=42)

  _best_tour, stats = ga.evolve(ctx, customers, max_generations=5, patience=10)

  # Verify all phases executed
  timings = stats["phase_timings"]
  assert "initialization" in timings
  assert "host_sync" in timings
  assert "selection" in timings
  assert "crossover" in timings
  assert "mutation" in timings
  assert "evaluation" in timings
  assert "local_search" in timings

  # PostInit local search fired exactly once (on Generation 1)
  assert timings["local_search"]["calls"] == 1
  # Core phases fired across all completed generations
  gens = stats["generations_completed"]
  assert timings["selection"]["calls"] == gens
  assert timings["crossover"]["calls"] == gens
  assert timings["mutation"]["calls"] == gens


def test_elite_only_step_in_full_ga_evolution(burma14_problem: Problem) -> None:
  """
  Verify EliteOnlyLocalSearchStep composed in a full GA runs local search each generation
  while standard GA steps (selection, crossover, mutation) also run each generation.
  """
  seed = 42
  customers = list(range(burma14_problem.dimension))
  ctx = ProblemContext(burma14_problem, xp=np, seed=seed)

  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt(max_iterations=5, mode="best")
  elite_step = EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=3)

  ga = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.05,
    seed=seed,
    initialization=InitializationRandom(seed=seed),
    selection=SelectionTournament(k=3, seed=seed),
    crossover=CrossoverOrder(seed=seed),
    mutation=MutationSwap(seed=seed),
    evaluation=eval_strat,
    local_search=ls_strat,
    local_search_stage="none",
    pipeline=[elite_step],
  )

  # Check that elite_step is placed after SurvivalStep
  step_names = [type(s).__name__ for s in ga.pipeline]
  assert step_names[-1] == "EliteOnlyLocalSearchStep"
  assert "SelectionStep" in step_names
  assert "SurvivalStep" in step_names

  _, stats = ga.evolve(ctx, customers, max_generations=5, patience=10)

  timings = stats["phase_timings"]
  gens = stats["generations_completed"]
  assert gens == 5
  assert timings["local_search"]["calls"] == 5
  assert timings["selection"]["calls"] == 5
  assert timings["crossover"]["calls"] == 5
  assert timings["mutation"]["calls"] == 5


def test_periodic_step_in_full_ga_evolution(burma14_problem: Problem) -> None:
  """
  Verify PeriodicLocalSearchStep composed in a full GA runs local search only on
  scheduled generations (interval=3 in 10 generations -> 3 calls), while GA steps run all 10 gens.
  """
  seed = 42
  customers = list(range(burma14_problem.dimension))
  ctx = ProblemContext(burma14_problem, xp=np, seed=seed)

  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt(max_iterations=5, mode="best")
  periodic_step = PeriodicLocalSearchStep(
    local_search=ls_strat,
    interval=3,
    target="population",
    re_evaluate=eval_strat,
  )

  ga = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.05,
    seed=seed,
    initialization=InitializationRandom(seed=seed),
    selection=SelectionTournament(k=3, seed=seed),
    crossover=CrossoverOrder(seed=seed),
    mutation=MutationSwap(seed=seed),
    evaluation=eval_strat,
    local_search=ls_strat,
    local_search_stage="none",
    pipeline=[periodic_step],
  )

  _, stats = ga.evolve(ctx, customers, max_generations=10, patience=20)

  timings = stats["phase_timings"]
  gens = stats["generations_completed"]
  assert gens == 10
  # Fired on gen 3, 6, 9 -> exactly 3 calls
  assert timings["local_search"]["calls"] == 3
  assert timings["selection"]["calls"] == 10
  assert timings["crossover"]["calls"] == 10
  assert timings["mutation"]["calls"] == 10


# =============================================================================
# Declarative Stage Anchors, Pipeline Introspection & Checkpoint Serialization
# =============================================================================


def test_stage_anchor_on_all_pipeline_steps() -> None:
  """WHAT: every extension step declares where it belongs in the generation (BUG-30).

  WHY: `_compose_pipeline` used to match on class names, so a new step type silently
  landed at the end of the pipeline regardless of its semantics.
  EXPECTED: post-init refines before selection, elite-only after survival, and the
  periodic step follows its target (offspring -> before evaluation, population -> after survival).
  """
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  assert PostInitLocalSearchStep(local_search=ls_strat, evaluation=eval_strat).stage_anchor == "pre_selection"
  assert EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat).stage_anchor == "post_survival"
  assert (
    PeriodicLocalSearchStep(local_search=ls_strat, interval=5, target="offspring").stage_anchor == "pre_evaluation"
  )
  assert (
    PeriodicLocalSearchStep(local_search=ls_strat, interval=5, target="population", re_evaluate=eval_strat).stage_anchor
    == "post_survival"
  )


def test_public_params_expose_declared_knobs() -> None:
  """WHAT: `public_params()` returns the parameters that define each step's schedule.

  WHY: A checkpoint recording only the class name cannot be reproduced (`BUG-30`).
  EXPECTED: post-init has no parameters; elite-only exposes `elite_count`; periodic
  exposes `interval` and `target`.
  """
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  assert PostInitLocalSearchStep(local_search=ls_strat, evaluation=eval_strat).public_params() == {}
  assert EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=3).public_params() == {
    "elite_count": 3
  }
  assert PeriodicLocalSearchStep(
    local_search=ls_strat, interval=7, target="population", re_evaluate=eval_strat
  ).public_params() == {"interval": 7, "target": "population"}


def test_strategies_reports_pipeline_owned_strategies() -> None:
  """WHAT: `strategies()` exposes exactly the strategy objects each step executes.

  WHY: Extension steps can own the only local search instance in the run
  (`local_search_stage="none"`), so telemetry must reach them.
  EXPECTED: post-init/elite-only report `[local_search, evaluation]`; periodic
  reports `[local_search]` plus `re_evaluate` only when it is configured.
  """
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  assert PostInitLocalSearchStep(local_search=ls_strat, evaluation=eval_strat).strategies() == [ls_strat, eval_strat]
  assert EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat).strategies() == [ls_strat, eval_strat]
  assert PeriodicLocalSearchStep(local_search=ls_strat, interval=2, target="offspring").strategies() == [ls_strat]
  assert PeriodicLocalSearchStep(
    local_search=ls_strat, interval=2, target="population", re_evaluate=eval_strat
  ).strategies() == [ls_strat, eval_strat]

  reg = MagicMock()
  assert PeriodicLocalSearchStep(
    local_search=ls_strat, interval=2, target="offspring", re_evaluate=reg
  ).strategies() == [ls_strat, reg]


def _make_ga(pipeline: list[GenerationStep], local_search: LocalSearchStrategy | None = None) -> GeneticAlgorithm:
  """Build a minimal CPU GA with a custom pipeline for anchor/composition tests."""
  eval_strat = EvaluationCPUCost()
  return GeneticAlgorithm(
    population_size=8,
    mutation_rate=0.02,
    seed=42,
    initialization=InitializationRandom(seed=42),
    selection=SelectionTournament(k=3, seed=42),
    crossover=CrossoverOrder(seed=42),
    mutation=MutationSwap(seed=42),
    evaluation=eval_strat,
    local_search=local_search,
    local_search_stage="none",
    pipeline=pipeline,
  )


class _AnchorProbeStep:
  """Minimal third-party step used to prove dispatch is anchor-driven, not name-driven."""

  def __init__(self, anchor: StageAnchor) -> None:
    self.stage_anchor: StageAnchor = anchor

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """No-op probe."""


def test_compose_pipeline_dispatches_on_stage_anchor() -> None:
  """WHAT: `_compose_pipeline` places steps by `stage_anchor` alone (BUG-30).

  WHY: A step whose class name is unknown to the core must still land where its
  semantics require; name matching made that impossible.
  EXPECTED: `pre_selection` at index 0, `pre_evaluation` immediately before the
  offspring `EvaluationStep`, `post_survival` last.
  """
  pre_selection = _AnchorProbeStep("pre_selection")
  pre_evaluation = _AnchorProbeStep("pre_evaluation")
  post_survival = _AnchorProbeStep("post_survival")

  ga = _make_ga([post_survival, pre_evaluation, pre_selection])
  pipeline = ga.pipeline

  assert pipeline[0] is pre_selection
  assert pipeline[-1] is post_survival
  eval_idx = next(
    i
    for i, step in enumerate(pipeline)
    if type(step).__name__ == "EvaluationStep" and getattr(step, "target", "") == "offspring"
  )
  assert pipeline[eval_idx - 1] is pre_evaluation


def test_compose_pipeline_keeps_declaration_order_within_anchor() -> None:
  """WHAT: Multiple steps sharing an anchor keep their declared order.

  WHY: Two pre-selection refinements must compose deterministically, not by hash order.
  EXPECTED: Both sit at the head of the pipeline in declaration order.
  """
  first = _AnchorProbeStep("pre_selection")
  second = _AnchorProbeStep("pre_selection")
  ga = _make_ga([first, second])

  assert ga.pipeline[0] is first
  assert ga.pipeline[1] is second


def test_custom_pipeline_steps_are_recorded_on_instance() -> None:
  """WHAT: `custom_pipeline_steps` preserves the caller-declared steps.

  WHY: `pipeline` holds the composed list, so without this the orchestration layer
  cannot distinguish declared steps from core ones (`BUG-30`).
  EXPECTED: Declared steps are kept verbatim; default construction records none.
  """
  step = PostInitLocalSearchStep(local_search=LocalSearchCPUTwoOpt(), evaluation=EvaluationCPUCost())
  ga = _make_ga([step])
  assert ga.custom_pipeline_steps == [step]

  eval_strat = EvaluationCPUCost()
  ga_default = GeneticAlgorithm(
    population_size=8,
    initialization=InitializationRandom(seed=1),
    selection=SelectionTournament(k=3, seed=1),
    crossover=CrossoverOrder(seed=1),
    mutation=MutationSwap(seed=1),
    evaluation=eval_strat,
    local_search_stage="none",
  )
  assert ga_default.custom_pipeline_steps == []


def test_all_strategies_includes_pipeline_owned_strategies(burma14_problem: Problem) -> None:
  """WHAT: telemetry traverses pipeline-owned strategies when the core has none (BUG-30).

  WHY: With `local_search_stage="none"` and `local_search=None`, the only local search
  and evaluator instances live inside the extension step; ignoring them zeroed
  `h2d_bytes` / `kernel_launches` in every checkpoint.
  EXPECTED: Both step strategies appear once, and their counters are aggregated.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  step_ls = LocalSearchCPUTwoOpt(max_iterations=2)
  step_eval = EvaluationCPUCost()
  step = PostInitLocalSearchStep(local_search=step_ls, evaluation=step_eval)

  ga = _make_ga([step])
  collected = ga._all_strategies()

  assert step_ls in collected
  assert step_eval in collected
  assert collected.count(step_ls) == 1
  assert collected.count(step_eval) == 1

  ga._reset_run_metrics()
  step_ls.h2d_bytes = 512
  step_ls.kernel_launches = 7
  ga._collect_strategy_metrics(context)

  assert ga.h2d_bytes >= 512
  assert ga.kernel_launches == 7

def test_reset_run_metrics_clears_pipeline_owned_counters(burma14_problem: Problem) -> None:
  """WHAT: `_reset_run_metrics` resets pipeline-owned strategies too.

  WHY: A step strategy that survives a previous run would leak H2D/kernel counts
  into the next repetition, inflating the reported cost.
  EXPECTED: Counters set before the reset read back as zero.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  step_ls = LocalSearchCPUTwoOpt(max_iterations=2)
  step = EliteOnlyLocalSearchStep(local_search=step_ls, evaluation=EvaluationCPUCost(), elite_count=2)
  ga = _make_ga([step])

  step_ls.h2d_bytes = 4096
  step_ls.d2h_bytes = 2048
  step_ls.kernel_launches = 9
  ga._reset_run_metrics()

  assert step_ls.h2d_bytes == 0
  assert step_ls.d2h_bytes == 0
  assert step_ls.kernel_launches == 0
  assert ga.h2d_bytes == 0
  assert context is not None


def test_serialize_custom_pipeline_preserves_params() -> None:
  """WHAT: checkpoint serialization keeps the knobs that define each step (BUG-30).

  WHY: Recording only `class` produced unreproducible coortes and exposed internal
  core steps (`SelectionStep`, ...) as if they were user configuration.
  EXPECTED: Only declared steps are serialized, each with its `params`.
  """
  eval_strat = EvaluationCPUCost()
  ls_strat = LocalSearchCPUTwoOpt()

  steps: list[Any] = [
    PostInitLocalSearchStep(local_search=ls_strat, evaluation=eval_strat),
    EliteOnlyLocalSearchStep(local_search=ls_strat, evaluation=eval_strat, elite_count=8),
    PeriodicLocalSearchStep(local_search=ls_strat, interval=10, target="population", re_evaluate=eval_strat),
  ]

  serialized = _serialize_custom_pipeline(steps)

  assert serialized == [
    {"class": "PostInitLocalSearchStep"},
    {"class": "EliteOnlyLocalSearchStep", "params": {"elite_count": 8}},
    {"class": "PeriodicLocalSearchStep", "params": {"interval": 10, "target": "population"}},
  ]


def test_example_config_never_combines_pipeline_and_local_search_stage() -> None:
  """WHAT: no example variant declares both a pipeline and a local search stage.

  WHY: Declaring `pipeline=[PostInitLocalSearchStep]` with
  `local_search_stage="post_evaluation"` refined P_0 twice with no genetic operator
  between the passes, which is exactly the `BUG-21` artefact.
  EXPECTED: Every variant with a non-empty `pipeline` uses `local_search_stage="none"`.
  """
  config_path = Path("configs")
  _ga_params, configs = load_algorithms_config(config_path, config_name="algorithms.full.example.json")

  pipelined = {name: cfg for name, cfg in configs.items() if cfg.params.get("pipeline")}
  assert pipelined, "example config must document at least one pipeline variant"
  for name, cfg in pipelined.items():
    assert cfg.params.get("local_search_stage", "post_mutation") == "none", (
      f"{name} declares a pipeline and a local_search_stage at the same time"
    )


