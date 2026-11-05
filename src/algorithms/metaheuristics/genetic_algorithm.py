"""
Composable, Strategy-Driven Genetic Algorithm Metaheuristic.

Every evolutionary phase (initialization, selection, crossover, mutation,
local search, and fitness evaluation) is injected as an interchangeable strategy
sharing standardized `Route` / `Routes` (`RouteOrRoutes`) contracts.

Execution within each generation is driven by a composable Step Pipeline:
- Pure GA (`local_search=None`) executes selection -> crossover -> mutation -> evaluation -> survival.
- Memetic GA (`local_search != None`) places local search according to `local_search_stage`:
  - `"post_crossover"`: immediately after crossover on offspring routes
  - `"post_mutation"`: after mutation on offspring routes (canonical memetic GA)
  - `"post_evaluation"`: after fitness evaluation & survival on surviving population routes
  - `"post_evolution"`: applied once to the final population after all generations complete
  - `"none"`: local search disabled (pure GA)
- Custom pipelines: users can pass `pipeline=[...]` directly for non-standard operator sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import numpy as np

from protocols.algorithm_strategies import (
  CrossoverStrategy,
  EvaluationStrategy,
  LocalSearchStrategy,
  MutationStrategy,
  SelectionStrategy,
)
from protocols.backend import cupy_backend, require_gpu
from protocols.types import (
  Fitness,
  LocalSearchStage,
  Route,
  Routes,
)
from utils.phase_timer import PhaseTimer
from utils.route_helpers import is_gpu_array

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.algorithm_strategies import (
    CrossoverStrategy,
    EvaluationStrategy,
    InitializationStrategy,
    LocalSearchStrategy,
    MutationStrategy,
    SelectionStrategy,
  )
  from protocols.problem_context import ProblemContext


# =============================================================================
# Composable Pipeline Step Abstraction
# =============================================================================


@dataclass
class GenerationState:
  """Mutable state carried through evolutionary pipeline steps each generation."""

  population: Routes
  fitness: Fitness
  context: ProblemContext
  offspring: Routes | None = None
  offspring_fitness: Fitness | None = None
  generation: int = 0


class GenerationStep(Protocol):
  """Protocol for an individual step within an evolutionary generation."""

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Execute the evolutionary step, modifying `state` in-place."""
    ...


class SelectionStep:
  """Parent selection step producing `offspring` candidate routes."""

  def __init__(self, strategy: SelectionStrategy, population_size: int) -> None:
    self.strategy: SelectionStrategy = strategy
    self.population_size: int = population_size

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    state.offspring = timer.measure(
      "selection",
      self.strategy.select_batch,
      state.population,
      state.fitness,
      self.population_size,
      state.context,
    )


class CrossoverStep:
  """Route recombination step transforming `offspring` routes."""

  def __init__(self, strategy: CrossoverStrategy) -> None:
    self.strategy: CrossoverStrategy = strategy

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    if state.offspring is None:
      raise RuntimeError("CrossoverStep called before state.offspring was populated.")
    state.offspring = timer.measure(
      "crossover",
      self.strategy.crossover_batch,
      state.offspring,
      state.context,
    )


class MutationStep:
  """Route mutation step mutating `offspring` routes."""

  def __init__(self, strategy: MutationStrategy, mutation_rate: float) -> None:
    self.strategy: MutationStrategy = strategy
    self.mutation_rate: float = mutation_rate

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    if state.offspring is None:
      raise RuntimeError("MutationStep called before state.offspring was populated.")
    state.offspring = timer.measure(
      "mutation",
      self.strategy.mutate_batch,
      state.offspring,
      self.mutation_rate,
      state.context,
    )


class EvaluationStep:
  """Route cost evaluation step computing fitness on offspring or population."""

  def __init__(
    self,
    strategy: EvaluationStrategy,
    target: Literal["offspring", "population"] = "offspring",
  ) -> None:
    self.strategy: EvaluationStrategy = strategy
    self.target: Literal["offspring"] | Literal["population"] = target

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("EvaluationStep target='offspring' called with state.offspring is None.")
      state.offspring_fitness = timer.measure(
        "evaluation",
        self.strategy.evaluate_batch,
        state.offspring,
        state.context,
      )
    else:
      state.fitness = timer.measure(
        "evaluation",
        self.strategy.evaluate_batch,
        state.population,
        state.context,
      )


class SurvivalStep:
  """Survival selection step combining parents and offspring into next generation."""

  def __init__(self, survivor_selector: Any) -> None:
    self.survivor_selector = survivor_selector

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    if state.offspring is None or state.offspring_fitness is None:
      raise RuntimeError("SurvivalStep called before offspring and offspring_fitness were available.")
    state.population, state.fitness = timer.measure(
      "survival",
      self.survivor_selector,
      state.population,
      state.fitness,
      state.offspring,
      state.offspring_fitness,
    )


class LocalSearchStep:
  """Local search refinement step operating on offspring or population."""

  def __init__(
    self,
    strategy: LocalSearchStrategy,
    target: Literal["offspring", "population"] = "offspring",
    re_evaluate: EvaluationStrategy | None = None,
  ) -> None:
    self.strategy: LocalSearchStrategy = strategy
    self.target: Literal["offspring"] | Literal["population"] = target
    self.re_evaluate: EvaluationStrategy | None = re_evaluate

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("LocalSearchStep target='offspring' called with state.offspring is None.")
      state.offspring = timer.measure(
        "local_search",
        self.strategy.improve_batch,
        state.offspring,
        state.context,
      )
    else:
      state.population = timer.measure(
        "local_search",
        self.strategy.improve_batch,
        state.population,
        state.context,
      )
      if self.re_evaluate is not None:
        state.fitness = timer.measure(
          "evaluation",
          self.re_evaluate.evaluate_batch,
          state.population,
          state.context,
        )


# =============================================================================
# GeneticAlgorithm Metaheuristic
# =============================================================================


class GeneticAlgorithm:
  """
  Modular Genetic Algorithm orchestrated via injectable operator strategies.

  Decoupled from concrete strategy implementations, concrete hardware backends,
  and strategy-specific parameters.
  """

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    seed: int | None = 42,
    *,
    initialization: InitializationStrategy,
    selection: SelectionStrategy,
    crossover: CrossoverStrategy,
    mutation: MutationStrategy,
    evaluation: EvaluationStrategy,
    local_search: LocalSearchStrategy | None = None,
    local_search_stage: LocalSearchStage = "post_mutation",
    enable_phase_timing: bool = True,
    pipeline: list[GenerationStep] | None = None,
  ) -> None:
    """
    Initialize modular Genetic Algorithm.

    Args:
        population_size: Number of routes in the population (`N`).
        mutation_rate: Probability of mutating each offspring route.
        seed: Random seed for reproducibility.
        initialization: Strategy to generate initial `(N, n_cities)` routes.
        selection: Strategy to select parent routes from `(routes, fitness)`.
        crossover: Strategy to recombine parent routes into offspring routes.
        mutation: Strategy to mutate routes with probability `mutation_rate`.
        evaluation: Strategy to compute route costs `(N,)`.
        local_search: Optional strategy to refine routes (Pure GA when `None`).
        local_search_stage: Where to place `local_search` when provided:
            `"post_crossover"`, `"post_mutation"`, `"post_evaluation"`,
            `"post_evolution"`, or `"none"`.
        enable_phase_timing: Record per-phase execution times via `PhaseTimer`.
        pipeline: Optional custom list of `GenerationStep`s (overrides default pipeline).
    """
    valid_stages = get_args(LocalSearchStage.__value__)
    if local_search_stage not in valid_stages:
      raise ValueError(
        f"Invalid local_search_stage '{local_search_stage}'. Valid stages: {list(valid_stages)}"
      )

    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.seed: int | None = seed
    self.local_search_stage: LocalSearchStage = local_search_stage

    self.initialization: InitializationStrategy = initialization
    self.selection: SelectionStrategy = selection
    self.crossover: CrossoverStrategy = crossover
    self.mutation: MutationStrategy = mutation
    self.evaluation: EvaluationStrategy = evaluation
    self.local_search: LocalSearchStrategy | None = local_search

    self.timer = PhaseTimer(enabled=enable_phase_timing)
    self.generation: int = 0
    self.best_cost_history: list[float] = []
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

    # Build or set the generational pipeline
    if pipeline is not None:
      self.pipeline: list[GenerationStep] = pipeline
    else:
      self.pipeline = self._build_default_pipeline()

  def _build_default_pipeline(self) -> list[GenerationStep]:
    """Construct standard generational pipeline based on configured strategies and stage."""
    steps: list[GenerationStep] = [
      SelectionStep(self.selection, self.population_size),
      CrossoverStep(self.crossover),
    ]

    has_ls: bool = self.local_search is not None and self.local_search_stage != "none"
    ls: LocalSearchStrategy | None = self.local_search

    if has_ls and ls is not None and self.local_search_stage == "post_crossover":
      steps.append(LocalSearchStep(ls, target="offspring"))

    steps.append(MutationStep(self.mutation, self.mutation_rate))

    if has_ls and ls is not None and self.local_search_stage == "post_mutation":
      steps.append(LocalSearchStep(ls, target="offspring"))

    steps.append(EvaluationStep(self.evaluation, target="offspring"))
    steps.append(SurvivalStep(self._select_survivors))

    if has_ls and ls is not None and self.local_search_stage == "post_evaluation":
      steps.append(
        LocalSearchStep(
          ls,
          target="population",
          re_evaluate=self.evaluation,
        )
      )

    return steps

  # ===========================================================================
  # Internal Helpers
  # ===========================================================================

  def _all_strategies(self) -> list[Any]:
    """Return list of active strategy objects for metric aggregation."""
    strats: list[Any] = [
      self.initialization,
      self.selection,
      self.crossover,
      self.mutation,
      self.evaluation,
    ]
    if self.local_search is not None:
      strats.append(self.local_search)
    return strats

  def _reset_run_metrics(self) -> None:
    """Reset counters on self and all injected strategies before a run."""
    self.generation = 0
    self.best_cost_history = []
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0
    self.timer.reset()
    for strat in self._all_strategies():
      if hasattr(strat, "reset_metrics"):
        strat.reset_metrics()

  def _collect_strategy_metrics(self, context: ProblemContext) -> None:
    """Aggregate H2D, D2H, and kernel launch counts from all active strategies."""
    total_h2d = 0
    total_d2h = 0
    total_kernels = 0
    for strat in self._all_strategies():
      total_h2d += int(getattr(strat, "h2d_bytes", 0))
      total_d2h += int(getattr(strat, "d2h_bytes", 0))
      total_kernels += int(getattr(strat, "kernel_launches", 0))

    gpu_dist = getattr(context, "_gpu_distances", None)
    if gpu_dist is not None and hasattr(gpu_dist, "nbytes"):
      total_h2d += int(gpu_dist.nbytes)

    self.h2d_bytes = total_h2d
    self.d2h_bytes = total_d2h
    self.kernel_launches = total_kernels

  def _select_survivors(
    self,
    old_pop: Routes,
    old_fitness: Fitness,
    new_pop: Routes,
    new_fitness: Fitness,
  ) -> tuple[Routes, Fitness]:
    """
    Deterministic `(mu + lambda)` elitist survival selection.

    Combines `old_pop` and `new_pop` `(2N, n_cities)` and keeps top `N` routes.
    """
    is_gpu: bool = is_gpu_array(old_pop)
    xp: Any = cupy_backend() if is_gpu else np

    combined_pop = xp.vstack([old_pop, new_pop])
    combined_fitness = xp.concatenate([xp.asarray(old_fitness), xp.asarray(new_fitness)])
    best_indices = xp.argsort(combined_fitness)[: self.population_size]
    return combined_pop[best_indices], combined_fitness[best_indices]

  @staticmethod
  def _extract_best(population: Routes, fitness: Fitness) -> tuple[Route, float]:
    """Return `(best_route_np, best_cost_float)` from population and fitness."""
    if is_gpu_array(fitness):
      cp = require_gpu()
      best_idx = int(cp.argmin(fitness).get())
      best_cost = float(fitness[best_idx].get())
    else:
      best_idx = int(np.argmin(fitness))
      best_cost = float(fitness[best_idx])

    best_route_raw = population[best_idx]
    if is_gpu_array(best_route_raw):
      best_route: Route = require_gpu().asnumpy(best_route_raw).astype(np.int32)
    else:
      best_route = np.asarray(best_route_raw, dtype=np.int32).copy()
    return best_route, best_cost

  # ===========================================================================
  # Main Evolutionary Loop
  # ===========================================================================

  def evolve(
    self,
    context: ProblemContext,
    customers: Sequence[int],
    max_generations: int,
    optimal_cost: float | None = None,
    patience: int | None = None,
  ) -> tuple[Route, dict[str, Any]]:
    """
    Execute modular Genetic Algorithm evolutionary pipeline.

    Args:
        context: `ProblemContext` providing distance matrices and RNG state.
        customers: Sequence of customer indices `[0..n-1]`.
        max_generations: Maximum number of generations to evolve.
        optimal_cost: Optional known optimal tour cost for early stopping.
        patience: Stagnation window in generations (`None` = `2 * sqrt(n)`).

    Returns:
        Tuple `(best_tour, stats)` where `best_tour` is a 1D `Route` `(n_cities,)`
        and `stats` contains convergence, transfer, and phase-timing metrics.
    """
    if len(customers) == 0:
      raise ValueError("customers sequence cannot be empty")

    n: int = len(customers)
    if patience is None:
      patience = int(2 * math.sqrt(n))

    self._reset_run_metrics()

    # 1. Initialize population: shape (population_size, n)
    population: Routes = self.timer.measure(
      "initialization",
      self.initialization.initialize,
      customers,
      self.population_size,
      context,
    )

    # 2. Initial population evaluation
    has_ls: bool = self.local_search is not None and self.local_search_stage != "none"
    ls: LocalSearchStrategy | None = self.local_search

    if has_ls and ls is not None and self.local_search_stage in ("post_crossover", "post_mutation"):
      population = self.timer.measure(
        "local_search",
        ls.improve_batch,
        population,
        context,
      )
      fitness: Fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
    elif has_ls and ls is not None and self.local_search_stage == "post_evaluation":
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
      population = self.timer.measure(
        "local_search",
        ls.improve_batch,
        population,
        context,
      )
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
    else:
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )

    best_tour, best_cost = self._extract_best(population, fitness)
    initial_best_cost: float = best_cost
    self.best_cost_history = [best_cost]

    generations_without_improvement = 0
    stopped_early = False
    stop_reason = "max_generations"

    state = GenerationState(population=population, fitness=fitness, context=context)

    # 3. Generational evolution loop driven by the pipeline
    for gen in range(max_generations):
      self.generation = gen + 1
      state.generation = self.generation

      for step in self.pipeline:
        step.run(state, self.timer)

      gen_best_tour, gen_best_cost = self._extract_best(state.population, state.fitness)
      if gen_best_cost < best_cost - 1e-6:
        best_cost = gen_best_cost
        best_tour = gen_best_tour
        generations_without_improvement = 0
      else:
        generations_without_improvement += 1

      self.best_cost_history.append(best_cost)

      # Early stopping check 1: true optimal reached or within 1% of known optimal cost
      if optimal_cost is not None and optimal_cost > 0:
        if best_cost <= optimal_cost + 1e-6:
          stopped_early = True
          stop_reason = "hit_optimal"
          break
        gap_percent: float = (best_cost - optimal_cost) / optimal_cost * 100.0
        if gap_percent < 1.0:
          stopped_early = True
          stop_reason = "within_1pct_optimal"
          break

      # Early stopping check 2: stagnation patience exceeded
      if generations_without_improvement >= patience:
        stopped_early = True
        stop_reason = "no_improvements"
        break

    # 4. Post-evolution stage: refine the full final population once after loop ends
    if has_ls and ls is not None and self.local_search_stage == "post_evolution":
      state.population = self.timer.measure(
        "local_search",
        ls.improve_batch,
        state.population,
        context,
      )
      state.fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        state.population,
        context,
      )
      post_best_tour, post_best_cost = self._extract_best(state.population, state.fitness)
      if post_best_cost <= best_cost:
        best_tour = post_best_tour
        best_cost = post_best_cost
        self.best_cost_history[-1] = best_cost

    self._collect_strategy_metrics(context)
    phase_timings: dict[str, dict[str, float | int]] = self.timer.summary()

    stats: dict[str, Any] = {
      "initial_fitness": initial_best_cost,
      "best_fitness": best_cost,
      "generations_completed": self.generation,
      "stopped_early": stopped_early,
      "stop_reason": stop_reason,
      "convergence_history": self.best_cost_history,
      "phase_timings": phase_timings,
      "h2d_bytes": self.h2d_bytes,
      "d2h_bytes": self.d2h_bytes,
      "kernel_launches": self.kernel_launches,
    }
    return best_tour, stats


__all__: list[str] = [
  "CrossoverStep",
  "EvaluationStep",
  "GenerationState",
  "GenerationStep",
  "GeneticAlgorithm",
  "LocalSearchStage",
  "LocalSearchStep",
  "MutationStep",
  "SelectionStep",
  "SurvivalStep",
]
