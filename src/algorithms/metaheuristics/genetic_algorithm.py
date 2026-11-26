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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

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
  from collections.abc import Callable, Mapping, Sequence

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
# StageSpec and Unified Stage Dispatch Table (DP11-8, DP11-9, BUG-21)
# =============================================================================


@dataclass(frozen=True)
class StageSpec:
  """Specification of pipeline steps and post-loop execution for a LocalSearchStage."""

  pipeline_steps: Callable[[GeneticAlgorithm], list[GenerationStep]]
  post_loop_hook: bool = False
  requires_local_search: bool = True


def _steps_standard(ga: GeneticAlgorithm) -> list[GenerationStep]:
  return [
    SelectionStep(ga.selection, ga.population_size),
    CrossoverStep(ga.crossover),
    MutationStep(ga.mutation, ga.mutation_rate),
    EvaluationStep(ga.evaluation, target="offspring"),
    SurvivalStep(ga._select_survivors),
  ]


def _steps_post_crossover(ga: GeneticAlgorithm) -> list[GenerationStep]:
  if ga.local_search is None:
    return _steps_standard(ga)
  return [
    SelectionStep(ga.selection, ga.population_size),
    CrossoverStep(ga.crossover),
    LocalSearchStep(ga.local_search, target="offspring"),
    MutationStep(ga.mutation, ga.mutation_rate),
    EvaluationStep(ga.evaluation, target="offspring"),
    SurvivalStep(ga._select_survivors),
  ]


def _steps_post_mutation(ga: GeneticAlgorithm) -> list[GenerationStep]:
  if ga.local_search is None:
    return _steps_standard(ga)
  return [
    SelectionStep(ga.selection, ga.population_size),
    CrossoverStep(ga.crossover),
    MutationStep(ga.mutation, ga.mutation_rate),
    LocalSearchStep(ga.local_search, target="offspring"),
    EvaluationStep(ga.evaluation, target="offspring"),
    SurvivalStep(ga._select_survivors),
  ]


def _steps_post_evaluation(ga: GeneticAlgorithm) -> list[GenerationStep]:
  if ga.local_search is None:
    return _steps_standard(ga)
  return [
    SelectionStep(ga.selection, ga.population_size),
    CrossoverStep(ga.crossover),
    MutationStep(ga.mutation, ga.mutation_rate),
    EvaluationStep(ga.evaluation, target="offspring"),
    SurvivalStep(ga._select_survivors),
    LocalSearchStep(ga.local_search, target="population", re_evaluate=ga.evaluation),
  ]


STAGE_SPECS: Mapping[LocalSearchStage, StageSpec] = {
  "none": StageSpec(pipeline_steps=_steps_standard, post_loop_hook=False, requires_local_search=False),
  "post_crossover": StageSpec(pipeline_steps=_steps_post_crossover, post_loop_hook=False, requires_local_search=True),
  "post_mutation": StageSpec(pipeline_steps=_steps_post_mutation, post_loop_hook=False, requires_local_search=True),
  "post_evaluation": StageSpec(pipeline_steps=_steps_post_evaluation, post_loop_hook=False, requires_local_search=True),
  "post_evolution": StageSpec(pipeline_steps=_steps_standard, post_loop_hook=True, requires_local_search=True),
}


#: Attributes a pipeline step may expose without declaring `strategies()`, used as
#: a fallback so third-party steps still contribute their telemetry counters.
_PIPELINE_STRATEGY_ATTRS: tuple[str, ...] = ("strategy", "local_search", "evaluation", "re_evaluate")


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
    if local_search_stage not in STAGE_SPECS:
      raise ValueError(
        f"Invalid local_search_stage '{local_search_stage}'. Valid stages: {list(STAGE_SPECS.keys())}"
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
    self.custom_pipeline_steps: list[GenerationStep] = list(pipeline) if pipeline is not None else []
    if pipeline is not None:
      has_core_step = any(
        isinstance(step, (SelectionStep, CrossoverStep, MutationStep, SurvivalStep))
        for step in pipeline
      )
      if has_core_step:
        self.pipeline: list[GenerationStep] = pipeline
      else:
        self.pipeline = self._compose_pipeline(pipeline)
    else:
      self.pipeline = self._build_default_pipeline()

  def _compose_pipeline(self, extension_steps: list[GenerationStep]) -> list[GenerationStep]:
    """Insert extension steps into the default pipeline at their declared `stage_anchor`.

    Anchors keep placement declarative (`BUG-30`): a step states where it belongs
    instead of `GeneticAlgorithm` matching on its class name, so a new step type
    needs no edit here. `pre_selection` steps run before parent selection, in the
    order declared; `pre_evaluation` steps run on offspring before they are scored;
    everything else (including steps that declare no anchor) is appended after survival.
    """
    steps = self._build_default_pipeline()
    pre_selection_steps: list[GenerationStep] = []
    for step in extension_steps:
      anchor: str = getattr(step, "stage_anchor", "post_survival")
      if anchor == "pre_selection":
        pre_selection_steps.append(step)
      elif anchor == "pre_evaluation":
        eval_idx = next(
          (
            i
            for i, s in enumerate(steps)
            if isinstance(s, EvaluationStep) and getattr(s, "target", "") == "offspring"
          ),
          len(steps),
        )
        steps.insert(eval_idx, step)
      else:
        steps.append(step)
    return [*pre_selection_steps, *steps]

  def _build_default_pipeline(self) -> list[GenerationStep]:
    """Construct standard generational pipeline based on configured strategies and stage."""
    spec = STAGE_SPECS[self.local_search_stage]
    return spec.pipeline_steps(self)

  # ===========================================================================
  # Internal Helpers
  # ===========================================================================

  def _all_strategies(self) -> list[Any]:
    """Return every strategy that can own telemetry, including pipeline-owned ones.

    Extension steps (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`,
    `PeriodicLocalSearchStep`) carry their own local search and evaluator, which may
    be the *only* instance in the run when `local_search_stage="none"`; collecting
    them here keeps H2D/D2H/kernel counters from silently reading zero (`BUG-30`).
    Identity de-duplication keeps shared strategies from being reset or summed twice.
    """
    candidates: list[Any] = [
      self.initialization,
      self.selection,
      self.crossover,
      self.mutation,
      self.evaluation,
      self.local_search,
    ]
    for step in self.pipeline:
      declared = getattr(step, "strategies", None)
      if callable(declared):
        candidates.extend(declared())
      else:
        candidates.extend(getattr(step, attr, None) for attr in _PIPELINE_STRATEGY_ATTRS)

    unique: list[Any] = []
    seen: set[int] = set()
    for strat in candidates:
      if strat is None or id(strat) in seen:
        continue
      seen.add(id(strat))
      unique.append(strat)
    return unique

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

  def warmup(self, context: ProblemContext) -> None:
    """Pre-compile every kernel this pipeline can launch (`DP11-12`, `DP11-21`, `BUG-25`).

    Replaces the discarded one-generation run, which only covered the operators that
    happen to fire in generation 1: `PeriodicLocalSearchStep(interval=10)` never fires
    there, so `two_opt_batch.cu` was JIT-compiled at the first *measured* generation 10
    (26.27x spike, 1651.73 ms vs 64.44 ms). This method walks the whole pipeline — every
    step and every strategy they own, plus the six injected slots — so coverage is a
    property of the configuration, not of the generation counter.

    It performs no evolutionary work: no generation counter, population state or
    `PhaseTimer` bucket of this instance changes. Each strategy's hook compiles its
    kernel and launches it once on a `(rows, n)` batch of valid permutations
    (`arange`, never `cp.zeros` — `BUG-24`) so the driver's module load also happens
    here, not inside a measurement. On the isolated clone the runner uses, the RNG
    advanced by those launches is discarded with the clone.

    Args:
        context: Run context used by strategy-level warm-up hooks.
    """
    seen: set[int] = set()
    for step in self.pipeline:
      step_hook = getattr(step, "warmup", None)
      if not callable(step_hook):
        continue
      step_hook(context)
      # Steps warm the strategies they own; recording those identities keeps the
      # per-strategy pass below from compiling (and counting) them a second time.
      owned = getattr(step, "strategies", None)
      if callable(owned):
        seen.update(id(strategy) for strategy in owned())

    for strategy in self._all_strategies():
      if id(strategy) in seen:
        continue
      seen.add(id(strategy))
      strategy_hook = getattr(strategy, "warmup", None)
      if callable(strategy_hook):
        strategy_hook(context)

  def _collect_strategy_metrics(self, context: ProblemContext) -> None:
    """Aggregate H2D, D2H, and kernel launch counts from all active strategies.

    Context-owned GPU caches (`ProblemContext.h2d_bytes`) are already attributed once
    to whichever strategy triggered the first physical upload (`BUG-95`), so this
    collector must not re-add `context._gpu_distances.nbytes`.
    """
    _ = context
    total_h2d = 0
    total_d2h = 0
    total_kernels = 0
    for strat in self._all_strategies():
      total_h2d += int(getattr(strat, "h2d_bytes", 0))
      total_d2h += int(getattr(strat, "d2h_bytes", 0))
      total_kernels += int(getattr(strat, "kernel_launches", 0))

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
  def _best_cost(fitness: Fitness) -> float:
    """Return only the best fitness value, transferring a single scalar D2H (`BUG-30`).

    A generation that does not beat the incumbent needs the cost and nothing else;
    materializing the winning route here is what used to copy a full `(n,)` tour
    device-to-host on every generation.
    """
    if is_gpu_array(fitness):
      return float(require_gpu().min(fitness).get())
    return float(np.min(fitness))

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

    # 2. Initial population evaluation (Generation 0 evaluated raw for all 5 stages; BUG-21)
    fitness: Fitness = self.timer.measure(
      "evaluation",
      self.evaluation.evaluate_batch,
      population,
      context,
    )

    best_tour, best_cost = self.timer.measure(
      "host_sync",
      self._extract_best,
      population,
      fitness,
    )
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

      gen_best_cost = self.timer.measure(
        "host_sync",
        self._best_cost,
        state.fitness,
      )
      if gen_best_cost < best_cost - 1e-6:
        # The route crosses PCIe only on strict improvement (`BUG-30`), so a stagnant
        # generation pays one scalar read instead of a full `(n,)` device-to-host copy.
        best_tour, _ = self.timer.measure(
          "host_sync",
          self._extract_best,
          state.population,
          state.fitness,
        )
        best_cost = gen_best_cost
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

    # 4. Post-evolution stage hook if registered in StageSpec (DP11-8)
    spec = STAGE_SPECS[self.local_search_stage]
    if self.local_search is not None and spec.post_loop_hook:
      state.population = self.timer.measure(
        "local_search",
        self.local_search.improve_batch,
        state.population,
        context,
      )
      state.fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        state.population,
        context,
      )
      post_best_tour, post_best_cost = self.timer.measure(
        "host_sync",
        self._extract_best,
        state.population,
        state.fitness,
      )
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
  "STAGE_SPECS",
  "CrossoverStep",
  "EvaluationStep",
  "GenerationState",
  "GenerationStep",
  "GeneticAlgorithm",
  "LocalSearchStage",
  "LocalSearchStep",
  "MutationStep",
  "SelectionStep",
  "StageSpec",
  "SurvivalStep",
]
