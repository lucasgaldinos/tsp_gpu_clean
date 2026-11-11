"""
Custom composable generational pipeline steps for GeneticAlgorithm (DP11-6, DP11-10).

Provides modular GenerationStep implementations for non-standard evolutionary pipelines,
such as post-initialization local search refinement on Generation 0 / 1 (PostInitLocalSearchStep),
elite-only local search (EliteOnlyLocalSearchStep), and periodic local search
(PeriodicLocalSearchStep).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from protocols.backend import cupy_backend
from utils.route_helpers import is_gpu_array

if TYPE_CHECKING:
  from algorithms.metaheuristics.genetic_algorithm import GenerationState
  from protocols.algorithm_strategies import EvaluationStrategy, LocalSearchStrategy
  from protocols.problem_context import ProblemContext
  from utils.phase_timer import PhaseTimer

#: Declarative insertion anchor of a pipeline step inside a generation.
#:
#: `pre_selection` runs before parent selection (generation-0/1 refinement),
#: `pre_evaluation` runs on the offspring right before it is scored, and
#: `post_survival` runs on the surviving population at the end of the generation.
type StageAnchor = Literal["pre_selection", "pre_evaluation", "post_survival"]


def _warmup_owned_strategies(strategies: list[Any], context: ProblemContext) -> None:
  """Call `warmup(context)` on every strategy a pipeline step owns.

  Strategies are duck-typed here (`getattr`, not a hard import) so third-party steps
  and test doubles that own plain callables still compose into a pipeline.
  """
  for strategy in strategies:
    hook = getattr(strategy, "warmup", None)
    if callable(hook):
      hook(context)


class _BaseLocalSearchPipelineStep:
  """Shared strategy injection, validation and introspection for local search steps."""

  #: Where `GeneticAlgorithm._compose_pipeline` inserts this step.
  stage_anchor: ClassVar[StageAnchor] = "post_survival"

  def __init__(
    self,
    local_search: LocalSearchStrategy,
    evaluation: EvaluationStrategy,
  ) -> None:
    if local_search is None:
      raise ValueError(f"{self.__class__.__name__} requires a local_search strategy")
    if evaluation is None:
      raise ValueError(f"{self.__class__.__name__} requires an evaluation strategy")
    self.local_search: LocalSearchStrategy = local_search
    self.evaluation: EvaluationStrategy = evaluation

  def strategies(self) -> list[Any]:
    """Return every strategy this step owns, for telemetry reset and aggregation."""
    return [self.local_search, self.evaluation]

  def public_params(self) -> dict[str, Any]:
    """Return the declarative parameters of this step for checkpoint serialization."""
    return {}

  def warmup(self, context: ProblemContext) -> None:
    """Compile the kernels owned by this step, whatever its cadence (`DP11-12`, `BUG-25`).

    Generation-gated steps (`PostInitLocalSearchStep` at generation 1) and periodic
    steps (`PeriodicLocalSearchStep`, every `interval` generations) are otherwise only
    first executed deep inside the measured repetitions; warming their strategies here
    keeps that first execution free of JIT compilation.
    """
    _warmup_owned_strategies(self.strategies(), context)


class PostInitLocalSearchStep(_BaseLocalSearchPipelineStep):
  """
  Refine `state.population` only once at the start of generation 1 (DP11-6).

  Permits a memetic warm-start without modifying Generation 0's raw evaluation contract.
  Executes `local_search.improve_batch` when `state.generation == 1` and re-evaluates
  `state.fitness` so downstream selection uses refined costs. No-op for all subsequent generations.
  """

  stage_anchor: ClassVar[StageAnchor] = "pre_selection"

  def __init__(
    self,
    local_search: LocalSearchStrategy,
    evaluation: EvaluationStrategy,
  ) -> None:
    super().__init__(local_search, evaluation)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Execute local search and re-evaluation on state.population if state.generation == 1."""
    if state.generation != 1:
      return

    state.population = timer.measure(
      "local_search",
      self.local_search.improve_batch,
      state.population,
      state.context,
    )
    state.fitness = timer.measure(
      "evaluation",
      self.evaluation.evaluate_batch,
      state.population,
      state.context,
    )


class EliteOnlyLocalSearchStep(_BaseLocalSearchPipelineStep):
  """
  Apply local search exclusively to the top-k lowest-cost routes in `state.population` (DP11-10).

  Operates on the population after survival selection, preserving relative order of
  unrefined individuals and without displacing non-elite routes.
  """

  stage_anchor: ClassVar[StageAnchor] = "post_survival"

  def __init__(
    self,
    local_search: LocalSearchStrategy,
    evaluation: EvaluationStrategy,
    elite_count: int = 10,
  ) -> None:
    super().__init__(local_search, evaluation)
    if elite_count <= 0:
      raise ValueError(f"elite_count must be > 0, got {elite_count}")
    self.elite_count: int = elite_count

  def public_params(self) -> dict[str, Any]:
    """Expose `elite_count` so checkpoints record the refined fraction."""
    return {"elite_count": self.elite_count}

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Refine only the elite_count individuals with lowest fitness."""
    pop_size: int = int(state.population.shape[0])
    k: int = min(self.elite_count, pop_size)
    if k <= 0:
      return

    is_gpu: bool = is_gpu_array(state.fitness)
    xp: Any = cupy_backend() if is_gpu else np

    elite_indices = xp.argsort(state.fitness)[:k]
    elite_routes = state.population[elite_indices]

    improved_elites = timer.measure(
      "local_search",
      self.local_search.improve_batch,
      elite_routes,
      state.context,
    )
    improved_fitness = timer.measure(
      "evaluation",
      self.evaluation.evaluate_batch,
      improved_elites,
      state.context,
    )

    state.population[elite_indices] = improved_elites
    state.fitness[elite_indices] = improved_fitness


class PeriodicLocalSearchStep:
  """
  Execute local search once every `interval` generations (DP11-10).

  Supports target='offspring' (before evaluation) or target='population' (after survival,
  requiring re-evaluation).
  """

  def __init__(
    self,
    local_search: LocalSearchStrategy,
    interval: int = 10,
    target: Literal["offspring", "population"] = "population",
    re_evaluate: EvaluationStrategy | None = None,
  ) -> None:
    if local_search is None:
      raise ValueError("PeriodicLocalSearchStep requires a local_search strategy")
    if interval <= 0:
      raise ValueError(f"interval must be > 0, got {interval}")
    if target not in ("offspring", "population"):
      raise ValueError(f"target must be 'offspring' or 'population', got {target!r}")
    if target == "population" and re_evaluate is None:
      raise ValueError("re_evaluate EvaluationStrategy is required when target='population'")
    self.local_search: LocalSearchStrategy = local_search
    self.interval: int = interval
    self.target: Literal["offspring", "population"] = target
    self.re_evaluate: EvaluationStrategy | None = re_evaluate

  @property
  def stage_anchor(self) -> StageAnchor:
    """Refining offspring must precede its evaluation; refining survivors must follow survival."""
    return "pre_evaluation" if self.target == "offspring" else "post_survival"

  def strategies(self) -> list[Any]:
    """Return the local search strategy and the optional post-refinement evaluator."""
    owned: list[Any] = [self.local_search]
    if self.re_evaluate is not None:
      owned.append(self.re_evaluate)
    return owned

  def public_params(self) -> dict[str, Any]:
    """Expose the cadence and the refinement target so checkpoints stay reproducible."""
    return {"interval": self.interval, "target": self.target}

  def warmup(self, context: ProblemContext) -> None:
    """Compile the kernels this step launches, independent of `interval` (`BUG-25`).

    `run` only fires on generation multiples of `interval`; a one-generation
    warm-up therefore never reached `two_opt_batch.cu`, which was then compiled by
    the first *measured* generation-10 hit (26.27x spike). This hook ignores the
    cadence by construction.
    """
    _warmup_owned_strategies(self.strategies(), context)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Execute local search if state.generation % interval == 0."""
    if state.generation % self.interval != 0:
      return

    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("PeriodicLocalSearchStep(target='offspring') called before offspring exists.")
      state.offspring = timer.measure(
        "local_search",
        self.local_search.improve_batch,
        state.offspring,
        state.context,
      )
    else:
      state.population = timer.measure(
        "local_search",
        self.local_search.improve_batch,
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


PIPELINE_STEP_REGISTRY: dict[str, type] = {
  "EliteOnlyLocalSearchStep": EliteOnlyLocalSearchStep,
  "PeriodicLocalSearchStep": PeriodicLocalSearchStep,
  "PostInitLocalSearchStep": PostInitLocalSearchStep,
}

__all__: list[str] = [
  "PIPELINE_STEP_REGISTRY",
  "EliteOnlyLocalSearchStep",
  "PeriodicLocalSearchStep",
  "PostInitLocalSearchStep",
  "StageAnchor",
]
