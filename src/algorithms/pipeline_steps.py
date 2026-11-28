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

from algorithms.vrp_pipeline_steps import FleetBinPackStep, PenaltyAdaptationStep
from protocols.backend import cupy_backend, require_gpu
from protocols.vrp_strategies import RouteAssessment, assess_with_evaluator
from utils.route_helpers import ensure_routes_batch, is_gpu_array

if TYPE_CHECKING:
  from algorithms.metaheuristics.genetic_algorithm import GenerationState
  from protocols.algorithm_strategies import EvaluationStrategy, LocalSearchStrategy
  from protocols.problem_context import ProblemContext
  from protocols.vrp_strategies import PenaltyPolicy, RouteDecoder
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
    pop_on_gpu = is_gpu_array(state.population)
    keep_gpu = bool(pop_on_gpu and getattr(self.evaluation, "return_gpu", False))
    assessment = timer.measure(
      "evaluation",
      assess_with_evaluator,
      self.evaluation,
      state.population,
      state.context,
      return_gpu=keep_gpu,
    )
    state.population_assessment = assessment
    state.fitness = assessment.objectives


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
    pop_on_gpu = is_gpu_array(improved_elites)
    keep_gpu = bool(pop_on_gpu and getattr(self.evaluation, "return_gpu", False))
    elite_assessment = timer.measure(
      "evaluation",
      assess_with_evaluator,
      self.evaluation,
      improved_elites,
      state.context,
      return_gpu=keep_gpu,
    )

    state.population[elite_indices] = improved_elites
    state.fitness[elite_indices] = elite_assessment.objectives
    if state.population_assessment is not None:
      pa = state.population_assessment
      pa.objectives[elite_indices] = elite_assessment.objectives
      pa.travel_costs[elite_indices] = elite_assessment.travel_costs
      pa.is_feasible[elite_indices] = elite_assessment.is_feasible
      pa.violations[elite_indices] = elite_assessment.violations
      pa.normalized_violations[elite_indices] = elite_assessment.normalized_violations
      if pa.raw_violation_components is not None and elite_assessment.raw_violation_components is not None:
        pa.raw_violation_components[elite_indices] = elite_assessment.raw_violation_components
      if pa.num_vehicles is not None and elite_assessment.num_vehicles is not None:
        pa.num_vehicles[elite_indices] = elite_assessment.num_vehicles


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
        pop_on_gpu = is_gpu_array(state.population)
        keep_gpu = bool(pop_on_gpu and getattr(self.re_evaluate, "return_gpu", False))
        assessment = timer.measure(
          "evaluation",
          assess_with_evaluator,
          self.re_evaluate,
          state.population,
          state.context,
          return_gpu=keep_gpu,
        )
        state.population_assessment = assessment
        state.fitness = assessment.objectives


class SplitDecodeStep:
  """Decode `(N, n_cust)` Giant-Tours into `(N, W)` `0`-delimited routes (`DP17-2`, `DP17-3`).

  Note:
      `LocalSearchCPURouteTwoOptStar` and `LocalSearchGPURouteTwoOptStar` decode Giant-Tour `(N, n - 1)` inputs
      natively, so `PeriodicLocalSearchStep` alone suffices without inserting off-generation
      `SplitDecodeStep` passes. When present, `SplitDecodeStep` skips already `0`-delimited
      batches (`width >= context.dimension`) to prevent redundant double `Split` (`BUG-112`).
  """

  def __init__(
    self,
    mode: Literal["prins", "greedy"] = "prins",
    target: Literal["offspring", "population"] = "offspring",
    stage_anchor: StageAnchor = "pre_evaluation",
    use_gpu: bool | None = None,
    max_vehicles: int | None = None,
    max_shared_customers: int = 2048,
    max_matrix_dimension: int | None = None,
    penalty_policy: PenaltyPolicy | None = None,
    decoder: RouteDecoder | None = None,
  ) -> None:
    if mode not in ("prins", "greedy"):
      raise ValueError(f"mode must be 'prins' or 'greedy', got {mode!r}")
    if target not in ("offspring", "population"):
      raise ValueError(f"target must be 'offspring' or 'population', got {target!r}")
    if stage_anchor not in ("pre_selection", "pre_evaluation", "post_survival"):
      raise ValueError(f"Invalid stage_anchor {stage_anchor!r}")
    self.mode: Literal["prins", "greedy"] = mode
    self.target: Literal["offspring", "population"] = target
    self._stage_anchor: StageAnchor = stage_anchor
    if decoder is not None:
      self.decoder: Any = decoder
    else:
      from algorithms.strategies.decoding import GreedySplitDecoder, PrinsSplitDecoder

      decoder_cls = PrinsSplitDecoder if mode == "prins" else GreedySplitDecoder
      self.decoder = decoder_cls(
        use_gpu=use_gpu,
        max_vehicles=max_vehicles,
        max_shared_customers=max_shared_customers,
        max_matrix_dimension=max_matrix_dimension,
        penalty_policy=penalty_policy,
      )

  @property
  def stage_anchor(self) -> StageAnchor:
    """Where `GeneticAlgorithm._compose_pipeline` inserts this decoding step."""
    return self._stage_anchor

  def strategies(self) -> list[Any]:
    """Return the owned Split decoder so telemetry and warm-up propagate."""
    return [self.decoder]

  def public_params(self) -> dict[str, Any]:
    """Expose declarative parameters for checkpoint serialization."""
    return {"mode": self.mode, "target": self.target, "stage_anchor": self._stage_anchor}

  def warmup(self, context: ProblemContext) -> None:
    """Compile CUDA Split/Strip kernels ahead of the measured loop (`BUG-25`)."""
    _warmup_owned_strategies(self.strategies(), context)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Transform `state.offspring` or `state.population` from Giant-Tour to `0`-delimited routes."""
    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("SplitDecodeStep target='offspring' called before state.offspring was populated.")
      batch, _ = ensure_routes_batch(state.offspring)
      if batch.shape[1] >= state.context.dimension:
        return
      res = timer.measure("decoding", self.decoder.decode_batch, state.offspring, state.context)
      state.offspring = res.routes
    else:
      batch, _ = ensure_routes_batch(state.population)
      if batch.shape[1] >= state.context.dimension:
        return
      res = timer.measure("decoding", self.decoder.decode_batch, state.population, state.context)
      state.population = res.routes
      if isinstance(state.population_assessment, RouteAssessment):
        state.population_assessment = None


class GiantTourEncodeStep:
  """Strip `0` depot delimiters from `(N, W)` routes back to `(N, n_cust)` Giant-Tours (`DP17-2`)."""

  def __init__(
    self,
    target: Literal["offspring", "population"] = "offspring",
    stage_anchor: StageAnchor = "pre_evaluation",
  ) -> None:
    if target not in ("offspring", "population"):
      raise ValueError(f"target must be 'offspring' or 'population', got {target!r}")
    if stage_anchor not in ("pre_selection", "pre_evaluation", "post_survival"):
      raise ValueError(f"Invalid stage_anchor {stage_anchor!r}")
    self.target: Literal["offspring", "population"] = target
    self._stage_anchor: StageAnchor = stage_anchor

  @property
  def stage_anchor(self) -> StageAnchor:
    """Where `GeneticAlgorithm._compose_pipeline` inserts this encoding step."""
    return self._stage_anchor

  def strategies(self) -> list[Any]:
    """Return empty list (kernel telemetry is minimal and stateless)."""
    return []

  def public_params(self) -> dict[str, Any]:
    """Expose declarative parameters for checkpoint serialization."""
    return {"target": self.target, "stage_anchor": self._stage_anchor}

  def warmup(self, context: ProblemContext) -> None:
    """Compile and launch `strip_depot_delimiters_kernel` once when GPU is active."""
    if not hasattr(context.xp, "RawKernel") or context.dimension <= 1:
      return
    from algorithms.strategies.decoding import strip_depot_delimiters

    cp = require_gpu()
    n_cust = context.dimension - 1
    probe = cp.zeros((1, context.dimension + 1), dtype=cp.int32)
    probe[0, 1 : n_cust + 1] = cp.arange(1, context.dimension, dtype=cp.int32)
    strip_depot_delimiters(probe, n_cust)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Strip depot `0` delimiters from `state.offspring` or `state.population`."""
    from algorithms.strategies.decoding import strip_depot_delimiters

    n_customers = state.context.dimension - 1
    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("GiantTourEncodeStep target='offspring' called before state.offspring was populated.")
      batch, _ = ensure_routes_batch(state.offspring)
      if batch.shape[1] == n_customers:
        return
      state.offspring = timer.measure(
        "decoding",
        strip_depot_delimiters,
        state.offspring,
        n_customers,
      )
    else:
      batch, _ = ensure_routes_batch(state.population)
      if batch.shape[1] == n_customers:
        return
      state.population = timer.measure(
        "decoding",
        strip_depot_delimiters,
        state.population,
        n_customers,
      )


PIPELINE_STEP_REGISTRY: dict[str, type] = {
  "EliteOnlyLocalSearchStep": EliteOnlyLocalSearchStep,
  "PeriodicLocalSearchStep": PeriodicLocalSearchStep,
  "PostInitLocalSearchStep": PostInitLocalSearchStep,
}


def register_pipeline_step(name: str, step_cls: type, *, overwrite: bool = False) -> None:
  """Register a custom ``GenerationStep`` class in ``PIPELINE_STEP_REGISTRY`` without modifying ``src/``.

  Args:
      name: Unique registry key referenced from declarative JSON pipeline specifications.
      step_cls: Class implementing ``run(self, state, timer) -> None``.
      overwrite: Allow replacing an existing registration with the same ``name``.

  Raises:
      ValueError: If ``name`` is empty or already registered with ``overwrite=False``.
      TypeError: If ``step_cls`` is not a class or lacks a callable ``run`` method.
  """
  if not isinstance(name, str) or not name.strip():
    raise ValueError("Pipeline step name must be a non-empty string.")
  if not isinstance(step_cls, type):
    raise TypeError(f"Expected a pipeline step class (type), got {type(step_cls).__name__}.")
  if not callable(getattr(step_cls, "run", None)):
    raise TypeError(f"Pipeline step class '{step_cls.__name__}' must define a callable 'run(state, timer)' method.")
  if name in PIPELINE_STEP_REGISTRY and not overwrite:
    raise ValueError(f"Pipeline step '{name}' is already registered. Pass overwrite=True to replace it.")
  PIPELINE_STEP_REGISTRY[name] = step_cls


register_pipeline_step("SplitDecodeStep", SplitDecodeStep)
register_pipeline_step("GiantTourEncodeStep", GiantTourEncodeStep)
register_pipeline_step("PenaltyAdaptationStep", PenaltyAdaptationStep)
register_pipeline_step("FleetBinPackStep", FleetBinPackStep)


__all__: list[str] = [
  "PIPELINE_STEP_REGISTRY",
  "EliteOnlyLocalSearchStep",
  "FleetBinPackStep",
  "GiantTourEncodeStep",
  "PenaltyAdaptationStep",
  "PeriodicLocalSearchStep",
  "PostInitLocalSearchStep",
  "SplitDecodeStep",
  "StageAnchor",
  "register_pipeline_step",
]
