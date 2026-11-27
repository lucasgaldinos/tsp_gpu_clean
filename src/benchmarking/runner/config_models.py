"""Typed configuration dataclasses for GA parameters, algorithms, benchmark settings, and problems."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, cast

from data_models.problem import VALID_PROBLEM_TYPES

from .spec_resolver import VALID_STRATEGY_SLOTS, build_component_spec, load_external_module

if TYPE_CHECKING:
  from collections.abc import Mapping
  from inspect import Signature
  from pathlib import Path

  from algorithms.metaheuristics.genetic_algorithm import GenerationStep
  from protocols.types import LocalSearchStage


@dataclass
class GAParams:
  """Default Genetic Algorithm parameters."""

  population_size: int = 256
  mutation_rate: float = 0.02
  seed: int = 42

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for passing to algorithm constructors."""
    return {
      "population_size": self.population_size,
      "mutation_rate": self.mutation_rate,
      "seed": self.seed,
    }


@dataclass
class AlgorithmConfig:
  """Configuration for a single algorithm variant (modular `GeneticAlgorithm` or monolithic class)."""

  name: str
  use_gpu: bool
  description: str
  metaheuristic: str = "GeneticAlgorithm"
  class_module: str | None = None
  class_name: str | None = None
  params: dict[str, Any] = dc_field(default_factory=dict)
  strategies: dict[str, Any] = dc_field(default_factory=dict)
  source_dir: Path | None = dc_field(default=None, repr=False, compare=False)

  def __post_init__(self) -> None:
    if self.params is None:
      self.params = {}
    if self.strategies is None:
      self.strategies = {}

  def get_class(self) -> type:
    """Dynamically import and return the algorithm class for monolithic configs."""
    if not self.class_module or not self.class_name:
      from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm

      return GeneticAlgorithm
    module = load_external_module(self.class_module, base_dir=self.source_dir)
    return cast("type", getattr(module, self.class_name))

  def instantiate(
    self,
    default_params: GAParams,
    seed: int | None = None,
    local_search_stage: str | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  ) -> Any:
    """Instantiate the algorithm, applying optional seed, parameter, and strategy overrides."""
    for slot in self.strategies:
      if slot not in VALID_STRATEGY_SLOTS:
        raise ValueError(
          f"Unknown strategy slot '{slot}' in algorithm '{self.name}'. Valid slots: {sorted(VALID_STRATEGY_SLOTS)}"
        )
    if strategy_overrides:
      for slot in strategy_overrides:
        if slot not in VALID_STRATEGY_SLOTS:
          raise ValueError(f"Unknown strategy slot '{slot}'. Valid slots: {sorted(VALID_STRATEGY_SLOTS)}")

    if self.class_module and self.class_name:
      return self._instantiate_monolithic(
        default_params=default_params,
        seed=seed,
        local_search_stage=local_search_stage,
        param_overrides=param_overrides,
        strategy_overrides=strategy_overrides,
      )

    return self._instantiate_modular(
      default_params=default_params,
      seed=seed,
      local_search_stage=local_search_stage,
      param_overrides=param_overrides,
      strategy_overrides=strategy_overrides,
    )

  def _instantiate_monolithic(
    self,
    *,
    default_params: GAParams,
    seed: int | None,
    local_search_stage: str | None,
    param_overrides: Mapping[str, Any] | None,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
  ) -> Any:
    cls = self.get_class()
    kwargs: dict[str, Any] = {
      "population_size": default_params.population_size,
      "mutation_rate": default_params.mutation_rate,
      "seed": seed if seed is not None else default_params.seed,
    }
    kwargs.update(self.params)
    if param_overrides:
      kwargs.update(param_overrides)
    if seed is not None:
      kwargs["seed"] = seed
    if local_search_stage is not None:
      kwargs["local_search_stage"] = local_search_stage

    if strategy_overrides:
      if "local_search" in strategy_overrides and "max_iterations" in strategy_overrides["local_search"]:
        kwargs["two_opt_iterations"] = strategy_overrides["local_search"]["max_iterations"]
      if "selection" in strategy_overrides and "k" in strategy_overrides["selection"]:
        kwargs["tournament_size"] = strategy_overrides["selection"]["k"]

    sig: Signature = inspect.signature(cls)
    has_var_kw: bool = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_kw:
      kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**kwargs)

  def _instantiate_modular(
    self,
    *,
    default_params: GAParams,
    seed: int | None,
    local_search_stage: str | None,
    param_overrides: Mapping[str, Any] | None,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
  ) -> Any:
    from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
    from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
    from algorithms.strategies import STRATEGY_REGISTRY, SUB_STRATEGY_REGISTRY

    if seed is not None:
      resolved_seed: int | Any = seed
    elif param_overrides and "seed" in param_overrides:
      resolved_seed = param_overrides["seed"]
    else:
      resolved_seed = self.params.get("seed", default_params.seed)

    def _build_sub_strategy(sub_slot: str) -> Any:
      raw_spec = (
        param_overrides.get(sub_slot)
        if (param_overrides and sub_slot in param_overrides)
        else self.params.get(sub_slot)
      )
      return build_component_spec(
        raw_spec,
        SUB_STRATEGY_REGISTRY,
        "sub-strategy",
        resolved_seed=resolved_seed,
        slot=sub_slot,
        base_dir=self.source_dir,
        strategy_registry=STRATEGY_REGISTRY,
        sub_strategy_registry=SUB_STRATEGY_REGISTRY,
      )

    shared_sub_defaults = {
      "penalty_policy": _build_sub_strategy("penalty_policy"),
      "decoder": _build_sub_strategy("decoder"),
      "bin_packing": _build_sub_strategy("bin_packing"),
    }

    def _build_slot(slot_name: str) -> Any:
      return build_component_spec(
        self.strategies.get(slot_name),
        STRATEGY_REGISTRY,
        "strategy",
        resolved_seed=resolved_seed,
        slot=slot_name,
        base_dir=self.source_dir,
        strategy_overrides=strategy_overrides,
        default_kwargs={k: v for k, v in shared_sub_defaults.items() if v is not None} or None,
        strategy_registry=STRATEGY_REGISTRY,
        sub_strategy_registry=SUB_STRATEGY_REGISTRY,
      )

    init_strat = _build_slot("initialization")
    sel_strat = _build_slot("selection")
    cx_strat = _build_slot("crossover")
    mut_strat = _build_slot("mutation")
    eval_strat = _build_slot("evaluation")
    ls_strat = _build_slot("local_search")

    built_pipeline: list[GenerationStep] | None = None
    pipeline_specs = (
      param_overrides.get("pipeline")
      if (param_overrides and "pipeline" in param_overrides)
      else self.params.get("pipeline")
    )
    if pipeline_specs is not None:
      built_pipeline = []
      step_defaults = {
        "local_search": ls_strat,
        "evaluation": eval_strat,
        "re_evaluate": eval_strat,
        **shared_sub_defaults,
      }
      for step_spec in pipeline_specs:
        step_inst = build_component_spec(
          step_spec,
          PIPELINE_STEP_REGISTRY,
          "pipeline step",
          resolved_seed=resolved_seed,
          base_dir=self.source_dir,
          default_kwargs=step_defaults,
          strategy_registry=STRATEGY_REGISTRY,
          sub_strategy_registry=SUB_STRATEGY_REGISTRY,
        )
        if step_inst is not None:
          built_pipeline.append(step_inst)

    effective_ls_stage = cast(
      "LocalSearchStage",
      local_search_stage
      or (param_overrides.get("local_search_stage") if param_overrides else None)
      or self.params.get("local_search_stage", "post_mutation"),
    )

    pop_size = self.params.get("population_size", default_params.population_size)
    if param_overrides and "population_size" in param_overrides:
      pop_size = param_overrides["population_size"]

    mut_rate = self.params.get("mutation_rate", default_params.mutation_rate)
    if param_overrides and "mutation_rate" in param_overrides:
      mut_rate = param_overrides["mutation_rate"]

    return GeneticAlgorithm(
      population_size=int(pop_size),
      mutation_rate=float(mut_rate),
      seed=resolved_seed,
      initialization=init_strat,
      selection=sel_strat,
      crossover=cx_strat,
      mutation=mut_strat,
      evaluation=eval_strat,
      local_search=ls_strat,
      local_search_stage=effective_ls_stage,
      pipeline=built_pipeline,
    )


@dataclass
class BenchmarkConfig:
  """Benchmark execution parameters."""

  repetitions: int = 10
  patience: int | None = None
  max_generations: int | None = None
  cpu_size_threshold: int | None = None
  incremental_runs: bool = False
  line_width: int = 80

  @property
  def separator(self) -> str:
    """Full-width separator line."""
    return "=" * self.line_width

  @property
  def short_sep(self) -> str:
    """Full-width dash separator."""
    return "-" * self.line_width


@dataclass
class ProblemConfig:
  """Configuration for a single routing problem instance with optional declarative overrides (`DP17-10`)."""

  name: str
  optimal: float | None
  size: int
  problem_type: str | None = None
  return_to_depot: bool | None = None
  max_vehicles: int | None = None
  max_route_distance: float | None = None
  capacity: int | None = None

  def __post_init__(self) -> None:
    if self.problem_type is not None:
      if not isinstance(self.problem_type, str) or self.problem_type.upper().strip() not in VALID_PROBLEM_TYPES:
        raise ValueError(
          f"Problem '{self.name}' 'problem_type' must be one of {sorted(VALID_PROBLEM_TYPES)}, "
          f"got {self.problem_type!r}"
        )
      self.problem_type = self.problem_type.upper().strip()

    if self.return_to_depot is not None and not isinstance(self.return_to_depot, bool):
      raise ValueError(
        f"Problem '{self.name}' 'return_to_depot' must be a boolean, got {type(self.return_to_depot).__name__}"
      )

    if self.max_vehicles is not None:
      if isinstance(self.max_vehicles, bool) or not isinstance(self.max_vehicles, int) or self.max_vehicles < 1:
        raise ValueError(
          f"Problem '{self.name}' 'max_vehicles' must be an integer >= 1, got {self.max_vehicles!r}"
        )

    if self.capacity is not None:
      if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
        raise ValueError(f"Problem '{self.name}' 'capacity' must be an integer >= 1, got {self.capacity!r}")

    if self.max_route_distance is not None:
      if (
        isinstance(self.max_route_distance, bool)
        or not isinstance(self.max_route_distance, (int, float))
        or not math.isfinite(float(self.max_route_distance))
        or float(self.max_route_distance) <= 0.0
      ):
        raise ValueError(
          f"Problem '{self.name}' 'max_route_distance' must be a finite number > 0, got {self.max_route_distance!r}"
        )
      self.max_route_distance = float(self.max_route_distance)

  def has_constraint_overrides(self) -> bool:
    """Return True when any declarative VRP constraint override is explicitly set."""
    return any(
      val is not None
      for val in (self.return_to_depot, self.max_vehicles, self.max_route_distance, self.capacity)
    )


__all__: list[str] = [
  "AlgorithmConfig",
  "BenchmarkConfig",
  "GAParams",
  "ProblemConfig",
]
