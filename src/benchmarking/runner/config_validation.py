"""Per-section validation helpers for resolved benchmark configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, get_args

from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY

from .spec_resolver import MODULAR_ALGO_PARAMS, SUB_STRATEGY_SLOTS, VALID_STRATEGY_SLOTS, _validate_class_spec

if TYPE_CHECKING:
  from .config_models import AlgorithmConfig, BenchmarkConfig, GAParams, ProblemConfig


def _validate_ga_params(ga_params: GAParams) -> list[str]:
  """Validate default GA parameters."""
  warnings: list[str] = []
  if ga_params.population_size <= 0:
    warnings.append(f"Invalid population_size: {ga_params.population_size} (must be > 0)")
  if not (0.0 <= ga_params.mutation_rate <= 1.0):
    warnings.append(f"Invalid mutation_rate: {ga_params.mutation_rate} (must be in [0, 1])")
  return warnings


def _validate_algorithm_params(
  algo_name: str,
  algo_cfg: AlgorithmConfig,
  valid_ls_stages: frozenset[str],
) -> list[str]:
  """Validate scalar parameters and slot param bounds for a single algorithm entry."""
  warnings: list[str] = []
  is_monolithic = bool(algo_cfg.class_module and algo_cfg.class_name)
  if bool(algo_cfg.class_module) != bool(algo_cfg.class_name):
    warnings.append(f"Algorithm '{algo_name}': 'class_module' and 'class_name' must be supplied together")
  if not is_monolithic and algo_cfg.metaheuristic != "GeneticAlgorithm":
    warnings.append(
      f"Algorithm '{algo_name}': metaheuristic '{algo_cfg.metaheuristic}' requires 'class_module' and 'class_name'"
    )
  if not is_monolithic:
    extra_params = sorted(set(algo_cfg.params.keys()) - MODULAR_ALGO_PARAMS)
    if extra_params:
      warnings.append(
        f"Algorithm '{algo_name}': unsupported param(s) for modular GeneticAlgorithm: {extra_params}. "
        f"Accepted: {sorted(MODULAR_ALGO_PARAMS)}"
      )

  if "population_size" in algo_cfg.params:
    algo_pop = algo_cfg.params["population_size"]
    if isinstance(algo_pop, bool) or not isinstance(algo_pop, int) or algo_pop <= 0:
      warnings.append(f"Invalid population_size in {algo_name}.params: {algo_pop!r} (must be > 0)")

  if "mutation_rate" in algo_cfg.params:
    algo_mut = algo_cfg.params["mutation_rate"]
    if isinstance(algo_mut, bool) or not isinstance(algo_mut, (int, float)) or not (0.0 <= float(algo_mut) <= 1.0):
      warnings.append(f"Invalid mutation_rate in {algo_name}.params: {algo_mut!r} (must be in [0, 1])")

  if "local_search_stage" in algo_cfg.params:
    algo_stage = algo_cfg.params["local_search_stage"]
    if not isinstance(algo_stage, str) or algo_stage not in valid_ls_stages:
      warnings.append(
        f"Invalid local_search_stage in {algo_name}.params: {algo_stage!r} "
        f"(expected one of {sorted(valid_ls_stages)})"
      )

  sel_params = algo_cfg.strategies.get("selection", {}).get("params", {})
  tournament_k = sel_params.get("k")
  if tournament_k is not None and (
    isinstance(tournament_k, bool) or not isinstance(tournament_k, int) or tournament_k <= 0
  ):
    warnings.append(f"Invalid k in {algo_name}.selection: {tournament_k} (must be > 0)")

  ls_params = algo_cfg.strategies.get("local_search", {}).get("params", {})
  max_iterations = ls_params.get("max_iterations")
  if max_iterations is not None and (
    isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0
  ):
    warnings.append(f"Invalid max_iterations in {algo_name}.local_search: {max_iterations} (must be >= 0)")
  return warnings


def _validate_algorithms(algorithms: Mapping[str, AlgorithmConfig]) -> list[str]:
  """Validate all algorithm entries, class resolution, strategy slots, and pipeline specs."""
  from algorithms.strategies import STRATEGY_REGISTRY, SUB_STRATEGY_REGISTRY
  from protocols.types import LocalSearchStage as _LocalSearchStage

  valid_ls_stages: frozenset[str] = frozenset(get_args(getattr(_LocalSearchStage, "__value__", _LocalSearchStage)))
  warnings: list[str] = []

  for algo_name, algo_cfg in algorithms.items():
    warnings.extend(_validate_algorithm_params(algo_name, algo_cfg, valid_ls_stages))

  for name, algo in algorithms.items():
    try:
      algo.get_class()
    except (ModuleNotFoundError, AttributeError, FileNotFoundError, ImportError, OSError) as exc:
      warnings.append(f"Algorithm '{name}' class not found: {exc}")
      continue

    if algo.class_module and algo.class_name:
      continue

    for slot, spec in algo.strategies.items():
      if slot not in VALID_STRATEGY_SLOTS:
        warnings.append(
          f"Algorithm '{name}': unknown strategy slot '{slot}'. Valid slots: {sorted(VALID_STRATEGY_SLOTS)}"
        )
        continue
      if isinstance(spec, Mapping):
        warnings.extend(
          _validate_class_spec(
            spec,
            STRATEGY_REGISTRY,
            "strategy",
            context_label=f"Algorithm '{name}' slot '{slot}'",
            slot=slot,
            strategy_registry=STRATEGY_REGISTRY,
            sub_strategy_registry=SUB_STRATEGY_REGISTRY,
            base_dir=algo.source_dir,
          )
        )

    for sub_slot in sorted(SUB_STRATEGY_SLOTS):
      sub_spec = algo.params.get(sub_slot)
      if sub_spec is None:
        continue
      if not isinstance(sub_spec, Mapping):
        warnings.append(f"Algorithm '{name}' params.{sub_slot}: sub-strategy spec must be a mapping")
      else:
        warnings.extend(
          _validate_class_spec(
            sub_spec,
            SUB_STRATEGY_REGISTRY,
            "sub-strategy",
            context_label=f"Algorithm '{name}' params.{sub_slot}",
            slot=sub_slot,
            strategy_registry=STRATEGY_REGISTRY,
            sub_strategy_registry=SUB_STRATEGY_REGISTRY,
            base_dir=algo.source_dir,
          )
        )

    pipeline_specs = algo.params.get("pipeline")
    if pipeline_specs is not None:
      if not isinstance(pipeline_specs, list):
        warnings.append(f"Algorithm '{name}': 'pipeline' must be a list of step specs")
      else:
        for idx, step_spec in enumerate(pipeline_specs):
          if not isinstance(step_spec, Mapping):
            warnings.append(f"Algorithm '{name}' pipeline[{idx}]: step spec must be a mapping")
          else:
            warnings.extend(
              _validate_class_spec(
                step_spec,
                PIPELINE_STEP_REGISTRY,
                "pipeline step",
                context_label=f"Algorithm '{name}' pipeline[{idx}]",
                strategy_registry=STRATEGY_REGISTRY,
                sub_strategy_registry=SUB_STRATEGY_REGISTRY,
                base_dir=algo.source_dir,
              )
            )
  return warnings


def _validate_benchmark(benchmark: BenchmarkConfig) -> list[str]:
  """Validate benchmark execution parameters."""
  warnings: list[str] = []
  if benchmark.repetitions <= 0:
    warnings.append(f"Invalid repetitions: {benchmark.repetitions} (must be > 0)")
  if benchmark.patience is not None and benchmark.patience <= 0:
    warnings.append(f"Invalid patience: {benchmark.patience} (must be > 0 or None)")
  if benchmark.max_generations is not None and benchmark.max_generations <= 0:
    warnings.append(f"Invalid max_generations: {benchmark.max_generations} (must be > 0 or None)")
  if benchmark.cpu_size_threshold is not None and benchmark.cpu_size_threshold <= 0:
    warnings.append(f"Invalid cpu_size_threshold: {benchmark.cpu_size_threshold} (must be > 0)")
  return warnings


def _validate_problems(problems: Sequence[ProblemConfig]) -> list[str]:
  """Validate problem catalogue entries and detect duplicate names."""
  warnings: list[str] = []
  if not problems:
    warnings.append("No problems defined in problems.json")
    return warnings

  for problem in problems:
    if problem.size <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid size: {problem.size} (must be > 0)")
    if problem.optimal is not None and problem.optimal <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid optimal: {problem.optimal} (must be > 0)")

  problem_names = [p.name for p in problems]
  if len(problem_names) != len(set(problem_names)):
    duplicates = [name for name in problem_names if problem_names.count(name) > 1]
    warnings.append(f"Duplicate problem names found: {set(duplicates)}")
  return warnings


def validate_configs(configs: dict[str, Any]) -> list[str]:
  """Validate loaded configurations for consistency and completeness."""
  warnings: list[str] = []
  warnings.extend(_validate_ga_params(configs["ga_params"]))
  warnings.extend(_validate_algorithms(configs["algorithms"]))
  warnings.extend(_validate_benchmark(configs["benchmark"]))
  warnings.extend(_validate_problems(configs["problems"]))
  return warnings


__all__: list[str] = [
  "_validate_algorithms",
  "_validate_benchmark",
  "_validate_ga_params",
  "_validate_problems",
  "validate_configs",
]
