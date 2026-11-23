"""
Configuration Loader for the Benchmark Runner
=============================================

Loads and validates JSON configuration files for the modular benchmark system.

This module provides type-safe configuration loading with dataclasses,
replacing the constants that were previously hardcoded in the chapter 4
validation script (removed during the benchmarking consolidation).

Usage:
    from pathlib import Path
    from benchmarking.runner.config_loader import load_all_configs

    configs = load_all_configs(Path("configs"))
    ga_params = configs["ga_params"]
    algorithms = configs["algorithms"]
    problems = configs["problems"]
    benchmark_config = configs["benchmark"]

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
from loaders.tsplib_loader import TsplibIngestionBundle, parse_tsplib_file, resolve_opt_tour_cost

if TYPE_CHECKING:
  from inspect import Signature

  from algorithms.metaheuristics.genetic_algorithm import GenerationStep
  from protocols.types import LocalSearchStage

# =============================================================================
# Configuration Data Classes
# =============================================================================


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
  """
  Configuration for a single algorithm variant.

  Supports both declarative strategy composition for modular `GeneticAlgorithm`
  and direct dynamic class loading for monolithic algorithms.
  """

  name: str
  use_gpu: bool
  description: str
  metaheuristic: str = "GeneticAlgorithm"
  class_module: str | None = None
  class_name: str | None = None
  params: dict[str, Any] = dc_field(default_factory=dict)
  strategies: dict[str, Any] = dc_field(default_factory=dict)

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
    module = importlib.import_module(self.class_module)
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

    # 1. Monolithic / class-based instantiation
    if self.class_module and self.class_name:
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

      import inspect

      sig: Signature = inspect.signature(cls)
      has_var_kw: bool = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
      if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
      return cls(**kwargs)

    # 2. Declarative GeneticAlgorithm strategy composition
    from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
    from algorithms.strategies import STRATEGY_REGISTRY

    resolved_seed: int | Any
    if seed is not None:
      resolved_seed = seed
    elif param_overrides and "seed" in param_overrides:
      resolved_seed = param_overrides["seed"]
    else:
      resolved_seed = self.params.get("seed", default_params.seed)

    def _build_spec(
      spec: dict[str, Any] | None,
      registry: dict[str, type],
      registry_name: str,
      slot: str | None = None,
      default_kwargs: dict[str, Any] | None = None,
    ) -> Any:
      if not spec:
        return None
      class_name = spec.get("class")
      if not class_name:
        return None

      target_cls = _resolve_class_spec(spec, registry, registry_name, slot=slot)
      params = dict(spec.get("params", {}))

      import inspect

      sig: Signature = inspect.signature(target_cls)
      has_var_kw: bool = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

      if slot and strategy_overrides and slot in strategy_overrides:
        for k, v in strategy_overrides[slot].items():
          if not has_var_kw and k not in sig.parameters:
            raise ValueError(
              f"Strategy class '{class_name}' in slot '{slot}' does not accept parameter '{k}'. "
              f"Accepted parameters: {list(sig.parameters.keys())}"
            )
          params[k] = v

      if default_kwargs:
        for k, v in default_kwargs.items():
          if k in sig.parameters and k not in params and v is not None:
            params[k] = v

      # Resolve nested strategy specifications if provided (registered or external module)
      for k, v in list(params.items()):
        if isinstance(v, dict) and "class" in v and ("module" in v or v["class"] in STRATEGY_REGISTRY):
          params[k] = _build_spec(v, STRATEGY_REGISTRY, "strategy")

      if "seed" in sig.parameters and "seed" not in params:
        params["seed"] = resolved_seed

      if not has_var_kw:
        for k in params:
          if k not in sig.parameters:
            raise ValueError(
              f"{registry_name} class '{class_name}' does not accept parameter '{k}'. "
              f"Accepted parameters: {list(sig.parameters.keys())}"
            )

      return target_cls(**params)

    init_strat = _build_spec(self.strategies.get("initialization"), STRATEGY_REGISTRY, "strategy", "initialization")
    sel_strat = _build_spec(self.strategies.get("selection"), STRATEGY_REGISTRY, "strategy", "selection")
    cx_strat = _build_spec(self.strategies.get("crossover"), STRATEGY_REGISTRY, "strategy", "crossover")
    mut_strat = _build_spec(self.strategies.get("mutation"), STRATEGY_REGISTRY, "strategy", "mutation")
    eval_strat = _build_spec(self.strategies.get("evaluation"), STRATEGY_REGISTRY, "strategy", "evaluation")
    ls_strat = _build_spec(self.strategies.get("local_search"), STRATEGY_REGISTRY, "strategy", "local_search")

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
      }
      for step_spec in pipeline_specs:
        step_inst = _build_spec(
          step_spec,
          PIPELINE_STEP_REGISTRY,
          "pipeline step",
          default_kwargs=step_defaults,
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
  """Configuration for a single TSP problem instance."""

  name: str
  optimal: float | None
  size: int


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _resolve_class_spec(
  spec: Mapping[str, Any],
  registry: Mapping[str, type],
  registry_name: str,
  slot: str | None = None,
) -> type:
  """Resolve a built-in registry or external ``{"module", "class"}`` specification to a class."""
  class_name = spec.get("class")
  if not isinstance(class_name, str) or not class_name.strip():
    slot_ctx = f" in slot '{slot}'" if slot else ""
    raise ValueError(f"Missing or invalid 'class' name for {registry_name}{slot_ctx}.")

  module_name = spec.get("module")
  if module_name is not None:
    if not isinstance(module_name, str) or not module_name.strip():
      raise ValueError(f"Invalid 'module' value {module_name!r} for {registry_name} '{class_name}'.")
    mod = importlib.import_module(module_name)
    loaded_cls = getattr(mod, class_name, None)
    if not isinstance(loaded_cls, type):
      raise ValueError(f"External {registry_name} '{class_name}' not found as a class in module '{module_name}'.")
    return loaded_cls

  if class_name not in registry:
    slot_ctx = f" in slot '{slot}'" if slot else ""
    raise ValueError(
      f"Unknown {registry_name} class '{class_name}'{slot_ctx}. "
      f"Available: {sorted(registry.keys())} (or provide 'module' to import an external class)"
    )
  return registry[class_name]


VALID_STRATEGY_SLOTS: frozenset[str] = frozenset(
  {"initialization", "selection", "crossover", "mutation", "evaluation", "local_search"}
)


def _slot_protocol_map() -> dict[str, type]:
  """Return the ``@runtime_checkable`` protocol expected for each declarative strategy slot."""
  from protocols.algorithm_strategies import (
    CrossoverStrategy,
    EvaluationStrategy,
    InitializationStrategy,
    LocalSearchStrategy,
    MutationStrategy,
    SelectionStrategy,
  )

  return {
    "initialization": InitializationStrategy,
    "selection": SelectionStrategy,
    "crossover": CrossoverStrategy,
    "mutation": MutationStrategy,
    "evaluation": EvaluationStrategy,
    "local_search": LocalSearchStrategy,
  }


def _validate_class_spec(
  spec: Mapping[str, Any] | None,
  registry: Mapping[str, type],
  registry_name: str,
  *,
  context_label: str,
  slot: str | None = None,
  strategy_registry: Mapping[str, type] | None = None,
) -> list[str]:
  """Validate class resolution, slot-protocol conformance, constructor params, and nested specs."""
  if not spec or ("class" not in spec and "module" not in spec):
    return []

  import inspect

  if slot is not None and registry_name == "strategy" and slot not in VALID_STRATEGY_SLOTS:
    return [f"{context_label}: unknown strategy slot '{slot}'. Valid slots: {sorted(VALID_STRATEGY_SLOTS)}"]

  errors: list[str] = []
  try:
    target_cls = _resolve_class_spec(spec, registry, registry_name, slot=slot)
  except (ImportError, ValueError, AttributeError) as exc:
    return [f"{context_label}: {exc}"]

  if slot is not None and registry_name == "strategy":
    expected_protocol = _slot_protocol_map().get(slot)
    if expected_protocol is not None and not issubclass(target_cls, expected_protocol):
      errors.append(
        f"{context_label}: class '{target_cls.__name__}' does not implement {expected_protocol.__name__} "
        f"required for slot '{slot}'"
      )

  raw_params = spec.get("params", {})
  if not isinstance(raw_params, Mapping):
    return [f"{context_label}: 'params' must be a mapping, got {type(raw_params).__name__}"]

  sig = inspect.signature(target_cls)
  has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
  for k, v in raw_params.items():
    if not has_var_kw and k not in sig.parameters:
      errors.append(
        f"{context_label}: {registry_name} class '{target_cls.__name__}' does not accept parameter '{k}'. "
        f"Accepted parameters: {list(sig.parameters.keys())}"
      )
    if (
      strategy_registry is not None
      and isinstance(v, Mapping)
      and "class" in v
      and ("module" in v or v["class"] in strategy_registry)
    ):
      errors.extend(
        _validate_class_spec(
          v,
          strategy_registry,
          "strategy",
          context_label=f"{context_label}.params.{k}",
          strategy_registry=strategy_registry,
        )
      )
  return errors


def _resolve_config_file(path: str | Path, default_filename: str) -> Path:
  """Resolve either a direct JSON file path or a directory containing ``default_filename``."""
  candidate = Path(path)
  if candidate.is_file():
    return candidate
  if candidate.is_dir():
    return candidate / default_filename
  if candidate.suffix.lower() == ".json":
    return candidate
  return candidate / default_filename


def load_algorithms_config(
  config_path: str | Path,
  *,
  config_name: str = "algorithms.json",
) -> tuple[GAParams, dict[str, AlgorithmConfig]]:
  """
  Load an algorithms configuration file containing default parameters and variants.

  Args:
      config_path: Path to either a configuration directory or a direct JSON file.
      config_name: Configuration file name when ``config_path`` is a directory.

  Returns:
      Tuple of (GAParams, dict of algorithm_name -> AlgorithmConfig)
  """
  resolved_path: Path = _resolve_config_file(config_path, config_name)

  if not resolved_path.exists():
    raise FileNotFoundError(f"Algorithms config not found: {resolved_path}")

  with open(resolved_path) as f:
    data = json.load(f)

  params_dict = data.get("default_params") or data.get("ga_params", {})
  ga_params = GAParams(
    population_size=int(params_dict.get("population_size", 256)),
    mutation_rate=float(params_dict.get("mutation_rate", 0.02)),
    seed=int(params_dict.get("seed", 42)),
  )

  if "algorithm_configs" not in data:
    raise ValueError("algorithms.json missing 'algorithm_configs' section")

  algorithms = {}
  for name, config_dict in data["algorithm_configs"].items():
    algorithms[name] = AlgorithmConfig(
      name=name,
      metaheuristic=config_dict.get("metaheuristic", "GeneticAlgorithm"),
      class_module=config_dict.get("class_module"),
      class_name=config_dict.get("class_name"),
      use_gpu=bool(config_dict.get("use_gpu", False)),
      description=config_dict.get("description", ""),
      params=config_dict.get("params", {}),
      strategies=config_dict.get("strategies", {}),
    )

  return ga_params, algorithms


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
  """
  Load benchmark execution parameters from a JSON file or configuration directory.

  Args:
      config_path: Path to either a configuration directory (containing ``benchmark.json``)
          or a direct JSON file.

  Returns:
      BenchmarkConfig instance
  """
  resolved_path: Path = _resolve_config_file(config_path, "benchmark.json")

  if not resolved_path.exists():
    raise FileNotFoundError(f"Benchmark config not found: {resolved_path}")

  with open(resolved_path) as f:
    data = json.load(f)

  required_fields: list[str] = [
    "repetitions",
    "patience",
  ]
  for field in required_fields:
    if field not in data:
      raise ValueError(f"benchmark.json missing required field: {field}")

  raw_threshold = data.get("cpu_size_threshold")
  cpu_size_threshold = int(raw_threshold) if raw_threshold is not None else None

  raw_max_gen = data.get("max_generations")
  if isinstance(raw_max_gen, bool):
    raise ValueError(f"benchmark.json 'max_generations' must be an integer or null, got bool: {raw_max_gen!r}")
  max_generations = int(raw_max_gen) if raw_max_gen is not None else None

  line_width = data.get("display", {}).get("line_width", 80)

  return BenchmarkConfig(
    repetitions=data["repetitions"],
    patience=data["patience"],
    max_generations=max_generations,
    cpu_size_threshold=cpu_size_threshold,
    incremental_runs=bool(data.get("incremental_runs", False)),
    line_width=line_width,
  )


def load_problems_config(config_path: str | Path) -> list[ProblemConfig]:
  """
  Load TSP problem definitions from a JSON file or configuration directory.

  The document is an object with a `$schema` reference and a `problems` array, so
  editors validate it against `configs/schemas/problems.schema.json` (`DP11-5`).

  Args:
      config_path: Path to either a configuration directory (containing ``problems.json``)
          or a direct JSON file.

  Returns:
      List of ProblemConfig instances
  """
  resolved_path: Path = _resolve_config_file(config_path, "problems.json")

  if not resolved_path.exists():
    raise FileNotFoundError(f"Problems config not found: {resolved_path}")

  with open(resolved_path) as f:
    data = json.load(f)

  if not isinstance(data, dict):
    raise ValueError("problems.json must contain a JSON object with a 'problems' array")

  entries = data.get("problems")
  if not isinstance(entries, list):
    raise ValueError("problems.json must contain a JSON object with a 'problems' array")

  problems = []
  required_fields: list[str] = ["name", "optimal", "size"]

  for idx, problem_dict in enumerate(entries):
    for field in required_fields:
      if field not in problem_dict:
        raise ValueError(f"Problem at index {idx} missing required field: {field}")

    raw_optimal = problem_dict["optimal"]
    problems.append(
      ProblemConfig(
        name=problem_dict["name"],
        optimal=float(raw_optimal) if raw_optimal is not None else None,
        size=int(problem_dict["size"]),
      )
    )

  return problems


def load_all_configs(config_dir: str | Path) -> dict[str, Any]:
  """
  Load all configuration files from the specified directory.

  Args:
      config_dir: Path to configuration directory

  Returns:
      Dictionary with keys: "ga_params", "algorithms", "benchmark", "problems"
  """
  config_path = Path(config_dir)

  if not config_path.exists():
    raise FileNotFoundError(f"Configuration directory not found: {config_path}")

  if not config_path.is_dir():
    raise ValueError(f"Config path must be a directory: {config_path}")

  ga_params, algorithms = load_algorithms_config(config_path)
  benchmark: BenchmarkConfig = load_benchmark_config(config_path)
  problems: list[ProblemConfig] = load_problems_config(config_path)

  return {
    "ga_params": ga_params,
    "algorithms": algorithms,
    "benchmark": benchmark,
    "problems": problems,
  }


def resolve_benchmark_configs(
  *,
  config_dir: str | Path | None = None,
  algorithms_config: str | Path | None = None,
  benchmark_config: str | Path | None = None,
  problems_config: str | Path | None = None,
  require_algorithms: bool = True,
) -> dict[str, Any]:
  """Resolve benchmark configurations from optional individual files and/or a config directory.

  Precedence: explicit ``*_config`` path > file inside ``config_dir`` > neutral defaults
  (for ``benchmark`` and ``problems``). ``algorithms`` is required unless
  ``require_algorithms=False`` (e.g., when ``algorithm_configs_override`` is supplied).
  """
  resolved_dir: Path | None = None
  if config_dir is not None:
    resolved_dir = Path(config_dir)
    if not resolved_dir.exists():
      raise FileNotFoundError(f"Configuration directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
      raise ValueError(f"Config path must be a directory: {resolved_dir}")

  if algorithms_config is not None:
    ga_params, algorithms = load_algorithms_config(algorithms_config)
  elif resolved_dir is not None:
    ga_params, algorithms = load_algorithms_config(resolved_dir)
  elif require_algorithms:
    raise ValueError("Either config_dir or algorithms_config must be provided.")
  else:
    ga_params, algorithms = GAParams(), {}

  if benchmark_config is not None:
    benchmark = load_benchmark_config(benchmark_config)
  elif resolved_dir is not None and (resolved_dir / "benchmark.json").is_file():
    benchmark = load_benchmark_config(resolved_dir)
  else:
    benchmark = BenchmarkConfig()

  if problems_config is not None:
    problems = load_problems_config(problems_config)
  elif resolved_dir is not None and (resolved_dir / "problems.json").is_file():
    problems = load_problems_config(resolved_dir)
  else:
    problems = []

  return {
    "ga_params": ga_params,
    "algorithms": algorithms,
    "benchmark": benchmark,
    "problems": problems,
  }


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_configs(configs: dict[str, Any]) -> list[str]:
  """
  Validate loaded configurations for consistency and completeness.

  Args:
      configs: Dictionary returned by load_all_configs()

  Returns:
      List of warning messages (empty list if no issues)
  """
  warnings = []

  ga_params = configs["ga_params"]
  algorithms = configs["algorithms"]
  benchmark = configs["benchmark"]
  problems = configs["problems"]

  if ga_params.population_size <= 0:
    warnings.append(f"Invalid population_size: {ga_params.population_size} (must be > 0)")

  if not (0.0 <= ga_params.mutation_rate <= 1.0):
    warnings.append(f"Invalid mutation_rate: {ga_params.mutation_rate} (must be in [0, 1])")

  from typing import get_args

  from protocols.types import LocalSearchStage as _LocalSearchStage

  valid_ls_stages: frozenset[str] = frozenset(get_args(getattr(_LocalSearchStage, "__value__", _LocalSearchStage)))

  for algo_name, algo_cfg in algorithms.items():
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
    if tournament_k is not None and tournament_k <= 0:
      warnings.append(f"Invalid k in {algo_name}.selection: {tournament_k} (must be > 0)")
    ls_params = algo_cfg.strategies.get("local_search", {}).get("params", {})
    max_iterations = ls_params.get("max_iterations")
    if max_iterations is not None and max_iterations <= 0:
      warnings.append(f"Invalid max_iterations in {algo_name}.local_search: {max_iterations} (must be > 0)")

  if benchmark.repetitions <= 0:
    warnings.append(f"Invalid repetitions: {benchmark.repetitions} (must be > 0)")

  if benchmark.patience is not None and benchmark.patience <= 0:
    warnings.append(f"Invalid patience: {benchmark.patience} (must be > 0 or None)")

  if benchmark.max_generations is not None and benchmark.max_generations <= 0:
    warnings.append(f"Invalid max_generations: {benchmark.max_generations} (must be > 0 or None)")

  if benchmark.cpu_size_threshold is not None and benchmark.cpu_size_threshold <= 0:
    warnings.append(f"Invalid cpu_size_threshold: {benchmark.cpu_size_threshold} (must be > 0)")

  if not problems:
    warnings.append("No problems defined in problems.json")

  for problem in problems:
    if problem.size <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid size: {problem.size} (must be > 0)")
    if problem.optimal is not None and problem.optimal <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid optimal: {problem.optimal} (must be > 0)")

  problem_names = [p.name for p in problems]
  if len(problem_names) != len(set(problem_names)):
    duplicates = [name for name in problem_names if problem_names.count(name) > 1]
    warnings.append(f"Duplicate problem names found: {set(duplicates)}")

  from algorithms.strategies import STRATEGY_REGISTRY

  for name, algo in algorithms.items():
    try:
      algo.get_class()
    except (ModuleNotFoundError, AttributeError) as e:
      warnings.append(f"Algorithm '{name}' class not found: {e}")
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
              )
            )

  return warnings


def build_problem_configs_from_tsp_files(
  tsp_files: Sequence[str | Path],
  existing_configs: Sequence[ProblemConfig] | None = None,
  tsp_dir: str | Path | None = None,
  known_optimals: Mapping[str, float] | None = None,
) -> tuple[list[ProblemConfig], dict[str, Path]]:
  """
  Build ProblemConfig instances directly from .tsp file paths using `routing_data`.

  Args:
      tsp_files: Sequence of paths to .tsp files (or problem names when `tsp_dir` is supplied).
      existing_configs: Optional list of already loaded ProblemConfig instances.
      tsp_dir: Optional base directory to resolve bare problem names against.
      known_optimals: Optional name -> optimal tour cost map, normally derived from
          a `problems.json` catalogue (`{cfg.name: cfg.optimal}`). This function performs no
          file I/O beyond the `.tsp` parse, so the reference costs stay a caller concern
          (`DP11-4`).

  Returns:
      Tuple of ProblemConfig list and path map dictionary.
  """
  existing_map: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}
  optimal_map: Mapping[str, float] = known_optimals or {}
  base_dir: Path | None = Path(tsp_dir) if tsp_dir is not None else None

  resolved_configs: list[ProblemConfig] = []
  tsp_path_map: dict[str, Path] = {}

  for item in tsp_files:
    candidate = Path(item)
    if not candidate.exists():
      if base_dir is not None:
        stem: str = candidate.stem if candidate.suffix == ".tsp" else str(item)
        alt: Path = base_dir / f"{stem}.tsp"
        if alt.exists():
          candidate = alt
        else:
          raise FileNotFoundError(f"TSPLIB file not found: {item}")
      else:
        raise FileNotFoundError(f"TSPLIB file not found: {item}")

    bundle: TsplibIngestionBundle = parse_tsplib_file(candidate)
    name = str(bundle.problem.name)
    dimension = int(bundle.problem.dimension)
    tsp_path_map[name] = candidate.resolve()

    optimal: float | None
    if name in existing_map:
      optimal = existing_map[name].optimal
    elif name in optimal_map:
      optimal = float(optimal_map[name])
    elif candidate.stem in optimal_map:
      optimal = float(optimal_map[candidate.stem])
    else:
      optimal = resolve_opt_tour_cost(candidate, bundle=bundle)

    resolved_configs.append(ProblemConfig(name=name, optimal=optimal, size=dimension))

  return resolved_configs, tsp_path_map


def build_problem_configs_from_duckdb(
  db_path: str | Path,
  problem_names: Sequence[str] | None = None,
  existing_configs: Sequence[ProblemConfig] | None = None,
) -> list[ProblemConfig]:
  """
  Build ProblemConfig instances by querying DuckDB for dimension and optimal solution cost.

  Queries the normalized DuckDB schema:
  - `problems` joined with `tsp_problems` for dimension and name
  - `solutions` for the optimal cost, falling back to `existing_configs` (loaded from
    a `problems.json` catalogue) when the database carries no optimal solution (`DP11-4`).

  Args:
      db_path: Path to DuckDB database file.
      problem_names: Optional sequence of problem names to filter by. If None,
          all TSP problems in the database are returned.
      existing_configs: Optional list of already loaded ProblemConfig instances.

  Returns:
      List of ProblemConfig instances. Preserves `problem_names` order if supplied.
  """
  import duckdb

  db_file = Path(db_path)
  if not db_file.exists():
    raise FileNotFoundError(f"Database file not found: {db_file}")

  existing_map: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}
  conn = duckdb.connect(str(db_file), read_only=True)
  try:
    if problem_names is not None:
      names_list = list(problem_names)
      placeholders = ", ".join("?" for _ in names_list)
      query = f"""
        SELECT p.id, p.name, t.dimension,
               (SELECT s.cost FROM solutions s
                WHERE s.problem_id = p.id AND (s.kind = 'optimal' OR s.quality = 'optimal')
                LIMIT 1) as solution_cost
        FROM problems p
        JOIN tsp_problems t ON p.id = t.problem_id
        WHERE p.name IN ({placeholders})
        ORDER BY t.dimension ASC, p.name ASC
      """  # noqa: S608
      rows = conn.execute(query, names_list).fetchall()
    else:
      query = """
        SELECT p.id, p.name, t.dimension,
               (SELECT s.cost FROM solutions s
                WHERE s.problem_id = p.id AND (s.kind = 'optimal' OR s.quality = 'optimal')
                LIMIT 1) as solution_cost
        FROM problems p
        JOIN tsp_problems t ON p.id = t.problem_id
        ORDER BY t.dimension ASC, p.name ASC
      """
      rows = conn.execute(query).fetchall()

    resolved_configs: list[ProblemConfig] = []
    found_names: set[str] = set()

    for _prob_id, name, dimension, solution_cost in rows:
      found_names.add(name)
      optimal: float | None
      if solution_cost is not None:
        optimal = float(solution_cost)
      elif name in existing_map:
        optimal = existing_map[name].optimal
      else:
        optimal = None
      resolved_configs.append(ProblemConfig(name=name, optimal=optimal, size=dimension))

    if problem_names is not None:
      missing: list[str] = [n for n in problem_names if n not in found_names]
      if missing:
        raise KeyError(f"Problem(s) not found in DuckDB database: {missing}")
      name_to_config = {cfg.name: cfg for cfg in resolved_configs}
      return [name_to_config[n] for n in problem_names]

    return resolved_configs
  finally:
    conn.close()
