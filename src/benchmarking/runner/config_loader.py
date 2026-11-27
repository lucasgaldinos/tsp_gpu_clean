"""JSON configuration loaders and public facade for `benchmarking.runner.config_loader`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Mapping

from data_models.problem import VALID_PROBLEM_TYPES

from .config_models import AlgorithmConfig, BenchmarkConfig, GAParams, ProblemConfig
from .config_validation import validate_configs
from .problem_catalog import (
  _resolve_config_problem_type,
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
)
from .spec_resolver import (
  MODULAR_ALGO_PARAMS,
  VALID_STAGE_ANCHORS,
  VALID_STRATEGY_SLOTS,
  _infer_nested_strategy_slot,
  _resolve_class_spec,
  _resolve_init_type_hints,
  _slot_protocol_map,
  _validate_class_spec,
  is_file_module_ref,
  load_external_module,
  unwrap_type_annotation,
)


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
  """Load an algorithms configuration file containing default parameters and variants."""
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

  source_dir = resolved_path.parent.resolve()
  algorithms: dict[str, AlgorithmConfig] = {}
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
      source_dir=source_dir,
    )

  return ga_params, algorithms


def _parse_optional_int_field(data: Mapping[str, Any], field_name: str, file_label: str) -> int | None:
  """Parse an optional integer-or-null field, rejecting booleans and non-integer values."""
  raw_val = data.get(field_name)
  if raw_val is None:
    return None
  if isinstance(raw_val, bool) or not isinstance(raw_val, int):
    raise ValueError(
      f"{file_label} '{field_name}' must be an integer or null, got {type(raw_val).__name__}: {raw_val!r}"
    )
  return int(raw_val)


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
  """Load benchmark execution parameters from a JSON file or configuration directory."""
  resolved_path: Path = _resolve_config_file(config_path, "benchmark.json")
  if not resolved_path.exists():
    raise FileNotFoundError(f"Benchmark config not found: {resolved_path}")

  with open(resolved_path) as f:
    data = json.load(f)

  if not isinstance(data, dict):
    raise ValueError("benchmark.json must contain a JSON object")
  if "repetitions" not in data:
    raise ValueError("benchmark.json missing required field: repetitions")

  raw_reps = data["repetitions"]
  if isinstance(raw_reps, bool) or not isinstance(raw_reps, int):
    raise ValueError(f"benchmark.json 'repetitions' must be an integer, got {type(raw_reps).__name__}: {raw_reps!r}")

  patience = _parse_optional_int_field(data, "patience", "benchmark.json")
  max_generations = _parse_optional_int_field(data, "max_generations", "benchmark.json")
  cpu_size_threshold = _parse_optional_int_field(data, "cpu_size_threshold", "benchmark.json")

  raw_incremental = data.get("incremental_runs", False)
  if not isinstance(raw_incremental, bool):
    raise ValueError(
      f"benchmark.json 'incremental_runs' must be a boolean, got {type(raw_incremental).__name__}: {raw_incremental!r}"
    )

  raw_display = data.get("display", {})
  if raw_display is None:
    raw_display = {}
  if not isinstance(raw_display, dict):
    raise ValueError(f"benchmark.json 'display' must be an object, got {type(raw_display).__name__}")
  raw_line_width = raw_display.get("line_width", 80)
  if isinstance(raw_line_width, bool) or not isinstance(raw_line_width, int):
    raise ValueError(
      f"benchmark.json 'display.line_width' must be an integer, got {type(raw_line_width).__name__}: {raw_line_width!r}"
    )

  return BenchmarkConfig(
    repetitions=int(raw_reps),
    patience=patience,
    max_generations=max_generations,
    cpu_size_threshold=cpu_size_threshold,
    incremental_runs=raw_incremental,
    line_width=int(raw_line_width),
  )


def load_problems_config(config_path: str | Path) -> list[ProblemConfig]:
  """Load problem definitions from a JSON file or configuration directory."""
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

  problems: list[ProblemConfig] = []
  for idx, problem_dict in enumerate(entries):
    if not isinstance(problem_dict, dict):
      raise ValueError(f"Problem at index {idx} must be a JSON object")
    for field in ("name", "size"):
      if field not in problem_dict:
        raise ValueError(f"Problem at index {idx} missing required field: {field}")

    raw_size = problem_dict["size"]
    if isinstance(raw_size, bool) or not isinstance(raw_size, int):
      raise ValueError(f"Problem at index {idx} 'size' must be an integer, got {type(raw_size).__name__}: {raw_size!r}")

    raw_optimal = problem_dict.get("optimal")
    if raw_optimal is not None and (isinstance(raw_optimal, bool) or not isinstance(raw_optimal, (int, float))):
      raise ValueError(
        f"Problem at index {idx} 'optimal' must be a number or null, got {type(raw_optimal).__name__}: {raw_optimal!r}"
      )

    raw_problem_type = problem_dict.get("problem_type")
    problem_type: str | None = None
    if raw_problem_type is not None:
      if not isinstance(raw_problem_type, str) or raw_problem_type.upper().strip() not in VALID_PROBLEM_TYPES:
        raise ValueError(
          f"Problem at index {idx} 'problem_type' must be one of {sorted(VALID_PROBLEM_TYPES)}, "
          f"got {raw_problem_type!r}"
        )
      problem_type = raw_problem_type.upper().strip()

    problems.append(
      ProblemConfig(
        name=problem_dict["name"],
        optimal=float(raw_optimal) if raw_optimal is not None else None,
        size=int(raw_size),
        problem_type=problem_type,
        return_to_depot=problem_dict.get("return_to_depot"),
        max_vehicles=problem_dict.get("max_vehicles"),
        max_route_distance=problem_dict.get("max_route_distance"),
        capacity=problem_dict.get("capacity"),
      )
    )

  return problems


def load_all_configs(config_dir: str | Path) -> dict[str, Any]:
  """Load all configuration files from the specified directory."""
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
  """Resolve benchmark configurations from optional individual files and/or a config directory."""
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


__all__: list[str] = [
  "MODULAR_ALGO_PARAMS",
  "VALID_STAGE_ANCHORS",
  "VALID_STRATEGY_SLOTS",
  "AlgorithmConfig",
  "BenchmarkConfig",
  "GAParams",
  "ProblemConfig",
  "_infer_nested_strategy_slot",
  "_parse_optional_int_field",
  "_resolve_class_spec",
  "_resolve_config_file",
  "_resolve_config_problem_type",
  "_resolve_init_type_hints",
  "_slot_protocol_map",
  "_validate_class_spec",
  "build_problem_configs_from_duckdb",
  "build_problem_configs_from_tsp_files",
  "is_file_module_ref",
  "load_algorithms_config",
  "load_all_configs",
  "load_benchmark_config",
  "load_external_module",
  "load_problems_config",
  "resolve_benchmark_configs",
  "unwrap_type_annotation",
  "validate_configs",
]
