"""
Configuration Loader for Benchmark V2
======================================

Loads and validates JSON configuration files for the modular benchmark system.

This module provides type-safe configuration loading with dataclasses,
replacing the hardcoded constants from the monolithic chapter4_validation.py.

Usage:
    from benchmarking.runner.config_loader import default_config_dir, load_all_configs

    configs = load_all_configs(default_config_dir())
    ga_params = configs["ga_params"]
    algorithms = configs["algorithms"]
    problems = configs["problems"]
    benchmark_config = configs["benchmark"]

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loaders.tsplib_loader import parse_tsp_file_with_tsplib95

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SELECTED_TSP_FILES: tuple[Path, ...] = (
  PROJECT_ROOT / "datasets" / "tsp" / "att48.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "bayg29.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "bays29.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "berlin52.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "brg180.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "gr17.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "gr21.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "gr24.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "gr48.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "gr96.tsp",
)


def default_config_dir() -> Path:
  """Directory holding `benchmark.json`, `algorithms.json` and `problems.json`.

  `configs/` sits beside `src/` at the repository root: these files are run
  parameters you edit, not code, so they live outside the package tree.

  Returns:
      Path to the configuration directory (`<repo root>/configs`).
  """
  return PROJECT_ROOT / "configs"


# Canonical TSPLIB95 optimal tour costs (from Reinelt 1991 / TSPLIB95 FAQ)
KNOWN_TSPLIB_OPTIMALS: dict[str, float] = {
  "burma14": 3323.0,
  "ulysses16": 6859.0,
  "gr17": 2085.0,
  "gr21": 2707.0,
  "ulysses22": 7013.0,
  "gr24": 1272.0,
  "fri26": 937.0,
  "bayg29": 1610.0,
  "bays29": 2020.0,
  "dantzig42": 699.0,
  "swiss42": 1273.0,
  "att48": 10628.0,
  "gr48": 5046.0,
  "hk48": 11461.0,
  "eil51": 426.0,
  "berlin52": 7542.0,
  "brazil58": 25395.0,
  "st70": 675.0,
  "eil76": 538.0,
  "pr76": 108159.0,
  "gr96": 55209.0,
  "rat99": 1211.0,
  "kroA100": 21282.0,
  "kroB100": 22141.0,
  "kroC100": 20749.0,
  "kroD100": 21294.0,
  "kroE100": 22068.0,
  "rd100": 7910.0,
  "eil101": 629.0,
  "lin105": 14379.0,
  "pr107": 44303.0,
  "gr120": 6942.0,
  "pr124": 59030.0,
  "bier127": 118282.0,
  "ch130": 6110.0,
  "pr136": 96772.0,
  "gr137": 69853.0,
  "pr144": 58537.0,
  "ch150": 6528.0,
  "kroA150": 26524.0,
  "kroB150": 26130.0,
  "pr152": 73682.0,
  "u159": 42080.0,
  "si175": 21407.0,
  "brg180": 1950.0,
  "rat195": 2323.0,
  "d198": 15780.0,
  "kroA200": 29368.0,
  "kroB200": 29437.0,
  "ts225": 126643.0,
  "tsp225": 3916.0,
  "pr226": 80369.0,
  "gil262": 2378.0,
  "pr264": 49135.0,
  "a280": 2579.0,
  "pr299": 48191.0,
  "lin318": 42029.0,
  "linhp318": 41345.0,
  "rd400": 15281.0,
  "fl417": 11861.0,
  "gr431": 171414.0,
  "pr439": 107217.0,
  "pcb442": 50778.0,
  "d493": 35002.0,
  "att532": 27686.0,
  "ali535": 202339.0,
  "si535": 48450.0,
  "pa561": 2763.0,
  "u574": 36905.0,
  "rat575": 6773.0,
  "p654": 34643.0,
  "d657": 48912.0,
  "gr666": 294358.0,
  "u724": 41910.0,
  "rat783": 8806.0,
  "dsj1000": 18659688.0,
  "pr1002": 259045.0,
  "si1032": 92650.0,
  "u1060": 224094.0,
  "vm1084": 239297.0,
  "pcb1173": 56892.0,
  "d1291": 50801.0,
  "rl1304": 252948.0,
  "rl1323": 270199.0,
  "nrw1379": 56638.0,
  "fl1400": 20127.0,
  "u1432": 152970.0,
  "fl1577": 22249.0,
  "d1655": 62128.0,
  "vm1748": 336556.0,
  "u1817": 57201.0,
  "rl1889": 316536.0,
  "d2103": 80450.0,
  "u2152": 64253.0,
  "u2319": 234256.0,
  "pr2392": 378032.0,
}

# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class GAParams:
  """Genetic Algorithm parameters (ISO-algorithmic across all variants)."""

  population_size: int
  mutation_rate: float
  tournament_size: int
  two_opt_iterations: int
  seed: int

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for passing to algorithm constructors."""
    return {
      "population_size": self.population_size,
      "mutation_rate": self.mutation_rate,
      "tournament_size": self.tournament_size,
      "two_opt_iterations": self.two_opt_iterations,
      "seed": self.seed,
    }


@dataclass
class AlgorithmConfig:
  """Configuration for a single algorithm variant."""

  name: str
  class_module: str
  class_name: str
  use_gpu: bool
  description: str

  def get_class(self) -> type:
    """Dynamically import and return the algorithm class."""
    module = importlib.import_module(self.class_module)
    return cast("type", getattr(module, self.class_name))

  def instantiate(
    self,
    ga_params: GAParams,
    local_search_stage: str | None = None,
  ) -> Any:
    """Instantiate the algorithm, applying optional local_search_stage override."""
    algo_class = self.get_class()
    kwargs: dict[str, Any] = ga_params.to_dict()
    if local_search_stage is not None:
      # Pass local_search_stage if accepted by constructor
      try:
        import inspect

        sig = inspect.signature(algo_class)
        if "local_search_stage" in sig.parameters or any(
          p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
          kwargs["local_search_stage"] = local_search_stage
      except Exception:
        kwargs["local_search_stage"] = local_search_stage
    return algo_class(**kwargs)


@dataclass
class BenchmarkConfig:
  """Benchmark execution parameters."""

  repetitions: int
  patience: int | None
  cpu_size_threshold: int
  skip_cpu_default: bool
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
  optimal: float
  size: int


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def load_algorithms_config(
  config_dir: Path,
) -> tuple[GAParams, dict[str, AlgorithmConfig]]:
  """
  Load algorithms.json containing GA parameters and algorithm configurations.

  Args:
      config_dir: Path to configuration directory

  Returns:
      Tuple of (GAParams, dict of algorithm_name -> AlgorithmConfig)
  """
  config_path = config_dir / "algorithms.json"

  if not config_path.exists():
    raise FileNotFoundError(f"Algorithms config not found: {config_path}")

  with open(config_path) as f:
    data = json.load(f)

  if "ga_params" not in data:
    raise ValueError("algorithms.json missing 'ga_params' section")
  if "algorithm_configs" not in data:
    raise ValueError("algorithms.json missing 'algorithm_configs' section")

  ga_params_dict = data["ga_params"]
  required_ga_fields = [
    "population_size",
    "mutation_rate",
    "tournament_size",
    "two_opt_iterations",
    "seed",
  ]
  for field in required_ga_fields:
    if field not in ga_params_dict:
      raise ValueError(f"ga_params missing required field: {field}")

  ga_params = GAParams(
    population_size=ga_params_dict["population_size"],
    mutation_rate=ga_params_dict["mutation_rate"],
    tournament_size=ga_params_dict["tournament_size"],
    two_opt_iterations=ga_params_dict["two_opt_iterations"],
    seed=ga_params_dict["seed"],
  )

  algorithms = {}
  for name, config_dict in data["algorithm_configs"].items():
    required_algo_fields = ["class_module", "class_name", "use_gpu", "description"]
    for field in required_algo_fields:
      if field not in config_dict:
        raise ValueError(f"Algorithm '{name}' missing required field: {field}")

    algorithms[name] = AlgorithmConfig(
      name=name,
      class_module=config_dict["class_module"],
      class_name=config_dict["class_name"],
      use_gpu=config_dict["use_gpu"],
      description=config_dict["description"],
    )

  return ga_params, algorithms


def load_benchmark_config(config_dir: Path) -> BenchmarkConfig:
  """
  Load benchmark.json containing execution parameters.

  Args:
      config_dir: Path to configuration directory

  Returns:
      BenchmarkConfig instance
  """
  config_path = config_dir / "benchmark.json"

  if not config_path.exists():
    raise FileNotFoundError(f"Benchmark config not found: {config_path}")

  with open(config_path) as f:
    data = json.load(f)

  required_fields = [
    "repetitions",
    "patience",
    "cpu_size_threshold",
    "skip_cpu_default",
  ]
  for field in required_fields:
    if field not in data:
      raise ValueError(f"benchmark.json missing required field: {field}")

  line_width = data.get("display", {}).get("line_width", 80)

  return BenchmarkConfig(
    repetitions=data["repetitions"],
    patience=data["patience"],
    cpu_size_threshold=data["cpu_size_threshold"],
    skip_cpu_default=data["skip_cpu_default"],
    incremental_runs=bool(data.get("incremental_runs", False)),
    line_width=line_width,
  )


def load_problems_config(config_dir: Path) -> list[ProblemConfig]:
  """
  Load problems.json containing TSP problem definitions.

  Args:
      config_dir: Path to configuration directory

  Returns:
      List of ProblemConfig instances
  """
  config_path = config_dir / "problems.json"

  if not config_path.exists():
    raise FileNotFoundError(f"Problems config not found: {config_path}")

  with open(config_path) as f:
    data = json.load(f)

  if not isinstance(data, list):
    raise ValueError("problems.json must contain a JSON array")

  problems = []
  required_fields = ["name", "optimal", "size"]

  for idx, problem_dict in enumerate(data):
    for field in required_fields:
      if field not in problem_dict:
        raise ValueError(f"Problem at index {idx} missing required field: {field}")

    problems.append(
      ProblemConfig(
        name=problem_dict["name"],
        optimal=float(problem_dict["optimal"]),
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
  benchmark = load_benchmark_config(config_path)
  problems = load_problems_config(config_path)

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

  if ga_params.tournament_size <= 0:
    warnings.append(f"Invalid tournament_size: {ga_params.tournament_size} (must be > 0)")

  if ga_params.two_opt_iterations <= 0:
    warnings.append(f"Invalid two_opt_iterations: {ga_params.two_opt_iterations} (must be > 0)")

  if benchmark.repetitions <= 0:
    warnings.append(f"Invalid repetitions: {benchmark.repetitions} (must be > 0)")

  if benchmark.patience is not None and benchmark.patience <= 0:
    warnings.append(f"Invalid patience: {benchmark.patience} (must be > 0 or None)")

  if benchmark.cpu_size_threshold <= 0:
    warnings.append(f"Invalid cpu_size_threshold: {benchmark.cpu_size_threshold} (must be > 0)")

  if not problems:
    warnings.append("No problems defined in problems.json")

  for problem in problems:
    if problem.size <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid size: {problem.size} (must be > 0)")
    if problem.optimal <= 0:
      warnings.append(f"Problem '{problem.name}' has invalid optimal: {problem.optimal} (must be > 0)")

  problem_names = [p.name for p in problems]
  if len(problem_names) != len(set(problem_names)):
    duplicates = [name for name in problem_names if problem_names.count(name) > 1]
    warnings.append(f"Duplicate problem names found: {set(duplicates)}")

  for name, algo in algorithms.items():
    try:
      algo.get_class()
    except (ModuleNotFoundError, AttributeError) as e:
      warnings.append(f"Algorithm '{name}' class not found: {e}")

  return warnings


def build_problem_configs_from_tsp_files(
  tsp_files: Sequence[str | Path],
  existing_configs: Sequence[ProblemConfig] | None = None,
) -> tuple[list[ProblemConfig], dict[str, Path]]:
  """
  Build ProblemConfig instances directly from .tsp file paths using tsplib95.

  Args:
      tsp_files: Sequence of paths to .tsp files (or problem names in datasets/tsp).
      existing_configs: Optional list of already loaded ProblemConfig instances.

  Returns:
      Tuple of ProblemConfig list and path map dictionary.
  """
  existing_map: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}
  default_tsp_dir: Path = PROJECT_ROOT / "datasets" / "tsp"

  resolved_configs: list[ProblemConfig] = []
  tsp_path_map: dict[str, Path] = {}

  for item in tsp_files:
    candidate = Path(item)
    if not candidate.exists():
      stem: str = candidate.stem if candidate.suffix == ".tsp" else str(item)
      alt = default_tsp_dir / f"{stem}.tsp"
      if alt.exists():
        candidate = alt
      else:
        raise FileNotFoundError(f"TSPLIB file not found: {item}")

    bundle = parse_tsp_file_with_tsplib95(candidate)
    name = str(bundle.problem.name)
    dimension = int(bundle.problem.dimension)
    tsp_path_map[name] = candidate.resolve()

    if name in existing_map:
      optimal: float = existing_map[name].optimal
    elif name in KNOWN_TSPLIB_OPTIMALS:
      optimal = KNOWN_TSPLIB_OPTIMALS[name]
    elif candidate.stem in KNOWN_TSPLIB_OPTIMALS:
      optimal = KNOWN_TSPLIB_OPTIMALS[candidate.stem]
    else:
      raise KeyError(
        f"Known optimal tour cost not found for '{name}' ({candidate}). "
        f"Add it to KNOWN_TSPLIB_OPTIMALS or problems.json."
      )

    resolved_configs.append(ProblemConfig(name=name, optimal=float(optimal), size=dimension))

  return resolved_configs, tsp_path_map
