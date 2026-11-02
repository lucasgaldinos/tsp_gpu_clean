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

from loaders.tsplib_loader import TsplibIngestionBundle, parse_tsp_file_with_tsplib95

if TYPE_CHECKING:
  from inspect import Signature

  from protocols.types import LocalSearchStage

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
    valid_slots = {"initialization", "selection", "crossover", "mutation", "evaluation", "local_search"}
    if strategy_overrides:
      for slot in strategy_overrides:
        if slot not in valid_slots:
          raise ValueError(
            f"Unknown strategy slot '{slot}'. Valid slots: {sorted(valid_slots)}"
          )

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

    valid_slots = {"initialization", "selection", "crossover", "mutation", "evaluation", "local_search"}
    if strategy_overrides:
      for slot_name in strategy_overrides:
        if slot_name not in valid_slots:
          raise ValueError(
            f"Unknown strategy slot '{slot_name}'. Valid slots are: {sorted(valid_slots)}"
          )

    def _build_strategy(slot: str) -> Any:
      strat_spec = self.strategies.get(slot)
      if not strat_spec:
        return None
      class_name = strat_spec.get("class")
      if not class_name:
        return None
      if class_name not in STRATEGY_REGISTRY:
        raise ValueError(
          f"Unknown strategy class '{class_name}' in slot '{slot}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
      strat_cls: type = STRATEGY_REGISTRY[class_name]
      strat_params = dict(strat_spec.get("params", {}))

      import inspect

      sig: Signature = inspect.signature(strat_cls)
      has_var_kw: bool = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

      if strategy_overrides and slot in strategy_overrides:
        for k, v in strategy_overrides[slot].items():
          if not has_var_kw and k not in sig.parameters:
            raise ValueError(
              f"Strategy class '{class_name}' in slot '{slot}' does not accept parameter '{k}'. "
              f"Accepted parameters: {list(sig.parameters.keys())}"
            )
          strat_params[k] = v

      if "seed" in sig.parameters and "seed" not in strat_params:
        strat_params["seed"] = resolved_seed
      return strat_cls(**strat_params)

    init_strat = _build_strategy("initialization")
    sel_strat = _build_strategy("selection")
    cx_strat = _build_strategy("crossover")
    mut_strat = _build_strategy("mutation")
    eval_strat = _build_strategy("evaluation")
    ls_strat = _build_strategy("local_search")

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
    )


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
  *,
  config_name: str = "algorithms.json",
) -> tuple[GAParams, dict[str, AlgorithmConfig]]:
  """
  Load an algorithms configuration file containing default parameters and variants.

  Args:
      config_dir: Path to configuration directory
      config_name: Configuration file name. Pass `"algorithms.archive.json"` to load the
          archived monolithic baseline, which the active configuration no longer lists.

  Returns:
      Tuple of (GAParams, dict of algorithm_name -> AlgorithmConfig)
  """
  config_path: Path = config_dir / config_name

  if not config_path.exists():
    raise FileNotFoundError(f"Algorithms config not found: {config_path}")

  with open(config_path) as f:
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


def load_benchmark_config(config_dir: Path) -> BenchmarkConfig:
  """
  Load benchmark.json containing execution parameters.

  Args:
      config_dir: Path to configuration directory

  Returns:
      BenchmarkConfig instance
  """
  config_path: Path = config_dir / "benchmark.json"

  if not config_path.exists():
    raise FileNotFoundError(f"Benchmark config not found: {config_path}")

  with open(config_path) as f:
    data = json.load(f)

  required_fields: list[str] = [
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
  config_path: Path = config_dir / "problems.json"

  if not config_path.exists():
    raise FileNotFoundError(f"Problems config not found: {config_path}")

  with open(config_path) as f:
    data = json.load(f)

  if not isinstance(data, list):
    raise ValueError("problems.json must contain a JSON array")

  problems = []
  required_fields: list[str] = ["name", "optimal", "size"]

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
  benchmark: BenchmarkConfig = load_benchmark_config(config_path)
  problems: list[ProblemConfig] = load_problems_config(config_path)

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

  for algo_name, algo_cfg in algorithms.items():
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
  tsp_dir: str | Path | None = None,
) -> tuple[list[ProblemConfig], dict[str, Path]]:
  """
  Build ProblemConfig instances directly from .tsp file paths using tsplib95.

  Args:
      tsp_files: Sequence of paths to .tsp files (or problem names when `tsp_dir` is supplied).
      existing_configs: Optional list of already loaded ProblemConfig instances.
      tsp_dir: Optional base directory to resolve bare problem names against.

  Returns:
      Tuple of ProblemConfig list and path map dictionary.
  """
  existing_map: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}
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

    bundle: TsplibIngestionBundle = parse_tsp_file_with_tsplib95(candidate)
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


def build_problem_configs_from_duckdb(
  db_path: str | Path,
  problem_names: Sequence[str] | None = None,
  existing_configs: Sequence[ProblemConfig] | None = None,
) -> list[ProblemConfig]:
  """
  Build ProblemConfig instances by querying DuckDB for dimension and optimal solution cost.

  Queries the normalized DuckDB schema:
  - `problems` joined with `tsp_problems` for dimension and name
  - `solutions` for optimal cost, falling back to `existing_configs` or `KNOWN_TSPLIB_OPTIMALS`.

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
      if solution_cost is not None:
        optimal = float(solution_cost)
      elif name in existing_map:
        optimal = existing_map[name].optimal
      elif name in KNOWN_TSPLIB_OPTIMALS:
        optimal = KNOWN_TSPLIB_OPTIMALS[name]
      else:
        raise KeyError(
          f"Optimal tour cost not found for '{name}' in solutions table, existing_configs, or KNOWN_TSPLIB_OPTIMALS."
        )
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
