"""
Configuration Loader for Benchmark V2
======================================

Loads and validates JSON configuration files for the modular benchmark system.

This module provides type-safe configuration loading with dataclasses,
replacing the hardcoded constants from the monolithic chapter4_validation.py.

Usage:
    from src.benchmarking_v2.config_loader import load_all_configs

    configs = load_all_configs("code/benchmarks_v2/configs/")
    ga_params = configs["ga_params"]
    algorithms = configs["algorithms"]
    problems = configs["problems"]
    benchmark_config = configs["benchmark"]

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional


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

    def to_dict(self) -> Dict[str, Any]:
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

    def get_class(self):
        """Dynamically import and return the algorithm class."""
        module = importlib.import_module(self.class_module)
        return getattr(module, self.class_name)


@dataclass
class BenchmarkConfig:
    """Benchmark execution parameters."""

    repetitions: int
    patience: int
    cpu_size_threshold: int
    skip_cpu_default: bool
    incremental_runs: bool = False  # NEW: Enable incremental checkpoint appending
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
) -> tuple[GAParams, Dict[str, AlgorithmConfig]]:
    """
    Load algorithms.json containing GA parameters and algorithm configurations.

    Args:
        config_dir: Path to configuration directory

    Returns:
        Tuple of (GAParams, dict of algorithm_name -> AlgorithmConfig)

    Raises:
        FileNotFoundError: If algorithms.json doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = config_dir / "algorithms.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Algorithms config not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    # Validate required sections
    if "ga_params" not in data:
        raise ValueError("algorithms.json missing 'ga_params' section")
    if "algorithm_configs" not in data:
        raise ValueError("algorithms.json missing 'algorithm_configs' section")

    # Parse GA parameters
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

    # Parse algorithm configurations
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

    Raises:
        FileNotFoundError: If benchmark.json doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = config_dir / "benchmark.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = [
        "repetitions",
        "patience",
        "cpu_size_threshold",
        "skip_cpu_default",
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"benchmark.json missing required field: {field}")

    # Optional display settings
    line_width = data.get("display", {}).get("line_width", 80)

    return BenchmarkConfig(
        repetitions=data["repetitions"],
        patience=data["patience"],
        cpu_size_threshold=data["cpu_size_threshold"],
        skip_cpu_default=data["skip_cpu_default"],
        line_width=line_width,
    )


def load_problems_config(config_dir: Path) -> List[ProblemConfig]:
    """
    Load problems.json containing TSP problem definitions.

    Args:
        config_dir: Path to configuration directory

    Returns:
        List of ProblemConfig instances

    Raises:
        FileNotFoundError: If problems.json doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = config_dir / "problems.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Problems config not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("problems.json must contain a JSON array")

    problems = []
    required_fields = ["name", "optimal", "size"]

    for idx, problem_dict in enumerate(data):
        for field in required_fields:
            if field not in problem_dict:
                raise ValueError(
                    f"Problem at index {idx} missing required field: {field}"
                )

        problems.append(
            ProblemConfig(
                name=problem_dict["name"],
                optimal=float(problem_dict["optimal"]),
                size=int(problem_dict["size"]),
            )
        )

    return problems


def load_all_configs(config_dir: str | Path) -> Dict[str, Any]:
    """
    Load all configuration files from the specified directory.

    Args:
        config_dir: Path to configuration directory (string or Path object)

    Returns:
        Dictionary with keys:
            - "ga_params": GAParams instance
            - "algorithms": Dict[str, AlgorithmConfig]
            - "benchmark": BenchmarkConfig instance
            - "problems": List[ProblemConfig]

    Raises:
        FileNotFoundError: If config directory or files don't exist
        ValueError: If any configuration is invalid

    Example:
        >>> configs = load_all_configs("code/benchmarks_v2/configs/")
        >>> print(f"Running {configs['benchmark'].repetitions} repetitions")
        >>> print(f"Testing {len(configs['problems'])} problems")
        >>> for name, algo_config in configs['algorithms'].items():
        ...     print(f"{name}: {algo_config.description}")
    """
    config_path = Path(config_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_path}")

    if not config_path.is_dir():
        raise ValueError(f"Config path must be a directory: {config_path}")

    # Load all configs
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


def validate_configs(configs: Dict[str, Any]) -> List[str]:
    """
    Validate loaded configurations for consistency and completeness.

    Args:
        configs: Dictionary returned by load_all_configs()

    Returns:
        List of warning messages (empty list if no issues)

    Example:
        >>> configs = load_all_configs("configs/")
        >>> warnings = validate_configs(configs)
        >>> if warnings:
        ...     for warning in warnings:
        ...         print(f"WARNING: {warning}")
    """
    warnings = []

    ga_params = configs["ga_params"]
    algorithms = configs["algorithms"]
    benchmark = configs["benchmark"]
    problems = configs["problems"]

    # Check GA parameters
    if ga_params.population_size <= 0:
        warnings.append(
            f"Invalid population_size: {ga_params.population_size} (must be > 0)"
        )

    if not (0.0 <= ga_params.mutation_rate <= 1.0):
        warnings.append(
            f"Invalid mutation_rate: {ga_params.mutation_rate} (must be in [0, 1])"
        )

    if ga_params.tournament_size <= 0:
        warnings.append(
            f"Invalid tournament_size: {ga_params.tournament_size} (must be > 0)"
        )

    if ga_params.two_opt_iterations <= 0:
        warnings.append(
            f"Invalid two_opt_iterations: {ga_params.two_opt_iterations} (must be > 0)"
        )

    # Check benchmark config
    if benchmark.repetitions <= 0:
        warnings.append(f"Invalid repetitions: {benchmark.repetitions} (must be > 0)")

    if benchmark.patience <= 0:
        warnings.append(f"Invalid patience: {benchmark.patience} (must be > 0)")

    if benchmark.cpu_size_threshold <= 0:
        warnings.append(
            f"Invalid cpu_size_threshold: {benchmark.cpu_size_threshold} (must be > 0)"
        )

    # Check problems
    if not problems:
        warnings.append("No problems defined in problems.json")

    for problem in problems:
        if problem.size <= 0:
            warnings.append(
                f"Problem '{problem.name}' has invalid size: {problem.size} (must be > 0)"
            )
        if problem.optimal <= 0:
            warnings.append(
                f"Problem '{problem.name}' has invalid optimal: {problem.optimal} (must be > 0)"
            )

    # Check for duplicate problem names
    problem_names = [p.name for p in problems]
    if len(problem_names) != len(set(problem_names)):
        duplicates = [name for name in problem_names if problem_names.count(name) > 1]
        warnings.append(f"Duplicate problem names found: {set(duplicates)}")

    # Check algorithm class imports (verify modules exist)
    for name, algo in algorithms.items():
        try:
            algo.get_class()
        except (ModuleNotFoundError, AttributeError) as e:
            warnings.append(f"Algorithm '{name}' class not found: {e}")

    return warnings
