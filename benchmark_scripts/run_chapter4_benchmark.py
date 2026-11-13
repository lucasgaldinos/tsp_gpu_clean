#!/usr/bin/env python3
"""
Chapter 4 Validation Benchmark Runner
=====================================

Executes the iso-algorithmic four-variant comparison of Chapter 4 through the
current modular pipeline (`benchmarking.runner.orchestration`).

The comparison is the same experiment as the pre-modularisation cohort, mapped
onto the renamed variants:

    slot          config variant      execution strategy
    ------------  ------------------  -------------------------------------------------
    cpu           CPU                 NumPy everywhere; `CPUTwoOpt` local search
    hybridnaive   HybridNaive         per-route GPU 2-opt; CPU evaluation
    hybrid        HybridOptimized     batched GPU 2-opt; GPU evaluation
    fullgpu       FullGPUModular      every operator on GPU (per-phase kernels)

The four variants are iso-algorithmic: identical GA parameters (population,
mutation rate, seed), identical tournament size and 2-opt iteration budget, and
identical operators except for the execution backend.

`validate_comparison_contract` fails loudly when `configs/algorithms.json` drifts
from that contract, and `validate_existing_checkpoints` refuses to resume from
checkpoints produced by a different architecture (the v2/v3 cohorts predate
`local_search_stage`, `raw_best_tours`, `problem_type` and `edge_type`, and
record only three strategy slots).

Usage:
    python -m benchmark_scripts.run_chapter4_benchmark
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.runner.algorithm_runner import adaptive_generations
from benchmarking.runner.checkpoint_io import load_checkpoint
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  BenchmarkConfig,
  GAParams,
  ProblemConfig,
  load_all_configs,
  validate_configs,
)
from benchmarking.runner.orchestration import run_comprehensive_benchmark
from protocols.backend import cupy_backend, gpu_available

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from protocols.types import LocalSearchStage

# =============================================================================
# Experiment declaration (explicit inputs, no hidden parsing)
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DB_PATH: Path = PROJECT_ROOT / "datasets" / "tsp.duckdb"
FALLBACK_TSP_DIR: Path = PROJECT_ROOT / "datasets" / "tsp"
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v4"
BENCHMARK_NAME: str = "chapter4_validation"
BASELINE_ALGORITHM: str = "CPU"

#: Thesis table slots -> active config variant names.
COMPARISON_SLOTS: dict[str, str] = {
  "cpu": "CPU",
  "hybridnaive": "HybridNaive",
  "hybrid": "HybridOptimized",
  "fullgpu": "FullGPUModular",
}
ALGORITHMS: tuple[str, ...] = tuple(COMPARISON_SLOTS.values())

#: Chapter 4 evaluation cohort spanning small, medium, and large instances across diverse edge-weight formats.
PROBLEMS: tuple[str, ...] = (
  "berlin52",
  "eil51",
  "att48",
  "gr96",
  "kroA100",
  "ch150",
  "d198",
  "pr264",
  "lin318",
  "pcb442",
  "att532",
  "si535",
  "gr666",
  "rat783",
  "pr1002",
)

CANDIDATE_K: int = 20
LOCAL_SEARCH_MAX_ITERATIONS: int = 10
SELECTION_K: int = 5
SEED: int = 42
POPULATION_SIZE: int = 256
MUTATION_RATE: float = 0.02
CPU_SIZE_THRESHOLD: int = 400
REPETITIONS: int = 30
INCREMENTAL_RUNS: bool = True
LOCAL_SEARCH_STAGE: LocalSearchStage = "post_mutation"

# =============================================================================
# Comparison contract
# =============================================================================

STRATEGY_SLOTS: tuple[str, ...] = (
  "initialization",
  "selection",
  "crossover",
  "mutation",
  "local_search",
  "evaluation",
)

#: Iso-algorithmic parameter values documented in the thesis.
EXPECTED_GA_PARAMS: dict[str, float] = {"population_size": 256.0, "mutation_rate": 0.02, "seed": 42.0}

#: Operator composition that defines each variant's execution backend (`DP11-1` naming).
EXPECTED_STRATEGIES: dict[str, dict[str, str]] = {
  "CPU": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchCPUTwoOpt",
    "evaluation": "EvaluationCPUCost",
  },
  "HybridNaive": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationCPUCost",
  },
  "HybridOptimized": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationGPUCost",
  },
  "FullGPUModular": {
    "initialization": "InitializationRandom",
    "selection": "SelectionGPUTournament",
    "crossover": "CrossoverGPUOrder",
    "mutation": "MutationGPUSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationGPUCost",
  },
}

#: Strategy parameters that must be identical across the four variants.
EXPECTED_STRATEGY_PARAMS: tuple[tuple[str, str, float], ...] = (
  ("selection", "k", 5.0),
  ("local_search", "max_iterations", 10.0),
  ("local_search", "candidate_k", 20.0),
)

#: `LocalSearchGPUTwoOpt.mode` selects the transfer strategy; `CPU` runs the vectorised best-improvement loop.
EXPECTED_LOCAL_SEARCH_MODE: dict[str, str] = {
  "CPU": "best",
  "HybridNaive": "naive",
  "HybridOptimized": "batch",
  "FullGPUModular": "batch",
}

#: `return_gpu` for (initialization, evaluation); only the full-GPU variant keeps routes in VRAM.
EXPECTED_RETURN_GPU: dict[str, tuple[bool, bool]] = {
  "CPU": (False, False),
  "HybridNaive": (False, False),
  "HybridOptimized": (False, False),
  "FullGPUModular": (True, True),
}

#: Keys written by the current runner; older cohorts lack them and must not be resumed.
REQUIRED_CHECKPOINT_KEYS: tuple[str, ...] = (
  "strategies",
  "algorithm_config",
  "problem_type",
  "edge_type",
  "raw_best_tours",
  "raw_seeds",
)

#: Cohorts written before Design B encode the old class names and lack `strategy_params`.
REQUIRED_ALGORITHM_CONFIG_KEYS: tuple[str, ...] = ("population_size", "mutation_rate", "seed", "local_search_stage")
REQUIRED_STRATEGY_PARAMS_KEY: str = "strategy_params"


def setup_logging(
  output_dir: Path | str = OUTPUT_DIR,
  benchmark_name: str = BENCHMARK_NAME,
) -> Path:
  """Configure dual-target console and file logging for benchmark execution."""
  return setup_benchmark_logging(output_dir, benchmark_name)


def _strategy_names(config: AlgorithmConfig) -> dict[str, str]:
  """Return `{slot: class name}` for a declarative variant configuration."""
  return {slot: str(config.strategies.get(slot, {}).get("class", "None")) for slot in STRATEGY_SLOTS}


def validate_comparison_contract(
  algorithms: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
) -> list[str]:
  """Check that `configs/` still encodes the iso-algorithmic comparison."""
  errors: list[str] = []

  for slot, variant in COMPARISON_SLOTS.items():
    if variant not in algorithms:
      errors.append(f"slot '{slot}': variant '{variant}' missing from configs/algorithms.json")
      continue

    config = algorithms[variant]
    actual = _strategy_names(config)
    for strategy_slot, expected_class in EXPECTED_STRATEGIES[variant].items():
      if actual[strategy_slot] != expected_class:
        errors.append(
          f"slot '{slot}' ({variant}).{strategy_slot}: expected {expected_class}, found {actual[strategy_slot]}"
        )

    expected_use_gpu = variant != "CPU"
    if config.use_gpu != expected_use_gpu:
      errors.append(f"slot '{slot}' ({variant}).use_gpu: expected {expected_use_gpu}, found {config.use_gpu}")

    stage = str(config.params.get("local_search_stage", LOCAL_SEARCH_STAGE))
    if stage != LOCAL_SEARCH_STAGE:
      errors.append(f"slot '{slot}' ({variant}).local_search_stage: expected {LOCAL_SEARCH_STAGE}, found {stage}")

    for strategy_slot, key, expected_value in EXPECTED_STRATEGY_PARAMS:
      params = config.strategies.get(strategy_slot, {}).get("params", {})
      if key not in params:
        errors.append(f"slot '{slot}' ({variant}).{strategy_slot}.{key}: missing (expected {expected_value:g})")
      elif float(params[key]) != expected_value:
        errors.append(
          f"slot '{slot}' ({variant}).{strategy_slot}.{key}: expected {expected_value:g}, found {params[key]}"
        )

  overrides: dict[str, float | int | None] = {
    "population_size": POPULATION_SIZE,
    "mutation_rate": MUTATION_RATE,
    "seed": SEED,
  }
  actual_ga = {
    "population_size": float(ga_params.population_size),
    "mutation_rate": float(ga_params.mutation_rate),
    "seed": float(ga_params.seed),
  }
  for key, expected_value in EXPECTED_GA_PARAMS.items():
    if overrides[key] is not None:
      continue
    if actual_ga[key] != expected_value:
      errors.append(f"GA {key}: expected {expected_value:g}, found {actual_ga[key]:g} in configs/algorithms.json")

  return errors


def validate_effective_instances(
  algorithms: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
) -> list[str]:
  """Instantiate every variant and compare the values the run will actually use.

  `validate_comparison_contract` only reads `configs/`; this step catches a harder
  failure class: a parameter accepted by the config but then replaced by a
  constructor default or by an override inside `AlgorithmConfig.instantiate`.
  """
  errors: list[str] = []
  expected_seed = SEED if SEED is not None else 42
  expected_population = POPULATION_SIZE if POPULATION_SIZE is not None else 256
  expected_mutation = MUTATION_RATE if MUTATION_RATE is not None else 0.02

  for slot, variant in COMPARISON_SLOTS.items():
    if variant not in algorithms:
      continue

    instance = algorithms[variant].instantiate(
      ga_params,
      seed=expected_seed,
      local_search_stage=LOCAL_SEARCH_STAGE,
    )

    effective: dict[str, Any] = {
      "population_size": getattr(instance, "population_size", None),
      "mutation_rate": getattr(instance, "mutation_rate", None),
      "seed": getattr(instance, "seed", None),
      "local_search_stage": getattr(instance, "local_search_stage", None),
    }
    expected: dict[str, Any] = {
      "population_size": expected_population,
      "mutation_rate": expected_mutation,
      "seed": expected_seed,
      "local_search_stage": LOCAL_SEARCH_STAGE,
    }
    for key, expected_value in expected.items():
      if effective[key] != expected_value:
        errors.append(f"slot '{slot}' ({variant}): effective {key}={effective[key]!r}, declared {expected_value!r}")

    for strategy_slot, expected_class in EXPECTED_STRATEGIES[variant].items():
      strategy = getattr(instance, strategy_slot, None)
      actual_class = type(strategy).__name__ if strategy is not None else "None"
      if actual_class != expected_class:
        errors.append(f"slot '{slot}' ({variant}): effective {strategy_slot}={actual_class}, declared {expected_class}")

    for strategy_slot, key, expected_value in EXPECTED_STRATEGY_PARAMS:
      strategy = getattr(instance, strategy_slot, None)
      actual_value = getattr(strategy, key, None)
      if actual_value != expected_value:
        errors.append(
          f"slot '{slot}' ({variant}): effective {strategy_slot}.{key}={actual_value!r}, declared {expected_value!r}"
        )

    expected_init_gpu, expected_eval_gpu = EXPECTED_RETURN_GPU[variant]
    for strategy_slot, expected_gpu in (("initialization", expected_init_gpu), ("evaluation", expected_eval_gpu)):
      strategy = getattr(instance, strategy_slot, None)
      # A strategy without a `return_gpu` knob cannot return device arrays at all.
      if strategy is None or not hasattr(strategy, "return_gpu"):
        continue
      actual_gpu = strategy.return_gpu
      if actual_gpu != expected_gpu:
        errors.append(
          f"slot '{slot}' ({variant}): effective {strategy_slot}.return_gpu={actual_gpu!r}, declared {expected_gpu!r}"
        )

    local_search = getattr(instance, "local_search", None)
    expected_mode = EXPECTED_LOCAL_SEARCH_MODE[variant]
    actual_mode = getattr(local_search, "mode", None)
    if actual_mode != expected_mode:
      errors.append(
        f"slot '{slot}' ({variant}): effective local_search.mode={actual_mode!r}, declared {expected_mode!r}"
      )

  return errors


def validate_generation_budget(
  problems: Sequence[ProblemConfig],
  target_problem_names: Sequence[str] = PROBLEMS,
) -> list[str]:
  """Confirm the pipeline's hardcoded generation budget is the documented formula.

  `run_comprehensive_benchmark` computes `max_generations = 2 * n * sqrt(n)` internally;
  no caller argument reaches it, so a drift there would change every row of the table.
  """
  errors: list[str] = []
  target_set = set(target_problem_names)
  for problem in problems:
    if problem.name not in target_set:
      continue
    expected = int(2 * problem.size * math.sqrt(problem.size))
    actual = adaptive_generations(problem.size)
    if actual != expected:
      errors.append(f"problem '{problem.name}': adaptive_generations({problem.size})={actual}, expected {expected}")
  return errors


def validate_gpu_backend() -> list[str]:
  """Confirm the GPU variants get a real CuPy module.

  `run_single_algorithm` now resolves `xp = cupy_backend() if use_gpu else np`, and
  `cupy_backend()` returns `None` when the probe fails, so the failure surfaces loudly
  inside the run instead of silently executing NumPy code under a "GPU" label. This
  check moves that failure to the preflight, before any checkpoint is written.
  """
  errors: list[str] = []
  if not gpu_available():
    errors.append("no usable CUDA device (`protocols.backend.gpu_available()` is False)")
  if cupy_backend() is None:
    errors.append("`protocols.backend.cupy_backend()` returned None; GPU variants have no backend")
  return errors


def validate_problem_list(
  problems: Sequence[ProblemConfig],
  target_problem_names: Sequence[str] = PROBLEMS,
) -> list[str]:
  """Check that every target instance is configured with a usable optimal cost."""
  configured = {problem.name: problem for problem in problems}
  errors: list[str] = []
  for name in target_problem_names:
    problem = configured.get(name)
    if problem is None:
      errors.append(f"problem '{name}' missing from configs/problems.json")
    elif problem.optimal <= 0:
      errors.append(f"problem '{name}' has non-positive optimal cost ({problem.optimal})")
  return errors


def validate_existing_checkpoints(
  checkpoint_dir: Path,
  target_problems: Sequence[str] = PROBLEMS,
  algorithms: Sequence[str] = ALGORITHMS,
) -> list[str]:
  """Reject checkpoints written by a different architecture before resuming.

  The orchestrator reuses any checkpoint whose `successful_runs` reaches the
  target, so a stale cohort in the output directory would otherwise be trusted.
  """
  if not checkpoint_dir.exists():
    return []

  errors: list[str] = []
  target_prob_set = set(target_problems)
  target_alg_set = set(algorithms)
  for path in sorted(checkpoint_dir.glob("*.json")):
    data = load_checkpoint(path)
    algorithm = str(data.get("algorithm", ""))
    problem = str(data.get("problem_name", ""))
    if algorithm not in target_alg_set or problem not in target_prob_set:
      continue

    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in data]
    if missing:
      errors.append(f"{path.name}: missing {missing} (written before the modular checkpoint schema)")
      continue

    recorded = data.get("strategies") or {}
    for slot, expected_class in EXPECTED_STRATEGIES[algorithm].items():
      if recorded.get(slot) != expected_class:
        errors.append(f"{path.name}: strategies.{slot} is {recorded.get(slot)!r}, expected {expected_class!r}")

    algorithm_config = data.get("algorithm_config") or {}
    expected_config: tuple[tuple[str, Any], ...] = (
      ("population_size", POPULATION_SIZE if POPULATION_SIZE is not None else 256),
      ("mutation_rate", MUTATION_RATE if MUTATION_RATE is not None else 0.02),
      ("seed", SEED if SEED is not None else 42),
    )
    for key, expected_value in expected_config:
      if algorithm_config.get(key) != expected_value:
        errors.append(
          f"{path.name}: algorithm_config.{key} is {algorithm_config.get(key)!r}, expected {expected_value!r}"
        )

    missing_config = [key for key in REQUIRED_ALGORITHM_CONFIG_KEYS if key not in algorithm_config]
    if missing_config:
      errors.append(f"{path.name}: algorithm_config missing {missing_config} (pre-Design-B checkpoint)")
    elif REQUIRED_STRATEGY_PARAMS_KEY not in algorithm_config:
      errors.append(
        f"{path.name}: algorithm_config.{REQUIRED_STRATEGY_PARAMS_KEY} missing "
        "(checkpoint predates the parameter-level record)"
      )
    elif str(algorithm_config["local_search_stage"]) != LOCAL_SEARCH_STAGE:
      errors.append(
        f"{path.name}: algorithm_config.local_search_stage is "
        f"{algorithm_config['local_search_stage']!r}, expected {LOCAL_SEARCH_STAGE!r}"
      )

  return errors


def resolve_problem_inputs(problem_names: Sequence[str] | None = None) -> dict[str, Any]:
  """Prefer the DuckDB catalogue; fall back to explicit TSPLIB files."""
  target_problems = list(problem_names or PROBLEMS)
  if DB_PATH.exists():
    return {
      "problem_names": target_problems,
      "db_path": DB_PATH,
      "fallback_tsp_dir": FALLBACK_TSP_DIR,
    }

  tsp_files = [FALLBACK_TSP_DIR / f"{name}.tsp" for name in target_problems]
  missing = [str(path) for path in tsp_files if not path.exists()]
  if missing:
    raise FileNotFoundError(f"Neither {DB_PATH} nor the TSPLIB files {missing} are available")
  logging.warning(f"DuckDB catalogue {DB_PATH} not found; loading instances from {FALLBACK_TSP_DIR}")
  return {"tsp_files": tsp_files}


def log_effective_parameters(
  ga_params: GAParams,
  benchmark_config: BenchmarkConfig,
  algorithms: Mapping[str, AlgorithmConfig],
  repetitions: int,
  output_dir: Path,
  *,
  baseline_algorithm: str = BASELINE_ALGORITHM,
  cpu_size_threshold: int = CPU_SIZE_THRESHOLD,
  target_problems: Sequence[str] = PROBLEMS,
) -> None:
  """Log the values actually used, so a run stays auditable after the fact."""
  effective_seed = SEED if SEED is not None else ga_params.seed
  effective_population = POPULATION_SIZE if POPULATION_SIZE is not None else ga_params.population_size
  effective_mutation = MUTATION_RATE if MUTATION_RATE is not None else ga_params.mutation_rate

  logging.info("=" * 80)
  logging.info("CHAPTER 4 ISO-ALGORITHMIC COMPARISON")
  logging.info("=" * 80)
  logging.info(f"  Problems ({len(target_problems)}): {', '.join(target_problems)}")
  logging.info(f"  Algorithms: {', '.join(ALGORITHMS)}")
  for slot, variant in COMPARISON_SLOTS.items():
    ls_params = algorithms[variant].strategies.get("local_search", {}).get("params", {})
    ls_mode = ls_params.get("mode", EXPECTED_LOCAL_SEARCH_MODE.get(variant, "N/A"))
    cand_k = ls_params.get("candidate_k", CANDIDATE_K)
    max_iter = ls_params.get("max_iterations", LOCAL_SEARCH_MAX_ITERATIONS)
    logging.info(
      f"    {slot:12s} -> {variant} (use_gpu={algorithms[variant].use_gpu}, mode={ls_mode}, "
      f"candidate_k={cand_k}, max_iterations={max_iter})"
    )
  logging.info(f"  Repetitions: {repetitions} | seed={effective_seed}")
  logging.info(f"  Population: {effective_population} | mutation_rate: {effective_mutation}")
  logging.info(f"  Selection tournament k: {SELECTION_K}")
  logging.info(f"  Candidate list width (candidate_k): {CANDIDATE_K} (active across all 2-opt search passes)")
  logging.info(f"  Local search max iterations: {LOCAL_SEARCH_MAX_ITERATIONS}")
  logging.info(f"  Local search stage: {LOCAL_SEARCH_STAGE} | cpu_size_threshold: {cpu_size_threshold}")
  logging.info(f"  Baseline algorithm: {baseline_algorithm}")
  logging.info(f"  Early stopping patience: {benchmark_config.patience} (None = adaptive 2*sqrt(n))")
  logging.info(f"  Output: {output_dir}")
  logging.info("=" * 80)


def log_generation_budget(
  problems: Sequence[ProblemConfig],
  target_problems: Sequence[str] = PROBLEMS,
) -> None:
  """Log the internal generation budget per instance (`2 * n * sqrt(n)`, not caller-controlled)."""
  logging.info("  Generation budget (adaptive_generations = 2*n*sqrt(n)):")
  target_set = set(target_problems)
  for problem in problems:
    if problem.name in target_set:
      logging.info(f"    {problem.name:9s} n={problem.size:<5d} max_generations={adaptive_generations(problem.size)}")


def log_comparison(
  results: Mapping[str, Mapping[str, dict[str, Any]]],
  *,
  cpu_size_threshold: int = CPU_SIZE_THRESHOLD,
) -> None:
  """Print the four-slot comparison table for the completed cohort."""
  logging.info("")
  logging.info("=" * 80)
  logging.info("CHAPTER 4 COMPARISON (mean gap and mean time per problem)")
  logging.info("=" * 80)
  header = f"{'problem':10s} " + " ".join(f"{slot:>22s}" for slot in COMPARISON_SLOTS)
  logging.info(header)
  logging.info("-" * len(header))

  for problem, algorithms in results.items():
    cells: list[str] = []
    for variant in ALGORITHMS:
      record = algorithms.get(variant)
      if record is None:
        cells.append(f"{'(skipped)':>22s}")
      else:
        gap = float(record.get("mean_gap", float("nan")))
        elapsed = float(record.get("mean_time", float("nan")))
        cells.append(f"{gap:8.2f}% {elapsed:9.1f}s".rjust(22))
    logging.info(f"{problem:10s} " + " ".join(cells))

  logging.info("-" * len(header))
  logging.info(f"'skipped' marks the CPU variant above cpu_size_threshold (n > {cpu_size_threshold}).")


def run(
  *,
  repetitions: int = REPETITIONS,
  output_dir: Path = OUTPUT_DIR,
  benchmark_name: str = BENCHMARK_NAME,
  cpu_size_threshold: int = CPU_SIZE_THRESHOLD,
  baseline_algorithm: str = BASELINE_ALGORITHM,
  skip_cpu: bool = False,
  incremental_runs: bool = INCREMENTAL_RUNS,
  problem_names: Sequence[str] | None = None,
  dry_run: bool = False,
) -> int:
  """Execute the Chapter 4 comparison and return a process exit code."""
  resolved_output_dir = Path(output_dir)
  target_problems = list(problem_names or PROBLEMS)

  configs = load_all_configs(CONFIG_DIR)
  ga_params: GAParams = configs["ga_params"]
  algorithms: dict[str, AlgorithmConfig] = configs["algorithms"]
  benchmark_config: BenchmarkConfig = configs["benchmark"]
  problems: list[ProblemConfig] = configs["problems"]

  for warning in validate_configs(configs):
    logging.warning(f"config: {warning}")

  errors = [
    *validate_comparison_contract(algorithms, ga_params),
    *validate_effective_instances(algorithms, ga_params),
    *validate_problem_list(problems, target_problems),
    *validate_generation_budget(problems, target_problems),
    *validate_existing_checkpoints(resolved_output_dir / "checkpoints", target_problems, ALGORITHMS),
    *validate_gpu_backend(),
  ]
  if errors:
    logging.error("Refusing to run: the declared comparison is not reproducible against the current state.")
    for error in errors:
      logging.error(f"  - {error}")
    logging.error("Move or delete %s to start a fresh cohort.", resolved_output_dir / "checkpoints")
    return 2

  log_effective_parameters(
    ga_params,
    benchmark_config,
    algorithms,
    repetitions,
    resolved_output_dir,
    baseline_algorithm=baseline_algorithm,
    cpu_size_threshold=cpu_size_threshold,
    target_problems=target_problems,
  )
  log_generation_budget(problems, target_problems)

  if dry_run:
    logging.info("Dry-run requested: all preflight validation checks passed. Exiting without execution.")
    return 0

  results = run_comprehensive_benchmark(
    config_dir=CONFIG_DIR,
    output_dir=resolved_output_dir,
    **resolve_problem_inputs(target_problems),
    algorithms=list(ALGORITHMS),
    repetitions=repetitions,
    benchmark_name=benchmark_name,
    local_search_stage=LOCAL_SEARCH_STAGE,
    cpu_size_threshold=cpu_size_threshold,
    baseline_algorithm=baseline_algorithm,
    skip_cpu=skip_cpu,
    incremental_runs=incremental_runs,
    seed=SEED,
    population_size=POPULATION_SIZE,
    mutation_rate=MUTATION_RATE,
  )

  log_comparison(results, cpu_size_threshold=cpu_size_threshold)
  return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parse command-line arguments for the Chapter 4 benchmark runner."""
  parser = argparse.ArgumentParser(
    description=(
      "Chapter 4 Validation Benchmark Runner: executes iso-algorithmic "
      "comparison across CPU and GPU variants."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=OUTPUT_DIR,
    help="Output directory for checkpoints, tables, and statistics.",
  )
  parser.add_argument(
    "-r",
    "--repetitions",
    type=int,
    default=REPETITIONS,
    help="Number of independent stochastic repetitions per configuration.",
  )
  parser.add_argument(
    "--cpu-threshold",
    type=int,
    default=CPU_SIZE_THRESHOLD,
    dest="cpu_size_threshold",
    help="Maximum problem dimension (n) for CPU execution.",
  )
  parser.add_argument(
    "--baseline",
    type=str,
    default=BASELINE_ALGORITHM,
    dest="baseline_algorithm",
    help="Baseline algorithm for paired speedup calculation (e.g. CPU, HybridNaive).",
  )
  parser.add_argument(
    "--benchmark-name",
    type=str,
    default=BENCHMARK_NAME,
    help="Identifier prefix for generated report tables and TikZ diagrams.",
  )
  parser.add_argument(
    "--skip-cpu",
    action="store_true",
    help="Omit the CPU baseline completely from benchmark execution.",
  )
  parser.add_argument(
    "--incremental",
    action=argparse.BooleanOptionalAction,
    default=INCREMENTAL_RUNS,
    dest="incremental_runs",
    help="Resume from existing valid checkpoints and run missing repetitions.",
  )
  parser.add_argument(
    "--problems",
    nargs="+",
    dest="problem_names",
    help="Custom subset of TSPLIB problem instances to execute (overrides pilot cohort).",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate environment, configuration contracts, and generation budgets without running.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Execute the Chapter 4 validation benchmark CLI."""
  setup_logging()
  args = parse_args(argv)
  logging.info("Starting Chapter 4 Validation Benchmark...")
  return run(
    repetitions=args.repetitions,
    output_dir=args.output_dir,
    benchmark_name=args.benchmark_name,
    cpu_size_threshold=args.cpu_size_threshold,
    baseline_algorithm=args.baseline_algorithm,
    skip_cpu=args.skip_cpu,
    incremental_runs=args.incremental_runs,
    problem_names=args.problem_names,
    dry_run=args.dry_run,
  )


if __name__ == "__main__":
  sys.exit(main())
