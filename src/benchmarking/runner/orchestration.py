"""
Orchestration Module for Benchmark V2
======================================

Main benchmark loop orchestrating all v2 modules.

This module provides the high-level benchmark workflow:
1. Load configurations
2. Check for existing checkpoints
3. Execute benchmark loop (Problems x Algorithms)
4. Perform per-problem statistical analysis
5. Perform cross-problem analysis (stratified)
6. Generate result tables

Usage:
    from benchmarking.runner.orchestration import run_comprehensive_benchmark
    from argparse import Namespace

    args = Namespace(
        repetitions=30,
        skip_cpu=False,
        test_mode=False
    )

    results = run_comprehensive_benchmark(args)

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import gc
import json
import logging
import os
import tempfile
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
  from data_models.problem import Problem

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False

from benchmarking.analysis.aggregate_statistics import (
  perform_cross_problem_analysis_stratified,
)
from benchmarking.analysis.problem_statistics import perform_statistical_analysis
from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.runner.algorithm_runner import (
  adaptive_generations,
  run_single_algorithm,
)
from benchmarking.runner.checkpoint_io import (
  CheckpointManager,
  load_checkpoint,
  numpy_to_json_converter,
  save_checkpoint_atomic,
)
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  BenchmarkConfig,
  GAParams,
  ProblemConfig,
  build_problem_configs_from_tsp_files,
  default_config_dir,
  load_all_configs,
)
from loaders.database_loader import DatabaseLoader
from loaders.tsplib_loader import TsplibLoader

# =============================================================================
# Helper Functions
# =============================================================================


def _log_benchmark_configuration(
  args: Namespace,
  selected_algorithms: list[str],
  problem_configs: list[ProblemConfig],
  ga_params: GAParams,
  benchmark_config: BenchmarkConfig,
) -> None:
  """Log benchmark configuration at startup."""
  logging.info("=" * 80)
  logging.info("CHAPTER 4 VALIDATION: GPU-ACCELERATED GENETIC ALGORITHM BENCHMARK")
  logging.info("=" * 80)
  logging.info("")

  logging.info("Configuration:")
  logging.info(f"  Algorithms: {', '.join(selected_algorithms)}")
  logging.info(f"  Problems: {len(problem_configs)} TSP instances")
  logging.info(f"  Repetitions: {args.repetitions}")
  logging.info(f"  CPU threshold: n ≤ {benchmark_config.cpu_size_threshold}")
  logging.info("")

  logging.info("GA Parameters:")
  logging.info(f"  Population size: {ga_params.population_size}")
  logging.info(f"  Mutation rate: {ga_params.mutation_rate}")
  logging.info(f"  Tournament size: {ga_params.tournament_size}")
  logging.info("  Early stop patience: Adaptive (2x√n generations)")
  logging.info("")


def _determine_problem_algorithms(
  problem_size: int,
  selected_algorithms: list[str],
  algorithm_configs: dict[str, AlgorithmConfig],
  cpu_threshold: int,
) -> list[str]:
  """Determine which algorithms to run for a problem based on size."""
  if problem_size <= cpu_threshold:
    return selected_algorithms
  else:
    return [alg for alg in selected_algorithms if algorithm_configs[alg].use_gpu]


def _save_problem_statistics_file(
  problem_name: str,
  problem_size: int,
  optimal_cost: float,
  problem_results: dict[str, dict[str, Any]],
  statistical_test_results: dict[str, Any],
  checkpoint_manager: CheckpointManager,
) -> None:
  """Save per-problem statistics to JSON file."""
  stats_path: Path = checkpoint_manager.get_problem_stats_path(problem_name)

  try:
    problem_stats = {
      "problem_info": {
        "name": problem_name,
        "size": problem_size,
        "optimal_cost": optimal_cost,
      },
      "algorithms": {},
      "statistical_tests": statistical_test_results,
    }

    for alg_name, results in problem_results.items():
      times = np.array(results.get("raw_times", results.get("times", [])))
      initial_costs = np.array(results.get("raw_initial_costs", results.get("initial_costs", [])))
      costs = np.array(results.get("raw_costs", results.get("final_costs", [])))
      gaps = np.array(results.get("raw_gaps", results.get("gaps", [])))
      generations = np.array(results.get("raw_generations", results.get("generations", [])))

      valid_mask = ~np.isnan(times)
      problem_stats["algorithms"][alg_name] = {
        "mean_time": float(np.nanmean(times)),
        "std_time": float(np.nanstd(times)),
        "mean_initial_cost": float(np.nanmean(initial_costs)),
        "std_initial_cost": float(np.nanstd(initial_costs)),
        "mean_cost": float(np.nanmean(costs)),
        "std_cost": float(np.nanstd(costs)),
        "mean_gap": float(np.nanmean(gaps)),
        "std_gap": float(np.nanstd(gaps)),
        "mean_generations": float(np.nanmean(generations)),
        "std_generations": float(np.nanstd(generations)),
        "success_rate": float(np.sum(valid_mask) / len(times)) if len(times) > 0 else 0.0,
        "backend": results.get("backend", "unknown"),
        "algorithm_config": results.get("algorithm_config", {}),
        "strategies": results.get("strategies", {}),
      }

    with open(stats_path, "w", encoding="utf-8") as f:
      json.dump(problem_stats, f, indent=2, default=numpy_to_json_converter)
    logging.info(f"✓ Problem statistics saved: {stats_path.name}\n")
  except Exception as e:
    logging.warning(f"Failed to save problem statistics: {e}\n")


# =============================================================================
# Main Benchmark Function
# =============================================================================


def run_comprehensive_benchmark(
  args: Namespace,
  *,
  tsp_files: Sequence[str | Path] | None = None,
  problem_configs: list[ProblemConfig] | None = None,
  algorithms: Sequence[str] | None = None,
  output_dir: str | Path | None = None,
  benchmark_name: str | None = None,
  local_search_stage: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
  """Run comprehensive benchmark on all configured problems and algorithms."""
  logging.info("Loading configurations...")

  cli_config_dir: Any | None = getattr(args, "config_dir", None)
  config_dir: Path = Path(cli_config_dir) if cli_config_dir else default_config_dir()
  configs: dict[str, Any] = load_all_configs(config_dir)

  ga_params = configs["ga_params"]
  algorithm_configs = configs["algorithms"]
  benchmark_config = configs["benchmark"]
  default_problems = configs["problems"]

  if hasattr(args, "incremental_runs") and args.incremental_runs is not None:
    logging.info(f"Applying CLI override: incremental_runs={args.incremental_runs}")
    benchmark_config.incremental_runs = args.incremental_runs

  if hasattr(args, "cpu_size_threshold") and args.cpu_size_threshold is not None:
    benchmark_config.cpu_size_threshold = int(args.cpu_size_threshold)

  tsp_path_map: dict[str, Path] = {}
  cli_tsp_files: Sequence[str | Path] | Any | None = (
    tsp_files or getattr(args, "tsp_files", None) or getattr(args, "problems", None)
  )
  if problem_configs is not None:
    resolved_problems: list[ProblemConfig] = list(problem_configs)
  elif cli_tsp_files:
    resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
      cli_tsp_files,
      existing_configs=default_problems,
    )
  else:
    resolved_problems = list(default_problems)

  problem_configs = resolved_problems

  if getattr(args, "test_mode", False) and not cli_tsp_files:
    problem_configs = problem_configs[:2]
    logging.info("=" * 80)
    logging.info("⚠️  TEST MODE ENABLED")
    logging.info("=" * 80)
    logging.info(f"Running on first 2 problems only ({problem_configs[0].name}, {problem_configs[1].name})")
    logging.info("Use without --test-mode for full benchmark")
    logging.info("=" * 80)
    logging.info("")

  cli_algorithms: Sequence[str] | Any | None = algorithms or getattr(args, "algorithms", None)
  candidate_algorithms: list[str] | list[Any] = (
    list(cli_algorithms) if cli_algorithms else list(algorithm_configs.keys())
  )

  selected_algorithms = []
  for alg_name in candidate_algorithms:
    if alg_name not in algorithm_configs:
      raise ValueError(f"Unknown algorithm '{alg_name}'. Available: {list(algorithm_configs.keys())}")
    if getattr(args, "skip_cpu", False) and alg_name == "CPU":
      continue
    selected_algorithms.append(alg_name)

  _log_benchmark_configuration(args, selected_algorithms, problem_configs, ga_params, benchmark_config)

  raw_out: str | Path | Any = output_dir if output_dir is not None else getattr(args, "output_dir", "results_v3")
  resolved_output_dir: Path = Path(str(raw_out)) if raw_out is not None else Path("results_v3")
  resolved_benchmark_name = str(benchmark_name or getattr(args, "benchmark_name", "chapter4_validation"))

  checkpoint_manager = CheckpointManager(
    checkpoint_dir=resolved_output_dir / "checkpoints",
    problem_stats_dir=resolved_output_dir / "problem_statistics",
  )

  completed, total = checkpoint_manager.count_completed_checkpoints(
    [p.__dict__ for p in problem_configs],
    selected_algorithms,
    args.repetitions,
    benchmark_config.cpu_size_threshold,
  )
  if completed > 0:
    logging.info(f"\n✓ Found {completed}/{total} completed checkpoints")
    logging.info("  Resuming from previous run...\n")

  project_root: Path = Path(__file__).resolve().parents[3]
  db_path: Path = project_root / "datasets" / "routing.duckdb"
  tsp_fallback_dir: Path = project_root / "datasets" / "tsp"

  if not db_path.exists() and not tsp_fallback_dir.exists() and not tsp_path_map:
    raise FileNotFoundError(f"Neither database ({db_path}) nor TSPLIB directory ({tsp_fallback_dir}) found")

  analyzer = StatisticalAnalyzer()
  problem_results: dict[str, dict[str, dict[str, Any]]] = {}

  for prob_idx, problem_config in enumerate(problem_configs, 1):
    problem_name: str = problem_config.name
    optimal_cost: float = problem_config.optimal
    problem_size: int = problem_config.size

    logging.info("\n" + "=" * 80)
    logging.info(f"PROBLEM {prob_idx}/{len(problem_configs)}: {problem_name}")
    logging.info(f"Size: {problem_size} cities | Optimal: {optimal_cost}")
    logging.info("=" * 80)

    problem_algorithms: list[str] = _determine_problem_algorithms(
      problem_size,
      selected_algorithms,
      algorithm_configs,
      benchmark_config.cpu_size_threshold,
    )

    if problem_size <= benchmark_config.cpu_size_threshold:
      logging.info(f"Running {len(problem_algorithms)} algorithm(s) (n≤{benchmark_config.cpu_size_threshold})\n")
    else:
      logging.info(f"Running {len(problem_algorithms)} GPU algorithms only (n>{benchmark_config.cpu_size_threshold})\n")

    try:
      if problem_name in tsp_path_map:
        tsp_path: Path = tsp_path_map[problem_name]
        with TsplibLoader(tsp_path.parent) as tsp_loader:
          problem: Problem = tsp_loader.load(tsp_path)
      else:
        with DatabaseLoader(str(db_path), fallback_tsp_dir=tsp_fallback_dir) as db_loader:
          problem = db_loader.load(problem_name)
    except Exception as e:
      logging.error(f"Failed to load {problem_name}: {e}")
      continue

    problem_results[problem_name] = {}
    max_generations: int = adaptive_generations(problem_size)
    customers: list[int] = list(range(problem_size))

    for algo_idx, alg_name in enumerate(problem_algorithms, 1):
      alg_config = algorithm_configs[alg_name]
      backend_type: Literal["GPU"] | Literal["CPU"] = "GPU" if alg_config.use_gpu else "CPU"

      checkpoint_path: Path = checkpoint_manager.get_checkpoint_path(problem_name, alg_name)
      existing_checkpoint: dict[str, Any] | None = None
      incremental_mode = benchmark_config.incremental_runs

      checkpoint_exists: bool = checkpoint_path.exists()

      if checkpoint_exists:
        existing_checkpoint = load_checkpoint(checkpoint_path)
        existing_runs = existing_checkpoint.get("successful_runs", 0) if existing_checkpoint else 0

        if incremental_mode:
          new_reps_to_add = args.repetitions
          total_target_reps = existing_runs + new_reps_to_add

          logging.info(
            f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
            f"✓ Resuming from checkpoint ({existing_runs} runs) - "
            f"adding {new_reps_to_add} more runs → {total_target_reps} total"
          )
          actual_repetitions = new_reps_to_add
        else:
          if existing_runs >= args.repetitions:
            logging.info(
              f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
              f"✓ Loading from checkpoint (complete)"
            )
            if existing_checkpoint is not None:
              problem_results[problem_name][alg_name] = existing_checkpoint

            if alg_config.use_gpu and CUPY_AVAILABLE:
              cp.get_default_memory_pool().free_all_blocks()

            continue
          else:
            actual_repetitions = args.repetitions - existing_runs
            logging.info(
              f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
              f"Resuming: {existing_runs}/{args.repetitions} runs exist, "
              f"completing remaining {actual_repetitions}"
            )
            incremental_mode = True
      else:
        actual_repetitions = args.repetitions
        logging.info(
          f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
          f"Running {actual_repetitions} repetitions..."
        )

      effective_stage = local_search_stage or getattr(args, "local_search_stage", None)
      algorithm_instance = alg_config.instantiate(ga_params, local_search_stage=effective_stage)

      results: dict[str, Any] = run_single_algorithm(
        algorithm_name=alg_name,
        algorithm_instance=algorithm_instance,
        problem=problem,
        problem_name=problem_name,
        problem_size=problem_size,
        customers=customers,
        max_generations=max_generations,
        optimal_cost=optimal_cost,
        repetitions=actual_repetitions,
        use_gpu=alg_config.use_gpu,
        patience=benchmark_config.patience,
        existing_checkpoint=existing_checkpoint,
        incremental_mode=incremental_mode and existing_checkpoint is not None,
      )

      def _get_strat_name(strat: Any, default: str) -> str:
        if strat is None:
          return default
        return getattr(strat, "__name__", type(strat).__name__)

      results["problem_name"] = problem_name
      results["problem_size"] = problem_size
      results["problem_type"] = getattr(problem, "problem_type", "TSP")
      results["edge_type"] = getattr(problem, "edge_type", "EUC_2D")
      results["optimal_cost"] = optimal_cost
      results["max_generations_configured"] = max_generations
      results["backend"] = "GPU" if alg_config.use_gpu else "CPU"
      results["timestamp"] = datetime.now().isoformat()

      results["algorithm_config"] = {
        "population_size": getattr(algorithm_instance, "population_size", ga_params.population_size),
        "mutation_rate": getattr(algorithm_instance, "mutation_rate", ga_params.mutation_rate),
        "tournament_size": getattr(algorithm_instance, "tournament_size", ga_params.tournament_size),
        "two_opt_iterations": getattr(
          algorithm_instance,
          "two_opt_iterations",
          ga_params.two_opt_iterations,
        ),
        "seed": getattr(algorithm_instance, "seed", ga_params.seed),
      }

      results["strategies"] = {
        "initialization": _get_strat_name(
          getattr(algorithm_instance, "initialization", None), "RandomPermutationInitialization"
        ),
        "selection": _get_strat_name(getattr(algorithm_instance, "selection", None), "TournamentSelection"),
        "crossover": _get_strat_name(getattr(algorithm_instance, "crossover", None), "OrderCrossover"),
        "mutation": _get_strat_name(getattr(algorithm_instance, "mutation", None), "SwapMutation"),
        "local_search": _get_strat_name(getattr(algorithm_instance, "local_search", None), "BatchGPUTwoOpt"),
        "evaluation": _get_strat_name(getattr(algorithm_instance, "evaluation", None), "GPUCostEvaluation"),
      }

      results["construction_heuristic"] = "random_initialization"

      checkpoint_path = checkpoint_manager.get_checkpoint_path(problem_name, alg_name)
      save_checkpoint_atomic(checkpoint_path, results)
      problem_results[problem_name][alg_name] = results

      del algorithm_instance
      gc.collect()
      if alg_config.use_gpu and CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    logging.info(f"\n{'─' * 80}")
    logging.info(f"STATISTICS FOR {problem_name}")
    logging.info(f"{'─' * 80}")

    statistical_test_results = perform_statistical_analysis(
      problem_name=problem_name,
      optimal_cost=optimal_cost,
      results=problem_results[problem_name],
      analyzer=analyzer,
    )

    _save_problem_statistics_file(
      problem_name,
      problem_size,
      optimal_cost,
      problem_results[problem_name],
      statistical_test_results,
      checkpoint_manager,
    )

    del problem
    gc.collect()

  logging.info("\n" + "=" * 80)
  logging.info("CROSS-PROBLEM ANALYSIS")
  logging.info("=" * 80)

  if len(problem_results) > 1:
    perform_cross_problem_analysis_stratified(
      all_results=problem_results,
      problem_configs=problem_configs,
      analyzer=analyzer,
      cpu_threshold=benchmark_config.cpu_size_threshold,
    )

  if problem_results:
    tables_dir = resolved_output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_result_tables(
      all_results=problem_results,
      problem_configs=problem_configs,
      output_dir=tables_dir,
      benchmark_name=resolved_benchmark_name,
    )

  logging.info("=" * 80)
  logging.info("BENCHMARK COMPLETE")
  logging.info("=" * 80)

  return problem_results
