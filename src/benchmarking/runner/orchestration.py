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

    results = run_comprehensive_benchmark(
        config_dir="configs",
        output_dir="results",
        repetitions=30,
        skip_cpu=False,
        test_mode=False,
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import gc
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
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
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
  load_all_configs,
)
from loaders.database_loader import DatabaseLoader
from loaders.tsplib_loader import TsplibLoader

# =============================================================================
# Helper Functions
# =============================================================================


def _log_benchmark_configuration(
  repetitions: int,
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
  logging.info(f"  Repetitions: {repetitions}")
  logging.info(f"  CPU threshold: n ≤ {benchmark_config.cpu_size_threshold}")
  logging.info("")

  logging.info("GA Parameters:")
  logging.info(f"  Population size: {ga_params.population_size}")
  logging.info(f"  Mutation rate: {ga_params.mutation_rate}")
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
  problem_metadata: dict[str, Any] | None = None,
) -> None:
  """Save per-problem statistics to JSON file."""
  stats_path: Path = checkpoint_manager.get_problem_stats_path(problem_name)

  try:
    problem_info: dict[str, Any] = {
      "name": problem_name,
      "size": problem_size,
      "optimal_cost": optimal_cost,
    }
    if problem_metadata:
      problem_info.update(problem_metadata)

    problem_stats = {
      "problem_info": problem_info,
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
  *,
  config_dir: str | Path,
  output_dir: str | Path,
  tsp_files: Sequence[str | Path] | None = None,
  problem_names: Sequence[str] | None = None,
  db_path: str | Path | None = None,
  fallback_tsp_dir: str | Path | None = None,
  problem_configs: list[ProblemConfig] | None = None,
  algorithms: Sequence[str] | None = None,
  algorithm_configs_override: dict[str, AlgorithmConfig] | None = None,
  repetitions: int | None = None,
  benchmark_name: str = "chapter4_validation",
  local_search_stage: str | None = None,
  skip_cpu: bool = False,
  incremental_runs: bool | None = None,
  cpu_size_threshold: int | None = None,
  test_mode: bool = False,
  patience: int | None = None,
  seed: int | None = None,
  population_size: int | None = None,
  mutation_rate: float | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
  """Run comprehensive benchmark on all configured problems and algorithms."""
  logging.info("Loading configurations...")

  resolved_config_dir: Path = Path(config_dir)
  configs: dict[str, Any] = load_all_configs(resolved_config_dir)

  ga_params: GAParams = configs["ga_params"]
  algorithm_configs: dict[str, AlgorithmConfig] = algorithm_configs_override or configs["algorithms"]
  benchmark_config: BenchmarkConfig = configs["benchmark"]
  default_problems: list[ProblemConfig] = configs["problems"]

  if patience is not None:
    logging.info(f"Applying override: patience={patience}")
    benchmark_config.patience = patience

  effective_seed: int = seed if seed is not None else ga_params.seed

  param_overrides: dict[str, Any] = {}
  if population_size is not None:
    param_overrides["population_size"] = population_size
  if mutation_rate is not None:
    param_overrides["mutation_rate"] = mutation_rate

  if incremental_runs is not None:
    logging.info(f"Applying override: incremental_runs={incremental_runs}")
    benchmark_config.incremental_runs = incremental_runs

  if cpu_size_threshold is not None:
    benchmark_config.cpu_size_threshold = int(cpu_size_threshold)

  effective_repetitions = repetitions if repetitions is not None else benchmark_config.repetitions
  effective_skip_cpu = skip_cpu
  effective_test_mode = test_mode

  resolved_db_path: Path | None = Path(db_path) if db_path is not None else None
  resolved_fallback_dir: Path | None = Path(fallback_tsp_dir) if fallback_tsp_dir is not None else None

  tsp_path_map: dict[str, Path] = {}
  if problem_configs is not None:
    resolved_problems: list[ProblemConfig] = list(problem_configs)
  elif problem_names:
    if resolved_db_path is None:
      raise ValueError("db_path must be provided when problem_names is specified")
    resolved_problems = build_problem_configs_from_duckdb(
      db_path=resolved_db_path,
      problem_names=problem_names,
      existing_configs=default_problems,
    )
  elif tsp_files:
    resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
      tsp_files,
      existing_configs=default_problems,
    )
  else:
    resolved_problems = list(default_problems)

  problem_configs = resolved_problems

  if effective_test_mode and not tsp_files:
    problem_configs = problem_configs[:2]
    logging.info("=" * 80)
    logging.info("⚠️  TEST MODE ENABLED")
    logging.info("=" * 80)
    logging.info(f"Running on first 2 problems only ({problem_configs[0].name}, {problem_configs[1].name})")
    logging.info("Use without --test-mode for full benchmark")
    logging.info("=" * 80)
    logging.info("")

  candidate_algorithms: list[str] = (
    list(algorithms) if algorithms else list(algorithm_configs.keys())
  )

  selected_algorithms = []
  for alg_name in candidate_algorithms:
    if alg_name not in algorithm_configs:
      raise ValueError(f"Unknown algorithm '{alg_name}'. Available: {list(algorithm_configs.keys())}")
    if effective_skip_cpu and not algorithm_configs[alg_name].use_gpu:
      continue
    selected_algorithms.append(alg_name)

  _log_benchmark_configuration(effective_repetitions, selected_algorithms, problem_configs, ga_params, benchmark_config)

  resolved_output_dir: Path = Path(output_dir)
  resolved_benchmark_name = str(benchmark_name)

  checkpoint_manager = CheckpointManager(
    checkpoint_dir=resolved_output_dir / "checkpoints",
    problem_stats_dir=resolved_output_dir / "problem_statistics",
  )

  completed, total = checkpoint_manager.count_completed_checkpoints(
    [p.__dict__ for p in problem_configs],
    selected_algorithms,
    effective_repetitions,
    benchmark_config.cpu_size_threshold,
  )
  if completed > 0:
    logging.info(f"\n✓ Found {completed}/{total} completed checkpoints")
    logging.info("  Resuming from previous run...\n")

  if not tsp_path_map:
    has_db = resolved_db_path is not None and resolved_db_path.exists()
    has_fallback = resolved_fallback_dir is not None and resolved_fallback_dir.exists()
    if not has_db and not has_fallback:
      raise FileNotFoundError(
        f"Neither database ({resolved_db_path}) nor TSPLIB directory ({resolved_fallback_dir}) found"
      )

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
        if resolved_db_path is None:
          raise ValueError(f"db_path is required to load problem '{problem_name}' from database")
        with DatabaseLoader(str(resolved_db_path), fallback_tsp_dir=resolved_fallback_dir) as db_loader:
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
          new_reps_to_add = effective_repetitions
          total_target_reps = existing_runs + new_reps_to_add

          logging.info(
            f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
            f"✓ Resuming from checkpoint ({existing_runs} runs) - "
            f"adding {new_reps_to_add} more runs → {total_target_reps} total"
          )
          actual_repetitions = new_reps_to_add
        else:
          if existing_runs >= effective_repetitions:
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
            actual_repetitions = effective_repetitions - existing_runs
            logging.info(
              f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
              f"Resuming: {existing_runs}/{effective_repetitions} runs exist, "
              f"completing remaining {actual_repetitions}"
            )
            incremental_mode = True
      else:
        actual_repetitions = effective_repetitions
        logging.info(
          f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
          f"Running {actual_repetitions} repetitions..."
        )

      effective_stage = local_search_stage

      def _factory(s: int, ac: AlgorithmConfig = alg_config, stage: str | None = effective_stage) -> Any:
        return ac.instantiate(
          ga_params,
          seed=s,
          local_search_stage=stage,
          param_overrides=param_overrides or None,
          strategy_overrides=strategy_overrides,
        )

      algorithm_instance = _factory(effective_seed)

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
        seed_base=effective_seed,
        existing_checkpoint=existing_checkpoint,
        incremental_mode=incremental_mode and existing_checkpoint is not None,
        algorithm_factory=_factory,
      )

      def _get_strat_name(strat: Any) -> str:
        if strat is None:
          return "None"
        return getattr(strat, "__name__", type(strat).__name__)

      results["problem_name"] = problem_name
      results["problem_size"] = problem_size
      results["problem_type"] = getattr(problem, "problem_type", "TSP")
      results["edge_type"] = getattr(problem, "edge_type", "EUC_2D")
      results["optimal_cost"] = optimal_cost
      results["max_generations_configured"] = max_generations
      results["backend"] = "GPU" if alg_config.use_gpu else "CPU"
      results["timestamp"] = datetime.now().isoformat()
      results["problem_metadata"] = problem.to_metadata_dict(optimal_cost)
      results["comment"] = problem.comment
      results["edge_weight_format"] = problem.edge_weight_format
      results["tsplib_name"] = problem.tsplib_name
      results["source_file"] = problem.source_file
      results["file_checksum"] = problem.file_checksum

      algo_cfg_data: dict[str, Any] = {
        "population_size": getattr(algorithm_instance, "population_size", ga_params.population_size),
        "mutation_rate": getattr(algorithm_instance, "mutation_rate", ga_params.mutation_rate),
        "seed": getattr(algorithm_instance, "seed", ga_params.seed),
      }
      if hasattr(algorithm_instance, "local_search_stage"):
        algo_cfg_data["local_search_stage"] = str(algorithm_instance.local_search_stage)
      results["algorithm_config"] = algo_cfg_data

      results["strategies"] = {
        "initialization": _get_strat_name(getattr(algorithm_instance, "initialization", None)),
        "selection": _get_strat_name(getattr(algorithm_instance, "selection", None)),
        "crossover": _get_strat_name(getattr(algorithm_instance, "crossover", None)),
        "mutation": _get_strat_name(getattr(algorithm_instance, "mutation", None)),
        "local_search": _get_strat_name(getattr(algorithm_instance, "local_search", None)),
        "evaluation": _get_strat_name(getattr(algorithm_instance, "evaluation", None)),
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

    statistical_test_results: dict[str, Any] = perform_statistical_analysis(
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
      problem_metadata=problem.to_metadata_dict(optimal_cost),
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
