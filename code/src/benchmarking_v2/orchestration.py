"""
Orchestration Module for Benchmark V2
======================================

Main benchmark loop orchestrating all v2 modules.

This module provides the high-level benchmark workflow:
1. Load configurations
2. Check for existing checkpoints
3. Execute benchmark loop (Problems × Algorithms)
4. Perform per-problem statistical analysis
5. Perform cross-problem analysis (stratified)
6. Generate result tables

Uses all v2 modules to provide a clean, modular benchmark execution.

Usage:
    from src.benchmarking_v2.orchestration import run_comprehensive_benchmark
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
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from src.loaders.database_loader import DatabaseLoader
from src.benchmarking_v2.statistics import StatisticalAnalyzer

from src.benchmarking_v2.config_loader import (
    load_all_configs,
    GAParams,
    AlgorithmConfig,
    BenchmarkConfig,
    ProblemConfig,
)
from src.benchmarking_v2.checkpoint_io import (
    CheckpointManager,
    save_checkpoint_atomic,
    load_checkpoint,
    is_valid_checkpoint,
    can_resume_checkpoint,
    get_remaining_repetitions,
)
from src.benchmarking_v2.algorithm_runner import (
    adaptive_generations,
    run_single_algorithm,
)
from src.benchmarking_v2.problem_statistics import perform_statistical_analysis
from src.benchmarking_v2.aggregate_statistics import (
    perform_cross_problem_analysis_stratified,
)
from src.benchmarking_v2.report_generator import generate_result_tables


# =============================================================================
# Helper Functions
# =============================================================================


def _log_benchmark_configuration(
    args: Namespace,
    selected_algorithms: List[str],
    problem_configs: List[ProblemConfig],
    ga_params: GAParams,
    benchmark_config: BenchmarkConfig,
) -> None:
    """
    Log benchmark configuration at startup.

    Args:
        args: Command-line arguments
        selected_algorithms: List of algorithm names to run
        problem_configs: List of ProblemConfig instances
        ga_params: GA parameter configuration
        benchmark_config: Benchmark execution configuration
    """
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
    logging.info(f"  Early stop patience: Adaptive (2×√n generations)")
    logging.info("")


def _determine_problem_algorithms(
    problem_size: int,
    selected_algorithms: List[str],
    algorithm_configs: Dict[str, AlgorithmConfig],
    cpu_threshold: int,
) -> List[str]:
    """
    Determine which algorithms to run for a problem based on size.

    For small problems (n ≤ threshold): Run all selected algorithms
    For large problems (n > threshold): Run GPU algorithms only

    Args:
        problem_size: Number of cities in problem
        selected_algorithms: List of algorithm names from CLI
        algorithm_configs: Dictionary of algorithm configurations
        cpu_threshold: Size threshold for CPU execution

    Returns:
        List of algorithm names to run for this problem
    """
    if problem_size <= cpu_threshold:
        return selected_algorithms
    else:
        return [alg for alg in selected_algorithms if algorithm_configs[alg].use_gpu]


def _save_problem_statistics_file(
    problem_name: str,
    problem_size: int,
    optimal_cost: float,
    problem_results: Dict[str, Dict[str, Any]],
    statistical_test_results: Dict[str, Any],
    checkpoint_manager: CheckpointManager,
) -> None:
    """
    Save per-problem statistics to JSON file.

    Args:
        problem_name: Name of problem (e.g., "berlin52")
        problem_size: Number of cities
        optimal_cost: Known optimal solution cost
        problem_results: Results for all algorithms on this problem
        statistical_test_results: Statistical analysis results
        checkpoint_manager: CheckpointManager instance for path resolution
    """
    stats_path = checkpoint_manager.get_problem_stats_path(problem_name)

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
            # Handle both checkpoint format (raw_*) and direct format
            times = np.array(results.get("raw_times", results.get("times", [])))
            initial_costs = np.array(
                results.get("raw_initial_costs", results.get("initial_costs", []))
            )
            costs = np.array(results.get("raw_costs", results.get("final_costs", [])))
            gaps = np.array(results.get("raw_gaps", results.get("gaps", [])))
            generations = np.array(
                results.get("raw_generations", results.get("generations", []))
            )

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
                "success_rate": float(np.sum(valid_mask) / len(times))
                if len(times) > 0
                else 0.0,
                "backend": results.get("backend", "unknown"),
                "algorithm_config": results.get("algorithm_config", {}),
                "strategies": results.get("strategies", {}),
            }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(problem_stats, f, indent=2)
        logging.info(f"✓ Problem statistics saved: {stats_path.name}\n")
    except Exception as e:
        logging.warning(f"Failed to save problem statistics: {e}\n")


# =============================================================================
# Main Benchmark Function
# =============================================================================


def run_comprehensive_benchmark(
    args: Namespace,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run comprehensive benchmark on all configured problems and algorithms.

    This is the main orchestration function that coordinates all v2 modules:
    1. Load configurations from JSON files
    2. Setup checkpoint system
    3. Execute main benchmark loop (Problems × Algorithms)
    4. Perform per-problem statistical analysis
    5. Perform cross-problem analysis with stratified approach
    6. Generate result tables in Markdown/LaTeX

    Args:
        args: Command-line arguments containing:
            - repetitions: Number of runs per algorithm (default: 30)
            - skip_cpu: Skip CPU variant (bool)
            - test_mode: Run on subset of problems for testing (bool)

    Returns:
        Dictionary mapping: problem_name -> algorithm_name -> metrics

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(repetitions=30, skip_cpu=False, test_mode=False)
        >>> results = run_comprehensive_benchmark(args)
        >>> print(results["berlin52"]["HybridOptimized"]["mean_gap"])
        1.23
    """
    # =========================================================================
    # Step 1: Load Configurations
    # =========================================================================

    logging.info("Loading configurations...")

    # Determine config directory path (from code/src/benchmarking_v2 -> code/benchmarks_v2/configs)
    config_dir = Path(__file__).parent.parent.parent / "benchmarks_v2" / "configs"
    configs = load_all_configs(config_dir)

    ga_params = configs["ga_params"]
    algorithm_configs = configs["algorithms"]
    benchmark_config = configs["benchmark"]
    problem_configs = configs["problems"]

    # CLI override for incremental_runs flag
    if hasattr(args, "incremental_runs") and args.incremental_runs is not None:
        logging.info(f"Applying CLI override: incremental_runs={args.incremental_runs}")
        benchmark_config.incremental_runs = args.incremental_runs

    # Filter problems if test mode
    if args.test_mode:
        problem_configs = problem_configs[:2]
        logging.info("=" * 80)
        logging.info("⚠️  TEST MODE ENABLED")
        logging.info("=" * 80)
        logging.info(
            f"Running on first 2 problems only ({problem_configs[0].name}, {problem_configs[1].name})"
        )
        logging.info("Use without --test-mode for full benchmark")
        logging.info("=" * 80)
        logging.info("")

    # Determine which algorithms to run
    selected_algorithms = []
    for alg_name in algorithm_configs.keys():
        if args.skip_cpu and alg_name == "CPU":
            continue
        selected_algorithms.append(alg_name)

    _log_benchmark_configuration(
        args, selected_algorithms, problem_configs, ga_params, benchmark_config
    )

    # =========================================================================
    # Step 2: Setup Checkpoint System
    # =========================================================================

    checkpoint_manager = CheckpointManager()

    # Check for existing progress
    completed, total = checkpoint_manager.count_completed_checkpoints(
        [p.__dict__ for p in problem_configs],
        selected_algorithms,
        args.repetitions,
        benchmark_config.cpu_size_threshold,
    )
    if completed > 0:
        logging.info(f"\n✓ Found {completed}/{total} completed checkpoints")
        logging.info("  Resuming from previous run...\n")

    # Database path (from code/src/benchmarking_v2 -> project root is 3 levels up)
    db_path = Path(__file__).parent.parent.parent.parent / "datasets" / "routing.duckdb"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Statistical analyzer
    analyzer = StatisticalAnalyzer()

    # =========================================================================
    # Step 3: Main Benchmark Loop (Problems × Algorithms)
    # =========================================================================

    # Results storage: problem -> algorithm -> metrics
    problem_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for prob_idx, problem_config in enumerate(problem_configs, 1):
        problem_name = problem_config.name
        optimal_cost = problem_config.optimal
        problem_size = problem_config.size

        logging.info("\n" + "=" * 80)
        logging.info(f"PROBLEM {prob_idx}/{len(problem_configs)}: {problem_name}")
        logging.info(f"Size: {problem_size} cities | Optimal: {optimal_cost}")
        logging.info("=" * 80)

        # Determine algorithms for this problem
        problem_algorithms = _determine_problem_algorithms(
            problem_size,
            selected_algorithms,
            algorithm_configs,
            benchmark_config.cpu_size_threshold,
        )

        if problem_size <= benchmark_config.cpu_size_threshold:
            logging.info(
                f"Running {len(problem_algorithms)} algorithm(s) "
                f"(n≤{benchmark_config.cpu_size_threshold})\n"
            )
        else:
            logging.info(
                f"Running {len(problem_algorithms)} GPU algorithms only "
                f"(n>{benchmark_config.cpu_size_threshold})\n"
            )

        # Initialize results for this problem
        problem_results[problem_name] = {}

        # Load problem once for all algorithms
        try:
            with DatabaseLoader(str(db_path)) as loader:
                problem = loader.load(problem_name)
        except Exception as e:
            logging.error(f"Failed to load {problem_name}: {e}")
            continue

        max_generations = adaptive_generations(problem_size)
        customers = list(range(problem_size))

        # Run each algorithm on this problem
        for algo_idx, alg_name in enumerate(problem_algorithms, 1):
            alg_config = algorithm_configs[alg_name]
            backend_type = "GPU" if alg_config.use_gpu else "CPU"

            # Check for existing checkpoint (complete or partial)
            checkpoint_path = checkpoint_manager.get_checkpoint_path(
                problem_name, alg_name
            )
            existing_checkpoint = None
            incremental_mode = benchmark_config.incremental_runs

            # Check if checkpoint exists
            checkpoint_exists = checkpoint_path.exists()

            if checkpoint_exists:
                existing_checkpoint = load_checkpoint(checkpoint_path)
                existing_runs = existing_checkpoint.get("successful_runs", 0)

                if incremental_mode:
                    # ADDITIVE INCREMENT: Add new reps to existing total
                    new_reps_to_add = args.repetitions
                    total_target_reps = existing_runs + new_reps_to_add

                    logging.info(
                        f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
                        f"✓ Resuming from checkpoint ({existing_runs} runs) - "
                        f"adding {new_reps_to_add} more runs → {total_target_reps} total"
                    )
                    actual_repetitions = new_reps_to_add
                else:
                    # REUSE MODE: Skip if target already met
                    if existing_runs >= args.repetitions:
                        logging.info(
                            f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
                            f"✓ Loading from checkpoint (complete)"
                        )
                        problem_results[problem_name][alg_name] = existing_checkpoint

                        # CRITICAL: Cleanup GPU memory even when loading from checkpoint
                        if alg_config.use_gpu and CUPY_AVAILABLE:
                            cp.get_default_memory_pool().free_all_blocks()

                        continue
                    else:
                        # Need to complete to target - treat as incremental internally
                        actual_repetitions = args.repetitions - existing_runs
                        logging.info(
                            f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
                            f"Resuming: {existing_runs}/{args.repetitions} runs exist, "
                            f"completing remaining {actual_repetitions}"
                        )
                        # Enable incremental mode internally to merge data
                        incremental_mode = True
            else:
                # Fresh start - no checkpoint exists
                actual_repetitions = args.repetitions
                logging.info(
                    f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
                    f"Running {actual_repetitions} repetitions..."
                )

            # Create algorithm instance
            algorithm_class = alg_config.get_class()
            algorithm_instance = algorithm_class(
                population_size=ga_params.population_size,
                mutation_rate=ga_params.mutation_rate,
                tournament_size=ga_params.tournament_size,
                two_opt_iterations=ga_params.two_opt_iterations,
                seed=ga_params.seed,
            )

            # Run benchmark
            results = run_single_algorithm(
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

            # Enhance checkpoint with metadata
            from datetime import datetime

            results["problem_name"] = problem_name
            results["problem_size"] = problem_size
            results["optimal_cost"] = optimal_cost
            results["max_generations_configured"] = max_generations
            results["backend"] = "GPU" if alg_config.use_gpu else "CPU"
            results["timestamp"] = datetime.now().isoformat()

            # Extract algorithm configuration
            results["algorithm_config"] = {
                "population_size": getattr(
                    algorithm_instance, "population_size", ga_params.population_size
                ),
                "mutation_rate": getattr(
                    algorithm_instance, "mutation_rate", ga_params.mutation_rate
                ),
                "tournament_size": getattr(
                    algorithm_instance, "tournament_size", ga_params.tournament_size
                ),
                "two_opt_iterations": getattr(
                    algorithm_instance,
                    "two_opt_iterations",
                    ga_params.two_opt_iterations,
                ),
                "seed": getattr(algorithm_instance, "seed", ga_params.seed),
            }

            # Strategy information
            results["strategies"] = {
                "selection": getattr(
                    algorithm_instance.selection,
                    "__class__.__name__",
                    "TournamentSelection",
                )
                if hasattr(algorithm_instance, "selection")
                else "TournamentSelection",
                "crossover": getattr(
                    algorithm_instance.crossover, "__class__.__name__", "OrderCrossover"
                )
                if hasattr(algorithm_instance, "crossover")
                else "OrderCrossover",
                "mutation": getattr(
                    algorithm_instance.mutation, "__class__.__name__", "SwapMutation"
                )
                if hasattr(algorithm_instance, "mutation")
                else "SwapMutation",
            }

            results["construction_heuristic"] = "random_initialization"

            # Save checkpoint immediately (fault tolerance)
            checkpoint_path = checkpoint_manager.get_checkpoint_path(
                problem_name, alg_name
            )
            save_checkpoint_atomic(checkpoint_path, results)
            problem_results[problem_name][alg_name] = results

            # Cleanup algorithm instance
            del algorithm_instance
            gc.collect()
            if alg_config.use_gpu and CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

        # =====================================================================
        # Step 4: Per-Problem Statistical Analysis
        # =====================================================================

        logging.info(f"\n{'─' * 80}")
        logging.info(f"STATISTICS FOR {problem_name}")
        logging.info(f"{'─' * 80}")

        statistical_test_results = perform_statistical_analysis(
            problem_name=problem_name,
            optimal_cost=optimal_cost,
            results=problem_results[problem_name],
            analyzer=analyzer,
        )

        # Save problem statistics to file
        _save_problem_statistics_file(
            problem_name,
            problem_size,
            optimal_cost,
            problem_results[problem_name],
            statistical_test_results,
            checkpoint_manager,
        )

        # Cleanup problem
        del problem
        gc.collect()

    # =========================================================================
    # Step 5: Cross-Problem Analysis (Stratified)
    # =========================================================================

    logging.info("\n" + "=" * 80)
    logging.info("CROSS-PROBLEM ANALYSIS")
    logging.info("=" * 80)

    if len(problem_results) > 1:
        cross_problem_stats = perform_cross_problem_analysis_stratified(
            all_results=problem_results,
            problem_configs=problem_configs,
            analyzer=analyzer,
            cpu_threshold=benchmark_config.cpu_size_threshold,
        )

    # =========================================================================
    # Step 6: Generate Result Tables
    # =========================================================================

    tables_dir = Path("results_v3/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_result_tables(
        all_results=problem_results,
        problem_configs=problem_configs,
        output_dir=tables_dir,
        benchmark_name="chapter4_validation",
    )

    logging.info("=" * 80)
    logging.info("BENCHMARK COMPLETE")
    logging.info("=" * 80)

    return problem_results
