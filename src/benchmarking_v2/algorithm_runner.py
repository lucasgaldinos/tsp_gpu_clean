"""
Algorithm Runner Module for Benchmark V2
=========================================

Executes algorithm instances with proper resource management.

This module provides the core execution logic for running genetic algorithms
on TSP problems, including:
- Adaptive generation calculation (2×n×√n formula)
- Single algorithm execution with error handling
- GPU memory cleanup
- Comprehensive metrics collection

Extracted from chapter4_validation.py lines 346-600.

Usage:
    from src.benchmarking_v2.algorithm_runner import (
        adaptive_generations, run_single_algorithm
    )

    # Calculate generations for problem
    max_gens = adaptive_generations(problem_size=52)

    # Run algorithm
    results = run_single_algorithm(
        algorithm_name="HybridOptimized",
        algorithm_instance=algorithm,
        problem=problem,
        problem_name="berlin52",
        problem_size=52,
        customers=list(range(52)),
        max_generations=max_gens,
        optimal_cost=7542.0,
        repetitions=30,
        use_gpu=True
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import gc
import logging
import math
import time
from typing import Dict, Any, List, Optional

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from src.protocols.problem_context import ProblemContext


# =============================================================================
# Generation Calculation
# =============================================================================


def adaptive_generations(n: int) -> int:
    """
    Calculate adaptive generations using formula: 2 × n × sqrt(n).

    This formula balances convergence time with solution quality across
    different problem sizes. Larger problems get proportionally more
    generations to explore the search space.

    Args:
        n: Problem size (number of cities)

    Returns:
        Number of generations to run

    Example:
        >>> adaptive_generations(52)
        748
        >>> adaptive_generations(100)
        2000
        >>> adaptive_generations(318)
        11324
    """
    return int(2 * n * math.sqrt(n))


def adaptive_patience(n: int) -> int:
    """
    Calculate adaptive patience (stagnation window) using formula: 2 × sqrt(n).

    Patience scales with problem complexity to avoid:
    - Small problems: Excessive patience window (was 50 for n=52)
    - Large problems: Insufficient patience (was 50 for n=1002)

    This adaptive approach reduces outlier effects where algorithms waste time
    in patience window after finding optimal solution early. The patience is defined empirically (kind of)

    Args:
        n: Problem size (number of cities)

    Returns:
        Patience value (generations without improvement before stopping)

    Example:
        >>> adaptive_patience(52)
        14
        >>> adaptive_patience(100)
        20
        >>> adaptive_patience(318)
        36
        >>> adaptive_patience(1002)
        63
    """
    return int(2 * math.sqrt(n))


# =============================================================================
# Algorithm Execution
# =============================================================================


def run_single_algorithm(
    algorithm_name: str,
    algorithm_instance,
    problem,
    problem_name: str,
    problem_size: int,
    customers: List[int],
    max_generations: int,
    optimal_cost: float,
    repetitions: int,
    use_gpu: bool,
    patience: Optional[int] = None,
    seed_base: int = 42,
    existing_checkpoint: Dict[str, Any] = None,
    incremental_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run single algorithm for specified repetitions.

    Executes the algorithm multiple times with different seeds, collecting
    comprehensive metrics for statistical analysis.

    Supports incremental execution: if existing_checkpoint is provided and
    incremental_mode=True, appends new runs to existing data and recalculates
    statistics.

    Args:
        algorithm_name: Name of algorithm variant (e.g., "HybridOptimized")
        algorithm_instance: Algorithm instance (must have evolve() method)
        problem: Problem object from DatabaseLoader
        problem_name: Problem identifier (e.g., "berlin52")
        problem_size: Number of cities
        customers: Customer indices to visit
        max_generations: Maximum number of generations
        optimal_cost: Known optimal solution cost
        repetitions: Number of NEW runs to perform (appended if incremental)
        use_gpu: Whether to use GPU backend
        patience: Early stopping patience (default: 50)
        seed_base: Base seed for reproducibility (default: 42, ignored if incremental)
        existing_checkpoint: Existing checkpoint data (for incremental mode)
        incremental_mode: If True, append to existing checkpoint instead of replacing

    Returns:
        Dictionary with aggregated results containing:
            - Summary statistics (mean, std, min, max)
            - Raw arrays for statistical tests
            - Convergence metrics
            - Memory transfer metrics (GPU only)
            - Seed list (for reproducibility)

    Example:
        >>> # Fresh run
        >>> results = run_single_algorithm(
        ...     algorithm_name="CPU",
        ...     algorithm_instance=algorithm,
        ...     problem=problem,
        ...     problem_name="berlin52",
        ...     problem_size=52,
        ...     customers=list(range(52)),
        ...     max_generations=748,
        ...     optimal_cost=7542.0,
        ...     repetitions=30,
        ...     use_gpu=False
        ... )

        >>> # Incremental run (append 10 more)
        >>> existing = load_checkpoint(path)
        >>> results = run_single_algorithm(
        ...     algorithm_name="CPU",
        ...     algorithm_instance=algorithm,
        ...     problem=problem,
        ...     problem_name="berlin52",
        ...     problem_size=52,
        ...     customers=list(range(52)),
        ...     max_generations=748,
        ...     optimal_cost=7542.0,
        ...     repetitions=10,  # 10 NEW runs
        ...     use_gpu=False,
        ...     existing_checkpoint=existing,
        ...     incremental_mode=True
        ... )
        >>> print(f"Total runs: {results['successful_runs']}")  # 40 total
    """
    backend = "GPU" if use_gpu else "CPU"
    
    # Calculate adaptive patience if not provided: 2 × sqrt(n)
    if patience is None:
        patience = adaptive_patience(problem_size)

    # Handle incremental mode
    if incremental_mode and existing_checkpoint:
        existing_runs = existing_checkpoint["successful_runs"]
        total_runs_after = existing_runs + repetitions
        logging.info(
            f"\n  [{algorithm_name}] INCREMENTAL MODE: Extending {problem_name} "
            f"from {existing_runs} to {total_runs_after} runs"
        )
        logging.info(
            f"    Backend: {backend} | Early stopping: patience={patience} (adaptive: 2×√{problem_size}) | "
            f"New repetitions: {repetitions}"
        )

        # Load existing data
        results = {
            "times": existing_checkpoint.get("raw_times", []).copy(),
            "initial_costs": existing_checkpoint.get("raw_initial_costs", []).copy(),
            "final_costs": existing_checkpoint.get("raw_costs", []).copy(),
            "gaps": existing_checkpoint.get("raw_gaps", []).copy(),
            "improvements": [],  # Recalculate from scratch
            "generations": existing_checkpoint.get("raw_generations", []).copy(),
            "stop_reasons": existing_checkpoint.get("raw_stop_reasons", []).copy(),
            "h2d_bytes": [],
            "d2h_bytes": [],
            "kernel_launches": [],
            "best_tours": [],
            "seeds": existing_checkpoint.get(
                "raw_seeds", []
            ).copy(),  # Load existing seeds
        }
    else:
        logging.info(
            f"\n  [{algorithm_name}] Processing {problem_name} "
            f"(n={problem_size}, optimal={optimal_cost:.0f})"
        )
        logging.info(
            f"    Backend: {backend} | Early stopping: {patience} gen patience | "
            f"Repetitions: {repetitions}"
        )

        results = {
            "times": [],
            "initial_costs": [],
            "final_costs": [],
            "gaps": [],
            "improvements": [],
            "generations": [],
            "stop_reasons": [],
            "h2d_bytes": [],
            "d2h_bytes": [],
            "kernel_launches": [],
            "best_tours": [],
            "seeds": [],  # Track seeds for reproducibility
        }

    for rep in range(repetitions):
        # Generate random seed for this run (not sequential)
        run_seed = np.random.randint(0, 2**31 - 1)
        results["seeds"].append(run_seed)

        # Create fresh context for each run with random seed
        xp = cp if use_gpu and CUPY_AVAILABLE else np
        context = ProblemContext(problem, xp=xp, seed=run_seed)

        # Log repetition start (adjusted for incremental mode)
        if incremental_mode and existing_checkpoint:
            current_rep = existing_checkpoint["successful_runs"] + rep + 1
            total_target = existing_checkpoint["successful_runs"] + repetitions
            logging.info(
                f"    Run {current_rep}/{total_target} (new #{rep + 1}/{repetitions}, seed={run_seed})..."
            )
        else:
            logging.info(f"    Repetition {rep + 1}/{repetitions} (seed={run_seed})...")

        # Run algorithm
        start_time = time.perf_counter()
        try:
            tour, stats = algorithm_instance.evolve(
                context,
                customers,
                max_generations,
                optimal_cost=optimal_cost,
                patience=patience,
            )
            elapsed = time.perf_counter() - start_time

            # Extract metrics
            initial_cost = stats["initial_fitness"]
            final_cost = stats["best_fitness"]
            gap = (final_cost - optimal_cost) / optimal_cost * 100
            improvement = (initial_cost - final_cost) / initial_cost * 100
            generations = stats.get("generations_completed", 0)
            stop_reason = stats.get("stop_reason", "completed")

            results["times"].append(elapsed)
            results["initial_costs"].append(initial_cost)
            results["final_costs"].append(final_cost)
            results["gaps"].append(gap)
            results["improvements"].append(improvement)
            results["generations"].append(generations)
            results["stop_reasons"].append(stop_reason)
            results["h2d_bytes"].append(stats.get("h2d_bytes", 0))
            results["d2h_bytes"].append(stats.get("d2h_bytes", 0))
            results["kernel_launches"].append(stats.get("kernel_launches", 0))
            results["best_tours"].append(tour)

            # Log completion with key metrics
            logging.info(
                f"      → Completed in {elapsed:.2f}s | "
                f"Cost: {final_cost:.0f} (gap: {gap:.2f}%) | "
                f"Generations: {generations}/{max_generations}"
            )

        except Exception as e:
            logging.error(f"    Rep {rep + 1} FAILED: {e}")
            import traceback

            traceback.print_exc()

            # CRITICAL: Append to ALL lists to maintain synchronization
            results["times"].append(np.nan)
            results["initial_costs"].append(np.nan)
            results["final_costs"].append(np.nan)
            results["gaps"].append(np.nan)
            results["improvements"].append(np.nan)
            results["generations"].append(0)
            results["stop_reasons"].append("error")
            results["h2d_bytes"].append(0)
            results["d2h_bytes"].append(0)
            results["kernel_launches"].append(0)
            results["best_tours"].append(None)

    # Calculate statistics (from combined data if incremental)
    times = np.array(results["times"])
    initial_costs_arr = np.array(results["initial_costs"])
    final_costs = np.array(results["final_costs"])
    gaps = np.array(results["gaps"])
    generations_arr = np.array(results["generations"])
    seeds_arr = np.array(results["seeds"])

    # Recalculate improvements from scratch
    improvements = (initial_costs_arr - final_costs) / initial_costs_arr * 100

    h2d = (
        np.array(results["h2d_bytes"]) if results["h2d_bytes"] else np.zeros(len(times))
    )
    d2h = (
        np.array(results["d2h_bytes"]) if results["d2h_bytes"] else np.zeros(len(times))
    )

    # Remove NaN values for valid statistics
    valid_mask = ~np.isnan(times)
    total_successful = int(np.sum(valid_mask))

    summary = {
        "algorithm": algorithm_name,
        "repetitions": len(times),  # Total runs (existing + new if incremental)
        "successful_runs": total_successful,
        # Time statistics
        "mean_time": float(np.nanmean(times)),
        "std_time": float(np.nanstd(times)),
        "min_time": float(np.nanmin(times)),
        "max_time": float(np.nanmax(times)),
        # Solution quality
        "mean_cost": float(np.nanmean(final_costs)),
        "std_cost": float(np.nanstd(final_costs)),
        "best_cost": float(np.nanmin(final_costs)),
        "worst_cost": float(np.nanmax(final_costs)),
        # Initial costs
        "mean_initial_cost": float(np.nanmean(initial_costs_arr)),
        "std_initial_cost": float(np.nanstd(initial_costs_arr)),
        # Gap to optimal
        "mean_gap": float(np.nanmean(gaps)),
        "std_gap": float(np.nanstd(gaps)),
        "best_gap": float(np.nanmin(gaps)),
        # Improvement from initial
        "mean_improvement": float(np.nanmean(improvements)),
        # Convergence metrics
        "mean_generations": float(np.nanmean(generations_arr)),
        "std_generations": float(np.nanstd(generations_arr)),
        "min_generations": float(np.nanmin(generations_arr)),
        "max_generations": float(np.nanmax(generations_arr)),
        # Memory transfers (GPU only)
        "mean_h2d_mb": float(np.nanmean(h2d)) / 1e6 if len(h2d) > 0 else 0.0,
        "mean_d2h_mb": float(np.nanmean(d2h)) / 1e6 if len(d2h) > 0 else 0.0,
        "total_transfer_mb": (float(np.nanmean(h2d)) + float(np.nanmean(d2h))) / 1e6
        if len(h2d) > 0
        else 0.0,
        # Kernel launches (GPU only)
        "mean_kernels": float(np.nanmean(results["kernel_launches"]))
        if results["kernel_launches"]
        else 0.0,
        # Raw data for statistical tests
        "raw_times": times[valid_mask].tolist(),
        "raw_initial_costs": initial_costs_arr[valid_mask].tolist(),
        "raw_costs": final_costs[valid_mask].tolist(),
        "raw_gaps": gaps[valid_mask].tolist(),
        "raw_generations": generations_arr[valid_mask].tolist(),
        "raw_stop_reasons": [
            results["stop_reasons"][i] for i in range(len(valid_mask)) if valid_mask[i]
        ]
        if results["stop_reasons"]
        else [],
        "raw_seeds": seeds_arr[
            valid_mask
        ].tolist(),  # NEW: seed list for reproducibility
    }

    if incremental_mode and existing_checkpoint:
        logging.info(
            f"\n    ✓ {algorithm_name} INCREMENTAL summary for {problem_name} "
            f"({total_successful} total runs): "
            f"Cost={summary['mean_cost']:.0f}±{summary['std_cost']:.1f} | "
            f"Gap={summary['mean_gap']:.2f}%±{summary['std_gap']:.2f}% | "
            f"Time={summary['mean_time']:.2f}s±{summary['std_time']:.2f}s"
        )
    else:
        logging.info(
            f"\n    ✓ {algorithm_name} summary for {problem_name}: "
            f"Cost={summary['mean_cost']:.0f}±{summary['std_cost']:.1f} | "
            f"Gap={summary['mean_gap']:.2f}%±{summary['std_gap']:.2f}% | "
            f"Time={summary['mean_time']:.2f}s±{summary['std_time']:.2f}s"
        )

    return summary


# =============================================================================
# GPU Memory Management
# =============================================================================


def cleanup_gpu_memory() -> None:
    """
    Free all GPU memory blocks.

    Should be called after each algorithm run to prevent memory leaks
    when running multiple benchmarks sequentially.

    Example:
        >>> # After running GPU algorithm
        >>> cleanup_gpu_memory()
    """
    if CUPY_AVAILABLE and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        logging.debug("GPU memory pool freed")


def cleanup_resources(use_gpu: bool = False) -> None:
    """
    Clean up algorithm resources and memory.

    Performs garbage collection and GPU memory cleanup if applicable.

    Args:
        use_gpu: Whether GPU was used (triggers GPU cleanup)

    Example:
        >>> # After algorithm execution
        >>> cleanup_resources(use_gpu=True)
    """
    gc.collect()
    if use_gpu:
        cleanup_gpu_memory()
