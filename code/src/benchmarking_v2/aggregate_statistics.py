"""
Aggregate Statistics Module for Benchmark V2
=============================================

Performs cross-problem statistical analysis with stratified approach.

This module fixes the critical bug in the original cross-problem analysis
where CPU (12 problems) and GPU (38 problems) results were compared directly,
causing ValueError in Friedman test due to unpaired data.

**BUG FIX**: Stratified analysis approach:
- Stratum 1: Small problems (n<100) with ALL algorithms (CPU + GPU)
- Stratum 2: All problems (n≤1002) with GPU-only algorithms

This maintains maximum statistical power while avoiding pairing issues.

Original bug location: chapter4_validation.py line 984
Error: ValueError: All data_sets must have the same number of observations

Usage:
    from src.benchmarking_v2.aggregate_statistics import (
        perform_cross_problem_analysis_stratified
    )
    from src.benchmarking_v2.statistics import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    all_results = {
        "berlin52": {"CPU": {...}, "HybridOptimized": {...}},
        "eil51": {"CPU": {...}, "HybridOptimized": {...}},
        ...
    }

    stats = perform_cross_problem_analysis_stratified(
        all_results=all_results,
        problem_configs=problem_configs,
        analyzer=analyzer,
        cpu_threshold=100
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import logging
from typing import Dict, Any, List

import numpy as np

from src.benchmarking_v2.statistics import StatisticalAnalyzer
from src.benchmarking_v2.config_loader import ProblemConfig


# =============================================================================
# Helper Functions
# =============================================================================


def _get_problem_size_map(problem_configs: List[ProblemConfig]) -> Dict[str, int]:
    """
    Create mapping from problem name to size.

    Args:
        problem_configs: List of ProblemConfig instances

    Returns:
        Dictionary mapping problem_name -> size
    """
    return {p.name: p.size for p in problem_configs}


def _extract_algorithm_gaps(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    problem_names: List[str],
    algorithm_name: str,
) -> List[float]:
    """
    Extract mean gaps for an algorithm across specified problems.

    Args:
        all_results: Nested dict: problem -> algorithm -> metrics
        problem_names: List of problems to extract from
        algorithm_name: Algorithm to extract gaps for

    Returns:
        List of mean gap values
    """
    gaps = []
    for problem in problem_names:
        if algorithm_name in all_results[problem]:
            gaps.append(all_results[problem][algorithm_name]["mean_gap"])
    return gaps


# =============================================================================
# Stratified Cross-Problem Analysis (BUG FIX)
# =============================================================================


def perform_cross_problem_analysis_stratified(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    problem_configs: List[ProblemConfig],
    analyzer: StatisticalAnalyzer,
    cpu_threshold: int = 100,
    separator: str = "=" * 80,
) -> Dict[str, Any]:
    """
    Perform stratified cross-problem statistical analysis.

    **CRITICAL BUG FIX**: This function replaces the monolithic cross-problem
    analysis that failed with:

        ValueError: All data_sets must have the same number of observations.
        Got lengths: [12, 38, 38, 38]

    **Root Cause**: CPU ran on 12 small problems (n<100), GPU ran on all 38.

    **Solution**: Stratify analysis into two independent comparisons:
    1. Small problems (n<100): Compare CPU + all GPU algorithms (12 problems, 4 algorithms)
    2. All problems: Compare GPU-only algorithms (38 problems, 3 algorithms)

    This maintains maximum statistical power for each stratum while avoiding
    the pairing issue.

    Args:
        all_results: Nested dictionary: problem_name -> algorithm_name -> metrics
        problem_configs: List of ProblemConfig instances with name, size, optimal
        analyzer: StatisticalAnalyzer instance
        cpu_threshold: Size threshold for CPU runs (default: 100)
        separator: Separator line for logging (default: 80 equals signs)

    Returns:
        Dictionary containing:
            - stratum1_small_problems: Results for small problems with CPU
            - stratum2_all_gpu_problems: Results for all problems with GPU only
            - summary_statistics: Aggregate metrics across strata

    Example:
        >>> stats = perform_cross_problem_analysis_stratified(
        ...     all_results, problem_configs, analyzer
        ... )
        >>> print(stats["stratum1_small_problems"]["friedman_test"]["p_value"])
        >>> print(stats["stratum2_all_gpu_problems"]["friedman_test"]["p_value"])
    """
    logging.info(separator)
    logging.info("CROSS-PROBLEM ANALYSIS (STRATIFIED APPROACH)")
    logging.info(separator)
    logging.info("")
    logging.info(
        "Using stratified analysis to handle CPU/GPU problem coverage difference:"
    )
    logging.info(f"  - CPU runs: Problems with n < {cpu_threshold}")
    logging.info("  - GPU runs: All problems")
    logging.info("")

    # Build problem size map
    problem_sizes = _get_problem_size_map(problem_configs)
    problem_names = list(all_results.keys())

    # Identify all algorithms present
    all_algorithm_names = set()
    for problem_results in all_results.values():
        all_algorithm_names.update(problem_results.keys())
    all_algorithm_names = sorted(all_algorithm_names)

    logging.info(f"Total problems analyzed: {len(problem_names)}")
    logging.info(f"Algorithms present: {', '.join(all_algorithm_names)}")
    logging.info("")

    # =========================================================================
    # STRATUM 1: Small Problems (n < cpu_threshold) with ALL Algorithms
    # =========================================================================

    small_problems = [p for p in problem_names if problem_sizes[p] < cpu_threshold]
    small_problem_algos = [
        a
        for a in all_algorithm_names
        if all(a in all_results[p] for p in small_problems)
    ]

    stratum1_results = None
    if len(small_problem_algos) >= 3 and len(small_problems) >= 3:
        logging.info(separator)
        logging.info(f"STRATUM 1: Small Problems (n < {cpu_threshold})")
        logging.info(separator)
        logging.info(
            f"Problems: {len(small_problems)} ({', '.join(sorted(small_problems)[:5])}{'...' if len(small_problems) > 5 else ''})"
        )
        logging.info(
            f"Algorithms: {len(small_problem_algos)} ({', '.join(small_problem_algos)})"
        )
        logging.info("")

        # Collect gap data for each algorithm
        small_data = []
        for algo in small_problem_algos:
            algo_gaps = _extract_algorithm_gaps(all_results, small_problems, algo)
            small_data.append(np.array(algo_gaps))
            logging.info(
                f"  {algo}: mean_gap={np.mean(algo_gaps):.2f}% ± {np.std(algo_gaps):.2f}%"
            )

        logging.info("")

        # Friedman test
        friedman_result = analyzer.friedman_test(small_data)
        logging.info("Friedman Test:")
        logging.info(f"  Statistic: {friedman_result['statistic']:.4f}")
        logging.info(f"  p-value: {friedman_result['p_value']:.6f}")
        logging.info(f"  Significant: {friedman_result['significant']}")

        stratum1_results = {
            "problems": small_problems,
            "algorithms": small_problem_algos,
            "n_problems": len(small_problems),
            "n_algorithms": len(small_problem_algos),
            "friedman_test": friedman_result,
        }

        # Nemenyi post-hoc if significant
        if friedman_result["post_hoc_required"]:
            logging.info("\nNemenyi Post-Hoc Test:")
            nemenyi_result = analyzer.nemenyi_posthoc(small_data, small_problem_algos)
            stratum1_results["nemenyi_posthoc"] = nemenyi_result

            logging.info(
                f"  Critical distance: {nemenyi_result['critical_distance']:.4f}"
            )

            if nemenyi_result["significant_pairs"]:
                logging.info("  Significant pairwise differences:")
                for i, j, p_val in nemenyi_result["significant_pairs"]:
                    name_i = nemenyi_result["algorithm_names"][i]
                    name_j = nemenyi_result["algorithm_names"][j]
                    logging.info(f"    {name_i} vs {name_j}: p={p_val:.4f}")
            else:
                logging.info("  No significant pairwise differences detected")

        logging.info("")
    else:
        logging.info(f"STRATUM 1: Skipped (need ≥3 algorithms and ≥3 problems)")
        logging.info(
            f"  Found: {len(small_problem_algos)} algorithms, {len(small_problems)} problems"
        )
        logging.info("")

    # =========================================================================
    # STRATUM 2: All Problems with GPU-Only Algorithms
    # =========================================================================

    gpu_algos = [a for a in all_algorithm_names if a != "CPU"]
    all_gpu_problems = [
        p for p in problem_names if all(a in all_results[p] for a in gpu_algos)
    ]

    stratum2_results = None
    if len(gpu_algos) >= 3 and len(all_gpu_problems) >= 3:
        logging.info(separator)
        logging.info("STRATUM 2: All Problems (GPU Algorithms Only)")
        logging.info(separator)

        size_range = [problem_sizes[p] for p in all_gpu_problems]
        logging.info(
            f"Problems: {len(all_gpu_problems)} "
            f"(n={min(size_range)} to {max(size_range)})"
        )
        logging.info(f"Algorithms: {len(gpu_algos)} ({', '.join(gpu_algos)})")
        logging.info("")

        # Collect gap data
        all_data = []
        for algo in gpu_algos:
            algo_gaps = _extract_algorithm_gaps(all_results, all_gpu_problems, algo)
            all_data.append(np.array(algo_gaps))
            logging.info(
                f"  {algo}: mean_gap={np.mean(algo_gaps):.2f}% ± {np.std(algo_gaps):.2f}%"
            )

        logging.info("")

        # Calculate geometric mean speedup vs baseline
        logging.info("Geometric mean speedup (vs HybridNaive baseline):")
        baseline_algo = "HybridNaive" if "HybridNaive" in gpu_algos else gpu_algos[0]
        for algo in gpu_algos:
            if algo == baseline_algo:
                logging.info(f"  {algo}: 1.00x (baseline)")
            else:
                speedups = []
                for problem in all_gpu_problems:
                    baseline_time = all_results[problem][baseline_algo]["mean_time"]
                    algo_time = all_results[problem][algo]["mean_time"]
                    if algo_time > 0:
                        speedups.append(baseline_time / algo_time)

                if speedups:
                    geomean_speedup = np.exp(np.mean(np.log(speedups)))
                    logging.info(f"  {algo}: {geomean_speedup:.2f}x vs {baseline_algo}")

        logging.info("")

        # Friedman test
        friedman_result = analyzer.friedman_test(all_data)
        logging.info("Friedman Test:")
        logging.info(f"  Statistic: {friedman_result['statistic']:.4f}")
        logging.info(f"  p-value: {friedman_result['p_value']:.6f}")
        logging.info(f"  Significant: {friedman_result['significant']}")

        stratum2_results = {
            "problems": all_gpu_problems,
            "algorithms": gpu_algos,
            "n_problems": len(all_gpu_problems),
            "n_algorithms": len(gpu_algos),
            "friedman_test": friedman_result,
        }

        # Nemenyi post-hoc if significant
        if friedman_result["post_hoc_required"]:
            logging.info("\nNemenyi Post-Hoc Test:")
            nemenyi_result = analyzer.nemenyi_posthoc(all_data, gpu_algos)
            stratum2_results["nemenyi_posthoc"] = nemenyi_result

            logging.info(
                f"  Critical distance: {nemenyi_result['critical_distance']:.4f}"
            )

            if nemenyi_result["significant_pairs"]:
                logging.info("  Significant pairwise differences:")
                for i, j, p_val in nemenyi_result["significant_pairs"]:
                    name_i = nemenyi_result["algorithm_names"][i]
                    name_j = nemenyi_result["algorithm_names"][j]
                    logging.info(f"    {name_i} vs {name_j}: p={p_val:.4f}")
            else:
                logging.info("  No significant pairwise differences detected")

        logging.info("")
    else:
        logging.info(f"STRATUM 2: Skipped (need ≥3 algorithms and ≥3 problems)")
        logging.info(
            f"  Found: {len(gpu_algos)} algorithms, {len(all_gpu_problems)} problems"
        )
        logging.info("")

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    logging.info(separator)
    logging.info("CROSS-PROBLEM SUMMARY")
    logging.info(separator)

    summary_stats = {}
    for algo in all_algorithm_names:
        all_gaps = []
        for problem in problem_names:
            if algo in all_results[problem]:
                all_gaps.append(all_results[problem][algo]["mean_gap"])

        if all_gaps:
            summary_stats[algo] = {
                "mean_gap": float(np.mean(all_gaps)),
                "std_gap": float(np.std(all_gaps)),
                "n_problems": len(all_gaps),
            }
            logging.info(
                f"{algo}: mean_gap={np.mean(all_gaps):.2f}% ± {np.std(all_gaps):.2f}% "
                f"(n={len(all_gaps)} problems)"
            )

    logging.info("")

    # =========================================================================
    # Return Results
    # =========================================================================

    return {
        "stratum1_small_problems": stratum1_results,
        "stratum2_all_gpu_problems": stratum2_results,
        "summary_statistics": summary_stats,
        "cpu_threshold": cpu_threshold,
        "total_problems": len(problem_names),
        "total_algorithms": len(all_algorithm_names),
    }
