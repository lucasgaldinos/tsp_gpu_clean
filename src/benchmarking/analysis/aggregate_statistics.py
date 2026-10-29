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

Usage:
    from benchmarking.analysis.aggregate_statistics import (
        perform_cross_problem_analysis_stratified,
        compare_algorithms,
        compare_algorithms_by_group,
    )
    from benchmarking.analysis.statistics import StatisticalAnalyzer

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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.statistics import StatisticalAnalyzer

if TYPE_CHECKING:
  from benchmarking.runner.config_loader import ProblemConfig

# =============================================================================
# Helper Functions
# =============================================================================


def _get_problem_size_map(problem_configs: list[ProblemConfig]) -> dict[str, int]:
  """
  Create mapping from problem name to size.

  Args:
      problem_configs: List of ProblemConfig instances

  Returns:
      Dictionary mapping problem_name -> size
  """
  return {p.name: p.size for p in problem_configs}


def _extract_algorithm_gaps(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_name: str,
) -> list[float]:
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
# Generalized Algorithm Comparison (Decoupled from thresholds)
# =============================================================================


def compare_algorithms(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problems: list[str],
  algorithms: list[str],
  analyzer: StatisticalAnalyzer | None = None,
  baseline_algorithm: str | None = None,
  label: str = "Algorithm Comparison",
  separator: str = "=" * 80,
) -> dict[str, Any] | None:
  """
  Compare a subset of algorithms across a designated subset of problems.

  Validates that each problem has results for all requested algorithms,
  executes the Friedman test, runs Nemenyi post-hoc if p < 0.05, and computes
  geometric mean speedups if baseline_algorithm is provided.

  Args:
      all_results: Nested dictionary: problem_name -> algorithm_name -> metrics
      problems: Candidate problem names to compare across
      algorithms: Algorithm names to include in comparison
      analyzer: Optional StatisticalAnalyzer instance (instantiated if None)
      baseline_algorithm: Optional baseline for geometric speedup calculations
      label: Human-readable header label for logging
      separator: Separator line for logging

  Returns:
      Comparison dictionary or None if len(problems) < 3 or len(algorithms) < 3.
  """
  valid_problems = [
    p for p in problems
    if p in all_results and all(a in all_results[p] for a in algorithms)
  ]

  if len(algorithms) < 3 or len(valid_problems) < 3:
    logging.info(separator)
    logging.info(f"{label}: Skipped (need ≥3 algorithms and ≥3 problems)")
    logging.info(
      f"  Found: {len(algorithms)} algorithms, {len(valid_problems)} complete problems"
    )
    logging.info(separator)
    return None

  if analyzer is None:
    analyzer = StatisticalAnalyzer()

  logging.info(separator)
  logging.info(label)
  logging.info(separator)
  logging.info(
    f"Problems: {len(valid_problems)} "
    f"({', '.join(sorted(valid_problems)[:5])}"
    f"{'...' if len(valid_problems) > 5 else ''})"
  )
  logging.info(f"Algorithms: {len(algorithms)} ({', '.join(algorithms)})")
  logging.info("")

  data = []
  for algo in algorithms:
    algo_gaps = _extract_algorithm_gaps(all_results, valid_problems, algo)
    data.append(np.array(algo_gaps))
    logging.info(
      f"  {algo}: mean_gap={np.mean(algo_gaps):.2f}% ± {np.std(algo_gaps):.2f}%"
    )

  logging.info("")

  speedup_summary: dict[str, float] = {}
  if baseline_algorithm and baseline_algorithm in algorithms:
    logging.info(f"Geometric mean speedup (vs {baseline_algorithm} baseline):")
    for algo in algorithms:
      if algo == baseline_algorithm:
        logging.info(f"  {algo}: 1.00x (baseline)")
        speedup_summary[algo] = 1.0
      else:
        speedups = []
        for problem in valid_problems:
          baseline_res = all_results[problem].get(baseline_algorithm, {})
          algo_res = all_results[problem].get(algo, {})
          baseline_time = baseline_res.get("mean_time", 0.0)
          algo_time = algo_res.get("mean_time", 0.0)
          if algo_time > 0 and baseline_time > 0:
            speedups.append(baseline_time / algo_time)

        if speedups:
          geomean_speedup = float(np.exp(np.mean(np.log(speedups))))
          speedup_summary[algo] = geomean_speedup
          logging.info(f"  {algo}: {geomean_speedup:.2f}x vs {baseline_algorithm}")

    logging.info("")

  friedman_result = analyzer.friedman_test(data)
  logging.info("Friedman Test:")
  logging.info(f"  Statistic: {friedman_result['statistic']:.4f}")
  logging.info(f"  p-value: {friedman_result['p_value']:.6f}")
  logging.info(f"  Significant: {friedman_result['significant']}")

  comparison_results: dict[str, Any] = {
    "problems": valid_problems,
    "algorithms": algorithms,
    "n_problems": len(valid_problems),
    "n_algorithms": len(algorithms),
    "friedman_test": friedman_result,
  }
  if speedup_summary:
    comparison_results["speedups"] = speedup_summary

  if friedman_result.get("post_hoc_required", False):
    logging.info("\nNemenyi Post-Hoc Test:")
    nemenyi_result = analyzer.nemenyi_posthoc(data, algorithms)
    comparison_results["nemenyi_posthoc"] = nemenyi_result

    logging.info(f"  Critical distance: {nemenyi_result['critical_distance']:.4f}")

    if nemenyi_result.get("significant_pairs"):
      logging.info("  Significant pairwise differences:")
      for i, j, p_val in nemenyi_result["significant_pairs"]:
        name_i = nemenyi_result["algorithm_names"][i]
        name_j = nemenyi_result["algorithm_names"][j]
        logging.info(f"    {name_i} vs {name_j}: p={p_val:.4f}")
    else:
      logging.info("  No significant pairwise differences detected")

  logging.info("")
  return comparison_results


def compare_algorithms_by_group(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  group_key: str = "edge_type",
  analyzer: StatisticalAnalyzer | None = None,
  separator: str = "=" * 80,
) -> dict[str, Any]:
  """
  Compare algorithms grouped by problem metadata (e.g. edge_type, size bracket).

  Args:
      all_results: Nested dictionary: problem_name -> algorithm_name -> metrics
      problem_configs: List of ProblemConfig instances
      group_key: Attribute of ProblemConfig or key in result metadata to group by
      analyzer: Optional StatisticalAnalyzer instance
      separator: Separator line for logging

  Returns:
      Dictionary mapping group_name -> comparison results dict or None
  """
  groups: dict[str, list[str]] = {}
  config_map = {p.name: p for p in problem_configs}

  for problem_name, problem_data in all_results.items():
    cfg = config_map.get(problem_name)
    group_val: Any = None
    if cfg is not None and hasattr(cfg, group_key):
      group_val = getattr(cfg, group_key)
    elif problem_data:
      first_algo_res = next(iter(problem_data.values()), {})
      group_val = first_algo_res.get(group_key)

    group_name = str(group_val) if group_val is not None else "unknown"
    groups.setdefault(group_name, []).append(problem_name)

  results_by_group: dict[str, Any] = {}
  for group_name, prob_list in sorted(groups.items()):
    common_algos = sorted({
      a for p in prob_list for a in all_results.get(p, {})
      if all(a in all_results[sp] for sp in prob_list)
    })
    group_label = f"GROUP COMPARISON: {group_key}={group_name}"
    res = compare_algorithms(
      all_results=all_results,
      problems=prob_list,
      algorithms=common_algos,
      analyzer=analyzer,
      label=group_label,
      separator=separator,
    )
    results_by_group[group_name] = res

  return results_by_group


# =============================================================================
# Stratified Cross-Problem Analysis (BUG FIX)
# =============================================================================


def perform_cross_problem_analysis_stratified(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  analyzer: StatisticalAnalyzer,
  cpu_threshold: int = 100,
  separator: str = "=" * 80,
) -> dict[str, Any]:
  """
  Perform stratified cross-problem statistical analysis.

  Args:
      all_results: Nested dictionary: problem_name -> algorithm_name -> metrics
      problem_configs: List of ProblemConfig instances with name, size, optimal
      analyzer: StatisticalAnalyzer instance
      cpu_threshold: Size threshold for CPU runs (default: 100)
      separator: Separator line for logging (default: 80 equals signs)

  Returns:
      Dictionary containing stratum results and summary statistics
  """
  logging.info(separator)
  logging.info("CROSS-PROBLEM ANALYSIS (STRATIFIED APPROACH)")
  logging.info(separator)
  logging.info("")
  logging.info("Using stratified analysis to handle CPU/GPU problem coverage difference:")
  logging.info(f"  - CPU runs: Problems with n < {cpu_threshold}")
  logging.info("  - GPU runs: All problems")
  logging.info("")

  problem_sizes = _get_problem_size_map(problem_configs)
  problem_names = list(all_results.keys())

  all_algorithms_set: set[str] = set()
  for problem_results in all_results.values():
    all_algorithms_set.update(problem_results.keys())
  all_algorithm_names: list[str] = sorted(all_algorithms_set)

  logging.info(f"Total problems analyzed: {len(problem_names)}")
  logging.info(f"Algorithms present: {', '.join(all_algorithm_names)}")
  logging.info("")

  # STRATUM 1: Small Problems with CPU + GPU Algorithms
  small_problems: list[str] = [
    p for p in problem_names if problem_sizes.get(p, 0) <= cpu_threshold
  ]
  small_problem_algos = sorted({
    a
    for p in small_problems
    for a in all_results.get(p, {})
    if all(a in all_results[sp] for sp in small_problems)
  })

  stratum1_results = compare_algorithms(
    all_results=all_results,
    problems=small_problems,
    algorithms=small_problem_algos,
    analyzer=analyzer,
    label=f"STRATUM 1: Small Problems (n ≤ {cpu_threshold})",
    separator=separator,
  )

  # STRATUM 2: All Problems with GPU-Only Algorithms
  gpu_algos = [a for a in all_algorithm_names if a != "CPU"]
  all_gpu_problems = [
    p for p in problem_names if all(a in all_results[p] for a in gpu_algos)
  ]
  baseline_algo = "HybridNaive" if "HybridNaive" in gpu_algos else (gpu_algos[0] if gpu_algos else None)

  stratum2_results = compare_algorithms(
    all_results=all_results,
    problems=all_gpu_problems,
    algorithms=gpu_algos,
    analyzer=analyzer,
    baseline_algorithm=baseline_algo,
    label="STRATUM 2: All Problems (GPU Algorithms Only)",
    separator=separator,
  )

  # Summary Statistics
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
        f"{algo}: mean_gap={np.mean(all_gaps):.2f}% ± {np.std(all_gaps):.2f}% (n={len(all_gaps)} problems)"
      )

  logging.info("")

  return {
    "stratum1_small_problems": stratum1_results,
    "stratum2_all_gpu_problems": stratum2_results,
    "summary_statistics": summary_stats,
    "cpu_threshold": cpu_threshold,
    "total_problems": len(problem_names),
    "total_algorithms": len(all_algorithm_names),
  }
