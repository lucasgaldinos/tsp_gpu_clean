"""
Aggregate Statistics Module for Benchmark V2
=============================================

Performs cross-problem statistical analysis across benchmarked algorithms.

Supports both unified cross-problem analysis (when ``cpu_threshold is None``
and all variants cover the cohort) and stratified fallback analysis (when a
size cutoff limits a subset of algorithms to smaller instances). Comparisons
are metric-agnostic and handle both solution quality (``mean_gap``) and
execution time (``mean_time`` / geometric speedup).

Usage:
    from benchmarking.analysis.aggregate_statistics import (
        perform_cross_problem_analysis_stratified,
        compare_algorithms,
        compare_algorithms_by_group,
    )
    from benchmarking.analysis.statistics import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    all_results = {
        "instance_1": {"AlgA": {...}, "AlgB": {...}},
        "instance_2": {"AlgA": {...}, "AlgB": {...}},
        ...
    }

    stats = perform_cross_problem_analysis_stratified(
        all_results=all_results,
        problem_configs=problem_configs,
        analyzer=analyzer,
        cpu_threshold=None,
        baseline_algorithm="AlgA",
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.cohort import BenchmarkCohort, resolve_algorithm_names
from benchmarking.analysis.statistics import StatisticalAnalyzer, compute_mean_ranks

if TYPE_CHECKING:
  from benchmarking.runner.config_loader import ProblemConfig
  from data_models.records import NemenyiResult

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
  metric_key: str = "mean_gap",
  higher_is_better: bool = False,
  metric_unit: str | None = None,
  *,
  cohort: BenchmarkCohort | None = None,
) -> dict[str, Any] | None:
  """Compare a subset of algorithms across a designated subset of problems.

  Validates that each problem has finite results for all requested algorithms,
  executes the Friedman test (with Iman-Davenport correction), runs Nemenyi
  post-hoc if significant, and computes geometric mean speedups when
  ``baseline_algorithm`` is provided.
  """
  active_cohort = cohort or BenchmarkCohort.from_results(all_results, sort_problems=False)
  valid_problems, matrix = active_cohort.metric_matrix(
    metric_key,
    algorithms=algorithms,
    problems=problems,
  )

  if len(algorithms) < 3 or len(valid_problems) < 3:
    logging.info(separator)
    logging.info(f"{label}: Skipped (need ≥3 algorithms and ≥3 problems)")
    logging.info(f"  Found: {len(algorithms)} algorithms, {len(valid_problems)} complete problems")
    logging.info(separator)
    return None

  if analyzer is None:
    analyzer = StatisticalAnalyzer()

  if metric_unit is None:
    unit_suffix = "%" if metric_key == "mean_gap" else ("s" if metric_key == "mean_time" else "")
  else:
    unit_suffix = metric_unit

  logging.info(separator)
  logging.info(label)
  logging.info(separator)
  logging.info(
    f"Problems: {len(valid_problems)} "
    f"({', '.join(sorted(valid_problems)[:5])}"
    f"{'...' if len(valid_problems) > 5 else ''})"
  )
  logging.info(f"Algorithms: {len(algorithms)} ({', '.join(algorithms)})")
  logging.info(f"Metric: {metric_key} ({'higher is better' if higher_is_better else 'lower is better'})")
  logging.info("")

  data: list[np.ndarray] = [matrix[:, idx] for idx in range(len(algorithms))]
  for algo, col in zip(algorithms, data, strict=True):
    std_val = float(np.std(col, ddof=1)) if col.size > 1 else 0.0
    logging.info(f"  {algo}: {metric_key}={float(np.mean(col)):.2f}{unit_suffix} ± {std_val:.2f}{unit_suffix}")

  logging.info("")

  speedup_summary: dict[str, float] = {}
  if baseline_algorithm and baseline_algorithm in algorithms:
    logging.info(f"Geometric mean speedup (vs {baseline_algorithm} baseline):")
    for algo in algorithms:
      if algo == baseline_algorithm:
        logging.info(f"  {algo}: 1.00x (baseline)")
        speedup_summary[algo] = 1.0
      else:
        sp = active_cohort.paired_geometric_speedup(algo, baseline_algorithm, problems=valid_problems)
        if sp.geomean is not None:
          speedup_summary[algo] = sp.geomean
          logging.info(f"  {algo}: {sp.geomean:.2f}x vs {baseline_algorithm}")

    logging.info("")

  rank_oriented_data = [-d for d in data] if higher_is_better else data
  friedman_result: dict[str, Any] = analyzer.friedman_test(rank_oriented_data)
  mean_ranks: dict[str, float] = compute_mean_ranks(matrix, algorithms, higher_is_better=higher_is_better)
  logging.info("Friedman Test (Iman-Davenport):")
  logging.info(f"  Chi-square: {friedman_result['statistic']:.4f} (p={friedman_result['p_value']:.6f})")
  logging.info(
    f"  F_F({friedman_result['df1']}, {friedman_result['df2']}): "
    f"{friedman_result['statistic_f']:.4f} (p={friedman_result['p_value_f']:.6f})"
  )
  logging.info(f"  Significant: {friedman_result['significant']}")
  logging.info("  Mean ranks: " + ", ".join(f"{name}={rank:.2f}" for name, rank in mean_ranks.items()))

  comparison_results: dict[str, Any] = {
    "problems": valid_problems,
    "algorithms": algorithms,
    "n_problems": len(valid_problems),
    "n_algorithms": len(algorithms),
    "friedman_test": friedman_result,
    "mean_ranks": mean_ranks,
  }
  if speedup_summary:
    comparison_results["speedups"] = speedup_summary

  if friedman_result.get("post_hoc_required", False):
    logging.info("\nNemenyi Post-Hoc Test:")
    nemenyi_result: NemenyiResult = analyzer.nemenyi_posthoc(rank_oriented_data, algorithms)
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
  baseline_algorithm: str | None = None,
  metric_key: str = "mean_gap",
  higher_is_better: bool = False,
) -> dict[str, Any]:
  """Compare algorithms grouped by problem metadata (e.g. edge_type, size bracket)."""
  cohort = BenchmarkCohort.from_results(all_results, problem_configs, sort_problems=False)
  groups: dict[str, list[str]] = {}
  config_map: dict[str, ProblemConfig] = {p.name: p for p in problem_configs}

  for problem_name, problem_data in all_results.items():
    cfg: ProblemConfig | None = config_map.get(problem_name)
    group_val: Any = None
    if cfg is not None and hasattr(cfg, group_key):
      group_val = getattr(cfg, group_key)
    elif problem_data:
      first_algo_res: dict[str, Any] = next(iter(problem_data.values()), {})
      group_val = first_algo_res.get(group_key)

    group_name = str(group_val) if group_val is not None else "unknown"
    groups.setdefault(group_name, []).append(problem_name)

  results_by_group: dict[str, Any] = {}
  for group_name, prob_list in sorted(groups.items()):
    ordered_group_algos: list[str] = resolve_algorithm_names({p: all_results[p] for p in prob_list})
    common_algos: list[str] = [a for a in ordered_group_algos if all(a in all_results[sp] for sp in prob_list)]
    results_by_group[group_name] = compare_algorithms(
      all_results=all_results,
      problems=prob_list,
      algorithms=common_algos,
      analyzer=analyzer,
      baseline_algorithm=baseline_algorithm,
      label=f"GROUP COMPARISON: {group_key}={group_name}",
      separator=separator,
      metric_key=metric_key,
      higher_is_better=higher_is_better,
      cohort=cohort,
    )

  return results_by_group


# =============================================================================
# Cross-Problem Analysis (Unified when unthresholded; Stratified if gated)
# =============================================================================


def _compute_cross_problem_summary(
  cohort: BenchmarkCohort,
  separator: str,
) -> dict[str, dict[str, Any]]:
  """Compute and log per-algorithm cross-problem mean/std summary for gap and time."""
  logging.info(separator)
  logging.info("CROSS-PROBLEM SUMMARY")
  logging.info(separator)

  summary_stats: dict[str, dict[str, Any]] = {}
  for algo in cohort.algorithms:
    agg = cohort.summarize_algorithm(algo)
    if agg is None or (agg.mean_gap is None and agg.mean_time is None):
      continue

    n_gap_or_time = max(
      len(cohort.algorithm_metric_values(algo, "mean_gap")),
      len(cohort.algorithm_metric_values(algo, "mean_time")),
    )
    entry: dict[str, Any] = {"n_problems": n_gap_or_time}
    if agg.mean_gap is not None and agg.std_gap is not None:
      entry["mean_gap"] = agg.mean_gap
      entry["std_gap"] = agg.std_gap
    if agg.mean_time is not None and agg.std_time is not None:
      entry["mean_time"] = agg.mean_time
      entry["std_time"] = agg.std_time

    summary_stats[algo] = entry
    gap_str = f"mean_gap={agg.mean_gap:.2f}% ± {agg.std_gap:.2f}%" if agg.mean_gap is not None else "mean_gap=N/A"
    time_str = f"mean_time={agg.mean_time:.2f}s ± {agg.std_time:.2f}s" if agg.mean_time is not None else "mean_time=N/A"
    logging.info(f"{algo}: {gap_str} | {time_str} (n={entry['n_problems']} problems)")

  logging.info("")
  return summary_stats


def perform_cross_problem_analysis_stratified(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  analyzer: StatisticalAnalyzer,
  cpu_threshold: int | None = None,
  separator: str = "=" * 80,
  baseline_algorithm: str | None = None,
) -> dict[str, Any]:
  """Perform cross-problem statistical analysis across all benchmarked algorithms.

  When ``cpu_threshold is None`` (no size cutoff), runs a unified multi-metric
  analysis (quality ``mean_gap`` or ``mean_cost`` fallback, and runtime ``mean_time``
  with geometric mean speedups against ``baseline_algorithm``) with no stratum split.

  When an integer ``cpu_threshold`` is provided, falls back to stratified analysis:
  Stratum 1 compares algorithms present on problems with ``size <= cpu_threshold``;
  Stratum 2 compares algorithms covering instances above the threshold across all
  complete problems.
  """
  cohort = BenchmarkCohort.from_results(all_results, problem_configs, sort_problems=False)
  problem_names = list(cohort.problems)
  all_algorithm_names = list(cohort.algorithms)
  effective_baseline = cohort.baseline(baseline_algorithm)

  if cpu_threshold is None:
    logging.info(separator)
    logging.info("CROSS-PROBLEM ANALYSIS")
    logging.info(separator)
    logging.info("")
    logging.info(f"Total problems analyzed: {len(problem_names)}")
    logging.info(f"Algorithms present: {', '.join(all_algorithm_names)}")
    logging.info(f"Baseline algorithm: {effective_baseline}")
    logging.info("")

    complete_for_all = cohort.complete_problems(algorithms=all_algorithm_names)
    if len(complete_for_all) >= 3:
      target_algos = all_algorithm_names
    else:
      full_cov = [a for a in all_algorithm_names if all(a in all_results[p] for p in problem_names)]
      target_algos = full_cov if len(full_cov) >= 3 else all_algorithm_names

    comp_baseline: str | None = effective_baseline if effective_baseline in target_algos else None
    quality_metric = cohort.quality_metric_key(
      algorithms=target_algos,
      problems=problem_names,
      require_complete=False,
    )
    gap_comparison = compare_algorithms(
      all_results=all_results,
      problems=problem_names,
      algorithms=target_algos,
      analyzer=analyzer,
      baseline_algorithm=None,
      label=f"CROSS-PROBLEM QUALITY COMPARISON ({quality_metric})",
      separator=separator,
      metric_key=quality_metric,
      higher_is_better=False,
      cohort=cohort,
    )
    time_comparison = compare_algorithms(
      all_results=all_results,
      problems=problem_names,
      algorithms=target_algos,
      analyzer=analyzer,
      baseline_algorithm=comp_baseline,
      label="CROSS-PROBLEM RUNTIME COMPARISON (mean_time & speedup)",
      separator=separator,
      metric_key="mean_time",
      higher_is_better=False,
      cohort=cohort,
    )

    summary_stats = _compute_cross_problem_summary(cohort, separator)
    return {
      "stratified": False,
      "gap_comparison": gap_comparison,
      "time_comparison": time_comparison,
      "summary_statistics": summary_stats,
      "cpu_threshold": None,
      "baseline_algorithm": effective_baseline,
      "total_problems": len(problem_names),
      "total_algorithms": len(all_algorithm_names),
    }

  logging.info(separator)
  logging.info("CROSS-PROBLEM ANALYSIS (STRATIFIED APPROACH)")
  logging.info(separator)
  logging.info("")
  logging.info(f"Stratified analysis active for size threshold n ≤ {cpu_threshold}:")
  logging.info(f"  - Stratum 1: Problems with n ≤ {cpu_threshold}")
  logging.info("  - Stratum 2: Full-cohort algorithms covering all instances")
  logging.info("")
  logging.info(f"Total problems analyzed: {len(problem_names)}")
  logging.info(f"Algorithms present: {', '.join(all_algorithm_names)}")
  logging.info("")

  small_problems = [p for p in problem_names if (cohort.problem_size(p) or 0) <= cpu_threshold]
  small_problem_algos = [
    a for a in all_algorithm_names if small_problems and all(a in all_results.get(sp, {}) for sp in small_problems)
  ]
  stratum1_baseline: str | None = (
    effective_baseline
    if effective_baseline in small_problem_algos
    else (small_problem_algos[0] if small_problem_algos else None)
  )
  stratum1_metric = cohort.quality_metric_key(
    algorithms=small_problem_algos,
    problems=small_problems,
    require_complete=False,
  )
  stratum1_results: dict[str, Any] | None = compare_algorithms(
    all_results=all_results,
    problems=small_problems,
    algorithms=small_problem_algos,
    analyzer=analyzer,
    baseline_algorithm=stratum1_baseline,
    label=f"STRATUM 1: Small Problems (n ≤ {cpu_threshold})",
    separator=separator,
    metric_key=stratum1_metric,
    higher_is_better=False,
    cohort=cohort,
  )

  large_problems = [p for p in problem_names if (cohort.problem_size(p) or 0) > cpu_threshold]
  coverage_target = large_problems if large_problems else problem_names
  full_coverage_algos = [a for a in all_algorithm_names if all(a in all_results.get(p, {}) for p in coverage_target)]
  stratum2_problems = cohort.complete_problems(algorithms=full_coverage_algos, problems=problem_names)
  stratum2_baseline: str | None = (
    baseline_algorithm
    if baseline_algorithm and baseline_algorithm in full_coverage_algos
    else (full_coverage_algos[0] if full_coverage_algos else None)
  )
  stratum2_metric = cohort.quality_metric_key(
    algorithms=full_coverage_algos,
    problems=stratum2_problems,
    require_complete=False,
  )
  stratum2_results: dict[str, Any] | None = compare_algorithms(
    all_results=all_results,
    problems=stratum2_problems,
    algorithms=full_coverage_algos,
    analyzer=analyzer,
    baseline_algorithm=stratum2_baseline,
    label="STRATUM 2: Full-Coverage Algorithms (All Problems)",
    separator=separator,
    metric_key=stratum2_metric,
    higher_is_better=False,
    cohort=cohort,
  )

  summary_stats = _compute_cross_problem_summary(cohort, separator)

  return {
    "stratified": True,
    "stratum1_small_problems": stratum1_results,
    "stratum2_all_gpu_problems": stratum2_results,
    "summary_statistics": summary_stats,
    "cpu_threshold": cpu_threshold,
    "baseline_algorithm": effective_baseline,
    "total_problems": len(problem_names),
    "total_algorithms": len(all_algorithm_names),
  }
