"""
Problem Statistics Module for Benchmark V2
===========================================

Performs statistical analysis on single-problem results.

This module analyzes results from multiple algorithms on a single TSP problem,
including:
- Friedman test for k-algorithm comparison
- Nemenyi post-hoc test
- Pairwise comparisons with Holm-Bonferroni correction
- Summary statistics and speedup calculations

Extracted from the analysis helpers formerly embedded in the chapter 4 validation script.

Usage:
    from benchmarking.analysis.problem_statistics import perform_statistical_analysis
    from benchmarking.analysis.statistics import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    results = {
        "CPU": {...},  # Results dict from algorithm_runner
        "HybridOptimized": {...},
        "FullGPUMonolithic": {...}
    }

    stats = perform_statistical_analysis(
        problem_name="berlin52",
        optimal_cost=7542.0,
        results=results,
        analyzer=analyzer
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, LiteralString

import numpy as np

from benchmarking.analysis.stats_utils import resolve_baseline_algorithm

if TYPE_CHECKING:
  from collections.abc import Sequence

  from numpy.typing import NDArray

  from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary
  from data_models.records import NemenyiResult

# =============================================================================
# Helper Functions
# =============================================================================


def _get_data_field(results_dict: dict[str, Any], field: str) -> list[Any]:
  """
  Safely extract data field with fallback for checkpoint vs direct format.

  Handles two formats:
  - Checkpoint format: uses raw_* prefix (raw_costs, raw_times, raw_gaps)
  - Direct format: uses direct names (final_costs, times, gaps)

  Args:
      results_dict: Algorithm results dictionary
      field: Field name ('costs', 'times', or 'gaps')

  Returns:
      List of values
  """
  raw_field: str = f"raw_{field}"
  direct_field: str = "final_costs" if field == "costs" else field
  val = results_dict.get(raw_field, results_dict.get(direct_field, []))
  return val if isinstance(val, list) else list(val)


def _resolve_metric_mean(
  results_dict: dict[str, Any],
  metric_key: str,
  raw_series: Sequence[float] | np.ndarray,
  alg_name: str,
) -> float:
  """Resolve mean metric value lazily from dictionary or raw data series.

  Prefers `results_dict[metric_key]` if present and not None. Otherwise,
  computes the mean of non-NaN elements in `raw_series`.
  Raises KeyError if neither a precomputed metric nor valid raw data is present.
  """
  val = results_dict.get(metric_key)
  if val is not None:
    return float(val)

  if len(raw_series) > 0:
    arr = np.asarray(raw_series, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
      return float(np.mean(valid))

  raise KeyError(
    f"Algorithm '{alg_name}' is missing both precomputed '{metric_key}' "
    f"and valid raw data series."
  )


def _resolve_gap_mean(
  results_dict: dict[str, Any],
  mean_cost: float,
  optimal_cost: float,
  raw_gaps: Sequence[float] | np.ndarray,
  alg_name: str,
) -> float:
  """Resolve mean gap percentage lazily from dictionary, cost, or raw gaps."""
  val = results_dict.get("mean_gap")
  if val is not None:
    return float(val)

  if optimal_cost > 0:
    return float((mean_cost - optimal_cost) / optimal_cost * 100.0)

  if len(raw_gaps) > 0:
    arr = np.asarray(raw_gaps, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
      return float(np.mean(valid))

  raise KeyError(
    f"Algorithm '{alg_name}' is missing both precomputed 'mean_gap' "
    f"and valid raw gap data."
  )


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
  """
  Apply Holm-Bonferroni multiple comparison correction.

  Corrects p-values for family-wise error rate when performing
  multiple pairwise comparisons.

  Args:
      p_values: List of p-values to correct
      alpha: Significance level (default: 0.05)

  Returns:
      Tuple of (reject_list, pvals_corrected) where:
      - reject_list: Boolean list indicating which hypotheses to reject
      - pvals_corrected: Adjusted p-values
  """
  n: int = len(p_values)
  if n == 0:
    return [], []

  # Sort p-values with original indices
  sorted_indices = np.argsort(p_values)
  sorted_pvals = np.array(p_values)[sorted_indices]

  # Adjusted p-values (step-down procedure)
  raw_adjusted = np.minimum((n - np.arange(n)) * sorted_pvals, 1.0)
  pvals_corrected_sorted = np.maximum.accumulate(raw_adjusted)

  # Step-down rejection: an hypothesis is rejected if and only if its adjusted p-value <= alpha.
  # This guarantees step-down termination at the first non-rejected hypothesis and ensures
  # strict consistency between `reject` and `pvals_corrected` (BUG-46 / DP12-23).
  reject_sorted = pvals_corrected_sorted <= alpha

  # Restore original order
  reject = np.zeros(n, dtype=bool)
  pvals_corrected = np.zeros(n)
  reject[sorted_indices] = reject_sorted
  pvals_corrected[sorted_indices] = pvals_corrected_sorted

  return [bool(x) for x in reject], [float(x) for x in pvals_corrected]


# =============================================================================
# Main Analysis Function
# =============================================================================


def perform_statistical_analysis(
  problem_name: str,
  optimal_cost: float,
  results: dict[str, dict[str, Any]],
  analyzer: StatisticalAnalyzer,
  short_sep: str = "-" * 80,
  baseline_algorithm: str | None = None,
) -> dict[str, Any]:
  """
  Perform statistical analysis on results for a single problem.

  Executes comprehensive statistical tests comparing multiple algorithms
  on the same problem instance:

  1. Friedman test (if k ≥ 3 algorithms)
  2. Nemenyi post-hoc test (if Friedman significant)
  3. Pairwise comparisons with Holm-Bonferroni correction
  4. Summary statistics and speedup calculations

  Args:
      problem_name: Name of TSPLIB instance (e.g., "berlin52")
      optimal_cost: Known optimal solution cost
      results: Dictionary mapping algorithm_name -> results_dict
      analyzer: StatisticalAnalyzer instance
      short_sep: Separator line for logging (default: 80 dashes)
      baseline_algorithm: Optional baseline algorithm name for speedup calculation.
          Resolved by `resolve_baseline_algorithm`: `None` selects the first
          algorithm in `results` insertion order; a requested name that is absent
          logs a warning and falls back to that same first algorithm; an empty
          `results` mapping resolves to `"Unknown"`.

  Returns:
      Dictionary containing statistical test results
  """
  logging.info(f"\nStatistical Analysis: {problem_name}")
  logging.info(short_sep)

  algorithm_names = list(results.keys())

  # Guard against insufficient data
  if len(algorithm_names) < 2:
    logging.warning(
      f"Skipping statistical analysis for {problem_name}: only {len(algorithm_names)} algorithm(s) available (need ≥2)"
    )
    return {
      "problem_name": problem_name,
      "optimal_cost": optimal_cost,
      "error": "insufficient_data",
      "available_algorithms": algorithm_names,
    }

  cost_data: list[NDArray[Any]] = [np.array(_get_data_field(results[name], "costs")) for name in algorithm_names]
  time_data: list[NDArray[Any]] = [np.array(_get_data_field(results[name], "times")) for name in algorithm_names]
  gap_data: list[NDArray[Any]] = [np.array(_get_data_field(results[name], "gaps")) for name in algorithm_names]

  # Check for ragged sample lengths across algorithms (Task 12.7, BUG-33)
  cost_lengths: list[int] = [len(d) for d in cost_data]
  min_cost_len: int = min(cost_lengths) if cost_lengths else 0
  if len(set(cost_lengths)) > 1:
    logging.warning(
      "Mismatched sample counts across algorithms for problem '%s': lengths=%s. "
      "Truncating to common minimum length %d for paired Friedman test.",
      problem_name,
      cost_lengths,
      min_cost_len,
    )
    cost_data = [d[:min_cost_len] for d in cost_data]

  time_lengths: list[int] = [len(d) for d in time_data if len(d) > 0]
  if time_lengths and len(set(time_lengths)) > 1:
    min_time_len: int = min(time_lengths)
    time_data = [d[:min_time_len] for d in time_data]

  # =========================================================================
  # Friedman Test (k ≥ 3 algorithms)
  # =========================================================================

  friedman_result: dict[str, Any] | None = None
  nemenyi_result: NemenyiResult | None = None
  mean_ranks: dict[str, float] = {}

  if len(algorithm_names) >= 3:
    if min_cost_len < 2:
      logging.info(
        "Skipping Friedman test for '%s': common sample size %d < 2 (insufficient for paired test).",
        problem_name,
        min_cost_len,
      )
      friedman_result = {
        "statistic": 0.0,
        "p_value": 1.0,
        "significant": False,
        "post_hoc_required": False,
      }
    else:
      friedman_result = analyzer.friedman_test(cost_data)

      logging.info(f"Friedman Test (k={len(algorithm_names)} algorithms):")
      logging.info(f"  Statistic: {friedman_result['statistic']:.4f}")
      logging.info(f"  p-value: {friedman_result['p_value']:.6f}")
      logging.info(f"  Significant: {friedman_result['significant']}")

      # Calculate mean ranks
      from scipy.stats import rankdata

      n_obs: int = len(cost_data[0])
      for i, name in enumerate(algorithm_names):
        ranks = []
        for obs_idx in range(n_obs):
          obs_costs = [cost_data[j][obs_idx] for j in range(len(algorithm_names))]
          obs_ranks: NDArray[np.float64] = rankdata(obs_costs, method="average")
          ranks.append(obs_ranks[i])
        mean_ranks[name] = float(np.mean(ranks))

      # Nemenyi post-hoc if significant
      if friedman_result.get("post_hoc_required"):
        logging.info("\nNemenyi Post-Hoc Test:")
        nemenyi_res: NemenyiResult = analyzer.nemenyi_posthoc(cost_data, algorithm_names)
        nemenyi_result = nemenyi_res

        logging.info(f"  Critical distance: {nemenyi_res['critical_distance']:.4f}")
        logging.info("  Mean ranks:")
        for name, rank in mean_ranks.items():
          logging.info(f"    {name}: {rank:.2f}")

        logging.info("\n  Significant pairwise differences:")
        if nemenyi_res["significant_pairs"]:
          for i, j, p_val in nemenyi_res["significant_pairs"]:
            name_i: str = nemenyi_res["algorithm_names"][i]
            name_j: str = nemenyi_res["algorithm_names"][j]
            rank_diff: float = abs(mean_ranks[name_i] - mean_ranks[name_j])
            logging.info(f"    {name_i} vs {name_j}: rank_diff={rank_diff:.2f}, p={p_val:.4f}")
        else:
          logging.info("    None (no significant differences detected)")

  # =========================================================================
  # Pairwise Comparisons
  # =========================================================================

  logging.info("\nPairwise Comparisons:")
  logging.info("Following statistical methodology:")
  logging.info("  - Normality testing (Shapiro-Wilk)")
  logging.info("  - Test selection (paired t-test vs Wilcoxon)")
  logging.info("  - Effect size (Cohen's d)")
  logging.info("  - 95% Confidence Intervals")
  logging.info("  - Holm-Bonferroni multiple comparison correction\n")

  comparisons = []
  for i, name1 in enumerate(algorithm_names):
    for j, name2 in enumerate(algorithm_names[i + 1 :], start=i + 1):
      summary: StatisticalSummary = analyzer.paired_comparison(
        cost_data[i],
        cost_data[j],
        f"{name1} vs {name2}",
        metric_name="cost",
      )
      comparisons.append(summary)

      if summary.test_used == "no_test_needed":
        logging.info(f"  {name1} vs {name2}:")
        logging.info("    All measurements identical (no variance)")
        logging.info(f"    Mean {name1}: {summary.mean_a:.2f}")
        logging.info(f"    Mean {name2}: {summary.mean_b:.2f}")
        logging.info("    No statistical test needed")
        continue

      logging.info(f"  {name1} vs {name2}:")
      logging.info("    Normality (Shapiro-Wilk):")
      logging.info(
        f"      {name1}: p={summary.normality_p_value_a:.4f} "
        f"{'(normal)' if summary.normality_p_value_a >= 0.05 else '(non-normal)'}"
      )
      logging.info(
        f"      {name2}: p={summary.normality_p_value_b:.4f} "
        f"{'(normal)' if summary.normality_p_value_b >= 0.05 else '(non-normal)'}"
      )
      logging.info(f"    Test selected: {summary.test_used}")
      logging.info(f"    Mean difference: {summary.mean_difference:.2f}")
      logging.info(f"    95% CI difference: [{summary.ci_95_difference[0]:.2f}, {summary.ci_95_difference[1]:.2f}]")
      logging.info(f"    p-value (uncorrected): {summary.p_value:.6f}")
      logging.info(f"    Cohen's d: {summary.effect_size:.4f}")

      abs_d: float = abs(summary.effect_size)
      if abs_d < 0.2:
        effect_interp = "negligible"
      elif abs_d < 0.5:
        effect_interp = "small"
      elif abs_d < 0.8:
        effect_interp = "medium"
      else:
        effect_interp = "large"
      logging.info(f"    Effect size interpretation: {effect_interp}")

  # Apply Holm-Bonferroni correction
  reject: list[bool] = []
  pvals_corrected: list[float] = []
  if len(comparisons) > 1:
    p_values = [c.p_value for c in comparisons]
    reject, pvals_corrected = holm_bonferroni_correction(p_values, alpha=0.05)

    logging.info("\n  Multiple Comparison Correction (Holm-Bonferroni, alpha=0.05):")
    for comp, p_orig, p_adj, sig in zip(comparisons, p_values, pvals_corrected, reject, strict=True):
      logging.info(f"    {comp.comparison_label}:")
      logging.info(f"      Original p-value: {p_orig:.6f}")
      logging.info(f"      Adjusted p-value: {p_adj:.6f}")
      logging.info(f"      Significant (after correction): {sig}")

  # =========================================================================
  # Summary Table
  # =========================================================================

  logging.info(f"\nSummary: {problem_name} (optimal={optimal_cost})")
  logging.info(short_sep)
  header_fmt: LiteralString = (
    f"{'Algorithm':<20} {'Mean Cost':<15} {'Gap (%)':<12} {'Time (s)':<12} {'H2D (MB)':<12} {'D2H (MB)':<12}"
  )
  logging.info(header_fmt)
  logging.info(short_sep)

  for i, name in enumerate(algorithm_names):
    r = results[name]
    m_cost = _resolve_metric_mean(r, "mean_cost", cost_data[i], name)
    m_gap = _resolve_gap_mean(r, m_cost, optimal_cost, gap_data[i], name)
    m_time = _resolve_metric_mean(r, "mean_time", time_data[i], name)
    row_fmt: str = (
      f"{name:<20} "
      f"{m_cost:<15.2f} "
      f"{m_gap:<12.2f} "
      f"{m_time:<12.2f} "
      f"{r.get('mean_h2d_mb', 0.0):<12.2f} "
      f"{r.get('mean_d2h_mb', 0.0):<12.2f}"
    )
    logging.info(row_fmt)

  # Calculate speedups
  logging.info("\nSpeedups (relative to baseline):")
  baseline_name: str = resolve_baseline_algorithm(algorithm_names, baseline_algorithm)
  base_idx: int = algorithm_names.index(baseline_name)
  base_r = results[baseline_name]
  baseline_time = _resolve_metric_mean(base_r, "mean_time", time_data[base_idx], baseline_name)

  for i, name in enumerate(algorithm_names):
    if name == baseline_name:
      logging.info(f"  {name}: 1.00x (baseline)")
    else:
      cand_time = _resolve_metric_mean(results[name], "mean_time", time_data[i], name)
      speedup = (baseline_time / cand_time) if (cand_time > 0 and baseline_time > 0) else 1.0
      logging.info(f"  {name}: {speedup:.2f}x vs {baseline_name}")

  logging.info("")

  # =========================================================================
  # Build Return Dictionary
  # =========================================================================

  statistical_results: dict[str, Any] = {
    "problem_name": problem_name,
    "optimal_cost": optimal_cost,
    "baseline_algorithm": baseline_name,
  }

  if friedman_result:
    statistical_results["friedman_test"] = {
      "statistic": float(friedman_result["statistic"]),
      "p_value": float(friedman_result["p_value"]),
      "significant": bool(friedman_result["significant"]),
      "post_hoc_required": bool(friedman_result["post_hoc_required"]),
    }

    if nemenyi_result:
      statistical_results["nemenyi_posthoc"] = {
        "critical_distance": float(nemenyi_result["critical_distance"]),
        "mean_ranks": mean_ranks,
        "algorithm_names": nemenyi_result["algorithm_names"],
        "significant_pairs": [
          {
            "algorithm_i": nemenyi_result["algorithm_names"][i],
            "algorithm_j": nemenyi_result["algorithm_names"][j],
            "p_value": float(p_val),
            "rank_difference": abs(
              mean_ranks[nemenyi_result["algorithm_names"][i]] - mean_ranks[nemenyi_result["algorithm_names"][j]]
            ),
          }
          for i, j, p_val in nemenyi_result["significant_pairs"]
        ],
      }

  pairwise_comps: list[dict[str, Any]] = []
  for i, comp in enumerate(comparisons):
    comparison_dict: dict[str, Any] = {
      "comparison": comp.comparison_label,
      "test_used": comp.test_used,
      "p_value": float(comp.p_value),
      "mean_difference": float(comp.mean_difference),
      "ci_95_lower": float(comp.ci_95_difference[0]),
      "ci_95_upper": float(comp.ci_95_difference[1]),
      "cohens_d": float(comp.effect_size),
      "normality_p_value_a": float(comp.normality_p_value_a),
      "normality_p_value_b": float(comp.normality_p_value_b),
    }

    if len(comparisons) > 1 and i < len(pvals_corrected):
      comparison_dict["p_value_corrected"] = float(pvals_corrected[i])
      comparison_dict["significant_after_correction"] = bool(reject[i])

    abs_d = abs(comp.effect_size)
    if abs_d < 0.2:
      comparison_dict["effect_interpretation"] = "negligible"
    elif abs_d < 0.5:
      comparison_dict["effect_interpretation"] = "small"
    elif abs_d < 0.8:
      comparison_dict["effect_interpretation"] = "medium"
    else:
      comparison_dict["effect_interpretation"] = "large"

    pairwise_comps.append(comparison_dict)

  statistical_results["pairwise_comparisons"] = pairwise_comps
  return statistical_results
