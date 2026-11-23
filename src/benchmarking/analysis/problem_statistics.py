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
        "AlgA": {...},  # Results dict from algorithm_runner
        "AlgB": {...},
        "AlgC": {...}
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

from benchmarking.analysis.statistics import compute_mean_ranks
from benchmarking.analysis.stats_utils import (
  finite_metric,
  holm_bonferroni_correction,
  resolve_baseline_algorithm,
)

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
  val = results_dict.get(raw_field)
  if val is None:
    val = results_dict.get(direct_field)
  if val is None:
    return []
  return val if isinstance(val, list) else list(val)


def _resolve_metric_mean(
  results_dict: dict[str, Any],
  metric_key: str,
  raw_series: Sequence[float] | np.ndarray,
  alg_name: str,
) -> float:
  """Resolve mean metric value lazily from dictionary or raw data series.

  Prefers `results_dict[metric_key]` if present and finite. Otherwise,
  computes the mean of finite elements in `raw_series`.
  Raises KeyError if neither a precomputed metric nor valid raw data is present.
  """
  val = finite_metric(results_dict.get(metric_key))
  if val is not None:
    return val

  if len(raw_series) > 0:
    arr = np.asarray(raw_series, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
      return float(np.mean(valid))

  raise KeyError(f"Algorithm '{alg_name}' is missing both precomputed '{metric_key}' and valid raw data series.")


def _resolve_gap_mean(
  results_dict: dict[str, Any],
  mean_cost: float,
  optimal_cost: float | None,
  raw_gaps: Sequence[float] | np.ndarray,
  alg_name: str,
) -> float | None:
  """Resolve mean gap percentage lazily from dictionary, cost, or raw gaps.

  Returns ``None`` when ``optimal_cost`` is ``None`` and no finite gap data is available.
  """
  val = finite_metric(results_dict.get("mean_gap"))
  if val is not None:
    return val

  if optimal_cost is not None and optimal_cost > 0:
    return float((mean_cost - optimal_cost) / optimal_cost * 100.0)

  if len(raw_gaps) > 0:
    arr = np.asarray(raw_gaps, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
      return float(np.mean(valid))

  if optimal_cost is None:
    return None

  raise KeyError(f"Algorithm '{alg_name}' is missing both precomputed 'mean_gap' and valid raw gap data.")


# =============================================================================
# Main Analysis Function
# =============================================================================


def _run_omnibus_friedman_nemenyi(
  *,
  problem_name: str,
  metric_label: str,
  series_data: list[NDArray[Any]],
  algorithm_names: list[str],
  analyzer: StatisticalAnalyzer,
) -> tuple[dict[str, Any] | None, NemenyiResult | None, dict[str, float]]:
  """Run Friedman (Iman-Davenport) and Nemenyi post-hoc tests on paired series."""
  if len(algorithm_names) < 3:
    return None, None, {}

  min_len = min((len(d) for d in series_data), default=0)
  if min_len < 2:
    logging.info(
      "Skipping Friedman test (%s) for '%s': common sample size %d < 2 (insufficient for paired test).",
      metric_label,
      problem_name,
      min_len,
    )
    k_algos = len(algorithm_names)
    neutral = {
      "statistic": 0.0,
      "p_value": 1.0,
      "statistic_f": 0.0,
      "df1": max(0, k_algos - 1),
      "df2": max(0, (k_algos - 1) * (min_len - 1)),
      "p_value_f": 1.0,
      "significant": False,
      "post_hoc_required": False,
    }
    return neutral, None, {}

  friedman_result = analyzer.friedman_test(series_data)
  logging.info(f"Friedman Test [{metric_label}] (k={len(algorithm_names)} algorithms):")
  logging.info(f"  Chi-square: {friedman_result['statistic']:.4f} (p={friedman_result['p_value']:.6f})")
  logging.info(
    f"  F_F({friedman_result['df1']}, {friedman_result['df2']}): "
    f"{friedman_result['statistic_f']:.4f} (p={friedman_result['p_value_f']:.6f})"
  )
  logging.info(f"  Significant: {friedman_result['significant']}")

  mean_ranks = compute_mean_ranks(series_data, algorithm_names)
  nemenyi_result: NemenyiResult | None = None

  if friedman_result.get("post_hoc_required"):
    logging.info(f"\nNemenyi Post-Hoc Test [{metric_label}]:")
    nemenyi_res: NemenyiResult = analyzer.nemenyi_posthoc(series_data, algorithm_names)
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

  return friedman_result, nemenyi_result, mean_ranks


def _serialize_friedman_section(
  friedman_result: dict[str, Any],
  nemenyi_result: NemenyiResult | None,
  mean_ranks: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Convert Friedman and optional Nemenyi outputs into JSON-ready dictionaries."""
  friedman_dict: dict[str, Any] = {
    "statistic": float(friedman_result["statistic"]),
    "p_value": float(friedman_result["p_value"]),
    "statistic_f": float(friedman_result["statistic_f"]),
    "df1": int(friedman_result["df1"]),
    "df2": int(friedman_result["df2"]),
    "p_value_f": float(friedman_result["p_value_f"]),
    "significant": bool(friedman_result["significant"]),
    "post_hoc_required": bool(friedman_result["post_hoc_required"]),
  }
  nemenyi_dict: dict[str, Any] | None = None
  if nemenyi_result:
    nemenyi_dict = {
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
  return friedman_dict, nemenyi_dict


def _interpret_effect_size(effect_size: float) -> str:
  """Classify rank-biserial effect magnitude into qualitative brackets."""
  abs_d: float = abs(effect_size)
  if abs_d < 0.2:
    return "negligible"
  if abs_d < 0.5:
    return "small"
  if abs_d < 0.8:
    return "medium"
  return "large"


def _run_pairwise_metric_comparisons(
  *,
  algorithm_names: list[str],
  series_data: list[NDArray[Any]],
  metric_name: str,
  analyzer: StatisticalAnalyzer,
  decimals: int = 2,
) -> list[dict[str, Any]]:
  """Run paired Wilcoxon + Holm-Bonferroni + Hodges-Lehmann + rank-biserial analysis."""
  logging.info(f"\nPairwise Comparisons ({metric_name}):")
  logging.info("Following statistical methodology (Demšar 2006):")
  logging.info("  - Normality diagnostics (Shapiro-Wilk, descriptive only)")
  logging.info("  - Paired test (Wilcoxon signed-rank, unconditional default)")
  logging.info("  - Effect size (matched-pairs rank-biserial correlation)")
  logging.info("  - 95% Confidence Intervals (Hodges-Lehmann distribution-free)")
  logging.info("  - Holm-Bonferroni multiple comparison correction\n")

  comparisons: list[StatisticalSummary] = []
  fmt = f".{decimals}f"
  for i, name1 in enumerate(algorithm_names):
    for j, name2 in enumerate(algorithm_names[i + 1 :], start=i + 1):
      summary: StatisticalSummary = analyzer.paired_comparison(
        series_data[i],
        series_data[j],
        f"{name1} vs {name2}",
        metric_name=metric_name,
      )
      comparisons.append(summary)

      if summary.test_used == "no_test_needed":
        logging.info(f"  {name1} vs {name2}:")
        if summary.test_justification == "constant_difference_no_test_needed":
          logging.info("    Constant paired difference (zero variance across differences)")
          logging.info(f"    Mean difference: {summary.mean_difference:{fmt}}")
        else:
          logging.info("    All measurements identical (no variance)")
        logging.info(f"    Mean {name1}: {summary.mean_a:{fmt}}")
        logging.info(f"    Mean {name2}: {summary.mean_b:{fmt}}")
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
      logging.info(f"    Mean difference: {summary.mean_difference:{fmt}}")
      logging.info(f"    95% CI difference: [{summary.ci_95_difference[0]:{fmt}}, {summary.ci_95_difference[1]:{fmt}}]")
      logging.info(f"    p-value (uncorrected): {summary.p_value:.6f}")
      logging.info(f"    Effect size (rank-biserial): {summary.effect_size:.4f}")
      logging.info(f"    Effect size interpretation: {_interpret_effect_size(summary.effect_size)}")

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

  pairwise_comps: list[dict[str, Any]] = []
  for i, comp in enumerate(comparisons):
    comparison_dict: dict[str, Any] = {
      "comparison": comp.comparison_label,
      "metric_name": comp.metric_name,
      "test_used": comp.test_used,
      "test_statistic": float(comp.test_statistic),
      "df1": comp.df1,
      "df2": comp.df2,
      "p_value": float(comp.p_value),
      "mean_difference": float(comp.mean_difference),
      "ci_95_lower": float(comp.ci_95_difference[0]),
      "ci_95_upper": float(comp.ci_95_difference[1]),
      "effect_size": float(comp.effect_size),
      "test_justification": comp.test_justification,
      "zero_differences_count": comp.zero_differences_count,
      "effective_sample_size": comp.effective_sample_size,
      "insufficient_power_warning": comp.insufficient_power_warning,
      "normality_p_value_a": float(comp.normality_p_value_a),
      "normality_p_value_b": float(comp.normality_p_value_b),
      "effect_interpretation": _interpret_effect_size(comp.effect_size),
    }

    if len(comparisons) > 1 and i < len(pvals_corrected):
      comparison_dict["p_value_corrected"] = float(pvals_corrected[i])
      comparison_dict["significant_after_correction"] = bool(reject[i])

    pairwise_comps.append(comparison_dict)

  return pairwise_comps


def perform_statistical_analysis(
  problem_name: str,
  optimal_cost: float | None,
  results: dict[str, dict[str, Any]],
  analyzer: StatisticalAnalyzer,
  short_sep: str = "-" * 80,
  baseline_algorithm: str | None = None,
  strict_lengths: bool = False,
) -> dict[str, Any]:
  """
  Perform statistical analysis on results for a single problem.

  Executes comprehensive statistical tests comparing multiple algorithms
  on the same problem instance across both solution cost and execution time:

  1. Friedman test (if k ≥ 3 algorithms)
  2. Nemenyi post-hoc test (if Friedman significant)
  3. Paired Wilcoxon signed-rank comparisons with Holm-Bonferroni correction,
     Hodges-Lehmann distribution-free 95% CI, and rank-biserial effect size
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
      strict_lengths: If True, raise ValueError whenever algorithm sample counts
          differ or when raw time arrays are missing/incompatible across variants.

  Returns:
      Dictionary containing statistical test results for cost and time
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

  # Validate paired sample lengths across algorithms without corrupting full per-algorithm means (Task 12.7 / Task 14.2)
  cost_lengths: list[int] = [len(d) for d in cost_data]
  paired_cost_data: list[NDArray[Any]] = cost_data
  if strict_lengths and any(length == 0 for length in cost_lengths):
    raise ValueError(f"Missing raw cost arrays across algorithms for problem '{problem_name}': lengths={cost_lengths}.")
  if len(set(cost_lengths)) > 1:
    min_cost_len: int = min(cost_lengths) if cost_lengths else 0
    if strict_lengths:
      raise ValueError(
        f"Mismatched cost sample counts across algorithms for problem '{problem_name}': lengths={cost_lengths}."
      )
    logging.warning(
      "Mismatched sample counts across algorithms for problem '%s': lengths=%s. "
      "Truncating to common minimum length %d for paired Friedman test.",
      problem_name,
      cost_lengths,
      min_cost_len,
    )
    paired_cost_data = [d[:min_cost_len] for d in cost_data]

  time_lengths: list[int] = [len(d) for d in time_data]
  all_have_times: bool = all(length > 0 for length in time_lengths)
  paired_time_data: list[NDArray[Any]] = time_data
  run_time_inference: bool = False
  if not all_have_times:
    if strict_lengths:
      raise ValueError(
        f"Incomplete raw time arrays across algorithms for problem '{problem_name}': lengths={time_lengths}."
      )
    if any(length > 0 for length in time_lengths):
      logging.warning(
        "Incomplete raw time sample counts across algorithms for problem '%s': lengths=%s. "
        "Skipping paired time inference.",
        problem_name,
        time_lengths,
      )
  else:
    if len(set(time_lengths)) > 1:
      min_time_len: int = min(time_lengths)
      if strict_lengths:
        raise ValueError(
          f"Mismatched time sample counts across algorithms for problem '{problem_name}': lengths={time_lengths}."
        )
      logging.warning(
        "Mismatched time sample counts across algorithms for problem '%s': lengths=%s. "
        "Truncating to common minimum length %d for paired time analysis.",
        problem_name,
        time_lengths,
        min_time_len,
      )
      paired_time_data = [d[:min_time_len] for d in time_data]
    if cost_lengths != time_lengths:
      if strict_lengths:
        raise ValueError(
          f"Incompatible cost vs time sample counts for problem '{problem_name}': "
          f"cost_lengths={cost_lengths}, time_lengths={time_lengths}."
        )
      logging.warning(
        "Cost vs time sample counts differ for problem '%s': cost_lengths=%s, time_lengths=%s.",
        problem_name,
        cost_lengths,
        time_lengths,
      )
    run_time_inference = True

  # =========================================================================
  # Friedman & Pairwise Inference: Cost
  # =========================================================================

  friedman_result, nemenyi_result, mean_ranks = _run_omnibus_friedman_nemenyi(
    problem_name=problem_name,
    metric_label="cost",
    series_data=paired_cost_data,
    algorithm_names=algorithm_names,
    analyzer=analyzer,
  )
  pairwise_comps = _run_pairwise_metric_comparisons(
    algorithm_names=algorithm_names,
    series_data=paired_cost_data,
    metric_name="cost",
    analyzer=analyzer,
    decimals=2,
  )

  # =========================================================================
  # Friedman & Pairwise Inference: Execution Time
  # =========================================================================

  friedman_time_result: dict[str, Any] | None = None
  nemenyi_time_result: NemenyiResult | None = None
  mean_ranks_time: dict[str, float] = {}
  pairwise_time_comps: list[dict[str, Any]] = []

  if run_time_inference:
    friedman_time_result, nemenyi_time_result, mean_ranks_time = _run_omnibus_friedman_nemenyi(
      problem_name=problem_name,
      metric_label="time",
      series_data=paired_time_data,
      algorithm_names=algorithm_names,
      analyzer=analyzer,
    )
    pairwise_time_comps = _run_pairwise_metric_comparisons(
      algorithm_names=algorithm_names,
      series_data=paired_time_data,
      metric_name="time",
      analyzer=analyzer,
      decimals=4,
    )

  # =========================================================================
  # Summary Table
  # =========================================================================

  opt_label = f"{optimal_cost}" if optimal_cost is not None else "unknown"
  logging.info(f"\nSummary: {problem_name} (optimal={opt_label})")
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
    gap_cell = f"{m_gap:<12.2f}" if m_gap is not None else f"{'—':<12}"
    row_fmt: str = (
      f"{name:<20} "
      f"{m_cost:<15.2f} "
      f"{gap_cell} "
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
    f_dict, n_dict = _serialize_friedman_section(friedman_result, nemenyi_result, mean_ranks)
    statistical_results["friedman_test"] = f_dict
    if n_dict is not None:
      statistical_results["nemenyi_posthoc"] = n_dict

  statistical_results["pairwise_comparisons"] = pairwise_comps

  if friedman_time_result:
    ft_dict, nt_dict = _serialize_friedman_section(friedman_time_result, nemenyi_time_result, mean_ranks_time)
    statistical_results["friedman_test_time"] = ft_dict
    if nt_dict is not None:
      statistical_results["nemenyi_posthoc_time"] = nt_dict

  statistical_results["pairwise_comparisons_time"] = pairwise_time_comps
  return statistical_results
