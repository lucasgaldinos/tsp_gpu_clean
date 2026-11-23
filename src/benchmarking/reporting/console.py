"""Console log renderers for ``ProblemAnalysis`` and ``CohortAnalysis`` snapshots.

Separates human-readable terminal reporting from pure statistical computation so
``benchmarking.analysis`` remains side-effect-free while CLIs and orchestration
retain rich structured progress logs.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from benchmarking.analysis.cohort_analysis import CohortAnalysis, StratumAnalysis
  from benchmarking.analysis.problem_statistics import PairwiseComparisonRecord, ProblemAnalysis
  from benchmarking.analysis.statistics import OmnibusTestResult


def _log_omnibus(metric_label: str, omnibus: OmnibusTestResult) -> None:
  """Emit Friedman, Iman-Davenport F_F, ranks, and Nemenyi summary lines."""
  logging.info(
    "Friedman Test [%s] (k=%d algorithms, N=%d blocks):",
    metric_label,
    omnibus.k_algorithms,
    omnibus.n_blocks,
  )
  logging.info("  Chi-square: %.4f (p=%.6f)", omnibus.chi2_stat, omnibus.chi2_pvalue)
  logging.info(
    "  F_F(%d, %d): %.4f (p=%.6f)",
    omnibus.df1,
    omnibus.df2,
    omnibus.friedman_f,
    omnibus.friedman_pvalue,
  )
  logging.info("  Significant (alpha=%.3g): %s", omnibus.alpha, omnibus.omnibus_significant)
  if omnibus.low_power:
    logging.info("  Advisory: low block count (N=%d < 5); asymptotic F_F has limited power.", omnibus.n_blocks)
  logging.info(
    "  Mean ranks: %s",
    ", ".join(f"{name}={rank:.2f}" for name, rank in omnibus.ranks.items()),
  )
  logging.info("  Nemenyi CD: %.4f", omnibus.critical_difference)
  if omnibus.omnibus_significant:
    if omnibus.significant_pairs:
      logging.info("  Significant Nemenyi pairs:")
      for name_i, name_j, p_val in omnibus.significant_pairs:
        rank_diff = abs(omnibus.ranks[name_i] - omnibus.ranks[name_j])
        logging.info("    %s vs %s: rank_diff=%.2f, p=%.4f", name_i, name_j, rank_diff, p_val)
    else:
      logging.info("  Significant Nemenyi pairs: None")


def _fmt_normality_line(p_val: float, alpha: float) -> str:
  """Format a descriptive Shapiro-Wilk normality p-value cleanly."""
  if not math.isfinite(p_val):
    return "N/A (untested or zero variance)"
  label = "(normal)" if p_val >= alpha else "(non-normal)"
  return f"p={p_val:.4f} {label}"


def _log_pairwise_section(
  metric_name: str,
  records: tuple[PairwiseComparisonRecord, ...],
  alpha: float,
  decimals: int,
) -> None:
  """Emit paired Wilcoxon + Holm-Bonferroni comparison details."""
  if not records:
    return

  fmt = f".{decimals}f"
  logging.info("\nPairwise Comparisons (%s, Wilcoxon + Holm-Bonferroni, alpha=%.3g):", metric_name, alpha)
  for rec in records:
    comp = rec.summary
    logging.info("  %s:", comp.comparison_label)
    if comp.test_used == "no_test_needed":
      logging.info("    No statistical test needed (%s)", comp.test_justification)
      logging.info(f"    Mean A={comp.mean_a:{fmt}}, Mean B={comp.mean_b:{fmt}}")
      continue

    logging.info("    Normality (Shapiro-Wilk, descriptive):")
    logging.info(
      "      A: %s | B: %s",
      _fmt_normality_line(comp.normality_p_value_a, alpha),
      _fmt_normality_line(comp.normality_p_value_b, alpha),
    )
    logging.info(f"    Mean difference: {comp.mean_difference:{fmt}}")
    lo, hi = comp.ci_95_difference
    if math.isfinite(lo) and math.isfinite(hi):
      logging.info(f"    {round((1.0 - alpha) * 100)}% Hodges-Lehmann CI: [{lo:{fmt}}, {hi:{fmt}}]")
    else:
      logging.info("    Hodges-Lehmann CI: N/A (insufficient sample size for requested coverage)")
    logging.info(
      "    p-value (raw)=%.6f | rank-biserial r=%.4f (%s)",
      comp.p_value,
      comp.effect_size,
      rec.effect_interpretation,
    )
    if rec.p_value_corrected is not None and rec.significant_after_correction is not None:
      logging.info(
        "    Holm-Bonferroni: p_adj=%.6f | significant=%s",
        rec.p_value_corrected,
        rec.significant_after_correction,
      )


def log_problem_analysis(
  analysis: ProblemAnalysis,
  *,
  short_sep: str = "-" * 80,
) -> None:
  """Log a formatted per-problem statistical report from a ``ProblemAnalysis`` snapshot."""
  logging.info("\nStatistical Analysis: %s", analysis.problem_name)
  logging.info(short_sep)

  if analysis.insufficient_data:
    logging.info(
      "Insufficient algorithms (%d) for statistical analysis on %s.",
      len(analysis.algorithms),
      analysis.problem_name,
    )
    return

  if analysis.cost_omnibus is not None:
    _log_omnibus("cost", analysis.cost_omnibus)
  _log_pairwise_section("cost", analysis.cost_pairwise, analysis.alpha, decimals=2)

  if analysis.time_omnibus is not None:
    _log_omnibus("time", analysis.time_omnibus)
  _log_pairwise_section("time", analysis.time_pairwise, analysis.alpha, decimals=4)

  opt_label = f"{analysis.optimal_cost}" if analysis.optimal_cost is not None else "unknown"
  logging.info("\nSummary: %s (optimal=%s)", analysis.problem_name, opt_label)
  logging.info(short_sep)

  show_gpu_transfers = any(row.mean_h2d_mb > 0.0 or row.mean_d2h_mb > 0.0 for row in analysis.rows)
  if show_gpu_transfers:
    logging.info(
      f"{'Algorithm':<20} {'Mean Cost':<15} {'Gap (%)':<12} {'Time (s)':<12} {'H2D (MB)':<12} {'D2H (MB)':<12}"
    )
  else:
    logging.info(f"{'Algorithm':<20} {'Mean Cost':<15} {'Gap (%)':<12} {'Time (s)':<12}")
  logging.info(short_sep)

  for row in analysis.rows:
    gap_cell = f"{row.mean_gap:<12.2f}" if row.mean_gap is not None else f"{'—':<12}"
    if show_gpu_transfers:
      logging.info(
        f"{row.algorithm:<20} {row.mean_cost:<15.2f} {gap_cell} {row.mean_time:<12.2f} "
        f"{row.mean_h2d_mb:<12.2f} {row.mean_d2h_mb:<12.2f}"
      )
    else:
      logging.info(f"{row.algorithm:<20} {row.mean_cost:<15.2f} {gap_cell} {row.mean_time:<12.2f}")

  logging.info("\nSpeedups (relative to %s):", analysis.baseline_algorithm)
  for row in analysis.rows:
    if row.algorithm == analysis.baseline_algorithm:
      logging.info("  %s: 1.00x (baseline)", row.algorithm)
    else:
      logging.info("  %s: %.2fx vs %s", row.algorithm, row.speedup, analysis.baseline_algorithm)
  logging.info("")


def _log_stratum(stratum: StratumAnalysis, separator: str) -> None:
  """Log one stratified subset comparison."""
  logging.info(separator)
  logging.info(stratum.label)
  logging.info(separator)
  if stratum.omnibus is None:
    logging.info(
      "Skipped (need >= 3 algorithms and >= 2 complete problems; found k=%d, N=%d).",
      len(stratum.algorithms),
      len(stratum.problems),
    )
    return

  unit = "%" if stratum.metric_key == "mean_gap" else ("s" if stratum.metric_key == "mean_time" else "")
  logging.info(
    "Problems (%d): %s",
    len(stratum.problems),
    ", ".join(stratum.problems[:5]) + ("..." if len(stratum.problems) > 5 else ""),
  )
  logging.info("Algorithms (%d): %s", len(stratum.algorithms), ", ".join(stratum.algorithms))
  for algo, (mean_val, std_val) in stratum.metric_means.items():
    logging.info("  %s: %s=%.2f%s ± %.2f%s", algo, stratum.metric_key, mean_val, unit, std_val, unit)
  if stratum.speedups and stratum.baseline_algorithm:
    logging.info("Geometric mean speedup (vs %s):", stratum.baseline_algorithm)
    for algo, sp in stratum.speedups.items():
      if sp.geomean is not None:
        logging.info("  %s: %.2fx (n=%d)", algo, sp.geomean, sp.sample_size)
  _log_omnibus(stratum.metric_key, stratum.omnibus)
  logging.info("")


def log_cohort_analysis(
  analysis: CohortAnalysis,
  *,
  separator: str = "=" * 80,
) -> None:
  """Log a formatted cross-problem report from a precomputed ``CohortAnalysis`` snapshot."""
  cohort = analysis.cohort
  logging.info(separator)
  if analysis.cpu_threshold is not None:
    logging.info("CROSS-PROBLEM ANALYSIS (STRATIFIED, n <= %d)", analysis.cpu_threshold)
  else:
    logging.info("CROSS-PROBLEM ANALYSIS")
  logging.info(separator)
  logging.info(
    "Problems analyzed: %d | Algorithms: %s | Baseline: %s",
    len(cohort.problems),
    ", ".join(cohort.algorithms),
    analysis.baseline_algorithm,
  )
  logging.info("")

  if analysis.strata:
    for stratum in analysis.strata:
      _log_stratum(stratum, separator)
  else:
    if analysis.quality_omnibus is not None:
      _log_omnibus(analysis.quality_metric_key, analysis.quality_omnibus)
      logging.info("")
    if analysis.time_omnibus is not None:
      _log_omnibus("mean_time", analysis.time_omnibus)
      logging.info("")

  logging.info("Geometric mean speedups (vs %s):", analysis.baseline_algorithm)
  for alg in cohort.algorithms:
    sp = analysis.baseline_speedups.get(alg)
    if alg == analysis.baseline_algorithm:
      logging.info("  %s: 1.00x (baseline)", alg)
    elif sp is not None and sp.geomean is not None:
      logging.info("  %s: %.2fx (n=%d)", alg, sp.geomean, sp.sample_size)
  logging.info("")

  logging.info("Aggregated Per-Algorithm Summary:")
  for alg in cohort.algorithms:
    agg = analysis.per_algorithm.get(alg)
    if agg is None:
      continue
    gap_str = (
      f"mean_gap={agg.mean_gap:.2f}% ± {agg.std_gap:.2f}%"
      if (agg.mean_gap is not None and agg.std_gap is not None)
      else (
        f"mean_cost={agg.mean_cost:.2f} ± {agg.std_cost:.2f}"
        if (agg.mean_cost is not None and agg.std_cost is not None)
        else "quality=N/A"
      )
    )
    time_str = (
      f"mean_time={agg.mean_time:.2f}s ± {agg.std_time:.2f}s"
      if (agg.mean_time is not None and agg.std_time is not None)
      else "mean_time=N/A"
    )
    logging.info("  %s: %s | %s (n=%d problems)", alg, gap_str, time_str, agg.n_problems)
  logging.info("")


__all__ = [
  "log_cohort_analysis",
  "log_problem_analysis",
]
