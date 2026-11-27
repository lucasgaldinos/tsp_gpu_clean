"""Single-pass cross-problem statistical cohort analysis.

Computes all cohort-level statistical summaries once from a ``BenchmarkCohort``:
- Per-algorithm aggregated means and sample SDs across instances
- Paired Fleming & Wallace (1986) geometric mean speedups (baseline and all ordered pairs)
- Unified Friedman + Iman-Davenport F_F + Nemenyi CD + cliques + pairwise p-values for
  both solution quality and execution time (enforcing ``k >= 3``, ``N >= 2``, ``low_power``
  when ``N < 5``, and explicit ``omnibus_significant`` gating)
- Paired Wilcoxon signed-rank + Holm-Bonferroni comparisons against the baseline
- Optional size-stratified analysis when a ``cpu_threshold`` cutoff is supplied

Every reporting surface (console logging, Markdown/LaTeX tables, quantitative LaTeX
macros, and Demšar TikZ CD diagrams) consumes the resulting ``CohortAnalysis`` directly
without recomputing any statistical test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.cohort import (
  AlgorithmAggregate,
  BenchmarkCohort,
  SpeedupSummary,
)
from benchmarking.analysis.statistics import (
  OmnibusTestResult,
  StatisticalAnalyzer,
  interpret_rank_biserial,
  run_friedman_nemenyi_analysis,
)
from benchmarking.analysis.stats_utils import holm_bonferroni_correction

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from benchmarking.runner.config_loader import ProblemConfig


@dataclass(frozen=True)
class BaselineWilcoxonComparison:
  """Paired cross-problem Wilcoxon + Holm-Bonferroni result against the baseline."""

  algorithm: str
  baseline: str
  metric_key: str
  n_paired: int
  w_statistic: float
  p_value: float
  p_value_holm: float
  significant_holm: bool
  rank_biserial_r: float
  effect_interpretation: str


@dataclass(frozen=True)
class StratumAnalysis:
  """Omnibus comparison and speedups over a designated subset of problems and algorithms."""

  label: str
  problems: tuple[str, ...]
  algorithms: tuple[str, ...]
  metric_key: str
  higher_is_better: bool
  metric_means: dict[str, tuple[float, float]]
  omnibus: OmnibusTestResult | None
  speedups: dict[str, SpeedupSummary]
  baseline_algorithm: str | None = None


@dataclass(frozen=True)
class CohortAnalysis:
  """Immutable single-pass statistical analysis across an entire benchmark cohort."""

  cohort: BenchmarkCohort
  baseline_algorithm: str
  alpha: float
  quality_metric_key: str
  cpu_threshold: int | None
  per_algorithm: dict[str, AlgorithmAggregate]
  baseline_speedups: dict[str, SpeedupSummary]
  pairwise_speedups: dict[tuple[str, str], SpeedupSummary]
  quality_omnibus: OmnibusTestResult | None
  time_omnibus: OmnibusTestResult | None
  quality_wilcoxon_vs_baseline: tuple[BaselineWilcoxonComparison, ...]
  time_wilcoxon_vs_baseline: tuple[BaselineWilcoxonComparison, ...]
  strata: tuple[StratumAnalysis, ...] = ()


def _compute_baseline_wilcoxon(
  cohort: BenchmarkCohort,
  baseline: str,
  metric_key: str,
  analyzer: StatisticalAnalyzer,
) -> tuple[BaselineWilcoxonComparison, ...]:
  """Run paired Wilcoxon + Holm-Bonferroni comparing every non-baseline algorithm to baseline."""
  non_baseline = [alg for alg in cohort.algorithms if alg != baseline]
  if baseline not in cohort.algorithms or not non_baseline or len(cohort.problems) < 2:
    return ()

  raw_records: list[tuple[str, int, float, float, float]] = []
  for cand in non_baseline:
    paired_probs, base_vals, cand_vals = cohort.paired_quality_vectors(baseline, cand, metric_key)
    if len(paired_probs) < 2:
      continue
    w_stat, p_val, r_eff = analyzer.wilcoxon_paired_summary(base_vals, cand_vals)
    raw_records.append((cand, len(paired_probs), w_stat, p_val, r_eff))

  if not raw_records:
    return ()

  reject, holm_p_vals = holm_bonferroni_correction([rec[3] for rec in raw_records], alpha=analyzer.alpha)
  return tuple(
    BaselineWilcoxonComparison(
      algorithm=cand,
      baseline=baseline,
      metric_key=metric_key,
      n_paired=n_paired,
      w_statistic=w_stat,
      p_value=p_val,
      p_value_holm=p_holm,
      significant_holm=is_sig,
      rank_biserial_r=r_eff,
      effect_interpretation=interpret_rank_biserial(r_eff),
    )
    for (cand, n_paired, w_stat, p_val, r_eff), p_holm, is_sig in zip(
      raw_records,
      holm_p_vals,
      reject,
      strict=True,
    )
  )


def _compute_stratum(
  cohort: BenchmarkCohort,
  *,
  label: str,
  problems: Sequence[str],
  algorithms: Sequence[str],
  metric_key: str,
  alpha: float,
  baseline_algorithm: str | None = None,
  higher_is_better: bool = False,
) -> StratumAnalysis:
  """Compute descriptive means, geometric speedups, and unified omnibus stats for a subset."""
  algo_list = list(algorithms)
  valid_problems, matrix = cohort.quality_score_matrix(metric_key, algorithms=algo_list, problems=problems)

  metric_means: dict[str, tuple[float, float]] = {}
  if valid_problems:
    has_infeas = metric_key in ("mean_cost", "mean_gap") and cohort.has_infeasibility(
      algorithms=algo_list, problems=valid_problems
    )
    if not has_infeas:
      for idx, algo in enumerate(algo_list):
        col = matrix[:, idx]
        std_val = float(np.std(col, ddof=1)) if col.size > 1 else 0.0
        metric_means[algo] = (float(np.mean(col)), std_val)
    else:
      for algo in algo_list:
        vals = cohort.algorithm_metric_values(algo, metric_key, problems=valid_problems)
        if vals:
          arr = np.asarray(vals, dtype=np.float64)
          std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
          metric_means[algo] = (float(np.mean(arr)), std_val)
        else:
          metric_means[algo] = (float("nan"), float("nan"))

  speedups: dict[str, SpeedupSummary] = {}
  if baseline_algorithm and baseline_algorithm in algo_list and valid_problems:
    for algo in algo_list:
      if algo == baseline_algorithm:
        speedups[algo] = SpeedupSummary(geomean=1.0, sample_size=len(valid_problems))
      else:
        speedups[algo] = cohort.paired_geometric_speedup(algo, baseline_algorithm, problems=valid_problems)

  omnibus: OmnibusTestResult | None = None
  if len(algo_list) >= 3 and len(valid_problems) >= 2:
    omnibus = run_friedman_nemenyi_analysis(
      matrix,
      algo_list,
      alpha=alpha,
      higher_is_better=higher_is_better,
    )

  return StratumAnalysis(
    label=label,
    problems=tuple(valid_problems),
    algorithms=tuple(algo_list),
    metric_key=metric_key,
    higher_is_better=higher_is_better,
    metric_means=metric_means,
    omnibus=omnibus,
    speedups=speedups,
    baseline_algorithm=baseline_algorithm if (baseline_algorithm in algo_list) else None,
  )


def compare_algorithms(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problems: Sequence[str],
  algorithms: Sequence[str],
  *,
  analyzer: StatisticalAnalyzer | None = None,
  baseline_algorithm: str | None = None,
  label: str = "Algorithm Comparison",
  metric_key: str = "mean_gap",
  higher_is_better: bool = False,
  cohort: BenchmarkCohort | None = None,
) -> StratumAnalysis | None:
  """Compare a subset of algorithms across a designated subset of problems.

  Enforces the unified ``k >= 3`` and ``N >= 2`` omnibus contract (returning ``None``
  when fewer than 3 algorithms or fewer than 2 complete problems are available).
  """
  active_cohort = cohort or BenchmarkCohort.from_results(all_results, sort_problems=False)
  alpha = analyzer.alpha if analyzer is not None else 0.05
  stratum = _compute_stratum(
    active_cohort,
    label=label,
    problems=problems,
    algorithms=algorithms,
    metric_key=metric_key,
    alpha=alpha,
    baseline_algorithm=baseline_algorithm,
    higher_is_better=higher_is_better,
  )
  if stratum.omnibus is None:
    return None
  return stratum


def analyze_cohort(
  cohort_or_results: BenchmarkCohort | Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None = None,
  *,
  baseline_algorithm: str | None = None,
  alpha: float = 0.05,
  cpu_threshold: int | None = None,
  algorithm_order: Sequence[str] | None = None,
) -> CohortAnalysis:
  """Run complete single-pass cross-problem statistical analysis for a benchmark cohort.

  Args:
      cohort_or_results: Either a prebuilt ``BenchmarkCohort`` or a raw nested results mapping.
      problem_configs: Optional problem metadata configs (used only when building a cohort
          from raw ``all_results``).
      baseline_algorithm: Optional baseline algorithm for speedups and Wilcoxon vs. baseline.
      alpha: Significance level for Friedman/Iman-Davenport, Nemenyi, and Holm-Bonferroni.
      cpu_threshold: Optional problem-size cutoff enabling stratified cross-problem analysis
          (Stratum 1: ``size <= cpu_threshold``; Stratum 2: full-coverage algorithms).
      algorithm_order: Optional explicit algorithm display order when constructing a cohort.

  Returns:
      Immutable ``CohortAnalysis`` consumed by console logging, tables, macros, and CD diagrams.
  """
  cohort = (
    cohort_or_results
    if isinstance(cohort_or_results, BenchmarkCohort)
    else BenchmarkCohort.from_results(
      cohort_or_results,
      problem_configs,
      algorithm_order=algorithm_order,
    )
  )
  effective_baseline = cohort.baseline(baseline_algorithm)
  quality_metric_key = cohort.quality_metric_key()
  analyzer = StatisticalAnalyzer(alpha=alpha)
  algorithms = list(cohort.algorithms)
  problems = list(cohort.problems)

  per_algorithm: dict[str, AlgorithmAggregate] = {}
  baseline_speedups: dict[str, SpeedupSummary] = {}
  for alg in algorithms:
    agg = cohort.summarize_algorithm(alg)
    if agg is not None:
      per_algorithm[alg] = agg
    if alg == effective_baseline:
      baseline_speedups[alg] = SpeedupSummary(
        geomean=1.0,
        sample_size=agg.n_problems if agg is not None else 0,
      )
    else:
      baseline_speedups[alg] = cohort.paired_geometric_speedup(alg, effective_baseline)

  pairwise_speedups: dict[tuple[str, str], SpeedupSummary] = {}
  for alg_a in algorithms:
    for alg_b in algorithms:
      if alg_a != alg_b:
        pairwise_speedups[(alg_a, alg_b)] = cohort.paired_geometric_speedup(alg_a, alg_b)

  quality_omnibus: OmnibusTestResult | None = None
  time_omnibus: OmnibusTestResult | None = None
  if len(algorithms) >= 3 and len(problems) >= 2:
    valid_q, mat_q = cohort.quality_score_matrix(quality_metric_key)
    if len(valid_q) >= 2:
      quality_omnibus = run_friedman_nemenyi_analysis(mat_q, algorithms, alpha=alpha, higher_is_better=False)

    valid_t, mat_t = cohort.metric_matrix("mean_time")
    if len(valid_t) >= 2:
      time_omnibus = run_friedman_nemenyi_analysis(mat_t, algorithms, alpha=alpha, higher_is_better=False)

  quality_wilcoxon = _compute_baseline_wilcoxon(cohort, effective_baseline, quality_metric_key, analyzer)
  time_wilcoxon = _compute_baseline_wilcoxon(cohort, effective_baseline, "mean_time", analyzer)

  strata: list[StratumAnalysis] = []
  if cpu_threshold is not None:
    small_problems = [p for p in problems if (cohort.problem_size(p) or 0) <= cpu_threshold]
    small_algos = [
      a for a in algorithms if small_problems and all(a in cohort.results.get(sp, {}) for sp in small_problems)
    ]
    stratum1_baseline = (
      effective_baseline if effective_baseline in small_algos else (small_algos[0] if small_algos else None)
    )
    stratum1_metric = cohort.quality_metric_key(algorithms=small_algos, problems=small_problems, require_complete=False)
    strata.append(
      _compute_stratum(
        cohort,
        label=f"STRATUM 1: Small Problems (n <= {cpu_threshold})",
        problems=small_problems,
        algorithms=small_algos,
        metric_key=stratum1_metric,
        alpha=alpha,
        baseline_algorithm=stratum1_baseline,
      )
    )

    large_problems = [p for p in problems if (cohort.problem_size(p) or 0) > cpu_threshold]
    coverage_target = large_problems if large_problems else problems
    full_cov_algos = [a for a in algorithms if all(a in cohort.results.get(p, {}) for p in coverage_target)]
    stratum2_problems = cohort.complete_problems(algorithms=full_cov_algos, problems=problems)
    stratum2_baseline = (
      baseline_algorithm
      if baseline_algorithm and baseline_algorithm in full_cov_algos
      else (full_cov_algos[0] if full_cov_algos else None)
    )
    stratum2_metric = cohort.quality_metric_key(
      algorithms=full_cov_algos,
      problems=stratum2_problems,
      require_complete=False,
    )
    strata.append(
      _compute_stratum(
        cohort,
        label="STRATUM 2: Full-Coverage Algorithms (All Problems)",
        problems=stratum2_problems,
        algorithms=full_cov_algos,
        metric_key=stratum2_metric,
        alpha=alpha,
        baseline_algorithm=stratum2_baseline,
      )
    )

  return CohortAnalysis(
    cohort=cohort,
    baseline_algorithm=effective_baseline,
    alpha=alpha,
    quality_metric_key=quality_metric_key,
    cpu_threshold=cpu_threshold,
    per_algorithm=per_algorithm,
    baseline_speedups=baseline_speedups,
    pairwise_speedups=pairwise_speedups,
    quality_omnibus=quality_omnibus,
    time_omnibus=time_omnibus,
    quality_wilcoxon_vs_baseline=quality_wilcoxon,
    time_wilcoxon_vs_baseline=time_wilcoxon,
    strata=tuple(strata),
  )


__all__ = [
  "BaselineWilcoxonComparison",
  "CohortAnalysis",
  "StratumAnalysis",
  "analyze_cohort",
  "compare_algorithms",
]
