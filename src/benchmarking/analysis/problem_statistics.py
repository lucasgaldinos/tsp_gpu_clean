"""Pure per-problem statistical analysis over paired algorithm replications.

Analyzes multiple algorithms on a single TSP instance across paired runs:
- Omnibus Friedman / Iman-Davenport F_F + Nemenyi post-hoc (when k >= 3 and n >= 2)
- All-pairs Wilcoxon signed-rank tests + Holm-Bonferroni FWER adjustment (using analyzer.alpha)
- Matched-pairs rank-biserial correlation with correlation-scale interpretation (Kerby 2014)
- Per-algorithm descriptive means and baseline speedups

This module performs pure computation with no console formatting or file I/O;
use ``benchmarking.reporting.console.log_problem_analysis`` for terminal output
and ``ProblemAnalysis.to_dict()`` for JSON checkpoint persistence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.constrained_statistics import (
  compute_deb_seed_rank_matrix,
  constrained_paired_comparison,
  extract_entry_normalized_violations,
)
from benchmarking.analysis.statistics import (
  OmnibusTestResult,
  StatisticalAnalyzer,
  StatisticalSummary,
  build_feasibility_aware_scores,
  interpret_rank_biserial,
  run_friedman_nemenyi_analysis,
)
from benchmarking.analysis.stats_utils import (
  finite_metric,
  holm_bonferroni_correction,
  resolve_baseline_algorithm,
)

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from numpy.typing import NDArray


def _finite_or_none(value: float) -> float | None:
  """Return ``float(value)`` when finite, else ``None`` for RFC-8259-safe JSON."""
  return float(value) if math.isfinite(value) else None


def _get_data_field(results_dict: Mapping[str, Any], field: str) -> list[Any]:
  """Extract raw repetition series with fallback for checkpoint vs. direct format.

  Handles two formats:
  - Checkpoint format: ``raw_costs``, ``raw_times``, ``raw_gaps``
  - Direct format: ``final_costs``, ``times``, ``gaps``
  """
  raw_field = f"raw_{field}"
  direct_field = "final_costs" if field == "costs" else field
  val = results_dict.get(raw_field)
  if val is None:
    val = results_dict.get(direct_field)
  if val is None and field == "costs":
    val = results_dict.get("raw_unpenalized_costs")
  if val is None:
    return []
  return val if isinstance(val, list) else list(val)


def _resolve_metric_mean(
  results_dict: Mapping[str, Any],
  metric_key: str,
  raw_series: Sequence[float] | np.ndarray,
  alg_name: str,
) -> float:
  """Resolve mean metric lazily from precomputed summary or raw finite series."""
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
  results_dict: Mapping[str, Any],
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


@dataclass(frozen=True)
class AlgorithmProblemRow:
  """Per-algorithm descriptive summary on a single benchmark instance."""

  algorithm: str
  mean_cost: float | None
  mean_gap: float | None
  mean_time: float
  speedup: float
  mean_h2d_mb: float = 0.0
  mean_d2h_mb: float = 0.0


@dataclass(frozen=True)
class PairwiseComparisonRecord:
  """One paired Wilcoxon comparison plus optional Holm-Bonferroni adjustment."""

  summary: StatisticalSummary
  effect_interpretation: str
  p_value_corrected: float | None = None
  significant_after_correction: bool | None = None

  def to_dict(self) -> dict[str, Any]:
    """Serialize into a JSON-ready dictionary matching checkpoint statistics format."""
    comp = self.summary
    payload: dict[str, Any] = {
      "comparison": comp.comparison_label,
      "metric_name": comp.metric_name,
      "test_used": comp.test_used,
      "test_statistic": float(comp.test_statistic),
      "df1": comp.df1,
      "df2": comp.df2,
      "p_value": float(comp.p_value),
      "mean_difference": _finite_or_none(comp.mean_difference),
      "ci_95_lower": _finite_or_none(comp.ci_95_difference[0]),
      "ci_95_upper": _finite_or_none(comp.ci_95_difference[1]),
      "effect_size": float(comp.effect_size),
      "test_justification": comp.test_justification,
      "zero_differences_count": comp.zero_differences_count,
      "effective_sample_size": comp.effective_sample_size,
      "insufficient_power_warning": comp.insufficient_power_warning,
      "normality_p_value_a": _finite_or_none(comp.normality_p_value_a),
      "normality_p_value_b": _finite_or_none(comp.normality_p_value_b),
      "effect_interpretation": self.effect_interpretation,
    }
    if comp.vargha_delaney_a12 is not None:
      payload["vargha_delaney_a12"] = float(comp.vargha_delaney_a12)
    if self.p_value_corrected is not None:
      payload["p_value_corrected"] = float(self.p_value_corrected)
    if self.significant_after_correction is not None:
      payload["significant_after_correction"] = bool(self.significant_after_correction)
    return payload


@dataclass(frozen=True)
class ProblemAnalysis:
  """Immutable statistical analysis result for one problem instance."""

  problem_name: str
  optimal_cost: float | None
  baseline_algorithm: str
  algorithms: tuple[str, ...]
  alpha: float
  insufficient_data: bool = False
  rows: tuple[AlgorithmProblemRow, ...] = ()
  cost_omnibus: OmnibusTestResult | None = None
  cost_pairwise: tuple[PairwiseComparisonRecord, ...] = ()
  time_omnibus: OmnibusTestResult | None = None
  time_pairwise: tuple[PairwiseComparisonRecord, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    """Serialize into the JSON-ready dict stored by ``save_problem_statistics``."""
    if self.insufficient_data:
      return {
        "problem_name": self.problem_name,
        "optimal_cost": self.optimal_cost,
        "error": "insufficient_data",
        "available_algorithms": list(self.algorithms),
      }

    payload: dict[str, Any] = {
      "problem_name": self.problem_name,
      "optimal_cost": self.optimal_cost,
      "baseline_algorithm": self.baseline_algorithm,
    }
    if self.cost_omnibus is not None:
      payload["friedman_test"] = self.cost_omnibus.to_friedman_dict()
      payload["nemenyi_posthoc"] = self.cost_omnibus.to_nemenyi_dict()

    payload["pairwise_comparisons"] = [rec.to_dict() for rec in self.cost_pairwise]

    if self.time_omnibus is not None:
      payload["friedman_test_time"] = self.time_omnibus.to_friedman_dict()
      payload["nemenyi_posthoc_time"] = self.time_omnibus.to_nemenyi_dict()

    payload["pairwise_comparisons_time"] = [rec.to_dict() for rec in self.time_pairwise]
    return payload


def _prepare_paired_problem_cost_inputs(
  results: Mapping[str, Mapping[str, Any]],
  algorithm_names: Sequence[str],
  paired_cost_data: list[NDArray[Any]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[NDArray[np.bool_]], bool]:
  """Extract aligned unpenalized costs, dimensionless ``hat_nu``, and feasibility masks.

  Returns ``(cost_cols, hat_nu_cols, feas_cols, has_infeasibility)``. When no VRP
  feasibility metadata is present or 100% of seeds are feasible, ``has_infeasibility``
  is ``False`` so callers execute the exact unconstrained Demšar (2006) path.
  """
  if not paired_cost_data:
    return [], [], [], False
  common_len = min((len(arr) for arr in paired_cost_data), default=0)
  if common_len == 0:
    empty_f = [np.empty(0, dtype=np.float64) for _ in algorithm_names]
    empty_b = [np.empty(0, dtype=bool) for _ in algorithm_names]
    return empty_f, empty_f, empty_b, False

  has_vrp_feasibility = any(
    results[name].get("raw_is_feasible") is not None for name in algorithm_names
  )
  if not has_vrp_feasibility:
    cost_cols = [np.asarray(arr[:common_len], dtype=np.float64) for arr in paired_cost_data]
    zero_viols = [np.zeros(common_len, dtype=np.float64) for _ in algorithm_names]
    all_feas = [np.ones(common_len, dtype=bool) for _ in algorithm_names]
    return cost_cols, zero_viols, all_feas, False

  cost_cols = []
  nu_cols: list[NDArray[np.float64]] = []
  feas_cols: list[NDArray[np.bool_]] = []

  for idx, name in enumerate(algorithm_names):
    entry = results[name]
    raw_unpen = entry.get("raw_unpenalized_costs")
    raw_feas = entry.get("raw_is_feasible")
    viols = extract_entry_normalized_violations(entry, expected_len=common_len)

    col_cost = (
      np.asarray(raw_unpen[:common_len], dtype=np.float64)
      if isinstance(raw_unpen, list) and len(raw_unpen) >= common_len
      else np.asarray(paired_cost_data[idx][:common_len], dtype=np.float64)
    )
    col_nu = (
      np.asarray(viols[:common_len], dtype=np.float64)
      if viols is not None
      else np.zeros(common_len, dtype=np.float64)
    )
    col_feas = (
      np.asarray(raw_feas[:common_len], dtype=bool)
      if isinstance(raw_feas, list) and len(raw_feas) >= common_len
      else np.ones(common_len, dtype=bool)
    )
    cost_cols.append(col_cost)
    nu_cols.append(col_nu)
    feas_cols.append(col_feas)

  has_infeas = not all(bool(np.all(col)) for col in feas_cols)
  return cost_cols, nu_cols, feas_cols, has_infeas


def _run_omnibus_if_eligible(
  *,
  series_data: list[NDArray[Any]],
  algorithm_names: list[str],
  alpha: float,
) -> OmnibusTestResult | None:
  """Run single-pass Friedman + Nemenyi when ``k >= 3`` and common ``n >= 2``."""
  if len(algorithm_names) < 3:
    return None
  min_len = min((len(d) for d in series_data), default=0)
  if min_len < 2:
    return None
  return run_friedman_nemenyi_analysis(series_data, algorithm_names, alpha=alpha, higher_is_better=False)


def _run_pairwise_comparisons(
  *,
  algorithm_names: list[str],
  series_data: list[NDArray[Any]],
  metric_name: str,
  analyzer: StatisticalAnalyzer,
  hat_nu_data: list[NDArray[np.float64]] | None = None,
  feasibility_data: list[NDArray[np.bool_]] | None = None,
) -> tuple[PairwiseComparisonRecord, ...]:
  """Run paired Wilcoxon (when 100% feasible) or constrained Binomial Sign Test + Holm-Bonferroni."""
  min_len = min((len(d) for d in series_data), default=0)
  if min_len < 2:
    return ()

  summaries: list[StatisticalSummary] = []
  for i, name1 in enumerate(algorithm_names):
    for j, name2 in enumerate(algorithm_names[i + 1 :], start=i + 1):
      if (
        feasibility_data is not None
        and hat_nu_data is not None
        and (not bool(np.all(feasibility_data[i])) or not bool(np.all(feasibility_data[j])))
      ):
        summaries.append(
          constrained_paired_comparison(
            series_data[i],
            hat_nu_data[i],
            feasibility_data[i],
            series_data[j],
            hat_nu_data[j],
            feasibility_data[j],
            label=f"{name1} vs {name2}",
            metric_name=metric_name,
            analyzer=analyzer,
          )
        )
      else:
        summaries.append(
          analyzer.paired_comparison(
            series_data[i],
            series_data[j],
            f"{name1} vs {name2}",
            metric_name=metric_name,
          )
        )

  reject: list[bool] = []
  pvals_corrected: list[float] = []
  if len(summaries) > 1:
    reject, pvals_corrected = holm_bonferroni_correction(
      [s.p_value for s in summaries],
      alpha=analyzer.alpha,
    )

  records: list[PairwiseComparisonRecord] = []
  for idx, summary in enumerate(summaries):
    records.append(
      PairwiseComparisonRecord(
        summary=summary,
        effect_interpretation=interpret_rank_biserial(summary.effect_size),
        p_value_corrected=pvals_corrected[idx] if len(summaries) > 1 else None,
        significant_after_correction=reject[idx] if len(summaries) > 1 else None,
      )
    )
  return tuple(records)


def analyze_problem(
  problem_name: str,
  optimal_cost: float | None,
  results: Mapping[str, Mapping[str, Any]],
  analyzer: StatisticalAnalyzer | None = None,
  *,
  baseline_algorithm: str | None = None,
  strict_lengths: bool = False,
) -> ProblemAnalysis:
  """Compute pure single-problem statistical analysis across multiple algorithms.

  Args:
      problem_name: Instance name (e.g., ``"berlin52"``).
      optimal_cost: Known optimal tour cost, or ``None`` if unknown.
      results: Mapping ``algorithm_name -> results_dict``.
      analyzer: Optional ``StatisticalAnalyzer`` (defaults to ``StatisticalAnalyzer(alpha=0.05)``).
      baseline_algorithm: Optional baseline algorithm name for per-problem speedup ratios.
      strict_lengths: If ``True``, raise ``ValueError`` when raw cost/time arrays are
          missing or have mismatched sample counts across algorithms.

  Returns:
      Immutable ``ProblemAnalysis`` containing omnibus tests, pairwise Wilcoxon + Holm
      comparisons, and per-algorithm descriptive rows.
  """
  active_analyzer = analyzer or StatisticalAnalyzer()
  algorithm_names = list(results.keys())

  if len(algorithm_names) < 2:
    logging.warning(
      "Skipping statistical analysis for %s: only %d algorithm(s) available (need >= 2).",
      problem_name,
      len(algorithm_names),
    )
    return ProblemAnalysis(
      problem_name=problem_name,
      optimal_cost=optimal_cost,
      baseline_algorithm=resolve_baseline_algorithm(algorithm_names, baseline_algorithm),
      algorithms=tuple(algorithm_names),
      alpha=active_analyzer.alpha,
      insufficient_data=True,
    )

  cost_data: list[NDArray[Any]] = [
    np.asarray(_get_data_field(results[name], "costs"), dtype=float) for name in algorithm_names
  ]
  time_data: list[NDArray[Any]] = [
    np.asarray(_get_data_field(results[name], "times"), dtype=float) for name in algorithm_names
  ]
  gap_data: list[NDArray[Any]] = [
    np.asarray(_get_data_field(results[name], "gaps"), dtype=float) for name in algorithm_names
  ]

  cost_lengths = [len(d) for d in cost_data]
  paired_cost_data = cost_data
  if strict_lengths and any(length == 0 for length in cost_lengths):
    raise ValueError(f"Missing raw cost arrays across algorithms for problem '{problem_name}': lengths={cost_lengths}.")
  if len(set(cost_lengths)) > 1:
    min_cost_len = min(cost_lengths) if cost_lengths else 0
    if strict_lengths:
      raise ValueError(
        f"Mismatched cost sample counts across algorithms for problem '{problem_name}': lengths={cost_lengths}."
      )
    logging.warning(
      "Mismatched sample counts across algorithms for problem '%s': lengths=%s. "
      "Truncating to common minimum length %d for paired cost analysis.",
      problem_name,
      cost_lengths,
      min_cost_len,
    )
    paired_cost_data = [d[:min_cost_len] for d in cost_data]

  time_lengths = [len(d) for d in time_data]
  all_have_times = all(length > 0 for length in time_lengths)
  paired_time_data = time_data
  run_time_inference = False
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
      min_time_len = min(time_lengths)
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

  cost_cols, nu_cols, feas_cols, has_infeasibility = _prepare_paired_problem_cost_inputs(
    results, algorithm_names, paired_cost_data
  )

  if has_infeasibility and len(cost_cols) >= 3 and min((len(c) for c in cost_cols), default=0) >= 2:
    deb_ranks = compute_deb_seed_rank_matrix(cost_cols, nu_cols, feas_cols)
    omnibus_series: list[NDArray[Any]] = [deb_ranks[:, j] for j in range(len(algorithm_names))]
  else:
    omnibus_series = list(cost_cols)

  cost_omnibus = _run_omnibus_if_eligible(
    series_data=omnibus_series,
    algorithm_names=algorithm_names,
    alpha=active_analyzer.alpha,
  )
  cost_pairwise = _run_pairwise_comparisons(
    algorithm_names=algorithm_names,
    series_data=list(cost_cols),
    metric_name="cost",
    analyzer=active_analyzer,
    hat_nu_data=nu_cols if has_infeasibility else None,
    feasibility_data=feas_cols if has_infeasibility else None,
  )

  time_omnibus: OmnibusTestResult | None = None
  time_pairwise: tuple[PairwiseComparisonRecord, ...] = ()
  if run_time_inference:
    time_omnibus = _run_omnibus_if_eligible(
      series_data=paired_time_data,
      algorithm_names=algorithm_names,
      alpha=active_analyzer.alpha,
    )
    time_pairwise = _run_pairwise_comparisons(
      algorithm_names=algorithm_names,
      series_data=paired_time_data,
      metric_name="time",
      analyzer=active_analyzer,
    )

  baseline_name = resolve_baseline_algorithm(algorithm_names, baseline_algorithm)
  base_idx = algorithm_names.index(baseline_name)
  baseline_time = _resolve_metric_mean(results[baseline_name], "mean_time", time_data[base_idx], baseline_name)

  rows: list[AlgorithmProblemRow] = []
  for idx, name in enumerate(algorithm_names):
    entry = results[name]
    raw_feas = entry.get("raw_is_feasible")
    m_cost: float | None
    m_gap: float | None
    if isinstance(raw_feas, list) and raw_feas:
      feas_mask = np.asarray(raw_feas, dtype=bool)
      unpen_raw = entry.get("raw_unpenalized_costs")
      cost_src = (
        np.asarray(unpen_raw, dtype=np.float64)
        if isinstance(unpen_raw, list) and len(unpen_raw) == len(feas_mask)
        else cost_data[idx]
      )
      if not bool(np.any(feas_mask)):
        m_cost, m_gap = None, None
      else:
        m_cost = finite_metric(entry.get("mean_cost"))
        if m_cost is None and len(cost_src) == len(feas_mask):
          valid_c = cost_src[feas_mask]
          valid_c = valid_c[np.isfinite(valid_c)]
          m_cost = (
            float(np.mean(valid_c))
            if valid_c.size > 0
            else _resolve_metric_mean(entry, "mean_cost", cost_data[idx], name)
          )
        elif m_cost is None:
          m_cost = _resolve_metric_mean(entry, "mean_cost", cost_data[idx], name)
        m_gap = finite_metric(entry.get("mean_gap"))
        if m_gap is None and optimal_cost is not None and optimal_cost > 0:
          m_gap = float((m_cost - optimal_cost) / optimal_cost * 100.0)
        elif m_gap is None and len(gap_data[idx]) == len(feas_mask):
          valid_g = gap_data[idx][feas_mask]
          valid_g = valid_g[np.isfinite(valid_g)]
          m_gap = float(np.mean(valid_g)) if valid_g.size > 0 else None
    elif entry.get("feasible_runs") == 0 and entry.get("mean_cost") is None:
      m_cost, m_gap = None, None
    else:
      m_cost = _resolve_metric_mean(entry, "mean_cost", cost_data[idx], name)
      m_gap = _resolve_gap_mean(entry, m_cost, optimal_cost, gap_data[idx], name)
    m_time = _resolve_metric_mean(entry, "mean_time", time_data[idx], name)
    speedup = (
      1.0 if name == baseline_name else ((baseline_time / m_time) if (m_time > 0.0 and baseline_time > 0.0) else 1.0)
    )
    rows.append(
      AlgorithmProblemRow(
        algorithm=name,
        mean_cost=m_cost,
        mean_gap=m_gap,
        mean_time=m_time,
        speedup=speedup,
        mean_h2d_mb=float(finite_metric(entry.get("mean_h2d_mb")) or 0.0),
        mean_d2h_mb=float(finite_metric(entry.get("mean_d2h_mb")) or 0.0),
      )
    )

  return ProblemAnalysis(
    problem_name=problem_name,
    optimal_cost=optimal_cost,
    baseline_algorithm=baseline_name,
    algorithms=tuple(algorithm_names),
    alpha=active_analyzer.alpha,
    insufficient_data=False,
    rows=tuple(rows),
    cost_omnibus=cost_omnibus,
    cost_pairwise=cost_pairwise,
    time_omnibus=time_omnibus,
    time_pairwise=time_pairwise,
  )


__all__ = [
  "AlgorithmProblemRow",
  "PairwiseComparisonRecord",
  "ProblemAnalysis",
  "analyze_problem",
  "build_feasibility_aware_scores",
]
