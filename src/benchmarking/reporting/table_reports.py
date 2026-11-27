"""Markdown and LaTeX benchmark summary table generators.

Provides independent, composable functions (``generate_markdown_tables`` and
``generate_latex_tables``) backed by ``BenchmarkCohort`` and parameterized by
``ReportStyle``. Row/aggregate computation lives here, while Markdown and LaTeX
rendering are delegated to ``table_markdown`` and ``table_latex``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.cohort import BenchmarkCohort, resolve_algorithm_names
from benchmarking.analysis.stats_utils import finite_metric
from benchmarking.reporting.style import ReportStyle, categorize_by_size
from benchmarking.reporting.table_latex import (
  render_phase_timings_latex,
  render_table1_latex,
  render_table2_latex,
  render_table3_latex,
)
from benchmarking.reporting.table_markdown import (
  render_phase_timings_markdown,
  render_table1_markdown,
  render_table2_markdown,
  render_table3_markdown,
)

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from benchmarking.analysis.cohort_analysis import CohortAnalysis
  from benchmarking.runner.config_loader import ProblemConfig

_PHASE_NAMES: tuple[str, ...] = (
  "initialization",
  "selection",
  "crossover",
  "mutation",
  "evaluation",
  "local_search",
  "survival",
  "host_sync",
)


@dataclass(frozen=True)
class Table1Row:
  """Computed row for Table 1 (per-problem summary statistics)."""

  problem: str
  algorithm: str
  mean_cost: float | None
  std_cost: float | None
  mean_gap: float | None
  mean_time: float
  speedup: float | None
  is_baseline: bool
  is_fallback: bool
  problem_row_index: int
  problem_row_span: int
  is_last_in_problem: bool
  is_last_problem: bool
  feasibility_rate: float | None = None
  mean_vehicles: float | None = None


@dataclass(frozen=True)
class PhaseTimingRow:
  """Computed row for Table 4 (phase execution shares and CV breakdown)."""

  problem: str
  algorithm: str
  mean_wall_ms: float
  has_phases: bool
  phase_shares: tuple[tuple[float, float], ...]
  sum_share: float


def _compute_baseline_times(
  cohort: BenchmarkCohort,
  effective_baseline: str,
) -> dict[str, float]:
  """Compute per-problem baseline mean runtime with local fallback when baseline is absent."""
  baseline_times: dict[str, float] = {}
  for problem in cohort.problems:
    base_time = cohort.metric(problem, effective_baseline, "mean_time")
    if base_time is not None:
      baseline_times[problem] = base_time
      continue
    first_alg = cohort.algorithms_for_problem(problem)[0]
    baseline_times[problem] = float(cohort.results[problem][first_alg]["mean_time"])
    logging.warning(f"Baseline {effective_baseline} not found for {problem}, using {first_alg}")
  return baseline_times


def _compute_table1_rows(
  cohort: BenchmarkCohort,
  baseline_algorithm: str,
  baseline_times: dict[str, float],
) -> tuple[list[Table1Row], bool]:
  """Compute structured Table 1 rows and whether any local fallback baseline was used."""
  rows: list[Table1Row] = []
  has_fallback = False
  last_problem_idx = len(cohort.problems) - 1

  for prob_idx, problem in enumerate(cohort.problems):
    algs_in_problem = cohort.algorithms_for_problem(problem)
    prob_results = cohort.results[problem]
    is_fallback = baseline_algorithm not in prob_results
    if is_fallback:
      has_fallback = True

    problem_base_alg = baseline_algorithm if not is_fallback else algs_in_problem[0]
    base_time = baseline_times.get(problem, float(prob_results[problem_base_alg]["mean_time"]))
    span = len(algs_in_problem)

    for alg_idx, alg in enumerate(algs_in_problem):
      rec = prob_results[alg]
      mean_time = float(rec["mean_time"])
      is_base = alg == problem_base_alg
      speedup: float | None = (
        1.0 if is_base else ((base_time / mean_time) if mean_time >= 1e-6 and base_time > 0 else None)
      )
      rows.append(
        Table1Row(
          problem=problem,
          algorithm=alg,
          mean_cost=finite_metric(rec.get("mean_cost")),
          std_cost=finite_metric(rec.get("std_cost")),
          mean_gap=finite_metric(rec.get("mean_gap")),
          mean_time=mean_time,
          speedup=speedup,
          is_baseline=is_base,
          is_fallback=is_fallback,
          problem_row_index=alg_idx,
          problem_row_span=span,
          is_last_in_problem=alg_idx == span - 1,
          is_last_problem=prob_idx == last_problem_idx,
          feasibility_rate=finite_metric(rec.get("feasibility_rate")),
          mean_vehicles=finite_metric(rec.get("mean_vehicles")),
        )
      )

  return rows, has_fallback


def _compute_table2_stats(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
  *,
  cohort: BenchmarkCohort | None = None,
  analysis: CohortAnalysis | None = None,
) -> list[dict[str, Any]]:
  """Compute aggregated statistics across all problems for each algorithm (Table 2)."""
  active_cohort = cohort or (
    analysis.cohort
    if analysis is not None
    else BenchmarkCohort.from_results(
      all_results,
      algorithm_order=algorithm_names,
      sort_problems=False,
    )
  )
  can_reuse_analysis = (
    analysis is not None
    and analysis.baseline_algorithm == baseline_algorithm
    and list(analysis.cohort.problems) == problem_names
  )
  stats: list[dict[str, Any]] = []

  for alg in algorithm_names:
    agg = (
      analysis.per_algorithm.get(alg)
      if can_reuse_analysis and analysis is not None
      else active_cohort.summarize_algorithm(alg, problems=problem_names)
    )
    if agg is None or (agg.mean_gap is None and agg.mean_time is None):
      continue

    if alg == baseline_algorithm:
      avg_speedup: float | None = 1.0
      sample_size: int | None = None
    else:
      sp = (
        analysis.baseline_speedups[alg]
        if can_reuse_analysis and analysis is not None and alg in analysis.baseline_speedups
        else active_cohort.paired_geometric_speedup(alg, baseline_algorithm, problems=problem_names)
      )
      avg_speedup = sp.geomean
      sample_size = sp.sample_size

    stats.append(
      {
        "algorithm": alg,
        "mean_gap": agg.mean_gap,
        "std_gap": agg.std_gap,
        "feasibility_rate": agg.mean_feasibility_rate,
        "mean_vehicles": agg.mean_vehicles,
        "mean_time": agg.mean_time if agg.mean_time is not None else 0.0,
        "std_time": agg.std_time if agg.std_time is not None else 0.0,
        "avg_speedup": avg_speedup,
        "sample_size": sample_size,
        "is_baseline": alg == baseline_algorithm,
      }
    )

  return stats


def _compute_table3_size_category_winners(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  algorithm_names: list[str],
  size_thresholds: tuple[int, int] | None = None,
  *,
  cohort: BenchmarkCohort | None = None,
) -> list[dict[str, Any]]:
  """Determine the best performing algorithm by problem size category (Table 3, BUG-50)."""
  active_cohort = cohort or BenchmarkCohort.from_results(
    all_results,
    problem_configs,
    algorithm_order=algorithm_names,
    sort_problems=False,
  )
  thresholds = ReportStyle().size_thresholds if size_thresholds is None else size_thresholds
  categories = categorize_by_size(
    list(active_cohort.problems),
    active_cohort.problem_configs,
    size_thresholds=thresholds,
  )
  winners: list[dict[str, Any]] = []

  for category, problems in categories.items():
    if not problems:
      continue

    score_metric = active_cohort.quality_metric_key(
      algorithms=algorithm_names,
      problems=problems,
      require_complete=False,
    )
    use_gap = score_metric == "mean_gap"

    best_alg: str | None = None
    best_score = float("inf")
    best_gap: float | None = None
    best_time: float | None = None

    for alg in algorithm_names:
      scores = active_cohort.algorithm_metric_values(alg, score_metric, problems=problems)
      times = active_cohort.algorithm_metric_values(alg, "mean_time", problems=problems)
      if len(scores) != len(problems) or len(times) != len(problems):
        continue

      avg_score = float(np.mean(scores))
      if avg_score < best_score:
        best_score = avg_score
        best_alg = alg
        best_gap = avg_score if use_gap else None
        best_time = float(np.mean(times))

    winners.append(
      {
        "category": category,
        "problems": problems,
        "best_alg": best_alg,
        "best_gap": best_gap,
        "best_time": best_time,
      }
    )

  return winners


def _generate_table3_markdown(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  algorithm_names: list[str],
  style: ReportStyle,
  *,
  cohort: BenchmarkCohort | None = None,
) -> list[str]:
  """Generate Markdown format for Table 3 (best by size category)."""
  winners = _compute_table3_size_category_winners(
    all_results,
    problem_configs,
    algorithm_names,
    size_thresholds=style.size_thresholds,
    cohort=cohort,
  )
  return render_table3_markdown(winners, style)


def _generate_table3_latex(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  algorithm_names: list[str],
  style: ReportStyle,
  *,
  cohort: BenchmarkCohort | None = None,
) -> list[str]:
  """Generate LaTeX format for Table 3 (best by size category)."""
  winners = _compute_table3_size_category_winners(
    all_results,
    problem_configs,
    algorithm_names,
    size_thresholds=style.size_thresholds,
    cohort=cohort,
  )
  return render_table3_latex(winners, style)


def _has_phase_timings(
  cohort_or_results: BenchmarkCohort | Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
  """Check if at least one algorithm across problems has mean_phase_timings data (BUG-35)."""
  results_map = cohort_or_results.results if isinstance(cohort_or_results, BenchmarkCohort) else cohort_or_results
  return any(bool(data.get("mean_phase_timings")) for algs in results_map.values() for data in algs.values())


def _compute_phase_timing_rows(cohort: BenchmarkCohort) -> tuple[list[PhaseTimingRow], bool]:
  """Extract structured Table 4 phase-share rows and whether any sub-100ms warning applies."""
  rows: list[PhaseTimingRow] = []
  has_floor_warning = False

  for problem in cohort.problems:
    for alg in cohort.algorithms_for_problem(problem):
      data = cohort.results[problem][alg]
      mean_wall_ms = (finite_metric(data.get("mean_time")) or 0.0) * 1000.0
      has_phases = bool(data.get("mean_phase_timings"))
      if has_phases and mean_wall_ms < 100.0:
        has_floor_warning = True

      if not has_phases:
        rows.append(
          PhaseTimingRow(
            problem=problem,
            algorithm=alg,
            mean_wall_ms=mean_wall_ms,
            has_phases=False,
            phase_shares=(),
            sum_share=0.0,
          )
        )
        continue

      share_pct = data.get("phase_share_pct", {})
      cv_pct = data.get("phase_share_cv_pct", {})
      shares = tuple((float(share_pct.get(p, 0.0)), float(cv_pct.get(p, 0.0))) for p in _PHASE_NAMES)
      rows.append(
        PhaseTimingRow(
          problem=problem,
          algorithm=alg,
          mean_wall_ms=mean_wall_ms,
          has_phases=True,
          phase_shares=shares,
          sum_share=sum(s for s, _ in shares),
        )
      )

  return rows, has_floor_warning


def _prepare_table_inputs(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  baseline_algorithm: str | None,
  style: ReportStyle | None,
  cohort: BenchmarkCohort | None,
  analysis: CohortAnalysis | None,
) -> tuple[ReportStyle, BenchmarkCohort, str, list[Table1Row], bool, list[dict[str, Any]]]:
  """Resolve cohort, baseline, Table 1 rows, and Table 2 aggregates."""
  if analysis is None and not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_style = style or ReportStyle()
  active_cohort = cohort or (
    analysis.cohort
    if analysis is not None
    else BenchmarkCohort.from_results(
      all_results,
      problem_configs,
      algorithm_order=effective_style.algorithm_order,
    )
  )
  effective_baseline = (
    analysis.baseline_algorithm
    if analysis is not None and baseline_algorithm is None
    else active_cohort.baseline(baseline_algorithm)
  )
  baseline_times = _compute_baseline_times(active_cohort, effective_baseline)
  t1_rows, has_fallback = _compute_table1_rows(active_cohort, effective_baseline, baseline_times)
  t2_stats = _compute_table2_stats(
    active_cohort.results,
    list(active_cohort.problems),
    list(active_cohort.algorithms),
    effective_baseline,
    cohort=active_cohort,
    analysis=analysis,
  )
  return effective_style, active_cohort, effective_baseline, t1_rows, has_fallback, t2_stats


def generate_markdown_tables(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "benchmark",
  baseline_algorithm: str | None = None,
  *,
  include_size_category_table: bool = False,
  style: ReportStyle | None = None,
  cohort: BenchmarkCohort | None = None,
  analysis: CohortAnalysis | None = None,
) -> Path:
  """Generate standalone Markdown benchmark summary tables (``<benchmark_name>.md``)."""
  effective_style, active_cohort, effective_baseline, t1_rows, has_fallback, t2_stats = _prepare_table_inputs(
    all_results, problem_configs, baseline_algorithm, style, cohort, analysis
  )
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  show_vrp_columns = active_cohort.is_vrp_cohort
  md_lines: list[str] = []
  md_lines.extend(
    render_table1_markdown(
      t1_rows,
      has_fallback,
      effective_baseline,
      benchmark_name,
      effective_style,
      show_vrp_columns=show_vrp_columns,
    )
  )
  md_lines.extend(render_table2_markdown(t2_stats, effective_style, show_vrp_columns=show_vrp_columns))
  if include_size_category_table:
    md_lines.extend(
      _generate_table3_markdown(
        active_cohort.results,
        active_cohort.problem_configs,
        list(active_cohort.algorithms),
        effective_style,
        cohort=active_cohort,
      )
    )
  if _has_phase_timings(active_cohort):
    phase_rows, has_floor_warning = _compute_phase_timing_rows(active_cohort)
    md_lines.extend(render_phase_timings_markdown(phase_rows, has_floor_warning, effective_style, len(_PHASE_NAMES)))

  md_path = out_dir / f"{benchmark_name}.md"
  md_path.write_text("\n".join(md_lines), encoding="utf-8")
  return md_path


def generate_latex_tables(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "benchmark",
  baseline_algorithm: str | None = None,
  *,
  include_size_category_table: bool = False,
  style: ReportStyle | None = None,
  cohort: BenchmarkCohort | None = None,
  analysis: CohortAnalysis | None = None,
) -> Path:
  """Generate standalone LaTeX benchmark summary tables (``<benchmark_name>.tex``)."""
  effective_style, active_cohort, effective_baseline, t1_rows, has_fallback, t2_stats = _prepare_table_inputs(
    all_results, problem_configs, baseline_algorithm, style, cohort, analysis
  )
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  show_vrp_columns = active_cohort.is_vrp_cohort
  tex_lines: list[str] = []
  tex_lines.extend(
    render_table1_latex(
      t1_rows,
      has_fallback,
      effective_baseline,
      benchmark_name,
      effective_style,
      show_vrp_columns=show_vrp_columns,
    )
  )
  tex_lines.extend(
    render_table2_latex(t2_stats, effective_baseline, effective_style, show_vrp_columns=show_vrp_columns)
  )
  if include_size_category_table:
    tex_lines.extend(
      _generate_table3_latex(
        active_cohort.results,
        active_cohort.problem_configs,
        list(active_cohort.algorithms),
        effective_style,
        cohort=active_cohort,
      )
    )
  if _has_phase_timings(active_cohort):
    phase_rows, has_floor_warning = _compute_phase_timing_rows(active_cohort)
    tex_lines.extend(render_phase_timings_latex(phase_rows, has_floor_warning, effective_style, len(_PHASE_NAMES)))
  tex_lines.append("\n\\end{document}")

  tex_path = out_dir / f"{benchmark_name}.tex"
  tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
  return tex_path


__all__ = [
  "PhaseTimingRow",
  "Table1Row",
  "generate_latex_tables",
  "generate_markdown_tables",
  "resolve_algorithm_names",
]
