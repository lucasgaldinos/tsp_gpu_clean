"""Markdown and LaTeX benchmark summary table generators.

Provides independent, composable functions (``generate_markdown_tables`` and
``generate_latex_tables``) backed by ``BenchmarkCohort`` and parameterized by
``ReportStyle``. Computation of every table (1-4) is cleanly separated from
Markdown and LaTeX rendering, with zero hardcoded experiment names, variant
display maps, or language presets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString

import numpy as np

from benchmarking.analysis.cohort import BenchmarkCohort, resolve_algorithm_names
from benchmarking.analysis.stats_utils import finite_metric
from benchmarking.reporting.style import (
  ReportStyle,
  categorize_by_size,
  escape_latex_identifier,
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
  mean_cost: float
  std_cost: float
  mean_gap: float | None
  mean_time: float
  speedup: float | None
  is_baseline: bool
  is_fallback: bool
  problem_row_index: int
  problem_row_span: int
  is_last_in_problem: bool
  is_last_problem: bool


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
    first_alg = next(iter(cohort.results[problem].keys()))
    baseline_times[problem] = float(cohort.results[problem][first_alg]["mean_time"])
    logging.warning(f"Baseline {effective_baseline} not found for {problem}, using {first_alg}")
  return baseline_times


# =============================================================================
# Table 1: Summary Statistics by Problem (Compute -> Render)
# =============================================================================


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

    problem_base_alg = baseline_algorithm if not is_fallback else next(iter(prob_results.keys()))
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
          mean_cost=float(rec["mean_cost"]),
          std_cost=float(rec["std_cost"]),
          mean_gap=finite_metric(rec.get("mean_gap")),
          mean_time=mean_time,
          speedup=speedup,
          is_baseline=is_base,
          is_fallback=is_fallback,
          problem_row_index=alg_idx,
          problem_row_span=span,
          is_last_in_problem=alg_idx == span - 1,
          is_last_problem=prob_idx == last_problem_idx,
        )
      )

  return rows, has_fallback


def _generate_table1_markdown(
  rows: Sequence[Table1Row],
  has_fallback: bool,
  baseline_algorithm: str,
  benchmark_name: str,
  style: ReportStyle,
) -> list[str]:
  """Render Table 1 rows in Markdown format."""
  report_title = benchmark_name.replace("_", " ").title()
  baseline_label = style.display_name(baseline_algorithm)
  lines: list[str] = [
    f"# {report_title}\n",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    "## Table 1: Summary Statistics by Problem\n",
    "| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |",
    "|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|",
  ]

  for row in rows:
    dagger = "†" if row.is_fallback else ""
    speedup: LiteralString = f"{row.speedup:.2f}x{dagger}" if row.speedup is not None else "—"
    gap_cell = f"{row.mean_gap:.2f}" if row.mean_gap is not None else "—"
    alg_label = style.display_name(row.algorithm)
    lines.append(
      f"| {row.problem} | {alg_label} | {row.mean_cost:.2f} | {row.std_cost:.2f} | "
      f"{gap_cell} | {row.mean_time:.2f} | {speedup} |"
    )

  lines.append("")
  if has_fallback:
    lines.append(
      f"*Note: Speedup calculated relative to {baseline_label}. "
      f"† Indicates speedup calculated relative to local fallback baseline when "
      f"{baseline_label} is unavailable for that problem.*\n"
    )
  else:
    lines.append(f"*Note: Speedup calculated relative to {baseline_label}.*\n")

  return lines


def _generate_table1_latex(
  rows: Sequence[Table1Row],
  has_fallback: bool,
  baseline_algorithm: str,
  benchmark_name: str,
  style: ReportStyle,
) -> list[str]:
  """Render Table 1 rows in LaTeX format."""
  report_title = benchmark_name.replace("_", " ").title()
  safe_baseline = escape_latex_identifier(style.display_name(baseline_algorithm))

  lines: list[str] = [
    f"% {report_title} - Auto-generated",
    f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"% Compile with: pdflatex {benchmark_name}.tex\n",
    "\\documentclass[11pt,a4paper]{article}",
    "\\usepackage{booktabs}  % Professional tables",
    "\\usepackage{multirow}  % Row spanning",
    "\\usepackage{geometry}  % Page margins",
    "\\usepackage{caption}   % Better captions",
    "\\geometry{margin=1in}",
    "\\captionsetup{font=small,labelfont=bf}\n",
    f"\\title{{{report_title}}}",
    f"\\date{{{time.strftime('%B %d, %Y')}}}",
    "\\author{Automated Benchmark Report}\n",
    "\\begin{document}",
    "\\maketitle\n",
    "\\section{Summary Statistics by Problem}\n",
    "\\begin{table}[htbp]",
    "\\centering",
    f"\\caption{{Summary Statistics by Problem. Speedup relative to baseline algorithm ({safe_baseline}).}}",
    f"\\label{{{style.table_label('summary')}}}",
    "\\begin{tabular}{llrrrrr}",
    "\\toprule",
    "Problem & Algorithm & Mean Cost & Std Dev & Gap (\\%) & Time (s) & Speedup \\\\",
    "\\midrule",
  ]

  for row in rows:
    dagger_tex = "^{\\dagger}" if row.is_fallback else ""
    tex_speedup = f"{row.speedup:.2f}$\\times{dagger_tex}$" if row.speedup is not None else "—"
    problem_cell = (
      f"\\multirow{{{row.problem_row_span}}}{{*}}{{{escape_latex_identifier(row.problem)}}}"
      if row.problem_row_index == 0
      else ""
    )
    safe_alg = escape_latex_identifier(style.display_name(row.algorithm))
    gap_tex = f"{row.mean_gap:7.2f}" if row.mean_gap is not None else f"{'—':>7}"
    lines.append(
      f"{problem_cell:20} & {safe_alg:15} & {row.mean_cost:9.2f} & {row.std_cost:7.2f} & "
      f"{gap_tex} & {row.mean_time:8.2f} & {tex_speedup:>12} \\\\"
    )
    if row.is_last_in_problem and not row.is_last_problem:
      lines.append("\\midrule")

  lines.extend(
    [
      "\\bottomrule",
      "\\end{tabular}",
      "\\vspace{0.5em}",
    ]
  )
  if has_fallback:
    lines.append(
      f"{{\\small \\textit{{Note:}} Speedup calculated relative to "
      f"{safe_baseline}. $\\dagger$ Indicates speedup calculated relative to local fallback baseline when "
      f"{safe_baseline} is unavailable for that problem.}}"
    )
  else:
    lines.append(f"{{\\small \\textit{{Note:}} Speedup calculated relative to {safe_baseline}.}}")
  lines.extend(["\\end{table}", "\\clearpage\n"])
  return lines


# =============================================================================
# Table 2: Algorithm Performance Comparison (Compute -> Render)
# =============================================================================


def _compute_table2_stats(
  all_results: dict[str, dict[str, dict[str, Any]]],
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
        "mean_time": agg.mean_time if agg.mean_time is not None else 0.0,
        "std_time": agg.std_time if agg.std_time is not None else 0.0,
        "avg_speedup": avg_speedup,
        "sample_size": sample_size,
        "is_baseline": alg == baseline_algorithm,
      }
    )

  return stats


def _generate_table2_markdown(
  table2_stats: Sequence[dict[str, Any]],
  style: ReportStyle,
) -> list[str]:
  """Render Table 2 rows in Markdown format."""
  lines: list[str] = [
    "## Table 2: Algorithm Performance Comparison (All Problems)\n",
    "| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |",
    "|:----------|-------------:|--------:|--------------:|---------:|------------:|",
  ]

  for s in table2_stats:
    if s["is_baseline"]:
      speedup_str = "1.00x"
    elif s["avg_speedup"] is not None:
      speedup_str = f"{s['avg_speedup']:.2f}x (n={s['sample_size']})"
    else:
      speedup_str = "—"

    gap_str = f"{s['mean_gap']:.2f} | {s['std_gap']:.2f}" if s["mean_gap"] is not None else "— | —"
    time_str = f"{s['mean_time']:.2f} | {s['std_time']:.2f}"
    alg_label = style.display_name(s["algorithm"])
    lines.append(f"| {alg_label} | {gap_str} | {time_str} | {speedup_str} |")

  return lines


def _generate_table2_latex(
  table2_stats: Sequence[dict[str, Any]],
  baseline_algorithm: str,
  style: ReportStyle,
) -> list[str]:
  """Render Table 2 rows in LaTeX format."""
  lines: list[str] = [
    "\\section{Algorithm Performance Comparison}\n",
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Algorithm Performance Comparison Across All Problems (aggregated statistics)}",
    f"\\label{{{style.table_label('algorithm_comparison')}}}",
    "\\begin{tabular}{lrrrrr}",
    "\\toprule",
    "Algorithm & Mean Gap (\\%) & Std Gap & Mean Time (s) & Std Time & Avg Speedup \\\\",
    "\\midrule",
  ]

  for s in table2_stats:
    if s["is_baseline"]:
      tex_speedup = "1.00$\\times$"
    elif s["avg_speedup"] is not None:
      tex_speedup = f"{s['avg_speedup']:.2f}$\\times$ ($n={s['sample_size']}$)"
    else:
      tex_speedup = "—"

    safe_alg = escape_latex_identifier(style.display_name(s["algorithm"]))
    gap_tex = f"{s['mean_gap']:8.2f} & {s['std_gap']:7.2f}" if s["mean_gap"] is not None else f"{'—':>8} & {'—':>7}"
    time_tex = f"{s['mean_time']:8.2f} & {s['std_time']:8.2f}"
    lines.append(f"{safe_alg:15} & {gap_tex} & {time_tex} & {tex_speedup:>12} \\\\")

  safe_baseline = escape_latex_identifier(style.display_name(baseline_algorithm))
  lines.extend(
    [
      "\\bottomrule",
      "\\end{tabular}",
      "\\vspace{0.5em}",
      f"{{\\small \\textit{{Note:}} Speedup calculated relative to {safe_baseline}.}}",
      "\\end{table}",
      "\\clearpage\n",
    ]
  )
  return lines


# =============================================================================
# Table 3: Best Algorithm by Size Category (Compute -> Render)
# =============================================================================


def _compute_table3_size_category_winners(
  all_results: dict[str, dict[str, dict[str, Any]]],
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
  categories = categorize_by_size(list(all_results.keys()), problem_configs, size_thresholds=thresholds)
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
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  algorithm_names: list[str],
  style: ReportStyle,
  *,
  cohort: BenchmarkCohort | None = None,
) -> list[str]:
  """Generate Markdown format for Table 3 (best by size category)."""
  lines: list[str] = [
    "\n## Table 3: Best Algorithm by Problem Size\n",
    "| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |",
    "|:--------------|:---------|:---------------|-------------:|--------------:|",
  ]

  for entry in _compute_table3_size_category_winners(
    all_results,
    problem_configs,
    algorithm_names,
    size_thresholds=style.size_thresholds,
    cohort=cohort,
  ):
    problems_str = ", ".join(entry["problems"])
    if entry["best_alg"] is not None:
      best_alg_str = style.display_name(str(entry["best_alg"]))
      gap_str = f"{entry['best_gap']:.2f}" if entry["best_gap"] is not None else "—"
      time_str = f"{entry['best_time']:.2f}"
    else:
      best_alg_str = "—"
      gap_str = "—"
      time_str = "—"
    lines.append(f"| {entry['category']} | {problems_str} | {best_alg_str} | {gap_str} | {time_str} |")

  lines.append("")
  lines.append(
    "*Note: Best algorithm determined by lowest mean gap to known optimum (or mean cost when optima are absent).*\n"
  )
  return lines


def _generate_table3_latex(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  algorithm_names: list[str],
  style: ReportStyle,
  *,
  cohort: BenchmarkCohort | None = None,
) -> list[str]:
  """Generate LaTeX format for Table 3 (best by size category)."""
  lines: list[str] = [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Best Performing Algorithm by Problem Size Category}",
    f"\\label{{{style.table_label('size_categories')}}}",
    "\\begin{tabular}{lllrr}",
    "\\toprule",
    "Size Category & Problems & Best Algorithm & Mean Gap (\\%) & Mean Time (s) \\\\",
    "\\midrule",
  ]

  for entry in _compute_table3_size_category_winners(
    all_results,
    problem_configs,
    algorithm_names,
    size_thresholds=style.size_thresholds,
    cohort=cohort,
  ):
    problems_str = ", ".join(entry["problems"])
    if len(problems_str) > 60:
      problems_str = problems_str[:57] + "..."
    problems_str = escape_latex_identifier(problems_str)

    if entry["best_alg"] is not None:
      best_alg_str = escape_latex_identifier(style.display_name(str(entry["best_alg"])))
      gap_str = f"{entry['best_gap']:8.2f}" if entry["best_gap"] is not None else f"{'—':>8}"
      time_str = f"{entry['best_time']:8.2f}"
    else:
      best_alg_str = "—"
      gap_str = f"{'—':>8}"
      time_str = f"{'—':>8}"

    lines.append(f"{entry['category']:20} & {problems_str:40} & {best_alg_str:15} & {gap_str} & {time_str} \\\\")

  lines.extend(
    [
      "\\bottomrule",
      "\\end{tabular}",
      "\\vspace{0.5em}",
      (
        "{\\small \\textit{Note:} Best algorithm determined by lowest mean gap to known optimum "
        "(or mean cost when optima are absent).}"
      ),
      "\\end{table}",
    ]
  )
  return lines


# =============================================================================
# Table 4: Phase Timings Breakdown (Compute -> Render)
# =============================================================================


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


def _generate_phase_timings_markdown(
  rows: Sequence[PhaseTimingRow],
  has_floor_warning: bool,
  style: ReportStyle,
) -> list[str]:
  """Render Table 4 rows in Markdown format."""
  lines: list[str] = [
    "\n## Table 4: Average Phase Timings and Relative Share Breakdown",
    "",
    "Phase execution times (ms), relative share of wall time (%), and CV (%) across repetitions.",
    "",
    (
      "| Problem | Algorithm | Wall (ms) | Init (%) | Sel (%) | CX (%) | Mut (%) | "
      "Eval (%) | LS (%) | Surv (%) | Host Sync (%) | Sum/Wall (%) |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
  ]

  missing_cells = " | ".join(["-"] * len(_PHASE_NAMES))
  for row in rows:
    alg_label = style.display_name(row.algorithm)
    if not row.has_phases:
      lines.append(f"| {row.problem} | {alg_label} | {row.mean_wall_ms:.1f} | {missing_cells} | - |")
      continue
    cells = [f"{s:.1f}% (CV {cv:.1f}%)" if s > 0 else "-" for s, cv in row.phase_shares]
    lines.append(
      f"| {row.problem} | {alg_label} | {row.mean_wall_ms:.1f} | {' | '.join(cells)} | {row.sum_share:.1f}% |"
    )

  if has_floor_warning:
    lines.append("")
    lines.append(
      "> [!WARNING]\n"
      "> **Runtime Floor Warning:** Executions with mean runtime < 100 ms exhibit elevated\n"
      "> relative host synchronization and dispatch overhead (~0.5-1.0 ms fixed latency).\n"
      "> Phase shares for sub-100ms runs should be interpreted with caution."
    )
  return lines


def _generate_phase_timings_latex(
  rows: Sequence[PhaseTimingRow],
  has_floor_warning: bool,
  style: ReportStyle,
) -> list[str]:
  """Render Table 4 rows in LaTeX format."""
  lines: list[str] = [
    "\n\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Average Phase Timings, Relative Share, and Dispersion (\\% $\\pm$ CV \\%)}",
    f"\\label{{{style.table_label('phase_timings')}}}",
    "\\begin{tabular}{llrccccccccr}",
    "\\toprule",
    "Problem & Algorithm & Wall (ms) & Init & Sel & CX & Mut & Eval & LS & Surv & Host Sync & Sum/Wall \\\\",
    "\\midrule",
  ]

  missing_cells = " & ".join(["-"] * len(_PHASE_NAMES))
  for row in rows:
    safe_problem = escape_latex_identifier(row.problem)
    safe_alg = escape_latex_identifier(style.display_name(row.algorithm))
    if not row.has_phases:
      lines.append(f"{safe_problem} & {safe_alg} & {row.mean_wall_ms:.1f} & {missing_cells} & - \\\\")
      continue
    cells = [f"{s:.1f}\\% ({cv:.0f}\\%)" if s > 0 else "-" for s, cv in row.phase_shares]
    lines.append(
      f"{safe_problem} & {safe_alg} & {row.mean_wall_ms:.1f} & {' & '.join(cells)} & {row.sum_share:.1f}\\% \\\\"
    )

  lines.extend(["\\bottomrule", "\\end{tabular}", "\\vspace{0.5em}"])
  if has_floor_warning:
    lines.append(
      "{\\small \\textbf{Warning:} Executions with mean runtime $< 100$\\,ms exhibit elevated "
      "relative host synchronization and dispatch overhead. Phase shares for sub-100\\,ms runs "
      "should be interpreted with caution.}"
    )
  lines.append("\\end{table}")
  return lines


# =============================================================================
# Public Composable Table Generators
# =============================================================================


def generate_markdown_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
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
  if analysis is None and not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_style = style or ReportStyle()
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

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
    all_results,
    list(active_cohort.problems),
    list(active_cohort.algorithms),
    effective_baseline,
    cohort=active_cohort,
    analysis=analysis,
  )

  md_lines: list[str] = []
  md_lines.extend(_generate_table1_markdown(t1_rows, has_fallback, effective_baseline, benchmark_name, effective_style))
  md_lines.extend(_generate_table2_markdown(t2_stats, effective_style))
  if include_size_category_table:
    md_lines.extend(
      _generate_table3_markdown(
        all_results,
        problem_configs,
        list(active_cohort.algorithms),
        effective_style,
        cohort=active_cohort,
      )
    )
  if _has_phase_timings(active_cohort):
    phase_rows, has_floor_warning = _compute_phase_timing_rows(active_cohort)
    md_lines.extend(_generate_phase_timings_markdown(phase_rows, has_floor_warning, effective_style))

  md_path = out_dir / f"{benchmark_name}.md"
  md_path.write_text("\n".join(md_lines), encoding="utf-8")
  return md_path


def generate_latex_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
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
  if analysis is None and not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_style = style or ReportStyle()
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

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
    all_results,
    list(active_cohort.problems),
    list(active_cohort.algorithms),
    effective_baseline,
    cohort=active_cohort,
    analysis=analysis,
  )

  tex_lines: list[str] = []
  tex_lines.extend(_generate_table1_latex(t1_rows, has_fallback, effective_baseline, benchmark_name, effective_style))
  tex_lines.extend(_generate_table2_latex(t2_stats, effective_baseline, effective_style))
  if include_size_category_table:
    tex_lines.extend(
      _generate_table3_latex(
        all_results,
        problem_configs,
        list(active_cohort.algorithms),
        effective_style,
        cohort=active_cohort,
      )
    )
  if _has_phase_timings(active_cohort):
    phase_rows, has_floor_warning = _compute_phase_timing_rows(active_cohort)
    tex_lines.extend(_generate_phase_timings_latex(phase_rows, has_floor_warning, effective_style))
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
