"""
Report Generator Module (`DP14-2`, `DP14-5`)
===========================================

Generates formatted output tables, quantitative LaTeX macros, Demšar Critical
Difference TikZ diagrams, and empirical scaling vector plots.

Output contract:
- Markdown and LaTeX tables: ``<benchmark_name>.md`` and ``<benchmark_name>.tex``
  in ``output_dir``;
- Quantitative LaTeX macros: ``benchmark_macros.tex`` in ``output_dir`` with
  strictly numeric ``\\newcommand`` definitions (zero prose);
- Analytical rank diagrams: ``<benchmark_name>_cd_diagram_{gap,time}.tikz`` in
  ``output_dir`` (TikZ snippets inheriting the document typeface);
- Empirical scaling curves: ``scaling_time_gap.{pdf,svg}`` in the sibling
  ``figures/`` directory (Matplotlib Type 42 vector output).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString

import numpy as np
from scipy import stats as scipy_stats

from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.analysis.stats_utils import resolve_baseline_algorithm

if TYPE_CHECKING:
  from collections.abc import Sequence

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

# Canonical size-stratum thresholds aligned with Table 3.3 of the thesis (DP13-7 / BUG-65).
SMALL_SIZE_MAX: int = 100
MEDIUM_SIZE_MAX: int = 400
SMALL_SIZE_LABEL: str = f"Small (n≤{SMALL_SIZE_MAX})"
MEDIUM_SIZE_LABEL: str = f"Medium ({SMALL_SIZE_MAX}<n≤{MEDIUM_SIZE_MAX})"
LARGE_SIZE_LABEL: str = f"Large (n>{MEDIUM_SIZE_MAX})"
SIZE_CATEGORY_LABELS: tuple[str, str, str] = (
  SMALL_SIZE_LABEL,
  MEDIUM_SIZE_LABEL,
  LARGE_SIZE_LABEL,
)

# =============================================================================
# Helper Functions
# =============================================================================


def escape_latex_identifier(name: str) -> str:
  """Escape LaTeX special characters (`_` and `%`) in identifiers.

  Replaces `_` with `\\_` and `%` with `\\%` so algorithm, problem, and metric
  names render properly without causing math-mode subscript errors or LaTeX comment truncations.
  """
  return name.replace("_", "\\_").replace("%", "\\%")


def _get_problem_size(problem_name: str, problem_configs: list[ProblemConfig]) -> int | None:
  """Get size of a problem from configs."""
  for config in problem_configs:
    if config.name == problem_name:
      return int(config.size)
  return None


def _categorize_by_size(problem_names: list[str], problem_configs: list[ProblemConfig]) -> dict[str, list[str]]:
  """Categorize problems by canonical size strata defined in ``SIZE_CATEGORY_LABELS``."""
  categories: dict[str, list[str]] = {label: [] for label in SIZE_CATEGORY_LABELS}

  for problem in problem_names:
    size = _get_problem_size(problem, problem_configs)
    if size is None:
      continue

    if size <= SMALL_SIZE_MAX:
      categories[SMALL_SIZE_LABEL].append(problem)
    elif size <= MEDIUM_SIZE_MAX:
      categories[MEDIUM_SIZE_LABEL].append(problem)
    else:
      categories[LARGE_SIZE_LABEL].append(problem)

  return categories


# =============================================================================
# Table 1: Summary Statistics by Problem
# =============================================================================


def _generate_table1_markdown(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
  baseline_times: dict[str, float],
  benchmark_name: str,
) -> list[str]:
  """Generate Markdown format for Table 1 (summary by problem)."""
  lines = []

  report_title = benchmark_name.replace("_", " ").title()
  lines.append(f"# {report_title}\n")
  lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
  lines.append("## Table 1: Summary Statistics by Problem\n")
  lines.append("| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |")
  lines.append("|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|")

  has_fallback = False
  for problem in problem_names:
    algs_in_problem: list[str] = [alg for alg in algorithm_names if alg in all_results[problem]]
    is_fallback = baseline_algorithm not in all_results[problem]
    if is_fallback:
      has_fallback = True

    problem_base_alg: str = baseline_algorithm if not is_fallback else next(iter(all_results[problem].keys()))
    base_time = (
      baseline_times[problem]
      if problem in baseline_times
      else float(all_results[problem][problem_base_alg]["mean_time"])
    )

    for alg in algs_in_problem:
      r = all_results[problem][alg]
      dagger_str = "†" if is_fallback else ""

      if alg == problem_base_alg:
        speedup: LiteralString = f"1.00x{dagger_str}"
      elif r["mean_time"] >= 1e-6 and base_time > 0:
        speedup_val = base_time / r["mean_time"]
        speedup = f"{speedup_val:.2f}x{dagger_str}"
      else:
        speedup = "—"

      lines.append(
        f"| {problem} | {alg} | {r['mean_cost']:.2f} | {r['std_cost']:.2f} | "
        f"{r['mean_gap']:.2f} | {r['mean_time']:.2f} | {speedup} |"
      )

  lines.append("")
  if has_fallback:
    lines.append(
      f"*Note: Speedup calculated relative to {baseline_algorithm}. "
      f"† Indicates speedup calculated relative to local fallback baseline when "
      f"{baseline_algorithm} is unavailable for that problem.*\n"
    )
  else:
    lines.append(f"*Note: Speedup calculated relative to {baseline_algorithm}.*\n")

  return lines


def _generate_table1_latex(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
  baseline_times: dict[str, float],
  benchmark_name: str,
) -> list[str]:
  """Generate LaTeX format for Table 1 (summary by problem)."""
  lines = []

  report_title = benchmark_name.replace("_", " ").title()

  lines.append(f"% {report_title} - Auto-generated")
  lines.append(f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append(f"% Compile with: pdflatex {benchmark_name}.tex\n")
  lines.append("\\documentclass[11pt,a4paper]{article}")
  lines.append("\\usepackage{booktabs}  % Professional tables")
  lines.append("\\usepackage{multirow}  % Row spanning")
  lines.append("\\usepackage{geometry}  % Page margins")
  lines.append("\\usepackage{caption}   % Better captions")
  lines.append("\\geometry{margin=1in}")
  lines.append("\\captionsetup{font=small,labelfont=bf}\n")
  lines.append(f"\\title{{{report_title}}}")
  lines.append(f"\\date{{{time.strftime('%B %d, %Y')}}}")
  lines.append("\\author{Automated Benchmark Report}\n")
  lines.append("\\begin{document}")
  lines.append("\\maketitle\n")
  lines.append("\\section{Summary Statistics by Problem}\n")

  lines.append("\\begin{table}[htbp]")
  lines.append("\\centering")
  safe_baseline = escape_latex_identifier(baseline_algorithm)
  lines.append(f"\\caption{{Summary Statistics by Problem. Speedup relative to baseline algorithm ({safe_baseline}).}}")
  lines.append("\\label{tab:chapter4_summary}")
  lines.append("\\begin{tabular}{llrrrrr}")
  lines.append("\\toprule")
  lines.append("Problem & Algorithm & Mean Cost & Std Dev & Gap (\\%) & Time (s) & Speedup \\\\")
  lines.append("\\midrule")

  has_fallback = False
  for problem in problem_names:
    algs_in_problem = [alg for alg in algorithm_names if alg in all_results[problem]]
    is_fallback = baseline_algorithm not in all_results[problem]
    if is_fallback:
      has_fallback = True

    problem_base_alg = baseline_algorithm if not is_fallback else next(iter(all_results[problem].keys()))
    base_time = (
      baseline_times[problem]
      if problem in baseline_times
      else float(all_results[problem][problem_base_alg]["mean_time"])
    )

    for i, alg in enumerate(algs_in_problem):
      r = all_results[problem][alg]
      dagger_tex = "^{\\dagger}" if is_fallback else ""

      if alg == problem_base_alg:
        tex_speedup = f"1.00$\\times{dagger_tex}$"
      elif r["mean_time"] >= 1e-6 and base_time > 0:
        speedup_val = base_time / r["mean_time"]
        tex_speedup = f"{speedup_val:.2f}$\\times{dagger_tex}$"
      else:
        tex_speedup = "—"

      if i == 0:
        problem_cell = f"\\multirow{{{len(algs_in_problem)}}}{{*}}{{{escape_latex_identifier(problem)}}}"
      else:
        problem_cell = ""

      safe_alg = escape_latex_identifier(alg)
      lines.append(
        f"{problem_cell:20} & {safe_alg:15} & {r['mean_cost']:9.2f} & {r['std_cost']:7.2f} & "
        f"{r['mean_gap']:7.2f} & {r['mean_time']:8.2f} & {tex_speedup:>12} \\\\"
      )

    if problem != problem_names[-1]:
      lines.append("\\midrule")

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  if has_fallback:
    lines.append(
      f"{{\\small \\textit{{Note:}} Speedup calculated relative to "
      f"{safe_baseline}. $\\dagger$ Indicates speedup calculated relative to local fallback baseline when "
      f"{safe_baseline} is unavailable for that problem.}}"
    )
  else:
    lines.append(f"{{\\small \\textit{{Note:}} Speedup calculated relative to {safe_baseline}.}}")
  lines.append("\\end{table}")
  lines.append("\\clearpage\n")

  return lines


# =============================================================================
# Table 2: Algorithm Performance Comparison
# =============================================================================


def _compute_table2_stats(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
) -> list[dict[str, Any]]:
  """Compute aggregated statistics across all problems for each algorithm (Table 2).

  Returns a list of dictionaries with keys:
    algorithm, mean_gap, std_gap, mean_time, std_time, avg_speedup, sample_size, is_baseline
  """
  stats: list[dict[str, Any]] = []

  for alg in algorithm_names:
    all_gaps = []
    all_times = []

    for problem in problem_names:
      if alg in all_results[problem]:
        all_gaps.append(all_results[problem][alg]["mean_gap"])
        all_times.append(all_results[problem][alg]["mean_time"])

    if not all_gaps:
      continue

    mean_gap = float(np.mean(all_gaps))
    std_gap = float(np.std(all_gaps, ddof=1)) if len(all_gaps) > 1 else 0.0
    mean_time = float(np.mean(all_times))
    std_time = float(np.std(all_times, ddof=1)) if len(all_times) > 1 else 0.0

    # Geometric mean of paired speedups per problem instance (Fleming & Wallace 1986, BUG-34)
    paired_speedups = []
    if alg != baseline_algorithm:
      for problem in problem_names:
        if problem in all_results:
          base_time = float(all_results[problem].get(baseline_algorithm, {}).get("mean_time", 0.0))
          cand_time = float(all_results[problem].get(alg, {}).get("mean_time", 0.0))
          if base_time > 0.0 and cand_time > 0.0:
            paired_speedups.append(base_time / cand_time)

    avg_speedup: float | None = None
    if alg == baseline_algorithm:
      avg_speedup = 1.0
    elif paired_speedups:
      avg_speedup = float(np.exp(np.mean(np.log(paired_speedups))))

    stats.append(
      {
        "algorithm": alg,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "mean_time": mean_time,
        "std_time": std_time,
        "avg_speedup": avg_speedup,
        "sample_size": len(paired_speedups) if alg != baseline_algorithm else None,
        "is_baseline": alg == baseline_algorithm,
      }
    )

  return stats


def _generate_table2_markdown(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
) -> list[str]:
  """Generate Markdown format for Table 2 (algorithm comparison)."""
  lines = []

  lines.append("## Table 2: Algorithm Performance Comparison (All Problems)\n")
  lines.append("| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |")
  lines.append("|:----------|-------------:|--------:|--------------:|---------:|------------:|")

  for s in _compute_table2_stats(all_results, problem_names, algorithm_names, baseline_algorithm):
    speedup_str = "—"
    if s["is_baseline"]:
      speedup_str = "1.00x"
    elif s["avg_speedup"] is not None:
      speedup_str = f"{s['avg_speedup']:.2f}x (n={s['sample_size']})"

    gap_str = f"{s['mean_gap']:.2f} | {s['std_gap']:.2f}"
    time_str = f"{s['mean_time']:.2f} | {s['std_time']:.2f}"
    lines.append(f"| {s['algorithm']} | {gap_str} | {time_str} | {speedup_str} |")

  return lines


def _generate_table2_latex(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
) -> list[str]:
  """Generate LaTeX format for Table 2 (algorithm comparison)."""
  lines = []

  lines.append("\\section{Algorithm Performance Comparison}\n")
  lines.append("\\begin{table}[htbp]")
  lines.append("\\centering")
  lines.append("\\caption{Algorithm Performance Comparison Across All Problems (aggregated statistics)}")
  lines.append("\\label{tab:chapter4_algorithm_comparison}")
  lines.append("\\begin{tabular}{lrrrrr}")
  lines.append("\\toprule")
  lines.append("Algorithm & Mean Gap (\\%) & Std Gap & Mean Time (s) & Std Time & Avg Speedup \\\\")
  lines.append("\\midrule")

  for s in _compute_table2_stats(all_results, problem_names, algorithm_names, baseline_algorithm):
    tex_speedup = "—"
    if s["is_baseline"]:
      tex_speedup = "1.00$\\times$"
    elif s["avg_speedup"] is not None:
      tex_speedup = f"{s['avg_speedup']:.2f}$\\times$ ($n={s['sample_size']}$)"

    safe_alg = escape_latex_identifier(s["algorithm"])
    gap_tex = f"{s['mean_gap']:8.2f} & {s['std_gap']:7.2f}"
    time_tex = f"{s['mean_time']:8.2f} & {s['std_time']:8.2f}"
    lines.append(f"{safe_alg:15} & {gap_tex} & {time_tex} & {tex_speedup:>12} \\\\")

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  safe_baseline = escape_latex_identifier(baseline_algorithm)
  lines.append(f"{{\\small \\textit{{Note:}} Speedup calculated relative to {safe_baseline}.}}")
  lines.append("\\end{table}")
  lines.append("\\clearpage\n")

  return lines


# =============================================================================
# Table 3: Best Algorithm by Size Category
# =============================================================================


def _compute_table3_size_category_winners(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  algorithm_names: list[str],
) -> list[dict[str, Any]]:
  """Determine the best performing algorithm by problem size category (Table 3).

  Returns a list of dictionaries with keys:
    category, problems, best_alg, best_gap, best_time
  """
  problem_names = list(all_results.keys())
  categories = _categorize_by_size(problem_names, problem_configs)
  winners: list[dict[str, Any]] = []

  for category, problems in categories.items():
    if not problems:
      continue

    best_alg = None
    best_gap = float("inf")
    best_time = 0.0

    for alg in algorithm_names:
      gaps = []
      times = []
      for prob in problems:
        if prob in all_results and alg in all_results[prob]:
          gaps.append(all_results[prob][alg]["mean_gap"])
          times.append(all_results[prob][alg]["mean_time"])

      if len(gaps) == len(problems) and len(problems) > 0:
        avg_gap = float(np.mean(gaps))
        if avg_gap < best_gap:
          best_gap = avg_gap
          best_alg = alg
          best_time = float(np.mean(times))

    winners.append(
      {
        "category": category,
        "problems": problems,
        "best_alg": best_alg,
        "best_gap": best_gap if best_alg is not None else None,
        "best_time": best_time if best_alg is not None else None,
      }
    )

  return winners


def _generate_table3_markdown(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  algorithm_names: list[str],
) -> list[str]:
  """Generate Markdown format for Table 3 (best by size category)."""
  lines = []

  lines.append("\n## Table 3: Best Algorithm by Problem Size\n")
  lines.append("| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |")
  lines.append("|:--------------|:---------|:---------------|-------------:|--------------:|")

  for entry in _compute_table3_size_category_winners(all_results, problem_configs, algorithm_names):
    problems_str = ", ".join(entry["problems"])
    if entry["best_alg"] is not None:
      best_alg_str = str(entry["best_alg"])
      gap_str = f"{entry['best_gap']:.2f}"
      time_str = f"{entry['best_time']:.2f}"
    else:
      best_alg_str = "—"
      gap_str = "—"
      time_str = "—"
    lines.append(f"| {entry['category']} | {problems_str} | {best_alg_str} | {gap_str} | {time_str} |")

  lines.append("")
  lines.append("*Note: Best algorithm determined by lowest mean gap to known optimum.*\n")

  return lines


def _generate_table3_latex(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  algorithm_names: list[str],
) -> list[str]:
  """Generate LaTeX format for Table 3 (best by size category)."""
  lines = []

  lines.append("\\begin{table}[htbp]")
  lines.append("\\centering")
  lines.append("\\caption{Best Performing Algorithm by Problem Size Category}")
  lines.append("\\label{tab:chapter4_size_categories}")
  lines.append("\\begin{tabular}{lllrr}")
  lines.append("\\toprule")
  lines.append("Size Category & Problems & Best Algorithm & Mean Gap (\\%) & Mean Time (s) \\\\")
  lines.append("\\midrule")

  for entry in _compute_table3_size_category_winners(all_results, problem_configs, algorithm_names):
    problems_str = ", ".join(entry["problems"])
    if len(problems_str) > 60:
      problems_str = problems_str[:57] + "..."
    problems_str = escape_latex_identifier(problems_str)

    if entry["best_alg"] is not None:
      best_alg_str = escape_latex_identifier(str(entry["best_alg"]))
      gap_str = f"{entry['best_gap']:8.2f}"
      time_str = f"{entry['best_time']:8.2f}"
    else:
      best_alg_str = "—"
      gap_str = f"{'—':>8}"
      time_str = f"{'—':>8}"

    lines.append(f"{entry['category']:20} & {problems_str:40} & {best_alg_str:15} & {gap_str} & {time_str} \\\\")

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  lines.append("{\\small \\textit{Note:} Best algorithm determined by lowest mean gap to known optimum.}")
  lines.append("\\end{table}")

  return lines


def _has_phase_timings(all_results: dict[str, dict[str, dict[str, Any]]]) -> bool:
  """Check if at least one algorithm across problems has mean_phase_timings data (BUG-35)."""
  for _prob, algs in all_results.items():
    for _alg, data in algs.items():
      if data.get("mean_phase_timings"):
        return True
  return False


def _generate_phase_timings_markdown(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
) -> list[str]:
  """Generate Table 4: Phase Timing and Relative Share Breakdown in Markdown."""
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

  phases = _PHASE_NAMES
  has_floor_warning = False

  for problem in problem_names:
    for alg in algorithm_names:
      if alg not in all_results[problem]:
        continue
      data = all_results[problem][alg]
      mean_wall_ms = float(data.get("mean_time", 0.0)) * 1000.0
      has_phases = bool(data.get("mean_phase_timings"))

      if has_phases and mean_wall_ms < 100.0:
        has_floor_warning = True

      if not has_phases:
        missing_cells = ["-"] * len(phases)
        row = f"| {problem} | {alg} | {mean_wall_ms:.1f} | " + " | ".join(missing_cells) + " | - |"
        lines.append(row)
        continue

      share_pct = data.get("phase_share_pct", {})
      cv_pct = data.get("phase_share_cv_pct", {})

      phase_cells: list[str] = []
      sum_share = 0.0
      for p in phases:
        s = float(share_pct.get(p, 0.0))
        cv = float(cv_pct.get(p, 0.0))
        sum_share += s
        if s > 0:
          phase_cells.append(f"{s:.1f}% (CV {cv:.1f}%)")
        else:
          phase_cells.append("-")

      row = f"| {problem} | {alg} | {mean_wall_ms:.1f} | " + " | ".join(phase_cells) + f" | {sum_share:.1f}% |"
      lines.append(row)

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
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
) -> list[str]:
  """Generate Table 4: Phase Timing and Relative Share Breakdown in LaTeX."""
  lines: list[str] = [
    "\n\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Average Phase Timings, Relative Share, and Dispersion (\\% $\\pm$ CV \\%)}",
    "\\label{tab:phase_timings}",
    "\\begin{tabular}{llrccccccccr}",
    "\\toprule",
    "Problem & Algorithm & Wall (ms) & Init & Sel & CX & Mut & Eval & LS & Surv & Host Sync & Sum/Wall \\\\",
    "\\midrule",
  ]

  phases = _PHASE_NAMES
  has_floor_warning = False

  for problem in problem_names:
    for alg in algorithm_names:
      if alg not in all_results[problem]:
        continue
      data = all_results[problem][alg]
      mean_wall_ms = float(data.get("mean_time", 0.0)) * 1000.0
      has_phases = bool(data.get("mean_phase_timings"))

      if has_phases and mean_wall_ms < 100.0:
        has_floor_warning = True

      safe_problem = escape_latex_identifier(problem)
      safe_alg = escape_latex_identifier(alg)

      if not has_phases:
        missing_cells = ["-"] * len(phases)
        row = f"{safe_problem} & {safe_alg} & {mean_wall_ms:.1f} & " + " & ".join(missing_cells) + " & - \\\\"
        lines.append(row)
        continue

      share_pct = data.get("phase_share_pct", {})
      cv_pct = data.get("phase_share_cv_pct", {})

      phase_cells = []
      sum_share = 0.0
      for p in phases:
        s = float(share_pct.get(p, 0.0))
        cv = float(cv_pct.get(p, 0.0))
        sum_share += s
        if s > 0:
          phase_cells.append(f"{s:.1f}\\% ({cv:.0f}\\%)")
        else:
          phase_cells.append("-")

      row = (
        f"{safe_problem} & {safe_alg} & {mean_wall_ms:.1f} & " + " & ".join(phase_cells) + f" & {sum_share:.1f}\\% \\\\"
      )
      lines.append(row)

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  if has_floor_warning:
    lines.append(
      "{\\small \\textbf{Warning:} Executions with mean runtime $< 100$\\,ms exhibit elevated "
      "relative host synchronization and dispatch overhead. Phase shares for sub-100\\,ms runs "
      "should be interpreted with caution.}"
    )
  lines.append("\\end{table}")
  return lines


# =============================================================================
# Quantitative LaTeX Macros (`DP14-5`)
# =============================================================================

_MACRO_DIGIT_WORDS: dict[str, str] = {
  "0": "Zero",
  "1": "One",
  "2": "Two",
  "3": "Three",
  "4": "Four",
  "5": "Five",
  "6": "Six",
  "7": "Seven",
  "8": "Eight",
  "9": "Nine",
}


def _latex_macro_token(name: str) -> str:
  """Sanitize an identifier into a valid ASCII-letter-only LaTeX macro token."""
  parts: list[str] = []
  for ch in name:
    if ch.isascii() and ch.isalpha():
      parts.append(ch)
    elif ch in _MACRO_DIGIT_WORDS:
      parts.append(_MACRO_DIGIT_WORDS[ch])
  return "".join(parts) or "Algorithm"


def _build_algorithm_macro_tokens(algorithm_names: Sequence[str]) -> dict[str, str]:
  """Map each algorithm name to a unique valid LaTeX command suffix."""
  tokens: dict[str, str] = {}
  used: dict[str, int] = {}
  for alg in algorithm_names:
    base = _latex_macro_token(alg)
    if base not in used:
      used[base] = 1
      tokens[alg] = base
    else:
      used[base] += 1
      suffix_idx = used[base]
      candidate = f"{base}{_MACRO_DIGIT_WORDS.get(str(suffix_idx), 'Extra')}"
      while candidate in used:
        suffix_idx += 1
        candidate = f"{base}{suffix_idx * 'X'}"
      used[candidate] = 1
      tokens[alg] = candidate
  return tokens


def _format_decimal_latex(
  value: float,
  decimals: int = 2,
  decimal_separator: str = "{,}",
) -> str:
  """Format a finite or infinite float for LaTeX with configurable decimal separator."""
  if math.isnan(value):
    return r"\mathrm{NaN}"
  if math.isinf(value):
    return r"\ensuremath{\infty}" if value > 0 else r"\ensuremath{-\infty}"
  if abs(value) < 0.5 * (10.0**-decimals):
    value = 0.0
  return f"{value:.{decimals}f}".replace(".", decimal_separator)


def _format_pvalue_latex(p_val: float, decimal_separator: str = "{,}") -> str:
  """Format a statistical p-value in fixed decimal or LaTeX scientific notation."""
  if math.isnan(p_val):
    return r"\mathrm{NaN}"
  if p_val <= 0.0:
    return r"\ensuremath{< 10^{-6}}"
  if p_val < 1e-3:
    exp = math.floor(math.log10(p_val))
    mant = p_val / (10.0**exp)
    if round(mant, 2) >= 10.0:
      mant /= 10.0
      exp += 1
    mant_str = f"{mant:.2f}".replace(".", decimal_separator)
    return rf"\ensuremath{{{mant_str} \times 10^{{{exp}}}}}"
  if p_val >= 0.99995:
    return f"1{decimal_separator}00"
  return f"{p_val:.4f}".replace(".", decimal_separator)


def _holm_adjusted_pvalues(raw_p_values: Sequence[float]) -> list[float]:
  """Compute step-down Holm-Bonferroni adjusted p-values preserving input order."""
  m = len(raw_p_values)
  if m == 0:
    return []
  indexed = sorted(enumerate(raw_p_values), key=lambda item: item[1])
  adjusted = [1.0] * m
  running_max = 0.0
  for rank_idx, (orig_idx, p_val) in enumerate(indexed):
    scaled = min(1.0, (m - rank_idx) * float(p_val))
    running_max = max(running_max, scaled)
    adjusted[orig_idx] = running_max
  return adjusted


def generate_latex_macros(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None,
  output_dir: Path | str,
  baseline_algorithm: str | None = None,
  *,
  filename: str = "benchmark_macros.tex",
  decimal_separator: str = "{,}",
  alpha: float = 0.05,
) -> Path:
  """Export strictly quantitative LaTeX ``\\newcommand`` macros to ``benchmark_macros.tex``.

  Emits global cohort counts, per-algorithm aggregated means/standard deviations,
  paired geometric speedups (relative to ``baseline_algorithm`` and across all
  ordered pairs), Friedman/Iman-Davenport/Nemenyi omnibus and post-hoc statistics,
  and paired Wilcoxon + Holm-Bonferroni statistics against ``baseline_algorithm``.
  Contains zero hardcoded thesis algorithm names or prose.
  """
  if not all_results:
    raise ValueError("all_results dictionary is empty")

  _ = problem_configs  # Retained for signature symmetry with generate_result_tables.
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  macro_path = out_dir / filename

  problem_names: list[str] = sorted(all_results.keys())
  algorithm_names: list[str] = []
  for prob_res in all_results.values():
    for algo in prob_res:
      if algo not in algorithm_names:
        algorithm_names.append(algo)

  effective_baseline: str = resolve_baseline_algorithm(algorithm_names, baseline_algorithm)
  tokens = _build_algorithm_macro_tokens(algorithm_names)

  rep_counts: list[int] = []
  for prob_res in all_results.values():
    for entry in prob_res.values():
      reps = entry.get("repetitions")
      if not (isinstance(reps, (int, float)) and not isinstance(reps, bool) and int(reps) > 0):
        reps = entry.get("successful_runs")
      if not (isinstance(reps, (int, float)) and not isinstance(reps, bool) and int(reps) > 0):
        raw_times = entry.get("raw_times")
        if isinstance(raw_times, (list, tuple)):
          reps = len(raw_times)
      if isinstance(reps, (int, float)) and not isinstance(reps, bool) and int(reps) > 0:
        rep_counts.append(int(reps))
  total_runs = sum(rep_counts)

  lines: list[str] = [
    "% Auto-generated quantitative benchmark macros (`DP14-5`)",
    "% Strictly numeric values formatted with configurable decimal separator.",
    f"\\newcommand{{\\benchmarkNumProblems}}{{{len(problem_names)}}}",
    f"\\newcommand{{\\benchmarkNumAlgorithms}}{{{len(algorithm_names)}}}",
    f"\\newcommand{{\\benchmarkBaselineAlgorithm}}{{{escape_latex_identifier(effective_baseline)}}}",
    f"\\newcommand{{\\benchmarkTotalRuns}}{{{total_runs}}}",
  ]
  if rep_counts:
    min_rep, max_rep = min(rep_counts), max(rep_counts)
    rep_macro = str(min_rep) if min_rep == max_rep else f"{min_rep}--{max_rep}"
    lines.append(f"\\newcommand{{\\benchmarkRepetitions}}{{{rep_macro}}}")

  # Aggregated per-algorithm means, standard deviations, and baseline speedups
  lines.append("")
  lines.append("% Aggregated per-algorithm statistics across evaluated instances")
  for alg in algorithm_names:
    tok = tokens[alg]
    alg_gaps = [
      float(all_results[p][alg]["mean_gap"])
      for p in problem_names
      if alg in all_results[p] and "mean_gap" in all_results[p][alg]
    ]
    alg_costs = [
      float(all_results[p][alg]["mean_cost"])
      for p in problem_names
      if alg in all_results[p] and "mean_cost" in all_results[p][alg]
    ]
    alg_times = [
      float(all_results[p][alg]["mean_time"])
      for p in problem_names
      if alg in all_results[p] and "mean_time" in all_results[p][alg]
    ]
    if alg_gaps:
      mean_gap_tex = _format_decimal_latex(float(np.mean(alg_gaps)), 2, decimal_separator)
      std_gap_val = float(np.std(alg_gaps, ddof=1)) if len(alg_gaps) > 1 else 0.0
      std_gap_tex = _format_decimal_latex(std_gap_val, 2, decimal_separator)
      lines.append(f"\\newcommand{{\\meanGap{tok}}}{{{mean_gap_tex}}}")
      lines.append(f"\\newcommand{{\\stdGap{tok}}}{{{std_gap_tex}}}")
    elif alg_costs:
      mean_cost_tex = _format_decimal_latex(float(np.mean(alg_costs)), 2, decimal_separator)
      std_cost_val = float(np.std(alg_costs, ddof=1)) if len(alg_costs) > 1 else 0.0
      std_cost_tex = _format_decimal_latex(std_cost_val, 2, decimal_separator)
      lines.append(f"\\newcommand{{\\meanCost{tok}}}{{{mean_cost_tex}}}")
      lines.append(f"\\newcommand{{\\stdCost{tok}}}{{{std_cost_tex}}}")
    if alg_times:
      mean_time_tex = _format_decimal_latex(float(np.mean(alg_times)), 2, decimal_separator)
      std_time_val = float(np.std(alg_times, ddof=1)) if len(alg_times) > 1 else 0.0
      std_time_tex = _format_decimal_latex(std_time_val, 2, decimal_separator)
      lines.append(f"\\newcommand{{\\meanTime{tok}}}{{{mean_time_tex}}}")
      lines.append(f"\\newcommand{{\\stdTime{tok}}}{{{std_time_tex}}}")

    if alg == effective_baseline:
      lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(1.0, 2, decimal_separator)}}}")
    else:
      paired_speedups: list[float] = []
      for problem in problem_names:
        base_t = float(all_results[problem].get(effective_baseline, {}).get("mean_time", 0.0))
        cand_t = float(all_results[problem].get(alg, {}).get("mean_time", 0.0))
        if base_t > 0.0 and cand_t > 0.0:
          paired_speedups.append(base_t / cand_t)
      if paired_speedups:
        avg_sp = float(np.exp(np.mean(np.log(paired_speedups))))
        lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(avg_sp, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\speedupSampleSize{tok}}}{{{len(paired_speedups)}}}")

  # Pairwise geometric speedups (alg_a speedup over alg_b = mean_time(b) / mean_time(a))
  pairwise_lines: list[str] = []
  for alg_a in algorithm_names:
    for alg_b in algorithm_names:
      if alg_a == alg_b:
        continue
      ratios: list[float] = []
      for problem in problem_names:
        if alg_a in all_results[problem] and alg_b in all_results[problem]:
          t_a = float(all_results[problem][alg_a].get("mean_time", 0.0))
          t_b = float(all_results[problem][alg_b].get("mean_time", 0.0))
          if t_a > 0.0 and t_b > 0.0:
            ratios.append(t_b / t_a)
      if ratios:
        geomean = float(np.exp(np.mean(np.log(ratios))))
        tok_a = tokens[alg_a]
        tok_b = tokens[alg_b]
        pairwise_lines.append(
          f"\\newcommand{{\\speedup{tok_a}Vs{tok_b}}}{{{_format_decimal_latex(geomean, 2, decimal_separator)}}}"
        )
  if pairwise_lines:
    lines.append("")
    lines.append("% Pairwise geometric mean speedups (Fleming & Wallace, 1986)")
    lines.extend(pairwise_lines)

  # Omnibus Friedman / Iman-Davenport / Nemenyi and paired Wilcoxon + Holm
  analyzer = StatisticalAnalyzer(alpha=alpha)
  complete_problems = [p for p in problem_names if all(alg in all_results[p] for alg in algorithm_names)]
  has_all_gaps = bool(
    complete_problems and all("mean_gap" in all_results[p][alg] for p in complete_problems for alg in algorithm_names)
  )
  gap_metric_key = "mean_gap" if has_all_gaps else "mean_cost"

  if len(algorithm_names) >= 3 and len(complete_problems) >= 2:
    from benchmarking.analysis.cd_diagram import compute_cd_diagram_data

    for metric_tag, metric_key in (("Gap", gap_metric_key), ("Time", "mean_time")):
      valid_problems = [
        p for p in complete_problems if all(metric_key in all_results[p][alg] for alg in algorithm_names)
      ]
      if len(valid_problems) < 2:
        continue
      matrix = np.array(
        [[float(all_results[p][alg][metric_key]) for alg in algorithm_names] for p in valid_problems],
        dtype=float,
      )
      cd_data = compute_cd_diagram_data(matrix, algorithm_names, alpha=alpha)
      friedman_res = analyzer.friedman_test([matrix[:, j] for j in range(len(algorithm_names))])

      chi2_2 = _format_decimal_latex(friedman_res["statistic"], 2, decimal_separator)
      chi2_4 = _format_decimal_latex(friedman_res["statistic"], 4, decimal_separator)
      chi2_p = _format_pvalue_latex(friedman_res["p_value"], decimal_separator)
      f_tex = _format_decimal_latex(cd_data.friedman_f, 4, decimal_separator)
      f_p_tex = _format_pvalue_latex(cd_data.friedman_pvalue, decimal_separator)
      cd_tex = _format_decimal_latex(cd_data.critical_difference, 3, decimal_separator)

      lines.append("")
      lines.append(f"% Omnibus Friedman / Iman-Davenport & Nemenyi statistics ({metric_tag})")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}N}}{{{len(valid_problems)}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}K}}{{{len(algorithm_names)}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}ChiSq}}{{{chi2_2}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}ChiSqExact}}{{{chi2_4}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}ChiSqP}}{{{chi2_p}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}F}}{{{f_tex}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}DfOne}}{{{friedman_res['df1']}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}DfTwo}}{{{friedman_res['df2']}}}")
      lines.append(f"\\newcommand{{\\friedman{metric_tag}P}}{{{f_p_tex}}}")
      lines.append(f"\\newcommand{{\\nemenyi{metric_tag}CD}}{{{cd_tex}}}")
      for alg in algorithm_names:
        tok = tokens[alg]
        rank_tex = _format_decimal_latex(cd_data.ranks[alg], 2, decimal_separator)
        lines.append(f"\\newcommand{{\\meanRank{metric_tag}{tok}}}{{{rank_tex}}}")

      if not bool(np.all(np.ptp(matrix, axis=1) == 0.0)):
        nemenyi_res = analyzer.nemenyi_posthoc([matrix[:, j] for j in range(len(algorithm_names))], algorithm_names)
        p_mat = np.asarray(nemenyi_res["p_values_matrix"], dtype=float)
        if bool(np.all(np.isfinite(p_mat))):
          for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
              tok_i = tokens[algorithm_names[i]]
              tok_j = tokens[algorithm_names[j]]
              p_str = _format_pvalue_latex(float(p_mat[i, j]), decimal_separator)
              lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_i}Vs{tok_j}}}{{{p_str}}}")
              lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_j}Vs{tok_i}}}{{{p_str}}}")

  # Paired Wilcoxon signed-rank + Holm-Bonferroni against baseline
  non_baseline = [alg for alg in algorithm_names if alg != effective_baseline]
  if effective_baseline in algorithm_names and non_baseline and len(problem_names) >= 2:
    for metric_tag, metric_key in (("Gap", gap_metric_key), ("Time", "mean_time")):
      paired_records: list[tuple[str, float, float, float]] = []
      raw_p_vals: list[float] = []
      for cand in non_baseline:
        paired_probs = [
          p
          for p in problem_names
          if effective_baseline in all_results[p]
          and cand in all_results[p]
          and metric_key in all_results[p][effective_baseline]
          and metric_key in all_results[p][cand]
        ]
        if len(paired_probs) < 2:
          continue
        base_vals = np.array([float(all_results[p][effective_baseline][metric_key]) for p in paired_probs], dtype=float)
        cand_vals = np.array([float(all_results[p][cand][metric_key]) for p in paired_probs], dtype=float)
        if np.allclose(base_vals, cand_vals):
          w_stat, p_val, r_eff = 0.0, 1.0, 0.0
        else:
          wilcoxon_res = scipy_stats.wilcoxon(base_vals, cand_vals, zero_method="zsplit")
          w_stat = float(wilcoxon_res.statistic)
          p_val = float(wilcoxon_res.pvalue)
          r_eff = float(analyzer.rank_biserial_correlation(base_vals, cand_vals))
        paired_records.append((cand, w_stat, p_val, r_eff))
        raw_p_vals.append(p_val)

      if paired_records:
        holm_p_vals = _holm_adjusted_pvalues(raw_p_vals)
        lines.append("")
        lines.append(f"% Paired Wilcoxon + Holm-Bonferroni vs baseline ({metric_tag})")
        for (cand, w_stat, p_val, r_eff), p_holm in zip(paired_records, holm_p_vals, strict=True):
          tok = tokens[cand]
          w_tex = _format_decimal_latex(w_stat, 1, decimal_separator)
          p_tex = _format_pvalue_latex(p_val, decimal_separator)
          p_holm_tex = _format_pvalue_latex(p_holm, decimal_separator)
          eff_tex = _format_decimal_latex(r_eff, 2, decimal_separator)
          lines.append(f"\\newcommand{{\\wilcoxon{metric_tag}W{tok}}}{{{w_tex}}}")
          lines.append(f"\\newcommand{{\\wilcoxon{metric_tag}P{tok}}}{{{p_tex}}}")
          lines.append(f"\\newcommand{{\\wilcoxon{metric_tag}PHolm{tok}}}{{{p_holm_tex}}}")
          lines.append(f"\\newcommand{{\\wilcoxon{metric_tag}Effect{tok}}}{{{eff_tex}}}")

  macro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return macro_path


# =============================================================================
# Main API Function
# =============================================================================


def generate_result_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "chapter4_validation",
  baseline_algorithm: str | None = None,
  *,
  include_size_category_table: bool = False,
) -> None:
  """
  Generate comprehensive result tables, LaTeX macros, CD diagrams, and scaling figures.

  Creates Markdown/LaTeX tables (``<benchmark_name>.{md,tex}``), strictly
  quantitative LaTeX macros (``benchmark_macros.tex``), Demšar Critical Difference
  TikZ diagrams (``<benchmark_name>_cd_diagram_{gap,time}.tikz``), and empirical
  scaling vector figures (``scaling_time_gap.{pdf,svg}`` in sibling ``figures/``).
  """
  if not all_results:
    raise ValueError("all_results dictionary is empty")

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  problem_names: list[str] = sorted(all_results.keys())
  algorithm_names: list[str] = []
  for prob_res in all_results.values():
    for algo in prob_res:
      if algo not in algorithm_names:
        algorithm_names.append(algo)

  effective_baseline: str = resolve_baseline_algorithm(algorithm_names, baseline_algorithm)

  logging.info(f"Generating result tables with {len(problem_names)} problems, {len(algorithm_names)} algorithms")
  logging.info(f"Baseline algorithm: {effective_baseline}")

  baseline_times = {}
  for problem in problem_names:
    if effective_baseline in all_results[problem]:
      baseline_times[problem] = all_results[problem][effective_baseline]["mean_time"]
    else:
      first_alg: str = next(iter(all_results[problem].keys()))
      baseline_times[problem] = all_results[problem][first_alg]["mean_time"]
      logging.warning(f"Baseline {effective_baseline} not found for {problem}, using {first_alg}")

  md_lines = []
  tex_lines = []

  md_lines.extend(
    _generate_table1_markdown(
      all_results,
      problem_names,
      algorithm_names,
      effective_baseline,
      baseline_times,
      benchmark_name,
    )
  )
  tex_lines.extend(
    _generate_table1_latex(
      all_results,
      problem_names,
      algorithm_names,
      effective_baseline,
      baseline_times,
      benchmark_name,
    )
  )

  md_lines.extend(_generate_table2_markdown(all_results, problem_names, algorithm_names, effective_baseline))
  tex_lines.extend(_generate_table2_latex(all_results, problem_names, algorithm_names, effective_baseline))

  if include_size_category_table:
    md_lines.extend(_generate_table3_markdown(all_results, problem_configs, algorithm_names))
    tex_lines.extend(_generate_table3_latex(all_results, problem_configs, algorithm_names))

  if _has_phase_timings(all_results):
    md_lines.extend(_generate_phase_timings_markdown(all_results, problem_names, algorithm_names))
    tex_lines.extend(_generate_phase_timings_latex(all_results, problem_names, algorithm_names))

  tex_lines.append("\n\\end{document}")

  md_path = output_dir / f"{benchmark_name}.md"
  tex_path = output_dir / f"{benchmark_name}.tex"

  with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

  with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

  # Generate strictly quantitative LaTeX macros (`DP14-5` / Task 14.7)
  try:
    macros_path = generate_latex_macros(
      all_results=all_results,
      problem_configs=problem_configs,
      output_dir=output_dir,
      baseline_algorithm=effective_baseline,
    )
    logging.info(f"  LaTeX Macros: {macros_path}")
  except Exception as e:
    logging.warning("Could not generate LaTeX macros: %s", e)

  # Attempt to generate Demšar CD diagrams for both Quality (Gap/Cost) and Execution Time
  if len(algorithm_names) >= 3 and len(problem_names) >= 2:
    from benchmarking.analysis.cd_diagram import compute_cd_diagram_data, export_cd_diagram_tikz

    complete_problems = [p for p in problem_names if all(alg in all_results[p] for alg in algorithm_names)]
    gap_metric = (
      "mean_gap"
      if complete_problems
      and all("mean_gap" in all_results[p][alg] for p in complete_problems for alg in algorithm_names)
      else "mean_cost"
    )
    cd_specs: tuple[tuple[str, str, str], ...] = (
      (gap_metric, f"{benchmark_name}_cd_diagram_gap.tikz", "Optimality Gap"),
      ("mean_time", f"{benchmark_name}_cd_diagram_time.tikz", "Execution Time"),
    )
    for metric_key, filename, metric_label in cd_specs:
      valid_problems = [
        p
        for p in problem_names
        if all(alg in all_results[p] and metric_key in all_results[p][alg] for alg in algorithm_names)
      ]
      if len(valid_problems) < 2:
        continue
      try:
        matrix = np.array([[float(all_results[p][alg][metric_key]) for alg in algorithm_names] for p in valid_problems])
        cd_data = compute_cd_diagram_data(matrix, algorithm_names)
        tikz_path = output_dir / filename
        export_cd_diagram_tikz(cd_data, tikz_path, metric_label=metric_label)
        logging.info(f"  CD Diagram TikZ ({metric_label}): {tikz_path}")
      except Exception as e:
        logging.warning("Could not generate Demšar CD diagram for %s: %s", metric_label, e)

  # Generate scaling & trade-off figures (Task 13.28 / DP13-9 / DP14-3)
  figures_dir = (output_dir.parent / "figures") if output_dir.name == "tables" else (output_dir / "figures")
  try:
    from benchmarking.analysis.scaling_plots import generate_scaling_plots

    generated_plots = generate_scaling_plots(
      all_results=all_results,
      problem_configs=problem_configs,
      output_dir=figures_dir,
    )
    logging.info("  Scaling Plots: %s", ", ".join(str(p) for p in generated_plots))
  except Exception as e:
    logging.warning("Could not generate scaling plots: %s", e)

  logging.info("Tables generated:")
  logging.info(f"  Markdown: {md_path}")
  logging.info(f"  LaTeX: {tex_path}")
  logging.info("")
