"""
Report Generator Module for Benchmark V2
=========================================

Generates formatted output tables in Markdown and LaTeX formats.

Creates three comprehensive tables:
1. Summary Statistics by Problem - Detailed per-problem results
2. Algorithm Performance Comparison - Aggregated across all problems
3. Best Algorithm by Size Category - Performance by problem size range

Output formats:
- Markdown: Human-readable tables for documentation
- LaTeX: Camera-ready tables for thesis/publication

Usage:
    from benchmarking.analysis.report_generator import generate_result_tables
    from pathlib import Path

    output_dir = Path("results_v3/tables")
    generate_result_tables(
        all_results=results_dict,
        problem_configs=configs,
        output_dir=output_dir,
        benchmark_name="chapter4_validation"
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.stats_utils import resolve_baseline_algorithm

if TYPE_CHECKING:
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
  """Categorize problems by size ranges."""
  categories: dict[str, list[str]] = {
    "Small (n<100)": [],
    "Medium (100≤n<300)": [],
    "Large (n≥300)": [],
  }

  for problem in problem_names:
    size = _get_problem_size(problem, problem_configs)
    if size is None:
      continue

    if size < 100:
      categories["Small (n<100)"].append(problem)
    elif size < 300:
      categories["Medium (100≤n<300)"].append(problem)
    else:
      categories["Large (n≥300)"].append(problem)

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

    for alg in algs_in_problem:
      r = all_results[problem][alg]
      dagger_str = "†" if is_fallback else ""

      if alg == problem_base_alg:
        speedup = f"1.00x{dagger_str}"
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
  lines.append(
    f"\\caption{{Summary Statistics by Problem. Speedup relative to baseline algorithm ({safe_baseline}).}}"
  )
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

    stats.append({
      "algorithm": alg,
      "mean_gap": mean_gap,
      "std_gap": std_gap,
      "mean_time": mean_time,
      "std_time": std_time,
      "avg_speedup": avg_speedup,
      "sample_size": len(paired_speedups) if alg != baseline_algorithm else None,
      "is_baseline": alg == baseline_algorithm,
    })

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

    winners.append({
      "category": category,
      "problems": problems,
      "best_alg": best_alg,
      "best_gap": best_gap if best_alg is not None else None,
      "best_time": best_time if best_alg is not None else None,
    })

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
    lines.append(
      f"| {entry['category']} | {problems_str} | {best_alg_str} | "
      f"{gap_str} | {time_str} |"
    )

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

    lines.append(
      f"{entry['category']:20} & {problems_str:40} & {best_alg_str:15} & "
      f"{gap_str} & {time_str} \\\\"
    )

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
# Main API Function
# =============================================================================


def generate_result_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "chapter4_validation",
  baseline_algorithm: str | None = None,
) -> None:
  """
  Generate comprehensive result tables in Markdown and LaTeX formats.

  Creates three tables:
  1. Summary Statistics by Problem (per-problem detailed results)
  2. Algorithm Performance Comparison (aggregated across all problems)
  3. Best Algorithm by Size Category (categorized by problem size)
  """
  if not all_results:
    raise ValueError("all_results dictionary is empty")

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  problem_names: list[str] = sorted(all_results.keys())
  algo_set: set[str] = set()
  for prob_res in all_results.values():
    algo_set.update(prob_res.keys())
  algorithm_names: list[str] = sorted(algo_set)

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

  # Attempt to generate Demšar CD diagram if conditions are met
  if len(algorithm_names) >= 3 and len(problem_names) >= 2:
    valid_problems = [
      p
      for p in problem_names
      if all(alg in all_results[p] and "mean_cost" in all_results[p][alg] for alg in algorithm_names)
    ]
    if len(valid_problems) >= 2:
      try:
        from benchmarking.analysis.cd_diagram import compute_cd_diagram_data, export_cd_diagram_tikz

        matrix = np.array(
          [[float(all_results[p][alg]["mean_cost"]) for alg in algorithm_names] for p in valid_problems]
        )
        cd_data = compute_cd_diagram_data(matrix, algorithm_names)
        tikz_path = output_dir / f"{benchmark_name}_cd_diagram.tikz"
        export_cd_diagram_tikz(cd_data, tikz_path)
        logging.info(f"  CD Diagram TikZ: {tikz_path}")
      except Exception as e:
        logging.warning("Could not generate Demšar CD diagram: %s", e)

  logging.info("Tables generated:")
  logging.info(f"  Markdown: {md_path}")
  logging.info(f"  LaTeX: {tex_path}")
  logging.info("")
