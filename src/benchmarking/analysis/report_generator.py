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

if TYPE_CHECKING:
  from benchmarking.runner.config_loader import ProblemConfig

# =============================================================================
# Helper Functions
# =============================================================================


def _get_baseline_algorithm(algorithm_names: list[str]) -> str:
  """Determine baseline algorithm for speedup calculations."""
  if "HybridNaive" in algorithm_names:
    return "HybridNaive"
  return algorithm_names[0] if algorithm_names else "Unknown"


def _get_problem_size(problem_name: str, problem_configs: list[ProblemConfig]) -> int | None:
  """Get size of a problem from configs."""
  for config in problem_configs:
    if config.name == problem_name:
      return int(config.size)
  return None


def _categorize_by_size(
  problem_names: list[str], problem_configs: list[ProblemConfig]
) -> dict[str, list[str]]:
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
) -> list[str]:
  """Generate Markdown format for Table 1 (summary by problem)."""
  lines = []

  lines.append("# Chapter 4 Validation Results\n")
  lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
  lines.append("## Table 1: Summary Statistics by Problem\n")
  lines.append("| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |")
  lines.append("|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|")

  for problem in problem_names:
    algs_in_problem = [alg for alg in algorithm_names if alg in all_results[problem]]

    for alg in algs_in_problem:
      r = all_results[problem][alg]

      speedup = "—"
      if problem in baseline_times and alg != baseline_algorithm:
        if r["mean_time"] >= 1e-6:
          speedup_val = baseline_times[problem] / r["mean_time"]
          speedup = f"{speedup_val:.2f}x"

      lines.append(
        f"| {problem} | {alg} | {r['mean_cost']:.2f} | {r['std_cost']:.2f} | "
        f"{r['mean_gap']:.2f} | {r['mean_time']:.2f} | {speedup} |"
      )

  lines.append("")
  lines.append(f"*Note: Speedup calculated relative to {baseline_algorithm} (— indicates baseline algorithm).*\n")

  return lines


def _generate_table1_latex(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_names: list[str],
  algorithm_names: list[str],
  baseline_algorithm: str,
  baseline_times: dict[str, float],
) -> list[str]:
  """Generate LaTeX format for Table 1 (summary by problem)."""
  lines = []

  lines.append("% Chapter 4 Validation Results - Auto-generated")
  lines.append(f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("% Compile with: pdflatex chapter4_validation.tex\n")
  lines.append("\\documentclass[11pt,a4paper]{article}")
  lines.append("\\usepackage{booktabs}  % Professional tables")
  lines.append("\\usepackage{multirow}  % Row spanning")
  lines.append("\\usepackage{geometry}  % Page margins")
  lines.append("\\usepackage{caption}   % Better captions")
  lines.append("\\geometry{margin=1in}")
  lines.append("\\captionsetup{font=small,labelfont=bf}\n")
  lines.append("\\title{Chapter 4 Validation: GPU-Accelerated Genetic Algorithm Benchmark}")
  lines.append(f"\\date{{{time.strftime('%B %d, %Y')}}}")
  lines.append("\\author{Automated Benchmark Report}\n")
  lines.append("\\begin{document}")
  lines.append("\\maketitle\n")
  lines.append("\\section{Summary Statistics by Problem}\n")

  lines.append("\\begin{table}[htbp]")
  lines.append("\\centering")
  lines.append(
    "\\caption{Summary Statistics by Problem. Speedup relative to baseline "
    "algorithm (HybridNaive when CPU not available).}"
  )
  lines.append("\\label{tab:chapter4_summary}")
  lines.append("\\begin{tabular}{llrrrrr}")
  lines.append("\\toprule")
  lines.append("Problem & Algorithm & Mean Cost & Std Dev & Gap (\\%) & Time (s) & Speedup \\\\")
  lines.append("\\midrule")

  for problem in problem_names:
    algs_in_problem = [alg for alg in algorithm_names if alg in all_results[problem]]

    for i, alg in enumerate(algs_in_problem):
      r = all_results[problem][alg]

      tex_speedup = "—"
      if problem in baseline_times and alg != baseline_algorithm:
        if r["mean_time"] >= 1e-6:
          speedup_val = baseline_times[problem] / r["mean_time"]
          tex_speedup = f"{speedup_val:.2f}$\\times$"

      if i == 0:
        problem_cell = f"\\multirow{{{len(algs_in_problem)}}}{{*}}{{{problem}}}"
      else:
        problem_cell = ""

      lines.append(
        f"{problem_cell:20} & {alg:15} & {r['mean_cost']:9.2f} & {r['std_cost']:7.2f} & "
        f"{r['mean_gap']:7.2f} & {r['mean_time']:8.2f} & {tex_speedup:>12} \\\\"
      )

    if problem != problem_names[-1]:
      lines.append("\\midrule")

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  lines.append(
    f"{{\\small \\textit{{Note:}} Speedup calculated relative to "
    f"{baseline_algorithm}. — indicates baseline algorithm.}}"
  )
  lines.append("\\end{table}")
  lines.append("\\clearpage\n")

  return lines


# =============================================================================
# Table 2: Algorithm Performance Comparison
# =============================================================================


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

  baseline_times_all = [
    all_results[problem][baseline_algorithm]["mean_time"]
    for problem in problem_names
    if baseline_algorithm in all_results[problem]
  ]
  baseline_mean_time = float(np.mean(baseline_times_all)) if baseline_times_all else None

  for alg in algorithm_names:
    all_gaps = []
    all_times = []

    for problem in problem_names:
      if alg in all_results[problem]:
        all_gaps.append(all_results[problem][alg]["mean_gap"])
        all_times.append(all_results[problem][alg]["mean_time"])

    if not all_gaps:
      continue

    mean_gap = np.mean(all_gaps)
    std_gap = np.std(all_gaps)
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)

    speedup_str = "—"
    if alg != baseline_algorithm and baseline_mean_time is not None and mean_time > 0:
      avg_speedup = baseline_mean_time / mean_time
      speedup_str = f"{avg_speedup:.2f}x"

    lines.append(f"| {alg} | {mean_gap:.2f} | {std_gap:.2f} | {mean_time:.2f} | {std_time:.2f} | {speedup_str} |")

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

  baseline_times_all = [
    all_results[problem][baseline_algorithm]["mean_time"]
    for problem in problem_names
    if baseline_algorithm in all_results[problem]
  ]
  baseline_mean_time = float(np.mean(baseline_times_all)) if baseline_times_all else None

  for alg in algorithm_names:
    all_gaps = []
    all_times = []

    for problem in problem_names:
      if alg in all_results[problem]:
        all_gaps.append(all_results[problem][alg]["mean_gap"])
        all_times.append(all_results[problem][alg]["mean_time"])

    if not all_gaps:
      continue

    mean_gap = np.mean(all_gaps)
    std_gap = np.std(all_gaps)
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)

    tex_speedup = "—"
    if alg != baseline_algorithm and baseline_mean_time is not None and mean_time > 0:
      avg_speedup = baseline_mean_time / mean_time
      tex_speedup = f"{avg_speedup:.2f}$\\times$"

    lines.append(
      f"{alg:15} & {mean_gap:8.2f} & {std_gap:7.2f} & {mean_time:8.2f} & {std_time:8.2f} & {tex_speedup:>12} \\\\"
    )

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  lines.append(
    f"{{\\small \\textit{{Note:}} Speedup calculated relative to "
    f"{baseline_algorithm}. — indicates baseline algorithm.}}"
  )
  lines.append("\\end{table}")
  lines.append("\\clearpage\n")

  return lines


# =============================================================================
# Table 3: Best Algorithm by Size Category
# =============================================================================


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

  problem_names = list(all_results.keys())
  categories = _categorize_by_size(problem_names, problem_configs)

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
        if alg in all_results[prob]:
          gaps.append(all_results[prob][alg]["mean_gap"])
          times.append(all_results[prob][alg]["mean_time"])

      if gaps:
        avg_gap = float(np.mean(gaps))
        if avg_gap < best_gap:
          best_gap = avg_gap
          best_alg = alg
          best_time = float(np.mean(times))

    problems_str = ", ".join(problems)

    lines.append(f"| {category} | {problems_str} | {best_alg} | {best_gap:.2f} | {best_time:.2f} |")

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

  problem_names = list(all_results.keys())
  categories = _categorize_by_size(problem_names, problem_configs)

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
        if alg in all_results[prob]:
          gaps.append(all_results[prob][alg]["mean_gap"])
          times.append(all_results[prob][alg]["mean_time"])

      if gaps:
        avg_gap = float(np.mean(gaps))
        if avg_gap < best_gap:
          best_gap = avg_gap
          best_alg = alg
          best_time = float(np.mean(times))

    problems_str = ", ".join(problems)

    if len(problems_str) > 60:
      problems_str = problems_str[:57] + "..."

    lines.append(f"{category:20} & {problems_str:40} & {best_alg:15} & {best_gap:8.2f} & {best_time:8.2f} \\\\")

  lines.append("\\bottomrule")
  lines.append("\\end{tabular}")
  lines.append("\\vspace{0.5em}")
  lines.append("{\\small \\textit{Note:} Best algorithm determined by lowest mean gap to known optimum.}")
  lines.append("\\end{table}")
  lines.append("\n\\end{document}")

  return lines


# =============================================================================
# Main API Function
# =============================================================================


def generate_result_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: list[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "chapter4_validation",
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

  problem_names = sorted(all_results.keys())
  algo_set: set[str] = set()
  for prob_res in all_results.values():
    algo_set.update(prob_res.keys())
  algorithm_names = sorted(algo_set)

  baseline_algorithm = _get_baseline_algorithm(algorithm_names)

  logging.info(f"Generating result tables with {len(problem_names)} problems, {len(algorithm_names)} algorithms")
  logging.info(f"Baseline algorithm: {baseline_algorithm}")

  baseline_times = {}
  for problem in problem_names:
    if baseline_algorithm in all_results[problem]:
      baseline_times[problem] = all_results[problem][baseline_algorithm]["mean_time"]
    else:
      first_alg = next(iter(all_results[problem].keys()))
      baseline_times[problem] = all_results[problem][first_alg]["mean_time"]
      logging.warning(f"Baseline {baseline_algorithm} not found for {problem}, using {first_alg}")

  md_lines = []
  tex_lines = []

  md_lines.extend(
    _generate_table1_markdown(
      all_results,
      problem_names,
      algorithm_names,
      baseline_algorithm,
      baseline_times,
    )
  )
  tex_lines.extend(
    _generate_table1_latex(
      all_results,
      problem_names,
      algorithm_names,
      baseline_algorithm,
      baseline_times,
    )
  )

  md_lines.extend(_generate_table2_markdown(all_results, problem_names, algorithm_names, baseline_algorithm))
  tex_lines.extend(_generate_table2_latex(all_results, problem_names, algorithm_names, baseline_algorithm))

  md_lines.extend(_generate_table3_markdown(all_results, problem_configs, algorithm_names))
  tex_lines.extend(_generate_table3_latex(all_results, problem_configs, algorithm_names))

  md_path = output_dir / f"{benchmark_name}.md"
  tex_path = output_dir / f"{benchmark_name}.tex"

  with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

  with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

  logging.info("Tables generated:")
  logging.info(f"  Markdown: {md_path}")
  logging.info(f"  LaTeX: {tex_path}")
  logging.info("")
