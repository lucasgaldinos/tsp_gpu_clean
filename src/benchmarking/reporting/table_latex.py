"""LaTeX renderers for benchmark summary tables (Tables 1-4)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from benchmarking.reporting.style import escape_latex_identifier

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.reporting.style import ReportStyle
  from benchmarking.reporting.table_reports import PhaseTimingRow, Table1Row


def render_table1_latex(
  rows: Sequence[Table1Row],
  has_fallback: bool,
  baseline_algorithm: str,
  benchmark_name: str,
  style: ReportStyle,
  *,
  show_vrp_columns: bool = False,
) -> list[str]:
  """Render Table 1 rows in LaTeX format."""
  report_title = benchmark_name.replace("_", " ").title()
  safe_baseline = escape_latex_identifier(style.display_name(baseline_algorithm))

  tabular_spec = "\\begin{tabular}{llrrrrrrr}" if show_vrp_columns else "\\begin{tabular}{llrrrrr}"
  header_tex = (
    "Problem & Algorithm & Mean Cost & Std Dev & Gap (\\%) & Feas. (\\%) & Vehicles & Time (s) & Speedup \\\\"
    if show_vrp_columns
    else "Problem & Algorithm & Mean Cost & Std Dev & Gap (\\%) & Time (s) & Speedup \\\\"
  )
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
    tabular_spec,
    "\\toprule",
    header_tex,
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
    cost_tex = f"{row.mean_cost:9.2f}" if row.mean_cost is not None else f"{'—':>9}"
    std_tex = f"{row.std_cost:7.2f}" if row.std_cost is not None else f"{'—':>7}"
    gap_tex = f"{row.mean_gap:7.2f}" if row.mean_gap is not None else f"{'—':>7}"
    if show_vrp_columns:
      feas_tex = f"{row.feasibility_rate:6.1f}" if row.feasibility_rate is not None else f"{'—':>6}"
      veh_tex = f"{row.mean_vehicles:6.2f}" if row.mean_vehicles is not None else f"{'—':>6}"
      lines.append(
        f"{problem_cell:20} & {safe_alg:15} & {cost_tex} & {std_tex} & "
        f"{gap_tex} & {feas_tex} & {veh_tex} & {row.mean_time:8.2f} & {tex_speedup:>12} \\\\"
      )
    else:
      lines.append(
        f"{problem_cell:20} & {safe_alg:15} & {cost_tex} & {std_tex} & "
        f"{gap_tex} & {row.mean_time:8.2f} & {tex_speedup:>12} \\\\"
      )
    if row.is_last_in_problem and not row.is_last_problem:
      lines.append("\\midrule")

  lines.extend(["\\bottomrule", "\\end{tabular}", "\\vspace{0.5em}"])
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


def render_table2_latex(
  table2_stats: Sequence[dict[str, Any]],
  baseline_algorithm: str,
  style: ReportStyle,
  *,
  show_vrp_columns: bool = False,
) -> list[str]:
  """Render Table 2 rows in LaTeX format."""
  tabular_spec = "\\begin{tabular}{lrrrrrrr}" if show_vrp_columns else "\\begin{tabular}{lrrrrr}"
  header_tex = (
    "Algorithm & Mean Gap (\\%) & Std Gap & Feas. (\\%) & Vehicles & Mean Time (s) & Std Time & Avg Speedup \\\\"
    if show_vrp_columns
    else "Algorithm & Mean Gap (\\%) & Std Gap & Mean Time (s) & Std Time & Avg Speedup \\\\"
  )
  lines: list[str] = [
    "\\section{Algorithm Performance Comparison}\n",
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Algorithm Performance Comparison Across All Problems (aggregated statistics)}",
    f"\\label{{{style.table_label('algorithm_comparison')}}}",
    tabular_spec,
    "\\toprule",
    header_tex,
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
    if show_vrp_columns:
      feas_tex = f"{s['feasibility_rate']:6.1f}" if s.get("feasibility_rate") is not None else f"{'—':>6}"
      veh_tex = f"{s['mean_vehicles']:6.2f}" if s.get("mean_vehicles") is not None else f"{'—':>6}"
      lines.append(f"{safe_alg:15} & {gap_tex} & {feas_tex} & {veh_tex} & {time_tex} & {tex_speedup:>12} \\\\")
    else:
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


def render_table3_latex(
  winners: Sequence[dict[str, Any]],
  style: ReportStyle,
) -> list[str]:
  """Render Table 3 rows in LaTeX format."""
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

  for entry in winners:
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


def render_phase_timings_latex(
  rows: Sequence[PhaseTimingRow],
  has_floor_warning: bool,
  style: ReportStyle,
  phase_count: int,
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

  missing_cells = " & ".join(["-"] * phase_count)
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


__all__ = [
  "render_phase_timings_latex",
  "render_table1_latex",
  "render_table2_latex",
  "render_table3_latex",
]
