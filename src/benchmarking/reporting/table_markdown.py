"""Markdown renderers for benchmark summary tables (Tables 1-4)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, LiteralString

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.reporting.style import ReportStyle
  from benchmarking.reporting.table_reports import PhaseTimingRow, Table1Row


def render_table1_markdown(
  rows: Sequence[Table1Row],
  has_fallback: bool,
  baseline_algorithm: str,
  benchmark_name: str,
  style: ReportStyle,
  *,
  show_vrp_columns: bool = False,
) -> list[str]:
  """Render Table 1 rows in Markdown format."""
  report_title = benchmark_name.replace("_", " ").title()
  baseline_label = style.display_name(baseline_algorithm)
  if show_vrp_columns:
    header_cols = (
      "| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Feas. (%) | Vehicles | Mean Time (s) | Speedup |"
    )
    align_cols = (
      "|:--------|:----------|----------:|--------:|--------:|----------:|---------:|--------------:|--------:|"
    )
  else:
    header_cols = "| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |"
    align_cols = "|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|"
  lines: list[str] = [
    f"# {report_title}\n",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    "## Table 1: Summary Statistics by Problem\n",
    header_cols,
    align_cols,
  ]

  for row in rows:
    dagger = "†" if row.is_fallback else ""
    speedup: LiteralString = f"{row.speedup:.2f}x{dagger}" if row.speedup is not None else "—"
    cost_cell = f"{row.mean_cost:.2f}" if row.mean_cost is not None else "—"
    std_cell = f"{row.std_cost:.2f}" if row.std_cost is not None else "—"
    gap_cell = f"{row.mean_gap:.2f}" if row.mean_gap is not None else "—"
    alg_label = style.display_name(row.algorithm)
    if show_vrp_columns:
      feas_cell = f"{row.feasibility_rate:.1f}" if row.feasibility_rate is not None else "—"
      veh_cell = f"{row.mean_vehicles:.2f}" if row.mean_vehicles is not None else "—"
      lines.append(
        f"| {row.problem} | {alg_label} | {cost_cell} | {std_cell} | "
        f"{gap_cell} | {feas_cell} | {veh_cell} | {row.mean_time:.2f} | {speedup} |"
      )
    else:
      lines.append(
        f"| {row.problem} | {alg_label} | {cost_cell} | {std_cell} | "
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


def render_table2_markdown(
  table2_stats: Sequence[dict[str, Any]],
  style: ReportStyle,
  *,
  show_vrp_columns: bool = False,
) -> list[str]:
  """Render Table 2 rows in Markdown format."""
  if show_vrp_columns:
    header_cols = (
      "| Algorithm | Mean Gap (%) | Std Gap | Feas. (%) | Vehicles | Mean Time (s) | Std Time | Avg Speedup |"
    )
    align_cols = (
      "|:----------|-------------:|--------:|----------:|---------:|--------------:|---------:|------------:|"
    )
  else:
    header_cols = "| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |"
    align_cols = "|:----------|-------------:|--------:|--------------:|---------:|------------:|"
  lines: list[str] = [
    "## Table 2: Algorithm Performance Comparison (All Problems)\n",
    header_cols,
    align_cols,
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
    if show_vrp_columns:
      feas_str = f"{s['feasibility_rate']:.1f}" if s.get("feasibility_rate") is not None else "—"
      veh_str = f"{s['mean_vehicles']:.2f}" if s.get("mean_vehicles") is not None else "—"
      lines.append(f"| {alg_label} | {gap_str} | {feas_str} | {veh_str} | {time_str} | {speedup_str} |")
    else:
      lines.append(f"| {alg_label} | {gap_str} | {time_str} | {speedup_str} |")

  return lines


def render_table3_markdown(
  winners: Sequence[dict[str, Any]],
  style: ReportStyle,
) -> list[str]:
  """Render Table 3 rows in Markdown format."""
  lines: list[str] = [
    "\n## Table 3: Best Algorithm by Problem Size\n",
    "| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |",
    "|:--------------|:---------|:---------------|-------------:|--------------:|",
  ]

  for entry in winners:
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


def render_phase_timings_markdown(
  rows: Sequence[PhaseTimingRow],
  has_floor_warning: bool,
  style: ReportStyle,
  phase_count: int,
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

  missing_cells = " | ".join(["-"] * phase_count)
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


__all__ = [
  "render_phase_timings_markdown",
  "render_table1_markdown",
  "render_table2_markdown",
  "render_table3_markdown",
]
