"""Pure Demšar Critical Difference (CD) TikZ diagram renderer.

Consumes a precomputed ``OmnibusTestResult`` from ``benchmarking.analysis.statistics``
without performing any statistical computation itself, emitting publication-ready
LaTeX/TikZ vector code with deduplicated coordinate identifiers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarking.reporting.style import escape_latex_identifier

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from benchmarking.analysis.statistics import OmnibusTestResult
  from benchmarking.reporting.style import ReportStyle


def _build_tikz_coord_ids(names: Sequence[str]) -> dict[str, str]:
  """Map each canonical algorithm name to a unique alphanumeric TikZ coordinate ID."""
  coord_ids: dict[str, str] = {}
  used: dict[str, int] = {}
  for name in names:
    base = re.sub(r"[^a-zA-Z0-9]", "", name) or "Alg"
    if base not in used:
      used[base] = 1
      coord_ids[name] = base
    else:
      used[base] += 1
      suffix = used[base]
      candidate = f"{base}_{suffix}"
      while candidate in used:
        suffix += 1
        candidate = f"{base}_{suffix}"
      used[candidate] = 1
      coord_ids[name] = candidate
  return coord_ids


def export_cd_diagram_tikz(
  omnibus: OmnibusTestResult,
  output_path: Path | str | None = None,
  standalone: bool = False,
  metric_label: str | None = None,
  *,
  display_names: Mapping[str, str] | None = None,
  rank_label: str | None = None,
  decimal_sep: str | None = None,
  style: ReportStyle | None = None,
) -> str:
  """Export publication-grade TikZ vector code from an ``OmnibusTestResult``.

  Args:
      omnibus: Precomputed ``OmnibusTestResult`` from ``run_friedman_nemenyi_analysis``
          or ``CohortAnalysis.{quality_omnibus, time_omnibus}``.
      output_path: Optional path to write the generated ``.tikz`` file.
      standalone: If ``True``, wraps in a complete compilable standalone LaTeX document.
      metric_label: Optional metric label recorded in the generated TikZ header comment.
      display_names: Optional mapping from canonical algorithm keys to rendered labels
          (defaults to ``style.display_names`` when ``style`` is provided).
      rank_label: Label placed at the right end of the rank axis (defaults to
          ``style.cd_rank_label`` when ``style`` is provided, else ``"Rank"``).
      decimal_sep: Decimal separator for rendered numbers (defaults to
          ``style.cd_decimal_sep`` when ``style`` is provided, else ``"."``).
      style: Optional ``ReportStyle`` providing presentation defaults overridden by
          any explicitly supplied keyword arguments.

  Returns:
      TikZ code as a string.
  """
  effective_display_names = display_names if display_names is not None else (style.display_names if style else None)
  effective_rank_label = rank_label if rank_label is not None else (style.cd_rank_label if style else "Rank")
  effective_decimal_sep = decimal_sep if decimal_sep is not None else (style.cd_decimal_sep if style else ".")

  k = omnibus.k_algorithms
  cd = omnibus.critical_difference
  sorted_items = sorted(omnibus.ranks.items(), key=lambda x: (x[1], x[0]))
  coord_ids = _build_tikz_coord_ids([name for name, _ in sorted_items])
  top_clique_y = 0.35 + max(0, len(omnibus.cliques) - 1) * 0.15
  cd_bar_y = max(1.1, top_clique_y + 0.30)

  def _fmt_text(value: float, decimals: int) -> str:
    formatted = f"{value:.{decimals}f}"
    return formatted.replace(".", effective_decimal_sep) if effective_decimal_sep != "." else formatted

  def _fmt_math(value: float, decimals: int) -> str:
    formatted = f"{value:.{decimals}f}"
    return formatted.replace(".", f"{{{effective_decimal_sep}}}") if effective_decimal_sep != "." else formatted

  safe_rank_label = escape_latex_identifier(effective_rank_label)

  tikz_body: list[str] = [
    "% Auto-generated Demšar Critical Difference Diagram in TikZ",
    *([f"% Metric: {metric_label}"] if metric_label else []),
    (
      f"% Generated for k = {k} algorithms across N = {omnibus.n_blocks} instances "
      f"(CD = {cd:.3f}, alpha = {omnibus.alpha}, F_F = {omnibus.friedman_f:.3f}, "
      f"p = {omnibus.friedman_pvalue:.4g})"
    ),
    r"\begin{tikzpicture}[",
    r"    scale=2.2,",
    r"    axis/.style={thick, -{Stealth[length=2.5mm]}},",
    r"    tick/.style={thick},",
    r"    subtick/.style={thin, gray},",
    r"    clique/.style={line width=2.4pt, color=blue!75!black, line cap=round},",
    r"    cdbar/.style={line width=1.5pt, {Bar[width=4mm]}-{Bar[width=4mm]}},",
    r"    algomarker/.style={circle, fill=black, inner sep=1.2pt},",
    r"    labelnode/.style={font=\small\bfseries, align=center}",
    r"]",
    "",
    "    % 1. CD Reference Bar at the Top",
    (
      f"    \\draw[cdbar] (1.0, {cd_bar_y:.2f}) -- "
      f"node[above=2pt, font=\\footnotesize\\bfseries] "
      f"{{CD = {_fmt_text(cd, 3)} ($\\alpha = {_fmt_math(omnibus.alpha, 2)}$)}} ({1.0 + cd:.3f}, {cd_bar_y:.2f});"
    ),
    "",
    "    % 2. Main Horizontal Rank Axis (1.0 to k.0)",
    (
      f"    \\draw[axis] (0.8, 0.0) -- ({k + 0.3:.1f}, 0.0) "
      f"node[right, font=\\footnotesize\\bfseries] {{{safe_rank_label}}};"
    ),
    "",
    "    % Major Ticks",
  ]

  for tick in range(1, k + 1):
    tikz_body.append(f"    \\draw[tick] ({tick}.0, 0.08) -- ({tick}.0, -0.08);")

  tikz_body.append("")
  tikz_body.append("    % Minor Sub-ticks (every 0.5)")
  for tick in range(1, k):
    tikz_body.append(f"    \\draw[subtick] ({tick}.5, 0.04) -- ({tick}.5, -0.04);")

  tikz_body.append("")
  tikz_body.append("    % 3. Algorithm Data Points and Drop Lines (staggered Demšar elbows avoid overlap on ties)")
  mid = k // 2
  left_items = sorted_items[:mid]
  right_items = sorted_items[mid:]
  x_left_elbow = 0.72
  x_left_label = 0.68
  x_right_elbow = float(k) + 0.28
  x_right_label = float(k) + 0.32

  for idx, (name, r) in enumerate(left_items):
    shown_name = effective_display_names.get(name, name) if effective_display_names else name
    safe_name = escape_latex_identifier(shown_name)
    rank_text = _fmt_text(r, 2)
    coord_name = coord_ids[name]
    y_drop = -0.48 - 0.50 * idx
    tikz_body.extend(
      [
        f"    \\coordinate (A_{coord_name}) at ({r:.3f}, 0.0);",
        f"    \\node[algomarker] at (A_{coord_name}) {{}};",
        f"    \\draw[thick] (A_{coord_name}) -- ({r:.3f}, {y_drop:.2f}) -- ({x_left_elbow:.2f}, {y_drop:.2f});",
        f"    \\node[labelnode, anchor=east] at ({x_left_label:.2f}, {y_drop:.2f}) {{{safe_name}\\\\({rank_text})}};",
      ]
    )

  for idx, (name, r) in enumerate(right_items):
    shown_name = effective_display_names.get(name, name) if effective_display_names else name
    safe_name = escape_latex_identifier(shown_name)
    rank_text = _fmt_text(r, 2)
    coord_name = coord_ids[name]
    depth_idx = len(right_items) - 1 - idx
    y_drop = -0.48 - 0.50 * depth_idx
    tikz_body.extend(
      [
        f"    \\coordinate (A_{coord_name}) at ({r:.3f}, 0.0);",
        f"    \\node[algomarker] at (A_{coord_name}) {{}};",
        f"    \\draw[thick] (A_{coord_name}) -- ({r:.3f}, {y_drop:.2f}) -- ({x_right_elbow:.2f}, {y_drop:.2f});",
        f"    \\node[labelnode, anchor=west] at ({x_right_label:.2f}, {y_drop:.2f}) {{{safe_name}\\\\({rank_text})}};",
      ]
    )

  tikz_body.append("")
  tikz_body.append("    % Major Tick Labels (drawn after drop lines so white fill masks crossings)")
  for tick in range(1, k + 1):
    tick_label = f"{tick}{effective_decimal_sep}0"
    tikz_body.append(
      f"    \\node[below=2pt, font=\\footnotesize, fill=white, inner sep=1.5pt] at ({tick}.0, -0.08) {{{tick_label}}};"
    )

  tikz_body.append("")
  tikz_body.append("    % 4. Non-Significant Cliques (|R_i - R_j| <= CD)")
  y_offset = 0.35
  for _, _, r_start, r_end in omnibus.cliques:
    tikz_body.extend(
      [
        f"    \\draw[clique] ({r_start:.3f}, {y_offset:.2f}) -- ({r_end:.3f}, {y_offset:.2f});",
        f"    \\draw[thin, blue!75!black, dashed] ({r_start:.3f}, 0.0) -- ({r_start:.3f}, {y_offset:.2f});",
        f"    \\draw[thin, blue!75!black, dashed] ({r_end:.3f}, 0.0) -- ({r_end:.3f}, {y_offset:.2f});",
      ]
    )
    y_offset += 0.15

  tikz_body.append(r"\end{tikzpicture}")

  if standalone:
    full_doc = [
      r"\documentclass[tikz,border=10pt]{standalone}",
      r"\usepackage{tikz}",
      r"\usetikzlibrary{arrows.meta,positioning,calc}",
      "",
      r"\begin{document}",
      "\n".join(tikz_body),
      r"\end{document}",
      "",
    ]
    final_output = "\n".join(full_doc)
  else:
    final_output = "\n".join(tikz_body) + "\n"

  if output_path is not None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
      f.write(final_output)

  return final_output


__all__ = ["export_cd_diagram_tikz"]
