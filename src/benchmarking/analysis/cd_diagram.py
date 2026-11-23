"""Automated Demšar Critical Difference (CD) diagram generator in LaTeX/TikZ.

Implements the Demšar (2006) protocol for multi-algorithm metaheuristic comparisons:
- Non-parametric omnibus Friedman test (Friedman 1937, Iman & Davenport 1980);
- Average rank conservation constraint validation (sum R_j = k(k+1)/2);
- Nemenyi (1963) post-hoc Critical Difference with exact Studentized range scaling;
- Connected maximal non-significant clique detection (|R_i - R_j| <= CD);
- Publication-ready vector TikZ code generation for thesis figures.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from benchmarking.analysis.report_generator import escape_latex_identifier
from benchmarking.analysis.statistics import compute_iman_davenport_f, compute_mean_ranks
from benchmarking.analysis.stats_utils import compute_nemenyi_critical_distance

if TYPE_CHECKING:
  from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CDResults:
  """Results from Demšar Critical Difference statistical analysis.

  Attributes:
      k_algorithms: Number of compared algorithms (k >= 3).
      n_instances: Number of benchmark instances (N >= 2).
      ranks: Mapping from algorithm name to its average rank across instances.
      rank_sum: Sum of average ranks across all algorithms (conserved at k(k+1)/2).
      critical_difference: Nemenyi Critical Difference (CD) threshold.
      friedman_f: Iman-Davenport F_F omnibus test statistic.
      friedman_pvalue: F-distribution p-value for the Iman-Davenport omnibus test.
      cliques: List of maximal non-significant cliques as (algo1, algo2, rank1, rank2).
      alpha: Significance level (default 0.05).
  """

  k_algorithms: int
  n_instances: int
  ranks: dict[str, float]
  rank_sum: float
  critical_difference: float
  friedman_f: float
  friedman_pvalue: float
  cliques: list[tuple[str, str, float, float]]
  alpha: float = 0.05


def compute_cd_diagram_data(
  ranks_matrix: np.ndarray,
  algorithm_names: list[str],
  alpha: float = 0.05,
  higher_is_better: bool = False,
) -> CDResults:
  """Compute Demšar ranks, Iman-Davenport F_F omnibus test, Nemenyi CD, and cliques.

  Args:
      ranks_matrix: 2D array of shape (N, k) containing raw performance scores or ranks.
          Rows represent benchmark instances, columns represent algorithms.
      algorithm_names: Names of algorithms corresponding to columns of ranks_matrix.
      alpha: Significance level (default 0.05).
      higher_is_better: If True, higher values receive lower ranks (rank 1 = best).
          Default False (lower cost or runtime receives rank 1).

  Returns:
      CDResults containing statistical metrics, ranks, CD threshold, and cliques.

  Raises:
      ValueError: If matrix dimensions or algorithm names are invalid, or if the
          rank conservation constraint is violated.
  """
  mat = np.asarray(ranks_matrix, dtype=float)
  if mat.ndim != 2:
    raise ValueError(f"ranks_matrix must be 2-dimensional (N instances, k algorithms). Got shape {mat.shape}.")

  n_instances, k_algorithms = mat.shape
  if len(algorithm_names) != k_algorithms:
    raise ValueError(
      f"algorithm_names length ({len(algorithm_names)}) must match columns in ranks_matrix ({k_algorithms})."
    )

  if k_algorithms < 3:
    raise ValueError(f"Demšar CD analysis requires at least 3 algorithms (k >= 3). Got k={k_algorithms}.")

  if n_instances < 2:
    raise ValueError(f"Demšar CD analysis requires at least 2 instances (N >= 2). Got N={n_instances}.")

  # 1 & 2. Compute instance-level ranks (1 = best) and average across instances
  ranks_dict = compute_mean_ranks(mat, algorithm_names, higher_is_better=higher_is_better)
  rank_sum = float(sum(ranks_dict.values()))

  # Rank conservation law: sum_{j=1}^k R_j = k(k + 1) / 2
  expected_sum = (k_algorithms * (k_algorithms + 1)) / 2.0
  if not np.isclose(rank_sum, expected_sum, atol=1e-5):
    raise ValueError(
      f"Rank conservation law violated: sum of average ranks={rank_sum:.6f}, expected={expected_sum:.6f}."
    )

  # 3. Friedman omnibus test with Iman-Davenport (1980) F_F correction
  if bool(np.all(np.ptp(mat, axis=1) == 0.0)):
    f_stat, p_val_f = 0.0, 1.0
  else:
    try:
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        chi2, _ = stats.friedmanchisquare(*[mat[:, j] for j in range(k_algorithms)])
      if np.isnan(chi2):
        f_stat, p_val_f = 0.0, 1.0
      else:
        f_stat, _, _, p_val_f = compute_iman_davenport_f(float(chi2), n_instances, k_algorithms)
    except Exception as e:
      logger.warning("Friedman omnibus test computation failed: %s. Using default p=1.0.", e)
      f_stat, p_val_f = 0.0, 1.0

  # 4. Critical Difference (Nemenyi 1963 / Demšar 2006, BUG-47 / DP12-24)
  cd = compute_nemenyi_critical_distance(k_algorithms, n_instances, alpha)

  # 5. Connected maximal cliques detection (|R_i - R_j| <= CD)
  sorted_algos = sorted(ranks_dict.items(), key=lambda item: item[1])
  all_cliques: list[tuple[int, int]] = []
  for i in range(k_algorithms):
    for j in range(i + 1, k_algorithms):
      diff = sorted_algos[j][1] - sorted_algos[i][1]
      if diff <= cd + 1e-9:
        all_cliques.append((i, j))

  # Filter to maximal cliques (not strictly contained in another clique)
  maximal_cliques: list[tuple[str, str, float, float]] = []
  for i, j in all_cliques:
    is_subset = False
    for i_other, j_other in all_cliques:
      if (i_other, j_other) == (i, j):
        continue
      if i_other <= i and j_other >= j:
        is_subset = True
        break
    if not is_subset:
      maximal_cliques.append((sorted_algos[i][0], sorted_algos[j][0], sorted_algos[i][1], sorted_algos[j][1]))

  return CDResults(
    k_algorithms=k_algorithms,
    n_instances=n_instances,
    ranks=ranks_dict,
    rank_sum=rank_sum,
    critical_difference=cd,
    friedman_f=float(f_stat),
    friedman_pvalue=float(p_val_f),
    cliques=maximal_cliques,
    alpha=alpha,
  )


def export_cd_diagram_tikz(
  cd_results: CDResults,
  output_path: Path | str | None = None,
  standalone: bool = False,
  metric_label: str | None = None,
  *,
  display_names: Mapping[str, str] | None = None,
  rank_label: str = "Rank",
  decimal_sep: str = ".",
) -> str:
  """Export publication-grade TikZ vector code from CDResults.

  Args:
      cd_results: CDResults instance from compute_cd_diagram_data.
      output_path: Optional path to write the generated .tikz file.
      standalone: If True, wraps in a complete compilable standalone document.
      metric_label: Optional metric label recorded in the generated TikZ header comment.
      display_names: Optional mapping from canonical algorithm keys to rendered labels.
      rank_label: Label placed at the right end of the rank axis (default ``"Rank"``).
      decimal_sep: Decimal separator for rendered numbers (e.g. ``","`` for pt-BR).

  Returns:
      TikZ code as a string.
  """
  k = cd_results.k_algorithms
  cd = cd_results.critical_difference
  sorted_items = sorted(cd_results.ranks.items(), key=lambda x: (x[1], x[0]))
  top_clique_y = 0.35 + max(0, len(cd_results.cliques) - 1) * 0.15
  cd_bar_y = max(1.1, top_clique_y + 0.30)

  def _fmt_text(value: float, decimals: int) -> str:
    formatted = f"{value:.{decimals}f}"
    return formatted.replace(".", decimal_sep) if decimal_sep != "." else formatted

  def _fmt_math(value: float, decimals: int) -> str:
    formatted = f"{value:.{decimals}f}"
    return formatted.replace(".", f"{{{decimal_sep}}}") if decimal_sep != "." else formatted

  safe_rank_label = escape_latex_identifier(rank_label)

  tikz_body: list[str] = [
    "% Auto-generated Demšar Critical Difference Diagram in TikZ",
    *(
      [f"% Metric: {metric_label}"]
      if metric_label
      else []
    ),
    (
      f"% Generated for k = {k} algorithms across N = {cd_results.n_instances} instances "
      f"(CD = {cd:.3f}, alpha = {cd_results.alpha}, F_F = {cd_results.friedman_f:.3f}, "
      f"p = {cd_results.friedman_pvalue:.4g})"
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
      f"{{CD = {_fmt_text(cd, 3)} ($\\alpha = {_fmt_math(cd_results.alpha, 2)}$)}} ({1.0 + cd:.3f}, {cd_bar_y:.2f});"
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
    shown_name = display_names.get(name, name) if display_names else name
    safe_name = escape_latex_identifier(shown_name)
    rank_text = _fmt_text(r, 2)
    coord_name = re.sub(r"[^a-zA-Z0-9]", "", name)
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
    shown_name = display_names.get(name, name) if display_names else name
    safe_name = escape_latex_identifier(shown_name)
    rank_text = _fmt_text(r, 2)
    coord_name = re.sub(r"[^a-zA-Z0-9]", "", name)
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
    tick_label = f"{tick}{decimal_sep}0"
    tikz_body.append(
      f"    \\node[below=2pt, font=\\footnotesize, fill=white, inner sep=1.5pt] at ({tick}.0, -0.08) {{{tick_label}}};"
    )

  tikz_body.append("")
  tikz_body.append("    % 4. Non-Significant Cliques (|R_i - R_j| <= CD)")
  y_offset = 0.35
  for _, _, r_start, r_end in cd_results.cliques:
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
