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

import numpy as np
from scipy import stats

from benchmarking.analysis.report_generator import escape_latex_identifier
from benchmarking.analysis.stats_utils import compute_nemenyi_critical_distance

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
      friedman_chi2: Friedman omnibus chi-square test statistic.
      friedman_pvalue: Asymptotic p-value for the Friedman omnibus test.
      cliques: List of maximal non-significant cliques as (algo1, algo2, rank1, rank2).
      alpha: Significance level (default 0.05).
  """

  k_algorithms: int
  n_instances: int
  ranks: dict[str, float]
  rank_sum: float
  critical_difference: float
  friedman_chi2: float
  friedman_pvalue: float
  cliques: list[tuple[str, str, float, float]]
  alpha: float = 0.05


def compute_cd_diagram_data(
  ranks_matrix: np.ndarray,
  algorithm_names: list[str],
  alpha: float = 0.05,
  higher_is_better: bool = False,
) -> CDResults:
  """Compute Demšar ranks, Friedman omnibus test, Nemenyi CD, and maximal cliques.

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

  # 1. Compute instance-level ranks (1 = best)
  multiplier = -1.0 if higher_is_better else 1.0
  ranked = np.empty_like(mat)
  for i in range(n_instances):
    ranked[i] = stats.rankdata(multiplier * mat[i])

  # 2. Average ranks across instances
  avg_ranks = np.mean(ranked, axis=0)
  ranks_dict = {name: float(avg_ranks[idx]) for idx, name in enumerate(algorithm_names)}
  rank_sum = float(np.sum(avg_ranks))

  # Rank conservation law: sum_{j=1}^k R_j = k(k + 1) / 2
  expected_sum = (k_algorithms * (k_algorithms + 1)) / 2.0
  if not np.isclose(rank_sum, expected_sum, atol=1e-5):
    raise ValueError(
      f"Rank conservation law violated: sum of average ranks={rank_sum:.6f}, expected={expected_sum:.6f}."
    )

  # 3. Friedman omnibus test
  try:
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=RuntimeWarning)
      chi2, p_val = stats.friedmanchisquare(*[ranked[:, j] for j in range(k_algorithms)])
    if np.isnan(chi2) or np.isnan(p_val):
      chi2, p_val = 0.0, 1.0
  except Exception as e:
    logger.warning("Friedman omnibus test computation failed: %s. Using default p=1.0.", e)
    chi2, p_val = 0.0, 1.0

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
    friedman_chi2=float(chi2),
    friedman_pvalue=float(p_val),
    cliques=maximal_cliques,
    alpha=alpha,
  )


def export_cd_diagram_tikz(
  cd_results: CDResults,
  output_path: Path | str | None = None,
  standalone: bool = False,
) -> str:
  """Export publication-grade TikZ vector code from CDResults.

  Args:
      cd_results: CDResults instance from compute_cd_diagram_data.
      output_path: Optional path to write the generated .tikz file.
      standalone: If True, wraps in a complete compilable standalone document.

  Returns:
      TikZ code as a string.
  """
  k = cd_results.k_algorithms
  cd = cd_results.critical_difference
  sorted_items = sorted(cd_results.ranks.items(), key=lambda x: x[1])

  tikz_body: list[str] = [
    "% Auto-generated Demšar Critical Difference Diagram in TikZ",
    (
      f"% Generated for k = {k} algorithms across N = {cd_results.n_instances} instances "
      f"(CD = {cd:.3f}, alpha = {cd_results.alpha})"
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
      f"    \\draw[cdbar] (1.0, 1.1) -- "
      f"node[above=2pt, font=\\footnotesize\\bfseries] "
      f"{{CD = {cd:.3f} ($\\alpha = {cd_results.alpha:.2f}$)}} ({1.0 + cd:.3f}, 1.1);"
    ),
    "",
    "    % 2. Main Horizontal Rank Axis (1.0 to k.0)",
    f"    \\draw[axis] (0.8, 0.0) -- ({k + 0.3:.1f}, 0.0) node[right, font=\\footnotesize\\bfseries] {{Rank}};",
    "",
    "    % Major Ticks and Numbers",
  ]

  for tick in range(1, k + 1):
    tikz_body.append(
      f"    \\draw[tick] ({tick}.0, 0.08) -- ({tick}.0, -0.08) node[below=2pt, font=\\footnotesize] {{{tick}.0}};"
    )

  tikz_body.append("")
  tikz_body.append("    % Minor Sub-ticks (every 0.5)")
  for tick in range(1, k):
    tikz_body.append(f"    \\draw[subtick] ({tick}.5, 0.04) -- ({tick}.5, -0.04);")

  tikz_body.append("")
  tikz_body.append("    % 3. Algorithm Data Points and Drop Lines")
  for name, r in sorted_items:
    safe_name = escape_latex_identifier(name)
    coord_name = re.sub(r"[^a-zA-Z0-9]", "", name)
    tikz_body.extend(
      [
        f"    \\coordinate (A_{coord_name}) at ({r:.3f}, 0.0);",
        f"    \\node[algomarker] at (A_{coord_name}) {{}};",
        f"    \\draw[thick] (A_{coord_name}) -- ({r:.3f}, -0.45);",
        f"    \\node[labelnode, anchor=north] at ({r:.3f}, -0.45) {{{safe_name}\\\\({r:.2f})}};",
      ]
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
