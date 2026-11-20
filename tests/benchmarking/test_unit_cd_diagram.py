"""Unit tests for Demšar Critical Difference (CD) diagram generation (`src.benchmarking.analysis.cd_diagram`).

WHAT: Tests Demšar rank transformation, Friedman omnibus test, Nemenyi Critical Difference (CD)
calculation with SciPy Studentized range scaling convention, connected maximal clique detection,
rank conservation law validation, and TikZ export.
WHY: DP12-6 — Critical Difference diagrams represent a core statistical thesis artifact for
multi-algorithm metaheuristic comparisons. Unscaled CD or flawed clique detection leads to
severe reporting inaccuracies (e.g. 41.4% inflated thresholds).
EXPECTED: Exact CD threshold matching Demšar (2006) formulation, strict rank conservation
sum(R_j) = k(k+1)/2, maximal cliques without redundant sub-intervals, and syntactically valid TikZ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarking.analysis.cd_diagram import (
  CDResults,
  compute_cd_diagram_data,
  export_cd_diagram_tikz,
)

if TYPE_CHECKING:
  from pathlib import Path


def test_cd_calculation_and_rank_conservation_k4_n38() -> None:
  """WHAT: k=4 algorithms across N=38 instances with known ranking.

  WHY: Validates Demšar (2006) Nemenyi CD formula: CD = (Q_alpha / sqrt(2)) * sqrt(k(k+1)/(6N)).
  At k=4, N=38, alpha=0.05, q_alpha ~= 2.569 and CD ~= 0.761.
  EXPECTED: CD ~= 0.7609, sum of average ranks == 10.00, p-value significant.
  """
  np.random.seed(42)
  n_instances = 38
  k_algorithms = 4
  alg_names = ["HybridOpt", "FullGPU", "HybridNaive", "CPU"]

  # Generate synthetic scores where HybridOpt < FullGPU < HybridNaive < CPU consistently
  base_scores = np.tile([100.0, 150.0, 250.0, 400.0], (n_instances, 1))
  noise = np.random.normal(0, 5.0, size=(n_instances, k_algorithms))
  results_matrix = base_scores + noise

  results = compute_cd_diagram_data(results_matrix, alg_names, alpha=0.05)

  assert isinstance(results, CDResults)
  assert results.k_algorithms == 4
  assert results.n_instances == 38

  # Verify rank conservation law: sum(R_j) = 4 * 5 / 2 = 10.0
  assert results.rank_sum == pytest.approx(10.0, abs=1e-5)

  # Check average ranks order
  assert results.ranks["HybridOpt"] < results.ranks["FullGPU"]
  assert results.ranks["FullGPU"] < results.ranks["HybridNaive"]
  assert results.ranks["HybridNaive"] < results.ranks["CPU"]

  # Critical difference check for k=4, N=38: CD ~= 0.7609
  assert results.critical_difference == pytest.approx(0.7609, abs=1e-3)

  # Friedman Iman-Davenport F_F test must reject null hypothesis (p < 0.001)
  assert results.friedman_pvalue < 1e-3
  assert results.friedman_f > 50.0


def test_cd_diagram_computes_iman_davenport_f_and_handles_ties() -> None:
  """WHAT: compute_cd_diagram_data computes finite Iman-Davenport F_F and handles all-tied inputs.

  WHY: Task 13.9 (BUG-61 / DP13-3) — CDResults exposes only friedman_f (matching Iman & Davenport 1980)
  with no legacy chi-square field, and complete intra-block ties must yield F_F=0.0, p=1.0.
  """
  from dataclasses import fields

  from scipy import stats

  expected_fields = {
    "k_algorithms",
    "n_instances",
    "ranks",
    "rank_sum",
    "critical_difference",
    "friedman_f",
    "friedman_pvalue",
    "cliques",
    "alpha",
  }
  assert {f.name for f in fields(CDResults)} == expected_fields

  alg_names = ["A", "B", "C"]
  # N=6, k=3 -> chi2=7.0, F_F=7.0, df1=2, df2=10
  mat = np.array(
    [
      [1.0, 2.0, 3.0],
      [2.0, 1.0, 3.0],
      [1.0, 3.0, 2.0],
      [2.0, 1.0, 3.0],
      [1.0, 2.0, 3.0],
      [1.0, 2.0, 3.0],
    ]
  )
  results = compute_cd_diagram_data(mat, alg_names, alpha=0.05)
  assert results.friedman_f == pytest.approx(7.0, rel=1e-12)
  assert results.friedman_pvalue == pytest.approx(float(stats.f.sf(7.0, 2, 10)), rel=1e-12)

  tied_mat = np.full((4, 3), 5.0)
  tied_results = compute_cd_diagram_data(tied_mat, alg_names, alpha=0.05)
  assert tied_results.friedman_f == 0.0
  assert tied_results.friedman_pvalue == 1.0


def test_connected_maximal_clique_detection() -> None:
  """WHAT: Ranks with controlled pairwise distances to test clique boundaries.

  WHY: Cliques must group contiguous algorithms with |R_i - R_j| <= CD without
  redundant sub-intervals (e.g. if A, B, C form a clique, AB and BC are not duplicated).
  EXPECTED: Maximal cliques correctly identified.
  """
  alg_names = ["A", "B", "C", "D"]
  # Construct a 20-instance matrix where:
  # 10 rows: [1, 2, 3, 4]
  # 10 rows: [2, 1, 3, 4]
  # Average ranks: A=1.5, B=1.5, C=3.0, D=4.0
  # At N=20, k=4:
  # q_0.05 ~= 2.569 -> CD = 2.569 * sqrt(20 / 120) = 2.569 * 0.408 ~= 1.049
  # |B - A| = 0.0 <= 1.049 -> {A, B} is a maximal clique
  # |C - B| = 1.5 > 1.049 -> separated
  # |D - C| = 1.0 <= 1.049 -> {C, D} is a maximal clique
  ranks = []
  for _ in range(10):
    ranks.append([1.0, 2.0, 3.0, 4.0])
  for _ in range(10):
    ranks.append([2.0, 1.0, 3.0, 4.0])

  results = compute_cd_diagram_data(np.array(ranks), alg_names, alpha=0.05)

  assert results.critical_difference == pytest.approx(1.049, abs=0.01)

  # Exactly 2 maximal cliques: (A, B) and (C, D)
  assert len(results.cliques) == 2
  assert results.cliques[0][:2] == ("A", "B")
  assert results.cliques[0][2] == pytest.approx(1.5)
  assert results.cliques[0][3] == pytest.approx(1.5)
  assert results.cliques[1][:2] == ("C", "D")
  assert results.cliques[1][2] == pytest.approx(3.0)
  assert results.cliques[1][3] == pytest.approx(4.0)


def test_export_cd_diagram_tikz_contains_required_elements(tmp_path: Path) -> None:
  """WHAT: export_cd_diagram_tikz creates a complete, compilable TikZ snippet.

  WHY: TikZ output must include rank axis, ticks, CD bar, algorithm coordinates,
  and clique connectors matching the second_draft.tex thesis styling.
  EXPECTED: File is created on disk, string contains all TikZ directives.
  """
  alg_names = ["HybridOpt", "FullGPU", "CPU"]
  # 5 instances
  scores = np.array(
    [
      [10.0, 15.0, 30.0],
      [12.0, 14.0, 28.0],
      [11.0, 13.0, 29.0],
      [9.0, 16.0, 31.0],
      [10.5, 14.5, 30.5],
    ]
  )

  results = compute_cd_diagram_data(scores, alg_names, alpha=0.05)
  tikz_path = tmp_path / "cd_diagram.tikz"

  tikz_code = export_cd_diagram_tikz(results, output_path=tikz_path)

  assert tikz_path.exists()
  content = tikz_path.read_text(encoding="utf-8")
  assert content == tikz_code

  # Check required TikZ primitives
  assert r"\begin{tikzpicture}" in tikz_code
  assert r"\end{tikzpicture}" in tikz_code
  assert r"\draw[cdbar]" in tikz_code
  assert r"\draw[axis]" in tikz_code
  assert "CD =" in tikz_code
  assert "HybridOpt" in tikz_code
  assert "FullGPU" in tikz_code
  assert "CPU" in tikz_code

  # Standalone document mode
  standalone_code = export_cd_diagram_tikz(results, standalone=True, metric_label="Execution Time")
  assert r"\documentclass[tikz,border=10pt]{standalone}" in standalone_code
  assert r"\usepackage{tikz}" in standalone_code
  assert r"\usetikzlibrary{arrows.meta,positioning,calc}" in standalone_code
  assert "% Metric: Execution Time" in standalone_code


def test_export_cd_diagram_tikz_separates_tied_and_proximate_rank_labels() -> None:
  """WHAT: Tied and proximate average ranks receive distinct label coordinates.

  WHY: Identical variants (e.g. HybridNaive and HybridOptimized) tie in rank; placing
  all labels at (r, -0.45) causes exact text overlap in compiled TikZ figures.
  EXPECTED: Every algorithm labelnode is placed at a unique (x, y) coordinate.
  """
  import re

  tied_cd = CDResults(
    k_algorithms=4,
    n_instances=15,
    ranks={"CPU": 2.933, "HybridNaive": 2.300, "HybridOptimized": 2.300, "FullGPUModular": 2.467},
    rank_sum=10.0,
    critical_difference=1.211,
    friedman_f=0.890,
    friedman_pvalue=0.4543,
    cliques=[("HybridNaive", "CPU", 2.300, 2.933)],
    alpha=0.05,
  )
  tikz_code = export_cd_diagram_tikz(tied_cd, metric_label="Optimality Gap")
  positions = re.findall(r"\\node\[labelnode[^\]]*\]\s+at\s+\(([^)]+)\)", tikz_code)
  assert len(positions) == 4
  assert len(set(positions)) == 4


def test_compute_cd_diagram_data_validation_errors() -> None:
  """WHAT: Invalid inputs (k < 3, N < 2, 1D array, mismatched name count).

  WHY: Preconditions must fail fast with descriptive ValueError.
  EXPECTED: ValueError on all violation dimensions.
  """
  # 1D array
  with pytest.raises(ValueError, match="must be 2-dimensional"):
    compute_cd_diagram_data(np.array([1.0, 2.0, 3.0]), ["A", "B", "C"])

  # k < 3
  with pytest.raises(ValueError, match="at least 3 algorithms"):
    compute_cd_diagram_data(np.array([[1.0, 2.0], [3.0, 4.0]]), ["A", "B"])

  # N < 2
  with pytest.raises(ValueError, match="at least 2 instances"):
    compute_cd_diagram_data(np.array([[1.0, 2.0, 3.0]]), ["A", "B", "C"])

  # Mismatched algorithm names
  with pytest.raises(ValueError, match="algorithm_names length"):
    compute_cd_diagram_data(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ["A", "B"])
