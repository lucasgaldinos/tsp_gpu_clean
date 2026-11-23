"""Unit tests for Demšar Critical Difference analysis (`run_friedman_nemenyi_analysis`)
and pure TikZ rendering (`benchmarking.reporting.cd_diagram.export_cd_diagram_tikz`).

WHAT: Tests Demšar rank transformation, Friedman / Iman-Davenport omnibus test, Nemenyi
Critical Difference calculation, maximal non-significant clique detection, rank conservation,
low-power / omnibus_significant flags, and TikZ export with coordinate collision protection.
WHY: DP12-6 / A3 — Single-pass `OmnibusTestResult` guarantees that CD diagrams, LaTeX macros,
and console reports agree identically on ranks, F_F, p-values, CD, and cliques.
EXPECTED: Exact CD threshold matching Demšar (2006), strict rank conservation sum(R_j)=k(k+1)/2,
maximal cliques without redundant sub-intervals, unique TikZ coordinates for colliding names
like `ga_v1` vs `ga-v1`, and strict input validation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats

from benchmarking.analysis.statistics import (
  OmnibusTestResult,
  run_friedman_nemenyi_analysis,
)
from benchmarking.reporting.cd_diagram import export_cd_diagram_tikz

if TYPE_CHECKING:
  from pathlib import Path


def test_cd_calculation_and_rank_conservation_k4_n38() -> None:
  """WHAT: k=4 algorithms across N=38 instances with known ranking."""
  rng = np.random.default_rng(42)
  n_instances = 38
  k_algorithms = 4
  alg_names = ["HybridOpt", "FullGPU", "HybridNaive", "CPU"]

  base_scores = np.tile([100.0, 150.0, 250.0, 400.0], (n_instances, 1))
  results_matrix = base_scores + rng.normal(0, 5.0, size=(n_instances, k_algorithms))

  results = run_friedman_nemenyi_analysis(results_matrix, alg_names, alpha=0.05)

  assert isinstance(results, OmnibusTestResult)
  assert results.k_algorithms == 4
  assert results.n_blocks == 38
  assert results.low_power is False
  assert results.omnibus_significant is True
  assert results.rank_sum == pytest.approx(10.0, abs=1e-5)

  assert results.ranks["HybridOpt"] < results.ranks["FullGPU"]
  assert results.ranks["FullGPU"] < results.ranks["HybridNaive"]
  assert results.ranks["HybridNaive"] < results.ranks["CPU"]

  assert results.critical_difference == pytest.approx(0.7609, abs=1e-3)
  assert results.friedman_pvalue < 1e-3
  assert results.friedman_f > 50.0


def test_omnibus_computes_iman_davenport_f_and_handles_ties() -> None:
  """WHAT: run_friedman_nemenyi_analysis computes exact Iman-Davenport F_F and handles all-tied inputs."""
  alg_names = ["A", "B", "C"]
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
  results = run_friedman_nemenyi_analysis(mat, alg_names, alpha=0.05)
  assert results.chi2_stat == pytest.approx(7.0, rel=1e-12)
  assert results.friedman_f == pytest.approx(7.0, rel=1e-12)
  assert results.df1 == 2
  assert results.df2 == 10
  assert results.friedman_pvalue == pytest.approx(float(stats.f.sf(7.0, 2, 10)), rel=1e-12)
  assert results.omnibus_significant is True
  assert results.low_power is False

  tied_mat = np.full((4, 3), 5.0)
  tied_results = run_friedman_nemenyi_analysis(tied_mat, alg_names, alpha=0.05)
  assert tied_results.friedman_f == 0.0
  assert tied_results.friedman_pvalue == 1.0
  assert tied_results.omnibus_significant is False
  assert tied_results.low_power is True
  assert tied_results.significant_pairs == ()
  assert all(p == pytest.approx(1.0) for row in tied_results.nemenyi_pvalues for p in row)


def test_connected_maximal_clique_detection() -> None:
  """WHAT: Ranks with controlled pairwise distances verify maximal non-significant cliques."""
  alg_names = ["A", "B", "C", "D"]
  ranks = [[1.0, 2.0, 3.0, 4.0]] * 10 + [[2.0, 1.0, 3.0, 4.0]] * 10

  results = run_friedman_nemenyi_analysis(np.array(ranks), alg_names, alpha=0.05)

  assert results.critical_difference == pytest.approx(1.049, abs=0.01)
  assert len(results.cliques) == 2
  assert results.cliques[0][:2] == ("A", "B")
  assert results.cliques[0][2] == pytest.approx(1.5)
  assert results.cliques[0][3] == pytest.approx(1.5)
  assert results.cliques[1][:2] == ("C", "D")
  assert results.cliques[1][2] == pytest.approx(3.0)
  assert results.cliques[1][3] == pytest.approx(4.0)


def test_export_cd_diagram_tikz_contains_required_elements(tmp_path: Path) -> None:
  """WHAT: export_cd_diagram_tikz creates a complete, compilable TikZ snippet."""
  alg_names = ["HybridOpt", "FullGPU", "CPU"]
  scores = np.array(
    [
      [10.0, 15.0, 30.0],
      [12.0, 14.0, 28.0],
      [11.0, 13.0, 29.0],
      [9.0, 16.0, 31.0],
      [10.5, 14.5, 30.5],
    ]
  )

  results = run_friedman_nemenyi_analysis(scores, alg_names, alpha=0.05)
  tikz_path = tmp_path / "cd_diagram.tikz"
  tikz_code = export_cd_diagram_tikz(results, output_path=tikz_path)

  assert tikz_path.exists()
  assert tikz_path.read_text(encoding="utf-8") == tikz_code
  assert r"\begin{tikzpicture}" in tikz_code
  assert r"\end{tikzpicture}" in tikz_code
  assert r"\draw[cdbar]" in tikz_code
  assert r"\draw[axis]" in tikz_code
  assert "CD =" in tikz_code
  assert "HybridOpt" in tikz_code
  assert "FullGPU" in tikz_code
  assert "CPU" in tikz_code

  standalone_code = export_cd_diagram_tikz(results, standalone=True, metric_label="Execution Time")
  assert r"\documentclass[tikz,border=10pt]{standalone}" in standalone_code
  assert r"\usepackage{tikz}" in standalone_code
  assert r"\usetikzlibrary{arrows.meta,positioning,calc}" in standalone_code
  assert "% Metric: Execution Time" in standalone_code


def test_export_cd_diagram_tikz_separates_tied_labels_and_deduplicates_colliding_coords() -> None:
  """WHAT: Tied ranks get unique label positions and sanitized name collisions get distinct TikZ coords.

  WHY: Algorithm names such as `ga_v1` and `ga-v1` both strip to `gav1`; without deduplication
  TikZ redefines `(A_gav1)` and draws both connectors from the same point.
  """
  tied_omnibus = OmnibusTestResult(
    k_algorithms=4,
    n_blocks=15,
    algorithm_names=("ga_v1", "ga-v1", "ga.v1", "baseline"),
    chi2_stat=2.5,
    chi2_pvalue=0.47,
    friedman_f=0.890,
    df1=3,
    df2=42,
    friedman_pvalue=0.4543,
    alpha=0.05,
    omnibus_significant=False,
    low_power=False,
    ranks={"baseline": 2.933, "ga_v1": 2.300, "ga-v1": 2.300, "ga.v1": 2.467},
    rank_sum=10.0,
    critical_difference=1.211,
    cliques=(("ga-v1", "baseline", 2.300, 2.933),),
    nemenyi_pvalues=tuple(tuple(1.0 for _ in range(4)) for _ in range(4)),
    significant_pairs=(),
  )
  tikz_code = export_cd_diagram_tikz(tied_omnibus, metric_label="Optimality Gap")

  label_positions = re.findall(r"\\node\[labelnode[^\]]*\]\s+at\s+\(([^)]+)\)", tikz_code)
  assert len(label_positions) == 4
  assert len(set(label_positions)) == 4

  coord_names = re.findall(r"\\coordinate\s+\((A_[^)]+)\)", tikz_code)
  assert len(coord_names) == 4
  assert len(set(coord_names)) == 4


def test_run_friedman_nemenyi_analysis_validation_errors() -> None:
  """WHAT: Invalid inputs (1D, k < 3, N < 2, mismatched names, non-finite) raise ValueError."""
  with pytest.raises(ValueError, match="must be 2-dimensional"):
    run_friedman_nemenyi_analysis(np.array([1.0, 2.0, 3.0]), ["A", "B", "C"])

  with pytest.raises(ValueError, match="at least 3 algorithms"):
    run_friedman_nemenyi_analysis(np.array([[1.0, 2.0], [3.0, 4.0]]), ["A", "B"])

  with pytest.raises(ValueError, match="at least 2 paired blocks"):
    run_friedman_nemenyi_analysis(np.array([[1.0, 2.0, 3.0]]), ["A", "B", "C"])

  with pytest.raises(ValueError, match="algorithm_names length"):
    run_friedman_nemenyi_analysis(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ["A", "B"])

  with pytest.raises(ValueError, match="non-finite"):
    run_friedman_nemenyi_analysis(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]), ["A", "B", "C"])
