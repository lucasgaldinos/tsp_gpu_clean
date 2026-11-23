"""Behavioral and resilience unit tests for report generator and statistical analysis.

WHAT: Tests anomalous, boundary, and non-standard conditions across the benchmarking pipeline:
1. Case 1: Arbitrary & exotic algorithm names (special tokens, hyphens, underscores).
2. Case 2: Baseline disabled on large instances (local dagger fallback \\dagger).
3. Case 3: Zero-variance / absolute ranking ties in Friedman omnibus test.
4. Case 4: Single-repetition cohorts (N=1, std=0.0, zero Bessel degrees-of-freedom warnings).
5. Case 5: Ragged and disjoint problem subsets across algorithms (|P_cap| = 1, |P_cap| = 0).

WHY: Task 12.24 (DP12-17) — Protect dissertation benchmarking pipeline against unhandled
runtime warnings, division by zero, or crashes under unusual empirical data distributions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.analysis.problem_statistics import (
  _resolve_gap_mean,
  _resolve_metric_mean,
  perform_statistical_analysis,
)
from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.analysis.report_style import escape_latex_identifier
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.analysis.stats_utils import resolve_baseline_algorithm
from benchmarking.analysis.table_reports import _compute_table2_stats
from benchmarking.runner.algorithm_runner import compute_series_stats
from benchmarking.runner.config_loader import ProblemConfig

if TYPE_CHECKING:
  from pathlib import Path


def test_behavioral_case1_arbitrary_exotic_algorithm_names(tmp_path: Path) -> None:
  """WHAT: Benchmarking with arbitrary and exotic algorithm identifiers.

  WHY: Task 12.24 (Case 1) — Algorithm names may contain hyphens, underscores, or
  camelCase strings (e.g. 'CrazyCpuGpuLocalSearchEveryStep', 'GA-Custom_v2').
  EXPECTED: Table generation produces valid Markdown and LaTeX without crashes;
  dynamic baseline selection picks the first exotic name when baseline is omitted.
  """
  exotic_algs = [
    "2-Opt_GPU",
    "GA-Custom_v2",
    "CrazyCpuGpuLocalSearchEveryStep",
  ]
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob_exotic_1": {
      "2-Opt_GPU": {"mean_time": 0.5, "mean_cost": 102.0, "std_cost": 0.0, "mean_gap": 2.0},
      "GA-Custom_v2": {"mean_time": 1.0, "mean_cost": 105.0, "std_cost": 0.0, "mean_gap": 5.0},
      "CrazyCpuGpuLocalSearchEveryStep": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob_exotic_2": {
      "2-Opt_GPU": {"mean_time": 1.0, "mean_cost": 204.0, "std_cost": 0.0, "mean_gap": 2.0},
      "GA-Custom_v2": {"mean_time": 2.0, "mean_cost": 210.0, "std_cost": 0.0, "mean_gap": 5.0},
      "CrazyCpuGpuLocalSearchEveryStep": {"mean_time": 4.0, "mean_cost": 200.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="prob_exotic_1", optimal=100.0, size=50),
    ProblemConfig(name="prob_exotic_2", optimal=200.0, size=100),
  ]

  # 1. Verify dynamic baseline selection with exotic names (insertion order in table generator)
  resolved_baseline = resolve_baseline_algorithm(exotic_algs)
  assert resolved_baseline == "2-Opt_GPU"

  # 2. Generate result tables without explicit baseline override (picks 2-Opt_GPU dynamically)
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="exotic_names_test",
    baseline_algorithm=None,
  )

  md_content = (tmp_path / "exotic_names_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "exotic_names_test.tex").read_text(encoding="utf-8")

  # Verify all algorithm names are present in both outputs (unescaped in MD, escaped in TeX)
  for name in exotic_algs:
    assert name in md_content
    assert escape_latex_identifier(name) in tex_content

  # Dynamic baseline (2-Opt_GPU) must have 1.00x speedup
  assert "1.00x" in md_content
  assert "1.00$\\times$" in tex_content

  # GA-Custom_v2 (time ratio 0.5/1.0 = 0.50x, 1.0/2.0 = 0.50x -> 0.50x (n=2))
  assert "0.50x (n=2)" in md_content
  assert "0.50$\\times$ ($n=2$)" in tex_content

  # 3. Explicit baseline override works with long exotic name
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="exotic_explicit_test",
    baseline_algorithm="CrazyCpuGpuLocalSearchEveryStep",
  )
  md_explicit = (tmp_path / "exotic_explicit_test.md").read_text(encoding="utf-8")
  tex_explicit = (tmp_path / "exotic_explicit_test.tex").read_text(encoding="utf-8")
  assert "2.00x (n=2)" in md_explicit
  assert "2.00$\\times$ ($n=2$)" in tex_explicit


def test_behavioral_case2_baseline_disabled_local_dagger_fallback(tmp_path: Path) -> None:
  """WHAT: Baseline algorithm omitted on large instances (e.g. CPU skipped for n > 400).

  WHY: Task 12.24 (Case 2) — When the configured baseline ('CPU') was skipped on large problems,
  the user-commissioned local fallback with dagger ('1.00x†' / '1.00\\times^{\\dagger}')
  must activate on rows missing CPU, while normal rows retain the primary baseline.
  EXPECTED: Local dagger fallback rendered on skipped instances with footnote explanation.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "small_50": {
      "CPU": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "GPU_Fast": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "huge_1000": {
      # CPU skipped due to threshold! Local fallback is GPU_Fast
      "GPU_Fast": {"mean_time": 50.0, "mean_cost": 2000.0, "std_cost": 0.0, "mean_gap": 0.0},
      "GPU_Ultra": {"mean_time": 10.0, "mean_cost": 1950.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="small_50", optimal=100.0, size=50),
    ProblemConfig(name="huge_1000", optimal=2000.0, size=1000),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="dagger_fallback_test",
    baseline_algorithm="CPU",
  )

  md_content = (tmp_path / "dagger_fallback_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "dagger_fallback_test.tex").read_text(encoding="utf-8")

  # On small_50, CPU is present: rows omit dagger in both Markdown and LaTeX
  small_md_rows = [line for line in md_content.splitlines() if line.startswith("| small_50 |")]
  assert len(small_md_rows) == 2
  for row in small_md_rows:
    assert "†" not in row
  assert small_md_rows[0].endswith("| 1.00x |")
  assert small_md_rows[1].endswith("| 5.00x |")

  # In LaTeX Table 1: small_50 rows do not contain \dagger
  tex_lines = tex_content.splitlines()
  cpu_small_tex = next(line for line in tex_lines if "small\\_50" in line and "CPU" in line)
  assert "1.00$\\times$" in cpu_small_tex and "\\dagger" not in cpu_small_tex
  gpu_small_tex = next(line for line in tex_lines if "GPU\\_Fast" in line and "5.00" in line)
  assert "5.00$\\times$" in gpu_small_tex and "\\dagger" not in gpu_small_tex

  # On huge_1000, CPU is missing: GPU_Fast is local fallback 1.00x†, GPU_Ultra is 5.00x†
  huge_md_rows = [line for line in md_content.splitlines() if line.startswith("| huge_1000 |")]
  assert len(huge_md_rows) == 2
  assert huge_md_rows[0].endswith("| 1.00x† |")
  assert huge_md_rows[1].endswith("| 5.00x† |")

  gpu_huge_tex = next(line for line in tex_lines if "huge\\_1000" in line and "GPU\\_Fast" in line)
  assert "1.00$\\times^{\\dagger}$" in gpu_huge_tex
  ultra_huge_tex = next(line for line in tex_lines if "GPU\\_Ultra" in line and "5.00" in line)
  assert "5.00$\\times^{\\dagger}$" in ultra_huge_tex

  # Note explaining dagger must appear
  assert "† Indicates speedup calculated relative to local fallback baseline" in md_content
  assert "$\\dagger$ Indicates speedup calculated relative to local fallback baseline" in tex_content


def test_behavioral_case3_friedman_absolute_ranking_ties_zero_variance() -> None:
  """WHAT: Absolute ranking ties across all algorithms on every problem in Friedman test.

  WHY: Task 12.24 (Case 3) — When multiple algorithms find identical solutions on every
  problem instance, ranking ties lead to 0/0 division in standard chi-square formulas.
  EXPECTED: Zero unhandled RuntimeWarning, statistic=0.0, p_value=1.0, significant=False;
  scale-invariant detection does not collapse small-scale distinct data; non-finite rejected.
  """
  analyzer = StatisticalAnalyzer()

  # 1. 3 algorithms with identical scores across 4 problems (complete ranking tie across all blocks)
  scores_a = np.array([100.0, 200.0, 300.0, 400.0])
  scores_b = np.array([100.0, 200.0, 300.0, 400.0])
  scores_c = np.array([100.0, 200.0, 300.0, 400.0])

  # Turn all warnings into errors to verify zero unhandled RuntimeWarning
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    friedman_res = analyzer.friedman_test([scores_a, scores_b, scores_c])

  assert friedman_res["statistic"] == 0.0
  assert friedman_res["p_value"] == 1.0
  assert friedman_res["significant"] is False
  assert friedman_res["post_hoc_required"] is False

  # 2. Scale invariance: distinct arrays at small scale (1e-11) must NOT be collapsed to p=1.0
  d1 = np.array([1e-11, 1e-11, 1e-11])
  d2 = np.array([2e-11, 2e-11, 2e-11])
  d3 = np.array([3e-11, 3e-11, 3e-11])
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    res_small = analyzer.friedman_test([d1, d2, d3])
  assert res_small["statistic"] == pytest.approx(6.0)
  assert res_small["p_value"] < 0.05
  assert res_small["significant"] is True

  # 3. Non-finite values (NaN / Inf) must raise ValueError rather than being swallowed
  with pytest.raises(ValueError, match="non-finite"):
    analyzer.friedman_test([np.array([np.nan, 1.0]), np.array([1.0, 2.0]), np.array([2.0, 3.0])])

  with pytest.raises(ValueError, match="non-finite"):
    analyzer.friedman_test([np.array([np.inf, 1.0]), np.array([1.0, 2.0]), np.array([2.0, 3.0])])

  # 4. Verify perform_statistical_analysis with real runner keys (raw_costs, raw_times)
  single_prob_results = {
    "Alg1": {"raw_costs": [100.0, 100.0, 100.0], "raw_times": [1.0, 1.0, 1.0]},
    "Alg2": {"raw_costs": [100.0, 100.0, 100.0], "raw_times": [1.0, 1.0, 1.0]},
    "Alg3": {"raw_costs": [100.0, 100.0, 100.0], "raw_times": [1.0, 1.0, 1.0]},
  }
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    stat_summary = perform_statistical_analysis(
      problem_name="tied_problem",
      optimal_cost=100.0,
      results=single_prob_results,
      analyzer=analyzer,
    )

  assert "friedman_test" in stat_summary
  assert stat_summary["friedman_test"]["statistic"] == 0.0
  assert stat_summary["friedman_test"]["p_value"] == 1.0
  assert stat_summary["friedman_test"]["significant"] is False


def test_behavioral_case4_single_repetition_cohorts(tmp_path: Path) -> None:
  """WHAT: Single-repetition benchmark runs (N = 1).

  WHY: Task 12.24 (Case 4) — For single-run experiments, sample standard deviation with
  Bessel's correction (ddof=1) must return 0.0 instead of NaN or 'Degrees of freedom <= 0' warning.
  EXPECTED: std == 0.0 everywhere, valid tables without NaN entries.
  """
  # 1. Test compute_series_stats directly
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    mean_val, std_val, min_val, max_val = compute_series_stats(np.array([42.5]), total_successful=1)

  assert mean_val == 42.5
  assert std_val == 0.0
  assert min_val == 42.5
  assert max_val == 42.5

  # 2. Test perform_statistical_analysis and paired_comparison with N=1
  analyzer = StatisticalAnalyzer()
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    comp = analyzer.paired_comparison(
      np.array([100.0]),
      np.array([90.0]),
      label="Algo1 vs Algo2",
      metric_name="cost",
    )

  assert comp.sample_size == 1
  assert comp.std_a == 0.0
  assert comp.std_b == 0.0
  assert comp.test_used == "no_test_needed"

  single_rep_results = {
    "Algo1": {"raw_costs": [100.0], "raw_times": [2.0]},
    "Algo2": {"raw_costs": [90.0], "raw_times": [1.0]},
  }
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    analysis = perform_statistical_analysis(
      problem_name="single_rep_prob",
      optimal_cost=80.0,
      results=single_rep_results,
      analyzer=analyzer,
    )

  pairwise_comp = analysis["pairwise_comparisons"][0]
  assert pairwise_comp["test_used"] == "no_test_needed"
  assert pairwise_comp["p_value"] == 1.0
  assert pairwise_comp["mean_difference"] == pytest.approx(10.0)

  # 3. Test generate_result_tables with N=1 summary data
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "Algo1": {"mean_time": 2.0, "std_time": 0.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "Algo2": {"mean_time": 1.0, "std_time": 0.0, "mean_cost": 90.0, "std_cost": 0.0, "mean_gap": 0.0},
    }
  }
  problem_configs = [ProblemConfig(name="p1", optimal=80.0, size=20)]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="n1_test",
    baseline_algorithm="Algo1",
  )

  md_content = (tmp_path / "n1_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "n1_test.tex").read_text(encoding="utf-8")

  assert "NaN" not in md_content
  assert "NaN" not in tex_content
  assert "2.00x (n=1)" in md_content
  assert "2.00$\\times$ ($n=1$)" in tex_content

  # Table 2: std_gap == 0.00 and std_time == 0.00 in both Markdown and LaTeX
  assert "| Algo1 | 0.00 | 0.00 | 2.00 | 0.00 | 1.00x |" in md_content
  assert "| Algo2 | 0.00 | 0.00 | 1.00 | 0.00 | 2.00x (n=1) |" in md_content
  assert "Algo1           &     0.00 &    0.00 &     2.00 &     0.00 & 1.00$\\times$ \\\\" in tex_content
  assert "Algo2           &     0.00 &    0.00 &     1.00 &     0.00 & 2.00$\\times$ ($n=1$) \\\\" in tex_content


def test_behavioral_case5_ragged_and_disjoint_problem_subsets(tmp_path: Path) -> None:
  """WHAT: Ragged and disjoint problem subsets across algorithms.

  WHY: Task 12.24 (Case 5) — Heterogeneous experimental setups where different algorithms
  were evaluated on disjoint problem subsets must not cause KeyError, IndexError, or NaN speedups.
  EXPECTED: Overlapping pairs calculate paired geometric mean with exact sample size annotation;
  disjoint pairs report '—' speedup gracefully without crashing.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob_A": {
      "Base": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "Overlapping": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob_B": {
      # Only Base ran here
      "Base": {"mean_time": 20.0, "mean_cost": 200.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob_C": {
      # Base did NOT run here; only Disjoint ran
      "Disjoint": {"mean_time": 5.0, "mean_cost": 300.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="prob_A", optimal=100.0, size=50),
    ProblemConfig(name="prob_B", optimal=200.0, size=100),
    ProblemConfig(name="prob_C", optimal=300.0, size=150),
  ]

  # 1. Verify _compute_table2_stats directly
  table2_stats = _compute_table2_stats(
    all_results=all_results,
    problem_names=["prob_A", "prob_B", "prob_C"],
    algorithm_names=["Base", "Overlapping", "Disjoint"],
    baseline_algorithm="Base",
  )
  assert len(table2_stats) == 3

  # Base
  assert table2_stats[0]["algorithm"] == "Base"
  assert table2_stats[0]["is_baseline"] is True
  assert table2_stats[0]["avg_speedup"] == 1.0

  # Overlapping (overlap |P_cap| = 1 on prob_A: speedup 10/2 = 5.00x)
  assert table2_stats[1]["algorithm"] == "Overlapping"
  assert table2_stats[1]["sample_size"] == 1
  assert table2_stats[1]["avg_speedup"] == pytest.approx(5.0)

  # Disjoint (overlap |P_cap| = 0 with Base: avg_speedup is None)
  assert table2_stats[2]["algorithm"] == "Disjoint"
  assert table2_stats[2]["sample_size"] == 0
  assert table2_stats[2]["avg_speedup"] is None

  # 2. Verify table generation renders '—' for Disjoint
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="ragged_test",
    baseline_algorithm="Base",
  )

  md_content = (tmp_path / "ragged_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "ragged_test.tex").read_text(encoding="utf-8")

  # Table 2: Overlapping has 5.00x (n=1), Disjoint has '—' in both Markdown and LaTeX
  assert "5.00x (n=1)" in md_content
  assert "5.00$\\times$ ($n=1$)" in tex_content

  # Explicitly verify '—' in the Disjoint speedup cell for Table 2
  disjoint_md_rows = [line for line in md_content.splitlines() if line.startswith("| Disjoint |")]
  assert len(disjoint_md_rows) == 1
  assert disjoint_md_rows[0].endswith("| — |")

  disjoint_tex_rows = [line for line in tex_content.splitlines() if line.startswith("Disjoint") and "&" in line]
  assert len(disjoint_tex_rows) == 1
  assert disjoint_tex_rows[0].endswith("— \\\\")


def test_behavioral_lazy_metric_resolution_and_validation() -> None:
  """WHAT: Unit test lazy metric resolution in problem_statistics.py (BLOCKING #3).

  WHY: Task 12.24 (DP12-17) — Prevent eager evaluation of np.nanmean on empty/NaN slices
  and prevent silent fabrication of 0.0 / 1.0 on malformed input dictionaries.
  EXPECTED: No RuntimeWarning when mean is supplied alongside NaN arrays; KeyError raised
  when both metric and raw data are missing.
  """
  # 1. No RuntimeWarning when precomputed mean is supplied alongside all-NaN raw series
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    val = _resolve_metric_mean(
      results_dict={"mean_cost": 150.0},
      metric_key="mean_cost",
      raw_series=[np.nan, np.nan],
      alg_name="TestAlg",
    )
  assert val == 150.0

  # 2. Computes mean from raw_series when precomputed key is missing
  val_raw = _resolve_metric_mean(
    results_dict={},
    metric_key="mean_cost",
    raw_series=[100.0, 200.0],
    alg_name="TestAlg",
  )
  assert val_raw == 150.0

  # 3. Fails loudly with KeyError when both precomputed and raw data are missing
  with pytest.raises(KeyError, match="TestAlg"):
    _resolve_metric_mean(
      results_dict={},
      metric_key="mean_cost",
      raw_series=[],
      alg_name="TestAlg",
    )

  # 4. Fails loudly with KeyError when raw series has only NaNs and no precomputed metric
  with pytest.raises(KeyError, match="TestAlg"):
    _resolve_metric_mean(
      results_dict={},
      metric_key="mean_cost",
      raw_series=[np.nan],
      alg_name="TestAlg",
    )

  # 5. _resolve_gap_mean uses optimal_cost if mean_gap is absent
  gap = _resolve_gap_mean(
    results_dict={},
    mean_cost=110.0,
    optimal_cost=100.0,
    raw_gaps=[],
    alg_name="TestAlg",
  )
  assert gap == pytest.approx(10.0)

  # 6. _resolve_gap_mean raises KeyError if both mean_gap and valid sources are missing
  with pytest.raises(KeyError, match="TestAlg"):
    _resolve_gap_mean(
      results_dict={},
      mean_cost=110.0,
      optimal_cost=0.0,
      raw_gaps=[],
      alg_name="TestAlg",
    )


def test_behavioral_latex_escaping_across_all_four_tables(tmp_path: Path) -> None:
  """WHAT: Verification of LaTeX identifier escaping across Tables 1 through 4 (Task 12.29, BUG-45).

  WHY: Underscores ('_') and percent signs ('%') in problem/algorithm names cause LaTeX syntax
  errors (math subscript without $ and premature comment truncation) if unescaped.
  EXPECTED: All names with '_' and '%' are escaped in .tex output (\\_, \\%) across Tables 1-4,
  while remaining completely unescaped in .md output.
  """
  # Helper function unit validation
  assert escape_latex_identifier("ga_v2%_gpu") == "ga\\_v2\\%\\_gpu"
  assert escape_latex_identifier("plain") == "plain"
  assert escape_latex_identifier("") == ""

  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob_alpha_1": {
      "Base_Alg": {
        "mean_time": 10.0,
        "std_time": 0.5,
        "mean_cost": 500.0,
        "std_cost": 10.0,
        "mean_gap": 0.0,
        "mean_phase_timings": {},
      },
      "Fast_100%_GPU": {
        "mean_time": 2.0,
        "std_time": 0.1,
        "mean_cost": 500.0,
        "std_cost": 10.0,
        "mean_gap": 0.0,
        "mean_phase_timings": {"initialization": 0.5, "crossover": 1.5},
        "phase_share_pct": {"initialization": 25.0, "crossover": 75.0},
        "phase_share_cv_pct": {"initialization": 5.0, "crossover": 10.0},
      },
    },
    "prob_beta%_2": {
      "Base_Alg": {
        "mean_time": 20.0,
        "std_time": 1.0,
        "mean_cost": 1000.0,
        "std_cost": 20.0,
        "mean_gap": 0.0,
        "mean_phase_timings": {},
      },
      "Fast_100%_GPU": {
        "mean_time": 4.0,
        "std_time": 0.2,
        "mean_cost": 1000.0,
        "std_cost": 20.0,
        "mean_gap": 0.0,
        "mean_phase_timings": {"initialization": 1.0, "crossover": 3.0},
        "phase_share_pct": {"initialization": 25.0, "crossover": 75.0},
        "phase_share_cv_pct": {"initialization": 5.0, "crossover": 10.0},
      },
    },
  }
  problem_configs = [
    ProblemConfig(name="prob_alpha_1", optimal=500.0, size=50),
    ProblemConfig(name="prob_beta%_2", optimal=1000.0, size=150),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="latex_escaping_test",
    baseline_algorithm="Base_Alg",
    include_size_category_table=True,
  )

  md_content = (tmp_path / "latex_escaping_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "latex_escaping_test.tex").read_text(encoding="utf-8")

  # 1. In Markdown, all special characters remain verbatim / unescaped
  assert "prob_alpha_1" in md_content
  assert "prob_beta%_2" in md_content
  assert "Base_Alg" in md_content
  assert "Fast_100%_GPU" in md_content
  assert "\\_" not in md_content
  assert "\\%" not in md_content

  # 2. In LaTeX Table 1 (Summary Statistics by Problem):
  # Caption and note have escaped baseline:
  assert "Base\\_Alg" in tex_content
  # Problem cells and algorithm cells are escaped:
  assert "prob\\_alpha\\_1" in tex_content
  assert "prob\\_beta\\%\\_2" in tex_content
  assert "Fast\\_100\\%\\_GPU" in tex_content

  # 3. In LaTeX Table 2 (Algorithm Comparison):
  # Rows start with escaped algorithm names:
  tex_lines = tex_content.splitlines()
  table2_base = next(line for line in tex_lines if line.startswith("Base\\_Alg"))
  assert "1.00$\\times$" in table2_base
  table2_fast = next(line for line in tex_lines if line.startswith("Fast\\_100\\%\\_GPU"))
  assert "5.00$\\times$ ($n=2$)" in table2_fast

  # 4. In LaTeX Table 3 (Size Categories):
  # Problem list and winner algorithm names are escaped:
  table3_small = next(line for line in tex_lines if line.startswith("Small (n≤100)"))
  assert "prob\\_alpha\\_1" in table3_small
  assert "Base\\_Alg" in table3_small or "Fast\\_100\\%\\_GPU" in table3_small
  table3_medium = next(line for line in tex_lines if line.startswith("Medium"))
  assert "prob\\_beta\\%\\_2" in table3_medium

  # 5. In LaTeX Table 4 (Phase Timings):
  # Problem and algorithm columns in phase timing table are escaped:
  table4_rows = [line for line in tex_lines if "prob\\_alpha\\_1" in line and "Fast\\_100\\%\\_GPU" in line]
  assert len(table4_rows) == 1
  assert "2000.0" in table4_rows[0]

  # 6. Negative checks: Verify unescaped forms DO NOT appear as identifiers in .tex
  assert "prob_alpha_1" not in tex_content
  assert "prob_beta%_2" not in tex_content
  assert "Fast_100%_GPU" not in tex_content
