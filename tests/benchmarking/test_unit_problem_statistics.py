"""Unit tests for single-problem statistical analysis (`benchmarking.analysis.problem_statistics`)
and console logging (`benchmarking.reporting.console.log_problem_analysis`).

WHAT: Covers `analyze_problem`, ragged repetition truncation and strict validation,
Holm-Bonferroni `analyzer.alpha` propagation (Bug 3), correlation-scale rank-biserial
effect interpretation (Bug 2), summary-only dict safety without `raw_costs` (Bug 4),
exact Wilcoxon on constant non-zero diffs (Bug 1), pure side-effect-free computation,
and conditional GPU transfer columns in `log_problem_analysis`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.analysis.problem_statistics import ProblemAnalysis, analyze_problem
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.analysis.stats_utils import holm_bonferroni_correction
from benchmarking.reporting.console import log_problem_analysis
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.runner.config_loader import ProblemConfig

if TYPE_CHECKING:
  from pathlib import Path


def _mock_result(costs: list[float], times: list[float] | None = None) -> dict[str, Any]:
  """Construct a minimal algorithm result dictionary for problem_statistics."""
  if times is None:
    times = [1.0] * len(costs)
  return {
    "raw_costs": costs,
    "raw_times": times,
    "mean_cost": float(np.mean(costs)),
    "mean_time": float(np.mean(times)),
    "mean_gap": 0.5,
  }


def test_analyze_problem_ragged_arrays_truncates_and_warns(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Algorithms with mismatched sample counts [5, 10, 10, 10] warn and truncate."""
  rng = np.random.default_rng(42)
  analyzer = StatisticalAnalyzer(alpha=0.05)

  results: dict[str, dict[str, Any]] = {
    "Alg_CPU": _mock_result(list(rng.normal(100.0, 5.0, size=5))),
    "Alg_GPU_Naive": _mock_result(list(rng.normal(95.0, 4.0, size=10))),
    "Alg_GPU_Opt": _mock_result(list(rng.normal(90.0, 3.0, size=10))),
    "Alg_GPU_Mono": _mock_result(list(rng.normal(88.0, 3.0, size=10))),
  }

  with caplog.at_level(logging.WARNING):
    analysis = analyze_problem(
      problem_name="berlin52",
      optimal_cost=7542.0,
      results=results,
      analyzer=analyzer,
    )

  warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
  assert any(
    "Mismatched sample counts across algorithms for problem 'berlin52'" in r.getMessage() for r in warning_records
  )

  assert isinstance(analysis, ProblemAnalysis)
  assert analysis.cost_omnibus is not None
  stats = analysis.to_dict()
  assert "friedman_test" in stats
  assert "statistic_f" in stats["friedman_test"]
  assert "p_value_f" in stats["friedman_test"]
  assert stats["problem_name"] == "berlin52"


def test_analyze_problem_min_len_less_than_two_skips_omnibus_and_pairwise() -> None:
  """WHAT: Common sample length < 2 skips omnibus and pairwise inference cleanly."""
  analyzer = StatisticalAnalyzer(alpha=0.05)
  results: dict[str, dict[str, Any]] = {
    "Alg1": _mock_result([100.0]),
    "Alg2": _mock_result([95.0, 94.0, 93.0, 92.0, 91.0]),
    "Alg3": _mock_result([90.0, 89.0, 88.0, 87.0, 86.0]),
  }

  analysis = analyze_problem("kroA100", 21282.0, results, analyzer)
  assert analysis.cost_omnibus is None
  assert analysis.cost_pairwise == ()
  stats = analysis.to_dict()
  assert "friedman_test" not in stats
  assert stats["pairwise_comparisons"] == []


def test_analyze_problem_summary_only_dicts_do_not_crash() -> None:
  """WHAT: Regression test for Bug 4 — summary-only dicts without `raw_costs`/`raw_times` succeed cleanly."""
  analyzer = StatisticalAnalyzer(alpha=0.05)
  results = {
    "AlgA": {"mean_cost": 102.0, "mean_time": 4.0, "mean_gap": 2.0},
    "AlgB": {"mean_cost": 100.5, "mean_time": 1.0, "mean_gap": 0.5},
  }

  analysis = analyze_problem("summary_only", 100.0, results, analyzer)
  assert analysis.insufficient_data is False
  assert analysis.cost_omnibus is None
  assert analysis.cost_pairwise == ()
  assert analysis.time_pairwise == ()
  assert len(analysis.rows) == 2
  assert analysis.rows[1].speedup == pytest.approx(4.0)


def test_analyze_problem_propagates_custom_alpha_to_holm_bonferroni() -> None:
  """WHAT: Regression test for Bug 3 — Holm-Bonferroni uses `analyzer.alpha`, never hardcoded 0.05."""
  # For n = 6 with 3 algorithms: 3 pairs, all-positive constant differences give raw p = 2^-5 = 0.03125.
  # Actually use n = 7 -> raw p = 2^-6 = 0.015625.
  # With 3 pairs: step-down adjusted p = 3 * 0.015625 = 0.046875 -> significant at 0.05, NOT significant at 0.01!
  base = np.arange(1.0, 8.0)
  results = {
    "A": {"raw_costs": list(base + 3.0), "raw_times": [3.0] * 7},
    "B": {"raw_costs": list(base + 1.0), "raw_times": [2.0] * 7},
    "C": {"raw_costs": list(base), "raw_times": [1.0] * 7},
  }

  analysis_05 = analyze_problem("alpha_test", 1.0, results, StatisticalAnalyzer(alpha=0.05))
  analysis_01 = analyze_problem("alpha_test", 1.0, results, StatisticalAnalyzer(alpha=0.01))

  assert all(rec.significant_after_correction is True for rec in analysis_05.cost_pairwise)
  assert all(rec.significant_after_correction is False for rec in analysis_01.cost_pairwise)
  assert all(rec.p_value_corrected == pytest.approx(0.046875) for rec in analysis_01.cost_pairwise)


def test_analyze_problem_equal_lengths_runs_without_info_or_warning_spam(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Equal-length analysis is pure computation (zero INFO logs, zero mismatch warnings)."""
  rng = np.random.default_rng(123)
  analyzer = StatisticalAnalyzer(alpha=0.05)

  results: dict[str, dict[str, Any]] = {
    "Alg1": _mock_result(list(rng.normal(100.0, 5.0, size=10))),
    "Alg2": _mock_result(list(rng.normal(95.0, 4.0, size=10))),
    "Alg3": _mock_result(list(rng.normal(90.0, 3.0, size=10))),
  }

  with caplog.at_level(logging.INFO):
    analysis = analyze_problem("eil51", 426.0, results, analyzer)

  assert caplog.records == []
  assert analysis.cost_omnibus is not None


def test_holm_bonferroni_step_down_and_boundaries() -> None:
  """WHAT: Step-down Holm-Bonferroni termination, ordering, and threshold equality."""
  reject, pvals_corrected = holm_bonferroni_correction([0.03, 0.04], alpha=0.05)
  assert reject == [False, False]
  assert pvals_corrected == [0.06, 0.06]

  assert holm_bonferroni_correction([]) == ([], [])

  reject_unsorted, pvals_unsorted = holm_bonferroni_correction([0.04, 0.01], alpha=0.05)
  assert reject_unsorted == [True, True]
  assert pvals_unsorted == [0.04, 0.02]

  reject_eq, pvals_eq = holm_bonferroni_correction([0.025, 0.05], alpha=0.05)
  assert reject_eq == [True, True]
  assert pvals_eq == [0.05, 0.05]


def test_analyze_problem_baseline_resolution_and_report_parity(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Baseline resolution supports explicit override, warns on missing, and matches table reports."""
  analyzer = StatisticalAnalyzer()
  results = {
    "HybridNaive": {
      "raw_costs": [90.0, 92.0],
      "raw_times": [5.0, 6.0],
      "mean_time": 5.5,
      "mean_cost": 91.0,
      "std_cost": 1.0,
      "mean_gap": 1.0,
    },
    "CPU": {
      "raw_costs": [100.0, 102.0],
      "raw_times": [10.0, 12.0],
      "mean_time": 11.0,
      "mean_cost": 101.0,
      "std_cost": 1.0,
      "mean_gap": 2.0,
    },
    "FullGPUModular": {
      "raw_costs": [85.0, 87.0],
      "raw_times": [2.0, 3.0],
      "mean_time": 2.5,
      "mean_cost": 86.0,
      "std_cost": 1.0,
      "mean_gap": 0.5,
    },
  }

  # Explicit override
  assert analyze_problem("eil51", 426.0, results, analyzer, baseline_algorithm="CPU").baseline_algorithm == "CPU"

  # Missing override warns and falls back to first insertion-order algorithm
  with caplog.at_level(logging.WARNING):
    fallback = analyze_problem("eil51", 426.0, results, analyzer, baseline_algorithm="MissingAlg")
  assert fallback.baseline_algorithm == "HybridNaive"
  assert any("MissingAlg" in r.getMessage() for r in caplog.records)

  # Default picks insertion-order first (even when CPU exists later) and matches report tables
  default_analysis = analyze_problem("eil51", 426.0, results, analyzer, baseline_algorithm=None)
  assert default_analysis.baseline_algorithm == "HybridNaive"

  generate_result_tables(
    all_results={"eil51": results},
    problem_configs=[ProblemConfig(name="eil51", optimal=426.0, size=51)],
    output_dir=tmp_path,
    benchmark_name="canonical_baseline_test",
    baseline_algorithm=None,
  )
  md_content = (tmp_path / "canonical_baseline_test.md").read_text(encoding="utf-8")
  assert "| HybridNaive |" in md_content
  assert "| 1.00x |" in md_content
  assert "2.20x" in md_content


def test_analyze_problem_pairwise_diagnostics_and_effect_size_interpretation() -> None:
  """WHAT: Pairwise comparisons expose tie diagnostics and correlation-scale rank-biserial labels."""
  analyzer = StatisticalAnalyzer()
  results = {
    "A": {"raw_costs": [10.0, 10.0, 10.0, 12.0], "raw_times": [1.0, 1.0, 1.0, 1.0], "mean_time": 1.0},
    "B": {"raw_costs": [10.0, 10.0, 10.0, 10.0], "raw_times": [2.0, 2.0, 2.0, 2.0], "mean_time": 2.0},
  }

  stats = analyze_problem("test_prob", 10.0, results, analyzer, baseline_algorithm="A").to_dict()
  pairwise = stats["pairwise_comparisons"]
  assert len(pairwise) == 1
  comp = pairwise[0]
  assert "cohens_d" not in comp
  assert comp["zero_differences_count"] == 3
  assert comp["effective_sample_size"] == 1
  assert comp["insufficient_power_warning"] is True
  assert comp["effect_interpretation"] == "large"
  # At n=4 (< 6 for 95% coverage), CI bounds serialize cleanly as None rather than invalid NaN JSON
  assert comp["ci_95_lower"] is None
  assert comp["ci_95_upper"] is None


def test_analyze_problem_time_inference_and_console_gpu_column_decoupling(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Time inference runs cleanly and `log_problem_analysis` hides GPU columns on CPU-only cohorts."""
  analyzer = StatisticalAnalyzer()
  results = {
    "BaseAlg": {
      "raw_costs": [100.0, 102.0, 101.0, 100.0, 101.0, 102.0],
      "raw_times": [10.0, 10.5, 10.2, 10.1, 10.4, 10.3],
    },
    "MidAlg": {
      "raw_costs": [100.0, 102.0, 101.0, 100.0, 101.0, 102.0],
      "raw_times": [5.0, 5.2, 5.1, 5.05, 5.15, 5.25],
    },
    "FastAlg": {
      "raw_costs": [100.0, 102.0, 101.0, 100.0, 101.0, 102.0],
      "raw_times": [1.0, 1.05, 1.02, 1.01, 1.04, 1.03],
    },
  }

  analysis = analyze_problem("berlin52", 100.0, results, analyzer, baseline_algorithm="BaseAlg")
  stats = analysis.to_dict()

  assert "friedman_test_time" in stats
  assert stats["friedman_test_time"]["significant"] is True
  assert "nemenyi_posthoc_time" in stats
  assert len(stats["pairwise_comparisons_time"]) == 3
  for comp in stats["pairwise_comparisons_time"]:
    assert comp["metric_name"] == "time"
    assert comp["test_used"] == "wilcoxon_signed_rank"
    assert comp["ci_95_lower"] <= comp["ci_95_upper"]
    assert comp["effect_size"] == pytest.approx(1.0)
    assert comp["effect_interpretation"] == "large"

  # Pure CPU cohort -> H2D/D2H columns omitted in console output
  with caplog.at_level(logging.INFO):
    log_problem_analysis(analysis)
  log_text = "\n".join(r.getMessage() for r in caplog.records)
  assert "paired t-test" not in log_text
  assert "Cohen's d" not in log_text
  assert "Wilcoxon" in log_text
  assert "rank-biserial" in log_text
  assert "Hodges-Lehmann" in log_text
  assert "H2D (MB)" not in log_text

  # Cohort with GPU transfer telemetry -> H2D/D2H columns shown
  caplog.clear()
  gpu_results = {k: {**v, "mean_h2d_mb": 1.25} for k, v in results.items()}
  gpu_analysis = analyze_problem("berlin52", 100.0, gpu_results, analyzer, baseline_algorithm="BaseAlg")
  with caplog.at_level(logging.INFO):
    log_problem_analysis(gpu_analysis)
  gpu_log = "\n".join(r.getMessage() for r in caplog.records)
  assert "H2D (MB)" in gpu_log


def test_analyze_problem_time_length_validation_and_strict_mode(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Validate time length mismatch warnings, incomplete raw_times fallback, and strict_lengths=True."""
  analyzer = StatisticalAnalyzer()

  mixed_results = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [], "mean_time": 2.0},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2], "mean_time": 1.1},
  }
  with caplog.at_level(logging.WARNING):
    mixed_stats = analyze_problem("p_mixed", 10.0, mixed_results, analyzer).to_dict()
  assert mixed_stats["pairwise_comparisons_time"] == []
  assert any("Incomplete raw time sample counts" in r.getMessage() for r in caplog.records)

  caplog.clear()
  ragged_times = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [2.0, 2.1, 2.2]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2, 1.3]},
  }
  with caplog.at_level(logging.WARNING):
    ragged_stats = analyze_problem("p_ragged_t", 10.0, ragged_times, analyzer).to_dict()
  assert len(ragged_stats["pairwise_comparisons_time"]) == 1
  assert any("Mismatched time sample counts" in r.getMessage() for r in caplog.records)

  with pytest.raises(ValueError, match="Mismatched time sample counts"):
    analyze_problem("p_ragged_t", 10.0, ragged_times, analyzer, strict_lengths=True)

  ragged_costs = {
    "A": {"raw_costs": [10.0, 11.0], "raw_times": [2.0, 2.1]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2]},
  }
  with pytest.raises(ValueError, match="Mismatched cost sample counts"):
    analyze_problem("p_ragged_c", 10.0, ragged_costs, analyzer, strict_lengths=True)

  missing_all_times = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "mean_time": 2.0},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "mean_time": 1.0},
  }
  with pytest.raises(ValueError, match="Incomplete raw time arrays"):
    analyze_problem("p_no_times", 10.0, missing_all_times, analyzer, strict_lengths=True)

  ragged_no_mean = {
    "A": {"raw_costs": [10.0, 10.0], "raw_times": [1.0, 1.0]},
    "B": {"raw_costs": [10.0, 10.0, 100.0, 100.0], "raw_times": [1.0, 1.0, 10.0, 10.0]},
  }
  ragged_analysis = analyze_problem("p_ragged_nomean", 10.0, ragged_no_mean, analyzer, strict_lengths=False)
  assert ragged_analysis.rows[1].mean_cost == pytest.approx(55.0)
  assert ragged_analysis.rows[1].mean_time == pytest.approx(5.5)

  # Constant non-zero offset at n=3 runs exact Wilcoxon (p=0.25, never p=0.0)
  const_offset = {
    "A": {"raw_costs": [11.0, 12.0, 13.0], "raw_times": [2.0, 2.0, 2.0]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.0, 1.0]},
  }
  const_analysis = analyze_problem("p_const_offset", 10.0, const_offset, analyzer, strict_lengths=True)
  assert const_analysis.cost_pairwise[0].summary.test_used == "wilcoxon_signed_rank"
  assert const_analysis.cost_pairwise[0].summary.p_value == pytest.approx(0.25)
  assert const_analysis.cost_pairwise[0].summary.insufficient_power_warning is True
