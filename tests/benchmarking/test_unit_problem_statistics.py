"""Unit tests for single-problem statistical analysis and ragged array handling.

WHAT: Covers `perform_statistical_analysis` in `benchmarking.analysis.problem_statistics`,
specifically verifying that heterogeneous repetition counts across algorithms (BUG-33)
are safely truncated to the common minimum length instead of crashing the paired Friedman test.
WHY: When checkpoints contain different repetition counts (e.g. CPU with 5 runs and GPU with 10),
or when isolated runs fail, statistical analysis must remain resilient without raising ValueError.
EXPECTED: Mismatched sample counts emit a warning, truncate arrays to min_len, and complete
the analysis cleanly. If min_len < 2, the Friedman test is skipped with informative logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.problem_statistics import perform_statistical_analysis
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.analysis.stats_utils import holm_bonferroni_correction

if TYPE_CHECKING:
  from pathlib import Path

  import pytest


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


def test_perform_statistical_analysis_ragged_arrays_truncates_and_warns(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Algorithms with mismatched sample counts [5, 10, 10, 10].

  WHY: BUG-33 — Checkpoints with heterogeneous repetitions previously raised
  `ValueError: All data_sets must have the same number of observations`.
  EXPECTED: Emits a warning naming problem, lengths, and min_len; truncates
  cost arrays to 5 observations; completes without ValueError; computes Friedman test.
  """
  rng = np.random.default_rng(42)
  analyzer = StatisticalAnalyzer(alpha=0.05)

  results: dict[str, dict[str, Any]] = {
    "Alg_CPU": _mock_result(list(rng.normal(100.0, 5.0, size=5))),
    "Alg_GPU_Naive": _mock_result(list(rng.normal(95.0, 4.0, size=10))),
    "Alg_GPU_Opt": _mock_result(list(rng.normal(90.0, 3.0, size=10))),
    "Alg_GPU_Mono": _mock_result(list(rng.normal(88.0, 3.0, size=10))),
  }

  with caplog.at_level(logging.WARNING):
    stats = perform_statistical_analysis(
      problem_name="berlin52",
      optimal_cost=7542.0,
      results=results,
      analyzer=analyzer,
    )

  # Verify warning message was logged
  warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
  expected_msg = (
    "Mismatched sample counts across algorithms for problem 'berlin52': lengths=[5, 10, 10, 10]. "
    "Truncating to common minimum length 5 for paired Friedman test."
  )
  assert any(expected_msg in r.getMessage() for r in warning_records)

  # Analysis completes without crashing and includes friedman_test
  assert "friedman_test" in stats
  assert stats["friedman_test"] is not None
  assert "statistic" in stats["friedman_test"]
  assert "p_value" in stats["friedman_test"]
  assert "statistic_f" in stats["friedman_test"]
  assert "df1" in stats["friedman_test"]
  assert "df2" in stats["friedman_test"]
  assert "p_value_f" in stats["friedman_test"]
  assert stats["problem_name"] == "berlin52"


def test_perform_statistical_analysis_min_len_less_than_two_skips_friedman(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Algorithms where common minimum sample length is < 2 (e.g. [1, 5, 5]).

  WHY: A paired Friedman test requires at least 2 paired observations (N >= 2).
  EXPECTED: Emits warning on mismatch, logs informative message skipping Friedman,
  and returns neutral/empty Friedman test result.
  """
  analyzer = StatisticalAnalyzer(alpha=0.05)

  results: dict[str, dict[str, Any]] = {
    "Alg1": _mock_result([100.0]),
    "Alg2": _mock_result([95.0, 94.0, 93.0, 92.0, 91.0]),
    "Alg3": _mock_result([90.0, 89.0, 88.0, 87.0, 86.0]),
  }

  with caplog.at_level(logging.INFO):
    stats = perform_statistical_analysis(
      problem_name="kroA100",
      optimal_cost=21282.0,
      results=results,
      analyzer=analyzer,
    )

  assert "friedman_test" in stats
  assert stats["friedman_test"]["statistic"] == 0.0
  assert stats["friedman_test"]["p_value"] == 1.0
  assert stats["friedman_test"]["statistic_f"] == 0.0
  assert stats["friedman_test"]["df1"] == 2
  assert stats["friedman_test"]["df2"] == 0
  assert stats["friedman_test"]["p_value_f"] == 1.0
  assert stats["friedman_test"]["significant"] is False


def test_perform_statistical_analysis_equal_lengths_runs_without_warning(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Standard case where all algorithms have equal length (e.g. [10, 10, 10]).

  EXPECTED: No mismatch warning is logged, Friedman test is executed normally.
  """
  rng = np.random.default_rng(123)
  analyzer = StatisticalAnalyzer(alpha=0.05)

  results: dict[str, dict[str, Any]] = {
    "Alg1": _mock_result(list(rng.normal(100.0, 5.0, size=10))),
    "Alg2": _mock_result(list(rng.normal(95.0, 4.0, size=10))),
    "Alg3": _mock_result(list(rng.normal(90.0, 3.0, size=10))),
  }

  with caplog.at_level(logging.WARNING):
    stats = perform_statistical_analysis(
      problem_name="eil51",
      optimal_cost=426.0,
      results=results,
      analyzer=analyzer,
    )

  warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
  assert not any("Mismatched sample counts" in r.getMessage() for r in warning_records)
  assert "friedman_test" in stats
  assert stats["friedman_test"] is not None


def test_holm_bonferroni_step_down_termination_non_monotonic() -> None:
  """WHAT: Test Holm-Bonferroni correction with non-monotonic adjusted thresholds [0.03, 0.04].

  WHY: BUG-46 / DP12-23 — When hypothesis 1 fails (0.03 >= 0.05/2 = 0.025), step-down testing
  must terminate, marking subsequent hypotheses as non-rejected even if 0.04 < 0.05/1.
  EXPECTED: reject=[False, False] and pvals_corrected=[0.06, 0.06].
  """
  reject, pvals_corrected = holm_bonferroni_correction([0.03, 0.04], alpha=0.05)
  assert reject == [False, False]
  assert pvals_corrected == [0.06, 0.06]


def test_holm_bonferroni_empty_and_unsorted_and_threshold_equality() -> None:
  """WHAT: Test Holm-Bonferroni correction on empty input, unsorted input, and threshold equality.

  WHY: Review Gate 22c181dd — Holm-Bonferroni must cleanly handle empty lists, properly restore
  original ordering when input p-values are not pre-sorted, and correctly reject at exact threshold
  equality (adjusted p-value == alpha).
  """
  # 1. Empty list
  reject_empty, pvals_empty = holm_bonferroni_correction([])
  assert reject_empty == []
  assert pvals_empty == []

  # 2. Unsorted inputs restoring original indices
  reject_unsorted, pvals_unsorted = holm_bonferroni_correction([0.04, 0.01], alpha=0.05)
  assert reject_unsorted == [True, True]
  assert pvals_unsorted == [0.04, 0.02]

  # 3. Exact threshold equality
  reject_eq, pvals_eq = holm_bonferroni_correction([0.025, 0.05], alpha=0.05)
  assert reject_eq == [True, True]
  assert pvals_eq == [0.05, 0.05]


def test_perform_statistical_analysis_custom_baseline_algorithm() -> None:
  """WHAT: Test perform_statistical_analysis respects custom baseline_algorithm parameter.

  WHY: Task 12.39 (BUG-55 / DP12-28) — Make baseline_algorithm configurable instead of
  hardcoded to 'CPU'.
  """
  analyzer = StatisticalAnalyzer()
  results = {
    "CPU": {"raw_costs": [100.0, 102.0], "raw_times": [10.0, 12.0], "mean_time": 11.0},
    "HybridNaive": {"raw_costs": [90.0, 92.0], "raw_times": [5.0, 6.0], "mean_time": 5.5},
    "FullGPUModular": {"raw_costs": [85.0, 87.0], "raw_times": [1.0, 2.0], "mean_time": 1.5},
  }

  stats = perform_statistical_analysis(
    problem_name="eil51",
    optimal_cost=426.0,
    results=results,
    analyzer=analyzer,
    baseline_algorithm="HybridNaive",
  )
  assert stats["baseline_algorithm"] == "HybridNaive"


def test_perform_statistical_analysis_missing_baseline_warns_and_falls_back(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: Test missing baseline_algorithm logs warning and falls back to CPU or first algorithm.

  WHY: Task 12.39 (BUG-55 / DP12-28) — Resilient fallback when user specifies unavailable baseline.
  """
  analyzer = StatisticalAnalyzer()
  results = {
    "HybridNaive": {"raw_costs": [90.0, 92.0], "raw_times": [5.0, 6.0], "mean_time": 5.5},
    "FullGPUModular": {"raw_costs": [85.0, 87.0], "raw_times": [1.0, 2.0], "mean_time": 1.5},
  }

  with caplog.at_level(logging.WARNING):
    stats = perform_statistical_analysis(
      problem_name="eil51",
      optimal_cost=426.0,
      results=results,
      analyzer=analyzer,
      baseline_algorithm="NonExistentAlg",
    )

  assert stats["baseline_algorithm"] == "HybridNaive"
  assert any("Requested baseline_algorithm 'NonExistentAlg' not found" in r.getMessage() for r in caplog.records)


def test_perform_statistical_analysis_default_baseline_picks_first_even_when_cpu_present_at_index_gt_0() -> None:
  """WHAT: When baseline_algorithm=None, perform_statistical_analysis picks algorithm_names[0]
  even if 'CPU' is present later in results.

  WHY: Task 12.39 (BUG-55 / DP12-9 / DP12-28) — Eliminate hardcoded 'CPU' fallback so baseline
  resolution cannot diverge from report_generator.
  """
  analyzer = StatisticalAnalyzer()
  # Dict insertion order ensures Hybrid is index 0 and CPU is index 1
  results = {
    "Hybrid": {"raw_costs": [90.0, 92.0], "raw_times": [5.0, 6.0], "mean_time": 5.5},
    "CPU": {"raw_costs": [100.0, 102.0], "raw_times": [10.0, 12.0], "mean_time": 11.0},
  }

  stats = perform_statistical_analysis(
    problem_name="eil51",
    optimal_cost=426.0,
    results=results,
    analyzer=analyzer,
    baseline_algorithm=None,
  )
  assert stats["baseline_algorithm"] == "Hybrid"


def test_baseline_resolution_matches_between_statistics_and_report_generator(tmp_path: Path) -> None:
  """WHAT: When baseline_algorithm=None, statistics and report tables agree on baseline.

  WHY: Task 12.40 (BUG-56 / DP12-29) — Prevent divergence where statistics picked insertion
  order while report_generator picked sorted order.
  """
  from benchmarking.analysis.report_generator import generate_result_tables
  from benchmarking.runner.config_loader import ProblemConfig

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

  stats = perform_statistical_analysis(
    problem_name="eil51",
    optimal_cost=426.0,
    results=results,
    analyzer=analyzer,
    baseline_algorithm=None,
  )
  # Stats resolves to first in execution/insertion order: HybridNaive
  assert stats["baseline_algorithm"] == "HybridNaive"

  all_results = {"eil51": results}
  problem_configs = [ProblemConfig(name="eil51", optimal=426.0, size=51)]
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="canonical_baseline_test",
    baseline_algorithm=None,
  )
  md_content = (tmp_path / "canonical_baseline_test.md").read_text(encoding="utf-8")
  # HybridNaive must be the 1.00x baseline in Table 2, not FullGPUModular (sorted first)
  assert "| HybridNaive |" in md_content
  assert "| 1.00x |" in md_content
  # FullGPUModular speedup is 5.5 / 2.5 = 2.20x vs HybridNaive
  assert "2.20x" in md_content


def test_perform_statistical_analysis_pairwise_includes_effect_size_and_tie_diagnostics() -> None:
  """WHAT: pairwise_comparisons in problem statistics contains effect_size and tie diagnostics.

  WHY: Task 13.11 (BUG-60 / DP13-4) — Diagnostics from StatisticalSummary must be exposed
  in the problem statistics dict for downstream consumers and JSON serialization.
  """
  analyzer = StatisticalAnalyzer()
  results = {
    "A": {"raw_costs": [10.0, 10.0, 10.0, 12.0], "raw_times": [1.0, 1.0, 1.0, 1.0], "mean_time": 1.0},
    "B": {"raw_costs": [10.0, 10.0, 10.0, 10.0], "raw_times": [2.0, 2.0, 2.0, 2.0], "mean_time": 2.0},
  }

  stats = perform_statistical_analysis(
    problem_name="test_prob",
    optimal_cost=10.0,
    results=results,
    analyzer=analyzer,
    baseline_algorithm="A",
  )

  pairwise = stats["pairwise_comparisons"]
  assert len(pairwise) == 1
  comp = pairwise[0]
  assert "effect_size" in comp
  assert "test_statistic" in comp
  assert "df1" in comp
  assert "df2" in comp
  assert "cohens_d" not in comp
  assert "test_justification" in comp
  assert "zero_differences_count" in comp
  assert "effective_sample_size" in comp
  assert "insufficient_power_warning" in comp
  # In costs: diffs are [0, 0, 0, 2]; 3 zero diffs out of 4 -> majority zeros
  assert comp["zero_differences_count"] == 3
  assert comp["effective_sample_size"] == 1
  assert comp["insufficient_power_warning"] is True


def test_perform_statistical_analysis_includes_time_inference_and_clean_log_headers(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: perform_statistical_analysis runs paired Wilcoxon/Holm/Friedman on time_data
  alongside cost_data and omits stale 'paired t-test' / 'Cohen's d' log headers.

  WHY: Task 14.2 (DP14-2 / BUG-92) — Validate both quality equivalence and runtime speedup significance.
  """
  import pytest

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

  with caplog.at_level(logging.INFO):
    stats = perform_statistical_analysis(
      problem_name="berlin52",
      optimal_cost=100.0,
      results=results,
      analyzer=analyzer,
      baseline_algorithm="BaseAlg",
    )

  log_text = "\n".join(r.getMessage() for r in caplog.records)
  assert "paired t-test" not in log_text
  assert "Cohen's d" not in log_text
  assert "Wilcoxon signed-rank" in log_text
  assert "rank-biserial" in log_text
  assert "Hodges-Lehmann" in log_text

  assert "friedman_test_time" in stats
  assert stats["friedman_test_time"]["significant"] is True
  assert "nemenyi_posthoc_time" in stats
  assert len(stats["pairwise_comparisons_time"]) == 3
  for comp in stats["pairwise_comparisons_time"]:
    assert comp["metric_name"] == "time"
    assert comp["test_used"] == "wilcoxon_signed_rank"
    assert "p_value_corrected" in comp
    assert "significant_after_correction" in comp
    assert comp["ci_95_lower"] <= comp["ci_95_upper"]
    assert comp["effect_size"] == pytest.approx(1.0)


def test_perform_statistical_analysis_time_length_validation_and_strict_mode(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: Validate time length mismatch warnings, incomplete raw_times fallback, and strict_lengths=True."""
  import pytest

  analyzer = StatisticalAnalyzer()

  # Case 1: Mixed empty and non-empty raw_times warns and skips time inference cleanly
  mixed_results = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [], "mean_time": 2.0},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2], "mean_time": 1.1},
  }
  with caplog.at_level(logging.WARNING):
    mixed_stats = perform_statistical_analysis("p_mixed", 10.0, mixed_results, analyzer)
  assert mixed_stats["pairwise_comparisons_time"] == []
  assert any("Incomplete raw time sample counts" in r.getMessage() for r in caplog.records)

  # Case 2: Mismatched time_lengths warns loudly (never silently truncates)
  caplog.clear()
  ragged_times = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [2.0, 2.1, 2.2]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2, 1.3]},
  }
  with caplog.at_level(logging.WARNING):
    ragged_stats = perform_statistical_analysis("p_ragged_t", 10.0, ragged_times, analyzer)
  assert len(ragged_stats["pairwise_comparisons_time"]) == 1
  assert any("Mismatched time sample counts" in r.getMessage() for r in caplog.records)

  # Case 3: strict_lengths=True raises ValueError on mismatched lengths
  with pytest.raises(ValueError, match="Mismatched time sample counts"):
    perform_statistical_analysis("p_ragged_t", 10.0, ragged_times, analyzer, strict_lengths=True)

  ragged_costs = {
    "A": {"raw_costs": [10.0, 11.0], "raw_times": [2.0, 2.1]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.1, 1.2]},
  }
  with pytest.raises(ValueError, match="Mismatched cost sample counts"):
    perform_statistical_analysis("p_ragged_c", 10.0, ragged_costs, analyzer, strict_lengths=True)

  # Case 4: strict_lengths=True raises ValueError even when ALL variants omit raw_times
  missing_all_times = {
    "A": {"raw_costs": [10.0, 11.0, 12.0], "mean_time": 2.0},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "mean_time": 1.0},
  }
  with pytest.raises(ValueError, match="Incomplete raw time arrays"):
    perform_statistical_analysis("p_no_times", 10.0, missing_all_times, analyzer, strict_lengths=True)

  # Case 5: strict_lengths=False with ragged arrays and no precomputed means preserves true full means in summary
  caplog.clear()
  ragged_no_mean = {
    "A": {"raw_costs": [10.0, 10.0], "raw_times": [1.0, 1.0]},
    "B": {"raw_costs": [10.0, 10.0, 100.0, 100.0], "raw_times": [1.0, 1.0, 10.0, 10.0]},
  }
  with caplog.at_level(logging.INFO):
    perform_statistical_analysis("p_ragged_nomean", 10.0, ragged_no_mean, analyzer, strict_lengths=False)
  log_text = "\n".join(r.getMessage() for r in caplog.records)
  assert "55.00" in log_text
  assert "5.50" in log_text

  # Case 6: constant non-zero offset logs constant paired difference (not identical)
  caplog.clear()
  const_offset = {
    "A": {"raw_costs": [11.0, 12.0, 13.0], "raw_times": [2.0, 2.0, 2.0]},
    "B": {"raw_costs": [10.0, 11.0, 12.0], "raw_times": [1.0, 1.0, 1.0]},
  }
  with caplog.at_level(logging.INFO):
    perform_statistical_analysis("p_const_offset", 10.0, const_offset, analyzer, strict_lengths=True)
  offset_log = "\n".join(r.getMessage() for r in caplog.records)
  assert "Constant paired difference" in offset_log
