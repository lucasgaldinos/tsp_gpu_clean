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

from benchmarking.analysis.problem_statistics import holm_bonferroni_correction, perform_statistical_analysis
from benchmarking.analysis.statistics import StatisticalAnalyzer

if TYPE_CHECKING:
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




