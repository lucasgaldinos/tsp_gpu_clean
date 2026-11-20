"""Unit tests for the `compare_algorithms` aggregation contract.

WHAT: Covers the result shape of `benchmarking.analysis.aggregate_statistics.
compare_algorithms`, its minimum-design skip, incomplete-problem filtering and
baseline speedups.
WHY: Report tables read these fixed keys, counts and orderings; a drift would
silently corrupt the cross-problem analysis output.
EXPECTED: Aggregation dictionaries with the promised keys, counts and input
ordering, and `None` for insufficient designs.
"""

from __future__ import annotations

import pytest

from benchmarking.analysis.aggregate_statistics import compare_algorithms

EXPECTED_COMPARISON_KEYS = {
  "problems",
  "algorithms",
  "n_problems",
  "n_algorithms",
  "friedman_test",
  "mean_ranks",
}
EXPECTED_FRIEDMAN_KEYS = [
  "statistic",
  "p_value",
  "statistic_f",
  "df1",
  "df2",
  "p_value_f",
  "significant",
  "post_hoc_required",
]


def _constant_results(
  problems: list[str],
  algorithms: list[str],
  mean_times: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
  """Build `problem -> algorithm -> metrics` records with constant gaps.

  Constant gaps make the Friedman statistic degenerate (0.0), so the aggregation
  test never triggers the Nemenyi post-hoc path.
  """
  return {
    problem: {algo: {"mean_gap": 5.0, "mean_time": mean_times[algo]} for algo in algorithms} for problem in problems
  }


# ============================================================================
# compare_algorithms aggregation contract
# ============================================================================


@pytest.mark.parametrize(
  ("problems", "algorithms"),
  [
    pytest.param(["a", "b"], ["CPU", "G", "H"], id="too_few_problems"),
    pytest.param(["a", "b", "c"], ["CPU", "G"], id="too_few_algorithms"),
  ],
)
def test_compare_algorithms_skips_insufficient_designs(problems: list[str], algorithms: list[str]) -> None:
  """WHAT: Aggregation with fewer than 3 algorithms or 3 complete problems.

  WHY: Friedman needs a 3x3 minimum; the caller must distinguish "skipped" from
  "computed with zeros".
  EXPECTED: `None`.
  """
  # Arrange
  results = _constant_results(["a", "b", "c"], ["CPU", "G", "H"], {"CPU": 10.0, "G": 5.0, "H": 2.0})

  # Act
  outcome = compare_algorithms(results, problems, algorithms)

  # Assert
  assert outcome is None


def test_compare_algorithms_reports_counts_keys_and_input_order() -> None:
  """WHAT: Successful 3-algorithm x 3-problem aggregation.

  WHY: Report code reads fixed keys; the problem list must follow the caller's
  order, not the dict insertion order of `all_results`.
  EXPECTED: exact key set, counts 3/3, problems ["c","a","b"], algorithms preserved.
  """
  # Arrange
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["b", "a", "c"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})

  # Act
  outcome = compare_algorithms(results, ["c", "a", "b"], algorithms)

  # Assert
  assert outcome is not None
  assert set(outcome) == EXPECTED_COMPARISON_KEYS
  assert outcome["n_problems"] == 3
  assert outcome["n_algorithms"] == 3
  assert outcome["problems"] == ["c", "a", "b"], "problem order must follow the caller's list"
  assert outcome["algorithms"] == algorithms
  assert list(outcome["friedman_test"]) == EXPECTED_FRIEDMAN_KEYS


def test_compare_algorithms_filters_incomplete_problems() -> None:
  """WHAT: Problems missing one algorithm's results.

  WHY: Friedman requires complete pairing; incomplete problems must be excluded,
  and the surviving problems must keep their relative order.
  EXPECTED: n_problems 3 and "d" absent from the reported list.
  """
  # Arrange
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["a", "b", "c", "d"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})
  del results["d"]["H"]

  # Act
  outcome = compare_algorithms(results, ["d", "a", "b", "c"], algorithms)

  # Assert
  assert outcome is not None
  assert outcome["n_problems"] == 3
  assert outcome["problems"] == ["a", "b", "c"]


def test_compare_algorithms_computes_geometric_mean_speedups() -> None:
  """WHAT: Baseline speedups when `baseline_algorithm` is supplied.

  WHY: The speedup column is the headline result; baseline must be exactly 1.0
  and the others the geometric mean of per-problem time ratios.
  EXPECTED: CPU 1.0, G 2.0, H 5.0.
  """
  # Arrange
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["a", "b", "c"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})

  # Act
  outcome = compare_algorithms(results, ["a", "b", "c"], algorithms, baseline_algorithm="CPU")

  # Assert
  assert outcome is not None
  assert set(outcome) == EXPECTED_COMPARISON_KEYS | {"speedups"}
  assert outcome["speedups"]["CPU"] == 1.0
  assert outcome["speedups"]["G"] == pytest.approx(2.0, rel=1e-12)
  assert outcome["speedups"]["H"] == pytest.approx(5.0, rel=1e-12)


def test_compare_algorithms_omits_speedups_without_baseline() -> None:
  """WHAT: No baseline requested.

  WHY: Consumers test key presence to decide whether to render the speedup column.
  EXPECTED: "speedups" absent from the result.
  """
  # Arrange
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["a", "b", "c"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})

  # Act
  outcome = compare_algorithms(results, ["a", "b", "c"], algorithms)

  # Assert
  assert outcome is not None
  assert "speedups" not in outcome


def test_compare_algorithms_returns_mean_ranks() -> None:
  """WHAT: compare_algorithms always returns per-algorithm mean_ranks across problems.

  WHY: Task 13.10 (BUG-62 / DP13-3) — Delivering CD without mean_ranks exposes only half
  of the Critical Difference artifact; mean_ranks must obey sum(R_j) = k(k+1)/2.
  """
  algorithms = ["Best", "Mid", "Worst"]
  results = {
    p: {
      "Best": {"mean_gap": 0.5, "mean_time": 1.0},
      "Mid": {"mean_gap": 2.0, "mean_time": 2.0},
      "Worst": {"mean_gap": 5.0, "mean_time": 3.0},
    }
    for p in ["p1", "p2", "p3", "p4"]
  }

  outcome = compare_algorithms(results, ["p1", "p2", "p3", "p4"], algorithms)

  assert outcome is not None
  assert outcome["mean_ranks"] == {
    "Best": pytest.approx(1.0),
    "Mid": pytest.approx(2.0),
    "Worst": pytest.approx(3.0),
  }
  assert sum(outcome["mean_ranks"].values()) == pytest.approx(6.0)


def test_compare_algorithms_supports_time_metric_and_higher_is_better() -> None:
  """WHAT: compare_algorithms parameterizes metric_key and higher_is_better rank direction.

  WHY: Task 14.2 (DP14-2) — Cross-problem comparison must evaluate execution time
  and speedup in addition to optimality gap.
  EXPECTED: Lower mean_time receives rank 1 when higher_is_better=False;
  higher score receives rank 1 when higher_is_better=True.
  """
  algorithms = ["Slow", "Medium", "Fast"]
  results = {
    p: {
      "Slow": {"mean_gap": 1.0, "mean_time": 10.0, "score": 10.0},
      "Medium": {"mean_gap": 2.0, "mean_time": 5.0, "score": 50.0},
      "Fast": {"mean_gap": 3.0, "mean_time": 1.0, "score": 100.0},
    }
    for p in ["p1", "p2", "p3", "p4"]
  }

  time_outcome = compare_algorithms(
    results,
    ["p1", "p2", "p3", "p4"],
    algorithms,
    baseline_algorithm="Slow",
    metric_key="mean_time",
    higher_is_better=False,
  )
  assert time_outcome is not None
  assert time_outcome["mean_ranks"] == {
    "Fast": pytest.approx(1.0),
    "Medium": pytest.approx(2.0),
    "Slow": pytest.approx(3.0),
  }
  assert time_outcome["friedman_test"]["significant"] is True
  assert time_outcome["speedups"]["Fast"] == pytest.approx(10.0)

  score_outcome = compare_algorithms(
    results,
    ["p1", "p2", "p3", "p4"],
    algorithms,
    metric_key="score",
    higher_is_better=True,
  )
  assert score_outcome is not None
  assert score_outcome["mean_ranks"] == {
    "Fast": pytest.approx(1.0),
    "Medium": pytest.approx(2.0),
    "Slow": pytest.approx(3.0),
  }


def test_perform_cross_problem_analysis_stratified_threshold_behavior(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: perform_cross_problem_analysis_stratified drops STRATUM 1/2 split when
  cpu_threshold is None and runs unified multi-metric analysis with baseline forwarding,
  while retaining stratified behavior when an integer threshold is supplied.

  WHY: Task 14.2 (DP14-2 / BUG-92) — Eliminate obsolete STRATUM 1/2 when cpu_threshold is None.
  """
  import logging

  from benchmarking.analysis.aggregate_statistics import perform_cross_problem_analysis_stratified
  from benchmarking.analysis.statistics import StatisticalAnalyzer
  from benchmarking.runner.config_loader import ProblemConfig

  analyzer = StatisticalAnalyzer()
  problem_configs = [
    ProblemConfig(name="p1", size=50, optimal=100.0),
    ProblemConfig(name="p2", size=80, optimal=200.0),
    ProblemConfig(name="p3", size=500, optimal=500.0),
  ]

  results = {
    p.name: {
      "BaseAlg": {"mean_gap": 1.0, "mean_time": 10.0},
      "GPU_A": {"mean_gap": 1.5, "mean_time": 5.0},
      "GPU_B": {"mean_gap": 2.0, "mean_time": 1.0},
    }
    for p in problem_configs
  }

  # Case 1: cpu_threshold=None -> unified multi-metric analysis, zero STRATUM 1/2 logs
  with caplog.at_level(logging.INFO):
    outcome_none = perform_cross_problem_analysis_stratified(
      all_results=results,
      problem_configs=problem_configs,
      analyzer=analyzer,
      cpu_threshold=None,
      baseline_algorithm="BaseAlg",
    )
  log_text = "\n".join(r.getMessage() for r in caplog.records)
  assert "STRATUM 1" not in log_text
  assert "STRATUM 2" not in log_text
  assert outcome_none["stratified"] is False
  assert "stratum1_small_problems" not in outcome_none
  assert "stratum2_all_gpu_problems" not in outcome_none
  assert outcome_none["cpu_threshold"] is None
  assert outcome_none["baseline_algorithm"] == "BaseAlg"
  assert outcome_none["gap_comparison"] is not None
  assert outcome_none["gap_comparison"]["problems"] == ["p1", "p2", "p3"]
  assert outcome_none["time_comparison"] is not None
  assert outcome_none["time_comparison"]["speedups"]["GPU_B"] == pytest.approx(10.0)

  # Case 2: cpu_threshold=100 -> Stratum 1 includes only p1, p2 (size <= 100), < 3 problems so skipped
  outcome_gated = perform_cross_problem_analysis_stratified(
    all_results=results,
    problem_configs=problem_configs,
    analyzer=analyzer,
    cpu_threshold=100,
    baseline_algorithm="BaseAlg",
  )
  assert outcome_gated["stratified"] is True
  assert outcome_gated["stratum1_small_problems"] is None
  assert outcome_gated["cpu_threshold"] == 100
