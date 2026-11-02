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

EXPECTED_COMPARISON_KEYS = {"problems", "algorithms", "n_problems", "n_algorithms", "friedman_test"}
EXPECTED_FRIEDMAN_KEYS = ["statistic", "p_value", "significant", "post_hoc_required"]


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
