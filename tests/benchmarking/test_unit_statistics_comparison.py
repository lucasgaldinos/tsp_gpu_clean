"""Unit tests for the comparison branches of `StatisticalAnalyzer`.

WHAT: Covers normality selection (`test_normality`), the paired-comparison
branches (paired t-test, Wilcoxon, short-circuits) and the Friedman test guards
and degenerate result.
WHY: These branches decide which statistical test is reported and whether the
Nemenyi post-hoc path opens; wrong values would silently corrupt every results
chapter.
EXPECTED: Documented branch selection, exact statistics, and exact error guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.analysis.statistics import StatisticalAnalyzer

EXPECTED_FRIEDMAN_KEYS = ["statistic", "p_value", "significant", "post_hoc_required"]


# ============================================================================
# StatisticalAnalyzer.test_normality
# ============================================================================


@pytest.mark.parametrize(
  ("data", "expected"),
  [
    pytest.param([], (0.0, False), id="empty"),
    pytest.param([1.0], (0.0, False), id="single"),
    pytest.param([1.0, 2.0], (0.0, False), id="two_points"),
  ],
)
def test_normality_rejects_small_samples(data: list[float], expected: tuple[float, bool]) -> None:
  """WHAT: `test_normality` with n < 3.

  WHY: Shapiro-Wilk is not defined below three points; the guard decides the
  test-selection branch for tiny paired samples.
  EXPECTED: exactly `(0.0, False)`.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  result = analyzer.test_normality(np.array(data))

  # Assert
  assert result == expected


def test_normality_constant_data_is_normal() -> None:
  """WHAT: `test_normality` on zero-variance data.

  WHY: The fast path avoids a Shapiro call that would error or warn; the value is
  pinned so the branch cannot drift.
  EXPECTED: exactly `(1.0, True)`.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  result = analyzer.test_normality(np.array([2.0, 2.0, 2.0, 2.0]))

  # Assert
  assert result == (1.0, True)


def test_normality_accepts_linear_data_and_rejects_outlier() -> None:
  """WHAT: Shapiro branch on a near-normal ramp and on a heavy outlier.

  WHY: The paired-comparison branch (t-test vs Wilcoxon) hinges on this decision.
  EXPECTED: ramp is normal (p >= 0.05); [1,1,1,1,20] is not (p < 0.05).
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  p_ramp, ramp_is_normal = analyzer.test_normality(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
  p_outlier, outlier_is_normal = analyzer.test_normality(np.array([1.0, 1.0, 1.0, 1.0, 20.0]))

  # Assert
  assert bool(ramp_is_normal) is True
  assert 0.05 <= p_ramp <= 1.0
  assert bool(outlier_is_normal) is False
  assert p_outlier < 0.05


# ============================================================================
# StatisticalAnalyzer.paired_comparison
# ============================================================================


def test_paired_comparison_rejects_mismatched_lengths() -> None:
  """WHAT: Paired comparison with different sample sizes.

  WHY: Pairing requires equal-length arrays; silently truncating would corrupt
  the paired test.
  EXPECTED: `ValueError` mentioning the lengths contract.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act / Assert
  with pytest.raises(ValueError, match="same length"):
    analyzer.paired_comparison(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "L", "m")


def test_paired_comparison_single_sample_short_circuits() -> None:
  """WHAT: n = 1 branch.

  WHY: No test can run on a single pair; the summary must still carry the means
  and a collapsed CI.
  EXPECTED: sample_size 1, `no_test_needed`, difference 2.0, CI (2.0, 2.0).
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  summary = analyzer.paired_comparison(np.array([3.0]), np.array([1.0]), "CPU vs GPU", "runtime_seconds")

  # Assert
  assert summary.sample_size == 1
  assert summary.test_used == "no_test_needed"
  assert summary.mean_a == pytest.approx(3.0)
  assert summary.mean_b == pytest.approx(1.0)
  assert summary.mean_difference == pytest.approx(2.0)
  assert summary.ci_95_difference[0] == pytest.approx(2.0)
  assert summary.ci_95_difference[1] == pytest.approx(2.0)
  assert summary.p_value == 1.0


def test_paired_comparison_empty_sample_reports_zero_means() -> None:
  """WHAT: n = 0 branch.

  WHY: An empty metric list must not raise; the summary documents zero means.
  EXPECTED: sample_size 0, means 0.0, `no_test_needed`.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  summary = analyzer.paired_comparison(np.array([]), np.array([]), "L", "m")

  # Assert
  assert summary.sample_size == 0
  assert summary.mean_a == 0.0
  assert summary.mean_b == 0.0
  assert summary.test_used == "no_test_needed"


def test_paired_comparison_identical_samples_skip_test() -> None:
  """WHAT: Zero-variance difference branch with identical samples.

  WHY: A t-test on identical pairs is undefined; the short-circuit is the
  documented behavior.
  EXPECTED: `no_test_needed`, p = 1.0, difference 0.0, no effect.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data = np.array([2.0, 4.0, 6.0])

  # Act
  summary = analyzer.paired_comparison(data, data, "L", "m")

  # Assert
  assert summary.test_used == "no_test_needed"
  assert summary.p_value == 1.0
  assert summary.mean_difference == 0.0
  assert summary.effect_size == 0.0


def test_paired_comparison_zero_variance_reports_true_mean_difference() -> None:
  """WHAT: mean difference when every pair differs by the same constant.

  WHY: `StatisticalSummary.mean_difference` is documented as mean(A - B); a
  constant offset is still a real difference and must be reported.
  EXPECTED: 1.0, not 0.0.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act
  summary = analyzer.paired_comparison(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), "L", "m")

  # Assert
  assert summary.mean_difference == pytest.approx(1.0)


def test_paired_comparison_normal_path_uses_paired_t_test() -> None:
  """WHAT: Both samples normal -> paired t-test branch.

  WHY: The reported test name and statistics must match the branch chosen.
  EXPECTED: `paired_t_test`, difference -11.5, Cohen's d -3.8877..., p ~ 2.34e-05.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  data_b = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

  # Act
  summary = analyzer.paired_comparison(data_a, data_b, "CPU vs GPU", "runtime_seconds")

  # Assert
  assert summary.test_used == "paired_t_test"
  assert bool(summary.is_normal) is True
  assert summary.mean_difference == pytest.approx(-11.5)
  assert summary.std_a == pytest.approx(1.8708286933869707)
  assert summary.std_b == pytest.approx(3.7416573867739413)
  assert summary.effect_size == pytest.approx(-3.8877095717511763, rel=1e-9)
  assert summary.p_value == pytest.approx(2.3404444710375406e-05, rel=1e-6)
  assert summary.get_speedup() == pytest.approx(3.5 / 15.0)


def test_paired_comparison_non_normal_path_uses_wilcoxon() -> None:
  """WHAT: One non-normal sample -> Wilcoxon signed-rank branch.

  WHY: The methodology section promises Wilcoxon when normality fails.
  EXPECTED: `wilcoxon_signed_rank`, mean difference 2.8, effect size ~0.466.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
  data_b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

  # Act
  summary = analyzer.paired_comparison(data_a, data_b, "L", "m")

  # Assert
  assert summary.test_used == "wilcoxon_signed_rank"
  assert bool(summary.is_normal) is False
  assert summary.mean_difference == pytest.approx(2.8)
  assert summary.effect_size == pytest.approx(0.46601986570902426, rel=1e-9)
  assert 0.0 < summary.p_value <= 1.0


# ============================================================================
# StatisticalAnalyzer.friedman_test
# ============================================================================


def test_friedman_requires_equal_lengths() -> None:
  """WHAT: Friedman test with unpaired data sets.

  WHY: Unpaired sets were the original cross-problem analysis bug this module
  documents; the guard must stay.
  EXPECTED: `ValueError` mentioning paired samples.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act / Assert
  with pytest.raises(ValueError, match="same number of observations"):
    analyzer.friedman_test([np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0])])


def test_friedman_requires_three_algorithms() -> None:
  """WHAT: Friedman test with fewer than three data sets.

  WHY: The test is defined for k >= 3; two sets must route to a paired test.
  EXPECTED: `ValueError` mentioning k >= 3.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()

  # Act / Assert
  with pytest.raises(ValueError, match="at least 3 algorithms"):
    analyzer.friedman_test([np.array([1.0, 2.0]), np.array([2.0, 1.0])])


def test_friedman_constant_data_returns_neutral_result() -> None:
  """WHAT: Degenerate (zero-variance) Friedman input.

  WHY: Constant gaps carry no ranking information; the neutral result keeps the
  post-hoc path closed.
  EXPECTED: keys in documented order with statistic 0.0, p 1.0, both booleans False.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  constant = np.array([5.0, 5.0, 5.0])

  # Act
  result = analyzer.friedman_test([constant, constant, constant])

  # Assert
  assert list(result) == EXPECTED_FRIEDMAN_KEYS
  assert result["statistic"] == 0.0
  assert result["p_value"] == 1.0
  assert result["significant"] is False
  assert result["post_hoc_required"] is False
