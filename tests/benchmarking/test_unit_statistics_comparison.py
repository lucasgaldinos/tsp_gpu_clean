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
  assert summary.test_statistic == 0.0
  assert summary.df1 is None
  assert summary.df2 is None


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
  EXPECTED: `no_test_needed`, p = 1.0, difference 0.0, no effect, zero_differences_count=n.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data = np.array([2.0, 4.0, 6.0])

  # Act
  summary = analyzer.paired_comparison(data, data, "L", "m")

  # Assert
  assert summary.test_used == "no_test_needed"
  assert summary.test_justification == "identical_samples_no_test_needed"
  assert summary.p_value == 1.0
  assert summary.test_statistic == 0.0
  assert summary.df1 is None
  assert summary.df2 is None
  assert summary.mean_difference == 0.0
  assert summary.effect_size == 0.0
  assert summary.zero_differences_count == 3
  assert summary.effective_sample_size == 0
  assert summary.insufficient_power_warning is True


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
  assert summary.test_used == "no_test_needed"
  assert summary.test_justification == "constant_difference_no_test_needed"
  assert summary.effect_size == pytest.approx(1.0)
  assert summary.zero_differences_count == 0
  assert summary.effective_sample_size == 3
  assert summary.insufficient_power_warning is False


def test_paired_comparison_normal_path_uses_wilcoxon() -> None:
  """WHAT: Both samples normal -> still uses Wilcoxon signed-rank test.

  WHY: Task 13.11 (BUG-60 / DP13-4) — Fixed non-parametric protocol per Demšar (2006).
  Normality is purely descriptive and does NOT branch to a paired t-test.
  EXPECTED: `wilcoxon_signed_rank`, difference -11.5, rank-biserial effect size -1.0.
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  data_b = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

  # Act
  summary = analyzer.paired_comparison(data_a, data_b, "CPU vs GPU", "runtime_seconds")

  # Assert
  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.test_justification == "demsar2006_wilcoxon_default"
  assert summary.mean_difference == pytest.approx(-11.5)
  assert summary.std_a == pytest.approx(1.8708286933869707)
  assert summary.std_b == pytest.approx(3.7416573867739413)
  # All differences (data_a - data_b) are negative -> rank-biserial correlation = -1.0
  assert summary.effect_size == pytest.approx(-1.0)
  assert 0.0 < summary.p_value <= 1.0
  assert summary.test_statistic == pytest.approx(0.0)
  assert summary.df1 is None
  assert summary.df2 is None
  assert summary.get_speedup() == pytest.approx(3.5 / 15.0)


def test_paired_comparison_normal_samples_n30_uses_wilcoxon_not_t() -> None:
  """WHAT: Two perfectly normal samples with n=30 receive Wilcoxon, proving data-independence.

  WHY: Task 13.11 (DoD requirement) — Two normal samples with n=30 must receive Wilcoxon signed-rank
  (not paired t-test), proving that test selection does not depend on the data.
  """
  # Arrange: construct perfectly symmetric, normal-passing data (p > 0.05)
  rng = np.random.default_rng(42)
  analyzer = StatisticalAnalyzer()
  data_a = rng.normal(loc=100.0, scale=10.0, size=30)
  data_b = data_a + rng.normal(loc=5.0, scale=2.0, size=30)

  _p_norm_a, is_norm_a = analyzer.test_normality(data_a)
  _p_norm_b, is_norm_b = analyzer.test_normality(data_b)
  assert bool(is_norm_a) and bool(is_norm_b), "Precondition: both samples must pass normality"

  # Act
  summary = analyzer.paired_comparison(data_a, data_b, "NormA vs NormB", "cost")

  # Assert
  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.test_justification == "demsar2006_wilcoxon_default"
  assert summary.sample_size == 30
  assert summary.ci_95_difference[0] < summary.ci_95_difference[1]


def test_paired_comparison_non_normal_path_uses_wilcoxon() -> None:
  """WHAT: One non-normal sample -> Wilcoxon signed-rank branch.

  WHY: Task 13.11 — The methodology section promises Wilcoxon when normality fails.
  EXPECTED: `wilcoxon_signed_rank`, mean difference 2.8, rank-biserial effect size -0.333...
  """
  # Arrange
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
  data_b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

  # Act
  summary = analyzer.paired_comparison(data_a, data_b, "L", "m")

  # Assert
  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.mean_difference == pytest.approx(2.8)
  # 4 diffs are -1, 1 diff is +18; ranks: 2.5 for -1s, 5 for +18; r = (5 - 10) / 15 = -0.333333333
  assert summary.effect_size == pytest.approx(-1.0 / 3.0)
  assert 0.0 < summary.p_value <= 1.0


def test_paired_comparison_tie_diagnostics_warns_on_majority_zeros(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: Tie diagnostics warn when more than half of differences are zero.

  WHY: Task 13.11 (BUG-60) — When > 50% of differences are zero, effective n is reduced
  and power is insufficient.
  """
  import logging
  analyzer = StatisticalAnalyzer()
  # 6 pairs, 4 have zero difference: [0, 0, 0, 0, 1, 2]
  data_a = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 7.0])
  data_b = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

  with caplog.at_level(logging.WARNING):
    summary = analyzer.paired_comparison(data_a, data_b, "A vs B", "cost")

  assert summary.zero_differences_count == 4
  assert summary.effective_sample_size == 2
  assert summary.insufficient_power_warning is True
  assert any("More than half of paired differences are zero" in r.getMessage() for r in caplog.records)


def test_compute_hodges_lehmann_location_and_ci() -> None:
  """WHAT: Test Hodges-Lehmann estimator and distribution-free CI on known differences."""
  analyzer = StatisticalAnalyzer()
  diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  hl_point, (ci_low, ci_high) = analyzer.compute_hodges_lehmann(diffs, confidence=0.95)

  assert hl_point == pytest.approx(3.0)
  assert ci_low <= hl_point <= ci_high
  assert ci_low >= 1.0
  assert ci_high <= 5.0


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
  assert result["statistic_f"] == 0.0
  assert result["df1"] == 2
  assert result["df2"] == 4
  assert result["p_value_f"] == 1.0
  assert result["significant"] is False
  assert result["post_hoc_required"] is False


def test_friedman_iman_davenport_f_matches_manual_calculation() -> None:
  """WHAT: Friedman test computes Iman-Davenport F_F matching manual calculation.

  WHY: Task 13.8 (BUG-61 / DP13-3) — Verify F_F = ((N-1)*chi2)/(N*(k-1)-chi2),
  df1 = k-1, df2 = (k-1)*(N-1), and p_value_f = f.sf(F_F, df1, df2) on a
  hand-computable N=6, k=3 matrix.
  """
  from scipy import stats

  analyzer = StatisticalAnalyzer(alpha=0.05)
  # Row ranks: [1,2,3], [2,1,3], [1,3,2], [2,1,3], [1,2,3], [1,2,3]
  # Rank sums: 8, 11, 17 -> chi2 = (12/(6*3*4))*(64+121+289) - 3*6*4 = 7.0
  # F_F = (5 * 7.0) / (6 * 2 - 7.0) = 35 / 5 = 7.0; df1 = 2, df2 = 10
  alg_a = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0])
  alg_b = np.array([2.0, 1.0, 3.0, 1.0, 2.0, 2.0])
  alg_c = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 3.0])

  result = analyzer.friedman_test([alg_a, alg_b, alg_c])

  assert list(result) == EXPECTED_FRIEDMAN_KEYS
  assert result["statistic"] == pytest.approx(7.0, rel=1e-12)
  assert result["statistic_f"] == pytest.approx(7.0, rel=1e-12)
  assert result["df1"] == 2
  assert result["df2"] == 10
  expected_p_f = float(stats.f.sf(7.0, 2, 10))
  assert result["p_value_f"] == pytest.approx(expected_p_f, rel=1e-12)
  assert result["significant"] is True
  assert result["post_hoc_required"] is True


def test_friedman_significance_uses_iman_davenport_p_value_f() -> None:
  """WHAT: When p_value >= alpha but p_value_f < alpha, significance follows p_value_f.

  WHY: Task 13.8 (BUG-61 / DP13-3) — Iman-Davenport F_F is the decision gate for
  omnibus significance and Nemenyi post-hoc activation.
  """
  analyzer = StatisticalAnalyzer(alpha=0.05)
  # N=3, k=4 matrix: chi2 = 7.0 (p_chi2 ~= 0.0719 >= 0.05), F_F = 7.0 (p_F ~= 0.0219 < 0.05)
  mat = np.array(
    [
      [1.0, 2.0, 3.0, 4.0],
      [1.0, 3.0, 2.0, 4.0],
      [2.0, 1.0, 3.0, 4.0],
    ]
  )

  result = analyzer.friedman_test([mat[:, j] for j in range(4)])

  assert result["p_value"] >= 0.05
  assert result["p_value_f"] < 0.05
  assert result["df1"] == 3
  assert result["df2"] == 6
  assert result["significant"] is True
  assert result["post_hoc_required"] is True


def test_friedman_boundary_single_observation_and_perfect_concordance() -> None:
  """WHAT: N < 2 returns neutral F_F=0.0, p_value_f=1.0; perfect concordance returns inf, p_value_f=0.0.

  WHY: Task 13.8 — Avoid ZeroDivisionError or NaN when denominator N*(k-1)-chi2 == 0 or N < 2.
  """
  analyzer = StatisticalAnalyzer(alpha=0.05)

  res_n1 = analyzer.friedman_test([np.array([1.0]), np.array([2.0]), np.array([3.0])])
  assert res_n1["statistic"] == 0.0
  assert res_n1["statistic_f"] == 0.0
  assert res_n1["df1"] == 2
  assert res_n1["df2"] == 0
  assert res_n1["p_value_f"] == 1.0
  assert res_n1["significant"] is False

  # Perfect concordance across N=5, k=3 -> chi2 = 10.0 = N*(k-1) -> denominator = 0
  res_perf = analyzer.friedman_test(
    [
      np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
      np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
      np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
    ]
  )
  assert res_perf["statistic"] == pytest.approx(10.0)
  assert res_perf["statistic_f"] == float("inf")
  assert res_perf["df1"] == 2
  assert res_perf["df2"] == 8
  assert res_perf["p_value_f"] == 0.0
  assert res_perf["significant"] is True
  assert res_perf["post_hoc_required"] is True
