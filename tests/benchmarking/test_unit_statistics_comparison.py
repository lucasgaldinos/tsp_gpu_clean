"""Unit tests for the comparison branches of `StatisticalAnalyzer`.

WHAT: Covers normality diagnostics (`test_normality`), unconditional Wilcoxon paired
comparisons (including constant non-zero differences at small n), exact/asymptotic
Hodges-Lehmann CIs, and Friedman / Iman-Davenport test guards.
WHY: Verifies that small-n constant differences never emit false p=0.0, normality
diagnostics never emit misleading 0.0/1.0 sentinels, marginal and difference CIs are
consistently distribution-free, and Friedman/Iman-Davenport F_F decisions are exact.
EXPECTED: Exact Wilcoxon p = 2^{-(n-1)} on constant non-zero diffs, NaN normality
sentinels for n < 3 or constant series, exact Hodges-Lehmann coverage guards for n <= 5,
and exact Friedman / Iman-Davenport statistics.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy import stats

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
  "low_power",
]


# ============================================================================
# StatisticalAnalyzer.test_normality
# ============================================================================


@pytest.mark.parametrize(
  "data",
  [
    pytest.param([], id="empty"),
    pytest.param([1.0], id="single"),
    pytest.param([1.0, 2.0], id="two_points"),
  ],
)
def test_normality_returns_nan_for_small_samples(data: list[float]) -> None:
  """WHAT: `test_normality` with n < 3 returns (NaN, False) rather than false p=0.0."""
  analyzer = StatisticalAnalyzer()
  p_val, is_normal = analyzer.test_normality(np.array(data))
  assert math.isnan(p_val)
  assert is_normal is False


def test_normality_constant_data_returns_nan_not_gaussian() -> None:
  """WHAT: `test_normality` on zero-variance data returns (NaN, False), not (1.0, True)."""
  analyzer = StatisticalAnalyzer()
  p_val, is_normal = analyzer.test_normality(np.array([2.0, 2.0, 2.0, 2.0]))
  assert math.isnan(p_val)
  assert is_normal is False


def test_normality_accepts_linear_data_and_rejects_outlier() -> None:
  """WHAT: Shapiro branch on a near-normal ramp and on a heavy outlier."""
  analyzer = StatisticalAnalyzer()
  p_ramp, ramp_is_normal = analyzer.test_normality(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
  p_outlier, outlier_is_normal = analyzer.test_normality(np.array([1.0, 1.0, 1.0, 1.0, 20.0]))

  assert bool(ramp_is_normal) is True
  assert 0.05 <= p_ramp <= 1.0
  assert bool(outlier_is_normal) is False
  assert p_outlier < 0.05


# ============================================================================
# StatisticalAnalyzer.paired_comparison
# ============================================================================


def test_paired_comparison_rejects_mismatched_lengths() -> None:
  """WHAT: Paired comparison rejects unequal-length arrays."""
  analyzer = StatisticalAnalyzer()
  with pytest.raises(ValueError, match="same length"):
    analyzer.paired_comparison(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "L", "m")


def test_paired_comparison_single_sample_short_circuits() -> None:
  """WHAT: n = 1 branch short-circuits cleanly with NaN CIs and p=1.0."""
  analyzer = StatisticalAnalyzer()
  summary = analyzer.paired_comparison(np.array([3.0]), np.array([1.0]), "CPU vs GPU", "runtime_seconds")

  assert summary.sample_size == 1
  assert summary.test_used == "no_test_needed"
  assert summary.mean_a == pytest.approx(3.0)
  assert summary.mean_b == pytest.approx(1.0)
  assert summary.mean_difference == pytest.approx(2.0)
  assert math.isnan(summary.ci_95_difference[0])
  assert math.isnan(summary.ci_95_difference[1])
  assert summary.p_value == 1.0
  assert summary.test_statistic == 0.0
  assert summary.df1 is None
  assert summary.df2 is None
  assert summary.insufficient_power_warning is True


def test_paired_comparison_empty_sample_reports_nan_means() -> None:
  """WHAT: n = 0 branch does not raise and returns NaN means."""
  analyzer = StatisticalAnalyzer()
  summary = analyzer.paired_comparison(np.array([]), np.array([]), "L", "m")

  assert summary.sample_size == 0
  assert math.isnan(summary.mean_a)
  assert math.isnan(summary.mean_b)
  assert summary.test_used == "no_test_needed"


def test_paired_comparison_identical_samples_skip_test_and_preserve_real_std() -> None:
  """WHAT: Identical paired samples short-circuit Wilcoxon while preserving true sample stds."""
  analyzer = StatisticalAnalyzer()
  data = np.array([2.0, 4.0, 6.0])
  summary = analyzer.paired_comparison(data, data, "L", "m")

  assert summary.test_used == "no_test_needed"
  assert summary.test_justification == "identical_samples_no_test_needed"
  assert summary.p_value == 1.0
  assert summary.std_a == pytest.approx(2.0)
  assert summary.std_b == pytest.approx(2.0)
  assert summary.mean_difference == 0.0
  assert summary.effect_size == 0.0
  assert summary.zero_differences_count == 3
  assert summary.effective_sample_size == 0
  assert summary.insufficient_power_warning is True


@pytest.mark.parametrize(
  ("n", "expected_p", "expected_low_power"),
  [
    pytest.param(3, 0.25, True, id="n3_impossible_to_reject_at_005"),
    pytest.param(5, 0.0625, True, id="n5_impossible_to_reject_at_005"),
    pytest.param(6, 0.03125, False, id="n6_exact_rejection_boundary"),
  ],
)
def test_paired_comparison_constant_nonzero_diff_uses_exact_wilcoxon(
  n: int,
  expected_p: float,
  expected_low_power: bool,
) -> None:
  """WHAT: Constant non-zero paired differences run exact Wilcoxon (p = 2^{-(n-1)}), never p = 0.0.

  WHY: Regression test for Bug 1 — the legacy paired t-test zero-variance short-circuit
  returned p=0.0 even at n=3, causing false statistical significance after Holm-Bonferroni.
  """
  analyzer = StatisticalAnalyzer(alpha=0.05)
  a = np.arange(1, n + 1, dtype=float)
  b = a - 1.0

  summary = analyzer.paired_comparison(a, b, "A vs B", "cost")

  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.test_justification == "demsar2006_wilcoxon_default"
  assert summary.mean_difference == pytest.approx(1.0)
  assert summary.p_value == pytest.approx(expected_p, rel=1e-12)
  assert summary.effect_size == pytest.approx(1.0)
  assert summary.zero_differences_count == 0
  assert summary.effective_sample_size == n
  assert summary.insufficient_power_warning is expected_low_power


def test_paired_comparison_normal_path_uses_wilcoxon_and_nonparametric_marginal_cis() -> None:
  """WHAT: Both samples normal -> still uses Wilcoxon and Hodges-Lehmann marginal CIs."""
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  data_b = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

  summary = analyzer.paired_comparison(data_a, data_b, "CPU vs GPU", "runtime_seconds")

  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.test_justification == "demsar2006_wilcoxon_default"
  assert summary.mean_difference == pytest.approx(-11.5)
  assert summary.std_a == pytest.approx(1.8708286933869707)
  assert summary.std_b == pytest.approx(3.7416573867739413)
  assert summary.effect_size == pytest.approx(-1.0)
  assert summary.p_value == pytest.approx(0.03125, rel=1e-12)
  assert summary.test_statistic == pytest.approx(0.0)
  assert summary.df1 is None
  assert summary.df2 is None
  assert summary.get_speedup() == pytest.approx(3.5 / 15.0)
  # Marginal CIs match one-sample Hodges-Lehmann (not parametric Student-t)
  _, expected_ci_a = analyzer.compute_hodges_lehmann(data_a, confidence=0.95)
  _, expected_ci_b = analyzer.compute_hodges_lehmann(data_b, confidence=0.95)
  assert summary.ci_95_a == pytest.approx(expected_ci_a)
  assert summary.ci_95_b == pytest.approx(expected_ci_b)


def test_paired_comparison_normal_samples_n30_uses_wilcoxon_not_t() -> None:
  """WHAT: Two perfectly normal samples with n=30 receive Wilcoxon, proving data-independence."""
  rng = np.random.default_rng(42)
  analyzer = StatisticalAnalyzer()
  data_a = rng.normal(loc=100.0, scale=10.0, size=30)
  data_b = data_a + rng.normal(loc=5.0, scale=2.0, size=30)

  _, is_norm_a = analyzer.test_normality(data_a)
  _, is_norm_b = analyzer.test_normality(data_b)
  assert bool(is_norm_a) and bool(is_norm_b)

  summary = analyzer.paired_comparison(data_a, data_b, "NormA vs NormB", "cost")

  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.test_justification == "demsar2006_wilcoxon_default"
  assert summary.sample_size == 30
  assert summary.ci_95_difference[0] < summary.ci_95_difference[1]


def test_paired_comparison_non_normal_path_uses_wilcoxon() -> None:
  """WHAT: One non-normal sample -> Wilcoxon signed-rank with exact rank-biserial r."""
  analyzer = StatisticalAnalyzer()
  data_a = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
  data_b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

  summary = analyzer.paired_comparison(data_a, data_b, "L", "m")

  assert summary.test_used == "wilcoxon_signed_rank"
  assert summary.mean_difference == pytest.approx(2.8)
  assert summary.effect_size == pytest.approx(-1.0 / 3.0)
  assert 0.0 < summary.p_value <= 1.0


def test_paired_comparison_tie_diagnostics_warns_on_majority_zeros(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: Tie diagnostics warn when more than half of differences are zero."""
  analyzer = StatisticalAnalyzer()
  data_a = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 7.0])
  data_b = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

  with caplog.at_level(logging.WARNING):
    summary = analyzer.paired_comparison(data_a, data_b, "A vs B", "cost")

  assert summary.zero_differences_count == 4
  assert summary.effective_sample_size == 2
  assert summary.insufficient_power_warning is True
  assert any("More than half of paired differences are zero" in r.getMessage() for r in caplog.records)


def test_compute_hodges_lehmann_small_n_guard_and_exact_ci() -> None:
  """WHAT: Hodges-Lehmann returns NaN CI when 2^{-(n-1)} > alpha (n <= 5 at 95%) and exact bounds for n >= 6."""
  analyzer = StatisticalAnalyzer()

  # n = 5 at 95%: maximal possible Walsh-average coverage is 1 - 2^{-4} = 93.75% < 95%
  pt5, (lo5, hi5) = analyzer.compute_hodges_lehmann(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), confidence=0.95)
  assert pt5 == pytest.approx(3.0)
  assert math.isnan(lo5)
  assert math.isnan(hi5)

  # n = 6 at 95%: exact critical rank k = 1 -> [W_(1), W_(21)] = [1.0, 6.0]
  pt6, (lo6, hi6) = analyzer.compute_hodges_lehmann(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), confidence=0.95)
  assert pt6 == pytest.approx(3.5)
  assert lo6 == pytest.approx(1.0)
  assert hi6 == pytest.approx(6.0)

  # n = 10 at 95%: exact critical rank k = 9 (c = 8, 2*P(T+<=8)=0.0488 <= 0.05)
  vals10 = np.arange(1.0, 11.0)
  pt10, (lo10, hi10) = analyzer.compute_hodges_lehmann(vals10, confidence=0.95)
  assert pt10 == pytest.approx(5.5)
  assert 1.0 < lo10 < pt10 < hi10 < 10.0


# ============================================================================
# StatisticalAnalyzer.friedman_test
# ============================================================================


def test_friedman_requires_equal_lengths() -> None:
  """WHAT: Friedman test rejects unpaired data sets."""
  analyzer = StatisticalAnalyzer()
  with pytest.raises(ValueError, match="same number of observations"):
    analyzer.friedman_test([np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0])])


def test_friedman_requires_three_algorithms() -> None:
  """WHAT: Friedman test rejects fewer than three data sets."""
  analyzer = StatisticalAnalyzer()
  with pytest.raises(ValueError, match="at least 3 algorithms"):
    analyzer.friedman_test([np.array([1.0, 2.0]), np.array([2.0, 1.0])])


def test_friedman_constant_data_returns_neutral_result() -> None:
  """WHAT: Degenerate (zero-variance) Friedman input returns neutral result with low_power flag."""
  analyzer = StatisticalAnalyzer()
  constant = np.array([5.0, 5.0, 5.0])
  result = analyzer.friedman_test([constant, constant, constant])

  assert list(result) == EXPECTED_FRIEDMAN_KEYS
  assert result["statistic"] == 0.0
  assert result["p_value"] == 1.0
  assert result["statistic_f"] == 0.0
  assert result["df1"] == 2
  assert result["df2"] == 4
  assert result["p_value_f"] == 1.0
  assert result["significant"] is False
  assert result["post_hoc_required"] is False
  assert result["low_power"] is True


def test_friedman_iman_davenport_f_matches_manual_calculation() -> None:
  """WHAT: Friedman test computes Iman-Davenport F_F matching manual calculation."""
  analyzer = StatisticalAnalyzer(alpha=0.05)
  alg_a = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0])
  alg_b = np.array([2.0, 1.0, 3.0, 1.0, 2.0, 2.0])
  alg_c = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 3.0])

  result = analyzer.friedman_test([alg_a, alg_b, alg_c])

  assert list(result) == EXPECTED_FRIEDMAN_KEYS
  assert result["statistic"] == pytest.approx(7.0, rel=1e-12)
  assert result["statistic_f"] == pytest.approx(7.0, rel=1e-12)
  assert result["df1"] == 2
  assert result["df2"] == 10
  assert result["p_value_f"] == pytest.approx(float(stats.f.sf(7.0, 2, 10)), rel=1e-12)
  assert result["significant"] is True
  assert result["post_hoc_required"] is True
  assert result["low_power"] is False


def test_friedman_significance_uses_iman_davenport_p_value_f() -> None:
  """WHAT: When p_value >= alpha but p_value_f < alpha, significance follows p_value_f."""
  analyzer = StatisticalAnalyzer(alpha=0.05)
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
  assert result["low_power"] is True


def test_friedman_boundary_single_observation_and_perfect_concordance() -> None:
  """WHAT: N < 2 returns neutral F_F=0.0, p_value_f=1.0; perfect concordance returns inf, p_value_f=0.0."""
  analyzer = StatisticalAnalyzer(alpha=0.05)

  res_n1 = analyzer.friedman_test([np.array([1.0]), np.array([2.0]), np.array([3.0])])
  assert res_n1["statistic"] == 0.0
  assert res_n1["statistic_f"] == 0.0
  assert res_n1["df1"] == 2
  assert res_n1["df2"] == 0
  assert res_n1["p_value_f"] == 1.0
  assert res_n1["significant"] is False
  assert res_n1["low_power"] is True

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
  assert res_perf["low_power"] is False
