"""Unit tests for the analysis/reporting facades and `StatisticalSummary` / Nemenyi primitives.

WHAT: Covers `benchmarking.analysis` and `benchmarking.reporting` package facades,
`StatisticalSummary` helpers, `interpret_rank_biserial` correlation-scale brackets,
and parity between `StatisticalAnalyzer.nemenyi_posthoc` and `run_friedman_nemenyi_analysis`.
WHY: Prevents broken re-exports, mislabeled rank-biserial effect sizes, and Nemenyi CD drift.
EXPECTED: Every exported name resolves cleanly, rank-biserial brackets follow Cohen (1988) /
Kerby (2014) cutoffs (0.1 / 0.3 / 0.5), and Nemenyi parity holds with tie/orientation safety.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

import benchmarking.analysis as analysis_facade
import benchmarking.reporting as reporting_facade
from benchmarking.analysis.cohort_analysis import compare_algorithms
from benchmarking.analysis.statistics import (
  OmnibusTestResult,
  StatisticalAnalyzer,
  StatisticalSummary,
  interpret_rank_biserial,
  run_friedman_nemenyi_analysis,
)
from benchmarking.analysis.stats_utils import compute_nemenyi_critical_distance


def _summary(*, mean_a: float = 4.0, mean_b: float = 2.0, p_value: float = 0.04) -> StatisticalSummary:
  """Build a `StatisticalSummary` with only the fields a test cares about varied."""
  return StatisticalSummary(
    comparison_label="A vs B",
    metric_name="runtime_seconds",
    sample_size=10,
    mean_a=mean_a,
    mean_b=mean_b,
    std_a=0.1,
    std_b=0.1,
    ci_95_a=(mean_a, mean_a),
    ci_95_b=(mean_b, mean_b),
    mean_difference=mean_a - mean_b,
    ci_95_difference=(mean_a - mean_b, mean_a - mean_b),
    p_value=p_value,
    effect_size=0.0,
    test_used="wilcoxon_signed_rank",
    normality_p_value_a=0.5,
    normality_p_value_b=0.5,
    test_statistic=0.0,
    df1=None,
    df2=None,
  )


def test_statistical_summary_requires_test_statistic_and_dfs() -> None:
  """WHAT: StatisticalSummary requires test_statistic, df1, and df2 explicitly."""
  with pytest.raises(TypeError):
    StatisticalSummary(  # type: ignore[call-arg]
      comparison_label="A vs B",
      metric_name="cost",
      sample_size=10,
      mean_a=1.0,
      mean_b=2.0,
      std_a=0.1,
      std_b=0.1,
      ci_95_a=(0.9, 1.1),
      ci_95_b=(1.9, 2.1),
      mean_difference=-1.0,
      ci_95_difference=(-1.1, -0.9),
      p_value=0.05,
      effect_size=0.0,
      test_used="wilcoxon_signed_rank",
      normality_p_value_a=0.5,
      normality_p_value_b=0.5,
    )


def test_analysis_and_reporting_facades_reexport_public_api() -> None:
  """WHAT: Every name in `benchmarking.analysis.__all__` and `benchmarking.reporting.__all__` resolves."""
  for name in analysis_facade.__all__:
    assert hasattr(analysis_facade, name), f"analysis facade is missing exported name {name!r}"
  for name in reporting_facade.__all__:
    assert hasattr(reporting_facade, name), f"reporting facade is missing exported name {name!r}"

  assert analysis_facade.StatisticalAnalyzer is StatisticalAnalyzer
  assert analysis_facade.StatisticalSummary is StatisticalSummary
  assert analysis_facade.compare_algorithms is compare_algorithms
  assert analysis_facade.OmnibusTestResult is OmnibusTestResult


@pytest.mark.parametrize(
  ("r_val", "expected"),
  [
    pytest.param(0.0, "negligible", id="zero"),
    pytest.param(-0.09, "negligible", id="below_0_1"),
    pytest.param(0.10, "small", id="at_0_1"),
    pytest.param(0.29, "small", id="below_0_3"),
    pytest.param(-0.30, "medium", id="at_0_3"),
    pytest.param(0.49, "medium", id="below_0_5_not_small"),
    pytest.param(0.50, "large", id="at_0_5"),
    pytest.param(-0.79, "large", id="at_0_79_not_medium"),
    pytest.param(1.0, "large", id="perfect_dominance"),
    pytest.param(float("nan"), "undefined", id="nan"),
  ],
)
def test_interpret_rank_biserial_uses_correlation_scale(r_val: float, expected: str) -> None:
  """WHAT: Rank-biserial classification uses correlation cutoffs (0.1/0.3/0.5), not Cohen's d (0.2/0.5/0.8)."""
  assert interpret_rank_biserial(r_val) == expected


@pytest.mark.parametrize(
  ("mean_a", "mean_b", "expected"),
  [
    pytest.param(4.0, 2.0, 2.0, id="two_x_speedup"),
    pytest.param(5.0, 5.0, 1.0, id="no_speedup"),
    pytest.param(3.0, 0.0, 0.0, id="zero_denominator"),
    pytest.param(0.0, 4.0, 0.0, id="zero_numerator"),
  ],
)
def test_summary_speedup(mean_a: float, mean_b: float, expected: float) -> None:
  """WHAT: `StatisticalSummary.get_speedup` = mean_a / mean_b with a zero guard."""
  summary = _summary(mean_a=mean_a, mean_b=mean_b)
  assert summary.get_speedup() == pytest.approx(expected)


@pytest.mark.parametrize(
  ("p_value", "alpha", "expected"),
  [
    pytest.param(0.049, 0.05, True, id="below_alpha"),
    pytest.param(0.05, 0.05, False, id="equal_is_not_significant"),
    pytest.param(0.0001, 0.001, True, id="far_below"),
    pytest.param(0.2, 0.05, False, id="above_alpha"),
  ],
)
def test_summary_is_significant_uses_strict_inequality(p_value: float, alpha: float, expected: bool) -> None:
  """WHAT: `is_significant` boundary behavior uses strict p < alpha."""
  summary = _summary(p_value=p_value)
  assert summary.is_significant(alpha) is expected


def test_nemenyi_critical_distance_parity_with_omnibus_analysis() -> None:
  """WHAT: Parity between StatisticalAnalyzer.nemenyi_posthoc and run_friedman_nemenyi_analysis."""
  rng = np.random.default_rng(42)
  n_instances = 38
  k_algorithms = 4
  alg_names = ["AlgA", "AlgB", "AlgC", "AlgD"]
  base_scores = np.tile([100.0, 150.0, 250.0, 400.0], (n_instances, 1))
  results_matrix = base_scores + rng.normal(0, 5.0, size=(n_instances, k_algorithms))

  data = [results_matrix[:, j] for j in range(k_algorithms)]

  analyzer = StatisticalAnalyzer(alpha=0.05)
  nemenyi_res = analyzer.nemenyi_posthoc(data, algorithm_names=alg_names)
  omnibus = run_friedman_nemenyi_analysis(results_matrix, alg_names, alpha=0.05)

  assert nemenyi_res["critical_distance"] == pytest.approx(omnibus.critical_difference, rel=1e-6)

  ranks = np.apply_along_axis(rankdata, 1, results_matrix)
  mean_ranks = np.mean(ranks, axis=0)
  cd = nemenyi_res["critical_distance"]
  p_mat = nemenyi_res["p_values_matrix"]

  for i in range(k_algorithms):
    for j in range(i + 1, k_algorithms):
      rank_diff = abs(mean_ranks[i] - mean_ranks[j])
      p_val = p_mat[i, j]
      assert (rank_diff > cd) == (p_val < 0.05)


def test_nemenyi_posthoc_higher_is_better_and_all_tied_guard() -> None:
  """WHAT: `nemenyi_posthoc` handles all-tied blocks cleanly without NaN and supports higher_is_better."""
  analyzer = StatisticalAnalyzer(alpha=0.05)

  tied = [np.array([5.0, 5.0, 5.0, 5.0]) for _ in range(3)]
  res_tied = analyzer.nemenyi_posthoc(tied, ["A", "B", "C"])
  assert np.all(np.isfinite(res_tied["p_values_matrix"]))
  assert np.allclose(res_tied["p_values_matrix"], np.ones((3, 3)))
  assert res_tied["significant_pairs"] == []

  # Higher-is-better produces identical pairwise p-values to negated lower-is-better
  cols = [
    np.array([1.0, 2.0, 1.5, 1.0, 2.0]),
    np.array([5.0, 5.5, 5.2, 5.1, 5.3]),
    np.array([10.0, 11.0, 10.5, 9.8, 10.2]),
  ]
  res_high = analyzer.nemenyi_posthoc(cols, ["A", "B", "C"], higher_is_better=True)
  res_neg = analyzer.nemenyi_posthoc([-c for c in cols], ["A", "B", "C"], higher_is_better=False)
  assert np.allclose(res_high["p_values_matrix"], res_neg["p_values_matrix"])


@pytest.mark.parametrize(
  ("k", "n", "alpha", "match"),
  [
    pytest.param(1, 10, 0.05, "at least 2 algorithms", id="k_less_than_2"),
    pytest.param(0, 10, 0.05, "at least 2 algorithms", id="k_zero"),
    pytest.param(4, 0, 0.05, "at least 1 observation", id="n_zero"),
    pytest.param(4, -1, 0.05, "at least 1 observation", id="n_negative"),
    pytest.param(4, 10, 0.0, r"strictly in \(0.0, 1.0\)", id="alpha_zero"),
    pytest.param(4, 10, 1.0, r"strictly in \(0.0, 1.0\)", id="alpha_one"),
    pytest.param(4, 10, -0.05, r"strictly in \(0.0, 1.0\)", id="alpha_negative"),
  ],
)
def test_compute_nemenyi_critical_distance_guards(k: int, n: int, alpha: float, match: str) -> None:
  """WHAT: Boundary validation in compute_nemenyi_critical_distance."""
  with pytest.raises(ValueError, match=match):
    compute_nemenyi_critical_distance(k, n, alpha)


def test_nemenyi_posthoc_guards() -> None:
  """WHAT: Boundary validation in StatisticalAnalyzer.nemenyi_posthoc."""
  analyzer = StatisticalAnalyzer()
  with pytest.raises(ValueError, match="at least 2 algorithms"):
    analyzer.nemenyi_posthoc([np.array([1.0, 2.0])])
  with pytest.raises(ValueError, match="at least 1 observation"):
    analyzer.nemenyi_posthoc([np.array([]), np.array([])])
