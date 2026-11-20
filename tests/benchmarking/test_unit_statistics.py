"""Unit tests for the statistics facade and the `StatisticalSummary` helpers.

WHAT: Covers the `benchmarking.analysis` package facade and the
`StatisticalSummary` helper methods (speedup, bootstrap CI, significance).
WHY: Notebooks import from the package root and the report tables quote these
helpers; a broken re-export or a wrong boundary value silently corrupts them.
EXPECTED: Every exported name resolves and the helpers return the exact
documented values, including at the significance boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

import benchmarking.analysis as analysis_facade
from benchmarking.analysis.aggregate_statistics import compare_algorithms
from benchmarking.analysis.cd_diagram import compute_cd_diagram_data
from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary
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
    test_used="paired_t_test",
    normality_p_value_a=0.5,
    normality_p_value_b=0.5,
    test_statistic=0.0,
    df1=None,
    df2=None,
  )


def test_statistical_summary_requires_test_statistic_and_dfs() -> None:
  """WHAT: StatisticalSummary requires test_statistic, df1, and df2 explicitly.

  WHY: Task 13.10 (BUG-62 / DP13-3) — Omitting test_statistic or degrees of freedom
  must fail at construction time so no summary silently lacks reporting metadata.
  """
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


# ============================================================================
# Package facade
# ============================================================================


def test_analysis_facade_reexports_public_api() -> None:
  """WHAT: Every name in `benchmarking.analysis.__all__` resolves on the package.

  WHY: Notebooks import from the package root; a broken re-export breaks them at
  import time, not at use time.
  EXPECTED: all names exist and the key classes are the same objects.
  """
  # Arrange / Act / Assert
  for name in analysis_facade.__all__:
    assert hasattr(analysis_facade, name), f"facade is missing exported name {name!r}"
  assert analysis_facade.StatisticalAnalyzer is StatisticalAnalyzer
  assert analysis_facade.StatisticalSummary is StatisticalSummary
  assert analysis_facade.compare_algorithms is compare_algorithms


# ============================================================================
# StatisticalSummary helpers
# ============================================================================


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
  """WHAT: `StatisticalSummary.get_speedup` = mean_a / mean_b with a zero guard.

  WHY: Speedup is the headline metric for CPU/GPU comparisons.
  EXPECTED: 2.0, 1.0, and 0.0 whenever the denominator is zero.
  """
  # Arrange
  summary = _summary(mean_a=mean_a, mean_b=mean_b)

  # Act / Assert
  assert summary.get_speedup() == pytest.approx(expected)


def test_summary_speedup_ci_uses_percentiles() -> None:
  """WHAT: 95% speedup CI from bootstrap samples.

  WHY: The CI is quoted next to the speedup; percentile bounds must be 2.5/97.5.
  EXPECTED: `arange(101)` -> (2.5, 97.5).
  """
  # Arrange
  summary = _summary()
  samples = np.arange(101, dtype=float)

  # Act
  low, high = summary.get_speedup_ci_95(samples)

  # Assert
  assert low == pytest.approx(2.5)
  assert high == pytest.approx(97.5)


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
  """WHAT: `is_significant` boundary behavior.

  WHY: Whether p == alpha counts as significant changes borderline claims in the
  results chapter.
  EXPECTED: True only when p < alpha (strict).
  """
  # Arrange
  summary = _summary(p_value=p_value)

  # Act / Assert
  assert summary.is_significant(alpha) is expected


def test_nemenyi_critical_distance_parity_with_cd_diagram_and_significance() -> None:
  """WHAT: Tests parity between StatisticalAnalyzer.nemenyi_posthoc and cd_diagram.

  WHY: BUG-47 / DP12-24 — Demšar (2006, Eq. 8) defines the Nemenyi critical distance
  with q_alpha = q_tukey / sqrt(2). Both statistical post-hoc and CD diagram generation
  must yield identical critical difference values and consistent significance decisions:
  |R_i - R_j| > CD <=> p_val < alpha.
  """
  np.random.seed(42)
  n_instances = 38
  k_algorithms = 4
  alg_names = ["AlgA", "AlgB", "AlgC", "AlgD"]
  base_scores = np.tile([100.0, 150.0, 250.0, 400.0], (n_instances, 1))
  noise = np.random.normal(0, 5.0, size=(n_instances, k_algorithms))
  results_matrix = base_scores + noise

  data = [results_matrix[:, j] for j in range(k_algorithms)]

  analyzer = StatisticalAnalyzer(alpha=0.05)
  nemenyi_res = analyzer.nemenyi_posthoc(data, algorithm_names=alg_names)
  cd_res = compute_cd_diagram_data(results_matrix, alg_names, alpha=0.05)

  assert nemenyi_res["critical_distance"] == pytest.approx(cd_res.critical_difference, rel=1e-6)

  ranks = np.apply_along_axis(rankdata, 1, results_matrix)
  mean_ranks = np.mean(ranks, axis=0)
  cd = nemenyi_res["critical_distance"]
  p_mat = nemenyi_res["p_values_matrix"]

  for i in range(k_algorithms):
    for j in range(i + 1, k_algorithms):
      rank_diff = abs(mean_ranks[i] - mean_ranks[j])
      p_val = p_mat[i, j]
      assert (rank_diff > cd) == (p_val < 0.05)


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
  """WHAT: Boundary validation in compute_nemenyi_critical_distance.

  WHY: Review Gate 22c181dd — Invalid k, n, or alpha must raise ValueError instead of
  emitting RuntimeWarning (sqrt invalid), NaN, or ZeroDivisionError.
  """
  with pytest.raises(ValueError, match=match):
    compute_nemenyi_critical_distance(k, n, alpha)


def test_nemenyi_posthoc_guards() -> None:
  """WHAT: Boundary validation in StatisticalAnalyzer.nemenyi_posthoc.

  WHY: Review Gate 22c181dd — Ensure k < 2 and empty observations raise ValueError.
  """
  analyzer = StatisticalAnalyzer()
  # k = 1
  with pytest.raises(ValueError, match="at least 2 algorithms"):
    analyzer.nemenyi_posthoc([np.array([1.0, 2.0])])

  # n = 0
  with pytest.raises(ValueError, match="at least 1 observation"):
    analyzer.nemenyi_posthoc([np.array([]), np.array([])])


