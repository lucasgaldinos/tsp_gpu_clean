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

import benchmarking.analysis as analysis_facade
from benchmarking.analysis.aggregate_statistics import compare_algorithms
from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary


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
    is_normal=True,
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
