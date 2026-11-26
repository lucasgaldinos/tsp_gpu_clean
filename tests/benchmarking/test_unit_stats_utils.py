"""Unit tests for the shared statistical helpers in `benchmarking.analysis.stats_utils`.

WHAT: Exercises `finite_metric`, `geometric_mean_ratios`, `resolve_baseline_algorithm`,
`holm_bonferroni_correction`, and `compute_nemenyi_critical_distance`.
WHY: Every cross-problem statistic and report table depends on these low-level primitives;
silent coercion, threshold, or scaling regressions would corrupt downstream inference.
EXPECTED: Exact closed-form values, bool/NaN/Inf rejection, step-down FWER termination,
and strict parameter guards.
"""

from __future__ import annotations

import logging
import math

import pytest

from benchmarking.analysis.stats_utils import (
  compute_nemenyi_critical_distance,
  finite_metric,
  geometric_mean_ratios,
  holm_bonferroni_correction,
  resolve_baseline_algorithm,
)


@pytest.mark.parametrize(
  ("raw", "expected"),
  [
    pytest.param(3.5, 3.5, id="float"),
    pytest.param(7, 7.0, id="int"),
    pytest.param("4.25", 4.25, id="numeric_string"),
    pytest.param(None, None, id="none"),
    pytest.param(True, None, id="bool_true"),
    pytest.param(False, None, id="bool_false"),
    pytest.param(float("nan"), None, id="nan"),
    pytest.param(float("inf"), None, id="pos_inf"),
    pytest.param(float("-inf"), None, id="neg_inf"),
    pytest.param("abc", None, id="bad_string"),
  ],
)
def test_finite_metric_coercion(raw: object, expected: float | None) -> None:
  """WHAT: `finite_metric` accepts only finite non-boolean numeric values."""
  result = finite_metric(raw)
  if expected is None:
    assert result is None
  else:
    assert result == pytest.approx(expected)


def test_geometric_mean_ratios_exact_and_filtering() -> None:
  """WHAT: Fleming & Wallace geometric mean skips non-positive/non-finite entries."""
  assert geometric_mean_ratios([]) is None
  assert geometric_mean_ratios([0.0, -3.0, float("nan"), float("inf")]) is None
  assert geometric_mean_ratios([2.0, 8.0]) == pytest.approx(4.0, rel=1e-12)
  assert geometric_mean_ratios([1.0, 4.0, 16.0]) == pytest.approx(4.0, rel=1e-12)
  assert geometric_mean_ratios([0.0, 2.0, 8.0, float("nan")]) == pytest.approx(4.0, rel=1e-12)


def test_resolve_baseline_algorithm_behavior(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: Baseline resolution picks explicit match, first item by default, or warns on fallback."""
  assert resolve_baseline_algorithm(["A", "B"], "B") == "B"
  assert resolve_baseline_algorithm(["A", "B"], None) == "A"
  assert resolve_baseline_algorithm([], None) == "Unknown"

  with caplog.at_level(logging.WARNING):
    resolved = resolve_baseline_algorithm(["A", "B"], "Missing")
  assert resolved == "A"
  assert any("Missing" in r.getMessage() for r in caplog.records)


def test_holm_bonferroni_step_down_and_ordering() -> None:
  """WHAT: Step-down Holm-Bonferroni preserves input order and enforces monotonicity."""
  assert holm_bonferroni_correction([]) == ([], [])

  reject, p_adj = holm_bonferroni_correction([0.03, 0.04], alpha=0.05)
  assert reject == [False, False]
  assert p_adj == pytest.approx([0.06, 0.06])

  reject_u, p_adj_u = holm_bonferroni_correction([0.04, 0.01], alpha=0.05)
  assert reject_u == [True, True]
  assert p_adj_u == pytest.approx([0.04, 0.02])

  reject_strict, p_adj_strict = holm_bonferroni_correction([0.02, 0.04], alpha=0.01)
  assert reject_strict == [False, False]
  assert p_adj_strict == pytest.approx([0.04, 0.04])


def test_compute_nemenyi_critical_distance_reference_and_guards() -> None:
  """WHAT: Exact Studentized-range Nemenyi CD matches Demšar (2006) reference values."""
  cd = compute_nemenyi_critical_distance(k=4, n=38, alpha=0.05)
  assert math.isfinite(cd)
  assert cd == pytest.approx(0.7609, abs=1e-3)

  with pytest.raises(ValueError, match="at least 2 algorithms"):
    compute_nemenyi_critical_distance(1, 10, 0.05)
  with pytest.raises(ValueError, match="at least 1 observation"):
    compute_nemenyi_critical_distance(4, 0, 0.05)
  with pytest.raises(ValueError, match=r"strictly in \(0.0, 1.0\)"):
    compute_nemenyi_critical_distance(4, 10, 1.0)
