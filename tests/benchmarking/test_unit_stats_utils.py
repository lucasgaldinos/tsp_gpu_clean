"""Unit tests for the shared statistical helpers in `benchmarking.analysis.stats_utils`.

WHAT: Exercises the regression metrics (`calculate_r2`, `adjusted_r2`,
`calculate_aic_bic`), the error metrics (MAE, RMSE, MAPE) and `loocv_regression`
on small hand-computed inputs.
WHY: These helpers feed the reported benchmark tables; a silent formula change
would alter published numbers without any other test failing.
EXPECTED: Exact closed-form values (floats via `pytest.approx`), documented
empty/single-element boundary behavior, and `ValueError` on mismatched lengths.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarking.analysis.stats_utils import (
  adjusted_r2,
  calculate_aic_bic,
  calculate_r2,
  loocv_regression,
  mean_absolute_error,
  mean_absolute_percentage_error,
  root_mean_squared_error,
)

if TYPE_CHECKING:
  from collections.abc import Callable


def _quiet(compute: Callable[[], float]) -> float:
  """Run `compute` while silencing numpy's expected empty-slice/singular warnings.

  WHAT: Wraps a numpy computation that legitimately emits `RuntimeWarning`
  (mean of an empty slice, 0/0) and returns its result.
  WHY: Empty-input boundary tests assert the produced value, so the warning noise
  would otherwise fail lint-clean runs or obscure real warnings.
  EXPECTED: The same return value, no warning raised.
  """
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    return compute()


def _linear_model(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
  """Linear model `slope * x + intercept` used as the LOOCV fit function."""
  return slope * x + intercept


# ============================================================================
# calculate_r2
# ============================================================================


@pytest.mark.parametrize(
  ("y_true", "y_pred", "expected"),
  [
    pytest.param([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1.0, id="perfect_fit"),
    pytest.param([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], -0.2, id="always_predict_mean"),
    pytest.param([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], 0.5, id="one_residual"),
  ],
)
def test_calculate_r2_exact_values(y_true: list[float], y_pred: list[float], expected: float) -> None:
  """WHAT: R² on inputs small enough to compute by hand.

  WHY: R² is quoted in the TCC results chapter; the formula must match the
  documented `1 - SS_res/SS_tot` definition.
  EXPECTED: perfect fit -> 1.0; predicting the mean -> 1 - 6/5 = -0.2;
  single residual of 1 with SS_tot 2 -> 0.5.
  """
  # Arrange
  true_array = np.array(y_true)
  pred_array = np.array(y_pred)

  # Act
  result = calculate_r2(true_array, pred_array)

  # Assert
  assert result == pytest.approx(expected, rel=1e-12), f"R² mismatch for {y_true} vs {y_pred}"


def test_calculate_r2_single_observation_is_undefined() -> None:
  """WHAT: R² with a single point (SS_tot == 0).

  WHY: The statistic is mathematically undefined for constant `y_true`; callers
  must not mistake the value for a valid score.
  EXPECTED: NaN (0/0), not 1.0 and not an exception.
  """
  # Arrange / Act
  result = _quiet(lambda: calculate_r2(np.array([5.0]), np.array([5.0])))

  # Assert
  assert math.isnan(result), "R² of a single observation must be NaN where SS_tot is zero"


def test_calculate_r2_empty_input_is_nan() -> None:
  """WHAT: R² over empty arrays.

  WHY: Empty aggregation can reach the reporting layer; it must not crash or
  silently return a finite score.
  EXPECTED: NaN.
  """
  # Arrange / Act
  result = _quiet(lambda: calculate_r2(np.array([]), np.array([])))

  # Assert
  assert math.isnan(result), "R² of empty input must be NaN"


# ============================================================================
# adjusted_r2
# ============================================================================


@pytest.mark.parametrize(
  ("r2", "n", "k", "expected"),
  [
    pytest.param(0.9, 10, 2, 0.8714285714285714, id="penalised"),
    pytest.param(1.0, 10, 2, 1.0, id="perfect_stays_one"),
    pytest.param(0.5, 20, 0, 0.5, id="no_predictors"),
  ],
)
def test_adjusted_r2_exact_values(r2: float, n: int, k: int, expected: float) -> None:
  """WHAT: Adjusted R² formula `1 - (1-R²)(n-1)/(n-k-1)`.

  WHY: The penalty term is easy to invert; the published tables depend on it.
  EXPECTED: 1 - 0.1*9/7 for (0.9, 10, 2); identity when R²=1 or k=0.
  """
  # Arrange / Act
  result = adjusted_r2(r2, n, k)

  # Assert
  assert result == pytest.approx(expected, rel=1e-12)


def test_adjusted_r2_with_zero_degrees_of_freedom_raises() -> None:
  """WHAT: adjusted_r2 when n == k + 1 (no residual degrees of freedom).

  WHY: The denominator becomes zero; callers must get a loud error instead of an
  "inf" score in a report.
  EXPECTED: `ZeroDivisionError`.
  """
  # Arrange / Act / Assert
  with pytest.raises(ZeroDivisionError):
    adjusted_r2(0.9, 4, 3)


# ============================================================================
# calculate_aic_bic
# ============================================================================


@pytest.mark.parametrize(
  ("residuals", "k", "n", "expected_aic", "expected_bic"),
  [
    pytest.param([1.0, -1.0, 1.0, -1.0], 2, 4, 4.0, 2.772588722239781, id="unit_residuals"),
    pytest.param([1.0, 2.0, 3.0], 1, 3, 6.621335122841447, 5.719947411509557, id="ss_res_14"),
  ],
)
def test_calculate_aic_bic_exact_values(
  residuals: list[float],
  k: int,
  n: int,
  expected_aic: float,
  expected_bic: float,
) -> None:
  """WHAT: AIC/BIC from residuals.

  WHY: Model-selection tables depend on `n*ln(SS_res/n) + penalty`; an off-by-one
  in `k` changes the reported winner.
  EXPECTED: [1,-1,1,-1] -> SS_res=4, AIC=4, BIC=2*ln(4); [1,2,3] -> AIC and BIC
  with SS_res=14.
  """
  # Arrange / Act
  aic, bic = calculate_aic_bic(np.array(residuals), k, n)

  # Assert
  assert aic == pytest.approx(expected_aic, rel=1e-12)
  assert bic == pytest.approx(expected_bic, rel=1e-12)


def test_calculate_aic_bic_empty_input_is_nan() -> None:
  """WHAT: AIC/BIC with zero observations.

  WHY: n=0 makes SS_res/n undefined; the helper must not fabricate a finite value.
  EXPECTED: both values NaN.
  """
  # Arrange / Act
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    aic, bic = calculate_aic_bic(np.array([]), 0, 0)

  # Assert
  assert math.isnan(aic)
  assert math.isnan(bic)


# ============================================================================
# Error metrics: MAE / RMSE / MAPE
# ============================================================================


@pytest.mark.parametrize(
  ("y_true", "y_pred", "expected"),
  [
    pytest.param([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 2.0 / 3.0, id="two_thirds"),
    pytest.param([5.0], [3.0], 2.0, id="single_element"),
  ],
)
def test_mean_absolute_error_exact_values(y_true: list[float], y_pred: list[float], expected: float) -> None:
  """WHAT: MAE on hand-computed inputs.

  WHY: MAE is reported alongside R² in the notebooks; the absolute value (not
  squared) must be used.
  EXPECTED: mean(|diff|) = 2/3 and 2.0.
  """
  # Arrange / Act
  result = mean_absolute_error(np.array(y_true), np.array(y_pred))

  # Assert
  assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
  ("y_true", "y_pred", "expected"),
  [
    pytest.param([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 0.816496580927726, id="sqrt_two_thirds"),
    pytest.param([5.0], [3.0], 2.0, id="single_element"),
  ],
)
def test_root_mean_squared_error_exact_values(y_true: list[float], y_pred: list[float], expected: float) -> None:
  """WHAT: RMSE on hand-computed inputs.

  WHY: RMSE weights large residuals differently from MAE; both are published.
  EXPECTED: sqrt((1+0+1)/3) = sqrt(2/3) and 2.0 for a single pair.
  """
  # Arrange / Act
  result = root_mean_squared_error(np.array(y_true), np.array(y_pred))

  # Assert
  assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
  ("y_true", "y_pred", "expected"),
  [
    pytest.param([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 44.44444444444444, id="percent_scale"),
    pytest.param([5.0], [3.0], 40.0, id="single_element"),
  ],
)
def test_mean_absolute_percentage_error_is_percent(
  y_true: list[float],
  y_pred: list[float],
  expected: float,
) -> None:
  """WHAT: MAPE on hand-computed inputs.

  WHY: The helper documents a percent result; a factor-of-100 regression would
  silently rescale every reported gap.
  EXPECTED: mean(|diff/y_true|) * 100 = 44.44...% and 40%.
  """
  # Arrange / Act
  result = mean_absolute_percentage_error(np.array(y_true), np.array(y_pred))

  # Assert
  assert result == pytest.approx(expected, rel=1e-12)


def test_mean_absolute_percentage_error_zero_actual_is_inf() -> None:
  """WHAT: MAPE when a true value is zero.

  WHY: Division by zero is reachable with synthetic data; the helper documents
  no guard, so the behavior must be pinned.
  EXPECTED: +inf (numpy divide-by-zero), not an exception.
  """
  # Arrange / Act
  result = _quiet(lambda: mean_absolute_percentage_error(np.array([0.0, 2.0]), np.array([1.0, 2.0])))

  # Assert
  assert math.isinf(result)


@pytest.mark.parametrize(
  "metric",
  [
    pytest.param(calculate_r2, id="calculate_r2"),
    pytest.param(mean_absolute_error, id="mean_absolute_error"),
    pytest.param(root_mean_squared_error, id="root_mean_squared_error"),
    pytest.param(mean_absolute_percentage_error, id="mean_absolute_percentage_error"),
  ],
)
def test_pairwise_metrics_reject_mismatched_lengths(metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
  """WHAT: Pairwise metrics with non-broadcastable lengths (2 vs 3).

  WHY: Silent per-element pairing of different runs would mix repetitions from
  different sample sizes into one score.
  EXPECTED: numpy `ValueError` mentioning broadcasting.
  """
  # Arrange
  y_true = np.array([1.0, 2.0])
  y_pred = np.array([1.0, 2.0, 3.0])

  # Act / Assert
  with pytest.raises(ValueError, match="broadcast"):
    metric(y_true, y_pred)


@pytest.mark.parametrize(
  "metric",
  [
    pytest.param(calculate_r2, id="calculate_r2"),
    pytest.param(mean_absolute_error, id="mean_absolute_error"),
    pytest.param(root_mean_squared_error, id="root_mean_squared_error"),
    pytest.param(mean_absolute_percentage_error, id="mean_absolute_percentage_error"),
  ],
)
def test_pairwise_metrics_empty_input_is_nan(metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
  """WHAT: Pairwise metrics over empty arrays.

  WHY: An empty aggregation stage must propagate "no data" instead of a zero
  that reads as a perfect score.
  EXPECTED: NaN for every metric.
  """
  # Arrange / Act
  result = _quiet(lambda: metric(np.array([]), np.array([])))

  # Assert
  assert math.isnan(result), f"{metric.__name__} of empty input must be NaN"


# ============================================================================
# loocv_regression
# ============================================================================


def test_loocv_regression_recovers_exact_linear_relation() -> None:
  """WHAT: Leave-one-out CV on data that lies exactly on a line.

  WHY: With a perfect line, every held-out point must be predicted exactly; this
  pins the indexing (fit on n-1, predict the excluded point) and the returned
  metric tuple order (MAE, RMSE, MAPE, predictions).
  EXPECTED: zero error metrics, predictions equal to the observed times in
  input order.
  """
  # Arrange
  sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  times = 2.0 * sizes

  # Act: a perfect line has an unestimable parameter covariance; the
  # OptimizeWarning is expected and unrelated to the index/metric contract.
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message="Covariance of the parameters could not be estimated",
    )
    mae, rmse, mape, predictions = loocv_regression(sizes, times, _linear_model)

  # Assert
  assert predictions.shape == (5,), "predictions must keep one entry per observation"
  assert predictions == pytest.approx(times, rel=1e-8, abs=1e-8)
  assert mae == pytest.approx(0.0, abs=1e-8)
  assert rmse == pytest.approx(0.0, abs=1e-8)
  assert mape == pytest.approx(0.0, abs=1e-8)
