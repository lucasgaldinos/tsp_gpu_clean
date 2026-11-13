"""Shared statistical utilities for benchmark analysis.

These helpers were originally defined inside
`my_notes/notebooks/results_and_stats_v3.ipynb` and are extracted
here so that multiple notebooks (v1 correction, v2 aggregation, etc.)
can reuse the same implementation.

The goal is to keep behavior identical to the original notebook
functions while providing a small, well-scoped API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

# ============================================================================
# Core regression metrics
# ============================================================================


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate coefficient of determination (R²).

  This matches the implementation used in the v3 notebook:

      SS_res = sum((y_true - y_pred)**2)
      SS_tot = sum((y_true - mean(y_true))**2)
      R2 = 1 - SS_res / SS_tot
  """
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)

  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
  return float(1.0 - (ss_res / ss_tot))


def adjusted_r2(r2: float, n: int, k: int) -> float:
  """Calculate adjusted R² (penalizes model complexity).

  Args:
      r2: Coefficient of determination
      n: Number of observations
      k: Number of predictors (excluding intercept)
  """
  return float(1.0 - (1.0 - r2) * (n - 1) / (n - k - 1))


def calculate_aic_bic(residuals: np.ndarray, k: int, n: int) -> tuple[float, float]:
  """Calculate AIC and BIC for a regression model.

  Args:
      residuals: Model residuals (y_true - y_pred)
      k: Number of parameters (including intercept)
      n: Number of observations

  Returns:
      Tuple of (AIC, BIC)
  """
  residuals = np.asarray(residuals, dtype=float)
  ss_res = np.sum(residuals**2)
  aic = n * np.log(ss_res / n) + 2 * k
  bic = n * np.log(ss_res / n) + k * np.log(n)
  return float(aic), float(bic)


# ============================================================================
# Error metrics
# ============================================================================


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Mean Absolute Error (MAE)."""
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)
  return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Root Mean Squared Error (RMSE)."""
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)
  return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calculate Mean Absolute Percentage Error (MAPE, in %)."""
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)
  return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# ============================================================================
# Leave-One-Out Cross-Validation
# ============================================================================


def loocv_regression(
  baseline_sizes: np.ndarray,
  baseline_times: np.ndarray,
  model_func: Callable[..., np.ndarray],
) -> tuple[float, float, float, np.ndarray]:
  """Perform Leave-One-Out Cross-Validation for a TSP regression model.

  This mirrors the original v3 notebook implementation, using
  ``scipy.optimize.curve_fit`` to refit the provided ``model_func``
  on ``n-1`` points at each iteration and predicting the held-out
  observation.

  Args:
      baseline_sizes: Array of problem sizes (n cities) used for model fitting
      baseline_times: Array of observed execution times (seconds)
      model_func: Model function (e.g., quadratic, quasilinear)

  Returns:
      (MAE, RMSE, MAPE, predictions)
  """
  sizes = np.asarray(baseline_sizes, dtype=float)
  times = np.asarray(baseline_times, dtype=float)

  n = len(sizes)
  predictions = np.zeros(n, dtype=float)

  for i in range(n):
    baseline_mask = np.ones(n, dtype=bool)
    baseline_mask[i] = False

    sizes_fit = sizes[baseline_mask]
    times_fit = times[baseline_mask]
    size_val = sizes[~baseline_mask]

    params, _ = curve_fit(model_func, sizes_fit, times_fit)
    predictions[i] = model_func(size_val, *params)[0]

  mae = mean_absolute_error(times, predictions)
  rmse = root_mean_squared_error(times, predictions)
  mape = mean_absolute_percentage_error(times, predictions)

  return mae, rmse, mape, predictions


def compute_nemenyi_critical_distance(k: int, n: int, alpha: float = 0.05) -> float:
  """Compute Nemenyi Critical Difference (CD) per Demšar (2006, Eq. 8).

  CD = (q_tukey / sqrt(2)) * sqrt(k * (k + 1) / (6 * n))
  where q_tukey is the Studentized range critical value at confidence 1 - alpha
  (BUG-47 / DP12-24).
  """
  if k < 2:
    raise ValueError(f"Nemenyi test requires at least 2 algorithms (k >= 2), got k={k}")
  if n < 1:
    raise ValueError(f"Nemenyi test requires at least 1 observation (n >= 1), got n={n}")
  if not (0.0 < alpha < 1.0):
    raise ValueError(f"Significance level alpha must be strictly in (0.0, 1.0), got alpha={alpha}")

  from scipy.stats import studentized_range

  q_tukey = float(studentized_range.ppf(1.0 - alpha, k, np.inf))
  q_alpha = q_tukey / np.sqrt(2.0)
  return float(q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n)))


def resolve_baseline_algorithm(
  algorithm_names: Sequence[str],
  requested_baseline: str | None = None,
) -> str:
  """Determine baseline algorithm for speedup calculations (DP12-9 / DP12-28 / BUG-55).

  If `requested_baseline` is provided and present in `algorithm_names`, returns it.
  If `requested_baseline` is provided but absent, logs a warning and falls back to
  the first algorithm in `algorithm_names`.
  If `requested_baseline` is None, returns the first algorithm in `algorithm_names`.
  If `algorithm_names` is empty, returns "Unknown".
  """
  if requested_baseline is not None:
    if requested_baseline in algorithm_names:
      return requested_baseline
    import logging

    fallback = algorithm_names[0] if algorithm_names else "Unknown"
    logging.warning(
      "Requested baseline_algorithm '%s' not found in algorithm names (%s); falling back to '%s'.",
      requested_baseline,
      list(algorithm_names),
      fallback,
    )
    return fallback
  return algorithm_names[0] if algorithm_names else "Unknown"



