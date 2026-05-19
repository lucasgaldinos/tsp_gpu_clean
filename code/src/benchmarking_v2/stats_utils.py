"""Shared statistical utilities for benchmark analysis.

These helpers were originally defined inside
`my_notes/notebooks/results_and_stats_v3.ipynb` and are extracted
here so that multiple notebooks (v1 correction, v2 aggregation, etc.)
can reuse the same implementation.

The goal is to keep behavior identical to the original notebook
functions while providing a small, well‑scoped API.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import curve_fit


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
    return 1.0 - (ss_res / ss_tot)


def adjusted_r2(r2: float, n: int, k: int) -> float:
    """Calculate adjusted R² (penalizes model complexity).

    Args:
        r2: Coefficient of determination
        n: Number of observations
        k: Number of predictors (excluding intercept)
    """

    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


def calculate_aic_bic(residuals: np.ndarray, k: int, n: int) -> Tuple[float, float]:
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
    return aic, bic


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
) -> Tuple[float, float, float, np.ndarray]:
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
        # Create baseline/validation split (leave one out)
        baseline_mask = np.ones(n, dtype=bool)
        baseline_mask[i] = False

        sizes_fit = sizes[baseline_mask]
        times_fit = times[baseline_mask]
        size_val = sizes[~baseline_mask]

        # Fit model on baseline data
        params, _ = curve_fit(model_func, sizes_fit, times_fit)

        # Predict on validation point
        predictions[i] = model_func(size_val, *params)[0]

    # Calculate metrics
    mae = mean_absolute_error(times, predictions)
    rmse = root_mean_squared_error(times, predictions)
    mape = mean_absolute_percentage_error(times, predictions)

    return mae, rmse, mape, predictions
