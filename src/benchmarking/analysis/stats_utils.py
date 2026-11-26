"""Shared low-level statistical utilities for benchmark analysis.

Provides pure mathematical helpers reused across ``benchmarking.analysis``:
- Finite numeric coercion (`finite_metric`)
- Fleming & Wallace (1986) geometric mean over positive ratios (`geometric_mean_ratios`)
- Baseline algorithm resolution (`resolve_baseline_algorithm`)
- Step-down Holm-Bonferroni FWER adjustment (`holm_bonferroni_correction`)
- Exact Studentized-range Nemenyi Critical Difference (`compute_nemenyi_critical_distance`)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Sequence


def compute_nemenyi_critical_distance(k: int, n: int, alpha: float = 0.05) -> float:
  """Compute Nemenyi Critical Difference (CD) per Demšar (2006, Eq. 8).

  ``CD = (q_tukey / sqrt(2)) * sqrt(k * (k + 1) / (6 * n))`` where ``q_tukey``
  is the Studentized range critical value at confidence ``1 - alpha``
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

  If ``requested_baseline`` is provided and present in ``algorithm_names``, returns it.
  If ``requested_baseline`` is provided but absent, logs a warning and falls back to
  the first algorithm in ``algorithm_names``.
  If ``requested_baseline`` is ``None``, returns the first algorithm in ``algorithm_names``.
  If ``algorithm_names`` is empty, returns ``"Unknown"``.
  """
  if requested_baseline is not None:
    if requested_baseline in algorithm_names:
      return requested_baseline
    fallback = algorithm_names[0] if algorithm_names else "Unknown"
    logging.warning(
      "Requested baseline_algorithm '%s' not found in algorithm names (%s); falling back to '%s'.",
      requested_baseline,
      list(algorithm_names),
      fallback,
    )
    return fallback
  return algorithm_names[0] if algorithm_names else "Unknown"


def finite_metric(val: Any) -> float | None:
  """Return ``float(val)`` when ``val`` is a finite real number (excluding booleans), else ``None``."""
  if val is None or isinstance(val, bool):
    return None
  try:
    num = float(val)
  except (TypeError, ValueError):
    return None
  return num if math.isfinite(num) else None


def geometric_mean_ratios(ratios: Sequence[float]) -> float | None:
  """Compute the Fleming & Wallace (1986) geometric mean over strictly positive finite ratios."""
  valid = [num for raw in ratios if (num := finite_metric(raw)) is not None and num > 0.0]
  if not valid:
    return None
  return float(np.exp(np.mean(np.log(np.asarray(valid, dtype=float)))))


def holm_bonferroni_correction(
  p_values: Sequence[float],
  alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
  """Apply step-down Holm-Bonferroni multiple comparison correction (DP12-23, BUG-46).

  Returns:
      Tuple ``(reject_list, pvals_corrected)`` in the original input order, where
      each hypothesis is rejected if and only if its step-down adjusted p-value ``<= alpha``.
  """
  n = len(p_values)
  if n == 0:
    return [], []

  arr = np.asarray(p_values, dtype=float)
  sorted_indices = np.argsort(arr)
  sorted_pvals = arr[sorted_indices]

  raw_adjusted = np.minimum((n - np.arange(n)) * sorted_pvals, 1.0)
  pvals_corrected_sorted = np.maximum.accumulate(raw_adjusted)
  reject_sorted = pvals_corrected_sorted <= alpha

  reject = np.zeros(n, dtype=bool)
  pvals_corrected = np.zeros(n, dtype=float)
  reject[sorted_indices] = reject_sorted
  pvals_corrected[sorted_indices] = pvals_corrected_sorted

  return [bool(x) for x in reject], [float(x) for x in pvals_corrected]


__all__ = [
  "compute_nemenyi_critical_distance",
  "finite_metric",
  "geometric_mean_ratios",
  "holm_bonferroni_correction",
  "resolve_baseline_algorithm",
]
