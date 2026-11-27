"""Constrained statistical inference primitives over paired benchmark replications.

Implements the Deb (2000) constrained statistical protocol (`DP17-10`, `BUG-110`):
- Dimensionless ``hat_nu`` extraction (`raw_normalized_violations` -> legacy `raw_violations`)
- Per-seed Deb lexicographic ranking across all ``R`` paired seeds for Friedman/Nemenyi
- Exact two-sided Binomial Sign Test with Kerby-signed dominance and Vargha-Delaney ``A_12``
- Feasible-only descriptive metrics (``NaN`` CI when ``0`` feasible or ``2^{-(n-1)} > alpha``)
- Seed-averaged Deb rank matrix ``R_bar_{p,j}`` for Tier-2 multi-problem cohorts
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary
from benchmarking.analysis.stats_utils import finite_metric

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from numpy.typing import NDArray

_TIE_ABS_TOL = 1e-12


def extract_entry_normalized_violations(
  entry: Mapping[str, Any],
  expected_len: int | None = None,
) -> list[float] | None:
  """Extract per-run dimensionless ``hat_nu`` preferring ``raw_normalized_violations``."""
  for key in ("raw_normalized_violations", "raw_violations"):
    raw = entry.get(key)
    if isinstance(raw, list) and raw and (expected_len is None or len(raw) >= expected_len):
      return [float(v) for v in raw]
  return None


def _deb_keys_close(a: tuple[int, float, float], b: tuple[int, float, float], *, abs_tol: float = _TIE_ABS_TOL) -> bool:
  """Return True when two Deb lexicographic tuples represent an exact/ULP tie."""
  if a[0] != b[0]:
    return False
  idx = 2 if a[0] == 0 else 1
  va, vb = a[idx], b[idx]
  return (va == vb) if not (math.isfinite(va) and math.isfinite(vb)) else abs(va - vb) <= abs_tol


def rank_deb_lexicographic(
  keys: Sequence[tuple[int, float, float]],
  *,
  abs_tol: float = _TIE_ABS_TOL,
) -> NDArray[np.float64]:
  """Assign 1-based average ranks (1 = best) under Deb's strict lexicographic order."""
  k = len(keys)
  if k == 0:
    return np.empty(0, dtype=np.float64)

  sanitized = [
    (
      int(t[0]),
      float(t[1]) if math.isfinite(t[1]) else float("inf"),
      float(t[2]) if math.isfinite(t[2]) else float("inf"),
    )
    for t in keys
  ]
  order = sorted(range(k), key=lambda idx: sanitized[idx])
  ranks = np.empty(k, dtype=np.float64)

  start = 0
  while start < k:
    end = start + 1
    while end < k and _deb_keys_close(sanitized[order[start]], sanitized[order[end]], abs_tol=abs_tol):
      end += 1
    avg_rank = (start + 1 + end) / 2.0
    for pos in range(start, end):
      ranks[order[pos]] = avg_rank
    start = end
  return ranks


def compute_deb_seed_rank_matrix(
  costs: Sequence[NDArray[np.float64]],
  hat_nu: Sequence[NDArray[np.float64]],
  is_feasible: Sequence[NDArray[np.bool_]],
) -> NDArray[np.float64]:
  """Build an ``(R, k)`` matrix of per-seed Deb (2000) average ranks."""
  k = len(costs)
  if k == 0:
    return np.empty((0, 0), dtype=np.float64)
  n_seeds = min(len(c) for c in costs)
  rank_mat = np.empty((n_seeds, k), dtype=np.float64)
  for r in range(n_seeds):
    keys: list[tuple[int, float, float]] = []
    for j in range(k):
      feas_r = bool(is_feasible[j][r])
      cost_r = float(costs[j][r])
      nu_r = float(hat_nu[j][r])
      if feas_r and math.isfinite(cost_r):
        keys.append((0, 0.0, cost_r))
      else:
        keys.append((1, nu_r if math.isfinite(nu_r) else float("inf"), 0.0))
    rank_mat[r, :] = rank_deb_lexicographic(keys)
  return rank_mat


def _feasible_descriptive_stats(
  values: NDArray[np.float64],
  feas_mask: NDArray[np.bool_],
  analyzer: StatisticalAnalyzer,
) -> tuple[float, float, tuple[float, float], float]:
  """Compute ``(mean, std, ci_95, normality_p)`` strictly over finite feasible runs."""
  nan_ci = (float("nan"), float("nan"))
  valid = values[feas_mask]
  valid = valid[np.isfinite(valid)]
  n_feas = int(valid.size)
  if n_feas == 0:
    return float("nan"), float("nan"), nan_ci, float("nan")
  mean_val = float(np.mean(valid))
  std_val = float(np.std(valid, ddof=1)) if n_feas > 1 else 0.0
  _, ci_val = analyzer.compute_hodges_lehmann(valid, confidence=1.0 - analyzer.alpha)
  p_norm, _ = analyzer.test_normality(valid)
  return mean_val, std_val, ci_val, p_norm


def _compare_seed_deb(
  feas_a: bool,
  nu_a: float,
  cost_a: float,
  feas_b: bool,
  nu_b: float,
  cost_b: float,
) -> int:
  """Return ``-1`` if A beats B, ``+1`` if B beats A, and ``0`` on tie under Deb (2000)."""
  key_a = (0, 0.0, cost_a) if (feas_a and math.isfinite(cost_a)) else (1, nu_a, 0.0)
  key_b = (0, 0.0, cost_b) if (feas_b and math.isfinite(cost_b)) else (1, nu_b, 0.0)
  if _deb_keys_close(key_a, key_b):
    return 0
  sa = (key_a[0], key_a[1] if math.isfinite(key_a[1]) else float("inf"), key_a[2])
  sb = (key_b[0], key_b[1] if math.isfinite(key_b[1]) else float("inf"), key_b[2])
  return -1 if sa < sb else 1


def constrained_paired_comparison(
  cost_a: NDArray[np.float64],
  hat_nu_a: NDArray[np.float64],
  feas_a: NDArray[np.bool_],
  cost_b: NDArray[np.float64],
  hat_nu_b: NDArray[np.float64],
  feas_b: NDArray[np.bool_],
  *,
  label: str,
  metric_name: str,
  analyzer: StatisticalAnalyzer,
) -> StatisticalSummary:
  """Compare two constrained solvers across all ``R`` paired seeds without jump distortion."""
  r_total = min(len(cost_a), len(cost_b), len(hat_nu_a), len(hat_nu_b), len(feas_a), len(feas_b))
  ca, cb = np.asarray(cost_a[:r_total], dtype=np.float64), np.asarray(cost_b[:r_total], dtype=np.float64)
  nua, nub = np.asarray(hat_nu_a[:r_total], dtype=np.float64), np.asarray(hat_nu_b[:r_total], dtype=np.float64)
  fa, fb = np.asarray(feas_a[:r_total], dtype=bool), np.asarray(feas_b[:r_total], dtype=bool)

  mean_a, std_a, ci_a, p_norm_a = _feasible_descriptive_stats(ca, fa, analyzer)
  mean_b, std_b, ci_b, p_norm_b = _feasible_descriptive_stats(cb, fb, analyzer)

  mutual = fa & fb & np.isfinite(ca) & np.isfinite(cb)
  if bool(np.any(mutual)):
    diffs_mutual = ca[mutual] - cb[mutual]
    mean_diff = float(np.mean(diffs_mutual))
    _, ci_diff = analyzer.compute_hodges_lehmann(diffs_mutual, confidence=1.0 - analyzer.alpha)
  else:
    mean_diff, ci_diff = float("nan"), (float("nan"), float("nan"))

  wins_a = wins_b = ties = 0
  for idx in range(r_total):
    cmp = _compare_seed_deb(
      bool(fa[idx]), float(nua[idx]), float(ca[idx]), bool(fb[idx]), float(nub[idx]), float(cb[idx])
    )
    if cmp < 0:
      wins_a += 1
    elif cmp > 0:
      wins_b += 1
    else:
      ties += 1

  effective_n = wins_a + wins_b
  if r_total < 2:
    a_12 = 0.5 if r_total == 0 else float((wins_a + 0.5 * ties) / r_total)
    return StatisticalSummary(
      comparison_label=label,
      metric_name=metric_name,
      sample_size=r_total,
      mean_a=mean_a,
      mean_b=mean_b,
      std_a=std_a,
      std_b=std_b,
      ci_95_a=ci_a,
      ci_95_b=ci_b,
      mean_difference=mean_diff,
      ci_95_difference=ci_diff,
      p_value=1.0,
      effect_size=0.0,
      test_used="no_test_needed",
      normality_p_value_a=p_norm_a,
      normality_p_value_b=p_norm_b,
      test_statistic=0.0,
      df1=None,
      df2=None,
      test_justification="insufficient_sample_size",
      zero_differences_count=ties,
      effective_sample_size=effective_n,
      insufficient_power_warning=True,
      vargha_delaney_a12=a_12,
    )

  p_value = float(stats.binomtest(wins_a, effective_n, p=0.5).pvalue) if effective_n > 0 else 1.0
  # Match StatisticalAnalyzer.rank_biserial_correlation(data_a, data_b) sign convention:
  # negative when A beats B, positive when B beats A.
  effect_size = float((wins_b - wins_a) / r_total)
  a_12 = float((wins_a + 0.5 * ties) / r_total)
  exact_min_p = 1.0 if effective_n == 0 else float(2.0 ** -(effective_n - 1))
  insufficient_power = bool((ties > (r_total // 2)) or (effective_n == 0) or (exact_min_p > analyzer.alpha + 1e-15))

  return StatisticalSummary(
    comparison_label=label,
    metric_name=metric_name,
    sample_size=r_total,
    mean_a=mean_a,
    mean_b=mean_b,
    std_a=std_a,
    std_b=std_b,
    ci_95_a=ci_a,
    ci_95_b=ci_b,
    mean_difference=mean_diff,
    ci_95_difference=ci_diff,
    p_value=p_value,
    effect_size=effect_size,
    test_used="exact_binomial_sign_deb",
    normality_p_value_a=p_norm_a,
    normality_p_value_b=p_norm_b,
    test_statistic=float(wins_a),
    df1=None,
    df2=None,
    test_justification=f"deb2000_exact_binomial_sign(wins_a={wins_a},wins_b={wins_b},ties={ties},A12={a_12:.4f})",
    zero_differences_count=ties,
    effective_sample_size=effective_n,
    insufficient_power_warning=insufficient_power,
    vargha_delaney_a12=a_12,
  )


def _entry_summary_mean_violation(rec: Mapping[str, Any]) -> float:
  """Return mean dimensionless violation ``hat_nu`` for a summary-only entry."""
  viols = extract_entry_normalized_violations(rec)
  if viols:
    arr = np.asarray(viols, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
      return float(np.mean(finite))
  feas_rate = finite_metric(rec.get("feasibility_rate"))
  return max(0.0, 100.0 - feas_rate) if feas_rate is not None else 1.0


def _summary_only_deb_ranks(entries: Sequence[Mapping[str, Any]], metric_key: str) -> NDArray[np.float64] | None:
  """Rank summary-only entries by ``(-feasibility_rate, mean_hat_nu, mean_feasible_cost)``."""
  k = len(entries)
  tuples: list[tuple[float, float, float]] = []
  for rec in entries:
    raw_feas = rec.get("raw_is_feasible")
    feas_rate = finite_metric(rec.get("feasibility_rate"))
    if feas_rate is None:
      if isinstance(raw_feas, list) and raw_feas:
        feas_rate = 100.0 * sum(bool(v) for v in raw_feas) / len(raw_feas)
      else:
        feas_runs, succ_runs = rec.get("feasible_runs"), rec.get("successful_runs")
        if isinstance(feas_runs, int) and isinstance(succ_runs, int) and succ_runs > 0:
          feas_rate = 100.0 * feas_runs / succ_runs
        elif isinstance(feas_runs, int) and feas_runs == 0 and finite_metric(rec.get("mean_cost")) is None:
          feas_rate = 0.0
        else:
          feas_rate = 100.0

    if feas_rate <= 0.0:
      mean_cost_val, mean_nu = 0.0, _entry_summary_mean_violation(rec)
    else:
      val = finite_metric(rec.get(metric_key))
      if val is None and metric_key == "mean_gap":
        val = finite_metric(rec.get("mean_cost"))
      if val is None:
        return None
      mean_cost_val = val
      mean_nu = 0.0 if feas_rate >= 100.0 - 1e-9 else _entry_summary_mean_violation(rec)

    tuples.append((-feas_rate, mean_nu, mean_cost_val))

  order = sorted(range(k), key=lambda idx: tuples[idx])
  ranks = np.empty(k, dtype=np.float64)
  start = 0
  while start < k:
    end = start + 1
    while end < k:
      ta, tb = tuples[order[start]], tuples[order[end]]
      if all(abs(ta[m] - tb[m]) <= _TIE_ABS_TOL for m in range(3)):
        end += 1
      else:
        break
    avg_rank = (start + 1 + end) / 2.0
    for pos in range(start, end):
      ranks[order[pos]] = avg_rank
    start = end
  return ranks


def compute_seed_averaged_deb_ranks(
  entries: Sequence[Mapping[str, Any]],
  metric_key: str,
) -> NDArray[np.float64] | None:
  """Compute ``R_bar_{p,j} = (1 / R_p) * sum_r rank_Deb(p, r, j)`` for one problem."""
  if not entries:
    return np.empty(0, dtype=np.float64)

  seed_lengths: list[int] = []
  cost_lists: list[list[float]] = []
  feas_lists: list[list[bool]] = []
  has_all_per_seed = True

  for rec in entries:
    raw_unpen, raw_costs = rec.get("raw_unpenalized_costs"), rec.get("raw_costs")
    cost_src = raw_unpen if isinstance(raw_unpen, list) and raw_unpen else raw_costs
    raw_feas = rec.get("raw_is_feasible")
    if not isinstance(cost_src, list) or not cost_src:
      has_all_per_seed = False
      break

    if isinstance(raw_feas, list) and raw_feas:
      r_len = min(len(cost_src), len(raw_feas))
      feas_src = [bool(v) for v in raw_feas[:r_len]]
    else:
      feas_rate = finite_metric(rec.get("feasibility_rate"))
      if feas_rate is not None and feas_rate < 100.0 - 1e-9:
        has_all_per_seed = False
        break
      r_len = len(cost_src)
      feas_src = [True] * r_len

    if r_len < 1:
      has_all_per_seed = False
      break
    seed_lengths.append(r_len)
    cost_lists.append([float(v) for v in cost_src[:r_len]])
    feas_lists.append(feas_src)

  if not has_all_per_seed:
    return _summary_only_deb_ranks(entries, metric_key)

  common_r = min(seed_lengths)
  cost_arrays: list[NDArray[np.float64]] = []
  feas_arrays: list[NDArray[np.bool_]] = []
  nu_arrays: list[NDArray[np.float64]] = []

  for idx, rec in enumerate(entries):
    cost_arrays.append(np.asarray(cost_lists[idx][:common_r], dtype=np.float64))
    feas_arrays.append(np.asarray(feas_lists[idx][:common_r], dtype=bool))
    viols = extract_entry_normalized_violations(rec, expected_len=common_r)
    if viols is not None:
      nu_arrays.append(np.asarray(viols[:common_r], dtype=np.float64))
    else:
      nu_arrays.append(np.where(feas_arrays[-1], 0.0, 1.0).astype(np.float64))

  rank_mat = compute_deb_seed_rank_matrix(cost_arrays, nu_arrays, feas_arrays)
  return np.mean(rank_mat, axis=0)


__all__ = [
  "compute_deb_seed_rank_matrix",
  "compute_seed_averaged_deb_ranks",
  "constrained_paired_comparison",
  "extract_entry_normalized_violations",
  "rank_deb_lexicographic",
]
