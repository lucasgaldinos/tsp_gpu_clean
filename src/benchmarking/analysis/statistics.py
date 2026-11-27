"""Non-parametric statistical primitives for benchmark analysis.

Implements the Demšar (2006) protocol for comparing stochastic optimization algorithms:
- Descriptive Shapiro-Wilk normality diagnostics (never gates test selection; DP13-4)
- Unconditional paired Wilcoxon signed-rank hypothesis testing (`zero_method="zsplit"`)
- Matched-pairs rank-biserial correlation effect size (Kerby 2014) and classification
- Distribution-free Hodges-Lehmann location estimator and exact/asymptotic Walsh-average CIs
- Step-down Holm-Bonferroni FWER adjustment (Holm 1979)
- Single-pass Friedman omnibus test with Iman-Davenport (1980) F_F correction, Nemenyi (1963)
  Critical Difference, pairwise p-values, and maximal non-significant cliques
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from benchmarking.analysis.stats_utils import (
  compute_nemenyi_critical_distance,
)
from benchmarking.analysis.stats_utils import (
  holm_bonferroni_correction as _holm_bonferroni,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from numpy.typing import NDArray

  from data_models.records import NemenyiResult


def build_feasibility_aware_scores(
  costs: Sequence[float] | np.ndarray,
  violations: Sequence[float] | np.ndarray | None = None,
  is_feasible: Sequence[bool] | np.ndarray | None = None,
) -> NDArray[np.float64]:
  """Transform raw solution costs into Deb (2000) lexicographic feasibility scores.

  For unconstrained or 100%-feasible inputs (``is_feasible is None`` or
  ``all(is_feasible)``), returns ``np.asarray(costs, dtype=np.float64)`` unchanged
  so unconstrained TSP statistics remain bit-identical.

  When any entry is infeasible, defines
  :math:`C^{\\max} = \\max_{i:\\,\\text{feasible}[i]} c_i` (or ``0.0`` when zero
  feasible entries exist) and maps:

  .. math::

      \\tilde{z}_i = \\begin{cases}
          c_i, & \\text{if feasible}[i] \\\\
          C^{\\max} + 1.0 + \\nu_i, & \\text{otherwise}
      \\end{cases}

  Guarantees that every feasible solution strictly beats every infeasible solution
  regardless of arbitrary penalty weights :math:`\\omega_Q`, while infeasible
  solutions rank monotonically by lower physical violation :math:`\\nu_i`.
  """
  costs_arr: NDArray[np.float64] = np.asarray(costs, dtype=np.float64)
  if is_feasible is None:
    return costs_arr

  feas_mask: NDArray[np.bool_] = np.asarray(is_feasible, dtype=bool)
  if feas_mask.shape != costs_arr.shape:
    raise ValueError(
      f"is_feasible shape {feas_mask.shape} must match costs shape {costs_arr.shape}."
    )

  if bool(np.all(feas_mask)):
    return costs_arr

  viol_arr: NDArray[np.float64] = (
    np.zeros_like(costs_arr, dtype=np.float64)
    if violations is None
    else np.asarray(violations, dtype=np.float64)
  )
  if viol_arr.shape != costs_arr.shape:
    raise ValueError(
      f"violations shape {viol_arr.shape} must match costs shape {costs_arr.shape}."
    )

  feasible_costs = costs_arr[feas_mask]
  finite_feasible = feasible_costs[np.isfinite(feasible_costs)]
  c_max = float(np.max(finite_feasible)) if finite_feasible.size > 0 else 0.0

  return np.where(feas_mask, costs_arr, c_max + 1.0 + viol_arr).astype(np.float64, copy=False)

POSTHOCS_AVAILABLE = find_spec("scikit_posthocs") is not None

# Exact DP table cap for the Wilcoxon signed-rank null distribution; beyond n = 50
# the continuity-corrected normal approximation matches exact quantiles to <1e-4.
_EXACT_WILCOXON_MAX_N = 50

# Minimum block count below which asymptotic Friedman/Iman-Davenport inference
# carries an explicit low-power advisory flag (Demšar 2006).
_OMNIBUS_LOW_POWER_MIN_BLOCKS = 5


def interpret_rank_biserial(effect_size: float) -> str:
  """Classify matched-pairs rank-biserial correlation |r| in [0, 1] (Cohen 1988; Kerby 2014).

  Uses standard correlation magnitude cutoffs rather than Cohen's d cutoffs:
  - ``|r| < 0.1``: ``"negligible"``
  - ``0.1 <= |r| < 0.3``: ``"small"``
  - ``0.3 <= |r| < 0.5``: ``"medium"``
  - ``|r| >= 0.5``: ``"large"``
  """
  if not math.isfinite(effect_size):
    return "undefined"
  abs_r = abs(effect_size)
  if abs_r < 0.1:
    return "negligible"
  if abs_r < 0.3:
    return "small"
  if abs_r < 0.5:
    return "medium"
  return "large"


def _hodges_lehmann_critical_rank(n: int, alpha: float) -> int | None:
  """Return the 1-indexed lower Walsh-average rank ``k`` for a ``(1 - alpha)`` CI.

  A distribution-free Hodges-Lehmann interval ``[W_(k), W_(m - k + 1)]`` (where
  ``m = n * (n + 1) // 2``) has exact coverage ``1 - 2 * P(T^+ <= k - 1)``.
  When ``2^{-(n - 1)} > alpha`` (e.g., ``n <= 5`` at ``alpha = 0.05``), even the
  widest interval ``[W_(1), W_(m)]`` cannot attain ``1 - alpha`` coverage, so
  ``None`` is returned rather than mislabeling a lower-coverage range.
  """
  if n < 2 or not (0.0 < alpha < 1.0):
    return None
  if (2.0 ** -(n - 1)) > alpha + 1e-15:
    return None

  m = n * (n + 1) // 2
  if n <= _EXACT_WILCOXON_MAX_N:
    counts = np.zeros(m + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in range(1, n + 1):
      counts[r:] += counts[:-r]
    total = float(2**n)
    two_sided_cdf = 2.0 * np.cumsum(counts) / total
    valid_c = np.flatnonzero(two_sided_cdf <= alpha + 1e-15)
    if valid_c.size == 0:
      return None
    c = int(valid_c[-1])
    return min(c + 1, m)

  z = float(stats.norm.ppf(1.0 - alpha / 2.0))
  e_w = n * (n + 1) / 4.0
  var_w = n * (n + 1) * (2 * n + 1) / 24.0
  c_approx = math.floor(e_w - 0.5 - z * math.sqrt(var_w))
  if c_approx < 0:
    return None
  return max(1, min(c_approx + 1, m))


@dataclass(frozen=True)
class OmnibusTestResult:
  """Unified, immutable result of a single-pass Friedman + Iman-Davenport + Nemenyi analysis.

  Serves as the single source of truth consumed by console logging, JSON serialization,
  LaTeX macros, and Demšar Critical Difference TikZ rendering.

  Attributes:
      k_algorithms: Number of compared algorithms (``k >= 3``).
      n_blocks: Number of paired blocks (instances or replications, ``N >= 2``).
      algorithm_names: Canonical algorithm identifiers matching matrix columns.
      chi2_stat: Friedman chi-square statistic.
      chi2_pvalue: Asymptotic chi-square p-value for Friedman's test.
      friedman_f: Iman-Davenport (1980) ``F_F`` omnibus statistic.
      df1: Numerator degrees of freedom ``k - 1``.
      df2: Denominator degrees of freedom ``(k - 1) * (N - 1)``.
      friedman_pvalue: F-distribution p-value for ``F_F``.
      alpha: Configured significance level.
      omnibus_significant: ``True`` iff ``friedman_pvalue < alpha``.
      low_power: ``True`` when ``n_blocks < 5`` (asymptotic F_F advisory).
      ranks: Mapping from algorithm name to average rank across blocks (1 = best).
      rank_sum: Sum of average ranks (conserved at ``k * (k + 1) / 2``).
      critical_difference: Nemenyi Critical Difference (CD) at ``alpha``.
      cliques: Maximal non-significant cliques ``(algo_start, algo_end, rank_start, rank_end)``
          with ``|R_i - R_j| <= CD``.
      nemenyi_pvalues: Symmetric ``(k, k)`` matrix of Nemenyi pairwise p-values.
      significant_pairs: Pairs ``(algo_i, algo_j, p_value)`` with Nemenyi ``p_value < alpha``.
  """

  k_algorithms: int
  n_blocks: int
  algorithm_names: tuple[str, ...]
  chi2_stat: float
  chi2_pvalue: float
  friedman_f: float
  df1: int
  df2: int
  friedman_pvalue: float
  alpha: float
  omnibus_significant: bool
  low_power: bool
  ranks: dict[str, float]
  rank_sum: float
  critical_difference: float
  cliques: tuple[tuple[str, str, float, float], ...]
  nemenyi_pvalues: tuple[tuple[float, ...], ...]
  significant_pairs: tuple[tuple[str, str, float], ...]

  def to_friedman_dict(self) -> dict[str, Any]:
    """Serialize omnibus Friedman / Iman-Davenport metrics into a JSON-ready dictionary."""
    return {
      "statistic": float(self.chi2_stat),
      "p_value": float(self.chi2_pvalue),
      "statistic_f": float(self.friedman_f),
      "df1": int(self.df1),
      "df2": int(self.df2),
      "p_value_f": float(self.friedman_pvalue),
      "significant": bool(self.omnibus_significant),
      "post_hoc_required": bool(self.omnibus_significant),
      "low_power": bool(self.low_power),
    }

  def to_nemenyi_dict(self) -> dict[str, Any]:
    """Serialize Nemenyi post-hoc metrics into a JSON-ready dictionary."""
    return {
      "critical_distance": float(self.critical_difference),
      "mean_ranks": dict(self.ranks),
      "algorithm_names": list(self.algorithm_names),
      "omnibus_significant": bool(self.omnibus_significant),
      "significant_pairs": [
        {
          "algorithm_i": name_i,
          "algorithm_j": name_j,
          "p_value": float(p_val),
          "rank_difference": abs(self.ranks[name_i] - self.ranks[name_j]),
        }
        for name_i, name_j, p_val in self.significant_pairs
      ],
    }


@dataclass
class StatisticalSummary:
  """Statistical summary for a paired two-algorithm comparison.

  Attributes:
      comparison_label: Human-readable comparison (e.g., ``"AlgA vs AlgB"``).
      metric_name: Measured quantity (e.g., ``"cost"``, ``"time"``).
      sample_size: Number of paired observations (``n``).
      mean_a: Arithmetic mean for configuration A.
      mean_b: Arithmetic mean for configuration B.
      std_a: Sample standard deviation (``ddof=1``) for A.
      std_b: Sample standard deviation (``ddof=1``) for B.
      ci_95_a: Distribution-free one-sample Hodges-Lehmann CI for A as ``(low, high)``.
      ci_95_b: Distribution-free one-sample Hodges-Lehmann CI for B as ``(low, high)``.
      mean_difference: Mean of ``A - B``.
      ci_95_difference: Distribution-free paired Hodges-Lehmann CI for ``A - B``.
      p_value: Two-sided Wilcoxon signed-rank p-value (or ``1.0`` when all diffs are zero).
      effect_size: Matched-pairs rank-biserial correlation ``r`` in ``[-1, 1]``.
      test_used: ``"wilcoxon_signed_rank"`` or ``"no_test_needed"``.
      normality_p_value_a: Descriptive Shapiro-Wilk p-value for A (``NaN`` if untested/constant).
      normality_p_value_b: Descriptive Shapiro-Wilk p-value for B (``NaN`` if untested/constant).
      test_statistic: Primary test statistic (Wilcoxon ``W``).
      df1: Numerator degrees of freedom when applicable, else ``None``.
      df2: Denominator degrees of freedom when applicable, else ``None``.
      test_justification: Rationale tag for test selection.
      zero_differences_count: Number of paired observations with zero difference.
      effective_sample_size: Non-zero difference count available to Wilcoxon.
      insufficient_power_warning: ``True`` when >50% of diffs are zero or ``2^{-(n_eff-1)} > alpha``.
      vargha_delaney_a12: Optional Vargha-Delaney :math:`\\hat{A}_{12}` dominance probability
          for constrained Binomial Sign Test comparisons (``None`` for unconstrained Wilcoxon).
  """

  comparison_label: str
  metric_name: str
  sample_size: int
  mean_a: float
  mean_b: float
  std_a: float
  std_b: float
  ci_95_a: tuple[float, float]
  ci_95_b: tuple[float, float]
  mean_difference: float
  ci_95_difference: tuple[float, float]
  p_value: float
  effect_size: float
  test_used: str
  normality_p_value_a: float
  normality_p_value_b: float
  test_statistic: float
  df1: float | int | None
  df2: float | int | None
  test_justification: str = "demsar2006_wilcoxon_default"
  zero_differences_count: int = 0
  effective_sample_size: int = 0
  insufficient_power_warning: bool = False
  vargha_delaney_a12: float | None = None

  def get_speedup(self) -> float:
    """Calculate speedup ratio ``mean_a / mean_b``, returning ``0.0`` when ``mean_b == 0.0``."""
    if self.mean_b == 0.0:
      return 0.0
    return self.mean_a / self.mean_b

  def is_significant(self, alpha: float = 0.05) -> bool:
    """Check if result is statistically significant at level ``alpha``."""
    return self.p_value < alpha


def compute_iman_davenport_f(
  chi2: float,
  n_observations: int,
  k_algorithms: int,
) -> tuple[float, int, int, float]:
  """Compute the Iman-Davenport (1980) ``F_F`` correction from Friedman's chi-square.

  Converts Friedman's ``chi2`` into ``F_F = ((N - 1) * chi2) / (N * (k - 1) - chi2)``
  with ``df1 = k - 1`` and ``df2 = (k - 1) * (N - 1)``, returning
  ``(F_F, df1, df2, p_value_f)``. Handles ``N < 2``, non-finite ``chi2``, and
  perfect rank concordance (``denominator <= 1e-12``).
  """
  df1 = max(0, k_algorithms - 1)
  df2 = max(0, (k_algorithms - 1) * (n_observations - 1))

  if n_observations < 2 or k_algorithms < 2 or not np.isfinite(chi2):
    return 0.0, df1, df2, 1.0

  denominator = n_observations * (k_algorithms - 1) - chi2
  if denominator <= 1e-12:
    if chi2 > 0.0:
      return float("inf"), df1, df2, 0.0
    return 0.0, df1, df2, 1.0

  statistic_f = float(((n_observations - 1) * chi2) / denominator)
  p_value_f = float(stats.f.sf(statistic_f, df1, df2))
  return statistic_f, df1, df2, p_value_f


def compute_mean_ranks(
  data_sets: Sequence[np.ndarray] | np.ndarray,
  algorithm_names: Sequence[str],
  *,
  higher_is_better: bool = False,
) -> dict[str, float]:
  """Compute average per-block ranks (1 = best) for each algorithm."""
  mat = np.asarray(data_sets, dtype=float) if isinstance(data_sets, np.ndarray) else np.column_stack(data_sets)
  if mat.ndim != 2 or mat.shape[1] != len(algorithm_names):
    raise ValueError(
      f"Expected 2D matrix with {len(algorithm_names)} columns matching algorithm_names; got shape {mat.shape}."
    )
  multiplier = -1.0 if higher_is_better else 1.0
  ranked = np.apply_along_axis(stats.rankdata, 1, multiplier * mat, method="average")
  avg_ranks = np.mean(ranked, axis=0)
  return {name: float(avg_ranks[i]) for i, name in enumerate(algorithm_names)}


def _compute_maximal_cliques(
  ranks: dict[str, float],
  critical_difference: float,
) -> tuple[tuple[str, str, float, float], ...]:
  """Identify maximal non-significant rank intervals ``|R_i - R_j| <= CD``."""
  sorted_algos = sorted(ranks.items(), key=lambda item: (item[1], item[0]))
  k = len(sorted_algos)
  intervals: list[tuple[int, int]] = []
  for i in range(k):
    for j in range(i + 1, k):
      if sorted_algos[j][1] - sorted_algos[i][1] <= critical_difference + 1e-9:
        intervals.append((i, j))

  maximal: list[tuple[str, str, float, float]] = []
  for i, j in intervals:
    if not any((i_o, j_o) != (i, j) and i_o <= i and j_o >= j for i_o, j_o in intervals):
      maximal.append((sorted_algos[i][0], sorted_algos[j][0], sorted_algos[i][1], sorted_algos[j][1]))
  return tuple(maximal)


class StatisticalAnalyzer:
  """Performs non-parametric statistical analysis on benchmark results.

  Implements the Demšar (2006) protocol (`DP13-4` / `BUG-60`):
  1. Descriptive Shapiro-Wilk normality diagnostics (never gates test choice)
  2. Unconditional paired Wilcoxon signed-rank testing (`zero_method="zsplit"`)
  3. Matched-pairs rank-biserial correlation effect size (Kerby 2014)
  4. Distribution-free Hodges-Lehmann location & Walsh-average confidence intervals
  5. Step-down Holm-Bonferroni multiple comparison correction
  6. Friedman omnibus test with Iman-Davenport ``F_F`` and Nemenyi post-hoc testing
  """

  def __init__(self, alpha: float = 0.05) -> None:
    """Initialize analyzer with significance level ``alpha`` in ``(0, 1)``."""
    if not (0.0 < alpha < 1.0):
      raise ValueError(f"Significance level alpha must be strictly in (0.0, 1.0), got alpha={alpha}")
    self.alpha = alpha

  def test_normality(self, data: np.ndarray) -> tuple[float, bool]:
    """Compute descriptive Shapiro-Wilk normality diagnostic ``(p_value, is_normal)``.

    Returns ``(float("nan"), False)`` when ``len(data) < 3`` or the sample has
    zero variance (``std < 1e-10``), avoiding false ``p = 0.0`` or ``p = 1.0``
    sentinels. Never used to switch away from Wilcoxon signed-rank testing.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size < 3 or not np.all(np.isfinite(arr)):
      return (float("nan"), False)

    if float(np.std(arr)) < 1e-10:
      return (float("nan"), False)

    _, p_value = stats.shapiro(arr)
    p_float = float(p_value)
    return (p_float, bool(p_float >= self.alpha))

  def compute_hodges_lehmann(
    self,
    values: np.ndarray,
    confidence: float = 0.95,
  ) -> tuple[float, tuple[float, float]]:
    """Compute Hodges-Lehmann location estimator and distribution-free CI.

    Calculates Walsh averages ``(v_i + v_j) / 2`` for ``1 <= i <= j <= n``.
    Uses the exact Wilcoxon signed-rank null distribution for ``n <= 50`` and the
    continuity-corrected normal approximation for ``n > 50``. When ``2^{-(n - 1)} > alpha``
    (e.g., ``n <= 5`` at 95% confidence), no Walsh-average interval reaches the
    requested coverage, so the CI bounds are returned as ``(nan, nan)``.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
      return float("nan"), (float("nan"), float("nan"))
    if n == 1:
      return float(arr[0]), (float("nan"), float("nan"))

    i_idx, j_idx = np.triu_indices(n)
    walsh = (arr[i_idx] + arr[j_idx]) / 2.0
    walsh_sorted = np.sort(walsh)
    m = walsh_sorted.size
    point_estimate = float(np.median(walsh_sorted))

    alpha = 1.0 - confidence
    k = _hodges_lehmann_critical_rank(n, alpha)
    if k is None:
      return point_estimate, (float("nan"), float("nan"))

    low = float(walsh_sorted[k - 1])
    high = float(walsh_sorted[m - k])
    return point_estimate, (low, high)

  def rank_biserial_correlation(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
  ) -> float:
    """Compute matched-pairs rank-biserial correlation effect size (Kerby 2014).

    For differences ``D = A - B``, ranks absolute non-zero differences ``|D|``.
    ``r = (W_+ - W_-) / (W_+ + W_-)`` in ``[-1.0, 1.0]``; returns ``0.0`` if all
    differences are zero.
    """
    arr_a = np.asarray(data_a, dtype=float)
    arr_b = np.asarray(data_b, dtype=float)
    diffs = arr_a - arr_b
    nonzero = diffs[diffs != 0.0]
    if nonzero.size == 0:
      return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos_ranks = float(np.sum(ranks[nonzero > 0.0]))
    neg_ranks = float(np.sum(ranks[nonzero < 0.0]))
    total = pos_ranks + neg_ranks
    if total == 0.0:
      return 0.0
    return float((pos_ranks - neg_ranks) / total)

  @staticmethod
  def _count_zero_differences(differences: np.ndarray, *, tolerance: float = 1e-10) -> int:
    """Count paired differences that vanish within absolute ``tolerance``."""
    return int(np.sum(np.abs(differences) < tolerance))

  def paired_comparison(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
    label: str,
    metric_name: str,
  ) -> StatisticalSummary:
    """Perform an unconditional non-parametric paired comparison of ``data_a`` vs ``data_b``.

    Always applies Wilcoxon signed-rank testing (including constant non-zero
    differences, where exact ``p = 2^{-(n-1)}`` prevents false-positive ``p = 0.0``
    at small ``n``). Short-circuits with ``test_used="no_test_needed"`` only when
    ``n < 2`` or all paired differences are zero within ``1e-10``.
    """
    arr_a = np.asarray(data_a, dtype=float)
    arr_b = np.asarray(data_b, dtype=float)
    if arr_a.size != arr_b.size:
      raise ValueError(f"Arrays must have same length: {arr_a.size} != {arr_b.size}")

    n = int(arr_a.size)
    nan_ci = (float("nan"), float("nan"))

    if n < 2:
      mean_a = float(np.mean(arr_a)) if n == 1 else float("nan")
      mean_b = float(np.mean(arr_b)) if n == 1 else float("nan")
      mean_diff = mean_a - mean_b if n == 1 else float("nan")
      zero_diffs = int(abs(arr_a[0] - arr_b[0]) < 1e-10) if n == 1 else 0
      return StatisticalSummary(
        comparison_label=label,
        metric_name=metric_name,
        sample_size=n,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=0.0,
        std_b=0.0,
        ci_95_a=nan_ci,
        ci_95_b=nan_ci,
        mean_difference=mean_diff,
        ci_95_difference=nan_ci,
        p_value=1.0,
        effect_size=0.0,
        test_used="no_test_needed",
        normality_p_value_a=float("nan"),
        normality_p_value_b=float("nan"),
        test_statistic=0.0,
        df1=None,
        df2=None,
        test_justification="insufficient_sample_size",
        zero_differences_count=zero_diffs,
        effective_sample_size=n - zero_diffs,
        insufficient_power_warning=True,
      )

    differences = arr_a - arr_b
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    std_a = float(np.std(arr_a, ddof=1))
    std_b = float(np.std(arr_b, ddof=1))
    mean_diff = float(np.mean(differences))
    p_norm_a, _ = self.test_normality(arr_a)
    p_norm_b, _ = self.test_normality(arr_b)

    confidence = 1.0 - self.alpha
    _, ci_a = self.compute_hodges_lehmann(arr_a, confidence=confidence)
    _, ci_b = self.compute_hodges_lehmann(arr_b, confidence=confidence)

    zero_diffs = self._count_zero_differences(differences)
    effective_n = n - zero_diffs
    exact_min_p = 1.0 if effective_n == 0 else float(2.0 ** -(effective_n - 1))
    insufficient_power = bool((zero_diffs > (n // 2)) or (effective_n == 0) or (exact_min_p > self.alpha + 1e-15))

    if effective_n == 0:
      return StatisticalSummary(
        comparison_label=label,
        metric_name=metric_name,
        sample_size=n,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        ci_95_a=ci_a,
        ci_95_b=ci_b,
        mean_difference=0.0,
        ci_95_difference=(0.0, 0.0),
        p_value=1.0,
        effect_size=0.0,
        test_used="no_test_needed",
        normality_p_value_a=p_norm_a,
        normality_p_value_b=p_norm_b,
        test_statistic=0.0,
        df1=None,
        df2=None,
        test_justification="identical_samples_no_test_needed",
        zero_differences_count=zero_diffs,
        effective_sample_size=0,
        insufficient_power_warning=True,
      )

    if zero_diffs > (n // 2):
      logging.warning(
        "More than half of paired differences are zero (%d/%d); "
        "effective n=%d yields insufficient power for Wilcoxon test.",
        zero_diffs,
        n,
        effective_n,
      )

    res = stats.wilcoxon(arr_a, arr_b, zero_method="zsplit")
    p_value = float(res.pvalue)
    effect_size = self.rank_biserial_correlation(arr_a, arr_b)
    _, ci_diff = self.compute_hodges_lehmann(differences, confidence=confidence)

    return StatisticalSummary(
      comparison_label=label,
      metric_name=metric_name,
      sample_size=n,
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
      test_used="wilcoxon_signed_rank",
      normality_p_value_a=p_norm_a,
      normality_p_value_b=p_norm_b,
      test_statistic=float(res.statistic),
      df1=None,
      df2=None,
      test_justification="demsar2006_wilcoxon_default",
      zero_differences_count=zero_diffs,
      effective_sample_size=effective_n,
      insufficient_power_warning=insufficient_power,
    )

  def wilcoxon_paired_summary(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
  ) -> tuple[float, float, float]:
    """Compute paired Wilcoxon signed-rank ``(W, p_value, rank_biserial_r)`` using ``zsplit``.

    Returns ``(0.0, 1.0, 0.0)`` cleanly when paired vectors have ``n < 2`` or all
    paired differences vanish within the same scale-invariant absolute tolerance
    (``1e-10``) used by ``paired_comparison``.
    """
    arr_a = np.asarray(data_a, dtype=float)
    arr_b = np.asarray(data_b, dtype=float)
    if arr_a.size != arr_b.size:
      raise ValueError(f"Arrays must have same length: {arr_a.size} != {arr_b.size}")
    if arr_a.size < 2:
      return 0.0, 1.0, 0.0
    differences = arr_a - arr_b
    if self._count_zero_differences(differences) == int(arr_a.size):
      return 0.0, 1.0, 0.0
    wilcoxon_res = stats.wilcoxon(arr_a, arr_b, zero_method="zsplit")
    w_stat = float(wilcoxon_res.statistic)
    p_val = float(wilcoxon_res.pvalue)
    r_eff = float(self.rank_biserial_correlation(arr_a, arr_b))
    return w_stat, p_val, r_eff

  def holm_bonferroni_correction(
    self,
    p_values: Sequence[float],
    alpha: float | None = None,
  ) -> list[tuple[int, float, bool]]:
    """Apply step-down Holm-Bonferroni correction using ``alpha`` (defaults to ``self.alpha``)."""
    if not p_values:
      return []
    effective_alpha = self.alpha if alpha is None else alpha
    reject, _ = _holm_bonferroni(p_values, alpha=effective_alpha)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    return [(orig_idx, float(p_val), reject[orig_idx]) for orig_idx, p_val in indexed]

  def friedman_test(self, data_sets: Sequence[np.ndarray]) -> dict[str, Any]:
    """Perform Friedman omnibus test with Iman-Davenport ``F_F`` correction.

    Following Demšar (2006): computes both ``chi2_F`` and the less conservative
    Iman-Davenport ``F_F`` statistic, deciding significance via ``p_value_f < self.alpha``.
    """
    if len(set(len(d) for d in data_sets)) > 1:
      raise ValueError(
        "All data_sets must have the same number of observations (paired samples). "
        f"Got lengths: {[len(d) for d in data_sets]}"
      )

    if len(data_sets) < 3:
      raise ValueError(f"Friedman test requires at least 3 algorithms (k >= 3). Got {len(data_sets)} algorithms.")

    data_matrix = np.column_stack(data_sets)
    if not np.all(np.isfinite(data_matrix)):
      raise ValueError("data_sets contain non-finite (NaN or Inf) values.")

    n_observations, k_algorithms = data_matrix.shape

    if n_observations < 2 or bool(np.all(np.ptp(data_matrix, axis=1) == 0.0)):
      chi2, p_value = 0.0, 1.0
    else:
      raw_chi2, raw_p = stats.friedmanchisquare(*data_sets)
      if np.isnan(raw_chi2) or np.isnan(raw_p):
        chi2, p_value = 0.0, 1.0
      else:
        chi2, p_value = float(raw_chi2), float(raw_p)

    statistic_f, df1, df2, p_value_f = compute_iman_davenport_f(
      chi2,
      n_observations,
      k_algorithms,
    )
    is_significant = bool(p_value_f < self.alpha)

    return {
      "statistic": chi2,
      "p_value": p_value,
      "statistic_f": float(statistic_f),
      "df1": int(df1),
      "df2": int(df2),
      "p_value_f": float(p_value_f),
      "significant": is_significant,
      "post_hoc_required": is_significant,
      "low_power": bool(n_observations < _OMNIBUS_LOW_POWER_MIN_BLOCKS),
    }

  def nemenyi_posthoc(
    self,
    data_sets: Sequence[np.ndarray],
    algorithm_names: Sequence[str] | None = None,
    *,
    higher_is_better: bool = False,
  ) -> NemenyiResult:
    """Perform Nemenyi pairwise post-hoc test over paired algorithm blocks.

    Supports ``higher_is_better`` orientation directly and guards all-tied block
    matrices so callers never receive ``NaN`` p-value matrices from ``scikit_posthocs``.
    """
    k = len(data_sets)
    if k < 2:
      raise ValueError(f"Nemenyi post-hoc test requires at least 2 algorithms (k >= 2), got k={k}")

    if len(set(len(d) for d in data_sets)) > 1:
      raise ValueError(
        f"All data_sets must have the same number of observations. Got lengths: {[len(d) for d in data_sets]}"
      )

    n_instances = len(data_sets[0])
    if n_instances <= 0:
      raise ValueError(f"Nemenyi post-hoc test requires at least 1 observation (n >= 1), got n={n_instances}")

    resolved_names = [f"Alg{i + 1}" for i in range(k)] if algorithm_names is None else list(algorithm_names)
    if len(resolved_names) != k:
      raise ValueError(f"algorithm_names length ({len(resolved_names)}) must match data_sets length ({k})")

    data_array = np.column_stack(data_sets).astype(np.float64, copy=False)
    if not np.all(np.isfinite(data_array)):
      raise ValueError("data_sets contain non-finite (NaN or Inf) values.")

    critical_distance = compute_nemenyi_critical_distance(k, n_instances, self.alpha)

    if bool(np.all(np.ptp(data_array, axis=1) == 0.0)):
      p_values_matrix = np.ones((k, k), dtype=np.float64)
      return {
        "p_values_matrix": p_values_matrix,
        "significant_pairs": [],
        "algorithm_names": resolved_names,
        "critical_distance": float(critical_distance),
      }

    oriented = -data_array if higher_is_better else data_array
    from scipy.stats import rankdata, studentized_range

    if POSTHOCS_AVAILABLE:
      import scikit_posthocs as sp

      p_matrix = sp.posthoc_nemenyi_friedman(oriented)
      p_values_matrix = np.asarray(p_matrix.values, dtype=np.float64)
    else:
      ranks = np.apply_along_axis(rankdata, 1, oriented)
      mean_ranks = np.mean(ranks, axis=0)
      se = np.sqrt(k * (k + 1) / (12.0 * n_instances))
      p_values_matrix = np.ones((k, k), dtype=np.float64)
      for i in range(k):
        for j in range(i + 1, k):
          q_stat = abs(mean_ranks[i] - mean_ranks[j]) / se
          p_val = float(studentized_range.sf(q_stat, k, np.inf))
          p_values_matrix[i, j] = p_val
          p_values_matrix[j, i] = p_val

    significant_pairs: list[tuple[int, int, float]] = []
    for i in range(k):
      for j in range(i + 1, k):
        p_val = float(p_values_matrix[i, j])
        if p_val < self.alpha:
          significant_pairs.append((i, j, p_val))

    return {
      "p_values_matrix": p_values_matrix,
      "significant_pairs": significant_pairs,
      "algorithm_names": resolved_names,
      "critical_distance": float(critical_distance),
    }


def run_friedman_nemenyi_analysis(
  data_matrix: Sequence[np.ndarray] | np.ndarray,
  algorithm_names: Sequence[str],
  *,
  alpha: float = 0.05,
  higher_is_better: bool = False,
) -> OmnibusTestResult:
  """Run unified Friedman + Iman-Davenport + Nemenyi + clique analysis in a single pass.

  Enforces the repository-wide multi-algorithm omnibus contract:
  - Mathematical domain: ``k >= 3`` algorithms and ``N >= 2`` paired blocks (``df2 >= 2``)
  - Explicit ``low_power = (N < 5)`` advisory flag
  - Rank conservation check ``sum(R_j) == k * (k + 1) / 2``
  - Computes Nemenyi CD, maximal cliques, and pairwise p-values alongside explicit
    ``omnibus_significant = (p_value_f < alpha)`` so every consumer shares one source of truth.
  """
  mat = (
    np.asarray(data_matrix, dtype=float)
    if isinstance(data_matrix, np.ndarray)
    else np.column_stack(data_matrix).astype(float, copy=False)
  )
  if mat.ndim != 2:
    raise ValueError(f"data_matrix must be 2-dimensional (N blocks, k algorithms). Got shape {mat.shape}.")

  n_blocks, k_algorithms = mat.shape
  names_tuple = tuple(algorithm_names)
  if len(names_tuple) != k_algorithms:
    raise ValueError(f"algorithm_names length ({len(names_tuple)}) must match columns in data_matrix ({k_algorithms}).")
  if k_algorithms < 3:
    raise ValueError(f"Friedman/Nemenyi analysis requires at least 3 algorithms (k >= 3). Got k={k_algorithms}.")
  if n_blocks < 2:
    raise ValueError(f"Friedman/Nemenyi analysis requires at least 2 paired blocks (N >= 2). Got N={n_blocks}.")
  if not np.all(np.isfinite(mat)):
    raise ValueError("data_matrix contains non-finite (NaN or Inf) values.")

  ranks = compute_mean_ranks(mat, names_tuple, higher_is_better=higher_is_better)
  rank_sum = float(sum(ranks.values()))
  expected_sum = (k_algorithms * (k_algorithms + 1)) / 2.0
  if not np.isclose(rank_sum, expected_sum, atol=1e-5):
    raise ValueError(
      f"Rank conservation law violated: sum of average ranks={rank_sum:.6f}, expected={expected_sum:.6f}."
    )

  analyzer = StatisticalAnalyzer(alpha=alpha)
  cols = [mat[:, j] for j in range(k_algorithms)]
  friedman_res = analyzer.friedman_test(cols)
  nemenyi_res = analyzer.nemenyi_posthoc(cols, names_tuple, higher_is_better=higher_is_better)

  p_mat = np.asarray(nemenyi_res["p_values_matrix"], dtype=float)
  p_rows = tuple(tuple(float(v) for v in row) for row in p_mat)
  sig_pairs = tuple((names_tuple[i], names_tuple[j], float(p_val)) for i, j, p_val in nemenyi_res["significant_pairs"])
  cd = float(nemenyi_res["critical_distance"])
  cliques = _compute_maximal_cliques(ranks, cd)

  return OmnibusTestResult(
    k_algorithms=k_algorithms,
    n_blocks=n_blocks,
    algorithm_names=names_tuple,
    chi2_stat=float(friedman_res["statistic"]),
    chi2_pvalue=float(friedman_res["p_value"]),
    friedman_f=float(friedman_res["statistic_f"]),
    df1=int(friedman_res["df1"]),
    df2=int(friedman_res["df2"]),
    friedman_pvalue=float(friedman_res["p_value_f"]),
    alpha=alpha,
    omnibus_significant=bool(friedman_res["significant"]),
    low_power=bool(n_blocks < _OMNIBUS_LOW_POWER_MIN_BLOCKS),
    ranks=ranks,
    rank_sum=rank_sum,
    critical_difference=cd,
    cliques=cliques,
    nemenyi_pvalues=p_rows,
    significant_pairs=sig_pairs,
  )


__all__ = [
  "POSTHOCS_AVAILABLE",
  "OmnibusTestResult",
  "StatisticalAnalyzer",
  "StatisticalSummary",
  "build_feasibility_aware_scores",
  "compute_iman_davenport_f",
  "compute_mean_ranks",
  "interpret_rank_biserial",
  "run_friedman_nemenyi_analysis",
]
