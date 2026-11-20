"""
Statistical analysis for benchmark results.

Implements the statistical methodology from Section 3.4 (Protocolo de Análise Estatística), including:
- Normality diagnostics (Shapiro-Wilk, descriptive only)
- Hypothesis testing (Wilcoxon signed-rank default)
- Effect size calculation (matched-pairs rank-biserial correlation)
- Distribution-free Hodges-Lehmann and bootstrap confidence intervals
- Multiple comparison correction (Holm-Bonferroni)
- Friedman test for multiple algorithm comparison
- Nemenyi post-hoc test for pairwise rankings
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
from scipy import stats

from benchmarking.analysis.stats_utils import compute_nemenyi_critical_distance
from data_models.records import NemenyiResult

POSTHOCS_AVAILABLE = find_spec("scikit_posthocs") is not None


@dataclass
class StatisticalSummary:
  """
  Statistical summary for a pair of benchmark configurations.

  Contains all statistics needed for Section 4 (RESULTADOS) reporting.

  Attributes:
      comparison_label: Human-readable comparison (e.g., "SA: NumPy vs CuPy")
      metric_name: What was measured (e.g., "runtime_seconds", "final_tour_cost")
      sample_size: Number of paired observations (n)
      mean_a: Mean for configuration A
      mean_b: Mean for configuration B
      std_a: Standard deviation for configuration A
      std_b: Standard deviation for configuration B
      ci_95_a: 95% CI for A as (low, high)
      ci_95_b: 95% CI for B as (low, high)
      mean_difference: Mean of (A - B)
      ci_95_difference: 95% CI for difference as (low, high)
      p_value: Statistical significance
      effect_size: Effect size measure (e.g., rank-biserial correlation for Wilcoxon)
      test_used: Which test was applied (e.g., "paired_t_test", "wilcoxon")
      normality_p_value_a: Shapiro-Wilk p-value for A
      normality_p_value_b: Shapiro-Wilk p-value for B
      test_statistic: Primary test statistic (e.g., Wilcoxon W)
      df1: Numerator degrees of freedom when applicable, else None
      df2: Denominator degrees of freedom when applicable, else None
      test_justification: Rationale for test selection (e.g., "demsar2006_wilcoxon_default")
      zero_differences_count: Number of paired observations where diff == 0
      effective_sample_size: Non-zero difference count available to Wilcoxon
      insufficient_power_warning: True when zero_differences_count exceeds n / 2
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
  mean_difference: float  # mean(A - B)
  ci_95_difference: tuple[float, float]
  p_value: float
  effect_size: float
  test_used: str
  normality_p_value_a: float
  normality_p_value_b: float
  test_statistic: float
  df1: float | int | None
  df2: float | int | None
  test_justification: str = "wilcoxon_default"
  zero_differences_count: int = 0
  effective_sample_size: int = 0
  insufficient_power_warning: bool = False

  def get_speedup(self) -> float:
    """
    Calculate speedup ratio (A / B).

    For runtime metrics, this gives GPU speedup if A=CPU, B=GPU.
    Returns 0.0 if B is zero.

    Note:
        0.0 is a sentinel for an undefined ratio, not a measurement: 0.0 is also a legal
        speedup value, so the two are indistinguishable. Do not feed this result into an
        arithmetic or geometric mean -- a single sentinel would drag the aggregate to zero.
        Report a placeholder (e.g. an em dash) instead of averaging it.
    """
    if self.mean_b == 0.0:
      return 0.0
    return self.mean_a / self.mean_b

  def get_speedup_ci_95(self, bootstrap_samples: np.ndarray) -> tuple[float, float]:
    """
    Calculate 95% CI for speedup using bootstrap samples.

    Args:
        bootstrap_samples: Array of (A/B) ratios from bootstrap resampling

    Returns:
        (low, high) 95% CI for speedup
    """
    return (
      float(np.percentile(bootstrap_samples, 2.5)),
      float(np.percentile(bootstrap_samples, 97.5)),
    )

  def is_significant(self, alpha: float = 0.05) -> bool:
    """Check if result is statistically significant at level alpha."""
    return self.p_value < alpha


def compute_iman_davenport_f(
  chi2: float,
  n_observations: int,
  k_algorithms: int,
) -> tuple[float, int, int, float]:
  """Compute the Iman-Davenport (1980) F_F correction from Friedman's chi-square.

  Converts Friedman's chi2 into F_F = ((N - 1) * chi2) / (N * (k - 1) - chi2)
  with df1 = k - 1 and df2 = (k - 1) * (N - 1), returning (F_F, df1, df2, p_value_f).
  Handles N < 2, non-finite chi2, and perfect rank concordance (denominator <= 1e-12).
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
  data_sets: list[np.ndarray] | np.ndarray,
  algorithm_names: list[str],
  *,
  higher_is_better: bool = False,
) -> dict[str, float]:
  """Compute average per-instance ranks (1 = best) for each algorithm."""
  mat = np.asarray(data_sets, dtype=float) if isinstance(data_sets, np.ndarray) else np.column_stack(data_sets)
  if mat.ndim != 2 or mat.shape[1] != len(algorithm_names):
    raise ValueError(
      f"Expected 2D matrix with {len(algorithm_names)} columns matching algorithm_names; got shape {mat.shape}."
    )
  multiplier = -1.0 if higher_is_better else 1.0
  ranked = np.apply_along_axis(stats.rankdata, 1, multiplier * mat, method="average")
  avg_ranks = np.mean(ranked, axis=0)
  return {name: float(avg_ranks[i]) for i, name in enumerate(algorithm_names)}


class StatisticalAnalyzer:
  """
  Performs statistical analysis on benchmark results.

  Implements methodology from Section 3.4 (Protocolo de Análise Estatística):
  1. Normality testing (Shapiro-Wilk)
  2. Test selection based on normality
  3. Effect size calculation
  4. Confidence interval construction
  5. Multiple comparison correction
  """

  def __init__(self, alpha: float = 0.05) -> None:
    """
    Initialize analyzer.

    Args:
        alpha: Significance level for hypothesis tests (default: 0.05)
    """
    self.alpha = alpha

  def test_normality(self, data: np.ndarray) -> tuple[float, bool]:
    """
    Test data for normality using Shapiro-Wilk test.

    Provides descriptive normality diagnostics per Section 3.4 (Protocolo de Análise
    Estatística). Note that paired comparisons use Wilcoxon signed-rank tests regardless
    of normality outcomes (DP13-4 / BUG-60).

    Args:
        data: 1D array of observations (n >= 3)

    Returns:
        (p_value, is_normal) where is_normal = (p >= alpha)
    """
    if len(data) < 3:
      return (0.0, False)

    if np.std(data) < 1e-10:
      return (1.0, True)

    _, p_value = stats.shapiro(data)
    is_normal = p_value >= self.alpha
    return (float(p_value), is_normal)

  def compute_hodges_lehmann(
    self,
    differences: np.ndarray,
    confidence: float = 0.95,
  ) -> tuple[float, tuple[float, float]]:
    """Compute Hodges-Lehmann location estimator and distribution-free confidence interval.

    Calculates Walsh averages `(d_i + d_j) / 2` for `1 <= i <= j <= n`.
    The point estimate is the median of the Walsh averages.
    The confidence interval `(low, high)` is derived from the critical rank of
    the Wilcoxon signed-rank distribution with normal approximation.
    """
    n = len(differences)
    if n == 0:
      return 0.0, (0.0, 0.0)
    if n == 1:
      val = float(differences[0])
      return val, (val, val)

    i_idx, j_idx = np.triu_indices(n)
    walsh = (differences[i_idx] + differences[j_idx]) / 2.0
    walsh_sorted = np.sort(walsh)
    m = len(walsh_sorted)

    point_estimate = float(np.median(walsh_sorted))

    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    e_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    k = int(np.floor(e_w - z * np.sqrt(var_w)))
    k = max(1, min(k, m))

    low = float(walsh_sorted[k - 1])
    high = float(walsh_sorted[m - k])
    return point_estimate, (low, high)

  def rank_biserial_correlation(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
  ) -> float:
    """Compute matched-pairs rank-biserial correlation effect size (Kerby 2014).

    For differences `D = A - B`, ranks absolute non-zero differences `|D|`.
    `r = (W_+ - W_-) / (W_+ + W_-)` where `W_+` is sum of ranks with `D > 0` and
    `W_-` is sum of ranks with `D < 0`. Range is `[-1.0, 1.0]`.
    Returns 0.0 if all differences are zero.
    """
    diffs = data_a - data_b
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) == 0:
      return 0.0
    abs_diffs = np.abs(nonzero)
    ranks = stats.rankdata(abs_diffs)
    pos_ranks = float(np.sum(ranks[nonzero > 0]))
    neg_ranks = float(np.sum(ranks[nonzero < 0]))
    total = pos_ranks + neg_ranks
    if total == 0.0:
      return 0.0
    return float((pos_ranks - neg_ranks) / total)

  def paired_comparison(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
    label: str,
    metric_name: str,
  ) -> StatisticalSummary:
    """
    Perform paired statistical comparison (e.g., CPU vs GPU).

    Implements Section 3.4 (Protocolo de Análise Estatística):
    1. Descriptive normality assessment (Shapiro-Wilk)
    2. Non-parametric paired hypothesis test (Wilcoxon signed-rank test in all cases, DP13-4)
    3. Effect size (rank-biserial correlation) and distribution-free Hodges-Lehmann CI

    Args:
        data_a: Measurements from configuration A (e.g., CPU runtimes)
        data_b: Measurements from configuration B (e.g., GPU runtimes)
        label: Human-readable comparison label
        metric_name: What was measured (e.g., "runtime_seconds")

    Returns:
        StatisticalSummary with all test results

    Raises:
        ValueError: If arrays have different lengths
    """
    if len(data_a) != len(data_b):
      raise ValueError(f"Arrays must have same length: {len(data_a)} != {len(data_b)}")

    n: int = len(data_a)

    if n < 2:
      mean_a = float(np.mean(data_a)) if n == 1 else 0.0
      mean_b = float(np.mean(data_b)) if n == 1 else 0.0
      mean_diff = mean_a - mean_b
      zero_diffs = int(data_a[0] == data_b[0]) if n == 1 else 0
      return StatisticalSummary(
        comparison_label=label,
        metric_name=metric_name,
        sample_size=n,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=0.0,
        std_b=0.0,
        ci_95_a=(mean_a, mean_a),
        ci_95_b=(mean_b, mean_b),
        mean_difference=mean_diff,
        ci_95_difference=(mean_diff, mean_diff),
        p_value=1.0,
        effect_size=0.0,
        test_used="no_test_needed",
        normality_p_value_a=1.0,
        normality_p_value_b=1.0,
        test_statistic=0.0,
        df1=None,
        df2=None,
        test_justification="insufficient_sample_size",
        zero_differences_count=zero_diffs,
        effective_sample_size=n - zero_diffs,
        insufficient_power_warning=True,
      )

    # Check for zero variance across differences
    differences = data_a - data_b
    variance_threshold = 1e-10
    mean_diff = float(np.mean(differences))
    zero_diffs = int(np.sum(differences == 0.0))
    effective_n = n - zero_diffs
    insufficient_power = zero_diffs > (n // 2)

    if np.std(differences) < variance_threshold:
      mean_a = float(np.mean(data_a))
      mean_b = float(np.mean(data_b))
      eff_size = 0.0 if mean_diff == 0.0 else (1.0 if mean_diff > 0.0 else -1.0)
      justification = "identical_samples_no_test_needed" if mean_diff == 0.0 else "constant_difference_no_test_needed"
      # Zero variance makes the paired t-test/Cohen's d undefined (division by zero).
      # A constant offset is different from no offset, so only a zero mean difference
      # collapses to the identity summary; any nonzero constant is fully separated.
      return StatisticalSummary(
        comparison_label=label,
        metric_name=metric_name,
        sample_size=n,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=0.0,
        std_b=0.0,
        ci_95_a=(mean_a, mean_a),
        ci_95_b=(mean_b, mean_b),
        mean_difference=mean_diff,
        ci_95_difference=(mean_diff, mean_diff),
        p_value=1.0 if mean_diff == 0.0 else 0.0,
        effect_size=eff_size,
        test_used="no_test_needed",
        normality_p_value_a=1.0,
        normality_p_value_b=1.0,
        test_statistic=0.0,
        df1=None,
        df2=None,
        test_justification=justification,
        zero_differences_count=zero_diffs,
        effective_sample_size=effective_n,
        insufficient_power_warning=insufficient_power,
      )

    if insufficient_power:
      import logging

      logging.warning(
        "More than half of paired differences are zero (%d/%d); "
        "effective n=%d yields insufficient power for Wilcoxon test.",
        zero_diffs,
        n,
        effective_n,
      )

    p_norm_a, _ = self.test_normality(data_a)
    p_norm_b, _ = self.test_normality(data_b)

    mean_a = float(np.mean(data_a))
    mean_b = float(np.mean(data_b))
    std_a = float(np.std(data_a, ddof=1))
    std_b = float(np.std(data_b, ddof=1))

    se_a = std_a / np.sqrt(n)
    se_b = std_b / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)

    ci_95_a = (mean_a - t_critical * se_a, mean_a + t_critical * se_a)
    ci_95_b = (mean_b - t_critical * se_b, mean_b + t_critical * se_b)

    # Always use Wilcoxon signed-ranks for paired comparison per Demšar (2006) protocol (DP13-4 / BUG-60)
    test_name = "wilcoxon_signed_rank"
    test_justification = "demsar2006_wilcoxon_default"
    res = stats.wilcoxon(data_a, data_b, zero_method="zsplit")
    p_value = float(res.pvalue)

    # Effect size: matched-pairs rank-biserial correlation (Kerby 2014)
    effect_size = self.rank_biserial_correlation(data_a, data_b)

    # Distribution-free Hodges-Lehmann confidence interval
    _, ci_95_diff = self.compute_hodges_lehmann(differences, confidence=1.0 - self.alpha)

    return StatisticalSummary(
      comparison_label=label,
      metric_name=metric_name,
      sample_size=n,
      mean_a=mean_a,
      mean_b=mean_b,
      std_a=std_a,
      std_b=std_b,
      ci_95_a=ci_95_a,
      ci_95_b=ci_95_b,
      mean_difference=mean_diff,
      ci_95_difference=ci_95_diff,
      p_value=p_value,
      effect_size=effect_size,
      test_used=test_name,
      normality_p_value_a=p_norm_a,
      normality_p_value_b=p_norm_b,
      test_statistic=float(res.statistic),
      df1=None,
      df2=None,
      test_justification=test_justification,
      zero_differences_count=zero_diffs,
      effective_sample_size=effective_n,
      insufficient_power_warning=insufficient_power,
    )

  def cohens_d(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired samples.

    d = (μ₁ - μ₂) / sigma_pooled
    where sigma_pooled = sqrt((sigma₁² + sigma₂²) / 2)
    """
    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    std_a = np.std(data_a, ddof=1)
    std_b = np.std(data_b, ddof=1)

    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    if pooled_std == 0.0:
      return 0.0

    return float((mean_a - mean_b) / pooled_std)

  def bootstrap_ci(
    self,
    data: np.ndarray,
    statistic_func: Callable[[Any], float],
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
  ) -> tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for any statistic.

    Following Section 3.4 (Protocolo de Análise Estatística), uses percentile method from
    Efron & Tibshirani (1993).
    """
    alpha = 1.0 - confidence_level
    bootstrap_samples = np.zeros(n_resamples)

    for i in range(n_resamples):
      resample = np.random.choice(data, size=len(data), replace=True)
      bootstrap_samples[i] = statistic_func(resample)

    ci_low = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))

    return (float(ci_low), float(ci_high), bootstrap_samples)

  def holm_bonferroni_correction(self, p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Following Section 3.4 (Protocolo de Análise Estatística) Step 4, controls family-wise error rate
    while being less conservative than Bonferroni.
    """
    m = len(p_values)
    if m == 0:
      return []

    indexed_p = list(enumerate(p_values))
    indexed_p.sort(key=lambda x: x[1])

    results = []
    for i, (original_idx, p_val) in enumerate(indexed_p):
      threshold = alpha / (m - i)
      is_significant = p_val <= threshold
      results.append((original_idx, p_val, is_significant))

      if not is_significant:
        for j in range(i + 1, len(indexed_p)):
          orig_idx, p_v = indexed_p[j]
          results.append((orig_idx, p_v, False))
        break

    return results

  def friedman_test(self, data_sets: list[np.ndarray]) -> dict[str, Any]:
    """
    Perform Friedman non-parametric omnibus test with Iman-Davenport F_F correction.

    Following Section 3.4 / Demšar (2006): computes both chi2_F and the less
    conservative Iman-Davenport F_F statistic, deciding significance via p_value_f < alpha.
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

    # Guard against insufficient samples (N < 2) or complete intra-block ranking ties
    # (where every row has identical values across all treatments, yielding Q = 0.0, p = 1.0).
    # Detecting exact zero range across every block avoids 0/0 scalar division in SciPy tie correction
    # without scale-dependent variance thresholds or blanket warning suppression.
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
    }

  def nemenyi_posthoc(
    self,
    data_sets: list[np.ndarray],
    algorithm_names: list[str] | None = None,
  ) -> NemenyiResult:
    """
    Perform Nemenyi post-hoc test after significant Friedman result.
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

    if algorithm_names is None:
      algorithm_names = [f"Alg{i + 1}" for i in range(k)]
    elif len(algorithm_names) != k:
      raise ValueError(f"algorithm_names length ({len(algorithm_names)}) must match data_sets length ({k})")
    data_array = np.column_stack(data_sets)
    from scipy.stats import rankdata, studentized_range

    if POSTHOCS_AVAILABLE:
      import scikit_posthocs as sp

      p_matrix = sp.posthoc_nemenyi_friedman(data_array)
      p_values_matrix = p_matrix.values
    else:
      ranks = np.apply_along_axis(rankdata, 1, data_array)
      mean_ranks = np.mean(ranks, axis=0)
      se = np.sqrt(k * (k + 1) / (12.0 * n_instances))
      p_values_matrix = np.ones((k, k), dtype=np.float64)
      for i in range(k):
        for j in range(i + 1, k):
          q_stat = abs(mean_ranks[i] - mean_ranks[j]) / se
          p_val = float(studentized_range.sf(q_stat, k, np.inf))
          p_values_matrix[i, j] = p_val
          p_values_matrix[j, i] = p_val

    significant_pairs = []
    for i in range(k):
      for j in range(i + 1, k):
        p_val = p_values_matrix[i, j]
        if p_val < self.alpha:
          significant_pairs.append((i, j, float(p_val)))

    critical_distance = compute_nemenyi_critical_distance(k, n_instances, self.alpha)

    return {
      "p_values_matrix": p_values_matrix,
      "significant_pairs": significant_pairs,
      "algorithm_names": algorithm_names,
      "critical_distance": float(critical_distance),
    }
