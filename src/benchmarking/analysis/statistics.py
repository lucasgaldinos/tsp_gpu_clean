"""
Statistical analysis for benchmark results.

Implements the statistical methodology from Section 3.5.3, including:
- Normality testing (Shapiro-Wilk)
- Hypothesis testing (paired t-test, Wilcoxon)
- Effect size calculation (Cohen's d)
- Bootstrap confidence intervals
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
      effect_size: Cohen's d (for normally distributed data)
      test_used: Which test was applied (e.g., "paired_t_test", "wilcoxon")
      normality_p_value_a: Shapiro-Wilk p-value for A
      normality_p_value_b: Shapiro-Wilk p-value for B
      is_normal: Whether both distributions passed normality test
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
  is_normal: bool

  def get_speedup(self) -> float:
    """
    Calculate speedup ratio (A / B).

    For runtime metrics, this gives GPU speedup if A=CPU, B=GPU.
    Returns 0.0 if B is zero.
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


class StatisticalAnalyzer:
  """
  Performs statistical analysis on benchmark results.

  Implements methodology from Section 3.5.3:
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

    Following Section 3.5.3 Step 1, tests null hypothesis that data
    comes from normal distribution.

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

  def paired_comparison(
    self,
    data_a: np.ndarray,
    data_b: np.ndarray,
    label: str,
    metric_name: str,
  ) -> StatisticalSummary:
    """
    Perform paired statistical comparison (e.g., CPU vs GPU).

    Implements Section 3.5.3 Steps 1-3:
    1. Test normality (Shapiro-Wilk)
    2. Choose appropriate test (t-test vs Wilcoxon)
    3. Calculate effect size and confidence intervals

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

    n = len(data_a)

    if n < 2:
      mean_a = float(np.mean(data_a)) if n == 1 else 0.0
      mean_b = float(np.mean(data_b)) if n == 1 else 0.0
      mean_diff = mean_a - mean_b
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
        is_normal=True,
      )

    # Check for zero variance across differences
    differences = data_a - data_b
    variance_threshold = 1e-10

    if np.std(differences) < variance_threshold:
      mean_a = float(np.mean(data_a))
      mean_b = float(np.mean(data_b))
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
        mean_difference=0.0,
        ci_95_difference=(0.0, 0.0),
        p_value=1.0,
        effect_size=0.0,
        test_used="no_test_needed",
        normality_p_value_a=1.0,
        normality_p_value_b=1.0,
        is_normal=True,
      )

    p_norm_a, is_normal_a = self.test_normality(data_a)
    p_norm_b, is_normal_b = self.test_normality(data_b)
    both_normal = is_normal_a and is_normal_b

    mean_a = float(np.mean(data_a))
    mean_b = float(np.mean(data_b))
    std_a = float(np.std(data_a, ddof=1))
    std_b = float(np.std(data_b, ddof=1))

    se_a = std_a / np.sqrt(n)
    se_b = std_b / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)

    ci_95_a = (mean_a - t_critical * se_a, mean_a + t_critical * se_a)
    ci_95_b = (mean_b - t_critical * se_b, mean_b + t_critical * se_b)

    differences = data_a - data_b
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    se_diff = std_diff / np.sqrt(n)
    ci_95_diff = (
      mean_diff - t_critical * se_diff,
      mean_diff + t_critical * se_diff,
    )

    if both_normal:
      _, p_value = stats.ttest_rel(data_a, data_b)
      test_name = "paired_t_test"
    else:
      _, p_value = stats.wilcoxon(data_a, data_b, zero_method="zsplit")
      test_name = "wilcoxon_signed_rank"

    effect_size = self.cohens_d(data_a, data_b)

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
      p_value=float(p_value),
      effect_size=effect_size,
      test_used=test_name,
      normality_p_value_a=p_norm_a,
      normality_p_value_b=p_norm_b,
      is_normal=both_normal,
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

    Following Section 3.5.3, uses percentile method from
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

  def holm_bonferroni_correction(
    self, p_values: list[float], alpha: float = 0.05
  ) -> list[tuple[int, float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Following Section 3.5.3 Step 4, controls family-wise error rate
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

  def friedman_test(self, data_sets: list[np.ndarray]) -> dict[str, float]:
    """
    Perform Friedman non-parametric test to compare k related samples.

    Following Section 3.5.3 Step 5.
    """
    if len(set(len(d) for d in data_sets)) > 1:
      raise ValueError(
        "All data_sets must have the same number of observations (paired samples). "
        f"Got lengths: {[len(d) for d in data_sets]}"
      )

    if len(data_sets) < 3:
      raise ValueError(
        f"Friedman test requires at least 3 algorithms (k >= 3). Got {len(data_sets)} algorithms."
      )

    all_data = np.concatenate(data_sets)
    variance_threshold = 1e-10

    if np.std(all_data) < variance_threshold or len(data_sets[0]) < 2:
      return {
        "statistic": 0.0,
        "p_value": 1.0,
        "significant": False,
        "post_hoc_required": False,
      }

    statistic, p_value = stats.friedmanchisquare(*data_sets)
    if np.isnan(statistic) or np.isnan(p_value):
      return {
        "statistic": 0.0,
        "p_value": 1.0,
        "significant": False,
        "post_hoc_required": False,
      }

    is_significant = bool(p_value < self.alpha)

    return {
      "statistic": float(statistic),
      "p_value": float(p_value),
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
    if len(set(len(d) for d in data_sets)) > 1:
      raise ValueError(
        f"All data_sets must have the same number of observations. Got lengths: {[len(d) for d in data_sets]}"
      )

    k = len(data_sets)
    if algorithm_names is None:
      algorithm_names = [f"Alg{i + 1}" for i in range(k)]
    elif len(algorithm_names) != k:
      raise ValueError(
        f"algorithm_names length ({len(algorithm_names)}) must match data_sets length ({k})"
      )

    n_instances = len(data_sets[0])
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

    q_critical = studentized_range.ppf(1 - self.alpha, k, np.inf)
    critical_distance = q_critical * np.sqrt(k * (k + 1) / (6 * n_instances))

    return {
      "p_values_matrix": p_values_matrix,
      "significant_pairs": significant_pairs,
      "algorithm_names": algorithm_names,
      "critical_distance": float(critical_distance),
    }

