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

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import stats

try:
    import scikit_posthocs as sp

    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False


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

    Example:
        >>> summary = StatisticalSummary(
        ...     comparison_label="SA: NumPy vs CuPy",
        ...     metric_name="runtime_seconds",
        ...     sample_size=30,
        ...     mean_a=12.5,
        ...     mean_b=1.3,
        ...     p_value=0.001,
        ...     effect_size=2.45,
        ...     test_used="wilcoxon"
        ... )
    """

    comparison_label: str
    metric_name: str
    sample_size: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    ci_95_a: Tuple[float, float]
    ci_95_b: Tuple[float, float]
    mean_difference: float  # mean(A - B)
    ci_95_difference: Tuple[float, float]
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

    def get_speedup_ci_95(self, bootstrap_samples: np.ndarray) -> Tuple[float, float]:
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

    def __init__(self, alpha: float = 0.05):
        """
        Initialize analyzer.

        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha

    def test_normality(self, data: np.ndarray) -> Tuple[float, bool]:
        """
        Test data for normality using Shapiro-Wilk test.

        Following Section 3.5.3 Step 1, tests null hypothesis that data
        comes from normal distribution.

        Args:
            data: 1D array of observations (n >= 3)

        Returns:
            (p_value, is_normal) where is_normal = (p >= alpha)

        Example:
            >>> data = np.array([1.2, 1.5, 1.3, 1.4, 1.6, ...])  # n=30
            >>> p_value, is_normal = analyzer.test_normality(data)
            >>> if is_normal:
            ...     print("Use parametric tests (t-test, ANOVA)")
            ... else:
            ...     print("Use non-parametric tests (Wilcoxon, Kruskal-Wallis)")
        """
        if len(data) < 3:
            # Shapiro-Wilk requires n >= 3
            return (0.0, False)

        statistic, p_value = stats.shapiro(data)
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

        Example:
            >>> cpu_times = np.array([12.5, 11.8, 13.2, ...])  # n=30
            >>> gpu_times = np.array([1.3, 1.2, 1.4, ...])     # n=30
            >>> summary = analyzer.paired_comparison(
            ...     cpu_times, gpu_times,
            ...     "SA: NumPy vs CuPy",
            ...     "runtime_seconds"
            ... )
            >>> print(f"p-value: {summary.p_value:.4f}")
            >>> print(f"Effect size: {summary.effect_size:.2f}")
            >>> print(f"Test used: {summary.test_used}")
        """
        if len(data_a) != len(data_b):
            raise ValueError(
                f"Arrays must have same length: {len(data_a)} != {len(data_b)}"
            )

        n = len(data_a)

        # Check for zero variance (degenerate case: all values identical)
        # This occurs when all algorithms achieve optimal solution
        differences = data_a - data_b
        variance_threshold = 1e-10
        
        if np.std(differences) < variance_threshold:
            # All measurements identical - no statistical test needed
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
                p_value=1.0,  # No difference
                effect_size=0.0,
                test_used="no_test_needed",
                normality_p_value_a=1.0,
                normality_p_value_b=1.0,
                is_normal=True,
            )

        # Step 1: Test normality for both samples
        p_norm_a, is_normal_a = self.test_normality(data_a)
        p_norm_b, is_normal_b = self.test_normality(data_b)
        both_normal = is_normal_a and is_normal_b

        # Calculate basic statistics
        mean_a = float(np.mean(data_a))
        mean_b = float(np.mean(data_b))
        std_a = float(np.std(data_a, ddof=1))
        std_b = float(np.std(data_b, ddof=1))

        # Calculate 95% CI for means (using t-distribution)
        se_a = std_a / np.sqrt(n)
        se_b = std_b / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n - 1)  # 97.5th percentile for two-tailed

        ci_95_a = (mean_a - t_critical * se_a, mean_a + t_critical * se_a)
        ci_95_b = (mean_b - t_critical * se_b, mean_b + t_critical * se_b)

        # Calculate differences
        differences = data_a - data_b
        mean_diff = float(np.mean(differences))
        std_diff = float(np.std(differences, ddof=1))
        se_diff = std_diff / np.sqrt(n)
        ci_95_diff = (
            mean_diff - t_critical * se_diff,
            mean_diff + t_critical * se_diff,
        )

        # Step 2: Choose appropriate test
        if both_normal:
            # Parametric: Paired t-test
            statistic, p_value = stats.ttest_rel(data_a, data_b)
            test_name = "paired_t_test"
        else:
            # Non-parametric: Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data_a, data_b)
            test_name = "wilcoxon_signed_rank"

        # Step 3: Calculate effect size (Cohen's d)
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

        Following Section 3.5.3 footnote [^cohens_d]:
        d = (μ₁ - μ₂) / σ_pooled
        where σ_pooled = sqrt((σ₁² + σ₂²) / 2)

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large

        Args:
            data_a: First sample
            data_b: Second sample

        Returns:
            Cohen's d effect size

        Example:
            >>> cpu_times = np.array([12.0, 11.5, 12.3])
            >>> gpu_times = np.array([1.2, 1.1, 1.3])
            >>> d = analyzer.cohens_d(cpu_times, gpu_times)
            >>> print(f"Effect size: {d:.2f} (large)")
        """
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        std_a = np.std(data_a, ddof=1)
        std_b = np.std(data_b, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

        if pooled_std == 0.0:
            return 0.0

        return float((mean_a - mean_b) / pooled_std)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_func: callable,
        confidence_level: float = 0.95,
        n_resamples: int = 10000,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Calculate bootstrap confidence interval for any statistic.

        Following Section 3.5.3, uses percentile method from
        Efron & Tibshirani (1993).

        Args:
            data: Original data array
            statistic_func: Function to calculate statistic (e.g., np.mean)
            confidence_level: CI level (default: 0.95)
            n_resamples: Bootstrap iterations (default: 10000)

        Returns:
            (ci_low, ci_high, bootstrap_samples)

        Example:
            >>> speedups = cpu_times / gpu_times
            >>> ci_low, ci_high, samples = analyzer.bootstrap_ci(
            ...     speedups, np.mean, confidence_level=0.95
            ... )
            >>> print(f"Mean speedup: {np.mean(speedups):.2f}x")
            >>> print(f"95% CI: [{ci_low:.2f}x, {ci_high:.2f}x]")
        """
        alpha = 1.0 - confidence_level
        bootstrap_samples = np.zeros(n_resamples)

        for i in range(n_resamples):
            # Resample with replacement
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples[i] = statistic_func(resample)

        # Percentile method
        ci_low = np.percentile(bootstrap_samples, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))

        return (float(ci_low), float(ci_high), bootstrap_samples)

    def holm_bonferroni_correction(
        self, p_values: List[float], alpha: float = 0.05
    ) -> List[Tuple[int, float, bool]]:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.

        Following Section 3.5.3 Step 4, controls family-wise error rate
        while being less conservative than Bonferroni.

        Procedure:
        1. Sort p-values in ascending order
        2. Compare p_i to α/(m-i+1) where m is number of tests
        3. Reject H₀ if p_i ≤ α/(m-i+1), stop at first non-rejection

        Args:
            p_values: List of p-values from multiple tests
            alpha: Family-wise error rate (default: 0.05)

        Returns:
            List of (original_index, p_value, is_significant) tuples

        Example:
            >>> p_vals = [0.001, 0.02, 0.03, 0.15]  # 4 comparisons
            >>> results = analyzer.holm_bonferroni_correction(p_vals)
            >>> for idx, p, sig in results:
            ...     print(f"Test {idx}: p={p:.3f}, significant={sig}")
        """
        m = len(p_values)
        if m == 0:
            return []

        # Create (index, p_value) pairs and sort by p-value
        indexed_p = list(enumerate(p_values))
        indexed_p.sort(key=lambda x: x[1])

        results = []
        for i, (original_idx, p_val) in enumerate(indexed_p):
            # Holm threshold: α / (m - i)
            threshold = alpha / (m - i)
            is_significant = p_val <= threshold
            results.append((original_idx, p_val, is_significant))

            # Stop rejecting after first non-rejection
            if not is_significant:
                # Mark remaining as non-significant
                for j in range(i + 1, len(indexed_p)):
                    orig_idx, p_v = indexed_p[j]
                    results.append((orig_idx, p_v, False))
                break

        return results

    def friedman_test(self, data_sets: List[np.ndarray]) -> Dict[str, float]:
        """
        Perform Friedman non-parametric test to compare k related samples.

        The Friedman test is used when comparing the performance (e.g., final cost,
        optimality gap) of multiple algorithms (k > 2) across the same set of problem
        instances. It's the non-parametric alternative to repeated measures ANOVA.

        Following Section 3.5.3 Step 5, this test answers: "Do at least two algorithms
        differ significantly in their performance across instances?"

        If null hypothesis is rejected (p < α), proceed with post-hoc test (Nemenyi)
        to identify which specific pairs differ significantly.

        Args:
            data_sets: List of arrays, each containing results for one algorithm
                      across all problem instances (paired/related samples).
                      Example: [algo1_gaps, algo2_gaps, algo3_gaps, algo4_gaps]
                      where each array has shape (n_instances,)

        Returns:
            Dictionary with:
                - 'statistic': Friedman test statistic (chi-square distribution)
                - 'p_value': Probability of observing results under null hypothesis
                - 'significant': Boolean indicating if p < alpha
                - 'post_hoc_required': Whether Nemenyi test should be performed

        Raises:
            ValueError: If data sets have different lengths (must be paired)

        Example:
            >>> # Compare 4 algorithms on 12 instances
            >>> naive_gaps = np.array([5.2, 3.8, 7.1, ...])  # 12 instances
            >>> optimized_gaps = np.array([2.1, 1.5, 3.2, ...])
            >>> fullgpu_gaps = np.array([1.8, 1.2, 2.9, ...])
            >>> cpu_gaps = np.array([5.5, 4.1, 7.3, ...])
            >>> result = analyzer.friedman_test([
            ...     naive_gaps, optimized_gaps, fullgpu_gaps, cpu_gaps
            ... ])
            >>> print(f"p-value: {result['p_value']:.4f}")
            >>> if result['post_hoc_required']:
            ...     print("Perform Nemenyi post-hoc test")
        """
        # Validate input
        if len(set(len(d) for d in data_sets)) > 1:
            raise ValueError(
                "All data_sets must have the same number of observations (paired samples). "
                f"Got lengths: {[len(d) for d in data_sets]}"
            )

        if len(data_sets) < 3:
            raise ValueError(
                "Friedman test requires at least 3 algorithms (k >= 3). "
                f"Got {len(data_sets)} algorithms."
            )

        # Check for zero variance across all algorithms (degenerate case)
        # This occurs when all algorithms achieve identical results
        all_data = np.concatenate(data_sets)
        variance_threshold = 1e-10
        
        if np.std(all_data) < variance_threshold:
            # All algorithms produced identical results - no test needed
            return {
                "statistic": 0.0,
                "p_value": 1.0,  # No difference
                "significant": False,
                "post_hoc_required": False,
            }

        # Perform Friedman test
        # scipy.stats.friedmanchisquare expects *args, not a list
        statistic, p_value = stats.friedmanchisquare(*data_sets)

        is_significant = p_value < self.alpha

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": is_significant,
            "post_hoc_required": is_significant,
        }

    def nemenyi_posthoc(
        self,
        data_sets: List[np.ndarray],
        algorithm_names: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Perform Nemenyi post-hoc test after significant Friedman result.

        The Nemenyi test is used to identify which specific pairs of algorithms
        differ significantly after the Friedman test rejects the null hypothesis.
        It's analogous to Tukey's HSD for parametric ANOVA.

        Following Section 3.5.3 Step 5, this test controls family-wise error rate
        when making all possible pairwise comparisons.

        Args:
            data_sets: Same list of arrays used in Friedman test
            algorithm_names: Optional list of algorithm names for readable output
                           (default: ["Alg1", "Alg2", ..., "AlgK"])

        Returns:
            Dictionary with:
                - 'p_values_matrix': 2D array of pairwise p-values (k × k)
                - 'significant_pairs': List of (i, j, p_value) tuples for significant pairs
                - 'algorithm_names': List of algorithm names used
                - 'critical_distance': Nemenyi critical distance at alpha level

        Raises:
            ImportError: If scikit-posthocs not installed
            ValueError: If data sets have different lengths

        Example:
            >>> # After significant Friedman test
            >>> result = analyzer.nemenyi_posthoc(
            ...     [naive_gaps, optimized_gaps, fullgpu_gaps, cpu_gaps],
            ...     algorithm_names=["Naive", "Optimized", "FullGPU", "CPU"]
            ... )
            >>> for i, j, p_val in result['significant_pairs']:
            ...     name_i = result['algorithm_names'][i]
            ...     name_j = result['algorithm_names'][j]
            ...     print(f"{name_i} vs {name_j}: p={p_val:.4f} *")
        """
        if not POSTHOCS_AVAILABLE:
            raise ImportError(
                "scikit-posthocs is required for Nemenyi test. "
                "Install with: pip install scikit-posthocs"
            )

        # Validate input
        if len(set(len(d) for d in data_sets)) > 1:
            raise ValueError(
                "All data_sets must have the same number of observations. "
                f"Got lengths: {[len(d) for d in data_sets]}"
            )

        k = len(data_sets)
        if algorithm_names is None:
            algorithm_names = [f"Alg{i + 1}" for i in range(k)]
        elif len(algorithm_names) != k:
            raise ValueError(
                f"algorithm_names length ({len(algorithm_names)}) "
                f"must match data_sets length ({k})"
            )

        # Prepare data in format expected by scikit-posthocs
        # Need to convert to long-form DataFrame or use array directly
        n_instances = len(data_sets[0])

        # Stack data: shape (n_instances, k_algorithms)
        data_array = np.column_stack(data_sets)

        # Perform Nemenyi test
        # Returns a symmetric matrix of p-values
        import scikit_posthocs as sp

        p_matrix = sp.posthoc_nemenyi_friedman(data_array)

        # Convert to numpy array
        p_values_matrix = p_matrix.values

        # Extract significant pairs (upper triangle only to avoid duplicates)
        significant_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                p_val = p_values_matrix[i, j]
                if p_val < self.alpha:
                    significant_pairs.append((i, j, float(p_val)))

        # Calculate critical distance for interpretation
        # CD = q_α * sqrt(k(k+1) / (6*n))
        # where q_α is studentized range statistic
        from scipy.stats import studentized_range

        q_critical = studentized_range.ppf(1 - self.alpha, k, np.inf)
        critical_distance = q_critical * np.sqrt(k * (k + 1) / (6 * n_instances))

        return {
            "p_values_matrix": p_values_matrix,
            "significant_pairs": significant_pairs,
            "algorithm_names": algorithm_names,
            "critical_distance": float(critical_distance),
        }
