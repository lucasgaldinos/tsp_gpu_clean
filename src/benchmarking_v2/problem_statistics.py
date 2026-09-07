"""
Problem Statistics Module for Benchmark V2
===========================================

Performs statistical analysis on single-problem results.

This module analyzes results from multiple algorithms on a single TSP problem,
including:
- Friedman test for k-algorithm comparison
- Nemenyi post-hoc test
- Pairwise comparisons with Holm-Bonferroni correction
- Summary statistics and speedup calculations

Extracted from chapter4_validation.py lines 702-870.

Usage:
    from src.benchmarking_v2.problem_statistics import perform_statistical_analysis
    from src.benchmarking_v2.statistics import StatisticalAnalyzer
    
    analyzer = StatisticalAnalyzer()
    results = {
        "CPU": {...},  # Results dict from algorithm_runner
        "HybridOptimized": {...},
        "FullGPU": {...}
    }
    
    stats = perform_statistical_analysis(
        problem_name="berlin52",
        optimal_cost=7542.0,
        results=results,
        analyzer=analyzer
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import logging
from typing import Dict, Any, List

import numpy as np

from src.benchmarking_v2.statistics import StatisticalAnalyzer


# =============================================================================
# Helper Functions
# =============================================================================

def _get_data_field(results_dict: Dict[str, Any], field: str) -> List:
    """
    Safely extract data field with fallback for checkpoint vs direct format.
    
    Handles two formats:
    - Checkpoint format: uses raw_* prefix (raw_costs, raw_times, raw_gaps)
    - Direct format: uses direct names (final_costs, times, gaps)
    
    Args:
        results_dict: Algorithm results dictionary
        field: Field name ('costs', 'times', or 'gaps')
        
    Returns:
        List of values
        
    Example:
        >>> results = {"raw_costs": [7543, 7542, 7544], ...}
        >>> costs = _get_data_field(results, "costs")
        >>> print(costs)
        [7543, 7542, 7544]
    """
    # Checkpoint format uses raw_* prefix
    raw_field = f"raw_{field}"
    # Direct format uses final_costs, times, gaps
    direct_field = "final_costs" if field == "costs" else field
    
    return results_dict.get(raw_field, results_dict.get(direct_field, []))


def holm_bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> tuple[List[bool], List[float]]:
    """
    Apply Holm-Bonferroni multiple comparison correction.
    
    Corrects p-values for family-wise error rate when performing
    multiple pairwise comparisons.
    
    Args:
        p_values: List of p-values to correct
        alpha: Significance level (default: 0.05)
        
    Returns:
        Tuple of (reject_list, pvals_corrected) where:
        - reject_list: Boolean list indicating which hypotheses to reject
        - pvals_corrected: Adjusted p-values
        
    Example:
        >>> p_vals = [0.001, 0.02, 0.03, 0.15]
        >>> reject, corrected = holm_bonferroni_correction(p_vals)
        >>> print(reject)  # [True, True, False, False]
    """
    n = len(p_values)
    if n == 0:
        return [], []
    
    # Sort p-values with original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    
    # Holm-Bonferroni critical values
    critical_values = alpha / (n - np.arange(n))
    
    # Reject hypotheses where p < critical value
    reject_sorted = sorted_pvals < critical_values
    
    # Adjusted p-values (step-down procedure)
    # CRITICAL: Must use maximum.accumulate for monotonicity
    raw_adjusted = np.minimum((n - np.arange(n)) * sorted_pvals, 1.0)
    pvals_corrected_sorted = np.maximum.accumulate(raw_adjusted)
    
    # Restore original order
    reject = np.zeros(n, dtype=bool)
    pvals_corrected = np.zeros(n)
    reject[sorted_indices] = reject_sorted
    pvals_corrected[sorted_indices] = pvals_corrected_sorted
    
    return reject.tolist(), pvals_corrected.tolist()


# =============================================================================
# Main Analysis Function
# =============================================================================

def perform_statistical_analysis(
    problem_name: str,
    optimal_cost: float,
    results: Dict[str, Dict[str, Any]],
    analyzer: StatisticalAnalyzer,
    short_sep: str = "-" * 80
) -> Dict[str, Any]:
    """
    Perform statistical analysis on results for a single problem.
    
    Executes comprehensive statistical tests comparing multiple algorithms
    on the same problem instance:
    
    1. Friedman test (if k ≥ 3 algorithms)
    2. Nemenyi post-hoc test (if Friedman significant)
    3. Pairwise comparisons with Holm-Bonferroni correction
    4. Summary statistics and speedup calculations
    
    Args:
        problem_name: Name of TSPLIB instance (e.g., "berlin52")
        optimal_cost: Known optimal solution cost
        results: Dictionary mapping algorithm_name -> results_dict
        analyzer: StatisticalAnalyzer instance
        short_sep: Separator line for logging (default: 80 dashes)
        
    Returns:
        Dictionary containing:
            - problem_name: Problem identifier
            - optimal_cost: Known optimum
            - baseline_algorithm: Baseline for speedup calculation
            - friedman_test: Friedman test results (if applicable)
            - nemenyi_posthoc: Nemenyi results (if significant)
            - pairwise_comparisons: List of pairwise test results
            
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> results = {
        ...     "CPU": cpu_results,
        ...     "HybridOptimized": hybrid_results,
        ...     "FullGPU": fullgpu_results
        ... }
        >>> stats = perform_statistical_analysis(
        ...     "berlin52", 7542.0, results, analyzer
        ... )
        >>> print(stats["friedman_test"]["p_value"])
    """
    logging.info(f"\nStatistical Analysis: {problem_name}")
    logging.info(short_sep)
    
    algorithm_names = list(results.keys())
    
    # Guard against insufficient data
    if len(algorithm_names) < 2:
        logging.warning(
            f"Skipping statistical analysis for {problem_name}: "
            f"only {len(algorithm_names)} algorithm(s) available (need ≥2)"
        )
        return {
            "problem_name": problem_name,
            "optimal_cost": optimal_cost,
            "error": "insufficient_data",
            "available_algorithms": algorithm_names,
        }
    
    cost_data = [
        np.array(_get_data_field(results[name], "costs"))
        for name in algorithm_names
    ]
    
    # =========================================================================
    # Friedman Test (k ≥ 3 algorithms)
    # =========================================================================
    
    friedman_result = None
    nemenyi_result = None
    
    if len(algorithm_names) >= 3:
        friedman_result = analyzer.friedman_test(cost_data)
        
        logging.info(f"Friedman Test (k={len(algorithm_names)} algorithms):")
        logging.info(f"  Statistic: {friedman_result['statistic']:.4f}")
        logging.info(f"  p-value: {friedman_result['p_value']:.6f}")
        logging.info(f"  Significant: {friedman_result['significant']}")
        
        # Calculate mean ranks
        from scipy.stats import rankdata
        mean_ranks = {}
        n_obs = len(cost_data[0])
        for i, name in enumerate(algorithm_names):
            ranks = []
            for obs_idx in range(n_obs):
                obs_costs = [cost_data[j][obs_idx] for j in range(len(algorithm_names))]
                obs_ranks = rankdata(obs_costs, method="average")
                ranks.append(obs_ranks[i])
            mean_ranks[name] = float(np.mean(ranks))
        
        # Nemenyi post-hoc if significant
        if friedman_result["post_hoc_required"]:
            logging.info("\nNemenyi Post-Hoc Test:")
            nemenyi_result = analyzer.nemenyi_posthoc(cost_data, algorithm_names)
            
            logging.info(f"  Critical distance: {nemenyi_result['critical_distance']:.4f}")
            logging.info("  Mean ranks:")
            for name, rank in mean_ranks.items():
                logging.info(f"    {name}: {rank:.2f}")
            
            logging.info("\n  Significant pairwise differences:")
            if nemenyi_result["significant_pairs"]:
                for i, j, p_val in nemenyi_result["significant_pairs"]:
                    name_i = nemenyi_result["algorithm_names"][i]
                    name_j = nemenyi_result["algorithm_names"][j]
                    rank_diff = abs(mean_ranks[name_i] - mean_ranks[name_j])
                    logging.info(
                        f"    {name_i} vs {name_j}: "
                        f"rank_diff={rank_diff:.2f}, p={p_val:.4f}"
                    )
            else:
                logging.info("    None (no significant differences detected)")
    
    # =========================================================================
    # Pairwise Comparisons
    # =========================================================================
    
    logging.info("\nPairwise Comparisons:")
    logging.info("Following statistical methodology:")
    logging.info("  - Normality testing (Shapiro-Wilk)")
    logging.info("  - Test selection (paired t-test vs Wilcoxon)")
    logging.info("  - Effect size (Cohen's d)")
    logging.info("  - 95% Confidence Intervals")
    logging.info("  - Holm-Bonferroni multiple comparison correction\n")
    
    comparisons = []
    for i, name1 in enumerate(algorithm_names):
        for name2 in algorithm_names[i + 1:]:
            summary = analyzer.paired_comparison(
                np.array(_get_data_field(results[name1], "costs")),
                np.array(_get_data_field(results[name2], "costs")),
                f"{name1} vs {name2}",
                metric_name="cost",
            )
            comparisons.append(summary)
            
            # Handle degenerate case (zero variance)
            if summary.test_used == "no_test_needed":
                logging.info(f"  {name1} vs {name2}:")
                logging.info("    All measurements identical (no variance)")
                logging.info(f"    Mean {name1}: {summary.mean_a:.2f}")
                logging.info(f"    Mean {name2}: {summary.mean_b:.2f}")
                logging.info("    No statistical test needed")
                continue
            
            logging.info(f"  {name1} vs {name2}:")
            logging.info("    Normality (Shapiro-Wilk):")
            logging.info(
                f"      {name1}: p={summary.normality_p_value_a:.4f} "
                f"{'(normal)' if summary.normality_p_value_a >= 0.05 else '(non-normal)'}"
            )
            logging.info(
                f"      {name2}: p={summary.normality_p_value_b:.4f} "
                f"{'(normal)' if summary.normality_p_value_b >= 0.05 else '(non-normal)'}"
            )
            logging.info(f"    Test selected: {summary.test_used}")
            logging.info(f"    Mean difference: {summary.mean_difference:.2f}")
            logging.info(
                f"    95% CI difference: "
                f"[{summary.ci_95_difference[0]:.2f}, {summary.ci_95_difference[1]:.2f}]"
            )
            logging.info(f"    p-value (uncorrected): {summary.p_value:.6f}")
            logging.info(f"    Cohen's d: {summary.effect_size:.4f}")
            
            # Effect size interpretation
            abs_d = abs(summary.effect_size)
            if abs_d < 0.2:
                effect_interp = "negligible"
            elif abs_d < 0.5:
                effect_interp = "small"
            elif abs_d < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            logging.info(f"    Effect size interpretation: {effect_interp}")
    
    # Apply Holm-Bonferroni correction
    if len(comparisons) > 1:
        p_values = [c.p_value for c in comparisons]
        reject, pvals_corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        
        logging.info("\n  Multiple Comparison Correction (Holm-Bonferroni, α=0.05):")
        for comp, p_orig, p_adj, sig in zip(
            comparisons, p_values, pvals_corrected, reject
        ):
            logging.info(f"    {comp.comparison_label}:")
            logging.info(f"      Original p-value: {p_orig:.6f}")
            logging.info(f"      Adjusted p-value: {p_adj:.6f}")
            logging.info(f"      Significant (after correction): {sig}")
    
    # =========================================================================
    # Summary Table
    # =========================================================================
    
    logging.info(f"\nSummary: {problem_name} (optimal={optimal_cost})")
    logging.info(short_sep)
    header_fmt = (
        f"{'Algorithm':<20} {'Mean Cost':<15} {'Gap (%)':<12} "
        f"{'Time (s)':<12} {'H2D (MB)':<12} {'D2H (MB)':<12}"
    )
    logging.info(header_fmt)
    logging.info(short_sep)
    
    for name in algorithm_names:
        r = results[name]
        row_fmt = (
            f"{name:<20} "
            f"{r['mean_cost']:<15.2f} "
            f"{r['mean_gap']:<12.2f} "
            f"{r['mean_time']:<12.2f} "
            f"{r.get('mean_h2d_mb', 0.0):<12.2f} "
            f"{r.get('mean_d2h_mb', 0.0):<12.2f}"
        )
        logging.info(row_fmt)
    
    # Calculate speedups
    logging.info("\nSpeedups (relative to baseline):")
    baseline_name = "CPU" if "CPU" in results else algorithm_names[0]
    baseline_time = results[baseline_name]["mean_time"]
    
    for name in algorithm_names:
        if name == baseline_name:
            logging.info(f"  {name}: 1.00x (baseline)")
        else:
            speedup = baseline_time / results[name]["mean_time"]
            logging.info(f"  {name}: {speedup:.2f}x vs {baseline_name}")
    
    logging.info("")
    
    # =========================================================================
    # Build Return Dictionary
    # =========================================================================
    
    statistical_results = {
        "problem_name": problem_name,
        "optimal_cost": optimal_cost,
        "baseline_algorithm": baseline_name,
    }
    
    # Add Friedman test results
    if friedman_result:
        statistical_results["friedman_test"] = {
            "statistic": float(friedman_result["statistic"]),
            "p_value": float(friedman_result["p_value"]),
            "significant": bool(friedman_result["significant"]),
            "post_hoc_required": bool(friedman_result["post_hoc_required"]),
        }
        
        # Add Nemenyi results if performed
        if nemenyi_result:
            statistical_results["nemenyi_posthoc"] = {
                "critical_distance": float(nemenyi_result["critical_distance"]),
                "mean_ranks": mean_ranks,
                "algorithm_names": nemenyi_result["algorithm_names"],
                "significant_pairs": [
                    {
                        "algorithm_i": nemenyi_result["algorithm_names"][i],
                        "algorithm_j": nemenyi_result["algorithm_names"][j],
                        "p_value": float(p_val),
                        "rank_difference": abs(
                            mean_ranks[nemenyi_result["algorithm_names"][i]]
                            - mean_ranks[nemenyi_result["algorithm_names"][j]]
                        ),
                    }
                    for i, j, p_val in nemenyi_result["significant_pairs"]
                ],
            }
    
    # Add pairwise comparison results
    statistical_results["pairwise_comparisons"] = []
    for i, comp in enumerate(comparisons):
        comparison_dict = {
            "comparison": comp.comparison_label,
            "test_used": comp.test_used,
            "p_value": float(comp.p_value),
            "mean_difference": float(comp.mean_difference),
            "ci_95_lower": float(comp.ci_95_difference[0]),
            "ci_95_upper": float(comp.ci_95_difference[1]),
            "cohens_d": float(comp.effect_size),
            "normality_p_value_a": float(comp.normality_p_value_a),
            "normality_p_value_b": float(comp.normality_p_value_b),
        }
        
        # Add corrected p-value if available
        if len(comparisons) > 1 and i < len(pvals_corrected):
            comparison_dict["p_value_corrected"] = float(pvals_corrected[i])
            comparison_dict["significant_after_correction"] = bool(reject[i])
        
        # Effect size interpretation
        abs_d = abs(comp.effect_size)
        if abs_d < 0.2:
            comparison_dict["effect_interpretation"] = "negligible"
        elif abs_d < 0.5:
            comparison_dict["effect_interpretation"] = "small"
        elif abs_d < 0.8:
            comparison_dict["effect_interpretation"] = "medium"
        else:
            comparison_dict["effect_interpretation"] = "large"
        
        statistical_results["pairwise_comparisons"].append(comparison_dict)
    
    return statistical_results
