# Chapter 4 Validation Benchmark Guide

## Overview

The `chapter4_validation.py` benchmark implements **complete academic statistical methodology** for comparing **CPU baseline vs 3 GPU-based Genetic Algorithm variants** on TSP instances.

## Quick Start

```bash
# Test run with CPU (2 reps per algorithm, ~10-15 minutes)
python code/benchmarks/chapter4_validation.py --repetitions 2

# Full academic benchmark with CPU (30 reps, ~24-30 hours)
python code/benchmarks/chapter4_validation.py

# GPU-only comparison (skip CPU baseline, faster)
python code/benchmarks/chapter4_validation.py --skip-cpu
```

## Algorithm Comparison

The benchmark compares **CPU baseline vs 3 GPU implementations**:

| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **CPU** | Pure NumPy baseline | Sequential processing, early stopping enabled |
| **HybridNaive** | Sequential 2-opt on GPU | Individual kernel launches per tour |
| **HybridOptimized** | Batched 2-opt on GPU | Batch processing with optimized kernels |
| **FullGPU** | Fujimoto's parallel GA | Complete GPU implementation with early stopping |

### CPU Baseline Strategy

**Important**: CPU is only run on **small problems (n < 100)** due to runtime constraints:

- Small problems (13 instances): CPU included for direct CPU vs GPU comparison
- Medium/Large problems (25 instances): CPU skipped (would take days to complete)

This size-stratified approach is academically valid and explicitly documented in results.

## Output Structure

```
results/
├── chapter4_validation.log           # Full execution log
└── tables/
    ├── chapter4_validation.md        # Markdown tables (GitHub-friendly)
    └── chapter4_validation.tex       # LaTeX tables (thesis-ready)
```

### Generated Tables

#### Table 1: Summary Statistics Per Problem

- **Content**: Problems × Algorithms (38 total problems, 4 algorithms for small problems, 3 for large)
- **Columns**: Problem name, algorithm, mean cost, std dev, gap%, time, speedup
- **Speedup**: Relative to CPU baseline (when available) or HybridNaive (GPU baseline)

#### Table 2: Algorithm Comparison (Aggregated)

- **Content**: Mean statistics across all problems
- **Columns**: Algorithm, mean gap%, std gap, mean time, std time, avg speedup
- **Purpose**: Overall performance comparison across all problem sizes

#### Table 3: Best Algorithm by Problem Size

- **Content**: Winner for each size category
- **Categories**:
  - Small: <100 cities (includes CPU baseline)
  - Medium: 100-300 cities (GPU only)
  - Large: ≥300 cities (GPU only)
- **Purpose**: Scalability analysis and algorithm selection guidance

### Per-Problem Speedup Logging

After each problem, speedups are logged relative to the baseline:

```
Speedups (relative to baseline):
  CPU: 1.00x (baseline)
  HybridNaive: 8.45x vs CPU
  HybridOptimized: 47.23x vs CPU
  FullGPU: 65.18x vs CPU
```

For problems where CPU is skipped (n ≥ 100), HybridNaive becomes the baseline:

```
Speedups (relative to baseline):
  HybridNaive: 1.00x (baseline)
  HybridOptimized: 5.59x vs HybridNaive
  FullGPU: 7.71x vs HybridNaive
```

## Statistical Methodology

Implements **first_draft.md Section 3.5.3** requirements:

### Per-Problem Analysis

1. **Shapiro-Wilk normality tests** (α=0.05) on each algorithm's cost distribution
2. **Test selection**: Paired t-test (parametric) vs Wilcoxon signed-rank (non-parametric)
3. **Effect size**: Cohen's d with interpretation (negligible/small/medium/large)
4. **Confidence intervals**: 95% CI for mean differences
5. **Multiple comparison correction**: Holm-Bonferroni method (α=0.05)

### Cross-Problem Analysis

1. **Friedman test**: Non-parametric ANOVA across all problems
2. **Nemenyi post-hoc**: Pairwise ranking comparisons
3. **Aggregate statistics**: Mean gap and time across all instances

## Sample Output

### Pairwise Comparison (Enhanced)

```
HybridNaive vs HybridOptimized:
  Normality (Shapiro-Wilk):
    HybridNaive: p=0.0234 (non-normal)
    HybridOptimized: p=0.0456 (non-normal)
  Test selected: wilcoxon_signed_rank
  Mean difference: 0.50
  95% CI difference: [-5.85, 6.85]
  p-value (uncorrected): 0.1234
  Cohen's d: 0.3456
  Effect size interpretation: small
```

### Multiple Comparison Correction

```
Multiple Comparison Correction (Holm-Bonferroni, α=0.05):
  HybridNaive vs HybridOptimized:
    Original p-value: 0.1234
    Adjusted p-value: 0.2468
    Significant (after correction): False
  HybridNaive vs FullGPU:
    Original p-value: 0.0012
    Adjusted p-value: 0.0036
    Significant (after correction): True
```

## Configuration

### Default Parameters (BENCHMARK_PARAMS)

```python
{
    "repetitions": 30,              # Statistical rigor per first_draft.md
    "patience": 50,                 # Early stopping generations
    "skip_cpu_default": False,      # Include CPU baseline by default
    "cpu_size_threshold": 100       # Only run CPU on problems with n < 100
}
```

**CPU Strategy**: The threshold prevents CPU from running on medium/large problems where it would take days to complete. This is documented in results and is academically acceptable for computational constraints.

### GA Parameters (GA_PARAMS)

```python
{
    "population_size": 256,
    "mutation_rate": 0.02,
    "tournament_size": 5,
    "two_opt_iterations": 10,
    "seed": 42                  # Reproducibility
}
```

### Adaptive Generations

- Formula: `max_gens = 14.3 × n - 46`
- Range: 728 gens (n=51) to 63,435 gens (n=1002)
- Early stopping: Patience of 50 generations without improvement

## Problem Set (38 TSP Instances)

| Size Range | Count | Examples | Optimal Range |
|------------|-------|----------|---------------|
| Small (<100) | 13 | eil51, berlin52, st70 | 426 - 1,211 |
| Medium (100-300) | 17 | kroA100, pr264, pr299 | 6,110 - 126,643 |
| Large (≥300) | 8 | rd400, rat783, pr1002 | 8,806 - 259,045 |

**Total**: 38 problems × 3 algorithms × 30 reps = **3,420 total runs**

## Estimated Runtime

### With Early Stopping (--repetitions 30)

- **Small problems**: ~2-3 min per problem
- **Medium problems**: ~5-10 min per problem
- **Large problems**: ~15-30 min per problem
- **Total**: ~18-20 hours (with early stopping)

### Test Run (--repetitions 2)

- **Total**: ~5-10 minutes
- **Purpose**: Verify statistical analysis works correctly

## Academic Compliance

### Citations Required

- **Kirkpatrick et al. (1983)**: Simulated Annealing (methodology)
- **Fujimoto & Tsutsui (2011)**: GPU-accelerated GA baseline
- **Demšar (2006)**: Friedman + Nemenyi statistical methodology
- **Cohen (1988)**: Effect size interpretation

### Methodology References

- `first_draft.md` Section 3.5: Experimental Design
- `gpu_cpu_comparison_methodology.md`: Statistical rigor guidelines

## Troubleshooting

### "CUDA out of memory"

- Reduce population size in GA_PARAMS
- Skip large problems (modify PROBLEM_SET)
- System has 4GB VRAM (GTX 1050 Mobile) with 65% limit (2.6GB usable)

### "No statistical significance found"

- Expected with only 2 repetitions (test mode)
- Run full 30 repetitions for statistical power
- Small problems may reach optimal quickly (all algorithms tie)

### "Tables not generated"

- Check `results/tables/` directory
- Ensure all problems complete successfully
- Tables generated at end of benchmark (after all problems)

## Next Steps

1. **Test run**: `--repetitions 2` to verify everything works (~5 min)
2. **Full benchmark**: Default 30 repetitions (~18-20 hours)
3. **Table review**: Check `results/tables/chapter4_validation.{md,tex}`
4. **Thesis integration**: LaTeX tables ready for Chapter 4 RESULTADOS section

## Implementation Status

✅ **COMPLETE - Academic Methodology**:

- Shapiro-Wilk normality testing
- Proper test selection (paired t-test / Wilcoxon)
- Cohen's d effect size calculation
- 95% Confidence intervals
- Holm-Bonferroni correction (manual implementation)
- Friedman + Nemenyi post-hoc tests
- 3-table generation system (MD + LaTeX formats)
- Academic citations and compliance
