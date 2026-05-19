# GPU vs CPU Algorithm Comparison Methodology

**Document Status:** Research findings compiled 2025-11-01  
**Research Context:** Task 5 - Statistical methodology for performance comparison  
**Memory Graph:** Findings stored in knowledge graph with 5 entities and 7 relations

---

## TOC

- [GPU vs CPU Algorithm Comparison Methodology](#gpu-vs-cpu-algorithm-comparison-methodology)
  - [TOC](#toc)
  - [Overview](#overview)
  - [Primary Sources](#primary-sources)
    - [1. Hoefler \& Belli (2015)](#1-hoefler--belli-2015)
    - [2. Hothorn, Leisch, Zeileis, Hornik (2005)](#2-hothorn-leisch-zeileis-hornik-2005)
  - [Statistical Methodology](#statistical-methodology)
    - [Core Principles](#core-principles)
  - [Experimental Design Best Practices](#experimental-design-best-practices)
    - [Dataset/Problem Instance Selection](#datasetproblem-instance-selection)
  - [Performance Reporting Standards](#performance-reporting-standards)
    - [Required Metrics](#required-metrics)
    - [Visualization Guidelines](#visualization-guidelines)
  - [TSP/Routing-Specific Considerations](#tsprouting-specific-considerations)
    - [Benchmark Selection](#benchmark-selection)
    - [Algorithm-Specific Metrics](#algorithm-specific-metrics)
  - [Reproducibility Checklist](#reproducibility-checklist)
    - [Essential Documentation](#essential-documentation)
    - [Code and Data Availability](#code-and-data-availability)
  - [Statistical Tests Reference](#statistical-tests-reference)
    - [When to Use Which Test](#when-to-use-which-test)
    - [Assumption Checks](#assumption-checks)
  - [Python Implementation Notes](#python-implementation-notes)
    - [Recommended Libraries](#recommended-libraries)
    - [Example: Paired Comparison (CPU vs GPU)](#example-paired-comparison-cpu-vs-gpu)
  - [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
  - [References](#references)
    - [Core Methodology Papers](#core-methodology-papers)
    - [Statistical Methods References](#statistical-methods-references)
    - [TSP GPU Implementation References](#tsp-gpu-implementation-references)
  - [Memory Graph Structure](#memory-graph-structure)
  - [Application to This Project](#application-to-this-project)
    - [Immediate Applicability](#immediate-applicability)
    - [Hardware Constraints](#hardware-constraints)

---

## Overview

This document synthesizes academic best practices for statistically rigorous GPU vs CPU algorithm performance comparison, based on authoritative sources in high-performance computing and statistical benchmarking.

## Primary Sources

### 1. Hoefler & Belli (2015)

**Title:** "Scientific Benchmarking of Parallel Computing Systems: Twelve ways to tell the masses when reporting performance results"  
**Published:** SC '15 Conference Proceedings  
**DOI:** [10.1145/2807591.2807644](https://doi.org/10.1145/2807591.2807644)  
**Citations:** 186 (highly influential in HPC community)  
**Key Contribution:** Investigated 120 papers and found most lack clarity on whether performance improvements are deterministic or observed by chance. Proposes statistically sound analysis and reporting techniques.

### 2. Hothorn, Leisch, Zeileis, Hornik (2005)

**Title:** "The Design and Analysis of Benchmark Experiments"  
**Published:** American Statistical Association, Institute of Mathematical Statistics, Interface Foundation  
**Key Contribution:** Theoretical framework for comparison of candidate algorithms and algorithm selection using cross-validation and standard statistical test procedures.

---

## Statistical Methodology

### Core Principles

1. **Report Uncertainty, Not Just Point Estimates**
   - Always include confidence intervals (e.g., 95% CI)
   - Report standard deviation or interquartile ranges
   - Use bootstrap/resampling methods (Efron & Tibshirani 1993) for robust estimates
   - Consider quantile regression (Koenker 2005) to understand performance distribution

2. **Statistical Significance Testing**
   - Test whether performance differences are real vs. chance occurrences
   - Use appropriate tests based on data distribution:
     - **Paired t-test:** When normality holds (use Shapiro-Wilk test to check)
     - **Wilcoxon signed-rank test:** Non-parametric alternative for paired comparisons
     - **Bootstrap confidence intervals:** When distribution is unknown or non-normal

3. **Multiple Comparison Corrections**
   - When comparing >2 algorithms, apply correction methods:
     - Bonferroni correction (conservative)
     - Holm correction (less conservative, more power)
   - Use Kruskal-Wallis test for multiple algorithm comparison (non-parametric ANOVA)

4. **Effect Size vs. Statistical Significance**
   - Report practical significance, not just p-values
   - Calculate effect sizes (e.g., Cohen's d) to quantify magnitude of differences
   - Small p-value doesn't always mean practically important difference

---

## Experimental Design Best Practices

### Dataset/Problem Instance Selection

1. **Diversity Requirement**
   - Test on multiple problem instances - avoid single-instance claims
   - For TSP/VRP: Use standard benchmarks (TSPLIB, Reinelt's collection)
   - Cover range of problem sizes: small (n ≤ 100), medium (100 < n ≤ 1000), large (n > 1000)
   - Include different problem characteristics (clustered, random, structured)

2. **Repetition and Variability**
   - Report number of repetitions for each experiment
   - Document how variability is measured (across runs, across instances, both)
   - Minimum 30 runs recommended for robust statistics (Central Limit Theorem)
   - Control for system noise and external factors:
     - Pin CPU processes to cores
     - Disable CPU frequency scaling
     - Monitor thermal throttling
     - Isolate benchmark processes

3. **Parameter Documentation**
   - Hardware specifications (CPU model, GPU model, VRAM, RAM)
   - Software versions (CUDA toolkit, Python, CuPy, NumPy)
   - Compiler flags and optimization levels
   - Runtime settings and environment variables
   - For GPU: block size, grid dimensions, shared memory configuration

---

## Performance Reporting Standards

### Required Metrics

1. **Execution Time**
   - Mean execution time with confidence interval
   - Standard deviation or IQR
   - Minimum/maximum if showing full distribution
   - Time measurement methodology (wall-clock, CPU time, GPU kernel time)

2. **Speedup Factors**
   - Report speedup with statistical backing (not single measurements)
   - Include confidence intervals on speedup estimates
   - Clearly define baseline (e.g., "GPU speedup relative to single-threaded NumPy CPU")
   - Distinguish between weak scaling and strong scaling

3. **Memory Usage**
   - Critical for GPU with limited VRAM
   - Report peak memory consumption
   - Document memory transfer overhead (host-to-device, device-to-host)

4. **Scalability**
   - Performance across different problem sizes
   - Identify inflection points where one approach becomes superior
   - Report complexity trends (does it match theoretical O() analysis?)

### Visualization Guidelines

1. **Appropriate Plot Types**
   - **Box plots:** Show distribution, median, quartiles, outliers
   - **Violin plots:** Similar to box plots but shows full distribution shape
   - **Error bars:** On bar charts, must specify if showing std dev, std err, or CI
   - **Avoid:** Simple bar charts without error bars or variability indicators

2. **Clarity Requirements**
   - Clear axis labels with units
   - Legend identifying all series
   - Error bars must specify what they represent
   - Use log scale when appropriate (large performance differences)
   - Annotate statistical significance when comparing groups

---

## TSP/Routing-Specific Considerations

### Benchmark Selection

1. **Standard Instances**
   - TSPLIB: Reinelt's classic benchmark collection
   - Use instances with known optimal solutions for correctness verification
   - Include both symmetric and asymmetric TSP instances

2. **Instance Characteristics**
   - Euclidean distance matrices (most common)
   - Geographic instances (real-world cities)
   - Random instances (for stress testing)

### Algorithm-Specific Metrics

1. **Construction Heuristics**
   - Solution quality (tour length vs. optimal)
   - Construction time
   - Memory footprint

2. **Meta-heuristics**
   - Solution quality over iterations
   - Convergence speed
   - Population/swarm size optimization
   - Document random seed handling for reproducibility

3. **GPU-Specific Parameters**
   - Optimal CUDA block size (typically 128-512 threads)
   - Grid dimensions impact
   - Shared memory vs. global memory tradeoffs
   - Occupancy metrics
   - Memory coalescing efficiency

---

## Reproducibility Checklist

### Essential Documentation

- [ ] Hardware platform details (CPU, GPU, memory)
- [ ] Operating system and kernel version
- [ ] Software dependencies with exact versions
- [ ] Random seeds used (if applicable)
- [ ] Number of repetitions per experiment
- [ ] Statistical tests applied
- [ ] Significance level (α, typically 0.05)
- [ ] Confidence interval level (typically 95%)
- [ ] Data preprocessing steps
- [ ] Outlier handling methodology

### Code and Data Availability

- [ ] Source code accessible (GitHub, etc.)
- [ ] Benchmark datasets documented or included
- [ ] Build/compilation instructions
- [ ] Execution instructions with expected output
- [ ] Results data in machine-readable format (CSV, JSON)

---

## Statistical Tests Reference

### When to Use Which Test

| Scenario | Parametric Test | Non-parametric Alternative |
|----------|----------------|---------------------------|
| Two related samples (same algorithm, CPU vs GPU) | Paired t-test | Wilcoxon signed-rank test |
| Two independent samples | Independent t-test | Mann-Whitney U test |
| Multiple related samples | Repeated measures ANOVA | Friedman test |
| Multiple independent samples | One-way ANOVA | Kruskal-Wallis test |

### Assumption Checks

1. **Normality:** Shapiro-Wilk test (for n < 50), Kolmogorov-Smirnov (for n ≥ 50)
2. **Homogeneity of Variance:** Levene's test, Bartlett's test
3. **Independence:** Experimental design consideration (proper randomization)

---

## Python Implementation Notes

### Recommended Libraries

```python
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon, ttest_rel, shapiro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For violin plots, box plots
```

### Example: Paired Comparison (CPU vs GPU)

```python
# Assuming cpu_times and gpu_times are arrays of execution times
# Both should have same number of measurements

# 1. Check normality assumption
stat_cpu, p_cpu = shapiro(cpu_times)
stat_gpu, p_gpu = shapiro(gpu_times)
normal = (p_cpu > 0.05) and (p_gpu > 0.05)

# 2. Choose appropriate test
if normal:
    # Parametric: paired t-test
    statistic, p_value = ttest_rel(cpu_times, gpu_times)
    test_name = "Paired t-test"
else:
    # Non-parametric: Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(cpu_times, gpu_times)
    test_name = "Wilcoxon signed-rank test"

# 3. Report results
print(f"{test_name}: statistic={statistic:.4f}, p-value={p_value:.4e}")
if p_value < 0.05:
    print("Statistically significant difference at α=0.05")
else:
    print("No statistically significant difference at α=0.05")

# 4. Bootstrap confidence interval for speedup
from scipy.stats import bootstrap

speedups = cpu_times / gpu_times
ci = bootstrap((speedups,), np.mean, confidence_level=0.95, 
               n_resamples=10000, method='percentile')
print(f"Mean speedup: {np.mean(speedups):.2f}x")
print(f"95% CI: [{ci.confidence_interval.low:.2f}x, "
      f"{ci.confidence_interval.high:.2f}x]")
```

---

## Common Pitfalls to Avoid

1. **Single-Run Comparisons**
   - ❌ "GPU is 10x faster" based on one measurement
   - ✅ "GPU shows mean speedup of 10.2x (95% CI: [9.5x, 10.9x]) over 100 runs"

2. **Cherry-Picking Problem Instances**
   - ❌ Only showing instances where GPU excels
   - ✅ Report results across all benchmark instances, including unfavorable cases

3. **Ignoring Variability**
   - ❌ Bar chart with no error bars
   - ✅ Box plot or bar chart with confidence interval error bars

4. **P-Hacking**
   - ❌ Trying multiple tests until finding p < 0.05
   - ✅ Pre-specify statistical test based on data characteristics

5. **Overlooking Practical Significance**
   - ❌ "p < 0.001, therefore GPU is better"
   - ✅ "Statistically significant (p < 0.001) speedup of 1.05x - not practically meaningful given implementation complexity"

---

## References

### Core Methodology Papers

1. Hoefler, T., & Belli, R. (2015). Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results. *SC '15: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 1-12. <https://doi.org/10.1145/2807591.2807644>

2. Hothorn, T., Leisch, F., Zeileis, A., & Hornik, K. (2005). The Design and Analysis of Benchmark Experiments. *Journal of Computational and Graphical Statistics*.

3. Bailey, D. H. (1991). Twelve ways to fool the masses when giving performance results on parallel computers. *Supercomputing Review*, 54-55.

4. Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically rigorous java performance evaluation. *OOPSLA '07*, 57-76.

### Statistical Methods References

5. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

6. Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

7. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895-1923.

### TSP GPU Implementation References

8. Robinson, T., et al. (2018). Analysis of a High-Performance TSP Solver on the GPU. *ACM Journal of Experimental Algorithmics*.

9. Various authors. Performance investigations of genetic algorithms on graphics cards (TSP applications).

---

## Memory Graph Structure

This methodology is stored in the knowledge graph with the following entities:

- **GPU-CPU Comparison Methodology** (Academic Methodology)
- **Benchmark Experimental Design** (Research Methodology)
- **Performance Reporting Standards** (Academic Standard)
- **TSP GPU Benchmarking** (Domain-Specific Practice)
- **Statistical Tests for Algorithm Comparison** (Statistical Method)

Relations established:

- GPU-CPU Comparison Methodology *requires* Benchmark Experimental Design
- GPU-CPU Comparison Methodology *follows* Performance Reporting Standards
- GPU-CPU Comparison Methodology *uses* Statistical Tests for Algorithm Comparison
- TSP GPU Benchmarking *applies* GPU-CPU Comparison Methodology
- And others...

Use memory search tools to query these entities for specific guidance.

---

## Application to This Project

### Immediate Applicability

1. **Section 4 RESULTADOS Analysis**
   - Apply these principles when reporting backend comparison results
   - Use appropriate statistical tests for NumPy vs CuPy comparisons
   - Create visualizations following reporting standards

2. **Benchmark Suite Design**
   - Select diverse TSPLIB instances covering size range
   - Plan repetition strategy (minimum 30 runs per configuration)
   - Document all experimental parameters

3. **Q1 Notebook Enhancement (Task 3)**
   - Add confidence intervals to existing plots
   - Apply statistical significance testing
   - Improve visualizations (box plots for multi-run results)

### Hardware Constraints

Given project hardware (GTX 1050 Mobile, 4GB VRAM):

- Maximum problem size ~3,000 nodes (VRAM limitation)
- Document this constraint in methodology
- Report memory usage alongside execution time
- Compare against CPU baseline with same problem size limits

---

**Last Updated:** 2025-11-01  
**Stored in Knowledge Graph:** Yes (5 entities, 7 relations)  
**Next Steps:** Apply to Task 3 (Q1 Notebook Enhancement) and Section 4 analysis
