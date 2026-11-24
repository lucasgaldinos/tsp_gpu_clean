---
title: "Statistical Testing Suite for Metaheuristics: Architecture and Master Index"
description: >
  Master index, architectural roadmap, and navigation guide for the modular statistical
  testing suite: two-tier validation protocol (within-instance runs vs across-instance benchmarks),
  end-to-end decision workflow, standardized symbol legend, and module directory.
created: 2026-11-05
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Initial creation of the master index and architectural roadmap for the modularized
          statistical testing documentation suite, establishing the two-tier Demšar protocol,
          mathematical symbol taxonomy, and cross-module referencing standards.
  - date_modified: 2026-11-23
    modifications:
      - description: >
          Aligned the two-tier workflow diagram and execution sequence with the active
          production protocol in `src/benchmarking/analysis/` (`DP13-3`, `DP13-4`, `BUG-60`,
          `BUG-97`): unconditional paired Wilcoxon + Hodges–Lehmann CI + Kerby rank-biserial
          (`0.1/0.3/0.5`), descriptive-only Shapiro–Wilk, and unified Friedman + Iman–Davenport
          + Nemenyi omnibus (`k >= 3, N >= 2`).
related_files:
  - "[guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md)"
  - "[guide-statistical-normality-and-outliers.md](./guide-statistical-normality-and-outliers.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md)"
  - "[guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md)"
  - "[guide-statistical-demsar-post-hoc-tests.md](./guide-statistical-demsar-post-hoc-tests.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
  - "[notes_types_of_variables_statistics.md](./notes_types_of_variables_statistics.md)"
  - "[statistics.py](../../src/benchmarking/analysis/statistics.py)"
tags:
  - guide/testing
  - guide/analysis
  - analysis/statistical
  - analysis/benchmarking
  - benchmark/statistical-tests
  - notes/theory
  - notes/mathematics
  - algorithm/tsp
  - algorithm/genetic-algorithm
  - review/quality
---

# Statistical Testing Suite for Metaheuristics: Architecture and Master Index

- [Statistical Testing Suite for Metaheuristics: Architecture and Master Index](#statistical-testing-suite-for-metaheuristics-architecture-and-master-index)
  + [1. Mission and Architecture of the Testing Suite](#1-mission-and-architecture-of-the-testing-suite)
  + [2. The Two-Tier Experimental Validation Protocol](#2-the-two-tier-experimental-validation-protocol)
    - [2.1 Tier 1: Within-Instance Stochastic Validation ($n = 15$ Runs)](#21-tier-1-within-instance-stochastic-validation-n--15-runs)
    - [2.2 Tier 2: Across-Instance Scalability Validation ($N = 38$ Instances)](#22-tier-2-across-instance-scalability-validation-n--38-instances)
  + [3. Modular Documentation Directory](#3-modular-documentation-directory)
  + [4. Global End-to-End Decision Workflow](#4-global-end-to-end-decision-workflow)
  + [5. Standardized Mathematical Symbol Legend](#5-standardized-mathematical-symbol-legend)
  + [6. python Scientific Computing Ecosystem Mapping](#6-python-scientific-computing-ecosystem-mapping)
  + [7. Academic References](#7-academic-references)

---

## 1. Mission and Architecture of the Testing Suite

When assessing metaheuristics and GPU-accelerated optimization algorithms, empirical benchmarking easily suffers from severe statistical pitfalls:

1. **The Multiplicity Trap**: Running unadjusted pairwise $t$-tests across $k$ algorithms inflates the Family-Wise Error Rate ($\alpha_{\text{FWER}} = 1 - (1 - \alpha)^m$), yielding spurious "breakthrough" claims.
2. **The Aggregation Conflation**: Pooling repeated stochastic runs ($n = 15$) together with heterogeneous problem instances ($N = 38$) violates independence assumptions and dilutes between-problem variance.
3. **The Non-Normality Blindspot**: Assuming runtime distributions are Gaussian without checking skewness, multimodal stagnation states, or heavy tails.

This modular statistical documentation suite establishes an exhaustive, mathematically rigorous, and publication-ready experimental validation framework tailored to the `gpu_accelerated_clean` codebase.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULAR STATISTICAL TESTING SUITE                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [Index & Roadmap] (guide-statistical-testing-index.md)                               │
│         │                                                                              │
│         ├──► [1. Parametric Foundations] (guide-statistical-parametric-foundations.md) │
│         │    - Student's t, Gamma function, degrees of freedom geometry, Bessel ddof   │
│         │                                                                              │
│         ├──► [2. Normality & Outliers] (guide-statistical-normality-and-outliers.md)   │
│         │    - Shapiro-Wilk matrix derivation, multimodal scales, GA stagnation bounds │
│         │                                                                              │
│         ├──► [3. Pairwise Comparisons] (guide-statistical-pairwise-comparisons.md)     │
│         │    - Paired t-test vs Wilcoxon, exact rank sum identities, p-value logic     │
│         │                                                                              │
│         ├──► [4. Multi-Comparison Tests] (guide-statistical-multicomparison-tests.md)  │
│         │    - Friedman, Iman-Davenport, Nemenyi CD, Dunn-Holm, Benjamini-Hochberg FDR │
│         │                                                                              │
│         ├──► [5. Effect Sizes & Power] (guide-statistical-effect-sizes-and-power.md)   │
│         │    - Cohen's d, U_3 non-overlap, Cliff's delta, power & sample size (ARE)    │
│         │                                                                              │
│         └──► [6. Demšar Protocol Deep-Dive] (guide-statistical-demsar-post-hoc-tests.md)│
│              - Two-tier protocol, Studentized range scaling, CD diagrams               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Two-Tier Experimental Validation Protocol

Following the gold standard established by **Demšar (2006)** and **García et al. (2009)**, the evaluation of heuristic solvers on standard benchmark libraries (such as TSPLIB) must strictly separate **within-instance stochastic variation** from **across-instance generalizability**:

```mermaid
graph TD
    subgraph Tier 1: Within-Instance Analysis
        D1[Instance j: n Matched-Seed Runs per Algorithm]
        D1 --> D2[Shapiro-Wilk Diagnostic - Descriptive Only]
        D1 --> D4[Unconditional Wilcoxon Signed-Rank zsplit + Hodges-Lehmann CI + Kerby r_rb]
        D4 --> D4b[Holm-Bonferroni Step-Down Family-Wise Adjustment]
        D4b --> D5[Instance Summary: Mean Cost, Gap, Runtime]
    end

    subgraph Tier 2: Across-Instance Meta-Analysis
        D5 --> E1[Complete-Case Matrix M in R^{N x k}: N >= 2 Blocks x k >= 3 Treatments]
        E1 --> E2[Friedman Omnibus chi2_F & Iman-Davenport F_F]
        E2 --> E3[OmnibusTestResult: Mean Ranks, Nemenyi CD, Cliques, Pairwise p-Values]
        E3 --> E4[All-vs-All: Nemenyi Critical Difference CD Diagram]
        E3 --> E5[1-vs-Baseline: Scale-Invariant Paired Wilcoxon + Holm Step-Down]
    end
```

> [!IMPORTANT]
> **Production Protocol Alignment (`DP13-3`, `DP13-4`, `BUG-60`, `BUG-97`):** Following Demšar (2006) §3.1.2, [`statistics.py::StatisticalAnalyzer.paired_comparison`](../../src/benchmarking/analysis/statistics.py) **never** switches between Student's $t$-test and Wilcoxon based on a Shapiro–Wilk pre-test (which distorts the nominal $\alpha = 0.05$ error rate on small samples). Shapiro–Wilk is recorded strictly as a descriptive diagnostic (`(nan, False)` for $n < 3$ or zero variance), while every paired comparison runs Wilcoxon signed-rank (`zero_method="zsplit"`), distribution-free Hodges–Lehmann CIs (guarded when $2^{-(n-1)} > \alpha$), and Kerby rank-biserial correlation $|r_{\text{rb}}|$ classified on the correlation scale (`0.1 / 0.3 / 0.5`). Cohort-level baseline Wilcoxon tests via [`statistics.py::StatisticalAnalyzer.wilcoxon_paired_summary`](../../src/benchmarking/analysis/statistics.py) share the exact `1e-10` absolute zero-difference tolerance (`BUG-97`). The parametric derivations in [guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md) and [guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md) remain as theoretical reference.

### 2.1 Tier 1: Within-Instance Stochastic Validation ($n = 15$ Runs)

- **Unit of Study**: A single problem instance $P_j$ (e.g., `berlin52` or `ch130`).
- **Objective**: Determine whether Algorithm A (e.g., `FullGPU`) is significantly faster or finds lower tour costs than Algorithm B (e.g., `CPU`) on this specific instance.
- **Sample Structure**: $n = 15$ stochastic seeds. Because both algorithms solve the identical instance starting from matched initial tour structures, data is **paired**.
- **Applicable Guide**: [guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md).

### 2.2 Tier 2: Across-Instance Scalability Validation ($N = 38$ Instances)

- **Unit of Study**: The entire benchmark suite consisting of $N = 38$ TSPLIB instances.
- **Objective**: Establish whether the ranking hierarchy (e.g., $\text{FullGPU} \succ \text{HybridOptimized} \succ \text{HybridNaive} \succ \text{CPU}$) holds consistently across different problem scales ($n_{\text{cities}} \in [51, 1002]$) and topology types.
- **Sample Structure**: Each problem instance acts as a single matched **block** in a Randomized Complete Block Design (RCBD). We aggregate the $n = 15$ runs per cell into a single robust central tendency (median runtime or mean solution gap).
- **Applicable Guides**: [guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md) and [guide-statistical-demsar-post-hoc-tests.md](./guide-statistical-demsar-post-hoc-tests.md).

---

## 3. Modular Documentation Directory

The table below outlines each module in the documentation suite, its scope, and primary takeaways:

| Module File | Primary Scope | Key Formulations | User Questions Resolved |
| :--- | :--- | :--- | :--- |
| **[guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md)** | Distribution theory & geometry of inference | $\Gamma(\alpha)$, Student's $t$ PDF, degrees of freedom $\nu = n-1$, Bessel `ddof=1` vs `ddof=0` | Gamma function intuition, degrees of freedom geometry, $t_{n, 0.975}$ quantile notation, `ddof` explanation |
| **[guide-statistical-normality-and-outliers.md](./guide-statistical-normality-and-outliers.md)** | Normality checks & runtime distribution diagnosis | Shapiro-Wilk matrix form ($W = \frac{(\mathbf{a}^T\mathbf{m})^2}{\mathbf{x}^T\mathbf{C}\mathbf{x}}$), Coefficient of Variation ($CV$), GA stagnation termination | Shapiro-Wilk matrix math, scale illusions, bimodal distributions, GA early-stagnation outliers |
| **[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)** | Two-treatment within-instance testing | Paired $t$-test, Wilcoxon Signed-Rank ($T^+, T^-$, $\mathbb{E}[T], \text{Var}(T)$), $p$-value definition | Paired definition, frequentist $p$-value meaning, $T^+ + T^-$ identity derivation, worked $n=15$ examples |
| **[guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md)** | Multi-algorithm across-instance testing | Friedman omnibus $\chi_F^2$, Iman-Davenport $F_F$, Nemenyi $CD$, Dunn-Holm, Benjamini-Hochberg FDR | Multi-paired testing rationale, Dunn's test, Nemenyi CD derivation, CD diagram reading, FDR control |
| **[guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md)** | Effect quantification & experimental power | Cohen's $d$, pooled variance derivation, Cohen's $U_3$ non-overlap, sample size $n(\alpha, \beta)$, ARE | Pooled variance algebra, percentile shift $\Phi(d)$, APA reporting format, empirical TSPLIB grounding |
| **[guide-statistical-demsar-post-hoc-tests.md](./guide-statistical-demsar-post-hoc-tests.md)** | Post-hoc multi-comparison & CD diagrams | Studentized range division by $\sqrt{2}$, exact rank sum identity $\sum R_j = 10.00$, CD cliques | Critical Difference diagram interpretation, SciPy `studentized_range.ppf` scaling convention |
| **[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)** | Three-level data architecture & empirical benchmarks | Complete 38-instance TSPLIB analysis, python pipeline, empirical speedup and gap tables | Three-tier data structure, within-problem vs across-problem meta-analysis |
| **[notes_types_of_variables_statistics.md](./notes_types_of_variables_statistics.md)** | Variables taxonomy in dynamical & optimization systems | Mathematical characterization ($\mathbf{x}, \mathbf{u}, \mathbf{w}, \boldsymbol{\theta}$), DAE constraints, GPU memory hierarchy mappings | Taxonomy of variables, state/control/decision distinctions, GPU solver variables |

---

## 4. Global End-to-End Decision Workflow

When conducting a statistical comparison between routing solver strategies, adhere strictly to the following execution sequence:

```text
START: Experimental Evaluation of Heuristics (src/benchmarking/analysis/)
  │
  ├─► [Step 1: Within-Instance Paired Comparison (analyze_problem)]
  │     │
  │     ├── Extract paired vectors d_i = x_A,i - x_B,i (for both cost and time)
  │     ├── Record Shapiro-Wilk p-values descriptively (never used for test selection)
  │     ├── Execute Unconditional Wilcoxon Signed-Rank Test (zero_method="zsplit")
  │     │     ├── Exact small-n minimum p = 2^{-(n-1)} (no false p=0.0 on constant diffs)
  │     │     ├── Hodges-Lehmann location & 95% CI (NaN when 2^{-(n-1)} > alpha)
  │     │     └── Kerby rank-biserial correlation r_rb (0.1 / 0.3 / 0.5 scale)
  │     └── Adjust all C(k, 2) pairwise p-values with monotonic Holm-Bonferroni step-down
  │
  ├─► [Step 2: Construct Across-Instance Matrix (analyze_cohort, N >= 2, k >= 3)]
  │     │
  │     ├── Form complete-case matrix M in R^{N x k} (gap/cost for quality; mean_time for speed)
  │     ├── Rank algorithms within each problem row: r_ij in {1, ..., k}
  │     ├── Compute column average ranks: R_j = (1/N) * sum_i r_ij
  │     └── Verify rank conservation: sum_j R_j = k(k + 1) / 2
  │
  ├─► [Step 3: Omnibus Non-Parametric Significance Test (run_friedman_nemenyi_analysis)]
  │     │
  │     ├── Compute Friedman chi2_F and Iman-Davenport F_F (flagging low_power when N < 5)
  │     ├── Record omnibus_significant = (p_value_f < alpha)
  │     └── Emit immutable OmnibusTestResult (reused once across console, macros, and CD plots)
  │
  └─► [Step 4: Post-Hoc Multi-Comparisons & Baseline Speedups]
        │
        ├─ All-vs-All: Nemenyi CD = (q_Tukey / sqrt(2)) * sqrt(k(k+1)/(6N)) + CD TikZ diagram
        ├─ 1-vs-Baseline: Scale-invariant wilcoxon_paired_summary (1e-10 abs tol) + Holm step-down
        └─ Speedups: Fleming & Wallace (1986) paired geometric mean across common instances
```

---

## 5. Standardized Mathematical Symbol Legend

To maintain uncompromising mathematical rigor across all modules, every symbol is defined under a unified standard:

| Symbol | Mathematical Meaning | Standard Context / Dimensions |
| :--- | :--- | :--- |
| $N$ | Number of independent benchmark problem instances | Blocks in RCBD ($N = 38$ TSPLIB instances) |
| $n$ | Number of stochastic seed repetitions per instance | Within-instance sample size ($n = 15$ runs) |
| $n_{\text{cities}}$ | Dimension of TSP problem instance | Number of vertices / coordinates ($51 \le n_{\text{cities}} \le 1{,}002$) |
| $k$ | Number of candidate algorithms compared | Treatments in RCBD ($k = 4$ algorithm variants) |
| $m$ | Total number of pairwise comparisons | $m = \binom{k}{2} = \frac{k(k - 1)}{2}$ ($m = 6$ for $k = 4$) |
| $\alpha$ | Significance level (Type I error threshold) | Typically $\alpha = 0.05$ |
| $\beta$ | Type II error probability ($1 - \beta = \text{Power}$) | Typically $\beta = 0.20 \implies \text{Power} = 0.80$ |
| $\mu, \bar{x}$ | Population mean and sample mean | $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ |
| $\sigma^2, s^2$ | Population variance and sample variance | $s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2$ (`ddof=1`) |
| $\nu, df$ | Degrees of freedom | $\nu = n - 1$ for one-sample/paired tests |
| $\Gamma(\alpha)$ | Euler Gamma function | Extension of factorial: $\Gamma(n) = (n - 1)!$ |
| $d_i$ | Paired difference between treatments | $d_i = x_{1, i} - x_{2, i}$ |
| $W$ | Shapiro-Wilk test statistic | $W \in (0, 1]$, measures agreement with Gaussian |
| $T^+, T^-$ | Sum of positive and negative Wilcoxon ranks | $T^+ + T^- = \frac{n(n + 1)}{2}$ |
| $r_{ij}$ | Rank of algorithm $j$ on instance $i$ | $r_{ij} \in \{1, \dots, k\}$ |
| $R_j$ | Average rank of algorithm $j$ across $N$ instances | $R_j = \frac{1}{N} \sum_{i=1}^N r_{ij}$ |
| $\chi_F^2$ | Friedman omnibus chi-square test statistic | Asymptotically distributed as $\chi^2(k - 1)$ |
| $F_F$ | Iman-Davenport corrected $F$-statistic | Distributed as $F(k - 1, (k - 1)(N - 1))$ |
| $q_\alpha$ | Studentized range critical value divided by $\sqrt{2}$ | Used in Demšar Nemenyi Critical Difference |
| $CD$ | Critical Difference threshold | Minimal rank delta for significance: $\|R_i - R_j\| > CD$ |
| $d, d_z$ | Cohen's effect size (independent vs paired) | Standardized difference in means |
| $\delta$ | Cliff's Delta non-parametric effect size | $\delta \in [-1, 1]$, probability of stochastic dominance |
| $ARE$ | Asymptotic Relative Efficiency | Large-sample efficiency ratio between tests |

---

## 6. python Scientific Computing Ecosystem Mapping

The repository's statistical analysis module ([statistics.py](../../src/benchmarking/analysis/statistics.py)) interfaces directly with three specialized python libraries:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        python SCIENTIFIC COMPUTING SUITE                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. scipy.stats:                                                                        │
│    - Parametric tests: ttest_rel, ttest_ind, f_oneway                                  │
│    - Non-parametric tests: wilcoxon, mannwhitneyu, friedmanchisquare                   │
│    - Distribution quantiles: t.ppf, f.ppf, chi2.ppf, studentized_range.ppf             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. statsmodels.stats:                                                                  │
│    - Multiple testing adjustments: multipletests (Bonferroni, Holm, Benjamini-Hochberg)│
│    - Power and sample size: TTestPower, tt_solve_power                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. scikit_posthocs:                                                                    │
│    - Non-parametric post-hoc pairwise matrices: posthoc_nemenyi_friedman, posthoc_dunn │
│    - Critical Difference diagram generation: critical_difference_diagram               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Installation requirements:

```bash
pip install scipy statsmodels scikit-posthocs
```

---

## 7. Academic References

1. **Demšar, J. (2006).** Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
2. **García, S., Fernández, A., Luengo, J., & Herrera, F. (2009).** A Study of Statistical Techniques and Performance Measures for Genetics-Based Machine Learning: Accuracy and Interpretability. *Soft Computing*, 13(10), 959–977. DOI: [10.1007/s00500-008-0392-y](https://doi.org/10.1007/s00500-008-0392-y).
3. **Friedman, M. (1937).** The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance. *Journal of the American Statistical Association*, 32(200), 675–701. DOI: [10.1080/01621459.1937.10503522](https://doi.org/10.1080/01621459.1937.10503522).
4. **Iman, R. L., & Davenport, J. M. (1980).** Approximations of the Critical Value of the Kruskal-Wallis and Friedman Tests to Significance Levels by Means of the F-Distribution. *Communications in Statistics - Theory and Methods*, 9(6), 571–595. DOI: [10.1080/03610928008827904](https://doi.org/10.1080/03610928008827904).
5. **Nemenyi, P. B. (1963).** *Distribution-Free Multiple Comparisons*. PhD Thesis, Princeton University.
6. **Wilcoxon, F. (1945).** Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83. DOI: [10.2307/3001968](https://doi.org/10.2307/3001968).
7. **Gosset, W. S. ["Student"] (1908).** The Probable Error of a Mean. *Biometrika*, 6(1), 1–25. DOI: [10.1093/biomet/6.1.1](https://doi.org/10.1093/biomet/6.1.1).
8. **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates, Hillsdale, NJ.
9. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. DOI: [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
10. **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
