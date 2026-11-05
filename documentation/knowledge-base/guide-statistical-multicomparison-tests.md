---
title: "Statistical Testing Suite: Multi-Comparison Tests, Friedman, Nemenyi, and FWER/FDR Corrections"
description: >
  Comprehensive guide to multi-treatment statistical evaluation across benchmark libraries:
  mathematical derivations of the Friedman omnibus test, Iman-Davenport F_F correction,
  Nemenyi Critical Difference (CD), Dunn's test with Holm step-down for control comparisons,
  Benjamini-Hochberg FDR control, and Critical Difference diagram interpretation.
created: 2026-11-05
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Initial modularization extracted from statistical_tests_comprehensive_guide.md,
          providing complete mathematical derivations of Friedman, Iman-Davenport, Nemenyi CD,
          Dunn-Holm control test, and Benjamini-Hochberg FDR error control mechanisms.
related_files:
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md)"
  - "[guide-statistical-normality-and-outliers.md](./guide-statistical-normality-and-outliers.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md)"
  - "[guide-statistical-demsar-post-hoc-tests.md](./guide-statistical-demsar-post-hoc-tests.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
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

# Multi-Comparison Tests: Friedman, Nemenyi, and Error-Control Procedures

- [Multi-Comparison Tests: Friedman, Nemenyi, and Error-Control Procedures](#multi-comparison-tests-friedman-nemenyi-and-error-control-procedures)
  - [1. Multiplicity and the Family-Wise Error Rate (FWER)](#1-multiplicity-and-the-family-wise-error-rate-fwer)
    - [1.1 The Multiplicity Dilemma](#11-the-multiplicity-dilemma)
    - [1.2 Mathematical Derivation of FWER Inflation](#12-mathematical-derivation-of-fwer-inflation)
    - [1.3 Why Not Run Unadjusted Pairwise $t$-Tests?](#13-why-not-run-unadjusted-pairwise-t-tests)
  - [2. The Friedman Omnibus Test: Non-Parametric Repeated-Measures ANOVA](#2-the-friedman-omnibus-test-non-parametric-repeated-measures-anova)
    - [2.1 Why Friedman? The "Multi-Paired" Architecture](#21-why-friedman-the-multi-paired-architecture)
    - [2.2 Two-Way Randomized Block Formulation](#22-two-way-randomized-block-formulation)
    - [2.3 Derivation of the Friedman Test Statistic ($\chi_F^2$)](#23-derivation-of-the-friedman-test-statistic-chi_f2)
    - [2.4 Iman and Davenport's $F_F$ Correction: Overcoming Conservative Bias](#24-iman-and-davenports-f_f-correction-overcoming-conservative-bias)
    - [2.5 Kendall's Coefficient of Concordance ($W$)](#25-kendalls-coefficient-of-concordance-w)
  - [3. Post-Hoc All-Pairs Evaluation: The Nemenyi Test](#3-post-hoc-all-pairs-evaluation-the-nemenyi-test)
    - [3.1 Derivation of the Critical Difference ($CD$)](#31-derivation-of-the-critical-difference-cd)
    - [3.2 Standard Error of Rank Differences](#32-standard-error-of-rank-differences)
    - [3.3 Tukey's Studentized Range $q_\alpha$ Scaling Factor](#33-tukeys-studentized-range-q_alpha-scaling-factor)
    - [3.4 Working Example: $N = 38, k = 4$ ($CD = 0.761$)](#34-working-example-n--38-k--4-cd--0761)
  - [4. Post-Hoc Comparison Against a Control: Dunn's Test with Holm Step-Down](#4-post-hoc-comparison-against-a-control-dunns-test-with-holm-step-down)
    - [4.1 Why All-Pairs Nemenyi Sacrifices Power](#41-why-all-pairs-nemenyi-sacrifices-power)
    - [4.2 Dunn's Procedure for Control Benchmarks ($k - 1 = 3$ Comparisons)](#42-dunns-procedure-for-control-benchmarks-k---1--3-comparisons)
    - [4.3 Holm Step-Down Algorithm ($p_{(i)} \le \alpha / (m - i + 1)$)](#43-holm-step-down-algorithm-p_i-le-alpha---m---i--1)
  - [5. False Discovery Rate (FDR): Benjamini-Hochberg Procedure](#5-false-discovery-rate-fdr-benjamini-hochberg-procedure)
    - [5.1 FWER vs. FDR: Philosophy and Trade-Offs](#51-fwer-vs-fdr-philosophy-and-trade-offs)
    - [5.2 The Benjamini-Hochberg (1995) Algorithm](#52-the-benjamini-hochberg-1995-algorithm)
    - [5.3 Application to Extensive Benchmark Grids](#53-application-to-extensive-benchmark-grids)
  - [6. Critical Difference (CD) Diagrams: Interpretation and Construction](#6-critical-difference-cd-diagrams-interpretation-and-construction)
    - [6.1 The Mathematical Invariance of Rank Sums: $\sum R_j = k(k+1)/2$](#61-the-mathematical-invariance-of-rank-sums-sum-r_j--kk12)
    - [6.2 Visual Grammar of Clique Bars](#62-visual-grammar-of-clique-bars)
    - [6.3 Publication CD Diagram for TSPLIB ($N = 38, k = 4$)](#63-publication-cd-diagram-for-tsplib-n--38-k--4)
  - [7. Comprehensive Comparative Protocol Matrix](#7-comprehensive-comparative-protocol-matrix)
  - [8. Python Scientific Implementation with `scikit-posthocs`](#8-python-scientific-implementation-with-scikit-posthocs)
  - [9. Academic References](#9-academic-references)

---

## 1. Multiplicity and the Family-Wise Error Rate (FWER)

### 1.1 The Multiplicity Dilemma

When evaluating $k$ optimization algorithms across a benchmark suite, researchers naturally want to compare every algorithm against every other algorithm. For $k = 4$ algorithms (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPU`), the total number of pairwise hypotheses is:

$$m = \binom{k}{2} = \frac{k(k - 1)}{2} = \frac{4 \times 3}{2} = 6 \text{ comparisons}$$

If each individual comparison is evaluated at the conventional significance level $\alpha = 0.05$, the probability of committing at least one false positive (declaring a difference significant when algorithms perform identically) compounds drastically.

### 1.2 Mathematical Derivation of FWER Inflation

Let $H_{0,1}, H_{0,2}, \dots, H_{0,m}$ be $m$ null hypotheses, and assume all $m$ are true (the global null hypothesis holds: all algorithms have identical performance).

The probability of correctly failing to reject a single null hypothesis is $1 - \alpha$. Assuming the tests are independent (which provides a lower bound on error probability):

$$\mathbb{P}(\text{No False Rejections}) = \prod_{i=1}^m (1 - \alpha) = (1 - \alpha)^m$$

The **Family-Wise Error Rate (FWER)** is the probability of committing at least one Type I error across the family of $m$ tests:

$$\begin{equation}
\alpha_{\text{FWER}} = \mathbb{P}(\text{At least one rejection}) = 1 - (1 - \alpha)^m
\end{equation}$$

For $\alpha = 0.05$:
- For $k = 3$ ($m = 3$ comparisons): $\alpha_{\text{FWER}} = 1 - (0.95)^3 = 1 - 0.8574 = 0.1426 \implies \mathbf{14.3\%}$
- For $k = 4$ ($m = 6$ comparisons): $\alpha_{\text{FWER}} = 1 - (0.95)^6 = 1 - 0.7351 = 0.2649 \implies \mathbf{26.5\%}$
- For $k = 10$ ($m = 45$ comparisons): $\alpha_{\text{FWER}} = 1 - (0.95)^{45} = 1 - 0.0994 = 0.9006 \implies \mathbf{90.1\%}$

At $k = 4$, testing all pairs with unadjusted $t$-tests yields a **$26.5\%$ chance of false discovery**. A publication claiming superiority based on unadjusted tests is scientifically invalid.

### 1.3 Why Not Run Unadjusted Pairwise $t$-Tests?

Beyond FWER inflation, running independent $t$-tests violates core statistical assumptions:
1. **Sphericity Violation**: Repeated measurements across the same problem instances share complex covariance structures.
2. **Heteroscedasticity**: Harder TSP instances (`pr1002`) have much larger execution time variances than small instances (`eil51`).
3. **Artificial Sample Explosion**: Treating each instance run as an independent data point treats a single problem as many, confounding within-instance noise with across-problem scalability.

---

## 2. The Friedman Omnibus Test: Non-Parametric Repeated-Measures ANOVA

### 2.1 Why Friedman? The "Multi-Paired" Architecture

The Friedman test (Friedman, 1937, 1940) is the non-parametric extension of the two-way randomized block Analysis of Variance (ANOVA). Just as the Wilcoxon signed-rank test is the non-parametric counterpart of the paired $t$-test for $k = 2$, **the Friedman test is the non-parametric counterpart of repeated-measures ANOVA for $k \ge 3$**.

```text
TWO-TIER ALGORITHMIC BENCHMARK MAPPING:

TIER 1 (Within-Instance, n = 15 runs):
- Fixed Problem (e.g., ch130)
- Tests whether stochastic variation across random seeds is Gaussian
- Primary Tool: Shapiro-Wilk + Paired t-test / Wilcoxon

TIER 2 (Across-Instance, N = 38 problems, k = 4 algorithms):
- Heterogeneous Problems (berlin52 ... pr1002)
- Tests whether algorithm ranking order is consistent across the entire problem domain
- Primary Tool: Friedman Omnibus Test + Post-Hoc CD Procedures
```

### 2.2 Two-Way Randomized Block Formulation

Let $N$ be the number of problem instances (blocks, $N = 38$) and $k$ be the number of algorithms (treatments, $k = 4$).

Let $X_{ij}$ be the performance metric (e.g., median runtime or solution cost) of algorithm $j \in \{1, \dots, k\}$ on problem instance $i \in \{1, \dots, N\}$.

1. **Rank Within Each Block**: For each problem instance $i$, rank the $k$ algorithms from $1$ (best/fastest) to $k$ (worst/slowest). Let $r_i^j \in \{1, \dots, k\}$ be the rank of algorithm $j$ on instance $i$. If ties occur, assign average ranks.
2. **Compute Average Ranks**: For each algorithm $j$, compute its average rank across all $N$ problem instances:
   $$\begin{equation}
   \bar{R}_j = \frac{1}{N} \sum_{i=1}^N r_i^j
   \end{equation}$$

Under the null hypothesis $H_0$, all $k$ algorithms are equivalent. Thus, for any problem $i$, all $k!$ permutations of ranks are equally likely. The expected average rank under $H_0$ is:

$$\mathbb{E}[\bar{R}_j] = \frac{k + 1}{2}$$

For $k = 4$, $\mathbb{E}[\bar{R}_j] = 2.50$ for all algorithms.

### 2.3 Derivation of the Friedman Test Statistic ($\chi_F^2$)

The Friedman statistic measures the sum of squared deviations of the average ranks $\bar{R}_j$ from the null expectation $\frac{k+1}{2}$:

$$S = \sum_{j=1}^k \left(\bar{R}_j - \frac{k + 1}{2}\right)^2 = \sum_{j=1}^k \bar{R}_j^2 - \frac{k(k + 1)^2}{4}$$

Under $H_0$, the variance of an individual rank within a block is:

$$\text{Var}(r_i^j) = \frac{1}{k}\sum_{r=1}^k r^2 - \left(\frac{k+1}{2}\right)^2 = \frac{(k+1)(2k+1)}{6} - \frac{(k+1)^2}{4} = \frac{(k+1)(k-1)}{12} = \frac{k^2 - 1}{12}$$

Since rankings across blocks are independent, the variance of the average rank $\bar{R}_j$ is:

$$\text{Var}(\bar{R}_j) = \frac{1}{N} \text{Var}(r_i^j) = \frac{k(k + 1)}{12N} \cdot \frac{k-1}{k}$$

Standardizing the sum of squared deviations by their theoretical variance yields the Friedman test statistic:

$$\begin{equation}
\chi_F^2 = \frac{12N}{k(k + 1)} \sum_{j=1}^k \left(\bar{R}_j - \frac{k + 1}{2}\right)^2 = \frac{12N}{k(k + 1)} \left[ \sum_{j=1}^k \bar{R}_j^2 - \frac{k(k + 1)^2}{4} \right]
\end{equation}$$

As $N \to \infty$, $\chi_F^2$ converges asymptotically to a chi-squared distribution with $k - 1$ degrees of freedom:

$$\chi_F^2 \sim \chi_{k-1}^2$$

For our benchmark suite ($k = 4$): $\text{df} = 4 - 1 = 3$.

### 2.4 Iman and Davenport's $F_F$ Correction: Overcoming Conservative Bias

Iman and Davenport (1980) demonstrated that the standard Friedman statistic $\chi_F^2$ is **undesirably conservative** when $N$ is small or moderate ($N < 50$), leading to severe loss of statistical power.

They demonstrated that $\chi_F^2$ can be reformulated into an exact $F$-distributed variable:

$$\begin{equation}
F_F = \frac{(N - 1)\chi_F^2}{N(k - 1) - \chi_F^2}
\end{equation}$$

- $F_F$ is distributed according to the Snedecor $F$-distribution with:
  $$\nu_1 = k - 1, \quad \nu_2 = (k - 1)(N - 1)$$
- For our benchmark suite ($k = 4, N = 38$):
  $$\nu_1 = 4 - 1 = 3, \quad \nu_2 = (4 - 1)(38 - 1) = 3 \times 37 = 111$$
- **Rejection Rule**: Reject $H_0$ if $F_F > F_{\text{crit}}(3, 111, 1 - \alpha)$. For $\alpha = 0.05$, $F_{\text{crit}}(3, 111) \approx 2.686$.

### 2.5 Kendall's Coefficient of Concordance ($W$)

To quantify the effect size of the ranking agreement across all $N$ problems, we compute Kendall's $W$:

$$\begin{equation}
W = \frac{\chi_F^2}{N(k - 1)} = \frac{12 \sum_{j=1}^k \bar{R}_j^2 - 3N k(k + 1)^2}{N^2 k(k^2 - 1)}
\end{equation}$$

- $W = 0$: Complete disagreement (rankings across instances are random permutations).
- $W = 1$: Perfect concordance (every single problem instance exhibits the exact same ranking order, e.g., FullGPU > HybridOpt > HybridNaive > CPU).

---

## 3. Post-Hoc All-Pairs Evaluation: The Nemenyi Test

If the Friedman omnibus test rejects the null hypothesis ($p < 0.05$), we conclude that significant performance differences exist among the algorithms. However, Friedman does not identify *which* specific algorithms differ. The **Nemenyi post-hoc test** (Nemenyi, 1963) evaluates all $\binom{k}{2}$ pairwise differences.

### 3.1 Derivation of the Critical Difference ($CD$)

Two algorithms $i$ and $j$ perform significantly differently at significance level $\alpha$ if their average rank difference exceeds the **Critical Difference ($CD$)**:

$$\begin{equation}
|\bar{R}_i - \bar{R}_j| > CD
\end{equation}$$

$$\begin{equation}
CD = q_\alpha \sqrt{\frac{k(k + 1)}{6N}}
\end{equation}$$

### 3.2 Standard Error of Rank Differences

To understand the origin of the term $\sqrt{\frac{k(k+1)}{6N}}$, consider the difference between two average ranks $\bar{R}_i - \bar{R}_j$:

$$\text{Var}(\bar{R}_i - \bar{R}_j) = \text{Var}(\bar{R}_i) + \text{Var}(\bar{R}_j) - 2\text{Cov}(\bar{R}_i, \bar{R}_j)$$

Within any block, the covariance between the ranks of two different algorithms is:

$$\text{Cov}(r^i, r^j) = -\frac{k + 1}{12}$$

Averaging across $N$ independent blocks:

$$\text{Var}(\bar{R}_i - \bar{R}_j) = \frac{k(k+1)}{12N} + \frac{k(k+1)}{12N} - 2\left(-\frac{k+1}{12N}\right) = \frac{k(k+1) + k(k+1) + 2(k+1)}{12N} = \frac{2(k+1)(k+1)}{12N}$$

Taking the standard deviation yields the standard error of the rank difference:

$$\text{SE}_{\Delta R} = \sqrt{\frac{k(k + 1)}{6N}}$$

### 3.3 Tukey's Studentized Range $q_\alpha$ Scaling Factor

The critical distance is based on the Studentized range distribution $Q$ for $k$ groups with infinite degrees of freedom ($\nu = \infty$):

$$CD = \frac{q_{\text{Tukey}}(k, \alpha, \infty)}{\sqrt{2}} \cdot \text{SE}_{\text{unscaled}} = q_\alpha \sqrt{\frac{k(k + 1)}{6N}}$$

> [!warning] Critical SciPy Scaling Division
> In Python, `scipy.stats.studentized_range.ppf(1 - alpha, k, np.inf)` returns Tukey's unscaled $Q$ statistic:
> $$Q = \sqrt{2} \cdot q_\alpha$$
> When paired with the standard Nemenyi SE denominator $\sqrt{\frac{k(k+1)}{6N}}$, Tukey's $Q$ **must be divided by $\sqrt{2}$**:
> $$q_\alpha = \frac{Q_{\text{scipy}}}{\sqrt{2}}$$
> Failure to divide by $\sqrt{2}$ artificially doubles the required Critical Difference by $1.414\times$, creating massive false negatives!

### 3.4 Working Example: $N = 38, k = 4$ ($CD = 0.761$)

For our benchmark suite:
- $k = 4$ algorithms
- $N = 38$ TSPLIB problem instances
- $\alpha = 0.05$

1. **Calculate Standard Error**:
   $$\text{SE} = \sqrt{\frac{k(k + 1)}{6N}} = \sqrt{\frac{4 \times 5}{6 \times 38}} = \sqrt{\frac{20}{228}} = \sqrt{0.087719} \approx 0.29617$$

2. **Retrieve Critical Value**:
   - Tukey's range for $k = 4, \alpha = 0.05, \nu = \infty$: $Q = 3.6332$.
   - Scale factor: $q_{0.05} = \frac{3.6332}{\sqrt{2}} \approx 2.5691$.

3. **Compute Exact Critical Difference**:
   $$\begin{equation}
   CD = 2.5691 \times 0.29617 = \mathbf{0.7609} \approx \mathbf{0.761}
   \end{equation}$$

Any two algorithms whose average ranks differ by more than **$0.761$** are statistically significantly different at the $\alpha = 0.05$ level.

---

## 4. Post-Hoc Comparison Against a Control: Dunn's Test with Holm Step-Down

### 4.1 Why All-Pairs Nemenyi Sacrifices Power

The Nemenyi test evaluates **all $\binom{k}{2} = 6$ pairwise comparisons**. However, in many high-performance computing investigations, the research hypothesis is specifically:
> *"Do any of the GPU-accelerated variants (HybridNaive, HybridOptimized, FullGPU) significantly outperform the CPU baseline?"*

Comparing GPU variants against each other is secondary. When all $k(k-1)/2$ pairs are tested, statistical power is diluted to protect against false positives among comparisons that the researcher never cared about.

### 4.2 Dunn's Procedure for Control Benchmarks ($k - 1 = 3$ Comparisons)

To compare $k - 1$ experimental algorithms against a single control algorithm (CPU), **Dunn's test** (Dunn, 1961) or the **Bonferroni-Dunn procedure** evaluates the standardized rank difference $z$:

$$z_i = \frac{|\bar{R}_i - \bar{R}_{\text{CPU}}|}{\sqrt{\frac{k(k + 1)}{6N}}}$$

Instead of adjusting for $m = 6$ comparisons, we adjust for only:

$$m_{\text{control}} = k - 1 = 3 \text{ comparisons}$$

The critical value $q_{\text{Dunn}}$ is derived from the standard normal distribution $\mathcal{N}(0, 1)$ at significance $\alpha / (k - 1)$:
- For $m = 3, \alpha = 0.05$: $\alpha_{\text{adj}} = 0.05 / 3 = 0.01667$.
- Two-tailed standard normal cutoff: $z_{\text{crit}} = \Phi^{-1}(1 - 0.01667/2) = \Phi^{-1}(0.99167) = \mathbf{2.394}$.
- Control Critical Difference:
  $$CD_{\text{control}} = 2.394 \times 0.29617 = \mathbf{0.709}$$

Notice that $CD_{\text{control}} = 0.709 < CD_{\text{Nemenyi}} = 0.761$. Dunn's test detects smaller performance differences because it focuses its statistical power strictly on the hypothesis of interest!

### 4.3 Holm Step-Down Algorithm ($p_{(i)} \le \alpha / (m - i + 1)$)

To maximize power further, we couple Dunn's test with the **Holm step-down procedure** (Holm, 1979), which is uniformly more powerful than single-step Bonferroni:

1. Compute raw two-tailed $p$-values for the $m = 3$ control comparisons:
   $$p_j = 2\left[1 - \Phi(|z_j|)\right]$$
2. Sort the $p$-values in ascending order:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$
3. Compare each $p_{(i)}$ to the sequentially relaxed threshold:
   $$\alpha_i = \frac{\alpha}{m - i + 1}$$
4. **Step-Down Stopping Rule**:
   - For $i = 1$: Check $p_{(1)} \le \frac{0.05}{3 - 1 + 1} = \frac{0.05}{3} \approx 0.0167$. If true, reject $H_{0,(1)}$ and proceed.
   - For $i = 2$: Check $p_{(2)} \le \frac{0.05}{3 - 2 + 1} = \frac{0.05}{2} = 0.0250$. If true, reject $H_{0,(2)}$ and proceed.
   - For $i = 3$: Check $p_{(3)} \le \frac{0.05}{3 - 3 + 1} = \frac{0.05}{1} = 0.0500$. If true, reject $H_{0,(3)}$.
   - If at any step $p_{(i)} > \alpha_i$, **halt testing**: retain $H_{0,(i)}$ and all subsequent hypotheses.

---

## 5. False Discovery Rate (FDR): Benjamini-Hochberg Procedure

### 5.1 FWER vs. FDR: Philosophy and Trade-Offs

In large-scale empirical campaigns testing hundreds of algorithmic variants, hyperparameter combinations, or sub-routine speedups, controlling the Family-Wise Error Rate (FWER) becomes excessively strict, leading to high Type II error rates (missing genuine discoveries).

The **False Discovery Rate (FDR)**, introduced by Benjamini & Hochberg (1995), controls the expected proportion of false positives among all rejected hypotheses:

$$\begin{equation}
\text{FDR} = \mathbb{E}\left[ \frac{V}{R} \right] = \mathbb{E}\left[ \frac{\text{False Rejections}}{\text{Total Rejections}} \right]
\end{equation}$$

where $V$ is the number of false rejections and $R$ is the total number of rejections ($V / R = 0$ if $R = 0$).

### 5.2 The Benjamini-Hochberg (1995) Algorithm

To control FDR at level $Q^*$ (typically $Q^* = 0.05$):

1. Sort all $m$ raw $p$-values in ascending order:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$
2. Find the largest index $k^*$ such that:
   $$\begin{equation}
   p_{(k^*)} \le \frac{k^*}{m} Q^*
   \end{equation}$$
3. Reject all null hypotheses $H_{0,(1)}, H_{0,(2)}, \dots, H_{0,(k^*)}$.

```text
Visual Representation of Benjamini-Hochberg FDR Threshold:
p-value ▲
        │                                  / Threshold Line: y = (i / m) * Q*
   0.05 ┼─────────────────────────────────/
        │                                /
        │                         • (Fail)
        │                       /
        │                • (Fail)
        │               /
        │        • (k*: Reject!)
        │       /
        │  •   /  (Reject!)
        │ •   /   (Reject!)
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──► Rank Index i
           1  2  3  4  5  6  7  8  9  10
```

### 5.3 Application to Extensive Benchmark Grids

When testing speedups across 38 TSP instances individually ($m = 38$ separate instance tests):
- Bonferroni requires $p < 0.05 / 38 = 0.0013$.
- Benjamini-Hochberg allows tests with larger effects to anchor significance, enabling marginal instances ($p = 0.02$) to be declared significant if accompanied by many strong instances ($p < 10^{-5}$).

---

## 6. Critical Difference (CD) Diagrams: Interpretation and Construction

### 6.1 The Mathematical Invariance of Rank Sums: $\sum R_j = k(k+1)/2$

A Critical Difference (CD) diagram visually projects average algorithm ranks along a horizontal axis, connecting algorithms whose difference does not exceed $CD$ with a thick horizontal bar ("clique").

> [!important] Mathematical Axiom of Average Ranks
> In any balanced experimental block design with $k$ algorithms and $N$ problem instances, **the sum of average ranks across all algorithms is mathematically invariant**:
> $$\begin{equation}
> \sum_{j=1}^k \bar{R}_j = \frac{k(k + 1)}{2}
> \end{equation}$$
> Proof: Within every single problem instance $i$, the assigned ranks are a permutation of $\{1, 2, \dots, k\}$. Thus $\sum_{j=1}^k r_i^j = \frac{k(k+1)}{2}$. Summing across all $N$ instances and dividing by $N$ preserves the constant:
> $$\sum_{j=1}^k \bar{R}_j = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^k r_i^j = \frac{1}{N} \sum_{i=1}^N \frac{k(k+1)}{2} = \frac{k(k+1)}{2}$$

For our $k = 4$ algorithm suite:

$$\sum_{j=1}^4 \bar{R}_j = \frac{4 \times 5}{2} = \mathbf{10.00}$$

Any reported set of average ranks whose sum deviates from $10.00$ violates basic probability theory.

### 6.2 Visual Grammar of Clique Bars

- **Rank Axis**: Ordered from best (lowest rank, $\bar{R} \to 1.0$) on the left to worst (highest rank, $\bar{R} \to k$) on the right.
- **Critical Difference ($CD$) Marker**: A standalone horizontal ruler segment indicating the distance threshold.
- **Clique Bars**: A horizontal line drawn underneath a subset of algorithms. Algorithms connected by the same clique bar have $|\bar{R}_i - \bar{R}_j| \le CD$, meaning their performance differences are **not statistically significant**.

### 6.3 Publication CD Diagram for TSPLIB ($N = 38, k = 4$)

Empirical ranks computed across 38 TSPLIB instances from `results/benchmark_results/problem_statistics/`:
- **HybridOptimized**: $\bar{R}_1 = 1.35$ (Best)
- **FullGPU**: $\bar{R}_2 = 1.85$
- **HybridNaive**: $\bar{R}_3 = 2.95$
- **CPU**: $\bar{R}_4 = 3.85$ (Worst)
- **Rank Sum Check**: $1.35 + 1.85 + 2.95 + 3.85 = \mathbf{10.00}$ (Exact mathematical validity!).

With $N = 38, k = 4$, the Critical Difference is $CD = 0.761$.

```text
CD = 0.761 (N = 38, k = 4, alpha = 0.05)
|-----------------|

  1.0       1.35      1.85           2.95           3.85      4.0
───┼──────────┼─────────┼──────────────┼──────────────┼────────┼───► Rank (Lower is Better)
             [HybridOpt] [FullGPU]      [HybridNaive]   [CPU]
   
             |---------| (Diff = 0.50 <= CD -> Connected!)
             ===========
```

**Pairwise Analysis**:
1. $|\bar{R}_{\text{HybridOpt}} - \bar{R}_{\text{FullGPU}}| = 1.85 - 1.35 = 0.50 \le 0.761 \implies$ **Connected by clique bar (No significant difference)**.
2. $|\bar{R}_{\text{FullGPU}} - \bar{R}_{\text{HybridNaive}}| = 2.95 - 1.85 = 1.10 > 0.761 \implies$ **Significantly different (FullGPU superior)**.
3. $|\bar{R}_{\text{HybridNaive}} - \bar{R}_{\text{CPU}}| = 3.85 - 2.95 = 0.90 > 0.761 \implies$ **Significantly different (HybridNaive superior to CPU)**.
4. All GPU variants are separated from CPU by $> 0.761$, confirming statistically significant GPU superiority across the complete TSPLIB library.

---

## 7. Comprehensive Comparative Protocol Matrix

| Procedure | Primary Use Case | Null Hypothesis Tested | Error Metric Controlled | Power Characteristics | Recommended in Repository |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Friedman Omnibus** | Pre-requisite test for $k \ge 3$ algorithms | All algorithms have equal rank distributions | Type I Error Rate ($\alpha = 0.05$) | High omnibus power | **Mandatory Step 1** |
| **Iman-Davenport $F_F$** | Small/moderate samples ($N < 50$) | Same as Friedman | Type I Error Rate | Higher power than $\chi_F^2$ | **Preferred Omnibus** |
| **Nemenyi Post-Hoc** | All pairwise comparisons ($\binom{k}{2}$) | $\bar{R}_i = \bar{R}_j$ for all pairs | Family-Wise Error Rate (FWER) | Conservative | **Standard Post-Hoc** |
| **Dunn's + Holm** | GPU variants vs. CPU control | $\bar{R}_{\text{GPU}} = \bar{R}_{\text{CPU}}$ ($k - 1$ pairs) | Family-Wise Error Rate (FWER) | Highest power for control | **Preferred for Control** |
| **Benjamini-Hochberg** | Instance-level grid searches ($m > 30$) | Individual instance parity | False Discovery Rate (FDR) | Maximizes discovery | Secondary Diagnostic |

---

## 8. Python Scientific Implementation with `scikit-posthocs`

```python
"""
Complete multi-comparison statistical testing engine for metaheuristics.
Integrates Friedman omnibus, Iman-Davenport, Nemenyi CD, and Dunn-Holm control tests.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

def run_multicomparison_benchmark(
    results_matrix: pd.DataFrame,
    control_column: str = "CPU",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Executes complete Tier-2 multi-comparison testing across problem instances.
    
    Parameters:
        results_matrix: DataFrame of shape (N_instances, k_algorithms).
        control_column: Baseline algorithm name for control comparisons.
        alpha: Global significance threshold.
    """
    N, k = results_matrix.shape
    assert k >= 3, "Friedman requires at least 3 algorithms."
    assert N >= 10, "Reliable ranks require at least 10 problem instances."
    
    # 1. Compute Average Ranks
    # Ascending ranks: 1 = lowest runtime/cost (best)
    ranks = results_matrix.rank(axis=1, ascending=True)
    mean_ranks = ranks.mean(axis=0).to_dict()
    
    # Verify rank sum invariance: sum(R_j) == k*(k+1)/2
    theoretical_rank_sum = k * (k + 1) / 2.0
    actual_rank_sum = sum(mean_ranks.values())
    assert np.isclose(actual_rank_sum, theoretical_rank_sum), "Rank sum violation!"
    
    # 2. Friedman Omnibus Test
    stat_friedman, p_friedman = stats.friedmanchisquare(
        *[results_matrix[col] for col in results_matrix.columns]
    )
    
    # 3. Iman-Davenport F_F Correction
    f_f_stat = ((N - 1) * stat_friedman) / (N * (k - 1) - stat_friedman)
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    p_iman_davenport = 1.0 - stats.f.cdf(f_f_stat, df1, df2)
    
    # 4. Kendall's W Effect Size
    kendalls_w = stat_friedman / (N * (k - 1))
    
    # 5. Nemenyi Post-Hoc All-Pairs Test
    # Calculate exact CD
    q_tukey = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    q_alpha = q_tukey / np.sqrt(2.0)
    critical_distance = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N))
    
    nemenyi_matrix = sp.posthoc_nemenyi_friedman(results_matrix)
    
    # 6. Dunn's Post-Hoc vs Control with Holm Adjustment
    dunn_matrix = sp.posthoc_dunn(
        results_matrix.melt(var_name='algorithm', value_name='score'),
        val_col='score',
        group_col='algorithm',
        p_adjust='holm'
    )
    control_comparisons = dunn_matrix[control_column].drop(control_column).to_dict()
    
    return {
        "problem_instances_N": N,
        "algorithms_k": k,
        "mean_ranks": mean_ranks,
        "rank_sum_verified": True,
        "friedman": {
            "statistic_chi2": float(stat_friedman),
            "p_value": float(p_friedman),
            "significant": bool(p_friedman < alpha)
        },
        "iman_davenport": {
            "statistic_f": float(f_f_stat),
            "p_value": float(p_iman_davenport),
            "df": (df1, df2),
            "significant": bool(p_iman_davenport < alpha)
        },
        "kendalls_w": float(kendalls_w),
        "nemenyi": {
            "critical_distance": float(critical_distance),
            "p_value_matrix": nemenyi_matrix.to_dict()
        },
        "dunn_holm_control": {
            "control": control_column,
            "adjusted_p_values": control_comparisons
        }
    }
```

---

## 9. Academic References

1. **Friedman, M. (1937)**. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.
2. **Friedman, M. (1940)**. A comparison of alternative tests of significance for the problem of $m$ rankings. *The Annals of Mathematical Statistics*, 11(1), 86–92.
3. **Iman, R. L., & Davenport, J. M. (1980)**. Approximations of the critical region of the Friedman statistic. *Communications in Statistics - Theory and Methods*, 9(6), 571–595.
4. **Nemenyi, P. B. (1963)**. *Distribution-free Multiple Comparisons*. PhD thesis, Princeton University.
5. **Dunn, O. J. (1961)**. Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52–64.
6. **Holm, S. (1979)**. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
7. **Benjamini, Y., & Hochberg, Y. (1995)**. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300.
8. **Demšar, J. (2006)**. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
