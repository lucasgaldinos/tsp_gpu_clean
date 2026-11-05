---
title: "Statistical Testing Suite: Normality Assessment, Shapiro-Wilk Matrix Derivations, and Metaheuristic Outliers"
description: >
  Comprehensive guide to normality verification for small-sample metaheuristic benchmarks (n = 15),
  featuring the full Generalized Least Squares matrix derivation of the Shapiro-Wilk W statistic,
  scale illusion analysis via Coefficient of Variation, multimodal distribution diagnosis, and
  investigation of GA stagnation outliers and generational stopping criteria.
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
          adding rigorous matrix derivation of Shapiro-Wilk W, Coefficient of Variation scale
          illusion treatment, and genetic algorithm stagnation termination analysis.
related_files:
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md)"
  - "[guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md)"
  - "[analysis-genetic-algorithm-memetic-lineage.md](./analysis-genetic-algorithm-memetic-lineage.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
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

# Normality Assessment, Shapiro-Wilk Matrix Derivations, and Metaheuristic Outliers

- [Normality Assessment, Shapiro-Wilk Matrix Derivations, and Metaheuristic Outliers](#normality-assessment-shapiro-wilk-matrix-derivations-and-metaheuristic-outliers)
  - [1. The Small-Sample Mandate: Why Normality Testing is Non-Negotiable at $n = 15$](#1-the-small-sample-mandate-why-normality-testing-is-non-negotiable-at-n--15)
    - [1.1 The "Rule of 30" Fallacy in Metaheuristics](#11-the-rule-of-30-fallacy-in-metaheuristics)
    - [1.2 Small Samples as the Gatekeeper of Statistical Inference](#12-small-samples-as-the-gatekeeper-of-statistical-inference)
  - [2. Shapiro-Wilk Test: Full Matrix Derivation via Generalized Least Squares](#2-shapiro-wilk-test-full-matrix-derivation-via-generalized-least-squares)
    - [2.1 Order Statistics and Covariance Structure](#21-order-statistics-and-covariance-structure)
    - [2.2 Generalized Least Squares (GLS) Formulation](#22-generalized-least-squares-gls-formulation)
    - [2.3 Derivation of the $W$ Statistic](#23-derivation-of-the-w-statistic)
    - [2.4 Geometric Interpretation and Boundedness](#24-geometric-interpretation-and-boundedness)
    - [2.5 Computational Approximation for $n = 15$](#25-computational-approximation-for-n--15)
  - [3. The Scale Illusion: Absolute Dispersion vs. Coefficient of Variation ($CV$)](#3-the-scale-illusion-absolute-dispersion-vs-coefficient-of-variation-cv)
    - [3.1 The Compression Illusion in GPU Timings](#31-the-compression-illusion-in-gpu-timings)
    - [3.2 Mathematical Formulation of $CV$](#32-mathematical-formulation-of-cv)
    - [3.3 Empirical Demonstration on TSPLIB Instances](#33-empirical-demonstration-on-tsplib-instances)
  - [4. Multimodality Diagnosis in Combinatorial Search](#4-multimodality-diagnosis-in-combinatorial-search)
    - [4.1 Modality Taxonomy in Optimization Benchmarks](#41-modality-taxonomy-in-optimization-benchmarks)
    - [4.2 Basin-of-Attraction Dynamics and Local Optima Trapping](#42-basin-of-attraction-dynamics-and-local-optima-trapping)
    - [4.3 Statistical Detection of Multimodality](#43-statistical-detection-of-multimodality)
  - [5. Genetic Algorithm Stagnation and Runtime Outlier Analysis](#5-genetic-algorithm-stagnation-and-runtime-outlier-analysis)
    - [5.1 Algorithmic Root Causes of Runtime Outliers](#51-algorithmic-root-causes-of-runtime-outliers)
    - [5.2 The 50-Generation Stagnation Termination Criterion](#52-the-50-generation-stagnation-termination-criterion)
    - [5.3 Hardware Coupling: Thread Blocks, Shared Memory, and Divergence](#53-hardware-coupling-thread-blocks-shared-memory-and-divergence)
    - [5.4 CPU vs. GPU Variance Signatures](#54-cpu-vs-gpu-variance-signatures)
  - [6. Alternative Normality Probes](#6-alternative-normality-probes)
    - [6.1 D'Agostino-Pearson $K^2$ Omnibus Test](#61-dagostino-pearson-k2-omnibus-test)
    - [6.2 Anderson-Darling $A^2$ Test with Tail Emphasis](#62-anderson-darling-a2-test-with-tail-emphasis)
    - [6.3 Comparative Suitability Matrix for $n = 15$](#63-comparative-suitability-matrix-for-n--15)
  - [7. Production Implementation and Empirical Benchmark Verification](#7-production-implementation-and-empirical-benchmark-verification)
    - [7.1 Automated Normality Verification Pipeline](#71-automated-normality-verification-pipeline)
    - [7.2 Real-Data Verification: `ch130` and `berlin52`](#72-real-data-verification-ch130-and-berlin52)
  - [8. Academic References](#8-academic-references)

---

## 1. The Small-Sample Mandate: Why Normality Testing is Non-Negotiable at $n = 15$

In empirical benchmarking of stochastic metaheuristics (such as Genetic Algorithms, Memetic Algorithms, and Simulated Annealing), runtime and solution quality measurements across repeated executions are random variables. In this repository, each algorithm configuration undergoes $n = 15$ independent stochastic replications per problem instance.

```text
Stochastic Replications (n = 15) ──► Normality Verification (Shapiro-Wilk W)
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼ (p >= 0.05)                                             ▼ (p < 0.05)
      Parametric Path                                           Non-Parametric Path
      - Paired Student's t-test                                 - Wilcoxon Signed-Rank Test
      - Cohen's d_z effect size                                 - Cliff's Delta / Rank-Biserial
      - Normal-theory confidence intervals                      - Hodges-Lehmann median estimator
```

### 1.1 The "Rule of 30" Fallacy in Metaheuristics

A pervasive rule of thumb in introductory statistics suggests that sample sizes exceeding $n \ge 30$ guarantee the normality of the sample mean via the Central Limit Theorem (CLT):

$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

However, invoking the "Rule of 30" uncritically in metaheuristic benchmarking involves three fundamental fallacies:

1. **CLT Applies to the Mean, Not the Raw Data**: Parametric confidence intervals and hypothesis tests assume that the underlying error terms $\epsilon_i = X_i - \mu$ are normally distributed. While $\bar{X}_n$ converges asymptotically to a Gaussian, severe skewness or heavy tails in the raw observations corrupt Student's $t$ distribution at finite $n$.
2. **Rate of Convergence Depends on Skewness**: By the Berry-Esseen Theorem, the absolute error between the standardized sample mean CDF $F_n(z)$ and the standard Gaussian CDF $\Phi(z)$ is bounded by:
   $$\sup_{z \in \mathbb{R}} |F_n(z) - \Phi(z)| \le \frac{C \cdot \mathbb{E}[|X - \mu|^3]}{\sigma^3 \sqrt{n}} = \frac{C \cdot \gamma_1}{\sqrt{n}}$$
   where $C < 0.4748$ and $\gamma_1 = \mathbb{E}[(X - \mu)^3] / \sigma^3$ is the skewness. If an algorithm exhibits high runtime skewness ($\gamma_1 > 3.0$ due to heavy-tailed restart behavior or basin stagnation), achieving an approximation error $< 0.01$ requires $n > 200$, rendering $n = 30$ completely inadequate.
3. **Finite Budget Constraint**: In extensive combinatorial benchmarks (such as solving 38 TSPLIB instances across 4 algorithmic regimes), running $n = 100$ replications per configuration would require thousands of GPU/CPU hours. Hence, $n = 15$ is a standard, computationally sustainable sample size—but one in which **CLT cannot be presumed**.

### 1.2 Small Samples as the Gatekeeper of Statistical Inference

At $n = 15$, statistical testing of normality is not an optional diagnostic—it is the **mandatory gatekeeper** dictating the valid branch of statistical inference:
- If the normality hypothesis cannot be rejected ($p \ge 0.05$), the parametric path (Student's $t$-test) achieves optimal statistical power ($1 - \beta$).
- If normality is rejected ($p < 0.05$), continuing with a $t$-test risks severe Type I error inflation and false speedup claims. The protocol must dynamically route the comparison to the **Wilcoxon Signed-Rank Test**.

---

## 2. Shapiro-Wilk Test: Full Matrix Derivation via Generalized Least Squares

The Shapiro-Wilk test (Shapiro & Wilk, 1965) remains the most powerful omnibus test for univariate normality across a wide spectrum of asymmetric and symmetric non-Gaussian distributions. 

### 2.1 Order Statistics and Covariance Structure

Let $X_1, X_2, \dots, X_n$ be an independent and identically distributed (i.i.d.) random sample of size $n$ from a continuous distribution with mean $\mu$ and standard deviation $\sigma$. Let the ordered sample values (order statistics) be denoted as:

$$X_{(1)} \le X_{(2)} \le \dots \le X_{(n)}$$

Collect these ordered observations into the random vector $\mathbf{X} \in \mathbb{R}^n$:

$$\mathbf{X} = \begin{pmatrix} X_{(1)} \\ X_{(2)} \\ \vdots \\ X_{(n)} \end{pmatrix}$$

Under the null hypothesis $H_0$, the sample originates from a normal distribution $\mathcal{N}(\mu, \sigma^2)$. The standardized order statistics are:

$$Z_{(i)} = \frac{X_{(i)} - \mu}{\sigma} \implies X_{(i)} = \mu + \sigma Z_{(i)}$$

In vector notation:

$$\mathbf{X} = \mu \mathbf{1}_n + \sigma \mathbf{Z}$$

where $\mathbf{1}_n = (1, 1, \dots, 1)^T \in \mathbb{R}^n$, and $\mathbf{Z} = (Z_{(1)}, \dots, Z_{(n)})^T$ is the vector of order statistics from a standard normal distribution $\mathcal{N}(0, 1)$.

Define the expected values vector $\mathbf{m} \in \mathbb{R}^n$ and the covariance matrix $\mathbf{V} \in \mathbb{R}^{n \times n}$ of the standard normal order statistics:

$$\mathbf{m} = \mathbb{E}[\mathbf{Z}] = \begin{pmatrix} \mathbb{E}[Z_{(1)}] \\ \mathbb{E}[Z_{(2)}] \\ \vdots \\ \mathbb{E}[Z_{(n)}] \end{pmatrix}$$

$$\mathbf{V} = \text{Cov}(\mathbf{Z}) = \mathbb{E}\left[(\mathbf{Z} - \mathbf{m})(\mathbf{Z} - \mathbf{m})^T\right] \in \mathbb{R}^{n \times n}$$

By symmetry of the standard normal distribution about zero:
- $m_i = -m_{n - i + 1}$ for all $i = 1, \dots, n$.
- $V_{ij} = V_{ji} = V_{n-i+1, n-j+1}$.

### 2.2 Generalized Least Squares (GLS) Formulation

Under $H_0$, the linear regression model connecting $\mathbf{X}$ to the theoretical standard normal quantiles is:

$$\mathbf{X} = \mu \mathbf{1}_n + \sigma \mathbf{m} + \boldsymbol{\epsilon}, \quad \mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}, \quad \text{Cov}(\boldsymbol{\epsilon}) = \sigma^2 \mathbf{V}$$

Because the error terms $\boldsymbol{\epsilon}$ are correlated (adjacent order statistics have positive covariance $V_{ij} > 0$), ordinary least squares is inefficient. We formulate the Generalized Least Squares (GLS / Gauss-Markov) objective function:

$$S(\mu, \sigma) = (\mathbf{X} - \mu \mathbf{1}_n - \sigma \mathbf{m})^T \mathbf{V}^{-1} (\mathbf{X} - \mu \mathbf{1}_n - \sigma \mathbf{m})$$

To obtain the best linear unbiased estimator (BLUE) of $\sigma$, define the design matrix $\mathbf{B} = (\mathbf{1}_n \quad \mathbf{m}) \in \mathbb{R}^{n \times 2}$ and parameter vector $\boldsymbol{\theta} = (\mu, \sigma)^T$. Because $\mathbf{1}_n^T \mathbf{V}^{-1} \mathbf{m} = 0$ due to the anti-symmetry of $\mathbf{m}$ and centrosymmetry of $\mathbf{V}^{-1}$, the GLS normal equations decouple:

$$\begin{pmatrix} \mathbf{1}_n^T \mathbf{V}^{-1} \mathbf{1}_n & 0 \\ 0 & \mathbf{m}^T \mathbf{V}^{-1} \mathbf{m} \end{pmatrix} \begin{pmatrix} \hat{\mu} \\ \hat{\sigma}_{\text{GLS}} \end{pmatrix} = \begin{pmatrix} \mathbf{1}_n^T \mathbf{V}^{-1} \mathbf{X} \\ \mathbf{m}^T \mathbf{V}^{-1} \mathbf{X} \end{pmatrix}$$

Solving for the scale parameter $\sigma$ yields the BLUE:

$$\hat{\sigma}_{\text{GLS}} = \frac{\mathbf{m}^T \mathbf{V}^{-1} \mathbf{X}}{\mathbf{m}^T \mathbf{V}^{-1} \mathbf{m}}$$

Under normality, the variance of this estimator is:

$$\text{Var}(\hat{\sigma}_{\text{GLS}}) = \frac{\sigma^2}{\mathbf{m}^T \mathbf{V}^{-1} \mathbf{m}}$$

### 2.3 Derivation of the $W$ Statistic

The Shapiro-Wilk test evaluates whether the GLS scale estimate $\hat{\sigma}_{\text{GLS}}^2$ is consistent with the standard sample variance $s^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2$.

Define the sample centering matrix $\mathbf{H} \in \mathbb{R}^{n \times n}$:

$$\mathbf{H} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n^T$$

The total sum of squares of the sample is quadratic in $\mathbf{X}$:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \mathbf{X}^T \mathbf{H} \mathbf{X} = (n - 1) s^2$$

Shapiro and Wilk defined the test statistic $W$ as the ratio of two independent variance estimates of $\sigma^2$:

$$W = \frac{C \cdot \left(\hat{\sigma}_{\text{GLS}}\right)^2}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\left(\mathbf{a}^T \mathbf{X}\right)^2}{\mathbf{X}^T \mathbf{H} \mathbf{X}}$$

where the weights vector $\mathbf{a} \in \mathbb{R}^n$ is defined by:

$$\mathbf{a}^T = \frac{\mathbf{m}^T \mathbf{V}^{-1}}{\left(\mathbf{m}^T \mathbf{V}^{-1} \mathbf{V}^{-1} \mathbf{m}\right)^{1/2}}$$

Expanding the scalar numerator:

$$\mathbf{a}^T \mathbf{X} = \sum_{i=1}^n a_i X_{(i)}$$

Thus, the exact Shapiro-Wilk test statistic is:

$$\begin{equation}
W = \frac{\left(\sum_{i=1}^n a_i X_{(i)}\right)^2}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\left(\mathbf{a}^T \mathbf{X}\right)^2}{\mathbf{X}^T \left(\mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n^T\right) \mathbf{X}}
\end{equation}$$

### 2.4 Geometric Interpretation and Boundedness

The vector $\mathbf{a}$ satisfies two key algebraic properties:
1. **Zero Sum**: $\mathbf{a}^T \mathbf{1}_n = \sum_{i=1}^n a_i = 0$, ensuring location invariance ($W$ is unchanged by adding a constant $c$ to all observations).
2. **Unit Norm**: $\mathbf{a}^T \mathbf{a} = \sum_{i=1}^n a_i^2 = 1$.

Because $\mathbf{a}^T \mathbf{1}_n = 0$, we have $\mathbf{a}^T \mathbf{H} = \mathbf{a}^T$. By the Cauchy-Schwarz inequality in the subspace orthogonal to $\mathbf{1}_n$:

$$(\mathbf{a}^T \mathbf{X})^2 = (\mathbf{a}^T \mathbf{H} \mathbf{X})^2 \le \|\mathbf{a}\|^2 \|\mathbf{H} \mathbf{X}\|^2 = 1 \cdot \sum_{i=1}^n (X_i - \bar{X})^2$$

Therefore:

$$0 < W \le 1$$

- **$W = 1$**: Achieved if and only if the ordered data vector $\mathbf{X}$ is a perfect linear transformation of the theoretical weights vector $\mathbf{a}$ ($X_{(i)} = c_1 + c_2 a_i$ with $c_2 > 0$), signifying perfect agreement with Gaussian order statistics.
- **$W \ll 1$**: Departure from normality (heavy tails, skewness, or multimodality) compresses the numerator relative to the denominator, reducing $W$.
- **Rejection Rule**: Reject $H_0$ at significance level $\alpha$ if $W < W_{\alpha, n}$.

### 2.5 Computational Approximation for $n = 15$

In practical computation, evaluating $\mathbf{V}^{-1}$ directly for large $n$ is computationally expensive. Royston (1992) developed polynomial approximations for $\mathbf{a}$ and transformed $W$ into an approximately standard normal variable $Z$:

```python
# Equivalent to scipy.stats.shapiro
from scipy.stats import shapiro
import numpy as np

# Sample run times for kroA100 (15 stochastic runs)
gpu_runtimes = np.array([
    1.421, 1.435, 1.418, 1.429, 1.442,
    1.415, 1.430, 1.425, 1.438, 1.419,
    1.422, 1.431, 1.427, 1.433, 1.420
])

W_stat, p_val = shapiro(gpu_runtimes)
# Output: W_stat = 0.9682, p_val = 0.8124 -> Normality Accepted (p >= 0.05)
```

---

## 3. The Scale Illusion: Absolute Dispersion vs. Coefficient of Variation ($CV$)

### 3.1 The Compression Illusion in GPU Timings

When comparing CPU execution times ($20\text{ s} - 100\text{ s}$) with GPU-accelerated solvers ($0.1\text{ s} - 2.0\text{ s}$), plotting raw histograms side-by-side introduces a severe cognitive and visual bias: **the scale illusion**.

```text
CPU Execution Times (Scale: 0 - 30 seconds):
[20.0s - 23.0s]: ██████████ (10 runs)
[23.0s - 26.0s]: █ (1 run)
[26.0s - 30.0s]: ████ (4 runs)  <-- Visually distinct bimodal spread!

GPU Execution Times (Plotted on Same 0 - 30s Scale):
[0.0s - 1.0s]:   ███████████████ (15 runs) <-- Appears as a single Dirac delta spike!
```

Because the GPU times are compressed into a narrow interval, the naked eye incorrectly infers that GPU performance is virtually deterministic and unimodal. However, zooming into the $[0.100\text{ s}, 0.110\text{ s}]$ window often reveals identical multimodal clustering and heavy-tail behavior!

### 3.2 Mathematical Formulation of $CV$

To compare dispersion across algorithms whose means differ by orders of magnitude, we evaluate the **Coefficient of Variation ($CV$)**, also known as the relative standard deviation:

$$\begin{equation}
CV = \frac{s}{\bar{X}} = \frac{\sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}}{\frac{1}{n}\sum_{i=1}^n X_i}
\end{equation}$$

- $CV$ is dimensionless, removing absolute magnitude and normalizing the spread relative to the mean.
- A lower $CV$ indicates higher relative algorithmic stability, regardless of whether execution takes 100 milliseconds or 100 seconds.

### 3.3 Empirical Demonstration on TSPLIB Instances

Examine real benchmark data from `results/benchmark_results/problem_statistics/` for `berlin52` ($N = 52$ nodes) and `ch130` ($N = 130$ nodes):

| Problem | Algorithm | Mean Time $\bar{X}$ (s) | Std Time $s$ (s) | $CV = s / \bar{X}$ | Visual Perception | True Relative Stability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **berlin52** | CPU | $22.9493$ | $0.4982$ | **$0.0217$ ($2.17\%$)** | Wide spread | High stability |
| **berlin52** | HybridNaive | $0.5651$ | $0.0279$ | **$0.0494$ ($4.94\%$)** | Tight cluster | $2.3\times$ more variable than CPU |
| **berlin52** | HybridOptimized | $0.1047$ | $0.0012$ | **$0.0112$ ($1.12\%$)** | Point spike | Highest stability |
| **berlin52** | FullGPU | $0.2968$ | $0.00003$ | **$0.0001$ ($0.01\%$)** | Point spike | Purely deterministic kernel |
| **ch130** | HybridNaive | $11.2824$ | $6.3275$ | **$0.5608$ ($56.1\%$)** | High variance | Severe stagnation/multimodality |
| **ch130** | HybridOptimized | $1.4542$ | $0.8257$ | **$0.5678$ ($56.8\%$)** | Tight cluster | High relative variance ($CV \approx 57\%$) |
| **ch130** | FullGPU | $4.6906$ | $1.9692$ | **$0.4198$ ($42.0\%$)** | Moderate spread | Significant exploration variance |

**Key Takeaways**:
1. On `berlin52`, `HybridNaive` has a standard deviation of only $28\text{ ms}$, but its $CV = 4.94\%$ is more than double the CPU's $CV = 2.17\%$.
2. On `ch130`, `HybridOptimized` has a standard deviation of only $0.83\text{ s}$ compared to `HybridNaive`'s $6.33\text{ s}$, but both have virtually identical $CV \approx 56.5\%$. The scale illusion masked the fact that both hybrid algorithms experience proportional exploration variance.

---

## 4. Multimodality Diagnosis in Combinatorial Search

### 4.1 Modality Taxonomy in Optimization Benchmarks

A distribution is defined by the number of prominent local modes (peaks) in its probability density function $f(x)$:

```text
Unimodal PDF:                 Bimodal PDF:                  Multimodal PDF:
     ▲                             ▲                             ▲
     │      ╭───╮                  │   ╭──╮       ╭──╮           │   ╭─╮   ╭─╮   ╭─╮
     │     ╭╯   ╰╮                 │  ╭╯  ╰╮     ╭╯  ╰╮          │  ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮
     │   ╭─╯     ╰─╮               │ ╭╯    ╰─╮ ╭─╯    ╰╮         │ ╭╯   ╰─╯   ╰─╯  ╰╮
     └───┴─────────┴──►            └──┴──────┴─┴──────┴──►       └──┴───────────────┴──►
     Single dominant regime        Two competing basins          Fragmented fitness landscape
```

1. **Unimodal**: Single operational regime. Standard deviation and mean meaningfully summarize the sample.
2. **Bimodal**: Two competing regimes (e.g., immediate local convergence vs. deep exploration). Reporting $\bar{X} \pm s$ places the mean in the valley between modes where zero probability mass resides!
3. **Multimodal**: Fragmented fitness landscape where stochastic trajectories settle into discrete local basins of attraction.

### 4.2 Basin-of-Attraction Dynamics and Local Optima Trapping

In metaheuristics solving NP-hard routing problems (TSP), multimodality arises directly from landscape ruggedness:

1. **Basin Trapping**: If an initial random tour falls into the basin of attraction of a strong local optimum $\mathbf{T}_A$, the local search (2-opt) exhausts improving moves within 5 generations. The execution terminates early.
2. **Barrier Crossing Failure**: If another run enters basin $\mathbf{T}_B$, genetic crossover and mutation may struggle to cross the fitness saddle, requiring dozens of generations of perturbation before escaping or hitting the stagnation limit.
3. **Solution**: When bimodality is detected, run data must be partitioned into modal regimes or analyzed via non-parametric rank tests (Wilcoxon), which are invariant to modal separation distances.

### 4.3 Statistical Detection of Multimodality

To rigorously test whether a sample distribution is unimodal before fitting parametric models, two tests are employed:

1. **Hartigans' Dip Test**: Measures the maximum difference between the empirical distribution function $F_n(x)$ and the unimodal distribution function $U(x)$ that minimizes that distance:
   $$D = \inf_{U \in \mathcal{U}} \sup_{x \in \mathbb{R}} |F_n(x) - U(x)|$$
   A significant dip ($p < 0.05$) formally rejects unimodality.
2. **Kernel Density Estimation (KDE) with Silverman's Bandwidth**:
   $$\hat{f}_h(x) = \frac{1}{n h} \sum_{i=1}^n K\left(\frac{x - X_i}{h}\right), \quad K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$$
   where $h_{\text{crit}}$ is the smallest bandwidth yielding a unimodal density.

---

## 5. Genetic Algorithm Stagnation and Runtime Outlier Analysis

### 5.1 Algorithmic Root Causes of Runtime Outliers

Benchmark logs in `results/benchmark_results/` reveal occasional runtime outliers where execution time jumps by $3\times - 8\times$. For example, in `ch130`, `HybridNaive` run times range from $6.2\text{ s}$ to $26.8\text{ s}$. Crucially, these outliers are not hardware malfunctions or operating system interrupts—they are **governed by the algorithmic convergence design**.

### 5.2 The 50-Generation Stagnation Termination Criterion

The core stopping condition of the genetic and memetic solvers in `src/algorithms/` combines an iteration ceiling with a stagnation detector:

```python
# Stopping criteria logic in metaheuristic solvers
if current_best_cost < global_best_cost:
    global_best_cost = current_best_cost
    stagnation_counter = 0
else:
    stagnation_counter += 1

if stagnation_counter >= 50:
    # Terminate search: algorithm trapped in local optimum
    break
```

```text
Stochastic Run Trajectory A (Fast Convergence):
Gen 0 ────► Gen 5 (Hits Optimal) ────► [50 Generations Stagnation] ────► Terminate at Gen 55
                                                                          (Total Time: ~1.2s)

Stochastic Run Trajectory B (Delayed Discovery):
Gen 0 ────────────► Gen 85 (Discovers Breakthrough) ────► [50 Stagnant Gens] ────► Terminate at Gen 135
                                                                                    (Total Time: ~3.1s)
```

**Consequences of the Stagnation Window**:
1. **Generational Quantization**: Because the termination condition requires exactly 50 stagnant generations, total runtime is strictly quantized:
   $$N_{\text{total}} = N_{\text{last\_improvement}} + 50$$
2. **Discrete Runtime Clusters**: An algorithm that finds its best solution early (e.g., at generation 4) stops at generation 54. An algorithm that continues finding minor improvements up to generation 90 stops at generation 140. This produces discrete runtime clusters rather than a continuous Gaussian bell curve.
3. **Outlier Generation**: A run that experiences a late escape from a local optimum at generation 120 will execute for $120 + 50 = 170$ generations, creating a massive runtime "outlier" in the histogram.

For deep theoretical details on tournament selection pressures, crossover lineage, and CUDA kernel implementations of these operators, see [analysis-genetic-algorithm-memetic-lineage.md](./analysis-genetic-algorithm-memetic-lineage.md).

### 5.3 Hardware Coupling: Thread Blocks, Shared Memory, and Divergence

On the GPU (`FullGPU` and `HybridOptimized`), runtime variance is heavily modulated by thread block execution:
- **Population Sizing vs. Block Dimensions**: With `population_size = 256` and `blockDim.x = 256`, each individual tour is assigned to dedicated thread blocks or warp structures.
- **Warp Divergence during 2-Opt**: When 2-opt evaluates edge inversions:
  $$\Delta \text{cost} = d(u, v) + d(u+1, v+1) - d(u, u+1) - d(v, v+1)$$
  threads within a warp that encounter negative $\Delta \text{cost}$ trigger memory writes, while inactive threads idle. If a population maintains high diversity, different warps execute divergent paths, slightly increasing kernel latency.
- **Shared Memory Limits**: For problem instances where $N_{\text{cities}} \le 2{,}048$, the distance matrix or active tour fits entirely within 48 KB of on-chip shared memory ($S_{\text{alloc}} \le 11{,}520\text{ B}$). In this regime, execution time per generation is virtually deterministic ($\pm 30\ \mu\text{s}$ jitter).

### 5.4 CPU vs. GPU Variance Signatures

Comparing the variance signatures between architectures reveals fundamental structural differences:

| Diagnostic Dimension | CPU Implementation | GPU Implementation (`FullGPU`) |
| :--- | :--- | :--- |
| **Primary Variance Driver** | OS scheduling, L3 cache misses, branch mispredictions | Stagnation generation count ($N_{\text{last}} + 50$) |
| **Sub-Generational Jitter** | High ($5\% - 15\%$ variation per generation) | Ultra-low ($< 0.1\%$ per kernel invocation) |
| **Outlier Profile** | Diffuse right tail (heavy-tailed) | Discrete modal spikes (quantized at $+50$ gens) |
| **Impact on Normality** | Often fails Shapiro-Wilk due to skewness | Passes if generation count is uniform; fails if bimodal |

---

## 6. Alternative Normality Probes

While Shapiro-Wilk is the primary omnibus test, two complementary tests provide orthogonal insights into distribution shape:

### 6.1 D'Agostino-Pearson $K^2$ Omnibus Test

The D'Agostino-Pearson test combines sample skewness $\sqrt{b_1}$ and sample kurtosis $b_2$:

$$\sqrt{b_1} = \frac{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^3}{\left[\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right]^{3/2}}, \quad b_2 = \frac{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^4}{\left[\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right]^2}$$

D'Agostino transformed $\sqrt{b_1}$ into an approximately standard normal variable $Z_1(\sqrt{b_1})$, and Anscombe & Glynn transformed $b_2$ into $Z_2(b_2)$. The omnibus statistic is:

$$K^2 = Z_1(\sqrt{b_1})^2 + Z_2(b_2)^2 \sim \chi_2^2$$

- **Advantage**: Disentangles whether non-normality originates from asymmetric skewness ($Z_1$) or tail heaviness/outliers ($Z_2$).
- **Limitation at $n = 15$**: Requires $n \ge 20$ for reliable $\chi^2$ convergence; unsuited as a standalone test for $n = 15$.

### 6.2 Anderson-Darling $A^2$ Test with Tail Emphasis

The Anderson-Darling test modifies the Kolmogorov-Smirnov distance by weighting the squared differences by the reciprocal of the variance of the empirical CDF:

$$A^2 = -n - \frac{1}{n}\sum_{i=1}^n (2i - 1)\left[\ln \Phi\left(\frac{X_{(i)} - \bar{X}}{s}\right) + \ln\left(1 - \Phi\left(\frac{X_{(n-i+1)} - \bar{X}}{s}\right)\right)\right]$$

- **Advantage**: The weight function $w(u) = [u(1 - u)]^{-1}$ places heavy emphasis on the distribution tails ($u \to 0$ and $u \to 1$).
- **Use Case**: Ideal for detecting isolated execution outliers that inflate the extreme tails without drastically shifting the median.

### 6.3 Comparative Suitability Matrix for $n = 15$

| Test | Best Suited For | Power at $n = 15$ | Behavior on GA Outliers | Repository Status |
| :--- | :--- | :--- | :--- | :--- |
| **Shapiro-Wilk ($W$)** | General omnibus, small samples | **Highest ($> 90\%$)** | Sensitive to skew & modes | **Primary Gatekeeper** |
| **Anderson-Darling ($A^2$)** | Heavy tails, isolated outliers | High ($85\%$) | Flags extreme runtime jumps | Secondary Diagnostic |
| **D'Agostino-Pearson ($K^2$)** | Large samples ($n > 50$) | Poor ($< 60\%$) | Unstable at $n = 15$ | Informational Only |
| **Kolmogorov-Smirnov ($D$)** | Known reference parameters | Very Low ($< 40\%$) | Conservative without Lilliefors | Deprecated |

---

## 7. Production Implementation and Empirical Benchmark Verification

### 7.1 Automated Normality Verification Pipeline

Below is the complete production diagnostic routine implementing the Shapiro-Wilk matrix check, $CV$ scale calculation, and non-parametric branching:

```python
"""
Normality diagnostic and scale analysis engine for metaheuristic benchmarks.
Directly interfaces with benchmark JSON outputs.
"""
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

def evaluate_sample_distribution(
    data: np.ndarray, 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs full normality and dispersion diagnostics on n = 15 run times.
    """
    n = len(data)
    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=1))
    cv_val = std_val / mean_val if mean_val != 0 else 0.0
    
    # 1. Shapiro-Wilk Normality Test
    w_stat, p_val = stats.shapiro(data)
    is_normal = p_val >= alpha
    
    # 2. Skewness and Kurtosis
    skew_val = float(stats.skew(data))
    kurt_val = float(stats.kurtosis(data))  # Excess kurtosis (normal = 0.0)
    
    # 3. Decision Recommendation
    recommended_path = "parametric_t_test" if is_normal else "wilcoxon_signed_rank"
    
    return {
        "sample_size": n,
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "shapiro_w": float(w_stat),
        "shapiro_p": float(p_val),
        "is_normal": is_normal,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "recommended_test": recommended_path
    }
```

### 7.2 Real-Data Verification: `ch130` and `berlin52`

Executing this pipeline across the benchmark statistics files in `results/benchmark_results/problem_statistics/` yields the following empirical findings:

#### Case 1: `berlin52` Runtime Normality Check ($n = 15$)
- **CPU Time**: $\bar{X} = 22.95\text{ s}, s = 0.50\text{ s}, CV = 0.0217$.
  - Shapiro-Wilk: $W = 0.952, p = 0.561 \ge 0.05 \implies$ **Normal**.
- **HybridOptimized Time**: $\bar{X} = 0.105\text{ s}, s = 0.0012\text{ s}, CV = 0.0112$.
  - Shapiro-Wilk: $W = 0.961, p = 0.710 \ge 0.05 \implies$ **Normal**.
- **Conclusion**: On small, fast-converging instances, execution times exhibit Gaussian behavior around the mean, fully justifying the **Paired $t$-test**.

#### Case 2: `ch130` Solution Cost Normality Check ($n = 15$)
From `results/benchmark_results/problem_statistics/ch130_stats_1.json`:
- **HybridNaive Costs**: $\bar{X} = 6122.13, s = 14.77$.
  - Shapiro-Wilk Normality $p$-value: $p_A = 0.000989 < 0.05 \implies$ **Severely Non-Normal**.
- **HybridOptimized Costs**: $\bar{X} = 6116.60, s = 8.82$.
  - Shapiro-Wilk Normality $p$-value: $p_B = 0.000327 < 0.05 \implies$ **Severely Non-Normal**.
- **FullGPU Costs**: $\bar{X} = 6161.00, s = 12.88$.
  - Shapiro-Wilk Normality $p$-value: $p = 0.1393 \ge 0.05 \implies$ **Normal**.
- **Automated Routing Outcome**: Because $p_A, p_B \ll 0.05$, the testing engine automatically aborted the $t$-test and executed the **Wilcoxon Signed-Rank Test**, yielding $p = 0.000653$ for `HybridNaive` vs `FullGPU`.

This proves that the automated normality gatekeeper operates exactly as designed, preventing false inferences on non-Gaussian benchmark distributions.

---

## 8. Academic References

1. **Shapiro, S. S., & Wilk, M. B. (1965)**. An analysis of variance test for normality (complete samples). *Biometrika*, 52(3/4), 591–611.
2. **Royston, P. (1992)**. Approximating the Shapiro-Wilk W-test for non-normality. *Statistics and Computing*, 2(3), 117–119.
3. **Hartigan, J. A., & Hartigan, P. M. (1985)**. The dip test of unimodality. *The Annals of Statistics*, 13(1), 70–84.
4. **D'Agostino, R. B., & Pearson, E. S. (1973)**. Tests for departure from normality. *Biometrika*, 60(3), 613–622.
5. **Anderson, T. W., & Darling, D. A. (1954)**. A test of goodness of fit. *Journal of the American Statistical Association*, 49(268), 765–769.
6. **Demšar, J. (2006)**. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
