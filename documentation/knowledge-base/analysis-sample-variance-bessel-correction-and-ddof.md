---
title: "Sample Variance, Bessel's Correction, and Delta Degrees of Freedom (ddof)"
description: >
  Rigorous mathematical derivation and practical guide to Bessel's correction, sample
  versus population variance, geometric degrees of freedom in R^N, and NumPy's ddof
  parameter. Quantifies the empirical underestimation of standard deviation across sample
  sizes (N in [2, 30]) and demonstrates the impact on Student's t confidence intervals
  and Coefficient of Variation (CV) in metaheuristic benchmarking.
created: 2026-11-11
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-11
    modifications:
      - description: >
          Authored formal derivation of Bessel's correction: algebraic expectation proof,
          geometric hyperplane projection (N-1 degrees of freedom), NumPy ddof API semantics,
          bias quantification table across benchmark sample sizes, and CI error propagation.
related_files:
  - "[guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md)"
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
  - "[statistics.py](../../src/benchmarking/analysis/statistics.py)"
tags:
  - analysis/benchmarking
  - analysis/mathematics
  - analysis/statistics
  - benchmark/statistical-tests
  - guide/methodology
  - guide/statistics
  - guide/testing
  - notes/mathematics
  - notes/theory
  - review/quality
---

# Sample Variance, Bessel's Correction, and Delta Degrees of Freedom (ddof)

- [Sample Variance, Bessel's Correction, and Delta Degrees of Freedom (ddof)](#sample-variance-bessels-correction-and-delta-degrees-of-freedom-ddof)
  + [1. Executive Mathematical Overview](#1-executive-mathematical-overview)
  + [2. Complete Mathematical Derivation of Bessel's Correction](#2-complete-mathematical-derivation-of-bessels-correction)
    - [2.1 Population vs. Sample Variance Definitions](#21-population-vs-sample-variance-definitions)
    - [2.2 Algebraic Proof of Downward Estimator Bias](#22-algebraic-proof-of-downward-estimator-bias)
    - [2.3 Exact Compensation by the Factor $\\frac{N}{N-1}$](#23-exact-compensation-by-the-factor-fracnn-1)
  + [3. Geometric Interpretation and Degrees of Freedom](#3-geometric-interpretation-and-degrees-of-freedom)
    - [3.1 The Constraint Hyperplane in $\\mathbb{R}^N$](#31-the-constraint-hyperplane-in-mathbbrn)
    - [3.2 Orthogonal Projection and the Loss of 1 Degree of Freedom](#32-orthogonal-projection-and-the-loss-of-1-degree-of-freedom)
  + [4. The `ddof` Parameter in NumPy and SciPy](#4-the-ddof-parameter-in-numpy-and-scipy)
    - [4.1 API Semantics and the Generalized Divisor $N - \\text{ddof}$](#41-api-semantics-and-the-generalized-divisor-n---textddof)
    - [4.2 Practical Differences: `ddof=0` (MLE) vs. `ddof=1` (MVUE)](#42-practical-differences-ddof0-mle-vs-ddof1-mvue)
    - [4.3 Repository Implementation Anchor (`statistics.py`)](#43-repository-implementation-anchor-statisticspy)
  + [5. Practical Impact on Metaheuristic Benchmarking](#5-practical-impact-on-metaheuristic-benchmarking)
    - [5.1 Quantification of Standard Deviation Bias Across Sample Sizes](#51-quantification-of-standard-deviation-bias-across-sample-sizes)
    - [5.2 Impact on Student's $t$ Confidence Intervals and False Positive Rates](#52-impact-on-students-t-confidence-intervals-and-false-positive-rates)
    - [5.3 Relevance to Dissertation Table 4 and Telemetry Coefficient of Variation (CV)](#53-relevance-to-dissertation-table-4-and-telemetry-coefficient-of-variation-cv)
  + [6. References](#6-references)

---

## 1. Executive Mathematical Overview

In empirical algorithmic benchmarking, quantifying the dispersion of runtimes and solution costs across stochastic repetitions is as critical as measuring central tendency. However, computing the variance of a finite sample of $N$ experimental runs using the naive population divisor $N$ introduces a systematic negative bias:

$$\mathbb{E}\left[ \frac{1}{N} \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \frac{N - 1}{N} \sigma^2 < \sigma^2$$

This downward bias arises because deviations are measured from the **sample mean** $\bar{X}$ (which minimizes the sum of squared deviations for that specific sample) rather than the true **population mean** $\mu$. Bessel's correction compensates for this shrinkage by replacing the divisor $N$ with $N - 1$, yielding the Minimum Variance Unbiased Estimator (MVUE) $s^2$.

In python's computational ecosystem (NumPy and SciPy), this distinction is controlled by the `ddof` (Delta Degrees of Freedom) parameter. While `numpy.var` and `numpy.std` default to `ddof=0` (the biased Maximum Likelihood Estimator), rigorous inferential statistics and confidence interval construction strictly require `ddof=1`.

---

## 2. Complete Mathematical Derivation of Bessel's Correction

### 2.1 Population vs. Sample Variance Definitions

Let $X$ be a random variable distributed according to an unknown probability distribution with true population mean $\mu = \mathbb{E}[X]$ and finite population variance $\sigma^2 = \text{Var}(X) = \mathbb{E}[(X - \mu)^2]$.

Consider an independent and identically distributed (i.i.d.) random sample of size $N$:

$$X_1, X_2, \dots, X_N \overset{\text{i.i.d.}}{\sim} (\mu, \sigma^2)$$

The **sample mean** $\bar{X}$ is defined as:

$$\bar{X} = \frac{1}{N} \sum_{i=1}^N X_i$$

By the linearity of expectation:

$$\mathbb{E}[\bar{X}] = \frac{1}{N} \sum_{i=1}^N \mathbb{E}[X_i] = \frac{1}{N} (N \mu) = \mu$$

The variance of the sample mean (standard error squared) is:

$$\text{Var}(\bar{X}) = \mathbb{E}\left[ (\bar{X} - \mu)^2 \right] = \text{Var}\left( \frac{1}{N} \sum_{i=1}^N X_i \right) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}(X_i) = \frac{1}{N^2} (N \sigma^2) = \frac{\sigma^2}{N}$$

Now consider two candidate estimators for the population variance $\sigma^2$:

1. **The Naive Sample Variance ($S_n^2$)**:
   $$S_n^2 = \frac{1}{N} \sum_{i=1}^N (X_i - \bar{X})^2$$
2. **The Bessel-Corrected Sample Variance ($s^2$)**:
   $$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (X_i - \bar{X})^2$$

### 2.2 Algebraic Proof of Downward Estimator Bias

To evaluate the expectation of the naive estimator $S_n^2$, we expand the deviation term $(X_i - \bar{X})$ by adding and subtracting the population mean $\mu$:

$$X_i - \bar{X} = (X_i - \mu) - (\bar{X} - \mu)$$

Squaring both sides yields:

$$(X_i - \bar{X})^2 = (X_i - \mu)^2 - 2(X_i - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^2$$

Summing this expansion across all $N$ observations:

$$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2(\bar{X} - \mu) \sum_{i=1}^N (X_i - \mu) + \sum_{i=1}^N (\bar{X} - \mu)^2$$

Notice that by the definition of $\bar{X}$:

$$\sum_{i=1}^N (X_i - \mu) = \sum_{i=1}^N X_i - N\mu = N\bar{X} - N\mu = N(\bar{X} - \mu)$$

Substituting this identity into the middle term:

$$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2(\bar{X} - \mu) \cdot N(\bar{X} - \mu) + N(\bar{X} - \mu)^2$$

$$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2N(\bar{X} - \mu)^2 + N(\bar{X} - \mu)^2$$

$$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - N(\bar{X} - \mu)^2$$

Now, we take the mathematical expectation of both sides:

$$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \sum_{i=1}^N \mathbb{E}\left[ (X_i - \mu)^2 \right] - N \cdot \mathbb{E}\left[ (\bar{X} - \mu)^2 \right]$$

Using the fundamental expectations derived in Section 2.1:

1. $\mathbb{E}[(X_i - \mu)^2] = \sigma^2$ for every observation $i$.
2. $\mathbb{E}[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \frac{\sigma^2}{N}$.

Substituting these values:

$$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \sum_{i=1}^N \sigma^2 - N \left( \frac{\sigma^2}{N} \right) = N \sigma^2 - \sigma^2 = (N - 1) \sigma^2$$

Dividing by the sample size $N$:

$$\mathbb{E}\left[ S_n^2 \right] = \mathbb{E}\left[ \frac{1}{N} \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \frac{N - 1}{N} \sigma^2$$

$$\mathbb{E}\left[ S_n^2 \right] = \sigma^2 - \frac{\sigma^2}{N}$$

**Conclusion**: The naive sample variance $S_n^2$ systematically underestimates the true population variance by exactly $\frac{\sigma^2}{N}$.

### 2.3 Exact Compensation by the Factor $\frac{N}{N-1}$

Because the expectation operator is linear, multiplying $S_n^2$ by the reciprocal of the bias factor $\frac{N}{N - 1}$ removes the bias completely:

$$s^2 = \frac{N}{N - 1} S_n^2 = \frac{N}{N - 1} \left( \frac{1}{N} \sum_{i=1}^N (X_i - \bar{X})^2 \right) = \frac{1}{N - 1} \sum_{i=1}^N (X_i - \bar{X})^2$$

Taking the expectation:

$$\mathbb{E}[s^2] = \frac{N}{N - 1} \mathbb{E}[S_n^2] = \frac{N}{N - 1} \left( \frac{N - 1}{N} \sigma^2 \right) = \sigma^2$$

Thus, $s^2$ is an **unbiased estimator** of $\sigma^2$ ($\text{Bias}(s^2) = \mathbb{E}[s^2] - \sigma^2 = 0$).

---

## 3. Geometric Interpretation and Degrees of Freedom

### 3.1 The Constraint Hyperplane in $\mathbb{R}^N$

The algebraic derivation of Bessel's correction has an elegant geometric interpretation in $N$-dimensional Euclidean space $\mathbb{R}^N$ (Kenney & Keeping, 1951).

Let the sample of $N$ observations be represented as a single point or vector $\mathbf{X} \in \mathbb{R}^N$:

$$\mathbf{X} = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_N \end{pmatrix}$$

The deviation vector from the sample mean is:

$$\mathbf{d} = \mathbf{X} - \bar{X} \mathbf{1} = \begin{pmatrix} X_1 - \bar{X} \\ X_2 - \bar{X} \\ \vdots \\ X_N - \bar{X} \end{pmatrix}$$

where $\mathbf{1} = (1, 1, \dots, 1)^{\mathsf{T}}$ is the vector of ones.

The squared Euclidean length (norm squared) of the deviation vector $\mathbf{d}$ is precisely the sum of squared deviations:

$$\|\mathbf{d}\|^2 = \sum_{i=1}^N (X_i - \bar{X})^2$$

However, the components of $\mathbf{d}$ are not linearly independent. By definition of the sample mean, their sum is identically zero:

$$\sum_{i=1}^N (X_i - \bar{X}) = \mathbf{1}^{\mathsf{T}} \mathbf{d} = 0$$

This single linear equation defines an $(N - 1)$-dimensional linear subspace (a hyperplane passing through the origin in $\mathbb{R}^N$) orthogonal to the vector $\mathbf{1}$.

### 3.2 Orthogonal Projection and the Loss of 1 Degree of Freedom

The sample mean vector $\bar{X} \mathbf{1}$ is the orthogonal projection of $\mathbf{X}$ onto the 1-dimensional subspace spanned by $\mathbf{1}$:

$$P_{\mathbf{1}} = \frac{\mathbf{1} \mathbf{1}^{\mathsf{T}}}{\mathbf{1}^{\mathsf{T}} \mathbf{1}} = \frac{1}{N} \mathbf{J}_N$$

where $\mathbf{J}_N$ is the $N \times N$ matrix of all ones. The deviation vector $\mathbf{d}$ is the projection of $\mathbf{X}$ onto the orthogonal complement subspace:

$$\mathbf{d} = (I_N - P_{\mathbf{1}}) \mathbf{X} = M \mathbf{X}$$

The projection matrix $M = I_N - \frac{1}{N} \mathbf{J}_N$ has the following properties:

1. **Symmetric**: $M^{\mathsf{T}} = M$.
2. **Idempotent**: $M^2 = M$.
3. **Trace (Dimensionality)**:
   $$\text{tr}(M) = \text{tr}(I_N) - \frac{1}{N} \text{tr}(\mathbf{J}_N) = N - \frac{1}{N}(N) = N - 1$$

The squared norm $\|\mathbf{d}\|^2 = \mathbf{X}^{\mathsf{T}} M \mathbf{X}$ represents the squared length of a random vector projected into an $(N - 1)$-dimensional space. The expected value of a quadratic form $\mathbf{X}^{\mathsf{T}} M \mathbf{X}$ for i.i.d. random variables with variance $\sigma^2$ is:

$$\mathbb{E}[\mathbf{X}^{\mathsf{T}} M \mathbf{X}] = \sigma^2 \cdot \text{tr}(M) = (N - 1) \sigma^2$$

Therefore, dividing $\|\mathbf{d}\|^2$ by the true dimension of the subspace ($N - 1$) is required to obtain an unbiased estimate of $\sigma^2$. The "loss of 1 degree of freedom" corresponds directly to the single constraint $\sum (X_i - \bar{X}) = 0$ used to estimate $\mu$ with $\bar{X}$.

---

## 4. The `ddof` Parameter in NumPy and SciPy

### 4.1 API Semantics and the Generalized Divisor $N - \text{ddof}$

In the scientific python stack (NumPy, SciPy, Pandas), variance and standard deviation functions generalize the calculation via the parameter `ddof` (Delta Degrees of Freedom):

$$\text{Divisor} = N - \text{ddof}$$

$$\text{Var}_{\text{ddof}}(X) = \frac{1}{N - \text{ddof}} \sum_{i=1}^N (X_i - \bar{X})^2$$

$$\text{Std}_{\text{ddof}}(X) = \sqrt{\frac{1}{N - \text{ddof}} \sum_{i=1}^N (X_i - \bar{X})^2}$$

The parameter specifies the number of degrees of freedom subtracted from the sample size $N$.

### 4.2 Practical Differences: `ddof=0` (MLE) vs. `ddof=1` (MVUE)

| Property | `ddof=0` (NumPy Default) | `ddof=1` (Inferential Standard) |
| :--- | :--- | :--- |
| **Formula Divisor** | $N$ | $N - 1$ |
| **Statistical Role** | Maximum Likelihood Estimator (MLE) | Minimum Variance Unbiased Estimator (MVUE) |
| **Expected Value** | $\mathbb{E}[S_n^2] = \frac{N-1}{N} \sigma^2$ (Biased) | $\mathbb{E}[s^2] = \sigma^2$ (Unbiased) |
| **Target Population** | Assumes the array represents the *entire population* | Assumes the array is a *sample* drawn from a larger population |
| **python Standard** | `numpy.var`, `numpy.std` default | `statistics.variance`, `pandas.Series.std` default |
| **Usage in Benchmarking** | **Incorrect** for sample runs | **Mandatory** for sample runs ($N=15, 30$) |

> [!IMPORTANT] **The Default Divergence Trap:**
> Notice that `pandas.DataFrame.std()` and the python standard library `statistics.stdev()` default to `ddof=1`, whereas `numpy.std()` defaults to `ddof=0`. Mixing NumPy and Pandas calculations without explicitly specifying `ddof=1` produces inconsistent standard deviations and corrupted t-test statistics!

### 4.3 Repository Implementation Anchor (`statistics.py`)

In this repository, statistical evaluations of runtime and solution quality explicitly enforce `ddof=1`. As implemented in [`src/benchmarking/analysis/statistics.py`](../../src/benchmarking/analysis/statistics.py#L238-L255):

```python
# Excerpt from src/benchmarking/analysis/statistics.py
mean_a = float(np.mean(data_a))
mean_b = float(np.mean(data_b))

# Explicit ddof=1 ensures unbiased sample standard deviation s
std_a = float(np.std(data_a, ddof=1))
std_b = float(np.std(data_b, ddof=1))

se_a = std_a / np.sqrt(n)
se_b = std_b / np.sqrt(n)
t_critical = stats.t.ppf(0.975, n - 1)

ci_95_a = (mean_a - t_critical * se_a, mean_a + t_critical * se_a)
ci_95_b = (mean_b - t_critical * se_b, mean_b + t_critical * se_b)

std_diff = float(np.std(differences, ddof=1))
se_diff = std_diff / np.sqrt(n)
ci_95_diff = (
    mean_diff - t_critical * se_diff,
    mean_diff + t_critical * se_diff,
)
```

---

## 5. Practical Impact on Metaheuristic Benchmarking

### 5.1 Quantification of Standard Deviation Bias Across Sample Sizes

While Bessel's correction completely eliminates bias in sample **variance** ($s^2$), taking the square root to obtain sample **standard deviation** introduces a minor non-linear bias via Jensen's inequality:

$$\mathbb{E}[s] = \mathbb{E}\left[\sqrt{s^2}\right] \le \sqrt{\mathbb{E}[s^2]} = \sigma$$

However, the naive standard deviation $\sigma_n = \sqrt{S_n^2}$ (obtained via `ddof=0`) suffers from a severe, systematic downward bias factor:

$$\text{Bias Factor} = \sqrt{\frac{N - 1}{N}}$$

$$\text{Underestimation of } \sigma \text{ (\%)} = \left( 1 - \sqrt{\frac{N - 1}{N}} \right) \times 100\%$$

The table below documents the percentage underestimation of variance and standard deviation when using `ddof=0` across typical sample sizes in metaheuristic benchmarking:

| Sample Size ($N$) | Divisor Ratio $\frac{N-1}{N}$ | Variance Underestimation | Std Dev Factor $\sqrt{\frac{N-1}{N}}$ | Std Dev Underestimation | Practical Context in Benchmarks |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$2$** | $0.5000$ | **$-50.00\%$** | $0.7071$ | **$-29.29\%$** | Minimum paired run (smoke tests) |
| **$3$** | $0.6667$ | **$-33.33\%$** | $0.8165$ | **$-18.35\%$** | Quick triaging / parameter sweeps |
| **$5$** | $0.8000$ | **$-20.00\%$** | $0.8944$ | **$-10.56\%$** | Fast exploratory profiling |
| **$10$** | $0.9000$ | **$-10.00\%$** | $0.9487$ | **$-5.13\%$** | Pilot statistical screening |
| **$15$** | $0.9333$ | **$-6.67\%$** | $0.9656$ | **$-3.44\%$** | **Our Core TSPLIB Protocol ($N=15$)** |
| **$20$** | $0.9500$ | **$-5.00\%$** | $0.9747$ | **$-2.53\%$** | Extended robustness evaluation |
| **$30$** | $0.9667$ | **$-3.33\%$** | $0.9832$ | **$-1.68\%$** | Central Limit Theorem asymptotic rule |
| **$50$** | $0.9800$ | **$-2.00\%$** | $0.9899$ | **$-1.01\%$** | Large-scale statistical validation |
| **$100$** | $0.9900$ | **$-1.00\%$** | $0.9950$ | **$-0.50\%$** | Asymptotically negligible difference |

### 5.2 Impact on Student's $t$ Confidence Intervals and False Positive Rates

In parametric inferential statistics, the $95\%$ confidence interval for the population mean $\mu$ based on a sample of size $N$ is:

$$CI_{95\%} = \bar{X} \pm t_{N-1, 0.975} \cdot \frac{s}{\sqrt{N}}$$

where $t_{N-1, 0.975}$ is the critical quantile of Student's $t$-distribution with $\nu = N - 1$ degrees of freedom (Student, 1908).

If a researcher mistakenly computes the sample standard error using `ddof=0` ($\sigma_n = \sqrt{S_n^2}$):

$$\widehat{SE}_{\text{biased}} = \frac{\sigma_n}{\sqrt{N}} = \sqrt{\frac{N - 1}{N}} \cdot \frac{s}{\sqrt{N}} = \sqrt{\frac{N - 1}{N}} \cdot SE_{\text{true}}$$

**The Consequence of Underestimated Standard Error**:

1. **Artificially Narrow Confidence Intervals**: At $N = 5$, the confidence interval width is shrunk by $10.56\%$. At our protocol size $N = 15$, it is shrunk by $3.44\%$.
2. **Inflated Type I Error Rates (False Positives)**: In paired $t$-tests:
   $$t_{\text{calc}} = \frac{\bar{D}}{\widehat{SE}_{\text{biased}}} = \sqrt{\frac{N}{N - 1}} \cdot \frac{\bar{D}}{SE_{\text{true}}}$$
   The calculated $t$-statistic is artificially inflated by $\sqrt{N / (N - 1)}$ ($+11.8\%$ at $N=5$, $+3.56\%$ at $N=15$).
3. **Compound Distortion with Gaussian $z$-Scores**: If the researcher simultaneously uses $z_{0.975} = 1.960$ instead of $t_{14, 0.975} = 2.145$ (which is $9.44\%$ larger), the total margin of error is underestimated by:
   $$\text{Total Margin Deflation} = \left(1 - \frac{1.960 \times 0.9656}{2.145}\right) \times 100\% = 11.77\%$$
   This combined error causes actual coverage to drop well below the advertised $95\%$ nominal level, falsely declaring insignificant performance differences as statistically significant.

### 5.3 Relevance to Dissertation Table 4 and Telemetry Coefficient of Variation (CV)

In Chapter 5 of the thesis (`second_draft.tex`), Table 4 documents the fine-grained runtime, speedup ratios, and stability metrics across benchmark instances. Two metrics are directly governed by Bessel's correction:

1. **Coefficient of Variation ($CV$)**:
   The relative dispersion metric is defined as:
   $$CV = \frac{s}{\bar{X}} \times 100\% = \frac{\text{np.std}(X, \text{ddof}=1)}{\text{np.mean}(X)} \times 100\%$$
   Computing $CV$ with `ddof=0` underestimates algorithmic variability by $3.44\%$ across our $N = 15$ runs. In hardware stability analyses where we assess whether GPU kernel execution exhibits deterministic timing ($CV < 2\%$), using `ddof=0` would generate false claims of kernel stability.
2. **Paired Difference Margins ($\pm \text{CI}_{95\%}$)**:
   Reporting $\Delta = \bar{X}_{\text{CPU}} - \bar{X}_{\text{GPU}} \pm \text{CI}_{95\%}$ requires $s_{\text{diff}}$ calculated strictly with `ddof=1`. This guarantees that confidence intervals around speedup ratios have exact, valid frequentist coverage across all 113 TSPLIB95 instances.

---

## 6. References

1. **Bessel, F. W.** (1815). *Ueber die Bestimmung der Genauigkeit der Beobachtungen*. Zeitschrift für Astronomie und verwandte Wissenschaften, 1, 125–130.
2. **Bessel, F. W.** (1838). *Untersuchungen über die Wahrscheinlichkeit der Beobachtungsfehler*. Astronomische Nachrichten, 15(358-359), 369–404.
3. **Kenney, J. F., & Keeping, E. S.** (1951). *Mathematics of Statistics, Part Two*. 2nd ed., D. Van Nostrand Company, New York.
4. **Student [Gosset, W. S.]** (1908). *The probable error of a mean*. Biometrika, 6(1), 1–25 (`student1908probable`).
