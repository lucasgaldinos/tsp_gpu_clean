---
title: "Statistical Testing Guide: Mathematical Derivations and the Demšar Protocol"
description: >
  Rigorous statistical foundation and practical guide for metaheuristic benchmarking:
  derivation of the Gamma function and Student's t distribution, geometric intuition of
  degrees of freedom, non-parametric effect sizes, and the two-tier Demšar (2006) protocol
  covering paired Wilcoxon, Friedman omnibus, Iman-Davenport, Nemenyi, and CD diagrams.
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
          Initial creation of the definitive statistical testing guide detailing
          mathematical derivations (Gamma function, Student's t PDF normalization, degrees
          of freedom geometry), effect sizes (Cliff's Delta, rank-biserial), and the
          two-tier Demšar protocol (within-instance Wilcoxon vs across-instances Friedman/Nemenyi).
      - description: >
          Corrected CD diagram ranks to mathematically valid sum sum(R_j) = 10.0 for k=4,
          labeled all 4 algorithms (HybridOpt, FullGPU, HybridNaive, CPU) with exact ticks,
          clique bars (CD = 0.761, N=38), added warning regarding scipy studentized_range
          scaling convention (division by sqrt(2)), and synchronized related_files.
related_files:
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
  - "[notes_types_of_variables_statistics.md](./notes_types_of_variables_statistics.md)"
  - "[analysis-genetic-algorithm-memetic-lineage.md](./analysis-genetic-algorithm-memetic-lineage.md)"
  - "[guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[statistics.py](../../src/benchmarking/analysis/statistics.py)"
  - "[genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py)"
tags:
  - guide/testing
  - guide/analysis
  - analysis/statistical
  - analysis/benchmarking
  - notes/theory
  - notes/mathematics
  - benchmark/statistical-tests
  - review/quality
  - algorithm/tsp
  - algorithm/genetic-algorithm
---

# Statistical Testing Guide: Mathematical Derivations and the Demšar Protocol

- [Statistical Testing Guide: Mathematical Derivations and the Demšar Protocol](#statistical-testing-guide-mathematical-derivations-and-the-demšar-protocol)
  - [1. Mathematical Foundations of Parametric Distributions](#1-mathematical-foundations-of-parametric-distributions)
    - [1.1 Derivation of the Euler Gamma Function $\Gamma(\alpha)$](#11-derivation-of-the-euler-gamma-function-gammaalpha)
    - [1.2 Derivation of the Student's $t$-Distribution PDF Normalization Constant](#12-derivation-of-the-students-t-distribution-pdf-normalization-constant)
    - [1.3 Geometric Intuition of Degrees of Freedom ($df = n - 1$)](#13-geometric-intuition-of-degrees-of-freedom-df--n---1)
    - [1.4 Two-Tailed Critical Values for Benchmark Runs ($n = 15$)](#14-two-tailed-critical-values-for-benchmark-runs-n--15)
  - [2. Non-Parametric Effect Sizes for Stochastic Metaheuristics](#2-non-parametric-effect-sizes-for-stochastic-metaheuristics)
    - [2.1 Why $p$-Values Alone Are Insufficient](#21-why-p-values-alone-are-insufficient)
    - [2.2 Cliff's Delta ($\delta$)](#22-cliffs-delta-delta)
    - [2.3 Matched-Pairs Rank-Biserial Correlation ($r_{\text{rb}}$)](#23-matched-pairs-rank-biserial-correlation-r_textrb)
  - [3. The Two-Tier Demšar (2006) Benchmarking Protocol](#3-the-two-tier-demšar-2006-benchmarking-protocol)
    - [3.1 The Incommensurability Problem Across Benchmark Instances](#31-the-incommensurability-problem-across-benchmark-instances)
    - [3.2 Tier 1: Within-Instance Pairwise Comparison (Wilcoxon Signed-Rank)](#32-tier-1-within-instance-pairwise-comparison-wilcoxon-signed-rank)
    - [3.3 Tier 2: Across-Instances Omnibus Comparison (Friedman \& Iman-Davenport)](#33-tier-2-across-instances-omnibus-comparison-friedman--iman-davenport)
    - [3.4 Post-Hoc Pairwise Multi-Comparison Procedures](#34-post-hoc-pairwise-multi-comparison-procedures)
    - [3.5 Constructing and Interpreting Critical Difference (CD) Diagrams](#35-constructing-and-interpreting-critical-difference-cd-diagrams)
  - [4. Implementation Recipe for Algorithmic Benchmarks](#4-implementation-recipe-for-algorithmic-benchmarks)
  - [5. Academic References](#5-academic-references)

---

## 1. Mathematical Foundations of Parametric Distributions

Rigorous benchmarking of stochastic metaheuristics (such as GPU-accelerated Genetic Algorithms and 2-opt local search) requires understanding the exact mathematical underpinnings of statistical test statistics.

### 1.1 Derivation of the Euler Gamma Function $\Gamma(\alpha)$

The Gamma function extends the discrete factorial operator to the continuous and complex domain. For $\alpha > 0$, it is defined as the improper integral:

$$\Gamma(\alpha) = \int_0^\infty x^{\alpha - 1} e^{-x} \, dx$$

#### Fundamental Recurrence Relation
Applying integration by parts with $u = x^{\alpha - 1}$ and $dv = e^{-x} \, dx$:

$$\frac{du}{dx} = (\alpha - 1) x^{\alpha - 2}, \qquad v = -e^{-x}$$

$$\Gamma(\alpha) = \left[ -x^{\alpha - 1} e^{-x} \right]_0^\infty - \int_0^\infty -(\alpha - 1) x^{\alpha - 2} e^{-x} \, dx$$

Because $\lim_{x \to \infty} x^{\alpha - 1} e^{-x} = 0$ for all finite $\alpha$, the boundary evaluation vanishes:

$$\Gamma(\alpha) = (\alpha - 1) \int_0^\infty x^{\alpha - 2} e^{-x} \, dx = (\alpha - 1) \Gamma(\alpha - 1)$$

Replacing $\alpha$ with $\alpha + 1$ yields the fundamental functional equation:

$$\Gamma(\alpha + 1) = \alpha \Gamma(\alpha)$$

#### Base Values and Factorial Equivalence
For $\alpha = 1$:

$$\Gamma(1) = \int_0^\infty e^{-x} \, dx = \left[ -e^{-x} \right]_0^\infty = 0 - (-1) = 1$$

By mathematical induction, for any positive integer $k \in \mathbb{N}^+$:

$$\Gamma(k) = (k - 1)!$$

#### Half-Integer Evaluation ($\Gamma(1/2)$)
For $\alpha = 1/2$:

$$\Gamma\left(\frac{1}{2}\right) = \int_0^\infty x^{-1/2} e^{-x} \, dx$$

Substitute $x = u^2 \implies dx = 2u \, du$:

$$\Gamma\left(\frac{1}{2}\right) = \int_0^\infty \frac{1}{u} e^{-u^2} (2u \, du) = 2 \int_0^\infty e^{-u^2} \, du$$

Recognizing the famous Gaussian (Euler-Poisson) integral $\int_{-\infty}^\infty e^{-u^2} \, du = \sqrt{\pi}$:

$$\Gamma\left(\frac{1}{2}\right) = 2 \left( \frac{\sqrt{\pi}}{2} \right) = \mathbf{\sqrt{\pi}}$$

---

### 1.2 Derivation of the Student's $t$-Distribution PDF Normalization Constant

Published in 1908 by **William Sealy Gosset** under the pseudonym *"Student"*, the $t$-distribution models the sample mean when the true population variance $\sigma^2$ is unknown and estimated from a small sample $n$.

#### Definition from Normal and Chi-Square Variates
Let $Z \sim \mathcal{N}(0, 1)$ and $V \sim \chi^2(\nu)$ be two independent random variables, where $\nu$ represents degrees of freedom.
The random variable $T$ is defined as:

$$T = \frac{Z}{\sqrt{V / \nu}}$$

The probability density functions of $Z$ and $V$ are:

$$f_Z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}, \quad z \in \mathbb{R}$$

$$f_V(v) = \frac{1}{2^{\nu/2} \Gamma(\nu/2)} v^{\nu/2 - 1} e^{-v/2}, \quad v > 0$$

Because $Z$ and $V$ are mutually independent, their joint PDF is the product:

$$f_{Z, V}(z, v) = f_Z(z) f_V(v) = \frac{1}{\sqrt{2\pi} 2^{\nu/2} \Gamma(\nu/2)} v^{\nu/2 - 1} \exp\left( -\frac{z^2 + v}{2} \right)$$

#### Bivariate Transformation
Define the transformation:

$$T = \frac{Z}{\sqrt{V / \nu}}, \qquad U = V$$

The inverse transformation is:

$$Z = T \sqrt{\frac{U}{\nu}}, \qquad V = U$$

Compute the Jacobian matrix $J$ of the inverse transformation:

$$J = \begin{pmatrix} \frac{\partial z}{\partial t} & \frac{\partial z}{\partial u} \\ \frac{\partial v}{\partial t} & \frac{\partial v}{\partial u} \end{pmatrix} = \begin{pmatrix} \sqrt{\frac{u}{\nu}} & \frac{t}{2\sqrt{\nu u}} \\ 0 & 1 \end{pmatrix}$$

The absolute determinant of the Jacobian is:

$$|J| = \left| \sqrt{\frac{u}{\nu}} \cdot 1 - 0 \right| = \sqrt{\frac{u}{\nu}} = u^{1/2} \nu^{-1/2}$$

The joint density $f_{T, U}(t, u)$ is:

$$f_{T, U}(t, u) = f_{Z, V}\left( t \sqrt{\frac{u}{\nu}}, \; u \right) \cdot |J|$$

Substitute $z^2 = t^2 \frac{u}{\nu}$:

$$f_{T, U}(t, u) = \frac{1}{\sqrt{2\pi} 2^{\nu/2} \Gamma(\nu/2)} u^{\nu/2 - 1} \exp\left( -\frac{t^2 \frac{u}{\nu} + u}{2} \right) \cdot u^{1/2} \nu^{-1/2}$$

Combine terms of $u$:

$$f_{T, U}(t, u) = \frac{1}{\sqrt{2\pi \nu} 2^{\nu/2} \Gamma(\nu/2)} u^{\frac{\nu + 1}{2} - 1} \exp\left( -u \cdot \frac{1 + t^2/\nu}{2} \right)$$

#### Marginalization Over $U$
To find the marginal density $f_T(t)$, integrate over all $u \in (0, \infty)$:

$$f_T(t) = \int_0^\infty f_{T, U}(t, u) \, du = \frac{1}{\sqrt{2\pi \nu} 2^{\nu/2} \Gamma(\nu/2)} \int_0^\infty u^{\frac{\nu + 1}{2} - 1} \exp\left( -u \cdot \frac{1 + t^2/\nu}{2} \right) \, du$$

Let $w = u \cdot \frac{1 + t^2/\nu}{2} \implies du = \left( \frac{1 + t^2/\nu}{2} \right)^{-1} dw$.
The integral matches the standard Gamma integral $\int_0^\infty w^{\alpha - 1} e^{-w} dw = \Gamma(\alpha)$ with $\alpha = \frac{\nu + 1}{2}$:

$$\int_0^\infty u^{\frac{\nu+1}{2}-1} e^{-u \beta} du = \beta^{-\frac{\nu+1}{2}} \Gamma\left( \frac{\nu + 1}{2} \right), \quad \text{where } \beta = \frac{1 + t^2/\nu}{2}$$

Substituting back:

$$f_T(t) = \frac{1}{\sqrt{2\pi \nu} 2^{\nu/2} \Gamma(\nu/2)} \left( \frac{1 + t^2/\nu}{2} \right)^{-\frac{\nu + 1}{2}} \Gamma\left( \frac{\nu + 1}{2} \right)$$

Cancel the powers of 2 ($2^{-\nu/2} \cdot 2^{(\nu+1)/2} = 2^{1/2} = \sqrt{2}$, which cancels the $\sqrt{2}$ in $\sqrt{2\pi\nu}$):

$$f_T(t) = \frac{\Gamma\left( \frac{\nu + 1}{2} \right)}{\sqrt{\pi \nu} \, \Gamma\left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}} \quad \blacksquare$$

This proves the exact closed-form Student's $t$ PDF normalization constant.

---

### 1.3 Geometric Intuition of Degrees of Freedom ($df = n - 1$)

Why is sample variance divided by $n - 1$ instead of $n$?
In high-dimensional Euclidean space $\mathbb{R}^n$, an experimental run of $n$ independent trials produces a single sample vector:

$$\mathbf{x} = (x_1, x_2, \dots, x_n)^T \in \mathbb{R}^n$$

The sample space has $n$ orthogonal dimensions of geometric freedom.
Now define the sample mean:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$

Notice that the vector of means is the projection of $\mathbf{x}$ onto the 1-dimensional subspace spanned by the all-ones vector $\mathbf{1} = (1, 1, \dots, 1)^T$:

$$\bar{\mathbf{x}} = (\bar{x}, \bar{x}, \dots, \bar{x})^T = \left( \frac{\mathbf{x} \cdot \mathbf{1}}{\|\mathbf{1}\|^2} \right) \mathbf{1}$$

The vector of sample residuals (deviations) is:

$$\mathbf{e} = \mathbf{x} - \bar{\mathbf{x}} = \begin{pmatrix} x_1 - \bar{x} \\ x_2 - \bar{x} \\ \vdots \\ x_n - \bar{x} \end{pmatrix}$$

By definition of the mean, the sum of deviations is identically zero:

$$\sum_{i=1}^n (x_i - \bar{x}) = 0 \iff \mathbf{e} \cdot \mathbf{1} = 0$$

```text
               R^n Space (n Dimensions)
                     ▲
                     │      x (Sample Vector)
                     │     /│
                     │    / │
                     │   /  │  e = x - x_bar (Residuals)
                     │  /   │  Lies in (n-1)-dim Hyperplane!
                     │ /    ▼
                     │/─────┴─────────► 1 = (1, 1, ..., 1)^T (1 Dimension)
                     O      x_bar
```

**Geometric Conclusion:**
The vector of residuals $\mathbf{e}$ is constrained to lie on an **$(n - 1)$-dimensional hyperplane** orthogonal to $\mathbf{1}$. Once the first $n - 1$ deviations are known, the $n$-th deviation is completely determined:

$$e_n = -\sum_{i=1}^{n-1} e_i$$

The sample variance $s^2 = \frac{1}{n-1} \|\mathbf{e}\|^2$ measures the squared length of a vector in an $(n-1)$-dimensional linear subspace. Thus, the system has exactly **$df = n - 1$ degrees of freedom**.

---

### 1.4 Two-Tailed Critical Values for Benchmark Runs ($n = 15$)

In our benchmark experiments, each algorithmic configuration is executed across $n = 15$ independent stochastic seeds per TSP/CVRP instance:

$$df = \nu = n - 1 = 15 - 1 = \mathbf{14}$$

For a standard significance level $\alpha = 0.05$ (two-tailed test, with $\alpha/2 = 0.025$ in each tail):
The critical value $t_{\text{crit}}$ satisfies:

$$\int_{-\infty}^{-t_{\text{crit}}} f_T(t) \, dt = \int_{t_{\text{crit}}}^\infty f_T(t) \, dt = 0.025 \iff F_T(t_{\text{crit}}) = 0.975$$

Evaluating the cumulative Student's $t$ distribution with $\nu = 14$:

$$t_{14, \; 0.975} = \mathbf{2.144787} \approx \mathbf{2.145}$$

**Decision Rule:**
- If the observed test statistic $|t_{\text{stat}}| > 2.145$, reject the null hypothesis $H_0$ at the $95\%$ confidence level.
- If $|t_{\text{stat}}| \le 2.145$, fail to reject $H_0$; the observed difference is indistinguishable from random sampling variance.

---

## 2. Non-Parametric Effect Sizes for Stochastic Metaheuristics

### 2.1 Why $p$-Values Alone Are Insufficient

A $p$-value indicates only whether an observed difference is statistically unlikely to have occurred under the null hypothesis; **it does not quantify the practical magnitude of the performance difference**. In large benchmark samples, a trivial $0.001\%$ speedup can yield $p < 0.001$, whereas a massive $40\%$ speedup in a high-variance small sample might yield $p = 0.06$. Effect size metrics are mandatory for publication.

### 2.2 Cliff's Delta ($\delta$)

Introduced by **Norman Cliff (1993)**, Cliff's Delta is a non-parametric effect size measuring the ordinal dominance of one distribution over another. It requires no normality or equal variance assumptions.

#### Mathematical Definition
Given sample $X = \{x_1, \dots, x_m\}$ from Algorithm A and sample $Y = \{y_1, \dots, y_n\}$ from Algorithm B:

$$\delta = \frac{\#\{x_i > y_j\} - \#\{x_i < y_j\}}{m \cdot n} = \frac{1}{m \cdot n} \sum_{i=1}^m \sum_{j=1}^n \operatorname{sgn}(x_i - y_j)$$

where:
$$\operatorname{sgn}(x - y) = \begin{cases} +1 & \text{if } x > y \\ 0 & \text{if } x = y \\ -1 & \text{if } x < y \end{cases}$$

- **Range:** $\delta \in [-1, +1]$.
- **$\delta = +1$:** Every run of Algorithm A yields a higher value than every run of Algorithm B.
- **$\delta = 0$:** Perfect stochastic equivalence.
- **$\delta = -1$:** Every run of Algorithm B dominates Algorithm A.

#### Interpretation Thresholds (Romano et al., 2006)

| Absolute Value $|\delta|$ | Qualitative Magnitude | Practical Meaning in Routing Solvers |
| :--- | :--- | :--- |
| $|\delta| < 0.147$ | **Negligible** | Stochastic noise; no meaningful difference between solvers. |
| $0.147 \le |\delta| < 0.330$ | **Small** | Noticeable shift, but overlapping performance. |
| $0.330 \le |\delta| < 0.474$ | **Medium** | Clear competitive advantage in solution quality or speed. |
| $|\delta| \ge 0.474$ | **Large** | Decisive algorithmic superiority (e.g., GPU 2-opt vs CPU). |

---

### 2.3 Matched-Pairs Rank-Biserial Correlation ($r_{\text{rb}}$)

When comparing algorithms on paired runs (e.g., the exact same 15 random seeds), the appropriate non-parametric effect size is the **Rank-Biserial Correlation** derived from the Wilcoxon signed-rank test ([Kerby, 2014](#5-academic-references)).

#### Mathematical Definition
Let $W^+$ be the sum of positive signed ranks and $W^-$ be the sum of negative signed ranks:

$$W^+ = \sum_{d_i > 0} R_i, \qquad W^- = \sum_{d_i < 0} R_i$$

Total sum of ranks:
$$W^+ + W^- = \frac{n(n + 1)}{2}$$

The rank-biserial correlation $r_{\text{rb}}$ is:

$$r_{\text{rb}} = \frac{W^+ - W^-}{W^+ + W^-} = \frac{W^+ - W^-}{\frac{n(n+1)}{2}} = \frac{2(W^+ - W^-)}{n(n+1)}$$

- **Range:** $r_{\text{rb}} \in [-1, +1]$.
- If all differences favor Algorithm A ($W^- = 0$), $r_{\text{rb}} = +1.0$.
- If ranks are evenly split ($W^+ = W^-$), $r_{\text{rb}} = 0.0$.

---

## 3. The Two-Tier Demšar (2006) Benchmarking Protocol

In 2006, **Janez Demšar** published the landmark paper in the *Journal of Machine Learning Research*:
> *"Statistical Comparisons of Classifiers over Multiple Data Sets"* (Demšar, 2006).

Demšar demonstrated that the empirical optimization and machine learning communities routinely commit fatal statistical errors by applying standard paired $t$-tests or repeated-measures ANOVA across multiple benchmark instances.

### 3.1 The Incommensurability Problem Across Benchmark Instances

Why can we not average speedups or tour costs across 38 TSPLIB instances and run a $t$-test?
1. **Violation of Commensurability:** In TSPLIB, `pcb442` has tour cost $\approx 50{,}000$, while `pla85900` has cost $\approx 140{,}000{,}000$. A $1\%$ error on `pla85900` ($\Delta = 1{,}400{,}000$) completely dwarfs a $100\%$ error on `pcb442` ($\Delta = 50{,}000$).
2. **Violation of Normality:** Algorithmic runtimes across instances follow heavy-tailed log-normal or Pareto distributions.
3. **Outlier Sensitivity:** A single pathological instance where an algorithm times out will skew the parametric mean and standard deviation, producing false rejections or false non-rejections.

To resolve this, Demšar established the **Two-Tier Benchmarking Protocol**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TWO-TIER DEMŠAR (2006) BENCHMARKING PROTOCOL                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: Within-Instance Pairwise Comparison (n = 15 runs per problem)                 │
│   - Objective: Compare Algorithm A vs Algorithm B on a SINGLE instance.                │
│   - Test: Wilcoxon Signed-Rank Test (paired seeds)                                     │
│   - Effect Size: Rank-Biserial Correlation r_rb or Cliff's Delta                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Across-Instances Omnibus & Multi-Comparison (N instances, k algorithms)        │
│   - Objective: Establish overall performance hierarchy across the ENTIRE benchmark.   │
│   - Step 2.1: Rank all k algorithms on each of the N instances (1 = best, k = worst). │
│   - Step 2.2: Compute average ranks R_j = (1/N) * sum_i r_i^j.                         │
│   - Step 2.3: Friedman Omnibus Test (χ_F^2) and Iman-Davenport Correction (F_F).      │
│   - Step 2.4: If p < 0.05, Post-Hoc Tests (Nemenyi / Dunn-Holm / FDR).                 │
│   - Step 2.5: Critical Difference (CD) Diagram visualization.                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Tier 1: Within-Instance Pairwise Comparison (Wilcoxon Signed-Rank)

When evaluating two algorithms (e.g., GPU 2-Opt vs CPU 2-Opt) on a specific instance (e.g., `ch130.tsp`) across $n = 15$ seeds:
1. Compute pairwise differences $d_i = \text{Cost}_A^{(i)} - \text{Cost}_B^{(i)}$ for seed $i \in \{1, \dots, 15\}$.
2. Exclude zero differences ($d_i = 0$). Let the remaining count be $n_r$.
3. Rank the absolute differences $|d_i|$ in ascending order, assigning average ranks to ties.
4. Sum positive ranks $W^+$ and negative ranks $W^-$. Test statistic is $W = \min(W^+, W^-)$.
5. Compare $W$ against the exact Wilcoxon distribution critical table for $n = 15$.

---

### 3.3 Tier 2: Across-Instances Omnibus Comparison (Friedman & Iman-Davenport)

To compare $k$ algorithms across $N$ benchmark problem instances (e.g., $k = 4$ solvers across $N = 38$ TSPLIB instances):

#### Step 1: Compute Ranks
For each instance $i \in \{1, \dots, N\}$, rank the $k$ algorithms from $1$ (best performance) to $k$ (worst performance). Let $r_i^j$ be the rank of algorithm $j$ on instance $i$. In case of ties, assign the average of the ranks.

Compute the **average rank** for each algorithm $j$:

$$R_j = \frac{1}{N} \sum_{i=1}^N r_i^j$$

#### Step 2: The Friedman Statistic ($\chi_F^2$)
Developed by **Milton Friedman (1937)**, the Friedman test is the non-parametric equivalent of repeated-measures ANOVA:

$$\chi_F^2 = \frac{12N}{k(k + 1)} \left[ \sum_{j=1}^k R_j^2 - \frac{k(k + 1)^2}{4} \right]$$

Under the null hypothesis $H_0$ (all algorithms perform equivalently, so each has expected average rank $\bar{R} = \frac{k+1}{2}$), $\chi_F^2$ is asymptotically distributed as Chi-square with $k - 1$ degrees of freedom:

$$\chi_F^2 \sim \chi^2(k - 1)$$

#### Step 3: The Iman-Davenport Correction ($F_F$)
As proven by **Iman & Davenport (1980)**, Friedman's $\chi_F^2$ is notoriously conservative when $N$ is small or moderate, failing to detect true differences. They derived a dramatically more powerful $F$-distributed statistic:

$$F_F = \frac{(N - 1) \chi_F^2}{N(k - 1) - \chi_F^2}$$

which is distributed according to the $F$-distribution with:

$$\nu_1 = k - 1 \quad \text{and} \quad \nu_2 = (k - 1)(N - 1) \text{ degrees of freedom}$$

If $p < 0.05$ under the $F_F$ test, reject $H_0$ and proceed to post-hoc tests.

---

### 3.4 Post-Hoc Pairwise Multi-Comparison Procedures

Once the omnibus Friedman/Iman-Davenport test rejects $H_0$, we must determine **which specific algorithm pairs differ significantly**.

#### 1. Nemenyi Test (All-vs-All Comparison)
When comparing all $k$ algorithms against each other without designating a control:
Two algorithms $i$ and $j$ perform significantly differently if their average ranks differ by at least the **Critical Difference (CD)**:

$$|R_i - R_j| > CD$$

The Critical Difference is given by:

$$CD = q_\alpha \sqrt{\frac{k(k + 1)}{6N}}$$

where $q_\alpha$ is the critical value for the two-tailed Studentized range statistic divided by $\sqrt{2}$.

**Standard Values of $q_\alpha$ ($\alpha = 0.05$):**
- $k = 2$: $q_{0.05} = 1.960$
- $k = 3$: $q_{0.05} = 2.343$
- $k = 4$: $q_{0.05} = 2.569$
- $k = 5$: $q_{0.05} = 2.728$

> [!WARNING] **Studentized Range Statistic Scaling Convention**
> When obtaining critical values computationally, note that `scipy.stats.studentized_range.ppf(1 - alpha, k, np.inf)` returns Tukey's unscaled Studentized range statistic critical value $Q_\alpha = \sqrt{2} \cdot q_\alpha$ ($\approx 3.633$ for $k = 4, \alpha = 0.05$). When pairing with the standard Demšar denominator $\sqrt{k(k + 1) / (6N)}$, you **must divide** by $\sqrt{2}$:
> $$q_\alpha = \frac{Q_\alpha}{\sqrt{2}} = \frac{\text{studentized\_range.ppf}(1 - \alpha, k, \infty)}{\sqrt{2}} \approx \frac{3.633}{1.4142} \approx 2.569$$
> Alternatively, Tukey's unscaled $Q_\alpha$ can be paired directly with the alternative denominator formulation $CD = Q_\alpha \sqrt{\frac{k(k + 1)}{12N}}$, yielding identical results. Failing to divide by $\sqrt{2}$ when using $\sqrt{6N}$ in the denominator artificially inflates the critical difference by $41.4\%$!

#### 2. Dunn-Bonferroni and Holm Step-Down (Comparison Against Control)
When comparing $k-1$ proposed algorithms against a single baseline/control algorithm (e.g., comparing GPU Hybrid, Monolithic GPU, and Memetic GPU against CPU Baseline):
The standard error of rank differences is:

$$SE = \sqrt{\frac{k(k + 1)}{6N}}$$

Compute the $z$-statistic for each comparison against control algorithm $c$:

$$z_j = \frac{R_j - R_c}{SE}, \qquad p_j = 2 \left( 1 - \Phi(|z_j|) \right)$$

Order the unadjusted $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(k-1)}$.
Under **Holm's Step-Down Procedure** (which strictly controls Family-Wise Error Rate $\alpha = 0.05$ without the extreme conservatism of Bonferroni):
Compare each $p_{(i)}$ sequentially against:

$$\alpha_{\text{Holm}}^{(i)} = \frac{\alpha}{k - i}$$

Reject $H_0^{(i)}$ as long as $p_{(i)} \le \alpha_{\text{Holm}}^{(i)}$. At the first index where $p_{(i)} > \alpha_{\text{Holm}}^{(i)}$, stop testing; all remaining hypotheses are retained.

#### 3. Benjamini-Hochberg False Discovery Rate (FDR)
When testing a large number of pairs (large $k$), controlling the Family-Wise Error Rate (FWER) becomes overly strict. The **Benjamini-Hochberg (1995)** procedure controls the expected proportion of false positive findings (FDR):

$$p_{(i)} \le \frac{i}{m} Q^*$$

where $m = \binom{k}{2}$ and $Q^* = 0.05$.

---

### 3.5 Constructing and Interpreting Critical Difference (CD) Diagrams

A Critical Difference (CD) diagram visualizes the results of the Demšar protocol in a single clean graphic:

```text
       Critical Difference (CD = 0.761 at α = 0.05, N = 38, k = 4)
       ├─────────────────┤

   1.0       1.35      1.85           2.95           3.85      4.0 (Rank Axis)
───┼──────────┼─────────┼──────────────┼──────────────┼────────┼───►
              ▲         ▲              ▲              ▲
   ┌──────────┴─────────┴─┐            │              │
   │      Clique Bar      │            │              │
   │  (|1.85-1.35| <= CD) │            │              │
   └──────────┬─────────┬─┘            │              │
              │         │              │              │
       ┌──────┴───┐ ┌───┴────┐   ┌─────┴───────┐   ┌──┴──┐
       │HybridOpt │ │FullGPU │   │HybridNaive  │   │ CPU │
       │(Best:1.35│ │(1.85)  │   │(2.95)       │   │(3.85│
       └──────────┘ └────────┘   └─────────────┘   └─────┘
                                   (Isolated)     (Isolated:
                                                 Inferior)
```

**How to Read a CD Diagram:**
1. **The Rank Axis:** Ranks are displayed along a horizontal axis from $1.0$ (best performance on the left) to $k.0$ (worst performance on the right). For $k = 4$, average ranks across all algorithms must satisfy the mathematical conservation constraint $\sum_{j=1}^4 R_j = \frac{k(k + 1)}{2} = 10.00$ ($1.35 + 1.85 + 2.95 + 3.85 = 10.00$).
2. **Algorithm Markers:** Each algorithm is plotted along the axis at its exact average rank $R_j$.
3. **The Critical Difference Bar:** A horizontal scale bar at the top displays the length of $CD = 2.569 \sqrt{\frac{20}{6 \times 38}} \approx 0.761$.
4. **Cliques / Thick Connecting Lines:** A thick horizontal line connects any subset of algorithms whose rank difference is **less than $CD$** ($|R_i - R_j| \le CD$).
5. **Statistical Interpretation:**
   - Algorithms connected by the same horizontal bar have **no statistically significant difference** in performance.
   - Any two algorithms **not connected** by a common horizontal bar are **statistically significantly different** at the $\alpha = 0.05$ level.
   - In the diagram above, HybridOpt ($R_1 = 1.35$) and FullGPU ($R_2 = 1.85$) are connected (difference $0.50 \le 0.761$), but both are separated from HybridNaive ($R_3 = 2.95$, difference $1.10 > CD$) and CPU ($R_4 = 3.85$, difference $2.00 > CD$) by gaps exceeding $CD$, proving conclusive superiority over baseline configurations.

---

## 4. Implementation Recipe for Algorithmic Benchmarks

The repository provides production implementations of the Friedman omnibus test and Nemenyi post-hoc ranking in [statistics.py](../../src/benchmarking/analysis/statistics.py#L358-L450). Below is a self-contained reference script demonstrating the mathematical calculation of the Demšar protocol:

```python
"""Demšar (2006) Two-Tier Statistical Protocol for Metaheuristics."""

import numpy as np
from scipy import stats


def friedman_iman_davenport(ranks_matrix: np.ndarray) -> dict[str, float]:
  """Computes Friedman chi-square and Iman-Davenport F_F from an (N, k) rank matrix.

  Args:
      ranks_matrix: Shape (N, k) where N is problem instances, k is algorithms.

  Returns:
      Dict with chi2_F, p_friedman, F_F, p_iman_davenport.
  """
  N, k = ranks_matrix.shape
  avg_ranks = np.mean(ranks_matrix, axis=0)

  # Friedman statistic
  chi2_F = (12.0 * N / (k * (k + 1))) * (np.sum(avg_ranks**2) - (k * (k + 1) ** 2) / 4.0)
  p_friedman = 1.0 - stats.chi2.cdf(chi2_F, df=k - 1)

  # Iman-Davenport correction
  F_F = ((N - 1) * chi2_F) / (N * (k - 1) - chi2_F)
  df1 = k - 1
  df2 = (k - 1) * (N - 1)
  p_iman = 1.0 - stats.f.cdf(F_F, dfn=df1, dfd=df2)

  return {
    "avg_ranks": avg_ranks.tolist(),
    "chi2_F": float(chi2_F),
    "p_friedman": float(p_friedman),
    "F_F": float(F_F),
    "p_iman_davenport": float(p_iman),
  }


def nemenyi_critical_difference(N: int, k: int, alpha: float = 0.05, q_alpha: float | None = None) -> float:
  """Calculates Nemenyi Critical Difference CD for N instances and k algorithms.

  Note: If q_alpha is None, it is derived from scipy.stats.studentized_range.
  scipy returns Tukey's unscaled Q = sqrt(2) * q_alpha; we divide by sqrt(2).
  """
  if q_alpha is None:
    # Scale Tukey's Q critical value by 1 / sqrt(2) to obtain Demšar's q_alpha
    q_alpha = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))
  return float(q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N)))
```

---

## 5. Academic References

1. **Gosset, W. S. ["Student"] (1908).** The Probable Error of a Mean. *Biometrika*, 6(1), 1–25. DOI: [10.1093/biomet/6.1.1](https://doi.org/10.1093/biomet/6.1.1).
2. **Demšar, J. (2006).** Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
3. **Friedman, M. (1937).** The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance. *Journal of the American Statistical Association*, 32(200), 675–701. DOI: [10.1080/01621459.1937.10503522](https://doi.org/10.1080/01621459.1937.10503522).
4. **Iman, R. L., & Davenport, J. M. (1980).** Approximations of the Critical Value of the Kruskal-Wallis and Friedman Tests to Significance Levels by Means of the F-Distribution. *Communications in Statistics - Theory and Methods*, 9(6), 571–595. DOI: [10.1080/03610928008827904](https://doi.org/10.1080/03610928008827904).
5. **Nemenyi, P. B. (1963).** *Distribution-Free Multiple Comparisons*. PhD Thesis, Princeton University.
6. **Cliff, N. (1993).** Dominance Statistics: Ordinal Analyses to Answer Ordinal Questions. *Psychological Bulletin*, 114(3), 494–509. DOI: [10.1037/0033-2909.114.3.494](https://doi.org/10.1037/0033-2909.114.3.494).
7. **Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006).** Appropriate Statistics for Ordinal Level Data: Should We Really Be Using t-test and Cohen's d for Evaluating Group Differences on the NSSE and Other Surveys? In *Proceedings of the Florida Association for Institutional Research*, pp. 1–33.
8. **Kerby, D. S. (2014).** The Simple Difference Formula: An Approach to Teaching Nonparametric Correlation. *Comprehensive Psychology*, 3, Article 1. DOI: [10.2466/11.IT.3.1](https://doi.org/10.2466/11.IT.3.1).
9. **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
10. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. DOI: [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
