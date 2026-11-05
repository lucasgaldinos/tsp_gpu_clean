---
title: "Parametric Statistical Foundations: Gamma Function, Student's t, and Degrees of Freedom"
description: >
  Mathematical foundations of parametric inferential statistics: Euler Gamma function derivation,
  analytical derivation of Student's t distribution, geometric intuition of degrees of freedom in
  R^n, asymptotic convergence to Gaussian, quantile notation (t_{nu, 0.975}), and Bessel's ddof correction.
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
          Extracted and modularized parametric distribution theory from the statistical testing monolith,
          providing complete mathematical derivations for the Gamma function, Student's t PDF,
          geometric degrees of freedom hyperplane projections, Bessel ddof proofs, and asymptotic limits.
related_files:
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[guide-statistical-normality-and-outliers.md](./guide-statistical-normality-and-outliers.md)"
  - "[guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md)"
  - "[guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md)"
  - "[guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md)"
  - "[statistics.py](../../src/benchmarking/analysis/statistics.py)"
tags:
  - guide/analysis
  - guide/testing
  - notes/theory
  - notes/mathematics
  - analysis/statistical
  - analysis/benchmarking
  - benchmark/statistical-tests
  - algorithm/tsp
  - algorithm/genetic-algorithm
  - review/quality
---

# Parametric Statistical Foundations: Gamma Function, Student's t, and Degrees of Freedom

- [Parametric Statistical Foundations: Gamma Function, Student's t, and Degrees of Freedom](#parametric-statistical-foundations-gamma-function-students-t-and-degrees-of-freedom)
  - [1. The Parametric Inference Framework](#1-the-parametric-inference-framework)
  - [2. The Euler Gamma Function $\Gamma(\alpha)$](#2-the-euler-gamma-function-gammaalpha)
    - [2.1 Analytical Definition and Convergence](#21-analytical-definition-and-convergence)
    - [2.2 Derivation of the Fundamental Recurrence Relation](#22-derivation-of-the-fundamental-recurrence-relation)
    - [2.3 Half-Integer Evaluation via the Gaussian Integral](#23-half-integer-evaluation-via-the-gaussian-integral)
    - [2.4 Why the Gamma Function Governs Normalizing Constants](#24-why-the-gamma-function-governs-normalizing-constants)
  - [3. Analytical Derivation of Student's $t$-Distribution](#3-analytical-derivation-of-students-t-distribution)
    - [3.1 The Ratio of Independent Normal and Chi-Square Variates](#31-the-ratio-of-independent-normal-and-chi-square-variates)
    - [3.2 Joint Distribution and Bivariate Change of Variables](#32-joint-distribution-and-bivariate-change-of-variables)
    - [3.3 Marginalization and the Exact Probability Density Function](#33-marginalization-and-the-exact-probability-density-function)
  - [4. Geometric Origin of Degrees of Freedom ($\nu = n - 1$)](#4-geometric-origin-of-degrees-of-freedom-nu--n---1)
    - [4.1 Hyperplane Projection in $n$-Dimensional Space $\mathbb{R}^n$](#41-hyperplane-projection-in-n-dimensional-space-mathbbrn)
    - [4.2 Why Estimating the Sample Mean Costs Exactly One Dimension](#42-why-estimating-the-sample-mean-costs-exactly-one-dimension)
  - [5. Mathematical Proof of Bessel's Correction and Delta Degrees of Freedom (`ddof`)](#5-mathematical-proof-of-bessels-correction-and-delta-degrees-of-freedom-ddof)
    - [5.1 Expectation of the Biased Estimator $S_n^2$ (`ddof = 0`)](#51-expectation-of-the-biased-estimator-s_n2-ddof--0)
    - [5.2 Derivation of the Unbiased Estimator $s^2$ (`ddof = 1`)](#52-derivation-of-the-unbiased-estimator-s2-ddof--1)
    - [5.3 When Does `ddof > 1` Apply?](#53-when-does-ddof--1-apply)
  - [6. Asymptotic Proof: Convergence of Student's $t$ to the Gaussian ($\nu \to \infty$)](#6-asymptotic-proof-convergence-of-students-t-to-the-gaussian-nu-to-infty)
  - [7. Quantile and Tail Notation ($t_{\nu, 1 - \alpha/2}$)](#7-quantile-and-tail-notation-t_nu-1---alpha2)
    - [7.1 Demystifying the $0.975$ Quantile Subscript](#71-demystifying-the-0975-quantile-subscript)
    - [7.2 Numerical Comparison Table: Student's $t$ vs Standard Normal $Z$](#72-numerical-comparison-table-students-t-vs-standard-normal-z)
  - [8. Python Scientific Implementation](#8-python-scientific-implementation)
  - [9. Academic References](#9-academic-references)

---

## 1. The Parametric Inference Framework

In computational benchmarking, parametric inferential statistics allow us to infer properties of an infinite population of stochastic executions from a finite sample of runs ($n = 15$). 

Parametric tests operate under the assumption that observations $X_1, X_2, \dots, X_n$ are independent, identically distributed realizations from a Gaussian distribution:

$$X_i \sim \mathcal{N}(\mu, \sigma^2)$$

When the true population variance $\sigma^2$ is known, the standardized sample mean follows the Standard Normal distribution:

$$Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)$$

However, in experimental metaheuristics, the true population variance $\sigma^2$ is **strictly unknown**. We must estimate $\sigma^2$ using the sample variance $s^2$. Substituting $s$ for $\sigma$ introduces supplementary sampling error, transforming the test statistic into **Student's $t$ statistic**, whose properties depend fundamentally on the **Euler Gamma Function** and the geometry of **degrees of freedom**.

---

## 2. The Euler Gamma Function $\Gamma(\alpha)$

### 2.1 Analytical Definition and Convergence

The Euler Gamma function $\Gamma: \mathbb{R}^+ \to \mathbb{R}^+$ extends the discrete factorial operator to the continuous and complex domain. For any real $\alpha > 0$, it is defined by the improper integral:

$$\Gamma(\alpha) = \int_0^\infty t^{\alpha - 1} e^{-t} \, \mathrm{d}t$$

The integral converges absolutely for all $\alpha > 0$:
- At the lower limit $t \to 0^+$, the integrand behaves as $t^{\alpha - 1}$, which is integrable if and only if $\alpha - 1 > -1 \iff \alpha > 0$.
- At the upper limit $t \to \infty$, the exponential decay $e^{-t}$ dominates any polynomial growth $t^{\alpha - 1}$.

### 2.2 Derivation of the Fundamental Recurrence Relation

Applying integration by parts to $\Gamma(\alpha + 1)$:

$$\Gamma(\alpha + 1) = \int_0^\infty t^\alpha e^{-t} \, \mathrm{d}t$$

Let:
- $u = t^\alpha \implies \mathrm{d}u = \alpha t^{\alpha - 1} \, \mathrm{d}t$
- $\mathrm{d}v = e^{-t} \, \mathrm{d}t \implies v = -e^{-t}$

Using the integration by parts formula $\int u \, \mathrm{d}v = [uv] - \int v \, \mathrm{d}u$:

$$\Gamma(\alpha + 1) = \left[ -t^\alpha e^{-t} \right]_0^\infty - \int_0^\infty (-e^{-t}) \left( \alpha t^{\alpha - 1} \right) \, \mathrm{d}t$$

Evaluating the boundary term:
- As $t \to \infty$, by L'Hôpital's Rule, $\lim_{t \to \infty} \frac{t^\alpha}{e^t} = 0$.
- At $t = 0$, $0^\alpha = 0$ (for $\alpha > 0$).
- Thus, the boundary term vanishes identically: $\left[ -t^\alpha e^{-t} \right]_0^\infty = 0$.

Therefore:

$$\Gamma(\alpha + 1) = \alpha \int_0^\infty t^{\alpha - 1} e^{-t} \, \mathrm{d}t = \alpha \Gamma(\alpha) \quad \blacksquare$$

Evaluating the base case $\alpha = 1$:

$$\Gamma(1) = \int_0^\infty e^{-t} \, \mathrm{d}t = \left[ -e^{-t} \right]_0^\infty = 0 - (-1) = 1$$

By repeated application of the recurrence for positive integers $n \in \mathbb{N}$:

$$\Gamma(n) = (n - 1) \Gamma(n - 1) = (n - 1)(n - 2) \cdots 1 \cdot \Gamma(1) = (n - 1)!$$

Thus, $\Gamma(n)$ matches the discrete factorial $(n - 1)!$.

### 2.3 Half-Integer Evaluation via the Gaussian Integral

In the derivation of Student's $t$ and Chi-Square distributions, degrees of freedom introduce evaluations at half-integers $\Gamma\left(\frac{k}{2}\right)$.

Consider $\Gamma(1/2)$:

$$\Gamma(1/2) = \int_0^\infty t^{-1/2} e^{-t} \, \mathrm{d}t$$

Apply the change of variables $t = u^2 \implies \mathrm{d}t = 2u \, \mathrm{d}u$:

$$\Gamma(1/2) = \int_0^\infty (u^2)^{-1/2} e^{-u^2} (2u \, \mathrm{d}u) = 2 \int_0^\infty e^{-u^2} \, \mathrm{d}u$$

Since $e^{-u^2}$ is symmetric about $u = 0$:

$$2 \int_0^\infty e^{-u^2} \, \mathrm{d}u = \int_{-\infty}^\infty e^{-u^2} \, \mathrm{d}u = \sqrt{\pi}$$

Therefore:

$$\Gamma(1/2) = \sqrt{\pi}$$

Using the recurrence relation $\Gamma(\alpha + 1) = \alpha \Gamma(\alpha)$, we evaluate subsequent half-integers recursively:

$$\Gamma(3/2) = \frac{1}{2} \Gamma(1/2) = \frac{\sqrt{\pi}}{2} \approx 0.8862$$

$$\Gamma(5/2) = \frac{3}{2} \Gamma(3/2) = \frac{3}{4} \sqrt{\pi} \approx 1.3293$$

$$\Gamma(7/2) = \frac{5}{2} \Gamma(5/2) = \frac{15}{8} \sqrt{\pi} \approx 3.3234$$

### 2.4 Why the Gamma Function Governs Normalizing Constants

When deriving continuous probability distributions on hyperspheres or orthogonal coordinate systems, integrating radial components in $n$ dimensions requires evaluating:

$$\int_0^\infty r^{n - 1} e^{-r^2 / 2} \, \mathrm{d}r = 2^{(n/2) - 1} \Gamma\left(\frac{n}{2}\right)$$

The Gamma function emerges naturally as the **volume normalizing constant** of high-dimensional Gaussian shells.

---

## 3. Analytical Derivation of Student's $t$-Distribution

### 3.1 The Ratio of Independent Normal and Chi-Square Variates

Let $X_1, \dots, X_n \sim \text{i.i.d. } \mathcal{N}(\mu, \sigma^2)$. The sample mean $\bar{X}$ and sample variance $s^2$ are:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i, \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

By **Cochran's Theorem**:
1. $\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \implies Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)$
2. $V = \frac{(n - 1)s^2}{\sigma^2} \sim \chi^2(\nu)$, where $\nu = n - 1$ degrees of freedom
3. $Z$ and $V$ are **statistically independent**.

The Student's $t$ statistic is defined as:

$$T = \frac{\bar{X} - \mu}{s / \sqrt{n}} = \frac{\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}}{\sqrt{\frac{s^2}{\sigma^2}}} = \frac{Z}{\sqrt{V / \nu}}$$

### 3.2 Joint Distribution and Bivariate Change of Variables

Since $Z$ and $V$ are independent, their joint probability density function is the product of their marginal densities:

$$f_{Z, V}(z, v) = f_Z(z) \cdot f_V(v)$$

where:

$$f_Z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}, \qquad f_V(v) = \frac{1}{2^{\nu/2} \Gamma(\nu/2)} v^{(\nu/2) - 1} e^{-v/2} \quad (v > 0)$$

Thus:

$$f_{Z, V}(z, v) = \frac{1}{\sqrt{2\pi} 2^{\nu/2} \Gamma(\nu/2)} v^{(\nu/2) - 1} \exp\left( -\frac{z^2 + v}{2} \right)$$

Define the bivariate transformation:

$$\begin{cases} T = \frac{Z}{\sqrt{V / \nu}} \\ U = V \end{cases} \iff \begin{cases} Z = T \sqrt{U / \nu} \\ V = U \end{cases}$$

The Jacobian matrix of the inverse transformation is:

$$\mathbf{J} = \begin{bmatrix} \frac{\partial z}{\partial t} & \frac{\partial z}{\partial u} \\ \frac{\partial v}{\partial t} & \frac{\partial v}{\partial u} \end{bmatrix} = \begin{bmatrix} \sqrt{u / \nu} & \frac{t}{2\sqrt{\nu u}} \\ 0 & 1 \end{bmatrix}$$

The Jacobian determinant is:

$$\det(\mathbf{J}) = \sqrt{\frac{u}{\nu}} = u^{1/2} \nu^{-1/2}$$

The joint density $f_{T, U}(t, u)$ is:

$$f_{T, U}(t, u) = f_{Z, V}\left( t \sqrt{u / \nu}, u \right) |\det(\mathbf{J})|$$

$$f_{T, U}(t, u) = \frac{1}{\sqrt{2\pi\nu} 2^{\nu/2} \Gamma(\nu/2)} u^{(\nu/2) - 1 + 1/2} \exp\left( -\frac{1}{2} \left[ \frac{t^2 u}{\nu} + u \right] \right)$$

$$f_{T, U}(t, u) = \frac{1}{\sqrt{2\pi\nu} 2^{\nu/2} \Gamma(\nu/2)} u^{\frac{\nu + 1}{2} - 1} \exp\left( -u \cdot \frac{1 + t^2/\nu}{2} \right)$$

### 3.3 Marginalization and the Exact Probability Density Function

To find the marginal distribution $f_T(t)$, integrate over all $u \in (0, \infty)$:

$$f_T(t) = \int_0^\infty f_{T, U}(t, u) \, \mathrm{d}u = \frac{1}{\sqrt{2\pi\nu} 2^{\nu/2} \Gamma(\nu/2)} \int_0^\infty u^{\frac{\nu + 1}{2} - 1} \exp\left( -u \cdot \frac{1 + t^2/\nu}{2} \right) \, \mathrm{d}u$$

Recall the definition of the Gamma integral:

$$\int_0^\infty u^{\alpha - 1} e^{-\lambda u} \, \mathrm{d}u = \frac{\Gamma(\alpha)}{\lambda^\alpha}$$

Setting $\alpha = \frac{\nu + 1}{2}$ and $\lambda = \frac{1 + t^2/\nu}{2}$:

$$\int_0^\infty u^{\frac{\nu + 1}{2} - 1} \exp\left( -u \frac{1 + t^2/\nu}{2} \right) \, \mathrm{d}u = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\left( \frac{1 + t^2/\nu}{2} \right)^{\frac{\nu + 1}{2}}} = \frac{\Gamma\left(\frac{\nu + 1}{2}\right) 2^{\frac{\nu + 1}{2}}}{\left( 1 + \frac{t^2}{\nu} \right)^{\frac{\nu + 1}{2}}}$$

Substitute this back into the marginal density expression:

$$f_T(t) = \frac{1}{\sqrt{2\pi\nu} 2^{\nu/2} \Gamma(\nu/2)} \cdot \frac{\Gamma\left(\frac{\nu + 1}{2}\right) 2^{\nu/2} 2^{1/2}}{\left( 1 + \frac{t^2}{\nu} \right)^{\frac{\nu + 1}{2}}}$$

Notice that the powers of 2 cancel: $\frac{2^{\nu/2} \sqrt{2}}{\sqrt{2\pi\nu} 2^{\nu/2}} = \frac{1}{\sqrt{\pi\nu}}$.

We arrive at the **exact Probability Density Function of Student's $t$-Distribution**:

$$f_T(t) = \frac{\Gamma\left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu\pi} \, \Gamma\left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}} \quad \blacksquare$$

---

## 4. Geometric Origin of Degrees of Freedom ($\nu = n - 1$)

### 4.1 Hyperplane Projection in $n$-Dimensional Space $\mathbb{R}^n$

A sample of $n$ independent observations $\mathbf{X} = [X_1, X_2, \dots, X_n]^T$ is a random vector inhabiting an $n$-dimensional Euclidean vector space $\mathbb{R}^n$.

Consider the equiangular vector $\mathbf{1} = [1, 1, \dots, 1]^T \in \mathbb{R}^n$. The sample mean $\bar{X}$ represents the orthogonal projection of $\mathbf{X}$ onto the 1-dimensional subspace spanned by $\mathbf{1}$:

$$\mathbf{P}_{\mathbf{1}} \mathbf{X} = \frac{\mathbf{1}^T \mathbf{X}}{\mathbf{1}^T \mathbf{1}} \mathbf{1} = \frac{\sum_{i=1}^n X_i}{n} \mathbf{1} = \bar{X} \mathbf{1} = \begin{bmatrix} \bar{X} \\ \bar{X} \\ \vdots \\ \bar{X} \end{bmatrix}$$

The residual deviation vector is the orthogonal complement:

$$\mathbf{e} = \mathbf{X} - \mathbf{P}_{\mathbf{1}} \mathbf{X} = \begin{bmatrix} X_1 - \bar{X} \\ X_2 - \bar{X} \\ \vdots \\ X_n - \bar{X} \end{bmatrix}$$

### 4.2 Why Estimating the Sample Mean Costs Exactly One Dimension

The sum of deviations around the sample mean is identically zero:

$$\sum_{i=1}^n (X_i - \bar{X}) = \mathbf{1}^T \mathbf{e} = 0$$

This single linear constraint ($\sum e_i = 0$) confines the residual vector $\mathbf{e}$ to an **$(n - 1)$-dimensional hyperplane** orthogonal to $\mathbf{1}$.

```text
               R^n Space (Dimension n)
                   ▲
                   │     / X (Sample vector)
                   │    /
                   │   / |
                   │  /  │ e = X - X_bar * 1
                   │ /   │ (Restricted to (n - 1) dimensions)
                   │/    ▼
       ────────────┼──────────────► Span(1) (Dimension 1: X_bar * 1)
                   │
```

The sum of squared deviations is the squared Euclidean norm of $\mathbf{e}$:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \|\mathbf{e}\|^2$$

Because $\mathbf{e}$ is confined to an $(n - 1)$-dimensional subspace, $\|\mathbf{e}\|^2 / \sigma^2$ is distributed as a Chi-Square variate with **$\nu = n - 1$ degrees of freedom**, not $n$. Estimating the unknown population mean $\mu$ with the sample mean $\bar{X}$ permanently consumes one degree of freedom.

---

## 5. Mathematical Proof of Bessel's Correction and Delta Degrees of Freedom (`ddof`)

### 5.1 Expectation of the Biased Estimator $S_n^2$ (`ddof = 0`)

The naive variance estimator divides the sum of squared deviations by $n$ (`ddof = 0` in NumPy):

$$S_n^2 = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2$$

Let us derive its mathematical expectation $\mathbb{E}[S_n^2]$:

$$(X_i - \bar{X}) = (X_i - \mu) - (\bar{X} - \mu)$$

Squaring both sides:

$$(X_i - \bar{X})^2 = (X_i - \mu)^2 - 2(X_i - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^2$$

Summing over all $i \in \{1, \dots, n\}$:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n (X_i - \mu) + n(\bar{X} - \mu)^2$$

Since $\sum_{i=1}^n (X_i - \mu) = n(\bar{X} - \mu)$:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - 2n(\bar{X} - \mu)^2 + n(\bar{X} - \mu)^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

Taking expectations on both sides:

$$\mathbb{E}\left[ \sum_{i=1}^n (X_i - \bar{X})^2 \right] = \sum_{i=1}^n \mathbb{E}\left[ (X_i - \mu)^2 \right] - n \mathbb{E}\left[ (\bar{X} - \mu)^2 \right]$$

By definition:
- $\mathbb{E}[(X_i - \mu)^2] = \text{Var}(X_i) = \sigma^2$
- $\mathbb{E}[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \frac{\sigma^2}{n}$

Therefore:

$$\mathbb{E}\left[ \sum_{i=1}^n (X_i - \bar{X})^2 \right] = n\sigma^2 - n\left(\frac{\sigma^2}{n}\right) = n\sigma^2 - \sigma^2 = (n - 1)\sigma^2$$

Taking the expectation of $S_n^2$:

$$\mathbb{E}[S_n^2] = \mathbb{E}\left[ \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2 \right] = \frac{n - 1}{n} \sigma^2 = \sigma^2 - \frac{\sigma^2}{n} < \sigma^2 \quad \blacksquare$$

The naive estimator systematically **underestimates** the true variance by a factor of $\frac{n - 1}{n}$.

### 5.2 Derivation of the Unbiased Estimator $s^2$ (`ddof = 1`)

To construct an strictly unbiased estimator $s^2$ such that $\mathbb{E}[s^2] = \sigma^2$, we multiply by the Bessel correction factor $\frac{n}{n - 1}$:

$$s^2 = \frac{n}{n - 1} S_n^2 = \frac{1}{n - 1} \sum_{i=1}^n (X_i - \bar{X})^2$$

Check expectation:

$$\mathbb{E}[s^2] = \frac{n}{n - 1} \mathbb{E}[S_n^2] = \frac{n}{n - 1} \left( \frac{n - 1}{n} \sigma^2 \right) = \sigma^2 \quad \blacksquare$$

In Python (`numpy`):
- `np.std(data, ddof=0)` computes $\sqrt{S_n^2}$ (biased sample standard deviation; true population $\sigma$ when $\mu$ is known a priori).
- `np.std(data, ddof=1)` computes $\sqrt{s^2}$ (unbiased sample standard deviation with Bessel's correction). In inferential statistics and $t$-tests, **`ddof=1` must always be used**.

### 5.3 When Does `ddof > 1` Apply?

`ddof` represents the number of linear constraints or fitted parameters consumed during estimation:
- **`ddof = 0`**: Used when computing variance of an entire known population, or when the true mean $\mu$ is known analytically without estimation from data.
- **`ddof = 1`**: Standard sample variance ($s^2$); one parameter ($\bar{x}$) was estimated.
- **`ddof = 2`**: Residual variance in simple linear regression $y = \beta_0 + \beta_1 x + \epsilon$, where two parameters ($\hat{\beta}_0, \hat{\beta}_1$) are estimated from data ($s_e^2 = \frac{1}{n - 2} \sum e_i^2$).
- **`ddof = p`**: Residual variance in multiple linear regression with $p$ coefficients ($\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, $s_e^2 = \frac{1}{n - p} \|\mathbf{e}\|^2$).
- **`ddof = k`**: Within-group variance in One-Way ANOVA across $k$ groups ($s_{\text{pooled}}^2 = \frac{1}{N - k} SS_{\text{within}}$).

---

## 6. Asymptotic Proof: Convergence of Student's $t$ to the Gaussian ($\nu \to \infty$)

Student's $t$ density is given by:

$$f_T(t) = C_\nu \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}, \qquad \text{where } C_\nu = \frac{\Gamma\left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu\pi} \, \Gamma\left( \frac{\nu}{2} \right)}$$

Let us evaluate the limit as $\nu \to \infty$ in two parts.

**Part 1: The algebraic kernel**
Recall the classical definition of the exponential function:

$$\lim_{m \to \infty} \left( 1 + \frac{x}{m} \right)^m = e^x$$

Rewriting the kernel:

$$\lim_{\nu \to \infty} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}} = \lim_{\nu \to \infty} \left[ \left( 1 + \frac{t^2}{\nu} \right)^\nu \right]^{-1/2} \cdot \lim_{\nu \to \infty} \left( 1 + \frac{t^2}{\nu} \right)^{-1/2}$$

As $\nu \to \infty$:
- $\left( 1 + \frac{t^2}{\nu} \right)^\nu \to e^{t^2}$
- $\left( 1 + \frac{t^2}{\nu} \right)^{-1/2} \to 1$

Thus:

$$\lim_{\nu \to \infty} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}} = \left( e^{t^2} \right)^{-1/2} = e^{-t^2 / 2}$$

**Part 2: The normalizing constant**
By **Stirling's approximation** for the Gamma function ($\Gamma(x) \sim \sqrt{\frac{2\pi}{x}} \left(\frac{x}{e}\right)^x$ as $x \to \infty$):

$$\lim_{x \to \infty} \frac{\Gamma(x + 1/2)}{\Gamma(x)} = \sqrt{x}$$

Let $x = \nu / 2$:

$$\lim_{\nu \to \infty} \frac{\Gamma\left( \frac{\nu}{2} + \frac{1}{2} \right)}{\Gamma\left( \frac{\nu}{2} \right)} = \sqrt{\frac{\nu}{2}}$$

Substituting this into the normalizing constant $C_\nu$:

$$\lim_{\nu \to \infty} C_\nu = \lim_{\nu \to \infty} \frac{1}{\sqrt{\nu\pi}} \cdot \sqrt{\frac{\nu}{2}} = \frac{1}{\sqrt{\pi}} \cdot \frac{\sqrt{\nu}}{\sqrt{2\nu}} = \frac{1}{\sqrt{2\pi}}$$

Multiplying Part 1 and Part 2 together:

$$\lim_{\nu \to \infty} f_T(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2 / 2} = \phi(t) = \mathcal{N}(0, 1) \quad \blacksquare$$

The $t$-distribution converges pointwise and uniformly to the Standard Gaussian distribution as degrees of freedom approach infinity.

---

## 7. Quantile and Tail Notation ($t_{\nu, 1 - \alpha/2}$)

### 7.1 Demystifying the $0.975$ Quantile Subscript

In statistical texts, you will frequently see notation such as:

$$t_{14, 0.975} = 2.145$$

Why $0.975$ when testing at the $\alpha = 0.05$ significance level?

Because standard algorithmic hypothesis testing is **two-tailed**: we test against the null hypothesis $H_0: \mu_{\text{CPU}} = \mu_{\text{GPU}}$ against $H_1: \mu_{\text{CPU}} \ne \mu_{\text{GPU}}$.
The total rejection area $\alpha = 0.05$ must be divided equally between both tails:
- Lower critical tail: $\alpha / 2 = 0.025$ in the extreme negative tail ($t < -2.145$).
- Upper critical tail: $\alpha / 2 = 0.025$ in the extreme positive tail ($t > +2.145$).
- Central acceptance region: $1 - \alpha = 0.95$ of the total probability mass lies between $[-2.145, +2.145]$.

```text
                  t-Distribution Density (df = 14)
                                ▲
                                │
                              ┌─┴─┐
                             │     │
                            │       │
      Alpha/2 = 0.025      │         │      Alpha/2 = 0.025
       (Rejection)        │  0.950    │      (Rejection)
    ─────────────────────┴─────────────┴─────────────────────►
                    -2.145      0    +2.145
                      ▲                ▲
                      │                │
            Cumulative P = 0.025   Cumulative P = 0.975
```

The quantile function (Percent Point Function, `ppf`) returns the value $x$ such that $\mathbb{P}(T \le x) = p$.
Therefore, to leave $0.025$ in the upper tail, the cumulative area up to that cutoff must be:

$$p = 1 - \frac{\alpha}{2} = 1 - 0.025 = 0.975$$

Hence, $t_{14, 0.975} \equiv \text{Critical threshold for two-tailed } \alpha = 0.05 \text{ test at } \nu = 14$.

### 7.2 Numerical Comparison Table: Student's $t$ vs Standard Normal $Z$

The lower the degrees of freedom $\nu$, the heavier the tails of Student's $t$, requiring larger critical values to overcome uncertainty:

| Sample Size ($n$) | Degrees of Freedom ($\nu = n - 1$) | Two-Tailed Critical $t_{\nu, 0.975}$ | Relative Inflation over $Z_{0.975} = 1.960$ | Practical Meaning for Metaheuristic Benchmarks |
| :--- | :--- | :--- | :--- | :--- |
| $n = 3$ | $\nu = 2$ | **$4.303$** | $+119.5\%$ | Severe noise; requires massive speedup to achieve significance |
| $n = 5$ | $\nu = 4$ | **$2.776$** | $+41.6\%$ | High uncertainty; heavy penalty for small sample |
| $n = 10$ | $\nu = 9$ | **$2.262$** | $+15.4\%$ | Moderate uncertainty |
| **$n = 15$** | **$\nu = 14$** | **$2.145$** | **$+9.4\%$** | **Our Codebase Setting**: Balanced power vs compute overhead |
| $n = 30$ | $\nu = 29$ | **$2.045$** | $+4.3\%$ | Standard rule of thumb; approximates Gaussian |
| $n = 100$ | $\nu = 99$ | **$1.984$** | $+1.2\%$ | Virtually identical to normal |
| $n \to \infty$ | $\nu \to \infty$ | **$1.960$** | **$0.0\%$** | Exact Standard Gaussian limit ($Z_{0.975}$) |

---

## 8. Python Scientific Implementation

Below is a self-contained reference script demonstrating the evaluation of the Gamma function, degrees of freedom, Bessel `ddof`, and critical quantiles using `scipy.stats` and `numpy`:

```python
"""Parametric Foundations Demonstration Script."""

import numpy as np
from scipy import special, stats


def verify_parametric_properties():
  # 1. Half-integer Gamma evaluation
  gamma_half = special.gamma(0.5)
  gamma_three_halves = special.gamma(1.5)
  gamma_five_halves = special.gamma(2.5)

  print(f"Gamma(1/2) = {gamma_half:.6f} (Expected sqrt(pi) = {np.sqrt(np.pi):.6f})")
  print(f"Gamma(3/2) = {gamma_three_halves:.6f} (Expected 0.5 * sqrt(pi) = {0.5 * np.sqrt(np.pi):.6f})")
  print(f"Gamma(5/2) = {gamma_five_halves:.6f} (Expected 0.75 * sqrt(pi) = {0.75 * np.sqrt(np.pi):.6f})")

  # 2. Bessel's Correction Demonstration on n=15 sample
  np.random.seed(42)
  true_sigma = 5.0
  sample = np.random.normal(loc=100.0, scale=true_sigma, size=15)

  std_biased = np.std(sample, ddof=0)
  std_unbiased = np.std(sample, ddof=1)
  bessel_factor = np.sqrt(15.0 / 14.0)

  print(f"\nBiased Std (ddof=0)  : {std_biased:.4f}")
  print(f"Unbiased Std (ddof=1): {std_unbiased:.4f}")
  print(f"Ratio (ddof=1 / ddof=0): {std_unbiased / std_biased:.4f} (Expected sqrt(15/14) = {bessel_factor:.4f})")

  # 3. Two-Tailed t Critical Quantiles
  alpha = 0.05
  df = 14  # n = 15 runs
  t_crit_two_tailed = stats.t.ppf(1.0 - alpha / 2.0, df=df)
  z_crit_two_tailed = stats.norm.ppf(1.0 - alpha / 2.0)

  print(f"\nCritical t(df=14, p=0.975): {t_crit_two_tailed:.4f}")
  print(f"Critical Z(p=0.975)       : {z_crit_two_tailed:.4f}")
  print(f"Inflation Factor          : {t_crit_two_tailed / z_crit_two_tailed:.4f}")


if __name__ == "__main__":
  verify_parametric_properties()
```

---

## 9. Academic References

1. **Gosset, W. S. ["Student"] (1908).** The Probable Error of a Mean. *Biometrika*, 6(1), 1–25. DOI: [10.1093/biomet/6.1.1](https://doi.org/10.1093/biomet/6.1.1).
2. **Fisher, R. A. (1925).** Applications of "Student's" Distribution. *Metron*, 5, 90–104.
3. **Cochran, W. G. (1934).** The Distribution of Quadratic Forms in a Normal System, with Applications to the Analysis of Covariance. *Mathematical Proceedings of the Cambridge Philosophical Society*, 30(2), 178–191. DOI: [10.1017/S0305004100016564](https://doi.org/10.1017/S0305004100016564).
4. **Artin, E. (1964).** *The Gamma Function*. Holt, Rinehart and Winston, New York.
5. **Casella, G., & Berger, R. L. (2002).** *Statistical Inference* (2nd ed.). Duxbury Press, Pacific Grove, CA.
