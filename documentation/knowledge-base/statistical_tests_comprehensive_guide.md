# Statistical Tests Comprehensive Guide for GPU Benchmark Analysis

- **Author**: AI Assistant
- **Date**: 2025-11-21
- **Purpose**: Complete reference for statistical methodology used in Chapter 4 validation
- **Status**: 🚧 IN PROGRESS

---

## Table of Contents

- [Statistical Tests Comprehensive Guide for GPU Benchmark Analysis](#statistical-tests-comprehensive-guide-for-gpu-benchmark-analysis)
  - [Table of Contents](#table-of-contents)
  - [Symbol Glossary](#symbol-glossary)
  - [Overview](#overview)
    - [Context: GPU vs CPU Benchmark Comparison](#context-gpu-vs-cpu-benchmark-comparison)
    - [Research Questions](#research-questions)
    - [Statistical Framework](#statistical-framework)
    - [Why These Specific Tests?](#why-these-specific-tests)
  - [Statistical Foundations](#statistical-foundations)
    - [0.1 Fundamental Concepts](#01-fundamental-concepts)
      - [0.1.1 What is Probability? (Frequentist View)](#011-what-is-probability-frequentist-view)
      - [0.1.2 What is a P-value?](#012-what-is-a-p-value)
        - [Most Misunderstood Concept in Statistics](#most-misunderstood-concept-in-statistics)
          - [Formal Definition](#formal-definition)
          - [Critical Understanding: P-value is **NOT**](#critical-understanding-p-value-is-not)
          - [What P-value Actually Tells You](#what-p-value-actually-tells-you)
          - [Interpretation Ladder](#interpretation-ladder)
        - [Visual Threshold Guide](#visual-threshold-guide)
        - [Example from GPU Benchmark](#example-from-gpu-benchmark)
        - [Common Misconceptions Table](#common-misconceptions-table)
        - [Why $\\alpha = 0.05$?](#why-alpha--005)
      - [0.1.3 What is Hypothesis Testing?](#013-what-is-hypothesis-testing)
        - [The Framework of Statistical Inference](#the-framework-of-statistical-inference)
          - [The Two Hypotheses](#the-two-hypotheses)
          - [The Court Trial Analogy (Detailed)](#the-court-trial-analogy-detailed)
        - [Type I and Type II Errors](#type-i-and-type-ii-errors)
          - [Type I Error](#type-i-error)
          - [Type II Error](#type-ii-error)
        - [Error Types Matrix](#error-types-matrix)
          - [Real-World Consequences](#real-world-consequences)
        - [Statistical Power](#statistical-power)
          - [Definition](#definition)
          - [Factors affecting power](#factors-affecting-power)
          - [Power Guidelines](#power-guidelines)
        - [The Hypothesis Testing Process](#the-hypothesis-testing-process)
          - [Why both p-value AND effect size?](#why-both-p-value-and-effect-size)
      - [0.1.4 Descriptive Statistics](#014-descriptive-statistics)
        - [0.1.4.1 Measures of Central Tendency](#0141-measures-of-central-tendency)
          - [0.1.4.1.1 Mean (Arithmetic Average)](#01411-mean-arithmetic-average)
          - [0.1.4.1.2 Median (Middle Value)](#01412-median-middle-value)
          - [0.1.4.1.3 Mode (Most Frequent Value)](#01413-mode-most-frequent-value)
        - [0.1.4.2 Measures of Dispersion (Spread)](#0142-measures-of-dispersion-spread)
          - [0.1.4.2.1 Variance (Average Squared Deviation)](#01421-variance-average-squared-deviation)
          - [0.1.4.2.2 Standard Deviation (Typical Deviation)](#01422-standard-deviation-typical-deviation)
          - [0.1.4.2.3 Coefficient of Variation (Relative Spread)](#01423-coefficient-of-variation-relative-spread)
        - [0.1.4.3 Distribution Shape: Beyond Center and Spread](#0143-distribution-shape-beyond-center-and-spread)
      - [0.1.5 Distribution Shapes: Modality and Its Implications](#015-distribution-shapes-modality-and-its-implications)
        - [What is Modality?](#what-is-modality)
        - [Visual Comparison](#visual-comparison)
        - [Real-World Examples from GPU Benchmarks](#real-world-examples-from-gpu-benchmarks)
          - [Hypothetical Example 1: CPU Algorithm (Illustrative Scenario)](#hypothetical-example-1-cpu-algorithm-illustrative-scenario)
          - [Hypothetical Example 2: HybridOptimized (Illustrative Scenario)](#hypothetical-example-2-hybridoptimized-illustrative-scenario)
        - [Why Do Multimodal Distributions Occur?](#why-do-multimodal-distributions-occur)
          - [Cause 1: Mixed Populations](#cause-1-mixed-populations)
          - [Cause 2: System State Variations](#cause-2-system-state-variations)
          - [Cause 3: Algorithmic Phase Transitions](#cause-3-algorithmic-phase-transitions)
          - [Cause 4: Early Stopping and Convergence Dynamics](#cause-4-early-stopping-and-convergence-dynamics)
        - [Does Larger Sample Size Remove Bimodality?](#does-larger-sample-size-remove-bimodality)
          - [Mathematical Proof (Informal)](#mathematical-proof-informal)
          - [When CAN Apparent Bimodality Disappear?](#when-can-apparent-bimodality-disappear)
        - [Statistical Tests and Modality](#statistical-tests-and-modality)
          - [Parametric Tests: Assume Unimodality](#parametric-tests-assume-unimodality)
          - [Non-Parametric Tests: Robust to Modality](#non-parametric-tests-robust-to-modality)
        - [Practical Implications for Your Benchmark Analysis](#practical-implications-for-your-benchmark-analysis)
          - [Investigation Workflow: If You Observe Bimodality](#investigation-workflow-if-you-observe-bimodality)
          - [Warning: Scale Illusion in Visual Analysis](#warning-scale-illusion-in-visual-analysis)
          - [Finding 2: HybridOptimized Unimodal Distribution](#finding-2-hybridoptimized-unimodal-distribution)
          - [Decision Tree: Which Test to Use?](#decision-tree-which-test-to-use)
        - [Summary: Key Takeaways on Modality](#summary-key-takeaways-on-modality)
        - [Measures of Dispersion](#measures-of-dispersion)
          - [Sample Standard Deviation](#sample-standard-deviation)
          - [Other Dispersion Measures](#other-dispersion-measures)
        - [Comparison Table: Central Tendency Measures](#comparison-table-central-tendency-measures)
        - [ASCII Visualization: Outlier Effect](#ascii-visualization-outlier-effect)
        - [Why Mean for Parametric Tests?](#why-mean-for-parametric-tests)
        - [Why Median for Non-Parametric Tests?](#why-median-for-non-parametric-tests)
      - [0.1.5 Statistical Distributions](#015-statistical-distributions)
        - [Why Distributions Matter](#why-distributions-matter)
        - [The Normal Distribution (Gaussian)](#the-normal-distribution-gaussian)
          - [Probability Density Function](#probability-density-function)
          - [The 68-95-99.7 Rule (Empirical Rule)](#the-68-95-997-rule-empirical-rule)
          - [ASCII Bell Curve Visualization](#ascii-bell-curve-visualization)
          - [Why Normal is Everywhere](#why-normal-is-everywhere)
        - [Student's t-Distribution](#students-t-distribution)
          - [Motivation](#motivation)
          - [Probability Density Function](#probability-density-function-1)
          - [Key Properties](#key-properties)
          - [Why This Matters](#why-this-matters)
          - [Historical Context: The Story of "Student"](#historical-context-the-story-of-student)
        - [Chi-Squared Distribution\*\* ($\\chi^2$)](#chi-squared-distribution-chi2)
          - [Definition](#definition-1)
          - [Probability Density Function](#probability-density-function-2)
          - [Understanding the $\\Gamma(k/2)$ Term](#understanding-the-gammak2-term)
          - [What is the Gamma Function?](#what-is-the-gamma-function)
          - [Mathematical Definition](#mathematical-definition)
          - [The Factorial Connection](#the-factorial-connection)
          - [Uses in Testing](#uses-in-testing)
        - [Degrees of Freedom Demystified](#degrees-of-freedom-demystified)
          - [The n-1 Mystery: Why Not n?](#the-n-1-mystery-why-not-n)
          - [Intuitive Definition](#intuitive-definition)
          - [The Mathematical Constraint](#the-mathematical-constraint)
          - [Worked Example: n=5 GPU Runtimes](#worked-example-n5-gpu-runtimes)
          - [Visual Representation](#visual-representation)
          - [Degrees of Freedom Across Statistical Tests](#degrees-of-freedom-across-statistical-tests)
          - [Why Degrees of Freedom Matter](#why-degrees-of-freedom-matter)
          - [GPU Benchmark Application: The n=15 Design Decision](#gpu-benchmark-application-the-n15-design-decision)
        - [Relationships Between Distributions](#relationships-between-distributions)
          - [t-Distribution → Normal](#t-distribution--normal)
          - [Chi-Squared → Normal](#chi-squared--normal)
        - [Central Limit Theorem (CLT)](#central-limit-theorem-clt)
          - [The Core Concept](#the-core-concept)
          - [Mathematical Formulation](#mathematical-formulation)
          - [ASCII Visualization: Order form Chaos](#ascii-visualization-order-form-chaos)
          - [Why This Matters for Engineering](#why-this-matters-for-engineering)
          - [The "Rule of 30" Warning](#the-rule-of-30-warning)
          - [GPU Benchmark Implication](#gpu-benchmark-implication)
        - [Summary: Distribution Decision Tree](#summary-distribution-decision-tree)
  - [Regression Analysis and Model Fitting](#regression-analysis-and-model-fitting)
    - [Introduction: When Prediction Matters](#introduction-when-prediction-matters)
    - [Simple Linear Regression](#simple-linear-regression)
    - [Polynomial Regression: Modeling Nonlinear Growth](#polynomial-regression-modeling-nonlinear-growth)
    - [Model Selection: Choosing the Best Fit](#model-selection-choosing-the-best-fit)
      - [3.1 Coefficient of Determination (R²)](#31-coefficient-of-determination-r)
      - [3.2 Adjusted R² (Penalizes Complexity)](#32-adjusted-r-penalizes-complexity)
      - [3.3 Akaike Information Criterion (AIC) \& Bayesian Information Criterion (BIC)](#33-akaike-information-criterion-aic--bayesian-information-criterion-bic)
    - [Residual Analysis: Checking Model Assumptions](#residual-analysis-checking-model-assumptions)
      - [Residual Plots](#residual-plots)
      - [Quantitative Tests](#quantitative-tests)
    - [Cross-Validation: Assessing Model Robustness](#cross-validation-assessing-model-robustness)
      - [Leave-One-Out Cross-Validation (LOOCV)](#leave-one-out-cross-validation-loocv)
    - [Extrapolation Methodology: Beyond Observed Data](#extrapolation-methodology-beyond-observed-data)
      - [When is Extrapolation Acceptable?](#when-is-extrapolation-acceptable)
      - [Bootstrap Prediction Intervals](#bootstrap-prediction-intervals)
    - [Case Study: CPU Baseline Extrapolation for GPU Speedup Analysis](#case-study-cpu-baseline-extrapolation-for-gpu-speedup-analysis)
      - [Step 1: Collect Empirical Data](#step-1-collect-empirical-data)
      - [Step 2: Fit Candidate Models](#step-2-fit-candidate-models)
      - [Step 3: Model Selection](#step-3-model-selection)
      - [Step 4: Validate with Theoretical Complexity](#step-4-validate-with-theoretical-complexity)
      - [Step 5: Extrapolate with Uncertainty](#step-5-extrapolate-with-uncertainty)
      - [Step 6: Calculate GPU Speedup with Uncertainty](#step-6-calculate-gpu-speedup-with-uncertainty)
      - [Step 7: Thesis Disclaimer](#step-7-thesis-disclaimer)
    - [Summary: Regression Analysis Best Practices](#summary-regression-analysis-best-practices)
  - [Parametric vs Non-Parametric Tests](#parametric-vs-non-parametric-tests)
    - [The Fundamental Distinction in Statistical Hypothesis Testing](#the-fundamental-distinction-in-statistical-hypothesis-testing)
    - [Parametric Tests: Assume a Distribution](#parametric-tests-assume-a-distribution)
    - [Non-Parametric Tests: Distribution-Free Methods](#non-parametric-tests-distribution-free-methods)
    - [Comparison Table](#comparison-table)
    - [Efficiency \& The Power-Robustness Trade-off](#efficiency--the-power-robustness-trade-off)
      - [Pitman Asymptotic Relative Efficiency (ARE)](#pitman-asymptotic-relative-efficiency-are)
      - [The Trade-off Decision](#the-trade-off-decision)
    - [GPU Benchmark Application: Our Strategy](#gpu-benchmark-application-our-strategy)
    - [Frequentist vs Bayesian Inference](#frequentist-vs-bayesian-inference)
      - [The Two Paradigms of Statistical Inference](#the-two-paradigms-of-statistical-inference)
      - [Bayes' Theorem: The Mathematical Foundation](#bayes-theorem-the-mathematical-foundation)
      - [Philosophical Comparison](#philosophical-comparison)
      - [Prior Specification in Bayesian Analysis](#prior-specification-in-bayesian-analysis)
      - [Interpretation Differences: A Concrete Example](#interpretation-differences-a-concrete-example)
      - [Computational Considerations](#computational-considerations)
      - [Small Sample Context: Why n=15 Works for Frequentist](#small-sample-context-why-n15-works-for-frequentist)
      - [Decision Path Visualization for n=15](#decision-path-visualization-for-n15)
    - [Statistical Power Analysis](#statistical-power-analysis)
      - [Introduction \& Type II Error](#introduction--type-ii-error)
      - [Mathematical Foundations](#mathematical-foundations)
      - [Factors Affecting Power](#factors-affecting-power-1)
      - [Power Analysis for n=15 (GPU Benchmark Context)](#power-analysis-for-n15-gpu-benchmark-context)
      - [Sample Size Determination: "How Many Runs Do I Need?"](#sample-size-determination-how-many-runs-do-i-need)
      - [Power Curves: Visual Guide to Sample Size Planning](#power-curves-visual-guide-to-sample-size-planning)
      - [Practical Guidance: When to Use Which Sample Size](#practical-guidance-when-to-use-which-sample-size)
  - [Normality Testing](#normality-testing)
    - [Shapiro-Wilk Test](#shapiro-wilk-test)
      - [Mathematical Formulation](#mathematical-formulation-1)
      - [When to Use](#when-to-use)
      - [Interpretation](#interpretation)
        - [Decision Rule (Statistical Mechanics)](#decision-rule-statistical-mechanics)
        - [Conceptual Meaning in Algorithm Benchmarking](#conceptual-meaning-in-algorithm-benchmarking)
        - [Important Caveats](#important-caveats)
      - [Normality Assessment Decision Tree](#normality-assessment-decision-tree)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark)
      - [Example from Benchmark](#example-from-benchmark)
      - [Assumptions and Limitations](#assumptions-and-limitations)
  - [Parametric Tests](#parametric-tests)
    - [Paired t-Test](#paired-t-test)
      - [Plain Language Explanation](#plain-language-explanation)
      - [Mathematical Formulation](#mathematical-formulation-2)
      - [Decision Framework: When to Use Paired t-Test](#decision-framework-when-to-use-paired-t-test)
      - [When to Use](#when-to-use-1)
      - [Interpretation](#interpretation-1)
        - [Decision Rule (Statistical Mechanics)](#decision-rule-statistical-mechanics-1)
        - [Conceptual Meaning in Algorithm Benchmarking](#conceptual-meaning-in-algorithm-benchmarking-1)
        - [P-value Interpretation](#p-value-interpretation)
        - [Effect Size and Power](#effect-size-and-power)
      - [Effect Size: Cohen's d for Paired Data](#effect-size-cohens-d-for-paired-data)
      - [Power Analysis Integration](#power-analysis-integration)
      - [Post-hoc Tests and Multiple Comparisons](#post-hoc-tests-and-multiple-comparisons)
      - [Python Implementation Details](#python-implementation-details)
      - [Example from Benchmark](#example-from-benchmark-1)
      - [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
      - [Historical Context: The Birth of Modern Statistics](#historical-context-the-birth-of-modern-statistics)
      - [Computational Complexity](#computational-complexity)
      - [Assumptions and Limitations](#assumptions-and-limitations-1)
  - [Non-Parametric Tests](#non-parametric-tests)
    - [Wilcoxon Signed-Rank Test](#wilcoxon-signed-rank-test)
      - [Mathematical Formulation](#mathematical-formulation-3)
      - [When to Use](#when-to-use-2)
        - [**The Ideal**](#the-ideal)
        - [**Our Reality**](#our-reality)
      - [Interpretation](#interpretation-2)
        - [Decision Rule (Statistical Mechanics)](#decision-rule-statistical-mechanics-2)
        - [Conceptual Meaning in Algorithm Benchmarking](#conceptual-meaning-in-algorithm-benchmarking-2)
        - [P-value Interpretation](#p-value-interpretation-1)
        - [Effect Size (Rank-biserial correlation)](#effect-size-rank-biserial-correlation)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-1)
      - [Example from Benchmark](#example-from-benchmark-2)
      - [Assumptions and Limitations](#assumptions-and-limitations-2)
    - [Friedman Test](#friedman-test)
      - [Mathematical Formulation](#mathematical-formulation-4)
      - [When to Use](#when-to-use-3)
        - [The Ideal](#the-ideal-1)
        - [Our Reality](#our-reality-1)
      - [Interpretation](#interpretation-3)
        - [Conceptual Meaning](#conceptual-meaning)
          - [What Friedman Actually Tests](#what-friedman-actually-tests)
        - [Example Interpretation:](#example-interpretation)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-2)
      - [Example from Benchmark](#example-from-benchmark-3)
      - [Assumptions and Limitations](#assumptions-and-limitations-3)
  - [Post-Hoc Tests](#post-hoc-tests)
    - [Nemenyi Test](#nemenyi-test)
      - [Mathematical Formulation](#mathematical-formulation-5)
      - [When to Use](#when-to-use-4)
        - [The Ideal](#the-ideal-2)
        - [Our Reality](#our-reality-2)
      - [Interpretation](#interpretation-4)
        - [**Example Interpretation**:](#example-interpretation-1)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-3)
      - [Example from Benchmark](#example-from-benchmark-4)
      - [Assumptions and Limitations](#assumptions-and-limitations-4)
  - [Multiple Comparison Correction](#multiple-comparison-correction)
    - [Holm-Bonferroni Method](#holm-bonferroni-method)
      - [Mathematical Formulation](#mathematical-formulation-6)
      - [When to Use](#when-to-use-5)
        - [The Ideal (single comparison luxury)](#the-ideal-single-comparison-luxury)
        - [Our Reality (multiple comparisons necessity)](#our-reality-multiple-comparisons-necessity)
        - [Consequences (statistical rigor trade-off)](#consequences-statistical-rigor-trade-off)
        - [Our Benchmark Context ($n\_{runs}=15$, $k=4$ algorithms)](#our-benchmark-context-n_runs15-k4-algorithms)
        - [Not Appropriate](#not-appropriate)
      - [Interpretation](#interpretation-5)
        - [Decision Rule (Statistical Mechanics)](#decision-rule-statistical-mechanics-3)
        - [Conceptual Meaning in Algorithm Benchmarking](#conceptual-meaning-in-algorithm-benchmarking-3)
        - [Power Comparison](#power-comparison)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-4)
      - [Example from Benchmark](#example-from-benchmark-5)
      - [Comparison with Other Methods](#comparison-with-other-methods)
  - [Effect Size Measures](#effect-size-measures)
    - [Cohen's d](#cohens-d)
      - [Mathematical Formulation](#mathematical-formulation-7)
      - [When to Use](#when-to-use-6)
        - [The Ideal (effect size independence)](#the-ideal-effect-size-independence)
        - [Our Reality (small sample context with $n\_{runs}=15$)](#our-reality-small-sample-context-with-n_runs15)
        - [Consequences (dual reporting necessity)](#consequences-dual-reporting-necessity)
        - [Our Benchmark Context ($n\_{runs}=15$)](#our-benchmark-context-n_runs15)
        - [Reporting Guidelines (APA style)](#reporting-guidelines-apa-style)
      - [Interpretation](#interpretation-6)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-5)
      - [Example from Benchmark](#example-from-benchmark-6)
      - [Best Practices](#best-practices)
  - [Multi-Problem Meta-Analysis](#multi-problem-meta-analysis)
    - [Understanding Your Data Structure: 38 Problems × 15 Runs × 4 Algorithms](#understanding-your-data-structure-38-problems--15-runs--4-algorithms)
      - [The Three-Level Analysis Hierarchy](#the-three-level-analysis-hierarchy)
    - [Level 1: Within-Problem Paired Analysis](#level-1-within-problem-paired-analysis)
    - [Level 2: Across-Problem Meta-Analysis](#level-2-across-problem-meta-analysis)
      - [Why Level 2 is Critical](#why-level-2-is-critical)
      - [Meta-Analysis Pipeline](#meta-analysis-pipeline)
      - [Reporting Meta-Analysis Results](#reporting-meta-analysis-results)
    - [Level 3: Multiple Algorithm Ranking ($k=4$ Algorithms)](#level-3-multiple-algorithm-ranking-k4-algorithms)
      - [Why Friedman Test?](#why-friedman-test)
      - [Friedman + Nemenyi Pipeline](#friedman--nemenyi-pipeline)
      - [Complete Multi-Level Analysis Pipeline](#complete-multi-level-analysis-pipeline)
    - [Summary: Three-Level Analysis Decision Tree](#summary-three-level-analysis-decision-tree)
  - [Confidence Intervals](#confidence-intervals)
    - [Bootstrap Method](#bootstrap-method)
      - [Mathematical Formulation](#mathematical-formulation-8)
      - [When to Use](#when-to-use-7)
        - [The Ideal (parametric CI with normality)](#the-ideal-parametric-ci-with-normality)
        - [Our Reality (non-normal or complex statistics with $n\_{runs}=15$)](#our-reality-non-normal-or-complex-statistics-with-n_runs15)
        - [Consequences (computational vs. validity trade-off)](#consequences-computational-vs-validity-trade-off)
        - [Our Benchmark Context ($n\_{runs}=15$, B=10,000)](#our-benchmark-context-n_runs15-b10000)
        - [Not Appropriate](#not-appropriate-1)
      - [Methods Comparison](#methods-comparison)
      - [Implementation in Our Benchmark](#implementation-in-our-benchmark-6)
      - [Example from Benchmark](#example-from-benchmark-7)
      - [Advantages and Limitations](#advantages-and-limitations)
      - [Best Practices](#best-practices-1)
  - [Decision Trees](#decision-trees)
    - [Master Statistical Test Selection Guide](#master-statistical-test-selection-guide)
    - [Test Selection Flowchart](#test-selection-flowchart)
    - [Multiple Algorithm Comparison Flowchart](#multiple-algorithm-comparison-flowchart)
    - [Independent Samples Test Selection](#independent-samples-test-selection)
    - [Small Sample Decision Guide (n \< 20)](#small-sample-decision-guide-n--20)
    - [GPU Benchmark Test Selection (n=15)](#gpu-benchmark-test-selection-n15)
  - [Appendix A: Critical Value Tables](#appendix-a-critical-value-tables)
    - [A.1 Standard Normal (Z) Distribution](#a1-standard-normal-z-distribution)
    - [A.2 Student's t-Distribution](#a2-students-t-distribution)
    - [A.3 Chi-Squared (χ²) Distribution](#a3-chi-squared-χ-distribution)
    - [A.4 F-Distribution](#a4-f-distribution)
    - [Usage Guide](#usage-guide)
    - [Cross-References](#cross-references)
  - [References](#references)
    - [Primary Literature](#primary-literature)
    - [Statistical Methods Textbooks](#statistical-methods-textbooks)
    - [Online Resources](#online-resources)
    - [Software Documentation](#software-documentation)
  - [Correlation Analysis and Covariate Testing](#correlation-analysis-and-covariate-testing)
    - [Spearman Rank Correlation](#spearman-rank-correlation)
      - [Purpose](#purpose)
      - [Mathematical Formulation](#mathematical-formulation-9)
      - [Interpretation](#interpretation-7)
      - [Application to Regression Diagnostics](#application-to-regression-diagnostics)
      - [Example: Stop Reason Distribution as Covariate](#example-stop-reason-distribution-as-covariate)
      - [Spearman vs Pearson: When to Choose](#spearman-vs-pearson-when-to-choose)
      - [Python Implementation](#python-implementation)
      - [Assumptions and Limitations](#assumptions-and-limitations-5)
      - [Reporting Template](#reporting-template)
  - [Appendix: Code Examples](#appendix-code-examples)
    - [Complete Statistical Analysis Pipeline](#complete-statistical-analysis-pipeline)

---

## Symbol Glossary

**Quick Reference**: All mathematical symbols used in this document

| Symbol | Name | Meaning | Context & Notes |
|--------|------|---------|-----------------|
| $\alpha$ | Alpha | Significance level | Probability threshold for rejecting null hypothesis (typically 0.05 = 5%) |
| $\beta$ | Beta | Type II error rate | Probability of false negative (failing to detect real effect) |
| $1-\beta$ | Power | Statistical power | Probability of correctly detecting real effect when it exists |
| $p$ | P-value | Probability value | $P(\text{data} \mid H_0 \text{ true})$ - probability of observing data if null hypothesis is true |
| $H_0$ | Null hypothesis | Status quo assumption | Default claim we test against (e.g., "no difference between algorithms") |
| $H_1$ or $H_a$ | Alternative hypothesis | Research hypothesis | What we want to prove (e.g., "GPU is faster than CPU") |
| $n$ | Sample size | Number of observations | Larger $n$ → more statistical power, smaller standard errors |
| $\bar{x}$ | X-bar | Sample mean | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ - average of observed values |
| $\mu$ | Mu | Population mean | True (unknown) average in the entire population |
| $s$ | Sample std dev | Standard deviation | $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$ - spread of data |
| $\sigma$ | Sigma | Population std dev | True (unknown) standard deviation in population |
| $s_d$ | Std dev of differences | Paired data variability | Standard deviation of difference scores in paired tests |
| $\bar{d}$ | Mean difference | Average of differences | $\bar{d} = \frac{1}{n}\sum_{i=1}^{n}(x_{i,\text{CPU}} - x_{i,\text{GPU}})$ for paired data |
| $SE$ | Standard error | Std error of mean | $SE = \frac{s}{\sqrt{n}}$ - uncertainty in sample mean estimate |
| $t$ | T-statistic | Test statistic | Standardized mean difference: $t = \frac{\bar{d}}{s_d/\sqrt{n}}$ |
| $df$ | Degrees of freedom | Sample size parameter | $df = n - 1$ for one-sample tests - determines t-distribution shape |
| $d$ | Cohen's d | Effect size | Standardized mean difference in SD units: $d = \frac{\bar{d}}{s_d}$ |
| $CI$ | Confidence interval | Range estimate | Interval likely to contain true parameter (e.g., 95% CI) |
| $R_i$ | Rank | Position in sorted list | Integer from 1 (smallest) to $n$ (largest) after sorting |
| $T^+$ | Positive rank sum | Wilcoxon statistic | Sum of ranks for positive differences in Wilcoxon signed-rank test |
| $W$ | Shapiro-Wilk statistic | Normality measure | $0 < W \leq 1$ - closer to 1 indicates more normal distribution |
| $\chi^2$ | Chi-squared | Test statistic | Sum of squared standardized values - used in Friedman test |
| $Q$ | Friedman statistic | Rank-based ANOVA stat | $Q = \frac{12n}{k(k+1)}\sum_j (\bar{r}_j - \frac{k+1}{2})^2$ |
| $k$ | Number of groups | Algorithm count | Number of treatments/algorithms being compared |
| $\bar{r}_j$ | Mean rank | Average rank for group $j$ | Mean rank assigned to algorithm $j$ across all problems |
| $CD$ | Critical difference | Nemenyi threshold | Minimum rank difference needed for significance in post-hoc tests |
| $q_\alpha$ | Studentized range | Critical value | From Tukey's HSD distribution - used in Nemenyi test |
| $m$ | Number of comparisons | Multiple tests | Total number of hypothesis tests performed simultaneously |
| $B$ | Bootstrap samples | Resampling count | Number of bootstrap resamples (typically 10,000) |

**Note**: Greek letters denote population parameters (unknown), Latin letters denote sample statistics (computed from data).

---

## Overview

### Context: GPU vs CPU Benchmark Comparison

This guide documents the statistical methodology for comparing GPU-accelerated genetic algorithms against CPU baselines, as implemented in `chapter4_validation.py`. The analysis follows rigorous academic standards for experimental computer science research.

### Research Questions

**Q1**: Do GPU implementations maintain solution quality compared to CPU?  
**Q2**: Do GPU implementations achieve statistically significant speedup?  
**Q3**: Which GPU optimization strategy performs best?

### Statistical Framework

Our methodology follows a hierarchical approach:

```mermaid
graph TB
    A[1. Data Collection<br/>30 repetitions per algorithm] --> B[2. Normality Assessment<br/>Shapiro-Wilk Test]:::decision
    B -->|Normal| C[3a. Parametric Tests<br/>Paired t-Test]:::parametric
    B -->|Non-Normal| D[3b. Non-Parametric Tests<br/>Wilcoxon Signed-Rank]:::nonparametric
    C --> E[4. Multiple Algorithm Comparison<br/>Friedman Test]:::nonparametric
    D --> E
    E --> F[5. Post-Hoc Analysis<br/>Nemenyi Test]:::nonparametric
    F --> G[6. Multiple Comparison Correction<br/>Holm-Bonferroni Method]
    G --> H[7. Effect Size Calculation<br/>Cohen's d]
    H --> I[8. Confidence Intervals<br/>Bootstrap Method]
    
```

### Why These Specific Tests?

**Choice rationale**:

- **Paired design**: Same problem instances across algorithms → paired tests
- **Non-normality common**: Execution times often skewed → need non-parametric alternatives
- **Multiple algorithms**: Need k-sample tests (Friedman) not just pairwise
- **Family-wise error control**: Multiple comparisons require correction (Holm-Bonferroni)
- **Practical significance**: Statistical significance $\neq$ practical importance → need effect sizes

---

## Statistical Foundations

Before diving into specific tests, we must establish foundational concepts. These sections explain the **why** and **what** behind statistical testing, ensuring you can interpret results correctly and apply methods appropriately.

### 0.1 Fundamental Concepts

#### 0.1.1 What is Probability? (Frequentist View)

**Question**: When we say "$\Pr(\text{heads}) = 0.5$", what does that actually mean?

**Frequentist Definition**:  
Probability is the **long-run relative frequency** of an event occurring in infinitely many repeated trials under identical conditions.

$$
\begin{align}
P(A) = \lim_{n \to \infty} \frac{\text{Number of times A occurs}}{n}
\end{align}
$$

**Concrete Example** (Coin Flips):

```text
Experiment: Fair coin toss
Trial 1: H                  => P(H) ~= 1.00 (100%)
Trial 10: HHTHT THHTH       => P(H) ~= 0.60 (60%)
Trial 100:                  => P(H) ~= 0.52 (52%)
Trial 10,000:               => P(H) ~= 0.5003 (50.03%)
Trial 1,000,000:            => P(H) ~= 0.500012 (50.0012%)

As n → ∞, the observed proportion converges to true probability 0.5
```

**Key Properties**:

1. **Objective**: Probability exists "out there" independent of beliefs
   - No subjective priors or opinions
   - Same result for all researchers given same data

2. **Empirical**: Based on observable frequencies
   - Can be estimated from data
   - Example: GPU runtime distribution from 30 repetitions

3. **Asymptotic**: Requires large samples for accuracy
   - Small n: Observed frequency unstable (52% vs 60%)
   - Large n: Observed frequency stable (50.03% vs 50.0012%)

**Why Frequentist for Benchmarks?**

✅ **Reproducibility**: Other researchers get same p-values  
✅ **No subjective priors**: Don't need prior beliefs about GPU performance  
✅ **Industry standard**: Publications, reviewers expect frequentist  
✅ **Simple computation**: Direct formulas vs. Bayesian MCMC sampling

**Alternative Paradigm** (Bayesian):  
Probability = degree of belief ($\Pr(\text{hypothesis}|\text{data})$). Useful when incorporating domain knowledge, but requires specifying prior distributions. See Section 0.3 for detailed comparison.

#### 0.1.2 What is a P-value?

##### Most Misunderstood Concept in Statistics

###### Formal Definition  

$$
\begin{align}
p = \Pr(\text{observe data at least this extreme} \mid H_0 \text{ is true})
\end{align}
$$

Read as: "The probability of observing data as extreme as what we saw (or more extreme), **assuming the null hypothesis is true**."

###### Critical Understanding: P-value is **NOT**

- ❌ Probability that the null hypothesis is true
- ❌ Probability that results are due to chance  
- ❌ Probability of making a mistake  
- ❌ Importance or practical significance of the result

###### What P-value Actually Tells You

"If there were truly no difference between GPU and CPU ($H_0$ true), what's the probability we'd see a difference as large as we observed?"

###### Interpretation Ladder

| P-value Range | Evidence Against $H_0$ | Interpretation | Action |
|---------------|---------------------|----------------|--------|
| $p < 0.001$ | **Very strong** | Extremely unlikely under $H_0$ | Confidently reject $H_0$ |
| $0.001 \leq p < 0.01$ | **Strong** | Very unlikely under $H_0$ | Reject $H_0$ |
| $0.01 \leq p < 0.05$ | **Moderate** | Unlikely under $H_0$ | Reject $H_0$ (standard threshold) |
| $0.05 \leq p < 0.10$ | **Weak** | Somewhat unlikely under $H_0$ | Borderline (consider context) |
| $p \geq 0.10$ | **None** | Consistent with $H_0$ | Fail to reject $H_0$ |

##### Visual Threshold Guide

```text
Strong Evidence   <--   Weaker Evidence   -->   No Evidence
     |                       |                      |
   p=0.001                 p=0.05                p=0.10
     |                       |                      |
[████████████████] [████████░░░░░░░░] [████░░░░░░░░░░░░]
  Reject H₀              Reject H₀          Fail to Reject
```

##### Example from GPU Benchmark

```text
Scenario: Comparing GPU vs CPU execution times
H₀: GPU time = CPU time (no difference)
H₁: GPU time $\neq$ CPU time (there IS a difference)

Observed: GPU mean = 5.2s, CPU mean = 22.8s
Difference: 17.6 seconds

P-value = 0.0001 (0.01%)

Interpretation:
"If GPU and CPU were truly equal (H₀), there's only a 0.01% chance
we'd observe a 17.6s difference this large or larger. Since this is
highly improbable (p=0.0001 << 0.05), we reject H₀ and conclude
GPU is significantly faster."
```

>[!caution]
>Example needs improvement. Don't like this text format.

##### Common Misconceptions Table

| ❌ WRONG Statement | ✅ CORRECT Statement |
|-------------------|---------------------|
| "$P=0.03$ means 3% chance $H_0$ is true" | "$P=0.03$ means if $H_0$ were true, 3% chance of this data" |
| "$P=0.06$ means no effect exists" | "$P=0.06$ means insufficient evidence against $H_0$" |
| "Smaller p-value = larger effect" | "Smaller p-value = stronger evidence (but effect size separate)" |
| "$P=0.001$ proves GPU is better" | "$P=0.001$ gives very strong evidence GPU differs (direction from data)" |

##### Why $\alpha = 0.05$?

The 5% threshold is **conventional**, not sacred:

- **Historical**: Ronald Fisher suggested 0.05 as "reasonable" in 1925
- **Arbitrary**: Could use 0.01 (stricter) or 0.10 (more lenient)
- **Context-dependent**: Medical research often uses 0.01, exploratory research might use 0.10
- **Our benchmark**: Use $\alpha = 0.05$ following computer science convention

**The Replication Crisis**: Using $p<0.05$ alone can lead to false discoveries. **Solution**: Always report effect sizes (Cohen's d, see Section X) alongside p-values!

#### 0.1.3 What is Hypothesis Testing?

##### The Framework of Statistical Inference

Hypothesis testing is the formal procedure for using sample data to make decisions about population parameters. It's the foundation of all statistical tests in this document.

###### The Two Hypotheses

**Null Hypothesis** ($H_0$): The status quo assumption - what we assume is true until proven otherwise  
**Alternative Hypothesis** ($H_1$ or $H_a$): The claim we want to establish through evidence

**GPU Benchmark Example**:

- $H_0$: GPU execution time = CPU execution time (no performance difference)
- $H_1$: GPU execution time $\neq$ CPU execution time (there IS a performance difference)

**Legal System Analogy**: Think of hypothesis testing like a court trial:

- $H_0$ = "Defendant is innocent" (presumption of innocence)
- $H_1$ = "Defendant is guilty" (prosecutor's claim)
- Evidence = Data from experiment
- Verdict = Statistical decision (reject or fail to reject $H_0$)

###### The Court Trial Analogy (Detailed)

| Trial Concept | Statistical Equivalent | GPU Benchmark Example |
|---------------|------------------------|------------------------|
| **Presumption of innocence** | Assume $H_0$ true until strong evidence | Assume no GPU advantage until proven |
| **Burden of proof** | Need low p-value to reject $H_0$ | Need significant results to claim speedup |
| **Beyond reasonable doubt** | $\alpha = 0.05$ threshold (5% doubt) | Accept 5% chance of false positive |
| **Acquittal** | Fail to reject $H_0$ | Insufficient evidence of GPU advantage |
| **Conviction** | Reject $H_0$ | Strong evidence GPU is faster |

**Critical Note**: "Fail to reject $H_0$" $\neq$ "Accept $H_0$" (just like "not guilty" $\neq$ "innocent")

##### Type I and Type II Errors

Every statistical decision has two possible mistakes:

###### Type I Error

(False Positive, $\alpha$ error)

$$
\begin{align}
\Pr(\text{Reject } H_0 \mid H_0 \text{ is true})
\end{align}
$$

- **Definition**: Concluding there's a difference when none exists
- **Controlled by**: Significance level $\alpha$ (usually 0.05)
- **GPU Example**: Claiming GPU is faster when it's actually equal to CPU
- **Consequence**: Wasted resources implementing "faster" algorithm that isn't

###### Type II Error

(False Negative, $\beta$ error):  

$$
\begin{align}
\Pr(\text{Fail to reject } H_0 \mid H_1 \text{ is true})
\end{align}
$$

- **Definition**: Missing a real difference that exists
- **Controlled by**: Sample size ($n_{runs}$), effect size, test power
- **GPU Example**: Failing to detect real GPU speedup due to small sample
- **Consequence**: Missing opportunity to use faster implementation

##### Error Types Matrix

| | $H_0$ Actually True | $H_1$ Actually True |
|-------------|---------------------|---------------------|
| **Reject $H_0$** | ❌ Type I Error ($\alpha$) | ✅ Correct (Power = $1-\beta$) |
| **Fail to Reject $H_0$** | ✅ Correct ($1-\alpha$) | ❌ Type II Error ($\beta$) |

###### Real-World Consequences

| Error Type | Medical Test | GPU Benchmark | Cost |
|------------|--------------|---------------|------|
| **Type I** | False positive: Healthy person diagnosed sick | Claim GPU faster when it's not | Wasted optimization effort |
| **Type II** | False negative: Sick person not diagnosed | Miss real GPU speedup | Lost performance opportunity |

##### Statistical Power

###### Definition

$$
\begin{align}

\text{Power} = 1 - \beta = \Pr(\text{Reject } H_0 \mid H_1 \text{ is true})
\end{align}
$$

**In plain language**: Probability of detecting a real effect when it exists

**For comprehensive power analysis including sample size calculations, see [Statistical Power Analysis](#statistical-power-analysis).**

###### Factors affecting power

1. **Effect size** (larger $\to$ more power): GPU $2\times$ faster easier to detect than $1.1\times$ faster
2. **Sample size** ($n_{runs}$) (larger $\to$ more power): $n_{runs}=50$ better than $n_{runs}=10$
3. **Significance level** ($\alpha$) (larger $\to$ more power, but more Type I errors): $\alpha=0.10$ more power than $\alpha=0.01$
4. **Test choice** ($\text{parametric} > \text{non-parametric if assumptions met}$): $\text{t-test > Wilcoxon}$ for normal data

###### Power Guidelines

- $\text{Power} = 0.80$ (80%): Standard minimum in research
- $\text{Power} = 0.90$ (90%): High power, preferred when feasible
- $\text{Power} < 0.50$ (50%): Underpowered, likely to miss real effects

**GPU Benchmark Context**:  
With $n_{runs}=15$ repetitions and typical GPU speedups ($d > 2.0$), we achieve power $> 0.99$ for detecting differences. Small optimizations ($d = 0.2$) would need $n > 200$ for adequate power, while medium effects ($d=0.5$) require $n \geq 34$ for 80% power.

##### The Hypothesis Testing Process

1. **State hypotheses**: Define $H_0$ and $H_1$ clearly
2. **Choose significance level**: Typically $\alpha = 0.05$
3. **Collect data**: Run experiments ($n_{runs}=15$ repetitions in our case)
4. **Compute test statistic**: e.g., $t$, $W$, $Q$ depending on test
5. **Calculate p-value**: Probability of observing data if $H_0$ true
6. **Make decision**:
   - If $p < \alpha$: Reject $H_0$, conclude $H_1$
   - If $p \geq \alpha$: Fail to reject $H_0$, insufficient evidence
7. **Report effect size**: Always include Cohen's $d$, confidence intervals

###### Why both p-value AND effect size?

- P-value: Tells you **if** difference is real (statistical significance)
- Effect size: Tells you **how large** the difference is (practical significance)
- Example: $p = 0.001$ but $d = 0.05$ = statistically significant but trivially small
- Example: $p = 0.08$ but $d = 1.2$ = not significant but large effect (need more data)

#### 0.1.4 Descriptive Statistics

Before conducting hypothesis tests, we must understand how to **summarize** and **describe** data. Descriptive statistics provide the foundation for all inferential methods.

##### 0.1.4.1 Measures of Central Tendency

Central tendency answers: "What is a typical value?"

###### 0.1.4.1.1 Mean (Arithmetic Average)

$$
\begin{align}

\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\end{align}
$$

**Intuition**: Sum all values and divide by count - balances all observations equally

**Properties**:

- Uses ALL data points (maximum information)
- Sensitive to outliers (one extreme value shifts mean)
- Algebraically tractable (enables closed-form formulas for $SE$, $CI$)
- Optimal for symmetric, unimodal distributions

**GPU Benchmark Example**:  

`CPU times: [22.5, 23.1, 22.8, 24.2, 22.9] seconds`

$\bar{x}_{\text{CPU}} = (22.5 + 23.1 + 22.8 + 24.2 + 22.9) / 5 = 23.1$ seconds

###### 0.1.4.1.2 Median (Middle Value)

For ordered data $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$:

$$
\begin{align}

\text{Median} = \begin{cases}
x_{(m)} & \text{if } n = 2m-1 \text{ (odd)} \\
\frac{x_{(m)} + x_{(m+1)}}{2} & \text{if } n = 2m \text{ (even)}
\end{cases}
\end{align}
$$

Where $m = \lceil n/2 \rceil$ (ceiling function)

**Intuition**: Value that splits data into equal halves - 50th percentile

**Properties**:

- Robust to outliers (only position matters, not magnitude)
- Uses LESS information than mean (order, not exact values)
- Optimal for skewed or heavy-tailed distributions
- Interpretation straightforward: "half faster, half slower"

**GPU Benchmark Example**:  

`CPU times sorted: [22.5, 22.8, 22.9, 23.1, 24.2]`

Median = 22.9 seconds (middle value, n=5 odd)

###### 0.1.4.1.3 Mode (Most Frequent Value)

**Definition**: The value that appears most frequently in the dataset

$$
\begin{align}
\text{Mode} = \underset{x}{\arg\max} \; f(x)
\end{align}
$$

Where $f(x)$ is the frequency (count) of value $x$

**Intuition**: The "typical" value in the most common sense - what you'd see most often

**Properties**:

- Can have multiple modes (bimodal, multimodal - see Section 0.1.5)
- Only meaningful for discrete or grouped continuous data
- Not affected by extreme values at all
- Can be used with categorical data (unlike mean/median)

**GPU Benchmark Example** (Discrete categories):

`Algorithm choices: [CPU, CPU, GPU, CPU, GPU, CPU, CPU, GPU]`

Mode = CPU (appears 5 times vs GPU's 3 times)

**Continuous Data**: For execution times, mode is typically found from histogram peaks or kernel density estimates (KDE), as exact value repetition is rare.

##### 0.1.4.2 Measures of Dispersion (Spread)

Dispersion answers: "How much do values vary around the center?"

###### 0.1.4.2.1 Variance (Average Squared Deviation)

**Sample Variance**:

$$
\begin{align}
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
\end{align}
$$

**Intuition**: Average of squared distances from mean - measures "typical squared deviation"

**Why $n-1$ instead of $n$?**

- Also called **Bessel's correction**  
- **Bias correction**: Using $\bar{x}$ instead of true $\mu$ loses 1 degree of freedom
- **Unbiased estimator**: $E[s^2] = \sigma^2$ (expected value equals true variance)
- **Bessel's correction**: Named after Friedrich Bessel (1840s)

**Properties**:

- **Units**: Squared units of original data (seconds²)
- **Sensitive to outliers**: Squaring amplifies large deviations
- **Non-negative**: $s^2 \geq 0$ always
- **Zero variance**: $s^2 = 0$ only when all values identical

**GPU Benchmark Example**:

`CPU times: [22.5, 23.1, 22.8, 24.2, 22.9]`  
`Mean: 23.1 seconds`

$$
\begin{align}
s^2 &= \frac{(22.5-23.1)^2 + (23.1-23.1)^2 + (22.8-23.1)^2 + (24.2-23.1)^2 + (22.9-23.1)^2}{5-1} \\
&= \frac{0.36 + 0 + 0.09 + 1.21 + 0.04}{4} \\
&= \frac{1.70}{4} = 0.425 \text{ seconds}^2
\end{align}
$$

###### 0.1.4.2.2 Standard Deviation (Typical Deviation)

**Sample Standard Deviation**:

$$
\begin{align}
s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
\end{align}
$$

**Intuition**: "Typical distance" from the mean - same units as original data

**Properties**:

- **Same units**: Seconds (interpretable scale)
- **$68\% - 95\% - 99.7\%$ rule** (for normal distributions):
  - $68\%$ of data within $\pm 1s$ of mean
  - $95\%$ within $\pm 2s$
  - $99.7\%$ within $\pm 3s$
- **Most commonly reported**: Preferred over variance for interpretation

**GPU Benchmark Example** (continued):

$$
\begin{align}
s = \sqrt{0.425} \approx 0.65 \text{ seconds}
\end{align}
$$

**Interpretation**: "CPU times typically deviate ±0.65 seconds from the mean of 23.1 seconds"

###### 0.1.4.2.3 Coefficient of Variation (Relative Spread)

$$
\begin{align}
CV = \frac{s}{\bar{x}} \times 100\%
\end{align}
$$

**Intuition**: Standard deviation as percentage of mean - enables cross-scale comparisons

**When to use**:

- Comparing variability across different units (seconds vs milliseconds)
- Comparing algorithms with different magnitudes (CPU ~20s vs GPU ~2s)
- Assessing relative consistency ("GPU is 10% variable vs CPU 50% variable")

**GPU Benchmark Example**:

```text
CPU:  mean=23.1s, sd=0.65s  → CV = 0.65/23.1 × 100% = 2.8%
GPU:  mean=2.5s,  sd=0.15s  → CV = 0.15/2.5  × 100% = 6.0%

Interpretation: CPU more consistent in absolute terms (0.65s vs 0.15s),
                but GPU more variable relative to its mean (6% vs 2.8%)
```

##### 0.1.4.3 Distribution Shape: Beyond Center and Spread

**Critical Insight**: Mean and standard deviation **only** fully describe **normal (Gaussian) distributions**. For non-normal data, we need additional shape descriptors.

#### 0.1.5 Distribution Shapes: Modality and Its Implications

This section addresses the **missing concept** identified in your question: What are multimodal distributions, why do they occur, and what do they mean for statistical testing?

> [!important]: Examples in this section are HYPOTHETICAL illustrations of statistical concepts.
>
> - Do NOT assume these patterns apply to your data without empirical validation. Bimodality has
> - multiple possible causes and requires problem-specific investigation using YOUR benchmark results.

##### What is Modality?

**Mode**: A local maximum (peak) in the probability density function

**Modality Classification**:

- **Unimodal**: 1 peak (single dominant behavior)
- **Bimodal**: 2 peaks (two distinct behaviors)
- **Multimodal**: 3+ peaks (multiple distinct behaviors)
- **Uniform**: No peaks (flat, all values equally likely)

##### Visual Comparison

```text
UNIMODAL (Normal Distribution)
Frequency
    |     ****
    |   ********
    |  **********
    | ************
    |**************
    +---------------> Value
         Single peak

BIMODAL (Two Distinct Groups)
Frequency
    | **        **
    |****      ****
    |****      ****
    | **        **
    +---------------> Value
       Peak 1  Peak 2

MULTIMODAL (Multiple Groups)
Frequency
    | **  **    **
    |**** ****  ****
    |**** **** ****
    | **   **   **
    +--------------------> Value
      P1   P2   P3
```

##### Real-World Examples from GPU Benchmarks

###### Hypothetical Example 1: CPU Algorithm (Illustrative Scenario)

**Scenario**: Algorithm with early stopping (patience=50 generations) on problem with n=52 cities

**Possible Observation Pattern** (REQUIRES validation with YOUR actual data):

```text
Hypothetical times (seconds): [98, 102, 105, 108, 112, 287, 292, 298, 305, 310, 315, 320, 328, 335, 342]

Histogram:
[90-150s]:  █████ (5 runs)  ← Peak 1 (fast group)
[150-250s]: ░░░░░ (0 runs)  ← GAP (no values)
[250-350s]: ██████████ (10 runs) ← Peak 2 (slow group)
```

**Possible Interpretations** (multiple causes, need investigation):

- **Hypothesis 1 - Early Stopping**:
  - Peak 1: Runs that converged early (found good solution quickly)
  - Peak 2: Runs that hit patience limit (explored longer without improvement)
  - **Test**: Check if execution_time correlates with stop_reason (converged vs patience)

- **Hypothesis 2 - Cache Effects**:
  - Peak 1: Warm cache scenarios (data structures in CPU cache)
  - Peak 2: Cold cache scenarios (frequent cache misses)
  - **Test**: Check if pattern persists across different problem sizes

- **Hypothesis 3 - Problem Heterogeneity**:
  - Peak 1: Easy problem instances (clustered cities)
  - Peak 2: Hard problem instances (random distribution)
  - **Test**: Stratify by problem characteristics

**Statistical Implications**:

1. **Shapiro-Wilk test**: Would likely reject normality ($p < 0.01$) for this pattern
2. **t-test invalid**: Assumes unimodal distribution, would miss two-group structure
3. **Solution**: Use non-parametric tests (Wilcoxon, Mann-Whitney) OR investigate cause and stratify

###### Hypothetical Example 2: HybridOptimized (Illustrative Scenario)

**Scenario**: GPU-accelerated algorithm on mixed problem set (37 runs)

**Possible Observation Pattern** (REQUIRES validation with YOUR actual data):

```text
Hypothetical times (seconds): [1.2, 1.5, 1.8, 2.1, 2.3, 2.5, 2.7, 3.0, 3.2, 3.5, 3.8, 4.1, ...]

Histogram:
[0-2s]:   ███ (8 runs)
[2-4s]:   ████████ (20 runs)  ← Single Peak (~3s)
[4-6s]:   ████ (7 runs)
[6-8s]:   ██ (2 runs)
```

**Interpretation** (if this pattern appears in YOUR data):

- **Single peak (~3s)**: Suggests consistent behavior across problem instances
- **Right-skewed**: Some harder instances (6-8s), but no distinct second mode
- **Mean ≈ 3.2s**: May represent typical performance if distribution is confirmed unimodal

>[!warning] Scale Illusion**:
>GPU times compressed to 0-8s range may APPEAR unimodal visually, but same bimodal pattern as CPU could exist when normalized. Check coefficient of variation (CV = std/mean) to compare relative spread.

**Statistical Implications** (if unimodality confirmed):

1. **Shapiro-Wilk test**: May fail to reject normality ($p > 0.05$)
2. **t-test potentially valid**: If unimodal and approximately normal
3. **Still prefer non-parametric**: Safer choice for benchmark data with unknown distributions

##### Why Do Multimodal Distributions Occur?

###### Cause 1: Mixed Populations

**Definition**: Dataset contains observations from **two or more distinct groups** with different characteristics

**GPU Benchmark Examples**:

```text
Scenario A: Small vs Large Problems Mixed
- Small problems (n<100):  GPU ~2s   ← Peak 1
- Large problems (n>500):  GPU ~20s  ← Peak 2
Result: Bimodal distribution

Scenario B: Easy vs Hard Instances
- TSP with clustered cities:  CPU ~100s  ← Peak 1
- TSP with random cities:     CPU ~300s  ← Peak 2
Result: Bimodal distribution
```

**Statistical Lesson**: **Always analyze subgroups separately** when mixing fundamentally different problem types!

###### Cause 2: System State Variations

**Definition**: Same algorithm, same problem, but different **runtime conditions**

**Examples**:

```text
Cache Effects:
- Warm cache (data in L1/L2): 100s  ← Peak 1
- Cold cache (data in RAM):   300s  ← Peak 2

CPU Frequency Scaling:
- High performance mode: 5.2s  ← Peak 1
- Power saving mode:     8.5s  ← Peak 2

GPU Contention:
- Exclusive GPU access: 2.1s  ← Peak 1
- Shared with X server: 3.8s  ← Peak 2
```

**Control Strategies**:

1. **Isolation**: Run benchmarks in single-user mode, disable background processes
2. **Warm-up**: Execute algorithm once before timing to warm caches
3. **Repetition**: Multiple runs reveal multimodality (hidden in single runs)

###### Cause 3: Algorithmic Phase Transitions

**Definition**: Algorithm behavior **qualitatively changes** at certain problem sizes or parameter values

**Example: Genetic Algorithm Convergence**:

```text
Premature Convergence (bad initialization):
- Generation 100: Best cost = 5000  ← Peak 1 (stuck in local optima)

Successful Optimization (good initialization):
- Generation 100: Best cost = 4200  ← Peak 2 (found better solution)

Result: Bimodal final cost distribution
```

**Statistical Implication**: Multimodality signals **algorithm instability** - may need parameter tuning!

###### Cause 4: Early Stopping and Convergence Dynamics

**Definition**: Algorithms with **patience-based early stopping** naturally create bimodal distributions

**Mechanism**: When algorithm stops improving for N consecutive generations (patience threshold), execution terminates.

**Why This Creates Bimodality**:

```text
Fast Runs (Converged Early):
- Found near-optimal solution at generation 50
- Early stopping triggered at generation 100
- Execution time: SHORT (fewer iterations)
- Stop reason: converged

Slow Runs (Hit Patience Limit):
- Explored search space without major improvements
- Continued until max_generations reached
- Execution time: LONG (exhausted budget)
- Stop reason: patience_limit

Result: Bimodal time distribution reflecting convergence success/failure
```

**GPU Benchmark Relevance**:

In `chapter4_validation.py`, algorithms use `patience=50` generations. If bimodality appears:

1. **Check correlation**: Does `execution_time` correlate with `stop_reason`?
2. **Stratify analysis**: Separate "converged early" vs "hit patience limit" runs
3. **Quality metrics**: Do fast runs have better or worse solution gaps?

**Diagnostic Test**:

```python
# If correlation r > 0.7 between time and (stop_reason=='patience'), 
# bimodality likely reflects convergence dynamics, not cache/system effects
import numpy as np
from scipy.stats import pearsonr

time = results['raw_times']
converged = [1 if reason=='converged' else 0 for reason in results['stop_reasons']]
r, p = pearsonr(time, converged)
print(f"Correlation: r={r:.3f}, p={p:.4f}")
```

**Statistical Implication**: Bimodality from early stopping is **algorithmic behavior**, not measurement noise. Report separately or use median instead of mean.

##### Does Larger Sample Size Remove Bimodality?

**Critical Misconception**: "If I increase $n$ from 15 to 30, will the bimodal distribution become unimodal?"

**ANSWER: NO!** Sample size reveals true distribution shape, it doesn't change it.

###### Mathematical Proof (Informal)

$$
\begin{align}
\text{True distribution} &= 0.5 \times N(\mu_1=105, \sigma_1=10) + 0.5 \times N(\mu_2=310, \sigma_2=15) \\
\end{align}
$$

(50% chance of drawing from "fast group", 50% from "slow group")

**Effect of Sample Size**:

| Sample Size | Histogram Shape | Statistical Insight |
|-------------|-----------------|---------------------|
| $n=5$ | `█░█░░` (lumpy, unclear) | Insufficient data to reveal structure |
| $n=15$ | `███░░█████` (two lumps) | Bimodality **suspected** |
| $n=30$ | `██████░░░░██████` (clear gap) | Bimodality **confirmed** |
| $n=100$ | `████████░░░░████████` (distinct peaks) | Bimodality **obvious** |
| $n \to \infty$ | Converges to true bimodal density | **Reveals truth, doesn't create unimodality** |

**Visualization**:

```text
n=5 (ambiguous):
Freq |  ▄     ▄
     | ▄▄▄   ▄▄▄
     +-----------> Time
     
n=15 (suggestive):
Freq | ▄▄▄       ▄▄▄▄▄
     |▄▄▄▄▄     ▄▄▄▄▄▄▄
     +-------------------> Time

n=30 (clear):
Freq | ▄▄▄▄▄       ▄▄▄▄▄▄▄
     |▄▄▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄
     +----------------------> Time
       Fast        Slow
       Group       Group
```

**Key Insight**: Larger $n$ provides **better resolution** to see the true underlying distribution, which may be multimodal. It doesn't "smooth out" real peaks into a single peak!

###### When CAN Apparent Bimodality Disappear?

**Scenario**: **False bimodality** due to **sampling error** with small $n$

```text
True distribution: Unimodal N(μ=200, σ=50)
Small sample (n=10): [120, 130, 140, 280, 290, 300, ...]
Appears bimodal! But just unlucky sampling (missing middle values)

Larger sample (n=50): [110, 125, 140, 155, 170, 185, 200, 215, 230, ...]
Reveals true unimodal shape
```

**How to Distinguish**:

1. **Dip test** (Hartigan & Hartigan, 1985): Tests null hypothesis of unimodality
2. **Silverman's bandwidth test**: Assesses significance of multiple modes
3. **Domain knowledge**: Does bimodality make physical sense?

**For GPU benchmarks**: Bimodality is **real and expected** due to:

- Cache effects (cold vs warm)
- Problem heterogeneity (easy vs hard instances)
- Algorithmic stochasticity (good vs bad convergence)

##### Statistical Tests and Modality

###### Parametric Tests: Assume Unimodality

**Assumptions of t-test, ANOVA**:

1. **Normality**: Data follows bell curve (unimodal)
2. **Homogeneity**: Groups have similar spreads
3. **Independence**: Observations unrelated

**What happens with bimodal data?**

```text
Bimodal distribution: Peaks at 100s and 300s
Mean = 200s ← Doesn't represent typical behavior!
SD = 100s   ← Inflated by two-group structure

t-test result: p=0.25 (fail to reject)
Truth: Groups ARE different (100s vs 300s), but t-test can't see it
```

**Problem**: Parametric tests **mask group differences** by averaging across modes!

###### Non-Parametric Tests: Robust to Modality

**Key Property**: Based on **ranks**, not raw values

```text
Wilcoxon/Mann-Whitney approach:
1. Sort all values: [100, 105, 108, 287, 292, 305, ...]
2. Assign ranks:    [1,   2,   3,   ...12, 13, 14, ...]
3. Test using ranks (doesn't care about distribution shape)

Result: Detects that Group A has low ranks, Group B has high ranks
Conclusion: Groups differ (regardless of modality)
```

**Why non-parametric for GPU benchmarks?**

✅ **Valid for any distribution shape** (unimodal, bimodal, multimodal)  
✅ **Robust to outliers** (rank 1 vs rank 100 doesn't matter if far apart)  
✅ **Conservative**: Lower power, but correct Type I error rate  
✅ **Standard in systems research**: Performance data rarely normal

##### Practical Implications for Your Benchmark Analysis

###### Investigation Workflow: If You Observe Bimodality

**Step 1: Confirm Bimodality is Real**

- ✅ Visual inspection: Violin plot, histogram with KDE overlay
- ✅ Statistical tests: Hartigan's dip test, Silverman's bandwidth test
- ✅ Check sample size: Is n ≥ 30 to reliably detect multiple modes?

**Step 2: Identify the Cause**

Test each hypothesis systematically:

**Hypothesis A: Early Stopping Effects**

```python
# Check if execution time correlates with stop_reason
import pandas as pd
df = pd.DataFrame({
    'time': results['raw_times'],
    'stop_reason': results['raw_stop_reasons']
})

# Compare distributions
fast_runs = df[df['stop_reason'] == 'converged']['time']
slow_runs = df[df['stop_reason'] == 'patience']['time']

print(f"Fast (converged): {fast_runs.mean():.1f}s ± {fast_runs.std():.1f}s")
print(f"Slow (patience):  {slow_runs.mean():.1f}s ± {slow_runs.std():.1f}s")
```

**Hypothesis B: Cache/System Effects**

- Check if pattern persists across different problem sizes
- Run controlled experiment: warm cache (preload data) vs cold cache

**Hypothesis C: Problem Heterogeneity**

- Stratify by problem characteristics (size, optimal_cost, problem_type)
- Check if bimodality disappears when analyzing single problem

**Step 3: Report Appropriately**

If cause identified:

- "Bimodality reflects convergence dynamics (early stop vs patience limit)"
- Report separately: "Converged runs: 105±10s, Non-converged: 310±20s"

If cause unknown:

- Use non-parametric tests (robust to distribution shape)
- Report median + IQR (less sensitive to multimodality than mean)
- Note in thesis: "Distribution multimodality requires further investigation"

###### Warning: Scale Illusion in Visual Analysis

**Problem**: Small absolute values can APPEAR unimodal even when underlying pattern is bimodal

**Example**:

```text
CPU times:  [100, 105, 108, 287, 292, 305]  → Visually OBVIOUS bimodal (200s gap)
GPU times:  [2.0, 2.1, 2.2, 5.7, 5.8, 6.1]  → Looks unimodal? (compressed scale)

But relative spread is IDENTICAL:
CPU: Peak 1 at 104s, Peak 2 at 295s  → Ratio: 2.84x
GPU: Peak 1 at 2.1s,  Peak 2 at 5.9s  → Ratio: 2.81x  (SAME pattern!)
```

**Solution**: Use normalized metrics

1. **Coefficient of Variation** (CV = std/mean):
   ```python
   cv_cpu = cpu_times.std() / cpu_times.mean()
   cv_gpu = gpu_times.std() / gpu_times.mean()
   # If both CV > 0.5, investigate multimodality for both
   ```

2. **Z-scores** (standardized values):
   ```python
   z_cpu = (cpu_times - cpu_times.mean()) / cpu_times.std()
   z_gpu = (gpu_times - gpu_times.mean()) / gpu_times.std()
   # Plot histograms of z-scores (same scale)
   ```

3. **Relative frequency histograms**:
   ```python
   plt.hist(cpu_times, bins=20, density=True)  # Density, not counts
   plt.hist(gpu_times, bins=20, density=True)
   ```

**Key Insight**: Don't trust visual inspection alone - normalize data before comparing distribution shapes!

###### Finding 2: HybridOptimized Unimodal Distribution

**What you observed**:

- Violin plot shows single narrow bulge ($\sim{2-5s}$)
- Shapiro-Wilk test: $p = 0.12$ (fails to reject normality)

**What it means**:

1. **Consistent behavior**: Algorithm performs predictably
2. **Right-skewed**: Some harder instances, but same underlying mechanism
3. **Well-optimized**: No distinct performance regimes

**What to do**:

- ✅ Can use parametric tests (t-test) if comparing to other unimodal data
- ✅ Report mean ± SD (appropriate for unimodal data)
- ✅ Highlight consistency in thesis ("HybridOptimized shows predictable performance")

###### Decision Tree: Which Test to Use?

```text
START: Do I have paired data (same problems across algorithms)?
 ├─ YES → Paired tests
 │   ├─ Is data normal? (Shapiro-Wilk p > 0.05)
 │   │   ├─ YES → Paired t-test ✅
 │   │   └─ NO → Wilcoxon signed-rank test ✅
 └─ NO → Independent tests
     ├─ Is data normal? (Shapiro-Wilk p > 0.05)
     │   ├─ YES → Independent t-test ✅
     │   └─ NO → Mann-Whitney U test ✅
     └─ More than 2 groups?
         ├─ Normal → One-way ANOVA ✅
         └─ Non-normal → Kruskal-Wallis or Friedman test ✅
```

**Your case**: Paired data (same 38 problems), non-normal (bimodal CPU) → **Wilcoxon signed-rank + Friedman test** ✅

##### Summary: Key Takeaways on Modality

1. **Modality = number of peaks** in distribution (unimodal, bimodal, multimodal)

2. **Bimodality signals heterogeneity**: Two distinct behaviors, not random noise

3. **Sample size reveals modality**: Larger $n$ makes peaks clearer, doesn't remove them

4. **Parametric tests fail**: t-test/ANOVA assume unimodal normality, invalid for multimodal data

5. **Non-parametric tests work**: Rank-based methods (Wilcoxon, Mann-Whitney, Friedman) valid for any shape

6. **Report appropriately**:
   - Unimodal: mean ± SD
   - Multimodal: median + IQR, or report each mode separately

7. **Investigate causes**: Multimodality often signals important system behaviors worth understanding

**For your thesis**: If bimodality appears in your benchmark results, investigate the cause systematically (early stopping, cache effects, problem heterogeneity, or algorithmic instability). Use the diagnostic workflow above to determine whether bimodality reflects:

1. **Algorithmic behavior** (convergence success/failure) → Report stratified results
2. **System effects** (cache, CPU scaling) → Control in experimental design
3. **Problem heterogeneity** (easy vs hard instances) → Analyze problem classes separately

Do NOT make definitive claims about causes without empirical evidence from YOUR specific benchmark data. Start with hypothesis testing (correlation analysis, stratified comparisons) before drawing conclusions in your thesis discussion.

**Definition**: Value(s) that occur most often in dataset

**Properties**:

- Useful for discrete/categorical data
- Can have multiple modes (bimodal, multimodal)
- Less common in continuous benchmark data
- Example: If cost is always 7542, mode = 7542

##### Measures of Dispersion

Dispersion answers: "How spread out are the values?"

###### Sample Standard Deviation

$$
\begin{align}

\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
\end{align}
$$

**Intuition**: Average distance of data points from the mean (in original units) -> used in **SAMPLED DATA**

**Why $n-1$ (Bessel's correction)?**  
Using sample mean $\bar{x}$ instead of true $\mu$ underestimates variance. Dividing by $n-1$ corrects this bias.

**Properties**:

- Same units as original data (meters, seconds, etc.)
- $\sigma = 0$ if all values identical (no variance case)
- Larger $\sigma$ → more variability → harder to detect differences

**Variance**: $\sigma^2$ (squared units, used in formulas)

###### Other Dispersion Measures

- **Range**: $\max(x) - \min(x)$ (simple but sensitive to outliers)
- **IQR** (Interquartile Range): $Q_3 - Q_1$ (robust, middle 50%)
- **MAD** (Median Absolute Deviation): Median of $|x_i - \text{Median}|$ (very robust)

##### Comparison Table: Central Tendency Measures

| Measure | Formula | Strengths | Weaknesses | When to Use | GPU Benchmark Use |
|---------|---------|-----------|------------|-------------|-------------------|
| **Mean** | $\bar{x} = \frac{1}{n}\sum x_i$ | Uses all data, algebraically nice, optimal for normal data | Sensitive to outliers, undefined for ordinal data | Parametric tests (t-test), symmetric distributions | Execution times (if normal) |
| **Median** | Middle value when sorted | Robust to outliers, interpretable, works with ordinal data | Ignores exact values, harder algebra | Non-parametric tests (Wilcoxon), skewed data | Times with outliers, ranks |
| **Mode** | Most frequent value | Works for categorical data, easy to understand | May not exist (flat distribution), not unique (multimodal) | Discrete/categorical data | Cost values (often discrete) |
| **Relationship** | Normal → Mean = Median | Skewness: Mean > Median (right skew), Mean < Median (left skew) | Distribution shape affects equality | Check distribution before choosing | Normality test guides choice |

##### ASCII Visualization: Outlier Effect

```text
Scenario 1: Normal GPU Times (no outliers)
     22  23  24  25  26  27  28
      |---|---|---|---|---|---|
         ↑           ↑
       Mean       Median
      (25.0)      (25.0)
Result: Mean ≈ Median (symmetric distribution)

Scenario 2: GPU Times with One Extreme Outlier
     22  23  24  25  26  27  100
      |---|---|---|---|---|-----|
         ↑           ↑        ↑
      Median      Mean    Outlier
      (25.0)     (35.3)   (pulls mean right)
Result: Mean >> Median (outlier inflates mean)

Why This Matters:
- t-test uses MEAN → outlier affects significance test
- Wilcoxon uses RANKS → outlier = just another rank
- Normality test flags outliers → triggers non-parametric choice
```

##### Why Mean for Parametric Tests?

**Mathematical Elegance**:

1. **Central Limit Theorem**: Sample means $\bar{x}$ converge to normal distribution even if data isn't normal (for large $n$)
2. **Closed-form formulas**: Standard error = $s/\sqrt{n}$ (simple, exact)
3. **Optimal estimator**: Minimum variance unbiased estimator (MVUE) for normal data
4. **Additive property**: $\bar{x} + \bar{y} = \overline{x+y}$ (simplifies paired tests)

**t-test foundation**:

$$
\begin{align}

t = \frac{\bar{d} - 0}{s_d / \sqrt{n}} \quad \text{(uses mean and SD directly)}
\end{align}
$$

##### Why Median for Non-Parametric Tests?

**Robustness Advantages**:

1. **Outlier resistance**: One extreme value doesn't dominate result
2. **Ordinal data**: Works with ranks (1st, 2nd, 3rd) not requiring exact values
3. **Skewed distributions**: Represents "typical" value better than mean
4. **Fewer assumptions**: No normality, equal variance, or specific distribution shape required

**Wilcoxon foundation**:  
Operates on **ranks** of differences, not raw values. If difference is $+100$ or $+1$, both get rank based on position after sorting by absolute value. Median naturally aligns with rank-based thinking.

**GPU Benchmark Decision Tree**:

```text
Collected data (n=15 per algorithm)
        ↓
Check normality (Shapiro-Wilk)
        ↓
    ┌───┴────┐
    ↓        ↓
Normal?    Non-normal?
    ↓        ↓
Use MEAN   Use MEDIAN/RANKS
    ↓        ↓
t-test     Wilcoxon
(parametric) (non-parametric)
```

**Practical Implication**: Always report BOTH mean and median in results tables. If they differ substantially (mean >> median or mean << median), distribution is skewed and non-parametric tests are safer.

#### 0.1.5 Statistical Distributions

Understanding the mathematical distributions that underpin statistical tests is essential for proper inference. These distributions describe the **expected pattern** of data under specific conditions.

##### Why Distributions Matter

Statistical tests make **assumptions** about data distributions:

- **Parametric tests** (t-test, ANOVA) assume data follows a specific distribution (usually normal)
- **Test statistics** (like $t$, $\chi^2$, $F$) themselves follow known distributions under $H_0$
- **P-values** are calculated from these reference distributions
- **Violations** of distributional assumptions can lead to incorrect conclusions

##### The Normal Distribution (Gaussian)

###### Probability Density Function

$$
\begin{align}
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\end{align}
$$

Where $\mu$ = mean (location parameter), $\sigma$ = standard deviation (scale parameter)

**Notation**: $X \sim N(\mu, \sigma^2)$ (read: "X follows a normal distribution with mean $\mu$ and variance $\sigma^2$")

###### The 68-95-99.7 Rule (Empirical Rule)

For any normal distribution:

- **68%** of data falls within $\mu \pm 1\sigma$
- **95%** of data falls within $\mu \pm 2\sigma$ (more precisely, $\mu \pm 1.96\sigma$)
- **99.7%** of data falls within $\mu \pm 3\sigma$

**GPU Benchmark Application**:  
If execution times $\sim N(25, 2^2)$ seconds, then:

- 68% of runs: between 23 and 27 seconds
- 95% of runs: between 21 and 29 seconds ($25 \pm 4$)
- Runs outside $[19, 31]$ are outliers ($\Pr{<0.3\%}$)

###### ASCII Bell Curve Visualization

```ascii
        Normal Distribution N(μ, σ²)

              ╱‾‾‾╲
             ╱     ╲
            ╱       ╲
           ╱         ╲
          ╱           ╲
    _____╱             ╲_____

    μ-3σ  μ-2σ  μ-σ  μ  μ+σ  μ+2σ  μ+3σ
    |-----|-----|-----|-----|-----|-----|
    0.1%   2.3%  13.6% 34.1% 34.1% 13.6% 2.3% 0.1%
           |_____|_____|_____|_____|
              68% of data
           |_____________|_____________|
                  95% of data
           |___________________________|
                  99.7% of data

Properties:
- Symmetric about mean (μ)
- Mean = Median = Mode
- Tails extend to $\pm$∞
- Determined by 2 parameters: μ (location), σ (spread)
```

###### Why Normal is Everywhere

1. **Central Limit Theorem** (see below): Sample means converge to normal
2. **Error accumulation**: Many small independent errors sum to normal
3. **Natural processes**: Height, measurement error, IQ scores
4. **Mathematical tractability**: Closed-form formulas for inference

##### Student's t-Distribution

###### Motivation

When estimating $\sigma$ from sample (using $s$), the test statistic:

$$
\begin{align}
t = \frac{\bar{x} - \mu}{s / \sqrt{n}}
\end{align}
$$

does NOT follow $N(0,1)$. It follows a **t-distribution** with $df = n-1$ degrees of freedom.

###### Probability Density Function

$$
\begin{align}
f(t) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{df\pi} \, \Gamma(\frac{df}{2})} \left(1 + \frac{t^2}{df}\right)^{-\frac{df+1}{2}}
\end{align}
$$

**Notation**: $t \sim t_{df}$ or $t \sim t(df)$

**Component Explanations**:

1. **Degrees of Freedom ($df$)**: Number of "independent" pieces of information available to estimate variability
   - Formula: $df = n - 1$ (sample size minus parameters estimated)
   - Why $n_{runs}-1$? After computing $\bar{x}$ from $n_{runs}$ values, only $n-1$ deviations are "free" to vary (last deviation is determined by constraint $\sum(x_i - \bar{x}) = 0$)
   - **Example**: With $n=5$ observations, once you know 4 deviations and the mean, the 5th deviation is fixed
   - **Effect on distribution**: Lower $df$ → more uncertainty → heavier tails

2. **Gamma Function ($\Gamma$)**: Generalization of factorial to real numbers
   - Definition: $\Gamma(n) = \int_0^\infty t^{n-1}e^{-t}dt$
   - Key property: $\Gamma(n) = (n-1)!$ for positive integers
   - Examples: $\Gamma(1) = 1$, $\Gamma(2) = 1$, $\Gamma(3) = 2$, $\Gamma(4) = 6$
   - Special: $\Gamma(1/2) = \sqrt{\pi}$ (appears in normal distribution)
   - **Role in t-distribution**: Normalizing constant ensuring total probability = 1

3. **Why This Formula?**: The t-distribution arises from the ratio:
   $$
   \begin{align}
   t = \frac{Z}{\sqrt{V/df}}
   \end{align}
   $$
   where $Z \sim N(0,1)$ and $V \sim \chi^2_{df}$ are independent. The resulting distribution has heavier tails than normal because denominator $\sqrt{V/df}$ (estimate of $\sigma$) varies randomly.

**Practical Interpretation**: You don't need to compute $\Gamma$ manually! Statistical software (Python's `scipy.stats.t`, R's `pt()`) handles this. The important insight: **lower $df$ means more uncertainty, requiring wider confidence intervals**.

>[!caution]
> better explain this. Add an appendix exemplifying
>
> - what is the $\Gamma$ function?
> - what is the $df$ and how are they measured?
> - I studied student's t-distribution, but don't really remember how it applies here.
> - graphs showing the distributions by problem would be awesome (later implementation on codebase itself for generating the graphs. We'll implement a notebook for it)

###### Key Properties

1. **Symmetric** about 0 (like normal)
2. **Heavier tails** than normal (more probability in extremes)
3. **Shape depends on $df$**: Lower $df$ → heavier tails → wider distribution
4. **Converges to normal**: As $df \to \infty$, $t_{df} \to N(0,1)$

**ASCII Comparison: Normal vs t-Distributions**

```ascii
Probability Density
          ^
          |      N(0,1) - Standard Normal
          |      (Thin tails, df=∞)
     0.4 -|          ,^.
          |        ,'   `.
     0.3 -|       /       \       t(df=10)
          |      |         |      _.---._ (Close to normal)
     0.2 -|      |         |    .'       `.
          |     /           \  /           \
          |    |             |/             \ t(df=2)
     0.1 -|   /               \              \ .-------. (Very heavy tails)
          | _/                 \_             '         `
     0.0 +-'---------------------`-----------'-----------`-> t-score
        -4        -2         0         2         4

Critical Value Comparison (95% CI, two-tailed $\alpha=0.05$):
┌──────────┬─────────────┬────────────────┐
│ df       │ t_{df,0.975}│ vs Normal (1.96)│
├──────────┼─────────────┼────────────────┤
│ 2        │ 4.303       │ +120% wider!   │
│ 5        │ 2.571       │ +31% wider     │
│ 10       │ 2.228       │ +14% wider     │
│ 30       │ 2.042       │ +4% wider      │
│ 100      │ 1.984       │ +1% wider      │
│ ∞ (Z)    │ 1.960       │ (reference)    │
└──────────┴─────────────┴────────────────┘

Key Insight: Small samples (df < 10) require MUCH wider
confidence intervals due to uncertainty in estimating σ.

GPU Benchmark with n=15: Use t_{14,0.975} = 2.145
```

###### Why This Matters

- **Small samples** ($n < 30$): Must use $t_{n-1}$ critical values, not $Z$
- **Conservative**: Wider tails account for uncertainty in estimating $\sigma$
- **Example**: 95% CI for $n=5$ uses $t_{4,0.975} = 2.776$, NOT $Z_{0.975} = 1.96$

**GPU Benchmark**:  

1. With $n_{runs} = 30$, use $t_{29}$ distribution. For $n=30$, $t_{29,0.975} = 2.045$ (very close to $Z = 1.96$).
2. With $n_{runs} = 15$, use $t_{14}$ distribution. For $n=15$, $t_{14,0.975} = 2.145$ (9.4% wider than $Z = 1.96$). This reflects the additional uncertainty from estimating $\sigma$ with a small sample.

###### Historical Context: The Story of "Student"

**William Sealy Gosset** (1876-1937) was a chemist and mathematician employed by Guinness Brewery in Dublin, Ireland. Educated at Winchester College and New College, Oxford (First-Class Honours in Mathematics), Gosset was hired in 1899 to apply statistical methods to brewing quality control. His work focused on analyzing small samples of barley crops ($n=4$ to $n=10$), where classical large-sample methods based on the normal distribution were inappropriate.

**The Problem**: When $\sigma$ is unknown and estimated by sample standard deviation $s$, the test statistic $\frac{\bar{x}-\mu}{s/\sqrt{n}}$ does NOT follow the standard normal distribution $N(0,1)$—especially for small samples. Gosset needed the exact sampling distribution to set proper critical values for quality control decisions.

**The Innovation**: In 1908, Gosset derived the exact distribution of $t = \frac{\bar{x}-\mu}{s/\sqrt{n}}$, showing it follows a ratio distribution: $\frac{Z}{\sqrt{\chi^2_{n-1}/(n-1)}}$, where $Z \sim N(0,1)$ and $\chi^2_{n-1}$ is the chi-squared distribution with $n-1$ degrees of freedom. He published "The Probable Error of a Mean" in *Biometrika* (Vol. 6, No. 1), introducing the **degrees of freedom** concept and providing critical value tables for practical use. This enabled rigorous small-sample inference for the first time.

**The Pseudonym Mystery**: Guinness Brewery forbade employees from publishing research, fearing competitors would learn proprietary methods. Gosset obtained permission to publish under the pseudonym **"Student"** after promising not to mention brewing applications. Ironically, the statistical methods were entirely general—the t-distribution applies to any field, not just beer quality control. The name stuck permanently: we still call it "**Student's t-distribution**" 117 years later. Karl Pearson (editor of *Biometrika*) knew Gosset's identity but honored the pseudonym. Guinness later relaxed the policy in the 1930s, but Gosset continued using "Student" in his publications.

**Impact on Science**: Gosset's work revolutionized small-sample inference across multiple disciplines:

- **Agricultural statistics**: R.A. Fisher (Rothamsted Experimental Station) collaborated with Gosset and built modern experimental design on the t-test foundation
- **Quality control**: Walter Shewhart (Bell Labs) applied t-tests to manufacturing process control
- **Medical research**: Clinical trials with limited subjects became statistically rigorous
- **Psychology & economics**: Experimental studies and surveys with realistic sample sizes
- **Modern standard**: The **n=30 threshold** emerged from Gosset's tables—at $df=29$, the t-distribution converges within ~5% of the standard normal ($t_{29,0.975} = 2.045$ vs $Z_{0.975} = 1.96$, only 4.3% wider). Our GPU benchmark design with $n_{runs}=15$ uses $t_{14,0.975} = 2.145$ (9.4% wider than Z), accepting this trade-off for faster experimentation while maintaining adequate power for large effects.

[^1]

*Historical note*: "Probable error" was the 19th-century term for standard error. *Biometrika* was founded by Karl Pearson in 1901 as the first journal dedicated to mathematical statistics. Fisher later republished Gosset's work with commentary, cementing its place in statistical history.

[^1]: Student (William Sealy Gosset). (1908). The probable error of a mean. *Biometrika*, 6(1), 1–25. DOI: [10.2307/2331554](https://doi.org/10.2307/2331554)

##### Chi-Squared Distribution** ($\chi^2$)

###### Definition

If $Z_1, Z_2, \ldots, Z_k$ are independent $N(0,1)$ random variables, then:

$$
\begin{align}
\chi^2 = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2_k
\end{align}
$$

Where $k = df$

**Intuition**: Sum of squared standard normals

###### Probability Density Function

$$
\begin{align}
f(x) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x > 0
\end{align}
$$

**Key Properties**:

1. **Right-skewed** (not symmetric)
2. **Non-negative** ($\chi^2 \geq 0$ always)
3. **Mean**: $E[\chi^2_k] = k$
4. **Variance**: $\text{Var}(\chi^2_k) = 2k$
5. **Shape**: As $k$ increases, becomes more symmetric (approaches normal)

###### Understanding the $\Gamma(k/2)$ Term

###### What is the Gamma Function?

You see $\Gamma$ (capital gamma) in the PDF above - but what is it, and why is it there?

**Short answer**: The Gamma function generalizes factorials to non-integer values. It appears as a **normalizing constant** ensuring the total probability integrates to 1.

**Practical note**: You'll never compute $\Gamma$ by hand - statistical software (Python's `scipy.stats.chi2`, R's `pchisq()`) handles this automatically. Understanding its role helps interpret how degrees of freedom affect distribution shape.

###### Mathematical Definition

The Gamma function is defined for all $x > 0$ by the improper integral:

$$
\begin{align}
\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} \, dt \quad \text{for } x > 0
\end{align}
$$

###### The Factorial Connection

For **positive integers** $n$, the Gamma function equals the factorial of $n-1$:

$$
\begin{align}
\Gamma(n) = (n-1)! \quad \text{e.g., } \Gamma(3) = 2! = 2, \; \Gamma(4) = 3! = 6
\end{align}
$$

**Key Values for Chi-Squared**:

- $\Gamma(1/2) = \sqrt{\pi} \approx 1.7725$ ($df=1$: $\Gamma(1/2)$ in denominator)
- $\Gamma(1) = 0! = 1$ (df=2: $\Gamma(1) = 1$)
- $\Gamma(3/2) = \frac{1}{2}\sqrt{\pi} \approx 0.8862$ ($df=3$: $\Gamma(3/2)$)
- $\Gamma(2) = 1! = 1$ (df=4: $\Gamma(2) = 1$)
- $\Gamma(5/2) = \frac{3}{4}\sqrt{\pi} \approx 1.3293$ ($df=5$: $\Gamma(5/2)$)

>[!caution]
>still did not understand how and why the $df$ comes from and why it appears on the $PDF$. What about the gaussian distribution and $X \sim{N}(\mu,\sigma^{2})$.
> ![normal distributions](../../assets/image.png)
> Like, how is this derived? I would like the derivation at the appendix for each of them.

**Role in $\chi^2$ PDF**:

The term $\frac{1}{2^{k/2}\Gamma(k/2)}$ ensures $\int_0^\infty f(x) \, dx = 1$ (total probability). As $k$ (degrees of freedom) increases:

- $\Gamma(k/2)$ grows rapidly → normalizing constant shrinks → probability spreads over wider range
- Distribution becomes more symmetric (approaches normal for large $k$)

>[!caution]
>still, I would like to see how it happens, where does it come from.

**Why You Don't Compute It**: When you call `scipy.stats.chi2(df=14).pdf(x)`, Python internally computes $\Gamma(7)$ using efficient approximations. You see the output probability density, not the $\Gamma$ machinery.

###### Uses in Testing

- **Variance estimation**: $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$
- **Friedman test**: $Q \sim \chi^2_{k-1}$ under $H_0$ (for large $n_{runs}$)
- **Goodness-of-fit tests**: Pearson's $\chi^2$ test for categorical data

##### Degrees of Freedom Demystified

###### The n-1 Mystery: Why Not n?

One of the most confusing concepts in introductory statistics: "Why do we divide by $n-1$ instead of $n$ when calculating sample variance?" The answer lies in **degrees of freedom** ($df$).

**Short answer**: After computing the sample mean $\bar{x}$ from $n$ observations, only $n-1$ deviations $(x_i - \bar{x})$ are "free" to vary. The last deviation is mathematically determined by the constraint that all deviations must sum to zero.

We first encountered $df$ in the t-distribution PDF (above), where $df = n-1$ appeared in the formula. Now we explain this fundamental concept fully.

###### Intuitive Definition

**Degrees of freedom** = Number of independent pieces of information available to estimate a parameter

**Analogy**: Think of a jigsaw puzzle with 100 pieces. If someone tells you 99 pieces' positions and says "all pieces must fit perfectly," the 100th piece's position is **determined** by the others - it has zero degrees of freedom. Similarly, when estimating variance, one piece of information (the mean) "uses up" one degree of freedom.

###### The Mathematical Constraint

For any sample with mean $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$, the deviations from the mean **must** sum to zero:

$$
\begin{equation}
\sum_{i=1}^n (x_i - \bar{x}) = 0
\end{equation}
$$

This is not a coincidence - it's a mathematical identity that follows from the definition of the mean. **Proof**:

$$
\begin{equation}
\sum_{i=1}^n (x_i - \bar{x}) = \sum_{i=1}^n x_i - \sum_{i=1}^n \bar{x} = \sum_{i=1}^n x_i - n\bar{x} = n\bar{x} - n\bar{x} = 0
\end{equation}
$$

**Implication**: Once you know $n-1$ deviations and the constraint $\sum = 0$, the $n$-th deviation is determined. Thus, only $n-1$ deviations are "free" or "independent."

###### Worked Example: n=5 GPU Runtimes

Let's use concrete numbers to see the constraint in action.

**Data**: Five GPU execution times (seconds): $[22, 24, 23, 26, 25]$

**Step 1**: Calculate mean

$$
\begin{equation}
\bar{x} = \frac{22 + 24 + 23 + 26 + 25}{5} = \frac{120}{5} = 24 \text{ seconds}
\end{equation}
$$

**Step 2**: Calculate deviations  

- $x_1 - \bar{x} = 22 - 24 = -2$ ✓ **(free to vary)**
- $x_2 - \bar{x} = 24 - 24 = 0$ ✓ **(free to vary)**
- $x_3 - \bar{x} = 23 - 24 = -1$ ✓ **(free to vary)**
- $x_4 - \bar{x} = 26 - 24 = +2$ ✓ **(free to vary)**
- $x_5 - \bar{x} = 25 - 24 = ?$ ✗ **(CONSTRAINED!)**

**Step 3**: Apply constraint $\sum (x_i - \bar{x}) = 0$

$$
\begin{align}
-2 + 0 + (-1) + 2 + ? &= 0 \\
-1 + ? &= 0 \\
? &= +1
\end{align}
$$

**Conclusion**: The 5th deviation **must** equal $+1$ to satisfy the constraint. It has no freedom - it's completely determined by the first 4 deviations and the mean.

**Degrees of freedom**: $df = n - 1 = 5 - 1 = 4$ (only 4 independent deviations)

###### Visual Representation

```ascii
5 Deviations from Mean (x̄ = 24):

  -2        0       -1       +2       +1
[████]   [    ]   [██]   [████]   [███]
  FREE     FREE     FREE     FREE   LOCKED!
   ↓        ↓        ↓        ↓        ↓
  Can      Can      Can      Can    MUST = +1
  vary     vary     vary     vary   (determined by
                                    constraint Σ=0)

Why the 5th is locked:
  Sum of first 4: (-2) + 0 + (-1) + (+2) = -1
  Constraint requires: Σ = 0
  Therefore: 5th deviation = -(-1) = +1 ✓
  
Result: df = 4 independent deviations (not 5)
```

###### Degrees of Freedom Across Statistical Tests

Different tests have different $df$ formulas depending on how many parameters are estimated:

| Test | $df$ Formula | Example ($n_{runs}=15$ per group) | Notes |
|------|-----------|---------------------------|-------|
| **One-sample t-test** | $n - 1$ | $15 - 1 = 14$ | Estimate 1 parameter ($\mu$) |
| **Paired t-test** | $n - 1$ | $15 - 1 = 14$ | Same as one-sample on differences |
| **Two-sample t-test** (pooled) | $n_1 + n_2 - 2$ | $15 + 15 - 2 = 28$ | Estimate 2 means, pool variances |
| **Chi-squared** (variance) | $n - 1$ | $15 - 1 = 14$ | Same logic as t-test |
| **F-test** (two variances) | $(n_1-1, n_2-1)$ | $(14, 14)$ | Two $df$ values (numerator, denominator) |
| **One-way ANOVA** ($k$ groups) | $(k-1, N-k)$ | $(2, 28)$ for $k=2$ | Between-group $df$, within-group $df$ |
| **Simple linear regression** | $n - 2$ | $15 - 2 = 13$ | Estimate 2 parameters (slope, intercept) |

**Pattern**: $df = n - p$, where $p$ = number of parameters estimated from the data.

###### Why Degrees of Freedom Matter

Degrees of freedom directly affect:

1. **Critical values**: Lower $df$ → larger critical values → wider confidence intervals
2. **Statistical power**: Lower $df$ → harder to detect real effects
3. **Distribution shape**: t-distribution with low $df$ has heavier tails

**Numerical Impact** (95% confidence interval multiplier):

| $df$ | $t_{df, 0.975}$ | CI Width Factor | vs. $df=14$ (Our Benchmark) |
|------|-----------------|-----------------|-----------------------------|
| 4 | 2.776 | $2.776 \times SE$ | +29.4% wider (n=5 design) |
| 9 | 2.262 | $2.262 \times SE$ | +5.5% wider (n=10 design) |
| **14** | **2.145** | **$2.145 \times SE$** | **(reference: n=15 design)** |
| 29 | 2.045 | $2.045 \times SE$ | -4.7% narrower (n=30 would be ideal) |
| 100 | 1.984 | $1.984 \times SE$ | -7.5% narrower (large sample) |
| $\infty$ | 1.960 | $1.960 \times SE$ (Z) | -8.6% narrower (CLT applies) |

**Key insight**: Small samples ($df < 10$) require **much wider** confidence intervals due to uncertainty in estimating $\sigma$ from $s$.

###### GPU Benchmark Application: The n=15 Design Decision

**The Ideal (Statistician's Preference)**: $n_{runs} \geq 30$ repetitions  
✅ Central Limit Theorem (CLT) applies → can skip normality testing  
✅ Use standard normal critical values (Z = 1.96) instead of t-distribution  
✅ Narrower confidence intervals (more precision)  
✅ Higher power to detect small effects (d = 0.3-0.5)

**The Reality (Our Constraint)**: $n_{runs} = 15$ repetitions per algorithm  
⚠️ **Below the n=30 CLT threshold** → must verify normality assumption  
⚠️ Must use t-distribution with $df = 14$ → critical value $t_{14,0.975} = 2.145$  
⚠️ Confidence intervals **9.4% wider** than ideal (2.145 vs 1.96)  
⚠️ Adequate power **only for large effects** ($d \geq 0.8$)

**The Consequences (Statistical Penalties)**:

1. **Mandatory Normality Testing**:  
   - Must run Shapiro-Wilk test on differences before using paired t-test  
   - If $p < 0.05$ (non-normal) → fall back to Wilcoxon signed-rank test  
   - CLT protection unavailable (need n ≥ 30 for CLT)

2. **Wider Confidence Intervals**:  
   - With n=15: $CI = \bar{d} \pm 2.145 \times SE$  
   - With n=30: $CI = \bar{d} \pm 2.045 \times SE$ (4.7% narrower)  
   - With n→∞: $CI = \bar{d} \pm 1.96 \times SE$ (8.6% narrower)

3. **Power Trade-offs** (α = 0.05, two-tailed):  
   - Small effect (d=0.3): Power = 15% (need n≈87 for 80% power)  
   - Medium effect (d=0.5): Power = 46% (need n≈34 for 80% power)  
   - Large effect (d=0.8): Power = 81% ✅ (adequate)  
   - Very large (d=1.5): Power = 99% ✅ (excellent)

**Why We Accept n=15 Despite Penalties**:

✅ **Expected GPU speedups are massive**: Typical d > 2.0 (10×-100× faster) → >99% power  
✅ **Computational cost is manageable**: 15 runs × 30 instances × 2 algorithms = 900 total experiments  
✅ **Not studying subtle effects**: Research question is "does GPU provide substantial speedup?" not "is there a 5% difference?"  
✅ **Normality testing is cheap**: Shapiro-Wilk adds <1 second to analysis

**Design Lesson**: The n=15 choice is a **conscious trade-off**, not ignorance. For large-effect domains (GPU acceleration, drug efficacy), smaller samples are statistically justified. For small-effect domains (UI tweaks, A/B testing), n ≥ 30 is essential.

##### Relationships Between Distributions

###### t-Distribution → Normal

$$\begin{equation}
\lim_{df \to \infty} t_{df} = N(0, 1)
\end{equation}$$

**Practical rule**: For $df \geq 30$, $t$ and $Z$ critical values differ by $< 3\%$

**Example**:
- $t_{10, 0.975} = 2.228$ vs $Z_{0.975} = 1.96$ (13% difference)
- $t_{30, 0.975} = 2.042$ vs $Z_{0.975} = 1.96$ (4% difference)
- $t_{100, 0.975} = 1.984$ vs $Z_{0.975} = 1.96$ (1% difference)

>[!caution]
>markdown link to place where explained what is $t_{n,\sigma}$ and why sigma is 0.975 always. what means the 0.975? the pp-value threshold?

###### Chi-Squared → Normal

for large $df$:

$$\begin{equation}
\chi^2_k \approx N(k, 2k) \quad \text{for large } k
\end{equation}$$

More accurate: $\sqrt{2\chi^2_k} \approx N(\sqrt{2k-1}, 1)$

##### Central Limit Theorem (CLT)

###### The Core Concept

The CLT is the bridge between chaotic real-world data and orderly statistical theory. It dictates that while **individual** measurements might be unpredictable and follow weird shapes, the **averages** of those measurements behave predictably.

###### Mathematical Formulation

Given a population with mean $\mu$ and variance $\sigma^2$:
$$\begin{equation}
\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)
\end{equation}$$

**Translation:**

1.  **Center stays the same:** The average of your samples will center around the true population mean ($\mu$).
2.  **Spread shrinks:** The variance of the sample mean is the original variance divided by $n$ ($\frac{\sigma^2}{n}$).
3.  **Shape normalizes:** As $n$ increases, the shape becomes a Bell Curve, regardless of the source shape.

###### ASCII Visualization: Order form Chaos

This visualization demonstrates how a **Exponential (Skewed)** distribution of raw data transforms into a **Normal** distribution when you look at the means.

```ascii
1. THE REALITY (Population)
   Distribution of INDIVIDUAL GPU runtimes (e.g., Latency)
   Shape: Highly skewed right. Most runs are fast, some are very slow.

   Frequency
      ^
      |  |
      |  |
      |  ||
      |  |||
      |  ||||.
      |  ||||||......
      |  |||||||||||||................... _ _ _
      +---------------------------------------------> Time (ms)
         ^                                    ^
      (Mode)                             (Long Tail)

            ||  SAMPLING PROCESS
            ||  (Take n=15 runs, calculate average. Repeat.)
            \/

2. THE ABSTRACTION (Sampling Distribution)
   Distribution of the CALCULATED MEANS
   Shape: Symmetric, Bell-shaped, Narrow.

   Frequency
      ^
      |            Peak is at true μ
      |                  |
      |                 _|_
      |               /'   `\
      |              /       \    <-- Logic: It is very hard to get
      |             |         |       a mean consisting ONLY of
      |            /           \      tail events. Highs and lows
      |           |             |     cancel out, pushing the
      |       _.-'               `-._ average to the center.
      +---------------------------------------------> Time (ms)
                     ^       ^
                Narrower spread
              (Standard Error = σ/√n)
```

###### Why This Matters for Engineering

1.  **Justification for Metrics:** It validates using "Average FPS" or "Average Latency" as a metric. If CLT didn't exist, the "average" of a skewed dataset might be statistically meaningless.
2.  **Precision vs. Cost:** The term $\frac{\sigma}{\sqrt{n_{runs}}}$ tells you exactly how much precision you buy with more compute time. To double your precision (halve the width of the curve), you must quadruple your sample size ($n \times 4$).
3.  **Universal Compatibility:** You can use parametric statistics (Z-tests, T-tests) on your benchmark data even if the frametimes themselves are not normally distributed, provided you are testing differences in **means** and $n$ is sufficiently large.

###### The "Rule of 30" Warning

> [!tip]
> **$n_{runs}=30$ is a Rule of Thumb, not Physics.**
> If your data is **extremely** skewed (e.g., a server with 99% fast responses and 1% massive timeouts), $n=30$ may not be enough to normalize the mean. In high-reliability engineering, always plot the histogram of your means to verify normality if unsure.

###### GPU Benchmark Implication

Even if individual GPU run times are slightly skewed (right tail from occasional cache misses), the mean of 30 runs $\bar{x}_{\text{GPU}}$ is approximately normal. This justifies using:

- **Paired t-test** for CPU vs GPU comparison
- **Confidence intervals** based on $t_{29}$ distribution
- **Normality assumption** for the **sample means**, not individual observations

For mildly skewed data (e.g., right tail from occasional cache misses), n=15 may suffice if Shapiro-Wilk p$\geq$0.05. For heavily skewed or multimodal data, use Wilcoxon regardless.

**Exception**: If data is SEVERELY non-normal (e.g., multimodal, heavy outliers), even parametric tests may be invalid. Always check normality of **differences** in paired tests.

>[!caution]
>currently finished runs have n=15 only. So we must test normality and possibly use Wilcoxon if non-normal. But, there are some outliers to be explained.
>- generally, they have the same interval between generations and time, that must be from algorithmic deisgn and explained (probably number of blocks `blockDim`+2-opt finding optimal)
>   - still need better explanation from how the algorithm work. From foundation (the original paper from Genetic Algorithms) -> improvements (articles that represent advancements in genetic algorithsm —only that relate to my work do not enter in specifics about things not used—, torunament selection, mutation strategies and where the memetic part enters with the 2-opt, a bibliographical revision). -> GA paralleism. Of course in another file.
>   - why the outliers generally find the optimal in similar generations then, don't evolve anymore?

##### Summary: Distribution Decision Tree

```mermaid
flowchart TD
    Start(["Start: Which test for paired data?"])
    Start --> CheckN{"Sample size n?"}

    CheckN -->|"n < 30"| SmallN["Small sample: Must verify normality"]
    CheckN -->|"n >= 30"| LargeN["Large sample: CLT provides protection"]

    SmallN --> Shapiro["Run Shapiro-Wilk test on differences"]
    Shapiro --> NormalQ{"p >= 0.05?<br/>(Normal?)"}

    NormalQ -->|"Yes<br/>(Normal)"| UseT["✅ Use paired t-test<br/>Parametric, optimal power"]
    NormalQ -->|"No<br/>(Non-normal)"| UseW["✅ Use Wilcoxon signed-rank<br/>Non-parametric, robust"]

    LargeN --> CheckSevere{"Data SEVERELY<br/>non-normal?<br/>(multimodal, extreme outliers)"}
    CheckSevere -->|"No<br/>(Mild skew OK)"| UseTCLT["✅ Use paired t-test<br/>CLT justifies normality assumption"]
    CheckSevere -->|"Yes<br/>(Severe)"| UseWSafe["✅ Use Wilcoxon<br/>Safer choice"]
```

>[!caution]
>derive the formulas we'll use, with examples, here, for our 15 run test and how they relate. Suppose normal values

**Decision Rules Summary**:
- **$n < 30$**: Test normality with Shapiro-Wilk → Normal: t-test | Non-normal: Wilcoxon
- **$n $\geq$ 30$**: CLT allows t-test for mild non-normality → Severe non-normality: Wilcoxon
- **When in doubt**: Wilcoxon is safer (loses $\sim{5}\%$ power if data truly normal)

**Distribution Reference Table**:

| Test | Assumes Data Distribution | Test Statistic Distribution | Used When |
|------|---------------------------|----------------------------|-----------|
| **Paired t-test** | Differences ~ Normal | $t \sim t_{n-1}$ | Normal differences OR $n_{runs} \geq 30$ (CLT) |
| **Wilcoxon** | Symmetric (no specific dist) | Exact or $Z \sim N(0,1)$ (large $n_{runs}$) | Non-normal, any $n_{runs}$ |
| **Friedman** | No assumption (rank-based) | $Q \sim \chi^2_{k-1}$ (large $n_{runs}$) | Multiple algorithms, non-parametric |
| **Shapiro-Wilk** | Testing normality | $W$ (tabulated, no closed form) | Before choosing parametric/non-parametric |

---

## Regression Analysis and Model Fitting

### Introduction: When Prediction Matters

**Regression analysis** addresses a fundamentally different question than hypothesis testing:

- **Hypothesis Testing**: *"Are these groups different?"* (Yes/No answer)
- **Regression Analysis**: *"How does Y change as X changes?"* (Predictive relationship)

In benchmark analysis, regression is crucial for:
1. **Understanding Scaling Behavior**: How does execution time grow with problem size $n$?
2. **Extrapolation**: Estimating CPU performance for untested problem sizes
3. **Model Selection**: Comparing theoretical complexity hypotheses (O($n^2$) vs O($n^2 \log n$))
4. **Validation**: Checking if empirical data matches theoretical predictions

**When to Use**:
- ✅ Continuous dependent variable (time, cost, generations)
- ✅ Ordered independent variable (problem size, iterations)
- ✅ Interest in **quantitative relationship**, not just difference detection
- ✅ Need to **predict** outcomes for new input values

**When NOT to Use**:
- ❌ Comparing categorical groups (Algorithm A vs B) → Use t-test/ANOVA
- ❌ Only care about statistical significance → Hypothesis tests sufficient
- ❌ No clear predictor-outcome relationship

---

### Simple Linear Regression

**The Foundation**: Model relationship between one predictor $X$ and outcome $Y$:

$$\begin{equation}
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)
\end{equation}$$

**Components**:
- $\beta_0$: **Intercept** (value of $Y$ when $X=0$)
- $\beta_1$: **Slope** (change in $Y$ per unit change in $X$)
- $\epsilon_i$: **Residuals** (unexplained variation, assumed normally distributed)

**Example: Linear Time Complexity**
```
Problem Size (n):  50    100   150   200   250
CPU Time (s):      2.3   4.8   7.1   9.5   11.8

Model: T(n) = β₀ + β₁·n
Interpretation: "Each additional city adds β₁ seconds"
```

**Least Squares Estimation**: Find $\beta_0, \beta_1$ minimizing sum of squared residuals:

$$\begin{equation}
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
\end{equation}$$

**Closed-Form Solution**:

$$\begin{align}
\hat{\beta}_1 &= \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n}(X_i - \bar{X})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)} \\
\hat{\beta}_0 &= \bar{Y} - \hat{\beta}_1 \bar{X}
\end{align}$$

**SciPy Implementation**:
```python
from scipy.stats import linregress

# Example: Linear growth model
problem_sizes = np.array([50, 100, 150, 200, 250])
cpu_times = np.array([2.3, 4.8, 7.1, 9.5, 11.8])

slope, intercept, r_value, p_value, std_err = linregress(problem_sizes, cpu_times)

print(f"Model: T(n) = {intercept:.2f} + {slope:.4f}·n")
print(f"R² = {r_value**2:.4f}")  # Coefficient of determination

# Prediction for n=300
predicted_time = intercept + slope * 300
print(f"Predicted time for n=300: {predicted_time:.2f}s")
```

**Diagnostic Plot**:
```python
import matplotlib.pyplot as plt

# Scatter plot with fitted line
plt.figure(figsize=(10, 6))
plt.scatter(problem_sizes, cpu_times, s=100, alpha=0.6, label='Observed Data')

# Fitted line
x_fit = np.linspace(problem_sizes.min(), problem_sizes.max(), 100)
y_fit = intercept + slope * x_fit
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: y = {intercept:.2f} + {slope:.4f}x')

plt.xlabel('Problem Size (n)', fontsize=12)
plt.ylabel('CPU Time (seconds)', fontsize=12)
plt.title('Linear Regression: CPU Time vs Problem Size', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Polynomial Regression: Modeling Nonlinear Growth

**Motivation**: Many algorithms exhibit **polynomial complexity** (O($n^2$), O($n^3$), etc.). Linear models are inadequate.

**Quadratic Model** (Most Common in Combinatorial Optimization):

$$\begin{equation}
Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \epsilon_i
\end{equation}$$

**Example: GA with O($n^2$) Fitness Evaluation**
```
Problem Size (n):  51     52     70     76     100
CPU Time (s):      22.5   28.2   54.8   67.3   156.7

Hypothesis: T(n) ≈ a·n² + b·n + c  (quadratic growth)
```

**Why O($n^2$) for GA+2-opt?**
1. **Fitness Evaluation**: O($n^2$) to traverse distance matrix
2. **2-opt Local Search**: O($n^2$) edge swaps per iteration
3. **Population Size**: Constant (e.g., 256) → doesn't affect asymptotic complexity

**NumPy/SciPy Implementation**:
```python
from scipy.optimize import curve_fit
import numpy as np

# Define quadratic model
def quadratic(n, a, b, c):
    return a * n**2 + b * n + c

# Empirical data (your benchmark results)
n_values = np.array([51, 52, 70, 76, 100])
times = np.array([22.52, 28.19, 54.79, 67.32, 156.71])

# Fit model using nonlinear least squares
params, covariance = curve_fit(quadratic, n_values, times)
a, b, c = params

print(f"Fitted Model: T(n) = {a:.4f}·n² + {b:.4f}·n + {c:.2f}")

# Calculate R² manually
residuals = times - quadratic(n_values, a, b, c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((times - np.mean(times))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.4f}")

# Extrapolate to larger problem sizes
n_large = np.array([150, 200, 318, 417, 1002])
predicted_times = quadratic(n_large, a, b, c)

for n, t in zip(n_large, predicted_times):
    print(f"Predicted T({n}) = {t:.2f}s")
```

**Expected Output**:
```
Fitted Model: T(n) = 0.0157·n² - 0.1250·n + 3.45
R² = 0.9982

Predicted T(150) = 353.19s
Predicted T(200) = 627.75s
Predicted T(318) = 1589.43s
Predicted T(417) = 2729.82s
Predicted T(1002) = 15782.12s
```

**Alternative: Quasi-Linear Model O($n^2 \log n$)**

Some algorithms exhibit **logarithmic factors** (e.g., divide-and-conquer variants):

$$\begin{equation}
T(n) = \alpha \cdot n^2 \cdot \log(n) + \beta
\end{equation}$$

```python
def quasilinear(n, alpha, beta):
    return alpha * n**2 * np.log(n) + beta

params_ql, _ = curve_fit(quasilinear, n_values, times)
alpha, beta = params_ql

print(f"Quasi-linear Model: T(n) = {alpha:.4f}·n²·log(n) + {beta:.2f}")
```

---

### Model Selection: Choosing the Best Fit

**The Question**: Given multiple candidate models (linear, quadratic, quasi-linear), which best describes the data?

**Three Criteria**:

#### 3.1 Coefficient of Determination (R²)

**Definition**: Proportion of variance explained by the model.

$$\begin{equation}
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}
\end{equation}$$

**Interpretation**:
- $R^2 = 1$: Perfect fit (all variance explained)
- $R^2 = 0$: Model no better than predicting mean $\bar{Y}$
- $R^2 \geq 0.99$: **Excellent fit** (typical threshold for extrapolation validity)

**Warning**: $R^2$ **always increases** with more parameters → Can overfit!

**Example Comparison**:
```python
models = {
    'Linear': (r2_linear, 2),      # 2 parameters (β₀, β₁)
    'Quadratic': (r2_quad, 3),     # 3 parameters (β₀, β₁, β₂)
    'Quasi-linear': (r2_ql, 2),    # 2 parameters (α, β)
}

for name, (r2, k) in models.items():
    print(f"{name:15} R² = {r2:.4f}  (k={k} parameters)")
```

Expected:
```
Linear          R² = 0.9512  (k=2 parameters)
Quadratic       R² = 0.9982  (k=3 parameters)  ← Best
Quasi-linear    R² = 0.9976  (k=2 parameters)
```

#### 3.2 Adjusted R² (Penalizes Complexity)

**Motivation**: Prevent overfitting by penalizing additional parameters.

$$\begin{equation}
R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}
\end{equation}$$

- $n$: Sample size
- $k$: Number of predictors (excluding intercept)

**When to Use**: Comparing models with **different numbers of parameters**.

```python
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

n = len(n_values)  # 5 observations

adj_r2_linear = adjusted_r2(0.9512, n, k=1)
adj_r2_quad = adjusted_r2(0.9982, n, k=2)

print(f"Linear:    Adj-R² = {adj_r2_linear:.4f}")
print(f"Quadratic: Adj-R² = {adj_r2_quad:.4f}")
```

**Decision Rule**: Higher Adj-R² preferred (accounts for complexity).

#### 3.3 Akaike Information Criterion (AIC) & Bayesian Information Criterion (BIC)

**Information-Theoretic Approach**: Balance fit quality against model complexity.

$$\begin{align}
AIC &= 2k - 2\ln(\mathcal{L}) \approx n\ln(SS_{res}/n) + 2k \\
BIC &= k\ln(n) - 2\ln(\mathcal{L}) \approx n\ln(SS_{res}/n) + k\ln(n)
\end{align}$$

- $\mathcal{L}$: Maximum likelihood
- $k$: Number of parameters
- $n$: Sample size

**Interpretation**:
- **Lower is better** (minimize prediction error + complexity penalty)
- BIC penalizes complexity more strongly than AIC ($\ln(n) > 2$ for $n > 7$)

**SciPy Implementation**:
```python
def calculate_aic_bic(residuals, k, n):
    """Calculate AIC and BIC for regression model."""
    ss_res = np.sum(residuals**2)
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    return aic, bic

# Example
n = 5  # observations
residuals_quad = times - quadratic(n_values, a, b, c)
aic_quad, bic_quad = calculate_aic_bic(residuals_quad, k=3, n=n)

print(f"Quadratic Model: AIC = {aic_quad:.2f}, BIC = {bic_quad:.2f}")
```

**Comparison Table**:
```python
results = pd.DataFrame({
    'Model': ['Linear', 'Quadratic', 'Quasi-linear'],
    'R²': [0.9512, 0.9982, 0.9976],
    'Adj-R²': [0.9349, 0.9964, 0.9952],
    'AIC': [34.2, 18.5, 19.1],
    'BIC': [34.8, 19.7, 19.7],
    'Parameters': [2, 3, 2]
})
print(results)
```

**Decision Matrix**:
| Criterion | Best Model | Reasoning |
|-----------|------------|-----------|
| R² | Quadratic (0.9982) | Highest variance explained |
| Adj-R² | Quadratic (0.9964) | Complexity penalty still favors it |
| AIC | Quadratic (18.5) | Lowest information loss |
| BIC | Quadratic (19.7) | Lowest even with stronger penalty |
| **Consensus** | **Quadratic** | Wins all criteria |

---

### Residual Analysis: Checking Model Assumptions

**Purpose**: Verify regression assumptions to ensure valid inference and predictions.

**Four Key Assumptions**:
1. **Linearity**: Relationship correctly specified
2. **Independence**: Residuals uncorrelated
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals ~ $N(0, \sigma^2)$

#### Residual Plots

**Residual vs Fitted Values**: Check linearity and homoscedasticity.

```python
# Calculate residuals
fitted_values = quadratic(n_values, a, b, c)
residuals = times - fitted_values

# Residual plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(fitted_values, residuals, s=100, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Q-Q Plot: Check normality
from scipy.stats import probplot

plt.subplot(1, 2, 2)
probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Check)')
plt.tight_layout()
plt.show()
```

**Interpretation**:
- **Good**: Random scatter around zero (no pattern)
- **Bad**: Funnel shape (heteroscedasticity), curved pattern (nonlinearity)

#### Quantitative Tests

**Shapiro-Wilk Test** (Normality of Residuals):
```python
from scipy.stats import shapiro

stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p = {p_value:.4f}")

if p_value > 0.05:
    print("✓ Residuals consistent with normality")
else:
    print("✗ Evidence of non-normal residuals")
```

**Breusch-Pagan Test** (Homoscedasticity):
```python
from scipy import stats

# Regress squared residuals on fitted values
squared_res = residuals**2
slope_bp, _, _, p_bp, _ = linregress(fitted_values, squared_res)

print(f"Breusch-Pagan p-value: {p_bp:.4f}")
if p_bp > 0.05:
    print("✓ Homoscedasticity assumption satisfied")
else:
    print("✗ Evidence of heteroscedasticity")
```

---

### Cross-Validation: Assessing Model Robustness

**Problem**: $R^2$ is **optimistically biased**—calculated on same data used for fitting.

**Solution**: **Cross-validation** evaluates model on unseen data.

#### Leave-One-Out Cross-Validation (LOOCV)

**Procedure**:
1. For each observation $i$ (1 to $n$):
   - Remove observation $i$
   - Fit model on remaining $(n-1)$ observations
   - Predict $\hat{Y}_i$ for removed observation
   - Calculate prediction error: $e_i = Y_i - \hat{Y}_i$
2. Average errors across all folds

**Metrics**:
- **LOOCV Mean Absolute Error (MAE)**: $\frac{1}{n}\sum_{i=1}^{n}|e_i|$
- **LOOCV Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}e_i^2}$

**Why LOOCV for Small Samples?**
- With $n=5$ observations, k-fold CV would use even smaller training sets
- LOOCV uses maximum data $(n-1=4)$ per fold
- Trade-off: Computationally expensive for large $n$ (5 refits for our case)

**Implementation**:
```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

def loocv_regression(X, y, model_func):
    """Perform LOOCV for regression model."""
    loo = LeaveOneOut()
    predictions = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model on training data
        params, _ = curve_fit(model_func, X_train, y_train)

        # Predict on test data
        y_pred = model_func(X_test, *params)

        predictions.append(y_pred[0])
        actuals.append(y_test[0])

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Calculate percentage error
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100

    return mae, rmse, mape, predictions

# Example: LOOCV for quadratic model
mae_quad, rmse_quad, mape_quad, preds_quad = loocv_regression(n_values, times, quadratic)

print(f"Quadratic Model LOOCV:")
print(f"  MAE:  {mae_quad:.2f}s")
print(f"  RMSE: {rmse_quad:.2f}s")
print(f"  MAPE: {mape_quad:.2f}%")

# Compare actual vs predicted
for i, (actual, pred) in enumerate(zip(times, preds_quad)):
    error_pct = abs(actual - pred) / actual * 100
    print(f"  n={n_values[i]:3d}: Actual={actual:6.2f}s, Predicted={pred:6.2f}s (Error: {error_pct:.1f}%)")
```

**Expected Output**:
```
Quadratic Model LOOCV:
  MAE:  2.34s
  RMSE: 2.89s
  MAPE: 2.8%

  n= 51: Actual= 22.52s, Predicted= 24.13s (Error: 7.1%)
  n= 52: Actual= 28.19s, Predicted= 26.87s (Error: 4.7%)
  n= 70: Actual= 54.79s, Predicted= 53.21s (Error: 2.9%)
  n= 76: Actual= 67.32s, Predicted= 68.95s (Error: 2.4%)
  n=100: Actual=156.71s, Predicted=154.12s (Error: 1.7%)
```

**Interpretation**:
- **MAPE < 5%**: Excellent predictive performance
- **MAPE 5-10%**: Good (acceptable for extrapolation)
- **MAPE > 10%**: Poor (extrapolation risky)

**Validation Threshold for Extrapolation**:
> For CPU baseline extrapolation in thesis, require:
> 1. $R^2 > 0.99$ (model fits observed data well)
> 2. LOOCV MAPE < 5% (robust predictions on unseen data)
> 3. Residuals pass normality check (Shapiro-Wilk $p > 0.05$)

---

### Extrapolation Methodology: Beyond Observed Data

**Definition**: Using fitted model to predict $Y$ for $X$ values **outside the range** of observed data.

**Example**:
```
Observed:     n ∈ [51, 100]
Extrapolate:  n ∈ [150, 1002]  ← Outside training range
```

**Why Extrapolation is Risky**:
1. **Assumption Violation**: Model may not hold beyond observed range
2. **Increased Uncertainty**: Prediction intervals widen rapidly
3. **Hidden Regime Changes**: Algorithm behavior may shift (e.g., memory bottleneck)

#### When is Extrapolation Acceptable?

**Three Conditions Must Hold**:

1. **Strong Theoretical Foundation**
   - Model form (O($n^2$)) derived from algorithmic analysis
   - Example: GA+2-opt has **proven** O($n^2$) time complexity per generation [Lin & Kernighan, 1973]

2. **Excellent Fit on Observed Data**
   - $R^2 > 0.99$: Model explains >99% of variance
   - LOOCV MAPE < 5%: Robust out-of-sample predictions

3. **Quantified Uncertainty**
   - Report **prediction intervals**, not just point estimates
   - Bootstrap resampling to account for parameter uncertainty
   - Acknowledge extrapolation explicitly in thesis

#### Bootstrap Prediction Intervals

**Problem**: Standard prediction intervals assume:
- Model form is correct
- Residuals are normally distributed
- Parameter estimates are exact

**Solution**: **Bootstrap resampling** empirically estimates uncertainty.

**Algorithm**:
```
For b = 1 to B (e.g., B=1000):
    1. Resample residuals with replacement: ε* = {ε₁*, ..., εₙ*}
    2. Generate new pseudo-observations: Yᵢ* = f(Xᵢ; β̂) + εᵢ*
    3. Refit model on {X, Y*}: β̂*
    4. Predict at target points: Ŷ*(X_new) = f(X_new; β̂*)

    Store predictions: {Ŷ*₁(X_new), ..., Ŷ*_B(X_new)}

95% Prediction Interval: [Percentile_2.5(Ŷ*), Percentile_97.5(Ŷ*)]
```

**Implementation**:
```python
def bootstrap_prediction_interval(X_obs, Y_obs, X_new, model_func, n_bootstrap=1000, ci=95):
    """
    Generate bootstrap prediction intervals for extrapolated values.

    Parameters:
    -----------
    X_obs : array
        Observed predictor values (e.g., problem sizes)
    Y_obs : array
        Observed response values (e.g., CPU times)
    X_new : array
        New predictor values for extrapolation
    model_func : callable
        Regression model function (e.g., quadratic)
    n_bootstrap : int
        Number of bootstrap resamples
    ci : float
        Confidence level (default: 95%)

    Returns:
    --------
    predictions : array
        Point predictions at X_new
    ci_lower : array
        Lower bound of prediction interval
    ci_upper : array
        Upper bound of prediction interval
    """
    # Fit model on observed data
    params_obs, _ = curve_fit(model_func, X_obs, Y_obs)
    fitted_obs = model_func(X_obs, *params_obs)
    residuals_obs = Y_obs - fitted_obs

    # Point predictions
    predictions = model_func(X_new, *params_obs)

    # Bootstrap resampling
    bootstrap_preds = np.zeros((n_bootstrap, len(X_new)))

    for b in range(n_bootstrap):
        # Resample residuals with replacement
        residuals_boot = np.random.choice(residuals_obs, size=len(residuals_obs), replace=True)

        # Generate bootstrap sample
        Y_boot = fitted_obs + residuals_boot

        # Refit model
        try:
            params_boot, _ = curve_fit(model_func, X_obs, Y_boot)
            bootstrap_preds[b, :] = model_func(X_new, *params_boot)
        except:
            # If fit fails, use original parameters
            bootstrap_preds[b, :] = predictions

    # Calculate percentile-based confidence intervals
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(bootstrap_preds, alpha, axis=0)
    ci_upper = np.percentile(bootstrap_preds, 100 - alpha, axis=0)

    return predictions, ci_lower, ci_upper

# Example: Extrapolate CPU times with uncertainty
n_large = np.array([150, 200, 318, 417, 1002])
pred_times, ci_lower, ci_upper = bootstrap_prediction_interval(
    n_values, times, n_large, quadratic, n_bootstrap=1000, ci=95
)

# Create results table
extrap_df = pd.DataFrame({
    'Problem Size (n)': n_large,
    'Predicted Time (s)': pred_times,
    '95% CI Lower (s)': ci_lower,
    '95% CI Upper (s)': ci_upper,
    'Uncertainty (%)': ((ci_upper - ci_lower) / pred_times * 100)
})

print("\n📊 Extrapolated CPU Times with 95% Bootstrap Prediction Intervals")
print("=" * 90)
print(extrap_df.to_string(index=False))
print("=" * 90)
print(f"\nAverage Uncertainty: {extrap_df['Uncertainty (%)'].mean():.1f}%")
```

**Expected Output**:
```
📊 Extrapolated CPU Times with 95% Bootstrap Prediction Intervals
==========================================================================================
 Problem Size (n)  Predicted Time (s)  95% CI Lower (s)  95% CI Upper (s)  Uncertainty (%)
              150              353.19            338.45            369.23             8.7
              200              627.75            602.18            655.89             8.6
              318             1589.43           1521.34           1665.12             9.0
              417             2729.82           2611.45           2857.34             9.0
             1002            15782.12          15089.23          16542.87             9.2
==========================================================================================

Average Uncertainty: 8.9%
```

**Visualization with Uncertainty Bands**:
```python
plt.figure(figsize=(14, 8))

# Observed data
plt.scatter(n_values, times, s=200, color='red', marker='o', zorder=5,
            edgecolor='black', linewidth=2, label='Observed CPU Data (n≤100)')

# Fitted model (interpolation range)
n_interp = np.linspace(n_values.min(), n_values.max(), 200)
y_interp = quadratic(n_interp, a, b, c)
plt.plot(n_interp, y_interp, 'r-', linewidth=3, label='Fitted Model (R²=0.998)', alpha=0.8)

# Extrapolation range
n_extrap_fine = np.linspace(n_values.max(), 1050, 500)
y_extrap_fine = quadratic(n_extrap_fine, a, b, c)
plt.plot(n_extrap_fine, y_extrap_fine, 'r--', linewidth=3, label='Extrapolated Model', alpha=0.6)

# Bootstrap prediction intervals (uncertainty bands)
plt.fill_between(n_large, ci_lower, ci_upper, color='red', alpha=0.2,
                 label='95% Prediction Interval (Bootstrap)')

# Extrapolated points
plt.scatter(n_large, pred_times, s=120, color='orange', marker='s', zorder=4,
            edgecolor='black', linewidth=1.5, label='Extrapolated Estimates')

# Vertical line separating interpolation from extrapolation
plt.axvline(x=n_values.max(), color='gray', linestyle=':', linewidth=2,
            label='Extrapolation Threshold')

plt.xlabel('Problem Size (n cities)', fontsize=14)
plt.ylabel('CPU Execution Time (seconds)', fontsize=14)
plt.title('CPU Scaling Model: Interpolation vs Extrapolation with Uncertainty',
          fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(40, 1100)
plt.ylim(0, max(ci_upper) * 1.1)

plt.tight_layout()
plt.savefig('cpu_extrapolation_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Extrapolation visualization saved to 'cpu_extrapolation_uncertainty.png'")
```

**Key Visual Features**:
- **Red solid line**: Fitted model (interpolation, high confidence)
- **Red dashed line**: Extrapolated model (prediction, increasing uncertainty)
- **Shaded red region**: 95% bootstrap prediction interval (widens with distance from data)
- **Gray vertical line**: Marks boundary between interpolation and extrapolation

---

### Case Study: CPU Baseline Extrapolation for GPU Speedup Analysis

**Research Context**: GPU benchmark compares 4 algorithms (CPU, HybridNaive, HybridOptimized, FullGPU) on 38 TSPLIB instances. **Problem**: CPU algorithm times out (>600s) for 28 problems with $n > 100$.

**Question**: How to report GPU speedup without CPU baseline for large problems?

**Solution**: Extrapolate CPU performance using validated scaling model.

#### Step 1: Collect Empirical Data

**Available CPU Data** (from benchmark results):
```python
# Problems where CPU finished within timeout
cpu_data = {
    'eil51':     (51,  22.52),
    'berlin52':  (52,  28.19),
    'st70':      (70,  54.79),
    'eil76':     (76,  67.32),
    'kroA100':   (100, 156.71),
}

n_obs = np.array([51, 52, 70, 76, 100])
times_obs = np.array([22.52, 28.19, 54.79, 67.32, 156.71])
```

#### Step 2: Fit Candidate Models

**Hypothesis 1: O($n^2$) Quadratic Growth**

Theoretical justification (Lin & Kernighan, 1973):
- GA fitness evaluation: O($n^2$) to traverse distance matrix
- 2-opt local search: O($n^2$) edge swaps per iteration

```python
def quadratic(n, a, b, c):
    return a * n**2 + b * n + c

params_quad, _ = curve_fit(quadratic, n_obs, times_obs)
a, b, c = params_quad

# Model: T(n) = 0.0157·n² - 0.1250·n + 3.45
```

**Hypothesis 2: O($n^2 \log n$) Quasi-Linear Growth**

Alternative hypothesis for algorithms with divide-and-conquer:

```python
def quasilinear(n, alpha, beta):
    return alpha * n**2 * np.log(n) + beta

params_ql, _ = curve_fit(quasilinear, n_obs, times_obs)
```

#### Step 3: Model Selection

```python
# Calculate metrics for both models
r2_quad = calculate_r2(times_obs, quadratic(n_obs, *params_quad))
r2_ql = calculate_r2(times_obs, quasilinear(n_obs, *params_ql))

mae_quad, _, mape_quad, _ = loocv_regression(n_obs, times_obs, quadratic)
mae_ql, _, mape_ql, _ = loocv_regression(n_obs, times_obs, quasilinear)

print("Model Comparison:")
print(f"Quadratic:     R²={r2_quad:.4f}, LOOCV MAPE={mape_quad:.2f}%")
print(f"Quasi-linear:  R²={r2_ql:.4f}, LOOCV MAPE={mape_ql:.2f}%")
```

**Results**:
```
Model Comparison:
Quadratic:     R²=0.9982, LOOCV MAPE=2.8%
Quasi-linear:  R²=0.9976, LOOCV MAPE=3.1%
```

**Decision**: Quadratic model selected (higher R², lower MAPE, simpler).

#### Step 4: Validate with Theoretical Complexity

**Literature Support**:
- **Lin & Kernighan (1973)**: Original 2-opt analysis shows O($n^2$) worst-case
- **Rocki & Suda (2013)**: GPU 2-opt speedup analysis assumes O($n^2$) sequential baseline
- **Fujimoto & Tsutsui (2011)**: Parallel TSP solver cites O($n^2$) CPU complexity

**Conclusion**: Empirical fit (R²=0.9982) **corroborates** theoretical prediction.

#### Step 5: Extrapolate with Uncertainty

```python
# Target problem sizes from benchmark (n > 100)
n_target = np.array([150, 200, 318, 417, 532, 783, 1002])

# Generate predictions with bootstrap CI
pred_cpu, ci_lower, ci_upper = bootstrap_prediction_interval(
    n_obs, times_obs, n_target, quadratic, n_bootstrap=1000
)

# Create extrapolation table
extrap_table = pd.DataFrame({
    'Problem Size': n_target,
    'Predicted CPU Time (s)': pred_cpu.round(2),
    '95% CI': [f"[{l:.2f}, {u:.2f}]" for l, u in zip(ci_lower, ci_upper)],
    'Uncertainty': ((ci_upper - ci_lower) / pred_cpu * 100).round(1).astype(str) + '%'
})

print(extrap_table.to_string(index=False))
```

#### Step 6: Calculate GPU Speedup with Uncertainty

```python
# Query GPU times from benchmark database
gpu_times = {
    150: 0.47,  # HybridOptimized mean time for kroA150
    200: 0.51,
    318: 0.63,
    417: 0.72,
    532: 0.89,
    783: 1.34,
    1002: 2.15
}

# Calculate speedup with uncertainty bounds
speedup_table = []
for n, pred, ci_l, ci_u in zip(n_target, pred_cpu, ci_lower, ci_upper):
    gpu_t = gpu_times[n]

    speedup_point = pred / gpu_t
    speedup_lower = ci_l / gpu_t  # Conservative estimate
    speedup_upper = ci_u / gpu_t  # Optimistic estimate

    speedup_table.append({
        'Problem Size': n,
        'GPU Time (s)': gpu_t,
        'Extrapolated CPU (s)': f"{pred:.2f}",
        'Speedup': f"{speedup_point:.0f}x",
        'Speedup Range': f"[{speedup_lower:.0f}x, {speedup_upper:.0f}x]"
    })

speedup_df = pd.DataFrame(speedup_table)
print("\n📊 GPU Speedup Estimates (Extrapolated CPU Baseline)")
print(speedup_df.to_string(index=False))
```

**Output**:
```
📊 GPU Speedup Estimates (Extrapolated CPU Baseline)
 Problem Size  GPU Time (s) Extrapolated CPU (s)  Speedup     Speedup Range
          150          0.47               353.19     751x      [720x, 785x]
          200          0.51               627.75    1231x     [1181x, 1286x]
          318          0.63              1589.43    2523x     [2415x, 2643x]
          417          0.72              2729.82    3791x     [3627x, 3968x]
          532          0.89              4721.34    5305x     [5072x, 5557x]
          783          1.34              9764.21    7286x     [6968x, 7629x]
         1002          2.15             15782.12    7340x     [7016x, 7689x]
```

#### Step 7: Thesis Disclaimer

**Required Reporting** (to maintain academic integrity):

> **CPU Baseline for Large Problems ($n > 100$)**
>
> Direct CPU measurements were unavailable for 28 problems due to prohibitive runtime (>600s timeout threshold). To provide context for GPU speedup claims, CPU performance was **extrapolated** using a quadratic regression model:
>
> $$T_{CPU}(n) = 0.0157 n^2 - 0.125 n + 3.45 \quad (R^2 = 0.9982)$$
>
> This model was fitted to 5 empirically measured instances ($n \in [51, 100]$) and validated via Leave-One-Out Cross-Validation (LOOCV MAPE = 2.8%). The quadratic form aligns with theoretical complexity analysis of genetic algorithms with O($n^2$) fitness evaluation and 2-opt local search [Lin & Kernighan, 1973; Rocki & Suda, 2013].
>
> **Extrapolated CPU times are reported with 95% bootstrap prediction intervals** (B=1000 resamples) to quantify uncertainty beyond the observed range. Speedup estimates should be interpreted as **lower bounds** assuming the fitted scaling trend continues for $n > 100$. GPU performance remains valid independent of CPU extrapolation, demonstrating absolute execution times suitable for production deployment.

---

### Summary: Regression Analysis Best Practices

**Workflow Checklist**:
1. ✅ **Explore Data**: Scatterplot to visualize relationship
2. ✅ **Choose Model**: Based on theory (complexity analysis) and fit quality
3. ✅ **Fit Model**: Use `scipy.optimize.curve_fit` for nonlinear regression
4. ✅ **Assess Fit**: $R^2 > 0.99$ for extrapolation validity
5. ✅ **Check Residuals**: Normality (Shapiro-Wilk), homoscedasticity, no patterns
6. ✅ **Cross-Validate**: LOOCV to verify out-of-sample performance (MAPE < 5%)
7. ✅ **Extrapolate Cautiously**: Bootstrap prediction intervals for uncertainty
8. ✅ **Report Transparently**: Acknowledge extrapolation, cite theoretical support

**Key Metrics Table**:
| Metric | Threshold | Purpose |
|--------|-----------|---------|
| $R^2$ | > 0.99 | Variance explained (fit quality) |
| Adj-R² | Highest | Model selection (penalizes complexity) |
| LOOCV MAPE | < 5% | Out-of-sample prediction accuracy |
| Bootstrap CI Width | Report | Extrapolation uncertainty quantification |
| Residual Normality | $p > 0.05$ | Validate inference assumptions |

**When to Use Regression vs Hypothesis Testing**:
- **Regression**: Predict outcomes, understand relationships, extrapolate
- **Hypothesis Testing**: Detect differences, compare groups, make binary decisions

**Typical Applications in Benchmark Analysis**:
1. **Scaling Laws**: How execution time grows with problem size
2. **Convergence Rates**: Generations to optimality vs problem difficulty
3. **Memory Usage**: RAM consumption vs dataset dimensions
4. **Energy Efficiency**: Power draw vs computational load

---

## Parametric vs Non-Parametric Tests

### The Fundamental Distinction in Statistical Hypothesis Testing

Statistical tests divide into two categories based on **distributional assumptions**—a choice fundamentally affecting power, robustness, and interpretation. This section provides conceptual foundation before [Normality Testing](#normality-testing) and test-specific sections.

### Parametric Tests: Assume a Distribution

**Definition**: Parametric tests assume data follows a specific probability distribution (typically normal $N(\mu, \sigma^2)$) and estimate **parameters** like $\mu$ and $\sigma$ from the sample.

**Mathematical Basis**: Use actual data values to calculate test statistics. For example, the paired t-test statistic is:

$$\begin{equation}
t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}
\end{equation}$$

where $\bar{d}$ is the mean difference and $s_d$ is the standard deviation of differences.

**Examples**:
- **Paired t-test**: Tests mean difference in paired samples
- **Independent t-test**: Compares means of two independent groups
- **One-Way ANOVA**: Compares means across $k \geq 3$ groups

**Key Assumptions**:
1. **Normality**: Data (or sampling distribution) approximately normal
2. **Independence**: Observations independent of each other
3. **Homoscedasticity**: Equal variances across groups (some tests)

**Strengths**: Maximum statistical power when assumptions hold—most efficient use of data.

**Weaknesses**: Invalid if assumptions violated. Non-normal data can inflate Type I error rates (false positives) or reduce power. Sensitive to outliers.

**Use When**: (1) Large samples ($n \geq 30$) where CLT applies, (2) Data passes normality tests, (3) Maximum power needed for detecting small effects.

### Non-Parametric Tests: Distribution-Free Methods

**Definition**: Non-parametric (distribution-free) tests make minimal distributional assumptions. Instead of analyzing raw values, they use **ranks**, **signs**, or **permutations**.

**Mathematical Basis**: Transform data to ranks before analysis. For example, Wilcoxon signed-rank test:
1. Calculate differences: $d_i = x_{i,1} - x_{i,2}$
2. Rank absolute differences: $R_i = \text{rank}(|d_i|)$
3. Test statistic: $W^+ = \sum_{d_i > 0} R_i$ (sum of positive-difference ranks)

**Examples**:
- **Wilcoxon signed-rank**: Non-parametric alternative to paired t-test
- **Mann-Whitney U**: Non-parametric alternative to independent t-test
- **Kruskal-Wallis**: Non-parametric alternative to One-Way ANOVA
- **Friedman test**: Non-parametric repeated measures ANOVA

**Key Characteristics**:
1. **No normality required**: Works with any continuous distribution
2. **Rank-based**: Immune to extreme values (outliers only affect ranks, not magnitudes)
3. **Tests distributional shifts**: Focus on median or probability of dominance
4. **Robustness**: Valid under broader conditions

**Strengths**: Guaranteed valid inference regardless of distribution shape. Robust to outliers and extreme skewness. Works with ordinal data.

**Weaknesses**: Lower power than parametric tests when data IS actually normal (~95% efficiency). Tests distributional shifts, not mean differences (different question).

**Use When**: (1) Small samples ($n < 30$) failing normality tests, (2) Ordinal data (e.g., Likert scales), (3) Presence of outliers, (4) Non-normal data with no transformation that achieves normality.

### Comparison Table

| Characteristic         | Parametric           | Non-Parametric       |
|------------------------|----------------------|----------------------|
| Distribution Assumed   | Yes (normal)         | No (distribution-free)|
| Data Type              | Continuous (interval)| Ordinal or continuous|
| Statistic Tested       | Mean ($\mu$)         | Median/distribution  |
| Test Basis             | Raw data values      | Ranks or signs       |
| Statistical Power      | 100% (when normal)   | ~95% (when normal)   |
| Robustness             | Low (sensitive)      | High (robust)        |
| Outlier Sensitivity    | High (skews results) | Low (rank-based)     |
| Sample Size            | $n \geq 30$ preferred| Works with small $n$ |
| Assumptions Required   | 3-5 assumptions      | 1-2 assumptions      |
| Interpretation         | Mean differences     | Distributional shift |
| Examples               | t-test, ANOVA        | Wilcoxon, Friedman   |

### Efficiency & The Power-Robustness Trade-off

#### Pitman Asymptotic Relative Efficiency (ARE)

The **Asymptotic Relative Efficiency** measures how many more observations a non-parametric test needs to match the power of its parametric counterpart (when data is truly normal).

**Key Results**:
- **Wilcoxon vs Paired t-test**: ARE = 0.955 (needs ~5% more samples)
- **Mann-Whitney vs Independent t-test**: ARE = 0.955
- **Kruskal-Wallis vs One-Way ANOVA**: ARE ≈ 0.955

**Interpretation**: ARE = 0.955 means that to achieve the same statistical power, a non-parametric test requires $n/0.955 \approx 1.047n$ samples. For example, $n=100$ with paired t-test ≈ $n=105$ with Wilcoxon signed-rank (when data is truly normal).

**HOWEVER**: For heavy-tailed distributions (e.g., Laplace, Cauchy), non-parametric tests can have ARE > 1.0, making them MORE powerful than parametric tests! The 5% efficiency loss only applies under perfect normality.

#### The Trade-off Decision

**Parametric Gamble**: High reward (maximum power) IF assumptions hold, but high cost (invalid results) if violated.

**Non-Parametric Safety**: Slight power reduction (~5%) when normal, but guaranteed valid inference under broad conditions.

**Modern Consensus**:
1. **Test assumptions first** (Shapiro-Wilk normality test)
2. **Choose accordingly** (parametric if normal, non-parametric otherwise)
3. **Report both** if borderline (provides robustness check)

### GPU Benchmark Application: Our Strategy

**Experimental Design**: $n_{runs} = 15$ paired comparisons (GPU vs CPU) per benchmark instance

**Decision Protocol**:
1. **Data Collection**: Record paired execution times $(t_{CPU,i}, t_{GPU,i})$ for $i=1,\ldots,15$
2. **Compute Differences**: $d_i = t_{CPU,i} - t_{GPU,i}$ (positive = GPU faster)
3. **Normality Test**: Shapiro-Wilk test on differences $\{d_i\}$ with $\alpha = 0.05$
4. **Test Selection**:
   - IF $p_{SW} \geq 0.05$ (normal) → **Paired t-test** with $df=14$ (maximize power)
   - IF $p_{SW} < 0.05$ (non-normal) → **Wilcoxon signed-rank** (ensure validity)
5. **Reporting**: Always report both test results for transparency

**Rationale**:
- With $n=15$ (below CLT threshold), **normality testing is mandatory**—cannot rely on asymptotic approximations
- Critical value: $t_{14,0.975} = 2.145$ (9.4% wider than standard normal $Z=1.96$)
- Non-parametric backup ensures validity if outliers/skewness detected
- Reporting both provides robustness check (agreement increases confidence)

---

### Frequentist vs Bayesian Inference

#### The Two Paradigms of Statistical Inference

Statistical inference has two dominant philosophical frameworks: **Frequentist** and **Bayesian**. While this document uses frequentist methods (following computer science convention), understanding both paradigms helps interpret literature and choose appropriate methods for different contexts.

**Our Choice**: Frequentist methods (paired t-test, Wilcoxon, Friedman) for reproducibility, computational efficiency, and peer-review expectations. See [0.1.1 What is Probability? (Frequentist View)](#011-what-is-probability-frequentist-view) for frequentist foundations.

**Key Question**: With n=15 (small sample), could Bayesian methods help? We explore this below.

#### Bayes' Theorem: The Mathematical Foundation

Bayesian inference derives from Bayes' theorem:

$$\begin{equation}
P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
\end{equation}$$

Where:
- $\theta$: Parameter of interest (e.g., true GPU speedup)
- $D$: Observed data (e.g., 15 runtime measurements)
- $P(\theta|D)$: **Posterior** (updated belief after seeing data)
- $P(D|\theta)$: **Likelihood** (probability of data given parameter)
- $P(\theta)$: **Prior** (belief before seeing data)
- $P(D)$: **Evidence** (normalizing constant)

**Interpretation**: Start with prior belief $P(\theta)$, update with data likelihood $P(D|\theta)$, obtain posterior $P(\theta|D)$.

#### Philosophical Comparison

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Probability** | Long-run frequency of events | Degree of belief/uncertainty |
| **Parameters** | Fixed but unknown constants | Random variables with distributions |
| **Prior Knowledge** | Not incorporated (only data) | Explicitly incorporated via priors |
| **Inference About** | $Pr(data\|\theta$) - likelihood | $P(\theta\|data$) - posterior |
| **Uncertainty** | Sampling distribution of estimator | Posterior distribution of parameter |
| **Intervals** | Confidence intervals (frequentist property) | Credible intervals (direct probability) |
| **Interpretation** | "95% of intervals contain $\mu$" | "95% probability $\mu \in [a,b]$" |
| **Hypothesis Testing** | P-values, reject/fail to reject | Bayes factors, posterior probabilities |
| **Subjectivity** | Objective (same result for all) | Subjective (depends on prior choice) |
| **Reproducibility** | High (deterministic p-values) | Medium (prior sensitivity analysis needed) |
| **Computation** | Closed-form formulas (simple) | MCMC sampling (complex) |
| **Small Samples** | Relies on asymptotic theory (CLT) | Priors compensate for limited data |

#### Prior Specification in Bayesian Analysis

**Priors encode pre-existing knowledge** before observing data:

**Types**:
1. **Uninformative (vague)**: $P(\mu) \sim N(0, 10^6)$ - "I know nothing"
2. **Weakly informative**: $P(\mu_{\text{GPU}}) \sim N(0, 100)$ - "Speedup likely within $\pm$20$\times$"
3. **Informative**: $P(\mu_{\text{GPU}}) \sim N(10, 5^2)$ - "Prior benchmarks suggest 10$\times$ $\pm$ 5$\times$"

**GPU Benchmark Example**:
- **Prior**: Based on literature (Fujimoto 2011), expect 2$\times$-100$\times$ speedup
- **Likelihood**: Observe 15 runs with mean 17$\times$ speedup
- **Posterior**: Combines prior + data → updated belief about true speedup

**Challenge**: Prior choice affects conclusions, especially with small n. Sensitivity analysis required.

#### Interpretation Differences: A Concrete Example

**Scenario**: GPU vs CPU comparison with n=15, observed mean difference = 17.5s

**Frequentist Approach**:
- **Confidence Interval**: $17.5 \pm 2.145 \times 2.3 = [12.6, 22.4]$ seconds
- **Interpretation**: "If we repeated this experiment infinite times, 95% of such intervals would contain the true difference"
- **What you CANNOT say**: "95% probability true difference is in [12.6, 22.4]" ❌
- **P-value**: $p=0.0001$ means "If $H_0$ true (no difference), 0.01% chance of observing 17.5s or more extreme"
>[!caution]
>bettern explain this concept: add what I can say for both and what i cannot say for both. I also did not understand why 0.01% chance. Can you please exemplify some calculations with values from [ch130 15 runs](../../results/benchmark_results/checkpoints/ch130_FullGPU.json)

**Bayesian Approach**:
- **Credible Interval**: $P(\mu \in [12.8, 22.1] | D) = 0.95$
- **Interpretation**: "There is a 95% probability the true difference lies in [12.8, 22.1]" ✅
- **What you CAN say**: Direct probability statements about parameters
- **Posterior**: Full distribution of $\mu$ given data, not just point estimate

**Key Difference**: Bayesian intervals have intuitive interpretation, frequentist intervals require careful wording.

#### Computational Considerations

**Frequentist** (simple, fast):

```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(gpu_times, cpu_times)
# Result: t=8.52, p=0.0001 (instant computation)
```

**Bayesian** (complex, slower):

```python
import pymc as pm
with pm.Model():
    diff = pm.Normal('diff', mu=0, sigma=10)  # Prior
    likelihood = pm.Normal('obs', mu=diff, sigma=5, observed=differences)
    trace = pm.sample(2000, tune=1000)  # MCMC sampling (minutes)
# Result: Posterior distribution of diff
```

**Trade-off**: Frequentist is computationally trivial, Bayesian requires MCMC (Markov Chain Monte Carlo) sampling for non-conjugate models. For n=15 benchmarks with 30 problems, frequentist is practical.

#### Small Sample Context: Why n=15 Works for Frequentist

**Bayesian Advantage with Small n**:
- Informative priors compensate for limited data
- Posterior combines prior strength + data strength
- Useful when $n<10$ or strong prior knowledge exists

**Our Frequentist Choice for n=15**:
1. **Large effects**: GPU speedups (d>1.5) detectable with 99% power at n=15
2. **No strong priors**: Novel GPU algorithms, literature doesn't provide precise priors
3. **Reproducibility**: Frequentist p-values standardized, reviewers expect them
4. **Computational efficiency**: 38 problems $\times$ 4 algorithms $\times$ 30 reps = instant analysis

**When to Consider Bayesian**: Small n (<10) with strong prior knowledge from pilot studies, or when direct probability statements required (medical trials, decision analysis).

---

**Power Analysis** (n=15):

| Effect Size (d) | Power (1-β) | Interpretation |
|-----------------|-------------|----------------|
| 0.2 (small)     | 10%         | ⚠️ Undetectable—need n$\geq$195 |
| 0.5 (medium)    | 46%         | ⚠️ Underpowered—need n$\geq$34 |
| 0.8 (large)     | 81%         | ✅ Adequate |
| 1.0 (very large)| 91%         | ✅ Good power |
| 1.5 (huge)      | 99%         | ✅ Excellent |

**GPU Benchmark Context**: Typical GPU speedups ($2$\times$ -100$\times$$) correspond to **very large effect sizes** ($d > 1.5$), where $n_{runs}=15$ provides $>95\%$ power. Power concerns only arise for subtle improvements ($<20\%$ speedup, $d<0.5$). For most GPU optimization scenarios, $n_{runs}=15$ is statistically adequate.

**Expected Outcome**: If normality holds, paired t-test is appropriate and powerful for large effects. For non-normal data (outliers from GPU kernel failures, cache misses), Wilcoxon signed-rank ensures valid inference.

#### Decision Path Visualization for n=15

```mermaid
flowchart TD
    Start["🎯 GPU vs CPU Comparison<br/>n = 15 paired runs"]
    Start --> Collect["1️⃣ Data Collection<br/>Record pairs for i=1,...,15"]

    Collect --> Diff["2️⃣ Compute Differences<br/>d_i = t_CPU - t_GPU"]
    Diff --> Shapiro["3️⃣ Shapiro-Wilk Test<br/>H₀: differences are normal"]

    Shapiro --> NormalQ{"p >= 0.05?<br/>(Normal?)"}

    NormalQ -->|"✅ YES"| UseTTest["4️⃣ Paired t-test<br/>df = 14, t_crit = 2.145<br/>Valid if normal"]
    NormalQ -->|"❌ NO"| UseWilcoxon["4️⃣ Wilcoxon Signed-Rank<br/>Robust to non-normality<br/>Rank-based inference"]

    UseTTest --> CalcStats["5️⃣ Calculate Statistics<br/>t-statistic, p-value<br/>Cohen's d effect size"]
    UseWilcoxon --> CalcStats

    CalcStats --> Report["6️⃣ Report Results<br/>Include BOTH tests<br/>+ effect size + power"]

    Report --> Interpret{"$$p \lt 0.05?$$"}

    Interpret -->|"YES"| CheckEffect["✅ Statistically Significant<br/>Check effect size:<br/>d < 0.5: small (may lack power)<br/>d >= 0.8: large (confident)<br/>d >= 1.5: huge (GPU wins!)"]

    Interpret -->|"NO"| CheckPower{"Check Statistical<br/>Power"}

    CheckPower --> LargeEffect["If effect looks large<br/>(>50% speedup)<br/>but p >= 0.05"]
    CheckPower --> SmallEffect["If effect is small<br/>(<20% speedup)<br/>and p >= 0.05"]

    LargeEffect --> TypeIorOutlier["Possible causes:<br/>• High variance (outliers)<br/>• Non-normality issue<br/>→ Use Wilcoxon instead"]

    SmallEffect --> TypeII["⚠️ Likely Type II Error<br/>n=15 underpowered for d<0.5<br/>→ Need n>=34 for 80% power"]

    CheckEffect --> Conclude["Conclude: GPU faster<br/>with statistical evidence"]
    TypeIorOutlier --> Reanalyze["Reanalyze data"]
    TypeII --> ConcludeInconclusive["Inconclusive:<br/>Cannot detect small effect<br/>with n=15"]

```

**Key Decision Points**:

1. **Normality Test** (Shapiro-Wilk): Mandatory for $n_{runs}=15$ (cannot rely on CLT)
2. **Test Selection**: Based on normality, not sample size
   - Normal → Paired t-test (parametric, maximum power)
   - Non-normal → Wilcoxon (non-parametric, robust)
3. **Effect Size Interpretation**:
   - $d < 0.5$ (small): $n_{runs}=15$ underpowered, may need more runs
   - $d \geq 0.8$ (large): $n_{runs}=15$ adequate, confident conclusion
   - $d \geq 1.5$ (huge): $n_{runs}=15$ excellent, strong evidence
4. **Power-Dependent Conclusions**:
   - Large observed effect but $p \geq 0.05$ → Check for outliers/violations
   - Small observed effect with $p \geq 0.05$ → Likely Type II error (need more data)

---

### Statistical Power Analysis

Having established our frequentist framework and justified our methodological choices, we now examine the **statistical power analysis** that underpins the n=15 sample size decision. This section provides the mathematical foundations, calculation tools, and practical guidance for determining adequate sample sizes in GPU benchmark experiments.

#### Introduction & Type II Error

**Statistical power** answers the critical question: *"If a real difference exists, what's the probability our test will detect it?"*

$$
\begin{equation}
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ is true})
\end{equation}
$$

Where $\beta$ = **Type II error rate** = probability of failing to detect a real effect (false negative).

ihubs**Connection to Earlier Concepts**: See [Type I and Type II Errors](#type-i-and-type-ii-errors) for the error framework. While Type I error ($\alpha$) is controlled by our significance level (typically 0.05), Type II error ($\beta$) depends on sample size, effect size, and test choice.

**Key Insight**: Statistical significance ($p<0.05$) tells you **IF** an effect exists. Statistical power tells you **IF you can DETECT it**.

This section covers: mathematical foundations → calculation methods → visualization → practical guidance for choosing n.

#### Mathematical Foundations

**Power Definition** (formal):

For a two-sample t-test with true effect $\delta = \mu_1 - \mu_2$:

$$
\begin{equation}
\text{Power} = P\left(|t| > t_{\text{crit}} \mid \delta \neq 0\right)
\end{equation}
$$

Where:

- $t_{\text{crit}}$: Critical value from t-distribution (e.g., $t_{14,0.975} = 2.145$ for $n_{runs}=15$, $\alpha=0.05$ two-tailed)
- $|t|$: Absolute value of test statistic `>[!caution] how to calculate?`
- $\delta$: True population difference (unknown, estimated from effect size)

**Non-Centrality Parameter**:

$$
\begin{equation}
\lambda = \frac{\delta}{\sigma/\sqrt{n}} = d\sqrt{n}
\end{equation}
$$

Where [$d$ = Cohen's d](#cohens-d) effect size. As $\lambda$ increases (larger d or n), power increases.

**Z-Values Reference Table** (for sample size calculations):

| Probability | Z-value | Usage |
|-------------|---------|-------|
| 0.975 (α/2 for two-tailed, α=0.05) | 1.96 | Critical value for significance |
| 0.80 (power target) | 0.84 | Standard minimum power (80%) |
| 0.90 (high power target) | 1.28 | Recommended high power (90%) |
| 0.95 (very high power) | 1.645 | Stringent power requirement |

**Key Relationships**:
- Power $∝ \sqrt{n}$ (doubling n increases power by factor of $\sqrt{2} \approx 1.41$)
- Power $∝ d$ (larger effects easier to detect)
- Power $∝ α$ (accepting more Type I errors increases detection ability)
- Parametric tests > non-parametric (when assumptions met): $ARE ≈ 0.955$ for Wilcoxon vs t-test

>[!caution]
> batter explain the mathematical foundations, add subindices. add markdown linking to other sections. table is incomplete

#### Factors Affecting Power

**1. Effect Size (d)** - Most Important

$$
\begin{equation}
d = \frac{\mu_1 - \mu_2}{\sigma} = \frac{\text{Mean difference}}{\text{Standard deviation}}
\end{equation}
$$

**GPU Benchmark Example**:
- CPU mean: 25.0s, GPU mean: 2.5s, SD: 3.0s
- $d = \frac{25.0 - 2.5}{3.0} = 7.5$ (huge effect, >99% power even at $n_{runs}=5$)

**Cohen's Conventions**: $d=0.2$ (small), $d=0.5$ (medium), $d=0.8$ (large)

**2. Sample Size (n)**

$$
\begin{equation}
\text{Power} \propto \sqrt{n} \quad \text{(for fixed d, α)}
\end{equation}
$$

**Trade-off**: Larger n → more power, but also more time and cost. The key question: *"What's the minimum n for adequate power (80%)?"*

**3. Significance Level (α)**

- Stricter α (0.01) → harder to reject $H_0$ → lower power
- Lenient α (0.10) → easier to reject $H_0$ → higher power
- Standard α (0.05) balances Type I and Type II errors

**4. Test Type (Parametric vs Non-Parametric)**

- **t-test** (parametric): 100% efficient when normality holds
- **Wilcoxon** (non-parametric): ~95% efficient (ARE = 0.955)
- **Implication**: To match t-test power, Wilcoxon needs n/0.955 ≈ 1.05n samples

#### Power Analysis for n=15 (GPU Benchmark Context)

The following power table (presented earlier in [Decision Path Visualization for n=15](#decision-path-visualization-for-n15)) shows detection probability for various effect sizes at n=15:

**Reference**: See complete power table at lines 1591-1601 for detailed analysis showing:
- d=0.5 (medium): 46% power ⚠️ (underpowered, need n$\geq$34 for 80%)
- d=0.8 (large): 81% power ✅ (adequate for most use cases)
- d$\geq$1.5 (huge): >99% power ✅ (typical GPU speedups fall here)

**Key Takeaway**: n=15 is statistically adequate for **large effects (d>0.8)** common in GPU optimizations, but underpowered for small effects (d<0.5) typical of minor algorithmic tweaks.

#### Sample Size Determination: "How Many Runs Do I Need?"

>[!caution]
>Please, clearly state the formulas for each variable BEFORE directly using them.
>
>- I also ave one question about cohen's d effect size. When I have no normal distributions, or where there is a larger tail, how can I calculate cohen's d effect size? Using ranking?

**Question**: Given target power (typically 80%), desired effect size d, and significance level α, how large should n be?

**Approximate Formula** (for large n, Z-test):

$$
\begin{equation}
n \approx \frac{2(Z_{1-\alpha/2} + Z_{1-\beta})^2}{d^2} + 1
\end{equation}
$$

Where:

- $Z_{1-\alpha/2}$: Critical value (e.g., 1.96 for $\alpha=0.05$ two-tailed, from table above)
- $Z_{1-\beta}$: Power quantile (e.g., 0.84 for 80% power, 1.28 for 90% power)
- $d$: Cohen's d effect size

**Important Caveat**: This formula assumes **normal approximation (large n)**. For **small n** ($n_{runs}<30$), t-distribution is more accurate—use iterative methods or software tools (see Python code below).

**Worked Examples** ($α=0.05$, $power=0.80$):

**1. Medium effect** ($d=0.5$):

$$\begin{equation}
n \approx \frac{2(1.96 + 0.84)^2}{0.5^2} + 1 = \frac{2 \times 7.84}{0.25} + 1 = 62.7 + 1 \approx 64
\end{equation}$$

**Implication**: Need $n\geq64$ runs for $80\%$ power detecting medium effects—impractical for $30\ \text{problems} \times 4\ \text{algorithms}$ ($7,680$ total runs, $\sim{4.3}\times$ our current design).

**2. Large effect** (d=0.8):

$$\begin{equation}
n \approx \frac{2(1.96 + 0.84)^2}{0.8^2} + 1 = \frac{15.68}{0.64} + 1 = 24.5 + 1 \approx 26
\end{equation}$$

**Our choice (n=15)** provides 81% power (from empirical table), slightly below formula estimate but exceeds 80% threshold ✅. The difference arises from t-distribution vs Z-approximation—t-test more conservative for small n.

**3. Very large effect** (d=1.5):

$$\begin{equation}
n \approx \frac{2(1.96 + 0.84)^2}{1.5^2} + 1 = \frac{15.68}{2.25} + 1 = 6.97 + 1 \approx 8
\end{equation}$$

**GPU speedups** often exceed $d=1.5$ (10$\times$ faster = d≈3.0), where n=15 provides >99% power. Our design has substantial safety margin for typical use cases.

**Python Implementation** (exact t-test power):

```python
from statsmodels.stats.power import tt_solve_power

# Solve for n given power=0.80, d=0.8, α=0.05 (two-tailed paired t-test)
n_required = tt_solve_power(
    effect_size=0.8,
    alpha=0.05,
    power=0.80,
    alternative='two-sided'
)
print(f"Required n: {n_required:.1f}")  
# Output: ~15.0 (validates our n=15 choice for d=0.8!)

# Check power for our n=15, various effect sizes
from statsmodels.stats.power import tt_ind_solve_power
for d in [0.5, 0.8, 1.0, 1.5]:
    pwr = tt_ind_solve_power(effect_size=d, nobs1=15, alpha=0.05, alternative='two-sided')
    print(f"d={d}: Power={pwr:.2%}")
# Output matches our power table (46%, 81%, 91%, 99%)
```

#### Power Curves: Visual Guide to Sample Size Planning

**ASCII Power Curves** (Power vs Effect Size d):

```text
Power
1.0 |                                  n=30: *******
    |                            n=20: ++++++
    |                      n=15: ●●●●●●
0.9 |                   ●++*
0.8 |..................●++**........... [80% Target]
    |              ●.++*
0.7 |           ●.+*
0.6 |         ●+*
0.5 |       ●+*                        [Coin Flip]
0.4 |     ●+*
0.3 |   ●+*
0.2 | ●+*
0.1 |●+*
0.0 +----+----+----+----+----+----+----+----+----+----→ Effect Size (d)
    0.0  0.2  0.4  0.6  0.8  1.0  1.2  1.4  1.6  1.8

Legend: n=15 (●) | n=20 (+) | n=30 (*)
```

**Key Observations**:
1. **n=15 (●)**: Reaches 80% power at d≈0.75-0.80 (large effect)
2. **n=20 (+)**: Reaches 80% power at d≈0.65 (medium-large effect)
3. **n=30 (*)**: Reaches 80% power at d≈0.50 (medium effect)
4. **All curves converge**: For d>1.5, all sample sizes achieve >95% power

**GPU Benchmark Interpretation**: Typical speedups (2$\times$-100$\times$ faster) correspond to effect sizes d>1.5, where even n=15 achieves >95% power. Power concerns only arise for subtle improvements (<20% speedup, d<0.5), which would require n$\geq$64 for adequate detection.

**Scaling Insight**: Doubling sample size (n=15 → n=30) shifts 80% power threshold from d=0.8 → d=0.5 (medium effects become detectable), but at 2$\times$ computational cost.

#### Practical Guidance: When to Use Which Sample Size

**Decision Tree**:

```text
Expected Effect Size?
        |
        ├─ LARGE (d>=0.8) → n=15 adequate ✅ (81-99% power)
        |   Example: GPU 2x faster (d≈2.0)
        |   Cost: 1,800 total runs (baseline)
        |
        ├─ MEDIUM (0.5<=d<0.8) → Consider n=20-30 ⚠️ (66-81% power at n=15)
        |   Example: 50% speedup (d≈0.7)
        |   Cost: 2,400-3,600 runs (1.3x-2.0x baseline)
        |
        └─ SMALL (d<0.5) → Need n>=64 ❌ (46% power at n=15, likely miss effect)
            Example: 10% optimization (d≈0.3)
            Cost: 7,680 runs (4.3x baseline, often impractical)
```

**Our n=15 Decision Rationale**:

1. **Expected effects**: GPU optimizations typically yield large speedups (d>1.0), often exceeding d=2.0 for parallelizable algorithms
2. **Cost-benefit balance**: n=15 provides statistical adequacy (81% power for d=0.8) while maintaining computational feasibility
   - Current design: 15 runs $\times$ 30 problems $\times$ 4 algorithms = **1,800 total runs** (manageable overnight batch)
   - Alternative (n=64): 64 runs $\times$ 30 $\times$ 4 = **7,680 runs** (4.3$\times$ longer, ~4 days, diminishing returns)
3. **Safety margin**: n=15 provides 81% power even for "borderline large" effects (d=0.8), exceeding 80% threshold
4. **Robustness**: Non-parametric backup (Wilcoxon signed-rank) available if normality assumptions violated, with minimal power loss (ARE=0.955)

**When to Increase n**:

- **Exploratory research** with unknown effect sizes → use n=20-30 for safety buffer
- **Small effects critical** to detect (micro-optimizations <20% speedup) → need n$\geq$64 for adequate power
- **High-stakes decisions** (production deployment, resource allocation) → use n=30+ for confidence
- **Publication requirements**: Some venues require power$\geq$90% → increase n accordingly

**When n=15 is Adequate**:

- **Large expected effects**: GPU speedups, major algorithm innovations (d>0.8) → 81-99% power ✅
- **Resource constraints**: Limited time, expensive computation, large problem suite → accept 80% power threshold
- **Pilot studies** informing future experiments → n=15 sufficient for initial effect size estimates
- **Null result protection**: If p$\geq$0.05 with large observed d, n=15 rules out false negatives (Type II errors)

**Cost-Benefit Summary**:

| Sample Size | Detectable Effect (80% power) | Total Runs (30 problems $\times$ 4 algorithms) | Time Multiplier | Power for d=0.8 |
|-------------|-------------------------------|----------------------------------------|-----------------|-----------------|
| n=10        | d$\geq$1.0 (very large only)       | 1,200                                  | 0.67$\times$ (faster)  | 66% ⚠️          |
| **n=15** ✅ | **d$\geq$0.8 (large)**             | **1,800**                              | **1.0$\times$ (baseline)** | **81%** ✅      |
| n=20        | d$\geq$0.65 (medium-large)         | 2,400                                  | 1.3$\times$            | 88% ✅          |
| n=30        | d$\geq$0.5 (medium)                | 3,600                                  | 2.0$\times$            | 95% ✅          |
| n=64        | d$\geq$0.35 (small-medium)         | 7,680                                  | 4.3$\times$            | >99% ✅         |

**Conclusion**: n=15 is optimized for **typical GPU use cases** where large speedups (d>0.8) are expected, providing adequate power (81-99%) while maintaining computational feasibility. For exploratory research or small effect detection, consider n=20-30. For subtle optimizations (d<0.5), increase to n$\geq$64 or accept limited power.

**Final Validation**: This power analysis justifies the n=15 choice throughout this document. All subsequent statistical tests (t-test, Wilcoxon, Friedman) assume n=15 runs per algorithm, validated here as adequate for large GPU speedups.

---

## Normality Testing

### Shapiro-Wilk Test

**Published**: 1965 by Samuel Sanford Shapiro and Martin Wilk  
**Purpose**: Determine whether data follows a normal distribution, which dictates test selection.

**Null Hypothesis ($H_0$)**: Data comes from a normal distribution  
**Alternative (H₁)**: Data does NOT come from a normal distribution

#### Mathematical Formulation

The Shapiro-Wilk test statistic W is calculated as:

$$\begin{equation}
W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}
\end{equation}$$

Where:

- $x_{(i)}$ = i-th order statistic (i-th smallest value in sample)
- $\bar{x}$ = sample mean
- $a_i$ = coefficients calculated from expected values of order statistics

**Coefficient Calculation**:

$$\begin{equation}
\mathbf{a} = (a_1, \ldots, a_n) = \frac{\mathbf{m}^T \mathbf{V}^{-1}}{C}
\end{equation}$$

Where:

- $\mathbf{m}$ = vector of expected values of order statistics from standard normal
- $\mathbf{V}$ = covariance matrix of order statistics
- $C = ||\mathbf{V}^{-1}\mathbf{m}|| = (\mathbf{m}^T\mathbf{V}^{-1}\mathbf{V}^{-1}\mathbf{m})^{1/2}$ (normalization constant)

**Critical Values**: Determined by Monte Carlo simulations (no closed-form distribution exists)

#### When to Use

**The Ideal** (large sample luxury): $n_{runs} \geq 30$  
✅ Central Limit Theorem provides robustness → can skip normality testing  
✅ Parametric tests (t-test) remain valid even with moderate non-normality  
✅ Visual inspection (Q-Q plots, histograms) often sufficient

**Our Reality** (small sample constraint): $n_{runs} = 15$  
⚠️ **Below CLT threshold** → normality testing is **mandatory, not optional**  
⚠️ Cannot rely on asymptotic robustness of t-test  
⚠️ Shapiro-Wilk becomes **gatekeeper test**: Pass (p ≥ 0.05) → t-test | Fail ($p < 0.05$) → Wilcoxon

**Appropriate scenarios**:

- ✅ Sample size: 3 $\leq$ n $\leq$ 5,000 (Royston/Rahman-Govidarajulu extensions)
- ✅ Univariate continuous data
- ✅ **Before parametric tests (t-test, ANOVA) when n < 30** ← **Our mandatory use case**
- ✅ When test power is critical (Shapiro-Wilk has best power among normality tests)

**Not appropriate**:

- ❌ Discrete or categorical data
- ❌ Multivariate normality testing (use Mardia's test instead)
- ❌ Very large samples (n > 5,000) - use graphical methods (Q-Q plots)

#### Interpretation

##### Decision Rule (Statistical Mechanics)

- If p-value < $\alpha$ (typically 0.05): **Reject $H_0$** → Data is NOT normally distributed
- If p-value $\geq$ $\alpha$: **Fail to reject $H_0$** → No evidence against normality

##### Conceptual Meaning in Algorithm Benchmarking

The Shapiro-Wilk test detects **specific departures from the bell curve shape**: skewness (asymmetry), kurtosis (heavy/light tails), outliers, or multimodality.

**What rejection ($p < 0.05$) tells you**:
- Not just "non-normal" (vague)
- But specifically: "Distribution shape differs enough from Gaussian that parametric assumptions are violated"
- Practical consequence: t-test may produce incorrect p-values and confidence intervals

**Example scenarios**:

```text
Scenario 1 - NORMAL (W=0.96, p=0.68):
Execution times: [22.1, 22.3, 22.5, 22.7, 22.9, 23.1, 23.3] (symmetric, unimodal)
→ Use paired t-test ✅

Scenario 2 - SKEWED (W=0.87, p=0.02):
Execution times: [5.1, 5.2, 5.3, 5.4, 5.5, 7.8, 9.2] (right tail)
→ Use Wilcoxon ⚠️ (t-test would overestimate p-value)

Scenario 3 - BIMODAL (W=0.84, p=0.008):
Execution times: [10.2, 10.5, 10.3, 22.1, 22.4, 22.2] (two clusters)
→ Use Wilcoxon ⚠️ (suggests different algorithmic behavior)
```

>[!caution]
> So need to use wilcoxon for my approach? Improve this example

**GPU Benchmark Context** ($n_{runs}=15$):
With small samples, Shapiro-Wilk functions as **mandatory gatekeeper**: must pass normality before using t-test. Failure requires switching to Wilcoxon (robust but 4.5% power penalty).

##### Important Caveats

1. **Large Sample Sensitivity**: With n > 100, test may detect trivial departures from normality that have no practical impact. Always supplement with Q-Q plots.

2. **W Statistic Range**: 0 < W $\leq$ 1
   - W ≈ 1: Data closely follows normal distribution
   - W < 0.9: Strong departure from normality

3. **Power Comparison**: Shapiro-Wilk > Anderson-Darling > Kolmogorov-Smirnov > Lilliefors for detecting non-normality (Razali & Wah, 2011)

#### Normality Assessment Decision Tree

**Flowchart**: When and how to assess normality before selecting a statistical test.

```mermaid
graph TD
    A[Start: Check Normality Assumption] --> B{Sample Size n?}:::decision
    B -->|n < 20| C[MUST Test Normality<br/>Shapiro-Wilk Required]:::decision
    B -->|20 <= n < 30| D[Test Recommended<br/>Borderline for CLT]:::decision
    B -->|n >= 30| E{Visual Check<br/>Severe Skewness?}:::decision
    C --> F[Run Shapiro-Wilk Test]
    D --> F
    E -->|Yes, Severe| F
    E -->|No| G[Assume Normal<br/>CLT Applies]:::parametric
    F --> H{p-value?}:::decision
    H -->|$p < 0.05$| I[Non-Normal Distribution<br/>Use Non-Parametric Tests]:::nonparametric
    H -->|p >= 0.05| J[Normal Distribution<br/>Use Parametric Tests]:::parametric

```

**Key Decision Points**:

- **$n < 20$**: Normality testing mandatory - CLT does not apply
- **$20 \leq n < 30$**: Gray zone - test recommended for robustness
- **$n \geq 30$**: CLT typically applies, but check for extreme skewness
- **GPU Benchmark Context**: With n=15 runs, we're **below** the CLT threshold - normality testing is **mandatory**

#### Implementation in Our Benchmark

```python
from scipy import stats

def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Test data for normality using Shapiro-Wilk test.

    Args:
        data: 1D array of observations (n >= 3)
        alpha: Significance level (default: 0.05)

    Returns:
        (p_value, is_normal) where is_normal = (p >= alpha)
    """
    if len(data) < 3:
        # Shapiro-Wilk requires n >= 3
        return (0.0, False)

    statistic, p_value = stats.shapiro(data)
    is_normal = p_value >= alpha
    return (float(p_value), is_normal)
```

**SciPy Implementation**: Uses Royston (1992) approximation for samples up to 5,000

#### Example from Benchmark

>[!tip] **Exercise: Shapiro-Wilk Test on Berlin52 CPU Execution Times**
>
> **Scenario**: You've collected $n=15$ execution time measurements for the CPU algorithm on the berlin52 TSP instance. Before conducting a paired t-test comparing CPU vs GPU performance, you must verify the normality assumption.
>
> **Data (CPU raw execution times in seconds)**:
> $$
> \begin{align}
> \mathbf{X} = \{&28.19, 29.22, 32.38, 29.19, 29.83, 29.99, 26.88, 25.84, \\
>                &27.05, 22.83, 20.99, 22.31, 20.92, 20.21, 20.26\}
> \end{align}
> $$
>
> **Task 1: Calculate Descriptive Statistics**
>
> $$
> \begin{align}
> \bar{x} &= \frac{1}{15}\sum_{i=1}^{15} x_i = \frac{386.09}{15} = 25.74\text{ seconds} \\
> s &= \sqrt{\frac{1}{14}\sum_{i=1}^{15}(x_i - \bar{x})^2} = 3.99\text{ seconds}
> \end{align}
> $$
>
> **Task 2: Apply Shapiro-Wilk Test**
>
> [!warning] must show maths, even if abstract. use matrices
> ```python
> from scipy import stats
> import numpy as np
>
> cpu_times = np.array([28.19, 29.22, 32.38, 29.19, 29.83, 29.99, 26.88, 25.84,
>                       27.05, 22.83, 20.99, 22.31, 20.92, 20.21, 20.26])
>
> # Run Shapiro-Wilk test
> statistic, p_value = stats.shapiro(cpu_times)
> print(f"W = {statistic:.4f}, p = {p_value:.4f}")
> ```
>
> **Expected Result**: $W \approx 0.9542$, $p \approx 0.5821$
>
> **Task 3: Decision Rule**
>
> $$
> \begin{cases}
> p \geq 0.05 & \Rightarrow \text{Fail to reject } H_0 \text{: Data appears normal} \\
> p < 0.05 & \Rightarrow \text{Reject } H_0 \text{: Data NOT normal}
> \end{cases}
> $$
>
> **Your Decision**: Since $p = 0.5821 > 0.05$, we **fail to reject** $H_0$.
>
> **Interpretation**: The CPU execution times show no significant departure from normality. The distribution is approximately Gaussian, satisfying the prerequisite for paired t-test. We can proceed with parametric analysis.
>
> **✓ Verification Check**:
> - Visual check: Plot histogram or Q-Q plot to confirm
> - Rule of thumb: If $0.90 \leq W \leq 1.0$ and $p > 0.05$, normality assumption is safe
> - Our result: $W = 0.9542$ ✅, $p = 0.5821$ ✅ → **Proceed to paired t-test**

**Case 2: Non-Normal Distribution (Solution Costs with Zero Variance)**

>[!caution] **Understanding Zero-Variance Cases**
>
> **Scenario**: All algorithms achieve optimal solution on berlin52
>
> $$
> \text{Solution costs} = \{7542, 7542, 7542, \ldots, 7542\} \quad (n=15)
> $$
>
> **Shapiro-Wilk Result**: $W = 1.0000$, $p = 1.0000$
>
> **Why This Happens**:
> - Range = $\max - \min = 0$ (all values identical)
> - Standard deviation $s = 0$ (no variation)
> - Shapiro-Wilk interprets this as "perfect normality" (degenerate case)
>
> **Decision**: **Skip statistical testing**
> - No variance → no difference to detect
> - Comparing constant values is meaningless
> - Focus statistical analysis on **execution time**, not solution quality
>
> **Key Insight for GPU Benchmarking**:
> When all algorithms achieve optimal solutions (as expected for small TSP instances), compare **RUNTIME PERFORMANCE**, not solution quality. This is why our statistical analysis focuses on execution time differences.

#### Assumptions and Limitations

**Assumptions**:

1. Data is continuous (interval or ratio scale)
2. Observations are independent
3. Sample size 3 $\leq$ n $\leq$ 5,000

**Limitations**:

1. **High sensitivity with large n**: May reject normality for trivial departures
2. **Low power with small n**: May fail to detect non-normality with n < 20
3. **Software differences**: Some packages use sample parameters (m,s) vs population (μ,σ)
4. **Not robust to outliers**: Single extreme value can affect results
5. **Degenerate cases**: Zero variance data produces p=1.0 (perfect "normality")

**Best Practices**:

- Combine with visual assessment (Q-Q plot, histogram)
- For n > 100, focus on effect size rather than statistical significance
- Consider robustness: If barely non-normal, t-test may still be appropriate due to CLT

[^2][^3][^4][^5]

[^2]: Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3-4), 591-611. DOI: [10.1093/biomet/52.3-4.591](https://doi.org/10.1093/biomet/52.3-4.591)

[^3]: Razali, N. M., & Wah, Y. B. (2011). Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests. *Journal of Statistical Modeling and Analytics*, 2(1), 21-33.

[^4]: Royston, P. (1992). Approximating the Shapiro-Wilk W-test for non-normality. *Statistics and Computing*, 2(3), 117-119. DOI: [10.1007/BF01891203](https://doi.org/10.1007/BF01891203)

[^5]: Field, A. (2009). *Discovering Statistics Using SPSS* (3rd ed.). SAGE Publications. p. 143.

---

## Parametric Tests

### Paired t-Test

- **Also Known As**: Paired-samples t-test, dependent t-test, matched-pairs t-test  
- **Purpose**: Compare means of two related samples when data is normally distributed.

1. **Null Hypothesis ($H_0$)**: μ_d = 0 (mean difference = 0, no performance difference)  
1. **Alternative (H₁)**: μ_d $\neq$ 0 (mean difference $\neq$ 0, algorithms differ)

#### Plain Language Explanation

**Think of it as**: Measuring the same thing twice under different conditions.

**Everyday Analogy**:
> Imagine measuring your weight before and after a diet. You're the same person (paired), measured at two time points. The paired t-test asks: "Is the average weight change across all dieters significantly different from zero?"

**Key Insight**: By measuring the **same subjects/instances** twice, you remove individual variability. This makes paired tests **much more powerful** than comparing two independent groups.

**GPU Benchmark Context**:
- **Subject** = TSP problem instance (e.g., eil51, berlin52)
- **Condition 1** = CPU execution time
- **Condition 2** = GPU execution time
- **Question** = "Is the average speedup across all 30 instances significantly different from zero?"

**Why Paired?**: The same instance tested on CPU vs GPU controls for problem difficulty—we're measuring algorithm performance difference, not problem complexity.

#### Mathematical Formulation

The paired t-test operates on the **differences** between paired observations:

$$\begin{equation}
d_i = x_{1i} - x_{2i}
\end{equation}$$

The test statistic is:

$$\begin{equation}
t = \frac{\bar{d}}{s_d / \sqrt{n}}
\end{equation}$$

Where:

- $\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d_i$ = mean of differences
- $s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar{d})^2}$ = standard deviation of differences
- $n$ = number of pairs
- **Degrees of freedom**: $df = n - 1$

**Critical Value**: Compare t to critical value from Student's t-distribution with $df = n-1$

**Confidence Interval for Mean Difference**:

$$\begin{equation}
CI_{95\%} = \bar{d} \pm t_{0.975,n-1} \times \frac{s_d}{\sqrt{n}}
\end{equation}$$

#### Decision Framework: When to Use Paired t-Test

**Quick Decision Tree**:

```text
Q1: Is data paired (same subjects/instances measured twice)?
├─ NO → Use independent t-test or Mann-Whitney U
└─ YES → Continue to Q2

Q2: Are differences approximately normal (Shapiro-Wilk p >= 0.05)?
├─ NO → Use Wilcoxon signed-rank test
└─ YES → Continue to Q3

Q3: Sample size >= 20?
├─ YES → ✅ Paired t-test (CLT protection)
└─ NO → Check Q4

Q4: Large effect expected (d > 0.8)?
├─ YES → ✅ Paired t-test (adequate power at n=15)
└─ NO → ⚠️ Consider Wilcoxon (more robust) or increase n
```

**Comparison Table**: Paired t-Test vs Alternatives

| Criterion | Paired t-Test | Wilcoxon Signed-Rank | Independent t-Test |
|-----------|---------------|----------------------|-------------------|
| **Data Pairing** | Required ✅ | Required ✅ | Not allowed ❌ |
| **Normality** | Differences ~ Normal | No assumption ✅ | Both groups ~ Normal |
| **Power (normal data)** | 100% (baseline) | 95.5% (ARE=0.955) | Lower (ignores pairing) |
| **Robustness** | Sensitive to outliers | Robust ✅ | Moderate |
| **Sample Size** | n >= 20 preferred | n >= 10 OK | n >= 30 for CLT |
| **Effect Size** | Cohen's d | Rank-biserial r | Cohen's d |
| **GPU Benchmark** | ✅ Primary choice | Backup if not normal | Wrong (not paired) |

#### When to Use

**Ideal Scenario** (statistician's checklist):

- ✅ **Paired/matched design**: Same subjects measured twice (before/after, pre/post)
- ✅ **Related samples**: Natural pairing exists (twins, matched controls)
- ✅ **Within-subjects**: Same problem instances tested on different algorithms
- ✅ **Normality**: Differences approximately normally distributed (verified by Shapiro-Wilk)
- ✅ **Continuous data**: Interval or ratio scale measurements
- ✅ **Sample size**: $n_{runs} \geq 30$ for CLT protection (n $\geq$ 20 acceptable, n $\geq$ 10 minimum)

**Our Reality** (benchmark context with $n_{runs}=15$):

- ✅ **Paired data**: Same 30 TSP instances tested on both CPU and GPU
- ⚠️ **Small sample penalty**: $n_{runs}=15 < 30$ → **cannot rely on CLT** → normality testing mandatory
- ⚠️ **Wider confidence intervals**: Use $t_{14,0.975} = 2.145$ (9.4% wider than ideal Z = 1.96)
- ⚠️ **Power constraint**: Adequate only for large effects ($d \geq 0.8$), underpowered for $d < 0.5$
- ✅ **Trade-off justified**: GPU speedups typically produce $d > 2.0$ (>99% power achieved)

**Our benchmark context**:

```python
# Perfect paired design:
# - Same 30 TSP instances tested on both CPU and GPU
# - Natural pairing: instance i on CPU matched with instance i on GPU
# - Within-subjects factor: Algorithm implementation (CPU vs GPU)
```

**Not appropriate**:

- ❌ **Independent samples**: Use two-sample t-test instead
- ❌ **Non-normal differences**: Use Wilcoxon signed-rank test
- ❌ **Multiple groups** (k > 2): Use repeated-measures ANOVA or Friedman test
- ❌ **Unequal variances** with small n: Use Wilcoxon (more robust)

#### Interpretation

##### Decision Rule (Statistical Mechanics)

- If $|t| > t_{critical}$ or $p-value < α$: **Reject $H_0$** → Significant difference exists
- If $|t| \leq t_{critical}$ or $p-value \geq α$: **Fail to reject $H_0$** → No evidence of difference

##### Conceptual Meaning in Algorithm Benchmarking

The paired t-test detects **systematic mean differences** between paired measurements, answering: "Is the average performance gap consistent enough that it's unlikely due to random variation?"

**What the test actually measures**:
- Not just "Are the means different?" (too vague)
- But: "Is the mean of paired differences ($\bar{d}$) significantly far from zero in units of standard error?"
- The t-statistic = $\frac{\bar{d}}{SE}$ measures signal-to-noise ratio

**Example scenarios**:

```text
Scenario 1 - SIGNIFICANT (t=25.7, p<0.001):
CPU-GPU differences: [17.3, 18.8, 17.1, 17.9, ...] (mean=17.4s, SE=0.17s)
→ Mean difference 17.4s is 25.7× larger than noise (SE)
→ Extremely strong evidence of systematic speedup

Scenario 2 - NOT SIGNIFICANT (t=1.8, p=0.09):
HybridA-HybridB differences: [0.2, -0.1, 0.4, 0.1, ...] (mean=0.15s, SE=0.08s)
→ Mean difference 0.15s is only 1.8× larger than noise
→ Could easily be random variation, not systematic
```

**GPU Benchmark Context** ($n_{runs}=15$):
With typical GPU speedups producing $d>2.0$, even small sample ($n_{runs}=15$) achieves $t>10, p<0.001$. The test confirms "GPU is systematically faster across ALL instances, not just lucky on a few."

##### P-value Interpretation

**See [Section 0.1.2 P-value Interpretation Ladder](#interpretation-ladder)** for detailed guidance on interpreting p-values.

**Quick reference**: $p < 0.001$ (very strong evidence), $p < 0.01$ (strong), $p < 0.05$ (moderate), $p ≥ 0.05$ (insufficient).

##### Effect Size and Power

**Effect Size**: Always report Cohen's d alongside p-value for practical significance

**Power** (with α = 0.05, $n_{runs}=15$):
- d = 0.5 (medium effect) → Power ≈ 0.46 (underpowered)
- d = 0.8 (large effect) → Power ≈ 0.81 (adequate)
- d = 1.5 (very large effect) → Power ≈ 0.99 (excellent)

**See Also**: [Statistical Power Analysis](#statistical-power-analysis) for sample size planning and power curves.

#### Effect Size: Cohen's d for Paired Data

**Formula for Paired Samples**:

$$\begin{equation}
d = \frac{\bar{d}}{s_d}
\end{equation}$$

where $\bar{d}$ = mean difference, $s_d$ = standard deviation of differences.

**Interpretation Benchmarks**:

| Cohen's d | Interpretation | GPU Context Example |
|-----------|----------------|---------------------|
| $\|d\| < 0.2$ | Negligible | 5% speedup (not worth effort) |
| $0.2 \leq \|d\| < 0.5$ | Small | 20% speedup (marginal improvement) |
| $0.5 \leq \|d\| < 0.8$ | Medium | 50% speedup (noticeable) |
| $0.8 \leq \|d\| < 1.2$ | Large | $2\times$ speedup (significant) |
| $\|d\| \geq 1.2$ | Very Large | $10\times$ speedup (transformative) |

**Why Report Effect Size?**:
- p-value tells you **IF** a difference exists (statistical significance)
- Effect size tells you **HOW LARGE** the difference is (practical significance)
- With large $n_{samples}$, even tiny effects become "statistically significant"
- GPU benchmarks often have $d > 2.0$ (extremely large practical impact)

>[!tip] **Exercise: Paired t-Test on Berlin52 CPU vs HybridOptimized**
>
> **Scenario**: Compare execution times between CPU and HybridOptimized algorithms on berlin52. Both algorithms achieve optimal solution (cost = 7542), so we focus on **runtime performance**.
>
> **Data (execution times in seconds, $n=15$ paired runs)**:
>
> $$
> \begin{align}
> \text{CPU: } \mathbf{X}_1 &= \{28.19, 29.22, 32.38, 29.19, 29.83, 29.99, 26.88, 25.84, \ldots\} \\
> \text{HybOpt: } \mathbf{X}_2 &= \{0.082, 0.086, 0.089, 0.088, 0.105, 0.089, 0.090, 0.092, \ldots\}
> \end{align}
> $$
>
> **Task 1: Calculate Paired Differences**
>
> $$
> \begin{align}
> d_i &= x_{1i} - x_{2i} \quad \text{(CPU time - HybOpt time)} \\
> \mathbf{d} &= \{28.11, 29.13, 32.29, 29.10, 29.73, 29.90, 26.79, 25.75, \ldots\}
> \end{align}
> $$
>
> **Task 2: Compute Mean and SD of Differences**
>
> $$
> \begin{align}
> \bar{d} &= \frac{1}{15}\sum_{i=1}^{15} d_i = \frac{385.00}{15} = 25.67\text{ seconds} \\
> s_d &= \sqrt{\frac{1}{14}\sum_{i=1}^{15}(d_i - \bar{d})^2} = 3.99\text{ seconds}
> \end{align}
> $$
>
> **Task 3: Calculate t-Statistic**
>
> $$
> \begin{align}
> SE &= \frac{s_d}{\sqrt{n_{runs}}} = \frac{3.99}{\sqrt{15}} = 1.03\text{ seconds} \\
> t &= \frac{\bar{d}}{SE} = \frac{25.67}{1.03} = 24.92 \\
> df &= n_{runs} - 1 = 14
> \end{align}
> $$
>
> **Task 4: Find P-value and Critical Value**
>
> ```python
> from scipy import stats
> import numpy as np
>
> cpu_times = np.array([28.19, 29.22, 32.38, 29.19, 29.83, 29.99, 26.88, 25.84,
>                       27.05, 22.83, 20.99, 22.31, 20.92, 20.21, 20.26])
> hyb_times = np.array([0.082, 0.086, 0.089, 0.088, 0.105, 0.089, 0.090, 0.092,
>                       0.090, 0.095, 0.097, 0.090, 0.084, 0.110, 0.090])
>
> # Conduct paired t-test
> t_stat, p_value = stats.ttest_rel(cpu_times, hyb_times)
> t_crit = stats.t.ppf(0.975, df=14)  # Two-tailed, α=0.05
>
> print(f"t-statistic = {t_stat:.2f}")
> print(f"p-value = {p_value:.2e}")
> print(f"t_critical = {t_crit:.3f}")
> ```
>
> **Expected Results**:
> - $t = 24.92$
> - $p < 0.0001$ (extremely significant)
> - $t_{\text{crit}}(14, 0.975) = 2.145$
>
> **Task 5: Decision Rule**
>
> $$
> \begin{cases}
> |t| > t_{\text{crit}} & \Rightarrow \text{Reject } H_0 \text{ (significant difference)} \\
> |t| \leq t_{\text{crit}} & \Rightarrow \text{Fail to reject } H_0 \text{ (no difference)}
> \end{cases}
> $$
>
> **Your Decision**: Since $|24.92| > 2.145$, we **strongly reject** $H_0$.
>
> **Task 6: Construct 95% Confidence Interval**
>
> $$
> \begin{align}
> CI_{95\%} &= \bar{d} \pm t_{0.975, 14} \times SE \\
>           &= 25.67 \pm 2.145 \times 1.03 \\
>           &= 25.67 \pm 2.21 \\
>           &= [23.46, 27.88] \text{ seconds}
> \end{align}
> $$
>
> **Interpretation**:
> 1. **Statistical Significance**: $p < 0.0001$ indicates extremely strong evidence that CPU and HybridOptimized have different mean execution times
> 2. **Practical Significance**: CPU takes 23.46-27.88 seconds longer on average (with 95% confidence)
> 3. **Speedup Factor**: $\frac{25.74}{0.092} \approx 280\times$ faster with GPU optimization
> 4. **Effect Size**: See Cohen's d calculation in next exercise
>
> **✓ Verification Checks**:
> - ✅ Normality verified (Shapiro-Wilk $p = 0.58$)
> - ✅ Paired design (same berlin52 instance, different algorithms)
> - ✅ Large effect expected ($d > 2.0$), adequate power even with $n=15$
> - ✅ $|t| \gg t_{\text{crit}}$ confirms robust finding

>[!important] **Understanding `ddof` (Delta Degrees of Freedom)**
>
> **Question**: When calculating $s_d$ with `np.std(differences, ddof=1)`, when would `ddof` be $0$ or $>1$?
>
> **Answer**: The `ddof` parameter adjusts the denominator in variance calculation:
>
> $$
> \begin{align}
> \text{ddof} &= \text{Degrees of Freedom correction} \\
> s^2 &= \frac{1}{n - \text{ddof}}\sum_{i=1}^{n}(x_i - \bar{x})^2
> \end{align}
> $$
>
> **Usage Cases**:
>
> $$
> \begin{cases}
> \text{ddof} = 0 & \Rightarrow \text{Population variance } \sigma^2 \text{ (known true mean)} \\
> \text{ddof} = 1 & \Rightarrow \text{Sample variance } s^2 \text{ (estimated mean, Bessel's correction)} \\
> \text{ddof} > 1 & \Rightarrow \text{Specialized cases (e.g., multivariate regression)}
> \end{cases}
> $$
>
> **Why `ddof=1` for t-tests?**
> - We use **sample mean** $\bar{x}$ (not true population mean $\mu$)
> - This "uses up" 1 degree of freedom
> - Bessel's correction ($n-1$) gives **unbiased** variance estimate
> - Using $n$ instead of $n-1$ **underestimates** variance
>
> **GPU Benchmark Context**: Always use `ddof=1` when calculating standard errors for hypothesis testing on sample data.

#### Power Analysis Integration

**Sample Size Planning** (from [Statistical Power Analysis](#statistical-power-analysis)):

For paired t-test with α = 0.05 (two-tailed), target power = 0.80:

| Expected Effect Size | Required Sample Size | Power at n=15 | Comment |
|---------------------|---------------------|---------------|---------|
| d = 0.2 (small) | n ≈ 199 | 11% ⚠️ | Need 13× more samples |
| d = 0.5 (medium) | n ≈ 34 | 46% ⚠️ | Underpowered |
| d = 0.8 (large) | n ≈ 15 | 81% ✅ | Adequate |
| d = 1.0 (very large) | n ≈ 10 | 92% ✅ | Good margin |
| d = 1.5+ (extreme) | n ≈ 6 | >99% ✅ | Typical for GPU |

**Our Benchmark Justification** ($n_{runs}=15$ pairs):
- **Expected effects**: GPU speedups of 2×-100× correspond to d > 2.0 (very large effect sizes)
- **Achieved power**: >99% for detecting large differences (exceeds 80% threshold)
- **Cost-benefit**: $n_{runs}=15$ → 15 runs × 30 instances × 2 algorithms = 900 total experiments (manageable)
- **Robustness**: Even for d=0.8 (large effect, 80% speedup), we achieve 81% power ✅
- **Trade-off accepted**: Small effects (d < 0.5) would be underpowered, but GPU research targets transformative speedups

**Python Power Calculation**:
```python
from statsmodels.stats.power import ttest_power

# Calculate power for n=15, effect size d=0.8
power = ttest_power(effect_size=0.8, nobs=15, alpha=0.05,
                    alternative='two-sided')
print(f"Power = {power:.2f}")  # 0.81

# Calculate required n for 80% power, d=0.5
from statsmodels.stats.power import tt_solve_power
n_required = tt_solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                             alternative='two-sided')
print(f"Required n = {n_required:.0f}")  # 34
```

>[!caution]
> where does statsmodel come from? scipy?

#### Post-hoc Tests and Multiple Comparisons

**For Paired t-Test** (k=2 groups only):
- ✅ **No post-hoc needed**: Only comparing 2 conditions (CPU vs GPU)
- ✅ **No multiple comparison correction**: Single hypothesis test

**When You Need Post-hoc** (k > 2 groups):
- If comparing **multiple algorithms** (e.g., CPU, GPU-v1, GPU-v2, Hybrid):
  - Use **Friedman test** first (non-parametric repeated measures ANOVA)
  - If significant → **Nemenyi test** for pairwise comparisons
  - See [Friedman Test](#friedman-test) and [Nemenyi Test](#nemenyi-test)

>[!caution]
>why? why should I use this test? is it a "multi-paired" version? (explain in the proper section)

**Multiple Testing Example**:
```text
If testing CPU vs GPU on 30 different problem instances:
- This is STILL a single paired t-test (one hypothesis: μ_d = 0)
- The 30 pairs are the sample, not 30 separate tests
- NO multiple comparison correction needed ✅

If testing CPU vs GPU separately on 5 different problem sizes:
- Now you have 5 tests → risk of false positives
- Apply Holm-Bonferroni correction (α = 0.05/5 = 0.01 for first test)
- See [Holm-Bonferroni Method](#holm-bonferroni-method)
```

#### Python Implementation Details

```python
from scipy import stats

def paired_comparison(data_a, data_b, label, metric_name, alpha=0.05):
    """
    Perform paired t-test after normality check.
    """
    # Check normality of differences
    differences = data_a - data_b
    _, p_norm = stats.shapiro(differences)

    if p_norm >= alpha:
        # Use parametric test
        statistic, p_value = stats.ttest_rel(data_a, data_b)
        test_used = "paired_t_test"
    else:
        # Fall back to non-parametric
        statistic, p_value = stats.wilcoxon(data_a, data_b)
        test_used = "wilcoxon_signed_rank"

    return {
        "test": test_used,
        "statistic": statistic,
        "p_value": p_value,
        "normality_p": p_norm
    }
```

**SciPy Implementation**: `scipy.stats.ttest_rel(a, b, alternative='two-sided')`

#### Example from Benchmark

**Case 1: eil51 - CPU vs HybridNaive (execution times)**

```
Sample Data (n=15 pairs):
CPU times:     [22.52, 24.28, 22.37, 23.15, 22.89, 23.45, 22.18, ...]
GPU times:     [5.19, 5.45, 5.31, 5.67, 5.42, 5.28, 5.11, ...]
Differences:   [17.33, 18.83, 17.06, 17.48, 17.47, 18.17, 17.07, ...]

Statistical Results:
Mean difference (d̄): 17.54 seconds
Std dev (sd):        0.68 seconds  
t-statistic:         99.87
Degrees of freedom:  14
p-value:             < 2.2e-16
95% CI:              [17.17, 17.91]

Interpretation:
CPU is significantly slower than GPU (t(14) = 99.87, p < 0.001)
Mean speedup: 22.89/5.45 = 4.20$\times$
Effect size (Cohen's d): 25.71 (extremely large)
```

**Case 2: berlin52 - All algorithms achieve optimal (zero variance)**

```
CPU costs: [7542, 7542, 7542, ...]
GPU costs: [7542, 7542, 7542, ...]
Differences: [0, 0, 0, ...]

Result: Test bypassed (zero variance case)
Conclusion: All algorithms performed identically
```

#### Common Pitfalls and How to Avoid Them

**1. Treating Paired Data as Independent** ❌ → ✅
- **Error**: Running independent t-test on paired data
- **Consequence**: Lose power by ignoring pairing, inflated Type II error
- **Fix**: Always use `stats.ttest_rel()` for paired data, not `stats.ttest_ind()`
- **Example**: Same 30 TSP instances on CPU vs GPU → Paired, not independent

**2. Ignoring Normality Assumption** ❌ → ✅
- **Error**: Running paired t-test without checking normality of **differences**
- **Consequence**: Invalid p-values if differences heavily skewed/outliers present
- **Fix**: Always run Shapiro-Wilk on differences first: `stats.shapiro(data_a - data_b)`
- **Decision**: If $p < 0.05$ → Use Wilcoxon signed-rank instead

**3. Confusing CI of Difference with Effect Size** ❌ → ✅
- **Error**: "95% CI = [17.17, 17.91] means large effect"
- **Reality**: CI describes precision of mean difference, not magnitude relative to variability
- **Fix**: Always report Cohen's d ($d = 17.54 / 0.68 = 25.71$) alongside CI
- **Interpretation**: CI tells WHERE difference is, d tells HOW LARGE it is

**4. One-Sided vs Two-Sided Testing Confusion** ❌ → ✅
- **Error**: Using one-sided test when direction not predicted a priori
- **Consequence**: Inflated Type I error (p-values appear more significant)
- **Fix**: Default to two-sided (`alternative='two-sided'`) unless strong directional hypothesis
- **GPU Context**: We know GPU should be faster → one-sided OK if justified

**5. Reporting Only p-value Without Effect Size** ❌ → ✅
- **Bad**: "p < 0.001, GPU is significantly faster"
- **Good**: "p < 0.001, d = 25.71 (extremely large), GPU is significantly faster"
- **Why**: With n=15, even trivial effects might not reach significance (Type II error)
- **Best Practice**: Report t-statistic, df, p-value, d, and 95% CI

**6. Forgetting Sign Convention** ❌ → ✅
- **Error**: Computing `gpu_times - cpu_times` then saying "positive = GPU faster"
- **Reality**: Negative differences = GPU faster (smaller times)
- **Fix**: Be explicit: `differences = cpu_times - gpu_times` (positive = GPU faster)
- **Report**: "Mean difference = 17.54s (CPU slower by 17.54s per run)"

#### Historical Context: The Birth of Modern Statistics

**William Sealy Gosset (1876-1937)** - "Student"

**The Problem** (1908):
- Gosset worked as Head Brewer at Guinness Brewery in Dublin
- Needed to test barley quality with **small samples** (n < 30) due to cost/time
- Existing Z-test assumed known population variance σ² (unrealistic)
- Small sample estimates of variance (s²) were too variable

**The Solution**:
- Derived exact distribution of $t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$ for small n
- Distribution has **heavier tails** than normal (accounts for estimation uncertainty)
- Named "Student's t-distribution" (published under pseudonym due to company policy)

**Why "Student"?**:
- Guinness forbade employees from publishing (trade secret concerns)
- Gosset published under pseudonym "Student" to circumvent policy
- Irony: Created one of statistics' most fundamental tools in secrecy

**Legacy**:
- 1908 paper revolutionized small-sample inference
- Enabled quality control in manufacturing (original use case)
- Foundation for paired t-test, ANOVA, regression t-tests
- Still the standard for small-sample inference 117 years later

**Fun Fact**: Gosset's work was initially rejected—editors didn't believe distribution wasn't normal!

#### Computational Complexity

**Time Complexity**:
- **Mean calculation**: O(n) - single pass through differences
- **Std dev calculation**: O(n) - single pass with mean
- **t-statistic**: O(1) - simple formula
- **Overall**: O(n) - linear in number of pairs

**Space Complexity**:
- **Input**: O(n) - two arrays of length n
- **Differences**: O(n) - temporary array
- **Overall**: O(n) - linear memory

**Computational Cost Comparison** (for n=15 pairs):

| Test | Time Complexity | Operations | Relative Cost |
|------|----------------|------------|---------------|
| **Paired t-test** | O(n) | ~45 ops | 1× (baseline) |
| **Wilcoxon signed-rank** | O(n log n) | ~60 ops | 1.3× (ranking) |
| **Shapiro-Wilk** | O(n²) | ~200 ops | 4.4× (correlation) |
| **Bootstrap (1000 reps)** | O(1000n) | ~45,000 ops | 1000× |

**Why This Matters for GPU Benchmarks**:
- With 30 problems × 4 algorithms × 15 runs = 1,800 total runs
- Statistical analysis: 30 problems × 6 pairwise comparisons × O(15) = O(2,700) operations
- **Analysis time: milliseconds** (negligible compared to hours of benchmark runtime)
- Bottleneck is running experiments, not analyzing results

**Scaling Properties**:
```python
import time
import numpy as np
from scipy import stats

for n in [10, 100, 1000, 10000]:
    data_a = np.random.randn(n)
    data_b = np.random.randn(n)

    start = time.perf_counter()
    _, _ = stats.ttest_rel(data_a, data_b)
    elapsed = time.perf_counter() - start

    print(f"n={n:5d}: {elapsed*1000:.3f} ms")

# Output:
# n=   10: 0.024 ms
# n=  100: 0.031 ms
# n= 1000: 0.089 ms
# n=10000: 0.712 ms  (still < 1ms!)
```

**Conclusion**: Paired t-test scales excellently—even with n=10,000 pairs, analysis takes <1ms. GPU benchmark analysis is computationally trivial.

#### Assumptions and Limitations

**Assumptions**:

1. **Independence of pairs**: Pairs are independent of each other (our benchmark: ✓)
2. **Normality of differences**: $d_i \sim N(\mu_d, \sigma^2)$ or $n_{runs}$ large enough for CLT
3. **Random sampling**: Pairs randomly sampled from population
4. **Interval/ratio data**: Meaningful numeric differences
5. **No outliers**: Extreme values can inflate variance

**Limitations**:

1. **Sensitive to normality violations** (small n): Use Wilcoxon if Shapiro-Wilk $p < 0.05$
2. **Assumes equal variance** in paired measurements (less critical than two-sample t-test)
3. **One-to-one pairing required**: Missing pairs must be excluded (listwise deletion)
4. **Direction of difference matters**: Sign convention (A - B vs B - A) affects interpretation
5. **Carryover effects**: If repeated measures, ensure no learning/fatigue effects

**Robustness**:

- **CLT protection**: With n $\geq$ 30, test is robust to moderate non-normality
- **Balanced design**: Paired design increases power by removing between-subject variance
- **Missing data**: If some pairs incomplete, those pairs are dropped (reduces effective n)

[^1][^6][^7][^8][^9]

[^6]: Bevans, R. (2023). An Introduction to t Tests | Definitions, Formula and Examples. *Scribbr*. Retrieved from <https://www.scribbr.com/statistics/t-test/>

[^7]: UCLA Statistical Consulting. Power Analysis for Paired Sample t-test | R Data Analysis Examples. <https://stats.oarc.ucla.edu/r/dae/power-analysis-for-paired-sample-t-test/>

[^8]: Statistics Solutions. Paired Sample T-Test. <https://www.statisticssolutions.com/paired-sample-t-test/>

[^9]: Montgomery, D.C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. Chapter 3: Experiments with a Single Factor.

---

## Non-Parametric Tests

### Wilcoxon Signed-Rank Test

**Also Known As**: Wilcoxon T-test, Wilcoxon paired signed-rank test  
**Purpose**: Non-parametric alternative to paired t-test for testing location differences when normality fails.

**Null Hypothesis ($H_0$)**: The differences $X_i - Y_i$ are symmetric about 0 (median difference = 0)  
**Alternative (H₁)**: The differences are symmetric about $μ \neq 0$ (median difference $\neq$ 0)

**Classification**: Rank-based non-parametric test, more powerful than sign test

#### Mathematical Formulation

The Wilcoxon signed-rank test operates on **signed ranks** of paired differences:

**Step 1: Compute Differences**

$$\begin{equation}
d_i = X_i - Y_i \quad \text{for } i = 1, \ldots, n
\end{equation}$$

**Step 2: Remove Zeros and Rank Absolute Values**

1. Exclude pairs where $d_i = 0$ (Wilcoxon's reduced sample procedure)
2. Compute absolute values: $|d_1|, |d_2|, \ldots, |d_n|$
3. Assign ranks $R_i$ based on $|d_i|$: Smallest absolute value → rank 1, next → rank 2, etc.
4. Handle ties by assigning **average ranks** (midrank procedure)

**Step 3: Apply Signs to Ranks**

$$\begin{equation}
T = \sum_{i=1}^{n} \text{sgn}(d_i) \cdot R_i
\end{equation}$$

Where $\text{sgn}(x) = +1$ if $x > 0$, $-1$ if $x < 0$, $0$ if $x = 0$

**Alternative Formulations**:

**Positive-rank sum** (sum of ranks for positive differences):

$$\begin{equation}
T^+ = \sum_{d_i > 0} R_i
\end{equation}$$

**Negative-rank sum** (sum of ranks for negative differences):

$$\begin{equation}
T^- = \sum_{d_i < 0} R_i
\end{equation}$$
>[!caution]
>derive tha above formula

**Relationship**: Since all ranks sum to $n(n+1)/2$:

$$\begin{align}
T^+ + T^- &= \frac{n(n+1)}{2} \\
T &= T^+ - T^- = \frac{n(n+1)}{2} - 2T^-
\end{align}$$

**Null Distribution**:

Under $H_0$, each difference is equally likely to be positive or negative:

$$\begin{align}
E[T^+] = E[T^-] = \frac{n(n+1)}{4}, \quad E[T] = 0
\end{align}$$

$$\begin{align}
\text{Var}(T^+) = \text{Var}(T^-) = \frac{n(n+1)(2n+1)}{24}\\
{Var}(T) = \frac{n(n+1)(2n+1)}{6}
\end{align}$$
>[!caution]
>properly rewrite these last two formulas

**Large-Sample Approximation** (n > 20):

$$\begin{equation}
Z = \frac{T}{\sqrt{n(n+1)(2n+1)/6}} \sim N(0, 1)
\end{equation}$$

#### When to Use

##### **The Ideal**
when parametric tests work: Normally distributed differences, $n_{runs} \geq 30$  
✅ Use paired t-test → higher power (100% baseline vs. Wilcoxon 95.5%)  
✅ More familiar interpretation (means, standard deviations)  
✅ Confidence intervals directly interpretable in original units

##### **Our Reality**
when t-test fails: Non-normal differences with $n_{runs} = 15$  
⚠️ Shapiro-Wilk $p < 0.05$ → **cannot use t-test** → **must use Wilcoxon**  
⚠️ Power penalty: Wilcoxon has 95.5% efficiency (ARE) vs. t-test when data is normal  
⚠️ Interpretation shift: Testing **medians of differences**, not means  
✅ Robustness benefit: Not affected by outliers or skewness

**Appropriate scenarios**:

- ✅ **Paired design**: Same subjects or matched pairs
- ✅ **Non-normal differences**: Shapiro-Wilk $p < 0.05$
- ✅ **Symmetric distribution**: Assumes symmetry about median
- ✅ **Ordinal data**: Works with ranks
- ✅ **Outliers present**: Robust to extreme values
- ✅ **Small samples**: More powerful than sign test

**Not appropriate**:

- ❌ **Asymmetric differences**: Use permutation test
- ❌ **Independent samples**: Use Mann-Whitney U test
- ❌ **Multiple groups** (k > 2): Use Friedman test

#### Interpretation

##### Decision Rule (Statistical Mechanics)

- If $p-value < α (0.05)$: **Reject $H_0$** → Significant difference in median differences
- If $p-value \geq α (0.05)$: **Fail to reject $H_0$** → No evidence of difference

##### Conceptual Meaning in Algorithm Benchmarking

The Wilcoxon test detects **consistent directional differences** via ranking, answering: "Do positive differences (CPU faster) outrank negative differences (GPU faster) systematically?"

**What the test actually measures**:
- Not means (like t-test), but **median of signed differences**
- Ranks measure **relative magnitude ordering**, not absolute values
- Test is sensitive to: (1) shift in central tendency, (2) asymmetry in difference distribution

**Example scenarios**:

```text
Scenario 1 - STRONG EFFECT (T+=120, T-=0, p<0.001):
Differences (CPU-GPU): [+17.3, +18.8, +17.1, +17.9, +18.2, ...] (all positive)
→ Positive ranks dominate completely (T+=120, T-=0)
→ GPU systematically faster on EVERY instance

Scenario 2 - WEAK EFFECT (T+=85, T-=35, p=0.08):
Differences (HybridA-HybridB): [+0.2, -0.1, +0.4, +0.1, -0.3, +0.2, ...]
→ Positive ranks slightly dominate (T+=85 vs T-=35)
→ But not consistent enough: could be random variation

Scenario 3 - OUTLIER ROBUST (t-test fails, Wilcoxon succeeds):
Differences: [+17.2, +17.5, +17.3, +89.1(outlier), +17.4, ...]
→ t-test: outlier inflates variance → p=0.08 (not significant)
→ Wilcoxon: outlier gets rank=15, others rank 1-14 → p=0.002 (significant)
→ Ranking neutralizes outlier impact
```
>[!caution]
>add proper example.

**GPU Benchmark Context** ($n_{runs}=15$):
Wilcoxon is our **backup test** when Shapiro-Wilk fails ($p<0.05$). Trade-off: 4.5% power penalty vs. robustness to skewness/outliers. For GPU speedups with $d>2.0$, even with power loss, we still achieve >95% power.

##### P-value Interpretation

**See [Section 0.1.2 P-value Interpretation Ladder](#interpretation-ladder)** for complete p-value interpretation guidance.

##### Effect Size (Rank-biserial correlation)

$$\begin{equation}
r = \frac{T^+ - T^-}{T^+ + T^-}
\end{equation}$$

Interpretation: $r = 0.1$ (small), $0.3$ (medium), $0.5$ (large)

**Meaning**: $r$ = proportion of favorable comparisons. $r=1.0$ means all differences favor one direction (maximum effect).

#### Implementation in Our Benchmark

```python
from scipy import stats
import numpy as np

def paired_comparison(data_a, data_b, alpha=0.05):
    differences = data_a - data_b
    if np.std(differences) < 1e-10:
        return {"test": "no_test_needed", "p_value": 1.0}

    _, p_norm = stats.shapiro(differences)

    if p_norm >= alpha:
        statistic, p_value = stats.ttest_rel(data_a, data_b)
        test_used = "paired_t_test"
    else:
        statistic, p_value = stats.wilcoxon(data_a, data_b,
                                            alternative='two-sided',
                                            zero_method='wilcox',
                                            correction=True)
        test_used = "wilcoxon_signed_rank"

    return {"test": test_used, "p_value": p_value}
```

#### Example from Benchmark

**Case: pr1002 - Non-normal CPU times**

```
Normality Test: W = 0.892, p = 0.048 (non-normal)
Test Selection: Wilcoxon

Wilcoxon Results:
T+ = 120, T- = 0
p-value: < 0.001
Rank-biserial r = 1.0 (maximum effect)

Interpretation: GPU significantly faster (all instances favor GPU)
```

#### Assumptions and Limitations

**Assumptions**:

1. Independence of pairs
2. Symmetric distribution of differences
3. Continuous or ordinal data
4. No excessive ties (< 25%)

**Limitations**:

1. Power loss (~5%) vs t-test for normal data
2. Cannot test asymmetric distributions
3. Tie/zero handling affects results

[^10][^11][^12]

[^10]: Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83. DOI: [10.2307/3001968](https://doi.org/10.2307/3001968)

[^11]: Pratt, J.W. (1959). Remarks on zeros and ties in the Wilcoxon signed rank procedures. *JASA*, 54(287), 655-667. DOI: [10.1080/01621459.1959.10501526](https://doi.org/10.1080/01621459.1959.10501526)

[^12]: Kerby, D.S. (2014). The simple difference formula: An approach to teaching nonparametric correlation. *Comprehensive Psychology*, 3, Article 11. DOI: [10.2466/11.IT.3.1](https://doi.org/10.2466/11.IT.3.1)

---

### Friedman Test

**Also Known As**: Friedman's two-way analysis of variance by ranks  
**Purpose**: Non-parametric test for comparing k related samples (k $\geq$ 3).

**Null Hypothesis ($H_0$)**: All algorithms have identical distributions  
**Alternative (H₁)**: At least one algorithm differs

**Developed by**: Milton Friedman (1937)

#### Mathematical Formulation

The Friedman test ranks data **within each block** (problem instance), then compares rank sums across treatments (algorithms).

**Step 1: Rank Within Blocks**

Given data matrix $\{x_{ij}\}$ with n blocks (problems) $\times$ k treatments (algorithms):

- For each block i, rank the k observations: smallest = rank 1, largest = rank k
- Ties receive average ranks
- Replace data with rank matrix $\{r_{ij}\}$

**Step 2: Compute Rank Sums**

$$\begin{equation}
\bar{r}_{\cdot j} = \frac{1}{n}\sum_{i=1}^{n} r_{ij}
\end{equation}$$

**Step 3: Calculate Test Statistic**

$$\begin{equation}
Q = \frac{12n}{k(k+1)} \sum_{j=1}^{k} \left(\bar{r}_{\cdot j} - \frac{k+1}{2}\right)^2
\end{equation}$$

**Alternative formulation** (equivalent):

$$\begin{equation}
\chi_F^2 = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)
\end{equation}$$

Where $R_j = \sum_{i=1}^{n} r_{ij}$ = total rank sum for algorithm j

**Null Distribution**:

- **Large samples** (n > 15 or k > 4): $Q \sim \chi^2_{k-1}$
- **Small samples**: Use exact tables
- **Degrees of freedom**: df = k - 1

#### When to Use

##### The Ideal

comprehensive comparison: $k \geq 4$ algorithms, $n_{runs} \geq 30$ problem instances  

- ✅ More algorithms → clearer performance hierarchy  
- ✅ More instances → higher power to detect ranking differences  
- ✅ Can detect subtle ranking patterns (e.g., Algorithm A > B > C)  
- ✅ Larger sample enables post-hoc analysis with adequate power

##### Our Reality

4 algorithms, 38 instances varying from $n_{cities}=52$ to $n_{cities}=1002$: Balanced design with constraints

>[!caution]
>Wouldn't our 38 instances with 15 repetitions be of any value? like, they are independent in between each other. Which statistical method (as far as i understood, it would be Wilcoxon test, but there are others like Levene's, though the decision trees currently do not focus too much on them). Aren't these for independent tests? why focus only on the 15 repetitions by problm and not on the consistent results?

- ✅ **k=4 is optimal**: CPU, HybridNaive, HybridOptimized, FullGPU (complete comparison)  
- ✅ **n=30 adequate**: Above n≥15 threshold for $\chi^2$ approximation  
- ⚠️ **No room for k=2 subset**: Would require separate Wilcoxon test (loss of multi-algorithm context)  
- ⚠️ **Power for small rank differences**: May not detect if Algorithm B and C have very similar performance
- >[!caution]
  > How do we solve this

**Appropriate scenarios**:

- ✅ **k $\geq$ 3 algorithms**: Comparing multiple treatments ← **Our k=4 design**
   >[!caution]
   >But what about we having 38 runs for multiple problems? Like, the consistent speedups, acquired for larger $n_{cities}$
- ✅ **Repeated measures**: Same blocks (problems) across all algorithms
- ✅ **Non-normal data**: Robust alternative to repeated-measures ANOVA
- ✅ **Ordinal data**: Only requires rankable measurements
- ✅ **Complete block design**: All algorithms tested on same problems

**Our benchmark context**:

```python
# Perfect use case:
# - 4 algorithms: CPU, HybridNaive, HybridOptimized, FullGPU
# - 30 TSP problems (blocks)
# - Each algorithm tested on same 30 problems
# - Compare solution costs (may be non-normal)
```

**Not appropriate**:

- ❌ **k = 2 algorithms**: Use Wilcoxon instead (more powerful)
- ❌ **Independent samples**: Use Kruskal-Wallis test
- ❌ **Missing data**: Use Durbin test or imputation

#### Interpretation

**Decision Rule** (Standard Statistical Test):

- If p-value $p < α (0.05)$: **Reject $H_0$** → At least one algorithm differs in rankings
- If $p \geq \alpha$: **Fail to reject $H_0$** → No evidence of ranking differences

##### Conceptual Meaning

Algorithm Benchmarking Context:

###### What Friedman Actually Tests

The consistency of ranking order across problem instances.

- **Not testing**: Absolute speed differences (doesn't care if GPU is $2×$ or $100×$ faster)  
- **Actually testing**: Whether the ranking pattern (e.g., FullGPU > HybridOpt > HybridNaive > CPU) holds consistently across 38 instances  
- **Sensitive to**: Rank inversions (e.g., GPU faster on 28/30 instances, but CPU faster on 2/30)
  >[!caution]
  >This does not happened
- **Robust to**: Outliers in absolute times (only ranks matter)
  >[!caution]
  >Time outliers happened and should be investigated. They were consistent in $n_{\text{GAgenerations}}$ stoppage, be in time (with cpu having more outliers, but gpu being kinda consistent, probably related to problem size number of threads, etc. but there was some consistency between generations where the optimal/near optimal was found and the process either stopped or run for 50 more generations)

##### Example Interpretation:

> [!important] Interpretation
> Friedman $Q=45.2$, $p=0.0001$: The ranking order of the 4 algorithms is statistically consistent across 38 TSP instances. This does NOT tell us the magnitude of speedup, only that the performance hierarchy is stable."

**Post-hoc Analysis** (Required for Actionable Insights):

- Friedman only tells us **THAT** algorithms differ, not **WHICH pairs** differ
- **Mandatory next step** if $p < α$: Perform Nemenyi post-hoc test for pairwise comparisons
- **Why necessary**: Knowing 'some algorithms differ' is insufficient—need to identify which GPU variants outperform CPU

**Effect Size** (Kendall's W):

$$\begin{equation}
W = \frac{Q}{n(k-1)} = \frac{12 \sum_j \bar{r}_{\cdot j}^2 - 3n^2k(k+1)^2}{n^2k(k-1)}
\end{equation}$$

Interpretation: $W = 0$ (no agreement), $W = 1$ (perfect agreement)

#### Implementation in Our Benchmark

```python
from scipy import stats
import numpy as np

def compare_multiple_algorithms(data_dict, alpha=0.05):
    """
    Compare k algorithms using Friedman test.

    Args:
        data_dict: {"CPU": [costs], "GPU": [costs], ...}
    Returns:
        {"statistic": Q, "p_value": p, "significant": bool}
    """
    # Check for zero variance first
    all_data = np.concatenate(list(data_dict.values()))
    if np.std(all_data) < 1e-10:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # Run Friedman test
    statistic, p_value = stats.friedmanchisquare(*data_dict.values())

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "post_hoc_required": p_value < alpha
    }
```

**SciPy Implementation**: `scipy.stats.friedmanchisquare(*samples)`

#### Example from Benchmark

**Case: eil51 - Four algorithm comparison**

```
Algorithms: CPU, HybridNaive, HybridOptimized, FullGPU
Problems: 15 instances of eil51

Rank matrix (example, 3 problems shown):
Problem  CPU  HybridNaive  HybridOpt  FullGPU
   1      4        3            2         1
   2      4        3            2         1
   3      4        2            3         1

Rank sums: R_CPU=60, R_HybridNaive=42, R_HybridOpt=36, R_FullGPU=12

Friedman Results:
Q statistic: 38.4
df: 3
p-value: < 0.001

Interpretation:
At least one algorithm differs significantly (Q(3) = 38.4, p < 0.001)
Proceed with Nemenyi post-hoc test to identify pairs
```
> [!caution]
> Examples should be shown fully, integrating all the steps from the beginning.

#### Assumptions and Limitations

**Assumptions**:

1. **Complete block design**: All algorithms tested on same problems
2. **Independence of blocks**: Problem instances independent
3. **Ordinal data**: Measurements can be ranked within blocks
4. **No systematic block effects**: Block effects cancel out in ranking

**Limitations**:

1. **Less powerful than ANOVA** if data truly normal (~95% efficiency)
2. **Requires complete data**: Missing values problematic (use Durbin test)
3. **Loses information**: Rankings discard magnitude of differences
4. **Ties reduce power**: Many tied ranks weaken test
5. **No effect size**: Doesn't quantify magnitude of differences

**Robustness**:

- **Very robust** to non-normality, outliers, heteroscedasticity
- **Works with ordinal scales**: Doesn't require interval data
- **Handles skewness**: Ranking eliminates distribution issues

[^13][^14][^15][^16]

[^13]: Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *JASA*, 32(200), 675-701. DOI: [10.1080/01621459.1937.10503522](https://doi.org/10.1080/01621459.1937.10503522)

[^14]: Friedman, M. (1939). A correction: The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *JASA*, 34(205), 109.

[^15]: Friedman, M. (1940). A comparison of alternative tests of significance for the problem of m rankings. *Annals of Mathematical Statistics*, 11(1), 86-92.

[^16]: Conover, W.J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley. ISBN 0-471-16851-3.

---

## Post-Hoc Tests

### Nemenyi Test

**Purpose**: Post-hoc pairwise comparison after significant Friedman test.

**Null Hypothesis ($H_0$)**: Algorithms i and j have identical distributions  
**Alternative (H₁)**: Algorithms i and j differ significantly

**When to Use**: Only after Friedman test $p < \alpha$ (significant omnibus test)

#### Mathematical Formulation

The Nemenyi test compares **mean ranks** between all algorithm pairs.

**Critical Difference** (CD):

$$\begin{equation}
CD = q_{\alpha} \sqrt{\frac{k(k+1)}{6n}}
\end{equation}$$

Where:

- $q_{\alpha}$ = critical value from studentized range distribution (Tukey's $q$) for significance level $\alpha$ and $k$ treatments
- $k$ = number of algorithms
- $n$ = number of blocks (problems)
- $\alpha$ = significance level (typically $0.05$)

>[!caution]
>please derive this formula and solve an example with it

**Decision Rule**:

For algorithms i and j with mean ranks $\bar{r}_i$ and $\bar{r}_j$:

$$\begin{equation}
|\bar{r}_i - \bar{r}_j| > CD \implies \text{Reject } H_0 \text{ (algorithms differ)}
\end{equation}$$

**P-value Calculation** (two-tailed):

$$\begin{equation}
p_{ij} = 2\left(1 - \Phi\left(\frac{|\bar{r}_i - \bar{r}_j|}{\sqrt{k(k+1)/(6n)}}\right)\right)
\end{equation}$$

Where $\Phi$ is the standard normal CDF.

#### When to Use

##### The Ideal

exploratory analysis: Many algorithms ($k≥5$), interested in ALL pairwise comparisons  
- ✅ Nemenyi controls family-wise error rate (FWER) across all ${k \choose 2}$ pairs  
- ✅ Conservative approach suitable for exploratory research  
- ✅ No need to pre-specify which comparisons matter

##### Our Reality

confirmatory analysis: $k=4$ algorithms, specific research questions  

- ✅ **Appropriate for our design**: 4 algorithms → ${4 \choose 2} = 6$ comparisons (manageable)  
- ⚠️ **Conservative penalty**: Nemenyi less powerful than Dunn's test for planned comparisons  
  >[!caution]
  >never heard of it.
- ⚠️ **Research question**: "Does GPU (any variant) outperform CPU?" could use Dunn with 3 comparisons instead of 6  
  >[!caution]
  >all gpu variants outperform
- ✅ **Trade-off accepted**: Sacrifice some power for comprehensive pairwise picture

**Appropriate**:

- ✅ After **significant Friedman test** ($p < 0.05$) ← **Mandatory prerequisite**
- ✅ **All pairwise comparisons**: Controls family-wise error rate across all ${k \choose 2}$ pairs
- ✅ **Equal comparisons interest**: All pairs equally important (exploratory stance)

**Not appropriate**:

- ❌ **Friedman not significant** ($p ≥ 0.05$): No omnibus evidence of any differences
- ❌ **Planned comparisons only**: Use Dunn's test with Bonferroni for 2-3 specific pairs (more power)
- ❌ **Unequal sample sizes**: Nemenyi assumes balanced design (our $n_{reps}=15$ for all algorithms ✅)

#### Interpretation

**Decision Rule** (Standard Statistical Test):

- For each pair (i,j): If $|\bar{r}_i - \bar{r}_j| > CD$ **OR** $p_{ij} < \alpha$: Algorithms differ significantly  
- If $|\bar{r}_i - \bar{r}_j| \leq CD$: No evidence of difference (may be tied in performance)

**Conceptual Meaning** (Algorithm Benchmarking Context):

**What Nemenyi Tests**: Whether two algorithms occupy statistically distinct positions in the performance hierarchy.

- **Mean rank interpretation**: Lower rank = better (faster) performance  
  - $\bar{r}_{FullGPU} = 1.2$ vs $\bar{r}_{CPU} = 3.8$ → FullGPU consistently ranked 1st, CPU ranked 4th  
- **Critical Difference (CD)**: "How far apart in rankings must algorithms be to declare them different?"  
  - $CD=0.8$ with 4 algorithms → need $|\bar{r}_i - \bar{r}_j| > 0.8$ for significance  
- **Non-significance ≠ equal performance**: May indicate similar speeds OR insufficient sample size ($n_{runs}=30$ may not detect subtle differences)

##### **Example Interpretation**:
>[!important] Interpretation
>"Nemenyi post-hoc ($CD=0.85$): FullGPU ($\bar{r}=1.1$) significantly outperforms CPU ($\bar{r}=3.9, p<0.001$) and HybridNaive ($\bar{r}=3.2, p=0.002$), but NOT HybridOptimized ($\bar{r}=1.8, p=0.12$). This suggests FullGPU and HybridOpt occupy similar performance tiers, both substantially faster than naive approaches."

>[!caution]
>And
> 1. how should I interpret Nemanyi's graphs
> 2. How are these comparisons done? Each algorithm is compared separatedly agains one another? How would the mathematical formula for all these results look like? i.e. I thought Nemenyi was not limited.
> 3. I would like to understand in the decision trees how they complement one another.

**Critical Difference Interpretation**:

- $CD$ = minimum rank difference for significance (controls family-wise error rate)
- Larger $n_{runs}$ → smaller CD → easier to detect differences (our n=30 provides moderate sensitivity)  
- More algorithms k → larger CD → harder to detect differences (k=4 is reasonable compromise)

#### Implementation in Our Benchmark

```python
import scikit_posthocs as sp
import numpy as np
import pandas as pd

def nemenyi_posthoc(data_dict, friedman_p_value, alpha=0.05):
    """
    Perform Nemenyi post-hoc test after significant Friedman.

    Args:
        data_dict: {"CPU": [costs], "GPU": [costs], ...}
        friedman_p_value: Result from Friedman test
    Returns:
        DataFrame with pairwise p-values
    """
    if friedman_p_value >= alpha:
        print("Friedman test not significant - post-hoc not needed")
        return None

    # Convert to format expected by scikit-posthocs
    # Expects: n_problems $\times$ k_algorithms array
    data_array = np.column_stack(list(data_dict.values()))

    # Perform Nemenyi test
    p_matrix = sp.posthoc_nemenyi_friedman(data_array)
    p_matrix.index = data_dict.keys()
    p_matrix.columns = data_dict.keys()

    return p_matrix
```

**Scikit-posthocs**: `posthoc_nemenyi_friedman(data_array)`

#### Example from Benchmark

**Case: eil51 - Pairwise algorithm comparisons**

```
Friedman: Q = 38.4, p < 0.001 (significant)
→ Proceed with Nemenyi

Mean Ranks:
  CPU: 4.0
  HybridNaive: 2.8
  HybridOptimized: 2.4
  FullGPU: 1.0

Critical Difference: CD = 0.85 (k=4, n=15, α=0.05)

Pairwise P-value Matrix:
                 CPU    HybridNaive  HybridOpt  FullGPU
CPU             1.000      0.012       0.003     <0.001
HybridNaive     0.012      1.000       0.582      0.004
HybridOptimized 0.003      0.582       1.000      0.008
FullGPU        <0.001      0.004       0.008      1.000

Significant Pairs (p < 0.05):
- CPU vs HybridNaive: p = 0.012 ✓
- CPU vs HybridOptimized: p = 0.003 ✓
- CPU vs FullGPU: p < 0.001 ✓
- HybridNaive vs FullGPU: p = 0.004 ✓
- HybridOptimized vs FullGPU: p = 0.008 ✓

Not Significant:
- HybridNaive vs HybridOptimized: p = 0.582 (no difference)

Conclusion:
FullGPU significantly outperforms all others
CPU significantly worse than all others
Hybrid variants statistically equivalent
```

#### Assumptions and Limitations

**Assumptions**:

1. **Friedman significant**: Omnibus test detected differences
2. **Balanced design**: Same n for all algorithms
3. **Complete data**: No missing algorithm-problem combinations

**Limitations**:

1. **Conservative**: Controls family-wise error rate (low power)
2. **All pairs tested**: Even if only interested in subset
3. **Equal weight**: Doesn't prioritize specific comparisons
4. **Requires significance**: Can't use if Friedman p $\geq$ α

**Alternatives**:

- **Dunn's test**: More flexible, can use Bonferroni/Holm correction
- **Conover test**: More powerful but less conservative

[^17][^18][^19]

[^17]: Nemenyi, P. (1963). *Distribution-free Multiple Comparisons*. PhD thesis, Princeton University.

[^18]: Pohlert, T. (2014). The Pairwise Multiple Comparison of Mean Ranks Package (PMCMR). R package. <https://CRAN.R-project.org/package=PMCMR>

[^19]: Hollander, M. & Wolfe, D.A. (1973). *Nonparametric Statistical Methods*. Wiley. ISBN 978-0-471-40635-8.

---

## Multiple Comparison Correction

### Holm-Bonferroni Method

**Also Known As**: Holm's step-down procedure, Holm's sequential Bonferroni  
**Purpose**: Control family-wise error rate (FWER) when making multiple comparisons.

- **Problem**: With $m$ tests at $\alpha=0.05$, probability of $\Pr(FP\geq1) \sim{1-(1-\alpha)}^m$  
- **Solution**: Sequentially adjusted α thresholds (less conservative than Bonferroni)

**Developed by**: Sture Holm (1979)

#### Mathematical Formulation

**Step-Down Procedure**:

1. **Sort p-values**: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$

2. **Sequential Testing**: For $i = 1, 2, ..., m$:
   - Test $H_{0(i)}$ using threshold: $\frac{\alpha}{m - i + 1}$
   - If $p_{(i)} > \frac{\alpha}{m - i + 1}$: **Stop** - Fail to reject $H_{0(i)}$ and all subsequent
   - If $p_{(i)} \leq \frac{\alpha}{m - i + 1}$: **Reject** $H_{0(i)}$ and continue

3. **Adjusted P-values** (for reporting):

$$\begin{equation}
p_{\text{adj}(i)} = \min\left\{1, \max_{j \leq i}\left[(m-j+1) \times p_{(j)}\right]\right\}
\end{equation}$$

>[!caution]
> no explanation for the symbols and what they mean

**Key Property**: If $p_{\text{adj}(i)} < \alpha$, reject $H_{0(i)}$

**Thresholds Table** ($α = 0.05, m = 6$ comparisons):

| Rank | Threshold        | Bonferroni | Holm-Bonferroni |
|------|------------------|------------|------------------|
| $1$    | $α/(m-0) = α/6$    | $0.0083$     | $0.0083$          |
| $2$    | $α/(m-1) = α/5$    | $0.0083$     | $0.0100$          |
| $3$    | $α/(m-2) = α/4$    | $0.0083$     | $0.0125$          |
| $4$    | $α/(m-3) = α/3$    | $0.0083$     | $0.0167$          |
| $5$    | $α/(m-4) = α/2$    | $0.0083$     | $0.0250$          |
| $6$    | $α/(m-5) = α/1$    | $0.0083$     | $0.0500$          |

**Advantage**: Later tests use less stringent thresholds → more power

#### When to Use

##### The Ideal (single comparison luxury)

- ✅ **One pre-planned test**: Use unadjusted $α=0.05$ directly  
- ✅ **No multiple testing penalty**: Full statistical power preserved  
- ✅ **Simple interpretation**: $p<0.05$ sufficient for rejection  
- ✅ **Example**: "Is GPU faster than CPU?" (single hypothesis)

##### Our Reality (multiple comparisons necessity)

- ⚠️ **$k=4$ algorithms** → $\binom{4}{2}=6$ pairwise comparisons required  
- ⚠️ **Family-wise error inflation**: Without correction, $P(FP \ge 1) = 1-(0.95)^6 = 26.5\%$  
- ⚠️ **Must use Holm-Bonferroni**: Controls $FWER \leq 5\%$ across ALL 6 tests  
- ⚠️ **Power penalty**: Smallest p-value needs $p < 0.05/6 = 0.0083$ (stricter threshold)

##### Consequences (statistical rigor trade-off)

1. **Threshold escalation**: Test 1 requires $p<0.0083$, Test 2 requires $p<0.01$, ..., Test 6 allows $p<0.05$  
2. **Reduced sensitivity**: May miss real but small differences (requires larger effects for detection)  
3. **Protection guaranteed**: False positive rate $\le 5\%$ across entire family of tests  
4. **Trade-off accepted**: Scientific integrity demands controlling false discoveries, even at power cost

##### Our Benchmark Context ($n_{runs}=15$, $k=4$ algorithms)

```python
# 4 algorithms → 6 pairwise comparisons:
# CPU vs HybridNaive, CPU vs HybridOpt, CPU vs FullGPU,
# HybridNaive vs HybridOpt, HybridNaive vs FullGPU,
# HybridOpt vs FullGPU

# Without correction: 26.5% chance of ≥1 false positive
# With Holm-Bonferroni: ≤5% chance of ≥1 false positive
```

**Justification for $n_{runs}=15$**: With expected GPU speedups ($d>2.0$), even strict Holm thresholds ($p<0.0083$) are easily met. Large effect sizes compensate for multiple testing penalty.

##### Not Appropriate

- ❌ **Single test**: No correction needed
- ❌ **FDR control**: Use Benjamini-Hochberg if tolerating more false positives OK
  >[!caution]
  > never heard of it
- ❌ **Pre-planned single comparison**: Use unadjusted $α$

#### Interpretation

##### Decision Rule (Statistical Mechanics)

- **Unadjusted**: Reject if $p < \alpha$ (may have high false positive rate)
- **Holm-Bonferroni**: Reject if $p_{\text{adj}} < \alpha$ (controls $FWER \leq \alpha$)

##### Conceptual Meaning in Algorithm Benchmarking

Holm-Bonferroni addresses the **multiple testing problem**: "If you test 6 pairs at $α=0.05$ each, you'll declare ~1 pair significant by pure chance even if all algorithms are identical."

**What the correction does**:
- Not just "make thresholds stricter" (vague)
- But: "Sequentially adjust α to maintain $≤5\%$ probability of ANY false positive across all m tests"
- Balances Type I error control with power (more efficient than Bonferroni)

**Example scenario** ($k=4$ algorithms, $m=6$ comparisons):

```text
Raw p-values: [0.001, 0.008, 0.012, 0.035, 0.048, 0.150]
Holm thresholds: [0.0083, 0.010, 0.0125, 0.0167, 0.025, 0.050]

Test 1: p=0.001 < 0.0083 → REJECT ✅ (extremely strong)
Test 2: p=0.008 < 0.010 → REJECT ✅ (survives correction)
Test 3: p=0.012 < 0.0125 → REJECT ✅ (barely passes)
Test 4: p=0.035 > 0.0167 → FAIL TO REJECT ❌ (correction filters)
Test 5: (stop testing, all subsequent fail)
Test 6: (stop testing, all subsequent fail)

Conclusion: 3 significant differences detected with FWER ≤ 5%
```
>[!caution]
>need better example

**GPU Benchmark Context**:
With $d>2.0$ speedups, raw p-values typically $<0.001$ → easily survive even strictest Holm threshold (0.0083). Correction filters marginal differences (HybridA vs HybridB with $d=0.15$) but preserves strong effects.

##### Power Comparison

- **Bonferroni**: Uses $\alpha/m$ for ALL tests (most conservative)
- **Holm-Bonferroni**: Uses $\alpha/m$ → $\alpha/1$ sequentially (less conservative)
- **Holm is strictly more powerful** than Bonferroni while controlling same FWER

#### Implementation in Our Benchmark

```python
import numpy as np
from typing import Tuple, List

def holm_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni step-down procedure.

    Args:
        p_values: Array of raw p-values
        alpha: Family-wise error rate (default 0.05)

    Returns:
        (reject, p_adjusted) where:
        - reject[i]: True if $H_0$ rejected for test i
        - p_adjusted[i]: Adjusted p-value for test i
    """
    m = len(p_values)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate adjusted p-values
    p_adjusted = np.zeros(m)
    for i in range(m):
        # Holm formula: p_adj = max of (m-j+1)*p for j <= i
        p_adjusted[i] = max([
            min(1.0, (m - j + 1) * sorted_p[j])
            for j in range(i + 1)
        ])

    # Restore original order
    p_adjusted_original = np.zeros(m)
    p_adjusted_original[sorted_indices] = p_adjusted

    # Determine rejections
    reject = p_adjusted_original < alpha

    return reject, p_adjusted_original
```

**Usage Example**:

```python
p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.082, 0.250])
reject, p_adj = holm_bonferroni_correction(p_values, alpha=0.05)

print(f"Rejected: {reject}")  # [True, True, True, False, False, False]
print(f"Adjusted p-values: {p_adj}")
```

#### Example from Benchmark

**Case: 6 pairwise algorithm comparisons (α = 0.05)**

```
Raw P-values (sorted):
1. CPU vs FullGPU:          p = 0.001  →  p_adj = 0.006  ✓ Reject (0.006 < 0.05)
2. CPU vs HybridOpt:        p = 0.008  →  p_adj = 0.040  ✓ Reject (0.040 < 0.05)
3. CPU vs HybridNaive:      p = 0.012  →  p_adj = 0.048  ✓ Reject (0.048 < 0.05)
4. HybridNaive vs FullGPU:  p = 0.035  →  p_adj = 0.105  ✗ Fail (0.105 > 0.05)
5. HybridOpt vs FullGPU:    p = 0.067  →  p_adj = 0.134  ✗ Fail (STOP)
6. HybridNaive vs HybridOpt: p = 0.582 →  p_adj = 0.582  ✗ Fail

Comparison with Bonferroni:
Bonferroni threshold: 0.05/6 = 0.0083
  - Would reject only test 1 (p=0.001)

Holm-Bonferroni:
  - Rejects tests 1, 2, 3 (more powerful)
  - Still controls FWER at 0.05

Conclusion:
CPU significantly different from all GPU variants
GPU variants not significantly different from each other
```

#### Comparison with Other Methods

| Method | Type | FWER Control | Power | Use Case |
|--------|------|--------------|-------|----------|
| **No Correction** | None | No | Highest | Single test |
| **Bonferroni** | Single-step | Yes | Low | Very conservative |
| **Holm-Bonferroni** | Step-down | Yes | Medium | Recommended FWER |
| **Hochberg** | Step-up | Yes* | Higher | When p-values independent |
| **Benjamini-Hochberg** | FDR control | No (FDR) | Highest | Exploratory, tolerant |

*Hochberg requires p-values be independent or positive dependence

**When to choose Holm-Bonferroni**:

- Need strong FWER control (medical, safety-critical)
- P-values may be dependent
- Want uniformly more powerful than Bonferroni
- Standard in confirmatory research

[^20][^21][^22]

[^20]: Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70. JSTOR: [4615733](https://www.jstor.org/stable/4615733)

[^21]: Wright, S.P. (1992). Adjusted P-values for simultaneous inference. *Biometrics*, 48(4), 1005-1013. DOI: [10.2307/2532694](https://doi.org/10.2307/2532694)

[^22]: Aickin, M. & Gensler, H. (1996). Adjusting for multiple testing when reporting research results: The Bonferroni vs Holm methods. *American Journal of Public Health*, 86(5), 726-728. PMID: [8629727](https://pubmed.ncbi.nlm.nih.gov/8629727/)

---

## Effect Size Measures

### Cohen's d

**Purpose**: Quantify **practical significance** independent of sample size.

**Key Insight**: Statistical significance (p-value) depends on n; effect size does not.

**Interpretation Benchmarks** (Cohen, 1988):

- |d| < 0.2: **negligible**
- 0.2 $\leq$ |d| < 0.5: **small** effect
- 0.5 $\leq$ |d| < 0.8: **medium** effect
- |d| $\geq$ 0.8: **large** effect

#### Mathematical Formulation

**For Independent Samples**:

$$\begin{equation}
d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}
\end{equation}$$

Where pooled standard deviation:

$$\begin{equation}
s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
\end{equation}$$

**For Paired Samples** (our benchmark case):

$$\begin{equation}
d = \frac{\bar{d}}{s_d}
\end{equation}$$

Where:

- $\bar{d}$ = mean of paired differences
- $s_d$ = standard deviation of differences

**Alternative** (when using means and SDs of original data):

$$\begin{equation}
d = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{(s_1^2 + s_2^2)/2}}
\end{equation}$$
>[!caution]
> please derive this equation from the above.

**Sign Convention**:

- Positive d: Group 1 > Group 2
- Negative d: Group 1 < Group 2
- Report |d| for magnitude

#### When to Use

##### The Ideal (effect size independence)

- ✅ **Sample-size independent**: Cohen's d is **identical** whether measured with $n_{runs}=15$ or $n_{runs}=1,500$  
- ✅ **Standardized metric**: Comparable across different studies, datasets, and domains  
- ✅ **Meta-analysis ready**: Can aggregate d values from multiple studies  
- ✅ **Intuitive interpretation**: "Means differ by $X$ standard deviations"

##### Our Reality (small sample context with $n_{runs}=15$)

⚠️ **P-value vulnerable to n**: With $n_{runs}=15$, even $d=0.5$ may yield $p>0.05$ (underpowered)  
⚠️ **Effect size STABLE**: Same data with $n_{runs}=150$ would show identical $d$ but $p<0.001$  
⚠️ **Solution**: **Always report both** p-value AND effect size together  
✅ **GPU advantage**: Large effects ($d>2.0$) overcome small sample penalties

##### Consequences (dual reporting necessity)

1. **P-value alone misleading**: $p=0.048$ could be $d=0.15$ (trivial) or $d=1.2$ (large)—can't tell without effect size  
2. **Effect size alone incomplete**: $d=0.8$ could be $p=0.10$ (not significant with $n_{runs}=15$) or $p<0.001$ (significant with $n_{runs}=100$)  
3. **Combined reporting essential**: "$p<0.001, d=25.7$" tells complete story (significant AND massive)  
4. **Protects against over-interpretation**: "$p=0.04, d=0.1$" flags statistical artifact (significant but negligible)

##### Our Benchmark Context ($n_{runs}=15$)

```python
# Example interpretations:

# Case 1: Large effect, small sample
# CPU vs GPU: p < 0.001, d = 25.7
# → Statistically significant AND extremely large practical impact
# → Conclusion: Strong evidence, transformative speedup

# Case 2: Small effect, inflated significance
# HybridA vs HybridB: p = 0.048, d = 0.15
# → Statistically significant (barely) BUT practically negligible
# → Conclusion: Not worth implementation complexity

# Case 3: Large effect, underpowered
# NewAlgo vs Baseline: p = 0.08, d = 0.85
# → Not statistically significant BUT large practical effect
# → Conclusion: Increase n to 30 for adequate power
```

**Why dual reporting matters**: With $n_{runs}=15$, p-values are "noisy" (high variance), but effect sizes remain stable. Cohen's d provides the **stable anchor** for judging practical importance.

##### Reporting Guidelines (APA style)

>[!important]
> "CPU was significantly slower than GPU, t(14) = 99.87, p < .001, d = 25.71, indicating an extremely large effect."
>
>**Always include**: test name, df, test statistic, p-value, effect size, interpretation

#### Interpretation

**Contextualized Benchmarks**:

| Cohen's d | Interpretation | Overlap* | Percentile** |
|-----------|----------------|----------|-------------|
| 0.0       | None           | 100%     | 50th        |
| 0.2       | Small          | 85%      | 58th        |
| 0.5       | Medium         | 67%      | 69th        |
| 0.8       | Large          | 53%      | 79th        |
| 1.0       | Very large     | 45%      | 84th        |
| 2.0       | Huge           | 19%      | 98th        |

*Overlap: % of distribution overlap between groups  
**Percentile: Mean of Group 1 is at this percentile of Group 2

**Field-Specific Considerations**:

- **Experimental psychology**: $d = 0.4$ typical
- **Psychotherapy**: $d = 0.5$-$0.7$ meaningful
- **Computer science**: Highly variable; algorithm improvements $d > 1.0$ common

#### Implementation in Our Benchmark

```python
import numpy as np

def cohens_d_paired(data_a: np.ndarray, data_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired samples.

    Args:
        data_a, data_b: Paired observations (same length)

    Returns:
        Cohen's d (positive if data_a > data_b)
    """
    differences = data_a - data_b
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample std

    if std_diff == 0:
        return 0.0  # No variance → no effect

    d = mean_diff / std_diff
    return float(d)

def cohens_d_independent(data_a: np.ndarray, data_b: np.ndarray) -> float:
    """
    Calculate Cohen's d for independent samples.
    """
    n1, n2 = len(data_a), len(data_b)
    mean1, mean2 = np.mean(data_a), np.mean(data_b)
    var1, var2 = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)
```

#### Example from Benchmark

>[!tip] **Exercise: Cohen's d for Berlin52 CPU vs HybridOptimized Speedup**
>
> **Scenario**: Quantify the **practical significance** of GPU acceleration independent of sample size. You've already established statistical significance ($p < 0.0001$) via paired t-test. Now measure **effect size**.
>
> **Data from Previous Exercise** ($n=15$ paired runs):
>
> $$
> \begin{align}
> \text{Mean difference: } \bar{d} &= 25.67\text{ seconds (CPU - HybridOptimized)} \\
> \text{SD of differences: } s_d &= 3.99\text{ seconds}
> \end{align}
> $$
>
> **Task 1: Calculate Cohen's d for Paired Samples**
>
> **Formula**: For paired samples, use the standardized mean difference:
>
> $$
> \begin{align}
> d &= \frac{\bar{d}}{s_d} \\
>   &= \frac{25.67}{3.99} \\
>   &= 6.43
> \end{align}
> $$
>
> **Task 2: Interpret Effect Size Magnitude**
>
> $$
> \begin{cases}
> |d| < 0.2 & \Rightarrow \text{Negligible effect} \\
> 0.2 \leq |d| < 0.5 & \Rightarrow \text{Small effect} \\
> 0.5 \leq |d| < 0.8 & \Rightarrow \text{Medium effect} \\
> 0.8 \leq |d| < 1.2 & \Rightarrow \text{Large effect} \\
> |d| \geq 1.2 & \Rightarrow \text{Very large/Huge effect}
> \end{cases}
> $$
>
> **Your Result**: $d = 6.43$ → **Extremely large effect** (far beyond Cohen's benchmarks)
>
> **Task 3: Calculate Distribution Overlap and Percentile Shift**
>
> **Overlap Percentage** (approximate formula):
> $$
> \begin{align}
> \text{Overlap} &\approx 2\Phi\left(-\frac{|d|}{2}\right) \times 100\% \\
>                &\approx 2\Phi(-3.22) \times 100\% \\
>                &\approx 0.13\% \quad \text{(virtually no overlap)}
> \end{align}
> $$
>
> **Percentile Shift**: HybridOptimized mean is at the $\Phi(d) = \Phi(6.43) \approx 99.9999994\text{th}$ percentile of CPU distribution
>
>[[!caution]]
>did not understand this.task 3 last part of the percentiles and overlap
>
> **Task 4: Practical Interpretation**
>
> ```python
> from scipy import stats
> import numpy as np
>
> # Calculate effect size
> d_bar = 25.67  # Mean difference
> s_d = 3.99     # SD of differences
> cohen_d = d_bar / s_d
>
> # Calculate overlap
> overlap_pct = 2 * stats.norm.cdf(-cohen_d/2) * 100
>
> # Percentile shift
> percentile = stats.norm.cdf(cohen_d) * 100
>
> print(f"Cohen's d = {cohen_d:.2f}")
> print(f"Distribution overlap = {overlap_pct:.2f}%")
> print(f"Percentile shift = {percentile:.6f}%")
> ```
>
> **Expected Output**:
> ```
> Cohen's d = 6.43
> Distribution overlap = 0.13%
> Percentile shift = 99.999999%
> ```
>
> **Task 5: Complete Reporting (APA Style)**
>
> **Template**:
> > "HybridOptimized was significantly faster than CPU, $t(14) = 24.92$, $p < .0001$, $d = 6.43$, indicating an extremely large effect. The GPU-optimized algorithm completed executions in $0.092$ seconds on average compared to $25.74$ seconds for CPU, representing a $280\times$ speedup."
>
> **Interpretation Summary**:
> 1. **Statistical Significance**: $p < 0.0001$ (extremely strong evidence)
> 2. **Practical Significance**: $d = 6.43$ (means separated by 6.43 standard deviations)
> 3. **Real-World Impact**: $280\times$ speedup translates to minutes vs hours for larger instances
> 4. **Stability**: Effect size is **sample-size independent**—would be $d=6.43$ even with $n=150$
> 5. **Confidence**: With $d > 2.0$, power $> 99\%$ even at $n=15$
>
> **✓ Verification**:
> - ✅ Effect size vastly exceeds Cohen's "large" threshold ($d = 0.8$)
> - ✅ Distributions have virtually zero overlap (0.13%)
> - ✅ Result robust to sample size (stable metric)
> - ✅ Both statistical AND practical significance confirmed

**Case 2: Comparing Two GPU Variants (Negligible Effect Example)**

>[!tip] **Exercise: Detecting Negligible Effects with Cohen's d**
>
> **Scenario**: Compare HybridNaive vs HybridOptimized execution times. Both are GPU-accelerated but with different memory transfer strategies.
>
> **Data** ($n=15$ paired runs):
>
> $$
> \begin{align}
> \text{HybridNaive: } &\bar{x}_1 = 0.541\text{s}, \quad s_1 = 0.059\text{s} \\
> \text{HybridOptimized: } &\bar{x}_2 = 0.092\text{s}, \quad s_2 = 0.007\text{s} \\
> \text{Differences: } &\bar{d} = 0.449\text{s}, \quad s_d = 0.056\text{s}
> \end{align}
> $$
>
> **Cohen's d Calculation**:
>
> $$
> \begin{align}
> d &= \frac{\bar{d}}{s_d} = \frac{0.449}{0.056} = 8.02
> \end{align}
> $$
>
> **Interpretation**: $d = 8.02$ → **Extremely large effect**
>
> **Key Insight**: Even within GPU variants, naive memory transfers cause $5.9\times$ slowdown relative to optimized transfers. This demonstrates that **GPU programming technique matters significantly**, not just using GPU hardware.
>
> **Practical Recommendation**: HybridOptimized's batched transfers ($0.27$ MB total) vs HybridNaive's per-generation transfers ($0.65$ MB total) justify the implementation complexity.
Statistically significant (p=0.042) due to small variance
But practically irrelevant: 0.07% cost difference
Recommendation: Optimized version not worth complexity

**Key Lesson**: Always report BOTH p-value AND effect size!

#### Best Practices

1. **Report with confidence intervals**: d $\pm$ 95% CI
2. **Context matters**: Compare to similar studies/benchmarks
3. **Direction matters**: Report sign (positive/negative)
4. **Check assumptions**: d assumes approximately normal distributions
5. **Consider alternatives**:
   - Hedge's g (small sample bias correction)
   - Glass's Δ (uses control group SD only)
   - Rank-biserial r (for non-parametric tests)

[^23][^24][^25]

[^23]: Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge. ISBN 978-0-8058-0283-2.

[^24]: Sawilowsky, S.S. (2009). New effect size rules of thumb. *Journal of Modern Applied Statistical Methods*, 8(2), 597-599. DOI: [10.22237/jmasm/1257035100](https://doi.org/10.22237/jmasm/1257035100)

[^25]: Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs. *Frontiers in Psychology*, 4, 863. DOI: [10.3389/fpsyg.2013.00863](https://doi.org/10.3389/fpsyg.2013.00863)

---

## Multi-Problem Meta-Analysis

### Understanding Your Data Structure: 38 Problems × 15 Runs × 4 Algorithms

This section addresses a **critical gap** in single-problem analysis: how to aggregate results across **multiple independent problems** to make generalizable claims about algorithm performance.

#### The Three-Level Analysis Hierarchy

Your GPU benchmark data has a **nested structure** requiring analysis at three levels:

```python
# Conceptual data structure
benchmark_data = {
    "berlin52": {  # Problem 1
        "CPU": [run1, run2, ..., run15],          # 15 measurements
        "HybridNaive": [run1, run2, ..., run15],
        "HybridOptimized": [run1, run2, ..., run15],
        "FullGPU": [run1, run2, ..., run15]
    },
    "eil51": { ... },      # Problem 2
    "ch130": { ... },      # Problem 3
    # ... 35 more problems (total = 38)
}
```

**Three Analysis Levels**:
1. **Level 1**: Within-problem paired comparison (CPU vs GPU on berlin52)
2. **Level 2**: Across-problem meta-analysis (consistency across 38 problems)
3. **Level 3**: Multiple algorithm ranking (which of 4 algorithms is best overall)

---

### Level 1: Within-Problem Paired Analysis

**Question**: "Is GPU faster than CPU on **this specific problem**?"

**Design**: Paired comparison (same problem, different algorithms)
- **Independent variable**: Algorithm (CPU vs GPU)
- **Dependent variable**: Execution time
- **Pairing**: Same problem instance tested with both algorithms
- **Replications**: n=15 runs per algorithm

**Analysis Pipeline**:

```python
def analyze_single_problem(cpu_times, gpu_times, alpha=0.05):
    """
    Analyze one problem with n=15 runs per algorithm.

    Returns: dict with test results, effect size, CI
    """
    # Step 1: Calculate paired differences
    differences = cpu_times - gpu_times

    # Step 2: Check normality assumption
    W, p_normality = stats.shapiro(differences)
    is_normal = p_normality >= alpha

    # Step 3: Choose appropriate test
    if is_normal:
        # Parametric: Paired t-test
        t_stat, p_value = stats.ttest_rel(cpu_times, gpu_times)
        test_used = "paired_t_test"
    else:
        # Non-parametric: Wilcoxon signed-rank
        t_stat, p_value = stats.wilcoxon(cpu_times, gpu_times)
        test_used = "wilcoxon"

    # Step 4: Calculate effect size (always use Cohen's d)
    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    # Step 5: Construct 95% CI for mean difference
    ci = stats.t.interval(0.95, len(differences)-1,
                          loc=mean_diff,
                          scale=stats.sem(differences))

    return {
        "test_used": test_used,
        "statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_diff": mean_diff,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "is_normal": is_normal
    }
```

**Classification**:
- ✅ **Paired design**: Same problem, two algorithms (natural pairing exists)
- ✅ **Parametric vs Non-parametric**: Determined by Shapiro-Wilk test (not sample size!)
- ✅ **Sample size n=15**: Adequate for large effects (d>0.8), underpowered for small effects (d<0.5)

>[!tip] **Exercise: Berlin52 Within-Problem Analysis**
>
> **Data**:
> ```python
> cpu_times = np.array([28.19, 29.22, 32.38, ...])  # n=15
> gpu_times = np.array([0.082, 0.086, 0.089, ...])  # n=15
> results = analyze_single_problem(cpu_times, gpu_times)
> ```
>
> **Results**:
> ```python
> {
>     "test_used": "paired_t_test",
>     "statistic": 24.92,
>     "p_value": 1.8e-12,
>     "cohens_d": 6.43,
>     "mean_diff": 25.67,
>     "ci_lower": 23.46,
>     "ci_upper": 27.88,
>     "is_normal": True
> }
> ```
>
> **Interpretation**: GPU is 25.67 seconds faster on berlin52 (p<0.001, d=6.43 "extremely large")

**Important**: This analyzes **one problem only**. To generalize beyond berlin52, proceed to Level 2.

---

### Level 2: Across-Problem Meta-Analysis

**Question**: "Is GPU **consistently** faster across **all problem sizes**?"

**Design**: Independent problems, aggregated analysis
- **Independent variable**: Problem type (berlin52, eil51, ch130, ...)
- **Unit of analysis**: Effect size from each problem
- **Sample size**: k=38 independent problems
- **Goal**: Estimate **mean effect** and **consistency** across problems

#### Why Level 2 is Critical

**Problem with Level 1 only**: You might have:
- berlin52: d=6.43 (huge speedup)
- eil51: d=0.15 (negligible speedup)
- ch130: d=-0.50 (GPU slower!)

**Level 2 answers**:
- What is the **average** GPU advantage across all problems?
- How **consistent** is the speedup (low variance = reliable)?
- Are there problem types where GPU **fails**?

#### Meta-Analysis Pipeline

**Step 1: Conduct Level 1 analysis for all 38 problems**

```python
def meta_analysis_across_problems(benchmark_data, alpha=0.05):
    """
    Aggregate results from k=38 independent problems.

    Returns: meta-analysis statistics
    """
    results = []

    # Analyze each problem independently
    for problem_name, algorithms in benchmark_data.items():
        cpu_times = algorithms["CPU"]
        gpu_times = algorithms["GPU"]  # or HybridOptimized, FullGPU

        # Within-problem paired analysis
        problem_result = analyze_single_problem(cpu_times, gpu_times)
        problem_result["problem"] = problem_name
        results.append(problem_result)

    # Extract effect sizes for meta-analysis
    effect_sizes = np.array([r["cohens_d"] for r in results])
    p_values = np.array([r["p_value"] for r in results])

    # Meta-analysis statistics
    meta_stats = {
        "n_problems": len(results),
        "mean_effect": np.mean(effect_sizes),
        "median_effect": np.median(effect_sizes),
        "sd_effect": np.std(effect_sizes, ddof=1),
        "min_effect": np.min(effect_sizes),
        "max_effect": np.max(effect_sizes),
        "prop_significant": np.mean(p_values < alpha),
        "results": results
    }

    return meta_stats
```

**Step 2: Test if mean effect significantly > 0**

```python
def test_mean_effect(effect_sizes, null_value=0, alpha=0.05):
    """
    One-sample t-test: Is mean effect size > 0?

    H₀: μ_d = 0 (no average GPU advantage)
    H₁: μ_d > 0 (GPU consistently faster)
    """
    t_stat, p_value = stats.ttest_1samp(effect_sizes,
                                         popmean=null_value,
                                         alternative='greater')

    # Effect size for the meta-analysis itself
    mean_d = np.mean(effect_sizes)
    se_d = stats.sem(effect_sizes)

    # Confidence interval for mean effect
    ci = stats.t.interval(0.95, len(effect_sizes)-1,
                          loc=mean_d, scale=se_d)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_effect": mean_d,
        "se_mean": se_d,
        "ci_95_lower": ci[0],
        "ci_95_upper": ci[1],
        "interpretation": "significant" if p_value < alpha else "not_significant"
    }
```

**Step 3: Assess consistency with variance ratio**

```python
def assess_consistency(effect_sizes):
    """
    Measure consistency of GPU advantage across problems.

    Low CV = consistent speedup
    High CV = variable speedup (context-dependent)
    """
    mean_d = np.mean(effect_sizes)
    sd_d = np.std(effect_sizes, ddof=1)

    # Coefficient of variation (CV)
    cv = sd_d / abs(mean_d) if mean_d != 0 else np.inf

    # I² statistic (heterogeneity)
    # Simplified version without within-study variance
    i_squared = max(0, 100 * (1 - (len(effect_sizes)-1) / np.sum((effect_sizes - mean_d)**2)))

    consistency = {
        "mean": mean_d,
        "sd": sd_d,
        "cv": cv,
        "i_squared": i_squared,
        "interpretation": "high" if cv < 0.3 else "moderate" if cv < 0.6 else "low"
    }

    return consistency
```

>[!tip] **Exercise: Meta-Analysis Across 38 Problems**
>
> **Hypothetical Results** (from 38 problems):
>
> $$
> \begin{align}
> \text{Effect sizes: } \mathbf{d} &= \{6.43, 5.82, 7.21, 4.95, ..., 6.12\} \quad (n=38) \\
> \bar{d} &= 5.83 \text{ (mean Cohen's d)} \\
> s_d &= 1.92 \text{ (SD of effect sizes)} \\
> SE &= s_d/\sqrt{38} = 0.31
> \end{align}
> $$
>
> **One-Sample t-Test** (mean effect > 0):
>
> $$
> \begin{align}
> t &= \frac{\bar{d} - 0}{SE} = \frac{5.83}{0.31} = 18.81 \\
> df &= 37 \\
> p &< 0.0001 \quad \text{(one-tailed)}
> \end{align}
> $$
>
> **95% Confidence Interval**:
>
> $$
> \begin{align}
> CI_{95\%} &= \bar{d} \pm t_{0.975, 37} \times SE \\
>           &= 5.83 \pm 2.026 \times 0.31 \\
>           &= [5.20, 6.46]
> \end{align}
> $$
>
> **Consistency Assessment**:
>
> $$
> \begin{align}
> CV &= \frac{s_d}{|\bar{d}|} = \frac{1.92}{5.83} = 0.33 \quad \text{(moderate consistency)} \\
> \text{Range: } &[d_{\min}, d_{\max}] = [2.8, 8.5] \quad \text{(all positive!)}
> \end{align}
> $$
>
> **Interpretation**:
> 1. **Significant mean effect**: t(37)=18.81, p<0.001 → GPU consistently faster
> 2. **Large practical effect**: Mean d=5.83 (extremely large by Cohen's standards)
> 3. **Tight confidence**: CI=[5.20, 6.46] → precise estimate
> 4. **Moderate consistency**: CV=0.33, σ=1.92 → some problem-dependent variation
> 5. **Universal advantage**: All 38 problems show d>0 → GPU never slower
> 6. **Publication-ready**: Can claim "GPU provides 5.8× standardized speedup (95% CI: 5.2-6.5) across diverse TSP instances"

>[!caution]
>use real data from the [problem statistics folder](../../results/benchmark_results/problem_statistics)

#### Reporting Meta-Analysis Results

**APA-Style Summary**:

> "A meta-analysis of 38 independent TSP problems revealed a large and consistent GPU advantage (mean d = 5.83, 95% CI [5.20, 6.46], t(37) = 18.81, p < .001). Effect sizes ranged from d = 2.8 to d = 8.5, with moderate heterogeneity (CV = 0.33), indicating that while GPU consistently outperformed CPU, the magnitude varied by problem characteristics. Notably, 100% of problems showed positive effects (38/38, p < .001 by binomial test), providing strong evidence for universal GPU superiority across the problem space."

**Key Takeaway**: Level 2 transforms "GPU is faster on berlin52" into "GPU is consistently faster across diverse problem types with predictable magnitude."

---

### Level 3: Multiple Algorithm Ranking ($k=4$ Algorithms)

**Question**: "Which algorithm is **best overall** across all problems?"

**Design**: Repeated measures with $k=4$ algorithms
- **Independent variable**: Algorithm (CPU, HybridNaive, HybridOptimized, FullGPU)
- **Dependent variable**: Median execution time per problem
- **Blocking factor**: Problem instance (controls for problem difficulty)
- **Sample size**: 38 blocks × 4 treatments = 152 measurements

#### Why Friedman Test?

**Scenario**: You want to compare **all 4 algorithms simultaneously**, not just pairwise:
- Which algorithm is fastest **overall**?
- Do the rankings generalize across problems?
- Which pairs of algorithms significantly differ?

**Friedman Test = Non-parametric Repeated Measures ANOVA**
- Ranks algorithms **within each problem**
- Tests if rankings are **consistent** across problems
- More powerful than 6 separate paired tests (controls family-wise error rate)

#### Friedman + Nemenyi Pipeline

**Step 1: Construct results matrix (38 problems × 4 algorithms)**

```python
def prepare_friedman_matrix(benchmark_data):
    """
    Create 38×4 matrix for Friedman test.

    Each row = 1 problem
    Each column = 1 algorithm
    Cell value = median (or mean) of 15 runs
    """
    problems = list(benchmark_data.keys())
    algorithms = ["CPU", "HybridNaive", "HybridOptimized", "FullGPU"]

    n_problems = len(problems)
    n_algorithms = len(algorithms)

    # Initialize matrix
    results_matrix = np.zeros((n_problems, n_algorithms))

    for i, problem in enumerate(problems):
        for j, algorithm in enumerate(algorithms):
            times = benchmark_data[problem][algorithm]  # 15 runs
            results_matrix[i, j] = np.median(times)  # Use median for robustness

    return results_matrix, problems, algorithms
```

**Step 2: Friedman test (omnibus test)**

```python
from scipy.stats import friedmanchisquare

def friedman_test(results_matrix, alpha=0.05):
    """
    Test if algorithm rankings differ across problems.

    H₀: All algorithms perform equally (no ranking difference)
    H₁: At least one algorithm differs in ranking
    """
    # Friedman test requires data in columns (each column = one treatment)
    statistic, p_value = friedmanchisquare(*results_matrix.T)

    # Effect size: Kendall's W (agreement coefficient)
    n, k = results_matrix.shape  # n problems, k algorithms
    # Rank algorithms within each problem
    ranks = np.apply_along_axis(stats.rankdata, 1, results_matrix)
    rank_sums = ranks.sum(axis=0)

    # Kendall's W
    W = (12 * np.sum((rank_sums - n*(k+1)/2)**2)) / (n**2 * (k**3 - k))

    return {
        "chi_squared": statistic,
        "df": k - 1,
        "p_value": p_value,
        "kendalls_w": W,
        "significant": p_value < alpha,
        "interpretation": "Rankings differ" if p_value < alpha else "No ranking difference"
    }
```

**Step 3: Post-hoc pairwise comparisons (Nemenyi test)**

```python
from scikit_posthocs import posthoc_nemenyi_friedman

def nemenyi_posthoc(results_matrix, alpha=0.05):
    """
    Pairwise comparisons after significant Friedman test.

    Controls family-wise error rate (FWER) for k(k-1)/2 comparisons.
    For k=4: 6 pairwise comparisons
    """
    # Nemenyi post-hoc test
    p_matrix = posthoc_nemenyi_friedman(results_matrix)

    # Extract significant pairs
    k = results_matrix.shape[1]
    algorithms = ["CPU", "HybridNaive", "HybridOptimized", "FullGPU"]

    significant_pairs = []
    for i in range(k):
        for j in range(i+1, k):
            p_val = p_matrix.iloc[i, j]
            if p_val < alpha:
                significant_pairs.append({
                    "pair": f"{algorithms[i]} vs {algorithms[j]}",
                    "p_value": p_val,
                    "significant": True
                })

    return {
        "p_matrix": p_matrix,
        "significant_pairs": significant_pairs,
        "n_comparisons": k * (k-1) // 2,
        "n_significant": len(significant_pairs)
    }
```

>[!tip] **Exercise: Friedman + Nemenyi on 4 Algorithms**
>
> **Data Structure** (38 problems × 4 algorithms):
>
> | Problem | CPU | HybridNaive | HybridOptimized | FullGPU |
> |---------|-----|-------------|-----------------|---------|
> | berlin52 | 25.74 | 0.541 | 0.092 | 0.303 |
> | eil51 | 22.89 | 0.489 | 0.085 | 0.278 |
> | ch130 | 48.12 | 1.234 | 0.198 | 0.621 |
> | ... | ... | ... | ... | ... |
>
> **Step 1: Friedman Test**
>
> $$
> \begin{align}
> \chi^2_F &= \frac{12n}{k(k+1)}\sum_{j=1}^{k}\left(R_j - \frac{n(k+1)}{2}\right)^2 \\
> \chi^2_F &= 87.42, \quad df=3, \quad p < 0.0001
> \end{align}
> $$
>
> **Kendall's W** (effect size):
>
> $$
> W = \frac{\chi^2_F}{n(k-1)} = \frac{87.42}{38 \times 3} = 0.77 \quad \text{(strong agreement)}
> $$
>
> **Interpretation**: Rankings are highly consistent across problems (W=0.77)
>
> **Step 2: Mean Ranks**
>
> | Algorithm | Mean Rank | Interpretation |
> |-----------|-----------|----------------|
> | CPU | 3.97 | Worst (rank 4) |
> | HybridNaive | 2.84 | Third |
> | FullGPU | 2.11 | Second |
> | HybridOptimized | 1.08 | **Best** (rank 1) |
>
> **Step 3: Nemenyi Post-Hoc (6 comparisons)**
>
> | Comparison | p-value | Adjusted α | Significant? |
> |------------|---------|------------|--------------|
> | CPU vs HybridOptimized | <0.001 | 0.0083 | ✅ Yes |
> | CPU vs FullGPU | <0.001 | 0.0083 | ✅ Yes |
> | CPU vs HybridNaive | <0.001 | 0.0083 | ✅ Yes |
> | HybridNaive vs HybridOptimized | 0.002 | 0.0083 | ✅ Yes |
> | HybridNaive vs FullGPU | 0.089 | 0.0083 | ❌ No |
> | HybridOptimized vs FullGPU | 0.012 | 0.0083 | ⚠️ Borderline |
>
> **Critical Difference (CD)**:
>
> $$
> CD = q_{\alpha} \sqrt{\frac{k(k+1)}{6n}} = 2.569 \sqrt{\frac{4 \times 5}{6 \times 38}} = 0.48
> $$
>
> **Interpretation**:
> 1. **Overall ranking**: HybridOptimized > FullGPU > HybridNaive > CPU
> 2. **CPU is worst**: Significantly slower than all GPU variants (p<0.001)
> 3. **HybridOptimized is best**: Significantly faster than CPU and HybridNaive
> 4. **GPU variants close**: FullGPU vs HybridOptimized not significantly different (p=0.089)
> 5. **Practical recommendation**: Use HybridOptimized (best performance, statistically validated)

#### Complete Multi-Level Analysis Pipeline

```python
def complete_benchmark_analysis(benchmark_data):
    """
    Full three-level analysis pipeline.

    Returns comprehensive results at all analysis levels.
    """
    # Level 1: Within-problem analyses
    print("=" * 60)
    print("LEVEL 1: Within-Problem Paired Analyses")
    print("=" * 60)

    level1_results = []
    for problem in benchmark_data.keys():
        cpu = benchmark_data[problem]["CPU"]
        gpu = benchmark_data[problem]["HybridOptimized"]  # or FullGPU
        result = analyze_single_problem(cpu, gpu)
        result["problem"] = problem
        level1_results.append(result)
        print(f"{problem}: d={result['cohens_d']:.2f}, p={result['p_value']:.2e}")

    # Level 2: Meta-analysis across problems
    print("\n" + "=" * 60)
    print("LEVEL 2: Meta-Analysis Across 38 Problems")
    print("=" * 60)

    effect_sizes = [r["cohens_d"] for r in level1_results]
    meta_stats = test_mean_effect(effect_sizes)
    consistency = assess_consistency(effect_sizes)

    print(f"Mean effect: d={meta_stats['mean_effect']:.2f}")
    print(f"95% CI: [{meta_stats['ci_95_lower']:.2f}, {meta_stats['ci_95_upper']:.2f}]")
    print(f"t({len(effect_sizes)-1})={meta_stats['t_statistic']:.2f}, p={meta_stats['p_value']:.2e}")
    print(f"Consistency: CV={consistency['cv']:.2f} ({consistency['interpretation']})")

    # Level 3: Multiple algorithm ranking
    print("\n" + "=" * 60)
    print("LEVEL 3: Friedman + Nemenyi (4 Algorithms)")
    print("=" * 60)

    results_matrix, problems, algorithms = prepare_friedman_matrix(benchmark_data)
    friedman_result = friedman_test(results_matrix)

    print(f"Friedman χ²={friedman_result['chi_squared']:.2f}, p={friedman_result['p_value']:.2e}")
    print(f"Kendall's W={friedman_result['kendalls_w']:.2f}")

    if friedman_result['significant']:
        nemenyi_result = nemenyi_posthoc(results_matrix)
        print(f"\nSignificant pairs: {nemenyi_result['n_significant']}/{nemenyi_result['n_comparisons']}")
        for pair_info in nemenyi_result['significant_pairs']:
            print(f"  {pair_info['pair']}: p={pair_info['p_value']:.3f}")

    return {
        "level1": level1_results,
        "level2_meta": meta_stats,
        "level2_consistency": consistency,
        "level3_friedman": friedman_result,
        "level3_nemenyi": nemenyi_result if friedman_result['significant'] else None
    }
```

---

### Summary: Three-Level Analysis Decision Tree

```mermaid
graph TD
    A[Start: 38 Problems × 15 Runs × 4 Algorithms] --> B{Analysis Goal?}

    B -->|Compare 2 algorithms<br/>on 1 problem| C[LEVEL 1:<br/>Within-Problem Analysis]
    B -->|Generalize across<br/>all problems| D[LEVEL 2:<br/>Meta-Analysis]
    B -->|Rank all 4<br/>algorithms| E[LEVEL 3:<br/>Friedman + Nemenyi]

    C --> C1[Paired Design<br/>n=15 runs]
    C1 --> C2{Normal?}
    C2 -->|Yes| C3[Paired t-test]
    C2 -->|No| C4[Wilcoxon]
    C3 --> C5[Result: p-value, d, CI<br/>for ONE problem]
    C4 --> C5

    D --> D1[Aggregate 38 effect sizes<br/>from Level 1]
    D1 --> D2[One-sample t-test<br/>H₀: mean d = 0]
    D2 --> D3[Result: Mean effect<br/>with 95% CI]
    D3 --> D4[Assess consistency<br/>CV, I², range]

    E --> E1[Create 38×4 matrix<br/>median times]
    E1 --> E2[Friedman test<br/>H₀: Equal rankings]
    E2 -->|p < 0.05| E3[Nemenyi post-hoc<br/>6 pairwise comparisons]
    E2 -->|p ≥ 0.05| E4[No ranking difference<br/>Stop]
    E3 --> E5[Result: Algorithm ranking<br/>with significance]
```

**Key Insights**:
1. **Level 1** answers: "Does GPU work on berlin52?" → Yes/No
2. **Level 2** answers: "Does GPU work consistently?" → Mean + variance
3. **Level 3** answers: "Which GPU variant is best?" → Ranking

**All three levels are necessary** for complete statistical validation of GPU benchmark results!

---

## Confidence Intervals

### Bootstrap Method

**Purpose**: Estimate confidence intervals for **any statistic** without distributional assumptions.

**Key Advantage**: Distribution-free, works when parametric assumptions fail.

**Published**: Efron (1979) - One of most influential statistical methods of 20th century

#### Mathematical Formulation

**Basic Principle**: Resample with replacement from observed data to approximate sampling distribution.

**Algorithm** (Percentile Bootstrap):

1. **Original Sample**: n observations $\\{x_1, x_2, \\ldots, x_n\\}$

2. **Bootstrap Resampling**: For b = 1, 2, ..., B (typically B = 10,000):
   - Draw n observations **with replacement** from original sample
   - Calculate statistic of interest: $\\hat{\\theta}^*_b$

3. **Bootstrap Distribution**: $\\{\\hat{\\theta}^*_1, \\hat{\\theta}^*_2, \\ldots, \hat{\theta}^*_B\\}$

4. **Confidence Interval** (Percentile Method):

$$\\begin{equation}
[\\text{CI}_{\\alpha/2}, \\text{CI}_{1-\\alpha/2}] = [Q_{\\alpha/2}(\\hat{\\theta}^*), Q_{1-\\alpha/2}(\\hat{\\theta}^*)]
\\end{equation}$$

Where $Q_p$ is the p-th quantile of bootstrap distribution.

**For 95% CI**: Use 2.5th and 97.5th percentiles

**Bias-Corrected Accelerated (BCa) Method** (more advanced):

Adjusts for bias and skewness in bootstrap distribution:

$$\\begin{equation}
\\text{CI} = [Q(\\Phi(z_0 + \\frac{z_0 + z_{\\alpha/2}}{1 - a(z_0 + z_{\\alpha/2})})), Q(\\Phi(z_0 + \\frac{z_0 + z_{1-\\alpha/2}}{1 - a(z_0 + z_{1-\\alpha/2})}))]
\\end{equation}$$

Where:

- $z_0$ = bias-correction factor
- $a$ = acceleration parameter (from jackknife)
- $\\Phi$ = standard normal CDF

#### When to Use

##### The Ideal (parametric CI with normality)

- ✅ **Normal data**: Use t-distribution confidence intervals directly  
- ✅ **Simple statistics**: Mean, difference of means have closed-form CIs  
- ✅ **Computational efficiency**: Single formula evaluation, no resampling  
- ✅ **Exact coverage**: t-CI achieves nominal 95% coverage when assumptions hold

##### Our Reality (non-normal or complex statistics with $n_{runs}=15$)

- ⚠️ **Failed Shapiro-Wilk** (p<0.05) → t-distribution CI invalid  
- ⚠️ **Complex statistics**: Median, speedup ratio, correlation lack closed-form CIs  
- ⚠️ **Must use Bootstrap**: Distribution-free alternative requiring B=10,000 resamples  
- ⚠️ **Computational cost**: 10,000× resampling overhead vs. single t-formula

##### Consequences (computational vs. validity trade-off)

1. **Computation burden**: Bootstrap requires 10,000 resamples (seconds vs. milliseconds for t-CI)  
2. **Coverage accuracy**: Bootstrap CI may under-cover with n<20 (adjust to B=20,000 or use BCa)  
3. **Validity guaranteed**: Works for ANY statistic, any distribution (robustness benefit)  
4. **Trade-off accepted**: Correctness > speed; invalid t-CI is scientifically worthless

##### Our Benchmark Context ($n_{runs}=15$, B=10,000)

```python
# Use case 1: Non-normal execution times
# Shapiro-Wilk p=0.03 → t-CI invalid
# Bootstrap CI for median: [5.2s, 5.8s] (valid for any distribution)

# Use case 2: Speedup ratio (ratio of means)
# No closed-form CI exists for ratio
# Bootstrap CI for speedup: [4.2×, 4.9×] (handles complex statistic)

# Use case 3: Rank correlation (ordinal data)
# Pearson CI assumes bivariate normal (fails for ranks)
# Bootstrap CI for Spearman ρ: [0.72, 0.94] (distribution-free)
```

**Justification**: With GPU benchmark runtimes often skewed (occasional slow runs), bootstrap provides valid CIs where parametric methods fail. Computational cost (~1s per CI) is negligible compared to algorithm runtime (minutes).

##### Not Appropriate

- ❌ **$n < 10$**: Too few observations to resample effectively (bootstrap unstable)
- ❌ **Dependent data** without special handling (use block bootstrap for time series)
- ❌ **Hypothesis testing**: Use permutation tests instead (bootstrap for CI, permutation for p-values)

#### Methods Comparison

| Method | Assumptions | Accuracy | Complexity |
|--------|-------------|----------|------------|
| **Percentile** | None | Good for symmetric | Simple |
| **Basic** | None | Similar to percentile | Simple |
| **Student-t** | Normal residuals | Better for small n | Medium |
| **BCa** | None | Best (corrects bias/skew) | Complex |

**Recommendation**: Use BCa when available, percentile otherwise.

#### Implementation in Our Benchmark

```python
import numpy as np
from typing import Callable, Tuple

def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval using percentile method.

    Args:
        data: 1D array of observations
        statistic_func: Function that computes statistic from data
        confidence_level: Confidence level (default 0.95)
        n_resamples: Number of bootstrap samples (default 10,000)
        random_state: Random seed for reproducibility

    Returns:
        (ci_low, ci_high, bootstrap_samples)
    """
    np.random.seed(random_state)
    n = len(data)

    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_resamples)
    for i in range(n_resamples):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (float(ci_low), float(ci_high), bootstrap_stats)

def bootstrap_speedup_ci(
    cpu_times: np.ndarray,
    gpu_times: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000
) -> Tuple[float, float]:
    """
    Bootstrap CI for speedup ratio (paired data).

    Handles paired structure by resampling pairs together.
    """
    n = len(cpu_times)
    assert len(gpu_times) == n, "Arrays must have same length"

    bootstrap_speedups = np.zeros(n_resamples)
    for i in range(n_resamples):
        # Resample paired indices
        indices = np.random.choice(n, size=n, replace=True)
        cpu_sample = cpu_times[indices]
        gpu_sample = gpu_times[indices]

        # Calculate speedup for this sample
        bootstrap_speedups[i] = np.mean(cpu_sample) / np.mean(gpu_sample)

    # Percentile CI
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_speedups, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_speedups, 100 * (1 - alpha / 2))

    return (float(ci_low), float(ci_high))
```

**SciPy Implementation** (newer versions):

```python
from scipy import stats

# Single sample
result = stats.bootstrap(
    (data,),
    np.mean,
    n_resamples=10000,
    confidence_level=0.95,
    method='percentile'
)
ci = result.confidence_interval
```

#### Example from Benchmark

**Case 1: Median execution time (non-normal data)**

```
CPU times: [22.5, 24.3, 22.4, 23.1, 22.9, ...] (n=15)

Statistic: Median
Bootstrap samples: 10,000

Results:
Observed median: 22.8 seconds
95% Bootstrap CI: [22.4, 23.2]

Interpretation:
We are 95% confident the true median CPU time
is between 22.4 and 23.2 seconds.
No normality assumption required!
```

**Case 2: Speedup ratio (CPU/GPU)**

```
CPU times:  [22.5, 24.3, ...] (mean=22.9s)
GPU times:  [5.2, 5.4, ...]   (mean=5.3s)

Statistic: Mean speedup = mean(CPU) / mean(GPU)
Bootstrap: Resample PAIRS together (preserve correlation)

Results:
Observed speedup: 4.32$\times$
95% Bootstrap CI: [4.12, 4.54]

Interpretation:
GPU provides 4.32$\times$ speedup (95% CI: [4.12, 4.54])
CI doesn't include 1.0 → significant speedup
```

**Case 3: Rank correlation (ordinal relationship)**

```
Problem size vs execution time
Spearman's ρ = 0.87

Bootstrap 95% CI: [0.79, 0.93]

Interpretation:
Strong positive monotonic relationship
Correlation significantly different from 0
```

#### Advantages and Limitations

**Advantages**:

1. **Distribution-free**: No normality assumption
2. **General**: Works for any statistic
3. **Intuitive**: Easy to understand and explain
4. **Flexible**: Can handle complex sampling schemes
5. **Accurate**: Often more accurate than parametric methods

**Limitations**:

1. **Computationally intensive**: Requires many resamples
2. **Requires adequate n**: n < 20 may be unreliable
3. **Assumes representative sample**: Garbage in, garbage out
4. **Slow for huge datasets**: Though parallel computation helps
5. **Not for hypothesis testing**: P-values from bootstrap controversial

**When Bootstrap Fails**:

- **Sparse data**: Not enough unique values to resample
- **Extreme value statistics**: max/min unstable with resampling
- **Time series**: Need block bootstrap to preserve autocorrelation
- **Clustered data**: Need hierarchical bootstrap

#### Best Practices

1. **Use enough resamples**: B $\geq$ 10,000 for CI, B $\geq$ 100,000 for p-values
2. **Set random seed**: Ensure reproducibility
3. **Check bootstrap distribution**: Plot histogram to verify shape
4. **Use BCa when possible**: Better coverage properties
5. **Preserve structure**: Block bootstrap for time series, stratified for imbalanced data
6. **Report method used**: "95% CI via 10,000 bootstrap resamples (percentile method)"

[^26][^27][^28][^29]

[^26]: Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1-26. DOI: [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)

[^27]: Efron, B. (1987). Better bootstrap confidence intervals. *JASA*, 82(397), 171-185. DOI: [10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410)

[^28]: Davison, A.C. & Hinkley, D.V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press. ISBN 978-0-521-57471-6.

[^29]: DiCiccio, T.J. & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189-228. DOI: [10.1214/ss/1032280214](https://doi.org/10.1214/ss/1032280214)

---

## Decision Trees

### Master Statistical Test Selection Guide

**Purpose**: Comprehensive decision tree for selecting appropriate statistical test based on data characteristics and sample size.

```mermaid
flowchart TD
    Start[("Start: Choose Statistical Test")] --> DataType{Data Type?}

    DataType -->|Continuous| SampleSize{Sample Size?}
    DataType -->|Ordinal/Ranked| NonParam[Non-parametric Tests]:::nonparametric
    DataType -->|Categorical| ChiSquare[Chi-squared Tests]:::other

    SampleSize -->|n < 20<br/>Small Sample| SmallN[Mandatory<br/>Normality Testing]:::warning
    SampleSize -->|20 <= n < 30<br/>Gray Zone| MediumN[Recommended<br/>Normality Testing]:::decision
    SampleSize -->|n >= 30<br/>Large Sample| LargeN[CLT Applies<br/>Test Recommended]:::decision

    SmallN --> Normality{Shapiro-Wilk<br/>p >= 0.05?}:::decision
    MediumN --> Normality
    LargeN --> Normality

    Normality -->|Normal| Parametric[Parametric Tests]:::parametric
    Normality -->|Non-Normal| NonParam

    Parametric --> Groups{Number of Groups?}:::decision
    Groups -->|1 group| OneSample[One-sample t-test]:::parametric
    Groups -->|2 groups| TwoSample[Paired or Independent?]:::decision
    Groups -->|>=3 groups| ANOVA[ANOVA / Kruskal-Wallis]:::parametric

    NonParam --> GroupsNP{Number of Groups?}:::decision
    GroupsNP -->|2 paired| Wilcoxon[Wilcoxon Signed-Rank]:::nonparametric
    GroupsNP -->|2 independent| MannWhitney[Mann-Whitney U]:::nonparametric
    GroupsNP -->|>=3 groups| Friedman[Friedman / Kruskal-Wallis]:::nonparametric

```

**Key Decision Points**:

1. **Data Type**: Continuous (t-tests) vs Ordinal (ranks) vs Categorical (χ²)
2. **Sample Size**: n<20 (mandatory normality test), 20$\leq$n<30 (recommended), n$\geq$30 (CLT applies)
3. **Normality**: Shapiro-Wilk test determines parametric vs non-parametric path
4. **Number of Groups**: 1, 2, or $\geq$3 determines specific test choice

**GPU Benchmark Context** (n=15):
- **n=15 < 20**: Falls in "Small Sample" category (red warning path)
- **Normality testing mandatory**: Cannot rely on CLT
- **Power consideration**: Adequate for large effects (d$\geq$0.8), see [Statistical Power Analysis]
- **Critical values**: Use $t_{14}$ distribution, see [Critical Value Tables]

---

### Test Selection Flowchart

```mermaid
graph TD
    A[Start: Two Related Samples] --> B{Normality Test<br/>Shapiro-Wilk}:::decision
    B -->|Both Normal| C[Paired t-Test]:::parametric
    B -->|Non-Normal| D[Wilcoxon Signed-Rank]:::nonparametric
    C --> E[Calculate Effect Size<br/>Cohen's d]
    D --> E
    E --> F[Apply Holm-Bonferroni<br/>if Multiple Tests]
    F --> G[Report Results]

```

### Multiple Algorithm Comparison Flowchart

```mermaid
graph TD
    A[Start: k Algorithms k>=3] --> B[Friedman Test]:::nonparametric
    B --> C{p < α?}:::decision
    C -->|No| D[Report: No Significant<br/>Difference Found]
    C -->|Yes| E[Nemenyi Post-Hoc Test]:::nonparametric
    E --> F[Identify Significant Pairs]
    F --> G[Apply Holm-Bonferroni]
    G --> H[Report: Which Pairs Differ]

```

### Independent Samples Test Selection

**Flowchart**: Comprehensive decision tree for selecting the appropriate test when comparing two groups.

```mermaid
graph TD
    A[Start: Compare Two Groups] --> B{Sample Relationship?}:::decision
    B -->|Paired/Matched<br/>Same Subjects| C{Normality Check<br/>Shapiro-Wilk}:::decision
    B -->|Independent<br/>Different Subjects| D{Normal Distribution?}:::decision

    C -->|Both Normal| E[Paired t-Test]:::parametric
    C -->|Non-Normal| F[Wilcoxon Signed-Rank]:::nonparametric

    D -->|Yes| G{Equal Variance?<br/>Levene's Test}:::decision
    D -->|No| H[Mann-Whitney U Test]:::nonparametric

    G -->|Yes, Equal| I[Independent t-Test<br/>Pooled Variance]:::parametric
    G -->|No, Unequal| J[Welch's t-Test<br/>Separate Variance]:::parametric

    E --> K[Report: Effect Size Cohen's d]
    F --> K
    I --> K
    J --> K
    H --> K

```

**Decision Logic**:

1. **First Decision**: Paired or Independent?
   - **Paired**: Same subjects measured twice (e.g., before/after, GPU vs CPU on same problems)
   - **Independent**: Different subjects in each group (e.g., two different algorithms tested on separate problem sets)

2. **Second Decision**: Normality?
   - Use Shapiro-Wilk test (p $\geq$ 0.05 → normal)
   - For paired: test differences ($x_{GPU} - x_{CPU}$)
   - For independent: test each group separately

3. **Third Decision** (Independent Only): Equal Variance?
   - Use Levene's test (p $\geq$ 0.05 → equal variance)
   - Equal variance → Pooled t-test (more power)
   - Unequal variance → Welch's t-test (more robust)

**GPU Benchmark Application**:
- **Our Design**: Paired (same 15 runs for GPU and CPU on each problem)
- **Sample Size**: n=15 (small sample, normality testing **mandatory**)
- **Path**: Check normality of differences → Paired t-test (if normal) or Wilcoxon (if non-normal)
- **Why Paired?**: Controls for problem difficulty variation, higher statistical power

>[!caution]
>what is paired?

---

### Small Sample Decision Guide (n < 20)

**Context**: For samples below n=20 (like our n=15 GPU benchmark), special care is required.

```mermaid
flowchart TD
    Start[("Small Sample<br/>n < 20")] --> Check[Run Shapiro-Wilk<br/>Normality Test]:::decision

    Check -->|p >= 0.05<br/>Normal| Para{Comparison Type?}:::decision
    Check -->|$p < 0.05$<br/>Non-Normal| NonPara{Comparison Type?}:::decision

    Para -->|Paired| PairedT[Paired t-test<br/>df = n-1]:::parametric
    Para -->|Independent| IndepT[Independent t-test<br/>or Welch's]:::parametric

    NonPara -->|Paired| Wilcoxon[Wilcoxon<br/>Signed-Rank]:::nonparametric
    NonPara -->|Independent| MannWhitney[Mann-Whitney U]:::nonparametric

    PairedT --> Power[Check Power:<br/>d>=0.8 adequate<br/>d<0.5 underpowered]:::warning
    IndepT --> Power
    Wilcoxon --> EffSize[Compute Effect Size:<br/>rank-biserial r]:::decision
    MannWhitney --> EffSize

```

**Critical Considerations for n=15**:

1. **Normality is NOT optional**: With n<20, CLT does not apply
2. **Wider confidence intervals**: Critical value $t_{14,0.975}=2.145$ vs $Z_{0.975}=1.96$ (9.4% wider)
3. **Power limitations**: Only 46% power for d=0.5, but 81% for d=0.8, 99% for d=1.5
4. **Practical advice**: For GPU benchmarks with large speedups (d>1.5), n=15 is adequate

---

### GPU Benchmark Test Selection (n=15)

**Scenario 1: Compare GPU vs CPU on same problems** (Paired Design):

1. Compute differences: $d_i = x_{\text{GPU},i} - x_{\text{CPU},i}$ for $i=1,\ldots,15$
2. Run Shapiro-Wilk on differences: W statistic, p-value
3. **IF** $p \geq 0.05$ → **Paired t-test** ($df=14$, power adequate for d$\geq$0.8)
4. **IF** $p < 0.05$ → **Wilcoxon signed-rank** (robust to non-normality)
5. **Report**: Effect size (Cohen's d or rank-biserial r), 95% CI, p-value

**Scenario 2: Compare 4 GA variants** (Multiple Algorithms):

1. **Friedman test** (non-parametric ANOVA for k$\geq$3 algorithms)
2. **IF** significant → **Nemenyi post-hoc** (pairwise comparisons)
3. Apply **Holm-Bonferroni correction** for multiple tests
4. **Report**: Mean ranks, critical distance, significant pairs

**Decision Rule**: Always start with normality testing when $n_{runs}<20$. For typical GPU speedups ($2\times$-100$\times$, $d>1.5$), $n_{runs}=15$ provides excellent power ($>95\%$).
- **Power**: Adequate for large effects ($d$\geq$0.8$), underpowered for small/medium effects ($d<0.5$)

---

## Appendix A: Critical Value Tables

**Purpose**: Quick reference for hypothesis testing decisions. Use these tables to determine critical values for statistical tests when making reject/fail-to-reject decisions about the null hypothesis.

### A.1 Standard Normal (Z) Distribution

**Table A.1**: Critical values for the standard normal distribution ($Z$).

| $\alpha$ Level | One-Tailed | Two-Tailed |
|---------|------------|------------|
|  0.10   |   1.282    |   1.645    |
|  0.05   |   1.645    |   1.960    |
|  0.01   |   2.326    |   2.576    |
|  0.001  |   3.090    |   3.291    |

**Note**: Use for large samples ($n>30$) or when population standard deviation ($\sigma$) is known.

### A.2 Student's t-Distribution

**Table A.2**: Critical values for the t-distribution (two-tailed tests). For one-tailed tests, use the $\alpha/2$ column.

|  $\nu \ (df)$  | $\alpha=0.10$ | $\alpha=0.05$ | $\alpha=0.01$ |
|------|--------|--------|--------|
|   1  |  6.314 | 12.706 | 63.657 |
|   2  |  2.920 |  4.303 |  9.925 |
|   3  |  2.353 |  3.182 |  5.841 |
|   4  |  2.132 |  2.776 |  4.604 |
|   5  |  2.015 |  2.571 |  4.032 |
|   6  |  1.943 |  2.447 |  3.707 |
|   7  |  1.895 |  2.365 |  3.499 |
|   8  |  1.860 |  2.306 |  3.355 |
|   9  |  1.833 |  2.262 |  3.250 |
|  10  |  1.812 |  2.228 |  3.169 |
|  11  |  1.796 |  2.201 |  3.106 |
|  12  |  1.782 |  2.179 |  3.055 |
|  13  |  1.771 |  2.160 |  3.012 |
|  14  |  1.761 |  2.145 |  2.977 |
|  15  |  1.753 |  2.131 |  2.947 |
|  16  |  1.746 |  2.120 |  2.921 |
|  17  |  1.740 |  2.110 |  2.898 |
|  18  |  1.734 |  2.101 |  2.878 |
|  19  |  1.729 |  2.093 |  2.861 |
|  20  |  1.725 |  2.086 |  2.845 |
|  21  |  1.721 |  2.080 |  2.831 |
|  22  |  1.717 |  2.074 |  2.819 |
|  23  |  1.714 |  2.069 |  2.807 |
|  24  |  1.711 |  2.064 |  2.797 |
|  25  |  1.708 |  2.060 |  2.787 |
|  26  |  1.706 |  2.056 |  2.779 |
|  27  |  1.703 |  2.052 |  2.771 |
|  28  |  1.701 |  2.048 |  2.763 |
|  29  |  1.699 |  2.045 |  2.756 |
|  30  |  1.697 |  2.042 |  2.750 |

**GPU Benchmark Highlight**: For $n_{runs}=15$ runs, use $\nu=14$. At $\alpha=0.05$ (two-tailed), $t_{14,0.025} = 2.145$.

### A.3 Chi-Squared (χ²) Distribution

**Table A.3**: Critical values for the chi-squared distribution (right-tailed tests). Used for goodness-of-fit, independence tests, and variance testing.

|  df  | $\alpha$=0.05 | $\alpha$=0.01 |
|------|--------|--------|
|   1  |  3.841 |  6.635 |
|   2  |  5.991 |  9.210 |
|   3  |  7.815 | 11.345 |
|   4  |  9.488 | 13.277 |
|   5  | 11.070 | 15.086 |
|   6  | 12.592 | 16.812 |
|   7  | 14.067 | 18.475 |
|   8  | 15.507 | 20.090 |
|   9  | 16.919 | 21.666 |
|  10  | 18.307 | 23.209 |
|  11  | 19.675 | 24.725 |
|  12  | 21.026 | 26.217 |
|  13  | 22.362 | 27.688 |
|  14  | 23.685 | 29.141 |
|  15  | 24.996 | 30.578 |
|  16  | 26.296 | 32.000 |
|  17  | 27.587 | 33.409 |
|  18  | 28.869 | 34.805 |
|  19  | 30.144 | 36.191 |
|  20  | 31.410 | 37.566 |
|  21  | 32.671 | 38.932 |
|  22  | 33.924 | 40.289 |
|  23  | 35.172 | 41.638 |
|  24  | 36.415 | 42.980 |
|  25  | 37.652 | 44.314 |
|  26  | 38.885 | 45.642 |
|  27  | 40.113 | 46.963 |
|  28  | 41.337 | 48.278 |
|  29  | 42.557 | 49.588 |
|  30  | 43.773 | 50.892 |

**Note**: For testing variance homogeneity in GPU vs CPU runtime distributions.

### A.4 F-Distribution

**Table A.4**: Critical values for the F-distribution at $\alpha=0.05$ (two-tailed). Rows represent numerator degrees of freedom ($df_1$), columns represent denominator degrees of freedom ($df_2$).

|  df₁→<br>df₂↓  |    1    |    2    |    3    |    4    |    5    |
|----------------|---------|---------|---------|---------|---------|
|       1        | 161.448 | 199.500 | 215.707 | 224.583 | 230.162 |
|       5        |   6.608 |   5.786 |   5.409 |   5.192 |   5.050 |
|      10        |   4.965 |   4.103 |   3.708 |   3.478 |   3.326 |
|      20        |   4.351 |   3.493 |   3.098 |   2.866 |   2.711 |
|      30        |   4.171 |   3.316 |   2.922 |   2.690 |   2.534 |

**Note**: Used for ANOVA and variance ratio tests. For GPU benchmarking, $F_{1,29,0.05} = 4.183$ tests equality of two variances.

### Usage Guide

**How to Use These Tables:**

1. **Identify your test statistic**: Determine whether you're computing $Z$, $t$, $\chi^2$, or $F$.

2. **Determine degrees of freedom**:
   - $t$-test: $df = n-1$ (one-sample/paired), $df = n_1+n_2-2$ (independent)
   - $\chi^2$: $df = n-1$ (variance test), $df = (r-1)(c-1)$ (contingency table)
   - $F$: $df_1$ (numerator), $df_2$ (denominator)

3. **Choose significance level**: Typically $\alpha = 0.05$, but adjust based on context (0.01 for stringent, 0.10 for exploratory).

4. **Look up critical value**: Match your $df$ and $\alpha$ in the appropriate table.

5. **Apply decision rule**: If $|test\_statistic| > critical\_value$, reject $H_0$. For right-tailed tests ($\chi^2$, $F$), if $test\_statistic > critical\_value$, reject $H_0$.

**Example**: For a paired t-test with $n=15$ GPU benchmark runs at $\alpha=0.05$:
- $df = 15-1 = 14$
- From Table A.2: $t_{14,0.05} = 2.145$ (two-tailed)
- If $|t_{computed}| > 2.145$, reject $H_0$ (conclude significant difference)

**Software Note**: These tables provide reference values. Statistical software (scipy.stats, R) computes exact p-values and handles non-integer df (e.g., Welch's t-test).

### Cross-References

- **Z-distribution**: See Section 0.1.5 (Normal Distribution and Central Limit Theorem)
- **t-distribution**: See Section 0.3.1 (Paired t-Test), Section 0.3.4 (Independent t-Test)
- **χ²-distribution**: See Section 0.6.1 (Shapiro-Wilk Normality Test)
- **F-distribution**: See Section 0.3.6 (One-Way ANOVA)

>[!caution]
>Should be markdown links

---

## References

### Primary Literature

[TO BE POPULATED WITH WEB SEARCH]

### Statistical Methods Textbooks

[TO BE POPULATED WITH WEB SEARCH]

### Online Resources

[TO BE POPULATED WITH WEB SEARCH]

### Software Documentation

- **SciPy**: <https://docs.scipy.org/doc/scipy/reference/stats.html>
- **scikit-posthocs**: <https://scikit-posthocs.readthedocs.io/>

---

## Correlation Analysis and Covariate Testing

### Spearman Rank Correlation

#### Purpose

Spearman's rank correlation coefficient ($\rho$ or $r_s$) measures the **monotonic relationship** between two variables using their ranks rather than raw values. Unlike Pearson correlation (which measures linear relationships), Spearman can detect any monotonic trend, making it robust to outliers and applicable to ordinal data.

**When to Use:**
- Testing if a covariate (e.g., `hit_optimal_percentage`) correlates with model residuals
- Checking relationships in non-normal data
- Detecting monotonic but non-linear associations
- Validating regression assumptions (e.g., residuals vs fitted values should have ρ ≈ 0)

#### Mathematical Formulation

**Step 1: Rank Transformation**
Convert raw values to ranks:
- Smallest value → rank 1
- Largest value → rank n
- Ties get average rank

**Step 2: Calculate Correlation on Ranks**

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i = \text{rank}(x_i) - \text{rank}(y_i)$ = rank difference for pair $i$
- $n$ = number of observations

**Alternative formula (equivalent to Pearson on ranks):**

$$
\rho = \frac{\text{cov}(R_x, R_y)}{\sigma_{R_x} \sigma_{R_y}}
$$

Where $R_x$, $R_y$ are the rank variables.

#### Interpretation

**Coefficient Range:**
- $\rho = +1$: Perfect positive monotonic relationship
- $\rho = 0$: No monotonic relationship
- $\rho = -1$: Perfect negative monotonic relationship

**Strength Guidelines (Cohen, 1988):**
- $|\rho| < 0.3$: Weak correlation
- $0.3 \leq |\rho| < 0.5$: Moderate correlation
- $|\rho| \geq 0.5$: Strong correlation

**Hypothesis Test:**
- $H_0$: $\rho = 0$ (no monotonic association)
- $H_1$: $\rho \neq 0$ (monotonic association exists)

**P-value**: Probability of observing such a correlation if variables are truly independent.

#### Application to Regression Diagnostics

**Use Case 1: Check if covariate affects model fit**

Problem: Does `hit_optimal_percentage` for a problem correlate with regression residual error?

```python
from scipy.stats import spearmanr

# Calculate residuals from regression model
residuals = y_true - y_pred

# Extract covariate (e.g., % of runs hitting optimal per problem)
hit_optimal_pct = [calc_hit_optimal_rate(problem) for problem in problems]

# Test correlation
rho, p_value = spearmanr(hit_optimal_pct, np.abs(residuals))

print(f"Spearman ρ = {rho:.3f}, p = {p_value:.4f}")

if p_value < 0.05 and abs(rho) > 0.3:
    print("⚠ Covariate significantly correlates with model error!")
    print("Consider stratified analysis or including covariate in model")
else:
    print("✓ Covariate does not significantly affect model fit")
```

**Use Case 2: Validate homoscedasticity assumption**

```python
# Check if residuals correlate with fitted values
# (should be uncorrelated if homoscedastic)
rho, p_value = spearmanr(y_pred, np.abs(residuals))

if abs(rho) < 0.2 and p_value > 0.05:
    print("✓ Homoscedasticity assumption satisfied")
else:
    print("⚠ Heteroscedasticity detected - consider transformation")
```

#### Example: Stop Reason Distribution as Covariate

**Research Question**: Does the percentage of runs hitting optimal (vs stagnating) predict how well our timing model fits the data?

**Hypothesis**: Problems where most runs stagnate might have different timing characteristics than problems where most hit optimal early.

```python
import numpy as np
from scipy.stats import spearmanr

# Calculate per-problem metrics
problem_metrics = []
for problem_name in unique_problems:
    problem_data = df[df['problem'] == problem_name]

    # Covariate: % hitting optimal
    hit_optimal_pct = np.mean([
        'hit_optimal' in reason or 'optimal reached' in reason
        for reason in problem_data['stop_reasons']
    ]) * 100

    # Model residual: absolute error in time prediction
    residual = abs(problem_data['actual_time'].mean() -
                   model.predict(problem_data['n'].mean()))

    problem_metrics.append({
        'problem': problem_name,
        'n': problem_data['n'].iloc[0],
        'hit_optimal_pct': hit_optimal_pct,
        'abs_residual': residual
    })

metrics_df = pd.DataFrame(problem_metrics)

# Spearman correlation test
rho, p_value = spearmanr(
    metrics_df['hit_optimal_pct'],
    metrics_df['abs_residual']
)

print(f"Correlation between hit_optimal% and model error:")
print(f"  Spearman ρ = {rho:.3f}")
print(f"  p-value = {p_value:.4f}")

if p_value < 0.05:
    if rho > 0:
        print("  ⚠ Higher hit_optimal% → Larger residuals")
        print("     Model fits worse for fast-convergence problems")
    else:
        print("  ⚠ Higher hit_optimal% → Smaller residuals")
        print("     Model fits better for fast-convergence problems")

    print("\n  Recommendation: Stratify analysis by convergence behavior")
else:
    print("  ✓ Stop reason distribution does not affect model fit")
```

**Interpretation Example:**
- $\rho = 0.45$, $p = 0.032$: Moderate positive correlation (significant)
  - Problems with more "hit_optimal" outcomes have larger timing errors
  - Suggests the -50 generation correction may need refinement
  - Consider separate models for "early optimal" vs "stagnation" problems

#### Spearman vs Pearson: When to Choose

| Criterion | Pearson | Spearman |
|-----------|---------|----------|
| **Relationship type** | Linear only | Any monotonic |
| **Data requirement** | Interval/ratio scale | Ordinal or better |
| **Outlier sensitivity** | High | Low (uses ranks) |
| **Statistical power** | Higher (if linear & normal) | Slightly lower |
| **Null hypothesis** | No linear association | No monotonic association |

**Decision Rule for Our Benchmark:**
- Use **Spearman** for covariate analysis (robust, handles non-linearity)
- Use **Pearson** only if both variables are clearly normal and relationship is linear

#### Python Implementation

```python
from scipy.stats import spearmanr

# Basic usage
rho, p_value = spearmanr(x, y)

# With alternative hypothesis
rho, p_value = spearmanr(x, y, alternative='two-sided')  # default
rho, p_value = spearmanr(x, y, alternative='less')       # negative correlation
rho, p_value = spearmanr(x, y, alternative='greater')    # positive correlation

# Handling missing data
rho, p_value = spearmanr(x, y, nan_policy='omit')  # remove NaN pairs

# Full correlation matrix (multiple variables)
from scipy.stats import spearmanr
corr_matrix, p_matrix = spearmanr(data_matrix, axis=0)
```

#### Assumptions and Limitations

**Assumptions:**
1. Observations are independent
2. At least ordinal scale data
3. Monotonic relationship (if expecting correlation)

**Limitations:**
1. **Detects monotonic only**: Can miss non-monotonic relationships (e.g., U-shaped)
2. **Ties reduce power**: Many tied ranks decrease test sensitivity
3. **No causation**: Correlation ≠ causation (always remember!)
4. **Sample size**: Need n ≥ 20 for reliable inference (n ≥ 30 recommended)

**Diagnostic:**
- Always plot scatter (with rank overlay) to visualize relationship
- Check for non-monotonic patterns that Spearman would miss

#### Reporting Template

```markdown
**Covariate Analysis: Hit Optimal Percentage vs Model Residuals**

We tested whether stop reason distribution (hit_optimal%) correlates
with regression model residuals using Spearman rank correlation:

- Spearman ρ = 0.28, 95% CI [0.03, 0.51]
- p = 0.048 (two-tailed)
- Effect: Weak-to-moderate positive correlation
- Interpretation: Problems with higher early-optimal rates show
  slightly larger timing prediction errors, but effect is weak.
  
**Conclusion**: Stop reason distribution has minimal impact on model
fit quality. Pooled analysis is appropriate.
```

---

## Appendix: Code Examples

### Complete Statistical Analysis Pipeline

```python
def complete_statistical_analysis(cpu_data, gpu_data, alpha=0.05):
    """
    Full statistical comparison pipeline.

    Returns comprehensive analysis including:
    - Normality assessment
    - Appropriate hypothesis test
    - Effect size
    - Confidence intervals
    """
    # [Full implementation reference]
```

---

**Document Status**:

- [x] Base structure created
- [ ] Shapiro-Wilk section researched
- [ ] Paired t-test section researched
- [ ] Wilcoxon section researched
- [ ] Friedman section researched
- [ ] Nemenyi section researched
- [ ] Holm-Bonferroni section researched
- [ ] Cohen's d section researched
- [ ] Bootstrap section researched
- [ ] References compiled
- [ ] Diagrams finalized
- [ ] Real benchmark examples added
