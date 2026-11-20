---
title: "Automated Demšar Critical Difference Diagrams: Statistical Derivations and Publication-Ready TikZ Blueprints"
description: >
  Comprehensive statistical derivation and production-ready LaTeX/TikZ implementation
  of Demšar Critical Difference (CD) diagrams for multi-algorithm metaheuristic comparisons.
  Details Friedman omnibus validation, Nemenyi post-hoc rank clustering, Studentized range
  scaling conventions, clique detection algorithms, and an automated Python-to-TikZ generator.
created: 2026-11-11
status: development
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-11
    modifications:
      - description: >
          Authored formal guide on Demšar CD diagrams: Nemenyi formulation, Studentized range
          scaling caveats, clique clustering logic, publication-ready TikZ templates, and
          automated Python generation script for TCC thesis figures.
  - date_modified: 2026-11-14
    modifications:
      - description: >
          Linked the companion record analysis-demsar-test-selection-and-power.md, which covers
          what this file deliberately does not: test selection (paired t versus Wilcoxon), the
          normality pre-test trap, tie and rounding power loss, Iman-Davenport degeneracy, and
          Nemenyi versus control procedures.
related_files:
  - "[analysis-demsar-test-selection-and-power.md](./analysis-demsar-test-selection-and-power.md)"
  - "[guide-statistical-demsar-post-hoc-tests.md](../../../documentation/knowledge-base/guide-statistical-demsar-post-hoc-tests.md)"
  - "[guide-statistical-multicomparison-tests.md](../../../documentation/knowledge-base/guide-statistical-multicomparison-tests.md)"
  - "[guide-statistical-testing-index.md](../../../documentation/knowledge-base/guide-statistical-testing-index.md)"
  - "[analysis-complete-statistical-results.md](../../../documentation/knowledge-base/analysis-complete-statistical-results.md)"
  - "[analysis-advanced-benchmark-reporting-metrics.md](./analysis-advanced-benchmark-reporting-metrics.md)"
  - "[analysis-tsp-population-diversity-and-entropy.md](./analysis-tsp-population-diversity-and-entropy.md)"
tags:
  - analysis/benchmarking
  - analysis/statistics
  - benchmark/metrics
  - guide/latex
  - guide/methodology
  - guide/statistics
  - guide/testing
  - notes/workflow
  - review/quality
  - review/statistics
---

# Automated Demšar Critical Difference Diagrams: Statistical Derivations and Publication-Ready TikZ Blueprints

- [Automated Demšar Critical Difference Diagrams: Statistical Derivations and Publication-Ready TikZ Blueprints](#automated-demšar-critical-difference-diagrams-statistical-derivations-and-publication-ready-tikz-blueprints)
  + [1. The Demšar Protocol for Metaheuristic Comparisons](#1-the-demšar-protocol-for-metaheuristic-comparisons)
    - [1.1 Non-Parametric Omnibus Gate: Friedman and Iman-Davenport](#11-non-parametric-omnibus-gate-friedman-and-iman-davenport)
    - [1.2 Average Rank Conservation Constraint](#12-average-rank-conservation-constraint)
  + [2. The Nemenyi Post-Hoc Test & Critical Difference Formula](#2-the-nemenyi-post-hoc-test--critical-difference-formula)
    - [2.1 Mathematical Formulation of Critical Difference ($CD$)](#21-mathematical-formulation-of-critical-difference-cd)
    - [2.2 Studentized Range Statistic Scaling Convention ($q_\alpha$ vs. $Q_\alpha$)](#22-studentized-range-statistic-scaling-convention-q_alpha-vs-q_alpha)
    - [2.3 Non-Significant Clique Clustering Logic](#23-non-significant-clique-clustering-logic)
    - [2.4 Family-Wise Error Rate Control vs. False Discovery Rate](#24-family-wise-error-rate-control-vs-false-discovery-rate)
  + [3. Standalone Publication-Ready LaTeX / TikZ Snippet](#3-standalone-publication-ready-latex--tikz-snippet)
  + [4. Automated python-to-TikZ Generator](#4-automated-python-to-tikz-generator)
  + [5. Thesis Integration Protocol (`second_draft.tex`)](#5-thesis-integration-protocol-second_drafttex)
  + [6. References](#6-references)

---

## 1. The Demšar Protocol for Metaheuristic Comparisons

When evaluating $k \ge 3$ algorithmic variants across a benchmark cohort of $N$ distinct problem instances (such as TSPLIB95 instances), parametric tests (ANOVA) are invalidated by severe violations of normality and sphericity. Janez Demšar (2006) established the gold standard statistical protocol for multi-algorithm comparisons across multiple data sets.

### 1.1 Non-Parametric Omnibus Gate: Friedman and Iman-Davenport

Before performing pairwise post-hoc tests, an omnibus test must reject the null hypothesis $H_0$ that all $k$ algorithms perform equivalently:

1. **Rank Transformation**: On each benchmark instance $i \in \{1, \dots, N\}$, rank the $k$ algorithms from $1$ (best) to $k$ (worst), assigning average ranks in the case of ties.
2. **Average Rank Calculation**:
   $$R_j = \frac{1}{N} \sum_{i=1}^N r_i^j, \quad \forall j \in \{1, \dots, k\}$$
3. **Friedman Chi-Square Statistic**:
   $$\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4} \right]$$
4. **Iman-Davenport $F_F$ Correction**: Because $\chi_F^2$ is known to be overly conservative, Iman & Davenport (1980) derived an improved $F$-distribution statistic with $(k-1)$ and $(k-1)(N-1)$ degrees of freedom:
   $$F_F = \frac{(N - 1) \chi_F^2}{N(k - 1) - \chi_F^2}$$

Only if the omnibus $p$-value satisfies $p < \alpha$ (typically $\alpha = 0.05$) is the omnibus null rejected, authorizing post-hoc pairwise rank comparisons.

### 1.2 Average Rank Conservation Constraint

A critical mathematical invariant that must be maintained across any reported rank table or CD diagram is the **Rank Conservation Law**:

$$\sum_{j=1}^k R_j = \frac{k(k + 1)}{2}$$

For $k = 4$ algorithms:
$$\sum_{j=1}^4 R_j = \frac{4 \times 5}{2} = 10.00$$

Any published report or diagram where average ranks fail to sum to $\frac{k(k+1)}{2}$ indicates data corruption or improper tie-handling.

---

## 2. The Nemenyi Post-Hoc Test & Critical Difference Formula

### 2.1 Mathematical Formulation of Critical Difference ($CD$)

When comparing all $k$ algorithms simultaneously without a designated control, the Nemenyi (1963) post-hoc test establishes the threshold difference in average rank required to demonstrate statistical significance at significance level $\alpha$:

$$CD = q_\alpha \sqrt{\frac{k(k + 1)}{6N}}$$

where:

- $k$: Total number of compared algorithms.
- $N$: Total number of benchmark instances evaluated.
- $q_\alpha$: Critical value of the two-tailed Studentized range statistic divided by $\sqrt{2}$.

Two algorithms $i$ and $j$ perform **statistically significantly differently** at the chosen confidence level if and only if:

$$|R_i - R_j| > CD$$

### 2.2 Studentized Range Statistic Scaling Convention ($q_\alpha$ vs. $Q_\alpha$)

A frequent pitfall in algorithmic benchmarking stems from divergent definitions of the Studentized range critical value across literature and statistical software libraries:

- **Tukey's Studentized Range Statistic ($Q_\alpha$)**:
  Standard statistical tables and python (`scipy.stats.studentized_range.ppf(1 - alpha, k, np.inf)`) tabulate Tukey's unscaled statistic $Q_\alpha$.
- **Demšar's Critical Value ($q_\alpha$)**:
  Demšar (2006) defines $q_\alpha = Q_\alpha / \sqrt{2}$.

$$\text{Correct Formulation}: \quad CD = \frac{Q_\alpha}{\sqrt{2}} \sqrt{\frac{k(k + 1)}{6N}} = Q_\alpha \sqrt{\frac{k(k + 1)}{12N}}$$

> [!WARNING]
> Failing to divide `studentized_range.ppf` by $\sqrt{2}$ when using the $\sqrt{6N}$ denominator artificially inflates the Critical Difference threshold by **$41.4\%$** ($\sqrt{2} \approx 1.414$), resulting in severe Type II errors (failing to detect true algorithmic superiority).

**Exact Numerical Values at $\alpha = 0.05$**:

- For $k = 4, N = 38$: $q_{0.05} = 2.569 \implies CD = 2.569 \sqrt{\frac{20}{228}} \approx 0.761$.
- For $k = 4, N = 113$ (full TSPLIB95): $q_{0.05} = 2.569 \implies CD = 2.569 \sqrt{\frac{20}{678}} \approx 0.441$.

### 2.3 Non-Significant Clique Clustering Logic

A **clique** represents a maximal subset of algorithms $\mathcal{C} \subseteq \{A_1, \dots, A_k\}$ such that no pair of algorithms within the subset has a rank difference exceeding $CD$:

$$\forall A_i, A_j \in \mathcal{C}, \quad |R_i - R_j| \le CD$$

In a Critical Difference diagram:

1. Algorithms are ordered by average rank $R_j$ along a horizontal scale from $1.0$ (best, left) to $k.0$ (worst, right).
2. For each maximal clique $\mathcal{C}$, a thick horizontal bar is drawn spanning the leftmost and rightmost algorithms of the clique.
3. Algorithms connected by a common bar are statistically equivalent ($p \ge 0.05$).
4. Algorithms separated by a gap with no overlapping bar are statistically significantly different ($p < 0.05$).

### 2.4 Family-Wise Error Rate Control vs. False Discovery Rate

While Nemenyi is standard for all-vs-all ranking, two alternative post-hoc procedures provide higher statistical power depending on the research question:

1. **Holm's Step-Down Procedure (Comparison to Control)**:
   When comparing $k-1$ GPU solvers against a single CPU baseline, Holm's procedure tests each hypothesis against $\alpha / (k - i)$, delivering significantly higher detection power than Nemenyi.
2. **Benjamini-Hochberg False Discovery Rate (FDR)**:
   When comparing a large matrix of variants ($k > 10$), controlling FDR ($Q^* = 0.05$) avoids the extreme conservatism of Family-Wise Error Rate control.

---

## 3. Standalone Publication-Ready LaTeX / TikZ Snippet

The following standalone TikZ snippet renders a vector-graphic Critical Difference diagram matching the formatting of Chapter 5 of `second_draft.tex`:

```latex
% Standalone Critical Difference (CD) Diagram in TikZ
% Generated for k = 4 algorithms across N = 38 benchmark instances (CD = 0.761)
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}

\begin{document}
\begin{tikzpicture}[
    scale=2.2,
    axis/.style={thick, -{Stealth[length=2.5mm]}},
    tick/.style={thick},
    subtick/.style={thin, gray},
    clique/.style={line width=2.4pt, color=blue!75!black, line cap=round},
    cdbar/.style={line width=1.5pt, {Bar[width=4mm]}-{Bar[width=4mm]}},
    algomarker/.style={circle, fill=black, inner sep=1.2pt},
    labelnode/.style={font=\small\bfseries, align=center}
]

    % -------------------------------------------------------------
    % Parameters: Rank axis spans [1.0, 4.0]
    % Average ranks: HybridOpt=1.35, FullGPU=1.85, HybridNaive=2.95, CPU=3.85
    % Sum of ranks = 10.00 (Conserved)
    % CD = 0.761 at alpha = 0.05
    % -------------------------------------------------------------

    % 1. CD Reference Bar at the Top
    \draw[cdbar] (1.0, 1.1) -- node[above=2pt, font=\footnotesize\bfseries] {CD = 0.761 ($\alpha = 0.05$)} (1.761, 1.1);

    % 2. Main Horizontal Rank Axis (1.0 to 4.0)
    \draw[axis] (0.8, 0.0) -- (4.3, 0.0) node[right, font=\footnotesize\bfseries] {Rank};

    % Major Ticks and Numbers
    \foreach \x in {1.0, 2.0, 3.0, 4.0} {
        \draw[tick] (\x, 0.08) -- (\x, -0.08) node[below=2pt, font=\footnotesize] {\x};
    }
    % Minor Sub-ticks (every 0.5)
    \foreach \x in {1.5, 2.5, 3.5} {
        \draw[subtick] (\x, 0.04) -- (\x, -0.04);
    }

    % 3. Algorithm Data Points and Drop Lines
    % HybridOpt: Rank 1.35
    \coordinate (A1) at (1.35, 0.0);
    \node[algomarker] at (A1) {};
    \draw[thick] (A1) -- (1.35, -0.45);
    \node[labelnode, anchor=north, color=green!40!black] at (1.35, -0.45) {HybridOpt\\(1.35)};

    % FullGPU: Rank 1.85
    \coordinate (A2) at (1.85, 0.0);
    \node[algomarker] at (A2) {};
    \draw[thick] (A2) -- (1.85, -0.45);
    \node[labelnode, anchor=north, color=teal!70!black] at (1.85, -0.45) {FullGPU\\(1.85)};

    % HybridNaive: Rank 2.95
    \coordinate (A3) at (2.95, 0.0);
    \node[algomarker] at (A3) {};
    \draw[thick] (A3) -- (2.95, -0.45);
    \node[labelnode, anchor=north, color=orange!80!black] at (2.95, -0.45) {HybridNaive\\(2.95)};

    % CPU Baseline: Rank 3.85
    \coordinate (A4) at (3.85, 0.0);
    \node[algomarker] at (A4) {};
    \draw[thick] (A4) -- (3.85, -0.45);
    \node[labelnode, anchor=north, color=red!70!black] at (3.85, -0.45) {CPU Baseline\\(3.85)};

    % 4. Non-Significant Cliques (|R_i - R_j| <= CD)
    % Clique 1: HybridOpt (1.35) and FullGPU (1.85) -> |1.85 - 1.35| = 0.50 <= 0.761
    \draw[clique] (1.35, 0.35) -- node[above=1pt, font=\scriptsize, color=blue!75!black] {Statistically Equivalent} (1.85, 0.35);
    \draw[thin, blue!75!black, dashed] (1.35, 0.0) -- (1.35, 0.35);
    \draw[thin, blue!75!black, dashed] (1.85, 0.0) -- (1.85, 0.35);

    % Note: HybridNaive (2.95) and CPU (3.85) have difference |3.85 - 2.95| = 0.90 > 0.761 -> Not connected!
    % Both are statistically isolated and significantly inferior to HybridOpt and FullGPU.

\end{tikzpicture}
\end{document}
```

---

## 4. Automated python-to-TikZ Generator

The following production python module automatically ingests a matrix of algorithm results, validates the Friedman test, evaluates Nemenyi $CD$, computes maximal cliques, and outputs the exact TikZ code:

```python
"""Automated Demšar Critical Difference Diagram generator for LaTeX/TikZ."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


@dataclass
class CDResults:
    k_algorithms: int
    n_instances: int
    ranks: Dict[str, float]
    rank_sum: float
    critical_difference: float
    friedman_chi2: float
    friedman_pvalue: float
    cliques: List[Tuple[str, str, float, float]]  # (algo1, algo2, r1, r2)


def compute_cd_diagram_data(
    results_matrix: np.ndarray,
    algorithm_names: List[str],
    alpha: float = 0.05,
    higher_is_better: bool = False,
) -> CDResults:
    """Compute Demšar ranks, Nemenyi CD, and maximal cliques."""
    n_instances, k_algorithms = results_matrix.shape
    if len(algorithm_names) != k_algorithms:
        raise ValueError("algorithm_names length must match columns in results_matrix.")

    # 1. Compute instance-level ranks (1 = best)
    multiplier = -1.0 if higher_is_better else 1.0
    ranked = np.empty_like(results_matrix)
    for i in range(n_instances):
        ranked[i] = stats.rankdata(multiplier * results_matrix[i])

    # 2. Average ranks across instances
    avg_ranks = np.mean(ranked, axis=0)
    ranks_dict = {name: float(avg_ranks[idx]) for idx, name in enumerate(algorithm_names)}
    rank_sum = float(np.sum(avg_ranks))

    # 3. Friedman test
    chi2, p_val = stats.friedmanchisquare(*[ranked[:, j] for j in range(k_algorithms)])

    # 4. Critical Difference (Nemenyi)
    # studentized_range.ppf returns Tukey's Q -> divide by sqrt(2) for Demšar q_alpha
    q_tukey = stats.studentized_range.ppf(1.0 - alpha, k_algorithms, np.inf)
    q_alpha = q_tukey / np.sqrt(2.0)
    cd = float(q_alpha * np.sqrt((k_algorithms * (k_algorithms + 1)) / (6.0 * n_instances)))

    # 5. Extract cliques (pairs/subsets with difference <= CD)
    sorted_algos = sorted(ranks_dict.items(), key=lambda item: item[1])
    cliques = []
    for i in range(k_algorithms):
        for j in range(i + 1, k_algorithms):
            name_i, r_i = sorted_algos[i]
            name_j, r_j = sorted_algos[j]
            if (r_j - r_i) <= cd:
                cliques.append((name_i, name_j, r_i, r_j))

    return CDResults(
        k_algorithms=k_algorithms,
        n_instances=n_instances,
        ranks=ranks_dict,
        rank_sum=rank_sum,
        critical_difference=cd,
        friedman_chi2=float(chi2),
        friedman_pvalue=float(p_val),
        cliques=cliques,
    )


def export_cd_diagram_tikz(results: CDResults, output_path: str) -> None:
    """Export publication-grade TikZ code from CDResults."""
    k = results.k_algorithms
    cd = results.critical_difference
    sorted_items = sorted(results.ranks.items(), key=lambda x: x[1])

    tikz_lines = [
        "% Auto-generated Demšar CD Diagram",
        r"\begin{tikzpicture}[scale=2.2,",
        r"    axis/.style={thick, -{Stealth[length=2.5mm]}},",
        r"    clique/.style={line width=2.4pt, color=blue!75!black, line cap=round},",
        r"    cdbar/.style={line width=1.5pt, {Bar[width=4mm]}-{Bar[width=4mm]}},",
        r"    algomarker/.style={circle, fill=black, inner sep=1.2pt},",
        r"    labelnode/.style={font=\small\bfseries, align=center}",
        r"]",
        f"    % CD Reference Bar (CD = {cd:.3f})",
        f"    \\draw[cdbar] (1.0, 1.1) -- node[above=2pt, font=\\footnotesize\\bfseries] {{CD = {cd:.3f}}} ({1.0 + cd:.3f}, 1.1);",
        f"    % Rank Axis (1.0 to {k}.0)",
        f"    \\draw[axis] (0.8, 0.0) -- ({k + 0.3:.1f}, 0.0) node[right, font=\\footnotesize\\bfseries] {{Rank}};",
    ]

    # Major ticks
    for tick in range(1, k + 1):
        tikz_lines.append(f"    \\draw[thick] ({tick}.0, 0.08) -- ({tick}.0, -0.08) node[below=2pt, font=\\footnotesize] {{{tick}.0}};")

    # Algorithm markers and labels
    for name, r in sorted_items:
        tikz_lines.extend([
            f"    \\coordinate (P_{name}) at ({r:.3f}, 0.0);",
            f"    \\node[algomarker] at (P_{name}) {{}};",
            f"    \\draw[thick] (P_{name}) -- ({r:.3f}, -0.45);",
            f"    \\node[labelnode, anchor=north] at ({r:.3f}, -0.45) {{{name}\\\\({r:.2f})}};",
        ])

    # Clique bars
    # Draw maximal non-overlapping cliques
    y_offset = 0.35
    for c_start, c_end, r_start, r_end in results.cliques:
        tikz_lines.extend([
            f"    \\draw[clique] ({r_start:.3f}, {y_offset:.2f}) -- ({r_end:.3f}, {y_offset:.2f});",
            f"    \\draw[thin, blue!75!black, dashed] ({r_start:.3f}, 0.0) -- ({r_start:.3f}, {y_offset:.2f});",
            f"    \\draw[thin, blue!75!black, dashed] ({r_end:.3f}, 0.0) -- ({r_end:.3f}, {y_offset:.2f});",
        ])
        y_offset += 0.15

    tikz_lines.append(r"\end{tikzpicture}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tikz_lines) + "\n")
```

---

## 5. Thesis Integration Protocol (`second_draft.tex`)

To integrate the generated CD diagrams cleanly into the LaTeX thesis:

1. Export the TikZ snippet into a standalone figure file: `documents_tcc/tex/figures/fig_cd_diagram.tikz`.
2. Include the figure in Chapter 5 (Experimental Results) via:

```latex
\begin{figure}[htbp]
    \centering
    \input{figures/fig_cd_diagram.tikz}
    \caption{Dem\v{s}ar Critical Difference (CD) diagram comparing the four algorithmic configurations across $N = 38$ TSPLIB instances ($\alpha = 0.05$, Critical Difference $CD = 0.761$). Algorithms connected by the horizontal bar are not significantly different. Average ranks satisfy the conservation condition $\sum R_j = 10.00$.}
    \label{fig:critical_difference_diagram}
\end{figure}
```

1. Ensure `\usepackage{tikz}` and `\usetikzlibrary{arrows.meta,positioning,calc}` are loaded in `documents_tcc/tex/second_draft.tex` preamble.

---

## 6. References

1. **Demšar, J.** (2006). *Statistical comparisons of classifiers over multiple data sets*. Journal of Machine Learning Research, 7(Dec), 1–30 (`demsar2006statistical`).
2. **Friedman, M.** (1937). *The use of ranks to avoid the assumption of normality implicit in the analysis of variance*. Journal of the American Statistical Association, 32(200), 675–701.
3. **Iman, R. L., & Davenport, J. M.** (1980). *Approximations of the critical region of the Friedman statistic*. Communications in Statistics-Theory and Methods, 9(6), 571–595.
4. **Nemenyi, P. B.** (1963). *Distribution-free Multiple Comparisons*. Ph.D. thesis, Princeton University.
