---
title: "Knowledge Base Index — GPU Routing Architecture, Statistical Testing, and Tooling"
description: >
  Thematic index of the curated knowledge-base directory covering GPU memory
  hierarchy, CUDA 2-opt parallelization and reduction patterns, Simulated Annealing
  dynamics, Memetic Genetic Algorithms, frequentist/non-parametric statistical
  validation, and Pandoc/LaTeX authoring guides.
created: 2025-01-28
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
          Rewrote the knowledge-base index after modularizing monolithic guides
          (statistical testing, 2-opt parallelization, Simulated Annealing, and
          GPU hardware/memory regimes) and enforcing repository Markdown standards.
related_files:
  - "[README.md](../README.md)"
  - "[guide-local-search-pipeline-stages.md](../guide-local-search-pipeline-stages.md)"
  - "[analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md)"
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
tags:
  - guide/architecture
  - guide/documentation
  - guide/cuda
  - guide/testing
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/simulated-annealing
  - algorithm/tsp
  - analysis/performance
  - analysis/statistics
  - benchmark/gpu
  - notes/cuda
---

# Knowledge base index

Curated technical reference articles, mathematical derivations, and engineering guides for the GPU-accelerated routing solver. For the repository-wide documentation map, see [documentation/README.md](../README.md).

## 1. GPU architecture, memory hierarchy, and synchronization

- [analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md) — Dense VRAM matrix vs on-the-fly coordinate evaluation, Pascal GP107 shared-memory limits, and multi-tour occupancy break-even ($T^* = 2048$).
- [analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md) — GPU memory latency hierarchy (registers, shared memory, L1/L2, GDDR5), bank conflicts, and 256-byte allocation granularity.
- [analysis-gpu-intra-inter-route-regimes.md](./analysis-gpu-intra-inter-route-regimes.md) — Intra-route (`two_opt_single.cu`), population batch (`two_opt_batch.cu`), and hybrid tiled execution regimes.
- [analysis-gpu-memory-calculation.md](./analysis-gpu-memory-calculation.md) — Analytical upper-bound VRAM sizing formulas across TSPLIB95 problem scales.
- [guide-gpu-vram-guardrails-buffer-pooling.md](./guide-gpu-vram-guardrails-buffer-pooling.md) — Pre-flight `VRAMInsufficientError` checks, persistent scratchpad pooling, P-Data vs P-Task streams, and PCIe Roofline diagnostics.
- [guide-cuda-synchronization-levels.md](./guide-cuda-synchronization-levels.md) — Thread, warp (`__shfl_down_sync`), block (`__syncthreads`), and host-device synchronization primitives.
- [analysis-non-power-of-two-blocks.md](./analysis-non-power-of-two-blocks.md) — Formal proof of orphaned-thread failure modes in non-power-of-two tree reductions.
- [guide-cupy-raw-kernels-optimization.md](./guide-cupy-raw-kernels-optimization.md) — Authoring and compiling custom CUDA C++ `RawKernel` modules in CuPy.
- [notes_cupy_amd_rocm_compatibility.md](./notes_cupy_amd_rocm_compatibility.md) — Portability notes for running CuPy kernels on AMD ROCm / HIP backends.

## 2. Parallel 2-opt local search and GPU reductions

- [analysis-gpu-2opt-performance-modeling.md](./analysis-gpu-2opt-performance-modeling.md) — Symbol table, Pascal GP107 hardware bounds, and analytical runtime/occupancy modeling for 2-opt.
- [analysis-gpu-2opt-parallel-architectures.md](./analysis-gpu-2opt-parallel-architectures.md) — Architectural comparison of HybridNaive, HybridOptimized, and FullGPU 2-opt execution pipelines.
- [analysis-gpu-2opt-reduction-strategies.md](./analysis-gpu-2opt-reduction-strategies.md) — Shared-memory tree reductions, warp-shuffle argmin patterns, and CUB integration trade-offs.
- [analysis-gpu-2opt-kernel-benchmarking.md](./analysis-gpu-2opt-kernel-benchmarking.md) — Micro-benchmark methodology, empirical speedup profiles, and synchronization overhead isolation.
- [analysis-gpu-2opt-literature-attribution.md](./analysis-gpu-2opt-literature-attribution.md) — Mapping implemented CUDA 2-opt choices to Rocki & Suda (2012, 2013) and O'Neil & Burtscher (2015).
- [analysis-gpu-2opt-integration-postmortem.md](./analysis-gpu-2opt-integration-postmortem.md) — Engineering postmortem on integrating GPU 2-opt into the modular strategy pipeline.
- [analysis-gpu-reduction-patterns.md](./analysis-gpu-reduction-patterns.md) — Deep-dive reference on parallel reduction bug root causes, warp masks, and VRP operator scope.
- [analysis-gpu-performance-bottlenecks.md](./analysis-gpu-performance-bottlenecks.md) — Profiling host-device transfer bottlenecks, Python dispatch overhead, and kernel occupancy limits.

## 3. Metaheuristics, problem formulations, and software design

- [analysis-problem-definition-tsp-cvrp.md](./analysis-problem-definition-tsp-cvrp.md) — Formal mathematical definitions for symmetric/asymmetric TSP and Capacitated VRP (CVRP).
- [guide-construction-heuristics.md](./guide-construction-heuristics.md) — Nearest Neighbor, Clarke-Wright Savings, and greedy tour construction heuristics.
- [analysis-compiling-tsp-heuristics.md](./analysis-compiling-tsp-heuristics.md) — Performance trade-offs across pure Python, NumPy vectorization, JIT compilation, and CUDA kernels.
- [analysis-genetic-algorithm-memetic-lineage.md](./analysis-genetic-algorithm-memetic-lineage.md) — Holland GA -> Moscato Memetic Algorithm lineage, discrete tournament selection math, and strategy composition.
- [guide-ga-hyperparameter-tuning.md](./guide-ga-hyperparameter-tuning.md) — Population sizing, crossover/mutation rates, and local-search intensity tuning benchmarks.
- [guide-iso-algorithmic-ga-validation.md](./guide-iso-algorithmic-ga-validation.md) — Protocol for verifying algorithmic equivalence between CPU and GPU Genetic Algorithm implementations.
- [guide-gpu-cpu-comparison-methodology.md](./guide-gpu-cpu-comparison-methodology.md) — Fair benchmarking rules, warm-up handling, stopping criteria, and hardware normalization caveats.
- [analysis-simulated-annealing-foundations.md](./analysis-simulated-annealing-foundations.md) — Talbi's S-Task vs P-Data taxonomy, Metropolis criterion, Kirkpatrick initial temperature, and adaptive cooling.
- [analysis-simulated-annealing-gpu-execution.md](./analysis-simulated-annealing-gpu-execution.md) — CUDA kernel launch latency vs arithmetic intensity, multi-tier SA parallelization, and `ch150` benchmarks.
- [analysis-strategy-hierarchy-and-context-contracts.md](./analysis-strategy-hierarchy-and-context-contracts.md) — Slot-prefixed `StrategyBase` hierarchy, `ProblemContext` GPU cache ownership, and three-tier TSPLIB95 distances.
- [notes_average_worst_case_analysis.md](./notes_average_worst_case_analysis.md) — Asymptotic worst-case vs empirical average-case complexity notes for routing heuristics.
- [guide-clean-python-practices.md](./guide-clean-python-practices.md) — Type annotations, protocol-driven design, and clean Python conventions used in `src/`.

## 4. Statistical validation suite

- [guide-statistical-testing-index.md](./guide-statistical-testing-index.md) — Master decision tree and navigation hub for the 6-part statistical testing series.
- [guide-statistical-parametric-foundations.md](./guide-statistical-parametric-foundations.md) — Hypothesis testing mechanics, Type I/II errors, confidence intervals, and parametric assumptions.
- [guide-statistical-normality-and-outliers.md](./guide-statistical-normality-and-outliers.md) — Shapiro-Wilk, D'Agostino-Pearson, Q-Q diagnostics, skewness/heavy tails in heuristic costs, and IQR outlier rules.
- [guide-statistical-pairwise-comparisons.md](./guide-statistical-pairwise-comparisons.md) — Paired $t$-test, Welch's $t$-test, Wilcoxon signed-rank, and Mann-Whitney U protocols.
- [guide-statistical-multicomparison-tests.md](./guide-statistical-multicomparison-tests.md) — One-way ANOVA, Friedman aligned/rank tests, Holm/Bonferroni family-wise error control, and omnibus workflows.
- [guide-statistical-demsar-post-hoc-tests.md](./guide-statistical-demsar-post-hoc-tests.md) — Demšar (2006) multi-problem protocol, Nemenyi Critical Difference diagrams, and SciPy `studentized_range` scaling.
- [guide-statistical-effect-sizes-and-power.md](./guide-statistical-effect-sizes-and-power.md) — Cohen's $d$, Cliff's $\delta$, Vargha-Delaney $\hat{A}_{12}$, Kendall's $W$, and sample-size power calculations.
- [analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md) — Empirical statistical benchmark evaluation across CPU, HybridNaive, HybridOpt, and FullGPU solvers.
- [notes_types_of_variables_statistics.md](./notes_types_of_variables_statistics.md) — Nominal, ordinal, interval, and ratio measurement scales mapped to benchmark metrics.

## 5. Literature notes, authoring guides, and LaTeX templates

- [useful_notes/notes_cook2012_pursuit_of_tsp.md](./useful_notes/notes_cook2012_pursuit_of_tsp.md) — Structured reading notes on Cook (2012), *In Pursuit of the Traveling Salesman*.
- [useful_notes/notes_crainic2003_metaheuristics.md](./useful_notes/notes_crainic2003_metaheuristics.md) — Structured reading notes on Crainic & Toulouse (2003), *Parallel Strategies for Meta-Heuristics*.
- [guide-latex-math-notation.md](./guide-latex-math-notation.md) — Mathematical typesetting conventions and LaTeX symbol reference for thesis chapters and Markdown docs.
- [guide-pandoc-latex-compilation.md](./guide-pandoc-latex-compilation.md) — Pandoc + LuaLaTeX/XeLaTeX + BibLaTeX build pipeline and troubleshooting guide.
- [guide-mermaid-diagram-authoring.md](./guide-mermaid-diagram-authoring.md) — Supported Mermaid diagram types, label quoting rules, and rendering tips.
- LaTeX templates & cheat sheets (preserved in `tex-docs/`):
  - [tex-docs/CheatSheet.tex](./tex-docs/CheatSheet.tex)
  - [tex-docs/Template.tex](./tex-docs/Template.tex)
