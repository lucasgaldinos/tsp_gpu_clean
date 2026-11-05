---
title: "Documentation Catalog — GPU-Accelerated Routing Optimization"
description: >
  Master catalog and navigation map for the repository documentation tree,
  organizing architectural analyses, GPU/statistical knowledge-base guides,
  academic literature appraisals, engineering phase walkthroughs, and archived
  exploratory notes.
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
          Created master documentation catalog indexing all 76 curated files across
          knowledge-base, analysis, critical_analysis, walkthroughs, and archive
          in compliance with repository Markdown standards.
related_files:
  - "[AGENTS.md](../AGENTS.md)"
  - "[markdown.instructions.md](../.agents/rules/markdown.instructions.md)"
  - "[guide-local-search-pipeline-stages.md](./guide-local-search-pipeline-stages.md)"
  - "[knowledge-base/README.md](./knowledge-base/README.md)"
tags:
  - guide/documentation
  - guide/architecture
  - guide/cuda
  - guide/testing
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/simulated-annealing
  - algorithm/tsp
  - analysis/architecture
  - analysis/performance
  - analysis/statistics
  - benchmark/gpu
---

# Documentation catalog

Central navigation hub for `documentation/`. All Markdown documents follow [markdown.instructions.md](../.agents/rules/markdown.instructions.md) (`<type-prefix><slug>.md` up to 50 characters, structured YAML frontmatter with 10–15 scoped tags, explicit code-fence identifiers, and verified relative links).

## Directory overview

| Directory / File | Purpose | Count |
| --- | --- | --- |
| [guide-local-search-pipeline-stages.md](./guide-local-search-pipeline-stages.md) | Root architectural guide on composing `LocalSearchStage` within Genetic Algorithms | 1 |
| [knowledge-base/](./knowledge-base/README.md) | Modular technical guides on GPU hardware, CUDA 2-opt, Simulated Annealing, Memetic GAs, statistics, and LaTeX tooling | 46 |
| `analysis/` | Implementation analyses of the four GA solver variants, benchmarking pipeline, thesis slides, and scaling trade-offs | 12 |
| `critical_analysis/` | Academic literature synthesis, prior-art comparison, and critical appraisal checklists | 4 |
| `walkthroughs/` | Phase-by-phase engineering walkthroughs (Phases 1–10) and test-suite verification records | 10 |
| `archive/` | Preserved historical/exploratory notes outside current solver scope (ACO, AODV, GLS, early 2024 reading log) | 4 |

## 1. Root pipeline guide

- [guide-local-search-pipeline-stages.md](./guide-local-search-pipeline-stages.md) — Integrating optional post-mutation and post-generation `LocalSearchStage` passes in the modular Genetic Algorithm pipeline.

## 2. Knowledge base (`knowledge-base/`)

See [knowledge-base/README.md](./knowledge-base/README.md) for the full thematic breakdown:

- **GPU memory, occupancy & CUDA kernels**:
  - [analysis-gpu-distance-and-shared-memory-limits.md](./knowledge-base/analysis-gpu-distance-and-shared-memory-limits.md)
  - [analysis-gpu-memory-latency-and-occupancy.md](./knowledge-base/analysis-gpu-memory-latency-and-occupancy.md)
  - [analysis-gpu-intra-inter-route-regimes.md](./knowledge-base/analysis-gpu-intra-inter-route-regimes.md)
  - [analysis-gpu-memory-calculation.md](./knowledge-base/analysis-gpu-memory-calculation.md)
  - [guide-gpu-vram-guardrails-buffer-pooling.md](./knowledge-base/guide-gpu-vram-guardrails-buffer-pooling.md)
  - [guide-cuda-synchronization-levels.md](./knowledge-base/guide-cuda-synchronization-levels.md) & [analysis-non-power-of-two-blocks.md](./knowledge-base/analysis-non-power-of-two-blocks.md)
  - [guide-cupy-raw-kernels-optimization.md](./knowledge-base/guide-cupy-raw-kernels-optimization.md) & [notes_cupy_amd_rocm_compatibility.md](./knowledge-base/notes_cupy_amd_rocm_compatibility.md)
- **2-opt parallelization & GPU reductions (8-file suite)**:
  - [analysis-gpu-2opt-performance-modeling.md](./knowledge-base/analysis-gpu-2opt-performance-modeling.md)
  - [analysis-gpu-2opt-parallel-architectures.md](./knowledge-base/analysis-gpu-2opt-parallel-architectures.md)
  - [analysis-gpu-2opt-reduction-strategies.md](./knowledge-base/analysis-gpu-2opt-reduction-strategies.md)
  - [analysis-gpu-2opt-kernel-benchmarking.md](./knowledge-base/analysis-gpu-2opt-kernel-benchmarking.md)
  - [analysis-gpu-2opt-literature-attribution.md](./knowledge-base/analysis-gpu-2opt-literature-attribution.md)
  - [analysis-gpu-2opt-integration-postmortem.md](./knowledge-base/analysis-gpu-2opt-integration-postmortem.md)
  - [analysis-gpu-reduction-patterns.md](./knowledge-base/analysis-gpu-reduction-patterns.md)
  - [analysis-gpu-performance-bottlenecks.md](./knowledge-base/analysis-gpu-performance-bottlenecks.md)
- **Metaheuristics & software architecture**:
  - [analysis-problem-definition-tsp-cvrp.md](./knowledge-base/analysis-problem-definition-tsp-cvrp.md)
  - [guide-construction-heuristics.md](./knowledge-base/guide-construction-heuristics.md)
  - [analysis-compiling-tsp-heuristics.md](./knowledge-base/analysis-compiling-tsp-heuristics.md)
  - [analysis-genetic-algorithm-memetic-lineage.md](./knowledge-base/analysis-genetic-algorithm-memetic-lineage.md)
  - [guide-ga-hyperparameter-tuning.md](./knowledge-base/guide-ga-hyperparameter-tuning.md)
  - [guide-iso-algorithmic-ga-validation.md](./knowledge-base/guide-iso-algorithmic-ga-validation.md)
  - [guide-gpu-cpu-comparison-methodology.md](./knowledge-base/guide-gpu-cpu-comparison-methodology.md)
  - [analysis-simulated-annealing-foundations.md](./knowledge-base/analysis-simulated-annealing-foundations.md) & [analysis-simulated-annealing-gpu-execution.md](./knowledge-base/analysis-simulated-annealing-gpu-execution.md)
  - [analysis-strategy-hierarchy-and-context-contracts.md](./knowledge-base/analysis-strategy-hierarchy-and-context-contracts.md)
  - [notes_average_worst_case_analysis.md](./knowledge-base/notes_average_worst_case_analysis.md) & [guide-clean-python-practices.md](./knowledge-base/guide-clean-python-practices.md)
- **Statistical validation suite (9-file suite)**:
  - [guide-statistical-testing-index.md](./knowledge-base/guide-statistical-testing-index.md) (entry point)
  - [guide-statistical-parametric-foundations.md](./knowledge-base/guide-statistical-parametric-foundations.md)
  - [guide-statistical-normality-and-outliers.md](./knowledge-base/guide-statistical-normality-and-outliers.md)
  - [guide-statistical-pairwise-comparisons.md](./knowledge-base/guide-statistical-pairwise-comparisons.md)
  - [guide-statistical-multicomparison-tests.md](./knowledge-base/guide-statistical-multicomparison-tests.md)
  - [guide-statistical-demsar-post-hoc-tests.md](./knowledge-base/guide-statistical-demsar-post-hoc-tests.md)
  - [guide-statistical-effect-sizes-and-power.md](./knowledge-base/guide-statistical-effect-sizes-and-power.md)
  - [analysis-complete-statistical-results.md](./knowledge-base/analysis-complete-statistical-results.md) & [notes_types_of_variables_statistics.md](./knowledge-base/notes_types_of_variables_statistics.md)
- **Reading notes, authoring & LaTeX templates**:
  - [notes_cook2012_pursuit_of_tsp.md](./knowledge-base/useful_notes/notes_cook2012_pursuit_of_tsp.md) & [notes_crainic2003_metaheuristics.md](./knowledge-base/useful_notes/notes_crainic2003_metaheuristics.md)
  - [guide-latex-math-notation.md](./knowledge-base/guide-latex-math-notation.md), [guide-pandoc-latex-compilation.md](./knowledge-base/guide-pandoc-latex-compilation.md), & [guide-mermaid-diagram-authoring.md](./knowledge-base/guide-mermaid-diagram-authoring.md)
  - [CheatSheet.tex](./knowledge-base/tex-docs/CheatSheet.tex) & [Template.tex](./knowledge-base/tex-docs/Template.tex)

## 3. Solver, benchmark, and presentation analyses (`analysis/`)

- **Algorithm variants (`analysis/algorithms/`)**:
  - [analysis-genetic-algorithm-cpu.md](./analysis/algorithms/analysis-genetic-algorithm-cpu.md)
  - [analysis-genetic-algorithm-hybrid-naive.md](./analysis/algorithms/analysis-genetic-algorithm-hybrid-naive.md)
  - [analysis-genetic-algorithm-hybrid-optimized.md](./analysis/algorithms/analysis-genetic-algorithm-hybrid-optimized.md)
  - [analysis-genetic-algorithm-full-gpu.md](./analysis/algorithms/analysis-genetic-algorithm-full-gpu.md)
  - [analysis-ga-module-interactions.md](./analysis/algorithms/analysis-ga-module-interactions.md)
- **Benchmarking & comparative synthesis**:
  - [analysis-benchmarking-system.md](./analysis/benchmarking/analysis-benchmarking-system.md)
  - [analysis-algorithm-comparison.md](./analysis/comparative/analysis-algorithm-comparison.md)
  - [analysis-future-scaling-and-type-decisions.md](./analysis/questions/analysis-future-scaling-and-type-decisions.md)
- **Presentations & executive summaries (`analysis/presentations/`)**:
  - [notes_thesis_presentation_slides.md](./analysis/presentations/notes_thesis_presentation_slides.md)
  - [report-thesis-summary.md](./analysis/presentations/report-thesis-summary.md)
  - [report-fact-check-presentation.md](./analysis/presentations/report-fact-check-presentation.md)
  - [benchmark_workflow.mmd](./analysis/presentations/benchmark_workflow.mmd)

## 4. Academic literature & critical appraisal (`critical_analysis/`)

- [analysis-literature-synthesis-prior-art.md](./critical_analysis/analysis-literature-synthesis-prior-art.md) — Consolidated literature synthesis positioning the solver against GPU TSP/VRP prior art.
- [analysis-routing-literature-corpus.md](./critical_analysis/analysis-routing-literature-corpus.md) — Paper-by-paper breakdown of foundational routing heuristics and GPU metaheuristics.
- [analysis-clrs-and-cook-foundations.md](./critical_analysis/analysis-clrs-and-cook-foundations.md) — Theoretical grounding from Cormen et al. (CLRS) and Cook (2012).
- [guide-academic-critical-appraisal.md](./critical_analysis/guide-academic-critical-appraisal.md) — Structured academic critical-appraisal rubric and checklist.

## 5. Engineering walkthroughs (`walkthroughs/`)

- [analysis-test-suite-walkthrough.md](./walkthroughs/analysis-test-suite-walkthrough.md) — Package-mirrored test suite architecture, xfail defect ledger, and verification commands.
- [analysis-phases-1-6-reorganization-walkthrough.md](./walkthroughs/analysis-phases-1-6-reorganization-walkthrough.md) — Phases 1–6 repository reorganization and modular strategy extraction.
- [analysis-phase-7-residency-rng-walkthrough.md](./walkthroughs/analysis-phase-7-residency-rng-walkthrough.md) — Phase 7 GPU array residency and deterministic RNG seeding.
- [analysis-phase-8-hermeticity-walkthrough.md](./walkthroughs/analysis-phase-8-hermeticity-walkthrough.md) — Phase 8 test hermeticity and fixture isolation.
- [analysis-phase-9-gpu-capability-walkthrough.md](./walkthroughs/analysis-phase-9-gpu-capability-walkthrough.md) — Phase 9 hardware capability probing and CPU fallback contracts.
- [analysis-walkthrough-10-15-pairwise-j-loop.md](./walkthroughs/analysis-walkthrough-10-15-pairwise-j-loop.md) — Tasks 10.15–10.16 device pairwise distances and 2-opt inner $j$-loop refactoring.
- [analysis-walkthrough-10-17-on-the-fly-tiling.md](./walkthroughs/analysis-walkthrough-10-17-on-the-fly-tiling.md) — Tasks 10.17–10.18 on-the-fly coordinate evaluation and shared-memory tiling.
- [analysis-walkthrough-10-19-vram-guard.md](./walkthroughs/analysis-walkthrough-10-19-vram-guard.md) — Task 10.19 pre-flight VRAM capacity guardrails (`VRAMInsufficientError`).
- [analysis-walkthrough-10-20-candidate-lists.md](./walkthroughs/analysis-walkthrough-10-20-candidate-lists.md) — Task 10.20 nearest-neighbor candidate lists and inverse position arrays.
- [analysis-walkthrough-10-21-docs-cleanup.md](./walkthroughs/analysis-walkthrough-10-21-docs-cleanup.md) — Task 10.21 legacy scripts and documentation cleanup record.

## 6. Archive (`archive/`)

Substantive historical or out-of-scope notes preserved for reference:

- [notes_simple_ant_colony_routing.md](./archive/notes_simple_ant_colony_routing.md) — Sanitized reading notes on Simple Ant Colony Optimization (SACO) in networks.
- [notes_aodv_routing_protocol.md](./archive/notes_aodv_routing_protocol.md) — Mobile ad-hoc network AODV routing protocol notes.
- [notes_guided_local_search_design.md](./archive/notes_guided_local_search_design.md) — Formal Guided Local Search (GLS) penalty formulation and typed prototype.
- [notes_early_reading_log_2024.md](./archive/notes_early_reading_log_2024.md) — Early 2024 exploratory literature reading log.
