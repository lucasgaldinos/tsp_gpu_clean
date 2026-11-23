---
title: "Thesis Chapter 4 Reproduction Guide"
description: >
  Isolated step-by-step guide for reproducing the Master's dissertation
  Chapter 4 iso-algorithmic GPU/CPU benchmark study and regenerating its
  Brazilian Portuguese tables, quantitative macros, Demšar CD diagrams, and
  scaling plots from saved JSON checkpoints.
created: 2026-11-23
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: guide
scope: thesis
modifications:
  - date_modified: 2026-11-23
    modifications:
      - description: >
          Created dedicated thesis Chapter 4 reproduction guide isolated from
          the generic framework getting-started tutorial, referencing
          benchmark_scripts/configs/chapter4/ and external report_style.json.
related_files:
  - "[README.md](../README.md)"
  - "[documentation/README.md](./README.md)"
  - "[guide-getting-started.md](./guide-getting-started.md)"
  - "[run_chapter4_benchmark.py](../benchmark_scripts/run_chapter4_benchmark.py)"
  - "[07-apendices.tex](../documents_tcc/tex/thesis/parts/07-apendices.tex)"
tags:
  - guide/reproduction
  - guide/thesis
  - benchmark/chapter4
  - benchmark/gpu
  - benchmark/reporting
  - analysis/statistics
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/tsp
  - architecture/decoupling
---

# Thesis Chapter 4 Reproduction Guide

This guide documents how to reproduce the iso-algorithmic four-variant benchmark study from Chapter 4 of the Master's dissertation (also summarized in [07-apendices.tex](../documents_tcc/tex/thesis/parts/07-apendices.tex)).

All dissertation-specific configuration files, iso-algorithmic contract assertions, and Brazilian Portuguese presentation settings live **strictly outside `src/`** under [`benchmark_scripts/`](../benchmark_scripts/):

- [`benchmark_scripts/run_chapter4_benchmark.py`](../benchmark_scripts/run_chapter4_benchmark.py) — Thesis experiment runner wrapping `tsp-benchmark` with preflight iso-algorithmic checks and joint statistical summary logging.
- [`benchmark_scripts/configs/chapter4/algorithms.json`](../benchmark_scripts/configs/chapter4/algorithms.json) — The four Chapter 4 variants (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`).
- [`benchmark_scripts/configs/chapter4/benchmark.json`](../benchmark_scripts/configs/chapter4/benchmark.json) — Execution protocol (`repetitions: 30`, adaptive patience $2\sqrt{n}$, adaptive generations $2n\sqrt{n}$, `incremental_runs: false`).
- [`benchmark_scripts/configs/chapter4/problems.json`](../benchmark_scripts/configs/chapter4/problems.json) — TSPLIB95 instance dimensions and reference optima.
- [`benchmark_scripts/configs/chapter4/report_style.json`](../benchmark_scripts/configs/chapter4/report_style.json) — Brazilian Portuguese display names, decimal commas (`{,}` / `,`), `\label{tab:chapter4_*}` prefix, and figure axis labels injected externally via `--report-style`.
- [`benchmark_scripts/configs/chapter4/algorithms.archive.json`](../benchmark_scripts/configs/chapter4/algorithms.archive.json) — Archived monolithic Fujimoto GPU baseline (`FullGPUMonolithic`) kept quarantined from active runs.

## 1. Environment & TSPLIB95 Dataset Setup

1. Install the locked environment (Python `>= 3.12`, `uv`, `git`, and NVIDIA CUDA 12.x driver for the GPU variants):

```bash
uv sync
```

2. Obtain the 15 TSPLIB95 `.tsp` instance files (`att48`, `eil51`, `berlin52`, `gr96`, `kroA100`, `ch150`, `d198`, `pr264`, `lin318`, `pcb442`, `att532`, `si535`, `gr666`, `rat783`, `pr1002`) and place them in `datasets/tsp/`.

> [!NOTE]
> TSPLIB `.tsp` files are loaded directly by [`run_chapter4_benchmark.py`](../benchmark_scripts/run_chapter4_benchmark.py) via `--tsp-dir datasets/tsp` — pre-ingesting `datasets/tsp.duckdb` is completely optional.

## 2. Preflight & Quick Verification (Dry Run)

Verify hardware detection, configuration schema parity, and iso-algorithmic strategy contracts without running any solver or touching `results_v4/`:

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir /tmp/chapter4_dryrun \
  --dry-run
```

## 3. Running the Chapter 4 Experiment

### Quick CPU smoke test on a single instance

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir /tmp/chapter4_smoke \
  --tsp-files tests/fixtures/tsp/berlin52.tsp \
  --algorithms CPU \
  --repetitions 2 \
  --max-generations 30
```

### Full 15-instance x 4-variant x 30-repetition campaign

> [!WARNING]
> Pass a dedicated `--output-dir` (e.g., `results_repro/`) if you want to preserve the frozen dissertation outputs in `results_v4/`.

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir results_repro \
  --tsp-dir datasets/tsp \
  --no-incremental \
  --repetitions 30
```

Every `(problem, variant)` pair writes an atomic checkpoint to `results_repro/checkpoints/<problem>_<variant>.json` after **each repetition**. If interrupted, re-running the exact command with `--no-incremental` completes only missing repetitions up to `30` and skips already-complete pairs.

## 4. Regenerating All Thesis Tables, Macros, CD Diagrams, and Figures Without Re-Running Solvers

Because `tsp-report` operates purely on JSON checkpoints and accepts the external thesis `--report-style`, you can regenerate every dissertation table, quantitative macro (`benchmark_macros.tex`), Demšar Critical Difference TikZ snippet, and vector scaling plot (`scaling_time_gap.{pdf,svg}`) in seconds without a GPU or DuckDB:

```bash
uv run tsp-report \
  --checkpoints results_v4/checkpoints \
  --output-dir /tmp/chapter4_reports \
  --problems-config benchmark_scripts/configs/chapter4/problems.json \
  --benchmark-name chapter4_validation \
  --baseline CPU \
  --report-style benchmark_scripts/configs/chapter4/report_style.json \
  --reports all
```

Generated artifacts under `/tmp/chapter4_reports/`:

- `tables/chapter4_validation.md` & `tables/chapter4_validation.tex` (`\label{tab:chapter4_*}`)
- `tables/benchmark_macros.tex` (Brazilian decimal comma `{,}` quantitative `\newcommand` macros)
- `tables/chapter4_validation_cd_diagram_{gap,time}.tikz` (pt-BR Demšar Critical Difference diagrams)
- `figures/scaling_time_gap.{pdf,svg}` (pt-BR Type 42 vector scaling plots)
- `problem_statistics/<problem>_stats.json` (per-instance Mann-Whitney U, Friedman, and descriptive stats)

## 5. Optional DuckDB Consolidation & Audit

```bash
uv run tsp-import-checkpoints \
  --input results_repro/checkpoints \
  --output-dir results_repro

uv run tsp-validate-db \
  --db1 results_repro/results.duckdb \
  --checkpoints results_repro/checkpoints
```
---
title: "Thesis Chapter 4 Reproduction Guide"
description: >
  Isolated step-by-step guide for reproducing the Master's dissertation
  Chapter 4 iso-algorithmic GPU/CPU benchmark study and regenerating its
  Brazilian Portuguese tables, quantitative macros, Demšar CD diagrams, and
  scaling plots from saved JSON checkpoints.
created: 2026-11-23
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: guide
scope: thesis
modifications:
  - date_modified: 2026-11-23
    modifications:
      - description: >
          Created dedicated thesis Chapter 4 reproduction guide isolated from
          the generic framework getting-started tutorial, referencing
          benchmark_scripts/configs/chapter4/ and external report_style.json.
related_files:
  - "[README.md](../README.md)"
  - "[documentation/README.md](./README.md)"
  - "[guide-getting-started.md](./guide-getting-started.md)"
  - "[run_chapter4_benchmark.py](../benchmark_scripts/run_chapter4_benchmark.py)"
  - "[07-apendices.tex](../documents_tcc/tex/thesis/parts/07-apendices.tex)"
tags:
  - guide/reproduction
  - guide/thesis
  - benchmark/chapter4
  - benchmark/gpu
  - benchmark/reporting
  - analysis/statistics
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/tsp
  - architecture/decoupling
---

# Thesis Chapter 4 Reproduction Guide

This guide documents how to reproduce the iso-algorithmic four-variant benchmark study from Chapter 4 of the Master's dissertation (also summarized in [07-apendices.tex](../documents_tcc/tex/thesis/parts/07-apendices.tex)).

All dissertation-specific configuration files, iso-algorithmic contract assertions, and Brazilian Portuguese presentation settings live **strictly outside `src/`** under [`benchmark_scripts/`](../benchmark_scripts/):

- [`benchmark_scripts/run_chapter4_benchmark.py`](../benchmark_scripts/run_chapter4_benchmark.py) — Thesis experiment runner wrapping `tsp-benchmark` with preflight iso-algorithmic checks and joint statistical summary logging.
- [`benchmark_scripts/configs/chapter4/algorithms.json`](../benchmark_scripts/configs/chapter4/algorithms.json) — The four Chapter 4 variants (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`).
- [`benchmark_scripts/configs/chapter4/benchmark.json`](../benchmark_scripts/configs/chapter4/benchmark.json) — Execution protocol (`repetitions: 30`, adaptive patience $2\sqrt{n}$, adaptive generations $2n\sqrt{n}$, `incremental_runs: false`).
- [`benchmark_scripts/configs/chapter4/problems.json`](../benchmark_scripts/configs/chapter4/problems.json) — TSPLIB95 instance dimensions and reference optima.
- [`benchmark_scripts/configs/chapter4/report_style.json`](../benchmark_scripts/configs/chapter4/report_style.json) — Brazilian Portuguese display names, decimal commas (`{,}` / `,`), `\label{tab:chapter4_*}` prefix, and figure axis labels injected externally via `--report-style`.
- [`benchmark_scripts/configs/chapter4/algorithms.archive.json`](../benchmark_scripts/configs/chapter4/algorithms.archive.json) — Archived monolithic Fujimoto GPU baseline (`FullGPUMonolithic`) kept quarantined from active runs.

## 1. Environment & TSPLIB95 Dataset Setup

1. Install the locked environment (Python `>= 3.12`, `uv`, `git`, and NVIDIA CUDA 12.x driver for the GPU variants):

```bash
uv sync
```

2. Obtain the 15 TSPLIB95 `.tsp` instance files (`att48`, `eil51`, `berlin52`, `gr96`, `kroA100`, `ch150`, `d198`, `pr264`, `lin318`, `pcb442`, `att532`, `si535`, `gr666`, `rat783`, `pr1002`) and place them in `datasets/tsp/`.

> [!NOTE]
> TSPLIB `.tsp` files are loaded directly by [`run_chapter4_benchmark.py`](../benchmark_scripts/run_chapter4_benchmark.py) via `--tsp-dir datasets/tsp` — pre-ingesting `datasets/tsp.duckdb` is completely optional.

## 2. Preflight & Quick Verification (Dry Run)

Verify hardware detection, configuration schema parity, and iso-algorithmic strategy contracts without running any solver or touching `results_v4/`:

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir /tmp/chapter4_dryrun \
  --dry-run
```

## 3. Running the Chapter 4 Experiment

### Quick CPU smoke test on a single instance

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir /tmp/chapter4_smoke \
  --tsp-files tests/fixtures/tsp/berlin52.tsp \
  --algorithms CPU \
  --repetitions 2 \
  --max-generations 30
```

### Full 15-instance x 4-variant x 30-repetition campaign

> [!WARNING]
> Pass a dedicated `--output-dir` (e.g., `results_repro/`) if you want to preserve the frozen dissertation outputs in `results_v4/`.

```bash
uv run python -m benchmark_scripts.run_chapter4_benchmark \
  --output-dir results_repro \
  --tsp-dir datasets/tsp \
  --no-incremental \
  --repetitions 30
```

Every `(problem, variant)` pair writes an atomic checkpoint to `results_repro/checkpoints/<problem>_<variant>.json` after **each repetition**. If interrupted, re-running the exact command with `--no-incremental` completes only missing repetitions up to `30` and skips already-complete pairs.

## 4. Regenerating All Thesis Tables, Macros, CD Diagrams, and Figures Without Re-Running Solvers

Because `tsp-report` operates purely on JSON checkpoints and accepts the external thesis `--report-style`, you can regenerate every dissertation table, quantitative macro (`benchmark_macros.tex`), Demšar Critical Difference TikZ snippet, and vector scaling plot (`scaling_time_gap.{pdf,svg}`) in seconds without a GPU or DuckDB:

```bash
uv run tsp-report \
  --checkpoints results_v4/checkpoints \
  --output-dir /tmp/chapter4_reports \
  --problems-config benchmark_scripts/configs/chapter4/problems.json \
  --benchmark-name chapter4_validation \
  --baseline CPU \
  --report-style benchmark_scripts/configs/chapter4/report_style.json \
  --reports all
```

Generated artifacts under `/tmp/chapter4_reports/`:

- `tables/chapter4_validation.md` & `tables/chapter4_validation.tex` (`\label{tab:chapter4_*}`)
- `tables/benchmark_macros.tex` (Brazilian decimal comma `{,}` quantitative `\newcommand` macros)
- `tables/chapter4_validation_cd_diagram_{gap,time}.tikz` (pt-BR Demšar Critical Difference diagrams)
- `figures/scaling_time_gap.{pdf,svg}` (pt-BR Type 42 vector scaling plots)
- `problem_statistics/<problem>_stats.json` (per-instance Mann-Whitney U, Friedman, and descriptive stats)

## 5. Optional DuckDB Consolidation & Audit

```bash
uv run tsp-import-checkpoints \
  --checkpoints results_repro/checkpoints \
  --db results_repro/results.duckdb

uv run tsp-validate-db \
  --db results_repro/results.duckdb \
  --checkpoints results_repro/checkpoints
```

