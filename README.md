# GPU-Accelerated Modular Evolutionary Solver for TSP

A composable, protocol-driven Genetic Algorithm (GA) and Memetic Algorithm framework for the Traveling Salesman Problem (TSP) in Python, supporting pure NumPy CPU execution, hybrid CPU/GPU pipelines, and device-resident CUDA execution via CuPy.

Every evolutionary phase is an interchangeable strategy behind PEP 544 `Protocol` contracts, allowing you to swap operators, compose custom generational pipelines, plug in your own strategies without editing `src/`, run reproducible multi-repetition benchmarks directly from TSPLIB `.tsp` files (no database required), and generate publication-ready statistical reports independently of benchmark collection.

## Key Capabilities

- **6 interchangeable strategy slots, 17 built-in strategies** (`src/algorithms/strategies/`): mix and match CPU and CUDA operators across initialization, selection, crossover, mutation, local search (CPU & GPU 2-opt with `first`, `best`, `multi_move`, `naive`, and `batch` modes), and cost evaluation.
- **Composable generational pipelines** (`src/algorithms/pipeline_steps.py`): configure built-in `local_search_stage` placements (`post_crossover`, `post_mutation`, `post_evaluation`, `post_evolution`, `none`) or assemble custom step pipelines (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep`).
- **Extensible without modifying `src/`**: plug custom strategies and custom `GenerationStep` classes via direct Python dependency injection, `register_strategy()` / `register_pipeline_step()`, or declarative JSON `"module"` + `"class"` references.
- **Direct TSPLIB95 `.tsp` ingestion**: load `.tsp` files directly (`load_tsplib_file`, `TsplibLoader`, or `--tsp-dir` / `--tsp-files`), with automatic sibling `.opt.tour` optimal-cost resolution and graceful handling when optima are unknown (`optimal_cost=None`). DuckDB storage is strictly optional.
- **Decoupled collection & composable reporting**: `tsp-benchmark` runs collection-only by default (atomic per-repetition JSON checkpoints + per-problem statistics); `tsp-report` (or opt-in `--reports`) independently renders Markdown tables, LaTeX tables, quantitative LaTeX macros, Demšar Critical Difference TikZ diagrams, and vector scaling plots (PDF/SVG), styled externally via `--report-style <file.json>` or `ReportStyle`.

## Requirements & Installation

- **Python** `>= 3.12`
- **`uv`** and **`git`** (used to fetch the public TSPLIB parser package `routing-data` pinned at tag `v0.2.0`)
- **Optional GPU:** NVIDIA GPU (Compute Capability `>= 5.0`) + CUDA 12.x driver for `use_gpu: true` strategies (all CPU strategies and the fast CPU test suite run without a GPU)

```bash
git clone https://github.com/lucasgaldinos/tsp_gpu_clean.git
cd tsp_gpu_clean
uv sync
```

Verify the environment (no GPU required):

```bash
uv run pytest -q -m "not gpu and not slow"
```

> [!NOTE]
> Use `uv sync` rather than `pip install -e .`: `routing-data` is sourced from Git via `[tool.uv.sources]` in [pyproject.toml](./pyproject.toml), which `uv` resolves automatically while plain `pip` ignores.

## Quickstart

### 1. Run a CPU benchmark directly on `.tsp` files (zero database, zero boilerplate)

Pass any algorithm config in [configs/](./configs/) and a directory of `.tsp` files (the tracked test fixtures under `tests/fixtures/tsp/` work out of the box):

```bash
uv run tsp-benchmark \
  --algorithms-config configs/cpu-operators.example.json \
  --tsp-dir tests/fixtures/tsp \
  --repetitions 3 \
  --max-generations 30 \
  --output-dir /tmp/tsp_demo
```

By default, `tsp-benchmark` performs **collection only**, writing atomic JSON checkpoints to `/tmp/tsp_demo/checkpoints/` and per-problem statistical summaries to `/tmp/tsp_demo/problem_statistics/`.

### 2. Generate reports independently from checkpoints (`tsp-report`)

Generate all report artifacts — or any subset (`markdown`, `latex`, `macros`, `cd`, `plots`) — directly from the saved JSON checkpoints, optionally applying a presentation style file:

```bash
uv run tsp-report \
  --checkpoints /tmp/tsp_demo/checkpoints \
  --benchmark-name cpu_operators_demo \
  --report-style configs/report-style.example.json \
  --reports markdown latex macros cd plots
```

Outputs are written under `/tmp/tsp_demo/tables/` (`.md`, `.tex`, `benchmark_macros.tex`, `*_cd_diagram_{gap,time}.tikz`) and `/tmp/tsp_demo/figures/` (`scaling_time_gap.{pdf,svg}`). You can also emit reports inline during collection by passing `--reports all` (or `--reports markdown plots`) directly to `tsp-benchmark`.

### 3. Compose a Genetic Algorithm in Python

```python
import numpy as np

from algorithms import GeneticAlgorithm
from algorithms.pipeline_steps import EliteOnlyLocalSearchStep, PostInitLocalSearchStep
from algorithms.strategies import (
  CrossoverCPUPermPartiallyMapped,
  EvaluationCPUCost,
  InitializationNearestNeighbor,
  LocalSearchCPUTourTwoOpt,
  MutationCPUPermInversion,
  SelectionCPUTournament,
)
from loaders import load_tsplib_file
from protocols.problem_context import ProblemContext

problem = load_tsplib_file("tests/fixtures/tsp/berlin52.tsp")
context = ProblemContext(problem, xp=np, seed=42)

local_search = LocalSearchCPUTourTwoOpt(mode="best", max_iterations=10, candidate_k=20)
evaluation = EvaluationCPUCost()

ga = GeneticAlgorithm(
  population_size=128,
  mutation_rate=0.05,
  seed=42,
  initialization=InitializationNearestNeighbor(k=5, seed=42),
  selection=SelectionCPUTournament(k=5, seed=42),
  crossover=CrossoverCPUPermPartiallyMapped(seed=42),
  mutation=MutationCPUPermInversion(seed=42),
  local_search=local_search,
  evaluation=evaluation,
  local_search_stage="none",
  pipeline=[
    PostInitLocalSearchStep(local_search=local_search, evaluation=evaluation),
    EliteOnlyLocalSearchStep(local_search=local_search, evaluation=evaluation, elite_count=16),
  ],
)

best_tour, stats = ga.evolve(
  context=context,
  customers=list(range(problem.dimension)),
  max_generations=50,
)
print(f"Best cost: {stats['best_fitness']:.2f} in {stats['generations_completed']} generations")
```

## Built-In Strategies & Pipeline Steps

Every strategy lives in [src/algorithms/strategies/](./src/algorithms/strategies/) and is registered in [`STRATEGY_REGISTRY`](./src/algorithms/strategies/__init__.py); pipeline steps live in [`PIPELINE_STEP_REGISTRY`](./src/algorithms/pipeline_steps.py).

| Slot / Component | CPU Implementations | GPU (CUDA / CuPy) Implementations |
| --- | --- | --- |
| **Initialization** | `InitializationRandom`, `InitializationNearestNeighbor` | `InitializationRandom(return_gpu=True)`, `InitializationNearestNeighbor(return_gpu=True)` |
| **Selection** | `SelectionCPUTournament`, `SelectionCPURouletteWheel` | `SelectionGPUTournament` |
| **Crossover** | `CrossoverCPUPermOrder`, `CrossoverCPUPermPartiallyMapped` | `CrossoverGPUPermOrder` |
| **Mutation** | `MutationCPUPermSwap`, `MutationCPUPermInversion` | `MutationGPUPermSwap`, `MutationGPUPermInversion` |
| **Local Search** | `LocalSearchCPUTourTwoOpt` (`first`, `best`, `multi_move`), `LocalSearchNoOp` | `LocalSearchGPUTourTwoOpt` (`naive`, `batch`) |
| **Evaluation** | `EvaluationCPUCost` | `EvaluationGPUCost` |
| **Pipeline Steps** | `PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep` | Backend-agnostic (operates on CPU or GPU arrays via injected strategies) |

## Example Configurations (`configs/`)

Ready-to-run declarative configs validated against Draft 2020-12 JSON Schemas ([configs/schemas/](./configs/schemas/)):

| File | Purpose | GPU Required? |
| --- | --- | --- |
| [configs/cpu-operators.example.json](./configs/cpu-operators.example.json) | Compares pure-GA operator combinations (`OX` vs `PMX`, `Swap` vs `Inversion`, `Tournament` vs `RouletteWheel`) against nearest-neighbor memetic 2-opt | No |
| [configs/local-search-pipelines.example.json](./configs/local-search-pipelines.example.json) | Compares `post_mutation`, `post_evolution`, `first`/`best`/`multi_move` 2-opt, and custom pipelines (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep`) | No |
| [configs/gpu-backends.example.json](./configs/gpu-backends.example.json) | Compares CPU baseline, hybrid batched GPU 2-opt, and full-GPU swap/inversion pipelines | Yes (or `--skip-cpu` / `--algorithms CPU_Baseline`) |
| [configs/algorithms.full.example.json](./configs/algorithms.full.example.json) | Full catalogue exercising all five 2-opt modes and every pipeline step | Mixed |
| [configs/report-style.example.json](./configs/report-style.example.json) | Example `--report-style` JSON customizing display names, LaTeX label prefix, size strata, and axis titles | No |

## Extending Without Modifying `src/`

You can add custom operators (e.g., 3-opt, Or-opt, custom mutation) or custom generational steps in your own project files without touching `src/`:

1. **Direct Python injection:** implement the target PEP 544 `Protocol` from [src/protocols/algorithm_strategies.py](./src/protocols/algorithm_strategies.py) (optionally subclassing `StrategyBase`) or `GenerationStep`, and pass your instance to `GeneticAlgorithm(...)`.
2. **Runtime registration:** call `register_strategy("MyLocalSearch", MyLocalSearch)` or `register_pipeline_step("MyStep", MyStep)` from `algorithms` in your benchmark script before calling `run_comprehensive_benchmark(...)`.
3. **Dynamic JSON loading from CLI:** reference any importable Python module directly in your `algorithms.json` via `{"module": "my_ops", "class": "MyLocalSearch", "params": {...}}` and run `tsp-benchmark --algorithms-config my_experiment.json`.

Complete runnable walkthroughs for all three extension patterns — plus a complete side-by-side **JSON-first experiment cookbook** (`algorithms.json`, `benchmark.json`, `problems.json`, `report_style.json`) and Level 1 / Level 2 external script templates — live in [documentation/guide-getting-started.md](./documentation/guide-getting-started.md#7-json-first-experiment-cookbook--external-script-templates).

> [!TIP]
> **Building your own solver variants or experiments?** Start with the [Getting Started & JSON-First Cookbook](./documentation/guide-getting-started.md), pair it with the [Local Search & Custom Pipeline Stages Guide](./documentation/guide-local-search-pipeline-stages.md), and consult the [Master Architecture Blueprint](./documentation/guide-project-architecture-blueprint.md) and subsystem guides under [Documentation](#documentation) below.

## CLI Entry Points

Five CLI commands are registered in [pyproject.toml](./pyproject.toml):

| Command | Role | DuckDB Required? |
| --- | --- | --- |
| `tsp-benchmark` | Runs declarative benchmarks from `--algorithms-config` (or `--config-dir`) and `--tsp-dir`/`--tsp-files` with preflight slot/protocol & checkpoint-drift checks, live corrupt-checkpoint quarantine, and nullable `optimal_cost` resume; collection-only by default, opt-in `--reports` and `--report-style` | Optional |
| `tsp-report` | Regenerates statistical JSONs (including retroactive optimality gaps via `--problems-config`), Markdown/LaTeX tables, `benchmark_macros.tex`, CD TikZ diagrams, and scaling figures directly from JSON checkpoints | No |
| `tsp-ingest` | Optional helper to validate and pre-ingest a `.tsp` directory into a reusable DuckDB file | Yes (optional workflow) |
| `tsp-import-checkpoints` | Optional idempotent import of `checkpoints/*.json` into `results.duckdb` for SQL queries | Yes (optional workflow) |
| `tsp-validate-db` | Optional full-table consistency audit comparing `results.duckdb` against source JSON checkpoints (and optionally a second DuckDB across every sorted row) | Yes (optional workflow) |

## Repository Structure

```text
.
├── src/
│   ├── algorithms/      # GeneticAlgorithm, 17 strategies, 3 pipeline steps, CUDA kernels
│   ├── benchmarking/    # runner/ (orchestration, configs, checkpoints), analysis/ (pure stats),
│   │                    # reporting/ (tables, macros, CD TikZ, plots, ReportStyle), database/ (optional)
│   ├── cli/             # tsp-benchmark, tsp-report, tsp-ingest, tsp-import-checkpoints, tsp-validate-db
│   ├── data_models/     # Problem & typed BenchmarkRunRecord dataclasses
│   ├── distances/       # Three-tier TSPLIB95 distance engines (CPU & CUDA)
│   ├── loaders/         # Direct .tsp loader (routing-data) & optional DuckDB loader
│   ├── protocols/       # PEP 544 strategy protocols, ProblemContext, backend guards
│   └── utils/           # PhaseTimer, deterministic RNG, route helpers
├── configs/             # Generic ready-to-run example configs & Draft 2020-12 JSON Schemas
├── benchmark_scripts/   # Repo doc/thesis checks & experiments-and-examples/ (Chapter 4 + custom studies)
├── tests/               # Unit, integration, contract, and validation suites + hermetic TSP fixtures
└── documentation/       # Getting-started tutorial, thesis reproduction guide, architecture blueprint, KB
```

## Documentation

### Practical Build, Extension & Reproduction Guides

- [documentation/guide-getting-started.md](./documentation/guide-getting-started.md) — End-to-end framework guide: direct `.tsp` loading without DuckDB, Python GA composition, custom external strategies & pipeline steps without editing `src/`, **4-file JSON-first experiment cookbook**, Level 1 / Level 2 script templates, and composable `tsp-report` styling (`ReportStyle`).
- [documentation/guide-local-search-pipeline-stages.md](./documentation/guide-local-search-pipeline-stages.md) — Deep dive into `GenerationState`, `GenerationStep`, `LocalSearchStage` semantics (`pre_selection`, `pre_evaluation`, `post_survival`), and custom multi-stage CPU/GPU pipelines.
- [documentation/guide-thesis-reproduction.md](./documentation/guide-thesis-reproduction.md) — Isolated reproduction guide for the Master's dissertation Chapter 4 iso-algorithmic study and artifact regeneration (`benchmark_scripts/experiments-and-examples/chapter4/`).
- [src/benchmarking/database/QUICKSTART.md](./src/benchmarking/database/QUICKSTART.md) — Optional DuckDB persistence guide (`tsp-import-checkpoints`, `tsp-validate-db`, schema layout, and SQL queries).

### Architecture Blueprints & Subsystem Guides

- [documentation/guide-project-architecture-blueprint.md](./documentation/guide-project-architecture-blueprint.md) — Master implementation-level architectural blueprint across all eight `src/` packages, CUDA memory models, layer rules, and ADRs.
- [documentation/analysis/guide-architecture-protocols-and-data-models.md](./documentation/analysis/guide-architecture-protocols-and-data-models.md) — Subsystem guide for `src/protocols/` and `src/data_models/` (`@runtime_checkable` operator protocols, `ProblemContext` lazy caches, VRAM guardrails, `Problem` invariants).
- [documentation/analysis/guide-architecture-loaders-and-distances.md](./documentation/analysis/guide-architecture-loaders-and-distances.md) — Subsystem guide for `src/loaders/` and `src/distances/` (TSPLIB95 / `routing-data` ingestion and 3-tier CPU/GPU distance engines).
- [documentation/analysis/guide-architecture-algorithms-and-strategies.md](./documentation/analysis/guide-architecture-algorithms-and-strategies.md) — Subsystem guide for `src/algorithms/` (`GeneticAlgorithm`, `STAGE_SPECS`, 17 `Design B` strategies, and 8 CUDA kernels).
- [documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md](./documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md) — Subsystem guide for `src/benchmarking/`, `src/cli/`, `src/utils/`, and `configs/schemas/`.
- [documentation/analysis/benchmarking/analysis-benchmarking-system.md](./documentation/analysis/benchmarking/analysis-benchmarking-system.md) — Benchmarking pipeline diagrams, pure statistical engine vs reporting topology, and checkpoint JSON schema reference.

### Methodology, GPU Optimization & Statistics Guides

- [documentation/knowledge-base/guide-statistical-testing-index.md](./documentation/knowledge-base/guide-statistical-testing-index.md) — Master statistical protocol guide (unconditional Wilcoxon + Hodges–Lehmann CI + Kerby rank-biserial, Holm–Bonferroni, Friedman + Iman–Davenport $F_F$, and Nemenyi CD diagrams).
- [documentation/knowledge-base/guide-iso-algorithmic-ga-validation.md](./documentation/knowledge-base/guide-iso-algorithmic-ga-validation.md) & [guide-gpu-cpu-comparison-methodology.md](./documentation/knowledge-base/guide-gpu-cpu-comparison-methodology.md) — Controlled CPU vs GPU experimental methodology and parameter parity checks.
- [documentation/knowledge-base/guide-ga-hyperparameter-tuning.md](./documentation/knowledge-base/guide-ga-hyperparameter-tuning.md) & [guide-construction-heuristics.md](./documentation/knowledge-base/guide-construction-heuristics.md) — Memetic GA parameter sizing and Nearest-Neighbor initialization trade-offs.
- [documentation/knowledge-base/guide-gpu-vram-guardrails-buffer-pooling.md](./documentation/knowledge-base/guide-gpu-vram-guardrails-buffer-pooling.md) & [guide-cupy-raw-kernels-optimization.md](./documentation/knowledge-base/guide-cupy-raw-kernels-optimization.md) — CUDA/CuPy memory guardrails, buffer pooling, shared-memory tiling, and RawKernel engineering.
- [documentation/knowledge-base/guide-clean-python-practices.md](./documentation/knowledge-base/guide-clean-python-practices.md) & [documentation/README.md](./documentation/README.md) — Repository Python conventions and the full 59+ guide catalog (knowledge base, critical analyses, and engineering walkthroughs).

## Thesis Chapter 4 Reproduction

This repository also accompanies a Master's dissertation comparing CPU, hybrid, and full-GPU memetic GA variants across 15 TSPLIB95 instances. All dissertation-specific scripts, configs, iso-algorithmic validation checks, and Brazilian Portuguese report styles are isolated **outside `src/`** under `benchmark_scripts/experiments-and-examples/chapter4/` (with [`run_chapter4_benchmark.py`](./benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py) and `configs/`).

See [documentation/guide-thesis-reproduction.md](./documentation/guide-thesis-reproduction.md) for the complete Chapter 4 reproduction workflow and checkpoint-to-figure regeneration commands.
