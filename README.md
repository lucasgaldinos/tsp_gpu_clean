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
  CrossoverPartiallyMapped,
  EvaluationCPUCost,
  InitializationNearestNeighbor,
  LocalSearchCPUTwoOpt,
  MutationInversion,
  SelectionTournament,
)
from loaders import load_tsplib_file
from protocols.problem_context import ProblemContext

problem = load_tsplib_file("tests/fixtures/tsp/berlin52.tsp")
context = ProblemContext(problem, xp=np, seed=42)

local_search = LocalSearchCPUTwoOpt(mode="best", max_iterations=10, candidate_k=20)
evaluation = EvaluationCPUCost()

ga = GeneticAlgorithm(
  population_size=128,
  mutation_rate=0.05,
  seed=42,
  initialization=InitializationNearestNeighbor(k=5, seed=42),
  selection=SelectionTournament(k=5, seed=42),
  crossover=CrossoverPartiallyMapped(seed=42),
  mutation=MutationInversion(seed=42),
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
| **Selection** | `SelectionTournament`, `SelectionRouletteWheel` | `SelectionGPUTournament` |
| **Crossover** | `CrossoverOrder`, `CrossoverPartiallyMapped` | `CrossoverGPUOrder` |
| **Mutation** | `MutationSwap`, `MutationInversion` | `MutationGPUSwap`, `MutationGPUInversion` |
| **Local Search** | `LocalSearchCPUTwoOpt` (`first`, `best`, `multi_move`), `LocalSearchNoOp` | `LocalSearchGPUTwoOpt` (`naive`, `batch`) |
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

Complete runnable walkthroughs for all three extension patterns live in [documentation/guide-getting-started.md](./documentation/guide-getting-started.md).

## CLI Entry Points

Five CLI commands are registered in [pyproject.toml](./pyproject.toml):

| Command | Role | DuckDB Required? |
| --- | --- | --- |
| `tsp-benchmark` | Runs declarative benchmarks from `--algorithms-config` (or `--config-dir`) and `--tsp-dir`/`--tsp-files`; collection-only by default, opt-in `--reports` and `--report-style` | Optional |
| `tsp-report` | Regenerates statistical JSONs, Markdown/LaTeX tables, `benchmark_macros.tex`, CD TikZ diagrams, and scaling figures directly from JSON checkpoints | No |
| `tsp-ingest` | Optional helper to validate and pre-ingest a `.tsp` directory into a reusable DuckDB file | Yes (optional workflow) |
| `tsp-import-checkpoints` | Optional idempotent import of `checkpoints/*.json` into `results.duckdb` for SQL queries | Yes (optional workflow) |
| `tsp-validate-db` | Optional consistency audit comparing `results.duckdb` against source JSON checkpoints (and optionally a second DuckDB) | Yes (optional workflow) |

## Repository Structure

```text
.
├── src/
│   ├── algorithms/      # GeneticAlgorithm, 17 strategies, 3 pipeline steps, CUDA kernels
│   ├── benchmarking/    # Runner, config loader, checkpoint I/O, stats, composable reporting, optional DuckDB
│   ├── cli/             # tsp-benchmark, tsp-report, tsp-ingest, tsp-import-checkpoints, tsp-validate-db
│   ├── data_models/     # Problem & typed BenchmarkRunRecord dataclasses
│   ├── distances/       # Three-tier TSPLIB95 distance engines (CPU & CUDA)
│   ├── loaders/         # Direct .tsp loader (routing-data) & optional DuckDB loader
│   ├── protocols/       # PEP 544 strategy protocols, ProblemContext, backend guards
│   └── utils/           # PhaseTimer, deterministic RNG, route helpers
├── configs/             # Generic ready-to-run example configs & Draft 2020-12 JSON Schemas
├── benchmark_scripts/   # External experiment scripts & thesis Chapter 4 configs (decoupled from src/)
├── tests/               # Unit, integration, contract, and validation suites + hermetic TSP fixtures
└── documentation/       # Getting-started tutorial, thesis reproduction guide, architecture blueprint, KB
```

## Documentation

- [documentation/guide-getting-started.md](./documentation/guide-getting-started.md) — Full framework guide: direct `.tsp` loading, custom strategies & pipelines without editing `src/`, custom CLI benchmarks, and composable reporting with `ReportStyle`.
- [documentation/guide-project-architecture-blueprint.md](./documentation/guide-project-architecture-blueprint.md) — Implementation-level architecture blueprint across all eight `src/` packages.
- [documentation/guide-local-search-pipeline-stages.md](./documentation/guide-local-search-pipeline-stages.md) — Deep dive into `LocalSearchStage` semantics and custom pipeline step placement.
- [documentation/README.md](./documentation/README.md) — Complete documentation catalog (knowledge base, subsystem guides, statistical references).

## Thesis Chapter 4 Reproduction

This repository also accompanies a Master's dissertation comparing CPU, hybrid, and full-GPU memetic GA variants across 15 TSPLIB95 instances. All dissertation-specific configs, iso-algorithmic validation checks, and Brazilian Portuguese report styles are isolated **outside `src/`** under `benchmark_scripts/configs/chapter4/` and `benchmark_scripts/run_chapter4_benchmark.py`.

See [documentation/guide-thesis-reproduction.md](./documentation/guide-thesis-reproduction.md) for the complete Chapter 4 reproduction workflow and checkpoint-to-figure regeneration commands.
