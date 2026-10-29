---
title: gpu_accelerated_clean
description: >
  Repository entry point: what the GPU-accelerated genetic-algorithm routing
  solver does, how the `src/` packages are organised, how to set the
  environment up with uv, and the exact commands for tests, lint, format and
  type checking.
created: 2026-10-27
modifications:
  - date_modified: 2026-10-27
    modifications:
      - description: >
          Created the root README so `pyproject.toml`'s `readme` field resolves
          and the build backend has a file to package.
related_files:
  - [pyproject.toml](./pyproject.toml)
  - [AGENTS.md](./AGENTS.md)
tags:
  - guide/readme
  - guide/setup
  - guide/testing
  - guide/typing
  - guide/linting
  - notes/environment
  - notes/project
  - notes/commands
  - notes/tooling
  - benchmark/gpu
  - algorithm/genetic
  - analysis/architecture
  - review/quality
---

# gpu_accelerated_clean

GPU-accelerated genetic algorithm for the travelling salesman problem, built as a
set of interchangeable strategy objects so CPU, hybrid and full-GPU variants share
one evolutionary loop.

## Layout

`src/` is the import root and is **not** itself a package: modules import
`from protocols.algorithm_strategies import Routes`, never `from src.protocols...`.

| Package | Responsibility |
| --- | --- |
| `src/algorithms/` | Evolutionary loop (`metaheuristics/`), monolithic GPU kernel path, pluggable `strategies/` |
| `src/protocols/` | `Route` / `Routes` contracts, strategy protocols, `ProblemContext`, backend module |
| `src/data_models/` | `Problem` and domain exceptions |
| `src/distances/` | Pairwise distance kernels and full-matrix construction |
| `src/loaders/` | TSPLIB parsing and DuckDB access |
| `src/benchmarking_v2/` | Orchestration, statistics, tables and checkpoints |
| `src/benchmarks_v2/` | CLI entry points and result-database scripts |
| `src/utils/` | CUDA kernel loading, phase timing |

`src/algorithms/archive/` holds pre-modularisation kernels and is excluded from
lint and type checking.

## Setup

```bash
uv sync          # runtime + dev groups; installs this project in editable mode
```

python 3.12+. CUDA execution needs `cupy-cuda12x` and a working driver; the CPU
and hybrid paths run without one.

## Commands

```bash
# Tests
.venv/bin/python -m pytest -q

# Lint and format
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests

# Types (strict; baseline recorded 2026-10-27: 143 errors / 26 files)
.venv/bin/python -m mypy src

# Benchmarks
.venv/bin/python -m benchmarks_v2.run_chapter4_benchmark --help
.venv/bin/python -m benchmarks_v2.run_selected_problems --repetitions 3
```

All tool configuration lives in `pyproject.toml`; do not add a second config file.

## Working rules

Contributor and agent conventions live in [AGENTS.md](./AGENTS.md): plan before
multi-file writes, bounded reads, executable explanations, and verification that
can fail.
