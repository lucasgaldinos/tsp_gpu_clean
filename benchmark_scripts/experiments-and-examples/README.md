---
title: Experiments and Examples Guide
description: >
  Practical reference for authoring, validating, executing, and interpreting
  self-contained JSON-first benchmark experiments outside src/, including custom
  external Python/CUDA operators, local schema generation, offline report
  regeneration, and interpreting asynchronous GPU phase timings.
created: 2026-11-24
type: guide
scope: local
modifications:
  - date_modified: 2026-11-24
    modifications:
      - description: >
          Created the experiments-and-examples README covering folder layout,
          JSON-first configs, external strategies/kernels, runner scripts,
          tsp-report regeneration, and Table 4 GPU Host Sync interpretation.
related_files:
  - "[guide-getting-started.md](../../documentation/guide-getting-started.md)"
  - "[algorithms.schema.json](../../configs/schemas/algorithms.schema.json)"
  - "[benchmark.schema.json](../../configs/schemas/benchmark.schema.json)"
  - "[problems.schema.json](../../configs/schemas/problems.schema.json)"
  - "[report_style.schema.json](../../configs/schemas/report_style.schema.json)"
  - "[analysis-ox-vs-pmx-no-2opt-and-gpu-dispatch.md](./experiment-comparing-OX-and-PMX/docs/analysis-ox-vs-pmx-no-2opt-and-gpu-dispatch.md)"
tags:
  - guide/experiments
  - guide/benchmarking
  - guide/configuration
  - guide/gpu
  - benchmark/workflow
  - benchmark/reproducibility
  - benchmark/profiling
  - algorithm/genetic-algorithm
  - algorithm/cuda-kernels
  - notes/architecture
---

# Experiments and Examples

This directory hosts self-contained, reproducible experiments isolated outside `src/`.
Each subfolder bundles its own JSON configurations, optional custom Python/CUDA operators,
runner script, documentation notes, and generated `results/` outputs without modifying core
library code or repository-wide schemas.

## Available Experiments

| Folder | Purpose |
| :--- | :--- |
| `chapter4/` | Dissertation Chapter 4 four-variant iso-algorithmic validation suite (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`) and local-search stage sweeps. |
| `experiment-comparing-OX-and-PMX/` | End-to-end example comparing built-in GPU Order Crossover (`CrossoverGPUPermOrder`) against an experiment-local CUDA Partially Mapped Crossover (`CrossoverGPUPermPartiallyMapped` + `kernels/pmx_crossover.cu`). |

---

## Standard Experiment Folder Layout

```text
my-experiment/
├── configs/
│   ├── algorithms.json          # Required: default_params + algorithm_configs
│   ├── algorithms.schema.json   # Optional: local schema with custom ops IntelliSense
│   ├── benchmark.json           # Optional: repetitions, patience, max_generations, incremental_runs
│   ├── problems.json            # Optional: instance names, exact sizes, optional optima
│   └── report_style.json        # Optional: display_names, algorithm_order, labels (passed via --report-style)
├── kernels/                     # Optional: experiment-local .cu CUDA kernels (auto-found by load_kernel)
│   └── my_op.cu
├── my_op.py                     # Optional: external StrategyBase / GenerationStep module
├── run_experiment.py            # Optional: thin CWD-safe wrapper delegating to cli.benchmark.main
├── docs/                        # Optional: experiment notes and empirical findings
└── results/                     # Generated checkpoints, problem_statistics, tables, figures, logs
```

Configuration precedence in [`config_loader.py::resolve_benchmark_configs`](../../src/benchmarking/runner/config_loader.py) is:

> **CLI flags** > **explicit `--*-config` paths** > **`--config-dir/{algorithms,benchmark,problems}.json`** > **neutral defaults**

---

## Adding Custom External Strategies and CUDA Kernels

1. **Implement the strategy outside `src/`:** subclass `StrategyBase` and the matching PEP 544 protocol (e.g., `CrossoverStrategy`, `MutationStrategy`) from [`algorithm_strategies.py`](../../src/protocols/algorithm_strategies.py).
2. **Place CUDA `.cu` files under `<experiment>/kernels/`:** calling `utils.gpu_helpers.load_kernel("my_kernel.cu", "kernel_fn", __file__)` resolves `<caller_dir>/kernels/my_kernel.cu` automatically and invalidates cached binaries on SHA-256 source changes.
3. **Reference it directly in `configs/algorithms.json`:** use a config-relative `.py` path in `"module"` (works even inside hyphenated folders with no `PYTHONPATH` or packaging):

```json
"crossover": {
  "module": "../gpu_pmx_crossover.py",
  "class": "CrossoverGPUPermPartiallyMapped"
}
```

4. **Generate an experiment-local schema for VS Code IntelliSense** (without touching `configs/schemas/algorithms.schema.json`):

```bash
uv run python -m benchmarking.runner.schema_generator \
  --extra-file benchmark_scripts/experiments-and-examples/my-experiment/my_op.py \
  --output benchmark_scripts/experiments-and-examples/my-experiment/configs/algorithms.schema.json
```

Point `"$schema": "./algorithms.schema.json"` inside your experiment's `algorithms.json` (and any variant config such as `algorithms.tuned.json`).

---

## Running and Regenerating Reports

### 1. CWD-Safe Runner Pattern

Anchor all paths to `EXP_DIR = Path(__file__).resolve().parent` and delegate argument parsing and orchestration to [`benchmark.py::main`](../../src/cli/benchmark.py). Because `uv run` walks upward to find `pyproject.toml` without changing your shell's working directory, anchoring paths to `__file__` guarantees `results/` is always written inside the experiment folder regardless of `pwd`.

```bash
# Preflight validation (configs, schemas, protocols, GPU availability, checkpoint provenance)
uv run python benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX/compare-pmx-ox-runner.py --dry-run

# Run with an alternate config into a separate output folder (prevents mixing incompatible checkpoints)
uv run python benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX/compare-pmx-ox-runner.py \
  --algorithms-config benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX/configs/algorithms.tuned.json \
  --output-dir benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX/results_tuned \
  --benchmark-name ox_vs_pmx_tuned \
  -r 30 --no-incremental
```

> [!IMPORTANT]
> Never reuse the same `--output-dir` after changing operators or `local_search_stage`: preflight provenance validation will intentionally abort rather than merge incompatible repetitions into the same checkpoint JSON. Use a new `--output-dir` (or clear `checkpoints/` first).

### 2. Offline Report Regeneration (`tsp-report`)

To regenerate Markdown/LaTeX tables, `benchmark_macros.tex`, Demšar Critical Difference `.tikz` diagrams, and vector scaling plots (`pdf`/`svg`) without re-running the GPU:

```bash
EXP="benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX"
uv run tsp-report \
  -i "$EXP/results/checkpoints" \
  -o "$EXP/results" \
  --config-dir "$EXP/configs" \
  --report-style "$EXP/configs/report_style.json" \
  --benchmark-name ox_vs_pmx \
  --baseline OX_GPU \
  --reports all
```

---

## Interpreting Table 4 on Full-GPU Pipelines (`Host Sync` vs. Dispatch Overhead)

By default, [`phase_timer.py::PhaseTimer`](../../src/utils/phase_timer.py) uses Python's host clock (`time.perf_counter()`) rather than inserting blocking `cp.cuda.Event.synchronize()` calls after every operator (`BUG-26`), so GPU kernels execute asynchronously back-to-back in a single CUDA stream.

This creates two distinct, complementary regimes in Table 4:

```text
Regime A — GPU-bound (e.g., WITH batched GPU 2-opt on pr1002):
  CPU: [Sel..LS..Surv enqueues: ~0.6 ms][====== waiting at Host Sync: ~10 ms (88-95%) ======]
  GPU:     [Sel][CX][Mut][============= executing 2-opt =============][Eval][Surv] -> 1 float

Regime B — Host-dispatch-bound (e.g., WITHOUT 2-opt on pure GPU GA):
  CPU: [Sel: 0.18][CX: 0.22][Mut: 0.26][Ev: 0.08][Surv: 0.29][Host Sync: 0.13 ms (~10%)]
  GPU:    [s]        [c]        [m]       [e]        [sv]    -> already nearly finished!
```

- **High `Host Sync (%)` (80–95%) with 2-opt is a healthy GPU-bound signature:** the GPU is executing heavy compute asynchronously while the CPU finishes enqueuing in <10% of wall time and waits at the single end-of-generation scalar convergence check (`float(cp.min(fitness))`).
- **Low `Host Sync (%)` (~10%) with flat ~1.1–1.2 ms/gen across problem sizes indicates host-dispatch-bound execution:** without 2-opt, each modular generation still issues ~10–12 CuPy/CUDA kernel launches plus ~6 small host-RNG H2D copies (`cut1`, `cut2`, `probs`, `pos1`, `pos2`, `candidates`) and 1 end-of-generation sync. Increasing `population_size` (e.g., `128 -> 512`) amortizes that fixed ~1.1 ms/gen host cost over 4× more routes per generation.
- For hardware-accurate per-phase GPU timings (including in-phase H2D/D2H transfers, at the cost of serializing the pipeline), construct `PhaseTimer(use_cuda_events=True)` or profile directly with `nsys profile` / `ncu`.
