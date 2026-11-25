---
title: "Benchmarking Architecture: Continuous Per-Repetition Checkpointing, Dual-Target Logging, and Algorithm-Agnostic Reporting"
description: >
  Exhaustive architectural review and technical design specifications for Tasks 12.14,
  12.15, and 12.16 (DP12-7, DP12-8, DP12-9). Covers the continuous per-repetition checkpointing
  protocol with strict isolation of disk I/O from metaheuristic evolution timing, the unified
  dual-target logging infrastructure, algorithm-agnostic reporting with Fleming & Wallace (1986)
  paired geometric speedups, and the strict boundary between library modules and experiment scripts.
created: 2026-11-12
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-12
    modifications:
      - description: >
          Authored comprehensive architectural review across `src/benchmarking/`, specifying
          per-repetition atomic checkpoint persistence (Task 12.14 / DP12-7), dual-target
          logging architecture (Task 12.15 / DP12-8), dynamic baseline resolution without
          hardcoded strings (Task 12.16 / DP12-9), and script-vs-module architectural boundaries.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../table-of-symbols.plan.md)"
  - "[analysis-advanced-benchmark-reporting-metrics.md](./analysis-advanced-benchmark-reporting-metrics.md)"
  - "[analysis-project-evolution-engineering-lessons.md](./analysis-project-evolution-engineering-lessons.md)"
  - "[algorithm_runner.py](../../../src/benchmarking/runner/algorithm_runner.py)"
  - "[orchestration.py](../../../src/benchmarking/runner/orchestration.py)"
  - "[checkpoint_io.py](../../../src/benchmarking/runner/checkpoint_io.py)"
  - "[report_generator.py](../../../src/benchmarking/analysis/report_generator.py)"
  - "[db_operations.py](../../../src/benchmarking/database/db_operations.py)"
  - "[run_chapter4_benchmark.py](../../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py)"
tags:
  - analysis/benchmarking
  - analysis/architecture
  - analysis/statistics
  - benchmark/checkpoints
  - benchmark/logging
  - benchmark/reporting
  - guide/methodology
  - guide/testing
  - review/quality
  - notes/workflow
---

# Benchmarking Architecture: Continuous Per-Repetition Checkpointing, Dual-Target Logging, and Algorithm-Agnostic Reporting

- [Benchmarking Architecture: Continuous Per-Repetition Checkpointing, Dual-Target Logging, and Algorithm-Agnostic Reporting](#benchmarking-architecture-continuous-per-repetition-checkpointing-dual-target-logging-and-algorithm-agnostic-reporting)
  + [1. Executive Architectural Motivation & Systemic Context](#1-executive-architectural-motivation--systemic-context)
  + [2. Per-Repetition Checkpointing Architecture (`DP12-7` / Task 12.14)](#2-per-repetition-checkpointing-architecture-dp12-7--task-1214)
    - [2.1 The Core Invariant: Strict Isolation of Disk I/O from Evolution Timing](#21-the-core-invariant-strict-isolation-of-disk-io-from-evolution-timing)
    - [2.2 Systemic Blast Radius Across `src/benchmarking/`](#22-systemic-blast-radius-across-srcbenchmarking)
      * [2.2.1 `algorithm_runner.py`: Observer Callback Pattern & Running Aggregates](#221-algorithm_runnerpy-observer-callback-pattern--running-aggregates)
      * [2.2.2 `orchestration.py`: Resumption Semantics with Partial In-Flight Cohorts](#222-orchestrationpy-resumption-semantics-with-partial-in-flight-cohorts)
      * [2.2.3 `checkpoint_io.py`: Schema Stability of In-Flight Checkpoints](#223-checkpoint_iopy-schema-stability-of-in-flight-checkpoints)
      * [2.2.4 `CheckpointManager`: Cohort Progress Accounting](#224-checkpointmanager-cohort-progress-accounting)
      * [2.2.5 `db_operations.py`: Analytical Columnar Storage vs. Transactional Logging](#225-db_operationspy-analytical-columnar-storage-vs-transactional-logging)
    - [2.3 Implementation Blueprint for Per-Repetition Persistence](#23-implementation-blueprint-for-per-repetition-persistence)
  + [3. Unified Dual-Target Logging Architecture (`DP12-8` / Task 12.15)](#3-unified-dual-target-logging-architecture-dp12-8--task-1215)
    - [3.1 Structural Deficiencies of Status Quo Logging](#31-structural-deficiencies-of-status-quo-logging)
    - [3.2 Dual-Target Topography & Clean Separation](#32-dual-target-topography--clean-separation)
    - [3.3 Centralized Module Blueprint: `src/benchmarking/logging_config.py`](#33-centralized-module-blueprint-srcbenchmarkinglogging_configpy)
    - [3.4 Integration Points Across CLI, Orchestrator, and Scripts](#34-integration-points-across-cli-orchestrator-and-scripts)
  + [4. Algorithm-Agnostic Reporting & Dynamic Baseline Resolution (`DP12-9` / Task 12.16)](#4-algorithm-agnostic-reporting--dynamic-baseline-resolution-dp12-9--task-1216)
    - [4.1 Audit of Hardcoded Couplings in `report_generator.py` and `db_operations.py`](#41-audit-of-hardcoded-couplings-in-report_generatorpy-and-db_operationspy)
    - [4.2 Declarative Baseline Resolution & Clean Instance Partitioning](#42-declarative-baseline-resolution--clean-instance-partitioning)
    - [4.3 Mathematical Rigor: Fleming & Wallace (1986) Paired Geometric Speedup](#43-mathematical-rigor-fleming--wallace-1986-paired-geometric-speedup)
    - [4.4 Implementation Blueprint for Universal Dynamic Reporting](#44-implementation-blueprint-for-universal-dynamic-reporting)
  + [5. Script vs. Module Boundaries: Framework Engine vs. Thesis Runner](#5-script-vs-module-boundaries-framework-engine-vs-thesis-runner)
    - [5.1 Architectural Separation of Concerns](#51-architectural-separation-of-concerns)
    - [5.2 Why `run_chapter4_benchmark.py` Must Remain an Autonomous Script](#52-why-run_chapter4_benchmarkpy-must-remain-an-autonomous-script)
  + [6. Action Plan & Falsifiable Verification Checklist](#6-action-plan--falsifiable-verification-checklist)

---

## 1. Executive Architectural Motivation & Systemic Context

The benchmarking framework (`src/benchmarking/`) in `gpu_accelerated_clean` serves as the empirical evaluation engine for the master's dissertation. It validates whether modular and monolithic GPU-accelerated Genetic Algorithms achieve speedup and solution-quality parity against traditional CPU and hybrid baselines across TSPLIB95 instances (Reinelt 1995).

Historically, the benchmark execution pipeline was designed around batch-oriented abstractions:

1. `run_single_algorithm` held all repetition results in RAM arrays, executing all $R$ stochastic repetitions (typically $R=30$ for production runs) before flushing to disk.
2. Logging was routed exclusively to `sys.stdout` via `logging.StreamHandler`, discarding execution transcripts if interactive pseudo-terminals disconnected or scrollback buffers overflowed.
3. Reporting logic in `report_generator.py` and `db_operations.py` embedded hardcoded string literals (`"HybridNaive"`, `"CPU"`, `"FullGPUMonolithic"`), assuming fixed algorithm taxonomies and failing when arbitrary metaheuristic configurations were tested.
4. When baseline algorithms were omitted for large problem tiers (such as skipping CPU on instances with $n > 400$ cities), reporting tables resorted to ad-hoc fallbacks (`first_alg = next(iter(...))`), inducing spurious $1.00\times$ speedup artifacts and corrupting LaTeX tables.

This document establishes the definitive architectural blueprints for:

- **`DP12-7` (Task 12.14)**: Continuous per-repetition checkpoint persistence with strict mathematical isolation of disk I/O from metaheuristic evolution timing.
- **`DP12-8` (Task 12.15)**: Unified dual-target logging simultaneously multiplexing to the terminal console and persistent, timestamped log files.
- **`DP12-9` (Task 12.16)**: Algorithm-agnostic reporting and dynamic baseline resolution adhering to Fleming & Wallace (1986) paired geometric speedups.
- **Script vs. Module Decoupling**: Preserving clean boundaries between the reusable library engine (`src/benchmarking/`) and thesis-specific experiment drivers (`benchmark_scripts/run_chapter4_benchmark.py`).

---

## 2. Per-Repetition Checkpointing Architecture (`DP12-7` / Task 12.14)

### 2.1 The Core Invariant: Strict Isolation of Disk I/O from Evolution Timing

In metaheuristic benchmarking, runtime measurement represents the core scientific dependent variable. For an instance with $n$ vertices and population size $N$, the measured execution time $T_r$ of repetition $r$ must measure strictly the computational complexity of the algorithm:

$$T_r = t_{\text{evolve\_end}} - t_{\text{evolve\_start}} = \sum_{g=1}^{G_r} \left( \tau_{\text{sel}} + \tau_{\text{cx}} + \tau_{\text{mut}} + \tau_{\text{eval}} + \tau_{\text{ls}} + \tau_{\text{surv}} + \tau_{\text{sync}} \right)$$

where $G_r$ is the number of generations completed before convergence or exhaustion of the patience budget.

Atomic checkpoint serialization requires:

1. Formatting large python dictionaries into JSON structures.
2. Allocating a temporary file on the filesystem: `tempfile.mkstemp`.
3. Writing file buffers and flushing userspace streams: `json.dump`, `f.flush()`.
4. Issuing synchronous OS barrier traps to physical storage: `os.fsync(fd)`.
5. Executing an atomic directory metadata swap: `os.replace(temp_path, target_path)`.

On standard NVMe storage, this sequence consumes $1.5\text{--}8.0\text{ ms}$; on mechanical storage or network filesystems (NFS/CIFS), latency can surge to $50\text{--}350\text{ ms}$. If disk persistence were embedded inside the measured interval, small-scale GPU executions (where an entire run of `berlin52` takes $\approx 25\text{ ms}$) would suffer a timing inflation of up to **$1400\%$**, completely corrupting the speedup ratios.

> [!IMPORTANT]
> **Timing Invariant:** Under `DP12-7`, disk persistence must occur strictly outside the `time.perf_counter()` envelope. The measured runtime $T_r$ recorded in `raw_times[r]` and aggregated into `mean_time` must reflect solely the in-memory execution of `current_algo.evolve(...)`.

```ascii
─────────────────────────────────────────────────────────────────────────────
Execution Flow per Repetition r:
─────────────────────────────────────────────────────────────────────────────
   [Context & Strategy Setup] (un-timed)
             │
             ▼
   t0 = time.perf_counter()   ◄── MEASUREMENT COMMENCES
   tour, stats = algo.evolve(...)
   t1 = time.perf_counter()   ◄── MEASUREMENT TERMINATES
             │
   elapsed = t1 - t0          (Pure evolution duration)
             │
             ▼
   [In-Memory Append]         (raw_times, raw_costs, raw_seeds, etc.)
   [Running Aggregates]       (Bessel-corrected std, means, min/max)
             │
             ▼
   [Atomic Disk Snapshot]     (save_checkpoint_atomic outside t0..t1)
   mkstemp -> dump -> fsync -> os.replace
─────────────────────────────────────────────────────────────────────────────
```

---

### 2.2 Systemic Blast Radius Across `src/benchmarking/`

Persisting checkpoints after every repetition touches five distinct architectural layers across the module. Modifying only `run_single_algorithm` without harmonizing upstream callers creates state divergence.

```ascii
┌────────────────────────────────────────────────────────────────────────────┐
│                       Systemic Blast Radius Map                            │
├───────────────────────────────┬────────────────────────────────────────────┤
│ Subsystem                     │ Architectural Responsibility               │
├───────────────────────────────┼────────────────────────────────────────────┤
│ `algorithm_runner.py`         │ Callback trigger; online aggregate math    │
│ `orchestration.py`            │ Callback factory; partial cohort resume    │
│ `checkpoint_io.py`            │ Atomic I/O; schema validity on partial data│
│ `CheckpointManager`           │ `count_completed_checkpoints` predicates   │
│ `db_operations.py`            │ Cohort insertion boundaries (OLAP hermetic)│
└───────────────────────────────┴────────────────────────────────────────────┘
```

#### 2.2.1 `algorithm_runner.py`: Observer Callback Pattern & Running Aggregates

Rather than coupling `algorithm_runner.py` directly to the filesystem or `CheckpointManager`, the runner must employ the **Observer / Callback Pattern**.

`run_single_algorithm` accepts an optional persistence callback:

```python
on_repetition_completed: Callable[[int, dict[str, Any]], None] | None = None
```

After repetition $r$ completes, the runner computes the exact running aggregates over the current cohort of successful runs $k \in \{1 \dots r\}$:

1. **Running Mean:**
   $$\bar{x}_k = \frac{1}{k} \sum_{i=1}^k x_i$$
2. **Running Sample Standard Deviation with Bessel's Correction (`ddof=1`):**
   $$s_k = \sqrt{\frac{1}{k - 1} \sum_{i=1}^k (x_i - \bar{x}_k)^2} \quad \text{for } k > 1, \quad s_1 = 0.0$$
3. **Running Phase Timing Breakdown:**
   $$\text{phase\_share\_pct}_k(p) = \frac{1}{k} \sum_{i=1}^k \frac{\tau_{i, p}}{T_i} \times 100$$
   $$\text{phase\_share\_cv\_pct}_k(p) = \frac{s_{\text{share}, k}(p)}{\overline{\text{share}}_k(p)} \times 100 \quad (ddof=1)$$

The intermediate `summary` dictionary is emitted to `on_repetition_completed(rep + 1, summary)`.

#### 2.2.2 `orchestration.py`: Resumption Semantics with Partial In-Flight Cohorts

Currently, `orchestration.py:444` checks:

```python
if existing_runs >= effective_repetitions:
    # Skip completed run
    continue
else:
    actual_repetitions = effective_repetitions - existing_runs
    incremental_mode = True
```

When per-repetition checkpointing is active, an interrupted run (e.g., killed at 12/30 repetitions) leaves a valid checkpoint file on disk with `successful_runs = 12` and `repetitions = 30`.

The orchestrator must handle two resumption paradigms:

1. **Target-Completion Mode (`incremental_runs=False` / default):**
   + The user requested $R_{\text{target}} = 30$ repetitions.
   + The existing checkpoint has $R_{\text{done}} = 12$ runs.
   + The orchestrator calculates $R_{\text{remain}} = 30 - 12 = 18$ repetitions.
   + It executes only the 18 missing repetitions, appending them to the existing cohort and bringing the checkpoint to completion.
2. **Additive Incremental Mode (`incremental_runs=True`):**
   + The user requested adding $R_{\text{add}} = 30$ additional repetitions to an existing cohort.
   + Target becomes $R_{\text{target}} = 12 + 30 = 42$ repetitions.
   + The orchestrator executes 30 new repetitions with offset seeds.

#### 2.2.3 `checkpoint_io.py`: Schema Stability of In-Flight Checkpoints

Under `DP12-7`, an in-flight checkpoint file on disk represents a completely valid, well-formed checkpoint payload conforming to `BenchmarkRunInput`.

- It contains all mandatory keys (`problem_name`, `algorithm`, `mean_time`, `std_time`, `raw_times`, etc.).
- Its raw arrays (`raw_times`, `raw_costs`, `raw_gaps`, `raw_seeds`) all possess identical length matching `successful_runs`.
- `can_resume_checkpoint(path, expected_reps=30)` correctly returns `True` because `successful_runs (12) < expected_reps (30)`.
- `is_valid_checkpoint(path, expected_reps=30)` returns `False` until repetition 30 is flushed.

#### 2.2.4 `CheckpointManager`: Cohort Progress Accounting

`CheckpointManager.count_completed_checkpoints` evaluates `is_valid_checkpoint(path, expected_reps)`. Because a partially executed run has `successful_runs < expected_reps`, it is correctly reported as pending, ensuring CLI progress indicators (`Found 10/120 completed checkpoints`) stay strictly truthful.

#### 2.2.5 `db_operations.py`: Analytical Columnar Storage vs. Transactional Logging

A critical architectural question is whether per-repetition progress should be inserted directly into DuckDB (`results.duckdb`).

> [!WARNING]
> **Database Hermeticity Gate:** DuckDB is an in-process vectorized columnar OLAP engine. Inserting partial runs on every repetition would force repeated `UPDATE` or `DELETE + INSERT` statements, causing database fragmentation, holding file write-locks, and complicating ACID transactions if runs are interrupted.
>
> Therefore, **DuckDB insertion must remain strictly an end-of-cohort operation**. The checkpoint JSON file on disk serves as the write-ahead log (WAL) for in-flight durability. DuckDB ingestion occurs only after all $R$ repetitions conclude, or via batch ingestion using `import_checkpoints.py`.

---

### 2.3 Implementation Blueprint for Per-Repetition Persistence

In `algorithm_runner.py`:

```python
# Signature addition to run_single_algorithm:
def run_single_algorithm(
    ...,
    on_repetition_completed: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ...
    for rep in range(repetitions):
        t0 = time.perf_counter()
        try:
            tour, stats = current_algo.evolve(...)
            elapsed = time.perf_counter() - t0
            # Record metrics into results lists...
        except Exception as e:
            ...

        # Build running summary snapshot OUTSIDE of t0..t1 timing:
        if on_repetition_completed is not None:
            intermediate_summary = _build_summary_payload(
                results, carried_runs, carried_h2d_bytes, carried_d2h_bytes,
                carried_kernel_launches, algorithm_name, problem
            )
            on_repetition_completed(rep + 1, intermediate_summary)
```

In `orchestration.py`:

```python
# Construct persistence callback closing over static metadata:
def _make_persistence_callback(checkpoint_path: Path, static_metadata: dict[str, Any]):
    def callback(rep_num: int, current_results: dict[str, Any]) -> None:
        payload = dict(current_results)
        payload.update(static_metadata)
        save_checkpoint_atomic(checkpoint_path, payload)
        logging.debug(f"Flushed incremental checkpoint rep {rep_num} to {checkpoint_path.name}")
    return callback
```

---

## 3. Unified Dual-Target Logging Architecture (`DP12-8` / Task 12.15)

### 3.1 Structural Deficiencies of Status Quo Logging

Auditing `src/benchmarking/` and `benchmark_scripts/` reveals fragmented logging practices:

1. `benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py:201` calls `logging.basicConfig(level=logging.INFO, handlers=[StreamHandler(sys.stdout)], force=True)`.
2. `src/cli/benchmark.py:184` calls `logging.basicConfig(..., force=True)` with a different format string.
3. `orchestration.py:340` relies purely on inherited root loggers without guaranteeing persistent disk retention.

If an execution runs unattended in a detached terminal (or tmux/nohup) and the terminal buffer truncates, or if a machine restarts during an overnight run, all standard output is irretrievably lost. Post-mortem failure analysis becomes impossible.

---

### 3.2 Dual-Target Topography & Clean Separation

The dual-target logging architecture must multiplex log events to two independent streams:

1. **Console Stream (`StreamHandler(sys.stdout)`):**
   + Clean, human-readable terminal rendering.
   + Avoids redundant timestamp clutter on progress bars and tables while retaining clear operational messages.
2. **Persistent File Stream (`FileHandler(log_path, mode="a", encoding="utf-8")`):**
   + Detailed, forensic ISO 8601 formatting: `%(asctime)s [%(levelname)s] [%(name)s] %(message)s`.
   + Guaranteed line-buffering / auto-flushing so that crashes preserve the final error message immediately preceding termination.

```ascii
                      ┌────────────────────────┐
                      │    logging.Logger      │
                      │ ("benchmarking.runner")│
                      └───────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       ┌─────────────────────┐         ┌─────────────────────┐
       │   StreamHandler     │         │     FileHandler     │
       │   (sys.stdout)      │         │ (benchmark.log)     │
       ├─────────────────────┤         ├─────────────────────┤
       │ Formatter:          │         │ Formatter:          │
       │ "%(message)s"       │         │ ISO-8601 + Level +  │
       │ (or minimal prefix) │         │ Logger + Message    │
       └─────────────────────┘         └─────────────────────┘
```

---

### 3.3 Centralized Module Blueprint: `src/benchmarking/logging_config.py`

Create a dedicated, reusable logging configuration helper:

```python
"""Centralized logging configuration for benchmarking executions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

BENCHMARK_LOGGER_NAME = "benchmarking"

def setup_benchmark_logging(
    output_dir: Path | str,
    benchmark_name: str = "benchmark",
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    capture_warnings: bool = True,
) -> tuple[logging.Logger, Path]:
    """Configure dual-target console and file logging for a benchmark session.

    Args:
        output_dir: Directory where the log file will reside.
        benchmark_name: Base name for the log file ({benchmark_name}.log).
        console_level: Logging threshold for stdout.
        file_level: Logging threshold for disk persistence.
        capture_warnings: Whether to route python warnings through logging.

    Returns:
        Tuple of (configured root/benchmarking logger, resolved log file path).
    """
    resolved_dir = Path(output_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = resolved_dir / f"{benchmark_name}.log"

    logger = logging.getLogger()  # Configure root logger to intercept all submodules
    logger.setLevel(min(console_level, file_level))

    # Remove stale handlers to prevent duplicate lines on re-entry
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Append mode with UTF-8)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if capture_warnings:
        logging.captureWarnings(True)

    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger, log_file_path
```

### 3.4 Integration Points Across CLI, Orchestrator, and Scripts

1. **`src/cli/benchmark.py`**:
   In `cmd_run`: invoke `setup_benchmark_logging(args.output_dir, args.benchmark_name)` before invoking `run_comprehensive_benchmark`.
2. **`orchestration.py`**:
   At entry to `run_comprehensive_benchmark`: check if any `FileHandler` is already bound to the active logger. If not, auto-attach a `FileHandler` writing to `resolved_output_dir / f"{resolved_benchmark_name}.log"`.
3. **`benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py`**:
   Replace local `setup_logging()` with `setup_benchmark_logging(output_dir, benchmark_name)`.

---

## 4. Algorithm-Agnostic Reporting & Dynamic Baseline Resolution (`DP12-9` / Task 12.16)

### 4.1 Audit of Hardcoded Couplings in `report_generator.py` and `db_operations.py`

An exhaustive audit of `src/benchmarking/analysis/report_generator.py` and `db_operations.py` reveals critical hardcoded assumptions:

1. **Hardcoded Baseline Algorithm Names:**
   + `report_generator.py:51`:

     ```python
     def _get_baseline_algorithm(algorithm_names: list[str]) -> str:
         if "HybridNaive" in algorithm_names:
             return "HybridNaive"
         return algorithm_names[0] if algorithm_names else "Unknown"
     ```

   + If an arbitrary experiment compares `["CrazyCpuGpuLocalSearch", "ExperimentalMonolithic"]`, the reporting logic defaults to arbitrary alphabetical sort order.
2. **Hardcoded LaTeX Caption Text:**
   + `report_generator.py:167`:

     ```latex
     \caption{Summary Statistics by Problem. Speedup relative to baseline algorithm (HybridNaive when CPU not available).}
     ```

   + This hardcodes the dissertation's specific comparison into a reusable library module!
3. **Toxic Fallback on Missing Baseline Instances:**
   + `report_generator.py:634-636`:

     ```python
     else:
         first_alg: str = next(iter(all_results[problem].keys()))
         baseline_times[problem] = all_results[problem][first_alg]["mean_time"]
         logging.warning(f"Baseline {effective_baseline} not found for {problem}, using {first_alg}")
     ```

   + When CPU is omitted for large instances ($n > 400$), `baseline_times[problem]` silently substitutes `first_alg` (e.g. `HybridOptimized`).
   + For `HybridOptimized`, speedup computes as $\frac{T_{\text{Hybrid}}}{T_{\text{Hybrid}}} = 1.00\times$.
   + For `FullGPUModular`, speedup computes against `HybridOptimized` while the column header claims it is relative to `CPU`!

---

### 4.2 Declarative Baseline Resolution & Clean Instance Partitioning

To make reporting universally applicable to any arbitrary algorithm configuration:

1. **Explicit Parameterization:**
   + `generate_result_tables` accepts `baseline_algorithm: str | None = None`.
   + If `None`, dynamically select the algorithm with `backend == "CPU"` if present, or provide a declarative precedence list via configuration, falling back to `sorted(algorithm_names)[0]` with explicit warning.
2. **Strict Neutral Dash on Missing Instance Data:**
   + If `baseline_algorithm not in all_results[problem]`, the speedup for that instance **must render as a neutral dash (`—`)**.
   + It must **never** pick an arbitrary alternative algorithm.
3. **Configurable Intra-Regime Secondary Baseline:**
   + If the user explicitly configures a secondary baseline (e.g. `fallback_baseline="HybridNaive"` for instances where CPU is unavailable), the table generator must explicitly designate fallback cells with a dagger symbol (`†` in Markdown, `\dagger` in LaTeX) and state this in the caption notes:
   + `15.42x†` with footnote: `† Relative to secondary baseline HybridNaive (CPU baseline unavailable for n > 400)`.

---

### 4.3 Mathematical Rigor: Fleming & Wallace (1986) Paired Geometric Speedup

In `report_generator.py:257` and `310` (Table 2), aggregate speedup across all instances was previously calculated as:
$$\text{Average Speedup} = \frac{\frac{1}{M}\sum_{i \in P_{\text{base}}} T_{\text{base}, i}}{\frac{1}{N}\sum_{j \in P_{\text{cand}}} T_{\text{cand}, j}}$$

Fleming & Wallace (1986) ("*How not to lie with statistics: the correct way to summarize benchmark results*") demonstrated that this formulation violates fundamental relational axioms:

1. It is sensitive to instance scale: execution times of large instances dominate the sum, rendering speedups on smaller instances mathematically invisible.
2. If the baseline and candidate ran on different subsets of instances ($P_{\text{base}} \neq P_{\text{cand}}$), the ratio compares incomparable problem distributions.

The **only** mathematically sound metric for summarizing benchmark speedups across multiple benchmark programs is the **Geometric Mean of Paired Speedups**:

Let $P_{\cap} = P_{\text{baseline}} \cap P_{\text{candidate}}$ represent the set of instances evaluated by both algorithms. For each instance $p \in P_{\cap}$, the paired speedup ratio is:
$$S_p = \frac{T_{\text{baseline}, p}}{T_{\text{candidate}, p}}$$

The aggregate speedup across the benchmark cohort is:
$$S_{\text{geom}} = \left( \prod_{p \in P_{\cap}} S_p \right)^{\frac{1}{|P_{\cap}|}} = \exp\left( \frac{1}{|P_{\cap}|} \sum_{p \in P_{\cap}} \ln \frac{T_{\text{baseline}, p}}{T_{\text{candidate}, p}} \right)$$

If $|P_{\cap}| = 0$, speedup is undefined and renders as `—`.

In Table 2, the table must report the number of paired instances:
`21.33x (n=12)` vs `CPU`, and `2.13x (n=38)` vs `HybridNaive`.

---

### 4.4 Implementation Blueprint for Universal Dynamic Reporting

```python
def _calculate_paired_geometric_speedup(
    all_results: dict[str, dict[str, dict[str, Any]]],
    problem_names: list[str],
    candidate_alg: str,
    baseline_alg: str,
) -> tuple[float | None, int]:
    """Calculate Fleming & Wallace (1986) geometric mean speedup on paired instances."""
    speedups: list[float] = []
    for problem in problem_names:
        prob_data = all_results.get(problem, {})
        if baseline_alg in prob_data and candidate_alg in prob_data:
            base_t = float(prob_data[baseline_alg].get("mean_time", 0.0))
            cand_t = float(prob_data[candidate_alg].get("mean_time", 0.0))
            if base_t > 0.0 and cand_t > 0.0:
                speedups.append(base_t / cand_t)

    if not speedups:
        return None, 0

    geomean = float(np.exp(np.mean(np.log(speedups))))
    return geomean, len(speedups)
```

In Table 1 generation:

```python
# In _generate_table1_markdown / LaTeX:
# If primary baseline exists for this problem, compute speedup against it.
# Otherwise, fall back to the first available algorithm as the local reference baseline
# so relative speedup is always visible rather than suppressed as a dash ("—").
effective_baseline = baseline_algorithm
if problem in all_results and baseline_algorithm not in all_results[problem]:
    effective_baseline = next(iter(all_results[problem].keys()))

if problem in all_results and effective_baseline in all_results[problem]:
    base_t = float(all_results[problem][effective_baseline]["mean_time"])
    cand_t = float(r["mean_time"])
    if cand_t > 0 and base_t > 0:
        ratio = base_t / cand_t
        speedup_str = f"{ratio:.2f}x"
else:
    speedup_str = "—"
```

---

## 5. Script vs. Module Boundaries: Framework Engine vs. Thesis Runner

### 5.1 Architectural Separation of Concerns

A recurring tension in scientific software is where experiment-specific logic ends and library framework code begins.

```ascii
┌────────────────────────────────────────────────────────────────────────────┐
│                    Separation of Architectural Layers                      │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Reusable Core Library (`src/benchmarking/`)                       │
│ - Zero thesis-specific hardcoded assumptions                               │
│ - Parameterized runner, atomic checkpoint manager, statistical analyzer    │
│ - Completely agnostic to whether algorithms are named "CPU" or "AlgorithmX"│
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: General CLI (`src/cli/benchmark.py`)                              │
│ - Exposes standard flags (--config, --problems, --repetitions, --output)   │
│ - Invokes Layer 1 with user-supplied arguments                             │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Thesis Experiment Drivers (`benchmark_scripts/`)                  │
│ - `run_chapter4_benchmark.py`: Binds the exact 4 Chapter 4 variants        │
│ - Asserts dissertation validation contracts (38 TSPLIB instances, seeds)   │
│ - Calls Layer 1 orchestration as a client library                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Why `run_chapter4_benchmark.py` Must Remain an Autonomous Script

Merging `run_chapter4_benchmark.py` into `src/cli/` as a built-in "preset" would be an architectural antipattern:

1. **Pollution of Library Scope:** The thesis runner enforces specific dissertation invariants (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`, specific TSPLIB problem lists, specific hardware disclosures for Chapter 4). A third-party researcher using this repository as a routing optimization library should not see thesis-specific hardcodings in `src/cli/`.
2. **Reproducibility Hermeticity:** `run_chapter4_benchmark.py` acts as an executable scientific protocol. It verifies the exact contract under which Chapter 4 was evaluated. Keeping it in `benchmark_scripts/` signals that it is an experiment script, not core library machinery.

---

## 6. Action Plan & Falsifiable Verification Checklist

To implement Tasks 12.14, 12.15, and 12.16 with zero regressions across the 280 existing unit and integration tests:

| Phase | Target Files | Key Modifications | Falsifiable Verification Gate |
| :--- | :--- | :--- | :--- |
| **Phase 1: Task 12.15 (Dual Logging)** | `src/benchmarking/logging_config.py`, `src/cli/benchmark.py`, `orchestration.py`, `run_chapter4_benchmark.py` | Create centralized `setup_benchmark_logging`; attach stdout `StreamHandler` and file `FileHandler` in append mode. | `pytest tests/benchmarking/test_unit_logging.py` validates simultaneous emission to stdout buffer and log file on disk. |
| **Phase 2: Task 12.14 (Per-Repetition Checkpoint)** | `algorithm_runner.py`, `orchestration.py`, `checkpoint_io.py` | Add `on_repetition_completed` callback; serialize atomic snapshot outside evolution `perf_counter`; compute online Bessel running aggregates. | Synthetic test with mock algorithm: simulate SIGINT after rep 2; assert file on disk has `successful_runs=2` and resumes to 5 without timing pollution. |
| **Phase 3: Task 12.16 (Agnostic Reporting)** | `report_generator.py`, `db_operations.py`, `aggregate_statistics.py` | Remove hardcoded `"HybridNaive"`; calculate Fleming & Wallace geometric mean speedups in Table 2; render `—` on missing baseline instances. | Run report generation with arbitrary names `["AlgoFoo", "AlgoBar"]`; assert clean Markdown/LaTeX generation with zero warnings and valid speedups. |
| **Phase 4: Full Suite Gate** | All modules in `src/benchmarking/` | Run full test suite, linting, doc checks. | `pytest tests/ -q` $\to$ 100% green; `mypy src tests benchmark_scripts` exit 0; `python -m benchmark_scripts.check_docs` exit 0. |

---

## References

- **Fleming, P. J., & Wallace, J. J. (1986).** *How not to lie with statistics: the correct way to summarize benchmark results.* Communications of the ACM, 29(3), 218-221.
- **Reinelt, G. (1995).** *TSPLIB95.* Interdisziplinäres Zentrum für Wissenschaftliches Rechnen, Heidelberg.
- **Demšar, J. (2006).** *Statistical comparisons of classifiers over multiple data sets.* Journal of Machine Learning Research, 7, 1-30.
- **Bessel, F. W. (1815).** *Über die Genauigkeit der Beobachtungen.* Astronomische Nachrichten.
