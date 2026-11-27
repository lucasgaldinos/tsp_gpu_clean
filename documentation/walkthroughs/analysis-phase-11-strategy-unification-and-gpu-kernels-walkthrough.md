---
title: "Phase 11 — Strategy Unification, Composable Pipeline Steps, and CUDA Kernel Architecture Walkthrough"
description: >
  Comprehensive technical walkthrough of Phase 11: StrategyBase architecture,
  STAGE_SPECS declarative dispatch (BUG-21), explicit host_sync PhaseTimer bucket (DP11-13),
  GPU benchmark warm-up (BUG-23), 1-warp-per-route Order Crossover (Format A, BUG-22),
  warp-shuffle reduction, cooperative 2-opt reversal (2OPT-A), and post-review remediations.
created: 2026-11-10
type: analysis
modifications:
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Initial creation documenting Phase 11 implementations (Tasks 11.1–11.16),
          empirical hardware measurements on GTX 1050 (GP107), occupancy calculations,
          falsifiable test gates, and code-review refactorings.
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Batch 4 closure of Tasks 11.7, 11.11 and 11.13 (BUG-30 telemetry half, BUG-23/BUG-25
          AOT warm-up, EVAL-A h2d accounting) with the measured zero-JIT probe; §10 added and
          §1/§6/§9 status notes reconciled.
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Post-audit reconciliation: Tasks 11.1, 11.6–11.11 and 11.12–11.16 reopened to `[ ]`
          under BUG-27–BUG-30, Task 11.17 created (BUG-26/DP11-22), §4.1 reclassified as the
          BUG-24 topology-cache poisoning, and §6 added with the per-task residual-gap table.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[src/algorithms/metaheuristics/genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py)"
  - "[src/algorithms/pipeline_steps.py](../../src/algorithms/pipeline_steps.py)"
  - "[src/algorithms/strategies/crossover/kernels/order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[src/algorithms/strategies/crossover/order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[src/algorithms/strategies/evaluation/kernels/tour_cost.cu](../../src/algorithms/strategies/evaluation/kernels/tour_cost.cu)"
  - "[src/algorithms/strategies/evaluation/cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py)"
  - "[src/algorithms/strategies/local_search/kernels/two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[src/algorithms/strategies/local_search/two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[src/utils/gpu_helpers.py](../../src/utils/gpu_helpers.py)"
  - "[crossover_scaling_sweep.py](../../benchmark_scripts/experiments-and-examples/chapter4/crossover_scaling_sweep.py)"
tags:
  - analysis/architecture
  - analysis/decisions
  - algorithm/gpu
  - benchmark/hardware
  - cuda/warp-intrinsics
  - review/quality
---

# Phase 11 — Strategy Unification, Composable Pipeline Steps, and CUDA Kernel Architecture

## 1. Overview and Executive Summary

Phase 11 represents the major architectural restructuring and GPU kernel overhaul of the genetic routing solver, executing decisions `DP11-1` through `DP11-18` across 16 formal tasks (`Tasks 11.1`–`11.16`).

> [!NOTE]
> **Status after the 2026-11-10 5-axis audit and the Batch 4 closure.** Every Phase 11 task (`Tasks 11.1`–`11.17`) is `[x]`: the reopening recorded in [§6](#6-post-audit-reopenings-2026-11-10) was closed by §7 (Tasks 11.1/11.16/11.17), §8 (Tasks 11.10/11.12/11.14/11.15), §9 (Tasks 11.6/11.8/11.9) and §10 (Tasks 11.7/11.11/11.13). The unresolved-gap column below is kept as the historical fault statement.

The phase achieved four primary technical objectives:

1. **Algorithmic Parity & Core Cleanliness**: Removed Generation 0 pre-loop local search (`BUG-21`), unifying dispatch under `STAGE_SPECS` and moving $P_0$ refinement into composable `GenerationStep`s (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep`).
2. **Deterministic Profiling & Accounting**: Integrated an explicit 7th `PhaseTimer` bucket (`host_sync`) isolating D2H synchronizations and `_extract_best` latency, and eliminated cold-start JIT contamination via automated GPU benchmark warm-up (`BUG-23`).
3. **GPU Strategy Kernel Overhaul**: Rewrote modular CUDA kernels across crossover, mutation, selection, local search, and evaluation, eliminating forced host synchronization (`Stream.null.synchronize()`), achieving $100\%$ CPU-GPU bit-identity, and saturated occupancy on Pascal GP107.
4. **Order Crossover Complexity Collapse (`BUG-22`, `OX-A`)**: Replaced the $\mathcal{O}(n^2)$ linear search with a Format A shared-memory direct byte table and 1-warp-per-route stream compaction (`__ballot_sync` + `__popc`), reducing $n=2000$ crossover time from $742.05\text{ ms}$ to $2.25\text{ ms}$ ($106.3\times$ speedup vs CPU).

---

## 2. Hardware Occupancy and Launch Mathematics (NVIDIA Pascal GP107 / GTX 1050)

### 2.1 Hardware Specifications

The target baseline GPU is the NVIDIA GeForce GTX 1050 (Pascal GP107, Compute Capability 6.1):

- **Streaming Multiprocessors (SMs)**: 5 SMs.
- **Warp Schedulers**: 4 warp schedulers per SM (each issues up to 1 instruction per cycle).
- **Maximum Resident Threads per SM**: $2{,}048\text{ threads} = 64\text{ warps}$.
- **Maximum Resident Thread Blocks per SM**: $32\text{ blocks}$.
- **Shared Memory Capacity**: $48\text{ KB} = 49{,}152\text{ bytes}$ per thread block / SM.
- **Register File**: $64\text{K}$ $32$-bit registers per SM ($256\text{ KB}$).

### 2.2 Why $\text{threads\_per\_block} = 256$ is the Theoretical Sweet Spot

In the modular kernels (`order_crossover.cu`, `inversion_mutation.cu`), execution is parallelized at the granularity of **1 warp (32 threads) per route**:
$$\text{warps\_per\_block} = \frac{\text{threads\_per\_block}}{32}$$

With $\text{threads\_per\_block} = 256$:

- Each block executes $8\text{ warps}$ ($8$ routes concurrently).
- To achieve $100\%$ theoretical occupancy ($64\text{ resident warps}$ per SM), an SM requires $\frac{64}{8} = 8\text{ resident thread blocks}$.
- Since GP107 permits up to $32\text{ blocks}$ per SM, $8$ blocks comfortably satisfies block limits ($8 \le 32$).

### 2.3 Shared Memory Capacity Ceiling (Format A)

Format A direct-address membership allocates an `unsigned char` array of size $n$ bytes per warp:
$$\text{Shared Memory Required} = W \times n \times 1\text{ byte} = 8n\text{ bytes}$$

Setting this against the $48\text{ KB}$ hardware limit:
$$8n \le 49{,}152\text{ bytes} \implies n \le 6{,}144\text{ cities}$$

For all instances in TSPLIB95 ($n \le 2{,}103$):
$$\text{At } n = 2{,}103: \quad 8 \times 2{,}103 = 16{,}824\text{ bytes} \approx 16.4\text{ KB} \quad (< 35\%\text{ of SM capacity})$$

When $n > 6{,}144$, the dynamic launcher in `order_crossover.py` iteratively halves block size:

- $n \in [6{,}145, 12{,}288]$: downscales to $128$ threads ($4$ warps, max $48\text{ KB}$).
- $n \in [12{,}289, 24{,}576]$: downscales to $64$ threads ($2$ warps, max $48\text{ KB}$).
- $n \in [24{,}577, 49{,}152]$: downscales to $32$ threads ($1$ warp, max $48\text{ KB}$).

---

## 3. The Power-Law Exponent $\beta$ and Host Overhead Analysis

In `crossover_scaling_sweep.py`, power-law scaling is evaluated as:
$$\log(T) = \beta \log(n) + \alpha \iff T(n) = \alpha \cdot n^\beta$$

### 3.1 Pure CUDA Kernel vs Whole Strategy Breakdown

Direct profiling with hardware CUDA events (`cp.cuda.Event`) reveals the true execution time of `order_crossover_kernel` on GTX 1050 ($N=256$):

| Dimension $n$ | Pure CUDA Kernel ($N=256$) | Host Cut Generation (python) | Whole Strategy Wall Time |
| :---: | :---: | :---: | :---: |
| $n = 52$ | **$0.0123\text{ ms}$** ($12.3\ \mu\text{s}$) | $\approx 1.05\text{ ms}$ | $2.405\text{ ms}$ |
| $n = 100$ | **$0.0143\text{ ms}$** ($14.3\ \mu\text{s}$) | $\approx 1.04\text{ ms}$ | $1.765\text{ ms}$ |
| $n = 200$ | **$0.0164\text{ ms}$** ($16.4\ \mu\text{s}$) | $\approx 1.10\text{ ms}$ | $1.911\text{ ms}$ |
| $n = 500$ | **$0.0246\text{ ms}$** ($24.6\ \mu\text{s}$) | $\approx 1.09\text{ ms}$ | $1.647\text{ ms}$ |
| $n = 1000$ | **$0.0389\text{ ms}$** ($38.9\ \mu\text{s}$) | $\approx 1.03\text{ ms}$ | $2.595\text{ ms}$ |
| $n = 2000$ | **$0.0881\text{ ms}$** ($88.1\ \mu\text{s}$) | $\approx 1.05\text{ ms}$ | $2.445\text{ ms}$ |

**Key Findings:**

1. **Pure Kernel Scaling**: The pure CUDA kernel scales with $\beta_{\text{kernel}} \approx \mathbf{0.539}$ ($n^{0.54}$), well below the linear ceiling $\beta \le 1.2$.
2. **Host Bottleneck**: Generating cut points on the host via python `for i in range(128): rng.choice(n, size=2, replace=False)` consumes $\approx 1.05\text{ ms}$ on CPU, representing **$>90\%$ of total strategy time**.
3. **Origin of Negative $\beta$ Artifact**: Without untimed per-size warm-up, the first tested size ($n=52$) incurred initial CuPy memory pool allocation and JIT overhead, measuring $3.08\text{ ms}$, while $n=2000$ measured $1.72\text{ ms}$. This artifactual downward tilt yielded a negative fitted slope ($\beta \approx -0.12$).
4. **Resolution**: Introducing an un-timed per-dimension warm-up pass in `crossover_scaling_sweep.py` stabilized measurement, yielding $\beta_{\text{strategy}} = \mathbf{0.0422} \le 1.2$.

---

## 4. Post-Review Architectural Remediations (DRY & SOLID)

During the multi-axis review gate, several critical defects and code quality issues were identified and immediately remediated:

### 4.1 Instance Cache for Route Topology (`order_crossover.py`) — superseded by `BUG-24`

- **Defect**: Lines 224–225 queried `first_0 = int(d_parents[0, 0].get())` on every generation to detect closed depot endpoints, causing synchronous pipeline stalls.
- **First fix**: `self._is_closed_cache` was added to `CrossoverGPUOrder`, inspecting route topology on generation 1 and caching the boolean, yielding **zero D2H transfers** on later generations.
- **Reclassification (2026-11-10, `BUG-24`)**: the instance cache is keyed only by `(n_cities, is_closed)` and is **poisoned** by zero-filled warm-up arrays (`0 == 0` ⇒ `is_closed=True`), so later open permutations of the same dimension diverge from `CrossoverOrder` (`E1 cache after zeros warm-up: (52, True)`, `GPU==CPU offspring after poisoned warm-up: False`). `DP11-19` therefore mandates removing the cache and deriving bounds from `resolve_route_bounds(n, context, route_sample)` instead of instance state.
- **Kept**: the hardware query replaced hardcoded `48 * 1024` with `cp.cuda.Device().attributes['MaxSharedMemoryPerBlock']`.

### 4.2 Centralization of `next_power_of_2` (`src/utils/gpu_helpers.py`)

- **Defect**: Duplicated private functions `_next_power_of_2` existed in `cost_evaluation.py` and `two_opt.py` with divergent clamping limits ($32$ vs $1$).
- **Fix**: Centralized canonical `next_power_of_2(value, min_val, max_val)` in [src/utils/gpu_helpers.py](../../src/utils/gpu_helpers.py#L67) and eliminated the local wrapper functions.

### 4.3 DRY Refactoring of Pipeline Steps (`src/algorithms/pipeline_steps.py`)

- **Defect**: `PostInitLocalSearchStep` and `EliteOnlyLocalSearchStep` duplicated strategy validation.
- **Fix**: Created `_BaseLocalSearchPipelineStep` encapsulating constructor injection and validation.

### 4.4 Falsifiable Test Gates

Added structural verification tests that inspect compiled kernels or PTX source, ensuring tests fail if optimizations are regressed:

- `test_cost_calculator_kernel_uses_warp_shuffle_and_stride` in `test_unit_gpu_cost_evaluation.py`.
- `test_gpu_tournament_kernel_emits_1d_winner_indices` in `test_unit_gpu_tournament_selection.py`.
- `test_gpu_two_opt_kernels_use_block_wide_cooperative_reversal` in `test_unit_two_opt_boundaries.py`.

---

## 5. Empirical Verification Record

All gates below were verified against the code as committed for `Tasks 11.1`–`11.16`; they do **not** certify the reopened items in §6.

- `pytest tests/algorithms tests/benchmarking -m "not slow" -q`: **587 passed in 9.57s**.
- `pytest tests/algorithms/strategies/ -q`: **268 passed in 2.04s**.
- `ruff check src tests benchmark_scripts`: **All checks passed!**
- `mypy src tests benchmark_scripts`: **Success: no issues found in 137 source files**.
- `PYTHONPATH=src python -m benchmark_scripts.check_docs`: **stale docstrings: 0, dead links: 0, undefined symbols: 0 (exit code 0)**.

## 6. Post-Audit Reopenings (2026-11-10)

The 5-axis audit reopened every task whose perimeter contained a residual defect. Each reopened task carries a fault statement and an executable **Proof of Concept (PoC)** in [plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md).

| Task | Decision(s) | Defect(s) | Residual gap |
| :--- | :--- | :--- | :--- |
| `11.1` | `DP11-1`, `DP11-2`, `DP11-19`, `DP11-20` | `BUG-20`, `BUG-27` | `_improve_grid` does per-route `int(tour.max())` D2H + re-uploads a NumPy candidate table; `mode` not validated; `StrategyBase` class-level counters |
| `11.6` | `DP11-6` | `BUG-30` | Pipeline step `params` dropped from checkpoint; `algorithms.full.example.json` runs 2-opt twice on $P_0$ |
| `11.7` | `DP11-7`, `DP11-13` | `BUG-30` | `survival` missing from Table 4; `np.std(ddof=0)`; `_extract_best` D2H every generation; `incremental_mode` loses telemetry; `phase_timings` absent from DuckDB |
| `11.8` | `DP11-8`, `DP11-9` | `BUG-21`, `BUG-30` | `_all_strategies()` ignores `self.pipeline`; `_compose_pipeline` anchors by class name instead of `stage_anchor` |
| `11.9` | `DP11-10` | `BUG-30` | Steps expose no `stage_anchor`; `elite_count`/`interval` never serialized |
| `11.10` | `DP11-11`, `DP11-19`, `DP11-20` | `BUG-22`, `BUG-24` | `_is_closed_cache` poisoning; serial cut sampling (1.217 ms vs 0.0168 ms) |
| `11.11` | `DP11-12`, `DP11-21` | `BUG-23`, `BUG-25` | Warm-up shares the measured instance; `max_generations=1` never triggers `PeriodicLocalSearchStep`, so JIT hits rep 1 (`rep1/median=26.27x`) |
| `11.12`, `11.15` | `DP11-14`, `DP11-17`, `DP11-19`, `DP11-20` | `BUG-28` | GPU mutations lack `candidate_k`, use `_mutation_bounds` (open tours lose the $0$/$n-1$ moves) and keep a serial sampling loop |
| `11.13` | `DP11-15` | `BUG-26`, `BUG-30` | Kernel is sound, but `@time_strategy` still blocks and `input_on_gpu=False` misses `self.h2d_bytes += int(d_population.nbytes)` |
| `11.14` | `DP11-16` | `BUG-28` | Candidates sampled as `int64`; `is_gpu=True` path uses `cp.random.randint` instead of `resolve_rng(context, self.rng)` |
| `11.16` | `DP11-18` | `BUG-29` | Missing `tour_idx >= num_tours` guard; 5 wasted `__syncthreads()` when `s_deltas[0] >= 0`; two serial reversals remain in `two_opt_multi_move.cu`; `multi_move` retention diverges from CPU |
| `11.17` | `DP11-22` | `BUG-26` | `@time_strategy` forces `end_evt.synchronize()` on every GPU array: `wrapped=260.840ms` vs `unwrapped_enqueue=0.363ms` |

## 7. Closure of `Tasks 11.1`, `11.16` and `11.17` (2026-11-10)

Three of the reopened items are closed here; the rows still listed as open at that point were closed later in §8–§10.

### 7.1 `Task 11.17` — `@time_strategy` no longer blocks (`DP11-22`, `BUG-26`)

The decorator used to inspect every positional argument with `isinstance(arg, cp.ndarray)` and, on a match, build a `cp.cuda.Event` pair ending in `end_evt.synchronize()`. Five to six strategies receive a GPU array per generation, so the generation carried five to six hidden host-device barriers, cancelling the asynchronous pipeline that `DP10-18`, `SWAP-A`, `EVAL-A`, `SEL-A` and `INV-A` were built for.

The default path now only reads `time.perf_counter()`, so a GPU strategy returns to the host as soon as its kernels are *enqueued*. Device-accurate timing stays reachable but is opt-in: a strategy exposes `use_gpu_events = True`, and `PhaseTimer(use_cuda_events=True)` keeps the barrier-and-event path for callers that need a device clock. `StrategyBase` declares `last_elapsed_ms`, `total_elapsed_ms` and `call_count` as typed instance attributes, giving the decorator a static home instead of a dynamic attribute.

```shellsession
$ enqueue probe — LocalSearchGPUTwoOpt(mode="batch"), N=256, n=500, resident population
enqueue=0.24ms wrapped_full=154.20ms last_elapsed=0.08ms
```

`last_elapsed_ms` (0.08 ms) tracks the enqueue, not the 154.20 ms device drain that used to be charged to every call. `grep -n "end_evt.synchronize" src/utils/phase_timer.py` now matches only line 59 (`PhaseTimer.measure`, opt-in by `use_cuda_events`) and line 118 (the `use_gpu_events` branch) — never the default path.

### 7.2 `Task 11.1` — candidate lists stay in VRAM (`DP11-1`, `BUG-20`, `BUG-27`)

`_improve_grid` used to rebuild the candidate table on the host, call `int(tour.max())` on the CUDA array (one D2H sync per generation) and re-upload `N_k` on every call. `_DistanceOperandCache.resolve_candidates(context, k)` now binds the table owned by `ProblemContext.get_gpu_candidate_list`, and `_uses_candidates` decides the candidate path from host metadata only (`problem_type == "CVRP"`, `n < context.dimension`, tiling), so the hot path never materializes a host copy. `_shared_mem_bytes(..., use_candidates=True)` also stops reserving `n * 4` bytes of `s_tour`, and both grid kernels place `s_deltas` at the first shared slot in that mode.

| Measurement (`n=100`, `k=20`, 5 generations, GPU-resident batch) | Before | After |
| :--- | :--- | :--- |
| `h2d_bytes` | 80,000 (matrix + 5 × 8,000) | **48,000** (matrix + 1 × 8,000) |
| `d2h_bytes` | > 0 (`int(tour.max())`) | **0** |

### 7.3 `Task 11.16` — kernel bounds, reduction and cooperative reversal (`DP11-18`, `BUG-29`)

`two_opt_batch.cu` gained `if (tour_idx >= num_tours) return;` before touching `tours` or shared memory. Both single-pass kernels now fold thread partials in parallel only while the stride exceeds one warp and let thread 0 scan just `1 .. min(block_size, 64) - 1` (it used to rescan all `block_size` partials the tree had already merged), short-circuiting when `s_deltas[0] >= 0`. `two_opt_multi_move.cu` keeps the inherently serial greedy selection in thread 0 but publishes the winning segment through `s_move_start`/`s_move_end` and reverses it block-cooperatively, so no serial `while (left < right)` remains in any kernel directory.

```bash
$ grep -rn "while (left < right)" src/algorithms/strategies/local_search/kernels/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache} | wc -l
0
$ grep -rn "for (int i = 1; i < block_size; i++)" src/algorithms/strategies/local_search/kernels/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache} | wc -l
0
$ grep -rn "if (tour_idx >= num_tours) return;" src/algorithms/strategies/local_search/kernels/ | wc -l
1
```

The multi-move candidate bound is now an explicit contract rather than a silent divergence: each thread retains one steepest candidate, so a pass offers at most `min(block_size, n / 2)` moves to the greedy disjoint step (Rocki & Suda, 2013). The CPU reference scans every pair, so the two backends descend in the same direction but not through identical intermediate tours — documented in both `two_opt_multi_move.cu` and `LocalSearchGPUTwoOpt`.

### 7.4 Verification record

- `pytest tests/utils/test_unit_phase_timer.py tests/algorithms/strategies/local_search -q`: **68 passed**.
- `pytest tests -m "not slow" -q`: **1022 passed, 3 failed, 2 deselected** — the three failures are `GEO` cases in `tests/distances/`, whose sources (`src/distances`, `src/loaders`, `pyproject.toml`, `uv.lock`) are unmodified.
- `ruff check src tests`: **All checks passed!**
- `mypy src tests benchmark_scripts`: **Success: no issues found in 137 source files**.
- `python -m benchmark_scripts.check_docs`: **exit 0**.

## 8. Closure of the Stochastic & Topology Batch (`Tasks 11.10`, `11.12`, `11.14`, `11.15`)

The four reopened tasks shared one root cause with two faces: route topology was being inferred from whatever bytes happened to be in the batch, and the host RNG protocol differed between the CPU and GPU siblings. Both were removed rather than patched.

### 8.1 Canonical topology and sampling helpers (`DP11-19`, `DP11-20`)

- `ProblemContext.is_closed` now declares the formulation (`CVRP` → `True`, `TSP`/`ATSP` → `False`), so crossover and mutation never read endpoints from the device.
- `resolve_route_bounds(n, context, route_sample)` resolves the mutable slice in this order: a host sample with identical endpoints `(n > 2)` → `(1, n - 1)`; otherwise `context.is_closed` decides; only a device sample *without* context pays a two-scalar D2H probe; with no information at all the depot-preserving `(1, n - 1)` wins. An open TSP tour therefore reaches `(0, n)` — recovering the $n!$ search space that `BUG-28` had truncated to `(n - 2)!` — while `[0, 1, 0]` stays a no-op.
- `sample_distinct_pairs(rng, num_pairs, low, high)` implements the rejection-free bijection $v = v' + \mathbb{1}\{v' \ge u\}$ with `u ~ U[low, high)` and $v' \sim \mathcal{U}[low, high-1)$, returning an `int32` `(N, 2)` array. Two flat integer draws replace `choice(..., replace=False)`, which is what the CPU and GPU siblings now share.
- `sample_mutation_positions(rng, rows, low, high, candidate_k, context)` is the single host oracle for swap and inversion, so `MutationSwap`/`MutationGPUSwap` and `MutationInversion`/`MutationGPUInversion` cannot drift.

### 8.2 Defect closure

| Defect | Evidence |
| ------ | -------- |
| `BUG-24` | `grep -rn "_is_closed_cache" src/` → 0; `test_gpu_order_crossover_recovers_from_zero_filled_warmup` passes after a `cp.zeros((32, 52))` batch. |
| `BUG-28` | `_mutation_bounds` deleted; open tours get `(0, n)`, depot tours `(1, n - 1)`; `grep "dtype=np.int64"`, `grep "cp.random.randint"` and `grep "rng.choice(np.arange"` in `src/algorithms/strategies/` → 0. |
| `BUG-22` | Bit-identity at $n \in \{52, 100, 200, 500, 1000, 2000\}$ with the shared cut sampler. |

### 8.3 Verification record

- `pytest tests/utils/test_unit_route_bounds_and_pairs.py -q`: **12 passed**.
- `pytest tests/algorithms/strategies/{mutation,crossover,selection} tests/utils/... -q`: **210 passed**.
- `pytest tests -m "not slow" -q`: **1038 passed, 3 failed, 2 deselected** — the three failures are the pre-existing `GEO` cases in `tests/distances/`, untouched here.
- `ruff check src tests benchmark_scripts`: **All checks passed!**
- `mypy src tests benchmark_scripts`: **Success: no issues found in 138 source files**.
- `python -m benchmark_scripts.check_docs`: **exit 0**.

## 9. Batch 3 — Declarative pipeline anchors, telemetry traversal and checkpoint parameters (`TASK 11.6`, `TASK 11.8`, `TASK 11.9`, `BUG-30`)

The three tasks converged on one question: who knows that an extension step belongs here, and how does that knowledge survive into the checkpoint?

### 9.1 The anchor replaces the class-name table

`_compose_pipeline` used to answer the placement question with `type(step).__name__`. That made the core file the single registry of every extension step: a step written outside `genetic_algorithm.py` fell through to `steps.append(step)` regardless of its semantics, and the periodic step's two placements lived inside an `if` that read `getattr(step, "target", "population")` — the algorithm knew the internals of a step it does not own.

Each step now answers the question itself through `stage_anchor`:

| Step | Anchor | Why |
| ---- | ------ | --- |
| `PostInitLocalSearchStep` | `pre_selection` | Refines `state.population` on generation 1, before the population is sampled. |
| `EliteOnlyLocalSearchStep` | `post_survival` | Elite refinement must see the survivors, not the parents. |
| `PeriodicLocalSearchStep(target="offspring")` | `pre_evaluation` | Refined offspring must be the ones scored. |
| `PeriodicLocalSearchStep(target="population")` | `post_survival` | Survivor refinement happens after survival selection. |

`_compose_pipeline` is now a three-branch dispatch on that attribute, with a buffer for `pre_selection` steps so two of them keep their declared order instead of being inserted in reverse. A step that declares nothing lands at the end, which is the old fallback behaviour and no longer a silent bug: the anchor is the documented contract.

### 9.2 Telemetry had a blind spot the size of the whole strategy

`_all_strategies()` listed the six constructor slots only. With `local_search_stage="none"` and `local_search=None` — the documented way to run local search exclusively through a pipeline step — the only `LocalSearchCPUTwoOpt` in the process lived inside `PostInitLocalSearchStep`, so `_reset_run_metrics` reset nothing that ran and `_collect_strategy_metrics` summed zero for H2D, D2H and kernel launches. The benchmark reported a memetic run with no GPU traffic.

The traversal now walks `self.pipeline` through `step.strategies()`, falling back to `_PIPELINE_STRATEGY_ATTRS = ("strategy", "local_search", "evaluation", "re_evaluate")` for third-party steps that expose attributes instead. De-duplication is by `id()` rather than `==`, so a shared evaluator is neither reset nor summed twice.

### 9.3 The checkpoint records what defines the run

`orchestration.py` dumped `[{"class": type(step).__name__} for step in algorithm.pipeline]`, which mixed internal core steps into the configuration and dropped `elite_count`, `interval` and `target` — the three numbers that define the schedule. `_serialize_custom_pipeline` now serializes only `algorithm_instance.custom_pipeline_steps` (the steps the caller declared) and calls `public_params()` on each, producing `{"class": "EliteOnlyLocalSearchStep", "params": {"elite_count": 8}}`.

`configs/algorithms.full.example.json` gained `ExperimentalMemeticPipeline` declaring all three steps with `local_search_stage: "none"`, and `test_example_config_never_combines_pipeline_and_local_search_stage` forbids the combination that reproduced `BUG-21` from the configuration side.

### 9.4 Verification record

| Gate | Command | Result |
| ---- | ------- | ------ |
| Correctness | `pytest tests/algorithms/test_unit_pipeline_steps.py -q` | 25 passed |
| DoD replay | `grep -n "local_search_stage ==" src/algorithms/metaheuristics/genetic_algorithm.py` | 0 |
| DoD replay | `grep -c "class EliteOnlyLocalSearchStep\|class PeriodicLocalSearchStep" src/algorithms/pipeline_steps.py` | 2 |
| Regression | `pytest tests/algorithms tests/benchmarking -q` | 609 passed |
| Types & lint | `mypy src tests benchmark_scripts` / `ruff check src tests benchmark_scripts` | 0 / All checks passed |
| Docs | `python -m benchmark_scripts.check_docs` | exit 0 |

At the time §9 was written, `BUG-30` still had its telemetry half open (Table 4 without `survival`, `ddof=0`, per-generation `_extract_best` D2H, `phase_timings` absent from DuckDB, `EvaluationGPUCost` under-reporting `h2d_bytes`); that half was closed in §10 by `TASK 11.7` and `TASK 11.13`.

## 10. Batch 4 — AOT warm-up, phase telemetry completion and evaluation accounting (`TASK 11.7`, `TASK 11.11`, `TASK 11.13`)

The last three open tasks shared one theme: a measurement was being produced, but a piece of it was either missing from the artifact or paid for inside the measured window.

### 10.1 `Task 11.11` — warm-up is a protocol, not a guess (`DP11-12`, `DP11-21`, `BUG-23`, `BUG-25`)

The old warm-up ran `evolve(max_generations=1)` on the measured instance. That has two independent defects. First, `PeriodicLocalSearchStep(interval=10)` never fires at generation 1, so `two_opt_batch.cu` was compiled at the first *measured* generation 10: repetition 1 cost `1651.73 ms` against a `64.44 ms` median (`26.27x`). Second, the warm-up advanced the RNG and the strategy counters that repetition 1 then observed.

The replacement is a `warmup(context)` hook at three levels:

| Level | Implementation | What it guarantees |
| ----- | -------------- | ------------------ |
| `StrategyBase` | `_get_kernel()` when the strategy has one | Every kernel reaches `_compile_kernel_cached` before measurement |
| GPU strategies | compile **and one launch** on `_warmup_batch(context, rows)`, a `(rows, n)` block of `arange` permutations | The driver's PTX→SASS module load (~230 ms measured) also leaves the measured window; `cp.zeros` is never used (`BUG-24`) |
| Pipeline steps + `GeneticAlgorithm.warmup` | visit every step (including periodic ones) and every strategy, deduplicated by `id()` | Coverage depends on configuration, never on `state.generation` |

The runner deep-copies the instance when `algorithm_factory is None`, warms the clone, and then enforces `_compile_kernel_cached.cache_info().currsize` constancy after the repetitions, raising `RuntimeError` when a kernel slipped into the measured loop.

Measured on `burma14` with `PeriodicLocalSearchStep(interval=10)`, `max_generations=12`, 3 repetitions: `raw_times_ms = [18.4, 18.2, 12.6]`, `rep1/median = 1.014x` (was `26.27x`), `currsize = 5` before and after.

### 10.2 `Task 11.7` — the telemetry half of `BUG-30`

| Residual defect | Closure |
| --------------- | ------- |
| Table 4 omitted `survival` | Added to the phase order in both emitters (`Surv (%)` / `Surv`), so `Sum/Wall %` no longer drops a bucket |
| `np.std(..., ddof=0)` | Sample standard deviation (`ddof=1`), the dispersion `DP11-7` promises |
| `_extract_best` every generation | `_best_cost` reads one scalar per generation; the route crosses PCIe only when `gen_best_cost < best_cost - 1e-6` |
| `incremental_mode` telemetry loss | Absolute totals reconstructed from `mean_h2d_mb`/`mean_d2h_mb`/`mean_kernels` × `successful_runs` and re-averaged over `carried_runs + new_runs` |
| `phase_timings` absent from DuckDB | `mean_phase_timings`, `phase_share_pct` and `phase_share_cv_pct` are `JSON` columns in `schema.sql`, `BenchmarkRunInput` and the `INSERT` (`_json_or_none`) |

### 10.3 `Task 11.13` — the evaluation upload is traffic too

`EvaluationGPUCost.evaluate_batch` counted the distance/coordinate operands but not the `cp.asarray(batch_np)` upload of the `input_on_gpu=False` path. The counter now records `d_population.nbytes`, and the GPU test asserts the *delta* between two consecutive calls so the cached operands cannot mask the result. The `BUG-26` check confirmed no strategy sets `use_gpu_events`, so `@time_strategy` measures `perf_counter` only and `phase_timer.py` needed no edit.

### 10.4 Verification record

| Gate | Command | Result |
| ---- | ------- | ------ |
| Correctness | `pytest tests/benchmarking tests/algorithms -q` | 619 passed |
| Warm-up protocol | `pytest tests/benchmarking/test_unit_runner_warmup.py -q` | 8 passed |
| DoD replay | `grep -rn "def warmup" src/ --exclude-dir=__pycache__` | 10 definitions |
| DoD replay | `grep -n '"survival"' src/benchmarking/analysis/report_generator.py` | lines 481, 548 |
| DoD replay | `grep -n "phase_timings\|phase_share" schema.sql db_operations.py records.py` | 10 (was 0) |
| Types & lint | `mypy src tests benchmark_scripts` / `ruff check src tests benchmark_scripts` | 140 files, 0 issues / All checks passed |
| Docs | `python -m benchmark_scripts.check_docs` | exit 0 |

## 11. Batch 5 — Contract synchronization (`BUG-31`), canonical `GEO` distance patch (`BUG-32`) and full 10-repetition benchmark validation (`results_v3/selected_10`)

The multi-axis quality audit of Phase 11 uncovered two contract defects and one benchmarking execution detail:

1. **`BUG-31` (`tests/data_models/test_unit_records.py`)**: `Task 11.7` added `mean_phase_timings`, `phase_share_pct` and `phase_share_cv_pct` as `NotRequired` fields to `BenchmarkRunInput` in [records.py](../../src/data_models/records.py#L116) and `schema.sql`, but the contract unit test [test_unit_records.py](../../tests/data_models/test_unit_records.py#L54) still pinned `len(OPTIONAL_FIELDS) == 13` and `len(FULL_PAYLOAD) == 40`, failing with `AssertionError`.
2. **`BUG-32` (`src/distances/pairwise.py`)**: In Phase 10 (commit `a20bffb`), `pairwise.compute_geographical` was replaced with an import from `routing_data.converter.utils.cost.geographical`. However, upstream `routing_data` v0.2.0 rounds the integer degree component rather than truncating DDD.MM coordinates as mandated by the TSPLIB95 specification (Reinelt 1995), corrupting distances whenever minutes $\ge 30$ (e.g., `burma14` d(1,3) reading 560 instead of the canonical 510). This broke three unit tests in `tests/distances/`.
3. **Execution of `run_selected_problems.py`**: Checkpoints in `results_v3/selected_10/checkpoints/` from prior runs had `successful_runs = 3` and lacked `mean_phase_timings`. In incremental mode, `algorithm_runner.py` skipped execution, causing `report_generator.py` to omit Table 4. Cleaning the directory and setting `REPETITIONS = 10` regenerated all 39 checkpoints with complete phase telemetry.

### 11.1 `Task 11.18` — Telemetry contract synchronization (`BUG-31`)

In [test_unit_records.py](../../tests/data_models/test_unit_records.py#L54-L190):

- Added `"mean_phase_timings"`, `"phase_share_pct"`, `"phase_share_cv_pct"` to `OPTIONAL_FIELDS` (expanding optional columns from 13 to 16).
- Included sample values in `FULL_PAYLOAD` (expanding declared payload keys from 40 to 43).
- Updated assertions `assert len(BenchmarkRunInput.__optional_keys__) == 16` and `assert len(FULL_PAYLOAD) == 43`.

All 15 tests in `tests/data_models/test_unit_records.py` now pass.

### 11.2 Canonical `GEO` distance patch (`BUG-32`)

To restore compliance with TSPLIB95 without waiting for an upstream `routing_data` release:

- Re-implemented `_to_radians_geo(coord_val: float) -> float` in [pairwise.py](../../src/distances/pairwise.py#L66):
  $$\text{deg} = \lfloor x \rfloor, \quad \text{min} = x - \text{deg}, \quad \text{rad} = \frac{\pi \cdot (\text{deg} + \frac{5}{3}\text{min})}{180}$$
  Truncating the degree part ensures that coordinates like `52.31` (52° 18.6') are correctly converted to radians instead of being rounded to 53°.
- Re-implemented `compute_geographical` with an explicit patch docstring noting that it overrides `converter.utils.cost.geographical`.
- Verified parity with the GPU kernel in [pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh#L105), which already implements the correct truncation (`(int)coord_val`).
- Updated [test_unit_pairwise_geographical.py](../../tests/distances/test_unit_pairwise_geographical.py#L23) to assert that `pairwise.compute_geographical` overrides upstream and validates all reference angles.

All 197 distance unit tests in `tests/distances/` now pass.

### 11.3 10-Repetition cohort execution and Table 4 closure

Executing [run_selected_problems.py](../../benchmark_scripts/experiments-and-examples/chapter4/run_selected_problems.py) with `REPETITIONS = 10` populated all checkpoints with `raw_phase_timings`, `mean_phase_timings`, `phase_share_pct`, and `phase_share_cv_pct` (`ddof=1`), closing Table 4 in Markdown and LaTeX.

### 11.4 Verification record

| Gate | Command | Result |
| ---- | ------- | ------ |
| Records contract (`BUG-31`) | `pytest tests/data_models/test_unit_records.py -q` | 15 passed |
| GEO distance patch (`BUG-32`) | `pytest tests/distances -q` | 197 passed |
| Full test suite | `pytest tests/ -q` | 1067 passed, 0 failed |
| Documentation check | `python -m benchmark_scripts.check_docs` | exit 0 (0 stale, 0 dead links, 0 undefined symbols) |
