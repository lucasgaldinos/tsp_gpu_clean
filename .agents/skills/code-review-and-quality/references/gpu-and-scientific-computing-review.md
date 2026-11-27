---
title: "GPU and Scientific-Computing Review Patterns"
description: >
  Review checklist for accelerated and benchmarked code: tracing hidden synchronization
  through decorators and wrappers, conserving CPU/GPU RNG bit-identity, auditing device
  residency and host scalar syncs on hot paths, sweeping sibling CUDA kernels for bounds
  and reduction defects, and verifying telemetry conservation end to end.
created: 2026-11-10
modifications:
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Created from the defect classes that survived Phase 11 review: decorator-level
          synchronization, unilateral RNG sampling changes, uncached host/device scalars,
          unverified DoD greps, and orphaned telemetry keys.
  - date_modified: 2026-11-26
    modifications:
      - description: >
          Added sections 6-8 from Phase 17 CVRP post-mortem: warmup() purity, CUDA warp
          SIMD utilization and fair vectorized CPU baselines, and partial-feasibility
          statistical invariance + doc/schema contract parity.
status: review
author:
  - "Lucas Galdino"
  - "GitHub Copilot"
type: guide
scope: global
related_files:
  - "[SKILL.md](../SKILL.md)"
  - "[guide-gpu-cpu-comparison-methodology.md](../../../../documentation/knowledge-base/guide-gpu-cpu-comparison-methodology.md)"
  - "[analysis-gpu-2opt-parallel-architectures.md](../../../../documentation/knowledge-base/analysis-gpu-2opt-parallel-architectures.md)"
tags:
  - review/quality
  - review/correctness
  - analysis/gpu
  - analysis/profiling
  - benchmark/gpu
  - benchmark/speedup
  - algorithm/2opt
  - notes/cuda
  - guide/benchmarking
  - notes/methodology
  - guide/testing
---

# GPU and Scientific-Computing Review Patterns

Five defect classes that pass a normal five-axis review and still invalidate a GPU
result. Each section states the claim reviewers accepted, the inspection that falsifies
it, and the command to attach.

## 1. Decorators and wrappers hide synchronization

**What got missed:** `grep -rn "Stream.null.synchronize()" src/algorithms/strategies/`
returned 0, so the async-GPU task was closed. Meanwhile `@time_strategy` /
`PhaseTimer.measure` called `end_evt.synchronize()` on *every* strategy call whenever
`use_cuda_events=True`, so the hot path never ran asynchronously.

**Inspection:** build the call path from public entry point to kernel and name every
blocking primitive on it (`synchronize()`, `.get()`, `.item()`, `cudaMemcpy` D2H).

```python
# Blocking primitives are not always in the module under review.
decorators = [d for d in dir(LocalSearchGPUTwoOpt.improve_batch) if "time" in d.lower()]
print(decorators)  # @time_strategy attaches here, not in two_opt.py
```

**Falsifiable check:** `rg -n "synchronize\(|\.get\(\)|\.item\(\)" src/utils/phase_timer.py`
plus a call-path note in the review. Zero matches inside a strategy module is not zero
on the call path.

## 2. Cross-backend RNG stream and bit-identity

**What got missed:** vectorizing `rng.choice(n, size=2, replace=False)` into
`rng.integers` only inside `CrossoverGPUOrder` sped up the GPU host loop 72x but broke
`OX-A` bit-identity with `CrossoverOrder` and left the CPU baseline penalized.

**Inspection:** any edit to sampling order, sample count, `dtype` (`int32` vs `int64`
changes `PCG64` stream consumption), or replacement semantics must land in one shared
helper imported by both classes.

```python
def sample_cut_pair(rng: np.random.Generator, n: int) -> tuple[int, int]:
  """Draw two distinct cut points from one vectorized call, shared by both backends."""
  first, second = rng.integers(0, n, size=2, dtype=np.int32)
  return (int(first), int(second)) if first < second else (int(second), int(first))
```

**Falsifiable check:** a parity test asserting `np.array_equal(cpu_out, cp.asnumpy(gpu_out))`
for one seed and the same parents, run in both directions (`OX-A`, `SWAP-A`, `INV-A`).

## 3. Device residency and host scalar syncs

**What got missed:** `.get()` was removed from `order_crossover.py`, but `two_opt.py`
kept `int(tour.max())` on a `cp.ndarray` and re-uploaded `cp.asarray(candidate_table)`
inside `_improve_grid` on every call. `ProblemContext.get_gpu_candidate_list` shipped
with zero callers.

**Inspection:** sweep every helper reachable from a GPU hot path for implicit
device→host conversions, then check residency with byte counters over a multi-generation
loop.

```bash
rg -n "int\([a-z_]+\.max\(\)\)|bool\([a-z_]+\[0\] ==|np\.asarray\([a-z_]+\)|\.item\(\)" \
  src/algorithms/strategies
rg -n "get_gpu_candidate_list" src  # 1 match = definition only = dead API
```

**Falsifiable check:** log `strategy.h2d_bytes` / `d2h_bytes` with `$G = 1$` and `$G = 5$`
generations; the delta per generation must be identical, and `h2d` must not grow with $G$.

## 4. Falsifiable DoD and whole-directory kernel sweep

**What got missed:** the DoD claimed
`grep -rn "while (left < right)" src/algorithms/strategies/local_search/kernels/ → 0`
while `two_opt_multi_move.cu` still had two occurrences. In the same directory a tree
reduction ended at `s > 32` and then finished with a full linear scan, and grid bounds
guards (`if (tour_idx >= num_tours) return;`) were absent from one sibling kernel.

**Inspection:** apply this checklist to **every** `.cu` in the directory, not only the
one named in the task.

| Check | Question |
| --- | --- |
| Grid bounds | Does each thread guard `idx >= num_tours` / `lane >= n` before the first load? |
| Reduction tail | After the `s > 32` tree phase, is the tail reduced by warps only, or does every thread run the full `1..block_size-1` scan? |
| Shared memory | Is `sizeof` under the 48 KB block limit for the configured `TILE_SIZE`, and is occupancy recomputed when the tile grows? |
| Divergence | Do lanes of one warp take identical branches on the same scalar (warp-per-route) or diverge? |
| Numerical parity | Does the kernel reproduce `nint = (int)floor(x + 0.5)` and the same rounding as the CPU reference? |

**Falsifiable check:** re-run the exact DoD command verbatim, including
`--exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}`, and paste the output.
Prefer structural assertions (kernel registry size, LRU compile-cache counters, AST
checks on the kernel source) over wall-clock thresholds in tests — timing assertions are
flaky and prove nothing about the defect.

## 5. Telemetry conservation end to end

**What got missed:** `host_sync` was added to `PhaseTimer` and `params.pipeline` to the
config, but the key stopped at the runner: the checkpoint validator did not require it,
the schema had no column, the importer dropped it, and the report table omitted
`survival` from `Sum/Wall %`.

**Inspection:** trace every new key through the five layers before approving.

| Layer | File | What to confirm |
| --- | --- | --- |
| Producer | `phase_timer.py`, `genetic_algorithm.py` | Key is written on every run, including early stopping |
| Aggregator | `algorithm_runner.py` | `mean`, `std` with `ddof=1`, CV over repetitions |
| Validator | `checkpoint_io.py` | Length equals `successful_runs`; missing key rejected or tolerated explicitly |
| Storage | `schema.sql`, `db_operations.py` | Column exists and the importer binds it |
| Report | `report_generator.py` | Column appears in Markdown and LaTeX; `Sum/Wall %` includes every phase |

**Falsifiable check:**

```bash
for f in phase_timer.py algorithm_runner.py checkpoint_io.py schema.sql \
  db_operations.py report_generator.py; do
  printf '%s: ' "$f"; rg -c "host_sync" "$f" || echo 0
done
```

A key present in four of five layers is a defect, not a follow-up.

## 6. `warmup()` counter, timer, and RNG purity

**What got missed:** New VRP strategy `warmup(context)` methods invoked the live
`@time_strategy`-decorated `evaluate_batch`/`improve_batch` directly on `self` and
`context`, mutating `h2d_bytes`, `kernel_launches`, `@time_strategy` buckets, and
advancing `context.rng` before repetition 1 even began.

**Inspection:** Every `warmup(context)` must compile/launch kernels via internal
un-decorated helpers using a private scratch RNG (`np.random.default_rng(0)`), and leave
`self.h2d_bytes`, `self.d2h_bytes`, `self.kernel_launches`, `context.h2d_bytes` (or
documented cache-only delta), and `context.rng.bit_generator.state` unchanged.

**Falsifiable check:** Snapshot `before = context.rng.bit_generator.state` and strategy
counters before `strategy.warmup(context)`; assert identical RNG state and `0` leaked
calls/timings afterward.

## 7. CUDA warp SIMD utilization & fair vectorized CPU baselines

**What got missed:**

1. A CUDA Split DP kernel launched 32-thread blocks while executing the entire DP loop
   inside `if (tid == 0)` (`3.1%` SIMD lane utilization, `96.9%` idle lanes) and stored
   per-tour buffers with stride-`width` uncoalesced global loads.
2. New CPU strategies (`EvaluationCPUCVRPCost`, `LocalSearchCPUCVRP`) iterated via
   pure-python `.tolist()` loops while existing TSP CPU baselines (`two_opt.py`) were
   NumPy-vectorized—artificially inflating reported GPU speedup ratios.

**Inspection:**

- In every `.cu` kernel, flag any inner `O(n)` or `O(n*B)` loop guarded by `if (tid == 0)`
  that could either map 1 thread = 1 independent tour (for small `n`) or parallelize the
  inner scan/reduction across warp lanes (`__shfl_down_sync`). Check global memory stride
  across consecutive lanes (`tid` should stride contiguous elements when possible).
- In every CPU sibling of a GPU strategy, verify the hot loop uses vectorized NumPy (or
  documented algorithmic parity, e.g. compiled/vectorized distance lookups) rather than
  nested `.tolist()` python loops, or explicitly flag it as a non-fair baseline.

**Falsifiable check:** `rg -n "\.tolist\(\)" src/algorithms/strategies/` on hot paths, plus
auditing `if (tid == 0)` block lengths in `src/algorithms/strategies/**/kernels/*.cu`.

## 8. Statistical validity under partial feasibility & config/doc contract honesty

**What got missed:**

1. Mapping infeasible runs to a jump surrogate $C^{\max} + 1 + \nu$ broke scale-invariance
   of cardinal-difference tests (`wilcoxon`, rank-biserial $r$, Hodges-Lehmann CI)—changing
   the arbitrary `+ 1.0` offset altered `p_value`—while `not all(raw_is_feasible)` collapsed
   `29/30` feasible runs below a `30/30` random baseline and summed incompatible units
   ($\nu_Q + \nu_L + \nu_K$).
2. Docs advertised `"Tier-1"` jargon (colliding with existing statistical Tiers) and
   `problems.json` override keys that the loader never wired.

**Inspection:**

- **Ordinal vs. Cardinal Separation:** Verify rank-based tests (Friedman/Nemenyi) vs.
  magnitude-sensitive paired tests (Wilcoxon/Hodges-Lehmann) are invariant to any positive
  rescaling of penalty offsets and normalize multi-unit violations ($\nu_Q/Q + \nu_L/L_{\max} + \nu_K/K_{\max}$).
- **Partial-Feasibility Test Case:** Require at least one unit test with mixed feasibility
  (`0% < feasibility_rate < 100%`, e.g., `9/10` feasible) and a perturbation test asserting
  changing an arbitrary jump constant does not change test p-values.
- **Doc & Schema Contract Parity:** Every JSON key or CLI flag mentioned in `documentation/`
  must be grepped in `src/` and validated against `configs/schemas/*.schema.json`.

## Related references

- [guide-gpu-cpu-comparison-methodology.md](../../../../documentation/knowledge-base/guide-gpu-cpu-comparison-methodology.md) — warm-up vs AOT precompilation, host serialization bottlenecks, timing boundaries.
- [analysis-gpu-2opt-parallel-architectures.md](../../../../documentation/knowledge-base/analysis-gpu-2opt-parallel-architectures.md) — thread mapping, wave serialization, shared-memory budget on GP107.
