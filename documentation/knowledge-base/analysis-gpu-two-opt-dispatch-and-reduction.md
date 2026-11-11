---
title: "GPU 2-Opt Dispatch Defects: Candidate-List Bypass, `.max()` D2H Stalls, and Multi-Move Compression"
description: >
  Audit of `LocalSearchGPUTwoOpt._improve_grid` and the three batched 2-opt CUDA
  kernels: the candidate-list path that bypasses `ProblemContext.get_gpu_candidate_list`
  and re-uploads `candidate_table` on every call, the `int(tour.max())` reduction that
  forces a synchronous D2H stall at the start of each generation, the `s_tour` shared
  memory region that stays allocated but unused in candidate mode, the serial
  `while (left < right)` reversals and one-candidate-per-thread retention that make
  `two_opt_multi_move.cu` diverge from the exhaustive CPU `multi_move` reference, the
  missing `tour_idx >= num_tours` guard, and the redundant tree-plus-linear reduction
  in `two_opt_batch.cu`/`two_opt_single.cu` (`BUG-27`, `BUG-29`, `DP10-20`, `2OPT-A`).
created: 2026-11-10
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[GitHub Copilot]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Initial creation: documented the candidate-list bypass and its measurable
          `h2d_bytes` growth, the `.max()` D2H synchronization stall, the dead `s_tour`
          shared-memory region in candidate mode, the residual serial reversals and
          per-thread single-candidate retention in `two_opt_multi_move.cu`, the missing
          out-of-range block guard in `two_opt_batch.cu`, and the O(block_size) serial
          remainder of the block reduction.
      - description: >
          Revision: added the verified source-anchor table (defect → `file:line`) and
          documented why the redundant reduction is doubly wasteful — it pays
          $\log_2(\text{block\_size}) - 3$ tree barriers and then re-derives the same
          minimum serially, and it pays them even on converged passes where
          `s_deltas[0] >= 0` means no move will be applied.
related_files:
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu)"
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[two_opt_common.cuh](../../src/distances/kernels/two_opt_common.cuh)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[analysis-gpu-2opt-parallel-architectures.md](./analysis-gpu-2opt-parallel-architectures.md)"
  - "[analysis-gpu-2opt-reduction-strategies.md](./analysis-gpu-2opt-reduction-strategies.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[analysis-timing-instrumentation-and-warmup.md](./analysis-timing-instrumentation-and-warmup.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
tags:
  - analysis/performance
  - analysis/architecture
  - algorithm/two-opt
  - benchmark/gpu
  - guide/cuda
  - notes/cuda
  - notes/hardware
  - analysis/profiling
  - review/quality
  - algorithm/genetic-algorithm
---

# GPU 2-Opt Dispatch Defects: Candidate-List Bypass, `.max()` D2H Stalls, and Multi-Move Compression

- [GPU 2-Opt Dispatch Defects: Candidate-List Bypass, `.max()` D2H Stalls, and Multi-Move Compression](#gpu-2-opt-dispatch-defects-candidate-list-bypass-max-d2h-stalls-and-multi-move-compression)
  + [1. Executive Summary](#1-executive-summary)
  + [2. Anatomy of `_improve_grid`](#2-anatomy-of-_improve_grid)
  + [3. Candidate-List Bypass and Repeated `h2d_bytes`](#3-candidate-list-bypass-and-repeated-h2d_bytes)
  + [4. The `.max()` D2H Synchronization Stall](#4-the-max-d2h-synchronization-stall)
  + [5. Dead `s_tour` Region in Candidate Mode](#5-dead-s_tour-region-in-candidate-mode)
  + [6. `two_opt_multi_move.cu`: Serial Reversals and One Candidate per Thread](#6-two_opt_multi_movecu-serial-reversals-and-one-candidate-per-thread)
    - [6.1 The Two `while (left < right)` Loops](#61-the-two-while-left--right-loops)
    - [6.2 Per-Thread Single Retention (`s_deltas[tid]`)](#62-per-thread-single-retention-s_deltastid)
  + [7. Missing Out-of-Range Block Guard](#7-missing-out-of-range-block-guard)
  + [8. Redundant Tree-Plus-Linear Reduction](#8-redundant-tree-plus-linear-reduction)
  + [9. Verification Recipe](#9-verification-recipe)
  + [10. References](#10-references)

---

## 1. Executive Summary

`LocalSearchGPUTwoOpt` is the hottest GPU strategy in `FullGPUModular`, and its host
wrapper accumulated five independent defects that all present as "the GPU is slow",
none of which is a hardware limit:

| # | Defect | Identifier | Consequence |
| :- | :--- | :--- | :--- |
| 1 | Candidate table resolved on the host and re-uploaded every call | `BUG-27` | `h2d_bytes` grows per generation; $8{,}000\ \text{B} \times$ generations wasted |
| 2 | `int(tour.max())` on a device array | `BUG-27` | Full device reduction + D2H copy + host block, once per generation |
| 3 | `s_tour` shared region allocated but never filled in candidate mode | `BUG-29` | Up to $n \times 4$ bytes of dead shared memory per block |
| 4 | Serial `while (left < right)` reversals in the multi-move kernel | `BUG-29` | `2OPT-A` applied to 2 of 3 kernels only |
| 5 | `O(\text{block\_size})` serial remainder of the block reduction | `BUG-29` | `255` serial iterations on thread 0 per 2-opt pass |

Defect 5 is the one that most often gets misattributed to "kernel launch latency": a
$O(\text{block\_size})$ single-threaded tail inside the kernel dominates any launch
overhead by an order of magnitude on small tours.

Verified source anchors (line numbers current as of the Phase 11 audit):

| Symbol | Anchor |
| :--- | :--- |
| `LocalSearchGPUTwoOpt._resolve_candidates` | [`two_opt.py:145`](../../src/algorithms/strategies/local_search/two_opt.py#L145) |
| `LocalSearchGPUTwoOpt._improve_grid` | [`two_opt.py:482`](../../src/algorithms/strategies/local_search/two_opt.py#L482) |
| `ProblemContext.get_gpu_candidate_list` | [`problem_context.py:415`](../../src/protocols/problem_context.py#L415) |
| Tree reduction with `s > 32` | [`two_opt_batch.cu:225`](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L225), [`two_opt_single.cu:100`](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L100) |
| Serial reversals surviving `2OPT-A` | [`two_opt_multi_move.cu:260`](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L260), [`...:271`](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L271) |

---

## 2. Anatomy of `_improve_grid`

```python
# src/algorithms/strategies/local_search/two_opt.py — LocalSearchGPUTwoOpt._improve_grid (abridged)
cp = require_gpu()
batch, was_1d = ensure_routes_batch(routes)
batch_size, n = batch.shape

input_on_gpu: bool = is_gpu_array(batch)
if input_on_gpu:
  d_population = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
else:
  batch_np = np.asarray(batch, dtype=np.int32)
  d_population = cp.asarray(batch_np, dtype=cp.int32)
  self.h2d_bytes += int(batch_np.nbytes)

(d_distances, d_coords, edge_type_code), uploaded_bytes = self._operands.resolve(
  context, n, self.max_matrix_dimension
)
self.h2d_bytes += uploaded_bytes

candidate_table = _resolve_candidates(context, batch, self.candidate_k, require_dense_ids=True)
candidate_count: int = int(candidate_table.shape[1]) if candidate_table is not None else 0
if candidate_count > 0 and candidate_table is not None:
  d_candidates = cp.ascontiguousarray(cp.asarray(candidate_table, dtype=cp.int32))
  d_positions = cp.empty((batch_size, n), dtype=cp.int32)
  self.h2d_bytes += int(candidate_table.nbytes)
```

The distance operands are cached in `self._operands` and only re-uploaded when the
problem identity changes (`DP10-17`). The candidate table has **no such cache**, which
is the asymmetry that defects 1 and 2 exploit.

---

## 3. Candidate-List Bypass and Repeated `h2d_bytes`

`ProblemContext` exposes both a CPU and a GPU candidate list specifically so a device
strategy never pays the transfer twice:

```python
get_cpu_candidate_list(k)   # -> np.ndarray (n, k_eff) int32
get_gpu_candidate_list(k)   # -> cp.ndarray (n, k_eff) int32, cached in the context
```

`_improve_grid` calls neither. It routes through `_resolve_candidates`, which calls
`context.get_cpu_candidate_list(...)` and then re-uploads:

```python
d_candidates = cp.ascontiguousarray(cp.asarray(candidate_table, dtype=cp.int32))
self.h2d_bytes += int(candidate_table.nbytes)
```

Because `improve_batch` is invoked once per generation, the table crosses PCIe once per
generation, forever, even though it is a static property of the problem. The counter
makes the leak observable without a profiler:

$$
\text{h2d\_bytes}(g) \;=\; \underbrace{48{,}000}_{\text{one-time}} \;+\; \underbrace{4 \cdot 8{,}000}_{\text{4 extra generations}} \;=\; 80{,}000\ \text{B}
\quad (n = 100,\ k = 20,\ 5\ \text{generations}).
$$

The table itself is $n \cdot k \cdot 4 = 100 \cdot 20 \cdot 4 = 8{,}000\ \text{B}$, so the excess is exactly four redundant uploads. On $n = 1{,}002$ with $k = 20$ the excess is $\approx 80\ \text{KB}$ per generation — small in absolute terms, but it serializes the host at the start of every local-search phase and it is pure waste.

> [!IMPORTANT]
> The fix is not "cache the table in the strategy". The context already owns a device
> cache (`get_gpu_candidate_list`) precisely so that `EvaluationGPUCost` and
> `LocalSearchGPUTwoOpt` share **one** VRAM copy. Caching a second copy inside the
> strategy reintroduces the duplication `DP10-17` removed for distance matrices.

---

## 4. The `.max()` D2H Synchronization Stall

```python
# src/algorithms/strategies/local_search/two_opt.py — _resolve_candidates
candidates = context.get_cpu_candidate_list(candidate_k)
largest_node = int(tour.max()) if tour.size else -1
if largest_node >= candidates.shape[0]:
  return None
if require_dense_ids and largest_node >= int(tour.shape[-1]):
  return None
```

`tour` here is the **unconverted** argument, so in the GPU path it is a
`cp.ndarray`. `tour.max()` returns a 0-d device array; `int(...)` on it performs an
implicit `.get()`, i.e.

1. enqueue a full $O(N \cdot n)$ reduction over the population,
2. copy the scalar to the host,
3. **block the host until the device drains**, and
4. do so *before* any kernel is launched for this generation.

This is a second hidden barrier per generation, independent of `_extract_best`, and it
is invisible in `phase_timings` because it happens inside `improve_batch` before the
first `kernel(...)` call — so it is attributed to `local_search` as "GPU time" when it
is host blocking. Worse, it defeats the entire purpose of the candidate path: the
candidate mode exists to avoid $O(n^2)$ scans, and it opens with an $O(N \cdot n)$
device reduction plus a device-wide sync.

Two clean resolutions:

- **Preferred (`DP11-19`/`DP10-20`)**: derive `largest_node` from problem metadata
  (`problem.dimension - 1`) or from `context.is_closed`/`problem_type`, never from the
  live batch. A dense TSP tour always satisfies `largest_node == n - 1`.
- **Fallback**: hoist the check out of the per-generation path by validating the route
  width once when the problem is bound, not when the strategy runs.

Also note the type annotation is `tour: np.ndarray` while the call site passes a CuPy
array — a typing hole that let the D2H copy survive review.

---

## 5. Dead `s_tour` Region in Candidate Mode

```python
# src/algorithms/strategies/local_search/kernels/two_opt_batch.cu
extern __shared__ int shared_mem[];
const bool use_candidates = candidate_k > 0 && candidates != nullptr && positions != nullptr;
int *s_tour = shared_mem;
int *s_deltas = tiled ? (int*)&s_tour[tour_tile_size + 1] : (int*)&s_tour[n];
```

```python
# src/algorithms/strategies/local_search/two_opt.py — _shared_mem_bytes
tour_ints = tile_size + 1 if tile_size > 0 else n
return (block_size * 3 + tour_ints) * 4
```

The `s_tour` region is carved out unconditionally, but the candidate branch never fills
it (`use_candidates` is checked before the window load) and `TOUR_AT` falls through to
global memory. On $n = 1{,}002$ that is $4{,}008\ \text{B}$ of dead shared memory per
block; on $n = 11{,}520$ the full-tour path is already at

$$
(256 \cdot 3 + 11{,}520) \cdot 4 = 46{,}080\ \text{B} \approx 45\ \text{KB},
$$

i.e. within $3\ \text{KB}$ of the $48\ \text{KB}$ per-block cap of the GP107. Wasted
bytes there are the difference between one and two resident blocks per SM, so the dead
region is not merely cosmetic. When `candidate_k > 0`, `_shared_mem_bytes` should omit
`tour_ints` entirely and the kernel should not advance `s_deltas` past it.

---

## 6. `two_opt_multi_move.cu`: Serial Reversals and One Candidate per Thread

### 6.1 The Two `while (left < right)` Loops

```cpp
// two_opt_multi_move.cu — inside `if (tid == 0)`
int left = swap_i + 1;
int right = swap_j;
if (use_candidates) {
  while (left < right) { int tmp = tour[left]; tour[left] = tour[right]; tour[right] = tmp; left++; right--; }
} else {
  while (left < right) { int tmp = TOUR_AT(left); TOUR_SET(left, TOUR_AT(right)); TOUR_SET(right, tmp); left++; right--; }
}
```

Both loops execute on a single thread while the other $255$ threads wait at the next
`__syncthreads()`. This is the exact defect `DP11-18` (`2OPT-A`) removed from
`two_opt_batch.cu` and `two_opt_single.cu`, which now reverse cooperatively:

```cpp
// two_opt_batch.cu — after DP11-18
if (use_candidates) {
  for (int k = tid; k < num_swaps; k += block_size) {
    int temp = tour[left + k]; tour[left + k] = tour[right - k]; tour[right - k] = temp;
  }
} else {
  for (int k = tid; k < num_swaps; k += block_size) {
    int temp = TOUR_AT(left + k); TOUR_SET(left + k, TOUR_AT(right - k)); TOUR_SET(right - k, temp);
  }
}
```

`2OPT-A` was applied to **two of three** kernels. The mirror pairs
$(left + k,\ right - k)$ are disjoint for $k \in [0, \lfloor L/2 \rfloor)$, so the
cooperative form is race-free and produces a bit-identical route; leaving the
multi-move kernel serial is a consistency defect, not a design choice.

### 6.2 Per-Thread Single Retention (`s_deltas[tid]`)

`two_opt_consider(..., s_deltas, s_swap_i, s_swap_j, tid)` stores at most **one**
candidate per thread — its best delta so far. The kernel can therefore apply at most

$$
\text{max\_moves} = \min(\text{block\_size},\ \lfloor n/2 \rfloor)
$$

disjoint moves per pass, and any thread that discovered three improving moves has
silently discarded two of them. The CPU reference does not do this:

```python
# LocalSearchCPUMultiMoveTwoOpt: every mutually disjoint improving move per pass
```

Consequence: the GPU and CPU implementations of `mode="multi_move"` converge along
**different trajectories**, and the divergence is not a rounding artifact — it was
measured on $15$ of $16$ routes. This breaks the iso-algorithmic contract that
`LocalSearchGPUTwoOpt(mode=...)` inherits from `Design B` (`DP11-1`), where the mode
parameter is supposed to select an acceptance *rule*, not a different search.

Two honest options, and only two:

1. **Restore parity**: give each thread a small per-thread candidate array (or rescan
   after applying a move) so the block collects every improving move found in the pass.
2. **Rename the mode**: call the GPU variant what it actually is — a *bounded* multi-move
   (at most one move per thread per pass) — and stop claiming bit-parity with
   `CPUMultiMoveTwoOpt` in the tests and in Chapter 4.

Shipping option 2's behaviour under option 1's name is the defect.

---

## 7. Missing Out-of-Range Block Guard

```cpp
// two_opt_batch.cu:58 — block 0 only; nothing above num_tours is rejected
int tour_idx = blockIdx.x;  // Which tour this block processes
int tid = threadIdx.x;
int block_size = blockDim.x;
```

There is no `if (tour_idx >= num_tours) return;`. Today the host always launches exactly
`blocks = batch_size`, so the guard is latent. It becomes an active out-of-bounds write
the moment any caller oversubscribes the grid — a common technique to fill the SM array
when `batch_size` is not a multiple of the resident-block count, and a natural future
optimization for small populations ($N = 32$ on 5 SMs already leaves the device mostly
idle). The guard costs a single predicate and one early exit; the alternative is
corrupting the next tour in the flattened `num_tours * n` buffer.

---

## 8. Redundant Tree-Plus-Linear Reduction

```cpp
// two_opt_batch.cu:225-243 and two_opt_single.cu:100-124
for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
  if (tid < s) {
    if (tid + s < block_size && s_deltas[tid + s] < s_deltas[tid]) {
      s_deltas[tid] = s_deltas[tid + s];
      s_swap_i[tid] = s_swap_i[tid + s];
      s_swap_j[tid] = s_swap_j[tid + s];
    }
  }
  __syncthreads();
}

if (tid == 0) {
  for (int i = 1; i < block_size; i++) {   // <-- full linear scan
    if (s_deltas[i] < s_deltas[0]) { ... }
  }
}
__syncthreads();
```

The waste is twofold. First, the tree already performed $\log_2(\text{block\_size}) - 3$
barriers (for $\text{block\_size} = 256$: $128, 64 \to$ **2** rounds, i.e. 2
`__syncthreads()`), and the serial scan then re-derives the same minimum by hand,
discarding the parallel work it just paid for. Second, the barriers are paid even when
`s_deltas[0] >= 0` — no improvement exists and the block is about to `continue` without
applying anything (`BUG-29`). Guarding the reduction with a cheap short-circuit
(`if (s_deltas[0] >= 0) { __syncthreads(); continue; }` evaluated after the warp-shuffle
aggregate) skips the remaining barriers on every converged pass, which is the majority of
passes once the tour stabilises.

The loop condition `s > 32` stops the tree reduction when the stride reaches `32`,
leaving **33** live values at indices `0..32` (not 32). The serial tail then scans
`1 .. block_size - 1`, i.e. it re-reads

- the 32 already-reduced lanes, and
- every lane above index 32, **which the tree has already folded into them**.

For `block_size = 256` that is $255$ serial iterations on thread 0, each with three
comparisons and three shared-memory reads, while $255$ threads idle. This is the
identical pattern that `EVAL-A` (`DP11-15`) removed from `cost_calculator.cu` by
finishing with a warp shuffle:

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
  int other = __shfl_down_sync(0xFFFFFFFFu, partial_cost, offset);
  partial_cost = other < partial_cost ? other : partial_cost;
}
```

The symmetric fix here is a `s_deltas`/`s_swap_i`/`s_swap_j` warp-shuffle epilogue for
the remaining 32 lanes. Correctness is unchanged — the reduction is associative and
commutative over a total order of deltas — so the resulting route is bit-identical and
the change is free to verify.

> [!NOTE]
> The `if (tid + s < block_size && ...)` inner guard exists because
> `next_power_of_2(min(threads_per_block, n - 2))` can produce non-power-of-two block
> sizes for small tours; it is correct but it also means the tree does *not* have the
> clean power-of-two invariant the shuffle epilogue assumes. The epilogue must therefore
> be preceded by an explicit padding of `s_deltas[block_size .. 31]` with `0`
> (no improvement), which is exactly the "orphaned thread" failure mode analyzed in
> [analysis-non-power-of-two-blocks.md](./analysis-non-power-of-two-blocks.md).

---

## 9. Verification Recipe

```bash
# 1. Candidate table must come from the context's device cache
grep -n "get_gpu_candidate_list" src/algorithms/strategies/local_search/two_opt.py   # expect >= 1
grep -n "cp.ascontiguousarray(cp.asarray(candidate_table" src/                        # expect 0

# 2. No host-side reduction over a live device batch
grep -n "int(tour.max())" src/algorithms/strategies/local_search/two_opt.py           # expect 0

# 3. No serial reversal left in any 2-opt kernel
grep -rn "while (left < right)" src/algorithms/strategies/local_search/kernels/       # expect 0

# 4. Every batched kernel guards its block index
grep -n "tour_idx >= num_tours" src/algorithms/strategies/local_search/kernels/two_opt_batch.cu
```

Expected: step 1 finds the context accessor and no local re-upload; steps 2–3 return
nothing; step 4 returns the guard.

---

## 10. References

1. Croes, G. A. (1958). A method for solving traveling-salesman problems. *Operations Research*, 6(6), 791–812.
2. Rocki, K., & Suda, R. (2012). Accelerating 2-opt and 3-opt local search using GPU in the travelling salesman problem. *IEEE HPCS*, 489–495.
3. Rocki, K., & Suda, R. (2013). An efficient GPU implementation of a multi-start TSP solver for large problem instances. *IEEE HPCS*.
4. O'Neil, M. A., & Burtscher, M. (2015). Rethinking the parallelization of random-restart hill climbing: a case study in optimizing a 2-opt TSP solver for GPU execution. *GPGPU-8*, 99–108.
5. Fujimoto, N., & Tsutsui, S. (2011). A highly-parallel TSP solver for a GPU computing platform. *ICNAAM*, 1389, 264–267.
6. Reinelt, G. (1991). TSPLIB — A Traveling Salesman Problem Library. *ORSA Journal on Computing*, 3(4), 376–384. (Infinite-precision Euclidean distance rounding.)
7. NVIDIA Corporation. *CUDA C++ Programming Guide* — warp shuffle functions, `__syncthreads`, and shared-memory capacity per block. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
