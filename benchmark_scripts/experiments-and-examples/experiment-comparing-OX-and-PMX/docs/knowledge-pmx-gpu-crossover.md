---
title: "GPU Partially Mapped Crossover (PMX) Kernel Architecture and Empirical Comparison with Order Crossover (OX)"
description: >
  Technical synthesis of the GPU Partially Mapped Crossover (PMX) CUDA kernel design,
  warp-private shared-memory mapping-chain resolution, host-device transfer contract,
  engineering pitfalls resolved during development, and empirical quality/runtime
  comparison against Order Crossover (OX) across TSPLIB instances with and without
  2-opt local search.
created: 2026-11-24
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: local
modifications:
  - date_modified: 2026-11-24
    modifications:
      - description: >
          Initial creation: synthesized the CUDA kernel architecture, shared-memory
          layout, host-device RNG/residency boundary, bugs and measurement artifacts
          discovered across the 10:46–12:37 experimental iterations, and statistical
          findings comparing PMX against OX with and without 2-opt local search.
related_files:
  - "[gpu_pmx_crossover.py](../gpu_pmx_crossover.py)"
  - "[pmx_crossover.cu](../kernels/pmx_crossover.cu)"
  - "[compare-pmx-ox-runner.py](../compare-pmx-ox-runner.py)"
  - "[algorithms.json](../configs/algorithms.json)"
  - "[problems.json](../configs/problems.json)"
  - "[ox_vs_pmx.md](../results/tables/ox_vs_pmx.md)"
  - "[pmx_crossover.py](../../../../src/algorithms/strategies/crossover/pmx_crossover.py)"
  - "[pmx_crossover.cu](../../../../src/algorithms/strategies/crossover/kernels/pmx_crossover.cu)"
  - "[order_crossover.py](../../../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[order_crossover.cu](../../../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[analysis-crossover-gpu-architecture-and-scaling.md](../../../../documentation/knowledge-base/analysis-crossover-gpu-architecture-and-scaling.md)"
tags:
  - algorithm/crossover
  - algorithm/pmx
  - algorithm/order-crossover
  - algorithm/genetic-algorithm
  - benchmark/gpu
  - benchmark/statistical-analysis
  - analysis/performance
  - analysis/scaling
  - analysis/architecture
  - guide/cuda
  - notes/cuda
  - notes/hardware
---

# GPU Partially Mapped Crossover (PMX) Kernel Architecture and Empirical Comparison with Order Crossover (OX)

This note captures the design, engineering lessons, and empirical findings from developing and benchmarking a warp-parallel GPU Partially Mapped Crossover (PMX) operator against Order Crossover (OX) on TSPLIB instances. The experiment was originally built in isolation under [experiment-comparing-OX-and-PMX](../compare-pmx-ox-runner.py) and its GPU PMX operator has since been promoted into the core framework as [pmx_crossover.py::CrossoverGPUPermPartiallyMapped](../../../../src/algorithms/strategies/crossover/pmx_crossover.py) backed by [pmx_crossover.cu::pmx_crossover_kernel](../../../../src/algorithms/strategies/crossover/kernels/pmx_crossover.cu).

## 1. Operator Semantics: Why PMX and OX Behave Differently on TSP

Both OX (Davis 1985) and PMX (Goldberg & Lingle 1985) recombine two parent permutations $(P_1, P_2)$ across a randomly sampled cut interval $[c_1, c_2]$, preserving the slice $P_1[c_1 \dots c_2]$ verbatim in the child $C$. They diverge fundamentally in what structural invariant they preserve outside $[c_1, c_2]$:

- **Order Crossover (OX — relative order and edge adjacency):** Fills positions outside $[c_1, c_2]$ by cyclically streaming cities from $P_2$ starting at $c_2 + 1$ in the relative order they appear in $P_2$. Because a TSP tour cost $\sum_i d(\pi_i, \pi_{(i+1) \bmod n})$ is invariant under cyclic shifts and depends exclusively on *city adjacency* (edges), OX preserves building blocks of contiguous edges (Oliver et al. 1987).
- **Partially Mapped Crossover (PMX — absolute index position):** Resolves conflicts for every uncopied city $v \in P_2[c_1 \dots c_2] \setminus P_1[c_1 \dots c_2]$ by following the bijection mapping induced by aligned positions inside $[c_1, c_2]$:
  $$i_0 = \text{idx}(v \text{ in } P_2), \qquad i_{k+1} = \text{inv}_{P_2}\bigl(P_1[i_k]\bigr) \quad \text{until } i_m \notin [c_1, c_2]$$
  placing $C[i_m] \leftarrow v$, and copying all remaining unmapped slots directly from $P_2[j]$.

PMX preserves *absolute array indices* (Goldberg & Lingle's O-schemata). On a cyclic TSP representation where absolute index $j$ carries no geometric meaning without edge context, enforcing absolute-position alignment disrupts existing sub-tours whenever parents are cyclically shifted relative to each other (Oliver et al. 1987).

## 2. CUDA Kernel Architecture (`pmx_crossover_kernel`)

The GPU PMX kernel in [pmx_crossover.cu::pmx_crossover_kernel](../../../../src/algorithms/strategies/crossover/kernels/pmx_crossover.cu) mirrors the 1-warp-per-route architecture established for OX in [order_crossover.cu::order_crossover_kernel](../../../../src/algorithms/strategies/crossover/kernels/order_crossover.cu), using disjoint parent pairs $(2k, 2k+1)$ that share cut points $(c_1, c_2)$, passthrough of a trailing odd route, and fixed depot endpoints when `is_closed != 0`.

### 2.1 Warp-Private Shared Memory Layout ($5n$ Bytes per Warp)

Whereas Format A OX needs only a 1-byte-per-city membership table (`in_seg[n]`, $n$ bytes/warp; see [analysis-crossover-gpu-architecture-and-scaling.md](../../../../documentation/knowledge-base/analysis-crossover-gpu-architecture-and-scaling.md)), PMX also requires an $O(1)$ inverse lookup `inv_p2[city] -> position in P2` during mapping-chain traversal. Each warp therefore claims a private slice of dynamic shared memory:

| Buffer | Element Type | Bytes / Warp | Purpose |
| :--- | :--- | ---: | :--- |
| `my_inv_p2` | `int32` (`int`) | $4n$ | Maps city ID $v \in [0, n)$ to its stream index in $P_2$ |
| `my_in_seg` | `uint8` (`unsigned char`) | $1n$ | Marks cities copied in $P_1[c_1 \dots c_2]$ for $O(1)$ exclusion |
| **Total** | — | **$5n$** | 4-byte aligned `int[]` region followed by `unsigned char[]` |

For a block of $W = \text{threads\_per\_block} / 32$ warps, total dynamic shared memory is:

$$\text{SMEM}_{\text{block}} = W \cdot 5n \le \text{MaxSharedMemoryPerBlock}$$

On Pascal GP107 (GTX 1050, $\text{MaxSharedMemoryPerBlock} = 49{,}152\text{ B}$), a single block supports:
- $W = 8$ warps (256 threads/block) up to $n = \lfloor 49{,}152 / 40 \rfloor = 1{,}228$ cities;
- $W = 1$ warp (32 threads/block) up to $n = \lfloor 49{,}152 / 5 \rfloor = 9{,}830$ cities.

Because CUDA requires block dimensions and dynamic shared-memory size to be reserved by the SM hardware block scheduler *before* kernel entry, [pmx_crossover.py::CrossoverGPUPermPartiallyMapped.crossover_batch](../../../../src/algorithms/strategies/crossover/pmx_crossover.py) dynamically halves `threads_per_block` ($256 \to 128 \to 64 \to 32$) until $W \cdot 5n \le \text{MaxSharedMemoryPerBlock}$, and raises `ValueError` if $5n$ exceeds hardware SMEM for $W = 1$.

### 2.2 Conflict-Free Parallel Bijection Traversal (Zero Atomics)

All 32 lanes of a warp cooperate across four `__syncwarp()`-bounded phases:

1. **Parallel Clear:** Lanes stride across $[0, n)$ setting `child[i] = -1` (or preserving `parent1[0]`/`parent1[n-1]` if `is_closed`), `my_inv_p2[i] = -1`, and `my_in_seg[i] = 0`.
2. **Inverse Map & Segment Copy:** Lanes populate `my_inv_p2[parent2[pos]] = pos` over `[stream_start, stream_start + stream_len)` and copy `parent1[c1..c2]` into `child[c1..c2]` while setting `my_in_seg[city] = 1`.
3. **Parallel Mapping-Chain Resolution:** Lanes stride across `idx in [c1, c2]`. Whenever `val = parent2[idx]` satisfies `my_in_seg[val] == 0`, that lane follows `pos = my_inv_p2[parent1[pos]]` until `pos` exits `[c1, c2]` (bounded by `steps <= seg_len`) and writes `child[pos] = val`. Because $P_1$ and $P_2$ are permutations of the same city set, the mapping restricted to $[c_1, c_2]$ decomposes into disjoint chains and cycles; distinct unmapped elements in $P_2[c_1 \dots c_2]$ are guaranteed to terminate at strictly distinct positions outside $[c_1, c_2]$. Hence, Step 3 has **zero write conflicts and needs zero atomics**.
4. **Parallel Unmapped Fill:** Lanes stride across `[stream_start, stream_start + stream_len)` and copy `parent2[pos]` wherever `child[pos] == -1`.

Compared to OX's branchless warp-ballot compaction (`__ballot_sync` + `__popc`), PMX trades smaller shared-memory occupancy ($5n$ vs. $1n$ B/warp) and data-dependent `while` loop divergence in Step 3 for $O(1)$ per-lane chain resolution without prefix-sum compaction.

## 3. Host-Device Boundary, RNG Parity, and Timing Semantics

In a full-GPU modular GA pipeline (`InitializationRandom(return_gpu=true)`, `SelectionGPUTournament`, `MutationGPUPermSwap`, `EvaluationGPUCost(return_gpu=true)`):

- **Zero Route Transfers:** `d_parents` and `d_offspring` remain 100% resident in GPU VRAM (`d2h_bytes == 0` on GPU-input calls). Topology detection via [route_helpers.py::resolve_route_bounds](../../../../src/utils/route_helpers.py) reads `context.is_closed` directly from host metadata without syncing `batch[0]`.
- **Control-Only Cut-Point H2D ($8N$ bytes/generation):** Host Python draws $(c_1, c_2)$ using [route_helpers.py::sample_distinct_pairs](../../../../src/utils/route_helpers.py) so CPU and GPU implementations consume the identical NumPy RNG stream and remain bit-identical (`2 * N * 4 = 1,024 B` at $N=128$, vs. $\sim 513\text{ KB}$ one-way if route arrays at $n=1{,}002$ crossed PCIe).
- **Host Stopwatch vs. CUDA Events:** Because `PhaseTimer` defaults to `use_cuda_events=False` (`time.perf_counter()`, measuring host enqueue latency rather than synchronized device completion), heavy asynchronous GPU kernels leak execution time into the next synchronization point (`Host Sync`). When GPU 2-opt was active, `LS` reported only $\sim 1.7\%$ while `Host Sync` absorbed $\sim 88\%$ of wall time; when 2-opt was disabled (`LocalSearchNoOp`), every GPU kernel finished in microseconds, `Host Sync` dropped to $9.0\%\text{–}13.6\%$, and per-generation wall time stabilized at $\sim 1.10\text{–}1.28\text{ ms/gen}$ (dominated by fixed Python/CuPy dispatch and host RNG overhead).

## 4. Engineering Pitfalls and Bugs Resolved

Several subtle configuration, schema, and benchmarking artifacts surfaced across the experimental iterations (`10:46` through `12:37`) and guided both the experiment setup and the `src/` promotion:

1. **Operator Aliasing & Case-Sensitive Slot Typo (`10:46`–`11:55`):** The initial `PMX` entry in `algorithms.json` accidentally specified `"class": "CrossoverCPUPermOrder"` (comparing OX against itself), and disabling 2-opt introduced `"LocalSearchNoOP"` (wrong capitalization) alongside stale `LocalSearchGPUTourTwoOpt` parameters (`mode`, `max_iterations`, `candidate_k`). Because `additionalProperties: false` in the generated schema rejects those extra keys, editor `oneOf` matching hid `LocalSearchNoOp` from autocompletion until the stale `params` block was removed.
2. **Working-Directory Path Resolution (`10:43`):** Invoking `uv run tsp-benchmark` from inside a nested directory failed on relative `--config-dir` paths because `uv` locates `pyproject.toml` upward without changing `cwd`. Anchoring `EXP_DIR = Path(__file__).resolve().parent` in [compare-pmx-ox-runner.py](../compare-pmx-ox-runner.py) made execution CWD-independent.
3. **Telemetry Reset Method Name:** Early parity scripts called a nonexistent `reset_telemetry()` method on `CrossoverGPUPermPartiallyMapped`; [base.py::StrategyBase.reset_metrics](../../../../src/algorithms/strategies/base.py) is the canonical interface (`h2d_bytes`, `d2h_bytes`, `kernel_launches`).
4. **Cold-Start CuPy Context Artifact (`10:59` & `11:27`):** In the `att48`/`berlin52` runs with 2-opt enabled, `OX_GPU` ran first and absorbed CuPy/CUDA context and memory-pool initialization spikes (`112.7 ms` on rep 1, `65.9 ms` on rep 6 vs. `11–16 ms` warm steady state, median $\sim 14.2\text{ ms}$), whereas `PMX_GPU` ran warm (`median ~14.2 ms`), falsely making `PMX_GPU` appear faster on `att48` (`14.6 ms` vs `18.8 ms`). On already-warm `berlin52`, `OX_GPU` (`14.5 ms`) was faster than `PMX_GPU` (`16.0 ms`). Calling `warmup(context)` ahead of measured repetitions (`DP11-12`, `BUG-24`) eliminates this bias.
5. **Sub-100 ms Runtime Floor & 2-Opt Dominance (`11:27`):** On `att48` and `berlin52` (30 reps with `LocalSearchGPUTourTwoOpt(mode="batch", max_iterations=10, candidate_k=20)`), CPU variants spent $96.0\%\text{–}96.5\%$ of wall time in 2-opt and only $3.2\%\text{–}3.8\%$ in crossover, while GPU variants compressed wall time to $14.5\text{–}18.8\text{ ms}$ ($115.6\times\text{–}124.7\times$ speedup over CPU), pushing execution below the 100 ms Runtime Floor where `Host Sync` took $\sim 30.6\%\text{–}32.9\%$. Moreover, 2-opt repaired both crossovers to near-optimal tours (`0.28%` OX GPU vs. `0.29%` PMX GPU mean gap), yielding $17\text{–}21$ zero paired differences out of $30$ and insufficient Wilcoxon power.
6. **Checkpoint Contamination Preflight Guard (`12:08`):** Switching `algorithms.json` from `LocalSearchGPUTourTwoOpt` (`post_mutation`) to `LocalSearchNoOp` (`none`) triggered the runner's checkpoint integrity guard, preventing silent mixing of 2-opt and no-LS checkpoints until old checkpoints were cleared with `--no-incremental`.
7. **Baseline Name Fallback:** Keeping `BASELINE_ALGORITHM = "OX"` in [compare-pmx-ox-runner.py](../compare-pmx-ox-runner.py) after removing CPU variants triggered `resolve_baseline_algorithm` fallback warnings until aligned with `"OX_GPU"`.

## 5. Empirical Results: OX vs. PMX Across Experimental Regimes

### 5.1 Regime A: Small Instances with 2-Opt (`11:27` Run, `att48` & `berlin52`, $r = 30$)

With batched 2-opt local search active on $n \in \{48, 52\}$, local search masks crossover differences almost completely:

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Avg Speedup vs. OX (CPU) |
| :--- | ---: | ---: | ---: | ---: |
| OX (CPU) | 0.36 | 0.30 | 1.93 | 1.00x |
| PMX (CPU) | 0.45 | 0.21 | 2.03 | 0.95x |
| OX (GPU) | 0.28 | 0.28 | 0.02 | 115.60x |
| PMX (GPU) | 0.29 | 0.20 | 0.02 | 124.65x |

### 5.2 Regime B: Larger Instances ($n > 150$) with 2-Opt (`12:00` Run, $r = 5$, GPU-Only)

Scaling to the 9 TSPLIB instances with $n \in [198, 1002]$ (`d198` through `pr1002`) with GPU 2-opt enabled separated solution quality even under local search: **OX (GPU)** achieved **2.01%** mean optimality gap (ranging from `0.78%` on `pr264` to `4.43%` on `rat783`) versus **6.28%** for **PMX (GPU)** (`1.19%` on `si535` to `10.65%` on `rat783`). PMX terminated sooner in wall time (`1.46 s` vs. `2.02 s` mean) on $n \ge 442$ because its populations stagnated earlier and triggered early stopping at worse tours.

### 5.3 Regime C: Pure Crossover Comparison Without Local Search (`12:37` Canonical Dataset, $r = 30$, 540 Runs)

To isolate the crossover operators without 2-opt repair, the final benchmark in [ox_vs_pmx.md](../results/tables/ox_vs_pmx.md) evaluated `OX_GPU` vs. `PMX_GPU` (`LocalSearchNoOp`, `local_search_stage="none"`, $N=128$, mutation rate $0.03$, $k=5$ tournament selection) across all 9 instances ($n > 150$, 30 repetitions each):

| Problem | $n$ | OX (GPU) Gap (%) | PMX (GPU) Gap (%) | OX Mean Time (s) | PMX Mean Time (s) | OX CX / Gen (ms) | PMX CX / Gen (ms) | OX Mean Gens | PMX Mean Gens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d198` | 198 | 128.63 | 343.79 | 0.49 | 1.02 | 0.205 | 0.201 | 421.9 | 930.9 |
| `pr264` | 264 | 251.93 | 665.38 | 0.75 | 1.71 | 0.209 | 0.216 | 637.5 | 1450.8 |
| `lin318` | 318 | 278.45 | 435.55 | 0.84 | 2.08 | 0.212 | 0.209 | 699.6 | 1840.5 |
| `pcb442` | 442 | 326.78 | 471.83 | 1.10 | 3.13 | 0.209 | 0.210 | 936.1 | 2753.8 |
| `att532` | 532 | 320.79 | 550.87 | 1.70 | 4.15 | 0.215 | 0.218 | 1410.3 | 3519.6 |
| `si535` | 535 | 54.30 | 91.25 | 1.45 | 3.55 | 0.211 | 0.216 | 1219.1 | 3027.9 |
| `gr666` | 666 | 346.61 | 536.87 | 2.07 | 5.91 | 0.227 | 0.221 | 1625.8 | 4950.6 |
| `rat783` | 783 | 403.42 | 592.18 | 2.56 | 7.22 | 0.227 | 0.220 | 2022.3 | 6072.3 |
| `pr1002` | 1002 | 524.58 | 748.19 | 3.36 | 10.69 | 0.229 | 0.223 | 2624.9 | 8904.6 |
| **Overall** | — | **292.83 ± 139.87** | **492.88 ± 192.80** | **1.59 ± 0.94** | **4.39 ± 3.09** | **~0.216** | **~0.215** | — | — |

#### Statistical Conclusions (Demšar 2006; Fleming & Wallace 1986)

- **Overall Cross-Instance Wilcoxon Signed-Rank + Holm-Bonferroni ($n = 9$ instances):**
  - **Solution Gap:** $W = 0.0$, $p = 0.0039$, $p_{\text{Holm}} = 0.0039$, rank-biserial effect size $r = -1.00$ (`OX_GPU` wins on 9/9 instances by $-200.05\text{ pp}$ overall).
  - **Wall Time:** $W = 0.0$, $p = 0.0039$, $p_{\text{Holm}} = 0.0039$, $r = -1.00$; geometric mean speedup of `PMX_GPU` vs. `OX_GPU` is **$0.39\times$** (equivalently, `OX_GPU` is **$2.58\times$** faster end-to-end).
- **Per-Instance Paired Wilcoxon Signed-Rank ($n = 30$ repetitions/instance):**
  - **Tour Cost:** $W = 0.0$, $p = 1.86 \times 10^{-9}$, $r = -1.00$ on all 9 instances (zero ties; large effect).
  - **Wall Time:** $W \in [0.0, 8.0]$, $p \le 4.66 \times 10^{-8}$, $r \in [-1.00, -0.966]$ on all 9 instances.

> [!IMPORTANT]
> **Root Cause of the 2.58x End-to-End Slowdown:** Per-generation crossover kernel runtime for `PMX_GPU` (`0.201–0.223 ms/call`) is **virtually identical** to `OX_GPU` (`0.205–0.229 ms/call`) across $n \in [198, 1002]$. `PMX_GPU` takes $2.58\times$ longer overall not because its CUDA kernel is slower per generation, but because PMX makes smaller, slower improvements per generation, running **$2.2\times\text{–}3.4\times$ more generations** before early stopping triggers (`8,904.6` vs. `2,624.9` generations on `pr1002`) while still converging to worse tours.

## 6. Promotion Into Core Framework

The validated GPU PMX operator has been promoted from the frozen experiment directory into the main framework:

- CUDA kernel: [pmx_crossover.cu::pmx_crossover_kernel](../../../../src/algorithms/strategies/crossover/kernels/pmx_crossover.cu)
- Strategy class: [pmx_crossover.py::CrossoverGPUPermPartiallyMapped](../../../../src/algorithms/strategies/crossover/pmx_crossover.py) (alongside [pmx_crossover.py::CrossoverCPUPermPartiallyMapped](../../../../src/algorithms/strategies/crossover/pmx_crossover.py), aligned to the same `sample_distinct_pairs` and `resolve_route_bounds` contract for bit-identical CPU/GPU parity)
- Registry & schema: registered as `"CrossoverGPUPermPartiallyMapped"` in [__init__.py::STRATEGY_REGISTRY](../../../../src/algorithms/strategies/__init__.py) and exposed strictly under `strategy_slot_crossover` in [algorithms.schema.json](../../../../configs/schemas/algorithms.schema.json)

## References

- **Davis, L.** (1985). *Applying Adaptive Algorithms to Epistatic Domains*. Proceedings of the 9th International Joint Conference on Artificial Intelligence (IJCAI), pp. 162–164.
- **Goldberg, D. E., & Lingle, R.** (1985). *Alleles, Loci, and the Traveling Salesman Problem*. Proceedings of the 1st International Conference on Genetic Algorithms (ICGA), pp. 154–159.
- **Oliver, I. M., Smith, D. J., & Holland, J. R. C.** (1987). *A Study of Permutation Crossover Operators on the Traveling Salesman Problem*. Proceedings of the 2nd International Conference on Genetic Algorithms (ICGA), pp. 224–230.
- **Fleming, P. J., & Wallace, J. J.** (1986). *How Not to Lie with Statistics: The Correct Way to Summarize Benchmark Results*. Communications of the ACM, 29(3), 218–221.
- **Demšar, J.** (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. Journal of Machine Learning Research, 7, 1–30.
