---
title: "GPU Strategy Bottleneck Analysis: Crossover, Selection, Mutation, Evaluation, and 2-Opt"
description: >
  Diagnostic record of the six GPU strategy bottlenecks found in the modular GA solver —
  Order Crossover's $O(n^2)$ membership scan and 2-of-5 SM occupancy, the stride-$n$
  tournament copy, warp divergence in inversion mutation, host-side RNG in swap mutation,
  chunk-strided cost reduction, and Thread-0 serial reversal in 2-opt — each paired with
  the Phase 11 resolution (`OX-A`, `SEL-A`, `INV-A`, `SWAP-A`, `EVAL-A`, `2OPT-A`) and with
  the pedagogical derivations for shared-memory bitpacking and intra-warp compaction.
created: 2026-11-09
status: final
author:
  - "Lucas Galdino"
type: notes
scope: global
modifications:
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Standardized to the repository Markdown standards: replaced the ad-hoc frontmatter
          (`Data`, `Autor`, `Status`, `file`) with the knowledge-base schema, translated the
          body from Portuguese to English, converted workstation-absolute hyperlinks to
          repository-relative links, tagged every fenced block, and replaced the chat-session
          quote markers (`Quote 1`–`Quote 4`) and the `<prior_attempt>` block with prose.
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Re-audited every code claim against `src/` after Phase 11 and rewrote the document
          as diagnosis plus resolution: the six defects are now implemented as `BUG-22` /
          `OX-A`, `SEL-A`, `INV-A`, `SWAP-A`, `EVAL-A` and `2OPT-A`. Line anchors were replaced
          by symbol references so the note survives kernel reorganization, a Table of Symbols
          was added, and the former "Remaining Questions" section became a resolution log with
          the residual gaps that are still open.
related_files:
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](../analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[analysis-crossover-gpu-architecture-and-scaling.md](../analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[analysis-crossover-matrix-formulation-and-mechanics.md](../analysis-crossover-matrix-formulation-and-mechanics.md)"
  - "[guide-cuda-warp-primitives-and-intrinsics.md](../guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[guide-cuda-hardware-bitwise-and-registers.md](../guide-cuda-hardware-bitwise-and-registers.md)"
  - "[analysis-gpu-memory-latency-and-occupancy.md](../analysis-gpu-memory-latency-and-occupancy.md)"
  - "[analysis-gpu-reduction-patterns.md](../analysis-gpu-reduction-patterns.md)"
  - "[order_crossover.cu](../../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[tournament_selection.cu](../../../src/algorithms/strategies/selection/kernels/tournament_selection.cu)"
  - "[inversion_mutation.cu](../../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu)"
  - "[cost_calculator.cu](../../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)"
  - "[two_opt_batch.cu](../../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_common.cuh](../../../src/distances/kernels/two_opt_common.cuh)"
  - "[plan-reorganization-and-typing.plan.md](../../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../../documents_tcc/plan/table-of-symbols.plan.md)"
tags:
  - analysis/gpu
  - analysis/architecture
  - analysis/bottlenecks
  - analysis/performance
  - algorithm/genetic-algorithm
  - algorithm/order-crossover
  - algorithm/two-opt
  - algorithm/mutation
  - algorithm/selection
  - benchmark/gpu
  - benchmark/speedup
  - notes/cuda
  - notes/warp-primitives
  - review/quality
  - notes/tcc
---

# GPU Strategy Bottleneck Analysis: Crossover, Selection, Mutation, Evaluation, and 2-Opt

> [!IMPORTANT]
> This note documents the diagnosis performed **before** Phase 11 and the resolution that
> Phase 11 implemented. Every claim was re-verified against `src/` on 2026-11-09: the six
> bottlenecks described in [§2](#2-strategy-level-bottlenecks) are resolved, and code
> references name the symbol (kernel entry point, class, or method) instead of a line
> number, because the kernels were rewritten across `TASK 11.10`–`TASK 11.16`.

## Table of Symbols

| Symbol | Definition | Physical / algorithmic meaning |
| :--- | :--- | :--- |
| $n$ | Tour length, $n = \lvert V \rvert$ | Cities per route |
| $N$ | Population size | Routes processed per generation |
| $\ell$ | Lane index inside a warp | $\ell \in [0, 31]$ on NVIDIA hardware |
| $p_\ell$ | Membership predicate of lane $\ell$ | $p_\ell = 1$ when the candidate city is absent from the donated segment |
| $B$ | Ballot mask | 32-bit word produced by `__ballot_sync` |
| $\mathbf{L}$ | Strictly lower-triangular lane mask matrix | Row $\ell$ equals $(1\text{u} \ll \ell) - 1\text{u}$ |
| $r_\ell$ | Exclusive prefix rank | $r_\ell = \operatorname{popc}(B \wedge \mathbf{L}_\ell)$, the lane's write offset |
| $p_m$ | Mutation probability | Fraction of routes mutated per generation |
| $T$ | Shared-memory tile length | `TWO_OPT_TILE_SIZE` $= 2048$ tour positions |
| $S_{\text{block}}$ | Shared memory per thread block | 48 KB on Pascal GP107 (GTX 1050) |
| $W$ | DRAM transaction granularity | 128-byte burst |
| $\beta$ | Scaling exponent | Power-law fit $\text{time} \propto n^{\beta}$ from the crossover sweep |
| $L = \lambda W$ | Little's law | Concurrent requests $L$ equal arrival rate $\lambda$ times latency $W$ |

---

## 1. Scope and Reading Guide

This note answers four questions that arose while auditing
`src/algorithms/strategies/` and records the resolution of each finding.

| Question | Section | Canonical treatment |
| :--- | :--- | :--- |
| What is wrong with each GPU strategy, what does the source actually do, and what was fixed? | [§2](#2-strategy-level-bottlenecks) | [analysis-gpu-ga-strategies-coalescing-and-divergence.md](../analysis-gpu-ga-strategies-coalescing-and-divergence.md) |
| Why cannot the 2-opt shared-memory tile be reused in Order Crossover? | [§3](#3-why-the-2-opt-tile-does-not-transfer-to-order-crossover) | [analysis-crossover-gpu-architecture-and-scaling.md](../analysis-crossover-gpu-architecture-and-scaling.md) |
| How are byte arrays and bitpacked arrays laid out in physical memory? | [§4](#4-storage-formats-byte-array-vs-bitpacked-array) | [analysis-crossover-gpu-architecture-and-scaling.md §6.3](../analysis-crossover-gpu-architecture-and-scaling.md#63-direct-address-lookup-byte-array-vs-bitpacked-array-mechanics) |
| How does a warp compact survivors into contiguous slots without locks? | [§5](#5-warp-compaction-via-a-lower-triangular-mask-and-__popc) | [guide-cuda-warp-primitives-and-intrinsics.md](../guide-cuda-warp-primitives-and-intrinsics.md) |

Sections [§3](#3-why-the-2-opt-tile-does-not-transfer-to-order-crossover) to
[§5](#5-warp-compaction-via-a-lower-triangular-mask-and-__popc) are timeless hardware
derivations. Section [§2](#2-strategy-level-bottlenecks) is a diagnostic log whose claims
must be read against the *pre-Phase-11* source; the present implementation is recorded in
§7 (Resolution Log and Residual Gaps).

---

## 2. Strategy-Level Bottlenecks

### 2.1 Crossover — `CrossoverGPUOrder`

**Diagnosis (pre-Phase-11).** The `order_crossover_kernel` performed membership testing by
linear scan: for each of the $n$ cities drawn from $P_2$ it ran `for (int j = c1; j <= c2; j++)`
against global memory. With an average segment of $n/2$ positions and $n = 2103$, that is
$\approx 1.47 \times 10^6$ uncoalesced global reads per route, i.e. $\mathcal{O}(n^2)$ per
route against the $\mathcal{O}(n)$ `set` used by `CrossoverOrder._apply_ox`.

Three further limitations compounded it:

1. **Occupancy.** One thread per route with 128 threads per block launches
   $\lceil 256/128 \rceil = 2$ blocks. On a 5-SM GP107, three SMs stayed idle and the two
   active ones ran 4 warps each — 6.25 % of the 64-warp capacity. With one warp per
   scheduler, DRAM latency (200–800 cycles) cannot be hidden: Little's law
   $L = \lambda W$ pins achievable throughput to the number of in-flight requests.
2. **Cut sampling.** The GPU drew `cut1` and `cut2` with replacement, admitting degenerate
   segments ($c_1 = c_2$), while the CPU used `rng.choice(..., replace=False)`.
3. **Parent pairing and depots.** The GPU paired parents cyclically $(k, k+1)$ and had no
   depot guard, so closed routes lost their endpoints; the CPU pairs disjointly and
   preserves `[depot, ..., depot]`.

**Resolution (`OX-A`, `DP11-11`, `BUG-22`, `TASK 11.10`).**

- Membership moved to a **direct-address table in shared memory (Format A)**: one
  `unsigned char` per city per warp (`extern __shared__ unsigned char s_in_seg[]`), with a
  warp-private slice `&s_in_seg[warp_in_block * n]`. Lookup is $\mathcal{O}(1)$ and writes
  are contention-free, so no `atomicOr` is needed.
- **One warp per route** (`route_idx = global_tid >> 5`, `lane_id = threadIdx.x & 31`),
  which raises the launched parallelism from $N$ threads to $32N$.
- Compaction of $P_2$ uses `__ballot_sync` plus `__popc` ([§5](#5-warp-compaction-via-a-lower-triangular-mask-and-__popc)).
- The launcher (`CrossoverGPUOrder.crossover_batch`) halves `threads_per_block` while
  `warps_per_block * n_cities` exceeds `MaxSharedMemoryPerBlock`, so the Format A budget is
  respected automatically: 8 warps $\times$ 2103 B $= 16.8$ KB at $n = 2103$, and the limit
  is reached only at $n = 6144$.
- Semantics aligned with the CPU: `rng.choice(..., replace=False)` (strict cuts inside
  $[1, n-2]$ for closed routes), disjoint pairing $(2k, 2k+1)$ sharing the same cuts, and
  `is_closed` preservation. **Constraint:** for identical parents and cuts, the GPU
  offspring must stay bit-identical to `CrossoverOrder`.

**Measured outcome.** Before: `CrossoverGPUOrder` won by 11.2× at $n = 52$, tied near
$n \approx 800$, and lost by 3.6× at $n = 2000$; the power fit gave CPU $\propto n^{0.87}$
and GPU $\propto n^{1.88}$, proving an asymptotic cause rather than a constant factor.
After: 26 parity cases ($n \in \{52, 100, 200, 500, 1000, 2000\}$) are bit-identical, and
`benchmark_scripts/experiments-and-examples/chapter4/crossover_scaling_sweep.py --smoke` reports $\beta_{\text{GPU}} \approx 0.0114$
with a $101.46\times$ gain at $n = 2000$. Treat that exponent as evidence that the curve is
no longer quadratic — at these sizes it is dominated by fixed launch overhead — not as a
predictive constant; the full sweep remains the reference for the Chapter 4 curve.

| Criterion | `OX-A` (warp-cooperative + shared table + ballot) | CPU fallback (multiprocess OX) |
| :--- | :--- | :--- |
| Complexity per route | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| Expected speedup | $30\times$–$50\times$ over CPU at $n = 2103$ | $2\times$–$4\times$ over serial CPU |
| GPU occupancy | 8 warps per block, all 5 SMs busy | none (GPU idle during crossover) |
| PCIe traffic | zero when routes are already device-resident | $2Nn$ int32 per generation |
| Implementation risk | warp primitives and shared-memory budgeting | low |

### 2.2 Tournament Selection — `SelectionGPUTournament`

**Diagnosis (pre-Phase-11).** In `tournament_selection_kernel`, thread `parent_idx` copied
all $n$ cities of the winning route, so consecutive threads wrote addresses $n$ integers
apart. Each lane therefore issued its own 32-byte DRAM transaction: with $n = 1000$ the
bus efficiency fell below 10 % (96.9 % of the fetched bytes were unused). For
$N_{\text{select}} = 100$ with `threads_per_block = 128`, the kernel launched a single
block, and `cp.cuda.Stream.null.synchronize()` blocked the host after every call.

**Resolution (`SEL-A`, `DP11-16`, `TASK 11.14`).**

- The kernel now emits only the 1D vector `winner_indices[parent_idx] = best_idx`; the
  2D route gather is delegated to CuPy (`d_parents = d_pop[d_winner_indices]`), which uses
  the CUB `DeviceGather` path with vectorized `int4` loads and full coalescing.
- Candidate sampling stays on the host through `ProblemContext.rng` (`rng.integers(...)`,
  shape `(n_select, k)`), which preserves bit-identity with `SelectionTournament`.
- No explicit `Stream.null.synchronize()` survives anywhere in `src/`.

> [!NOTE]
> The launcher still issues $\lceil N/128 \rceil$ blocks — one block for $N = 100$. This is
> no longer a bottleneck: each thread now performs $\mathcal{O}(k)$ work, and the
> bandwidth-bound copy happens inside the gather kernel, which is grid-sized by CuPy. What
> remains is the $\mathcal{O}(Nk)$ candidate transfer (~2.5 KB at $N = 100$, $k = 5$).

### 2.3 Inversion Mutation — `MutationGPUInversion`

**Diagnosis (pre-Phase-11).** Two defects:

1. `d_routes[0, 0].get()` and `d_routes[0, -1].get()` transferred 4 bytes each way and
   imposed two host–device barriers **per generation**, although the host already knows
   whether the route is closed.
2. With one thread per route and $p_m = 0.10$, roughly 29 of every 32 lanes in a warp sat
   idle in lockstep while 3 lanes executed up to $n/2$ sequential swaps. The reversal itself
   (`while (left < right)`) was serial.

**Resolution (`INV-A`, `DP11-17`, `TASK 11.15`).**

- **One warp per route** (`route_idx = global_tid >> 5`, `lane_id = threadIdx.x & 31`) and a
  cooperative reversal `for (int k = lane_id; k < num_swaps; k += 32)`. All 32 lanes test the
  same scalar `probs[route_idx] < mutation_rate`, so intra-warp divergence drops to zero;
  non-mutated routes retire in one cycle, mutated routes split the swaps across lanes.
- The endpoint probe is replaced by `MutationGPUInversion._is_closed_cache`, populated once
  per `n_cities`, eliminating both `.get()` calls.
- Launch geometry: `num_routes * 32` threads in blocks of 256, i.e. 13 blocks for $N = 100$
  and full SM coverage.

| Criterion | `INV-A` (host flag + warp inversion) | Stream compaction of mutated routes |
| :--- | :--- | :--- |
| PCIe latency | zero device probes | zero |
| Intra-warp divergence | none for the predicate; cooperative inside the reversed segment | none |
| Reversal throughput | up to $32\times$ vs. a single lane | $32\times$–$256\times$ |
| Kernel launches | 1 | 2 (compaction + reversal) |

### 2.4 Swap Mutation — `MutationGPUSwap`

**Diagnosis (pre-Phase-11).** `pos1`, `pos2` and `probs` were generated with NumPy on the
host and uploaded, costing roughly $25\,\mu\text{s}$ of host time plus PCIe transfer to
perform a swap that executes in under $0.3\,\mu\text{s}$ on the device. A
`Stream.null.synchronize()` prevented overlap with the following evaluation.

**Resolution (`SWAP-A`, `DP11-14`, `TASK 11.12`).**

- The unconditional stream synchronization is removed; device arrays produced in
  `mutate_batch` can flow directly into the next stage.
- The RNG **deliberately remains on the host** (`resolve_rng(context, self.rng)`), because
  `MutationGPUSwap` must be bit-identical to `MutationSwap` for the iso-algorithmic
  comparison of Chapter 4. The transfer of `probs` (float64), `pos1` and `pos2` (int32) —
  about $16N$ bytes per generation — is the accepted price of that equivalence.

> [!WARNING]
> Moving this draw to `cp.random` or cuRAND would eliminate ~$16N$ bytes per generation but
> immediately breaks reproducibility against the CPU reference, and with it the iso-algorithmic
> claim. The trade-off was decided in favour of equivalence (`DP11-14`); any future change
> here must re-open that decision rather than optimize locally.

### 2.5 Cost Evaluation — `EvaluationGPUCost`

**Diagnosis (pre-Phase-11).** `cost_calculator_kernel` partitioned the tour with
`start = tid * chunk`, so adjacent threads read cities `chunk` positions apart (36-byte
strides at $n = 2103$), fragmenting the 128-byte DRAM transactions. The block reduction ran
`__syncthreads()` at every halving down to `stride = 1`, even with a single active warp.

**Resolution (`EVAL-A`, `DP11-15`, `TASK 11.13`).**

- The loop is now block-strided and therefore fully coalesced:

  ```cpp
  for (int i = tid; i < n; i += block_size) {
      int from_node = tour[i];
      int to_node = tour[(i + 1) % n];
      partial_cost += (long long)gpu_pairwise_distance(from_node, to_node, distances, coords, n, edge_type_code);
  }
  ```

- The shared reduction descends only to `stride = 32`; warp 0 then finishes the last 32
  partial sums in registers with `__shfl_down_sync`, removing the tail barriers.
- Shared memory holds `blockDim.x` partials of 8 bytes (2 KB at 256 threads), so the
  footprint is independent of $n$ and the kernel never required tiling at $n = 11\,425$.

> [!NOTE]
> The 256-thread ceiling in `_next_power_of_2` is **not** a work cap under a stride loop: each
> thread iterates over multiple cities. It is an occupancy choice that keeps the shared-memory
> array at 2 KB. The earlier framing of this limit as a bottleneck was incorrect and is
> corrected here.

### 2.6 2-Opt Local Search — `LocalSearchGPUTwoOpt`

**Diagnosis (pre-Phase-11).** After the block selected the best improving move
(`s_deltas[0] < 0`), **Thread 0** reversed the winning segment alone
(`while (left < right)`), while the other 255 threads waited at `__syncthreads()`. A 1000-city
segment cost 500 sequential iterations with 0.4 % of the block doing useful work.

**Resolution (`2OPT-A`, `DP11-18`, `TASK 11.16`).**

- The reversal is now block-cooperative in both the tiled and the candidate paths:

  ```cpp
  int num_swaps = (right - left + 1) / 2;
  if (use_candidates) {
      for (int k = tid; k < num_swaps; k += block_size) {
          int temp = tour[left + k];
          tour[left + k] = tour[right - k];
          tour[right - k] = temp;
      }
  } else {
      for (int k = tid; k < num_swaps; k += block_size) {
          int temp = TOUR_AT(left + k);
          TOUR_SET(left + k, TOUR_AT(right - k));
          TOUR_SET(right - k, temp);
      }
  }
  ```

- Symmetric pairs $(left+k,\ right-k)$ are mutually disjoint, so the parallel form is
  race-free and leaves the sequence of accepted moves — hence the tour — bit-identical to
  `LocalSearchCPUTwoOpt(mode="best")`.
- Tiling remains intact: above `max_shared_tour_cities` (11 520 by default) the kernel
  streams `s_tour_tile[TWO_OPT_TILE_SIZE + 1]` through `TOUR_AT`/`TOUR_SET`.

**Residual gap.** The final reduction after the `stride > 32` shared phase is still a serial
loop executed by Thread 0 over `block_size` partials, and the tile sentinel
`s_tour[window_len]` is written by Thread 0. Both are $\mathcal{O}(\text{block size})$ or
$\mathcal{O}(1)$ per pass, negligible against the $\Theta(n^2)$ scan, but they are the last
single-threaded sections of the kernel and are tracked in §7 (Resolution Log and Residual Gaps).

---

## 3. Why the 2-opt Tile Does Not Transfer to Order Crossover

Two independent reasons: **structural** (access locality) and **quantitative** (memory
footprint).

### 3.1 Access Structure: Spatial Locality (2-opt) versus Stochastic Scatter (OX)

```text
2-Opt (contiguous edge locality):
Route: [ ... | 2048, 2049, 2050, 2051, 2052, ... | ... ]
             └────────── tile in shared memory ──────────┘
Moves (i, j) inside the tile touch shared memory only.

Order Crossover (stochastic city scatter):
Parent 2: [ 1482,  12, 49201,   385, 22104,    3, 10450, ... ]
Lookup:     miss   hit?  miss   miss   miss   hit?  miss
The cities of P2 are uniformly spread over [0, n-1].
```

In 2-opt the tour is a topological sequence: promising pairs $(i, j)$ are close along the
route, so slicing the tour into contiguous blocks of $T = 2048$ positions lets the kernel
evaluate all intra-tile moves out of shared memory.

In OX the goal is different: filter the cities of $P_2$ that are absent from the segment
donated by $P_1$. Because $P_2$ is a random permutation, its cities carry no spatial
correlation with city indices — lane 0 may inspect city 45 102 while lane 1 inspects city
14. Tiling the *membership table* would give a miss rate above 95 %, forcing constant tile
reloads and destroying the benefit.

### 3.2 The Footprint Paradox: Integers versus Bits

The 2-opt tile exists because a full tour of int32 costs 4 bytes per city:

$$\text{2-opt for } n = 50{,}000: \quad 50{,}000 \times 4\text{ B} = 200\text{ KB} > 48\text{ KB} = S_{\text{block}}$$

Tiling 2048 integers (8 KB) is therefore mandatory. The OX table stores a **predicate**,
not a position:

$$\text{Format B (bitpacked) at } n = 2103: \quad \lceil 2103/32 \rceil \times 4\text{ B} = 264\text{ B}$$
$$\text{Format B (bitpacked) at } n = 50{,}000: \quad \lceil 50{,}000/32 \rceil \times 4\text{ B} = 6250\text{ B} \approx 6.1\text{ KB}$$

Conclusion: tiling is unnecessary because the table is tiny; and tiling the table is
harmful because the access pattern has no locality. The implemented kernel therefore keeps
the **whole** table resident — one byte per city per warp (Format A), 16.8 KB at $n = 2103$
with 8 warps — and sizes the block accordingly.

---

## 4. Storage Formats: Byte Array vs. Bitpacked Array

### 4.1 Physical Layout

Memory is organized in binary cells (bits). The smallest directly addressable unit is a
**byte** (8 bits); integers such as `uint32_t` group 4 consecutive bytes.

```text
Physical view of the byte array (uint8_t in_segment[8]):
Byte 0       Byte 1       Byte 2       Byte 3       Byte 4
[00000001]   [00000000]   [00000001]   [00000001]   [00000000] ...
City 0       City 1       City 2       City 3       City 4
 (present)   (absent)     (present)    (present)    (absent)
     ▲
     └─ 7 bits wasted as zero in every byte (87.5 % overhead)

Physical view of the bitpacked array (uint32_t bitmap[1]):
A single 32-bit word stores 32 cities simultaneously:
Bit:    31 30 29 ...  4  3  2  1  0
Value: [ 0  0  1 ...  0  1  1  0  1 ]
City:   31 30 29 ...  4  3  2  1  0
```

### 4.2 Byte Array (`uint8_t`)

- **Mechanics:** each city owns a one-byte drawer. Read with `bool exists = in_segment[c];`,
  write with `in_segment[c] = 1;`.
- **Advantage:** direct indexing, no arithmetic, and *distinct addresses for distinct
  cities*, hence zero write contention between lanes.
- **Disadvantage:** wastes 7 of every 8 bits. At $n = 50{,}000$ it needs 50 KB, exceeding the
  48 KB block budget.

### 4.3 Bitpacked Array (`uint32_t`)

Each 32-bit word is a drawer with 32 compartments (bits). Two arithmetic steps locate a city:

1. **Which word?** $\lfloor c/32 \rfloor$, implemented as `c >> 5` (shift by 5 bits).
2. **Which bit?** $c \bmod 32$, implemented as `c & 31` (mask with $2^5 - 1$).

Worked examples:

- **City 5:** word $5 \gg 5 = 0$ (`bitmap[0]`), bit $5 \ \& \ 31 = 5$.
- **City 35:** word $35 \gg 5 = 1$ (`bitmap[1]`), bit $35 \ \& \ 31 = 3$, because $35 = 1 \times 32 + 3$.
- **City 75:** word $75 \gg 5 = 2$ (`bitmap[2]`), bit $75 \ \& \ 31 = 11$, because $75 = 2 \times 32 + 11$.

```cpp
// Test membership
uint32_t word = bitmap[c >> 5];
bool exists = (word >> (c & 31)) & 1u;

// Mark present (bitwise OR) — requires atomicOr: several lanes may share one word
atomicOr(&bitmap[c >> 5], 1u << (c & 31));

// Clear (bitwise AND with NOT)
bitmap[c >> 5] &= ~(1u << (c & 31));
```

> [!IMPORTANT]
> The implemented kernel uses **Format A** (`uint8_t in_segment[n]`), because distinct bytes
> mean distinct addresses and therefore no `atomicOr` contention. Format B is preserved as
> the scalability roadmap for $n > 10{,}000$ (`DP12-1`, `TASK 12.1`): it is 8× smaller but
> pays for one atomic per marked city. See
> [analysis-crossover-gpu-architecture-and-scaling.md §6.4](../analysis-crossover-gpu-architecture-and-scaling.md#64-shared-memory-capacity-verification-across-tsplib-instances--hardware-occupancy-math)
> for the capacity table.

---

## 5. Warp Compaction via a Lower-Triangular Mask and `__popc`

This section explains how 32 lanes write valid data into contiguous memory positions in one
clock cycle, without atomics and without races. The kernel applies the 32-lane form; the
8-lane example below keeps the arithmetic readable.

### 5.1 The Stream-Compaction Problem

Eight lanes read eight candidate cities of $P_2$:

$$\text{cities per lane } \ell \in [0..7]: \quad [14, 7, 29, 3, 41, 18, 9, 22]$$

Consulting the membership table yields the predicate vector (0 = already donated by $P_1$):

$$\mathbf{p} = [0, 0, \mathbf{1}, 0, \mathbf{1}, \mathbf{1}, 0, \mathbf{1}]$$

Only lanes 2, 4, 5 and 7 hold valid cities $[29, 41, 18, 22]$. How can those four lanes
write to `child[10]`, `child[11]`, `child[12]`, `child[13]` concurrently without clobbering
each other?

### 5.2 The Role of `__ballot_sync`

`__ballot_sync(0xFFFFFFFFu, is_valid)` has every lane publish its predicate bit
$p_\ell \in \{0,1\}$; the hardware folds the 32 votes into one mask and broadcasts it:

$$\text{mask } B = \texttt{0b10110100} \quad (\text{bits } 2, 4, 5, 7 \text{ set})$$

### 5.3 The Crucial Question per Lane and the Matrix $\mathbf{L}$

Each lane $\ell$ only needs to answer: *how many lanes **before me** are also valid?*

- none → I am first, offset `+0`;
- one → I am second, offset `+1`;
- two → I am third, offset `+2`.

That count is an **exclusive prefix sum**. Isolating the lanes below $\ell$ uses

$$\texttt{lower\_lanes\_mask} = (1\text{u} \ll \ell) - 1\text{u}$$

which, evaluated for every lane, produces the strictly lower-triangular matrix $\mathbf{L}$:

$$\begin{equation}
\mathbf{L} = \begin{array}{r|cccccccc|l}
& \text{Bit 0} & \text{Bit 1} & \text{Bit 2} & \text{Bit 3} & \text{Bit 4} & \text{Bit 5} & \text{Bit 6} & \text{Bit 7} & (1\text{u} \ll \ell) - 1\text{u} \\
\hline
\text{Lane } 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0b00000000} \ (2^0 - 1 = 0) \\
\text{Lane } 1 & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0b00000001} \ (2^1 - 1 = 1) \\
\text{Lane } 2 & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0b00000011} \ (2^2 - 1 = 3) \\
\text{Lane } 3 & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & \texttt{0b00000111} \ (2^3 - 1 = 7) \\
\text{Lane } 4 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & \texttt{0b00001111} \ (2^4 - 1 = 15) \\
\text{Lane } 5 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & \texttt{0b00011111} \ (2^5 - 1 = 31) \\
\text{Lane } 6 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & \texttt{0b00111111} \ (2^6 - 1 = 63) \\
\text{Lane } 7 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & \texttt{0b01111111} \ (2^7 - 1 = 127)
\end{array}
\end{equation}$$

### 5.4 Matrix-Vector Form via `__popc`

```cpp
int rank = __popc(ballot_mask & lower_lanes_mask);
```

This is row $\ell$ of $\mathbf{L}$ multiplied by $\mathbf{p}$: `&` performs the element-wise
product and `__popc` sums the bits in one native instruction.

$$\begin{equation}
\begin{bmatrix} r_0 \\ r_1 \\ \mathbf{r_2} \\ r_3 \\ \mathbf{r_4} \\ \mathbf{r_5} \\ r_6 \\ \mathbf{r_7} \end{bmatrix}
=
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
0 \\ 0 \\ \mathbf{1} \\ 0 \\ \mathbf{1} \\ \mathbf{1} \\ 0 \\ \mathbf{1}
\end{bmatrix}
=
\begin{bmatrix}
0 \\ 0 \\ \mathbf{0} \\ 1 \\ \mathbf{1} \\ \mathbf{2} \\ 3 \\ \mathbf{3}
\end{bmatrix}
\end{equation}$$

### 5.5 Full Numerical Trace: Destination of Each Lane

With `write_pos = 10` at the start of the round:

| Step | Lane 0 | Lane 1 | **Lane 2** | Lane 3 | **Lane 4** | **Lane 5** | Lane 6 | **Lane 7** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate city from $P_2$** | `14` | `7` | **`29`** | `3` | **`41`** | **`18`** | `9` | **`22`** |
| **Inside the $P_1$ segment?** | yes | yes | **no** | yes | **no** | **no** | yes | **no** |
| **Predicate `is_valid`** | `0` | `0` | **`1`** | `0` | **`1`** | **`1`** | `0` | **`1`** |
| **`rank` $= \mathbf{L}_\ell \cdot \mathbf{p}$** | `0` | `0` | **`0`** | `1` | **`1`** | **`2`** | `3` | **`3`** |
| **Write destination** | skip | skip | **`child[10]`** | skip | **`child[11]`** | **`child[12]`** | skip | **`child[13]`** |

The four valid cities land in contiguous positions `10, 11, 12, 13` with no conflict, no
atomics, and deterministic bit-identity.

### 5.6 `__popc` Across Numeric Types

| Type | Width | Correct call | Hardware cost | Pitfall if called naively |
| :--- | :---: | :--- | :---: | :--- |
| `uint32_t` (`unsigned int`) | 32 | `__popc(x)` | 1 cycle (native `POPC`) | none |
| `int32_t` (`int`) | 32 | `__popc((unsigned int)x)` | 1 cycle | safe: two's complement reinterpretation |
| `uint64_t` (`unsigned long long`) | 64 | `__popcll(x)` | 2 `POPC` + 1 `IADD` | `__popc(x)` **truncates** the upper 32 bits |
| `int64_t` (`long long`) | 64 | `__popcll((unsigned long long)x)` | 2 `POPC` + 1 `IADD` | same truncation |
| `uint8_t` / `uint16_t` | 8/16 | `__popc((unsigned int)x)` | 1 cycle | safe: zero-extended |
| `int8_t` (`signed char`) | 8 | `__popc((unsigned int)(uint8_t)x)` | 1 cycle | **danger:** $x = -5$ (`11111011`) sign-extends to 24 ones and returns **31**, not 7 |
| `float` (IEEE-754 FP32) | 32 | `__popc(__float_as_uint(x))` | 0-cycle cast + 1 `POPC` | **danger:** `__popc(3.5f)` converts to integer `3` first |
| `double` (IEEE-754 FP64) | 64 | `__popcll(__double_as_ull(x))` | 0-cycle cast + 2 `POPC` | requires the bit reinterpretation intrinsic |

---

## 6. Verification Commands

Every claim above is executable. The parity tests are the ones that would fail if a kernel
silently diverged from its CPU twin.

```bash
# Cross-stage parity and boundary coverage (CPU, no GPU marker required)
rtk .venv/bin/python -m pytest tests/algorithms/strategies -q

# CPU/GPU bit-identity for Order Crossover (n in {52, 100, 200, 500, 1000, 2000})
rtk .venv/bin/python -m pytest tests/algorithms/strategies/crossover/test_unit_gpu_order_crossover.py -q

# Selection, mutation, evaluation and 2-opt parity suites
rtk .venv/bin/python -m pytest tests/algorithms/strategies/selection tests/algorithms/strategies/mutation \
  tests/algorithms/strategies/evaluation tests/algorithms/strategies/local_search -q

# Crossover scaling curve (beta_GPU and speedup at n = 2000)
rtk .venv/bin/python benchmark_scripts/experiments-and-examples/chapter4/crossover_scaling_sweep.py --smoke
```

---

## 7. Resolution Log and Residual Gaps

| Identifier | Finding | Resolution | Status |
| :--- | :--- | :--- | :--- |
| `BUG-22` / `OX-A` | OX membership scan $\mathcal{O}(n^2)$, 1 thread per route, divergent cuts/pairing, depot loss | `TASK 11.10` — Format A table, 1 warp per route, `replace=False`, disjoint pairing, `is_closed` | implemented |
| `SEL-A` | stride-$n$ route copy, 96.9 % bandwidth waste, unconditional sync | `TASK 11.14` — `winner_indices` + CuPy gather | implemented |
| `INV-A` | two blocking `.get()` per generation, 29/32 idle lanes, serial reversal | `TASK 11.15` — `_is_closed_cache`, 1 warp per route, `k += 32` | implemented |
| `SWAP-A` | host RNG upload, redundant `Stream.null.synchronize()` | `TASK 11.12` — sync removed, host RNG kept for bit-identity | implemented |
| `EVAL-A` | chunk-stride reads, barrier chain in the reduction | `TASK 11.13` — block-stride loop, warp-shuffle tail | implemented |
| `2OPT-A` | Thread-0 serial segment reversal | `TASK 11.16` — cooperative block reversal | implemented |
| `DP12-1` | Format B (bitpacked membership) for $n > 10{,}000$ | `TASK 12.1` — dissertation future-work section | open |
| — | Serial tail reduction in `two_opt_batch.cu` and the Thread-0 tile sentinel | not scheduled; $\mathcal{O}(\text{block size})$ per pass | open |
| — | Device-side RNG (`cp.random` / cuRAND) for mutations | **rejected** by `DP11-14`/`DP11-17`: would break CPU/GPU bit-identity | decided |

> [!TIP]
> The three items that the earlier draft listed as "remaining questions" are all closed:
> the plan registration happened in `TASK 11.10`, the RNG placement was decided in
> `DP11-14`/`DP11-17` (host, for equivalence), and the bit-identity test scope was
> implemented and is enforced by the parity suites in [§6](#6-verification-commands).

---

## 8. References

- NVIDIA Corporation. *CUDA C++ Programming Guide* — warp vote (`__ballot_sync`), warp
  shuffle (`__shfl_down_sync`), `__popc`/`__popcll`, and shared-memory organization.
- Harris, M. (2007). *Optimizing Parallel Reduction in CUDA*. NVIDIA Developer Technology.
- Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors: A Hands-on
  Approach* (3rd ed.). Morgan Kaufmann — coalescing, occupancy and Little's law reasoning.
- Davis, L. (1985). Applying Adaptive Algorithms to Epistatic Domains. *Proceedings of
  IJCAI-85*, 162–164 — origin of the Order Crossover operator.
- Reinelt, G. (1991). TSPLIB — A Traveling Salesman Problem Library. *ORSA Journal on
  Computing*, 3(4), 376–384 — instance definitions up to `fl1577`/`pr1002`.
- Little, J. D. C. (1961). A Proof for the Queuing Formula: $L = \lambda W$. *Operations
  Research*, 9(3), 383–387.
