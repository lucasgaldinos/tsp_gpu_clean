---
title: "GPU Distance Matrix vs On-the-Fly Coordinates and Shared Memory Limits in 2-Opt"
description: >
  Technical analysis of the cache-vs-ALU Roofline crossover between precomputed
  (n, n) distance matrices and on-the-fly coordinate distance calculation on GPU,
  together with the 48 KB shared-memory ceiling (n <= 11,520) in block-per-tour
  2-opt and the Shared-Memory Tiling and Candidate-List remedies.
created: 2026-11-03
status: review
author:
  - "[[Lucas Galdino]]"
  - "[[Claude Opus 4.6]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Created comprehensive knowledge-base guide covering why precomputed
          matrices are faster for n <= 10,000 while on-the-fly coordinates win
          for n > 10,000, why uint16 matrices cause overflow/divergence, why the
          48 KB shared-memory limit caps s_tour at n = 11,520, and how
          shared-memory tiling and candidate lists overcome both limits.
      - description: >
          Expanded Section 3 with the provenance of Shared-Memory Tiling (Rocki &
          Suda, 2012; O'Neil & Burtscher, 2015; Nyland et al., 2007), visual KaTeX
          block-matrix formulation, register-to-shared-memory result storage
          lifecycle, lossless j-read optimizations (register carry-over and d_ab
          hoisting), and Section 4 on k-NN Candidate Lists across TSP and CVRP.
      - description: >
          Added didactic code analogies for matrix partitioning into B x T tiles,
          demystified the outer-product (x) notation via Python/CUDA nested loops,
          documented the 4-step sliding window lifecycle, and clarified the strictly
          single segment reversal per pass rule in best-improvement 2-opt.
  - date_modified: 2026-11-04
    modifications:
      - description: >
          Added Section 3.6 explaining the SM occupancy trade-off behind
          TILE_SIZE = 2048 (96 KB per SM vs 48 KB per block on Pascal CC 6.1 and
          why TILE_SIZE must be a multiple of blockDim.x rather than a power of 2),
          Section 3.7 on CUDA block decomposition when data must be shared across
          routes (intra-route TSP vs inter-route CVRP 2-opt* vs 2-phase global
          reduction), and Section 4.5 on extending inverse position maps pos[u] to
          sparse CVRP sub-routes (BUG-19).
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Resolved user observations in Sections 3.6 and 3.7: corrected T=4096 SM occupancy
          row to 50.0% (4 blocks) due to CC 6.1 256-byte hardware allocation granularity,
          fixed barrier count to 3 __syncthreads() per window (two_opt_batch.cu#L235-L261),
          added concise occupancy formulas and derivation of T* = 2048, replaced the two
          note callouts with clean prose and cross-links to analysis-gpu-memory-latency-and-occupancy.md,
          streamlined Section 3.7 with a cross-link to analysis-gpu-intra-inter-route-regimes.md,
          and updated related_files.
related_files:
  - "[analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md)"
  - "[analysis-gpu-intra-inter-route-regimes.md](./analysis-gpu-intra-inter-route-regimes.md)"
  - "[analysis-gpu-memory-calculation.md](./analysis-gpu-memory-calculation.md)"
  - "[analysis-strategy-hierarchy-and-context-contracts.md](./analysis-strategy-hierarchy-and-context-contracts.md)"
  - "[analysis-gpu-batch.md](../../documents_tcc/plan/knowledge-base/analysis-gpu-batch.md)"
  - "[analysis-future-scaling-and-type-decisions.md](../analysis/questions/analysis-future-scaling-and-type-decisions.md)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - analysis/performance
  - analysis/memory
  - analysis/scaling
  - algorithm/two-opt
  - algorithm/genetic-algorithm
  - benchmark/gpu
  - guide/architecture
  - notes/cuda
  - notes/vram
  - review/quality
---

# GPU Distance Matrix vs On-the-Fly Coordinates and Shared Memory Limits in 2-Opt

- [GPU Distance Matrix vs On-the-Fly Coordinates and Shared Memory Limits in 2-Opt](#gpu-distance-matrix-vs-on-the-fly-coordinates-and-shared-memory-limits-in-2-opt)
  + [1. Why `uint16` Matrices Are Fragile and On-the-Fly Coordinates Win at Scale](#1-why-uint16-matrices-are-fragile-and-on-the-fly-coordinates-win-at-scale)
    - [1.1 The Problem with `uint16` (`0..65,535`)](#11-the-problem-with-uint16-065535)
    - [1.2 Why Precomputed `int32[n, n]` Is Faster for $n \\le 10,000$, While On-the-Fly `float2 coords[n]` Is Faster for $n \> 10,000$](#12-why-precomputed-int32n-n-is-faster-for-n-le-10000-while-on-the-fly-float2-coordsn-is-faster-for-n--10000)
  + [2. Blast Radius for Adding On-the-Fly CUDA Pairwise Distance Kernels](#2-blast-radius-for-adding-on-the-fly-cuda-pairwise-distance-kernels)
  + [3. Deep Dive: The $48\\text{ KB}$ Shared-Memory Ceiling ($n\_{\\max} = 11,520$) and How to Overcome It](#3-deep-dive-the-48text-kb-shared-memory-ceiling-n_max--11520-and-how-to-overcome-it)
    - [3.1 Clarifying Mechanism 1 vs Mechanism 2 When $n \> 11,520$](#31-clarifying-mechanism-1-vs-mechanism-2-when-n--11520)
    - [3.2 Visual Mathematical Explanation Using KaTeX and Block Matrices](#32-visual-mathematical-explanation-using-katex-and-block-matrices)
      + [How Tiling Partitions $\\mathbf{\\Delta}$ into $B \\times T$ Rectangular Sub-Blocks](#how-tiling-partitions-mathbfdelta-into-b-times-t-rectangular-sub-blocks)
        - [Translating the Outer-Product ($\\otimes$) to Code](#translating-the-outer-product-otimes-to-code)
        - [The 4-Step Sliding Window Lifecycle (Rocki \& Suda, 2012; O'Neil \& Burtscher, 2015)](#the-4-step-sliding-window-lifecycle-rocki--suda-2012-oneil--burtscher-2015)
    - [3.3 Where Are the Results Stored During Tiling? (Zero Global Memory Overhead)](#33-where-are-the-results-stored-during-tiling-zero-global-memory-overhead)
    - [3.4 Can We Safely Reduce the Number of `j` Reads Without Changing the Exact 2-Opt Result?](#34-can-we-safely-reduce-the-number-of-j-reads-without-changing-the-exact-2-opt-result)
    - [3.6 Why `TILE_SIZE = 2048` ($\\approx 8\\text{ KB}$) Instead of Near the $48\\text{ KB}$ Ceiling ($\\approx 11{,}000$)?](#36-why-tile_size--2048-approx-8text-kb-instead-of-near-the-48text-kb-ceiling-approx-11000)
      + [3.6.1 Per-Block Ceiling ($48\\text{ KB}$) vs Per-SM Budget ($96\\text{ KB}$)](#361-per-block-ceiling-48text-kb-vs-per-sm-budget-96text-kb)
      + [3.6.2 Must `TILE_SIZE` Be a Power of $2$ ($2^k$)?](#362-must-tile_size-be-a-power-of-2-2k)
    - [3.7 CUDA Block Decomposition When Data Must Be Shared: Intra-Route TSP vs Inter-Route CVRP ($2\\text{-opt}^\*$) vs Single Huge Tour](#37-cuda-block-decomposition-when-data-must-be-shared-intra-route-tsp-vs-inter-route-cvrp-2text-opt-vs-single-huge-tour)
      + [3.7.1 Why Standard CUDA Kernels Do Not Synchronize Across Blocks](#371-why-standard-cuda-kernels-do-not-synchronize-across-blocks)
      + [3.7.2 How to Partition Blocks \& Threads Across the 3 Routing Regimes](#372-how-to-partition-blocks--threads-across-the-3-routing-regimes)
  + [4. Breaking the $\\Theta(n^2)$ Barrier: $k$-Nearest Neighbor ($k$-NN) Candidate Lists](#4-breaking-the-thetan2-barrier-k-nearest-neighbor-k-nn-candidate-lists)
    - [4.1 Candidate List vs Nearest-Neighbor Heuristic: What Is the Difference?](#41-candidate-list-vs-nearest-neighbor-heuristic-what-is-the-difference)
    - [4.2 How $k$-NN Candidate Lists Reduce 2-Opt from $\\Theta(n^2)$ to $\\Theta(n \\cdot k)$](#42-how-k-nn-candidate-lists-reduce-2-opt-from-thetan2-to-thetan-cdot-k)
    - [4.3 Where Else Can $\\mathcal{N}\_k(u)$ Be Used Beyond 2-Opt?](#43-where-else-can-mathcaln_ku-be-used-beyond-2-opt)
    - [4.4 Software Architecture: Is $k$-NN a New "Strategy Type" or a Shared Context Service?](#44-software-architecture-is-k-nn-a-new-strategy-type-or-a-shared-context-service)
      + [Inputs, Outputs, and Strategy Compatibility Matrix for $\\mathcal{N}\_k$](#inputs-outputs-and-strategy-compatibility-matrix-for-mathcaln_k)
    - [4.5 Using the Same Operators for TSP and CVRP: Intra-Route Sub-Routes (`BUG-19`) vs Inter-Route $2\\text{-opt}^\*$](#45-using-the-same-operators-for-tsp-and-cvrp-intra-route-sub-routes-bug-19-vs-inter-route-2text-opt)
      + [1. Intra-Route Operators Work on Both TSP and CVRP Once `pos[u]` Is Sized by Node ID (`BUG-19`)](#1-intra-route-operators-work-on-both-tsp-and-cvrp-once-posu-is-sized-by-node-id-bug-19)
      + [2. Why Inter-Route $2\\text{-opt}^\*$ (Potvin \& Rousseau, 1995) Is a Separate Operator Family](#2-why-inter-route-2text-opt-potvin--rousseau-1995-is-a-separate-operator-family)
  + [5. Academic References](#5-academic-references)

## 1. Why `uint16` Matrices Are Fragile and On-the-Fly Coordinates Win at Scale

### 1.1 The Problem with `uint16` (`0..65,535`)

While storing the distance matrix as `uint16` halves VRAM ($2 n^2$ bytes instead of $4 n^2$ bytes), it introduces three hazards:

1. **Range overflow on real TSPLIB instances**: Many TSPLIB95 instances have coordinates spanning $[0, 10^5]$ or $[0, 10^6]$ (e.g. `pla33810`, `d15112`, `usa13509`), producing pairwise distances well above $65,535$. Any edge weight $> 65,535$ silently wraps modulo $2^{16}$, corrupting tour costs and 2-opt deltas.
2. **In-kernel promotion required**: Even if individual edge weights fit in `uint16`, summing two edges (`old_dist = d1 + d2`) or accumulating tour cost requires immediate promotion to `int32` / `int64` in registers.
3. **Still $O(n^2)$ memory**: At $n = 30,000$, even `uint16` requires $1.8\text{ GB}$ of VRAM, which still overflows the GPU's $1\text{ MB}$ L2 cache by a factor of $1,800\times$.

### 1.2 Why Precomputed `int32[n, n]` Is Faster for $n \le 10,000$, While On-the-Fly `float2 coords[n]` Is Faster for $n > 10,000$

At first glance, it seems paradoxical that **computing** `sqrtf(dx*dx + dy*dy)` millions of times inside a kernel can be faster than **reading** a precomputed number `dist[u * n + v]` from an array. The reason lies in the GPU memory hierarchy on the GTX 1050 (Pascal GP107):

| Memory Tier | Capacity on GTX 1050 | Latency (approx.) | Bandwidth |
| :--- | :--- | :--- | :--- |
| **Registers** | $256\text{ KB}$ per SM | $1\text{ cycle}$ | $> 10,000\text{ GB/s}$ |
| **Shared Memory / L1 Cache** | $48\text{ KB}$ per block / $64\text{ KB}$ per SM | $\approx 20\text{–}30\text{ cycles}$ | $\approx 1,500\text{ GB/s}$ |
| **L2 Cache** | $1,024\text{ KB}$ ($1\text{ MB}$ total chip-wide) | $\approx 150\text{ cycles}$ | $\approx 400\text{ GB/s}$ |
| **Global VRAM (GDDR5)** | $4,096\text{ MB}$ ($4\text{ GB}$) | $\approx 400\text{–}600\text{ cycles}$ | $112\text{ GB/s}$ |

Now compare what happens during 1 pass of 2-opt (`two_opt_batch.cu`):

- **Case A: Small/Medium $n$ ($n \le 1,500$, e.g. `berlin52` to `pr1002`):**
  + The `int32` distance matrix is $4 n^2 \le 4\text{ MB}$ (and for $n \le 512$, $\le 1\text{ MB}$).
  + Because all blocks in the population access the same cities repeatedly, the active rows of `dist` stay hot in the **L2 cache** (or require minimal DRAM transactions).
  + Reading `dist[u * n + v]` takes **zero math instructions** — which is especially advantageous for `GEO` (which otherwise requires `cosf`, `acosf`, degree-minute conversion) and `ATT`.
- **Case B: Large $n$ ($n > 10,000$, e.g. `d15112` or `rl11849`):**
  + The `int32` distance matrix is $400\text{ MB}$ to $913\text{ MB}$ — **400 to 900 times larger than the $1\text{ MB}$ L2 cache**.
  + Moreover, inside `two_opt_batch.cu`, thread `tid` checks `node_i = s_tour[i]` against `node_j = s_tour[j]`. Because `s_tour` is a permutation of cities, `node_i` and `node_j` jump pseudo-randomly across $[0, n-1]$.
  + Consequently, the 32 threads in a warp access 32 completely different rows `dist[node_i * n + node_j]` scattered across $900\text{ MB}$ of DRAM (**uncoalesced, L2-cache-missing global memory reads** costing $\approx 400\text{–}600\text{ cycles}$ each!).
  + Now look at **On-the-Fly Coordinates (`float2 coords[n]`)**:
    - At $n = 15,112$, the entire coordinate array `(x, y)` in `float32` (`float2`) is $15,112 \times 8\text{ bytes} = \mathbf{120.9\text{ KB}}$.
    - **$120.9\text{ KB}$ fits 8 times over inside the $1\text{ MB}$ L2 cache!** Every coordinate read `coords[node_i]` and `coords[node_j]` is a guaranteed **L1/L2 cache hit** ($\approx 20\text{–}150\text{ cycles}$), and computing `dx*dx + dy*dy` + `sqrtf` + `(int)(d + 0.5f)` takes only $\approx 15\text{ cycles}$ on the GPU's floating-point units ([Rocki & Suda, 2012](#5-academic-references); [O'Neil & Burtscher, 2015](#5-academic-references)).

---

## 2. Blast Radius for Adding On-the-Fly CUDA Pairwise Distance Kernels

To avoid dead code while supporting both precomputed matrices ($n \le 10,000$ or `EXPLICIT`) and on-the-fly coordinate evaluation ($n > 10,000$), only the following files in `gpu_accelerated_clean` need to participate:

| File | Role in the On-the-Fly Distance Pipeline |
| :--- | :--- |
| `src/distances/kernels/pairwise_device.cuh` *(new header)* | Defines `enum EdgeWeightTypeCode` and `__device__ __forceinline__ int gpu_pairwise_distance(int u, int v, const int *dist, const double *coords, int n, int edge_type_code)` implementing `EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`, `CEIL_2D`, `ATT`, `GEO` on GPU registers when `dist == nullptr`, and returning `dist[u * n + v]` when `dist != nullptr`. |
| [gpu_helpers.py](../../src/utils/gpu_helpers.py#L14-L54) | Passes `-I<src/distances/kernels>` to `cp.RawKernel(..., options=...)` so `.cu` kernels can `#include "pairwise_device.cuh"`. |
| [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu#L26-L65) | Calls `gpu_pairwise_distance(...)` instead of raw `distances[from_node * n + to_node]`. |
| [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L38-L85) | Calls `gpu_pairwise_distance(...)` for `old_dist` and `new_dist`. |
| [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L98-L160) | Calls `gpu_pairwise_distance(...)` for candidate move evaluation. |
| [problem_context.py](../../src/protocols/problem_context.py#L295-L360) | Exposes `max_matrix_dimension: int = 10_000` (and `VRAMInsufficientError` check via `cp.cuda.Device().mem_info`): when `n <= max_matrix_dimension` or `edge_type == "EXPLICIT"`, provides `d_distances` (`cp.int32`); when `n > max_matrix_dimension`, keeps `d_distances = None` and provides `get_gpu_coordinates()`. |
| [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L120-L185) & [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L265-L335) | Passes either `(d_distances, nullptr)` or `(nullptr, d_coords)` to the kernel. |
| `tests/distances/test_unit_gpu_pairwise_device.py` *(new test)* | Verifies 100% integer parity between `src/distances/pairwise.py` (CPU) and `pairwise_device.cuh` (GPU) across all TSPLIB95 distance types, plus end-to-end `BatchGPUCostEvaluation` and `BatchGPUTwoOpt` in coordinate-only mode (`d_distances = None`). |

---

## 3. Deep Dive: The $48\text{ KB}$ Shared-Memory Ceiling ($n_{\max} = 11,520$) and How to Overcome It

### 3.1 Clarifying Mechanism 1 vs Mechanism 2 When $n > 11,520$

In [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L67-L85), look at the two nested loops inside the kernel:

```cpp
// Outer loop: each thread tid has a DIFFERENT index i = tid, tid + 256, ...
for (int i = tid; i < n - 2; i += block_size) {
    int node_i  = s_tour[i];       // <--- Read 1 (depends on i, different per thread)
    int node_i1 = s_tour[i + 1];   // <--- Read 2 (depends on i, different per thread)

    // Inner loop: all threads step j = i + 2 ... n - 1
    for (int j = i + 2; j < n; ++j) {
        int node_j  = s_tour[j];           // <--- Read 3 (executed n times per i!)
        int node_j1 = s_tour[(j + 1) % n]; // <--- Read 4 (executed n times per i!)
        ...
    }
}
```

Now let's answer your question: **"Wouldn't passing `s_tour` via global memory (`tour[j]`) incur a bottleneck because it consults global memory every time?"**

- **Look at how many times each read happens:**
  + `node_i = tour[i]` and `node_i1 = tour[i + 1]` are read **outside** the `j` loop — only **$O(n)$ times total**, and because `i = tid + k * block_size`, adjacent threads `tid = 0, 1, 2, ..., 31` read `tour[0], tour[1], ..., tour[31]`, which is a **100% coalesced global memory read** (all 32 threads served in a single 128-byte memory transaction!), stored in two registers `node_i` and `node_i1`.
  + The bottleneck is the **inner `j` loop** (`node_j = tour[j]`), which executes **$O(n^2)$ times**!
- **Why naive Global Memory in the `j` loop is slower than Shared Memory, and how Shared-Memory Tiling (Mechanism 2) fixes it completely:**
  1. **If you just replace `s_tour[j]` with global `tour[j]` (Naive Global Fallback):**
     Because thread `tid` starts its inner loop at `j = i + 2 = tid + 2`, thread 0 reads `tour[2]`, thread 1 reads `tour[3]`, ..., thread 31 reads `tour[33]`. That is a contiguous 128-byte cache line in L1/L2 cache! However, for $n = 15,112$, the tour is $60\text{ KB}$, which is larger than the $24\text{–}48\text{ KB}$ L1 cache per SM, so it falls back to L2 cache ($\approx 150\text{ cycles}$ instead of shared memory's $\approx 25\text{ cycles}$).
  2. **The Optimal Solution Used in GPU Literature: Shared-Memory Tour Tiling ([Rocki & Suda, 2012](#5-academic-references); [O'Neil & Burtscher, 2015](#5-academic-references)):**
     - **Where this pattern comes from:** In GPU computing, partitioning an all-pairs $\Theta(n^2)$ interaction loop into fixed-size shared-memory windows while holding the outer entity in registers was introduced for N-body gravitational simulation by [Nyland, Harris, & Prins (2007)](#5-academic-references) (*GPU Gems 3*, Chapter 31) and adapted specifically to upper-triangular TSP 2-opt tour segments by [Rocki & Suda (2012)](#5-academic-references) (Section III-B, *"Shared Memory Optimization / Chunking for Large Tours"*) and [O'Neil & Burtscher (2015)](#5-academic-references) (Section 3.3, *"Extended-Size Solvers"*).
     - Instead of trying to load the **entire** $n$-city tour into shared memory at once (which fails when $n > 11,520$), the kernel allocates a **fixed-size tile** in shared memory — `TILE_SIZE = 2048` ints ($8\text{ KB}$ of shared memory, which works for **any $n$ up to $n = 10,000,000$!**).

### 3.2 Visual Mathematical Explanation Using KaTeX and Block Matrices

Let a tour of length $n$ be the permutation vector $\boldsymbol{\pi} = (\pi_0, \pi_1, \dots, \pi_{n-1})^\top \in \{0, \dots, n-1\}^n$, with wrap-around $\pi_n \equiv \pi_0$. Each tour position $k \in \{0, \dots, n-1\}$ defines a directed tour edge:

$$
e_k = (\pi_k, \pi_{k+1}), \quad c(e_k) = d(\pi_k, \pi_{k+1})
$$

A 2-opt move on index pair $(i, j)$ with $0 \le i < i + 2 \le j < n$ (excluding $(i=0, j=n-1)$) replaces edges $e_i = (\pi_i, \pi_{i+1})$ and $e_j = (\pi_j, \pi_{j+1})$ with $(\pi_i, \pi_j)$ and $(\pi_{i+1}, \pi_{j+1})$, producing the cost change:

$$
\Delta_{i,j} = \bigl(d(\pi_i, \pi_j) + d(\pi_{i+1}, \pi_{j+1})\bigr) - \bigl(d(\pi_i, \pi_{i+1}) + d(\pi_j, \pi_{j+1})\bigr)
$$

Conceptually, evaluating one pass of steepest-descent 2-opt finds the minimum entry in the strict upper triangle of the $n \times n$ move matrix $\mathbf{\Delta} \in \mathbb{Z}^{n \times n}$:

$$
\mathbf{\Delta} =
\begin{bmatrix}
\text{—} & \text{—} & \Delta_{0,2} & \Delta_{0,3} & \cdots & \Delta_{0,n-2} & \text{—} \\
\text{—} & \text{—} & \text{—} & \Delta_{1,3} & \cdots & \Delta_{1,n-2} & \Delta_{1,n-1} \\
\vdots & \vdots & \ddots & \ddots & \ddots & \vdots & \vdots \\
\text{—} & \text{—} & \text{—} & \text{—} & \cdots & \text{—} & \Delta_{n-3,n-1} \\
\text{—} & \text{—} & \text{—} & \text{—} & \cdots & \text{—} & \text{—}
\end{bmatrix}
$$

#### How Tiling Partitions $\mathbf{\Delta}$ into $B \times T$ Rectangular Sub-Blocks

> [!NOTE]
> **Intuitive Mental Model — Breaking a Giant Matrix into Small Tiles:**
> In 2-opt, we must evaluate every candidate edge $(i, i+1)$ against every non-adjacent candidate edge $(j, j+1)$. This forms a giant virtual matrix of candidate move gains $\mathbf{\Delta}$ of dimension roughly $n \times n$ (specifically the upper-triangular pairs $\frac{n(n-3)}{2}$).
> For $n = 15{,}000$, materializing this full matrix would take $15{,}000 \times 15{,}000 \times 4\text{ B} = \mathbf{900\text{ MB}}$ of VRAM, and fitting the whole tour of 15,000 nodes at once would require $60\text{ KB}$, exceeding the GPU hardware ceiling of 48 KB of shared memory per block.
> **Tiling breaks the virtual $n \times n$ matrix into small rectangular tiles of dimension $B \times T$**:
>
> - Height $B = 256$ (the number of threads in our CUDA block, where each thread handles one row $i$).
> - Width $T = 2048$ (a window of 2048 tour nodes loaded into shared memory, occupying only $\approx 8\text{ KB}$).
> - Number of row strips: $K = \left\lceil \frac{n-2}{B} \right\rceil$.
> - Number of column tiles: $M = \left\lceil \frac{n}{T} \right\rceil$.
> The GPU visits these small $256 \times 2048$ tiles one by one, evaluates them on-the-fly, updates only the best move found so far in registers, and discards the rest. **Zero bytes of global memory are allocated for the matrix.**

Instead of materializing $\mathbf{\Delta}$ in memory, we partition the rows $i$ into horizontal strips of height $B = \text{block\_size} = 256$ (one row per CUDA thread `tid` $\in \{0, \dots, B-1\}$) and the columns $j$ into vertical tiles of width $T = \text{TILE\_SIZE} = 2048$:

$$
\mathbf{\Delta} =
\begin{bmatrix}
\mathbf{\Delta}^{(0,0)}_{B \times T} & \mathbf{\Delta}^{(0,1)}_{B \times T} & \cdots & \mathbf{\Delta}^{(0, M-1)}_{B \times T} \\
\mathbf{0} & \mathbf{\Delta}^{(1,1)}_{B \times T} & \cdots & \mathbf{\Delta}^{(1, M-1)}_{B \times T} \\
\vdots & \ddots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{\Delta}^{(K-1, M-1)}_{B \times T}
\end{bmatrix},
\quad
K = \left\lceil \frac{n-2}{B} \right\rceil,\;
M = \left\lceil \frac{n}{T} \right\rceil
$$

For each sub-block $\mathbf{\Delta}^{(p, q)}_{B \times T}$ (where row strip $p$ starts at $i_{\text{base}} = p B$ and column tile $q$ starts at $t_{\text{start}} = q T$):

1. **Row Vector in Registers ($B = 256$ edges, 1 per thread):**
   Each thread $\tau \in \{0, \dots, B-1\}$ holds its own row index $i = i_{\text{base}} + \tau$, its two tour endpoints, and its precomputed edge length in **private registers**:

   $$
   \mathbf{r}_\tau^{(\text{row})} = \bigl(\pi_i,\; \pi_{i+1},\; d_{ab} = d(\pi_i, \pi_{i+1})\bigr) \in \text{Registers}(\tau)
   $$

2. **Column Vector in Shared Memory ($T + 1 = 2049$ nodes, shared by all 256 threads):**
   All $B = 256$ threads cooperatively load the contiguous slice of $T + 1$ tour nodes from global memory into the $8\text{ KB}$ shared-memory buffer $\mathbf{s}_{\text{tile}}$:

   $$
   \mathbf{s}_{\text{tile}} = \bigl(\pi_{t_{\text{start}}},\; \pi_{t_{\text{start}}+1},\; \dots,\; \pi_{t_{\text{start}}+T}\bigr) \in \text{SharedMem}\quad (2049 \times 4\text{ B} \approx 8\text{ KB})
   $$

3. **Outer-Product Evaluation of the $B \times T$ Tile:**
   Thread $\tau$ streams $k \in \{0, \dots, T-1\}$ across $\mathbf{s}_{\text{tile}}$, computing column $j = t_{\text{start}} + k$ of row $i = i_{\text{base}} + \tau$:

   $$
   \underbrace{
   \begin{bmatrix}
   (\pi_{i_{\text{base}}+0},\, \pi_{i_{\text{base}}+1}) \\
   (\pi_{i_{\text{base}}+1},\, \pi_{i_{\text{base}}+2}) \\
   \vdots \\
   (\pi_{i_{\text{base}}+255},\, \pi_{i_{\text{base}}+256})
   \end{bmatrix}
   }_{256\text{ thread register pairs}}
   \otimes
   \underbrace{
   \begin{bmatrix}
   (s_{\text{tile}, 0},\, s_{\text{tile}, 1}) &
   \cdots &
   (s_{\text{tile}, T-1},\, s_{\text{tile}, T})
   \end{bmatrix}
   }_{2048\text{ shared-memory column edges}}
   \longrightarrow \mathbf{\Delta}^{(p,q)}_{256 \times 2048}
   $$

##### Translating the Outer-Product ($\otimes$) to Code

The mathematical notation $\otimes$ ("outer product") simply represents a parallel nested loop where each of the 256 threads matches its own edge against all 2048 edges in shared memory:

```python
# Conceptual python equivalent of what CUDA executes in parallel:
# Outer loop: 256 threads running simultaneously in hardware
# Thread tau (tid = tau) holds its own row edge i:
edge_i = (tour[i_base + tid], tour[i_base + tid + 1])

# Inner loop: Each thread streams through the 2048 nodes in the shared tile (s_tile)
for k in range(2048):
  j = t_start + k
  edge_j = (s_tile[k], s_tile[k + 1])  # loaded from fast 8 KB shared memory

  # Evaluate 2-opt delta between edge_i and edge_j:
  delta = (
      dist(edge_i[0], edge_j[0])
      + dist(edge_i[1], edge_j[1])
      - dist(edge_i[0], edge_i[1])
      - dist(edge_j[0], edge_j[1])
  )

  # Keep only the single best improvement found so far in private registers:
  if delta < my_best_delta:
    my_best_delta = delta
    my_best_i = i_base + tid
    my_best_j = j
```

##### The 4-Step Sliding Window Lifecycle (Rocki & Suda, 2012; O'Neil & Burtscher, 2015)

For each column tile $q \in \{0, \dots, M-1\}$:

1. **Step A (Cooperative Global-to-Shared Load):** All 256 threads collaboratively copy $2049$ nodes from global `tour[t_start ... t_start + 2048]` into `s_tile[0 ... 2048]` (only $\approx 8$ reads per thread).
2. **Step B (`__syncthreads()`):** A block barrier ensures that all 2049 nodes are resident in shared memory before evaluation begins.
3. **Step C (100% On-Chip Inner Loop):** Every thread evaluates its row against all 2048 columns reading exclusively from `s_tile` in shared memory and coordinates in L1/L2 cache (zero DRAM reads in the inner loop).
4. **Step D (Advance Window):** Advance $t_{\text{start}} \leftarrow t_{\text{start}} + 2048$ and repeat until the tour completes.

### 3.3 Where Are the Results Stored During Tiling? (Zero Global Memory Overhead)

A natural question is: *"Since we loop through tiles of size $2048$, where are the results stored? In global memory?"*

**No — intermediate deltas are never written to global memory or even shared memory during the tile loops!** Here is the exact 3-stage storage lifecycle:

1. **Stage 1 — Inside the $(i_{\text{base}}, t_{\text{start}}, j)$ loops (100% in Thread Registers):**
   Each thread $\tau \in \{0, \dots, 255\}$ maintains **only 3 scalar 32-bit registers** across the entire kernel execution:

   $$
   \bigl(\Delta^*_\tau,\; i^*_\tau,\; j^*_\tau\bigr) \leftarrow (0,\; -1,\; -1)
   $$

   Whenever thread $\tau$ evaluates a pair $(i, j)$ inside any tile $(p, q)$ and finds $\Delta_{i,j} < \Delta^*_\tau$, it overwrites those 3 registers in $1\text{ clock cycle}$:

   $$
   \text{if } \Delta_{i,j} < \Delta^*_\tau \implies (\Delta^*_\tau, i^*_\tau, j^*_\tau) \leftarrow (\Delta_{i,j}, i, j)
   $$

2. **Stage 2 — After all tiles finish (Write 1 winner per thread to $3\text{ KB}$ Shared Memory):**
   Once the outer and inner loops finish, each thread has reduced $\approx \frac{n(n-3)}{2 \times 256}$ candidate moves down to **1 single winner** in its registers. Only then does each thread write its 3 registers to the 3 shared-memory reduction arrays ($256 \times 4\text{ B} = 1\text{ KB}$ each, $3\text{ KB}$ total):

   ```cpp
   s_deltas[tid] = my_best_delta;
   s_swap_i[tid] = my_best_i;
   s_swap_j[tid] = my_best_j;
   __syncthreads();
   ```

3. **Stage 3 — Block Reduction & In-Place Tour Reversal in Global Memory:**
   A standard $\log_2(256) = 8$-step parallel reduction in `s_deltas[0..255]` places the overall tour-best move $(\Delta^*, i^*, j^*)$ into `s_deltas[0], s_swap_i[0], s_swap_j[0]`. If $\Delta^* < 0$, the 256 threads cooperatively reverse the sub-segment `tour[i* + 1 ... j*]` **in-place** in global memory `tours[tour_idx * n + ...]`.

> [!IMPORTANT] Key Takeaway: Exactly One Segment Reversal per Pass (Steepest Descent)
> *Does this mean only one change is made per pass?* **Yes, exactly one.**
> In standard Best-Improvement 2-Opt (steepest descent):
>
> 1. One full pass scans the entire combinatorial search space of all $\approx \frac{n(n-3)}{2}$ candidate edge pairs across all tiles.
> 2. The single move offering the most negative cost change ($\Delta^* < 0$) is selected as the global winner.
> 3. That single move reverses the sub-segment `tour[i* + 1 ... j*]`.
> 4. The pass terminates, and the next pass begins evaluation on the newly improved tour until no negative delta exists (local optimum reached).
>
> **Why only one change at a time?** Inverting a sub-segment alters the sequence and indices of all visited cities within that range. If two overlapping 2-opt swaps were executed simultaneously in the same pass (e.g., reversing indices $5 \dots 20$ and $12 \dots 35$), the second reversal would corrupt the topology of the first, rendering the precalculated cost delta completely invalid. *(Only specialized Multi-Move variants apply multiple moves per pass, and only after strictly verifying that their move intervals are mutually disjoint).*

### 3.4 Can We Safely Reduce the Number of `j` Reads Without Changing the Exact 2-Opt Result?

Yes! Even before introducing heuristic candidate lists, there are **three exact (100% lossless) optimizations** that stack together in both **GPU** and **CPU** 2-opt while guaranteeing the **exact same steepest-descent 2-opt move**:

1. **Optimization 1 — Upper-Triangle Loop Bounds ($j \ge i + 2$, $\approx 50\%$ fewer tiles/pairs scanned):**
   + **Why this is completely different from `np.triu_indices` in `matrix.py` (`BUG-13`):**
     - In [matrix.py](../../src/distances/matrix.py#L126), we rejected `np.triu_indices(n, k=1)` for constructing/storing the **Distance Matrix $\mathbf{D} \in \mathbb{Z}^{n \times n}$**, because `np.triu_indices` allocates two huge 64-bit index arrays in RAM ($8 n^2$ bytes), requires slow scatter indexing, and a 1D packed triangle breaks asymmetric (`ATSP`) distance lookups $d(u, v) \ne d(v, u)$.
     - Here in [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L71) and [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L77), we are indexing the **2-Opt Move Matrix $\mathbf{\Delta}_{i,j}$** (pairs of tour positions $(i, j)$ to reverse), **not** the distance matrix $\mathbf{D}$! Reversing the tour segment between positions $(i, j)$ is the **exact same move** as reversing between $(j, i)$, and adjacent edges ($j \le i + 1$) cannot form a valid 2-opt swap.
     - Starting a `for` loop at `for (int j = i + 2; j < n; ++j)` (or starting the tile loop at `for (int tile_start = i_base + 2; tile_start < n; tile_start += 2048)`) **allocates zero index arrays** and **performs zero triangular index math** — it simply initializes the loop counter at `i + 2` (or `i_base + 2`), skipping all lower-triangular blocks $\mathbf{\Delta}^{(p, q)}$ with $q T + T \le p B + 2$ for free!

2. **Optimization 2 — Register / Variable Carry-Over of `node_j = node_j1` ($50\%$ fewer tour reads in the `j` loop):**
   In the naive inner loop:

   ```cpp
   int node_j  = s_tour_tile[local_j];
   int node_j1 = s_tour_tile[local_j + 1];
   ```

   Every iteration $j$ performs **two** tour reads (`s_tour_tile[local_j]` and `s_tour_tile[local_j + 1]` on GPU, or `improved[j]` and `improved[(j + 1) % n]` on CPU). Yet for consecutive iterations $j$ and $j + 1$, the start node of edge $e_{j+1}$ is the **exact same city** as the end node of edge $e_j$ ($\pi_{j+1}$)!

   By priming `int node_j = s_tour_tile[j_start - tile_start];` once before the `j` loop and advancing `node_j = node_j1;` in a register at the end of each step, each iteration of `j` performs **only 1 tour read** — cutting tour read traffic in half on both GPU and CPU!

3. **Optimization 3 — Hoisting `(node_i, node_i1, d_ab)` Outside the `j` Loop ($25\%$ fewer distance lookups):**
   In $\Delta_{i,j} = d(\pi_i, \pi_j) + d(\pi_{i+1}, \pi_{j+1}) - d(\pi_i, \pi_{i+1}) - d(\pi_j, \pi_{j+1})$, the endpoints $(\pi_i, \pi_{i+1})$ and their edge weight $d_{ab} = d(\pi_i, \pi_{i+1})$ depend **only on $i$**, never on $j$.
   + In [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L80-L83), `a, b = int(improved[i]), int(improved[i + 1])` and `distances[a, b]` were previously evaluated **inside** the `for j` loop ($n$ times per $i$ instead of once per $i$!).
   + Hoisting `a, b`, row slices `dist_a = distances[a]`, `dist_b = distances[b]`, and `d_ab = int(dist_a[b])` outside the `j` loop (combined with `c = d` carry-over from Optimization 2) speeds up CPU 2-opt by **$1.91\times$** ($137.9\text{ ms} \to 72.1\text{ ms}$ on $n = 150$, $10$ passes) with $100\%$ identical output!

```cpp
// Complete Register-Optimized Shared-Memory Tiling Pattern (Exact 2-Opt)
__shared__ int s_tour_tile[2049]; // 2048 + 1 for wrap-around (8 KB)

int my_best_delta = 0;
int my_best_i = -1;
int my_best_j = -1;

for (int i_base = 0; i_base < n - 2; i_base += block_size) {
    int i = i_base + tid;
    int node_i  = -1;
    int node_i1 = -1;
    int d_ab    = 0;
    if (i < n - 2) {
        node_i  = tour[i];
        node_i1 = tour[i + 1];
        d_ab    = gpu_pairwise_distance(node_i, node_i1, dist, coords, n, edge_type);
    }

    for (int tile_start = i_base + 2; tile_start < n; tile_start += 2048) {
        int tile_len = min(2049, n - tile_start);
        for (int t = tid; t < tile_len; t += block_size) {
            s_tour_tile[t] = tour[tile_start + t];
        }
        if (tile_start + 2048 >= n && tid == 0) {
            s_tour_tile[n - tile_start] = tour[0]; // wrap-around (j+1)%n
        }
        __syncthreads();

        if (i < n - 2) {
            int j_start = max(i + 2, tile_start);
            int j_end   = min(n, tile_start + 2048);
            if (i == 0 && j_end == n) {
                j_end = n - 1; // skip adjacent wrap-around pair (0, n-1)
            }
            if (j_start < j_end) {
                // Prime node_j in register once before the inner loop:
                int node_j = s_tour_tile[j_start - tile_start];
                for (int j = j_start; j < j_end; ++j) {
                    int node_j1 = s_tour_tile[(j - tile_start) + 1]; // ONLY 1 shared-mem read!
                    int old_dist = d_ab + gpu_pairwise_distance(node_j, node_j1, dist, coords, n, edge_type);
                    int new_dist = gpu_pairwise_distance(node_i, node_j, dist, coords, n, edge_type)
                                 + gpu_pairwise_distance(node_i1, node_j1, dist, coords, n, edge_type);
                    int delta = new_dist - old_dist;
                    if (delta < my_best_delta) {
                        my_best_delta = delta;
                        my_best_i = i;
                        my_best_j = j;
                    }
                    node_j = node_j1; // Register carry-over for next j!
                }
            }
        }
        __syncthreads();
    }
}
// Write thread-best registers to 3 KB shared memory once for block reduction:
s_deltas[tid] = my_best_delta;
s_swap_i[tid] = my_best_i;
s_swap_j[tid] = my_best_j;
__syncthreads();
```

### 3.6 Why `TILE_SIZE = 2048` ($\approx 8\text{ KB}$) Instead of Near the $48\text{ KB}$ Ceiling ($\approx 11{,}000$)?

A natural question arises when looking at `_TILE_SIZE: int = 2048` in [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L51) and `#define TWO_OPT_TILE_SIZE 2048` in [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L52):

> *"If a CUDA thread block can allocate up to $48\text{ KB}$ ($11{,}520\text{ ints}$) of dynamic shared memory, why use a tile of only $2{,}048\text{ ints}$ ($\approx 8\text{ KB}$) instead of a tile just below the maximum (say, $11{,}000\text{ ints}$)? And does `TILE_SIZE` have to be a power of $2$ ($2^k$)?"*

The answer comes from two hardware rules on NVIDIA GPUs: **Streaming Multiprocessor (SM) Occupancy** and **Warp-Coalesced Memory Transactions**.

#### 3.6.1 Per-Block Ceiling ($48\text{ KB}$) vs Per-SM Budget ($96\text{ KB}$)

On the GTX 1050 (Pascal architecture, Compute Capability 6.1, GP107), each Streaming Multiprocessor (SM) has:

- **Maximum active threads per SM:** $2{,}048\text{ threads}$ ($= 8\text{ blocks}$ of $B = 256\text{ threads}$).
- **Maximum dynamic shared memory per single block:** $48\text{ KB}$ ($49{,}152\text{ B}$).
- **Total shared memory per SM (shared across ALL simultaneously resident blocks on that SM):** $96\text{ KB}$ ($98{,}304\text{ B}$).

Now compare what happens on each SM when we set $T = 11{,}000$ versus $T = 2{,}048$:

$$
\text{SharedMem}_{\text{block}}(T) = \underbrace{3 \times 256 \times 4\text{ B}}_{\text{reduction arrays } (3{,}072\text{ B})} + \underbrace{(T + 1) \times 4\text{ B}}_{\text{tour tile } s\_\text{tour\_tile}}
$$

| Tile Width $T$ | Per-Block Shared Memory | Max Resident Blocks per SM ($\lfloor 96\text{ KB} / S_{\text{alloc}} \rfloor$) | Active Threads per SM (out of $2{,}048$) | Theoretical SM Occupancy | Global Tour Read Reduction Ratio ($B : 1$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$T = 11{,}000$** *(near $48\text{ KB}$ cap)* | $3{,}072 + 44{,}004 = \mathbf{47{,}076\text{ B}}$ ($S_{\text{alloc}} = 47{,}360\text{ B}$) | $\left\lfloor \frac{98{,}304}{47{,}360} \right\rfloor = \mathbf{2\text{ blocks}}$ | $2 \times 256 = \mathbf{512\text{ threads}}$ | **$25.0\%$** *(75% of warp slots idle!)* | $256 : 1$ ($99.61\%$ reuse) |
| **$T = 4{,}096$** | $3{,}072 + 16{,}388 = \mathbf{19{,}460\text{ B}}$ ($S_{\text{alloc}} = 19{,}712\text{ B}$) | $\left\lfloor \frac{98{,}304}{19{,}712} \right\rfloor = \mathbf{4\text{ blocks}}$ | $4 \times 256 = \mathbf{1{,}024\text{ threads}}$ | **$50.0\%$** *(5 blocks overflow SM by 256 B!)* | $256 : 1$ ($99.61\%$ reuse) |
| **$T = 2{,}048$** *(chosen sweet spot)* | $3{,}072 + 8{,}196 = \mathbf{11{,}268\text{ B}}$ ($S_{\text{alloc}} = 11{,}520\text{ B}$) | $\left\lfloor \frac{98{,}304}{11{,}520} \right\rfloor = \mathbf{8\text{ blocks}}$ | $8 \times 256 = \mathbf{2{,}048\text{ threads}}$ | **$100.0\%$** *(full latency hiding!)* | $256 : 1$ ($99.61\%$ reuse) |

- **Why $25\%$ Occupancy Degrades Throughput:** When only 2 blocks fit on an SM ($T = 11{,}000$), the SM hosts only 16 active warps divided across 4 warp schedulers (4 warps per scheduler). Whenever warps hit a `__syncthreads()` barrier or stall waiting on L2/DRAM memory fetches ($200\text{--}800$ clock cycles), the schedulers run out of eligible ready warps, leaving the execution ALUs starved and completely idle.
- **The $B : 1 = 256 : 1$ Global Tour Read Reduction and $99.61\%$ Reuse Ratio:** In naive 2-opt without shared memory, every thread in the block would read its target node $j$ directly from global VRAM DRAM, generating $B \times T = 256 \times T$ separate global memory read transactions. With cooperative shared-memory tiling, the $B = 256$ threads collaborate to read the window of length $T$ exactly once ($1 \times T$ global read transactions) into fast L1/Shared SRAM (`s_tour`), from which all 256 threads subsequently evaluate candidate edges. This reduces global DRAM read traffic by a factor of $B : 1 = 256 : 1$, eliminating $\frac{B - 1}{B} = \frac{255}{256} = \mathbf{99.61\%}$ of global tour memory traffic.
- **Three `__syncthreads()` Barriers per Window:** Verified in [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L235-L261), each sliding window encounters exactly **three** hardware barrier synchronizations:
  1. `Line 235`: Pre-load barrier preventing Write-After-Read (WAR) hazards on `s_tour` from the preceding window;
  2. `Line 244`: Post-load barrier preventing Read-After-Write (RAW) hazards before threads evaluate deltas;
  3. `Line 261`: Post-compute barrier ensuring delta updates to `s_deltas` finish before the next window overwrites `s_tour`.

>[!tip] **Deep Architectural Analysis: Memory Latency and Warp Scheduling**
> For physical clock cycle calculations ($\tau \approx 0.71\text{ ns}$), memory hierarchy tier physics (Register File vs SRAM vs L2 vs GDDR5), Little's Law, and SIMT warp scheduler starvation mechanics, see the dedicated knowledge-base guide: [analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md).

#### 3.6.2 Must `TILE_SIZE` Be a Power of $2$ ($2^k$)?

**No.** `TILE_SIZE` does **not** need to be a power of $2$, because tree reduction (`s >>= 1`) operates over `s_deltas[blockDim.x]` (which depends on `blockDim.x = 256`, **not** on `TILE_SIZE`).

Instead, `TILE_SIZE` should be an **exact integer multiple of `blockDim.x` ($256$)** (which is automatically a multiple of the $32$-thread warp size):

$$
T = m \times \text{blockDim.x} = m \times 256 \qquad (m \in \mathbb{Z}^+, \text{ e.g., } 1{,}536 = 6 \times 256,\; 2{,}048 = 8 \times 256,\; 2{,}560 = 10 \times 256)
$$

When $T = m \times 256$, the cooperative load loop:

```cpp
for (int idx = window_start + tid; idx < window_end; idx += block_size) {
    s_tour[idx - window_start] = tour[idx];
}
```

executes **exact $m$ full iterations across all $256$ threads** with:

1. **Zero warp divergence** on full tiles (every thread in every warp executes the exact same branch count $m$); and
2. **100% coalesced 128-byte global memory transactions** ($32\text{ threads/warp} \times 4\text{ B} = 128\text{ B}$ per warp per iteration).

##### Mathematical Derivation of Unique Optimum $T^* = 2{,}048$

To maximize theoretical SM occupancy to $100.0\%$ (8 resident blocks of 256 threads on Pascal CC 6.1), the allocated shared memory per block cannot exceed:

$$S_{\text{alloc}}(T) \le \left\lfloor \frac{S_{\text{SM}}}{8} \right\rfloor = \left\lfloor \frac{98{,}304\text{ B}}{8} \right\rfloor = 12{,}288\text{ B}$$

Given reduction buffer overhead $S_{\text{overhead}} = 3 \times 256 \times 4\text{ B} = 3{,}072\text{ B}$:

$$\left\lceil \frac{3{,}072 + 4(T + 1)}{256} \right\rceil \times 256 \le 12{,}288 \implies T \le 2{,}303$$

To eliminate warp divergence during the cooperative load loop and ensure 100% coalesced 128-byte DRAM transactions, $T$ must be an exact integer multiple of the block dimension:

$$T = m \times \text{blockDim.x} = m \times 256 \qquad (m \in \mathbb{Z}^+)$$

Evaluating the bound:

$$m \le \frac{2{,}303}{256} \approx 8.996 \implies m^* = 8 \implies \mathbf{T^* = 8 \times 256 = 2{,}048}$$

>[!tip] **Complete Occupancy Formulation and Allocation Granularities**
> For the comprehensive derivation including 256-register allocation chunks, Little's Law latency-hiding constraints, and microarchitectural proofs, see [analysis-gpu-memory-latency-and-occupancy.md#6-mathematical-derivation-of-unique-optimum-t--2048](./analysis-gpu-memory-latency-and-occupancy.md#6-mathematical-derivation-of-unique-optimum-t--2048).

---

### 3.7 CUDA Block Decomposition When Data Must Be Shared: Intra-Route TSP vs Inter-Route CVRP ($2\text{-opt}^*$) vs Single Huge Tour

The header comment of [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L15) states:

> `No inter-block synchronization (tours are independent)`

What does **inter-block synchronization** mean in CUDA, and how does the block decomposition change if we move from intra-route TSP 2-opt to **inter-route CVRP operators** (such as $2\text{-opt}^*$ or Relocate) or to optimizing a **single gigantic tour** ($P = 1, n > 50{,}000$)?

#### 3.7.1 Why Standard CUDA Kernels Do Not Synchronize Across Blocks

In CUDA's hardware model:

- **Intra-block (`__syncthreads()`):** All $256$ threads of a single block (`blockIdx.x = b`) execute on the **same SM**, share the **same $48\text{ KB}$ shared memory**, and synchronize in hardware in $\sim 20$ clock cycles.
- **Inter-block (across `blockIdx.x = 0, 1, ...`):** Blocks are independent work units dispatched to any SM in any order (and when $P$ exceeds the GPU's concurrent block capacity, later blocks do not even start executing until earlier blocks finish!). Therefore, **Block $0$ cannot read Block $1$'s shared memory**, and synchronising across blocks requires either ending the kernel launch (implicit grid barrier) or using CUDA Cooperative Groups (`cg::this_grid().sync()`).

#### 3.7.2 How to Partition Blocks & Threads Across the 3 Routing Regimes

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ REGIME 1: Population-Based Intra-Route TSP (Current `two_opt_batch.cu`, P >= 64)       │
│   Grid = P blocks  │  1 Block = 1 Individual Tour  │  256 Threads scan (i, j) pairs   │
│   -> Tours p_0 and p_1 never interact => ZERO inter-block communication needed!       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 2A: Population-Based Inter-Route CVRP 2-opt* (GA Population, P >= 64)           │
│   Grid = P blocks  │  1 Block = 1 Complete CVRP Individual (all m vehicle routes!)    │
│   -> Shared memory holds giant tour + route_id[n] + prefix_load[n].                   │
│   -> Threads evaluate inter-route pairs (i in R_a, j in R_b, a != b) inside the SAME  │
│      block => STILL ZERO inter-block communication needed!                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 2B: Single-Solution Inter-Route CVRP 2-opt* (P = 1, m vehicles)                 │
│   Grid = m(m-1)/2 blocks  │  1 Block = 1 Route Pair (R_a, R_b)                        │
│   -> Phase 1 Kernel: Each block finds best 2-opt* swap between R_a and R_b -> VRAM.   │
│   -> Phase 2 Kernel (1 block): Resolves route conflicts & applies disjoint swaps.     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 3: Single Huge Tour (P = 1, n > 50,000)                                         │
│   Grid = G blocks  │  1 Block = 1 Row Stripe [i_start .. i_end) of the SAME tour      │
│   -> Phase 1 Kernel: Block b scans its row stripe and writes block_best[b] to VRAM.   │
│   -> Phase 2 Kernel (1 block): Reduces block_best[0..G-1] and reverses best [i+1..j]. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Notice the critical insight in **Regime 2A**:

- In a Genetic Algorithm for CVRP with population size $P = 256$, **each individual $p \in \{0, \dots, P-1\}$ is a complete CVRP solution** containing $m$ vehicle routes $R_1, R_2, \dots, R_m$ (total length $\approx n + m$).
- While route $R_a$ and route $R_b$ **inside** individual $p$ must exchange customers during inter-route $2\text{-opt}^*$, **individual $p_0$ and individual $p_1$ are still 100% independent!**
- Therefore, by assigning **1 CUDA block per CVRP individual** (`gridDim.x = P`), all $m$ routes of that individual reside together in the **same block's shared memory**, and threads inside that block evaluate inter-route moves $(i \in R_a, j \in R_b)$ without any inter-block synchronization! Only when optimizing a single solution ($P = 1$, Regimes 2B and 3) do we split a single solution across multiple blocks using a 2-phase kernel launch (where the boundary between Kernel 1 and Kernel 2 acts as a zero-cost global barrier).

>[!tip] **Dedicated Architecture Guide: Intra- vs Inter-Route Regimes**
> For full architectural specifications, prefix-sum $O(1)$ capacity feasibility checks, and two-phase conflict-resolution CUDA kernels for Regimes 2A, 2B, and 3, see the dedicated knowledge-base document: [analysis-gpu-intra-inter-route-regimes.md](./analysis-gpu-intra-inter-route-regimes.md).

---

## 4. Breaking the $\Theta(n^2)$ Barrier: $k$-Nearest Neighbor ($k$-NN) Candidate Lists

### 4.1 Candidate List vs Nearest-Neighbor Heuristic: What Is the Difference?

In routing literature, "$k$-Nearest Neighbor" appears in two distinct roles that should not be conflated:

1. **Nearest-Neighbor Construction Heuristic ([Rosenkrantz, Stearns, & Lewis, 1977](#5-academic-references)):**
   A greedy **tour-building algorithm** (already implemented in your codebase as `NearestNeighborInitialization`) that starts at a depot/city $v_0$ and repeatedly appends the closest unvisited city $v_{t+1} = \arg\min_{u \notin \text{visited}} d(v_t, u)$ in $\Theta(n^2)$ time.
2. **$k$-Nearest Neighbor ($k$-NN) Candidate List ([Reinelt, 1994](#5-academic-references); [Johnson & McGeoch, 1997](#5-academic-references); [Helsgaun, 2000](#5-academic-references)):**
   A **static, problem-level sparse neighborhood structure** $\mathcal{N}_k \in \mathbb{Z}^{n \times k}$ computed **once** when the problem is loaded into `ProblemContext`. For each node $u \in \{0, \dots, n-1\}$, row $u$ stores the indices of the $k$ closest nodes to $u$ sorted by non-decreasing distance:

   $$
   \mathcal{N}_k(u) = \bigl[v_{u,1},\; v_{u,2},\; \dots,\; v_{u,k}\bigr], \quad
   \text{where } d(u, v_{u,1}) \le d(u, v_{u,2}) \le \dots \le d(u, v_{u,k}) \le \min_{w \notin \mathcal{N}_k(u) \cup \{u\}} d(u, w)
   $$

   Typical values in TSP/CVRP solvers are $k \in [15, 40]$ (default $k = 20$). At $n = 15{,}112$ and $k = 20$, the entire matrix $\mathcal{N}_{20} \in \mathbb{Z}^{15112 \times 20}$ (`int32`) occupies only:

   $$
   15{,}112 \times 20 \times 4\text{ bytes} = \mathbf{1.21\text{ MB}}
   $$

   + **Tiled / Row-Chunked Construction of $\mathcal{N}_k$ (Building $\mathcal{N}_k$ Without Ever Allocating the $n \times n$ Matrix):**
     If we constructed $\mathcal{N}_k$ at $n = 15{,}112$ by first materializing the full $15{,}112 \times 15{,}112$ distance matrix ($913.5\text{ MB}$) and then sorting every row, we would still pay the $\Theta(n^2)$ memory peak during preprocessing. Instead, we apply the **exact same row-chunking / tiling principle** (`chunk_size = 512` rows):
     1. For each strip of $B = 512$ nodes $u \in [i_{\text{base}}, i_{\text{base}} + B)$, compute only the rectangular distance strip $\mathbf{D}_{\text{strip}} \in \mathbb{Z}^{B \times n}$ against all $n$ coordinates ($512 \times 15{,}112 \times 4\text{ B} \approx \mathbf{30.9\text{ MB}}$ instead of $913.5\text{ MB}$!), setting the self-distance diagonal $d(u, u) = +\infty$.
     2. Select the $k$ smallest neighbor indices in $O(B \cdot n)$ linear time per strip using `xp.argpartition(d_strip, kth=k, axis=1)[:, :k]` and sort only those $k$ winners per row.
     3. Write the $B \times k$ block into $\mathcal{N}_k[i_{\text{base}} : i_{\text{base}} + B, :]$ and discard $\mathbf{D}_{\text{strip}}$. Peak memory is bounded by $O(B \cdot n + n \cdot k)$ ($\approx 32\text{ MB}$ at $n = 15{,}112$)!

### 4.2 How $k$-NN Candidate Lists Reduce 2-Opt from $\Theta(n^2)$ to $\Theta(n \cdot k)$

Why do we not need to check all $n$ positions $j$ in a tour?
Suppose we consider swapping edges $(\pi_i, \pi_{i+1})$ and $(\pi_j, \pi_{j+1})$ into $(\pi_i, \pi_j)$ and $(\pi_{i+1}, \pi_{j+1})$. For $\Delta_{i,j} < 0$, the sum of the two new edges must be strictly less than the sum of the two old edges:

$$
d(\pi_i, \pi_j) + d(\pi_{i+1}, \pi_{j+1}) < d(\pi_i, \pi_{i+1}) + d(\pi_j, \pi_{j+1})
$$

In any reasonable tour, at least one of the newly added edges — either $(\pi_i, \pi_j)$ or $(\pi_{i+1}, \pi_{j+1})$ — must be short, which means either $\pi_j \in \mathcal{N}_k(\pi_i)$ or $\pi_{j+1} \in \mathcal{N}_k(\pi_{i+1})$ ([Reinelt, 1994](#5-academic-references), Chapter 6; [Johnson & McGeoch, 1997](#5-academic-references), Section 3.1).

To evaluate only the $k$ candidates of $\pi_i$ in $O(k)$ time per city $i$, we maintain an **inverse tour permutation** $\text{pos} \in \{0, \dots, n-1\}^n$ defined by:

$$
\text{pos}[\pi_idx] = idx \quad \Longleftrightarrow \quad \pi_{\text{pos}[u]} = u \quad \forall u \in \{0, \dots, n-1\}
$$

Building $\text{pos}$ takes a single parallel pass of $O(n)$ writes (`pos[tour[i]] = i`). Then, for each tour index $i \in \{0, \dots, n-1\}$ (with city $u = \pi_i$):

1. Iterate only over the $r \in \{0, \dots, k-1\}$ nearest neighbors $v = \mathcal{N}_k(u)[r]$.
2. Look up where city $v$ currently sits in the tour in $O(1)$ time: $j = \text{pos}[v]$.
3. Evaluate the 2-opt delta $\Delta_{i,j}$ (and optionally the predecessor $j - 1$ corresponding to connecting $(\pi_{i+1}, v)$).

| Metric | Exhaustive 2-Opt | $k$-NN Candidate-List 2-Opt ($k = 20$) | Speedup / Reduction Factor |
| :--- | :--- | :--- | :--- |
| **Pairs evaluated per pass** | $\frac{n(n-3)}{2} = \Theta(n^2)$ | $n \cdot k = \Theta(n \cdot k)$ | $\frac{n - 3}{2k}$ |
| **Evaluations at $n = 1{,}002$ (`pr1002`)** | $499{,}500$ | $20{,}040$ | **$24.9\times$ fewer checks** |
| **Evaluations at $n = 15{,}112$ (`d15112`)** | $114{,}171{,}164$ | $302{,}240$ | **$377.8\times$ fewer checks** |

### 4.3 Where Else Can $\mathcal{N}_k(u)$ Be Used Beyond 2-Opt?

A precomputed $k$-NN candidate table `context.get_cpu_candidate_list(k)` / `context.get_gpu_candidate_list(k)` is **not limited to 2-opt**. It is a foundational building block across **five different algorithmic families** in combinatorial routing:

1. **Higher-Order $k$-Opt Local Search (3-Opt, Or-Opt, Lin-Kernighan / LKH):**
   + Exhaustive 3-opt requires $\Theta(n^3)$ checks per pass ($10^9$ checks at $n = 1{,}000$), which is intractable even on GPU.
   + By restricting the second and third broken edges to $\mathcal{N}_k(\pi_i)$ and $\mathcal{N}_k(\pi_j)$, 3-opt and Or-opt (relocating chains of $1, 2, 3$ consecutive cities) drop from $\Theta(n^3)$ to **$\Theta(n \cdot k^2)$** or **$\Theta(n \cdot k)$** ([Lin & Kernighan, 1973](#5-academic-references); [Helsgaun, 2000](#5-academic-references); [Rocki & Suda, 2012](#5-academic-references)).
2. **CVRP Intra-Route & Inter-Route Local Search (Granular Neighborhoods):**
   + In Capacitated Vehicle Routing (CVRP), local search requires inter-route operators (**Relocate / 1-0 Shift**, **Swap / 1-1 Exchange**, **2-Opt\***, **Cross-Exchange**).
   + Scanning all pairs of customers across all vehicle routes is $\Theta(n^2)$. [Toth & Vigo (2003)](#5-academic-references) (*Granular Tabu Search*) and [Vidal et al. (2012)](#5-academic-references) (*Hybrid Genetic Search, HGS-CVRP*) proved that restricting inter-route insertions of customer $u$ to positions adjacent to $v \in \mathcal{N}_k(u)$ (**granular neighborhoods**) preserves $>99.5\%$ of improving CVRP moves while reducing complexity to $\Theta(n \cdot k)$.
3. **Randomized Nearest-Neighbor / GRASP Population Initialization:**
   + Standard `NearestNeighborInitialization` is deterministic from a given start node and scans all $n$ nodes at each step.
   + Using $\mathcal{N}_k(u)$, a **GRASP / Restricted Candidate List (RCL)** initializer ([Feo & Resende, 1995](#5-academic-references)) immediately picks a random unvisited neighbor from $\mathcal{N}_k(u)$ in $O(k)$ time, generating diverse, high-quality initial populations orders of magnitude faster than random permutations.
4. **Edge-Preserving Crossovers (Order Crossover / PMX / Edge Recombination Fallback):**
   + When an offspring tour reaches a city $u$ whose parent neighbors are already visited, standard crossover falls back to the next sequential or random unvisited city (creating a long "jump" edge that increases tour cost). Consulting $\mathcal{N}_k(u)$ first selects an unvisited geographically close neighbor in $O(k)$ time.
5. **Guided / Proximity-Biased Mutation (`SwapMutation`, `InversionMutation`):**
   + Uniformly random `SwapMutation` or `InversionMutation` picks two random tour indices $(i, j) \in \{0, \dots, n-1\}^2$. In $>95\%$ of cases on medium/large instances, cities $\pi_i$ and $\pi_j$ are far apart across the map, so swapping or reversing between them injects two huge edges and produces a worse tour that selection immediately discards.
   + With a proximity-bias parameter `candidate_k: int | None = None`, mutation picks index $i$ uniformly at random, samples a neighbor city $v \in \mathcal{N}_k(\pi_i)$, and sets $j = \text{pos}[v]$ (or $j - 1$), focusing mutation directly on plausible short-edge reconnections!

### 4.4 Software Architecture: Is $k$-NN a New "Strategy Type" or a Shared Context Service?

Because $\mathcal{N}_k$ is **not a heuristic stage in itself** (it does not transform a population of routes on its own), it should **not** be a 7th strategy slot on `GeneticAlgorithm`. Instead:

1. **Shared Module + `ProblemContext` Service (`DP10-20`):**
   + The pure computation function `compute_candidate_list(problem_or_coords, k=20, xp=np, chunk_size=512)` lives in [src/distances/matrix.py](../../src/distances/matrix.py).
   + [ProblemContext](../../src/protocols/problem_context.py) exposes two lazy, cached accessors:
     - `context.get_cpu_candidate_list(k: int = 20) -> np.ndarray` (shape `(n, k_eff)`, `dtype=np.int32`)
     - `context.get_gpu_candidate_list(k: int = 20) -> CupyArray` (shape `(n, k_eff)`, `dtype=cp.int32`)
   + Because `ProblemContext` is already passed to **every** strategy method (`initialize(pop_size, context)`, `select(pop, fitness, context)`, `crossover(parents, context)`, `mutate(offspring, context)`, `improve_batch(routes, context)`, `evaluate_batch(routes, context)`), **any strategy can request `context.get_cpu_candidate_list(k)` or `context.get_gpu_candidate_list(k)` on demand without changing a single Protocol signature or hardcoding $k$-NN construction inside any strategy!**
2. **Opt-In via Strategy Parameter (`candidate_k: int | None = None`):**
   + Existing strategies simply accept `candidate_k: int | None = None` in `__init__` (configurable directly from `configs/algorithms.json` under `strategy_params`).
   + When `candidate_k is None`, the strategy runs its classic full-range / uniform behavior.
   + When `candidate_k = 20`, the strategy calls `context.get_cpu_candidate_list(20)` (or `get_gpu_candidate_list(20)`) and restricts its neighborhood lookup to the $k$ nearest neighbors.

#### Inputs, Outputs, and Strategy Compatibility Matrix for $\mathcal{N}_k$

- **Inputs to `compute_candidate_list`**:
  + Either a precomputed distance matrix $\mathbf{D} \in \mathbb{Z}^{n \times n}$ (`EXPLICIT` problems: `TSP`, `ATSP`, `SOP`, `CVRP`), **or** node coordinates $\mathbf{X} \in \mathbb{R}^{n \times d}$ + `edge_weight_type` (`EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`, `CEIL_2D`, `ATT`, `GEO`, `XRAY1`), plus integer $k \ge 1$.
  + *Note on Asymmetric Problems (`ATSP`):* Row $u$ of $\mathcal{N}_k^{\text{out}}(u)$ stores the $k$ smallest **outgoing** arcs $\arg\min_{v \ne u} d(u, v)$.
- **Output of `compute_candidate_list`**:
  + A 2D integer array $\mathcal{N}_k \in \{0, \dots, n-1\}^{n \times k_{\text{eff}}}$ (`int32`, where $k_{\text{eff}} = \min(k, n - 1)$), where $\mathcal{N}_k[u, r]$ is the $(r+1)$-th closest node to node $u$ ($0 \le r < k_{\text{eff}}$, $u \notin \mathcal{N}_k[u, :]$).

| Strategy Family | Can Use $\mathcal{N}_k$? | How It Uses $\mathcal{N}_k$ (`candidate_k: int \| None`) | When It CANNOT Be Used |
| :--- | :--- | :--- | :--- |
| **Local Search** (`CPUTwoOpt`, `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`, `BatchGPUTwoOpt`, `BatchGPUMultiMoveTwoOpt`) | **Yes (Primary Consumer)** | Builds inverse map `pos[tour[m]] = m` ($O(n)$) and checks only $j = \text{pos}[\mathcal{N}_k[\pi_i, r]]$ for $r \in [0, k_{\text{eff}})$, reducing pass time from $\Theta(n^2)$ to $\Theta(n \cdot k)$. | Unweighted graphs without distances (`HCP` with pure 0/1 adjacency where all non-adjacent distances are equal), or $n \le 4$ where $k_{\text{eff}} = n - 1$ equals exhaustive search. |
| **Mutation** (`CPUSwapMutation`, `GPUSwapMutation`, `CPUInversionMutation`, `GPUInversionMutation`) | **Yes (Proximity-Guided Mode)** | Picks first position $i$ uniformly at random, picks random $r \in [0, k_{\text{eff}})$, and sets second position $j = \text{pos}[\mathcal{N}_k[\pi_i, r]]$ so mutations reconnect nearby cities instead of random cross-map jumps. | If pure uniform exploration is desired (`candidate_k=None`), or on tiny tours ($n \le 4$). |
| **Initialization** (`NearestNeighborInitialization`, `RandomInitialization` / GRASP) | **Yes (RCL / Fast Greedy)** | Instead of scanning all $n$ nodes to find the closest unvisited node, inspects $\mathcal{N}_k[u, 0 \dots k_{\text{eff}}-1]$ in $O(k)$ time and only falls back to full scan if all $k$ neighbors are already visited. | Pure uniform `RandomInitialization` (`candidate_k=None`) intentionally ignores distances to maximize initial entropy. |
| **Crossover** (`CPUOrderCrossover`, `GPUOrderCrossover`, `PMXCrossover`) | **Optional (Fallback / Edge-Repair)** | Order Crossover (OX) and PMX are pure positional permutation operators (they preserve relative order from parents without looking at distances). $\mathcal{N}_k$ is used if extended to **Edge Recombination (EX)** or **greedy gap repair** when connecting sliced sub-segments. | Standard OX/PMX without edge repair do not consult distances. |
| **Selection** (`TournamentSelection`, `RouletteWheelSelection`) | **No** | Selection operates solely on scalar fitness values `fitness: float64[P]` (survival of the fittest) and never inspects city indices or distances. | Never applicable (selection is representation-agnostic). |
| **Evaluation** (`CPUCostEvaluation`, `BatchGPUCostEvaluation`) | **No** | Evaluation sums the exact closed-loop tour cost $\sum_{i=0}^{n-1} d(\pi_i, \pi_{i+1})$ along the already-constructed tour; it requires exact pairwise distances $d(\pi_i, \pi_{i+1})$, not neighbor rankings. | Never applicable ($\pi_{i+1}$ in an arbitrary tour is not guaranteed to be in $\mathcal{N}_k(\pi_i)$). |

---

### 4.5 Using the Same Operators for TSP and CVRP: Intra-Route Sub-Routes (`BUG-19`) vs Inter-Route $2\text{-opt}^*$

Can the exact same `2-opt`, `SwapMutation`, and `InversionMutation` implementations be used for both **TSP** and **CVRP**? The answer depends on the distinction between **intra-route** (within one vehicle's route) and **inter-route** (across two vehicles' routes) operators:

#### 1. Intra-Route Operators Work on Both TSP and CVRP Once `pos[u]` Is Sized by Node ID (`BUG-19`)

In a CVRP instance with $n = 22$ total nodes (`0` = depot, `1..21` = customers), suppose vehicle 1 serves a sub-route of length $m = 5$:

$$
R_1 = \bigl[0,\; 5,\; 12,\; 19,\; 0\bigr] \qquad (\text{length } m = 5, \text{ max node ID } = 19)
$$

Inside $R_1$:

- **Capacity invariance:** Reversing a customer segment or swapping two customers inside $R_1$ never changes the set of customers $\{5, 12, 19\}$ assigned to vehicle 1. Therefore, vehicle 1's total demand $\sum_{u \in R_1} q_u \le Q$ is **automatically preserved** without any capacity check!
- **Depot endpoint protection:** `_mutation_bounds(m)` returns `(1, m - 1) = (1, 4)`, and `2-opt` skips `(i == 0 and j == m - 1)`, so the start and end depots at indices `0` and `m - 1` never move.
- **The Inverse Position Map Trap (`BUG-19`) and Its Fix:**
  In a full TSP tour, `len(tour) == n` and the node IDs are a permutation of $\{0, \dots, n - 1\}$, so `positions = np.empty(len(tour), dtype=np.int32); positions[tour] = np.arange(len(tour))` is safe.
  However, on the CVRP sub-route $R_1 = [0, 5, 12, 19, 0]$, `len(R_1) == 5` while customer ID `19 >= 5`! Indexing `positions[19]` on an array of size $5$ raises `IndexError: index 19 is out of bounds for axis 0 with size 5`.
  + **Fix (`Task 10.20` / `BUG-19`):** Allocate `positions` to cover the largest node ID present (or problem dimension $n$), initialized to `-1` (meaning *"node not present in this vehicle's sub-route"*):

    ```python
    def inverse_positions(route: np.ndarray, num_nodes: int | None = None) -> np.ndarray:
      """Map each node ID `u` to its index in `route`, or `-1` if `u` is not in `route`."""
      if route.size == 0:
        return np.empty(0, dtype=np.int32)
      size = max(int(route.size), int(route.max()) + 1)
      if num_nodes is not None:
        size = max(size, int(num_nodes))
      positions = np.full(size, -1, dtype=np.int32)
      positions[route] = np.arange(route.size, dtype=np.int32)
      return positions
    ```

  + When scanning $v = \mathcal{N}_k[\pi_i, r]$ for a node $\pi_i \in R_1$, if `j = positions[v]` is `-1`, customer $v$ belongs to a *different* vehicle's route, so the intra-route operator simply skips $v$ (`if j <= i + 1: continue` already skips `j == -1`!). Thus, **one and the same function handles both full TSP tours and individual CVRP sub-routes**.

#### 2. Why Inter-Route $2\text{-opt}^*$ ([Potvin & Rousseau, 1995](#5-academic-references)) Is a Separate Operator Family

While intra-route `2-opt` optimizes each vehicle's tour independently, CVRP also requires **inter-route** operators to move customers between vehicles. The most important inter-route counterpart is **$2\text{-opt}^*$** ([Potvin & Rousseau, 1995](#5-academic-references); [Toth & Vigo, 2003](#5-academic-references)):

| Property | Intra-Route `2-opt` (Croes, 1958) | Inter-Route `2-opt*` (Potvin & Rousseau, 1995) |
| :--- | :--- | :--- |
| **Where edges are cut** | Two edges $(u_i, u_{i+1})$ and $(u_j, u_{j+1})$ inside the **same** route $R_a$. | One edge $(u_i, u_{i+1})$ in route $R_a$ and one edge $(v_j, v_{j+1})$ in a **different** route $R_b$ ($a \ne b$). |
| **How segments reconnect** | **Inverts** the middle segment: $(0, \dots, u_i, \mathbf{u_j, \dots, u_{i+1}}, u_{j+1}, \dots, 0)$. | **Swaps the two route tails without inversion** (preserving customer orientation):<br>$R_a' = (0, \dots, u_i, \; v_{j+1}, \dots, 0)$<br>$R_b' = (0, \dots, v_j, \; u_{i+1}, \dots, 0)$ |
| **Cost delta $\Delta$** | $d(u_i, u_j) + d(u_{i+1}, u_{j+1}) - d(u_i, u_{i+1}) - d(u_j, u_{j+1})$ | $d(u_i, v_{j+1}) + d(v_j, u_{i+1}) - d(u_i, u_{i+1}) - d(v_j, v_{j+1})$ |
| **Capacity check required?** | **No** (total route load is unchanged). | **Yes ($O(1)$ via prefix loads $L_a[i] = \sum_{t=0}^{i} q_{R_a[t]}$):**<br>$L_a[i] + \bigl(L_b[\text{end}] - L_b[j]\bigr) \le Q \quad \text{and} \quad L_b[j] + \bigl(L_a[\text{end}] - L_a[i]\bigr) \le Q$ |
| **Candidate list $\mathcal{N}_k$ lookup** | Checks $v \in \mathcal{N}_k(u_i)$ where `pos[v]` is in the **same** route. | Checks $v \in \mathcal{N}_k(u_i)$ where `route_of[v] != route_of[u_i]` (only neighbors belonging to **other** routes!). |

---

## 5. Academic References

1. **Croes, G. A. (1958).** *A method for solving traveling-salesman problems.* Operations Research, 6(6), 791–812. (Introduced the 2-opt edge-exchange heuristic).
2. **Lin, S., & Kernighan, B. W. (1973).** *An effective heuristic algorithm for the traveling-salesman problem.* Operations Research, 21(2), 498–516. (Introduced candidate-list pruning for variable-$k$-opt).
3. **Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977).** *An analysis of several heuristics for the traveling salesman problem.* SIAM Journal on Computing, 6(3), 563–581. (Formal analysis of the Nearest-Neighbor construction heuristic).
4. **Reinelt, G. (1994).** *The Traveling Salesman: Computational Solutions for TSP Applications.* Lecture Notes in Computer Science, Vol. 840, Springer-Verlag. (Formalized $k$-nearest neighbor candidate lists, quadrant neighbor lists, and inverse tour position arrays reducing 2-opt from $O(n^2)$ to $O(n \cdot k)$).
5. **Feo, T. A., & Resende, M. G. (1995).** *Greedy randomized adaptive search procedures.* Journal of Global Optimization, 6(2), 109–133. (Restricted Candidate Lists / RCL for randomized greedy initialization).
6. **Potvin, J.-Y., & Rousseau, J.-M. (1995).** *An exchange heuristic for routeing problems with time windows.* Journal of the Operational Research Society, 46(12), 1433–1446. (Introduced the inter-route $2\text{-opt}^*$ tail-exchange heuristic for vehicle routing).
7. **Johnson, D. S., & McGeoch, L. A. (1997).** *The traveling salesman problem: A case study in local optimization.* In *Local Search in Combinatorial Optimization* (pp. 215–310), Wiley. (Comprehensive analysis of first-improvement vs best-improvement 2-opt and $k$-NN candidate lists).
8. **Helsgaun, K. (2000).** *An effective implementation of the Lin-Kernighan traveling salesman heuristic.* European Journal of Operational Research, 126(1), 106–130. (Alpha-nearness and candidate-set pruning in LKH).
9. **Toth, P., & Vigo, D. (2003).** *The granular tabu search and its application to the vehicle-routing problem.* INFORMS Journal on Computing, 15(4), 333–346. (Granular $k$-NN neighborhoods for CVRP relocate, swap, and 2-opt\* operators).
10. **Nyland, L., Harris, M., & Prins, J. (2007).** *Fast N-body simulation with CUDA.* In *GPU Gems 3* (Chapter 31, pp. 677–695), Addison-Wesley. (Foundational CUDA shared-memory strip-mining / tiling pattern for all-pairs $\Theta(n^2)$ interactions).
11. **Fujimoto, N., & Tsutsui, S. (2011).** *A highly-parallel TSP solver for a GPU computing platform.* In International Conference on Numerical Analysis and Applied Mathematics (ICNAAM), Vol. 1389, pp. 264–267. (Block-per-tour GPU GA + 2-opt in shared memory).
12. **Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012).** *A hybrid genetic algorithm for multidepot and periodic vehicle routing problems.* Operations Research, 60(3), 611–624. (HGS-CVRP using $k$-NN granular local search).
13. **Rocki, K., & Suda, R. (2012).** *Accelerating 2-opt and 3-opt local search using GPU in the travelling salesman problem.* In IEEE International Conference on High Performance Computing and Simulation (HPCS), pp. 489–495. (Shared-memory tiling and on-the-fly coordinate distance evaluation for large TSP instances).
14. **Rocki, K., & Suda, R. (2013).** *High performance GPU accelerated local optimization in TSP.* In IEEE International Symposium on Parallel & Distributed Processing, Workshops and Phd Forum (IPDPSW), pp. 1788–1796. (Multiple non-conflicting 2-opt moves per GPU pass).
15. **O'Neil, M. A., & Burtscher, M. (2015).** *Rethinking the parallelization of random-restart hill climbing: A case study in optimizing a 2-opt TSP solver for GPU execution.* In Proceedings of the 8th Workshop on General Purpose Processing using GPUs (GPGPU-8), pp. 99–108. (Roofline analysis of coordinate caching vs global distance matrix lookup and tiled shared-memory 2-opt on GPU).
