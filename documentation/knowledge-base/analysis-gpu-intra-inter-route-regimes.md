---
title: "CUDA Intra- vs Inter-Block Synchronization and Routing Parallelization Regimes"
description: >
  Comprehensive architectural analysis of CUDA intra-block vs inter-block synchronization,
  hardware execution constraints, and the four fundamental routing parallelization regimes
  spanning population-based intra-route TSP, population-based inter-route CVRP 2-opt*,
  single-solution multi-vehicle CVRP, and single huge tour two-phase reductions.
created: 2026-11-05
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Created dedicated architectural knowledge-base document analyzing CUDA
          synchronization boundaries (intra-block vs inter-block), hardware grid
          independence, and the four routing regimes across TSP and CVRP with capacity
          formulas, memory layout, and conflict-resolution kernels.
      - description: >
          Corrected Section 5 codebase mapping table: distinguished two_opt_single.cu as a
          1-block untiled full-shared-memory baseline (n <= 11k) from proposed two_opt_stripe_scan.cu
          in Regime 3 (n > 50k two-phase reduction).
related_files:
  - "[analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md)"
  - "[analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu)"
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
tags:
  - analysis/performance
  - analysis/architecture
  - analysis/scaling
  - algorithm/two-opt
  - algorithm/cvrp
  - algorithm/tsp
  - benchmark/gpu
  - guide/cuda
  - notes/cuda
  - notes/architecture
  - notes/memory
---

# CUDA Intra- vs Inter-Block Synchronization and Routing Parallelization Regimes

- [CUDA Intra- vs Inter-Block Synchronization and Routing Parallelization Regimes](#cuda-intra--vs-inter-block-synchronization-and-routing-parallelization-regimes)
  - [1. CUDA Synchronization Fundamentals: Intra-Block vs Inter-Block](#1-cuda-synchronization-fundamentals-intra-block-vs-inter-block)
    - [1.1 The Hardware Execution Model](#11-the-hardware-execution-model)
    - [1.2 Intra-Block Synchronization (`__syncthreads()`)](#12-intra-block-synchronization-__syncthreads)
    - [1.3 The Inter-Block Synchronization Barrier Dilemma](#13-the-inter-block-synchronization-barrier-dilemma)
  - [2. The Four Routing Parallelization Regimes](#2-the-four-routing-parallelization-regimes)
    - [2.1 High-Level Architecture Overview](#21-high-level-architecture-overview)
    - [2.2 Regime 1: Population-Based Intra-Route TSP (Current Codebase)](#22-regime-1-population-based-intra-route-tsp-current-codebase)
    - [2.3 Regime 2A: Population-Based Inter-Route CVRP $2\text{-opt}^*$](#23-regime-2a-population-based-inter-route-cvrp-2text-opt)
    - [2.4 Regime 2B: Single-Solution Multi-Vehicle CVRP ($P = 1, m$ vehicles)](#24-regime-2b-single-solution-multi-vehicle-cvrp-p--1-m-vehicles)
    - [2.5 Regime 3: Single Huge Tour ($P = 1, n > 50{,}000$)](#25-regime-3-single-huge-tour-p--1-n--50000)
  - [3. CVRP Mathematical Formulations and Capacity Checks on GPU](#3-cvrp-mathematical-formulations-and-capacity-checks-on-gpu)
    - [3.1 Capacitated Vehicle Routing Problem (CVRP) Formulation](#31-capacitated-vehicle-routing-problem-cvrp-formulation)
    - [3.2 The $2\text{-opt}^*$ Inter-Route Move](#32-the-2text-opt-inter-route-move)
    - [3.3 Shared-Memory Capacity and Delta Evaluation](#33-shared-memory-capacity-and-delta-evaluation)
  - [4. Multi-Phase Conflict Resolution and Reduction Patterns](#4-multi-phase-conflict-resolution-and-reduction-patterns)
    - [4.1 Why Independent Sets of Swaps Are Required in Regime 2B](#41-why-independent-sets-of-swaps-are-required-in-regime-2b)
    - [4.2 Phase 1 / Phase 2 Kernel Partitioning Pattern](#42-phase-1--phase-2-kernel-partitioning-pattern)
    - [4.3 Two-Phase Row-Stripe Reduction in Regime 3](#43-two-phase-row-stripe-reduction-in-regime-3)
  - [5. Codebase Mapping and Extension Roadmap](#5-codebase-mapping-and-extension-roadmap)
  - [6. Academic References](#6-academic-references)

---

## 1. CUDA Synchronization Fundamentals: Intra-Block vs Inter-Block

In GPU computing, algorithms cannot simply place barriers wherever threads need to coordinate. The CUDA architecture enforces a strict hierarchical execution model that dictates which threads can communicate, how they synchronize, and what physical hardware units coordinate their actions.

### 1.1 The Hardware Execution Model

A CUDA application issues work to the GPU via **grids** of **thread blocks**:

$$\text{Grid} = \{ \text{Block}_0, \; \text{Block}_1, \; \dots, \; \text{Block}_{G-1} \}$$

$$\text{Block}_b = \{ \text{Thread}_0, \; \text{Thread}_1, \; \dots, \; \text{Thread}_{B-1} \}$$

The physical execution rules governing this hierarchy are:
1. **Thread Block Localization:** Every thread inside a single block $\text{Block}_b$ is guaranteed to execute on the **exact same Streaming Multiprocessor (SM)**. They share the SM's physical on-chip SRAM ($48\text{ KB}$ per block, up to $96\text{ KB}$ total per SM on Pascal CC 6.1).
2. **Block Independence:** Thread blocks $\text{Block}_a$ and $\text{Block}_b$ ($a \ne b$) may execute on **different SMs**, or on the same SM at **different times**. The CUDA hardware scheduler dispatches blocks to SMs in arbitrary, non-deterministic order.
3. **No Direct Inter-Block Hardware Channel:** An SM running Block 0 has no physical interconnect to read the shared memory of an SM running Block 1. Global GDDR5 DRAM is the only unified memory layer accessible by all blocks.

### 1.2 Intra-Block Synchronization (`__syncthreads()`)

Within a single block, synchronization is handled by the CUDA primitive `__syncthreads()`:
- **Hardware Implementation:** A dedicated barrier synchronization unit inside each SM sub-core.
- **Latency:** $\approx 20\text{ clock cycles}$ ($\approx 14\text{ ns}$ on GP107).
- **Semantics:** Execution halts for all warps within that block until every thread has reached the barrier instruction. Furthermore, all memory writes (shared and global) issued before the barrier are guaranteed to be visible to all threads in the block after the barrier.
- **Scope:** Strictly **intra-block**. Threads in Block 0 synchronizing at `__syncthreads()` have zero impact on or visibility into Block 1.

### 1.3 The Inter-Block Synchronization Barrier Dilemma

Why can standard CUDA kernels not invoke a global barrier across blocks (e.g., `__syncgrid()`)?

**The Hardware Deadlock Hazard:**
Suppose a GPU has 5 SMs, each capable of hosting 4 concurrent blocks. The GPU can execute at most $5 \times 4 = 20\text{ blocks}$ simultaneously.
If a kernel is launched with grid size $G = 100\text{ blocks}$:
- Blocks $0 \dots 19$ are resident on the SMs.
- Blocks $20 \dots 99$ are queued in the driver/hardware work distributor; they cannot even begin execution until resident blocks complete and retire their allocated registers and shared memory.

If Block 0 executes a speculative global barrier waiting for Block 50 to reach the same barrier, Block 0 will wait indefinitely. Block 50 will never start because Block 0 refuses to release its SM resources. The GPU enters an unrecoverable **hardware deadlock**.

To synchronize across all blocks in a grid, CUDA developers have two robust architectural options:
1. **The Kernel Boundary Barrier (Implicit Host-Dispatched Barrier):**
   Split the algorithm into two kernel launches: Kernel 1 computes block-level partial results and writes them to global VRAM; the kernel launch terminates; the CPU driver/stream acts as a zero-cost hardware barrier; Kernel 2 launches and reads the committed results.
2. **Cooperative Groups Grid Synchronization (`cg::this_grid().sync()`):**
   Introduced in CUDA 9. Requires that the grid size is strictly bounded by the physical concurrency of the device (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`). If the grid size exceeds hardware capacity, the launch is rejected.

---

## 2. The Four Routing Parallelization Regimes

Combinatorial routing problems (TSP, CVRP, VRPTW) exhibit distinct parallelism patterns depending on whether the solver operates on a population of solutions (Genetic Algorithms) or a single solution (Iterated Local Search), and whether local moves affect a single route or multiple interacting routes.

### 2.1 High-Level Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ REGIME 1: Population-Based Intra-Route TSP (Current `two_opt_batch.cu`, P >= 64)               │
│   Grid = P blocks  │  1 Block = 1 Individual Tour  │  B = 256 Threads scan (i, j) pairs        │
│   -> Tours p_0 and p_1 never interact => ZERO inter-block communication needed!                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 2A: Population-Based Inter-Route CVRP 2-opt* (GA Population, P >= 64)                   │
│   Grid = P blocks  │  1 Block = 1 Complete CVRP Individual (all m vehicle routes!)             │
│   -> Shared memory holds giant tour + route_id[n] + prefix_load[n].                            │
│   -> Threads evaluate inter-route pairs (i in R_a, j in R_b, a != b) inside the SAME block     │
│      => STILL ZERO inter-block communication needed!                                           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 2B: Single-Solution Inter-Route CVRP 2-opt* (P = 1, m vehicles)                         │
│   Grid = m(m-1)/2 blocks  │  1 Block = 1 Route Pair (R_a, R_b)                                 │
│   -> Phase 1 Kernel: Each block finds best 2-opt* swap between R_a and R_b -> writes to VRAM.  │
│   -> Phase 2 Kernel (1 block): Resolves route conflicts & applies disjoint swaps to solution.  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ REGIME 3: Single Huge Tour (P = 1, n > 50,000)                                                 │
│   Grid = G blocks  │  1 Block = 1 Row Stripe [i_start .. i_end) of the SAME tour               │
│   -> Phase 1 Kernel: Block b scans its row stripe and writes block_best[b] to VRAM.            │
│   -> Phase 2 Kernel (1 block): Reduces block_best[0..G-1] and reverses best [i+1..j].          │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Regime 1: Population-Based Intra-Route TSP (Current Codebase)

This regime represents the existing production kernel in [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L15).

- **Problem Model:** Traveling Salesperson Problem ($1$ single Hamiltonian cycle of length $n$).
- **Metaheuristic Context:** Genetic Algorithm / Island Model with population size $P \ge 64$ individuals.
- **CUDA Grid Decomposition:**
  $$\text{Grid Dimension} = P \text{ blocks}, \quad \text{Block Dimension} = B = 256\text{ threads}$$
- **Mapping:** Block $b \in \{0, \dots, P-1\}$ is assigned exclusively to tour $b$ (`tour = tours + (size_t)b * n`).
- **Communication Pattern:** Tour $p_0$ and tour $p_1$ are completely independent solutions in the GA population. They do not share state, swap edges, or compete during local search.
- **Synchronization:** Strictly **intra-block**. Threads within block $b$ synchronize using `__syncthreads()` to cooperatively load sliding windows into shared memory (`s_tour`), evaluate deltas, and reduce to find the best local 2-opt move for tour $b$.
- **Inter-Block Sync Needed:** **Zero.** No grid-level barriers or host syncs are required during the 2-opt pass.

### 2.3 Regime 2A: Population-Based Inter-Route CVRP $2\text{-opt}^*$

When extending the solver from TSP to the Capacitated Vehicle Routing Problem (CVRP), a common misconception is that cross-route optimization requires inter-block communication. In a population-based metaheuristic, this is entirely false.

- **Problem Model:** CVRP with $n$ customers, vehicle capacity $Q$, and $m$ vehicle routes.
- **Metaheuristic Context:** Genetic Algorithm with population size $P \ge 64$ individuals.
- **The Key Insight:** In a GA for CVRP, **each individual $p$ is an entire, valid CVRP solution** consisting of all $m$ vehicle routes:
  $$S^{(p)} = \{ R_1^{(p)}, \; R_2^{(p)}, \; \dots, \; R_m^{(p)} \}$$
- **CUDA Grid Decomposition:**
  $$\text{Grid Dimension} = P \text{ blocks}, \quad \text{Block Dimension} = B = 256\text{ threads}$$
- **Shared Memory Allocation per Block:**
  Instead of holding only a single tour, the block's shared memory holds the chromosome representation of individual $p$:
  1. `s_tour[n + m]`: The giant tour array containing all $n$ customers separated by depot delimiters ($0$ or $-1$).
  2. `s_route_id[n]`: A direct lookup array mapping customer node $u \in \{1, \dots, n\}$ to its route index $r \in \{0, \dots, m-1\}$.
  3. `s_prefix_load[n]`: The cumulative vehicle load from the depot up to customer $u$.
- **Execution Workflow:**
  Threads in Block $b$ scan pairs of edges $(u, v)$ where customer $u \in R_a$ and customer $v \in R_b$ ($a \ne b$). Because routes $R_a$ and $R_b$ both belong to individual $p$ and reside in the **same block's shared memory**, threads evaluate inter-route swaps and capacity feasibility inside the block.
- **Inter-Block Sync Needed:** **Zero.** Individual $p_0$ and individual $p_1$ remain 100% independent.

### 2.4 Regime 2B: Single-Solution Multi-Vehicle CVRP ($P = 1, m$ vehicles)

When solving a single CVRP instance ($P = 1$) via single-solution metaheuristics (Iterated Local Search, Tabu Search, or Simulated Annealing), assigning only 1 block to the problem would leave $95\%$ of the GPU's compute cores idle. We must parallelize across the routes of the single solution.

- **Problem Model:** Single CVRP solution ($P = 1$) with $m$ vehicle routes.
- **Candidate Route Pairs:** There are $\binom{m}{2} = \frac{m(m-1)}{2}$ distinct pairs of routes that could exchange customers. For $m = 30$ vehicles, there are $435\text{ candidate pairs}$.
- **CUDA Grid Decomposition:**
  $$\text{Grid Dimension} = \frac{m(m-1)}{2} \text{ blocks}, \quad \text{Block Dimension} = 256\text{ threads}$$
- **Two-Phase Kernel Decomposition:**
  - **Phase 1 Kernel (`cvrp_inter_route_eval`):**
    Block $b$ is assigned to route pair $(R_a, R_b)$. The block loads routes $R_a$ and $R_b$ into its shared memory, evaluates all valid capacity-respecting $2\text{-opt}^*$ moves between them, and writes the best improving move:
    $$\text{block\_best}[b] = \langle \Delta_{\text{best}}, \; i^*, \; j^*, \; a, \; b \rangle$$
    into global VRAM.
  - **Implicit Global Barrier:** Kernel 1 finishes execution.
  - **Phase 2 Kernel (`cvrp_conflict_resolution`, 1 block):**
    Because multiple route pairs may involve the same route (e.g., $(R_1, R_2)$ and $(R_1, R_3)$ both proposing modifications to $R_1$), they cannot both be applied simultaneously without creating race conditions and invalid solutions. Phase 2 reads all candidate moves from $\text{block\_best}$, selects a conflict-free set of moves (or greedily applies the single global best move), and updates the master solution in VRAM.

### 2.5 Regime 3: Single Huge Tour ($P = 1, n > 50{,}000$)

When solving an enormous single TSP tour ($n > 50{,}000$ cities) on a GPU, the problem cannot fit into a single block's shared memory, and a single block cannot saturate the hardware.

- **Problem Model:** Single TSP tour ($P = 1$), large scale ($n > 50{,}000$).
- **Row Stripe Decomposition:** The 2-opt search space consists of $\frac{n(n-3)}{2}$ candidate edge pairs $(i, j)$. We partition the outer loop index $i \in \{0, \dots, n-3\}$ into $G$ contiguous row stripes:
  $$\text{Stripe}_b = [i_{\text{start}}^{(b)}, \; i_{\text{end}}^{(b)}), \quad \text{where } i_{\text{start}}^{(b)} = b \times \frac{n}{G}$$
- **Two-Phase Kernel Decomposition:**
  - **Phase 1 Kernel (`two_opt_stripe_scan`, $G$ blocks):**
    Block $b$ scans candidate moves for all $i \in \text{Stripe}_b$ and all $j \in \{i+2, \dots, n-1\}$ using shared-memory sliding windows. Each block writes its local minimum delta to VRAM:
    $$\text{block\_best}[b] = \langle \Delta_{\text{best}}^{(b)}, \; i^{*(b)}, \; j^{*(b)} \rangle$$
  - **Implicit Global Barrier:** Kernel 1 finishes.
  - **Phase 2 Kernel (`two_opt_apply_best`, 1 block, 256 threads):**
    Loads the $G$ candidates from $\text{block\_best}$, performs a parallel tree reduction across threads to find the global optimum $(i^*, j^*)$ with $\Delta^* < 0$, and inverts the tour sub-segment $v[i^*+1 \dots j^*]$ in global memory.

---

## 3. CVRP Mathematical Formulations and Capacity Checks on GPU

### 3.1 Capacitated Vehicle Routing Problem (CVRP) Formulation

Let:
- $V = \{0, 1, \dots, n\}$ be the set of nodes, where $0$ is the depot and $V_C = \{1, \dots, n\}$ are customers.
- $q_i > 0$ be the non-negative delivery demand of customer $i$ ($q_0 = 0$).
- $Q$ be the maximum vehicle capacity.
- $m$ be the number of homogeneous vehicles.

A feasible CVRP solution partitions customers into $m$ routes $R_1, \dots, R_m$ such that:
1. Every customer $i \in V_C$ is visited exactly once by exactly one route.
2. Every route begins and terminates at the depot ($0$).
3. The total vehicle load does not exceed capacity $Q$:
   $$\text{Demand}(R_k) = \sum_{u \in R_k} q_u \le Q, \quad \forall k \in \{1, \dots, m\}$$

### 3.2 The $2\text{-opt}^*$ Inter-Route Move

Originally introduced by [Potvin & Rousseau (1995)](#6-academic-references), the $2\text{-opt}^*$ operator is the canonical inter-route exchange mechanism. Unlike intra-route 2-opt (which reverses a segment within the same route), $2\text{-opt}^*$ cuts two edges belonging to **two distinct routes** and swaps their tail segments:

$$\text{Route } R_a = \langle 0, \; u_1, \; \dots, \; u_i, \quad u_{i+1}, \; \dots, \; u_{|R_a|}, \; 0 \rangle$$

$$\text{Route } R_b = \langle 0, \; v_1, \; \dots, \; v_j, \quad v_{j+1}, \; \dots, \; v_{|R_b|}, \; 0 \rangle$$

Edges $(u_i, u_{i+1})$ and $(v_j, v_{j+1})$ are severed. The tails are exchanged to produce two new routes:

$$R_a' = \langle 0, \; u_1, \; \dots, \; u_i, \quad v_{j+1}, \; \dots, \; v_{|R_b|}, \; 0 \rangle$$

$$R_b' = \langle 0, \; v_1, \; \dots, \; v_j, \quad u_{i+1}, \; \dots, \; u_{|R_a|}, \; 0 \rangle$$

*(Alternatively, the inverted tail variation connects $u_i$ to $v_j$ in reverse order).*

```text
Before 2-opt*:
Route A:  Depot ───> ... ───> u_i ──(cut)──> u_{i+1} ───> ... ───> Depot
Route B:  Depot ───> ... ───> v_j ──(cut)──> v_{j+1} ───> ... ───> Depot

After 2-opt* (Tail Exchange):
Route A': Depot ───> ... ───> u_i ─────────> v_{j+1} ───> ... ───> Depot
                                    ╲   ╱
                                     ╳
                                    ╱   ╲
Route B': Depot ───> ... ───> v_j ─────────> u_{i+1} ───> ... ───> Depot
```

### 3.3 Shared-Memory Capacity and Delta Evaluation

In naive CPU algorithms, verifying whether $R_a'$ and $R_b'$ satisfy capacity requires $O(|R_a| + |R_b|)$ steps to re-sum customer demands. On a GPU, this would introduce disastrous warp divergence and latency.

**$O(1)$ Prefix-Sum Capacity Check:**
By maintaining a prefix-sum array of demands in shared memory:

$$\text{prefix\_load}[u_k] = \sum_{t=1}^k q_{u_t}$$

the total demand of route $R_a$ is simply $L(R_a) = \text{prefix\_load}[u_{|R_a|}]$.
When evaluating the swap at cut points $u_i$ and $v_j$:

$$\text{Demand}(R_a') = \text{prefix\_load}[u_i] + \Bigl( L(R_b) - \text{prefix\_load}[v_j] \Bigr)$$

$$\text{Demand}(R_b') = \text{prefix\_load}[v_j] + \Bigl( L(R_a) - \text{prefix\_load}[u_i] \Bigr)$$

The capacity feasibility condition is evaluated in registers in **2 arithmetic cycles**:

$$\text{Feasible} \iff \Bigl(\text{Demand}(R_a') \le Q\Bigr) \;\land\; \Bigl(\text{Demand}(R_b') \le Q\Bigr)$$

If feasible, the delta in travel cost is computed using the 4 boundary edges:

$$\Delta_{\text{cost}} = \Bigl( d(u_i, v_{j+1}) + d(v_j, u_{i+1}) \Bigr) - \Bigl( d(u_i, u_{i+1}) + d(v_j, v_{j+1}) \Bigr)$$

A candidate move is improving if and only if:

$$\text{Feasible} \;\land\; (\Delta_{\text{cost}} < 0)$$

---

## 4. Multi-Phase Conflict Resolution and Reduction Patterns

### 4.1 Why Independent Sets of Swaps Are Required in Regime 2B

In Regime 2B, the Phase 1 kernel evaluates $\frac{m(m-1)}{2}$ route pairs concurrently.
Suppose:
- Block 1 evaluates pair $(R_1, R_4)$ and finds an improving swap of $\Delta = -42.0$.
- Block 8 evaluates pair $(R_1, R_9)$ and finds an improving swap of $\Delta = -38.5$.

Both blocks propose modifications to Route $R_1$. If both modifications were applied concurrently to global memory, the resulting route $R_1$ would be corrupted, customer nodes would be duplicated or omitted, and the solution would become infeasible.

### 4.2 Phase 1 / Phase 2 Kernel Partitioning Pattern

To guarantee solution feasibility while maintaining massive parallelism, Regime 2B employs a two-phase architecture:

```text
Phase 1: Parallel Evaluation Kernel (m*(m-1)/2 Blocks)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐         ┌──────────────┐
│ Block 0      │  │ Block 1      │  │ Block 2      │  . . .  │ Block K-1    │
│ Pair (R0, R1)│  │ Pair (R0, R2)│  │ Pair (R0, R3)│         │ Pair (Ra, Rb)│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘         └──────┬───────┘
       │                 │                 │                        │
       ▼                 ▼                 ▼                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Global Memory Scratchpad: block_best[0 ... K-1]                           │
│ [ <delta, i, j, a, b>, <delta, i, j, a, b>, ... ]                         │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                         === IMPLICIT GRID BARRIER ===
                                      │
                                      ▼
Phase 2: Conflict Resolution Kernel (1 Block, 256 Threads)
┌───────────────────────────────────────────────────────────────────────────┐
│ 1. Load candidate moves into shared memory.                               │
│ 2. Parallel sort candidate moves by delta ascending (best moves first).    │
│ 3. Greedy Independent Set: Track modified_route[m] bitmask.               │
│    For each candidate <delta, i, j, a, b>:                                │
│       If NOT (modified[a] OR modified[b]) AND delta < 0:                  │
│          Commit swap to master solution;                                  │
│          Set modified[a] = True; modified[b] = True;                      │
└───────────────────────────────────────────────────────────────────────────┘
```

Because Phase 2 executes with a single block, it has access to fast shared memory and warp shuffle operations, resolving conflicts across hundreds of route candidates in microseconds.

### 4.3 Two-Phase Row-Stripe Reduction in Regime 3

For gigantic single tours ($P = 1, n > 50{,}000$), the two-phase kernel pattern operates as follows:

```cpp
// Kernel 1: Launched with G blocks (e.g., G = 128)
__global__ void two_opt_stripe_scan_kernel(
    const int* __restrict__ tour,
    const float* __restrict__ coords,
    int n,
    CandidateMove* __restrict__ block_best)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Each block owns a contiguous row slice of the tour
    int rows_per_block = (n - 2 + gridDim.x - 1) / gridDim.x;
    int i_start = bid * rows_per_block;
    int i_end = min(i_start + rows_per_block, n - 2);

    CandidateMove local_best = {0.0f, -1, -1};

    for (int i = i_start + tid; i < i_end; i += blockDim.x) {
        // Tiled sliding window scan across j in [i+2, n)
        // Update local_best...
    }

    // Intra-block tree reduction across threads in block
    local_best = block_reduce_min(local_best);
    if (tid == 0) {
        block_best[bid] = local_best;
    }
}

// Kernel 2: Launched with 1 block (256 threads)
__global__ void two_opt_apply_best_kernel(
    int* __restrict__ tour,
    int n,
    const CandidateMove* __restrict__ block_best,
    int G)
{
    int tid = threadIdx.x;
    
    // Cooperatively load G candidates into shared memory
    CandidateMove best_move = {0.0f, -1, -1};
    if (tid < G) {
        best_move = block_best[tid];
    }
    
    // Tree reduction across the G blocks to find global minimum
    best_move = block_reduce_min(best_move);

    // Apply the winning 2-opt reversal in parallel
    if (best_move.delta < 0.0f) {
        int i = best_move.i;
        int j = best_move.j;
        // Parallel in-place sub-tour reversal: swap(tour[i + 1 + t], tour[j - t])
        int len = (j - i) / 2;
        for (int t = tid; t < len; t += blockDim.x) {
            int temp = tour[i + 1 + t];
            tour[i + 1 + t] = tour[j - t];
            tour[j - t] = temp;
        }
    }
}
```

This architecture completely eliminates CPU-GPU round-trips: the entire 2-opt iteration, from row scanning to tree reduction and sub-tour reversal, takes place on the GPU with zero host intervention.

---

## 5. Codebase Mapping and Extension Roadmap

The following table summarizes the mapping between these regimes and the codebase modules:

| Regime | Problem Scope | Codebase File / Target | Shared Memory Content | Synchronization Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime 1** | Population TSP ($P \ge 64$) | [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L15) | 1 Tour Tile ($T = 2{,}048$) + Reductions | Strictly Intra-Block (`__syncthreads()`) | **Active in Production** |
| **Regime 2A** | Population CVRP ($P \ge 64$) | Proposed `cvrp_two_opt_batch.cu` | Full Solution ($n + m$) + Route IDs + Prefix Loads | Strictly Intra-Block (`__syncthreads()`) | **Roadmap: CVRP Extension** |
| **Regime 2B** | Single CVRP ($P = 1$) | Proposed `cvrp_two_opt_pair.cu` | Route Pair $(R_a, R_b)$ + Prefix Loads | 2-Phase (Kernel 1: $\binom{m}{2}$ blocks; Kernel 2: 1 block) | **Roadmap: CVRP Extension** |
| **Baseline Single** | Single Small/Medium TSP ($n \lesssim 11\text{k}$) | [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L1) | Full Single Tour (`s_tour[n]`) + Intra-Block Reduction | Strictly Intra-Block (1 block, $P=1$) | **Implemented Baseline** |
| **Regime 3** | Single Huge TSP ($n > 50\text{k}$) | Proposed `two_opt_stripe_scan.cu` | Tour Window + Stripe Reduction | 2-Phase (Kernel 1: $G$ blocks; Kernel 2: 1 block) | **Roadmap: Scaled TSP** |

---

## 6. Academic References

1. **Potvin, J. Y., & Rousseau, J. M. (1995).** An Exchange Heuristic for Routeing Problems with Time Windows. *Journal of the Operational Research Society*, 46(12), 1433–1446. DOI: [10.1057/jors.1995.204](https://doi.org/10.1057/jors.1995.204).
2. **Cordeau, J. F., Gendreau, M., Laporte, G., Potvin, J. Y., & Semet, F. (2002).** A Guide to Vehicle Routing Heuristics. *Journal of the Operational Research Society*, 53(5), 512–522. DOI: [10.1057/palgrave.jors.2601319](https://doi.org/10.1057/palgrave.jors.2601319).
3. **Rocki, K., & Suda, R. (2012).** Accelerating 2-Opt Local Search for the Traveling Salesman Problem on NVIDIA GPUs. In *Proceedings of the 2012 IEEE International Conference on High Performance Computing and Simulation (HPCS)*, pp. 489–495. DOI: [10.1109/HPCSim.2012.6266965](https://doi.org/10.1109/HPCSim.2012.6266965).
4. **Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012).** A Hybrid Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems. *Operations Research*, 60(3), 611–624. DOI: [10.1287/opre.1120.1048](https://doi.org/10.1287/opre.1120.1048).
5. **NVIDIA Corporation (2024).** *CUDA C++ Programming Guide: Cooperative Groups and Grid Synchronization*. Document ID: PG-02829-001_v12.4.
