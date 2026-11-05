---
title: "GPU VRAM Capacity Guardrails, Buffer Pooling, and PCIe Transfer Diagnostics"
description: >
  Comprehensive engineering guide to GPU memory safety and throughput optimization:
  pre-flight VRAM capacity estimation formulas, active hardware guardrails, persistent scratchpad
  buffer pooling across GA, SA, and ACO metaheuristics, streaming concurrency patterns (P-Data vs P-Task),
  and quantitative diagnosis of single-batch vs per-iteration PCIe transfer bottlenecks.
created: 2026-11-05
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Initial creation of the production guide covering pre-flight VRAM guardrails
          (ProblemContext DP10-19), persistent scratchpad pooling patterns to prevent
          runtime allocator fragmentation, CUDA stream concurrency regimes, and PCIe
          interconnect Roofline diagnostics.
      - description: >
          Corrected Crainic & Toulouse concurrency classification terminology from S-Task
          to P-Task for multi-search/island-model concurrency, maintaining S-Task for sequential
          single-chain execution.
related_files:
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[analysis-walkthrough-10-19-vram-guard.md](../walkthroughs/analysis-walkthrough-10-19-vram-guard.md)"
  - "[test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py)"
  - "[exceptions.py](../../src/data_models/exceptions.py)"
  - "[analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md)"
tags:
  - guide/cuda
  - guide/architecture
  - guide/memory
  - analysis/memory
  - analysis/performance
  - notes/cuda
  - notes/vram
  - benchmark/gpu
  - algorithm/genetic-algorithm
  - algorithm/tsp
---

# GPU VRAM Capacity Guardrails, Buffer Pooling, and PCIe Transfer Diagnostics

- [GPU VRAM Capacity Guardrails, Buffer Pooling, and PCIe Transfer Diagnostics](#gpu-vram-capacity-guardrails-buffer-pooling-and-pcie-transfer-diagnostics)
  - [1. Pre-Flight VRAM Capacity Guardrails (ProblemContext DP10-19)](#1-pre-flight-vram-capacity-guardrails-problemcontext-dp10-19)
    - [1.1 The Out-Of-Memory Catastrophe in GPU Metaheuristics](#11-the-out-of-memory-catastrophe-in-gpu-metaheuristics)
    - [1.2 Mathematical Formulation of VRAM Demand](#12-mathematical-formulation-of-vram-demand)
    - [1.3 Active Hardware Interrogation via `cp.cuda.Device().mem_info`](#13-active-hardware-interrogation-via-cpcudadevicemem_info)
    - [1.4 The Fail-Fast Protocol and `VRAMInsufficientError`](#14-the-fail-fast-protocol-and-vraminsufficienterror)
  - [2. Temporary Allocations vs Persistent Scratchpad Buffer Pooling](#2-temporary-allocations-vs-persistent-scratchpad-buffer-pooling)
    - [2.1 The Latency and Fragmentation Cost of Naive `cudaMalloc`](#21-the-latency-and-fragmentation-cost-of-naive-cudamalloc)
    - [2.2 The Persistent Scratchpad Pooling Architecture](#22-the-persistent-scratchpad-pooling-architecture)
    - [2.3 Multi-Strategy Buffer Pooling Patterns](#23-multi-strategy-buffer-pooling-patterns)
  - [3. Streaming Concurrency: P-Data vs P-Task Regimes](#3-streaming-concurrency-p-data-vs-p-task-regimes)
    - [3.1 The Two Concurrency Paradigms](#31-the-two-concurrency-paradigms)
    - [3.2 Stream Pipelining for P-Data Batching](#32-stream-pipelining-for-p-data-batching)
    - [3.3 Asynchronous Island Migration in P-Task Execution](#33-asynchronous-island-migration-in-p-task-execution)
  - [4. Quantitative Diagnosis of PCIe Transfer Bottlenecks](#4-quantitative-diagnosis-of-pcie-transfer-bottlenecks)
    - [4.1 Theoretical vs Real-World PCIe Interconnect Bandwidth](#41-theoretical-vs-real-world-pcie-interconnect-bandwidth)
    - [4.2 The Naive Hybrid Antipattern: Per-Iteration Round-Trips](#42-the-naive-hybrid-antipattern-per-iteration-round-trips)
    - [4.3 The Single-Batch GPU-Resident Paradigm](#43-the-single-batch-gpu-resident-paradigm)
    - [4.4 Roofline Model Comparison](#44-roofline-model-comparison)
  - [5. Academic and Industry References](#5-academic-and-industry-references)

---

## 1. Pre-Flight VRAM Capacity Guardrails (ProblemContext DP10-19)

### 1.1 The Out-Of-Memory Catastrophe in GPU Metaheuristics

In iterative stochastic optimization, encountering a `CUDA_ERROR_OUT_OF_MEMORY` error at generation 9,500 of a 10,000-generation run is catastrophic. The CUDA driver immediately corrupts the host context, all asynchronous streams are terminated, in-flight GPU memory states become unrecoverable, and the entire Python process is aborted.

Traditional heuristic codebases rely on reactive exception handling (`try ... except cp.cuda.memory.OutOfMemoryError`). This approach is deeply flawed on modern GPUs because:
1. CuPy memory pool fragmentation can trigger driver-level aborts before Python catches the exception;
2. Reactive catching does not prevent thrashing virtual memory or locking the system display driver;
3. It cannot inform the user beforehand how to rescale parameters (such as population size $P$ or switching to coordinate-based distance evaluation).

To solve this, our codebase implements an **active, proactive pre-flight VRAM capacity guardrail** inside [problem_context.py](../../src/protocols/problem_context.py#L486-L537), documented in [analysis-walkthrough-10-19-vram-guard.md](../walkthroughs/analysis-walkthrough-10-19-vram-guard.md) and verified by 22 unit tests in [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py).

### 1.2 Mathematical Formulation of VRAM Demand

Before allocating a single device array, [estimate_vram_bytes(pop_size)](../../src/protocols/problem_context.py#L486) computes the exact upper-bound VRAM bytes required to execute population-based metaheuristics for $n$ cities and population size $P$:

$$\text{VRAM}_{\text{required}}(P, n, d) = \mathbf{M}_{\text{matrix}}(n) + \mathbf{B}_{\text{population}}(P, n) + \mathbf{B}_{\text{fitness}}(P) + \mathbf{B}_{\text{auxiliary}}(n, d)$$

```text
┌────────────────────────────────────────────────────────────────────────┐
│               ProblemContext VRAM Memory Budget Breakdown              │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Distance Representation:                                            │
│    - Dense Mode:        4 * n^2 bytes (int32 / float32 distance matrix) │
│    - On-the-Fly Mode:   0 bytes       (evaluated from coordinates)     │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Population Buffers (12 * P * n bytes):                              │
│    - Current Population:    P * n * 4 bytes (int32)                    │
│    - Offspring Population:  P * n * 4 bytes (int32)                    │
│    - Mating / Temp Pool:    P * n * 4 bytes (int32)                    │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Fitness Buffers (24 * P bytes):                                     │
│    - Current Fitness:       P * 8 bytes (float64)                      │
│    - Offspring Fitness:     P * 8 bytes (float64)                      │
│    - Generation Best:       P * 8 bytes (float64)                      │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Coordinate & Auxiliary Buffers (8 * d * n bytes):                   │
│    - Coordinates Array:     d * n * 8 bytes (d dimensions, float64)    │
│    - Node Demands / IDs:    (d = 1 fallback when no 2D coords exist)   │
└────────────────────────────────────────────────────────────────────────┘
```

The exact formula implemented in code is:

$$\text{VRAM}_{\text{required}}(P, n, d) = \begin{cases} 4n^2 + 12Pn + 24P + 8dn & \text{if } \text{uses\_on\_the\_fly\_distances}(n) = \text{False} \\ 12Pn + 24P + 8dn & \text{if } \text{uses\_on\_the\_fly\_distances}(n) = \text{True} \end{cases}$$

**Concrete Numerical Example:**
For an instance of $n = 15{,}112$ cities (`d15112.tsp`) and population $P = 256$:
- **Dense Mode:**
  $$\text{Matrix} = 4 \times (15{,}112)^2 = 913{,}501{,}696\text{ B} \approx \mathbf{913.5\text{ MB}}$$
  $$\text{Population} = 12 \times 256 \times 15{,}112 = 46{,}424{,}064\text{ B} \approx \mathbf{46.4\text{ MB}}$$
  $$\text{Fitness} = 24 \times 256 = 6{,}144\text{ B} \approx \mathbf{6.1\text{ KB}}$$
  $$\text{Coordinates} (d=2) = 8 \times 2 \times 15{,}112 = 241{,}792\text{ B} \approx \mathbf{241.8\text{ KB}}$$
  $$\text{Total Dense} \approx \mathbf{960.2\text{ MB}}$$
- **On-the-Fly Mode:**
  The $913.5\text{ MB}$ matrix is omitted!
  $$\text{Total On-the-Fly} = 46.4\text{ MB} + 0.006\text{ MB} + 0.24\text{ MB} \approx \mathbf{46.65\text{ MB}}$$

By switching to on-the-fly coordinate evaluation when $n > 10{,}000$, total VRAM consumption drops by **$95.1\%$**, enabling the solver to run on modest $4\text{ GB}$ laptop GPUs.

### 1.3 Active Hardware Interrogation via `cp.cuda.Device().mem_info`

The solver inspects physical device memory in real time:

```python
# File: src/protocols/problem_context.py, Lines 471-476
def _query_free_vram_bytes(self) -> int | None:
  if self.xp == np:
    return None
  cuda: Any = cast("Any", self.xp).cuda
  free_bytes, _total_bytes = cuda.Device().mem_info
  return int(free_bytes)
```

`cuda.Device().mem_info` queries the NVIDIA driver directly via the CUDA Runtime API (`cudaMemGetInfo`), returning:
- `free_bytes`: Truly unallocated, physically available VRAM on the GPU;
- `_total_bytes`: Total physical VRAM capacity of the graphics card (e.g., $4{,}294{,}967{,}296\text{ bytes}$ for a 4 GB card).

### 1.4 The Fail-Fast Protocol and `VRAMInsufficientError`

When a caller requests a population size $P$ or requests the dense distance matrix via `get_gpu_distances()`:

```python
# File: src/protocols/problem_context.py, Lines 477-484
def _raise_if_insufficient(
  self, required_bytes: int, available_bytes: int | None, operation: str
) -> None:
  """Raise VRAMInsufficientError when required_bytes exceeds available VRAM."""
  if available_bytes is not None and required_bytes > available_bytes:
    raise VRAMInsufficientError(
      f"Insufficient VRAM for {operation}",
      required_bytes=required_bytes,
      available_bytes=available_bytes,
    )
```

If `required_bytes > available_bytes`, the solver immediately raises `VRAMInsufficientError` (defined in [exceptions.py](../../src/data_models/exceptions.py#L1)).
The error formats the byte counts into human-readable gigabytes:

```text
VRAMInsufficientError: Insufficient VRAM for population evaluation (pop_size=2048, n=15112).
Required: 1.85 GB, Available: 0.42 GB.
Recommendation: Reduce pop_size or enable on-the-fly coordinate evaluation.
```

The process fails fast and safely, preserving the Python interpreter state and allowing automatic fallback to CPU processing or reduced batch sizes.

---

## 2. Temporary Allocations vs Persistent Scratchpad Buffer Pooling

### 2.1 The Latency and Fragmentation Cost of Naive `cudaMalloc`

A widespread anti-pattern in naive GPU implementations is allocating temporary device memory inside evolutionary loops:

```python
# NAIVE ANTIPATTERN (DO NOT USE)
for generation in range(10000):
  offspring = cp.empty((P, n), dtype=cp.int32)  # Triggers device allocation!
  fitness = cp.empty(P, dtype=cp.float64)  # Triggers device allocation!
  # Run crossover, mutation, 2-opt...
```

Why does this destroy performance?
1. **Memory Allocator Mutex Lock:** Every call to `cudaMalloc` (or CuPy's internal pool allocator) acquires a global thread mutex lock in user space.
2. **Driver Synchronizations:** If CuPy's memory pool requires allocating a new memory block from the OS kernel, it issues an implicit `cudaStreamSynchronize()`, draining all GPU pipelines and destroying concurrency.
3. **Memory Fragmentation:** Allocating and freeing millions of small buffers with variable shapes creates non-contiguous memory holes in VRAM. Eventually, an allocation request for a contiguous $100\text{ MB}$ buffer fails even though $2\text{ GB}$ of free memory is technically available in fragmented fragments.
4. **Allocation Latency:** Calling `cudaMalloc` incurs a latency of $10\text{--}50\,\mu\text{s}$. Across $10{,}000\text{ generations}$ with 4 operators per generation, allocation overhead alone wastes:
   $$10{,}000 \times 4 \times 30\,\mu\text{s} = 1.2\text{ seconds of pure CPU idle wait!}$$

### 2.2 The Persistent Scratchpad Pooling Architecture

In high-performance GPU metaheuristics, device memory allocation should follow a **single-allocation lifecycle**:

$$\text{Allocate Once at Init} \implies \text{Reuse Across 10,000 Generations} \implies \text{Deallocate at Teardown}$$

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   PERSISTENT SCRATCHPAD MEMORY POOL                    │
├────────────────────────────────────────────────────────────────────────┤
│ Pre-allocated once during Metaheuristic Initialization:                │
│                                                                        │
│   d_population:           cp.ndarray (shape=(P, n),     dtype=int32)   │
│   d_offspring:            cp.ndarray (shape=(P, n),     dtype=int32)   │
│   d_mating_pool:          cp.ndarray (shape=(P, n),     dtype=int32)   │
│   d_fitness:              cp.ndarray (shape=(P,),       dtype=float64) │
│   d_offspring_fitness:    cp.ndarray (shape=(P,),       dtype=float64) │
│   d_selection_indices:    cp.ndarray (shape=(P,),       dtype=int32)   │
│   d_reduction_scratch:    cp.ndarray (shape=(P, 256),   dtype=int32)   │
└────────────────────────────────────────────────────────────────────────┘
```

During the generational loop:
- Operators write directly into the pre-allocated buffers.
- Double-buffering / pointer swapping replaces physical copies:
  ```python
  # Zero-cost pointer swap (Lamarckian survival)
  d_population, d_offspring = d_offspring, d_population
  ```
- Runtime memory allocation inside the loop drops to **identically zero**.

### 2.3 Multi-Strategy Buffer Pooling Patterns

The scratchpad pooling pattern extends naturally across all optimization strategies in our codebase:

1. **Genetic Algorithm (GA):**
   - Holds persistent population buffers `d_pop` and `d_offspring` of shape $(P, n)$.
   - Holds persistent tournament index buffer `d_winners` of shape $(P,)$.
2. **Simulated Annealing (SA / Batch SA):**
   - Evaluates a batch of $P$ independent Markov chains in parallel on the GPU.
   - Holds persistent current state `d_current` $(P, n)$, candidate state `d_candidate` $(P, n)$, and temperature vector `d_temp` $(P,)$.
   - Metropolis acceptance criterion is evaluated in-place using vector operations:
     $$\Delta = \text{cost}(\text{candidate}) - \text{cost}(\text{current})$$
     $$\text{Accept} \iff (\Delta < 0) \;\lor\; \left( \mathcal{U}(0, 1) < \exp(-\Delta / T) \right)$$
3. **Ant Colony Optimization (Future ACO Operator):**
   - Holds persistent pheromone matrix $\boldsymbol{\tau} \in \mathbb{R}^{n \times n}$ (`float32`).
   - Holds ant tour scratchpad $(m_{\text{ants}}, n)$ (`int32`).
   - Holds visited bitmasks $(m_{\text{ants}}, \lceil n / 32 \rceil)$ (`uint32`) for $O(1)$ tabu checks.
   - Evaporation ($\boldsymbol{\tau} \leftarrow (1 - \rho)\boldsymbol{\tau}$) and deposition ($\boldsymbol{\tau} \leftarrow \boldsymbol{\tau} + \Delta\boldsymbol{\tau}$) execute entirely in-place on GPU via vectorized kernel launches.

---

## 3. Streaming Concurrency: P-Data vs P-Task Regimes

CUDA streams allow concurrent execution of kernels and memory transfers. Following the foundational taxonomy of **Crainic & Toulouse (2003, 2010)**, parallel metaheuristics are categorized along two orthogonal dimensions: information shared (Type 1: Low-level operation parallelism, Type 2: Domain-level solution parallelism, Type 3: Multi-search parallelism) and control strategy (Synchronous vs Asynchronous). In GPU stream concurrency, execution is organized into two primary parallel regimes: **P-Data** (data-parallel decomposition of a single population/search) and **P-Task** (task-parallel multi-search / concurrent Island Model), contrasted with sequential single-search progression (**S-Task**).

### 3.1 The Two Concurrency Paradigms

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        STREAMING CONCURRENCY REGIMES                   │
├────────────────────────────────────────────────────────────────────────┤
│ 1. P-Data (Data-Parallel Batch Partitioning):                          │
│    - One single large population (e.g., P = 1,024) partitioned into    │
│      chunks of 256 across multiple CUDA streams.                       │
│    - Stream 0: Evaluates Chunk 0  (Tours 0 - 255)                      │
│    - Stream 1: Evaluates Chunk 1  (Tours 256 - 511)                    │
│    - Purpose: Maximize SM saturation; overlap host-device transfers.   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. P-Task (Task-Parallel Multi-Search / Island Model):                 │
│    - Multiple independent metaheuristic instances run in parallel.     │
│    - Stream 0: Island 0 (Fast GA, high mutation rate)                  │
│    - Stream 1: Island 1 (Memetic GA with deep 2-opt)                   │
│    - Stream 2: Island 2 (Simulated Annealing explorer)                 │
│    - Purpose: Algorithmic diversity; asynchronous migration.           │
│    - Contrasted with S-Task (single sequential search chain).          │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Stream Pipelining for P-Data Batching

When population size $P$ is very large (e.g., $P = 2{,}048$) and cannot fit into VRAM alongside a large distance matrix, the population is divided into $K$ chunks of size $P_{\text{chunk}} = P / K$.
Using two alternating CUDA streams (`stream_A`, `stream_B`) with pinned host memory (`cudaHostAlloc`), we pipeline data transfer with kernel execution:

```text
Time ──►
Stream A: [ Copy Chunk 0 (H2D) ] [ Kernel 2-Opt (Chunk 0) ] [ Copy Chunk 0 (D2H) ]
Stream B:                        [ Copy Chunk 1 (H2D)     ] [ Kernel 2-Opt (Chunk 1) ]
```

While the GPU's copy engine is transferring Chunk 1 over PCIe, the GPU's compute engines are simultaneously executing 2-opt on Chunk 0. The PCIe transfer latency is completely hidden behind computation.

### 3.3 Asynchronous Island Migration in P-Task Execution

In an Island Model metaheuristic operating in the P-Task regime, each island executes independently on its own dedicated CUDA stream. Every $G_{\text{mig}}$ generations (e.g., every 50 generations):
1. Island 0 copies its top 5 best tours to a shared host-pinned migration buffer using an asynchronous transfer on `stream_0`:
   `cp.cuda.Stream(null).record()`
2. Island 1 reads those tours and injects them into its mating pool.
3. Because the transfer is non-blocking, neither island ever halts or waits for the other.

---

## 4. Quantitative Diagnosis of PCIe Transfer Bottlenecks

The most critical architectural performance flaw in GPU computing is the **PCIe bus bottleneck**. Developers who do not understand PCIe bandwidth often write code that spends $90\%$ of its time moving data across the motherboard rather than computing on GPU cores.

### 4.1 Theoretical vs Real-World PCIe Interconnect Bandwidth

Data transferred between host RAM and GPU VRAM traverses the PCI Express (PCIe) bus.

| PCIe Generation | Bus Width | Theoretical Unidirectional Bandwidth | Real-World Sustained Bandwidth | Typical Small-Transfer Latency |
| :--- | :--- | :--- | :--- | :--- |
| **PCIe Gen 3.0** | $\times 16$ lanes | $15.75\text{ GB/s}$ | $\approx 12.0\text{--}12.5\text{ GB/s}$ | $10\text{--}15\,\mu\text{s}$ |
| **PCIe Gen 4.0** | $\times 16$ lanes | $31.51\text{ GB/s}$ | $\approx 24.5\text{--}26.0\text{ GB/s}$ | $8\text{--}12\,\mu\text{s}$ |
| **PCIe Gen 5.0** | $\times 16$ lanes | $63.02\text{ GB/s}$ | $\approx 50.0\text{--}54.0\text{ GB/s}$ | $5\text{--}8\,\mu\text{s}$ |

Contrast this with the internal memory bandwidth of GPU VRAM:
- **Pascal GP107 (GTX 1050) Internal VRAM Bandwidth:** $\mathbf{112.1\text{ GB/s}}$ ($\approx \mathbf{9.3\times}$ faster than PCIe Gen3!)
- **Ampere GA102 (RTX 3080) Internal VRAM Bandwidth:** $\mathbf{760.0\text{ GB/s}}$ ($\approx \mathbf{31\times}$ faster than PCIe Gen4!)

The PCIe bus is an extremely narrow, high-latency bridge connecting two vast memory continents.

### 4.2 The Naive Hybrid Antipattern: Per-Iteration Round-Trips

Consider the naive hybrid architecture commonly seen in unoptimized research prototypes:
- Python performs Selection and Crossover on the CPU (using NumPy).
- It copies the offspring to the GPU (`cp.asarray(offspring)`).
- GPU runs 2-opt local search ([two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)).
- It copies the optimized tours back to the CPU (`offspring.get()`) to compute survival.

```text
NAIVE HYBRID ANTIPATTERN:
[ CPU: Crossover ] ──( PCIe Transfer: 2 MB )──► [ GPU: 2-Opt ] ──( PCIe Transfer: 2 MB )──► [ CPU: Survival ]
▲                                                                                                          │
└────────────────────────── Repeat for 10,000 Generations! ───────────────────────────────────────────────┘
```

#### Quantitative Bandwidth & Latency Calculation:
Let population $P = 256$, number of cities $n = 1{,}000$, and generations $G = 10{,}000$.
In each generation:
1. **Host-to-Device (H2D):**
   $$\text{Bytes}_{\text{H2D}} = P \times n \times 4\text{ B} = 256 \times 1{,}000 \times 4 = 1{,}048{,}576\text{ B} = \mathbf{1.0\text{ MB}}$$
2. **Device-to-Host (D2H):**
   $$\text{Bytes}_{\text{D2H}} = P \times n \times 4\text{ B} = \mathbf{1.0\text{ MB}}$$
3. **Total Data Moved per Generation:**
   $$\text{Bytes}_{\text{gen}} = 2.0\text{ MB}$$

Over a complete run of $10{,}000\text{ generations}$:

$$\text{Total PCIe Traffic} = 10{,}000 \times 2.0\text{ MB} = \mathbf{20{,}000\text{ MB}} = \mathbf{20.0\text{ GB!}}$$

Now calculate the time spent waiting on PCIe on a Gen 3 system ($12\text{ GB/s}$ sustained, with $15\,\mu\text{s}$ driver invocation latency per transfer):

$$t_{\text{transfer}} = \frac{20.0\text{ GB}}{12.0\text{ GB/s}} = \mathbf{1.67\text{ seconds of pure PCIe bus transit}}$$

$$t_{\text{latency}} = 10{,}000 \times 2\text{ transfers} \times 15\,\mu\text{s} = \mathbf{0.30\text{ seconds of driver overhead}}$$

$$\text{Total PCIe Overhead} = 1.67 + 0.30 = \mathbf{1.97\text{ seconds!}}$$

If the GPU 2-opt kernel itself executes in $0.50\text{ seconds}$, the solver spends **$80\%$ of its total runtime waiting for data to crawl across the PCIe bus**. The GPU's massive computational speedup is completely erased!

### 4.3 The Single-Batch GPU-Resident Paradigm

In our architecture, we strictly enforce the **Single-Batch GPU-Resident Paradigm**:

```text
SINGLE-BATCH RESIDENT PARADIGM:
1. Load Problem & Init Population ──( ONE-TIME PCIe: 1 MB )──► VRAM
                                                                 │
                                                       [ Loop 10,000 Generations ]
                                                       - Tournament Selection (GPU)
                                                       - Order Crossover (GPU)
                                                       - Swap/Inversion Mutation (GPU)
                                                       - Batched 2-Opt (GPU)
                                                       - Cost Evaluation (GPU)
                                                       - (μ + λ) Survival (GPU)
                                                                 │
                                                                 ▼
[ Return Final Best Tour Only ] ◄──( ONE-TIME PCIe: 4 KB )─── VRAM
```

#### Quantitative Comparison:
1. **Initial Problem Data Load (H2D):**
   $$t_{\text{init}} = \frac{1.0\text{ MB}}{12.0\text{ GB/s}} + 15\,\mu\text{s} \approx \mathbf{0.0001\text{ s}} \quad (0.1\text{ ms})$$
2. **Generational Loop (VRAM Resident):**
   $$10{,}000\text{ generations} \implies \mathbf{0\text{ PCIe transfers!}}$$
   All selection indices, crossover masks, mutation swaps, and 2-opt deltas are evaluated and committed in on-chip SRAM and high-bandwidth GDDR5 VRAM ($112\text{ GB/s}$).
3. **Final Result Return (D2H):**
   Only the single winning tour $\pi^*$ of length $n = 1{,}000$ integers ($4\text{ KB}$) is copied back to host RAM:
   $$t_{\text{final}} \approx \mathbf{0.00001\text{ s}} \quad (0.01\text{ ms})$$

$$\text{Total PCIe Time} = 0.1\text{ ms} + 0.01\text{ ms} = \mathbf{0.11\text{ ms}} \quad (\text{vs } 1{,}970\text{ ms in the naive hybrid!})$$

**Speedup Factor:**
Eliminating per-iteration PCIe round-trips yields a **$17{,}900\times$ reduction in data transfer overhead**.

### 4.4 Roofline Model Comparison

The **Roofline Model** ([Williams, Waterman, & Patterson, 2009](#5-academic-references)) captures how algorithmic performance is bounded by either computational peak performance ($P_{\text{peak}}$, in GFLOP/s or GOp/s) or memory bandwidth ($B$, in GB/s), as a function of **Operational Intensity** $I$ (operations per byte):

$$\text{Attainable Performance} = \min\left( P_{\text{peak}}, \; I \times B \right)$$

```text
Attainable
Performance ▲
(GOp/s)     │                        Peak Compute Ceiling (P_peak)
            │                     ┌───────────────────────────────
            │                    /
            │                   /
            │                  /  Slope = VRAM Bandwidth (112 GB/s)
            │                 /
            │   Slope =      /
            │   PCIe Gen3   /
            │   (12 GB/s)  /
            │             /
            └────────────┴───────┴───────────────────────────────►
                         I_PCIe  I_VRAM             Operational Intensity
                                                    (Operations / Byte)
```

1. **Naive Hybrid GA:**
   - Bound by the PCIe slope: $B_{\text{PCIe}} \approx 12\text{ GB/s}$.
   - Operational intensity is throttled to near zero because $2\text{ MB}$ of data is moved across the bus for every minor evolutionary step.
   - The algorithm is severely **interconnect-bound**, operating in the lowest left quadrant of the Roofline chart.
2. **GPU-Resident Solver:**
   - Operates entirely within the VRAM domain: $B_{\text{VRAM}} = 112\text{ GB/s}$ (and on-chip SRAM bandwidth $> 1{,}600\text{ GB/s}$).
   - Operational intensity is high: once a tour tile is loaded into shared memory, thousands of 2-opt evaluations are performed per byte loaded.
   - The algorithm ascends the Roofline slope into the compute-saturated regime, unlocking the full arithmetic power of the GPU cores.

---

## 5. Academic and Industry References

1. **Williams, S., Waterman, A., & Patterson, D. (2009).** Roofline: An Insightful Visual Performance Model for Multicore Architectures. *Communications of the ACM*, 52(4), 65–76. DOI: [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785).
2. **NVIDIA Corporation (2024).** *CUDA C++ Best Practices Guide: Asynchronous Concurrent Execution and Memory Management*. Document ID: PG-02830-001_v12.4.
3. **Harris, M. (2012).** *How to Overlap Data Transfers in CUDA C/C++*. NVIDIA Developer Technical Blog.
4. **Kirk, D. B., & Hwu, W. W. (2016).** *Programming Massively Parallel Processors: A Hands-on Approach* (3rd ed.). Morgan Kaufmann, Cambridge, MA.
5. **Crainic, T. G., & Toulouse, M. (2003).** Parallel Strategies for Metaheuristics. In *Handbook of Metaheuristics*, pp. 475–513. Springer, Boston, MA.
6. **Rocki, K., & Suda, R. (2012).** Accelerating 2-Opt Local Search for the Traveling Salesman Problem on NVIDIA GPUs. In *Proceedings of the IEEE HPCS 2012*, pp. 489–495.
