---
title: "CUDA warp-level primitives: vote, shuffle, and parallel stream compaction mechanics"
description: >
  Pedagogical reference and visual guide to CUDA warp-level collective primitives
  (__ballot_sync, __all_sync, __any_sync, __activemask, and __shfl_*_sync). Explains
  Pascal GP107 (CC 6.1) vs. Volta+ (CC >= 7.0) execution models, mask RSVP semantics,
  sub-warp partitioning, classroom-announcement broadcast analogies, and step-by-step
  deterministic stream compaction for Davis (1985) Order Crossover (DP11-11).
created: 2026-11-08
date_created: 2026-11-08
date_changed: 2026-11-09
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-08
    modifications:
      - description: >
          Authored comprehensive guide on CUDA bit-manipulation intrinsics and warp-level
          vote/shuffle primitives.
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Refactored into dedicated warp-level primitives guide; decoupled foundational hardware
          and register bitwise arithmetic into companion guide; clarified Pascal CC 6.1 vs Volta+
          scheduling, sub-warps, teacher-broadcast analogies, and Davis 1985 algorithmic identity.
related_files:
  - "[guide-cuda-hardware-bitwise-and-registers.md](./guide-cuda-hardware-bitwise-and-registers.md)"
  - "[analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md)"
  - "[analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py)"
tags:
  - guide/cuda
  - guide/architecture
  - notes/cuda
  - notes/hardware
  - analysis/architecture
  - analysis/performance
  - algorithm/crossover
  - algorithm/two-opt
  - algorithm/genetic-algorithm
  - benchmark/gpu
---

# CUDA warp-level primitives: vote, shuffle, and parallel stream compaction mechanics

- [CUDA warp-level primitives: vote, shuffle, and parallel stream compaction mechanics](#cuda-warp-level-primitives-vote-shuffle-and-parallel-stream-compaction-mechanics)
  + [1. Table of symbols and notation](#1-table-of-symbols-and-notation)
  + [2. Terminology and official NVIDIA documentation](#2-terminology-and-official-nvidia-documentation)
    - [2.1 Intrinsics vs. Primitives vs. Library Functions](#21-intrinsics-vs-primitives-vs-library-functions)
    - [2.2 Where NVIDIA documents these functions and how developers discover them](#22-where-nvidia-documents-these-functions-and-how-developers-discover-them)
  + [3. Warp vote primitives (`__ballot_sync`, `__all_sync`, `__any_sync`, `__activemask`)](#3-warp-vote-primitives-__ballot_sync-__all_sync-__any_sync-__activemask)
    - [3.1 Compute Capability: Pascal GTX 1050 (CC 6.1) vs. Volta+ (CC $\\ge 7.0$)](#31-compute-capability-pascal-gtx-1050-cc-61-vs-volta-cc-ge-70)
    - [3.2 The `mask` argument: the RSVP attendance list](#32-the-mask-argument-the-rsvp-attendance-list)
    - [3.3 Signatures and behavior of the four vote primitives](#33-signatures-and-behavior-of-the-four-vote-primitives)
    - [3.4 Visual trace of `__ballot_sync` on an 8-city tour segment](#34-visual-trace-of-__ballot_sync-on-an-8-city-tour-segment)
    - [3.5 "And then what?": from ballot mask to output placement](#35-and-then-what-from-ballot-mask-to-output-placement)
  + [4. Warp shuffle primitives (`__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`)](#4-warp-shuffle-primitives-__shfl_sync-__shfl_up_sync-__shfl_down_sync-__shfl_xor_sync)
    - [4.1 Direct register-to-register communication without memory](#41-direct-register-to-register-communication-without-memory)
    - [4.2 Sub-Warps Decoded: Physical Hardware Reality, Register Isolation, and Limits on Parallelization](#42-sub-warps-decoded-physical-hardware-reality-register-isolation-and-limits-on-parallelization)
      + [1. The Physical Hardware Rule: Warps are Indivisible in Silicon](#1-the-physical-hardware-rule-warps-are-indivisible-in-silicon)
      + [2. What, Then, is a "Sub-Warp" in CUDA? (A Logical Shuffle Crossbar Mask)](#2-what-then-is-a-sub-warp-in-cuda-a-logical-shuffle-crossbar-mask)
      + [3. Do Sub-Warps Share Registers and Memory?](#3-do-sub-warps-share-registers-and-memory)
      + [4. What Limits GPU Parallelization?](#4-what-limits-gpu-parallelization)
      + [5. EXPLICIT RECTIFICATION: Why Our Real TSP Solver NEVER Uses Sub-Warps!](#5-explicit-rectification-why-our-real-tsp-solver-never-uses-sub-warps)
    - [4.3 Demystifying Direct Indexed Shuffle: the classroom announcement analogy](#43-demystifying-direct-indexed-shuffle-the-classroom-announcement-analogy)
    - [4.4 Matrix transformations of the four shuffle modes](#44-matrix-transformations-of-the-four-shuffle-modes)
  + [5. Step-by-step walkthrough: warp stream compaction in Order Crossover (`DP11-11`)](#5-step-by-step-walkthrough-warp-stream-compaction-in-order-crossover-dp11-11)
    - [5.1 The mental model: Empty Chairs and the Bus Queue with Blinders](#51-the-mental-model-empty-chairs-and-the-bus-queue-with-blinders)
    - [5.2 The "Blinders" (*Antolho*) bitmask matrix (`lower_lanes_mask`)](#52-the-blinders-antolho-bitmask-matrix-lower_lanes_mask)
    - [5.3 Complete numerical trace on $P\_1 = \[C, D, H, B, G, A, F, E\]$ and $P\_2 = \[A, B, C, D, E, F, G, H\]$](#53-complete-numerical-trace-on-p_1--c-d-h-b-g-a-f-e-and-p_2--a-b-c-d-e-f-g-h)
    - [5.4 Historical Lineage and the 40-Year Evolution of Order Crossover (1985–2026)](#54-historical-lineage-and-the-40-year-evolution-of-order-crossover-19852026)
      + [Phase 1: Lawrence Davis's Original 1985 Proposal (Symbolic Artificial Intelligence in Lisp)](#phase-1-lawrence-daviss-original-1985-proposal-symbolic-artificial-intelligence-in-lisp)
      + [Phase 2: The C and TSPLIB95 Era (Late 1980s – 1990s)](#phase-2-the-c-and-tsplib95-era-late-1980s--1990s)
      + [Phase 3: The Early GPGPU Era (2000s – 2010s: The 1-Thread Bottleneck)](#phase-3-the-early-gpgpu-era-2000s--2010s-the-1-thread-bottleneck)
      + [Phase 4: Modern Warp-Level Stream Compaction (2010s – Present: `DP11-11`)](#phase-4-modern-warp-level-stream-compaction-2010s--present-dp11-11)
    - [5.5 Strict iso-algorithmic bit-identity contract (`DP11-11`)](#55-strict-iso-algorithmic-bit-identity-contract-dp11-11)

---

## 1. Table of symbols and notation

| Symbol | C++ / CUDA Identifier | Mathematical Definition & Domain | Physical / Algorithmic Meaning |
| :--- | :--- | :--- | :--- |
| $w$ | `warpSize` | $w = 32$ | Constant number of threads (lanes) in one hardware warp |
| $\ell$ | `lane_id` | $\ell = \text{threadIdx.x} \bmod 32 \in \{0, 1, \dots, 31\}$ | Index of a thread inside its containing 32-thread warp |
| $M$ | `mask` | $M \in \{0, 1\}^{32}$ (e.g. `0xFFFFFFFFu`) | 32-bit participation bitmask specifying which lanes take part |
| $P(\ell)$ | `predicate` | $P(\ell) \in \{0, 1\}$ | Per-lane boolean condition evaluated independently by lane $\ell$ |
| $r_\ell$ | `rank` | $r_\ell = \sum_{k=0}^{\ell-1} P(k)$ | Number of active threads strictly to the left of lane $\ell$ (compaction offset) |
| $c_1, c_2$ | `c1, c2` | $0 \le c_1 < c_2 < n$ | Crossover cut points enclosing the preserved segment |

---

## 2. Terminology and official NVIDIA documentation

### 2.1 Intrinsics vs. Primitives vs. Library Functions

In CUDA programming, developers distinguish three tiers of functionality:

1. **Intrinsics (Hardware Instructions)**: Built-in compiler functions that map directly to **a single assembly instruction (SASS)** on the GPU ALU, such as `__popc(x)` (`POPC`) or `__clz(x)` (`FLO.CLZ`). They have zero function call overhead and execute in 1 to 4 clock cycles.
2. **Warp Primitives (Collectives)**: Functions that coordinate data exchange across the 32 threads of a warp using internal hardware datapath crossbars, such as `__ballot_sync` and `__shfl_sync`. They do not touch shared or global memory.
3. **Library Functions**: Multi-instruction routines implemented in software (e.g., CUDA Math `sin()`, Thrust algorithms, CUB primitives).

```ascii
Software Abstraction Layers in GPU Computing:
┌────────────────────────────────────────────────────────┐
│ High-Level Library Functions: std::sort(), sin(x)       │  → Written in C++; composed of dozens of assembly instructions.
├────────────────────────────────────────────────────────┤
│ Compiler Intrinsics: __popc(x), __clz(x)               │  → Built into nvcc compiler; map 1-to-1 to a single silicon instruction!
├────────────────────────────────────────────────────────┤
│ Warp-Level Primitives: __ballot_sync(), __shfl_sync()  │  → Hardware crossbar wires connecting the 32 registers of a warp.
├────────────────────────────────────────────────────────┤
│ Block-Level Synchronization: __syncthreads()           │  → Pauses up to 1024 threads; communicates through shared memory (SRAM).
└────────────────────────────────────────────────────────┘
```

NVIDIA documents these operations in two official locations in the *CUDA C++ Programming Guide*:

- **Appendix C.2 (Math Data Type Functions)**: Documents bit manipulation intrinsics (`__popc`, `__clz`, `__ffs`, `__brev`).
- **Section 7.22 (Warp Vote and Shuffle Functions)**: Documents warp collective primitives (`__ballot_sync`, `__all_sync`, `__any_sync`, `__shfl_*_sync`).

1. **Warp-Level Primitives (*Primitivas no Nível de Warp*)**:
   A "primitive" is the smallest indivisible hardware coordination operation. The 32 threads in a warp live on the same physical execution sub-core. They can share register values directly using dedicated internal silicon wires (**crossbar networks**) without reading or writing to cache or RAM[^harris-warp].

2. **Wall-Clock Time vs. Clock Cycles**:
   + **Wall-clock time** (`time.perf_counter()`): Physical time measured on a human stopwatch (milliseconds, seconds). It includes everything: kernel execution, PCIe transfers, operating system latency, and idle waiting.
   + **Clock cycles** (*ciclos de clock*): The internal discrete heartbeats of the GPU chip. At 1.493 GHz, 1 cycle takes 0.67 nanoseconds. Hardware instructions take an integer number of clock cycles to complete.

[^harris-warp]: Harris, M., & Perelygin, K. (2017). [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). NVIDIA Technical Blog.

### 2.2 Where NVIDIA documents these functions and how developers discover them

A developer does not have to memorize arbitrary names. NVIDIA documents these features in four official locations:

| Layer | Official Location | What It Contains |
| :--- | :--- | :--- |
| **1. CUDA Programming Guide** | Appendix B.14 & B.15[^cuda-vote] | Function signatures and semantics for `__all_sync`, `__any_sync`, `__ballot_sync`, and `__shfl_*_sync`. |
| **2. Math API Reference** | Appendix C.2 & C.3[^cuda-cast] | Bitwise operations (`__popc`, `__clz`, `__ffs`, `__brev`) and type casting (`__float_as_uint`). |
| **3. Toolkit Header Files** | `/usr/local/cuda/include/sm_30_intrinsics.h` | Raw C++ function declarations automatically included by `nvcc`. |
| **4. SASS Assembly Dump** | Command: `cuobjdump --dump-sass kernel.cubin` | Disassembles GPU machine code to confirm that `__popc` emitted a single `POPC R2, R0;` instruction. |

[^cuda-vote]: [CUDA C++ Programming Guide — Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions).
[^cuda-cast]: [CUDA C++ Programming Guide — Type Casting Intrinsics](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#type-casting-intrinsics).

---

## 3. Warp vote primitives (`__ballot_sync`, `__all_sync`, `__any_sync`, `__activemask`)

### 3.1 Compute Capability: Pascal GTX 1050 (CC 6.1) vs. Volta+ (CC $\ge 7.0$)

The user asked:
> *"Is `__ballot_sync` not used anymore in newer architectures with CC > 6.1?"*

**The exact opposite is true! `__ballot_sync` is the modern, mandatory standard for all CUDA architectures!**

Here is what actually happened across GPU generations:

- **Prior to CUDA 9 (Pascal CC 6.1 and older)**:
  NVIDIA provided unsuffixed functions: `__ballot(predicate)` and `__shfl(val, src)`.
  On Pascal GPUs (like the GeForce GTX 1050), all 32 threads in a warp executed instructions in strict hardware lockstep with a single shared Program Counter (PC).
- **Starting with Volta (CC 7.0+)**:
  NVIDIA introduced **Independent Thread Scheduling (ITS)**. Each thread has its own Program Counter and call stack, allowing threads inside the same warp to branch independently and make forward progress asynchronously.
  Under ITS, if threads call a warp collective without synchronizing, threads might execute at different times, causing deadlocks or reading stale values!
- **The Modern Standard**:
  To support Independent Thread Scheduling safely, NVIDIA introduced the `_sync` variants (`__ballot_sync`, `__shfl_sync`) requiring an explicit participation `mask`.
  The old unsuffixed intrinsics (`__ballot`, `__shfl`) were **officially deprecated** in CUDA 9 and are completely removed or issue compiler errors in modern CUDA toolchains.
  Therefore, `__ballot_sync` is **mandatory** on all modern architectures (Volta CC 7.0, Turing CC 7.5, Ampere CC 8.6, Ada Lovelace CC 8.9, Hopper CC 9.0, and Blackwell CC 10.0)!

### 3.2 The `mask` argument: the RSVP attendance list

Every warp collective function begins with `uint32_t mask`:

```cpp
uint32_t __ballot_sync(uint32_t mask, int predicate);
```

Think of `mask` as an **RSVP attendance list for a meeting**:

- Bit $k$ specifies whether Lane $k$ is expected to attend.
- When all 32 threads participate, every bit is `1`, written in hexadecimal as `0xFFFFFFFFu`:
  $$\texttt{0xFFFFFFFFu} = \texttt{0b}\underbrace{\texttt{11111111\_11111111\_11111111\_11111111}}_{\text{32 ones}}$$

> [!CAUTION]
> **The Deadlock Trap:**
> If you pass `mask = 0xFFFFFFFFu`, you are telling the hardware: *"Wait until all 32 threads arrive at this line."*
>
> If you put this inside an `if (threadIdx.x < 16)` statement, Threads 0..15 will wait forever for Threads 16..31 to arrive. The kernel hangs indefinitely in a hardware **deadlock**!
>
> When all 32 threads participate uniformly, passing `0xFFFFFFFFu` is completely safe and optimal.

### 3.3 Signatures and behavior of the four vote primitives

| Intrinsic Signature | Return Type | What it does in plain python terms | Practical TSP Solver Use Case |
| :--- | :--- | :--- | :--- |
| `__all_sync(mask, pred)` | `int` (`0` or `1`) | `all(pred[i] for i in range(32))` | Convergence check: did all 32 threads confirm tour validity? |
| `__any_sync(mask, pred)` | `int` (`0` or `1`) | `any(pred[i] for i in range(32))` | Early exit: did any thread find an improving 2-opt move ($\Delta < 0$)? |
| `__ballot_sync(mask, pred)` | `uint32_t` (32 bits) | Gathers 32 boolean flags into a single 32-bit integer | **Stream compaction**: identifying which candidate cities from $P_2$ are missing. |
| `__activemask()` | `uint32_t` (32 bits) | Bitmask of threads currently executing at this line | Dynamic warp attendance query for diverged branches. |

### 3.4 Visual trace of `__ballot_sync` on an 8-city tour segment

Consider an 8-thread group ($\ell \in \{0 \dots 7\}$) evaluating candidate cities from $P_2 = [A, B, C, D, E, F, G, H]$.

Suppose cities $\{A, B, D, G\}$ are already present in $P_1$'s segment, so only cities $\{C, E, F, H\}$ (at lanes $2, 4, 5, 7$) are valid:

$$\begin{array}{rcccccccc}
\text{Lane } \ell: & \text{Lane 0} & \text{Lane 1} & \textbf{Lane 2} & \text{Lane 3} & \textbf{Lane 4} & \textbf{Lane 5} & \text{Lane 6} & \textbf{Lane 7} \\
\text{City}: & A & B & \mathbf{C} & D & \mathbf{E} & \mathbf{F} & G & \mathbf{H} \\
\text{Predicate } P(\ell): & 0 & 0 & \mathbf{1} & 0 & \mathbf{1} & \mathbf{1} & 0 & \mathbf{1} \\
\text{Bit Weight } 2^\ell: & 1 & 2 & 4 & 8 & 16 & 32 & 64 & 128
\end{array}$$

When the threads execute `ballot_mask = __ballot_sync(0xFFFFFFFFu, is_valid);`:
The hardware bundles the 8 private boolean conditions into a single integer:

$$B = 0 \cdot 1 + 0 \cdot 2 + 1 \cdot 4 + 0 \cdot 8 + 1 \cdot 16 + 1 \cdot 32 + 0 \cdot 64 + 1 \cdot 128 = 180 = \texttt{0b10110100}$$

This 32-bit integer $B$ is simultaneously deposited into a private register of **every thread in the warp** in **1 single clock cycle**!

### 3.5 "And then what?": from ballot mask to output placement

The user asked:
> *"We get the ballot mask `0b10110100`. And then what?"*

Now, every thread uses this integer to calculate its exact write position without communicating or colliding with any other thread:

1. **Each thread creates blinders (`lower_lanes_mask`)**:
   Lane $\ell$ only cares about threads to its left:
   ```cpp
   uint32_t lower_lanes_mask = (1u << lane_id) - 1u;
   ```
2. **Each thread computes its write rank**:
   ```cpp
   int rank = __popc(ballot_mask & lower_lanes_mask);
   ```
   - **Lane 2 ($C$)**: `lower_lanes_mask` is `0b00000011`.
     `0b10110100 & 0b00000011 = 0`.
     `rank = __popc(0) = 0` $\implies$ writes to **Slot 0**!
   - **Lane 4 ($E$)**: `lower_lanes_mask` is `0b00001111`.
     `0b10110100 & 0b00001111 = 0b00000100`.
     `rank = __popc(4) = 1` $\implies$ writes to **Slot 1**!
   - **Lane 5 ($F$)**: `lower_lanes_mask` is `0b00011111`.
     `0b10110100 & 0b00011111 = 0b00010100`.
     `rank = __popc(20) = 2` $\implies$ writes to **Slot 2**!
   - **Lane 7 ($H$)**: `lower_lanes_mask` is `0b01111111`.
     `0b10110100 & 0b01111111 = 0b00110100`.
     `rank = __popc(52) = 3` $\implies$ writes to **Slot 3**!
3. **Each valid thread writes to the child tour**:
   ```cpp
   if (is_valid) {
       child[write_pos + rank] = my_city;
   }
   ```
4. **All threads update the shared write position**:
   ```cpp
   write_pos += __popc(ballot_mask); // Exactly 4 cities were written!
   ```

In **less than 5 clock cycles**, all missing cities are packed contiguously into the child tour with zero shared memory traffic and zero race conditions!

---

## 4. Warp shuffle primitives (`__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`)

### 4.1 Direct register-to-register communication without memory

While `__ballot_sync` exchanges 1 bit per thread, **Warp Shuffle primitives** exchange full 32-bit variables (`int`, `float`) directly between thread registers across high-speed internal silicon wires (the register file crossbar).

### 4.2 Sub-Warps Decoded: Physical Hardware Reality, Register Isolation, and Limits on Parallelization

The user asked:
> *"When do we use sub-warps? Until now I thought warps (32 threads) were the smallest hardware size I could use. Can I partition into sub-warps and use each thread individually? Do sub-warps share memory and registers? If I can use warps and sub-warps, what limits parallelization? Cache memory?*
>
> *Or is a sub-warp not real hardware and was just an example? If that's the case, please rectify the examples!"*

This is one of the most vital questions for understanding GPU hardware! Let us establish the physical ground truth:

#### 1. The Physical Hardware Rule: Warps are Indivisible in Silicon

Inside an NVIDIA Streaming Multiprocessor (SM), the instruction dispatcher **physically operates ONLY on 32-thread warps**.
- There is **no silicon circuit** that can dispatch an instruction to 8 threads alone or 4 threads alone.
- When an SM executes an instruction, exactly 32 ALU pipelines fire on the exact same clock cycle.
- **A warp of 32 threads is the absolute smallest execution unit that exists in NVIDIA hardware!**

#### 2. What, Then, is a "Sub-Warp" in CUDA? (A Logical Shuffle Crossbar Mask)

In CUDA, the term "sub-warp" (or sub-warp collective, configured via the `width` parameter in `__shfl_sync` or `cg::tiled_partition<8>`) is **purely a software/routing filter inside the register crossbar**.
- All 32 threads in the warp still execute the instruction at the **exact same physical clock tick**.
- However, the internal crossbar switches are instructed to **partition communication boundaries**:

  ```ascii
  Hardware Warp (All 32 threads execute together at Cycle T):
  ┌───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
  │ Sub-Warp 0: Lanes 0..7│ Sub-Warp 1: Lanes 8..15│ Sub-Warp 2: Lanes 16..23│ Sub-Warp 3: Lanes 24..31│
  └───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
  ◄──── Partition 0 ─────►◄──── Partition 1 ─────►◄──── Partition 2 ─────►◄──── Partition 3 ─────►
  ```

- If `width = 8`, Lane 9 reads from Lane 8 (the base of its 8-thread partition), completely isolated from Lane 0.
- **When is this used in GPU engineering?**
  Only when an algorithm has very small, independent micro-tasks (e.g. processing $4 \times 4$ image pixel tiles or 8-element quaternions) and you want 1 single 32-thread hardware warp to process 4 independent micro-tasks simultaneously without crosstalk.

#### 3. Do Sub-Warps Share Registers and Memory?

- **Registers are NEVER Shared!**
  Each thread in CUDA has its own private, isolated physical 32-bit registers (`R0..R255`). Lane 0 cannot see Lane 1's registers, and Sub-Warp 0 cannot see Sub-Warp 1's registers, except through explicit shuffle instructions.
- **Shared Memory is Shared at the Block Level**:
  All threads in the entire thread block (typically 256 threads = 8 warps) share the exact same 48 KB of on-chip SRAM.

#### 4. What Limits GPU Parallelization?
The total number of concurrent warps an SM can run is strictly governed by four physical hardware ceilings:
1. **Register Budget**: A GP107 SM has 65,536 registers. If a kernel uses 32 registers per thread, each warp consumes $32 \times 32 = 1{,}024$ registers. Maximum warps supported by registers = $65{,}536 / 1{,}024 = \mathbf{64\text{ warps}}$ ($2{,}048$ threads). If a kernel spills and uses 64 registers per thread, occupancy drops to 32 warps.
2. **Shared Memory Capacity**: Each GP107 SM has 48 KB of SRAM. If each block requests 16 KB, the SM can host at most $48 / 16 = \mathbf{3\text{ blocks}}$ simultaneously.
3. **Warp Scheduler Limits**: Pascal GP107 has 4 warp schedulers per SM, with a maximum architectural cap of **64 active warps (2,048 threads) per SM**.
4. **Memory Bus Bandwidth (DRAM Roofline)**: GDDR5 memory delivers 128 bytes per memory transaction. If threads access memory without coalescing, the memory bus chokes and starves the compute pipelines.

#### 5. EXPLICIT RECTIFICATION: Why Our Real TSP Solver NEVER Uses Sub-Warps!
> [!IMPORTANT]
> **Clarification of Documentation Examples:**
> In our real GPU TSP solver (`order_crossover.cu`, `tournament_selection.cu`, `inversion_mutation.cu`), **WE NEVER USE SUB-WARPS!**
> We strictly use the **full 32-thread hardware warp** (`width = 32`, `mask = 0xFFFFFFFFu`).
>
> **Why did the documentation use 8 lanes in the diagram?**
> The 8-lane example was **strictly a pedagogical visual aid**.
> An 8-city tour ($P_1 = [C, D, H, B, G, A, F, E]$) fits cleanly on a single page, and an 8-bit mask (`0b11110000`) can be traced in your head.
> If we displayed a 32-city tour with a 32-column table and a 32-digit binary mask (`0b11111111_00000000_...`), the table would be completely unreadable on a screen!
> **In actual execution: 1 Warp = 32 Threads, processing routes of size $n \in \{52, 100, 318, 1002, 2103\}$ using warp-stride loops of 32!**

### 4.3 Demystifying Direct Indexed Shuffle: the classroom announcement analogy

The user asked:
> *"Why is $x_0$ substituted on Direct Indexed Shuffle? This is greek to me."*

Let us use an intuitive analogy:
> **The Classroom Analogy**:
> Imagine a classroom with 32 students (Lanes 0..31).
> The teacher (Lane 0) runs the tournament selection and finds that the winning route is **Route 42**.
> Currently:
> - Student 0's notebook says: `42`
> - Student 1's notebook says: `15`
> - Student 2's notebook says: `88`
> - ...
> The teacher speaks into a microphone: *"Everyone, today's winning route is 42!"*
> All 32 students immediately write `42` into their notebooks.

This is exactly what `__shfl_sync(0xFFFFFFFFu, my_route, 0)` does:

```cpp
// Lane 0 holds best_route. All lanes execute:
int winner = __shfl_sync(0xFFFFFFFFu, best_route, 0);
```

- Parameter `srcLane = 0` tells the hardware: *"Read the register of Lane 0 and copy its value to everyone."*
- Every thread's local variable `winner` now holds `42`.
- It takes **1 single clock cycle**!
- Why not use Shared Memory? Because writing to shared memory, waiting for `__syncthreads()`, and reading it back takes **30 to 50 clock cycles** and consumes shared memory SRAM. The shuffle broadcast is 40× faster and uses 0 bytes of memory!

### 4.4 Matrix transformations of the four shuffle modes

Let $\mathbf{x} = \begin{bmatrix} x_0 & x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & x_7 \end{bmatrix}$ be register values in an 8-lane sub-warp before the shuffle:

1. **Broadcast (`__shfl_sync(M, x, 0)`)**:
   Every lane receives Lane 0's value:

   $$\begin{bmatrix} x_0 & x_0 & x_0 & x_0 & x_0 & x_0 & x_0 & x_0 \end{bmatrix}$$

   *Application: Broadcasting tournament winner ID or crossover cut points.*

2. **Shift Up with $\delta = 2$ (`__shfl_up_sync(M, x, 2)`)**:
   Lanes $\ell \ge 2$ copy from $\ell - 2$; lower lanes keep their own value:

   $$\begin{bmatrix} x_0 & x_1 & \mathbf{x_0} & \mathbf{x_1} & \mathbf{x_2} & \mathbf{x_3} & \mathbf{x_4} & \mathbf{x_5} \end{bmatrix}$$

   *Application: Parallel prefix scan (cumulative tour distance).*

3. **Shift Down with $\delta = 4$ (`__shfl_down_sync(M, x, 4)`)**:
   Lanes copy from $\ell + 4$:

   $$\begin{bmatrix} \mathbf{x_4} & \mathbf{x_5} & \mathbf{x_6} & \mathbf{x_7} & x_4 & x_5 & x_6 & x_7 \end{bmatrix}$$

   *Application: Log-step tree reductions (finding the minimum cost edge or 2-opt swap in 5 steps).*

4. **Butterfly XOR with `laneMask = 1` (`__shfl_xor_sync(M, x, 1)`)**:
   Neighbors swap values pairwise ($0 \leftrightarrow 1, 2 \leftrightarrow 3, \dots$):

   $$\begin{bmatrix} \mathbf{x_1} & \mathbf{x_0} & \mathbf{x_3} & \mathbf{x_2} & \mathbf{x_5} & \mathbf{x_4} & \mathbf{x_7} & \mathbf{x_6} \end{bmatrix}$$

   *Application: Bitonic sorting networks and symmetric neighbor exchanges.*

---

## 5. Step-by-step walkthrough: warp stream compaction in Order Crossover (`DP11-11`)

### 5.1 The mental model: Empty Chairs and the Bus Queue with Blinders

Order Crossover copies a slice from Parent 1 into `child`, and then fills the vacant positions with the remaining cities of Parent 2 in their original relative order.

Imagine 8 passengers (threads $\ell \in \{0 \dots 7\}$) boarding a bus:
- Passengers whose cities were already copied from $P_1$ step out of line (`is_valid = 0`).
- Passengers who need a seat raise their hand (`is_valid = 1`).
- Every passenger puts on **horse blinders (*antolhos*)** so they can only look to their left!
- Passenger $\ell$ counts how many raised hands are visible to their left:
  + 0 hands visible $\implies$ take **Seat 0**!
  + 1 hand visible $\implies$ take **Seat 1**!
  + 2 hands visible $\implies$ take **Seat 2**!
  + 3 hands visible $\implies$ take **Seat 3**!

Nobody cuts in line. Seat assignment is **100% mathematically deterministic** and preserves the exact order of Parent 2!

### 5.2 The "Blinders" (*Antolho*) bitmask matrix (`lower_lanes_mask`)

Each thread computes its blinders mask in 1 cycle:

$$\texttt{lower\_lanes\_mask} = (1\text{u} \ll \text{lane\_id}) - 1\text{u}$$

Stacking `lower_lanes_mask` for an 8-lane sub-warp produces the strictly lower-triangular matrix $\mathbf{L}$:

$$\mathbf{L} = \begin{array}{r|cccccccc|l}
& \text{Bit 0} & \text{Bit 1} & \text{Bit 2} & \text{Bit 3} & \text{Bit 4} & \text{Bit 5} & \text{Bit 6} & \text{Bit 7} & \text{Hex Value} \\
\hline
\text{Lane } 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0x00} \quad (0) \\
\text{Lane } 1 & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0x01} \quad (1) \\
\text{Lane } 2 & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & 0 & \texttt{0x03} \quad (3) \\
\text{Lane } 3 & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & 0 & \texttt{0x07} \quad (7) \\
\text{Lane } 4 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & \texttt{0x0F} \quad (15) \\
\text{Lane } 5 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & \texttt{0x1F} \quad (31) \\
\text{Lane } 6 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & 0 & \texttt{0x3F} \quad (63) \\
\text{Lane } 7 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} & 0 & \texttt{0x7F} \quad (127)
\end{array}$$

Row $\ell$ has ones **only for threads strictly to the left of lane $\ell$**.
When Lane $\ell$ computes `rank = __popc(ballot_mask & lower_lanes_mask);`:
1. `ballot_mask & lower_lanes_mask` zeroes out all threads to the right.
2. `__popc(...)` counts the remaining valid threads to the left in **1 clock cycle**!

### 5.3 Complete numerical trace on $P_1 = [C, D, H, B, G, A, F, E]$ and $P_2 = [A, B, C, D, E, F, G, H]$

- **Cities**: $A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7$.
- **Parent 1**: $P_1 = [2, 3, 7, 1, 6, 0, 5, 4]$
- **Parent 2**: $P_2 = [0, 1, 2, 3, 4, 5, 6, 7]$
- **Cut Points**: $c_1 = 2, c_2 = 5$.
  Preserved segment: $\text{child}[2..5] = P_1[2..5] = [H, B, G, A] = [7, 1, 6, 0]$.
  Lookup table in shared memory: `in_segment[7] = in_segment[1] = in_segment[6] = in_segment[0] = 1`.
  The 4 vacant positions in `child` (starting cyclically at $(c_2 + 1) \bmod 8 = \mathbf{6}$) are slots **6, 7, 0, 1**.

The 8 threads stream candidate cities from $P_2$ starting from cyclic offset 6 (`p2_pos = (6 + lane_id) % 8`):

| Variable / Step | Lane 0 | Lane 1 | Lane 2 | Lane 3 | **Lane 4** | **Lane 5** | **Lane 6** | **Lane 7** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate Index `p2_pos`** | $6$ | $7$ | $0$ | $1$ | $\mathbf{2}$ | $\mathbf{3}$ | $\mathbf{4}$ | $\mathbf{5}$ |
| **City from $P_2$** | $G(6)$ | $H(7)$ | $A(0)$ | $B(1)$ | **$C(2)$** | **$D(3)$** | **$E(4)$** | **$F(5)$** |
| **`in_segment[city]`** | `1` (skip) | `1` (skip) | `1` (skip) | `1` (skip) | **`0` (valid)** | **`0` (valid)** | **`0` (valid)** | **`0` (valid)** |
| **Predicate `is_valid`** | `0` | `0` | `0` | `0` | **`1`** | **`1`** | **`1`** | **`1`** |
| **`ballot_mask`** | `0b11110000` | `0b11110000` | `0b11110000` | `0b11110000` | `0b11110000` | `0b11110000` | `0b11110000` | `0b11110000` |
| **`lower_lanes_mask`** | `0b00000000` | `0b00000001` | `0b00000011` | `0b00000111` | `0b00001111` | `0b00011111` | `0b00111111` | `0b01111111` |
| **Bitwise AND** | `0b00000000` | `0b00000000` | `0b00000000` | `0b00000000` | `0b00000000` | `0b00010000` | `0b00110000` | `0b01110000` |
| **`rank = __popc(...)`** | `0` | `0` | `0` | `0` | **`0`** | **`1`** | **`2`** | **`3`** |
| **Target in `child`** | *Skipped* | *Skipped* | *Skipped* | *Skipped* | **Slot 6** | **Slot 7** | **Slot 0** | **Slot 1** |
| **Value Written** | — | — | — | — | **$C$** | **$D$** | **$E$** | **$F$** |

The resulting child tour is:

$$\text{child} = [\underbrace{E}_{\text{Slot 0}}, \, \underbrace{F}_{\text{Slot 1}}, \, \underbrace{H, \, B, \, G, \, A}_{\text{Preserved } P_1[2..5]}, \, \underbrace{C}_{\text{Slot 6}}, \, \underbrace{D}_{\text{Slot 7}}]$$

Every single city is visited exactly once, and the cyclic relative order of $P_2$ ($C \to D \to E \to F$) is strictly preserved.

### 5.4 Historical Lineage and the 40-Year Evolution of Order Crossover (1985–2026)

The user asked:
> *"Did Davis propose using bitwise operations even on serialized sequential loops? Python didn't exist in 1985, so this must have been done in C or Lisp and could maybe be parallelized. How was the parallelization of OX developed over time? Did the original proposal use bit manipulation techniques?"*

Let us trace the four-decade historical and algorithmic journey of Order Crossover from its invention to modern CUDA GPU architecture:

#### Phase 1: Lawrence Davis's Original 1985 Proposal (Symbolic Artificial Intelligence in Lisp)

In 1985, Lawrence Davis published *Applying adaptive algorithms to epistatic domains* at IJCAI 1985[^davis1985].
- **Did Davis use bitwise operations?** **NO, absolutely not.**
- Davis was an AI researcher working in **Lisp**. In Lisp, individuals were represented as symbolic linked lists of symbols: `'(C D H B G A F E)`.
- Davis defined Order Crossover purely as a **high-level symbolic permutation rule**:
  1. Copy a contiguous sublist from Parent 1 to Child.
  2. Starting immediately after the cut, iterate through the elements of Parent 2.
  3. If an element is already a member of the child sublist (tested via Lisp's `member` predicate), skip it; otherwise, append it to the child in cyclic order.
- The concept of bit manipulation, bit shifts, or binary masks was completely absent from Davis's formulation.

#### Phase 2: The C and TSPLIB95 Era (Late 1980s – 1990s)

As Genetic Algorithms transitioned into C and Fortran (e.g. David Goldberg's Simple Genetic Algorithm in C, 1989; and Gerhard Reinelt's TSPLIB95 benchmark suite, 1995):
- Tours were represented as 1D integer arrays: `int parent1[n], parent2[n], child[n]`.
- Because C lacks high-level set types, developers implemented the membership test in one of two ways:
  1. **Nested Linear Scan ($\mathcal{O}(n^2)$)**:
     ```c
     // Classic naive C loop: O(n) scan inside an O(n) loop = O(n^2) work!
     for (int j = 0; j < n; j++) {
         int city = parent2[(c2 + 1 + j) % n];
         int found = 0;
         for (int k = c1; k <= c2; k++) {
             if (parent1[k] == city) { found = 1; break; }
         }
         if (!found) child[write_pos++] = city;
     }
     ```
  2. **Direct-Address Byte Lookup ($\mathcal{O}(n)$)**:
     ```c
     // Optimized C implementation: O(1) table lookup
     char in_segment[MAX_CITIES] = {0};
     for (int k = c1; k <= c2; k++) in_segment[parent1[k]] = 1;
     for (int j = 0; j < n; j++) {
         int city = parent2[(c2 + 1 + j) % n];
         if (!in_segment[city]) child[write_pos++] = city;
     }
     ```
- Notice: Even in C, no bitwise operations were used. An ordinary array of `char` or `int` was standard.

#### Phase 3: The Early GPGPU Era (2000s – 2010s: The 1-Thread Bottleneck)
When CUDA was introduced in 2007, early GPU GA researchers (e.g., Wong & Wong, 2004; Tsutsui & Fujimoto, 2009; Jaros, 2012) attempted to parallelize Genetic Algorithms by assigning **1 single GPU thread per route**:
- Thread $t$ ran the exact serial C `while` loop to construct Route $t$.
- **Why this failed**: As we proved in `BUG-22` and [analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md), assigning 1 thread per route causes:
  1. Massive cache thrashing and uncoalesced memory access.
  2. Severe warp divergence when routes have different cut lengths.
  3. Underutilization of the GPU: launching $N=256$ threads occupied only 2 out of 5 SMs on a GTX 1050, leaving the GPU 40% idle!

#### Phase 4: Modern Warp-Level Stream Compaction (2010s – Present: `DP11-11`)
The breakthrough that enabled true GPU acceleration of Order Crossover came from modern parallel prefix-sum and stream-compaction research (Billeter et al., 2009, *Efficient Stream Compaction on Many-Core Architectures*; Duane Merrill & Mark Harris, NVIDIA CUB):
- **The Core Realization**:
  The task *"take an array, test a condition, and pack only the surviving elements contiguously"* is called **Stream Compaction**.
- On a CPU, stream compaction is done with a sequential loop (`write_pos++`).
- On a GPU, 32 threads can execute stream compaction in **1 single clock cycle** using hardware vote and population count instructions:
  1. `__ballot_sync` gathers 32 boolean conditions into a 32-bit register.
  2. `__popc(ballot & lower_lanes_mask)` computes each thread's destination rank directly in ALU silicon!
- **Conclusion**:
  Bitwise operations (`__ballot_sync`, `__popc`, `1u << lane`) are **NOT a modification of Davis's algorithm**!
  They are the **parallel hardware realization** that allows 32 GPU threads to execute Davis's exact 1985 sequential logic simultaneously without serial loops, without thread collisions, and with **100% bit-identity** to the original mathematical specification!

### 5.5 Strict iso-algorithmic bit-identity contract (`DP11-11`)

Because the rank calculation is a pure mathematical function of `lane_id`:

$$r_\ell = \sum_{k=0}^{\ell-1} P(k)$$

For any two active lanes $\ell_1 < \ell_2$, it is strictly guaranteed that $r_{\ell_1} < r_{\ell_2}$.
Lane 4 is guaranteed to write before Lane 5, which writes before Lane 6, which writes before Lane 7.
There is **zero non-determinism**. For the same parents and cut points, `CrossoverGPUOrder` produces output that is identical to `CrossoverOrder` to the exact bit!
