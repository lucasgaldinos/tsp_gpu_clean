---
title: "Analysis of Shared-Memory Crossover Formats: Direct Byte Array (Format A) vs. Bitpacked Bitmap (Format B)"
description: >
  In-depth comparative analysis of CUDA shared-memory data structures for Order Crossover (OX)
  duplicate detection in GPU-accelerated Genetic Algorithms: Direct Byte Array (Format A) versus
  Bitpacked Bitmap (Format B). Evaluates instruction-level memory latency, write-hazard serialization
  and atomic contention, and shared-memory occupancy limits on Pascal GP107 hardware across
  the complete TSPLIB95 instance suite.
created: 2026-11-11
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-11
    modifications:
      - description: >
          Authored technical analysis contrasting Format A (uint8_t byte array) with Format B
          (uint32_t bitpacked bitmap) for Order Crossover shared-memory duplicate detection.
          Formalized algorithmic requirements, LDS.U8 versus atomicOr instruction mechanics,
          GTX 1050 (GP107) hardware occupancy budgets (48 KB SRAM limit), and TSPLIB95 scalability boundaries.
related_files:
  - "[analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md)"
  - "[guide-cuda-hardware-bitwise-and-registers.md](./guide-cuda-hardware-bitwise-and-registers.md)"
  - "[guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md)"
  - "[order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
tags:
  - algorithm/crossover
  - algorithm/genetic-algorithm
  - algorithm/tsp
  - analysis/architecture
  - analysis/memory
  - analysis/performance
  - analysis/scaling
  - guide/cuda
  - notes/cuda
  - notes/hardware
---

# Analysis of Shared-Memory Crossover Formats: Direct Byte Array (Format A) vs. Bitpacked Bitmap (Format B)

- [Analysis of Shared-Memory Crossover Formats: Direct Byte Array (Format A) vs. Bitpacked Bitmap (Format B)](#analysis-of-shared-memory-crossover-formats-direct-byte-array-format-a-vs-bitpacked-bitmap-format-b)
  + [1. The Algorithmic Need: Duplicate Membership Testing in Order Crossover (OX)](#1-the-algorithmic-need-duplicate-membership-testing-in-order-crossover-ox)
  + [2. Format A: Direct Byte Array (`uint8_t in_segment[n]`)](#2-format-a-direct-byte-array-uint8_t-in_segmentn)
    - [2.1 Structural Representation](#21-structural-representation)
    - [2.2 Parallel Marking Phase (Zero Collisions, Zero Atomics)](#22-parallel-marking-phase-zero-collisions-zero-atomics)
    - [2.3 Query Phase (Direct LDS.U8 Load)](#23-query-phase-direct-ldsu8-load)
    - [2.4 Occupancy & Capacity Math on Pascal GP107 (GeForce GTX 1050)](#24-occupancy--capacity-math-on-pascal-gp107-geforce-gtx-1050)
    - [2.5 Concrete Shared-Memory Footprints Across the TSPLIB95 Cohort](#25-concrete-shared-memory-footprints-across-the-tsplib95-cohort)
  + [3. Format B: Bitpacked Array](#3-format-b-bitpacked-array)
    - [3.1 Structural Representation & Footprint Reduction](#31-structural-representation--footprint-reduction)
    - [3.2 Bitwise Coordinate Arithmetic](#32-bitwise-coordinate-arithmetic)
    - [3.3 The Write Hazard: Race Conditions & Atomic Serialization](#33-the-write-hazard-race-conditions--atomic-serialization)
    - [3.4 Query Phase Overhead](#34-query-phase-overhead)
    - [3.5 Scalability Ceiling](#35-scalability-ceiling)
  + [4. Architectural Decision Matrix & Comparative Analysis](#4-architectural-decision-matrix--comparative-analysis)
    - [4.1 Comprehensive Architectural Comparison](#41-comprehensive-architectural-comparison)
    - [4.2 Production Rationalization for TSPLIB95](#42-production-rationalization-for-tsplib95)
    - [4.3 Scalability Roadmap Integration](#43-scalability-roadmap-integration)
  + [5. Repository Implementation Anchors](#5-repository-implementation-anchors)
  + [6. References](#6-references)

---

## 1. The Algorithmic Need: Duplicate Membership Testing in Order Crossover (OX)

In genetic algorithms applied to permutation optimization problems such as the Traveling Salesman Problem (TSP), recombination operators must preserve edge and ordering information while producing strictly valid permutation offspring without duplicate or omitted cities (Davis, 1985; Oliver et al., 1987).

Order Crossover (`OX`) operates on two parent permutations, $P_1$ and $P_2$, by selecting two cut boundaries $i$ and $j$ ($0 \le i \le j < n$). The offspring $C$ inherits the designated permutation segment directly from Parent 1:

$$C[k] = P_1[k], \quad \forall k \in [i, j]$$

The remaining positions in $C$ must be filled by iterating through Parent 2 in circular (cyclic) order starting immediately after the second cut point, $(j + 1) \bmod n$, and copying candidate cities while strictly skipping any city that was already transferred from $P_1[i:j]$. This step requires evaluating a set membership predicate:

$$\text{is\_duplicate}(\text{city}) = \begin{cases} 1, & \text{if } \text{city} \in P_1[i:j] \\ 0, & \text{otherwise} \end{cases}$$

On general-purpose CPUs (e.g., standard CPython execution), membership testing is implemented via hash tables (`std::unordered_set` or python `set`), achieving average-case $\mathcal{O}(1)$ query time and $\mathcal{O}(n)$ total crossover runtime. However, when porting Order Crossover naively to NVIDIA GPUs, allocating dynamic hash tables per thread or per route is prohibitive. A naive GPU implementation that linearly scans the transferred slice $P_1[i:j]$ for each candidate city incurs $\mathcal{O}(n)$ steps per city, inflating overall kernel execution time to $\mathcal{O}(n^2)$ (`BUG-22`).

To restore optimal $\mathcal{O}(n)$ linear scaling on GPU architectures, the membership test must be evaluated in on-chip CUDA Shared Memory (`__shared__`, SRAM). Two distinct physical formats exist for structuring this lookup table:

1. **Format A**: Direct Byte Array (`uint8_t in_segment[n]`), allocating one full byte per city.
2. **Format B**: Bitpacked Bitmap (`uint32_t bitmap[\lceil n/32 \rceil]`), compressing 32 city flags into each 32-bit integer word.

---

## 2. Format A: Direct Byte Array (`uint8_t in_segment[n]`)

### 2.1 Structural Representation

In Format A, every city index $c \in \{0, 1, \dots, n-1\}$ maps directly to an 8-bit unsigned integer (`uint8_t` or `unsigned char`) allocated in shared memory:

$$\text{SRAM Footprint per Route} = n \times 1\text{ byte} = n\text{ bytes}$$

Because city indices in standard TSP benchmarks form a contiguous range $[0, n-1]$, the city identifier serves directly as its own memory offset. No hash functions, perturbation chains, or modulo reductions are required.

```text
City Index c:        0      1      2      3     ...    n-2    n-1
Shared Memory Byte: [0x01, 0x00, 0x01, 0x00,   ...,  0x00,  0x01]
                     ▲             ▲                        ▲
                     └─ Preserved ─┴────────────────────────┘
```

### 2.2 Parallel Marking Phase (Zero Collisions, Zero Atomics)

During the segment inheritance phase, threads in the warp cooperatively read the slice $P_1[i:j]$ and mark the corresponding flags in shared memory:

```cpp
// Cooperative mark phase executed by threads in the warp
for (int k = lane_id; k < seg_len; k += 32) {
    int city = parent1[c1 + k];
    if (city >= 0 && city < n) {
        my_in_seg[city] = 1;
    }
}
__syncwarp();
```

Because every city in a valid TSP tour is strictly unique, no two threads in the warp can ever process the same city index. Consequently:

- **Unique Memory Addresses**: Every thread writes to an independent byte address in shared memory.
- **Zero Race Conditions**: Write-after-write (WAW) hazards are mathematically impossible.
- **Zero Atomics**: Marking is achieved through standard register-to-shared stores without requiring `atomicOr` or serialization barriers.

### 2.3 Query Phase (Direct LDS.U8 Load)

When streaming candidates from Parent 2, each thread queries membership via a single direct load:

```cpp
bool is_duplicate = (my_in_seg[city] != 0);
```

At the hardware instruction level (SASS), this compiles to a single `LDS.U8` (Load Shared 8-bit Unsigned) instruction. On NVIDIA Pascal hardware (Compute Capability 6.1), shared memory latency is approximately $19\text{ clock cycles}$, with zero ALU bit-shift operations, zero masking overhead, and zero register pressure from intermediate coordinate calculations.

### 2.4 Occupancy & Capacity Math on Pascal GP107 (GeForce GTX 1050)

The target NVIDIA GeForce GTX 1050 (GP107-300, Compute Capability 6.1) exhibits the following physical architectural limits:

- **Shared Memory per SM / Block**: $48\text{ KB} = 49{,}152\text{ bytes}$.
- **Warps per SM**: Maximum 64 resident warps ($2{,}048\text{ threads}$).
- **Thread Block Sizing**: $256\text{ threads per block} = 8\text{ warps per block}$.

Under our 1-warp-per-route execution model (`route_idx = global_tid >> 5`), each thread block processes $W = 8$ routes concurrently. The total shared memory required per thread block is:

$$\text{Shared Memory Required} = W \times n \times 1\text{ byte} = 8n\text{ bytes}$$

Setting this demand against the $48\text{ KB}$ hardware limit establishes the exact mathematical boundary for single-block Format A execution:

$$8n \le 49{,}152\text{ bytes} \implies n \le \frac{49{,}152}{8} = 6{,}144\text{ cities}$$

Thus, any TSP instance up to $n = 6{,}144$ cities fits entirely within shared memory on Pascal hardware without dropping block concurrency below 8 routes per block.

### 2.5 Concrete Shared-Memory Footprints Across the TSPLIB95 Cohort

The official TSPLIB95 library (Reinelt, 1991) contains 113 symmetric TSP instances spanning problem sizes from $n = 14$ (`burma14`) to $n = 2{,}103$ (`d2103`). The table below presents the exact shared-memory allocation and headroom for Format A across representative instances under 8 concurrent routes per block:

| TSPLIB Instance | Cities ($n$) | Format A Footprint ($8n$ bytes) | Pascal GP107 Headroom (48 KB Max) | Utilization |
| :--- | :---: | :---: | :---: | :---: |
| `berlin52` | 52 | $416\text{ bytes}$ | **99.1% free** ($48{,}736\text{ bytes}$ free) | $0.85\%$ |
| `kroA100` | 100 | $800\text{ bytes}$ | **98.3% free** ($48{,}352\text{ bytes}$ free) | $1.63\%$ |
| `lin318` | 318 | $2{,}544\text{ bytes}$ | **94.8% free** ($46{,}608\text{ bytes}$ free) | $5.18\%$ |
| `pr1002` | 1,002 | $8{,}016\text{ bytes}$ | **83.6% free** ($41{,}136\text{ bytes}$ free) | $16.31\%$ |
| `d2103` | 2,103 | $16{,}824\text{ bytes}$ | **65.7% free** ($32{,}328\text{ bytes}$ free) | $34.23\%$ |

Across the entire TSPLIB95 benchmark suite, Format A consumes at most $16.8\text{ KB}$ of shared memory, leaving greater than $65\%$ of SRAM completely unallocated. Bank conflicts are non-existent, and threads achieve maximum memory bandwidth without serialization.

---

## 3. Format B: Bitpacked Array

### 3.1 Structural Representation & Footprint Reduction

Format B compresses the membership representation by mapping each city to a single bit within an array of 32-bit unsigned words (`uint32_t`):

$$\text{Words Required} = \left\lceil \frac{n}{32} \right\rceil$$

$$\text{SRAM Footprint per Route} = \left\lceil \frac{n}{32} \right\rceil \times 4\text{ bytes} \approx \frac{n}{8}\text{ bytes}$$

Format B delivers an $8\times$ reduction in shared-memory consumption relative to Format A. Each 32-bit word packs flags for 32 consecutive cities:

```text
Physical 32-bit Shared Memory Word (uint32_t):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ b31 │ b30 │ b29 │ ... │ b5  │ b4  │ b3  │ b2  │ b1  │ b0  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
City 31                                               City 0
```

### 3.2 Bitwise Coordinate Arithmetic

Accessing a city flag requires converting the scalar city index into two coordinates:

1. **Word Index**: Calculated by dividing the city index by 32 using a right bit-shift:
   $$\text{word\_idx} = \text{city} \gg 5 = \left\lfloor \frac{\text{city}}{32} \right\rfloor$$
2. **Bit Offset**: Calculated modulo 32 using bitwise AND:
   $$\text{bit\_offset} = \text{city} \ \& \ 31 = \text{city} \pmod{32}$$
3. **Bitmask**: Generated via single-bit left-shift:
   $$\text{mask} = 1\text{u} \ll \text{bit\_offset}$$

For example, for City 75:

- $\text{word\_idx} = 75 \gg 5 = 2$ (lives in `bitmap[2]`).
- $\text{bit\_offset} = 75 \ \& \ 31 = 11$ (occupies Bit 11).
- $\text{mask} = 1\text{u} \ll 11 = 0\text{x}00000800$.

### 3.3 The Write Hazard: Race Conditions & Atomic Serialization

While Format B reduces memory footprint by $87.5\%$, it introduces a critical concurrency hazard during the marking phase:

- Suppose Thread 0 marks City 70 ($\text{word\_idx} = 2$, $\text{bit\_offset} = 6$).
- Simultaneously, Thread 1 marks City 75 ($\text{word\_idx} = 2$, $\text{bit\_offset} = 11$).

Both threads must update `bitmap[2]`. If implemented via standard read-modify-write (`bitmap[2] |= mask`), a **Write-After-Write (WAW) race condition** occurs: both threads read the original word concurrently, apply their respective masks in registers, and store the modified word back, causing one thread's update to overwrite and erase the other's!

To ensure correctness, threads must use atomic bitwise operations:

```cpp
atomicOr(&bitmap[city >> 5], 1u << (city & 31));
```

Under NVIDIA CUDA architecture (NVIDIA Corporation, 2024):

- Shared-memory atomics (`ATOMS.OR`) targeting the same word within a warp are **hardware-serialized**.
- Multiple threads contending for the same 32-bit address stall the instruction pipeline, increasing the marking phase latency by up to $32\times$ in the worst-case scenario where all warp threads target the same word.

### 3.4 Query Phase Overhead

During candidate filtering from Parent 2, verifying membership requires reading the 32-bit word, right-shifting by the bit offset, and masking the least significant bit:

```cpp
bool is_duplicate = (bitmap[city >> 5] >> (city & 31)) & 1;
```

This sequence replaces a single `LDS.U8` instruction with:

1. `SHR` / `LEA` for word index calculation.
2. `LDS.U32` for 32-bit shared memory load.
3. `AND` for bit offset calculation.
4. `SHR` (right bit-shift).
5. `LOP3.LUT` or `AND` (extract least significant bit).

This additional ALU instruction latency runs across every single inspected city candidate.

### 3.5 Scalability Ceiling

The primary justification for Format B is its extended scalability limit. Under 8 warps per block ($W = 8$), the maximum problem instance fitting in $48\text{ KB}$ shared memory expands to:

$$8 \times \left(\frac{n}{8}\right) \le 49{,}152 \implies n \le 49{,}152\text{ cities}$$

For an architecture using 1 warp per block ($W = 1$), the ceiling extends to:

$$1 \times \left(\frac{n}{8}\right) \le 49{,}152 \implies n \le 393{,}216\text{ cities}$$

---

## 4. Architectural Decision Matrix & Comparative Analysis

### 4.1 Comprehensive Architectural Comparison

The following table contrasts Format A and Format B across all hardware, algorithmic, and performance dimensions:

| Architectural Metric | Format A: Direct Byte Array | Format B: Bitpacked Bitmap | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Data Type** | `uint8_t in_seg[n]` | `uint32_t bitmap[⌈n/32⌉]` | Format B uses 32-bit words |
| **Footprint per Route** | $n\text{ bytes}$ | $\approx n/8\text{ bytes}$ | Format B is $8\times$ smaller |
| **Marking Instruction** | Standard store (`STS.U8`) | Atomic OR (`ATOMS.OR`) | Format A requires zero atomics |
| **Write Collision Risk** | **Zero** (unique city indices) | **High** (cities share words) | Format B requires serialization |
| **Marking Contention** | None (parallel writes) | Serialized on shared word | Format A avoids warp stalls |
| **Query Instruction** | Direct load (`LDS.U8`) | Load + shift + mask (`LDS.U32`, `SHR`, `AND`) | Format A saves 3–4 ALU instructions |
| **Query Latency** | $\sim 19\text{ cycles}$ | $\sim 19\text{ cycles} + \text{ALU shifts}$ | Format A minimizes instruction count |
| **SRAM per Block ($W=8$)** | $8n\text{ bytes}$ | $n\text{ bytes}$ | Format A fits up to $n = 6{,}144$ |
| **Pascal GP107 Limit ($W=8$)** | $n \le 6{,}144\text{ cities}$ | $n \le 49{,}152\text{ cities}$ | Both exceed TSPLIB95 max ($2{,}103$) |
| **TSPLIB95 Headroom** | $> 65.7\%\text{ free}$ ($n=2{,}103$) | $> 95.7\%\text{ free}$ ($n=2{,}103$) | Format A has abundant headroom |
| **Bank Conflict Profile** | 0 conflicts (byte stride) | 0 conflicts (broadcast reads) | Both formats avoid bank stalls |
| **Code Simplicity** | Clean, direct indexing | Requires bit-shift logic | Format A is simpler and less error-prone |

### 4.2 Production Rationalization for TSPLIB95

For the complete TSPLIB95 suite ($n \le 2{,}103$), **Format A is decisively optimal** for the following reasons:

1. **Zero Atomic Contention**: Every thread writes to an independent byte address, eliminating the `atomicOr` pipeline stalls that plague Format B.
2. **Zero ALU Shift Latency**: Querying duplicates requires a single 8-bit memory load, bypassing bit shifts and mask evaluations.
3. **Abundant Hardware Headroom**: Even for the largest TSPLIB95 instance (`d2103`, $n = 2{,}103$), Format A requires only $16.8\text{ KB}$ across 8 active warps, leaving $65.7\%$ of the $48\text{ KB}$ shared memory untouched.

### 4.3 Scalability Roadmap Integration

Format B is preserved in the project roadmap as a future-proofing scalability extension:

- **Instances $n \le 6{,}144$**: Format A is the default production implementation in `src/algorithms/strategies/crossover/kernels/order_crossover.cu`.
- **Instances $6{,}144 < n \le 49{,}152$**: Format B allows maintaining 8 warps per block on GP107 hardware without spilling shared memory to L2 cache or DRAM.
- **Instances $n > 49{,}152$**: Format B combined with dynamic block downscaling ($W = 1$ to $4$ warps per block) extends GPU Order Crossover up to $393{,}216$ cities.

---

## 5. Repository Implementation Anchors

The production CUDA implementation of Format A resides in:

- Kernel implementation: [`order_crossover.cu`](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L25-L124)
- python strategy driver: [`order_crossover.py`](../../src/algorithms/strategies/crossover/order_crossover.py)

Key implementation excerpts demonstrating Format A in production:

```cpp
// Excerpt from src/algorithms/strategies/crossover/kernels/order_crossover.cu
extern __shared__ unsigned char s_in_seg[];

int warp_in_block = threadIdx.x >> 5;
unsigned char* my_in_seg = &s_in_seg[warp_in_block * n];

// Step 1: Parallel zero-initialization of warp membership table
for (int c = lane_id; c < n; c += 32) {
    my_in_seg[c] = 0;
}
__syncwarp();

// Step 2: Parallel copy and mark from Parent 1 (Zero atomics, unique addresses)
for (int k = lane_id; k < seg_len; k += 32) {
    int city = parent1[c1 + k];
    child[c1 + k] = city;
    if (city >= 0 && city < n) {
        my_in_seg[city] = 1;
    }
}
__syncwarp();

// Step 3: Stream from Parent 2 and query membership with direct LDS.U8 load
for (int step = 0; step < stream_len; step += 32) {
    int idx = step + lane_id;
    int retain = 0;
    int city = -1;
    if (idx < stream_len) {
        int p2_pos = stream_start + ((c2_rel + 1 + idx) % stream_len);
        city = parent2[p2_pos];
        if (city >= 0 && city < n) {
            if (my_in_seg[city] == 0) {
                retain = 1; // Not in segment: retain city in child
            }
        }
    }
    // Warp-level compaction via ballot and popcount
    unsigned int mask = __ballot_sync(0xFFFFFFFFu, retain);
    if (retain) {
        unsigned int lower_mask = (1u << lane_id) - 1u;
        int rank = __popc(mask & lower_mask);
        int write_idx = stream_start + ((child_pos + dest_offset + rank) % stream_len);
        child[write_idx] = city;
    }
    dest_offset += __popc(mask);
}
```

---

## 6. References

1. **Davis, L.** (1985). *Applying adaptive algorithms to epistatic domains*. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI-85), pp. 162–164.
2. **NVIDIA Corporation** (2024). *CUDA C++ Best Practices Guide*. Version 13.0. URL: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/> (`nvidia2024cuda`).
3. **Oliver, I. M., Smith, D. J., & Holland, J. R. C.** (1987). *A study of permutation crossover operators on the traveling salesman problem*. In Genetic Algorithms and their Applications: Proceedings of the Second International Conference on Genetic Algorithms, pp. 224–230.
4. **O'Neil, M. A., & Burtscher, M.** (2014). *Floating-point data compression at 22 GB/s on a GPU*. Proceedings of the 23rd International Conference on Parallel Architectures and Compilation Techniques (PACT), pp. 155–166.
5. **Reinelt, G.** (1991). *TSPLIB—A Traveling Salesman Problem Library*. ORSA Journal on Computing, 3(4), 376–384 (`reinelt1991tsplib`).
6. **Rocki, K., & Suda, R.** (2012). *High-performance GPU implementation of genetic algorithms for the Traveling Salesman Problem*. International Conference on High Performance Computing and Simulation (HPCS), IEEE, pp. 488–495.
