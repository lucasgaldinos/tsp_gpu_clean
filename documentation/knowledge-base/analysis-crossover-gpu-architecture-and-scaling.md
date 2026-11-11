---
title: "Order Crossover (OX) Algorithmic Scaling, GPU Memory Hierarchies, and Warp-Level Parallelism"
description: >
  Rigorous architectural and algorithmic analysis of Order Crossover (OX) for the TSP:
  mathematical formulation, Python CPU hash-table lookup O(n), the O(n^2) scaling bottleneck of
  naive GPU linear scanning (BUG-22), CUDA shared-memory direct-address bitmaps vs. hash sets,
  Pascal GP107 hardware occupancy limits (N=256 threads across 5 SMs), cooperative warp-level
  parallelism (1 warp per route), warp-synchronous prefix scans for iso-algorithmic
  bit-identical offspring generation (DP11-11, Task 11.10), and resolution of the three CPU-vs-GPU
  algorithmic divergences (OX-A).
created: 2026-11-08
date_created: 2026-11-08
date_changed: 2026-11-10
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-08
    modifications:
      - description: >
          Initial creation of deep-dive architectural document addressing the crossover scaling
          bottleneck (BUG-22), deriving CPU O(n) hash-table vs. naive GPU O(n^2) linear scan mechanics,
          evaluating CUDA direct-address shared-memory bitmaps, analyzing SM occupancy on GTX 1050 (GP107),
          and establishing the 1-warp-per-route cooperative architecture with warp shuffle intrinsics.
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Significantly expanded and refined: clarified cyclic fill modulo arithmetic and circular clock indexing;
          detailed CPython set internals (open addressing, perturb >>= 5 quadratic shift, and power-of-two bitmasking);
          added concrete 8-city traces for sequential O(n^2) GPU scans vs. cooperative 8-lane warp bit transformations;
          decoded byte array vs. bitpacked array mechanics (c >> 5 and c & 31); proved why 2-opt tiling (T=2048) does
          not apply to crossover; and documented the three critical CPU-vs-GPU algorithmic divergences (OX-A: disjoint
          vs. ring pairing, replace=False cuts, and depot preservation).
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Documented empirical host-side RNG bottleneck in order_crossover.py where unvectorized
          cut sampling takes ~1.22 ms vs 0.012–0.088 ms CUDA execution (>90% runtime); detailed
          NumPy vectorized integer sampling (c1, c2 += c2 >= c1) reducing overhead to ~0.017 ms (72x speedup);
          and analyzed GPU warm-up nuances (pure AOT compilation vs execution warm-up) and cache poisoning
          hazards from dummy zero arrays on topology detection heuristics (_is_closed_cache).
related_files:
  - "[order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[crossover_scaling_sweep.py](../../benchmark_scripts/crossover_scaling_sweep.py)"
  - "[analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md)"
  - "[guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[guide-cuda-hardware-bitwise-and-registers.md](./guide-cuda-hardware-bitwise-and-registers.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md)"
  - "[analysis-initial-population-policy-and-experiment-comparability.md](./analysis-initial-population-policy-and-experiment-comparability.md)"
  - "[guide-iso-algorithmic-ga-validation.md](./guide-iso-algorithmic-ga-validation.md)"
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
tags:
  - algorithm/crossover
  - algorithm/genetic-algorithm
  - benchmark/gpu
  - guide/cuda
  - notes/cuda
  - notes/hardware
  - analysis/scaling
  - analysis/performance
  - analysis/architecture
---

# Order Crossover (OX) Algorithmic Scaling, GPU Memory Hierarchies, and Warp-Level Parallelism

- [Order Crossover (OX) Algorithmic Scaling, GPU Memory Hierarchies, and Warp-Level Parallelism](#order-crossover-ox-algorithmic-scaling-gpu-memory-hierarchies-and-warp-level-parallelism)
  + [1. Executive Summary and the Empirical Scaling Discrepancy](#1-executive-summary-and-the-empirical-scaling-discrepancy)
    - [1.1 Measured Microbenchmark Data ($N = 256$)](#11-measured-microbenchmark-data-n--256)
    - [1.2 The Power-Law Divergence: $\mathcal{O}(n^{0.87})$ vs. $\mathcal{O}(n^{1.88})$](#12-the-power-law-divergence-mathcalon087-vs-mathcalon188)
    - [1.3 Macroscopic Impact on Genetic Algorithm Cohorts](#13-macroscopic-impact-on-genetic-algorithm-cohorts)
  + [2. Mathematical Formulation of Order Crossover (OX1)](#2-mathematical-formulation-of-order-crossover-ox1)
    - [2.1 Permutation Algebra and Cycle Preservation](#21-permutation-algebra-and-cycle-preservation)
    - [2.2 Step-by-Step Operator Specification](#22-step-by-step-operator-specification)
  + [3. Algorithmic Dissection: Why the CPU Implementation Scales as $\mathcal{O}(n)$](#3-algorithmic-dissection-why-the-cpu-implementation-scales-as-mathcalon)
    - [3.1 Source Inspection of `CrossoverOrder._ox_core` and the Cyclic Clock Fill](#31-source-inspection-of-crossoverorder_ox_core-and-the-cyclic-clock-fill)
    - [3.2 CPython `set` Internals: Open Addressing, `perturb >>= 5`, and Modulo vs. Bitmasking](#32-cpython-set-internals-open-addressing-perturb--5-and-modulo-vs-bitmasking)
    - [3.3 Exact Asymptotic Operation Count on CPU](#33-exact-asymptotic-operation-count-on-cpu)
  + [4. Algorithmic Dissection: Why the Current GPU Kernel Scales as $\mathcal{O}(n^2)$ (`BUG-22`)](#4-algorithmic-dissection-why-the-current-gpu-kernel-scales-as-mathcalon2-bug-22)
    - [4.1 Source Inspection of `order_crossover_kernel`](#41-source-inspection-of-order_crossover_kernel)
    - [4.2 The Nested Linear Scan Bottleneck](#42-the-nested-linear-scan-bottleneck)
    - [4.3 Arithmetic Derivation of Inner Loop Executions](#43-arithmetic-derivation-of-inner-loop-executions)
    - [4.4 Concrete Walkthrough: $\mathcal{O}(n^2)$ Comparisons on an 8-City Tour ($P_1, P_2$)](#44-concrete-walkthrough-mathcalon2-comparisons-on-an-8-city-tour-p_1-p_2)
  + [5. Hardware Architecture: SM Starvation and Thread-Level Parallelism Bottlenecks](#5-hardware-architecture-sm-starvation-and-thread-level-parallelism-bottlenecks)
    - [5.1 NVIDIA Pascal GP107 Hardware Model (GTX 1050)](#51-nvidia-pascal-gp107-hardware-model-gtx-1050)
    - [5.2 Analysis of Grid Launch Configuration: Only 2 of 5 SMs Active](#52-analysis-of-grid-launch-configuration-only-2-of-5-sms-active)
    - [5.3 Little's Law and Scheduler Starvation at 6.25% SM Occupancy](#53-littles-law-and-scheduler-starvation-at-625-sm-occupancy)
  + [6. Why CUDA Cannot Use python's `set` — and Why Direct-Address Arrays are Superior](#6-why-cuda-cannot-use-pythons-set--and-why-direct-address-arrays-are-superior)
    - [6.1 The Pitfalls of Dynamic Hash Tables on SIMT Hardware](#61-the-pitfalls-of-dynamic-hash-tables-on-simt-hardware)
    - [6.2 The TSP Domain Guarantee: Dense Contiguous Integer Keys](#62-the-tsp-domain-guarantee-dense-contiguous-integer-keys)
    - [6.3 Direct-Address Lookup: Byte Array vs. Bitpacked Array Mechanics](#63-direct-address-lookup-byte-array-vs-bitpacked-array-mechanics)
    - [6.4 Shared Memory Capacity Verification across TSPLIB Instances](#64-shared-memory-capacity-verification-across-tsplib-instances)
    - [6.5 Why 2-Opt Tiling ($T=2048$) Does Not Apply to Order Crossover](#65-why-2-opt-tiling-t2048-does-not-apply-to-order-crossover)
  + [7. The Redesigned Architecture: 1 Warp per Route and Warp-Level Cooperativity](#7-the-redesigned-architecture-1-warp-per-route-and-warp-level-cooperativity)
    - [7.1 Decomposing Route Construction across 32 Threads](#71-decomposing-route-construction-across-32-threads)
    - [7.2 Grid Sizing and SM Occupancy: Achieving 75%–100% Saturation](#72-grid-sizing-and-sm-occupancy-achieving-75100-saturation)
    - [7.3 Intra-Warp Cooperative Primitives: `__ballot_sync` and `__popc`](#73-intra-warp-cooperative-primitives-__ballot_sync-and-__popc)
    - [7.4 Algorithmic Walkthrough of the Warp-Parallel OX Kernel](#74-algorithmic-walkthrough-of-the-warp-parallel-ox-kernel)
    - [7.5 Warp-Parallel Trace on $P_1 = [C, D, H, B, G, A, F, E]$ and $P_2 = [A, B, C, D, E, F, G, H]$](#75-warp-parallel-trace-on-p_1--c-d-h-b-g-a-f-e-and-p_2--a-b-c-d-e-f-g-h)
  + [8. Strict Iso-Algorithmic Bit-Identity Contract (`DP11-11`) and CPU vs. GPU Divergences (`OX-A`)](#8-strict-iso-algorithmic-bit-identity-contract-dp11-11-and-cpu-vs-gpu-divergences-ox-a)
    - [8.1 The Bit-Identity Invariant](#81-the-bit-identity-invariant)
    - [8.2 Architectural Enablers of Bit-Identity](#82-architectural-enablers-of-bit-identity)
    - [8.3 The Three Critical CPU-vs-GPU Algorithmic Divergences (`OX-A`)](#83-the-three-critical-cpu-vs-gpu-algorithmic-divergences-ox-a)
    - [8.4 Host-Side RNG Bottleneck: Serial Sampling vs. Vectorized NumPy Generation](#84-host-side-rng-bottleneck-serial-sampling-vs-vectorized-numpy-generation)
    - [8.5 GPU Warm-Up Nuances: AOT Precompilation, Execution Priming, and Cache-Poisoning Traps](#85-gpu-warm-up-nuances-aot-precompilation-execution-priming-and-cache-poisoning-traps)
  + [9. Comparison Summary and Scaling Projection](#9-comparison-summary-and-scaling-projection)
  + [10. References](#10-references)

---

## 1. Executive Summary and the Empirical Scaling Discrepancy

During the systematic validation of the modular Genetic Algorithm (GA) architecture (`Design B`, Phase 11), a critical performance divergence was uncovered between the CPU implementation (`CrossoverOrder`) and the GPU CUDA implementation (`CrossoverGPUOrder`). While the GPU kernel was intended to massively accelerate crossover across population batches, microbenchmarks revealed that its speedup collapsed as the problem dimension $n$ grew, eventually causing the GPU to become significantly slower than a single CPU core.

### 1.1 Measured Microbenchmark Data ($N = 256$)

Measurements were conducted on an NVIDIA GeForce GTX 1050 Mobile (Pascal GP107, 5 SMs, 640 CUDA cores) paired with an Intel Core i7 host CPU. A batch of $N = 256$ offspring routes was generated under fixed random seeds after a dedicated warm-up run (mediana of 3 runs):

| Dimension ($n$) | Problem Class / Anchor | CPU Time (`ms`) | GPU Time (`ms`) | Speedup ($\frac{\text{CPU}}{\text{GPU}}$) | Dominant Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| $n = 52$ | `berlin52` | $8.79$ | $0.78$ | **$11.27\times$** | GPU dominates (high constant-factor parallelism) |
| $n = 100$ | `kroA100` | $14.20$ | $2.15$ | **$6.60\times$** | GPU wins comfortably |
| $n = 200$ | synthetic uniform | $25.10$ | $7.80$ | **$3.22\times$** | GPU advantage narrows |
| $n = 318$ | `lin318` | $38.40$ | $19.20$ | **$2.00\times$** | GPU advantage decaying rapidly |
| $n \approx 800$ | crossover break-even | $\approx 92.00$ | $\approx 92.00$ | **$1.00\times$** | **Empirical Break-Even Point** |
| $n = 1{,}000$ | `pr1002` | $112.50$ | $168.40$ | **$0.67\times$** | GPU is $1.5\times$ slower than CPU |
| $n = 2{,}000$ | `d2103` / large TSPLIB | $206.81$ | $742.05$ | **$0.28\times$** | **GPU is $3.59\times$ slower than CPU** |

### 1.2 The Power-Law Divergence: $\mathcal{O}(n^{0.87})$ vs. $\mathcal{O}(n^{1.88})$

A least-squares log-log power-law regression ($t(n) = \alpha \cdot n^{\beta}$) fitted across the empirical timing data yielded:

$$\beta_{\text{CPU}} \approx 0.87 \quad (\text{sub-linear to linear scaling, consistent with } \mathcal{O}(n))$$

$$\beta_{\text{GPU}} \approx 1.88 \quad (\text{quadratic scaling, consistent with } \mathcal{O}(n^2))$$

This quantitative divergence proves conclusively that the performance inversion at $n \ge 800$ is **not** a hardware bandwidth bottleneck or launch latency artifact. It is an **asymptotic algorithmic defect** (`BUG-22`): the GPU implementation implements an algorithm with a higher computational complexity class than the CPU reference implementation.

Fitting the empirical data to a power law $T(n) = \alpha \cdot n^\beta$:

- **CPU Scaling**: $T_{\text{CPU}}(n) \propto n^{0.87}$ (sub-linear to linear, dominated by memory allocator efficiency).
- **GPU Scaling**: $T_{\text{GPU}}(n) \propto n^{1.88}$ (strictly quadratic $\approx \mathcal{O}(n^2)$).

This confirms an **asymptotic computational complexity defect**, not a constant memory transfer overhead.

### 1.3 Macroscopic Impact on Genetic Algorithm Cohorts

n end-to-end GA runs, the crossover operator is invoked once per generation. In profiling the phase composition of complete runs:

- In `HybridOptimized` (GPU 2-opt local search + host operators), crossover accounts for **$\approx 70\%$** of the entire generation wall-clock time across `berlin52`, `kroA100`, and `lin318`.
- In `FullGPUModular` on `pr1002` ($n = 1{,}002$), crossover accounts for **$63.6\%$** of the entire generation runtime.

In the modular `FullGPUModular` solver pipeline, each generation executes: Selection $\to$ Crossover $\to$ Mutation $\to$ Local Search $\to$ Evaluation.
Because crossover accounts for over $63\%$ of per-generation runtime on $n \ge 1{,}000$, `BUG-22` acts as a severe brake on the entire solver, obscuring any benefits of GPU acceleration.

---

## 2. Mathematical Formulation of Order Crossover (OX1)

Order Crossover, introduced by Davis (1985) and formalized by Oliver, Smith, and Holland (1987), is a specialized genetic permutation recombination operator designed specifically for combinatorial ordering problems such as the Traveling Salesperson Problem (TSP).

### 2.1 Permutation Algebra and Cycle Preservation

Order Crossover (OX1, Davis, 1985; Oliver et al., 1987) is designed for permutation representations of the TSP where each tour is a bijection $\pi: \{0, \dots, n-1\} \to \{0, \dots, n-1\}$. OX preserves the relative order of vertices from both parents while guaranteeing that no duplicate cities appear in the offspring.

### 2.2 Step-by-Step Operator Specification

Given two parent permutations $P_1$ and $P_2$:

1. **Draw Cut Points**: Sample two random indices $0 \le c_1 < c_2 < n$.
2. **Preserve Segment**: Copy the contiguous substring $P_1[c_1 \dots c_2]$ directly into the identical indices of the child:
   $$\text{child}[k] = P_1[k] \quad \forall k \in [c_1, c_2]$$
3. **Cyclic Fill**: Starting immediately after the cut segment at index $(c_2 + 1) \bmod n$, traverse parent $P_2$ in cyclic order. For each city in $P_2$, if that city is **not** present in the preserved segment $P_1[c_1 \dots c_2]$, insert it into the next available slot in `child`, wrapping around to index $0$ until all $n$ cities are placed.

---

## 3. Algorithmic Dissection: Why the CPU Implementation Scales as $\mathcal{O}(n)$

### 3.1 Source Inspection of `CrossoverOrder._ox_core` and the Cyclic Clock Fill

The CPU reference implementation resides in [`src/algorithms/strategies/crossover/order_crossover.py`](../../src/algorithms/strategies/crossover/order_crossover.py#L40-L85):

```python
# src/algorithms/strategies/crossover/order_crossover.py:59-82
def _ox_core(
    p1: np.ndarray,
    p2: np.ndarray,
    c1: int,
    c2: int,
    child: np.ndarray,
) -> None:
    n = len(p1)
    # Direct slice extraction
    segment = p1[c1 : c2 + 1]
    child[c1 : c2 + 1] = segment

    # Set membership structure
    in_segment = set(segment)  # <--- CRUCIAL: O(L) hash table construction

    # Cyclic fill starting from c2 + 1
    current = (c2 + 1) % n
    for i in range(n):
        idx = (c2 + 1 + i) % n
        gene = p2[idx]
        if gene not in in_segment:  # <--- O(1) average lookup
            child[current] = gene
            current = (current + 1) % n
```

#### Why Modulo `(c2 + 1 + i) % n` Wraps Like a Clock

Why does the algorithm use `(c2 + 1 + i) % n`?
Think of a circular 12-hour clock face. If an event ends at 10 o'clock ($c_2 = 10$) and lasts for 4 hours, what hours do you visit?
$11$, $12$, then $13 \bmod 12 = 1$, then $14 \bmod 12 = 2$.
In a computer array of length $n = 8$, indices run from $0$ to $7$. If the cut ends at $c_2 = 5$:

- $i=0 \implies (5 + 1 + 0) \bmod 8 = 6$
- $i=1 \implies (5 + 1 + 1) \bmod 8 = 7$
- $i=2 \implies (5 + 1 + 2) \bmod 8 = 8 \bmod 8 = \mathbf{0}$ (wraps back to the beginning!)
- $i=3 \implies (5 + 1 + 3) \bmod 8 = 9 \bmod 8 = \mathbf{1}$

The modulo operator `% n` makes the flat linear array behave as a seamless **circular ring**, preserving relative tour adjacency across the array boundaries.

### 3.2 CPython `set` Internals: Open Addressing, `perturb >>= 5`, and Modulo vs. Bitmasking

The user asked: *"What does `perturb >>= 5` mean, and why is `mask` used instead of modulo?"*

In CPython (the standard python interpreter written in C, see `Objects/setobject.c` and `dictobject.c`), a `set` is implemented as an open-addressing hash table:

#### 1. The Probing Sequence and `perturb >>= 5`

When two keys hash to the same slot (a collision), the hash table must inspect another slot. If it simply checked the next slot ($i + 1$), long contiguous blocks of occupied slots would form (clustering), degrading lookups.
Instead, CPython uses quadratic perturbation:

$$\text{next\_slot} = (5 \times i + 1 + \text{perturb}) \ \& \ \text{mask}$$

In each collision step, CPython updates:

```c
perturb >>= 5;
```

Syntax `perturb >>= 5` means: **shift the bits of `perturb` to the right by 5 positions and reassign**.
Because the initial value of `perturb` is the full 64-bit hash code, shifting right by 5 bits on every collision gradually injects the **higher-order bits** of the hash code into the probe index. This scrambles the probe sequence across the entire table, preventing clustering!

#### 2. Why `& mask` Replaces `% table_size`

Why does CPython compute `slot = hash & mask` instead of `slot = hash % table_size`?
CPython always allocates hash tables with a capacity that is an exact **power of two**:

$$\text{table\_size} = 2^k \quad (\text{e.g., } 8, 16, 32, 64, \dots)$$

When the divisor is $2^k$, the remainder $x \bmod 2^k$ is mathematically identical to the value of the **lowest $k$ bits** of $x$.
By defining $\text{mask} = 2^k - 1$ (which consists of $k$ ones in binary: $\texttt{0b}\underbrace{11\dots1}_{k\text{ ones}}$):

$$x \bmod 2^k \equiv x \ \& \ (2^k - 1)$$

On modern CPUs:

- Hardware integer division `hash % table_size` requires an ALU division circuit taking **$15\text{–}30\text{ clock cycles}$**.
- Bitwise AND `hash & mask` executes in **exactly 1 clock cycle**!

Because table lookups take 1 cycle, checking `gene not in in_segment` runs in $\mathcal{O}(1)$ average time.

### 3.3 Exact Asymptotic Operation Count on CPU

Let $L = c_2 - c_1 + 1$ denote the preserved slice length ($\mathbb{E}[L] \approx n/3$).
The CPU steps are:

1. Slice copy: $L$ elements $\implies \mathcal{O}(L)$.
2. Set construction: $L$ inserts into hash table $\implies \mathcal{O}(L)$.
3. Cyclic fill loop: $n$ iterations $\times$ $\mathcal{O}(1)$ hash probe $\implies \mathcal{O}(n)$.

Total CPU operational complexity per route:

$$T_{\text{CPU}}(n) = \mathcal{O}(L) + \mathcal{O}(L) + \mathcal{O}(n) = \mathbf{\mathcal{O}(n)}$$

---

## 4. Algorithmic Dissection: Why the Current GPU Kernel Scales as $\mathcal{O}(n^2)$ (`BUG-22`)

### 4.1 Source Inspection of `order_crossover_kernel`

The CUDA kernel resides in [`src/algorithms/strategies/crossover/kernels/order_crossover.cu`](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L10-L60):

```c++ cuda
// src/algorithms/strategies/crossover/kernels/order_crossover.cu:44-59
// Fill remaining positions from parent2
int p2_idx = (c2 + 1) % n;
int child_idx = (c2 + 1) % n;

for (int i = 0; i < n; i++) {
    int val = p2[p2_idx];

    // Linear search in parent1 segment: O(L) scan!
    bool in_segment = false;
    for (int j = c1; j <= c2; j++) {
        if (p1[j] == val) {
            in_segment = true;
            break;
        }
    }

    if (!in_segment) {
        child[child_idx] = val;
        child_idx = (child_idx + 1) % n;
    }
    p2_idx = (p2_idx + 1) % n;
}
```

### 4.2 The Nested Linear Scan Bottleneck

Instead of using an associative lookup table, the kernel executes a brute-force nested linear search `for (int j = c1; j <= c2; j++)`.
For every city in $P_2$, a single thread walks sequentially through the preserved slice in $P_1$, checking if `p1[j] == val`.

### 4.3 Arithmetic Derivation of Inner Loop Executions

With $\mathbb{E}[L] \approx n/3$, the expected number of comparison operations per route is:

$$P(\text{val} \in S) = \frac{L}{n}$$

For cities that **are** in $S$ (which happens $L$ times), the linear scan terminates on average after $\frac{L + 1}{2}$ comparisons.
For cities that **are not** in $S$ (which happens $n - L$ times), the linear scan fails to find a match and must inspect **all $L$ elements** before concluding `in_segment = false`.

Thus, the expected number of comparison operations $C(n, L)$ per route is:

$$C(n, L) = L \cdot \left(\frac{L + 1}{2}\right) + (n - L) \cdot L = n \cdot L - \frac{L^2 - L}{2}$$

Substituting the expected slice length $\mathbb{E}[L] = \frac{n}{3}$:

Evaluating this quadratic function across problem dimensions:

$$\mathbb{E}[C(n)] \approx n \cdot \left(\frac{n}{3}\right) - \frac{1}{2} \left(\frac{n^2}{9}\right) = \frac{n^2}{3} - \frac{n^2}{18} = \frac{5}{18} n^2 \approx \mathbf{0.278 \, n^2}$$

- At $n = 100$: $\approx 2{,}777$ comparisons.
- At $n = 1{,}002$: $\approx 278{,}890$ comparisons.
- At $n = 2{,}000$: $\approx \mathbf{1{,}111{,}111 \text{ comparisons per route!}}$

For a batch of $N = 256$ routes at $n = 2{,}000$, the GPU must execute over **$284 \text{ million comparisons}$** sequentially in memory, explaining the empirical exponent $\beta_{\text{GPU}} \approx 1.88$.

### 4.4 Concrete Walkthrough: $\mathcal{O}(n^2)$ Comparisons on an 8-City Tour ($P_1, P_2$)

Let us trace the exact comparison operations the current GPU kernel performs on an 8-city instance:

- $P_1 = [C, D, H, B, G, A, F, E]$ (numeric: $[2, 3, 7, 1, 6, 0, 5, 4]$)
- $P_2 = [A, B, C, D, E, F, G, H]$ (numeric: $[0, 1, 2, 3, 4, 5, 6, 7]$)
- Cut points: $c_1 = 2, c_2 = 5$.
  + Preserved segment in $P_1$: indices $2 \dots 5 \implies [H, B, G, A] = [7, 1, 6, 0]$.
  + Slice length $L = 4$.

Now the GPU thread begins the outer loop at `p2_idx = (c2 + 1) % n = 6` ($P_2[6] = G = 6$):

1. **Step 1 ($i = 0$, city $G = 6$)**:
   + Compare with $P_1[2] = H$ ($6 \neq 7$) $\to$ step inner loop.
   + Compare with $P_1[3] = B$ ($6 \neq 1$) $\to$ step inner loop.
   + Compare with $P_1[4] = G$ ($6 == 6$) $\implies$ **Match found after 3 comparisons!** (`in_segment = true`, discard $G$).
2. **Step 2 ($i = 1$, city $H = 7$)**:
   + Compare with $P_1[2] = H$ ($7 == 7$) $\implies$ **Match found after 1 comparison!** (`in_segment = true`, discard $H$).
3. **Step 3 ($i = 2$, city $A = 0$, index wrapped to 0)**:
   + Compare with $H$ ($0 \neq 7$).
   + Compare with $B$ ($0 \neq 1$).
   + Compare with $G$ ($0 \neq 6$).
   + Compare with $A$ ($0 == 0$) $\implies$ **Match found after 4 comparisons!** (`in_segment = true`, discard $A$).
4. **Step 4 ($i = 3$, city $B = 1$)**:
   + Compare with $H$ ($1 \neq 7$).
   + Compare with $B$ ($1 == 1$) $\implies$ **Match found after 2 comparisons!** (`in_segment = true`, discard $B$).
5. **Step 5 ($i = 4$, city $C = 2$)**:
   + Compare with $H$ ($2 \neq 7$).
   + Compare with $B$ ($2 \neq 1$).
   + Compare with $G$ ($2 \neq 6$).
   + Compare with $A$ ($2 \neq 0$).
   + End of slice reached! **4 comparisons executed to discover $C$ is missing!** (`in_segment = false`, write $C$ to `child[6]`).
6. **Step 6 ($i = 5$, city $D = 3$)**:
   + Compare with all 4 elements in slice $\implies$ **4 comparisons executed!** (Write $D$ to `child[7]`).
7. **Step 7 ($i = 6$, city $E = 4$)**:
   + Compare with all 4 elements in slice $\implies$ **4 comparisons executed!** (Write $E$ to `child[0]`).
8. **Step 8 ($i = 7$, city $F = 5$)**:
   + Compare with all 4 elements in slice $\implies$ **4 comparisons executed!** (Write $F$ to `child[1]`).

**Total comparisons for just this 1 route**: $3 + 1 + 4 + 2 + 4 + 4 + 4 + 4 = \mathbf{26 \text{ comparisons}}$!
When $n = 2{,}000$, this inner scan does over $1.1 \text{ million comparisons}$ per route!

---

## 5. Hardware Architecture: SM Starvation and Thread-Level Parallelism Bottlenecks

ven if the membership lookup were instantaneous, the current kernel suffers from a second, independent architectural defect: **catastrophic under-utilization of GPU Streaming Multiprocessors (SMs)**.

### 5.1 NVIDIA Pascal GP107 Hardware Model (GTX 1050)

The target GPU (GTX 1050 Mobile, Pascal GP107, Compute Capability 6.1) possesses:

- **5 Streaming Multiprocessors (SMs)**.
- 128 CUDA cores per SM ($5 \times 128 = 640$ cores).
- 4 Warp Schedulers per SM.
- Up to **$2{,}048$ resident threads per SM** ($10{,}240$ threads total).
- 48 KB Shared Memory per SM.

### 5.2 Analysis of Grid Launch Configuration: Only 2 of 5 SMs Active

In [`order_crossover.py#L196-L200`](../../src/algorithms/strategies/crossover/order_crossover.py#L196), the grid launch was configured as:

```python
block_size = 128
grid_size = (num_routes + block_size - 1) // block_size
# For standard population N = 256:
# grid_size = (256 + 128 - 1) // 128 = 2 blocks!
```

This launch produces:

- Grid size: **2 blocks** of 128 threads.
- Total threads launched: $2 \times 128 = \mathbf{256 \text{ threads}}$.

The hardware CUDA Block Scheduler maps thread blocks to SMs. When 2 blocks are launched:

- Total blocks launched: **2 blocks**.
- Block 0 assigned to **SM 0**; Block 1 assigned to **SM 1**.
- **SM 2, SM 3, and SM 4 receive ZERO blocks and sit completely idle (60% of the GPU unallocated)!**

### 5.3 Little's Law and Scheduler Starvation at 6.25% SM Occupancy

Even on the two active SMs (SM 0 and SM 1):
$$\text{SM Occupancy} = \frac{128 \text{ active threads}}{2{,}048 \text{ maximum capacity}} = \mathbf{6.25\%}$$

Each SM has 128 threads, which corresponds to exactly:
$$\text{Active Warps per SM} = \frac{128}{32} = \mathbf{4 \text{ warps}}$$

Because a Pascal SM contains **4 independent warp schedulers**, each scheduler manages on average only **1 warp**.

According to Little's Law applied to SIMT instruction pipelines:

$$\text{Concurrency Required} = \text{Instruction Throughput} \times \text{Latency}$$

A global memory load instruction takes $\approx 200\text{--}400$ clock cycles to complete. To keep the execution pipelines saturated while awaiting memory data, each scheduler requires at least 4 to 8 independent active warps that can be scheduled in round-robin fashion.

Each SM contains 4 independent warp schedulers, meaning each scheduler has on average only **1 active warp** ($128 / 32 = 4\text{ warps}$).
By Little's Law ($\text{Concurrency} = \text{Throughput} \times \text{Latency}$), hiding a 200-cycle global memory stall requires at least 4 to 8 active warps per scheduler.
With only 1 warp per scheduler, the moment that warp loads `p1[j]`, the scheduler has no other warp to issue. The SM stalls completely.

With only 1 warp per scheduler, the moment that single warp initiates a memory load (such as fetching `p1[j]` in the inner loop), the scheduler has **zero** other eligible warps to issue. The SM core pipelines stall completely, entering **scheduler starvation**.

---

## 6. Why CUDA Cannot Use python's `set` — and Why Direct-Address Arrays are Superior

### 6.1 The Pitfalls of Dynamic Hash Tables on SIMT Hardware

Why not instantiate a hash set inside the CUDA kernel?

1. **Dynamic Heap Memory**: Hash tables require dynamic resizing and pointer chaining (`malloc`), which serializes execution across GPU DRAM.
2. **Branch Divergence**: Hash collisions cause different threads in a warp to follow different collision chains, serializing SIMT execution.
3. **Register Pressure**: Managing hash state, table pointers, and collision counters consumes tens of registers per thread, drastically degrading occupancy.
4. **Memory Scattering**: Hash function evaluations scatter memory access patterns randomly across the cache lines, completely defeating memory coalescing.

### 6.2 The TSP Domain Guarantee: Dense Contiguous Integer Keys

Crucially, **a hash table is completely unnecessary for the TSP**.
Hash tables are only required when keys are sparse or unbounded (strings, 64-bit UUIDs).
In the Traveling Salesperson Problem, city identifiers are guaranteed to be **dense, contiguous integers**:

$$\text{city} \in \{0, 1, 2, \dots, n-1\}$$

Because the keys are dense integers, we can use a **Direct-Address Table** (an array indexed directly by the city ID):

$$\text{in\_segment}[\text{city}] \in \{0, 1\}$$

- **Zero hash calculations**: `city` is the index directly.
- **Zero collisions**: every city owns a unique slot.
- **Deterministic $\mathcal{O}(1)$ access**: 1 single memory read.
- **Zero branch divergence**: all threads execute identical instructions.

### 6.3 Direct-Address Lookup: Byte Array vs. Bitpacked Array Mechanics

The user noted:
> *"I would like to understand how the direct-address lookup table works in physical shared memory. Why compare a Byte Array vs. a Bitpacked Array? How does the hardware access it, and what does the mathematical matrix formulation look like for 2 parent routes?"*

To understand this from first principles, consider the fundamental algorithmic question Order Crossover must answer:
> *"Given a candidate city from Parent 2 (say, City $c$), is it already present in the preserved segment of Parent 1?"*

On a sequential CPU, you might check this by searching the segment linearly ($\mathcal{O}(L)$) or using a hash table (`std::unordered_set` or python `set`). But on a GPU:

- Linear search takes hundreds of clock cycles and causes branch divergence.
- Hash tables require computing hash functions, resolving collisions, and allocating dynamic linked lists—disastrous in SIMT architecture!

The GPU-optimal solution is **Direct-Address Table Lookup in On-Chip Shared Memory (`__shared__`)** (latency $\approx 19\text{ clock cycles}$):

```ascii
                     THE CHALKBOARD MENTAL MODEL: DIRECT-ADDRESS LOOKUP
Preserved Segment from P1: [ H, B, G, A ] = Cities [ 7, 1, 6, 0 ]

Direct-Address Table in Shared Memory (n = 8 boxes):
Box Index:     [ 0 ]   [ 1 ]   [ 2 ]   [ 3 ]   [ 4 ]   [ 5 ]   [ 6 ]   [ 7 ]
City Mapped:     A       B       C       D       E       F       G       H
Value Stored:    1       1       0       0       0       0       1       1
               ▲       ▲                                       ▲       ▲
               └── Preserved from P1 (Marked 1) ───────────────┴───────┘
```

1. **Zero Hash Computations**: The city index $c$ is directly the memory offset.
2. **Zero Collisions**: Every city owns its own dedicated box.
3. **Deterministic $\mathcal{O}(1)$ Access**: Exactly 1 memory read instruction.
4. **Zero Branch Divergence**: All 32 threads in the warp execute the exact same memory fetch simultaneously.

There are two physical representations for this table in shared memory:

#### Format A: Byte Array (`uint8_t in_segment[n]`) — The Recommended Architecture

In Format A, each city gets **1 full byte** (`uint8_t`) in shared memory:

- **Memory Footprint**: Exactly $n$ bytes per route.
- **Physical Memory Layout**: For our 8-city instance, 8 contiguous bytes in SRAM:
  $$\texttt{in\_segment} = [1, \, 1, \, 0, \, 0, \, 0, \, 0, \, 1, \, 1]$$
- **Parallel Mark Phase**:
  The 32 threads read the segment of Parent 1 in parallel. Thread $k$ reads city $c = P_1[c_1 + k]$ and writes:

  ```cpp
  in_segment[c] = 1;
  ```

  Because every city appears at most once in a TSP route, **every thread writes to a completely distinct byte address**! There are zero collisions and zero race conditions.
- **Query Phase**:
  When inspecting candidate city $c$ from Parent 2:

  ```cpp
  bool is_duplicate = in_segment[c]; // 0 if missing (valid to copy); 1 if duplicate (skip)
  ```

  This is a single 8-bit load instruction (`LDS.U8`) taking $\sim 19$ cycles.

#### Format B: Bitpacked Array (`uint32_t bitmap[n / 32]`) — The Scalability Extension

Why would anyone consider an alternative to the simplicity of Format A?
Because on NVIDIA Pascal GP107, shared memory is capped at **48 KB per thread block**.
If a problem instance had $n = 100{,}000$ cities, a byte array would require $100\text{ KB}$ per route, exceeding hardware capacity!

In Format B, instead of allocating an entire 8-bit byte for each city, each city gets **1 single bit**:

- **Memory Footprint**: $\lceil n / 32 \rceil \times 4\text{ bytes}$ ($8\times$ smaller than Format A).
- **Physical Layout**: Each 32-bit integer (`uint32_t`) packs the flags for **32 consecutive cities**:

  ```ascii
  Physical 32-bit Word in Shared Memory:
  ┌────┬────┬────┬────┬────┬────┬───┬────┬────┬────┬────┬────┬────┬────┐
  │b31 │b30 │b29 │b28 │b27 │b26 │...│ b5 │ b4 │ b3 │ b2 │ b1 │ b0 │
  └────┴────┴────┴────┴────┴────┴───┴────┴────┴────┴────┴────┴────┴────┘
  City 31                                                  City 0
  ```

- **Access Arithmetic Step-by-Step (Concrete Example with City 75)**:
  1. **Word Index (`city >> 5`)**:
     Dividing by 32 in 1 cycle via bit shift ($75 \gg 5 = \lfloor 75 / 32 \rfloor = 2$). City 75 lives inside `bitmap[2]`.
  2. **Bit Position (`city & 31`)**:
     Computing $75 \bmod 32$ in 1 cycle via bitwise AND ($75 \ \& \ 31 = 11$). City 75 occupies **Bit 11** inside `bitmap[2]`.
  3. **Marking the City (`atomicOr`)**:
     Because multiple threads might mark different cities that happen to share the same 32-bit word (e.g. City 70 and City 75 both live in `bitmap[2]`), an ordinary write `bitmap[2] |= mask` would cause a write-after-write race condition! Threads must use atomic bitwise OR:

     ```cpp
     atomicOr(&bitmap[city >> 5], 1u << (city & 31));
     ```

  4. **Testing if the City is Present**:

     ```cpp
     bool is_duplicate = (bitmap[city >> 5] >> (city & 31)) & 1;
     ```

> [!IMPORTANT]
> **Why Format A is decisively superior for all benchmark instances ($n \le 2{,}103$):**
>
> 1. **Zero Atomic Contention**: Format A writes to unique byte addresses, needing zero `atomicOr` instructions.
> 2. **Zero Bit-Shift Overhead**: No `>> 5`, `& 31`, or `1u << k` ALU operations on every single city access.
> 3. **Abundant Shared Memory Headroom**: As proven in the table below, even for $n = 2{,}103$, Format A consumes only $16.8\text{ KB}$ across 8 concurrent routes, leaving $>65\%$ of shared memory completely free!
>
> Format B is formally preserved in our roadmap (`DP12-1`, `Task 12.1`) as a future-proof scalability extension for instances exceeding $n > 10{,}000$ cities.

> [!NOTE]
> **Mathematical Matrix Formalization (The 2-Route Formulation):**
> For the complete linear algebraic derivation proving how this direct-address table corresponds to the incidence vector $\mathbf{v} = \mathbf{1}^{\mathsf{T}}(\mathbf{S}\mathbf{\Pi}_1)$ and why dense general matrix multiplication (GEMM, $\mathcal{O}(n^3)$) is rejected in favor of warp streaming ($\mathcal{O}(n)$) across 2 parent routes, see the companion analysis:
> [analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md).

### 6.4 Shared Memory Capacity Verification across TSPLIB Instances & Hardware Occupancy Math

On Pascal GP107 (GTX 1050), each Streaming Multiprocessor (SM) provides:

- **Max Resident Threads per SM**: $2{,}048\text{ threads} = 64\text{ warps}$.
- **Max Thread Blocks per SM**: $32\text{ blocks}$.
- **Warp Schedulers**: $4\text{ warp schedulers}$ per SM (capable of issuing instructions for at most 4 warps concurrently per clock cycle, switching among up to 64 resident warps to hide memory latency).
- **Shared Memory Limit**: **48 KB** ($49{,}152\text{ bytes}$) per thread block and per SM.

#### Occupancy Derivation for 256 Threads (8 Warps / Block)

In our 1-warp-per-route execution model (`route_idx = global_tid >> 5`), each thread block hosts $W = \text{threads\_per\_block} / 32$ warps, processing $W$ routes concurrently.

- For $\text{threads\_per\_block} = 256$, $W = 8\text{ warps}$ ($8$ routes per block).
- Theoretical maximum SM thread occupancy requires $\frac{64\text{ warps}}{8\text{ warps/block}} = 8\text{ blocks per SM}$, which fits easily within the $32$-block hardware limit.
- Total shared memory required per block with Format A ($n$ bytes per warp) is:
  $$\text{Shared Memory Required} = W \times n \times 1\text{ byte} = 8n\text{ bytes}$$

#### Mathematical Boundary for Format A

Setting $8n \le 49{,}152\text{ bytes}$ yields the exact single-block hardware threshold:
$$n \le \frac{49{,}152}{8} = 6{,}144\text{ cities}$$

#### Shared Memory Footprint across TSPLIB Instances

For a block running 8 routes concurrently:

| TSPLIB Instance | $n$ | Byte Array (8 Routes / Block) | Bitpacked (8 Routes / Block) | Headroom on 48 KB Max |
| :--- | :---: | :---: | :---: | :--- |
| `berlin52` | 52 | $416\text{ bytes}$ | $64\text{ bytes}$ | **99.1% free** |
| `kroA100` | 100 | $800\text{ bytes}$ | $128\text{ bytes}$ | **98.3% free** |
| `lin318` | 318 | $2{,}544\text{ bytes}$ | $320\text{ bytes}$ | **94.8% free** |
| `pr1002` | 1,002 | $8{,}016\text{ bytes}$ | $1{,}024\text{ bytes}$ | **83.6% free** |
| `d2103` | 2,103 | $16{,}824\text{ bytes}$ | $2{,}112\text{ bytes}$ | **65.7% free** |

For all benchmark instances ($n \le 2{,}103$), the simple byte array comfortably fits in shared memory without needing bitpacking or global memory spills.

#### Dynamic Block Downscaling for $n > 6{,}144$

When problem dimension exceeds $n > 6{,}144$, maintaining 8 warps per block ($256$ threads) would require $> 48\text{ KB}$. The launcher in `order_crossover.py` dynamically queries the driver's `MaxSharedMemoryPerBlock` attribute and halves the block size:

- $n \in [6{,}145, 12{,}288]$: downscales to $128\text{ threads}$ ($4\text{ warps} \implies 4n \le 48\text{ KB}$).
- $n \in [12{,}289, 24{,}576]$: downscales to $64\text{ threads}$ ($2\text{ warps} \implies 2n \le 48\text{ KB}$).
- $n \in [24{,}577, 49{,}152]$: downscales to $32\text{ threads}$ ($1\text{ warp} \implies 1n \le 48\text{ KB}$).

Beyond $n = 49{,}152$ (where even a single warp's byte table exceeds $48\text{ KB}$), Format B (bitpacked array `uint32_t bitmap[n/32]`, `DP12-1`, `Task 12.1`) becomes necessary, reducing shared memory by $8\times$ and extending the single-warp ceiling to $n \le 393{,}216$ cities.

### 6.5 Why 2-Opt Tiling ($T=2048$) Does Not Apply to Order Crossover

The user asked: *"Why is the tiling trick ($T=2048$) used in 2-opt not used in Order Crossover?"*

This stems from the fundamental difference in dimensionality between the two algorithms:

1. **In 2-Opt Local Search**:
   2-opt evaluates all pairs of edges $(i, j)$ across the tour. This requires an $n \times n$ distance matrix or coordinate evaluations ($\Theta(n^2)$ pairs).
   When $n > 2{,}048$, the entire $n$-city tour cannot fit inside 48 KB of shared memory ($2048 \times 4\text{ bytes} \times 2 = 16\text{ KB}$ for coordinates, plus edge caches). Therefore, 2-opt must **tile** the tour into blocks of size $T = 2{,}048$ and process tiles in successive passes.

2. **In Order Crossover (OX)**:
   OX operates on **1D linear arrays** of size $n$, never an $n \times n$ matrix.
   + For $n \le 2{,}103$, the entire lookup table is only $2\text{ KB}$, which easily fits in shared memory.
   + The streaming of parent $P_2$ is done in chunks of 32 cities directly from global memory into registers.
   + Because OX never compares all pairs of cities against all other cities, tiling an $n \times n$ space is mathematically meaningless.
   + For a complete formal linear algebraic matrix development comparing $n \times n$ permutation matrix representations, general matrix multiplication (GEMM), and linear warp streaming across 2 parent routes, see the companion analysis: [analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md).

---

## 7. The Redesigned Architecture: 1 Warp per Route and Warp-Level Cooperativity

### 7.1 Decomposing Route Construction across 32 Threads

Instead of assigning 1 thread to sequentially build 1 route, we assign **1 Warp (32 threads)** cooperatively to build 1 offspring route.

```
Execution Architecture Comparison:
Naive Kernel (1 Thread per Route):
Thread 0   ──> Route 0 (does all n cities sequentially)
Thread 1   ──> Route 1
...
Thread 255 ──> Route 255  (Only 2 blocks launched; 3 of 5 SMs idle!)

Redesigned Kernel (1 Warp per Route):
Warp 0 (Threads 0..31)     ──> Cooperates on Route 0
Warp 1 (Threads 32..63)    ──> Cooperates on Route 1
...
Warp 255 (Threads 8160..8191) ──> Cooperates on Route 255  (32 blocks launched; 100% SM saturation!)
```

### 7.2 Grid Sizing and SM Occupancy: Achieving 75%–100% Saturation

For a population batch of $N = 256$ offspring routes:

$$\text{Total Threads} = 256 \text{ routes} \times 32 \frac{\text{threads}}{\text{route}} = \mathbf{8{,}192 \text{ threads}}$$

Configuring 8 warps (256 threads) per block yields:

$$\text{Grid Size} = \frac{8{,}192}{256} = \mathbf{32 \text{ blocks}}$$

Distributing 32 blocks across 5 SMs gives $\approx 6.4$ blocks per SM:

- Active threads per SM: $1{,}536\text{ to }1{,}792$.
- **Active SM Occupancy**: **$75.0\%\text{ to }87.5\%$** (optimal).
- **All 5 SMs saturated (0 idle SMs)**.
- Each warp scheduler manages $12\text{ to }14$ active warps, completely hiding memory latency.

### 7.3 Intra-Warp Cooperative Primitives: `__ballot_sync` and `__popc`

When 32 threads inspect a chunk of 32 cities from parent $P_2$:

1. Each thread $\ell$ checks `is_valid = (in_segment[gene] == 0)`.
2. `ballot_mask = __ballot_sync(0xFFFFFFFFu, is_valid);` packs all 32 booleans into 1 register.
3. `lower_lanes_mask = (1u << lane_id) - 1u;` isolates threads to the left.
4. `rank = __popc(ballot_mask & lower_lanes_mask);` gives each active thread its exact sequential write offset.
5. Active threads write to `child[write_pos + rank]` in a coalesced burst.
6. `write_pos += __popc(ballot_mask);` advances the cursor for the next chunk.

### 7.4 Algorithmic Walkthrough of the Warp-Parallel OX Kernel

```
Step 1: Parallel Table Clear
Lanes 0..31 clear the shared-memory byte array in parallel:
Lane 0 clears 0, 32, 64...  Lane 1 clears 1, 33, 65...
(Takes ceil(n / 32) cycles)

Step 2: Parallel Segment Copy and Marking
Lanes collaboratively read P1[c1..c2], copy to child[c1..c2],
and write in_segment[gene] = 1 in parallel.
(Takes ceil(L / 32) cycles)

Step 3: Cooperative Streaming and Compaction from P2
Lanes stream 32 cities at a time from P2:
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬───────┬──────────────┐
│    Lane 0    │    Lane 1    │    Lane 2    │    Lane 3    │    Lane 4    │  ...  │   Lane 31    │
│  (Thread 0)  │  (Thread 1)  │  (Thread 2)  │  (Thread 3)  │  (Thread 4)  │       │ (Thread 31)  │
│    City A    │    City B    │    City C    │    City D    │    City E    │       │    City H    │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴───────┴──────┬───────┘
       │              │              │              │              │                      │
       │              │  O(1) SRAM Lookup: in_segment[city]        │                      │
       ▼              ▼              ▼              ▼              ▼                      ▼
 [Duplicate]    [Duplicate]     [Missing]      [Missing]      [Missing]              [Duplicate]
 (Skip / 0)     (Skip / 0)     (Keep / 1)     (Keep / 1)     (Keep / 1)              (Skip / 0)
       │              │              │              │              │                      │
       └──────────────┴──────────────┴──────┬───────┴──────────────┴──────────────────────┘
                                            ▼
           __ballot_sync() gathers the 32 votes into 1 compact bitmask: 0b...011100
           __popc() gives each [Missing] lane its exact destination slot in the child!
           Surviving lanes write cities directly to child in 1 single coalesced burst!
```

Complexity per route on GPU:

$$T_{\text{GPU-warp}}(n) = \mathcal{O}\left(\frac{n}{32}\right) = \mathbf{\mathcal{O}(n)}$$

The GPU implementation is restored to the **linear complexity class**, identical to CPU.

### 7.5 Warp-Parallel Trace on $P_1 = [C, D, H, B, G, A, F, E]$ and $P_2 = [A, B, C, D, E, F, G, H]$

Let us trace the warp-parallel execution on the 8-city instance following the exact cyclic Order Crossover specification:

- Preserved slice from $P_1$: $[H, B, G, A]$ at indices $2 \dots 5$.
- `in_segment` marked as `1` in Shared Memory for cities $\{H, B, G, A\} = \{7, 1, 6, 0\}$.
- Cities $\{C, D, E, F\} = \{2, 3, 4, 5\}$ have `in_segment == 0`.
- Vacant positions in `child` starting from cyclic offset $(c_2 + 1) \bmod 8 = \mathbf{6}$:
  $$\text{Slot 6, Slot 7, Slot 0, Slot 1}$$

#### Demystifying "Lanes" and the Abstract Parallel Stream Compaction

The user asked:
> *"What does 'lanes' mean here? The number of threads? And those binary masks are way too specific — what do they even mean? Isn't there something more abstract than writing pure binary code? How should I understand that?"*

Let us understand this from the top down, separating the abstract algorithmic concept from the binary hardware implementation:

1. **What is a "Lane"?**
   A lane is simply **one individual thread within a 32-thread hardware warp**, indexed from $0$ to $31$ (`lane_id = threadIdx.x % 32`).
   Think of a 32-lane highway: each thread drives in its own dedicated lane. Thread 0 is in Lane 0, Thread 1 is in Lane 1, ..., Thread 31 is in Lane 31.
2. **What is the abstract problem? (Parallel Stream Compaction)**
   In python, Order Crossover does:

   ```python
   # Abstract CPU view: filter parent 2 to find cities not yet in the child
   missing_cities = [city for city in parent2 if city not in segment]
   ```

   On a sequential CPU, a single thread runs a loop, appending valid cities to a list one by one.
   On a GPU, 32 threads evaluate 32 cities simultaneously. If 4 of those cities are missing, **how do those 4 threads know which output slot (0, 1, 2, or 3) belongs to them without colliding?**
3. **The High-Level Collaborative Table (No Binary Required!)**:
   Instead of writing binary bitmasks, look at the collaborative decision table:

   | Thread (Lane) | City Evaluated from $P_2$ | Already in $P_1$? | Status | Raised Hand? | Number of Raised Hands to My Left | Destination Slot in Output |
   | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | Lane 0 | $G$ | Yes ($P_1[4]$) | Duplicate | No | 0 | *Does not write* |
   | Lane 1 | $H$ | Yes ($P_1[2]$) | Duplicate | No | 0 | *Does not write* |
   | Lane 2 | $A$ | Yes ($P_1[5]$) | Duplicate | No | 0 | *Does not write* |
   | Lane 3 | $B$ | Yes ($P_1[3]$) | Duplicate | No | 0 | *Does not write* |
   | **Lane 4** | **$C$** | **No** | **Missing** | **YES** | **0** | **1st vacant slot (Slot 6)** |
   | **Lane 5** | **$D$** | **No** | **Missing** | **YES** | **1** | **2nd vacant slot (Slot 7)** |
   | **Lane 6** | **$E$** | **No** | **Missing** | **YES** | **2** | **3rd vacant slot (Slot 0)** |
   | **Lane 7** | **$F$** | **No** | **Missing** | **YES** | **3** | **4th vacant slot (Slot 1)** |

4. **Why did we write binary numbers like `0b11110000`?**
   Because inside the GPU chip, the 8 threads "raise their hands" by sending their 1-bit boolean flag (`0` or `1`) to the SM execution unit. The hardware packs those 8 hands into an 8-bit integer:
   $$\text{Lane 7}\to 1, \quad \text{Lane 6}\to 1, \quad \text{Lane 5}\to 1, \quad \text{Lane 4}\to 1, \quad \text{Lane 3}\to 0, \quad \text{Lane 2}\to 0, \quad \text{Lane 1}\to 0, \quad \text{Lane 0}\to 0$$
   Reading from left to right (Lane 7 down to Lane 0), this is written as `0b11110000`.
   + `__ballot_sync` is just the hardware instruction that collects the raised hands!
   + `__popc` (population count) is just the hardware instruction that counts how many hands were raised!
   There is no "hacky ultra-optimization" here: this is the standard, mathematically proven primitive for **Parallel Prefix Scan and Stream Compaction** used across all parallel computing libraries (CUDA CUB, Thrust, and Kokkos).

#### Execution Trace: Writing to Child Slots in 1 Clock Cycle

- `ballot_mask = __ballot_sync(0xFF, is_valid) = 0b11110000` ($240_{10}$).
- Lane 4: 0 raised hands to its left $\implies \text{rank} = 0$. Writes $C$ to `child[(6+0)%8] = child[6]`.
- Lane 5: 1 raised hand to its left $\implies \text{rank} = 1$. Writes $D$ to `child[(6+1)%8] = child[7]`.
- Lane 6: 2 raised hands to its left $\implies \text{rank} = 2$. Writes $E$ to `child[(6+2)%8] = child[0]`.
- Lane 7: 3 raised hands to its left $\implies \text{rank} = 3$. Writes $F$ to `child[(6+3)%8] = child[1]`.

The final child route is:

$$\text{child} = [E, F, H, B, G, A, C, D]$$

All 4 missing cities are placed into their correct slots in **1 single clock cycle**, with zero nested loops and 100% bit-identity to CPU `CrossoverOrder._ox_core`!

---

## 8. Strict Iso-Algorithmic Bit-Identity Contract (`DP11-11`) and CPU vs. GPU Divergences (`OX-A`)

### 8.1 The Bit-Identity Invariant

Under `DP11-11` and `Task 11.10`:
> For identical parent routes $P_1, P_2$ and identical cut points $(c_1, c_2)$, `CrossoverGPUOrder` must produce an offspring route array that is **bit-for-bit identical** to the output of `CrossoverOrder._ox_core`:
>
> $$\text{child}_{\text{GPU}}[k] \equiv \text{child}_{\text{CPU}}[k] \quad \forall k \in \{0, 1, \dots, n-1\}$$

### 8.2 Architectural Enablers of Bit-Identity

1. **Cut Points Sampled on Host**: As established in Phase 10, cut points $(c_1, c_2)$ are sampled by `ProblemContext.rng` on the host CPU and passed as uniform batch arrays to the GPU kernel.
2. **Deterministic Lane Ordering**: Warp compaction guarantees that active lanes write in strict increasing order of `lane_id`.

### 8.3 The Three Critical CPU-vs-GPU Algorithmic Divergences (`OX-A`)

During our architectural audit, three subtle but major algorithmic discrepancies were uncovered between the CPU reference and GPU crossover implementations. These discrepancies break iso-algorithmic comparability and must be resolved together under `OX-A` (`Task 11.10`):

#### Divergence 1: Parent Pairing Mechanics

- **CPU Reference** ([`order_crossover.py#L130-L141`](../../src/algorithms/strategies/crossover/order_crossover.py#L130)):
  Pairs parents in **disjoint pairs** $(P_{2k}, P_{2k+1})$:
  $$(P_0 \times P_1) \implies \text{Child}_0, \text{Child}_1; \quad (P_2 \times P_3) \implies \text{Child}_2, \text{Child}_3$$
  Both offspring in a pair share the exact same cut points $(c_1, c_2)$, generating symmetric complementary children.
- **Current GPU Kernel** ([`order_crossover.cu#L22`](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L22)):
  Pairs parents in a **cyclic ring**:
  $$(P_i \times P_{(i+1) \bmod N}) \implies \text{Child}_i$$
  Route $i$ crosses with route $i+1$, and each route receives a completely independent random cut point!
- **Resolution in `OX-A`**: Align the GPU launcher to generate disjoint pairs $(2k, 2k+1)$ with identical cut points, matching CPU semantics.

#### Divergence 2: Cut Points Sampling Strategy

- **CPU Reference** ([`order_crossover.py#L106-L113`](../../src/algorithms/strategies/crossover/order_crossover.py#L106)):
  Samples 2 points **without replacement**:

  ```python
  c1, c2 = np.sort(rng.choice(n, size=2, replace=False))
  ```

  This strictly guarantees $c_1 < c_2$ (cut segment length $L \ge 2$).
- **Current GPU Kernel**:
  Draws cut points independently using `rng.integers(0, n, size=N)`, which allows $c_1 = c_2$ (cut segment of length 1).
- **Resolution in `OX-A`**: Draw cut points with `replace=False` on the host CPU so $c_1 < c_2$ is guaranteed on both CPU and GPU.

#### Divergence 3: Closed-Route Depot Preservation

- **CPU Reference** ([`order_crossover.py#L93-L97`](../../src/algorithms/strategies/crossover/order_crossover.py#L93)):
  Detects if the route is a closed tour ($P[0] == P[-1]$) and constrains cut points to $[1, n-2]$, keeping the fixed depot at index $0$ and index $n-1$ untouched.
- **Current GPU Kernel**:
  Completely ignores depots and draws cuts across $[0, n-1]$, overwriting fixed depot positions!
- **Resolution in `OX-A`**: Enforce depot preservation bounds in the GPU kernel launcher.

### 8.4 Host-Side RNG Bottleneck: Serial Sampling vs. Vectorized NumPy Generation

Following the implementation of `order_crossover_kernel` with 1 warp per route (`DP11-11`), high-precision GPU hardware profiling with CUDA events (`cp.cuda.Event`) demonstrated that the pure CUDA kernel executes with extreme efficiency on the GTX 1050 (GP107) for $N = 256$ individuals:

- $n = 52$ (`berlin52`): **$0.0123\text{ ms}$** ($12.3\ \mu\text{s}$)
- $n = 100$ (`kroA100`): **$0.0143\text{ ms}$** ($14.3\ \mu\text{s}$)
- $n = 200$: **$0.0164\text{ ms}$** ($16.4\ \mu\text{s}$)
- $n = 500$: **$0.0246\text{ ms}$** ($24.6\ \mu\text{s}$)
- $n = 1{,}000$ (`pr1002`): **$0.0389\text{ ms}$** ($38.9\ \mu\text{s}$)
- $n = 2{,}000$ (`d2103`): **$0.0881\text{ ms}$** ($88.1\ \mu\text{s}$)

The pure CUDA kernel scales with a fitted power-law exponent of:

$$\beta_{\text{kernel}} \approx 0.539 \ll 1.20$$

However, when profiling the end-to-end strategy method [`crossover_batch`](../../src/algorithms/strategies/crossover/order_crossover.py#L220), total wall-clock runtime measured $\approx 2.45\text{ ms}$ across all problem dimensions. Host-side flame graphs and timing breakdowns revealed an unexpected bottleneck: **over $90\%$ of total strategy runtime was consumed by the host CPU in Python before kernel launch**.

#### 1. The Root Cause: Serial Python Loop and Dynamic Allocation

To satisfy `OX-A` (disjoint pairing $(2k, 2k+1)$ and cut point sampling without replacement), cut points were initially generated on the host CPU using an unvectorized serial Python loop:

```python
cut1_np = np.zeros(num_routes, dtype=np.int32)
cut2_np = np.zeros(num_routes, dtype=np.int32)

for i in range(0, num_routes - 1, 2):
    if is_closed:
        cuts = rng.choice(np.arange(1, n_cities - 1), size=2, replace=False)
    else:
        cuts = rng.choice(n_cities, size=2, replace=False)
    c1, c2 = int(min(cuts)), int(max(cuts))
    cut1_np[i] = c1
    cut2_np[i] = c2
    cut1_np[i + 1] = c1
    cut2_np[i + 1] = c2
```

For a population of $N = 256$ individuals ($128$ parent pairs), this serial construct introduces devastating interpreter overhead:
1. **$128$ Temporary Array Allocations**: `np.arange(1, n_cities - 1)` allocates and frees an intermediate NumPy buffer on every loop iteration.
2. **$128$ Dynamic Method Invocations**: `rng.choice(..., replace=False)` incurs CPython bytecode dispatch, Python tuple packing, dynamic type validation, and partial Fisher-Yates array shuffling.
3. **$256$ Scalar Casts and $512$ Individual Array Writes**: Repeatedly converting NumPy integers to Python `int` and writing scalar values into NumPy arrays across interpreter boundaries.

On an Intel Core i7 host CPU, this serial loop consumes **$1.05\text{ ms}$ to $1.22\text{ ms}$**. Comparing this host preparation duration to pure CUDA kernel execution reveals a severe Amdahl imbalance:

$$\frac{T_{\text{host}}}{T_{\text{CUDA}}(n=52)} = \frac{1.220\text{ ms}}{0.0123\text{ ms}} \approx 99.2\times \quad (\mathbf{\approx 100\times \text{ slower than CUDA}})$$

$$\frac{T_{\text{host}}}{T_{\text{CUDA}}(n=2{,}000)} = \frac{1.220\text{ ms}}{0.0881\text{ ms}} \approx 13.8\times \quad (\mathbf{\approx 14\times \text{ slower than CUDA}})$$

The unvectorized Python cut generation loop takes $14\times$ to $100\times$ longer than the CUDA kernel itself, dominating $>90\%$ of total strategy runtime.

#### 2. Vectorized Integer Sampling in NumPy

To preserve 100% bit-identity with the CPU host PRNG state without offloading RNG to GPU cuRAND (which would break deterministic cross-platform reproducibility), cut point sampling can be vectorized across all $K = N / 2$ pairs simultaneously.

To sample two distinct integers $c_1 \neq c_2$ uniformly at random from $[a, b)$ of length $L = b - a$ without replacement:
1. Draw $c_1 \sim \mathcal{U}\{a, b - 1\}$ uniformly of size $K$.
2. Draw $c_2' \sim \mathcal{U}\{a, b - 2\}$ uniformly of size $K$ from an alphabet reduced by 1.
3. Apply a branchless conditional increment:
   $$c_2 = c_2' + \mathbb{I}(c_2' \ge c_1)$$
   Because $c_2'$ is drawn from an alphabet of $L - 1$ symbols, shifting all values $\ge c_1$ upward by 1 maps $c_2'$ bijectively onto $\{a, \dots, b - 1\} \setminus \{c_1\}$. This guarantees $c_1 \neq c_2$ with exact uniform distribution $\mathcal{P}(c_1 = x, c_2 = y) = \frac{1}{L(L - 1)}$ for all $x \neq y$, requiring **zero rejection or sequential retry**.
4. Compute cut boundaries using vectorized SIMD minimum and maximum:
   $$\text{start} = \min(c_1, c_2), \quad \text{end} = \max(c_1, c_2)$$
5. Replicate across disjoint pairs $(2k, 2k + 1)$ via `np.repeat(..., 2)`.

```python
# Vectorized host cut generation in NumPy (O(1) dynamic Python dispatches)
num_pairs = num_routes // 2
low = 1 if is_closed else 0
high = (n_cities - 1) if is_closed else n_cities

# 1. Vectorized uniform integer draws
c1 = rng.integers(low, high, size=num_pairs, dtype=np.int32)
c2 = rng.integers(low, high - 1, size=num_pairs, dtype=np.int32)

# 2. Branchless collision avoidance without rejection
c2 += (c2 >= c1)

# 3. Vectorized ordering (guarantees start < end)
start = np.minimum(c1, c2)
end = np.maximum(c1, c2)

# 4. Disjoint pair replication: pairs (2k, 2k+1) share identical cuts
cut1_np = np.repeat(start, 2)
cut2_np = np.repeat(end, 2)
```

#### 3. Empirical Performance Impact

| Sampling Implementation | Host Execution Time ($N=256$) | Speedup vs. Serial Loop | Dominant Operation |
| :--- | :---: | :---: | :--- |
| **Serial Python Loop** (`rng.choice`) | **$1.220\text{ ms}$** ($1220\ \mu\text{s}$) | $1.0\times$ (Baseline) | CPython loop overhead, 128 allocations |
| **Vectorized NumPy** (`c2 += c2 >= c1`) | **$0.017\text{ ms}$** ($17\ \mu\text{s}$) | **$71.8\times$ faster** | 2 C-level SIMD RNG draws + vectorized shift |

Vectorized integer sampling reduces host cut generation overhead from $\approx 1.22\text{ ms}$ to $\approx 0.017\text{ ms}$ ($72\times$ faster). This brings host-side preparation time down to parity with the asynchronous CUDA driver kernel launch latency ($\approx 15\text{--}20\ \mu\text{s}$), completely eliminating the host Amdahl bottleneck.

---

### 8.5 GPU Warm-Up Nuances: AOT Precompilation, Execution Priming, and Cache-Poisoning Traps

Microbenchmarking GPU algorithms requires rigorous separation between compilation, initialization, and steady-state execution. During the empirical validation of `CrossoverGPUOrder`, two critical benchmarking traps were identified: the distinction between AOT precompilation versus execution warm-up, and topology cache poisoning caused by unrepresentative dummy inputs.

#### 1. AOT Precompilation vs. Execution Warm-Up

A common testing pitfall is assuming that compiling the CUDA kernel ahead of time (AOT) is sufficient to eliminate first-call measurement noise:

- **Ahead-Of-Time (AOT) Precompilation**: CuPy compiles the CUDA C++ source code string into PTX via NVRTC and translates it to target binary machine code (cubin), loading the module into CUDA driver memory. This compiles the code but **leaves the GPU runtime hardware completely unprimed**.
- **Execution Warm-Up (Runtime Priming)**: Dispatching an untimed kernel launch with active memory buffers. AOT precompilation alone fails to warm the hardware pipeline due to four distinct mechanisms:
  1. **CuPy Memory Pool Allocation**: The initial allocation of device memory buffers (`cp.empty`, `cp.asarray`) triggers expensive OS-level driver calls (`cudaMalloc`). Subsequent allocations recycle pre-allocated chunks from the pool in microseconds.
  2. **CUDA Stream Context Initialization**: The CUDA driver establishes internal stream queues, command buffers, and hardware event synchronization primitives on the first physical kernel dispatch.
  3. **Hardware Clock P-State Transitions**: Consumer and mobile GPUs (e.g. GTX 1050 Mobile) idle at low-power P-states (P8/P5, $\approx 139\text{ MHz}$). Upon receiving a kernel, the GPU hardware power management requires several milliseconds to ramp core clocks to maximum boost frequency (P0, $\approx 1{,}493\text{ MHz}$).
  4. **Instruction Cache (I-Cache) and L2 Priming**: The first kernel launch incurs cold misses in the SM instruction cache and GPU L2 cache, loading kernel instructions and page table entries (TLB) into on-chip SRAM.

In initial runs of `benchmark_scripts/crossover_scaling_sweep.py`, measuring $n=52$ without an untimed per-size warm-up yielded $3.08\text{ ms}$ (due to pool expansion for new buffer dimensions), while $n=2{,}000$ measured $1.72\text{ ms}$ on subsequent runs. This artifactual downward measurement tilt produced a false negative power-law exponent ($\beta \approx -0.12$). Introducing an untimed per-dimension execution warm-up stabilized the benchmark to its true scaling behavior: $\beta_{\text{strategy}} = \mathbf{0.0422} \le 1.20$.

#### 2. The Cache-Poisoning Hazard: Dummy Zero Arrays and Topology Heuristics

While execution warm-up is mandatory, **using dummy zero arrays during warm-up can silently poison stateful strategy heuristics**.

In `src/algorithms/strategies/crossover/order_crossover.py` (lines 230–236), route topology detection (whether a route is a closed tour with identical depot endpoints or an open tour) was optimized using an internal cache:

```python
if self._is_closed_cache is not None and self._is_closed_cache[0] == n_cities:
    is_closed = self._is_closed_cache[1]
else:
    first_0 = int(d_parents[0, 0].get()) if input_on_gpu else int(batch[0, 0])
    first_end = int(d_parents[0, -1].get()) if input_on_gpu else int(batch[0, -1])
    is_closed = bool(n_cities > 2 and first_0 == first_end)
    self._is_closed_cache = (n_cities, is_closed)
```

> [!CAUTION]
> **The Zero-Array Cache Poisoning Trap**:
> If a test harness or warm-up script executes an untimed pass using a placeholder dummy array:
> ```python
> warm_pop = cp.zeros((pop_size, n), dtype=cp.int32)
> gpu_strategy.crossover_batch(warm_pop, context=warm_ctx)
> ```
> 1. Because the entire array is filled with zeros, `warm_pop[0, 0] == 0` and `warm_pop[0, -1] == 0`.
> 2. The equality check `first_0 == first_end` evaluates to `0 == 0`, which is **`True`**!
> 3. The strategy incorrectly infers that the problem is a **closed tour** and commits `self._is_closed_cache = (n, True)`.
> 4. When the actual benchmark or GA run subsequently executes on an **Open TSP** instance of the same dimension $n$ (where start and end cities are distinct), `self._is_closed_cache[0] == n` matches!
> 5. The poisoned cache returns `is_closed = True`, forcing cut point bounds into $[1, n-2]$ and freezing indices $0$ and $n-1$.
> 6. Boundary cities are treated as fixed depots, cut segments are truncated, duplicate cities are injected, and valid open tour permutations are corrupted!

Furthermore, notice that inspecting `d_parents[0, 0].get()` and `d_parents[0, -1].get()` triggers synchronous Device-to-Host (D2H) memory transfers on cache misses, stalling the host CPU and causing GPU pipeline bubbles.

#### 3. Architectural Prevention Rules

To prevent cache poisoning and eliminate hidden host synchronizations:

1. **Explicit Context Metadata (`ProblemContext.is_closed`)**:
   Topology flags must be defined declaratively in the problem specification (`Problem.is_closed` via `ProblemContext`) rather than inferred by inspecting array contents at runtime (`DP11-10`, `OX-A`). This completely removes runtime heuristic sniffing and eliminates all synchronous `.get()` D2H transfers from the hot path.
2. **Permutation-Valid Warm-Up Payloads**:
   Whenever execution warm-up passes are dispatched, input arrays must be initialized with structurally valid permutations (e.g. `cp.stack([cp.arange(n, dtype=cp.int32) for _ in range(pop_size)])`) reflecting the actual target problem topology (open or closed), ensuring all runtime invariants are strictly satisfied.

---

## 9. Comparison Summary and Scaling Projection

| Architectural Dimension | CPU Reference (`CrossoverOrder`) | Current GPU Kernel (`BUG-22`) | Proposed Redesign (`Task 11.10` / `DP11-11`) | Empirically Measured (`DP11-11`, GTX 1050) |
| :--- | :--- | :--- | :--- | :--- |
| **Membership Structure** | python `set` (Hash table) | None (Brute-force linear scan) | Direct-address array in `__shared__` memory | Direct-address array in `__shared__` memory (Format A) |
| **Membership Complexity** | $\mathcal{O}(1)$ average | $\mathcal{O}(n)$ per gene ($n/3$ comparisons) | $\mathcal{O}(1)$ deterministic (1 SRAM cycle) | $\mathcal{O}(1)$ deterministic (1 SRAM cycle) |
| **Algorithmic Class** | $\mathcal{O}(n)$ per route | $\mathbf{\mathcal{O}(n^2)}$ per route | $\mathbf{\mathcal{O}(n)}$ per route | $\mathbf{\mathcal{O}(n)}$ per route |
| **Empirical Exponent** | $\beta = 0.87$ | $\beta = 1.88$ | $\beta \le 1.05$ (Projected) | **$\beta_{\text{kernel}} = 0.539$ / $\beta_{\text{strategy}} = 0.0422$** |
| **Concurrency Granularity** | 1 thread per route | 1 thread per route | **1 warp (32 threads) per route** | **1 warp (32 threads) per route** |
| **Grid Sizing ($N = 256$)** | Serial loop / joblib | 2 blocks of 128 threads | **32 blocks of 256 threads** | **32 blocks of 256 threads** |
| **Active SMs (GTX 1050)** | N/A (1 CPU core) | **2 of 5 SMs active (60% idle)** | **5 of 5 SMs active (100% saturated)** | **5 of 5 SMs active (100% saturated)** |
| **SM Occupancy** | N/A | **6.25%** (Extreme starvation) | **75.0% – 87.5%** (Optimal SIMT hiding) | **75.0% – 87.5%** (Optimal SIMT hiding) |
| **$n = 52$ Runtime** | $8.79\text{ ms}$ | $0.78\text{ ms}$ ($11.2\times$) | $\approx 0.15\text{ ms}$ ($\approx 58\times$) | **$0.0123\text{ ms}$** kernel ($714\times$) / $2.405\text{ ms}$ strategy |
| **$n = 1{,}002$ Runtime** | $112.50\text{ ms}$ | $168.40\text{ ms}$ ($0.67\times$, loss) | $\approx 3.20\text{ ms}$ ($\approx 35\times$) | **$0.0389\text{ ms}$** kernel ($2{,}892\times$) / $2.595\text{ ms}$ strategy |
| **$n = 2{,}000$ Runtime** | $206.81\text{ ms}$ | $742.05\text{ ms}$ ($0.28\times$, massive loss) | $\approx 6.50\text{ ms}$ ($\approx 31\times$) | **$0.0881\text{ ms}$** kernel ($2{,}347\times$) / $2.445\text{ ms}$ strategy |
| **Speedup vs CPU ($n=2000$)** | $1.0\times$ (Baseline) | $0.28\times$ (Slowdown) | $\approx 31\times$ (Projected) | **$84.58\times$ to $107.39\times$ (Sweep Measured)** |
| **Bit-Identity to Reference** | **Reference Truth** | Divergent (random cuts/pairing) | **100% Bit-Identical Guarantee (`OX-A`)** | **100% Bit-Identical Verified (`pytest` PASS)** |

---

## 10. References

1. **Davis, L.** (1985). *Applying adaptive algorithms to epistatic domains*. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI-85), pp. 162–164.
2. **Oliver, I. M., Smith, D. J., & Holland, J. R. C.** (1987). *A study of permutation crossover operators on the traveling salesman problem*. In Genetic Algorithms and their Applications: Proceedings of the Second International Conference on Genetic Algorithms, pp. 224–230.
3. **Reinelt, G.** (1994). *The Traveling Salesman: Computational Solutions for TSP Applications*. Lecture Notes in Computer Science, vol. 840, Springer-Verlag.
4. **Reinelt, G.** (1995). *TSPLIB95*. Interdisziplinäres Zentrum für Wissenschaftliches Rechnen, Heidelberg.
5. **Rocki, K., & Suda, R.** (2012). *High-performance GPU implementation of genetic algorithms for the Traveling Salesman Problem*. International Conference on High Performance Computing and Simulation (HPCS), IEEE, pp. 488–495.
6. **O'Neil, M. A., & Burtscher, M.** (2014). *Floating-point data compression at 22 GB/s on a GPU*. Proceedings of the 23rd International Conference on Parallel Architectures and Compilation Techniques (PACT), pp. 155–166.
7. **NVIDIA Corporation** (2023). *CUDA C++ Programming Guide (Compute Capabilities: Pascal GP100/GP102/GP104/GP107)*. NVIDIA Developer Documentation.
