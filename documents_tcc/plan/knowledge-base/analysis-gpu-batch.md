---
title: "Interface Granularity Analysis"
created: 2025-02-23
description: >
  An analysis of interface granularity strategies for evolutionary
  algorithms on GPU architectures, evaluating per-individual versus
  whole-population batch approaches based on hardware constraints.
modifications: []
related_files: []
tags:
  - analysis/gpu-architecture
  - analysis/hardware-acceleration
  - algorithm/evolutionary
  - algorithm/evaluation
  - architecture/interfaces
  - architecture/patterns
  - performance/batching
  - performance/latency
  - hardware/pcie-bottleneck
  - hardware/simt
  - implementation/numba
  - implementation/cupy
---

# Strategy Interface Granularity: Architectural Evaluation

This document evaluates the proposed architectural shift from a "Per-individual interface" to a "Whole-population batch interface" for heuristic and evolutionary algorithm strategies, targeting optimal execution on Hardware Accelerators (GPUs) using libraries like Numba and CuPy.

## The Problem: Granularity Bottlenecks in GPU Computing

Traditional object-oriented evolutionary algorithms often employ an orchestrator (typically running on the CPU) that loops over individuals in a population and applies computational strategies (e.g., fitness evaluation, crossover, local search) per individual:

```python
# Traditional (CPU-bound) approach
for ind in population:
    ind = strategy.apply(ind, distances)
```

When porting this architecture to GPUs, maintaining this granularity results in launching thousands of tiny, micro-kernels per generation. 

GPUs excel at **Data Parallelism**—applying the same instruction set to massive arrays of data simultaneously. Launching a kernel incurs a host-to-device communication overhead. If the computational work within the kernel (e.g., evaluating a single TSP tour) is smaller than the overhead to launch the kernel, the GPU will drastically underperform compared to a CPU. 

## Evaluation of Proposed Strategies

### Option A & B: Per-Individual Loops (The Legacy Approach)

Operating at the single-entity level and wrapping elements in `@cuda.jit` or orchestrating the loop on the CPU fundamentally starves the GPU of work. The PCIe bus is saturated with kernel launch commands and small memory transfers, while the thousands of streaming multiprocessors (SMs) on the GPU remain idle.

### Option C: Whole-population Batch Interface (Recommended)

This architecture standardizes the interface to accept the entire data batch simultaneously:

```python
# Batch (GPU-optimized) approach
population = strategy.apply(population, distances, rng)
```

In this model, the `population` is a 2D array (e.g., `(pop_size, num_cities)` in TSP). A single Numba CUDA kernel is launched, mapping a 1D or 2D thread grid directly to the population matrix. 

#### Advantages:

1. **Chromosome-level Granularity**: As established in modern GPU computing literature, mapping one GPU thread (or a cooperative block of threads) to a single chromosome ensures that all SMs are fully utilized in a single parallel sweep.
2. **Zero-Overhead Kernels**: Kernel launch latency is paid once per generation, instead of `pop_size` times.
3. **CAI (CUDA Array Interface) Compatibility**: By keeping the `population` as a continuous device array (e.g., CuPy array or Numba device array), state transitions can occur entirely on the device without costly host-device memory transfers.

## Academic and Industry Grounding

This architectural shift is strongly backed by both specific computational literature and practical constraints of the Numba CUDA backend.

### Evidence from Project Bibliography (`refs.bib`)

1. **`tsp_gpu` (Rocki & Suda)**: Emphasizes the need for high-performance localized optimization mapping directly to SM architecture, avoiding CPU round-trips.
2. **`benaini2018genetic`**: Highlights that for large dynamic Vehicle Routing Problems (VRP), genetic algorithms on GPUs require data structures that allow parallel fitness and crossover evaluation over the entire population matrix to achieve expected speedups.
3. **`schulz2013gpu`**: Demonstrates that discrete optimization on GPUs relies heavily on minimizing algorithmic branching and maximizing coalesced memory access, which is only possible when evaluating batches of solutions together rather than individually.

### Numba Implementation Realities

Recent evaluations of Numba's `@cuda.jit` backend confirm that:
* **Host-Device Transfers**: Must be minimized. A batch interface allows maintaining the population state entirely on the device (`cuda.to_device()` called once at initialization).
* **Launch Overheads**: GPU parallelism requires sufficient work. Launching a kernel for a micro-task (like computing the length of a 50-node tour) takes longer than doing the math. Passing the `(5000, 50)` array matrix allows the kernel grid `eval_kernel[blocks, threads](population, ...)` to hide launch latencies effectively.
* **Vectorized RNG**: The Numba parallel PRNG (e.g., xoroshiro128p) is designed to evaluate arrays of random states across grids, inherently favoring batch interfaces over sequential calls.

## Conclusion

I **strongly agree** with the shift to **Option C: Whole-population batch interface**. 

It is structurally required to unlock the hardware capabilities of the GPU. Any architecture that delegates loops to the python CPU space will bottleneck the algorithmic throughput. By shifting to analytical arrays and bulk operations, the codebase will not only be "GPU-compatible" but "GPU-optimized".
