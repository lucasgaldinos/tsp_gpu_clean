---
title: "Modularizing CUDA Kernels"
source: "https://gemini.google.com/app/55776cec8670d1c0"
author:
  - "[[Lucas Galdino]]"
published:
created: 2026-05-31
description: "Architectural refactoring plan for modularizing GPU-accelerated Genetic Algorithms (GAs) using CuPy, CUDA C++, and Numba, focusing on decoupling monolithic kernels into reusable strategy-based operators."
tags:
  - "computer-science/parallel-computing/cuda,computer-science/software-engineering/design-patterns/strategy,operations-research/optimization/metaheuristics/memetic,optimization/algorithms/combinatorial/traveling-salesman,programming/python/cupy,programming/python/numba,software-engineering/architecture/refactoring"
image: "https://www.gstatic.com/lamda/images/gemini_aurora_thumbnail_4g_e74822ff0ca4259beb718.png"
icon: "https://www.gstatic.com/lamda/images/gemini_sparkle_aurora_33f86dc0c0257da337c63.svg"
---

### 2 & 3. The Strategy Pattern, Timing, and the Base Class

Since python orchestrates every step and the arrays stay on the GPU, you **do not need C++ macros** to turn 2-opt on or off. The `use_2opt` logic is handled entirely by dependency injection in python.

If you want a "Full GPU" run, you simply inject GPU-backed strategies into the base GA. If you want a "CPU" run, you inject CPU-backed strategies.

Here is how detailed and clean `genetic_algorithm.py` becomes:

```py
import time
import cupy as cp

def time_strategy(func):
    """Decorator to measure time taken by any strategy."""
    def wrapper(*args, **kwargs):
        cp.cuda.Stream.null.synchronize()  # Block until previous GPU work finishes
        start = time.perf_counter()
        result = func(*args, **kwargs)
        cp.cuda.Stream.null.synchronize()  # Block until THIS strategy finishes
        elapsed = time.perf_counter() - start
        # You can log this to a dictionary passed to the instance, or print it
        return result, elapsed
    return wrapper

class GeneticAlgorithm:
    def __init__(self, selection_strat, crossover_strat, mutation_strat, local_search_strat=None):
        self.selection = selection_strat
        self.crossover = crossover_strat
        self.mutation = mutation_strat
        self.local_search = local_search_strat  # use_2opt is inherently handled here!
        self.timings = {"selection": 0, "crossover": 0, "mutation": 0, "local_search": 0}

    def run(self, population, distances, generations):
        for gen in range(generations):
            # 1. Selection
            population, t = self._timed_apply(self.selection, population, distances)
            self.timings["selection"] += t

            # 2. Crossover
            population, t = self._timed_apply(self.crossover, population, distances)
            self.timings["crossover"] += t

            # 3. Mutation
            population, t = self._timed_apply(self.mutation, population, distances)
            self.timings["mutation"] += t

            # 4. Local Search (Pluggable!)
            if self.local_search:
                # Can easily accept candidate lists here
                population, t = self._timed_apply(self.local_search, population, distances)
                self.timings["local_search"] += t

        return population

    @time_strategy
    def _timed_apply(self, strategy, pop, dist):
        return strategy.apply(pop, dist)
```

**How the Strategy loads the kernel:**

```python
# strategies/local_search/two_opt.py
import cupy as cp
from ..base import LocalSearchStrategy

class TwoOptGPU(LocalSearchStrategy):
    def __init__(self):
        # Load and compile only once when instantiated
        with open('kernels/two_opt.cu', 'r') as f:
            self.module = cp.RawModule(code=f.read())
        self.kernel = self.module.get_function('two_opt_kernel')

    def apply(self, population, distances):
        # Execute the kernel
        self.kernel((grid,), (block,), (population, distances))
        return population
```

### 4\. NumPy Vectorization vs. Numba for CPU

You asked if you should use NumPy vectorization instead of `Numba` loops for the CPU baselines.

For the TSP, **NumPy vectorization is largely ineffective for the core operators.** Vectorization works when the same operation is applied independently across an entire matrix (like `A = B + C`). However, TSP operators (like OX1 and 2-opt) are highly sequential and dependent on dynamic array indices. In 2-opt, reversing a sub-tour `[i:j]` cannot be easily vectorized across a population because `i` and `j` are different for every single chromosome, and evaluating the cost delta requires scalar lookups in the distance matrix based on those specific moving indices.

Numba is the standard in OR (Operations Research) for python because it takes your readable `for` loops and compiles them to LLVM machine code, bypassing the python interpreter entirely.

### 5 & 6. Is 2-opt doing the same as Crossover? (OX1 Destruction)

> "Thinking about the problem, isn't 2-opt, in a way, doing the same as the crossover?"

No, they serve fundamentally opposite roles in the search space.

- **Crossover (Exploration):** The goal of crossover is to make massive jumps across the search space by recombining large "building blocks" (macros-structures) from two good parents.
- **2-Opt (Exploitation):** The goal of 2-opt is strict, greedy, localized refinement. It fixes micro-inefficiencies (like crossing edges) to pull a specific chromosome down to the absolute bottom of its local mathematical valley.

**Why OX1 ruins 2-opt:** If 2-opt spends computational time perfectly organizing a cluster of 15 cities, the OX1 operator will randomly select a slice index that cuts right through the middle of that cluster, destroying the optimized edges. This is why standard GAs fail when hybridized with heavy local search.

As a researcher, if you want your GPU 2-opt to actually yield results, you must implement a **respectful crossover**. A respectful crossover (like Edge Assembly Crossover - EAX, or Generalized Partition Crossover - GPX) guarantees that if Parent A and Parent B share a common edge, that edge is 100% preserved in the offspring. This allows the GA to recombine the *un-optimized* parts of the tours while preserving the micro-refinements generated by 2-opt.

---

**You**

Nice. So, now what stands is:

<Current-chat-related-questions>

\- Does these new operator also behave well for numba workflows?

\- What about the other steps in GA and its current methods when using parallelization?

\- Are there drawbacks in the proposed solutions?

\- How would the orchestration and benchmarks have to be changed?

</Current-chat-related-questions>

<refactor-questions>

1\. Which articles should I focus on? Please use my references in refs.bib inside the repo provided before, but also @google scholar search for other references, e.g. each and GPX.

2\. The new tree considering proposed methods and the questions above for the current chat?

</refactor-questions>

---

**Gemini**

This is an excellent line of questioning. You are moving away from treating the GPU as a "magic bullet" and starting to look at the algorithmic architecture as a system of interacting, heterogeneous components.

Here is the candid, technical breakdown of how these changes impact your research, the literature you need to back it up, and how your repository should be structured.

## Part 1: Technical Implications

### 1\. Do these new operators (EAX/GPX) behave well for Numba workflows?

**Yes, but with a major caveat regarding data structures.** Both Edge Assembly Crossover (EAX) and Generalized Partition Crossover (GPX) are fundamentally **graph traversal algorithms**. They require finding cycles, alternating paths, and connected components.

- **The Good:** Numba is exceptionally fast at compiling `while` loops and array indexing, which is exactly what graph traversal requires. It will run at near-C++ speeds.
- **The Bad:** Numba's `nopython=True` mode does not play well with dynamic python data structures (like `set()`, `dict()`, or lists of varying sizes). To implement GPX or EAX in Numba, you must represent the parent edge-graphs using pre-allocated flat arrays (like a CSR - Compressed Sparse Row matrix) or fixed-size adjacency lists. Once you map the graph data structures to NumPy arrays, Numba will chew through them effortlessly.

### 2\. What about the other GA steps when using parallelization?

When you break the monolithic GA into a strategy pattern, you will find that different operators have different ideal hardware backends:

- **Selection (Tournament/Roulette):** Extremely parallelizable on the GPU. You can dedicate one thread or warp per tournament. It is memory-bound but trivial for CUDA.
- **Mutation (Swap/Inversion):** Perfectly parallelizable on the GPU. One thread per chromosome.
- **Local Search (2-opt):** Excellent on the GPU, provided you manage memory well (using shared memory for sub-tours).
- **Crossover (EAX/GPX):** **Terrible for the GPU.** Graph partitioning requires variable-length path tracing and cycle detection. This causes massive **thread divergence** (threads in the same warp taking different `if/else` paths) and requires dynamic memory allocation. This is why researchers almost universally run EAX/GPX on the CPU.

### 3\. Are there drawbacks in the proposed solutions?

Yes. The primary drawback of the modular, strategy-based architecture is the **PCI-e Bus Bottleneck**. If your pipeline uses a CPU-based Numba strategy for Crossover (because EAX is bad on GPUs) and a CuPy/CUDA strategy for 2-Opt (because it's great on GPUs), you must transfer the entire population matrix from CPU RAM to GPU VRAM and back *every single generation*.

- **The Mitigation:** You have to weigh the cost. If 2-opt is doing 95% of the computational work, the penalty of transferring the population across the PCI-e bus is worth it. If you run a fast, low-iteration 2-opt, the transfer time might exceed the compute time. Your orchestrator's decorators will reveal this immediately.

### 4\. How do orchestration and benchmarks change?

You are no longer benchmarking `HybridNaive` vs `FullGPU`. You are benchmarking **Pipelines**. Your `algorithms.json` config will change from defining monolithic variants to defining combinations of operators.

```
{
  "pipeline_1": {
    "selection": "TournamentGPU",
    "crossover": "OX1_GPU",
    "local_search": "None"
  },
  "pipeline_2": {
    "selection": "TournamentGPU",
    "crossover": "GPX_CPU",
    "local_search": "2OptGPU"
  }
}
```

Your benchmark results will now output fine-grained times: `crossover_time_avg`, `local_search_time_avg`, and `transfer_time_avg`. This allows you to write a thesis chapter detailing *exactly* where the GPU provides a speedup and where it creates friction.

## Part 2: The Reading List (Focus & New Articles)

Based on your repository's `refs.md` and the current state of TSP research, here is exactly what you should cite and study to justify a CPU-Crossover + GPU-Local Search architecture.

**From your existing `refs.md`:**

1. **Whitley et al. (2010) - *whitley2010hybrid.pdf***: You already have this. This is your primary defense for why standard crossover destroys 2-opt progress, and why **GPX** is required to "tunnel" between local optima.
2. **Fujimoto et al. (2011) - *fujimoto2011highly.pdf***: Use this to justify your GPU memory access patterns for the parallel 2-opt and tournament selection steps.

**New Literature to Search For & Add:**

1. **Nagata & Kobayashi (1997/2013)**: *"A Powerful Genetic Algorithm Using Edge Assembly Crossover for the Traveling Salesman Problem"*.

  - *Why you need it:* EAX is widely considered the most powerful crossover operator for TSP. You need to cite Nagata to explain how building alternating cycles from parent edges works.

1. **Tinós, Whitley, & Ochoa (2019)**: *"A New Generalized Partition Crossover for the Traveling Salesman Problem: Tunneling Between Local Optima"*.

  - *Why you need it:* This introduces **GPX2**, an $O(n)$ complexity evolution of GPX that finds more recombining components. It is the modern standard for Memetic TSP algorithms.

1. **Varadarajan & Whitley (2021)**: *"A Parallel Ensemble Genetic Algorithm for the Traveling Salesman Problem"*.

  - *Why you need it:* This paper specifically discusses running parallel ensembles of EAX and GPX. It provides modern context on how to parallelize these heavy operators (usually via coarse-grained parallel islands, not SIMT/GPU threads).

## Part 3: The Updated Research Directory Tree

This tree reflects a professional research repository. It explicitly separates the hardware backends (CPU vs GPU) into distinct strategy implementations, allowing the orchestrator to dynamically build the GA pipeline without monolithic files.

```
code/src/
├── algorithms/
│   ├── metaheuristics/
│   │   └── genetic_algorithm.py      # Core loop. Injects strategies and times them.
│   ├── strategies/
│   │   ├── base.py                   # Strategy Interfaces (ABC)
│   │   ├── selection/
│   │   │   ├── cpu_roulette.py       # Numba implementation
│   │   │   ├── gpu_tournament.py     # CuPy implementation
│   │   │   └── kernels/
│   │   │       └── selection.cu      # CUDA C++ helpers
│   │   ├── crossover/
│   │   │   ├── cpu_eax.py            # Numba implementation of Edge Assembly (Nagata)
│   │   │   ├── cpu_gpx2.py           # Numba implementation of GPX2 (Whitley)
│   │   │   ├── gpu_ox1.py            # CuPy implementation of standard OX1
│   │   │   └── kernels/
│   │   │       └── ox1.cu
│   │   ├── mutation/
│   │   │   ├── gpu_swap.py
│   │   │   └── kernels/
│   │   │       └── mutation.cu
│   │   └── local_search/
│   │       ├── cpu_2opt.py           # Numba baseline
│   │       ├── gpu_2opt_batch.py     # CuPy implementation
│   │       └── kernels/
│   │           └── two_opt.cu        # The core GPU optimization logic
├── benchmarking_v2/
│   ├── algorithm_runner.py           # Builds the pipeline based on JSON configs
│   ├── pipeline_statistics.py        # NEW: Tracks timing per strategy & PCI-e overhead
│   └── configs/
│       └── pipelines.json            # Defines combinations (e.g. CPU_EAX + GPU_2OPT)
```

By transitioning to this architecture, your research shifts from a simple "CPU vs GPU" comparison to a highly sophisticated analysis of operator-hardware matching.

Do you want to focus on writing the Numba implementation of GPX/EAX next, or should we refine the python orchestrator class that handles the CPU/GPU memory handoffs?

---

**You**

<questions>

<question n=1>

\>Crossover (EAX/GPX): Terrible for the GPU. Graph partitioning requires variable-length path tracing and cycle detection. This causes massive thread divergence (threads in the same warp taking different if/else paths) and requires dynamic memory allocation. This is why researchers almost universally run EAX/GPX on the CPU.

I get it, but aren't there other methods that fit my requirement while being able to be parallelized?

</question>

\---

<question n=2>

\>Yes. The primary drawback of the modular, strategy-based architecture is the PCI-e Bus Bottleneck.

If your pipeline uses a CPU-based Numba strategy for Crossover (because EAX is bad on GPUs) and a CuPy/CUDA strategy for 2-Opt (because it's great on GPUs), you must transfer the entire population matrix from CPU RAM to GPU VRAM and back every single generation.

Ok, but it's just one batched memory transfer. The overhead is possibly negligible, given my vram is 4GB and pcie bus is 16GB/s, the max amount is $0.25s \\times 2$, which in the whole process would be nothing. Doing multiple kernel calls hurts much more, as compiling the code multiple times.

</question>

\---

<question n=3>

\>Varadarajan & Whitley (2021): "A Parallel Ensemble Genetic Algorithm for the Traveling Salesman Problem".

\>

\> Why you need it: This paper specifically discusses running parallel ensembles of EAX and GPX. It provides modern context on how to parallelize these heavy operators (usually via coarse-grained parallel islands, not SIMT/GPU threads).

Well, but if this is the parallel implementation of EAX and GPX, it's a direct contradiction of last answer's excerpt on "crossover strategies", cited in current's `<question n=1>`.

</question>

---

**Gemini**

These are excellent, highly critical questions. You are absolutely right to push back on these points—this is exactly the level of scrutiny required for an algorithmic refactor at the research level. Let's break down each of your points.

### : Are there GPU-friendly, parallelizable alternatives to EAX/GPX?

Yes. The problem with EAX and GPX on the GPU is **cycle detection** (graph traversal). Graph traversal causes *massive* warp divergence because thread $A$ might trace a cycle of 5 cities, while thread $B$ traces a cycle of 500 cities. In a GPU warp (SIMT architecture), thread $A$ has to sit completely idle waiting for thread $B$ to finish.

However, you can achieve the "Respectful" property (preserving common edges) without doing complex cycle detection. Here are two GPU-friendly alternatives:

**1\. The "Common Edge / Greedy Fill" Crossover (Adjacency-Based)** Instead of finding alternating cycles, you strictly focus on **Intersection**.

- **Step 1 (Fully Parallel):** Each GPU thread compares Parent 1 and Parent 2 to find identical edges (e.g., City 5 goes to City 12 in both parents).
- **Step 2 (Fully Parallel):** The offspring inherits *only* these common edges. This creates a highly fragmented set of sub-tours.
- **Step 3 (GPU Friendly):** Instead of intelligently connecting the fragments like GPX does, you just use a simple, deterministic parallel heuristic to reconnect the sub-tours (e.g., sequentially appending missing cities or randomly patching them).
- *Why it works:* Because you are running a highly aggressive **2-opt** immediately after! You don't need the crossover to be perfect; you just need it to transfer the "good" edges from the parents and let 2-opt fix the chaotic greedy patches.

**2\. Sort-Based Crossovers (e.g., Random Keys)** You can map the TSP permutation into an array of continuous floating-point numbers (Random Keys). Crossover becomes a simple mathematical average of the two parent arrays (Standard Arithmetic Crossover), which is $O(1)$ per gene and perfectly parallel on the GPU. You then run a parallel Radix Sort on the GPU (e.g., using `cupy.argsort`) to convert the floats back into a valid TSP permutation. It completely bypasses graph theory and relies on math and sorting—two things GPUs excel at.

### : Is the PCI-e Bottleneck really an issue?

**You are completely correct.** Your math and intuition here highlight a common fallacy in GPU computing literature.

Let's do the exact math for a standard TSP batch:

- Population Size: $10,000$
- Cities (e.g., `pr1002`): $\sim 1,000$
- Data Type: 32-bit Integer (4 bytes)
- Total Matrix Size = $10,000 \times 1,000 \times 4 \text{ bytes} \approx 40 \text{ MB}$.

Transferring 40 MB over a PCIe 3.0 or 4.0 x16 bus (16 GB/s to 32 GB/s) takes roughly **0.0025 seconds**. Even if you run 1,000 generations, the *total* transfer time for the entire benchmark is just **2.5 seconds**.

**Your conclusion is spot on:** The PCI-e transfer of the population array is absolutely negligible. The *actual* overheads in python-GPU hybrid systems are:

1. **JIT Compilation** (which you are fixing by compiling once with `RawModule`).
2. **Kernel Launch Latency:** Every time you call a CUDA kernel from python, there is a $\sim 5-10$ microsecond CPU-GPU handshake overhead. If you have 5 sequential kernels inside a loop of 1,000 generations, that's 5,000 launches, which adds up.
3. **Data Structure Conversions:** Bouncing between NumPy and CuPy arrays if not managed carefully.

*Refactored Takeaway:* Because you correctly identified that the bandwidth is not the bottleneck, the **Hybrid Batch** approach (where the CPU orchestrates, perhaps runs Numba EAX, and hands 40MB off to the GPU for a massively parallel 2-Opt) is computationally sound and arguably the most optimal pipeline for this specific problem size.

### : The Contradiction of "Parallel" EAX/GPX

This is a classic case of ambiguous terminology in High-Performance Computing (HPC) literature.

When Varadarajan & Whitley (2021) title their paper *"A Parallel Ensemble Genetic Algorithm..."*, they are using the word **Parallel** to mean **MIMD (Multiple Instruction, Multiple Data) multi-core CPU parallelism**, *not* **SIMT (Single Instruction, Multiple Threads) GPU parallelism.**

**The Distinction:**

- **CPU Parallelism (MIMD):** If you have a 64-core AWS CPU, Core 1 can execute a complex `while` loop to find a GPX cycle of length 10, and Core 2 can execute a completely different `while` loop to find a cycle of length 500. They operate independently. This is called the "Island Model" or "Coarse-Grained Parallelism." EAX and GPX scale beautifully here.
- **GPU Parallelism (SIMT):** On a GPU, threads are grouped into "Warps" of 32 threads. They share the same instruction fetcher. If Thread 1 is stuck in a `while` loop tracking a 500-node cycle, the other 31 threads in that warp *cannot move on to the next instruction* until Thread 1 finishes. This is **Warp Divergence**.

*To summarize:* EAX/GPX are excellent for **CPU multi-threading** (which the 2021 paper demonstrates), but they are functionally terrible for **GPU hardware**. Therefore, there is no contradiction—the paper validates scaling these operators across CPU clusters, further supporting the idea that if you want to use GPX, you should map it to your CPU (via Numba) and map your 2-opt to your GPU (via CUDA).

---

**You**

Nice, knowing all these, I need a full implementation plan regarding, considering the nuances of what was said. I'll provide what a good plan may need. Consider each step.

I provided a `plan` folder, where there's a `plan.md` file and related files. Focus on the writing the plan inside a snippet. Be extremely thorough when writing those, please.

---

**Gemini**

Here is a comprehensive, step-by-step implementation plan to refactor your repository from monolithic kernels into a highly modular, research-grade Strategy architecture.

This plan accounts for the hardware nuances, PCIe transfer logging, Numba graph-traversal compatibility, and JIT compilation bottlenecks we discussed.

# Comprehensive Implementation Plan: GPU-Accelerated GA Refactor

## Phase 1: Directory Restructuring & Core Interfaces

**Objective:** Decouple algorithmic logic into domain-specific strategy folders and establish strict input/output contracts.

1. **Restructure Directory Tree**
   + Create `code/src/algorithms/strategies/` with subdirectories: `selection/`, `crossover/`, `mutation/`, and `local_search/`.
   + Within each subdirectory, create a `kernels/` folder specifically for `.cu` and `.cuh` files.
2. **Define Base Strategy Contracts (`base.py`)**
   + Create `AbstractBaseClass` definitions for each operator.
   + Enforce the signature: `def apply(self, population: cp.ndarray, distances: cp.ndarray) -> cp.ndarray:`
   + *Nuance Check:* Ensure CPU/Numba strategies accept `cp.ndarray`, handle the `.get()` (VRAM to RAM) conversion internally, and return `.asarray()` back to VRAM.

## Phase 2: The Unified Orchestrator & Micro-Benchmarking

**Objective:** Replace monolithic algorithm variants with a single pipeline executor that accurately tracks hardware handoffs.

1. **Implement `GeneticAlgorithm` (`genetic_algorithm.py`)**
   + Deprecate all previous subclass variants (`HybridNaive`, `FullGPU`, etc.).
   + Implement constructor using Dependency Injection to receive instantiated strategy classes (e.g., `crossover_strat`, `local_search_strat`).
2. **Implement `@time_strategy` Decorator**
   + Wrap strategy calls to capture precise execution times.
   + **Critical:** Insert `cupy.cuda.Stream.null.synchronize()` before and after the `time.perf_counter()` to force the GPU to finish execution, preventing asynchronous launch latency from corrupting operator timings.
3. **PCIe Handoff Tracking**
   + Add logic within the orchestrator to track total time spent transferring arrays between CPU/GPU when heterogeneous pipelines (CPU Crossover + GPU 2-Opt) are assembled.

## Phase 3: Hardware-Specific Strategy Implementations

**Objective:** Migrate existing logic into modular classes and introduce new robust crossovers.

1. **GPU Strategies (CuPy `RawModule`)**
   + **Local Search (`local_search/gpu_two_opt.py`):** Wrap the existing `two_opt_device` helper. Implement Class-Level caching for `cp.RawModule` so the C++ code is only compiled *once* per benchmark suite execution, not once per instance.
   + **Selection/Mutation:** Wrap tournament selection and swap mutation into their own isolated strategy classes.
2. **CPU Strategies (Numba `njit`)**
   + **Respectful Crossover (`crossover/cpu_gpx.py`):** Implement Generalized Partition Crossover (GPX2).
   + *Nuance Check:* Since GPX requires graph cycle detection (terrible for GPUs), implement this purely in python. Decorate with `@numba.njit(nogil=True)`. Map dynamic graph components into pre-allocated NumPy flat arrays (CSR format) to ensure Numba can compile it optimally.

## Phase 4: Benchmarking Suite Overhaul (`benchmarking_v2/`)

**Objective:** Adapt the runner to handle dynamic pipeline generation rather than static class mapping.

1. **Refactor `configs/algorithms.json`**
   + Deprecate old variant maps.
   + Introduce Pipeline definitions. Example:

     ```json
     {
       "Pipeline_EAX_2Opt": {
         "selection": "TournamentGPU",
         "crossover": "EAX_CPU",
         "mutation": "SwapGPU",
         "local_search": "TwoOptGPU"
       }
     }
     ```

2. **Update `algorithm_runner.py`**
   + Implement a Strategy Factory/Builder that reads `algorithms.json`, initializes the requested strategies (fetching from the JIT cache), and injects them into the base `GeneticAlgorithm` class.
3. **Database & Report Generation Adaptation**
   + Update SQLite schemas/JSON writers to accept dictionary objects for timings rather than a single `total_time` float. Add fields for `crossover_time_avg`, `local_search_time_avg`, and `pcie_transfer_time`.

## Phase 5: Validation & "Iso-algorithmic" Checks

**Objective:** Ensure the new modular pipeline produces mathematically identical outputs to the monolithic approach.

1. **Seed Synchronization**
   + Expose the CuPy/CUDA RNG seeds at the Strategy level to guarantee deterministic runs.
2. **Unit Tests (`tests/validation/`)**
   + Run the new `Pipeline_FullGPU` (using standard OX1 crossover) against the old `FullGPU` monolithic kernel over 5 generations on `berlin52.tsp`.
   + Assert that the exact same optimal path and fitness scores are achieved to prove that the modular boundary handoffs are functionally invisible.

### Strategic Recommendations for Execution

1. **Start with Phase 1 & 2:** Create the "dummy" strategies first. Move your existing, basic operators (OX1, naive 2-opt) into the new folder structure. Get the Orchestrator running and verifying that the `@time_strategy` decorator correctly splits your timings.
2. **Tackle Numba Second (Phase 3.2):** Implementing EAX or GPX2 in Numba will be the mathematically hardest part of this refactor because of the array-flattening requirement. Isolate this in a separate Jupyter Notebook before pasting it into the main repository pipeline.
3. **Tackle the Runner Last:** Once you can manually instantiate the `GeneticAlgorithm` and prove it works, then update the `algorithms.json` and your massive benchmark runner to read the new format.
