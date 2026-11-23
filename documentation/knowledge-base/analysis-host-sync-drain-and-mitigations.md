---
title: "Understanding the `host_sync` Timing Bucket and Asynchronous CUDA Mitigation Strategies"
description: >
  Technical explanation of why `host_sync` (`Sinc.`) captures 36%–95% of wall-clock time in
  `FullGPUModular` under host-clock profiling, separating real asynchronous GPU kernel compute
  from fixed launch/synchronization latency, and evaluating four architectural mitigations:
  batched K-generation convergence checks, kernel fusion and shared-memory tour caching, CUDA
  Graphs, and device-side RNG.
created: 2026-11-22
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-22
    modifications:
      - description: >
          Created reference note explaining the physical meaning of `host_sync` under
          `PhaseTimer(use_cuda_events=False)` and documenting four concrete mitigation paths
          with plain-language mental models, accounting equations, and code anchors.
related_files:
  - "[genetic_algorithm.py::GeneticAlgorithm.evolve](../../src/algorithms/metaheuristics/genetic_algorithm.py)"
  - "[phase_timer.py::PhaseTimer](../../src/utils/phase_timer.py)"
  - "[two_opt_batch.cu::two_opt_batch_kernel](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[order_crossover.cu::order_crossover_kernel](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[analysis-timing-instrumentation-and-warmup.md](./analysis-timing-instrumentation-and-warmup.md)"
  - "[analysis-gpu-tsp-future-improvements.md](./analysis-gpu-tsp-future-improvements.md)"
  - "[analysis-gpu-two-opt-dispatch-and-reduction.md](./analysis-gpu-two-opt-dispatch-and-reduction.md)"
tags:
  - analysis/performance
  - analysis/profiling
  - analysis/architecture
  - analysis/optimization
  - benchmark/gpu
  - benchmark/performance
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - guide/cuda
  - notes/cuda
---

# Understanding the `host_sync` Timing Bucket and Asynchronous CUDA Mitigation Strategies

## 1. What `host_sync` (`Sinc.`) Actually Measures

In Chapter 4 phase-timing tables, the `FullGPUModular` variant reports between $36.0\%$ (`berlin52`) and $94.8\%$ (`pr1002`) of total wall-clock time under the `Sinc.` (`host_sync`) column. **This bucket is not idle waiting waste.** Under host-side timing, `host_sync` captures **all asynchronous GPU kernel compute queued during the generation**, plus a smaller fixed dispatch and device-to-host (`D2H`) synchronization cost.

### 1.1 Asynchronous Enqueue vs. Blocking Drain

Think of the CPU as an order clerk writing work tickets and the GPU as a factory floor processing those tickets in a FIFO queue (the CUDA stream):

1. **Host clock by default:** Both [phase_timer.py::PhaseTimer](../../src/utils/phase_timer.py) (`use_cuda_events=False`) and [phase_timer.py::time_strategy](../../src/utils/phase_timer.py) default to `time.perf_counter()` on the CPU inside [phase_timer.py::_timed_call](../../src/utils/phase_timer.py).
2. **Non-blocking kernel dispatch:** During each generation of `FullGPUModular`, `selection`, `crossover`, `mutation`, `local_search`, `evaluation`, and `survival` merely enqueue CUDA kernels onto the GPU stream and return control to Python in microseconds. Their individual phase buckets therefore record only *enqueue latency* ($0.3\%$ to $10.0\%$).
3. **End-of-generation drain (`_best_cost`):** At the end of each generation in [genetic_algorithm.py::GeneticAlgorithm.evolve](../../src/algorithms/metaheuristics/genetic_algorithm.py), the stopping check calls [genetic_algorithm.py::GeneticAlgorithm._best_cost](../../src/algorithms/metaheuristics/genetic_algorithm.py):

```python
if is_gpu_array(fitness):
  return float(require_gpu().min(fitness).get())
```

In CuPy, `.get()` issues a synchronous device-to-host copy (`cudaMemcpy`), which forces the CPU to pause inside `"host_sync"` until **every previously queued kernel in that generation finishes executing on the GPU** and returns the scalar minimum (plus the full route via [genetic_algorithm.py::GeneticAlgorithm._extract_best](../../src/algorithms/metaheuristics/genetic_algorithm.py) only when a strict improvement occurs).

$$
T_{\text{host\_sync}} \;=\; \underbrace{\sum_{k \in \text{stream}} T_{\text{kernel\_compute}}(k)}_{\text{real GPU work (dominant on } n \ge 500\text{)}} \;+\; \underbrace{T_{\text{sync\_barrier}} + T_{\text{D2H\_scalar}}}_{\text{fixed per-generation latency}}
$$

By contrast, `HybridOptimized` reports $\approx 0.1\%$ in `host_sync` because its `local_search` and `evaluation` steps already copy route matrices or fitness vectors back to CPU via `.get()` *inside* their own strategy calls, paying the wait inside `BL` and `Aval.`.

> [!NOTE]
> See [analysis-timing-instrumentation-and-warmup.md §4](./analysis-timing-instrumentation-and-warmup.md#4-the-async-bleed-survivalstep-and-the-host_sync-bucket) for the original wall-clock closure equation showing why omitting `use_cuda_events=True` between operators avoids inserting artificial waits that slow down the pipeline.

---

## 2. Scale Regime Determines What Dominates `host_sync`

| Problem scale | Example instance | Mean ms/generation (`FullGPUModular`) | Dominant component inside `host_sync` |
| :--- | :--- | ---: | :--- |
| Small ($n < 100$) | `att48`, `berlin52` | $\approx 3.2\text{–}4.5\ \text{ms}$ | Fixed per-generation launch, driver dispatch, and D2H sync latency ($\approx 0.5\text{–}1.0\ \text{ms}$) |
| Large ($n \ge 500$) | `pr1002` | $\approx 28.6\ \text{ms}$ | Actual GPU kernel compute ($2$-opt neighborhood scans and evolutionary operators) |

---

## 3. Mitigation Strategies

Because the dominant term shifts with problem dimension $n$, mitigations fall into two distinct groups: reducing **synchronization/dispatch overhead** (helps small instances) vs. reducing **GPU kernel execution time** (helps large instances).

### 3.1 Batched Convergence Checks Every $K$ Generations

- **Mental model:** Instead of asking *"are we done yet?"* after every single generation—forcing the GPU queue to drain completely each time—the CPU submits work tickets for $K$ consecutive generations (for example, $K = 5$ or $K = 10$) and calls `_best_cost()` only once at the end of each $K$-generation block.
- **What improves:** Amortizes the blocking D2H synchronization latency and keeps the GPU pipeline continuously fed without inter-generation stalls.
- **Trade-off:** If the target $\varepsilon_{\text{alvo}} = 1\%$ is crossed on generation 2 of a 5-generation block, the solver executes up to $K - 1$ extra generations before observing the stopping condition. Best suited for fast instances ($n < 150$) where per-generation compute is only a few milliseconds.

### 3.2 Kernel Fusion and Shared-Memory Caching

On large instances such as `pr1002`, accelerating the kernels themselves is the only way to substantially shrink `host_sync`:

1. **Fuse lightweight evolutionary kernels:** Today a single generation dispatches separate kernels for selection, [order_crossover.cu::order_crossover_kernel](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu), swap mutation, [two_opt_batch.cu::two_opt_batch_kernel](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu), cost evaluation, and survival. Between `order_crossover_kernel` and mutation, the full $(n_{\text{pop}} \times n)$ offspring matrix is written to global VRAM and immediately read back from global VRAM just to swap two positions per selected route. Fusing crossover and mutation into one kernel keeps the freshly generated offspring in fast registers or shared memory and applies the swap before committing to VRAM, saving one global-memory round trip and one kernel launch.
2. **Cache the tour in shared memory under candidate-list 2-opt:** In [two_opt_batch.cu::two_opt_batch_kernel](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu), when `candidate_k > 0` (`use_candidates`), shared memory holds only `s_deltas`, `s_swap_i`, and `s_swap_j`, so city lookups along `tour` hit global VRAM repeatedly instead of on-chip shared memory. Caching the active route in `s_tour` (whenever $n$ fits within the 48 KiB block limit) reduces global-memory read pressure inside the dominant $2$-opt kernel.

### 3.3 CUDA Graphs (Pre-Recorded Execution Templates)

- **Mental model:** Even when kernels launch asynchronously, every call (`selection` $\to$ `OX` $\to$ `mutation` $\to$ `2-opt` $\to$ `eval` $\to$ `survival`) requires Python and the CUDA driver to assemble and validate a fresh launch descriptor (kernel pointer, grid/block dimensions, and VRAM buffer arguments).
- **How CUDA Graphs work:** On the first iteration of a $K$-generation block (with fixed VRAM buffer addresses), the CPU records the exact chain of kernel launches into a **CUDA Graph**—essentially a pre-validated execution macro stored in the driver. Subsequent iterations replay the entire multi-kernel chain with a single lightweight launch call instead of six separate driver submissions.
- **Impact:** Removes host/driver launch overhead on small and medium instances; minimal effect when `two_opt_batch_kernel` runtime dominates ($n \ge 500$).

### 3.4 Device-Side Random Number Generation (Production Mode Only)

Currently the CPU draws stochastic control vectors on NumPy and uploads $\approx 11.26\ \text{KB}$ per generation (`H2D`) so that `FullGPUModular` follows the exact same seed stream as the CPU baseline (`guide-iso-algorithmic-ga-validation`). Replacing those host draws with GPU-side PRNG (e.g., `curandState_t` or CuPy RNG) eliminates the per-generation `H2D` copy when CPU bit-identity is not required.

### 3.5 Diagnostic-Only Device Timing (`PhaseTimer(use_cuda_events=True)`)

Passing `use_cuda_events=True` to [phase_timer.py::PhaseTimer](../../src/utils/phase_timer.py) instruments each phase with `cp.cuda.Event()` pairs (`armed_end.synchronize()`), redistributing the GPU compute out of `host_sync` into `local_search`, `crossover`, etc. This clarifies internal GPU phase shares for profiling, **but introduces inter-operator waits that inflate wall-clock time** and must remain disabled during production timing runs.

---

## 4. Verification Anchor

Inspect the two `host_sync` call sites and the default `PhaseTimer` configuration:

```bash
rtk rg -n "host_sync|use_cuda_events" src/algorithms/metaheuristics/genetic_algorithm.py src/utils/phase_timer.py
```

Expected: `PhaseTimer.__init__` defaults to `use_cuda_events: bool = False`, and `GeneticAlgorithm.evolve` records `"host_sync"` solely around `_best_cost` and `_extract_best`.

---

## 5. References

1. NVIDIA Corporation (2024). *CUDA C++ Programming Guide* — Streams, Events, Asynchronous Execution, and CUDA Graphs. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
2. CuPy Team. *CuPy User Guide* — Streams, Events, and implicit synchronization on `cupy.ndarray.get()`. <https://docs.cupy.dev/en/stable/user_guide/performance.html>
