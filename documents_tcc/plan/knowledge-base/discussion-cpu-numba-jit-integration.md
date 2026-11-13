---
title: "Architectural Specification and Implementation Plan for CPU Numba JIT Acceleration"
description: >
  Architectural design, LLVM compilation patterns, ahead-of-time (AOT) warm-up protocols,
  OpenMP multi-core parallelization, and graceful fallback mechanisms for Numba JIT
  acceleration in `gpu_accelerated_clean`.
created: 2026-11-12
status: development
author:
  - "[[Lucas Galdino]]"
type: plan
scope: global
modifications:
  - date_modified: 2026-11-12
    modifications:
      - description: >
          Created dedicated Numba JIT integration specification: OpenMP multi-core
          parallelization (prange), AOT warm-up protocol integration with algorithm_runner.py,
          zero bit-parity regression safeguards, and fallback helper decorators.
related_files:
  - "documents_tcc/plan/knowledge-base/analysis-current-cpu-bottlenecks.md"
  - "src/algorithms/strategies/local_search/two_opt.py"
  - "src/algorithms/strategies/evaluation/cost_evaluation.py"
  - "src/algorithms/strategies/base.py"
  - "src/algorithms/metaheuristics/genetic_algorithm.py"
  - "src/benchmarking/runner/algorithm_runner.py"
tags:
  - "plan/numba"
  - "architecture/jit"
  - "parallelism/openmp"
  - "metaheuristics/cpu"
subagents_ids:
  - "893e9832-65d2-401a-8e02-929d2f2948b8"
  - "c9ac42ab-bdb2-42c2-b5a7-32ed0c0ca4bb"
---

# Architectural Specification and Implementation Plan for CPU Numba JIT Acceleration

- [Architectural Specification and Implementation Plan for CPU Numba JIT Acceleration](#architectural-specification-and-implementation-plan-for-cpu-numba-jit-acceleration)
  + [1. Executive Summary & Core Objectives](#1-executive-summary--core-objectives)
  + [2. Architectural Principles & Non-Negotiables](#2-architectural-principles--non-negotiables)
  + [3. Multi-Threading Control & Thread Allocation](#3-multi-threading-control--thread-allocation)
  + [4. The Ahead-of-Time (AOT) Warm-Up Protocol](#4-the-ahead-of-time-aot-warm-up-protocol)
    - [4.1 The Mechanism (Eliminating the 300 ms JIT Spike)](#41-the-mechanism-eliminating-the-300-ms-jit-spike)
    - [4.2 Updates Required in `algorithm_runner.py`](#42-updates-required-in-algorithm_runnerpy)
  + [5. Concrete Kernel Definitions & Blueprint Code](#5-concrete-kernel-definitions--blueprint-code)
    - [5.1 Numba Helper & Fallback (`src/utils/numba_helpers.py`)](#51-numba-helper--fallback-srcutilsnumba_helperspy)
    - [5.2 Parallel Numba 2-Opt Batch Kernel (`two_opt_numba.py`)](#52-parallel-numba-2-opt-batch-kernel-two_opt_numbapy)
    - [5.3 Zero-Allocation Fused Cost Evaluation (`cost_evaluation_numba.py`)](#53-zero-allocation-fused-cost-evaluation-cost_evaluation_numbapy)
    - [5.4 Strategy Integration & Warm-Up Hook (`LocalSearchCPUTwoOpt`)](#54-strategy-integration--warm-up-hook-localsearchcputwoopt)
  + [6. Step-by-Step Implementation Roadmap](#6-step-by-step-implementation-roadmap)
  + [7. Verification Standards & Acceptance Gate](#7-verification-standards--acceptance-gate)

---

## 1. Executive Summary & Core Objectives

This document specifies the architecture, compilation strategy, and integration roadmap for accelerating the CPU solver using **Numba JIT (LLVM compilation) with OpenMP multi-threading**.

Complementing the pure NumPy optimizations documented in [`analysis-current-cpu-bottlenecks.md`](analysis-current-cpu-bottlenecks.md), Numba JIT provides:

1. **$30\times$ to $80\times$ execution speedups on CPU**: Compiling the $\mathcal{O}(n^2)$ 2-opt inner loop into native machine code and executing across all available CPU physical cores via `numba.prange`.
2. **Zero-Allocation Fused Loops**: Evaluating tour costs and move deltas in $\mathcal{O}(1)$ memory without creating intermediate NumPy array buffers.
3. **100% Architectural Compatibility**: Full adherence to `StrategyBase`, `LocalSearchStrategy`, and the declarative `STAGE_SPECS` dispatch table.
4. **Ahead-of-Time (AOT) Warm-Up Protocol**: Ensuring that LLVM compilation occurs before benchmark measurements begin, preventing spurious 300 ms spikes in Repetition 1.

---

## 2. Architectural Principles & Non-Negotiables

1. **Lego Blocks Organization**:
   Numba JIT kernels must remain isolated from high-level strategy orchestration. Mirroring the design of CUDA kernels (`.cu` files), Numba kernels reside in dedicated files:
   + `src/algorithms/strategies/local_search/two_opt_numba.py`
   + `src/algorithms/strategies/evaluation/cost_evaluation_numba.py`
   + `src/algorithms/strategies/crossover/crossover_numba.py`
2. **Graceful Fallback & Zero Mandatory Dependencies**:
   Numba must not become a hard requirement that breaks pure-python environments. [`src/utils/numba_helpers.py`](../../../src/utils/numba_helpers.py) provides a `@numba_njit` decorator that automatically falls back to pure NumPy if Numba is not installed or if `NUMBA_DISABLE_JIT=1` is exported in the environment.
3. **Deterministic Bit-Parity (No `fastmath=True` on 2-Opt)**:
   Do **not** use `fastmath=True` in 2-opt kernels. Floating-point re-association can alter delta tie-breaking when `delta < 0`, violating strict bit-parity tests with existing CPU/GPU acceptance tests ([`test_unit_two_opt_boundaries.py#L378`](../../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py#L378)).
4. **Memory-Conscious Typing**:
   Kernels must accept integer distance matrices (`int32`) and route permutations (`int32`), matching the TSPLIB95 specification audit.

---

## 3. Multi-Threading Control & Thread Allocation

Numba executes loops marked with `nb.prange` using an internal OpenMP or TBB thread pool.

```text
Population Batch: routes (N = 256, n_cities)
       │
       ├── Core 0 / Thread 0 ──> processes routes[0..31]
       ├── Core 1 / Thread 1 ──> processes routes[32..63]
       ├── Core 2 / Thread 2 ──> processes routes[64..95]
       ├── Core 3 / Thread 3 ──> processes routes[96..127]
       ├── Core 4 / Thread 4 ──> processes routes[128..159]
       ├── Core 5 / Thread 5 ──> processes routes[160..191]
       ├── Core 6 / Thread 6 ──> processes routes[192..223]
       └── Core 7 / Thread 7 ──> processes routes[224..255]
```

### Thread Contention Mitigation

When running multi-instance benchmark sweeps (e.g. executing independent problems concurrently with python `multiprocessing`), each worker process could potentially launch an 8-thread Numba pool, creating severe thread over-subscription and CPU cache thrashing.

- **Control API**: Expose an explicit configuration in `AlgorithmConfig` / `GeneticAlgorithm`:

  ```python
  import numba

  if cpu_threads is not None:
    numba.set_num_threads(cpu_threads)

```

---

## 4. The Ahead-of-Time (AOT) Warm-Up Protocol

### 4.1 The Mechanism (Eliminating the 300 ms JIT Spike)

In Phase 11, we established the ahead-of-time warm-up protocol ([`StrategyBase.warmup` in `src/algorithms/strategies/base.py#L77-L110`](../../../src/algorithms/strategies/base.py#L77-L110) and [`GeneticAlgorithm.warmup` in `src/algorithms/metaheuristics/genetic_algorithm.py#L472-L498`](../../../src/algorithms/metaheuristics/genetic_algorithm.py#L472-L498)).

We extend this mechanism to CPU Numba strategies:

```text
Benchmark Runner (algorithm_runner.py)
   │
   ├──> Creates isolated clone: warmup_instance = copy.deepcopy(algorithm_instance)
   │
   └──> Calls warmup_instance.warmup(warmup_context)
          │
          ├──> GeneticAlgorithm walks every pipeline step & strategy
          │      │
          │      └──> Strategy.warmup(context)
          │             │
          │             └──> Invokes Numba kernel with dummy batch: _warmup_batch(rows=2)
          │                    └──> Triggers LLVM JIT compilation AOT!
```

### 4.2 Updates Required in `algorithm_runner.py`

In [`src/benchmarking/runner/algorithm_runner.py#L168-L184`](../../../src/benchmarking/runner/algorithm_runner.py#L168-L184):

Currently, warm-up is guarded by `if use_gpu:`.

1. Update the condition to: `if repetitions > 0:` so CPU variants also pre-compile ahead of the measured loop.
2. Set `warmup_xp = (cupy_backend() if use_gpu else np) or np`.
3. Guard the CuPy-only check `_compile_kernel_cached.cache_info()` with `if use_gpu:`.

---

## 5. Concrete Kernel Definitions & Blueprint Code

### 5.1 Numba Helper & Fallback (`src/utils/numba_helpers.py`)

```python
"""Numba environment detection, compilation helpers, and fallback decorators."""

from __future__ import annotations

import os
from typing import Any, Callable

try:
  import numba as nb

  HAS_NUMBA = os.environ.get("NUMBA_DISABLE_JIT", "0") != "1"
except ImportError:
  HAS_NUMBA = False
  nb = None  # type: ignore[assignment]


def numba_njit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Drop-in decorator for @nb.njit with automatic fallback to pure python/NumPy."""
  if HAS_NUMBA:
    default_kwargs = {"nogil": True, "cache": True}
    default_kwargs.update(kwargs)
    return nb.njit(*args, **default_kwargs)  # type: ignore[no-any-return]

  def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    return func

  return decorator
```

### 5.2 Parallel Numba 2-Opt Batch Kernel (`two_opt_numba.py`)

```python
"""Numba-accelerated 2-opt batch kernels with OpenMP multi-core execution."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, nogil=True, cache=True)
def numba_2opt_batch_best(
  routes: np.ndarray,  # shape (N, n), int32
  distances: np.ndarray,  # shape (n, n), int32
  max_iterations: int,
) -> np.ndarray:
  """Run 2-opt steepest descent across N routes in parallel using all CPU cores."""
  N, n = routes.shape
  improved = routes.copy()

  for idx in nb.prange(N):
    tour = improved[idx]
    for _ in range(max_iterations):
      best_delta = 0
      best_i = -1
      best_j = -1

      for i in range(n - 2):
        a = tour[i]
        b = tour[i + 1]
        d_ab = distances[a, b]
        limit_j = n - 1 if i == 0 else n

        for j in range(i + 2, limit_j):
          c = tour[j]
          d = tour[0] if j == n - 1 else tour[j + 1]
          delta = (distances[a, c] + distances[b, d]) - (d_ab + distances[c, d])
          if delta < best_delta:
            best_delta = delta
            best_i = i
            best_j = j

      if best_delta < 0:
        # In-place two-pointer segment reversal (zero heap allocation)
        left = best_i + 1
        right = best_j
        while left < right:
          tmp = tour[left]
          tour[left] = tour[right]
          tour[right] = tmp
          left += 1
          right -= 1
      else:
        break

  return improved
```

### 5.3 Zero-Allocation Fused Cost Evaluation (`cost_evaluation_numba.py`)

```python
"""Numba-accelerated closed-loop TSP tour cost evaluation."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, nogil=True, cache=True)
def numba_evaluate_batch(
  routes: np.ndarray,  # shape (N, n), int32
  distances: np.ndarray,  # shape (n, n), int32
) -> np.ndarray:
  """Compute tour costs for N routes in O(1) extra memory without intermediate arrays."""
  N, n = routes.shape
  costs = np.empty(N, dtype=np.int64)

  for idx in nb.prange(N):
    tour = routes[idx]
    total = 0
    for j in range(n - 1):
      total += distances[tour[j], tour[j + 1]]
    total += distances[tour[n - 1], tour[0]]  # Close the loop
    costs[idx] = total

  return costs
```

### 5.4 Strategy Integration & Warm-Up Hook (`LocalSearchCPUTwoOpt`)

```python
class LocalSearchCPUTwoOpt(StrategyBase, LocalSearchStrategy):
  # ... existing initialization ...

  def warmup(self, context: ProblemContext) -> None:
    """Pre-compile Numba LLVM kernels ahead of the measured benchmark loop."""
    if HAS_NUMBA:
      dummy_batch = self._warmup_batch(context, rows=2)
      distances = context.get_cpu_distances()
      # JIT compile 2-opt with a 1-pass dummy run
      numba_2opt_batch_best(dummy_batch, distances, max_iterations=1)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    batch_np, was_1d, was_gpu = to_cpu_batch(routes)
    distances = context.get_cpu_distances()

    if HAS_NUMBA and self.mode == "best" and self.candidate_k is None:
      improved = numba_2opt_batch_best(batch_np, distances, self.max_iterations)
    else:
      # Seamless fallback to pure NumPy implementation
      improved = self._fallback_improve_batch(batch_np, distances, context)

    return restore_routes_shape(improved, was_1d, to_gpu=was_gpu)
```

---

## 6. Step-by-Step Implementation Roadmap

1. **Step 1: Utility & Fallback Harness**:
   Create `src/utils/numba_helpers.py` with `@numba_njit` and JIT toggle support.
2. **Step 2: Fused Evaluation Strategy**:
   Implement `cost_evaluation_numba.py` and connect it to `EvaluationCPUCost.evaluate_batch`.
3. **Step 3: Multi-Core 2-Opt Kernel**:
   Implement `two_opt_numba.py` with `prange(N)` parallelization and in-place segment reversal.
4. **Step 4: Warm-Up Hook Expansion**:
   Add `warmup(context)` to `LocalSearchCPUTwoOpt` and `EvaluationCPUCost`. Update `algorithm_runner.py` to trigger warm-up for CPU runs when `repetitions > 0`.
5. **Step 5: Verification & Bit-Parity**:
   Run the test suite (`pytest tests/algorithms/strategies -q`) and verify that Numba output matches the CPU/GPU baseline outputs to $100\%$ numerical precision.

---

## 7. Verification Standards & Acceptance Gate

1. **Test Gate**: All existing tests in `tests/algorithms/strategies/` must pass with zero regressions.
2. **Bit-Parity**: `numba_2opt_batch_best` must yield identical tour orderings and costs as the baseline `LocalSearchCPUTwoOpt` for identical initial tours.
3. **Warm-Up Verification**: Running `algorithm_runner.py` on a CPU preset must exhibit $< 5\%$ runtime variance between Repetition 1 and Repetition 2 (proving zero JIT latency leaks into the measured repetitions).
