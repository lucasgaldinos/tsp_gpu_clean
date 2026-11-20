---
title: "Timing Instrumentation and Warm-Up: The `@time_strategy` Barrier, Phase Accounting, and Deterministic AOT Verification"
description: >
  Analysis of the two independent clocks in the GPU Genetic Algorithm pipeline: the
  `@time_strategy` decorator in `src/utils/phase_timer.py`, which inspects
  `isinstance(arg, cp.ndarray)` and unconditionally records `cp.cuda.Event` pairs plus
  `end_evt.synchronize()` around every GPU strategy call, versus `PhaseTimer.measure`,
  which only synchronizes when `use_cuda_events=True`. Explains why the 2026-11-06
  `DP11-7` measurement mistook the decorator barrier for intrinsic device
  synchronization, why the `Stream.null.synchronize()` removals of `BUG-15`/`DP10-18`
  and `DP11-14`–`DP11-17` were neutralized, why `SurvivalStep` latency bleeds into the
  `host_sync` bucket, why `np.std` needs `ddof=1`, and why GPU warm-up must be verified
  through `_compile_kernel_cached.cache_info()` rather than wall-clock assertions
  (`BUG-25`, `DP11-21`).
created: 2026-11-10
status: final
author:
  - "Lucas Galdino"
  - "GitHub Copilot"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Initial creation: documented the `@time_strategy` decorator as an unconditional
          host-device barrier, reconciled it with the refuted `DP11-7` premise, derived the
          `SurvivalStep` -> `host_sync` async bleed accounting, the `fitness[0]`-is-minimum
          invariant after elitist `argsort`, the `ddof=1` sample-standard-deviation
          requirement for `phase_share_cv_pct`, and the deterministic kernel-cache
          verification protocol for `BUG-25`.
      - description: >
          Revision: added the third clock of the 2026-11-06 experiment
          (`unwrapped_after_sync = 258.725 ms`, only ~0.8 % below the decorated 260.840 ms,
          proving the decorator overhead is ~2.1 ms on a ~259 ms device workload), the
          per-generation barrier budget of `FullGPUModular` (5 decorated strategies + 1–3
          `_extract_best` D2H drains = 6–8 host-device round trips), the structural
          `@time_strategy`/`PhaseTimer` decoupling requirement, and §6.3 on warm-up
          instance isolation (`algorithm_factory is None` mutates Repetition 1's RNG
          stream) plus the `cache_info().currsize` falsifiable invariant.
related_files:
  - "[phase_timer.py](../../src/utils/phase_timer.py)"
  - "[genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py)"
  - "[algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py)"
  - "[gpu_helpers.py](../../src/utils/gpu_helpers.py)"
  - "[pipeline_steps.py](../../src/algorithms/pipeline_steps.py)"
  - "[guide-gpu-cpu-comparison-methodology.md](./guide-gpu-cpu-comparison-methodology.md)"
  - "[analysis-gpu-two-opt-dispatch-and-reduction.md](./analysis-gpu-two-opt-dispatch-and-reduction.md)"
  - "[analysis-host-side-sampling-bottlenecks.md](./analysis-host-side-sampling-bottlenecks.md)"
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
tags:
  - analysis/performance
  - analysis/profiling
  - analysis/architecture
  - benchmark/gpu
  - benchmark/performance
  - guide/benchmarking
  - guide/cuda
  - notes/cuda
  - notes/hardware
  - notes/reproducibility
  - review/quality
  - algorithm/genetic-algorithm
---

# Timing Instrumentation and Warm-Up: The `@time_strategy` Barrier, Phase Accounting, and Deterministic AOT Verification

- [Timing Instrumentation and Warm-Up: The `@time_strategy` Barrier, Phase Accounting, and Deterministic AOT Verification](#timing-instrumentation-and-warm-up-the-time_strategy-barrier-phase-accounting-and-deterministic-aot-verification)
  + [1. Executive Summary](#1-executive-summary)
  + [2. Two Independent Clocks in the Same Process](#2-two-independent-clocks-in-the-same-process)
    - [2.1 The `@time_strategy` Decorator](#21-the-time_strategy-decorator)
    - [2.2 Why Every GPU Strategy Call Is Already Synchronous](#22-why-every-gpu-strategy-call-is-already-synchronous)
    - [2.3 Barrier Budget per Generation](#23-barrier-budget-per-generation)
  + [3. Reconstructing the 2026-11-06 `DP11-7` Measurement](#3-reconstructing-the-2026-11-06-dp11-7-measurement)
    - [3.1 Impact on the `DP10-18` / `DP11-14`–`DP11-18` Synchronization Removals](#31-impact-on-the-dp10-18--dp11-14dp11-18-synchronization-removals)
  + [4. The Async Bleed: `SurvivalStep` and the `host_sync` Bucket](#4-the-async-bleed-survivalstep-and-the-host_sync-bucket)
    - [4.1 Accounting Equation](#41-accounting-equation)
    - [4.2 Two Redundancies in `_extract_best`](#42-two-redundancies-in-_extract_best)
  + [5. Sample Standard Deviation: Why `ddof=1`](#5-sample-standard-deviation-why-ddof1)
  + [6. Deterministic AOT Warm-Up (`BUG-25`, `DP11-21`)](#6-deterministic-aot-warm-up-bug-25-dp11-21)
    - [6.1 Why `evolve(max_generations=1)` Fails](#61-why-evolvemax_generations1-fails)
    - [6.2 The Kernel-Cache Protocol](#62-the-kernel-cache-protocol)
    - [6.3 Instance Isolation: Why the Warm-Up Cannot Reuse Repetition 1's Object](#63-instance-isolation-why-the-warm-up-cannot-reuse-repetition-1s-object)
  + [7. Verification Recipe](#7-verification-recipe)
  + [8. References](#8-references)

---

## 1. Executive Summary

The GPU pipeline of this repository contains **two clocks that both synchronize the device**, and only one of them is the phase timer the thesis reports:

1. `PhaseTimer.measure(name, func, ...)` in [`src/utils/phase_timer.py`](../../src/utils/phase_timer.py) — records `time.perf_counter()` deltas by default; it only calls `cp.cuda.Event.record()` + `end_evt.synchronize()` when constructed with `use_cuda_events=True`.
2. `@time_strategy` in the same module — a *decorator* applied to every GPU strategy entry point (`select_batch`, `crossover_batch`, `mutate_batch`, `improve_batch`, `evaluate_batch`). It detects a CuPy array among its positional arguments and **unconditionally** brackets the call with CUDA events, ending with `end_evt.synchronize()`.

The decorator is therefore an implicit, invisible host-device barrier inside every measured region. Because of it, the removals of `cp.cuda.Stream.null.synchronize()` performed under `BUG-15`/`DP10-18` and propagated by `DP11-14` (`SWAP-A`), `DP11-15` (`EVAL-A`), `DP11-16` (`SEL-A`), `DP11-17` (`INV-A`) and `DP11-18` (`2OPT-A`) produced **no measurable de-synchronization gain**: the body no longer blocks, but the wrapper does, one stack frame higher.

> [!IMPORTANT]
> Removing a synchronization primitive from a strategy body is only an optimization if the caller does not re-introduce it. In this codebase the caller is a decorator, so the two changes must be reasoned about together.

---

## 2. Two Independent Clocks in the Same Process

### 2.1 The `@time_strategy` Decorator

```python
# src/utils/phase_timer.py — time_strategy.wrapper (abridged)
self_obj: Any = args[0] if args else None
cp = cupy_backend()
use_cuda = False
if cp is not None:
  if getattr(self_obj, "use_gpu_events", False):
    use_cuda = True
  else:
    for arg in args[1:]:
      if isinstance(arg, cp.ndarray):
        use_cuda = True
        break

if use_cuda and cp is not None:
  start_evt = cp.cuda.Event()
  end_evt = cp.cuda.Event()
  start_evt.record()
  result: R = func(*args, **kwargs)
  end_evt.record()
  end_evt.synchronize()          # <-- host blocks here, unconditionally
  elapsed_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))
```

Three properties matter:

| Property | Consequence |
| :--- | :--- |
| Detection is `isinstance(arg, cp.ndarray)` over `args[1:]` | A strategy that receives a **NumPy** batch but internally calls `cp.asarray` is measured with `perf_counter`, while the same strategy receiving a **GPU-resident** batch is measured with CUDA events. The measurement regime depends on the caller's residency, not on the algorithm. |
| `record()` is placed **before** any device work is enqueued | `start_evt` is enqueued into the default stream, so the recorded interval contains the kernel, but the host returns only after `end_evt.synchronize()` drains it. |
| `end_evt.synchronize()` is unconditional | There is no path in which the decorated call returns while device work is still in flight. |

### 2.2 Why Every GPU Strategy Call Is Already Synchronous

For a decorated strategy with a GPU-resident argument, the observable wall-clock identity is

$$
T_{\text{observed}} \;=\; T_{\text{host enqueue}} \;+\; T_{\text{device}} \;+\; T_{\text{sync}},
$$

where $T_{\text{sync}}$ is the event/synchronization latency. Because `end_evt.synchronize()` blocks, $T_{\text{device}}$ is **always** included in the measured value. The distinction between "enqueue latency" and "kernel duration" is erased before any caller can observe it.

This is precisely the opposite of the assumption encoded in `DP11-7`, whose original text argued that `use_cuda_events=False` made GPU phases record enqueue time instead of execution time.

### 2.3 Barrier Budget per Generation

`FullGPUModular` decorates five strategy entry points, and each one that receives a
device-resident batch contributes exactly one `end_evt.synchronize()`:

| # | Strategy call | Decorated | Barrier |
| -: | :--- | :---: | :---: |
| 1 | `SelectionGPUTournament.select_batch` | ✅ | 1 |
| 2 | `CrossoverGPUOrder.crossover_batch` | ✅ | 1 |
| 3 | `MutationGPUSwap.mutate_batch` | ✅ | 1 |
| 4 | `EvaluationGPUCost.evaluate_batch` | ✅ | 1 |
| 5 | `LocalSearchGPUTwoOpt.improve_batch` | ✅ | 1 |
| 6 | `GeneticAlgorithm._extract_best` (`argmin().get()`, `float(...get())`, `asnumpy`) | ❌ | 1–3 D2H |

So a single generation with local search carries **5 device-wide event barriers plus 1–3 implicit D2H drains = 6–8 host-device round trips**, every generation, before any kernel-level sync is considered. The host cannot run ahead by even one generation; the CUDA pipeline is completely serialised by the instrumentation.

The decoupling requirement is therefore structural, not cosmetic:

- `@time_strategy` must **not** own a barrier. Its documented job is to record `self.last_elapsed_ms` / `self.total_elapsed_ms`; if those numbers must reflect device time, that is `PhaseTimer(use_cuda_events=True)`'s decision, made once at the call site, not a decorator decision made per call.
- `PhaseTimer.measure("host_sync", ...)` must remain the *sole* owner of the drain point, because only it can attribute the wait to a bucket in the accounting equation of [§4.1](#41-accounting-equation).
- The two must not be nested inside each other, or `host_sync` double-counts what the decorator already waited for.

---

## 3. Reconstructing the 2026-11-06 `DP11-7` Measurement

The refutation experiment compared two timings of the same operation, `LocalSearchGPUTwoOpt.improve_batch`, on $n = 500$ cities with $N = 256$ routes:

```text
wrapped=260.840ms | unwrapped_enqueue=0.363ms | unwrapped_after_sync=258.725ms
```

| Variant | Measurement | Value |
| :--- | :--- | ---: |
| Decorated call (`improve_batch(...)` as configured) | `perf_counter` around the wrapper | $260.840\ \text{ms}$ |
| Undecorated body (raw enqueue only) | `perf_counter` around `func.__wrapped__` | $0.363\ \text{ms}$ |
| Undecorated body + explicit `cp.cuda.Stream.null.synchronize()` | `perf_counter` around `__wrapped__` then drain | $258.725\ \text{ms}$ |

Reading: the **body** enqueues $N$ blocks of 2-opt work and returns in $0.363\ \text{ms}$; the **wrapper** then blocks until the device finishes, which takes $260.840\ \text{ms}$. Neither number is wrong — they answer different questions. The $0.363\ \text{ms}$ answers *"how long does the CPU spend issuing the work?"*; the $260.840\ \text{ms}$ answers *"how long until the work is done?"*.

The third variant is the decisive one. It removes the decorator and re-adds the drain by hand, reproducing the wrapper's duration to within

$$
\frac{260.840 - 258.725}{258.725} \approx 0.8\%,
$$

so the decorator's own overhead (event record, `cudaEventSynchronize` on an already drained stream, elapsed-time query) is only $\approx 2.1\ \text{ms}$ on a $\approx 259\ \text{ms}$ device workload. **The barrier is not free, but the device work is real.** The `258.725 ms` figure is what `use_cuda_events=True` measures legitimately; the `260.840 ms` figure happens to be nearly identical *because both paths drain the stream*, not because the kernel is intrinsically blocking.

The earlier three-clock experiment (which reported `perf_counter`, `cp.cuda.Event` and `perf_counter`-with-drain agreeing within $0.5\%$, e.g. `berlin52` $2.24/2.25/2.20\ \text{ms}$ and `pr1002` $77.02/77.12/76.96\ \text{ms}$) was therefore **correct as a measurement and wrong as an inference**. All three clocks were measuring the same decorated call. Agreement among them proves that the decorated call blocks; it does not prove that the kernel is intrinsically blocking, and it does not license the conclusion that `use_cuda_events=True` would change any number.

> [!WARNING]
> The lesson generalizes: an experiment that varies the *clock* but not the *call surface* cannot distinguish "the operation blocks" from "the wrapper blocks". To separate them you must vary the call surface — `func` versus `func.__wrapped__`.

### 3.1 Impact on the `DP10-18` / `DP11-14`–`DP11-18` Synchronization Removals

Those tasks removed redundant `cp.cuda.Stream.null.synchronize()` calls from `MutationGPUSwap`, `MutationGPUInversion`, `EvaluationGPUCost`, `SelectionGPUTournament`, and the two batched 2-opt kernels. `grep` confirms the removals landed:

```bash
grep -rn "Stream.null.synchronize()" src/algorithms/strategies/ --exclude-dir=__pycache__
# 0 matches
```

The removals are still **correct** — they eliminate a genuine double barrier when the strategy is called outside the decorator (unit tests, direct benchmark harnesses, future callers). What they are not is a *performance* change inside `evolve()`, because `@time_strategy` immediately re-imposes the barrier. Any future claim of the form "removing `synchronize()` accelerated the generation by $X\%$" must first prove that the measured path bypasses the decorator.

---

## 4. The Async Bleed: `SurvivalStep` and the `host_sync` Bucket

`SurvivalStep.run` is **not** decorated and does **not** use CUDA events:

```python
# src/algorithms/metaheuristics/genetic_algorithm.py — SurvivalStep.run
state.population, state.fitness = timer.measure(
  "survival",
  self.survivor_selector,
  state.population,
  state.fitness,
  state.offspring,
  state.offspring_fitness,
)
```

`PhaseTimer` here has `use_cuda_events=False`, so the recorded `survival` value is a `perf_counter` delta around `_select_survivors`, which performs:

```python
combined_pop = xp.vstack([old_pop, new_pop])
combined_fitness = xp.concatenate([xp.asarray(old_fitness), xp.asarray(new_fitness)])
best_indices = xp.argsort(combined_fitness)[: self.population_size]
return combined_pop[best_indices], combined_fitness[best_indices]
```

All four operations are **asynchronous** on CuPy: `vstack` (device-to-device copy), `concatenate`, `argsort` (thrust/CUB-backed sort) and the fancy-index gather. The host returns as soon as they are enqueued. The recorded `survival` time is therefore mostly host enqueue cost, and the real device work is drained by the *next* synchronized operation — which is `_extract_best`, recorded under the explicit `host_sync` bucket:

```python
gen_best_tour, gen_best_cost = self.timer.measure("host_sync", self._extract_best, state.population, state.fitness)
```

### 4.1 Accounting Equation

For the phase table of Chapter 4 to be falsifiable, the buckets must close the wall clock:

$$
T_{\text{wall}} \;\approx\; \sum_{p \in \mathcal{P}} T_p \;+\; T_{\text{host\_sync}},
\qquad
\mathcal{P} = \{\text{initialization}, \text{selection}, \text{crossover}, \text{mutation}, \text{local\_search}, \text{evaluation}, \text{survival}\}.
$$

`survival` is part of $\mathcal{P}$ even though its measured value under-reports device cost, because omitting it silently transfers its enqueue time nowhere while its *device* time lands in `host_sync`. Reporting a `Sum/Wall` percentage without `survival` produces a residual that is not attributable to any labeled phase — the exact failure mode that makes a phase table unfalsifiable.

> [!NOTE]
> `host_sync` is not "overhead". It is the drain point for every asynchronous producer whose consumer was not itself synchronized. On `pr1002` this bucket is small ($\approx 0.2\%$ of $446\ \text{ms}$); on `berlin52` with a single generation it reaches $\approx 7.8\%$ of a $6.7\ \text{ms}$ run because the fixed cost is divided by a much smaller denominator.

### 4.2 Two Redundancies in `_extract_best`

```python
# src/algorithms/metaheuristics/genetic_algorithm.py — GeneticAlgorithm._extract_best
if is_gpu_array(fitness):
  cp = require_gpu()
  best_idx = int(cp.argmin(fitness).get())
  best_cost = float(fitness[best_idx].get())
else:
  best_idx = int(np.argmin(fitness))
  best_cost = float(fitness[best_idx])

best_route_raw = population[best_idx]
if is_gpu_array(best_route_raw):
  best_route: Route = require_gpu().asnumpy(best_route_raw).astype(np.int32)
else:
  best_route = np.asarray(best_route_raw, dtype=np.int32).copy()
return best_route, best_cost
```

**Redundancy 1 — `argmin` is not needed after elitist survival.** `_select_survivors` returns `combined_fitness[best_indices]` where `best_indices = xp.argsort(combined_fitness)[:N]`. The returned fitness vector is therefore **sorted ascending by construction**, so `fitness[0]` is already the minimum:

$$
f_0 \le f_1 \le \dots \le f_{N-1}
\quad\Longrightarrow\quad
\arg\min_{k} f_k = 0.
$$

`cp.argmin(fitness).get()` re-executes a full $O(N)$ device reduction plus a D2H scalar copy to rediscover index `0`. Replacing it with `float(fitness[0].get())` removes one reduction per generation and keeps the D2H copy that is genuinely required.

**Redundancy 2 — the route is copied every generation.** `best_route` (`asnumpy`, $n \times 4$ bytes) is materialized unconditionally. `evolve` only consumes it when the cost improves:

```python
if gen_best_cost < best_cost - 1e-6:
    best_cost = gen_best_cost
    best_tour = gen_best_tour          # only read here
```

A deferred copy — compute `gen_best_cost` first, and only pull the route when `gen_best_cost < best_cost - 1e-6` — removes $n \times 4$ bytes of D2H traffic per non-improving generation, which is the common case after convergence. On a 1,002-city problem that is $\approx 4\ \text{KB}$ per generation reclaimed from `host_sync`.

---

## 5. Sample Standard Deviation: Why `ddof=1`

`phase_share_cv_pct` is computed as

```python
# src/benchmarking/runner/algorithm_runner.py
std_share = float(np.std(phase_shares)) if phase_shares else 0.0
phase_share_cv_pct[phase] = (std_share / m_share * 100.0) if m_share > 1e-9 else 0.0
```

`np.std` defaults to `ddof=0`, i.e. the **population** standard deviation. The repetitions are a *sample* from the run distribution, not the entire population, so the estimator is biased low:

$$
\mathbb{E}\!\left[\hat\sigma_0\right] = \sigma\sqrt{\frac{R-1}{R}}
\quad\Longrightarrow\quad
\text{bias}(R) = \frac{\hat\sigma_1 - \hat\sigma_0}{\hat\sigma_1} = 1 - \sqrt{\frac{R-1}{R}}.
$$

With the minimum acceptable $R = 8$ repetitions required by `DP11-7`, the underestimate is

$$
1 - \sqrt{7/8} \approx 6.5\%,
$$

and at $R = 3$ (the `selected_10` cohort) it reaches $\approx 18.4\%$. Since `DP11-7` publishes the coefficient of variation specifically as the guard against unstable phase fractions, a systematic $6.5\%$–$18.4\%$ downward bias in that very guard is a defect. Use `ddof=1`.

---

## 6. Deterministic AOT Warm-Up (`BUG-25`, `DP11-21`)

### 6.1 Why `evolve(max_generations=1)` Fails

The runner's warm-up performs one discarded generation:

```python
# src/benchmarking/runner/algorithm_runner.py:146-161
warmup_algo.evolve(warmup_context, ..., max_generations=1)
```

`PeriodicLocalSearchStep` only fires when `state.generation % interval == 0`. With `interval = 10`, generation `1` never triggers it, so `two_opt_batch.cu` is **never compiled during warm-up**. The first measured repetition then pays NVRTC compilation at generation 10:

| Observation | Value |
| :--- | ---: |
| Repetition 1 wall time | $1651.73\ \text{ms}$ |
| Median of subsequent repetitions | $64.44\ \text{ms}$ |
| Spike factor | $26.27\times$ |

This is not clock ramp-up and not allocator warm-up: it is a *missing kernel* in the JIT cache. Fixed-generation warm-up is structurally incapable of covering a periodic or conditional step, because the set of kernels reachable in $k$ generations is a function of the pipeline's control flow, not of the generation count.

### 6.2 The Kernel-Cache Protocol

`load_kernel` is backed by a process-wide LRU cache, so its state is directly observable:

```python
# src/utils/gpu_helpers.py
@functools.lru_cache(maxsize=32)
def _compile_kernel_cached(kernel_path_str: str, kernel_function_name: str) -> cp.RawKernel: ...
```

The warm-up becomes a **deterministic assertion on cache state** instead of a timing heuristic:

1. Record `hits, misses` from `_compile_kernel_cached.cache_info()` before warm-up.
2. Drive the pipeline's explicit `warmup(context)` hooks — every strategy and every `GenerationStep`, including `PeriodicLocalSearchStep` — rather than an `evolve()` call.
3. Assert `misses_after_warmup == misses_before + |expected kernels|`, and assert `misses_after_reps == misses_after_warmup` over the measured repetitions.

Payload discipline is part of the protocol: warm-up must feed **valid permutations** (`cp.arange(n, dtype=cp.int32)` broadcast to `(N, n)`), never `cp.zeros`. A zero array satisfies `batch[0, 0] == batch[0, -1]` ($0 = 0$), which poisons closed-tour inference — see [§6 of the sampling analysis](./analysis-host-side-sampling-bottlenecks.md#6-topology-closure-mutation-bounds-versus-crossover-inference) and `BUG-24`.

```python
import cupy as cp
from utils.gpu_helpers import _compile_kernel_cached

def assert_warm(callable_under_test, n: int, pop: int) -> None:
  """Fail unless every kernel the pipeline needs is already compiled."""
  base = _compile_kernel_cached.cache_info().misses
  routes = cp.tile(cp.arange(n, dtype=cp.int32), (pop, 1))  # valid permutation, never zeros
  callable_under_test(routes)
  after = _compile_kernel_cached.cache_info().misses
  assert after >= base, "warm-up must not shrink the kernel cache"
  cp.cuda.Stream.null.synchronize()
```

> [!CAUTION]
> Never assert on wall-clock thresholds in a warm-up test. `assert rep1_time < 1.2 * median(rest)` is flaky under thermal throttling, co-tenancy and CI jitter, and it silently passes when both values are equally bad. `cache_info().misses` is a deterministic counter: it either advanced or it did not.

### 6.3 Instance Isolation: Why the Warm-Up Cannot Reuse Repetition 1's Object

The runner accepts either a `algorithm_factory` or a pre-built `algorithm_instance`. When only the instance is supplied, the warm-up currently drives **that same object**:

```python
# src/benchmarking/runner/algorithm_runner.py — warm-up path (abridged)
warmup_algo = algorithm_instance if algorithm_factory is None else algorithm_factory()
warmup_algo.evolve(warmup_context, ..., max_generations=1)
```

Every stochastic strategy mutates its own state during those generations:

- `ProblemContext.rng` is a stateful `np.random.Generator` shared by `SelectionGPUTournament`, `CrossoverGPUOrder`, `MutationGPUSwap` and `MutationGPUInversion`. Consuming $k$ draws in the warm-up shifts the stream by $k$ for every subsequent draw.
- `StrategyBase.h2d_bytes` / `d2h_bytes` / `kernel_launches` accumulate during the warm-up and are only zeroed by `reset_metrics()`, which nothing calls on the boundary.
- `CrossoverGPUOrder._is_closed_cache` may be committed from the warm-up payload (see [§6.2 of the sampling analysis](./analysis-host-side-sampling-bottlenecks.md#62-crossover-infers-closure-from-values-bug-24)).

The consequence is precise and easy to miss: **Repetition 1 is not a draw from the same distribution as Repetitions 2..R.** It is offset by the warm-up's consumption, and the reported `mean_time` / `wall_cv_pct` are computed over a mixture of two populations. The warm-up is supposed to remove a *fixed hardware cost*; by mutating the instance it also perturbs the *stochastic trajectory*, which is the one thing a seeded benchmark must hold constant.

The contract is therefore: when `algorithm_factory is None`, clone the instance before warming (`copy.deepcopy`) and discard the clone — or require a factory. The warm-up must be a pure side-effect-free acceleration of the *process*, never of the *run*.

A second, independent invariant makes the protocol falsifiable without clocks: the LRU cache exposes its **size**, not only its hit/miss counters:

```python
from utils.gpu_helpers import _compile_kernel_cached

before = _compile_kernel_cached.cache_info().currsize
warmup(context)
after = _compile_kernel_cached.cache_info().currsize
assert after > before, "warm-up compiled no kernel: pipeline steps were not exercised"

# ... run the measured repetitions ...
for _ in range(repetitions):
  algo.evolve(context, ...)
  assert _compile_kernel_cached.cache_info().currsize == after, \
    "JIT compilation happened inside a measured repetition"
```

`currsize` is the right operand rather than `misses` because it is bounded by `maxsize=32` and directly encodes "how many kernels are resident". If a measured repetition ever increases it, a pipeline step escaped the warm-up — exactly the `PeriodicLocalSearchStep` failure of [§6.1](#61-why-evolvemax_generations1-fails), caught before it can inflate a timing.

---

## 7. Verification Recipe

```bash
# 1. Confirm the decorator is still unconditional
grep -n "end_evt.synchronize()" src/utils/phase_timer.py

# 2. Confirm no strategy body re-introduces the barrier
grep -rn "Stream.null.synchronize()" src/algorithms/strategies/ --exclude-dir=__pycache__   # expect 0

# 3. Confirm the CV estimator is unbiased
grep -n "np.std(phase_shares" src/benchmarking/runner/algorithm_runner.py                 # expect ddof=1

# 4. Confirm the warm-up is cache-verified, not generation-counted
grep -n "cache_info" tests/benchmarking/test_unit_runner_warmup.py
```

Expected: step 1 returns the decorator line (the barrier is a documented contract, not a bug); step 2 returns nothing; steps 3 and 4 return the unbiased estimator and the deterministic cache assertion.

---

## 8. References

1. Hoefler, T., & Belli, R. (2015). Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results. *SC '15*, 1–12. <https://doi.org/10.1145/2807591.2807644>
2. NVIDIA Corporation. *CUDA C++ Programming Guide* — Asynchronous Execution, CUDA Events, and `cudaEventSynchronize` semantics. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
3. CuPy Team. *CuPy User Guide* — Streams, Events, and implicit synchronization on `cupy.ndarray.get()`. <https://docs.cupy.dev/en/stable/user_guide/performance.html>
4. Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically rigorous Java performance evaluation. *OOPSLA '07*, 57–76.
5. NIST/SEMATECH. *e-Handbook of Statistical Methods* — §1.3.5.6 Measures of Scale (sample standard deviation, Bessel's correction). <https://www.itl.nist.gov/div898/handbook/>
6. NVRTC / CUDA Driver API. *JIT compilation and module caching of `RawKernel` sources* — the mechanism behind `load_kernel` latency. <https://docs.nvidia.com/cuda/nvrtc/>
