---
title: "Host-Side Sampling Bottlenecks and Route Topology Contracts in the GPU Genetic Algorithm"
description: >
  Analysis of the host RNG serialization that dominates crossover and mutation wall
  time in the modular Genetic Algorithm: why `rng.choice(np.arange(low, high), size=2,
  replace=False)` inside a per-route Python loop costs $\mathcal{O}(N \cdot n)$ per
  generation, the exact $\mathcal{O}(1)$-per-pair vectorized formula for sampling two
  distinct integers from $[a, b)$, why vectorizing only the GPU variant breaks the
  `OX-A`/`SWAP-A`/`INV-A` bit-identity invariants, and why the topology contract must
  move from heuristic inference (`_has_depot_endpoints`, `_is_closed_cache`) to a
  canonical `ProblemContext` property (`BUG-24`, `DP11-19`, `DP11-20`).
created: 2026-11-10
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[GitHub Copilot]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Initial creation: derived the cost of `rng.choice(np.arange(...), size=2,
          replace=False)` per route, proved the vectorized rejection-free formula for two
          distinct integers, documented why a GPU-only vectorization breaks the
          `OX-A`/`SWAP-A`/`INV-A` bit-identity contracts and why the whole per-generation
          draw (including `rng.random()`) must move together, and reconciled
          `_mutation_bounds(n) -> (1, n - 1)` against `_is_closed_cache` poisoning and the
          canonical `ProblemContext` topology property.
      - description: >
          Revision: added the search-space cardinality framing for the frozen-endpoint
          defect ($n! \to (n-2)!$ reachable permutations, a factor of $n(n-1) = 2{,}652$
          configurations at $n = 52$), the verbatim `_mutation_bounds` docstring with its
          `route_helpers.py:31-39` anchor, and the `order_crossover.py:227-233` anchor for
          the dimension-keyed `_is_closed_cache` probe.
related_files:
  - "[order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py)"
  - "[inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py)"
  - "[route_helpers.py](../../src/utils/route_helpers.py)"
  - "[rng.py](../../src/utils/rng.py)"
  - "[guide-iso-algorithmic-ga-validation.md](./guide-iso-algorithmic-ga-validation.md)"
  - "[analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[analysis-timing-instrumentation-and-warmup.md](./analysis-timing-instrumentation-and-warmup.md)"
  - "[guide-gpu-cpu-comparison-methodology.md](./guide-gpu-cpu-comparison-methodology.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
tags:
  - analysis/performance
  - analysis/architecture
  - analysis/scaling
  - benchmark/gpu
  - benchmark/performance
  - algorithm/crossover
  - algorithm/mutation
  - algorithm/genetic-algorithm
  - guide/benchmarking
  - notes/reproducibility
  - review/quality
---

# Host-Side Sampling Bottlenecks and Route Topology Contracts in the GPU Genetic Algorithm

- [Host-Side Sampling Bottlenecks and Route Topology Contracts in the GPU Genetic Algorithm](#host-side-sampling-bottlenecks-and-route-topology-contracts-in-the-gpu-genetic-algorithm)
  + [1. Executive Summary](#1-executive-summary)
  + [2. The Three Sampling Sites](#2-the-three-sampling-sites)
  + [3. Why `rng.choice(np.arange(...), size=2, replace=False)` Is Expensive](#3-why-rngchoicenparange-size2-replacefalse-is-expensive)
  + [4. The Vectorized Rejection-Free Formula](#4-the-vectorized-rejection-free-formula)
    - [4.1 Derivation and Correctness Proof](#41-derivation-and-correctness-proof)
    - [4.2 Implementation](#42-implementation)
  + [5. Why Vectorizing Only the GPU Variant Breaks Bit-Identity](#5-why-vectorizing-only-the-gpu-variant-breaks-bit-identity)
    - [5.1 The Invariants at Stake](#51-the-invariants-at-stake)
    - [5.2 The Interleaved-Stream Hazard](#52-the-interleaved-stream-hazard)
    - [5.3 The Shared-Sampler Resolution](#53-the-shared-sampler-resolution)
  + [6. Topology Closure: Mutation Bounds versus Crossover Inference](#6-topology-closure-mutation-bounds-versus-crossover-inference)
    - [6.1 `_mutation_bounds` Freezes Endpoints Unconditionally](#61-_mutation_bounds-freezes-endpoints-unconditionally)
    - [6.2 Crossover Infers Closure from Values (`BUG-24`)](#62-crossover-infers-closure-from-values-bug-24)
    - [6.3 The Canonical Property (`DP11-19`)](#63-the-canonical-property-dp11-19)
  + [7. Verification Recipe](#7-verification-recipe)
  + [8. References](#8-references)

---

## 1. Executive Summary

Crossover and mutation in this codebase generate their stochastic parameters **on the host, one route at a time**. The device kernels are fast — `order_crossover.cu` executes in $0.012$–$0.088\ \text{ms}$ for $N = 256$ routes — while the python loop that feeds them consumes $\approx 1.22\ \text{ms}$, i.e. $>90\%$ of the strategy's wall time and $14\times$ the kernel duration. Amdahl's law makes the kernel optimization irrelevant until the sampler is vectorized.

The measured floor for the vectorized form is $0.0168\ \text{ms}$ — a $72\times$ reduction — but the fix cannot be applied to one class in isolation without destroying the CPU/GPU bit-identity contracts (`OX-A`, `SWAP-A`, `INV-A`) that make the Chapter 4 comparison *iso-algorithmic* rather than *"two different algorithms that both happen to solve TSP"*.

Finally, route *topology* (open tour vs. closed depot route) is currently inferred in two incompatible ways, and one of them caches the inference by dimension alone, which is poisonable by a warm-up payload (`BUG-24`).

---

## 2. The Three Sampling Sites

| Site | Loop | Draw |
| :--- | :--- | :--- |
| [`CrossoverOrder.crossover_batch`](../../src/algorithms/strategies/crossover/order_crossover.py) | `for i in range(0, num_routes - 1, 2)` | `rng.choice(np.arange(1, n - 1), size=2, replace=False)` when closed, else `rng.choice(n, size=2, replace=False)` |
| [`CrossoverGPUOrder.crossover_batch`](../../src/algorithms/strategies/crossover/order_crossover.py) | `for i in range(0, num_routes - 1, 2)` | identical draw, then scatter into `cut1_np` / `cut2_np` |
| [`MutationSwap.mutate_batch`](../../src/algorithms/strategies/mutation/swap_mutation.py) / [`MutationGPUSwap.mutate_batch`](../../src/algorithms/strategies/mutation/swap_mutation.py) | `for i in range(num_routes)` | `p = rng.random()`, then **if** `p < mutation_rate`: `rng.choice(np.arange(low, high), size=2, replace=False)` |
| [`MutationInversion`](../../src/algorithms/strategies/mutation/inversion_mutation.py) / [`MutationGPUInversion`](../../src/algorithms/strategies/mutation/inversion_mutation.py) | `for i in range(num_routes)` | `rng.integers(low, high)` plus the same `rng.choice(..., size=2, replace=False)` pair |

The crossover loop iterates $N/2$ times; the mutation loop iterates $N$ times and conditionally draws. All of them live on the host, and all of them run identically for the CPU and the GPU variant — which is *why* the two variants are bit-identical in the first place.

---

## 3. Why `rng.choice(np.arange(...), size=2, replace=False)` Is Expensive

Three cost components per call, none of them $O(1)$:

1. **`np.arange(low, high)` allocates a fresh `int64` array of length $L = \text{high} - \text{low}$.** The result is discarded immediately after `choice` returns. For $n = 1{,}002$ this is an $8\ \text{KB}$ allocation, $N/2 = 128$ times per generation.
2. **`Generator.choice(..., replace=False)` is not a constant-time two-sample.** NumPy's implementation of sampling without replacement builds a permutation of the whole population (via a Fisher–Yates shuffle) and returns its first `size` entries. Its cost is $\Theta(L)$, not $\Theta(\text{size})$.
3. **python-level loop overhead** — up to three `int(...)` casts, two array index assignments, and a branch per route.

Per generation the aggregate is therefore

$$
T_{\text{host}} \;\approx\; m \cdot \bigl(\underbrace{c_{\text{arange}}\, L}_{\text{allocation}} \;+\; \underbrace{c_{\text{choice}}\, L}_{\Theta(L)\ \text{shuffle}} \;+\; c_{\text{loop}}\bigr),
\qquad
m = \begin{cases} N/2 & \text{crossover} \\ N & \text{mutation} \end{cases}
$$

which is $\mathcal{O}(N \cdot n)$ — a second full pass over the population matrix, in python, every generation, on data whose entire purpose is to produce $N$ pairs of small integers.

> [!IMPORTANT]
> The asymptotic class is the same as the population matrix itself. Vectorizing the *kernel* while leaving this loop serial caps the achievable speedup at whatever fraction the sampler does not own — here, under $10\%$. This is the textbook Amdahl's-law ceiling and it is the reason `BUG-24`/`OX-B` treat the sampler as a first-class bottleneck rather than a micro-optimization.

Measured ($N = 256$):

| Variant | Host sampling time |
| :--- | ---: |
| python loop with `rng.choice` | $1.217\ \text{ms}$ |
| Vectorized NumPy (`rng.integers` + comparison) | $0.0168\ \text{ms}$ |
| Speedup | $72\times$ |

---

## 4. The Vectorized Rejection-Free Formula

Goal: draw $m$ independent ordered pairs $(u_i, v_i)$ with $u_i \ne v_i$, both uniform over $\{a, \dots, b-1\}$, using only two vectorized `integers` calls and no rejection loop.

### 4.1 Derivation and Correctness Proof

Let $L = b - a \ge 2$ and fix a pair $(u, v)$ with $u \ne v$. Draw

$$
u \sim \mathcal{U}\{a, \dots, b-1\},
\qquad
v' \sim \mathcal{U}\{a, \dots, b-2\},
\qquad
v = v' + \mathbb{I}\left[v' \ge u\right].
$$

Define the map $\varphi_u : \{a, \dots, b-2\} \to \{a, \dots, b-1\} \setminus \{u\}$ by $\varphi_u(x) = x + \mathbb{I}[x \ge u]$.

- **Injectivity.** If $x < y$ then $\varphi_u(x) \le \varphi_u(y)$, with equality only if $x = u - 1$ and $y = u$. But $x = u - 1 \Rightarrow \varphi_u(x) = u - 1 \ne u = \varphi_u(y)$ when $y = u$. Hence $\varphi_u$ is injective.
- **Surjectivity.** $|\{a, \dots, b-2\}| = L - 1 = |\{a, \dots, b-1\} \setminus \{u\}|$, and an injective map between finite sets of equal cardinality is bijective.

Therefore $v$ is uniform over the $L - 1$ values other than $u$, and $u$ is uniform over the $L$ values, so every ordered pair with $u \ne v$ has probability

$$
\Pr(u, v) = \frac{1}{L} \cdot \frac{1}{L-1},
$$

which is exactly the distribution of `rng.choice(np.arange(a, b), size=2, replace=False)` — **the same distribution, in $\mathcal{O}(1)$ per pair, with zero rejections** (contrast with rejection sampling, whose expected number of draws per pair is $L/(L-1) \to 1$ but whose *worst case* is unbounded, which is unacceptable inside a timed generation).

The final relation $v = \min(u, v) + \mathbb{I}[\cdot]$ used in practice is the same construction with the pair sorted, matching `c1, c2 = int(min(cuts)), int(max(cuts))`.

### 4.2 Implementation

```python
import numpy as np
from numpy.random import Generator


def sample_distinct_pairs(rng: Generator, num_pairs: int, low: int, high: int) -> np.ndarray:
  """Draw `(num_pairs, 2)` uniformly among distinct integer pairs in `[low, high)`.

  Uses the bijection `v = v' + I[v' >= u]` from `[low, high - 1]` onto
  `[low, high) \\ {u}`, so the result is uniform over ordered distinct pairs
  without a rejection loop. Requires `high - low >= 2`.

  Returns:
      `int32` array of shape `(num_pairs, 2)` in `[low, high)`.
  """
  if high - low < 2:
    raise ValueError(f"need at least two positions, got [{low}, {high})")
  first = rng.integers(low, high, size=num_pairs)
  second = rng.integers(low, high - 1, size=num_pairs)
  second += (second >= first).astype(np.int32)
  return np.stack([first, second], axis=1).astype(np.int32, copy=False)
```

---

## 5. Why Vectorizing Only the GPU Variant Breaks Bit-Identity

### 5.1 The Invariants at Stake

| Identifier | Contract |
| :--- | :--- |
| `OX-A` | For the same parents and the same cut points, the offspring of `CrossoverGPUOrder` is **bit-identical** to `CrossoverOrder` |
| `SWAP-A` | For the same seed, the routes mutated by `MutationGPUSwap` are **bit-identical** to `MutationSwap` |
| `INV-A` | For the same seed, the routes mutated by `MutationGPUInversion` are **bit-identical** to `MutationInversion` |

`DP11-20` states the resolution precisely: a **shared** sampler (`sample_distinct_pairs`) must be consumed by both the CPU and the GPU class of each pair. If only `CrossoverGPUOrder` is vectorized, its `cut1_np`/`cut2_np` tensors are drawn from a different sequence than `CrossoverOrder`'s, so

$$
\{c_1^{(g)}, c_2^{(g)}\}_{\text{GPU}} \neq \{c_1^{(g)}, c_2^{(g)}\}_{\text{CPU}}
\quad\Longrightarrow\quad
\text{offspring}_{\text{GPU}} \neq \text{offspring}_{\text{CPU}},
$$

and the entire premise of the Chapter 4 speedup measurement — *the GPU runs the same algorithm faster* — collapses. What the benchmark would then be comparing is two different genetic searches, and any quality difference between them (`best_fitness`, `gap_percent`) is uninterpretable.

> [!WARNING]
> Vectorization is a **distribution-preserving but stream-changing** transformation. `rng.integers` and `rng.choice` both yield uniform pairs, but they consume the generator's bit stream differently. Distributional equivalence is not algorithmic equivalence. The invariant must be enforced by construction — one helper, both call sites — not by hoping the draws coincide.

### 5.2 The Interleaved-Stream Hazard

In the mutation classes the pair draws are **interleaved** with an unconditional draw:

```python
for i in range(num_routes):
  p = float(rng.random())                       # always consumed
  probs_np[i] = p
  if p < mutation_rate:                         # conditional
    first, second = rng.choice(np.arange(low, high), size=2, replace=False)
```

The number of `choice` calls depends on how many `p < mutation_rate` routes appear, which depends on the values of `p`, which depend on the stream position. Vectorizing only the conditional branch changes not just the pair values but the *positions at which subsequent `rng.random()` calls read*. The correct transformation vectorizes the whole per-generation draw:

```python
probs_np = rng.random(num_routes)                                # one call
mask = probs_np < mutation_rate
pairs = sample_distinct_pairs(rng, int(mask.sum()), low, high)   # one call
pos1_np = np.zeros(num_routes, dtype=np.int32)
pos2_np = np.zeros(num_routes, dtype=np.int32)
pos1_np[mask] = pairs[:, 0]
pos2_np[mask] = pairs[:, 1]
```

This still changes the stream relative to the loop version, which is why step (3) of `DP11-20` is: **convert CPU and GPU together, in the same change, and re-baseline the checkpoint cohort**. A partially converted pair of classes is worse than no conversion at all.

### 5.3 The Shared-Sampler Resolution

1. Put `sample_distinct_pairs(rng, num_pairs, low, high)` in a single module (`utils/route_helpers.py` or `utils/rng.py`).
2. Call it from `CrossoverOrder`, `CrossoverGPUOrder`, `MutationSwap`, `MutationGPUSwap`, `MutationInversion` and `MutationGPUInversion` — six call sites, one implementation.
3. Re-run the parity tests that pin `OX-A`, `SWAP-A` and `INV-A`, and only then re-generate the result cohort.

---

## 6. Topology Closure: Mutation Bounds versus Crossover Inference

### 6.1 `_mutation_bounds` Freezes Endpoints Unconditionally

```python
# src/utils/route_helpers.py:31-39
def _mutation_bounds(n: int) -> tuple[int, int]:
  """Return the mutable index slice `[low, high)` for a route of length `n`.

  Endpoints (index `0` and `n - 1`) are never mutated, so depot delimiters stay
  fixed. For `n <= 3` the slice is empty (`high - low < 2`), making mutation a
  safe no-op when there is no interior position to swap or invert.
  """
  return 1, n - 1
```

The docstring justifies this as preserving depot delimiters. It is correct for a **closed** route `[depot, ..., depot]`, where index `0` and `n - 1` are the same depot. It is wrong for an **open** TSP tour, which is a permutation of $\{0, \dots, n-1\}$ with no repeated element: index `0` and index $n - 1$ are ordinary cities.

Freezing two of $n$ positions does not merely remove a few pairs — it removes two degrees of freedom from the genotype, and the reachable set stops being a group acting transitively on the tour:

$$
\underbrace{n!}_{\text{reachable as }S_n}\;\longrightarrow\;\underbrace{(n-2)!}_{\text{only the frozen interior}}
$$

because every permutation reachable through swap/inversion on the slice $[1, n-1)$ is a permutation of the *remaining* $n-2$ cities with positions `0` and `n-1` pinned. For $n = 52$ that is a factor of $52 \cdot 51 = 2{,}652$ fewer configurations, and the operator bias is structural rather than statistical:

- shrinks the reachable neighbourhood from $\binom{n}{2}$ to $\binom{n-2}{2} + (n - 2)$ pairs (a loss of $2n - 3$ of the $\binom{n}{2}$ pairs, i.e. $4/n \approx 7.7\%$ at $n = 52$), and
- biases the operator, because cities at the tour extremities are systematically less mutable than interior cities.

The loss is small per generation but systematic: it is not noise, it does not average out over repetitions, and it differs from the topology the crossover operator assumes for the same problem.

### 6.2 Crossover Infers Closure from Values (`BUG-24`)

Two different inference mechanisms coexist:

```python
# CrossoverOrder.crossover: closure requires BOTH parents to look closed
is_closed = _has_depot_endpoints(p1) and _has_depot_endpoints(p2)
```

```python
# CrossoverGPUOrder.crossover_batch — src/algorithms/strategies/crossover/order_crossover.py:227-233
if self._is_closed_cache is not None and self._is_closed_cache[0] == n_cities:
  is_closed = self._is_closed_cache[1]
else:
  first_0 = int(d_parents[0, 0].get()) if input_on_gpu else int(batch[0, 0])
  first_end = int(d_parents[0, -1].get()) if input_on_gpu else int(batch[0, -1])
  is_closed = bool(n_cities > 2 and first_0 == first_end)
  self._is_closed_cache = (n_cities, is_closed)
```

Two defects compound:

1. **The cache key is $n_{\text{cities}}$ alone.** Two problems of equal dimension but different topology (an open tour of 52 cities and a closed 52-stop route) map to the same key, so whichever ran first decides for both.
2. **The inference is poisonable.** Any batch whose first and last entries coincide reads as closed. A warm-up payload of `cp.zeros((N, n))` satisfies $0 = 0$, commits `(n, True)`, and every subsequent open-tour run of that dimension silently freezes cut boundaries and can duplicate cities. Empirically: `E1 cache after zeros warm-up: (52, True)`, `GPU==CPU offspring after poisoned warm-up: False`.

`reset_metrics()` clears the cache, but nothing forces a reset between a warm-up and a measured repetition, and `reset_metrics()` is a *bookkeeping* method — making correctness depend on calling it is a hidden ordering contract.

### 6.3 The Canonical Property (`DP11-19`)

Closure is a property of the **problem formulation**, not of the current array contents. It belongs next to `problem_type` and `dimension`:

- `ProblemContext.is_closed` (or `Problem.is_closed`), derived once from `problem.problem_type in ("CVRP", "TSPTW")` or from an explicit depot field written by the loader.
- Zero D2H transfers: no `d_parents[0, 0].get()` probe.
- Zero heuristic: a value comparison can no longer decide an algorithmic branch.
- Zero cache invalidation: an immutable per-problem property cannot be poisoned by a warm-up payload of any shape.
- `_mutation_bounds` becomes `resolve_route_bounds(n, context, route_sample)`, so mutation and crossover consult **one** topology source; open tours expose `[0, n)` to mutation, closed routes expose `[1, n - 1)`.

---

## 7. Verification Recipe

```bash
# 1. One sampler, six call sites
grep -rn "sample_distinct_pairs" src/algorithms/strategies/ | wc -l          # expect 6

# 2. No per-route choice() loop left in production code
grep -rn "rng.choice(np.arange" src/algorithms/strategies/ --exclude-dir=__pycache__   # expect 0

# 3. Topology comes from the problem, not from array contents
grep -rn "_is_closed_cache" src/ --exclude-dir=__pycache__                   # expect 0
grep -rn "context.is_closed" src/                                            # expect >= 4
```

---

## 8. References

1. Davis, L. (1985). Applying Adaptive Algorithms to Epistatic Domains. *IJCAI*, 162–164. (Origin of Order Crossover.)
2. Oliver, I. M., Smith, D. J., & Holland, J. R. C. (1987). A Study of Permutation Crossover Operators on the Traveling Salesman Problem. *ICGA*, 224–230.
3. Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms* (3rd ed.), §3.4.2 — random sampling and selection without replacement. Addison-Wesley.
4. NumPy Developers. *Random Generator — `choice`, `integers`, `permutation`*; implementation notes on `replace=False` shuffling. <https://numpy.org/doc/stable/reference/random/generator.html>
5. Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Spring Joint Computer Conference*, 483–485.
6. Reinelt, G. (1991). TSPLIB — A Traveling Salesman Problem Library. *ORSA Journal on Computing*, 3(4), 376–384.
