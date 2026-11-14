---
title: "Exhaustive Multi-Axis Review: Schemas, Operator Mechanics, Benchmarking Architecture, and Testing Rigor"
description: >
  In-depth architectural and code quality review across three batches: (1) JSON schema
  integrity and operator mechanics in cost evaluation, Davis Order Crossover (OX1), and
  inversion mutation; (2) benchmarking architecture, DRY audit between orchestration and
  report generator, and Fleming & Wallace (1986) dynamic baseline resolution; (3) runner
  telemetry verbosity reduction and behavioral/property-based integration testing quality.
created: 2026-11-12
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: review
scope: global
modifications:
  - date_modified: 2026-11-12
    modifications:
      - description: >
          Conducted comprehensive multi-axis review covering JSON schema strictness,
          vectorized cost evaluation, OX1 indicator array mechanics, mutation memory aliasing,
          report generator coupling, DRY architectural restructuring, and property-based testing.
related_files:
  - "[algorithms.schema.json](../../../configs/schemas/algorithms.schema.json)"
  - "[problems.schema.json](../../../configs/schemas/problems.schema.json)"
  - "[cost_evaluation.py](../../../src/algorithms/strategies/evaluation/cost_evaluation.py)"
  - "[order_crossover.py](../../../src/algorithms/strategies/crossover/order_crossover.py)"
  - "[inversion_mutation.py](../../../src/algorithms/strategies/mutation/inversion_mutation.py)"
  - "[swap_mutation.py](../../../src/algorithms/strategies/mutation/swap_mutation.py)"
  - "[report_generator.py](../../../src/benchmarking/analysis/report_generator.py)"
  - "[orchestration.py](../../../src/benchmarking/runner/orchestration.py)"
  - "[algorithm_runner.py](../../../src/benchmarking/runner/algorithm_runner.py)"
  - "[test_integration_pipeline.py](../../../tests/benchmarking/test_integration_pipeline.py)"
  - "[analysis-benchmarking-architecture-checkpoints-logging-reporting.md](./analysis-benchmarking-architecture-checkpoints-logging-reporting.md)"
tags:
  - review/quality
  - analysis/architecture
  - analysis/benchmarking
  - analysis/complexity
  - algorithm/crossover
  - algorithm/mutation
  - algorithm/evaluation
  - benchmark/reporting
  - benchmark/metrics
  - guide/methodology
  - guide/testing
---

# Exhaustive Multi-Axis Review: Schemas, Operator Mechanics, Benchmarking Architecture, and Testing Rigor

- [Exhaustive Multi-Axis Review: Schemas, Operator Mechanics, Benchmarking Architecture, and Testing Rigor](#exhaustive-multi-axis-review-schemas-operator-mechanics-benchmarking-architecture-and-testing-rigor)
  + [1. Executive Architectural Summary](#1-executive-architectural-summary)
  + [2. Batch 1: Schema Integrity & Operator Implementation Mechanics](#2-batch-1-schema-integrity--operator-implementation-mechanics)
    - [2.1 JSON Schema Audit (`algorithms.schema.json` & `problems.schema.json`)](#21-json-schema-audit-algorithmsschemajson--problemsschemajson)
      * [2.1.1 Rationale and Impact of `"additionalProperties": false`](#211-rationale-and-impact-of-additionalproperties-false)
      * [2.1.2 Cross-Examination: Schema Properties vs. Strategy Constructors](#212-cross-examination-schema-properties-vs-strategy-constructors)
      * [2.1.3 The `re_evaluate` Omission Defect in `PeriodicLocalSearchStep`](#213-the-re_evaluate-omission-defect-in-periodiclocalsearchstep)
      * [2.1.4 Recommended Schema Architecture](#214-recommended-schema-architecture)
    - [2.2 Cost Evaluation Vectorization (`cost_evaluation.py:47-57`)](#22-cost-evaluation-vectorization-cost_evaluationpy47-57)
      * [2.2.1 Mathematical & Memory Layout Analysis: `flat_dist` vs. `np.roll`](#221-mathematical--memory-layout-analysis-flat_dist-vs-nproll)
      * [2.2.2 Dtype Preservation and Return Contract Violation](#222-dtype-preservation-and-return-contract-violation)
      * [2.2.3 Edge Cases: 1D vs. 2D, Open vs. Closed Routes](#223-edge-cases-1d-vs-2d-open-vs-closed-routes)
    - [2.3 Order Crossover Mechanics (`order_crossover.py:56-83`)](#23-order-crossover-mechanics-order_crossoverpy56-83)
      * [2.3.1 Mathematical Equivalence to Davis Order Crossover (OX1)](#231-mathematical-equivalence-to-davis-order-crossover-ox1)
      * [2.3.2 Routing Bounds Compliance: `max_node` Safety Invariant](#232-routing-bounds-compliance-max_node-safety-invariant)
      * [2.3.3 Endpoint Invariance for Closed Topologies](#233-endpoint-invariance-for-closed-topologies)
    - [2.4 Inversion Mutation Safety (`inversion_mutation.py:72-99`)](#24-inversion-mutation-safety-inversion_mutationpy72-99)
      * [2.4.1 Memory Aliasing Verification (`copy=False` vs. `batch_np.copy()`)](#241-memory-aliasing-verification-copyfalse-vs-batch_npcopy)
      * [2.4.2 In-Place Slice Reversal Safety in NumPy](#242-in-place-slice-reversal-safety-in-numpy)
      * [2.4.3 Sibling Inconsistency: `MutationSwap` vs. `MutationInversion`](#243-sibling-inconsistency-mutationswap-vs-mutationinversion)
  + [3. Batch 2: Benchmarking Architecture, Reporting & DRY Analysis](#3-batch-2-benchmarking-architecture-reporting--dry-analysis)
    - [3.1 Multi-Axis Review of `report_generator.py`](#31-multi-axis-review-of-report_generatorpy)
      * [3.1.1 Correctness & Agnosticism Defect (`_get_baseline_algorithm`)](#311-correctness--agnosticism-defect-_get_baseline_algorithm)
      * [3.1.2 Toxic Fallback on Missing Baseline Instances (`first_alg = next(iter(...))`)](#312-toxic-fallback-on-missing-baseline-instances-first_alg--nextiter)
      * [3.1.3 Fleming & Wallace (1986) Formulation on Irregular Matrices](#313-fleming--wallace-1986-formulation-on-irregular-matrices)
      * [3.1.4 Architectural Duplication: Dual Formatter Bloat](#314-architectural-duplication-dual-formatter-bloat)
    - [3.2 DRY Architectural Audit: `orchestration.py` vs. `report_generator.py`](#32-dry-architectural-audit-orchestrationpy-vs-report_generatorpy)
      * [3.2.1 Redundant Metric Derivations and Problem Queries](#321-redundant-metric-derivations-and-problem-queries)
      * [3.2.2 Proposed Structural Decoupling: Intermediate Table Models](#322-proposed-structural-decoupling-intermediate-table-models)
  + [4. Batch 3: Runner Telemetry Verbosity & Behavioral Testing Quality](#4-batch-3-runner-telemetry-verbosity--behavioral-testing-quality)
    - [4.1 Telemetry Runner Verbosity (`algorithm_runner.py:74-208`)](#41-telemetry-runner-verbosity-algorithm_runnerpy74-208)
      * [4.1.1 Deconstructing the Boilerplate: 18 Ternary Guards](#411-deconstructing-the-boilerplate-18-ternary-guards)
      * [4.1.2 The DRY Refactored Structure](#412-the-dry-refactored-structure)
      * [4.1.3 Strict Invariant Preservation: C1, Bessel Correction, Schema](#413-strict-invariant-preservation-c1-bessel-correction-schema)
    - [4.2 Integration Testing Quality & Hardening](#42-integration-testing-quality--hardening)
      * [4.2.1 Brittle Assertions in `test_integration_pipeline.py`](#421-brittle-assertions-in-test_integration_pipelinepy)
      * [4.2.2 Behavioral Deficiencies in Existing Benchmarking Tests](#422-behavioral-deficiencies-in-existing-benchmarking-tests)
      * [4.2.3 Proposed Property-Based and Edge-Case Test Suite](#423-proposed-property-based-and-edge-case-test-suite)
  + [5. Comprehensive Recommendations & Action Plan](#5-comprehensive-recommendations--action-plan)

---

## 1. Executive Architectural Summary

This review conducts an exhaustive, domain-grounded inspection of the recent changes across the optimization algorithms, schemas, runner telemetry, and reporting pipeline in `gpu_accelerated_clean`.

Key conclusions:
1. **Schema Integrity**: The introduction of `"additionalProperties": false` in `algorithms.schema.json` correctly enforces configuration rigor, but contains a severe gap: `re_evaluate` is omitted from `pipeline_step.params`, making valid JSON configurations using `PeriodicLocalSearchStep(target="population")` fail schema validation.
2. **Operator Mechanics**:
   - `EvaluationCPUCost` achieves significant speedup and zero-allocation view slicing via 1D flattened indexing `flat_dist[u * n + v]`. However, it introduces a type contract violation: when distances are integer (`np.int32`), it returns an `int32` array cast to `Fitness`, diverging from GPU evaluation which always returns `np.float64`.
   - `CrossoverOrder._apply_ox` implements Davis Order Crossover (OX1) flawlessly without `set` allocations, strictly adhering to the `max_node = max(n, int(p1.max()) + 1, int(p2.max()) + 1)` buffer safety rule.
   - `MutationInversion` manages memory aliasing safely via `to_cpu_batch(..., copy=False)` paired with `batch_np if was_gpu else batch_np.copy()`. However, its sibling `MutationSwap` performs a redundant double copy.
3. **Reporting & Baseline Resolution**:
   - While Table 2 now uses the geometric mean of paired speedups (Fleming & Wallace 1986), `_get_baseline_algorithm` still hardcodes `"CPU"` and `"HybridNaive"`.
   - Table 1 continues to rely on `first_alg = next(iter(all_results[problem].keys()))` when the baseline is absent, generating spurious `1.00x†` speedups against arbitrary dictionary-ordered algorithms.
4. **DRY & Telemetry Verbosity**:
   - `algorithm_runner._build_summary_payload` contains 18 redundant ternary guards (`if total_successful > 0 else 0.0`), inflating code size without adding safety.
   - `report_generator.py` (767 lines) duplicates identical aggregation, filtering, and metric calculation logic across Markdown and LaTeX generator functions.
5. **Testing Rigor**:
   - `test_integration_pipeline.py` relies on brittle magic strings (`"20.00x"`, `"0.50x"`), testing happy-path constants rather than mathematical invariants, baseline omission behaviors, or arbitrary algorithm names.

---

## 2. Batch 1: Schema Integrity & Operator Implementation Mechanics

### 2.1 JSON Schema Audit (`algorithms.schema.json` & `problems.schema.json`)

#### 2.1.1 Rationale and Impact of `"additionalProperties": false`

In JSON Schema (Draft 2020-12), `additionalProperties` defaults to `true`. In complex metaheuristic systems, omitting `"additionalProperties": false` creates a failure mode: typos in parameter names (`"candidat_k": 20`, `"population_siz": 256`, `"mode": "naiv"`) are silently ignored by the JSON parser, falling back to default constructor arguments without raising an error.

Adding `"additionalProperties": false` closes this safety loophole. However, it imposes an absolute operational invariant: **the schema must remain in exact 1:1 synchronization with the underlying Python constructors**.

#### 2.1.2 Cross-Examination: Schema Properties vs. Strategy Constructors

We cross-examined all 17 strategy classes registered in `STRATEGY_REGISTRY` against `strategy_slot.params.properties` in `algorithms.schema.json:193-238`:

```
┌─────────────────────────────────┬────────────────────────────────────────┬─────────────┐
│ Strategy Class                  │ Constructor Parameters (__init__)      │ In Schema?  │
├─────────────────────────────────┼────────────────────────────────────────┼─────────────┤
│ `InitializationRandom`          │ `seed`, `return_gpu`                   │ Yes         │
│ `InitializationNearestNeighbor` │ `k`, `seed`, `return_gpu`, `candidate_k`│ Yes         │
│ `SelectionTournament`           │ `k`, `seed`                            │ Yes         │
│ `SelectionRouletteWheel`        │ `seed`                                 │ Yes         │
│ `SelectionGPUTournament`        │ `k`, `seed`                            │ Yes         │
│ `CrossoverOrder`                │ `seed`                                 │ Yes         │
│ `CrossoverPartiallyMapped`      │ `seed`                                 │ Yes         │
│ `CrossoverGPUOrder`             │ `seed`                                 │ Yes         │
│ `MutationSwap`                  │ `seed`, `candidate_k`                  │ Yes         │
│ `MutationInversion`             │ `seed`, `candidate_k`                  │ Yes         │
│ `MutationGPUSwap`               │ `seed`, `candidate_k`                  │ Yes         │
│ `MutationGPUInversion`          │ `seed`, `candidate_k`                  │ Yes         │
│ `LocalSearchCPUTwoOpt`          │ `max_iterations`, `mode`, `candidate_k`│ Yes         │
│ `LocalSearchGPUTwoOpt`          │ `max_iterations`, `mode`, `threads_per_block`, │ Yes   │
│                                 │ `max_matrix_dimension`, `max_shared_tour_cities`,│   │
│                                 │ `candidate_k`                          │             │
│ `LocalSearchNoOp`               │ `max_iterations`                       │ Yes         │
│ `EvaluationCPUCost`             │ None                                   │ Yes         │
│ `EvaluationGPUCost`             │ `return_gpu`, `threads_per_block`,     │ Yes         │
│                                 │ `max_matrix_dimension`                 │             │
└─────────────────────────────────┴────────────────────────────────────────┴─────────────┘
```

All 9 constructor parameters (`candidate_k`, `k`, `max_iterations`, `max_matrix_dimension`, `max_shared_tour_cities`, `mode`, `return_gpu`, `seed`, `threads_per_block`) are declared under `strategy_slot.params.properties`.

#### 2.1.3 The `re_evaluate` Omission Defect in `PeriodicLocalSearchStep`

Examining `PIPELINE_STEP_REGISTRY` reveals an immediate, high-severity defect:
- In `src/algorithms/pipeline_steps.py:182-196`:
  ```python
  class PeriodicLocalSearchStep(GenerationStep):
      def __init__(
          self,
          local_search: LocalSearchStrategy,
          interval: int = 10,
          target: Literal["offspring", "population"] = "population",
          re_evaluate: EvaluationStrategy | None = None,
      ) -> None:
          ...
          if target == "population" and re_evaluate is None:
              raise ValueError("re_evaluate EvaluationStrategy is required when target='population'")
  ```
- In `configs/schemas/algorithms.schema.json:257-282`:
  ```json
  "params": {
    "type": "object",
    "properties": {
      "local_search": { "$ref": "#/$defs/strategy_slot" },
      "evaluation": { "$ref": "#/$defs/strategy_slot" },
      "elite_count": { "type": "integer" },
      "interval": { "type": "integer" },
      "target": { "type": "string", "enum": ["offspring", "population"] }
    },
    "additionalProperties": false
  }
  ```

> [!CAUTION]
> **Schema Breaking Defect:** `re_evaluate` is completely omitted from `pipeline_step.params.properties`.
> If an operator declares a pipeline step with `target="population"` and supplies the mandatory `re_evaluate` strategy configuration in JSON, `jsonschema.validate` will raise:
> `ValidationError: Additional properties are not allowed ('re_evaluate' was unexpected)`.
> If the user omits `re_evaluate` to appease the schema, Python raises:
> `ValueError: re_evaluate EvaluationStrategy is required when target='population'`.
> This renders declared population periodic local search impossible via JSON configuration!

#### 2.1.4 Recommended Schema Architecture

1. **Immediate Patch**: Add `re_evaluate` to `pipeline_step.params`:
   ```json
   "re_evaluate": {
     "$ref": "#/$defs/strategy_slot",
     "description": "Evaluation strategy used to re-evaluate the population after periodic local search refinement."
   }
   ```
2. **Problems Schema Robustness**: In `configs/schemas/problems.schema.json`, allow optional metadata fields (`edge_weight_type`, `comment`, `source`) or change `additionalProperties: false` to allow non-breaking informative metadata on TSPLIB problem records.

---

### 2.2 Cost Evaluation Vectorization (`cost_evaluation.py:47-57`)

#### 2.2.1 Mathematical & Memory Layout Analysis: `flat_dist` vs. `np.roll`

In `src/algorithms/strategies/evaluation/cost_evaluation.py:50-56`:
```python
batch_np, _, _ = to_cpu_batch(routes)
distances: np.ndarray = context.get_cpu_distances()
flat_dist = distances.ravel()
n = distances.shape[1]
costs = flat_dist[batch_np[:, :-1] * n + batch_np[:, 1:]].sum(axis=1, dtype=distances.dtype)
costs += flat_dist[batch_np[:, -1] * n + batch_np[:, 0]]
return cast("Fitness", costs)
```

**Memory & Complexity Analysis:**
1. **Elimination of Intermediate Array Allocation**:
   The status-quo implementation used `next_cities = np.roll(batch_np, -1, axis=1)`. For population size $N=256$ and city count $n=1000$, `np.roll` allocates a brand-new $(256, 1000)$ array of `int32` ($1.024\text{ MB}$) and copies all $256,000$ integers on every evaluation call.
   In contrast, view slicing `batch_np[:, :-1]` and `batch_np[:, 1:]` creates lightweight NumPy array view descriptors (0 bytes allocated on heap; stride manipulation only).
2. **Flat 1D Indexing vs. 2D Fancy Indexing**:
   In C-contiguous row-major storage, matrix entry $D[u, v]$ resides at byte offset:
   $$\text{Offset}(u, v) = (u \times n + v) \times \text{sizeof}(\text{dtype})$$
   By using `flat_dist[batch_np[:, :-1] * n + batch_np[:, 1:]]`, NumPy executes a single flattened 1D gather operation rather than dispatching through 2D fancy indexing routines, yielding lower interpreter overhead and better L1/L2 cache prefetching.

#### 2.2.2 Dtype Preservation and Return Contract Violation

Notice line 54:
```python
costs = flat_dist[batch_np[:, :-1] * n + batch_np[:, 1:]].sum(axis=1, dtype=distances.dtype)
costs += flat_dist[batch_np[:, -1] * n + batch_np[:, 0]]
return cast("Fitness", costs)
```

In `src/protocols/types.py:29`:
```python
type Fitness = NDArray[np.float64]
```
In `src/protocols/types.py:32`:
```python
type DistanceMatrix = NDArray[np.int32]
```

When evaluated on standard TSPLIB instances (e.g. `berlin52`, `att48`), `distances` is an integer matrix (`np.int32`).
- `costs.sum(..., dtype=distances.dtype)` produces an `NDArray[np.int32]`.
- `costs += ...` maintains `int32`.
- `return cast("Fitness", costs)` masks the type error with a mypy cast, but at runtime `costs.dtype` is `int32`!

In contrast, `EvaluationGPUCost.evaluate_batch` (`cost_evaluation.py:248`) explicitly returns `costs_cpu = d_costs.get().astype(np.float64)`.

> [!WARNING]
> **Subtle Divergence:** CPU cost evaluation returns integer arrays when distances are integer, while GPU cost evaluation always returns `float64`. Downstream statistical calculations or NumPy vector operations expecting `float64` precision face type instability.
> **Fix**: Explicitly sum or cast with `dtype=np.float64`:
> ```python
> costs = flat_dist[batch_np[:, :-1] * n + batch_np[:, 1:]].sum(axis=1, dtype=np.float64)
> costs += flat_dist[batch_np[:, -1] * n + batch_np[:, 0]].astype(np.float64)
> ```

#### 2.2.3 Edge Cases: 1D vs. 2D, Open vs. Closed Routes

- **1D Route Support**: `to_cpu_batch(routes)` ensures 1D inputs $(n,)$ are expanded to $(1, n)$. Slices `[:, :-1]` produce $(1, n-1)$, and `sum(axis=1)` returns $(1,)$, which matches the expected batch shape.
- **Closed Route Invariant**: The term `costs += flat_dist[batch_np[:, -1] * n + batch_np[:, 0]]` represents the return edge $(v_{n-1} \to v_0)$. This is correct for closed TSP tours. If applied to CVRP routes where depot endpoints are already duplicated (`[0, c_1, ..., c_k, 0]`), adding $D[0, 0] = 0$ is mathematically benign only if the diagonal is strictly zero. For pure TSP permutations, it closes the loop cleanly.

---

### 2.3 Order Crossover Mechanics (`order_crossover.py:56-83`)

#### 2.3.1 Mathematical Equivalence to Davis Order Crossover (OX1)

Davis (1985) Order Crossover (OX1) is defined as follows:
Given two parents $P_1$ and $P_2$ of length $L$, and two cut points $c_1, c_2$ with $0 \le c_1 \le c_2 < L$:
1. The child $C$ copies the substring $P_1[c_1 \dots c_2]$ directly into positions $c_1 \dots c_2$.
2. Starting from position $c_2 + 1$ and wrapping cyclically modulo $L$, inspect the genes of $P_2$ in cyclic order starting from $c_2 + 1$.
3. Append each gene of $P_2$ that does not already occur in $P_1[c_1 \dots c_2]$ into the empty positions of $C$, also proceeding cyclically from $c_2 + 1$.

In `src/algorithms/strategies/crossover/order_crossover.py:69-82`:
```python
max_node = max(n, int(p1.max()) + 1, int(p2.max()) + 1)
in_seg = np.zeros(max_node, dtype=bool)
in_seg[p1[c1 : c2 + 1]] = True

p2_stream = p2[stream_start : stream_start + stream_len]
shift = (c2_rel + 1) % stream_len

p2_rotated = np.roll(p2_stream, -shift)
remaining_genes = p2_rotated[~in_seg[p2_rotated]]

rotated_positions = (np.arange(stream_len) + shift) % stream_len
empty_mask = child[stream_start + rotated_positions] == -1
child[stream_start + rotated_positions[empty_mask]] = remaining_genes
```

**Proof of Equivalence:**
1. `np.roll(p2_stream, -shift)` rotates $P_2$ so that the gene at index $(c_2 + 1)$ becomes index $0$.
2. The boolean mask `~in_seg[p2_rotated]` filters out all genes contained in $P_1[c_1 \dots c_2]$ in $\mathcal{O}(L)$ time, preserving the relative cyclic order of $P_2$.
3. `rotated_positions` maps cyclic slots starting at $c_2 + 1$ back to chromosome coordinates.
4. `empty_mask` selects only positions not already occupied by the preserved $P_1$ slice.
5. The assignment writes the filtered genes into these positions in exact cyclic order.

This is an exact vectorization of Davis OX1. It eliminates the previous $\mathcal{O}(L)$ Python `set` creation and sequential iteration without changing the output permutation.

#### 2.3.2 Routing Bounds Compliance: `max_node` Safety Invariant

Rule 1 from `code-review-and-quality/SKILL.md` states:
> *In routing operators (TSP/CVRP), verify lookup buffers (such as inverse_positions) size against max node ID (`max(len(route), int(route.max()) + 1)`), never `len(route)`.*

Line 69 adheres to this standard:
```python
max_node = max(n, int(p1.max()) + 1, int(p2.max()) + 1)
in_seg = np.zeros(max_node, dtype=bool)
```
If an operator processes a CVRP cluster or sub-tour where the cities are non-contiguous IDs (e.g., cities $[102, 54, 880]$ with $n=3$), allocating `in_seg = np.zeros(n)` would trigger `IndexError: index 880 is out of bounds for axis 0 with size 3`. Taking the maximum over `n`, `p1.max() + 1`, and `p2.max() + 1` mathematically guarantees zero out-of-bounds indexing.

#### 2.3.3 Endpoint Invariance for Closed Topologies

Lines 59-66 handle closed route topologies:
```python
stream_start = 1 if is_closed else 0
stream_len = (n - 2) if is_closed else n
c2_rel = (c2 - 1) if is_closed else c2

if is_closed:
  child[0] = p1[0]
  child[-1] = p1[-1]
```
When `is_closed=True`, cut points are restricted to $[1, n-2]$. The depot endpoints at $0$ and $n-1$ are frozen, the cyclic shift operations are performed solely on the internal permutation of length $n-2$, and the closed depot constraints remain intact.

---

### 2.4 Inversion Mutation Safety (`inversion_mutation.py:72-99`)

#### 2.4.1 Memory Aliasing Verification (`copy=False` vs. `batch_np.copy()`)

In `src/algorithms/strategies/mutation/inversion_mutation.py:78-94`:
```python
batch_np, was_1d, was_gpu = to_cpu_batch(routes, copy=False)
...
out = batch_np if was_gpu else batch_np.copy()
for i in np.flatnonzero(probs < mutation_rate):
  start = int(min(pos1[i], pos2[i]))
  end = int(max(pos1[i], pos2[i]))
  out[i, start : end + 1] = out[i, start : end + 1][::-1]
return restore_routes_shape(out, was_1d, to_gpu=was_gpu)
```

**Memory Flow Trace:**
- **Case 1: Input is GPU array (`was_gpu == True`)**:
  `to_cpu_batch(routes, copy=False)` calls `routes.get()`, allocating a brand-new host NumPy array. Thus, `batch_np` is already a private copy of the device data. Setting `out = batch_np` performs zero redundant host allocations. Mutating `out` in place modifies this temporary host array. Finally, `restore_routes_shape(out, ..., to_gpu=True)` uploads the mutated data to a new GPU allocation via `cp.asarray(out)`. The caller's GPU array is never aliased or modified.
- **Case 2: Input is CPU array (`was_gpu == False`)**:
  `to_cpu_batch(routes, copy=False)` returns a view or the original array. Setting `out = batch_np.copy()` creates an explicit copy of the array. Mutating `out` modifies only the newly created copy. The caller's original CPU array is preserved.

#### 2.4.2 In-Place Slice Reversal Safety in NumPy

Line 98 executes:
```python
out[i, start : end + 1] = out[i, start : end + 1][::-1]
```
The right-hand side `out[i, start : end + 1][::-1]` is a strided view with negative step `stride = -itemsize`.

**Memory Overlap Analysis:**
In C, a naive `memcpy` between overlapping memory regions with negative strides produces undefined behavior (data corruption).
However, NumPy's assignment engine (`PyArray_AssignArray`) performs internal overlap detection using `PyArray_SharesMemoryWith`. When overlapping source and destination buffers are detected, NumPy creates an internal temporary buffer to read the reversed slice before writing it back into memory.
Therefore, in-place slice reversal is safe in NumPy.

#### 2.4.3 Sibling Inconsistency: `MutationSwap` vs. `MutationInversion`

In `src/algorithms/strategies/mutation/swap_mutation.py:77-93`:
```python
batch_np, was_1d, was_gpu = to_cpu_batch(routes, copy=True)
...
out = batch_np.copy()
```
`MutationSwap` calls `to_cpu_batch(..., copy=True)` (allocating a copy) and then immediately calls `out = batch_np.copy()` (allocating a second copy)!
For CPU inputs, `MutationSwap` allocates **two complete copies** of the population batch, whereas `MutationInversion` allocates only one.

> [!TIP]
> **Performance Alignment**: Align `swap_mutation.py` with `inversion_mutation.py` by using `copy=False` in `to_cpu_batch` and `out = batch_np if was_gpu else batch_np.copy()`.

---

## 3. Batch 2: Benchmarking Architecture, Reporting & DRY Analysis

### 3.1 Multi-Axis Review of `report_generator.py`

#### 3.1.1 Correctness & Agnosticism Defect (`_get_baseline_algorithm`)

In `src/benchmarking/analysis/report_generator.py:49-55`:
```python
def _get_baseline_algorithm(algorithm_names: list[str]) -> str:
  """Determine baseline algorithm for speedup calculations."""
  if "CPU" in algorithm_names:
    return "CPU"
  if "HybridNaive" in algorithm_names:
    return "HybridNaive"
  return algorithm_names[0] if algorithm_names else "Unknown"
```

**Why this violates Task 12.16 (`DP12-9`):**
1. It retains hardcoded string literals `"CPU"` and `"HybridNaive"`.
2. If an experiment benchmarks arbitrary algorithms:
   `["MyExperimentalGA", "StandardGA"]`
   `_get_baseline_algorithm` arbitrarily picks `"MyExperimentalGA"` simply because `"M" < "S"` alphabetically!
3. If an experiment intentionally sets `baseline_algorithm="CustomBaseline"`, but calls helper functions where `baseline_algorithm` is defaulted, the hardcoded check for `"CPU"` or `"HybridNaive"` overrides the caller's architecture.

**The Clean Pattern**:
Baseline algorithm resolution must follow an explicit hierarchy:
1. `caller_baseline`: If explicitly passed to the generator, it is authoritative.
2. Metadata detection: If an algorithm declares `backend == "CPU"` in its metadata, it is selected as the primary baseline.
3. Fallback: If neither is available, select the first algorithm in candidate order and issue an explicit `logging.info` noting the dynamically resolved baseline.

#### 3.1.2 Toxic Fallback on Missing Baseline Instances (`first_alg = next(iter(...))`)

In `src/benchmarking/analysis/report_generator.py:119` and `694-696`:
```python
problem_base_alg = baseline_algorithm if not is_fallback else next(iter(all_results[problem].keys()))
base_time = (
  baseline_times[problem]
  if problem in baseline_times
  else float(all_results[problem][problem_base_alg]["mean_time"])
)
```

**Scientific Impact:**
Suppose the configured baseline is `CPU`. CPU is intentionally skipped for instances with $n > 400$ cities.
On `att532`, `all_results["att532"]` contains:
- `HybridOptimized`: $2.50\text{ s}$
- `FullGPUModular`: $0.25\text{ s}$

Under the current code:
- `next(iter(...))` selects `HybridOptimized` (or whichever algorithm was inserted first in the dictionary).
- For `HybridOptimized`, it calculates $\frac{2.50}{2.50} = 1.00\times^\dagger$.
- For `FullGPUModular`, it calculates $\frac{2.50}{0.25} = 10.00\times^\dagger$.
- **The table column header reads: `Speedup vs CPU`!**

The reader is presented with a speedup claiming that `FullGPUModular` is $10\times$ faster than `CPU`, when in reality it was computed against `HybridOptimized`!
Furthermore, because dictionary iteration order depends on execution history, if `FullGPUModular` was inserted first, `FullGPUModular` becomes the baseline, rendering `HybridOptimized` as $0.10\times^\dagger$!

> [!IMPORTANT]
> **Mathematical Rule for Missing Baselines**:
> If the baseline algorithm was not evaluated for a given problem instance, **the speedup cell must render as a neutral dash (`—`)**.
> An arbitrary algorithm must **never** be silently substituted as a pseudo-baseline.
> If an intra-regime secondary baseline is desired (e.g. comparing GPU variants against `HybridNaive` when `CPU` is absent), this must be an **explicit, user-configured secondary baseline parameter**, not a dictionary-order accident.

#### 3.1.3 Fleming & Wallace (1986) Formulation on Irregular Matrices

In Table 2 (`report_generator.py:289-304`), the paired geometric speedup is implemented as:
```python
paired_speedups = []
if alg != baseline_algorithm:
  for problem in problem_names:
    if problem in all_results:
      base_time = float(all_results[problem].get(baseline_algorithm, {}).get("mean_time", 0.0))
      cand_time = float(all_results[problem].get(alg, {}).get("mean_time", 0.0))
      if base_time > 0.0 and cand_time > 0.0:
        paired_speedups.append(base_time / cand_time)

if paired_speedups:
  avg_speedup = float(np.exp(np.mean(np.log(paired_speedups))))
  speedup_str = f"{avg_speedup:.2f}x"
```

**Mathematical Evaluation:**
Let $P_{\text{baseline}}$ be the problems evaluated by the baseline, and $P_{\text{cand}}$ be the problems evaluated by candidate algorithm $A$.
The paired set is:
$$P_{\cap} = P_{\text{baseline}} \cap P_{\text{cand}}$$
The geometric mean of ratios:
$$\mu_g(A) = \exp\left( \frac{1}{|P_{\cap}|} \sum_{p \in P_{\cap}} \ln \frac{T_{\text{base}, p}}{T_{A, p}} \right)$$
satisfies the relational axioms established by Fleming & Wallace (1986):
1. **Dimension Invariance**: Multiplicative scaling of any problem's time unit does not alter the ratio.
2. **Transitivity**: $\mu_g(A / B) = \mu_g(A) / \mu_g(B)$ across identical instance sets.

**The Missing Metric**:
The current implementation reports `speedup_str = "12.50x"`, but omits $|P_{\cap}|$!
If `CPU` was evaluated on 12 instances ($n < 100$) and `HybridNaive` on 38 instances, reporting `21.33x` for CPU vs `2.10x` for HybridNaive without stating the cohort size gives the false impression that both speedups were evaluated over the same 38 problems.
Table 2 must report: `21.33x (n=12)` or annotate the row with the instance count.

#### 3.1.4 Architectural Duplication: Dual Formatter Bloat

`report_generator.py` spans 767 lines. It defines four tables, each implemented as two separate functions:
- Table 1: `_generate_table1_markdown` (68 lines) + `_generate_table1_latex` (92 lines)
- Table 2: `_generate_table2_markdown` (50 lines) + `_generate_table2_latex` (68 lines)
- Table 3: `_generate_table3_markdown` (58 lines) + `_generate_table3_latex` (71 lines)
- Table 4: `_generate_phase_timings_markdown` (67 lines) + `_generate_phase_timings_latex` (73 lines)

**The Flaw**: Every analytical computation (extracting numbers from nested dicts, computing `np.mean`, calculating Bessel-corrected standard deviations, computing paired geometric speedups, categorizing by problem size) is **implemented twice**!
If a bug is fixed in Markdown Table 2 (e.g. geometric mean speedup), the developer must remember to duplicate the fix in LaTeX Table 2.

---

### 3.2 DRY Architectural Audit: `orchestration.py` vs. `report_generator.py`

#### 3.2.1 Redundant Metric Derivations and Problem Queries

A line-by-line comparison between `src/benchmarking/runner/orchestration.py` and `src/benchmarking/analysis/report_generator.py` identifies four major duplicated abstractions:

1. **Problem Size Lookup**:
   - `report_generator.py:58-63`:
     ```python
     def _get_problem_size(problem_name: str, problem_configs: list[ProblemConfig]) -> int | None:
       for config in problem_configs:
         if config.name == problem_name:
           return int(config.size)
       return None
     ```
   - `orchestration.py` repeatedly performs linear scans or dict comprehensions `{p.name: p.size for p in problem_configs}`.
2. **Bessel Standard Deviation Calculation**:
   - Both modules implement ad-hoc inline ternary expressions for Bessel's correction:
     `float(np.std(data, ddof=1)) if len(data) > 1 else 0.0`.
3. **Problem Categorization by Size**:
   - `report_generator.py:67-84` (`_categorize_by_size`) partitions problems into `Small (n<100)`, `Medium (100<=n<300)`, `Large (n>=300)`.
   - `aggregate_statistics.py:322` partitions problems into `small_problems` using `cpu_threshold`.
4. **Table 1 Speedup Calculation**:
   - Both modules duplicate logic for calculating per-problem speedup ratios.

#### 3.2.2 Proposed Structural Decoupling: Intermediate Table Models

To eliminate 300+ lines of duplicated code, extract the analytical computation into strongly typed, immutable dataclasses representing the intermediate table models:

```python
# src/benchmarking/analysis/table_models.py

@dataclass(frozen=True)
class Table1Row:
    problem_name: str
    problem_size: int
    algorithm: str
    mean_cost: float
    std_cost: float
    mean_gap: float
    mean_time: float
    speedup: float | None  # None renders as "—"
    is_fallback: bool

@dataclass(frozen=True)
class Table2Row:
    algorithm: str
    mean_gap: float
    std_gap: float
    mean_time: float
    std_time: float
    geometric_speedup: float | None
    paired_instances_count: int
```

The workflow cleanly splits into two steps:
1. **Data Builder**: `build_table1_rows(all_results, problem_configs, baseline) -> list[Table1Row]` (contains 100% of the math, zero string formatting).
2. **Presenters**:
   - `render_table1_markdown(rows: list[Table1Row]) -> str`
   - `render_table1_latex(rows: list[Table1Row]) -> str`

---

## 4. Batch 3: Runner Telemetry Verbosity & Behavioral Testing Quality

### 4.1 Telemetry Runner Verbosity (`algorithm_runner.py:74-208`)

#### 4.1.1 Deconstructing the Boilerplate: 18 Ternary Guards

In `src/benchmarking/runner/algorithm_runner.py:166-183`:
```python
summary: dict[str, Any] = {
  "algorithm": algorithm_name,
  "repetitions": len(times),
  "successful_runs": total_successful,
  "mean_time": float(np.nanmean(times)) if total_successful > 0 else 0.0,
  "std_time": float(np.nanstd(times, ddof=std_ddof)) if total_successful > 1 else 0.0,
  "min_time": float(np.nanmin(times)) if total_successful > 0 else 0.0,
  "max_time": float(np.nanmax(times)) if total_successful > 0 else 0.0,
  "mean_cost": float(np.nanmean(final_costs)) if total_successful > 0 else 0.0,
  "std_cost": float(np.nanstd(final_costs, ddof=std_ddof)) if total_successful > 1 else 0.0,
  "best_cost": float(np.nanmin(final_costs)) if total_successful > 0 else 0.0,
  "worst_cost": float(np.nanmax(final_costs)) if total_successful > 0 else 0.0,
  "mean_initial_cost": float(np.nanmean(initial_costs_arr)) if total_successful > 0 else 0.0,
  "std_initial_cost": float(np.nanstd(initial_costs_arr, ddof=std_ddof)) if total_successful > 1 else 0.0,
  "mean_gap": float(np.nanmean(gaps)) if total_successful > 0 else 0.0,
  "std_gap": float(np.nanstd(gaps, ddof=std_ddof)) if total_successful > 1 else 0.0,
  "best_gap": float(np.nanmin(gaps)) if total_successful > 0 else 0.0,
  "mean_improvement": float(np.nanmean(improvements)) if total_successful > 0 else 0.0,
  "mean_generations": float(np.nanmean(generations_arr)) if total_successful > 0 else 0.0,
  "std_generations": float(np.nanstd(generations_arr, ddof=std_ddof)) if total_successful > 1 else 0.0,
  "min_generations": float(np.nanmin(generations_arr)) if total_successful > 0 else 0.0,
  "max_generations": float(np.nanmax(generations_arr)) if total_successful > 0 else 0.0,
...
```

**Critique:**
- The condition `if total_successful > 0 else 0.0` is copy-pasted 13 times.
- The condition `if total_successful > 1 else 0.0` is copy-pasted 5 times.
- It spans 30+ lines of visual noise.
- If the default fallback or Bessel rule needs modification, an engineer must edit 18 lines individually.

#### 4.1.2 The DRY Refactored Structure

Extract a focused statistical reduction helper:

```python
def _compute_series_stats(
    arr: np.ndarray,
    n_valid: int,
    ddof: int,
) -> tuple[float, float, float, float]:
    """Compute (mean, std, min, max) with Bessel's correction and zero-guards."""
    if n_valid <= 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = float(np.nanmean(arr))
    std_val = float(np.nanstd(arr, ddof=ddof)) if n_valid > 1 else 0.0
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    return mean_val, std_val, min_val, max_val
```

Then `_build_summary_payload` simplifies cleanly:

```python
m_time, s_time, min_time, max_time = _compute_series_stats(times, total_successful, std_ddof)
m_cost, s_cost, b_cost, w_cost = _compute_series_stats(final_costs, total_successful, std_ddof)
m_init, s_init, _, _ = _compute_series_stats(initial_costs_arr, total_successful, std_ddof)
m_gap, s_gap, b_gap, _ = _compute_series_stats(gaps, total_successful, std_ddof)
m_gen, s_gen, min_gen, max_gen = _compute_series_stats(generations_arr, total_successful, std_ddof)
m_imp = float(np.nanmean(improvements)) if total_successful > 0 else 0.0

summary: dict[str, Any] = {
    "algorithm": algorithm_name,
    "repetitions": len(times),
    "successful_runs": total_successful,
    "mean_time": m_time,
    "std_time": s_time,
    "min_time": min_time,
    "max_time": max_time,
    "mean_cost": m_cost,
    "std_cost": s_cost,
    "best_cost": b_cost,
    "worst_cost": w_cost,
    "mean_initial_cost": m_init,
    "std_initial_cost": s_init,
    "mean_gap": m_gap,
    "std_gap": s_gap,
    "best_gap": b_gap,
    "mean_improvement": m_imp,
    "mean_generations": m_gen,
    "std_generations": s_gen,
    "min_generations": min_gen,
    "max_generations": max_gen,
    ...
}
```

#### 4.1.3 Strict Invariant Preservation: C1, Bessel Correction, Schema

This refactoring strictly preserves all required contracts:
1. **Negative Constraint C1 (Timing Isolation)**: `_compute_series_stats` operates strictly on in-memory arrays after the `perf_counter` block. Zero I/O is invoked.
2. **Bessel's Correction**: `std_ddof` is passed into `_compute_series_stats`, ensuring `ddof=1` when `total_successful > 1`.
3. **DuckDB / Checkpoint Schema Compatibility**: All 27 required keys and all nullable keys are emitted under identical dictionary names and types.

---

### 4.2 Integration Testing Quality & Hardening

#### 4.2.1 Brittle Assertions in `test_integration_pipeline.py`

In `tests/benchmarking/test_integration_pipeline.py:170-179`:
```python
generate_result_tables(
  all_results=all_results,
  problem_configs=problem_configs,
  output_dir=tmp_path,
  benchmark_name="test_table",
  baseline_algorithm="HybridNaive",
)
md_content: str = (tmp_path / "test_table.md").read_text(encoding="utf-8")
assert "20.00x" in md_content
assert "0.50x" in md_content
```

**Critique:**
1. **Magic Constant Assertions**: Testing `assert "20.00x" in md_content` checks that a hardcoded string exists somewhere in the markdown document. It does not verify:
   - Which table row contains `"20.00x"`.
   - Which algorithm column it was assigned to.
   - Whether the LaTeX table (`test_table.tex`) contains the corresponding `20.00\times` string.
   - Whether the baseline algorithm row itself displays `1.00x` or `—`.
2. **Happy-Path Monoculture**: The test provides identical coverage across all algorithms (every algorithm ran on every problem). It fails to test irregular matrices where algorithms ran on differing subsets of instances.

#### 4.2.2 Behavioral Deficiencies in Existing Benchmarking Tests

The current test suite (`tests/benchmarking/`) lacks property-based tests for:
- **Arbitrary Algorithm Names**: What happens when algorithm names contain spaces, underscores, or unconventional characters (`"Algorithm-V2 (Experimental)"`)?
- **Missing Baseline Problem Tier**: What happens when `baseline_algorithm="CPU"`, but the test data contains problems where `CPU` was omitted?
- **Corrupted In-Flight Checkpoints**: What happens when an in-flight checkpoint has 0 bytes or truncated JSON? (Task 12.11 solved this in `orchestration.py`, but integration tests do not assert recovery).
- **Single-Repetition Cohorts ($N=1$)**: Ensuring `std_time == 0.0` and no `Degrees of freedom <= 0 for slice` warnings are emitted.

#### 4.2.3 Proposed Property-Based and Edge-Case Test Suite

Add `tests/benchmarking/test_unit_report_generator_behavioral.py`:

```python
"""Behavioral and property-based tests for dynamic report generation."""

import pytest
import numpy as np
from pathlib import Path
from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.runner.config_loader import ProblemConfig

def test_report_generator_with_arbitrary_algorithm_names(tmp_path: Path):
    """Verify tables render cleanly when algorithms have arbitrary non-canonical names."""
    all_results = {
        "eil51": {
            "Metaheuristic_A": {"mean_time": 1.0, "mean_cost": 500.0, "std_cost": 5.0, "mean_gap": 1.0, "std_gap": 0.1},
            "CrazyCpuGpuEveryStep": {"mean_time": 0.2, "mean_cost": 495.0, "std_cost": 2.0, "mean_gap": 0.0, "std_gap": 0.0},
        }
    }
    configs = [ProblemConfig(name="eil51", optimal=495.0, size=51)]
    
    generate_result_tables(
        all_results=all_results,
        problem_configs=configs,
        output_dir=tmp_path,
        benchmark_name="arbitrary_test",
        baseline_algorithm="Metaheuristic_A",
    )
    
    md_text = (tmp_path / "arbitrary_test.md").read_text()
    tex_text = (tmp_path / "arbitrary_test.tex").read_text()
    
    # Must contain arbitrary names without raising
    assert "CrazyCpuGpuEveryStep" in md_text
    assert "CrazyCpuGpuEveryStep" in tex_text
    assert "5.00x" in md_text

def test_report_generator_renders_dash_when_baseline_absent(tmp_path: Path):
    """Verify speedup renders as neutral dash '—' when baseline was not evaluated on instance."""
    all_results = {
        "lin318": {
            "HybridOptimized": {"mean_time": 2.0, "mean_cost": 42000.0, "std_cost": 10.0, "mean_gap": 0.5, "std_gap": 0.1},
            "FullGPUModular": {"mean_time": 0.2, "mean_cost": 42000.0, "std_cost": 5.0, "mean_gap": 0.5, "std_gap": 0.1},
        }
    }
    configs = [ProblemConfig(name="lin318", optimal=42029.0, size=318)]
    
    generate_result_tables(
        all_results=all_results,
        problem_configs=configs,
        output_dir=tmp_path,
        benchmark_name="missing_baseline_test",
        baseline_algorithm="CPU",  # CPU was NOT run on lin318
    )
    
    md_text = (tmp_path / "missing_baseline_test.md").read_text()
    # Must NOT substitute a random algorithm; speedup must be a dash
    assert "| lin318 | HybridOptimized |" in md_text
    assert "| — |" in md_text
```

---

## 5. Comprehensive Recommendations & Action Plan

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  Actionable Recommendations Matrix                         │
├────┬─────────────────────────────┬────────────────────────────────────────┤
│ No.│ Component                   │ Remediation Action                     │
├────┼─────────────────────────────┼────────────────────────────────────────┤
│ 1  │ `algorithms.schema.json`    │ Add `re_evaluate` to `pipeline_step`.  │
│ 2  │ `cost_evaluation.py`        │ Enforce `float64` return dtype for CPU.│
│ 3  │ `swap_mutation.py`          │ Eliminate redundant double copy.       │
│ 4  │ `report_generator.py`       │ Remove hardcoded baseline names;       │
│    │                             │ render `—` on missing baseline rows;   │
│    │                             │ decouple computation from formatters.  │
│ 5  │ `algorithm_runner.py`       │ Factor out `_compute_series_stats`     │
│    │                             │ to eliminate 18 ternary guards.        │
│ 6  │ Test Suite                  │ Add property-based & irregular matrix  │
│    │                             │ integration tests.                     │
└────┴─────────────────────────────┴────────────────────────────────────────┘
```

### Verification Standard
Each proposed modification directly improves architectural cleanliness, memory safety, or mathematical correctness without breaking existing tests. All changes can be validated using:
- `.venv/bin/python -m benchmark_scripts.check_docs` $\to$ Exit 0.
- `mypy src tests benchmark_scripts` $\to$ 0 issues.
- `pytest tests/benchmarking/` $\to$ 100% pass rate.
