---
title: "Strategy Class Hierarchy, ProblemContext Contracts, and Three-Tier Distance Architecture"
description: >
  Architectural reference documenting the Design B slot-prefixed strategy class
  hierarchy (`StrategyBase` + `Initialization*`, `Selection*`, `Crossover*`,
  `Mutation*`, `LocalSearch*`, `Evaluation*`), `ProblemContext` GPU cache
  ownership (`cp = require_gpu()`), the three-tier TSPLIB95 distance layer, and
  benchmark checkpoint provenance.
created: 2026-11-04
status: review
author:
  - "[[Lucas Galdino]]"
  - "[[Claude Opus 4.6]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-04
    modifications:
      - description: >
          Created architectural knowledge-base document synthesizing the
          post-Phase 10 review findings: unifying the 6 TwoOpt classes into
          LocalSearchCPUTwoOpt and LocalSearchGPUTwoOpt, comparing Design A
          (umbrella classes) vs Design B (slot-prefixed operator-family classes
          inheriting from StrategyBase), fixing ProblemContext GPU cache
          ownership (BUG-14), aligning matrix materialization thresholds
          (DP11-3), and documenting the three-tier distance architecture.
related_files:
  - "[analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md)"
  - "[analysis-gpu-memory-calculation.md](./analysis-gpu-memory-calculation.md)"
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py)"
  - "[matrix.py](../../src/distances/matrix.py)"
  - "[pairwise.py](../../src/distances/pairwise.py)"
tags:
  - guide/architecture
  - analysis/planning
  - review/quality
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/local-search
  - notes/cuda
  - notes/vram
  - benchmark/gpu
  - guide/testing
---

# Strategy Class Hierarchy, `ProblemContext` Contracts, and Three-Tier Distance Architecture

This document records the architectural principles and root-cause analyses established during the Phase 10 / Phase 11 review of `gpu_accelerated_clean` and `routing_data`.

---

## 1. Strategy Class Hierarchy: Umbrella vs Operator-Family Classes (`Design B` / `DP11-1`)

### 1.1 Every Strategy in a Slot Shares the Exact Same Input/Output Contract

In [src/protocols/algorithm_strategies.py](../../src/protocols/algorithm_strategies.py), each of the six evolutionary operator slots is governed by a structural `Protocol`:

| Strategy Slot | Protocol Method Signature | Input Shape | Output Shape |
| :--- | :--- | :--- | :--- |
| **Initialization** | `initialize(customers, population_size, context)` | `Sequence[int]`, `int` | `Routes` `(P, n)` `int32` |
| **Selection** | `select(population, fitness_scores, num_parents, context)` | `Routes` `(P, n)`, `Fitness` `(P,)` | `Routes` `(M, n)` `int32` |
| **Crossover** | `crossover_batch(routes, context)` | `RouteOrRoutes` `(P, n)` or `(n,)` | `RouteOrRoutes` `int32` |
| **Mutation** | `mutate_batch(routes, mutation_rate, context)` | `RouteOrRoutes`, `float` | `RouteOrRoutes` `int32` |
| **Local Search** | `improve_batch(routes, context)` | `RouteOrRoutes` `(P, n)` or `(n,)` | `RouteOrRoutes` `int32` |
| **Evaluation** | `evaluate_batch(routes, context)` | `RouteOrRoutes` `(P, n)` or `(n,)` | `Fitness` `(P,)` `float64` |

Because all implementations in a given slot consume and return the **exact same types**, duplicating telemetry counters (`h2d_bytes`, `d2h_bytes`, `kernel_launches`, `reset_metrics()`), RNG resolution (`seed`, `rng`), and batch normalization (`ensure_routes_batch`, `to_cpu_batch`, `restore_routes_shape`) across classes violates **DRY (Don't Repeat Yourself)** and **SRP (Single Responsibility Principle)**.

### 1.2 Why the 6-Class Split in `two_opt.py` Violated DRY and Liskov Substitution (LSP)

In [src/algorithms/strategies/local_search/two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py), six separate classes existed:

- **CPU:** `CPUTwoOpt`, `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`
- **GPU:** `NaiveGPUTwoOpt`, `BatchGPUTwoOpt`, `BatchGPUMultiMoveTwoOpt`

Inspecting the three CPU classes reveals that they are **90% identical**:

1. **Identical outer loop and batch wrapper:** `improve(route, context)` and `improve_batch(routes, context)` are copy-pasted three times.
2. **Identical neighborhood scan:** All three classes iterate $i \in \{0, \dots, n - 3\}$, hoist `(a, b, d_ab)`, and scan either the upper triangle $j \in \{i + 2, \dots, n - 1\}$ (with carry-over `c = d`) or the $k$-NN candidate list $r \in \{0, \dots, k_{\text{eff}} - 1\}$ (`j = pos[candidates[a, r]]`).
3. **Only the pass-level acceptance rule differs:**
   + `best` (*steepest descent*, Croes, 1958): track `(best_delta, best_i, best_j)` over the whole pass and reverse once at the end of the pass.
   + `first` (*first improvement*, Johnson & McGeoch, 1997): reverse immediately upon encountering the first $\Delta_{i,j} < 0$ and restart the pass.
   + `multi_move` (*disjoint greedy*, Rocki & Suda, 2013): collect all $(\Delta_{i,j}, i, j)$ with $\Delta_{i,j} < 0$, sort ascending by $\Delta_{i,j}$, and apply all non-overlapping reversals $[i_1, j_1] \cap [i_2, j_2] = \emptyset$ in the same pass.

Furthermore, `BatchGPUMultiMoveTwoOpt.__init__` omitted `candidate_k: int | None = None`, whereas `CPUTwoOpt`, `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`, and `BatchGPUTwoOpt` accepted `candidate_k`. Passing `strategy_overrides={"local_search": {"candidate_k": 20}}` therefore worked on `BatchGPUTwoOpt` but raised `TypeError` on `BatchGPUMultiMoveTwoOpt` — a **constructor-level Liskov Substitution Principle (LSP) violation**.

### 1.3 Design A (Umbrella per Slot) vs Design B (Slot-Prefixed Operator-Family Classes + `StrategyBase`)

We evaluated two ways to eliminate this duplication while making every strategy's slot immediately obvious from its class name:

| Criterion | Design A: One Umbrella Class per Slot & Backend (`MutationCPU(method="swap"\|"inversion")`) | Design B (`DP11-1` — Selected): Slot-Prefixed Operator-Family Classes + `StrategyBase` |
| :--- | :--- | :--- |
| **Naming clarity** | Slot + Backend (`InitializationCPU`, `MutationCPU`, `LocalSearchCPU`). Algorithm family is hidden inside `params["method"]`. | **Slot + Algorithm Family** (`InitializationRandom`, `InitializationNearestNeighbor`, `MutationSwap`, `MutationInversion`, `LocalSearchCPUTwoOpt`, `LocalSearchGPUTwoOpt`). Both the slot and the mathematical operator are visible in the class name. |
| **Constructor type safety** | Umbrella `__init__` must accept the union of all parameters for every algorithm in that slot (even parameters irrelevant to a specific method). | Each operator-family class accepts **only** the parameters meaningful to that mathematical operator family. |
| **Boilerplate elimination** | Shared inside the umbrella class. | Shared via inheritance from **`StrategyBase`** (`h2d_bytes`, `d2h_bytes`, `kernel_launches`, `reset_metrics()`, `seed`, `rng`). |
| **Where `mode` / `candidate_k` live** | Combined with `method`. | Variations of the **same** operator family (`mode: Literal["best", "first", "multi_move"] = "best"` and `candidate_k: int \| None = None`) are parameters of that family's class! |

#### Why Initialization Is Split into `InitializationRandom` and `InitializationNearestNeighbor`, While Mutation Keeps `candidate_k` Inside `MutationSwap` / `MutationInversion`

- **Initialization (`InitializationRandom` vs `InitializationNearestNeighbor`):**
  + `InitializationRandom` is a zero-distance-knowledge uniform Fisher–Yates permutation generator ($O(n)$ time, maximum initial entropy).
  + `InitializationNearestNeighbor` (GRASP Restricted Candidate List, Feo & Resende, 1995) is a **constructive heuristic** ($O(n \cdot k)$ time) that greedily builds low-cost tours from $\mathcal{N}_k(u)$. In the benchmark database (`benchmark_runs.construction_heuristic`), random vs greedy initialization defines two distinct experimental cohorts.
- **Mutation (`MutationSwap(candidate_k=...)` and `MutationInversion(candidate_k=...)`):**
  + Whether `candidate_k` is `None` (uniform random second index $j$) or an integer $k$ (second index $j = \text{pos}[\mathcal{N}_k[\pi_i, r]]$ chosen from the $k$ nearest neighbors of $\pi_i$), the topological perturbation applied to the tour is still a **2-gene swap** (`MutationSwap`) or a **contiguous segment inversion** (`MutationInversion`). Keeping `candidate_k: int | None = None` as an optional parameter on `MutationSwap` and `MutationInversion` (and sharing `select_proximity_index` in `route_helpers.py`) avoids multiplying the mutation classes while recording `"candidate_k": 20` in `algorithm_config.strategy_params`.

#### Complete `Design B` Class Map (`DP11-1`)

| Slot Prefix | CPU Classes | GPU Classes |
| :--- | :--- | :--- |
| `Initialization*` | `InitializationRandom`, `InitializationNearestNeighbor` | *(both support `return_gpu: bool = False`)* |
| `Selection*` | `SelectionTournament`, `SelectionRouletteWheel` | `SelectionGPUTournament` |
| `Crossover*` | `CrossoverOrder`, `CrossoverPartiallyMapped` | `CrossoverGPUOrder` |
| `Mutation*` | `MutationSwap`, `MutationInversion` | `MutationGPUSwap`, `MutationGPUInversion` |
| `LocalSearch*` | `LocalSearchCPUTwoOpt(mode="best"\|"first"\|"multi_move")`, `LocalSearchNoOp` | `LocalSearchGPUTwoOpt(mode="batch"\|"multi_move"\|"naive")` |
| `Evaluation*` | `EvaluationCPUCost` | `EvaluationGPUCost` |

---

## 2. `ProblemContext` as the Single Owner of Backend Caches and Thresholds

### 2.1 Why `ProblemContext.get_gpu_*` Must Use `cp = require_gpu()` Instead of `self.xp` (`BUG-14`)

In [src/protocols/problem_context.py](../../src/protocols/problem_context.py#L250-L365), `ProblemContext` was originally written with a backend hint `self.xp: BackendModule = np`.

- `get_cpu_distances()` always uses `np` directly, even when `self.xp` is CuPy.
- However, `get_gpu_distances()`, `get_gpu_coordinates()`, `get_gpu_demands()`, and `get_gpu_candidate_list()` checked:

  ```python
  if self.xp == np:
    raise ValueError("get_gpu_distances() requires xp to be CuPy, not NumPy")
  ```

**Why this caused a hidden bug in Hybrid algorithms and `two_opt.py`:**

1. The method name `get_gpu_distances()` is **already explicitly GPU-specific**. Calling `get_gpu_distances()` means the caller is a GPU strategy (`EvaluationGPUCost`, `LocalSearchGPUTwoOpt`) that needs a `cp.ndarray` in VRAM.
2. When a `ProblemContext(problem)` is instantiated with default `xp=np` (for instance, in a hybrid CPU+GPU pipeline or a unit test) and passed to `BatchGPUTwoOpt`, `context.get_gpu_distances()` raised `ValueError`!
3. To work around that `ValueError`, `BatchGPUTwoOpt._get_distances_int32_gpu` ([two_opt.py:432](../../src/algorithms/strategies/local_search/two_opt.py#L432)) bypassed `context.get_gpu_distances()` and allocated its **own** private GPU matrix `cp.asarray(context.get_cpu_distances(), dtype=cp.int32)` keyed only by `problem.name`!
   + **Consequence 1 (Double VRAM allocation):** In `FullGPUModular` and `HybridOptimized`, the $n \times n$ distance matrix was allocated **twice** in VRAM — once inside `context._gpu_distances` (for `GPUCostEvaluation`) and a second time inside `BatchGPUTwoOpt._cached_distances_gpu`!
   + **Consequence 2 (Cache collision):** Two synthetic `Problem` objects with the same `name` but different coordinates/weights collided in `BatchGPUTwoOpt._cached_problem_name`.

**The Structural Fix (`Task 10.14` / `DP10-17`):**

- Every `get_gpu_*` method on `ProblemContext` calls `cp = require_gpu()` directly (never `self.xp`), so `context.get_gpu_distances()`, `context.get_gpu_coordinates()`, and `context.get_gpu_candidate_list(k)` work regardless of whether `context` was initialized with `xp=np` or `xp=cp`.
- `ProblemContext` exposes `context.get_gpu_distance_operands(n, max_matrix_dimension) -> tuple[Any, Any, int]` returning `(d_distances, d_coords, edge_type_code)`, deleting all three duplicated copies of `_get_distances_int32_gpu`, `_get_coords_gpu`, `_distance_operands`, and `_uses_on_the_fly_distances` from `cost_evaluation.py` and `two_opt.py`.

### 2.2 The Gray-Zone Trap: `max_eager_matrix_dimension = 4,000` vs `max_matrix_dimension = 10,000` (`DP11-3`)

Before Phase 11, two different matrix dimension thresholds existed in the codebase:

1. **Loader threshold (`max_eager_matrix_dimension = 4000`):** In [tsplib_loader.py:249](../../src/loaders/tsplib_loader.py#L249) and [database_loader.py:293](../../src/loaders/database_loader.py#L293), coordinate-based problems with $n \le 4{,}000$ eagerly computed `problem.distances` ($n \times n$ `int32` on CPU) during `loader.load()`, while problems with $n > 4{,}000$ left `problem.distances = None`.
2. **GPU/Context threshold (`max_matrix_dimension = 10_000`):** In [problem_context.py:115](../../src/protocols/problem_context.py#L115), `ProblemContext` used dense $(n, n)$ matrices up to $n = 10{,}000$ and only switched to on-the-fly coordinate distances for $n > 10{,}000$.

**What went wrong in the gray zone $4{,}000 < n \le 10{,}000$ (e.g., `rl5915`, `rl5934`, `pla7397`)?**

- For $n = 3{,}795$ (`fl3795`), `DatabaseLoader.load()` computed the dense matrix **before** the benchmark timer started in `run_single_algorithm`.
- For $n = 7{,}397$ (`pla7397`), `DatabaseLoader.load()` skipped matrix computation (`problem.distances = None`), **but** on Generation 1, `ProblemContext` saw $7{,}397 \le 10{,}000$ (dense mode!) and computed the $7{,}397 \times 7{,}397$ matrix ($219\text{ MB}$) **inside the timed evolution loop**, distorting generation-1 timing comparisons!
- Unifying the threshold on `DEFAULT_MAX_MATRIX_DIMENSION = 10_000` (`DP11-3`) eliminates this gray zone.

---

## 3. Three-Tier TSPLIB95 Distance Architecture (`DP10-13` & `DP10-19`)

A common question is: *"Why do we have `pairwise_device.cuh`, `distances/matrix.py`, and `distances/pairwise.py` (`converter.utils.cost`), and does keeping `pairwise.py` as python `math` re-exports hurt performance?"*

The three modules serve **three completely disjoint execution tiers**, so python's `math` module is never called in any hot loop:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: GPU On-the-Fly Registers (`src/distances/kernels/pairwise_device.cuh`)          │
│   - Used when: GPU evaluation / 2-opt with n > 10,000 (or max_matrix_dimension).        │
│   - Execution: 100% compiled CUDA C++ (`__device__ __forceinline__ gpu_pairwise_distance`)│
│   - Memory: O(n) coordinates in L1/L2 cache; ZERO (n, n) matrix allocation.             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Vectorized 2D / Row-Chunked Broadcast (`src/distances/matrix.py`)               │
│   - Used when: Building dense (n, n) matrices (n <= 10,000) or k-NN strips (512 x n).   │
│   - Execution: 100% vectorized C/CUDA array ops via `xp` (`numpy` or `cupy`).           │
│   - Memory: Direct 2D broadcast for n <= 2,048; 512-row strips for n > 2,048 (`BUG-13`).│
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: Scalar Reference & Dispatch Codes (`src/distances/pairwise.py` <- `routing_data`)│
│   - Used when: (a) Mapping `edge_type` string -> integer `GpuEdgeTypeCode` once before  │
│     kernel launch; (b) Ground-truth scalar oracle in unit tests (`tests/distances/`);   │
│     (c) Rare `XRAY1` crystallography fallback.                                          │
│   - Execution: Pure python `math` functions imported from `converter.utils.cost`.       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### The TSPLIB95 `GEO` Degree-Truncation Rule (`int(x)` vs `nint(x)`)

In TSPLIB95 ([spec §2.4](../../documents_tcc/plan/knowledge-base/tsplib95_format.md)), geographical coordinates are written in `DDD.MM` format (for example, `16.47` means $16^\circ\, 47'$). The official C specification extracts degrees via **integer truncation toward zero**:

```cpp
int deg = (int) x[i];
double min = x[i] - deg;
double rad = PI * (deg + 5.0 * min / 3.0) / 180.0;
```

If an implementation mistakenly uses `deg = nint(x[i])` ($\lfloor x + 0.5 \rfloor$) instead of `(int) x[i]`, any coordinate whose minutes are $\ge 0.50$ (such as `52.51` in `burma14`) is rounded up to the next degree ($53^\circ$ with negative minutes $-0.49$), shifting `burma14` distance $d(1, 3)$ from its canonical value **`510`** to **`560`**. Both `routing_data/src/converter/utils/cost.py`, `src/distances/matrix.py` (`xp.trunc`), and `src/distances/kernels/pairwise_device.cuh` (`(int)coord_val`) enforce truncation `(int)x` to guarantee bit-for-bit parity across all three tiers.

---

## 4. Hardcoded Constants Inventory & Checkpoint Provenance (`BUG-17`–`BUG-20`)

The table below summarizes all hardcoded constants audited in `gpu_accelerated_clean/src/` and their resolution across Fase 10 and Fase 11:

| Identifier | Location | Hardcoded Constant | Failure Mechanism | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **`BUG-17`** | [checkpoint_io.py:232, 297](../../src/benchmarking/runner/checkpoint_io.py#L232) | `expected_reps: int = 30` | Calling `is_valid_checkpoint(path)` with default args rejects valid checkpoints from `benchmark.json` (`15` reps) or `selected_10` (`3` reps). | **Task 10.23:** Default `expected_reps: int \| None = None` and validate `successful_runs >= data["repetitions"]` when `None`. |
| **`BUG-18`** | [orchestration.py:465-485](../../src/benchmarking/runner/orchestration.py#L465-L485) | `construction_heuristic = "random_initialization"` and class-name-only `strategies` dict | Discards all strategy hyperparameters (`candidate_k`, `mode`, `max_iterations`, `k`, `return_gpu`) from saved checkpoints and `results.duckdb`. | **Task 10.23 & Task 10.22:** Serialize `strategy_params` per slot into `algorithm_config` and derive `construction_heuristic` dynamically. |
| **`BUG-19`** | [two_opt.py:88](../../src/algorithms/strategies/local_search/two_opt.py#L88), [swap_mutation.py:82](../../src/algorithms/strategies/mutation/swap_mutation.py#L82) | `np.empty(len(tour))` and `route.max() >= route.size` | Raises `IndexError` or silently disables `candidate_k` on CVRP vehicle sub-routes where customer node IDs exceed `len(sub_route) - 1`. | **Task 10.20:** Centralize `inverse_positions` in `route_helpers.py` sized to `max(len(route), int(route.max()) + 1)` with `-1` fill. |
| **`BUG-20`** | [two_opt_multi_move.cu:160](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L160) | `int applied_start[16];` (`pass < 16`) | Caps GPU `multi_move` at 16 disjoint reversals per pass while CPU `multi_move` is uncapped, causing CPU/GPU divergence on large tours ($n > 500$). | **Task 11.1 (`DP11-2`):** Use shared memory (`block_size` entries) for `applied_start`/`applied_end`. |
| **`DP11-3`** | [tsplib_loader.py:249](../../src/loaders/tsplib_loader.py#L249), [database_loader.py:293](../../src/loaders/database_loader.py#L293) | `max_eager_matrix_dimension: int = 4000` vs `10_000` | Creates a gray zone ($4{,}000 < n \le 10{,}000$) where loaders skip matrix creation but `ProblemContext` allocates it inside the timed run. | **Task 11.3 (`DP11-3`):** Unify on `DEFAULT_MAX_MATRIX_DIMENSION = 10_000`. |
| **`DP11-4`** | [config_loader.py:42-160](../../src/benchmarking/runner/config_loader.py#L42-L160) | `KNOWN_OPTIMAL_COSTS` (~110 hardcoded floats) | Duplicates optimal tour costs in python source instead of querying `solutions.cost` in DuckDB or `configs/problems.json`. | **Task 11.3 (`DP11-4`):** Resolve optimal costs from DuckDB `solutions` table and `configs/problems.json`. |
