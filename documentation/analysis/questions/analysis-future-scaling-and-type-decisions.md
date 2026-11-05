---
title: "Analysis of Future Scaling and Type Decisions Across routing_data and gpu_accelerated_clean"
description: >
  Catalog of architectural and data-type decisions that satisfy current TSPLIB95
  and GPU constraints (such as integer distance matrices and dense 2D VRAM layouts)
  but represent known scaling or generalization trade-offs for future problem
  formulations.
created: 2026-11-03
status: draft
author:
  - "[[Lucas Galdino]]"
  - "[[Claude Opus 4.6]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Initial creation mapping future escalation points: integer vs floating-point
          explicit distance matrices (M3), dense 2D vs triangular GPU distance layouts,
          and cross-repository schema boundaries.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[matrix.py](../../../src/distances/matrix.py)"
  - "[two_opt.py](../../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - analysis/architecture
  - analysis/scaling
  - analysis/typing
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - benchmark/database
  - benchmark/gpu
  - guide/architecture
  - notes/planning
  - notes/questions
  - review/quality
---

# Future Scaling and Type Decisions

> [!NOTE]
> This document records architectural compromises that are intentionally locked for the current TSPLIB95 / TCC scope (`DP10-*`), alongside the conditions under which they will need to be revisited when extending the solver to broader routing problem classes.

## 1. Integer (`int32` / `INTEGER[][]`) vs Floating-Point Distance Matrices (`M3`)

### Current Decision (`DP10-12`)

- **Database (`routing_data`)**: `edge_weight_matrices.matrix` is stored as `INTEGER[][]` in DuckDB, strictly following the canonical TSPLIB95 specification (§1.2.8 and §2: all distances and explicit weights are rounded to nearest integers via `nint(x) = int(x + 0.5)`).
- **Solver (`gpu_accelerated_clean`)**: CUDA kernels ([two_opt_batch.cu](../../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu), [two_opt_multi_move.cu](../../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu), [cost_calculator.cu](../../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)) operate on `int32` (`const int* distances`) to avoid floating-point rounding drift (`delta < 0` false positives) and halve shared/global memory bandwidth compared to `float64`.

### Future Escalation Trigger

1. **Extended non-TSPLIB95 benchmarks (Solomon, Homberger, real-world GIS matrices)**: Many CVRPTW and modern VRP benchmarks define Euclidean distances with 1 or 2 decimal places of precision or supply explicit floating-point travel-time matrices without integer rounding.
2. **Overflow risk with fixed-point scaling**: Scaling floats by $10^k$ into `int32` works for moderate coordinates, but long tours or large scaling factors ($10^4$–$10^6$) risk `int32` accumulation overflow above $2^{31} - 1 \approx 2.14 \times 10^9$.

### Locked Decision (`DP10-12`)

- **Everything stays `int32` (`INTEGER[][]` in DuckDB)**:
  + Neither a precision scale factor (`matrix_scale`) nor a `DOUBLE[][]` column will be added to `routing_data` now; all distances follow the TSPLIB95 integer rounding convention (`_nint(x) = int(x + 0.5)`).
  + In `gpu_accelerated_clean`, `DatabaseLoader._load_distance_matrix` ([database_loader.py](../../../src/loaders/database_loader.py#L265)) and `compute_distance_matrix` ([matrix.py](../../../src/distances/matrix.py#L226)) both materialize `Problem.distances` as `int32`, and `ProblemContext.get_gpu_distances()` caches a single `cp.int32` matrix shared across all GPU strategies (`DP10-17`, resolving `BUG-14`).

### Recorded Reference Pattern for Future Non-Rounded Extensions (`template <typename WeightT>`)

If a future solver extension requires unrounded floating-point distances (`float` / `double`), the CUDA kernels can support both `int32` and `float`/`double` without code duplication via C++ function templates compiled by CuPy `RawKernel`:

```cpp
template <typename WeightT, typename AccumT>
__device__ void two_opt_batch_impl(
    int *tours,
    const WeightT *dist,
    int n,
    int num_tours
) {
    // Kernel body identical; AccumT is int for int32 and double for float/double
}

extern "C" __global__ void two_opt_batch_int32(int *tours, const int *dist, int n, int num_tours) {
    two_opt_batch_impl<int, int>(tours, dist, n, num_tours);
}

extern "C" __global__ void two_opt_batch_float64(int *tours, const double *dist, int n, int num_tours) {
    two_opt_batch_impl<double, double>(tours, dist, n, num_tours);
}
```

---

## 2. Dense Square `(n, n)` Matrix vs On-the-Fly Coordinates in GPU VRAM (`BUG-13`, `DP10-19`)

### Current & Planned Architecture (`DP10-19`)

- **For $n \le 10,000$ (or `EXPLICIT` edge weight type)**: Distance matrices are materialized as full `(n, n)` 2D `int32` arrays (`4 * n^2` bytes in VRAM). Full 2D indexing (`distances[i * n + j]`) hits L2 cache for small/medium instances, requires zero math instructions per edge lookup, and supports both symmetric (`TSP`, `CVRP`) and asymmetric (`ATSP`, `SOP`) instances uniformly.
- **For $n > 10,000$ with coordinate-based metrics (`EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`, `CEIL_2D`, `ATT`, `GEO`)**: Materializing a $>400\text{ MB}$ matrix overflows the GPU's $1\text{ MB}$ L2 cache, causing uncached GDDR5 DRAM reads ($\approx 400\text{–}600\text{ cycles}$). Switching automatically to **On-the-Fly Coordinate Evaluation** (`src/distances/kernels/pairwise_device.cuh`) keeps only the `(n, d)` coordinates in VRAM ($\approx 120\text{ KB}$ at $n = 15,112$), which fits entirely in L1/L2 cache ($\approx 20\text{–}150\text{ cycles}$) and computes TSPLIB95-rounded distances in registers.
- **Pre-flight VRAM Check**: `ProblemContext` computes the exact required VRAM $\text{VRAM}_{\text{total}}(n, P) = 4 n^2 + 12 P n + 24 P + 8 n$ against `cp.cuda.Device().mem_info` and raises [VRAMInsufficientError](../../../src/data_models/exceptions.py#L210) when an `EXPLICIT` matrix exceeds available VRAM.
- See [analysis-gpu-distance-and-shared-memory-limits.md](../../knowledge-base/analysis-gpu-distance-and-shared-memory-limits.md) and [analysis-gpu-memory-calculation.md](../../knowledge-base/analysis-gpu-memory-calculation.md) for the full Roofline analysis and academic citations (Rocki & Suda, 2012, 2013; O'Neil & Burtscher, 2015).

---

## 3. Shared Memory Limit (`s_tour`) in GPU 2-Opt Kernels for Large $n$ ($n > 11,520$)

### Hardware Constraint

- In [two_opt.py](../../../src/algorithms/strategies/local_search/two_opt.py#L305-L310), `shared_mem = 3 * block_size * 4 + n * 4` bytes is allocated per thread block so `s_tour` (`n * 4` bytes) resides in fast on-chip shared memory.
- On Pascal (GTX 1050, Compute Capability 6.1), the per-block shared memory limit is $48\text{ KB}$ ($49,152\text{ bytes}$). With `block_size = 256` ($3 \times 256 \times 4 = 3,072\text{ bytes}$), `s_tour` can hold at most $(49,152 - 3,072) / 4 = \mathbf{11,520\text{ cities}}$ before kernel launch fails with `CUDA_ERROR_INVALID_VALUE`.

### Planned Resolution (`DP10-19`: Shared-Memory Tour Tiling)

- When $n \le 11,520$, the full tour `s_tour[n]` is loaded into shared memory as before.
- When $n > 11,520$, each thread loads its outer-loop endpoints `(node_i, node_i1)` into registers (1 coalesced global read per outer step) and the block streams the inner-loop `j` positions through a fixed-size **Shared-Memory Tile** (`s_tour_tile[2049]`, using only $8\text{ KB}$ of shared memory regardless of $n$). Every inner-loop `j` lookup still executes from fast shared memory without exceeding the $48\text{ KB}$ hardware limit.
- See [analysis-gpu-distance-and-shared-memory-limits.md](../../knowledge-base/analysis-gpu-distance-and-shared-memory-limits.md) for the complete tiled kernel walkthrough and $O(n \cdot k)$ candidate-list literature comparison.
