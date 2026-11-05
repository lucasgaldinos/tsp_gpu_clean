---
title: "Knowledge: On-the-Fly GPU Distances and Shared-Memory Tour Tiling"
description: >
  Reusable reference for the two scalability mechanisms added in Tasks 10.17 and 10.18: the automatic switch from a dense `(n, n)` distance matrix to `gpu_pairwise_distance` computed from coordinates, and the tiled tour cache that keeps 2-opt under the 48 KB shared-memory limit for instances above 11,520 cities.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Created from the Task 10.17/10.18 execution: memory formulas, kernel signatures, parity evidence, and the pitfalls found while wiring both mechanisms.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
  - "[walkthrough-10.17-10.18-on-the-fly-and-tiling.md](../../../documentation/walkthroughs/walkthrough-10.17-10.18-on-the-fly-and-tiling.md)"
  - "[pairwise_device.cuh](../../../src/distances/kernels/pairwise_device.cuh)"
  - "[cost_calculator.cu](../../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)"
  - "[two_opt_batch.cu](../../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_multi_move.cu](../../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[two_opt.py](../../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - analysis/gpu
  - algorithm/tsp
  - algorithm/local-search
  - guide/cuda
  - guide/testing
  - notes/performance
  - analysis/memory
  - analysis/numerics
  - guide/architecture
  - analysis/scalability
---

# Knowledge: On-the-Fly GPU Distances and Shared-Memory Tour Tiling

## 1. Why the dense matrix stops scaling

A dense `int32` matrix costs $4n^2$ bytes. At $n = 15{,}112$ (`d15112`) that is $\approx 913$ MB; at $n = 33{,}810$ (`pla33810`) it is $\approx 4.57$ GB, before any population, offspring, or coordinate buffer is allocated. The on-the-fly path replaces that term with $12Pn + 24P + 8n$ (population plus coordinates), which grows linearly.

## 2. Dispatch contract

`gpu_pairwise_distance(u, v, dist, coords, n, edge_type_code)` is the only entry point:

- `dist != nullptr` → `dist[u * n + v]` (dense fast path);
- `dist == nullptr` → one of the 9 TSPLIB95 formulas on `coords`, selected by `edge_type_code` from `distances.pairwise.EDGE_TYPE_TO_CODE`.

python passes `0` for a null pointer; CuPy converts both `None` and `0` to `nullptr`, which is verified by the probe:

```python
import cupy as cp

src = r'''
extern "C" __global__ void probe(const int* p, int* out) { out[0] = (p == 0) ? 7 : 9; }
'''
k = cp.RawKernel(src, "probe")
out = cp.zeros(1, dtype=cp.int32)
k((1,), (1,), (0, out))  # out[0] == 7
```

The switch is centralized in `ProblemContext.uses_on_the_fly_distances(n, threshold)`:

```python
return (
  n > threshold
  and self.problem.edge_type != "EXPLICIT"
  and self.problem.coordinates is not None
)
```

`threshold` defaults to `max_matrix_dimension = 10_000`, overridable on the context or on each strategy (tests use `10`).

## 3. Shared-memory budget and tiling

The full-tour buffer is `4n` bytes plus three `block_size` int arrays:

$$
\text{bytes}(n) = 4n + 12 \cdot \text{block size}
$$

With `block_size = 256`, the 48 KB hardware limit is reached at $n = 11{,}520$: $46{,}080 + 3{,}072 = 49{,}152 = 48$ KiB. Above it the kernel switches to one tile:

```cuda
#define TWO_OPT_TILE_SIZE 2048
int *s_tour = shared_mem;                       // [TWO_OPT_TILE_SIZE + 1]
int *s_deltas = tiled ? &s_tour[tour_tile_size + 1] : &s_tour[n];
```

The outer loop loads `[tile_start, tile_end_exclusive)` cooperatively, then every thread scans its rows inside the window. `TOUR_AT` falls back to global memory for indices outside the window, and `TOUR_SET` writes through to global memory in tiled mode, so the block's global tour stays authoritative.

The tile width (2048) is a multiple of every supported power-of-two block size, so the row-to-thread mapping `i % block_size` is identical in both modes. That is what makes tiled and untiled results bit-identical instead of merely both valid.

## 4. Pitfalls found during execution

| Symptom | Cause | Fix |
| --- | --- | --- |
| `nvrtc: identifier "tile_start" is undefined` | `tile_start` declared in the `for` init, out of scope for the apply-phase macros | declare `int tile_start = 0;` before the loop |
| Strategy test asserted `context._gpu_distances` | `BatchGPUTwoOpt` caches the matrix on the strategy, not on `ProblemContext` | assert `strategy._cached_distances_gpu` / `_cached_coords_gpu` |
| Coordinate mode silently allocating the matrix | calling `_get_distances_int32_gpu` before the mode check | build `(d_distances, d_coords, code)` through one `_distance_operands` helper |

## 5. Evidence

```text
.venv/bin/python -m pytest tests/distances tests/algorithms/strategies -q
273 passed

.venv/bin/python -m mypy src tests
Success: no issues found in 116 source files

.venv/bin/python -m ruff check src tests
All checks passed!
```

Parity is asserted bit-for-bit (`np.testing.assert_array_equal`) on the int64 costs and on the improved tours for `EUC_2D` (`berlin52`) and `GEO` (`burma14`), and on a synthetic 2,100-city tour that spans two 2048-wide tiles.
