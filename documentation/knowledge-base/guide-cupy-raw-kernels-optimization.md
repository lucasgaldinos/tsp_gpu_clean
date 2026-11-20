---
title: "CuPy Raw Kernels for GPU Combinatorial Optimization"
description: >
  Practical guide to writing, compiling, launching, and profiling CuPy RawKernel
  objects in this repository: choosing between RawKernel, ReductionKernel and
  ElementwiseKernel, the shared-memory and launch contract used by the 2-opt kernels,
  the optimizations that changed the measured wall clock, and the failure modes that
  corrupt results without raising an error.
created: 2025-01-20
status: review
author:
  - "Lucas Galdino"
  - "GitHub Copilot"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Standardized document per markdown instructions: added formal YAML frontmatter,
          corrected malformed fence tags, and aligned kernel links.
  - date_modified: 2026-11-19
    modifications:
      - description: >
          Full rewrite. The file had accumulated an unedited chat transcript: prompt text,
          literal "Code snippet" labels, a tool truncation notice, three overlapping analyses
          of the same heuristics, and a section duplicated verbatim. Removed the transcript
          scaffolding, restored the math that escape handling had swallowed (Delta cost,
          distance-matrix indices, logical OR inside CUDA code), replaced the line-wrapped
          ASCII diagrams with ones that render, condensed the three parallel-taxonomy passes
          into a single corrected section, and grounded every optimization in the kernel that
          implements it. Frontmatter links and tags regenerated.
related_files:
  - "[guide-cuda-synchronization-levels.md](./guide-cuda-synchronization-levels.md)"
  - "[guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[analysis-gpu-distance-and-shared-memory-limits.md](./analysis-gpu-distance-and-shared-memory-limits.md)"
  - "[analysis-gpu-reduction-patterns.md](./analysis-gpu-reduction-patterns.md)"
  - "[gpu_helpers.py](../../src/utils/gpu_helpers.py)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
tags:
  - guide/cuda
  - guide/performance-tuning
  - guide/benchmarking
  - algorithm/cuda
  - algorithm/two-opt
  - analysis/gpu
  - analysis/profiling
  - notes/cuda
  - notes/optimization
  - notes/shared-memory
  - notes/coalescing
  - review/quality
---

# CuPy Raw Kernels for GPU Combinatorial Optimization

- [CuPy Raw Kernels for GPU Combinatorial Optimization](#cupy-raw-kernels-for-gpu-combinatorial-optimization)
  + [Scope and audience](#scope-and-audience)
  + [Choosing the right kernel class](#choosing-the-right-kernel-class)
  + [Compilation and launch contract](#compilation-and-launch-contract)
  + [Anatomy of a launch](#anatomy-of-a-launch)
  + [Optimization playbook](#optimization-playbook)
    - [1. One block per independent work item](#1-one-block-per-independent-work-item)
    - [2. Cache the hot array in shared memory](#2-cache-the-hot-array-in-shared-memory)
    - [3. Enumerate the upper triangle and hoist the row](#3-enumerate-the-upper-triangle-and-hoist-the-row)
    - [4. Coalesce with block-stride loops](#4-coalesce-with-block-stride-loops)
    - [5. Reduce inside the warp first](#5-reduce-inside-the-warp-first)
    - [6. Bound the neighbourhood with candidate lists](#6-bound-the-neighbourhood-with-candidate-lists)
    - [7. Never synchronize the stream per call](#7-never-synchronize-the-stream-per-call)
  + [Tiling past the shared memory ceiling](#tiling-past-the-shared-memory-ceiling)
  + [Coordinate mode without a distance matrix](#coordinate-mode-without-a-distance-matrix)
  + [Failure modes that corrupt results silently](#failure-modes-that-corrupt-results-silently)
    - [Stale JIT binaries](#stale-jit-binaries)
    - [Zero-filled warm-up](#zero-filled-warm-up)
    - [Silent fallback to CPU](#silent-fallback-to-cpu)
    - [Host-side serial sampling](#host-side-serial-sampling)
  + [What to measure](#what-to-measure)
  + [Where the parallelism lives](#where-the-parallelism-lives)
    - [Source of parallelism](#source-of-parallelism)
    - [Strategy of parallelism](#strategy-of-parallelism)
    - [Applying both to this project](#applying-both-to-this-project)
  + [References](#references)

## Scope and audience

This is the working guide for the hand-written CUDA layer of the solver: the `.cu` and
`.cuh` files under `src/algorithms/strategies/**/kernels/` and `src/distances/kernels/`,
compiled at run time through CuPy. It assumes you already know what a grid, a block and a
warp are, and it is organised around the questions that actually come up while editing one
of those files: which CuPy class to reach for, how the launch is wired, what the hardware
budget is, which optimizations pay, and which mistakes produce plausible numbers that are
wrong.

The running example is [`two_opt_batch.cu::two_opt_batch_kernel`](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu),
because it exercises every technique in the guide at once: one block per independent route,
a shared-memory cache with a capacity fallback, an upper-triangle scan, a block reduction,
an optional candidate table, and a second mode that computes distances on the fly instead of
reading a matrix.

## Choosing the right kernel class

CuPy exposes three levels of control. They are not interchangeable, and the difference that
matters is not "abstraction" in the abstract but whether you need threads to *cooperate*.

| Capability | `ElementwiseKernel` | `ReductionKernel` | `RawKernel` |
| --- | --- | --- | --- |
| What you write | body of one iteration | map and reduce expressions | the whole `__global__` function |
| Thread and block indices | hidden | hidden | explicit |
| Arbitrary index arithmetic | no | no | yes |
| Shared memory | no | no | yes |
| `__syncthreads()` | no | no | yes |
| Grid-stride loop | emitted by CuPy | emitted by CuPy | you write it |
| Reduction scope | none | whole input | per block, as designed |
| Device header via `-I` | limited | limited | yes |
| Fits a 2-opt pass | no | no | yes |

`ElementwiseKernel` is a template for a one-to-one map. You supply a C++ expression and CuPy
generates the kernel wrapper, including the loop that walks the output indices, so

```python
import cupy as cp

squared = cp.ElementwiseKernel(
    "T x, T y",
    "T z",
    "z = (x - y) * (x - y)",
    "squared_difference",
)
out = squared(cp.arange(4, dtype=cp.float32), cp.ones(4, dtype=cp.float32))
# out -> [1. 0. 1. 4.]
```

is one launch, not four. The class supports `raw` arguments for pointer-style access, which
is what makes it useful for lookups into a precomputed table. What it cannot express is
cooperation: there is no way for element `i` to see what element `j` computed.

That limitation is decisive here. A 2-opt pass is a reduction from $n^2$ candidate moves to
one winning move plus its two indices, and it is a *per-route* reduction: with a batch of
256 tours the launch must produce 256 independent minima, not one global minimum.
`ReductionKernel` reduces its entire input to a single scalar, so it cannot serve the batch,
and `ElementwiseKernel` cannot carry the argmin payload `(i, j)` out of the block. Emulating
either with an intermediate `(n, n)` delta matrix reintroduces exactly the quadratic memory
that the dense distance matrix already strains against. The 2-opt family therefore uses
`RawKernel`, and so does every other operator in this repository that needs shared memory or
a block barrier.

> [!NOTE]
> A GPU is **SIMT** (single instruction, multiple threads), not plain SIMD. The distinction
> matters when reading the taxonomy section below: a warp executes one instruction stream and
> masks divergent lanes, so data-parallel code that maps one lane to one data element runs at
> full rate, while code where lanes in the same warp take different branches — a mutation
> applied to only 10% of routes, for instance — runs at the rate of the slowest path and
> leaves the rest of the warp idle.

## Compilation and launch contract

Kernels are compiled at run time from source files by
[`gpu_helpers.py::load_kernel`](../../src/utils/gpu_helpers.py), which resolves the file
relative to the caller, compiles it once, and hands back a `cp.RawKernel`:

```python
from utils.gpu_helpers import load_kernel, prev_power_of_2

kernel = load_kernel("two_opt_batch.cu", "two_opt_batch_kernel", __file__)
```

Four properties of that path are worth knowing before editing a kernel:

- **Search order.** `load_kernel` looks in `<caller_dir>/kernels/`, then
  `<caller_dir>/../kernels/`, then `<caller_dir>/../../kernels/`, and raises
  `FileNotFoundError` naming all three when the file is missing.
- **The cache is invalidated by content, not just by path.** Compilation is memoised on the
  source path, which is what keeps it cheap, but a path says nothing about the file. A digest
  guard clears the cache when the `.cu` changes, so editing a kernel inside a long-lived
  process cannot leave it running a binary built from the previous revision.
- **The include path is injected.** Every kernel is built with
  `-I<src/distances/kernels>`, so `#include "pairwise_device.cuh"` and
  `#include "two_opt_common.cuh"` resolve without copying headers next to the `.cu`.
- **Entry points are `extern "C"`.** `cp.RawKernel` resolves the function by its symbol name;
  without the `extern "C"` linkage the name is C++-mangled and the lookup fails.

```cpp
#include "two_opt_common.cuh"

extern "C" __global__
void two_opt_batch_kernel(
    int *tours,             // flattened `num_tours * n`
    const int *dist,        // `n * n` int32 matrix, or nullptr in coordinate mode
    const double *coords,   // row-major coordinates, ignored when `dist != nullptr`
    int n,                  // nodes per tour
    int num_tours,          // independent routes in the batch
    int edge_type_code,     // `GpuEdgeTypeCode`, used only in coordinate mode
    int tour_tile_size,     // 0 = whole tour in shared memory, else tile width
    const int *candidates,  // `n * candidate_k` neighbour table, or nullptr
    int candidate_k,
    int *positions)         // `(batch, n)` scratch for the inverse tour map
{
    // ...
}
```

The GPU backend is obtained through `require_gpu()`, never through a guarded `import cupy`
that falls back to NumPy. A machine without a device must fail at the entry point: a GPU
variant that silently executes on the host records NumPy timings under a GPU label, and the
comparison it feeds is then meaningless.

## Anatomy of a launch

Three numbers decide how the kernel runs: the grid, the block, and the dynamic shared
memory. The batched 2-opt launch sets them like this:

```python
kernel(
    (batch_size,),      # grid: one block per independent tour
    (block_size,),      # block: threads cooperating on that tour
    (
        d_population, d_distances, d_coords,
        np.int32(n), np.int32(batch_size), np.int32(edge_type_code),
        np.int32(tile_size), d_candidates, np.int32(candidate_count), d_positions,
    ),
    shared_mem=shared_mem,
)
```

- **Grid = number of independent work items.** One block owns one route, so no block ever
  needs to talk to another; there is no inter-block barrier anywhere in the design.
- **Block = `prev_power_of_2(min(256, n - 2), max_val=256)`.** A power of two keeps the
  reduction tree balanced, and clamping to `n - 2` avoids launching threads that would find
  no `i` to work on. Above 256 the extra threads cost occupancy without adding parallelism
  inside a single block.
- **Dynamic shared memory must be requested explicitly.** Static `__shared__` arrays are not
  enough when the size depends on `n`; the byte count travels in `shared_mem=`.

The byte count itself is where the hard ceiling lives. With three `block_size` int buffers
for the reduction and the tour cached separately:

```text
budget per block                     49 152 B   (48 KiB)
- reduction buffers   3 * 256 * 4   - 3 072 B
= available for tour                 46 080 B
/ 4 B per position                  = 11 520 positions
```

That division is the origin of the constant `max_shared_tour_cities = 11_520`: it is not a
tuned value, it is the largest tour that still fits next to the reduction buffers at
`block_size = 256`. Changing the block size or adding a shared buffer moves it, and the
kernel must then switch to the tiled path described below.

## Optimization playbook

Each item below pairs the change with the reason it works and with a check that fails if the
change is undone.

### 1. One block per independent work item

Batching 256 tours into a single launch with `grid = (256,)` replaces 256 launches with one.
The saving is not the launch API call itself but the driver work behind it, which is charged
per launch and per iteration; a 10-iteration pass over 256 tours costs 10 launches instead of
2 560.

*Check:* the strategy's `kernel_launches` counter increments once per pass over the batch,
not once per route.

### 2. Cache the hot array in shared memory

Every thread in the block reads tour positions, most of them many times. Copying the tour
into `s_tour` once per launch turns those reads from global-memory transactions into
shared-memory reads, and the copy itself is one coalesced load.

The accessor contract is deliberately macro-based, because the tile fallback has to be
invisible to the scan:

```cpp
// tiled == false: the whole tour is in s_tour[n]
// tiled == true : only one window is cached; out-of-window reads go to global memory
#define TOUR_AT(idx)  (tiled ? (in_window(idx) ? s_tour[(idx) - tile_start] : tour[(idx)]) : s_tour[(idx)])
```

A kernel that includes `two_opt_common.cuh` must declare `tiled`, `tile_start`, `tile_end`,
`s_tour` and `tour` before the first use, because the macros read the caller's locals.

*Check:* the reserved bytes stay within the budget computed above; a tour that would exceed
it must take the tiled path, not silently overrun.

### 3. Enumerate the upper triangle and hoist the row

Two exact transforms remove work without changing a single delta:

```cpp
for (int i = tid; i < n - 2; i += block_size) {
    int node_i  = s_tour[i];
    int node_i1 = s_tour[i + 1];
    const int *row_i  = dist + (size_t)node_i  * (size_t)n;
    const int *row_i1 = dist + (size_t)node_i1 * (size_t)n;
    int d_ab = row_i[node_i1];
    int node_j = s_tour[i + 2];
    for (int j = i + 2; j < n; j++) {
        int node_j1 = s_tour[(j + 1) % n];
        if (i == 0 && j == n - 1) { node_j = node_j1; continue; }  // identity move
        // ... evaluate, then carry over: node_j = node_j1
        node_j = node_j1;
    }
}
```

- **Upper triangle only.** A 2-opt move is symmetric: the pair `(i, j)` breaks the same two
  edges as `(j, i)`, so scanning `j < i + 2` re-evaluates pairs already seen. Half the
  comparisons disappear.
- **`(0, n - 1)` is excluded.** In a cyclic tour that pair reverses the whole route, which
  reproduces the same tour rotated; it is the identity, not a move.
- **The row is hoisted.** `node_i`, `node_i1`, `d_ab` and the two distance-matrix rows do not
  depend on `j`, so they are loaded once per `i` instead of once per pair.
- **The node is carried over.** `tour[j + 1]` of one iteration is `tour[j]` of the next, so
  one read per iteration is replaced by a register copy.

*Check:* the tour produced on a small instance is identical to the exhaustive reference; the
optimization is exact, so any difference is a bug rather than a trade-off.

### 4. Coalesce with block-stride loops

Partitioning a range into one contiguous slice per thread looks natural and reads badly:

```cpp
// avoid: adjacent lanes touch addresses `chunk` apart
int start = tid * chunk;
for (int i = start; i < start + chunk; i++) { /* ... */ }

// prefer: adjacent lanes touch adjacent addresses
for (int i = tid; i < n; i += blockDim.x) { /* ... */ }
```

With the block-stride form, the 32 lanes of a warp read 32 consecutive positions in one
transaction. With the chunked form they read 32 addresses separated by `chunk` elements,
fragmenting the request into as many transactions as there are lanes. The evaluation kernel
`cost_calculator.cu` uses the block-stride form for exactly this reason.

*Check:* the stride in the loop increment is `blockDim.x` (or `block_size`), never a
per-thread range.

### 5. Reduce inside the warp first

Finding the best move is a minimum over `block_size` partial results. Doing it in shared
memory costs `log2(block_size)` full-block barriers. Folding within the warp first, using
`__shfl_down_sync`, costs register shuffles and no barrier; only one partial per warp then
reaches shared memory, where thread 0 merges at most `min(block_size, 64) - 1` entries.

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    partial = min(partial, __shfl_down_sync(0xFFFFFFFFu, partial, offset));
```

Two further shortcuts are worth keeping: the merge scan is skipped entirely when the current
best delta is already non-negative, because the pass will apply no move.

*Check:* the kernel's shared-memory usage does not grow with `log2(block_size)`.

### 6. Bound the neighbourhood with candidate lists

The exhaustive scan is $\Theta(n^2)$ per pass. A $k$-nearest-neighbour table
$\mathcal{N}_k(u)$ turns it into $\Theta(n \cdot k)$, which for $n = 10^4$ and $k = 20$ is
three orders of magnitude fewer evaluations. The kernel needs the inverse map to translate a
neighbour's node id into a tour position:

```cpp
// pos[tour[idx]] = idx, built once per pass in O(n)
for (int idx = tid; idx < n; idx += block_size)
    pos[tour[idx]] = idx;

for (int r = 0; r < candidate_k; ++r) {
    int j = pos[candidates[(size_t)node_i * (size_t)candidate_k + (size_t)r]];
    if (j <= i + 1 || (i == 0 && j == n - 1)) continue;
    // ...
}
```

Three constraints come with it, and each is enforced on the host before the launch:

- The inverse map needs a dense permutation over `0 .. n-1`. Sparse CVRP sub-routes, whose
  node ids exceed `len(route) - 1`, keep the exhaustive scan.
- The table must already be resident in VRAM. Binding it once per problem and reusing the
  device buffer across generations avoids re-uploading it on every pass.
- Candidate mode and tiled mode are mutually exclusive. A tiled block already streams the
  tour through shared memory, so an extra scratch buffer would compete for the same budget
  for no gain.

*Check:* with `candidate_k` set, `d2h_bytes` stays at zero across generations and the
`n x n` matrix is never materialised on the host to build the table.

### 7. Never synchronize the stream per call

A host-side `cp.cuda.Stream.null.synchronize()` after every kernel forces the CPU to wait for
the device, destroying the only overlap the pipeline has. CuPy operations are stream-ordered,
and the copy that actually needs the result — `array.get()` — synchronizes implicitly. The
same reasoning applies inside instrumentation: a timing decorator that records a CUDA event
and calls `event.synchronize()` on every invocation reintroduces one barrier per strategy per
generation, which is five or six per generation in a fully GPU pipeline.

*Check:* no unconditional `Stream.null.synchronize()` in the strategy; device-event timing is
opt-in per strategy, not the default.

## Tiling past the shared memory ceiling

Above `max_shared_tour_cities` the whole tour no longer fits, so the kernel caches one window
and streams the rest:

```text
n <= 11 520                        n > 11 520
+--------------------------+       +--------------------------+
| s_tour[0 .. n-1]         |       | s_tour[0 .. 2048] window |
| loaded once per launch   |       | reloaded as the scan moves|
+--------------------------+       +--------------------------+
                                   positions outside the window
                                   are read from global memory
```

`TOUR_AT` and `TOUR_SET` hide the difference: reads inside the window hit shared memory,
reads outside it fall back to the global tour, and writes go to both. The window is
`TWO_OPT_TILE_SIZE + 1 = 2049` positions — the extra slot lets the inner loop read
`tour[j + 1]` without a branch at the window boundary.

Tiling is a correctness ceiling, not a speedup. It keeps the launch legal for instance sizes
that would otherwise exceed the per-block budget; the out-of-window reads are slower than the
cached path by construction, so a tiled run should be reported as such rather than compared
against an untiled one as if they were the same kernel.

## Coordinate mode without a distance matrix

The dense `(n, n)` int32 matrix costs $4n^2$ bytes and is materialised in VRAM only up to
`max_matrix_dimension` (10 000 by default). Above it, the launch passes `dist = nullptr` and
the kernel computes each edge on the fly from the coordinates:

```cpp
int delta = use_matrix
    ? two_opt_delta(a, b, d_ab, c, d, row_a, row_b, dist, coords, n, edge_type_code, true)
    : gpu_pairwise_distance(a, c, nullptr, coords, n, edge_type_code)
      + gpu_pairwise_distance(b, d, nullptr, coords, n, edge_type_code)
      - d_ab - gpu_pairwise_distance(c, d, nullptr, coords, n, edge_type_code);
```

The device functions in `pairwise_device.cuh` must reproduce the host arithmetic exactly,
including the TSPLIB95 integer rounding of Euclidean distances and the degree truncation of
geographic coordinates. A one-unit difference between the two paths changes which move wins,
so the two modes are covered by parity tests that compare costs and resulting tours rather
than timing.

## Failure modes that corrupt results silently

None of these raises an exception. Each produces numbers that look reasonable and are wrong.

### Stale JIT binaries

The compilation cache is keyed by source path. Without the content digest guard described
above, editing a `.cu` inside a running benchmark keeps executing the previous binary, and the
collected timings describe code that is no longer in the tree.

*Guard:* the digest check in `load_kernel`, which clears the cache when the file changes.

### Zero-filled warm-up

Warm-up batches are often built with `cp.zeros((batch, n), dtype=cp.int32)`. Any component
that infers route topology from the data — for instance "closed if `route[0] == route[-1]`" —
sees `0 == 0` and permanently marks the instance as closed, so later open tours get their
endpoints frozen and the search silently loses two positions of freedom per route.

*Guard:* warm up with valid permutations, `cp.arange(n, dtype=cp.int32)`, and never infer
topology from a synthetic buffer.

### Silent fallback to CPU

An `import cupy` inside `try/except ImportError` with a NumPy fallback turns a missing driver
into a CPU run reported under a GPU variant name. The failure is invisible in the results and
poisons every comparison built on them.

*Guard:* `require_gpu()`, which raises instead of degrading.

### Host-side serial sampling

Drawing one random pair per route in a serial host loop costs microseconds per route and
blocks the host while the device is idle:

```python
# avoid: one interpreter round-trip per route, and one blocking draw per iteration
for route_index in range(num_routes):
    left, right = rng.choice(np.arange(low, high), size=2, replace=False)

# prefer: one vectorised draw, then a single transfer
pairs = sample_distinct_pairs(rng, num_pairs=num_routes, low=low, high=high)
```

The rule generalises: anything that runs once per route on the host belongs in the kernel or
in a batched array operation. The same applies to `int(array.max())` inside a hot loop, which
forces a device-to-host synchronization per generation just to learn a value the host already
knows from the problem metadata.

*Guard:* the host path performs a constant number of draws per batch, not one per route.

## What to measure

The counters that make the above verifiable live on the strategy base class and are reset per
run:

| Counter | What it answers |
| --- | --- |
| `kernel_launches` | is the batching actually batching? |
| `h2d_bytes` | is the same table being uploaded every generation? |
| `d2h_bytes` | is a result crossing the bus when nobody asked for it? |

Alongside them, per-phase timings attribute wall clock to initialization, selection,
crossover, mutation, evaluation, local search and the host-side synchronization bucket. Two
habits keep those numbers honest: run the compilation and allocation outside the measured
loop, so JIT cost does not land in repetition 1, and assert that the compilation cache size is
unchanged when the loop ends — a cache miss inside the loop means a kernel compiled during
measurement.

## Where the parallelism lives

Two classifications are often conflated in the literature and in review comments. They answer
different questions and compose rather than compete.

### Source of parallelism

This axis says *what* is being executed in parallel, and it is the older of the two.

- **S-Data** — one stream of data, processed in order.
- **P-Data** — the same instruction over many data elements. Evaluating every 2-opt move, or
  the fitness of every individual in a population, is P-Data.
- **S-Task** — the algorithm is a pipeline of stages with a dependency between them: generate,
  then evaluate, then accept. A hill-climb loop and a generational loop are both S-Task.
- **P-Task** — different instruction streams run concurrently and usually cooperate. A tabu
  search cooperating with a genetic algorithm is P-Task, and it needs MIMD hardware to be
  worth anything; on a GPU, lanes in a warp share one instruction stream, so P-Task is limited
  to block-level granularity or to separate kernel launches.

The practically important case is the composition of the first two, which is what almost every
metaheuristic on a GPU looks like:

```text
S-Task: the hill climb is sequential
+-------------------------------------------------------------------+
|  find best move   ->   apply move   ->   repeat until no improve   |
+-------------------------------------------------------------------+
              |
              |  P-Data: the "find" step is one thread per (i, j) pair
              v
+-------------------------------------------------------------------+
|  lane 0: (0,2)   lane 1: (0,3)   lane 2: (0,4)  ...  (n-3, n-1)    |
|  each lane computes delta(i, j) independently, no communication    |
+-------------------------------------------------------------------+
              |
              v
  block reduction: the smallest delta and its (i, j)  ->  one move per pass
```

### Strategy of parallelism

This axis says *how* the parallel algorithm is structured, and it is the one the
parallel-metaheuristics literature usually means by "type":

- **Type 1, inner-loop parallelism.** Parallelise one expensive step inside the sequential
  loop. The search path is unchanged; it is executed faster. This is the only type the kernel
  layer implements.
- **Type 2, domain decomposition.** Partition the problem's data, solve each part, combine.
  Rewrites the search rather than accelerating it.
- **Type 3, multiple searches.** Run several complete searches at once, either independent
  (multi-start, best result wins) or cooperative (populations or temperatures exchange
  solutions). Cooperative variants are P-Task by construction.

The two axes line up as follows: Type 1 exploits P-Data inside an S-Task algorithm, Type 3
independent is P-Data at the level of whole searches, and Type 3 cooperative is P-Task. The
three-type scheme is coarser than the data/task scheme and describes the design decision,
which is why it is the more useful vocabulary when arguing about *where to spend engineering
effort* rather than *what the hardware does*.

### Applying both to this project

| Heuristic | Source of parallelism | Strategy | State in this repository |
| --- | --- | --- | --- |
| 2-opt hill climb | S-Task loop around a P-Data evaluation | Type 1 | `LocalSearchCPUTwoOpt`, `LocalSearchGPUTwoOpt` |
| Genetic algorithm | S-Task generation loop around P-Data evaluation | Type 1 | `FullGPUModular` keeps the population in VRAM and runs the operators as kernels |
| Multi-start SA or GA | P-Data, independent instances | Type 3 independent | supported by the batched kernel API, not used by the benchmark |
| Island-model GA | P-Task, cooperating populations | Type 3 cooperative | not implemented |

The practical consequence is that the kernel work in this repository is all Type 1, and that
its ceiling is Amdahl's law applied to the generation: parallelising crossover can only help
in proportion to the share of the generation crossover occupies. Once the batched 2-opt
dominates the profile, further kernel tuning on the other operators buys almost nothing,
which is why the optimization effort belongs on local search and on the memory layout it
reads.

## References

- Crainic, T. G., and Toulouse, M. *Parallel Strategies for Meta-Heuristics.* In Handbook of
  Metaheuristics, Kluwer, 2003. Source of the Type 1 / Type 2 / Type 3 strategy classification.
- Fujimoto, N., and Tsutsui, S. *A Highly-Parallel TSP Solver for a GPU Computing Platform.*
  ICNAAM, vol. 1389, 2011. One CUDA block per tour with the tour cached in shared memory.
- Rocki, K., and Suda, R. *Accelerating 2-opt and 3-opt Local Search Using GPU in the
  Travelling Salesman Problem.* IEEE HPCS, 2012. Thread-cooperative move evaluation and
  shared-memory reduction.
- O'Neil, M. A., and Burtscher, M. *Rethinking the Parallelization of Random-Restart Hill
  Climbing: A Case Study in Optimizing a 2-Opt TSP Solver for GPU Execution.* GPGPU-8, 2015.
