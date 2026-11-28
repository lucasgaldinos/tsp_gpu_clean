---
title: "GPU Multi-Move Conflict Resolution for 2-Opt and 3-Opt Local Search"
description: >
  Architectural deep dive into Thread-0 greedy disjoint multi-move conflict
  resolution shared across `two_opt_multi_move.cu` and `three_opt_multi_move.cu`:
  the `!(tk < mi || mk < ti)` non-overlap invariant, `tour[0]` anchor invariance,
  O(block_size) bounded candidate retention versus O(n^2) CPU bookkeeping,
  cooperative block-level segment reversal (`BUG-29`, `2OPT-A`), and empirical
  convergence speedups across both 2-opt and 3-opt benchmarks.
created: 2026-11-24
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-24
    modifications:
      - description: >
          Initial creation: documented the shared Thread-0 greedy disjoint
          multi-move conflict-resolution architecture across `two_opt_multi_move.cu`
          and `three_opt_multi_move.cu`, proved the `!(tk < mi || mk < ti)`
          disjointness and `tour[0]` invariance contract, contrasted GPU
          O(block_size) retention against CPU O(n^2) sorting, and synthesized
          empirical 2-opt and 3-opt evidence.
related_files:
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_common.cuh](../../src/distances/kernels/two_opt_common.cuh)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[three_opt_multi_move.cu](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/kernels/three_opt_multi_move.cu)"
  - "[three_opt_common.cuh](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/kernels/three_opt_common.cuh)"
  - "[gpu_three_opt.py](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/gpu_three_opt.py)"
  - "[knowledge-three-opt-gpu-local-search.md](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/docs/knowledge-three-opt-gpu-local-search.md)"
  - "[analysis-gpu-two-opt-dispatch-and-reduction.md](./analysis-gpu-two-opt-dispatch-and-reduction.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
tags:
  - algorithm/two-opt
  - algorithm/three-opt
  - algorithm/tsp
  - analysis/architecture
  - analysis/performance
  - benchmark/gpu
  - guide/cuda
  - notes/cuda
  - notes/hardware
  - review/quality
---

# GPU Multi-Move Conflict Resolution for 2-Opt and 3-Opt Local Search

Single-move `batch` local search spends an entire $\Theta(n^2)$ or $\Theta(n k^2)$ parallel neighborhood scan only to apply **one** edge-exchange move per pass. On tours with hundreds of cities, applying one move per pass creates a serial convergence bottleneck: the GPU computes thousands of improving moves across disjoint regions of the tour and discards all but one. **Multi-move conflict resolution** (Rocki & Suda 2012, 2013; O'Neil & Burtscher 2015) harvests multiple mutually independent improving moves from a single scan before launching the next pass, driving both 2-opt and 3-opt to tighter tours in $4.7\times\text{–}7.4\times$ fewer generations.

## 1. Per-Thread Candidate Compression: $\mathcal{O}(\text{block\_size})$ vs. $\mathcal{O}(n^2)$

Exhaustive CPU multi-move ([two_opt.py::LocalSearchCPUTourTwoOpt._step_multi_move](../../src/algorithms/strategies/local_search/two_opt.py)) materializes every improving pair $(\Delta < 0)$, sorts all $\mathcal{O}(n^2)$ (or $\mathcal{O}(nk)$) candidates by gain, and greedily marks an `occupied[n]` bitmask. On GPU, allocating $\mathcal{O}(n^2)$ shared-memory buffers and running a block-wide sort per tour would exceed the 48 KB per-SM SRAM ceiling and destroy occupancy.

Instead, both [two_opt_multi_move.cu::two_opt_multi_move_batch_kernel](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu) and [three_opt_multi_move.cu::three_opt_multi_move_batch_kernel](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/kernels/three_opt_multi_move.cu) bound per-pass candidate storage to **one slot per thread**:

$$\text{Candidates per pass} \le \min(\text{block\_size},\; \lfloor n / 2 \rfloor), \qquad \text{block\_size} \le 256$$

During the parallel scan, each thread maintains only its own steepest move (`s_deltas[tid]`, plus cut coordinates `(s_swap_i[tid], s_swap_j[tid])` for 2-opt or `(s_i[tid], s_j[tid], s_k[tid], s_case[tid])` for 3-opt). That deliberate trade-off caps shared-memory selection state at **$3 \cdot B \cdot 4 = 3\text{ KB}$** (2-opt) or **$5 \cdot B \cdot 4 = 5\text{ KB}$** (3-opt) for $B = 256$, while preserving deterministic conflict resolution across every tour in the grid.

## 2. The Overlap-Invalidation Invariant and `tour[0]` Anchor Safety

Once all threads complete the neighborhood scan and reach `__syncthreads()`, Thread 0 iteratively picks the steepest remaining move (`s_deltas[best_t] < 0`) and broadcasts its cut span to the block:

- **2-opt (`two_opt_multi_move.cu`):** Reverses `[seg_start, seg_end] = [mi + 1, mj]`. A surviving candidate in thread `tid` spanning `[ms, me] = [s_swap_i[tid] + 1, s_swap_j[tid]]` has cut edges `(tour[ms - 1], tour[ms])` and `(tour[me], tour[(me + 1) % n])`. Its gain stays exact if and only if `[ms - 1, me + 1]` is strictly disjoint from `[seg_start, seg_end]`, i.e., `me + 1 < seg_start || seg_end + 1 < ms` (negated: `!(me + 1 < seg_start || seg_end + 1 < ms)` invalidates `s_deltas[tid] = 0`).
- **3-opt (`three_opt_multi_move.cu`):** Reconnects `[mi + 1, mk]` across sorted cuts $0 \le m_i < m_j < m_k \le k_{\max}$. A surviving candidate `(ti, tj, tk)` modifies `[ti + 1, tk]` and touches cut endpoints `{ti, ti+1, tj, tj+1, tk, (tk+1) % n}`. Its exact $\Delta$ survives if and only if its entire closed cut span `[ti, tk]` lies strictly outside `[mi, mk]`:

$$\text{Independent} \iff (t_k < m_i) \;\lor\; (m_k < t_i)$$

Equivalently, any thread with `!(tk < mi || mk < ti)` sets `s_deltas[tid] = 0` in parallel without atomics or barriers inside the per-thread filter.

### Why `tour[0]` Invariance Prevents Wrap-Around Edge Contamination

Notice a subtle boundary condition: a move ending at `mk = n - 1` cuts the wrap-around edge `(tour[n - 1], tour[0])`, so why doesn't `mk = n - 1` invalidate an unrelated move starting at `ti = 0`?

Because **every reconnection starts at `seg_start = i + 1 >= 1`** (and never writes `tour[0]`), **`tour[0]` is strictly invariant across the entire multi-move pass**:
- The node at index `0` never moves, so `(tour[0], tour[1])` depends only on `tour[1]` (`ti + 1 >= 1`), while `(tour[n - 1], tour[0])` depends only on `tour[n - 1]` (`mk = n - 1`).
- If `mk < ti` (impossible when `mk = n - 1`) or `tk < mi` (`ti = 0, tk < mi <= n - 1`), the two moves' modified index intervals `[ti + 1, tk]` and `[mi + 1, mk]` share neither endpoints nor wrap-around vertices, preserving exact additive $\Delta$ independence!

## 3. Serial Selection vs. Cooperative Block Reversal (`BUG-29`, `2OPT-A`)

Earlier implementations suffered from a severe intra-block bottleneck (`BUG-29` / `2OPT-A`; see [analysis-gpu-two-opt-dispatch-and-reduction.md](./analysis-gpu-two-opt-dispatch-and-reduction.md)): Thread 0 both selected the winning segment *and* reversed `[seg_start, seg_end]` sequentially via `while (left < right)`, leaving 255 threads idle for up to $n / 2$ global or shared swaps per selected move.

In the unified architecture across [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu) and [three_opt_common.cuh::three_opt_apply_reconnection](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/kernels/three_opt_common.cuh):

1. **Only selection is serial:** Thread 0 scans $t \in [0, \text{block\_size})$ ($\le 256$ integer comparisons in shared memory, $< 1\,\mu\text{s}$), writes the winning bounds to shared scalars, and marks `s_deltas[best_t] = 0`.
2. **All 256 threads reverse cooperatively:** Every thread executes disjoint swap pairs `for (int s = tid; s < num_swaps; s += block_size)` across `[seg_start, seg_end]` (and `[mj+1, mk]` / `[mi+1, mk]` for 3-opt).
3. **Every thread filters its own slot in parallel:** Each thread tests its own `s_deltas[tid]` against the broadcast span in $O(1)$ before the next `__syncthreads()`.

## 4. Empirical Trade-Off: `multi_move` vs. Single-Steepest `batch`

Comparing single-steepest `batch` against greedy disjoint `multi_move` across the 6 completed TSPLIB instances (`d198`–`si535`, 10 repetitions each, `max_iterations=10`, `candidate_k=20`; full tables in [knowledge-three-opt-gpu-local-search.md](../../benchmark_scripts/experiments-and-examples/experiment-comparing-3opt-and-2opt/docs/knowledge-three-opt-gpu-local-search.md)) confirms that **multi-move dominates single-steepest `batch` on both speed and solution quality** across both neighborhood orders:

| Family | Mode | Mean Gap (%) | Mean Time (s) | Mean Generations | Per-Gen Time Range (ms) | Speedup vs. Own `batch` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **2-Opt GPU** | `batch` | 1.17 | 0.888 | 111.5 | 4.32 – 10.29 | 1.00x |
| **2-Opt GPU** | `multi_move` | **0.87** | **0.188** | **23.7** | 4.43 – 12.01 | **4.42x** |
| **3-Opt GPU** | `batch` | 1.14 | 11.063 | 67.6 | 99.54 – 397.52 | 1.00x |
| **3-Opt GPU** | `multi_move` | **0.67** | **2.022** | **9.1** | 101.33 – 417.53 | **5.47x** |

Two architectural takeaways stand out:
- **Zero per-generation penalty:** Thread-0 selection over $\le 256$ SMEM slots adds negligible cost compared to the $\Theta(n k)$ or $\Theta(n k^2)$ scan (`4.43` vs `4.32 ms/gen` on `d198` for 2-opt; `101.3` vs `99.5 ms/gen` for 3-opt).
- **Eliminating single-move stagnation:** On `att532`, applying one 3-opt move per pass (`ThreeOpt_Batch_GPU`) required `163.1` generations (`27.39 s`) and stagnated at `2.09%` gap, whereas `ThreeOpt_MultiMove_GPU` converged in `14.6` generations (`3.61 s`) to `0.85%` gap.

## References

- **Rocki, K., & Suda, R.** (2012). *Accelerating 2-opt and 3-opt Local Search Using GPU in the Travelling Salesman Problem*. IEEE HPCS, pp. 489–495.
- **Rocki, K., & Suda, R.** (2013). *High Performance GPU Accelerated Local Optimization in TSP*. IEEE IPDPSW, pp. 1788–1796.
- **O'Neil, M. A., & Burtscher, M.** (2015). *Rethinking the Parallelization of Random-Restart Hill Climbing: A Case Study in Optimizing a 2-Opt TSP Solver for GPU Execution*. GPGPU-8, pp. 99–108.
