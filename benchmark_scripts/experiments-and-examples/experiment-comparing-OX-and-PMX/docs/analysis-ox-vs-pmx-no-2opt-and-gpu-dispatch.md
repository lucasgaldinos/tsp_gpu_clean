---
title: Analysis of OX vs. PMX Without 2-Opt and GPU Dispatch Bottlenecks
description: >
  Empirical diagnosis of why disabling 2-opt under memetic defaults produced
  293%-493% optimality gaps on TSPLIB instances (n > 150), how host-side GPU
  kernel dispatch dominates pure-GA runtime (~1.1-1.2 ms/gen), and the parameter
  adjustments captured in configs/algorithms.tuned.json.
created: 2026-11-24
type: analysis
scope: local
modifications:
  - date_modified: 2026-11-24
    modifications:
      - description: >
          Documented the 30-repetition no-2-opt OX_GPU vs. PMX_GPU findings,
          root causes of poor pure-GA convergence, GPU dispatch cost anatomy,
          tuned pure-GA configuration rationale, worked LaTeX PMX vs. OX
          recombination example, and candidate-list caching semantics.
related_files:
  - "[algorithms.json](../configs/algorithms.json)"
  - "[algorithms.tuned.json](../configs/algorithms.tuned.json)"
  - "[ox_vs_pmx.md](../results/tables/ox_vs_pmx.md)"
  - "[gpu_pmx_crossover.py](../gpu_pmx_crossover.py)"
  - "[README.md](../../README.md)"
tags:
  - analysis/experiment
  - analysis/convergence
  - analysis/performance
  - algorithm/crossover
  - algorithm/genetic-algorithm
  - algorithm/cuda-kernels
  - benchmark/gpu
  - benchmark/profiling
  - benchmark/tuning
  - notes/findings
---

# OX vs. PMX Without 2-Opt: Convergence Diagnosis & GPU Dispatch Analysis

## 1. Observed Baseline Results (30 Repetitions, No 2-Opt, Memetic Defaults)

Running [`configs/algorithms.json`](../configs/algorithms.json) (`population_size=128`, `mutation_rate=0.03`, `InitializationRandom`, `SelectionGPUTournament(k=5)`, `MutationGPUPermSwap`, `LocalSearchNoOp`, `local_search_stage="none"`) across the 9 TSPLIB instances with $n > 150$ produced the results in [`results/tables/ox_vs_pmx.md`](../results/tables/ox_vs_pmx.md):

| Metric (9 instances, $n \in [198, 1002]$) | `OX_GPU` | `PMX_GPU` |
| :--- | ---: | ---: |
| Overall Mean Optimality Gap | **292.83%** | **492.88%** |
| Best Instance Gap (`si535`) | 54.30% | 91.25% |
| Worst Instance Gap (`pr1002`) | 524.58% | 748.19% |
| Mean Wall Time | 1.59 s | 4.39 s |
| `d198` Mean Generations / Time per Gen | 421.9 gens / **1.172 ms/gen** | 930.9 gens / **1.098 ms/gen** |
| `pr1002` Mean Generations / Time per Gen | 2,624.9 gens / **1.278 ms/gen** | 8,904.6 gens / **1.200 ms/gen** |

For comparison, the same pipeline **with** batched GPU 2-opt (`LocalSearchGPUTourTwoOpt(mode="batch")`, 5 reps) achieved **2.01% mean gap (`OX_GPU`)** and **6.28% mean gap (`PMX_GPU`)**.

---

## 2. Why Were the Pure-GA (No 2-Opt) Gaps So High?

Those hyperparameters were tuned for a **memetic GA** where 2-opt does the geometric edge refinement every generation. Removing 2-opt without retuning exposed five compounding weaknesses:

1. **Swap mutation (`MutationGPUPermSwap`) cannot uncross TSP edges effectively.** Swapping two cities breaks 4 edges and inserts 4 random ones. By contrast, segment inversion (`MutationGPUPermInversion`) reverses a sub-tour $[i..j]$, which breaks only 2 edges — structurally equivalent to a random 2-opt move.
2. **Starved mutation rate (`0.03`) on a tiny population (`128`).** On average, only $128 \times 0.03 \approx 3.8$ individuals mutate per generation. Without 2-opt introducing improving edges, diversity collapses quickly.
3. **Excessive selection pressure (`k = 5`) combined with $(\mu + \lambda)$ elitist survival.** High tournament size lets the best initial random tours take over the 128-route population within a few generations, locking in long crossing edges.
4. **Uniformly random initialization (`InitializationRandom`) on $n \le 1{,}002$.** Random permutations on `pr1002` start more than 15× above optimum; a pure GA with 128 routes and 3% swap mutation stalls long before untangling hundreds of crossings.
5. **Why `PMX_GPU` lagged `OX_GPU` by +200 percentage points AND ran 3.4× more generations:**
   In symmetric TSP, tour cost is a function of **edge adjacency**, not absolute array index. Order Crossover (OX) preserves relative city adjacency from both parents (Starkweather et al., 1991; Larrañaga et al., 1999), whereas Partially Mapped Crossover (PMX) preserves absolute positions — breaking intact geometric sub-tours when parents are shifted relative to each other. Consequently, `PMX_GPU` made tiny noisy improvements that kept resetting early-stopping patience (8,905 generations on `pr1002` vs. OX's 2,625) while converging to far worse tours.

---

## 3. GPU Dispatch Cost Anatomy & Mitigation

Across a 5× jump in tour length (`d198` $\to$ `pr1002`), per-generation wall time barely moved (**1.17 $\to$ 1.28 ms/gen** for `OX_GPU`; **1.10 $\to$ 1.20 ms/gen** for `PMX_GPU`), and Table 4 phase shares stayed nearly constant (`Sel ~15%`, `CX ~18%`, `Mut ~22%`, `Eval ~7%`, `Surv ~24%`, `Host Sync ~10–13%`).

### Dispatches per Generation (No 2-Opt Modular GPU GA)

Each generation executes roughly **10–12 CUDA kernel launches**, **6 small H2D copies**, and **1 blocking GPU $\to$ CPU scalar sync**:

| Phase | Approximate Share | What Runs on Host + Device per Generation |
| :--- | ---: | :--- |
| `Sel` | ~15% (~0.18 ms) | NumPy candidate draw + 1 H2D copy + 1 custom kernel (`tournament_selection_kernel`) |
| `CX` | ~18% (~0.22 ms) | NumPy `sample_distinct_pairs` + 2 H2D copies (`cut1`, `cut2`) + `cp.empty` + 1 custom kernel |
| `Mut` | ~22% (~0.26 ms) | NumPy `probs`/`pos1`/`pos2` + 3 H2D copies + 1 D2D copy + 1 custom kernel |
| `Eval` | ~7% (~0.08 ms) | No host RNG; 1 custom kernel (`evaluate_cost_kernel`) |
| `Surv` | ~24% (~0.29 ms) | ~4–6 CuPy built-in kernels (`concatenate` ×2, `argsort`, fancy indexing ×2) + temp allocations |
| `Host Sync` | ~10–13% (~0.13 ms) | `cp.min(fitness)` reduction + blocking `float(...)` GPU $\to$ CPU read for early stopping |

Because GPU kernels without 2-opt finish in microseconds on $N = 128$, the ~1.1 ms/gen floor is dominated by **Python/CuPy wrapper dispatch, small `cp.asarray` H2D transfers, and CuPy temporary allocations** — not GPU arithmetic. Over 8,905 generations on `pr1002`, that equals **~90,000–100,000 host-to-driver calls**.

### Mitigation Levers

1. **Amortize dispatch via larger `population_size` (zero code changes):** Raising `population_size` from `128` to `512` (or `1024`) keeps host dispatch nearly constant at ~1.1 ms/gen while giving the GPU 4×–8× more routes per launch, improving both SM occupancy and solution quality.
2. **Use stronger pure-GA operators (zero code changes):** `MutationGPUPermInversion`, `mutation_rate=0.20`, `k=3`, and optionally `InitializationNearestNeighbor(k=20, return_gpu=true)` cut wasted generations drastically.
3. **Host/runtime batching (code-level in `src/`):** Pack per-generation RNG arrays into a single pinned H2D buffer, check early stopping every $K$ generations instead of every generation, or capture the generation sequence in a **CUDA Graph**.
4. **Monolithic fused kernel (architectural):** Move the entire generational loop + `curand` inside a single `.cu` kernel (trading away modular strategy composition, as in [`full_gpu_monolithic.py`](../../../../src/algorithms/archive/full_gpu_monolithic.py)).

---

## 4. Tuned Pure-GA Configuration (`configs/algorithms.tuned.json`)

[`configs/algorithms.tuned.json`](../configs/algorithms.tuned.json) keeps `local_search_stage: "none"` and `LocalSearchNoOp` while applying four pure-GA adjustments across both random-start and nearest-neighbor-start pairs (`OX_GPU_Random`, `PMX_GPU_Random`, `OX_GPU_NN`, `PMX_GPU_NN`):

| Parameter | Baseline (`algorithms.json`) | Tuned (`algorithms.tuned.json`) | Rationale |
| :--- | :--- | :--- | :--- |
| `population_size` | `128` | `512` | 4× diversity and 4× GPU work per fixed ~1.1 ms/gen dispatch |
| `mutation_rate` | `0.03` | `0.20` | Restores edge exploration when 2-opt is disabled |
| `mutation` | `MutationGPUPermSwap` | `MutationGPUPermInversion` | 2-edge segment reversal uncrosses Euclidean tours cleanly |
| `selection.k` | `5` | `3` | Reduces premature diversity collapse under elitist survival |
| `initialization` | Random only | Random + `InitializationNearestNeighbor(k=20)` | Compares both cold-start untangling and geometric recombination |

### How to Run the Tuned Config (Into Its Own Output Folder)

Use `--patience 200` (optional, recommended for pure-GA runs without 2-opt) and a dedicated `--output-dir` so checkpoints never collide with the previous run:

```bash
ROOT="$(git rev-parse --show-toplevel)"
EXP="$ROOT/benchmark_scripts/experiments-and-examples/experiment-comparing-OX-and-PMX"

# 1. Preflight dry-run
uv run python "$EXP/compare-pmx-ox-runner.py" \
  --algorithms-config "$EXP/configs/algorithms.tuned.json" \
  --output-dir "$EXP/results_no_2opt_tuned" \
  --benchmark-name ox_vs_pmx_no_2opt_tuned \
  --baseline OX_GPU_Random \
  --patience 200 \
  -r 10 --no-incremental \
  --dry-run

# 2. Full run + tables, macros, CD diagrams, and scaling figures
uv run python "$EXP/compare-pmx-ox-runner.py" \
  --algorithms-config "$EXP/configs/algorithms.tuned.json" \
  --output-dir "$EXP/results_no_2opt_tuned" \
  --benchmark-name ox_vs_pmx_no_2opt_tuned \
  --baseline OX_GPU_Random \
  --patience 200 \
  -r 10 --no-incremental
```

---

## 5. Worked Mathematical Example: PMX vs. OX Side-by-Side

Consider two 0-indexed parent permutations of length $n = 8$ with cut points $c_1 = 2$ and $c_2 = 4$:

$$
p_1 = [0,\; 1,\; \underbrace{2,\; 3,\; 4}_{[c_1,\, c_2]},\; 5,\; 6,\; 7],
\qquad
p_2 = [4,\; 6,\; \underbrace{3,\; 5,\; 2}_{[c_1,\, c_2]},\; 7,\; 0,\; 1].
$$

### 5.1 Partially Mapped Crossover (PMX) — [`pmx_crossover.cu`](../kernels/pmx_crossover.cu)

1. **Copy segment $[c_1, c_2] = [2, 4]$ from $p_1$ and mark membership $S$:**

   $$
   c_{\text{PMX}} = [\_,\; \_,\; \mathbf{2},\; \mathbf{3},\; \mathbf{4},\; \_,\; \_,\; \_],
   \qquad
   S = \{p_1[i] : c_1 \le i \le c_2\} = \{2, 3, 4\}.
   $$

2. **Build the inverse lookup $\pi_2^{-1}(v) = \text{index of city } v \text{ in } p_2$** (`my_inv_p2` in warp-private shared memory):

   $$
   \pi_2^{-1}(0)=6,\; \pi_2^{-1}(1)=7,\; \pi_2^{-1}(2)=4,\; \pi_2^{-1}(3)=2,\;
   \pi_2^{-1}(4)=0,\; \pi_2^{-1}(5)=3,\; \pi_2^{-1}(6)=1,\; \pi_2^{-1}(7)=5.
   $$

3. **Resolve every uncopied city $v = p_2[i]$ inside $[c_1, c_2]$ ($v \notin S$) via bijection chaining:**
   Define $f(pos) = \pi_2^{-1}\!\big(p_1[pos]\big)$. Starting at $pos_0 = i$, iterate $pos_{t+1} = f(pos_t)$ until $pos_t \notin [c_1, c_2]$:
   - $i = 2$: $p_2[2] = 3 \in S \implies$ already placed, skip.
   - $i = 4$: $p_2[4] = 2 \in S \implies$ already placed, skip.
   - $i = 3$: $v = p_2[3] = 5 \notin S \implies$ follow the 3-hop chain:

     $$
     pos_0 = 3 \;\xrightarrow{p_1[3]=3}\; \pi_2^{-1}(3) = 2 \;\xrightarrow{p_1[2]=2}\; \pi_2^{-1}(2) = 4 \;\xrightarrow{p_1[4]=4}\; \pi_2^{-1}(4) = \mathbf{0} \notin [2, 4].
     $$

     Place $c_{\text{PMX}}[0] = \mathbf{5}$. Because $p_1$ and $p_2$ are bijections, distinct entry indices $i_1 \ne i_2$ trace strictly disjoint chains and write to strictly distinct slots outside $[c_1, c_2]$, so all 32 warp lanes in [`pmx_crossover.cu`](../kernels/pmx_crossover.cu) execute this loop in parallel without atomics.

4. **Copy all remaining empty positions $j$ directly from $p_2[j]$:**

   $$
   c_{\text{PMX}}[1] = 6,\quad c_{\text{PMX}}[5] = 7,\quad c_{\text{PMX}}[6] = 0,\quad c_{\text{PMX}}[7] = 1
   \;\Longrightarrow\;
   \boxed{c_{\text{PMX}} = [\mathbf{5},\; 6,\; \mathbf{2},\; \mathbf{3},\; \mathbf{4},\; 7,\; 0,\; 1]}.
   $$

### 5.2 Order Crossover (OX) — Same $p_1, p_2, c_1=2, c_2=4$

1. **Copy segment $[2, 4]$ from $p_1$:** $c_{\text{OX}} = [\_,\; \_,\; \mathbf{2},\; \mathbf{3},\; \mathbf{4},\; \_,\; \_,\; \_],\; S = \{2, 3, 4\}$.
2. **Stream $p_2$ cyclically from $(c_2 + 1) \bmod n = 5$ and filter out $S$** (`__ballot_sync` + `__popc` on GPU):

   $$
   p_2^{\text{rot}} = [7,\; 0,\; 1,\; \cancel{4},\; 6,\; \cancel{3},\; 5,\; \cancel{2}]
   \;\Longrightarrow\;
   \text{stream} = [7,\; 0,\; 1,\; 6,\; 5].
   $$

3. **Fill empty slots cyclically starting at index $5 \to 6 \to 7 \to 0 \to 1$:**

   $$
   \boxed{c_{\text{OX}} = [6,\; 5,\; \mathbf{2},\; \mathbf{3},\; \mathbf{4},\; 7,\; 0,\; 1]}.
   $$

### 5.3 Representation & GPU Hardware Contrast

- **Adjacency vs. absolute position:** A cyclic shift of $p_2$ represents the exact same TSP tour. OX preserves the relative cyclic order $7 \to 0 \to 1 \to 6 \to 5$ from $p_2$ regardless of index shift, whereas PMX anchors cities to absolute indices ($c[1]=p_2[1]=6$) and relocates city $5$ to slot $0$, inverting $(6, 5)$ into $(5, 6)$ and breaking edge adjacency.
- **GPU shared memory & warp divergence:** OX needs $1n$ byte/warp (`in_seg`) with branchless warp compaction (`__ballot_sync` + `__popc`), whereas PMX needs $5n$ bytes/warp (`inv_p2` + `in_seg`) and executes a data-dependent `while` chain whose hop count varies across warp lanes.

---

## 6. Candidate Lists vs. Tournament Contestants (Caching Semantics)

- **Spatial $k$-NN candidate list (`candidate_k`, used by 2-opt and `InitializationNearestNeighbor`):** Computed **once** on CPU in [`problem_context.py::ProblemContext.get_cpu_candidate_list`](../../../../src/protocols/problem_context.py) (reused across repetitions via `candidate_cache`) and uploaded **once** to VRAM by [`problem_context.py::ProblemContext.get_gpu_candidate_list`](../../../../src/protocols/problem_context.py). It is never recomputed per generation or inside a CUDA kernel. Setting `"candidate_k": null` in `strategies.local_search.params` disables candidate pruning and forces full $\Theta(n^2)$ all-pairs 2-opt.
- **Tournament contestant table (`d_candidates` in [`tournament_selection.py::SelectionGPUTournament`](../../../../src/algorithms/strategies/selection/tournament_selection.py)):** Unrelated to $k$-NN — this is a fresh $(N \times k)$ matrix of random parent indices drawn on CPU each generation to guarantee exact RNG parity with CPU selection. Eliminating that small per-generation H2D transfer requires moving RNG onto the GPU (e.g., `curand` inside `tournament_selection.cu`).

