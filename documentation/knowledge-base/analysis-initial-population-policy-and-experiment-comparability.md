---
title: "Generation-0 Policy, `initial_cost` Semantics, and When Two Benchmark Cohorts Are Comparable"
description: >
  Reference answers for the questions raised while comparing `results_v2` and `results_v4`: what generation 0 does in `GeneticAlgorithm.evolve`, why an implicit 2-opt pass over the random initial population once existed and why it inflated both `initial_fitness` and wall-clock time, why `post_evaluation` still refines $P_0$ (`BUG-21`), where per-phase timings are measured and why the runner discards them (`DP11-7`), how to refine $P_0$ by composing a pipeline step instead of editing the core loop (`DP11-6`), and which provenance fields must match before two checkpoints may be compared.
created: 2026-11-06
status: review
author:
  - "[[Lucas Galdino]]"
  - "[[GitHub Copilot]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-06
    modifications:
      - description: >
          Created from the `berlin52` cohort comparison (`results_v2` vs `results_v4/chapter4_pilot_pre_design_b`): recorded the generation-0 contract, the arithmetic that separates the two `initial_cost` definitions, the complexity argument that explains the removed pre-pass, the phase-timing gap, and the composition recipes that restore $P_0$ refinement without touching `evolve()`.
related_files:
  - "[guide-iso-algorithmic-ga-validation.md](./guide-iso-algorithmic-ga-validation.md)"
  - "[guide-gpu-cpu-comparison-methodology.md](./guide-gpu-cpu-comparison-methodology.md)"
  - "[analysis-strategy-hierarchy-and-context-contracts.md](./analysis-strategy-hierarchy-and-context-contracts.md)"
  - "[analysis-genetic-algorithm-memetic-lineage.md](./analysis-genetic-algorithm-memetic-lineage.md)"
  - "[guide-local-search-pipeline-stages.md](../guide-local-search-pipeline-stages.md)"
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py)"
  - "[algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py)"
  - "[phase_timer.py](../../src/utils/phase_timer.py)"
tags:
  - algorithm/genetic-algorithm
  - algorithm/memetic-algorithm
  - algorithm/local-search
  - algorithm/two-opt
  - benchmark/metrics
  - benchmark/reproducibility
  - analysis/methodology
  - analysis/architecture
  - analysis/performance
  - guide/algorithm-composition
  - guide/benchmarking
  - notes/reproducibility
  - review/quality
---

# Generation-0 Policy, `initial_cost` Semantics, and Cohort Comparability

## 0. The question this answers

Comparing `results_v2/problem_statistics/berlin52_stats.json` with the modular cohort produced numbers that looked "almost unchanged" while `mean_initial_cost` moved from $25{,}030$ to $14{,}315$. Three separate effects were tangled in that comparison, and only one of them was a change in the algorithm:

| Effect | Nature | What it changes in a comparison |
| --- | --- | --- |
| `initial_cost` sampled before vs. after a 2-opt pass | **Metric redefinition** | that one metric stops being comparable; final cost, gap and generations are unaffected |
| random seeds vs. `42…56`, 26 vs. 15 repetitions | **Sampling** | the samples are no longer paired, so a paired test is not licensed; validity is unaffected |
| `candidate_k`, Design B class names, kernels | **Implementation** | nothing to fix: a difference in what was run is the treatment being measured |

The useful question is therefore not "are these cohorts comparable?" but "which metric, compared with which test, licenses which claim?". §5 answers that field by field. Only two claims are genuinely unavailable across the v2/v4 pair: `initial_cost`/`improvement` (different $P_0$ policy) and `time` as "same algorithm, new implementation" (different code revision). Final cost, gap, generations and success rate remain comparable in both, with the caveat that different seeds force an unpaired test.

---

## 1. The generation-0 contract

`GeneticAlgorithm.evolve` has exactly two responsibilities before the generational loop starts ([genetic_algorithm.py:452-505](../../src/algorithms/metaheuristics/genetic_algorithm.py#L452)):

1. `population = initialization.initialize(customers, population_size, context)` — build $N = 256$ tours.
2. `fitness = evaluation.evaluate_batch(population, context)` — assign each of those tours its cost.

Everything else — selection, crossover, mutation, local search, survival — belongs to the per-generation pipeline built by `_build_default_pipeline` ([genetic_algorithm.py:304-340](../../src/algorithms/metaheuristics/genetic_algorithm.py#L304)).

The loop then assigns `state.generation = gen + 1` **before** running the steps ([genetic_algorithm.py:511-516](../../src/algorithms/metaheuristics/genetic_algorithm.py#L511)), so the first iteration is `state.generation == 1`. Nothing in that pair of calls depends on `local_search_stage`; that independence is the property the current code enforces and the property the removed block violated.

### 1.1 What `fitness` is, and why it is computed before anything else

`fitness` is the **objective value of each individual**, the only input that lets any later step prefer a better tour over a worse one.

Concretely, a tour $\pi$ of $n$ cities visits them in order and returns to the start, so its cost is the sum of the $n$ edges it closes:

$$c(\pi) \;=\; \sum_{i=1}^{n} d\bigl(\pi_i,\ \pi_{i+1}\bigr), \qquad \pi_{n+1} \equiv \pi_1$$

and `fitness` is the flat array holding that value once per individual:

$$\texttt{fitness} \;=\; \bigl[\,c(\pi_1),\ c(\pi_2),\ \dots,\ c(\pi_{256})\,\bigr] \; \in \; \mathbb{R}^{256}$$

`evaluate_batch` produces it for the whole population in one call, not in a python loop over 256 arrays — that batching is what makes the GPU path possible, but it does not change the meaning: one cost per tour.

Four consumers read it, which is why it cannot be deferred until something asks for it:

| Consumer | What it needs `fitness` for |
| --- | --- |
| `SelectionStep` → `select_batch(routes, fitness, ...)` | tournament selection compares the costs of $k=5$ drawn candidates and keeps the cheapest; without `fitness` every individual would be equally likely |
| `SurvivalStep` → `_select_survivors` | the $(\mu+\lambda)$ rule stacks parents and offspring ($2N$) and keeps the $N$ cheapest, which is a sort on `fitness` |
| `_extract_best` | returns `(best_tour, best_cost)`, i.e. `initial_fitness` before the loop and the reported solution after it |
| early stopping | compares `best_cost` against `optimal_cost` and detects stagnation across generations |

The distinction worth keeping: `population` is *what* is being searched, `fitness` is *how good each candidate currently is*. A population without fitness is an unranked set; selection, survival, termination and reporting all degenerate without it.

### 1.2 What the removed block did

Until 2026-11-05, `evolve` contained a pre-loop branch. The essential part was:

```python
if has_ls and ls is not None and self.local_search_stage in ("post_crossover", "post_mutation"):
  population = self.timer.measure("local_search", ls.improve_batch, population, context)
  fitness = self.timer.measure("evaluation", self.evaluation.evaluate_batch, population, context)
elif has_ls and ls is not None and self.local_search_stage == "post_evaluation":
  fitness = self.timer.measure("evaluation", self.evaluation.evaluate_batch, population, context)
  # ... then refine and re-evaluate
```

Read literally, that branch says: *choosing a memetic stage changes what generation 0 is.* The stage is a statement about **offspring**, and the code was applying it to **parents**, one generation early.

### 1.3 Why it only covered `post_crossover` and `post_mutation`

The author was reasoning within the memetic paradigm, which defines the search over the subspace of local optima (Moscato, 1989; Merz & Freisleben, 1997):

$$\mathcal{S}^{*} = \{\, x \in \mathcal{S} \;\mid\; \forall y \in \mathcal{N}(x),\ f(x) \le f(y) \,\}$$

For `post_crossover` and `post_mutation`, local search is applied to `state.offspring` inside the loop. At generation 0 there are no offspring, so the branch patched the parents to make the "every individual is a local optimum" invariant hold from the first selection onwards. `post_evolution` was excluded because that stage is defined as a pure GA during all generations with a single polish afterwards; and `post_evaluation` kept a variant of the branch because its contract ("evaluate, then refine survivors") is at least *shaped* like a refinement of an existing population. Both justifications are post-hoc rationalisations of an undocumented behaviour, not requirements of the algorithm.

### 1.4 Stage dispatch is written in three places, and `post_evaluation` is written twice

The stage switch is not centralised. The same `local_search_stage` comparison appears at three sites, each with its own copy of the `has_ls`/`ls` guard:

| Site | Stages handled there |
| --- | --- |
| [`_build_default_pipeline`](../../src/algorithms/metaheuristics/genetic_algorithm.py#L304) (constructor time) | `post_crossover`, `post_mutation`, `post_evaluation` |
| [`evolve`, pre-loop](../../src/algorithms/metaheuristics/genetic_algorithm.py#L465) | `post_evaluation`, as a separate pre-`P_0` invocation |
| [`evolve`, post-loop](../../src/algorithms/metaheuristics/genetic_algorithm.py#L546) | `post_evolution` |

`post_evaluation` is the one stage handled **twice**: once as the last element of the pipeline that runs inside every generation, and once as an extra pre-loop pass. That is not a stylistic repetition — it is the mechanism behind `BUG-21`, and it explains why the same stage name produced two consecutive 2-opt passes with no genetic operator between them.

Two consequences that are properties of the current structure, not opinions about it:

1. **A sixth stage would require editing three sites**, and nothing in the code enforces that they agree. Adding a stage to `LocalSearchStage` but not to one of the three sites yields a configuration that is silently accepted and silently ignored.
2. **The guard is duplicated, not shared.** `has_ls` and `ls` are recomputed in `_build_default_pipeline` and again in `evolve`, so the definition of "local search is active" has two independent copies.
3. **Two `post_evaluation` sites cannot stay consistent by construction**, because one is built from the other's data rather than derived from a single description of the stage.

Whether to centralise the switch is a design decision, deliberately not argued here.

---

## 2. What `initial_cost` actually measures

`initial_fitness` is whatever `_extract_best` returns immediately before the loop ([genetic_algorithm.py:501-503](../../src/algorithms/metaheuristics/genetic_algorithm.py#L501)). Its meaning is therefore inherited from the last operation applied to `population`:

| Generation-0 path | `initial_cost` is | `mean_improvement` is |
| --- | --- | --- |
| raw (`initialize` → `evaluate`) | best cost of $N$ random tours | full GA+LS improvement |
| post-2-opt (`initialize` → `improve_batch` → `evaluate`) | best cost of $N$ 2-opt tours | improvement **after** the pre-pass |
| `post_evaluation` branch (today, `BUG-21`) | best cost of $N$ 2-opt tours | same artefact as above |

Measured on `berlin52` with seed 42, calling the strategies directly rather than inferring from a cohort:

```python
import numpy as np
from pathlib import Path

from benchmarking.runner.config_loader import load_all_configs
from loaders.tsplib_loader import TsplibLoader
from protocols.problem_context import ProblemContext

configs = load_all_configs(Path("configs"))
with TsplibLoader(Path("datasets/tsp")) as loader:
  problem = loader.load(Path("datasets/tsp/berlin52.tsp"))

context = ProblemContext(problem, xp=np, seed=42)
ga = configs["algorithms"]["CPU"].instantiate(
  configs["ga_params"], seed=42, local_search_stage="post_mutation"
)
customers = list(range(problem.dimension))

population = ga.initialization.initialize(customers, ga.population_size, context)
print("raw     :", float(np.min(ga.evaluation.evaluate_batch(population, context))))

refined = ga.local_search.improve_batch(population, context)
print("2-opt   :", float(np.min(ga.evaluation.evaluate_batch(refined, context))))
```

Expected output (GTX 1050 host, `max_iterations=10`, `candidate_k=20`):

```text
raw     : 25478.0
2-opt   : 14634.0
```

The two numbers bracket the two cohort values ($25{,}030$ and $14{,}315$ are their means over different seed sets), which is what makes the attribution unambiguous rather than assumed.

### 2.1 The arithmetic that moved "improvement" in the opposite direction of quality

```text
results_v2 :  (25,030.08 − 7,554.65) / 25,030.08 = 69.82 %   final gap 0.168 %
results_v4 :  (14,315.07 − 7,547.20) / 14,315.07 = 47.28 %   final gap 0.069 %
```

The reported improvement *fell* while the final gap *improved*. The descent is the same total, partitioned differently:

```text
pre-pass + loop  ≈  (25,478 − 14,634) + (14,634 − 7,547)  =  10,844 + 7,087  ≈  17,931
raw  + loop      ≈   25,030 − 7,555                       ≈  17,475
```

The first term migrated from `mean_improvement` into `mean_time`. That is the whole explanation; no optimisation was lost or gained.

Consequence for the thesis: any "improvement %" column inherited from a cohort produced before this correction is not comparable with one produced after it. `mean_gap`, `mean_cost`, and `mean_time` remain comparable *in definition* (the first two exactly; the third only if the same work is performed).

---

## 3. Why the pre-pass dominated the runtime

Two independent multipliers apply to the pre-loop pass, and both are visible in the code rather than inferred from wall clocks.

**Multiplier 1 — the pre-pass always saturates its budget.** A random permutation of $n$ planar points has $\Theta(n^{2})$ edge crossings in expectation, while an offspring produced by order crossover and swap mutation inherits most of its parent's structure. Since `improve_batch` is capped at `max_iterations=10`, the pre-pass executed the full 10 descents for all $N$ tours, while in-loop offspring usually converge after 1–3.

**Multiplier 2 — it duplicated the run's entire local-search budget.** Each accepted 2-opt move reverses a tour segment in $\Theta(n)$, so a full descent is $\Theta(n^{3})$ in the worst case (Croes, 1958; Lin & Kernighan, 1973). At `local_search_stage="post_mutation"` the loop performs one descent per offspring per generation, and early stopping with $\textbf{patience} = 2\sqrt{n}$ terminated `berlin52` after $3.8—4.7$ generations. The pre-pass was therefore up to

$$\frac{\texttt{max\_iterations}}{\text{generations executed}} = \frac{10}{4} \approx 2.5\times$$

the entire remaining local-search volume of the run, spent before generation 1 and counted in `mean_time`.

That also explains why `mean_generations` dropped ($4.69 \rightarrow 3.80$): a population that starts near a local optimum reaches the stagnation window sooner. Fewer generations is not evidence of a faster algorithm; it is evidence of a different starting point.

> [!IMPORTANT]
> The exact split between `local_search` and the other phases **cannot** be measured in the current cohorts, because per-phase timings are not persisted. Closing that gap is `DP11-7` (`Task 11.7`); until then, the ratio above is a structural bound derived from `max_iterations` and the stopping rule, not a measurement.

---

## 4. Per-phase timing: measured, returned, discarded

Three facts, each verifiable in one place:

1. `PhaseTimer` records every phase. `evolve` wraps each operator in `timer.measure(...)` and returns the aggregate in `stats["phase_timings"]` ([genetic_algorithm.py:566-577](../../src/algorithms/metaheuristics/genetic_algorithm.py#L566)), shaped as `{phase: {"total_ms", "calls", "mean_ms"}}` ([phase_timer.py:68-77](../../src/utils/phase_timer.py#L68)).
2. `run_single_algorithm` reads only `h2d_bytes`, `d2h_bytes`, `kernel_launches`, `initial_fitness`, `best_fitness`, `generations_completed`, and `stop_reason` ([algorithm_runner.py:180-195](../../src/benchmarking/runner/algorithm_runner.py#L180)). `stats["phase_timings"]` is never appended to `results`, so it never reaches a checkpoint.
3. `_save_problem_statistics_file` writes mean time, cost, gap, generations, and transfer bytes ([orchestration.py:173-193](../../src/benchmarking/runner/orchestration.py#L173)) — no phase breakdown.

So the answer to *"which operator consumed the most time?"* is: **the number exists at runtime and is thrown away at the first hop.** It is available today only by instrumenting a script directly:

```python
tour, stats = ga.evolve(context, customers, max_generations=10, optimal_cost=None)
for phase, data in sorted(stats["phase_timings"].items(), key=lambda kv: -kv[1]["total_ms"]):
  print(f"{phase:14s} {data['total_ms'] / 1000:8.3f} s  ({data['calls']} calls)")
```

Expected shape on CPU for `berlin52`:

```text
local_search     12.xxx s  (N calls)
evaluation        1.xxx s  (N calls)
selection         0.xxx s  (N calls)
...
```

Two caveats that make `DP11-7` more than a plumbing task:

- `GeneticAlgorithm` constructs `PhaseTimer(enabled=enable_phase_timing)` with `use_cuda_events` defaulting to `False` ([genetic_algorithm.py:291](../../src/algorithms/metaheuristics/genetic_algorithm.py#L291), [phase_timer.py:33-40](../../src/utils/phase_timer.py#L33)). On GPU, `perf_counter` around an asynchronous kernel launch measures **enqueueing**, not execution, so GPU phases are underestimated and their sum will not match `mean_time`.
- Reporting must be conditional on presence of the key, or old cohorts break the table generator.

---

## 5. Reading a cohort comparison: what each field constrains

Comparing different algorithms, different seeds and different repetition counts is normal and necessary. What changes is never *permission*; it is **which claim the numbers support**. The table below separates three outcomes: a metric that keeps its meaning, a metric that does not, and a test that is no longer licensed.

| Field | What it records | What it changes in the comparison |
| --- | --- | --- |
| `raw_seeds` | the seed of each repetition | If seeds match, observations are paired and a paired test (Wilcoxon signed-rank) is available, which removes between-seed variance. If seeds differ, the samples are independent: use Mann–Whitney U, or Friedman on per-problem ranks. Both are valid; only the paired option is lost. |
| `repetitions` / `successful_runs` | sample size per cell | 26 vs 15 changes the standard error of each mean, not the meaning of the mean. Means with different $n$ are comparable; the CI widths differ, and unequal-$n$ tests (Welch, Brunner–Munzel) may be more appropriate than equal-variance ones. |
| `strategies.*` | operator classes actually instantiated | Identifies the treatment. A difference here is the intervention being measured, not an error. |
| `algorithm_config.strategy_params` | `candidate_k`, `mode`, `max_iterations`, `k` | Same role as above at parameter level; it is what makes "which knob caused the speedup" answerable. |
| `algorithm_config.local_search_stage` | where 2-opt sits in the pipeline | Determines whether $P_0$ was refined, so it decides whether `initial_cost`/`improvement` share a definition (§2). It does **not** affect the comparability of final cost, gap or generations — those are post-run quantities. |
| `problem_type`, `edge_type` | the distance convention | The one field pair that must agree: `EUC_2D` and `GEO` define different objective functions, so comparing their costs compares two problems. |
| `mean_initial_cost` | best cost of the starting population | Comparable only when the $P_0$ policy is the same. Across policies it is a different quantity, and so is any `improvement` percentage derived from it. |
| `mean_time` | wall clock of `evolve()` | Comparable as a *performance* figure for two implementations of the same algorithm; comparable as a *speedup* figure only within one code revision, because a removed pre-pass or an added `candidate_k` changes how much work a second counts. |

### 5.1 The three tiers, stated plainly

1. **Metric definition** (hard limit, one metric). `initial_cost` and `improvement` are the only quantities whose meaning depends on the $P_0$ policy. Final cost, gap, generations, success rate
   and time keep their definitions.
2. **Code revision** (soft limit, one metric). Two revisions can be compared on time — that is a perfectly good experiment — but the claim is "implementation A vs implementation B", not "the same algorithm got faster". Within one revision, time differences between variants are a clean speedup measurement.
3. **Sampling** (no limit, affects the test). Different seeds or different repetition counts mean the two cohorts are independent samples. The comparison stands; the paired test is what is unavailable, and $n$ plus a dispersion measure must be reported.

Stated this way, the v2/v4 table is usable for cost, gap, generations and success rate, and usable for time only as an implementation comparison — which is exactly what it is.

### 5.2 Class names no longer identify the transfer strategy

A provenance caveat worth recording, because it weakens the assumption that `strategies.local_search` alone describes what ran. Design B collapsed the former `NaiveGPUTwoOpt` and `BatchGPUTwoOpt` into one class:

```text
before Design B :  NaiveGPUTwoOpt          BatchGPUTwoOpt
after  Design B :  LocalSearchGPUTwoOpt(mode="naive")   LocalSearchGPUTwoOpt(mode="batch")
```

The class name `LocalSearchGPUTwoOpt` is identical for both, so a checkpoint whose `strategies.local_search` reads `LocalSearchGPUTwoOpt` does **not** say whether the run used per-route transfers or batched transfers. That information now lives only in `algorithm_config.strategy_params.local_search.mode` ([two_opt.py:371-397](../../src/algorithms/strategies/local_search/two_opt.py#L371)). The pre-Design-B checkpoint from the pilot cohort, by contrast, recorded `NaiveGPUTwoOpt` or `BatchGPUTwoOpt` directly.

The same applies to `LocalSearchCPUTwoOpt`, whose `mode` selects `best`, `first` or `multi_move` acceptance. The general consequence: **`strategies.*` is a class inventory, not a description of the treatment.** Any claim about which 2-opt variant ran has to read `strategy_params` as well.

### 5.3 Why "not much change" was the expected observation on `berlin52`

`berlin52` cannot discriminate the four variants at all. With $N = 256$, 2-opt on the initial population and a `< 1\%` early-stopping rule, every variant lands within $0.2\%$ of $7542$ in three to five generations. The Friedman test on the modular cohort returns $p = 0.417$ (not significant), against $p = 0.006$ on the v2 cohort — a difference produced by seeds and metric definitions, not by algorithms. Discrimination requires instances where the implementation differences are actually exercised: `candidate_k` pays off above $n \approx 100$, the dense-matrix threshold at $n > 2{,}048$, coordinate mode above $n > 10{,}000$, and shared-memory tiling above $n > 11{,}520$.

---

## 6. Verification recipe

A fix in this area ships with an observation that fails without it. The generation-0 contract can be pinned without a GPU and without repetition noise:

```python
import numpy as np
from pathlib import Path

from benchmarking.runner.config_loader import load_all_configs
from loaders.tsplib_loader import TsplibLoader
from protocols.problem_context import ProblemContext

STAGES = ("none", "post_crossover", "post_mutation", "post_evaluation", "post_evolution")

configs = load_all_configs(Path("configs"))
with TsplibLoader(Path("datasets/tsp")) as loader:
  problem = loader.load(Path("datasets/tsp/berlin52.tsp"))
customers = list(range(problem.dimension))

initial_costs = {}
for stage in STAGES:
  ga = configs["algorithms"]["CPU"].instantiate(configs["ga_params"], seed=42, local_search_stage=stage)
  context = ProblemContext(problem, xp=np, seed=42)
  _, stats = ga.evolve(context, customers, max_generations=1, optimal_cost=None)
  initial_costs[stage] = stats["initial_fitness"]

print(initial_costs)
```

Expected after `Task 11.8`: one value repeated five times (the raw best of the same 256 seeded tours).

Measured on 2026-11-06, before `Task 11.8`, `berlin52`, 1 generation:

```text
{'none': 25478.0, 'post_crossover': 25478.0, 'post_mutation': 25478.0,
 'post_evaluation': 14634.0, 'post_evolution': 25478.0}
```

Four stages agree on the raw value and `post_evaluation` alone disagrees — `BUG-21` reproduced with an observation that fails without the fix and passes with it.

---

## 7. Tasks that must update this document

This file states facts about the code as of 2026-11-06. Three open tasks will invalidate parts of it, so each carries the exact section to revise. When a task closes, edit the listed section **in place** in this file; do not append an errata block at the end.

| Identifier | Task | What changes in the code | Section of this file to revise |
| --- | --- | --- | --- |
| `BUG-21` / `DP11-8` | [Task 11.8](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L1458) | removes the pre-loop `post_evaluation` branch, so $P_0$ is raw for all five stages | [§1](#1-the-generation-0-contract) (the pre-loop pair becomes unconditional), [§1.1](#11-what-fitness-is-and-why-it-is-computed-before-anything-else) (no local-search consumer before the loop), [§1.4](#14-stage-dispatch-is-written-in-three-places-and-post_evaluation-is-written-twice) (two dispatch sites remain, not three), [§2](#2-what-initial_cost-actually-measures) (drop the `post_evaluation` row), [§5](#5-reading-a-cohort-comparison-what-each-field-constrains) (row on `local_search_stage`), [§6](#6-verification-recipe) (the five values must become identical) |
| `DP11-6` | [Task 11.6](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L1427) | adds `params.pipeline` with name resolution and checkpoint recording | §1 (what `evolve` does before the loop gains a declared stage), §5 (new provenance field to read) |
| `DP11-7` | [Task 11.7](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L1442) | persists `raw_phase_timings`, `mean_phase_timings`, `phase_share_pct`; fixes `use_cuda_events` for GPU phases | [§3](#3-why-the-pre-pass-dominated-the-runtime) (the bound becomes a measurement), [§4](#4-per-phase-timing-measured-returned-discarded) (the discard no longer happens) |

Until `Task 11.8` lands, read §2's `improvement` column as stage-dependent and §3 as a structural bound rather than a measurement.

---

## 8. References

- Croes, G. A. (1958). A method for solving traveling-salesman problems. *Operations Research*, 6(6), 791–812.
- Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem. *Operations Research*, 21(2), 498–516.
- Moscato, P. (1989). On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms. *Caltech Concurrent Computation Program, C3P Report 826*.
- Merz, P., & Freisleben, B. (1997). A genetic local search approach to the quadratic assignment problem. *ICGA-97*, 465–472.
- Reinelt, G. (1991). TSPLIB—A traveling salesman problem library. *ORSA Journal on Computing*, 3(4), 376–384.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30.
- Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82.
