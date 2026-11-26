---
title: "How $k$, $N$ and the stratified analysis work in the Friedman protocol"
description: >
  Reference note on the statistical machinery of the thesis comparison: what $k$
  and $N$ count, how Friedman builds ranks and why Iman–Davenport's $F_F$ is the
  decision statistic, why the historical CPU/GPU coverage split produced two
  degenerate rank tests, what full coverage changes, why `lin318` needs 79
  generations (the 1% early-stopping discontinuity), the two senses of
  "bit-to-bit identical", and the validation/parser split between the thesis
  script and `src/`.
created: 2026-11-16
status: review
author:
  - "Lucas Galdino"
type: notes
scope: global
modifications:
  - date_modified: 2026-11-17
    modifications:
      - description: >
          Added the missing YAML frontmatter and rewrote the file, which was a
          chat reply missing its first two sections and carrying first-person
          asides. Restructured into numbered sections, marked the coverage split
          as historical (the CPU now covers all 15 instances), fixed an
          arithmetic slip in the `lin318` threshold analysis, and reconciled the
          validation and threshold recommendations with `TASK 13.25` and
          `TASK 13.26`, which implemented them.
related_files:
  - "[genetic_algorithm.py](../../../src/algorithms/metaheuristics/genetic_algorithm.py)"
  - "[statistics.py](../../../src/benchmarking/analysis/statistics.py)"
  - "[aggregate_statistics.py](../../../src/benchmarking/analysis/aggregate_statistics.py)"
  - "[run_chapter4_benchmark.py](../../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py)"
  - "[orchestration.py](../../../src/benchmarking/runner/orchestration.py)"
tags:
  - analysis/statistics
  - guide/statistics
  - analysis/benchmark
  - benchmark/statistics
  - notes/planning
  - algorithm/tsp
  - analysis/architecture
  - guide/benchmarking
  - review/quality
  - notes/performance
---

# How `k`, `N` and the stratified analysis work

Reference note for the Friedman/Nemenyi protocol used in the Chapter 4 comparison: the meaning of the counts, the two rank tests that the uneven CPU/GPU coverage used to force, and the practical questions that came out of the consolidated data (generation counts and the duplication between the thesis script and `src/`).

## 1. What `k` and `N` count

**$k$** = number of algorithms compared simultaneously. **$N$** = number of *independent problems* — instances, not repetitions.

For each problem $j$, the $k$ algorithms are ranked by quality: best gets 1, worst gets $k$.
The ranks are then averaged across problems, and the Friedman statistic compares the averaged ranks against the null hypothesis "all algorithms are equivalent":

$$R_i = \frac{1}{N}\sum_{j=1}^{N} r_{ij}, \qquad Q = \frac{12N}{k(k+1)}\left\lbrack\sum_{i=1}^{k} R_i^2 - \frac{k(k+1)^2}{4}\right\rbrack$$

$Q$ is compared to $\chi^2_{k-1}$, or — as the implementation in [statistics.py](../../../src/benchmarking/analysis/statistics.py) does — to Iman–Davenport's $F_F$, which is better conditioned for small $N$. Rejecting the omnibus licenses a post-hoc test: Nemenyi for all pairs, Holm–Bonferroni against the control.

## 2. The historical CPU/GPU coverage split

While the CPU was limited to $n \le 400$, the 15 instances divided into two strata with different algorithm coverage:

| | instances | algorithms available | test |
| --- | --- | --- | --- |
| stratum 1 | 9 (n ≤ 318) | all 4 | $k=4$, $N=9$ |
| stratum 2 | 6 (n > 400) | GPU only | $k=3$, $N=6$ |

A single Friedman cannot run on uneven data: the rank of each problem must involve the *same* $k$ algorithms.
The code therefore had only two options — drop the CPU (losing the comparison the thesis is about) or drop the 6 large instances (losing scale) — and it did both, separately.

Both halves were weak in practice:

+ $N=9$ is a small sample for a rank test, and on the stored data $k=4/N=9$ gave $p = 0.8896$ — failure to reject.
+ $k=3/N=6$ was **degenerate**: $N(k-1) - \chi_F^2 = 12 - 12 = 0$, so the Iman–Davenport $F_F$ divides by zero and diverges.

With full coverage — the CPU executing all 15 instances after the size cut was removed (`TASK 13.26`, `DP13-1`) — there is one test instead of two: $k=4$, $N=15$, one rank table and one CD diagram with four mean ranks, exactly what Demšar §3.2 prescribes.
The degenerate case is gone: $N(k-1) = 45$ against a $\chi_F^2$ of order 15 leaves $F_F$ finite and well-conditioned.

The stratified helper `aggregate_statistics.py::perform_cross_problem_analysis_stratified` exists to handle unequal coverage: it logs *"Using stratified analysis to handle CPU/GPU problem coverage difference"* and builds the first stratum from `n ≤ cpu_threshold`.
With every instance carrying every algorithm, both strata contain the same 15 instances and the function computes the same thing twice — the justification for it disappears, not just its output.

## 3. Why `lin318` needs 79 generations

**Generations are not an input — they are an output.** The input is a *ceiling* of $2n\sqrt{n}$ (63,435 for `lin318`), which is never reached. What actually decides run length is [`genetic_algorithm.py::GeneticAlgorithm.evolve`](../../../src/algorithms/metaheuristics/genetic_algorithm.py) with its three stop conditions:

1. `best_cost ≤ optimal` → `hit_optimal`
2. `gap < 1.0%` → **`within_1pct_optimal`**
3. `generations_without_improvement ≥ patience`, where `patience = ⌊2√n⌋` → `no_improvements`

Same seed gives the same run, so the count is deterministic — but it is *data-dependent*, not constant.
Two instances with the same $n$ can legitimately differ.
The table below comes from the archived scalar-CPU cohort; the counts remain valid after the vectorization because the tours are bit-identical (Section 4).

| instance | n | patience ⌊2√n⌋ | mean gens | max gens | mean gap |
| --- | --- | --- | --- | --- | --- |
| att48 | 48 | 13 | 4.2 | 5 | 0.43% |
| eil51 | 51 | 14 | 5.2 | 7 | 0.53% |
| berlin52 | 52 | 14 | 4.7 | 5 | 0.12% |
| gr96 | 96 | 19 | 10.2 | 12 | 0.62% |
| kroA100 | 100 | 20 | 10.6 | 12 | 0.49% |
| ch150 | 150 | 24 | 18.9 | 22 | 0.77% |
| d198 | 198 | 28 | 25.9 | 29 | 0.85% |
| pr264 | 264 | 32 | 36.9 | 44 | 0.85% |
| **lin318** | **318** | **35** | **79.2** | **147** | **1.04%** |

**`lin318` is the only instance whose mean gap exceeds 1%.** For the other eight, rule 2 fires: the run enters the 1% band and stops — which is why their generation counts sit near or below patience and cluster tightly (`att48`: 4.2 mean, 5 max).

For `lin318`, rule 2 is out of reach. Its runs must satisfy rule 3 instead, and rule 3 resets on every improvement, so the run length becomes *"generations until improvements stop, plus 35 more"* — 79 on average, 147 worst case.
Improvements kept arriving until generation ~112 in the slowest run.

The flip is narrow: a mean gap a few hundredths of a percentage point above the 1.0% cutoff switches the entire stopping regime and nearly doubles the runtime.
That discontinuity is why a cost estimate for this instance has to be a range rather than a point.

## 4. The two senses of "bit-to-bit identical"

"Bit-to-bit identical" means every value is exactly equal — the same integers in the same positions in the array — not merely close.
Two claims of that kind arein play:

+ The **vectorization** produces the same `int32` tour array as the scalar scan. Verified 3,510 times with `np.array_equal`: not approximately, identically.
  That is why `mean_cost`, `mean_gap`, rankings, Friedman and CD are untouched while wall time changes (`DP13-5`).
+ The **two hybrid variants** produce identical `raw_costs` on all 15 instances (`BUG-64`).

It is a stronger statement than "statistically equivalent", because it pins the exact computation path: no reordering, no rounding drift.

## 5. What the thesis script duplicates from `src/`

The duplication claim is partly true, and the parts differ in kind — which matters because each needs a different fix.

**The engine is not reimplemented.** [run_chapter4_benchmark.py](../../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py)
imports `setup_benchmark_logging`, `adaptive_generations`, `load_checkpoint`, `load_all_configs`, `run_comprehensive_benchmark` and `gpu_available` from `src/`.
The GA, the runner, the analysis and the logging setup are all library code; there is no duplicated engine and no duplicated logging setup.

**Two things are genuinely duplicated:**

| duplication | where | size |
| --- | --- | --- |
| An argparse front-end that is a **subset** of `src/cli/benchmark.py` (13 flags vs ~20) | both files | ~55 lines |
| Report formatting: `log_effective_parameters`, `log_generation_budget`, `log_comparison` | script only | ~90 lines |

On top of that, the configs are loaded **twice**: `run()` calls `load_all_configs(CONFIG_DIR)` for validation and logging, and then `run_comprehensive_benchmark` loads them again internally.

**The script-local validations split into two groups:**
They are not repository invariants; they are assertions about *this chapter's experimental claim*:

| validation | what it asserts | scope |
| --- | --- | --- |
| `validate_comparison_contract` | each slot maps to the expected class (`CPU` → `EvaluationCPUCost`, …) and `use_gpu` matches — i.e. **"the four variants differ only in backend"** | thesis-specific: the iso-algorithmic claim, not a code property |
| `validate_effective_instances` | the 15 declared instances exist | thesis-specific: experiment definition |
| `validate_problem_list` | instances present in `configs/problems.json` | generic |
| `validate_generation_budget` | seed / population / mutation / adaptive generations pinned | generic |
| `validate_existing_checkpoints` | checkpoint schema compatibility | generic |
| `validate_gpu_backend` | a CUDA device is available | generic |

**Status (2026-11-17).** The four generic checks now live in the runner's `orchestration.py::_run_preflight`, which every entry point calls (`TASK 13.25`), and the two thesis-specific assertions remain in the script, as intended by `DP13-8`.
What is still outstanding is the parser: `run_chapter4_benchmark.py` keeps a second `ArgumentParser` with a subset of the flags, and the target shape is a **pinned preset** — keep the protocol constants, the three report formatters and `validate_comparison_contract`, move the formatting into `report_generator.py`, and delegate execution by building an `argv` list for `cli.benchmark.main(argv)` (`TASK 13.24`).
That removes the flag drift for good, because there would be exactly one parser.

Target shape, as illustration:

```python
def build_preset_argv(args: argparse.Namespace) -> list[str]:
  return [
    "--repetitions", str(args.repetitions),
    "--benchmark-name", args.benchmark_name,
    "--output-dir", str(args.output_dir),
    "--problems", *PROBLEMS,
    *([] if args.incremental_runs else ["--no-incremental"]),
  ]
```

The former blocker is gone: `cpu_size_threshold` is now `int | None` with `None` meaning *no cut* (`TASK 13.26`), so a preset can express "all instances, all algorithms" without a synthetic ceiling.
