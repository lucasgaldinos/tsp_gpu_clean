---
title: "Phases 1–6 — import-root, typing, checkpoint and benchmark-modularization walkthrough"
description: >
  Reconstruction of what actually happened across Phases 1–6 of the repository plan: the
  decision rounds that fixed the strict `src/` import root, the non-generic backend protocol,
  the checkpoint payload, the pure `compare_algorithms` API, the consolidation of two benchmark
  packages into `src/benchmarking/`, and the removal of presets and strategy shims. Every claim
  is anchored to a session step or to a repository line.
created: 2026-10-31
modifications:
  - date_modified: 2026-10-31
    modifications:
      - description: >
          Created by reading the Phase 1–6 execution session
          (`4f72da0b-6375-4cc1-ac90-dd8390461096`), the plan file, the earlier test-suite
          walkthrough, and the current working tree. The three transcripts named in the
          original brief (`e8dd0e92`, `8ad95c8b`, `57bedbaa`) were read first and contain no
          Phase 1–6 content; that finding is recorded under "Defects and deviations".
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[analysis-phase-7-residency-rng-walkthrough.md](./analysis-phase-7-residency-rng-walkthrough.md)"
  - "[test-suite-walkthrough.md](./test-suite-walkthrough.md)"
  - "[guide-local-search-pipeline-stages.md](../guide-local-search-pipeline-stages.md)"
  - "[algorithms.json](../../configs/algorithms.json)"
  - "[types.py](../../src/protocols/types.py)"
  - "[route_helpers.py](../../src/utils/route_helpers.py)"
  - "[benchmark_scripts/](../../benchmark_scripts/)"
tags:
  - analysis/walkthrough
  - analysis/architecture
  - analysis/typing
  - analysis/refactoring
  - algorithm/genetic-algorithm
  - algorithm/monolithic
  - benchmark/modular
  - guide/reorganization
  - notes/session
  - notes/decisions
  - review/quality
  - review/findings
---

# Phases 1–6 — import-root, typing, checkpoint and benchmark-modularization walkthrough

## Context

The solver had grown a second benchmark package beside the first (`src/benchmarking_v2/` and `src/benchmarks_v2/`), CLI argument parsing embedded inside library modules, an import prefix that disagreed with `pythonpath = ["src"]`, 68 `mypy` errors, and checkpoints that computed a best tour per repetition and then threw it away before writing the summary. None of that is visible in a benchmark table, which is precisely why it blocked the thesis: the numbers in the Chapter 4 tables could not be traced back to the runs that produced them.

The six objectives agreed before any file was touched are listed above the [Decisões Fixadas](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md) section (plan lines 51–61): one import root, pragmatic PEP 695/698 typing without a cascade of generic parameters, checkpoints enriched with routes and metadata, a stricter early-stopping vocabulary, a size-independent `compare_algorithms`, and one `src/benchmarking/` package plus one `benchmark_scripts/` directory. The five decisions D1–D5 that constrain every later phase live in that section (plan lines 63–72).

Two conventions are used throughout this document:

- **Phase N** means the tasks numbered `N.x` in the task registry (plan line 73 onward). The plan itself only writes `## Fase 7`, `## Fase 8` and `## Fase 9` headings (plan lines 285, 397, 623), so "Phases 1–6" is a numbering convention, not a document structure. See "Defects and deviations".
- **`step <n>`** means the record with `step_index == n` in `4f72da0b-6375-4cc1-ac90-dd8390461096/.system_generated/logs/transcript_full.jsonl`, reproducible with `jq 'select(.step_index==<n>)'`. The short `transcript.jsonl` in the same directory truncates every content field to 4120 characters and sets `truncated_fields: ["content"]`; only `transcript_full.jsonl` is authoritative for long messages.

## Timeline

Session `4f72da0b-6375-4cc1-ac90-dd8390461096` runs from `2026-10-28T17:47:24Z` to `2026-10-30T17:34:16Z` and holds 16 `USER_INPUT` records. It is the session that executed Tasks 1.1 through 7.5; Phase 7 continues in the companion walkthrough.

| When (UTC) | Where | Decision | Alternative rejected | Why |
| :-- | :-- | :-- | :-- | :-- |
| 2026-10-28T17:47 | step 0 | Restart the work as an explicit decision round before editing | Continue the inherited conversation | The predecessor model "started trying to type everything and disconsidering the whole of the project"; the user asked to "understand what can and cannot be done, while maintaining the project consistency" |
| 2026-10-28T18:18 | step 110 | Re-read the code against the plan before executing | Trust the plan artifact as written | "/plan review the code and the plan again … just check if everything said really makes sense" |
| 2026-10-28T18:36 | step 198 | Approve `implementation_plan.md` and begin execution | Keep planning | Plan review closed with no open questions |
| 2026-10-28T18:38 | step 214 | Accept Phase 1 on the suite alone | Add a typing gate to Task 1.2 | Task 1.2's Definition of Done was behavioural: the dynamic-import tests had to load again |
| 2026-10-28T18:51 | step 361 | Accept Phase 2 on `mypy` over the five core packages | Type the whole tree at once | Scoping kept the change inside the ~100–200 line step budget |
| 2026-10-28T21:34 → 21:48 | steps 805, 863 | Re-read the predecessor conversation, then approve the revised plan | Assume the summary captured it | Open questions on the plan remained after the first pass |
| 2026-10-28T22:11 | step 1205 | Accept Phase 5 with an empirical benchmark attached | Accept the refactor on unit tests alone | The user had asked for evidence of *where* 2-opt should sit, not just that it moved |
| 2026-10-29T14:13 | step 1208 | Rewrite the modular layer: eight named critiques, then Task 6.1 | Keep `legacy_kwargs`, `presets.py`, `argparse` scripts and monolithic strategy files | "those seem repetitive benchmarks. Why couldn't you use the orchestrator?"; hard-coded `two_opt_iterations` blocks 2-opt\*, k-opt, 3-opt, Or-opt and node insertion |
| 2026-10-29T15:01 | step 1260 | Approve the revised plan | — | Plan matched the eight critiques |
| 2026-10-29T15:28 | step 1615 (task-1608 log) | Accept Task 6.1 on 19 passed in 29.18 s | Keep the 10.52 s gate as the reference | Test time tripled because strategies were split into one file per algorithm; correctness, not wall clock, was the gate |
| 2026-10-29T19:18 | step 1869 | Six questions become the Phase 7 backlog | Treat the duplication as acceptable boilerplate | See the companion walkthrough |
| 2026-10-29T19:56 | step 1953 | Plan before writing; user answers three Phase 7 design questions | Implement while planning | "your job was to plan the changes, not make them" |

## Changes

### Phase 1 — the plan exists and imports resolve (Task 1.1, Task 1.2)

- **Task 1.1** (plan line 75) registered `documents_tcc/plan/plan-reorganization-and-typing.plan.md` and linked it from `plan-roteiro.md`. Mechanism: a 5W2H registry, one block per task, with the Solution written back into the same block after execution.
- **Task 1.2** (plan line 90) removed the `src.` prefix from every `class_module` in [algorithms.json](../../configs/algorithms.json) (plan cites lines 11–29) and aligned `pyproject.toml` (plan cites lines 70–105) with `pythonpath = ["src"]`. Mechanism: the registry is loaded through `importlib.import_module`, so `"src.algorithms...."` raised `ModuleNotFoundError: No module named 'src'` once `src/` itself was the import root.

### Phase 2 — typing and checkpoint payload (Task 2.1, Task 2.2)

- **Task 2.1** (plan line 105) made `BackendModule` a non-generic protocol returning `Array` (`NDArray[Any]`) and gained `arccos`; annotated the lazy caches on `ProblemContext` as `np.ndarray | None`; migrated `phase_timer.measure` to PEP 695 `[**P, R]`; applied `typing.override` on the concrete strategies; and repaired return types and casts in `src/distances/matrix.py`, `src/distances/pairwise.py`, `src/loaders/tsplib_loader.py`, `src/loaders/database_loader.py` and `src/algorithms/monolithic/full_gpu_monolithic.py`. Mechanism: one non-generic boundary (`Array`) instead of threading `ArrayT` through `SelectionStrategy`, `ProblemContext` and every call site — decision D1 at plan line 65.
- **Task 2.2** (plan line 133) split the early-stopping vocabulary and widened the checkpoint payload. `"hit_optimal"` is now reserved for `best_cost <= optimal_cost + 1e-6`; gaps under 1 % that are not optimal record `"within_1pct_optimal"` (decision D5). Checkpoints gained `best_tour`, `raw_best_tours`, `problem_type`, `edge_type` and the six real strategy names, captured with `type(strategy).__name__` rather than a class-introspection fallback. Touched: `src/algorithms/metaheuristics/genetic_algorithm.py`, `src/algorithms/monolithic/full_gpu_monolithic.py`, `src/benchmarking_v2/algorithm_runner.py`, `src/benchmarking_v2/orchestration.py`, `tests/validation/test_early_stopping.py`.

### Phase 3 — a pure comparison function (Task 3.1)

- **Task 3.1** (plan line 153) extracted `compare_algorithms(...)` and `compare_algorithms_by_group(...)` in `src/benchmarking_v2/aggregate_statistics.py` (plan cites lines 98–358). The function validates that every instance has paired coverage, runs Friedman, computes Nemenyi critical distances and geometric speedups against a configurable baseline. Mechanism: `perform_cross_problem_analysis_stratified` now delegates both strata to the pure function and computes `small_problem_algos` dynamically, which removes the undefined name (8 `ruff` `F821`) and the static `cpu_threshold = 100` (7 `mypy` errors).

### Phase 4 — one benchmark package, one script directory (Task 4.1)

- **Task 4.1** (plan line 171) consolidated `src/benchmarking_v2/` and `src/benchmarks_v2/` into `src/benchmarking/{runner,analysis,database}/`, moved the four entry points to `benchmark_scripts/`, and deleted the legacy directories (both are absent from the tree today). Mechanism: public names are re-exported from `src/benchmarking/__init__.py`; nullable `cursor.fetchone()` results were narrowed; the `runner`↔`analysis` import cycle was broken with `from __future__ import annotations` plus `if TYPE_CHECKING:` imports. `DEFAULT_SELECTED_TSP_FILES` was centralised in the library at `src/benchmarking/runner/config_loader.py` — decision D3, later reversed by Task 7.5.

### Phase 5 — presets out of `metaheuristics/`, local search over the population (Task 5.1)

- **Task 5.1** (plan line 212) moved the four preset wrappers to `src/algorithms/presets.py`, reduced `src/algorithms/metaheuristics/` to `genetic_algorithm.py`, replaced the hard-coded local-search branch with an ordered `LocalSearchStage` pipeline, corrected the ordering so `post_evaluation` evaluates fitness before 2-opt, and rewrote `post_evolution` to refine the whole `(N, n_cities)` population through `improve_batch`/`evaluate_batch` instead of a single tour — the change that stopped 255 of 256 CUDA blocks idling.
- The empirical deliverable is `benchmark_scripts/compare_local_search_stages.py` over `att48`, `bayg29` and `bays29`, with tables written to `results_v3/local_search_comparison/tables/` (the Markdown table is present today).

### Phase 6 — no compatibility layer (Task 6.1)

- **Task 6.1** (plan line 242) deleted `src/algorithms/presets.py` (absent today, by design), deleted the ten `*_strategies.py` shims, and split every strategy family into one file per algorithm (`order_crossover.py`, `pmx_crossover.py`, `two_opt.py`, `no_op.py`, `swap_mutation.py`, `inversion_mutation.py`, `tournament_selection.py`, `roulette_wheel_selection.py`, `cost_evaluation.py`, `random_initialization.py`).
- Domain types moved to `src/protocols/types.py` and batch helpers to `src/utils/route_helpers.py`. `GeneticAlgorithm` became a pure GA by default (`local_search=None`) governed by a `GenerationStep` pipeline. `configs/algorithms.json` became declarative and resolves strategies through `STRATEGY_REGISTRY` (`src/algorithms/strategies/__init__.py`, line 40 today). The three benchmark scripts were rewritten as plain python that calls the orchestrator — the direct answer to critique 5 and 6 at step 1208.

The eight critiques that produced Task 6.1 are worth keeping verbatim in summary form, because the plan only says "resolve as críticas 1 a 8": (1) where `type LocalSearchStage` belongs; (2) why `tournament_size` and `two_opt_iterations` became `legacy_kwargs` and whether a pure GA is reachable; (3) why an `if` was added for loading; (4) why `presets.py` is returned from inside `genetic_algorithm.py`; (5) why a benchmark needs `argparse` instead of fixed parameters, and why table generation is not automated; (6) why the three scripts duplicate each other; (7) why `algorithms.json` points at presets instead of supporting a user's own `algorithm_configs`; (8) why strategy files are monolithic and identically named.

## Verification

Historical gates, quoted from the transcripts.

Phase 1 — Task 1.2's Definition of Done, observed at step 214 (2026-10-28T18:38:15Z):

```text
rootdir: /home/lucas_galdino/chimera/gpu_accelerated_clean
configfile: pyproject.toml
collected 18 items

tests/test_benchmark_selected_problems.py .....                          [ 27%]
tests/test_modular_architecture.py ...........                           [ 88%]
tests/validation/test_early_stopping.py .                                [ 94%]
tests/validation/test_incremental_checkpoints.py .                       [100%]

============================== 18 passed in 9.89s ==============================
```

Phase 2 — Task 2.1, observed at step 361 (2026-10-28T18:51Z), command `mypy src/protocols src/algorithms src/distances src/loaders src/utils`:

```text
Success: no issues found in 41 source files
```

Phase 5 — Task 5.1, reported at step 1205 (2026-10-28T22:11:31Z):

```text
- Automated Tests: pytest passed 19/19 tests with 0 failures in 10.52s.
- Type Checking : mypy src benchmark_scripts passed with 0 issues in strict mode.
- Linter        : ruff check src tests benchmark_scripts passed with 0 issues.
```

Phase 6 — Task 6.1, reported by the worker log at step 1615 (2026-10-29T15:28:34Z, `.system_generated/tasks/task-1608.log`):

```text
Pytest: 19 passed in 29.18s
```

Current gate (2026-10-31), confirmed by the operator:

```text
$ .venv/bin/python -m pytest tests -q
711 passed, 6 xfailed

$ .venv/bin/python -m mypy src tests
Success: no issues found in 111 source files

$ .venv/bin/python -m ruff check src tests benchmark_scripts
All checks passed!
```

> [!NOTE]
> The three numbers above are quoted, not reproduced in this session. An independent re-run reached `219 passed, 2 xfailed` and was then killed by a `KeyboardInterrupt` at `src/benchmarking/runner/orchestration.py:495` because the shell was being driven concurrently by another agent. Treat the quoted gate as the operator's measurement.

## Defects and deviations

1. **The three transcripts named in the brief are not Phase 1–6 sources.** `e8dd0e92`, `8ad95c8b` and `57bedbaa` are all timestamped `2026-10-30T23:09Z`–`2026-10-30T23:56Z` and discuss Phase 9 exclusively: counting case-insensitively, `"Fase 2"` through `"Fase 8"`, `"residency"` and `"Task 7."` each return zero hits in all three, while `"Fase 9"` returns 11, 17 and 31 respectively. The real Phase 1–7 session is `4f72da0b-6375-4cc1-ac90-dd8390461096`.
2. **The Copilot transcript named as a secondary source postdates the work.** Session `86299fac-0e20-41ba-99d0-171ed05945a7` starts `2026-10-29T13:15Z` and its first user message is `2026-10-30T14:24Z`; its last two user messages are the two archaeology briefs that commissioned these documents. It reads the plan, so it mentions every phase, but it executes none of them.
3. **Phases 1–6 have no headings.** The plan writes a single `## Registro de Tarefas 5W2H` heading (line 73) and only adds `## Fase 7`/`8`/`9` headings later (lines 285, 397, 623). The "phase" grouping for Tasks 1.1–6.1 exists only in the task numbering.
4. **Two files the plan cites as changed no longer exist.** `tests/test_modular_architecture.py` and `tests/test_benchmark_selected_problems.py` are listed in the Task 4.1 and Task 6.1 Solution blocks but are absent; the suite was later reorganised into a package-mirrored layout, recorded in [test-suite-walkthrough.md](./test-suite-walkthrough.md) under [10.4 Test reorganization](./test-suite-walkthrough.md#104-test-reorganization).
5. **The Phase 5 report contradicts its own table.** Step 1205 says `post_evolution` is "faster overall", but the same message's table reports `0.852 s` for `att48`/`HybridOptimized` `post_evolution` against `0.037 s` for `post_evaluation`. Only the gap claim survives (`post_evaluation` reaches 0.00 % on `bayg29`); the speed claim does not.
6. **The eight critiques are stored truncated.** In `transcript.jsonl`, step 1208 carries 4120 characters and `truncated_fields: ["content"]`, which cuts the list after critique 3. The full 8320-character list exists only in `transcript_full.jsonl`.
7. **The `mypy` denominator grows across phases.** `41` files (Phase 2), `68` (Phase 7), `111` today. A "0 errors" claim is only comparable within one phase.
8. **None of Phases 1–6 is committed.** `git log --oneline` shows `62edcbd` as `HEAD` ("Finally ending modularization of the algorithms"), and the working tree is still dirty. Phase 7's deliverables were later found reverted to that same commit — see the companion walkthrough.

## Open questions

1. Were Tasks 1.1–6.1 meant to carry `## Fase` headings that were dropped, or is the numbering a report convention invented when Phase 7 was written?
2. Is `62edcbd` the intended boundary commit for "end of Phases 1–6", or should that state be reconstructed into a commit of its own?
3. Task 5.1's benchmark table prices `post_evaluation` at `0.037 s` on `att48`; the plan's summary claims "up to 50 %" savings. Which comparison is the intended headline number?
4. Should the six pre-existing strict xfails be fixed inside this plan, or tracked as a separate defect register?
5. Does `config_loader.py` still need any problem list at all, given that Task 7.5 moved problem selection into `benchmark_scripts/run_selected_problems.py`?
