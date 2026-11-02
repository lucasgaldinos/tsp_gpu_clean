---
title: "Phase 7 — route residency, RNG resolution, GPU inversion and 2-opt variants walkthrough"
description: >
  Reconstruction of Phase 7: why route array residency and random-generator resolution moved
  into `src/utils/`, how `GPUInversionMutation` and the three 2-opt variants were motivated and
  built, and how `datasets/tsp.duckdb` replaced the hardcoded problem list. Includes the
  verification gates captured at the time, the defects that survived as strict xfails, and the
  eight-file revert that had to be undone in Phase 8.
created: 2026-10-31
modifications:
  - date_modified: 2026-10-31
    modifications:
      - description: >
          Created from the Phase 7 execution session
          (`4f72da0b-6375-4cc1-ac90-dd8390461096`), its `walkthrough.md` artifact, the plan
          file, the Phase 8 recovery request in `ea9a2b63-9509-4d3c-8a28-5461f8b7b756`, and
          the current working tree. The three transcripts named in the original brief were
          read and contain Phase 9 only; see "Defects and deviations".
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[analysis-phases-1-6-reorganization-walkthrough.md](./analysis-phases-1-6-reorganization-walkthrough.md)"
  - "[guide-local-search-pipeline-stages.md](../guide-local-search-pipeline-stages.md)"
  - "[route_helpers.py](../../src/utils/route_helpers.py)"
  - "[rng.py](../../src/utils/rng.py)"
  - "[inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py)"
  - "[inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[strategies/__init__.py](../../src/algorithms/strategies/__init__.py)"
tags:
  - analysis/walkthrough
  - analysis/refactoring
  - algorithm/genetic-algorithm
  - algorithm/local-search
  - algorithm/mutation
  - algorithm/gpu-kernels
  - algorithm/memetic
  - benchmark/modular
  - guide/reorganization
  - notes/session
  - notes/decisions
  - review/defects
  - review/findings
---

# Phase 7 — route residency, RNG resolution, GPU inversion and 2-opt variants walkthrough

## Context

Phase 6 left the solver compositional: strategies are separate classes, `GeneticAlgorithm` is a pure GA unless a local-search stage is configured, and `configs/algorithms.json` names strategies through a registry. What it did not leave is a single way to *touch an array*. Eight strategy files each carried their own `try: import cupy` block and their own `isinstance(batch, cp.ndarray)` conversions — roughly twelve sites — and eleven of them repeated `if context is not None and hasattr(context, "rng")` before drawing random numbers (plan lines 303–307). Every one of those copies is a place where a CPU run and a GPU run can diverge.

Three further gaps were open when Phase 7 started. `InversionMutation` had no GPU counterpart because no kernel reversed a sub-segment independently of the cost matrix (plan lines 332–336). The 2-opt implementation evaluated all $n^2/2$ pairs and then applied a single best reversal per sweep, so disjoint improving moves — which have additive deltas — were re-discovered on the next sweep (plan lines 351–355). And `DatabaseLoader` could not open `datasets/tsp.duckdb` at all: it raised `BinderError: Referenced column "dimension" not found in FROM clause`, while `config_loader.py` kept a hardcoded `DEFAULT_SELECTED_TSP_FILES` (plan lines 370–374).

**`step <n>`** below means the record with `step_index == n` in `4f72da0b-6375-4cc1-ac90-dd8390461096/.system_generated/logs/transcript_full.jsonl`. The short `transcript.jsonl` truncates content at 4120 characters and flags it in `truncated_fields`; use the full file for anything longer than a paragraph.

## Timeline

| When (UTC) | Where | Decision | Alternative rejected | Why |
| :-- | :-- | :-- | :-- | :-- |
| 2026-10-29T19:18 | step 1869 | Accept six open questions as the Phase 7 backlog | Leave the duplication and the single-move 2-opt alone | The user's own framing: repeated array checks should be "a function in utils … instead of repeating the code over and over"; a 2-opt that re-scans everything is "basically evaluating the same routes over and over" |
| 2026-10-29T19:56 | step 1953 | Plan the backlog before writing it | Implement while planning, as the previous round had done | "your job was to plan the changes, not make them" |
| 2026-10-29T19:56 | step 1953 | Drop database back-compatibility entirely | Keep a `routing.duckdb`/`nodes` fallback inside `DatabaseLoader` | "No, no compatibility. The new version of the db is the main one" |
| 2026-10-29T19:56 | step 1953 | Own the problem list in the runner, not the library | Keep `DEFAULT_SELECTED_TSP_FILES` in `config_loader.py` | "run_selected_problems.py should specify. Or query from the db using database_loader.py" |
| 2026-10-29T19:56 | step 1953 | Build the 2-opt variants | Ship only the existing best-improvement sweep | "I would like different variations for two opt, yes" |
| 2026-10-29T20:19 | step 1987 | Approve the Phase 7 implementation plan | — | Plan matched the six questions and the four answers |
| 2026-10-29T20:53 | step 2451 | Close Phase 7: 22 passed, `mypy` clean on 68 files, `ruff` clean | Extend the phase to commit the work | Definition of Done met; committing was never part of it |
| 2026-10-30T17:42 | `ea9a2b63` step 723 | Restore eight Phase 7 files that had been reverted to `HEAD` | Re-implement from scratch | The verified end-of-Phase-7 state still existed in the transcript; re-deriving it would have lost the reviewed code |

The six questions at step 1869 map one-to-one onto the five tasks, which is why the phase reads as
a single decision rather than five unrelated refactors:

| Question at step 1869 | Task |
| :-- | :-- |
| "document the usage of this for each possibility inside `./documentation`" | Task 7.1 |
| "wouldn't it be better defining a function in utils that checks this" | Task 7.2 (residency) |
| "couldn't this also be a utils function … the parameter would be a callable" | Task 7.2 (`SeedLike`) |
| "So it's possible, just wasn't done? Is it too hard to implement?" | Task 7.3 |
| "why would I make another full iteration instead of directly applying those 2 unrelated nodes?" | Task 7.4 |
| "I don't like these defaults hardcoded in a configloader … there's a db in `./datasets/tsp.duckdb`" | Task 7.5 |

## Changes

### Task 7.1 — the pipeline guide (plan line 287)

`documentation/guide-local-search-pipeline-stages.md` was written as a contract document rather than a tutorial. Its current headings show the shape: core architecture and data contracts (line 49), the `GenerationState` carrier (line 53), the `GenerationStep` protocol (line 81), pipeline execution flow (line 96), the five built-in `LocalSearchStage` modes (line 119, one subsection each from lines 135 to 237), four executable custom-step patterns (lines 264–559), the declarative `configs/algorithms.json` form (line 600), and a verification snippet (line 676).

Mechanism: the guide documents `GenerationStep.run(state, timer)` as the only extension point, so a new local-search placement is a new class rather than a new branch inside `GeneticAlgorithm`.

### Task 7.2 — one way to convert arrays, one way to resolve a seed (plan line 303)

Two utilities absorbed the duplication.

- **Residency.** `src/utils/route_helpers.py` now exposes `is_gpu_array` (line 27), `ensure_routes_batch` (line 32), `to_cpu_batch` (line 59) and `restore_routes_shape(..., to_gpu: bool = False)` (line 82). The contract is the three-tuple from `to_cpu_batch` — `(batch_np, was_1d, was_gpu)` — which lets a strategy do its arithmetic on the host and then hand the array back to the device it came from.
- **Seeds.** `type SeedLike = int | Generator | Callable[[], Generator] | None` lives at `src/protocols/types.py` line 17. `make_rng` and `resolve_rng` now live at `src/utils/rng.py` lines 22 and 43, **not** in `route_helpers.py` as the plan's Task 7.2 Solution states.

The RNG move is a documented correction, not a silent drift: the plan carries a warning at line 330 stating that grouping `make_rng`/`resolve_rng` with shape helpers created "acoplamento indevido" and that Task 8.10 extracted them into `src/utils/rng.py` with a re-export from `src/utils/__init__.py`.

Also in this task: `GenerationState` gained `generation: int = 0`, synchronised once per generation so a periodic step can branch on `state.generation % K`.

### Task 7.3 — `GPUInversionMutation` (plan line 332)

- `src/algorithms/strategies/mutation/kernels/inversion_mutation.cu` — 40 lines today. One thread per route; reverses `route[left..right]` when `probs[idx] < mutation_rate`. It is independent of `two_opt.cu` because reversal needs no distance matrix.
- `GPUInversionMutation` at `src/algorithms/strategies/mutation/inversion_mutation.py` line 112, tracking `h2d_bytes`, `d2h_bytes` and `kernel_launches` in the same shape as the other GPU strategies.
- Registered as `"GPUInversionMutation"` in `STRATEGY_REGISTRY` (`src/algorithms/strategies/__init__.py`, line 40).

### Task 7.4 — three 2-opt variants (plan line 351)

`src/algorithms/strategies/local_search/two_opt.py` now holds six classes: `CPUTwoOpt` (line 46), `NaiveGPUTwoOpt` (126), `BatchGPUTwoOpt` (221), and the Phase 7 additions `CPUFirstImprovementTwoOpt` (337), `CPUMultiMoveTwoOpt` (407) and `BatchGPUMultiMoveTwoOpt` (485).

- **First improvement** applies the first $\Delta < 0$ reversal it meets instead of finishing the sweep.
- **Multi-move** collects improving pairs whose reversal intervals do not overlap ($j_1 + 1 < i_2$) and applies them in the same pass, because their deltas are additive: $\Delta_{\text{total}} = \sum_k \Delta_k$. This is the direct answer to the question at step 1869 about applying the 5↔6 and 8↔10 swaps at once instead of rescanning.
- **`BatchGPUMultiMoveTwoOpt`** backs that with `two_opt_multi_move.cu` (126 lines today): threads evaluate candidate pairs in parallel and the collected non-conflicting improvements are applied together.

All three are registered in `STRATEGY_REGISTRY`.

### Task 7.5 — `datasets/tsp.duckdb` as the only schema (plan line 370)

- `src/data_models/problem.py` gained the metadata fields and `to_metadata_dict(optimal_cost)`.
- `src/loaders/database_loader.py` and `ingest_tsplib_directory_to_duckdb` in `src/loaders/tsplib_loader.py` were rewritten against the normalized schema (`problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `solutions`, `file_tracking`), reading `coords DOUBLE[][]` and `matrix INTEGER[][]` natively instead of parsing JSON.
- `DEFAULT_SELECTED_TSP_FILES` was deleted from `src/benchmarking/runner/config_loader.py` and replaced by `build_problem_configs_from_duckdb`. Verified today: searching `src/` for the constant returns no hits.
- `benchmark_scripts/run_selected_problems.py` owns the selection (`SELECTED_PROBLEMS = ("att48", "bayg29", "bays29", "berlin52", "brg180", "gr17", "gr21", "gr24", "gr48", "gr96")`).
- `src/benchmarking/runner/orchestration.py` writes `problem_metadata` into both checkpoint JSON and `problem_statistics`.

## Verification

Phase 7's own report artifact is `4f72da0b-6375-4cc1-ac90-dd8390461096/walkthrough.md`, sections 2.1–2.3:

```text
$ .venv/bin/pytest
22 passed in 32.88s
```

with the reported split `tests/test_benchmark_selected_problems.py` 6, `tests/test_modular_architecture.py` 14, `tests/validation/test_early_stopping.py` 1, `tests/validation/test_incremental_checkpoints.py` 1.

```text
$ .venv/bin/mypy src benchmark_scripts tests
Success: no issues found in 68 source files
```

```text
$ .venv/bin/ruff check src tests benchmark_scripts
All checks passed!
```

The same numbers are repeated in the Phase 7 closing message at step 2451 (2026-10-29T20:53Z), which also records that the plan registry and the `## Handoff` section were updated at `plan-reorganization-and-typing.plan.md` lines 260–337.

Current gate (2026-10-31), confirmed by the operator:

```text
$ .venv/bin/python -m pytest tests -q
711 passed, 6 xfailed

$ .venv/bin/python -m mypy src tests
Success: no issues found in 111 source files

$ .venv/bin/python -m ruff check src tests benchmark_scripts
All checks passed!
```

Six of the 711 are `xfail`; three of those six pin Phase 7 code — see below.

> [!NOTE]
> The gate above is quoted, not reproduced here. A re-run in this session was interrupted at `219 passed, 2 xfailed` by a `KeyboardInterrupt` raised at `src/benchmarking/runner/orchestration.py:495`, because the shell was shared with a concurrent agent. The Phase 7 figures (`22 passed`) are worth reading as "the whole suite at that moment", not as a subset that still exists in that form.

## Defects and deviations

1. **Eight Phase 7 files were reverted to `HEAD` and had to be restored.** The Phase 8 recovery request at `ea9a2b63-9509-4d3c-8a28-5461f8b7b756` step 723 (2026-10-30T17:42Z) describes "the broken working tree caused by 8 Phase 7 files being unintentionally reverted to HEAD" and names them as "`utils.route_helpers`, `loaders.database_loader`, `benchmarking.__init__`, and the 5 modular strategy files". The user's verdict in the same record is blunt: "This goal description is when you undone your work" and "You undid a job." Phase 8 Task 8.1 (plan line 397 onward) exists solely to restore them.
   The two sources do not enumerate the same eight: [analysis-phase-8-hermeticity-walkthrough.md](./analysis-phase-8-hermeticity-walkthrough.md) dates the reset to 2026-10-29 between 19:10 and 19:17 and lists `route_helpers.py`, `database_loader.py`, `two_opt.py`, `order_crossover.py`, `swap_mutation.py`, `tournament_selection.py`, `roulette_wheel_selection.py` and `mutation/__init__.py`. Both agree on the count and on the restoration path (reconstruct from the Phase 7 transcript rather than re-design); the third non-strategy file differs — `benchmarking/__init__.py` in the recovery goal, `mutation/__init__.py` in the Phase 8 record.
2. **Phase 7 was never committed.** `git log --oneline` puts `62edcbd` at `HEAD`, and the working tree is still dirty. The revert found files by returning them to *that* commit, which means a `git checkout` against `HEAD` was sufficient to erase the phase.
3. **`resolve_rng` never accepted the context generator it was written for.** The strict xfail at `tests/utils/test_unit_rng.py` line 163 records: "DEFECT: `src/utils/rng.py:56` resolve_rng only accepts `np.random.Generator`", while `ProblemContext.rng` is a `RandomState`. The same defect appears earlier as BUG-2 in [test-suite-walkthrough.md](./test-suite-walkthrough.md) ([section 6](./test-suite-walkthrough.md#6-defects-found-strict-xfails-not-fixed-in-this-change)), when the function still lived in `route_helpers.py`.
4. **The depot-endpoint guard is off by one route length.** `_mutation_bounds` (defined at `src/algorithms/strategies/mutation/inversion_mutation.py` line 38) only protects depot endpoints when `n > 3`; line 48 is the offending condition. A three-node `[depot, node, depot]` route can therefore invert across a depot endpoint. Two strict xfails pin it: `tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py` line 77 and `tests/algorithms/strategies/mutation/test_unit_swap_mutation.py` line 76. These are BUG-5 and BUG-6 in [test-suite-walkthrough.md](./test-suite-walkthrough.md#6-defects-found-strict-xfails-not-fixed-in-this-change).
5. **The plan's line anchors for this phase are stale.** Task 7.2 cites `types.py#L10-L54` (SeedLike is at line 17) and `route_helpers.py#L1-L61` (the file now runs past line 88). Task 7.4 cites `two_opt.py#L380-L597` while the three new classes sit at 337, 407 and 485. Task 7.3 cites `inversion_mutation.cu#L1-L55` for a 40-line file. Task 7.4 cites `two_opt_multi_move.cu#L1-L150` for a 126-line file.
6. **`tests/test_modular_architecture.py` is cited but does not exist.** Task 7.3, 7.4 and 7.5 all name it as the validation site; the suite was later reorganised (see [test-suite-walkthrough.md](./test-suite-walkthrough.md), [section 10.4](./test-suite-walkthrough.md#104-test-reorganization)).
7. **Task 7.2's Solution block contradicts the tree.** It lists `make_rng` and `resolve_rng` as `route_helpers.py` additions; they are in `src/utils/rng.py`. The plan acknowledges this at line 330, but the Solution block was not rewritten, so a reader who stops at the task text gets the wrong location.
8. **`LocalSearchStage` moved too.** The Phase 7 closing message points at `src/protocols/types.py#L18-L26`; the `Literal` is at line 35 today.

## Open questions

1. Should `resolve_rng` accept `np.random.RandomState`, or should `ProblemContext.rng` migrate to a `Generator` so one type is used everywhere?
2. Is `[depot, node, depot]` a legal chromosome in this solver? If it is not, should the mutation guards be widened rather than "fixed"?
3. Was the eight-file revert caused by an agent action, a `git checkout`, or an editor rollback — and is there a guard today that would catch it?
4. Should `restore_routes_shape(to_gpu=True)` be the only device-conversion path, or is per-strategy residency still allowed?
5. Should Task 7.2's `Where` field be rewritten to point at `src/utils/rng.py`, or is the line-330 warning the intended correction record?
