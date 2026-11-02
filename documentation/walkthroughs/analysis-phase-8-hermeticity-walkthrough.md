---
title: "Phase 8 — hermetic tests and strict CLI/library separation (reconstructed)"
description: >
  Reconstruction of Phase 8 of the reorganization plan from the Antigravity session
  transcripts: what the plan records, what the session states at each decision round,
  which alternatives were rejected and why, and where the plan, the transcripts and the
  current tree disagree.
created: 2026-10-31
type: analysis
modifications:
  - date_modified: 2026-10-31
    modifications:
      - description: >
          Created from `implementation_plan.md`, `phase_8_followup_plan.md` and
          `phase_8_walkthrough.md` in session `ea9a2b63`, the `transcript_full.jsonl`
          records of 2026-10-29T22:20–2026-10-30T19:24, the Copilot session log, and the
          plan file lines 397–622; re-verified the 2026-10-31 gate.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[test-suite-walkthrough.md](./test-suite-walkthrough.md)"
  - "[pyproject.toml](../../pyproject.toml)"
  - "[src/utils/rng.py](../../src/utils/rng.py)"
  - "[src/loaders/problem_type_handlers.py](../../src/loaders/problem_type_handlers.py)"
  - "[src/cli/benchmark.py](../../src/cli/benchmark.py)"
  - "[src/cli/tsplib.py](../../src/cli/tsplib.py)"
  - "[tests/conftest.py](../../tests/conftest.py)"
tags:
  - analysis/hermeticity
  - analysis/architecture
  - notes/timeline
  - notes/decisions
  - review/defects
  - guide/tests
  - guide/cli
  - algorithm/strategies
  - benchmark/test-suite
  - notes/session
---

# Phase 8 — hermetic tests and strict CLI/library separation (reconstructed)

## Context

Phase 8 answers four business questions about the routing solver package.

1. **Can the library be used outside this laptop clone?** Every loader and benchmarking entry point carried a workspace-relative default (`datasets/tsp`, `datasets/tsp.duckdb`, `PROJECT_ROOT / "configs"`, `results_v3/checkpoints`), and `datasets/tsp.duckdb` is gitignored. A fresh clone therefore imported fine and failed at runtime.
2. **Can the test suite prove the library works, or only prove that this machine is configured?** `tests/test_benchmark_selected_problems.py` imported `SELECTED_PROBLEMS` from a top-level experiment script, which is why `[tool.pytest.ini_options].pythonpath` had been widened to `["src", "."]`. Five of that file's six tests exercised library components, not the script.
3. **Who owns the terminal interface?** `argparse` blocks and `if __name__ == "__main__":` guards lived inside `tsplib_loader.py`, `import_checkpoints.py` and `validate_consistency.py`, so the library could not be imported without dragging CLI concerns along.
4. **What are the experiment scripts?** They are reproducible notebooks, run by editing constants, not a second CLI. `benchmark_scripts/` had to call the library with explicit paths and no `argparse`.

The plan records the resulting work as Tasks 8.1–8.7 in [plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L397 "Fase 8 heading"), then appends Tasks 8.10, 8.9 and 8.11 as follow-up (lines [552](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L552), [571](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L571), [594](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L594)).

**Citation convention.** Session artifacts live under `~/.gemini/antigravity/brain/ea9a2b63-9509-4d3c-8a28-5461f8b7b756`. "step N" is the `step_index` field of `.system_generated/logs/transcript_full.jsonl` in that session, with the record's `created_at` timestamp. "CP Lnnnn" is a line of the VS Code Copilot transcript `86299fac-0e20-41ba-99d0-171ed05945a7.jsonl`. Repository citations are `path:line`.

## Timeline

### Round 1 — 2026-10-29T22:20 — the `pythonpath` question opens the phase

The session opens with the user quoting a reviewer warning about `pyproject.toml:L43-L44` (step 0, `2026-10-29T22:20:07Z`): adding `"."` to `pythonpath` makes every root directory importable, contradicts the `src/` import-root comment and `[tool.setuptools.packages.find] where = ["src"]`, and makes pytest and mypy disagree about whether the code compiles.

Rejected alternative: **move `benchmark_scripts/` under `src/`** (the warning's own suggestion). It lost because the scripts are meant to be executed as notebooks, not packaged; the coupling was removed from the other side instead — the script-dependent test was deleted, `benchmark_scripts` was dropped from `known-first-party`, and `pythonpath` went back to `["src"]`. The root-cause chain is recorded in the session checkpoint: *"`tests/test_benchmark_selected_problems.py` imported `SELECTED_PROBLEMS` from `benchmark_scripts/run_selected_problems.py`, which caused the previous agent to add `"."` to `pytest.ini_options.pythonpath`"* (checkpoint, step 269, `2026-10-29T23:42:53Z`).

### Round 2 — the 8.1 restore-after-reset episode

Task 8.1 exists because of a self-inflicted regression. Between **19:10 and 19:17 on 2026-10-29**, eight `src/` files were reset to `HEAD` (`62edcbd`) while their consumers already imported the new APIs (step 77, `2026-10-29T23:03:54Z`: *"Eight files were silently reset to HEAD between 19:10 and 19:17. Checking whether that wiped Phase 7 work"*; repeated in checkpoint 269).

- **Chosen remedy:** reconstruct the eight files from the Phase 7 conversation `4f72da0b-6375-4cc1-ac90-dd8390461096`, `transcript_full.jsonl` steps 1998–2383, and re-apply them on top of `HEAD`. The eight files are listed in `phase_8_walkthrough.md` §1.1 and match the plan's `Where:` at [plan#L403](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L403): `route_helpers.py`, `database_loader.py`, `two_opt.py`, `order_crossover.py`, `swap_mutation.py`, `tournament_selection.py`, `roulette_wheel_selection.py`, `mutation/__init__.py`.
- **Rejected alternative:** accept the `HEAD` state and rebuild Phase 7 from the plan text. The user rejected this framing outright when reviewing the plan goal: *"This goal description is when you undone your work. … That was the plan, as shown in the plan. And if it isn't there, that's a flaw. If is, the flaw was in your analysis. You undid a job."* (step 723, `2026-10-30T17:42:51Z`). The blast radius — every strategy module importing `is_gpu_array` — made a from-scratch rebuild the larger risk.
- **Second-order episode (same round).** At `2026-10-29 22:24:38 -0300` the user reverted `src/utils/route_helpers.py` to its 63-line Task 7.1 form, which removed `make_rng`/`resolve_rng`/`is_gpu_array` and broke collection. A first investigator blamed the Phase 8 worker for writing truncated code; a second investigator refuted that claim with file timestamps and transcript steps (step 674, `2026-10-30T01:52:58Z`). This is the origin of Task 8.10: the RNG helpers had been bundled into `route_helpers.py` by Task 7.2 itself, which the plan still records as the decision.

### Round 3 — 2026-10-29T23:33 — plan approval and the batch gate

The implementation plan is approved (step 148, `2026-10-29T23:33:03Z`) as five sequential batches: 8.1 → (8.2, 8.3) → (8.4, 8.5) → (8.6, 8.7) → 8.8. Task 8.9 is written as *"When: Phase 9 or next technical debt refactoring batch"* — it was **not** part of the approved Phase 8 scope; it moved into Phase 8 later, at Round 4.

### Round 4 — 2026-10-30T01:24 — six questions, three new tasks

The user sends six follow-up questions (step 670, `2026-10-30T01:24:30Z`): why the scripts `return 0`, why `sys.exit(main())`, a Pylance `reportUnknownArgumentType`, whether RNG really belonged in `route_helpers.py`, whether the CLI is missing parameters, and a request — *"I can already feel the problem with the if statement per problem. But keep it that way, just add it as a task in the plan"* — which is what created Task 8.9.

`phase_8_followup_plan.md` then asks four questions and answers three of them against an alternative:

| Question | Chosen | Rejected alternative | Why the rejection |
| :-- | :-- | :-- | :-- |
| Q1 backwards-compatible re-exports of `make_rng`/`resolve_rng` in `route_helpers.py` | No | Keep a re-export | *"A re-export would keep the SRP violation visible through the old import path"* |
| Q2 move `is_gpu_array`/`to_cpu_batch` to `gpu_helpers.py` | Leave in `route_helpers.py` | Move them | `gpu_helpers.py` hard-imported CuPy at module top, so CPU-only callers would need CuPy; the proper move requires making that import lazy first |
| Q3 expose two-opt iterations as a dedicated flag | Generic `--strategy-param SLOT.KEY=VALUE` | `--two-opt-iterations` | The generic form covers every strategy knob, including future ones |
| Q4 ingest `.opt.tour` solutions now | Defer | Implement `--solutions-dir` | `datasets/tsp/` contained **0** `.opt.tour` files, so nothing could exercise it with real data |

The follow-up plan is approved at step 753 (`2026-10-30T18:43:53Z`).

### Round 5 — 2026-10-30T18:50–19:24 — execution order 8.10 → 8.11 → 8.9

`phase_8_followup_plan.md` fixes the order explicitly: *"Order: **8.10 → 8.11 → 8.9**"*. The plan file registers them in a different order (8.10 at line 552, 8.9 at 571, 8.11 at 594), so the registry no longer reflects the order in which the work happened. At step 886 (`2026-10-30T18:50:31Z`) the user adds the warning that now matters for this document: *"there are changes that were made and were not done by you. You can read the walkthrough … to see more."* Those changes were made in the parallel Copilot session and include deleting the test files the plan cites (see [Defects and deviations](#defects-and-deviations)).

## Changes

**Task 8.1 — restore eight reverted modules.** Mechanism: reconstruct from the Phase 7 transcript and re-apply, not re-design. Files as listed in Round 2. Success criterion: `python -c "import utils, algorithms, algorithms.strategies, benchmarking, loaders"` exits 0.

**Task 8.2 — explicit loader paths.** `TsplibLoader(tsp_dir)`, `ingest_tsplib_directory_to_duckdb(tsp_dir, db_path)` and `DatabaseLoader(db_path, fallback_tsp_dir=None)` lose their defaults, so a zero-argument call raises `TypeError` instead of reading the wrong directory. Mechanism: signature change plus docstring rewrite in `problem.py` and `problem_context.py`.

**Task 8.3 — benchmarking paths and `Namespace` removal.** `PROJECT_ROOT` and `default_config_dir()` are deleted from `config_loader.py` and from both `__init__` re-export lists; `CheckpointManager`, `ensure_checkpoint_dirs` and `import_checkpoints` require their directories; `run_comprehensive_benchmark` becomes pure keyword arguments with required `config_dir` and `output_dir`, and stops importing `argparse.Namespace`.

**Task 8.4 — `src/cli/` package.** New `src/cli/{__init__,tsplib,benchmark,checkpoints}.py`, each exposing `main(argv: Sequence[str] | None = None) -> int`, registered as `[project.scripts]`: `tsp-ingest`, `tsp-benchmark`, `tsp-import-checkpoints`, `tsp-validate-db`. Mechanism: move the parser out of the library module and keep the library function importable.

**Task 8.5 — `benchmark_scripts/` as notebooks.** The five scripts keep module-level constants (`PROJECT_ROOT`, `CONFIG_DIR`, `DB_PATH`, `OUTPUT_DIR`) and call the library directly. `validate_consistency.py` keeps `if __name__ == "__main__": sys.exit(main())` so the exit code reaches the shell while `main()` stays testable — the rationale is spelled out in step 674 (§2 of the investigator report).

**Task 8.6 — hermetic fixtures.** `pythonpath` back to `["src"]`, `"benchmark_scripts"` removed from `known-first-party`, five compact `.tsp` instances (~5.4 KB, covering `GEO`, `EXPLICIT` lower-diagonal, `EXPLICIT` upper-row, `ATT`, `EUC_2D`) plus three JSON configs under `tests/fixtures/`, and session-scoped fixtures in `tests/conftest.py` including a DuckDB ingested once into `tmp_path_factory`.

**Task 8.7 — test replacement.** `tests/test_benchmark_selected_problems.py` deleted; `tests/test_benchmarking_pipeline.py` (7 tests) and `tests/test_cli.py` (7 tests) created; `test_modular_architecture.py`, `test_early_stopping.py` and `test_incremental_checkpoints.py` moved onto the fixtures. This is the task whose file citations are now dead — see below.

**Task 8.10 — RNG extracted.** New [src/utils/rng.py](../../src/utils/rng.py) holds `make_rng` and `resolve_rng`; `route_helpers.py` keeps only array-shape and residency helpers; `utils/__init__.py` re-exports both from the new module; the eight stochastic strategy modules import `from utils.rng import …`; test cases move to `tests/utils/test_unit_rng.py`.

**Task 8.11 — CLI parity, seed bug, overwrite guard.** Four behaviour changes, not just flags:

1. `AlgorithmConfig.instantiate` gains `param_overrides` and `strategy_overrides`, applied **last** so a per-algorithm `params` block cannot silently shadow them; an unknown strategy slot or key raises `ValueError`.
2. `run_single_algorithm` now receives `seed_base=effective_seed` in [orchestration.py](../../src/benchmarking/runner/orchestration.py); previously repetitions fell back to the hardcoded `seed_base = 42` and ignored the configured seed.
3. `ingest_tsplib_directory_to_duckdb` gains `overwrite: bool = False` and raises `DatabaseNotEmptyError` instead of running `DELETE FROM` on seven tables; [src/cli/tsplib.py](../../src/cli/tsplib.py#L47) exposes `--overwrite` (and `--max-eager-matrix-dimension` at line 52).
4. `--local-search-stage` choices are derived from `LocalSearchStage` instead of the non-existent `never/all_generations` list, and `GeneticAlgorithm.__init__` rejects unknown stages.

New flags on [src/cli/benchmark.py](../../src/cli/benchmark.py): `--patience`, `--seed`, `--population-size`, `--mutation-rate`, `--strategy-param` (line 157), `--incremental-runs`/`--no-incremental-runs` via `argparse.BooleanOptionalAction` (line 122), `--log-level`, `--local-search-stage` (line 109).

**Task 8.9 — polymorphic `DatabaseLoader`.** New [src/loaders/problem_type_handlers.py](../../src/loaders/problem_type_handlers.py): `ProblemTypeHandler(Protocol)` with `list_problems(conn)` and `load_metadata(conn, base)`, a frozen `BaseProblemRow` carrier, concrete `TspHandler`/`AtspHandler`/`CvrpHandler`, the `PROBLEM_TYPE_HANDLERS` registry and `register_problem_type_handler()`. `DatabaseLoader` takes `handlers=` for injection and raises `UnsupportedProblemTypeError` for an unknown type instead of falling into the CVRP branch (the original symptom: a `VRPTW` row loaded as CVRP and failed with a misleading `Missing record in cvrp_problems`).

## Verification

Transcript-era observations, in order. The first block is the Phase 8 gate, the second the follow-up gate.

```text
$ .venv/bin/python -m pytest tests            # step 559, 2026-10-29T23:58:43Z
collected 30 items
tests/test_benchmarking_pipeline.py .......                              [ 23%]
tests/test_cli.py .......                                                [ 46%]
tests/test_modular_architecture.py .F............                        [ 93%]
tests/validation/test_early_stopping.py .                                [ 96%]
tests/validation/test_incremental_checkpoints.py .                       [100%]
FAILED tests/test_modular_architecture.py::test_database_loader_fallback_and_duckdb_ingestion
  data_models.exceptions.InstanceNotFoundError: Instance 'brg180' not found in database.
1 failed, 29 passed in 10.91s

$ .venv/bin/python -m pytest tests            # step 567, 2026-10-29T23:59:17Z
30 passed in 9.61s

$ .venv/bin/python -m pytest tests            # step 632, 2026-10-30T00:04:06Z
30 passed in 8.22s

$ pytest --rootdir=<repo> -c <repo>/pyproject.toml <repo>/tests   # step 634
30 passed in 8.76s

$ .venv/bin/python -m mypy src tests benchmark_scripts           # step 636
Success: no issues found in 74 source files
```

The `brg180` failure is the observation that fails without the fix: the test asked a fixture-only DuckDB for an instance that only the full corpus contains, which is exactly the hermeticity defect Phase 8 was removing.

```text
$ .venv/bin/python -m pytest tests                 # step 902, 2026-10-30T19:23:31Z
559 passed, 6 xfailed in 12.00s

$ .venv/bin/python -m mypy src benchmark_scripts   # step 904, 2026-10-30T19:23:35Z
Success: no issues found in 70 source files

$ grep -n 'problem_type ==\|ptype ==' src/loaders/database_loader.py
0 hits

$ grep -rn "argparse" src/ --include=*.py --exclude-dir=cli
0 hits
```

Re-verified on 2026-10-31 in this workspace:

```text
$ .venv/bin/python -m pytest tests -q
711 passed, 6 xfailed in 11.70s

$ .venv/bin/python -m mypy src tests
Success: no issues found in 111 source files

$ .venv/bin/python -m ruff check src tests benchmark_scripts
All checks passed!
```

The test counts are not comparable across rounds — 30 at the Phase 8 gate, 559 at the follow-up gate, 703 at the Phase 9 baseline and 711 today — because the suite was restructured and grown in a parallel session between `2026-10-30T19:24` and `2026-10-30T21:38`.

## Defects and deviations

1. **Four test files cited by Phase 8 no longer exist, and no task records their removal.** [plan#L535](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L535) and [plan#L544-L546](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L544) cite `tests/test_benchmarking_pipeline.py`, `tests/test_cli.py` and `tests/test_modular_architecture.py`; all three are absent today. They were deleted in the Copilot session with `rm -f tests/test_cli.py tests/test_benchmarking_pipeline.py tests/test_modular_architecture.py` under the goal *"Delete the three legacy flat test files whose replacements now pass"* (CP L3007, `2026-10-30T18:08:37`), following the user's split instruction at CP L1668 (`2026-10-30T17:58:23`). No Phase 8 or Phase 9 task mentions the deletion.
2. **`tests/strategy_fixtures.py` is cited once, by Task 9.3, not by Phase 8.** The file did not exist during Phase 8: it was created at CP L3435/L3460 (`2026-10-30T19:44`) as a rename of `tests/algorithms/strategies/conftest.py`, then deleted by Task 9.3 ([plan#L679](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L679)). Treating it as a Phase 8 artifact is a mis-attribution.
3. **The type gate was narrowed, not just re-run.** Phase 8 closed on `mypy src tests benchmark_scripts` (74 files); the follow-up closed on `mypy src benchmark_scripts` (70 files). The reason is visible at step 839 (`2026-10-30T18:47:55Z`): `mypy src tests` reported an error on `tests/conftest.py` because two `conftest.py` modules mapped to the same module name. `tests` came back into the gate only later, once `explicit_package_bases = true` was set ([pyproject.toml](../../pyproject.toml#L65)); today `mypy src tests` reports 111 files.
4. **The plan's task order contradicts the execution order.** The plan lists 8.10 → 8.9 → 8.11 (lines 552, 571, 594); `phase_8_followup_plan.md` states *"Order: 8.10 → 8.11 → 8.9"*, and the checkpoint of 19:22 presents them in that order.
5. **Task 8.1's Solution count does not match its inventory.** The Solution says *"0 erros nos 24 arquivos de estratégia e loaders"* ([plan#L420](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L420)) while the task restores 8 files.
6. **Task 8.9 was executed outside its recorded window.** The approved implementation plan says `When: Phase 9 or next technical debt refactoring batch`; it ran in Phase 8 at the user's explicit request (step 670, question 6).
7. **The `datasets/` cleanup has a docstring residue.** The Phase 8 walkthrough reports `grep -rn "datasets/" src/` as *"PASS (0 code matches)"*. Today the command yields four documentation lines in `src/loaders/database_loader.py` (lines 5, 8, 20, 59) plus stale `.pyc` hits. The claim survives only under a "code, not docstrings" reading.
8. **`tests/gpu_probe.py` and `tests/strategy_fixtures.py` still exist as bytecode.** `grep -rn "gpu_probe" tests/` returns two hits today, both `tests/__pycache__/gpu_probe.cpython-31{2,3}.pyc`. Any Gate phrased as "grep returns 0" is sensitive to stale caches.

## Open questions

1. Should Task 8.7's `Where:`/`Solution:` be rewritten to cite the surviving files (`tests/benchmarking/test_integration_pipeline.py`, `tests/cli/*`, `tests/loaders/test_integration_loaders.py`, `tests/algorithms/test_integration_ga_composition.py`), or does `update-plan.instructions.md` forbid touching an executed task's evidence?
2. Is the narrowed type gate (`mypy src benchmark_scripts`, 70 files) acceptable as Phase 8's recorded evidence, given that Phase 9 closed on `mypy src tests` (111 files)?
3. Should `~/.pyc` hits be excluded from the DoD greps, or should the repository standardise on `git grep` so caches cannot satisfy or defeat a criterion?
4. Does Task 8.9's ingestion-side counterpart (`if p.problem_type == "TSP"` in `tsplib_loader.py`) warrant a task now, or does it stay a pending note?
5. Which order is authoritative for the follow-up tasks — the plan's numeric order or the execution order recorded in `phase_8_followup_plan.md`?
