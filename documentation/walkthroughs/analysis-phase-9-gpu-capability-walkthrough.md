---
title: "Phase 9 — GPU capability, archive isolation and config contract (reconstructed)"
description: >
  Reconstruction of Phase 9 of the reorganization plan from the Antigravity session
  transcripts: the seven fixed decisions DP9-1 to DP9-7, the alternatives each one
  displaced, the six review rounds on 2026-10-30 that produced them, and the places
  where the plan, the transcripts and the current tree disagree.
created: 2026-10-31
type: analysis
modifications:
  - date_modified: 2026-10-31
    modifications:
      - description: >
          Created from `phase_9_plan.md` in session `ea9a2b63`, its
          `transcript_full.jsonl` records of 2026-10-30T21:09–2026-10-31T00:02, the
          Copilot session log, plan lines 623–830 and walkthrough §10.6–10.8;
          re-verified the 2026-10-31 gate and the archived-baseline smoke script.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[test-suite-walkthrough.md](./test-suite-walkthrough.md)"
  - "[analysis-phase-8-hermeticity-walkthrough.md](./analysis-phase-8-hermeticity-walkthrough.md)"
  - "[src/protocols/backend.py](../../src/protocols/backend.py)"
  - "[src/algorithms/archive/full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py)"
  - "[src/algorithms/archive/README.md](../../src/algorithms/archive/README.md)"
  - "[benchmark_scripts/check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py)"
  - "[tests/protocols/test_unit_gpu_capability.py](../../tests/protocols/test_unit_gpu_capability.py)"
  - "[tests/benchmarking/test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py)"
  - "[configs/algorithms.json](../../configs/algorithms.json)"
tags:
  - analysis/architecture
  - analysis/decisions
  - algorithm/gpu
  - algorithm/archive
  - benchmark/configs
  - benchmark/baseline
  - notes/timeline
  - notes/decisions
  - review/defects
  - guide/testing
  - notes/session
---

# Phase 9 — GPU capability, archive isolation and config contract (reconstructed)

## Context

Phase 9 closes four gaps that Phase 8 exposed.

1. **GPU capability belongs to the core, not to the tests.** The only device check lived in `tests/gpu_probe.py`, while [problem_context.py](../../src/protocols/problem_context.py) tested whether CuPy was importable and never whether a device existed; [gpu_helpers.py](../../src/utils/gpu_helpers.py) imported CuPy at module top, so importing the library on a CPU-only machine failed. A published package offered no way to ask "is there a GPU?" and no actionable error when the answer was no.
2. **Valid input must be rejected at the boundary.** With `n = 0`, `patience` collapses to 0, every empty route costs 0.0, and `evolve` returns a fictitious optimum — which masks a loader or filtering bug rather than reporting it ([plan#L699](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L699)).
3. **The archived monolithic baseline must not be part of the product.** `GeneticAlgorithmFullGPUMonolithic` was exported from `algorithms.__all__` and parametrised into two behavioural suites, and five legacy modules under `src/algorithms/archive/` imported modules that no longer existed. The thesis still needs the baseline; the package should not advertise it and the suite should not execute it.
4. **The production configuration needs a contract.** No test read `configs/algorithms.json`, so renaming or dropping a variant only in the production file kept the suite green and broke the real benchmark.

**Citation convention.** Session artifacts live under `~/.gemini/antigravity/brain/ea9a2b63-9509-4d3c-8a28-5461f8b7b756`. "step N" is the `step_index` of `.system_generated/logs/transcript_full.jsonl` plus its `created_at`. "CP Lnnnn" is a line of `86299fac-0e20-41ba-99d0-171ed05945a7.jsonl`. Repository citations are `path:line`.

## Timeline

Unlike Phase 8, Phase 9 started with the decisions already written: **DP9-1 to DP9-7 were authored by the user in the plan** ([plan#L627-L633](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L627)) before any code. The session's work was to audit them, then implement them.

### Round 1 — 2026-10-30T21:09 — "I just updated Fase 9; audit it" (step 918)

The user hands over the freshly written Fase 9 and asks for an investigation of doubtful decisions and, if none remain, a plan for the planned tasks. The request also states the repository constraint that shapes the whole phase: every dispatched worker must follow the visible rules *and* the unsaid ones, because *"the repo shall not be a mess."*

### Round 2 — 2026-10-30T21:38 — the audit that corrected the audit (step 922)

A DeepInvestigator report arrives with two purposes: refuting a previous investigator, and auditing Fase 9.

**Baseline measured on the real machine:** `703 passed, 9 xfailed in 14.37s`, `mypy src tests` clean in 113 files, `ruff` clean, and an **NVIDIA GeForce GTX 1050** with `getDeviceCount() == 1`. The earlier investigator's claim that CUDA was inactive is labelled a *"FATAL FACTUAL ERROR"*; running `pytest -m gpu` produced `10 passed, 2 xfailed in 1.83s`, the two xfails being the `best_cost_history` defect (BUG-7).

**Eight issues raised**, each with a resolution that later appears in `phase_9_plan.md` §1:

| # | Issue | Resolution adopted |
| :-- | :-- | :-- |
| 1 | Task 9.10 said "script com argparse mínimo", contradicting the zero-`argparse` rule of Task 8.5 | pure python script with module constants |
| 2 | Task 9.9 filtered archive only in the composition test, but `test_early_stopping.py` also calls `evolve()` on every configured variant | extend the filter to early stopping |
| 3 | Task 9.3 removed `tests` from `mypy_path` while seven strategy test modules imported `from strategy_fixtures import population` | migrate `population` into `conftest.py`, then delete the module |
| 4 | Task 9.9's DoD ("no test imports `algorithms.archive`") contradicts Task 9.8 and `test_benchmark_algorithms_json_loads_all_classes` | restate the DoD as "no behavioural test runs `evolve()` on archive variants" |
| 5 | `cupy_backend()` return contract unspecified | `gpu_available() -> bool`, `require_gpu() -> Any`, `cupy_backend() -> Any \| None` |
| 6 | `CUPY_CACHE_DIR` breaks in read-only homes | set the fallback inside `backend.py` |
| 7 | `gpu_probe` orphaned in `known-first-party` | remove it with the file |
| 8 | `archive/README.md` needs frontmatter | add frontmatter compliance to Task 9.6 |

Issue 5 also carries a finding the plan never absorbed: `full_gpu_monolithic.py` *"accepts `customers` without validation, allocating 0-sized device buffers and launching CUDA kernels with `n = 0`"* — so the empty-input guard of Task 9.5 had to be mirrored in the archive.

### Round 3 — 2026-10-30T21:39 — `phase_9_plan.md` and five batches (step 939)

The execution plan is written: the eight-row resolution table, a target-architecture diagram, and five ordered batches (9.1+9.4 → 9.2+9.3 → 9.5+9.11 → 9.6+9.7+9.8+9.9+9.10 → 9.12), with a DoD table that fixes the phase target at **≥704 passed and exactly 6 xfails**.

### Round 4 — 2026-10-30T23:09 — the user adjudicates seven points (step 940)

Seven inline comments on `phase_9_plan.md`. Three changed the plan, and they are the ones the plan file does not explain:

- **Archive in tests — rejected mechanism.** The plan proposed filtering archive variants in `test_early_stopping.py` with `if cfg.class_module and "archive" in cfg.class_module: continue`. The user rejected it: *"Nah, files on archive do not need tests. They just need to be updated. Just remove them from there and don't validate them. Tests shouldn't depend on archive files. They are archive. I still need them and to instantiate them in a script, for example, but they are not part of the suite."* The compromise that survived is narrower: removal from the parametrised lists, with the `continue` filter still present in [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L39).
- **`CUPY_CACHE_DIR` — put on hold.** *"Why is this useful? Do not do this task yet. Let's put it on hold on this turn."* Issue 6 was therefore **not** applied to `src/protocols/backend.py`; it survives in [tests/conftest.py](../../tests/conftest.py#L15) and [check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py#L10) instead, and is recorded as an open decision in the Handoff ([plan#L828](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L828)).
- **README plus plan update.** *"Do that. Also, update the planning files after finishing."*

The other four comments were confirmations ("Concordo", "Certo", *"Just don't test the monoliths"*, and the SRP remark *"Tests can import from src, but makes 0 SENSE importing from tests in the src code"*).

### Round 5 — 2026-10-30T23:57 — worker report: four defects and a fifth guard (step 944)

The worker reports six problems in the first pass. Five are fixed immediately:

1. the archived monolithic lacked the empty-`customers` guard (issue 5 of Round 2, never transcribed into a task);
2. `ProblemContext.__init__` still documented `ImportError` after the refactor to `GPUUnavailableError`;
3. the Handoff still linked `src/algorithms/monolithic/full_gpu_monolithic.py`, a directory that no longer exists;
4. the plan's frontmatter carried four tags under an unapproved `plan/` namespace instead of 10–15 under approved namespaces;
5. `conftest.py` exposed only `make_population`, so a `population` fixture alias was added.

The sixth is environmental: `rtk pytest` resolved `/usr/bin/pytest` and failed with `ModuleNotFoundError: No module named 'duckdb'`.

### Round 6 — 2026-10-31T00:02 — self-review, no subagents (step 1022)

The user had asked for `/code-review-and-quality` **without subagents**, and the review runs in the orchestrator. Verdict: **APROVADO**. The diffstat covering the whole working tree since `HEAD` is 64 files, **+1,934 / −6,279**, of which Phase 9 deletes `src/algorithms/monolithic/*`, `src/algorithms/presets.py` and the five legacy archive modules, and adds `src/protocols/backend.py` (77 lines).

### DP9-1 to DP9-7: what each decision displaced

| Decision | Chosen | Alternative rejected | Why it lost |
| :-- | :-- | :-- | :-- |
| **DP9-1** GPU API location | `gpu_available()`, `require_gpu()`, `cupy_backend()` in `src/protocols/backend.py`; no `__init__` probes a device at import | Leave the device check in `tests/gpu_probe.py`; probe at import time | A test helper cannot be the only implementation of a capability the package ships; a package-level probe makes `import` fail on CPU |
| **DP9-2** failure contract | `GPUUnavailableError(RuntimeError)` covers "CuPy absent" and "no device"; `ValueError` stays reserved for `get_gpu_*` on a NumPy backend | Reuse `ImportError` for a missing package | The incumbent contract *was* `ImportError` — step 944 item 2 shows the stale docstring still promising it; one error type per user action beats two |
| **DP9-3** packaging | `cupy-cuda12x` stays a required dependency; every `import cupy` becomes lazy | Make CuPy an optional extra | Changing the install contract for existing users was not on the table; lazy imports already deliver CPU-only `import` ([pyproject.toml](../../pyproject.toml#L27)) |
| **DP9-4** archive | Move to `src/algorithms/archive/`, drop exports from `algorithms.__init__`, delete the five unimportable legacy modules | Keep `src/algorithms/monolithic/` as a first-class package | *"O monolítico não faz parte da arquitetura modular e não deve ser anunciado na API do pacote"* ([plan#L714](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L714)) |
| **DP9-5** configurations | Keep `FullGPU`/`FullGPUMonolithic` in the default config pointing at `algorithms.archive…`; keep archive out of behavioural tests; cover the baseline with a smoke script | Delete the two variants from the config; or keep them in the test matrix behind the "archive" filter | Deleting them removes the thesis baseline; the filter keeps executing archive code, which the user rejected at step 940 |
| **DP9-6** empty input | `evolve` raises `ValueError` naming `customers` before computing `patience` | Let `n = 0` run | It yields a fictitious optimum that hides a loader bug |
| **DP9-7** config parity | Compare structure, variant names, class paths and parameter keys; numeric tuning values may differ; descriptions are synchronised | Byte-for-byte JSON equality | Research and test configs legitimately differ in tuning values but not in shape |

## Changes

**Task 9.1 — capability API.** [src/protocols/backend.py](../../src/protocols/backend.py) gains a lazily cached `_probe_gpu()` (`functools.lru_cache(maxsize=1)`, `importlib.util.find_spec`, `cp.cuda.runtime.getDeviceCount()`), `gpu_available() -> bool`, `require_gpu() -> Any`, `cupy_backend() -> Any | None` and `clear_gpu_cache() -> None`. `GPUUnavailableError(RuntimeError)` is added to `exceptions.py` and re-exported from `data_models`.

**Task 9.2 — consumer adoption.** `problem_context.py`, `gpu_helpers.py` and the archived monolithic delegate to `require_gpu()`; `import cupy` moves under `TYPE_CHECKING` and into function bodies; NumPy-backend accessors keep raising `ValueError`.

**Task 9.3 — retire the test-side shim.** `tests/gpu_probe.py` and `tests/strategy_fixtures.py` are deleted; tests import `gpu_available`/`cupy_backend` from `protocols.backend`; the permutation helper becomes the `make_population` fixture in [tests/conftest.py](../../tests/conftest.py#L47) (with a `population` alias at line 59); `pythonpath` and `mypy_path` are `["src"]` ([pyproject.toml](../../pyproject.toml#L49), [L62](../../pyproject.toml#L62)); `gpu_probe` leaves `known-first-party`.

**Task 9.4 — capability tests.** [tests/protocols/test_unit_gpu_capability.py](../../tests/protocols/test_unit_gpu_capability.py) covers five CPU-only cases: CuPy absent, `getDeviceCount() == 0`, driver exception, device present, and cache reuse cleared by `clear_gpu_cache()`.

**Task 9.5 — empty customers.** `if len(customers) == 0: raise ValueError("customers sequence cannot be empty")` at [genetic_algorithm.py:453](../../src/algorithms/metaheuristics/genetic_algorithm.py#L453); the same guard is mirrored at [full_gpu_monolithic.py:119](../../src/algorithms/archive/full_gpu_monolithic.py#L119). The `xfail(strict=True)` comes off `test_empty_customers_are_rejected`.

**Task 9.6 — archive migration.** `git mv` of `full_gpu_monolithic.py` and `kernels/` into `src/algorithms/archive/`; the five legacy modules deleted; `archive/__init__.py` exposes only `GeneticAlgorithmFullGPUMonolithic`; [README.md](../../src/algorithms/archive/README.md) gets repository frontmatter; the statistics key is corrected from `best_cost_history` to `convergence_history` at [full_gpu_monolithic.py:248](../../src/algorithms/archive/full_gpu_monolithic.py#L248), which resolves BUG-7; exports leave `algorithms.__all__`.

**Task 9.7 — class paths.** `class_module` becomes `"algorithms.archive.full_gpu_monolithic"` for both variants in [configs/algorithms.json](../../configs/algorithms.json#L164) and in the fixture ([L175](../../configs/algorithms.json#L175) is the second variant in both files).

**Task 9.8 — config contract.** [tests/benchmarking/test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py) loads both JSON files and asserts equal sections, variant sets, `class_module`/`class_name`, strategy slots and parameter keys, and that every production variant imports (`test_production_variants_importable`, line 83).

**Task 9.9 — archive out of the suite.** The archive variants are removed from the two `parametrize` lists in `tests/algorithms/test_integration_ga_composition.py`; the two `xfail(strict=True)` markers for BUG-7 are deleted; the early-stopping loop keeps the `"archive" in cfg.class_module` skip.

**Task 9.10 — baseline smoke.** [benchmark_scripts/check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py) is a pure python script (zero `argparse`) with `PROBLEM_PATH`, `MAX_GENERATIONS = 10`, `PATIENCE = 5`, `POPULATION_SIZE = 64`, `SEED = 42`, returning 2 when no device is visible, 1 on contract violation, 0 on success. It validates nine required statistics keys, the absence of `best_cost_history` **in the returned dict**, `len(convergence_history) == generations_completed + 1`, and tour length.

**Task 9.11 — typing cleanup.** `AlgorithmConfig.params`/`.strategies` use `dc_field(default_factory=dict)`; the two `type: ignore` comments and the `default_config_dir` docstring reference disappear.

**Task 9.12 — documentation.** Section 10.8 is appended to [test-suite-walkthrough.md](./test-suite-walkthrough.md#L259), every Task 9.x `Solution` block is filled in, and the Handoff is rewritten.

## Verification

```text
# Phase 9 baseline, step 922 / 934 (2026-10-30T21:38–21:38:51Z)
703 passed, 9 xfailed in 14.37s
Success: no issues found in 113 source files
All checks passed!

$ pytest -m gpu          # step 922, live GTX 1050
10 passed, 2 xfailed in 1.83s

# Phase 9 close, step 956 (2026-10-30T23:57:59Z)
collected 717 items
711 passed, 6 xfailed in 12.79s
Success: no issues found in 111 source files
All checks passed!

$ .venv/bin/python benchmark_scripts/check_archive_baseline.py     # step 944
[SUCCESS] Archive baseline smoke test passed!
          Best fitness: 3323.0
          Generations: 6
          Stop reason: no_improvements
exit code 0
```

The 9 → 6 xfail reduction is exactly BUG-7 (two parametrised cases) plus BUG-8 (one case). Baseline was 703 passed, target was "≥704 passed, exactly 6 xfails"; the close reports 711 passed and 6 xfailed over 717 collected items, so `711 + 6 = 717` is internally consistent.

Re-verified on 2026-10-31 in this workspace:

```text
$ .venv/bin/python -m pytest tests -q
711 passed, 6 xfailed in 11.70s

$ .venv/bin/python -m mypy src tests
Success: no issues found in 111 source files

$ .venv/bin/python -m ruff check src tests benchmark_scripts
All checks passed!

$ .venv/bin/python benchmark_scripts/check_archive_baseline.py
[INFO] Loading problem from tests/fixtures/tsp/burma14.tsp...
[INFO] Running evolve for up to 10 generations (patience=5)...
[SUCCESS] Archive baseline smoke test passed!
          Best fitness: 3323.0
          Generations: 6
          Stop reason: no_improvements
exit code 0
```

The smoke script reproduces the transcript-era numbers bit for bit, which is the strongest available evidence that the archived baseline still executes and honours the statistics contract.

## Defects and deviations

1. **Task 9.6's DoD is not met as written.** The DoD requires `grep -rn "monolithic" src/algorithms/__init__.py` to return 0 ([plan#L721](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L721)); that same file's module docstring contains *"- `archive/`: Legacy pre-modularization implementations and monolithic baseline"* at [line 8](../../src/algorithms/__init__.py#L8). The Solution admits the residue (*"retorna apenas comentário de documentação"*), so the criterion and the evidence contradict each other in the same entry.
2. **Task 9.9's DoD is false today, and was never corrected.** It still reads *"nenhum teste importa `algorithms.archive`"* ([plan#L766](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L766)), although audit issue 4 and the user's *"Just don't test the monoliths"* both asked for the narrower wording. Two tests import the archive package: [test_integration_ga_composition.py:149-161](../../tests/algorithms/test_integration_ga_composition.py#L149) resolves and instantiates **every** variant in the fixture config, and [test_unit_algorithms_config_contract.py:83](../../tests/benchmarking/test_unit_algorithms_config_contract.py#L83) imports production class paths. Only the narrower invariant holds: no parametrised test calls `evolve()` on an archive variant.
3. **The `archive` filter the user rejected still runs.** DP9-5 states that behavioural tests do not execute archive code, yet [test_early_stopping.py:39](../../tests/validation/test_early_stopping.py#L39) keeps `if cfg.class_module and "archive" in cfg.class_module: continue` — the exact mechanism rejected at step 940. The plan's Task 9.9 `Solution` records the filter, so the deviation is disclosed but never reconciled with the instruction that produced it.
4. **Six strict xfails remain and no task registers them.** The current markers are [test_unit_statistics_comparison.py:173](../../tests/benchmarking/test_unit_statistics_comparison.py#L173), [test_unit_checkpoint_io.py:240](../../tests/benchmarking/test_unit_checkpoint_io.py#L240), [test_unit_inversion_mutation.py:77](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py#L77), [test_unit_swap_mutation.py:76](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L76), [test_unit_problem_validation.py:124](../../tests/data_models/test_unit_problem_validation.py#L124) and [test_unit_rng.py:163](../../tests/utils/test_unit_rng.py#L163). The plan mentions xfails only as counts ([plan#L823](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md#L823)).
5. **Three incompatible `D`-numberings coexisted — now resolved.** The walkthrough used `D1` = `Problem.__eq__`, `D2` = `resolve_rng`, `D3` = zero variance, `D4` = checkpoint validity, `D5` = swap mutation, `D6` = inversion mutation; the Phase 9 checkpoint used a different order (`D1` = 3-node mutation, `D3` = checkpoints, `D4` = zero variance, `D5` = `Problem.__eq__`, `D6` = `resolve_rng`); and the plan used `D1`–`D5` for *design decisions* (typing, layout, imports, checkpoints, early stopping). The three schemes are unified as `BUG-1`–`BUG-8` (defects, same order as `D1`–`D8`) and `DP-1`–`DP-5` (decisions) in [table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md).
6. **Task 9.3's grep DoD is cache-sensitive.** *"`grep -rn "gpu_probe" tests` retorna 0"* returns two hits today, both `tests/__pycache__/gpu_probe.cpython-31{2,3}.pyc`.
7. **The `CUPY_CACHE_DIR` hold was honoured in the library only.** `src/protocols/backend.py` has no fallback, but [tests/conftest.py:15-16](../../tests/conftest.py#L15) and [benchmark_scripts/check_archive_baseline.py:10-11](../../benchmark_scripts/check_archive_baseline.py#L10) both set one. The Handoff records the hold as an open decision, so this is a deliberate split rather than drift — but it means the library still fails on a read-only home.
8. **The archive is outside both static gates.** `mypy` excludes `^src/algorithms/archive/` ([pyproject.toml:69](../../pyproject.toml#L69)) and `ruff` excludes `src/algorithms/archive` ([pyproject.toml:89](../../pyproject.toml#L89)). "0 errors in 111 source files" therefore says nothing about the baseline; only `check_archive_baseline.py` covers it.
9. **Task 9.5's scope grew without a record.** The guard in `full_gpu_monolithic.py` was added in Round 5 because the audit flagged it; the Task 9.5 `Solution` lists only `genetic_algorithm.py` and the boundary test.
10. **Task 9.12's recorded path is stale.** At execution time the walkthrough lived at `documentation/analysis/test-suite-walkthrough.md` (step 1011 read that path); it now lives at `documentation/walkthroughs/test-suite-walkthrough.md` and `documentation/analysis/` holds no such file.

## Open questions

1. Should Task 9.6's DoD be rewritten as "no symbol of archive in `algorithms.__all__`" instead of a grep for a word that legitimately appears in a docstring?
2. Should Task 9.9's DoD adopt the audit's wording (no `evolve()` on archive variants) now that two tests provably import the package?
3. ~~Should the six remaining xfails be registered as Phase 10 tasks with the walkthrough's D1–D6 numbering, and the plan's design-decision IDs renamed to avoid the collision?~~ **Answered:** the six are registered as `BUG-1`–`BUG-6` in [table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md), and the plan's decisions became `DP-1`–`DP-5`.
4. Is the `CUPY_CACHE_DIR` hold still intended, given that the fallback already lives in `conftest.py` and the smoke script?
5. Should `src/algorithms/archive/` stay excluded from mypy and ruff, or is a relaxed per-directory configuration preferable to no coverage at all?
