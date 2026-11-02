---
title: "Testing rules consolidation and suite build-out — walkthrough"
description: >
  Record of consolidating the instruction files into `.agents/rules`, merging the
  two routing_data testing rules, writing repo-specific testing standards for
  gpu_accelerated_clean, and building 517 new tests across five parallel batches,
  including the decisions taken, the alternatives rejected, and the defects found.
created: 2026-10-30
type: analysis
modifications:
  - date_modified: 2026-10-30
    modifications:
      - description: >
          Created the walkthrough after consolidating the rule files, running the
          five test batches, removing one speculative test, and verifying the full
          suite (541 passed, 6 xfailed, 0 warnings).
      - description: >
          Round 2: corrected both testing rules (current-state intros, descriptive
          frontmatter, config-input mechanism, responsibility-based splitting, full
          TSPLIB rounding table, EXPLICIT error guidance), split every oversized and
          legacy test file into responsibility-scoped files, and re-verified the 547
          collected tests.
      - description: >
          Round 4 (Fase 9): GPU capability API moved to core protocols.backend (lazy import),
          tests/gpu_probe.py retired and tests removed from pythonpath/mypy_path, make_population
          fixture moved to conftest.py, empty customers validation resolved (D8 fixed),
          archive/ isolated from test suite with D7 resolved in full_gpu_monolithic, configs
          structural contract test added, and full test suite verified (711 passed, 6 xfailed).
related_files:
  - "[testing.instructions.md](../../.agents/rules/testing.instructions.md)"
  - "[sequential-thinking.instructions.md](../../.agents/rules/sequential-thinking.instructions.md)"
  - "[markdown.instructions.md](../../.agents/rules/markdown.instructions.md)"
  - "[testing.instructions.md (routing_data)](../../../routing_data/.agents/rules/testing.instructions.md)"
  - "[conftest.py](../../tests/conftest.py)"
  - "[pyproject.toml](../../pyproject.toml)"
tags:
  - analysis/testing
  - analysis/instructions
  - analysis/consolidation
  - notes/session
  - notes/decisions
  - notes/risks
  - guide/testing
  - guide/agents
  - review/findings
  - review/defects
  - benchmark/test-suite
  - notes/pytest
---

# Testing rules consolidation and suite build-out — walkthrough

## 1. What shipped

- One instruction source per repository: every rule now lives in `.agents/rules/`, with the Antigravity key block and the VS Code key block in the same YAML frontmatter, separated by comments.
- `.github/instructions/` is empty in both repositories; the duplicated bodies were deleted after the unique content was merged forward.
- Two new merged testing rules: `gpu_accelerated_clean/.agents/rules/testing.instructions.md` and `routing_data/.agents/rules/testing.instructions.md`.
- 517 new tests in `gpu_accelerated_clean`, developed by five parallel subagents under those rules.
- Pytest marker registry (`gpu`, `slow`) and `testpaths` added before any test used a custom mark.

## 2. Instruction consolidation

| Repository | File | Action |
| :-- | :-- | :-- |
| gpu_accelerated_clean | `exploration.instructions.md` | `.github` copy deleted (body identical to `.agents`) |
| gpu_accelerated_clean | `markdown.instructions.md` | `<referencing_rules>` block merged into `.agents`; `.github` copy deleted |
| gpu_accelerated_clean | `sequentialthinking.instructions.md` | `.github` copy deleted (body identical) |
| gpu_accelerated_clean | `testing.instructions.md` | New, repo-specific |
| routing_data | `documentation.instructions.md` | `.github` copy deleted (body identical) |
| routing_data | `test.instructions.md` + `testing_instructions.instructions.md` | Merged into `.agents/rules/testing.instructions.md`; both `.github` copies deleted |

Reference updates after the moves: `routing_data/AGENTS.md` (§4, §7), `routing_data/docs/ARCHITECTURE.md`, `routing_data/src/tsplib_parser/TODO.md` (PP7, references), `routing_data/tests/README.md` (`related_files`).

## 3. Testing rules design

Shared skeleton: philosophy (enforce behavior, never document a bug), value assertions over structure, fixture reuse, layout and filename-encoded levels, banned patterns, exception assertions, write-time checklist, four-step workflow, commands.

Repository deltas:

| Concern | routing_data | gpu_accelerated_clean |
| :-- | :-- | :-- |
| Layout | `tests/tsplib_parser/`, `tests/converter/`, `tests/e2e/` | Mirrored package folders for new tests; legacy flat files stay |
| Inputs | `requires_dataset` marker + conftest fixtures | `tests/fixtures/` only; no `datasets/` or repo `configs/` reads |
| Device rules | None | `gpu` marker, explicit skipif, CPU/GPU parity tests |
| Determinism | Dataset fixtures | Explicit seeds; no reliance on the GA global `np.random.seed` |
| Numeric rules | Parse-value assertions | `int32` matrices, TSPLIB `nint` and ATT `+1` rounding |

## 4. Test development batches

Five parallel subagents, disjoint file targets, each running its own tests and ruff:

| Batch | Files | New tests | Result |
| :-- | :-- | :-- | :-- |
| distances | `test_unit_distance_matrix.py`, `test_unit_pairwise_functions.py` | 151 | 151 passed |
| data_models | `test_unit_problem_invariants.py`, `test_unit_records.py` | 102 | 101 passed, 1 xfailed |
| utils | `test_unit_route_helpers.py`, `test_unit_phase_timer.py` | 55 | 54 passed, 1 xfailed |
| benchmarking | `test_unit_stats_utils.py`, `test_unit_statistics.py`, `test_unit_config_loader.py`, `test_unit_checkpoint_io.py` | 127 | 125 passed, 2 xfailed |
| protocols + algorithms | `test_unit_problem_context.py`, `test_unit_strategies.py` | 82 | 80 passed, 2 xfailed |

Total: 517 new tests (511 passing, 6 strict xfails). Full suite: `541 passed, 6 xfailed in 8.75s`, zero warnings, `ruff check tests` clean.

## 5. Decisions and rejected alternatives

| Decision | Chosen | Alternative rejected | Why |
| :-- | :-- | :-- | :-- |
| Instruction location | Single source in `.agents/rules` | Keep duplicated `.github/instructions` copies | The user asked for consolidation; the harness discovers `.agents/rules/*.instructions.md` (the files appear in the instruction list). Duplicates drift: the gpu markdown copies had already diverged |
| Frontmatter | Two commented key blocks in one YAML | Two separate files per host | One file cannot drift from itself; both hosts read the same description |
| Testing rules shape | One merged file per repo | Generic file + per-repo layout file | The philosophy section would be duplicated; one concern per file |
| Test layout | New tests in mirrored folders; legacy files untouched | Move the six existing flat files first | Moving 379-line and 267-line files churns imports without adding coverage; the user asked for development now |
| Stochastic assertions | Invariants (`sorted(child) == sorted(parent)`) with explicit seeds | Exact expected sequences | Exact sequences encode one RNG implementation, not the algorithm contract |
| Defect handling | `xfail(strict=True, reason="DEFECT: path:line ...")` | Skip with a bug comment; weaken the assertion | A strict xfail fails when the defect is fixed, forcing marker removal; weakening hides the defect |
| Speculative feature test | Deleted (`PhaseTimer` context manager) | Keep the xfail | `PhaseTimer` never promised `__enter__`/`__exit__`; an xfail for an invented requirement is noise, not a finding |
| Marker strategy | Filename encodes level; register only `gpu` and `slow` | Hand-written `unit`/`integration` markers | Duplicates information already in the filename; unregistered marks warn |

## 6. Defects found (strict xfails, not fixed in this change)

| ID | Evidence | Test |
| :-- | :-- | :-- |
| D1 | `src/data_models/problem.py:30` — frozen dataclass equality compares `ndarray` fields with `==`, raising `ValueError`; `hash()` raises `TypeError` | `tests/data_models/test_unit_problem_invariants.py::test_problems_with_equal_content_compare_equal` |
| D2 | `src/utils/route_helpers.py:145` — `resolve_rng` rejects `ProblemContext.rng` because it is a `RandomState` (`src/protocols/problem_context.py:187`), so the seeded context RNG is silently ignored | `tests/utils/test_unit_route_helpers.py::test_resolve_rng_uses_seeded_problem_context_rng` |
| D3 | `src/benchmarking/analysis/statistics.py:220` — zero-variance branch hardcodes `mean_difference=0.0` and `ci_95_difference=(0.0, 0.0)` | `tests/benchmarking/test_unit_statistics.py::test_paired_comparison_zero_variance_reports_true_mean_difference` |
| D4 | `src/benchmarking/runner/checkpoint_io.py:266` — `is_valid_checkpoint` ignores `raw_*` array lengths that `can_resume_checkpoint` checks | `tests/benchmarking/test_unit_checkpoint_io.py::test_validity_rejects_arrays_inconsistent_with_successful_runs` |
| D5 | `src/algorithms/strategies/mutation/swap_mutation.py:48` — depot-endpoint guard only applies when `len > 3` | `tests/algorithms/test_unit_strategies.py::test_mutation_preserves_three_node_depot_endpoints[swap]` |
| D6 | `src/algorithms/strategies/mutation/inversion_mutation.py:48` — same guard gap | `tests/algorithms/test_unit_strategies.py::test_mutation_preserves_three_node_depot_endpoints[inversion]` |

Documentation findings (verified behavior differs from the inline examples; no test asserts the example):

- `src/distances/pairwise.py:291` and `:502` — GEO example says `504`, actual `505`.
- `src/distances/pairwise.py:347` — ATT example says `6495`, actual `1495`.
- `src/distances/matrix.py:95` — GEO example returns `505` as `int32`, not `504.0`.
- `src/data_models/exceptions.py:88` — error message lists four valid edge types, `Problem` accepts `CEIL_2D` too.

## 7. Verification evidence

```text
$ .venv/bin/python -m pytest tests -q
541 passed, 6 xfailed in 8.75s

$ .venv/bin/python -m ruff check tests
All checks passed!

$ python3 <frontmatter scan>   # every .agents/rules/*.md
gpu: exploration, markdown, sequential-thinking, testing, update-plan -> parsed
routing_data: documentation, testing -> parsed
```

Chain of reasoning used during the work: six `sequentialthinking` calls (scope, consolidation, instruction shape branch, layout, batching, acceptance criteria), followed by the five test batches.

## 8. Residual risks

- If a client scans only `.github/instructions/**/*.instructions.md`, the consolidated rules no longer load there. The harness instruction list includes `.agents/rules/*.instructions.md`, so this workspace is covered; other clients need verification.
- Existing flat tests were not moved to the mirrored layout; the two layouts coexist until a dedicated refactor.
- Six strict xfails turn red as soon as the defects are fixed, by design.
- `test_early_stopping.py` still runs 80 generations for four algorithms without a `slow` marker.
- The `gpu` marker selects nothing today; parity tests will populate it when a device-backed run is scheduled.

## 9. Follow-ups

1. Fix D1–D6 and remove the strict xfail markers.
2. Move the six legacy flat test files into mirrored folders; update `tests/README`-equivalent documentation if one is added.
3. Add a `slow` marker to the validation tests and a `gpu` parity suite.
4. Correct the four distance/exception documentation examples.

## 10. Round 2 — rule corrections and responsibility split

### 10.1 Rule corrections

| Rule | Before | After |
| :-- | :-- | :-- |
| Intros | gpu rule claimed the suite was thin; routing_data rule described the merge history | Both describe the current organized suite |
| Frontmatter descriptions | Short | Descriptive one-liners naming layout, splitting, fixtures, parity, rounding |
| Config inputs | Ambiguous exception for config-contract tests | File loading uses `fixture_config_dir`; contract tests pass inline dicts to `validate_configs()`; malformed input writes JSON under `tmp_path`; file IO is never mocked |
| EXPLICIT input | Generic `ValueError` | Same error, message names `Problem.distances` and the loaders that populate it |
| Rounding | `nint` for EUC_2D/GEO, ATT `+1` | Seven-row table: EUC_2D/3D, MAN_2D/3D, MAX_2D/3D, CEIL_2D, GEO, ATT, XRAY1, with source anchors |
| File size | No rule | One responsibility per file, one source module or one strategy class; no line cap, split on the second concern |

### 10.2 EXPLICIT input decision

`compute_distance_matrix(coordinates, edge_type)` never receives a `Problem`, so it cannot return `Problem.distances`; a warning would still have to fail for lack of a return value. `Problem.needs_distance_computation` already returns `False` for EXPLICIT, so the guard only fires on misuse. A dispatcher (`resolve_distance_matrix(problem)`) remains an option if call sites multiply; it was not added.

### 10.3 Typing question (`Any` vs `float`)

Time values are `float` by construction (`time.perf_counter`, CuPy elapsed time). `calls` is the only integer, inside `dict[str, float | int]`. `Any` is unrelated to time: it types the decorator's `self_obj`, which `ParamSpec` cannot express. Per the user's decision, `phase_timer.py` is unchanged.

### 10.4 Test reorganization

Every oversized and legacy file was split or migrated. The tree now mirrors `src/` (40 python files under `tests/`, including `conftest.py`), with strategy tests under `tests/algorithms/strategies/<family>`. The GPU probe moved into `tests/conftest.py` and `pytest_collection_modifyitems` skips `gpu`-marked tests when no device is visible.

| Moved or split | Targets |
| :-- | :-- |
| `test_unit_strategies.py` (604 L) | 8 files under `tests/algorithms/strategies/` + shared conftest |
| `test_unit_problem_invariants.py` (676 L) + `test_unit_records.py` (512 L) | validation, metadata, records, exceptions |
| `test_unit_pairwise_functions.py` (451 L) | 6 family files including the function factory |
| `test_unit_statistics.py` (538 L), `test_unit_config_loader.py` (555 L) | statistics/comparison/aggregate; loader/validation |
| `test_unit_route_helpers.py` (309 L) | route helpers + RNG helpers |
| `test_unit_problem_context.py` (237 L) | CPU accessors + GPU guards |
| Legacy flat files | `tests/cli/` (3 files), `tests/benchmarking/test_integration_pipeline.py`, `tests/loaders/test_integration_loaders.py`, `tests/algorithms/test_integration_ga_composition.py`, `tests/algorithms/test_unit_strategy_contracts.py` |

Collected test count is unchanged at 547 (541 passed, 6 xfailed).

### 10.5 Verification (round 2)

```text
$ .venv/bin/python -m pytest tests -q
541 passed, 6 xfailed in 11.11s

$ .venv/bin/python -m ruff check tests
All checks passed!

$ .venv/bin/python -m pytest tests --collect-only -q
547 tests collected in 1.31s
```

### 10.6 Independent review — status after migration and remediation

An unanchored agent explored `src/` and `tests/`; the findings below are now reconciled against the post-migration code (Phases 7-8: RNG extracted to `src/utils/rng.py`, `local_search_stage` validated, config schema changed to `default_params` + `algorithm_configs`, presets removed).

| Severity | Finding | Status | Evidence of resolution |
| :-- | :-- | :-- | :-- |
| High | `src/benchmarking/database/*` has no behavioral coverage | Fixed | `tests/benchmarking/database/test_unit_db_operations.py` (117 cases: schema, insert, validation errors, summary round-trip) |
| High | Unknown `local_search_stage` silently disables local search | Fixed before remediation | Migration added `ValueError` at `genetic_algorithm.py:279-284`; pinned by `test_ga_rejects_unknown_local_search_stage` |
| High | Constructor mutates global NumPy RNG state | Fixed before remediation | RNG moved to `src/utils/rng.py`; pinned by `test_ga_init_does_not_mutate_global_numpy_rng_state` |
| High | Protocol test covers 13 families; `isinstance(Protocol)` is structure-only | Partially fixed | Duplicate route-helper coverage removed; per-class behavior tests exist under `tests/algorithms/strategies/**`. The `isinstance` sweep remains as an explicit contract smoke test |
| High | 2-opt asserted as `<=` cost; no GPU cost parity | Fixed | Strict `<` against the reversed burma14 route; `test_multi_move_two_opt_cpu_gpu_parity` asserts equal per-route costs on device |
| High | Private CuPy probe; 80-generation test unmarked | Fixed | Probe now `tests/gpu_probe.py` (real module, static-analysis friendly); conftest auto-skips `gpu` marks; validation tests marked `slow` |
| High | Production `configs/` read by two test files | Fixed | Both tests now take `fixture_config_dir`; `tests/fixtures/configs/algorithms.json` expanded to all six variants |
| Medium | `assert x is not None` on analysis outputs | Fixed | Replaced with counts, group membership, Friedmann bounds and speedup values |
| Medium | `hasattr(instance, "evolve")` and key-presence assertions | Fixed | Class-name contract against `cfg.metaheuristic`; stats contract (`best_fitness <= initial_fitness`, `len(convergence_history) == generations_completed + 1`, byte counters, `stop_reason` set) |
| Medium | Duplicated coverage of route batch helpers | Fixed | Duplicate removed from `test_unit_strategy_contracts.py`; canonical coverage in `tests/utils/` |
| Medium | Integration run unmarked; hardcoded `7542.0` | Fixed | `slow` markers added; `KNOWN_TSPLIB_OPTIMALS["berlin52"]` used |
| Missing | `n < 4`, `max_iterations <= 0`, `mutation_rate=0.0`, `max_generations=0`, `patience=0`, empty customers | Mostly fixed | New boundary files for two-opt, mutation rate, GA limits. Empty customers exposed a defect (D8) |

**Defects found during remediation (strict xfails):**

| ID | Defect | Test |
| :-- | :-- | :-- |
| D7 | `GeneticAlgorithmFullGPUMonolithic.evolve` returns `best_cost_history` instead of the protocol key `convergence_history` (`full_gpu_monolithic.py:250`) | `test_gpu_variants_and_modular_vs_monolithic[FullGPU]`, `[FullGPUMonolithic]` |
| D8 | `evolve` accepts empty `customers`, returns a `(0,)` tour and `generations_completed=1` instead of rejecting the instance (`genetic_algorithm.py:452`) | `test_empty_customers_are_rejected` |

**Observations, not filed as defects** (no documented promise contradicts them): `validate_benchmark_data` NaN-checks 13 aggregate fields but not `optimal_cost`; `insert_benchmark_run` stores `"...Z"` timestamps shifted by the DuckDB session timezone.

### 10.7 Round 3 — probe fix, typing, and verification

- `gpu_probe` was only a `sys.modules` alias published by `conftest.py`, so mypy reported `Cannot find implementation or library stub for module named "gpu_probe"`. Replaced by a real `tests/gpu_probe.py`; `pythonpath` and `mypy_path` now include `tests`, and `explicit_package_bases = true` keeps the two `conftest.py` modules from colliding.
- Typing clean-up in tests: `ParameterSet` annotations, `to_cpu_batch` dtype note, named function instead of an untyped lambda, `setattr` for the frozen-dataclass check.
- The strategies `conftest.py` helper moved to `tests/strategy_fixtures.py` (a `conftest` cannot be imported by sibling test modules reliably).

```text
$ .venv/bin/python -m pytest tests -q
703 passed, 9 xfailed in 15.00s

$ .venv/bin/python -m mypy src tests
Success: no issues found in 113 source files

$ .venv/bin/python -m ruff check tests src
All checks passed!
```

### 10.8 Round 4 — Core GPU capability, archive isolation, and config contracts (Fase 9)

- **GPU capability in core**: Implemented lazy capability API in [backend.py](../../src/protocols/backend.py) (`gpu_available()`, `require_gpu()`, `cupy_backend()`, `clear_gpu_cache()`) with `GPUUnavailableError` in [exceptions.py](../../src/data_models/exceptions.py). Unit tests added in [test_unit_gpu_capability.py](../../tests/protocols/test_unit_gpu_capability.py) covering missing package, missing device, driver error, and cache reuse.
- **Retired `tests/gpu_probe.py` and strict sys.path**: Tests and mypy path now restricted to `["src"]` in [pyproject.toml](../../pyproject.toml). `tests/gpu_probe.py` deleted; all tests import capability from `protocols.backend`.
- **`make_population` fixture**: Migrated permutation builder from `tests/strategy_fixtures.py` to [conftest.py](../../tests/conftest.py) as an injected fixture, updated all 7 strategy test files, and deleted `tests/strategy_fixtures.py`.
- **Defect D8 resolved**: `GeneticAlgorithm.evolve` in [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py) validates `len(customers) > 0` with `ValueError("customers sequence cannot be empty")`; strict xfail removed from `test_empty_customers_are_rejected` (now passing).
- **Archive isolation & D7 resolved**: `full_gpu_monolithic.py` and `kernels/` moved to `src/algorithms/archive/` ([README.md](../../src/algorithms/archive/README.md)); 5 legacy non-importable files deleted; monolithic re-exports removed from `algorithms.__all__`. Key `"best_cost_history"` corrected to `"convergence_history"` (resolving D7).
- **Archive excluded from test suite**: Strict repository rule applied (no tests depend on archive). Removed the two monolithic `xfail(strict=True)` from [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py) and excluded archive variants in [test_early_stopping.py](../../tests/validation/test_early_stopping.py).
- **Smoke script & config contract**: Created pure-Python [check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py) (zero argparse, runs standalone on CUDA) and [test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py) ensuring structural parity between `configs/algorithms.json` and the test fixture.

```text
$ .venv/bin/pytest -q
711 passed, 6 xfailed in 16.12s

$ .venv/bin/mypy src tests
Success: no issues found in 111 source files

$ .venv/bin/ruff check src tests benchmark_scripts
All checks passed!
```

**Documentation caveat:** findings recorded before the migration (for example the `local_search_stage` and global-RNG rows) described the old code. Sections 10.6-10.8 are the current record; earlier documentation written against the pre-migration layout may still be stale and should be audited separately.
