---
# --- Antigravity rule keys ---
trigger: glob
globs: "tests/**/*.py"
description: "Use when writing, refactoring, reviewing, or splitting tests in gpu_accelerated_clean: mirrored package layout, one responsibility per file, hermetic fixtures, deterministic seeds, CPU/GPU parity, numeric rounding contracts, and defect-tracking xfails." # shared with VS Code
# --- VS Code instruction keys ---
name: "Testing Standards (gpu_accelerated_clean)"
applyTo: "tests/**/*.py"
---

# Testing standards — gpu_accelerated_clean

The suite covers the numeric distance core, data-model invariants, statistics, runner configuration, protocol guards and strategy contracts, organized in folders that mirror `src/`. New tests go beside the module they test, one responsibility per file.

> [!IMPORTANT]
> GPU availability is not a license to skip: a test that needs a device must be marked `gpu` and skipped explicitly, and every GPU behavior needs a CPU-side twin that runs everywhere.

## 1. Philosophy: enforce behavior, do not document bugs

- A new test must fail against the pre-fix code and pass after it.
- Never `@pytest.mark.skip` with a comment explaining a known bug. Fix it, or `xfail` it with a live register ID.
- Do not assert that a bug exists.

## 2. Hermetic inputs

- Fixtures live in `tests/fixtures/` and are served by [tests/conftest.py](../../tests/conftest.py): `fixture_tsp_dir`, `fixture_config_dir`, `fixture_duckdb_path`, `burma14_problem`, `berlin52_problem`, `db_loader`.
- File-loading tests use `fixture_config_dir`. The repository `configs/` directory is production state: a test must never read or write it, because a config edit would silently break tests that have nothing to do with the change.
- Contract tests pass inline dictionaries to `validate_configs()` (`src/benchmarking/runner/config_loader.py`); malformed-input tests write JSON under `tmp_path`.
- Do not mock file IO: the behavior under test is real parsing, and a mock turns the test into a test of the mock.
- Write only under `tmp_path` or `tmp_path_factory`. No test writes to `results*/`, `checkpoints/`, `tables/` or the repository root.

## 3. Layout, naming and file responsibility

- Mirror the source package: `tests/distances/`, `tests/data_models/`, `tests/utils/`, `tests/benchmarking/`, `tests/protocols/`, `tests/algorithms/`, `tests/loaders/`, `tests/cli/`.
- Strategy tests mirror the strategy families: `tests/algorithms/strategies/<initialization|selection|crossover|mutation|evaluation>/test_unit_<class>.py`.
- One responsibility per file: one source module, or one strategy class. There is no line cap; split when a file starts covering a second concern, a second class, or a second failure mode.
- Name files `test_<level>_<module>[_<concern>].py`, where `<level>` is `unit`, `integration`, `e2e` or `performance`.
- `tests/validation/` holds end-to-end algorithm validations; keep them marked `slow`.
- Do not test `src/algorithms/archive/` — excluded from lint and typing on purpose.

## 4. Determinism

- Pass an explicit `seed=` to every strategy and GA under test. `GeneticAlgorithm.__init__` calls the global `np.random.seed`; relying on it couples test order.
- Compare costs and statistics with `pytest.approx`, never exact float equality. Distance matrices are integer (`int32`) and may be asserted exactly.
- No wall-clock assertions; `PhaseTimer` tests assert bookkeeping, not duration.

## 5. GPU discipline

- Reuse the availability probe defined in `tests/conftest.py` (`GPU_AVAILABLE` / `cp`); do not duplicate the probe.
- Mark device tests `@pytest.mark.gpu`; `tests/conftest.py::pytest_collection_modifyitems` skips them automatically when the probe reports no device, so a second `skipif(not GPU_AVAILABLE, ...)` is redundant.
- Every GPU strategy gets a CPU/GPU parity test on a small fixture (burma14 or berlin52) asserting route validity and cost equality within tolerance.
- Never assert device memory, stream identity or kernel names.

## 6. Numeric contracts

- `compute_distance_matrix` returns `int32`, has a zero diagonal, and rejects coordinate-based computation for `EXPLICIT`. That input carries its matrix in `Problem.distances`, produced by the loaders; the error message names that field so the caller is sent to the right API rather than to a warning that cannot return a value.
- Rounding is part of the contract and differs per edge type. Pin one fractional coordinate pair per row:

  | Edge type | Rule | Source |
  | :-- | :-- | :-- |
  | `EUC_2D`, `EUC_3D` | `nint` = `int(x + 0.5)` | `pairwise.py:50` |
  | `MAN_2D`, `MAN_3D` | `nint` per coordinate sum | `pairwise.py:143,170` |
  | `MAX_2D`, `MAX_3D` | `nint` per coordinate, then `max` | `pairwise.py:201,226` |
  | `CEIL_2D` | `ceil` with no `+1` | `pairwise.py:399` |
  | `GEO` | `int(d + 1.0)` truncation after the haversine `d` | `pairwise.py:314` |
  | `ATT` | `t = nint(r)`; `t + 1` when `t < r` | `pairwise.py:359` |
  | `XRAY1` | `nint(100 * cost)` | `pairwise.py:452` |

  The vectorized `matrix.py` reproduces the same rules; a rounding test belongs beside the pairwise function and, when the matrix path diverges, beside the matrix too.

## 7. Error contracts

- `Problem.__post_init__` raises for its invariants; assert type and message with `pytest.raises(..., match=...)`.
- GPU accessors on a numpy backend raise `ValueError`; test them without a device.

## 8. Write-time checklist

- [ ] Test all elements, not a sample.
- [ ] Validate values, not only structure.
- [ ] Cover boundaries: minimum, maximum, empty, single element.
- [ ] Cover invalid input and error paths.
- [ ] Docstring states WHAT is tested, WHY it matters, EXPECTED result.
- [ ] Assertions validate the property, with informative messages.
- [ ] Multiple cases via `parametrize` (or Hypothesis for invariants).
- [ ] Fixtures reused; no duplicate setup.

## 9. Workflow

1. THINK: identify testable functions; list happy, edge and error cases. Use `sequentialthinking` when the module has many branches.
2. CHECK: read this file and `tests/conftest.py`; find an existing fixture before writing setup.
3. CODE: write the test with the AAA structure (arrange, act, assert).
4. CRITIC: re-read the checklist; fix any failing box before running the suite.

## 10. Commands

```bash
.venv/bin/python -m pytest tests/distances -q
.venv/bin/python -m pytest tests -m "not slow and not gpu" -q
.venv/bin/python -m pytest tests -m gpu -q
.venv/bin/python -m ruff check tests
```
