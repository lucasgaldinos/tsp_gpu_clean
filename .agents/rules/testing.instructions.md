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
- **Truth-grounded oracles over "written-to-pass" snapshots:** Never write characterization tests that merely snapshot current (potentially buggy) output. Assert domain invariants and independent mathematical oracles (e.g., recomputing route cost from coordinates against declared optimal cost, verifying a mutation of indexing/rounding logic causes the test to fail).
- Never `@pytest.mark.skip` with a comment explaining a known bug. Fix it, or `xfail` it with a live register ID (`@pytest.mark.xfail(strict=True, reason="BUG-n")`).
- Do not assert that a bug exists.
- Annotate every test function and fixture with Python 3.12+ type hints (`-> None`, `list[int]`, `X | None`), and avoid `monkeypatch` mutations of private module dictionaries when testing public contracts.

## 2. Hermetic inputs

- Fixtures live in `tests/fixtures/` and are served by [tests/conftest.py](../../tests/conftest.py): `fixture_tsp_dir`, `fixture_config_dir`, `fixture_duckdb_path`, `burma14_problem`, `berlin52_problem`, `db_loader`.
- File-loading tests use `fixture_config_dir`. The repository `configs/` directory is production state: a test must never read or write it, because a config edit would silently break tests that have nothing to do with the change.
- Contract tests pass inline dictionaries to `validate_configs()` (`src/benchmarking/runner/config_loader.py`); malformed-input tests write JSON under `tmp_path`.
- Do not mock file IO: the behavior under test is real parsing, and a mock turns the test into a test of the mock.
- Write only under `tmp_path` or `tmp_path_factory`. No test writes to `results*/`, `checkpoints/`, `tables/` or the repository root.

## 3. Layout, naming and file responsibility

- Mirror the source package: `tests/distances/`, `tests/data_models/`, `tests/utils/`, `tests/benchmarking/`, `tests/protocols/`, `tests/algorithms/`, `tests/loaders/`, `tests/cli/`.
- Strategy tests mirror the strategy families: `tests/algorithms/strategies/<initialization|selection|crossover|mutation|local_search|evaluation>/test_unit_<class>.py`.
- One responsibility per file: one source module, or one strategy class. Keep test files maintainable (~250–400 lines target; never build 1,000+ line monolith test files) — split when a file grows past ~400 lines or starts covering a second concern, class, or failure mode.
- Name files `test_<level>_<module>[_<concern>].py`, where `<level>` is `unit`, `integration`, `e2e` or `performance` (keep the `test_e2e_` prefix even inside `tests/e2e/`). Apply orthogonal pytest markers as per-test decorators, never module-wide globals.
- `tests/validation/` holds end-to-end algorithm validations; keep them marked `slow` (`> 1.0s`).
- Do not test `src/algorithms/archive/` — excluded from lint and typing on purpose.

## 4. Determinism

- Pass an explicit `seed=` to every strategy and GA under test. `GeneticAlgorithm.__init__` calls the global `np.random.seed`; relying on it couples test order.
- Compare costs and statistics with `pytest.approx`, never exact float equality. Distance matrices are integer (`int32`) and may be asserted exactly.
- No wall-clock assertions; `PhaseTimer` tests assert bookkeeping, not duration.

## 5. GPU discipline

- Import GPU capability from production (`protocols.backend.gpu_available` / `require_gpu`); never inject dynamic `sys.modules` aliases (such as `gpu_probe`) in `tests/conftest.py`, which break static `mypy` resolution and leak test helpers into core APIs.
- Mark device tests `@pytest.mark.gpu`; `tests/conftest.py::pytest_collection_modifyitems` skips them automatically when the probe reports no device, so a second `skipif(not GPU_AVAILABLE, ...)` is redundant.
- Every GPU strategy gets a CPU/GPU parity test on a small fixture (burma14 or berlin52) asserting route validity and cost equality within tolerance.
- Never assert device memory, stream identity or kernel names.

## 6. Numeric contracts

- `compute_distance_matrix` returns `int32`, has a zero diagonal, and rejects coordinate-based computation for `EXPLICIT`. That input carries its matrix in `Problem.distances`, produced by the loaders; the error message names that field so the caller is sent to the right API rather than to a warning that cannot return a value.
- Rounding is part of the contract and differs per edge type. Pin one fractional coordinate pair per row:

  | Edge type | Rule | Source |
  | :-- | :-- | :-- |
  | `EUC_2D`, `EUC_3D` | `nint` = `int(x + 0.5)` | `matrix.py` / `cost.py` |
  | `MAN_2D`, `MAN_3D` | `nint` per coordinate sum | `matrix.py` / `cost.py` |
  | `MAX_2D`, `MAX_3D` | `nint` per coordinate, then `max` | `matrix.py` / `cost.py` |
  | `CEIL_2D` | `ceil` with no `+1` | `matrix.py` / `cost.py` |
  | `GEO` | C-style `int(coord)` degree truncation (never `nint`), then `int(RRR * acos(...) + 1.0)` (`burma14` `d(1,3) == 510`) | `matrix.py` / `cost.py` |
  | `ATT` | `t = nint(r)`; `t + 1` when `t < r` | `matrix.py` / `cost.py` |
  | `XRAY1` | `nint(100 * cost)` | `matrix.py` / `cost.py` |

  The vectorized `matrix.py` and CUDA `pairwise_device.cuh` reproduce the same rules (`pairwise.py` is a 0-math re-export layer); pin rounding parity across CPU matrix, scalar re-export, and GPU on-the-fly kernels.

## 7. Error & route-boundary contracts

- `Problem.__post_init__` and parsers raise specific exceptions for broken invariants; forbid bare `pytest.raises(Exception)` and always assert exact exception type plus `match=...`.
- Explicitly-named `ProblemContext.get_gpu_*()` accessors call `require_gpu()` directly (usable regardless of `context.xp`) and raise `GPUUnavailableError` only when no CUDA device is available.
- Route/tour operator tests (mutation, crossover, 2-opt, candidate lists) must cover both minimal `n == 3` closed tours (preserving endpoints) and non-contiguous CVRP sub-routes where `route.max() >= len(route)` (e.g., `[0, 5, 18, 0]`).

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
rtk .venv/bin/python -m pytest tests/distances -q
rtk .venv/bin/python -m pytest tests -m "not slow and not gpu" -q
rtk .venv/bin/python -m pytest tests -m gpu -q
rtk .venv/bin/python -m ruff check tests
rtk .venv/bin/python -m mypy tests
```
