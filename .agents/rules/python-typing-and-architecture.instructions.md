---
# --- Antigravity rule keys ---
trigger: glob
globs: "**/*.py, **/*.cu, **/*.cuh, !**/.agents/**, !**/.github/**, !**/archive/**"
description: "Use when writing, typing, refactoring, or reviewing Python 3.12+ and CUDA/CuPy code: import root, PEP 695/698 typing, TypeGuard and descriptor patterns, NumPy/CuPy boundary protocols, GPU caching, TSPLIB95/CVRP invariants, and DuckDB safety." # shared with VS Code
# --- VS Code instruction keys ---
name: "Python 3.12, Typing & GPU/Routing Architecture"
applyTo: "**/*.py, **/*.cu, **/*.cuh, !**/.agents/**, !**/.github/**, !**/archive/**"
---

# Python 3.12+, Typing & GPU/Routing Architecture Standards

## 1. Architecture Precedes Typing

- Establish functional behavior, database schemas, and input/output contracts **before** pursuing strict static type gymnastics. Never invest passes typing broken or unverified pipelines.
- Never silently strip forward-compatibility stubs or planned enum/section branches without user confirmation via `ask-clarification-questions`.
- Comments and docstrings must explain **domain/business rules and algorithmic provenance** (WHY — e.g., TSPLIB95 rounding rules, Pascal CC 6.1 SM occupancy, Rocki & Suda 2012), never restate Python/CUDA syntax.

## 2. Import Root & Configuration Authority

- `src/` is the import root (`pythonpath = ["src"]`, `mypy_path = ["src"]`), **not** an importable package.
  - **Banned:** `from src.*` and cross-package relative escaping (`from ..data_models import ...`).
  - **Allowed:** Absolute package imports (`from protocols.backend import require_gpu`) and intra-package relative imports (`from .backend import ...`).
- [pyproject.toml](../../pyproject.toml) is the single configuration source of truth for `mypy`, `ruff`, and `pytest`. Never reintroduce `mypy.ini`, `ruff.toml`, or `setup.cfg`.
- Run linters and type checkers via `rtk .venv/bin/python -m ruff check <files>` and `rtk .venv/bin/python -m mypy <files>` (never bare system `pytest`/`mypy`).

## 3. Modern Python 3.12+ Static Typing

Every public and private function carries explicit parameter and return annotations (`-> None` included):

- **PEP 585 & PEP 604:** Use built-in generics (`list[T]`, `dict[K, V]`, `tuple[T, ...]`, `set[T]`) and `X | Y | None` unions. Imports of `List`, `Dict`, `Tuple`, `Set`, `Optional`, and `Union` from `typing` are banned.
- **PEP 695 (`requires-python >= 3.12`):** Use `type Alias = ...` and `class Box[T]:` / `def fn[T](x: T) -> T:` instead of legacy `TypeVar` and `TypeAlias`. Ensure `pyproject.toml` declares `requires-python = ">=3.12"` whenever PEP 695 syntax is used.
- **PEP 698:** Decorate every implementation of a `Protocol` or `ABC` method with `@override` (`from typing import override`).
- **`object` Narrowing via `TypeGuard`:** At untrusted JSON/dict boundaries, `isinstance(x, dict)` on `object` or `Any` narrows to `dict[Unknown, Unknown]` under Pyright strict mode, and `cast("dict[str, Any]", x)` triggers `reportRedundantCast`. Write a dedicated `TypeGuard` helper (`def _is_str_dict(val: object) -> TypeGuard[dict[str, Any]]: return isinstance(val, dict)`) instead.
- **Non-Data Descriptors on Class Bodies:** Pyright synthesizes typed attribute writes only for data descriptors (defining `__set__`). When a class body uses non-data descriptors (`__get__` only) populated into `instance.__dict__` at parse time, type the class attribute with `cast("TargetType", SomeField(...))` (and use `cast("Callable[[str], T]", lambda x: x)` for polymorphic identity defaults).
- **No Unjustified Suppressions:** `# type: ignore[code]` and `# noqa: CODE` require a specific error code and concrete inline reason; never use bare suppressions as a shortcut.

## 4. Dual-Backend (NumPy / CuPy) & GPU Capability Contracts

- **Production GPU Capability:** Hardware availability lives in `src/protocols/backend.py` (`gpu_available()`, `require_gpu()`, raising `GPUUnavailableError`), **never** in `tests/` or dynamic `sys.modules` shims.
- **Boundary Typing (`CupyArray`):** Do **not** install `cupy-stubs` (its incomplete PEP 561 `__init__.pyi` breaks `zeros`, `sum`, `min`, `max`, `int32`, `newaxis`). Keep `cupy.*` under `[[tool.mypy.overrides]]`, parameterize `BackendModule[ArrayT]`, keep `ProblemContext` non-generic (making `ProblemContext[ArrayT]` generic causes wide call-site error cascades), and type explicit GPU array boundaries with `type CupyArray = Any` (never import private `cupy._core.core.ndarray`).
- **Centralized `ProblemContext` GPU Caching (`DP10-17`, `BUG-14`):** Explicitly-named `ProblemContext.get_gpu_*()` accessors call `cp = require_gpu()` and `cp.asarray(...)` directly — never guard them with `if self.xp == np: raise ValueError(...)`, which breaks hybrid CPU+GPU pipelines. Strategies must consume shared GPU distance/coordinate/candidate operands from `ProblemContext` rather than caching `_cached_distances_gpu` locally per strategy instance.
- **Stream Synchronization Discipline (`BUG-15`):** Avoid redundant per-call `cp.cuda.Stream.null.synchronize()` inside hot loops when `input_on_gpu=True`; synchronize only at host readback (`.get()` / `cp.asnumpy()`) or timing boundaries.

## 5. Routing Domain, DuckDB, & Decoupled Benchmark Invariants

- **TSPLIB95 Specification Authority:**
  - TSPLIB node and depot IDs in problem/tour files are **1-based**; `.sol` files use customer ordinals — verify tour/route indexing against coordinate cost recomputation.
  - `GEO` coordinate conversion truncates degrees toward zero via `int(coord_val)` (Reinelt 1991), **never** nearest-integer `_nint(coord_val)` (`burma14` `d(1,3) == 510`, not `560`).
  - Reject malformed sections loudly (`ParseError`); never swallow section errors into empty arrays or launder `ValidationError` into `ParseError`.
- **CVRP Sub-Route Indexing Bounds (`BUG-19`):** A CVRP vehicle sub-route (e.g., `[0, 5, 18, 0]`) has `route.max() >= len(route)`. Never allocate inverse-position lookup tables as `np.empty(len(route))` or silently disable `candidate_k` when `route.max() >= route.size`; allocate `np.full(max(len(route), int(route.max()) + 1), -1, dtype=np.int32)` and skip candidates with position `< 0`.
- **No Hardcoded Corpus Shapes or Run Counts (`C1`–`C3`, `BUG-17`, `BUG-18`):** Library, CLI, and config defaults must never hardcode `datasets_raw/**`, sibling folder layouts (`parent / "tour"`), `expected_reps = 30`, or hardcoded `construction_heuristic` strings; serialize full strategy hyperparameters into checkpoints.
- **DuckDB Safety & Decoupled Analysis:**
  - DuckDB lacks SQL `SAVEPOINT`; isolate batch ingestion via Python pre-flight validation + per-item transactions + `ON CONFLICT` upserts.
  - Guard nullable `cursor.fetchone()` calls with a typed helper (`_one(row, error_msg)`), mirror SQL column names as constants in `schema.py`, and type insert payloads with `TypedDict`.
  - Benchmark analysis/reporting is a **pure consumer** of checkpoint/DB files and must never depend on runner execution size thresholds.
