---
title: "Discussion on joining `routing_data` and `gpu_accelerated_clean`"
description: >
  Question-by-question record of the review that maps the duplication between
  `routing_data` and `gpu_accelerated_clean`, lists the five-axis and type-safety
  findings for both source trees, and inventories stale documentation, together with
  the decisions requested from the project owner.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Added YAML frontmatter and one heading per numbered item, so every finding has
          a stable anchor that mirrors the numbering used in
          `discussion-on-routing_data-gpu_accelerated_clean-answers.md`.
related_files:
  - "[discussion-on-routing_data-gpu_accelerated_clean-answers.md](./discussion-on-routing_data-gpu_accelerated_clean-answers.md)"
  - "[analysis-gpu-batch.md](./analysis-gpu-batch.md)"
tags:
  - review/architecture
  - review/quality
  - analysis/duplication
  - analysis/type-safety
  - analysis/performance
  - analysis/documentation
  - notes/decisions
  - notes/discussion
  - notes/cross-repository
  - guide/migration
  - guide/packaging
---

# Discussion on joining `routing_data` and `gpu_accelerated_clean`

## 1. Duplication and Ovelap map

### 1.1 Problem file parsing

> **Problem file parsing** | `src/loaders/tsplib_loader.py:94` (`parse_tsp_file_with_tsplib95`, wraps third-party `tsplib95.load_problem`) | `src/tsplib_parser/parser.py:76` (`FormatParser.parse_file`) + `src/tsplib_parser/cordeau/` | `routing_data` is a full self-contained two-pass parser (`StandardProblem`, `ExtendedProblem`, Cordeau/CVRPLIB preprocessor, `validation.py`) with **zero `tsplib95` dependency**. `gpu_accelerated_clean` still depends on `tsplib95>=0.6.1` (`pyproject.toml:29`).

I said in a past iteration that tsplib was temporary. The objective is to use `routing_data`. Please make the "substitution". I tested `routing_data` and it currently just cannot properly parse Solomon and Homberger problems. Though it's still in development, the current implementation already serves its purpose for `TSP` and some `CVRP` problems.

Knowing that, what do I need to do in `routing_data` before using it as a parser and db generator? Should I make a release? How to transform `routing_data/src` code into a package (I should not import the whole repo and all the docs and things, just the package, right?)

### 1.2 DuckDB v2 schema and ETL

> **DuckDB v2 schema & ETL** | `src/loaders/tsplib_loader.py:418-660` (`ingest_tsplib_directory_to_duckdb`) + `src/cli/tsplib.py:60` (`tsp-ingest`) | `src/converter/database/operations.py:53-245` (`DatabaseManager`) + `src/converter/api.py:250` (`to_database`, `process_directory`) | **Exact duplicate DDL** for `problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `file_tracking`, `solutions`. In fact, `gpu_accelerated_clean/datasets/tsp.duckdb` on disk today is **byte-identical in table counts** to `routing_data/datasets/db/routing.duckdb` (9 tables including `hcp_problems` and `sop_problems`, 113 TSP problems, 17 matrices). `tsplib_loader.py` is explicitly marked as a temporary placeholder (`tsplib_loader.py:6-8`).

Yes, those are the same databases. I copied it, so I could test `gpu_accelerated_clean`. `tsplib_loader.py` is in fact temporary. I would prefer to use the more robust `routing_data` to parse problems, be them in a `duckdb` database or transformed into a tsplib problem and consumed directly. the tsplib_loader should use the methods on routing_data instead.

### 1.3 Pairwise TSPLIB95 distances

> **Pairwise TSPLIB95 distances** | `src/distances/pairwise.py:29-460` (`_nint`, `EUC_2D/3D`, `CEIL_2D`, `MAN_2D/3D`, `MAX_2D/3D`, `GEO`, `ATT`, `XRAY1/2`) | `src/converter/utils/cost.py:29-145` (`_nint`, `_euclidean_2d/3d`, `_ceiling_2d`, `_manhattan_2d/3d`, `_maximum_2d/3d`, `_geographical`, `_att`, `_xray`) | **100% duplicated math.** `cost.py:12` explicitly states it was copied to match `pairwise.py`. Worse, `gpu_accelerated_clean/src/distances/matrix.py:157-206` re-implements vectorized versions of the same rounding rules a third time.

So, should we also join them all? And another thing: I know they are repeated, but who should handle this responsibility? `gpu_accelerated_clean` or `routing_data`? Because when parsing, I may need that the matrices and calculations generate a `numpy` or `cupy` matrix instead (which is gpu's repo job), but for the db generation, I don't really know what should be done. Maybe a converter for `numpy`/`cupy` arrays?

Need to better discuss this topic before I can take a decision.

### 1.4 Database reader (`Problem` loader)

> **Database reader (`Problem` loader)** | `src/loaders/database_loader.py:54` (`DatabaseLoader`) + `src/loaders/problem_type_handlers.py:46-236` (`TspHandler`, `AtspHandler`, `CvrpHandler`) | `src/converter/database/operations.py:625` (`query_problems`, `get_problem_by_id`) | `DatabaseLoader` in `gpu_accelerated_clean` already queries `routing_data`'s v2 DuckDB schema (`problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `file_tracking`), but lacks `HCP`/`SOP` handlers and solution loading (`solutions` table).

So, it can query all same as `routing_data` also can query (i.e., repeating DDL, functions, etc.), or will I need some custom handlers and DDL fo `gpu_accelerated_clean`? I would like more and better separation of concerns.

Need more info before taking a decision on this.

The main problem here is defining where one repo's responsibility ends and where the other's begin.

The executable anchor result is expected. As I said, they are the same database I've copied into this repo.

## 2. Five-Axis (+ Type Safety) Code Review: `gpu_accelerated_clean/src/`

### 2.1 Critical / High (Blockers & Correctness)

#### 2.1.1 Broken seed determinism

>**Critical (Correctness — Broken Seed Determinism):** `src/utils/rng.py:57` vs `src/protocols/problem_context.py:166-169`
>
>- `ProblemContext.__init__` sets `self.rng = xp.random.RandomState(seed)`.
>- `resolve_rng(context)` checks `isinstance(context.rng, np.random.Generator)`, which is **always `False`** for `RandomState`, falling back to `make_rng(None)` → fresh unseeded entropy on every call when no `fallback` seed is passed.
>- **Executable observation that fails today:**
>
> ```bash
> rtk /home/lucas_galdino/chimera/gpu_accelerated_clean/.venv/bin/python - <<'EOF'
> import numpy as np
> from data_models.problem import Problem
> from protocols.problem_context import ProblemContext
> from utils.rng import resolve_rng
> p = Problem(name="t", dimension=3, edge_type="EUC_2D", coordinates=np.zeros((3, 2)))
> ctx1 = ProblemContext(p, xp=np, seed=42)
> ctx2 = ProblemContext(p, xp=np, seed=42)
> assert resolve_rng(ctx1).integers(0, 10**9) == resolve_rng(ctx2).integers(0, 10**9)
> EOF
>  ```
>
> Fails with `AssertionError`.
>
>- **Remedy:** Standardize `ProblemContext.rng` on `xp.random.default_rng(seed)` (`Generator`) or accept `(np.random.Generator, np.random.RandomState)` at the boundary.

Nice observation. That is indeed a problem. I want to know more on the proposed solution before giving the decision.

#### 2.1.2 Distance matrix full broadcast and formula duplication

> **High (Correctness & Performance — Distance Matrix Full Broadcast + Formula Duplication):** `src/distances/matrix.py:126-206`
>
> - Docstring (`matrix.py:38`) claims upper-triangle computation (`n*(n-1)/2`), but line 127 allocates full `diff = coords[:, None, :] - coords[None, :, :]` (`(n, n, d)` float64 = ~3.6 GB at `n=15,000`) and duplicates every rounding formula from `src/distances/pairwise.py`.
> - **Remedy:** Extract shared vector/scalar kernel rules or compute upper-triangle chunks when `n` exceeds a threshold.

More on this problem before I can decide. Please better explain.

#### 2.1.3 Device cache keyed only by `problem.name`

> **High (Correctness — Device Cache Keyed Only by `problem.name`):** `src/algorithms/strategies/evaluation/cost_evaluation.py:123-127` and `src/algorithms/strategies/local_search/two_opt.py:265-273,520-528`
>
> - Caches GPU distance matrix by `problem.name` string only. Two synthetic problems in tests or benchmarks sharing a name (e.g. `"test"`) with different dimensions or coordinates silently reuse the stale GPU matrix.
> - **Remedy:** Key on `(problem.name, problem.dimension, problem.file_checksum)` or delegate GPU matrix caching solely to `ProblemContext.get_distances()`.

More on this problem's proposed solution, pros and cons. Caching shall be maitained.

#### 2.1.4 Per-call `Stream.null.synchronize()`

> **High (Performance — Per-Call `Stream.null.synchronize()`):** `src/algorithms/strategies/local_search/two_opt.py:327,580`
>
> - Full device synchronization on every 2-opt call stalls the CUDA stream pipeline.
> - **Remedy:** Synchronize only at host readback boundaries.

Also need to know more on this problem: How does it affect performance? What happens when I have the synchroinization done this way? It does seem to be a performance bottleneck. How much time does it take from a run performance? You can test in a small problem.

### 2.2 Medium / Required (Architecture & Type Contracts)

#### 2.2.1 Duplicate TSPLIB ETL in solver repo

> **Medium (Architecture — Duplicate TSPLIB ETL in Solver Repo):** `src/loaders/tsplib_loader.py:1-705` and `src/cli/tsplib.py:1-155`
>
> - `tsplib_loader.py` mixes `ProblemRowRecord`/`NodeRowRecord` (v1 leftovers with `nodes` table docstring at `:6,57`) with v2 DuckDB DDL (`:452-533`), silently drops ATSP display coords (`:585`), coerces missing CVRP capacity to `0` (`:627`), and runs outside a transaction (`:575-660`).
> - **Remedy:** Replace with `routing-data` dependency (`tsplib_parser` + `converter.database.DatabaseManager`).

Agree with this decision.

#### 2.2.2 Swallowed `FileNotFoundError`

> **Medium (Correctness — Swallowed `FileNotFoundError`):** `src/loaders/database_loader.py:105-111`
>
> - Raises `FileNotFoundError`, immediately caught by `except Exception as e: raise DatabaseConnectionError(self.db_path, e)` — obscuring missing file vs corrupt DB.

Well, and what are the options for fixing and how does this affect the current code?

#### 2.2.3 Type and contract safety

> **Medium (Type & Contract Safety):**
>
> - `src/distances/pairwise.py:460`: `Callable[[Any, Any], int]` → replace with a `PairwiseDistanceFn` `Protocol`.
> - `src/loaders/problem_type_handlers.py:30,41`: `dict[str, Any]` metadata bag across handler/loader boundary → replace with `ProblemMetadataRow` `TypedDict`.
> - `src/protocols/backend.py:120-160,225-228`: `*args: Any, **kwargs: Any` protocol signatures and `# type: ignore` monkey-patching of `cache_clear`.

Agree with proposed solutions.

---

## 3. Five-Axis (+ Type Safety) Code Review: `routing_data/src/`

Cross-referenced against `routing_data/TODO.md`:

### 3.1 Critical / High (Blockers & Correctness)

#### 3.1.1 Silent no-op on re-ingestion (`B1`)

> **Critical (`B1` — Silent No-Op on Re-Ingestion):** `src/converter/database/operations.py:409`
>
> - `insert_problem` runs a plain `INSERT INTO problems` without `ON CONFLICT`, while `problems` has `UNIQUE (name, type)` (line 109). Re-running without `--force` fails inside `insert_problem`, logs a warning, returns `None`, and exits `0` with nothing updated.
> - **Remedy:** Implement `ON CONFLICT (name, type) DO UPDATE` or explicit `update_problem` upsert path.

Agree with these solutions.

#### 3.1.2 Whole-batch transaction abort (`B3`)

> **High (`B3` — Whole-Batch Transaction Abort):** `src/converter/database/operations.py:552`
>
> - `batch_insert_problems` wraps all rows in one transaction; a single bad row rolls back the entire batch, and `operations.py:586` logs `"Batch insert completed: 1000/1000"` before commit.
> - **Remedy:** Per-instance savepoint or isolated commit with accurate post-commit count.

I may agree with _Per-instance savepoint_. On the topic of isolated commits, if you're talking about git commits, I do not agree. Which type of commits are you talking about?
But, beyond that, a batch operation does that in batches. Would the performance be impacted with any of those solutions when writing to db? I also agree taht on bad row breaks the entire batch.

#### 3.1.3 `INTEGER[][]` truncates fractional explicit weights (`M3`)

> **High (`M3` — `matrix INTEGER[][]` Truncates Fractional Explicit Weights):** `src/converter/database/operations.py:191-195` (and `gpu_accelerated_clean/src/loaders/tsplib_loader.py:508`)
>
> - Both repos declare `edge_weight_matrices.matrix INTEGER[][]`, silently rounding float explicit matrices (`1.5 → 2`).
> - **Remedy:** Change column type to `DOUBLE[][]` in `routing_data` (single source of truth).

Isn't rounding a `tsplib95` principle, as described in [tsplib95 format](../../../../routing_data/docs/reference/tsplib95_format.md)? And remember, `routing_data` repo extends the format beyond as well. Or can the `edge_weights` be a float? Following the rule in tsplib format is a must. If that's not possible, I didn't understand the question and need clarification.

#### 3.1.4 Swallowed transform errors (`M4`)

> **High (`M4` — Swallowed Transform Errors):** `src/converter/core/transformer.py:78-84`
>
> - `transform_problem` catches `Exception`, logs a warning, and returns `None`, hiding parser/contract bugs as skipped files.
> - **Remedy:** Raise typed `TransformError` so caller records a structured `WorkerFailure`.

I didn't even understand the problem, really. Isn't logging and raising a warning, good enough in this context? Why is it hiding parser/contract bugs, if the problem may be in the file that is not well written?
Please better explain the problem before I can take a decision.

#### 3.1.5 Symmetry guessed from format name (`M7`)

> **High (`M7` — Symmetry Guessed from Format Name Instead of Matrix Values):** `src/tsplib_parser/parser.py:727`
>
> - `FULL_MATRIX` is hardcoded `is_symmetric = False` even when the matrix is symmetric (`A == A.T`), while `src/tsplib_parser/validation.py:124` already implements `check_symmetry()`.
> - **Remedy:** Compute `is_symmetric` from the matrix values via `check_symmetry`.

That really is a bug that was overlooked. Not all `FULL_MATRIX` problems are symmetric, though most are. Assuming a `FULL_MATRIX` problem is not symmetric, avoids checking symmetry, but flags a problem that could be a `TSP` as an `ATSP`, for example.

### 3.2 Medium / Required (Architecture, Security, Type Contracts)

#### 3.2.1 No-op `converter validate` command (`M11`)

> **Medium (`M11` — No-Op `converter validate` Command):** `src/converter/cli/commands.py:231-269`
>
> - Violates `AGENTS.md §5.8` ("Validation Must Be Executable"): `converter validate` only prints `get_database_stats()` and exits `0`.

Please better explain these problem before I can take a decision.

#### 3.2.2 Path sanitization and SQL interpolation (`M2` / `m17`)

> **Medium (`M2` / `m17` — Path Sanitization & SQL Interpolation):**
>
> - `src/converter/output/json_writer.py:64,84`: uses `problem_data["name"]` from file header (`NAME : ../../foo`) without character whitelist validation.
> - `src/converter/output/parquet_writer.py:112,150`: interpolates `problem_type` and paths via f-strings instead of bound parameters.

How this affects the repo (e.g. possible cases where this would be a problem) and what are your proposed changes?

#### 3.2.3 Typing debt and untyped dict contracts (`T1`–`T5`)

> **Medium (`T1`–`T5` — Typing Debt & Untyped Dict Contracts):**
>
> - `src/converter/core/transformer.py:17,35`: `DataTransformer.transform_problem` returns `dict[str, Any] | None` instead of a named `TransformedProblem` `TypedDict` (`AGENTS.md §5.5`), and collides in name with `tsplib_parser/transformers.py` (`TODO.md` Batch 4).
> - Root scripts `convert_vrp.py` and `converter_cli.py` bypass the `[project.scripts] converter` entry point and hardcode `Vrp-Set-Solomon`.

Fully agree. This was a missed implementation flaw.

## 4. How to Use `routing_data` in `gpu_accelerated_clean`

`routing_data` is **already configured as a valid PEP 621 package** (`name = "routing-data"`, `where = ["src"]`, exporting top-level packages `tsplib_parser` and `converter` with `py.typed` markers in `routing_data/pyproject.toml:8-59`).

### 4.1 Why it isn't linked yet & small incompatibilities to resolve

#### 4.1.1 Dependency declaration

> **Dependency declaration:** `gpu_accelerated_clean/pyproject.toml` does not list `routing-data` under `dependencies` or `[tool.uv.sources]`. Adding `routing-data` as an editable path dependency (`{ path = "../routing_data", editable = true }`) in `gpu_accelerated_clean/pyproject.toml` makes `import tsplib_parser` and `import converter` immediately available in `gpu_accelerated_clean/.venv`.

I would prefer if the dependency used the repo directly, not the local path in my pc.
Isn't there a way to import the code as a package, with e.g. `uv add ...`? It seems better than an edittable dependency.

#### 4.1.2 DuckDB version constraint

> **DuckDB version constraint:** `routing_data/pyproject.toml:31` requires `duckdb>=1.5.6`, whereas `gpu_accelerated_clean/.venv` has `duckdb==1.4.1` (`gpu_accelerated_clean/pyproject.toml:22`). Relaxing `routing_data/pyproject.toml` to `duckdb>=1.4.0` makes both environments compatible without network downloads.

I don't know how each version is different, so if relaxing works, that's ok for me as well.

#### 4.1.3 Direct `Problem` construction from `tsplib_parser` + `DatabaseManager`

> **Direct `Problem` construction from `tsplib_parser` + `DatabaseManager`:**
>
> - **Generating / updating databases from `gpu_accelerated_clean`:** `tsplib.py` (`tsp-ingest`) and `tsplib_loader.py` can call `converter.api.process_directory` / `converter.database.operations.DatabaseManager` and `tsplib_parser.parser.FormatParser` directly instead of maintaining a 700-line copy of the parser + DDL in `tsplib_loader.py`.
> - **Accessing databases & solutions in `gpu_accelerated_clean`:** `DatabaseLoader` (`database_loader.py`) already reads `routing_data`'s DuckDB schema (`problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `file_tracking`). Adding a `load_solution(instance_name)` method (or populating optimal cost/tour from `solutions`) and pointing `TsplibLoader` fallback to `tsplib_parser.FormatParser` eliminates the `tsplib95` third-party dependency completely.

I agree with this decision.

---

## 5. Outdated, Duplicated, and Stale Documentation Inventory

Per `AGENTS.md` Rule 1 & §3.9, no files have been edited or deleted yet. Below is the complete verified inventory across both repositories for your approval.

### 5.A `routing_data` — Recommended **DELETE** (16 scratch files in `tmp/`)

#### 5.A.1 `tmp/src_exploration_report.md`

> `routing_data/tmp/src_exploration_report.md` | Claims `models.py` is 967 lines containing `Field` & `_SectionAccumulator` (now 345 lines; decomposed into `fields.py`, `transformers.py`, `tokenization.py`). Superseded by `ARCHITECTURE.md`.

Please delete.

#### 5.A.2 `tmp/analysis-test-suite-organization.md`

> `routing_data/tmp/analysis-test-suite-organization.md` | One-off migration note from before the `tests/` reorganization; superseded by `README.md`.

Yeah, outdated. Delete.

#### 5.A.3 Historical timing dumps

> `routing_data/tmp/baseline-durations.txt`, `routing_data/tmp/batch3-durations.txt` | Historical timing dumps referencing deleted `tests/test_converter/...` paths.

Delete. Super old files.

#### 5.A.4 One-off migration scripts

> `routing_data/tmp/*.py` (12 one-off migration scripts: `add_dataset_fixture.py`, `annotate_tests.py`, `fix_cli_fixtures.py`, `fix_imports.py`, `merge_tests.py`, `probe_adj.py`, `probe_folders.py`, `probe_parse.py`, `probe_parse2.py`, `rewrite_matrix_test.py`, `split_cordeau.py`, `tag_tests.py`, `tmp_path_migration.py`) | One-shot test migration scripts from completed batches; unreferenced anywhere in `src/` or `tests/`.

Yeah, those are something like "side scripts". They could be either deleted or be standalone helper scripts used as examples. Though, that would require renaming, reorganization, etc. Maybe this could be done by a subagent, but also should improve the scripts. This task also cannot be done earlier, as the API may change.

### 5.B `routing_data` — Recommended **EDIT** (Stale Code Anchors & Broken Links)

| File Path | Stale Claim (`path:line`) | Actual Code Truth |
| :--- | :--- | :--- |
| `routing_data/docs/ARCHITECTURE.md:175,250,472` | `:175` says `models.py` is 967 lines with `Field`/`:409` and `_SectionAccumulator`/`:723`; `:250` claims `_preprocess_vrp_file` writes a temp file unlinked in `finally`; `:472` lists `detect_special_distance_type`. | `models.py` is 345 lines (`fields.py` has 14 `Field` classes); `_preprocess_vrp_file` (`parser.py:221`) is pure in-memory text; `detect_special_distance_type` was deleted. |
| `routing_data/docs/src/tsplib_parser/parser-pipeline.md:42-132` | Links to `models.py#L832-L920`, `parser.py:854` (`detect_special_distance_type`), and temp-file preprocessing (`:54,:118`). | All anchors point to deleted/moved code (`StandardProblem.parse` is at `models.py:121`, `validate_problem` at `parser.py:560`). |
| `routing_data/docs/src/tsplib_parser/models.md:39,62,109` | `:109` shows `from tsplib_parser.models import FuncT, IntegerField` and `:62` links `models.py#L832-L920`. | Import fails (`IntegerField` is in `fields.py:92`, `FuncT` in `transformers.py`). |
| `routing_data/docs/studies/python-logic/**` (4 files: `practical/factory-pattern.md:209,241,316`, `practical/fields-vs-dataclasses.md:101`, `theoretical/classes-instances-cls.md:317-338`, `theoretical/descriptors-fields.md:288,308`) | References deleted `_SectionAccumulator`, `models.py:853`, and temp-file VRP preprocessing. | Update anchors to `fields.py`, `models.py:121`, and in-memory `parser.py:221`. |
| `routing_data/README.md:20,24,47,60,228,265,369` | `:60` `pip install -r requirements.txt` (file does not exist); `:20,265,369` link to non-existent `docs/reference/CONTRIBUTING.md`; `:24` vs `:47` contradict on test counts (134 vs 287 vs actual 406). | Replace `requirements.txt` with `uv sync`, point contributing links to `AGENTS.md`, align test counts with `README.md`. |
| `routing_data/docs/README.md:47,53` | Links to missing `reference/CONTRIBUTING.md` and missing `development/` folder; omits `src` and `studies`. | Fix index links to match actual `docs` tree. |
| `routing_data/docs/guides/USER_GUIDE.md:677-683` | Commands use deleted test paths `pytest tests/test_converter` and `tests/test_integration`. | Actual paths are `converter`, `tsplib_parser`, `e2e`. |
| `routing_data/src/converter/api.py:46` & `parquet_writer.py:27` | Docstrings still describe the dropped v1 `nodes` table. | Update docstrings to v2 schema (`tsp_problems`, `cvrp_problems`, etc.). |

Agree with everything. Though this must also be done after the tasks with the problems from before [chapter 5's problems](#5-outdated-duplicated-and-stale-documentation-inventory).

### 5.C `gpu_accelerated_clean` — Recommended **DELETE** (5 Superseded / Duplicate Markdown Files)

| File Path | Evidence Why Obsolete / Duplicated |
| :--- | :--- |
| `gpu_accelerated_clean/daily_planning.md` | Explicitly superseded by `notes_project_direction_questions.md:4` (_"Consolidates `meeting-notes.md` and daily_planning.md"_). |
| `gpu_accelerated_clean/meeting-notes.md` | Explicitly superseded by `notes_project_direction_questions.md:4`. |
| `gpu_accelerated_clean/modularizing-cuda-kernels.md` | Historical pre-modularization plan referencing `benchmarking_v2/`; superseded by completed walkthroughs in `walkthroughs`. |
| `gpu_accelerated_clean/documentation/analysis/presentations/key_findings.md` | Scratch action plan referencing deleted `tmp/recompute_stats.py`, `tmp/stats_output/`, and `src.benchmarking_v2`; content duplicated in `thesis_summary.md` and `thesis_presentation.md`. |
| `gpu_accelerated_clean/documentation/critical_analysis/criticas_analysis_bib.md` | Near-duplicate of `documentation/critical_analysis/critical_analysis.md` with broken `../gpu_accelerated/TCC_context/...` path. |

Agree with this as well.

### 5.D `gpu_accelerated_clean` — Recommended **EDIT** (Stale Paths in Docs)

| File Path | Stale Claim (`path:line`) | Actual Code Truth |
| :--- | :--- | :--- |
| `gpu_accelerated_clean/README.md:52-53,82-83` | References `src/benchmarking_v2/` and `src/benchmarks_v2/`. | Actual packages are `benchmarking` and `cli` (`tsp-ingest`, `tsp-benchmark`, `tsp-import-checkpoints`, `tsp-validate-db`). |
| `gpu_accelerated_clean/HARDWARE_SPECS.md:87,99` | References `GA_PARAMS` and `BENCHMARK_PARAMS` python constants. | Moved to `algorithms.json` and `benchmark.json`. |
| `gpu_accelerated_clean/documentation/analysis/benchmarking/benchmarking_system.md:26,944-961`, `module_interactions.md:25,806`, `algorithm_comparison.md:782`, `fact_check_report.md:11` | Cite `src/benchmarking_v2/`, `code/benchmarks_v2/`, and `tmp/recompute_stats.py`. | Update paths to `benchmarking`, `configs`, and `benchmark_scripts`. |

Agree with this as well.
