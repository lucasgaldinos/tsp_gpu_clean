---
title: "Quick Start: Benchmark Results Database (Optional DuckDB Layer)"
description: >
  Operational reference for the optional DuckDB results persistence module under
  `src/benchmarking/database/`, showing CLI checkpoint ingestion, consistency
  validation, and direct SQL queries.
created: 2025-11-23
modifications:
  - date_modified: 2026-11-23
    modifications:
      - description: >
          Replaced dead `python -m benchmarking.database.*` invocations and broken
          `README.md` reference with the registered `tsp-import-checkpoints` and
          `tsp-validate-db` CLIs, clarifying that DuckDB consolidation is optional
          and adding YAML frontmatter compliant with repository standards.
      - description: >
          Noted full-table deterministic `(problem_name, algorithm)` comparison and
          guaranteed connection cleanup in `validate_consistency.py::compare_databases`.
  - date_modified: 2026-11-27
    modifications:
      - description: >
          Updated schema notes for VRP Tier 1 support: `54`-column `benchmark_runs`
          table with `problem_type` (`DEFAULT 'TSP'`), nullable cost/gap/VRP columns,
          and `(problem_name, problem_type, algorithm)` natural-key disambiguation;
          removed internal plan/bug identifiers from user-facing guide.
related_files:
  - "[schema.sql](./schema.sql)"
  - "[db_operations.py](./db_operations.py)"
  - "[checkpoints.py](../../cli/checkpoints.py)"
  - "[guide-getting-started.md](../../../documentation/guide-getting-started.md)"
tags:
  - guide/database
  - guide/quickstart
  - guide/cli
  - benchmark/database
  - benchmark/checkpoints
  - benchmark/validation
  - notes/duckdb
  - notes/tooling
  - analysis/benchmarking
  - review/quality
---

# Quick Start: Benchmark Results Database

> [!NOTE]
> Consolidating JSON checkpoints into DuckDB is **optional**. Both `tsp-benchmark` and `tsp-report` generate all Markdown/LaTeX tables, quantitative macros, Demšar CD diagrams, and scaling figures directly from JSON files under `checkpoints/`. Use this module when you want SQL querying or archival consistency auditing across runs.

## Module Layout

**Directory:** `src/benchmarking/database/`

- [schema.sql](./schema.sql) — DuckDB `benchmark_runs` table DDL (`54` columns) and indexes (`problem_type VARCHAR NOT NULL DEFAULT 'TSP'`, nullable `mean_cost`/`std_cost`/`best_cost`/`worst_cost` for `feasible_runs == 0`, nullable `optimal_cost`/gap columns, and optional VRP feasibility/fleet scalars and `LIST` arrays).
- [schema.py](./schema.py) — Table name and reusable SQL query constants.
- [db_operations.py](./db_operations.py) — Schema creation, payload validation ([db_operations.py::validate_benchmark_data](./db_operations.py)), insertion ([db_operations.py::insert_benchmark_run](./db_operations.py)), and query helpers.
- [import_checkpoints.py](./import_checkpoints.py) — Idempotent batch importer keyed on `(problem_name, problem_type, algorithm)` exposed through the `tsp-import-checkpoints` CLI.
- [validate_consistency.py](./validate_consistency.py) — Cross-validation between DuckDB and source JSON checkpoints (`validate_against_checkpoints`), plus full-table row-by-row database comparison ordered by `(problem_name, problem_type, algorithm, timestamp, run_id)` ([validate_consistency.py::compare_databases](./validate_consistency.py)), exposed through the `tsp-validate-db` CLI.

## 1. Import Checkpoints via CLI

```bash
# Import every *.json checkpoint into ./results/results.duckdb
uv run tsp-import-checkpoints --input ./results/checkpoints --output-dir ./results

# Validate the database against the source JSON checkpoints (--db2 is optional)
uv run tsp-validate-db --db1 ./results/results.duckdb --checkpoints ./results/checkpoints
```

## 2. Inspect with SQL

```bash
duckdb ./results/results.duckdb
```

```sql
-- Count consolidated runs
SELECT COUNT(*) FROM benchmark_runs;

-- Compare mean time and optimality gap for one instance
SELECT algorithm, mean_time, mean_cost, mean_gap
FROM benchmark_runs
WHERE problem_name = 'berlin52'
ORDER BY mean_time;
```

## 3. Programmatic Validation & Queries

```python
import json
from benchmarking.database.db_operations import (
  compute_speedup_matrix,
  create_database,
  get_problems_with_cpu,
  insert_benchmark_run,
  validate_benchmark_data,
)

conn = create_database("results/results.duckdb")

with open("results/checkpoints/berlin52_CPU.json", encoding="utf-8") as fh:
  payload = validate_benchmark_data(json.load(fh))

run_id = insert_benchmark_run(conn, payload)
print(f"Inserted run_id={run_id}")
print("CPU instances:", get_problems_with_cpu(conn))
print("Speedups:", compute_speedup_matrix(conn, "CPU", ["AlgA", "AlgB"]))
conn.close()
```
