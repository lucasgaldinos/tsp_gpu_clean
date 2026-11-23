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

- [schema.sql](./schema.sql) — DuckDB `benchmark_runs` table DDL and indexes (`optimal_cost` and gap columns are nullable for custom `.tsp` instances without a known optimum).
- [schema.py](./schema.py) — Table name and reusable SQL query constants.
- [db_operations.py](./db_operations.py) — Schema creation, payload validation ([db_operations.py::validate_benchmark_data](./db_operations.py)), insertion ([db_operations.py::insert_benchmark_run](./db_operations.py)), and query helpers.
- [import_checkpoints.py](./import_checkpoints.py) — Idempotent batch importer exposed through the `tsp-import-checkpoints` CLI.
- [validate_consistency.py](./validate_consistency.py) — Cross-validation between DuckDB and source JSON checkpoints exposed through the `tsp-validate-db` CLI.

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
