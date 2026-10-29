"""Table and query names shared by the benchmark result database.

`schema.sql` in this directory is the authority for columns, types and
nullability. This module only removes the literal repetitions from the Python
side, so renaming the table is one edit instead of nineteen.

Column names are deliberately *not* constant-ised. The `INSERT` is a
hand-written SQL string, so a `COL_MEAN_TIME` constant would make the statement
noisier without catching anything extra; what catches a mistyped column is the
`BenchmarkRunInput` TypedDict on the payload (`data_models.records`), because
`data["mean_tme"]` is a checker error while `mean_tme` inside a SQL string is
not visible to any checker.
"""

#: Main results table: one row per checkpoint file.
TABLE_RUNS = "benchmark_runs"

#: Every run, oldest first. Used by the consistency validators.
ALL_RUNS_BY_ID = f"SELECT * FROM {TABLE_RUNS} ORDER BY run_id"

#: Row count, used before and after an import to report how much was written.
COUNT_RUNS = f"SELECT COUNT(*) FROM {TABLE_RUNS}"

