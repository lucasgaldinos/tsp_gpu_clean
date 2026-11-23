"""
Database Consistency Validation Script
=======================================

Validates that:
1. Two databases contain identical data
2. Database contents match source JSON files exactly

Usage:
------
    python -m benchmarking.database.validate_consistency \\
        --db1 results.duckdb \\
        --db2 my_notes/results.duckdb \\
        --checkpoints code/benchmarks/results/checkpoints/

Author: GPU Accelerated Routing Optimization Team
Date: 2025-11-23
"""

import json
import math
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd

from benchmarking.database.schema import ALL_RUNS_BY_ID, COUNT_RUNS, TABLE_RUNS

_EXCLUDED_COMPARE_COLUMNS: frozenset[str] = frozenset({"run_id", "created_at"})


def _is_scalar_na(value: Any) -> bool:
  """Return True only when ``value`` is a scalar NULL/NaN (never raises on arrays/lists)."""
  if value is None:
    return True
  if isinstance(value, (list, tuple, dict, np.ndarray)):
    return False
  try:
    return bool(pd.isna(value))
  except (TypeError, ValueError):
    return False


def _values_equal(val1: Any, val2: Any) -> bool:
  """Compare two DuckDB/pandas cell values safely across scalars, floats, NULLs, and LIST arrays."""
  if _is_scalar_na(val1) and _is_scalar_na(val2):
    return True
  if _is_scalar_na(val1) or _is_scalar_na(val2):
    return False

  if isinstance(val1, (list, tuple, np.ndarray)) and isinstance(val2, (list, tuple, np.ndarray)):
    seq1 = list(val1)
    seq2 = list(val2)
    return len(seq1) == len(seq2) and all(_values_equal(a, b) for a, b in zip(seq1, seq2, strict=True))

  if isinstance(val1, bool) or isinstance(val2, bool):
    return isinstance(val1, bool) and isinstance(val2, bool) and val1 == val2

  if isinstance(val1, (int, float, np.integer, np.floating)) and isinstance(
    val2, (int, float, np.integer, np.floating)
  ):
    return math.isclose(float(val1), float(val2), rel_tol=1e-9, abs_tol=1e-12)

  return bool(val1 == val2)


def load_checkpoint_json(filepath: Path) -> dict[str, Any]:
  """Load checkpoint JSON file."""
  with open(filepath) as f:
    return cast("dict[str, Any]", json.load(f))


def compare_databases(db1_path: str, db2_path: str) -> tuple[bool, list[str]]:
  """Compare two DuckDB benchmark databases across every row ordered by ``(problem_name, algorithm)``.

  Returns:
      ``(is_identical, list_of_differences)`` with DuckDB connections always closed.
  """
  print("=" * 80)
  print("PHASE 1: Comparing Two Databases")
  print("=" * 80)
  print()

  conn1: duckdb.DuckDBPyConnection | None = None
  conn2: duckdb.DuckDBPyConnection | None = None
  differences: list[str] = []

  try:
    conn1 = duckdb.connect(db1_path, read_only=True)
    conn2 = duckdb.connect(db2_path, read_only=True)

    r1 = conn1.execute(COUNT_RUNS).fetchone()
    r2 = conn2.execute(COUNT_RUNS).fetchone()
    count1 = r1[0] if r1 else 0
    count2 = r2[0] if r2 else 0

    print(f"Database 1: {count1} rows")
    print(f"Database 2: {count2} rows")

    if count1 != count2:
      differences.append(f"Row count mismatch: {count1} vs {count2}")
      print("❌ Row count mismatch!")
      return False, differences

    print(f"✓ Row counts match: {count1} rows")
    print()

    if count1 == 0:
      print("✅ Databases are identical (both empty)!")
      return True, []

    df1 = conn1.execute(ALL_RUNS_BY_ID).df()
    df2 = conn2.execute(ALL_RUNS_BY_ID).df()

    if list(df1.columns) != list(df2.columns):
      differences.append(f"Column sets differ: {set(df1.columns) ^ set(df2.columns)}")
      print("❌ Different table columns!")
      return False, differences

    problems1 = set(df1["problem_name"].unique())
    problems2 = set(df2["problem_name"].unique())
    if problems1 != problems2:
      differences.append(f"Problem sets differ: {problems1 ^ problems2}")
      print("❌ Different problems!")
      return False, differences
    print(f"✓ Same {len(problems1)} unique problems")

    algos1 = set(df1["algorithm"].unique())
    algos2 = set(df2["algorithm"].unique())
    if algos1 != algos2:
      differences.append(f"Algorithm sets differ: {algos1 ^ algos2}")
      print("❌ Different algorithms!")
      return False, differences
    print(f"✓ Same {len(algos1)} algorithms")
    print()

    sort_cols = [c for c in ("problem_name", "algorithm", "timestamp", "run_id") if c in df1.columns]
    df1_sorted = df1.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    df2_sorted = df2.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    compare_cols = [c for c in df1_sorted.columns if c not in _EXCLUDED_COMPARE_COLUMNS]
    print(f"Comparing all {len(df1_sorted)} records across {len(compare_cols)} columns...")

    for idx in range(len(df1_sorted)):
      row1 = df1_sorted.iloc[idx]
      row2 = df2_sorted.iloc[idx]
      row_label = f"{row1['problem_name']} / {row1['algorithm']}"
      for field in compare_cols:
        val1 = row1[field]
        val2 = row2[field]
        if not _values_equal(val1, val2):
          differences.append(f"Row {idx} ({row_label}), field {field}: {val1!r} vs {val2!r}")

    if differences:
      print(f"❌ Found {len(differences)} cell difference(s)!")
      return False, differences

    print("✅ Databases are identical!")
    return True, []
  finally:
    if conn1 is not None:
      conn1.close()
    if conn2 is not None:
      conn2.close()


def validate_against_checkpoints(db_path: str, checkpoint_dir: str) -> tuple[bool, pd.DataFrame]:
  """
  Validate database contents against source JSON files.

  Returns:
      (all_valid, mismatches_dataframe)
  """
  print()
  print("=" * 80)
  print("PHASE 2: Validating Database Against Source JSON Files")
  print("=" * 80)
  print()

  conn = duckdb.connect(db_path, read_only=True)
  df = conn.execute(f"SELECT * FROM {TABLE_RUNS} ORDER BY problem_name, algorithm").df()
  conn.close()

  checkpoint_path = Path(checkpoint_dir)
  mismatches = []
  validated_count = 0

  print(f"Validating {len(df)} database records...")
  print()

  for _, row in df.iterrows():
    problem = row["problem_name"]
    algo = row["algorithm"]

    checkpoint_file = checkpoint_path / f"{problem}_{algo}.json"

    if not checkpoint_file.exists():
      mismatches.append(
        {
          "run_id": row["run_id"],
          "problem": problem,
          "algorithm": algo,
          "issue": "checkpoint_file_not_found",
          "field": "N/A",
          "db_value": "N/A",
          "json_value": "N/A",
        }
      )
      continue

    json_data = load_checkpoint_json(checkpoint_file)

    fields_to_check = [
      "algorithm",
      "repetitions",
      "successful_runs",
      "mean_time",
      "std_time",
      "min_time",
      "max_time",
      "mean_cost",
      "std_cost",
      "best_cost",
      "worst_cost",
      "mean_gap",
      "std_gap",
      "best_gap",
      "problem_name",
      "problem_size",
      "optimal_cost",
    ]

    record_valid = True
    for field in fields_to_check:
      if field not in json_data:
        continue

      db_value = row[field]
      json_value = json_data[field]

      if json_value is None and pd.isna(db_value):
        continue
      if isinstance(db_value, float) and isinstance(json_value, (int, float)):
        if abs(db_value - json_value) > 1e-6:
          record_valid = False
          mismatches.append(
            {
              "run_id": row["run_id"],
              "problem": problem,
              "algorithm": algo,
              "issue": "value_mismatch",
              "field": field,
              "db_value": db_value,
              "json_value": json_value,
            }
          )
      else:
        if db_value != json_value:
          record_valid = False
          mismatches.append(
            {
              "run_id": row["run_id"],
              "problem": problem,
              "algorithm": algo,
              "issue": "value_mismatch",
              "field": field,
              "db_value": db_value,
              "json_value": json_value,
            }
          )

    if record_valid:
      validated_count += 1

  print(f"✓ Validated {validated_count}/{len(df)} records")

  if mismatches:
    print(f"❌ Found {len(mismatches)} mismatches")
    mismatch_df = pd.DataFrame(mismatches)
    return False, mismatch_df
  else:
    print("✅ All records match source JSON files!")
    return True, pd.DataFrame()
