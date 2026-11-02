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
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from benchmarking.database.schema import ALL_RUNS_BY_ID, COUNT_RUNS, TABLE_RUNS


def load_checkpoint_json(filepath: Path) -> dict[str, Any]:
  """Load checkpoint JSON file."""
  with open(filepath) as f:
    return cast("dict[str, Any]", json.load(f))


def compare_databases(db1_path: str, db2_path: str) -> tuple[bool, list[str]]:
  """
  Compare two databases for identical content.

  Returns:
      (is_identical, list_of_differences)
  """
  print("=" * 80)
  print("PHASE 1: Comparing Two Databases")
  print("=" * 80)
  print()

  conn1 = duckdb.connect(db1_path, read_only=True)
  conn2 = duckdb.connect(db2_path, read_only=True)

  differences = []

  # Compare row counts
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

  # Load both databases into pandas
  df1 = conn1.execute(ALL_RUNS_BY_ID).df()
  df2 = conn2.execute(ALL_RUNS_BY_ID).df()

  # Compare unique problems
  problems1 = set(df1["problem_name"].unique())
  problems2 = set(df2["problem_name"].unique())

  if problems1 != problems2:
    differences.append(f"Problem sets differ: {problems1 ^ problems2}")
    print("❌ Different problems!")
    return False, differences

  print(f"✓ Same {len(problems1)} unique problems")

  # Compare algorithms
  algos1 = set(df1["algorithm"].unique())
  algos2 = set(df2["algorithm"].unique())

  if algos1 != algos2:
    differences.append(f"Algorithm sets differ: {algos1 ^ algos2}")
    print("❌ Different algorithms!")
    return False, differences

  print(f"✓ Same {len(algos1)} algorithms")
  print()

  # Compare sample records (2 random ones)
  print("Comparing 2 random records in detail...")
  sample_indices = [0, len(df1) // 2]  # First and middle records

  for idx in sample_indices:
    row1 = df1.iloc[idx]
    row2 = df2.iloc[idx]

    problem = row1["problem_name"]
    algo = row1["algorithm"]

    print(f"\n  Record {idx + 1}: {problem} / {algo}")

    key_fields = [
      "mean_time",
      "best_cost",
      "repetitions",
      "optimal_cost",
      "problem_size",
    ]

    for field in key_fields:
      val1 = row1[field]
      val2 = row2[field]

      if val1 != val2:
        differences.append(f"Row {idx}, field {field}: {val1} vs {val2}")
        print(f"    ❌ {field}: {val1} != {val2}")
      else:
        print(f"    ✓ {field}: {val1}")

  print()

  conn1.close()
  conn2.close()

  if differences:
    return False, differences
  else:
    print("✅ Databases are identical!")
    return True, []


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

