"""
Database Operations Module for Benchmark Results
=================================================

Solution 1: Pure Functions Module (~150 LOC, YAGNI-compliant)

This module provides type-safe database operations for storing and querying
GPU-accelerated routing optimization benchmark results.

Features:
- Type safety: NaN validation, NULL checks, array length consistency
- Pure functions: No global state, explicit connection passing
- DuckDB-specific: Uses native LIST and JSON types
- Error handling: Comprehensive validation with clear error messages

Usage Example:
--------------
    import duckdb
    from benchmarking.database.db_operations import create_database, insert_benchmark_run

    # Create database with schema
    conn = create_database('results.duckdb')

    # Insert benchmark data
    data = {
        'problem_name': 'berlin52',
        'algorithm': 'HybridOptimized',
        'repetitions': 15,
        'mean_time': 1.234,
        'raw_times': [1.2, 1.3, 1.1, ...],
        # ... other fields
    }
    insert_benchmark_run(conn, data)

    conn.close()

Dependencies:
-------------
    pip install duckdb

Author: GPU Accelerated Routing Optimization Team
Date: 2025-11-23
"""

import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb

from benchmarking.database.schema import TABLE_RUNS
from data_models.records import BenchmarkRunInput

# ============================================================================
# Database Initialization
# ============================================================================


def create_database(db_path: str) -> duckdb.DuckDBPyConnection:
  """
  Create or connect to DuckDB database and initialize schema.

  Args:
      db_path: Path to database file (e.g., 'results.duckdb')

  Returns:
      DuckDB connection object

  Example:
      >>> conn = create_database("results/results.duckdb")
  """
  conn = duckdb.connect(db_path)

  # Read and execute schema
  schema_path = Path(__file__).parent / "schema.sql"
  with open(schema_path) as f:
    schema_sql = f.read()

  # Execute schema (handles IF NOT EXISTS)
  conn.execute(schema_sql)

  return conn


# ============================================================================
# Type Safety Validation
# ============================================================================


def validate_benchmark_data(data: Mapping[str, Any]) -> BenchmarkRunInput:
  """
  Validate benchmark data for type safety before insertion.

  Checks:
  - Required fields present
  - No NaN values in numeric fields
  - Array lengths match repetitions

  Args:
      data: Dictionary containing benchmark run data

  Returns:
      Validated BenchmarkRunInput TypedDict

  Raises:
      ValueError: If validation fails with specific error message

  Example:
      >>> record = validate_benchmark_data(checkpoint_json)
      >>> insert_benchmark_run(conn, record)
  """
  # Check required fields
  required_fields = list(BenchmarkRunInput.__required_keys__)

  missing = [f for f in required_fields if f not in data]
  if missing:
    raise ValueError(f"Missing required fields: {', '.join(missing)}")

  # Check numeric fields for NaN
  numeric_fields = [
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
    "mean_generations",
    "std_generations",
  ]

  for field in numeric_fields:
    val = data.get(field)
    if val is not None and isinstance(val, (int, float)) and math.isnan(val):
      raise ValueError(f"Field '{field}' cannot be NaN")

  # Optional numeric fields (check if present)
  optional_numeric = [
    "mean_initial_cost",
    "std_initial_cost",
    "mean_improvement",
    "mean_h2d_mb",
    "mean_d2h_mb",
    "total_transfer_mb",
    "mean_kernels",
  ]

  for field in optional_numeric:
    val = data.get(field)
    if val is not None and isinstance(val, (int, float)) and math.isnan(val):
      raise ValueError(f"Optional field '{field}' cannot be NaN")

  # Validate array lengths match repetitions
  repetitions = data["repetitions"]
  array_fields = ["raw_times", "raw_costs", "raw_gaps"]

  for field in array_fields:
    arr = data.get(field)
    if arr is not None and len(arr) != repetitions:
      raise ValueError(
        f"Array length mismatch for '{field}': expected {repetitions}, got {len(arr)}"
      )

  # Check optional arrays if present
  optional_arrays = ["raw_initial_costs", "raw_generations", "raw_stop_reasons"]
  for field in optional_arrays:
    arr = data.get(field)
    if arr is not None and len(arr) != repetitions:
      raise ValueError(
        f"Array length mismatch for optional '{field}': expected {repetitions}, got {len(arr)}"
      )

  return cast("BenchmarkRunInput", data)


# ============================================================================
# Insert Operations
# ============================================================================


def insert_benchmark_run(
  conn: duckdb.DuckDBPyConnection, data: BenchmarkRunInput
) -> int:
  """
  Insert a benchmark run into DuckDB.

  Uses native DuckDB types:
  - LIST for raw arrays (INTEGER[], DOUBLE[], VARCHAR[])
  - JSON for nested configurations (parsed JSON, not string)
  - TIMESTAMP for run time

  Args:
      conn: DuckDB connection
      data: Validated BenchmarkRunInput dictionary

  Returns:
      run_id: Auto-generated primary key

  Example:
      >>> record = validate_benchmark_data(checkpoint)
      >>> run_id = insert_benchmark_run(conn, record)
      >>> print(f"Inserted run {run_id}")
  """
  # Parse timestamp if string
  ts_raw = data["timestamp"]
  ts: datetime
  if isinstance(ts_raw, str):
    # Try ISO format
    try:
      ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except ValueError:
      # Try common format: "2025-11-23 12:00:00"
      ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
  else:
    ts = ts_raw

  # Convert configs to JSON string if dict
  algorithm_config = data["algorithm_config"]
  if isinstance(algorithm_config, dict):
    algorithm_config = json.dumps(algorithm_config)

  strategies = data["strategies"]
  if isinstance(strategies, dict):
    strategies = json.dumps(strategies)

  # Insert query using parameterized values
  query = f"""
        INSERT INTO {TABLE_RUNS} (
            problem_name,
            problem_size,
            optimal_cost,
            algorithm,
            backend,
            repetitions,
            successful_runs,
            timestamp,
            max_generations_configured,
            mean_time,
            std_time,
            min_time,
            max_time,
            mean_cost,
            std_cost,
            best_cost,
            worst_cost,
            mean_initial_cost,
            std_initial_cost,
            mean_gap,
            std_gap,
            best_gap,
            mean_improvement,
            mean_generations,
            std_generations,
            min_generations,
            max_generations,
            mean_h2d_mb,
            mean_d2h_mb,
            total_transfer_mb,
            mean_kernels,
            raw_times,
            raw_costs,
            raw_gaps,
            raw_initial_costs,
            raw_generations,
            raw_stop_reasons,
            algorithm_config,
            strategies,
            construction_heuristic
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        RETURNING run_id
    """

  result = conn.execute(
    query,
    [
      data["problem_name"],
      data["problem_size"],
      data["optimal_cost"],
      data["algorithm"],
      data["backend"],
      data["repetitions"],
      data["successful_runs"],
      ts,
      data["max_generations_configured"],
      data["mean_time"],
      data["std_time"],
      data["min_time"],
      data["max_time"],
      data["mean_cost"],
      data["std_cost"],
      data["best_cost"],
      data["worst_cost"],
      data.get("mean_initial_cost"),
      data.get("std_initial_cost"),
      data["mean_gap"],
      data["std_gap"],
      data["best_gap"],
      data.get("mean_improvement"),
      data["mean_generations"],
      data["std_generations"],
      data.get("min_generations"),
      data.get("max_generations"),
      data.get("mean_h2d_mb"),
      data.get("mean_d2h_mb"),
      data.get("total_transfer_mb"),
      data.get("mean_kernels"),
      data["raw_times"],
      data["raw_costs"],
      data["raw_gaps"],
      data.get("raw_initial_costs"),
      data.get("raw_generations"),
      data.get("raw_stop_reasons"),
      algorithm_config,
      strategies,
      data.get("construction_heuristic"),
    ],
  )

  row = result.fetchone()
  if row is None:
    raise RuntimeError("Failed to insert run: no row returned from RETURNING run_id")
  return int(row[0])


# ============================================================================
# Query Utilities
# ============================================================================


def get_problems_with_cpu(conn: duckdb.DuckDBPyConnection) -> list[str]:
  """
  Get list of problems that have CPU benchmark data.

  Returns:
      List of problem names (e.g., ['berlin52', 'st70', ...])

  Example:
      >>> problems = get_problems_with_cpu(conn)
      >>> print(f"CPU data available for {len(problems)} problems")
  """
  result = conn.execute(f"""
        SELECT DISTINCT problem_name
        FROM {TABLE_RUNS}
        WHERE algorithm = 'CPU'
        ORDER BY problem_size
    """).fetchall()

  return [row[0] for row in result]


def get_algorithm_summary(
  conn: duckdb.DuckDBPyConnection, problem_name: str, algorithm: str
) -> dict[str, Any] | None:
  """
  Get summary statistics for a specific algorithm on a problem.

  Args:
      conn: DuckDB connection
      problem_name: Problem name
      algorithm: Algorithm name

  Returns:
      Dictionary with summary stats or None if not found

  Example:
      >>> stats = get_algorithm_summary(conn, "berlin52", "HybridOptimized")
      >>> print(f"Mean time: {stats['mean_time']:.2f}s")
  """
  result = conn.execute(
    f"""
        SELECT
            mean_time, std_time,
            mean_cost, std_cost,
            best_cost, best_gap,
            repetitions, is_test,
            timestamp
        FROM {TABLE_RUNS}
        WHERE problem_name = ? AND algorithm = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """,
    [problem_name, algorithm],
  ).fetchone()

  if result is None:
    return None

  return {
    "mean_time": result[0],
    "std_time": result[1],
    "mean_cost": result[2],
    "std_cost": result[3],
    "best_cost": result[4],
    "best_gap": result[5],
    "repetitions": result[6],
    "is_test": result[7],
    "timestamp": result[8],
  }


def compute_speedup_matrix(
  conn: duckdb.DuckDBPyConnection,
  baseline: str = "CPU",
  comparisons: list[str] | None = None,
) -> list[tuple[str, str, float, float]]:
  """
  Compute speedup of each algorithm vs baseline across all problems.

  Args:
      conn: DuckDB connection
      baseline: Baseline algorithm (default 'CPU')
      comparisons: List of algorithms to compare (default all GPU algorithms)

  Returns:
      List of (problem_name, algorithm, speedup, baseline_time) tuples

  Example:
      >>> speedups = compute_speedup_matrix(conn, "CPU", ["HybridOptimized", "FullGPU"])
      >>> for prob, alg, speedup, base_time in speedups:
      ...   print(f"{prob}: {alg} is {speedup:.2f}x faster than CPU")
  """
  if comparisons is None:
    comparisons = ["HybridNaive", "HybridOptimized", "FullGPU"]

  results = []

  # Get all problems with baseline data
  problems = conn.execute(
    f"""
        SELECT DISTINCT problem_name
        FROM {TABLE_RUNS}
        WHERE algorithm = ? AND is_test = FALSE
    """,
    [baseline],
  ).fetchall()

  for (problem_name,) in problems:
    # Get baseline time
    baseline_time = conn.execute(
      f"""
            SELECT mean_time
            FROM {TABLE_RUNS}
            WHERE problem_name = ? AND algorithm = ? AND is_test = FALSE
        """,
      [problem_name, baseline],
    ).fetchone()

    if baseline_time is None:
      continue

    base_time_val = float(baseline_time[0])

    # Compare each algorithm
    for comp_algorithm in comparisons:
      comp_time = conn.execute(
        f"""
                SELECT mean_time
                FROM {TABLE_RUNS}
                WHERE problem_name = ? AND algorithm = ? AND is_test = FALSE
            """,
        [problem_name, comp_algorithm],
      ).fetchone()

      if comp_time is not None:
        speedup = base_time_val / float(comp_time[0])
        results.append((problem_name, comp_algorithm, speedup, base_time_val))

  return results


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
  print("Database Operations Module - Quick Test")
  print("=" * 60)

  # Create test database
  test_conn = create_database(":memory:")  # In-memory database

  # Create test data (no need to insert problem separately)
  test_data: BenchmarkRunInput = {
    "problem_name": "test_problem",
    "problem_size": 10,
    "optimal_cost": 100.0,
    "algorithm": "TestAlgo",
    "backend": "CPU",
    "repetitions": 3,
    "successful_runs": 3,
    "timestamp": "2025-11-23 12:00:00",
    "max_generations_configured": 100,
    "mean_time": 1.5,
    "std_time": 0.1,
    "min_time": 1.4,
    "max_time": 1.6,
    "mean_cost": 100.0,
    "std_cost": 0.0,
    "best_cost": 100.0,
    "worst_cost": 100.0,
    "mean_gap": 0.0,
    "std_gap": 0.0,
    "best_gap": 0.0,
    "mean_generations": 10.0,
    "std_generations": 1.0,
    "raw_times": [1.4, 1.5, 1.6],
    "raw_costs": [100.0, 100.0, 100.0],
    "raw_gaps": [0.0, 0.0, 0.0],
    "algorithm_config": {"param1": 1, "param2": 2},
    "strategies": {"strategy1": "A", "strategy2": "B"},
  }

  # Validate and insert
  try:
    record = validate_benchmark_data(test_data)
    test_run_id = insert_benchmark_run(test_conn, record)
    print(f"✓ Inserted test benchmark run (run_id: {test_run_id})")

    # Query back
    summary = get_algorithm_summary(test_conn, "test_problem", "TestAlgo")
    if summary is not None:
      print(f"✓ Retrieved summary: mean_time={summary['mean_time']:.2f}s")

    print("\n✅ All tests passed! Module ready for use.")

  except Exception as e:
    print(f"❌ Test failed: {e}")

  finally:
    test_conn.close()
