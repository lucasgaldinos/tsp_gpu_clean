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
        'algorithm': 'MemeticGA',
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
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np

from benchmarking.database.schema import TABLE_RUNS
from data_models.records import BenchmarkRunInput, JsonPayload

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
    "mean_generations",
    "std_generations",
  ]

  for field in numeric_fields:
    val = data.get(field)
    if val is not None and isinstance(val, (int, float)) and math.isnan(val):
      raise ValueError(f"Field '{field}' cannot be NaN")

  # Optional numeric fields (check if present)
  optional_numeric = [
    "mean_cost",
    "std_cost",
    "best_cost",
    "worst_cost",
    "optimal_cost",
    "mean_gap",
    "std_gap",
    "best_gap",
    "mean_initial_cost",
    "std_initial_cost",
    "mean_improvement",
    "mean_h2d_mb",
    "mean_d2h_mb",
    "total_transfer_mb",
    "mean_kernels",
    "feasibility_rate",
    "mean_vehicles",
  ]

  for field in optional_numeric:
    val = data.get(field)
    if val is not None and isinstance(val, (int, float)) and math.isnan(val):
      raise ValueError(f"Optional field '{field}' cannot be NaN")

  # Validate array lengths match successful_runs (or repetitions if absent) (Task 12.10, BUG-36)
  repetitions = data["repetitions"]
  successful_runs = data.get("successful_runs", repetitions)
  if successful_runs < 0 or successful_runs > repetitions:
    raise ValueError(f"successful_runs ({successful_runs}) cannot be negative or exceed repetitions ({repetitions})")

  expected_len = successful_runs
  array_fields = ["raw_times", "raw_costs"]

  for field in array_fields:
    arr = data.get(field)
    if arr is not None and len(arr) != expected_len:
      raise ValueError(f"Array length mismatch for '{field}': expected {expected_len}, got {len(arr)}")

  # Check optional arrays if present
  optional_arrays = [
    "raw_gaps",
    "raw_initial_costs",
    "raw_generations",
    "raw_stop_reasons",
    "raw_is_feasible",
    "raw_unpenalized_costs",
    "raw_violations",
    "raw_num_vehicles",
  ]
  for field in optional_arrays:
    arr = data.get(field)
    if arr is not None and len(arr) != expected_len:
      raise ValueError(f"Array length mismatch for optional '{field}': expected {expected_len}, got {len(arr)}")

  return cast("BenchmarkRunInput", data)


def compute_run_aggregates(
  raw_times: list[float],
  raw_costs: list[float],
  raw_gaps: list[float] | None,
  optimal_cost: float | None,
  repetitions: int,
  raw_initial_costs: list[float] | None = None,
  raw_generations: list[int] | None = None,
  raw_stop_reasons: list[str] | None = None,
) -> dict[str, Any]:
  """Compute aggregate metrics for a benchmark run, strictly using successful_runs as denominator.

  Following BUG-36:
  - If successful_runs = len(raw_times) > 0, means are calculated dividing by successful_runs.
  - If successful_runs == 0, returns graceful zero/null values according to column contracts
    (0.0 for NOT NULL columns, None for nullable columns).

  Args:
      raw_times: Execution times for successful runs.
      raw_costs: Best tour costs for successful runs.
      raw_gaps: Percentage gaps above optimal for successful runs (or None when unknown).
      optimal_cost: Known optimal solution cost (or None when unknown).
      repetitions: Total attempted repetitions.
      raw_initial_costs: Optional initial random tour costs.
      raw_generations: Optional generations completed.
      raw_stop_reasons: Optional stop reasons for completed runs.

  Returns:
      Dictionary of aggregate metrics conforming to BenchmarkRunInput.
  """
  n_successful = len(raw_times)
  has_gaps = raw_gaps is not None and optimal_cost is not None
  if n_successful == 0:
    return {
      "repetitions": repetitions,
      "successful_runs": 0,
      "mean_time": 0.0,
      "std_time": 0.0,
      "min_time": 0.0,
      "max_time": 0.0,
      "mean_cost": 0.0,
      "std_cost": 0.0,
      "best_cost": 0.0,
      "worst_cost": 0.0,
      "mean_gap": 0.0 if has_gaps else None,
      "std_gap": 0.0 if has_gaps else None,
      "best_gap": 0.0 if has_gaps else None,
      "mean_generations": 0.0,
      "std_generations": 0.0,
      "mean_initial_cost": None,
      "std_initial_cost": None,
      "mean_improvement": None,
      "raw_times": [],
      "raw_costs": [],
      "raw_gaps": [] if has_gaps else None,
    }

  times_arr = np.array(raw_times, dtype=np.float64)
  costs_arr = np.array(raw_costs, dtype=np.float64)

  # Sample std uses ddof=1 when n_successful > 1, else 0.0
  std_ddof = 1 if n_successful > 1 else 0

  res: dict[str, Any] = {
    "repetitions": repetitions,
    "successful_runs": n_successful,
    "mean_time": float(np.sum(times_arr) / n_successful),
    "std_time": float(np.std(times_arr, ddof=std_ddof)) if n_successful > 1 else 0.0,
    "min_time": float(np.min(times_arr)),
    "max_time": float(np.max(times_arr)),
    "mean_cost": float(np.sum(costs_arr) / n_successful),
    "std_cost": float(np.std(costs_arr, ddof=std_ddof)) if n_successful > 1 else 0.0,
    "best_cost": float(np.min(costs_arr)),
    "worst_cost": float(np.max(costs_arr)),
    "raw_times": raw_times,
    "raw_costs": raw_costs,
  }

  if has_gaps and raw_gaps is not None and len(raw_gaps) > 0:
    gaps_arr = np.array(raw_gaps, dtype=np.float64)
    res["mean_gap"] = float(np.sum(gaps_arr) / n_successful)
    res["std_gap"] = float(np.std(gaps_arr, ddof=std_ddof)) if n_successful > 1 else 0.0
    res["best_gap"] = float(np.min(gaps_arr))
    res["raw_gaps"] = raw_gaps
  else:
    res["mean_gap"] = None
    res["std_gap"] = None
    res["best_gap"] = None
    res["raw_gaps"] = None

  if raw_generations is not None and len(raw_generations) > 0:
    gen_arr = np.array(raw_generations, dtype=np.float64)
    res["mean_generations"] = float(np.sum(gen_arr) / n_successful)
    res["std_generations"] = float(np.std(gen_arr, ddof=std_ddof)) if n_successful > 1 else 0.0
    res["raw_generations"] = raw_generations
  else:
    res["mean_generations"] = 0.0
    res["std_generations"] = 0.0

  if raw_initial_costs is not None and len(raw_initial_costs) > 0:
    init_arr = np.array(raw_initial_costs, dtype=np.float64)
    res["mean_initial_cost"] = float(np.sum(init_arr) / n_successful)
    res["std_initial_cost"] = float(np.std(init_arr, ddof=std_ddof)) if n_successful > 1 else 0.0
    res["raw_initial_costs"] = raw_initial_costs
    improvements = (init_arr - costs_arr) / init_arr * 100.0
    res["mean_improvement"] = float(np.sum(improvements) / n_successful)
  else:
    res["mean_initial_cost"] = None
    res["std_initial_cost"] = None
    res["mean_improvement"] = None

  if raw_stop_reasons is not None:
    res["raw_stop_reasons"] = raw_stop_reasons

  return res


# ============================================================================
# Insert Operations
# ============================================================================


def _json_or_none(value: JsonPayload | None) -> str | None:
  """Return a DuckDB-ready JSON string, or `None` for an absent payload.

  DuckDB binds a Python `dict` parameter as a `MAP`, which cannot be inserted into a
  `JSON` column, so structured telemetry is serialised here exactly like
  `algorithm_config` and `strategies` are.
  """
  if value is None or isinstance(value, str):
    return value
  return json.dumps(dict(value))


def insert_benchmark_run(conn: duckdb.DuckDBPyConnection, data: BenchmarkRunInput) -> int:
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
            problem_type,
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
            feasible_runs,
            feasibility_rate,
            mean_vehicles,
            best_vehicles,
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
            mean_phase_timings,
            phase_share_pct,
            phase_share_cv_pct,
            raw_times,
            raw_costs,
            raw_gaps,
            raw_initial_costs,
            raw_generations,
            raw_stop_reasons,
            raw_is_feasible,
            raw_unpenalized_costs,
            raw_violations,
            raw_num_vehicles,
            algorithm_config,
            strategies,
            construction_heuristic
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?
        )
        RETURNING run_id
    """

  result = conn.execute(
    query,
    [
      data["problem_name"],
      data.get("problem_type") or "TSP",
      data["problem_size"],
      data.get("optimal_cost"),
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
      data.get("mean_cost"),
      data.get("std_cost"),
      data.get("best_cost"),
      data.get("worst_cost"),
      data.get("mean_initial_cost"),
      data.get("std_initial_cost"),
      data.get("feasible_runs"),
      data.get("feasibility_rate"),
      data.get("mean_vehicles"),
      data.get("best_vehicles"),
      data.get("mean_gap"),
      data.get("std_gap"),
      data.get("best_gap"),
      data.get("mean_improvement"),
      data["mean_generations"],
      data["std_generations"],
      data.get("min_generations"),
      data.get("max_generations"),
      data.get("mean_h2d_mb"),
      data.get("mean_d2h_mb"),
      data.get("total_transfer_mb"),
      data.get("mean_kernels"),
      _json_or_none(data.get("mean_phase_timings")),
      _json_or_none(data.get("phase_share_pct")),
      _json_or_none(data.get("phase_share_cv_pct")),
      data["raw_times"],
      data["raw_costs"],
      data.get("raw_gaps"),
      data.get("raw_initial_costs"),
      data.get("raw_generations"),
      data.get("raw_stop_reasons"),
      data.get("raw_is_feasible"),
      data.get("raw_unpenalized_costs"),
      data.get("raw_violations"),
      data.get("raw_num_vehicles"),
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


def get_algorithm_summary(conn: duckdb.DuckDBPyConnection, problem_name: str, algorithm: str) -> dict[str, Any] | None:
  """
  Get summary statistics for a specific algorithm on a problem.

  Args:
      conn: DuckDB connection
      problem_name: Problem name
      algorithm: Algorithm name

  Returns:
      Dictionary with summary stats or None if not found

  Example:
      >>> stats = get_algorithm_summary(conn, "berlin52", "MemeticGA")
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
  include_missing: bool = False,
  fill_missing: float | None = None,
) -> list[tuple[str, str, float | None, float | None]]:
  """
  Compute speedup of each algorithm vs baseline across all problems.

  Executes a single aggregated analytical SQL query with GROUP BY problem_name, algorithm
  and calculates speedup ratios in memory across all comparison algorithms.

  Args:
      conn: DuckDB connection
      baseline: Baseline algorithm (default 'CPU')
      comparisons: Optional list of algorithms to compare (defaults to all non-baseline algorithms)
      include_missing: If True, include problems where candidate algorithm data is missing
          or invalid with `fill_missing` (default False).
      fill_missing: Value to fill for missing or invalid speedups when include_missing is True.

  Returns:
      List of (problem_name, algorithm, speedup, baseline_time) tuples

  Example:
      >>> speedups = compute_speedup_matrix(conn, "CPU", ["MemeticGPU", "FullGPU"])
      >>> for prob, alg, speedup, base_time in speedups:
      ...   print(f"{prob}: {alg} is {speedup:.2f}x faster than CPU")
  """
  # Single aggregated analytical query extracting mean_time grouped by problem_name and algorithm
  query = f"""
        SELECT
            problem_name,
            algorithm,
            AVG(mean_time) AS mean_time
        FROM {TABLE_RUNS}
        WHERE is_test = FALSE
        GROUP BY problem_name, algorithm
  """
  rows = conn.execute(query).fetchall()

  # Group by problem
  data_by_problem: dict[str, dict[str, float]] = defaultdict(dict)
  available_algs: set[str] = set()
  for prob, alg, m_time in rows:
    if m_time is not None:
      data_by_problem[prob][alg] = float(m_time)
      available_algs.add(alg)

  if comparisons is None:
    comparisons = sorted(available_algs - {baseline})

  results: list[tuple[str, str, float | None, float | None]] = []

  for problem_name in sorted(data_by_problem.keys()):
    prob_data = data_by_problem[problem_name]
    if baseline not in prob_data:
      continue

    base_time_val = prob_data[baseline]
    if base_time_val <= 0 or math.isnan(base_time_val):
      continue

    for comp_algorithm in comparisons:
      if comp_algorithm not in prob_data:
        if include_missing:
          results.append((problem_name, comp_algorithm, fill_missing, base_time_val))
        continue

      comp_time_val = prob_data[comp_algorithm]
      if comp_time_val <= 0 or math.isnan(comp_time_val):
        if include_missing:
          results.append((problem_name, comp_algorithm, fill_missing, base_time_val))
        continue

      speedup = base_time_val / comp_time_val
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
