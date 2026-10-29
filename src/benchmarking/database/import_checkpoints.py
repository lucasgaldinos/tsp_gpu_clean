"""
Import Checkpoint JSONs to DuckDB Database
==========================================

This script imports all benchmark checkpoint JSON files into the results database.

Usage:
------
    # Default: Create results.duckdb in current working directory
    python -m benchmarking.database.import_checkpoints --input ./checkpoints/

    # Specify output directory
    python -m benchmarking.database.import_checkpoints --input ./checkpoints/ --output-dir /path/to/results/
    python -m benchmarking.database.import_checkpoints -i ./checkpoints/ -o ./my_results/

Features:
---------
- CLI arguments for input and output directory specification
- Handles both old (2 reps) and new (15/30 reps) formats
- Validates data before insertion
- Reports progress and statistics
- Skips duplicates based on (problem_name, algorithm, timestamp)

Author: GPU Accelerated Routing Optimization Team
Date: 2025-11-23
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import duckdb

from benchmarking.database.db_operations import (
  create_database,
  insert_benchmark_run,
  validate_benchmark_data,
)
from benchmarking.database.schema import COUNT_RUNS, TABLE_RUNS

# ============================================================================
# CLI Argument Parsing
# ============================================================================


def parse_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments.

  Returns:
      Parsed arguments namespace
  """
  parser = argparse.ArgumentParser(
    description="Import benchmark checkpoint JSON files to DuckDB database",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Import from directory to current working directory
  python -m benchmarking.database.import_checkpoints --input ./checkpoints/

  # Import to custom output location
  python -m benchmarking.database.import_checkpoints --input ./checkpoints/ --output-dir ./results/
  python -m benchmarking.database.import_checkpoints -i ./checkpoints/ -o ./my_notes/

  # Import single file
  python -m benchmarking.database.import_checkpoints -i ./checkpoints/berlin52_CPU.json
        """,
  )

  parser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Input directory (bulk scan for *.json) or single JSON file path",
  )

  parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default=None,
    help="Output directory for results.duckdb (default: current working directory)",
  )

  return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================


def resolve_output_path(output_dir: str | None) -> Path:
  """
  Resolve output directory path.

  Args:
      output_dir: User-specified output directory or None

  Returns:
      Absolute path to output directory
  """
  if output_dir is None:
    return Path.cwd()
  path = Path(output_dir).resolve()
  path.mkdir(parents=True, exist_ok=True)
  return path


def resolve_input_paths(input_arg: str) -> list[Path]:
  """
  Resolve input argument to list of checkpoint JSON files.

  Args:
      input_arg: Path to directory (bulk scan) or single JSON file

  Returns:
      List of checkpoint JSON file paths

  Raises:
      FileNotFoundError: If input path doesn't exist
      ValueError: If no JSON files found in directory
  """
  input_path = Path(input_arg).resolve()

  if not input_path.exists():
    raise FileNotFoundError(f"Input path does not exist: {input_path}")

  if input_path.is_file():
    if input_path.suffix != ".json":
      raise ValueError(f"Input file must be JSON: {input_path}")
    return [input_path]

  if input_path.is_dir():
    json_files = list(input_path.glob("*.json"))
    if not json_files:
      raise ValueError(f"No JSON files found in directory: {input_path}")
    return sorted(json_files)

  raise ValueError(f"Invalid input path: {input_path}")


def load_checkpoint(filepath: Path) -> dict[str, Any]:
  """
  Load and parse checkpoint JSON file.

  Args:
      filepath: Path to checkpoint JSON file

  Returns:
      Dictionary containing checkpoint data
  """
  with open(filepath) as f:
    data = json.load(f)

  return cast("dict[str, Any]", data)


def check_duplicate(conn: duckdb.DuckDBPyConnection, data: dict[str, Any]) -> bool:
  """Check if benchmark run already exists in database."""
  result = conn.execute(
    f"""
        SELECT COUNT(*)
        FROM {TABLE_RUNS}
        WHERE problem_name = ?
          AND algorithm = ?
          AND timestamp = ?
    """,
    [data["problem_name"], data["algorithm"], data["timestamp"]],
  ).fetchone()

  if result is None:
    return False
  return bool(result[0] > 0)


# ============================================================================
# Main Import Function
# ============================================================================


def import_checkpoints(input_path: str, output_dir: str | None = None) -> None:
  """
  Main import function.

  Args:
      input_path: Input directory or JSON file path
      output_dir: Directory for results.duckdb (default: current working directory)
  """
  print("=" * 80)
  print("Importing Checkpoint JSONs to DuckDB Database")
  print("=" * 80)
  print()

  checkpoint_files = resolve_input_paths(input_path)
  output_path = resolve_output_path(output_dir)
  db_path = output_path / "results.duckdb"

  print(f"📥 Input: {Path(input_path).resolve()}")
  print(f"📂 Output directory: {output_path}")
  print(f"💾 Database path: {db_path}")
  print(f"📊 Found {len(checkpoint_files)} checkpoint file(s)")
  print()

  if not checkpoint_files:
    print("❌ No checkpoint files found!")
    return

  print("🗄️  Creating/connecting to database...")
  conn = create_database(str(db_path))
  print(f"   ✓ Database ready: {db_path}")
  print()

  print("🚀 Importing benchmark runs...")

  stats = {
    "total": len(checkpoint_files),
    "success": 0,
    "duplicates": 0,
    "validation_errors": 0,
    "other_errors": 0,
  }

  for filepath in checkpoint_files:
    try:
      data = load_checkpoint(filepath)

      if check_duplicate(conn, data):
        stats["duplicates"] += 1
        print(f"   ⊘ {filepath.name} (duplicate)")
        continue

      record = validate_benchmark_data(data)
      run_id = insert_benchmark_run(conn, record)
      stats["success"] += 1

      problem = data["problem_name"]
      algo = data["algorithm"]
      reps = data["repetitions"]
      is_test = "test" if reps <= 5 else "prod"

      print(f"   ✓ {problem:12s} {algo:16s} ({reps:2d} reps, {is_test}) -> run_id={run_id}")

    except ValueError as e:
      stats["validation_errors"] += 1
      print(f"   ✗ {filepath.name}: Validation error - {e}")

    except Exception as e:
      stats["other_errors"] += 1
      print(f"   ✗ {filepath.name}: {e}")

  print()
  print("=" * 80)
  print("Import Summary")
  print("=" * 80)
  print(f"Total files:          {stats['total']}")
  print(f"✓ Imported:           {stats['success']}")
  print(f"⊘ Duplicates:         {stats['duplicates']}")
  print(f"✗ Validation errors:  {stats['validation_errors']}")
  print(f"✗ Other errors:       {stats['other_errors']}")
  print()

  # Query statistics
  print("📊 Database Statistics")
  print("=" * 80)

  count_row = conn.execute(COUNT_RUNS).fetchone()
  total_runs = count_row[0] if count_row else 0
  print(f"Total benchmark runs: {total_runs}")

  by_algorithm = conn.execute(f"""
        SELECT algorithm, COUNT(*) as count
        FROM {TABLE_RUNS}
        GROUP BY algorithm
        ORDER BY count DESC
    """).fetchall()

  print("\nBy Algorithm:")
  for algo, count in by_algorithm:
    print(f"  {algo:20s}: {count:3d} runs")

  test_vs_prod = conn.execute(f"""
        SELECT is_test, COUNT(*) as count
        FROM {TABLE_RUNS}
        GROUP BY is_test
    """).fetchall()

  print("\nBy Type:")
  for is_test, count in test_vs_prod:
    run_type = "Test runs" if is_test else "Production runs"
    print(f"  {run_type:20s}: {count:3d} runs")

  prob_row = conn.execute(f"""
        SELECT COUNT(DISTINCT problem_name)
        FROM {TABLE_RUNS}
    """).fetchone()
  unique_problems = prob_row[0] if prob_row else 0
  print(f"\nUnique problems: {unique_problems}")

  cpu_row = conn.execute(f"""
        SELECT COUNT(DISTINCT problem_name)
        FROM {TABLE_RUNS}
        WHERE algorithm = 'CPU'
    """).fetchone()
  cpu_problems = cpu_row[0] if cpu_row else 0
  print(f"Problems with CPU data: {cpu_problems}")

  conn.close()
  print()
  print("✅ Import completed successfully!")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
  try:
    args = parse_arguments()
    import_checkpoints(input_path=args.input, output_dir=args.output_dir)
  except KeyboardInterrupt:
    print("\n\n⚠️  Import interrupted by user")
    sys.exit(1)
  except Exception as e:
    print(f"\n\n❌ Fatal error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
