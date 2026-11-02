"""CLI entrypoints for checkpoint import and database consistency validation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarking.database.import_checkpoints import import_checkpoints
from benchmarking.database.validate_consistency import (
  compare_databases,
  validate_against_checkpoints,
)


def build_import_parser() -> argparse.ArgumentParser:
  """Build the argument parser for importing checkpoint JSONs into DuckDB."""
  parser = argparse.ArgumentParser(
    prog="tsp-import-checkpoints",
    description="Import benchmark checkpoint JSON files to DuckDB database",
  )
  parser.add_argument(
    "--input",
    "-i",
    type=Path,
    required=True,
    help="Input directory (bulk scan for *.json) or single JSON file path",
  )
  parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    required=True,
    help="Output directory for results.duckdb",
  )
  return parser


def import_main(argv: Sequence[str] | None = None) -> int:
  """Execute the import checkpoints CLI command."""
  parser = build_import_parser()
  args = parser.parse_args(argv)

  if not args.input.exists():
    print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
    return 1

  try:
    import_checkpoints(input_path=args.input, output_dir=args.output_dir)
    return 0
  except Exception as e:
    print(f"Error importing checkpoints: {e}", file=sys.stderr)
    return 1


def build_validate_parser() -> argparse.ArgumentParser:
  """Build the argument parser for validating database consistency."""
  parser = argparse.ArgumentParser(
    prog="tsp-validate-db",
    description="Validate consistency between DuckDB databases and checkpoint JSON files",
  )
  parser.add_argument("--db1", type=Path, required=True, help="First database path")
  parser.add_argument("--db2", type=Path, required=True, help="Second database path")
  parser.add_argument("--checkpoints", type=Path, required=True, help="Checkpoint directory")
  return parser


def validate_main(argv: Sequence[str] | None = None) -> int:
  """Execute the database consistency validation CLI command."""
  parser = build_validate_parser()
  args = parser.parse_args(argv)

  if not args.db1.exists():
    print(f"❌ Database 1 not found: {args.db1}", file=sys.stderr)
    return 1
  if not args.db2.exists():
    print(f"❌ Database 2 not found: {args.db2}", file=sys.stderr)
    return 1
  if not args.checkpoints.exists():
    print(f"❌ Checkpoint directory not found: {args.checkpoints}", file=sys.stderr)
    return 1

  identical, db_diffs = compare_databases(str(args.db1), str(args.db2))
  if not identical:
    print("\nDatabase Differences:")
    for diff in db_diffs:
      print(f"  - {diff}")
    return 1

  all_valid, mismatch_df = validate_against_checkpoints(str(args.db1), str(args.checkpoints))
  if not all_valid:
    print("\nMismatches found:")
    print(mismatch_df.to_string(index=False))
    return 1

  print("✅ Validation passed: databases and checkpoints are fully consistent.")
  return 0


if __name__ == "__main__":
  sys.exit(import_main())

