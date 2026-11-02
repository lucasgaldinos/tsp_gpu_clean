#!/usr/bin/env python3
"""
Validate Database Consistency Against Checkpoint JSON Files
===========================================================

Pure Python experiment script comparing benchmark databases and verifying
consistency against raw JSON checkpoint records.
For CLI usage with command-line arguments, use `tsp-validate-db` instead.

Usage:
    python -m benchmark_scripts.validate_consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarking.database.validate_consistency import (
  compare_databases,
  validate_against_checkpoints,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DB1_PATH: Path = PROJECT_ROOT / "results_v3" / "results.duckdb"
DB2_PATH: Path = PROJECT_ROOT / "results_v3" / "backup_results.duckdb"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "results_v3" / "checkpoints"


def main() -> int:
  """Run database consistency validation."""
  if not DB1_PATH.exists() or not CHECKPOINTS_DIR.exists():
    print(f"Skipping validation: {DB1_PATH} or {CHECKPOINTS_DIR} does not exist.", file=sys.stderr)
    return 0

  if DB2_PATH.exists():
    identical, db_diffs = compare_databases(str(DB1_PATH), str(DB2_PATH))
    if not identical:
      print("\nDatabase differences found:")
      for diff in db_diffs:
        print(f"  - {diff}")
      return 1

  all_valid, mismatch_df = validate_against_checkpoints(str(DB1_PATH), str(CHECKPOINTS_DIR))
  if not all_valid:
    print("\nMismatches found:")
    print(mismatch_df.to_string(index=False))
    return 1

  print("✅ Validation successful: all records consistent.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
