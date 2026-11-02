#!/usr/bin/env python3
"""
Import Checkpoint JSON files into DuckDB Database
=================================================

Pure Python experiment script importing checkpoint JSON files directly into DuckDB.
For CLI usage with command-line arguments, use `tsp-import-checkpoints` instead.

Usage:
    python -m benchmark_scripts.import_checkpoints
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarking.database.import_checkpoints import import_checkpoints

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
INPUT_PATH: Path = PROJECT_ROOT / "results_v3" / "checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v3"


def main() -> int:
  """Run checkpoint import with configured paths."""
  try:
    import_checkpoints(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)
    return 0
  except KeyboardInterrupt:
    print("\n\n⚠️  Import interrupted by user", file=sys.stderr)
    return 130
  except Exception as e:
    print(f"\n\n❌ Fatal error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
