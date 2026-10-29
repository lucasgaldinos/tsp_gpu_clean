#!/usr/bin/env python3
"""
CLI wrapper for importing benchmark checkpoint JSON files into DuckDB.

Usage:
    python -m benchmark_scripts.import_checkpoints --input ./results_v3/checkpoints/
"""

import sys

from benchmarking.database.import_checkpoints import import_checkpoints, parse_arguments


def main() -> int:
  try:
    args = parse_arguments()
    import_checkpoints(input_path=args.input, output_dir=args.output_dir)
    return 0
  except KeyboardInterrupt:
    print("\n\n⚠️  Import interrupted by user", file=sys.stderr)
    return 130
  except Exception as e:
    print(f"\n\n❌ Fatal error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())

