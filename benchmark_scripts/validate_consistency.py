#!/usr/bin/env python3
"""
CLI wrapper for validating database consistency against checkpoint JSON files.

Usage:
    python -m benchmark_scripts.validate_consistency \\
        --db1 results.duckdb \\
        --db2 my_notes/results.duckdb \\
        --checkpoints results_v3/checkpoints/
"""

from benchmarking.database.validate_consistency import main

if __name__ == "__main__":
  main()
