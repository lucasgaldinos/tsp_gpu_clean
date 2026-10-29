#!/usr/bin/env python3
"""
Chapter 4 Validation Benchmark CLI
===================================

Command-line interface for GPU-accelerated genetic algorithm benchmarking.

All configuration is loaded from JSON files in `configs/` at the repository root
(override with `--config-dir`).

Usage:
    # Full benchmark (30 reps, all configured problems, all algorithms)
    python -m benchmark_scripts.run_chapter4_benchmark

    # Skip CPU variant (GPU-only comparison)
    python -m benchmark_scripts.run_chapter4_benchmark --skip-cpu

    # Test mode (2 problems only)
    python -m benchmark_scripts.run_chapter4_benchmark --test-mode

    # Custom repetitions
    python -m benchmark_scripts.run_chapter4_benchmark --repetitions 10

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmarking.runner.config_loader import default_config_dir
from benchmarking.runner.orchestration import run_comprehensive_benchmark


def setup_logging() -> None:
  """Configure logging for benchmark execution."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def parse_arguments() -> argparse.Namespace:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
    description="Chapter 4 Validation: GPU-Accelerated GA Benchmark",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Full benchmark (all algorithms, all problems)
  %(prog)s

  # GPU-only comparison (skip CPU)
  %(prog)s --skip-cpu

  # Quick test (2 problems only)
  %(prog)s --test-mode

  # Custom repetitions
  %(prog)s --repetitions 10

Configuration:
  Algorithms: configs/algorithms.json
  Problems: configs/problems.json
  Benchmark params: configs/benchmark.json
  (all overridable with --config-dir)

Output:
  Checkpoints: results_v3/checkpoints/
  Statistics: results_v3/problem_statistics/
  Tables: results_v3/tables/
        """,
  )

  parser.add_argument(
    "--skip-cpu",
    action="store_true",
    default=False,
    help="Skip CPU variant (compare GPU algorithms only). Default: False",
  )

  parser.add_argument(
    "--repetitions",
    type=int,
    default=30,
    metavar="N",
    help="Number of repetitions per algorithm. Default: 30",
  )

  parser.add_argument(
    "--incremental-runs",
    action="store_true",
    default=None,
    dest="incremental_runs",
    help=(
      "Enable incremental checkpoint mode: append new runs to existing checkpoints "
      "instead of skipping them. Overrides config file setting."
    ),
  )

  parser.add_argument(
    "--test-mode",
    action="store_true",
    default=False,
    help="Test mode: run on first 2 problems only. Default: False",
  )

  parser.add_argument(
    "--problems",
    "--tsp-files",
    nargs="+",
    default=None,
    dest="tsp_files",
    help="Optional list of .tsp file paths or problem names to run instead of problems.json.",
  )

  parser.add_argument(
    "--algorithms",
    nargs="+",
    default=None,
    help="Optional subset of algorithm names to run (e.g. CPU HybridNaive HybridOptimized FullGPU).",
  )

  parser.add_argument(
    "--output-dir",
    type=str,
    default="results_v3",
    help="Output root directory for checkpoints, problem_statistics, and tables. Default: results_v3",
  )

  parser.add_argument(
    "--config-dir",
    type=str,
    default=None,
    metavar="DIR",
    help="Directory holding benchmark.json, algorithms.json and problems.json. Default: <repo root>/configs",
  )

  parser.add_argument(
    "--version",
    action="version",
    version="Chapter 4 Validation Benchmark V2.0",
  )

  return parser.parse_args()


def validate_environment(config_dir: Path) -> bool:
  """Validate environment and dependencies."""
  required_configs = ["algorithms.json", "benchmark.json", "problems.json"]

  for config_file in required_configs:
    config_path = config_dir / config_file
    if not config_path.exists():
      logging.error(f"Missing configuration file: {config_path}")
      return False

  project_root = config_dir.parent
  db_path = project_root / "datasets" / "routing.duckdb"
  tsp_dir = project_root / "datasets" / "tsp"
  if not db_path.exists() and not tsp_dir.exists():
    logging.error(f"Missing database file ({db_path}) and TSPLIB fallback directory ({tsp_dir})")
    return False

  return True


def main() -> int:
  """Main entry point for benchmark CLI."""
  setup_logging()
  args = parse_arguments()

  logging.info("=" * 80)
  logging.info("Chapter 4 Validation Benchmark (Modular Architecture)")
  logging.info("=" * 80)
  logging.info("")

  if args.incremental_runs is not None:
    mode_str = "ENABLED" if args.incremental_runs else "DISABLED"
    logging.info(f"CLI Override: Incremental runs {mode_str}")
    logging.info("")

  config_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
  if not validate_environment(config_dir):
    logging.error("Environment validation failed. Cannot proceed.")
    return 1

  logging.info("Environment validated successfully")
  logging.info("")

  try:
    results = run_comprehensive_benchmark(args)

    logging.info("")
    logging.info("=" * 80)
    logging.info("BENCHMARK COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info(f"Total problems: {len(results)}")
    logging.info("")

    return 0

  except KeyboardInterrupt:
    logging.info("")
    logging.warning("Benchmark interrupted by user (Ctrl+C)")
    logging.info("Progress has been saved. Rerun to resume from checkpoints.")
    return 130

  except Exception as e:
    logging.error("")
    logging.error("=" * 80)
    logging.error("BENCHMARK FAILED")
    logging.error("=" * 80)
    logging.error(f"Error: {e}")
    logging.exception("Full traceback:")
    return 1


if __name__ == "__main__":
  sys.exit(main())
