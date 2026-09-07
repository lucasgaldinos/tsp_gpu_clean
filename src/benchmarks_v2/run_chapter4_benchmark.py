#!/usr/bin/env python3
"""
Chapter 4 Validation Benchmark CLI (V2)
========================================

Command-line interface for GPU-accelerated genetic algorithm benchmarking.

This script uses the modular v2 architecture for clean, maintainable execution.
All configuration is loaded from JSON files in benchmarks_v2/configs/.

Usage:
    # Full benchmark (30 reps, all 38 problems, all algorithms)
    python code/benchmarks_v2/run_chapter4_benchmark.py

    # Skip CPU variant (GPU-only comparison)
    python code/benchmarks_v2/run_chapter4_benchmark.py --skip-cpu

    # Test mode (2 problems only)
    python code/benchmarks_v2/run_chapter4_benchmark.py --test-mode

    # Custom repetitions
    python code/benchmarks_v2/run_chapter4_benchmark.py --repetitions 10

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import argparse
import logging
import sys
from pathlib import Path

# Add code directory to path (parent of benchmarks_v2)
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from src.benchmarking_v2.orchestration import run_comprehensive_benchmark


def setup_logging() -> None:
    """
    Configure logging for benchmark execution.

    Sets up INFO level logging with timestamp and level formatting.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace with parsed arguments:
            - skip_cpu: bool (skip CPU algorithm)
            - repetitions: int (number of runs per algorithm)
            - test_mode: bool (run on 2 problems only)
    """
    parser = argparse.ArgumentParser(
        description="Chapter 4 Validation: GPU-Accelerated GA Benchmark (V2 Modular Architecture)",
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
  Algorithms: benchmarks_v2/configs/algorithms.json
  Problems: benchmarks_v2/configs/problems.json
  Benchmark params: benchmarks_v2/configs/benchmark.json

Output:
  Checkpoints: results/benchmark_results/checkpoints/
  Statistics: results/benchmark_results/problem_stats/
  Tables: results/tables/
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
        default=None,  # None means use config file value
        dest="incremental_runs",
        help="Enable incremental checkpoint mode: append new runs to existing checkpoints instead of skipping them. Overrides config file setting.",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Test mode: run on first 2 problems only. Default: False (runs all 38 problems)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Chapter 4 Validation Benchmark V2.0 (Modular Architecture)",
    )

    return parser.parse_args()


def validate_environment() -> bool:
    """
    Validate environment and dependencies.

    Checks:
    - Configuration files exist
    - Dataset database exists
    - Required directories can be created

    Returns:
        True if environment is valid, False otherwise
    """
    # Check config files
    configs_dir = Path("code/benchmarks_v2/configs")
    required_configs = ["algorithms.json", "benchmark.json", "problems.json"]

    for config_file in required_configs:
        config_path = configs_dir / config_file
        if not config_path.exists():
            logging.error(f"Missing configuration file: {config_path}")
            return False

    # Check database
    db_path = Path("datasets/routing.duckdb")
    if not db_path.exists():
        logging.error(f"Missing database file: {db_path}")
        logging.error("Run dataset loading script to create database")
        return False

    # Check output directories can be created
    output_dirs = [
        Path("results/benchmark_results/checkpoints"),
        Path("results/benchmark_results/problem_stats"),
        Path("results/tables"),
    ]

    try:
        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Cannot create output directories: {e}")
        return False

    return True


def main() -> int:
    """
    Main entry point for benchmark CLI.

    Returns:
        Exit code: 0 for success, 1 for error
    """
    # Setup
    setup_logging()
    args = parse_arguments()

    # Log startup
    logging.info("=" * 80)
    logging.info("Chapter 4 Validation Benchmark V2 (Modular Architecture)")
    logging.info("=" * 80)
    logging.info("")

    # Log incremental mode if enabled via CLI
    if args.incremental_runs is not None:
        mode_str = "ENABLED" if args.incremental_runs else "DISABLED"
        logging.info(f"CLI Override: Incremental runs {mode_str}")
        logging.info("")

    # Validate environment
    if not validate_environment():
        logging.error("Environment validation failed. Cannot proceed.")
        return 1

    logging.info("Environment validated successfully")
    logging.info("")

    # Run benchmark
    try:
        results = run_comprehensive_benchmark(args)

        logging.info("")
        logging.info("=" * 80)
        logging.info("BENCHMARK COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        logging.info(f"Total problems: {len(results)}")
        logging.info("Output directories:")
        logging.info("  - Checkpoints: results_v2/checkpoints/")
        logging.info("  - Statistics: results_v2/problem_statistics/")
        logging.info("  - Tables: results_v2/tables/")
        logging.info("")

        return 0

    except KeyboardInterrupt:
        logging.info("")
        logging.warning("Benchmark interrupted by user (Ctrl+C)")
        logging.info("Progress has been saved. Rerun to resume from checkpoints.")
        return 130  # Standard exit code for SIGINT

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
