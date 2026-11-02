#!/usr/bin/env python3
"""
Chapter 4 Validation Benchmark Runner
=====================================

Executes the GPU-accelerated GA validation benchmark across all configured
problems and algorithms using configs from `configs/`.

Usage:
    python -m benchmark_scripts.run_chapter4_benchmark
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from benchmarking.runner.orchestration import run_comprehensive_benchmark

# Default experiment parameters
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DB_PATH: Path = PROJECT_ROOT / "datasets" / "tsp.duckdb"
FALLBACK_TSP_DIR: Path = PROJECT_ROOT / "datasets" / "tsp"
REPETITIONS: int = 30
SKIP_CPU: bool = False
TEST_MODE: bool = False
BENCHMARK_NAME: str = "chapter4_validation"
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v4"


def setup_logging() -> None:
  """Configure standard INFO logging for benchmark execution."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def main() -> int:
  """Execute the Chapter 4 validation benchmark."""
  setup_logging()
  logging.info("Starting Chapter 4 Validation Benchmark...")

  run_comprehensive_benchmark(
    config_dir=CONFIG_DIR,
    output_dir=OUTPUT_DIR,
    db_path=DB_PATH,
    fallback_tsp_dir=FALLBACK_TSP_DIR,
    repetitions=REPETITIONS,
    skip_cpu=SKIP_CPU,
    test_mode=TEST_MODE,
    benchmark_name=BENCHMARK_NAME,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
