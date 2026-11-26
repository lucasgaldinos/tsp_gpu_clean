#!/usr/bin/env python3
"""
Run Benchmark on Selected 10 TSPLIB Problems
============================================

Executes the modular benchmark pipeline across the 10 target TSPLIB instances
spanning all major edge weight formats (`ATT`, `EXPLICIT`, `EUC_2D`, `GEO`).

Usage:
    uv run python benchmark_scripts/experiments-and-examples/chapter4/run_selected_problems.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.runner.orchestration import run_comprehensive_benchmark

# Target 10 TSPLIB benchmark instances (48 <= n < 400; micro-instances n < 48 removed)
SELECTED_PROBLEMS: tuple[str, ...] = (
  "att48",
  "berlin52",
  "st70",
  "eil76",
  "gr96",
  "kroA100",
  "ch150",
  "brg180",
  "pr264",
  "lin318",
)

# Experiment parameters
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG_DIR: Path = SCRIPT_DIR / "configs"
DB_PATH: Path = PROJECT_ROOT / "datasets" / "tsp.duckdb"
REPETITIONS: int = 5
CPU_SIZE_THRESHOLD: int = 400
BASELINE_ALGORITHM: str = "CPU"
SKIP_CPU: bool = False
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v3" / "selected_10"
BENCHMARK_NAME: str = "selected_10_validation"


def setup_logging() -> Path:
  """Configure dual-target console and file logging for benchmark execution."""
  return setup_benchmark_logging(OUTPUT_DIR, BENCHMARK_NAME)


def main() -> int:
  """Execute the selected 10 problems benchmark."""
  setup_logging()
  logging.info("Starting Selected 10 TSPLIB Benchmark (n < 400, CPU uncapped, baseline=CPU, 5 repetitions)...")

  run_comprehensive_benchmark(
    config_dir=CONFIG_DIR,
    output_dir=OUTPUT_DIR,
    problem_names=SELECTED_PROBLEMS,
    db_path=DB_PATH,
    repetitions=REPETITIONS,
    cpu_size_threshold=CPU_SIZE_THRESHOLD,
    skip_cpu=SKIP_CPU,
    baseline_algorithm=BASELINE_ALGORITHM,
    benchmark_name=BENCHMARK_NAME,
    incremental_runs=True,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
