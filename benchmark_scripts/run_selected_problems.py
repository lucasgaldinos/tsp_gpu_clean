#!/usr/bin/env python3
"""
Run Benchmark on Selected 10 TSPLIB Problems
============================================

Executes the modular benchmark pipeline across the 10 target TSPLIB instances
spanning all major edge weight formats (`ATT`, `EXPLICIT`, `EUC_2D`, `GEO`).

Usage:
    python -m benchmark_scripts.run_selected_problems
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from benchmarking.runner.orchestration import run_comprehensive_benchmark

# Target 10 TSPLIB benchmark instances
SELECTED_PROBLEMS: tuple[str, ...] = (
  "att48",
  "bayg29",
  "bays29",
  "berlin52",
  "brg180",
  "gr17",
  "gr21",
  "gr24",
  "gr48",
  "gr96",
)

# Experiment parameters
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DB_PATH: Path = PROJECT_ROOT / "datasets" / "tsp.duckdb"
REPETITIONS: int = 3
SKIP_CPU: bool = False
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v3" / "selected_10"
BENCHMARK_NAME: str = "selected_10_validation"


def setup_logging() -> None:
  """Configure standard INFO logging for benchmark execution."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def main() -> int:
  """Execute the selected 10 problems benchmark."""
  setup_logging()
  logging.info("Starting Selected 10 TSPLIB Benchmark...")

  run_comprehensive_benchmark(
    config_dir=CONFIG_DIR,
    output_dir=OUTPUT_DIR,
    problem_names=SELECTED_PROBLEMS,
    db_path=DB_PATH,
    repetitions=REPETITIONS,
    skip_cpu=SKIP_CPU,
    benchmark_name=BENCHMARK_NAME,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
