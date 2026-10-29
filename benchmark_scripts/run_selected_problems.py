#!/usr/bin/env python3
"""
Run Benchmark on Selected 10 TSPLIB Problems
============================================

Executes the modular benchmark pipeline across the 10 target TSPLIB instances
spanning all major edge weight formats (`ATT`, `EXPLICIT` [`UPPER_ROW`, `FULL_MATRIX`,
`LOWER_DIAG_ROW`], `EUC_2D`, and `GEO`).

Usage:
    python -m benchmark_scripts.run_selected_problems --repetitions 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmarking.runner.config_loader import (
  DEFAULT_SELECTED_TSP_FILES,
  build_problem_configs_from_tsp_files,
)
from benchmarking.runner.orchestration import run_comprehensive_benchmark


def setup_logging() -> None:
  """Configure standard INFO logging to stdout."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse CLI arguments for the selected problems benchmark script."""
  parser = argparse.ArgumentParser(
    description="Run the modular GPU-accelerated GA benchmark on the 10 selected TSPLIB problems.",
  )
  parser.add_argument(
    "--tsp-files",
    nargs="+",
    default=[str(p) for p in DEFAULT_SELECTED_TSP_FILES],
    help="Paths to .tsp problem files (defaults to the 10 selected TSPLIB instances).",
  )
  parser.add_argument(
    "--repetitions",
    type=int,
    default=3,
    help="Number of repetitions per (problem, algorithm) pair (default: 3).",
  )
  parser.add_argument(
    "--skip-cpu",
    action="store_true",
    default=False,
    help="Skip the CPU baseline variant.",
  )
  parser.add_argument(
    "--algorithms",
    nargs="+",
    default=None,
    help="Optional subset of algorithms to run (default: all configured in algorithms.json).",
  )
  parser.add_argument(
    "--cpu-size-threshold",
    type=int,
    default=None,
    help="Override CPU problem size threshold (default from benchmark.json: 100).",
  )
  parser.add_argument(
    "--incremental-runs",
    action="store_true",
    default=None,
    dest="incremental_runs",
    help="Append new repetitions to existing checkpoints instead of skipping completed runs.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="results_v3/selected_10",
    help="Directory to store checkpoints, problem_statistics, and tables (default: results_v3/selected_10).",
  )
  parser.add_argument(
    "--benchmark-name",
    type=str,
    default="selected_10_validation",
    help="Base filename for generated Markdown and LaTeX tables (default: selected_10_validation).",
  )
  parser.add_argument(
    "--test-mode",
    action="store_true",
    default=False,
    help="Unused compatibility flag for orchestration.",
  )
  parser.add_argument(
    "--local-search-stage",
    type=str,
    default=None,
    choices=["post_crossover", "post_mutation", "post_evaluation", "post_evolution", "none"],
    help="Override local search stage in GeneticAlgorithm (default: post_mutation).",
  )
  parser.add_argument(
    "--config-dir",
    type=str,
    default=None,
    metavar="DIR",
    help="Directory holding benchmark.json, algorithms.json and problems.json. Default: <repo root>/configs",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  """Parse the 10 selected .tsp problems and run the full benchmark pipeline."""
  setup_logging()
  args = parse_arguments(argv)

  problem_configs, tsp_map = build_problem_configs_from_tsp_files(args.tsp_files)
  logging.info("=" * 80)
  logging.info(f"Parsed {len(problem_configs)} TSPLIB problem files via tsplib95:")
  for cfg in problem_configs:
    logging.info(f"  - {cfg.name:10s} (n={cfg.size:4d}, optimal={cfg.optimal:8.1f}) <- {tsp_map[cfg.name]}")
  logging.info("=" * 80)

  results = run_comprehensive_benchmark(
    args,
    tsp_files=args.tsp_files,
    algorithms=args.algorithms,
    output_dir=args.output_dir,
    benchmark_name=args.benchmark_name,
  )

  logging.info(f"Completed benchmark for {len(results)}/{len(problem_configs)} problems.")
  logging.info(f"Results saved under: {Path(args.output_dir).resolve()}")
  return 0 if len(results) == len(problem_configs) else 1


if __name__ == "__main__":
  sys.exit(main())