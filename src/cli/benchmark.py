"""CLI entrypoint for running the comprehensive GA routing benchmark."""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.runner.orchestration import run_comprehensive_benchmark
from protocols.types import LocalSearchStage

if TYPE_CHECKING:
  from benchmarking.runner.config_loader import AlgorithmConfig, GAParams, ProblemConfig


def parse_strategy_param(param_str: str) -> tuple[str, str, Any]:
  """
  Parse a strategy parameter override in the format SLOT.KEY=VALUE.

  Values are decoded via json.loads with fallback to raw string.
  """
  if "=" not in param_str:
    raise ValueError(
      f"Invalid strategy param '{param_str}'. Expected format: SLOT.KEY=VALUE (e.g. local_search.max_iterations=20)"
    )
  target, val_str = param_str.split("=", 1)
  if target.count(".") != 1:
    raise ValueError(
      f"Invalid strategy param '{param_str}'. Target must be SLOT.KEY (e.g. local_search.max_iterations=20)"
    )
  slot, key = target.split(".", 1)
  slot = slot.strip()
  key = key.strip()
  if not slot or not key:
    raise ValueError(
      f"Invalid strategy param '{param_str}'. "
      "Target must be SLOT.KEY with non-empty SLOT and KEY (e.g. local_search.max_iterations=20)"
    )
  try:
    val = json.loads(val_str)
  except (json.JSONDecodeError, ValueError):
    val = val_str
  return slot, key, val


def build_parser(*, epilog: str | None = None) -> argparse.ArgumentParser:
  """Build the canonical argument parser for benchmarking."""
  parser = argparse.ArgumentParser(
    prog="tsp-benchmark",
    description="Run GPU/CPU benchmarks on TSP instances with statistical evaluation.",
    epilog=epilog,
    formatter_class=argparse.RawDescriptionHelpFormatter if epilog else argparse.HelpFormatter,
  )
  parser.add_argument(
    "--config-dir",
    type=Path,
    required=True,
    help="Directory containing benchmark configuration JSON files (algorithms, benchmark, problems)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory to write checkpoints, statistics, and tables",
  )
  parser.add_argument(
    "--db-path",
    type=Path,
    default=None,
    help="Path to DuckDB database containing TSP problems",
  )
  parser.add_argument(
    "--fallback-tsp-dir",
    type=Path,
    default=None,
    help="Directory of .tsp files to fall back to if not found in database",
  )
  parser.add_argument(
    "--tsp-files",
    nargs="+",
    type=Path,
    default=None,
    help="Explicit list of .tsp file paths to benchmark",
  )
  parser.add_argument(
    "--problems",
    nargs="+",
    type=str,
    default=None,
    help="Specific problem names to benchmark",
  )
  parser.add_argument(
    "--algorithms",
    nargs="+",
    type=str,
    default=None,
    help="Algorithms to run (defaults to all configured in algorithms.json)",
  )
  parser.add_argument(
    "-r",
    "--repetitions",
    type=int,
    default=None,
    help="Number of independent repetitions per problem/algorithm",
  )
  parser.add_argument(
    "--benchmark-name",
    type=str,
    default="chapter4_validation",
    help="Name identifier for this benchmark run (used in table captions)",
  )
  parser.add_argument(
    "--local-search-stage",
    type=str,
    default=None,
    choices=get_args(LocalSearchStage.__value__),
    help="Override local search stage ('post_crossover', 'post_mutation', 'post_evaluation', 'post_evolution', 'none')",
  )
  parser.add_argument(
    "--skip-cpu",
    action="store_true",
    help="Skip CPU algorithms regardless of problem size",
  )
  parser.add_argument(
    "--incremental-runs",
    "--incremental",
    action=argparse.BooleanOptionalAction,
    default=None,
    dest="incremental_runs",
    help=(
      "Append a full block of --repetitions new runs to existing checkpoints "
      "(use --no-incremental / --no-incremental-runs to complete only missing repetitions "
      "up to --repetitions or skip already-complete checkpoints)"
    ),
  )
  parser.add_argument(
    "--cpu-size-threshold",
    type=int,
    default=None,
    help="City count threshold above which CPU algorithms are skipped (default: None, run on all instances)",
  )
  parser.add_argument(
    "--baseline-algorithm",
    "--baseline",
    type=str,
    default=None,
    dest="baseline_algorithm",
    help="Baseline algorithm name for speedup calculations (default: first algorithm in execution order)",
  )
  parser.add_argument(
    "--patience",
    type=int,
    default=None,
    help="Early stopping stagnation window in generations (default: adaptive 2*sqrt(n))",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Base random seed for benchmark runs",
  )
  parser.add_argument(
    "--population-size",
    type=int,
    default=None,
    help="Override population size for all algorithms",
  )
  parser.add_argument(
    "--mutation-rate",
    type=float,
    default=None,
    help="Override mutation rate for all algorithms",
  )
  parser.add_argument(
    "--strategy-param",
    action="append",
    default=None,
    metavar="SLOT.KEY=VALUE",
    help="Strategy parameter override in format SLOT.KEY=VALUE (repeatable, parsed via json.loads with str fallback)",
  )
  parser.add_argument(
    "--log-level",
    type=str.upper,
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
    help="Set logging verbosity level (default: INFO)",
  )
  parser.add_argument(
    "--test-mode",
    action="store_true",
    help="Run in test mode (fewer problems/repetitions for quick verification)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate environment, configuration contracts, and checkpoints without running",
  )
  return parser


def main(
  argv: Sequence[str] | None = None,
  *,
  preflight_validator: Callable[
    [Mapping[str, "AlgorithmConfig"], "GAParams", Sequence["ProblemConfig"]],
    Sequence[str],
  ]
  | None = None,
  on_complete: Callable[[argparse.Namespace, dict[str, dict[str, dict[str, Any]]]], None] | None = None,
  parser_epilog: str | None = None,
) -> int:
  """Execute the benchmark CLI command."""
  parser = build_parser(epilog=parser_epilog)
  args = parser.parse_args(argv)

  level = getattr(logging, args.log_level.upper(), logging.INFO)
  setup_benchmark_logging(args.output_dir, args.benchmark_name, log_level=level)

  strategy_overrides: dict[str, dict[str, Any]] = {}
  if args.strategy_param:
    for item in args.strategy_param:
      try:
        slot, key, val = parse_strategy_param(item)
        strategy_overrides.setdefault(slot, {})[key] = val
      except ValueError as e:
        logging.error(f"Failed to parse strategy param: {e}")
        return 2

  try:
    results = run_comprehensive_benchmark(
      config_dir=args.config_dir,
      output_dir=args.output_dir,
      db_path=args.db_path,
      fallback_tsp_dir=args.fallback_tsp_dir,
      tsp_files=args.tsp_files,
      problem_names=args.problems,
      algorithms=args.algorithms,
      repetitions=args.repetitions,
      benchmark_name=args.benchmark_name,
      local_search_stage=args.local_search_stage,
      skip_cpu=args.skip_cpu,
      incremental_runs=args.incremental_runs,
      cpu_size_threshold=args.cpu_size_threshold,
      baseline_algorithm=args.baseline_algorithm,
      test_mode=args.test_mode,
      patience=args.patience,
      seed=args.seed,
      population_size=args.population_size,
      mutation_rate=args.mutation_rate,
      strategy_overrides=strategy_overrides or None,
      dry_run=args.dry_run,
      preflight_validator=preflight_validator,
    )
    if not args.dry_run and on_complete is not None:
      on_complete(args, results)
    return 0
  except ValueError as e:
    logging.error(f"Benchmark validation failed: {e}")
    return 2
  except Exception as e:
    logging.exception(f"Benchmark execution failed: {e}")
    return 1


if __name__ == "__main__":
  sys.exit(main())
