"""CLI entrypoint for running the modular GA routing benchmark.

Executes collection-only by default (checkpoints JSON + per-problem statistics).
Pass ``--reports all`` (or a subset of ``markdown``, ``latex``, ``macros``, ``cd``,
``plots``) to emit reports inline, or run ``tsp-report`` separately on the saved
checkpoints directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.reporting.style import ReportStyle, load_report_style, parse_report_artifacts
from benchmarking.runner.orchestration import run_comprehensive_benchmark
from protocols.types import LocalSearchStage

if TYPE_CHECKING:
  from collections.abc import Callable, Mapping, Sequence

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
    default=None,
    help=(
      "Optional directory containing benchmark configuration JSON files "
      "(algorithms.json, and optionally benchmark.json and problems.json)"
    ),
  )
  parser.add_argument(
    "--algorithms-config",
    type=Path,
    default=None,
    help="Direct path to an algorithms JSON config file (required unless --config-dir is provided)",
  )
  parser.add_argument(
    "--benchmark-config",
    type=Path,
    default=None,
    help="Optional direct path to a benchmark execution JSON config file (overrides defaults)",
  )
  parser.add_argument(
    "--problems-config",
    type=Path,
    default=None,
    help=(
      "Optional direct path to a problems catalogue JSON file (auto-discovered from --tsp-dir/--tsp-files if omitted)"
    ),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory to write checkpoints, per-problem statistics, and optional reports",
  )
  parser.add_argument(
    "--tsp-dir",
    type=Path,
    default=None,
    help="Directory containing TSPLIB .tsp files (primary input without DuckDB; also used as fallback)",
  )
  parser.add_argument(
    "--db-path",
    type=Path,
    default=None,
    help="Optional path to DuckDB database containing pre-ingested TSP problems",
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
    help="Algorithms to run (defaults to all configured in algorithms JSON)",
  )
  parser.add_argument(
    "-r",
    "--repetitions",
    type=int,
    default=None,
    help="Number of independent repetitions per problem/algorithm (default: 10 if no benchmark.json)",
  )
  parser.add_argument(
    "--max-generations",
    type=int,
    default=None,
    help="Fixed generation budget per repetition (default: adaptive 2 * n * sqrt(n))",
  )
  parser.add_argument(
    "--benchmark-name",
    type=str,
    default="benchmark",
    help="Name identifier for this benchmark run (used in logs and report filenames; default: benchmark)",
  )
  parser.add_argument(
    "--reports",
    nargs="+",
    type=str,
    default=None,
    metavar="ARTIFACT",
    help=(
      "Opt-in report artifacts to generate after collection: 'none' (default), 'all', "
      "or any subset of markdown, latex, macros, cd, plots"
    ),
  )
  parser.add_argument(
    "--report-style",
    type=Path,
    default=None,
    help="Optional JSON file configuring display names, labels, decimal separators, and size thresholds",
  )
  parser.add_argument(
    "--include-size-category-table",
    action="store_true",
    help="Include Table 3 (best algorithm by problem size stratum) when generating Markdown/LaTeX tables",
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
    help="Run in test mode (fewer problems for quick verification)",
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
    [Mapping[str, AlgorithmConfig], GAParams, Sequence[ProblemConfig]],
    Sequence[str],
  ]
  | None = None,
  on_complete: Callable[[argparse.Namespace, dict[str, dict[str, dict[str, Any]]]], None] | None = None,
  parser_epilog: str | None = None,
) -> int:
  """Execute the benchmark CLI command."""
  parser = build_parser(epilog=parser_epilog)
  args = parser.parse_args(argv)

  if args.config_dir is None and args.algorithms_config is None:
    parser.error("one of the arguments --config-dir or --algorithms-config is required")

  level = getattr(logging, args.log_level.upper(), logging.INFO)
  setup_benchmark_logging(args.output_dir, args.benchmark_name, log_level=level)

  try:
    selected_reports = parse_report_artifacts(args.reports, default=())
  except ValueError as e:
    logging.error(f"Invalid --reports option: {e}")
    return 2

  report_style: ReportStyle | None = None
  if args.report_style is not None:
    try:
      report_style = load_report_style(args.report_style)
    except (FileNotFoundError, ValueError) as e:
      logging.error(f"Failed to load --report-style: {e}")
      return 2

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
      algorithms_config=args.algorithms_config,
      benchmark_config=args.benchmark_config,
      problems_config=args.problems_config,
      output_dir=args.output_dir,
      db_path=args.db_path,
      tsp_dir=args.tsp_dir,
      tsp_files=args.tsp_files,
      problem_names=args.problems,
      algorithms=args.algorithms,
      repetitions=args.repetitions,
      max_generations=args.max_generations,
      benchmark_name=args.benchmark_name,
      local_search_stage=args.local_search_stage,
      skip_cpu=args.skip_cpu,
      incremental_runs=args.incremental_runs,
      cpu_size_threshold=args.cpu_size_threshold,
      baseline_algorithm=args.baseline_algorithm,
      reports=selected_reports,
      report_style=report_style,
      include_size_category_table=args.include_size_category_table,
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
