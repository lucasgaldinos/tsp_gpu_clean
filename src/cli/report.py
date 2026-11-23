"""CLI entrypoint for regenerating benchmark reports directly from JSON checkpoints.

Reads ``checkpoints/*.json`` without re-running any solver or touching DuckDB,
and emits selectable per-problem statistics JSONs, Markdown/LaTeX summary tables,
``benchmark_macros.tex``, Demšar Critical Difference TikZ diagrams, and vector
scaling figures (PDF/SVG), styled via an optional ``--report-style`` JSON file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarking.analysis.cohort import derive_problem_configs
from benchmarking.analysis.cohort_analysis import analyze_cohort
from benchmarking.analysis.problem_statistics import analyze_problem
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.reporting.console import log_cohort_analysis, log_problem_analysis
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.reporting.style import (
  ALL_REPORT_ARTIFACTS,
  ReportStyle,
  load_report_style,
  parse_report_artifacts,
)
from benchmarking.runner.checkpoint_io import CheckpointManager, load_checkpoints_dir, save_problem_statistics
from benchmarking.runner.config_loader import ProblemConfig, load_problems_config

if TYPE_CHECKING:
  from collections.abc import Sequence


def _load_problem_configs(
  all_results: dict[str, dict[str, dict[str, Any]]],
  config_dir: Path | None,
  problems_config: Path | None = None,
) -> list[ProblemConfig]:
  """Build ``ProblemConfig`` entries from optional ``problems.json`` plus checkpoint metadata."""
  explicit: list[ProblemConfig] | None = None
  if problems_config is not None:
    explicit = load_problems_config(problems_config)
  elif config_dir is not None:
    if not config_dir.is_dir():
      raise FileNotFoundError(f"Configuration directory not found: '{config_dir}'")
    if (config_dir / "problems.json").is_file():
      explicit = load_problems_config(config_dir)
  return derive_problem_configs(all_results, explicit)


def build_parser() -> argparse.ArgumentParser:
  """Build the argument parser for standalone report regeneration."""
  parser = argparse.ArgumentParser(
    prog="tsp-report",
    description=(
      "Regenerate statistical summaries, Markdown/LaTeX tables, quantitative macros, "
      "Demšar CD TikZ diagrams, and scaling figures directly from JSON checkpoints (no DuckDB required)."
    ),
  )
  parser.add_argument(
    "-i",
    "--checkpoints",
    type=Path,
    required=True,
    help="Path to checkpoints directory containing *.json files (or parent results directory containing checkpoints/)",
  )
  parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
    default=None,
    help=(
      "Output root directory for tables/, figures/, and problem_statistics/ "
      "(default: parent of checkpoints/ when named 'checkpoints', else the input directory)"
    ),
  )
  parser.add_argument(
    "--config-dir",
    type=Path,
    default=None,
    help="Optional configuration directory containing problems.json for problem sizes and reference optima",
  )
  parser.add_argument(
    "--problems-config",
    type=Path,
    default=None,
    help="Optional direct path to a problems.json file for problem sizes and reference optima",
  )
  parser.add_argument(
    "--benchmark-name",
    type=str,
    default="benchmark",
    help="Base filename stem for generated tables and CD diagrams (default: benchmark)",
  )
  parser.add_argument(
    "--baseline-algorithm",
    "--baseline",
    type=str,
    default=None,
    dest="baseline_algorithm",
    help="Baseline algorithm name for speedup calculations (default: first algorithm found)",
  )
  parser.add_argument(
    "--reports",
    nargs="+",
    type=str,
    default=None,
    metavar="ARTIFACT",
    help=("Report artifacts to emit: 'all' (default), 'none', or any subset of markdown, latex, macros, cd, plots"),
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
    help="Include Table 3 (best algorithm by problem size stratum) in Markdown and LaTeX outputs",
  )
  parser.add_argument(
    "--skip-problem-stats",
    action="store_true",
    help="Skip writing per-problem JSON summaries under problem_statistics/",
  )
  parser.add_argument(
    "--log-level",
    type=str.upper,
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
    help="Set logging verbosity level (default: INFO)",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Execute the standalone checkpoint-to-report CLI command."""
  parser = build_parser()
  args = parser.parse_args(argv)

  resolved_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
  logging.basicConfig(
    level=resolved_log_level,
    format="%(levelname)s: %(message)s",
  )
  logging.getLogger().setLevel(resolved_log_level)

  try:
    selected_reports = parse_report_artifacts(args.reports, default=ALL_REPORT_ARTIFACTS)
  except ValueError as exc:
    logging.error("%s", exc)
    return 2

  report_style: ReportStyle | None = None
  if args.report_style is not None:
    try:
      report_style = load_report_style(args.report_style)
    except (FileNotFoundError, ValueError) as exc:
      logging.error("%s", exc)
      return 2

  try:
    all_results = load_checkpoints_dir(args.checkpoints)
  except FileNotFoundError as exc:
    logging.error("%s", exc)
    return 1

  if not all_results:
    logging.error("No valid checkpoint JSON files found in '%s'.", args.checkpoints)
    return 1

  ckpt_input = Path(args.checkpoints)
  if args.output_dir is not None:
    output_root = Path(args.output_dir)
  elif ckpt_input.name == "checkpoints":
    output_root = ckpt_input.parent
  else:
    output_root = ckpt_input

  try:
    problem_configs = _load_problem_configs(all_results, args.config_dir, args.problems_config)
  except (FileNotFoundError, ValueError) as exc:
    logging.error("%s", exc)
    return 2

  cfg_by_name = {cfg.name: cfg for cfg in problem_configs}
  analyzer = StatisticalAnalyzer()

  if not args.skip_problem_stats:
    resolved_ckpt_dir = (ckpt_input / "checkpoints") if (ckpt_input / "checkpoints").is_dir() else ckpt_input
    checkpoint_manager = CheckpointManager(
      checkpoint_dir=resolved_ckpt_dir,
      problem_stats_dir=output_root / "problem_statistics",
    )
    for problem_name, alg_results in sorted(all_results.items()):
      cfg = cfg_by_name[problem_name]
      first_entry = next(iter(alg_results.values()))
      meta = first_entry.get("problem_metadata") if isinstance(first_entry.get("problem_metadata"), dict) else None
      problem_analysis = analyze_problem(
        problem_name=problem_name,
        optimal_cost=cfg.optimal,
        results=alg_results,
        analyzer=analyzer,
        baseline_algorithm=args.baseline_algorithm,
      )
      log_problem_analysis(problem_analysis)
      save_problem_statistics(
        problem_name=problem_name,
        problem_size=cfg.size,
        optimal_cost=cfg.optimal,
        problem_results=alg_results,
        statistical_test_results=problem_analysis.to_dict(),
        checkpoint_manager=checkpoint_manager,
        problem_metadata=meta,
      )

  cohort_analysis = None
  if len(all_results) > 1 or selected_reports:
    cohort_analysis = analyze_cohort(
      all_results,
      problem_configs,
      baseline_algorithm=args.baseline_algorithm,
      alpha=analyzer.alpha,
      algorithm_order=report_style.algorithm_order if report_style is not None else None,
    )
    if len(all_results) > 1:
      log_cohort_analysis(cohort_analysis)

  if selected_reports:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_result_tables(
      all_results=all_results,
      problem_configs=problem_configs,
      output_dir=tables_dir,
      benchmark_name=args.benchmark_name,
      baseline_algorithm=args.baseline_algorithm,
      include_size_category_table=args.include_size_category_table,
      style=report_style,
      artifacts=selected_reports,
      analysis=cohort_analysis,
    )

  artifacts_desc = ",".join(selected_reports) if selected_reports else "none"
  logging.info(
    "Reports regenerated in %s (%d problem(s), artifacts=%s).", output_root, len(all_results), artifacts_desc
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
