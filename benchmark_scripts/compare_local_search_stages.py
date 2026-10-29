#!/usr/bin/env python3
"""
Compare Local Search Placement Stages in Genetic Algorithm
===========================================================

Executes and compares different local search (2-opt) pipeline stages:
- `post_mutation`: canonical memetic GA (local search applied after mutation)
- `post_evaluation`: local search applied after fitness evaluation and survival
- `post_evolution`: pure GA evolutionary search, with local search applied to
  the full population once after evolution terminates

Target default problems:
- `datasets/tsp/att48.tsp` (n=48, ATT)
- `datasets/tsp/bayg29.tsp` (n=29, EXPLICIT UPPER_ROW)
- `datasets/tsp/bays29.tsp` (n=29, EXPLICIT FULL_MATRIX)

Usage:
    python -m benchmark_scripts.compare_local_search_stages --repetitions 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from benchmarking.runner.algorithm_runner import adaptive_generations, run_single_algorithm
from benchmarking.runner.config_loader import (
  KNOWN_TSPLIB_OPTIMALS,
  build_problem_configs_from_tsp_files,
  default_config_dir,
  load_all_configs,
)
from loaders.tsplib_loader import TsplibLoader

DEFAULT_TARGET_PROBLEMS: list[str] = [
  "datasets/tsp/att48.tsp",
  "datasets/tsp/bayg29.tsp",
  "datasets/tsp/bays29.tsp",
]

DEFAULT_STAGES: list[str] = [
  "post_mutation",
  "post_evaluation",
  "post_evolution",
]

DEFAULT_ALGORITHMS: list[str] = [
  "CPU",
  "HybridNaive",
  "HybridOptimized",
  "FullGPUModular",
]


def setup_logging() -> None:
  """Configure logging to stdout."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse CLI arguments for stage comparison."""
  parser = argparse.ArgumentParser(
    description=(
      "Compare Genetic Algorithm local search placement stages "
      "(post_mutation vs post_evaluation vs post_evolution)."
    ),
  )
  parser.add_argument(
    "--tsp-files",
    nargs="+",
    default=DEFAULT_TARGET_PROBLEMS,
    help="Paths to .tsp problem files.",
  )
  parser.add_argument(
    "--stages",
    nargs="+",
    default=DEFAULT_STAGES,
    choices=["post_crossover", "post_mutation", "post_evaluation", "post_evolution", "none"],
    help="Local search stages to compare.",
  )
  parser.add_argument(
    "--algorithms",
    nargs="+",
    default=DEFAULT_ALGORITHMS,
    help="Algorithms to benchmark across stages.",
  )
  parser.add_argument(
    "--repetitions",
    type=int,
    default=3,
    help="Number of repetitions per (problem, algorithm, stage) tuple (default: 3).",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="results_v3/local_search_comparison",
    help="Directory to save checkpoints and comparison tables.",
  )
  parser.add_argument(
    "--config-dir",
    type=str,
    default=None,
    help="Config directory holding algorithms.json and benchmark.json.",
  )
  return parser.parse_args(argv)


def run_stage_comparison(args: argparse.Namespace) -> dict[str, Any]:
  """Execute comparison across problems, stages, and algorithms."""
  cfg_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
  all_configs = load_all_configs(cfg_dir)
  ga_params = all_configs["ga_params"]
  algo_configs = all_configs["algorithms"]
  benchmark_cfg = all_configs["benchmark"]

  problem_configs, tsp_map = build_problem_configs_from_tsp_files(args.tsp_files)
  output_path = Path(args.output_dir)
  checkpoints_dir = output_path / "checkpoints"
  tables_dir = output_path / "tables"
  checkpoints_dir.mkdir(parents=True, exist_ok=True)
  tables_dir.mkdir(parents=True, exist_ok=True)

  logging.info("=" * 80)
  logging.info("LOCAL SEARCH STAGES COMPARISON BENCHMARK")
  logging.info("=" * 80)
  logging.info(f"Target problems : {[c.name for c in problem_configs]}")
  logging.info(f"Target stages   : {args.stages}")
  logging.info(f"Algorithms      : {args.algorithms}")
  logging.info(f"Repetitions     : {args.repetitions}")
  logging.info(f"Output directory: {output_path.resolve()}")
  logging.info("=" * 80)

  records: list[dict[str, Any]] = []

  for prob_cfg in problem_configs:
    pname = prob_cfg.name
    optimal_cost = KNOWN_TSPLIB_OPTIMALS.get(pname, prob_cfg.optimal)
    tsp_path = tsp_map[pname]

    with TsplibLoader(tsp_path.parent) as loader:
      problem = loader.load(tsp_path)

    customers = list(range(problem.dimension))
    max_generations = adaptive_generations(problem.dimension)

    logging.info(f"\n>>> Running problem: {pname} (n={problem.dimension}, optimal={optimal_cost})")

    for stage in args.stages:
      for alg_name in args.algorithms:
        if alg_name not in algo_configs:
          logging.warning(f"Skipping unknown algorithm '{alg_name}'")
          continue

        alg_cfg = algo_configs[alg_name]
        logging.info(f"  -> Testing {alg_name:15s} | Stage: {stage:15s} ({args.repetitions} reps)...")

        # Instantiate algorithm with local_search_stage override
        algo_instance = alg_cfg.instantiate(ga_params, local_search_stage=stage)

        res = run_single_algorithm(
          algorithm_name=alg_name,
          algorithm_instance=algo_instance,
          problem=problem,
          problem_name=pname,
          problem_size=problem.dimension,
          customers=customers,
          max_generations=max_generations,
          optimal_cost=optimal_cost,
          repetitions=args.repetitions,
          use_gpu=alg_cfg.use_gpu,
          patience=benchmark_cfg.patience,
          existing_checkpoint=None,
          incremental_mode=False,
        )

        # Enrich checkpoint
        res["stage"] = stage
        res["optimal_cost"] = optimal_cost
        res["problem_name"] = pname
        res["problem_size"] = problem.dimension

        chk_name = f"{pname}_{alg_name}_{stage}.json"
        chk_file = checkpoints_dir / chk_name
        with open(chk_file, "w", encoding="utf-8") as f:
          json.dump(res, f, indent=2)

        mean_cost = res.get("mean_cost", 0.0)
        gap = (mean_cost - optimal_cost) / optimal_cost * 100.0 if optimal_cost else 0.0
        mean_time = res.get("mean_time", 0.0)
        stop_reason = res.get("stop_reason", "max_generations")

        records.append({
          "problem": pname,
          "dimension": problem.dimension,
          "optimal": optimal_cost,
          "stage": stage,
          "algorithm": alg_name,
          "mean_cost": mean_cost,
          "std_cost": res.get("std_cost", 0.0),
          "gap_pct": gap,
          "mean_time": mean_time,
          "std_time": res.get("std_time", 0.0),
          "stop_reason": stop_reason,
        })

  # Generate comparative Markdown and LaTeX tables
  md_table_path = tables_dir / "local_search_stage_comparison.md"
  tex_table_path = tables_dir / "local_search_stage_comparison.tex"

  _generate_comparison_markdown(records, md_table_path)
  _generate_comparison_latex(records, tex_table_path)

  logging.info("\n" + "=" * 80)
  logging.info("BENCHMARK COMPARISON COMPLETE")
  logging.info(f"Saved Markdown summary : {md_table_path.resolve()}")
  logging.info(f"Saved LaTeX summary    : {tex_table_path.resolve()}")
  logging.info("=" * 80)

  return {"records": records, "markdown_table": str(md_table_path), "latex_table": str(tex_table_path)}


def _generate_comparison_markdown(records: list[dict[str, Any]], out_path: Path) -> None:
  """Generate markdown comparison table."""
  lines: list[str] = [
    "# Local Search Stage Comparison Benchmark",
    f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "\nCompares solution quality, gap, and runtime across 2-opt placement stages:",
    "- `post_mutation`: Traditional memetic GA (refinement after mutation each generation)",
    "- `post_evaluation`: Refinement applied after fitness evaluation & survival",
    "- `post_evolution`: Pure GA evolution, with 2-opt refinement applied to the full final population",
    "\n| Problem | Algorithm | Stage | Mean Cost | Std Cost | Gap (%) | Mean Time (s) | Stop Reason |",
    "|:---|:---|:---|---:|---:|---:|---:|:---|",
  ]
  for r in records:
    row = (
      f"| {r['problem']} | {r['algorithm']} | `{r['stage']}` | "
      f"{r['mean_cost']:.2f} | {r['std_cost']:.2f} | {r['gap_pct']:.2f}% | "
      f"{r['mean_time']:.3f} | {r['stop_reason']} |"
    )
    lines.append(row)

  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_comparison_latex(records: list[dict[str, Any]], out_path: Path) -> None:
  """Generate LaTeX tabular representation."""
  lines: list[str] = [
    "% Auto-generated Local Search Stage Comparison Table",
    "\\begin{table}[htbp]",
    "\\centering",
    "\\small",
    "\\caption{Comparison of Local Search Placement Stages (post\\_mutation vs post\\_evaluation vs post\\_evolution)}",
    "\\label{tab:local_search_stages}",
    "\\begin{tabular}{lllrrrr}",
    "\\toprule",
    "Problem & Algorithm & Stage & Mean Cost & Std Dev & Gap (\\%) & Time (s) \\\\",
    "\\midrule",
  ]
  for r in records:
    stage_tex = r["stage"].replace("_", "\\_")
    row = (
      f"{r['problem']} & {r['algorithm']} & {stage_tex} & "
      f"{r['mean_cost']:.2f} & {r['std_cost']:.2f} & {r['gap_pct']:.2f} & {r['mean_time']:.3f} \\\\"
    )
    lines.append(row)
  lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
  ])
  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  """CLI entry point."""
  setup_logging()
  args = parse_arguments(argv)
  run_stage_comparison(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())