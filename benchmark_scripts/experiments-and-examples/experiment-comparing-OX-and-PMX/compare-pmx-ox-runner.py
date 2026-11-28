#!/usr/bin/env python3
"""GPU-only OX vs PMX experiment runner (TSPLIB instances with n > 150).

Delegates CLI parsing and orchestration to `cli.benchmark.main` while keeping
all configs, custom GPU PMX operator/kernel, and `results/` inside this folder.

Usage:
    uv run python .../experiment-comparing-OX-and-PMX/compare-pmx-ox-runner.py --dry-run
    uv run python .../experiment-comparing-OX-and-PMX/compare-pmx-ox-runner.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cli.benchmark as benchmark_cli

if TYPE_CHECKING:
  import argparse
  from collections.abc import Sequence

EXP_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = EXP_DIR.parents[2]
CONFIG_DIR: Path = EXP_DIR / "configs"
RESULTS_DIR: Path = EXP_DIR / "results"
REPORT_STYLE_PATH: Path = CONFIG_DIR / "report_style.json"
DEFAULT_DB_PATH: Path = REPO_ROOT / "datasets" / "db" / "routing.duckdb"
BENCHMARK_NAME: str = "ox_vs_pmx"
BASELINE_ALGORITHM: str = "OX_GPU"


def on_experiment_complete(
  args: argparse.Namespace,
  results: dict[str, dict[str, dict[str, Any]]],
) -> None:
  """Log a concise completion summary pointing to the experiment-local results directory."""
  variant_names = sorted({alg for per_prob in results.values() for alg in per_prob})
  logging.info(
    "Completed '%s' across %d instance(s) and %d variant(s) (%s). Outputs: %s",
    args.benchmark_name,
    len(results),
    len(variant_names),
    ", ".join(variant_names),
    args.output_dir,
  )


def main(argv: Sequence[str] | None = None) -> int:
  """Run the GPU-only OX vs PMX benchmark via `cli.benchmark.main`."""
  defaults: list[str] = [
    "--config-dir",
    str(CONFIG_DIR),
    "--db-path",
    str(DEFAULT_DB_PATH),
    "--output-dir",
    str(RESULTS_DIR),
    "--benchmark-name",
    BENCHMARK_NAME,
    "--baseline",
    BASELINE_ALGORITHM,
    "--report-style",
    str(REPORT_STYLE_PATH),
    "--reports",
    "all",
  ]
  extra = list(sys.argv[1:] if argv is None else argv)
  return benchmark_cli.main(
    [*defaults, *extra],
    on_complete=on_experiment_complete,
    parser_epilog="GPU-only OX vs PMX benchmark wrapper anchored to this experiment folder.",
  )


if __name__ == "__main__":
  raise SystemExit(main())
