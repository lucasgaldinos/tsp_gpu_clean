"""CLI for TSPLIB dataset parsing, inspection, and DuckDB ingestion."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from data_models.exceptions import DatabaseNotEmptyError, InvalidEdgeTypeError
from loaders.tsplib_loader import (
  TsplibIngestionBundle,
  TsplibLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  parse_tsplib_file,
)

if TYPE_CHECKING:
  from data_models.problem import Problem


def build_parser() -> argparse.ArgumentParser:
  """Build the argument parser for TSPLIB operations."""
  parser = argparse.ArgumentParser(
    prog="tsp-ingest",
    description="Parse TSPLIB .tsp problems with routing_data into DuckDB ingestion format.",
  )
  parser.add_argument(
    "--tsp-dir",
    type=Path,
    required=True,
    help="Directory containing .tsp files",
  )
  parser.add_argument(
    "--instance",
    type=str,
    default=None,
    help="Parse and validate a single instance name (e.g., berlin52)",
  )
  parser.add_argument(
    "--to-duckdb",
    type=Path,
    default=None,
    help="Optional path to write ingested records into a DuckDB database",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing records if database already contains data",
  )
  parser.add_argument(
    "--max-eager-matrix-dimension",
    type=int,
    default=0,
    help="Maximum problem dimension to eagerly compute distance matrix (default: 0)",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Execute the TSPLIB CLI command."""
  parser = build_parser()
  args = parser.parse_args(argv)

  if not args.tsp_dir.exists():
    print(f"Error: TSPLIB directory '{args.tsp_dir}' does not exist.", file=sys.stderr)
    return 1

  if args.instance is not None:
    with TsplibLoader(args.tsp_dir) as loader:
      problem: Problem = loader.load(args.instance)
      print(
        f"Loaded {problem.name}: type={problem.problem_type}, "
        f"dimension={problem.dimension}, edge_type={problem.edge_type}, "
        f"distances_shape={problem.distances.shape if problem.distances is not None else None}"
      )
    return 0

  if args.to_duckdb is not None:
    try:
      count: int = ingest_tsplib_directory_to_duckdb(args.tsp_dir, args.to_duckdb, overwrite=args.overwrite)
      print(f"Ingested {count} TSPLIB problems into {args.to_duckdb}")
      return 0
    except DatabaseNotEmptyError as e:
      print(f"Error: {e}", file=sys.stderr)
      return 1

  with TsplibLoader(args.tsp_dir) as loader:
    names: list[str] = loader.list_instances()
    valid_count = 0
    for idx, name in enumerate(names, start=1):
      try:
        bundle: TsplibIngestionBundle = parse_tsplib_file(loader.tsp_dir / f"{name}.tsp", problem_id=idx)
        bundle_to_problem(bundle, max_eager_matrix_dimension=args.max_eager_matrix_dimension)
        valid_count += 1
      except InvalidEdgeTypeError:
        continue
    print(f"Verified {valid_count}/{len(names)} .tsp files in {args.tsp_dir} via routing_data.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
