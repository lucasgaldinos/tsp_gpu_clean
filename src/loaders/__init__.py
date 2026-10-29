"""
Problem loaders for importing routing instances.

This package provides loaders for accessing routing problem instances
from various sources:
- `DatabaseLoader`: Context manager for loading from DuckDB database
- `TsplibLoader`: Context manager for loading `.tsp` files via `tsplib95`
"""

from .database_loader import DatabaseLoader
from .tsplib_loader import (
  MatrixRowRecord,
  NodeRowRecord,
  ProblemRowRecord,
  TsplibIngestionBundle,
  TsplibLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  parse_tsp_file_with_tsplib95,
)

__all__ = [
  "DatabaseLoader",
  "MatrixRowRecord",
  "NodeRowRecord",
  "ProblemRowRecord",
  "TsplibIngestionBundle",
  "TsplibLoader",
  "bundle_to_problem",
  "ingest_tsplib_directory_to_duckdb",
  "parse_tsp_file_with_tsplib95",
]
