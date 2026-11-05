"""
Problem loaders for importing routing instances.

This package provides loaders for accessing routing problem instances
from various sources:
- `DatabaseLoader`: Context manager for loading from DuckDB database
- `TsplibLoader`: Context manager for loading `.tsp` files through `routing_data`
"""

from .database_loader import DatabaseLoader
from .problem_type_handlers import (
  PROBLEM_TYPE_HANDLERS,
  AtspHandler,
  BaseProblemRow,
  CvrpHandler,
  ProblemTypeHandler,
  TspHandler,
  register_problem_type_handler,
)
from .tsplib_loader import (
  MatrixRowRecord,
  NodeRowRecord,
  ProblemRowRecord,
  TsplibIngestionBundle,
  TsplibLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  parse_tsplib_file,
)

__all__ = [
  "PROBLEM_TYPE_HANDLERS",
  "AtspHandler",
  "BaseProblemRow",
  "CvrpHandler",
  "DatabaseLoader",
  "MatrixRowRecord",
  "NodeRowRecord",
  "ProblemRowRecord",
  "ProblemTypeHandler",
  "TspHandler",
  "TsplibIngestionBundle",
  "TsplibLoader",
  "bundle_to_problem",
  "ingest_tsplib_directory_to_duckdb",
  "parse_tsplib_file",
  "register_problem_type_handler",
]
