"""Public facade for TSPLIB/VRP parsing, BKS resolution, and DuckDB ingestion.

Re-exports the cohesive submodules (`tsplib_models`, `tsplib_reader`, `tsplib_ingestor`)
so all callers importing from ``loaders.tsplib_loader`` remain 100% compatible (`BUG-116`).
"""

from __future__ import annotations

from .tsplib_ingestor import (
  ingest_tsplib_directory_to_duckdb,
)
from .tsplib_models import (
  MatrixRowRecord,
  NodeRowRecord,
  ProblemRowRecord,
  TsplibIngestionBundle,
  bundle_from_normalized,
  file_checksum,
  normalize_tsplib_file,
  parse_tsplib_file,
)
from .tsplib_reader import (
  TsplibLoader,
  bundle_to_problem,
  load_tsplib_file,
  resolve_opt_tour_cost,
)

__all__: list[str] = [
  "MatrixRowRecord",
  "NodeRowRecord",
  "ProblemRowRecord",
  "TsplibIngestionBundle",
  "TsplibLoader",
  "bundle_from_normalized",
  "bundle_to_problem",
  "file_checksum",
  "ingest_tsplib_directory_to_duckdb",
  "load_tsplib_file",
  "normalize_tsplib_file",
  "parse_tsplib_file",
  "resolve_opt_tour_cost",
]
