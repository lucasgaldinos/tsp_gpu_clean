"""DuckDB directory ingestion built on ``routing_data.converter.database.operations.DatabaseManager``."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
from converter.core.solutions import parse_solution_file, solution_stem
from converter.database.operations import DatabaseManager

from data_models.exceptions import (
  DatabaseNotEmptyError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
)
from data_models.problem import VALID_EDGE_TYPES, VRP_PROBLEM_TYPES, expected_coordinate_dimension

from .tsplib_models import file_checksum, normalize_tsplib_file

if TYPE_CHECKING:
  from converter.core.contracts import SolutionPayload
  from converter.core.transformer import NormalizedProblemData

logger = logging.getLogger(__name__)


def _problem_payload(
  tsp_file: Path,
  normalized: NormalizedProblemData,
) -> dict[str, Any]:
  """Build the flat schema-v2 payload `DatabaseManager.insert_problem` consumes."""
  problem_meta: dict[str, Any] = normalized["problem_data"]
  instance_name: str = tsp_file.stem
  problem_type: str = str(problem_meta.get("type") or "TSP")
  edge_type = str(problem_meta.get("edge_weight_type") or "").upper().strip()
  dimension = int(problem_meta.get("dimension") or 0)
  capacity_raw: Any = problem_meta.get("capacity")

  if edge_type not in VALID_EDGE_TYPES:
    raise InvalidEdgeTypeError(edge_type)
  if dimension <= 0:
    raise InvalidProblemDataError(instance_name, f"Invalid dimension: {dimension}")

  coords: list[list[float]] | None = normalized["coords"]
  display_coords: list[list[float]] | None = normalized["display_coords"]

  if edge_type != "EXPLICIT" and not coords:
    raise MissingCoordinatesError(instance_name, edge_type)

  expected_coord_dim: int = expected_coordinate_dimension(edge_type)
  if edge_type == "EXPLICIT":
    display_coords = coords or display_coords
    coords = None
    if display_coords is not None:
      for zero_idx, pt in enumerate(display_coords):
        if len(pt) != 2:
          raise InvalidProblemDataError(
            instance_name,
            f"EXPLICIT display_coords require 2D coordinates (found coordinate of length {len(pt)} at node {zero_idx})",
          )
        if not all(math.isfinite(float(c)) for c in pt):
          raise InvalidProblemDataError(
            instance_name,
            f"EXPLICIT display_coords contain non-finite value at node {zero_idx}",
          )
  else:
    if coords is not None:
      if len(coords) < dimension:
        raise MissingCoordinatesError(instance_name, edge_type)
      if len(coords) > dimension:
        raise InvalidProblemDataError(
          instance_name,
          f"Expected {dimension} coordinates for edge_type '{edge_type}', got {len(coords)}",
        )
      for zero_idx, pt in enumerate(coords):
        if len(pt) != expected_coord_dim:
          coord_len = len(pt)
          raise InvalidProblemDataError(
            instance_name,
            f"edge_type '{edge_type}' requires {expected_coord_dim}D coordinates "
            f"(found coordinate of length {coord_len} at node {zero_idx})",
          )
        floats = [float(c) for c in pt]
        if any(math.isnan(c) for c in floats):
          raise MissingCoordinatesError(instance_name, edge_type)
        if not all(math.isfinite(c) for c in floats):
          raise InvalidProblemDataError(
            instance_name,
            f"Non-finite coordinate at node {zero_idx} for edge_type '{edge_type}'",
          )
    if display_coords is not None:
      for zero_idx, pt in enumerate(display_coords):
        if len(pt) != 2:
          raise InvalidProblemDataError(
            instance_name,
            f"display_coords require 2D coordinates (found coordinate of length {len(pt)} at node {zero_idx})",
          )
        if not all(math.isfinite(float(c)) for c in pt):
          raise InvalidProblemDataError(
            instance_name,
            f"display_coords contain non-finite value at node {zero_idx}",
          )

  edge_weight_matrix: list[list[int]] | None = normalized.get("edge_weight_matrix")
  edge_weight_data: dict[str, Any] | None = None
  if edge_weight_matrix is not None:
    edge_weight_data = {
      "matrix": edge_weight_matrix,
      "matrix_format": problem_meta.get("edge_weight_format") or "FULL_MATRIX",
      "is_symmetric": problem_type != "ATSP",
    }

  return {
    "name": tsp_file.stem,
    "type": problem_type,
    "comment": problem_meta.get("comment"),
    "dimension": dimension,
    "capacity": int(capacity_raw) if capacity_raw is not None else None,
    "edge_weight_type": edge_type,
    "edge_weight_format": problem_meta.get("edge_weight_format"),
    "tsplib_name": problem_meta.get("name") or tsp_file.stem,
    "coords": coords,
    "display_coords": display_coords,
    "demands": normalized["demands"],
    "depots": normalized["depots"],
    "max_distance": problem_meta.get("max_distance"),
    "vehicles": problem_meta.get("vehicles"),
    "service_time": problem_meta.get("service_time"),
    "adjacency": normalized["edges"],
    "fixed_edges": normalized["fixed_edges"],
    "edge_weight_data": edge_weight_data,
    "file_path": str(tsp_file.resolve()),
    "checksum": file_checksum(tsp_file),
    "file_size": tsp_file.stat().st_size if tsp_file.exists() else 0,
  }


def _resolve_solution_problem_id(
  conn: duckdb.DuckDBPyConnection,
  instance_name: str,
  kind: str,
) -> int | None:
  """Resolve `(name, type)` uniquely for solution ingestion (`.sol` -> VRP, `.opt.tour` -> TSP/ATSP/SOP)."""
  rows = conn.execute(
    "SELECT id, type FROM problems WHERE name = ? ORDER BY id ASC",
    [instance_name],
  ).fetchall()
  if not rows:
    return None
  if len(rows) == 1:
    return int(rows[0][0])

  preferred = VRP_PROBLEM_TYPES if kind == "SOL" else frozenset({"TSP", "ATSP", "SOP"})
  matched = [int(pid) for pid, ptype in rows if str(ptype) in preferred]
  if len(matched) == 1:
    return matched[0]
  return None


def _ingest_opt_tour_solutions(db_path: Path, source_dir: Path) -> None:
  """Insert every ``*.opt.tour`` and ``*.sol`` next to instances via ``DatabaseManager.insert_solutions``."""
  solution_dirs: tuple[Path, ...] = tuple(
    directory
    for directory in (source_dir, source_dir / "tour", source_dir / "sol", source_dir / "solutions")
    if directory.is_dir()
  )
  solution_files: list[Path] = sorted(
    {path for directory in solution_dirs for ext in ("*.opt.tour", "*.sol") for path in directory.glob(ext)}
  )
  if not solution_files:
    return

  payloads: list[SolutionPayload] = []
  with duckdb.connect(str(db_path), read_only=True) as conn:
    for sol_file in solution_files:
      try:
        parsed = parse_solution_file(sol_file)
      except Exception as exc:
        logger.warning("Skipping unreadable solution file '%s': %s", sol_file, exc)
        continue

      instance_name: str = solution_stem(sol_file)
      problem_id = _resolve_solution_problem_id(conn, instance_name, str(parsed["kind"]))
      if problem_id is None:
        continue

      payload: SolutionPayload = {
        "kind": parsed["kind"],
        "quality": parsed["quality"],
        "source_file": str(sol_file.resolve()),
        "stated_cost": parsed["stated_cost"],
        "cost": parsed["cost"],
        "routes": parsed["routes"],
        "problem_id": problem_id,
        "checksum": file_checksum(sol_file),
      }
      payloads.append(payload)

  if payloads:
    DatabaseManager(str(db_path)).insert_solutions(payloads)


def ingest_tsplib_directory_to_duckdb(
  tsp_dir: str | Path,
  db_path: str | Path,
  *,
  overwrite: bool = False,
) -> int:
  """Parse all supported `.tsp` and `.vrp` files in `tsp_dir` and ingest them into DuckDB."""
  source_dir = Path(tsp_dir)
  if not source_dir.exists() or not source_dir.is_dir():
    raise FileNotFoundError(f"TSPLIB source directory not found: {tsp_dir}")
  target_db = Path(db_path)
  target_db.parent.mkdir(parents=True, exist_ok=True)

  if overwrite and target_db.exists():
    target_db.unlink()

  manager = DatabaseManager(str(target_db))
  stats: dict[str, Any] = manager.get_problem_stats()
  if int(stats["total_problems"]) > 0 and not overwrite:
    raise DatabaseNotEmptyError(str(target_db))

  ingested = 0
  instance_files = sorted([*source_dir.glob("*.tsp"), *source_dir.glob("*.vrp")])
  for tsp_file in instance_files:
    try:
      normalized = normalize_tsplib_file(tsp_file)
      manager.insert_problem(_problem_payload(tsp_file, normalized))
    except InvalidEdgeTypeError as exc:
      logger.warning("Skipping '%s': %s", tsp_file, exc)
      continue
    except Exception as exc:
      logger.warning("Skipping malformed instance '%s': %s", tsp_file, exc)
      continue
    ingested += 1

  _ingest_opt_tour_solutions(target_db, source_dir)
  return ingested


__all__: list[str] = [
  "ingest_tsplib_directory_to_duckdb",
]
