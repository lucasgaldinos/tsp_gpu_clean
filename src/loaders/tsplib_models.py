"""Structured TSPLIB/VRP ingestion dataclasses and file parsing helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from converter.core.transformer import NormalizedProblemData, ProblemNormalizer
from tsplib_parser.parser import FormatParser

from data_models.exceptions import (
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
)
from data_models.problem import VALID_EDGE_TYPES, expected_coordinate_dimension


@dataclass(frozen=True)
class ProblemRowRecord:
  """Row schema for problem records in DuckDB."""

  problem_id: int
  name: str
  problem_type: str
  dimension: int
  edge_weight_type: str
  capacity: int | None
  comment: str | None
  edge_weight_format: str | None = None
  tsplib_name: str | None = None
  source_file: str | None = None
  file_checksum: str | None = None
  max_distance: float | None = None
  vehicles: int | None = None
  service_time: float | None = None
  display_coords: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class NodeRowRecord:
  """Row schema for the `nodes` table in DuckDB (0-based `node_id`)."""

  problem_id: int
  node_id: int
  x: float | None
  y: float | None
  demand: int
  is_depot: bool
  z: float | None = None


@dataclass(frozen=True)
class MatrixRowRecord:
  """Row schema for the `edge_weight_matrices` table in DuckDB."""

  problem_id: int
  dimension: int
  matrix_format: str
  is_symmetric: bool
  matrix_json: str = ""
  matrix: list[list[int]] | list[list[float]] | None = None


@dataclass(frozen=True)
class TsplibIngestionBundle:
  """Complete structured ingestion payload produced from a `.tsp` or `.vrp` file."""

  problem: ProblemRowRecord
  nodes: tuple[NodeRowRecord, ...]
  edge_matrix: MatrixRowRecord | None


def file_checksum(path: Path) -> str | None:
  """Return the SHA-256 checksum of ``path`` or ``None`` when unreadable."""
  try:
    return hashlib.sha256(path.read_bytes()).hexdigest()
  except OSError:
    return None


def normalize_tsplib_file(tsp_path: str | Path) -> NormalizedProblemData:
  """Parse and normalize one TSPLIB/VRP file through the ``routing_data`` pipeline."""
  path = Path(tsp_path)
  parser = FormatParser()
  normalizer = ProblemNormalizer()
  try:
    parsed = parser.parse_file(path)
    return normalizer.transform_problem(parsed, file_info={"file_path": str(path)})
  except Exception as exc:
    raise InvalidProblemDataError(path.stem, f"tsplib_parser failed to parse '{path}': {exc}") from exc


def bundle_from_normalized(
  tsp_path: Path,
  normalized: NormalizedProblemData,
  problem_id: int,
) -> TsplibIngestionBundle:
  """Build the ingestion bundle from a normalized problem payload."""
  path = Path(tsp_path)
  instance_name: str = path.stem
  problem_meta: dict[str, Any] = normalized["problem_data"]

  problem_type: str = str(problem_meta.get("type") or "TSP")
  dimension = int(problem_meta.get("dimension") or 0)
  edge_type = str(problem_meta.get("edge_weight_type") or "").upper().strip()
  capacity_raw: Any = problem_meta.get("capacity")
  capacity: int | None = int(capacity_raw) if capacity_raw is not None else None
  comment_raw: Any = problem_meta.get("comment")
  comment: str | None = str(comment_raw) if comment_raw is not None else None
  edge_format_raw: Any = problem_meta.get("edge_weight_format")
  edge_weight_format: str | None = str(edge_format_raw).strip() if edge_format_raw else None
  tsplib_name: str | None = str(problem_meta.get("name") or instance_name)

  if edge_type not in VALID_EDGE_TYPES:
    raise InvalidEdgeTypeError(edge_type)
  if dimension <= 0:
    raise InvalidProblemDataError(instance_name, f"Invalid dimension: {dimension}")

  coords: list[list[float]] | None = normalized["coords"]
  display_coords: list[list[float]] | None = normalized["display_coords"]
  demands: list[int] = normalized["demands"]
  depots: list[int] = normalized["depots"]
  depot_set: set[int] = set(depots)

  node_coords: list[list[float]] | None = coords if edge_type != "EXPLICIT" else display_coords
  if edge_type != "EXPLICIT":
    if not node_coords:
      raise MissingCoordinatesError(instance_name, edge_type)
    if len(node_coords) > dimension:
      raise InvalidProblemDataError(
        instance_name,
        f"Expected {dimension} coordinates for edge_type '{edge_type}', got {len(node_coords)}",
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

  expected_coord_dim: int = expected_coordinate_dimension(edge_type)
  nodes_records: list[NodeRowRecord] = []
  for zero_idx in range(dimension):
    x_val: float | None = None
    y_val: float | None = None
    z_val: float | None = None
    if node_coords is not None and zero_idx < len(node_coords):
      coord_len = len(node_coords[zero_idx])
      if coord_len != expected_coord_dim:
        raise InvalidProblemDataError(
          instance_name,
          f"edge_type '{edge_type}' requires {expected_coord_dim}D coordinates "
          f"(found coordinate of length {coord_len} at node {zero_idx})",
        )
      x_val = float(node_coords[zero_idx][0])
      y_val = float(node_coords[zero_idx][1])
      if expected_coord_dim == 3:
        z_val = float(node_coords[zero_idx][2])
      coords_tuple = (x_val, y_val, z_val) if z_val is not None else (x_val, y_val)
      if any(math.isnan(c) for c in coords_tuple):
        raise MissingCoordinatesError(instance_name, edge_type)
      if not all(math.isfinite(c) for c in coords_tuple):
        raise InvalidProblemDataError(
          instance_name,
          f"Non-finite coordinate at node {zero_idx} for edge_type '{edge_type}'",
        )
    if edge_type != "EXPLICIT" and x_val is None:
      raise MissingCoordinatesError(instance_name, edge_type)
    demand_val: int = int(demands[zero_idx]) if zero_idx < len(demands) else 0
    is_depot: bool = (zero_idx in depot_set) or (zero_idx == 0 and not depot_set)
    nodes_records.append(
      NodeRowRecord(
        problem_id=problem_id,
        node_id=zero_idx,
        x=x_val,
        y=y_val,
        z=z_val,
        demand=demand_val,
        is_depot=is_depot,
      )
    )

  matrix_record: MatrixRowRecord | None = None
  edge_weight_matrix: list[list[int]] | None = normalized.get("edge_weight_matrix")
  if edge_weight_matrix is not None:
    matrix_record = MatrixRowRecord(
      problem_id=problem_id,
      dimension=len(edge_weight_matrix),
      matrix_format=edge_weight_format or "FULL_MATRIX",
      is_symmetric=problem_type != "ATSP",
      matrix=edge_weight_matrix,
    )

  max_distance_raw: Any = problem_meta.get("max_distance")
  vehicles_raw: Any = problem_meta.get("vehicles")
  service_time_raw: Any = problem_meta.get("service_time")
  display_coords_tuple: tuple[tuple[float, float], ...] | None = (
    tuple((float(pt[0]), float(pt[1])) for pt in display_coords) if display_coords is not None else None
  )

  problem_record = ProblemRowRecord(
    problem_id=problem_id,
    name=instance_name,
    problem_type=problem_type,
    dimension=dimension,
    edge_weight_type=edge_type,
    capacity=capacity,
    comment=comment,
    edge_weight_format=edge_weight_format,
    tsplib_name=tsplib_name,
    source_file=str(path.resolve()),
    file_checksum=file_checksum(path),
    max_distance=float(max_distance_raw) if max_distance_raw is not None else None,
    vehicles=int(vehicles_raw) if vehicles_raw is not None else None,
    service_time=float(service_time_raw) if service_time_raw is not None else None,
    display_coords=display_coords_tuple,
  )

  return TsplibIngestionBundle(
    problem=problem_record,
    nodes=tuple(nodes_records),
    edge_matrix=matrix_record,
  )


def parse_tsplib_file(tsp_path: str | Path, problem_id: int = 1) -> TsplibIngestionBundle:
  """Parse a single `.tsp` or `.vrp` file into DuckDB-ready ingestion records."""
  path = Path(tsp_path)
  if not path.exists():
    raise InstanceNotFoundError(path.stem)
  normalized = normalize_tsplib_file(path)
  return bundle_from_normalized(path, normalized, problem_id)


__all__: list[str] = [
  "MatrixRowRecord",
  "NodeRowRecord",
  "ProblemRowRecord",
  "TsplibIngestionBundle",
  "bundle_from_normalized",
  "file_checksum",
  "normalize_tsplib_file",
  "parse_tsplib_file",
]
