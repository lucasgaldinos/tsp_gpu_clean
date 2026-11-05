"""
TSPLIB95 loader and DuckDB ingestion built on the ``routing_data`` package.

This module no longer parses TSPLIB files itself (``tsplib95`` is gone, ``DP10-6``):

* ``tsplib_parser.parser.FormatParser`` produces the normalized schema-v2 parse
  result (header metadata, 0-based array columns, explicit weight matrix).
* ``converter.core.transformer.ProblemNormalizer`` converts it into the
  storage payload consumed by ``converter.database.operations.DatabaseManager``.
* ``DatabaseManager`` owns the DuckDB schema, idempotent problem upserts and the
  satellite rows (``edge_weight_matrices`` / ``file_tracking``), replacing the
  ~250 lines of duplicated DDL that used to live here.

The public surface stays the same for callers: :class:`TsplibLoader` returns
validated :class:`Problem` instances, :func:`parse_tsplib_file` produces a
:class:`TsplibIngestionBundle`, and :func:`ingest_tsplib_directory_to_duckdb`
populates a DuckDB database (problems plus ``*.opt.tour`` solutions).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
from converter.core.solutions import parse_solution_file, solution_stem
from converter.core.transformer import NormalizedProblemData, ProblemNormalizer
from converter.database.operations import DatabaseManager
from converter.utils.cost import compute_routes_cost
from tsplib_parser.parser import FormatParser

from data_models.exceptions import (
  DatabaseNotEmptyError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
)
from data_models.problem import VALID_EDGE_TYPES, Problem
from distances.matrix import compute_distance_matrix

if TYPE_CHECKING:
  from types import TracebackType

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class NodeRowRecord:
  """Row schema for the `nodes` table in DuckDB (0-based `node_id`)."""

  problem_id: int
  node_id: int
  x: float | None
  y: float | None
  demand: int
  is_depot: bool


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
  """
  Complete structured ingestion payload produced from a `.tsp` file.

  Mirrors the exact relational records required by the DuckDB schema
  (`problems`, node array columns and `edge_weight_matrices`).
  """

  problem: ProblemRowRecord
  nodes: tuple[NodeRowRecord, ...]
  edge_matrix: MatrixRowRecord | None


def _file_checksum(path: Path) -> str | None:
  """Return the SHA-256 checksum of ``path`` or ``None`` when unreadable."""
  try:
    return hashlib.sha256(path.read_bytes()).hexdigest()
  except OSError:
    return None


def _normalize_file(tsp_path: str | Path) -> NormalizedProblemData:
  """Parse and normalize one TSPLIB file through the ``routing_data`` pipeline."""
  path = Path(tsp_path)
  parser = FormatParser()
  normalizer = ProblemNormalizer()
  try:
    parsed = parser.parse_file(path)
    return normalizer.transform_problem(parsed, file_info={"file_path": str(path)})
  except Exception as exc:
    raise InvalidProblemDataError(path.stem, f"tsplib_parser failed to parse '{path}': {exc}") from exc


def _bundle_from_normalized(
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
  if edge_type != "EXPLICIT" and not node_coords:
    raise MissingCoordinatesError(instance_name, edge_type)

  nodes_records: list[NodeRowRecord] = []
  for zero_idx in range(dimension):
    x_val: float | None = None
    y_val: float | None = None
    if node_coords is not None and zero_idx < len(node_coords) and len(node_coords[zero_idx]) >= 2:
      x_val = float(node_coords[zero_idx][0])
      y_val = float(node_coords[zero_idx][1])
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
    file_checksum=_file_checksum(path),
  )

  return TsplibIngestionBundle(
    problem=problem_record,
    nodes=tuple(nodes_records),
    edge_matrix=matrix_record,
  )


def parse_tsplib_file(tsp_path: str | Path, problem_id: int = 1) -> TsplibIngestionBundle:
  """
  Parse a single `.tsp` file into DuckDB-ready ingestion records.

  Args:
      tsp_path: Path to the `.tsp` file.
      problem_id: Integer identifier assigned to the problem record.

  Returns:
      `TsplibIngestionBundle` ready for DuckDB insertion or direct conversion
      into a `Problem` dataclass.

  Raises:
      InstanceNotFoundError: If `tsp_path` does not exist.
      InvalidEdgeTypeError: If the problem's `EDGE_WEIGHT_TYPE` is unsupported.
      MissingCoordinatesError: If coordinates are required but missing.
      InvalidProblemDataError: If parsing fails or data is inconsistent.
  """
  path = Path(tsp_path)
  if not path.exists():
    raise InstanceNotFoundError(path.stem)
  normalized = _normalize_file(path)
  return _bundle_from_normalized(path, normalized, problem_id)


def bundle_to_problem(
  bundle: TsplibIngestionBundle,
  max_eager_matrix_dimension: int = 4000,
) -> Problem:
  """
  Convert a `TsplibIngestionBundle` into a validated `Problem` domain model.

  Args:
      bundle: Parsed TSPLIB ingestion bundle.
      max_eager_matrix_dimension: Maximum dimension for eager O(N^2) distance
          matrix materialization on coordinate-based problems. Above this
          threshold, `distances` is left as `None` for lazy computation via
          `ProblemContext`.

  Returns:
      Immutable `Problem` instance.
  """
  meta: ProblemRowRecord = bundle.problem
  coordinates: np.ndarray | None = None
  demands: np.ndarray | None = None
  distances: np.ndarray | None = None

  display_coords: np.ndarray | None = None
  display_list: list[list[float]] = []
  depots_list: list[int] = []

  if meta.edge_weight_type == "EXPLICIT":
    if bundle.edge_matrix is None or bundle.edge_matrix.matrix is None:
      raise InvalidProblemDataError(meta.name, "EXPLICIT edge_type but no matrix in edge_matrix record")
    distances = np.asarray(bundle.edge_matrix.matrix, dtype=np.int32)

    for node in bundle.nodes:
      if node.x is not None and node.y is not None:
        display_list.append([node.x, node.y])
      if node.is_depot:
        depots_list.append(node.node_id)
    if len(display_list) == meta.dimension:
      display_coords = np.asarray(display_list, dtype=np.float64)
  else:
    coords_list: list[list[float]] = []
    demands_list: list[int] = []
    for node in bundle.nodes:
      if node.x is None or node.y is None:
        raise MissingCoordinatesError(meta.name, meta.edge_weight_type)
      coords_list.append([node.x, node.y])
      demands_list.append(node.demand)
      if node.is_depot:
        depots_list.append(node.node_id)

    coordinates = np.asarray(coords_list, dtype=np.float64)
    if meta.dimension <= max_eager_matrix_dimension:
      distances = compute_distance_matrix(coordinates, meta.edge_weight_type)

    if meta.problem_type == "CVRP" or any(demand > 0 for demand in demands_list):
      demands = np.asarray(demands_list, dtype=np.int32)

  try:
    return Problem(
      name=meta.name,
      dimension=meta.dimension,
      problem_type=meta.problem_type,
      edge_type=meta.edge_weight_type,
      coordinates=coordinates,
      distances=distances,
      capacity=meta.capacity,
      demands=demands,
      problem_id=meta.problem_id,
      comment=meta.comment,
      edge_weight_format=meta.edge_weight_format,
      tsplib_name=meta.tsplib_name or meta.name,
      display_coords=display_coords,
      depots=np.asarray(depots_list, dtype=np.int32) if depots_list else None,
      source_file=meta.source_file,
      file_checksum=meta.file_checksum,
    )
  except ValueError as exc:
    raise InvalidProblemDataError(meta.name, str(exc)) from exc


class TsplibLoader:
  """
  Context-managed loader that parses `.tsp` files into `Problem` objects.

  Acts as a drop-in temporary replacement for `DatabaseLoader` while the DuckDB
  database is not yet built.
  """

  def __init__(self, tsp_dir: str | Path) -> None:
    """
    Initialize the TSPLIB directory loader.

    Args:
        tsp_dir: Directory containing `.tsp` instance files.
    """
    self.tsp_dir = Path(tsp_dir)
    self._active: bool = False

  def __enter__(self) -> TsplibLoader:
    """Enter context manager and verify directory existence."""
    if not self.tsp_dir.exists():
      raise FileNotFoundError(f"TSPLIB directory not found: {self.tsp_dir}")
    self._active = True
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Exit context manager."""
    self._active = False

  def load(self, instance_name: str | Path) -> Problem:
    """
    Load a TSP problem instance by name (e.g., `'berlin52'`) or path.

    Args:
        instance_name: Problem name (with or without `.tsp` suffix) or `Path`.

    Returns:
        Validated `Problem` instance.
    """
    if isinstance(instance_name, Path) or "/" in str(instance_name):
      candidate = Path(instance_name)
      if candidate.exists():
        return bundle_to_problem(parse_tsplib_file(candidate))
      instance_str: str = candidate.name
    else:
      instance_str = str(instance_name)

    stem: str = instance_str[:-4] if instance_str.endswith(".tsp") else instance_str
    tsp_path: Path = self.tsp_dir / f"{stem}.tsp"
    if not tsp_path.exists():
      raise InstanceNotFoundError(stem)

    return bundle_to_problem(parse_tsplib_file(tsp_path))

  def list_instances(self) -> list[str]:
    """Return sorted list of available `.tsp` problem names in `tsp_dir`."""
    if not self.tsp_dir.exists():
      return []
    return sorted(path.stem for path in self.tsp_dir.glob("*.tsp"))


def _problem_payload(
  tsp_file: Path,
  normalized: NormalizedProblemData,
) -> dict[str, Any]:
  """Build the flat schema-v2 payload `DatabaseManager.insert_problem` consumes."""
  problem_meta: dict[str, Any] = normalized["problem_data"]
  problem_type: str = str(problem_meta.get("type") or "TSP")
  edge_type = str(problem_meta.get("edge_weight_type") or "").upper().strip()
  dimension = int(problem_meta.get("dimension") or 0)
  capacity_raw: Any = problem_meta.get("capacity")

  coords: list[list[float]] | None = normalized["coords"]
  display_coords: list[list[float]] | None = normalized["display_coords"]
  if edge_type == "EXPLICIT":
    display_coords = coords or display_coords
    coords = None

  edge_weight_matrix: list[list[int]] | None = normalized.get("edge_weight_matrix")
  edge_weight_data: dict[str, Any] | None = None
  if edge_weight_matrix is not None:
    symmetry: Any = None
    edge_weight_data = {
      "matrix": edge_weight_matrix,
      "matrix_format": problem_meta.get("edge_weight_format") or "FULL_MATRIX",
      "is_symmetric": problem_type != "ATSP" if symmetry is None else bool(symmetry),
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
    "adjacency": normalized["edges"],
    "fixed_edges": normalized["fixed_edges"],
    "edge_weight_data": edge_weight_data,
    "file_path": str(tsp_file.resolve()),
    "checksum": _file_checksum(tsp_file),
    "file_size": tsp_file.stat().st_size if tsp_file.exists() else 0,
  }


def _ingest_opt_tour_solutions(db_path: Path, source_dir: Path) -> None:
  """Insert every ``*.opt.tour`` next to the instances into the ``solutions`` table."""
  tour_dirs: tuple[Path, ...] = tuple(
    directory for directory in (source_dir, source_dir / "tour") if directory.is_dir()
  )
  tour_files: list[Path] = sorted(path for directory in tour_dirs for path in directory.glob("*.opt.tour"))
  if not tour_files:
    return

  with duckdb.connect(str(db_path)) as conn:
    for tour_file in tour_files:
      try:
        parsed = parse_solution_file(tour_file)
      except Exception as exc:
        logger.warning("Skipping unreadable solution file '%s': %s", tour_file, exc)
        continue

      instance_name: str = solution_stem(tour_file)
      row: tuple[Any, ...] | None = conn.execute(
        "SELECT p.id, t.edge_weight_type, t.coords FROM problems p "
        "LEFT JOIN tsp_problems t ON p.id = t.problem_id "
        "WHERE p.name = ?",
        [instance_name],
      ).fetchone()
      if row is None:
        continue

      problem_id, edge_type, coords = row
      matrix_row: tuple[Any, ...] | None = conn.execute(
        "SELECT matrix FROM edge_weight_matrices WHERE problem_id = ?",
        [problem_id],
      ).fetchone()
      matrix: Any = matrix_row[0] if matrix_row is not None else None

      cost: float | None = parsed["cost"]
      if cost is None:
        cost = compute_routes_cost(
          parsed["routes"],
          matrix=matrix,
          coords=coords,
          edge_weight_type=str(edge_type) if edge_type else None,
        )

      conn.execute(
        """
        INSERT INTO solutions (problem_id, kind, quality, source_file, checksum, cost, routes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (source_file) DO UPDATE SET
          problem_id = EXCLUDED.problem_id,
          checksum = EXCLUDED.checksum,
          cost = EXCLUDED.cost,
          routes = EXCLUDED.routes
        """,
        [
          problem_id,
          parsed["kind"],
          parsed["quality"],
          str(tour_file.resolve()),
          _file_checksum(tour_file),
          cost,
          parsed["routes"],
        ],
      )


def ingest_tsplib_directory_to_duckdb(
  tsp_dir: str | Path,
  db_path: str | Path,
  *,
  overwrite: bool = False,
) -> int:
  """
  Parse all supported `.tsp` files in `tsp_dir` and ingest them into DuckDB.

  The schema, idempotent problem upserts and satellite rows are owned by
  ``converter.database.operations.DatabaseManager`` (``DP10-14``); this function
  only walks the directory, normalizes each file, and also ingests
  ``*.opt.tour`` solutions found next to the instances.

  Args:
      tsp_dir: Directory containing `.tsp` files.
      db_path: Output DuckDB file path.
      overwrite: Whether to recreate the database when it already holds rows.

  Returns:
      Number of successfully ingested problem instances.

  Raises:
      FileNotFoundError: If `tsp_dir` does not exist.
      DatabaseNotEmptyError: If the database already contains records and
          `overwrite` is False.
  """
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
  for tsp_file in sorted(source_dir.glob("*.tsp")):
    try:
      normalized = _normalize_file(tsp_file)
      manager.insert_problem(_problem_payload(tsp_file, normalized))
    except InvalidEdgeTypeError as exc:
      # Skip unsupported experimental edge types such as XRAY1/XRAY2.
      logger.warning("Skipping '%s': %s", tsp_file, exc)
      continue
    except Exception as exc:
      # Keep the batch going: a malformed file must not abort the directory.
      logger.warning("Skipping malformed instance '%s': %s", tsp_file, exc)
      continue
    ingested += 1

  _ingest_opt_tour_solutions(target_db, source_dir)
  return ingested


__all__ = [
  "MatrixRowRecord",
  "NodeRowRecord",
  "ProblemRowRecord",
  "TsplibIngestionBundle",
  "TsplibLoader",
  "bundle_to_problem",
  "ingest_tsplib_directory_to_duckdb",
  "parse_tsplib_file",
]
