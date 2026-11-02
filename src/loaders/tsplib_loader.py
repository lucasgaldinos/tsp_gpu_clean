"""
TSPLIB95 loader and temporary ingestion module for routing problem instances.

This module parses `.tsp` files using `tsplib95` into:
1. Standardized ingestion records (`TsplibIngestionBundle`) matching the
   DuckDB database schema (`problems`, `nodes`, `edge_weight_matrices`).
2. Immutable `Problem` domain objects via `TsplibLoader`, providing a
   drop-in placeholder while the DuckDB database is not yet populated.
3. A CLI ingestion entrypoint (`python -m loaders.tsplib_loader`) that
   parses `datasets/tsp/*.tsp` and optionally ingests them into DuckDB.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import tsplib95

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
  Complete structured ingestion payload produced from a `.tsp` file via `tsplib95`.

  Mirrors the exact relational records required by `DatabaseLoader` (`problems`,
  `nodes`, and `edge_weight_matrices`).
  """

  problem: ProblemRowRecord
  nodes: tuple[NodeRowRecord, ...]
  edge_matrix: MatrixRowRecord | None


def parse_tsp_file_with_tsplib95(
  tsp_path: str | Path,
  problem_id: int = 1,
) -> TsplibIngestionBundle:
  """
  Parse a single `.tsp` file using `tsplib95` into DuckDB-ready ingestion records.

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
      InvalidProblemDataError: If `tsplib95` fails to parse or data is inconsistent.
  """
  path = Path(tsp_path)
  if not path.exists():
    raise InstanceNotFoundError(path.stem)

  try:
    raw_problem: tsplib95.Problem = tsplib95.load_problem(str(path))
  except Exception as exc:
    raise InvalidProblemDataError(path.stem, f"tsplib95 failed to parse '{path}': {exc}") from exc

  instance_name: str = path.stem
  raw_type_full: str = str(getattr(raw_problem, "type", "TSP") or "TSP").upper().strip()
  raw_type: str = raw_type_full.split()[0] if raw_type_full else "TSP"
  problem_type: str = "CVRP" if raw_type == "VRP" else raw_type
  dimension = int(getattr(raw_problem, "dimension", 0) or 0)
  edge_type = str(getattr(raw_problem, "edge_weight_type", "") or "").upper().strip()
  capacity_raw: Any | None = getattr(raw_problem, "capacity", None)
  capacity: int | None = int(capacity_raw) if capacity_raw is not None else None
  comment_raw: Any | None = getattr(raw_problem, "comment", None)
  comment: str | None = str(comment_raw) if comment_raw is not None else None
  edge_format_raw: Any | None = getattr(raw_problem, "edge_weight_format", None)
  edge_weight_format: str | None = str(edge_format_raw).strip() if edge_format_raw else None
  tsplib_name: str | None = str(getattr(raw_problem, "name", "") or instance_name)

  checksum: str | None = None
  try:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
  except Exception:
    checksum = None
  source_file: str = str(path.resolve())

  if edge_type not in VALID_EDGE_TYPES:
    raise InvalidEdgeTypeError(edge_type)

  if dimension <= 0:
    raise InvalidProblemDataError(instance_name, f"Invalid dimension: {dimension}")

  node_coords: dict[Any, Any] = getattr(raw_problem, "node_coords", None) or {}
  demands_map: dict[Any, Any] = getattr(raw_problem, "demands", None) or {}
  depots_list: list[Any] = list(getattr(raw_problem, "depots", None) or [])
  depot_set = set(depots_list)

  nodes_records: list[NodeRowRecord] = []
  matrix_record: MatrixRowRecord | None = None

  if edge_type == "EXPLICIT":
    raw_nodes = list(raw_problem.get_nodes())
    if not raw_nodes:
      raw_nodes = list(range(1, dimension + 1))
    if len(raw_nodes) != dimension:
      raise InvalidProblemDataError(
        instance_name,
        f"EXPLICIT node count mismatch: declared={dimension}, actual={len(raw_nodes)}",
      )

    matrix_rows: list[list[float]] = []
    for u in raw_nodes:
      row_vals: list[float] = []
      for v in raw_nodes:
        weight: float = 0.0 if u == v else float(raw_problem.wfunc(u, v))
        row_vals.append(weight)
      matrix_rows.append(row_vals)

    matrix_format = str(getattr(raw_problem, "edge_weight_format", None) or "FULL_MATRIX")
    is_symmetric: bool = problem_type != "ATSP"
    int_matrix: list[list[int]] = [[round(w) for w in row] for row in matrix_rows]
    matrix_record = MatrixRowRecord(
      problem_id=problem_id,
      dimension=dimension,
      matrix_format=matrix_format,
      is_symmetric=is_symmetric,
      matrix_json=json.dumps(matrix_rows),
      matrix=int_matrix,
    )

    display_data: dict[Any, Any] = getattr(raw_problem, "display_data", None) or {}
    for zero_idx, node_key in enumerate(raw_nodes):
      coord: Any | None = display_data.get(node_key)
      x_val: float | None = float(coord[0]) if coord is not None and len(coord) >= 2 else None
      y_val: float | None = float(coord[1]) if coord is not None and len(coord) >= 2 else None
      demand_val = int(demands_map.get(node_key, 0) or 0)
      is_depot: bool = (node_key in depot_set) or (zero_idx == 0 and not depot_set)
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
  else:
    if not node_coords:
      raise MissingCoordinatesError(instance_name, edge_type)

    sorted_keys = sorted(node_coords.keys())
    if len(sorted_keys) != dimension:
      raise InvalidProblemDataError(
        instance_name,
        f"Dimension mismatch: declared={dimension}, actual={len(sorted_keys)}",
      )

    for zero_idx, node_key in enumerate(sorted_keys):
      coord = node_coords[node_key]
      if coord is None or len(coord) < 2:
        raise MissingCoordinatesError(instance_name, edge_type)
      x_val = float(coord[0])
      y_val = float(coord[1])
      demand_val = int(demands_map.get(node_key, 0) or 0)
      is_depot = (node_key in depot_set) or (zero_idx == 0 and not depot_set)
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
    source_file=source_file,
    file_checksum=checksum,
  )

  return TsplibIngestionBundle(
    problem=problem_record,
    nodes=tuple(nodes_records),
    edge_matrix=matrix_record,
  )


def bundle_to_problem(
  bundle: TsplibIngestionBundle,
  max_eager_matrix_dimension: int = 4000,
) -> Problem:
  """
  Convert a `TsplibIngestionBundle` into a validated `Problem` domain model.

  Uses the exact same transformation logic as `DatabaseLoader._construct_problem`.

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
    if bundle.edge_matrix is None:
      raise InvalidProblemDataError(
        meta.name,
        "EXPLICIT edge_type but no matrix in edge_matrix record",
      )
    if bundle.edge_matrix.matrix is not None:
      distances = np.array(bundle.edge_matrix.matrix, dtype=np.float64)
    elif bundle.edge_matrix.matrix_json:
      matrix_list = json.loads(bundle.edge_matrix.matrix_json)
      distances = np.array(matrix_list, dtype=np.float64)
    else:
      raise InvalidProblemDataError(
        meta.name,
        "EXPLICIT edge_type but no matrix data in edge_matrix record",
      )

    for node in bundle.nodes:
      if node.x is not None and node.y is not None:
        display_list.append([node.x, node.y])
      if node.is_depot:
        depots_list.append(node.node_id)
    if len(display_list) == meta.dimension:
      display_coords = np.array(display_list, dtype=np.float64)
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

    coordinates = np.array(coords_list, dtype=np.float64)
    if meta.dimension <= max_eager_matrix_dimension:
      distances = compute_distance_matrix(coordinates, meta.edge_weight_type)

    if meta.problem_type == "CVRP" or any(d > 0 for d in demands_list):
      demands = np.array(demands_list, dtype=np.int32)

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
      depots=np.array(depots_list, dtype=np.int32) if depots_list else None,
      source_file=meta.source_file,
      file_checksum=meta.file_checksum,
    )
  except ValueError as exc:
    raise InvalidProblemDataError(meta.name, str(exc)) from exc


class TsplibLoader:
  """
  Context-managed loader that parses `.tsp` files using `tsplib95` into `Problem` objects.

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
    Load a TSP problem instance by name (e.g., `'berlin52'`) or path using `tsplib95`.

    Args:
        instance_name: Problem name (with or without `.tsp` suffix) or `Path` to `.tsp` file.

    Returns:
        Validated `Problem` instance.
    """
    if isinstance(instance_name, Path) or "/" in str(instance_name):
      candidate = Path(instance_name)
      if candidate.exists():
        bundle: TsplibIngestionBundle = parse_tsp_file_with_tsplib95(candidate)
        return bundle_to_problem(bundle)
      instance_str: str = candidate.name
    else:
      instance_str = str(instance_name)

    stem: str = instance_str[:-4] if instance_str.endswith(".tsp") else instance_str
    tsp_path: Path = self.tsp_dir / f"{stem}.tsp"
    if not tsp_path.exists():
      raise InstanceNotFoundError(stem)

    bundle = parse_tsp_file_with_tsplib95(tsp_path)
    return bundle_to_problem(bundle)

  def list_instances(self) -> list[str]:
    """Return sorted list of available `.tsp` problem names in `tsp_dir`."""
    if not self.tsp_dir.exists():
      return []
    return sorted(p.stem for p in self.tsp_dir.glob("*.tsp"))


def ingest_tsplib_directory_to_duckdb(
  tsp_dir: str | Path,
  db_path: str | Path,
  *,
  overwrite: bool = False,
) -> int:
  """
  Parse all supported `.tsp` files in `tsp_dir` with `tsplib95` and ingest into DuckDB.

  Creates the normalized schema tables matching the DuckDB target schema:
  `problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`,
  `file_tracking`, and `solutions`.

  Args:
      tsp_dir: Directory containing `.tsp` files.
      db_path: Output DuckDB file path.
      overwrite: Whether to overwrite/clear existing database tables if not empty.

  Returns:
      Number of successfully ingested problem instances.

  Raises:
      DatabaseNotEmptyError: If database already contains records and `overwrite` is False.
  """
  source_dir = Path(tsp_dir)
  if not source_dir.exists() or not source_dir.is_dir():
    raise FileNotFoundError(f"TSPLIB source directory not found: {tsp_dir}")
  target_db = Path(db_path)
  target_db.parent.mkdir(parents=True, exist_ok=True)

  conn: duckdb.DuckDBPyConnection = duckdb.connect(str(target_db))
  try:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY,
        name VARCHAR UNIQUE NOT NULL,
        type VARCHAR NOT NULL,
        has_solution BOOLEAN NOT NULL DEFAULT false,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS tsp_problems (
        problem_id INTEGER PRIMARY KEY,
        dimension INTEGER NOT NULL,
        comment VARCHAR,
        edge_weight_type VARCHAR NOT NULL,
        edge_weight_format VARCHAR,
        tsplib_name VARCHAR,
        coords DOUBLE[][],
        display_coords DOUBLE[][]
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS atsp_problems (
        problem_id INTEGER PRIMARY KEY,
        dimension INTEGER NOT NULL,
        comment VARCHAR,
        tsplib_name VARCHAR
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS cvrp_problems (
        problem_id INTEGER PRIMARY KEY,
        dimension INTEGER NOT NULL,
        capacity INTEGER NOT NULL,
        comment VARCHAR,
        edge_weight_type VARCHAR NOT NULL,
        edge_weight_format VARCHAR,
        tsplib_name VARCHAR,
        coords DOUBLE[][],
        demands INTEGER[],
        depots INTEGER[]
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS edge_weight_matrices (
        problem_id INTEGER PRIMARY KEY,
        matrix_format VARCHAR NOT NULL,
        is_symmetric BOOLEAN NOT NULL,
        matrix INTEGER[][] NOT NULL
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS file_tracking (
        id INTEGER PRIMARY KEY,
        file_path VARCHAR UNIQUE NOT NULL,
        problem_id INTEGER,
        checksum VARCHAR,
        last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_size BIGINT
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS solutions (
        id INTEGER PRIMARY KEY,
        problem_id INTEGER,
        kind VARCHAR NOT NULL,
        quality VARCHAR NOT NULL,
        source_file VARCHAR UNIQUE NOT NULL,
        checksum VARCHAR,
        cost DOUBLE,
        routes INTEGER[][],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """
    )

    existing_count_row = conn.execute("SELECT COUNT(*) FROM problems").fetchone()
    existing_count = existing_count_row[0] if existing_count_row else 0
    if existing_count > 0:
      if not overwrite:
        raise DatabaseNotEmptyError(str(target_db))
      conn.execute("DELETE FROM solutions")
      conn.execute("DELETE FROM file_tracking")
      conn.execute("DELETE FROM edge_weight_matrices")
      conn.execute("DELETE FROM cvrp_problems")
      conn.execute("DELETE FROM atsp_problems")
      conn.execute("DELETE FROM tsp_problems")
      conn.execute("DELETE FROM problems")

    ingested = 0
    for idx, tsp_file in enumerate(sorted(source_dir.glob("*.tsp")), start=1):
      try:
        bundle: TsplibIngestionBundle = parse_tsp_file_with_tsplib95(tsp_file, problem_id=idx)
      except InvalidEdgeTypeError:
        # Skip unsupported experimental edge types such as XRAY1/XRAY2
        continue

      p: ProblemRowRecord = bundle.problem
      conn.execute(
        """
        INSERT INTO problems (id, name, type, has_solution)
        VALUES (?, ?, ?, false)
        """,
        [p.problem_id, p.name, p.problem_type],
      )

      coords_list: list[list[float]] | None = None
      display_coords_list: list[list[float]] | None = None
      demands_list: list[int] = []
      depots_list: list[int] = []

      for node in bundle.nodes:
        if node.x is not None and node.y is not None:
          if coords_list is None:
            coords_list = []
          coords_list.append([node.x, node.y])
        demands_list.append(node.demand)
        if node.is_depot:
          depots_list.append(node.node_id)

      if p.edge_weight_type == "EXPLICIT":
        display_coords_list = coords_list
        coords_list = None

      if p.problem_type == "TSP":
        conn.execute(
          """
          INSERT INTO tsp_problems (
            problem_id, dimension, comment, edge_weight_type, edge_weight_format, tsplib_name, coords, display_coords
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          """,
          [
            p.problem_id,
            p.dimension,
            p.comment,
            p.edge_weight_type,
            p.edge_weight_format,
            p.tsplib_name or p.name,
            coords_list,
            display_coords_list,
          ],
        )
      elif p.problem_type == "ATSP":
        conn.execute(
          """
          INSERT INTO atsp_problems (problem_id, dimension, comment, tsplib_name)
          VALUES (?, ?, ?, ?)
          """,
          [p.problem_id, p.dimension, p.comment, p.tsplib_name or p.name],
        )
      elif p.problem_type == "CVRP":
        conn.execute(
          """
          INSERT INTO cvrp_problems (
            problem_id, dimension, capacity, comment, edge_weight_type, edge_weight_format,
            tsplib_name, coords, demands, depots
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          [
            p.problem_id,
            p.dimension,
            p.capacity or 0,
            p.comment,
            p.edge_weight_type,
            p.edge_weight_format,
            p.tsplib_name or p.name,
            coords_list,
            demands_list,
            depots_list,
          ],
        )

      if bundle.edge_matrix is not None:
        m: MatrixRowRecord = bundle.edge_matrix
        mat_vals = m.matrix
        if mat_vals is None and m.matrix_json:
          raw_json = json.loads(m.matrix_json)
          mat_vals = [[round(w) for w in row] for row in raw_json]
        conn.execute(
          """
          INSERT INTO edge_weight_matrices (problem_id, matrix_format, is_symmetric, matrix)
          VALUES (?, ?, ?, ?)
          """,
          [m.problem_id, m.matrix_format, m.is_symmetric, mat_vals],
        )

      file_size = tsp_file.stat().st_size if tsp_file.exists() else None
      conn.execute(
        """
        INSERT INTO file_tracking (id, file_path, problem_id, checksum, file_size)
        VALUES (?, ?, ?, ?, ?)
        """,
        [idx, p.source_file or str(tsp_file.resolve()), p.problem_id, p.file_checksum, file_size],
      )

      ingested += 1

    return ingested
  finally:
    conn.close()
