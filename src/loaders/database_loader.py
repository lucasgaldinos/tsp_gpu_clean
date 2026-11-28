"""
Database loader for routing problem instances.

This module provides a context manager for loading problem instances from
the DuckDB database (`datasets/db/routing.duckdb`). It handles database connections, SQL queries,
data transformation, and error handling with domain-specific exceptions.

Database Schema (datasets/db/routing.duckdb):
    - problems: Metadata (id, name, type, has_solution, created_at, updated_at)
    - tsp_problems: TSP data (problem_id, dimension, comment, edge_weight_type,
      edge_weight_format, tsplib_name, coords, display_coords)
    - atsp_problems: ATSP data (problem_id, dimension, comment, tsplib_name)
    - cvrp_problems: CVRP data (problem_id, dimension, capacity, comment, edge_weight_type,
      edge_weight_format, tsplib_name, coords, demands, depots)
    - edge_weight_matrices: Distance matrices (problem_id, matrix_format, is_symmetric, matrix INTEGER[][])
    - file_tracking: Ingestion files (id, file_path, problem_id, checksum, last_processed, file_size)
    - solutions: Known solutions (id, problem_id, kind, quality, source_file, checksum, cost, routes)

Usage:
    >>> with DatabaseLoader("<path/to/file.duckdb>") as loader:
    ...   problem = loader.load("berlin52")
    ...   print(f"{problem.name}: {problem.dimension} nodes")
    berlin52: 52 nodes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np

from data_models.exceptions import (
  AmbiguousProblemError,
  DatabaseConnectionError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
  UnsupportedProblemTypeError,
)
from data_models.problem import VALID_EDGE_TYPES, Problem, expected_coordinate_dimension
from protocols.problem_context import DEFAULT_MAX_MATRIX_DIMENSION

from .problem_type_handlers import (
  PROBLEM_TYPE_HANDLERS,
  BaseProblemRow,
  ProblemMetadataRow,
  ProblemTypeHandler,
  build_vrp_constraints,
  validate_single_depot,
)

if TYPE_CHECKING:
  from collections.abc import Mapping
  from types import TracebackType

  from .tsplib_loader import TsplibLoader


class DatabaseLoader:
  """
  Context manager for loading routing problems from DuckDB database.

  This class provides a clean interface for loading TSP, ATSP, and CVRP
  instances from `datasets/db/routing.duckdb`. It handles connection lifecycle,
  data parsing, and validation with comprehensive error handling.
  """

  def __init__(
    self,
    db_path: str | Path,
    fallback_tsp_dir: str | Path | None = None,
    handlers: Mapping[str, ProblemTypeHandler] | None = None,
    max_eager_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
  ) -> None:
    """
    Initialize database loader with path to DuckDB file.

    Args:
        db_path: Path to DuckDB database file.
        fallback_tsp_dir: Optional directory with `.tsp` files to use via
            `TsplibLoader` when the database file has not been created yet.
        handlers: Optional mapping from problem formulation type to handler instance.
            Defaults to PROBLEM_TYPE_HANDLERS.
        max_eager_matrix_dimension: Maximum dimension for eager O(N^2) distance matrix
            materialization on coordinate-based problems; above it `distances` stays
            `None` and `ProblemContext` computes lazily (`DP11-3`).
    """
    self.db_path = str(db_path)
    self.fallback_tsp_dir = Path(fallback_tsp_dir) if fallback_tsp_dir is not None else None
    self.max_eager_matrix_dimension: int = max_eager_matrix_dimension
    self.conn: duckdb.DuckDBPyConnection | None = None
    self._tsplib_fallback: TsplibLoader | None = None
    self.handlers: dict[str, ProblemTypeHandler] = (
      dict(handlers) if handlers is not None else dict(PROBLEM_TYPE_HANDLERS)
    )

  def __enter__(self) -> DatabaseLoader:
    """
    Enter context manager and establish database connection.

    Returns:
        Self for use in with statement.

    Raises:
        DatabaseConnectionError: If database file doesn't exist or connection fails.
    """
    db_file = Path(self.db_path)
    if not db_file.exists():
      if self.fallback_tsp_dir is not None and self.fallback_tsp_dir.exists():
        from .tsplib_loader import TsplibLoader

        self._tsplib_fallback = TsplibLoader(self.fallback_tsp_dir).__enter__()
        return self
      raise DatabaseConnectionError(self.db_path, FileNotFoundError(f"Database file does not exist: {self.db_path}"))

    try:
      self.conn = duckdb.connect(str(db_file), read_only=True)
      return self
    except Exception as e:
      raise DatabaseConnectionError(self.db_path, e) from e

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Exit context manager and close database connection."""
    if self._tsplib_fallback is not None:
      self._tsplib_fallback.__exit__(exc_type, exc_val, exc_tb)
      self._tsplib_fallback = None
    if self.conn is not None:
      self.conn.close()
      self.conn = None

  def list_problems(self, problem_type: str = "TSP") -> list[dict[str, Any]]:
    """
    Return list of problem metadata summaries ordered by dimension and name.

    Args:
        problem_type: Target problem type ('TSP', 'ATSP', 'CVRP').

    Returns:
        List of dictionaries with metadata (name, dimension, edge_weight_type, etc.).
    """
    if self._tsplib_fallback is not None:
      names: list[str] = self._tsplib_fallback.list_instances()
      return [{"name": name, "dimension": 0, "edge_weight_type": "UNKNOWN"} for name in names]

    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    handler = self.handlers.get(problem_type)
    if handler is None:
      raise UnsupportedProblemTypeError(problem_type)
    return handler.list_problems(self.conn)

  def load(
    self,
    instance_name: str,
    problem_type: str | None = None,
    *,
    capacity: int | None = None,
    max_vehicles: int | None = None,
    max_route_distance: float | None = None,
    return_to_depot: bool | None = None,
  ) -> Problem:
    """
    Load a problem instance from the DuckDB database (or `TsplibLoader` fallback).

    When `problem_type` is `'OVRP'` or `'DCVRP'` and no exact row with that type
    exists in `problems`, falls back unambiguously to the matching `'CVRP'` row
    and derives the requested variant (`BUG-111`, `DP17-10`).
    """
    if self._tsplib_fallback is not None:
      return self._tsplib_fallback.load(
        instance_name,
        problem_type=problem_type,
        capacity=capacity,
        max_vehicles=max_vehicles,
        max_route_distance=max_route_distance,
        return_to_depot=return_to_depot,
      )

    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    metadata = self._load_metadata(instance_name, problem_type=problem_type)
    coordinates, demands, display_coords, depots = self._load_coordinates(metadata)
    distances, matrix_format, _is_symmetric = self._load_distance_matrix(metadata)
    metadata["edge_weight_format"] = metadata["edge_weight_format"] or matrix_format

    return self._construct_problem(
      metadata=metadata,
      coordinates=coordinates,
      demands=demands,
      distances=distances,
      display_coords=display_coords,
      depots=depots,
      capacity_override=capacity,
      max_vehicles_override=max_vehicles,
      max_route_distance_override=max_route_distance,
      return_to_depot_override=return_to_depot,
    )

  def _resolve_problem_row(
    self,
    instance_name: str,
    *,
    problem_type: str | None = None,
  ) -> tuple[tuple[Any, ...], bool]:
    """Resolve one row from `problems LEFT JOIN file_tracking` with `(name, type)` disambiguation.

    Returns ``(row, is_derived_variant)``. When ``problem_type`` is ``OVRP`` or ``DCVRP``
    and no exact row exists, falls back to the stored ``CVRP`` row and marks
    ``is_derived_variant=True`` so closed-CVRP solutions are never reused as OVRP/DCVRP BKS.
    """
    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    if problem_type is not None:
      normalized_type = problem_type.upper().strip()
      row = self.conn.execute(
        """
        SELECT p.id, p.name, p.type, p.has_solution, ft.file_path, ft.checksum
        FROM problems p
        LEFT JOIN file_tracking ft ON p.id = ft.problem_id
        WHERE p.name = ? AND p.type = ?
        """,
        [instance_name, normalized_type],
      ).fetchone()
      if row is not None:
        return row, False
      if normalized_type in ("OVRP", "DCVRP"):
        cvrp_row = self.conn.execute(
          """
          SELECT p.id, p.name, p.type, p.has_solution, ft.file_path, ft.checksum
          FROM problems p
          LEFT JOIN file_tracking ft ON p.id = ft.problem_id
          WHERE p.name = ? AND p.type = 'CVRP'
          """,
          [instance_name],
        ).fetchone()
        if cvrp_row is not None:
          return cvrp_row, True
      raise InstanceNotFoundError(instance_name)

    rows = self.conn.execute(
      """
      SELECT p.id, p.name, p.type, p.has_solution, ft.file_path, ft.checksum
      FROM problems p
      LEFT JOIN file_tracking ft ON p.id = ft.problem_id
      WHERE p.name = ?
      ORDER BY p.id ASC
      """,
      [instance_name],
    ).fetchall()
    if not rows:
      raise InstanceNotFoundError(instance_name)
    if len(rows) > 1:
      raise AmbiguousProblemError(instance_name, [str(r[2]) for r in rows])
    return rows[0], False

  def load_optimal_cost(
    self,
    instance_name: str,
    problem_type: str | None = None,
  ) -> float | None:
    """Return the best canonical solution cost stored for `(instance_name, problem_type)`.

    Returns ``None`` when ``problem_type`` (e.g. ``OVRP``) is derived from a stored
    closed ``CVRP`` row so closed-CVRP BKS values are never misapplied (`DP17-6`, `BUG-111`).
    """
    if self._tsplib_fallback is not None:
      return self._tsplib_fallback.resolve_optimal_cost(instance_name, problem_type=problem_type)

    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    row, is_derived = self._resolve_problem_row(instance_name, problem_type=problem_type)
    if is_derived:
      return None
    problem_id = int(row[0])

    has_solutions_row = self.conn.execute(
      "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'solutions'"
    ).fetchone()
    if has_solutions_row is None or int(has_solutions_row[0]) == 0:
      return None

    sol_row = self.conn.execute(
      """
      SELECT cost
      FROM solutions
      WHERE problem_id = ?
        AND cost IS NOT NULL
      ORDER BY CASE quality
        WHEN 'proven_optimal' THEN 0
        WHEN 'best_known' THEN 1
        ELSE 2
      END ASC,
      id ASC
      LIMIT 1
      """,
      [problem_id],
    ).fetchone()
    return float(sol_row[0]) if sol_row is not None else None

  def _load_metadata(
    self,
    instance_name: str,
    *,
    problem_type: str | None = None,
  ) -> ProblemMetadataRow:
    """Query and validate problem metadata across normalized tables."""
    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    row, is_derived = self._resolve_problem_row(instance_name, problem_type=problem_type)

    problem_id, name, ptype, has_solution, source_file, file_checksum = row
    effective_type = problem_type.upper().strip() if (is_derived and problem_type is not None) else ptype

    handler = self.handlers.get(effective_type)
    if handler is None:
      raise UnsupportedProblemTypeError(effective_type, instance_name=name)

    base = BaseProblemRow(
      problem_id=problem_id,
      name=name,
      problem_type=effective_type,
      has_solution=False if is_derived else bool(has_solution),
      source_file=source_file,
      file_checksum=file_checksum,
    )
    metadata = handler.load_metadata(self.conn, base)
    edge_type = metadata["edge_type"]
    if not isinstance(edge_type, str) or edge_type not in VALID_EDGE_TYPES:
      raise InvalidEdgeTypeError(edge_type if isinstance(edge_type, str) else str(edge_type or ""))
    return metadata

  def _load_coordinates(
    self, metadata: ProblemMetadataRow
  ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Parse node coordinates, display coordinates, demands, and depots."""
    edge_type = metadata["edge_type"]
    if not isinstance(edge_type, str) or edge_type not in VALID_EDGE_TYPES:
      raise InvalidEdgeTypeError(edge_type if isinstance(edge_type, str) else str(edge_type or ""))

    raw_coords = metadata["raw_coords"]
    coordinates: np.ndarray | None = None
    if edge_type != "EXPLICIT":
      if raw_coords is None:
        raise MissingCoordinatesError(metadata["name"], edge_type)
      expected_dim: int = expected_coordinate_dimension(edge_type)
      try:
        coords_arr = np.asarray(raw_coords, dtype=np.float64)
      except (TypeError, ValueError) as exc:
        raise InvalidProblemDataError(
          metadata["name"],
          f"Invalid coordinates for edge_type='{edge_type}': {exc}",
        ) from exc
      if coords_arr.size == 0 or np.isnan(coords_arr).any():
        raise MissingCoordinatesError(metadata["name"], edge_type)
      if not np.isfinite(coords_arr).all():
        raise InvalidProblemDataError(
          metadata["name"],
          f"Non-finite coordinates for edge_type='{edge_type}'",
        )
      if coords_arr.shape != (metadata["dimension"], expected_dim):
        raise InvalidProblemDataError(
          metadata["name"],
          f"coordinates.shape should be ({metadata['dimension']}, {expected_dim}), got {coords_arr.shape}",
        )
      coordinates = np.ascontiguousarray(coords_arr, dtype=np.float64)

    display_coords: np.ndarray | None = None
    if metadata.get("raw_display_coords") is not None:
      try:
        display_arr = np.asarray(metadata["raw_display_coords"], dtype=np.float64)
      except (TypeError, ValueError) as exc:
        raise InvalidProblemDataError(
          metadata["name"],
          f"Invalid display_coords: {exc}",
        ) from exc
      if display_arr.size > 0:
        if display_arr.shape != (metadata["dimension"], 2):
          raise InvalidProblemDataError(
            metadata["name"],
            f"display_coords.shape should be ({metadata['dimension']}, 2), got {display_arr.shape}",
          )
        if not np.isfinite(display_arr).all():
          raise InvalidProblemDataError(
            metadata["name"],
            "Non-finite display_coords",
          )
        display_coords = np.ascontiguousarray(display_arr, dtype=np.float64)

    demands: np.ndarray | None = None
    if metadata.get("raw_demands") is not None:
      demands = np.asarray(metadata["raw_demands"], dtype=np.int32)

    depots: np.ndarray | None = None
    if metadata.get("raw_depots") is not None:
      depots = np.asarray(metadata["raw_depots"], dtype=np.int32)

    return coordinates, demands, display_coords, depots

  def _load_distance_matrix(self, metadata: ProblemMetadataRow) -> tuple[np.ndarray | None, str | None, bool]:
    """Parse explicit distance matrix from `edge_weight_matrices` table."""
    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    if metadata["edge_type"] != "EXPLICIT":
      return None, None, True

    row = self.conn.execute(
      """
      SELECT matrix_format, is_symmetric, matrix
      FROM edge_weight_matrices
      WHERE problem_id = ?
      """,
      [metadata["problem_id"]],
    ).fetchone()

    if row is None:
      raise InvalidProblemDataError(
        metadata["name"],
        f"EXPLICIT edge_type but no matrix in edge_weight_matrices for problem_id={metadata['problem_id']}",
      )

    matrix_format, is_symmetric, matrix_raw = row
    if matrix_raw is None:
      raise InvalidProblemDataError(
        metadata["name"],
        f"EXPLICIT edge_type has NULL matrix in edge_weight_matrices for problem_id={metadata['problem_id']}",
      )
    distances = np.asarray(matrix_raw, dtype=np.int32)
    expected_shape = (metadata["dimension"], metadata["dimension"])
    if distances.shape != expected_shape:
      raise InvalidProblemDataError(
        metadata["name"],
        f"Matrix shape mismatch: expected {expected_shape}, got {distances.shape}",
      )

    return distances, str(matrix_format) if matrix_format else None, bool(is_symmetric)

  def _construct_problem(
    self,
    metadata: ProblemMetadataRow,
    coordinates: np.ndarray | None,
    demands: np.ndarray | None,
    distances: np.ndarray | None,
    display_coords: np.ndarray | None = None,
    depots: np.ndarray | None = None,
    *,
    capacity_override: int | None = None,
    max_vehicles_override: int | None = None,
    max_route_distance_override: float | None = None,
    return_to_depot_override: bool | None = None,
  ) -> Problem:
    """Assemble validated Problem instance from loaded data and optional declarative overrides."""
    validate_single_depot(metadata["name"], metadata["type"], depots)
    try:
      if distances is None and coordinates is not None and metadata["dimension"] <= self.max_eager_matrix_dimension:
        from distances.matrix import compute_distance_matrix

        distances = compute_distance_matrix(coordinates, metadata["edge_type"])

      effective_capacity = capacity_override if capacity_override is not None else metadata["capacity"]
      effective_max_distance = (
        max_route_distance_override if max_route_distance_override is not None else metadata.get("max_distance")
      )
      effective_vehicles = max_vehicles_override if max_vehicles_override is not None else metadata.get("vehicles")

      constraints = build_vrp_constraints(
        name=metadata["name"],
        problem_type=metadata["type"],
        dimension=metadata["dimension"],
        capacity=effective_capacity,
        max_distance=effective_max_distance,
        vehicles=effective_vehicles,
        service_time=metadata.get("service_time"),
        return_to_depot=return_to_depot_override,
      )

      return Problem(
        name=metadata["name"],
        dimension=metadata["dimension"],
        problem_type=metadata["type"],
        edge_type=metadata["edge_type"],
        coordinates=coordinates,
        distances=distances,
        capacity=effective_capacity,
        demands=demands,
        problem_id=metadata["problem_id"],
        comment=metadata["comment"],
        edge_weight_format=metadata["edge_weight_format"],
        tsplib_name=metadata["tsplib_name"],
        has_solution=metadata["has_solution"],
        display_coords=display_coords,
        depots=depots,
        source_file=metadata["source_file"],
        file_checksum=metadata["file_checksum"],
        constraints=constraints,
      )
    except ValueError as e:
      raise InvalidProblemDataError(metadata["name"], str(e)) from e
