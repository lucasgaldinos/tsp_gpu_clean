"""
Database loader for routing problem instances.

This module provides a context manager for loading problem instances from
the DuckDB database (`datasets/tsp.duckdb`). It handles database connections, SQL queries,
data transformation, and error handling with domain-specific exceptions.

Database Schema (datasets/tsp.duckdb):
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
    >>> with DatabaseLoader("datasets/tsp.duckdb") as loader:
    ...   problem = loader.load("berlin52")
    ...   print(f"{problem.name}: {problem.dimension} nodes")
    berlin52: 52 nodes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import duckdb
import numpy as np

from data_models.exceptions import (
  DatabaseConnectionError,
  InstanceNotFoundError,
  InvalidProblemDataError,
  MissingCoordinatesError,
  UnsupportedProblemTypeError,
)
from data_models.problem import Problem

from .problem_type_handlers import (
  PROBLEM_TYPE_HANDLERS,
  BaseProblemRow,
  ProblemTypeHandler,
)

if TYPE_CHECKING:
  from collections.abc import Mapping
  from types import TracebackType


class DatabaseLoader:
  """
  Context manager for loading routing problems from DuckDB database.

  This class provides a clean interface for loading TSP, ATSP, and CVRP
  instances from `datasets/tsp.duckdb`. It handles connection lifecycle,
  data parsing, and validation with comprehensive error handling.
  """

  def __init__(
    self,
    db_path: str | Path,
    fallback_tsp_dir: str | Path | None = None,
    handlers: Mapping[str, ProblemTypeHandler] | None = None,
  ) -> None:
    """
    Initialize database loader with path to DuckDB file.

    Args:
        db_path: Path to DuckDB database file.
        fallback_tsp_dir: Optional directory with `.tsp` files to use via
            `TsplibLoader` when the database file has not been created yet.
        handlers: Optional mapping from problem formulation type to handler instance.
            Defaults to PROBLEM_TYPE_HANDLERS.
    """
    self.db_path = str(db_path)
    self.fallback_tsp_dir = Path(fallback_tsp_dir) if fallback_tsp_dir is not None else None
    self.conn: duckdb.DuckDBPyConnection | None = None
    self._tsplib_fallback: Any = None
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
    try:
      db_file = Path(self.db_path)
      if not db_file.exists():
        if self.fallback_tsp_dir is not None and self.fallback_tsp_dir.exists():
          from .tsplib_loader import TsplibLoader

          self._tsplib_fallback = TsplibLoader(self.fallback_tsp_dir).__enter__()
          return self
        raise FileNotFoundError(f"Database file not found: {self.db_path}")

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

  def load(self, instance_name: str) -> Problem:
    """
    Load a problem instance from the DuckDB database (or `TsplibLoader` fallback).

    Args:
        instance_name: Name of the problem instance (e.g., 'berlin52').

    Returns:
        Immutable `Problem` instance with all data and metadata loaded.
    """
    if self._tsplib_fallback is not None:
      return cast("Problem", self._tsplib_fallback.load(instance_name))

    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    metadata = self._load_metadata(instance_name)
    coordinates, demands, display_coords, depots = self._load_coordinates(metadata)
    distances, matrix_format, is_symmetric = self._load_distance_matrix(metadata)
    metadata["edge_weight_format"] = metadata["edge_weight_format"] or matrix_format
    metadata["is_symmetric"] = is_symmetric

    return self._construct_problem(
      metadata=metadata,
      coordinates=coordinates,
      demands=demands,
      distances=distances,
      display_coords=display_coords,
      depots=depots,
    )

  def _load_metadata(self, instance_name: str) -> dict[str, Any]:
    """Query and validate problem metadata across normalized tables."""
    if self.conn is None:
      raise RuntimeError("DatabaseLoader must be used as context manager")

    row = self.conn.execute(
      """
      SELECT p.id, p.name, p.type, p.has_solution, ft.file_path, ft.checksum
      FROM problems p
      LEFT JOIN file_tracking ft ON p.id = ft.problem_id
      WHERE p.name = ?
      """,
      [instance_name],
    ).fetchone()

    if row is None:
      raise InstanceNotFoundError(instance_name)

    problem_id, name, ptype, has_solution, source_file, file_checksum = row

    handler = self.handlers.get(ptype)
    if handler is None:
      raise UnsupportedProblemTypeError(ptype, instance_name=name)

    base = BaseProblemRow(
      problem_id=problem_id,
      name=name,
      problem_type=ptype,
      has_solution=bool(has_solution),
      source_file=source_file,
      file_checksum=file_checksum,
    )
    return handler.load_metadata(self.conn, base)

  def _load_coordinates(
    self, metadata: dict[str, Any]
  ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Parse node coordinates, display coordinates, demands, and depots."""
    raw_coords = metadata["raw_coords"]
    coordinates: np.ndarray | None = None
    if metadata["edge_type"] != "EXPLICIT":
      if raw_coords is None:
        raise MissingCoordinatesError(metadata["name"], metadata["edge_type"])
      coords_arr = np.asarray(raw_coords, dtype=np.float64)
      if coords_arr.shape != (metadata["dimension"], 2) or np.isnan(coords_arr).any():
        raise MissingCoordinatesError(metadata["name"], metadata["edge_type"])
      coordinates = coords_arr

    display_coords: np.ndarray | None = None
    if metadata.get("raw_display_coords") is not None:
      display_coords = np.asarray(metadata["raw_display_coords"], dtype=np.float64)

    demands: np.ndarray | None = None
    if metadata.get("raw_demands") is not None:
      demands = np.asarray(metadata["raw_demands"], dtype=np.int32)

    depots: np.ndarray | None = None
    if metadata.get("raw_depots") is not None:
      depots = np.asarray(metadata["raw_depots"], dtype=np.int32)

    return coordinates, demands, display_coords, depots

  def _load_distance_matrix(self, metadata: dict[str, Any]) -> tuple[np.ndarray | None, str | None, bool]:
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
    distances = np.asarray(matrix_raw, dtype=np.float64)
    expected_shape = (metadata["dimension"], metadata["dimension"])
    if distances.shape != expected_shape:
      raise InvalidProblemDataError(
        metadata["name"],
        f"Matrix shape mismatch: expected {expected_shape}, got {distances.shape}",
      )

    return distances, str(matrix_format) if matrix_format else None, bool(is_symmetric)

  def _construct_problem(
    self,
    metadata: dict[str, Any],
    coordinates: np.ndarray | None,
    demands: np.ndarray | None,
    distances: np.ndarray | None,
    display_coords: np.ndarray | None = None,
    depots: np.ndarray | None = None,
  ) -> Problem:
    """Assemble validated Problem instance from loaded data."""
    if distances is None and coordinates is not None:
      from distances.matrix import compute_distance_matrix

      distances = compute_distance_matrix(coordinates, metadata["edge_type"])

    try:
      return Problem(
        name=metadata["name"],
        dimension=metadata["dimension"],
        problem_type=metadata["type"],
        edge_type=metadata["edge_type"],
        coordinates=coordinates,
        distances=distances,
        capacity=metadata["capacity"],
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
      )
    except ValueError as e:
      raise InvalidProblemDataError(metadata["name"], str(e)) from e
