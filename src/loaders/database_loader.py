"""
Database loader for routing problem instances.

This module provides a context manager for loading problem instances from
the DuckDB database. It handles database connections, SQL queries, data
transformation, and error handling with domain-specific exceptions.

Database Schema:
    - problems: Metadata (id, name, type, dimension, edge_weight_type, capacity, ...)
    - nodes: Node data (problem_id, node_id, x, y, z, demand, is_depot)
    - edge_weight_matrices: EXPLICIT distance matrices (problem_id, matrix_json)

Usage:
    >>> with DatabaseLoader() as loader:
    ...     problem = loader.load('berlin52')
    ...     print(f"{problem.name}: {problem.dimension} nodes")
    berlin52: 52 nodes
"""

import duckdb
import numpy as np
import json
from pathlib import Path
from typing import Optional

from ..data_models.problem import Problem
from ..data_models.exceptions import (
    InstanceNotFoundError,
    MissingCoordinatesError,
    DatabaseConnectionError,
    InvalidProblemDataError,
)


class DatabaseLoader:
    """
    Context manager for loading routing problems from DuckDB database.

    This class provides a clean interface for loading TSP, ATSP, and CVRP
    instances from the database. It handles connection lifecycle, data
    parsing, and validation with comprehensive error handling.

    The loader uses a context manager pattern to ensure proper resource
    cleanup and connection management.

    Attributes:
        db_path: Path to the DuckDB database file
        conn: Active database connection (None when outside context)

    Design Decisions:
        - Context manager ensures connections are always closed
        - Read-only connections prevent accidental database modifications
        - All queries use parameterized SQL to prevent injection
        - Data validation happens in Problem dataclass __post_init__
        - 0-based indexing (database uses 0-based, not TSPLIB 1-based)

    Example:
        >>> with DatabaseLoader('datasets/db/routing.duckdb') as loader:
        ...     # Load TSP instance
        ...     tsp = loader.load('berlin52')
        ...
        ...     # Load ATSP instance
        ...     atsp = loader.load('br17')
        ...
        ...     # Load CVRP instance
        ...     cvrp = loader.load('eil22')

    Error Handling:
        - Raises DatabaseConnectionError if database file doesn't exist
        - Raises InstanceNotFoundError if instance name not in database
        - Raises MissingCoordinatesError if coordinates NULL but required
        - Raises InvalidProblemDataError if data integrity checks fail
    """

    def __init__(self, db_path: str = "datasets/routing.duckdb"):
        """
        Initialize database loader with path to DuckDB file.

        Args:
            db_path: Path to DuckDB database file. Defaults to
                'datasets/routing.duckdb' (188 routing problems).

        Note:
            Connection is not established until entering context manager.
            This allows early initialization without holding resources.
        """
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def __enter__(self) -> "DatabaseLoader":
        """
        Enter context manager and establish database connection.

        Returns:
            Self for use in with statement

        Raises:
            DatabaseConnectionError: If database file doesn't exist or
                connection fails
        """
        try:
            # Verify database file exists
            db_file = Path(self.db_path)
            if not db_file.exists():
                raise FileNotFoundError(f"Database file not found: {self.db_path}")

            # Establish read-only connection
            self.conn = duckdb.connect(str(db_file), read_only=True)
            return self

        except Exception as e:
            # Wrap all connection errors in domain exception
            raise DatabaseConnectionError(self.db_path, e)

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        """
        Exit context manager and close database connection.

        Ensures connection is always closed even if errors occur during
        problem loading. Returns False to propagate any exceptions.
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        return False  # Propagate exceptions

    def load(self, instance_name: str) -> Problem:
        """
        Load a problem instance from the database.

        Elegantly decomposed into focused helper methods for clarity:
        - _load_metadata(): Query and validate problem metadata
        - _load_coordinates(): Extract node coordinates and demands
        - _load_distance_matrix(): Parse explicit distance matrix
        - _construct_problem(): Assemble validated Problem instance

        Args:
            instance_name: Name of the problem instance (e.g., 'berlin52')

        Returns:
            Immutable Problem instance with all data loaded

        Raises:
            InstanceNotFoundError: Instance name not found in database
            MissingCoordinatesError: Coordinates NULL but required
            InvalidProblemDataError: Data integrity violation

        Example:
            >>> with DatabaseLoader() as loader:
            ...     problem = loader.load('berlin52')
            ...     assert problem.dimension == 52
            ...     assert problem.edge_type == 'EUC_2D'
        """
        if self.conn is None:
            raise RuntimeError("DatabaseLoader must be used as context manager")

        # Decomposed workflow - each helper has single responsibility
        metadata = self._load_metadata(instance_name)
        coordinates, demands = self._load_coordinates(metadata)
        distances = self._load_distance_matrix(metadata)
        return self._construct_problem(metadata, coordinates, demands, distances)

    def _load_metadata(self, instance_name: str) -> dict:
        """
        Query and validate problem metadata.

        Returns:
            Dictionary with keys: problem_id, name, type, dimension,
            edge_type, capacity
        """
        assert self.conn is not None

        row = self.conn.execute(
            """
            SELECT id, name, type, dimension, edge_weight_type, capacity
            FROM problems
            WHERE name = ?
        """,
            [instance_name],
        ).fetchone()

        if row is None:
            raise InstanceNotFoundError(instance_name)

        problem_id, name, ptype, dimension, edge_type, capacity = row

        # Map database type to Problem type (VRP → CVRP)
        problem_type = "CVRP" if ptype == "VRP" else ptype

        return {
            "problem_id": problem_id,
            "name": name,
            "type": problem_type,
            "dimension": dimension,
            "edge_type": edge_type,
            "capacity": capacity,
        }

    def _load_coordinates(
        self, metadata: dict
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract node coordinates and demands for coordinate-based problems.

        Returns:
            (coordinates, demands) - both None if EXPLICIT edge_type
        """
        assert self.conn is not None

        if metadata["edge_type"] == "EXPLICIT":
            return None, None

        # Query node data ordered by ID
        # Note: Using DISTINCT to handle potential duplicate node entries in database
        rows = self.conn.execute(
            """
            SELECT DISTINCT node_id, x, y, demand
            FROM nodes
            WHERE problem_id = ?
            ORDER BY node_id
        """,
            [metadata["problem_id"]],
        ).fetchall()

        if not rows:
            raise InvalidProblemDataError(
                metadata["name"],
                f"No nodes found for problem_id={metadata['problem_id']}",
            )

        # Validate dimension consistency
        if len(rows) != metadata["dimension"]:
            raise InvalidProblemDataError(
                metadata["name"],
                f"Dimension mismatch: declared={metadata['dimension']}, actual={len(rows)}",
            )

        # Extract coordinates and demands
        coords_list: list[list[float]] = []
        demands_list: list[int] = []

        for _, x, y, demand in rows:
            if x is None or y is None:
                raise MissingCoordinatesError(metadata["name"], metadata["edge_type"])
            coords_list.append([x, y])
            demands_list.append(demand if demand is not None else 0)

        coordinates = np.array(coords_list, dtype=np.float64)

        # Only set demands if CVRP or non-zero demands exist
        if metadata["type"] == "CVRP" or any(d > 0 for d in demands_list):
            demands = np.array(demands_list, dtype=np.int32)
        else:
            demands = None

        return coordinates, demands

    def _load_distance_matrix(self, metadata: dict) -> Optional[np.ndarray]:
        """
        Parse explicit distance matrix from JSON.

        Returns:
            Distance matrix array, or None if not EXPLICIT edge_type
        """
        assert self.conn is not None

        if metadata["edge_type"] != "EXPLICIT":
            return None

        row = self.conn.execute(
            """
            SELECT matrix_json, dimension
            FROM edge_weight_matrices
            WHERE problem_id = ?
        """,
            [metadata["problem_id"]],
        ).fetchone()

        if row is None:
            raise InvalidProblemDataError(
                metadata["name"],
                "EXPLICIT edge_type but no matrix in edge_weight_matrices",
            )

        matrix_json, matrix_dim = row

        # Validate dimensions match
        if matrix_dim != metadata["dimension"]:
            raise InvalidProblemDataError(
                metadata["name"],
                f"Matrix dimension mismatch: problem={metadata['dimension']}, matrix={matrix_dim}",
            )

        # Parse and validate matrix
        try:
            matrix_list = json.loads(matrix_json)
            distances = np.array(matrix_list, dtype=np.float64)

            expected_shape = (metadata["dimension"], metadata["dimension"])
            if distances.shape != expected_shape:
                raise InvalidProblemDataError(
                    metadata["name"],
                    f"Matrix shape mismatch: expected {expected_shape}, got {distances.shape}",
                )

            return distances

        except json.JSONDecodeError as e:
            raise InvalidProblemDataError(
                metadata["name"], f"Failed to parse matrix JSON: {e}"
            )

    def _construct_problem(
        self,
        metadata: dict,
        coordinates: Optional[np.ndarray],
        demands: Optional[np.ndarray],
        distances: Optional[np.ndarray],
    ) -> Problem:
        """
        Assemble validated Problem instance from loaded data.

        The Problem.__post_init__ performs final validation.
        """
        # Compute distances from coordinates if needed
        if distances is None and coordinates is not None:
            from ..distances.matrix import compute_distance_matrix

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
            )
        except ValueError as e:
            raise InvalidProblemDataError(metadata["name"], str(e))
