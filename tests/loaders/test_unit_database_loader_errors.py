"""Unit tests for `DatabaseLoader.__enter__` failure paths.

A missing database file is a configuration error the caller must be able to branch on,
so it surfaces as `DatabaseConnectionError` carrying the original `FileNotFoundError`
(``BUG-16`` / ``DP10-11``). The existence check runs before the connection ``try``, so the
error is never re-captured and re-wrapped by the generic handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import duckdb
import pytest

from data_models.exceptions import DatabaseConnectionError, InvalidProblemDataError
from loaders.database_loader import DatabaseLoader

if TYPE_CHECKING:
  from pathlib import Path

  from converter.core.transformer import NormalizedProblemData


def test_missing_db_without_fallback_raises_database_connection_error(tmp_path: Path) -> None:
  """WHAT: entering a loader whose `db_path` does not exist and has no `.tsp` fallback.

  WHY: the pre-fix code raised `FileNotFoundError` inside the connection `try` and
  caught it with `except Exception`, producing a generic error with a self-referential
  cause chain instead of an actionable loader error.

  EXPECTED: `DatabaseConnectionError` naming the path with "does not exist", its
  `original_error` a `FileNotFoundError`, no `__cause__`, and no open connection.
  """
  # Arrange
  missing_db = tmp_path / "absent.duckdb"
  loader = DatabaseLoader(str(missing_db), fallback_tsp_dir=None)

  # Act / Assert
  with pytest.raises(DatabaseConnectionError) as excinfo:
    loader.__enter__()

  error = excinfo.value
  assert error.db_path == str(missing_db)
  assert isinstance(error.original_error, FileNotFoundError)
  assert "does not exist" in str(error)
  assert error.__cause__ is None, "the FileNotFoundError must not be captured and re-wrapped"
  assert loader.conn is None
  assert loader._tsplib_fallback is None


def test_missing_db_with_existing_fallback_directory_uses_tsplib_loader(
  tmp_path: Path,
  fixture_tsp_dir: Path,
) -> None:
  """WHAT: the same missing database, but with a populated `.tsp` fallback directory.

  WHY: the fallback must be chosen before the existence error, so the loader still works
  while the DuckDB file has not been built.

  EXPECTED: no exception, loads berlin52 through `TsplibLoader`, and the fallback is
  released on context exit.
  """
  # Arrange
  missing_db = tmp_path / "absent_with_fallback.duckdb"

  # Act
  with DatabaseLoader(str(missing_db), fallback_tsp_dir=fixture_tsp_dir) as loader:
    problem = loader.load("berlin52")
    fallback_active = loader._tsplib_fallback is not None

  # Assert
  assert problem.name == "berlin52"
  assert problem.dimension == 52
  assert fallback_active
  assert loader._tsplib_fallback is None


def test_database_loader_explicit_matrix_null_raises_invalid_problem_data_error(tmp_path: Path) -> None:
  """WHAT: EXPLICIT problem with NULL matrix in edge_weight_matrices raises InvalidProblemDataError.

  WHY: Task 12.38 (BUG-53 / DP12-28) — Guard against np.asarray(None) crash and raise
  explicit InvalidProblemDataError.
  """
  db_path = tmp_path / "null_matrix.duckdb"
  conn = duckdb.connect(str(db_path))
  conn.execute(
    """
    CREATE TABLE problems (
      id INTEGER PRIMARY KEY,
      name VARCHAR UNIQUE NOT NULL,
      type VARCHAR NOT NULL,
      has_solution BOOLEAN DEFAULT false
    );
    CREATE TABLE tsp_problems (
      problem_id INTEGER PRIMARY KEY,
      dimension INTEGER NOT NULL,
      comment VARCHAR,
      edge_weight_type VARCHAR NOT NULL,
      edge_weight_format VARCHAR,
      tsplib_name VARCHAR,
      coords DOUBLE[][],
      display_coords DOUBLE[][]
    );
    CREATE TABLE edge_weight_matrices (
      problem_id INTEGER PRIMARY KEY,
      dimension INTEGER NOT NULL,
      matrix_format VARCHAR,
      is_symmetric BOOLEAN,
      matrix INTEGER[][]
    );
    CREATE TABLE file_tracking (
      problem_id INTEGER PRIMARY KEY,
      file_path VARCHAR,
      checksum VARCHAR
    );
    INSERT INTO problems (id, name, type, has_solution) VALUES (1, 'null_mat_prob', 'TSP', false);
    INSERT INTO tsp_problems VALUES (1, 3, 'Test null matrix', 'EXPLICIT', 'FULL_MATRIX', 'null_mat_prob', NULL, NULL);
    INSERT INTO edge_weight_matrices VALUES (1, 3, 'FULL_MATRIX', true, NULL);
    """
  )
  conn.close()

  with DatabaseLoader(str(db_path), fallback_tsp_dir=None) as loader:
    with pytest.raises(InvalidProblemDataError, match="NULL matrix"):
      loader.load("null_mat_prob")


def test_tsplib_loader_rejects_3d_coordinates_and_3d_edge_types(tmp_path: Path) -> None:
  """WHAT: TSPLIB loader rejects EUC_3D and 3D coordinate sections.

  WHY: Task 12.38 (BUG-53 / DP12-28) — DuckDB and 2D solvers do not support 3D.
  Ingestion must explicitly reject 3D rather than silently truncating the z coordinate.
  """
  from data_models.exceptions import InvalidEdgeTypeError
  from loaders.tsplib_loader import parse_tsplib_file

  # 1. 3D edge weight type: EUC_3D
  p_euc3d = tmp_path / "test_euc3d.tsp"
  p_euc3d.write_text(
    "NAME: test_euc3d\n"
    "TYPE: TSP\n"
    "DIMENSION: 2\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 1.0 1.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  with pytest.raises(InvalidEdgeTypeError):
    parse_tsplib_file(p_euc3d)

  # 2. Valid 2D edge weight type, but coordinates have 3 components
  p_3d_coords = tmp_path / "test_3d_coords.tsp"
  p_3d_coords.write_text(
    "NAME: test_3d_coords\n"
    "TYPE: TSP\n"
    "DIMENSION: 2\n"
    "EDGE_WEIGHT_TYPE: EUC_2D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 5.0\n"
    "2 1.0 1.0 10.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  with pytest.raises(InvalidProblemDataError, match=r"requires 2-D coordinates|3D coordinates are not supported"):
    parse_tsplib_file(p_3d_coords)

  # 3. Direct _bundle_from_normalized validation of 3D coordinates
  from loaders.tsplib_loader import _bundle_from_normalized

  normalized_3d = {
    "problem_data": {
      "name": "synth_3d",
      "type": "TSP",
      "dimension": 2,
      "edge_weight_type": "EUC_2D",
    },
    "coords": [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
    "display_coords": None,
    "demands": [0, 0],
    "depots": [],
    "edges": [],
    "fixed_edges": [],
  }

  with pytest.raises(InvalidProblemDataError, match="3D coordinates are not supported"):
    _bundle_from_normalized(p_3d_coords, cast("NormalizedProblemData", normalized_3d), problem_id=1)

  # 4. Direct _problem_payload validation rejecting 3D EDGE_WEIGHT_TYPE
  from loaders.tsplib_loader import _problem_payload, ingest_tsplib_directory_to_duckdb

  normalized_euc3d = {
    "problem_data": {
      "name": "synth_3d",
      "type": "TSP",
      "dimension": 2,
      "edge_weight_type": "EUC_3D",
    },
    "coords": [[0.0, 0.0], [1.0, 1.0]],
    "display_coords": None,
    "demands": [0, 0],
    "depots": [],
    "edges": [],
    "fixed_edges": [],
  }
  with pytest.raises(InvalidEdgeTypeError):
    _problem_payload(p_euc3d, cast("NormalizedProblemData", normalized_euc3d))

  # 5. Direct _problem_payload validation rejecting 3D coordinates
  with pytest.raises(InvalidProblemDataError, match="3D coordinates are not supported"):
    _problem_payload(p_3d_coords, cast("NormalizedProblemData", normalized_3d))

  # 6. ingest_tsplib_directory_to_duckdb safely skips 3D files
  tsplib_3d_dir = tmp_path / "tsplib_3d_only"
  tsplib_3d_dir.mkdir()
  (tsplib_3d_dir / "inst_3d.tsp").write_text(p_euc3d.read_text(encoding="utf-8"), encoding="utf-8")
  out_db = tmp_path / "tsplib_3d_test.duckdb"
  count = ingest_tsplib_directory_to_duckdb(tsplib_3d_dir, out_db)
  assert count == 0


