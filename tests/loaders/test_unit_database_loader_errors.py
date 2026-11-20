"""Unit tests for `DatabaseLoader.__enter__` failure paths.

A missing database file is a configuration error the caller must be able to branch on,
so it surfaces as `DatabaseConnectionError` carrying the original `FileNotFoundError`
(``BUG-16`` / ``DP10-11``). The existence check runs before the connection ``try``, so the
error is never re-captured and re-wrapped by the generic handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import duckdb
import numpy as np
import pytest

from data_models.exceptions import (
  DatabaseConnectionError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
)
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


def test_tsplib_loader_accepts_euc3d_and_rejects_mismatched_coordinate_dimensions(tmp_path: Path) -> None:
  """WHAT: TSPLIB loader accepts `EUC_3D` with 3D coordinates, rejects dimension mismatches,
  and still skips unsupported edge types (`XRAY1`).

  WHY: `DP14-6` enables `EUC_3D` end-to-end while preserving `BUG-53`'s guarantee against
  silently truncating `z` on 2D edge types or accepting 2D coords for 3D edge types.
  """
  from loaders.tsplib_loader import (
    MatrixRowRecord,
    NodeRowRecord,
    ProblemRowRecord,
    TsplibIngestionBundle,
    _bundle_from_normalized,
    _problem_payload,
    bundle_to_problem,
    ingest_tsplib_directory_to_duckdb,
    parse_tsplib_file,
  )

  # 1. Valid EUC_3D instance parses, preserves z, and computes 3D distance matrix
  p_euc3d = tmp_path / "test_euc3d.tsp"
  p_euc3d.write_text(
    "NAME: test_euc3d\n"
    "TYPE: TSP\n"
    "DIMENSION: 3\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 2.0 3.0\n"
    "3 4.0 6.0 8.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  bundle_3d = parse_tsplib_file(p_euc3d)
  assert bundle_3d.problem.edge_weight_type == "EUC_3D"
  assert [(node.x, node.y, node.z) for node in bundle_3d.nodes] == [
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    (4.0, 6.0, 8.0),
  ]
  problem_3d = bundle_to_problem(bundle_3d)
  assert problem_3d.edge_type == "EUC_3D"
  assert problem_3d.coordinates is not None
  assert problem_3d.coordinates.shape == (3, 3)
  assert problem_3d.distances is not None
  np.testing.assert_array_equal(problem_3d.distances, np.array([[0, 4, 11], [4, 0, 7], [11, 7, 0]], dtype=np.int32))

  # 2. Valid 2D edge weight type with 3D coordinates must fail explicitly
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
  with pytest.raises(InvalidProblemDataError, match=r"requires 2-D coordinates|requires 2D coordinates"):
    parse_tsplib_file(p_3d_coords)

  # 3. EUC_3D with 2D coordinates must fail explicitly
  p_euc3d_2d_coords = tmp_path / "test_euc3d_2d.tsp"
  p_euc3d_2d_coords.write_text(
    "NAME: test_euc3d_2d\n"
    "TYPE: TSP\n"
    "DIMENSION: 2\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0\n"
    "2 1.0 1.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  with pytest.raises(InvalidProblemDataError, match=r"requires 3-D coordinates|requires 3D coordinates"):
    parse_tsplib_file(p_euc3d_2d_coords)

  # 4. Direct _bundle_from_normalized and _problem_payload validation for mismatches
  normalized_2d_type_3d_coords = {
    "problem_data": {
      "name": "synth_2d_with_3d",
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
  with pytest.raises(InvalidProblemDataError, match="requires 2D coordinates"):
    _bundle_from_normalized(p_3d_coords, cast("NormalizedProblemData", normalized_2d_type_3d_coords), problem_id=1)
  with pytest.raises(InvalidProblemDataError, match="requires 2D coordinates"):
    _problem_payload(p_3d_coords, cast("NormalizedProblemData", normalized_2d_type_3d_coords))

  normalized_3d_type_2d_coords = {
    "problem_data": {
      "name": "synth_3d_with_2d",
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
  with pytest.raises(InvalidProblemDataError, match="requires 3D coordinates"):
    _bundle_from_normalized(
      p_euc3d_2d_coords, cast("NormalizedProblemData", normalized_3d_type_2d_coords), problem_id=1
    )
  with pytest.raises(InvalidProblemDataError, match="requires 3D coordinates"):
    _problem_payload(p_euc3d_2d_coords, cast("NormalizedProblemData", normalized_3d_type_2d_coords))

  # 5. Direct bundle_to_problem checks for missing/unexpected z
  missing_z_bundle = TsplibIngestionBundle(
    problem=ProblemRowRecord(1, "missing_z", "TSP", 2, "EUC_3D", None, None),
    nodes=(
      NodeRowRecord(1, 0, 0.0, 0.0, 0, True, z=None),
      NodeRowRecord(1, 1, 1.0, 1.0, 0, False, z=None),
    ),
    edge_matrix=None,
  )
  with pytest.raises(MissingCoordinatesError):
    bundle_to_problem(missing_z_bundle)

  unexpected_z_2d_bundle = TsplibIngestionBundle(
    problem=ProblemRowRecord(1, "unexpected_z", "TSP", 2, "EUC_2D", None, None),
    nodes=(
      NodeRowRecord(1, 0, 0.0, 0.0, 0, True, z=5.0),
      NodeRowRecord(1, 1, 1.0, 1.0, 0, False, z=6.0),
    ),
    edge_matrix=None,
  )
  with pytest.raises(InvalidProblemDataError, match="requires 2D coordinates"):
    bundle_to_problem(unexpected_z_2d_bundle)

  unexpected_z_explicit_bundle = TsplibIngestionBundle(
    problem=ProblemRowRecord(1, "unexpected_z_exp", "ATSP", 2, "EXPLICIT", None, None),
    nodes=(
      NodeRowRecord(1, 0, 0.0, 0.0, 0, True, z=5.0),
      NodeRowRecord(1, 1, 1.0, 1.0, 0, False, z=6.0),
    ),
    edge_matrix=MatrixRowRecord(1, 2, "FULL_MATRIX", False, matrix=[[0, 1], [1, 0]]),
  )
  with pytest.raises(InvalidProblemDataError, match="EXPLICIT display coordinates require 2D coordinates"):
    bundle_to_problem(unexpected_z_explicit_bundle)

  # 6. Unsupported edge types (e.g. XRAY1) still raise InvalidEdgeTypeError and are skipped during ingestion
  p_xray = tmp_path / "test_xray.tsp"
  p_xray.write_text(
    "NAME: test_xray\n"
    "TYPE: TSP\n"
    "DIMENSION: 2\n"
    "EDGE_WEIGHT_TYPE: XRAY1\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 1.0 1.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  with pytest.raises(InvalidEdgeTypeError):
    parse_tsplib_file(p_xray)

  mixed_dir = tmp_path / "mixed_tsp_dir"
  mixed_dir.mkdir()
  (mixed_dir / "valid_3d.tsp").write_text(p_euc3d.read_text(encoding="utf-8"), encoding="utf-8")
  (mixed_dir / "unsupported_xray.tsp").write_text(p_xray.read_text(encoding="utf-8"), encoding="utf-8")
  out_db = tmp_path / "tsplib_mixed_test.duckdb"
  count = ingest_tsplib_directory_to_duckdb(mixed_dir, out_db)
  assert count == 1


def test_database_loader_euc3d_and_coordinate_shape_validation(tmp_path: Path) -> None:
  """WHAT: `DatabaseLoader` loads valid `EUC_3D` TSP/CVRP rows and rejects invalid coordinate shapes.

  WHY: DuckDB `DOUBLE[][]` columns can contain 2D, 3D, NULL, NaN, or ragged rows; the loader
  must accept `(dimension, 3)` for `EUC_3D` while catching mismatches explicitly.
  """
  db_path = tmp_path / "euc3d_validation.duckdb"
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
    CREATE TABLE cvrp_problems (
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
    INSERT INTO problems VALUES
      (1, 'tsp_euc3d_ok', 'TSP', false),
      (2, 'cvrp_euc3d_ok', 'CVRP', false),
      (3, 'euc3d_with_2d', 'TSP', false),
      (4, 'euc2d_with_3d', 'TSP', false),
      (5, 'euc3d_with_nan', 'TSP', false),
      (6, 'euc3d_ragged', 'TSP', false);
    INSERT INTO tsp_problems VALUES
      (1, 3, 'valid 3d tsp', 'euc_3d', NULL, 'tsp_euc3d_ok',
       [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], NULL),
      (3, 3, '2d coords on 3d', 'EUC_3D', NULL, 'euc3d_with_2d',
       [[0.0, 0.0], [1.0, 2.0], [4.0, 6.0]], NULL),
      (4, 3, '3d coords on 2d', 'EUC_2D', NULL, 'euc2d_with_3d',
       [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], NULL),
      (5, 2, 'nan in coords', 'EUC_3D', NULL, 'euc3d_with_nan',
       [[0.0, 0.0, 0.0], [1.0, NULL, 3.0]], NULL),
      (6, 2, 'ragged coords', 'EUC_3D', NULL, 'euc3d_ragged',
       [[0.0, 0.0, 0.0], [1.0, 2.0]], NULL);
    INSERT INTO cvrp_problems VALUES
      (2, 3, 25, 'valid 3d cvrp', 'EUC_3D', NULL, 'cvrp_euc3d_ok',
       [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], [0, 10, 15], [0]);
    """
  )
  conn.close()

  with DatabaseLoader(str(db_path), fallback_tsp_dir=None) as loader:
    tsp_prob = loader.load("tsp_euc3d_ok")
    assert tsp_prob.edge_type == "EUC_3D"
    assert tsp_prob.coordinates is not None
    assert tsp_prob.coordinates.shape == (3, 3)
    assert tsp_prob.distances is not None
    np.testing.assert_array_equal(tsp_prob.distances, np.array([[0, 4, 11], [4, 0, 7], [11, 7, 0]], dtype=np.int32))

    cvrp_prob = loader.load("cvrp_euc3d_ok")
    assert cvrp_prob.problem_type == "CVRP"
    assert cvrp_prob.edge_type == "EUC_3D"
    assert cvrp_prob.coordinates is not None
    assert cvrp_prob.coordinates.shape == (3, 3)
    assert cvrp_prob.capacity == 25
    assert cvrp_prob.demands is not None
    np.testing.assert_array_equal(cvrp_prob.demands, np.array([0, 10, 15], dtype=np.int32))

    with pytest.raises(InvalidProblemDataError, match=r"coordinates\.shape should be \(3, 3\), got \(3, 2\)"):
      loader.load("euc3d_with_2d")

    with pytest.raises(InvalidProblemDataError, match=r"coordinates\.shape should be \(3, 2\), got \(3, 3\)"):
      loader.load("euc2d_with_3d")

    with pytest.raises(MissingCoordinatesError):
      loader.load("euc3d_with_nan")

    with pytest.raises(InvalidProblemDataError, match="Invalid coordinates"):
      loader.load("euc3d_ragged")


def test_tsplib_loader_cvrp_euc3d_threed_coords_display_and_edge_guards(tmp_path: Path) -> None:
  """WHAT: TSPLIB loader handles CVRP `EUC_3D`, `THREED_COORDS` + `TWOD_DISPLAY`, and rejects bad payloads/edge types.

  WHY: Real TSPLIB95 files combine `EUC_3D` with `NODE_COORD_TYPE: THREED_COORDS`, `DISPLAY_DATA_TYPE: TWOD_DISPLAY`,
  or `TYPE: CVRP`, while malformed normalized payloads or invalid bundle `edge_weight_type` must fail cleanly.
  """
  from loaders.tsplib_loader import (
    NodeRowRecord,
    ProblemRowRecord,
    TsplibIngestionBundle,
    _bundle_from_normalized,
    _problem_payload,
    bundle_to_problem,
    ingest_tsplib_directory_to_duckdb,
    parse_tsplib_file,
  )

  # 1. CVRP with EUC_3D and TSP with THREED_COORDS + TWOD_DISPLAY
  tsp_dir = tmp_path / "euc3d_variants"
  tsp_dir.mkdir()
  cvrp_file = tsp_dir / "cvrp3d.tsp"
  cvrp_file.write_text(
    "NAME: cvrp3d\n"
    "TYPE: CVRP\n"
    "DIMENSION: 3\n"
    "CAPACITY: 20\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 2.0 3.0\n"
    "3 4.0 6.0 8.0\n"
    "DEMAND_SECTION\n"
    "1 0\n"
    "2 7\n"
    "3 9\n"
    "DEPOT_SECTION\n"
    "1\n"
    "-1\n"
    "EOF\n",
    encoding="utf-8",
  )
  cvrp_prob = bundle_to_problem(parse_tsplib_file(cvrp_file))
  assert cvrp_prob.problem_type == "CVRP"
  assert cvrp_prob.edge_type == "EUC_3D"
  assert cvrp_prob.coordinates is not None
  assert cvrp_prob.coordinates.shape == (3, 3)
  assert cvrp_prob.demands is not None
  np.testing.assert_array_equal(cvrp_prob.demands, np.array([0, 7, 9], dtype=np.int32))
  assert cvrp_prob.depots is not None
  np.testing.assert_array_equal(cvrp_prob.depots, np.array([0], dtype=np.int32))

  disp_file = tsp_dir / "disp3d.tsp"
  disp_file.write_text(
    "NAME: disp3d\n"
    "TYPE: TSP\n"
    "DIMENSION: 3\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_TYPE: THREED_COORDS\n"
    "DISPLAY_DATA_TYPE: TWOD_DISPLAY\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 2.0 3.0\n"
    "3 4.0 6.0 8.0\n"
    "DISPLAY_DATA_SECTION\n"
    "1 0.0 0.0\n"
    "2 1.0 2.0\n"
    "3 4.0 6.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  db_path = tmp_path / "variants.duckdb"
  assert ingest_tsplib_directory_to_duckdb(tsp_dir, db_path) == 2

  with DatabaseLoader(db_path, fallback_tsp_dir=None) as loader:
    loaded_disp = loader.load("disp3d")
    assert loaded_disp.coordinates is not None
    assert loaded_disp.coordinates.shape == (3, 3)
    assert loaded_disp.display_coords is not None
    assert loaded_disp.display_coords.shape == (3, 2)
    loaded_cvrp = loader.load("cvrp3d")
    assert loaded_cvrp.problem_type == "CVRP"
    assert loaded_cvrp.edge_type == "EUC_3D"

  # 2. Spec mismatch: EUC_3D with NODE_COORD_TYPE: TWOD_COORDS fails
  bad_header = tmp_path / "bad_header_3d.tsp"
  bad_header.write_text(
    "NAME: bad_header_3d\n"
    "TYPE: TSP\n"
    "DIMENSION: 2\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_TYPE: TWOD_COORDS\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 2.0 3.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  with pytest.raises(InvalidProblemDataError, match="TWOD_COORDS"):
    parse_tsplib_file(bad_header)

  # 3. Direct bundle_to_problem with unsupported edge_weight_type raises InvalidEdgeTypeError
  for bad_edge_type, coords in (("XRAY1", (0.0, 0.0, 0.0)), ("BOGUS", (0.0, 0.0, None))):
    bad_bundle = TsplibIngestionBundle(
      problem=ProblemRowRecord(1, "bad_edge", "TSP", 1, bad_edge_type, None, None),
      nodes=(NodeRowRecord(1, 0, coords[0], coords[1], 0, True, z=coords[2]),),
      edge_matrix=None,
    )
    with pytest.raises(InvalidEdgeTypeError):
      bundle_to_problem(bad_bundle)

  # 4. Extra rows or non-finite values in synthetic normalized payload fail explicitly
  extra_rows = {
    "problem_data": {"name": "extra", "type": "TSP", "dimension": 2, "edge_weight_type": "EUC_3D"},
    "coords": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "display_coords": None,
    "demands": [0, 0],
    "depots": [],
    "edges": [],
    "fixed_edges": [],
  }
  with pytest.raises(InvalidProblemDataError, match="Expected 2 coordinates"):
    _bundle_from_normalized(disp_file, cast("NormalizedProblemData", extra_rows), problem_id=1)
  with pytest.raises(InvalidProblemDataError, match="Expected 2 coordinates"):
    _problem_payload(disp_file, cast("NormalizedProblemData", extra_rows))

  inf_rows = {
    "problem_data": {"name": "inf_coords", "type": "TSP", "dimension": 2, "edge_weight_type": "EUC_3D"},
    "coords": [[0.0, 0.0, 0.0], [1.0, float("inf"), 3.0]],
    "display_coords": None,
    "demands": [0, 0],
    "depots": [],
    "edges": [],
    "fixed_edges": [],
  }
  with pytest.raises(InvalidProblemDataError, match="Non-finite coordinate"):
    _bundle_from_normalized(disp_file, cast("NormalizedProblemData", inf_rows), problem_id=1)
  with pytest.raises(InvalidProblemDataError, match="Non-finite coordinate"):
    _problem_payload(disp_file, cast("NormalizedProblemData", inf_rows))


def test_database_loader_rejects_unsupported_edge_types_and_bad_display_coords(tmp_path: Path) -> None:
  """WHAT: `DatabaseLoader` raises `InvalidEdgeTypeError` for unsupported DB edge types and bad display coords.

  WHY: Real DuckDB files may store `XRAY1`/`XRAY2` rows with 3D coordinates or corrupt `EDGE_WEIGHT_TYPE` values;
  those must surface `InvalidEdgeTypeError` (not a 2D shape mismatch or an uncaught `ValueError` in matrix build).
  """
  db_path = tmp_path / "edge_type_and_display_errors.duckdb"
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
    INSERT INTO problems VALUES
      (1, 'xray1_3d', 'TSP', false),
      (2, 'man3d_3d', 'TSP', false),
      (3, 'bogus_2d', 'TSP', false),
      (4, 'empty_edge', 'TSP', false),
      (5, 'euc3d_inf', 'TSP', false),
      (6, 'euc3d_3d_disp', 'TSP', false);
    INSERT INTO tsp_problems VALUES
      (1, 2, 'xray1 row', 'XRAY1', NULL, 'xray1_3d', [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], NULL),
      (2, 2, 'man3d row', 'MAN_3D', NULL, 'man3d_3d', [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], NULL),
      (3, 2, 'bogus row', 'BOGUS', NULL, 'bogus_2d', [[0.0, 0.0], [1.0, 1.0]], NULL),
      (4, 2, 'empty edge', '', NULL, 'empty_edge', [[0.0, 0.0], [1.0, 1.0]], NULL),
      (5, 2, 'inf coords', 'EUC_3D', NULL, 'euc3d_inf', [[0.0, 0.0, 0.0], [1e309, 2.0, 3.0]], NULL),
      (6, 2, '3d disp', 'EUC_3D', NULL, 'euc3d_3d_disp',
       [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
    """
  )
  conn.close()

  with DatabaseLoader(str(db_path), fallback_tsp_dir=None) as loader:
    for name in ("xray1_3d", "man3d_3d", "bogus_2d", "empty_edge"):
      with pytest.raises(InvalidEdgeTypeError):
        loader.load(name)

    with pytest.raises(InvalidProblemDataError, match="Non-finite coordinates"):
      loader.load("euc3d_inf")

    with pytest.raises(InvalidProblemDataError, match=r"display_coords\.shape should be \(2, 2\), got \(2, 3\)"):
      loader.load("euc3d_3d_disp")


