"""Unit tests for problem type handlers protocol, registry, and delegation in DatabaseLoader.

WHAT: Tests TspHandler, AtspHandler, CvrpHandler implementations, handler registry,
custom handler injection into DatabaseLoader, and UnsupportedProblemTypeError handling.
WHY: Removing hardcoded branching in DatabaseLoader requires that every problem formulation
type delegates to its registered ProblemTypeHandler, unknown formulations fail explicitly
with UnsupportedProblemTypeError instead of falling through, and custom handlers can be injected.
EXPECTED: Correct metadata dicts and problem listings for standard types; UnsupportedProblemTypeError
for unregistered types; custom handler dispatch when registered or injected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pytest

from data_models.exceptions import (
  InvalidProblemDataError,
  UnsupportedProblemTypeError,
)
from loaders.database_loader import DatabaseLoader
from loaders.problem_type_handlers import (
  PROBLEM_TYPE_HANDLERS,
  AtspHandler,
  BaseProblemRow,
  CvrpHandler,
  ProblemMetadataRow,
  ProblemTypeHandler,
  TspHandler,
  register_problem_type_handler,
)

if TYPE_CHECKING:
  from pathlib import Path


@pytest.fixture
def multi_problem_db(tmp_path: Path) -> Path:
  """Create a temporary DuckDB database containing TSP, ATSP, CVRP, and unknown (VRPTW) rows."""
  db_path = tmp_path / "multi_problem.duckdb"
  conn = duckdb.connect(str(db_path))
  conn.execute(
    """
    CREATE TABLE problems (
      id INTEGER PRIMARY KEY,
      name VARCHAR UNIQUE NOT NULL,
      type VARCHAR NOT NULL,
      has_solution BOOLEAN NOT NULL DEFAULT false,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    CREATE TABLE atsp_problems (
      problem_id INTEGER PRIMARY KEY,
      dimension INTEGER NOT NULL,
      comment VARCHAR,
      tsplib_name VARCHAR
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
      matrix_format VARCHAR NOT NULL,
      is_symmetric BOOLEAN NOT NULL,
      matrix INTEGER[][] NOT NULL
    );
    CREATE TABLE file_tracking (
      id INTEGER PRIMARY KEY,
      file_path VARCHAR UNIQUE NOT NULL,
      problem_id INTEGER,
      checksum VARCHAR,
      last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      file_size BIGINT
    );
    """
  )

  # 1. TSP row: id=1, name="tsp_test"
  conn.execute("INSERT INTO problems (id, name, type, has_solution) VALUES (1, 'tsp_test', 'TSP', false)")
  conn.execute(
    "INSERT INTO tsp_problems (problem_id, dimension, comment, edge_weight_type, coords) "
    "VALUES (1, 3, 'TSP comment', 'EUC_2D', [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])"
  )

  # 2. ATSP row: id=2, name="atsp_test"
  conn.execute("INSERT INTO problems (id, name, type, has_solution) VALUES (2, 'atsp_test', 'ATSP', false)")
  conn.execute(
    "INSERT INTO atsp_problems (problem_id, dimension, comment, tsplib_name) VALUES (2, 3, 'ATSP comment', 'atsp_test')"
  )
  conn.execute(
    "INSERT INTO edge_weight_matrices (problem_id, matrix_format, is_symmetric, matrix) "
    "VALUES (2, 'FULL_MATRIX', false, [[0, 5, 10], [10, 0, 5], [5, 10, 0]])"
  )

  # 3. CVRP row: id=3, name="cvrp_test"
  conn.execute("INSERT INTO problems (id, name, type, has_solution) VALUES (3, 'cvrp_test', 'CVRP', false)")
  conn.execute(
    "INSERT INTO cvrp_problems (problem_id, dimension, capacity, comment, edge_weight_type, coords, demands, depots) "
    "VALUES (3, 3, 50, 'CVRP comment', 'EUC_2D', [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], [0, 10, 20], [0])"
  )

  # 4. Unknown type row: id=4, name="vrptw_test", type="VRPTW"
  conn.execute("INSERT INTO problems (id, name, type, has_solution) VALUES (4, 'vrptw_test', 'VRPTW', false)")

  conn.close()
  return db_path


def test_tsp_handler_direct(multi_problem_db: Path) -> None:
  """Verify TspHandler queries problems and metadata directly."""
  conn = duckdb.connect(str(multi_problem_db), read_only=True)
  handler = TspHandler()
  assert handler.problem_type == "TSP"

  problems = handler.list_problems(conn)
  assert len(problems) == 1
  assert problems[0]["name"] == "tsp_test"
  assert problems[0]["dimension"] == 3
  assert problems[0]["edge_weight_type"] == "EUC_2D"

  base = BaseProblemRow(
    problem_id=1,
    name="tsp_test",
    problem_type="TSP",
    has_solution=False,
    source_file=None,
    file_checksum=None,
  )
  meta = handler.load_metadata(conn, base)
  assert meta["type"] == "TSP"
  assert meta["dimension"] == 3
  assert meta["edge_type"] == "EUC_2D"
  assert meta["raw_coords"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
  conn.close()


def test_atsp_handler_direct(multi_problem_db: Path) -> None:
  """Verify AtspHandler queries problems and metadata directly."""
  conn = duckdb.connect(str(multi_problem_db), read_only=True)
  handler = AtspHandler()
  assert handler.problem_type == "ATSP"

  problems = handler.list_problems(conn)
  assert len(problems) == 1
  assert problems[0]["name"] == "atsp_test"
  assert problems[0]["dimension"] == 3
  assert problems[0]["edge_weight_type"] == "EXPLICIT"

  base = BaseProblemRow(
    problem_id=2,
    name="atsp_test",
    problem_type="ATSP",
    has_solution=False,
    source_file=None,
    file_checksum=None,
  )
  meta = handler.load_metadata(conn, base)
  assert meta["type"] == "ATSP"
  assert meta["dimension"] == 3
  assert meta["edge_type"] == "EXPLICIT"
  assert meta["edge_weight_format"] is None
  conn.close()


def test_cvrp_handler_direct(multi_problem_db: Path) -> None:
  """Verify CvrpHandler queries problems and metadata directly."""
  conn = duckdb.connect(str(multi_problem_db), read_only=True)
  handler = CvrpHandler()
  assert handler.problem_type == "CVRP"

  problems = handler.list_problems(conn)
  assert len(problems) == 1
  assert problems[0]["name"] == "cvrp_test"
  assert problems[0]["dimension"] == 3
  assert problems[0]["edge_weight_type"] == "EUC_2D"

  base = BaseProblemRow(
    problem_id=3,
    name="cvrp_test",
    problem_type="CVRP",
    has_solution=False,
    source_file=None,
    file_checksum=None,
  )
  meta = handler.load_metadata(conn, base)
  assert meta["type"] == "CVRP"
  assert meta["dimension"] == 3
  assert meta["capacity"] == 50
  assert meta["raw_demands"] == [0, 10, 20]
  assert meta["raw_depots"] == [0]
  conn.close()


def test_database_loader_loads_atsp_and_cvrp(multi_problem_db: Path) -> None:
  """Verify DatabaseLoader loads ATSP and CVRP instances via registered handlers."""
  with DatabaseLoader(multi_problem_db) as loader:
    prob_atsp = loader.load("atsp_test")
    assert prob_atsp.name == "atsp_test"
    assert prob_atsp.problem_type == "ATSP"
    assert prob_atsp.dimension == 3
    assert prob_atsp.edge_type == "EXPLICIT"
    assert prob_atsp.distances is not None
    assert np.array_equal(prob_atsp.distances, np.array([[0, 5, 10], [10, 0, 5], [5, 10, 0]]))

    prob_cvrp = loader.load("cvrp_test")
    assert prob_cvrp.name == "cvrp_test"
    assert prob_cvrp.problem_type == "CVRP"
    assert prob_cvrp.dimension == 3
    assert prob_cvrp.capacity == 50
    assert prob_cvrp.demands is not None
    assert np.array_equal(prob_cvrp.demands, np.array([0, 10, 20]))


def test_unsupported_problem_type_raises_explicit_exception(multi_problem_db: Path) -> None:
  """Verify unknown problem type raises UnsupportedProblemTypeError, not InvalidProblemDataError."""
  with DatabaseLoader(multi_problem_db) as loader:
    with pytest.raises(UnsupportedProblemTypeError) as exc_load:
      loader.load("vrptw_test")
    assert exc_load.value.problem_type == "VRPTW"
    assert exc_load.value.instance_name == "vrptw_test"

    with pytest.raises(UnsupportedProblemTypeError) as exc_list:
      loader.list_problems("VRPTW")
    assert exc_list.value.problem_type == "VRPTW"


def test_custom_handler_injection(multi_problem_db: Path) -> None:
  """Verify custom handler injected via handlers parameter in DatabaseLoader constructor is called."""

  class VrptwHandler:
    problem_type: str = "VRPTW"

    def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
      return [{"name": "vrptw_test", "dimension": 10, "edge_weight_type": "EUC_2D"}]

    def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
      return {
        "problem_id": base.problem_id,
        "name": base.name,
        "type": "CVRP",
        "dimension": 2,
        "edge_type": "EUC_2D",
        "edge_weight_format": None,
        "tsplib_name": "vrptw_test",
        "comment": "custom vrptw",
        "capacity": 100,
        "has_solution": base.has_solution,
        "source_file": base.source_file,
        "file_checksum": base.file_checksum,
        "raw_coords": [[0.0, 0.0], [3.0, 4.0]],
        "raw_display_coords": None,
        "raw_demands": [0, 5],
        "raw_depots": [0],
      }

  custom_handlers: dict[str, ProblemTypeHandler] = {
    **PROBLEM_TYPE_HANDLERS,
    "VRPTW": VrptwHandler(),
  }

  with DatabaseLoader(multi_problem_db, handlers=custom_handlers) as loader:
    listed = loader.list_problems("VRPTW")
    assert len(listed) == 1
    assert listed[0]["name"] == "vrptw_test"

    prob = loader.load("vrptw_test")
    assert prob.problem_type == "CVRP"
    assert prob.dimension == 2
    assert prob.capacity == 100
    assert prob.distances is not None
    assert prob.distances.shape == (2, 2)


def test_register_problem_type_handler_and_restore() -> None:
  """Verify register_problem_type_handler adds to PROBLEM_TYPE_HANDLERS and cleans up properly."""

  class DummyHandler:
    problem_type: str = "DUMMY"

    def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
      return []

    def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
      raise NotImplementedError("registration test never loads metadata")

  dummy = DummyHandler()
  register_problem_type_handler(dummy)
  try:
    assert "DUMMY" in PROBLEM_TYPE_HANDLERS
    assert PROBLEM_TYPE_HANDLERS["DUMMY"] is dummy
  finally:
    PROBLEM_TYPE_HANDLERS.pop("DUMMY", None)


def test_missing_record_in_subtable_raises_invalid_problem_data_error(tmp_path: Path) -> None:
  """Verify missing record in type table raises InvalidProblemDataError with clear message."""
  db_path = tmp_path / "broken.duckdb"
  conn = duckdb.connect(str(db_path))
  conn.execute("""
    CREATE TABLE problems (
      id INTEGER PRIMARY KEY,
      name VARCHAR UNIQUE,
      type VARCHAR,
      has_solution BOOLEAN DEFAULT false
    );
    CREATE TABLE tsp_problems (
      problem_id INTEGER PRIMARY KEY,
      dimension INTEGER,
      comment VARCHAR,
      edge_weight_type VARCHAR,
      edge_weight_format VARCHAR,
      tsplib_name VARCHAR,
      coords DOUBLE[][],
      display_coords DOUBLE[][]
    );
    CREATE TABLE file_tracking (
      id INTEGER PRIMARY KEY,
      file_path VARCHAR UNIQUE,
      problem_id INTEGER,
      checksum VARCHAR,
      last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      file_size BIGINT
    );
    INSERT INTO problems VALUES (99, 'orphan_tsp', 'TSP', false);
  """)
  conn.close()

  with DatabaseLoader(db_path) as loader:
    with pytest.raises(InvalidProblemDataError, match="Missing record in tsp_problems for problem_id=99"):
      loader.load("orphan_tsp")
