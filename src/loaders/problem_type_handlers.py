"""
Problem type handlers for normalized DuckDB database loading.

Encapsulates problem-type specific SQL queries and metadata extraction
(TSP, ATSP, CVRP, and extensibility for future formulations like VRPTW, MDVRP)
without conditional branching in DatabaseLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from data_models.exceptions import InvalidProblemDataError

if TYPE_CHECKING:
  import duckdb


class ProblemMetadataRow(TypedDict):
  """Named contract for one problem's metadata row (hub + type table).

  Replaces the untyped ``dict[str, Any]`` bags that crossed the handler -> loader
  boundary (``DP10-8``). ``raw_*`` carry the DuckDB array columns verbatim.
  """

  problem_id: int
  name: str
  type: str
  dimension: int
  edge_type: str
  edge_weight_format: str | None
  tsplib_name: str | None
  comment: str | None
  capacity: int | None
  has_solution: bool
  source_file: str | None
  file_checksum: str | None
  raw_coords: list[list[float]] | None
  raw_display_coords: list[list[float]] | None
  raw_demands: list[int] | None
  raw_depots: list[int] | None


@dataclass(frozen=True, slots=True)
class BaseProblemRow:
  """Base problem row attributes from `problems LEFT JOIN file_tracking`."""

  problem_id: int
  name: str
  problem_type: str
  has_solution: bool
  source_file: str | None
  file_checksum: str | None


class ProblemTypeHandler(Protocol):
  """Protocol for problem-type specific metadata loading and problem listing."""

  problem_type: str

  def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """List all problems of this formulation type present in the database."""
    ...

  def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
    """Load metadata dictionary for the instance from type-specific tables."""
    ...


class TspHandler:
  """Handler for symmetric Traveling Salesperson Problem (TSP) instances."""

  problem_type: str = "TSP"

  def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = conn.execute(
      """
      SELECT p.id, p.name, p.type, t.dimension, t.edge_weight_type, t.comment, t.edge_weight_format
      FROM problems p
      JOIN tsp_problems t ON p.id = t.problem_id
      WHERE p.type = ?
      ORDER BY t.dimension ASC, p.name ASC
      """,
      [self.problem_type],
    ).fetchall()
    return [
      {
        "id": r[0],
        "name": r[1],
        "type": r[2],
        "dimension": r[3],
        "edge_weight_type": r[4],
        "comment": r[5],
        "edge_weight_format": r[6],
      }
      for r in rows
    ]

  def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
    tsp_row = conn.execute(
      """
      SELECT dimension, comment, edge_weight_type, edge_weight_format, tsplib_name, coords, display_coords
      FROM tsp_problems
      WHERE problem_id = ?
      """,
      [base.problem_id],
    ).fetchone()
    if tsp_row is None:
      raise InvalidProblemDataError(base.name, f"Missing record in tsp_problems for problem_id={base.problem_id}")
    dim, comment, edge_type, edge_format, tsplib_name, coords, display_coords = tsp_row
    return {
      "problem_id": base.problem_id,
      "name": base.name,
      "type": "TSP",
      "dimension": dim,
      "edge_type": edge_type,
      "edge_weight_format": edge_format,
      "tsplib_name": tsplib_name,
      "comment": comment,
      "capacity": None,
      "has_solution": base.has_solution,
      "source_file": base.source_file,
      "file_checksum": base.file_checksum,
      "raw_coords": coords,
      "raw_display_coords": display_coords,
      "raw_demands": None,
      "raw_depots": None,
    }


class AtspHandler:
  """Handler for Asymmetric Traveling Salesperson Problem (ATSP) instances."""

  problem_type: str = "ATSP"

  def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = conn.execute(
      """
      SELECT p.id, p.name, p.type, a.dimension, 'EXPLICIT' as edge_weight_type, a.comment, NULL as edge_weight_format
      FROM problems p
      JOIN atsp_problems a ON p.id = a.problem_id
      WHERE p.type = ?
      ORDER BY a.dimension ASC, p.name ASC
      """,
      [self.problem_type],
    ).fetchall()
    return [
      {
        "id": r[0],
        "name": r[1],
        "type": r[2],
        "dimension": r[3],
        "edge_weight_type": r[4],
        "comment": r[5],
        "edge_weight_format": r[6],
      }
      for r in rows
    ]

  def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
    atsp_row = conn.execute(
      """
      SELECT dimension, comment, tsplib_name
      FROM atsp_problems
      WHERE problem_id = ?
      """,
      [base.problem_id],
    ).fetchone()
    if atsp_row is None:
      raise InvalidProblemDataError(base.name, f"Missing record in atsp_problems for problem_id={base.problem_id}")
    dim, comment, tsplib_name = atsp_row
    return {
      "problem_id": base.problem_id,
      "name": base.name,
      "type": "ATSP",
      "dimension": dim,
      "edge_type": "EXPLICIT",
      "edge_weight_format": None,
      "tsplib_name": tsplib_name,
      "comment": comment,
      "capacity": None,
      "has_solution": base.has_solution,
      "source_file": base.source_file,
      "file_checksum": base.file_checksum,
      "raw_coords": None,
      "raw_display_coords": None,
      "raw_demands": None,
      "raw_depots": None,
    }


class CvrpHandler:
  """Handler for Capacitated Vehicle Routing Problem (CVRP) instances."""

  problem_type: str = "CVRP"

  def list_problems(self, conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = conn.execute(
      """
      SELECT p.id, p.name, p.type, c.dimension, c.edge_weight_type, c.comment, c.edge_weight_format
      FROM problems p
      JOIN cvrp_problems c ON p.id = c.problem_id
      WHERE p.type = ?
      ORDER BY c.dimension ASC, p.name ASC
      """,
      [self.problem_type],
    ).fetchall()
    return [
      {
        "id": r[0],
        "name": r[1],
        "type": r[2],
        "dimension": r[3],
        "edge_weight_type": r[4],
        "comment": r[5],
        "edge_weight_format": r[6],
      }
      for r in rows
    ]

  def load_metadata(self, conn: duckdb.DuckDBPyConnection, base: BaseProblemRow) -> ProblemMetadataRow:
    cvrp_row = conn.execute(
      """
      SELECT dimension, capacity, comment, edge_weight_type, edge_weight_format, tsplib_name, coords, demands, depots
      FROM cvrp_problems
      WHERE problem_id = ?
      """,
      [base.problem_id],
    ).fetchone()
    if cvrp_row is None:
      raise InvalidProblemDataError(base.name, f"Missing record in cvrp_problems for problem_id={base.problem_id}")
    dim, capacity, comment, edge_type, edge_format, tsplib_name, coords, demands, depots = cvrp_row
    return {
      "problem_id": base.problem_id,
      "name": base.name,
      "type": "CVRP",
      "dimension": dim,
      "edge_type": edge_type,
      "edge_weight_format": edge_format,
      "tsplib_name": tsplib_name,
      "comment": comment,
      "capacity": capacity,
      "has_solution": base.has_solution,
      "source_file": base.source_file,
      "file_checksum": base.file_checksum,
      "raw_coords": coords,
      "raw_display_coords": None,
      "raw_demands": demands,
      "raw_depots": depots,
    }


PROBLEM_TYPE_HANDLERS: dict[str, ProblemTypeHandler] = {
  "TSP": TspHandler(),
  "ATSP": AtspHandler(),
  "CVRP": CvrpHandler(),
}


def register_problem_type_handler(handler: ProblemTypeHandler) -> None:
  """Register or replace a handler in the global PROBLEM_TYPE_HANDLERS registry."""
  PROBLEM_TYPE_HANDLERS[handler.problem_type] = handler


__all__: list[str] = [
  "PROBLEM_TYPE_HANDLERS",
  "AtspHandler",
  "BaseProblemRow",
  "CvrpHandler",
  "ProblemMetadataRow",
  "ProblemTypeHandler",
  "TspHandler",
  "register_problem_type_handler",
]
