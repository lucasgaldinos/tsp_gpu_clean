"""
Problem type handlers for normalized DuckDB database loading.

Encapsulates problem-type specific SQL queries and metadata extraction
(TSP, ATSP, CVRP, and extensibility for future formulations like VRPTW, MDVRP)
without conditional branching in DatabaseLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

import numpy as np

from data_models.exceptions import InvalidProblemDataError
from data_models.problem import VRP_PROBLEM_TYPES
from data_models.vrp_constraints import VRPConstraints

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
  max_distance: NotRequired[float | None]
  vehicles: NotRequired[int | None]
  service_time: NotRequired[float | None]


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
    normalized_edge_type: str = edge_type.upper().strip() if isinstance(edge_type, str) else edge_type
    return {
      "problem_id": base.problem_id,
      "name": base.name,
      "type": "TSP",
      "dimension": dim,
      "edge_type": normalized_edge_type,
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
      "max_distance": None,
      "vehicles": None,
      "service_time": None,
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
      "max_distance": None,
      "vehicles": None,
      "service_time": None,
    }


class CvrpHandler:
  """Handler for Capacitated Vehicle Routing Problem variants (`CVRP`, `OVRP`, `DCVRP`)."""

  def __init__(self, problem_type: str = "CVRP") -> None:
    self.problem_type: str = problem_type

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
      SELECT dimension, capacity, comment, edge_weight_type, edge_weight_format,
             tsplib_name, coords, demands, depots,
             display_coords, max_distance, vehicles, service_time
      FROM cvrp_problems
      WHERE problem_id = ?
      """,
      [base.problem_id],
    ).fetchone()
    if cvrp_row is None:
      raise InvalidProblemDataError(base.name, f"Missing record in cvrp_problems for problem_id={base.problem_id}")
    (
      dim,
      capacity,
      comment,
      edge_type,
      edge_format,
      tsplib_name,
      coords,
      demands,
      depots,
      display_coords,
      max_distance,
      vehicles,
      service_time,
    ) = cvrp_row
    normalized_edge_type: str = edge_type.upper().strip() if isinstance(edge_type, str) else edge_type
    return {
      "problem_id": base.problem_id,
      "name": base.name,
      "type": base.problem_type,
      "dimension": dim,
      "edge_type": normalized_edge_type,
      "edge_weight_format": edge_format,
      "tsplib_name": tsplib_name,
      "comment": comment,
      "capacity": capacity,
      "has_solution": base.has_solution,
      "source_file": base.source_file,
      "file_checksum": base.file_checksum,
      "raw_coords": coords,
      "raw_display_coords": display_coords,
      "raw_demands": demands,
      "raw_depots": depots,
      "max_distance": float(max_distance) if max_distance is not None else None,
      "vehicles": int(vehicles) if vehicles is not None else None,
      "service_time": float(service_time) if service_time is not None else None,
    }


def validate_single_depot(name: str, problem_type: str, depots: np.ndarray | None) -> None:
  """Reject multi-depot (`MDVRP`) or non-zero-depot payloads outside single-depot VRP scope."""
  if problem_type not in VRP_PROBLEM_TYPES or depots is None or depots.size == 0:
    return
  if depots.size > 1 or int(depots[0]) != 0:
    raise InvalidProblemDataError(
      name,
      f"Multi-depot or non-zero depot VRP (MDVRP, depots={depots.tolist()}) "
      f"is outside single-depot {problem_type} scope.",
    )


def build_vrp_constraints(
  *,
  name: str,
  problem_type: str,
  dimension: int,
  capacity: int | None,
  max_distance: float | None = None,
  vehicles: int | None = None,
  service_time: float | None = None,
  return_to_depot: bool | None = None,
) -> VRPConstraints:
  """Construct validated `VRPConstraints` from flat VRP Tier 1 TSPLIB/DuckDB fields."""
  service_times: np.ndarray | None = None
  if service_time is not None:
    st_val = float(service_time)
    if st_val < 0.0:
      raise InvalidProblemDataError(name, f"service_time must be >= 0, got {st_val}")
    if st_val > 0.0:
      service_times = np.full(dimension, st_val, dtype=np.float64)
      service_times[0] = 0.0
  effective_return = return_to_depot if return_to_depot is not None else (problem_type != "OVRP")
  return VRPConstraints(
    capacity=capacity,
    max_vehicles=vehicles,
    max_route_distance=max_distance,
    return_to_depot=effective_return,
    service_times=service_times,
  )


PROBLEM_TYPE_HANDLERS: dict[str, ProblemTypeHandler] = {
  "TSP": TspHandler(),
  "ATSP": AtspHandler(),
  "CVRP": CvrpHandler("CVRP"),
  "OVRP": CvrpHandler("OVRP"),
  "DCVRP": CvrpHandler("DCVRP"),
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
  "build_vrp_constraints",
  "register_problem_type_handler",
  "validate_single_depot",
]
