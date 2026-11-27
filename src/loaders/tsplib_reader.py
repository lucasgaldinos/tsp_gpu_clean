"""TSPLIB/VRP problem materialization, solution BKS pricing, and directory loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from converter.core.solutions import parse_solution_file
from converter.utils.cost import compute_routes_cost

from data_models.exceptions import (
  AmbiguousProblemError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
)
from data_models.problem import VALID_EDGE_TYPES, VRP_PROBLEM_TYPES, Problem, expected_coordinate_dimension
from distances.matrix import compute_distance_matrix
from distances.pairwise import get_distance_function
from protocols.problem_context import DEFAULT_MAX_MATRIX_DIMENSION

from .problem_type_handlers import build_vrp_constraints, validate_single_depot
from .tsplib_models import ProblemRowRecord, TsplibIngestionBundle, parse_tsplib_file

if TYPE_CHECKING:
  from types import TracebackType

logger = logging.getLogger(__name__)


def bundle_to_problem(
  bundle: TsplibIngestionBundle,
  max_eager_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
) -> Problem:
  """Convert a `TsplibIngestionBundle` into a validated `Problem` domain model."""
  meta: ProblemRowRecord = bundle.problem
  if meta.edge_weight_type not in VALID_EDGE_TYPES:
    raise InvalidEdgeTypeError(meta.edge_weight_type)

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
        if node.z is not None:
          raise InvalidProblemDataError(
            meta.name,
            f"EXPLICIT display coordinates require 2D coordinates, got 3D coordinate at node {node.node_id}",
          )
        display_list.append([node.x, node.y])
      if node.is_depot:
        depots_list.append(node.node_id)
    if len(display_list) == meta.dimension:
      display_coords = np.asarray(display_list, dtype=np.float64)
  else:
    if not bundle.nodes:
      raise MissingCoordinatesError(meta.name, meta.edge_weight_type)
    coords_list: list[list[float]] = []
    demands_list: list[int] = []
    expected_coord_dim: int = expected_coordinate_dimension(meta.edge_weight_type)
    for node in bundle.nodes:
      if node.x is None or node.y is None:
        raise MissingCoordinatesError(meta.name, meta.edge_weight_type)
      if expected_coord_dim == 3:
        if node.z is None:
          raise MissingCoordinatesError(meta.name, meta.edge_weight_type)
        coords_list.append([node.x, node.y, node.z])
      else:
        if node.z is not None:
          raise InvalidProblemDataError(
            meta.name,
            f"edge_type '{meta.edge_weight_type}' requires 2D coordinates, got 3D coordinate at node {node.node_id}",
          )
        coords_list.append([node.x, node.y])
      demands_list.append(node.demand)
      if node.is_depot:
        depots_list.append(node.node_id)

    coordinates = np.asarray(coords_list, dtype=np.float64)
    if meta.display_coords is not None and len(meta.display_coords) == meta.dimension:
      display_coords = np.asarray(meta.display_coords, dtype=np.float64)

    if meta.problem_type in VRP_PROBLEM_TYPES or any(demand > 0 for demand in demands_list):
      demands = np.asarray(demands_list, dtype=np.int32)

  depots_arr: np.ndarray | None = np.asarray(depots_list, dtype=np.int32) if depots_list else None
  validate_single_depot(meta.name, meta.problem_type, depots_arr)

  try:
    if distances is None and coordinates is not None and meta.dimension <= max_eager_matrix_dimension:
      distances = compute_distance_matrix(coordinates, meta.edge_weight_type)

    constraints = build_vrp_constraints(
      name=meta.name,
      problem_type=meta.problem_type,
      dimension=meta.dimension,
      capacity=meta.capacity,
      max_distance=meta.max_distance,
      vehicles=meta.vehicles,
      service_time=meta.service_time,
    )

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
      depots=depots_arr,
      source_file=meta.source_file,
      file_checksum=meta.file_checksum,
      constraints=constraints,
    )
  except ValueError as exc:
    raise InvalidProblemDataError(meta.name, str(exc)) from exc


def load_tsplib_file(
  tsp_path: str | Path,
  max_eager_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
) -> Problem:
  """Load a single `.tsp` or `.vrp` file directly into a `Problem` instance."""
  bundle: TsplibIngestionBundle = parse_tsplib_file(tsp_path)
  return bundle_to_problem(bundle, max_eager_matrix_dimension=max_eager_matrix_dimension)


def _normalize_tour_nodes(route: list[int]) -> list[int]:
  """Strip a repeated terminal city from a closed tour so cyclic evaluation never double-counts."""
  if len(route) > 1 and route[0] == route[-1]:
    return route[:-1]
  return route


def _compute_problem_routes_cost(problem: Problem, routes: list[list[int]]) -> float:
  """Evaluate 0-based TSP tour(s) using this solver's canonical TSPLIB95 distances."""
  if problem.distances is not None:
    total = 0
    for route in routes:
      nodes = _normalize_tour_nodes(route)
      n_nodes = len(nodes)
      if n_nodes < 2:
        continue
      for idx in range(n_nodes):
        total += int(problem.distances[nodes[idx], nodes[(idx + 1) % n_nodes]])
    return float(total)

  if problem.coordinates is not None:
    dist_fn = get_distance_function(problem.edge_type)
    total = 0
    for route in routes:
      nodes = _normalize_tour_nodes(route)
      n_nodes = len(nodes)
      if n_nodes < 2:
        continue
      for idx in range(n_nodes):
        total += int(dist_fn(problem.coordinates[nodes[idx]], problem.coordinates[nodes[(idx + 1) % n_nodes]]))
    return float(total)

  raise InvalidProblemDataError(problem.name, "Cannot evaluate tour cost without coordinates or distance matrix")


def resolve_opt_tour_cost(
  tsp_path: str | Path,
  *,
  bundle: TsplibIngestionBundle | None = None,
) -> float | None:
  """Resolve optimal or best-known cost from a sibling ``.opt.tour`` or ``.sol`` file."""
  path = Path(tsp_path)
  tour_candidates: tuple[Path, ...] = (
    path.parent / f"{path.stem}.opt.tour",
    path.parent / "tour" / f"{path.stem}.opt.tour",
  )
  sol_candidates: tuple[Path, ...] = (
    path.parent / f"{path.stem}.sol",
    path.parent / "sol" / f"{path.stem}.sol",
    path.parent / "solutions" / f"{path.stem}.sol",
  )
  candidates: tuple[Path, ...] = (
    (*sol_candidates, *tour_candidates) if path.suffix.lower() == ".vrp" else (*tour_candidates, *sol_candidates)
  )
  tour_file: Path | None = next((cand for cand in candidates if cand.is_file()), None)
  if tour_file is None:
    return None

  try:
    parsed = parse_solution_file(tour_file)
  except Exception as exc:
    logger.warning("Skipping unreadable solution file '%s': %s", tour_file, exc)
    return None

  raw_cost: float | None = parsed.get("stated_cost")
  if raw_cost is None:
    raw_cost = parsed.get("cost")
  if parsed["kind"] != "SOL" and raw_cost is not None:
    return float(raw_cost)

  routes: list[list[int]] = parsed["routes"]
  if not routes:
    return float(raw_cost) if raw_cost is not None else None

  resolved_bundle: TsplibIngestionBundle = bundle if bundle is not None else parse_tsplib_file(path)
  if parsed["kind"] != "SOL" and resolved_bundle.problem.problem_type == "TSP":
    problem = bundle_to_problem(resolved_bundle, max_eager_matrix_dimension=0)
    return _compute_problem_routes_cost(problem, routes)

  matrix = resolved_bundle.edge_matrix.matrix if resolved_bundle.edge_matrix is not None else None
  coords: list[list[float]] | None = None
  if resolved_bundle.nodes and all(node.x is not None and node.y is not None for node in resolved_bundle.nodes):
    coords = [
      [node.x, node.y] if node.z is None else [node.x, node.y, node.z]
      for node in resolved_bundle.nodes
      if node.x is not None and node.y is not None
    ]

  is_open_sol = parsed["kind"] == "SOL" and resolved_bundle.problem.problem_type == "OVRP"
  if parsed["kind"] == "SOL":
    depot = next((node.node_id for node in resolved_bundle.nodes if node.is_depot), 0)
    eval_routes: list[list[int]] = (
      [[depot, *route] for route in routes if route]
      if is_open_sol
      else [[depot, *route, depot] for route in routes if route]
    )
  else:
    eval_routes = routes

  computed = compute_routes_cost(
    eval_routes,
    matrix=matrix,
    coords=coords,
    edge_weight_type=resolved_bundle.problem.edge_weight_type,
    closed=not is_open_sol,
  )
  if computed is not None:
    return float(computed)
  return float(raw_cost) if raw_cost is not None else None


class TsplibLoader:
  """Context-managed loader that parses `.tsp` and `.vrp` files into `Problem` objects."""

  def __init__(self, tsp_dir: str | Path) -> None:
    self.tsp_dir = Path(tsp_dir)
    self._active: bool = False

  def __enter__(self) -> TsplibLoader:
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
    self._active = False

  def _resolve_instance_path(
    self,
    instance_name: str | Path,
    *,
    problem_type: str | None = None,
  ) -> tuple[Path, bool]:
    """Resolve one `.tsp`/`.vrp` path and report whether ``problem_type`` was derived from ``CVRP``."""
    if isinstance(instance_name, Path) or "/" in str(instance_name):
      candidate = Path(instance_name)
      if candidate.exists():
        return candidate, False
      instance_str: str = candidate.name
    else:
      instance_str = str(instance_name)

    stem: str = instance_str[:-4] if instance_str.endswith((".tsp", ".vrp")) else instance_str
    matches: list[Path] = [
      cand for cand in (self.tsp_dir / f"{stem}.tsp", self.tsp_dir / f"{stem}.vrp") if cand.is_file()
    ]
    if not matches:
      raise InstanceNotFoundError(stem)

    if problem_type is not None:
      target_type = problem_type.upper().strip()
      parsed_matches = [(m, parse_tsplib_file(m).problem.problem_type) for m in matches]
      for match_path, ptype in parsed_matches:
        if ptype == target_type:
          return match_path, False
      if target_type in ("OVRP", "DCVRP"):
        for match_path, ptype in parsed_matches:
          if ptype == "CVRP":
            return match_path, True
      raise InstanceNotFoundError(stem)

    if len(matches) > 1:
      found_types = [parse_tsplib_file(m).problem.problem_type for m in matches]
      raise AmbiguousProblemError(stem, found_types)

    return matches[0], False

  def load(
    self,
    instance_name: str | Path,
    problem_type: str | None = None,
    *,
    capacity: int | None = None,
    max_vehicles: int | None = None,
    max_route_distance: float | None = None,
    return_to_depot: bool | None = None,
  ) -> Problem:
    """Load a problem instance by name or path, optionally deriving variants or overriding constraints."""
    resolved_path, is_derived = self._resolve_instance_path(instance_name, problem_type=problem_type)
    problem = load_tsplib_file(resolved_path)
    effective_type = problem_type.upper().strip() if (is_derived and problem_type is not None) else None
    if effective_type is not None or any(
      v is not None for v in (capacity, max_vehicles, max_route_distance, return_to_depot)
    ):
      problem = problem.with_constraints(
        problem_type=effective_type,
        capacity=capacity,
        max_vehicles=max_vehicles,
        max_route_distance=max_route_distance,
        return_to_depot=return_to_depot,
      )
    return problem

  def resolve_optimal_cost(
    self,
    instance_name: str | Path,
    problem_type: str | None = None,
  ) -> float | None:
    """Resolve optimal/best-known cost from sibling solution files (returns None for derived variants)."""
    resolved_path, is_derived = self._resolve_instance_path(instance_name, problem_type=problem_type)
    if is_derived:
      return None
    return resolve_opt_tour_cost(resolved_path)

  def list_instances(self) -> list[str]:
    """Return sorted list of available `.tsp` and `.vrp` problem names in `tsp_dir`."""
    if not self.tsp_dir.exists():
      return []
    return sorted({path.stem for ext in ("*.tsp", "*.vrp") for path in self.tsp_dir.glob(ext)})

  def list_instance_files(self) -> list[Path]:
    """Return sorted `.tsp` and `.vrp` paths in `tsp_dir`."""
    if not self.tsp_dir.exists():
      return []
    return sorted([*self.tsp_dir.glob("*.tsp"), *self.tsp_dir.glob("*.vrp")])


__all__: list[str] = [
  "TsplibLoader",
  "bundle_to_problem",
  "load_tsplib_file",
  "resolve_opt_tour_cost",
]
