"""
Problem dataclass for immutable routing problem representation.

This module provides the core data structure for representing TSP, ATSP,
and CVRP problem instances. The Problem class is an immutable value object
that encapsulates all data needed to solve a routing optimization problem.

Design Philosophy:
- Immutability (frozen dataclass) prevents accidental modifications
- Type safety through comprehensive type hints
- Value semantics (equality based on content, not identity)
- Backend agnostic (works with NumPy or CuPy arrays)

Example:
    >>> with DatabaseLoader("path/to/corpus.duckdb") as loader:
    ...   problem = loader.load("berlin52")
    ...   print(f"{problem.name}: {problem.dimension} nodes, {problem.edge_type}")
    berlin52: 52 nodes, EUC_2D
"""

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from data_models.exceptions import InvalidProblemDataError
from data_models.vrp_constraints import VRPConstraints

VRP_PROBLEM_TYPES: frozenset[str] = frozenset({"CVRP", "OVRP", "DCVRP"})
VALID_PROBLEM_TYPES: frozenset[str] = frozenset({"TSP", "ATSP"}) | VRP_PROBLEM_TYPES
VALID_EDGE_TYPES: frozenset[str] = frozenset({"EUC_2D", "EUC_3D", "CEIL_2D", "GEO", "ATT", "EXPLICIT"})
THREE_DIMENSIONAL_EDGE_TYPES: frozenset[str] = frozenset({"EUC_3D"})
_UNSET_CONSTRAINTS: VRPConstraints = VRPConstraints()


def expected_coordinate_dimension(edge_type: str) -> int:
  """Return expected coordinate width (`3` for 3D edge types, `2` for 2D/EXPLICIT).

  Raises:
      ValueError: If ``edge_type`` is not in ``VALID_EDGE_TYPES``.
  """
  if edge_type not in VALID_EDGE_TYPES:
    raise ValueError(f"Unsupported edge_type: '{edge_type}'. Valid types: {sorted(VALID_EDGE_TYPES)}")
  return 3 if edge_type in THREE_DIMENSIONAL_EDGE_TYPES else 2

_ARRAY_FIELDS: frozenset[str] = frozenset({"coordinates", "distances", "demands", "display_coords", "depots"})


def _arrays_equal(first: np.ndarray | None, second: np.ndarray | None) -> bool:
  """Compare two optional arrays by content, treating `None` as equal only to `None`."""
  if first is None or second is None:
    return first is second
  return bool(np.array_equal(first, second))


def _array_hash_key(value: np.ndarray | None) -> tuple[tuple[int, ...], str, bytes] | None:
  """Build a hashable content key for an optional array."""
  if value is None:
    return None
  return value.shape, value.dtype.str, value.tobytes()


@dataclass(frozen=True)
class Problem:
  """
  Immutable representation of a routing problem instance.

  This dataclass encapsulates all information needed to solve a routing
  optimization problem (TSP, ATSP, or CVRP). The frozen=True parameter
  makes instances immutable, providing referential transparency and
  thread safety.

  Attributes:
      name: Instance identifier (e.g., 'berlin52', 'br17', 'eil22')
      dimension: Number of nodes/cities in the problem
      problem_type: Problem variant - 'TSP', 'ATSP', or 'CVRP'
      edge_type: Distance calculation method:
          - 'EUC_2D': Euclidean distance in 2D plane
          - 'EUC_3D': Euclidean distance in 3D space
          - 'CEIL_2D': Ceiling of Euclidean distance in 2D plane
          - 'GEO': Geographic distance (haversine formula)
          - 'ATT': Pseudo-Euclidean distance (special TSPLIB format)
          - 'EXPLICIT': Pre-computed distance matrix
      coordinates: Node coordinates, shape (n, 2) or (n, 3) for EUC_3D
          - None for ATSP instances with EXPLICIT edge_type
          - NumPy array for TSP/CVRP with geometric coordinates
      distances: Explicit distance matrix, shape (n, n)
          - Required for ATSP (asymmetric distances)
          - None for TSP/CVRP (computed from coordinates)
      capacity: Vehicle capacity constraint (CVRP only)
          - None for TSP/ATSP
          - Positive integer for CVRP
      demands: Node demands, shape (n,)
          - None for TSP/ATSP
          - NumPy array for CVRP (demands[0] should be 0 for depot)

  Invariants:
      - dimension > 0
      - problem_type in {'TSP', 'ATSP', 'CVRP'}
      - edge_type in {'EUC_2D', 'EUC_3D', 'CEIL_2D', 'GEO', 'ATT', 'EXPLICIT'}
      - If edge_type != 'EXPLICIT': coordinates is not None
      - If edge_type == 'EXPLICIT': distances is not None
      - If problem_type == 'CVRP': capacity is not None and demands is not None
      - If coordinates is not None: coordinates.shape == (dimension, 3) for EUC_3D, else (dimension, 2)
      - If distances is not None: distances.shape == (dimension, dimension)
      - If demands is not None: demands.shape == (dimension,)

  Note on Immutability:
      While the dataclass is frozen (cannot reassign attributes), NumPy
      arrays stored in fields can still be modified in-place. Users should
      treat all arrays as read-only. If modification is needed, create a
      new Problem instance with updated arrays.

  Example - TSP Instance:
      >>> problem = Problem(
      ...   name="berlin52",
      ...   dimension=52,
      ...   problem_type="TSP",
      ...   edge_type="EUC_2D",
      ...   coordinates=np.array([[565, 575], [25, 185], ...]),  # shape (52, 2)
      ...   distances=None,
      ...   capacity=None,
      ...   demands=None,
      ... )

  Example - ATSP Instance:
      >>> problem = Problem(
      ...   name="br17",
      ...   dimension=17,
      ...   problem_type="ATSP",
      ...   edge_type="EXPLICIT",
      ...   coordinates=None,  # ATSP uses explicit distances
      ...   distances=asymmetric_matrix,  # shape (17, 17)
      ...   capacity=None,
      ...   demands=None,
      ... )

  Example - CVRP Instance:
      >>> problem = Problem(
      ...   name="eil22",
      ...   dimension=22,
      ...   problem_type="CVRP",
      ...   edge_type="EUC_2D",
      ...   coordinates=np.array([[...], ...]),  # shape (22, 2)
      ...   distances=None,
      ...   capacity=6000,
      ...   demands=np.array([0, 1200, 1500, ...]),  # shape (22,), depot has demand=0
      ... )
  """

  name: str
  dimension: int
  problem_type: str
  edge_type: str
  coordinates: np.ndarray | None
  distances: np.ndarray | None
  capacity: int | None = None
  demands: np.ndarray | None = None
  # Extended metadata from datasets/tsp.duckdb / TSPLIB95
  problem_id: int | None = None
  comment: str | None = None
  edge_weight_format: str | None = None
  tsplib_name: str | None = None
  has_solution: bool = False
  display_coords: np.ndarray | None = None
  depots: np.ndarray | None = None
  source_file: str | None = None
  file_checksum: str | None = None
  constraints: VRPConstraints = _UNSET_CONSTRAINTS

  def to_metadata_dict(self, optimal_cost: float | None = None) -> dict[str, Any]:
    """Return JSON-serializable problem metadata dictionary for checkpoints and statistics."""
    return {
      "problem_id": self.problem_id,
      "name": self.name,
      "tsplib_name": self.tsplib_name or self.name,
      "problem_type": self.problem_type,
      "dimension": self.dimension,
      "edge_type": self.edge_type,
      "edge_weight_type": self.edge_type,
      "edge_weight_format": self.edge_weight_format,
      "is_symmetric": self.is_symmetric,
      "comment": self.comment,
      "capacity": self.capacity,
      "has_coordinates": self.coordinates is not None,
      "has_display_coords": self.display_coords is not None,
      "has_explicit_matrix": self.edge_type == "EXPLICIT",
      "has_solution": self.has_solution,
      "optimal_cost": optimal_cost,
      "source_file": self.source_file,
      "file_checksum": self.file_checksum,
    }

  def with_constraints(
    self,
    *,
    problem_type: str | None = None,
    capacity: int | None = None,
    max_vehicles: int | None = None,
    max_route_distance: float | None = None,
    return_to_depot: bool | None = None,
  ) -> "Problem":
    """Return a new `Problem` with declarative constraint/variant overrides applied (`DP17-10`)."""
    target_type = problem_type.upper().strip() if problem_type is not None else self.problem_type
    new_capacity = capacity if capacity is not None else self.constraints.capacity
    new_max_vehicles = max_vehicles if max_vehicles is not None else self.constraints.max_vehicles
    new_max_distance = max_route_distance if max_route_distance is not None else self.constraints.max_route_distance
    if return_to_depot is not None:
      new_return = return_to_depot
    elif problem_type is not None and target_type == "OVRP":
      new_return = False
    else:
      new_return = self.constraints.return_to_depot

    new_constraints = replace(
      self.constraints,
      capacity=new_capacity,
      max_vehicles=new_max_vehicles,
      max_route_distance=new_max_distance,
      return_to_depot=new_return,
    )
    return replace(
      self,
      problem_type=target_type,
      capacity=new_capacity,
      constraints=new_constraints,
    )

  def __post_init__(self) -> None:
    """
    Validate problem instance after creation.

    Raises:
        ValueError: If invariants are violated
    """
    # Validate dimension
    if self.dimension <= 0:
      raise ValueError(f"dimension must be positive, got {self.dimension}")

    # Validate problem_type
    if self.problem_type not in VALID_PROBLEM_TYPES:
      raise ValueError(f"problem_type must be one of {set(VALID_PROBLEM_TYPES)}, got '{self.problem_type}'")

    # Validate edge_type
    if self.edge_type not in VALID_EDGE_TYPES:
      raise ValueError(f"edge_type must be one of {set(VALID_EDGE_TYPES)}, got '{self.edge_type}'")

    # Validate coordinates/distances relationship
    if self.edge_type == "EXPLICIT":
      if self.distances is None:
        raise ValueError("edge_type='EXPLICIT' requires distances matrix, got None")
    else:
      if self.coordinates is None:
        raise ValueError(f"edge_type='{self.edge_type}' requires coordinates, got None")

    # Validate array shapes, dtypes, finiteness, and contiguity if present
    if self.coordinates is not None:
      expected_coord_dim = expected_coordinate_dimension(self.edge_type)
      if self.coordinates.shape != (self.dimension, expected_coord_dim):
        raise ValueError(
          f"coordinates.shape should be ({self.dimension}, {expected_coord_dim}), got {self.coordinates.shape}"
        )
      if not (
        np.issubdtype(self.coordinates.dtype, np.integer)
        or np.issubdtype(self.coordinates.dtype, np.floating)
      ):
        raise ValueError(f"coordinates must have a real numeric dtype, got {self.coordinates.dtype}")
      coords_f64: np.ndarray = np.ascontiguousarray(self.coordinates, dtype=np.float64)
      if not np.isfinite(coords_f64).all():
        raise ValueError("coordinates must contain only finite values (no NaN or Inf)")
      if coords_f64 is not self.coordinates:
        object.__setattr__(self, "coordinates", coords_f64)

    if self.display_coords is not None:
      if self.display_coords.shape != (self.dimension, 2):
        raise ValueError(
          f"display_coords.shape should be ({self.dimension}, 2), got {self.display_coords.shape}"
        )
      if not (
        np.issubdtype(self.display_coords.dtype, np.integer)
        or np.issubdtype(self.display_coords.dtype, np.floating)
      ):
        raise ValueError(f"display_coords must have a real numeric dtype, got {self.display_coords.dtype}")
      display_f64: np.ndarray = np.ascontiguousarray(self.display_coords, dtype=np.float64)
      if not np.isfinite(display_f64).all():
        raise ValueError("display_coords must contain only finite values (no NaN or Inf)")
      if display_f64 is not self.display_coords:
        object.__setattr__(self, "display_coords", display_f64)

    if self.distances is not None:
      if self.distances.shape != (self.dimension, self.dimension):
        raise ValueError(f"distances.shape should be ({self.dimension}, {self.dimension}), got {self.distances.shape}")

    # Validate and synchronize VRPConstraints with capacity
    if not isinstance(self.constraints, VRPConstraints):
      raise ValueError(f"constraints must be a VRPConstraints instance, got {type(self.constraints).__name__}")

    if self.capacity is not None:
      if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
        raise ValueError(f"capacity must be a positive int, got {type(self.capacity).__name__}: {self.capacity!r}")
      if self.capacity <= 0:
        raise ValueError(f"capacity must be positive, got {self.capacity}")

    constraints_omitted = self.constraints is _UNSET_CONSTRAINTS
    if constraints_omitted:
      default_return = self.problem_type != "OVRP"
      if self.capacity is not None or not default_return:
        object.__setattr__(
          self,
          "constraints",
          VRPConstraints(capacity=self.capacity, return_to_depot=default_return),
        )
    else:
      if self.problem_type == "OVRP" and self.constraints.return_to_depot:
        raise InvalidProblemDataError(self.name, "OVRP instances require constraints.return_to_depot=False")
      if (
        self.capacity is not None
        and self.constraints.capacity is not None
        and self.capacity != self.constraints.capacity
      ):
        raise InvalidProblemDataError(
          self.name,
          (
            f"Conflicting capacity values: problem.capacity={self.capacity} "
            f"!= constraints.capacity={self.constraints.capacity}"
          ),
        )
      if self.capacity is None and self.constraints.capacity is not None:
        object.__setattr__(self, "capacity", self.constraints.capacity)
      elif self.capacity is not None and self.constraints.capacity is None:
        object.__setattr__(self, "constraints", replace(self.constraints, capacity=self.capacity))

    # Validate VRP requirements (CVRP, OVRP, DCVRP)
    if self.problem_type in VRP_PROBLEM_TYPES:
      if self.capacity is None:
        raise ValueError(f"{self.problem_type} instances require capacity")

      if self.demands is None:
        raise ValueError(f"{self.problem_type} instances require demands")
      if self.demands.shape != (self.dimension,):
        raise ValueError(f"demands.shape should be ({self.dimension},), got {self.demands.shape}")
      if self.demands.dtype == np.dtype(bool) or not np.issubdtype(self.demands.dtype, np.integer):
        raise ValueError(f"demands must have an integer dtype, got {self.demands.dtype}")
      demands_i32: np.ndarray = np.ascontiguousarray(self.demands, dtype=np.int32)
      if int(demands_i32[0]) != 0:
        raise ValueError(f"Depot demand demands[0] must be 0, got {int(demands_i32[0])}")
      if bool(np.any(demands_i32 < 0)):
        raise ValueError("demands must be non-negative")
      if demands_i32 is not self.demands:
        object.__setattr__(self, "demands", demands_i32)

      if self.problem_type == "DCVRP" and self.constraints.max_route_distance is None:
        raise ValueError("DCVRP instances require constraints.max_route_distance")

    self.constraints.validate_against_dimension(self.dimension)

  def __eq__(self, other: object) -> bool:
    """
    Compare two problems by content instead of dataclass field identity.

    NumPy arrays are compared with `np.array_equal` (and `None` fields are only
    equal to other `None` fields) so two independently built instances with the
    same data compare equal without raising `ValueError`.
    """
    if not isinstance(other, Problem):
      return NotImplemented

    for field in fields(self):
      if field.name in _ARRAY_FIELDS:
        if not _arrays_equal(getattr(self, field.name), getattr(other, field.name)):
          return False
      elif getattr(self, field.name) != getattr(other, field.name):
        return False
    return True

  def __hash__(self) -> int:
    """Hash problem content so frozen instances stay usable in sets and dict keys."""
    return hash(
      tuple(
        _array_hash_key(getattr(self, field.name)) if field.name in _ARRAY_FIELDS else getattr(self, field.name)
        for field in fields(self)
      )
    )

  def __repr__(self) -> str:
    """
    Detailed string representation for debugging.

    Returns:
        String with all problem attributes
    """
    coords_shape = self.coordinates.shape if self.coordinates is not None else None
    dists_shape = self.distances.shape if self.distances is not None else None
    demands_shape = self.demands.shape if self.demands is not None else None

    return (
      f"Problem("
      f"name='{self.name}', "
      f"dimension={self.dimension}, "
      f"problem_type='{self.problem_type}', "
      f"edge_type='{self.edge_type}', "
      f"coordinates={coords_shape}, "
      f"distances={dists_shape}, "
      f"capacity={self.capacity}, "
      f"demands={demands_shape}"
      f")"
    )

  def __str__(self) -> str:
    """
    Human-readable string representation.

    Returns:
        Brief problem description
    """
    if self.problem_type in VRP_PROBLEM_TYPES:
      return f"{self.name}: {self.problem_type} with {self.dimension} nodes, capacity={self.capacity}"
    else:
      return f"{self.name}: {self.problem_type} with {self.dimension} nodes"

  @property
  def is_symmetric(self) -> bool:
    """
    Check if problem has symmetric distances.

    Returns:
        True for TSP and VRP variants (CVRP, OVRP, DCVRP), False for ATSP
    """
    return self.problem_type != "ATSP"

  @property
  def has_capacity_constraints(self) -> bool:
    """
    Check if problem has vehicle capacity constraints.

    Returns:
        True for VRP variants or problems with explicit capacity constraints, False otherwise
    """
    return self.problem_type in VRP_PROBLEM_TYPES or self.constraints.capacity is not None

  @property
  def is_vrp(self) -> bool:
    """Return ``True`` when this instance is a VRP variant or carries active constraints."""
    return self.problem_type in VRP_PROBLEM_TYPES or self.constraints.is_constrained

  @property
  def default_evolutionary_nodes(self) -> list[int]:
    """Return default node sequence for GA evolution (`1..n-1` for VRP variants, `0..n-1` otherwise)."""
    start = 1 if self.problem_type in VRP_PROBLEM_TYPES else 0
    return list(range(start, self.dimension))

  @property
  def requires_distance_computation(self) -> bool:
    """
    Check if distances need to be computed from coordinates.

    Returns:
        True if edge_type is geometric (EUC_2D, EUC_3D, CEIL_2D, GEO, ATT),
        False if EXPLICIT (distances pre-computed)
    """
    return self.edge_type != "EXPLICIT"
