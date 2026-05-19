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
    >>> with DatabaseLoader() as loader:
    ...     problem = loader.load('berlin52')
    ...     print(f"{problem.name}: {problem.dimension} nodes, {problem.edge_type}")
    berlin52: 52 nodes, EUC_2D
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


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
            - 'GEO': Geographic distance (haversine formula)
            - 'ATT': Pseudo-Euclidean distance (special TSPLIB format)
            - 'EXPLICIT': Pre-computed distance matrix
        coordinates: Node coordinates, shape (n, 2)
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
        - edge_type in {'EUC_2D', 'GEO', 'ATT', 'EXPLICIT'}
        - If edge_type != 'EXPLICIT': coordinates is not None
        - If edge_type == 'EXPLICIT': distances is not None
        - If problem_type == 'CVRP': capacity is not None and demands is not None
        - If coordinates is not None: coordinates.shape == (dimension, 2)
        - If distances is not None: distances.shape == (dimension, dimension)
        - If demands is not None: demands.shape == (dimension,)

    Note on Immutability:
        While the dataclass is frozen (cannot reassign attributes), NumPy
        arrays stored in fields can still be modified in-place. Users should
        treat all arrays as read-only. If modification is needed, create a
        new Problem instance with updated arrays.

    Example - TSP Instance:
        >>> problem = Problem(
        ...     name='berlin52',
        ...     dimension=52,
        ...     problem_type='TSP',
        ...     edge_type='EUC_2D',
        ...     coordinates=np.array([[565, 575], [25, 185], ...]),  # shape (52, 2)
        ...     distances=None,
        ...     capacity=None,
        ...     demands=None
        ... )

    Example - ATSP Instance:
        >>> problem = Problem(
        ...     name='br17',
        ...     dimension=17,
        ...     problem_type='ATSP',
        ...     edge_type='EXPLICIT',
        ...     coordinates=None,  # ATSP uses explicit distances
        ...     distances=asymmetric_matrix,  # shape (17, 17)
        ...     capacity=None,
        ...     demands=None
        ... )

    Example - CVRP Instance:
        >>> problem = Problem(
        ...     name='eil22',
        ...     dimension=22,
        ...     problem_type='CVRP',
        ...     edge_type='EUC_2D',
        ...     coordinates=np.array([[...], ...]),  # shape (22, 2)
        ...     distances=None,
        ...     capacity=6000,
        ...     demands=np.array([0, 1200, 1500, ...])  # shape (22,), depot has demand=0
        ... )
    """

    name: str
    dimension: int
    problem_type: str
    edge_type: str
    coordinates: Optional[np.ndarray]
    distances: Optional[np.ndarray]
    capacity: Optional[int] = None
    demands: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Validate problem instance after creation.

        Raises:
            ValueError: If invariants are violated
        """
        # Validate dimension
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")

        # Validate problem_type
        valid_types = {"TSP", "ATSP", "CVRP"}
        if self.problem_type not in valid_types:
            raise ValueError(
                f"problem_type must be one of {valid_types}, got '{self.problem_type}'"
            )

        # Validate edge_type
        valid_edge_types = {"EUC_2D", "GEO", "ATT", "EXPLICIT"}
        if self.edge_type not in valid_edge_types:
            raise ValueError(
                f"edge_type must be one of {valid_edge_types}, got '{self.edge_type}'"
            )

        # Validate coordinates/distances relationship
        if self.edge_type == "EXPLICIT":
            if self.distances is None:
                raise ValueError(
                    f"edge_type='EXPLICIT' requires distances matrix, got None"
                )
        else:
            if self.coordinates is None:
                raise ValueError(
                    f"edge_type='{self.edge_type}' requires coordinates, got None"
                )

        # Validate array shapes if present
        if self.coordinates is not None:
            if self.coordinates.shape != (self.dimension, 2):
                raise ValueError(
                    f"coordinates.shape should be ({self.dimension}, 2), "
                    f"got {self.coordinates.shape}"
                )

        if self.distances is not None:
            if self.distances.shape != (self.dimension, self.dimension):
                raise ValueError(
                    f"distances.shape should be ({self.dimension}, {self.dimension}), "
                    f"got {self.distances.shape}"
                )

        # Validate CVRP requirements
        if self.problem_type == "CVRP":
            if self.capacity is None:
                raise ValueError("CVRP instances require capacity")
            if self.capacity <= 0:
                raise ValueError(f"capacity must be positive, got {self.capacity}")

            if self.demands is None:
                raise ValueError("CVRP instances require demands")
            if self.demands.shape != (self.dimension,):
                raise ValueError(
                    f"demands.shape should be ({self.dimension},), "
                    f"got {self.demands.shape}"
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
        if self.problem_type == "CVRP":
            return f"{self.name}: {self.problem_type} with {self.dimension} nodes, capacity={self.capacity}"
        else:
            return f"{self.name}: {self.problem_type} with {self.dimension} nodes"

    @property
    def is_symmetric(self) -> bool:
        """
        Check if problem has symmetric distances.

        Returns:
            True for TSP and CVRP (symmetric), False for ATSP (asymmetric)
        """
        return self.problem_type in {"TSP", "CVRP"}

    @property
    def has_capacity_constraints(self) -> bool:
        """
        Check if problem has vehicle capacity constraints.

        Returns:
            True for CVRP, False for TSP/ATSP
        """
        return self.problem_type == "CVRP"

    @property
    def requires_distance_computation(self) -> bool:
        """
        Check if distances need to be computed from coordinates.

        Returns:
            True if edge_type is geometric (EUC_2D, GEO, ATT),
            False if EXPLICIT (distances pre-computed)
        """
        return self.edge_type != "EXPLICIT"
