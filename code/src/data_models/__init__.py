"""
Core data models for routing problem representation.

This package provides immutable dataclasses for representing TSP, ATSP,
and CVRP problem instances.
"""

from .problem import Problem
from .exceptions import (
    ProblemLoaderError,
    InstanceNotFoundError,
    InvalidEdgeTypeError,
    MissingCoordinatesError,
)

__all__ = [
    "Problem",
    "ProblemLoaderError",
    "InstanceNotFoundError",
    "InvalidEdgeTypeError",
    "MissingCoordinatesError",
]
