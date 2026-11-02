"""
Core data models for routing problem representation.

This package provides immutable dataclasses for representing TSP, ATSP,
and CVRP problem instances.
"""

from .exceptions import (
  GPUUnavailableError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  MissingCoordinatesError,
  ProblemLoaderError,
)
from .problem import Problem

__all__ = [
  "GPUUnavailableError",
  "InstanceNotFoundError",
  "InvalidEdgeTypeError",
  "MissingCoordinatesError",
  "Problem",
  "ProblemLoaderError",
]
