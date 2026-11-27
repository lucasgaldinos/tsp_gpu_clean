"""
Core data models for routing problem representation.

This package provides immutable dataclasses for representing TSP, ATSP,
and CVRP problem instances.
"""

from .exceptions import (
  AmbiguousProblemError,
  GPUUnavailableError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
  ProblemLoaderError,
)
from .problem import VALID_PROBLEM_TYPES, VRP_PROBLEM_TYPES, Problem
from .vrp_constraints import VRPConstraints

__all__ = [
  "VALID_PROBLEM_TYPES",
  "VRP_PROBLEM_TYPES",
  "AmbiguousProblemError",
  "GPUUnavailableError",
  "InstanceNotFoundError",
  "InvalidEdgeTypeError",
  "InvalidProblemDataError",
  "MissingCoordinatesError",
  "Problem",
  "ProblemLoaderError",
  "VRPConstraints",
]
