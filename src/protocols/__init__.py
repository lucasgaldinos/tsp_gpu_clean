"""
Protocols for ISO-algorithmic Genetic Algorithms.

This package defines protocol interfaces for:
- Problem context (distance matrices, backend selection)
- Algorithm strategies (evolutionary operators and metaheuristics)
- Standardized route/batch type aliases (`Route`, `Routes`, `RouteOrRoutes`, `Fitness`, `DistanceMatrix`)
"""

from .algorithm_strategies import (
  CrossoverStrategy,
  EvaluationStrategy,
  InitializationStrategy,
  LocalSearchStrategy,
  MetaheuristicStrategy,
  MutationStrategy,
  SelectionStrategy,
  ensure_routes_batch,
  restore_routes_shape,
)
from .problem_context import ProblemContext
from .types import (
  DistanceMatrix,
  Fitness,
  LocalSearchStage,
  Route,
  RouteOrRoutes,
  Routes,
  SeedLike,
)

__all__: list[str] = [
  "CrossoverStrategy",
  "DistanceMatrix",
  "EvaluationStrategy",
  "Fitness",
  "InitializationStrategy",
  "LocalSearchStage",
  "LocalSearchStrategy",
  "MetaheuristicStrategy",
  "MutationStrategy",
  "ProblemContext",
  "Route",
  "RouteOrRoutes",
  "Routes",
  "SeedLike",
  "SelectionStrategy",
  "ensure_routes_batch",
  "restore_routes_shape",
]
