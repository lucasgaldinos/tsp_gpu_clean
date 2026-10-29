"""
Protocols for ISO-algorithmic Genetic Algorithms.

This package defines protocol interfaces for:
- Problem context (distance matrices, backend selection)
- Algorithm strategies (evolutionary operators and metaheuristics)
- Standardized route/batch type aliases (`Route`, `Routes`, `RouteOrRoutes`, `Fitness`, `DistanceMatrix`)
"""

from .algorithm_strategies import (
  CrossoverStrategy,
  DistanceMatrix,
  EvaluationStrategy,
  Fitness,
  InitializationStrategy,
  LocalSearchStage,
  LocalSearchStrategy,
  MetaheuristicStrategy,
  MutationStrategy,
  Route,
  RouteOrRoutes,
  Routes,
  SelectionStrategy,
  ensure_routes_batch,
  restore_routes_shape,
)
from .problem_context import ProblemContext

__all__ = [
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
  "SelectionStrategy",
  "ensure_routes_batch",
  "restore_routes_shape",
]
