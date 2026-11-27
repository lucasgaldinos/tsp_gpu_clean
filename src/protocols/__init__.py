"""
Protocols for ISO-algorithmic Genetic Algorithms.

This package defines protocol interfaces for:
- Problem context (distance matrices, backend selection)
- Algorithm strategies (evolutionary operators)
- Standardized route/batch type aliases (`Route`, `Routes`, `RouteOrRoutes`, `Fitness`, `DistanceMatrix`)
"""

from .algorithm_strategies import (
    CrossoverStrategy,
    EvaluationStrategy,
    InitializationStrategy,
    LocalSearchStrategy,
    MutationStrategy,
    SelectionStrategy,
)
from .problem_context import ProblemContext
from .types import (
    Array,
    CupyArray,
    DeviceDistanceMatrix,
    DeviceFitness,
    DeviceRoute,
    DeviceRoutes,
    DistanceMatrix,
    Fitness,
    LocalSearchStage,
    Route,
    RouteOrRoutes,
    Routes,
    SeedLike,
)
from .vrp_strategies import (
    BinPackingStrategy,
    PenaltyPolicy,
    RouteAssessment,
    RouteDecoder,
    assess_with_evaluator,
    canonicalize_with_evaluator,
)

__all__: list[str] = [
    "Array",
    "BinPackingStrategy",
    "CrossoverStrategy",
    "CupyArray",
    "DeviceDistanceMatrix",
    "DeviceFitness",
    "DeviceRoute",
    "DeviceRoutes",
    "DistanceMatrix",
    "EvaluationStrategy",
    "Fitness",
    "InitializationStrategy",
    "LocalSearchStage",
    "LocalSearchStrategy",
    "MutationStrategy",
    "PenaltyPolicy",
    "ProblemContext",
    "Route",
    "RouteAssessment",
    "RouteDecoder",
    "RouteOrRoutes",
    "Routes",
    "SeedLike",
    "SelectionStrategy",
    "assess_with_evaluator",
    "canonicalize_with_evaluator",
]
