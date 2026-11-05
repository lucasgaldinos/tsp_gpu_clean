"""
Modular evolutionary operator strategies for TSP and routing optimization.

All route-transforming operators (`CrossoverStrategy`, `MutationStrategy`,
`LocalSearchStrategy`, `EvaluationStrategy`) accept `RouteOrRoutes`
(`(n_cities,)` or `(N, n_cities)`), enabling arbitrary pipeline ordering.

Every class name is prefixed with the slot it fills (`Initialization*`, `Selection*`,
`Crossover*`, `Mutation*`, `LocalSearch*`, `Evaluation*`) and every class derives from
`StrategyBase`, so the registry key reads as "which slot, which variant" (`DP11-1`).
"""

from .base import StrategyBase
from .crossover import (
  CrossoverGPUOrder,
  CrossoverOrder,
  CrossoverPartiallyMapped,
)
from .evaluation import (
  EvaluationCPUCost,
  EvaluationGPUCost,
)
from .initialization import InitializationNearestNeighbor, InitializationRandom
from .local_search import LocalSearchCPUTwoOpt, LocalSearchGPUTwoOpt, LocalSearchNoOp
from .mutation import (
  MutationGPUInversion,
  MutationGPUSwap,
  MutationInversion,
  MutationSwap,
)
from .selection import (
  SelectionGPUTournament,
  SelectionRouletteWheel,
  SelectionTournament,
)

STRATEGY_REGISTRY: dict[str, type] = {
  "CrossoverGPUOrder": CrossoverGPUOrder,
  "CrossoverOrder": CrossoverOrder,
  "CrossoverPartiallyMapped": CrossoverPartiallyMapped,
  "EvaluationCPUCost": EvaluationCPUCost,
  "EvaluationGPUCost": EvaluationGPUCost,
  "InitializationNearestNeighbor": InitializationNearestNeighbor,
  "InitializationRandom": InitializationRandom,
  "LocalSearchCPUTwoOpt": LocalSearchCPUTwoOpt,
  "LocalSearchGPUTwoOpt": LocalSearchGPUTwoOpt,
  "LocalSearchNoOp": LocalSearchNoOp,
  "MutationGPUInversion": MutationGPUInversion,
  "MutationGPUSwap": MutationGPUSwap,
  "MutationInversion": MutationInversion,
  "MutationSwap": MutationSwap,
  "SelectionGPUTournament": SelectionGPUTournament,
  "SelectionRouletteWheel": SelectionRouletteWheel,
  "SelectionTournament": SelectionTournament,
}

__all__: list[str] = [
  "STRATEGY_REGISTRY",
  "CrossoverGPUOrder",
  "CrossoverOrder",
  "CrossoverPartiallyMapped",
  "EvaluationCPUCost",
  "EvaluationGPUCost",
  "InitializationNearestNeighbor",
  "InitializationRandom",
  "LocalSearchCPUTwoOpt",
  "LocalSearchGPUTwoOpt",
  "LocalSearchNoOp",
  "MutationGPUInversion",
  "MutationGPUSwap",
  "MutationInversion",
  "MutationSwap",
  "SelectionGPUTournament",
  "SelectionRouletteWheel",
  "SelectionTournament",
  "StrategyBase",
]
