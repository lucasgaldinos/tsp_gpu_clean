"""
Modular evolutionary operator strategies for TSP and routing optimization.

All route-transforming operators (`CrossoverStrategy`, `MutationStrategy`,
`LocalSearchStrategy`, `EvaluationStrategy`) accept `RouteOrRoutes`
(`(n_cities,)` or `(N, n_cities)`), enabling arbitrary pipeline ordering.
"""

from .crossover import (
  GPUOrderCrossover,
  OrderCrossover,
  PartiallyMappedCrossover,
)
from .evaluation import (
  CPUCostEvaluation,
  GPUCostEvaluation,
)
from .initialization import RandomPermutationInitialization
from .local_search import (
  BatchGPUTwoOpt,
  CPUTwoOpt,
  NaiveGPUTwoOpt,
  NoOpLocalSearch,
)
from .mutation import (
  GPUSwapMutation,
  InversionMutation,
  SwapMutation,
)
from .selection import (
  GPUTournamentSelection,
  RouletteWheelSelection,
  TournamentSelection,
)

__all__ = [
  "BatchGPUTwoOpt",
  "CPUCostEvaluation",
  "CPUTwoOpt",
  "GPUCostEvaluation",
  "GPUOrderCrossover",
  "GPUSwapMutation",
  "GPUTournamentSelection",
  "InversionMutation",
  "NaiveGPUTwoOpt",
  "NoOpLocalSearch",
  "OrderCrossover",
  "PartiallyMappedCrossover",
  "RandomPermutationInitialization",
  "RouletteWheelSelection",
  "SwapMutation",
  "TournamentSelection",
]
