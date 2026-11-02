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
  BatchGPUMultiMoveTwoOpt,
  BatchGPUTwoOpt,
  CPUFirstImprovementTwoOpt,
  CPUMultiMoveTwoOpt,
  CPUTwoOpt,
  NaiveGPUTwoOpt,
  NoOpLocalSearch,
)
from .mutation import (
  GPUInversionMutation,
  GPUSwapMutation,
  InversionMutation,
  SwapMutation,
)
from .selection import (
  GPUTournamentSelection,
  RouletteWheelSelection,
  TournamentSelection,
)

STRATEGY_REGISTRY: dict[str, type] = {
  "BatchGPUMultiMoveTwoOpt": BatchGPUMultiMoveTwoOpt,
  "BatchGPUTwoOpt": BatchGPUTwoOpt,
  "CPUCostEvaluation": CPUCostEvaluation,
  "CPUFirstImprovementTwoOpt": CPUFirstImprovementTwoOpt,
  "CPUMultiMoveTwoOpt": CPUMultiMoveTwoOpt,
  "CPUTwoOpt": CPUTwoOpt,
  "GPUCostEvaluation": GPUCostEvaluation,
  "GPUInversionMutation": GPUInversionMutation,
  "GPUOrderCrossover": GPUOrderCrossover,
  "GPUSwapMutation": GPUSwapMutation,
  "GPUTournamentSelection": GPUTournamentSelection,
  "InversionMutation": InversionMutation,
  "NaiveGPUTwoOpt": NaiveGPUTwoOpt,
  "NoOpLocalSearch": NoOpLocalSearch,
  "OrderCrossover": OrderCrossover,
  "PartiallyMappedCrossover": PartiallyMappedCrossover,
  "RandomPermutationInitialization": RandomPermutationInitialization,
  "RouletteWheelSelection": RouletteWheelSelection,
  "SwapMutation": SwapMutation,
  "TournamentSelection": TournamentSelection,
}

__all__: list[str] = [
  "STRATEGY_REGISTRY",
  "BatchGPUMultiMoveTwoOpt",
  "BatchGPUTwoOpt",
  "CPUCostEvaluation",
  "CPUFirstImprovementTwoOpt",
  "CPUMultiMoveTwoOpt",
  "CPUTwoOpt",
  "GPUCostEvaluation",
  "GPUInversionMutation",
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

