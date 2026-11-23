"""
Modular GPU-Accelerated Evolutionary Algorithms Package.

Structure:
- `metaheuristics/`: Composable `GeneticAlgorithm` metaheuristic orchestrator
- `strategies/`: Pluggable operator strategies (`initialization`, `selection`, `crossover`,
  `mutation`, `local_search`, `evaluation`) with modular CUDA `kernels/`
- `archive/`: Pre-modularization baseline kept for thesis comparison; not part of the
  public API and never imported by this package
"""

from algorithms.pipeline_steps import register_pipeline_step
from algorithms.strategies import register_strategy
from protocols.types import LocalSearchStage

from .metaheuristics import (
  GeneticAlgorithm,
  GeneticAlgorithmBase,
)

__all__: list[str] = [
  "GeneticAlgorithm",
  "GeneticAlgorithmBase",
  "LocalSearchStage",
  "register_pipeline_step",
  "register_strategy",
]
