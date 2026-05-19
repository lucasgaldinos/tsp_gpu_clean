"""
Strategy implementations for ISO-algorithmic Genetic Algorithms.

This package provides genetic algorithm operator strategies:
- Selection: Tournament selection
- Crossover: Order crossover (OX)
- Mutation: Swap mutation

These strategies are used by GeneticAlgorithmBase and its variants.
"""

from .selection_strategies import TournamentSelection
from .crossover_strategies import OrderCrossover
from .mutation_strategies import SwapMutation

__all__ = [
    "TournamentSelection",
    "OrderCrossover",
    "SwapMutation",
]
