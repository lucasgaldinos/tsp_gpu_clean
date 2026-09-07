"""
Protocols for ISO-algorithmic Genetic Algorithms.

This package defines protocol interfaces for:
- Problem context (distance matrices, backend selection)
- Algorithm strategies (metaheuristics base protocol)
"""

from .problem_context import ProblemContext
from .algorithm_strategies import MetaheuristicStrategy

__all__ = ["ProblemContext", "MetaheuristicStrategy"]
