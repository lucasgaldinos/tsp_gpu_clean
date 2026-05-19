"""
ISO-Algorithmic Genetic Algorithm variants for Chapter 4 validation.

This module provides genetic algorithm implementations with different
GPU acceleration strategies for memory transfer optimization research.

Classes:
    GeneticAlgorithmBase: Abstract base enforcing ISO-algorithmic flow
    GeneticAlgorithmCPU: Pure NumPy CPU implementation
    GeneticAlgorithmHybridNaive: Individual GPU kernel launches (bottleneck demo)
    GeneticAlgorithmHybridOptimized: Batch GPU with kernel chaining
    GeneticAlgorithmFullGPU: Fujimoto's full-GPU kernel adapter
"""

from .genetic_algorithm_base import GeneticAlgorithmBase
from .genetic_algorithm_cpu import GeneticAlgorithmCPU
from .genetic_algorithm_hybrid_naive import GeneticAlgorithmHybridNaive
from .genetic_algorithm_hybrid_optimized import GeneticAlgorithmHybridOptimized
from .genetic_algorithm_full_gpu_early_stop import (
    GeneticAlgorithmFullGPUEarlyStop as GeneticAlgorithmFullGPU,
)

__all__ = [
    "GeneticAlgorithmBase",
    "GeneticAlgorithmCPU",
    "GeneticAlgorithmHybridNaive",
    "GeneticAlgorithmHybridOptimized",
    "GeneticAlgorithmFullGPU",
]
