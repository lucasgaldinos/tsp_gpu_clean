"""
Modular GPU-Accelerated Evolutionary Algorithms Package.

Structure:
- `metaheuristics/`: Composable `GeneticAlgorithm` metaheuristic orchestrator
- `presets`: Preconfigured pipelines and legacy preset classes
- `strategies/`: Pluggable `initialization`, `selection`, `crossover`, `mutation`,
  `local_search`, and `evaluation` strategies (with modular CUDA `kernels/`)
- `monolithic/`: Separated Fujimoto monolithic Full-GPU kernel (`ga_fujimoto_early_stop.cu`)
- `archive/`: Legacy pre-modularization implementations preserved as thesis study objects
"""

from .metaheuristics import (
  GeneticAlgorithm,
  GeneticAlgorithmBase,
  LocalSearchStage,
)
from .monolithic.full_gpu_monolithic import (
  GeneticAlgorithmFullGPUEarlyStop,
  GeneticAlgorithmFullGPUMonolithic,
)
from .presets import (
  GeneticAlgorithmCPU,
  GeneticAlgorithmFullGPUModular,
  GeneticAlgorithmHybridNaive,
  GeneticAlgorithmHybridOptimized,
  create_cpu_ga,
  create_full_gpu_modular_ga,
  create_hybrid_naive_ga,
  create_hybrid_optimized_ga,
)

__all__: list[str] = [
  "GeneticAlgorithm",
  "GeneticAlgorithmBase",
  "GeneticAlgorithmCPU",
  "GeneticAlgorithmFullGPUEarlyStop",
  "GeneticAlgorithmFullGPUModular",
  "GeneticAlgorithmFullGPUMonolithic",
  "GeneticAlgorithmHybridNaive",
  "GeneticAlgorithmHybridOptimized",
  "LocalSearchStage",
  "create_cpu_ga",
  "create_full_gpu_modular_ga",
  "create_hybrid_naive_ga",
  "create_hybrid_optimized_ga",
]
