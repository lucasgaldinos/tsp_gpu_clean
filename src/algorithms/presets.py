"""
Algorithm presets and convenience factory functions.

This module provides factory functions and legacy preset wrappers for standard
CPU, Hybrid, and Modular Full-GPU Genetic Algorithm pipelines. Concrete strategies
are injected into the base `GeneticAlgorithm` metaheuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from protocols.algorithm_strategies import LocalSearchStage

from .metaheuristics.genetic_algorithm import GeneticAlgorithm
from .monolithic.full_gpu_monolithic import (
  GeneticAlgorithmFullGPUEarlyStop,
  GeneticAlgorithmFullGPUMonolithic,
)
from .strategies.crossover import GPUOrderCrossover, OrderCrossover
from .strategies.evaluation import CPUCostEvaluation, GPUCostEvaluation
from .strategies.initialization import RandomPermutationInitialization
from .strategies.local_search import BatchGPUTwoOpt, CPUTwoOpt, NaiveGPUTwoOpt
from .strategies.mutation import GPUSwapMutation, SwapMutation
from .strategies.selection import GPUTournamentSelection, TournamentSelection

if TYPE_CHECKING:
  from protocols.algorithm_strategies import LocalSearchStage


def create_cpu_ga(
  population_size: int = 256,
  mutation_rate: float = 0.02,
  tournament_size: int = 5,
  two_opt_iterations: int = 10,
  seed: int | None = 42,
  local_search_stage: LocalSearchStage = "post_mutation",
  enable_phase_timing: bool = True,
) -> GeneticAlgorithm:
  """Create a pure NumPy CPU Genetic Algorithm pipeline."""
  return GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    seed=seed,
    initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
    selection=TournamentSelection(k=tournament_size, seed=seed),
    crossover=OrderCrossover(seed=seed),
    mutation=SwapMutation(seed=seed),
    local_search=CPUTwoOpt(max_iterations=two_opt_iterations),
    evaluation=CPUCostEvaluation(),
    local_search_stage=local_search_stage,
    enable_phase_timing=enable_phase_timing,
    tournament_size=tournament_size,
    two_opt_iterations=two_opt_iterations,
  )


def create_hybrid_naive_ga(
  population_size: int = 256,
  mutation_rate: float = 0.02,
  tournament_size: int = 5,
  two_opt_iterations: int = 10,
  seed: int | None = 42,
  local_search_stage: LocalSearchStage = "post_mutation",
  enable_phase_timing: bool = True,
) -> GeneticAlgorithm:
  """Create the Hybrid-Naive GA preset (per-route GPU 2-opt via `two_opt_single.cu`)."""
  return GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    seed=seed,
    initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
    selection=TournamentSelection(k=tournament_size, seed=seed),
    crossover=OrderCrossover(seed=seed),
    mutation=SwapMutation(seed=seed),
    local_search=NaiveGPUTwoOpt(max_iterations=two_opt_iterations),
    evaluation=CPUCostEvaluation(),
    local_search_stage=local_search_stage,
    enable_phase_timing=enable_phase_timing,
    tournament_size=tournament_size,
    two_opt_iterations=two_opt_iterations,
  )


def create_hybrid_optimized_ga(
  population_size: int = 256,
  mutation_rate: float = 0.02,
  tournament_size: int = 5,
  two_opt_iterations: int = 10,
  seed: int | None = 42,
  local_search_stage: LocalSearchStage = "post_mutation",
  enable_phase_timing: bool = True,
) -> GeneticAlgorithm:
  """Create the Hybrid-Optimized GA preset (batch GPU 2-opt + batch GPU cost evaluation)."""
  return GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    seed=seed,
    initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
    selection=TournamentSelection(k=tournament_size, seed=seed),
    crossover=OrderCrossover(seed=seed),
    mutation=SwapMutation(seed=seed),
    local_search=BatchGPUTwoOpt(max_iterations=two_opt_iterations),
    evaluation=GPUCostEvaluation(return_gpu=False),
    local_search_stage=local_search_stage,
    enable_phase_timing=enable_phase_timing,
    tournament_size=tournament_size,
    two_opt_iterations=two_opt_iterations,
  )


def create_full_gpu_modular_ga(
  population_size: int = 256,
  mutation_rate: float = 0.02,
  tournament_size: int = 5,
  two_opt_iterations: int = 10,
  seed: int | None = 42,
  local_search_stage: LocalSearchStage = "post_mutation",
  enable_phase_timing: bool = True,
) -> GeneticAlgorithm:
  """Create the Modular Full-GPU GA preset using decomposed per-phase CUDA kernels."""
  return GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    seed=seed,
    initialization=RandomPermutationInitialization(seed=seed, return_gpu=True),
    selection=GPUTournamentSelection(k=tournament_size, seed=seed),
    crossover=GPUOrderCrossover(seed=seed),
    mutation=GPUSwapMutation(seed=seed),
    local_search=BatchGPUTwoOpt(max_iterations=two_opt_iterations),
    evaluation=GPUCostEvaluation(return_gpu=True),
    local_search_stage=local_search_stage,
    enable_phase_timing=enable_phase_timing,
    tournament_size=tournament_size,
    two_opt_iterations=two_opt_iterations,
  )


# =============================================================================
# Deprecated Preset Subclasses (Maintained outside metaheuristics/ for compat)
# =============================================================================


class GeneticAlgorithmCPU(GeneticAlgorithm):
  """Deprecated: Use `create_cpu_ga(...)` or declarative pipeline config."""

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
    **kwargs: Any,
  ) -> None:
    super().__init__(
      population_size=population_size,
      mutation_rate=mutation_rate,
      seed=seed,
      initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
      selection=TournamentSelection(k=tournament_size, seed=seed),
      crossover=OrderCrossover(seed=seed),
      mutation=SwapMutation(seed=seed),
      local_search=CPUTwoOpt(max_iterations=two_opt_iterations),
      evaluation=CPUCostEvaluation(),
      local_search_stage=local_search_stage,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      **kwargs,
    )


class GeneticAlgorithmHybridNaive(GeneticAlgorithm):
  """Deprecated: Use `create_hybrid_naive_ga(...)` or declarative pipeline config."""

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
    **kwargs: Any,
  ) -> None:
    super().__init__(
      population_size=population_size,
      mutation_rate=mutation_rate,
      seed=seed,
      initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
      selection=TournamentSelection(k=tournament_size, seed=seed),
      crossover=OrderCrossover(seed=seed),
      mutation=SwapMutation(seed=seed),
      local_search=NaiveGPUTwoOpt(max_iterations=two_opt_iterations),
      evaluation=CPUCostEvaluation(),
      local_search_stage=local_search_stage,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      **kwargs,
    )


class GeneticAlgorithmHybridOptimized(GeneticAlgorithm):
  """Deprecated: Use `create_hybrid_optimized_ga(...)` or declarative pipeline config."""

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
    **kwargs: Any,
  ) -> None:
    super().__init__(
      population_size=population_size,
      mutation_rate=mutation_rate,
      seed=seed,
      initialization=RandomPermutationInitialization(seed=seed, return_gpu=False),
      selection=TournamentSelection(k=tournament_size, seed=seed),
      crossover=OrderCrossover(seed=seed),
      mutation=SwapMutation(seed=seed),
      local_search=BatchGPUTwoOpt(max_iterations=two_opt_iterations),
      evaluation=GPUCostEvaluation(return_gpu=False),
      local_search_stage=local_search_stage,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      **kwargs,
    )


class GeneticAlgorithmFullGPUModular(GeneticAlgorithm):
  """Deprecated: Use `create_full_gpu_modular_ga(...)` or declarative pipeline config."""

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
    **kwargs: Any,
  ) -> None:
    super().__init__(
      population_size=population_size,
      mutation_rate=mutation_rate,
      seed=seed,
      initialization=RandomPermutationInitialization(seed=seed, return_gpu=True),
      selection=GPUTournamentSelection(k=tournament_size, seed=seed),
      crossover=GPUOrderCrossover(seed=seed),
      mutation=GPUSwapMutation(seed=seed),
      local_search=BatchGPUTwoOpt(max_iterations=two_opt_iterations),
      evaluation=GPUCostEvaluation(return_gpu=True),
      local_search_stage=local_search_stage,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      **kwargs,
    )


__all__: list[str] = [
  "GeneticAlgorithmCPU",
  "GeneticAlgorithmFullGPUEarlyStop",
  "GeneticAlgorithmFullGPUModular",
  "GeneticAlgorithmFullGPUMonolithic",
  "GeneticAlgorithmHybridNaive",
  "GeneticAlgorithmHybridOptimized",
  "create_cpu_ga",
  "create_full_gpu_modular_ga",
  "create_hybrid_naive_ga",
  "create_hybrid_optimized_ga",
]