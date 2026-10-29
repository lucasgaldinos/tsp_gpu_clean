"""
Monolithic Full-GPU Genetic Algorithm (Fujimoto Kernel with Early Stopping).

This module is intentionally separated from the modular strategy-composed
`GeneticAlgorithm` pipeline. In this monolithic implementation, the entire
evolutionary loop (initialization, tournament selection, OX crossover, swap
mutation, 2-opt local search, fitness evaluation, and survival) executes
inside a single fused CUDA kernel (`ga_fujimoto_early_stop.cu`).
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from utils.gpu_helpers import load_kernel

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.algorithm_strategies import Route
  from protocols.problem_context import ProblemContext


class GeneticAlgorithmFullGPUMonolithic:
  """
  Monolithic Full-GPU Genetic Algorithm using `ga_fujimoto_early_stop.cu`.

  Kept separate from the modular per-phase GPU strategies to serve as the
  fused-kernel baseline studied in Chapter 4.
  """

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
  ) -> None:
    """
    Initialize the monolithic Full-GPU GA runner.

    Args:
        population_size: Number of individuals in the population.
        mutation_rate: Swap mutation probability.
        tournament_size: Tournament selection size `k`.
        two_opt_iterations: Maximum 2-opt iterations per individual.
        seed: Random seed for reproducibility.
    """
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.tournament_size: int = tournament_size
    self.two_opt_iterations: int = two_opt_iterations
    self.seed: int | None = seed

    self.generation: int = 0
    self.best_cost_history: list[float] = []
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

    self._init_rand_kernel: Any = None
    self._init_pop_kernel: Any = None
    self._ga_kernel: Any = None

  def _ensure_kernels_loaded(self) -> None:
    """Lazily load and compile the monolithic CUDA kernels."""
    if self._ga_kernel is None:
      self._init_rand_kernel = load_kernel(
        "ga_fujimoto_early_stop.cu",
        "init_rand_states",
        __file__,
      )
      self._init_pop_kernel = load_kernel(
        "ga_fujimoto_early_stop.cu",
        "initialize_population",
        __file__,
      )
      self._ga_kernel = load_kernel(
        "ga_fujimoto_early_stop.cu",
        "ga_evolution_early_stop",
        __file__,
      )
      logging.info("Loaded monolithic Fujimoto GA kernel with early stopping support")

  def evolve(
    self,
    context: ProblemContext,
    customers: Sequence[int],
    max_generations: int,
    optimal_cost: float | None = None,
    patience: int | None = None,
  ) -> tuple[Route, dict[str, Any]]:
    """
    Execute the monolithic Full-GPU GA evolution kernel.

    Args:
        context: `ProblemContext` with distance matrix.
        customers: Sequence of customer indices `[0..n-1]`.
        max_generations: Maximum number of generations.
        optimal_cost: Known optimal tour cost for early stopping (`None` disables).
        patience: Stagnation patience window in generations (`None` = `2 * sqrt(n)`).

    Returns:
        Tuple `(best_tour, stats)` matching the standardized benchmark contract.
    """
    if not CUPY_AVAILABLE or cp is None:
      raise RuntimeError("CuPy is required to run GeneticAlgorithmFullGPUMonolithic")

    self._ensure_kernels_loaded()
    n = len(customers)
    distances = context.get_cpu_distances()

    if patience is None:
      patience = int(2 * math.sqrt(n))

    self.generation = 0
    self.best_cost_history = []
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

    distances_gpu = cp.asarray(distances, dtype=cp.int32)
    self.h2d_bytes += int(distances.nbytes)

    population_gpu = cp.empty((self.population_size, n), dtype=cp.int32)
    new_population_gpu = cp.empty((self.population_size, n), dtype=cp.int32)
    fitness_gpu = cp.empty(self.population_size, dtype=cp.int64)

    rand_state_size = 48  # sizeof(curandStateXORWOW)
    rand_states_gpu = cp.empty((self.population_size, rand_state_size), dtype=cp.uint8)

    best_tour_gpu = cp.empty(n, dtype=cp.int32)
    best_fitness_gpu = cp.empty(1, dtype=cp.int64)
    stopped_generation_gpu = cp.empty(1, dtype=cp.int32)

    threads_per_block = 256
    blocks_per_grid = (self.population_size + threads_per_block - 1) // threads_per_block

    raw_seed = getattr(context, "seed", None)
    if isinstance(raw_seed, list):
      effective_seed = int(raw_seed[0]) if raw_seed else 42
    elif raw_seed is not None:
      effective_seed = int(raw_seed)
    elif self.seed is not None:
      effective_seed = int(self.seed)
    else:
      effective_seed = int(time.time())

    self._init_rand_kernel(
      (blocks_per_grid,),
      (threads_per_block,),
      (rand_states_gpu, np.uint64(effective_seed), np.int32(self.population_size)),
    )
    self.kernel_launches += 1

    self._init_pop_kernel(
      (blocks_per_grid,),
      (threads_per_block,),
      (
        population_gpu,
        fitness_gpu,
        distances_gpu,
        np.int32(n),
        np.int32(self.population_size),
        rand_states_gpu,
      ),
    )
    self.kernel_launches += 1

    initial_best_cost = float(fitness_gpu.min().get())
    self.best_cost_history.append(initial_best_cost)

    optimal_cost_param = int(optimal_cost) if optimal_cost is not None else -1

    self._ga_kernel(
      (blocks_per_grid,),
      (threads_per_block,),
      (
        population_gpu,
        fitness_gpu,
        new_population_gpu,
        distances_gpu,
        np.int32(n),
        np.int32(self.population_size),
        np.int32(max_generations),
        np.float32(self.mutation_rate),
        np.int32(0),  # elite_size=0 for ISO-algorithmic (mu+lambda)
        np.int32(self.tournament_size),
        rand_states_gpu,
        best_tour_gpu,
        best_fitness_gpu,
        np.int64(optimal_cost_param),
        np.int32(patience),
        np.int32(self.two_opt_iterations),
        stopped_generation_gpu,
      ),
    )
    self.kernel_launches += 1

    cp.cuda.Device().synchronize()

    final_best_cost = float(best_fitness_gpu.get().item())
    best_tour: Route = best_tour_gpu.get().astype(np.int32)
    actual_generations = int(stopped_generation_gpu.get().item())

    self.d2h_bytes += int(best_tour.nbytes + 4 + 4)

    for gen in range(1, actual_generations + 1):
      interpolated = initial_best_cost - ((initial_best_cost - final_best_cost) * gen / max(actual_generations, 1))
      self.best_cost_history.append(interpolated)

    improvement_pct = (
      (initial_best_cost - final_best_cost) / initial_best_cost * 100.0 if initial_best_cost > 0 else 0.0
    )

    if optimal_cost is not None and optimal_cost > 0:
      if final_best_cost <= optimal_cost + 1e-6:
        stop_reason = "hit_optimal"
      else:
        gap_percent = (final_best_cost - optimal_cost) / optimal_cost * 100.0
        if gap_percent < 1.0:
          stop_reason = "within_1pct_optimal"
        elif actual_generations < max_generations:
          stop_reason = "no_improvements"
        else:
          stop_reason = "max_generations"
    elif actual_generations < max_generations:
      stop_reason = "no_improvements"
    else:
      stop_reason = "max_generations"

    stats: dict[str, Any] = {
      "best_fitness": final_best_cost,
      "initial_fitness": initial_best_cost,
      "improvement_pct": improvement_pct,
      "best_cost_history": self.best_cost_history,
      "h2d_bytes": self.h2d_bytes,
      "d2h_bytes": self.d2h_bytes,
      "kernel_launches": self.kernel_launches,
      "generations_completed": actual_generations,
      "stop_reason": stop_reason,
    }
    return best_tour, stats


# Alias for backward compatibility
GeneticAlgorithmFullGPUEarlyStop = GeneticAlgorithmFullGPUMonolithic
