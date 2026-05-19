#!/usr/bin/env python3
"""Full GPU Genetic Algorithm with Early Stopping (ISO-Algorithmic Variant).

Extended version of FullGPU with early stopping capability:
- Optimal cost detection (stops when optimal reached)
- Stagnation detection (stops when no improvement for N generations)

This makes FullGPU truly ISO-algorithmic with Hybrid variants.

Architecture:
    - Same as FullGPU but with convergence tracking in kernel
    - Still maintains minimal H2D/D2H transfers
    - Preserves Fujimoto's monolithic design advantages

Memory Transfer Pattern (entire run):
    - H2D: Distance matrix only (~n² × 8 bytes, once at start)
    - D2H: Best tour + stopped_generation (~n × 4 bytes, once at end)
    - Total: Minimal transfers, now with intelligent early stopping

Author: AI Assistant
Date: 2025-11-19
"""

import numpy as np
import cupy as cp
import logging
import time
from typing import Any, Optional

from src.algorithms.metaheuristics.genetic_algorithm_base import (
    GeneticAlgorithmBase,
)
from src.utils.gpu_helpers import load_kernel


class GeneticAlgorithmFullGPUEarlyStop(GeneticAlgorithmBase):
    """
    Full GPU GA with early stopping - Fujimoto kernel with convergence detection.

    Extends the original FullGPU implementation with:
    - Optimal cost detection
    - Stagnation detection (patience-based)
    - Actual generation tracking

    ISO-algorithmic with Hybrid variants: same stopping logic, different execution.
    """

    def __init__(
        self,
        population_size: int = 256,
        mutation_rate: float = 0.02,
        tournament_size: int = 5,
        two_opt_iterations: int = 10,
        seed: int = 42,
    ):
        """Initialize FullGPU GA with early stopping."""
        super().__init__(
            population_size=population_size,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            seed=seed,
        )
        self.two_opt_iterations = two_opt_iterations

        # Load kernels with early stopping (same file, different function name)
        self.init_rand_kernel = load_kernel(
            "ga_fujimoto_early_stop.cu", "init_rand_states", __file__
        )
        self.init_pop_kernel = load_kernel(
            "ga_fujimoto_early_stop.cu", "initialize_population", __file__
        )
        self.ga_kernel = load_kernel(
            "ga_fujimoto_early_stop.cu", "ga_evolution_early_stop", __file__
        )

        # Base class already logs initialization, just log kernel loading
        logging.info("Loaded Fujimoto GA kernel with early stopping support")

    def evolve(
        self,
        context,
        customers,
        max_generations: int,
        optimal_cost: Optional[float] = None,
        patience: Optional[int] = None,
    ):
        """Override evolve() to use Fujimoto's kernel with early stopping.

        Args:
            context: Problem context with distance matrix
            customers: List of customer indices
            max_generations: Maximum number of generations
            optimal_cost: Optimal cost for early stopping (None to disable)
            patience: Stagnation patience in generations (None = adaptive: 2×√n)

        Returns:
            (best_tour, stats) tuple
        """
        n = len(customers)
        distances = context.get_cpu_distances()
        
        # Use adaptive patience if not provided: 2 × sqrt(n)
        if patience is None:
            import math
            patience = int(2 * math.sqrt(n))

        # Reset statistics
        self.generation = 0
        self.best_cost_history = []
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.kernel_launches = 0

        # Transfer distance matrix to GPU (ONCE)
        distances_gpu = cp.asarray(distances, dtype=cp.int32)
        self.h2d_bytes += distances.nbytes

        # Allocate GPU memory
        population_gpu = cp.empty((self.population_size, n), dtype=cp.int32)
        new_population_gpu = cp.empty((self.population_size, n), dtype=cp.int32)
        fitness_gpu = cp.empty(self.population_size, dtype=cp.int64)

        # Random states (cuRAND)
        rand_state_size = 48  # sizeof(curandStateXORWOW)
        rand_states_gpu = cp.empty(
            (self.population_size, rand_state_size), dtype=cp.uint8
        )

        best_tour_gpu = cp.empty(n, dtype=cp.int32)
        best_fitness_gpu = cp.empty(1, dtype=cp.int64)
        stopped_generation_gpu = cp.empty(1, dtype=cp.int32)  # NEW: Track actual gens

        # Configure kernel launch
        threads_per_block = 256
        blocks_per_grid = (
            self.population_size + threads_per_block - 1
        ) // threads_per_block

        # Initialize random states
        self.init_rand_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (rand_states_gpu, int(time.time()), self.population_size),
        )
        self.kernel_launches += 1

        # Initialize population on GPU
        self.init_pop_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                population_gpu,
                fitness_gpu,
                distances_gpu,
                n,
                self.population_size,
                rand_states_gpu,
            ),
        )
        self.kernel_launches += 1

        # Get initial best cost
        initial_best_cost = float(fitness_gpu.min().get())
        self.best_cost_history.append(initial_best_cost)

        # Prepare early stopping parameters
        optimal_cost_param = optimal_cost if optimal_cost is not None else -1

        logging.info(
            f"FullGPUEarlyStop: Launching kernel (max={max_generations} gens, "
            f"optimal={optimal_cost if optimal_cost else 'disabled'}, patience={patience})"
        )

        # Launch main GA evolution kernel WITH EARLY STOPPING
        self.ga_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                population_gpu,
                fitness_gpu,
                new_population_gpu,
                distances_gpu,
                n,
                self.population_size,
                max_generations,
                self.mutation_rate,
                0,  # elite_size=0 for ISO-algorithmic (base uses mu+lambda)
                self.tournament_size,
                rand_states_gpu,
                best_tour_gpu,
                best_fitness_gpu,
                int(optimal_cost_param),  # int64 optimal cost
                patience,  # Patience
                self.two_opt_iterations,  # NEW: two_opt_iterations (was hardcoded in kernel)
                stopped_generation_gpu,  # Output
            ),
        )
        self.kernel_launches += 1

        cp.cuda.Device().synchronize()

        # D2H: Transfer results (ONCE)
        final_best_cost = float(best_fitness_gpu.get().item())
        best_tour = best_tour_gpu.get()
        actual_generations = int(stopped_generation_gpu.get().item())  # NEW

        self.d2h_bytes += best_tour.nbytes + 4 + 4  # tour + fitness + gen_count

        # Build convergence history (approximation based on actual generations)
        for gen in range(1, actual_generations + 1):
            interpolated = initial_best_cost - (
                (initial_best_cost - final_best_cost) * gen / actual_generations
            )
            self.best_cost_history.append(interpolated)

        improvement_pct = (
            (initial_best_cost - final_best_cost) / initial_best_cost * 100
        )

        # Determine stopping reason (standardized format - matches base algorithm)
        if optimal_cost is not None:
            gap_percent = (final_best_cost - optimal_cost) / optimal_cost * 100
            if gap_percent < 1.0:  # Within 1% of optimal
                stop_reason = "hit_optimal"
            elif actual_generations < max_generations:
                stop_reason = "no_improvements"  # Stagnation
            else:
                stop_reason = "max_generations"  # Ran full duration without hitting optimal
        elif actual_generations < max_generations:
            stop_reason = "no_improvements"  # Stagnation (no optimal cost provided)
        else:
            stop_reason = "max_generations"  # Completed all generations

        logging.info(
            f"FullGPUEarlyStop complete: best={final_best_cost:.2f}, "
            f"improvement={improvement_pct:.2f}%, gens={actual_generations}/{max_generations}, "
            f"stopped={stop_reason}, H2D={self.h2d_bytes / 1024 / 1024:.2f}MB, "
            f"D2H={self.d2h_bytes / 1024 / 1024:.2f}MB, kernels={self.kernel_launches}"
        )

        stats = {
            "best_fitness": final_best_cost,
            "initial_fitness": initial_best_cost,
            "improvement_pct": improvement_pct,
            "best_cost_history": self.best_cost_history,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "kernel_launches": self.kernel_launches,
            "generations_completed": actual_generations,  # FIXED: Use standard field name
            "stop_reason": stop_reason,  # Standardized format
        }

        return best_tour, stats

    # Abstract methods from base (not used in FullGPU, but required)
    def _improve_population(self, population, distances, xp):
        """Not used - improvement happens inside kernel."""
        return population

    def _evaluate_population(self, population, distances, xp):
        """Not used - evaluation happens inside kernel."""
        return cp.zeros(len(population))
