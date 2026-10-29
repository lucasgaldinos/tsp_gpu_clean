#!/usr/bin/env python3
"""Naive Hybrid Genetic Algorithm (ISO-Algorithmic Variant).

Individual GPU kernel launches with full D2H transfers.
This variant demonstrates the memory transfer bottleneck.

Architecture:
    - Selection, Crossover, Mutation: CPU (NumPy)
    - 2-opt improvement: GPU (256 individual kernel calls)
    - Fitness calculation: CPU (after D2H transfer)

Memory Transfer Pattern (per generation):
    - H2D: 256 tours × n cities × 4 bytes × 10 iterations = 10,240n bytes
    - D2H: 256 tours × n cities × 4 bytes × 10 iterations = 10,240n bytes
    - Total: 20,480n bytes per generation
    - Example (n=1000): ~20MB per generation

Author: AI Assistant
Date: 2025-01-28
"""

import numpy as np
import cupy as cp
import logging
from typing import Any
from pathlib import Path

from algorithms.metaheuristics.genetic_algorithm_base import (
    GeneticAlgorithmBase,
)


class GeneticAlgorithmHybridNaive(GeneticAlgorithmBase):
    """Naive hybrid GA with individual GPU kernel launches.

    This variant demonstrates the inefficiency of per-tour GPU processing.
    Each tour is transferred individually: H2D → GPU 2-opt → D2H.

    For 256 tours with 10 iterations, this results in 2,560 kernel launches
    and 2,560 memory transfers per generation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with kernel compilation."""
        super().__init__(*args, **kwargs)
        self._compile_kernel()

    def _compile_kernel(self):
        """Load single-tour 2-opt kernel."""
        kernel_dir = Path(__file__).parent.parent / "kernels"
        kernel_path = kernel_dir / "two_opt_single.cu"

        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel not found: {kernel_path}")

        with open(kernel_path, "r") as f:
            kernel_source = f.read()

        self.two_opt_kernel = cp.RawKernel(kernel_source, "two_opt_kernel")
        logging.info("Loaded two_opt_kernel for naive hybrid")

    def _improve_population(
        self,
        offspring: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """Apply 2-opt using INDIVIDUAL GPU kernel calls.

        For each tour:
            1. H2D transfer tour
            2. Launch GPU 2-opt kernel (10 iterations)
            3. D2H transfer improved tour

        This pattern maximizes memory transfer overhead.

        Args:
            offspring: Tours to improve, shape (population_size, n)
            distances: Distance matrix (NumPy array)
            xp: Backend module (cp for GPU)

        Returns:
            Improved tours, shape (population_size, n)
        """
        improved = np.zeros_like(offspring)
        n = offspring.shape[1]

        # Transfer distance matrix once (shared across all tours)
        distances_gpu = cp.asarray(distances, dtype=cp.int32)
        self.h2d_bytes += distances.nbytes

        threads_per_block = min(256, n - 2)
        shared_mem = (
            threads_per_block * 4  # s_deltas (int32)
            + threads_per_block * 4  # s_swap_i (int)
            + threads_per_block * 4  # s_swap_j (int)
            + n * 4  # s_tour (int)
        )

        # Process each tour individually
        for i in range(self.population_size):
            tour_np = offspring[i]

            # H2D: Transfer single tour
            tour_gpu = cp.asarray(tour_np, dtype=cp.int32)
            self.h2d_bytes += tour_np.nbytes

            # GPU 2-opt with multiple iterations
            for iteration in range(self.two_opt_iterations):
                self.two_opt_kernel(
                    (1,),  # Single tour
                    (threads_per_block,),
                    (tour_gpu, distances_gpu, n, 0),  # tour_idx=0 (single tour)
                    shared_mem=shared_mem,
                )
                self.kernel_launches += 1

            # D2H: Transfer improved tour
            improved[i] = tour_gpu.get()
            self.d2h_bytes += tour_np.nbytes

        return improved

    def _evaluate_population(
        self,
        population: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """Calculate fitness on CPU after D2H transfer.

        Since tours are already on CPU (from _improve_population D2H),
        we calculate fitness using NumPy.

        Args:
            population: Tours to evaluate, shape (population_size, n)
            distances: Distance matrix (NumPy array)
            xp: Backend module

        Returns:
            Fitness values (costs), shape (population_size,)
        """
        costs = np.zeros(self.population_size, dtype=np.int64)

        for i in range(self.population_size):
            tour = population[i]
            cost = 0.0

            for j in range(len(tour)):
                city_from = tour[j]
                city_to = tour[(j + 1) % len(tour)]
                cost += distances[city_from, city_to]

            costs[i] = cost

        # No additional GPU transfers (tours already on CPU)
        return costs
