#!/usr/bin/env python3
"""CPU-Only Genetic Algorithm (ISO-Algorithmic Variant).

Pure NumPy implementation for baseline comparison in Chapter 4 validation.
Uses identical algorithm flow to GPU variants.

Author: AI Assistant
Date: 2025-01-28
"""

import numpy as np
import logging
from typing import Any

from src.algorithms.metaheuristics.genetic_algorithm_base import (
    GeneticAlgorithmBase,
)


class GeneticAlgorithmCPU(GeneticAlgorithmBase):
    """
    CPU-only GA variant using pure NumPy operations.

    This serves as the baseline for measuring GPU speedup.
    All operations (selection, crossover, mutation, 2-opt, fitness)
    run sequentially on CPU.

    Memory Transfer Pattern:
        - H2D: 0 bytes (no GPU usage)
        - D2H: 0 bytes (no GPU usage)
        - Kernel launches: 0 (CPU only)
    """

    def _improve_population(
        self,
        offspring: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """
        Apply 2-opt improvement using CPU (NumPy).

        Sequential implementation: Process each tour independently
        with 10 iterations of 2-opt per tour.

        Args:
            offspring: Tours to improve, shape (population_size, n)
            distances: Distance matrix (NumPy array)
            xp: Backend module (np)

        Returns:
            Improved tours, shape (population_size, n)
        """
        improved = offspring.copy()
        n = offspring.shape[1]

        for i in range(self.population_size):
            tour = improved[i]

            # Apply 10 iterations of 2-opt
            for _ in range(self.two_opt_iterations):
                best_delta = 0.0
                best_i, best_j = -1, -1

                # Find best 2-opt swap
                for ii in range(n - 1):
                    for jj in range(ii + 2, n):
                        # Calculate delta for swap (i, i+1) with (j, j+1)
                        city_i = tour[ii]
                        city_i_next = tour[ii + 1]
                        city_j = tour[jj]
                        city_j_next = tour[(jj + 1) % n]

                        delta = (
                            distances[city_i, city_j]
                            + distances[city_i_next, city_j_next]
                            - distances[city_i, city_i_next]
                            - distances[city_j, city_j_next]
                        )

                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = ii, jj

                # Apply best swap if improvement found
                if best_delta < 0:
                    tour[best_i + 1 : best_j + 1] = tour[best_i + 1 : best_j + 1][::-1]
                else:
                    break  # No improvement, exit 2-opt loop

            improved[i] = tour

        # No GPU transfers
        self.h2d_bytes += 0
        self.d2h_bytes += 0
        self.kernel_launches += 0

        return improved

    def _evaluate_population(
        self,
        population: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """
        Calculate fitness (tour costs) using CPU (NumPy).

        Vectorized fitness calculation for entire population.

        Args:
            population: Tours to evaluate, shape (population_size, n)
            distances: Distance matrix (NumPy array)
            xp: Backend module (np)

        Returns:
            Fitness values (costs), shape (population_size,)
        """
        costs = np.zeros(self.population_size, dtype=np.int64)

        for i in range(self.population_size):
            tour = population[i]
            cost = 0.0

            # Sum edge costs
            for j in range(len(tour)):
                city_from = tour[j]
                city_to = tour[(j + 1) % len(tour)]
                cost += distances[city_from, city_to]

            costs[i] = cost

        # No GPU transfers
        self.h2d_bytes += 0
        self.d2h_bytes += 0
        self.kernel_launches += 0

        return costs
