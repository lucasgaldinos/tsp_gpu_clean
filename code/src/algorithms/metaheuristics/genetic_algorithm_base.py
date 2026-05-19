#!/usr/bin/env python3
"""
ISO-Algorithmic Genetic Algorithm Base Class for Chapter 4 Validation.

This base class ensures that ALL algorithm variants (CPU, HybridNaive,
HybridOptimized, FullGPU) use IDENTICAL solving logic, differing ONLY
in where computation happens and memory transfer patterns.

Academic Purpose:
    Enable apples-to-apples comparison for thesis Chapter 4 validation
    by eliminating algorithmic differences as a confounding variable.

Architecture:
    Template Method Pattern - defines algorithm skeleton with abstract
    methods for variant-specific operations (_improve_population,
    _evaluate_population).

Author: AI Assistant
Date: 2025-01-28
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import logging
import math

from src.data_models.problem import Problem
from src.protocols.problem_context import ProblemContext
from src.algorithms.strategies.selection_strategies import TournamentSelection
from src.algorithms.strategies.crossover_strategies import OrderCrossover
from src.algorithms.strategies.mutation_strategies import SwapMutation


class GeneticAlgorithmBase(ABC):
    """
    Abstract base class for ISO-algorithmic GA variants.

    Defines the EXACT algorithm flow:
        1. Initialize random population
        2. For each generation:
            a. Tournament selection (k=5)
            b. Order crossover (OX)
            c. Swap mutation (rate=0.02)
            d. 2-opt improvement (10 iterations per tour)
            e. Fitness evaluation
            f. (mu+lambda) survival selection
        3. Return best solution

    Subclasses MUST implement:
        - _improve_population(): Apply 2-opt to offspring
        - _evaluate_population(): Calculate fitness values

    All other operations (selection, crossover, mutation) are IDENTICAL
    across variants to ensure fair comparison.
    """

    def __init__(
        self,
        population_size: int = 256,
        mutation_rate: float = 0.02,
        tournament_size: int = 5,
        two_opt_iterations: int = 10,
        seed: int = 42,
    ):
        """
        Initialize GA with ISO-algorithmic parameters.

        Args:
            population_size: Number of individuals (mu = lambda = 256)
            mutation_rate: Probability of swap mutation (0.02)
            tournament_size: Tournament selection size (5)
            two_opt_iterations: 2-opt iterations per tour (10)
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.two_opt_iterations = two_opt_iterations
        self.seed = seed

        # Initialize operators (SAME for all variants)
        self.selection = TournamentSelection(tournament_size=tournament_size)
        self.crossover = OrderCrossover()
        self.mutation = SwapMutation()

        # Random state for reproducibility
        self.rng = np.random.RandomState(seed)

        # Statistics tracking
        self.generation = 0
        self.best_cost_history: List[float] = []
        self.h2d_bytes = 0  # Host-to-Device transfers
        self.d2h_bytes = 0  # Device-to-Host transfers
        self.kernel_launches = 0

        logging.info(
            f"{self.__class__.__name__} initialized: "
            f"pop_size={population_size}, mutation_rate={mutation_rate}, "
            f"tournament_k={tournament_size}, 2opt_iters={two_opt_iterations}"
        )

    @abstractmethod
    def _improve_population(
        self,
        offspring: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """
        Apply 2-opt improvement to offspring population.

        This is the KEY differentiation point between variants:
        - CPU: NumPy sequential 2-opt
        - HybridNaive: 256 individual GPU kernel calls
        - HybridOptimized: Single batch GPU call with kernel chaining
        - FullGPU: GPU-resident 2-opt (no transfers)

        Args:
            offspring: Tours to improve, shape (population_size, n)
            distances: Distance matrix
            xp: Backend module (np or cp)

        Returns:
            Improved tours, shape (population_size, n)

        Note:
            Subclasses MUST track h2d_bytes, d2h_bytes, kernel_launches
            to enable memory transfer analysis.
        """
        pass

    @abstractmethod
    def _evaluate_population(
        self,
        population: np.ndarray,
        distances: np.ndarray,
        xp: Any,
    ) -> np.ndarray:
        """
        Calculate fitness (tour costs) for population.

        This is the SECOND differentiation point:
        - CPU: NumPy vectorized fitness calculation
        - HybridNaive: Transfer tours to CPU, calculate on CPU
        - HybridOptimized: GPU fitness kernel (chained with 2-opt)
        - FullGPU: GPU-resident fitness (no transfers)

        Args:
            population: Tours to evaluate, shape (population_size, n)
            distances: Distance matrix
            xp: Backend module (np or cp)

        Returns:
            Fitness values (costs), shape (population_size,)

        Note:
            For HybridOptimized, this may be called WITHIN _improve_population
            via kernel chaining (no intermediate transfers).
        """
        pass

    def _initialize_population(self, n: int) -> np.ndarray:
        """
        Create initial random population (IDENTICAL for all variants).

        Args:
            n: Problem size (number of cities)

        Returns:
            Population array, shape (population_size, n)
        """
        population = np.zeros((self.population_size, n), dtype=np.int32)
        for i in range(self.population_size):
            population[i] = self.rng.permutation(n)
        return population

    def _select_parents(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """
        Tournament selection (IDENTICAL for all variants).

        Args:
            population: Current population, shape (pop_size, n)
            fitness: Fitness values, shape (pop_size,)

        Returns:
            Selected parent indices, shape (pop_size,)
        """
        parent_indices = np.zeros(self.population_size, dtype=np.int32)
        for i in range(self.population_size):
            # Tournament selection
            tournament_indices = self.rng.choice(
                self.population_size,
                size=self.tournament_size,
                replace=False,
            )
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parent_indices[i] = winner_idx
        return parent_indices

    def _create_offspring(
        self,
        population: np.ndarray,
        parent_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Apply crossover and mutation (IDENTICAL for all variants).

        Args:
            population: Current population
            parent_indices: Selected parent indices

        Returns:
            Offspring population, shape (pop_size, n)
        """
        n = population.shape[1]
        offspring = np.zeros((self.population_size, n), dtype=np.int32)

        for i in range(0, self.population_size, 2):
            # Get two parents
            parent1_idx = parent_indices[i]
            parent2_idx = parent_indices[min(i + 1, self.population_size - 1)]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Order crossover (OX) - produces 1 child per call
            child1 = self.crossover.crossover(parent1, parent2, None)
            child2 = self.crossover.crossover(
                parent2, parent1, None
            )  # Reverse parents for 2nd child

            # Swap mutation
            if self.rng.rand() < self.mutation_rate:
                child1 = self.mutation.mutate(child1, None)
            if self.rng.rand() < self.mutation_rate:
                child2 = self.mutation.mutate(child2, None)

            offspring[i] = child1
            if i + 1 < self.population_size:
                offspring[i + 1] = child2

        return offspring

    def _survival_selection(
        self,
        population: np.ndarray,
        offspring: np.ndarray,
        pop_fitness: np.ndarray,
        off_fitness: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (mu+lambda) survival selection (IDENTICAL for all variants).

        Args:
            population: Current population
            offspring: Offspring population
            pop_fitness: Current fitness
            off_fitness: Offspring fitness

        Returns:
            (new_population, new_fitness) tuple
        """
        # Combine parents and offspring
        combined = np.vstack([population, offspring])
        combined_fitness = np.concatenate([pop_fitness, off_fitness])

        # Select best mu individuals
        sorted_indices = np.argsort(combined_fitness)
        best_indices = sorted_indices[: self.population_size]

        new_population = combined[best_indices]
        new_fitness = combined_fitness[best_indices]

        return new_population, new_fitness

    def evolve(
        self,
        context: ProblemContext,
        customers: List[int],
        max_generations: int,
        optimal_cost: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run GA evolution (IDENTICAL algorithm flow for all variants).

        This is the CORE algorithm that remains CONSTANT across variants.
        Only _improve_population() and _evaluate_population() differ.

        Args:
            context: Problem context with distance matrix and backend
            customers: List of customer indices
            max_generations: Number of generations to evolve
            optimal_cost: Known optimal cost for early stopping (None to disable)
            patience: Stagnation patience in generations (None = adaptive: 2×√n)

        Returns:
            (best_tour, stats) where stats contains:
                - best_fitness: Final best cost
                - initial_fitness: Initial best cost
                - improvement_pct: Percentage improvement
                - best_cost_history: Cost per generation
                - h2d_bytes: Total H2D memory transfer
                - d2h_bytes: Total D2H memory transfer
                - kernel_launches: Total GPU kernel launches
                - generations_completed: Actual generations run
                - stop_reason: Why evolution stopped ("hit_optimal", "no_improvements", "max_generations")
        """
        n = len(customers)
        xp = context.xp
        distances = context.get_cpu_distances()
        
        # Use adaptive patience if not provided: 2 × sqrt(n)
        if patience is None:
            patience = int(2 * math.sqrt(n))

        # Reset statistics
        self.generation = 0
        self.best_cost_history = []
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.kernel_launches = 0

        # Step 1: Initialize population (SAME for all)
        population = self._initialize_population(n)

        # Step 2: Initial fitness evaluation
        fitness = self._evaluate_population(population, distances, xp)
        initial_best_cost = float(np.min(fitness))
        self.best_cost_history.append(initial_best_cost)

        # Step 3: Evolution loop
        last_improvement_gen = 0
        best_ever_cost = float("inf")
        stop_reason = "max_generations"  # Default if loop completes

        for gen in range(max_generations):
            self.generation = gen + 1

            # 3a. Selection (SAME for all)
            parent_indices = self._select_parents(population, fitness)

            # 3b. Crossover + Mutation (SAME for all)
            offspring = self._create_offspring(population, parent_indices)

            # 3c. Improvement (DIFFERS: CPU vs GPU-individual vs GPU-batch)
            offspring_improved = self._improve_population(offspring, distances, xp)

            # 3d. Fitness evaluation (DIFFERS: CPU vs GPU)
            offspring_fitness = self._evaluate_population(
                offspring_improved, distances, xp
            )

            # 3e. Survival selection (SAME for all)
            population, fitness = self._survival_selection(
                population, offspring_improved, fitness, offspring_fitness
            )

            # Track best cost
            best_cost = float(np.min(fitness))
            self.best_cost_history.append(best_cost)

            # Track improvement for early stopping
            if best_cost < best_ever_cost:
                best_ever_cost = best_cost
                last_improvement_gen = gen

            # Early stopping condition 1: Optimal reached (within 1% of optimal)
            if optimal_cost is not None:
                gap_percent = (best_cost - optimal_cost) / optimal_cost * 100
                if gap_percent < 1.0:  # Within 1% of optimal
                    logging.info(
                        f"Near-optimal solution reached at generation {gen + 1} "
                        f"(cost={best_cost:.2f}, optimal={optimal_cost:.2f}, gap={gap_percent:.2f}%)"
                    )
                    stop_reason = "hit_optimal"
                    break

            # Early stopping condition 2: Stagnation (patience threshold reached)
            if patience is not None and gen - last_improvement_gen >= patience:
                logging.info(
                    f"No improvement for {patience} generations. "
                    f"Stopping at generation {gen + 1}"
                )
                stop_reason = "no_improvements"
                break

            if (gen + 1) % 100 == 0:
                logging.info(
                    f"Generation {gen + 1}/{max_generations}: "
                    f"best={best_cost:.2f}, avg={np.mean(fitness):.2f}"
                )

        # Final statistics
        final_best_cost = float(np.min(fitness))
        best_idx = np.argmin(fitness)
        best_tour = population[best_idx]

        improvement_pct = (
            (initial_best_cost - final_best_cost) / initial_best_cost * 100
        )

        stats = {
            "best_fitness": final_best_cost,
            "initial_fitness": initial_best_cost,
            "improvement_pct": improvement_pct,
            "best_cost_history": self.best_cost_history,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "kernel_launches": self.kernel_launches,
            "generations_completed": self.generation,  # Actual completed generations
            "stop_reason": stop_reason,  # Why evolution stopped
        }

        logging.info(
            f"{self.__class__.__name__} evolution complete: "
            f"best={final_best_cost:.2f}, improvement={improvement_pct:.2f}%, "
            f"H2D={self.h2d_bytes / 1e6:.2f}MB, D2H={self.d2h_bytes / 1e6:.2f}MB, "
            f"kernels={self.kernel_launches}, stopped={stop_reason}"
        )

        return best_tour, stats
