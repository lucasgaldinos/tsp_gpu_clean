"""
Mutation strategies for Genetic Algorithms.

Implements MutationOperator protocol with various mutation operators:
- SwapMutation: Swap two random cities
- InversionMutation: Reverse random tour segment
- InsertionMutation: Remove city and reinsert elsewhere

All implementations are:
- Backend-agnostic (work with numpy or cupy)
- Type-safe (conform to MutationOperator protocol)
- Preserve tour validity (no duplicate cities, maintain depot)

References:
- Goldberg (1989): Genetic Algorithms in Search, Optimization, and Machine Learning
- Eiben & Smith (2015): Introduction to Evolutionary Computing
"""

from typing import Any
import numpy as np


class SwapMutation:
    """
    Swap two random cities in tour.

    Algorithm:
        1. Select two random positions (excluding depot)
        2. Swap cities at those positions

    Time Complexity: O(1) - constant time operation
    Space Complexity: O(n) - copy of tour

    Properties:
        - Simplest mutation operator
        - Small neighborhood (minimal disruption)
        - Fast (constant time selection and swap)
        - Most common mutation for TSP

    Use Cases:
        - Default mutation for most GA implementations
        - Good balance between exploration and exploitation
        - Works well with moderate mutation rates (0.1-0.2)

    Example:
        >>> strategy = SwapMutation()
        >>> tour = [0, 1, 2, 3, 4, 5, 0]
        >>> mutated = strategy.mutate(tour, problem=None, xp=np)
        >>> # mutated might be: [0, 1, 4, 3, 2, 5, 0]
        >>> # (cities at positions 2 and 4 swapped)
    """

    def mutate(
        self,
        tour: Any,
        problem: Any,
    ) -> Any:
        """
        Mutate tour by swapping two random cities.

        Phase 3.5 Hybrid Bridge Architecture:
            - S-Task (Sequential): Mutation operations are CPU-only
            - Uses NumPy for random number generation
            - Permutation operations are too small to benefit from GPU parallelism

        Args:
            tour: Tour [depot, c1, ..., ck, depot]
                - List or array of city indices
                - First and last elements are depot (typically 0)
            problem: TSPProblem instance (not used, required by protocol)

        Returns:
            Mutated tour [depot, c1, ..., ck, depot]
                - Same structure as input
                - Two cities swapped (excluding depot)

        Edge Cases:
            - Single customer (n=1): Returns copy (cannot swap)
            - Less than 2 customers: Returns copy
        """
        # Create copy to avoid modifying original
        if hasattr(tour, "copy"):
            mutated = tour.copy()
        else:
            mutated = list(tour)

        n = len(tour) - 1  # Exclude last depot
        interior_size = n - 1  # Exclude first depot too

        # Edge case: less than 2 customers to swap
        if interior_size < 2:
            return mutated

        # Select two random positions (not depot) - CPU-only
        indices = np.random.choice(range(1, n), size=2, replace=False)
        i, j = indices

        # Swap cities
        mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated


class InversionMutation:
    """
    Reverse random segment of tour (2-opt style mutation).

    Algorithm:
        1. Select two random positions [i, j]
        2. Reverse tour segment tour[i:j+1]

    Time Complexity: O(k) where k = j - i + 1 (segment length)
    Space Complexity: O(n) - copy of tour

    Properties:
        - Medium disruption (more than swap, less than scramble)
        - Related to 2-opt local search (single 2-opt move)
        - Good for escaping local optima
        - Preserves some tour structure

    Use Cases:
        - Alternative to swap mutation (more disruptive)
        - Hybrid with local search (mutation + 2-opt)
        - Problems where edge reversals are meaningful

    Reference:
        Related to 2-opt heuristic (Croes 1958)

    Example:
        >>> strategy = InversionMutation()
        >>> tour = [0, 1, 2, 3, 4, 5, 0]
        >>> mutated = strategy.mutate(tour, problem=None, xp=np)
        >>> # mutated might be: [0, 1, 4, 3, 2, 5, 0]
        >>> # (segment [2,3,4] reversed to [4,3,2])
    """

    def mutate(
        self,
        tour: Any,
        problem: Any,
    ) -> Any:
        """
        Mutate tour by reversing random segment.

        Phase 3.5 Hybrid Bridge Architecture:
            - S-Task (Sequential): Mutation operations are CPU-only
            - Uses NumPy for random number generation
            - Permutation operations are too small to benefit from GPU parallelism

        Args:
            tour: Tour [depot, c1, ..., ck, depot]
            problem: TSPProblem instance (not used, required by protocol)

        Returns:
            Mutated tour with reversed segment

        Edge Cases:
            - Single customer: Returns copy
            - Less than 2 customers: Returns copy
        """
        # Create copy
        if hasattr(tour, "copy"):
            mutated = tour.copy()
        else:
            mutated = list(tour)

        n = len(tour) - 1  # Exclude last depot
        interior_size = n - 1  # Exclude first depot

        # Edge case: less than 2 customers to invert
        if interior_size < 2:
            return mutated

        # Select two random positions (not depot) - CPU-only
        indices = np.random.choice(range(1, n), size=2, replace=False)
        i, j = sorted(indices)

        # Reverse segment [i, j] (inclusive)
        mutated[i : j + 1] = mutated[i : j + 1][::-1]

        return mutated


class InsertionMutation:
    """
    Remove city from tour and reinsert at different position.

    Algorithm:
        1. Select random position to remove city
        2. Select different random position to insert
        3. Remove city from original position
        4. Insert city at new position

    Time Complexity: O(n) - worst case for list insert/remove
    Space Complexity: O(n) - copy of tour

    Properties:
        - Medium disruption (between swap and inversion)
        - Preserves tour structure around insertion point
        - Good for fine-tuning solutions
        - Common in construction heuristics

    Use Cases:
        - Alternative mutation operator
        - Refinement phase of GA
        - Hybrid with construction heuristics

    Example:
        >>> strategy = InsertionMutation()
        >>> tour = [0, 1, 2, 3, 4, 5, 0]
        >>> mutated = strategy.mutate(tour, problem=None, xp=np)
        >>> # mutated might be: [0, 1, 3, 4, 2, 5, 0]
        >>> # (city 2 removed from position 2, inserted at position 4)
    """

    def mutate(
        self,
        tour: Any,
        problem: Any,
    ) -> Any:
        """
        Mutate tour by removing and reinserting city.

        Phase 3.5 Hybrid Bridge Architecture:
            - S-Task (Sequential): Mutation operations are CPU-only
            - Uses NumPy for random number generation
            - Permutation operations are too small to benefit from GPU parallelism

        Args:
            tour: Tour [depot, c1, ..., ck, depot]
            problem: TSPProblem instance (not used, required by protocol)

        Returns:
            Mutated tour with city relocated

        Edge Cases:
            - Single customer: Returns copy (cannot relocate)
            - Less than 2 customers: Returns copy
        """
        # Create copy
        if hasattr(tour, "copy"):
            mutated = tour.copy()
        else:
            mutated = list(tour)

        n = len(tour) - 1  # Exclude last depot
        interior_size = n - 1  # Exclude first depot

        # Edge case: less than 2 customers for meaningful insertion
        if interior_size < 2:
            return mutated

        # Select position to remove city from (CPU-only random generation)
        remove_pos = int(np.random.randint(1, n))

        # Select position to insert city at (must be different)
        insert_pos = int(np.random.randint(1, n))
        while insert_pos == remove_pos:
            insert_pos = int(np.random.randint(1, n))

        # Remove city
        city = mutated.pop(remove_pos)

        # Insert at new position
        mutated.insert(insert_pos, city)

        return mutated
