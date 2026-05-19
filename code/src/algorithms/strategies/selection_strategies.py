"""
Selection strategies for Genetic Algorithms.

Implements SelectionStrategy protocol with various selection methods:
- TournamentSelection: k-tournament (standard, strong pressure)
- RouletteWheelSelection: Fitness-proportional (weak pressure)
- CrowdingSelection: Deterministic (maintains diversity)

All implementations are:
- Vectorized (GPU-friendly, no Python loops)
- Backend-agnostic (work with numpy or cupy)
- Memory-efficient (return indices, not tours)
- Type-safe (conform to SelectionStrategy protocol)

References:
- Goldberg (1989): Genetic Algorithms in Search, Optimization, and Machine Learning
- Miller & Goldberg (1995): Genetic Algorithms, Tournament Selection, and Niching
- De Jong (1975): An Analysis of the Behavior of a Class of Genetic Adaptive Systems
- Mahfoud (1995): Niching Methods for Genetic Algorithms
- Fujimoto & Tsutsui (2011): A highly parallel TSP solver for GPUs
"""

from typing import Any
import numpy as np


class TournamentSelection:
    """
    k-tournament selection: sample k random individuals, select best.

    Algorithm:
        1. For each of n_select selections:
           a. Sample k candidates uniformly at random (with replacement)
           b. Select individual with lowest cost (best fitness)
        2. Return indices of selected individuals

    Time Complexity: O(n_select × k)
    Space Complexity: O(n_select × k) temporary arrays

    Selection Pressure:
        - k=2: Weak pressure, good exploration
        - k=3: Moderate pressure (typical default)
        - k=5-7: Strong pressure, fast convergence
        - As k → pop_size: approaches greedy selection (always best)

    Properties:
        - Parallel-friendly: Each tournament is independent
        - Robust: Works well with negative fitness or outliers
        - Tunable: Selection pressure controlled by k
        - Efficient: No sorting required (unlike rank-based)

    Reference:
        Miller & Goldberg (1995): "Genetic Algorithms, Tournament Selection,
        and the Effects of Noise"

    Example:
        >>> strategy = TournamentSelection(tournament_size=3)
        >>> population = np.random.rand(100, 51)  # 100 tours, 50 cities + depot
        >>> fitness = np.random.rand(100) * 1000  # Random costs
        >>> indices = strategy.select(population, fitness, n_select=100, xp=np)
        >>> # indices: best individuals from 100 tournaments of size 3
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of candidates per tournament (k)
                - Must be >= 2
                - Typical values: 2 (weak), 3 (moderate), 5-7 (strong)
                - Larger k increases selection pressure

        Raises:
            ValueError: If tournament_size < 2
        """
        if tournament_size < 2:
            raise ValueError(
                f"Tournament size must be >= 2, got {tournament_size}. "
                "Use tournament_size=2 for weak selection pressure, "
                "tournament_size=3 for moderate (typical), or "
                "tournament_size=5-7 for strong pressure."
            )
        self.tournament_size = tournament_size

    def select(
        self,
        population: Any,
        fitness: Any,
        n_select: int,
    ) -> Any:
        """
        Select n_select individuals via k-tournament.

        Phase 3.5: CPU-only execution using NumPy internally.

        Args:
            population: (pop_size, n+1) array of tours
            fitness: (pop_size,) array of costs (lower is better)
            n_select: Number of individuals to select

        Returns:
            (n_select,) array of indices into population

        Algorithm Details:
            1. Sample candidates: (n_select, k) random indices
            2. Get fitness: (n_select, k) costs of candidates
            3. Find best: (n_select,) argmin along tournaments (axis=1)
            4. Map to population indices using advanced indexing
        """
        pop_size = len(fitness)
        k = self.tournament_size

        # Sample candidates for all tournaments: (n_select, k)
        # Each row is one tournament with k candidates
        candidates = np.random.randint(0, pop_size, size=(n_select, k))

        # Get fitness of all candidates: (n_select, k)
        candidate_fitness = fitness[candidates]

        # Best in each tournament: (n_select,) - indices within each tournament [0, k)
        best_in_tournament = np.argmin(candidate_fitness, axis=1)

        # Map back to population indices: (n_select,)
        # Advanced indexing: for each i, select candidates[i, best_in_tournament[i]]
        selected_indices = candidates[np.arange(n_select), best_in_tournament]

        return selected_indices


class RouletteWheelSelection:
    """
    Fitness-proportional selection (roulette wheel).

    Algorithm:
        1. Convert costs to probabilities: p_i ∝ 1/cost_i (invert for minimization)
        2. Normalize: p_i = p_i / sum(p)
        3. Compute cumulative distribution: CDF[i] = sum(p[0:i+1])
        4. For each selection:
           a. Sample uniform random: u ~ U(0,1)
           b. Find i where CDF[i-1] < u <= CDF[i] (binary search)

    Time Complexity: O(pop_size + n_select × log(pop_size))
        - O(pop_size): compute CDF
        - O(n_select × log(pop_size)): binary search for each selection

    Space Complexity: O(pop_size) for CDF

    Selection Pressure:
        - Weak: Proportional to fitness ratio (not fitness difference)
        - Problem: Premature convergence if one individual dominates
        - Recommendation: Use tournament selection instead for most problems

    Properties:
        - Classic GA selection (Goldberg 1989)
        - Intuitive: Better fitness → higher selection probability
        - Weakness: Sensitive to fitness scaling
        - Requires: Positive fitness (handled by inversion + epsilon)

    Reference:
        Goldberg (1989): "Genetic Algorithms in Search, Optimization,
        and Machine Learning", Chapter 3.

    Example:
        >>> strategy = RouletteWheelSelection()
        >>> population = np.random.rand(100, 51)
        >>> fitness = np.random.rand(100) * 1000 + 100  # Positive costs
        >>> indices = strategy.select(population, fitness, n_select=100, xp=np)
        >>> # Lower fitness → higher selection probability
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize roulette wheel selection.

        Args:
            epsilon: Small value to prevent division by zero
                - Added to all fitness values before inversion
                - Ensures numerical stability for zero or very small costs
                - Default: 1e-6 (suitable for typical TSP costs)
        """
        self.epsilon = epsilon

    def select(
        self,
        population: Any,
        fitness: Any,
        n_select: int,
    ) -> Any:
        """
        Select n_select individuals via fitness-proportional sampling.

        Args:
            population: (pop_size,n+1) array of tours
            fitness: (pop_size,) array of costs (lower is better)
                - Must be positive (TSP costs are always positive)
                - Epsilon is added to prevent division by zero
            n_select: Number of individuals to select

        Returns:
            (n_select,) array of indices into population

        Algorithm Details:
            1. Invert fitness: inv_fitness = 1 / (fitness + epsilon)
            2. Normalize: probabilities = inv_fitness / sum(inv_fitness)
            3. Cumulative sum: cdf = cumsum(probabilities)
            4. Sample: random_values ~ U(0,1) for each selection
            5. Binary search: searchsorted(cdf, random_values)

        Notes:
            - Vectorized: Single searchsorted call for all selections
            - Numerically stable: epsilon prevents division by zero
            - Edge case handling: clip ensures valid indices [0, pop_size)
        """
        # Convert costs to probabilities (lower cost = higher probability)
        # Invert: p_i ∝ 1/cost_i
        inv_fitness = 1.0 / (fitness + self.epsilon)

        # Normalize to sum to 1
        probabilities = inv_fitness / np.sum(inv_fitness)

        # Cumulative distribution function
        cdf = np.cumsum(probabilities)

        # Sample uniform random values: (n_select,)
        random_values = np.random.uniform(0.0, 1.0, size=n_select)

        # Binary search in CDF (vectorized)
        # For each random_value, find index i where cdf[i-1] < value <= cdf[i]
        selected_indices = np.searchsorted(cdf, random_values)

        # Clamp to valid range (handles floating-point edge cases)
        # E.g., random_value=1.0 might give index=pop_size (out of bounds)
        selected_indices = np.clip(selected_indices, 0, len(fitness) - 1)

        return selected_indices


class CrowdingSelection:
    """
    Crowding selection for steady-state GA (deterministic replacement).

    Algorithm:
        Returns sequential indices [0, 1, 2, ..., n_select-1].
        Actual crowding logic happens in replacement phase (not selection).

    Time Complexity: O(n_select)
    Space Complexity: O(n_select)

    Selection Pressure:
        - None during selection (deterministic)
        - Diversity maintenance happens in replacement (offspring vs parents)

    Properties:
        - Deterministic (no randomness in selection phase)
        - Maintains population diversity via crowding replacement
        - Offspring replace most similar parent (distance-based)

    Crowding Replacement Logic (in GA, not here):
        1. For each offspring:
           a. Sample k random parents
           b. Compute distance (hamming, edit, tour similarity)
           c. Replace most similar parent if offspring is better

    Reference:
        - De Jong (1975): "An Analysis of the Behavior of a Class of
          Genetic Adaptive Systems"
        - Mahfoud (1995): "Niching Methods for Genetic Algorithms"
        - Fujimoto & Tsutsui (2011): "An Efficient Deterministic Crowding
          Genetic Algorithm for Many-objective Optimization" (DCM-GA)

    Design Note:
        This is unusual among selection strategies - it's trivial here
        because complexity is in replacement. This matches DCM-GA pattern
        where offspring deterministically replace most similar parent.

    Example:
        >>> strategy = CrowdingSelection()
        >>> population = np.random.rand(100, 51)
        >>> fitness = np.random.rand(100) * 1000
        >>> indices = strategy.select(population, fitness, n_select=50, xp=np)
        >>> # indices: [0, 1, 2, ..., 49]
            >>> # Replacement logic in GA determines which survive
    """

    def select(
        self,
        population: Any,
        fitness: Any,
        n_select: int,
    ) -> Any:
        """
        Select n_select individuals (deterministic sequential indices).

        Phase 3.5: CPU-only execution using NumPy internally.

        Args:
            population: (pop_size, n+1) array of tours (not used)
            fitness: (pop_size,) array of costs (not used)
            n_select: Number of individuals to select

        Returns:
            (n_select,) array of sequential indices [0, 1, ..., n_select-1]

        Notes:
            - Population and fitness are ignored (deterministic selection)
            - Crowding logic happens during replacement phase
            - This matches DCM-GA design (Fujimoto & Tsutsui 2011)
        """
        return np.arange(n_select, dtype=np.int32)
