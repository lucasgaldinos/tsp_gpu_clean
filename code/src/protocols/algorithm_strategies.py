"""
Algorithm strategy protocols for "Lego Blocks" compositional architecture.

This module defines protocol interfaces for interchangeable algorithm strategies:
- BinPackingStrategy: Capacity-based customer grouping algorithms
- TspConstructionStrategy: Tour construction heuristics
- ClusteringStrategy: Spatial clustering algorithms

These protocols enable runtime algorithm selection via dependency injection,
supporting the compositional "Lego Blocks" design pattern.

Design Pattern References:
    - Strategy Pattern (GoF Design Patterns)
    - Dependency Injection (Fowler, 2004)
    - Protocol-Based Programming (PEP 544)

Example Usage:
    >>> # Mix and match algorithms
    >>> solver = lego_cvrp_solver(
    ...     locations, demands, capacity,
    ...     bin_packing_strategy=FFDStrategy(),  # Swap with BFDStrategy()
    ...     tsp_strategy=NearestNeighborStrategy(),  # Swap with ChristofidesStrategy()
    ...     xp=cp  # Backend flows through
    ... )

See Also:
    - code/src/algorithms/strategies/bin_packing_strategies.py
    - code/src/algorithms/strategies/tsp_strategies.py
    - code/src/algorithms/compositional_cvrp_solver.py
"""

from typing import Protocol, List, TYPE_CHECKING, Tuple, Dict, Any
import numpy as np
from .backend import BackendModule

if TYPE_CHECKING:
    from .problem_context import ProblemContext


# ==============================================================================
# BIN PACKING STRATEGY PROTOCOL
# ==============================================================================


class BinPackingStrategy(Protocol):
    """
    Protocol for bin packing strategies (Class S - Sequential, CPU-only).

    **Architectural Classification: Class S (Sequential)**
    Bin packing strategies use greedy heuristics that are inherently sequential.
    They always run on CPU regardless of ProblemContext backend, using the
    context.get_cpu_demands() helper to extract demands.

    **Design Philosophy**:
    Bin packing is a PURE CAPACITY PROBLEM - it doesn't need to know about:
    - Depot locations or indices
    - Spatial coordinates
    - Problem-specific structure (CVRP vs MDVRP vs VRPTW)

    The compositional solver is responsible for:
    - Identifying which nodes are customers (vs depots, stations, etc.)
    - Passing the appropriate customer_indices to pack()
    - Handling problem-specific constraints outside of capacity

    This separation enables:
    - MDVRP: Pack customers for each depot independently
    - Cluster-first: Pack each cluster subset separately
    - Route-first: Pack all customers, then organize routes
    - Time windows: Pre-filter customers by time compatibility

    Methods:
        pack: Group customers into capacity-constrained bins

    Example (Single-depot CVRP):
        >>> from code.src.protocols import ProblemContext
        >>> from code.src.data import Problem
        >>> import numpy as np
        >>>
        >>> problem = Problem(
        ...     coordinates=np.array([[0,0], [1,1], [2,2], [3,3], [4,4]]),
        ...     demands=np.array([0, 30, 40, 20, 50]),
        ...     capacity=100
        ... )
        >>> context = ProblemContext(problem, xp=np)
        >>>
        >>> strategy = FFDStrategy()
        >>> all_customers = [1, 2, 3, 4]  # Exclude depot at index 0
        >>> bins = strategy.pack(context, all_customers)
        >>> # bins = [[4, 1], [2, 3]]  (50+30=80, 40+20=60)

    Example (Cluster-first CVRP):
        >>> cluster1 = [1, 2]  # Customers in spatial cluster 1
        >>> cluster2 = [3, 4]  # Customers in spatial cluster 2
        >>> bins_c1 = strategy.pack(context, cluster1)
        >>> bins_c2 = strategy.pack(context, cluster2)

    Example (MDVRP - Multi-depot):
        >>> depot1_customers = [1, 2, 5]  # Customers assigned to depot 0
        >>> depot2_customers = [3, 4, 6]  # Customers assigned to depot 7
        >>> bins_d1 = strategy.pack(context, depot1_customers)
        >>> bins_d2 = strategy.pack(context, depot2_customers)

    Time Complexity:
        O(n log n) for FFD/BFD strategies where n = len(customer_indices)

    Implementation Notes:
        Strategies should:
        1. Extract demands: demands = context.get_cpu_demands()[customer_indices]
        2. Call underlying algorithm: bins_local = algorithm.pack(demands, capacity)
        3. Map indices back: bins_global = [[customer_indices[i] for i in bin] for bin in bins_local]

        This pattern handles arbitrary customer subsets and problem structures.
    """

    def pack(
        self, context: "ProblemContext", customer_indices: List[int]
    ) -> List[List[int]]:
        """
        Group customers into capacity-constrained bins.

        Args:
            context: ProblemContext with demands and capacity on target backend
            customer_indices: Indices of customers to pack (excludes depots/stations)

        Returns:
            List of bins, where each bin is a list of customer indices from
            the input customer_indices. Indices are in the GLOBAL problem space.

        Raises:
            ValueError: If any customer demand exceeds capacity
            ValueError: If customer_indices contains invalid indices

        Example:
            >>> # Input: customer_indices = [1, 3, 5]  (3 customers to pack)
            >>> # Context: demands = [0, 30, 0, 40, 0, 50], capacity = 100
            >>> bins = strategy.pack(context, [1, 3, 5])
            >>> # bins = [[5, 1], [3]]  (Global indices: 50+30=80, 40)
        """
        ...


# ==============================================================================
# TSP CONSTRUCTION STRATEGY PROTOCOL
# ==============================================================================


class TspConstructionStrategy(Protocol):
    """
    Protocol for TSP tour construction heuristics (Class S - CPU only).

    **Architectural Classification: Class S (Sequential)**
    These algorithms are inherently sequential with step k+1 depending on step k.
    They run on CPU regardless of ProblemContext backend. When context is on GPU,
    strategies use context.get_cpu_distances() for explicit data transfer.

    This is acceptable because:
    1. Sequential algorithms cannot leverage GPU parallelism
    2. Forcing them to run with GPU kernel calls creates massive overhead
    3. The single transfer is cheaper than k × (compute + transfer) anti-pattern

    Protocol Requirements:
        - Accept ProblemContext and customer indices
        - Return tour as ordered list of customer indices
        - Tour must visit each customer exactly once
        - Tour implicitly returns to start (depot)
        - Use context.get_cpu_distances() for CPU-based computation

    Design Rationale:
        ProblemContext eliminates k × O(m²) redundant distance computations
        (where k = number of routes, m = avg route size). Distance matrix is
        computed ONCE during context creation, then cached.

    Implementations:
        - NearestNeighborStrategy: O(n²) greedy construction
        - ChristofidesStrategy: 1.5-approximation for metric TSP
        - RandomStrategy: Random permutation (baseline)
        - SweepStrategy: Angular sorting from depot

    Example:
        >>> from code.src.protocols import ProblemContext
        >>> import numpy as np
        >>>
        >>> context = ProblemContext(problem, xp=np)
        >>> strategy = NearestNeighborStrategy()
        >>> tour = strategy.build_tour(context, customers=[1, 2, 3])
        >>> # tour = [0, 1, 3, 2, 0]  (depot → 1 → 3 → 2 → depot)
    """

    def build_tour(self, context: "ProblemContext", customers: List[int]) -> List[int]:
        """
        Construct TSP tour for given customers using context's cached distances.

        Args:
            context: ProblemContext with precomputed distance matrix
            customers: List of customer indices to visit (NOT including depot)

        Returns:
            Tour as ordered list of node indices [depot, c1, c2, ..., depot]

        Raises:
            ValueError: If customers list is empty
            ValueError: If depot (0) is in customers list

        Note:
            Implementations should use context.get_cpu_distances() to obtain
            the distance matrix as a NumPy array for CPU-based computation.
            This handles the backend transfer transparently.
        """
        ...
        ...


# ==============================================================================
# TSP IMPROVEMENT STRATEGY PROTOCOL
# ==============================================================================


class TspImprovementStrategy(Protocol):
    """
    Protocol for TSP tour improvement (local search) algorithms (Class P - Parallelizable).

    **Architectural Classification: Class P (Parallel)**
    These algorithms have parallelizable components that can benefit from GPU acceleration.
    They are the primary target for CPU vs GPU benchmarking.

    Unlike Class S (construction), improvement algorithms evaluate many candidate moves
    in parallel (e.g., all O(n²) 2-opt swaps), making them suitable for GPU acceleration.

    Protocol Requirements:
        - Accept ProblemContext and existing tour
        - Return improved tour with same structure (depot at start/end)
        - Tour must remain valid (visits each customer exactly once)
        - Should reduce or maintain tour cost (no worsening)
        - Use context.distances directly (respects backend)

    Design Rationale:
        Separating improvement from construction enables:
        - Fast construction + intensive improvement pipeline
        - GPU acceleration of improvement phase (parallel neighborhood search)
        - Iterative improvement (apply multiple times)
        - Benchmarking improvement impact

    Implementations:
        - TwoOptCPU: CPU-based 2-opt local search (sequential evaluation)
        - TwoOptGPU: GPU-accelerated 2-opt (Fujimoto 2011, parallel evaluation)
        - ThreeOpt: 3-opt local search (more expensive, better solutions)
        - LinKernighan: Variable-depth search (state-of-the-art)
        - SimulatedAnnealing: Metaheuristic with probabilistic acceptance

    Example:
        >>> from code.src.protocols import ProblemContext
        >>> import numpy as np
        >>>
        >>> # Construct initial tour
        >>> context = ProblemContext(problem, xp=np)
        >>> construction = NearestNeighborStrategy()
        >>> tour = construction.build_tour(context, [1, 2, 3, 4, 5])
        >>>
        >>> # Improve with CPU 2-opt
        >>> improvement = TwoOptCPU()
        >>> improved_tour = improvement.improve_tour(context, tour)
        >>>
        >>> # Or use GPU for larger problems
        >>> import cupy as cp
        >>> gpu_context = ProblemContext(problem, xp=cp)
        >>> gpu_improvement = TwoOptGPU()
        >>> gpu_improved = gpu_improvement.improve_tour(gpu_context, tour)

    GPU Acceleration Note:
        GPU implementations (e.g., TwoOptGPU) should:
        - Use CuPy RawKernel for custom CUDA code
        - Exploit parallelism in neighborhood evaluation
        - Keep data on device (context.distances stays in VRAM)
        - Handle synchronization correctly
    """

    def improve_tour(self, context: "ProblemContext", tour: List[int]) -> List[int]:
        """
        Improve TSP tour using local search.

        Args:
            context: ProblemContext with distance matrix on target backend
            tour: Current tour [depot, c1, c2, ..., ck, depot]

        Returns:
            Improved tour with same structure (same customers, better cost)

        Raises:
            ValueError: If tour is invalid (duplicate nodes, missing customers)
            ValueError: If backend mismatch (e.g., GPU-only algorithm with CPU context)

        Note:
            Implementations should check context.xp to determine backend and
            use context.distances directly for computation. GPU implementations
            should guard with: if not hasattr(context.xp, 'RawKernel'): raise ValueError
        """
        ...


# ==============================================================================
# TSP METAHEURISTIC STRATEGY PROTOCOL
# ==============================================================================


class TspMetaheuristicStrategy(Protocol):
    """
    Protocol for metaheuristic TSP solvers (Class P - Parallelizable with Hyperparameters).

    **Architectural Classification: Class P-Meta (Parallel Metaheuristic)**
    Metaheuristics are iterative optimization algorithms that explore solution spaces
    using stochastic search strategies. They can benefit from GPU acceleration through:
    - Parallel fitness evaluation (population-based methods)
    - Batch neighbor generation (trajectory-based methods)
    - Vectorized update operations

    Unlike deterministic heuristics (NN, Christofides), metaheuristics:
    1. Require hyperparameter tuning (population size, temperature, iterations, etc.)
    2. Return optimization statistics (convergence history, best fitness, etc.)
    3. Use randomness and may produce different results on each run
    4. Can escape local optima (unlike greedy construction)

    **Protocol Requirements:**
    - Accept ProblemContext and customer indices
    - Support hyperparameter configuration via set_params()
    - Return tour with optimization statistics via build_tour_with_stats()
    - Provide statistics access via get_stats()
    - Support both CPU and GPU backends (check context.xp)

    **Design Rationale:**
    Separating metaheuristics from construction/improvement enables:
    - Hyperparameter optimization workflows (grid search, Bayesian optimization)
    - Performance profiling (convergence analysis, iteration timing)
    - Algorithm comparison (GA vs SA vs ACO with same interface)
    - Hybrid approaches (use NN for initialization, SA for refinement)

    **Implementations:**
    - GeneticAlgorithmStrategy: Population-based evolutionary search
      * Hyperparams: population_size, generations, mutation_rate, crossover_rate
      * GPU parallelism: Fitness evaluation, selection, crossover

    - SimulatedAnnealingStrategy: Trajectory-based probabilistic search
      * Hyperparams: initial_temp, cooling_rate, max_iterations
      * GPU parallelism: Batch neighbor generation, acceptance evaluation

    - AntColonyStrategy: Pheromone-based probabilistic construction
      * Hyperparams: num_ants, alpha, beta, evaporation_rate, iterations
      * GPU parallelism: Ant tour construction, pheromone updates

    - ParticleSwarmStrategy: Swarm intelligence optimization
      * Hyperparams: num_particles, inertia, cognitive, social, iterations
      * GPU parallelism: Particle position updates, fitness evaluation

    **Statistics Dictionary:**
    All implementations should return statistics with these standardized keys:

    ```python
    stats = {
        "best_fitness": float,          # Best tour cost found
        "final_fitness": float,          # Final tour cost (may differ from best)
        "iterations": int,               # Number of iterations executed
        "convergence_history": List[float],  # Fitness by iteration
        "runtime_seconds": float,        # Total execution time
        "hyperparameters": Dict[str, Any],   # Hyperparameter values used

        # Algorithm-specific (optional):
        "population_diversity": List[float],  # GA/PSO: diversity over time
        "temperature_schedule": List[float],   # SA: temperature by iteration
        "acceptance_rate": float,              # SA: fraction of moves accepted
        "crossover_count": int,                # GA: successful crossovers
        "mutation_count": int,                 # GA: successful mutations
    }
    ```

    **Example - Genetic Algorithm:**
    ```python
    >>> from code.src.protocols import ProblemContext
    >>> import numpy as np
    >>>
    >>> context = ProblemContext(problem, xp=np)
    >>> strategy = GeneticAlgorithmStrategy()
    >>>
    >>> # Configure hyperparameters
    >>> strategy.set_params(
    ...     population_size=100,
    ...     generations=200,
    ...     mutation_rate=0.1,
    ...     crossover_rate=0.8
    ... )
    >>>
    >>> # Solve with statistics
    >>> tour, stats = strategy.build_tour_with_stats(context, [1, 2, 3, 4, 5])
    >>> print(f"Best fitness: {stats['best_fitness']}")
    >>> print(f"Converged in {stats['iterations']} generations")
    >>>
    >>> # Access statistics later
    >>> final_stats = strategy.get_stats()
    ```

    **Example - Simulated Annealing:**
    ```python
    >>> strategy = SimulatedAnnealingStrategy()
    >>> strategy.set_params(
    ...     initial_temp=1000.0,
    ...     cooling_rate=0.95,
    ...     max_iterations=10000
    ... )
    >>> tour, stats = strategy.build_tour_with_stats(context, customers)
    >>>
    >>> # Analyze convergence
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(stats['convergence_history'])
    >>> plt.xlabel('Iteration')
    >>> plt.ylabel('Tour Cost')
    >>> plt.show()
    ```

    **GPU Acceleration Note:**
    GPU implementations should:
    - Check `hasattr(context.xp, 'RawKernel')` to verify CuPy backend
    - Use `context.distances` directly (stays on device)
    - Batch operations for parallel evaluation
    - Synchronize at statistics collection points
    - Report GPU-specific stats (kernel launches, memory transfers)

    **Time Complexity:**
    - Genetic Algorithm: $O(g \\cdot p \\cdot n^2)$ where $g$ = generations, $p$ = population
    - Simulated Annealing: $O(i \\cdot n^2)$ where $i$ = iterations
    - GPU acceleration typically provides 10-50x speedup for fitness evaluation
    """

    def set_params(self, **hyperparameters) -> None:
        """
        Configure metaheuristic hyperparameters.

        Args:
            **hyperparameters: Algorithm-specific hyperparameters
                Common examples:
                - population_size (int): Population size for GA/PSO
                - generations (int): Number of generations for GA
                - mutation_rate (float): Mutation probability for GA
                - crossover_rate (float): Crossover probability for GA
                - initial_temp (float): Initial temperature for SA
                - cooling_rate (float): Temperature decay for SA
                - max_iterations (int): Iteration limit for SA/ACO
                - num_ants (int): Colony size for ACO
                - alpha (float): Pheromone importance for ACO
                - beta (float): Heuristic importance for ACO

        Raises:
            ValueError: If hyperparameter values are invalid
            TypeError: If hyperparameter names are not recognized

        Example:
            >>> strategy = GeneticAlgorithmStrategy()
            >>> strategy.set_params(
            ...     population_size=200,
            ...     generations=500,
            ...     mutation_rate=0.05,
            ...     crossover_rate=0.9
            ... )
        """
        ...

    def build_tour_with_stats(
        self, context: "ProblemContext", customers: List[int]
    ) -> tuple[List[int], dict]:
        """
        Construct TSP tour using metaheuristic optimization with statistics.

        Args:
            context: ProblemContext with precomputed distance matrix
            customers: List of customer indices to visit (NOT including depot)

        Returns:
            Tuple of (tour, statistics):
            - tour: Best tour found [depot, c1, c2, ..., depot]
            - statistics: Dict with optimization metrics (see class docstring)

        Raises:
            ValueError: If customers list is empty
            ValueError: If depot (0) is in customers list
            ValueError: If hyperparameters not set (call set_params() first)
            RuntimeError: If optimization fails to converge

        Note:
            Implementations should:
            1. Validate hyperparameters are set
            2. Initialize population/starting solution
            3. Track best fitness and convergence history
            4. Use context.distances (respects backend)
            5. Return standardized statistics dictionary
        """
        ...

    def get_stats(self) -> dict:
        """
        Retrieve statistics from most recent optimization run.

        Returns:
            Dictionary with optimization statistics (see class docstring)
            Returns empty dict if build_tour_with_stats() not called yet

        Example:
            >>> tour, _ = strategy.build_tour_with_stats(context, customers)
            >>> stats = strategy.get_stats()
            >>> print(f"Best fitness: {stats['best_fitness']}")
            >>> print(f"Iterations: {stats['iterations']}")
            >>> print(f"Runtime: {stats['runtime_seconds']:.3f}s")
        """
        ...


# ==============================================================================
# CLUSTERING STRATEGY PROTOCOL
# ==============================================================================


class ClusteringStrategy(Protocol):
    """
    Protocol for spatial clustering algorithms.

    Implementations must partition customers into spatial clusters based on
    geographic proximity. This is Phase 0 of spatial-aware CVRP approaches,
    applied before bin packing and routing.

    Protocol Requirements:
        - Accept location coordinates, demands, backend module
        - Return list of Cluster objects (customer indices + metadata)
        - Clusters should be spatially coherent (minimize inter-cluster distance)
        - May consider demand distribution for load balancing

    Implementations:
        - KMeansStrategy: k-means clustering (requires k parameter)
        - DBSCANStrategy: Density-based clustering (finds k automatically)
        - SweepStrategy: Angular partitioning from depot
        - GridStrategy: Spatial grid decomposition

    Example:
        >>> strategy = KMeansStrategy(k=5)
        >>> clusters = strategy.cluster(locations, demands, xp=np)
        >>> # clusters[0].customer_indices = [1, 3, 7, 9]
        >>> # clusters[0].centroid = array([10.5, 20.3])

    Note:
        This protocol is OPTIONAL in compositional solver. If not provided,
        all customers treated as single cluster.
    """

    def cluster(
        self, locations: np.ndarray, demands: np.ndarray, xp: BackendModule = np
    ) -> List["Cluster"]:
        """
        Partition customers into spatial clusters.

        Args:
            locations: (n, 2) array of customer coordinates
            demands: (n,) array of customer demands
            xp: Backend module (NumPy or CuPy)

        Returns:
            List of Cluster objects with customer indices and metadata

        Raises:
            ValueError: If locations and demands have different lengths
        """
        ...


# ==============================================================================
# METAHEURISTIC STRATEGY PROTOCOL
# ==============================================================================


class MetaheuristicStrategy(Protocol):
    """
    Protocol for metaheuristic strategies.
    """

    def improve(
        self, initial_tour: np.ndarray, distances: np.ndarray, xp: BackendModule
    ) -> Tuple[np.ndarray, float]:
        """
        Improves a given tour using a metaheuristic strategy.

        Args:
            initial_tour (np.ndarray): The initial tour to improve.
            distances (np.ndarray): The distance matrix.
            xp: The backend module (NumPy or CuPy).

        Returns:
            A tuple containing the improved tour and its total distance.
        """
        ...
