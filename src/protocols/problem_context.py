"""
ProblemContext: Backend-Specific Problem Data Holder with Lazy Distance Matrix Caching.

⚠️ ARCHITECTURAL NOTE ⚠️
This module is placed in `protocols/` for organizational consistency with other
strategy protocols, but ProblemContext is NOT a Protocol (PEP 544) interface.
It is a concrete class that wraps the immutable Problem dataclass and adds
backend-specific lazy computation of distance matrices.

The ProblemContext solves a critical performance anti-pattern identified in
flaws.md: sequential algorithms calling compute_distance_matrix() repeatedly
inside Python loops, causing O(n³) complexity instead of O(n²).

**CRITICAL: Fully Lazy Loading (Phase 5 Cleanup)**
- `__init__` stores problem and xp but DOES NOT allocate matrices
- All caches (_distances_cpu, _distances_gpu, etc.) start as None
- No eager @property accessors - forces explicit method calls
- Use `get_cpu_distances()` or `get_gpu_distances()` explicitly
- This prevents 14.4GB allocation crashes on large problems (n=30,000)

Design Philosophy:
- Single Responsibility: Holds all backend-specific problem data
- Immutability: Based on frozen Problem dataclass, computed data cached
- Performance: Distance matrix computed ONCE on target backend
- Lazy Loading: NO allocation until first explicit access
- Hybrid Bridge Compatible: Supports both CPU-only (S-Task) and GPU (P-Task)

Example:
    >>> from code.src.data_models.problem import Problem
    >>> from code.src.protocols.problem_context import ProblemContext
    >>> import numpy as np
    >>>
    >>> # Load problem
    >>> with DatabaseLoader("path/to/tsp.duckdb") as loader:
    ...   problem = loader.load("berlin52")
    >>>
    >>> # Create context (NO allocation yet - fully lazy)
    >>> context = ProblemContext(problem, xp=np, seed=42)
    >>>
    >>> # S-Task algorithm: Use CPU distances explicitly
    >>> distances_cpu = context.get_cpu_distances()  # Allocated here
    >>>
    >>> # P-Task algorithm: Use GPU distances explicitly
    >>> if CUPY_AVAILABLE:
    ...   distances_gpu = context.get_gpu_distances()  # Transfers to GPU

References:
    - flaws.md: Documents the distance matrix recomputation anti-pattern
    - flaw_analysis.md: Explains Class S (sequential) vs Class P (parallel)
    - architectural-decisions-and-questions.md: ProblemContext design rationale
    - Phase 5 Cleanup: Removed eager @property accessors (distances, demands, coordinates)
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from data_models.exceptions import VRAMInsufficientError
from data_models.problem import Problem
from distances.matrix import compute_candidate_list, compute_distance_matrix
from protocols.backend import BackendModule, CupyArray, require_gpu
from utils.rng import make_rng

if TYPE_CHECKING:
  from numpy.random import Generator


class ProblemContext:
  """
  Backend-specific problem data holder with cached distance matrix.

  This class wraps the immutable Problem dataclass and precomputes the
  distance matrix on the specified backend (NumPy or CuPy), caching it
  to avoid redundant O(n²) computations.

  **Key Performance Fix:**
  Before ProblemContext, TspConstructionStrategy.build_tour() was called
  k times (once per route), each time computing the distance matrix:
      - Total: k x O(m²) redundant work
      - For GPU: k x (CPU→GPU transfer + compute + GPU→CPU transfer)
      - Example: eil51 with k=5 routes = 17-82x slowdown

  After ProblemContext:
      - Distance matrix computed ONCE: O(N²) where N = total nodes
      - Cached on target backend (CPU or GPU)
      - All algorithms reuse the same cached matrix
      - No redundant transfers

  Attributes:
      xp: Backend module (NumPy or CuPy)
      problem: Original immutable Problem instance
      dimension: Number of nodes (problem.dimension)
      capacity: Vehicle capacity (CVRP only, None for TSP/ATSP)
      demands: Customer demands on target backend (xp.ndarray)
      distances: Cached distance matrix on target backend (xp.ndarray)
      coordinates: Coordinates on target backend (xp.ndarray or None)

  Example:
      >>> # Class S (sequential CPU algorithm)
      >>> context = ProblemContext(problem, xp=np)
      >>> tour = nearest_neighbor_strategy.construct(context, customers)
      >>> # No recomputation - uses context.distances

      >>> # Class P (parallel GPU algorithm)
      >>> context_gpu = ProblemContext(problem, xp=cp)
      >>> improved_tours = two_opt_gpu.improve_tours(tours, context_gpu)
      >>> # Distance matrix stays in VRAM, zero transfer overhead
  """

  def __init__(
    self,
    problem: Problem,
    xp: BackendModule,
    seed: int | list[int] | None = None,
    max_matrix_dimension: int = 10_000,
  ) -> None:
    """
    Initialize ProblemContext with lazy computation strategy.

    Unlike the previous eager implementation, this version does NOT
    precompute distance matrices or transfer data in __init__. All
    computations are deferred until first access, eliminating the
    wasteful double-transfer pattern for Class S algorithms.

    Args:
        problem: Immutable Problem instance from database loader
        xp: Backend module (numpy or cupy) - preferred backend hint
        seed: Random seed for RNG (int, list of ints, or None).
              If None, OS entropy is used. For parallel execution,
              use sequence of integers: [worker_id, root_seed].
              See NumPy docs on "Parallel random number generation".
        max_matrix_dimension: Largest `n` for which GPU strategies allocate the
              dense `(n, n)` matrix. Above it, coordinate-based problems switch to
              on-the-fly distances (`DP10-19`). Configurable for testing.

    Raises:
        GPUUnavailableError: If xp is CuPy but CuPy is not installed or CUDA device is unavailable.

    Note:
        The xp parameter indicates the PREFERRED backend but does not
        force eager computation on that backend. Actual computation
        happens lazily when get_cpu_*() or get_gpu_*() is called.

        RNG Strategy: Uses NumPy's "Sequence of integer seeds" pattern
        for parallel safety. RandomState accepts both single integers
        and sequences, providing cross-backend compatibility (NumPy/CuPy).

    Example:
        >>> # Create GPU context (no computation yet)
        >>> import cupy as cp
        >>> context = ProblemContext(problem, xp=cp)
        >>> # Class S algorithm calls get_cpu_distances()
        >>> # → Computes on CPU directly (no GPU transfer waste)
        >>> distances_cpu = context.get_cpu_distances()

        >>> # Parallel execution with independent RNG (SAFE)
        >>> context = ProblemContext(problem, xp=np, seed=[worker_id, root_seed])
        >>> # Each worker gets independent RNG stream
        >>> random_value = context.rng.random()

        >>> # CuPy not available - clear error
        >>> context = ProblemContext(problem, xp=cp)  # Raises GPUUnavailableError
    """
    # Validate GPU backend availability
    if xp != np:
      require_gpu()

    self.xp = xp
    self.problem = problem
    self.dimension = problem.dimension
    self.capacity = problem.capacity
    self.max_matrix_dimension: int = max_matrix_dimension

    # Create an independent `np.random.Generator` (NumPy "Sequence of integer seeds" pattern).
    # Strategies and `resolve_rng` operate on `Generator`, so the context must expose the
    # same type or the seeded stream is silently ignored.
    self.rng: Generator = make_rng(seed)

    # Store seed for debugging/reproducibility
    self.seed = seed

    # Lazy computation cache (all None until first access)
    self._cpu_distances: np.ndarray | None = None
    self._gpu_distances: CupyArray | None = None
    self._cpu_demands: np.ndarray | None = None
    self._gpu_demands: CupyArray | None = None
    self._cpu_coordinates: np.ndarray | None = None
    self._gpu_coordinates: CupyArray | None = None
    # `k`-NN candidate lists `N_k`, cached per requested `k` (`DP10-20`).
    self._cpu_candidate_lists: dict[int, np.ndarray] = {}
    self._gpu_candidate_lists: dict[int, CupyArray] = {}

    # Validate problem has data to compute from
    if problem.distances is None and problem.coordinates is None:
      raise ValueError(f"Problem '{problem.name}' has no distances or coordinates to compute distance matrix from.")

  def get_cpu_demands(self) -> np.ndarray | None:
    """
    Get demands as NumPy array on CPU (lazy computation with caching).

    This method implements lazy computation: demands are only moved/
    computed on first access and then cached for subsequent calls.

    For Class S (sequential) algorithms requiring CPU data, this
    eliminates wasteful GPU transfers when context was created with
    xp=cupy but algorithm needs CPU data.

    Returns:
        Demands array as NumPy array on CPU, or None if no demands

    Example:
        >>> # GPU context (xp=cp) but Class S bin packing needs CPU
        >>> context = ProblemContext(problem, xp=cp)
        >>> # First call: computes on CPU directly (no GPU waste)
        >>> cpu_demands = context.get_cpu_demands()
        >>> # Subsequent calls: returns cached value
        >>> same_demands = context.get_cpu_demands()
    """
    if self._cpu_demands is None and self.problem.demands is not None:
      # Lazy computation: extract demands on CPU
      self._cpu_demands = self.problem.demands  # Already NumPy array
    return self._cpu_demands

  def get_cpu_distances(self) -> np.ndarray:
    """
    Get distance matrix as NumPy array on CPU (lazy computation with caching).

    This method implements lazy computation: the distance matrix is only
    computed on CPU when first requested, eliminating wasteful GPU
    computation and transfer for Class S algorithms.

    Returns:
        Distance matrix as NumPy array on CPU, shape (n, n)

    Raises:
        ValueError: If problem has no distances or coordinates

    Example:
        >>> # Context created with xp=cp (GPU hint)
        >>> context = ProblemContext(problem, xp=cp)
        >>> # Class S algorithm needs CPU distances
        >>> # First call: computes on CPU directly (no GPU waste!)
        >>> cpu_dist = context.get_cpu_distances()
        >>> # Subsequent calls: returns cached CPU matrix
        >>> same_dist = context.get_cpu_distances()
    """
    if self._cpu_distances is None:
      # Lazy computation on CPU
      if self.problem.distances is not None:
        # EXPLICIT edge type - use pre-computed matrix
        self._cpu_distances = self.problem.distances
      elif self.problem.coordinates is not None:
        # Compute from coordinates on CPU (using NumPy)
        self._cpu_distances = compute_distance_matrix(self.problem.coordinates, self.problem.edge_type, np)
      else:
        raise ValueError(
          f"Problem '{self.problem.name}' has no distances or coordinates to compute distance matrix from."
        )
    return self._cpu_distances

  def get_cpu_coordinates(self) -> np.ndarray | None:
    """
    Get coordinates as NumPy array on CPU (lazy with caching).

    Returns:
        Coordinates as NumPy array on CPU, or None if not available
    """
    if self._cpu_coordinates is None and self.problem.coordinates is not None:
      # Lazy extraction: coordinates already NumPy array
      self._cpu_coordinates = self.problem.coordinates
    return self._cpu_coordinates

  def get_gpu_demands(self) -> CupyArray | None:
    """
    Get demands on GPU (lazy computation with caching).

    Transfers from CPU cache if available, otherwise from problem data.
    Only valid when xp is CuPy.

    Returns:
        Demands as CuPy array on GPU, or None if no demands

    Raises:
        ValueError: If xp is NumPy (not CuPy)
    """
    if self.xp == np:
      raise ValueError("get_gpu_demands() requires xp to be CuPy, not NumPy")

    if self._gpu_demands is None and self.problem.demands is not None:
      if self._cpu_demands is not None:
        # Transfer from CPU cache
        self._gpu_demands = self.xp.asarray(self._cpu_demands)
      else:
        # Transfer from problem data
        self._gpu_demands = self.xp.asarray(self.problem.demands)
    return self._gpu_demands

  def get_gpu_distances(self) -> CupyArray | None:
    """
    Get distance matrix on GPU (lazy computation with caching).

    This method computes or transfers the distance matrix to GPU only
    when first requested. If CPU cache exists, transfers from there.
    Otherwise, computes directly on GPU or transfers from problem data.

    Returns:
        Distance matrix as CuPy array on GPU, shape (n, n)

    Raises:
        ValueError: If xp is NumPy (not CuPy) or no data to compute from

    Example:
        >>> # Context with GPU hint
        >>> context = ProblemContext(problem, xp=cp)
        >>> # Class P algorithm needs GPU distances
        >>> # First call: computes on GPU and caches
        >>> gpu_dist = context.get_gpu_distances()
        >>> # Subsequent calls: returns cached GPU matrix (stays in VRAM)
        >>> same_dist = context.get_gpu_distances()
    """
    if self.xp == np:
      raise ValueError("get_gpu_distances() requires xp to be CuPy, not NumPy")

    if self._gpu_distances is None:
      # Fail before the allocation, not inside CUDA: a `(n, n)` `int32` matrix is
      # `4n^2` bytes and the driver aborts the process when it does not fit.
      self._raise_if_insufficient(
        4 * self.dimension * self.dimension,
        self._query_free_vram_bytes(),
        f"dense distance matrix (n={self.dimension})",
      )
      # Priority: CPU cache > problem.distances > compute from coordinates.
      # Every path lands on ``int32`` so GPU kernels can bind the matrix without
      # an implicit dtype conversion (``BUG-14`` / ``DP10-17``).
      if self._cpu_distances is not None:
        # Transfer from CPU cache (most efficient if CPU was used first)
        self._gpu_distances = self.xp.asarray(self._cpu_distances, dtype=self.xp.int32)
      elif self.problem.distances is not None:
        # Transfer EXPLICIT matrix to GPU
        self._gpu_distances = self.xp.asarray(self.problem.distances, dtype=self.xp.int32)
      elif self.problem.coordinates is not None:
        # Compute from coordinates on GPU
        coords_gpu = self.xp.asarray(self.problem.coordinates)
        self._gpu_distances = self.xp.asarray(
          compute_distance_matrix(coords_gpu, self.problem.edge_type, self.xp), dtype=self.xp.int32
        )
      else:
        raise ValueError(
          f"Problem '{self.problem.name}' has no distances or coordinates to compute distance matrix from."
        )
    return self._gpu_distances

  def get_gpu_coordinates(self) -> CupyArray | None:
    """
    Get coordinates on GPU (lazy with caching).

    Returns:
        Coordinates as CuPy array on GPU, or None if not available

    Raises:
        ValueError: If xp is NumPy (not CuPy)
    """
    if self.xp == np:
      raise ValueError("get_gpu_coordinates() requires xp to be CuPy, not NumPy")

    if self._gpu_coordinates is None and self.problem.coordinates is not None:
      if self._cpu_coordinates is not None:
        # Transfer from CPU cache
        self._gpu_coordinates = self.xp.asarray(self._cpu_coordinates)
      else:
        # Transfer from problem data
        self._gpu_coordinates = self.xp.asarray(self.problem.coordinates)
    return self._gpu_coordinates

  def get_cpu_candidate_list(self, k: int = 20) -> np.ndarray:
    """Get the `k`-NN candidate list `N_k` on CPU (lazy computation with caching).

    `N_k` has shape `(n, min(k, n - 1))` and dtype `int32`; row `u` holds the
    `k_eff` nearest nodes to `u` sorted by ascending distance, self excluded
    (`DP10-20`). Rows are built in blocks, so a coordinate-only problem never
    allocates the full `(n, n)` matrix.

    Args:
        k: Requested neighbours per node; clamped to `min(k, n - 1)`.

    Returns:
        Candidate list as a NumPy array on CPU.

    Raises:
        ValueError: If the problem carries neither distances nor coordinates.
    """
    cached = self._cpu_candidate_lists.get(k)
    if cached is None:
      if self.problem.distances is not None:
        cached = compute_candidate_list(
          self.problem.coordinates,
          self.problem.edge_type,
          k,
          np,
          distances=self.problem.distances,
        )
      elif self.problem.coordinates is not None:
        cached = compute_candidate_list(self.problem.coordinates, self.problem.edge_type, k, np)
      else:
        raise ValueError(
          f"Problem '{self.problem.name}' has no distances or coordinates to compute a candidate list from."
        )
      self._cpu_candidate_lists[k] = cached
    return cached

  def get_gpu_candidate_list(self, k: int = 20) -> CupyArray:
    """Get the `k`-NN candidate list `N_k` on GPU (lazy computation with caching).

    The table is built on CPU (blocked, `O(chunk_size * n)` peak) and transferred
    once as `int32`; `N_k` itself is only `4 * n * k` bytes (`DP10-20`).

    Args:
        k: Requested neighbours per node; clamped to `min(k, n - 1)`.

    Returns:
        Candidate list as a CuPy array on GPU, shape `(n, min(k, n - 1))`.

    Raises:
        ValueError: If xp is NumPy (not CuPy).
    """
    if self.xp == np:
      raise ValueError("get_gpu_candidate_list() requires xp to be CuPy, not NumPy")

    cached = self._gpu_candidate_lists.get(k)
    if cached is None:
      cached = self.xp.asarray(self.get_cpu_candidate_list(k), dtype=self.xp.int32)
      self._gpu_candidate_lists[k] = cached
    return cached

  def uses_on_the_fly_distances(self, n: int, max_matrix_dimension: int | None = None) -> bool:
    """Return True when GPU kernels should compute distances from coordinates.

    The dense `(n, n)` matrix costs `4n^2` bytes in VRAM; above
    `max_matrix_dimension` a coordinate-based problem is served on the fly by
    `gpu_pairwise_distance` instead (`DP10-19`).

    Args:
        n: Number of cities in the current batch (matrix row width).
        max_matrix_dimension: Threshold override; falls back to the context value.

    Returns:
        True when `n` exceeds the threshold and the problem carries coordinates
        with a coordinate-based `EDGE_WEIGHT_TYPE`.
    """
    threshold = self.max_matrix_dimension if max_matrix_dimension is None else max_matrix_dimension
    return (
      n > threshold
      and self.problem.edge_type != "EXPLICIT"
      and self.problem.coordinates is not None
    )

  def _coordinate_bytes(self) -> int:
    """Return VRAM bytes for the coordinate buffer (`8 * d * n`, `float64`)."""
    n = self.dimension
    coordinates = self.problem.coordinates
    components = coordinates.shape[1] if coordinates is not None and coordinates.ndim == 2 else 1
    return 8 * components * n

  def _query_free_vram_bytes(self) -> int | None:
    """Return free device VRAM in bytes, or `None` on the NumPy backend.

    CuPy ships no type information, so `BackendModule` cannot declare `cuda`;
    this is the project's documented third-party `Any` boundary
    (`protocols/backend.py`).
    """
    if self.xp == np:
      return None
    cuda: Any = cast("Any", self.xp).cuda
    free_bytes, _total_bytes = cuda.Device().mem_info
    return int(free_bytes)

  def _raise_if_insufficient(self, required_bytes: int, available_bytes: int | None, operation: str) -> None:
    """Raise `VRAMInsufficientError` when `required_bytes` exceeds the available VRAM."""
    if available_bytes is not None and required_bytes > available_bytes:
      raise VRAMInsufficientError(
        f"Insufficient VRAM for {operation}",
        required_bytes=required_bytes,
        available_bytes=available_bytes,
      )

  def estimate_vram_bytes(self, pop_size: int) -> int:
    """Estimate the VRAM bytes needed to evaluate a population of `pop_size` tours.

    Dense-matrix mode (`DP10-19`) accounts for the `int32` distance matrix
    (`4n^2`), three `int32` population/offspring buffers (`12Pn`), three
    `int64`/`float64` per-individual fitness buffers (`24P`) and the coordinate
    or auxiliary buffers (`8dn`, `d=1` without a 2-D coordinate array).
    On-the-fly mode drops the matrix term, keeping memory `O(n)`.

    Args:
        pop_size: Number of candidate tours held at once.

    Returns:
        Required VRAM in bytes.

    Raises:
        ValueError: If `pop_size` is negative.
    """
    if pop_size < 0:
      raise ValueError(f"pop_size must be non-negative, got {pop_size}")
    n = self.dimension
    population_bytes = 12 * pop_size * n
    fitness_bytes = 24 * pop_size
    auxiliary_bytes = self._coordinate_bytes()
    if self.uses_on_the_fly_distances(n):
      return population_bytes + fitness_bytes + auxiliary_bytes
    return 4 * n * n + population_bytes + fitness_bytes + auxiliary_bytes

  def check_vram_capacity(self, pop_size: int, available_bytes: int | None = None) -> int:
    """Validate that `pop_size` tours fit in the available VRAM.

    Args:
        pop_size: Number of candidate tours held at once.
        available_bytes: Free VRAM override (tests and dry runs). When `None`
            and the backend is CuPy, free VRAM is read from
            `xp.cuda.Device().mem_info`; on the NumPy backend there is no VRAM
            budget, so the check reduces to returning the estimate.

    Returns:
        The required VRAM in bytes from `estimate_vram_bytes(pop_size)`.

    Raises:
        VRAMInsufficientError: If the estimate exceeds the available VRAM.
    """
    required_bytes = self.estimate_vram_bytes(pop_size)
    free_bytes = available_bytes if available_bytes is not None else self._query_free_vram_bytes()
    self._raise_if_insufficient(
      required_bytes,
      free_bytes,
      f"population evaluation (pop_size={pop_size}, n={self.dimension})",
    )
    return required_bytes

  def __repr__(self) -> str:
    """String representation showing problem and backend."""
    backend_name = "GPU (CuPy)" if hasattr(self.xp, "RawKernel") else "CPU (NumPy)"
    return f"<ProblemContext: {self.problem.name} ({self.problem.problem_type}, N={self.dimension}) on {backend_name}>"
