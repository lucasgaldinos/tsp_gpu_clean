"""
Initialization strategies for generating initial populations of routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from utils.phase_timer import time_strategy

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.algorithm_strategies import Routes
  from protocols.problem_context import ProblemContext


class RandomPermutationInitialization:
  """
  Initialize a population of routes `(population_size, n_cities)` as random permutations.

  Works with both CPU (NumPy) and GPU (CuPy) backends while maintaining
  deterministic reproducibility via the provided seed or `context.rng`.
  """

  def __init__(self, seed: int | None = None, return_gpu: bool = False) -> None:
    """
    Initialize random permutation strategy.

    Args:
        seed: Optional random seed override.
        return_gpu: If True and CuPy is available, return the initial routes
            batch directly on GPU (`cp.ndarray`).
    """
    self.seed = seed
    self.return_gpu = return_gpu
    self.rng = np.random.default_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """
    Generate `population_size` random route permutations of `customers`.

    Args:
        customers: Sequence of customer node indices to visit.
        population_size: Number of routes in the population batch.
        context: Optional `ProblemContext` providing a seeded RNG.

    Returns:
        2D routes array of shape `(population_size, len(customers))` with dtype `int32`.
    """
    n_cities = len(customers)
    customers_array = np.asarray(customers, dtype=np.int32)
    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    population = np.empty((population_size, n_cities), dtype=np.int32)
    for i in range(population_size):
      population[i] = rng.permutation(customers_array)

    if self.return_gpu and CUPY_AVAILABLE and cp is not None:
      self.h2d_bytes += int(population.nbytes)
      return cast("Routes", cp.asarray(population, dtype=cp.int32))
    return population
