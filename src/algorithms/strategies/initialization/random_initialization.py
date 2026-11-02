"""
Initialization strategies for generating initial populations of routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from utils.phase_timer import time_strategy
from utils.rng import make_rng, resolve_rng

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from collections.abc import Sequence

  from numpy.random._generator import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


class RandomPermutationInitialization:
  """
  Initialize a population of routes `(population_size, n_cities)` as random permutations.

  Works with both CPU (NumPy) and GPU (CuPy) backends while maintaining
  deterministic reproducibility via the provided seed or `context.rng`.
  """

  def __init__(self, seed: SeedLike = None, return_gpu: bool = False) -> None:
    """
    Initialize random permutation strategy.

    Args:
        seed: Optional random seed override (int, Generator, callable, or None).
        return_gpu: If True and CuPy is available, return the initial routes
            batch directly on GPU (`cp.ndarray`).
    """
    self.seed: SeedLike = seed
    self.return_gpu = return_gpu
    self.rng = make_rng(seed)
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
    n_cities: int = len(customers)
    customers_array: np.ndarray = np.asarray(customers, dtype=np.int32)
    rng: Generator = resolve_rng(context, self.rng)

    population: np.ndarray = np.empty((population_size, n_cities), dtype=np.int32)
    for i in range(population_size):
      population[i] = rng.permutation(customers_array)

    if self.return_gpu and CUPY_AVAILABLE and cp is not None:
      self.h2d_bytes += int(population.nbytes)
      return cast("Routes", cp.asarray(population, dtype=cp.int32))
    return population
