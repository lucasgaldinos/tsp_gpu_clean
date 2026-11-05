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

  def __init__(
    self,
    seed: SeedLike = None,
    return_gpu: bool = False,
    candidate_k: int | None = None,
  ) -> None:
    """
    Initialize random permutation strategy.

    Args:
        seed: Optional random seed override (int, Generator, callable, or None).
        return_gpu: If True and CuPy is available, return the initial routes
            batch directly on GPU (`cp.ndarray`).
        candidate_k: When set and a `context` is provided, tours are built as
            randomized greedy nearest-neighbour tours guided by the `k`-NN candidate
            list (`DP10-20`); `None` keeps uniform random permutations.
    """
    self.seed: SeedLike = seed
    self.return_gpu = return_gpu
    self.candidate_k: int | None = candidate_k
    self.rng = make_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  @staticmethod
  def _greedy_candidate_tour(
    customers: np.ndarray,
    candidates: np.ndarray,
    rng: Generator,
  ) -> np.ndarray:
    """Build one randomized nearest-neighbour tour restricted to `customers`.

    At each step the candidate list `N_k[current]` is walked in a random order and the
    first still-unvisited customer is appended (GRASP / restricted candidate list);
    when every candidate is already visited the tour falls back to a uniform draw from
    the remaining customers. The result is always a permutation of `customers`.
    """
    n_cities: int = int(customers.size)
    if n_cities == 0:
      return customers.astype(np.int32, copy=True)

    max_node: int = int(customers.max())
    index_of = np.full(max_node + 1, -1, dtype=np.int32)
    index_of[customers] = np.arange(n_cities, dtype=np.int32)
    candidates = candidates[: max_node + 1]

    visited = np.zeros(max_node + 1, dtype=bool)
    tour = np.empty(n_cities, dtype=np.int32)
    current: int = int(customers[rng.integers(n_cities)])
    tour[0] = current
    visited[current] = True

    for step in range(1, n_cities):
      chosen: int = -1
      for r in rng.permutation(int(candidates.shape[1])):
        neighbor = int(candidates[current, int(r)])
        if neighbor <= max_node and index_of[neighbor] >= 0 and not visited[neighbor]:
          chosen = neighbor
          break
      if chosen < 0:
        chosen = int(rng.choice(customers[~visited[customers]]))
      tour[step] = chosen
      visited[chosen] = True
      current = chosen

    return tour

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
        context: Optional `ProblemContext` providing a seeded RNG and, when
            `candidate_k` is set, the `k`-NN candidate list.

    Returns:
        2D routes array of shape `(population_size, len(customers))` with dtype `int32`.
    """
    n_cities: int = len(customers)
    customers_array: np.ndarray = np.asarray(customers, dtype=np.int32)
    rng: Generator = resolve_rng(context, self.rng)

    use_candidates: bool = self.candidate_k is not None and self.candidate_k > 0 and context is not None
    candidates: np.ndarray | None = None
    if use_candidates and context is not None and self.candidate_k is not None:
      candidates = context.get_cpu_candidate_list(self.candidate_k)

    population: np.ndarray = np.empty((population_size, n_cities), dtype=np.int32)
    for i in range(population_size):
      if candidates is not None:
        population[i] = self._greedy_candidate_tour(customers_array, candidates, rng)
      else:
        population[i] = rng.permutation(customers_array)

    if self.return_gpu and CUPY_AVAILABLE and cp is not None:
      self.h2d_bytes += int(population.nbytes)
      return cast("Routes", cp.asarray(population, dtype=cp.int32))
    return population
