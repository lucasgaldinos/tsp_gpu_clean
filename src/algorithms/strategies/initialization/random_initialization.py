"""
Initialization strategies for generating initial populations of routes.

Two constructions with different cost models (`DP11-1`):

- `InitializationRandom`: uniform Fisher-Yates permutations, `Theta(n)` per route.
- `InitializationNearestNeighbor`: randomized greedy nearest-neighbour tours guided by
  the `k`-NN candidate list (GRASP restricted candidate list), `Theta(n * k)` per route.

Keeping them apart makes the construction cost explicit in the configuration: a run that
asks for the uniform baseline cannot silently pay the candidate-guided price, and the
`candidate_k=None` fallback inside a single class disappears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from protocols.algorithm_strategies import InitializationStrategy
from protocols.backend import require_gpu
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng

from ..base import StrategyBase

if TYPE_CHECKING:
  from collections.abc import Sequence

  from numpy.random._generator import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


class InitializationRandom(StrategyBase, InitializationStrategy):
  """
  Uniform random permutation initialization.

  Each route is an independent Fisher-Yates shuffle of `customers`, so the population is
  a uniform sample of the `n!` feasible tours. Deterministic reproducibility comes from
  the explicit seed or `context.rng`.
  """

  def __init__(self, seed: SeedLike = None, return_gpu: bool = False) -> None:
    """
    Initialize the uniform permutation strategy.

    Args:
        seed: Optional random seed override (int, `SeedSequence`, existing
            `numpy.random.Generator`, zero-argument callable, or `None`).
        return_gpu: If True, return the routes batch on GPU (`cp.ndarray`); a machine
            without a CUDA device raises `GPUUnavailableError` instead of silently
            returning a host array (`DP9-2`).
    """
    super().__init__(seed)
    self.return_gpu: bool = return_gpu

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """
    Generate `population_size` uniform random permutations of `customers`.

    Args:
        customers: Sequence of customer node indices to visit.
        population_size: Number of routes in the population batch.
        context: Optional `ProblemContext` providing a seeded RNG.

    Returns:
        2D routes array of shape `(population_size, len(customers))` with dtype `int32`.

    Raises:
        GPUUnavailableError: When `return_gpu=True` and no CUDA device is reachable.
    """
    customers_array: np.ndarray = np.asarray(customers, dtype=np.int32)
    rng: Generator = resolve_rng(context, self.rng)

    population: np.ndarray = np.empty((population_size, len(customers)), dtype=np.int32)
    for i in range(population_size):
      population[i] = rng.permutation(customers_array)

    if self.return_gpu:
      cp = require_gpu()
      self.h2d_bytes += int(population.nbytes)
      return cast("Routes", cp.asarray(population, dtype=cp.int32))
    return population


class InitializationNearestNeighbor(StrategyBase, InitializationStrategy):
  """
  Randomized greedy nearest-neighbour (GRASP RCL) initialization over `N_k`.

  Each route starts at a uniformly drawn customer and then repeatedly picks the first
  still-unvisited entry of a randomly permuted `N_k` row. Restricting the draw to the
  candidate list turns the naive `Theta(n^2)` nearest-neighbour scan into
  `Theta(n * k)` while keeping the tour greedy -- the randomization of the candidate
  order is what restores diversity across the population.
  """

  def __init__(
    self,
    k: int = 20,
    seed: SeedLike = None,
    return_gpu: bool = False,
    candidate_k: int | None = None,
  ) -> None:
    """
    Initialize the candidate-guided nearest-neighbour strategy.

    Args:
        k: Candidate-list width: how many nearest neighbours each step may choose from.
        seed: Optional random seed override.
        return_gpu: If True and CuPy is available, return the routes batch on GPU.
        candidate_k: Alias for `k`; when provided it overrides `k`, so a configuration
            written against the old `candidate_k` key keeps working unchanged.
    """
    super().__init__(seed)
    resolved_k: int = k if candidate_k is None else candidate_k
    if resolved_k < 1:
      raise ValueError(f"Candidate-list width k must be >= 1, got {resolved_k}")
    self.k: int = resolved_k
    self.candidate_k: int = resolved_k
    self.return_gpu: bool = return_gpu

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
    Generate `population_size` randomized greedy nearest-neighbour tours of `customers`.

    Args:
        customers: Sequence of customer node indices to visit.
        population_size: Number of routes in the population batch.
        context: `ProblemContext` owning the cached `k`-NN candidate list. Required:
            the construction has no distances of its own.

    Returns:
        2D routes array of shape `(population_size, len(customers))` with dtype `int32`.
    """
    if context is None:
      raise ValueError("InitializationNearestNeighbor requires a ProblemContext to read the k-NN candidate list")

    customers_array: np.ndarray = np.asarray(customers, dtype=np.int32)
    rng: Generator = resolve_rng(context, self.rng)
    candidates: np.ndarray = context.get_cpu_candidate_list(self.candidate_k)

    population: np.ndarray = np.empty((population_size, len(customers)), dtype=np.int32)
    for i in range(population_size):
      population[i] = self._greedy_candidate_tour(customers_array, candidates, rng)

    if self.return_gpu:
      cp = require_gpu()
      self.h2d_bytes += int(population.nbytes)
      return cast("Routes", cp.asarray(population, dtype=cp.int32))
    return population


__all__: list[str] = ["InitializationNearestNeighbor", "InitializationRandom"]
