"""Boundary contracts for the CPU 2-opt implementations in `two_opt.py`.

Every variant must be a no-op for tours shorter than one 2-opt move and for a
non-positive iteration budget, while still returning a defensive copy so caller-owned
routes are never aliased.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.strategies import (
  BatchGPUMultiMoveTwoOpt,
  BatchGPUTwoOpt,
  CPUFirstImprovementTwoOpt,
  CPUMultiMoveTwoOpt,
  CPUTwoOpt,
)
from data_models.problem import Problem
from protocols import ProblemContext
from protocols.backend import cupy_backend

if TYPE_CHECKING:
  from collections.abc import Callable

CPU_TWO_OPT_FACTORIES = [
  pytest.param(CPUTwoOpt, id="best-improvement"),
  pytest.param(CPUFirstImprovementTwoOpt, id="first-improvement"),
  pytest.param(CPUMultiMoveTwoOpt, id="multi-move"),
]

GPU_TWO_OPT_FACTORIES = [
  pytest.param(BatchGPUTwoOpt, id="batch"),
  pytest.param(BatchGPUMultiMoveTwoOpt, id="multi-move"),
]


@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_short_tour_returns_equal_distinct_copy(
  factory: Callable[[int], CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  burma14_problem: Problem,
) -> None:
  """WHAT: A three-city tour through every CPU 2-opt variant.

  WHY: A 2-opt move needs four distinct positions; shorter tours must be returned
  untouched, but still as independent memory so callers cannot alias the result.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.array([0, 2, 1], dtype=np.int32)
  original = tour.copy()

  improved = factory(10).improve(tour, context=context)

  assert improved is not tour, "boundary path must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"


@pytest.mark.parametrize("max_iterations", [0, -1], ids=["zero", "negative"])
@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_non_positive_iteration_budget_returns_equal_distinct_copy(
  factory: Callable[[int], CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  max_iterations: int,
  burma14_problem: Problem,
) -> None:
  """WHAT: A full burma14 tour evaluated with `max_iterations` of 0 and -1.

  WHY: A non-positive budget disables improvement; returning the input itself would
  let a later in-place stage corrupt the caller's population.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.arange(burma14_problem.dimension, dtype=np.int32)
  original = tour.copy()

  improved = factory(max_iterations).improve(tour, context=context)

  assert improved is not tour, "non-positive budget must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"


def _cupy() -> Any:
  """Return the importable CuPy module, skipping the test when it is unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


def _random_tours(n: int, count: int, seed: int) -> np.ndarray:
  """Return `count` deterministic permutations of `0..n-1`."""
  rng = np.random.default_rng(seed)
  base = np.arange(n, dtype=np.int32)
  return np.stack([rng.permutation(base) for _ in range(count)]).astype(np.int32, copy=False)


def _synthetic_euclidean_problem(n: int, seed: int = 3) -> Problem:
  """Build a coordinate-only `EUC_2D` problem with no precomputed matrix."""
  rng = np.random.default_rng(seed)
  coordinates = rng.integers(0, 10_000, size=(n, 2)).astype(np.float64)
  return Problem(
    name=f"synthetic{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=coordinates,
    distances=None,
    capacity=None,
    demands=None,
  )


def _closed_cost(tour: np.ndarray, distances: np.ndarray) -> int:
  """Return the closed-loop tour cost under the dense matrix."""
  return int(sum(distances[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


@pytest.mark.gpu
@pytest.mark.parametrize("factory", GPU_TWO_OPT_FACTORIES)
def test_gpu_two_opt_coordinate_mode_matches_dense_matrix(
  factory: type[BatchGPUTwoOpt | BatchGPUMultiMoveTwoOpt],
  berlin52_problem: Problem,
) -> None:
  """WHAT: `improve_batch` with the dense matrix versus on-the-fly coordinates.

  WHY: above `max_matrix_dimension` a coordinate-based problem must skip the `(n, n)`
  allocation without changing a single swap decision.

  EXPECTED: byte-identical improved tours, a valid permutation per route, and no
  distance matrix materialized by the coordinate-mode context.
  """
  # Arrange
  cp = _cupy()
  n = berlin52_problem.dimension
  tours = _random_tours(n, 3, seed=7)
  distances = ProblemContext(berlin52_problem, xp=np, seed=0).get_cpu_distances()

  dense_context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  dense_strategy = factory(max_iterations=3)
  dense = dense_strategy.improve_batch(tours, dense_context)
  assert dense_strategy._cached_distances_gpu is not None, "the dense path must upload the matrix"

  coords_context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  coords_strategy = factory(max_iterations=3, max_matrix_dimension=10)
  on_the_fly = coords_strategy.improve_batch(tours, coords_context)

  # Assert
  np.testing.assert_array_equal(np.asarray(dense), np.asarray(on_the_fly))
  assert coords_strategy._cached_distances_gpu is None, "coordinate mode must not upload the matrix"
  assert coords_strategy._cached_coords_gpu is not None, "coordinate mode must upload coordinates"
  for before, after in zip(tours, np.asarray(on_the_fly), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)


@pytest.mark.gpu
@pytest.mark.parametrize("factory", GPU_TWO_OPT_FACTORIES)
def test_gpu_two_opt_tiled_shared_memory_matches_full_tour(
  factory: type[BatchGPUTwoOpt | BatchGPUMultiMoveTwoOpt],
  berlin52_problem: Problem,
) -> None:
  """WHAT: Tiled shared-memory scan versus the full-tour buffer on a small instance.

  WHY: a lowered `max_shared_tour_cities` threshold must route through the
  `s_tour_tile[2049]` path without changing the selected moves (`Task 10.18`).

  EXPECTED: the tiled strategy reports a 2048-wide tile and produces tours identical
  to the full-tour path.
  """
  # Arrange
  cp = _cupy()
  n = berlin52_problem.dimension
  tours = _random_tours(n, 3, seed=13)
  context = ProblemContext(berlin52_problem, xp=cp, seed=2)

  full_tour = factory(max_iterations=3).improve_batch(tours, context)
  tiled_strategy = factory(max_iterations=3, max_shared_tour_cities=4)
  tiled = tiled_strategy.improve_batch(tours, context)

  # Assert
  assert tiled_strategy._tile_size(n) == 2048
  np.testing.assert_array_equal(np.asarray(full_tour), np.asarray(tiled))


@pytest.mark.gpu
@pytest.mark.parametrize("factory", GPU_TWO_OPT_FACTORIES)
def test_gpu_two_opt_tiles_large_tour_beyond_shared_memory(
  factory: type[BatchGPUTwoOpt | BatchGPUMultiMoveTwoOpt],
) -> None:
  """WHAT: Tiled scan over a 2100-city tour, larger than one 2048 tile.

  WHY: the full-tour buffer fails to launch beyond 11,520 cities; with the tile
  threshold crossed the outer loop must reload the window and stay correct across
  tile boundaries.

  EXPECTED: the tiled result is a valid permutation and matches the untiled path
  exactly when both run in coordinate mode.
  """
  # Arrange
  cp = _cupy()
  n = 2100
  problem = _synthetic_euclidean_problem(n)
  context = ProblemContext(problem, xp=cp, seed=3)
  tours = _random_tours(n, 1, seed=9)

  untiled = factory(max_iterations=1, max_matrix_dimension=10, max_shared_tour_cities=10_000).improve_batch(
    tours, context
  )
  tiled_strategy = factory(max_iterations=1, max_matrix_dimension=10, max_shared_tour_cities=2048)
  tiled = tiled_strategy.improve_batch(tours, context)

  # Assert
  assert tiled_strategy._tile_size(n) == 2048
  np.testing.assert_array_equal(np.asarray(untiled), np.asarray(tiled))
  assert sorted(int(gene) for gene in np.asarray(tiled)[0]) == list(range(n))


@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_cpu_two_opt_candidate_k_keeps_tour_valid(
  factory: type[CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  berlin52_problem: Problem,
) -> None:
  """WHAT: every CPU 2-opt variant restricted to a 10-neighbour candidate list.

  WHY: `DP10-20` swaps the `Theta(n^2)` scan for `j = pos[N_k[tour[i], r]]`; a wrong
  inverse map would duplicate or drop genes and silently corrupt the population.

  EXPECTED: a valid permutation, a non-increasing closed cost, and an untouched input.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=5)
  distances = context.get_cpu_distances()
  tour = _random_tours(berlin52_problem.dimension, 1, seed=17)[0]
  original = tour.copy()

  improved = factory(10, candidate_k=10).improve(tour, context=context)

  assert sorted(int(gene) for gene in improved) == list(range(berlin52_problem.dimension))
  assert _closed_cost(improved, distances) <= _closed_cost(original, distances)
  np.testing.assert_array_equal(tour, original)


@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_cpu_two_opt_candidate_k_zero_matches_exhaustive(
  factory: type[CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  berlin52_problem: Problem,
) -> None:
  """WHAT: `candidate_k=0` versus the default `candidate_k=None`.

  WHY: a non-positive budget must disable the restriction rather than select an empty
  neighbour set, which would freeze local search at the initial tour.

  EXPECTED: byte-identical improved tours.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=6)
  tour = _random_tours(berlin52_problem.dimension, 1, seed=21)[0]

  restricted_off = factory(10, candidate_k=0).improve(tour, context=context)
  exhaustive = factory(10).improve(tour, context=context)

  np.testing.assert_array_equal(restricted_off, exhaustive)


@pytest.mark.gpu
def test_gpu_two_opt_candidate_list_path(berlin52_problem: Problem) -> None:
  """WHAT: `BatchGPUTwoOpt` with `candidate_k` on a 52-city instance.

  WHY: the kernel builds `pos[tour[idx]] = idx` per block and looks each neighbour up in
  `N_k`; a bad lookup would produce an invalid tour or a worse cost.

  EXPECTED: valid permutations and a non-increasing closed cost for every route.
  """
  # Arrange
  cp = _cupy()
  n = berlin52_problem.dimension
  tours = _random_tours(n, 3, seed=19)
  context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  distances = ProblemContext(berlin52_problem, xp=np, seed=0).get_cpu_distances()
  strategy = BatchGPUTwoOpt(max_iterations=5, candidate_k=10)

  # Act
  improved = strategy.improve_batch(tours, context)

  # Assert
  assert strategy.candidate_k == 10
  for before, after in zip(tours, np.asarray(improved), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)

