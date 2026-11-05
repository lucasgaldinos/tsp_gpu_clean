"""Boundary contracts for the unified 2-opt strategies in `two_opt.py`.

`LocalSearchCPUTwoOpt` covers the `best` / `first` / `multi_move` acceptance rules and
`LocalSearchGPUTwoOpt` covers the `batch` / `multi_move` / `naive` kernel families, so
every mode is exercised through the same constructor shape (`DP11-1`).

Each mode must be a no-op for tours shorter than one 2-opt move and for a zero
iteration budget, still returning a defensive copy so caller-owned routes are never
aliased, and each must reject a negative budget or an unknown mode before doing work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.strategies import LocalSearchCPUTwoOpt, LocalSearchGPUTwoOpt
from algorithms.strategies.evaluation.cost_evaluation import EvaluationGPUCost
from data_models.problem import Problem
from protocols import ProblemContext
from protocols.backend import cupy_backend

if TYPE_CHECKING:
  from algorithms.strategies.local_search import TwoOptCPUMode, TwoOptGPUMode

CPU_TWO_OPT_MODES: list[TwoOptCPUMode] = ["best", "first", "multi_move"]
GPU_TWO_OPT_MODES: list[TwoOptGPUMode] = ["batch", "multi_move"]


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_short_tour_returns_equal_distinct_copy(mode: TwoOptCPUMode, burma14_problem: Problem) -> None:
  """WHAT: A three-city tour through every CPU acceptance rule.

  WHY: A 2-opt move needs four distinct positions; shorter tours must be returned
  untouched, but still as independent memory so callers cannot alias the result.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.array([0, 2, 1], dtype=np.int32)
  original = tour.copy()

  improved = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode).improve(tour, context=context)

  assert improved is not tour, "boundary path must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_zero_iteration_budget_returns_equal_distinct_copy(mode: TwoOptCPUMode, burma14_problem: Problem) -> None:
  """WHAT: A full burma14 tour evaluated with `max_iterations=0`.

  WHY: A zero budget disables improvement; returning the input itself would let a later
  in-place stage corrupt the caller's population.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.arange(burma14_problem.dimension, dtype=np.int32)
  original = tour.copy()

  improved = LocalSearchCPUTwoOpt(max_iterations=0, mode=mode).improve(tour, context=context)

  assert improved is not tour, "zero budget must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"


def test_cpu_two_opt_rejects_unknown_mode() -> None:
  """WHAT: `LocalSearchCPUTwoOpt(mode="steepest")`.

  WHY: the mode string is configuration data; a typo must fail at construction time
  instead of silently running a different acceptance rule.

  EXPECTED: `ValueError` naming the valid modes.
  """
  with pytest.raises(ValueError, match="Invalid 2-opt CPU mode"):
    LocalSearchCPUTwoOpt(mode="steepest")  # type: ignore[arg-type]


def test_gpu_two_opt_rejects_unknown_mode() -> None:
  """WHAT: `LocalSearchGPUTwoOpt(mode="tiled")`.

  WHY: an unknown kernel family would otherwise compile the `batch` kernel and report
  results attributed to a kernel that never ran.

  EXPECTED: `ValueError` naming the valid modes.
  """
  with pytest.raises(ValueError, match="Invalid 2-opt GPU mode"):
    LocalSearchGPUTwoOpt(mode="tiled")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", [*CPU_TWO_OPT_MODES])
def test_negative_iteration_budget_is_rejected(mode: TwoOptCPUMode) -> None:
  """WHAT: `max_iterations=-1` for every CPU acceptance rule.

  WHY: a negative budget has no meaning and used to be silently treated as "disabled";
  rejecting it keeps the configuration honest.

  EXPECTED: `ValueError` at construction time.
  """
  with pytest.raises(ValueError, match="max_iterations must be >= 0"):
    LocalSearchCPUTwoOpt(max_iterations=-1, mode=mode)


@pytest.mark.parametrize("mode", GPU_TWO_OPT_MODES)
def test_gpu_negative_iteration_budget_is_rejected(mode: TwoOptGPUMode) -> None:
  """WHAT: `max_iterations=-1` for every GPU kernel family.

  WHY: same contract as the CPU class, so a config cannot depend on which backend runs.

  EXPECTED: `ValueError` at construction time.
  """
  with pytest.raises(ValueError, match="max_iterations must be >= 0"):
    LocalSearchGPUTwoOpt(max_iterations=-1, mode=mode)


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
@pytest.mark.parametrize("mode", GPU_TWO_OPT_MODES)
def test_gpu_two_opt_coordinate_mode_matches_dense_matrix(mode: TwoOptGPUMode, berlin52_problem: Problem) -> None:
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
  dense_strategy = LocalSearchGPUTwoOpt(max_iterations=3, mode=mode)
  dense = dense_strategy.improve_batch(tours, dense_context)
  assert dense_strategy._operands.distances is not None, "the dense path must bind the matrix"

  coords_context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  coords_strategy = LocalSearchGPUTwoOpt(max_iterations=3, mode=mode, max_matrix_dimension=10)
  on_the_fly = coords_strategy.improve_batch(tours, coords_context)

  # Assert
  np.testing.assert_array_equal(np.asarray(dense), np.asarray(on_the_fly))
  assert coords_strategy._operands.distances is None, "coordinate mode must not upload the matrix"
  assert coords_strategy._operands.coords is not None, "coordinate mode must bind coordinates"
  for before, after in zip(tours, np.asarray(on_the_fly), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", GPU_TWO_OPT_MODES)
def test_gpu_two_opt_tiled_shared_memory_matches_full_tour(mode: TwoOptGPUMode, berlin52_problem: Problem) -> None:
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

  full_tour = LocalSearchGPUTwoOpt(max_iterations=3, mode=mode).improve_batch(tours, context)
  tiled_strategy = LocalSearchGPUTwoOpt(max_iterations=3, mode=mode, max_shared_tour_cities=4)
  tiled = tiled_strategy.improve_batch(tours, context)

  # Assert
  assert tiled_strategy._tile_size(n) == 2048
  np.testing.assert_array_equal(np.asarray(full_tour), np.asarray(tiled))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", GPU_TWO_OPT_MODES)
def test_gpu_two_opt_tiles_large_tour_beyond_shared_memory(mode: TwoOptGPUMode) -> None:
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

  untiled = LocalSearchGPUTwoOpt(
    max_iterations=1, mode=mode, max_matrix_dimension=10, max_shared_tour_cities=10_000
  ).improve_batch(tours, context)
  tiled_strategy = LocalSearchGPUTwoOpt(
    max_iterations=1, mode=mode, max_matrix_dimension=10, max_shared_tour_cities=2048
  )
  tiled = tiled_strategy.improve_batch(tours, context)

  # Assert
  assert tiled_strategy._tile_size(n) == 2048
  np.testing.assert_array_equal(np.asarray(untiled), np.asarray(tiled))
  assert sorted(int(gene) for gene in np.asarray(tiled)[0]) == list(range(n))


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_cpu_two_opt_candidate_k_keeps_tour_valid(mode: TwoOptCPUMode, berlin52_problem: Problem) -> None:
  """WHAT: every CPU acceptance rule restricted to a 10-neighbour candidate list.

  WHY: `DP10-20` swaps the `Theta(n^2)` scan for `j = pos[N_k[tour[i], r]]`; a wrong
  inverse map would duplicate or drop genes and silently corrupt the population.

  EXPECTED: a valid permutation, a non-increasing closed cost, and an untouched input.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=5)
  distances = context.get_cpu_distances()
  tour = _random_tours(berlin52_problem.dimension, 1, seed=17)[0]
  original = tour.copy()

  improved = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode, candidate_k=10).improve(tour, context=context)

  assert sorted(int(gene) for gene in improved) == list(range(berlin52_problem.dimension))
  assert _closed_cost(improved, distances) <= _closed_cost(original, distances)
  np.testing.assert_array_equal(tour, original)


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_cpu_two_opt_candidate_k_zero_matches_exhaustive(mode: TwoOptCPUMode, berlin52_problem: Problem) -> None:
  """WHAT: `candidate_k=0` versus the default `candidate_k=None`.

  WHY: a non-positive budget must disable the restriction rather than select an empty
  neighbour set, which would freeze local search at the initial tour.

  EXPECTED: byte-identical improved tours.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=6)
  tour = _random_tours(berlin52_problem.dimension, 1, seed=21)[0]

  restricted_off = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode, candidate_k=0).improve(tour, context=context)
  exhaustive = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode).improve(tour, context=context)

  np.testing.assert_array_equal(restricted_off, exhaustive)


@pytest.mark.gpu
def test_gpu_two_opt_candidate_list_path(berlin52_problem: Problem) -> None:
  """WHAT: `LocalSearchGPUTwoOpt(mode="batch")` with `candidate_k` on a 52-city instance.

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
  strategy = LocalSearchGPUTwoOpt(max_iterations=5, mode="batch", candidate_k=10)

  # Act
  improved = strategy.improve_batch(tours, context)

  # Assert
  assert strategy.candidate_k == 10
  for before, after in zip(tours, np.asarray(improved), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)


@pytest.mark.gpu
def test_multi_move_gpu_candidate_list_path(berlin52_problem: Problem) -> None:
  """WHAT: `LocalSearchGPUTwoOpt(mode="multi_move")` with `candidate_k` on a 52-city instance.

  WHY: the multi-move kernel gained the same `N_k` scan as `two_opt_batch.cu`
  (`DP10-20`); a bad inverse map would corrupt the tour or report a worse cost.

  EXPECTED: valid permutations and a non-increasing closed cost for every route.
  """
  # Arrange
  cp = _cupy()
  n = berlin52_problem.dimension
  tours = _random_tours(n, 3, seed=23)
  context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  distances = ProblemContext(berlin52_problem, xp=np, seed=0).get_cpu_distances()
  strategy = LocalSearchGPUTwoOpt(max_iterations=5, mode="multi_move", candidate_k=10)

  # Act
  improved = strategy.improve_batch(tours, context)

  # Assert
  assert strategy.candidate_k == 10
  for before, after in zip(tours, np.asarray(improved), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)


@pytest.mark.gpu
def test_naive_mode_keeps_one_launch_per_route_and_iteration(berlin52_problem: Problem) -> None:
  """WHAT: `LocalSearchGPUTwoOpt(mode="naive")` on a three-route batch.

  WHY: the naive family is the Chapter 4 transfer baseline (`two_opt_single.cu`); it must
  still transfer each route individually and launch one kernel per route per iteration,
  which is exactly the cost the `batch` mode removes.

  EXPECTED: `3 routes x 2 iterations = 6` launches, versus 2 for the grid mode, valid
  permutations and a non-increasing closed cost.
  """
  # Arrange
  cp = _cupy()
  n = berlin52_problem.dimension
  tours = _random_tours(n, 3, seed=29)
  context = ProblemContext(berlin52_problem, xp=cp, seed=1)
  distances = ProblemContext(berlin52_problem, xp=np, seed=0).get_cpu_distances()
  naive = LocalSearchGPUTwoOpt(max_iterations=2, mode="naive")
  grid = LocalSearchGPUTwoOpt(max_iterations=2, mode="batch")

  # Act
  naive_improved = naive.improve_batch(tours, context)
  grid.improve_batch(tours, context)

  # Assert
  assert naive.kernel_launches == tours.shape[0] * 2
  assert grid.kernel_launches == 2
  for before, after in zip(tours, np.asarray(naive_improved), strict=True):
    assert sorted(int(gene) for gene in after) == list(range(n))
    assert _closed_cost(after, distances) <= _closed_cost(before, distances)


def _explicit_problem(name: str, distances: np.ndarray) -> Problem:
  """Build a small EXPLICIT TSP whose matrix distinguishes it from a same-named twin."""
  return Problem(
    name=name,
    dimension=distances.shape[0],
    problem_type="TSP",
    edge_type="EXPLICIT",
    coordinates=None,
    distances=distances,
    capacity=None,
    demands=None,
  )


#: `int64` matrices whose ``int32`` casts must be distinguishable (4-city: 2-opt needs n >= 4).
_FIRST_MATRIX: np.ndarray = np.array([[0, 2, 9, 4], [2, 0, 6, 3], [9, 6, 0, 5], [4, 3, 5, 0]], dtype=np.int64)
_SECOND_MATRIX: np.ndarray = np.array([[0, 7, 1, 8], [7, 0, 8, 2], [1, 8, 0, 6], [8, 2, 6, 0]], dtype=np.int64)
#: Sparse CVRP sub-route: node ids reach 19 while the route holds 6 stops (`BUG-19`).
_CVRP_SUB_ROUTE: np.ndarray = np.array([0, 5, 12, 19, 8, 0], dtype=np.int32)


@pytest.mark.gpu
def test_gpu_strategies_share_the_context_distance_matrix() -> None:
  """WHAT: `EvaluationGPUCost` and both grid 2-opt modes on one context.

  WHY: `DP10-17`/`BUG-14` require a single `(n, n)` matrix in VRAM; each strategy used to
  upload its own copy, tripling the footprint and letting the stages disagree. The
  context here is created with the default `xp=np`, i.e. a hybrid CPU+GPU pipeline.

  EXPECTED: all three strategies bind the very object `context.get_gpu_distances()`
  caches, and the context holds exactly one matrix.
  """
  # Arrange
  cupy: Any = _cupy()
  problem = _explicit_problem("shared_matrix", _FIRST_MATRIX)
  context = ProblemContext(problem, xp=np, seed=1)
  tours = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)

  # Act
  evaluator = EvaluationGPUCost()
  evaluator.evaluate_batch(tours, context)
  batch_strategy = LocalSearchGPUTwoOpt(max_iterations=1, mode="batch")
  batch_strategy.improve_batch(tours, context)
  multi_move_strategy = LocalSearchGPUTwoOpt(max_iterations=1, mode="multi_move", candidate_k=2)
  multi_move_strategy.improve_batch(tours, context)

  # Assert
  context_matrix = context.get_gpu_distances()
  assert context_matrix is not None
  assert isinstance(context_matrix, cupy.ndarray)
  assert evaluator._cached_distances_gpu is context_matrix
  assert batch_strategy._operands.distances is context_matrix
  assert multi_move_strategy._operands.distances is context_matrix


@pytest.mark.gpu
def test_gpu_two_opt_keeps_same_named_problems_isolated() -> None:
  """WHAT: one strategy fed two synthetic `Problem`s with the same `name`.

  WHY: `BUG-14` was a cache keyed only by `problem.name`, so the second instance reused
  the first one's device matrix and optimized the wrong cost function.

  EXPECTED: the strategy rebinds to each context's own matrix, never the twin's.
  """
  # Arrange
  cupy: Any = _cupy()
  first = ProblemContext(_explicit_problem("same_name", _FIRST_MATRIX), xp=np, seed=2)
  second = ProblemContext(_explicit_problem("same_name", _SECOND_MATRIX), xp=np, seed=2)
  strategy = LocalSearchGPUTwoOpt(max_iterations=1, mode="batch")
  tour = np.array([[0, 1, 2, 3]], dtype=np.int32)

  # Act
  strategy.improve_batch(tour, first)
  matrix_first = strategy._operands.distances
  strategy.improve_batch(tour, second)
  matrix_second = strategy._operands.distances

  # Assert
  assert matrix_first is not matrix_second
  np.testing.assert_array_equal(cupy.asnumpy(matrix_first), _FIRST_MATRIX.astype(np.int32))
  np.testing.assert_array_equal(cupy.asnumpy(matrix_second), _SECOND_MATRIX.astype(np.int32))


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_cpu_two_opt_candidate_k_supports_sparse_cvrp_sub_route(mode: TwoOptCPUMode, berlin52_problem: Problem) -> None:
  """WHAT: CPU 2-opt with `candidate_k=3` on `[0, 5, 12, 19, 8, 0]` (max id 19, 6 stops).

  WHY: the inverse map used to be allocated with `len(route)` slots, so a CVRP sub-route
  whose node ids exceed its length raised `IndexError` (`BUG-19`).

  EXPECTED: the gene multiset survives and no exception escapes; 2-opt still treats the
  route cyclically, so only the multiset is asserted.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=8)
  route = _CVRP_SUB_ROUTE.copy()

  # Act
  improved = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode, candidate_k=3).improve(route, context=context)

  # Assert
  assert sorted(int(gene) for gene in improved) == sorted(int(gene) for gene in route)
  np.testing.assert_array_equal(route, _CVRP_SUB_ROUTE)


@pytest.mark.gpu
def test_multi_move_gpu_candidate_k_supports_sparse_sub_route(berlin52_problem: Problem) -> None:
  """WHAT: `LocalSearchGPUTwoOpt(mode="multi_move", candidate_k=4)` on the sparse CVRP sub-route.

  WHY: the kernel candidate path indexes its `positions` buffer by node id, so a route
  whose ids exceed its width must fall back to the exhaustive scan instead of writing
  outside the buffer (`BUG-19`).

  EXPECTED: the route ends as a valid multiset of the same node ids and the launch
  succeeds (2-opt treats the route cyclically, so positions may rotate).
  """
  # Arrange
  cp = _cupy()
  context = ProblemContext(berlin52_problem, xp=cp, seed=4)
  strategy = LocalSearchGPUTwoOpt(max_iterations=2, mode="multi_move", candidate_k=4)

  # Act
  improved = strategy.improve_batch(_CVRP_SUB_ROUTE.copy(), context)

  # Assert
  assert strategy.candidate_k == 4
  result = np.asarray(improved)
  assert sorted(int(gene) for gene in result) == sorted(int(gene) for gene in _CVRP_SUB_ROUTE)
