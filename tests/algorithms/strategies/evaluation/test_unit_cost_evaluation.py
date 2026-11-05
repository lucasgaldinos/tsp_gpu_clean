"""Unit tests for `EvaluationCPUCost`.

Tour cost drives every selection and survival decision in the GA. The tests pin the
closed-loop cost on a hand-computable 4x4 EXPLICIT matrix, the agreement between the
single and batch APIs, the shape contract of a batch call on one route, and the
absence of in-place side effects on the population.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost, EvaluationGPUCost
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext

TINY_DISTANCES: np.ndarray = np.array(
  [
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 3],
    [10, 4, 3, 0],
  ],
  dtype=np.int32,
)


def _tiny_explicit_problem() -> Problem:
  """Build a 4-node EXPLICIT TSP whose closed-tour costs are hand-computable."""
  return Problem(
    name="tiny4",
    dimension=4,
    problem_type="TSP",
    edge_type="EXPLICIT",
    coordinates=None,
    distances=TINY_DISTANCES,
    capacity=None,
    demands=None,
  )


@pytest.mark.parametrize(
  ("tour", "expected_cost"),
  [
    pytest.param([0, 1, 2, 3], 21.0, id="ascending"),
    pytest.param([0, 2, 1, 3], 29.0, id="middle-swap"),
    pytest.param([1, 3, 2, 0], 18.0, id="rotated"),
    pytest.param([0], 0.0, id="single-gene"),
  ],
)
def test_cpu_cost_evaluation_matches_hand_computed_cost(
  tour: list[int],
  expected_cost: float,
) -> None:
  """WHAT: Closed-loop tour cost on a known 4x4 distance matrix.

  WHY: Cost drives every selection and survival decision; a wrong wrap-around edge
  silently biases the whole search.

  EXPECTED: The exact sum of the `n` tour edges, including the last-to-first edge.
  """
  # Arrange
  context = ProblemContext(_tiny_explicit_problem(), xp=np, seed=0)
  evaluator = EvaluationCPUCost()

  # Act
  cost = evaluator.evaluate(np.asarray(tour, dtype=np.int32), context)

  # Assert
  assert cost == pytest.approx(expected_cost)


def test_cpu_cost_evaluation_batch_matches_single_tours() -> None:
  """WHAT: Batch evaluation against per-tour evaluation.

  WHY: The GA evaluates whole populations; batch and single APIs must agree exactly.

  EXPECTED: float64 costs of shape `(N,)` matching the hand-computed values.
  """
  # Arrange
  context = ProblemContext(_tiny_explicit_problem(), xp=np, seed=0)
  evaluator = EvaluationCPUCost()
  tours = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [1, 3, 2, 0]], dtype=np.int32)
  expected = np.array([21.0, 29.0, 18.0], dtype=np.float64)

  # Act
  costs = evaluator.evaluate_batch(tours, context)

  # Assert
  assert costs.shape == (3,)
  assert costs.dtype == np.float64
  for cost, expected_cost in zip(costs, expected, strict=True):
    assert cost == pytest.approx(expected_cost)
  for tour, expected_cost in zip(tours, expected, strict=True):
    assert evaluator.evaluate(tour, context) == pytest.approx(expected_cost)


def test_cpu_cost_evaluation_single_route_batch_shape() -> None:
  """WHAT: `evaluate_batch` applied to a single 1D route.

  WHY: Callers may pass one route to the batch API; it must still return a 1D cost
  array rather than a scalar.

  EXPECTED: Shape `(1,)` with the tour's closed-loop cost.
  """
  # Arrange
  context = ProblemContext(_tiny_explicit_problem(), xp=np, seed=0)
  evaluator = EvaluationCPUCost()

  # Act
  costs = evaluator.evaluate_batch(np.array([0, 1, 2, 3], dtype=np.int32), context)

  # Assert
  assert costs.shape == (1,)
  assert costs[0] == pytest.approx(21.0)


def test_cpu_cost_evaluation_does_not_mutate_routes() -> None:
  """WHAT: Side-effect freedom of cost evaluation.

  WHY: Evaluation runs after every generation; mutating the population in place would
  corrupt later crossover and elite preservation.

  EXPECTED: The input batch is byte-for-byte unchanged.
  """
  # Arrange
  context = ProblemContext(_tiny_explicit_problem(), xp=np, seed=0)
  evaluator = EvaluationCPUCost()
  tours = np.array([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=np.int32)
  original = tours.copy()

  # Act
  evaluator.evaluate_batch(tours, context)

  # Assert
  np.testing.assert_array_equal(tours, original)


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


@pytest.mark.gpu
@pytest.mark.parametrize("fixture_name", ["berlin52_problem", "burma14_problem"])
def test_gpu_cost_evaluation_coordinate_mode_matches_dense_matrix(
  fixture_name: str,
  request: pytest.FixtureRequest,
) -> None:
  """WHAT: `EvaluationGPUCost` with the dense matrix versus on-the-fly coordinates.

  WHY: above `max_matrix_dimension` a coordinate-based problem must not allocate the
  `(n, n)` matrix, yet its costs must stay bit-identical to the matrix path.

  EXPECTED: identical int64 costs, and the coordinate-mode context never materializes
  the CPU or GPU distance matrix.
  """
  # Arrange
  cp = _cupy()
  problem: Problem = request.getfixturevalue(fixture_name)
  tours = _random_tours(problem.dimension, 4, seed=11)

  dense_context = ProblemContext(problem, xp=cp, seed=0)
  dense_costs = EvaluationGPUCost().evaluate_batch(tours, dense_context)
  assert dense_context._gpu_distances is not None, "the dense path must build the matrix"

  coords_context = ProblemContext(problem, xp=cp, seed=0)
  coords_costs = EvaluationGPUCost(max_matrix_dimension=10).evaluate_batch(tours, coords_context)

  # Assert
  np.testing.assert_array_equal(np.asarray(dense_costs), np.asarray(coords_costs))
  assert coords_context._gpu_distances is None, "coordinate mode must not allocate the dense matrix"
  assert coords_context._cpu_distances is None, "coordinate mode must not build the CPU matrix"
  expected = EvaluationCPUCost().evaluate_batch(tours, ProblemContext(problem, xp=np, seed=0))
  np.testing.assert_array_equal(np.asarray(coords_costs), expected)
