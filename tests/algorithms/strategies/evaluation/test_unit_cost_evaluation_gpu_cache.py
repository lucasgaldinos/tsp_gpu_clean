"""Unit tests for the GPU distance cache in `EvaluationGPUCost`.

The evaluator keeps its own device matrix when no `ProblemContext` is supplied. That
fallback cache is keyed by the full problem identity ``(name, dimension, edge_type,
id(problem))`` (``BUG-14`` / ``DP10-17``), so two instances sharing a name cannot alias
each other's costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.evaluation.cost_evaluation import EvaluationGPUCost
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext

cp = cupy_backend()

_FIRST_MATRIX: np.ndarray = np.array(
  [
    [0, 2, 9],
    [2, 0, 6],
    [9, 6, 0],
  ],
  dtype=np.int64,
)
_SECOND_MATRIX: np.ndarray = np.array(
  [
    [0, 7, 1],
    [7, 0, 8],
    [1, 8, 0],
  ],
  dtype=np.int64,
)


def _explicit_problem(name: str, distances: np.ndarray) -> Problem:
  """Build a 3-node EXPLICIT TSP whose matrix distinguishes it from its same-named twin."""
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


def _cupy() -> Any:
  """Return the importable CuPy module, skipping the test when it is unavailable."""
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


@pytest.mark.gpu
def test_fallback_cache_invalidates_on_same_named_problem_swap() -> None:
  """WHAT: `_get_distances_int32_gpu(problem=...)` called with two same-named problems.

  WHY: the pre-fix cache compared only `problem.name`, so the second problem silently
  reused the first one's device matrix and evaluated a different instance's costs.

  EXPECTED: the second call uploads its own matrix (values differ) and both payloads are
  ``int32``; repeating the call for the same object returns the cached array.
  """
  cupy: Any = _cupy()

  # Arrange
  evaluator = EvaluationGPUCost()
  problem_first = _explicit_problem("same_name", _FIRST_MATRIX)
  problem_second = _explicit_problem("same_name", _SECOND_MATRIX)

  # Act
  device_first = evaluator._get_distances_int32_gpu(problem=problem_first)
  device_second = evaluator._get_distances_int32_gpu(problem=problem_second)
  device_second_again = evaluator._get_distances_int32_gpu(problem=problem_second)

  # Assert
  assert device_first.dtype == cupy.int32
  assert device_second.dtype == cupy.int32
  np.testing.assert_array_equal(cupy.asnumpy(device_first), _FIRST_MATRIX.astype(np.int32))
  np.testing.assert_array_equal(cupy.asnumpy(device_second), _SECOND_MATRIX.astype(np.int32))
  assert device_second_again is device_second, "same problem object must reuse the cached matrix"


@pytest.mark.gpu
def test_context_takes_precedence_over_the_fallback_cache() -> None:
  """WHAT: `_get_distances_int32_gpu(context=...)` with a context supplied.

  WHY: the context owns one matrix per problem instance; the strategy-local fallback
  must not shadow it, or two call sites could disagree about the matrix in VRAM.

  EXPECTED: the returned array is exactly `context.get_gpu_distances()`.
  """
  cupy: Any = _cupy()

  # Arrange
  evaluator = EvaluationGPUCost()
  context = ProblemContext(_explicit_problem("context_owned", _FIRST_MATRIX), xp=cupy, seed=2)

  # Act
  device_distances = evaluator._get_distances_int32_gpu(context=context)
  context_distances = context.get_gpu_distances()

  # Assert
  assert context_distances is not None
  assert device_distances is context_distances
  assert context_distances.dtype == cupy.int32
