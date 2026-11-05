"""Unit tests for the device-resident distance matrix cache in `ProblemContext`.

`ProblemContext.get_gpu_distances()` is the single owner of the ``int32`` matrix that
every GPU kernel binds. These tests pin the dtype contract (``BUG-14`` / ``DP10-17``),
the cache identity on repeated calls, and the absence of cross-instance aliasing when
two `Problem` objects share a `name` but carry different matrices.

The numpy-backend guard lives in `test_unit_problem_context_gpu_guards.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from data_models.problem import Problem
from distances.pairwise import get_edge_type_code
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
  """Build a 3-node EXPLICIT TSP from an ``int64`` matrix, exercising the dtype cast."""
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


def _coordinate_problem(name: str) -> Problem:
  """Build a 3-node `EUC_2D` problem whose matrix must be computed on demand."""
  coordinates = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
  return Problem(
    name=name,
    dimension=3,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=coordinates,
    distances=None,
    capacity=None,
    demands=None,
  )


@pytest.mark.gpu
def test_get_gpu_distances_casts_to_int32_and_caches() -> None:
  """WHAT: `get_gpu_distances()` on an ``int64`` EXPLICIT matrix.

  WHY: kernels declare an ``int`` distance pointer; an implicit ``int64`` device array
  widens memory traffic and can be rejected by the compiled signature. The matrix is
  also requested once per algorithm stage, so it must be cached, not re-transferred.

  EXPECTED: a CuPy ``int32`` matrix equal to the source values, and the same object
  returned by a second call.
  """
  cupy: Any = _cupy()

  # Arrange
  context = ProblemContext(_explicit_problem("int32_cast", _FIRST_MATRIX), xp=cupy, seed=3)

  # Act
  first = context.get_gpu_distances()
  second = context.get_gpu_distances()

  # Assert
  assert first is not None
  assert isinstance(first, cupy.ndarray)
  assert first.dtype == cupy.int32
  np.testing.assert_array_equal(cupy.asnumpy(first), _FIRST_MATRIX.astype(np.int32))
  assert second is first, "the device matrix must be cached, not re-transferred"


@pytest.mark.gpu
def test_get_gpu_distances_keeps_same_named_problems_isolated() -> None:
  """WHAT: two `Problem` instances named identically but carrying different matrices.

  WHY: BUG-14 was a GPU cache keyed only by `problem.name`; a synthetic or benchmark
  variant reusing a sanctioned instance name could silently receive the other
  instance's costs.

  EXPECTED: each context returns its own matrix, so the per-instance cache never
  leaks values across two same-named problems.
  """
  cupy: Any = _cupy()

  # Arrange
  context_first = ProblemContext(_explicit_problem("same_name", _FIRST_MATRIX), xp=cupy, seed=11)
  context_second = ProblemContext(_explicit_problem("same_name", _SECOND_MATRIX), xp=cupy, seed=11)

  # Act
  matrix_first = context_first.get_gpu_distances()
  matrix_second = context_second.get_gpu_distances()

  # Assert
  assert matrix_first is not None
  assert matrix_second is not None
  np.testing.assert_array_equal(cupy.asnumpy(matrix_first), _FIRST_MATRIX.astype(np.int32))
  np.testing.assert_array_equal(cupy.asnumpy(matrix_second), _SECOND_MATRIX.astype(np.int32))
  assert not np.array_equal(cupy.asnumpy(matrix_first), cupy.asnumpy(matrix_second))


@pytest.mark.gpu
def test_get_gpu_distance_operands_reuses_the_context_matrix() -> None:
  """WHAT: `get_gpu_distance_operands` on a context created with `xp=np`.

  WHY: `DP10-17` lets GPU strategies request operands from a NumPy-preferred context;
  returning a private copy would double the `(n, n)` allocation in VRAM.

  EXPECTED: `(matrix, 0, 0)` where `matrix` is the very object `get_gpu_distances()`
  caches, in ``int32``.
  """
  cupy: Any = _cupy()

  # Arrange
  context = ProblemContext(_explicit_problem("operands_matrix", _FIRST_MATRIX), xp=np, seed=4)

  # Act
  d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(3)

  # Assert
  assert isinstance(d_distances, cupy.ndarray)
  assert d_distances.dtype == cupy.int32
  assert d_distances is context.get_gpu_distances(), "operands must reuse the context matrix"
  assert d_coords == 0
  assert edge_type_code == 0


@pytest.mark.gpu
def test_get_gpu_distance_operands_switches_to_coordinates() -> None:
  """WHAT: `get_gpu_distance_operands` above the matrix threshold.

  WHY: coordinate mode must hand the kernel a `nullptr` matrix plus the context-owned
  coordinates, so a large instance never allocates `(n, n)` in VRAM (`DP10-19`).

  EXPECTED: `(0, coords, EUC_2D code)` with `coords is context.get_gpu_coordinates()`
  and no distance matrix cached.
  """
  cupy: Any = _cupy()

  # Arrange
  context = ProblemContext(_coordinate_problem("operands_coords"), xp=np, seed=4, max_matrix_dimension=0)

  # Act
  d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(3)

  # Assert
  assert d_distances == 0
  assert isinstance(d_coords, cupy.ndarray)
  assert d_coords is context.get_gpu_coordinates()
  assert edge_type_code == get_edge_type_code("EUC_2D")
  assert context._gpu_distances is None, "coordinate mode must not materialize the matrix"
