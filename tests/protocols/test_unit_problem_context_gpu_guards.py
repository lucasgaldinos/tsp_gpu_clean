"""Unit tests for the `ProblemContext` GPU accessor guards.

The numpy-backend guard must fail fast without requiring a device; the device-marked
parity case is skipped when CUDA is absent. CPU accessor contracts live in
`test_unit_problem_context.py`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from protocols.backend import cupy_backend, gpu_available
from protocols.problem_context import ProblemContext

GPU_AVAILABLE = gpu_available()
cp = cupy_backend()

if TYPE_CHECKING:
  from data_models.problem import Problem


@pytest.mark.parametrize("method_name", ["get_gpu_distances", "get_gpu_coordinates", "get_gpu_demands"])
@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
def test_gpu_accessors_raise_value_error_on_numpy_backend(
  request: pytest.FixtureRequest,
  fixture_name: str,
  method_name: str,
) -> None:
  """WHAT: GPU-only accessors invoked on a NumPy backend.

  WHY: The guard must fail fast with an actionable message instead of returning a host
  array or raising an obscure attribute error.

  EXPECTED: `ValueError` naming the accessor and stating that CuPy is required.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=5)
  expected_message = f"{method_name}() requires xp to be CuPy, not NumPy"

  # Act / Assert
  with pytest.raises(ValueError, match=re.escape(expected_message)):
    getattr(context, method_name)()


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA GPU not available")
@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
@pytest.mark.parametrize("cpu_first", [False, True], ids=["gpu-first", "cpu-first"])
def test_gpu_accessors_match_cpu_backend(
  request: pytest.FixtureRequest,
  fixture_name: str,
  cpu_first: bool,
) -> None:
  """WHAT: GPU accessors on a real device, exercising both transfer orders.

  WHY: Lazy transfer may start from the problem data or from an existing CPU cache;
  both paths must produce exactly the CPU matrix and coordinates.

  EXPECTED: CuPy arrays with CPU-identical values and shapes, `None` demands for TSP
  fixtures, and cached device arrays on repeated calls.
  """
  if cp is None:
    pytest.skip("CuPy is not importable")

  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=cp, seed=9)

  if cpu_first:
    cpu_distances = context.get_cpu_distances()
    cpu_coordinates = context.get_cpu_coordinates()

  # Act
  gpu_distances = context.get_gpu_distances()
  gpu_coordinates = context.get_gpu_coordinates()
  gpu_demands = context.get_gpu_demands()

  if not cpu_first:
    cpu_distances = context.get_cpu_distances()
    cpu_coordinates = context.get_cpu_coordinates()

  # Assert
  assert isinstance(gpu_distances, cp.ndarray)
  assert isinstance(gpu_coordinates, cp.ndarray)
  assert gpu_distances.dtype == cp.int32
  assert gpu_distances.shape == (problem.dimension, problem.dimension)
  assert gpu_coordinates.shape == (problem.dimension, 2)
  assert cpu_coordinates is not None
  np.testing.assert_array_equal(cp.asnumpy(gpu_distances), cpu_distances)
  np.testing.assert_array_equal(cp.asnumpy(gpu_coordinates), cpu_coordinates)
  assert gpu_demands is None
  assert isinstance(cpu_distances, np.ndarray)
  assert not isinstance(cpu_distances, cp.ndarray)
  assert context.get_gpu_distances() is gpu_distances
