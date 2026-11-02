"""Unit tests for `ProblemContext` CPU accessor contracts.

The context is the single owner of problem data for both Class S (CPU) and Class P
(GPU) algorithms. These tests pin the host-array shapes, the lazy caching contract,
the seed propagation into the context RNG, and the CPU backend label. GPU accessor
guards and device-only parity cases live in `test_unit_problem_context_gpu_guards.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from data_models.problem import Problem


@pytest.mark.parametrize(
  ("fixture_name", "expected_dimension", "expected_edge_type"),
  [
    pytest.param("burma14_problem", 14, "GEO", id="burma14"),
    pytest.param("berlin52_problem", 52, "EUC_2D", id="berlin52"),
  ],
)
def test_cpu_accessors_return_expected_arrays(
  request: pytest.FixtureRequest,
  fixture_name: str,
  expected_dimension: int,
  expected_edge_type: str,
) -> None:
  """WHAT: CPU accessors on a NumPy-backed `ProblemContext`.

  WHY: Class S algorithms read host arrays directly and must never trigger a device
  transfer or receive a wrongly shaped matrix.

  EXPECTED: int32 `(n, n)` distances with a zero diagonal and symmetric values,
  `(n, 2)` coordinates matching the problem, and `None` demands for TSP instances.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=7)

  # Act
  distances = context.get_cpu_distances()
  coordinates = context.get_cpu_coordinates()
  demands = context.get_cpu_demands()

  # Assert
  assert context.xp is np
  assert context.problem is problem
  assert context.dimension == expected_dimension
  assert problem.edge_type == expected_edge_type
  assert context.capacity is None
  assert isinstance(distances, np.ndarray)
  assert distances.shape == (expected_dimension, expected_dimension)
  assert distances.dtype == np.int32
  assert np.all(np.diag(distances) == 0), "distance matrix diagonal must be zero"
  np.testing.assert_array_equal(distances, distances.T, err_msg="TSPLIB distance matrix must be symmetric")
  assert isinstance(coordinates, np.ndarray)
  assert coordinates.shape == (expected_dimension, 2)
  assert problem.coordinates is not None
  np.testing.assert_array_equal(coordinates, problem.coordinates)
  assert demands is None


@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
def test_cpu_accessors_are_lazy_and_cached(
  request: pytest.FixtureRequest,
  fixture_name: str,
) -> None:
  """WHAT: Lazy allocation and caching of the host-side arrays.

  WHY: Eager accessors previously allocated a full distance matrix for every context,
  producing multi-GB memory spikes on large instances.

  EXPECTED: Caches remain `None` until first access, and repeated calls return the
  same object instead of recomputing.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=11)
  assert context._cpu_distances is None
  assert context._cpu_coordinates is None

  # Act
  first_distances = context.get_cpu_distances()
  first_coordinates = context.get_cpu_coordinates()

  # Assert
  assert context.get_cpu_distances() is first_distances
  assert context.get_cpu_coordinates() is first_coordinates
  assert context._cpu_distances is first_distances
  assert context._cpu_coordinates is first_coordinates


def test_seed_controls_context_rng(berlin52_problem: Problem) -> None:
  """WHAT: Seed propagation into the context's private RNG.

  WHY: Parallel workers depend on independent, reproducible random streams; relying on
  the global NumPy seed would couple test and worker order.

  EXPECTED: Equal seeds produce equal draws, different seeds produce different draws.
  """
  # Arrange
  first = ProblemContext(berlin52_problem, xp=np, seed=123)
  second = ProblemContext(berlin52_problem, xp=np, seed=123)
  third = ProblemContext(berlin52_problem, xp=np, seed=124)

  # Act
  first_draw = first.rng.random_sample(8)
  second_draw = second.rng.random_sample(8)
  third_draw = third.rng.random_sample(8)

  # Assert
  assert first.rng is not second.rng
  np.testing.assert_array_equal(first_draw, second_draw)
  assert not np.array_equal(first_draw, third_draw)


@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
def test_repr_reports_cpu_backend(
  request: pytest.FixtureRequest,
  fixture_name: str,
) -> None:
  """WHAT: `repr` of a NumPy-backed context.

  WHY: Logs and failure messages must identify the backend a run actually used.

  EXPECTED: The problem name, dimension, and `CPU (NumPy)` backend label.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=3)

  # Act
  text = repr(context)

  # Assert
  assert problem.name in text
  assert f"N={problem.dimension}" in text
  assert "CPU (NumPy)" in text
