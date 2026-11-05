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
  first_draw = first.rng.random(8)
  second_draw = second.rng.random(8)
  third_draw = third.rng.random(8)

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


@pytest.mark.parametrize(
  ("fixture_name", "k", "expected_columns"),
  [
    pytest.param("berlin52_problem", 20, 20, id="k-below-n"),
    pytest.param("berlin52_problem", 100, 51, id="k-clamped-to-n-minus-1"),
    pytest.param("burma14_problem", 0, 0, id="k-zero"),
  ],
)
def test_cpu_candidate_list_contract(
  request: pytest.FixtureRequest,
  fixture_name: str,
  k: int,
  expected_columns: int,
) -> None:
  """WHAT: `get_cpu_candidate_list(k)` shape, dtype and neighbour invariants.

  WHY: `DP10-20` exposes `N_k` as a shared context service; consumers assume
  `k_eff = min(k, n - 1)` `int32` columns, no self-edge, ascending distances.

  EXPECTED: `(n, k_eff)` `int32`, `u` absent from its own row, distances non-decreasing.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=5)

  # Act
  candidates = context.get_cpu_candidate_list(k)

  # Assert
  assert candidates.shape == (problem.dimension, expected_columns)
  assert candidates.dtype == np.int32
  distances = context.get_cpu_distances()
  for node in range(problem.dimension):
    row = candidates[node]
    assert node not in set(int(neighbor) for neighbor in row)
    row_distances = [int(distances[node, int(neighbor)]) for neighbor in row]
    assert row_distances == sorted(row_distances)


def test_cpu_candidate_list_is_cached_per_k(berlin52_problem: Problem) -> None:
  """WHAT: Cache identity of the `k`-NN candidate lists.

  WHY: rebuilding `N_k` per mutation or per local-search pass would turn the `Theta(n k)`
  speed-up back into `Theta(n^2)` setup cost every generation.

  EXPECTED: repeated calls return the same object, and a different `k` gets its own
  entry with a different width.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=13)

  # Act
  first = context.get_cpu_candidate_list(10)
  again = context.get_cpu_candidate_list(10)
  wider = context.get_cpu_candidate_list(25)

  # Assert
  assert again is first
  assert wider is not first
  assert context._cpu_candidate_lists[10] is first
  assert wider.shape[1] == 25
  assert first.shape[1] == 10


def test_gpu_candidate_list_requires_cupy_backend(berlin52_problem: Problem) -> None:
  """WHAT: `get_gpu_candidate_list` on a NumPy-backed context.

  WHY: silently returning a host array would make a GPU strategy send a NumPy array into
  a CUDA launch; the guard must fail loudly instead.

  EXPECTED: `ValueError` naming the accessor.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=17)

  # Act / Assert
  with pytest.raises(ValueError, match="get_gpu_candidate_list"):
    context.get_gpu_candidate_list(5)

