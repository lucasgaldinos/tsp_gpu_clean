"""Unit tests for `MutationSwap`.

Swap mutation exchanges two positions of a route. The tests pin the invariants the GA
depends on: the permutation and its dtype survive, the parent array is never mutated
in place, a zero rate is an identity, and batch input keeps its dimensionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies.mutation.swap_mutation import MutationSwap
from protocols.problem_context import ProblemContext
from utils.route_helpers import select_proximity_index

if TYPE_CHECKING:
  from collections.abc import Callable

  from data_models.problem import Problem


@pytest.mark.parametrize("mutation_rate", [0.0, 1.0])
@pytest.mark.parametrize("n_genes", [1, 2, 5, 9])
def test_mutation_preserves_permutation(mutation_rate: float, n_genes: int) -> None:
  """WHAT: Swap mutation across rates and route sizes.

  WHY: Mutation must explore without breaking the permutation or mutating the parent
  array that the population still references.

  EXPECTED: Same shape and int32 dtype, same gene multiset, untouched input, and an
  identity result at rate 0.0.
  """
  # Arrange
  rng = np.random.default_rng(200 + n_genes)
  route = rng.permutation(np.arange(n_genes, dtype=np.int32)).astype(np.int32, copy=False)
  original = route.copy()

  # Act
  mutated = MutationSwap(seed=3).mutate(route, mutation_rate=mutation_rate)

  # Assert
  assert mutated.shape == route.shape
  assert mutated.dtype == np.int32
  assert sorted(mutated.tolist()) == sorted(route.tolist())
  np.testing.assert_array_equal(route, original)
  if mutation_rate == 0.0:
    np.testing.assert_array_equal(mutated, route)


@pytest.mark.parametrize("as_batch", [False, True], ids=["single-route", "batch"])
def test_mutation_batch_preserves_shape_and_permutations(
  as_batch: bool,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: `mutate_batch` on a 1D route and on a 2D population.

  WHY: Batch mutation is the path used by the GA pipeline; shape restoration must not
  flatten or wrap a single route.

  EXPECTED: Output keeps the input dimensionality and every row stays a permutation.
  """
  # Arrange
  routes = make_population(3, 6, seed=31)
  input_routes = routes if as_batch else routes[0]

  # Act
  mutated = MutationSwap(seed=13).mutate_batch(input_routes, mutation_rate=1.0)

  # Assert
  assert mutated.ndim == input_routes.ndim
  rows = mutated if as_batch else mutated[np.newaxis, :]
  assert rows.shape[1] == 6
  for row in rows:
    assert sorted(row.tolist()) == list(range(6))


def test_mutation_preserves_three_node_depot_endpoints() -> None:
  """WHAT: Mutation on the shortest depot-bounded route.

  WHY: Depot endpoints delimit vehicle routes; moving them turns a valid tour into an
  invalid chromosome for the constructor.

  EXPECTED: Both endpoints remain at the depot.
  """
  # Arrange
  route = np.array([0, 1, 0], dtype=np.int32)

  # Act
  mutated = MutationSwap(seed=0).mutate(route, mutation_rate=1.0)

  # Assert
  assert mutated[0] == 0
  assert mutated[-1] == 0
  assert sorted(mutated.tolist()) == sorted(route.tolist())


def test_candidate_k_swap_keeps_permutation_and_depot(berlin52_problem: Problem) -> None:
  """WHAT: `MutationSwap(candidate_k=10)` on a full 52-city tour.

  WHY: proximity-guided mutation changes how `idx2` is drawn; it must still exchange two
  interior positions and leave every gene exactly once (`DP10-20`).

  EXPECTED: valid permutation, depot endpoints untouched, parent array not mutated.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=9)
  route = np.random.default_rng(11).permutation(np.arange(52, dtype=np.int32)).astype(np.int32, copy=False)
  original = route.copy()

  # Act
  mutated = MutationSwap(seed=4, candidate_k=10).mutate(route, mutation_rate=1.0, context=context)

  # Assert
  assert sorted(mutated.tolist()) == list(range(52))
  assert mutated[0] == route[0]
  assert mutated[-1] == route[-1]
  np.testing.assert_array_equal(route, original)


def test_proximity_index_selects_a_candidate_neighbour(berlin52_problem: Problem) -> None:
  """WHAT: `select_proximity_index` on a full tour with `candidate_k=10`.

  WHY: the second swap position must be the tour position of a real `N_k` neighbour of
  the first city, in the mutable interior, and different from the first index.

  EXPECTED: an interior index whose city belongs to `N_k[route[idx1]]`.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=12)
  route = np.random.default_rng(13).permutation(np.arange(52, dtype=np.int32)).astype(np.int32, copy=False)
  rng = np.random.default_rng(5)
  first_index = 9
  candidates = context.get_cpu_candidate_list(10)

  # Act
  second_index = select_proximity_index(route, first_index, 1, len(route) - 1, candidates, rng)

  # Assert
  assert second_index != first_index
  assert 1 <= second_index <= len(route) - 2
  neighbours = {int(neighbor) for neighbor in candidates[int(route[first_index])]}
  assert int(route[second_index]) in neighbours


def test_candidate_k_swap_supports_sparse_cvrp_sub_route(berlin52_problem: Problem) -> None:
  """WHAT: `MutationSwap(candidate_k=3)` on `[0, 5, 12, 19, 8, 0]` (max id 19, 6 stops).

  WHY: node ids above `len(route) - 1` used to disable candidate mode or raise
  `IndexError`; the shared inverse map sizes itself by the largest id (`BUG-19`).

  EXPECTED: a valid sub-route permutation with both depot endpoints untouched.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=21)
  route = np.array([0, 5, 12, 19, 8, 0], dtype=np.int32)
  original = route.copy()

  # Act
  mutated = MutationSwap(seed=7, candidate_k=3).mutate(route, mutation_rate=1.0, context=context)

  # Assert
  assert sorted(mutated.tolist()) == sorted(route.tolist())
  assert mutated[0] == 0
  assert mutated[-1] == 0
  np.testing.assert_array_equal(route, original)
