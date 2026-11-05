"""Unit tests for `MutationInversion`.

Inversion mutation reverses a contiguous segment of a route. The tests pin the
invariants the GA depends on: the permutation and its dtype survive, the parent array
is never mutated in place, a zero rate is an identity, and batch input keeps its
dimensionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies.mutation.inversion_mutation import MutationInversion
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from collections.abc import Callable

  from data_models.problem import Problem


@pytest.mark.parametrize("mutation_rate", [0.0, 1.0])
@pytest.mark.parametrize("n_genes", [1, 2, 5, 9])
def test_mutation_preserves_permutation(mutation_rate: float, n_genes: int) -> None:
  """WHAT: Inversion mutation across rates and route sizes.

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
  mutated = MutationInversion(seed=3).mutate(route, mutation_rate=mutation_rate)

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
  mutated = MutationInversion(seed=13).mutate_batch(input_routes, mutation_rate=1.0)

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
  mutated = MutationInversion(seed=0).mutate(route, mutation_rate=1.0)

  # Assert
  assert mutated[0] == 0
  assert mutated[-1] == 0
  assert sorted(mutated.tolist()) == sorted(route.tolist())


def test_candidate_k_inversion_keeps_permutation_and_depot(berlin52_problem: Problem) -> None:
  """WHAT: `MutationInversion(candidate_k=10)` on a full 52-city tour.

  WHY: proximity-guided cuts change how the segment is chosen; the reversal must still
  produce a permutation and leave the depot endpoints fixed (`DP10-20`).

  EXPECTED: valid permutation, depot endpoints untouched, parent array not mutated.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=19)
  route = np.random.default_rng(23).permutation(np.arange(52, dtype=np.int32)).astype(np.int32, copy=False)
  original = route.copy()

  # Act
  mutated = MutationInversion(seed=6, candidate_k=10).mutate(route, mutation_rate=1.0, context=context)

  # Assert
  assert sorted(mutated.tolist()) == list(range(52))
  assert mutated[0] == route[0]
  assert mutated[-1] == route[-1]
  np.testing.assert_array_equal(route, original)


def test_candidate_k_inversion_supports_sparse_cvrp_sub_route(berlin52_problem: Problem) -> None:
  """WHAT: `MutationInversion(candidate_k=3)` on `[0, 5, 12, 19, 8, 0]` (max id 19, 6 stops).

  WHY: node ids above `len(route) - 1` used to disable candidate mode or raise
  `IndexError`; the shared inverse map sizes itself by the largest id (`BUG-19`).

  EXPECTED: a valid sub-route permutation with both depot endpoints untouched.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=22)
  route = np.array([0, 5, 12, 19, 8, 0], dtype=np.int32)
  original = route.copy()

  # Act
  mutated = MutationInversion(seed=8, candidate_k=3).mutate(route, mutation_rate=1.0, context=context)

  # Assert
  assert sorted(mutated.tolist()) == sorted(route.tolist())
  assert mutated[0] == 0
  assert mutated[-1] == 0
  np.testing.assert_array_equal(route, original)
