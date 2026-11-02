"""Unit tests for `RandomPermutationInitialization`.

Population initialization is the entry point of every GA generation: the routes it
emits are the ancestors of all later crossover and mutation products. The tests pin
the output contract (shape, dtype, gene multiset) and seed reproducibility rather than
a particular random sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.strategies.initialization.random_initialization import RandomPermutationInitialization


@pytest.mark.parametrize(
  ("customers", "population_size"),
  [
    pytest.param([0, 1, 2, 3, 4, 5], 1, id="single-route"),
    pytest.param([7, 3, 9, 1, 4], 8, id="non-contiguous-customers"),
  ],
)
def test_random_permutation_initialization_returns_valid_population(
  customers: list[int],
  population_size: int,
) -> None:
  """WHAT: `RandomPermutationInitialization.initialize` output.

  WHY: Every GA generation descends from this population, so a duplicated or missing
  gene would corrupt all downstream routes.

  EXPECTED: `(population_size, len(customers))` int32 routes, each a permutation of
  exactly the requested customer set.
  """
  # Arrange
  strategy = RandomPermutationInitialization(seed=17)

  # Act
  population = strategy.initialize(customers, population_size)

  # Assert
  assert population.shape == (population_size, len(customers))
  assert population.dtype == np.int32
  expected_genes: list[int] = sorted(customers)
  for route in population:
    assert sorted(route.tolist()) == expected_genes
    assert len(set(route.tolist())) == len(customers)


def test_random_permutation_initialization_respects_seed() -> None:
  """WHAT: Reproducibility of population initialization under an explicit seed.

  WHY: Tests and benchmarks require identical starting populations across runs.

  EXPECTED: Two strategies built with seed 123 emit identical arrays.
  """
  # Arrange
  customers: list[int] = list(range(6))

  # Act
  first = RandomPermutationInitialization(seed=123).initialize(customers, population_size=4)
  second = RandomPermutationInitialization(seed=123).initialize(customers, population_size=4)

  # Assert
  np.testing.assert_array_equal(first, second)
