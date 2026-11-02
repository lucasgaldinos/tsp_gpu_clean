"""Unit tests for `OrderCrossover`.

Order crossover copies a slice from the first parent and fills the remaining positions
in the order the second parent visits the missing genes. The tests pin the invariants
cost evaluation depends on: the child is a permutation of the parent genes, depot
endpoints survive, and whole-population batches keep one permutation per row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies.crossover.order_crossover import OrderCrossover

if TYPE_CHECKING:
  from collections.abc import Callable


@pytest.mark.parametrize("n_genes", [1, 2, 5, 9])
def test_crossover_returns_permutation_of_parent_genes(n_genes: int) -> None:
  """WHAT: Children produced by order crossover.

  WHY: A crossover that drops or duplicates a gene yields a tour that cost evaluation
  cannot interpret.

  EXPECTED: A length-`n_genes` int32 child containing every parent gene exactly once.
  """
  # Arrange
  rng = np.random.default_rng(1000 + n_genes)
  base = np.arange(n_genes, dtype=np.int32)
  parent1 = rng.permutation(base).astype(np.int32, copy=False)
  parent2 = rng.permutation(base).astype(np.int32, copy=False)
  strategy = OrderCrossover(seed=11)

  # Act
  child = strategy.crossover(parent1, parent2)

  # Assert
  assert child.shape == (n_genes,)
  assert child.dtype == np.int32
  assert sorted(child.tolist()) == sorted(parent1.tolist())
  assert len(set(child.tolist())) == n_genes


def test_crossover_preserves_depot_endpoints() -> None:
  """WHAT: Crossover on depot-bounded routes `[depot, ..., depot]`.

  WHY: Route-based formulations rely on the depot staying first and last so the
  constructor can split the chromosome into vehicle routes.

  EXPECTED: The child keeps the depot at both ends and a valid permutation inside.
  """
  # Arrange
  parent1 = np.array([0, 1, 2, 3, 4, 0], dtype=np.int32)
  parent2 = np.array([0, 4, 2, 1, 3, 0], dtype=np.int32)

  # Act
  child = OrderCrossover(seed=21).crossover(parent1, parent2)

  # Assert
  assert child.shape == parent1.shape
  assert child[0] == 0
  assert child[-1] == 0
  assert sorted(child[1:-1].tolist()) == [1, 2, 3, 4]


@pytest.mark.parametrize("num_routes", [1, 5])
def test_crossover_batch_returns_permutation_rows(
  num_routes: int,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: Batch crossover over one route and an odd-sized population.

  WHY: The GA recombines whole populations; single-route input and odd counts exercise
  the copy and wrap-around branches.

  EXPECTED: Output has the input shape, int32 dtype, and every row is a permutation.
  """
  # Arrange
  population = make_population(num_routes, 6, seed=7)

  # Act
  offspring = OrderCrossover(seed=5).crossover_batch(population)

  # Assert
  assert offspring.shape == population.shape
  assert offspring.dtype == np.int32
  for row in offspring:
    assert sorted(row.tolist()) == list(range(6))
