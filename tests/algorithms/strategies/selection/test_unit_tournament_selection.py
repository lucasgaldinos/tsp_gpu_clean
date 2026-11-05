"""Unit tests for `SelectionTournament`.

Tournament selection samples `k` competitors and keeps the cheapest. The tests pin the
contract the GA relies on: every winner is a real population member, `k` larger than
the population saturates safely, batch output is drawn from the population, and a
non-positive `k` is rejected by the constructor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies.selection.tournament_selection import SelectionTournament

if TYPE_CHECKING:
  from collections.abc import Callable


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("seed", [0, 4])
def test_tournament_select_returns_population_member(
  k: int,
  seed: int,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: Single-parent tournament selection for several tournament sizes.

  WHY: Tournament size drives selection pressure; every winner must still be a real
  population member, and `k` larger than the population must saturate safely.

  EXPECTED: A population route; when `k` covers the whole population, the cheapest one.
  """
  # Arrange
  population = make_population(4, 6, seed=77)
  fitness = np.array([10.0, 30.0, 20.0, 40.0], dtype=np.float64)
  strategy = SelectionTournament(k=k, seed=seed)

  # Act
  selected = strategy.select(population, fitness)

  # Assert
  assert selected.shape == (6,)
  assert any(np.array_equal(selected, member) for member in population)
  if k >= len(population):
    np.testing.assert_array_equal(selected, population[int(np.argmin(fitness))])


@pytest.mark.parametrize("num_parents", [1, 6])
def test_tournament_select_batch_returns_population_members(
  num_parents: int,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: Batch tournament selection.

  WHY: The GA pipeline consumes selected parents as a 2D batch; each row must come from
  the existing population.

  EXPECTED: `(num_parents, n_genes)` int32 rows matching population members.
  """
  # Arrange
  population = make_population(4, 6, seed=77)
  fitness = np.array([10.0, 30.0, 20.0, 40.0], dtype=np.float64)
  strategy = SelectionTournament(k=3, seed=8)

  # Act
  parents = strategy.select_batch(population, fitness, num_parents=num_parents)

  # Assert
  assert parents.shape == (num_parents, 6)
  assert parents.dtype == np.int32
  for parent in parents:
    assert any(np.array_equal(parent, member) for member in population)


def test_tournament_handles_single_member_population() -> None:
  """WHAT: Tournament selection with a one-route population.

  WHY: `k` must be clamped to the population size instead of requesting more distinct
  competitors than exist.

  EXPECTED: Every selection returns the only member.
  """
  # Arrange
  population = np.arange(4, dtype=np.int32).reshape(1, 4)
  fitness = np.array([7.0], dtype=np.float64)
  strategy = SelectionTournament(k=3, seed=6)

  # Act
  selected = strategy.select(population, fitness)
  parents = strategy.select_batch(population, fitness, num_parents=2)

  # Assert
  np.testing.assert_array_equal(selected, population[0])
  assert parents.shape == (2, 4)
  for parent in parents:
    np.testing.assert_array_equal(parent, population[0])


@pytest.mark.parametrize("k", [0, -1])
def test_tournament_rejects_non_positive_tournament_size(k: int) -> None:
  """WHAT: Constructor validation for the tournament size.

  WHY: A non-positive `k` would produce an empty candidate set and an undefined winner.

  EXPECTED: `ValueError` naming the minimum valid size.
  """
  with pytest.raises(ValueError, match="Tournament size k must be >= 1"):
    SelectionTournament(k=k, seed=0)
