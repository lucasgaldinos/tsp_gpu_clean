"""Unit tests for `SelectionRouletteWheel`.

Roulette-wheel selection samples parents with probability proportional to inverse
cost. The tests pin the contract the GA relies on: selected routes are real population
members returned as copies, cheaper routes dominate the draw, batch and single APIs
agree, and a one-route population is handled without dividing by zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies.selection.roulette_wheel_selection import SelectionRouletteWheel

if TYPE_CHECKING:
  from collections.abc import Callable


@pytest.mark.parametrize("seed", [0, 1, 5, 17])
def test_roulette_wheel_select_returns_population_member(
  seed: int,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: Single-parent roulette-wheel selection.

  WHY: Selection must return an existing route, never a synthetic one, and must return
  a copy so later mutation cannot corrupt the population.

  EXPECTED: A `(n_genes,)` int32 route equal to one population row and sharing no memory.
  """
  # Arrange
  population = make_population(4, 6, seed=99)
  fitness = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
  strategy = SelectionRouletteWheel(seed=seed)

  # Act
  selected = strategy.select(population, fitness)

  # Assert
  assert selected.shape == (6,)
  assert selected.dtype == np.int32
  assert any(np.array_equal(selected, member) for member in population)
  assert not np.shares_memory(selected, population)


def test_roulette_wheel_favours_lower_cost_routes(make_population: Callable[..., np.ndarray]) -> None:
  """WHAT: Fitness-proportionate weighting of roulette-wheel selection.

  WHY: Inverse-cost weights are the entire contract of this strategy; uniform sampling
  would silently degrade GA pressure.

  EXPECTED: The cheapest route dominates 50 seeded draws.
  """
  # Arrange
  population = make_population(4, 6, seed=5)
  fitness = np.array([1000.0, 1.0, 1000.0, 1000.0], dtype=np.float64)
  strategy = SelectionRouletteWheel(seed=42)

  # Act
  wins = sum(int(np.array_equal(strategy.select(population, fitness), population[1])) for _ in range(50))

  # Assert
  assert wins > 40, f"expected the cheapest route to dominate selection, got {wins}/50"


@pytest.mark.parametrize("num_parents", [1, 7])
def test_roulette_wheel_select_batch_returns_population_members(
  num_parents: int,
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: Batch roulette-wheel selection with `num_parents` smaller and larger than
  the population.

  WHY: The GA requests exactly `population_size` parents; undersampling and sampling
  with replacement must both stay valid.

  EXPECTED: `(num_parents, n_genes)` int32 rows drawn from the population.
  """
  # Arrange
  population = make_population(4, 6, seed=99)
  fitness = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
  strategy = SelectionRouletteWheel(seed=8)

  # Act
  parents = strategy.select_batch(population, fitness, num_parents=num_parents)

  # Assert
  assert parents.shape == (num_parents, 6)
  assert parents.dtype == np.int32
  for parent in parents:
    assert any(np.array_equal(parent, member) for member in population)


def test_roulette_wheel_handles_single_member_population() -> None:
  """WHAT: Roulette-wheel selection with a one-route population.

  WHY: Population size 1 is the lower boundary of the constructor and must not divide
  by zero or return an empty selection.

  EXPECTED: The only member is selected for both the single and batch APIs.
  """
  # Arrange
  population = np.arange(4, dtype=np.int32).reshape(1, 4)
  fitness = np.array([42.0], dtype=np.float64)
  strategy = SelectionRouletteWheel(seed=2)

  # Act
  selected = strategy.select(population, fitness)
  parents = strategy.select_batch(population, fitness, num_parents=3)

  # Assert
  np.testing.assert_array_equal(selected, population[0])
  assert parents.shape == (3, 4)
  for parent in parents:
    np.testing.assert_array_equal(parent, population[0])
