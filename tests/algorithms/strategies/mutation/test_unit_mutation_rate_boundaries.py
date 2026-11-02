"""Boundary contracts for `mutation_rate` in the CPU mutation strategies.

Rate 0.0 must be a bit-exact identity; rate 1.0 must guarantee a change for every
route of at least four genes. Both properties protect the GA from silently losing its
exploration operator or from corrupting a population at rate 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies import InversionMutation, SwapMutation

if TYPE_CHECKING:
  from collections.abc import Callable

MUTATION_FACTORIES = [
  pytest.param(SwapMutation, id="swap"),
  pytest.param(InversionMutation, id="inversion"),
]


@pytest.mark.parametrize("factory", MUTATION_FACTORIES)
def test_zero_rate_leaves_every_route_bit_identical(
  factory: Callable[[int], SwapMutation | InversionMutation],
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: A fixed six-route population mutated at `mutation_rate=0.0`.

  WHY: A zero rate is the GA's "no exploration" switch; any drift would silently
  alter parents that the next generation still references.

  EXPECTED: same shape and dtype, identical bytes, and an unchanged input batch.
  """
  routes = make_population(6, 8, seed=2026)
  original = routes.copy()

  mutated = factory(7).mutate_batch(routes, mutation_rate=0.0)

  assert mutated.shape == original.shape
  assert mutated.dtype == np.int32
  assert mutated.tobytes() == original.tobytes(), "rate 0.0 must not change any gene"
  assert np.array_equal(routes, original), "the input batch must stay untouched"


@pytest.mark.parametrize("factory", MUTATION_FACTORIES)
def test_full_rate_changes_every_route_of_length_at_least_four(
  factory: Callable[[int], SwapMutation | InversionMutation],
  make_population: Callable[..., np.ndarray],
) -> None:
  """WHAT: A fixed six-route population of eight-gene permutations at rate 1.0.

  WHY: The GA relies on `mutation_rate=1.0` meaning "mutate every offspring"; a
  probabilistic no-op would hide broken RNG plumbing behind a passing permutation check.

  EXPECTED: every row differs from its parent and remains a permutation.
  """
  routes = make_population(6, 8, seed=2026)

  mutated = factory(7).mutate_batch(routes, mutation_rate=1.0)

  assert mutated.shape == routes.shape
  for index, (before, after) in enumerate(zip(routes, mutated, strict=True)):
    assert not np.array_equal(before, after), f"row {index} was left unmutated at rate 1.0"
    assert sorted(after.tolist()) == sorted(before.tolist()), f"row {index} is no longer a permutation"
