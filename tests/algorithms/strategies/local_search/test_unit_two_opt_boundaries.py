"""Boundary contracts for the CPU 2-opt implementations in `two_opt.py`.

Every variant must be a no-op for tours shorter than one 2-opt move and for a
non-positive iteration budget, while still returning a defensive copy so caller-owned
routes are never aliased.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies import CPUFirstImprovementTwoOpt, CPUMultiMoveTwoOpt, CPUTwoOpt
from protocols import ProblemContext

if TYPE_CHECKING:
  from collections.abc import Callable

  from data_models.problem import Problem

CPU_TWO_OPT_FACTORIES = [
  pytest.param(CPUTwoOpt, id="best-improvement"),
  pytest.param(CPUFirstImprovementTwoOpt, id="first-improvement"),
  pytest.param(CPUMultiMoveTwoOpt, id="multi-move"),
]


@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_short_tour_returns_equal_distinct_copy(
  factory: Callable[[int], CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  burma14_problem: Problem,
) -> None:
  """WHAT: A three-city tour through every CPU 2-opt variant.

  WHY: A 2-opt move needs four distinct positions; shorter tours must be returned
  untouched, but still as independent memory so callers cannot alias the result.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.array([0, 2, 1], dtype=np.int32)
  original = tour.copy()

  improved = factory(10).improve(tour, context=context)

  assert improved is not tour, "boundary path must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"


@pytest.mark.parametrize("max_iterations", [0, -1], ids=["zero", "negative"])
@pytest.mark.parametrize("factory", CPU_TWO_OPT_FACTORIES)
def test_non_positive_iteration_budget_returns_equal_distinct_copy(
  factory: Callable[[int], CPUTwoOpt | CPUFirstImprovementTwoOpt | CPUMultiMoveTwoOpt],
  max_iterations: int,
  burma14_problem: Problem,
) -> None:
  """WHAT: A full burma14 tour evaluated with `max_iterations` of 0 and -1.

  WHY: A non-positive budget disables improvement; returning the input itself would
  let a later in-place stage corrupt the caller's population.

  EXPECTED: `result is not input`, equal values, and an unchanged input array.
  """
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  tour = np.arange(burma14_problem.dimension, dtype=np.int32)
  original = tour.copy()

  improved = factory(max_iterations).improve(tour, context=context)

  assert improved is not tour, "non-positive budget must not alias the caller's route"
  assert np.array_equal(improved, original)
  assert np.array_equal(tour, original), "the input route must stay untouched"
