"""Unit tests for `resolve_route_bounds` and `sample_distinct_pairs` (`DP11-19`, `DP11-20`).

These helpers decide which positions stochastic operators may touch and draw the
distinct pairs they need. The tests pin the two contracts that `BUG-24` and
`BUG-28` broke: depot-bounded routes keep both endpoints while open permutations
may move every position, and the rejection-free sampler yields ordered, distinct,
in-range integer pairs from a single host RNG stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_models.problem import Problem
from protocols.problem_context import ProblemContext
from utils.route_helpers import resolve_route_bounds, sample_distinct_pairs


def _make_problem(name: str, problem_type: str, dimension: int) -> Problem:
  """Build a minimal coordinate problem of the requested formulation."""
  is_cvrp = problem_type == "CVRP"
  return Problem(
    name=name,
    dimension=dimension,
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=np.zeros((dimension, 2), dtype=np.float64),
    distances=None,
    capacity=10 if is_cvrp else None,
    demands=np.ones(dimension, dtype=np.int32) if is_cvrp else None,
  )


def test_problem_context_is_closed_only_for_cvrp() -> None:
  """WHAT: `ProblemContext.is_closed` per formulation.

  WHY: crossover and mutation read this flag instead of syncing route endpoints
  from the device every generation (`DP11-19`).

  EXPECTED: True for CVRP, False for TSP and ATSP.
  """
  assert ProblemContext(_make_problem("c", "CVRP", 8), xp=np).is_closed is True
  assert ProblemContext(_make_problem("t", "TSP", 8), xp=np).is_closed is False
  assert ProblemContext(_make_problem("a", "ATSP", 8), xp=np).is_closed is False


def test_open_tsp_route_can_mutate_both_endpoints(berlin52_problem: Problem) -> None:
  """WHAT: Bounds of an open 52-city permutation under a TSP context.

  WHY: freezing positions `0` and `n - 1` truncated the search space to `(n-2)!`,
  which is the defect `BUG-28` records.

  EXPECTED: `(0, 52)`.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=1)
  route = np.random.default_rng(7).permutation(np.arange(52, dtype=np.int32)).astype(np.int32)

  assert resolve_route_bounds(len(route), context=context, route_sample=route) == (0, 52)


def test_closed_depot_route_keeps_endpoints_even_under_tsp_context() -> None:
  """WHAT: Bounds of a depot-bounded `[0, ..., 0]` route under a TSP context.

  WHY: unit tests and CVRP sub-routes pass closed tours with a TSP context; the
  host endpoint probe must win so the depot delimiters stay fixed (`BUG-5`).

  EXPECTED: `(1, n - 1)` for `n > 2`, and `(1, n - 1)` for the degenerate `[0, 1, 0]`.
  """
  context = ProblemContext(_make_problem("t", "TSP", 10), xp=np, seed=2)
  route = np.array([0, 3, 4, 5, 6, 7, 8, 9, 1, 0], dtype=np.int32)

  assert resolve_route_bounds(len(route), context=context, route_sample=route) == (1, 9)
  assert resolve_route_bounds(3, context=None, route_sample=np.array([0, 1, 0], dtype=np.int32)) == (1, 2)


def test_cvrp_context_forces_depot_bounds_without_a_sample() -> None:
  """WHAT: Bounds resolution when only the context is available.

  WHY: the GPU never inspects endpoints on the device, so `context.is_closed` must
  be sufficient (`BUG-24`).

  EXPECTED: CVRP -> `(1, n - 1)`; TSP -> `(0, n)`.
  """
  cvrp = ProblemContext(_make_problem("c", "CVRP", 12), xp=np)
  tsp = ProblemContext(_make_problem("t", "TSP", 12), xp=np)

  assert resolve_route_bounds(12, context=cvrp) == (1, 11)
  assert resolve_route_bounds(12, context=tsp) == (0, 12)


def test_no_information_keeps_the_conservative_slice() -> None:
  """WHAT: Bounds with neither context nor sample.

  WHY: the previous default preserved depots; losing it would silently move
  endpoints for callers that pass nothing (`TASK 10.10` contract).

  EXPECTED: `(1, n - 1)`.
  """
  assert resolve_route_bounds(9) == (1, 8)


@pytest.mark.parametrize("low, high", [(0, 2), (1, 3), (0, 52), (1, 51)])
def test_distinct_pairs_are_distinct_and_in_range(low: int, high: int) -> None:
  """WHAT: 4,000 draws of `sample_distinct_pairs` per range.

  WHY: `DP11-20` replaces `choice(replace=False)` with a bijection; any collision
  or out-of-range value silently corrupts crossover cuts and mutation swaps.

  EXPECTED: int32 `(num_pairs, 2)`, both columns in `[low, high)`, and `u != v`.
  """
  pairs = sample_distinct_pairs(np.random.default_rng(11), 4000, low, high)

  assert pairs.shape == (4000, 2)
  assert pairs.dtype == np.int32
  assert int(pairs.min()) >= low
  assert int(pairs.max()) < high
  assert np.all(pairs[:, 0] != pairs[:, 1])


def test_distinct_pairs_cover_every_admissible_second_value() -> None:
  """WHAT: Value coverage of the bijection on `[0, 4)`.

  WHY: a biased sampler would make cut selection reproducible-but-wrong; every
  admissible `v` must appear for every `u`.

  EXPECTED: for each `u`, the observed `v` set equals `[0, 4) \\ {u}`.
  """
  pairs = sample_distinct_pairs(np.random.default_rng(3), 4000, 0, 4)

  for first in range(4):
    observed = set(pairs[pairs[:, 0] == first][:, 1].tolist())
    assert observed == set(range(4)) - {first}


def test_distinct_pairs_is_stream_deterministic() -> None:
  """WHAT: Two generators with the same seed.

  WHY: CPU and GPU strategies share this oracle, so identity depends on it being a
  pure function of the stream (`SWAP-A`, `INV-A`, `OX-A`).

  EXPECTED: identical arrays.
  """
  left = sample_distinct_pairs(np.random.default_rng(2024), 64, 1, 51)
  right = sample_distinct_pairs(np.random.default_rng(2024), 64, 1, 51)

  np.testing.assert_array_equal(left, right)


def test_distinct_pairs_reject_degenerate_ranges() -> None:
  """WHAT: Ranges with fewer than two integers.

  WHY: closed tours with `n == 3` have a single interior position; callers must
  guard, and the helper must fail loud instead of returning a self-pair.

  EXPECTED: `ValueError`.
  """
  with pytest.raises(ValueError, match="at least two distinct integers"):
    sample_distinct_pairs(np.random.default_rng(0), 4, 1, 2)
