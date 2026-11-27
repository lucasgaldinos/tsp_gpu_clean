"""Unit tests for `PenaltyPolicy` implementations and `PenaltyAdaptationStep` (`DP17-9`, `Task 17.19`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GenerationState
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY, PenaltyAdaptationStep
from algorithms.strategies.penalties import (
  AdaptivePenaltyPolicy,
  FixedPenaltyPolicy,
  InstanceScaledPenaltyPolicy,
  StrictFeasibilityPolicy,
)
from data_models.problem import Problem
from data_models.vrp_constraints import VRPConstraints
from protocols.problem_context import ProblemContext
from protocols.vrp_strategies import PenaltyPolicy, RouteAssessment
from utils.phase_timer import PhaseTimer


def _make_cvrp_problem(scale: float = 1.0, *, max_vehicles: int | None = 2) -> Problem:
  coords = np.array(
    [
      [0.0, 0.0],
      [3.0, 4.0],
      [6.0, 0.0],
      [0.0, 8.0],
      [6.0, 8.0],
    ],
    dtype=np.float64,
  ) * scale
  demands = np.array([0, 4, 5, 3, 4], dtype=np.int32)
  return Problem(
    name=f"cvrp_scale_{scale}",
    dimension=5,
    problem_type="CVRP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=10,
    demands=demands,
    constraints=VRPConstraints(
      capacity=10,
      max_vehicles=max_vehicles,
      max_route_distance=50.0 * scale,
      return_to_depot=True,
    ),
  )


def test_instance_scaled_penalty_policy_rescaling_invariance_and_no_materialization_above_limit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: `InstanceScaledPenaltyPolicy` scales linearly with coordinate rescaling and skips O(n^2) above threshold."""
  policy = InstanceScaledPenaltyPolicy(safety_factor=2.0)
  assert isinstance(policy, PenaltyPolicy)

  ctx_small = ProblemContext(_make_cvrp_problem(scale=1.0), xp=np)
  ctx_large = ProblemContext(_make_cvrp_problem(scale=100.0), xp=np)

  wq1, wl1, wk1 = policy.weights(ctx_small)
  wq2, wl2, wk2 = policy.weights(ctx_large)

  d_max_1 = float(np.max(ctx_small.get_cpu_distances()))
  assert wq1 == pytest.approx(2.0 * d_max_1 + 1.0)
  assert wl1 == pytest.approx(2.0)
  assert wk1 == pytest.approx(2.0 * d_max_1 * 4.0)

  # Uniform 100x coordinate rescaling preserves edge penalty ratio and scales capacity/fleet bounds by ~100x.
  assert wl2 == pytest.approx(wl1)
  assert (wq2 - 1.0) == pytest.approx(100.0 * (wq1 - 1.0), rel=1e-3)
  assert wk2 == pytest.approx(100.0 * wk1, rel=1e-3)

  # Above `max_matrix_dimension`, D_max is bounded in O(n) from coordinates without calling get_cpu_distances.
  ctx_no_matrix = ProblemContext(_make_cvrp_problem(scale=10.0), xp=np, max_matrix_dimension=3)
  monkeypatch.setattr(
    ctx_no_matrix,
    "get_cpu_distances",
    lambda: pytest.fail("get_cpu_distances must not be called above max_matrix_dimension"),
  )
  wq_big, wl_big, wk_big = policy.weights(ctx_no_matrix)
  assert wq_big > 1.0
  assert wl_big == pytest.approx(2.0)
  assert wk_big > 0.0


def test_strict_feasibility_policy_avoids_ieee754_nan_on_zero_violations() -> None:
  """WHAT: `StrictFeasibilityPolicy.apply` returns finite travel cost on feasible rows (never `0.0 * inf == NaN`)."""
  policy = StrictFeasibilityPolicy()
  assert isinstance(policy, PenaltyPolicy)
  ctx = ProblemContext(_make_cvrp_problem(), xp=np)

  travel = np.array([123.0, 80.0, 95.0], dtype=np.float64)
  comps = np.array(
    [
      [0.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  out = policy.apply(travel, comps, ctx)
  assert out.dtype == np.float64
  assert not np.isnan(out).any()
  assert out[0] == pytest.approx(123.0)
  assert math.isinf(out[1]) and out[1] > 0.0
  assert math.isinf(out[2]) and out[2] > 0.0

  # `GenerationState.refresh_penalized_fitness` must also stay NaN-free under infinite weights.
  state = GenerationState(
    population=np.zeros((3, 5), dtype=np.int32),
    fitness=np.zeros(3, dtype=np.float64),
    context=ctx,
    population_assessment=RouteAssessment(
      objectives=travel.copy(),
      travel_costs=travel,
      is_feasible=np.array([True, False, False]),
      violations=np.array([0.0, 2.0, 1.0]),
      normalized_violations=np.array([0.0, 0.2, 0.5]),
      raw_violation_components=comps,
    ),
  )
  state.refresh_penalized_fitness(*policy.weights(ctx))
  assert not np.isnan(state.fitness).any()
  assert state.fitness[0] == pytest.approx(123.0)
  assert math.isinf(state.fitness[1])


def test_fixed_penalty_policy_validates_non_negative_non_bool() -> None:
  """WHAT: `FixedPenaltyPolicy` rejects `bool`, negatives, and non-finite inputs while applying static weights."""
  policy = FixedPenaltyPolicy(penalty_capacity=500.0, penalty_distance=10.0, penalty_vehicles=2000.0)
  assert isinstance(policy, PenaltyPolicy)
  ctx = ProblemContext(_make_cvrp_problem(), xp=np)
  assert policy.weights(ctx) == (500.0, 10.0, 2000.0)

  # Partial or all-None axes resolve only the None axes via InstanceScaledPenaltyPolicy (`BUG-118`).
  scaled_q, scaled_l, scaled_k = InstanceScaledPenaltyPolicy().weights(ctx)
  assert FixedPenaltyPolicy().weights(ctx) == pytest.approx((scaled_q, scaled_l, scaled_k))
  assert FixedPenaltyPolicy(penalty_capacity=123.0).weights(ctx) == pytest.approx((123.0, scaled_l, scaled_k))
  assert FixedPenaltyPolicy(penalty_distance=0.0, penalty_vehicles=50.0).weights(ctx) == pytest.approx(
    (scaled_q, 0.0, 50.0)
  )

  with pytest.raises(ValueError, match="bool"):
    FixedPenaltyPolicy(penalty_capacity=True)
  with pytest.raises(ValueError, match=">= 0"):
    FixedPenaltyPolicy(penalty_distance=-1.0)
  with pytest.raises(ValueError, match="finite"):
    FixedPenaltyPolicy(penalty_vehicles=math.inf)


def test_adaptive_penalty_policy_and_step_refresh_fitness_and_reset_metrics() -> None:
  """WHAT: `AdaptivePenaltyPolicy` + `PenaltyAdaptationStep` adapt weights and refresh `state.fitness` in O(N)."""
  assert "PenaltyAdaptationStep" in PIPELINE_STEP_REGISTRY
  policy = AdaptivePenaltyPolicy(
    epsilon_tol=0.10,
    up_factor=1.5,
    down_factor=0.5,
    target_feasible_low=0.25,
    target_feasible_high=0.75,
    min_scale=0.25,
    max_scale=4.0,
  )
  assert isinstance(policy, PenaltyPolicy)
  step = PenaltyAdaptationStep(penalty_policy=policy, interval=2)
  assert step.stage_anchor == "post_survival"

  ctx = ProblemContext(_make_cvrp_problem(), xp=np)
  base_weights = policy.weights(ctx)
  travel = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
  comps_low_feas = np.array(
    [
      [0.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
      [6.0, 0.0, 0.0],
      [4.0, 0.0, 0.0],
    ],
    dtype=np.float64,
  )
  state = GenerationState(
    population=np.zeros((4, 5), dtype=np.int32),
    fitness=np.zeros(4, dtype=np.float64),
    context=ctx,
    generation=1,
    population_assessment=RouteAssessment(
      objectives=travel.copy(),
      travel_costs=travel,
      is_feasible=np.array([True, False, False, False]),
      violations=np.array([0.0, 5.0, 6.0, 4.0]),
      normalized_violations=np.array([0.0, 0.5, 0.6, 0.4]),
      raw_violation_components=comps_low_feas,
    ),
  )

  # Off-cadence generation=1: no adaptation and no fitness mutation.
  step.run(state, PhaseTimer(enabled=False))
  assert policy.weights(ctx) == base_weights
  assert np.array_equal(state.fitness, np.zeros(4, dtype=np.float64))

  # On-cadence generation=2: low feasibility triggers `up_factor` and refreshes `state.fitness` immediately.
  state.generation = 2
  step.run(state, PhaseTimer(enabled=False))
  assert policy.weights(ctx) == pytest.approx(tuple(1.5 * w for w in base_weights))
  expected_fit = policy.apply(travel, comps_low_feas, ctx)
  np.testing.assert_allclose(state.fitness, expected_fit)
  assert state.population_assessment is not None
  np.testing.assert_allclose(state.population_assessment.objectives, expected_fit)

  # High feasibility triggers `down_factor`.
  state.population_assessment = RouteAssessment(
    objectives=travel.copy(),
    travel_costs=travel,
    is_feasible=np.array([True, True, True, True]),
    violations=np.zeros(4, dtype=np.float64),
    normalized_violations=np.zeros(4, dtype=np.float64),
    raw_violation_components=np.zeros((4, 3), dtype=np.float64),
  )
  state.generation = 4
  step.run(state, PhaseTimer(enabled=False))
  assert policy.weights(ctx) == pytest.approx(tuple(0.75 * w for w in base_weights))

  # `reset_metrics()` restores the dynamic multiplier scale back to 1.0x.
  policy.reset_metrics()
  assert policy.weights(ctx) == pytest.approx(base_weights)
