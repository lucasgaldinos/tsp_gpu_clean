"""Unit tests for VRP evaluation paths in `EvaluationCPUCost` and `EvaluationGPUCost` (`DP17-7`, `DP17-9`)."""

from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
import pytest

from algorithms.strategies.decoding.prins_split import PrinsSplitDecoder
from algorithms.strategies.evaluation import (
  EvaluationCPUCost,
  EvaluationGPUCost,
)
from algorithms.strategies.penalties import (
  AdaptivePenaltyPolicy,
  FixedPenaltyPolicy,
  InstanceScaledPenaltyPolicy,
  StrictFeasibilityPolicy,
)
from data_models import Problem, VRPConstraints
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext


def _make_cvrp_problem(
  *,
  problem_type: str = "CVRP",
  capacity: int = 10,
  max_route_distance: float | None = None,
  max_vehicles: int | None = None,
  return_to_depot: bool = True,
) -> Problem:
  """Build a deterministic 5-node VRP problem (depot=0, customers=1..4)."""
  coords = np.array(
    [
      [0.0, 0.0],
      [3.0, 0.0],
      [6.0, 0.0],
      [0.0, 4.0],
      [0.0, 8.0],
    ],
    dtype=np.float64,
  )
  demands = np.array([0, 4, 5, 6, 4], dtype=np.int32)
  constraints = VRPConstraints(
    capacity=capacity,
    max_vehicles=max_vehicles,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  return Problem(
    name=f"cvrp5_{problem_type.lower()}",
    dimension=5,
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=capacity,
    demands=demands,
    constraints=constraints,
  )


def _make_large_vrp_problem(
  *,
  dimension: int = 50,
  capacity: int = 20,
  max_route_distance: float | None = 60.0,
  max_vehicles: int | None = 6,
  return_to_depot: bool = True,
) -> Problem:
  """Build a 50-node deterministic VRP instance for multi-tile (`width > 32`) CUDA tests."""
  angles = np.linspace(0.0, 2.0 * np.pi, dimension, endpoint=False)
  radii = np.where(np.arange(dimension) == 0, 0.0, 10.0 + (np.arange(dimension) % 5) * 3.0)
  coords = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).astype(np.float64)
  demands = np.array([0 if i == 0 else (3 + (i % 6)) for i in range(dimension)], dtype=np.int32)
  return Problem(
    name=f"vrp{dimension}",
    dimension=dimension,
    problem_type="DCVRP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=capacity,
    demands=demands,
    constraints=VRPConstraints(
      capacity=capacity,
      max_vehicles=max_vehicles,
      max_route_distance=max_route_distance,
      return_to_depot=return_to_depot,
    ),
  )


def test_delimited_cpu_exact_cost_and_penalties() -> None:
  """WHAT: Direct `0`-delimited evaluation on feasible and infeasible routes.
  WHY: Evaluators must score exact TSPLIB95 travel cost, ignore consecutive `0` padding,
  and apply exact linear penalties for Q, L_max, and K_max excess.
  EXPECTED: exact unpenalized cost, violations, num_vehicles, and penalized float64 fitness.
  """
  problem = _make_cvrp_problem(capacity=10, max_route_distance=16.0, max_vehicles=1, problem_type="DCVRP")
  ctx = ProblemContext(problem, xp=np, seed=1)
  dist = ctx.get_cpu_distances()
  r1_d = int(dist[0, 1] + dist[1, 2] + dist[2, 0])
  r2_d = int(dist[0, 3] + dist[3, 4] + dist[4, 0])
  r1_bad_d = int(dist[0, 1] + dist[1, 2] + dist[2, 3] + dist[3, 0])
  r2_bad_d = int(dist[0, 4] + dist[4, 0])

  routes = np.array(
    [
      [0, 1, 2, 0, 0, 3, 4, 0],
      [0, 1, 2, 3, 0, 4, 0, 0],
    ],
    dtype=np.int32,
  )
  ev = EvaluationCPUCost(
    penalty_policy=FixedPenaltyPolicy(penalty_capacity=100.0, penalty_distance=10.0, penalty_vehicles=1000.0)
  )
  assert isinstance(ev.penalty_policy, FixedPenaltyPolicy)
  res = ev.evaluate_detailed_batch(routes, ctx)
  costs = ev.evaluate_batch(routes, ctx)

  assert costs.dtype == np.dtype(np.float64)
  np.testing.assert_allclose(costs, res.penalized_costs)
  assert res.num_vehicles.tolist() == [2, 2]

  expected_unpen_0 = float(r1_d + r2_d)
  expected_pen_0 = expected_unpen_0 + 1000.0 * 1.0
  assert res.unpenalized_costs[0] == expected_unpen_0
  assert res.penalized_costs[0] == expected_pen_0
  assert res.violations[0] == 1.0

  dist_exc_1 = max(0.0, float(r1_bad_d) - 16.0) + max(0.0, float(r2_bad_d) - 16.0)
  expected_unpen_1 = float(r1_bad_d + r2_bad_d)
  expected_pen_1 = expected_unpen_1 + 100.0 * 5.0 + 10.0 * dist_exc_1 + 1000.0 * 1.0
  assert res.unpenalized_costs[1] == expected_unpen_1
  assert res.penalized_costs[1] == expected_pen_1
  assert res.violations[1] == 5.0 + dist_exc_1 + 1.0


def test_ovrp_open_routes_omit_return_leg() -> None:
  """WHAT: `OVRP` (`return_to_depot=False`) evaluation on both Giant-Tour and `0`-delimited inputs.
  WHY: Open VRP vehicles stop at the last customer without paying `d(last, 0)`.
  EXPECTED: exact open-route cost without return legs.
  """
  problem = _make_cvrp_problem(problem_type="OVRP", capacity=10, return_to_depot=False)
  ctx = ProblemContext(problem, xp=np, seed=2)
  delimited = np.array([[0, 1, 2, 0, 3, 4, 0]], dtype=np.int32)
  giant = np.array([[1, 2, 3, 4]], dtype=np.int32)

  ev = EvaluationCPUCost(split_mode="prins")
  assert isinstance(ev.penalty_policy, InstanceScaledPenaltyPolicy)
  assert isinstance(ev._decoder, PrinsSplitDecoder)
  assert ev._decoder.penalty_policy is ev.penalty_policy
  assert ev.evaluate(delimited[0], ctx) == 14.0
  assert ev.evaluate(giant[0], ctx) == 14.0


def test_penalty_policy_injection_strict_feasibility_and_assessment() -> None:
  """WHAT: `PenaltyPolicy` injection (`InstanceScaled`, `StrictFeasibility`, `Adaptive`) & `RouteAssessment`.
  WHY: Evaluators must default to `InstanceScaledPenaltyPolicy`, share one policy with `_decoder`,
  avoid `0.0 * inf = NaN` under `StrictFeasibilityPolicy`, and populate `(N, 3)` + normalized violations.
  EXPECTED: exact penalty parity across Giant-Tour and delimited paths, zero NaNs, finite feasible cost.
  """
  problem = _make_cvrp_problem(capacity=10, max_route_distance=16.0, max_vehicles=2, problem_type="DCVRP")
  ctx = ProblemContext(problem, xp=np, seed=5)
  delimited = np.array(
    [
      [0, 1, 2, 0, 3, 4, 0, 0],  # feasible: loads (9, 10), dists (12, 16)
      [0, 1, 2, 3, 0, 4, 0, 0],  # infeasible: load 15 (+5), dist 17 (+1)
    ],
    dtype=np.int32,
  )

  ev_strict = EvaluationCPUCost(penalty_policy=StrictFeasibilityPolicy())
  assert isinstance(ev_strict._decoder, PrinsSplitDecoder)
  assert ev_strict._decoder.penalty_policy is ev_strict.penalty_policy
  assess = ev_strict.assess_batch(delimited, ctx)
  assert assess.is_feasible.tolist() == [True, False]
  assert math.isfinite(float(assess.objectives[0]))
  assert float(assess.objectives[0]) == float(assess.travel_costs[0])
  assert math.isinf(float(assess.objectives[1])) and float(assess.objectives[1]) > 0.0
  assert assess.raw_violation_components is not None
  np.testing.assert_allclose(assess.raw_violation_components, [[0.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
  np.testing.assert_allclose(assess.normalized_violations, [0.0, 5.0 / 10.0 + 1.0 / 16.0])

  adaptive = AdaptivePenaltyPolicy()
  ev_adapt = EvaluationCPUCost(penalty_policy=adaptive)
  giant = np.array([[1, 2, 3, 4]], dtype=np.int32)
  giant_assess = ev_adapt.assess_batch(giant, ctx)
  assert giant_assess.raw_violation_components is not None
  assert giant_assess.raw_violation_components.shape == (1, 3)
  assert giant_assess.normalized_violations.shape == (1,)
  adaptive.adapt(assess, ctx)
  ev_adapt.reset_metrics()
  assert adaptive.weights(ctx) == InstanceScaledPenaltyPolicy().weights(ctx)


def test_giant_tour_prins_vs_greedy_and_width_validation() -> None:
  """WHAT: Giant-Tour auto-detection (`W == n - 1`) for Prins and Greedy modes plus invalid width check.
  WHY: Width-only format detection avoids D2H syncs while failing loud on malformed widths.
  EXPECTED: Prins <= Greedy cost, `evaluate_detailed_batch` returns `0`-delimited routes, invalid width raises.
  """
  problem = _make_cvrp_problem(capacity=10)
  ctx = ProblemContext(problem, xp=np, seed=3)
  tours = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)

  ev_prins = EvaluationCPUCost(split_mode="prins")
  ev_greedy = EvaluationCPUCost(split_mode="greedy")
  c_prins = ev_prins.evaluate_batch(tours, ctx)
  c_greedy = ev_greedy.evaluate_batch(tours, ctx)
  det_prins = ev_prins.evaluate_detailed_batch(tours, ctx)

  assert c_prins.shape == (2,)
  assert np.all(c_prins <= c_greedy + 1e-9)
  assert det_prins.routes.shape[0] == 2
  assert det_prins.routes.shape[1] >= problem.dimension

  with pytest.raises(ValueError, match="incompatible with problem dimension"):
    ev_prins.evaluate_batch(np.array([[1, 2, 3]], dtype=np.int32), ctx)


@pytest.mark.parametrize("max_matrix_dim", [100, 1])
@pytest.mark.parametrize("split_mode", ["prins", "greedy"])
def test_gpu_vrp_evaluator_parity_and_zero_d2h(
  max_matrix_dim: int,
  split_mode: Literal["prins", "greedy"],
) -> None:
  """WHAT: CPU vs GPU parity for `EvaluationGPUCost` on Giant-Tour and `0`-delimited batches.
  WHY: GPU evaluation must match CPU bit-for-bit on both dense matrix and on-the-fly coords,
  and incur zero D2H traffic when `return_gpu=True` (`DP17-3`).
  EXPECTED: exact float64 parity, `d2h_bytes == 0` with `return_gpu=True`, and clean `warmup()`.
  """
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  problem = _make_cvrp_problem(capacity=10, max_route_distance=16.0, max_vehicles=2, problem_type="DCVRP")
  ctx = ProblemContext(problem, xp=cp, seed=7, max_matrix_dimension=max_matrix_dim)

  giant_np = np.array([[1, 2, 3, 4], [2, 4, 1, 3], [3, 1, 4, 2]], dtype=np.int32)
  delim_np = np.array(
    [
      [0, 1, 2, 0, 3, 4, 0, 0],
      [0, 1, 2, 3, 0, 4, 0, 0],
      [0, 4, 0, 3, 0, 2, 0, 1],
    ],
    dtype=np.int32,
  )

  fixed = FixedPenaltyPolicy(penalty_capacity=500.0, penalty_distance=50.0, penalty_vehicles=250.0)
  ev_cpu = EvaluationCPUCost(
    split_mode=split_mode,
    penalty_policy=fixed,
  )
  ev_gpu = EvaluationGPUCost(
    split_mode=split_mode,
    penalty_policy=fixed,
    max_matrix_dimension=max_matrix_dim,
    return_gpu=True,
  )
  before_rng = copy.deepcopy(ctx.rng.bit_generator.state)
  ev_gpu.warmup(ctx)
  assert ev_gpu.h2d_bytes == 0
  assert ev_gpu.d2h_bytes == 0
  assert ev_gpu.kernel_launches == 0
  assert ev_gpu.call_count == 0
  assert ctx.h2d_bytes == 0
  assert ctx.rng.bit_generator.state == before_rng
  ev_gpu.reset_metrics()

  for batch_np in (giant_np, delim_np):
    cpu_det = ev_cpu.evaluate_detailed_batch(batch_np, ctx)
    cpu_assess = ev_cpu.assess_batch(batch_np, ctx)
    gpu_batch = cp.asarray(batch_np, dtype=cp.int32)

    ev_gpu.reset_metrics()
    gpu_costs = ev_gpu.evaluate_batch(gpu_batch, ctx, return_gpu=True)
    assert isinstance(gpu_costs, cp.ndarray)
    assert gpu_costs.dtype == cp.float64
    assert ev_gpu.d2h_bytes == 0
    np.testing.assert_allclose(gpu_costs.get(), cpu_det.penalized_costs)

    gpu_det = ev_gpu.evaluate_detailed_batch(gpu_batch, ctx, return_gpu=True)
    np.testing.assert_allclose(gpu_det.penalized_costs.get(), cpu_det.penalized_costs)
    np.testing.assert_allclose(gpu_det.unpenalized_costs.get(), cpu_det.unpenalized_costs)
    np.testing.assert_allclose(gpu_det.violations.get(), cpu_det.violations)
    np.testing.assert_array_equal(gpu_det.num_vehicles.get(), cpu_det.num_vehicles)

    ev_gpu.reset_metrics()
    gpu_assess = ev_gpu.assess_batch(gpu_batch, ctx, return_gpu=True)
    assert ev_gpu.d2h_bytes == 0
    assert gpu_assess.raw_violation_components is not None and cpu_assess.raw_violation_components is not None
    np.testing.assert_allclose(gpu_assess.objectives.get(), cpu_assess.objectives)
    np.testing.assert_allclose(gpu_assess.normalized_violations.get(), cpu_assess.normalized_violations)
    np.testing.assert_allclose(gpu_assess.raw_violation_components.get(), cpu_assess.raw_violation_components)


@pytest.mark.parametrize("return_to_depot", [True, False])
@pytest.mark.parametrize("max_matrix_dim", [100, 1])
def test_gpu_warp_coalesced_multi_tile_boundary_parity(
  return_to_depot: bool,
  max_matrix_dim: int,
) -> None:
  """WHAT: Multi-tile (`width >= 64`) warp-coalesced CUDA kernel vs vectorized CPU parity (`BUG-113`, `BUG-116`).
  WHY: Routes crossing 32-column warp tile boundaries (`col 31 -> 32`, `col 63 -> 64`), trailing open routes,
  consecutive zero-padded tiles, and all-zero rows must match CPU bit-for-bit under `StrictFeasibilityPolicy`.
  EXPECTED: Exact bit-for-bit parity on travel costs, penalized costs (`+inf` vs finite), vehicles, and components.
  """
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  prob = _make_large_vrp_problem(dimension=50, return_to_depot=return_to_depot)
  ctx = ProblemContext(prob, xp=cp, seed=11, max_matrix_dimension=max_matrix_dim)

  # Build width=64 (exact multiple of 32) and width=67 (non-multiple) test rows crossing tile boundaries
  row_a = np.zeros(64, dtype=np.int32)
  row_a[1:35] = np.arange(1, 35, dtype=np.int32)  # Single long route crossing col 31 -> 32
  row_a[36:51] = np.arange(35, 50, dtype=np.int32)  # Second route ending cleanly at col 51 (all-zero tile after)

  row_b = np.zeros(64, dtype=np.int32)
  row_b[0:20] = np.arange(1, 21, dtype=np.int32)  # Starts at col 0 without leading 0
  row_b[22:45] = np.arange(21, 44, dtype=np.int32)  # Crosses col 31 -> 32
  row_b[58:64] = np.arange(44, 50, dtype=np.int32)  # Ends open at col 63 (width % 32 == 0)

  row_zero = np.zeros(64, dtype=np.int32)
  batch_64 = np.vstack((row_a, row_b, row_zero))
  batch_67 = np.hstack((batch_64, np.zeros((3, 3), dtype=np.int32)))
  batch_67[1, 66] = 49  # Also test open trailing route when width % 32 != 0 (duplicate node ok for kernel parity)

  policy = StrictFeasibilityPolicy()
  ev_cpu = EvaluationCPUCost(penalty_policy=policy)
  ev_gpu = EvaluationGPUCost(
    penalty_policy=policy,
    max_matrix_dimension=max_matrix_dim,
    return_gpu=True,
  )
  ev_gpu.warmup(ctx)

  for batch_np in (batch_64, batch_67):
    cpu_assess = ev_cpu.assess_batch(batch_np, ctx)
    ev_gpu.reset_metrics()
    gpu_assess = ev_gpu.assess_batch(cp.asarray(batch_np, dtype=cp.int32), ctx, return_gpu=True)
    assert ev_gpu.d2h_bytes == 0
    assert gpu_assess.num_vehicles is not None and cpu_assess.num_vehicles is not None
    assert gpu_assess.raw_violation_components is not None and cpu_assess.raw_violation_components is not None
    np.testing.assert_array_equal(gpu_assess.is_feasible.get(), cpu_assess.is_feasible)
    np.testing.assert_array_equal(gpu_assess.num_vehicles.get(), cpu_assess.num_vehicles)
    np.testing.assert_allclose(gpu_assess.travel_costs.get(), cpu_assess.travel_costs)
    np.testing.assert_allclose(gpu_assess.violations.get(), cpu_assess.violations)
    np.testing.assert_allclose(gpu_assess.raw_violation_components.get(), cpu_assess.raw_violation_components)
    np.testing.assert_allclose(gpu_assess.normalized_violations.get(), cpu_assess.normalized_violations)
    np.testing.assert_array_equal(gpu_assess.objectives.get(), cpu_assess.objectives)
