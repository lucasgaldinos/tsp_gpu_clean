"""Unit & parity tests for `LocalSearch{CPU,GPU}RouteTwoOptStar` (`Tasks 17.8`, `17.9`, `17.24`)."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.strategies.decoding import GreedySplitDecoder, PrinsSplitDecoder, strip_depot_delimiters
from algorithms.strategies.local_search import LocalSearchCPURouteTwoOptStar, LocalSearchGPURouteTwoOptStar
from algorithms.strategies.local_search.two_opt_star_helpers import (
  RouteState,
  eval_relocate_delta,
  eval_swap_delta,
  eval_two_opt_star_delta,
  extract_routes_from_delimited,
)
from algorithms.strategies.penalties import InstanceScaledPenaltyPolicy, StrictFeasibilityPolicy
from data_models.problem import Problem
from data_models.vrp_constraints import VRPConstraints
from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext


def _cupy() -> object:
  """Return importable CuPy module or skip GPU test."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


def _make_vrp_problem(
  name: str,
  coords: np.ndarray,
  demands: np.ndarray,
  *,
  problem_type: str = "CVRP",
  capacity: int | None = None,
  max_vehicles: int | None = None,
  max_route_distance: float | None = None,
  return_to_depot: bool = True,
) -> Problem:
  """Build an `EUC_2D` VRP problem with explicit `VRPConstraints`."""
  constraints = VRPConstraints(
    capacity=capacity,
    max_vehicles=max_vehicles,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  return Problem(
    name=name,
    dimension=coords.shape[0],
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=coords.astype(np.float64),
    distances=None,
    capacity=capacity,
    demands=demands.astype(np.int64),
    constraints=constraints,
  )


def _full_penalized_cost(
  routes: list[list[int]],
  distances: np.ndarray,
  demands: np.ndarray,
  constraints: VRPConstraints,
  *,
  omega_q: float = 10_000.0,
  omega_l: float = 10_000.0,
  omega_k: float = 10_000.0,
) -> float:
  """Recompute exact penalized solution objective directly from routes."""
  non_empty = [r for r in routes if r]
  total_dist = 0
  cap_viol = 0
  dist_viol = 0.0
  for r in non_empty:
    load = int(demands[r].sum())
    d = int(distances[0, r[0]])
    for k in range(len(r) - 1):
      d += int(distances[r[k], r[k + 1]])
    if constraints.return_to_depot:
      d += int(distances[r[-1], 0])
    total_dist += d
    if constraints.capacity is not None and load > constraints.capacity:
      cap_viol += load - constraints.capacity
    if constraints.max_route_distance is not None and float(d) > constraints.max_route_distance:
      dist_viol += float(d) - constraints.max_route_distance
  veh_viol = (
    max(0, len(non_empty) - constraints.max_vehicles)
    if constraints.max_vehicles is not None
    else 0
  )
  cap_pen = (omega_q * float(cap_viol)) if cap_viol > 0 else 0.0
  dist_pen = (omega_l * dist_viol) if dist_viol > 0.0 else 0.0
  veh_pen = (omega_k * float(veh_viol)) if veh_viol > 0 else 0.0
  return float(total_dist) + cap_pen + dist_pen + veh_pen


@pytest.mark.parametrize(
  ("problem_type", "return_to_depot", "max_route_distance", "max_vehicles"),
  [
    ("CVRP", True, None, None),
    ("OVRP", False, None, None),
    ("DCVRP", True, 85.0, None),
    ("CVRP", True, None, 2),
  ],
)
def test_cpu_operators_exact_delta_matches_full_recomputation(
  problem_type: str,
  return_to_depot: bool,
  max_route_distance: float | None,
  max_vehicles: int | None,
) -> None:
  """WHAT: Exhaustively compare O(1) `Relocate`, `Swap`, and `2-opt*` deltas against full recomputation."""
  rng = np.random.default_rng(42)
  n = 9
  coords = rng.uniform(0.0, 40.0, size=(n, 2))
  coords[0] = [20.0, 20.0]
  demands = np.array([0, 3, 4, 2, 5, 3, 2, 4, 3], dtype=np.int64)
  problem = _make_vrp_problem(
    f"delta_{problem_type.lower()}",
    coords,
    demands,
    problem_type=problem_type,
    capacity=10,
    max_vehicles=max_vehicles,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  ctx = ProblemContext(problem, xp=np, seed=0)
  dist = ctx.get_cpu_distances()
  dem = ctx.get_cpu_demands()
  assert dem is not None

  routes = [[1, 2, 3], [4, 5], [6, 7, 8]]
  state = RouteState(routes, dist, dem, n, return_to_depot=return_to_depot)
  base_cost = _full_penalized_cost(routes, dist, dem, problem.constraints)
  omega_q, omega_l, omega_k = 10_000.0, 10_000.0, 10_000.0

  for r_a in range(len(routes)):
    for r_b in range(len(routes)):
      if r_a == r_b:
        continue
      s_a = len(routes[r_a])
      s_b = len(routes[r_b])

      for pos_v in range(s_a):
        for ins_pos in range(s_b + 1):
          delta = eval_relocate_delta(
            state,
            r_a,
            pos_v,
            r_b,
            ins_pos,
            capacity=problem.constraints.capacity,
            max_route_distance=max_route_distance,
            max_vehicles=max_vehicles,
            omega_q=omega_q,
            omega_l=omega_l,
            omega_k=omega_k,
          )
          trial = [list(r) for r in routes]
          trial[r_b].insert(ins_pos, trial[r_a].pop(pos_v))
          expected = _full_penalized_cost(trial, dist, dem, problem.constraints) - base_cost
          assert delta == pytest.approx(expected, abs=1e-9)

      for pos_v in range(s_a):
        for pos_u in range(s_b):
          delta = eval_swap_delta(
            state,
            r_a,
            pos_v,
            r_b,
            pos_u,
            capacity=problem.constraints.capacity,
            max_route_distance=max_route_distance,
            omega_q=omega_q,
            omega_l=omega_l,
          )
          trial = [list(r) for r in routes]
          trial[r_a][pos_v], trial[r_b][pos_u] = trial[r_b][pos_u], trial[r_a][pos_v]
          expected = _full_penalized_cost(trial, dist, dem, problem.constraints) - base_cost
          assert delta == pytest.approx(expected, abs=1e-9)

      for cut_a in range(-1, s_a):
        for cut_b in range(-1, s_b):
          delta = eval_two_opt_star_delta(
            state,
            r_a,
            cut_a,
            r_b,
            cut_b,
            capacity=problem.constraints.capacity,
            max_route_distance=max_route_distance,
            max_vehicles=max_vehicles,
            omega_q=omega_q,
            omega_l=omega_l,
            omega_k=omega_k,
          )
          trial = [list(r) for r in routes]
          trial[r_a] = routes[r_a][: cut_a + 1] + routes[r_b][cut_b + 1 :]
          trial[r_b] = routes[r_b][: cut_b + 1] + routes[r_a][cut_a + 1 :]
          expected = _full_penalized_cost(trial, dist, dem, problem.constraints) - base_cost
          assert delta == pytest.approx(expected, abs=1e-9)


def test_cpu_inter_route_improves_cross_route_suboptimal_and_policy_injection() -> None:
  """WHAT: `LocalSearchCPURouteTwoOptStar` escapes intra-route 2-opt optima, honoring `PenaltyPolicy`/`RouteDecoder`."""
  coords = np.array(
    [[0.0, 0.0], [10.0, 0.0], [11.0, 0.0], [-10.0, 0.0], [-11.0, 0.0]],
    dtype=np.float64,
  )
  demands = np.array([0, 5, 5, 5, 5], dtype=np.int64)
  problem = _make_vrp_problem("cross_routes", coords, demands, capacity=10)
  ctx = ProblemContext(problem, xp=np, seed=0)
  dist = ctx.get_cpu_distances()

  delimited = np.array([0, 1, 3, 0, 2, 4, 0, 0, 0], dtype=np.int32)
  before_cost = _full_penalized_cost(extract_routes_from_delimited(delimited), dist, demands, problem.constraints)
  assert before_cost == 84.0

  for mode in ("best", "multi_move"):
    ls = LocalSearchCPURouteTwoOptStar(mode=mode, max_iterations=10, candidate_k=None)
    assert isinstance(ls, LocalSearchStrategy)
    assert isinstance(ls.penalty_policy, InstanceScaledPenaltyPolicy)
    out = ls.improve(delimited, ctx)
    assert out.shape == delimited.shape
    after_routes = extract_routes_from_delimited(out)
    assert _full_penalized_cost(after_routes, dist, demands, problem.constraints) == 44.0
    assert sorted(strip_depot_delimiters(out, 4).tolist()) == [1, 2, 3, 4]

  # Custom PenaltyPolicy + RouteDecoder injection + StrictFeasibilityPolicy NaN safety
  dec = GreedySplitDecoder(use_gpu=False, penalty_policy=StrictFeasibilityPolicy())
  ls_strict = LocalSearchCPURouteTwoOptStar(penalty_policy=StrictFeasibilityPolicy(), decoder=dec, candidate_k=None)
  assert ls_strict._decoder is dec
  out_strict = ls_strict.improve(delimited, ctx)
  assert _full_penalized_cost(extract_routes_from_delimited(out_strict), dist, demands, problem.constraints) == 44.0


@pytest.mark.parametrize("mode", ["best", "multi_move"])
def test_cpu_inter_route_giant_tour_and_delimited_preserve_bijection_and_monotonicity(mode: str) -> None:
  """WHAT: Run `LocalSearchCPURouteTwoOptStar` on both Giant-Tour `(N, n-1)` and `0`-delimited `(N, W)` batches."""
  rng = np.random.default_rng(123)
  n = 15
  coords = rng.uniform(0.0, 100.0, size=(n, 2))
  demands = rng.integers(1, 5, size=n, dtype=np.int64)
  demands[0] = 0
  problem = _make_vrp_problem(
    "bijection_15",
    coords,
    demands,
    problem_type="DCVRP",
    capacity=10,
    max_route_distance=220.0,
  )
  ctx = ProblemContext(problem, xp=np, seed=0)
  dist = ctx.get_cpu_distances()

  giant_batch = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(4)])
  ls = LocalSearchCPURouteTwoOptStar(mode=mode, max_iterations=15, candidate_k=6)  # type: ignore[arg-type]
  wq, wl, wk = ls.penalty_policy.weights(ctx)

  gt_out = ls.improve_batch(giant_batch, ctx)
  assert gt_out.shape == giant_batch.shape
  for row in gt_out:
    assert sorted(row.tolist()) == list(range(1, n))

  split_res = PrinsSplitDecoder(use_gpu=False, penalty_policy=ls.penalty_policy).decode_batch(giant_batch, ctx)
  delim_in = split_res.routes
  delim_out = ls.improve_batch(delim_in, ctx)
  assert delim_out.shape == delim_in.shape
  for idx in range(len(delim_in)):
    r_in = extract_routes_from_delimited(delim_in[idx])
    r_out = extract_routes_from_delimited(delim_out[idx])
    assert sorted(x for r in r_out for x in r) == list(range(1, n))
    assert _full_penalized_cost(
      r_out, dist, demands, problem.constraints, omega_q=wq, omega_l=wl, omega_k=wk
    ) <= _full_penalized_cost(r_in, dist, demands, problem.constraints, omega_q=wq, omega_l=wl, omega_k=wk)


@pytest.mark.gpu
@pytest.mark.parametrize(
  ("problem_type", "return_to_depot", "max_route_distance", "max_vehicles"),
  [
    ("CVRP", True, None, None),
    ("OVRP", False, None, None),
    ("DCVRP", True, 180.0, None),
    ("CVRP", True, None, 4),
  ],
)
@pytest.mark.parametrize("max_shared_dimension", [512, 4])
def test_gpu_batch_matches_cpu_best_across_variants_coords_and_global_fallback(
  problem_type: str,
  return_to_depot: bool,
  max_route_distance: float | None,
  max_vehicles: int | None,
  max_shared_dimension: int,
) -> None:
  """WHAT: Exact parity of GPU `mode="batch"` vs CPU `mode="best"` `LocalSearch*RouteTwoOptStar`."""
  from typing import Any

  cp: Any = _cupy()
  rng = np.random.default_rng(77)
  n = 12
  coords = rng.uniform(0.0, 60.0, size=(n, 2))
  coords[0] = [30.0, 30.0]
  demands = np.array([0, 3, 2, 4, 3, 5, 2, 3, 4, 2, 3, 4], dtype=np.int64)
  problem = _make_vrp_problem(
    f"gpu_parity_{problem_type.lower()}",
    coords,
    demands,
    problem_type=problem_type,
    capacity=10,
    max_vehicles=max_vehicles,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  cpu_ctx = ProblemContext(problem, xp=np, seed=1)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)

  tours = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(3)])
  delim_in = PrinsSplitDecoder(use_gpu=False).decode_batch(tours, cpu_ctx).routes

  for cand_k in (None, 5):
    cpu_ls = LocalSearchCPURouteTwoOptStar(mode="best", max_iterations=10, candidate_k=cand_k, intra_two_opt=True)
    cpu_out = cpu_ls.improve_batch(delim_in, cpu_ctx)

    for max_mat in (None, 3):
      gpu_ls = LocalSearchGPURouteTwoOptStar(
        mode="batch",
        max_iterations=10,
        candidate_k=cand_k,
        intra_two_opt=True,
        max_matrix_dimension=max_mat,
        max_shared_dimension=max_shared_dimension,
      )
      gpu_out = np.asarray(gpu_ls.improve_batch(delim_in, gpu_ctx))
      np.testing.assert_array_equal(gpu_out, cpu_out)


@pytest.mark.gpu
def test_gpu_multi_move_over_32_routes_uses_multi_word_bitmask_and_preserves_constraints() -> None:
  """WHAT: `LocalSearchGPURouteTwoOptStar(mode="multi_move")` on an instance with $m = 44 > 32$ active routes (`N2`)."""
  from typing import Any

  cp: Any = _cupy()
  rng = np.random.default_rng(314)
  n = 45
  coords = rng.uniform(0.0, 100.0, size=(n, 2))
  coords[0] = [50.0, 50.0]
  demands = np.ones(n, dtype=np.int64)
  demands[0] = 0
  problem = _make_vrp_problem("mask_45", coords, demands, capacity=3)
  cpu_ctx = ProblemContext(problem, xp=np, seed=2)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=2)
  dist = cpu_ctx.get_cpu_distances()

  row = np.zeros(n + (n - 1), dtype=np.int32)
  for idx in range(1, n):
    row[2 * idx - 1] = idx
  batch = np.tile(row, (2, 1))

  before_cost = _full_penalized_cost(extract_routes_from_delimited(row), dist, demands, problem.constraints)
  gpu_ls = LocalSearchGPURouteTwoOptStar(mode="multi_move", max_iterations=15, candidate_k=10, intra_two_opt=True)
  assert isinstance(gpu_ls, LocalSearchStrategy)
  out = np.asarray(gpu_ls.improve_batch(batch, gpu_ctx))

  assert out.shape == batch.shape
  for r_idx in range(2):
    routes = extract_routes_from_delimited(out[r_idx])
    assert sorted(x for r in routes for x in r) == list(range(1, n))
    for r in routes:
      assert int(demands[r].sum()) <= 3
    assert _full_penalized_cost(routes, dist, demands, problem.constraints) < before_cost


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", ["batch", "multi_move"])
def test_gpu_inter_route_zero_d2h_on_cupy_inputs_and_warmup(gpu_mode: str) -> None:
  """WHAT: CuPy-resident Giant-Tour & `0`-delimited inputs + `warmup(context)` + `StrictFeasibilityPolicy`."""
  from typing import Any

  cp: Any = _cupy()
  coords = np.array([[0.0, 0.0], [10.0, 0.0], [11.0, 0.0], [-10.0, 0.0], [-11.0, 0.0]], dtype=np.float64)
  demands = np.array([0, 5, 5, 5, 5], dtype=np.int64)
  problem = _make_vrp_problem("gpu_d2h_5", coords, demands, capacity=10)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=0)
  dist = gpu_ctx.get_cpu_distances()

  ls = LocalSearchGPURouteTwoOptStar(
    mode=gpu_mode,  # type: ignore[arg-type]
    max_iterations=10,
    candidate_k=None,
    penalty_policy=StrictFeasibilityPolicy(),
  )
  ls.warmup(gpu_ctx)
  assert ls._kernel is not None
  assert (ls.kernel_launches, ls.h2d_bytes, ls.d2h_bytes) == (0, 0, 0)

  ls.reset_metrics()
  delim_gpu = cp.asarray(np.array([[0, 1, 3, 0, 2, 4, 0, 0, 0]], dtype=np.int32))
  delim_out = ls.improve_batch(delim_gpu, gpu_ctx)
  assert ls.d2h_bytes == 0
  assert isinstance(delim_out, cp.ndarray)
  out_routes = extract_routes_from_delimited(cp.asnumpy(delim_out)[0])
  assert _full_penalized_cost(out_routes, dist, demands, problem.constraints) == 44.0

  ls.reset_metrics()
  gt_gpu = cp.asarray(np.array([[1, 3, 2, 4]], dtype=np.int32))
  gt_out = ls.improve_batch(gt_gpu, gpu_ctx)
  assert ls.d2h_bytes == 0
  assert isinstance(gt_out, cp.ndarray)
  assert gt_out.shape == (1, 4)
  assert sorted(cp.asnumpy(gt_out)[0].tolist()) == [1, 2, 3, 4]
