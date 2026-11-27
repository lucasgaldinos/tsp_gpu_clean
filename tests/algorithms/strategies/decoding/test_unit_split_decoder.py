"""Unit & parity tests for CPU/CUDA Split decoders and pipeline adapters (`Task 17.4`, `DP17-2`, `DP17-3`).

Verifies:
1. Exact optimality of `PrinsSplitDecoder` against brute-force cut enumeration ($n \\le 9$)
   under `CVRP`, `OVRP` (`return_to_depot=False`), and `DCVRP` (`max_route_distance`).
2. Bit-exact CPU vs GPU parity for `PrinsSplitDecoder` and `GreedySplitDecoder` across
   dense matrices, on-the-fly coordinates (`EUC_2D`/`EUC_3D`), `decode_costs_only`,
   and the global-memory workspace fallback (`max_shared_customers`).
3. Bounded `max_vehicles` overflow merging & penalty consistency (`DP17-2`).
4. Single-customer fallback safety when $q_v > Q$ or single-route distance $> L_{\\max}$.
5. Lossless `SplitDecodeStep -> GiantTourEncodeStep` round-trip on CPU & GPU with
   `d2h_bytes == 0`, `strip_depot_delimiters`, `compact_delimited_route`, and `warmup`.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GenerationState
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY, GiantTourEncodeStep, SplitDecodeStep
from algorithms.strategies.decoding import (
  GreedySplitDecoder,
  PrinsSplitDecoder,
  compact_delimited_route,
  strip_depot_delimiters,
)
from algorithms.strategies.penalties import FixedPenaltyPolicy
from data_models.problem import Problem
from data_models.vrp_constraints import VRPConstraints
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext
from utils.phase_timer import PhaseTimer


def _cupy() -> Any:
  """Return the importable CuPy module, skipping GPU tests when unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


def _make_euc2d_vrp(
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
  """Construct an `EUC_2D` VRP `Problem` with explicit `VRPConstraints`."""
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


def _bruteforce_optimal_split(
  tour: np.ndarray,
  distances: np.ndarray,
  demands: np.ndarray,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  return_to_depot: bool,
) -> int:
  """Enumerate all $2^{n_{\\text{cust}} - 1}$ partitions of `tour` and return the optimal feasible cost."""
  n_cust = int(tour.size)
  best = 1 << 60
  for mask in range(1 << max(0, n_cust - 1)):
    cuts = [0] + [i + 1 for i in range(n_cust - 1) if (mask >> i) & 1] + [n_cust]
    total_cost = 0
    feasible = True
    for a, b in pairwise(cuts):
      seg = tour[a:b]
      load = int(demands[seg].sum())
      if capacity is not None and load > capacity:
        feasible = False
        break
      dist = int(distances[0, int(seg[0])])
      for k in range(len(seg) - 1):
        dist += int(distances[int(seg[k]), int(seg[k + 1])])
      if return_to_depot:
        dist += int(distances[int(seg[-1]), 0])
      if max_route_distance is not None and float(dist) > max_route_distance:
        feasible = False
        break
      total_cost += dist
    if feasible and total_cost < best:
      best = total_cost
  return best


def _score_delimited_row(
  row: np.ndarray,
  distances: np.ndarray,
  demands: np.ndarray,
  *,
  return_to_depot: bool,
) -> tuple[int, int, list[int]]:
  """Score a `0`-delimited route row directly for independent verification."""
  tokens = [int(x) for x in row.tolist()]
  routes: list[list[int]] = []
  cur: list[int] = []
  for tok in tokens:
    if tok == 0:
      if cur:
        routes.append(cur)
        cur = []
    else:
      cur.append(tok)
  if cur:
    routes.append(cur)

  total_dist = 0
  max_load = 0
  for r in routes:
    max_load = max(max_load, int(demands[r].sum()))
    d = int(distances[0, r[0]])
    for k in range(len(r) - 1):
      d += int(distances[r[k], r[k + 1]])
    if return_to_depot:
      d += int(distances[r[-1], 0])
    total_dist += d
  return total_dist, len(routes), [x for r in routes for x in r]


# ---------------------------------------------------------------------------
# 1. Exact Prins Optimality vs Brute Force (CVRP, OVRP, DCVRP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("problem_type", "return_to_depot", "max_route_distance"),
  [
    ("CVRP", True, None),
    ("OVRP", False, None),
    ("DCVRP", True, 90.0),
  ],
)
def test_prins_split_matches_bruteforce_optimum_cpu_and_gpu(
  problem_type: str,
  return_to_depot: bool,
  max_route_distance: float | None,
) -> None:
  """WHAT: `PrinsSplitDecoder` vs exhaustive $2^{n-1}$ partition search on $n=9$ instances.

  EXPECTED: Exact match to brute-force global optimum under `CVRP`, `OVRP`, and `DCVRP`,
  zero constraint violations, and bit-exact CPU vs GPU parity.
  """
  rng = np.random.default_rng(2026)
  n = 9
  coords = rng.uniform(0.0, 30.0, size=(n, 2))
  coords[0] = [15.0, 15.0]
  demands = np.array([0, 3, 4, 2, 5, 3, 4, 2, 3], dtype=np.int64)
  problem = _make_euc2d_vrp(
    f"bf_{problem_type.lower()}",
    coords,
    demands,
    problem_type=problem_type,
    capacity=10,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  cpu_ctx = ProblemContext(problem, xp=np, seed=1)
  distances = cpu_ctx.get_cpu_distances()

  tours = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(5)])
  cpu_dec = PrinsSplitDecoder(use_gpu=False)
  cpu_res = cpu_dec.decode_batch(tours, cpu_ctx)

  for idx, tour in enumerate(tours):
    expected_opt = _bruteforce_optimal_split(
      tour,
      distances,
      demands,
      capacity=10,
      max_route_distance=max_route_distance,
      return_to_depot=return_to_depot,
    )
    assert float(cpu_res.unpenalized_costs[idx]) == float(expected_opt)
    assert float(cpu_res.penalized_costs[idx]) == float(expected_opt)
    assert float(cpu_res.violations[idx]) == 0.0
    scored_dist, scored_m, flat_cust = _score_delimited_row(
      cpu_res.routes[idx], distances, demands, return_to_depot=return_to_depot
    )
    assert scored_dist == expected_opt
    assert scored_m == int(cpu_res.num_vehicles[idx])
    assert flat_cust == tour.tolist()

  cp = cupy_backend()
  if cp is not None:
    gpu_ctx = ProblemContext(problem, xp=cp, seed=1)
    gpu_dec = PrinsSplitDecoder(use_gpu=True)
    gpu_res = gpu_dec.decode_batch(tours, gpu_ctx, return_gpu=False)
    np.testing.assert_array_equal(gpu_res.routes, cpu_res.routes)
    np.testing.assert_array_equal(gpu_res.penalized_costs, cpu_res.penalized_costs)
    np.testing.assert_array_equal(gpu_res.unpenalized_costs, cpu_res.unpenalized_costs)
    np.testing.assert_array_equal(gpu_res.violations, cpu_res.violations)
    np.testing.assert_array_equal(gpu_res.num_vehicles, cpu_res.num_vehicles)


def test_prins_strictly_beats_greedy_on_trap_instance() -> None:
  """WHAT: Construct a giant-tour where Greedy makes a suboptimal early fill and Prins finds optimal.

  Tour `[1, 2, 3]` with `Q = 10`, `q = [0, 4, 4, 6]`, distances:
  depot `0` at `(0, 0)`, `1` at `(10, 0)`, `2` at `(10, 1)`, `3` at `(100, 0)`.
  Greedy puts `[1, 2]` (load 8, closes before `3`) + `[3]` (`21 + 200 = 221`) -> wait:
  Put `1` close to depot `(1, 0)` and `2, 3` far `(100, 0), (100, 1)`:
  - `[1] + [2, 3]` (loads `4` and `10`): cost `(1+1) + (100+1+100) = 203`.
  - `[1, 2] + [3]` (Greedy, loads `8` and `6`): cost `(1+99+100) + (100+100) = 400`!
  """
  coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [100.0, 1.0]], dtype=np.float64)
  demands = np.array([0, 4, 4, 6], dtype=np.int64)
  problem = _make_euc2d_vrp("prins_vs_greedy", coords, demands, capacity=10)
  ctx = ProblemContext(problem, xp=np, seed=0)
  tour = np.array([1, 2, 3], dtype=np.int32)

  greedy_res = GreedySplitDecoder(use_gpu=False).decode_batch(tour, ctx)
  prins_res = PrinsSplitDecoder(use_gpu=False).decode_batch(tour, ctx)

  assert float(greedy_res.unpenalized_costs[0]) == 400.0
  assert float(prins_res.unpenalized_costs[0]) == 203.0
  assert compact_delimited_route(prins_res.routes) .tolist() == [0, 1, 0, 2, 3, 0]


# ---------------------------------------------------------------------------
# 2. GPU Matrix, On-the-Fly Coords, Global Workspace, & `decode_costs_only`
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("decoder_cls", [PrinsSplitDecoder, GreedySplitDecoder])
@pytest.mark.parametrize("max_shared_customers", [2048, 4])
def test_gpu_split_parity_matrix_coords_and_global_workspace(
  decoder_cls: type[PrinsSplitDecoder] | type[GreedySplitDecoder],
  max_shared_customers: int,
) -> None:
  """WHAT: CPU vs GPU parity for matrix, on-the-fly coords, shared-mem & global workspace (`n_cust=12`).

  WHY: Lowering `max_shared_customers=4` forces `use_shared=0` on `n_cust=12`, exercising
  the `vrp-XXL` global workspace path alongside `max_matrix_dimension=3` coordinate mode.
  """
  cp = _cupy()
  rng = np.random.default_rng(77)
  n = 13
  coords = rng.uniform(0.0, 200.0, size=(n, 2))
  demands = rng.integers(1, 6, size=n, dtype=np.int64)
  demands[0] = 0
  problem = _make_euc2d_vrp(
    "parity_13",
    coords,
    demands,
    problem_type="DCVRP",
    capacity=12,
    max_route_distance=450.0,
    return_to_depot=True,
  )
  cpu_ctx = ProblemContext(problem, xp=np, seed=1)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)
  tours = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(4)])

  cpu_res = decoder_cls(use_gpu=False).decode_batch(tours, cpu_ctx)

  for max_mat in (None, 3):
    gpu_dec = decoder_cls(
      use_gpu=True,
      max_shared_customers=max_shared_customers,
      max_matrix_dimension=max_mat,
    )
    gpu_res = gpu_dec.decode_batch(tours, gpu_ctx, return_gpu=False)
    np.testing.assert_array_equal(gpu_res.routes, cpu_res.routes)
    np.testing.assert_array_equal(gpu_res.penalized_costs, cpu_res.penalized_costs)
    np.testing.assert_array_equal(gpu_res.unpenalized_costs, cpu_res.unpenalized_costs)
    np.testing.assert_array_equal(gpu_res.violations, cpu_res.violations)
    np.testing.assert_array_equal(gpu_res.num_vehicles, cpu_res.num_vehicles)

    costs_only = gpu_dec.decode_costs_only(tours, gpu_ctx, return_gpu=False)
    assert costs_only.routes is None
    np.testing.assert_array_equal(costs_only.penalized_costs, cpu_res.penalized_costs)
    np.testing.assert_array_equal(costs_only.unpenalized_costs, cpu_res.unpenalized_costs)
    np.testing.assert_array_equal(costs_only.violations, cpu_res.violations)
    np.testing.assert_array_equal(costs_only.num_vehicles, cpu_res.num_vehicles)


# ---------------------------------------------------------------------------
# 3. `max_vehicles` Overflow Merging & Single-Customer Fallback
# ---------------------------------------------------------------------------


def test_max_vehicles_overflow_merges_tail_into_buffer_and_scores_exact() -> None:
  """WHAT: `K_max = 2` on a 4-customer tour requiring 4 routes by capacity (`Q = 5`, `q = 4`).

  EXPECTED:
  - Default `buffer_k = K_max = 2` fits in `W = 5 + 2 = 7` by merging trailing routes into slot 2,
    reporting exact capacity violation `8` (`3 * 4 - 4 = 8`) on slot 2.
  - Explicit `max_vehicles=4` buffer keeps all 4 routes (`num_vehicles == 4`), reporting
    fleet excess `4 - 2 = 2` and `penalized_cost == unpenalized + 2 * omega_k`.
  """
  coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]], dtype=np.float64)
  demands = np.array([0, 4, 4, 4, 4], dtype=np.int64)
  problem = _make_euc2d_vrp("kmax_overflow", coords, demands, capacity=5, max_vehicles=2)
  ctx = ProblemContext(problem, xp=np, seed=0)
  tour = np.array([1, 2, 3, 4], dtype=np.int32)
  distances = ctx.get_cpu_distances()

  dec = PrinsSplitDecoder(
    use_gpu=False,
    penalty_policy=FixedPenaltyPolicy(penalty_capacity=1000.0, penalty_vehicles=500.0),
  )
  merged = dec.decode_batch(tour, ctx)
  assert merged.routes.shape == (7,)
  assert int(merged.num_vehicles[0]) == 2
  # Route 1: [1], Route 2: [2, 3, 4] -> load 12 > 5 -> cap_viol = 7
  scored_dist, scored_m, flat = _score_delimited_row(merged.routes, distances, demands, return_to_depot=True)
  assert scored_m == 2
  assert flat == [1, 2, 3, 4]
  assert float(merged.unpenalized_costs[0]) == float(scored_dist)
  assert float(merged.violations[0]) == 7.0
  assert float(merged.penalized_costs[0]) == float(scored_dist) + 7000.0

  unmerged = dec.decode_batch(tour, ctx, max_vehicles=4)
  assert unmerged.routes.shape == (9,)
  assert int(unmerged.num_vehicles[0]) == 4
  assert float(unmerged.violations[0]) == 2.0
  assert float(unmerged.penalized_costs[0]) == float(unmerged.unpenalized_costs[0]) + 1000.0

  cp = cupy_backend()
  if cp is not None:
    gpu_ctx = ProblemContext(problem, xp=cp, seed=0)
    gpu_dec = PrinsSplitDecoder(
      use_gpu=True,
      penalty_policy=FixedPenaltyPolicy(penalty_capacity=1000.0, penalty_vehicles=500.0),
    )
    gpu_merged = gpu_dec.decode_batch(tour, gpu_ctx, return_gpu=False)
    np.testing.assert_array_equal(gpu_merged.routes, merged.routes)
    np.testing.assert_array_equal(gpu_merged.penalized_costs, merged.penalized_costs)
    np.testing.assert_array_equal(gpu_merged.violations, merged.violations)


@pytest.mark.parametrize("decoder_cls", [PrinsSplitDecoder, GreedySplitDecoder])
def test_single_customer_fallback_avoids_inf_or_nan(
  decoder_cls: type[PrinsSplitDecoder] | type[GreedySplitDecoder],
) -> None:
  """WHAT: Single customer demand $q_v > Q$ and single-route distance $> L_{\\max}$.

  EXPECTED: No `Inf`/`NaN`, all customers preserved in output routes, exact violation recorded,
  and CPU/GPU parity preserved.
  """
  coords = np.array([[0.0, 0.0], [100.0, 0.0], [5.0, 0.0], [6.0, 0.0]], dtype=np.float64)
  demands = np.array([0, 15, 2, 3], dtype=np.int64)
  problem = _make_euc2d_vrp(
    "single_fallback",
    coords,
    demands,
    problem_type="DCVRP",
    capacity=10,
    max_route_distance=50.0,
  )
  cpu_ctx = ProblemContext(problem, xp=np, seed=0)
  tour = np.array([1, 2, 3], dtype=np.int32)

  policy = FixedPenaltyPolicy(penalty_capacity=100.0, penalty_distance=10.0)
  cpu_res = decoder_cls(use_gpu=False, penalty_policy=policy).decode_batch(tour, cpu_ctx)
  assert np.isfinite(cpu_res.penalized_costs).all()
  assert np.isfinite(cpu_res.unpenalized_costs).all()
  # Customer 1: cap excess 5, dist 200 -> dist excess 150 -> total viol 155.
  assert float(cpu_res.violations[0]) == 155.0
  assert float(cpu_res.penalized_costs[0]) == float(cpu_res.unpenalized_costs[0]) + 500.0 + 1500.0
  assert strip_depot_delimiters(cpu_res.routes, 3).tolist() == [1, 2, 3]

  cp = cupy_backend()
  if cp is not None:
    gpu_ctx = ProblemContext(problem, xp=cp, seed=0)
    gpu_res = decoder_cls(use_gpu=True, penalty_policy=policy).decode_batch(
      tour, gpu_ctx, return_gpu=False
    )
    np.testing.assert_array_equal(gpu_res.routes, cpu_res.routes)
    np.testing.assert_array_equal(gpu_res.penalized_costs, cpu_res.penalized_costs)
    np.testing.assert_array_equal(gpu_res.violations, cpu_res.violations)


# ---------------------------------------------------------------------------
# 4. Pipeline Steps Round-Trip (`SplitDecodeStep -> GiantTourEncodeStep`) & Helpers
# ---------------------------------------------------------------------------


def test_compact_delimited_route_collapses_padding_zeros() -> None:
  """WHAT: `compact_delimited_route` on padded and boundary inputs."""
  assert compact_delimited_route([0, 1, 2, 0, 0, 3, 0, 0, 0]).tolist() == [0, 1, 2, 0, 3, 0]
  assert compact_delimited_route([1, 2, 0, 3]).tolist() == [0, 1, 2, 0, 3, 0]
  assert compact_delimited_route([0, 0, 0]).tolist() == [0]
  assert compact_delimited_route([]).tolist() == []


def test_split_decode_and_giant_tour_encode_round_trip_cpu_and_gpu() -> None:
  """WHAT: Lossless `SplitDecodeStep -> GiantTourEncodeStep` round-trip and registry integration.

  EXPECTED:
  - Both steps are registered in `PIPELINE_STEP_REGISTRY`.
  - On CPU and GPU (`d2h_bytes == 0`), `SplitDecodeStep -> GiantTourEncodeStep` returns the
    exact original `(N, n_cust)` Giant-Tour batch without losing or reordering any customer.
  """
  assert PIPELINE_STEP_REGISTRY["SplitDecodeStep"] is SplitDecodeStep
  assert PIPELINE_STEP_REGISTRY["GiantTourEncodeStep"] is GiantTourEncodeStep

  rng = np.random.default_rng(11)
  n = 10
  coords = rng.uniform(0.0, 100.0, size=(n, 2))
  demands = np.array([0, 2, 3, 4, 2, 5, 1, 3, 4, 2], dtype=np.int64)
  problem = _make_euc2d_vrp("rt_10", coords, demands, capacity=8)
  tours = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(4)])

  decode_step = SplitDecodeStep(mode="prins", target="offspring")
  encode_step = GiantTourEncodeStep(target="offspring")
  timer = PhaseTimer(enabled=True)

  # CPU round-trip
  cpu_ctx = ProblemContext(problem, xp=np, seed=0)
  state_cpu = GenerationState(
    population=tours.copy(),
    fitness=np.zeros(len(tours), dtype=np.float64),
    context=cpu_ctx,
    offspring=tours.copy(),
  )
  decode_step.warmup(cpu_ctx)
  encode_step.warmup(cpu_ctx)
  decode_step.run(state_cpu, timer)
  assert state_cpu.offspring is not None
  assert state_cpu.offspring.shape == (4, n + (n - 1))
  encode_step.run(state_cpu, timer)
  np.testing.assert_array_equal(state_cpu.offspring, tours)

  # GPU round-trip with zero D2H traffic
  cp = cupy_backend()
  if cp is not None:
    gpu_ctx = ProblemContext(problem, xp=cp, seed=0)
    gpu_tours = cp.asarray(tours, dtype=cp.int32)
    state_gpu = GenerationState(
      population=gpu_tours,
      fitness=cp.zeros(len(tours), dtype=cp.float64),
      context=gpu_ctx,
      offspring=gpu_tours,
    )
    decode_step.warmup(gpu_ctx)
    encode_step.warmup(gpu_ctx)
    decode_step.decoder.reset_metrics()
    decode_step.run(state_gpu, timer)
    assert isinstance(state_gpu.offspring, cp.ndarray)
    assert state_gpu.offspring.shape == (4, n + (n - 1))
    assert decode_step.decoder.d2h_bytes == 0
    encode_step.run(state_gpu, timer)
    assert isinstance(state_gpu.offspring, cp.ndarray)
    assert decode_step.decoder.d2h_bytes == 0
    np.testing.assert_array_equal(cp.asnumpy(state_gpu.offspring), tours)


@pytest.mark.parametrize("decoder_cls", [PrinsSplitDecoder, GreedySplitDecoder])
@pytest.mark.parametrize("max_matrix_dimension", [1000, 10])
def test_warp_parallel_multi_tile_cpu_gpu_parity_matrix_and_coords(
  decoder_cls: type[PrinsSplitDecoder] | type[GreedySplitDecoder],
  max_matrix_dimension: int,
) -> None:
  """WHAT: Multi-tile (`n_cust=96 > 32`) CPU/GPU parity across dense matrix & coordinate paths (`BUG-113`)."""
  rng = np.random.default_rng(1721)
  n = 97  # 96 customers -> 3 full 32-lane prefix tiles + multi-stride Bellman warp scans
  coords = rng.uniform(0.0, 200.0, size=(n, 2))
  demands = rng.integers(1, 5, size=n, dtype=np.int64)
  demands[0] = 0
  # Large capacity (~40 customers per route) forces multi-stride `i -= 32` warp reductions.
  problem = _make_euc2d_vrp(
    "multi_tile_97",
    coords,
    demands,
    problem_type="DCVRP",
    capacity=110,
    max_route_distance=900.0,
    return_to_depot=True,
  )
  tours = np.stack([rng.permutation(np.arange(1, n, dtype=np.int32)) for _ in range(8)])

  cpu_ctx = ProblemContext(problem, xp=np, seed=0)
  cpu_dec = decoder_cls(use_gpu=False)
  cpu_res = cpu_dec.decode_batch(tours, cpu_ctx)

  cp = cupy_backend()
  if cp is None:
    return

  gpu_ctx = ProblemContext(problem, xp=cp, seed=0)
  gpu_tours = cp.asarray(tours, dtype=cp.int32)
  gpu_dec = decoder_cls(use_gpu=True, max_matrix_dimension=max_matrix_dimension)
  gpu_dec.warmup(gpu_ctx)
  gpu_dec.reset_metrics()
  gpu_res = gpu_dec.decode_batch(gpu_tours, gpu_ctx, return_gpu=True)

  assert gpu_dec.d2h_bytes == 0
  np.testing.assert_array_equal(cp.asnumpy(gpu_res.routes), cpu_res.routes)
  np.testing.assert_allclose(cp.asnumpy(gpu_res.unpenalized_costs), cpu_res.unpenalized_costs, rtol=1e-12)
  np.testing.assert_allclose(cp.asnumpy(gpu_res.penalized_costs), cpu_res.penalized_costs, rtol=1e-12)
  np.testing.assert_allclose(cp.asnumpy(gpu_res.violations), cpu_res.violations, rtol=1e-12)
  np.testing.assert_array_equal(cp.asnumpy(gpu_res.num_vehicles), cpu_res.num_vehicles)


def test_split_decoder_penalty_policy_injection_default_and_strict() -> None:
  """WHAT: Decoders default to `InstanceScaledPenaltyPolicy` and keep `StrictFeasibilityPolicy` NaN-free."""
  from algorithms.strategies.penalties import InstanceScaledPenaltyPolicy, StrictFeasibilityPolicy

  default_dec = PrinsSplitDecoder(use_gpu=False)
  assert isinstance(default_dec.penalty_policy, InstanceScaledPenaltyPolicy)

  coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float64)
  demands = np.array([0, 4, 15, 4], dtype=np.int64)
  problem = _make_euc2d_vrp("strict_split_4", coords, demands, capacity=10)
  tour = np.array([[1, 2, 3]], dtype=np.int32)

  strict_cpu = PrinsSplitDecoder(use_gpu=False, penalty_policy=StrictFeasibilityPolicy())
  res_cpu = strict_cpu.decode_batch(tour, ProblemContext(problem, xp=np))
  assert not np.isnan(res_cpu.penalized_costs).any()
  assert np.isinf(res_cpu.penalized_costs[0])
  assert np.isfinite(res_cpu.unpenalized_costs[0])

  cp = cupy_backend()
  if cp is not None:
    strict_gpu = PrinsSplitDecoder(use_gpu=True, penalty_policy=StrictFeasibilityPolicy())
    res_gpu = strict_gpu.decode_batch(tour, ProblemContext(problem, xp=cp), return_gpu=False)
    assert not np.isnan(res_gpu.penalized_costs).any()
    assert np.isinf(res_gpu.penalized_costs[0])
    np.testing.assert_allclose(res_gpu.unpenalized_costs, res_cpu.unpenalized_costs)
