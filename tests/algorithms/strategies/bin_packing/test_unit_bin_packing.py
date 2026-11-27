"""Unit tests for CPU/GPU FFD & BFD bin-packing strategies and `FleetBinPackStep` (`Task 17.30`, `BUG-117`)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GenerationState
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY, FleetBinPackStep
from algorithms.strategies import SUB_STRATEGY_REGISTRY
from algorithms.strategies.bin_packing import (
  BinPackingCPUBFD,
  BinPackingCPUFFD,
  BinPackingGPUBFD,
  BinPackingGPUFFD,
)
from data_models.problem import Problem
from data_models.vrp_constraints import VRPConstraints
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext
from protocols.vrp_strategies import BinPackingStrategy
from utils.gpu_helpers import _compile_kernel_cached
from utils.phase_timer import PhaseTimer


def _make_vrp_problem(
  *,
  demands: np.ndarray | None = None,
  coords: np.ndarray | None = None,
  max_vehicles: int | None = 2,
  capacity: int | None = 10,
  max_route_distance: float | None = None,
  problem_type: str = "CVRP",
  return_to_depot: bool | None = None,
) -> Problem:
  if coords is None:
    coords = np.array(
      [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0],
        [0.0, 4.0],
      ],
      dtype=np.float64,
    )
  if demands is None:
    demands = np.array([0, 4, 3, 5, 2], dtype=np.int32)
  rtd = (problem_type != "OVRP") if return_to_depot is None else return_to_depot
  return Problem(
    name=f"bp_{problem_type.lower()}_{coords.shape[0]}",
    dimension=int(coords.shape[0]),
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=capacity,
    demands=demands,
    constraints=VRPConstraints(
      capacity=capacity,
      max_vehicles=max_vehicles,
      max_route_distance=max_route_distance,
      return_to_depot=rtd,
    ),
  )


def _extract_subroutes(row: np.ndarray) -> list[list[int]]:
  routes: list[list[int]] = []
  cur: list[int] = []
  for val in row.tolist():
    node = int(val)
    if node == 0:
      if cur:
        routes.append(cur)
        cur = []
    else:
      cur.append(node)
  if cur:
    routes.append(cur)
  return routes


def _extract_customers(row: np.ndarray) -> list[int]:
  return sorted(c for r in _extract_subroutes(row) for c in r)


@pytest.mark.parametrize(
  "cls",
  [BinPackingCPUFFD, BinPackingCPUBFD, BinPackingGPUFFD, BinPackingGPUBFD],
)
def test_registry_and_o1_self_gating(cls: type[Any], monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: All 4 bin-packing strategies implement `BinPackingStrategy`, register cleanly, and self-gate in O(1)."""
  if cls in (BinPackingGPUFFD, BinPackingGPUBFD) and not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")

  packer = cls()
  assert isinstance(packer, BinPackingStrategy)
  assert SUB_STRATEGY_REGISTRY[cls.__name__] is cls
  assert "FleetBinPackStep" in PIPELINE_STEP_REGISTRY

  tsp = Problem(
    name="tsp5",
    dimension=5,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.arange(10, dtype=np.float64).reshape(5, 2),
    distances=None,
  )
  ctx_tsp = ProblemContext(tsp, xp=np)
  monkeypatch.setattr(ctx_tsp, "get_cpu_distances", lambda: pytest.fail("TSP must skip distance access"))
  monkeypatch.setattr(ctx_tsp, "get_gpu_distances", lambda: pytest.fail("TSP must skip GPU distance access"))
  tsp_batch = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
  assert packer.pack_batch(tsp_batch, ctx_tsp) is tsp_batch

  # Giant-Tour width (< dimension) skips in O(1).
  ctx_vrp = ProblemContext(_make_vrp_problem(max_vehicles=2, capacity=10), xp=np)
  monkeypatch.setattr(ctx_vrp, "get_cpu_distances", lambda: pytest.fail("Giant-Tour must skip"))
  giant = np.array([[1, 2, 3, 4]], dtype=np.int32)
  assert packer.pack_batch(giant, ctx_vrp) is giant

  # Unconstrained VRP (capacity=None, max_vehicles=None) skips in O(1).
  ctx_unconstrained: Any = SimpleNamespace(
    problem=SimpleNamespace(is_vrp=True),
    constraints=VRPConstraints(capacity=None, max_vehicles=None),
    get_cpu_distances=lambda: pytest.fail("Unbounded VRP must skip"),
  )
  assert packer.pack_batch(tsp_batch, cast("ProblemContext", ctx_unconstrained)) is tsp_batch


def test_cpu_feasible_batch_skips_distance_matrix_and_fleet_step_gating(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Feasible CPU batches bypass distance-matrix lookup via `np.add.reduceat` screen."""
  ctx_no_fleet = ProblemContext(_make_vrp_problem(max_vehicles=None, capacity=10), xp=np)
  monkeypatch.setattr(ctx_no_fleet, "get_cpu_distances", lambda: pytest.fail("Feasible batch must skip D"))
  feasible = np.array([[0, 1, 2, 0, 3, 4, 0]], dtype=np.int32)
  for cls in (BinPackingCPUFFD, BinPackingCPUBFD):
    packer = cls()
    assert packer.pack_batch(feasible, ctx_no_fleet) is feasible

  step = FleetBinPackStep(target="offspring")
  assert isinstance(step.bin_packing, BinPackingCPUFFD)
  assert step.stage_anchor == "pre_evaluation"
  state = GenerationState(
    population=feasible.copy(),
    fitness=np.zeros(1, dtype=np.float64),
    context=ctx_no_fleet,
    offspring=feasible.copy(),
  )
  monkeypatch.setattr(
    step.bin_packing,
    "pack_batch",
    lambda *a, **kw: pytest.fail("Step must gate on max_vehicles=None"),
  )
  step.run(state, PhaseTimer(enabled=False))


def test_ffd_vs_bfd_behavioral_distinction_cpu_and_gpu() -> None:
  """WHAT: Crafted instance where FFD picks route 0 (larger slack) while BFD picks route 1 (tighter residual)."""
  coords = np.array(
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 2.0],
      [0.0, 3.0],
      [2.0, 0.0],
      [3.0, 0.0],
    ],
    dtype=np.float64,
  )
  demands = np.array([0, 4, 4, 4, 3, 2], dtype=np.int32)
  problem = _make_vrp_problem(demands=demands, coords=coords, max_vehicles=2, capacity=10)
  ctx = ProblemContext(problem, xp=np)

  # Route 0: [1] (load=4, slack=6); Route 1: [2,3,4,5] (load=13 > 10).
  # Truncating Route 1 ejects 5(q=2) and 4(q=3), leaving Route 1=[2,3] (load=8, slack=2).
  # 4(q=3) enters Route 0 -> load=7 (slack=3). Then 5(q=2) fits Route 0 (res=1) and Route 1 (res=0).
  row = np.array([0, 1, 0, 2, 3, 4, 5, 0], dtype=np.int32)

  cpu_ffd = BinPackingCPUFFD().pack_batch(row, ctx)
  cpu_bfd = BinPackingCPUBFD().pack_batch(row, ctx)
  assert isinstance(cpu_ffd, np.ndarray)
  assert isinstance(cpu_bfd, np.ndarray)

  ffd_routes = _extract_subroutes(cpu_ffd)
  bfd_routes = _extract_subroutes(cpu_bfd)
  ffd_loads = [sum(int(demands[c]) for c in r) for r in ffd_routes]
  bfd_loads = [sum(int(demands[c]) for c in r) for r in bfd_routes]

  assert _extract_customers(cpu_ffd) == [1, 2, 3, 4, 5]
  assert _extract_customers(cpu_bfd) == [1, 2, 3, 4, 5]
  # FFD placed customer 5 into route 0 (loads [9, 8]); BFD placed customer 5 into route 1 (loads [7, 10]).
  assert ffd_loads == [9, 8]
  assert bfd_loads == [7, 10]

  if gpu_available():
    cp = require_gpu()
    ctx_gpu = ProblemContext(problem, xp=cp)
    gpu_ffd = BinPackingGPUFFD().pack_batch(cp.asarray(row), ctx_gpu)
    gpu_bfd = BinPackingGPUBFD().pack_batch(cp.asarray(row), ctx_gpu)
    np.testing.assert_array_equal(cp.asnumpy(gpu_ffd)[: cpu_ffd.size], cpu_ffd)
    np.testing.assert_array_equal(cp.asnumpy(gpu_bfd)[: cpu_bfd.size], cpu_bfd)


@pytest.mark.parametrize("variant", ["CVRP", "OVRP", "DCVRP", "OVER_TIGHT"])
@pytest.mark.parametrize("mode", ["ffd", "bfd"])
def test_conservation_and_cpu_gpu_parity_across_variants(variant: str, mode: str) -> None:
  """WHAT: Permutation conservation and CPU/GPU parity across CVRP, OVRP, DCVRP, and overflow fallback."""
  coords = np.array(
    [
      [0.0, 0.0],
      [3.0, 0.0],
      [5.0, 0.0],
      [0.0, 4.0],
      [0.0, 6.0],
      [-3.0, 0.0],
      [-5.0, 0.0],
      [0.0, -4.0],
    ],
    dtype=np.float64,
  )
  demands = np.array([0, 4, 5, 3, 6, 2, 5, 4], dtype=np.int32)

  if variant == "CVRP":
    problem = _make_vrp_problem(demands=demands, coords=coords, max_vehicles=3, capacity=12, problem_type="CVRP")
  elif variant == "OVRP":
    problem = _make_vrp_problem(
      demands=demands, coords=coords, max_vehicles=3, capacity=12, problem_type="OVRP", return_to_depot=False
    )
  elif variant == "DCVRP":
    problem = _make_vrp_problem(
      demands=demands, coords=coords, max_vehicles=3, capacity=12, max_route_distance=20.0, problem_type="DCVRP"
    )
  else:
    # Over-tight: total demand = 29 > 2 * 10 = 20 -> hard K_max=2 cap forces overflow fallback.
    problem = _make_vrp_problem(
      demands=demands, coords=coords, max_vehicles=2, capacity=10, max_route_distance=10.0, problem_type="DCVRP"
    )

  # Mixed batch: row 0 violates fleet & cap, row 1 violates cap only, row 2 is feasible (or over-tight).
  batch = np.array(
    [
      [0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 0, 0, 0, 0],
      [0, 1, 2, 3, 4, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0],
      [0, 1, 5, 0, 2, 6, 0, 3, 4, 7, 0, 0, 0, 0, 0],
    ],
    dtype=np.int32,
  )
  # Ensure row 2 is feasible for CVRP/OVRP/DCVRP (loads: 6, 10, 13 -> tweak row 2 for feasible test).
  batch[2] = np.array([0, 1, 5, 3, 0, 2, 6, 0, 4, 7, 0, 0, 0, 0, 0], dtype=np.int32)  # loads 9, 10, 10 <= 12

  ctx_cpu = ProblemContext(problem, xp=np)
  cpu_cls = BinPackingCPUFFD if mode == "ffd" else BinPackingCPUBFD
  cpu_packer = cpu_cls()
  cpu_out = cpu_packer.pack_batch(batch, ctx_cpu)
  assert isinstance(cpu_out, np.ndarray)

  k_cap = problem.constraints.max_vehicles
  assert k_cap is not None
  for r_idx in range(batch.shape[0]):
    assert _extract_customers(cpu_out[r_idx]) == [1, 2, 3, 4, 5, 6, 7]
    assert len(_extract_subroutes(cpu_out[r_idx])) <= k_cap

  if gpu_available():
    cp = require_gpu()
    ctx_gpu = ProblemContext(problem, xp=cp)
    gpu_cls = BinPackingGPUFFD if mode == "ffd" else BinPackingGPUBFD
    for max_smem_dim, max_mat_dim in ((768, 10_000), (0, 0)):
      gpu_packer = gpu_cls(
        max_matrix_dimension=max_mat_dim,
        max_shared_dimension=max_smem_dim,
      )
      gpu_out = gpu_packer.pack_batch(cp.asarray(batch), ctx_gpu)
      assert isinstance(gpu_out, cp.ndarray)
      gpu_host = cp.asnumpy(gpu_out)
      assert gpu_packer.d2h_bytes == 0
      for r_idx in range(batch.shape[0]):
        assert _extract_subroutes(gpu_host[r_idx]) == _extract_subroutes(cpu_out[r_idx])


@pytest.mark.gpu
@pytest.mark.parametrize("cls", [BinPackingGPUFFD, BinPackingGPUBFD])
def test_gpu_zero_transfers_across_generations_and_warmup_purity(cls: type[Any]) -> None:
  """WHAT: GPU-resident FFD/BFD achieve zero H2D/D2H across G>1 generations and pure warmup()."""
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  problem = _make_vrp_problem(max_vehicles=2, capacity=10)
  ctx = ProblemContext(problem, xp=cp, seed=123)
  packer = cls(seed=77)

  rng_before = copy.deepcopy(packer.rng.bit_generator.state)
  ctx_rng_before = copy.deepcopy(ctx.rng.bit_generator.state)
  packer.warmup(ctx)

  assert packer.h2d_bytes == 0
  assert packer.d2h_bytes == 0
  assert packer.kernel_launches == 0
  assert packer.call_count == 0
  assert packer.rng.bit_generator.state == rng_before
  assert ctx.h2d_bytes == 0
  assert ctx.rng.bit_generator.state == ctx_rng_before

  cache_before = _compile_kernel_cached.cache_info().currsize

  feasible_gpu = cp.array(
    [
      [0, 1, 2, 0, 3, 4, 0, 0],
      [0, 1, 4, 0, 2, 3, 0, 0],
    ],
    dtype=cp.int32,
  )
  violating_gpu = cp.array(
    [
      [0, 1, 4, 0, 3, 0, 2, 0],
      [0, 1, 2, 3, 4, 0, 0, 0],
    ],
    dtype=cp.int32,
  )

  for _gen in range(3):
    out_f = packer.pack_batch(feasible_gpu, ctx)
    out_v = packer.pack_batch(violating_gpu, ctx)
    assert isinstance(out_f, cp.ndarray)
    assert isinstance(out_v, cp.ndarray)

  assert packer.h2d_bytes == 0
  assert packer.d2h_bytes == 0
  assert packer.kernel_launches == 6
  assert _compile_kernel_cached.cache_info().currsize == cache_before
