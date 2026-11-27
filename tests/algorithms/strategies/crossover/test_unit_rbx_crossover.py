"""Unit and integration tests for separated CPU/GPU VRP route crossover & mutation operators (`DP17-7`, `BUG-114`)."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import (
  CrossoverStep,
  EvaluationStep,
  GeneticAlgorithm,
  MutationStep,
  SelectionStep,
  SurvivalStep,
)
from algorithms.pipeline_steps import GiantTourEncodeStep, SplitDecodeStep
from algorithms.strategies import STRATEGY_REGISTRY
from algorithms.strategies.crossover import CrossoverGPURBX, CrossoverOrder, CrossoverRBX
from algorithms.strategies.evaluation import EvaluationCPUCost, EvaluationGPUCost
from algorithms.strategies.initialization import (
  InitializationCPUClarkeWright,
  InitializationCPUSweep,
  InitializationGPUClarkeWright,
)
from algorithms.strategies.local_search import LocalSearchNoOp
from algorithms.strategies.mutation import (
  MutationGPURouteRelocate,
  MutationGPURouteSwap,
  MutationInversion,
  MutationRouteRelocate,
  MutationRouteSwap,
  MutationSwap,
)
from algorithms.strategies.selection import SelectionTournament
from data_models import Problem, VRPConstraints
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext


def _make_cvrp9(
  *,
  max_vehicles: int | None = None,
  max_route_distance: float | None = None,
  return_to_depot: bool = True,
  capacity: int = 12,
) -> Problem:
  """Create a 9-node VRP instance (depot=0, customers=1..8)."""
  coords = np.array(
    [
      [0.0, 0.0],
      [5.0, 0.0],
      [6.0, 2.0],
      [2.0, 6.0],
      [0.0, 5.0],
      [-5.0, 1.0],
      [-6.0, -2.0],
      [-1.0, -5.0],
      [4.0, -4.0],
    ],
    dtype=np.float64,
  )
  demands = np.array([0, 4, 4, 3, 5, 4, 3, 5, 4], dtype=np.int32)
  return Problem(
    name="cvrp9_rbx",
    dimension=9,
    problem_type="OVRP" if not return_to_depot else ("DCVRP" if max_route_distance is not None else "CVRP"),
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


def _assert_valid_delimited_batch(batch: np.ndarray, dimension: int, expected_width: int) -> None:
  """Assert every row has width `expected_width` and contains `{1..dimension-1}` exactly once."""
  assert batch.dtype == np.dtype(np.int32)
  assert batch.shape[1] == expected_width
  expected = list(range(1, dimension))
  for row in batch:
    non_zero = [int(x) for x in row.tolist() if int(x) != 0]
    assert sorted(non_zero) == expected


def test_all_new_vrp_route_operators_are_registered() -> None:
  """WHAT: `STRATEGY_REGISTRY` membership for separated CPU/GPU VRP route crossover & mutation classes."""
  expected = {
    "EvaluationCPUCost": EvaluationCPUCost,
    "EvaluationGPUCost": EvaluationGPUCost,
    "InitializationCPUClarkeWright": InitializationCPUClarkeWright,
    "InitializationCPUSweep": InitializationCPUSweep,
    "CrossoverRBX": CrossoverRBX,
    "CrossoverGPURBX": CrossoverGPURBX,
    "MutationRouteRelocate": MutationRouteRelocate,
    "MutationGPURouteRelocate": MutationGPURouteRelocate,
    "MutationRouteSwap": MutationRouteSwap,
    "MutationGPURouteSwap": MutationGPURouteSwap,
  }
  for name, cls in expected.items():
    assert STRATEGY_REGISTRY[name] is cls


@pytest.mark.parametrize("max_vehicles", [None, 2, 3])
def test_rbx_crossover_preserves_bijection_width_and_cpu_gpu_parity(max_vehicles: int | None) -> None:
  """WHAT: `CrossoverRBX` & `CrossoverGPURBX` (`RBX`) on CPU and GPU.
  WHY: Must preserve exact `{1..n-1}` bijection, exact width `W`, single/batch `int32` RNG parity,
  reuse `d_visited` workspace, and incur zero D2H on GPU.
  """
  problem = _make_cvrp9(max_vehicles=max_vehicles)
  ctx_init = ProblemContext(problem, xp=np, seed=10)
  parents = InitializationCPUClarkeWright(output_format="delimited").initialize(
    list(range(1, problem.dimension)), 10, ctx_init
  )
  width = int(parents.shape[1])

  ctx_cpu = ProblemContext(problem, xp=np, seed=77)
  cx_cpu = CrossoverRBX()
  off_cpu = cx_cpu.crossover_batch(parents, context=ctx_cpu)
  _assert_valid_delimited_batch(off_cpu, problem.dimension, width)

  # Single-pair API matches batch[0] under identical seed (int32 RNG parity)
  ctx_single = ProblemContext(problem, xp=np, seed=77)
  single = cx_cpu.crossover(parents[0], parents[1], context=ctx_single)
  _assert_valid_delimited_batch(single.reshape(1, -1), problem.dimension, width)
  ctx_pair_batch = ProblemContext(problem, xp=np, seed=77)
  pair_batch = cx_cpu.crossover_batch(parents[:2], context=ctx_pair_batch)
  np.testing.assert_array_equal(single, pair_batch[0])

  if not gpu_available():
    return
  cp = require_gpu()
  ctx_gpu = ProblemContext(problem, xp=cp, seed=77)
  cx_gpu = CrossoverGPURBX(return_gpu=True)
  cx_gpu.warmup(ctx_gpu)
  cx_gpu.reset_metrics()
  ctx_gpu_run = ProblemContext(problem, xp=cp, seed=77)
  off_gpu = cx_gpu.crossover_batch(cp.asarray(parents, dtype=cp.int32), context=ctx_gpu_run)
  assert isinstance(off_gpu, cp.ndarray)
  assert cx_gpu.d2h_bytes == 0
  assert cx_gpu._ws_visited is not None
  ws_ref = cx_gpu._ws_visited
  # Second call reuses cached d_visited without reallocation
  _ = cx_gpu.crossover_batch(cp.asarray(parents, dtype=cp.int32), context=ctx_gpu_run)
  assert cx_gpu._ws_visited is ws_ref
  np.testing.assert_array_equal(off_gpu.get(), off_cpu)


@pytest.mark.parametrize("return_to_depot", [True, False])
def test_rbx_respects_dcvrp_max_route_distance_and_ovrp_open_routes(return_to_depot: bool) -> None:
  """WHAT: `RBX` splits Parent-2 residual routes on `max_route_distance` honoring `return_to_depot` (`BUG-114`).
  WHY: Previously `RBX` only checked capacity, silently creating long infeasible `DCVRP`/`OVRP` routes.
  EXPECTED: Residual routes split on `L_max`, `OVRP` allows longer open legs without return leg, CPU/GPU match.
  """
  # Loose capacity=100, tight L_max=15.0: closed routes cannot carry many nodes without splitting
  problem = _make_cvrp9(capacity=100, max_route_distance=15.0, return_to_depot=return_to_depot, max_vehicles=6)
  parents = np.array(
    [
      [0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0],
      [0, 2, 3, 4, 5, 6, 7, 8, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int32,
  )
  ctx_cpu = ProblemContext(problem, xp=np, seed=5)
  off_cpu = CrossoverRBX().crossover_batch(parents, context=ctx_cpu)
  _assert_valid_delimited_batch(off_cpu, problem.dimension, parents.shape[1])
  assess_off = EvaluationCPUCost().assess_batch(off_cpu, ctx_cpu)
  assert assess_off.num_vehicles is not None
  num_routes = int(assess_off.num_vehicles[0])
  assert num_routes > 2

  if gpu_available():
    cp = require_gpu()
    ctx_gpu = ProblemContext(problem, xp=cp, seed=5)
    cx_gpu = CrossoverGPURBX(return_gpu=True)
    off_gpu = cx_gpu.crossover_batch(cp.asarray(parents, dtype=cp.int32), context=ctx_gpu)
    assert cx_gpu.d2h_bytes == 0
    assert isinstance(off_gpu, cp.ndarray)
    np.testing.assert_array_equal(off_gpu.get(), off_cpu)


@pytest.mark.parametrize(
  ("cpu_cls", "gpu_cls"),
  [
    (MutationRouteRelocate, MutationGPURouteRelocate),
    (MutationRouteSwap, MutationGPURouteSwap),
  ],
)
def test_route_mutations_preserve_bijection_zeros_and_cpu_gpu_parity(cpu_cls: type, gpu_cls: type) -> None:
  """WHAT: Separated CPU/GPU `Mutation*RouteRelocate` & `Mutation*RouteSwap`.
  WHY: Must preserve `{1..n-1}` bijection, exact zero count, CPU/GPU parity, and zero D2H on GPU.
  """
  problem = _make_cvrp9(max_vehicles=4)
  ctx_init = ProblemContext(problem, xp=np, seed=11)
  pop = InitializationCPUSweep(output_format="delimited").initialize(
    list(range(1, problem.dimension)), 12, ctx_init
  )
  width = int(pop.shape[1])
  orig_zeros = np.sum(pop == 0, axis=1)

  ctx_cpu = ProblemContext(problem, xp=np, seed=99)
  mut_cpu = cpu_cls()
  out_cpu = mut_cpu.mutate_batch(pop, mutation_rate=1.0, context=ctx_cpu)
  _assert_valid_delimited_batch(out_cpu, problem.dimension, width)
  np.testing.assert_array_equal(np.sum(out_cpu == 0, axis=1), orig_zeros)

  if not gpu_available():
    return
  cp = require_gpu()
  ctx_gpu = ProblemContext(problem, xp=cp, seed=99)
  mut_gpu = gpu_cls(return_gpu=True)
  mut_gpu.warmup(ctx_gpu)
  mut_gpu.reset_metrics()
  ctx_gpu_run = ProblemContext(problem, xp=cp, seed=99)
  out_gpu = mut_gpu.mutate_batch(cp.asarray(pop, dtype=cp.int32), mutation_rate=1.0, context=ctx_gpu_run)
  assert isinstance(out_gpu, cp.ndarray)
  assert mut_gpu.d2h_bytes == 0
  np.testing.assert_array_equal(out_gpu.get(), out_cpu)


def test_relocate_reaches_index_zero_and_opens_empty_vehicle_slots() -> None:
  """WHAT: `MutationRouteRelocate` can insert at index `0` and into empty `0` delimiter slots (`BUG-114`).
  WHY: Old relocate could never move a customer into an empty `0` slot to open a new route or before index 0.
  EXPECTED: Over a batch of single-route `[0, 1, 2, 3, 4, 0, 0, 0]`, both index-0 placement and opening
  a second non-empty route (`num_vehicles == 2`) occur while preserving `{1..4}` bijection.
  """
  prob = _make_cvrp9(max_vehicles=4)
  assert prob.coordinates is not None and prob.demands is not None
  prob_small = Problem(
    name="reloc5",
    dimension=5,
    problem_type="CVRP",
    edge_type="EUC_2D",
    coordinates=prob.coordinates[:5],
    distances=None,
    capacity=20,
    demands=prob.demands[:5],
    constraints=VRPConstraints(capacity=20, max_vehicles=4),
  )
  row = np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32)
  batch = np.tile(row, (128, 1))
  ctx = ProblemContext(prob_small, xp=np, seed=7)
  out = MutationRouteRelocate().mutate_batch(batch, mutation_rate=1.0, context=ctx)
  assert isinstance(out, np.ndarray)
  _assert_valid_delimited_batch(out, 5, 8)
  # Check that index 0 receives a customer in at least one row
  assert bool(np.any(out[:, 0] > 0))
  # Check that opening a second route (unused 0 delimiter slot) occurs
  n_veh = EvaluationCPUCost().assess_batch(out, ctx).num_vehicles
  assert n_veh is not None
  assert bool(np.any(np.asarray(n_veh) >= 2))


def test_full_ga_evolve_cycle_for_all_three_vrp_representation_families() -> None:
  """WHAT: Complete `GeneticAlgorithm.evolve()` runs on CVRP under all 3 families (`DP17-2`)."""
  problem = _make_cvrp9(max_vehicles=4)
  customers = list(range(1, problem.dimension))

  def _assert_canonical_best(tour: np.ndarray, fitness: float) -> None:
    assert fitness > 0.0 and np.isfinite(fitness)
    assert tour[0] == 0 and tour[-1] == 0
    assert not any(int(tour[i]) == 0 and int(tour[i + 1]) == 0 for i in range(len(tour) - 1))
    non_zero = [int(x) for x in tour.tolist() if int(x) != 0]
    assert sorted(non_zero) == customers

  ctx1 = ProblemContext(problem, xp=np, seed=101)
  ga1 = GeneticAlgorithm(
    population_size=12,
    mutation_rate=0.2,
    initialization=InitializationCPUClarkeWright(output_format="giant_tour"),
    selection=SelectionTournament(k=3),
    crossover=CrossoverOrder(),
    mutation=MutationSwap(),
    local_search=LocalSearchNoOp(),
    evaluation=EvaluationCPUCost(split_mode="prins"),
  )
  tour1, stats1 = ga1.evolve(ctx1, customers, max_generations=5)
  _assert_canonical_best(tour1, stats1["best_fitness"])

  ctx2 = ProblemContext(problem, xp=np, seed=102)
  eval2 = EvaluationCPUCost(split_mode="prins")
  sel2 = SelectionTournament(k=3)
  cx2 = CrossoverOrder()
  mut2 = MutationInversion()
  ga2 = GeneticAlgorithm(
    population_size=12,
    mutation_rate=0.2,
    initialization=InitializationCPUSweep(output_format="giant_tour"),
    selection=sel2,
    crossover=cx2,
    mutation=mut2,
    local_search=LocalSearchNoOp(),
    evaluation=eval2,
  )
  ga2.pipeline = [
    SelectionStep(sel2, 12),
    CrossoverStep(cx2),
    MutationStep(mut2, 0.2),
    SplitDecodeStep(mode="prins", target="offspring"),
    GiantTourEncodeStep(target="offspring"),
    EvaluationStep(eval2, target="offspring"),
    SurvivalStep(ga2._select_survivors),
  ]
  tour2, stats2 = ga2.evolve(ctx2, customers, max_generations=5)
  _assert_canonical_best(tour2, stats2["best_fitness"])

  ctx3 = ProblemContext(problem, xp=np, seed=103)
  ga3 = GeneticAlgorithm(
    population_size=12,
    mutation_rate=0.3,
    initialization=InitializationCPUClarkeWright(output_format="delimited"),
    selection=SelectionTournament(k=3),
    crossover=CrossoverRBX(),
    mutation=MutationRouteRelocate(),
    local_search=LocalSearchNoOp(),
    evaluation=EvaluationCPUCost(),
  )
  tour3, stats3 = ga3.evolve(ctx3, customers, max_generations=5)
  _assert_canonical_best(tour3, stats3["best_fitness"])

  if gpu_available():
    cp = require_gpu()
    ctx3_gpu = ProblemContext(problem, xp=cp, seed=103)
    ga3_gpu = GeneticAlgorithm(
      population_size=12,
      mutation_rate=0.3,
      initialization=InitializationGPUClarkeWright(output_format="delimited", return_gpu=True),
      selection=SelectionTournament(k=3),
      crossover=CrossoverGPURBX(return_gpu=True),
      mutation=MutationGPURouteSwap(return_gpu=True),
      local_search=LocalSearchNoOp(),
      evaluation=EvaluationGPUCost(return_gpu=True),
    )
    tour3_g, stats3_g = ga3_gpu.evolve(ctx3_gpu, customers, max_generations=5)
    _assert_canonical_best(tour3_g, stats3_g["best_fitness"])
