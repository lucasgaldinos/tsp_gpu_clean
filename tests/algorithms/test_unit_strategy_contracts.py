"""Unit tests for strategy Protocol conformance and route-shape contracts.

Pins that every strategy family satisfies its Protocol, accepts both a 1D `Route` and a
2D `Routes` batch, and that the inversion / 2-opt variants produce valid tours on CPU
and, when a device exists, on GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from algorithms.strategies import (
  CrossoverOrder,
  CrossoverPartiallyMapped,
  EvaluationCPUCost,
  InitializationRandom,
  LocalSearchCPUTwoOpt,
  LocalSearchGPUTwoOpt,
  LocalSearchNoOp,
  MutationGPUInversion,
  MutationInversion,
  MutationSwap,
  SelectionRouletteWheel,
  SelectionTournament,
)
from protocols import (
  CrossoverStrategy,
  EvaluationStrategy,
  Fitness,
  InitializationStrategy,
  LocalSearchStrategy,
  MutationStrategy,
  ProblemContext,
  Route,
  RouteOrRoutes,
  Routes,
  SelectionStrategy,
)
from protocols.backend import cupy_backend, gpu_available

GPU_AVAILABLE = gpu_available()
cp = cupy_backend()

if TYPE_CHECKING:
  from data_models.problem import Problem


def test_all_strategies_satisfy_protocols_and_route_contracts(burma14_problem: Problem) -> None:
  """Verify all strategies satisfy their Protocol and accept both 1D Route and 2D Routes."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))
  n = len(customers)

  init_strat = InitializationRandom(seed=42)
  assert isinstance(init_strat, InitializationStrategy)
  routes: Routes = init_strat.initialize(customers, population_size=10, context=context)
  assert routes.shape == (10, n)

  eval_strat = EvaluationCPUCost()
  assert isinstance(eval_strat, EvaluationStrategy)
  fitness: Fitness = eval_strat.evaluate_batch(routes, context)
  assert fitness.shape == (10,)
  single_cost: float = eval_strat.evaluate(routes[0], context)
  assert np.isclose(single_cost, fitness[0])

  # Selection strategies
  for sel in (SelectionTournament(k=3, seed=42), SelectionRouletteWheel(seed=42)):
    assert isinstance(sel, SelectionStrategy)
    one_parent: Route = sel.select(routes, fitness, context=context)
    assert one_parent.shape == (n,)
    batch_parents: Routes = sel.select_batch(routes, fitness, num_parents=10, context=context)
    assert batch_parents.shape == (10, n)

  # Crossover strategies (1D pair and 2D batch)
  for cx in (CrossoverOrder(seed=42), CrossoverPartiallyMapped(seed=42)):
    assert isinstance(cx, CrossoverStrategy)
    child_1d: Route = cx.crossover(routes[0], routes[1], context=context)
    assert child_1d.shape == (n,)
    assert sorted(child_1d.tolist()) == customers

    children_2d: RouteOrRoutes = cx.crossover_batch(routes, context=context)
    assert children_2d.shape == (10, n)
    for row in children_2d:
      assert sorted(row.tolist()) == customers

  # Mutation strategies (1D route and 2D batch)
  for mut in (MutationSwap(seed=42), MutationInversion(seed=42)):
    assert isinstance(mut, MutationStrategy)
    mut_1d: RouteOrRoutes = mut.mutate_batch(routes[0], mutation_rate=1.0, context=context)
    assert mut_1d.shape == (n,)
    assert sorted(mut_1d.tolist()) == customers

    mut_2d: RouteOrRoutes = mut.mutate_batch(routes, mutation_rate=0.5, context=context)
    assert mut_2d.shape == (10, n)
    for row in mut_2d:
      assert sorted(row.tolist()) == customers

  # Local search strategies (1D route and 2D batch)
  for ls in (LocalSearchCPUTwoOpt(max_iterations=5), LocalSearchNoOp()):
    assert isinstance(ls, LocalSearchStrategy)
    imp_1d: RouteOrRoutes = ls.improve_batch(routes[0], context)
    assert imp_1d.shape == (n,)
    assert sorted(imp_1d.tolist()) == customers
    assert eval_strat.evaluate(imp_1d, context) <= eval_strat.evaluate(routes[0], context) + 1e-6

    imp_2d: RouteOrRoutes = ls.improve_batch(routes, context)
    assert imp_2d.shape == (10, n)


def test_inversion_mutation_cpu_and_gpu(burma14_problem: Problem) -> None:
  """Verify dedicated MutationInversion and MutationGPUInversion strategies."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  routes: np.ndarray = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (8, 1))

  cpu_mut = MutationInversion(seed=42)
  mutated_cpu = cpu_mut.mutate_batch(routes, mutation_rate=1.0, context=context)
  assert mutated_cpu.shape == routes.shape
  for r in mutated_cpu:
    assert sorted(r.tolist()) == list(range(burma14_problem.dimension))

  if GPU_AVAILABLE and cp is not None:
    gpu_context = ProblemContext(burma14_problem, xp=cp, seed=42)
    gpu_routes = cp.asarray(routes)
    gpu_mut = MutationGPUInversion(seed=42)
    mutated_gpu = gpu_mut.mutate_batch(gpu_routes, mutation_rate=1.0, context=gpu_context)
    assert mutated_gpu.shape == routes.shape
    host_routes = cp.asnumpy(mutated_gpu)
    for r in host_routes:
      assert sorted(r.tolist()) == list(range(burma14_problem.dimension))


def test_two_opt_variants(burma14_problem: Problem) -> None:
  """Verify 2-opt search variants: First-Improvement, Multi-Move, and BatchGPUMultiMove."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  assert context.get_cpu_distances().shape == (14, 14)
  evaluator = EvaluationCPUCost()

  # Inverted route gives suboptimal initial cost
  suboptimal_route = np.arange(burma14_problem.dimension - 1, -1, -1, dtype=np.int32)
  initial_cost = float(evaluator.evaluate(suboptimal_route, context=context))

  fi_opt = LocalSearchCPUTwoOpt(max_iterations=10, mode="first")
  fi_route = fi_opt.improve(suboptimal_route, context=context)
  fi_cost = float(evaluator.evaluate(fi_route, context=context))
  assert fi_cost < initial_cost, "first-improvement 2-opt must strictly improve the reversed tour"
  assert sorted(fi_route.tolist()) == list(range(burma14_problem.dimension))

  multi_opt = LocalSearchCPUTwoOpt(max_iterations=10, mode="multi_move")
  multi_route = multi_opt.improve(suboptimal_route, context=context)
  multi_cost = float(evaluator.evaluate(multi_route, context=context))
  assert multi_cost < initial_cost, "multi-move 2-opt must strictly improve the reversed tour"
  assert sorted(multi_route.tolist()) == list(range(burma14_problem.dimension))

  if GPU_AVAILABLE and cp is not None:
    gpu_context = ProblemContext(burma14_problem, xp=cp, seed=42)
    routes_batch = np.tile(suboptimal_route, (4, 1))
    gpu_routes = cp.asarray(routes_batch)
    gpu_multi_opt = LocalSearchGPUTwoOpt(max_iterations=10, mode="multi_move")
    gpu_improved = gpu_multi_opt.improve(gpu_routes, context=gpu_context)
    host_improved = cp.asnumpy(gpu_improved)
    for r in host_improved:
      assert sorted(r.tolist()) == list(range(burma14_problem.dimension))


@pytest.mark.gpu
def test_multi_move_two_opt_cpu_gpu_parity(burma14_problem: Problem) -> None:
  """Verify `LocalSearchCPUTwoOpt` and `LocalSearchGPUTwoOpt` agree route by route in multi-move mode.

  WHAT: Both implementations run on the same four reversed-tour routes.
  WHY: Equal costs alone can hide a permutation drift that changes crossover behavior;
  the GPU kernel must reproduce the CPU tours, not merely an equally cheap tour.
  EXPECTED: identical permutations and per-route costs within `rel=1e-9`.
  """
  assert cp is not None, "gpu-marked test requires CuPy"
  cpu_context = ProblemContext(burma14_problem, xp=np, seed=42)
  gpu_context = ProblemContext(burma14_problem, xp=cp, seed=42)
  suboptimal_route = np.arange(burma14_problem.dimension - 1, -1, -1, dtype=np.int32)
  routes = np.tile(suboptimal_route, (4, 1)).astype(np.int32, copy=False)

  cpu_improved = LocalSearchCPUTwoOpt(max_iterations=10, mode="multi_move").improve_batch(routes, cpu_context)
  gpu_improved = cp.asnumpy(
    LocalSearchGPUTwoOpt(max_iterations=10, mode="multi_move").improve_batch(cp.asarray(routes), gpu_context)
  )

  evaluator = EvaluationCPUCost()
  cpu_costs = [float(evaluator.evaluate(route, context=cpu_context)) for route in cpu_improved]
  gpu_costs = [float(evaluator.evaluate(route, context=cpu_context)) for route in gpu_improved]

  assert np.array_equal(cpu_improved, gpu_improved), "GPU multi-move must return the CPU tours"
  assert gpu_costs == pytest.approx(cpu_costs, rel=1e-9)
