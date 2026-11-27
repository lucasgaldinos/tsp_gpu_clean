"""Unit tests for CPU/GPU Clarke & Wright and Sweep initializers (`Task 17.31`)."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from algorithms.strategies.crossover.order_crossover import CrossoverOrder
from algorithms.strategies.evaluation import EvaluationCPUCost
from algorithms.strategies.initialization import (
  InitializationCPUClarkeWright,
  InitializationCPUSweep,
  InitializationGPUClarkeWright,
  InitializationGPUSweep,
)
from algorithms.strategies.local_search.no_op import LocalSearchNoOp
from algorithms.strategies.mutation.swap_mutation import MutationSwap
from algorithms.strategies.selection.tournament_selection import SelectionTournament
from benchmarking.runner.algorithm_runner import run_single_algorithm
from data_models import Problem, VRPConstraints
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext
from utils.gpu_helpers import _compile_kernel_cached


def _make_cvrp_problem(
  *,
  problem_type: str = "CVRP",
  capacity: int = 12,
  max_route_distance: float | None = None,
  max_vehicles: int | None = None,
  return_to_depot: bool = True,
  explicit_only: bool = False,
) -> Problem:
  """Create a 9-node VRP problem arranged around depot (0, 0)."""
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
  constraints = VRPConstraints(
    capacity=capacity,
    max_vehicles=max_vehicles,
    max_route_distance=max_route_distance,
    return_to_depot=return_to_depot,
  )
  if explicit_only:
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.floor(np.hypot(diff[:, :, 0], diff[:, :, 1]) + 0.5).astype(np.int32)
    return Problem(
      name="cvrp9_explicit",
      dimension=9,
      problem_type=problem_type,
      edge_type="EXPLICIT",
      coordinates=None,
      distances=dists,
      capacity=capacity,
      demands=demands,
      constraints=constraints,
    )
  return Problem(
    name=f"cvrp9_{problem_type.lower()}",
    dimension=9,
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=capacity,
    demands=demands,
    constraints=constraints,
  )


@pytest.mark.parametrize("strategy_cls", [InitializationCPUClarkeWright, InitializationCPUSweep])
@pytest.mark.parametrize("output_format", ["giant_tour", "delimited"])
def test_vrp_initializers_produce_valid_reproducible_feasible_populations(
  strategy_cls: type[Any],
  output_format: str,
) -> None:
  """WHAT: `InitializationCPUClarkeWright` and `InitializationCPUSweep` in both output formats."""
  assert "return_gpu" not in inspect.signature(strategy_cls.__init__).parameters

  problem = _make_cvrp_problem(capacity=12, max_route_distance=40.0, problem_type="DCVRP")
  ctx1 = ProblemContext(problem, xp=np, seed=123)
  ctx2 = ProblemContext(problem, xp=np, seed=123)
  customers = list(range(1, problem.dimension))

  pop1 = strategy_cls(output_format=output_format).initialize(customers, population_size=12, context=ctx1)
  pop2 = strategy_cls(output_format=output_format).initialize(customers, population_size=12, context=ctx2)

  np.testing.assert_array_equal(pop1, pop2)
  assert pop1.dtype == np.dtype(np.int32)
  assert pop1.shape[0] == 12

  expected_customers = set(customers)
  if output_format == "giant_tour":
    assert pop1.shape == (12, problem.dimension - 1)
    for row in pop1:
      assert 0 not in row
      assert set(int(x) for x in row.tolist()) == expected_customers
  else:
    assert pop1.shape == (12, problem.dimension + (problem.dimension - 1))
    for row in pop1:
      non_zero = [int(x) for x in row.tolist() if int(x) != 0]
      assert len(non_zero) == len(customers)
      assert set(non_zero) == expected_customers
    det = EvaluationCPUCost().evaluate_detailed_batch(pop1, ctx1)
    assert np.all(det.violations == 0.0)


def test_sweep_fallback_without_coordinates_and_ovrp_clarke_wright() -> None:
  """WHAT: `InitializationCPUSweep` on `EXPLICIT` (no coords) and `InitializationCPUClarkeWright` on `OVRP`."""
  explicit_prob = _make_cvrp_problem(explicit_only=True)
  ctx_exp = ProblemContext(explicit_prob, xp=np, seed=9)
  pop_sw = InitializationCPUSweep(output_format="delimited").initialize(list(range(1, 9)), 6, ctx_exp)
  det_sw = EvaluationCPUCost().evaluate_detailed_batch(pop_sw, ctx_exp)
  assert np.all(det_sw.violations == 0.0)

  ovrp_prob = _make_cvrp_problem(problem_type="OVRP", return_to_depot=False)
  ctx_ovrp = ProblemContext(ovrp_prob, xp=np, seed=9)
  pop_cw = InitializationCPUClarkeWright(output_format="delimited").initialize(list(range(1, 9)), 6, ctx_ovrp)
  det_cw = EvaluationCPUCost().evaluate_detailed_batch(pop_cw, ctx_ovrp)
  assert np.all(det_cw.violations == 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize(
  ("cpu_cls", "gpu_cls"),
  [
    (InitializationCPUSweep, InitializationGPUSweep),
    (InitializationCPUClarkeWright, InitializationGPUClarkeWright),
  ],
)
@pytest.mark.parametrize("output_format", ["giant_tour", "delimited"])
@pytest.mark.parametrize("variant", ["cvrp", "ovrp", "explicit", "on_the_fly"])
def test_vrp_initializers_cpu_gpu_bit_identity_and_warmup_purity(
  cpu_cls: type[Any],
  gpu_cls: type[Any],
  output_format: str,
  variant: str,
) -> None:
  """WHAT: Exact CPU==GPU bit-identity, `return_gpu=True` default, `h2d_bytes`, and `warmup()` purity."""
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()
  assert inspect.signature(gpu_cls.__init__).parameters["return_gpu"].default is True

  if variant == "ovrp":
    problem = _make_cvrp_problem(problem_type="OVRP", return_to_depot=False)
    max_mat = 10_000
  elif variant == "explicit":
    problem = _make_cvrp_problem(explicit_only=True)
    max_mat = 10_000
  elif variant == "on_the_fly":
    problem = _make_cvrp_problem(problem_type="DCVRP", capacity=12, max_route_distance=40.0)
    max_mat = 0
  else:
    problem = _make_cvrp_problem(problem_type="CVRP", capacity=12)
    max_mat = 10_000

  customers = list(range(1, problem.dimension))
  extra: dict[str, Any] = {"candidate_k": 4} if cpu_cls is InitializationCPUClarkeWright else {}

  ctx_warm = ProblemContext(problem, xp=cp, seed=11)
  gpu_init = gpu_cls(output_format=output_format, max_matrix_dimension=max_mat, seed=55, **extra)
  rng_before = copy.deepcopy(gpu_init.rng.bit_generator.state)
  ctx_rng_before = copy.deepcopy(ctx_warm.rng.bit_generator.state)
  gpu_init.warmup(ctx_warm)
  assert gpu_init.h2d_bytes == 0
  assert gpu_init.d2h_bytes == 0
  assert gpu_init.kernel_launches == 0
  assert gpu_init.call_count == 0
  assert gpu_init.rng.bit_generator.state == rng_before
  assert ctx_warm.h2d_bytes == 0
  assert ctx_warm.rng.bit_generator.state == ctx_rng_before
  cache_size = _compile_kernel_cached.cache_info().currsize

  ctx_cpu = ProblemContext(problem, xp=np, seed=202)
  ctx_gpu = ProblemContext(problem, xp=cp, seed=202)
  cpu_init = cpu_cls(output_format=output_format, seed=55, **extra)

  cpu_pop = cpu_init.initialize(customers, 10, ctx_cpu)
  gpu_pop = gpu_init.initialize(customers, 10, ctx_gpu)

  assert isinstance(gpu_pop, cp.ndarray)
  assert gpu_pop.dtype == cp.int32
  assert gpu_init.d2h_bytes == 0
  assert gpu_init.h2d_bytes > 0
  assert _compile_kernel_cached.cache_info().currsize == cache_size
  np.testing.assert_array_equal(cpu_pop, cp.asnumpy(gpu_pop))


def test_ga_evolve_normalizes_vrp_best_tour_and_runner_strips_depot_customer() -> None:
  """WHAT: `GeneticAlgorithm.evolve()` `best_tour` normalization and `run_single_algorithm` `customers`."""
  problem = _make_cvrp_problem()
  ctx = ProblemContext(problem, xp=np, seed=42)
  ga = GeneticAlgorithm(
    population_size=12,
    mutation_rate=0.2,
    initialization=InitializationCPUClarkeWright(output_format="giant_tour"),
    selection=SelectionTournament(k=3),
    crossover=CrossoverOrder(),
    mutation=MutationSwap(),
    local_search=LocalSearchNoOp(),
    evaluation=EvaluationCPUCost(split_mode="prins"),
  )
  best_tour, stats = ga.evolve(ctx, list(range(1, problem.dimension)), max_generations=5)
  assert best_tour[0] == 0
  assert best_tour[-1] == 0
  assert not any(int(best_tour[i]) == 0 and int(best_tour[i + 1]) == 0 for i in range(len(best_tour) - 1))
  non_zero = [int(x) for x in best_tour.tolist() if int(x) != 0]
  assert sorted(non_zero) == list(range(1, problem.dimension))
  assert stats["best_fitness"] > 0.0

  seen_customers: list[list[int]] = []

  class _SpyAlgo:
    evaluation = EvaluationCPUCost()

    def evolve(
      self,
      context: ProblemContext,
      customers: list[int],
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, max_generations, optimal_cost, patience
      seen_customers.append(list(customers))
      return np.array([0, *customers, 0], dtype=np.int32), {
        "initial_fitness": 100.0,
        "best_fitness": 90.0,
        "generations_completed": 1,
        "stop_reason": "max_generations",
        "phase_timings": {},
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "kernel_launches": 0,
      }

  run_single_algorithm(
    algorithm_name="spy",
    algorithm_instance=_SpyAlgo(),
    problem=problem,
    problem_name=problem.name,
    problem_size=problem.dimension,
    customers=list(range(problem.dimension)),
    max_generations=1,
    optimal_cost=None,
    repetitions=1,
    use_gpu=False,
  )
  assert seen_customers == [list(range(1, problem.dimension))]
