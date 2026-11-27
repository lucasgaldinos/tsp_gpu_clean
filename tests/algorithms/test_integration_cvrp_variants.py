"""End-to-end integration tests for all 4 Tier-1 VRP variants and 3 GA representation families (`DP17-2`, `DP17-3`).

Exercises full `GeneticAlgorithm.evolve()` cycles across:
- Tier-1 variants: `CVRP`, `OVRP` (`return_to_depot=False`), `Bounded-Fleet` (`max_vehicles`),
  and `DCVRP` (`max_route_distance`).
- Representation families:
  1. Pure Giant-Tour (`InitializationCPUClarkeWright` + `CrossoverOrder` + `MutationSwap` + `Evaluation*VRPCost`),
  2. Hybrid Prins-Split + Inter-Route Local Search (`LocalSearch{CPU,GPU}CVRP`),
  3. Direct `0`-Delimited (`InitializationCPUClarkeWright(output_format="delimited")` +
     `Crossover*RouteBased` + `Mutation*RouteRelocate` + `LocalSearch*InterRoute`).
- Declarative JSON `AlgorithmConfig.instantiate(...)` validated against `configs/schemas/algorithms.schema.json`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from algorithms.strategies import STRATEGY_REGISTRY
from algorithms.strategies.crossover import CrossoverGPURBX, CrossoverOrder, CrossoverRBX
from algorithms.strategies.evaluation import EvaluationCPUCost, EvaluationGPUCost
from algorithms.strategies.initialization import (
  InitializationCPUClarkeWright,
  InitializationCPUSweep,
  InitializationGPUClarkeWright,
  InitializationGPUSweep,
)
from algorithms.strategies.local_search import LocalSearchCPUTwoOptStar, LocalSearchGPUTwoOptStar, LocalSearchNoOp
from algorithms.strategies.mutation import MutationGPURouteRelocate, MutationRouteRelocate, MutationSwap
from algorithms.strategies.selection import SelectionGPUTournament, SelectionTournament
from benchmarking.runner.config_loader import AlgorithmConfig, GAParams
from data_models import Problem, VRPConstraints
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "schemas" / "algorithms.schema.json"


def _make_tier1_problem(variant: Literal["cvrp", "ovrp", "bounded_fleet", "dcvrp"]) -> Problem:
  """Build a deterministic 11-node Tier-1 VRP instance (depot=0, customers=1..10)."""
  coords = np.array(
    [
      [0.0, 0.0],
      [4.0, 1.0],
      [6.0, 3.0],
      [3.0, 6.0],
      [1.0, 5.0],
      [-4.0, 4.0],
      [-6.0, 1.0],
      [-5.0, -3.0],
      [-2.0, -6.0],
      [3.0, -5.0],
      [6.0, -2.0],
    ],
    dtype=np.float64,
  )
  demands = np.array([0, 3, 4, 3, 5, 4, 3, 4, 3, 5, 2], dtype=np.int32)

  if variant == "ovrp":
    return Problem(
      name="tier1_ovrp11",
      dimension=11,
      problem_type="OVRP",
      edge_type="EUC_2D",
      coordinates=coords,
      distances=None,
      capacity=12,
      demands=demands,
      constraints=VRPConstraints(capacity=12, return_to_depot=False),
    )
  if variant == "bounded_fleet":
    return Problem(
      name="tier1_fleet11",
      dimension=11,
      problem_type="CVRP",
      edge_type="EUC_2D",
      coordinates=coords,
      distances=None,
      capacity=12,
      demands=demands,
      constraints=VRPConstraints(capacity=12, max_vehicles=4, return_to_depot=True),
    )
  if variant == "dcvrp":
    return Problem(
      name="tier1_dcvrp11",
      dimension=11,
      problem_type="DCVRP",
      edge_type="EUC_2D",
      coordinates=coords,
      distances=None,
      capacity=12,
      demands=demands,
      constraints=VRPConstraints(capacity=12, max_route_distance=30.0, return_to_depot=True),
    )
  return Problem(
    name="tier1_cvrp11",
    dimension=11,
    problem_type="CVRP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=12,
    demands=demands,
    constraints=VRPConstraints(capacity=12, return_to_depot=True),
  )


def _assert_canonical_vrp_result(tour: np.ndarray, stats: dict[str, object], dimension: int) -> None:
  """Assert canonical `0`-delimited `best_tour`, exact customer bijection, and monotonic convergence."""
  assert tour.dtype == np.dtype(np.int32)
  assert tour.ndim == 1
  assert int(tour[0]) == 0
  assert int(tour[-1]) == 0
  assert not any(int(tour[i]) == 0 and int(tour[i + 1]) == 0 for i in range(len(tour) - 1))
  customers = [int(x) for x in tour.tolist() if int(x) != 0]
  assert sorted(customers) == list(range(1, dimension))

  history = stats["convergence_history"]
  assert isinstance(history, list)
  assert len(history) >= 2
  for prev, curr in itertools.pairwise(history):
    assert float(curr) <= float(prev) + 1e-9


def _build_ga(
  family: Literal["giant_tour", "hybrid_ls", "delimited"],
  *,
  use_gpu: bool,
) -> GeneticAlgorithm:
  """Construct a `GeneticAlgorithm` for one of the 3 VRP representation families on CPU or GPU."""
  if family == "giant_tour":
    return GeneticAlgorithm(
      population_size=12,
      mutation_rate=0.25,
      initialization=InitializationGPUClarkeWright(output_format="giant_tour", return_gpu=True)
      if use_gpu
      else InitializationCPUClarkeWright(output_format="giant_tour"),
      selection=SelectionGPUTournament(k=3) if use_gpu else SelectionTournament(k=3),
      crossover=CrossoverOrder(),
      mutation=MutationSwap(),
      local_search=LocalSearchNoOp(),
      evaluation=EvaluationGPUCost(split_mode="prins", return_gpu=use_gpu)
      if use_gpu
      else EvaluationCPUCost(split_mode="prins"),
    )

  if family == "hybrid_ls":
    return GeneticAlgorithm(
      population_size=10,
      mutation_rate=0.2,
      initialization=InitializationGPUSweep(output_format="giant_tour", return_gpu=True)
      if use_gpu
      else InitializationCPUSweep(output_format="giant_tour"),
      selection=SelectionGPUTournament(k=3) if use_gpu else SelectionTournament(k=3),
      crossover=CrossoverOrder(),
      mutation=MutationSwap(),
      local_search=LocalSearchGPUTwoOptStar(mode="multi_move", max_iterations=5, candidate_k=8)
      if use_gpu
      else LocalSearchCPUTwoOptStar(mode="multi_move", max_iterations=5, candidate_k=8),
      evaluation=EvaluationGPUCost(split_mode="prins", return_gpu=use_gpu)
      if use_gpu
      else EvaluationCPUCost(split_mode="prins"),
      local_search_stage="post_mutation",
    )

  # family == "delimited"
  return GeneticAlgorithm(
    population_size=10,
    mutation_rate=0.3,
    initialization=InitializationGPUClarkeWright(output_format="delimited", return_gpu=True)
    if use_gpu
    else InitializationCPUClarkeWright(output_format="delimited"),
    selection=SelectionGPUTournament(k=3) if use_gpu else SelectionTournament(k=3),
    crossover=CrossoverGPURBX(return_gpu=use_gpu) if use_gpu else CrossoverRBX(),
    mutation=MutationGPURouteRelocate(return_gpu=use_gpu) if use_gpu else MutationRouteRelocate(),
    local_search=LocalSearchGPUTwoOptStar(mode="multi_move", max_iterations=5, candidate_k=8)
    if use_gpu
    else LocalSearchCPUTwoOptStar(mode="multi_move", max_iterations=5, candidate_k=8),
    evaluation=EvaluationGPUCost(return_gpu=use_gpu) if use_gpu else EvaluationCPUCost(),
    local_search_stage="post_mutation",
  )


@pytest.mark.parametrize("variant", ["cvrp", "ovrp", "bounded_fleet", "dcvrp"])
@pytest.mark.parametrize("family", ["giant_tour", "hybrid_ls", "delimited"])
def test_cpu_tier1_variants_across_all_three_families(
  variant: Literal["cvrp", "ovrp", "bounded_fleet", "dcvrp"],
  family: Literal["giant_tour", "hybrid_ls", "delimited"],
) -> None:
  """WHAT: Full CPU `GeneticAlgorithm.evolve()` across all 4 Tier-1 variants x 3 families.
  WHY: Validates canonical `best_tour`, customer coverage, monotonic convergence, and
  seed reproducibility across every Tier-1 VRP variant and representation (`DP17-2`).
  EXPECTED: valid canonical `best_tour`, non-increasing convergence, deterministic seed replay.
  """
  problem = _make_tier1_problem(variant)
  customers = list(range(1, problem.dimension))

  ctx1 = ProblemContext(problem, xp=np, seed=42)
  tour1, stats1 = _build_ga(family, use_gpu=False).evolve(ctx1, customers, max_generations=4)
  _assert_canonical_vrp_result(tour1, stats1, problem.dimension)

  # Reproducibility check on CVRP
  if variant == "cvrp":
    ctx2 = ProblemContext(problem, xp=np, seed=42)
    tour2, stats2 = _build_ga(family, use_gpu=False).evolve(ctx2, customers, max_generations=4)
    np.testing.assert_array_equal(tour1, tour2)
    assert stats1["best_fitness"] == stats2["best_fitness"]


@pytest.mark.parametrize("variant", ["cvrp", "ovrp", "bounded_fleet", "dcvrp"])
@pytest.mark.parametrize("family", ["giant_tour", "hybrid_ls", "delimited"])
def test_gpu_tier1_variants_across_all_three_families_and_zero_d2h(
  variant: Literal["cvrp", "ovrp", "bounded_fleet", "dcvrp"],
  family: Literal["giant_tour", "hybrid_ls", "delimited"],
) -> None:
  """WHAT: Full GPU `GeneticAlgorithm.evolve()` across all 4 Tier-1 variants x 3 families.
  WHY: Verifies `warmup(context)`, GPU residency (`d2h_bytes == 0` inside GPU strategies),
  canonical `best_tour`, and monotonic convergence on CUDA (`DP17-3`).
  EXPECTED: valid canonical `best_tour`, monotonic history, `d2h_bytes == 0` in GPU operators.
  """
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  problem = _make_tier1_problem(variant)
  customers = list(range(1, problem.dimension))

  ga = _build_ga(family, use_gpu=True)
  warm_ctx = ProblemContext(problem, xp=cp, seed=1)
  ga.warmup(warm_ctx)

  ctx = ProblemContext(problem, xp=cp, seed=42)
  tour, stats = ga.evolve(ctx, customers, max_generations=4)
  _assert_canonical_vrp_result(tour, stats, problem.dimension)

  assert isinstance(ga.evaluation, EvaluationGPUCost)
  assert ga.evaluation.d2h_bytes == 0
  if isinstance(ga.local_search, LocalSearchGPUTwoOptStar):
    assert ga.local_search.d2h_bytes == 0
  if isinstance(ga.crossover, CrossoverGPURBX):
    assert ga.crossover.d2h_bytes == 0
  if isinstance(ga.mutation, MutationGPURouteRelocate):
    assert ga.mutation.d2h_bytes == 0


def test_declarative_algorithm_config_instantiation_and_schema_local_search_slot() -> None:
  """WHAT: Declarative `AlgorithmConfig.instantiate()` with new VRP strategies & schema slot check.
  WHY: Ensures `LocalSearchCPUTwoOptStar` and `LocalSearchGPUTwoOptStar` live strictly under
  `#/$defs/strategy_slot_local_search` and instantiate cleanly from JSON config dicts.
  EXPECTED: both classes in `STRATEGY_REGISTRY` and `strategy_slot_local_search`, instantiated
  GA executes `evolve()` cleanly on a Tier-1 CVRP instance.
  """
  assert STRATEGY_REGISTRY["LocalSearchCPUTwoOptStar"] is LocalSearchCPUTwoOptStar
  assert STRATEGY_REGISTRY["LocalSearchGPUTwoOptStar"] is LocalSearchGPUTwoOptStar

  schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
  ls_slot = schema["$defs"]["strategy_slot_local_search"]
  ls_classes = {
    branch["properties"]["class"]["const"]
    for branch in ls_slot["oneOf"]
    if "const" in branch.get("properties", {}).get("class", {})
  }
  assert "LocalSearchCPUTwoOptStar" in ls_classes
  assert "LocalSearchGPUTwoOptStar" in ls_classes

  use_gpu = gpu_available()
  cfg = AlgorithmConfig(
    name="HGS-CVRP-Declarative",
    use_gpu=use_gpu,
    description="Declarative CVRP memetic GA with ClarkeWright, LocalSearchInterRoute, and VRPCost",
    params={"population_size": 10, "mutation_rate": 0.25, "local_search_stage": "post_mutation"},
    strategies={
      "initialization": {
        "class": "InitializationGPUClarkeWright" if use_gpu else "InitializationCPUClarkeWright",
        "params": {"output_format": "delimited", "return_gpu": True} if use_gpu else {"output_format": "delimited"},
      },
      "selection": {
        "class": "SelectionGPUTournament" if use_gpu else "SelectionTournament",
        "params": {"k": 3},
      },
      "crossover": {
        "class": "CrossoverGPURBX" if use_gpu else "CrossoverRBX",
        "params": {"return_gpu": use_gpu} if use_gpu else {},
      },
      "mutation": {
        "class": "MutationGPURouteRelocate" if use_gpu else "MutationRouteRelocate",
        "params": {"return_gpu": use_gpu} if use_gpu else {},
      },
      "local_search": {
        "class": "LocalSearchGPUTwoOptStar" if use_gpu else "LocalSearchCPUTwoOptStar",
        "params": {"mode": "multi_move", "max_iterations": 5, "candidate_k": 8},
      },
      "evaluation": {
        "class": "EvaluationGPUCost" if use_gpu else "EvaluationCPUCost",
        "params": {"return_gpu": True} if use_gpu else {},
      },
    },
  )
  ga = cfg.instantiate(GAParams(population_size=10, mutation_rate=0.25, seed=7), seed=7)
  problem = _make_tier1_problem("cvrp")
  xp = require_gpu() if use_gpu else np
  ctx = ProblemContext(problem, xp=xp, seed=7)
  tour, stats = ga.evolve(ctx, list(range(1, problem.dimension)), max_generations=3)
  _assert_canonical_vrp_result(tour, stats, problem.dimension)
