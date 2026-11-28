"""Falsifiable unit tests for Tasks 17.14 & 17.15 (`DP17-8`, `BUG-112`, `BUG-115`).

Covers:
1. `StrategyBase.warmup(context)` telemetry (`h2d_bytes == d2h_bytes == kernel_launches == 0`),
   `@time_strategy` (`call_count == 0`), and `context.rng` bit-state purity across GPU/CPU strategies.
2. Polymorphic `EvaluationStrategy` (`canonicalize_route`, `assess_batch -> RouteAssessment`),
   custom `DP16-3` evaluator fallback (`assess_with_evaluator`), and `compute_normalized_violations`.
3. Strict Feasible-First Elitism in `GeneticAlgorithm` (feasible cost `500` beats infeasible `400`),
   `GenerationState.refresh_penalized_fitness` O(N) refresh, and zero VRP leakage.
4. `run_single_algorithm` honoring custom `split_mode="greedy"` / custom penalty evaluators,
   emitting `NaN` in `raw_costs` / excluding infeasible runs from `mean_improvement`, and
   recording `raw_normalized_violations`.
5. `SplitDecodeStep` `"decoding"` bucket and idempotence on already `0`-delimited routes.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GenerationState, GeneticAlgorithm
from algorithms.pipeline_steps import SplitDecodeStep
from algorithms.strategies import (
  CrossoverGPUPermOrder,
  CrossoverGPURouteRBX,
  EvaluationCPUCost,
  EvaluationGPUCost,
  FixedPenaltyPolicy,
  LocalSearchGPURouteTwoOptStar,
  LocalSearchGPUTourTwoOpt,
  LocalSearchNoOp,
  MutationGPUPermSwap,
  MutationGPURouteRelocate,
  MutationGPURouteSwap,
  SelectionGPUTournament,
)
from algorithms.strategies.decoding import PrinsSplitDecoder
from benchmarking.runner.algorithm_runner import run_single_algorithm
from data_models import Problem, VRPConstraints
from protocols import (
  BinPackingStrategy,
  EvaluationStrategy,
  PenaltyPolicy,
  ProblemContext,
  RouteAssessment,
  RouteDecoder,
  assess_with_evaluator,
  canonicalize_with_evaluator,
)
from protocols.backend import gpu_available, require_gpu
from protocols.vrp_strategies import compute_normalized_violations
from utils.phase_timer import PhaseTimer


def _make_cvrp_problem(*, capacity: int = 10, max_vehicles: int | None = None) -> Problem:
  """Build a deterministic 5-node CVRP instance (depot=0, customers=1..4)."""
  coords = np.array(
    [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [0.0, 4.0], [0.0, 8.0]],
    dtype=np.float64,
  )
  demands = np.array([0, 4, 5, 6, 4], dtype=np.int32)
  return Problem(
    name="poly_cvrp5",
    dimension=5,
    problem_type="CVRP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=capacity,
    demands=demands,
    constraints=VRPConstraints(capacity=capacity, max_vehicles=max_vehicles, return_to_depot=True),
  )


def test_vrp_protocols_and_tsp_evaluation_polymorphism(burma14_problem: Problem) -> None:
  """WHAT: `EvaluationCPUCost` + bare `DP16-3` evaluator satisfy `EvaluationStrategy` / `assess_with_evaluator`.
  WHY: `DP17-8` removes VRP branching by giving every evaluator `canonicalize_route` and `assess_batch`.
  EXPECTED: `RouteAssessment` populated cleanly, `PrinsSplitDecoder` matches `RouteDecoder`,
  and bare custom evaluator falls back cleanly via `assess_with_evaluator`.
  """
  assert isinstance(PrinsSplitDecoder(), RouteDecoder)
  assert isinstance(EvaluationCPUCost(), EvaluationStrategy)

  class _DummyPenalty:
    def weights(self, context: ProblemContext) -> tuple[float, float, float]:
      del context
      return (10.0, 10.0, 10.0)

    def apply(self, travel_costs: Any, raw_violation_components: Any, context: ProblemContext) -> Any:
      del raw_violation_components, context
      return travel_costs

  class _DummyPacker:
    def pack_batch(self, routes: Any, context: ProblemContext) -> Any:
      del context
      return routes

  assert isinstance(_DummyPenalty(), PenaltyPolicy)
  assert isinstance(_DummyPacker(), BinPackingStrategy)

  ctx = ProblemContext(burma14_problem, xp=np, seed=3)
  tours = np.tile(np.arange(burma14_problem.dimension, dtype=np.int32), (2, 1))
  ev = EvaluationCPUCost()
  assessment = ev.assess_batch(tours, ctx)
  assert isinstance(assessment, RouteAssessment)
  assert assessment.objectives.shape == (2,)
  assert np.all(assessment.is_feasible)
  assert np.all(assessment.violations == 0.0)
  assert np.all(assessment.normalized_violations == 0.0)
  np.testing.assert_array_equal(ev.canonicalize_route(tours[0], ctx), tours[0])

  class _BareCustomEvaluator:
    def evaluate(self, route: np.ndarray, context: ProblemContext) -> float:
      del context
      return float(np.sum(route))

    def evaluate_batch(self, routes: np.ndarray, context: ProblemContext) -> np.ndarray:
      del context
      return np.sum(routes, axis=1, dtype=np.float64)

  bare = _BareCustomEvaluator()
  bare_assessment = assess_with_evaluator(bare, tours, ctx)
  assert bare_assessment.objectives.shape == (2,)
  assert np.all(bare_assessment.is_feasible)
  np.testing.assert_array_equal(canonicalize_with_evaluator(bare, tours[0], ctx), tours[0])


def test_compute_normalized_violations_is_dimensionless() -> None:
  """WHAT: `compute_normalized_violations` over multi-unit `(N, 3)` excesses.
  WHY: `DP17-10` normalizes capacity, route distance, and fleet excess by `Q`, `L_max`, `K_max`.
  EXPECTED: exact `viol_q/Q + viol_l/L_max + viol_k/K_max`.
  """
  prob = Problem(
    name="norm5",
    dimension=5,
    problem_type="DCVRP",
    edge_type="EUC_2D",
    coordinates=np.zeros((5, 2), dtype=np.float64),
    distances=None,
    capacity=10,
    demands=np.array([0, 2, 2, 2, 2], dtype=np.int32),
    constraints=VRPConstraints(capacity=10, max_route_distance=20.0, max_vehicles=2),
  )
  ctx = ProblemContext(prob, xp=np, seed=0)
  comps = np.array([[5.0, 4.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
  norm = compute_normalized_violations(comps, ctx)
  np.testing.assert_allclose(norm, [0.5 + 0.2 + 0.5, 0.0])


def test_warmup_purity_preserves_counters_timings_and_rng_state(burma14_problem: Problem) -> None:
  """WHAT: `StrategyBase.warmup(ctx)` counter, timer, and `context.rng` isolation (`BUG-115`).
  WHY: Warm-up previously leaked `h2d_bytes`, `kernel_launches`, `@time_strategy` call counts,
  and mutated `context.rng` before repetition 1.
  EXPECTED: `h2d_bytes == 0`, `d2h_bytes == 0`, `kernel_launches == 0`, `call_count == 0`,
  `context.h2d_bytes == 0`, and identical `context.rng.bit_generator.state` after `warmup()`.
  """
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  tsp_ctx = ProblemContext(burma14_problem, xp=cp, seed=77)
  tsp_strategies: list[Any] = [
    EvaluationGPUCost(return_gpu=True),
    LocalSearchGPUTourTwoOpt(mode="batch", max_iterations=2),
    SelectionGPUTournament(k=3),
    CrossoverGPUPermOrder(),
    MutationGPUPermSwap(),
  ]
  for strat in tsp_strategies:
    before_rng = copy.deepcopy(tsp_ctx.rng.bit_generator.state)
    before_draw = tsp_ctx.rng.random(4)
    tsp_ctx.rng.bit_generator.state = copy.deepcopy(before_rng)

    strat.warmup(tsp_ctx)
    assert strat.h2d_bytes == 0
    assert strat.d2h_bytes == 0
    assert strat.kernel_launches == 0
    assert strat.call_count == 0
    assert strat.total_elapsed_ms == 0.0
    assert tsp_ctx.h2d_bytes == 0
    assert strat.rng.bit_generator.state == strat.rng.bit_generator.state
    after_draw = tsp_ctx.rng.random(4)
    np.testing.assert_array_equal(after_draw, before_draw)
    tsp_ctx.rng.bit_generator.state = copy.deepcopy(before_rng)

  vrp_prob = _make_cvrp_problem(capacity=10, max_vehicles=3)
  vrp_ctx = ProblemContext(vrp_prob, xp=cp, seed=88)
  vrp_strategies: list[Any] = [
    EvaluationGPUCost(return_gpu=True),
    LocalSearchGPURouteTwoOptStar(mode="batch", max_iterations=2),
    CrossoverGPURouteRBX(return_gpu=True),
    MutationGPURouteRelocate(return_gpu=True),
    MutationGPURouteSwap(return_gpu=True),
    PrinsSplitDecoder(use_gpu=True),
  ]
  for strat in vrp_strategies:
    before_rng = copy.deepcopy(vrp_ctx.rng.bit_generator.state)
    strat.warmup(vrp_ctx)
    assert strat.h2d_bytes == 0
    assert strat.d2h_bytes == 0
    assert strat.kernel_launches == 0
    assert strat.call_count == 0
    assert vrp_ctx.h2d_bytes == 0
    assert vrp_ctx.rng.bit_generator.state == before_rng


def test_feasible_first_elitism_beats_lower_penalized_infeasible() -> None:
  """WHAT: Feasible tour of cost `500` vs infeasible tour with lower penalized cost `400` (`DP17-8`).
  WHY: When adaptive penalties dip, an infeasible solution can score below a feasible one;
  feasible-first elitism must never discard or overwrite the feasible incumbent.
  EXPECTED: feasible `500` wins in Gen 0, survives across generations, and wins survival selection.
  """
  prob = _make_cvrp_problem(capacity=10)
  ctx = ProblemContext(prob, xp=np, seed=1)

  tour_feas = np.array([0, 1, 2, 0, 3, 4, 0, 0], dtype=np.int32)
  tour_infeas = np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32)

  class _FixedInit:
    def initialize(self, customers: Any, pop_size: int, context: ProblemContext) -> np.ndarray:
      del customers, pop_size, context
      return np.vstack([tour_infeas, tour_feas])

  class _NoOpSel:
    def select(self, pop: Any, fit: Any, context: ProblemContext | None = None) -> Any:
      del fit, context
      return pop[0]

    def select_batch(self, routes: Any, fit: Any, num_parents: int | None = None, context: Any = None) -> Any:
      del fit, num_parents, context
      return routes.copy()

  class _InfeasGenCx:
    def crossover(self, p1: Any, p2: Any, context: ProblemContext | None = None) -> Any:
      del p2, context
      return p1

    def crossover_batch(self, routes: Any, context: ProblemContext | None = None) -> Any:
      del context
      return np.vstack([tour_infeas, tour_infeas])

  class _NoOpMut:
    def mutate(self, ind: Any, rate: float, context: ProblemContext | None = None) -> Any:
      del rate, context
      return ind

    def mutate_batch(self, routes: Any, rate: float, context: ProblemContext | None = None) -> Any:
      del rate, context
      return routes

  class _ManipulatedEval:
    def evaluate(self, route: np.ndarray, context: ProblemContext) -> float:
      return float(self.evaluate_batch(route.reshape(1, -1), context)[0])

    def evaluate_batch(self, routes: np.ndarray, context: ProblemContext) -> np.ndarray:
      return np.asarray(self.assess_batch(routes, context).objectives, dtype=np.float64)

    def canonicalize_route(self, route: np.ndarray, context: ProblemContext) -> np.ndarray:
      del context
      return np.asarray(route, dtype=np.int32)

    def assess_batch(
      self,
      routes: np.ndarray,
      context: ProblemContext,
      return_gpu: bool = False,
    ) -> RouteAssessment:
      del context, return_gpu
      n = routes.shape[0]
      obj = np.empty(n, dtype=np.float64)
      travel = np.empty(n, dtype=np.float64)
      feas = np.empty(n, dtype=bool)
      viol = np.empty(n, dtype=np.float64)
      norm = np.empty(n, dtype=np.float64)
      comps = np.zeros((n, 3), dtype=np.float64)
      for i in range(n):
        if np.array_equal(routes[i], tour_feas):
          obj[i], travel[i], feas[i], viol[i], norm[i] = 500.0, 500.0, True, 0.0, 0.0
        else:
          obj[i], travel[i], feas[i], viol[i], norm[i] = 400.0, 350.0, False, 9.0, 0.9
          comps[i, 0] = 9.0
      return RouteAssessment(
        objectives=obj,
        travel_costs=travel,
        is_feasible=feas,
        violations=viol,
        normalized_violations=norm,
        raw_violation_components=comps,
        num_vehicles=np.ones(n, dtype=np.int32),
      )

  ga = GeneticAlgorithm(
    population_size=2,
    mutation_rate=0.0,
    initialization=_FixedInit(),
    selection=_NoOpSel(),
    crossover=_InfeasGenCx(),
    mutation=_NoOpMut(),
    local_search=LocalSearchNoOp(),
    evaluation=_ManipulatedEval(),
  )
  best_tour, stats = ga.evolve(ctx, [1, 2, 3, 4], max_generations=3)
  assert stats["best_is_feasible"] is True
  assert stats["best_fitness"] == 500.0
  assert stats["initial_feasible_cost"] == pytest.approx(500.0)
  np.testing.assert_array_equal(best_tour, tour_feas)

  # When Gen 0 has no feasible tour, initial_feasible_cost must be NaN
  class _AllInfeasInit:
    def initialize(self, customers: Any, pop_size: int, context: ProblemContext) -> np.ndarray:
      del customers, pop_size, context
      return np.vstack([tour_infeas, tour_infeas])

  ga_infeas = GeneticAlgorithm(
    population_size=2,
    mutation_rate=0.0,
    initialization=_AllInfeasInit(),
    selection=_NoOpSel(),
    crossover=_InfeasGenCx(),
    mutation=_NoOpMut(),
    local_search=LocalSearchNoOp(),
    evaluation=_ManipulatedEval(),
  )
  _, stats_infeas_gen0 = ga_infeas.evolve(ctx, [1, 2, 3, 4], max_generations=1)
  assert math.isnan(stats_infeas_gen0["initial_feasible_cost"])

  # Also verify O(N) GenerationState.refresh_penalized_fitness
  state = GenerationState(
    population=np.vstack([tour_infeas, tour_feas]),
    fitness=np.array([400.0, 500.0]),
    context=ctx,
    population_assessment=_ManipulatedEval().assess_batch(np.vstack([tour_infeas, tour_feas]), ctx),
  )
  state.refresh_penalized_fitness(omega_q=100.0, omega_l=0.0, omega_k=0.0)
  np.testing.assert_allclose(state.fitness, [350.0 + 900.0, 500.0])


def test_runner_honors_custom_evaluator_and_excludes_infeasible_from_costs_and_improvement() -> None:
  """WHAT: `run_single_algorithm` delegates VRP assessment to `current_algo.evaluation` (`BUG-112`).
  WHY: Hardcoding `EvaluationCPUCost()` ignored custom `split_mode="greedy"` and penalties,
  and recording penalized costs on infeasible runs fabricated fake improvements.
  EXPECTED: Custom evaluator is called; infeasible run yields `NaN` in `raw_costs` and
  `mean_improvement` averages only finite feasible runs while preserving `raw_normalized_violations`.
  """
  prob = _make_cvrp_problem(capacity=10)
  calls: list[str] = []

  class _SpyGreedyEval(EvaluationCPUCost):
    def __init__(self) -> None:
      super().__init__(split_mode="greedy", penalty_policy=FixedPenaltyPolicy(penalty_capacity=1.0))

    def assess_batch(self, routes: Any, context: ProblemContext, return_gpu: bool = False) -> RouteAssessment:
      calls.append("assess_batch")
      return super().assess_batch(routes, context, return_gpu=return_gpu)

  feas_tour = np.array([0, 1, 2, 0, 3, 4, 0], dtype=np.int32)
  infeas_tour = np.array([0, 1, 2, 3, 4, 0], dtype=np.int32)

  class _TwoRepAlgo:
    def __init__(self) -> None:
      self.evaluation = _SpyGreedyEval()
      self._rep = 0

    def evolve(
      self,
      context: ProblemContext,
      customers: list[int],
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, customers, max_generations, optimal_cost, patience
      idx = self._rep
      self._rep += 1
      if idx == 0:
        # Feasible run: initial=56, final=28 -> 50% improvement
        return feas_tour, {
          "initial_fitness": 56.0,
          "best_fitness": 28.0,
          "generations_completed": 1,
          "stop_reason": "max_generations",
          "phase_timings": {},
          "h2d_bytes": 0,
          "d2h_bytes": 0,
          "kernel_launches": 0,
        }
      # Infeasible run: penalized best_fitness=20, initial=100 -> must NOT report 80% improvement
      return infeas_tour, {
        "initial_fitness": 100.0,
        "best_fitness": 20.0,
        "generations_completed": 1,
        "stop_reason": "max_generations",
        "phase_timings": {},
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "kernel_launches": 0,
      }

  summary = run_single_algorithm(
    algorithm_name="SpyGreedy",
    algorithm_instance=_TwoRepAlgo(),
    problem=prob,
    problem_name=prob.name,
    problem_size=prob.dimension,
    customers=list(range(prob.dimension)),
    max_generations=1,
    optimal_cost=28.0,
    repetitions=2,
    use_gpu=False,
  )

  assert calls == ["assess_batch", "assess_batch"]
  assert summary["feasible_runs"] == 1
  assert summary["feasibility_rate"] == 50.0
  assert summary["raw_is_feasible"] == [True, False]
  assert math.isfinite(summary["raw_costs"][0])
  assert math.isnan(summary["raw_costs"][1])
  assert summary["raw_unpenalized_costs"][1] > 0.0
  assert summary["raw_violations"][1] == 9.0
  assert summary["raw_normalized_violations"] == [0.0, pytest.approx(0.9)]
  assert summary["mean_cost"] == pytest.approx(summary["raw_unpenalized_costs"][0])
  expected_imp = (56.0 - summary["raw_unpenalized_costs"][0]) / 56.0 * 100.0
  assert summary["mean_improvement"] == pytest.approx(expected_imp)


def test_split_decode_step_uses_decoding_bucket_and_skips_already_delimited() -> None:
  """WHAT: `SplitDecodeStep` timer bucket and idempotence on `0`-delimited batches (`BUG-112`).
  WHY: `SplitDecodeStep` previously logged under `'local_search'` and crashed if routes were
  already `0`-delimited (`width >= dimension`).
  EXPECTED: logs under `'decoding'` and leaves already `0`-delimited batches untouched.
  """
  prob = _make_cvrp_problem(capacity=10)
  ctx = ProblemContext(prob, xp=np, seed=5)
  step = SplitDecodeStep(mode="prins", target="offspring")
  timer = PhaseTimer(enabled=True)

  giant = np.array([[1, 2, 3, 4]], dtype=np.int32)
  state = GenerationState(
    population=giant.copy(),
    fitness=np.zeros(1, dtype=np.float64),
    context=ctx,
    offspring=giant.copy(),
  )
  step.run(state, timer)
  assert state.offspring is not None
  assert state.offspring.shape[1] >= prob.dimension
  timings = timer.summary()
  assert "decoding" in timings
  assert "local_search" not in timings
  assert timings["decoding"]["calls"] == 1

  # Calling again on already-delimited offspring is a clean no-op
  delimited_before = state.offspring.copy()
  step.run(state, timer)
  np.testing.assert_array_equal(state.offspring, delimited_before)
  assert timer.summary()["decoding"]["calls"] == 1


def test_runner_vrp_improvement_uses_initial_feasible_cost_and_excludes_infeasible_gen0() -> None:
  """WHAT: `run_single_algorithm` computes VRP `mean_improvement` from `initial_feasible_cost`.
  WHY: Using penalized `initial_fitness` (e.g., 50_000) against unpenalized `28` manufactures
  a bogus ~99.9% improvement even when Gen 0 was completely infeasible.
  EXPECTED: Run with `initial_feasible_cost=NaN` records `NaN` initial cost; run with
  `initial_feasible_cost=56.0` and penalized `initial_fitness=50000.0` yields `50.0%`.
  """
  prob = _make_cvrp_problem(capacity=10)
  feas_tour = np.array([0, 1, 2, 0, 3, 4, 0], dtype=np.int32)

  class _Gen0SpyAlgo:
    def __init__(self) -> None:
      self.evaluation = EvaluationCPUCost(penalty_policy=FixedPenaltyPolicy(penalty_capacity=1000.0))
      self._rep = 0

    def evolve(
      self,
      context: ProblemContext,
      customers: list[int],
      max_generations: int,
      optimal_cost: float | None = None,
      patience: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
      del context, customers, max_generations, optimal_cost, patience
      unpen_final = float(
        self.evaluation.assess_batch(
          feas_tour.reshape(1, -1), ProblemContext(prob, xp=np)
        ).travel_costs[0]
      )
      idx = self._rep
      self._rep += 1
      if idx == 0:
        return feas_tour, {
          "initial_fitness": 50_000.0,
          "initial_feasible_cost": float("nan"),
          "best_fitness": unpen_final,
          "generations_completed": 1,
          "stop_reason": "max_generations",
        }
      return feas_tour, {
        "initial_fitness": 50_000.0,
        "initial_feasible_cost": unpen_final * 2.0,
        "best_fitness": unpen_final,
        "generations_completed": 1,
        "stop_reason": "max_generations",
      }

  summary = run_single_algorithm(
    algorithm_name="Gen0Spy",
    algorithm_instance=_Gen0SpyAlgo(),
    problem=prob,
    problem_name=prob.name,
    problem_size=prob.dimension,
    customers=prob.default_evolutionary_nodes,
    max_generations=1,
    optimal_cost=None,
    repetitions=2,
    use_gpu=False,
  )
  assert math.isnan(summary["raw_initial_costs"][0])
  assert summary["mean_improvement"] == pytest.approx(50.0)
