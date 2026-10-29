"""
Composable, Strategy-Driven Genetic Algorithm Orchestrator.

Every evolutionary phase (initialization, selection, crossover, mutation,
local search, and fitness evaluation) is injected as a modular strategy
sharing the standardized `Route` / `Routes` (`RouteOrRoutes`) contract.

Because all route-transforming strategies accept `RouteOrRoutes` (`(n_cities,)`
or `(N, n_cities)`), `local_search` (e.g., 2-opt) can be placed at any point in
the pipeline via `local_search_stage`:
- `"post_crossover"`: immediately after crossover
- `"post_mutation"`: after mutation (canonical memetic GA default)
- `"post_evaluation"`: after fitness evaluation and `(mu + lambda)` survival
- `"post_evolution"`: only once at the end of evolution on the full population
- `"none"`: disabled
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from protocols.algorithm_strategies import (
  LocalSearchStage,
  Routes,
)
from utils.phase_timer import PhaseTimer

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.algorithm_strategies import (
    CrossoverStrategy,
    EvaluationStrategy,
    Fitness,
    InitializationStrategy,
    LocalSearchStrategy,
    MutationStrategy,
    Route,
    SelectionStrategy,
  )
  from protocols.problem_context import ProblemContext

  from ..monolithic.full_gpu_monolithic import GeneticAlgorithmFullGPUMonolithic


class GeneticAlgorithm:
  """
  Modular Genetic Algorithm orchestrated via injectable operator strategies.

  Supports CPU, Hybrid-Naive, Hybrid-Optimized, and Modular Full-GPU execution
  through strategy composition, as well as customizable local-search placement.
  """

  def __init__(
    self,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    seed: int | None = 42,
    *,
    initialization: InitializationStrategy | None = None,
    selection: SelectionStrategy | None = None,
    crossover: CrossoverStrategy | None = None,
    mutation: MutationStrategy | None = None,
    local_search: LocalSearchStrategy | None = None,
    evaluation: EvaluationStrategy | None = None,
    local_search_stage: LocalSearchStage = "post_mutation",
    enable_phase_timing: bool = True,
    **legacy_kwargs: Any,
  ) -> None:
    """
    Initialize the modular Genetic Algorithm with injectable strategies.

    Args:
        population_size: Number of routes in the population (`N`).
        mutation_rate: Probability of mutating each offspring route.
        seed: Random seed for reproducibility.
        initialization: Strategy to generate initial `(N, n_cities)` routes.
        selection: Strategy to select parent routes from `(routes, fitness)`.
        crossover: Strategy to recombine parent routes into offspring routes.
        mutation: Strategy to mutate routes with probability `mutation_rate`.
        local_search: Strategy to refine a single route or batch of routes (e.g., 2-opt).
        evaluation: Strategy to compute route costs `(N,)`.
        local_search_stage: Where to apply `local_search` in the pipeline:
            `"post_crossover"`, `"post_mutation"`, `"post_evaluation"`,
            `"post_evolution"`, or `"none"`.
        enable_phase_timing: Record per-phase execution times via `PhaseTimer`.
        **legacy_kwargs: Optional strategy parameters (`tournament_size`, `two_opt_iterations`)
            for backward compatibility.
    """
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.seed: int | None = seed
    self.local_search_stage: LocalSearchStage = local_search_stage

    # Store legacy attributes if passed or assign sensible defaults
    self.tournament_size: int = int(legacy_kwargs.get("tournament_size", 5))
    self.two_opt_iterations: int = int(legacy_kwargs.get("two_opt_iterations", 10))

    if seed is not None:
      np.random.seed(seed)
      if CUPY_AVAILABLE and cp is not None:
        try:
          cp.random.seed(seed)
        except Exception:
          logging.debug("CuPy seed skipped (no active CUDA device).")

    # Lazy default strategy initialization to keep this module decoupled from concrete backends
    if initialization is None:
      from ..strategies.initialization import RandomPermutationInitialization

      initialization = RandomPermutationInitialization(seed=seed)

    if selection is None:
      from ..strategies.selection import TournamentSelection

      selection = TournamentSelection(k=self.tournament_size, seed=seed)

    if crossover is None:
      from ..strategies.crossover import OrderCrossover

      crossover = OrderCrossover(seed=seed)

    if mutation is None:
      from ..strategies.mutation import SwapMutation

      mutation = SwapMutation(seed=seed)

    if local_search is None:
      from ..strategies.local_search import CPUTwoOpt

      local_search = CPUTwoOpt(max_iterations=self.two_opt_iterations)

    if evaluation is None:
      from ..strategies.evaluation import CPUCostEvaluation

      evaluation = CPUCostEvaluation()

    self.initialization: InitializationStrategy = initialization
    self.selection: SelectionStrategy = selection
    self.crossover: CrossoverStrategy = crossover
    self.mutation: MutationStrategy = mutation
    self.local_search: LocalSearchStrategy = local_search
    self.evaluation: EvaluationStrategy = evaluation

    self.timer = PhaseTimer(enabled=enable_phase_timing)
    self.generation: int = 0
    self.best_cost_history: list[float] = []
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  # ===========================================================================
  # Factory Presets (Delegated to algorithms.presets)
  # ===========================================================================

  @classmethod
  def cpu(
    cls,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
  ) -> GeneticAlgorithm:
    """Create a pure CPU NumPy Genetic Algorithm preset."""
    from ..presets import create_cpu_ga

    return create_cpu_ga(
      population_size=population_size,
      mutation_rate=mutation_rate,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      seed=seed,
      local_search_stage=local_search_stage,
    )

  @classmethod
  def hybrid_naive(
    cls,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
  ) -> GeneticAlgorithm:
    """Create the Hybrid-Naive GA preset (per-route GPU 2-opt via `two_opt_single.cu`)."""
    from ..presets import create_hybrid_naive_ga

    return create_hybrid_naive_ga(
      population_size=population_size,
      mutation_rate=mutation_rate,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      seed=seed,
      local_search_stage=local_search_stage,
    )

  @classmethod
  def hybrid_optimized(
    cls,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
  ) -> GeneticAlgorithm:
    """Create the Hybrid-Optimized GA preset (batch GPU 2-opt + batch GPU cost calculator)."""
    from ..presets import create_hybrid_optimized_ga

    return create_hybrid_optimized_ga(
      population_size=population_size,
      mutation_rate=mutation_rate,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      seed=seed,
      local_search_stage=local_search_stage,
    )

  @classmethod
  def full_gpu_modular(
    cls,
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
    local_search_stage: LocalSearchStage = "post_mutation",
  ) -> GeneticAlgorithm:
    """Create the Modular Full-GPU GA preset using decomposed per-phase CUDA kernels."""
    from ..presets import create_full_gpu_modular_ga

    return create_full_gpu_modular_ga(
      population_size=population_size,
      mutation_rate=mutation_rate,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      seed=seed,
      local_search_stage=local_search_stage,
    )

  @staticmethod
  def full_gpu_monolithic(
    population_size: int = 256,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    two_opt_iterations: int = 10,
    seed: int | None = 42,
  ) -> GeneticAlgorithmFullGPUMonolithic:
    """Create the separated Monolithic Full-GPU GA baseline (`ga_fujimoto_early_stop.cu`)."""
    from ..monolithic.full_gpu_monolithic import GeneticAlgorithmFullGPUMonolithic

    return GeneticAlgorithmFullGPUMonolithic(
      population_size=population_size,
      mutation_rate=mutation_rate,
      tournament_size=tournament_size,
      two_opt_iterations=two_opt_iterations,
      seed=seed,
    )

  # ===========================================================================
  # Internal Helpers
  # ===========================================================================

  def _all_strategies(self) -> list[Any]:
    """Return list of active strategy objects for metric aggregation."""
    return [
      self.initialization,
      self.selection,
      self.crossover,
      self.mutation,
      self.local_search,
      self.evaluation,
    ]

  def _reset_run_metrics(self) -> None:
    """Reset counters on self and all injected strategies before a run."""
    self.generation = 0
    self.best_cost_history = []
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0
    self.timer.reset()
    for strat in self._all_strategies():
      if hasattr(strat, "reset_metrics"):
        strat.reset_metrics()

  def _collect_strategy_metrics(self, context: ProblemContext) -> None:
    """Aggregate H2D, D2H, and kernel launch counts from all active strategies."""
    total_h2d = 0
    total_d2h = 0
    total_kernels = 0
    for strat in self._all_strategies():
      total_h2d += int(getattr(strat, "h2d_bytes", 0))
      total_d2h += int(getattr(strat, "d2h_bytes", 0))
      total_kernels += int(getattr(strat, "kernel_launches", 0))

    # Include one-time GPU distance matrix transfer if GPU was used
    gpu_dist = getattr(context, "_gpu_distances", None)
    if gpu_dist is not None and hasattr(gpu_dist, "nbytes"):
      total_h2d += int(gpu_dist.nbytes)

    self.h2d_bytes = total_h2d
    self.d2h_bytes = total_d2h
    self.kernel_launches = total_kernels

  def _select_survivors(
    self,
    old_pop: Routes,
    old_fitness: Fitness,
    new_pop: Routes,
    new_fitness: Fitness,
  ) -> tuple[Routes, Fitness]:
    """
    Deterministic `(mu + lambda)` elitist survival selection.

    Combines `old_pop` and `new_pop` `(2N, n_cities)` and keeps the top `N`
    routes with lowest cost. Works transparently on NumPy or CuPy arrays.
    """
    is_gpu: bool = CUPY_AVAILABLE and cp is not None and isinstance(old_pop, cp.ndarray)
    xp: Any = cp if is_gpu else np

    combined_pop = xp.vstack([old_pop, new_pop])
    combined_fitness = xp.concatenate([xp.asarray(old_fitness), xp.asarray(new_fitness)])
    best_indices = xp.argsort(combined_fitness)[: self.population_size]
    return combined_pop[best_indices], combined_fitness[best_indices]

  @staticmethod
  def _extract_best(population: Routes, fitness: Fitness) -> tuple[Route, float]:
    """Return `(best_route_np, best_cost_float)` from population and fitness."""
    if CUPY_AVAILABLE and cp is not None and isinstance(fitness, cp.ndarray):
      best_idx = int(cp.argmin(fitness).get())
      best_cost = float(fitness[best_idx].get())
    else:
      best_idx = int(np.argmin(fitness))
      best_cost = float(fitness[best_idx])

    best_route_raw = population[best_idx]
    if CUPY_AVAILABLE and cp is not None and isinstance(best_route_raw, cp.ndarray):
      best_route: Route = cp.asnumpy(best_route_raw).astype(np.int32)
    else:
      best_route = np.asarray(best_route_raw, dtype=np.int32).copy()
    return best_route, best_cost

  # ===========================================================================
  # Main Evolutionary Loop
  # ===========================================================================

  def evolve(
    self,
    context: ProblemContext,
    customers: Sequence[int],
    max_generations: int,
    optimal_cost: float | None = None,
    patience: int | None = None,
  ) -> tuple[Route, dict[str, Any]]:
    """
    Execute the modular Genetic Algorithm evolutionary pipeline.

    Args:
        context: `ProblemContext` providing distance matrices and RNG state.
        customers: Sequence of customer indices `[0..n-1]`.
        max_generations: Maximum number of generations to evolve.
        optimal_cost: Optional known optimal tour cost for early stopping.
        patience: Stagnation window in generations (`None` = `2 * sqrt(n)`).

    Returns:
        Tuple `(best_tour, stats)` where `best_tour` is a 1D `Route` `(n_cities,)`
        and `stats` contains convergence, transfer, and phase-timing metrics.
    """
    n: int = len(customers)
    if patience is None:
      patience = int(2 * math.sqrt(n))

    self._reset_run_metrics()

    # 1. Initialize population of routes: shape (population_size, n)
    population: Routes = self.timer.measure(
      "initialization",
      self.initialization.initialize,
      customers,
      self.population_size,
      context,
    )

    # 2. Initial population evaluation & local search handling
    if self.local_search_stage in ("post_crossover", "post_mutation"):
      # Canonical memetic initialization: refine initial random permutations prior to evaluation
      population = self.timer.measure(
        "local_search",
        self.local_search.improve_batch,
        population,
        context,
      )
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
    elif self.local_search_stage == "post_evaluation":
      # Post-evaluation: evaluate the initial population first, then refine and re-evaluate
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
      population = self.timer.measure(
        "local_search",
        self.local_search.improve_batch,
        population,
        context,
      )
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
    else:
      # "post_evolution" or "none": initial evaluation without local search
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )

    best_tour, best_cost = self._extract_best(population, fitness)
    initial_best_cost = best_cost
    self.best_cost_history = [best_cost]

    generations_without_improvement = 0
    stopped_early = False
    stop_reason = "max_generations"

    # 3. Generational evolution loop
    for gen in range(max_generations):
      self.generation = gen + 1

      # Step A: Parent selection -> (population_size, n)
      parents = self.timer.measure(
        "selection",
        self.selection.select_batch,
        population,
        fitness,
        self.population_size,
        context,
      )

      # Step B: Crossover -> (population_size, n)
      offspring = self.timer.measure(
        "crossover",
        self.crossover.crossover_batch,
        parents,
        context,
      )

      # Optional stage: immediately after crossover
      if self.local_search_stage == "post_crossover":
        offspring = self.timer.measure(
          "local_search",
          self.local_search.improve_batch,
          offspring,
          context,
        )

      # Step C: Mutation -> (population_size, n)
      offspring = self.timer.measure(
        "mutation",
        self.mutation.mutate_batch,
        offspring,
        self.mutation_rate,
        context,
      )

      # Canonical memetic stage: after mutation
      if self.local_search_stage == "post_mutation":
        offspring = self.timer.measure(
          "local_search",
          self.local_search.improve_batch,
          offspring,
          context,
        )

      # Step D: Evaluate offspring fitness -> (population_size,)
      offspring_fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        offspring,
        context,
      )

      # Step E: Deterministic (mu + lambda) elitist survival -> (population_size, n)
      population, fitness = self.timer.measure(
        "survival",
        self._select_survivors,
        population,
        fitness,
        offspring,
        offspring_fitness,
      )

      # Post-evaluation stage: applied after fitness evaluation & survival each generation
      if self.local_search_stage == "post_evaluation":
        population = self.timer.measure(
          "local_search",
          self.local_search.improve_batch,
          population,
          context,
        )
        fitness = self.timer.measure(
          "evaluation",
          self.evaluation.evaluate_batch,
          population,
          context,
        )

      gen_best_tour, gen_best_cost = self._extract_best(population, fitness)
      if gen_best_cost < best_cost - 1e-6:
        best_cost = gen_best_cost
        best_tour = gen_best_tour
        generations_without_improvement = 0
      else:
        generations_without_improvement += 1

      self.best_cost_history.append(best_cost)

      # Early stopping check 1: true optimal reached or within 1% of known optimal cost
      if optimal_cost is not None and optimal_cost > 0:
        if best_cost <= optimal_cost + 1e-6:
          stopped_early = True
          stop_reason = "hit_optimal"
          break
        gap_percent: float = (best_cost - optimal_cost) / optimal_cost * 100.0
        if gap_percent < 1.0:
          stopped_early = True
          stop_reason = "within_1pct_optimal"
          break

      # Early stopping check 2: stagnation patience exceeded
      if generations_without_improvement >= patience:
        stopped_early = True
        stop_reason = "no_improvements"
        break

    # 4. Post-evolution stage: refine the full final population after all evolution finishes
    if self.local_search_stage == "post_evolution":
      population = self.timer.measure(
        "local_search",
        self.local_search.improve_batch,
        population,
        context,
      )
      fitness = self.timer.measure(
        "evaluation",
        self.evaluation.evaluate_batch,
        population,
        context,
      )
      post_best_tour, post_best_cost = self._extract_best(population, fitness)
      if post_best_cost <= best_cost:
        best_tour = post_best_tour
        best_cost = post_best_cost
        self.best_cost_history[-1] = best_cost

    self._collect_strategy_metrics(context)

    improvement_pct = (initial_best_cost - best_cost) / initial_best_cost * 100.0 if initial_best_cost > 0 else 0.0

    stats: dict[str, Any] = {
      "best_fitness": best_cost,
      "initial_fitness": initial_best_cost,
      "improvement_pct": improvement_pct,
      "best_cost_history": self.best_cost_history,
      "h2d_bytes": self.h2d_bytes,
      "d2h_bytes": self.d2h_bytes,
      "kernel_launches": self.kernel_launches,
      "generations_completed": self.generation,
      "stopped_early": stopped_early,
      "stop_reason": stop_reason,
      "local_search_stage": self.local_search_stage,
      "phase_timings": self.timer.summary(),
    }
    return best_tour, stats


__all__: list[str] = [
  "GeneticAlgorithm",
  "LocalSearchStage",
]
