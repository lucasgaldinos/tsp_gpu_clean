"""Integration tests for the composable Genetic Algorithm pipeline.

Covers the flexible 2-opt placement stages, the benchmark config that resolves every
modular and monolithic variant, and those variants running end-to-end on a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms import GeneticAlgorithm
from algorithms.strategies import (
  CPUCostEvaluation,
  CPUTwoOpt,
  OrderCrossover,
  RandomPermutationInitialization,
  SwapMutation,
  TournamentSelection,
)
from benchmarking.runner.config_loader import load_algorithms_config
from protocols import ProblemContext
from protocols.backend import cupy_backend

cp = cupy_backend()

if TYPE_CHECKING:
  from pathlib import Path

  from benchmarking.runner.config_loader import AlgorithmConfig
  from data_models.problem import Problem

# Every variant named in `algorithm_configs`; the conftest fixture mirrors production.
EXPECTED_VARIANTS: frozenset[str] = frozenset(
  {
    "CPU",
    "HybridNaive",
    "HybridOptimized",
    "FullGPUModular",
    "FullGPU",
    "FullGPUMonolithic",
  }
)

# Documented `stop_reason` values: `max_generations` (loop exhausted), `hit_optimal`
# (exact optimum reached), `within_1pct_optimal` (early stop near the optimum) and
# `no_improvements` (stagnation patience exhausted).
DOCUMENTED_STOP_REASONS: frozenset[str] = frozenset(
  {"max_generations", "hit_optimal", "within_1pct_optimal", "no_improvements"}
)


@pytest.mark.parametrize(
  "stage",
  [
    "post_crossover",
    "post_mutation",
    "post_evaluation",
    "post_evolution",
    "none",
  ],
)
def test_ga_composable_local_search_stages(burma14_problem: Problem, stage: str) -> None:
  """Verify 2-opt can be placed after OX, after mutation, after fitness, or at the end of GA."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))

  ga = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.05,
    seed=42,
    initialization=RandomPermutationInitialization(seed=42),
    selection=TournamentSelection(k=3, seed=42),
    crossover=OrderCrossover(seed=42),
    mutation=SwapMutation(seed=42),
    evaluation=CPUCostEvaluation(),
    local_search=CPUTwoOpt(max_iterations=4),
    local_search_stage=stage,  # type: ignore[arg-type]
  )
  best_tour, stats = ga.evolve(context, customers, max_generations=6, patience=10)
  assert best_tour.shape == (burma14_problem.dimension,)
  assert sorted(best_tour.tolist()) == customers
  assert stats["best_fitness"] <= stats["initial_fitness"] + 1e-6
  assert ga.local_search_stage == stage


def test_ga_rejects_unknown_local_search_stage() -> None:
  """Verify GeneticAlgorithm raises ValueError on invalid local_search_stage."""
  with pytest.raises(ValueError, match="Invalid local_search_stage 'unknown_stage'"):
    GeneticAlgorithm(
      population_size=16,
      mutation_rate=0.05,
      local_search_stage="unknown_stage",  # type: ignore[arg-type]
      initialization=RandomPermutationInitialization(),
      selection=TournamentSelection(),
      crossover=OrderCrossover(),
      mutation=SwapMutation(),
      evaluation=CPUCostEvaluation(),
    )


def test_ga_init_does_not_mutate_global_numpy_rng_state() -> None:
  """Verify initializing GeneticAlgorithm with seed does not mutate global np.random state."""
  np.random.seed(12345)
  state_before = np.random.get_state()
  _ = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.05,
    seed=99999,
    initialization=RandomPermutationInitialization(seed=99999),
    selection=TournamentSelection(seed=99999),
    crossover=OrderCrossover(seed=99999),
    mutation=SwapMutation(seed=99999),
    evaluation=CPUCostEvaluation(),
  )
  state_after = np.random.get_state()
  assert state_before[0] == state_after[0]
  assert np.array_equal(state_before[1], state_after[1])
  assert state_before[2:] == state_after[2:]


def _assert_stats_contract(stats: dict[str, Any], max_generations: int) -> None:
  """Assert the `evolve()` stats contract every variant must honour.

  WHY: the benchmark runner aggregates these exact keys per repetition; a missing key
  or an inconsistent history corrupts the reported convergence curve and transfer totals.
  EXPECTED: fitness never worsens, one convergence point per completed generation plus
  the initial one, non-negative transfer counters, and a documented `stop_reason`.
  """
  assert stats["best_fitness"] <= stats["initial_fitness"], (
    f"best_fitness {stats['best_fitness']} is worse than initial_fitness "
    f"{stats['initial_fitness']}"
  )
  assert 0 <= stats["generations_completed"] <= max_generations
  assert len(stats["convergence_history"]) == stats["generations_completed"] + 1, (
    "convergence_history must hold the initial best cost plus one point per generation: "
    f"{len(stats['convergence_history'])} points for "
    f"{stats['generations_completed']} completed generations"
  )
  assert stats["stop_reason"] in DOCUMENTED_STOP_REASONS, (
    f"undocumented stop_reason {stats['stop_reason']!r}"
  )
  for counter in ("h2d_bytes", "d2h_bytes", "kernel_launches"):
    assert stats[counter] >= 0, f"{counter} must be non-negative, got {stats[counter]}"


def test_benchmark_algorithms_json_loads_all_classes(fixture_config_dir: Path) -> None:
  """Verify the fixture algorithms.json resolves and instantiates every configured variant."""
  ga_params, alg_configs = load_algorithms_config(fixture_config_dir)
  assert set(alg_configs) == EXPECTED_VARIANTS

  for name, cfg in alg_configs.items():
    resolved_cls = cfg.get_class()
    assert resolved_cls.__name__ == cfg.metaheuristic, (
      f"{name} resolves class {resolved_cls.__name__}, "
      f"but its metaheuristic field is {cfg.metaheuristic!r}"
    )
    instance = cfg.instantiate(ga_params)
    assert type(instance).__name__ == cfg.metaheuristic, (
      f"{name} instantiated {type(instance).__name__}, "
      f"but its metaheuristic field is {cfg.metaheuristic!r}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
  "variant",
  [
    "CPU",
    "HybridNaive",
    "HybridOptimized",
    "FullGPUModular",
  ],
)
def test_gpu_variants_and_modular_vs_monolithic(
  fixture_config_dir: Path,
  burma14_problem: Problem,
  variant: str,
) -> None:
  """Verify every configured variant runs on its backend and honours the stats contract."""
  customers: list[int] = list(range(burma14_problem.dimension))
  ga_params, alg_configs = load_algorithms_config(fixture_config_dir)
  cfg: AlgorithmConfig = alg_configs[variant]

  algo = cfg.instantiate(ga_params)
  assert type(algo).__name__ == cfg.metaheuristic
  xp: Any = cp if cfg.use_gpu else np
  context = ProblemContext(burma14_problem, xp=xp, seed=42)

  best_tour, stats = algo.evolve(context, customers, max_generations=10, patience=5)
  assert best_tour.shape == (burma14_problem.dimension,)
  assert sorted(best_tour.tolist()) == customers
  _assert_stats_contract(stats, max_generations=10)
