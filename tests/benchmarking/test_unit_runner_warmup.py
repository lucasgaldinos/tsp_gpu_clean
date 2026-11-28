"""
Unit tests for the GPU warm-up / AOT pre-compilation protocol (DP11-12, DP11-21, BUG-23, BUG-25).

Verifies that:
1. When `use_gpu=True`, `run_single_algorithm` warms an isolated clone once, before the
   measured repetitions, and records no metrics for it.
2. When `use_gpu=False`, zero warm-ups run.
3. Warm-up exceptions do not crash benchmark execution.
4. Warm-up still fires when resuming from an existing checkpoint.
5. A kernel compiled inside the measured loop fails the run (`BUG-25` regression guard).
6. `GeneticAlgorithm.warmup` reaches a `PeriodicLocalSearchStep` that never fires at
   generation 1, which is the exact blind spot of the previous one-generation warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from algorithms.pipeline_steps import PeriodicLocalSearchStep
from algorithms.strategies.crossover.order_crossover import CrossoverCPUPermOrder
from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost
from algorithms.strategies.initialization import InitializationCPUPermRandom
from algorithms.strategies.local_search.two_opt import LocalSearchCPUTourTwoOpt
from algorithms.strategies.mutation.swap_mutation import MutationCPUPermSwap
from algorithms.strategies.selection.tournament_selection import SelectionCPUTournament
from benchmarking.runner.algorithm_runner import run_single_algorithm
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext
from utils.gpu_helpers import _compile_kernel_cached

if TYPE_CHECKING:
  from data_models.problem import Problem


@dataclass
class _RecordingAlgorithm:
  """Stand-in GA that records warm-up and evolve calls instead of running one.

  A plain dataclass (not a `MagicMock`) because the runner deep-copies it: the clone
  must be a genuinely independent object for the isolation assertions to mean anything.
  """

  dimension: int
  warmup_calls: list[ProblemContext] = field(default_factory=list)
  evolve_generations: list[int] = field(default_factory=list)
  warmup_error: Exception | None = None

  def warmup(self, context: ProblemContext) -> None:
    """Record the warm-up and optionally fail, like an unavailable CUDA runtime would."""
    if self.warmup_error is not None:
      raise self.warmup_error
    self.warmup_calls.append(context)

  def evolve(
    self,
    context: ProblemContext,
    customers: Any,
    max_generations: int,
    optimal_cost: float | None = None,
    patience: int | None = None,
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic tour and the stats the runner consumes."""
    del context, customers, optimal_cost, patience
    self.evolve_generations.append(max_generations)
    stats: dict[str, Any] = {
      "initial_fitness": 100.0,
      "best_fitness": 50.0,
      "generations_completed": max_generations,
      "stop_reason": "completed",
      "phase_timings": {
        "initialization": {"total_ms": 1.0, "calls": 1},
        "evaluation": {"total_ms": 2.0, "calls": 2},
        "host_sync": {"total_ms": 0.5, "calls": 2},
      },
      "h2d_bytes": 0,
      "d2h_bytes": 0,
      "kernel_launches": 0,
    }
    return np.arange(self.dimension, dtype=np.int32), stats


class _MutableJitCache:
  """Fake `_compile_kernel_cached` whose `currsize` the test can move."""

  def __init__(self) -> None:
    self.currsize: int = 0

  def cache_info(self) -> SimpleNamespace:
    """Mimic `functools.lru_cache`'s introspection surface."""
    return SimpleNamespace(currsize=self.currsize)


@pytest.mark.gpu
def test_runner_gpu_warms_an_isolated_clone_once(burma14_problem: Problem) -> None:
  """Verify GPU execution pre-compiles once, on a clone, before the measured loop."""
  algorithm = _RecordingAlgorithm(dimension=burma14_problem.dimension)

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="RecordingGPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=3,
    seed_base=42,
    use_gpu=True,
  )

  # The measured instance was never warmed: only its deep copy was.
  assert algorithm.warmup_calls == []
  # Exactly `repetitions` evolve calls, all with the measured generation budget.
  assert algorithm.evolve_generations == [5, 5, 5]
  assert summary["repetitions"] == 3
  assert summary["successful_runs"] == 3
  assert len(summary["raw_times"]) == 3


@pytest.mark.gpu
def test_runner_gpu_warmup_context_targets_the_real_problem(burma14_problem: Problem) -> None:
  """Verify the warm-up receives the measured problem, never a synthetic dummy batch."""
  seen: list[ProblemContext] = []

  class _ContextProbe(_RecordingAlgorithm):
    def warmup(self, context: ProblemContext) -> None:
      """Record the context before deferring to the base recorder."""
      seen.append(context)
      super().warmup(context)

  algorithm = _ContextProbe(dimension=burma14_problem.dimension)
  run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="ProbeGPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=2,
    repetitions=1,
    seed_base=42,
    use_gpu=True,
  )

  assert len(seen) == 1
  assert seen[0].problem is burma14_problem
  assert seen[0].seed == 42


def test_runner_cpu_omits_warmup_run(burma14_problem: Problem) -> None:
  """Verify that CPU execution executes zero warm-up runs."""
  algorithm = _RecordingAlgorithm(dimension=burma14_problem.dimension)

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="RecordingCPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=3,
    seed_base=42,
    use_gpu=False,
  )

  assert algorithm.warmup_calls == []
  assert algorithm.evolve_generations == [5, 5, 5]
  assert summary["repetitions"] == 3
  assert len(summary["raw_times"]) == 3


@pytest.mark.gpu
def test_runner_gpu_warmup_failure_tolerated(burma14_problem: Problem) -> None:
  """Verify that a failing warm-up leaves the measured repetitions intact."""
  algorithm = _RecordingAlgorithm(
    dimension=burma14_problem.dimension,
    warmup_error=RuntimeError("CUDA out of memory during warmup"),
  )

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="RecordingGPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=2,
    seed_base=42,
    use_gpu=True,
  )

  assert summary["successful_runs"] == 2
  assert len(summary["raw_times"]) == 2


@pytest.mark.gpu
def test_runner_gpu_fires_warmup_run_in_incremental_mode(burma14_problem: Problem) -> None:
  """Verify that GPU warm-up still fires when resuming from an existing checkpoint."""
  algorithm = _RecordingAlgorithm(dimension=burma14_problem.dimension)
  existing_checkpoint: dict[str, Any] = {
    "algorithm": "RecordingGPUAlgorithm",
    "repetitions": 5,
    "successful_runs": 2,
    "mean_time": 0.05,
    "mean_cost": 100.0,
    "raw_times": [0.05, 0.05],
    "raw_initial_costs": [120.0, 120.0],
    "raw_costs": [100.0, 100.0],
    "raw_gaps": [5.0, 5.0],
    "raw_seeds": [42, 43],
    "raw_generations": [5, 5],
    "raw_stop_reasons": ["completed", "completed"],
    "raw_phase_timings": [{}, {}],
    "raw_best_tours": [[0] * 14, [0] * 14],
  }

  summary = run_single_algorithm(
    algorithm_instance=algorithm,
    algorithm_name="RecordingGPUAlgorithm",
    problem=burma14_problem,
    problem_name="burma14",
    problem_size=burma14_problem.dimension,
    optimal_cost=30.0,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
    repetitions=3,
    seed_base=42,
    use_gpu=True,
    incremental_mode=True,
    existing_checkpoint=existing_checkpoint,
  )

  assert summary["successful_runs"] == 5
  # 3 new repetitions only; the warm-up never reaches the measured instance.
  assert algorithm.evolve_generations == [5, 5, 5]


@pytest.mark.gpu
def test_runner_gpu_rejects_jit_compiled_inside_the_measured_loop(
  burma14_problem: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verify the run fails loudly when a kernel is compiled after the warm-up.

  WHY: `BUG-25` — the whole point of the protocol is that `currsize` is constant during
  the measured repetitions. A stale warm-up must not be reported as a valid cohort.
  """
  jit_cache = _MutableJitCache()
  algorithm = _RecordingAlgorithm(dimension=burma14_problem.dimension)

  def _evolve_that_compiles(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    jit_cache.currsize += 1
    return _RecordingAlgorithm.evolve(algorithm, *args, **kwargs)

  algorithm.evolve = _evolve_that_compiles  # type: ignore[method-assign]  # instance-level spy in a test
  monkeypatch.setattr("benchmarking.runner.algorithm_runner._compile_kernel_cached", jit_cache)

  with pytest.raises(RuntimeError, match="JIT compilation landed inside the measured loop"):
    run_single_algorithm(
      algorithm_instance=algorithm,
      algorithm_name="RecordingGPUAlgorithm",
      problem=burma14_problem,
      problem_name="burma14",
      problem_size=burma14_problem.dimension,
      optimal_cost=30.0,
      customers=list(range(burma14_problem.dimension)),
      max_generations=5,
      repetitions=2,
      seed_base=42,
      use_gpu=True,
    )


class _WarmupSpy(LocalSearchCPUTourTwoOpt):
  """CPU local search that records warm-up calls while keeping the real contract."""

  def __init__(self) -> None:
    super().__init__()
    self.warmup_contexts: list[ProblemContext] = []

  def warmup(self, context: ProblemContext) -> None:
    """Record the warm-up instead of compiling (the CPU path has no kernel)."""
    self.warmup_contexts.append(context)


def test_ga_warmup_covers_periodic_local_search_step(burma14_problem: Problem) -> None:
  """Verify `GeneticAlgorithm.warmup` reaches a step that never fires at generation 1.

  WHY: `PeriodicLocalSearchStep(interval=10)` is only executed at generation 10, so the
  old one-generation warm-up left `two_opt_batch.cu` to be compiled by the first measured
  repetition. This CPU twin asserts the *coverage* invariant without a GPU.
  """
  local_search = _WarmupSpy()
  evaluator = EvaluationCPUCost()
  ga = GeneticAlgorithm(
    population_size=8,
    mutation_rate=0.02,
    seed=42,
    initialization=InitializationCPUPermRandom(seed=42),
    selection=SelectionCPUTournament(k=3, seed=42),
    crossover=CrossoverCPUPermOrder(seed=42),
    mutation=MutationCPUPermSwap(seed=42),
    evaluation=evaluator,
    local_search=None,
    local_search_stage="none",
    pipeline=[
      PeriodicLocalSearchStep(
        local_search=local_search,
        interval=10,
        target="population",
        re_evaluate=evaluator,
      )
    ],
  )
  context = ProblemContext(burma14_problem, xp=np, seed=42)

  ga.warmup(context)

  assert local_search.warmup_contexts == [context], "the periodic step's strategy must be warmed"
  assert ga.generation == 0, "warming up must not advance the generation counter"


@pytest.mark.gpu
def test_gpu_strategy_warmup_leaves_no_jit_for_measured_calls() -> None:
  """Verify GPU strategies compile on `warmup` and perform zero JIT afterwards.

  Uses a valid permutation batch (`cp.arange`), never `cp.zeros` (`BUG-24`).
  """
  from algorithms.strategies.evaluation.cost_evaluation import EvaluationGPUCost
  from algorithms.strategies.local_search.two_opt import LocalSearchGPUTourTwoOpt
  from data_models.problem import Problem

  cp = cupy_backend()
  if cp is None:
    pytest.skip("CUDA device or CuPy unavailable")

  n = 30
  rng = np.random.default_rng(7)
  problem = Problem(
    name="warmup_gpu",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=rng.uniform(0.0, 100.0, size=(n, 2)),
    distances=None,
    capacity=None,
    demands=None,
  )
  context = ProblemContext(problem, xp=cp, seed=3)
  evaluator = EvaluationGPUCost()
  local_search = LocalSearchGPUTourTwoOpt(mode="batch", max_iterations=1)

  evaluator.warmup(context)
  local_search.warmup(context)
  before = _compile_kernel_cached.cache_info().currsize
  assert before >= 1, "warmup must compile at least one kernel"

  population = cp.tile(cp.arange(n, dtype=cp.int32), (8, 1))
  for _ in range(3):
    evaluator.evaluate_batch(population, context)
    local_search.improve_batch(population, context)

  assert _compile_kernel_cached.cache_info().currsize == before
