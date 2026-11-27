"""Unit tests for plugging external strategies and pipeline steps without modifying ``src/``.

Covers all three extension mechanisms:
1. Direct Python dependency injection into ``GeneticAlgorithm`` (PEP 544 protocols).
2. Runtime registration via ``register_strategy`` and ``register_pipeline_step``.
3. Dynamic external module loading via ``{"module": "...", "class": "...", "params": {...}}``
   in declarative ``AlgorithmConfig`` JSON specifications.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms import GeneticAlgorithm, register_pipeline_step, register_strategy
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
from algorithms.strategies import (
  STRATEGY_REGISTRY,
  CrossoverOrder,
  EvaluationCPUCost,
  InitializationCPURandom,
  SelectionTournament,
  StrategyBase,
)
from benchmarking.runner.config_loader import AlgorithmConfig, GAParams
from protocols.algorithm_strategies import LocalSearchStrategy, MutationStrategy
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from pathlib import Path

  from data_models.problem import Problem


class _ExternalShiftMutation(StrategyBase):
  """Custom mutation strategy defined outside ``src/``."""

  def __init__(self, shift_by: int = 1, seed: int | None = None) -> None:
    super().__init__(seed=seed)
    self.shift_by = shift_by

  def mutate(self, individual: Any, mutation_rate: float, context: Any = None) -> Any:
    del context
    route = np.array(individual, copy=True)
    if self.rng.random() < mutation_rate:
      route = np.roll(route, self.shift_by)
    return route

  def mutate_batch(self, routes: Any, mutation_rate: float, context: Any = None) -> Any:
    arr = np.asarray(routes)
    if arr.ndim == 1:
      return self.mutate(arr, mutation_rate, context)
    out = arr.copy()
    for i in range(out.shape[0]):
      out[i] = self.mutate(out[i], mutation_rate, context)
    return out


class _ExternalTraceStep:
  """Custom generational pipeline step defined outside ``src/``."""

  stage_anchor = "post_survival"

  def __init__(self, tag: str = "external") -> None:
    self.tag = tag
    self.generations_seen: list[int] = []

  def run(self, state: Any, timer: Any) -> None:
    del timer
    self.generations_seen.append(int(state.generation))


def test_direct_python_injection_runs_external_strategy_and_step(burma14_problem: Problem) -> None:
  """WHAT: Direct Python injection of external StrategyBase subclass and GenerationStep.

  WHY: Ensures users can compose custom operators and pipeline steps directly with
  GeneticAlgorithm via PEP 544 protocols without editing src/.
  EXPECTED: Evolution completes cleanly and executes the custom step every generation.
  """
  mutation = _ExternalShiftMutation(shift_by=2, seed=7)
  step = _ExternalTraceStep(tag="direct")

  assert isinstance(mutation, MutationStrategy)

  ga = GeneticAlgorithm(
    population_size=12,
    mutation_rate=0.25,
    seed=7,
    initialization=InitializationCPURandom(seed=7),
    selection=SelectionTournament(k=3, seed=7),
    crossover=CrossoverOrder(seed=7),
    mutation=mutation,
    evaluation=EvaluationCPUCost(),
    pipeline=[step],
  )
  context = ProblemContext(burma14_problem, xp=np, seed=7)
  best_route, stats = ga.evolve(
    context=context,
    customers=list(range(burma14_problem.dimension)),
    max_generations=5,
  )

  assert len(best_route) == burma14_problem.dimension
  assert stats["best_fitness"] > 0
  assert step.generations_seen == [1, 2, 3, 4, 5]


def test_register_strategy_and_pipeline_step_helpers(
  monkeypatch: pytest.MonkeyPatch,
  burma14_problem: Problem,
) -> None:
  """WHAT: Runtime registration helpers guard types, protocols, and duplicates.

  WHY: Allows external experiment scripts to register custom classes safely while
  failing fast on non-classes, non-conforming strategy classes, and accidental overwrites.
  EXPECTED: Valid classes register and instantiate cleanly; invalid types/protocols raise TypeError/ValueError.
  """
  monkeypatch.setitem(STRATEGY_REGISTRY, "__probe__", StrategyBase)
  monkeypatch.delitem(STRATEGY_REGISTRY, "__probe__")
  monkeypatch.setitem(PIPELINE_STEP_REGISTRY, "__probe__", _ExternalTraceStep)
  monkeypatch.delitem(PIPELINE_STEP_REGISTRY, "__probe__")

  try:
    register_strategy("ExternalShiftMutation", _ExternalShiftMutation)
    register_pipeline_step("ExternalTraceStep", _ExternalTraceStep)

    with pytest.raises(ValueError, match="already registered"):
      register_strategy("ExternalShiftMutation", _ExternalShiftMutation)
    with pytest.raises(ValueError, match="already registered"):
      register_pipeline_step("ExternalTraceStep", _ExternalTraceStep)

    register_strategy("ExternalShiftMutation", _ExternalShiftMutation, overwrite=True)
    register_pipeline_step("ExternalTraceStep", _ExternalTraceStep, overwrite=True)

    with pytest.raises(TypeError, match="Expected a strategy class"):
      register_strategy("NotAClass", "invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="does not implement any strategy protocol"):
      register_strategy("EmptyClass", type("EmptyClass", (), {}))
    with pytest.raises(TypeError, match="callable 'run"):
      register_pipeline_step("NoRunMethod", StrategyBase)

    cfg = AlgorithmConfig(
      name="CustomRegistered",
      use_gpu=False,
      description="Registered custom strategy and step",
      params={"pipeline": [{"class": "ExternalTraceStep", "params": {"tag": "reg"}}]},
      strategies={
        "initialization": {"class": "InitializationCPURandom"},
        "selection": {"class": "SelectionTournament", "params": {"k": 3}},
        "crossover": {"class": "CrossoverOrder"},
        "mutation": {"class": "ExternalShiftMutation", "params": {"shift_by": 3}},
        "evaluation": {"class": "EvaluationCPUCost"},
      },
    )
    ga = cfg.instantiate(GAParams(population_size=10, mutation_rate=0.2, seed=11))
    assert isinstance(ga.mutation, _ExternalShiftMutation)
    assert ga.mutation.shift_by == 3

    context = ProblemContext(burma14_problem, xp=np, seed=11)
    _, stats = ga.evolve(
      context=context,
      customers=list(range(burma14_problem.dimension)),
      max_generations=3,
    )
    assert stats["best_fitness"] > 0
  finally:
    STRATEGY_REGISTRY.pop("ExternalShiftMutation", None)
    PIPELINE_STEP_REGISTRY.pop("ExternalTraceStep", None)


def test_json_module_spec_loads_external_strategy_and_pipeline_step(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  burma14_problem: Problem,
) -> None:
  """WHAT: Declarative JSON {'module', 'class', 'params'} dynamic external class loading.

  WHY: Enables CLI users to benchmark custom strategies and pipeline steps from
  external Python modules without modifying src/.
  EXPECTED: External classes load, satisfy protocols, receive params, and fail fast if missing.
  """
  ext_module = tmp_path / "user_external_ops.py"
  ext_module.write_text(
    textwrap.dedent(
      """
      from __future__ import annotations
      import numpy as np
      from algorithms.strategies import StrategyBase

      class UserReverseLocalSearch(StrategyBase):
        def __init__(self, passes: int = 1, seed: int | None = None) -> None:
          super().__init__(seed=seed)
          self.passes = passes

        def improve(self, route, context):
          del context
          return np.array(route, copy=True)

        def improve_batch(self, routes, context):
          del context
          return np.array(routes, copy=True)

      class UserPeriodicAuditStep:
        stage_anchor = "post_survival"

        def __init__(self, local_search, evaluation, label: str = "audit") -> None:
          self.local_search = local_search
          self.evaluation = evaluation
          self.label = label
          self.calls = 0

        def run(self, state, timer) -> None:
          self.calls += 1
          state.population = timer.measure(
            "local_search",
            self.local_search.improve_batch,
            state.population,
            state.context,
          )
      """
    ),
    encoding="utf-8",
  )
  monkeypatch.syspath_prepend(str(tmp_path))

  cfg = AlgorithmConfig(
    name="ExternalModuleVariant",
    use_gpu=False,
    description="Loaded from external user module via JSON spec",
    params={
      "local_search_stage": "none",
      "pipeline": [
        {
          "module": "user_external_ops",
          "class": "UserPeriodicAuditStep",
          "params": {
            "label": "custom-audit",
            "local_search": {
              "module": "user_external_ops",
              "class": "UserReverseLocalSearch",
              "params": {"passes": 2},
            },
          },
        }
      ],
    },
    strategies={
      "initialization": {"class": "InitializationCPURandom"},
      "selection": {"class": "SelectionTournament", "params": {"k": 3}},
      "crossover": {"class": "CrossoverOrder"},
      "mutation": {"class": "MutationSwap"},
      "local_search": {
        "module": "user_external_ops",
        "class": "UserReverseLocalSearch",
        "params": {"passes": 4},
      },
      "evaluation": {"class": "EvaluationCPUCost"},
    },
  )

  ga = cfg.instantiate(GAParams(population_size=10, mutation_rate=0.1, seed=13))
  assert isinstance(ga.local_search, LocalSearchStrategy)
  ls_inst: Any = ga.local_search
  assert ls_inst.passes == 4

  custom_steps: list[Any] = list(ga.custom_pipeline_steps)
  assert len(custom_steps) == 1
  assert custom_steps[0].label == "custom-audit"
  assert custom_steps[0].local_search.passes == 2

  context = ProblemContext(burma14_problem, xp=np, seed=13)
  _, stats = ga.evolve(
    context=context,
    customers=list(range(burma14_problem.dimension)),
    max_generations=4,
  )
  assert stats["best_fitness"] > 0
  assert custom_steps[0].calls == 4

  bad_cfg = AlgorithmConfig(
    name="MissingExternalClass",
    use_gpu=False,
    description="Invalid external class name",
    strategies={
      "local_search": {"module": "user_external_ops", "class": "DoesNotExist"},
    },
  )
  with pytest.raises(ValueError, match="not found as a class in module 'user_external_ops'"):
    bad_cfg.instantiate(GAParams())


def test_config_relative_py_path_loads_from_hyphenated_experiment_dir(
  tmp_path: Path,
  burma14_problem: Problem,
) -> None:
  """WHAT: Config-relative `.py` external module loading inside hyphenated experiment directories (`BUG-104`, `DP16-3`).

  WHY: Experiment folders like `experiment-comparing-OX-and-PMX/` contain hyphens that Python's
  dotted `importlib.import_module` cannot traverse without `PYTHONPATH` workarounds.
  EXPECTED: `"module": "../my-ops.py"` resolves relative to `algorithms.json`, passes preflight,
  instantiates both top-level strategies and nested pipeline step strategies, and evolves cleanly.
  """
  import json

  from benchmarking.runner.config_loader import (
    BenchmarkConfig,
    ProblemConfig,
    load_algorithms_config,
    validate_configs,
  )

  exp_dir = tmp_path / "experiment-comparing-OX-and-PMX"
  configs_dir = exp_dir / "configs"
  configs_dir.mkdir(parents=True)

  (exp_dir / "my-ops.py").write_text(
    textwrap.dedent(
      """
      from __future__ import annotations
      import numpy as np
      from algorithms.strategies import StrategyBase

      class RelativeShiftMutation(StrategyBase):
        def __init__(self, shift_by: int = 2, seed: int | None = None) -> None:
          super().__init__(seed=seed)
          self.shift_by = shift_by

        def mutate(self, individual, mutation_rate: float, context=None):
          del context
          route = np.array(individual, copy=True)
          if self.rng.random() < mutation_rate:
            route = np.roll(route, self.shift_by)
          return route

        def mutate_batch(self, routes, mutation_rate: float, context=None):
          arr = np.asarray(routes)
          if arr.ndim == 1:
            return self.mutate(arr, mutation_rate, context)
          out = arr.copy()
          for i in range(out.shape[0]):
            out[i] = self.mutate(out[i], mutation_rate, context)
          return out

      class RelativeNoOpSearch(StrategyBase):
        def improve(self, route, context):
          del context
          return np.array(route, copy=True)

        def improve_batch(self, routes, context):
          del context
          return np.array(routes, copy=True)

      class RelativeCounterStep:
        stage_anchor = "post_survival"

        def __init__(self, local_search, evaluation, tag: str = "rel") -> None:
          self.local_search = local_search
          self.evaluation = evaluation
          self.tag = tag
          self.count = 0

        def run(self, state, timer) -> None:
          del timer
          self.count += 1
          state.population = self.local_search.improve_batch(state.population, state.context)
      """
    ),
    encoding="utf-8",
  )

  algo_json = configs_dir / "algorithms.json"
  algo_json.write_text(
    json.dumps(
      {
        "default_params": {"population_size": 10, "mutation_rate": 0.2, "seed": 19},
        "algorithm_configs": {
          "RelativeExp": {
            "metaheuristic": "GeneticAlgorithm",
            "use_gpu": False,
            "description": "Loaded via config-relative .py path",
            "params": {
              "local_search_stage": "none",
              "pipeline": [
                {
                  "module": "../my-ops.py",
                  "class": "RelativeCounterStep",
                  "params": {
                    "tag": "hyphen-dir",
                    "local_search": {"module": "../my-ops.py", "class": "RelativeNoOpSearch"},
                  },
                }
              ],
            },
            "strategies": {
              "initialization": {"class": "InitializationCPURandom"},
              "selection": {"class": "SelectionTournament", "params": {"k": 3}},
              "crossover": {"class": "CrossoverOrder"},
              "mutation": {
                "module": "../my-ops.py",
                "class": "RelativeShiftMutation",
                "params": {"shift_by": 3},
              },
              "local_search": {"module": "../my-ops.py", "class": "RelativeNoOpSearch"},
              "evaluation": {"class": "EvaluationCPUCost"},
            },
          }
        },
      }
    ),
    encoding="utf-8",
  )

  ga_params, algos = load_algorithms_config(algo_json)
  warnings = validate_configs(
    {
      "ga_params": ga_params,
      "algorithms": algos,
      "benchmark": BenchmarkConfig(),
      "problems": [ProblemConfig("burma14", 3323.0, 14)],
    }
  )
  assert warnings == []

  ga = algos["RelativeExp"].instantiate(ga_params)
  assert isinstance(ga.mutation, MutationStrategy)
  assert isinstance(ga.local_search, LocalSearchStrategy)
  mut_inst: Any = ga.mutation
  assert mut_inst.shift_by == 3

  context = ProblemContext(burma14_problem, xp=np, seed=19)
  _, stats = ga.evolve(context=context, customers=list(range(burma14_problem.dimension)), max_generations=3)
  assert stats["best_fitness"] > 0
  custom_steps: list[Any] = list(ga.custom_pipeline_steps)
  assert len(custom_steps) == 1
  assert custom_steps[0].count == 3

  # Missing relative file reports a clean preflight warning instead of an unhandled crash.
  algos["RelativeExp"].strategies["mutation"] = {"module": "../does-not-exist.py", "class": "Ghost"}
  missing_warnings = validate_configs(
    {
      "ga_params": ga_params,
      "algorithms": algos,
      "benchmark": BenchmarkConfig(),
      "problems": [ProblemConfig("burma14", 3323.0, 14)],
    }
  )
  assert any("External module file not found" in w for w in missing_warnings)
