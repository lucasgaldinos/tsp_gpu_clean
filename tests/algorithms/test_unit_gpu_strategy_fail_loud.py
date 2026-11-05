"""Fail-loud contract for every GPU strategy: no silent CPU fallback (`DP9-2`).

A GPU strategy that cannot reach a device must raise `GPUUnavailableError` instead of
delegating to its CPU twin. The old fallback was invisible to the caller: the benchmark
recorded the run under the GPU variant's name, so a host without CUDA produced "GPU"
timings that were really NumPy timings, and the CPU-versus-GPU comparison the thesis
depends on degenerated into NumPy against NumPy.

Both tests run on the CPU: they never touch a device, they only remove one. The first
test removes the device at the shared capability probe (end-to-end through the real
`require_gpu`), the second removes it at each strategy module's imported binding, which
guards the wiring itself -- a module that stopped calling `require_gpu` fails the
`setattr` instead of silently keeping its fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest

import protocols.backend as backend
from algorithms.strategies import (
  CrossoverGPUOrder,
  EvaluationGPUCost,
  InitializationRandom,
  LocalSearchGPUTwoOpt,
  MutationGPUInversion,
  MutationGPUSwap,
  SelectionGPUTournament,
)
from data_models.exceptions import GPUUnavailableError
from protocols import ProblemContext

if TYPE_CHECKING:
  from data_models.problem import Problem

#: Strategy modules that must resolve the device through `protocols.backend.require_gpu`.
GPU_STRATEGY_MODULES: tuple[str, ...] = (
  "algorithms.strategies.crossover.order_crossover",
  "algorithms.strategies.evaluation.cost_evaluation",
  "algorithms.strategies.initialization.random_initialization",
  "algorithms.strategies.local_search.two_opt",
  "algorithms.strategies.mutation.inversion_mutation",
  "algorithms.strategies.mutation.swap_mutation",
  "algorithms.strategies.selection.tournament_selection",
)

#: One label per GPU entry point that used to fall back to CPU.
GPU_ENTRY_POINTS: tuple[str, ...] = (
  "InitializationRandom.return_gpu",
  "LocalSearchGPUTwoOpt.improve_batch",
  "SelectionGPUTournament.select_batch",
  "CrossoverGPUOrder.crossover_batch",
  "MutationGPUSwap.mutate_batch",
  "MutationGPUInversion.mutate_batch",
  "EvaluationGPUCost.evaluate_batch",
)

_NO_CUDA_MESSAGE = "CUDA device unavailable"


def _raise_gpu_unavailable() -> NoReturn:
  """Stand-in for `require_gpu` that fails exactly like a host without a device."""
  raise GPUUnavailableError(_NO_CUDA_MESSAGE)


@pytest.fixture()
def cpu_only_probe(monkeypatch: pytest.MonkeyPatch) -> None:
  """Make the shared capability probe report "no device" for the duration of a test."""
  monkeypatch.setattr(backend._probe_gpu, "_cached", (None, GPUUnavailableError(_NO_CUDA_MESSAGE)))
  assert backend.gpu_available() is False, "the probe override must be visible to the public API"


def _call_entry_point(name: str, context: ProblemContext, routes: np.ndarray, fitness: np.ndarray) -> None:
  """Dispatch one GPU entry point by label, keeping the parametrization readable."""
  match name:
    case "InitializationRandom.return_gpu":
      customers: list[int] = list(range(context.problem.dimension))
      InitializationRandom(seed=42, return_gpu=True).initialize(
        customers,
        population_size=2,
        context=context,
      )
    case "LocalSearchGPUTwoOpt.improve_batch":
      LocalSearchGPUTwoOpt(max_iterations=2, mode="batch").improve_batch(routes, context)
    case "SelectionGPUTournament.select_batch":
      SelectionGPUTournament().select_batch(routes, fitness, num_parents=2, context=context)
    case "CrossoverGPUOrder.crossover_batch":
      CrossoverGPUOrder(seed=42).crossover_batch(routes, context=context)
    case "MutationGPUSwap.mutate_batch":
      MutationGPUSwap(seed=42).mutate_batch(routes, mutation_rate=0.5, context=context)
    case "MutationGPUInversion.mutate_batch":
      MutationGPUInversion(seed=42).mutate_batch(routes, mutation_rate=0.5, context=context)
    case "EvaluationGPUCost.evaluate_batch":
      EvaluationGPUCost().evaluate_batch(routes, context)
    case _:
      raise AssertionError(f"unknown GPU entry point {name!r}")


def _host_batch(problem: Problem) -> tuple[ProblemContext, np.ndarray, np.ndarray]:
  """Return a CPU context plus a NumPy route batch and fitness vector for `problem`."""
  context = ProblemContext(problem, xp=np, seed=42)
  routes = np.tile(np.arange(problem.dimension, dtype=np.int32), (4, 1))
  fitness = np.arange(routes.shape[0], dtype=np.float64)
  return context, routes, fitness


@pytest.mark.parametrize("entry_point", GPU_ENTRY_POINTS)
def test_gpu_strategy_raises_when_the_probe_reports_no_device(
  entry_point: str,
  cpu_only_probe: None,
  burma14_problem: Problem,
) -> None:
  """WHAT: every GPU entry point with the shared probe reporting "no device".

  WHY: the previous behaviour delegated to the CPU twin, so a benchmark row labelled
  `*GPU*` could be produced entirely on the host.

  EXPECTED: `GPUUnavailableError` naming the missing device, never a NumPy result.
  """
  context, routes, fitness = _host_batch(burma14_problem)
  with pytest.raises(GPUUnavailableError, match=_NO_CUDA_MESSAGE):
    _call_entry_point(entry_point, context, routes, fitness)


@pytest.mark.parametrize("entry_point", GPU_ENTRY_POINTS)
def test_gpu_strategy_module_binds_require_gpu(
  entry_point: str,
  monkeypatch: pytest.MonkeyPatch,
  burma14_problem: Problem,
) -> None:
  """WHAT: every GPU module's own `require_gpu` binding replaced by a raising stub.

  WHY: the contract is only real if each module actually calls the backend; a module
  that dropped the import would fail here instead of quietly restoring its fallback.

  EXPECTED: `setattr` resolves in all seven modules and each entry point raises.
  """
  for module_path in GPU_STRATEGY_MODULES:
    monkeypatch.setattr(f"{module_path}.require_gpu", _raise_gpu_unavailable)

  context, routes, fitness = _host_batch(burma14_problem)
  with pytest.raises(GPUUnavailableError, match=_NO_CUDA_MESSAGE):
    _call_entry_point(entry_point, context, routes, fitness)
