"""Unit tests for CPU/GPU Random and Nearest-Neighbor initializers (`Task 17.31`)."""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.strategies.initialization import (
  InitializationCPUPermRandom,
  InitializationCPUTourNearestNeighbor,
  InitializationGPUPermRandom,
  InitializationGPUTourNearestNeighbor,
)
from protocols.backend import gpu_available, require_gpu
from protocols.problem_context import ProblemContext
from utils.gpu_helpers import _compile_kernel_cached

if TYPE_CHECKING:
  from data_models.problem import Problem


def _closed_cost(route: np.ndarray, distances: np.ndarray) -> int:
  """Return the closed-loop tour cost under a dense distance matrix."""
  return int(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


@pytest.mark.parametrize(
  ("customers", "population_size"),
  [
    pytest.param([0, 1, 2, 3, 4, 5], 1, id="single-route"),
    pytest.param([7, 3, 9, 1, 4], 8, id="non-contiguous-customers"),
  ],
)
def test_random_permutation_initialization_returns_valid_population(
  customers: list[int],
  population_size: int,
) -> None:
  """WHAT: `InitializationCPUPermRandom.initialize` output and signature contract."""
  assert "return_gpu" not in inspect.signature(InitializationCPUPermRandom.__init__).parameters
  assert inspect.signature(InitializationGPUPermRandom.__init__).parameters["return_gpu"].default is True

  strategy = InitializationCPUPermRandom(seed=17)
  population = strategy.initialize(customers, population_size)
  assert isinstance(population, np.ndarray)
  assert population.shape == (population_size, len(customers))
  assert population.dtype == np.int32
  expected_genes: list[int] = sorted(customers)
  for route in population:
    assert sorted(route.tolist()) == expected_genes
    assert len(set(route.tolist())) == len(customers)


def test_random_permutation_initialization_respects_seed() -> None:
  """WHAT: Reproducibility of `InitializationCPUPermRandom` under an explicit seed."""
  customers: list[int] = list(range(6))
  first = InitializationCPUPermRandom(seed=123).initialize(customers, population_size=4)
  second = InitializationCPUPermRandom(seed=123).initialize(customers, population_size=4)
  np.testing.assert_array_equal(first, second)


def test_candidate_k_builds_greedy_nearest_neighbour_tours(berlin52_problem: Problem) -> None:
  """WHAT: `InitializationCPUTourNearestNeighbor(k=10)` beats uniform random and removes `candidate_k` alias."""
  assert "return_gpu" not in inspect.signature(InitializationCPUTourNearestNeighbor.__init__).parameters
  assert "candidate_k" not in inspect.signature(InitializationCPUTourNearestNeighbor.__init__).parameters
  assert inspect.signature(InitializationGPUTourNearestNeighbor.__init__).parameters["return_gpu"].default is True

  with pytest.raises(TypeError):
    InitializationCPUTourNearestNeighbor(k=3, candidate_k=10)  # type: ignore[call-arg]
  with pytest.raises(ValueError, match="Candidate-list width"):
    InitializationCPUTourNearestNeighbor(k=0)

  context = ProblemContext(berlin52_problem, xp=np, seed=3)
  distances = context.get_cpu_distances()
  customers = list(range(berlin52_problem.dimension))

  guided = InitializationCPUTourNearestNeighbor(k=10, seed=7).initialize(customers, 4, context)
  uniform = InitializationCPUPermRandom(seed=7).initialize(customers, 4, context)

  expected_genes = sorted(customers)
  for route in guided:
    assert sorted(route.tolist()) == expected_genes
  best_guided = min(_closed_cost(route, distances) for route in guided)
  best_uniform = min(_closed_cost(route, distances) for route in uniform)
  assert best_guided < best_uniform


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["random", "nn"])
def test_random_and_nn_cpu_gpu_bit_identity_and_warmup_purity(
  berlin52_problem: Problem,
  kind: str,
) -> None:
  """WHAT: Exact CPU==GPU bit-identity (`SplitMix64`), VRAM output, `h2d_bytes`, and `warmup()` purity."""
  if not gpu_available():
    pytest.skip("CUDA/CuPy unavailable")
  cp = require_gpu()

  customers = list(range(berlin52_problem.dimension))
  cpu_cls: Any = InitializationCPUPermRandom if kind == "random" else InitializationCPUTourNearestNeighbor
  gpu_cls: Any = InitializationGPUPermRandom if kind == "random" else InitializationGPUTourNearestNeighbor
  kwargs: dict[str, Any] = {"seed": 77} if kind == "random" else {"k": 15, "seed": 77}

  ctx_cpu = ProblemContext(berlin52_problem, xp=np, seed=314)
  ctx_gpu = ProblemContext(berlin52_problem, xp=cp, seed=314)
  ctx_warm = ProblemContext(berlin52_problem, xp=cp, seed=99)

  gpu_init = gpu_cls(**kwargs)
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

  cpu_init = cpu_cls(**kwargs)
  cpu_pop = cpu_init.initialize(customers, 16, ctx_cpu)
  gpu_pop = gpu_init.initialize(customers, 16, ctx_gpu)

  assert isinstance(gpu_pop, cp.ndarray)
  assert gpu_pop.dtype == cp.int32
  assert gpu_init.d2h_bytes == 0
  assert gpu_init.h2d_bytes > 0
  assert _compile_kernel_cached.cache_info().currsize == cache_size
  np.testing.assert_array_equal(cpu_pop, cp.asnumpy(gpu_pop))
