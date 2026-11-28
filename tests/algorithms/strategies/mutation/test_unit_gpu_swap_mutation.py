"""Unit tests for `MutationGPUPermSwap` (`DP11-14`, `SWAP-A`).

Verifies 100% bit-identity between `MutationGPUPermSwap` and `MutationCPUPermSwap`
when using the same seed and problem context, confirming removal of
unconditional stream synchronization and unified host RNG index sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.mutation.swap_mutation import MutationCPUPermSwap, MutationGPUPermSwap
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext


def _cupy() -> Any:
  """Return CuPy module or skip if unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CUDA device or CuPy unavailable")
  return cp


def _make_dummy_problem(n: int = 52) -> Problem:
  """Create a minimal problem for context initialization."""
  return Problem(
    name="dummy_swap",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.zeros((n, 2), dtype=np.float64),
    distances=None,
    capacity=None,
    demands=None,
  )


@pytest.mark.gpu
@pytest.mark.parametrize("mutation_rate", [0.0, 0.2, 0.8, 1.0])
@pytest.mark.parametrize("pop_size", [1, 16, 64])
@pytest.mark.parametrize("n_cities", [5, 20, 52])
def test_gpu_swap_mutation_matches_cpu_bit_identical(
  mutation_rate: float,
  pop_size: int,
  n_cities: int,
) -> None:
  """WHAT: Compare GPU and CPU swap mutation on a batch of routes.

  WHY: SWAP-A mandates iso-algorithmic bit-identity between MutationGPUPermSwap
  and MutationCPUPermSwap for identical seeds.

  EXPECTED: Exactly identical 2D mutated route arrays.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n_cities)

  rng_data = np.random.default_rng(500 + n_cities)
  base_tour = np.arange(n_cities, dtype=np.int32)
  if pop_size == 1:
    population = rng_data.permutation(base_tour).astype(np.int32)
  else:
    population = np.stack([rng_data.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=42)
  cpu_strategy = MutationCPUPermSwap()
  cpu_mutated = cpu_strategy.mutate_batch(population.copy(), mutation_rate=mutation_rate, context=cpu_ctx)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=42)
  gpu_strategy = MutationGPUPermSwap()
  gpu_mutated = gpu_strategy.mutate_batch(population.copy(), mutation_rate=mutation_rate, context=gpu_ctx)

  np.testing.assert_array_equal(
    np.asarray(gpu_mutated),
    np.asarray(cpu_mutated),
    err_msg=f"Mismatch for rate={mutation_rate}, pop={pop_size}, n={n_cities}",
  )


@pytest.mark.gpu
def test_gpu_swap_mutation_preserves_gpu_input_residence() -> None:
  """WHAT: GPU-resident arrays stay in VRAM after mutation."""
  cp = _cupy()
  problem = _make_dummy_problem(20)
  routes_gpu = cp.arange(4 * 20, dtype=cp.int32).reshape(4, 20)

  ctx = ProblemContext(problem, xp=cp, seed=123)
  strategy = MutationGPUPermSwap()
  result = strategy.mutate_batch(routes_gpu, mutation_rate=0.5, context=ctx)

  assert isinstance(result, cp.ndarray)
  assert result.shape == (4, 20)


@pytest.mark.gpu
def test_gpu_swap_mutation_single_1d_route() -> None:
  """WHAT: Mutate a single 1D route via `mutate()`."""
  cp = _cupy()
  n = 20
  problem = _make_dummy_problem(n)
  tour = np.arange(n, dtype=np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=42)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=42)

  cpu_res = MutationCPUPermSwap().mutate(tour.copy(), 1.0, context=cpu_ctx)
  gpu_res = MutationGPUPermSwap().mutate(tour.copy(), 1.0, context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_res), np.asarray(cpu_res))
  assert gpu_res.ndim == 1


@pytest.mark.gpu
def test_gpu_swap_mutation_closed_depot_routes_match_cpu() -> None:
  """WHAT: Mutate depot-bounded `[0, ..., 0]` batches on both backends.

  WHY: `resolve_route_bounds` must preserve the depots of closed routes while an
  open TSP permutation is free to move positions `0` and `n - 1` (`BUG-28`).

  EXPECTED: bit-identical offspring, both endpoints still at the depot.
  """
  cp = _cupy()
  n = 14
  problem = _make_dummy_problem(n)
  inner = np.arange(1, n - 1, dtype=np.int32)
  population = np.stack(
    [np.concatenate(([0], np.random.default_rng(800 + row).permutation(inner), [0])) for row in range(8)]
  ).astype(np.int32)

  cpu_mutated = MutationCPUPermSwap(seed=3).mutate_batch(
    population.copy(), mutation_rate=1.0, context=ProblemContext(problem, xp=np, seed=42)
  )
  gpu_mutated = MutationGPUPermSwap(seed=3).mutate_batch(
    population.copy(), mutation_rate=1.0, context=ProblemContext(problem, xp=cp, seed=42)
  )

  np.testing.assert_array_equal(np.asarray(gpu_mutated), np.asarray(cpu_mutated))
  assert np.all(np.asarray(gpu_mutated)[:, 0] == 0)
  assert np.all(np.asarray(gpu_mutated)[:, -1] == 0)


@pytest.mark.gpu
def test_gpu_swap_mutation_candidate_k_matches_cpu(berlin52_problem: Problem) -> None:
  """WHAT: `candidate_k` on both swap strategies over the same seed.

  WHY: `MutationGPUPermSwap` lacked the parameter, so a GPU run could not honour the
  configured neighbourhood and violated LSP (`BUG-28`).

  EXPECTED: identical offspring and a valid permutation per row.
  """
  cp = _cupy()
  base = np.arange(berlin52_problem.dimension, dtype=np.int32)
  population = np.stack(
    [np.random.default_rng(900 + row).permutation(base) for row in range(8)]
  ).astype(np.int32)

  cpu_mutated = MutationCPUPermSwap(seed=6, candidate_k=10).mutate_batch(
    population.copy(), mutation_rate=1.0, context=ProblemContext(berlin52_problem, xp=np, seed=13)
  )
  gpu_mutated = MutationGPUPermSwap(seed=6, candidate_k=10).mutate_batch(
    population.copy(), mutation_rate=1.0, context=ProblemContext(berlin52_problem, xp=cp, seed=13)
  )

  np.testing.assert_array_equal(np.asarray(gpu_mutated), np.asarray(cpu_mutated))
  for row in np.asarray(gpu_mutated):
    assert sorted(row.tolist()) == list(range(berlin52_problem.dimension))
