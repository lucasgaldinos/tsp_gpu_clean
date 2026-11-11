"""Unit tests for `CrossoverGPUOrder` (`DP11-11`, `BUG-22`, `OX-A`).

Verifies 100% bit-identity between `CrossoverGPUOrder` and `CrossoverOrder`
across problem dimensions n in {52, 100, 200, 500, 1000, 2000} and closed routes
with fixed depot endpoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.crossover.order_crossover import CrossoverGPUOrder, CrossoverOrder
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext


def _cupy() -> Any:
  """Return CuPy module or skip if unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CUDA device or CuPy unavailable")
  return cp


def _make_dummy_problem(n: int) -> Problem:
  """Create a dummy problem for context initialization."""
  return Problem(
    name=f"dummy_ox_{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.zeros((n, 2), dtype=np.float64),
    distances=None,
    capacity=None,
    demands=None,
  )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [52, 100, 200, 500, 1000, 2000])
def test_gpu_order_crossover_matches_cpu_bit_identical(n: int) -> None:
  """WHAT: Compare GPU and CPU order crossover across problem sizes.

  WHY: DP11-11 and OX-A mandate 100% bit-identity between CrossoverGPUOrder
  and CrossoverOrder on disjoint pairs with identical host-sampled cut points.

  EXPECTED: Exactly identical 2D offspring route arrays for all n.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  pop_size = 64

  rng = np.random.default_rng(300 + n)
  base_tour = np.arange(n, dtype=np.int32)
  population = np.stack([rng.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=77)
  cpu_strategy = CrossoverOrder()
  cpu_offspring = cpu_strategy.crossover_batch(population.copy(), context=cpu_ctx)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=77)
  gpu_strategy = CrossoverGPUOrder()
  gpu_offspring = gpu_strategy.crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(
    np.asarray(gpu_offspring),
    np.asarray(cpu_offspring),
    err_msg=f"Bit divergence in CrossoverGPUOrder at n={n}",
  )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [14, 52, 100])
def test_gpu_order_crossover_preserves_closed_route_depot_endpoints(n: int) -> None:
  """WHAT: Crossover on closed tours [0, ..., 0] with fixed depot endpoints.

  WHY: OX-A requires that depot endpoints stay at index 0 and index n-1,
  and offspring must match CPU bit-for-bit.

  EXPECTED: Endpoints untouched, interior valid permutation, bit-identical to CPU.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  pop_size = 16

  rng = np.random.default_rng(400 + n)
  inner_cities = np.arange(1, n - 1, dtype=np.int32)
  routes = []
  for _ in range(pop_size):
    perm = rng.permutation(inner_cities).astype(np.int32)
    routes.append(np.concatenate([[0], perm, [0]]).astype(np.int32))
  population = np.stack(routes).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=88)
  cpu_strategy = CrossoverOrder()
  cpu_offspring = cpu_strategy.crossover_batch(population.copy(), context=cpu_ctx)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=88)
  gpu_strategy = CrossoverGPUOrder()
  gpu_offspring = gpu_strategy.crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_offspring), np.asarray(cpu_offspring))
  for child in gpu_offspring:
    assert child[0] == 0
    assert child[-1] == 0
    assert sorted(child[1:-1].tolist()) == list(range(1, n - 1))


@pytest.mark.gpu
def test_gpu_order_crossover_preserves_gpu_input_residence() -> None:
  """WHAT: GPU-resident parent arrays remain in VRAM without host transfer."""
  cp = _cupy()
  problem = _make_dummy_problem(30)
  rng = np.random.default_rng(99)
  base = np.arange(30, dtype=np.int32)
  routes_np = np.stack([rng.permutation(base) for _ in range(8)]).astype(np.int32)
  routes_gpu = cp.asarray(routes_np)

  ctx = ProblemContext(problem, xp=cp, seed=99)
  strategy = CrossoverGPUOrder()
  offspring = strategy.crossover_batch(routes_gpu, context=ctx)

  assert isinstance(offspring, cp.ndarray)
  assert offspring.shape == (8, 30)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("pop_size", [1, 3, 5])
def test_gpu_order_crossover_small_n_open_routes(n: int, pop_size: int) -> None:
  """WHAT: Crossover on boundary dimensions n in {2, 3, 4} with odd and single populations.

  WHY: Boundary sizes must not crash or trigger division by zero in CUDA or host cut selection.

  EXPECTED: Bit-identical offspring to CPU reference.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  rng = np.random.default_rng(500 + n + pop_size)
  base_tour = np.arange(n, dtype=np.int32)
  if pop_size == 1:
    population = rng.permutation(base_tour).astype(np.int32)
  else:
    population = np.stack([rng.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=123)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=123)

  cpu_off = CrossoverOrder().crossover_batch(population.copy(), context=cpu_ctx)
  gpu_off = CrossoverGPUOrder().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_off), np.asarray(cpu_off))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("pop_size", [2, 3])
def test_gpu_order_crossover_small_n_closed_routes(n: int, pop_size: int) -> None:
  """WHAT: Crossover on small closed tours [0, ..., 0] including degenerate n=3.

  WHY: Closed tours with n=3 have only 1 interior city so cuts cannot be sampled without replacement.
  Both CPU and GPU must safely return the parents unchanged without raising ValueError.

  EXPECTED: Bit-identical offspring to CPU reference with depot ends preserved.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  rng = np.random.default_rng(600 + n)
  inner_cities = np.arange(1, n - 1, dtype=np.int32)
  routes = []
  for _ in range(pop_size):
    if len(inner_cities) > 0:
      perm = rng.permutation(inner_cities).astype(np.int32)
      routes.append(np.concatenate([[0], perm, [0]]).astype(np.int32))
    else:
      routes.append(np.array([0, 0], dtype=np.int32))
  population = np.stack(routes).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=124)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=124)

  cpu_off = CrossoverOrder().crossover_batch(population.copy(), context=cpu_ctx)
  gpu_off = CrossoverGPUOrder().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_off), np.asarray(cpu_off))


@pytest.mark.gpu
def test_gpu_order_crossover_recovers_from_zero_filled_warmup() -> None:
  """WHAT: A zero-filled batch followed by an open permutation batch of equal `n`.

  WHY: the removed `_is_closed_cache` keyed only on `n_cities`, so a warm-up batch
  of zeros (`0 == 0`) marked the dimension closed and made every later open-tour
  offspring diverge from the CPU reference (`BUG-24`).

  EXPECTED: the second call is bit-identical to `CrossoverOrder` under the same seed.
  """
  cp = _cupy()
  n = 52
  pop_size = 32
  problem = _make_dummy_problem(n)

  gpu_strategy = CrossoverGPUOrder(seed=5)
  poison_ctx = ProblemContext(problem, xp=cp, seed=1)
  gpu_strategy.crossover_batch(cp.zeros((pop_size, n), dtype=cp.int32), context=poison_ctx)

  base = np.arange(n, dtype=np.int32)
  population = np.stack(
    [np.random.default_rng(700 + row).permutation(base) for row in range(pop_size)]
  ).astype(np.int32)
  cpu_offspring = CrossoverOrder(seed=5).crossover_batch(
    population.copy(), context=ProblemContext(problem, xp=np, seed=77)
  )
  gpu_offspring = gpu_strategy.crossover_batch(
    population.copy(), context=ProblemContext(problem, xp=cp, seed=77)
  )

  np.testing.assert_array_equal(np.asarray(gpu_offspring), np.asarray(cpu_offspring))


@pytest.mark.gpu
def test_gpu_order_crossover_1d_pair() -> None:
  """WHAT: Crossover between two 1D parent vectors.

  EXPECTED: Bit-identical 1D child to CPU reference.
  """
  cp = _cupy()
  n = 20
  problem = _make_dummy_problem(n)
  p1 = np.arange(n, dtype=np.int32)
  p2 = np.random.default_rng(1).permutation(p1)

  cpu_ctx = ProblemContext(problem, xp=np, seed=42)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=42)

  c_cpu = CrossoverOrder().crossover(p1, p2, context=cpu_ctx)
  c_gpu = CrossoverGPUOrder().crossover(p1, p2, context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(c_gpu), np.asarray(c_cpu))
  assert c_gpu.ndim == 1
