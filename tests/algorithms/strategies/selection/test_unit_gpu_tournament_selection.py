"""Unit tests for `SelectionGPUTournament` (`DP11-16`, `SEL-A`).

Verifies 100% bit-identity between `SelectionGPUTournament` and `SelectionCPUTournament`
when using the same seed, confirming memory coalescing via 1D winner index emission
and CuPy 2D gather without unconditional stream synchronization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.selection.tournament_selection import SelectionCPUTournament, SelectionGPUTournament
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
    name="dummy_sel",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.zeros((n, 2), dtype=np.float64),
    distances=None,
    capacity=None,
    demands=None,
  )


@pytest.mark.gpu
@pytest.mark.parametrize("k", [2, 3, 5, 7])
@pytest.mark.parametrize("pop_size", [16, 64, 128])
@pytest.mark.parametrize("n_cities", [20, 52])
def test_gpu_tournament_selection_matches_cpu_bit_identical(
  k: int,
  pop_size: int,
  n_cities: int,
) -> None:
  """WHAT: Selection from a population with distinct fitnesses.

  WHY: SEL-A requires that SelectionGPUTournament outputs bit-identical routes
  to SelectionCPUTournament for identical RNG streams on host.

  EXPECTED: Exactly identical 2D parent routes.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n_cities)

  rng_data = np.random.default_rng(100 + k)
  base_tour = np.arange(n_cities, dtype=np.int32)
  population_np = np.stack([rng_data.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  # Unique non-colliding fitness values
  fitness_np = rng_data.uniform(10.0, 5000.0, size=pop_size).astype(np.float64)

  # Run CPU selection
  cpu_ctx = ProblemContext(problem, xp=np, seed=42)
  cpu_strategy = SelectionCPUTournament(k=k)
  cpu_parents = cpu_strategy.select_batch(population_np, fitness_np, num_parents=pop_size, context=cpu_ctx)

  # Run GPU selection with identical seed in context
  gpu_ctx = ProblemContext(problem, xp=cp, seed=42)
  gpu_strategy = SelectionGPUTournament(k=k)
  gpu_parents = gpu_strategy.select_batch(population_np, fitness_np, num_parents=pop_size, context=gpu_ctx)

  np.testing.assert_array_equal(
    np.asarray(gpu_parents),
    np.asarray(cpu_parents),
    err_msg=f"Mismatch for k={k}, pop_size={pop_size}, n_cities={n_cities}",
  )


@pytest.mark.gpu
def test_gpu_tournament_selection_gpu_input_remains_on_gpu() -> None:
  """WHAT: When population resides on GPU, parents remain on GPU."""
  cp = _cupy()
  problem = _make_dummy_problem(30)
  pop_gpu = cp.arange(10 * 30, dtype=cp.int32).reshape(10, 30)
  fitness_gpu = cp.linspace(100.0, 10.0, 10, dtype=cp.float64)

  ctx = ProblemContext(problem, xp=cp, seed=99)
  strategy = SelectionGPUTournament(k=3)
  selected = strategy.select_batch(pop_gpu, fitness_gpu, num_parents=6, context=ctx)

  assert isinstance(selected, cp.ndarray)
  assert selected.shape == (6, 30)


@pytest.mark.gpu
@pytest.mark.parametrize("num_parents", [1, 3, 7])
def test_gpu_tournament_selection_odd_num_parents(num_parents: int) -> None:
  """WHAT: Selection with odd numbers of parents (1, 3, 7)."""
  cp = _cupy()
  problem = _make_dummy_problem(20)
  pop_size = 10
  rng = np.random.default_rng(77)
  base = np.arange(20, dtype=np.int32)
  pop = np.stack([rng.permutation(base) for _ in range(pop_size)]).astype(np.int32)
  fit = rng.random(pop_size)

  cpu_ctx = ProblemContext(problem, xp=np, seed=77)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=77)

  cpu_sel = SelectionCPUTournament(k=3).select_batch(pop, fit, num_parents=num_parents, context=cpu_ctx)
  gpu_sel = SelectionGPUTournament(k=3).select_batch(pop, fit, num_parents=num_parents, context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_sel), np.asarray(cpu_sel))
  assert gpu_sel.shape == (num_parents, 20)


@pytest.mark.gpu
def test_gpu_tournament_selection_single_parent() -> None:
  """WHAT: Select a single 1D parent route via `select()`."""
  cp = _cupy()
  problem = _make_dummy_problem(20)
  pop_size = 10
  rng = np.random.default_rng(88)
  base = np.arange(20, dtype=np.int32)
  pop = np.stack([rng.permutation(base) for _ in range(pop_size)]).astype(np.int32)
  fit = rng.random(pop_size)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=88)
  gpu_p = SelectionGPUTournament(k=3).select(pop, fit, context=gpu_ctx)

  assert gpu_p.ndim == 1
  assert gpu_p.shape == (20,)
  assert any(np.array_equal(gpu_p, member) for member in pop)


def test_gpu_tournament_kernel_emits_1d_winner_indices() -> None:
  """WHAT: Falsifiable test confirming tournament_selection.cu emits 1D winner_indices.

  WHY: Task 11.14 (DP11-16, SEL-A) eliminated the 2D non-coalesced route gather in the kernel.
  """
  from pathlib import Path
  kernel_path = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "algorithms"
    / "strategies"
    / "selection"
    / "kernels"
    / "tournament_selection.cu"
  )
  import re

  source = kernel_path.read_text(encoding="utf-8")
  assert re.search(r"int\s*\*\s*__restrict__\s+winner_indices", source), (
    "tournament_selection.cu must emit 1D winner_indices"
  )
  assert "parents_out" not in source, "tournament_selection.cu must delegate 2D gather to CuPy"

