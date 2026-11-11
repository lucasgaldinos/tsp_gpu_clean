"""Unit tests for `EvaluationGPUCost` (`DP11-15`, `EVAL-A`).

Verifies 100% cost identity between `EvaluationGPUCost` and `EvaluationCPUCost`
across problem dimensions n in {52, 100, 318, 1002, 2103}, testing both dense distance
matrix and on-the-fly coordinate evaluation modes with coalesced block-stride loops
and intra-warp shuffle reduction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from algorithms.strategies.evaluation.cost_evaluation import EvaluationCPUCost, EvaluationGPUCost
from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext


def _cupy() -> Any:
  """Return CuPy module or skip if unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CUDA device or CuPy unavailable")
  return cp


def _make_euclidean_problem(n: int, seed: int = 42) -> Problem:
  """Create a synthetic 2D Euclidean problem with n cities."""
  rng = np.random.default_rng(seed)
  coords = rng.uniform(0.0, 1000.0, size=(n, 2))
  return Problem(
    name=f"syn_euc_{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=None,
    demands=None,
  )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [52, 100, 318, 1002, 2103])
@pytest.mark.parametrize("mode", ["dense", "coordinate"])
def test_gpu_cost_evaluation_matches_cpu_reference(n: int, mode: str) -> None:
  """WHAT: Evaluate batch tour costs on GPU and compare bit-for-bit with CPU.

  WHY: EVAL-A (DP11-15) rewrote the evaluation kernel with coalesced block-stride
  loops and warp shuffle reductions. Tour costs must remain 100% identical.

  EXPECTED: Exactly identical float64 costs for all tours across all dimensions.
  """
  cp = _cupy()
  problem = _make_euclidean_problem(n, seed=100 + n)

  # Generate 8 random tours
  rng = np.random.default_rng(200 + n)
  base_tour = np.arange(n, dtype=np.int32)
  tours_np = np.stack([rng.permutation(base_tour) for _ in range(8)]).astype(np.int32)

  # In dense mode, ensure matrix is built; in coordinate mode, limit max_matrix_dimension
  max_matrix_dim = 10000 if mode == "dense" else 10

  cpu_ctx = ProblemContext(problem, xp=np, seed=1, max_matrix_dimension=max_matrix_dim)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1, max_matrix_dimension=max_matrix_dim)

  cpu_eval = EvaluationCPUCost()
  gpu_eval = EvaluationGPUCost(max_matrix_dimension=max_matrix_dim)

  cpu_costs = cpu_eval.evaluate_batch(tours_np, cpu_ctx)
  gpu_costs = gpu_eval.evaluate_batch(tours_np, gpu_ctx)

  np.testing.assert_array_equal(
    np.asarray(cpu_costs, dtype=np.float64),
    np.asarray(gpu_costs, dtype=np.float64),
    err_msg=f"Cost mismatch at n={n}, mode={mode}",
  )


@pytest.mark.gpu
def test_gpu_cost_evaluation_1d_route() -> None:
  """WHAT: Evaluate a single 1D route via `evaluate()`."""
  cp = _cupy()
  n = 30
  problem = _make_euclidean_problem(n, seed=99)
  tour = np.arange(n, dtype=np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=1)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)

  cost_cpu = EvaluationCPUCost().evaluate(tour, cpu_ctx)
  cost_gpu = EvaluationGPUCost().evaluate(tour, gpu_ctx)

  assert cost_gpu == cost_cpu


@pytest.mark.gpu
def test_gpu_cost_evaluation_return_gpu() -> None:
  """WHAT: Return GPU-resident costs when input is on GPU and return_gpu=True."""
  cp = _cupy()
  n = 30
  problem = _make_euclidean_problem(n, seed=101)
  tour_gpu = cp.arange(n, dtype=cp.int32).reshape(1, n)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)
  gpu_eval = EvaluationGPUCost(return_gpu=True)

  costs = gpu_eval.evaluate_batch(tour_gpu, gpu_ctx)
  assert isinstance(costs, cp.ndarray)


@pytest.mark.gpu
def test_gpu_cost_evaluation_counts_h2d_for_cpu_input() -> None:
  """WHAT: `input_on_gpu=False` uploads the batch and must report it as H2D traffic.

  WHY: `BUG-30` — `cp.asarray(batch_np)` crossed PCIe without touching
  `h2d_bytes`, so a mixed CPU-population/GPU-evaluator run under-reported transfers.

  EXPECTED: `h2d_bytes` equals the `int32` payload size of the host batch, and stays
  zero-free across two identical calls (one upload per call, no silent caching).
  """
  cp = _cupy()
  n = 40
  problem = _make_euclidean_problem(n, seed=102)
  batch_np = np.tile(np.arange(n, dtype=np.int32), (6, 1))
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)

  gpu_eval = EvaluationGPUCost()
  gpu_eval.evaluate_batch(batch_np, gpu_ctx)
  # First call also uploads the cached distance operands; the *second* call only
  # crosses PCIe with the batch, so the delta isolates the upload under test.
  baseline_bytes = gpu_eval.h2d_bytes
  costs = gpu_eval.evaluate_batch(batch_np, gpu_ctx)

  assert costs.shape == (batch_np.shape[0],)
  assert gpu_eval.h2d_bytes - baseline_bytes == batch_np.nbytes


def test_cost_calculator_kernel_uses_warp_shuffle_and_stride() -> None:
  """WHAT: Falsifiable test confirming cost_calculator.cu uses warp shuffle and stride.

  WHY: Task 11.13 requires eliminating chunk partitioning and 5 shared-memory barriers.
  """
  from pathlib import Path
  kernel_path = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "algorithms"
    / "strategies"
    / "evaluation"
    / "kernels"
    / "cost_calculator.cu"
  )
  source = kernel_path.read_text(encoding="utf-8")
  assert "__shfl_down_sync" in source, "cost_calculator.cu must use __shfl_down_sync reduction"
  assert "for (int i = tid; i < n; i += block_size)" in source, (
    "cost_calculator.cu must use block-stride loop"
  )
  assert "int chunk = " not in source, "cost_calculator.cu must not use chunk partitioning"

