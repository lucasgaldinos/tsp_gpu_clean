# ruff: noqa: S101
"""Unit and GPU parity tests for `LocalSearchGPUThreeOpt` and the 3-opt CUDA kernel family.

Verifies all three kernel modes (`batch`, `multi_move`, `naive`) sharing
`kernels/three_opt_common.cuh`:
  1. Exact cost-delta parity across all 7 non-identity 3-opt reconnections (open & closed).
  2. Strict pure 3-opt improvement on a 2-opt-optimal tour (where 2-opt stalls).
  3. Exact single-pass `==` cost-delta parity of `batch` and `naive` vs. CPU 3-opt oracle,
     and `<=` disjoint multi-move parity for `multi_move`, on open TSP & closed CVRP tours.
  4. Permutation integrity & monotonically non-increasing costs across iterations.
  5. Dense-matrix vs. on-the-fly coordinate parity and shared-memory vs. global-memory parity.
  6. `LocalSearchStrategy` protocol conformance, `warmup(context)` idempotence, telemetry
     counters (`h2d_bytes`, `d2h_bytes`, `kernel_launches`), and config-relative loading via
     `load_algorithms_config` + `validate_configs`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest

from algorithms.strategies import LocalSearchGPUTwoOpt
from benchmarking.runner.config_loader import load_algorithms_config, load_all_configs, validate_configs
from data_models.problem import Problem
from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext
from utils.route_helpers import is_gpu_array

type GPUMode = Literal["batch", "multi_move", "naive"]

EXP_DIR: Path = Path(__file__).resolve().parent
CONFIG_DIR: Path = EXP_DIR / "configs"
ALL_GPU_MODES: tuple[GPUMode, ...] = ("batch", "multi_move", "naive")
GRID_GPU_MODES: tuple[GPUMode, ...] = ("batch", "multi_move")


def _load_local_module() -> Any:
  """Import `gpu_three_opt.py` directly from this experiment directory."""
  module_name = "_exp_gpu_three_opt_under_test"
  if module_name in sys.modules:
    return sys.modules[module_name]
  spec = importlib.util.spec_from_file_location(module_name, EXP_DIR / "gpu_three_opt.py")
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = mod
  spec.loader.exec_module(mod)
  return mod


gpu_three_opt = _load_local_module()
LocalSearchGPUThreeOpt = gpu_three_opt.LocalSearchGPUThreeOpt
apply_three_opt_case_cpu = gpu_three_opt.apply_three_opt_case_cpu
cpu_three_opt_best_move = gpu_three_opt.cpu_three_opt_best_move
cpu_three_opt_refine = gpu_three_opt.cpu_three_opt_refine
three_opt_delta_cpu = gpu_three_opt.three_opt_delta_cpu
tour_cost_cpu = gpu_three_opt.tour_cost_cpu


def _cupy() -> Any:
  """Return CuPy module or skip if unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


def _make_symmetric_explicit_problem(
  name: str,
  distances: np.ndarray,
  *,
  problem_type: str = "TSP",
) -> Problem:
  """Build an `EXPLICIT` Problem from a symmetric integer matrix."""
  n = int(distances.shape[0])
  return Problem(
    name=name,
    dimension=n,
    problem_type=problem_type,
    edge_type="EXPLICIT",
    coordinates=None,
    distances=distances.astype(np.int32, copy=False),
    capacity=100 if problem_type == "CVRP" else None,
    demands=np.array([0, *([1] * (n - 1))], dtype=np.int32) if problem_type == "CVRP" else None,
  )


def _make_two_opt_optimal_three_opt_improvable_problem() -> tuple[Problem, np.ndarray]:
  """Construct a 6-city symmetric TSP instance strictly 2-opt optimal yet 3-opt improvable.

  For `tour = [0, 1, 2, 3, 4, 5]` (cost = 33):
    - Kept edges `(1, 2), (3, 4), (5, 0)` have weight 1.
    - Broken edges `(0, 1), (2, 3), (4, 5)` have weight 10.
    - Pure 3-opt Case 5 edges `(0, 3), (1, 4), (2, 5)` have weight 6.
    - All remaining chords have weight 100.
  Every 2-opt move has `Delta >= +1` (strictly 2-opt optimal), while Case 5 (`S2 S1`)
  reorders the tour to `[0, 3, 4, 1, 2, 5]` with exact `Delta = -12` (cost 33 -> 21).
  """
  d = np.full((6, 6), 100, dtype=np.int32)
  np.fill_diagonal(d, 0)
  for u, v, w in (
    (1, 2, 1),
    (3, 4, 1),
    (5, 0, 1),
    (0, 1, 10),
    (2, 3, 10),
    (4, 5, 10),
    (0, 3, 6),
    (1, 4, 6),
    (2, 5, 6),
  ):
    d[u, v] = w
    d[v, u] = w
  problem = _make_symmetric_explicit_problem("pure3opt_trap6", d, problem_type="TSP")
  tour = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
  return problem, tour


def _make_euclidean_problem(n: int, seed: int = 11) -> Problem:
  """Build a coordinate-only `EUC_2D` TSP problem."""
  rng = np.random.default_rng(seed)
  coords = rng.integers(0, 5_000, size=(n, 2)).astype(np.float64)
  return Problem(
    name=f"euc_{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=coords,
    distances=None,
    capacity=None,
    demands=None,
  )


def _random_open_tours(n: int, count: int, seed: int) -> np.ndarray:
  """Generate `count` deterministic open permutations of `0..n-1`."""
  rng = np.random.default_rng(seed)
  base = np.arange(n, dtype=np.int32)
  return np.stack([rng.permutation(base) for _ in range(count)]).astype(np.int32, copy=False)


def _random_closed_tours(n_customers: int, count: int, seed: int) -> np.ndarray:
  """Generate `count` deterministic depot-bounded closed routes `[0, ..., 0]`."""
  rng = np.random.default_rng(seed)
  interior = np.arange(1, n_customers + 1, dtype=np.int32)
  rows = [np.concatenate(([0], rng.permutation(interior), [0])) for _ in range(count)]
  return np.stack(rows).astype(np.int32, copy=False)


# ==============================================================================
# CPU REFERENCE & PROTOCOL UNIT TESTS
# ==============================================================================


@pytest.mark.parametrize("is_closed", [False, True])
def test_cpu_three_opt_all_seven_cases_exact_delta_and_permutation(is_closed: bool) -> None:
  """Every reconnection case 1..7 matches exact tour cost difference and preserves permutation."""
  rng = np.random.default_rng(123 if not is_closed else 456)
  n_nodes = 10
  raw = rng.integers(1, 500, size=(n_nodes, n_nodes), dtype=np.int32)
  distances = ((raw + raw.T) // 2).astype(np.int32)
  np.fill_diagonal(distances, 0)

  if is_closed:
    tour = np.concatenate(([0], rng.permutation(np.arange(1, n_nodes, dtype=np.int32)), [0])).astype(np.int32)
    k_max = len(tour) - 2
  else:
    tour = rng.permutation(np.arange(n_nodes, dtype=np.int32)).astype(np.int32)
    k_max = len(tour) - 1

  base_cost = tour_cost_cpu(tour, distances, is_closed=is_closed)
  checked = 0
  for i in range(k_max - 1):
    for j in range(i + 1, k_max):
      for k in range(j + 1, k_max + 1):
        for case_id in range(1, 8):
          expected_delta = three_opt_delta_cpu(tour, distances, i, j, k, case_id)
          new_tour = apply_three_opt_case_cpu(tour, i, j, k, case_id)
          actual_delta = tour_cost_cpu(new_tour, distances, is_closed=is_closed) - base_cost
          assert actual_delta == expected_delta, f"Mismatch for case={case_id}, cuts=({i},{j},{k})"
          if is_closed:
            assert int(new_tour[0]) == 0 and int(new_tour[-1]) == 0
            assert sorted(int(c) for c in new_tour[1:-1]) == list(range(1, n_nodes))
          else:
            assert sorted(int(c) for c in new_tour) == list(range(n_nodes))
          checked += 1
  assert checked > 500


@pytest.mark.parametrize("mode", ALL_GPU_MODES)
def test_protocol_conformance_and_parameter_validation(mode: GPUMode) -> None:
  """`LocalSearchGPUThreeOpt` conforms to `LocalSearchStrategy` across modes and rejects bad args."""
  strategy = LocalSearchGPUThreeOpt(max_iterations=5, mode=mode, candidate_k=20)
  assert isinstance(strategy, LocalSearchStrategy)
  assert cast("Any", strategy).mode == mode

  with pytest.raises(ValueError, match="Invalid 3-opt GPU mode"):
    LocalSearchGPUThreeOpt(mode="steepest")
  with pytest.raises(ValueError, match="max_iterations"):
    LocalSearchGPUThreeOpt(max_iterations=-1, mode=mode)
  with pytest.raises(ValueError, match="threads_per_block"):
    LocalSearchGPUThreeOpt(threads_per_block=0, mode=mode)
  with pytest.raises(ValueError, match="max_shared_tour_cities"):
    LocalSearchGPUThreeOpt(max_shared_tour_cities=-1, mode=mode)
  with pytest.raises(ValueError, match="inner_candidate_k"):
    LocalSearchGPUThreeOpt(inner_candidate_k=-2, mode=mode)


def test_config_relative_loading_and_validation() -> None:
  """`load_algorithms_config` + `validate_configs` load all 4 iso-algorithmic 2-opt/3-opt variants."""
  ga_params, algorithms = load_algorithms_config(CONFIG_DIR)
  expected_names = ("TwoOpt_Batch_GPU", "TwoOpt_MultiMove_GPU", "ThreeOpt_Batch_GPU", "ThreeOpt_MultiMove_GPU")
  for name in expected_names:
    assert name in algorithms
  assert ga_params.population_size == 128

  all_configs = load_all_configs(CONFIG_DIR)
  warnings = validate_configs(all_configs)
  assert warnings == [], f"Unexpected config validation warnings: {warnings}"

  for expected_mode, two_name, three_name in (
    ("batch", "TwoOpt_Batch_GPU", "ThreeOpt_Batch_GPU"),
    ("multi_move", "TwoOpt_MultiMove_GPU", "ThreeOpt_MultiMove_GPU"),
  ):
    two_ls = algorithms[two_name].instantiate(ga_params).local_search
    three_ls = algorithms[three_name].instantiate(ga_params).local_search
    assert isinstance(two_ls, LocalSearchGPUTwoOpt)
    assert two_ls.mode == expected_mode
    assert isinstance(three_ls, LocalSearchStrategy)
    three_any = cast("Any", three_ls)
    assert type(three_any).__name__ == "LocalSearchGPUThreeOpt"
    assert three_any.mode == expected_mode
    assert two_ls.max_iterations == three_any.max_iterations == 10
    assert two_ls.candidate_k == three_any.candidate_k == 20


# ==============================================================================
# GPU EXECUTION, PARITY, AND INTEGRITY TESTS
# ==============================================================================


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ALL_GPU_MODES)
@pytest.mark.parametrize("candidate_k", [None, 2, 5])
def test_gpu_three_opt_improves_two_opt_optimal_trap(mode: GPUMode, candidate_k: int | None) -> None:
  """All 3 GPU 3-opt modes improve a strictly 2-opt-optimal trap (33 -> 21) while 2-opt stalls."""
  cp = _cupy()
  problem, tour = _make_two_opt_optimal_three_opt_improvable_problem()
  distances = problem.distances
  assert distances is not None
  assert tour_cost_cpu(tour, distances) == 33

  ctx_2opt = ProblemContext(problem, xp=cp, seed=42)
  two_opt_mode: GPUMode = "batch" if mode == "naive" else mode
  two_opt = LocalSearchGPUTwoOpt(mode=two_opt_mode, max_iterations=10, candidate_k=candidate_k)
  after_2opt = np.asarray(two_opt.improve(tour, ctx_2opt), dtype=np.int32)
  assert tour_cost_cpu(after_2opt, distances) == 33, "2-opt must not improve a 2-opt-optimal trap"

  ctx_3opt = ProblemContext(problem, xp=cp, seed=42)
  three_opt = LocalSearchGPUThreeOpt(max_iterations=5, mode=mode, candidate_k=candidate_k)
  after_3opt = np.asarray(three_opt.improve(tour, ctx_3opt), dtype=np.int32)
  cpu_ref = cpu_three_opt_refine(tour, distances, max_iterations=5)

  assert sorted(int(c) for c in after_3opt) == list(range(6))
  assert tour_cost_cpu(after_3opt, distances) == 21
  assert tour_cost_cpu(after_3opt, distances) == tour_cost_cpu(cpu_ref, distances)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ALL_GPU_MODES)
def test_gpu_three_opt_exact_single_pass_parity_open_and_closed(mode: str) -> None:
  """`batch` and `naive` match exact steepest CPU delta (`==`); `multi_move` satisfies `<=`."""
  cp = _cupy()
  # Open TSP (n=10)
  open_prob = _make_euclidean_problem(10, seed=7)
  open_ctx = ProblemContext(open_prob, xp=cp, seed=1)
  open_dist = open_ctx.get_cpu_distances()
  open_tours = _random_open_tours(10, 6, seed=19)

  gpu_out = np.asarray(
    LocalSearchGPUThreeOpt(max_iterations=1, mode=mode, candidate_k=None).improve_batch(open_tours, open_ctx),
    dtype=np.int32,
  )
  for before, after in zip(open_tours, gpu_out, strict=True):
    assert sorted(int(c) for c in after) == list(range(10))
    cpu_delta, *_ = cpu_three_opt_best_move(before, open_dist, is_closed=False)
    gpu_delta = tour_cost_cpu(after, open_dist) - tour_cost_cpu(before, open_dist)
    if mode in ("batch", "naive"):
      assert gpu_delta == cpu_delta
    else:
      assert gpu_delta <= cpu_delta
      if cpu_delta < 0:
        assert gpu_delta < 0

  # Closed CVRP tour `[0, c1, ..., c8, 0]` (length 10, matrix dim 9)
  rng = np.random.default_rng(29)
  raw = rng.integers(5, 300, size=(9, 9), dtype=np.int32)
  closed_dist = ((raw + raw.T) // 2).astype(np.int32)
  np.fill_diagonal(closed_dist, 0)
  closed_prob = _make_symmetric_explicit_problem("closed_cvrp9", closed_dist, problem_type="CVRP")
  closed_ctx = ProblemContext(closed_prob, xp=cp, seed=2)
  closed_tours = _random_closed_tours(8, 6, seed=31)

  gpu_closed = np.asarray(
    LocalSearchGPUThreeOpt(max_iterations=1, mode=mode, candidate_k=None).improve_batch(closed_tours, closed_ctx),
    dtype=np.int32,
  )
  for before, after in zip(closed_tours, gpu_closed, strict=True):
    assert int(after[0]) == 0 and int(after[-1]) == 0
    assert sorted(int(c) for c in after[1:-1]) == list(range(1, 9))
    cpu_delta, *_ = cpu_three_opt_best_move(before, closed_dist, is_closed=True)
    gpu_delta = tour_cost_cpu(after, closed_dist, is_closed=True) - tour_cost_cpu(before, closed_dist, is_closed=True)
    if mode in ("batch", "naive"):
      assert gpu_delta == cpu_delta
    else:
      assert gpu_delta <= cpu_delta


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ALL_GPU_MODES)
@pytest.mark.parametrize("candidate_k", [None, 12, 20])
def test_gpu_three_opt_non_increasing_costs_and_permutation_integrity(mode: str, candidate_k: int | None) -> None:
  """Tour costs are monotonically non-increasing across iterations and permutations stay intact."""
  cp = _cupy()
  n = 30
  problem = _make_euclidean_problem(n, seed=42)
  context = ProblemContext(problem, xp=cp, seed=42)
  distances = context.get_cpu_distances()
  current = _random_open_tours(n, 6, seed=99)
  prev_costs = [tour_cost_cpu(t, distances) for t in current]

  strategy = LocalSearchGPUThreeOpt(max_iterations=1, mode=mode, candidate_k=candidate_k)
  for _step in range(5):
    current = np.asarray(strategy.improve_batch(current, context), dtype=np.int32)
    costs = [tour_cost_cpu(t, distances) for t in current]
    for tour, before_c, after_c in zip(current, prev_costs, costs, strict=True):
      assert sorted(int(c) for c in tour) == list(range(n))
      assert after_c <= before_c
    prev_costs = costs

  assert sum(prev_costs) < sum(tour_cost_cpu(t, distances) for t in _random_open_tours(n, 6, seed=99))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", GRID_GPU_MODES)
@pytest.mark.parametrize("candidate_k", [None, 15])
def test_gpu_three_opt_coordinate_and_global_memory_paths_match_dense_shmem(
  mode: str,
  candidate_k: int | None,
) -> None:
  """On-the-fly coordinate mode and global-memory tour path match dense shared-memory output bit-for-bit."""
  cp = _cupy()
  n = 28
  problem = _make_euclidean_problem(n, seed=77)
  tours = _random_open_tours(n, 4, seed=88)

  dense_ctx = ProblemContext(problem, xp=cp, seed=1)
  dense_out = np.asarray(
    LocalSearchGPUThreeOpt(max_iterations=3, mode=mode, candidate_k=candidate_k).improve_batch(tours, dense_ctx),
    dtype=np.int32,
  )

  coords_ctx = ProblemContext(problem, xp=cp, seed=1)
  coords_strat = LocalSearchGPUThreeOpt(
    max_iterations=3, mode=mode, candidate_k=candidate_k, max_matrix_dimension=5
  )
  coords_out = np.asarray(coords_strat.improve_batch(tours, coords_ctx), dtype=np.int32)
  assert coords_strat._operands.distances is None
  assert coords_strat._operands.coords is not None
  np.testing.assert_array_equal(dense_out, coords_out)

  global_strat = LocalSearchGPUThreeOpt(
    max_iterations=3, mode=mode, candidate_k=candidate_k, max_shared_tour_cities=4
  )
  global_out = np.asarray(global_strat.improve_batch(tours, dense_ctx), dtype=np.int32)
  np.testing.assert_array_equal(dense_out, global_out)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ALL_GPU_MODES)
def test_gpu_three_opt_warmup_idempotence_residency_and_boundaries(mode: str) -> None:
  """Verify `warmup(context)` idempotence, CuPy device residency, telemetry, and short-tour no-op."""
  cp = _cupy()
  n = 24
  problem = _make_euclidean_problem(n, seed=5)
  context = ProblemContext(problem, xp=cp, seed=5)
  strategy = LocalSearchGPUThreeOpt(max_iterations=3, mode=mode, candidate_k=10)

  strategy.warmup(context)
  strategy.warmup(context)
  strategy.reset_metrics()
  assert strategy.h2d_bytes == 0
  assert strategy.d2h_bytes == 0
  assert strategy.kernel_launches == 0

  d_batch = cp.asarray(_random_open_tours(n, 4, seed=6), dtype=cp.int32)
  d_out = strategy.improve_batch(d_batch, context)
  assert is_gpu_array(d_out)
  assert d_out.dtype == cp.int32
  if mode in ("batch", "multi_move"):
    assert strategy.h2d_bytes == 0
    assert strategy.d2h_bytes == 0
    assert strategy.kernel_launches == 3
  else:
    assert strategy.h2d_bytes > 0
    assert strategy.d2h_bytes > 0
    assert strategy.kernel_launches == 4 * 3

  short = np.array([0, 2, 1], dtype=np.int32)
  short_out = strategy.improve(short, context)
  assert short_out is not short
  np.testing.assert_array_equal(short_out, short)

  zero_strat = LocalSearchGPUThreeOpt(max_iterations=0, mode=mode, candidate_k=10)
  full_tour = np.arange(n, dtype=np.int32)
  zero_out = zero_strat.improve(full_tour, context)
  assert zero_out is not full_tour
  np.testing.assert_array_equal(zero_out, full_tour)
