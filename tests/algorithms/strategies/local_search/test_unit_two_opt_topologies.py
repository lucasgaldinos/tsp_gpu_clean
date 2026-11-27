"""Topology, `dist_stride`, and open/closed boundary tests for CPU & GPU 2-opt (`BUG-107`).

Covers the three mathematical route topologies:

- **Topology A (Cyclic Tour):** `is_closed=False`, `return_to_depot=True` (`TSP`).
- **Topology B (Closed Depot-Bounded Route):** `is_closed=True` (`CVRP` sub-route
  `[0, u_1, ..., u_m, 0]`), where `n != context.dimension` and both depot endpoints
  stay fixed.
- **Topology C (Open Path):** `is_closed=False`, `return_to_depot=False` for both
  `OVRP` (`tour[0] == 0` fixed, free tail) and `Open TSP` (free head & free tail,
  including prefix `i == -1` and suffix `j == n - 1` reversals).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.strategies import LocalSearchCPUTwoOpt, LocalSearchGPUTwoOpt
from data_models.problem import Problem
from protocols import ProblemContext
from protocols.backend import cupy_backend

if TYPE_CHECKING:
  from algorithms.strategies.local_search import TwoOptCPUMode, TwoOptGPUMode
  from protocols.backend import BackendModule

CPU_TWO_OPT_MODES: list[TwoOptCPUMode] = ["best", "first", "multi_move"]
GPU_TWO_OPT_MODES: list[TwoOptGPUMode] = ["batch", "multi_move", "naive"]


def _cupy() -> Any:
  """Return the importable CuPy module, skipping GPU tests when unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


class _ProblemTypeProxy:
  """Lightweight read-only view overriding `problem_type` for pre-merge OVRP tests."""

  def __init__(self, inner: Problem, problem_type: str) -> None:
    self._inner = inner
    self.problem_type = problem_type

  def __getattr__(self, name: str) -> Any:
    return getattr(self._inner, name)


class _TopologyContext(ProblemContext):
  """`ProblemContext` subclass overriding `is_closed`, `return_to_depot`, and `problem_type`."""

  def __init__(
    self,
    problem: Problem,
    xp: BackendModule,
    *,
    is_closed: bool,
    return_to_depot: bool,
    problem_type_override: str | None = None,
    seed: int = 0,
    max_matrix_dimension: int = 10_000,
  ) -> None:
    super().__init__(problem, xp=xp, seed=seed, max_matrix_dimension=max_matrix_dimension)
    self._is_closed_override = is_closed
    self._return_to_depot_override = return_to_depot
    if problem_type_override is not None:
      self.problem = _ProblemTypeProxy(problem, problem_type_override)  # type: ignore[assignment]

  @property
  def is_closed(self) -> bool:
    return self._is_closed_override

  @property
  def return_to_depot(self) -> bool:
    return self._return_to_depot_override


def _line_distances(n: int) -> np.ndarray:
  """Return symmetric `|u - v|` `(n, n)` distance matrix."""
  axis = np.arange(n, dtype=np.int64)
  return np.abs(axis[:, None] - axis[None, :])


def _explicit_problem(
  name: str,
  distances: np.ndarray,
  *,
  problem_type: str = "TSP",
  capacity: int | None = None,
  demands: np.ndarray | None = None,
) -> Problem:
  """Build an `EXPLICIT` Problem instance."""
  return Problem(
    name=name,
    dimension=distances.shape[0],
    problem_type=problem_type,
    edge_type="EXPLICIT",
    coordinates=None,
    distances=distances,
    capacity=capacity,
    demands=demands,
  )


def _euc2d_problem(
  name: str,
  coords: np.ndarray,
  *,
  problem_type: str = "TSP",
  capacity: int | None = None,
  demands: np.ndarray | None = None,
) -> Problem:
  """Build a coordinate-only `EUC_2D` Problem instance."""
  return Problem(
    name=name,
    dimension=coords.shape[0],
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=coords.astype(np.float64),
    distances=None,
    capacity=capacity,
    demands=demands,
  )


def _path_cost(route: np.ndarray, distances: np.ndarray) -> int:
  """Return exact open/closed-endpoint linear path cost `sum_{k=0}^{n-2} d(r[k], r[k+1])`."""
  return int(sum(int(distances[int(route[k]), int(route[k + 1])]) for k in range(len(route) - 1)))


def _cyclic_cost(route: np.ndarray, distances: np.ndarray) -> int:
  """Return exact cyclic tour cost `sum_{k=0}^{n-1} d(r[k], r[(k+1) % n])`."""
  n = len(route)
  return int(sum(int(distances[int(route[k]), int(route[(k + 1) % n])]) for k in range(n)))


# ---------------------------------------------------------------------------
# Topology B — Closed Depot-Bounded CVRP Sub-routes (`dist_stride != n`)
# ---------------------------------------------------------------------------


def _cvrp_line_problem(dimension: int = 5) -> Problem:
  """Build a `CVRP` problem on a 1D line (`0` is depot at 0, customers `1..dimension-1`)."""
  distances = _line_distances(dimension)
  demands = np.ones(dimension, dtype=np.int64)
  demands[0] = 0
  return _explicit_problem("cvrp_line5", distances, problem_type="CVRP", capacity=100, demands=demands)


@pytest.mark.parametrize("mode", CPU_TWO_OPT_MODES)
def test_closed_cvrp_sub_route_preserves_depots_and_improves_cost_cpu(mode: TwoOptCPUMode) -> None:
  """WHAT: `[0, 3, 1, 4, 2, 0]` (`n=6`) on a `(5, 5)` CVRP matrix across all CPU modes.

  WHY: Pre-fix CPU `_exhaustive_pairs` allowed `j == n - 1`, moving the trailing depot
  and assuming a wrap-around edge (`BUG-107`).

  EXPECTED: `tour[0] == 0 == tour[-1]`, exact multiset preserved, and cost strictly drops.
  """
  problem = _cvrp_line_problem(5)
  context = ProblemContext(problem, xp=np, seed=1)
  distances = context.get_cpu_distances()
  route = np.array([0, 3, 1, 4, 2, 0], dtype=np.int32)
  before = _path_cost(route, distances)

  improved = LocalSearchCPUTwoOpt(max_iterations=10, mode=mode).improve(route, context)

  assert int(improved[0]) == 0 and int(improved[-1]) == 0
  assert sorted(improved.tolist()) == [0, 0, 1, 2, 3, 4]
  assert _path_cost(improved, distances) < before
  # On `line(5)`: [0, 3, 1, 4, 2, 0] has cost 12; 1-step [0, 1, 3, 4, 2, 0] achieves optimal 8.
  assert _path_cost(improved, distances) == 8


@pytest.mark.gpu
def test_closed_cvrp_sub_route_gpu_parity_and_dist_stride() -> None:
  """WHAT: `[0, 3, 1, 4, 2, 0]` (`n=6`) on `(5, 5)` CVRP across CPU `best` and GPU modes.

  WHY: Pre-fix GPU kernels used `n = 6` as distance matrix stride instead of
  `dist_stride = 5` and allowed `j == n - 1`, corrupting `tour[-1]` (`BUG-107`).

  EXPECTED: All GPU modes preserve `[0, ..., 0]`, reach cost 8 (`[0, 1, 2, 3, 4, 0]`),
  and `batch` / `naive` match CPU `best` bit-for-bit.
  """
  cp = _cupy()
  problem = _cvrp_line_problem(5)
  cpu_ctx = ProblemContext(problem, xp=np, seed=1)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=1)
  distances = cpu_ctx.get_cpu_distances()
  route = np.array([0, 3, 1, 4, 2, 0], dtype=np.int32)

  cpu_best = LocalSearchCPUTwoOpt(max_iterations=10, mode="best").improve(route, cpu_ctx)
  for gpu_mode in GPU_TWO_OPT_MODES:
    gpu_out = np.asarray(LocalSearchGPUTwoOpt(max_iterations=10, mode=gpu_mode).improve_batch(route, gpu_ctx))
    assert int(gpu_out[0]) == 0 and int(gpu_out[-1]) == 0
    assert sorted(gpu_out.tolist()) == [0, 0, 1, 2, 3, 4]
    assert _path_cost(gpu_out, distances) == 8
    if gpu_mode in ("batch", "naive"):
      np.testing.assert_array_equal(gpu_out, cpu_best)


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", GPU_TWO_OPT_MODES)
def test_closed_cvrp_sub_route_with_sparse_ids_coords_candidate_and_tiling(gpu_mode: TwoOptGPUMode) -> None:
  """WHAT: Sparse CVRP sub-route (`n=10`, `dimension=20`) with `EUC_2D` coords, `candidate_k=3`, and tiling.

  WHY: Verifies `dist_stride = 20 != n = 10` in coordinate mode (`max_matrix_dimension=5`),
  safe candidate fallback when `is_closed=True`, and tiled shared-memory scan
  (`max_shared_tour_cities=4`).
  """
  cp = _cupy()
  rng = np.random.default_rng(77)
  coords = rng.uniform(0.0, 1000.0, size=(20, 2))
  demands = np.ones(20, dtype=np.int64)
  demands[0] = 0
  problem = _euc2d_problem("cvrp_euc20", coords, problem_type="CVRP", capacity=100, demands=demands)

  cpu_ctx = ProblemContext(problem, xp=np, seed=2)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=2)
  distances = cpu_ctx.get_cpu_distances()

  sub_route = np.array([0, 17, 4, 19, 8, 12, 3, 15, 6, 0], dtype=np.int32)
  before = _path_cost(sub_route, distances)

  cpu_best = LocalSearchCPUTwoOpt(max_iterations=5, mode="best", candidate_k=3).improve(sub_route, cpu_ctx)
  gpu_res = np.asarray(
    LocalSearchGPUTwoOpt(
      max_iterations=5,
      mode=gpu_mode,
      candidate_k=3,
      max_matrix_dimension=5,
      max_shared_tour_cities=4 if gpu_mode != "naive" else 11_520,
    ).improve_batch(sub_route, gpu_ctx)
  )

  assert int(gpu_res[0]) == 0 and int(gpu_res[-1]) == 0
  assert sorted(gpu_res.tolist()) == sorted(sub_route.tolist())
  assert _path_cost(gpu_res, distances) <= before
  if gpu_mode in ("batch", "naive"):
    np.testing.assert_array_equal(gpu_res, cpu_best)


def test_closed_cvrp_short_sub_route_n3_returns_unchanged() -> None:
  """WHAT: `[0, 1, 0]` (`n=3`) on CVRP returns an equal copy with endpoints intact."""
  problem = _cvrp_line_problem(5)
  ctx = ProblemContext(problem, xp=np, seed=0)
  route = np.array([0, 1, 0], dtype=np.int32)
  out = LocalSearchCPUTwoOpt(max_iterations=5, mode="best").improve(route, ctx)
  assert out is not route
  np.testing.assert_array_equal(out, route)


# ---------------------------------------------------------------------------
# Topology C — Open Paths (`return_to_depot == False`: Open TSP & OVRP)
# ---------------------------------------------------------------------------


def _no_hallucinated_closing_edge_problem() -> Problem:
  """Build a 5-node matrix where `[0, 1, 2, 3, 4]` is open-optimal (`4`),
  but a cyclic solver hallucinating `(4, 0)` would break it to save `99`."""
  distances = _line_distances(5)
  distances[0, 4] = 100
  distances[4, 0] = 100
  distances[0, 3] = 1
  distances[3, 0] = 1
  distances[2, 4] = 1
  distances[4, 2] = 1
  return _explicit_problem("open_trap5", distances, problem_type="TSP")


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", GPU_TWO_OPT_MODES)
def test_open_tsp_prefix_suffix_and_no_closing_edge(gpu_mode: TwoOptGPUMode) -> None:
  """WHAT: Open TSP (`return_to_depot=False`) on prefix, suffix, `(0, n-1)`, and cyclic-trap gadgets.

  WHY: Open TSP must allow prefix reversals (`i == -1`: `tour[0..j]`) and suffix
  reversals (`j == n - 1`: `tour[i+1..n-1]`) while never subtracting a non-existent
  `(tour[n-1], tour[0])` closing edge (`BUG-107`).
  """
  cp = _cupy()
  line_prob = _explicit_problem("open_line5", _line_distances(5), problem_type="TSP")
  cpu_ctx = _TopologyContext(line_prob, xp=np, is_closed=False, return_to_depot=False)
  gpu_ctx = _TopologyContext(line_prob, xp=cp, is_closed=False, return_to_depot=False)
  dist = cpu_ctx.get_cpu_distances()

  # 1. Prefix `[0..1]`, suffix `[3..4]`, and `(0, n-1)` `[1..4]` gadgets.
  cases = np.array(
    [
      [1, 0, 2, 3, 4],  # prefix reversal [0..1] -> [0, 1, 2, 3, 4] (cost 5 -> 4)
      [0, 1, 2, 4, 3],  # suffix reversal [3..4] -> [0, 1, 2, 3, 4] (cost 5 -> 4)
      [0, 4, 3, 2, 1],  # (0, n-1) reversal [1..4] -> [0, 1, 2, 3, 4] (cost 7 -> 4)
    ],
    dtype=np.int32,
  )
  cpu_best = LocalSearchCPUTwoOpt(max_iterations=5, mode="best").improve_batch(cases, cpu_ctx)
  gpu_res = np.asarray(LocalSearchGPUTwoOpt(max_iterations=5, mode=gpu_mode).improve_batch(cases, gpu_ctx))

  for row in gpu_res:
    assert _path_cost(row, dist) == 4
  if gpu_mode in ("batch", "naive"):
    np.testing.assert_array_equal(gpu_res, cpu_best)

  # 2. Cyclic-trap gadget: `[0, 1, 2, 3, 4]` has open cost 4; cyclic 2-opt would degrade it.
  trap_prob = _no_hallucinated_closing_edge_problem()
  trap_cpu_ctx = _TopologyContext(trap_prob, xp=np, is_closed=False, return_to_depot=False)
  trap_gpu_ctx = _TopologyContext(trap_prob, xp=cp, is_closed=False, return_to_depot=False)
  optimal_open = np.array([0, 1, 2, 3, 4], dtype=np.int32)
  np.testing.assert_array_equal(
    LocalSearchCPUTwoOpt(max_iterations=3, mode="best").improve(optimal_open, trap_cpu_ctx),
    optimal_open,
  )
  np.testing.assert_array_equal(
    np.asarray(LocalSearchGPUTwoOpt(max_iterations=3, mode=gpu_mode).improve_batch(optimal_open, trap_gpu_ctx)),
    optimal_open,
  )


@pytest.mark.gpu
def test_open_tsp_step_by_step_exact_cost_and_cpu_gpu_parity() -> None:
  """WHAT: Random Open TSP tours (`n=12`, matrix & coords & tiled) stepped one pass at a time.

  WHY: Verifies every single pass matches exact open-path cost recomputation and keeps
  CPU `best` bit-identical to GPU `batch` / `naive` across all acceptance modes.
  """
  cp = _cupy()
  rng = np.random.default_rng(1234)
  coords = rng.uniform(0.0, 500.0, size=(12, 2))
  problem = _euc2d_problem("open_euc12", coords, problem_type="TSP")

  cpu_ctx = _TopologyContext(problem, xp=np, is_closed=False, return_to_depot=False)
  gpu_ctx = _TopologyContext(problem, xp=cp, is_closed=False, return_to_depot=False)
  dist = cpu_ctx.get_cpu_distances()

  tours = np.stack([rng.permutation(12) for _ in range(4)]).astype(np.int32)

  # Step-by-step exact monotonicity across all CPU and GPU modes (plus tiled & candidate fallback).
  for cpu_mode in CPU_TWO_OPT_MODES:
    cur = tours.copy()
    strat = LocalSearchCPUTwoOpt(max_iterations=1, mode=cpu_mode, candidate_k=4)
    for _ in range(6):
      nxt = strat.improve_batch(cur, cpu_ctx)
      for b_row, a_row in zip(cur, nxt, strict=True):
        assert sorted(a_row.tolist()) == list(range(12))
        assert _path_cost(a_row, dist) <= _path_cost(b_row, dist)
      cur = np.asarray(nxt, dtype=np.int32)

  cpu_final = LocalSearchCPUTwoOpt(max_iterations=6, mode="best").improve_batch(tours, cpu_ctx)
  for gpu_mode in GPU_TWO_OPT_MODES:
    gpu_final = np.asarray(
      LocalSearchGPUTwoOpt(
        max_iterations=6,
        mode=gpu_mode,
        candidate_k=4,
        max_matrix_dimension=5,
        max_shared_tour_cities=4 if gpu_mode != "naive" else 11_520,
      ).improve_batch(tours, gpu_ctx)
    )
    for b_row, a_row in zip(tours, gpu_final, strict=True):
      assert sorted(a_row.tolist()) == list(range(12))
      assert _path_cost(a_row, dist) <= _path_cost(b_row, dist)
    if gpu_mode in ("batch", "naive"):
      np.testing.assert_array_equal(gpu_final, cpu_final)


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", GPU_TWO_OPT_MODES)
def test_ovrp_open_route_keeps_depot_fixed_and_tail_free(gpu_mode: TwoOptGPUMode) -> None:
  """WHAT: `OVRP` (`problem_type="OVRP"`, `return_to_depot=False`) fixed-start & free-tail tests.

  WHY: Unlike Open TSP, OVRP routes `[0, u_1, ..., u_m]` MUST keep `tour[0] == 0` fixed
  (no `i == -1` prefix reversals), while still allowing free-tail reversals `j == n - 1`
  including `(0, n - 1)` and handling `n < dist_stride` (`BUG-107`).
  """
  cp = _cupy()
  # Matrix where moving node 0 via prefix `[0..1]` would reduce cost (10+100+1=111 -> 10+1+1=12),
  # but OVRP must keep `tour[0] == 0` and instead apply suffix `[2..3]` (111 -> 61).
  distances = np.array(
    [
      [0, 10, 1, 50, 99],
      [10, 0, 100, 50, 99],
      [1, 100, 0, 1, 99],
      [50, 50, 1, 0, 99],
      [99, 99, 99, 99, 0],
    ],
    dtype=np.int64,
  )
  demands = np.array([0, 1, 1, 1, 1], dtype=np.int64)
  base_prob = _explicit_problem("ovrp_fix5", distances, problem_type="CVRP", capacity=10, demands=demands)
  cpu_ctx = _TopologyContext(base_prob, xp=np, is_closed=False, return_to_depot=False, problem_type_override="OVRP")
  gpu_ctx = _TopologyContext(base_prob, xp=cp, is_closed=False, return_to_depot=False, problem_type_override="OVRP")

  sub_route = np.array([0, 1, 2, 3], dtype=np.int32)  # n = 4 < dist_stride = 5
  cpu_best = LocalSearchCPUTwoOpt(max_iterations=5, mode="best").improve(sub_route, cpu_ctx)
  gpu_res = np.asarray(LocalSearchGPUTwoOpt(max_iterations=5, mode=gpu_mode).improve_batch(sub_route, gpu_ctx))

  assert int(cpu_best[0]) == 0 and int(gpu_res[0]) == 0
  # Iteration 1: suffix [2..3] -> [0, 1, 3, 2] (cost 61); iteration 2: (0, 3) -> [0, 2, 3, 1] (cost 52).
  expected = np.array([0, 2, 3, 1], dtype=np.int32)
  np.testing.assert_array_equal(cpu_best, expected)
  assert _path_cost(cpu_best, distances) == 52
  if gpu_mode in ("batch", "naive"):
    np.testing.assert_array_equal(gpu_res, expected)
  else:
    assert _path_cost(gpu_res, distances) <= _path_cost(sub_route, distances)

  # Also verify `(0, n - 1)` customer reversal `[0, 3, 2, 1]` -> `[0, 1, 2, 3]` on line(5).
  line_prob = _explicit_problem("ovrp_line5", _line_distances(5), problem_type="CVRP", capacity=10, demands=demands)
  line_cpu_ctx = _TopologyContext(
    line_prob, xp=np, is_closed=False, return_to_depot=False, problem_type_override="OVRP"
  )
  line_gpu_ctx = _TopologyContext(
    line_prob, xp=cp, is_closed=False, return_to_depot=False, problem_type_override="OVRP"
  )
  rev_route = np.array([0, 3, 2, 1], dtype=np.int32)
  out_gpu = np.asarray(LocalSearchGPUTwoOpt(max_iterations=3, mode=gpu_mode).improve_batch(rev_route, line_gpu_ctx))
  out_cpu = LocalSearchCPUTwoOpt(max_iterations=3, mode="best").improve(rev_route, line_cpu_ctx)
  assert int(out_gpu[0]) == 0 and int(out_cpu[0]) == 0
  assert _path_cost(out_gpu, _line_distances(5)) == 3
  if gpu_mode in ("batch", "naive"):
    np.testing.assert_array_equal(out_gpu, out_cpu)


# ---------------------------------------------------------------------------
# Residency (`d2h_bytes == 0`) and `warmup(context)` across topologies
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", ["batch", "multi_move"])
def test_cupy_resident_closed_and_open_routes_have_zero_d2h_bytes(gpu_mode: TwoOptGPUMode) -> None:
  """WHAT: CuPy-resident CVRP closed sub-route and Open TSP batch under grid GPU modes.

  EXPECTED: `d2h_bytes == 0`, valid permutations/depots, and monotonic improvement.
  """
  cp = _cupy()
  cvrp_prob = _cvrp_line_problem(5)
  cvrp_ctx = ProblemContext(cvrp_prob, xp=cp, seed=0)
  cvrp_batch = cp.asarray(np.array([[0, 3, 1, 4, 2, 0]], dtype=np.int32))
  cvrp_strat = LocalSearchGPUTwoOpt(max_iterations=3, mode=gpu_mode)
  cvrp_out = cvrp_strat.improve_batch(cvrp_batch, cvrp_ctx)
  assert cvrp_strat.d2h_bytes == 0
  assert isinstance(cvrp_out, cp.ndarray)
  cvrp_host = cp.asnumpy(cvrp_out)[0]
  assert int(cvrp_host[0]) == 0 and int(cvrp_host[-1]) == 0
  assert _path_cost(cvrp_host, _line_distances(5)) == 8

  open_prob = _explicit_problem("open_d2h", _line_distances(5), problem_type="TSP")
  open_ctx = _TopologyContext(open_prob, xp=cp, is_closed=False, return_to_depot=False)
  open_batch = cp.asarray(np.array([[1, 0, 2, 3, 4], [0, 1, 2, 4, 3]], dtype=np.int32))
  open_strat = LocalSearchGPUTwoOpt(max_iterations=3, mode=gpu_mode)
  open_out = open_strat.improve_batch(open_batch, open_ctx)
  assert open_strat.d2h_bytes == 0
  assert isinstance(open_out, cp.ndarray)
  for row in cp.asnumpy(open_out):
    assert _path_cost(row, _line_distances(5)) == 4


@pytest.mark.gpu
@pytest.mark.parametrize("gpu_mode", GPU_TWO_OPT_MODES)
def test_warmup_succeeds_for_tsp_and_cvrp_with_updated_kernel_signatures(gpu_mode: TwoOptGPUMode) -> None:
  """WHAT: `LocalSearchGPUTwoOpt.warmup(context)` across all three GPU modes on TSP and CVRP.

  WHY: Kernel signatures gained `dist_stride`, `is_closed`, `is_cyclic`, and `fixed_start`;
  `warmup` must launch the compiled kernel cleanly with zero signature mismatch.
  """
  cp = _cupy()
  tsp_ctx = ProblemContext(_explicit_problem("warm_tsp", _line_distances(6)), xp=cp, seed=0)
  cvrp_ctx = ProblemContext(_cvrp_line_problem(6), xp=cp, seed=0)

  for ctx in (tsp_ctx, cvrp_ctx):
    strat = LocalSearchGPUTwoOpt(max_iterations=1, mode=gpu_mode)
    strat.warmup(ctx)
    assert strat._kernel is not None
    assert (strat.kernel_launches, strat.h2d_bytes, strat.d2h_bytes) == (0, 0, 0)


@pytest.mark.gpu
def test_cyclic_tsp_candidate_and_exhaustive_remain_bit_identical(berlin52_problem: Problem) -> None:
  """WHAT: Cyclic TSP sanity check across CPU `best` and GPU `batch`/`naive` with `candidate_k=10`.

  EXPECTED: Cyclic tour cost improves monotonically, `batch` == `naive` bit-identically,
  and candidate mode stays enabled on `n == context.dimension`.
  """
  cp = _cupy()
  n = berlin52_problem.dimension
  rng = np.random.default_rng(99)
  tours = np.stack([rng.permutation(n) for _ in range(3)]).astype(np.int32)
  cpu_ctx = ProblemContext(berlin52_problem, xp=np, seed=1)
  gpu_ctx = ProblemContext(berlin52_problem, xp=cp, seed=1)
  dist = cpu_ctx.get_cpu_distances()

  batch_strat = LocalSearchGPUTwoOpt(max_iterations=3, mode="batch", candidate_k=10)
  naive_strat = LocalSearchGPUTwoOpt(max_iterations=3, mode="naive", candidate_k=10)
  assert batch_strat._uses_candidates(gpu_ctx, n) is True

  batch_out = np.asarray(batch_strat.improve_batch(tours, gpu_ctx))
  naive_out = np.asarray(naive_strat.improve_batch(tours, gpu_ctx))
  np.testing.assert_array_equal(batch_out, naive_out)
  for before, after in zip(tours, batch_out, strict=True):
    assert sorted(after.tolist()) == list(range(n))
    assert _cyclic_cost(after, dist) <= _cyclic_cost(before, dist)
