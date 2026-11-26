"""Experiment-local GPU 3-opt local search strategy and CPU reference helper.

Implements `LocalSearchGPUThreeOpt(StrategyBase, LocalSearchStrategy)` outside `src/`,
mirroring `algorithms.strategies.local_search.LocalSearchGPUTwoOpt` with three kernel
families sharing one common device header (`kernels/three_opt_common.cuh`):

- `"batch"`: `kernels/three_opt_batch.cu` (`three_opt_batch_kernel`), one CUDA block
  per route, parallel block reduction applying the single steepest 3-opt move per pass.
- `"multi_move"`: `kernels/three_opt_multi_move.cu` (`three_opt_multi_move_batch_kernel`),
  one CUDA block per route, greedy disjoint multi-move 3-opt per pass (Rocki & Suda, 2013).
- `"naive"`: `kernels/three_opt_single.cu` (`three_opt_kernel`), per-route `CPU -> GPU -> CPU`
  transfers and single-block launches, mirroring `LocalSearchGPUTwoOpt(mode="naive")`.

Also exposes deterministic CPU 3-opt reference helpers (`tour_cost_cpu`,
`three_opt_delta_cpu`, `apply_three_opt_case_cpu`, `cpu_three_opt_best_move`,
`cpu_three_opt_refine`) for exact delta/reconnection parity verification in tests.

References:
    - Lin, S. (1965). "Computer Solutions of the Traveling Salesman Problem."
      Bell System Technical Journal, 44(10), 2245-2269.
    - Bentley, J. L. (1992). "Fast Algorithms for Geometric Traveling Salesman
      Problems." ORSA Journal on Computing, 4(4), 387-411.
    - Johnson, D. S., & McGeoch, L. A. (1997). "The Traveling Salesman Problem:
      A Case Study in Local Optimization." Local Search in Combinatorial Optimization.
    - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
      Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
    - Rocki, K., & Suda, R. (2013). "High Performance GPU Accelerated Local
      Optimization in TSP." IEEE IPDPSW, pp. 1788-1796.
    - O'Neil, M. A., & Burtscher, M. (2015). "Rethinking the Parallelization of
      Random-Restart Hill Climbing." GPGPU-8, pp. 99-108.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.two_opt import _DistanceOperandCache
from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import DISTANCE_KERNELS_DIR, prev_power_of_2
from utils.phase_timer import time_strategy
from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  resolve_route_bounds,
  restore_routes_shape,
)

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes

type ThreeOptGPUMode = Literal["batch", "multi_move", "naive"]
"""Kernel family selected by `LocalSearchGPUThreeOpt`, mirroring `TwoOptGPUMode`."""

_GPU_MODES: frozenset[str] = frozenset({"batch", "multi_move", "naive"})

#: Kernel source filename and CUDA entry point per GPU mode.
_GPU_KERNELS: dict[str, tuple[str, str]] = {
  "batch": ("three_opt_batch.cu", "three_opt_batch_kernel"),
  "multi_move": ("three_opt_multi_move.cu", "three_opt_multi_move_batch_kernel"),
  "naive": ("three_opt_single.cu", "three_opt_kernel"),
}

#: Local CUDA kernels directory containing `three_opt_common.cuh` and the three `.cu` kernels.
_KERNELS_DIR: Path = Path(__file__).resolve().parent / "kernels"

#: Shared header included by every 3-opt kernel (hashed alongside `.cu` for cache invalidation).
_COMMON_HEADER: Path = _KERNELS_DIR / "three_opt_common.cuh"

#: Last combined source+header digest seen per kernel path (`BUG-58` cache safety).
_KERNEL_DIGESTS: dict[str, str] = {}

#: Largest `n` cached in shared memory alongside `edge_costs` (`8*n + 20*256 <= 37,888 B < 48 KB`).
_DEFAULT_MAX_SHARED_TOUR_CITIES: int = 4_096

#: Per-thread shared-memory integer arrays (`s_deltas`, `s_i`, `s_j`, `s_k`, `s_case`).
_SHMEM_INTS_PER_THREAD: int = 5


@functools.lru_cache(maxsize=16)
def _compile_three_opt_kernel_cached(kernel_path_str: str, entry_point: str) -> Any:
  """Compile and cache a 3-opt CUDA kernel with include paths for both common headers."""
  cp = require_gpu()
  source = Path(kernel_path_str).read_text(encoding="utf-8")
  options = (f"-I{DISTANCE_KERNELS_DIR}", f"-I{_KERNELS_DIR}")
  return cp.RawKernel(source, entry_point, options=options)


def _load_three_opt_kernel(source_filename: str, entry_point: str) -> Any:
  """Load a 3-opt kernel from `kernels/`, invalidating the cache if source or header changed."""
  kernel_path = _KERNELS_DIR / source_filename
  if not kernel_path.exists():
    raise FileNotFoundError(f"3-opt kernel file not found: {kernel_path}")
  hasher = hashlib.sha256(kernel_path.read_bytes())
  if _COMMON_HEADER.exists():
    hasher.update(_COMMON_HEADER.read_bytes())
  digest = hasher.hexdigest()
  key = str(kernel_path)
  previous = _KERNEL_DIGESTS.get(key)
  if previous is not None and previous != digest:
    _compile_three_opt_kernel_cached.cache_clear()
  _KERNEL_DIGESTS[key] = digest
  return _compile_three_opt_kernel_cached(key, entry_point)


def _shared_mem_bytes(block_size: int, n: int, use_shared_tour: bool) -> int:
  """Return dynamic shared-memory bytes for one CUDA block of the 3-opt kernels.

  Layout:
    - Always reserves `block_size * 5 * 4` bytes (`s_deltas`, `s_i`, `s_j`, `s_k`, `s_case`).
    - When `use_shared_tour` is `True`, appends `2 * n * 4` bytes for `s_tour[n]` and
      `s_edge[n]`. Larger tours bypass shared-memory tour caching so dynamic shared
      memory stays capped at 5 KB (safe for GTX 1050 48 KB/block limit).
  """
  base = block_size * _SHMEM_INTS_PER_THREAD * 4
  if use_shared_tour:
    return base + 2 * n * 4
  return base


# ==============================================================================
# CPU 3-OPT REFERENCE HELPERS (EXACT DELTA & RECONNECTION PARITY ORACLE)
# ==============================================================================


def tour_cost_cpu(tour: np.ndarray, distances: np.ndarray, *, is_closed: bool = False) -> int:
  """Compute exact integer tour cost on CPU for open (cyclic TSP) or closed (CVRP) routes."""
  n = len(tour)
  if n < 2:
    return 0
  if is_closed:
    return int(sum(int(distances[int(tour[idx]), int(tour[idx + 1])]) for idx in range(n - 1)))
  return int(sum(int(distances[int(tour[idx]), int(tour[(idx + 1) % n])]) for idx in range(n)))


def three_opt_delta_cpu(
  tour: np.ndarray,
  distances: np.ndarray,
  i: int,
  j: int,
  k: int,
  case_id: int,
) -> int:
  """Return exact cost change `Delta` for applying 3-opt `case_id` (`1..7`) at cuts `(i, j, k)`.

  Cuts remove edges `(a, b) = (tour[i], tour[i + 1])`, `(c, d) = (tour[j], tour[j + 1])`,
  and `(e, f) = (tour[k], tour[(k + 1) % n])` for `0 <= i < j < k < n` (or pure 2-opt
  sentinel `0 <= i < j == k < n` when `case_id == 1`). Assumes symmetric distances.

  Cases:
    - 1 (2-opt):    `a -> S1' -> S2  -> f`
    - 2 (2-opt):    `a -> S1  -> S2' -> f`
    - 3 (2-opt):    `a -> S2' -> S1' -> f`
    - 4 (3-opt):    `a -> S1' -> S2' -> f`
    - 5 (3-opt):    `a -> S2  -> S1  -> f` (orientation-preserving block swap)
    - 6 (3-opt):    `a -> S2  -> S1' -> f`
    - 7 (3-opt):    `a -> S2' -> S1  -> f`
  """
  n = len(tour)
  a = int(tour[i])
  b = int(tour[i + 1])
  c = int(tour[j])
  d = int(tour[(j + 1) % n])

  if case_id == 1:
    return int(distances[a, c]) + int(distances[b, d]) - int(distances[a, b]) - int(distances[c, d])

  e = int(tour[k])
  f = int(tour[(k + 1) % n])
  d_ab = int(distances[a, b])
  d_cd = int(distances[c, d])
  d_ef = int(distances[e, f])
  old3 = d_ab + d_cd + d_ef

  if case_id == 2:
    return int(distances[c, e]) + int(distances[d, f]) - d_cd - d_ef
  if case_id == 3:
    return int(distances[a, e]) + int(distances[b, f]) - d_ab - d_ef
  if case_id == 4:
    return int(distances[a, c]) + int(distances[b, e]) + int(distances[d, f]) - old3
  if case_id == 5:
    return int(distances[a, d]) + int(distances[b, e]) + int(distances[c, f]) - old3
  if case_id == 6:
    return int(distances[a, d]) + int(distances[c, e]) + int(distances[b, f]) - old3
  if case_id == 7:
    return int(distances[a, e]) + int(distances[b, d]) + int(distances[c, f]) - old3
  raise ValueError(f"Invalid 3-opt case_id {case_id}; expected 1..7")


def apply_three_opt_case_cpu(
  tour: np.ndarray,
  i: int,
  j: int,
  k: int,
  case_id: int,
) -> np.ndarray:
  """Return a new route with 3-opt `case_id` (`1..7`) applied at cuts `(i, j, k)`.

  Mirrors the exact 3-step in-place reversal decomposition in `three_opt_common.cuh`.
  """
  result = np.asarray(tour, dtype=np.int32).copy()
  seg_start = i + 1
  if case_id in (1, 4, 5, 7):
    result[seg_start : j + 1] = result[seg_start : j + 1][::-1]
  if case_id in (2, 4, 5, 6):
    result[j + 1 : k + 1] = result[j + 1 : k + 1][::-1]
  if case_id in (3, 5, 6, 7):
    result[seg_start : k + 1] = result[seg_start : k + 1][::-1]
  return result


def cpu_three_opt_best_move(
  tour: np.ndarray,
  distances: np.ndarray,
  *,
  is_closed: bool = False,
) -> tuple[int, int, int, int, int]:
  """Find the steepest improving exhaustive 3-opt move `(delta, i, j, k, case_id)`.

  Returns `(0, -1, -1, -1, 0)` when the tour is 3-opt locally optimal.
  """
  n = len(tour)
  if n < 4:
    return (0, -1, -1, -1, 0)
  k_max = n - 2 if is_closed else n - 1

  best_delta = 0
  best_move = (0, -1, -1, -1, 0)
  for i in range(k_max - 1):
    for j in range(i + 1, k_max):
      if j > i + 1:
        delta1 = three_opt_delta_cpu(tour, distances, i, j, j, 1)
        if delta1 < best_delta:
          best_delta = delta1
          best_move = (delta1, i, j, j, 1)
      for k in range(j + 1, k_max + 1):
        for case_id in (2, 3, 4, 5, 6, 7):
          if case_id == 2 and k <= j + 1:
            continue
          if case_id == 3 and not is_closed and i == 0 and k == n - 1:
            continue
          delta = three_opt_delta_cpu(tour, distances, i, j, k, case_id)
          if delta < best_delta:
            best_delta = delta
            best_move = (delta, i, j, k, case_id)
  return best_move


def cpu_three_opt_refine(
  tour: np.ndarray,
  distances: np.ndarray,
  *,
  max_iterations: int = 10,
  is_closed: bool = False,
) -> np.ndarray:
  """Iteratively apply steepest-descent exhaustive CPU 3-opt up to `max_iterations` passes."""
  current = np.asarray(tour, dtype=np.int32).copy()
  if len(current) < 4 or max_iterations <= 0:
    return current
  for _ in range(max_iterations):
    delta, i, j, k, case_id = cpu_three_opt_best_move(current, distances, is_closed=is_closed)
    if delta >= 0 or case_id == 0:
      break
    current = apply_three_opt_case_cpu(current, i, j, k, case_id)
  return current


# ==============================================================================
# GPU 3-OPT LOCAL SEARCH STRATEGY
# ==============================================================================


class LocalSearchGPUThreeOpt(StrategyBase, LocalSearchStrategy):
  """Modular GPU CUDA 3-opt local search with selectable kernel family (`batch | multi_move | naive`).

  Mirrors `LocalSearchGPUTwoOpt`:
    - `"batch"`: `three_opt_batch.cu`, one CUDA block per route, single steepest 3-opt move per pass.
    - `"multi_move"`: `three_opt_multi_move.cu`, greedy disjoint improving 3-opt moves per pass.
    - `"naive"`: `three_opt_single.cu`, per-route `CPU -> GPU -> CPU` transfers and 1-block launches.

  All three families share `kernels/three_opt_common.cuh`, support open TSP and closed CVRP
  bounds, honour `candidate_k`, preserve CuPy device residency on grid modes, and track
  `h2d_bytes`, `d2h_bytes`, and `kernel_launches` with zero unnecessary synchronizations.
  """

  def __init__(
    self,
    max_iterations: int = 10,
    mode: ThreeOptGPUMode = "batch",
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    max_shared_tour_cities: int = _DEFAULT_MAX_SHARED_TOUR_CITIES,
    candidate_k: int | None = None,
    inner_candidate_k: int | None = None,
  ) -> None:
    """Initialize the GPU 3-opt strategy.

    Args:
        max_iterations: Maximum 3-opt passes per route; `0` disables refinement.
        mode: Kernel family, one of `"batch"`, `"multi_move"`, `"naive"`.
        threads_per_block: Maximum CUDA threads per block (clamped to power of 2 in `[1, 256]`).
        max_matrix_dimension: `n` above which coordinate-based problems switch to on-the-fly
            distances (`None` defers to `context.max_matrix_dimension`).
        max_shared_tour_cities: `n` above which `s_tour`/`s_edge` are read from global memory
            instead of shared memory (default `4096`, keeping shared memory <= 38 KB < 48 KB).
        candidate_k: Nearest-neighbor list width `N_k` for pruned 3-opt (`None` or `<= 0`
            selects exhaustive 3-opt scan).
        inner_candidate_k: Optional inner-loop width cap for secondary 3-opt candidate probes
            (`None` uses `candidate_k`).
    """
    super().__init__()
    if mode not in _GPU_MODES:
      raise ValueError(f"Invalid 3-opt GPU mode {mode!r}; expected one of {sorted(_GPU_MODES)}")
    if max_iterations < 0:
      raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    if threads_per_block <= 0:
      raise ValueError(f"threads_per_block must be > 0, got {threads_per_block}")
    if max_shared_tour_cities < 0:
      raise ValueError(f"max_shared_tour_cities must be >= 0, got {max_shared_tour_cities}")
    if inner_candidate_k is not None and inner_candidate_k < 0:
      raise ValueError(f"inner_candidate_k must be >= 0 or None, got {inner_candidate_k}")
    self.mode: ThreeOptGPUMode = mode
    self.max_iterations: int = max_iterations
    self.threads_per_block: int = threads_per_block
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.max_shared_tour_cities: int = max_shared_tour_cities
    self.candidate_k: int | None = candidate_k
    self.inner_candidate_k: int | None = inner_candidate_k
    self._kernel: Any = None
    self._operands: _DistanceOperandCache = _DistanceOperandCache()

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    super().reset_metrics()
    self._operands.reset()

  def _get_kernel(self) -> Any:
    """Lazily compile and return the CUDA kernel selected by `mode`."""
    if self._kernel is None:
      source, entry_point = _GPU_KERNELS[self.mode]
      self._kernel = _load_three_opt_kernel(source, entry_point)
    return self._kernel

  def warmup(self, context: ProblemContext) -> None:
    """Compile the selected CUDA kernel and launch it once on a valid permutation."""
    super().warmup(context)
    self.improve_batch(self._warmup_batch(context, rows=1), context)

  def _uses_candidates(self, context: ProblemContext, n: int, *, is_closed: bool) -> bool:
    """Return True when the candidate-list 3-opt path is admissible without D2H syncs."""
    if self.candidate_k is None or self.candidate_k <= 0:
      return False
    if is_closed or context.problem.problem_type == "CVRP":
      return False
    return n >= context.dimension

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` through the selected GPU 3-opt kernel."""
    return self.improve_batch(route, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route `(n_cities,)` or 2D batch `(N, n_cities)` on GPU."""
    require_gpu()
    if self.mode == "naive":
      return self._improve_naive(routes, context)
    return self._improve_grid(routes, context)

  def _improve_naive(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Run the per-route `three_opt_single.cu` baseline with one transfer pair per route."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    num_routes, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    batch_np = cp.asnumpy(batch) if input_on_gpu else np.asarray(batch, dtype=np.int32)

    low, high = resolve_route_bounds(n, context=context, route_sample=batch_np[0])
    is_closed: bool = (low, high) != (0, n)

    distances_cpu = context.get_cpu_distances()
    distances_gpu = cp.asarray(distances_cpu, dtype=cp.int32)
    self.h2d_bytes += int(distances_cpu.nbytes)
    dist_stride: int = int(distances_gpu.shape[1])

    dev_shared_mem = int(getattr(cp.cuda.Device(), "attributes", {}).get("MaxSharedMemoryPerBlock", 48 * 1024))
    max_anchors = n if not is_closed else max(1, n - 3)
    block_size: int = prev_power_of_2(min(self.threads_per_block, max(1, max_anchors)), min_val=1, max_val=256)
    use_shared_tour: bool = (
      n <= self.max_shared_tour_cities and _shared_mem_bytes(block_size, n, use_shared_tour=True) <= dev_shared_mem
    )
    shared_mem: int = _shared_mem_bytes(block_size, n, use_shared_tour=use_shared_tour)

    candidate_count: int = 0
    inner_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n, is_closed=is_closed):
      d_candidates, candidate_bytes = self._operands.resolve_candidates(context, int(self.candidate_k or 0))
      self.h2d_bytes += candidate_bytes
      candidate_count = int(d_candidates.shape[1])
      inner_count = (
        min(candidate_count, int(self.inner_candidate_k))
        if self.inner_candidate_k is not None and self.inner_candidate_k > 0
        else candidate_count
      )
      d_positions = cp.empty(n, dtype=cp.int32)

    d_edge_costs: Any = 0 if use_shared_tour else cp.empty(n, dtype=cp.int32)
    kernel = self._get_kernel()

    improved = np.empty_like(batch_np, dtype=np.int32)
    for idx in range(num_routes):
      tour = batch_np[idx]
      d_tour = cp.asarray(tour, dtype=cp.int32)
      self.h2d_bytes += int(tour.nbytes)

      for _ in range(self.max_iterations):
        kernel(
          (1,),
          (block_size,),
          (
            d_tour,
            distances_gpu,
            np.int32(n),
            np.int32(dist_stride),
            np.int32(1 if is_closed else 0),
            np.int32(1 if use_shared_tour else 0),
            d_candidates,
            np.int32(candidate_count),
            np.int32(inner_count),
            d_positions,
            d_edge_costs,
          ),
          shared_mem=shared_mem,
        )
        self.kernel_launches += 1

      improved[idx] = d_tour.get()
      self.d2h_bytes += int(tour.nbytes)

    if input_on_gpu:
      return restore_routes_shape(cp.asarray(improved, dtype=cp.int32), was_1d)
    return restore_routes_shape(improved, was_1d)

  def _improve_grid(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Run the `N`-block grid kernels (`batch` or `multi_move`) with one batched transfer."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    batch_size, n = batch.shape
    if n < 4 or self.max_iterations <= 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu: bool = is_gpu_array(batch)
    if input_on_gpu:
      d_population = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      batch_np = np.asarray(batch, dtype=np.int32)
      d_population = cp.asarray(batch_np, dtype=cp.int32)
      self.h2d_bytes += int(batch_np.nbytes)

    low, high = resolve_route_bounds(n, context=context, route_sample=batch[0])
    is_closed: bool = (low, high) != (0, n)

    (d_distances, d_coords, edge_type_code), uploaded_bytes = self._operands.resolve(
      context, context.dimension, self.max_matrix_dimension
    )
    self.h2d_bytes += uploaded_bytes
    dist_stride: int = context.dimension if isinstance(d_distances, int) else int(d_distances.shape[1])

    dev_shared_mem = int(getattr(cp.cuda.Device(), "attributes", {}).get("MaxSharedMemoryPerBlock", 48 * 1024))
    max_anchors = n if not is_closed else max(1, n - 3)
    block_size: int = prev_power_of_2(min(self.threads_per_block, max(1, max_anchors)), min_val=1, max_val=256)

    use_shared_tour: bool = (
      n <= self.max_shared_tour_cities and _shared_mem_bytes(block_size, n, use_shared_tour=True) <= dev_shared_mem
    )
    shared_mem: int = _shared_mem_bytes(block_size, n, use_shared_tour=use_shared_tour)

    candidate_count: int = 0
    inner_count: int = 0
    d_candidates: Any = 0
    d_positions: Any = 0
    if self._uses_candidates(context, n, is_closed=is_closed):
      d_candidates, candidate_bytes = self._operands.resolve_candidates(context, int(self.candidate_k or 0))
      self.h2d_bytes += candidate_bytes
      candidate_count = int(d_candidates.shape[1])
      inner_count = (
        min(candidate_count, int(self.inner_candidate_k))
        if self.inner_candidate_k is not None and self.inner_candidate_k > 0
        else candidate_count
      )
      d_positions = cp.empty((batch_size, n), dtype=cp.int32)

    d_edge_costs: Any = 0 if use_shared_tour else cp.empty((batch_size, n), dtype=cp.int32)

    kernel = self._get_kernel()
    for _ in range(self.max_iterations):
      kernel(
        (batch_size,),
        (block_size,),
        (
          d_population,
          d_distances,
          d_coords,
          np.int32(n),
          np.int32(dist_stride),
          np.int32(batch_size),
          np.int32(edge_type_code),
          np.int32(1 if is_closed else 0),
          np.int32(1 if use_shared_tour else 0),
          d_candidates,
          np.int32(candidate_count),
          np.int32(inner_count),
          d_positions,
          d_edge_costs,
        ),
        shared_mem=shared_mem,
      )
      self.kernel_launches += 1

    if input_on_gpu:
      return restore_routes_shape(d_population, was_1d)

    improved = d_population.get()
    self.d2h_bytes += int(improved.nbytes)
    return restore_routes_shape(improved, was_1d)


__all__: list[str] = [
  "LocalSearchGPUThreeOpt",
  "ThreeOptGPUMode",
  "apply_three_opt_case_cpu",
  "cpu_three_opt_best_move",
  "cpu_three_opt_refine",
  "three_opt_delta_cpu",
  "tour_cost_cpu",
]
