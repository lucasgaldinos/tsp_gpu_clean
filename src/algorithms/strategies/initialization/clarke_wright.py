"""Parallel Clarke & Wright (1964) savings CPU and GPU initializers (`Task 17.31`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.gpu_base import DistanceOperandCache, ensure_gpu_workspace
from protocols.algorithm_strategies import InitializationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.splitmix64 import draw_row_seeds, splitmix64_bounded_scalar

from .vrp_helpers import compute_clarke_wright_pairs, pack_vrp_routes_np, resolve_vrp_customers

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


def _validate_cw_params(
  shape_param: float,
  noise: float,
  candidate_k: int | None,
  output_format: str,
) -> tuple[float, float, int | None, Literal["giant_tour", "delimited"]]:
  if (
    isinstance(shape_param, bool)
    or not isinstance(shape_param, (int, float))
    or not math.isfinite(shape_param)
    or shape_param <= 0.0
  ):
    raise ValueError(f"shape_param must be a finite number > 0, got {shape_param!r}")
  if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0.0:
    raise ValueError(f"noise must be a finite number >= 0, got {noise!r}")
  if candidate_k is not None and (isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1):
    raise ValueError(f"candidate_k must be >= 1 or None, got {candidate_k!r}")
  if output_format not in ("giant_tour", "delimited"):
    raise ValueError(f"output_format must be 'giant_tour' or 'delimited', got {output_format!r}")
  return (
    float(shape_param),
    float(noise),
    candidate_k,
    cast("Literal['giant_tour', 'delimited']", output_format),
  )


def _compute_rcl_width(noise: float) -> int:
  """Map non-negative ``noise`` onto sliding-window RCL width in ``[1, 32]``."""
  if noise <= 0.0:
    return 1
  return max(1, min(32, math.floor(1.0 + noise * 32.0 + 0.5)))


def _reverse_chain_cpu(start_node: int, prev_node: np.ndarray, next_node: np.ndarray) -> None:
  cur = start_node
  while cur != -1:
    nxt = int(next_node[cur])
    next_node[cur] = prev_node[cur]
    prev_node[cur] = nxt
    cur = nxt


def _clarke_wright_row_cpu(
  customer_ids: np.ndarray,
  pairs_u: np.ndarray,
  pairs_v: np.ndarray,
  distances: np.ndarray,
  demands: np.ndarray,
  context: ProblemContext,
  *,
  seed_u64: int,
  rcl_width: int,
) -> list[np.ndarray]:
  """Run one parallel Clarke & Wright merge pass matching ``clarke_wright.cu`` bit-for-bit."""
  n_cust = int(customer_ids.size)
  if n_cust == 0:
    return []

  constraints = context.constraints
  capacity = constraints.capacity
  max_route_distance = constraints.max_route_distance
  return_to_depot = constraints.return_to_depot
  is_symmetric = context.problem.is_symmetric

  dim = context.dimension
  prev_node = np.full(dim, -1, dtype=np.int32)
  next_node = np.full(dim, -1, dtype=np.int32)
  other_end = np.arange(dim, dtype=np.int32)
  route_load = np.zeros(dim, dtype=np.int64)
  route_load[customer_ids] = demands[customer_ids].astype(np.int64)
  route_open_dist = np.zeros(dim, dtype=np.int64)

  state = int(seed_u64)
  n_pairs = int(pairs_u.size)
  win_base = 0
  used_mask = 0

  while win_base < n_pairs:
    while win_base < n_pairs and (used_mask & 1) != 0:
      win_base += 1
      used_mask >>= 1
    if win_base >= n_pairs:
      break

    avail = min(rcl_width, n_pairs - win_base)
    k = 0
    if rcl_width > 1 and avail > 1:
      state, r = splitmix64_bounded_scalar(state, avail)
      for step in range(avail):
        cand = (r + step) % avail
        if (used_mask & (1 << cand)) == 0:
          k = cand
          break
    used_mask |= 1 << k

    pair_idx = win_base + k
    u = int(pairs_u[pair_idx])
    v = int(pairs_v[pair_idx])
    if u == v or int(other_end[u]) == v:
      continue

    u_at_end = int(next_node[u]) == -1
    u_at_start = int(prev_node[u]) == -1
    v_at_start = int(prev_node[v]) == -1
    v_at_end = int(next_node[v]) == -1

    if is_symmetric:
      if (not u_at_end and not u_at_start) or (not v_at_start and not v_at_end):
        continue
    elif not u_at_end or not v_at_start:
      continue

    new_load = int(route_load[u]) + int(route_load[v])
    if capacity is not None and new_load > capacity:
      continue

    a = int(other_end[u])
    b = int(other_end[v])
    new_open = int(route_open_dist[u]) + int(distances[u, v]) + int(route_open_dist[v])
    if max_route_distance is not None:
      d_0a = int(distances[0, a])
      d_b0 = int(distances[b, 0]) if return_to_depot else 0
      if float(d_0a + new_open + d_b0) > max_route_distance:
        continue

    if is_symmetric:
      if not u_at_end:
        _reverse_chain_cpu(u, prev_node, next_node)
      if not v_at_start:
        _reverse_chain_cpu(b, prev_node, next_node)

    next_node[u] = v
    prev_node[v] = u
    other_end[a] = b
    other_end[b] = a
    route_load[a] = new_load
    route_load[b] = new_load
    route_open_dist[a] = new_open
    route_open_dist[b] = new_open

  routes: list[np.ndarray] = []
  for cid_raw in customer_ids:
    head = int(cid_raw)
    if int(prev_node[head]) != -1:
      continue
    chain: list[int] = []
    cur = head
    while cur != -1:
      chain.append(cur)
      cur = int(next_node[cur])
    routes.append(np.asarray(chain, dtype=np.int32))
  return routes


class InitializationCPUClarkeWright(StrategyBase, InitializationStrategy):
  """CPU Parallel Clarke & Wright (1964) savings initializer with ``SplitMix64`` RCL."""

  def __init__(
    self,
    shape_param: float = 1.0,
    noise: float = 0.15,
    candidate_k: int | None = 32,
    output_format: Literal["giant_tour", "delimited"] = "giant_tour",
    seed: SeedLike = None,
  ) -> None:
    super().__init__(seed)
    self.shape_param, self.noise, self.candidate_k, self.output_format = _validate_cw_params(
      shape_param, noise, candidate_k, output_format
    )

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` Clarke & Wright VRP routes on CPU."""
    if context is None:
      raise ValueError("InitializationCPUClarkeWright requires a ProblemContext")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    customer_ids = resolve_vrp_customers(customers, context)
    n_cust = int(customer_ids.size)
    k_max = context.constraints.max_vehicles or max(1, n_cust)
    row_width = n_cust if self.output_format == "giant_tour" else context.dimension + k_max
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0:
      return np.empty((0, row_width), dtype=np.int32)

    distances = context.get_cpu_distances()
    demands = context.get_cpu_demands()
    if demands is None:
      demands = np.zeros(context.dimension, dtype=np.int32)

    pairs_u, pairs_v = compute_clarke_wright_pairs(
      customer_ids,
      context,
      shape_param=self.shape_param,
      candidate_k=self.candidate_k,
      distances=distances,
    )
    rcl_nonzero = _compute_rcl_width(self.noise)

    pop = np.empty((population_size, row_width), dtype=np.int32)
    for idx in range(population_size):
      routes = _clarke_wright_row_cpu(
        customer_ids,
        pairs_u,
        pairs_v,
        distances,
        demands,
        context,
        seed_u64=int(row_seeds[idx]),
        rcl_width=1 if idx == 0 else rcl_nonzero,
      )
      pop[idx] = pack_vrp_routes_np(
        routes,
        dimension=context.dimension,
        k_max=k_max,
        output_format=self.output_format,
      )
    return pop


class InitializationGPUClarkeWright(StrategyBase, InitializationStrategy):
  """VRAM-native CUDA Clarke & Wright savings initializer bit-identical to ``InitializationCPUClarkeWright``."""

  def __init__(
    self,
    shape_param: float = 1.0,
    noise: float = 0.15,
    candidate_k: int | None = 32,
    output_format: Literal["giant_tour", "delimited"] = "giant_tour",
    max_matrix_dimension: int | None = None,
    return_gpu: bool = True,
    seed: SeedLike = None,
  ) -> None:
    if max_matrix_dimension is not None and (isinstance(max_matrix_dimension, bool) or max_matrix_dimension < 0):
      raise ValueError(f"max_matrix_dimension must be >= 0 or None, got {max_matrix_dimension!r}")
    super().__init__(seed)
    self.shape_param, self.noise, self.candidate_k, self.output_format = _validate_cw_params(
      shape_param, noise, candidate_k, output_format
    )
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None
    self._operands: DistanceOperandCache = DistanceOperandCache()
    self._ws_prev: Any = None
    self._ws_next: Any = None
    self._ws_other: Any = None
    self._ws_load: Any = None
    self._ws_open_dist: Any = None

  def reset_metrics(self) -> None:
    """Reset transfer/launch counters, operand bindings, and cached workspaces."""
    super().reset_metrics()
    self._operands.reset()
    self._ws_prev = None
    self._ws_next = None
    self._ws_other = None
    self._ws_load = None
    self._ws_open_dist = None

  def _get_kernel(self) -> Any:
    """Lazily compile and cache ``clarke_wright_init_kernel``."""
    if self._kernel is None:
      self._kernel = load_kernel("clarke_wright.cu", "clarke_wright_init_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch ``clarke_wright_init_kernel`` ahead of measured runs without mutating state."""
    require_gpu()
    super()._warmup_impl(context)
    self._get_kernel()
    if context.dimension <= 1:
      return
    self.initialize(list(range(1, context.dimension)), 2, context)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` Clarke & Wright VRP routes directly in VRAM."""
    if context is None:
      raise ValueError("InitializationGPUClarkeWright requires a ProblemContext")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    cp = require_gpu()
    customer_ids = resolve_vrp_customers(customers, context)
    n_cust = int(customer_ids.size)
    k_max = context.constraints.max_vehicles or max(1, n_cust)
    row_width = n_cust if self.output_format == "giant_tour" else context.dimension + k_max
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0:
      empty_gpu = cp.empty((0, row_width), dtype=cp.int32)
      return cast("Routes", empty_gpu) if self.return_gpu else np.empty((0, row_width), dtype=np.int32)

    pairs_u, pairs_v = compute_clarke_wright_pairs(
      customer_ids,
      context,
      shape_param=self.shape_param,
      candidate_k=self.candidate_k,
      distances=None,
    )
    n_pairs = int(pairs_u.size)

    d_customers = cp.asarray(customer_ids, dtype=cp.int32)
    d_seeds = cp.asarray(row_seeds, dtype=cp.uint64)
    if n_pairs > 0:
      d_pairs_u = cp.asarray(pairs_u, dtype=cp.int32)
      d_pairs_v = cp.asarray(pairs_v, dtype=cp.int32)
      self.h2d_bytes += int(pairs_u.nbytes + pairs_v.nbytes)
    else:
      d_pairs_u = cp.empty(1, dtype=cp.int32)
      d_pairs_v = cp.empty(1, dtype=cp.int32)
    self.h2d_bytes += int(customer_ids.nbytes + row_seeds.nbytes)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    operands, uploaded_bytes = self._operands.resolve(context, context.dimension, self.max_matrix_dimension)
    self.h2d_bytes += uploaded_bytes
    d_distances, d_coords, edge_type_code = operands

    need_ws = population_size * context.dimension
    self._ws_prev = ensure_gpu_workspace(self._ws_prev, (need_ws,), cp.int32)
    self._ws_next = ensure_gpu_workspace(self._ws_next, (need_ws,), cp.int32)
    self._ws_other = ensure_gpu_workspace(self._ws_other, (need_ws,), cp.int32)
    self._ws_load = ensure_gpu_workspace(self._ws_load, (need_ws,), cp.int64)
    self._ws_open_dist = ensure_gpu_workspace(self._ws_open_dist, (need_ws,), cp.int64)

    d_pop = cp.empty((population_size, row_width), dtype=cp.int32)
    constraints = context.constraints
    rcl_nonzero = _compute_rcl_width(self.noise)
    block = 128
    grid = (population_size + block - 1) // block
    kernel = self._get_kernel()
    kernel(
      (grid,),
      (block,),
      (
        d_customers,
        d_demands,
        d_distances,
        d_coords,
        d_pairs_u,
        d_pairs_v,
        d_seeds,
        self._ws_prev[:need_ws],
        self._ws_next[:need_ws],
        self._ws_other[:need_ws],
        self._ws_load[:need_ws],
        self._ws_open_dist[:need_ws],
        d_pop,
        np.int32(population_size),
        np.int32(n_cust),
        np.int32(n_pairs),
        np.int32(row_width),
        np.int32(context.dimension),
        np.int32(edge_type_code),
        np.int32(k_max),
        np.int32(rcl_nonzero),
        np.int32(1 if context.problem.is_symmetric else 0),
        np.int32(1 if self.output_format == "delimited" else 0),
        np.int32(1 if constraints.return_to_depot else 0),
        np.int32(1 if constraints.capacity is not None else 0),
        np.int64(constraints.capacity if constraints.capacity is not None else 0),
        np.int32(1 if constraints.max_route_distance is not None else 0),
        np.float64(constraints.max_route_distance if constraints.max_route_distance is not None else 0.0),
      ),
    )
    self.kernel_launches += 1

    if self.return_gpu:
      return cast("Routes", d_pop)
    host_pop = np.asarray(d_pop.get(), dtype=np.int32)
    self.d2h_bytes += int(host_pop.nbytes)
    return host_pop


__all__: list[str] = ["InitializationCPUClarkeWright", "InitializationGPUClarkeWright"]
