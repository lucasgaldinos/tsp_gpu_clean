"""GPU VRAM-resident VRP inter-route and intra-route local search (`LocalSearchGPUTwoOptStar`, `DP17-7`, `DP17-9`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.decoding import strip_depot_delimiters
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel, prev_power_of_2
from utils.phase_timer import time_strategy
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape

from .gpu_base import BaseLocalSearchGPU, ensure_gpu_workspace
from .two_opt_star_helpers import (
  _GPU_MODES,
  DEFAULT_VRP_LS_MAX_SHARED_DIMENSION,
  InterRouteGPUMode,
  VRPSplitMode,
  resolve_inter_route_policies,
)

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import RouteOrRoutes
  from protocols.vrp_strategies import PenaltyPolicy, RouteDecoder

_GPU_KERNELS: dict[str, tuple[str, str]] = {
  "batch": ("two_opt_star_steepest.cu", "two_opt_star_steepest_kernel"),
  "multi_move": ("two_opt_star_multi_move.cu", "two_opt_star_multi_move_kernel"),
}


def _vrp_ls_shared_mem_bytes(block_size: int, n_cust: int, dimension: int, *, use_shared: bool) -> int:
  """Return dynamic shared-memory bytes for `cvrp_inter_route_*_kernel`."""
  fixed_bytes = block_size * 8 + block_size * 6 * 4
  if not use_shared:
    return fixed_bytes
  i64_bytes = 5 * n_cust * 8
  int_bytes = (7 * n_cust + 2 * dimension) * 4
  return fixed_bytes + i64_bytes + int_bytes


class LocalSearchGPUTwoOptStar(BaseLocalSearchGPU):
  """VRAM-resident GPU VRP local search (`Relocate`, `Swap`, `2-opt*` + intra-route `2-opt`)."""

  def __init__(
    self,
    mode: InterRouteGPUMode = "multi_move",
    max_iterations: int = 50,
    candidate_k: int | None = 20,
    split_mode: VRPSplitMode = "prins",
    intra_two_opt: bool = True,
    max_matrix_dimension: int | None = None,
    threads_per_block: int = 128,
    max_shared_dimension: int = DEFAULT_VRP_LS_MAX_SHARED_DIMENSION,
    penalty_policy: PenaltyPolicy | None = None,
    decoder: RouteDecoder | None = None,
  ) -> None:
    if mode not in _GPU_MODES:
      raise ValueError(f"Invalid VRP GPU local search mode {mode!r}; expected one of {sorted(_GPU_MODES)}")
    if isinstance(threads_per_block, bool) or threads_per_block <= 0:
      raise ValueError(f"threads_per_block must be > 0, got {threads_per_block!r}")
    if isinstance(max_shared_dimension, bool) or max_shared_dimension < 0:
      raise ValueError(f"max_shared_dimension must be >= 0, got {max_shared_dimension!r}")
    super().__init__(
      threads_per_block=threads_per_block,
      max_matrix_dimension=max_matrix_dimension,
      candidate_k=candidate_k,
    )
    self.mode: InterRouteGPUMode = mode
    self.max_iterations: int = max_iterations
    self.split_mode: VRPSplitMode = split_mode
    self.intra_two_opt: bool = intra_two_opt
    self.max_shared_dimension: int = max_shared_dimension
    self.penalty_policy, self._decoder = resolve_inter_route_policies(
      split_mode=split_mode,
      max_iterations=max_iterations,
      candidate_k=candidate_k,
      max_matrix_dimension=max_matrix_dimension,
      penalty_policy=penalty_policy,
      decoder=decoder,
      use_gpu=True,
    )
    self._ws_int: Any = None
    self._ws_i64: Any = None
    self._ws_used: Any = None

  def reset_metrics(self) -> None:
    """Reset transfer/launch counters, workspaces, decoder, and penalty policy."""
    super().reset_metrics()
    self._ws_int = None
    self._ws_i64 = None
    self._ws_used = None
    for obj in (self._decoder, self.penalty_policy):
      hook = getattr(obj, "reset_metrics", None)
      if callable(hook):
        hook()

  def _get_kernel(self) -> Any:
    """Lazily compile and return the CUDA kernel selected by `mode`."""
    if self._kernel is None:
      source, entry_point = _GPU_KERNELS[self.mode]
      self._kernel = load_kernel(source, entry_point, __file__)
    return self._kernel

  def _ensure_workspace(
    self,
    batch_size: int,
    n_cust: int,
    dimension: int,
    *,
    need_route_buffers: bool,
    need_used_bytes: bool,
  ) -> tuple[Any, Any, Any]:
    """Allocate/reuse monotonic global-memory workspaces when shared-memory or 1024-route caps are exceeded."""
    cp = require_gpu()
    int_stride = 7 * n_cust + 2 * dimension
    i64_stride = 5 * n_cust
    ws_int: Any = 0
    ws_i64: Any = 0
    ws_used: Any = 0

    if need_route_buffers:
      need_int = batch_size * int_stride
      need_i64 = batch_size * i64_stride
      self._ws_int = ensure_gpu_workspace(self._ws_int, (need_int,), cp.int32)
      self._ws_i64 = ensure_gpu_workspace(self._ws_i64, (need_i64,), cp.int64)
      ws_int = self._ws_int[:need_int]
      ws_i64 = self._ws_i64[:need_i64]

    if need_used_bytes:
      need_used = batch_size * n_cust
      self._ws_used = ensure_gpu_workspace(self._ws_used, (need_used,), cp.uint8)
      ws_used = self._ws_used[:need_used]

    return ws_int, ws_i64, ws_used

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch CUDA kernels ahead of the measured evolutionary loop (`BUG-25`)."""
    require_gpu()
    super()._warmup_impl(context)
    cp = require_gpu()
    self._get_kernel()
    dec_warmup = getattr(self._decoder, "warmup", None)
    if callable(dec_warmup):
      dec_warmup(context)
    if context.dimension <= 2:
      return
    n_cust = context.dimension - 1
    probe = cp.zeros((1, context.dimension + 1), dtype=cp.int32)
    probe[0, 1 : n_cust + 1] = cp.arange(1, context.dimension, dtype=cp.int32)
    self.improve_batch(probe, context)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route or 2D batch on GPU with zero D2H traffic for CuPy inputs."""
    cp = require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    batch_size, width = batch.shape
    if context.dimension <= 2 or self.max_iterations <= 0 or batch_size == 0:
      return restore_routes_shape(batch.copy(), was_1d)

    input_on_gpu = is_gpu_array(batch)
    n_cust = context.dimension - 1
    is_giant_tour = width == n_cust

    if is_giant_tour:
      before_h2d = int(getattr(self._decoder, "h2d_bytes", 0))
      before_d2h = int(getattr(self._decoder, "d2h_bytes", 0))
      before_launches = int(getattr(self._decoder, "kernel_launches", 0))
      d_delimited = self._decoder.decode_batch(batch, context, return_gpu=True).routes
      self.h2d_bytes += max(0, int(getattr(self._decoder, "h2d_bytes", 0)) - before_h2d)
      self.d2h_bytes += max(0, int(getattr(self._decoder, "d2h_bytes", 0)) - before_d2h)
      self.kernel_launches += max(0, int(getattr(self._decoder, "kernel_launches", 0)) - before_launches)
    elif input_on_gpu:
      d_delimited = cp.ascontiguousarray(batch.copy(), dtype=cp.int32)
    else:
      batch_np = np.ascontiguousarray(batch, dtype=np.int32)
      d_delimited = cp.asarray(batch_np, dtype=cp.int32)
      self.h2d_bytes += int(batch_np.nbytes)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    d_distances, d_coords, edge_type_code = self._resolve_distance_operands(context, context.dimension)
    cand_k = 0
    d_candidates: Any = 0
    if self.candidate_k is not None:
      d_candidates = self._resolve_candidate_table(context, self.candidate_k)
      cand_k = int(d_candidates.shape[1])

    delim_width = int(d_delimited.shape[1])
    block_size = prev_power_of_2(min(self.threads_per_block, max(1, n_cust)), min_val=1, max_val=256)
    use_shared = context.dimension <= self.max_shared_dimension
    shared_mem = _vrp_ls_shared_mem_bytes(block_size, n_cust, context.dimension, use_shared=use_shared)
    need_used_bytes = self.mode == "multi_move" and n_cust > 1024
    ws_int, ws_i64, ws_used = self._ensure_workspace(
      batch_size,
      n_cust,
      context.dimension,
      need_route_buffers=not use_shared,
      need_used_bytes=need_used_bytes,
    )

    constraints = context.constraints
    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)
    kernel = self._get_kernel()
    args: list[Any] = [
      d_delimited,
      d_demands,
      d_distances,
      d_coords,
      d_candidates,
      np.int32(delim_width),
      np.int32(n_cust),
      np.int32(context.dimension),
      np.int32(batch_size),
      np.int32(edge_type_code),
      np.int32(cand_k),
      np.int32(self.max_iterations),
      np.int32(1 if self.intra_two_opt else 0),
      np.int32(1 if constraints.return_to_depot else 0),
      np.int32(1 if constraints.capacity is not None else 0),
      np.int64(constraints.capacity if constraints.capacity is not None else 0),
      np.int32(1 if constraints.max_route_distance is not None else 0),
      np.float64(constraints.max_route_distance if constraints.max_route_distance is not None else 0.0),
      np.int32(1 if constraints.max_vehicles is not None else 0),
      np.int32(constraints.max_vehicles if constraints.max_vehicles is not None else 0),
      np.float64(omega_q if math.isfinite(omega_q) else math.inf),
      np.float64(omega_l if math.isfinite(omega_l) else math.inf),
      np.float64(omega_k if math.isfinite(omega_k) else math.inf),
      np.int32(1 if use_shared else 0),
      ws_int,
      ws_i64,
    ]
    if self.mode == "multi_move":
      args.append(ws_used)

    kernel((batch_size,), (block_size,), tuple(args), shared_mem=shared_mem)
    self.kernel_launches += 1

    if is_giant_tour:
      d_out = strip_depot_delimiters(d_delimited, n_cust)
      if batch_size > 0 and n_cust > 0:
        self.kernel_launches += 1
    else:
      d_out = d_delimited

    if input_on_gpu:
      return restore_routes_shape(d_out, was_1d)
    out_host = d_out.get()
    self.d2h_bytes += int(out_host.nbytes)
    return restore_routes_shape(out_host, was_1d)


__all__: list[str] = ["LocalSearchGPUTwoOptStar"]
