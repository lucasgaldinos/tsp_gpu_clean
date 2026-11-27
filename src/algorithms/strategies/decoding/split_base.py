"""Shared CPU/GPU base decoder, prefix-table builder, and GPU launcher (`Task 17.21`, `DP17-7`, `DP17-9`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.penalties import InstanceScaledPenaltyPolicy
from protocols.backend import require_gpu
from protocols.vrp_strategies import PenaltyPolicy
from utils.gpu_helpers import load_kernel
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape, to_cpu_batch

from .delimiter_ops import (
  SplitBatchResult,
  build_cpu_prefix_tables,
  score_and_materialize_cpu_batch,
)

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import RouteOrRoutes

#: Default shared-memory customer ceiling respecting Pascal `sm_61` 48 KB limit (~32 KB at 1024).
DEFAULT_MAX_SHARED_CUSTOMERS: int = 1024

#: Maximum dynamic shared memory per block (44 KB leaves 4 KB headroom under `sm_61`'s 48 KB cap).
_MAX_SHARED_MEM_BYTES: int = 44 * 1024

#: Warp block size for `vrp_split_kernel` (1 full warp per Giant-Tour).
_SPLIT_BLOCK_SIZE: int = 32


def _split_shared_mem_bytes(n_cust: int) -> int:
  """Return dynamic shared-memory bytes for `V, W, P, pred, d0` (`32 * n_cust + 28`)."""
  return 32 * n_cust + 28


class _BaseSplitDecoder(StrategyBase):
  """Shared CPU/CUDA Giant-Tour -> `0`-delimited route decoder (`DP17-2`, `DP17-3`, `DP17-9`)."""

  _mode_prins: ClassVar[bool] = True

  def __init__(
    self,
    *,
    use_gpu: bool | None = None,
    return_gpu: bool | None = None,
    max_vehicles: int | None = None,
    max_shared_customers: int = DEFAULT_MAX_SHARED_CUSTOMERS,
    max_matrix_dimension: int | None = None,
    penalty_policy: PenaltyPolicy | None = None,
  ) -> None:
    super().__init__()
    if max_vehicles is not None and (isinstance(max_vehicles, bool) or max_vehicles <= 0):
      raise ValueError(f"max_vehicles must be > 0 when provided, got {max_vehicles!r}")
    if isinstance(max_shared_customers, bool) or max_shared_customers < 0:
      raise ValueError(f"max_shared_customers must be >= 0, got {max_shared_customers!r}")
    if penalty_policy is not None and not isinstance(penalty_policy, PenaltyPolicy):
      raise TypeError(f"penalty_policy must implement PenaltyPolicy, got {type(penalty_policy).__name__}")

    self.use_gpu: bool | None = use_gpu
    self.return_gpu: bool | None = return_gpu
    self.max_vehicles: int | None = max_vehicles
    self.max_shared_customers: int = max_shared_customers
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.penalty_policy: PenaltyPolicy = penalty_policy if penalty_policy is not None else InstanceScaledPenaltyPolicy()

    self._kernel: Any = None
    self._strip_kernel: Any = None
    self._ws_v: Any = None
    self._ws_w: Any = None
    self._ws_p: Any = None
    self._ws_pred: Any = None
    self._ws_d0: Any = None
    self._ws_shape: tuple[int, int] = (0, 0)

  def reset_metrics(self) -> None:
    """Reset telemetry counters, dynamic penalty state, and cached GPU workspaces."""
    super().reset_metrics()
    reset_hook = getattr(self.penalty_policy, "reset_metrics", None)
    if callable(reset_hook):
      reset_hook()
    self._ws_v = None
    self._ws_w = None
    self._ws_p = None
    self._ws_pred = None
    self._ws_d0 = None
    self._ws_shape = (0, 0)

  def _get_kernel(self) -> Any:
    """Lazily compile and cache `vrp_split_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel("vrp_split.cu", "vrp_split_kernel", __file__)
    return self._kernel

  def _get_strip_kernel(self) -> Any:
    """Lazily compile and cache `strip_depot_delimiters_kernel`."""
    if self._strip_kernel is None:
      self._strip_kernel = load_kernel("vrp_split.cu", "strip_depot_delimiters_kernel", __file__)
    return self._strip_kernel

  def _ensure_global_workspace(self, batch_size: int, n_cust: int) -> tuple[Any, Any, Any, Any, Any]:
    """Return cached `(d_V, d_W, d_P, d_pred, d_d0)` global buffers for `use_shared == 0`."""
    cp = require_gpu()
    state_len = n_cust + 1
    cust_len = max(1, n_cust)
    if self._ws_v is None or self._ws_shape[0] < batch_size or self._ws_shape[1] < state_len:
      rows = max(batch_size, self._ws_shape[0])
      cols = max(state_len, self._ws_shape[1])
      self._ws_v = cp.empty((rows, cols), dtype=cp.float64)
      self._ws_w = cp.empty((rows, cols), dtype=cp.int64)
      self._ws_p = cp.empty((rows, cols), dtype=cp.int64)
      self._ws_pred = cp.empty((rows, cols), dtype=cp.int32)
      self._ws_d0 = cp.empty((rows, max(1, cols - 1)), dtype=cp.int32)
      self._ws_shape = (rows, cols)
    return (
      self._ws_v[:batch_size, :state_len],
      self._ws_w[:batch_size, :state_len],
      self._ws_p[:batch_size, :state_len],
      self._ws_pred[:batch_size, :state_len],
      self._ws_d0[:batch_size, :cust_len],
    )

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile CUDA Split/Strip kernels and launch once on a valid Giant-Tour (`BUG-25`)."""
    super()._warmup_impl(context)
    run_gpu = self.use_gpu if self.use_gpu is not None else hasattr(context.xp, "RawKernel")
    if not run_gpu:
      return
    self._get_strip_kernel()
    n_cust = max(0, context.dimension - 1)
    if n_cust <= 0:
      return
    cp = require_gpu()
    warmup_tour = cp.arange(1, context.dimension, dtype=cp.int32).reshape(1, n_cust)
    self.decode_batch(warmup_tour, context, return_gpu=True)

  def decode_batch(
    self,
    giant_tours: RouteOrRoutes,
    context: ProblemContext,
    *,
    max_vehicles: int | None = None,
    return_gpu: bool | None = None,
  ) -> SplitBatchResult:
    """Partition `(N, n_cust)` or `(n_cust,)` Giant-Tours into `0`-delimited routes and scores."""
    return self._decode_impl(
      giant_tours, context, max_vehicles=max_vehicles, return_gpu=return_gpu, materialize_routes=True
    )

  def decode_costs_only(
    self,
    giant_tours: RouteOrRoutes,
    context: ProblemContext,
    *,
    max_vehicles: int | None = None,
    return_gpu: bool | None = None,
  ) -> SplitBatchResult:
    """Compute `(penalized_costs, unpenalized_costs, violations, num_vehicles)` without writing routes."""
    return self._decode_impl(
      giant_tours, context, max_vehicles=max_vehicles, return_gpu=return_gpu, materialize_routes=False
    )

  def _compute_cpu_predecessors(
    self,
    W: np.ndarray,
    P: np.ndarray,
    d0: np.ndarray,
    dr: np.ndarray,
    *,
    capacity: int | None,
    max_route_distance: float | None,
    omega_q: float,
    omega_l: float,
  ) -> np.ndarray:
    """Compute `(N, n_cust + 1)` predecessor matrix on CPU; overridden by Prins/Greedy subclasses."""
    raise NotImplementedError

  def _resolve_buffer_k(self, context: ProblemContext, n_cust: int, max_vehicles: int | None) -> int:
    req = max_vehicles if max_vehicles is not None else self.max_vehicles
    if req is not None:
      if isinstance(req, bool) or req <= 0:
        raise ValueError(f"max_vehicles must be > 0, got {req!r}")
      return req
    if context.constraints.max_vehicles is not None:
      return context.constraints.max_vehicles
    return max(1, n_cust)

  def _decode_impl(
    self,
    giant_tours: RouteOrRoutes,
    context: ProblemContext,
    *,
    max_vehicles: int | None,
    return_gpu: bool | None,
    materialize_routes: bool,
  ) -> SplitBatchResult:
    batch, was_1d = ensure_routes_batch(giant_tours)
    _, n_cust = batch.shape
    expected_cust = context.dimension - 1
    if n_cust != expected_cust:
      raise ValueError(f"Giant-Tour width must be dimension - 1 ({expected_cust}), got {n_cust}")

    buffer_k = self._resolve_buffer_k(context, n_cust, max_vehicles)
    input_on_gpu = is_gpu_array(batch)
    out_on_gpu = (
      return_gpu if return_gpu is not None else (self.return_gpu if self.return_gpu is not None else input_on_gpu)
    )
    run_gpu = self.use_gpu if self.use_gpu is not None else (input_on_gpu or hasattr(context.xp, "RawKernel"))
    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)

    if run_gpu:
      return self._decode_gpu(
        batch,
        was_1d=was_1d,
        input_on_gpu=input_on_gpu,
        out_on_gpu=out_on_gpu,
        context=context,
        buffer_k=buffer_k,
        materialize_routes=materialize_routes,
        omega_q=omega_q,
        omega_l=omega_l,
        omega_k=omega_k,
      )
    return self._decode_cpu(
      batch,
      was_1d=was_1d,
      out_on_gpu=out_on_gpu,
      context=context,
      buffer_k=buffer_k,
      materialize_routes=materialize_routes,
      omega_q=omega_q,
      omega_l=omega_l,
      omega_k=omega_k,
    )

  def _decode_cpu(
    self,
    batch: Any,
    *,
    was_1d: bool,
    out_on_gpu: bool,
    context: ProblemContext,
    buffer_k: int,
    materialize_routes: bool,
    omega_q: float,
    omega_l: float,
    omega_k: float,
  ) -> SplitBatchResult:
    batch_np, _, _ = to_cpu_batch(batch)
    distances = context.get_cpu_distances()
    demands = context.get_cpu_demands()
    if demands is None:
      demands = np.zeros(context.dimension, dtype=np.int32)

    constraints = context.constraints
    W, P, d0, dr = build_cpu_prefix_tables(batch_np, demands, distances, return_to_depot=constraints.return_to_depot)
    pred_batch = self._compute_cpu_predecessors(
      W,
      P,
      d0,
      dr,
      capacity=constraints.capacity,
      max_route_distance=constraints.max_route_distance,
      omega_q=omega_q,
      omega_l=omega_l,
    )
    routes_np, penalized, unpenalized, violations, num_vehicles = score_and_materialize_cpu_batch(
      batch_np,
      pred_batch,
      W,
      P,
      d0,
      dr,
      capacity=constraints.capacity,
      max_route_distance=constraints.max_route_distance,
      max_vehicles_constraint=constraints.max_vehicles,
      buffer_k=buffer_k,
      omega_q=omega_q,
      omega_l=omega_l,
      omega_k=omega_k,
      materialize_routes=materialize_routes,
    )

    routes_final: Any = restore_routes_shape(routes_np, was_1d, to_gpu=out_on_gpu) if routes_np is not None else None
    if out_on_gpu:
      cp = require_gpu()
      return SplitBatchResult(
        routes=routes_final,
        penalized_costs=cp.asarray(penalized, dtype=cp.float64),
        unpenalized_costs=cp.asarray(unpenalized, dtype=cp.float64),
        violations=cp.asarray(violations, dtype=cp.float64),
        num_vehicles=cp.asarray(num_vehicles, dtype=cp.int32),
      )
    return SplitBatchResult(
      routes=routes_final,
      penalized_costs=penalized,
      unpenalized_costs=unpenalized,
      violations=violations,
      num_vehicles=num_vehicles,
    )

  def _decode_gpu(
    self,
    batch: Any,
    *,
    was_1d: bool,
    input_on_gpu: bool,
    out_on_gpu: bool,
    context: ProblemContext,
    buffer_k: int,
    materialize_routes: bool,
    omega_q: float,
    omega_l: float,
    omega_k: float,
  ) -> SplitBatchResult:
    cp = require_gpu()
    batch_size, n_cust = batch.shape

    if input_on_gpu:
      d_tours = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      tours_np = np.ascontiguousarray(batch, dtype=np.int32)
      d_tours = cp.asarray(tours_np, dtype=cp.int32)
      self.h2d_bytes += int(tours_np.nbytes)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(
      context.dimension, self.max_matrix_dimension
    )
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)

    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    row_width = context.dimension + buffer_k
    d_routes: Any = cp.empty((batch_size, row_width), dtype=cp.int32) if materialize_routes else 0
    d_penalized = cp.empty(batch_size, dtype=cp.float64)
    d_unpenalized = cp.empty(batch_size, dtype=cp.float64)
    d_violations = cp.empty(batch_size, dtype=cp.float64)
    d_num_vehicles = cp.empty(batch_size, dtype=cp.int32)

    if batch_size > 0:
      req_smem = _split_shared_mem_bytes(n_cust)
      use_shared = n_cust <= self.max_shared_customers and req_smem <= _MAX_SHARED_MEM_BYTES
      shared_mem = req_smem if use_shared else 0
      if use_shared:
        d_ws_v: Any = 0
        d_ws_w: Any = 0
        d_ws_p: Any = 0
        d_ws_pred: Any = 0
        d_ws_d0: Any = 0
      else:
        d_ws_v, d_ws_w, d_ws_p, d_ws_pred, d_ws_d0 = self._ensure_global_workspace(batch_size, n_cust)

      constraints = context.constraints
      kernel = self._get_kernel()
      kernel(
        (batch_size,),
        (_SPLIT_BLOCK_SIZE,),
        (
          d_tours,
          d_demands,
          d_distances,
          d_coords,
          np.int32(n_cust),
          np.int32(context.dimension),
          np.int32(batch_size),
          np.int32(edge_type_code),
          np.int32(1 if self._mode_prins else 0),
          np.int32(1 if constraints.return_to_depot else 0),
          np.int32(1 if constraints.capacity is not None else 0),
          np.int64(constraints.capacity if constraints.capacity is not None else 0),
          np.int32(1 if constraints.max_route_distance is not None else 0),
          np.float64(constraints.max_route_distance if constraints.max_route_distance is not None else 0.0),
          np.int32(1 if constraints.max_vehicles is not None else 0),
          np.int32(constraints.max_vehicles if constraints.max_vehicles is not None else 0),
          np.int32(buffer_k),
          np.float64(omega_q if math.isfinite(omega_q) else math.inf),
          np.float64(omega_l if math.isfinite(omega_l) else math.inf),
          np.float64(omega_k if math.isfinite(omega_k) else math.inf),
          np.int32(1 if use_shared else 0),
          d_ws_v,
          d_ws_w,
          d_ws_p,
          d_ws_pred,
          d_ws_d0,
          d_routes,
          d_penalized,
          d_unpenalized,
          d_violations,
          d_num_vehicles,
        ),
        shared_mem=shared_mem,
      )
      self.kernel_launches += 1

    if out_on_gpu:
      routes_out = restore_routes_shape(d_routes, was_1d) if materialize_routes else None
      return SplitBatchResult(
        routes=routes_out,
        penalized_costs=d_penalized,
        unpenalized_costs=d_unpenalized,
        violations=d_violations,
        num_vehicles=d_num_vehicles,
      )

    pen_host = d_penalized.get()
    unpen_host = d_unpenalized.get()
    viol_host = d_violations.get()
    veh_host = d_num_vehicles.get()
    self.d2h_bytes += int(pen_host.nbytes + unpen_host.nbytes + viol_host.nbytes + veh_host.nbytes)
    routes_host: Any = None
    if materialize_routes:
      routes_2d = d_routes.get()
      self.d2h_bytes += int(routes_2d.nbytes)
      routes_host = restore_routes_shape(routes_2d, was_1d)
    return SplitBatchResult(
      routes=routes_host,
      penalized_costs=pen_host,
      unpenalized_costs=unpen_host,
      violations=viol_host,
      num_vehicles=veh_host,
    )


__all__: list[str] = [
  "DEFAULT_MAX_SHARED_CUSTOMERS",
  "_BaseSplitDecoder",
  "build_cpu_prefix_tables",
]
