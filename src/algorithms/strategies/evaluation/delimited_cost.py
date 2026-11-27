"""0-delimited VRP cost evaluation helpers (CPU NumPy + GPU kernel launcher) (`DP17-1`, `DP17-7`, `DP17-9`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from algorithms.strategies.decoding import (
  GreedySplitDecoder,
  PrinsSplitDecoder,
  compact_delimited_route,
)
from algorithms.strategies.penalties import InstanceScaledPenaltyPolicy
from protocols.backend import require_gpu
from protocols.vrp_strategies import (
  PenaltyPolicy,
  RouteAssessment,
  RouteDecoder,
  canonicalize_default_route,
  compute_normalized_violations,
)
from utils.route_helpers import restore_routes_shape

if TYPE_CHECKING:
  from algorithms.strategies.decoding import SplitBatchResult
  from protocols.problem_context import ProblemContext
  from protocols.types import Route

_EVAL_BLOCK_SIZE: int = 128
_WARPS_PER_BLOCK: int = _EVAL_BLOCK_SIZE // 32


def validate_split_mode(split_mode: str) -> Literal["prins", "greedy"]:
  """Validate `split_mode` literal."""
  if split_mode not in ("prins", "greedy"):
    raise ValueError(f"split_mode must be 'prins' or 'greedy', got {split_mode!r}")
  return cast("Literal['prins', 'greedy']", split_mode)


def _apply_safe_penalty(excess: np.ndarray, omega: float) -> np.ndarray:
  """Multiply positive violations by `omega` without producing `0.0 * inf = NaN`."""
  positive = excess > 0
  if math.isfinite(omega):
    return np.where(positive, omega * excess.astype(np.float64), 0.0)
  return np.where(positive, np.inf, 0.0)


def evaluate_delimited_cpu(
  batch_np: np.ndarray,
  context: ProblemContext,
  *,
  omega_q: float,
  omega_l: float,
  omega_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Evaluate `(N, W)` `0`-delimited routes on CPU in pure vectorized NumPy (`BUG-116`)."""
  batch = np.ascontiguousarray(batch_np, dtype=np.intp)
  batch_size, width = batch.shape
  if batch_size == 0 or width == 0:
    zeros_f = np.zeros(batch_size, dtype=np.float64)
    return (
      zeros_f,
      zeros_f.copy(),
      zeros_f.copy(),
      np.zeros(batch_size, dtype=np.int32),
      np.zeros((batch_size, 3), dtype=np.float64),
    )

  distances = context.get_cpu_distances()
  demands = context.get_cpu_demands()
  if demands is None:
    demands = np.zeros(context.dimension, dtype=np.int32)

  constraints = context.constraints
  return_to_depot = constraints.return_to_depot
  capacity = constraints.capacity
  max_route_distance = constraints.max_route_distance
  max_vehicles = constraints.max_vehicles

  nz = batch != 0
  prev = np.empty_like(batch)
  prev[:, 0] = 0
  prev[:, 1:] = batch[:, :-1]

  is_start = nz & (prev == 0)
  active_vehicles = np.sum(is_start, axis=1, dtype=np.int32)

  cell_load = np.where(nz, demands[batch], 0).astype(np.int64, copy=False)
  cell_dist = np.where(nz, distances[prev, batch], 0).astype(np.int64, copy=False)
  if return_to_depot:
    nxt = np.empty_like(batch)
    nxt[:, :-1] = batch[:, 1:]
    nxt[:, -1] = 0
    is_end = nz & (nxt == 0)
    cell_dist = cell_dist + np.where(is_end, distances[batch, 0], 0).astype(np.int64, copy=False)

  total_dist = np.sum(cell_dist, axis=1, dtype=np.int64)
  total_cap_viol = np.zeros(batch_size, dtype=np.int64)
  total_dist_viol = np.zeros(batch_size, dtype=np.float64)

  if capacity is not None or max_route_distance is not None:
    flat_nz = nz.ravel()
    if bool(np.any(flat_nz)):
      seg_starts = np.flatnonzero(is_start.ravel()[flat_nz])
      row_ids = np.repeat(np.arange(batch_size, dtype=np.intp), active_vehicles)
      if capacity is not None:
        route_loads = np.add.reduceat(cell_load.ravel()[flat_nz], seg_starts)
        cap_exc = np.maximum(0, route_loads - int(capacity))
        np.add.at(total_cap_viol, row_ids, cap_exc)
      if max_route_distance is not None:
        route_dists = np.add.reduceat(cell_dist.ravel()[flat_nz], seg_starts)
        dist_exc = np.maximum(0.0, route_dists.astype(np.float64) - float(max_route_distance))
        np.add.at(total_dist_viol, row_ids, dist_exc)

  veh_viol = (
    np.maximum(0, active_vehicles.astype(np.int64) - int(max_vehicles))
    if max_vehicles is not None
    else np.zeros(batch_size, dtype=np.int64)
  )
  cap_viol_f = total_cap_viol.astype(np.float64)
  veh_viol_f = veh_viol.astype(np.float64)

  unpenalized = total_dist.astype(np.float64)
  violations = cap_viol_f + total_dist_viol + veh_viol_f
  penalized = (
    unpenalized
    + _apply_safe_penalty(total_cap_viol, omega_q)
    + _apply_safe_penalty(total_dist_viol, omega_l)
    + _apply_safe_penalty(veh_viol, omega_k)
  )
  components = np.ascontiguousarray(
    np.column_stack((cap_viol_f, total_dist_viol, veh_viol_f)),
    dtype=np.float64,
  )
  return penalized, unpenalized, violations, active_vehicles, components


def launch_delimited_cost_gpu(
  kernel: Any,
  d_routes: Any,
  d_demands: Any,
  d_distances: Any,
  d_coords: Any,
  context: ProblemContext,
  *,
  edge_type_code: int,
  omega_q: float,
  omega_l: float,
  omega_k: float,
  need_components: bool = False,
) -> tuple[Any, Any, Any, Any, Any, bool]:
  """Launch warp-coalesced `vrp_delimited_cost_kernel` on `(N, W)` `int32` device routes."""
  cp = require_gpu()
  batch_size, width = d_routes.shape
  d_penalized = cp.empty(batch_size, dtype=cp.float64)
  d_unpenalized = cp.empty(batch_size, dtype=cp.float64)
  d_violations = cp.empty(batch_size, dtype=cp.float64)
  d_num_vehicles = cp.empty(batch_size, dtype=cp.int32)
  d_components: Any = cp.empty((batch_size, 3), dtype=cp.float64) if need_components else 0

  launched = False
  if batch_size > 0:
    constraints = context.constraints
    blocks = (batch_size + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
    kernel(
      (blocks,),
      (_EVAL_BLOCK_SIZE,),
      (
        d_routes,
        d_demands,
        d_distances,
        d_coords,
        np.int32(batch_size),
        np.int32(width),
        np.int32(context.dimension),
        np.int32(edge_type_code),
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
        d_penalized,
        d_unpenalized,
        d_violations,
        d_num_vehicles,
        d_components,
      ),
    )
    launched = True

  return d_penalized, d_unpenalized, d_violations, d_num_vehicles, d_components, launched


class UnifiedCostMixin:
  """Shared decoder/penalty policy wiring, VRP width check, and polymorphic canonicalization."""

  split_mode: Literal["prins", "greedy"]
  penalty_policy: PenaltyPolicy
  _decoder: RouteDecoder
  h2d_bytes: int
  d2h_bytes: int
  kernel_launches: int

  def _init_vrp_components(
    self,
    *,
    decoder: RouteDecoder | None,
    split_mode: Literal["prins", "greedy"],
    penalty_policy: PenaltyPolicy | None,
    max_matrix_dimension: int | None,
    use_gpu: bool,
  ) -> None:
    self.split_mode = validate_split_mode(split_mode)
    if penalty_policy is not None and not isinstance(penalty_policy, PenaltyPolicy):
      raise TypeError(f"penalty_policy must implement PenaltyPolicy, got {type(penalty_policy).__name__}")
    if decoder is not None and not isinstance(decoder, RouteDecoder):
      raise TypeError(f"decoder must implement RouteDecoder, got {type(decoder).__name__}")

    if penalty_policy is not None:
      self.penalty_policy = penalty_policy
    else:
      decoder_policy = getattr(decoder, "penalty_policy", None) if decoder is not None else None
      self.penalty_policy = (
        decoder_policy if isinstance(decoder_policy, PenaltyPolicy) else InstanceScaledPenaltyPolicy()
      )

    if decoder is not None:
      self._decoder = decoder
    else:
      decoder_cls = PrinsSplitDecoder if self.split_mode == "prins" else GreedySplitDecoder
      self._decoder = decoder_cls(
        use_gpu=use_gpu,
        max_matrix_dimension=max_matrix_dimension,
        penalty_policy=self.penalty_policy,
      )

  def _decoder_metrics_snapshot(self) -> tuple[int, int, int]:
    return (
      int(getattr(self._decoder, "h2d_bytes", 0)),
      int(getattr(self._decoder, "d2h_bytes", 0)),
      int(getattr(self._decoder, "kernel_launches", 0)),
    )

  def _absorb_decoder_metrics(self, before: tuple[int, int, int]) -> None:
    """Propagate telemetry deltas recorded by the internal Split decoder."""
    before_h2d, before_d2h, before_launches = before
    self.h2d_bytes += max(0, int(getattr(self._decoder, "h2d_bytes", 0)) - before_h2d)
    self.d2h_bytes += max(0, int(getattr(self._decoder, "d2h_bytes", 0)) - before_d2h)
    self.kernel_launches += max(0, int(getattr(self._decoder, "kernel_launches", 0)) - before_launches)

  def _reset_vrp_components(self) -> None:
    for obj in (self._decoder, self.penalty_policy):
      hook = getattr(obj, "reset_metrics", None)
      if callable(hook):
        hook()

  @staticmethod
  def _check_vrp_width(width: int, dimension: int) -> bool:
    """Return `True` for Giant-Tour `(dimension - 1)`, `False` for `0`-delimited `(>= dimension)`."""
    if width == dimension - 1:
      return True
    if width >= dimension:
      return False
    raise ValueError(
      f"Route width {width} is incompatible with problem dimension {dimension} "
      f"(expected {dimension - 1} for Giant-Tour or >= {dimension} for 0-delimited routes)"
    )

  def canonicalize_route(self, route: Route, context: ProblemContext) -> Route:
    """Canonicalize a winning route: passthrough for TSP/ATSP, Split+compact for VRP."""
    route_1d = canonicalize_default_route(route)
    if not context.problem.is_vrp:
      return route_1d
    if route_1d.size == context.dimension - 1:
      split_res = self._decoder.decode_batch(route_1d, context, return_gpu=False)
      return compact_delimited_route(split_res.routes)
    if route_1d.size >= context.dimension:
      return compact_delimited_route(route_1d)
    return route_1d


def build_cpu_vrp_detailed(
  batch_np: np.ndarray,
  pen: np.ndarray,
  unpen: np.ndarray,
  viol: np.ndarray,
  num_veh: np.ndarray,
  *,
  was_1d: bool,
  return_gpu: bool,
) -> SplitBatchResult:
  """Package CPU delimited evaluation arrays into `SplitBatchResult` on CPU or GPU."""
  from algorithms.strategies.decoding import SplitBatchResult

  routes_out: Any = restore_routes_shape(batch_np, was_1d, to_gpu=return_gpu)
  if return_gpu:
    cp = require_gpu()
    return SplitBatchResult(
      routes=routes_out,
      penalized_costs=cp.asarray(pen, dtype=cp.float64),
      unpenalized_costs=cp.asarray(unpen, dtype=cp.float64),
      violations=cp.asarray(viol, dtype=cp.float64),
      num_vehicles=cp.asarray(num_veh, dtype=cp.int32),
    )
  return SplitBatchResult(
    routes=routes_out,
    penalized_costs=pen,
    unpenalized_costs=unpen,
    violations=viol,
    num_vehicles=num_veh,
  )


def build_cpu_vrp_assessment(
  pen: np.ndarray,
  unpen: np.ndarray,
  viol: np.ndarray,
  num_veh: np.ndarray,
  comps: np.ndarray,
  context: ProblemContext,
  *,
  return_gpu: bool,
) -> RouteAssessment:
  """Package CPU VRP metrics into `RouteAssessment` on CPU or GPU."""
  is_feas = viol <= 1e-9
  norm_viol = compute_normalized_violations(comps, context)
  if return_gpu:
    cp = require_gpu()
    return RouteAssessment(
      objectives=cp.asarray(pen, dtype=cp.float64),
      travel_costs=cp.asarray(unpen, dtype=cp.float64),
      is_feasible=cp.asarray(is_feas, dtype=bool),
      violations=cp.asarray(viol, dtype=cp.float64),
      normalized_violations=cp.asarray(norm_viol, dtype=cp.float64),
      raw_violation_components=cp.asarray(comps, dtype=cp.float64),
      num_vehicles=cp.asarray(num_veh, dtype=cp.int32),
    )
  return RouteAssessment(
    objectives=pen,
    travel_costs=unpen,
    is_feasible=is_feas,
    violations=viol,
    normalized_violations=norm_viol,
    raw_violation_components=comps,
    num_vehicles=num_veh,
  )


def build_gpu_vrp_assessment(
  d_pen: Any,
  d_unpen: Any,
  d_viol: Any,
  d_veh: Any,
  d_comps: Any,
  context: ProblemContext,
  *,
  return_gpu: bool,
) -> tuple[RouteAssessment, int]:
  """Package GPU VRP arrays into `RouteAssessment`, returning `(assessment, d2h_bytes)`."""
  cp = require_gpu()
  d_feas = d_viol <= 1e-9
  d_norm = compute_normalized_violations(d_comps, context)
  if return_gpu:
    return (
      RouteAssessment(
        objectives=d_pen,
        travel_costs=d_unpen,
        is_feasible=d_feas,
        violations=d_viol,
        normalized_violations=d_norm,
        raw_violation_components=d_comps,
        num_vehicles=d_veh,
      ),
      0,
    )

  pen_h, unpen_h, viol_h = d_pen.get(), d_unpen.get(), d_viol.get()
  norm_h, comps_h, veh_h = d_norm.get(), d_comps.get(), d_veh.get()
  feas_h = cp.asnumpy(d_feas)
  d2h = int(
    pen_h.nbytes + unpen_h.nbytes + viol_h.nbytes + norm_h.nbytes + comps_h.nbytes + veh_h.nbytes + feas_h.nbytes
  )
  return (
    RouteAssessment(
      objectives=pen_h,
      travel_costs=unpen_h,
      is_feasible=feas_h,
      violations=viol_h,
      normalized_violations=norm_h,
      raw_violation_components=comps_h,
      num_vehicles=veh_h,
    ),
    d2h,
  )


__all__: list[str] = [
  "UnifiedCostMixin",
  "build_cpu_vrp_assessment",
  "build_cpu_vrp_detailed",
  "build_gpu_vrp_assessment",
  "evaluate_delimited_cpu",
  "launch_delimited_cost_gpu",
  "validate_split_mode",
]
