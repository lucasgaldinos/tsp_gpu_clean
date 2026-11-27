"""Zero-delimiter route conversion helpers and `SplitBatchResult` (`DP17-2`, `DP17-3`, `Task 17.21`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape


@dataclass(frozen=True, slots=True)
class SplitBatchResult:
  """Structured output returned by `PrinsSplitDecoder` and `GreedySplitDecoder`.

  Attributes:
      routes: `0`-delimited `int32` routes of shape `(N, W)` (or `(W,)` if the input was
          1D), where `W = context.dimension + buffer_k`; `None` on `decode_costs_only`.
      penalized_costs: `float64` array `(N,)` with `unpenalized_costs + weighted violations`.
      unpenalized_costs: `float64` array `(N,)` with exact travel costs of the routes.
      violations: `float64` array `(N,)` summing capacity, route-distance, and fleet excesses.
      num_vehicles: `int32` array `(N,)` reporting active non-empty routes per individual.
  """

  routes: Any
  penalized_costs: Any
  unpenalized_costs: Any
  violations: Any
  num_vehicles: Any


def compact_delimited_route(route_1d: Any) -> np.ndarray:
  """Collapse consecutive/trailing depot `0`s into canonical `[0, r_1..., 0, ..., 0]` in vectorized NumPy."""
  if is_gpu_array(route_1d):
    cp = require_gpu()
    arr = np.asarray(cp.asnumpy(route_1d), dtype=np.int32).ravel()
  else:
    arr = np.asarray(route_1d, dtype=np.int32).ravel()

  if arr.size == 0:
    return np.empty(0, dtype=np.int32)

  nonzero = arr != 0
  if not bool(np.any(nonzero)):
    return np.array([0], dtype=np.int32)

  keep = nonzero.copy()
  keep[1:] |= (~nonzero[1:]) & nonzero[:-1]
  compacted = arr[keep]
  if int(compacted[0]) != 0:
    compacted = np.concatenate((np.zeros(1, dtype=np.int32), compacted))
  if int(compacted[-1]) != 0:
    compacted = np.concatenate((compacted, np.zeros(1, dtype=np.int32)))
  return np.ascontiguousarray(compacted, dtype=np.int32)


def strip_depot_delimiters(delimited_routes: Any, n_customers: int) -> Any:
  """Strip depot `0` delimiters from `(N, W)` or `(W,)` routes into `(N, n_customers)`.

  Preserves the input array backend (`np.ndarray` or `cp.ndarray`) and 1D/2D shape.
  On CuPy inputs, executes via `strip_depot_delimiters_kernel` with zero D2H syncs.
  """
  if n_customers < 0:
    raise ValueError(f"n_customers must be >= 0, got {n_customers}")

  batch, was_1d = ensure_routes_batch(delimited_routes)
  batch_size, width = batch.shape

  if is_gpu_array(batch):
    cp = require_gpu()
    d_in = cp.ascontiguousarray(batch, dtype=cp.int32)
    d_out = cp.empty((batch_size, n_customers), dtype=cp.int32)
    if batch_size > 0 and n_customers > 0:
      kernel = load_kernel("vrp_split.cu", "strip_depot_delimiters_kernel", __file__)
      threads = 128
      blocks = (batch_size + threads - 1) // threads
      kernel(
        (blocks,),
        (threads,),
        (d_in, d_out, np.int32(batch_size), np.int32(width), np.int32(n_customers)),
      )
    return restore_routes_shape(d_out, was_1d)

  batch_np = np.ascontiguousarray(batch, dtype=np.int32)
  mask = batch_np != 0
  counts = np.sum(mask, axis=1)
  bad_rows = np.flatnonzero(counts != n_customers)
  if bad_rows.size > 0:
    bad_idx = int(bad_rows[0])
    raise ValueError(
      f"Delimited route at index {bad_idx} contains {int(counts[bad_idx])} non-zero customers, expected {n_customers}"
    )
  out_np = (
    batch_np[mask].reshape(batch_size, n_customers)
    if batch_size > 0
    else np.empty((0, n_customers), dtype=np.int32)
  )
  return restore_routes_shape(np.ascontiguousarray(out_np, dtype=np.int32), was_1d)


def build_cpu_prefix_tables(
  batch_np: np.ndarray,
  demands: np.ndarray,
  distances: np.ndarray,
  *,
  return_to_depot: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Vectorized NumPy batch computation of `(W, P, d0, dr)` across `(N, n_cust)`."""
  batch_size, n_cust = batch_np.shape
  W = np.zeros((batch_size, n_cust + 1), dtype=np.int64)
  P = np.zeros((batch_size, n_cust + 1), dtype=np.int64)
  if n_cust == 0:
    empty_2d = np.empty((batch_size, 0), dtype=np.int64)
    return W, P, empty_2d, empty_2d

  d0 = np.asarray(distances[0, batch_np], dtype=np.int64)
  dr = (
    np.asarray(distances[batch_np, 0], dtype=np.int64)
    if return_to_depot
    else np.zeros((batch_size, n_cust), dtype=np.int64)
  )
  q_batch = np.asarray(demands[batch_np], dtype=np.int64)
  np.cumsum(q_batch, axis=1, out=W[:, 1:])
  if n_cust > 1:
    step_d = np.asarray(distances[batch_np[:, :-1], batch_np[:, 1:]], dtype=np.int64)
    np.cumsum(step_d, axis=1, out=P[:, 2:])
  return W, P, d0, dr


def score_and_materialize_cpu_batch(
  batch_np: np.ndarray,
  pred_batch: np.ndarray,
  W: np.ndarray,
  P: np.ndarray,
  d0: np.ndarray,
  dr: np.ndarray,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  max_vehicles_constraint: int | None,
  buffer_k: int,
  omega_q: float,
  omega_l: float,
  omega_k: float,
  materialize_routes: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reconstruct `0`-delimited routes and score exact directional costs & violations on CPU."""
  batch_size, n_cust = batch_np.shape
  row_width = n_cust + 1 + buffer_k
  routes_np: np.ndarray | None = (
    np.zeros((batch_size, row_width), dtype=np.int32) if materialize_routes else None
  )
  penalized = np.zeros(batch_size, dtype=np.float64)
  unpenalized = np.zeros(batch_size, dtype=np.float64)
  violations = np.zeros(batch_size, dtype=np.float64)
  num_vehicles = np.zeros(batch_size, dtype=np.int32)

  if n_cust == 0 or batch_size == 0:
    return routes_np, penalized, unpenalized, violations, num_vehicles

  cut_buf = np.empty(n_cust + 1, dtype=np.int32)
  for row_idx in range(batch_size):
    pred_row = pred_batch[row_idx]
    m = 0
    cur = n_cust
    while cur > 0:
      cut_buf[m] = cur
      cur = int(pred_row[cur])
      m += 1
    cut_buf[m] = 0

    eff_m = m if m <= buffer_k else buffer_k
    rem_merges = m - eff_m

    cuts = np.empty(eff_m + 1, dtype=np.int32)
    cuts[eff_m] = n_cust
    cuts[:eff_m] = cut_buf[m : rem_merges : -1]

    starts = cuts[:-1]
    ends = cuts[1:]
    r_loads = W[row_idx, ends] - W[row_idx, starts]
    r_dists = d0[row_idx, starts] + (P[row_idx, ends] - P[row_idx, starts + 1]) + dr[row_idx, ends - 1]

    total_unpen = int(np.sum(r_dists))
    total_cap_viol = int(np.sum(np.maximum(0, r_loads - capacity))) if capacity is not None else 0
    total_dist_viol = (
      float(np.sum(np.maximum(0.0, r_dists.astype(np.float64) - max_route_distance)))
      if max_route_distance is not None
      else 0.0
    )
    veh_viol = max(0, eff_m - max_vehicles_constraint) if max_vehicles_constraint is not None else 0

    cap_pen = (omega_q * float(total_cap_viol)) if total_cap_viol > 0 else 0.0
    dist_pen = (omega_l * total_dist_viol) if total_dist_viol > 0.0 else 0.0
    veh_pen = (omega_k * float(veh_viol)) if veh_viol > 0 else 0.0

    penalized[row_idx] = float(total_unpen) + cap_pen + dist_pen + veh_pen
    unpenalized[row_idx] = float(total_unpen)
    violations[row_idx] = float(total_cap_viol) + total_dist_viol + float(veh_viol)
    num_vehicles[row_idx] = eff_m

    if routes_np is not None:
      row_out = routes_np[row_idx]
      tour_row = batch_np[row_idx]
      for r in range(eff_m):
        a = int(starts[r])
        b = int(ends[r])
        row_out[a + r] = 0
        row_out[a + r + 1 : b + r + 1] = tour_row[a:b]
      row_out[n_cust + eff_m] = 0

  return routes_np, penalized, unpenalized, violations, num_vehicles


__all__: list[str] = [
  "SplitBatchResult",
  "build_cpu_prefix_tables",
  "compact_delimited_route",
  "score_and_materialize_cpu_batch",
  "strip_depot_delimiters",
]
