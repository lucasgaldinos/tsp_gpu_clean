"""Vectorized NumPy batch feasibility screen and CPU FFD/BFD fleet repacking (`Task 17.30`, `BUG-117`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape, to_cpu_batch

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import RouteOrRoutes


def screen_violating_rows_cpu(
  batch_np: np.ndarray,
  demands: np.ndarray,
  *,
  capacity: int | None,
  max_vehicles: int | None,
) -> np.ndarray:
  """Screen ``(N, W)`` ``0``-delimited VRP rows for fleet-count or capacity violations.

  Uses ``np.add.reduceat`` over non-zero customer segments (zero host list conversions)
  so feasible batches skip distance-matrix lookup and row repacking completely.
  """
  batch_size, width = batch_np.shape
  if batch_size == 0 or width == 0:
    return np.zeros(batch_size, dtype=bool)

  has_zero = np.any(batch_np == 0, axis=1)
  padded = np.zeros((batch_size, width + 1), dtype=np.int32)
  padded[:, :width] = batch_np
  flat = padded.ravel()
  nz = flat != 0
  if not bool(np.any(nz)):
    return np.zeros(batch_size, dtype=bool)

  prev_nz = np.empty_like(nz)
  prev_nz[0] = False
  prev_nz[1:] = nz[:-1]
  starts = np.flatnonzero(nz & ~prev_nz)
  seg_rows = starts // (width + 1)
  route_counts = np.bincount(seg_rows, minlength=batch_size)

  demands_i64 = np.asarray(demands, dtype=np.int64).copy()
  demands_i64[0] = 0
  flat_demands = demands_i64[flat]
  seg_loads = np.add.reduceat(flat_demands, starts)

  max_loads = np.zeros(batch_size, dtype=np.int64)
  np.maximum.at(max_loads, seg_rows, seg_loads)

  viol = np.zeros(batch_size, dtype=bool)
  if max_vehicles is not None:
    viol |= route_counts > max_vehicles
  if capacity is not None:
    viol |= max_loads > capacity
  viol &= has_zero
  return viol


def _extract_segments_np(row_np: np.ndarray) -> list[np.ndarray]:
  """Split a ``0``-delimited 1D ``int32`` row into non-empty customer NumPy slices."""
  nz = row_np != 0
  if not bool(np.any(nz)):
    return []
  ext = np.zeros(row_np.size + 2, dtype=bool)
  ext[1:-1] = nz
  starts = np.flatnonzero(ext[1:-1] & ~ext[:-2])
  ends = np.flatnonzero(ext[1:-1] & ~ext[2:]) + 1
  return [row_np[int(s) : int(e)].copy() for s, e in zip(starts, ends, strict=True)]


def _route_distance_np(
  route: np.ndarray,
  distances: np.ndarray,
  *,
  return_to_depot: bool,
) -> float:
  """Compute travel distance of one customer NumPy slice starting at depot ``0``."""
  if route.size == 0:
    return 0.0
  total = float(distances[0, int(route[0])])
  if route.size > 1:
    total += float(np.sum(distances[route[:-1], route[1:]], dtype=np.int64))
  if return_to_depot:
    total += float(distances[int(route[-1]), 0])
  return total


def _best_detour_in_route_np(
  route: np.ndarray,
  node: int,
  distances: np.ndarray,
  *,
  max_route_distance: float | None,
  route_dist: float,
  return_to_depot: bool,
) -> tuple[float, int]:
  """Return ``(best_detour, best_pos)`` for inserting ``node`` into ``route`` (``pos=-1`` if infeasible)."""
  length = int(route.size)
  prev_nodes = np.empty(length + 1, dtype=np.int32)
  prev_nodes[0] = 0
  if length > 0:
    prev_nodes[1:] = route

  next_nodes = np.empty(length + 1, dtype=np.int32)
  if length > 0:
    next_nodes[:-1] = route
  next_nodes[-1] = 0

  d_prev_u = distances[prev_nodes, node].astype(np.float64)
  d_u_next = distances[node, next_nodes].astype(np.float64)
  d_prev_next = distances[prev_nodes, next_nodes].astype(np.float64)
  if not return_to_depot:
    d_u_next[-1] = 0.0
    d_prev_next[-1] = 0.0
  elif length == 0:
    d_prev_next[0] = 0.0

  detours = d_prev_u + d_u_next - d_prev_next
  if max_route_distance is not None:
    detours = np.where(route_dist + detours <= max_route_distance, detours, np.inf)

  best_pos = int(np.argmin(detours))
  best_detour = float(detours[best_pos])
  if not np.isfinite(best_detour):
    return np.inf, -1
  return best_detour, best_pos


def _repack_row_cpu(
  sub_routes: list[np.ndarray],
  demands: np.ndarray,
  distances: np.ndarray,
  *,
  mode: int,
  capacity: int | None,
  max_vehicles: int | None,
  max_route_distance: float | None,
  return_to_depot: bool,
) -> np.ndarray:
  """Repair one violating ``sub_routes`` partition via FFD (``mode=0``) or BFD (``mode=1``)."""
  route_loads = [int(np.sum(demands[r], dtype=np.int64)) for r in sub_routes]
  order = sorted(
    range(len(sub_routes)),
    key=lambda idx: (
      0 if (capacity is None or route_loads[idx] <= capacity) else 1,
      -route_loads[idx],
      idx,
    ),
  )

  k_limit = max_vehicles if max_vehicles is not None else len(order)
  kept_routes: list[np.ndarray] = []
  kept_loads: list[int] = []
  ejected_parts: list[np.ndarray] = []

  for rank, orig_idx in enumerate(order):
    r_arr = sub_routes[orig_idx]
    load = route_loads[orig_idx]
    if rank < k_limit:
      keep_len = int(r_arr.size)
      if capacity is not None and load > capacity:
        while keep_len > 0 and load > capacity:
          keep_len -= 1
          load -= int(demands[int(r_arr[keep_len])])
        ejected_parts.append(r_arr[keep_len:][::-1])
      if keep_len > 0:
        kept_routes.append(r_arr[:keep_len].copy())
        kept_loads.append(load)
    else:
      ejected_parts.append(r_arr)

  if ejected_parts:
    ejected_arr = np.concatenate(ejected_parts).astype(np.int32, copy=False)
    if ejected_arr.size > 1:
      lex = np.lexsort((ejected_arr, -demands[ejected_arr].astype(np.int64)))
      ejected_arr = ejected_arr[lex]
  else:
    ejected_arr = np.empty(0, dtype=np.int32)

  kept_dists = [_route_distance_np(r, distances, return_to_depot=return_to_depot) for r in kept_routes]

  for u_raw in ejected_arr:
    u = int(u_raw)
    q_u = int(demands[u])
    chosen_r = -1
    chosen_pos = -1
    chosen_detour = 0.0

    if mode == 0:
      for r_idx, route in enumerate(kept_routes):
        if capacity is not None and kept_loads[r_idx] + q_u > capacity:
          continue
        detour, pos = _best_detour_in_route_np(
          route,
          u,
          distances,
          max_route_distance=max_route_distance,
          route_dist=kept_dists[r_idx],
          return_to_depot=return_to_depot,
        )
        if pos >= 0:
          chosen_r, chosen_pos, chosen_detour = r_idx, pos, detour
          break
    else:
      best_bfd: tuple[int, float, int, int] | None = None
      for r_idx, route in enumerate(kept_routes):
        if capacity is not None and kept_loads[r_idx] + q_u > capacity:
          continue
        detour, pos = _best_detour_in_route_np(
          route,
          u,
          distances,
          max_route_distance=max_route_distance,
          route_dist=kept_dists[r_idx],
          return_to_depot=return_to_depot,
        )
        if pos >= 0:
          residual = (capacity - (kept_loads[r_idx] + q_u)) if capacity is not None else 0
          cand = (residual, detour, r_idx, pos)
          if best_bfd is None or cand < best_bfd:
            best_bfd = cand
      if best_bfd is not None:
        _, chosen_detour, chosen_r, chosen_pos = best_bfd

    if chosen_r >= 0:
      kept_routes[chosen_r] = np.insert(kept_routes[chosen_r], chosen_pos, np.int32(u))
      kept_loads[chosen_r] += q_u
      kept_dists[chosen_r] += chosen_detour
      continue

    if max_vehicles is None or len(kept_routes) < max_vehicles:
      single = np.array([u], dtype=np.int32)
      kept_routes.append(single)
      kept_loads.append(q_u)
      kept_dists.append(_route_distance_np(single, distances, return_to_depot=return_to_depot))
      continue

    # Hard fleet cap reached: preserve every customer by minimizing overflow.
    best_fb: tuple[int, float, float, int, int] | None = None
    for r_idx, route in enumerate(kept_routes):
      old_cap_ex = max(0, kept_loads[r_idx] - capacity) if capacity is not None else 0
      new_cap_ex = max(0, kept_loads[r_idx] + q_u - capacity) if capacity is not None else 0
      cap_delta = new_cap_ex - old_cap_ex
      detour, pos = _best_detour_in_route_np(
        route,
        u,
        distances,
        max_route_distance=None,
        route_dist=kept_dists[r_idx],
        return_to_depot=return_to_depot,
      )
      old_dist_ex = max(0.0, kept_dists[r_idx] - max_route_distance) if max_route_distance is not None else 0.0
      new_dist_ex = max(0.0, kept_dists[r_idx] + detour - max_route_distance) if max_route_distance is not None else 0.0
      cand_fb = (cap_delta, new_dist_ex - old_dist_ex, detour, r_idx, pos)
      if best_fb is None or cand_fb < best_fb:
        best_fb = cand_fb

    if best_fb is not None:
      _, _, best_detour, fb_r, fb_pos = best_fb
      kept_routes[fb_r] = np.insert(kept_routes[fb_r], fb_pos, np.int32(u))
      kept_loads[fb_r] += q_u
      kept_dists[fb_r] += best_detour
    else:
      single = np.array([u], dtype=np.int32)
      kept_routes.append(single)
      kept_loads.append(q_u)
      kept_dists.append(_route_distance_np(single, distances, return_to_depot=return_to_depot))

  non_empty = [r for r in kept_routes if r.size > 0]
  if not non_empty:
    return np.zeros(1, dtype=np.int32)

  total_cust = sum(int(r.size) for r in non_empty)
  encoded = np.zeros(total_cust + len(non_empty) + 1, dtype=np.int32)
  cursor = 1
  for r in non_empty:
    end = cursor + int(r.size)
    encoded[cursor:end] = r
    cursor = end + 1
  return encoded


def pack_batch_cpu(
  routes: RouteOrRoutes,
  context: ProblemContext,
  *,
  mode: int,
) -> tuple[RouteOrRoutes, int, int]:
  """Repack violating ``0``-delimited VRP rows on CPU; returns ``(result, d2h_bytes, h2d_bytes)``."""
  constraints = context.constraints
  if not context.problem.is_vrp or (constraints.max_vehicles is None and constraints.capacity is None):
    return routes, 0, 0

  batch, was_1d = ensure_routes_batch(routes)
  if int(batch.shape[0]) == 0 or int(batch.shape[1]) < context.dimension:
    return routes, 0, 0

  on_gpu = is_gpu_array(batch)
  batch_np, _, _ = to_cpu_batch(batch)
  d2h_bytes = int(batch_np.nbytes) if on_gpu else 0

  demands = context.get_cpu_demands()
  if demands is None:
    demands = np.zeros(context.dimension, dtype=np.int32)

  viol_mask = screen_violating_rows_cpu(
    batch_np,
    demands,
    capacity=constraints.capacity,
    max_vehicles=constraints.max_vehicles,
  )
  viol_indices = np.flatnonzero(viol_mask)
  if viol_indices.size == 0:
    return routes, d2h_bytes, 0

  distances = context.get_cpu_distances()
  width = int(batch_np.shape[1])
  repacked_rows: dict[int, np.ndarray] = {}
  max_len = width

  for idx_raw in viol_indices:
    idx = int(idx_raw)
    sub_routes = _extract_segments_np(batch_np[idx])
    if not sub_routes:
      continue
    encoded = _repack_row_cpu(
      sub_routes,
      demands,
      distances,
      mode=mode,
      capacity=constraints.capacity,
      max_vehicles=constraints.max_vehicles,
      max_route_distance=constraints.max_route_distance,
      return_to_depot=constraints.return_to_depot,
    )
    repacked_rows[idx] = encoded
    max_len = max(max_len, int(encoded.size))

  if not repacked_rows:
    return routes, d2h_bytes, 0

  out_np = np.zeros((int(batch_np.shape[0]), max_len), dtype=np.int32)
  out_np[:, :width] = batch_np
  for idx, encoded in repacked_rows.items():
    out_np[idx, :] = 0
    out_np[idx, : int(encoded.size)] = encoded

  result = restore_routes_shape(out_np, was_1d, to_gpu=on_gpu)
  h2d_bytes = int(getattr(result, "nbytes", out_np.nbytes)) if on_gpu else 0
  return result, d2h_bytes, h2d_bytes


__all__: list[str] = ["pack_batch_cpu", "screen_violating_rows_cpu"]
