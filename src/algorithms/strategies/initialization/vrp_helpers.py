"""Shared VRP customer, encoding, polar-order, and savings helpers for initializers (`Task 17.31`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.problem_context import ProblemContext


def resolve_vrp_customers(customers: Sequence[int], context: ProblemContext) -> np.ndarray:
  """Extract non-depot customer IDs ``{1..dimension-1}`` as an ``int32`` NumPy array."""
  arr = np.asarray(customers, dtype=np.int32)
  filtered = arr[arr != 0]
  if filtered.size > 0:
    return np.ascontiguousarray(filtered, dtype=np.int32)
  return np.arange(1, context.dimension, dtype=np.int32)


def pack_vrp_routes_np(
  routes: list[np.ndarray],
  *,
  dimension: int,
  k_max: int,
  output_format: Literal["giant_tour", "delimited"],
) -> np.ndarray:
  """Encode a list of customer route slices as a Giant-Tour or ``0``-delimited ``int32`` row."""
  non_empty = [r for r in routes if r.size > 0]
  if output_format == "giant_tour":
    if not non_empty:
      return np.empty(0, dtype=np.int32)
    return np.concatenate(non_empty).astype(np.int32, copy=False)

  if len(non_empty) > k_max:
    merged_tail = np.concatenate(non_empty[k_max - 1 :]).astype(np.int32, copy=False)
    non_empty = [*non_empty[: k_max - 1], merged_tail]

  row_width = dimension + k_max
  row = np.zeros(row_width, dtype=np.int32)
  pos = 1
  for route in non_empty:
    end = pos + int(route.size)
    row[pos:end] = route
    pos = end + 1
  return row


def compute_sorted_sweep_customers(
  customer_ids: np.ndarray,
  context: ProblemContext,
) -> np.ndarray:
  """Order ``customer_ids`` by depot polar angle (or ``(d(0, v), v)`` when coords are absent)."""
  if customer_ids.size <= 1:
    return customer_ids.copy()
  coords = context.get_cpu_coordinates()
  if coords is not None and coords.ndim == 2 and coords.shape[1] >= 2:
    dx = coords[customer_ids, 0] - coords[0, 0]
    dy = coords[customer_ids, 1] - coords[0, 1]
    angles = np.arctan2(dy, dx)
    order = np.lexsort((customer_ids, angles))
    return np.ascontiguousarray(customer_ids[order], dtype=np.int32)

  distances = context.get_cpu_distances()
  depot_dists = distances[0, customer_ids]
  order = np.lexsort((customer_ids, depot_dists))
  return np.ascontiguousarray(customer_ids[order], dtype=np.int32)


def compute_clarke_wright_pairs(
  customer_ids: np.ndarray,
  context: ProblemContext,
  *,
  shape_param: float,
  candidate_k: int | None,
  distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Build and sort positive Clarke & Wright savings pairs ``(sorted_u, sorted_v)``."""
  n_cust = int(customer_ids.size)
  if n_cust < 2:
    empty = np.empty(0, dtype=np.int32)
    return empty, empty

  return_to_depot = context.constraints.return_to_depot
  is_symmetric = context.problem.is_symmetric
  symmetric_closed = is_symmetric and return_to_depot

  if candidate_k is not None and n_cust > candidate_k:
    cand_table = context.get_cpu_candidate_list(candidate_k)
    k_eff = int(cand_table.shape[1])
    pu = np.repeat(customer_ids, k_eff).astype(np.int32, copy=False)
    pv = cand_table[customer_ids].ravel().astype(np.int32, copy=False)
    valid = (pv != 0) & (pv != pu) & np.isin(pv, customer_ids)
    pu = pu[valid]
    pv = pv[valid]
    if symmetric_closed:
      lo = np.minimum(pu, pv)
      hi = np.maximum(pu, pv)
      keys = lo.astype(np.int64) * np.int64(context.dimension) + hi.astype(np.int64)
      _, first_idx = np.unique(keys, return_index=True)
      first_idx.sort()
      pairs_u = lo[first_idx]
      pairs_v = hi[first_idx]
    else:
      keys = pu.astype(np.int64) * np.int64(context.dimension) + pv.astype(np.int64)
      _, first_idx = np.unique(keys, return_index=True)
      first_idx.sort()
      pairs_u = pu[first_idx]
      pairs_v = pv[first_idx]
  elif symmetric_closed:
    tri_i, tri_j = np.triu_indices(n_cust, k=1)
    pairs_u = customer_ids[tri_i]
    pairs_v = customer_ids[tri_j]
  else:
    grid_i, grid_j = np.where(~np.eye(n_cust, dtype=bool))
    pairs_u = customer_ids[grid_i]
    pairs_v = customer_ids[grid_j]

  if pairs_u.size == 0:
    empty = np.empty(0, dtype=np.int32)
    return empty, empty

  if distances is not None:
    d_u0 = distances[pairs_u, 0].astype(np.float64)
    d_0v = distances[0, pairs_v].astype(np.float64)
    d_uv = distances[pairs_u, pairs_v].astype(np.float64)
  else:
    coords = context.get_cpu_coordinates()
    if coords is None:
      distances = context.get_cpu_distances()
      d_u0 = distances[pairs_u, 0].astype(np.float64)
      d_0v = distances[0, pairs_v].astype(np.float64)
      d_uv = distances[pairs_u, pairs_v].astype(np.float64)
    else:
      from distances.pairwise import get_distance_function

      dist_fn = get_distance_function(context.problem.edge_type)
      c0 = coords[0]
      d_u0 = np.array([dist_fn(coords[int(u)], c0) for u in pairs_u], dtype=np.float64)
      d_0v = np.array([dist_fn(c0, coords[int(v)]) for v in pairs_v], dtype=np.float64)
      d_uv = np.array(
        [dist_fn(coords[int(u)], coords[int(v)]) for u, v in zip(pairs_u, pairs_v, strict=True)],
        dtype=np.float64,
      )

  depot_u = d_u0 if return_to_depot else 0.0
  savings = depot_u + d_0v - shape_param * d_uv
  pos_mask = savings > 0.0
  pairs_u = pairs_u[pos_mask]
  pairs_v = pairs_v[pos_mask]
  savings = savings[pos_mask]
  if pairs_u.size == 0:
    empty = np.empty(0, dtype=np.int32)
    return empty, empty

  order = np.lexsort((pairs_v, pairs_u, -savings))
  return (
    np.ascontiguousarray(pairs_u[order], dtype=np.int32),
    np.ascontiguousarray(pairs_v[order], dtype=np.int32),
  )


__all__: list[str] = [
  "compute_clarke_wright_pairs",
  "compute_sorted_sweep_customers",
  "pack_vrp_routes_np",
  "resolve_vrp_customers",
]
