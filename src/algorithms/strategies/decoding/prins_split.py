"""Vectorized CPU and warp-parallel CUDA Prins (2004) Split decoder (`Task 17.21`, `DP17-2`, `BUG-113`)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from .split_base import _BaseSplitDecoder


def prins_predecessors_cpu_batch(
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
  """Compute exact Prins (2004) DAG predecessor pointers `(N, n_cust + 1)` in vectorized NumPy."""
  batch_size = int(W.shape[0])
  n_cust = int(W.shape[1]) - 1
  pred = np.broadcast_to(np.arange(-1, n_cust, dtype=np.int32), (batch_size, n_cust + 1)).copy()
  if batch_size == 0 or n_cust == 0:
    return pred

  V = np.full((batch_size, n_cust + 1), np.inf, dtype=np.float64)
  V[:, 0] = 0.0
  row_arange = np.arange(batch_size, dtype=np.intp)
  i_min = 0

  for j in range(1, n_cust + 1):
    # Monotone lower-bound window tightening across the batch.
    if capacity is not None:
      while i_min < j - 1 and bool(np.all(W[:, j] - W[:, i_min] > capacity)):
        i_min += 1

    loads = W[:, j : j + 1] - W[:, i_min:j]
    suffix_chain = P[:, j : j + 1] - P[:, i_min + 1 : j + 1]
    arc_dists = d0[:, i_min:j] + suffix_chain + dr[:, j - 1 : j]
    cands = V[:, i_min:j] + arc_dists.astype(np.float64)

    # Multi-customer arcs (columns :-1) must satisfy capacity and distance limits strictly.
    if j - i_min > 1:
      infeasible = np.zeros((batch_size, j - i_min - 1), dtype=bool)
      if capacity is not None:
        infeasible |= loads[:, :-1] > capacity
      if max_route_distance is not None:
        infeasible |= arc_dists[:, :-1].astype(np.float64) > max_route_distance
      cands[:, :-1] = np.where(infeasible, np.inf, cands[:, :-1])

    # Single-customer arc (column -1, i = j - 1) is always admissible with IEEE-754-safe penalties.
    single_cost = V[:, j - 1] + arc_dists[:, -1].astype(np.float64)
    if capacity is not None:
      cap_ex = np.maximum(0, loads[:, -1] - capacity)
      cap_pen = np.where(cap_ex > 0, np.inf, 0.0) if math.isinf(omega_q) else (omega_q * cap_ex.astype(np.float64))
      single_cost = single_cost + cap_pen
    if max_route_distance is not None:
      dist_ex = np.maximum(0.0, arc_dists[:, -1].astype(np.float64) - max_route_distance)
      dist_pen = np.where(dist_ex > 0.0, np.inf, 0.0) if math.isinf(omega_l) else (omega_l * dist_ex)
      single_cost = single_cost + dist_pen
    cands[:, -1] = single_cost

    # `np.argmin` deterministically picks the smallest index `i` on ties (matches GPU `warp_reduce_argmin_double_int`).
    best_rel = np.argmin(cands, axis=1)
    V[:, j] = cands[row_arange, best_rel]
    pred[:, j] = np.int32(i_min) + best_rel.astype(np.int32)

  return pred


class PrinsSplitDecoder(_BaseSplitDecoder):
  """Exact Prins (2004) DAG shortest-path Split decoder on vectorized CPU and warp-parallel CUDA."""

  _mode_prins: ClassVar[bool] = True

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
    return prins_predecessors_cpu_batch(
      W,
      P,
      d0,
      dr,
      capacity=capacity,
      max_route_distance=max_route_distance,
      omega_q=omega_q,
      omega_l=omega_l,
    )


__all__: list[str] = ["PrinsSplitDecoder", "prins_predecessors_cpu_batch"]
