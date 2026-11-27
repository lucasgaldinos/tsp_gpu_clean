"""Vectorized CPU and cooperative CUDA Greedy Split decoder (`Task 17.21`, `DP17-2`, `BUG-116`)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .split_base import _BaseSplitDecoder


def greedy_predecessors_cpu_batch(
  W: np.ndarray,
  P: np.ndarray,
  d0: np.ndarray,
  dr: np.ndarray,
  *,
  capacity: int | None,
  max_route_distance: float | None,
) -> np.ndarray:
  """Compute greedy linear-scan predecessor pointers `(N, n_cust + 1)` vectorized across the batch."""
  batch_size = int(W.shape[0])
  n_cust = int(W.shape[1]) - 1
  pred = np.empty((batch_size, n_cust + 1), dtype=np.int32)
  pred[:, 0] = -1
  if batch_size == 0 or n_cust == 0:
    return pred

  row_arange = np.arange(batch_size, dtype=np.intp)
  route_start = np.zeros(batch_size, dtype=np.int32)

  for j in range(1, n_cust + 1):
    is_first = route_start == (j - 1)
    new_load = W[:, j] - W[row_arange, route_start]
    new_open = d0[row_arange, route_start] + (P[:, j] - P[row_arange, route_start + 1])
    new_arc = new_open + dr[:, j - 1]

    infeasible = np.zeros(batch_size, dtype=bool)
    if capacity is not None:
      infeasible |= new_load > capacity
    if max_route_distance is not None:
      infeasible |= new_arc.astype(np.float64) > max_route_distance

    start_new = (~is_first) & infeasible
    route_start = np.where(start_new, np.int32(j - 1), route_start)
    pred[:, j] = route_start

  return pred


class GreedySplitDecoder(_BaseSplitDecoder):
  """Linear-scan :math:`O(n_{\\text{cust}})` Greedy Split decoder on vectorized CPU and CUDA."""

  _mode_prins: ClassVar[bool] = False

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
    _ = (omega_q, omega_l)
    return greedy_predecessors_cpu_batch(
      W,
      P,
      d0,
      dr,
      capacity=capacity,
      max_route_distance=max_route_distance,
    )


__all__: list[str] = ["GreedySplitDecoder", "greedy_predecessors_cpu_batch"]
