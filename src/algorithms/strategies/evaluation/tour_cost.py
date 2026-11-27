"""Hamiltonian tour cost evaluation helpers on CPU (NumPy) and GPU (`tour_cost.cu`, `DP17-7`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from protocols.backend import require_gpu
from protocols.vrp_strategies import RouteAssessment

if TYPE_CHECKING:
  from algorithms.strategies.decoding import SplitBatchResult
  from protocols.problem_context import ProblemContext


def evaluate_tour_cpu(batch_np: np.ndarray, context: ProblemContext) -> np.ndarray:
  """Compute `(N,)` `float64` Hamiltonian tour costs on CPU, honoring `context.return_to_depot`."""
  batch_size, n = batch_np.shape
  if batch_size == 0 or n <= 1:
    return np.zeros(batch_size, dtype=np.float64)

  distances: np.ndarray = context.get_cpu_distances()
  flat_dist = distances.ravel()
  stride = distances.shape[1]
  costs = flat_dist[batch_np[:, :-1] * stride + batch_np[:, 1:]].sum(axis=1, dtype=np.float64)
  if context.return_to_depot:
    costs += flat_dist[batch_np[:, -1] * stride + batch_np[:, 0]]
  return cast("np.ndarray", costs)


def launch_tour_cost_gpu(
  kernel: Any,
  d_population: Any,
  d_distances: Any,
  d_coords: Any,
  *,
  edge_type_code: int,
  threads_per_block: int,
  return_to_depot: bool,
) -> Any:
  """Launch `tour_cost_kernel` on `(N, n)` `int32` device tours and return `(N,)` `int64` costs."""
  cp = require_gpu()
  batch_size, n = d_population.shape
  d_costs = cp.zeros(batch_size, dtype=cp.int64)
  if batch_size == 0:
    return d_costs

  cost_shared = threads_per_block * 8  # s_partials (long long)
  kernel(
    (batch_size,),
    (threads_per_block,),
    (
      d_population,
      d_distances,
      d_coords,
      d_costs,
      np.int32(n),
      np.int32(batch_size),
      np.int32(edge_type_code),
      np.int32(1 if return_to_depot else 0),
    ),
    shared_mem=cost_shared,
  )
  return d_costs


def build_tour_assessment(costs: Any, *, on_gpu: bool) -> RouteAssessment:
  """Build a feasible `RouteAssessment` with zero violations for unconstrained Hamiltonian tours."""
  xp: Any = require_gpu() if on_gpu else np
  costs_f = xp.asarray(costs, dtype=xp.float64)
  n_rows = int(costs_f.shape[0])
  zeros = xp.zeros(n_rows, dtype=xp.float64)
  return RouteAssessment(
    objectives=costs_f,
    travel_costs=costs_f,
    is_feasible=xp.ones(n_rows, dtype=bool),
    violations=zeros,
    normalized_violations=zeros.copy(),
    raw_violation_components=xp.zeros((n_rows, 3), dtype=xp.float64),
    num_vehicles=xp.ones(n_rows, dtype=xp.int32),
  )


def build_tour_detailed_result(routes: Any, costs: Any, *, on_gpu: bool) -> SplitBatchResult:
  """Build a `SplitBatchResult` for unconstrained Hamiltonian tours on CPU or GPU."""
  from algorithms.strategies.decoding import SplitBatchResult

  xp: Any = require_gpu() if on_gpu else np
  costs_f = xp.asarray(costs, dtype=xp.float64)
  n_rows = int(costs_f.shape[0])
  return SplitBatchResult(
    routes=routes,
    penalized_costs=costs_f,
    unpenalized_costs=costs_f,
    violations=xp.zeros(n_rows, dtype=xp.float64),
    num_vehicles=xp.ones(n_rows, dtype=xp.int32),
  )


__all__: list[str] = [
  "build_tour_assessment",
  "build_tour_detailed_result",
  "evaluate_tour_cpu",
  "launch_tour_cost_gpu",
]
