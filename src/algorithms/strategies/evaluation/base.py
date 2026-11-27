"""Shared base classes for CPU and GPU route cost evaluation strategies (`DP17-8`).

Centralizes:
- `EvaluationCPUBase`: single-route `evaluate()` delegation and `StrategyBase`
  polymorphic defaults (`canonicalize_route`, `assess_batch`).
- `EvaluationGPUBase`: `require_gpu()` device guard (`DP9-2`), context-owned
  distance operand caching with `BUG-95` H2D delta accounting, `return_gpu`
  output control, and D2H byte tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from algorithms.strategies.base import StrategyBase
from protocols.algorithm_strategies import EvaluationStrategy
from protocols.backend import require_gpu
from utils.route_helpers import is_gpu_array

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Fitness, Route, SeedLike


class EvaluationCPUBase(StrategyBase, EvaluationStrategy):
  """Shared base class for CPU evaluation strategies."""

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize CPU evaluator state."""
    super().__init__(seed=seed)

  def evaluate(self, route: Route, context: ProblemContext) -> float:
    """Compute scalar cost for a single 1D route `(n_cities,)`."""
    costs: Fitness = self.evaluate_batch(route, context)
    return float(costs[0])


class EvaluationGPUBase(StrategyBase, EvaluationStrategy):
  """Shared base class for GPU CUDA evaluation strategies (`DP9-2`, `BUG-95`)."""

  def __init__(
    self,
    *,
    return_gpu: bool = False,
    max_matrix_dimension: int | None = None,
    seed: SeedLike = None,
  ) -> None:
    """Initialize GPU evaluator state and operand cache mirrors."""
    super().__init__(seed=seed)
    self.return_gpu: bool = return_gpu
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self._kernel: Any = None
    self._cached_distances_gpu: Any = None
    self._cached_problem_key: tuple[str, int, str, int] | None = None
    self._cached_coords_gpu: Any = None
    self._cached_coords_key: tuple[str, int, str, int] | None = None

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters, dropping mirrored VRAM references."""
    super().reset_metrics()
    self._cached_distances_gpu = None
    self._cached_problem_key = None
    self._cached_coords_gpu = None
    self._cached_coords_key = None

  def _distance_operands(self, context: ProblemContext, n: int) -> tuple[Any, Any, int]:
    """Return `(d_distances, d_coords, edge_type_code)` charging physical H2D delta once (`BUG-95`)."""
    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(n, self.max_matrix_dimension)
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)

    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    if isinstance(d_distances, int):
      if d_coords is None:
        raise ValueError(f"Problem '{problem.name}' has no coordinates for the on-the-fly distance mode")
      self._cached_coords_gpu = d_coords
      self._cached_coords_key = problem_key
    else:
      self._cached_distances_gpu = d_distances
      self._cached_problem_key = problem_key
    return d_distances, d_coords, edge_type_code

  def _upload_routes_batch(self, batch: Any) -> tuple[Any, bool]:
    """Ensure 2D `int32` routes reside on GPU, charging H2D bytes if uploaded from host."""
    cp = require_gpu()
    input_on_gpu = is_gpu_array(batch)
    if input_on_gpu:
      return cp.ascontiguousarray(batch, dtype=cp.int32), True
    batch_np = np.ascontiguousarray(batch, dtype=np.int32)
    d_batch = cp.asarray(batch_np, dtype=cp.int32)
    self.h2d_bytes += int(d_batch.nbytes)
    return d_batch, False

  def _finalize_costs(
    self,
    d_costs: Any,
    *,
    input_on_gpu: bool,
    out_on_gpu: bool | None = None,
  ) -> Fitness:
    """Return `float64` costs on GPU or copy to host while recording `d2h_bytes`."""
    cp = require_gpu()
    keep_on_gpu = (input_on_gpu and self.return_gpu) if out_on_gpu is None else out_on_gpu
    if keep_on_gpu:
      return cast("Fitness", d_costs.astype(cp.float64))
    costs_cpu = d_costs.get().astype(np.float64)
    self.d2h_bytes += int(d_costs.nbytes)
    return cast("Fitness", costs_cpu)

  def evaluate(self, route: Route, context: ProblemContext) -> float:
    """Compute scalar cost for a single 1D route `(n_cities,)` on GPU."""
    costs: Fitness = self.evaluate_batch(route, context)
    if is_gpu_array(costs):
      return float(require_gpu().asnumpy(costs)[0])
    return float(costs[0])


__all__: list[str] = ["EvaluationCPUBase", "EvaluationGPUBase"]
