"""Shared base class and VRAM operand cache for GPU local search strategies (`DP17-8`).

Extracts `DistanceOperandCache` (context-owned distance, coordinate, and candidate-list
VRAM bindings with `BUG-95` H2D delta tracking) and `BaseLocalSearchGPU` (shared
configuration, operand resolution, and monotonic workspace buffer allocation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from algorithms.strategies.base import StrategyBase
from protocols.algorithm_strategies import LocalSearchStrategy
from protocols.backend import require_gpu

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Route, SeedLike


class DistanceOperandCache:
  """Strategy-local mirror of the GPU buffers owned by `ProblemContext`.

  The context is the single owner of the `int32` distance matrix, the coordinate
  buffer (`DP10-17`), and the `k`-NN candidate lists (`DP10-20`). This cache only
  records which buffers the strategy bound last and how many bytes the first
  physical upload cost (`BUG-95`).
  """

  def __init__(self) -> None:
    self.distances: Any = None
    self.distance_key: tuple[str, int, str, int] | None = None
    self.coords: Any = None
    self.coords_key: tuple[str, int, str, int] | None = None
    self.candidates: Any = None
    self.candidates_key: tuple[str, int, str, int] | None = None

  def reset(self) -> None:
    """Forget the mirrored buffers after a metrics reset."""
    self.distances = None
    self.distance_key = None
    self.coords = None
    self.coords_key = None
    self.candidates = None
    self.candidates_key = None

  def resolve(
    self,
    context: ProblemContext,
    n: int,
    max_matrix_dimension: int | None,
  ) -> tuple[tuple[Any, Any, int], int]:
    """Return `((d_distances, d_coords, edge_type_code), uploaded_bytes)`."""
    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_distances, d_coords, edge_type_code = context.get_gpu_distance_operands(n, max_matrix_dimension)
    uploaded_bytes = max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    if isinstance(d_distances, int):
      self.coords = d_coords
      self.coords_key = problem_key
      self.distances = None
      self.distance_key = None
    else:
      self.distances = d_distances
      self.distance_key = problem_key
      self.coords = None
      self.coords_key = None
    return (d_distances, d_coords, edge_type_code), uploaded_bytes

  def resolve_candidates(self, context: ProblemContext, k: int) -> tuple[Any, int]:
    """Return `(d_candidates, uploaded_bytes)` for the context-owned `N_k` table."""
    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_candidates = context.get_gpu_candidate_list(k)
    uploaded_bytes = max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    problem = context.problem
    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    self.candidates = d_candidates
    self.candidates_key = problem_key
    return d_candidates, uploaded_bytes


def ensure_gpu_workspace(buffer: Any, min_shape: tuple[int, ...], dtype: Any) -> Any:
  """Grow a persistent GPU workspace buffer monotonically without shrinking across generations."""
  cp = require_gpu()
  if (
    buffer is not None
    and buffer.dtype == cp.dtype(dtype)
    and len(buffer.shape) == len(min_shape)
    and all(cur >= req for cur, req in zip(buffer.shape, min_shape, strict=True))
  ):
    return buffer
  if buffer is not None and len(buffer.shape) == len(min_shape):
    alloc_shape = tuple(max(cur, req) for cur, req in zip(buffer.shape, min_shape, strict=True))
  else:
    alloc_shape = min_shape
  return cp.empty(alloc_shape, dtype=dtype)


class BaseLocalSearchGPU(StrategyBase, LocalSearchStrategy):
  """Shared base class for CUDA local search strategies (`DP17-8`)."""

  def __init__(
    self,
    *,
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    candidate_k: int | None = None,
    seed: SeedLike = None,
  ) -> None:
    """Initialize shared GPU local-search state."""
    super().__init__(seed=seed)
    self.threads_per_block: int = threads_per_block
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.candidate_k: int | None = candidate_k
    self._kernel: Any = None
    self._operands: DistanceOperandCache = DistanceOperandCache()

  def reset_metrics(self) -> None:
    """Reset transfer/kernel counters and mirrored operand bindings."""
    super().reset_metrics()
    self._operands.reset()

  def _resolve_distance_operands(
    self,
    context: ProblemContext,
    n: int,
  ) -> tuple[Any, Any, int]:
    """Bind context-owned GPU distance/coordinate operands and charge physical H2D delta."""
    operands, uploaded_bytes = self._operands.resolve(context, n, self.max_matrix_dimension)
    self.h2d_bytes += uploaded_bytes
    return operands

  def _resolve_candidate_table(self, context: ProblemContext, k: int) -> Any:
    """Bind context-owned `N_k` table and charge physical H2D delta once (`BUG-27`)."""
    d_candidates, candidate_bytes = self._operands.resolve_candidates(context, k)
    self.h2d_bytes += candidate_bytes
    return d_candidates

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D route `(n_cities,)` through `improve_batch`."""
    return self.improve_batch(route, context)


__all__: list[str] = [
  "BaseLocalSearchGPU",
  "DistanceOperandCache",
  "ensure_gpu_workspace",
]
