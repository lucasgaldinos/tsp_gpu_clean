"""
No-Op Local Search strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from protocols.algorithm_strategies import LocalSearchStrategy
from utils.phase_timer import time_strategy
from utils.route_helpers import ensure_routes_batch, restore_routes_shape

from ..base import StrategyBase

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes


class LocalSearchNoOp(StrategyBase, LocalSearchStrategy):
  """Identity local search strategy that returns routes unchanged."""

  def __init__(self, max_iterations: int = 0) -> None:
    """Initialize no-op local search."""
    super().__init__()
    self.max_iterations: int = max_iterations

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Return single 1D route unchanged."""
    return route.copy()

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Return 1D route or 2D batch of routes unchanged."""
    batch, was_1d = ensure_routes_batch(routes)
    return restore_routes_shape(batch.copy(), was_1d)


__all__: list[str] = ["LocalSearchNoOp"]
