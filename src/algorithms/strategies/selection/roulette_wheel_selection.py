"""
Roulette Wheel (fitness-proportionate) selection strategy for routing algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from utils.phase_timer import time_strategy
from utils.rng import make_rng, resolve_rng
from utils.route_helpers import ensure_routes_batch

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Fitness, Route, Routes, SeedLike


class RouletteWheelSelection:
  """
  Fitness-proportionate (roulette wheel) selection for route minimization.

  Converts route costs `c_i` into selection weights `w_i = 1 / (c_i + eps)` so
  shorter tours receive higher selection probability.
  """

  def __init__(self, seed: SeedLike = None) -> None:
    """Initialize roulette wheel selection."""
    self.seed: SeedLike = seed
    self.rng: Generator = make_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _compute_probabilities(self, fitness: Fitness) -> np.ndarray:
    """Compute inverse-cost selection probabilities."""
    fit_np = np.asarray(fitness, dtype=np.float64)
    inv_fitness = 1.0 / (np.maximum(fit_np, 0.0) + 1e-6)
    total = float(np.sum(inv_fitness))
    if total <= 0.0 or not np.isfinite(total):
      return np.full(len(fit_np), 1.0 / len(fit_np), dtype=np.float64)
    return inv_fitness / total

  def select(
    self,
    population: Routes,
    fitness: Fitness,
    context: ProblemContext | None = None,
  ) -> Route:
    """Select a single parent route `(n_cities,)` using roulette wheel probabilities."""
    pop_batch, _ = ensure_routes_batch(population)
    probs = self._compute_probabilities(fitness)
    rng: Generator = resolve_rng(context, self.rng)
    idx = int(rng.choice(len(pop_batch), p=probs))
    return cast("Route", pop_batch[idx].copy())

  @time_strategy
  def select_batch(
    self,
    routes: Routes,
    fitness: Fitness,
    num_parents: int | None = None,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Select a 2D batch of parent routes `(num_parents, n_cities)` via roulette wheel."""
    pop_batch, _ = ensure_routes_batch(routes)
    n_select = len(pop_batch) if num_parents is None else num_parents
    probs = self._compute_probabilities(fitness)
    rng: Generator = resolve_rng(context, self.rng)
    indices = rng.choice(len(pop_batch), size=n_select, replace=True, p=probs)
    return pop_batch[indices].copy()


__all__: list[str] = ["RouletteWheelSelection"]

