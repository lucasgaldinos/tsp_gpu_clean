"""
Selection strategies for evolutionary routing algorithms.

All selection strategies support both:
- `select(population, fitness, context=None) -> Route` (single 1D route `(n_cities,)`)
- `select_batch(routes, fitness, num_parents=None, context=None) -> Routes` (2D batch `(N, n_cities)`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from protocols.algorithm_strategies import Fitness, Route, Routes, ensure_routes_batch
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext


class TournamentSelection:
  """
  Tournament selection strategy for single-route and batch parent selection.

  Selects the individual with the lowest route cost among `k` randomly chosen
  competitors per tournament.
  """

  def __init__(self, k: int = 3, seed: int | None = None) -> None:
    """
    Initialize tournament selection.

    Args:
        k: Tournament size (number of competing routes per selection).
        seed: Optional RNG seed.
    """
    if k < 1:
      raise ValueError(f"Tournament size k must be >= 1, got {k}")
    self.k = k
    self.rng = np.random.default_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def select(
    self,
    population: Routes,
    fitness: Fitness,
    context: ProblemContext | None = None,
  ) -> Route:
    """
    Select a single winning parent route `(n_cities,)` via `k`-way tournament.

    Args:
        population: 2D routes array `(pop_size, n_cities)` or list of routes.
        fitness: 1D cost array `(pop_size,)` (lower cost is better).
        context: Optional `ProblemContext`.

    Returns:
        1D winning route `(n_cities,)`.
    """
    pop_batch, _ = ensure_routes_batch(population)
    fit_arr = np.asarray(fitness, dtype=np.float64)
    pop_size = len(pop_batch)
    k_eff = min(self.k, pop_size)

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    indices = rng.choice(pop_size, size=k_eff, replace=False)
    winner_local = int(np.argmin(fit_arr[indices]))
    winner_idx = int(indices[winner_local])
    return cast("Route", pop_batch[winner_idx].copy())

  @time_strategy
  def select_batch(
    self,
    routes: Routes,
    fitness: Fitness,
    num_parents: int | None = None,
    context: ProblemContext | None = None,
  ) -> Routes:
    """
    Select a 2D batch of winning parent routes `(num_parents, n_cities)`.

    Args:
        routes: 2D population routes array `(pop_size, n_cities)`.
        fitness: 1D cost array `(pop_size,)` (lower cost is better).
        num_parents: Number of parents to select (defaults to `len(routes)`).
        context: Optional `ProblemContext`.

    Returns:
        2D array of selected parent routes `(num_parents, n_cities)`.
    """
    pop_batch, _ = ensure_routes_batch(routes)
    pop_size = len(pop_batch)
    n_select = pop_size if num_parents is None else num_parents

    is_gpu = CUPY_AVAILABLE and cp is not None and isinstance(pop_batch, cp.ndarray)
    xp: Any = cp if is_gpu else np

    fit_arr = xp.asarray(fitness, dtype=xp.float64)
    if is_gpu:
      candidates = cp.random.randint(0, pop_size, size=(n_select, self.k), dtype=cp.int32)
      cand_fitness = fit_arr[candidates]
      best_cols = cp.argmin(cand_fitness, axis=1)
      winners = candidates[cp.arange(n_select), best_cols]
      return cast("Routes", pop_batch[winners].copy())

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng

    candidates_np = rng.integers(0, pop_size, size=(n_select, self.k))
    cand_fitness_np = fit_arr[candidates_np]
    best_cols_np = np.argmin(cand_fitness_np, axis=1)
    winners_np = candidates_np[np.arange(n_select), best_cols_np]
    return pop_batch[winners_np].copy()


class RouletteWheelSelection:
  """
  Fitness-proportionate (roulette wheel) selection for route minimization.

  Converts route costs `c_i` into selection weights `w_i = 1 / (c_i + eps)` so
  shorter tours receive higher selection probability.
  """

  def __init__(self, seed: int | None = None) -> None:
    """Initialize roulette wheel selection."""
    self.rng = np.random.default_rng(seed)
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
    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng
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
    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng
    indices = rng.choice(len(pop_batch), size=n_select, replace=True, p=probs)
    return pop_batch[indices].copy()


class GPUTournamentSelection:
  """
  Modular GPU CUDA kernel tournament selection strategy (`tournament_selection.cu`).

  Executes parent selection directly in GPU VRAM using a dedicated CUDA kernel.
  """

  def __init__(self, k: int = 5, seed: int | None = None) -> None:
    """
    Initialize GPU tournament selection.

    Args:
        k: Tournament size.
        seed: Optional RNG seed.
    """
    if k < 1:
      raise ValueError(f"Tournament size k must be >= 1, got {k}")
    self.k = k
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self._kernel: Any = None
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def _get_kernel(self) -> Any:
    """Lazily compile and return the `tournament_selection_kernel` CUDA kernel."""
    if self._kernel is None:
      self._kernel = load_kernel(
        "tournament_selection.cu",
        "tournament_selection_kernel",
        __file__,
      )
    return self._kernel

  def select(
    self,
    population: Routes,
    fitness: Fitness,
    context: ProblemContext | None = None,
  ) -> Route:
    """Select a single parent route `(n_cities,)`."""
    batch = self.select_batch(population, fitness, num_parents=1, context=context)
    return cast("Route", batch[0])

  @time_strategy
  def select_batch(
    self,
    routes: Routes,
    fitness: Fitness,
    num_parents: int | None = None,
    context: ProblemContext | None = None,
  ) -> Routes:
    """
    Select `num_parents` routes `(num_parents, n_cities)` on GPU via CUDA kernel.
    """
    if not CUPY_AVAILABLE or cp is None:
      fallback = TournamentSelection(k=self.k, seed=self.seed)
      return fallback.select_batch(routes, fitness, num_parents=num_parents, context=context)

    pop_batch, _ = ensure_routes_batch(routes)
    pop_size, n_cities = pop_batch.shape
    n_select = pop_size if num_parents is None else num_parents

    input_on_gpu = isinstance(pop_batch, cp.ndarray)
    if input_on_gpu:
      d_pop = cp.ascontiguousarray(pop_batch, dtype=cp.int32)
    else:
      d_pop = cp.asarray(pop_batch, dtype=cp.int32)
      self.h2d_bytes += int(d_pop.nbytes)

    if isinstance(fitness, cp.ndarray):
      d_fit = cp.ascontiguousarray(fitness, dtype=cp.float64)
    else:
      d_fit = cp.asarray(fitness, dtype=cp.float64)
      self.h2d_bytes += int(d_fit.nbytes)

    rng = self.rng
    if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
      rng = context.rng
    candidates_np = rng.integers(0, pop_size, size=(n_select, self.k), dtype=np.int32)
    d_candidates = cp.asarray(candidates_np, dtype=cp.int32)
    self.h2d_bytes += int(d_candidates.nbytes)

    d_parents = cp.empty((n_select, n_cities), dtype=cp.int32)

    threads_per_block = 128
    blocks = (n_select + threads_per_block - 1) // threads_per_block
    kernel = self._get_kernel()
    kernel(
      (blocks,),
      (threads_per_block,),
      (
        d_pop,
        d_fit,
        d_candidates,
        d_parents,
        np.int32(pop_size),
        np.int32(n_select),
        np.int32(n_cities),
        np.int32(self.k),
      ),
    )
    cp.cuda.Stream.null.synchronize()
    self.kernel_launches += 1

    if input_on_gpu:
      return cast("Routes", d_parents)
    self.d2h_bytes += int(d_parents.nbytes)
    return cast("Routes", cp.asnumpy(d_parents))
