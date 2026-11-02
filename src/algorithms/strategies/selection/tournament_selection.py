"""
Tournament Selection strategies for routing algorithms.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Tournament Selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import make_rng, resolve_rng
from utils.route_helpers import ensure_routes_batch

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Fitness, Route, Routes, SeedLike


class TournamentSelection:
  """
  Tournament selection strategy for single-route and batch parent selection.

  Selects the individual with the lowest route cost among `k` randomly chosen
  competitors per tournament.
  """

  def __init__(self, k: int = 3, seed: SeedLike = None) -> None:
    """
    Initialize tournament selection.

    Args:
        k: Tournament size (number of competing routes per selection).
        seed: Optional RNG seed (int, Generator, callable, or None).
    """
    if k < 1:
      raise ValueError(f"Tournament size k must be >= 1, got {k}")
    self.k: int = k
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

    rng: Generator = resolve_rng(context, self.rng)
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
    pop_size: int = len(pop_batch)
    n_select: int = pop_size if num_parents is None else num_parents

    is_gpu: bool = CUPY_AVAILABLE and cp is not None and isinstance(pop_batch, cp.ndarray)
    xp: Any = cp if is_gpu else np

    fit_arr = xp.asarray(fitness, dtype=xp.float64)
    if is_gpu:
      candidates = cp.random.randint(0, pop_size, size=(n_select, self.k), dtype=cp.int32)
      cand_fitness = fit_arr[candidates]
      best_cols = cp.argmin(cand_fitness, axis=1)
      winners = candidates[cp.arange(n_select), best_cols]
      return cast("Routes", pop_batch[winners].copy())

    rng: Generator = resolve_rng(context, self.rng)

    candidates_np = rng.integers(0, pop_size, size=(n_select, self.k))
    cand_fitness_np = fit_arr[candidates_np]
    best_cols_np = np.argmin(cand_fitness_np, axis=1)
    winners_np = candidates_np[np.arange(n_select), best_cols_np]
    return pop_batch[winners_np].copy()


class GPUTournamentSelection:
  """
  Modular GPU CUDA kernel tournament selection strategy (`tournament_selection.cu`).

  Executes parent selection directly in GPU VRAM using a dedicated CUDA kernel.
  """

  def __init__(self, k: int = 5, seed: SeedLike = None) -> None:
    """
    Initialize GPU tournament selection.

    Args:
        k: Tournament size.
        seed: Optional RNG seed (int, Generator, callable, or None).
    """
    if k < 1:
      raise ValueError(f"Tournament size k must be >= 1, got {k}")
    self.k: int = k
    self.seed: SeedLike = seed
    self.rng: Generator = make_rng(seed)
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

    input_on_gpu: bool = isinstance(pop_batch, cp.ndarray)
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

    rng: Generator = resolve_rng(context, self.rng)
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


__all__: list[str] = ["GPUTournamentSelection", "TournamentSelection"]

