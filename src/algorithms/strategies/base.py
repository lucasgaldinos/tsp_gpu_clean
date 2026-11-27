"""
Shared base class for every modular strategy slot.

The genetic algorithm composes six interchangeable slots (initialization, selection,
crossover, mutation, local search, evaluation). Each slot reports its own cost, so the
telemetry counters and the reproducible RNG contract live here once instead of being
copy-pasted into seventeen strategy classes (`DP11-1`).

References:
    - Strategy Pattern (GoF, 1994) and Dependency Injection (Fowler, 2004): the slots
      are interchangeable because they share one narrow base contract.
    - NumPy `Generator` seeding contract: `make_rng` accepts an integer, a
      `SeedSequence`, an existing `Generator`, a zero-argument factory or `None`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from protocols.vrp_strategies import (
  RouteAssessment,
  assess_from_evaluate_batch,
  canonicalize_default_route,
)
from utils.rng import make_rng

if TYPE_CHECKING:
  from typing import Any

  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes, SeedLike


class StrategyBase:
  """
  Common telemetry counters and RNG plumbing for the six strategy slots.

  Attributes:
      seed: The seed the strategy was built with (`None` for an unseeded draw).
      rng: The `numpy.random.Generator` derived from `seed` via `make_rng`.
      h2d_bytes: Host-to-device bytes moved by this strategy instance.
      d2h_bytes: Device-to-host bytes moved by this strategy instance.
      kernel_launches: CUDA kernel launches performed by this strategy instance.
      last_elapsed_ms: Duration of the last `@time_strategy` call, in milliseconds.
      total_elapsed_ms: Sum of every `@time_strategy` call of this instance.
      call_count: Number of `@time_strategy` calls seen by this instance.
  """

  # Declared, never assigned at class level: the counters below must stay instance
  # attributes, otherwise the class value shadows the instance slot and
  # `reset_metrics()` can disagree with what the strategy actually reports (`BUG-27`).
  last_elapsed_ms: float
  total_elapsed_ms: float
  call_count: int

  def __init__(self, seed: SeedLike = None) -> None:
    """
    Initialize the shared strategy state.

    Args:
        seed: Optional random seed override (int, `SeedSequence`, existing
            `numpy.random.Generator`, zero-argument callable, or `None`). Strategies
            without a stochastic step ignore the resulting generator.
    """
    self.seed: SeedLike = seed
    self.rng: Generator = make_rng(seed)
    self.h2d_bytes: int = 0
    self.d2h_bytes: int = 0
    self.kernel_launches: int = 0
    self.last_elapsed_ms = 0.0
    self.total_elapsed_ms = 0.0
    self.call_count = 0

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0

  def warmup(self, context: ProblemContext) -> None:
    """Compile and warm up CUDA kernels without mutating telemetry or RNG state (`DP11-12`, `BUG-115`).

    Template Method wrapping `_warmup_impl(context)` in a `try ... finally` guard that
    snapshots and restores `self` transfer/launch/timer counters, `self.rng`, and
    `context`'s transfer counters and `rng.bit_generator.state`.
    """
    prev_h2d = self.h2d_bytes
    prev_d2h = self.d2h_bytes
    prev_launches = self.kernel_launches
    prev_last_ms = self.last_elapsed_ms
    prev_total_ms = self.total_elapsed_ms
    prev_calls = self.call_count
    prev_self_rng = copy.deepcopy(self.rng.bit_generator.state) if hasattr(self.rng, "bit_generator") else None

    prev_ctx_h2d = getattr(context, "h2d_bytes", None)
    ctx_rng = getattr(context, "rng", None)
    prev_ctx_rng = (
      copy.deepcopy(ctx_rng.bit_generator.state) if ctx_rng is not None and hasattr(ctx_rng, "bit_generator") else None
    )

    try:
      self._warmup_impl(context)
    finally:
      self.h2d_bytes = prev_h2d
      self.d2h_bytes = prev_d2h
      self.kernel_launches = prev_launches
      self.last_elapsed_ms = prev_last_ms
      self.total_elapsed_ms = prev_total_ms
      self.call_count = prev_calls
      if prev_self_rng is not None and hasattr(self.rng, "bit_generator"):
        self.rng.bit_generator.state = prev_self_rng
      if prev_ctx_h2d is not None:
        context.h2d_bytes = prev_ctx_h2d
      if prev_ctx_rng is not None and ctx_rng is not None and hasattr(ctx_rng, "bit_generator"):
        ctx_rng.bit_generator.state = prev_ctx_rng

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Subclasses override to compile and launch warm-up kernels; default compiles `_get_kernel()`."""
    del context
    compile_kernel = getattr(self, "_get_kernel", None)
    if callable(compile_kernel):
      compile_kernel()

  def canonicalize_route(self, route: Route, context: ProblemContext) -> Route:
    """Normalize a single 1D winning route into canonical host representation (`DP17-8`)."""
    del context
    return canonicalize_default_route(route)

  def assess_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool = False,
  ) -> RouteAssessment:
    """Default `EvaluationStrategy.assess_batch` adapter wrapping `evaluate_batch` (`DP17-8`)."""
    return assess_from_evaluate_batch(self, routes, context, return_gpu=return_gpu)

  @staticmethod
  def _warmup_batch(context: ProblemContext, rows: int = 2) -> Any:
    """Return a `(rows, n)` batch of valid permutations on the context backend.

    A launch-based warm-up must never feed degenerate rows (`cp.zeros`): identical zero
    rows make `route[0] == route[-1]` read as a closed topology and poison every
    heuristic that inspects the sample (`BUG-24`). `arange` is a real permutation whose
    endpoints differ for `n > 1`.

    Args:
        context: Run context supplying the backend (`context.xp`) and the dimension `n`.
        rows: Number of routes; kernels that recombine pairs need at least 2.
    """
    xp: Any = context.xp
    return xp.tile(xp.arange(context.dimension, dtype=xp.int32), (rows, 1))


__all__: list[str] = ["StrategyBase"]
