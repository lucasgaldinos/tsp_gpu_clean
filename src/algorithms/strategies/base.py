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

from typing import TYPE_CHECKING

from utils.rng import make_rng

if TYPE_CHECKING:
  from typing import Any

  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import SeedLike


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
    """Compile this strategy's CUDA kernels ahead of the measured loop (`DP11-12`, `BUG-25`).

    The hook exists on *every* strategy so the warm-up protocol never has to guess
    which slot owns a kernel: CPU strategies keep the no-op body, GPU strategies
    override it to compile *and* launch their kernel once (see `_warmup_batch`).
    The default here only forces compilation through `_get_kernel()`, the single path
    that reaches `_compile_kernel_cached`, so even a strategy that has not overridden
    the hook cannot leave a JIT compilation for the measured repetitions.

    Args:
        context: The run's `ProblemContext`; unused by the compile-only default.
    """
    del context
    compile_kernel = getattr(self, "_get_kernel", None)
    if callable(compile_kernel):
      compile_kernel()

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
