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
  from numpy.random import Generator

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
  """

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

  def reset_metrics(self) -> None:
    """Reset transfer and kernel launch counters."""
    self.h2d_bytes = 0
    self.d2h_bytes = 0
    self.kernel_launches = 0


__all__: list[str] = ["StrategyBase"]
