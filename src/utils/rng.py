"""
Random number generator resolution and instantiation utilities.

Provides centralized helpers for reproducible stochastic operations across
genetic algorithms and metaheuristic strategies, supporting explicit seeds,
existing NumPy generators, and context-attached RNG instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from numpy.random import Generator

  from protocols.problem_context import ProblemContext
  from protocols.types import SeedLike


def make_rng(seed: SeedLike = None) -> Generator:
  """
  Construct or resolve a `np.random.Generator` from an int, existing Generator, or callable.

  Args:
      seed: Integer seed, existing `np.random.Generator`, zero-argument callable returning
          a `np.random.Generator`, or `None`.

  Returns:
      Configured `np.random.Generator` instance.
  """
  if isinstance(seed, np.random.Generator):
    return seed
  if callable(seed):
    rng = seed()
    if not isinstance(rng, np.random.Generator):
      raise TypeError(f"Callable seed factory must return np.random.Generator, got {type(rng)!r}")
    return rng
  return np.random.default_rng(seed)


def resolve_rng(
  context: ProblemContext | None = None,
  fallback: SeedLike = None,
) -> Generator:
  """
  Resolve the active `np.random.Generator` from `context.rng` if available, else `fallback`.

  Args:
      context: Optional `ProblemContext` that may carry a deterministic `context.rng`.
      fallback: Fallback `Generator`, callable factory, integer seed, or `None`.

  Returns:
      Active `np.random.Generator`.
  """
  if context is not None and hasattr(context, "rng") and isinstance(context.rng, np.random.Generator):
    return context.rng
  return make_rng(fallback)


__all__: list[str] = [
  "make_rng",
  "resolve_rng",
]
