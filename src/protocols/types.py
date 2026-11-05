"""
Standardized domain type aliases for routing algorithms.

Defines standardized data types for tours, batches, fitness arrays, and distance matrices
using PEP 695 type aliases and numpy.typing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

type SeedLike = int | Sequence[int] | Generator | Callable[[], Generator] | None
"""Random seed integer, integer seed sequence, existing `np.random.Generator`, zero-arg callable factory, or `None`."""

type Route = NDArray[np.int32]
"""Single tour represented as a 1D array of node indices with shape `(n_cities,)`."""

type Routes = NDArray[np.int32]
"""Batch/population of tours represented as a 2D array with shape `(N, n_cities)`."""

type RouteOrRoutes = Route | Routes
"""Either a single 1D `Route` `(n_cities,)` or a 2D `Routes` batch `(N, n_cities)`."""

type Fitness = NDArray[np.float64]
"""1D array of tour costs with shape `(N,)` (lower is better for TSP minimization)."""

type DistanceMatrix = NDArray[np.float64]
"""2D pairwise distance matrix with shape `(n_cities, n_cities)`."""

type LocalSearchStage = Literal[
  "post_crossover",
  "post_mutation",
  "post_evaluation",
  "post_evolution",
  "none",
]
"""
Pipeline stage in the evolutionary algorithm where `LocalSearchStrategy` is executed:
- `"post_crossover"`: immediately after crossover
- `"post_mutation"`: after mutation (canonical memetic GA default)
- `"post_evaluation"`: after fitness evaluation & survival selection each generation
- `"post_evolution"`: after all generations complete, applied to the final population
- `"none"`: disabled
"""

__all__: list[str] = [
  "DistanceMatrix",
  "Fitness",
  "LocalSearchStage",
  "Route",
  "RouteOrRoutes",
  "Routes",
  "SeedLike",
]
