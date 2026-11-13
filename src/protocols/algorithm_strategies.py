"""
Algorithm strategy protocols for modular evolutionary routing architecture.

This module defines protocol interfaces (PEP 544) for interchangeable GA pipeline operators:
- `InitializationStrategy`: Generates initial route population `(population_size, n_cities)`.
- `SelectionStrategy`: Selects parent route(s) based on population fitness.
- `CrossoverStrategy`: Recombines route pairs or full batches into offspring.
- `MutationStrategy`: Mutates route(s) with configured mutation rate.
- `LocalSearchStrategy`: Refines route(s) via neighborhood search (e.g., 2-opt).
- `EvaluationStrategy`: Evaluates tour costs for single routes or batches.

These protocols enable runtime operator selection via dependency injection,
supporting the compositional genetic algorithm architecture across CPU and GPU backends.

Design Pattern References:
    - Strategy Pattern (GoF Design Patterns)
    - Dependency Injection (Fowler, 2004)
    - Protocol-Based Programming (PEP 544)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import (
  DistanceMatrix,
  Fitness,
  LocalSearchStage,
  Route,
  RouteOrRoutes,
  Routes,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from .problem_context import ProblemContext

__all__: list[str] = [
  "CrossoverStrategy",
  "DistanceMatrix",
  "EvaluationStrategy",
  "Fitness",
  "InitializationStrategy",
  "LocalSearchStage",
  "LocalSearchStrategy",
  "MutationStrategy",
  "Route",
  "RouteOrRoutes",
  "Routes",
  "SelectionStrategy",
]


# ==============================================================================
# MODULAR EVOLUTIONARY OPERATOR PROTOCOLS (ROUTE / BATCH COMPOSABLE)
# ==============================================================================


@runtime_checkable
class InitializationStrategy(Protocol):
  """Protocol for generating an initial batch of routes `(population_size, n_cities)`."""

  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext,
  ) -> Routes:
    """Generate `population_size` initial routes over `customers`."""
    ...


@runtime_checkable
class SelectionStrategy(Protocol):
  """Protocol for parent selection operating on a population of routes and fitness costs."""

  def select(
    self,
    population: Routes,
    fitness: Fitness,
    context: ProblemContext | None = None,
  ) -> Route:
    """Select a single parent route `(n_cities,)` from `population`."""
    ...

  def select_batch(
    self,
    routes: Routes,
    fitness: Fitness,
    num_parents: int | None = None,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Select a batch of parent routes `(num_parents, n_cities)` from `routes`."""
    ...


@runtime_checkable
class CrossoverStrategy(Protocol):
  """Protocol for route recombination accepting a pair of routes or a 2D batch of routes."""

  def crossover(
    self,
    parent1: Route,
    parent2: Route,
    context: ProblemContext | None = None,
  ) -> Route:
    """Recombine two 1D parent routes into a single 1D child route."""
    ...

  def crossover_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Recombine a batch of parent routes `(N, n_cities)` into `(N, n_cities)` offspring."""
    ...


@runtime_checkable
class MutationStrategy(Protocol):
  """Protocol for route mutation accepting a single route `(n_cities,)` or batch `(N, n_cities)`."""

  def mutate(
    self,
    individual: Route,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> Route:
    """Mutate a single 1D route with probability `mutation_rate`."""
    ...

  def mutate_batch(
    self,
    routes: RouteOrRoutes,
    mutation_rate: float,
    context: ProblemContext | None = None,
  ) -> RouteOrRoutes:
    """Mutate a 1D route or 2D batch of routes `(N, n_cities)` with probability `mutation_rate`."""
    ...


@runtime_checkable
class LocalSearchStrategy(Protocol):
  """
  Protocol for local search refinement (e.g., 2-opt) on a route or batch of routes.

  Because `improve_batch` accepts `RouteOrRoutes` (`(n_cities,)` or `(N, n_cities)`),
  the exact same strategy instance can be placed:
  - After crossover (`post_crossover`)
  - After mutation (`post_mutation`)
  - After fitness evaluation (`post_evaluation`)
  - Only once at the end of evolution on the best route or final population (`post_evolution`)
  """

  def improve(
    self,
    route: Route,
    context: ProblemContext,
  ) -> Route:
    """Refine a single 1D route `(n_cities,)`."""
    ...

  def improve_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
  ) -> RouteOrRoutes:
    """Refine a 1D route `(n_cities,)` or 2D batch of routes `(N, n_cities)`."""
    ...


@runtime_checkable
class EvaluationStrategy(Protocol):
  """Protocol for evaluating route cost(s) on a single route or a 2D batch of routes."""

  def evaluate(
    self,
    route: Route,
    context: ProblemContext,
  ) -> float:
    """Compute total tour cost for a single 1D route."""
    ...

  def evaluate_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
  ) -> Fitness:
    """Compute 1D cost array `(N,)` for a 1D route or 2D batch of routes `(N, n_cities)`."""
    ...
