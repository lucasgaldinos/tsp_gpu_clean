"""Polymorphic VRP protocols, assessment payload, and evaluator fallback helpers (`DP17-8`).

Defines:
- `RouteAssessment`: immutable batch evaluation payload carrying penalized
  objectives, pure TSPLIB95 travel costs, feasibility flags, scalar and
  dimensionless normalized violations, per-component excesses `(N, 3)`, and
  vehicle counts.
- `RouteDecoder`, `PenaltyPolicy`, and `BinPackingStrategy`: `@runtime_checkable`
  PEP 544 sub-strategy protocols for composable VRP decoding, penalty policies,
  and fleet bin packing.
- `assess_with_evaluator`, `assess_from_evaluate_batch`, `canonicalize_default_route`,
  and `canonicalize_with_evaluator`: safe adapters so bare custom `EvaluationStrategy`
  implementations (`DP16-3`) work identically alongside full VRP evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from protocols.backend import cupy_backend, require_gpu
from utils.route_helpers import ensure_routes_batch, is_gpu_array

if TYPE_CHECKING:
    from protocols.problem_context import ProblemContext
    from protocols.types import Route, RouteOrRoutes


@dataclass(frozen=True)
class RouteAssessment:
    """Immutable batch evaluation payload returned by `EvaluationStrategy.assess_batch`.

    Attributes:
        objectives: `(N,)` `float64` minimized fitness (penalized for VRP, pure travel cost for TSP/ATSP).
        travel_costs: `(N,)` `float64` pure unpenalized TSPLIB95 travel distance.
        is_feasible: `(N,)` `bool` feasibility flags.
        violations: `(N,)` `float64` raw scalar excess (or `0.0` when feasible/unconstrained).
        normalized_violations: `(N,)` `float64` dimensionless violation
            :math:`\\hat{\\nu} = \\nu_Q/Q + \\nu_L/L_{\\max} + \\nu_K/K_{\\max}`.
        raw_violation_components: Optional `(N, 3)` `float64` array holding
            `[viol_q, viol_l, viol_k]` so adaptive penalty updates can refresh
            `objectives` in :math:`\\mathcal{O}(N)` without re-scanning routes.
        num_vehicles: Optional `(N,)` `int32` active vehicle counts (`1` for TSP/ATSP).
    """

    objectives: Any
    travel_costs: Any
    is_feasible: Any
    violations: Any
    normalized_violations: Any
    raw_violation_components: Any | None = None
    num_vehicles: Any | None = None


@runtime_checkable
class RouteDecoder(Protocol):
    """Protocol for decoding Giant-Tour `(N, n_cust)` permutations into `0`-delimited routes."""

    def decode_batch(
        self,
        giant_tours: RouteOrRoutes,
        context: ProblemContext,
        *,
        max_vehicles: int | None = None,
        return_gpu: bool | None = None,
    ) -> Any:
        """Decode Giant-Tour(s) into `0`-delimited routes and full cost/violation breakdown."""
        ...

    def decode_costs_only(
        self,
        giant_tours: RouteOrRoutes,
        context: ProblemContext,
        *,
        max_vehicles: int | None = None,
        return_gpu: bool | None = None,
    ) -> Any:
        """Compute cost/violation breakdown for Giant-Tour(s) without materializing routes."""
        ...


@runtime_checkable
class PenaltyPolicy(Protocol):
    """Protocol for VRP constraint-violation penalty weighting (`DP17-9`)."""

    def weights(self, context: ProblemContext) -> tuple[float, float, float]:
        """Return `(omega_q, omega_l, omega_k)` penalty multipliers for capacity, distance, and fleet."""
        ...

    def apply(
        self,
        travel_costs: Any,
        raw_violation_components: Any,
        context: ProblemContext,
    ) -> Any:
        """Compute `(N,)` `float64` penalized objectives from travel costs and `[viol_q, viol_l, viol_k]`."""
        ...


@runtime_checkable
class BinPackingStrategy(Protocol):
    """Protocol for fleet-repacking operators under hard vehicle limits (`DP17-9`)."""

    def pack_batch(
        self,
        routes: RouteOrRoutes,
        context: ProblemContext,
    ) -> RouteOrRoutes:
        """Repack a route or batch of routes to respect fleet constraints."""
        ...


def compute_normalized_violations(
    raw_violation_components: Any,
    context: ProblemContext,
) -> Any:
    """Compute dimensionless :math:`\\hat{\\nu} = \\nu_Q/Q + \\nu_L/L_{\\max} + \\nu_K/K_{\\max}`.

    Preserves the array backend (`np.ndarray` or `cp.ndarray`) with zero D2H traffic.
    """
    xp: Any = cupy_backend() if is_gpu_array(raw_violation_components) else np
    comps = xp.asarray(raw_violation_components, dtype=xp.float64)
    if comps.ndim != 2 or comps.shape[1] < 3:
        raise ValueError(
            f"raw_violation_components must have shape (N, 3), got {comps.shape}"
        )

    constraints = context.constraints
    norm = xp.zeros(comps.shape[0], dtype=xp.float64)
    if constraints.capacity is not None and constraints.capacity > 0:
        norm = norm + comps[:, 0] / float(constraints.capacity)
    if (
        constraints.max_route_distance is not None
        and constraints.max_route_distance > 0.0
    ):
        norm = norm + comps[:, 1] / float(constraints.max_route_distance)
    if constraints.max_vehicles is not None and constraints.max_vehicles > 0:
        norm = norm + comps[:, 2] / float(constraints.max_vehicles)
    return norm


def canonicalize_default_route(route: Route) -> Route:
    """Return an `int32` host copy of `route` without altering tour structure (TSP/ATSP default)."""
    if is_gpu_array(route):
        return np.asarray(require_gpu().asnumpy(route), dtype=np.int32).copy()
    return np.asarray(route, dtype=np.int32).copy()


def canonicalize_with_evaluator(
    evaluator: Any,
    route: Route,
    context: ProblemContext,
) -> Route:
    """Canonicalize `route` via `evaluator.canonicalize_route` when present, else host `int32` copy."""
    hook = getattr(evaluator, "canonicalize_route", None)
    if callable(hook):
        return np.asarray(hook(route, context), dtype=np.int32)
    return canonicalize_default_route(route)


def assess_from_evaluate_batch(
    evaluator: Any,
    routes: RouteOrRoutes,
    context: ProblemContext,
    *,
    return_gpu: bool = False,
) -> RouteAssessment:
    """Wrap a plain `evaluate_batch` result into a feasible `RouteAssessment`."""
    try:
        objectives = evaluator.evaluate_batch(routes, context, return_gpu=return_gpu)
    except TypeError:
        objectives = evaluator.evaluate_batch(routes, context)

    on_gpu = is_gpu_array(objectives)
    if return_gpu and not on_gpu:
        cp = require_gpu()
        objectives = cp.asarray(objectives, dtype=cp.float64)
        on_gpu = True
    elif not return_gpu and on_gpu:
        objectives = require_gpu().asnumpy(objectives).astype(np.float64, copy=False)
        on_gpu = False

    xp: Any = require_gpu() if on_gpu else np
    objectives = xp.asarray(objectives, dtype=xp.float64)
    n_rows = int(objectives.shape[0])
    return RouteAssessment(
        objectives=objectives,
        travel_costs=objectives.copy(),
        is_feasible=xp.ones(n_rows, dtype=bool),
        violations=xp.zeros(n_rows, dtype=xp.float64),
        normalized_violations=xp.zeros(n_rows, dtype=xp.float64),
        raw_violation_components=None,
        num_vehicles=xp.ones(n_rows, dtype=xp.int32),
    )


def assess_with_evaluator(
    evaluator: Any,
    routes: RouteOrRoutes,
    context: ProblemContext,
    *,
    return_gpu: bool = False,
) -> RouteAssessment:
    """Assess `routes` via `evaluator.assess_batch`, falling back to `evaluate_batch` for custom evaluators."""
    batch, _ = ensure_routes_batch(routes)
    hook = getattr(evaluator, "assess_batch", None)
    if callable(hook):
        try:
            result = hook(batch, context, return_gpu=return_gpu)
        except TypeError:
            result = hook(batch, context)
        if isinstance(result, RouteAssessment):
            return result
    return assess_from_evaluate_batch(evaluator, batch, context, return_gpu=return_gpu)


__all__: list[str] = [
    "BinPackingStrategy",
    "PenaltyPolicy",
    "RouteAssessment",
    "RouteDecoder",
    "assess_from_evaluate_batch",
    "assess_with_evaluator",
    "canonicalize_default_route",
    "canonicalize_with_evaluator",
    "compute_normalized_violations",
]
