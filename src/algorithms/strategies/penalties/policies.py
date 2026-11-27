"""VRP constraint-violation penalty policies (`DP17-9`).

Provides four composable :class:`protocols.vrp_strategies.PenaltyPolicy`
implementations derived from :class:`algorithms.strategies.base.StrategyBase`:

- :class:`InstanceScaledPenaltyPolicy`: stateless scale-invariant weights
  ``omega_q = safety_factor * D_max + 1.0``, ``omega_l = safety_factor``, and
  ``omega_k = safety_factor * D_max * max(1, n - 1)``.
- :class:`StrictFeasibilityPolicy`: assigns ``+inf`` to any violating route
  without multiplying ``0.0 * inf`` (IEEE-754 NaN-safe).
- :class:`FixedPenaltyPolicy`: constant non-negative multipliers.
- :class:`AdaptivePenaltyPolicy`: starts from instance-scaled base weights and
  adapts dynamically around the ``epsilon_tol`` feasibility corridor, resetting
  its dynamic multiplier on :meth:`reset_metrics`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.strategies.base import StrategyBase
from protocols.backend import cupy_backend, require_gpu
from utils.route_helpers import is_gpu_array

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import SeedLike
  from protocols.vrp_strategies import RouteAssessment

# Bounded O(n^2)-avoiding cache keyed by `id(problem)` with structural metadata guard.
_D_MAX_CACHE: dict[int, tuple[str, int, str, float]] = {}


def _validate_float_param(name: str, value: Any, *, allow_zero: bool = False) -> float:
  """Validate a finite numeric parameter (rejecting ``bool``)."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number, got {type(value).__name__}: {value!r}")
  num = float(value)
  if not math.isfinite(num):
    raise ValueError(f"{name} must be finite (no NaN or Inf), got {value!r}")
  if allow_zero:
    if num < 0.0:
      raise ValueError(f"{name} must be >= 0, got {value!r}")
  elif num <= 0.0:
    raise ValueError(f"{name} must be > 0, got {value!r}")
  return num


def _compute_instance_d_max(context: ProblemContext) -> float:
  """Compute or upper-bound :math:`D_{\\max} = \\max_{u,v} d(u, v)` without unnecessary allocation."""
  problem = context.problem
  cache_key = id(problem)
  signature = (problem.name, problem.dimension, problem.edge_type)
  cached = _D_MAX_CACHE.get(cache_key)
  if cached is not None and cached[:3] == signature:
    return cached[3]

  if problem.distances is not None or context.dimension <= context.max_matrix_dimension:
    dist_matrix = context.get_cpu_distances()
    d_max = float(np.max(dist_matrix)) if dist_matrix.size > 0 else 0.0
  else:
    coords = context.get_cpu_coordinates()
    if coords is None or coords.size == 0:
      d_max = 0.0
    elif problem.edge_type == "GEO":
      # Great-circle spherical diameter upper bound in km without O(n^2) matrix allocation.
      d_max = float(math.ceil(math.pi * 6378.388))
    else:
      span = np.max(coords, axis=0) - np.min(coords, axis=0)
      d_max = float(math.ceil(float(np.linalg.norm(span))))

  if not math.isfinite(d_max) or d_max < 0.0:
    d_max = 0.0

  if len(_D_MAX_CACHE) >= 512:
    _D_MAX_CACHE.clear()
  _D_MAX_CACHE[cache_key] = (*signature, d_max)
  return d_max


def _prepare_apply_inputs(
  travel_costs: Any,
  raw_violation_components: Any,
) -> tuple[Any, Any, Any]:
  """Normalize ``(N,)`` travel costs and ``(N, 3)`` violation components on their shared backend."""
  on_gpu = is_gpu_array(travel_costs) or is_gpu_array(raw_violation_components)
  xp: Any = require_gpu() if on_gpu else np
  travel = xp.asarray(travel_costs, dtype=xp.float64)
  comps = xp.asarray(raw_violation_components, dtype=xp.float64)
  if comps.ndim != 2 or comps.shape[1] < 3:
    raise ValueError(f"raw_violation_components must have shape (N, 3), got {comps.shape}")
  if travel.ndim != 1 or travel.shape[0] != comps.shape[0]:
    raise ValueError(f"travel_costs must have shape ({comps.shape[0]},), got {travel.shape}")
  return xp, travel, comps


class InstanceScaledPenaltyPolicy(StrategyBase):
  """Stateless scale-invariant VRP penalty policy anchored to :math:`D_{\\max}` (`DP17-9`)."""

  def __init__(self, safety_factor: float = 2.0, seed: SeedLike = None) -> None:
    super().__init__(seed=seed)
    self.safety_factor: float = _validate_float_param("safety_factor", safety_factor, allow_zero=False)

  def weights(self, context: ProblemContext) -> tuple[float, float, float]:
    """Return ``(omega_q, omega_l, omega_k)`` scaled proportionally to instance distances."""
    d_max = _compute_instance_d_max(context)
    n_cust = max(1, context.dimension - 1)
    omega_q = self.safety_factor * d_max + 1.0
    omega_l = self.safety_factor
    omega_k = self.safety_factor * d_max * float(n_cust)
    return omega_q, omega_l, omega_k

  def apply(
    self,
    travel_costs: Any,
    raw_violation_components: Any,
    context: ProblemContext,
  ) -> Any:
    """Compute penalized objectives while preserving array backend."""
    _, travel, comps = _prepare_apply_inputs(travel_costs, raw_violation_components)
    omega_q, omega_l, omega_k = self.weights(context)
    return travel + omega_q * comps[:, 0] + omega_l * comps[:, 1] + omega_k * comps[:, 2]


class StrictFeasibilityPolicy(StrategyBase):
  """Hard-feasibility policy assigning ``+inf`` to any violating individual (`DP17-9`)."""

  def __init__(self, seed: SeedLike = None) -> None:
    super().__init__(seed=seed)

  def weights(self, context: ProblemContext) -> tuple[float, float, float]:
    """Return ``(inf, inf, inf)`` for interface compatibility with weight consumers."""
    _ = context
    return (math.inf, math.inf, math.inf)

  def apply(
    self,
    travel_costs: Any,
    raw_violation_components: Any,
    context: ProblemContext,
  ) -> Any:
    """Return pure travel cost when feasible and ``+inf`` when violated (zero ``0.0 * inf`` NaN risk)."""
    _ = context
    xp, travel, comps = _prepare_apply_inputs(travel_costs, raw_violation_components)
    has_violation = (comps[:, 0] > 0.0) | (comps[:, 1] > 0.0) | (comps[:, 2] > 0.0)
    return xp.where(has_violation, xp.inf, travel)


class FixedPenaltyPolicy(StrategyBase):
  """Static constant penalty multipliers ``(penalty_capacity, penalty_distance, penalty_vehicles)``.

  Any axis left as ``None`` falls back to :class:`InstanceScaledPenaltyPolicy` for that axis
  only (`BUG-118`); when all three axes are explicit, :meth:`weights` returns in :math:`O(1)`.
  """

  def __init__(
    self,
    penalty_capacity: float | None = None,
    penalty_distance: float | None = None,
    penalty_vehicles: float | None = None,
    seed: SeedLike = None,
  ) -> None:
    super().__init__(seed=seed)
    self.penalty_capacity: float | None = (
      None if penalty_capacity is None else _validate_float_param("penalty_capacity", penalty_capacity, allow_zero=True)
    )
    self.penalty_distance: float | None = (
      None if penalty_distance is None else _validate_float_param("penalty_distance", penalty_distance, allow_zero=True)
    )
    self.penalty_vehicles: float | None = (
      None if penalty_vehicles is None else _validate_float_param("penalty_vehicles", penalty_vehicles, allow_zero=True)
    )
    self._fallback_policy: InstanceScaledPenaltyPolicy | None = (
      None
      if (self.penalty_capacity is not None and self.penalty_distance is not None and self.penalty_vehicles is not None)
      else InstanceScaledPenaltyPolicy(seed=seed)
    )

  def weights(self, context: ProblemContext) -> tuple[float, float, float]:
    """Return fixed non-negative penalty multipliers, resolving ``None`` axes via instance scaling."""
    if self.penalty_capacity is not None and self.penalty_distance is not None and self.penalty_vehicles is not None:
      return (self.penalty_capacity, self.penalty_distance, self.penalty_vehicles)
    fallback = self._fallback_policy or InstanceScaledPenaltyPolicy()
    base_q, base_l, base_k = fallback.weights(context)
    return (
      base_q if self.penalty_capacity is None else self.penalty_capacity,
      base_l if self.penalty_distance is None else self.penalty_distance,
      base_k if self.penalty_vehicles is None else self.penalty_vehicles,
    )

  def apply(
    self,
    travel_costs: Any,
    raw_violation_components: Any,
    context: ProblemContext,
  ) -> Any:
    """Compute penalized objectives using fixed penalty weights."""
    _, travel, comps = _prepare_apply_inputs(travel_costs, raw_violation_components)
    omega_q, omega_l, omega_k = self.weights(context)
    return travel + omega_q * comps[:, 0] + omega_l * comps[:, 1] + omega_k * comps[:, 2]


class AdaptivePenaltyPolicy(StrategyBase):
  """Self-calibrating penalty policy starting at instance-scaled weights (`DP17-9`)."""

  def __init__(
    self,
    epsilon_tol: float = 0.10,
    up_factor: float = 1.2,
    down_factor: float = 0.85,
    target_feasible_low: float = 0.20,
    target_feasible_high: float = 0.80,
    min_scale: float = 0.1,
    max_scale: float = 100.0,
    safety_factor: float = 2.0,
    seed: SeedLike = None,
  ) -> None:
    super().__init__(seed=seed)
    self.epsilon_tol: float = _validate_float_param("epsilon_tol", epsilon_tol, allow_zero=True)
    self.up_factor: float = _validate_float_param("up_factor", up_factor, allow_zero=False)
    self.down_factor: float = _validate_float_param("down_factor", down_factor, allow_zero=False)
    if self.up_factor < 1.0:
      raise ValueError(f"up_factor must be >= 1.0, got {up_factor!r}")
    if self.down_factor > 1.0:
      raise ValueError(f"down_factor must be <= 1.0, got {down_factor!r}")

    self.target_feasible_low: float = _validate_float_param("target_feasible_low", target_feasible_low, allow_zero=True)
    self.target_feasible_high: float = _validate_float_param(
      "target_feasible_high", target_feasible_high, allow_zero=True
    )
    if not (0.0 <= self.target_feasible_low <= self.target_feasible_high <= 1.0):
      raise ValueError(
        "target_feasible_low and target_feasible_high must satisfy 0 <= low <= high <= 1, "
        f"got low={target_feasible_low!r}, high={target_feasible_high!r}"
      )

    self.min_scale: float = _validate_float_param("min_scale", min_scale, allow_zero=False)
    self.max_scale: float = _validate_float_param("max_scale", max_scale, allow_zero=False)
    if self.min_scale > self.max_scale:
      raise ValueError(f"min_scale ({min_scale!r}) must be <= max_scale ({max_scale!r})")

    self.safety_factor: float = _validate_float_param("safety_factor", safety_factor, allow_zero=False)
    self._base_policy = InstanceScaledPenaltyPolicy(safety_factor=self.safety_factor, seed=seed)
    self._scale: float = 1.0

  def reset_metrics(self) -> None:
    """Reset transfer counters and dynamic adaptive penalty multiplier scale."""
    super().reset_metrics()
    self._scale = 1.0

  def weights(self, context: ProblemContext) -> tuple[float, float, float]:
    """Return active penalty multipliers ``self._scale * base_weights(context)``."""
    base_q, base_l, base_k = self._base_policy.weights(context)
    return (base_q * self._scale, base_l * self._scale, base_k * self._scale)

  def apply(
    self,
    travel_costs: Any,
    raw_violation_components: Any,
    context: ProblemContext,
  ) -> Any:
    """Compute penalized objectives using current dynamic weights."""
    _, travel, comps = _prepare_apply_inputs(travel_costs, raw_violation_components)
    omega_q, omega_l, omega_k = self.weights(context)
    return travel + omega_q * comps[:, 0] + omega_l * comps[:, 1] + omega_k * comps[:, 2]

  def adapt(self, assessment: RouteAssessment, context: ProblemContext) -> bool:
    """Adjust dynamic penalty multiplier scale around the feasibility/violation corridor."""
    _ = context
    xp: Any = cupy_backend() if is_gpu_array(assessment.is_feasible) else np
    if int(assessment.is_feasible.size) == 0:
      return False

    feasible_frac = float(xp.mean(assessment.is_feasible.astype(xp.float64)))
    mean_norm_viol = float(xp.mean(assessment.normalized_violations.astype(xp.float64)))

    if feasible_frac < self.target_feasible_low or mean_norm_viol > self.epsilon_tol:
      new_scale = min(self.max_scale, self._scale * self.up_factor)
    elif feasible_frac > self.target_feasible_high:
      new_scale = max(self.min_scale, self._scale * self.down_factor)
    else:
      return False

    changed = not math.isclose(new_scale, self._scale, rel_tol=1e-12, abs_tol=1e-15)
    self._scale = new_scale
    return changed


__all__: list[str] = [
  "AdaptivePenaltyPolicy",
  "FixedPenaltyPolicy",
  "InstanceScaledPenaltyPolicy",
  "StrictFeasibilityPolicy",
]
