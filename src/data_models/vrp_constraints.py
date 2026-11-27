"""Immutable global VRP constraints descriptor (VRP Tier 1 active, VRP Tier 2 reserved).

Following Vidal, Crainic, Gendreau & Prins (2014), ``VRPConstraints`` centralizes
vehicle-routing constraints on ``Problem`` and ``ProblemContext`` so evaluators, split
decoders, and local search operators read one validated contract instead of branching on
ad-hoc problem-type flags (``DP17-1``, ``DP17-3``).

VRP Tier 1 (active — symmetric prefix-additive constraints):
    - ``capacity``: homogeneous vehicle capacity :math:`Q > 0`.
    - ``max_vehicles``: optional fleet size upper bound :math:`K_{\\max} > 0`.
    - ``max_route_distance``: optional per-route distance/duration limit :math:`L_{\\max} > 0`.
    - ``return_to_depot``: whether routes close at the depot (``True`` for CVRP/DCVRP,
      ``False`` for OVRP / Open TSP).

VRP Tier 2 (reserved — asymmetric time-warp monoids for VRPTW/SDVRP without schema migration):
    - ``time_windows``: optional read-only C-contiguous ``float64`` array of shape ``(n, 2)``.
    - ``service_times``: optional read-only C-contiguous ``float64`` array of shape ``(n,)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


def _validate_optional_positive_int(name: str, value: Any) -> None:
  """Validate that an optional scalar is a strict positive ``int`` (rejecting ``bool``)."""
  if value is None:
    return
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{name} must be a positive int, got {type(value).__name__}: {value!r}")
  if value <= 0:
    raise ValueError(f"{name} must be positive, got {value}")


def _validate_optional_positive_float(name: str, value: Any) -> float | None:
  """Validate that an optional scalar is a finite positive real number and return ``float``."""
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite positive number, got {type(value).__name__}: {value!r}")
  as_float = float(value)
  if not math.isfinite(as_float):
    raise ValueError(f"{name} must be finite (no NaN or Inf), got {value!r}")
  if as_float <= 0.0:
    raise ValueError(f"{name} must be positive, got {value}")
  return as_float


def _coerce_readonly_float64(name: str, value: Any) -> NDArray[np.float64]:
  """Convert a real numeric array-like to a read-only C-contiguous ``float64`` copy."""
  raw = np.asarray(value)
  if raw.dtype == np.dtype(bool) or not (
    np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)
  ):
    raise ValueError(f"{name} must have a real numeric dtype, got {raw.dtype}")
  arr: NDArray[np.float64] = np.array(raw, dtype=np.float64, order="C", copy=True)
  if not np.isfinite(arr).all():
    raise ValueError(f"{name} must contain only finite values (no NaN or Inf)")
  arr.flags.writeable = False
  return arr


def _arrays_equal(first: NDArray[np.float64] | None, second: NDArray[np.float64] | None) -> bool:
  """Compare two optional NumPy arrays by content."""
  if first is None or second is None:
    return first is second
  return bool(np.array_equal(first, second))


def _array_hash_key(value: NDArray[np.float64] | None) -> tuple[tuple[int, ...], str, bytes] | None:
  """Build a hashable content key for an optional NumPy array."""
  if value is None:
    return None
  return value.shape, value.dtype.str, value.tobytes()


@dataclass(frozen=True, slots=True)
class VRPConstraints:
  """Immutable descriptor of VRP Tier 1 and reserved VRP Tier 2 constraints (Vidal et al., 2014).

  Attributes:
      capacity: Optional homogeneous vehicle capacity (strict ``int > 0``).
      max_vehicles: Optional maximum fleet size (strict ``int > 0``).
      max_route_distance: Optional maximum route distance/duration (finite ``float > 0``).
      return_to_depot: Whether vehicles return to the depot at route end.
      time_windows: Reserved VRP Tier 2 array of ``[earliest, latest]`` windows, shape ``(n, 2)``.
      service_times: Reserved VRP Tier 2 array of non-negative node service times, shape ``(n,)``.
  """

  # VRP Tier 1 (active)
  capacity: int | None = None
  max_vehicles: int | None = None
  max_route_distance: float | None = None
  return_to_depot: bool = True

  # VRP Tier 2 (reserved)
  time_windows: NDArray[np.float64] | None = None
  service_times: NDArray[np.float64] | None = None

  def __post_init__(self) -> None:
    """Validate scalar bounds, normalize optional arrays to read-only ``float64``, and check consistency."""
    _validate_optional_positive_int("capacity", self.capacity)
    _validate_optional_positive_int("max_vehicles", self.max_vehicles)

    normalized_dist = _validate_optional_positive_float("max_route_distance", self.max_route_distance)
    if normalized_dist is not None:
      object.__setattr__(self, "max_route_distance", normalized_dist)

    if not isinstance(self.return_to_depot, bool):
      raise ValueError(f"return_to_depot must be a bool, got {type(self.return_to_depot).__name__}")

    tw: NDArray[np.float64] | None = None
    if self.time_windows is not None:
      tw = _coerce_readonly_float64("time_windows", self.time_windows)
      if tw.ndim != 2 or tw.shape[1] != 2 or tw.shape[0] <= 0:
        raise ValueError(f"time_windows must have shape (n, 2) with n > 0, got {tw.shape}")
      if not bool(np.all(tw[:, 0] <= tw[:, 1])):
        raise ValueError("time_windows must satisfy earliest <= latest for every node")
      object.__setattr__(self, "time_windows", tw)

    st: NDArray[np.float64] | None = None
    if self.service_times is not None:
      st = _coerce_readonly_float64("service_times", self.service_times)
      if st.ndim != 1 or st.shape[0] <= 0:
        raise ValueError(f"service_times must have shape (n,) with n > 0, got {st.shape}")
      if not bool(np.all(st >= 0.0)):
        raise ValueError("service_times must be non-negative (>= 0.0) for every node")
      object.__setattr__(self, "service_times", st)

    if tw is not None and st is not None and tw.shape[0] != st.shape[0]:
      raise ValueError(
        f"time_windows length ({tw.shape[0]}) must match service_times length ({st.shape[0]})"
      )

  @property
  def is_constrained(self) -> bool:
    """Return ``True`` if any VRP Tier 1 or VRP Tier 2 constraint differs from an unconstrained closed tour."""
    return (
      self.capacity is not None
      or self.max_vehicles is not None
      or self.max_route_distance is not None
      or not self.return_to_depot
      or self.time_windows is not None
      or self.service_times is not None
    )

  def validate_against_dimension(self, dimension: int) -> None:
    """Verify that any node-indexed VRP Tier 2 arrays match ``dimension``.

    Args:
        dimension: Total number of nodes in the associated ``Problem``.

    Raises:
        ValueError: If ``dimension <= 0`` or an array's leading axis differs from ``dimension``.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
      raise ValueError(f"dimension must be a positive int, got {dimension!r}")
    if self.time_windows is not None and self.time_windows.shape[0] != dimension:
      raise ValueError(
        f"time_windows.shape should be ({dimension}, 2), got {self.time_windows.shape}"
      )
    if self.service_times is not None and self.service_times.shape[0] != dimension:
      raise ValueError(
        f"service_times.shape should be ({dimension},), got {self.service_times.shape}"
      )

  def __eq__(self, other: object) -> bool:
    """Compare constraints by content with array-safe NumPy equality."""
    if not isinstance(other, VRPConstraints):
      return NotImplemented
    return (
      self.capacity == other.capacity
      and self.max_vehicles == other.max_vehicles
      and self.max_route_distance == other.max_route_distance
      and self.return_to_depot == other.return_to_depot
      and _arrays_equal(self.time_windows, other.time_windows)
      and _arrays_equal(self.service_times, other.service_times)
    )

  def __hash__(self) -> int:
    """Hash constraint content so frozen instances stay usable in sets and dict keys."""
    return hash(
      (
        self.capacity,
        self.max_vehicles,
        self.max_route_distance,
        self.return_to_depot,
        _array_hash_key(self.time_windows),
        _array_hash_key(self.service_times),
      )
    )


__all__ = ["VRPConstraints"]
