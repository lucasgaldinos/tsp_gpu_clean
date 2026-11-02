"""Utility helpers for GPU kernel loading, strategy timing, RNG resolution, and route batches."""

from .gpu_helpers import load_kernel
from .phase_timer import PhaseTimer, time_strategy
from .rng import make_rng, resolve_rng
from .route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  to_cpu_batch,
)

__all__: list[str] = [
  "PhaseTimer",
  "ensure_routes_batch",
  "is_gpu_array",
  "load_kernel",
  "make_rng",
  "resolve_rng",
  "restore_routes_shape",
  "time_strategy",
  "to_cpu_batch",
]

