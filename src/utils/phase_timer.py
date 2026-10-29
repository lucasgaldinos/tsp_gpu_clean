"""
Non-intrusive strategy and phase timing instrumentation.

Provides `@time_strategy` and `PhaseTimer` for recording per-phase execution
durations in milliseconds using CUDA hardware events (`cp.cuda.Event`) when
available and `time.perf_counter()` on CPU.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Callable

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None


class PhaseTimer:
  """Accumulates per-phase execution times (in milliseconds) across generations."""

  def __init__(self, enabled: bool = True, use_cuda_events: bool = False) -> None:
    """
    Initialize phase timer.

    Args:
        enabled: Whether timing measurements are active.
        use_cuda_events: Whether to synchronize and measure with CUDA events.
    """
    self.enabled = enabled
    self.use_cuda_events = use_cuda_events and CUPY_AVAILABLE
    self._totals_ms: dict[str, float] = defaultdict(float)
    self._counts: dict[str, int] = defaultdict(int)

  def record(self, phase_name: str, elapsed_ms: float) -> None:
    """Record a single phase execution duration in milliseconds."""
    if not self.enabled:
      return
    self._totals_ms[phase_name] += elapsed_ms
    self._counts[phase_name] += 1

  def measure[**P, R](self, phase_name: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Execute `func(*args, **kwargs)` and record its duration under `phase_name`."""
    if not self.enabled:
      return func(*args, **kwargs)

    elapsed_ms: float
    if self.use_cuda_events and cp is not None:
      start_evt = cp.cuda.Event()
      end_evt = cp.cuda.Event()
      start_evt.record()
      result: R = func(*args, **kwargs)
      end_evt.record()
      end_evt.synchronize()
      elapsed_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))
    else:
      t0: float = time.perf_counter()
      result = func(*args, **kwargs)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0

    self.record(phase_name, elapsed_ms)
    return result

  def summary(self) -> dict[str, dict[str, float | int]]:
    """Return summary dictionary with total_ms, calls, and mean_ms per phase."""
    out: dict[str, dict[str, float | int]] = {}
    for phase, total_ms in self._totals_ms.items():
      calls: int = self._counts[phase]
      out[phase] = {
        "total_ms": total_ms,
        "calls": calls,
        "mean_ms": total_ms / calls if calls > 0 else 0.0,
      }
    return out

  def reset(self) -> None:
    """Clear all recorded phase timings."""
    self._totals_ms.clear()
    self._counts.clear()


def time_strategy[**P, R](func: Callable[P, R]) -> Callable[P, R]:
  """
  Decorator for strategy methods that measures execution time.

  Stores the last execution duration (in ms) on `self.last_elapsed_ms` and
  accumulates total time on `self.total_elapsed_ms` when `self` is a strategy
  instance. Uses `cp.cuda.Event` if any argument is a CuPy array or if
  `getattr(self, "use_gpu_events", False)` is True.
  """

  @functools.wraps(func)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    self_obj: Any = args[0] if args else None
    use_cuda = False
    if CUPY_AVAILABLE and cp is not None:
      if getattr(self_obj, "use_gpu_events", False):
        use_cuda = True
      else:
        for arg in args[1:]:
          if isinstance(arg, cp.ndarray):
            use_cuda = True
            break

    elapsed_ms: float
    if use_cuda and cp is not None:
      try:
        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()
        start_evt.record()
        result: R = func(*args, **kwargs)
        end_evt.record()
        end_evt.synchronize()
        elapsed_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))
      except Exception:
        t0: float = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    else:
      t0 = time.perf_counter()
      result = func(*args, **kwargs)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if self_obj is not None and hasattr(self_obj, "__dict__"):
      self_obj.last_elapsed_ms = elapsed_ms
      self_obj.total_elapsed_ms = getattr(self_obj, "total_elapsed_ms", 0.0) + elapsed_ms
      self_obj.call_count = getattr(self_obj, "call_count", 0) + 1

    return result

  return wrapper
