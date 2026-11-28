"""Synchronization contract for the batch GPU 2-opt strategies.

`LocalSearchGPUTourTwoOpt.improve_batch` in both grid modes (`batch` and `multi_move`)
must not issue
an explicit ``cp.cuda.Stream.null.synchronize()`` (``BUG-15`` / ``DP10-18``): a GPU-resident
result stays stream-ordered for the caller's next CuPy operation, and the host-returning
branch synchronizes implicitly inside ``d_population.get()``. The tests intercept the
device layer with a counting proxy, so no real synchronization can happen unnoticed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms.strategies import local_search
from algorithms.strategies.local_search import LocalSearchGPUTourTwoOpt
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext
from utils import phase_timer

cp = cupy_backend()

if TYPE_CHECKING:
  from algorithms.strategies.local_search import TwoOptGPUMode
  from data_models.problem import Problem

_BATCH_SIZE: int = 3
_TOUR_LENGTH: int = 8


def _cupy() -> Any:
  """Return the importable CuPy module, skipping the test when it is unavailable."""
  if cp is None:
    pytest.skip("CuPy is not importable")
  return cp


class _CountingNullStream:
  """Wrap the real null stream and count `synchronize()` calls."""

  def __init__(self, real_stream: Any, counter: dict[str, int]) -> None:
    self._real_stream = real_stream
    self._counter = counter

  def synchronize(self) -> None:
    self._counter["null_stream_synchronize"] += 1
    self._real_stream.synchronize()


class _CountingEvent:
  """Wrap a real CUDA event and count `synchronize()` calls."""

  def __init__(self, real_event: Any, counter: dict[str, int]) -> None:
    self._real_event = real_event
    self._counter = counter

  def record(self) -> None:
    self._real_event.record()

  def synchronize(self) -> None:
    self._counter["event_synchronize"] += 1
    self._real_event.synchronize()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._real_event, name)


class _CountingCuda:
  """Expose `Stream.null` and `Event` through counting wrappers."""

  def __init__(self, real_cuda: Any, counter: dict[str, int]) -> None:
    self._real_cuda = real_cuda
    self._counter = counter
    self.Stream = self

  @property
  def null(self) -> _CountingNullStream:
    return _CountingNullStream(self._real_cuda.Stream.null, self._counter)

  def Event(self) -> _CountingEvent:
    self._counter["cuda_events"] += 1
    return _CountingEvent(self._real_cuda.Event(), self._counter)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._real_cuda, name)


class _CountingBackend:
  """Proxy the CuPy module while counting null-stream synchronizations."""

  def __init__(self, real_backend: Any, counter: dict[str, int]) -> None:
    self._real_backend = real_backend
    self._counter = counter
    self.cuda = _CountingCuda(real_backend.cuda, counter)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._real_backend, name)


@pytest.fixture
def sync_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
  """Make the GPU seams return a synchronization-counting proxy.

  The strategy resolves the device through `require_gpu()` instead of a module-level
  CuPy import (`DP9-2`), and `@time_strategy` resolves the event backend through
  `cupy_backend()`; both seams are injected with the same counting proxy, so neither a
  null-stream barrier nor a CUDA event can happen unnoticed.
  """
  counter: dict[str, int] = {
    "null_stream_synchronize": 0,
    "cuda_events": 0,
    "event_synchronize": 0,
  }
  counting_backend = _CountingBackend(_cupy(), counter)
  monkeypatch.setattr(local_search.two_opt, "require_gpu", lambda: counting_backend)
  monkeypatch.setattr(phase_timer, "cupy_backend", lambda: counting_backend)
  return counter


@pytest.fixture
def device_context(berlin52_problem: Problem) -> ProblemContext:
  """Return a CuPy-backed context for the berlin52 fixture."""
  return ProblemContext(berlin52_problem, xp=_cupy(), seed=17)


@pytest.fixture
def route_batch() -> np.ndarray:
  """Return a deterministic host batch of three valid permutations."""
  rng = np.random.default_rng(5)
  base = np.arange(_TOUR_LENGTH, dtype=np.int32)
  return np.stack([rng.permutation(base) for _ in range(_BATCH_SIZE)]).astype(np.int32, copy=False)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["batch", "multi_move"])
def test_gpu_resident_batch_does_not_synchronize(
  mode: TwoOptGPUMode,
  device_context: ProblemContext,
  route_batch: np.ndarray,
  sync_counter: dict[str, int],
) -> None:
  """WHAT: `improve_batch` fed a CuPy-resident population.

  WHY: a host-device barrier per call costs roughly 12% of a generation on small
  instances while adding nothing: the returned device array is consumed by
  stream-ordered CuPy work.

  EXPECTED: zero null-stream synchronizations and a GPU-resident batch with every tour
  still a permutation of ``0..n-1``.
  """
  # Arrange
  cupy: Any = _cupy()
  device_batch = cupy.asarray(route_batch, dtype=cupy.int32)
  strategy = LocalSearchGPUTourTwoOpt(max_iterations=2, mode=mode)

  # Act
  improved = strategy.improve_batch(device_batch, context=device_context)

  # Assert
  assert sync_counter["null_stream_synchronize"] == 0
  assert isinstance(improved, cupy.ndarray)
  improved_host = cupy.asnumpy(improved)
  assert improved_host.shape == route_batch.shape
  for tour in improved_host:
    assert sorted(int(gene) for gene in tour) == list(range(_TOUR_LENGTH))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["batch", "multi_move", "naive"])
def test_time_strategy_enqueues_without_device_events(
  mode: TwoOptGPUMode,
  device_context: ProblemContext,
  route_batch: np.ndarray,
  sync_counter: dict[str, int],
) -> None:
  """WHAT: `improve_batch` decorated by `@time_strategy` on a CuPy-resident batch.

  WHY: the decorator created a CUDA event pair and called `end_evt.synchronize()` on any
  call that merely received a GPU array, reintroducing the 5-6 hidden host-device barriers
  per generation that `DP10-18`, `SWAP-A`, `EVAL-A`, `SEL-A` and `INV-A` removed
  (`BUG-26`). With the barrier in place the host could never run ahead of the device.

  EXPECTED: zero CUDA events, zero event synchronizations, and the host clock still
  records a duration for the call.
  """
  # Arrange
  cupy: Any = _cupy()
  device_batch = cupy.asarray(route_batch, dtype=cupy.int32)
  strategy = LocalSearchGPUTourTwoOpt(max_iterations=1, mode=mode)

  # Act
  improved = strategy.improve_batch(device_batch, context=device_context)

  # Assert
  assert sync_counter["cuda_events"] == 0
  assert sync_counter["event_synchronize"] == 0
  assert sync_counter["null_stream_synchronize"] == 0
  assert strategy.last_elapsed_ms >= 0.0
  assert isinstance(improved, cupy.ndarray)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["batch", "multi_move"])
def test_host_batch_returns_improved_routes_without_explicit_sync(
  mode: TwoOptGPUMode,
  device_context: ProblemContext,
  route_batch: np.ndarray,
  sync_counter: dict[str, int],
) -> None:
  """WHAT: `improve_batch` fed a NumPy population (the D2H-returning branch).

  WHY: that branch copies the result back with ``.get()``, which synchronizes by
  itself; an extra barrier is pure overhead and no improvement may be silently dropped.

  EXPECTED: zero null-stream synchronizations, a host array of the input shape, valid
  permutations, and a closed-tour cost no worse than the input population.
  """
  # Arrange
  strategy = LocalSearchGPUTourTwoOpt(max_iterations=2, mode=mode)
  distances = device_context.get_cpu_distances()

  def closed_cost(tour: np.ndarray) -> int:
    return int(sum(distances[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))

  # Act
  improved = strategy.improve_batch(route_batch, context=device_context)

  # Assert
  assert sync_counter["null_stream_synchronize"] == 0
  assert isinstance(improved, np.ndarray)
  assert improved.shape == route_batch.shape
  for before, after in zip(route_batch, improved, strict=True):
    assert sorted(int(gene) for gene in after) == list(range(_TOUR_LENGTH))
    assert closed_cost(after) <= closed_cost(before)
