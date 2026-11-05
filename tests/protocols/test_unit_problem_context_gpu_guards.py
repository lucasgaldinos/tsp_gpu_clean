"""Unit tests for the `ProblemContext` GPU accessor guards.

The accessors resolve the device through `require_gpu()` rather than `self.xp`
(`DP10-17`), so a NumPy-preferred context still serves GPU kernels and a host without
CUDA fails with `GPUUnavailableError`. CPU accessor contracts live in
`test_unit_problem_context.py`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from data_models.exceptions import GPUUnavailableError, VRAMInsufficientError
from protocols.backend import cupy_backend, gpu_available
from protocols.problem_context import ProblemContext

GPU_AVAILABLE = gpu_available()
cp = cupy_backend()

if TYPE_CHECKING:
  from data_models.problem import Problem


class _FakeDevice:
  """Minimal `cuda.Device` stand-in exposing the `mem_info` tuple."""

  def __init__(self, free_bytes: int) -> None:
    self._free_bytes = free_bytes

  @property
  def mem_info(self) -> tuple[int, int]:
    """Return `(free_bytes, total_bytes)` like CuPy's device object."""
    return self._free_bytes, 4 * self._free_bytes


class _FakeCuda:
  """Minimal `xp.cuda` stand-in returning one fake device."""

  def __init__(self, free_bytes: int) -> None:
    self._device = _FakeDevice(free_bytes)

  def Device(self) -> _FakeDevice:
    """Return the fake device."""
    return self._device


class _FakeBackend:
  """Minimal `xp` stand-in exposing only `cuda`."""

  def __init__(self, free_bytes: int) -> None:
    self.cuda: _FakeCuda = _FakeCuda(free_bytes)


def _dense_bytes(n: int, pop_size: int, components: int) -> int:
  """Return the `DP10-19` dense-mode estimate for `n`, `pop_size` and `d` components."""
  return 4 * n * n + 12 * pop_size * n + 24 * pop_size + 8 * components * n


def _on_the_fly_bytes(n: int, pop_size: int, components: int) -> int:
  """Return the `DP10-19` coordinate-mode estimate (no `4n^2` matrix term)."""
  return 12 * pop_size * n + 24 * pop_size + 8 * components * n


@pytest.mark.parametrize("method_name", ["get_gpu_distances", "get_gpu_coordinates", "get_gpu_demands"])
@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
def test_gpu_accessors_on_numpy_backend_use_the_device(
  request: pytest.FixtureRequest,
  fixture_name: str,
  method_name: str,
) -> None:
  """WHAT: GPU-only accessors invoked on a context created with `xp=np`.

  WHY: `DP10-17` resolves the device through `require_gpu()` instead of `self.xp`, so a
  hybrid CPU+GPU pipeline can keep a NumPy-preferred context and still hand device
  buffers to GPU kernels; without CUDA the failure must stay typed.

  EXPECTED: a CuPy array (or `None` demands) when a device is reachable, otherwise
  `GPUUnavailableError` - never the old `ValueError`.
  """
  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=np, seed=5)

  if not GPU_AVAILABLE or cp is None:
    with pytest.raises(GPUUnavailableError):
      getattr(context, method_name)()
    return

  # Act
  value = getattr(context, method_name)()

  # Assert
  if method_name == "get_gpu_demands":
    assert value is None
  else:
    assert isinstance(value, cp.ndarray)


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA GPU not available")
@pytest.mark.parametrize("fixture_name", ["burma14_problem", "berlin52_problem"])
@pytest.mark.parametrize("cpu_first", [False, True], ids=["gpu-first", "cpu-first"])
def test_gpu_accessors_match_cpu_backend(
  request: pytest.FixtureRequest,
  fixture_name: str,
  cpu_first: bool,
) -> None:
  """WHAT: GPU accessors on a real device, exercising both transfer orders.

  WHY: Lazy transfer may start from the problem data or from an existing CPU cache;
  both paths must produce exactly the CPU matrix and coordinates.

  EXPECTED: CuPy arrays with CPU-identical values and shapes, `None` demands for TSP
  fixtures, and cached device arrays on repeated calls.
  """
  if cp is None:
    pytest.skip("CuPy is not importable")

  # Arrange
  problem: Problem = request.getfixturevalue(fixture_name)
  context = ProblemContext(problem, xp=cp, seed=9)

  if cpu_first:
    cpu_distances = context.get_cpu_distances()
    cpu_coordinates = context.get_cpu_coordinates()

  # Act
  gpu_distances = context.get_gpu_distances()
  gpu_coordinates = context.get_gpu_coordinates()
  gpu_demands = context.get_gpu_demands()

  if not cpu_first:
    cpu_distances = context.get_cpu_distances()
    cpu_coordinates = context.get_cpu_coordinates()

  # Assert
  assert isinstance(gpu_distances, cp.ndarray)
  assert isinstance(gpu_coordinates, cp.ndarray)
  assert gpu_distances.dtype == cp.int32
  assert gpu_distances.shape == (problem.dimension, problem.dimension)
  assert gpu_coordinates.shape == (problem.dimension, 2)
  assert cpu_coordinates is not None
  np.testing.assert_array_equal(cp.asnumpy(gpu_distances), cpu_distances)
  np.testing.assert_array_equal(cp.asnumpy(gpu_coordinates), cpu_coordinates)
  assert gpu_demands is None
  assert isinstance(cpu_distances, np.ndarray)
  assert not isinstance(cpu_distances, cp.ndarray)
  assert context.get_gpu_distances() is gpu_distances


@pytest.mark.parametrize("pop_size", [0, 1, 8, 210])
def test_estimate_vram_bytes_uses_dense_matrix_formula(
  berlin52_problem: Problem,
  pop_size: int,
) -> None:
  """WHAT: `estimate_vram_bytes` on a problem that keeps the dense matrix.

  WHY: Below `max_matrix_dimension` the solver allocates the `int32` matrix
  (`4n^2`) plus population, offspring, fitness and coordinate buffers (`DP10-19`).

  EXPECTED: Exactly `4n^2 + 12Pn + 24P + 8dn` bytes with `d=2` for berlin52.
  """
  # Arrange
  n = berlin52_problem.dimension
  context = ProblemContext(berlin52_problem, xp=np, seed=3)

  # Act
  required_bytes = context.estimate_vram_bytes(pop_size)

  # Assert
  assert context.uses_on_the_fly_distances(n) is False
  assert required_bytes == _dense_bytes(n, pop_size, 2)


@pytest.mark.parametrize("pop_size", [0, 5])
def test_estimate_vram_bytes_drops_matrix_term_on_the_fly(
  berlin52_problem: Problem,
  pop_size: int,
) -> None:
  """WHAT: `estimate_vram_bytes` when the threshold forces coordinate mode.

  WHY: On-the-fly distances never materialize `(n, n)`, so charging `4n^2`
  would reject populations that actually fit (`DP10-19`).

  EXPECTED: Exactly `12Pn + 24P + 8dn` bytes, smaller than the dense estimate.
  """
  # Arrange
  n = berlin52_problem.dimension
  context = ProblemContext(berlin52_problem, xp=np, seed=3, max_matrix_dimension=n - 1)

  # Act
  required_bytes = context.estimate_vram_bytes(pop_size)

  # Assert
  assert context.uses_on_the_fly_distances(n) is True
  assert required_bytes == _on_the_fly_bytes(n, pop_size, 2)
  assert required_bytes < _dense_bytes(n, pop_size, 2)


def test_estimate_vram_bytes_without_coordinates_uses_one_component(
  berlin52_problem: Problem,
) -> None:
  """WHAT: `estimate_vram_bytes` for an EXPLICIT problem without coordinates.

  WHY: The auxiliary term is `8dn`; with no 2-D coordinate array `d` falls back
  to `1`, so a bare `8n` is charged instead of an accidental zero.

  EXPECTED: `4n^2 + 12Pn + 24P + 8n` bytes.
  """
  # Arrange
  n = berlin52_problem.dimension
  explicit_problem = dataclasses.replace(
    berlin52_problem,
    edge_type="EXPLICIT",
    coordinates=None,
    distances=np.ones((n, n), dtype=np.int32),
  )
  context = ProblemContext(explicit_problem, xp=np, seed=3)

  # Act
  required_bytes = context.estimate_vram_bytes(4)

  # Assert
  assert required_bytes == _dense_bytes(n, 4, 1)


def test_estimate_vram_bytes_rejects_negative_population(berlin52_problem: Problem) -> None:
  """WHAT: `estimate_vram_bytes` with an invalid population size.

  WHY: A negative population would produce a negative byte count that silently
  passes the capacity comparison.

  EXPECTED: `ValueError` naming `pop_size`.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=3)

  with pytest.raises(ValueError, match="pop_size must be non-negative"):
    context.estimate_vram_bytes(-1)


def test_check_vram_capacity_raises_with_capacity_attributes(berlin52_problem: Problem) -> None:
  """WHAT: `check_vram_capacity` with simulated insufficient VRAM.

  WHY: The guard must surface a typed error carrying the estimate and the budget
  instead of letting the driver abort the process.

  EXPECTED: `VRAMInsufficientError` whose `required_bytes` is the estimate and
  whose `available_bytes` is the injected budget, with both rendered in `str()`.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=3)
  required_bytes = context.estimate_vram_bytes(16)
  available_bytes = required_bytes - 1

  # Act / Assert
  with pytest.raises(VRAMInsufficientError) as exc_info:
    context.check_vram_capacity(16, available_bytes=available_bytes)

  error = exc_info.value
  assert error.required_bytes == required_bytes
  assert error.available_bytes == available_bytes
  assert error.args[0] == f"Insufficient VRAM for population evaluation (pop_size=16, n={context.dimension})"
  assert f"available: {available_bytes / 1e9:.2f}GB" in str(error)


def test_check_vram_capacity_accepts_exact_budget(berlin52_problem: Problem) -> None:
  """WHAT: `check_vram_capacity` when the estimate equals the free VRAM.

  WHY: The comparison is `required > available`; an exact fit must not be
  rejected, otherwise the reported requirement is one byte too strict.

  EXPECTED: No error, and the returned estimate equals `estimate_vram_bytes`.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=3)

  # Act
  returned_bytes = context.check_vram_capacity(16, available_bytes=context.estimate_vram_bytes(16))

  # Assert
  assert returned_bytes == context.estimate_vram_bytes(16)


def test_check_vram_capacity_numpy_backend_returns_estimate(berlin52_problem: Problem) -> None:
  """WHAT: `check_vram_capacity` with no injected budget on a NumPy-preferred context.

  WHY: The device is probed with `require_gpu()`; when no device exists there is no VRAM
  budget to compare against, so the check degrades to the estimate instead of touching
  CUDA or raising (`DP10-17`).

  EXPECTED: The estimate is returned and nothing is raised.
  """
  context = ProblemContext(berlin52_problem, xp=np, seed=3)

  assert context.check_vram_capacity(8) == context.estimate_vram_bytes(8)


def test_check_vram_capacity_reads_device_mem_info(
  berlin52_problem: Problem,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: `check_vram_capacity` querying `cuda.Device().mem_info` through the probe.

  WHY: The production path resolves the device with `require_gpu()` (not `self.xp`), so
  a NumPy-preferred context still validates the real budget (`DP10-17`).

  EXPECTED: The estimate is rejected when `mem_info` reports less free VRAM.
  """
  # Arrange
  context = ProblemContext(berlin52_problem, xp=np, seed=3)
  required_bytes = context.estimate_vram_bytes(32)
  free_bytes = required_bytes - 1
  monkeypatch.setattr("protocols.problem_context.require_gpu", lambda: _FakeBackend(free_bytes))

  # Act / Assert
  with pytest.raises(VRAMInsufficientError) as exc_info:
    context.check_vram_capacity(32)

  assert exc_info.value.required_bytes == required_bytes
  assert exc_info.value.available_bytes == free_bytes
