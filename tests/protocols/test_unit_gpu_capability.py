"""Unit tests for the GPU capability probe in `protocols.backend`.

Verifies detection of missing CuPy installation, missing CUDA devices,
driver errors, working GPU setups, and probe result caching.
All tests run on CPU without requiring a GPU or CuPy.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
  from collections.abc import Generator

from data_models.exceptions import GPUUnavailableError
from protocols.backend import (
  clear_gpu_cache,
  cupy_backend,
  gpu_available,
  require_gpu,
)


@pytest.fixture(autouse=True)
def _reset_probe_cache() -> Generator[None, None, None]:
  """Ensure every test starts and ends with a clean GPU capability cache."""
  clear_gpu_cache()
  try:
    yield
  finally:
    clear_gpu_cache()


def test_cupy_not_installed_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: CuPy is not installed in the Python environment.

  WHY: The core library must fail fast with an actionable installation message
  when a GPU-dependent operation is requested.

  EXPECTED: `gpu_available()` is False, `cupy_backend()` is None, and
  `require_gpu()` raises `GPUUnavailableError` with install hint.
  """
  real_find_spec = importlib.util.find_spec

  def mock_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "cupy":
      return None
    return real_find_spec(name, *args, **kwargs)

  monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

  assert not gpu_available()
  assert cupy_backend() is None
  with pytest.raises(GPUUnavailableError, match="CuPy is not installed"):
    require_gpu()


def test_cuda_device_count_zero_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: CuPy is installed but reports 0 CUDA devices.

  WHY: Running on a machine with CuPy installed but no GPU or with driver disabled
  must not crash with an obscure C-level CUDA error.

  EXPECTED: `gpu_available()` is False, `cupy_backend()` is None, and
  `require_gpu()` raises `GPUUnavailableError` stating device unavailable.
  """
  fake_spec = MagicMock()
  fake_cp = MagicMock()
  fake_cp.cuda.runtime.getDeviceCount.return_value = 0

  monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: fake_spec if name == "cupy" else None)
  monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cp)

  assert not gpu_available()
  assert cupy_backend() is None
  with pytest.raises(GPUUnavailableError, match="CUDA device unavailable"):
    require_gpu()


def test_cuda_device_probe_exception_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: CuPy is installed but `getDeviceCount()` raises a driver error.

  WHY: Driver mismatch or CUDA runtime initialization failure must be caught
  and surfaced cleanly as `GPUUnavailableError`.

  EXPECTED: `gpu_available()` is False, `cupy_backend()` is None, and
  `require_gpu()` raises `GPUUnavailableError` stating device unavailable.
  """
  fake_spec = MagicMock()
  fake_cp = MagicMock()
  fake_cp.cuda.runtime.getDeviceCount.side_effect = RuntimeError("CUDA driver mismatch")

  monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: fake_spec if name == "cupy" else None)
  monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cp)

  assert not gpu_available()
  assert cupy_backend() is None
  with pytest.raises(GPUUnavailableError, match="CUDA device unavailable"):
    require_gpu()


def test_cuda_device_available_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: CuPy is installed and at least 1 CUDA device is present.

  WHY: Normal GPU-accelerated execution path must succeed and return the module.

  EXPECTED: `gpu_available()` is True, `cupy_backend()` returns the module,
  and `require_gpu()` returns the module without raising.
  """
  fake_spec = MagicMock()
  fake_cp = MagicMock()
  fake_cp.cuda.runtime.getDeviceCount.return_value = 1

  monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: fake_spec if name == "cupy" else None)
  monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cp)

  assert gpu_available()
  assert cupy_backend() is fake_cp
  assert require_gpu() is fake_cp


def test_probe_results_are_cached_until_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: Repeated calls to capability probe functions.

  WHY: Probing device count repeatedly introduces unnecessary driver overhead.
  The result must be cached until explicitly cleared.

  EXPECTED: Second call reuses cached verdict without re-probing device count;
  clearing the cache causes re-evaluation.
  """
  fake_spec = MagicMock()
  fake_cp = MagicMock()
  fake_cp.cuda.runtime.getDeviceCount.return_value = 1

  monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: fake_spec if name == "cupy" else None)
  monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cp)

  assert gpu_available()
  assert fake_cp.cuda.runtime.getDeviceCount.call_count == 1

  # Second call must hit cache
  assert gpu_available()
  assert fake_cp.cuda.runtime.getDeviceCount.call_count == 1

  # After clearing cache, next call probes again
  clear_gpu_cache()
  assert gpu_available()
  assert fake_cp.cuda.runtime.getDeviceCount.call_count == 2
