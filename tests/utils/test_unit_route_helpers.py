"""CPU-only unit tests for the route normalization helpers in `utils.route_helpers`.

Covers shape normalization, host-batch conversion, shape restoration and device
detection. Every test is hermetic: synthetic routes and no CUDA device access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from utils.route_helpers import (
  ensure_routes_batch,
  is_gpu_array,
  restore_routes_shape,
  to_cpu_batch,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  from _pytest.mark.structures import ParameterSet

VALID_ROUTE_CASES: list[ParameterSet] = [
  pytest.param(np.arange(6, dtype=np.int32), (1, 6), True, id="1d-array"),
  pytest.param(np.arange(12, dtype=np.int32).reshape(3, 4), (3, 4), False, id="2d-batch"),
  pytest.param([0, 2, 1], (1, 3), True, id="1d-list"),
  pytest.param([[0, 1], [1, 0]], (2, 2), False, id="2d-list"),
  pytest.param(np.array([5], dtype=np.int32), (1, 1), True, id="single-city-route"),
]

INVALID_ROUTE_CASES: list[ParameterSet] = [
  pytest.param(np.array(7, dtype=np.int32), 0, id="0d-array"),
  pytest.param(7, 0, id="python-scalar"),
  pytest.param(np.zeros((2, 3, 4), dtype=np.int32), 3, id="3d-array"),
  pytest.param(np.zeros((2, 2, 2, 2), dtype=np.int32), 4, id="4d-array"),
]


@pytest.mark.parametrize("normalizer", [ensure_routes_batch, to_cpu_batch], ids=["ensure", "to_cpu"])
@pytest.mark.parametrize(("routes", "expected_shape", "expected_was_1d"), VALID_ROUTE_CASES)
def test_route_normalizers_accept_valid_shapes(
  normalizer: Callable[..., tuple[object, ...]],
  routes: object,
  expected_shape: tuple[int, int],
  expected_was_1d: bool,
) -> None:
  """WHAT: both normalizers applied to every valid 1D/2D route representation.

  WHY: callers mix single tours and populations, arrays and sequences.
  EXPECTED: a 2D array with the documented shape and the correct `was_1d` flag.
  """
  result = normalizer(routes)

  assert isinstance(result[0], np.ndarray)
  assert result[0].shape == expected_shape
  assert result[1] is expected_was_1d


@pytest.mark.parametrize("normalizer", [ensure_routes_batch, to_cpu_batch], ids=["ensure", "to_cpu"])
@pytest.mark.parametrize(("routes", "ndim"), INVALID_ROUTE_CASES)
def test_route_normalizers_reject_invalid_shapes(
  normalizer: Callable[..., tuple[object, ...]],
  routes: object,
  ndim: int,
) -> None:
  """WHAT: both normalizers applied to 0-d and 3-d+ inputs.

  WHY: the contract covers exactly 1D routes and 2D batches; other ranks are caller errors.
  EXPECTED: `ValueError` reporting the offending `ndim`.
  """
  with pytest.raises(ValueError, match=f"ndim={ndim}"):
    normalizer(routes)


def test_to_cpu_batch_reports_device_and_dtype() -> None:
  """WHAT: `to_cpu_batch` on a host 1D route.

  WHY: downstream kernels require an int32 host batch plus shape/device provenance.
  EXPECTED: `int32` dtype, `was_1d=True`, `was_gpu=False`, values preserved.
  """
  route = np.array([2, 0, 1], dtype=np.int64)

  batch, was_1d, was_gpu = to_cpu_batch(route)  # type: ignore[arg-type]  # runtime accepts any integer dtype; the stub narrows to int32

  assert batch.dtype == np.int32
  assert was_1d is True
  assert was_gpu is False
  assert batch.tolist() == [[2, 0, 1]]


def test_to_cpu_batch_copy_flag_controls_defensive_copy() -> None:
  """WHAT: `to_cpu_batch` with `copy=False` and `copy=True` on a host batch.

  WHY: the copy flag is the only defense against aliasing caller-owned routes.
  EXPECTED: the default aliases the input; `copy=True` returns independent memory.
  """
  routes = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)

  shared, _, _ = to_cpu_batch(routes)
  copies, _, _ = to_cpu_batch(routes, copy=True)

  assert np.shares_memory(shared, routes)
  assert not np.shares_memory(copies, routes)
  copies[0, 0] = 99
  assert routes[0, 0] == 0


def test_restore_routes_shape_roundtrips_1d_and_2d() -> None:
  """WHAT: normalize-then-restore for a single route and for a batch.

  WHY: restore is how helpers hand results back in the caller's original shape.
  EXPECTED: 1D input comes back 1D with equal values; 2D input stays 2D.
  """
  route = np.array([0, 3, 1, 2], dtype=np.int32)
  batch, was_1d = ensure_routes_batch(route)
  restored_route = restore_routes_shape(batch, was_1d)

  routes = np.array([[0, 1], [1, 0]], dtype=np.int32)
  batch_2d, was_1d_2d = ensure_routes_batch(routes)
  restored_batch = restore_routes_shape(batch_2d, was_1d_2d)

  assert restored_route.ndim == 1
  assert np.array_equal(restored_route, route)
  assert restored_batch.shape == (2, 2)
  assert np.array_equal(restored_batch, routes)


@pytest.mark.parametrize(
  "value",
  [
    pytest.param(np.arange(3), id="numpy-array"),
    pytest.param([1, 2, 3], id="python-list"),
    pytest.param(np.int32(5), id="numpy-scalar"),
  ],
)
def test_is_gpu_array_rejects_host_values(value: object) -> None:
  """WHAT: `is_gpu_array` on CPU-side values.

  WHY: device detection gates every transfer path and must never report host data as GPU.
  EXPECTED: `False` for numpy arrays, lists, and scalars.
  """
  assert is_gpu_array(value) is False
