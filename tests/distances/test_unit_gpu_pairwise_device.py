"""Bit-for-bit CPU/GPU parity for `gpu_pairwise_distance` (`Task 10.15`, `DP10-19`).

The header `src/distances/kernels/pairwise_device.cuh` re-implements the TSPLIB95
integer arithmetic of :mod:`distances.pairwise` in device code. A one-ulp drift in
`nint`, `ceil`, `ATT` or the `GEO` degree truncation silently changes tour lengths
and 2-opt deltas, so the contract is exact equality against both the scalar
functions and :func:`distances.matrix.compute_distance_matrix`.

EXPECTED: for every one of the nine coordinate metrics, the dispatcher returns the
same `int32` matrix in matrix mode (`dist != nullptr`) and coordinate mode
(`dist == nullptr`), including the zero diagonal (`u == v`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from distances.matrix import compute_distance_matrix
from distances.pairwise import EDGE_TYPE_TO_CODE, get_distance_function
from utils.gpu_helpers import load_kernel

pytestmark = pytest.mark.gpu

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
HEADER_PATH: Path = REPO_ROOT / "src" / "distances" / "kernels" / "pairwise_device.cuh"

# Values deliberately off the ``.5`` rounding boundary so a one-ulp difference in
# the device math cannot flip an integer and mask a real arithmetic mismatch.
COORDS_2D = np.array(
  [[0.0, 0.0], [3.1, 4.0], [1.5, 2.5], [7.25, -2.75], [12.0, 0.5]],
  dtype=np.float64,
)
COORDS_3D = np.array(
  [[0.0, 0.0, 0.0], [1.5, 2.25, 3.0], [4.0, 6.0, 8.0], [-2.5, 1.0, 0.25]],
  dtype=np.float64,
)
# burma14 cities 1-4 in TSPLIB DDD.MM notation (degrees truncated, minutes in the fraction).
COORDS_GEO = np.array(
  [[16.47, 96.10], [16.47, 94.44], [20.09, 92.54], [22.22, 93.37]],
  dtype=np.float64,
)

THREE_DIMENSIONAL_TYPES: frozenset[str] = frozenset({"EUC_3D", "MAN_3D", "MAX_3D"})


def _coords_for(edge_weight_type: str) -> np.ndarray:
  """Return the coordinate set whose dimensionality matches ``edge_weight_type``."""
  if edge_weight_type == "GEO":
    return COORDS_GEO
  if edge_weight_type in THREE_DIMENSIONAL_TYPES:
    return COORDS_3D
  return COORDS_2D


@pytest.fixture(scope="module")
def cupy() -> Any:
  """Import CuPy lazily so CPU-only machines collect and skip instead of erroring."""
  return pytest.importorskip("cupy")


def test_edge_type_codes_match_device_header() -> None:
  """WHAT: the Python code table and `enum GpuEdgeTypeCode` carry identical values.

  WHY: the dispatcher switches on a raw `int`; a drift between the two tables makes
  the kernel compute a different metric than the caller asked for, with no error.

  EXPECTED: `GPU_EDGE_<NAME> = <code>` pairs in the header equal `EDGE_TYPE_TO_CODE`.
  """
  header = HEADER_PATH.read_text(encoding="utf-8")
  enum_codes: dict[str, int] = {
    name.removeprefix("GPU_EDGE_"): int(code)
    for name, code in re.findall(r"GPU_EDGE_([A-Z0-9_]+) = (\d+)", header)
  }

  assert enum_codes == EDGE_TYPE_TO_CODE


@pytest.mark.parametrize("edge_weight_type", sorted(EDGE_TYPE_TO_CODE))
def test_device_dispatcher_matches_cpu(edge_weight_type: str, cupy: Any) -> None:
  """WHAT: matrix-mode and coordinate-mode dispatcher output against the CPU matrix.

  WHY: `dist == nullptr` is the path large instances take to avoid an `O(n^2)`
  VRAM allocation; any arithmetic difference from the dense path changes results.

  EXPECTED: `from_matrix == from_coords == compute_distance_matrix(...)` element-wise,
  and every off-diagonal entry equals the scalar `pairwise` function.
  """
  coords = _coords_for(edge_weight_type)
  n = int(coords.shape[0])
  matrix = compute_distance_matrix(coords, edge_weight_type)
  code = EDGE_TYPE_TO_CODE[edge_weight_type]
  kernel = load_kernel("probe_pairwise.cu", "probe_pairwise_kernel", __file__)

  d_coords = cupy.asarray(coords, dtype=cupy.float64)
  d_matrix = cupy.asarray(matrix, dtype=cupy.int32)
  d_out = cupy.zeros((n, n), dtype=cupy.int32)
  threads = 64
  blocks = (n * n + threads - 1) // threads

  kernel(
    (blocks,),
    (threads,),
    (d_matrix, d_coords, np.int32(n), np.int32(code), np.int32(1), d_out),
  )
  from_matrix = cupy.asnumpy(d_out)

  d_out.fill(0)
  kernel(
    (blocks,),
    (threads,),
    (d_matrix, d_coords, np.int32(n), np.int32(code), np.int32(0), d_out),
  )
  from_coords = cupy.asnumpy(d_out)

  assert np.array_equal(from_matrix, matrix), f"{edge_weight_type}: matrix-mode drift"
  assert np.array_equal(from_coords, matrix), f"{edge_weight_type}: coordinate-mode drift"

  distance_fn = get_distance_function(edge_weight_type)
  expected = np.zeros((n, n), dtype=np.int32)
  for u in range(n):
    for v in range(n):
      if u != v:
        expected[u, v] = distance_fn(coords[u], coords[v])
  assert np.array_equal(from_coords, expected), f"{edge_weight_type}: scalar mismatch"
