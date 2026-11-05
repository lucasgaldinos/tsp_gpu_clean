"""
Pairwise distance functions for TSPLIB95 edge weight types.

The scalar TSPLIB95 arithmetic lives in ``routing_data``'s
:mod:`converter.utils.cost` module (``DP10-13``). This module re-exports those
functions under the names the solver has always used, defines the
``EDGE_WEIGHT_TYPE`` dispatch table, and shares the TSPLIB95 constants with
:mod:`distances.matrix`.

Implementation Notes:
    - All functions return int (TSPLIB95 stores distances as integers)
    - Rounding: nint(x) = int(x + 0.5) (round to nearest integer)
    - Coordinates: sequences of floats (x, y) or (x, y, z)
    - GEO coordinates: DDD.MM format (degrees and decimal minutes)
    - XRAY coordinates: (PHI, CHI, TWOTH) crystallography angles

References:
    - TSPLIB95 spec: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
    - Original C code: TSPLIB95 dist.c
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from converter.utils.cost import (
  att as compute_att,
)
from converter.utils.cost import (
  ceil_2d as compute_ceiling_2d,
)
from converter.utils.cost import (
  euclidean_2d as compute_euclidean_2d,
)
from converter.utils.cost import (
  euclidean_3d as compute_euclidean_3d,
)
from converter.utils.cost import (
  geographical as compute_geographical,
)
from converter.utils.cost import (
  manhattan_2d as compute_manhattan_2d,
)
from converter.utils.cost import (
  manhattan_3d as compute_manhattan_3d,
)
from converter.utils.cost import (
  maximum_2d as compute_maximum_2d,
)
from converter.utils.cost import (
  maximum_3d as compute_maximum_3d,
)
from converter.utils.cost import (
  nint as _nint,
)
from converter.utils.cost import (
  xray1 as compute_xray1,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

#: TSPLIB95 uses this approximation of pi for DDD.MM coordinate conversion.
PI = 3.141592

#: Earth radius in km used by the TSPLIB95 geographical distance.
EARTH_RADIUS_KM = 6378.388


class PairwiseDistanceFn(Protocol):
  """Callable computing an integer TSPLIB95 distance from two coordinates.

  The parameters are positional-only so the ``routing_data`` scalar functions
  (``euclidean_2d(a, b)``, ``att(a, b)``, ...) satisfy the protocol without
  renaming their arguments (``DP10-13``).
  """

  def __call__(self, coord1: Sequence[float], coord2: Sequence[float], /) -> int:
    """Return the TSPLIB95 integer distance between ``coord1`` and ``coord2``."""
    ...


_DISTANCE_FUNCTION_MAP: dict[str, PairwiseDistanceFn] = {
  "EUC_2D": compute_euclidean_2d,
  "EUC_3D": compute_euclidean_3d,
  "MAN_2D": compute_manhattan_2d,
  "MAN_3D": compute_manhattan_3d,
  "MAX_2D": compute_maximum_2d,
  "MAX_3D": compute_maximum_3d,
  "GEO": compute_geographical,
  "ATT": compute_att,
  "CEIL_2D": compute_ceiling_2d,
  "XRAY1": compute_xray1,
  # XRAY2 would be similar to XRAY1 with different normalization
  # Not implemented as no TSPLIB instances use it
}


#: Coordinate-based ``EDGE_WEIGHT_TYPE`` -> integer code understood by
#: ``gpu_pairwise_distance`` in ``src/distances/kernels/pairwise_device.cuh``.
#: The codes must match ``enum GpuEdgeTypeCode`` in that header; the parity is
#: asserted by ``tests/distances/test_unit_gpu_pairwise_device.py``.
EDGE_TYPE_TO_CODE: dict[str, int] = {
  "EUC_2D": 0,
  "EUC_3D": 1,
  "MAN_2D": 2,
  "MAN_3D": 3,
  "MAX_2D": 4,
  "MAX_3D": 5,
  "CEIL_2D": 6,
  "ATT": 7,
  "GEO": 8,
}


def get_edge_type_code(edge_weight_type: str) -> int:
  """Return the device dispatcher code for a coordinate-based edge weight type.

  Args:
      edge_weight_type: TSPLIB95 edge weight type string (case-insensitive).

  Returns:
      The ``int`` code shared with ``pairwise_device.cuh``.

  Raises:
      ValueError: If the type has no on-the-fly formula (``EXPLICIT``,
          ``SPECIAL``, ``XRAY1``).
  """
  code: int | None = EDGE_TYPE_TO_CODE.get(edge_weight_type.upper())
  if code is None:
    raise ValueError(
      f"EDGE_WEIGHT_TYPE '{edge_weight_type}' has no device distance code. "
      f"Supported types: {', '.join(EDGE_TYPE_TO_CODE)}"
    )
  return code


def get_distance_function(edge_weight_type: str) -> PairwiseDistanceFn:
  """
  Get distance computation function for given EDGE_WEIGHT_TYPE.

  Returns the appropriate pairwise distance function based on TSPLIB95
  edge weight type string. Handles all 10 supported types.

  Args:
      edge_weight_type: TSPLIB95 edge weight type string
          (case-insensitive)

  Returns:
      Distance computation function taking two coordinate sequences

  Raises:
      ValueError: If edge_weight_type is unknown or requires special handling
          (EXPLICIT and SPECIAL are not computed functions)

  Supported Types:
      - EUC_2D: 2D Euclidean distance
      - EUC_3D: 3D Euclidean distance
      - MAN_2D: 2D Manhattan distance
      - MAN_3D: 3D Manhattan distance
      - MAX_2D: 2D Maximum distance
      - MAX_3D: 3D Maximum distance
      - GEO: Geographical distance
      - ATT: Pseudo-Euclidean (att48, att532)
      - CEIL_2D: Ceiling of Euclidean
      - XRAY1: Crystallography distance
  """
  func: PairwiseDistanceFn | None = _DISTANCE_FUNCTION_MAP.get(edge_weight_type.upper())
  if func:
    return func
  elif edge_weight_type.upper() in ["EXPLICIT", "SPECIAL"]:
    raise ValueError(
      f"Distance type '{edge_weight_type}' is handled via explicit data "
      "or special external logic, not a calculated function."
    )
  else:
    raise ValueError(
      f"Unknown or unsupported EDGE_WEIGHT_TYPE: '{edge_weight_type}'. "
      f"Supported types: {', '.join(_DISTANCE_FUNCTION_MAP.keys())}, EXPLICIT, SPECIAL"
    )


__all__ = [
  "EARTH_RADIUS_KM",
  "EDGE_TYPE_TO_CODE",
  "PI",
  "PairwiseDistanceFn",
  "_nint",
  "compute_att",
  "compute_ceiling_2d",
  "compute_euclidean_2d",
  "compute_euclidean_3d",
  "compute_geographical",
  "compute_manhattan_2d",
  "compute_manhattan_3d",
  "compute_maximum_2d",
  "compute_maximum_3d",
  "compute_xray1",
  "get_distance_function",
  "get_edge_type_code",
]
