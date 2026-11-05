"""
Distance matrix computation from coordinate data.

This module is the bridge between the pairwise distance functions and full
distance matrix computation. It uses the factory from :mod:`distances.pairwise`
to dispatch to the correct TSPLIB95 formula based on edge weight type.

Memory model (``BUG-13`` / ``DP10-16``): the naive 2D broadcast materialises an
``(n, n, d)`` tensor. For ``n <= DEFAULT_CHUNK_THRESHOLD`` that stays the direct
path; above it the rows are processed in blocks of ``chunk_size`` so peak
memory is ``O(chunk_size * n * d)`` instead of ``O(n^2 * d)``.

Usage:
    >>> from .matrix import compute_distance_matrix
    >>> coordinates = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    >>> distances = compute_distance_matrix(coordinates, "EUC_2D")
    >>> distances.shape
    (3, 3)
    >>> distances[0, 1]  # Distance from node 0 to node 1
    5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .pairwise import EARTH_RADIUS_KM, PI, get_distance_function

if TYPE_CHECKING:
  from protocols.backend import BackendModule

#: Largest ``n`` still computed with a single full 2D broadcast.
DEFAULT_CHUNK_THRESHOLD: int = 2048

#: Number of rows evaluated per broadcast block when ``n > chunk_threshold``.
DEFAULT_CHUNK_SIZE: int = 512

_VECTORIZED_TYPES: frozenset[str] = frozenset(
  ("EUC_2D", "EUC_3D", "MAN_2D", "MAN_3D", "MAX_2D", "MAX_3D", "CEIL_2D", "ATT", "GEO")
)


def _geo_to_radians(coord_array: Any, xp: BackendModule) -> Any:
  """Convert DDD.MM coordinates to radians (truncated degrees, minutes in the fraction)."""
  deg: Any = xp.trunc(coord_array)
  minutes: Any = coord_array - deg
  return PI * (deg + 5.0 * minutes / 3.0) / 180.0


def _xray_block(rows: Any, all_coords: Any, xp: BackendModule) -> Any:
  """Compute an ``(m, n)`` XRAY1 block with the scalar crystallography formula."""
  distance_fn = get_distance_function("XRAY1")
  row_count: int = int(rows.shape[0])
  column_count: int = int(all_coords.shape[0])
  result: Any = xp.zeros((row_count, column_count), dtype=xp.int32)
  for i in range(row_count):
    coord_i = tuple(float(value) for value in rows[i])
    for j in range(column_count):
      coord_j = tuple(float(value) for value in all_coords[j])
      result[i, j] = distance_fn(coord_i, coord_j)
  return result


def _distance_block(rows: Any, all_coords: Any, edge_weight_type: str, xp: BackendModule) -> Any:
  """Compute the ``(len(rows), n)`` distance block between ``rows`` and ``all_coords``."""
  if edge_weight_type == "XRAY1":
    return _xray_block(rows, all_coords, xp)

  coordinate_diff: Any = rows[:, np.newaxis, :] - all_coords[np.newaxis, :, :]

  if edge_weight_type == "EUC_2D":
    squared = xp.sum(coordinate_diff * coordinate_diff, axis=2)
    return (xp.sqrt(squared) + 0.5).astype(xp.int64)

  if edge_weight_type == "EUC_3D":
    squared = xp.sum(coordinate_diff * coordinate_diff, axis=2)
    return (xp.sqrt(squared) + 0.5).astype(xp.int64)

  if edge_weight_type == "MAN_2D":
    return (xp.sum(xp.abs(coordinate_diff), axis=2) + 0.5).astype(xp.int64)

  if edge_weight_type == "MAN_3D":
    return (xp.sum(xp.abs(coordinate_diff), axis=2) + 0.5).astype(xp.int64)

  if edge_weight_type == "MAX_2D":
    return (xp.max(xp.abs(coordinate_diff), axis=2) + 0.5).astype(xp.int64)

  if edge_weight_type == "MAX_3D":
    return (xp.max(xp.abs(coordinate_diff), axis=2) + 0.5).astype(xp.int64)

  if edge_weight_type == "CEIL_2D":
    squared = xp.sum(coordinate_diff * coordinate_diff, axis=2)
    return xp.ceil(xp.sqrt(squared)).astype(xp.int64)

  if edge_weight_type == "ATT":
    squared = xp.sum(coordinate_diff * coordinate_diff, axis=2)
    pseudo_euclidean: Any = xp.sqrt(squared / 10.0)
    rounded: Any = (pseudo_euclidean + 0.5).astype(xp.int64)
    return xp.where(rounded < pseudo_euclidean, rounded + 1, rounded)

  if edge_weight_type == "GEO":
    lat_rad: Any = _geo_to_radians(rows[:, 0], xp)
    lon_rad: Any = _geo_to_radians(rows[:, 1], xp)
    all_lat_rad: Any = _geo_to_radians(all_coords[:, 0], xp)
    all_lon_rad: Any = _geo_to_radians(all_coords[:, 1], xp)

    lat_i: Any = lat_rad[:, np.newaxis]
    lat_j: Any = all_lat_rad[np.newaxis, :]
    lon_i: Any = lon_rad[:, np.newaxis]
    lon_j: Any = all_lon_rad[np.newaxis, :]

    q1: Any = xp.cos(lon_i - lon_j)
    q2: Any = xp.cos(lat_i - lat_j)
    q3: Any = xp.cos(lat_i + lat_j)
    acos_arg: Any = xp.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    return (EARTH_RADIUS_KM * xp.arccos(acos_arg) + 1.0).astype(xp.int64)

  raise ValueError(
    f"Unsupported edge_weight_type: '{edge_weight_type}'. "
    f"Coordinate-based types: EUC_2D, EUC_3D, MAN_2D, MAN_3D, "
    f"MAX_2D, MAX_3D, GEO, ATT, CEIL_2D, XRAY1. "
    f"For EXPLICIT types, use Problem.distances (populated by "
    f"TsplibLoader/DatabaseLoader); nothing is computed from coordinates."
  )


def compute_candidate_list(
  coordinates: np.ndarray | None = None,
  edge_weight_type: str = "EUC_2D",
  k: int = 20,
  xp: BackendModule = np,
  *,
  distances: Any | None = None,
  chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
  """Build the ``k``-nearest-neighbour candidate list ``N_k`` of every node.

  ``N_k[u, r]`` is the ``(r + 1)``-th closest node to ``u``, self-edges excluded and
  ties broken by ascending node index. Rows are evaluated in blocks of ``chunk_size``
  so a coordinate-only problem never materialises the full ``(n, n)`` matrix when
  ``n > chunk_size`` (``DP10-20``).

  Args:
      coordinates: Node coordinates, shape ``(n, 2)`` or ``(n, 3)``. Ignored when
          ``distances`` is provided.
      edge_weight_type: TSPLIB95 edge weight type used for the coordinate blocks.
      k: Requested neighbours per node; clamped to ``min(k, n - 1)``.
      xp: Backend module (NumPy or CuPy). Default is NumPy.
      distances: Precomputed distance matrix, shape ``(n, n)``. Rows are read in
          blocks, so the matrix may stay on the target backend.
      chunk_size: Rows evaluated per block.

  Returns:
      ``int32`` array of shape ``(n, k_eff)`` with ``k_eff = min(k, n - 1)``, or
      ``(n, 0)`` when ``n <= 1`` or ``k <= 0``.

  Raises:
      ValueError: If neither ``distances`` nor ``coordinates`` is provided.

  Performance:
      Peak intermediate memory is ``O(chunk_size * n)`` plus the ``O(n * k)`` result,
      independent of ``n`` when ``distances`` is ``None``.
  """
  if distances is not None:
    n: int = int(distances.shape[0])
  elif coordinates is not None:
    n = int(coordinates.shape[0])
  else:
    raise ValueError("compute_candidate_list requires either `distances` or `coordinates`.")

  if n <= 1 or k <= 0:
    return np.zeros((n, 0), dtype=np.int32)

  k_eff: int = min(k, n - 1)
  step: int = max(1, chunk_size)
  # Self-edges are masked with the largest `int64`, which no real distance reaches.
  sentinel: int = int(np.iinfo(np.int64).max)
  result: Any = xp.empty((n, k_eff), dtype=xp.int32)

  for start in range(0, n, step):
    end: int = min(start + step, n)
    block: Any
    if distances is not None:
      block = xp.asarray(distances[start:end], dtype=xp.int64)
    else:
      if coordinates is None:  # pragma: no cover - guarded above
        raise ValueError("compute_candidate_list requires coordinates when `distances` is None.")
      block = _distance_block(coordinates[start:end], coordinates, edge_weight_type, xp).astype(xp.int64)

    rows: Any = xp.arange(end - start)
    block[rows, xp.arange(start, end)] = sentinel
    candidate_columns: Any = xp.argpartition(block, k_eff, axis=1)[:, :k_eff]
    candidate_distances: Any = block[rows[:, np.newaxis], candidate_columns]
    # `lexsort` uses its last key as primary: ascending distance, then node index.
    order: Any = xp.lexsort((candidate_columns, candidate_distances), axis=1)
    result[start:end, :] = candidate_columns[rows[:, np.newaxis], order]

  return cast("np.ndarray", result)


def compute_distance_matrix(
  coordinates: np.ndarray,
  edge_weight_type: str,
  xp: BackendModule = np,
  *,
  chunk_threshold: int = DEFAULT_CHUNK_THRESHOLD,
  chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
  """
  Compute full distance matrix from coordinates using specified edge type.

  Dispatches to the appropriate TSPLIB95 formula (EUC_2D, GEO, ATT, etc.) and
  returns the complete ``n x n`` matrix. All coordinate-based types are
  symmetric; the matrix is computed exactly once per ordered pair through the
  vectorized broadcast (no upper-triangle shortcut is taken, because the
  vectorized path is faster than a triangular Python loop).

  Args:
      coordinates: Node coordinates, shape ``(n, 2)`` or ``(n, 3)``.
      edge_weight_type: TSPLIB95 edge weight type string.
      xp: Backend module (NumPy or CuPy). Default is NumPy.
      chunk_threshold: Largest ``n`` computed with a single full broadcast.
      chunk_size: Rows per broadcast block when ``n > chunk_threshold``.

  Returns:
      Distance matrix, shape ``(n, n)``, dtype ``int32`` with a zero diagonal.

  Raises:
      ValueError: If ``edge_weight_type`` is unknown or unsupported.

  Performance:
      - Direct path: peak intermediate memory ``O(n^2 * d)``.
      - Chunked path (``n > chunk_threshold``): peak memory
        ``O(chunk_size * n * d)``, keeping large instances inside VRAM/RAM.
  """
  n: int = int(coordinates.shape[0])

  if edge_weight_type not in _VECTORIZED_TYPES and edge_weight_type != "XRAY1":
    raise ValueError(
      f"Unsupported edge_weight_type: '{edge_weight_type}'. "
      f"Coordinate-based types: EUC_2D, EUC_3D, MAN_2D, MAN_3D, "
      f"MAX_2D, MAX_3D, GEO, ATT, CEIL_2D, XRAY1. "
      f"For EXPLICIT types, use Problem.distances (populated by "
      f"TsplibLoader/DatabaseLoader); nothing is computed from coordinates."
    )

  if n <= chunk_threshold:
    distances: Any = _distance_block(coordinates, coordinates, edge_weight_type, xp)
  else:
    blocks: list[Any] = [
      _distance_block(coordinates[start : start + chunk_size], coordinates, edge_weight_type, xp)
      for start in range(0, n, chunk_size)
    ]
    distances = blocks[0] if len(blocks) == 1 else xp.concatenate(blocks, axis=0)

  # TSPLIB GEO returns 1 for identical points (`int(d + 1.0)`); the diagonal is
  # a hard contract, so it is zeroed explicitly for every type.
  result: Any = distances.astype(xp.int32)
  xp.fill_diagonal(result, 0)
  return cast("np.ndarray", result)
