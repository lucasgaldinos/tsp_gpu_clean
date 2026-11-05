"""Unit tests for :func:`distances.matrix.compute_distance_matrix`.

The matrix builder is the single funnel from raw coordinates to the ``int32`` cost
matrix consumed by every algorithm, so its dtype, zero diagonal, symmetry and TSPLIB
rounding are public contract rather than incidental implementation detail.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from distances.matrix import compute_candidate_list, compute_distance_matrix
from distances.pairwise import get_distance_function

if TYPE_CHECKING:
  from data_models.problem import Problem

# Every coordinate-based type `compute_distance_matrix` dispatches on (src/distances/matrix.py:126-220).
CARTESIAN_2D_TYPES: tuple[str, ...] = ("EUC_2D", "MAN_2D", "MAX_2D", "CEIL_2D", "ATT")
CARTESIAN_3D_TYPES: tuple[str, ...] = ("EUC_3D", "MAN_3D", "MAX_3D", "XRAY1")
ALL_SUPPORTED_TYPES: tuple[str, ...] = (*CARTESIAN_2D_TYPES, "GEO", *CARTESIAN_3D_TYPES)

# Three-point sets small enough to check by hand; chosen so that no two metric
# families produce the same matrix by accident (CEIL differs from EUC at (0,0)-(3.1,4.0)).
CARTESIAN_2D_POINTS = np.array([[0.0, 0.0], [3.1, 4.0], [1.5, 2.5]])
CARTESIAN_3D_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
GEO_POINTS = np.array([[16.47, 96.10], [16.47, 94.44], [20.09, 92.54]])  # burma14 cities 1-3


def _points_for(edge_weight_type: str) -> np.ndarray:
  """Return the coordinate set whose dimensionality matches ``edge_weight_type``."""
  if edge_weight_type == "GEO":
    return GEO_POINTS
  if edge_weight_type in CARTESIAN_3D_TYPES:
    return CARTESIAN_3D_POINTS
  return CARTESIAN_2D_POINTS


@pytest.mark.parametrize("edge_weight_type", ALL_SUPPORTED_TYPES)
def test_distance_matrix_returns_square_int32_matrix(edge_weight_type: str) -> None:
  """WHAT: every supported edge type returns an ``(n, n)`` matrix of exactly ``int32``.

  WHY: algorithms compare and sum these costs in tight loops; a silent float64 matrix
  changes equality semantics and doubles memory traffic.

  EXPECTED: shape ``(3, 3)`` and dtype ``int32`` for all ten supported types.
  """
  coordinates = _points_for(edge_weight_type)

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  assert matrix.shape == (3, 3), f"{edge_weight_type}: expected 3x3, got {matrix.shape}"
  assert matrix.dtype == np.dtype(np.int32), f"{edge_weight_type}: expected int32, got {matrix.dtype}"


@pytest.mark.parametrize("edge_weight_type", ALL_SUPPORTED_TYPES)
def test_distance_matrix_has_zero_diagonal(edge_weight_type: str) -> None:
  """WHAT: ``matrix[i, i]`` is exactly 0 for every supported type.

  WHY: a node reaching itself costs nothing; TSPLIB GEO returns 1 for identical points,
  so the builder must zero the diagonal explicitly instead of trusting the metric.

  EXPECTED: the diagonal equals ``[0, 0, 0]`` for all ten supported types.
  """
  coordinates = _points_for(edge_weight_type)

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  assert np.array_equal(matrix.diagonal(), np.zeros(3, dtype=np.int32)), (
    f"{edge_weight_type}: diagonal must be zero, got {matrix.diagonal()}"
  )


@pytest.mark.parametrize("edge_weight_type", ALL_SUPPORTED_TYPES)
def test_distance_matrix_is_symmetric(edge_weight_type: str) -> None:
  """WHAT: ``matrix[i, j] == matrix[j, i]`` for every coordinate-based type.

  WHY: the builder computes only the upper triangle mentally and mirrors it; a broken
  mirror yields asymmetric matrices that break 2-opt delta evaluation.

  EXPECTED: the transposed matrix is bit-identical for all ten supported types.
  """
  coordinates = _points_for(edge_weight_type)

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  assert np.array_equal(matrix, matrix.T), f"{edge_weight_type}: matrix is not symmetric"


@pytest.mark.parametrize(
  ("edge_weight_type", "expected"),
  [
    ("EUC_2D", [[0, 5, 3], [5, 0, 2], [3, 2, 0]]),
    ("CEIL_2D", [[0, 6, 3], [6, 0, 3], [3, 3, 0]]),
    ("MAN_2D", [[0, 7, 4], [7, 0, 3], [4, 3, 0]]),
    ("MAX_2D", [[0, 4, 3], [4, 0, 2], [3, 2, 0]]),
    ("ATT", [[0, 2, 1], [2, 0, 1], [1, 1, 0]]),
    ("EUC_3D", [[0, 4, 11], [4, 0, 7], [11, 7, 0]]),
    ("MAN_3D", [[0, 6, 18], [6, 0, 12], [18, 12, 0]]),
    ("MAX_3D", [[0, 3, 8], [3, 0, 5], [8, 5, 0]]),
    ("XRAY1", [[0, 300, 800], [300, 0, 500], [800, 500, 0]]),
    ("GEO", [[0, 153, 510], [153, 0, 422], [510, 422, 0]]),
  ],
)
def test_distance_matrix_exact_values(edge_weight_type: str, expected: list[list[int]]) -> None:
  """WHAT: the full matrix equals the values computed by hand from the TSPLIB95 formulas.

  WHY: nint (``int(x + 0.5)``), GEO's ``int(d + 1.0)`` and ATT's conditional ``+1`` are
  the difference between matching canonical tour lengths and missing them by 1 per edge.

  EXPECTED: the matrix equals ``expected`` cell by cell; sample derivation for EUC_2D
  ``d(0, 1) = nint(sqrt(3.1^2 + 4.0^2)) = nint(5.0606) = 5``.
  """
  coordinates = _points_for(edge_weight_type)

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  assert matrix.tolist() == expected, f"{edge_weight_type}: got {matrix.tolist()}, expected {expected}"


@pytest.mark.parametrize("edge_weight_type", ALL_SUPPORTED_TYPES)
def test_distance_matrix_agrees_with_pairwise_function(edge_weight_type: str) -> None:
  """WHAT: every off-diagonal matrix cell equals the module's pairwise function for that pair.

  WHY: the builder re-implements each formula vectorized; divergence between the two
  implementations makes the same instance cost differ depending on the entry point.

  EXPECTED: zero mismatches off the diagonal (the diagonal is excluded because the builder
  zeroes it while GEO's pairwise function returns 1 for identical points).
  """
  coordinates = _points_for(edge_weight_type)
  distance_function = get_distance_function(edge_weight_type)

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  mismatches = [
    (i, j, int(matrix[i, j]), distance_function(tuple(coordinates[i]), tuple(coordinates[j])))
    for i in range(len(coordinates))
    for j in range(len(coordinates))
    if i != j and int(matrix[i, j]) != distance_function(tuple(coordinates[i]), tuple(coordinates[j]))
  ]
  assert mismatches == [], f"{edge_weight_type}: matrix/pairwise mismatches (i, j, matrix, pairwise): {mismatches}"


def test_distance_matrix_with_two_nodes() -> None:
  """WHAT: the smallest non-trivial instance (2 nodes) produces a 2x2 matrix.

  WHY: the minimum size is where off-by-one diffs in broadcasting or mirroring surface.

  EXPECTED: ``[[0, 5], [5, 0]]`` for (0, 0)-(3, 4) under EUC_2D, dtype int32.
  """
  coordinates = np.array([[0.0, 0.0], [3.0, 4.0]])

  matrix = compute_distance_matrix(coordinates, "EUC_2D")

  assert matrix.tolist() == [[0, 5], [5, 0]]
  assert matrix.dtype == np.dtype(np.int32)


def test_distance_matrix_with_single_node() -> None:
  """WHAT: a one-node instance returns a 1x1 zero matrix instead of raising.

  WHY: degenerate problem dimensions come from sliced benchmark subsets; the builder must
  behave predictably rather than crash inside broadcasting.

  EXPECTED: shape ``(1, 1)`` and value ``[[0]]``.
  """
  coordinates = np.array([[2.0, 3.0]])

  matrix = compute_distance_matrix(coordinates, "EUC_2D")

  assert matrix.shape == (1, 1)
  assert matrix.tolist() == [[0]]


@pytest.mark.parametrize("edge_weight_type", CARTESIAN_2D_TYPES)
def test_distance_matrix_identical_coordinates_cost_zero(edge_weight_type: str) -> None:
  """WHAT: two coincident cities under a Cartesian metric have distance 0.

  WHY: duplicated coordinates exist in synthetic CVRP instances (two customers at one
  address); a nonzero self-pair distance inflates tour costs and breaks savings heuristics.

  EXPECTED: the whole 2x2 matrix is zero for every Cartesian 2D metric.
  """
  coordinates = np.array([[1.5, 2.5], [1.5, 2.5]])

  matrix = compute_distance_matrix(coordinates, edge_weight_type)

  assert matrix.tolist() == [[0, 0], [0, 0]], f"{edge_weight_type}: coincident points must cost 0"


def test_distance_matrix_identical_geographical_coordinates_cost_one() -> None:
  """WHAT: two coincident GEO cities cost 1 off the diagonal while the diagonal stays 0.

  WHY: TSPLIB95 GEO truncates ``d + 1.0``, so a great-circle distance of 0 becomes 1;
  the builder keeps the zero diagonal as a separate guarantee.

  EXPECTED: ``[[0, 1], [1, 0]]``.
  """
  coordinates = np.array([[16.47, 96.10], [16.47, 96.10]])

  matrix = compute_distance_matrix(coordinates, "GEO")

  assert matrix.tolist() == [[0, 1], [1, 0]]


@pytest.mark.parametrize("edge_weight_type", CARTESIAN_2D_TYPES)
def test_distance_matrix_negative_coordinates_are_translation_invariant(edge_weight_type: str) -> None:
  """WHAT: translating both cities into negative coordinates leaves distances unchanged.

  WHY: SOP/ATSP-style instances and normalized datasets carry negative Cartesian
  coordinates; sign handling in |dx|, max and squaring must not leak into the result.

  EXPECTED: the matrix for [(-7, -11), (-4, -7)] equals the matrix for [(0, 0), (3, 4)].
  """
  original = np.array([[0.0, 0.0], [3.0, 4.0]])
  translated = np.array([[-7.0, -11.0], [-4.0, -7.0]])

  expected = compute_distance_matrix(original, edge_weight_type)
  matrix = compute_distance_matrix(translated, edge_weight_type)

  assert matrix.tolist() == expected.tolist(), (
    f"{edge_weight_type}: expected {expected.tolist()}, got {matrix.tolist()}"
  )


@pytest.mark.parametrize("coordinate_dtype", [np.int64, np.int32])
def test_distance_matrix_accepts_integer_coordinate_arrays(coordinate_dtype: type[np.integer]) -> None:
  """WHAT: integer coordinate arrays yield the same matrix as float coordinates.

  WHY: TSPLIB NODE_COORD_SECTION values are often whole numbers and loaders may hand over
  integer arrays; integer arithmetic must not truncate before the square root.

  EXPECTED: ``[[0, 5], [5, 0]]`` for (0, 0)-(3, 4), dtype int32, no overflow at small values.
  """
  coordinates = np.array([[0, 0], [3, 4]], dtype=coordinate_dtype)

  matrix = compute_distance_matrix(coordinates, "EUC_2D")

  assert matrix.tolist() == [[0, 5], [5, 0]]
  assert matrix.dtype == np.dtype(np.int32)


@pytest.mark.parametrize("edge_weight_type", ["EXPLICIT", "SPECIAL", "XRAY2", "BOGUS"])
def test_distance_matrix_rejects_non_computable_types(edge_weight_type: str) -> None:
  """WHAT: types that have no coordinate-based formula raise ``ValueError``.

  WHY: EXPLICIT ships its own matrix and SPECIAL needs external documentation; silently
  computing a coordinate distance for them would corrupt the instance costs.

  EXPECTED: ``ValueError`` matching ``Unsupported edge_weight_type``.
  """
  with pytest.raises(ValueError, match="Unsupported edge_weight_type"):
    compute_distance_matrix(CARTESIAN_2D_POINTS, edge_weight_type)


def test_explicit_type_error_points_to_the_matrix_table() -> None:
  """WHAT: the EXPLICIT error names the replacement source of truth.

  WHY: the caller's next step is a database lookup, so the message must say where
  EXPLICIT costs live rather than only that the call failed.

  EXPECTED: ``ValueError`` whose text points at ``Problem.distances``.
  """
  with pytest.raises(ValueError, match=r"Problem\.distances"):
    compute_distance_matrix(CARTESIAN_2D_POINTS, "EXPLICIT")


def test_burma14_geo_matrix_contract(burma14_problem: Problem) -> None:
  """WHAT: the 14-city GEO fixture yields the documented matrix contract and known distances.

  WHY: validates the builder against a real TSPLIB95 instance instead of a synthetic one;
  burma14 is the smallest GEO instance with published canonical distances.

  EXPECTED: 14x14 int32, symmetric, zero diagonal, ``d(1,2)=153, d(1,3)=510, d(2,3)=422``.
  """
  assert burma14_problem.coordinates is not None
  assert burma14_problem.edge_type == "GEO"

  matrix = compute_distance_matrix(burma14_problem.coordinates, burma14_problem.edge_type)

  assert matrix.shape == (14, 14)
  assert matrix.dtype == np.dtype(np.int32)
  assert np.array_equal(matrix, matrix.T)
  assert np.array_equal(matrix.diagonal(), np.zeros(14, dtype=np.int32))
  assert matrix[0, :3].tolist() == [0, 153, 510]
  assert matrix[1, 2] == 422


def test_berlin52_euclidean_matrix_known_values(berlin52_problem: Problem) -> None:
  """WHAT: the 52-city EUC_2D fixture yields the hand-checked distances for cities 1-3.

  WHY: pins nint rounding on real data: city 1 (565, 575) to city 2 (25, 185) is
  ``sqrt(540^2 + 390^2) = 666.108...`` which must round to 666, not 667.

  EXPECTED: 52x52 int32; ``d(1,2)=666``, ``d(1,3)=281``, ``d(2,3)=649``.
  """
  assert berlin52_problem.coordinates is not None
  assert berlin52_problem.edge_type == "EUC_2D"

  matrix = compute_distance_matrix(berlin52_problem.coordinates, berlin52_problem.edge_type)

  assert matrix.shape == (52, 52)
  assert matrix.dtype == np.dtype(np.int32)
  assert matrix[0, 1] == 666
  assert matrix[0, 2] == 281
  assert matrix[1, 2] == 649


@pytest.mark.parametrize("chunk_size", [3, 10], ids=["multi-block", "single-block"])
@pytest.mark.parametrize("edge_weight_type", ALL_SUPPORTED_TYPES)
def test_distance_matrix_row_chunked_path_matches_full_broadcast(
  edge_weight_type: str,
  chunk_size: int,
) -> None:
  """WHAT: the row-chunked path (``n > chunk_threshold``) against the full 2D broadcast.

  WHY: BUG-13 bounds peak memory to ``O(chunk_size * n * d)`` for large instances; a
  chunked evaluation that disagrees with the direct broadcast changes costs depending
  only on instance size, and the single-block and multi-block branches differ.

  EXPECTED: bit-identical matrices for all ten types, with ``n = 7`` rows split into
  three blocks (``chunk_size=3``) and one block (``chunk_size=10``).
  """
  rng = np.random.default_rng(13)
  width = 3 if edge_weight_type in CARTESIAN_3D_TYPES else 2
  coordinates = rng.uniform(0.0, 60.0, size=(7, width))

  full = compute_distance_matrix(coordinates, edge_weight_type)
  chunked = compute_distance_matrix(coordinates, edge_weight_type, chunk_threshold=3, chunk_size=chunk_size)

  assert chunked.dtype == np.dtype(np.int32)
  np.testing.assert_array_equal(
    chunked,
    full,
    err_msg=f"{edge_weight_type}: chunked path (chunk_size={chunk_size}) diverged from the full broadcast",
  )


def _brute_force_candidates(
  matrix: np.ndarray,
  k: int,
) -> np.ndarray:
  """Return ``N_k`` by sorting every row of ``matrix`` (ties broken by node index)."""
  n = int(matrix.shape[0])
  k_eff = min(k, n - 1)
  expected = np.empty((n, k_eff), dtype=np.int32)
  for u in range(n):
    order = sorted((v for v in range(n) if v != u), key=lambda v: (int(matrix[u, v]), v))
    expected[u] = order[:k_eff]
  return expected


@pytest.mark.parametrize(
  ("n", "k", "expected_shape"),
  [
    pytest.param(3, 20, (3, 2), id="clamped-to-n-minus-1"),
    pytest.param(20, 5, (20, 5), id="k-below-n"),
    pytest.param(20, 0, (20, 0), id="k-zero"),
    pytest.param(20, -3, (20, 0), id="k-negative"),
    pytest.param(1, 20, (1, 0), id="single-node"),
  ],
)
def test_candidate_list_shape_and_dtype(n: int, k: int, expected_shape: tuple[int, int]) -> None:
  """WHAT: shape and dtype contract of the ``k``-NN candidate list.

  WHY: every consumer indexes ``N_k[u, r]`` assuming ``k_eff = min(k, n - 1)`` columns
  and ``int32`` node ids; a wrong width silently drops or repeats neighbours.

  EXPECTED: ``(n, k_eff)`` int32, with ``(n, 0)`` for ``k <= 0`` and ``n <= 1``.
  """
  rng = np.random.default_rng(7)
  coordinates = rng.uniform(0.0, 10_000.0, size=(n, 2))

  candidates = compute_candidate_list(coordinates, "EUC_2D", k)

  assert candidates.shape == expected_shape
  assert candidates.dtype == np.dtype(np.int32)


@pytest.mark.parametrize("edge_weight_type", ["EUC_2D", "MAN_2D", "GEO"])
def test_candidate_list_matches_brute_force_from_coordinates(edge_weight_type: str) -> None:
  """WHAT: blocked coordinate construction against a row-by-row brute force.

  WHY: ``argpartition`` + ``lexsort`` must return exactly the ``k`` smallest *distinct*
  nodes; an off-by-one in the partial selection silently substitutes a farther node.

  EXPECTED: identical tables for the hand-checked three-point fixtures.
  """
  coordinates = _points_for(edge_weight_type)
  matrix = compute_distance_matrix(coordinates, edge_weight_type)
  expected = _brute_force_candidates(matrix, k=2)

  candidates = compute_candidate_list(coordinates, edge_weight_type, 2)

  np.testing.assert_array_equal(candidates, expected)


@pytest.mark.parametrize("k", [1, 2, 5], ids=["k1", "k2", "k5"])
def test_candidate_list_from_distances_matches_brute_force(k: int) -> None:
  """WHAT: the ``distances=`` path against the brute force on the same matrix.

  WHY: EXPLICIT problems have no coordinates; the matrix overload must produce the same
  neighbour multiset as the coordinate path so candidate-guided search behaves identically.

  EXPECTED: identical distance multisets per row, and every node strictly nearer than the
  ``k``-th selected distance present (the tie at the selection boundary may resolve either
  way, because ``argpartition`` is a linear-time partial sort).
  """
  rng = np.random.default_rng(23)
  coordinates = rng.uniform(0.0, 500.0, size=(12, 2))
  matrix = compute_distance_matrix(coordinates, "EUC_2D")
  expected = _brute_force_candidates(matrix, k)

  candidates = compute_candidate_list(k=k, distances=matrix)

  assert candidates.shape == expected.shape
  for node in range(matrix.shape[0]):
    selected = [int(matrix[node, int(neighbor)]) for neighbor in candidates[node]]
    reference = [int(matrix[node, int(neighbor)]) for neighbor in expected[node]]
    assert selected == sorted(selected), f"row {node} is not ordered by distance"
    assert selected == reference, f"row {node} selected a different distance multiset"
    boundary = reference[-1]
    present = {int(neighbor) for neighbor in candidates[node]}
    for other in range(matrix.shape[0]):
      if other != node and matrix[node, other] < boundary:
        assert other in present, f"row {node} dropped the strictly nearer node {other}"



def test_candidate_list_blocked_path_matches_unchunked() -> None:
  """WHAT: a tiny ``chunk_size`` against the single-block evaluation.

  WHY: ``DP10-20`` evaluates rows in blocks so large instances never materialise the
  ``(n, n)`` matrix; a block boundary must not change a single neighbour.

  EXPECTED: bit-identical tables for ``chunk_size=7`` and ``chunk_size=512``.
  """
  rng = np.random.default_rng(31)
  coordinates = rng.uniform(0.0, 1_000.0, size=(37, 2))

  blocked = compute_candidate_list(coordinates, "EUC_2D", 6, chunk_size=7)
  single = compute_candidate_list(coordinates, "EUC_2D", 6, chunk_size=512)

  np.testing.assert_array_equal(blocked, single)


def test_candidate_list_never_materializes_the_full_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: row-block heights observed inside ``compute_candidate_list``.

  WHY: the whole point of ``DP10-20`` is bounding peak memory to ``O(chunk_size * n)``
  for coordinate-only problems; one leaked full-width block defeats it.

  EXPECTED: every ``_distance_block`` call evaluates at most ``chunk_size`` rows.
  """
  from distances import matrix as matrix_module

  rng = np.random.default_rng(41)
  n = 64
  chunk_size = 16
  coordinates = rng.uniform(0.0, 5_000.0, size=(n, 2))
  observed_rows: list[int] = []
  original = matrix_module._distance_block

  def _recording_distance_block(rows: np.ndarray, all_coords: np.ndarray, edge: str, xp: Any) -> np.ndarray:
    observed_rows.append(int(rows.shape[0]))
    return cast("np.ndarray", original(rows, all_coords, edge, xp))

  monkeypatch.setattr(matrix_module, "_distance_block", _recording_distance_block)

  candidates = compute_candidate_list(coordinates, "EUC_2D", 5, chunk_size=chunk_size)

  assert candidates.shape == (n, 5)
  assert observed_rows, "the blocked path must call _distance_block"
  assert max(observed_rows) <= chunk_size, f"block of {max(observed_rows)} rows exceeds chunk_size={chunk_size}"
  assert max(observed_rows) < n, "a full-width block means the (n, n) matrix was materialised"


def test_candidate_list_excludes_self_and_sorts_by_distance() -> None:
  """WHAT: self-edge exclusion plus ordering invariants of ``N_k``.

  WHY: a self-edge costs 0 and would occupy the first column of every row; unsorted
  columns turn the first improving move into an arbitrary one.

  EXPECTED: ``u not in N_k[u, :]``, distances non-decreasing along the row, and equal
  distances ordered by ascending node index.
  """
  rng = np.random.default_rng(53)
  coordinates = rng.integers(0, 8, size=(30, 2)).astype(np.float64)
  matrix = compute_distance_matrix(coordinates, "EUC_2D")

  candidates = compute_candidate_list(k=8, distances=matrix)

  for u in range(30):
    row = candidates[u]
    assert u not in set(int(v) for v in row), f"node {u} appears in its own candidate list"
    distances = [int(matrix[u, int(v)]) for v in row]
    assert distances == sorted(distances), f"row {u} is not ordered by distance"
    for left, right in itertools.pairwise(row):
      if matrix[u, int(left)] == matrix[u, int(right)]:
        assert int(left) < int(right), f"row {u} breaks ties by descending node index"

