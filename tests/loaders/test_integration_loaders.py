"""Integration tests for the TSPLIB95 / DuckDB loading layer.

Covers the `routing_data` parse into a `Problem` bundle, the `DatabaseLoader` fallback to
`TsplibLoader` when the database file is missing, the DuckDB ingestion round-trip, and
the normalized-schema metadata exposed by `Problem.to_metadata_dict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms import GeneticAlgorithm
from algorithms.strategies import (
  CrossoverOrder,
  EvaluationCPUCost,
  EvaluationGPUCost,
  InitializationRandom,
  LocalSearchCPUTwoOpt,
  LocalSearchGPUTwoOpt,
  MutationSwap,
  SelectionTournament,
)
from data_models.problem import Problem
from distances.matrix import compute_distance_matrix
from distances.pairwise import compute_euclidean_3d
from loaders import (
  DatabaseLoader,
  TsplibLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  load_tsplib_file,
  parse_tsplib_file,
  resolve_opt_tour_cost,
)
from protocols.backend import cupy_backend, gpu_available
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from pathlib import Path

  from loaders.tsplib_loader import TsplibIngestionBundle


def test_parse_and_bundle_to_problem(fixture_tsp_dir: Path) -> None:
  """Verify parsing EUC_2D and EXPLICIT .tsp instances through `routing_data`."""
  bundle_euc: TsplibIngestionBundle = parse_tsplib_file(fixture_tsp_dir / "berlin52.tsp", problem_id=1)
  assert bundle_euc.problem.name == "berlin52"
  assert bundle_euc.problem.dimension == 52
  assert bundle_euc.problem.edge_weight_type == "EUC_2D"
  assert len(bundle_euc.nodes) == 52
  assert bundle_euc.edge_matrix is None

  problem_euc: Problem = bundle_to_problem(bundle_euc)
  assert problem_euc.dimension == 52
  assert problem_euc.coordinates is not None
  assert problem_euc.coordinates.shape == (52, 2)
  assert problem_euc.distances is not None
  assert problem_euc.distances.shape == (52, 52)

  bundle_exp: TsplibIngestionBundle = parse_tsplib_file(fixture_tsp_dir / "gr17.tsp", problem_id=2)
  assert bundle_exp.problem.name == "gr17"
  assert bundle_exp.problem.edge_weight_type == "EXPLICIT"
  assert bundle_exp.edge_matrix is not None

  problem_exp: Problem = bundle_to_problem(bundle_exp)
  assert problem_exp.dimension == 17
  assert problem_exp.distances is not None
  assert problem_exp.distances.shape == (17, 17)

  direct_problem: Problem = load_tsplib_file(fixture_tsp_dir / "berlin52.tsp")
  assert direct_problem.name == "berlin52"
  assert direct_problem.dimension == 52
  assert direct_problem.distances is not None
  assert direct_problem.distances.shape == (52, 52)


def test_resolve_opt_tour_cost_uses_canonical_geo_distances(fixture_tsp_dir: Path, tmp_path: Path) -> None:
  """Verify `resolve_opt_tour_cost` returns `None` when absent and computes canonical GEO cost when present."""
  burma_copy = tmp_path / "burma14.tsp"
  burma_copy.write_text((fixture_tsp_dir / "burma14.tsp").read_text(encoding="utf-8"), encoding="utf-8")
  assert resolve_opt_tour_cost(burma_copy) is None

  # Known canonical optimal tour for burma14 (1-based TSPLIB IDs, optimal cost = 3323)
  opt_tour_nodes = [1, 2, 14, 3, 4, 5, 12, 6, 13, 7, 11, 8, 9, 10]
  tour_text = "\n".join(
    ["NAME : burma14.opt.tour", "TYPE : TOUR", "DIMENSION : 14", "TOUR_SECTION", *map(str, opt_tour_nodes), "-1", "EOF"]
  )
  (tmp_path / "burma14.opt.tour").write_text(tour_text + "\n", encoding="utf-8")
  resolved = resolve_opt_tour_cost(burma_copy)
  assert resolved is not None
  assert resolved > 0
  # Matches canonical Problem distance matrix (including BUG-32 GEO truncation patch)
  prob = load_tsplib_file(burma_copy)
  assert prob.distances is not None
  zero_based = [n - 1 for n in opt_tour_nodes]
  expected_cost = sum(int(prob.distances[zero_based[i], zero_based[(i + 1) % 14]]) for i in range(14))
  assert resolved == float(expected_cost)


def test_database_loader_fallback_and_duckdb_ingestion(tmp_path: Path, fixture_tsp_dir: Path) -> None:
  """Verify DatabaseLoader fallback to TsplibLoader and full DuckDB ingestion."""
  # 1. Test automatic fallback when routing.duckdb does not exist yet
  missing_default_db = tmp_path / "routing.duckdb"
  with DatabaseLoader(str(missing_default_db), fallback_tsp_dir=fixture_tsp_dir) as loader:
    prob = loader.load("berlin52")
    assert prob.name == "berlin52"
    assert prob.dimension == 52

  # 2. Test ingesting a subset of .tsp files into a temporary DuckDB file
  mini_tsp_dir: Path = tmp_path / "mini_tsp"
  mini_tsp_dir.mkdir()
  for name in ("berlin52.tsp", "gr17.tsp"):
    src_file: Path = fixture_tsp_dir / name
    (mini_tsp_dir / name).write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

  ingested_db: Path = tmp_path / "ingested.duckdb"
  count = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, ingested_db)
  assert count == 2
  assert ingested_db.exists()

  with DatabaseLoader(str(ingested_db), fallback_tsp_dir=None) as db_loader:
    p1 = db_loader.load("berlin52")
    p2 = db_loader.load("gr17")
    assert p1.dimension == 52
    assert p2.dimension == 17
    assert p2.distances is not None


def test_database_loader_normalized_schema_and_problem_metadata(fixture_duckdb_path: Path) -> None:
  """Verify DatabaseLoader queries normalized DuckDB and exposes metadata."""
  with DatabaseLoader(fixture_duckdb_path, fallback_tsp_dir=None) as loader:
    prob_b52 = loader.load("berlin52")
    assert prob_b52.name == "berlin52"
    assert prob_b52.dimension == 52
    assert prob_b52.edge_type == "EUC_2D"
    assert prob_b52.problem_id is not None
    meta_b52: dict[str, Any] = prob_b52.to_metadata_dict(optimal_cost=7542.0)
    assert meta_b52["name"] == "berlin52"
    assert meta_b52["dimension"] == 52
    assert meta_b52["optimal_cost"] == 7542.0
    assert "source_file" in meta_b52
    assert "file_checksum" in meta_b52

    prob_bayg29 = loader.load("bayg29")
    assert prob_bayg29.name == "bayg29"
    assert prob_bayg29.dimension == 29
    assert prob_bayg29.edge_type == "EXPLICIT"
    assert prob_bayg29.distances is not None
    assert prob_bayg29.distances.shape == (29, 29)
    meta_bayg29: dict[str, Any] = prob_bayg29.to_metadata_dict(optimal_cost=1610.0)
    assert meta_bayg29["dimension"] == 29
    assert meta_bayg29["edge_type"] == "EXPLICIT"


def test_ingest_refuses_non_empty_db(tmp_path: Path, fixture_tsp_dir: Path) -> None:
  """Verify ingest_tsplib_directory_to_duckdb raises DatabaseNotEmptyError unless overwrite=True."""
  import pytest

  from data_models.exceptions import DatabaseNotEmptyError

  mini_tsp_dir: Path = tmp_path / "mini_tsp_overwrite"
  mini_tsp_dir.mkdir()
  src_file: Path = fixture_tsp_dir / "gr17.tsp"
  (mini_tsp_dir / "gr17.tsp").write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

  test_db: Path = tmp_path / "test_overwrite.duckdb"

  # First ingestion succeeds
  count1 = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=False)
  assert count1 == 1

  # Second ingestion without overwrite raises DatabaseNotEmptyError
  with pytest.raises(DatabaseNotEmptyError, match="already contains records in 'problems'"):
    ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=False)

  # Second ingestion with overwrite=True succeeds
  count2 = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=True)
  assert count2 == 1


def test_ingest_missing_directory_raises(tmp_path: Path) -> None:
  """Verify ingest_tsplib_directory_to_duckdb raises FileNotFoundError if source directory does not exist."""
  with pytest.raises(FileNotFoundError, match="TSPLIB source directory not found"):
    ingest_tsplib_directory_to_duckdb(tmp_path / "non_existent_tsp", tmp_path / "out.duckdb")


def _write_euc3d_fixture(tsp_dir: Path) -> Path:
  """Write a 5-node EUC_3D TSPLIB instance whose z-axis alters pairwise distances."""
  tsp_file = tsp_dir / "cube5.tsp"
  tsp_file.write_text(
    "NAME: cube5\n"
    "TYPE: TSP\n"
    "COMMENT: 5-node synthetic 3D Euclidean TSP instance\n"
    "DIMENSION: 5\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 0.0 0.0 10.0\n"
    "3 3.0 4.0 0.0\n"
    "4 3.0 4.0 12.0\n"
    "5 6.0 8.0 5.0\n"
    "EOF\n",
    encoding="utf-8",
  )
  return tsp_file


def test_euc3d_end_to_end_loaders_distances_and_cpu_ga(tmp_path: Path) -> None:
  """Verify EUC_3D round-trips through TSPLIB, DuckDB, ProblemContext, and CPU GeneticAlgorithm."""
  tsp_dir = tmp_path / "euc3d_tsp"
  tsp_dir.mkdir()
  tsp_file = _write_euc3d_fixture(tsp_dir)

  # 1. TSPLIB parse -> Problem (eager and lazy)
  bundle = parse_tsplib_file(tsp_file, problem_id=1)
  eager_problem = bundle_to_problem(bundle)
  lazy_problem = bundle_to_problem(bundle, max_eager_matrix_dimension=0)

  with TsplibLoader(tsp_dir) as loader:
    loaded_from_dir = loader.load("cube5")

  assert eager_problem == loaded_from_dir
  assert eager_problem.edge_type == "EUC_3D"
  assert eager_problem.coordinates is not None
  assert eager_problem.coordinates.shape == (5, 3)
  assert eager_problem.distances is not None
  assert lazy_problem.distances is None

  expected_matrix = compute_distance_matrix(eager_problem.coordinates, "EUC_3D")
  euc2d_only_matrix = compute_distance_matrix(eager_problem.coordinates[:, :2], "EUC_2D")
  assert not np.array_equal(expected_matrix, euc2d_only_matrix), "z coordinate must affect 3D distances"
  np.testing.assert_array_equal(eager_problem.distances, expected_matrix)

  for i in range(5):
    for j in range(5):
      if i != j:
        assert int(expected_matrix[i, j]) == compute_euclidean_3d(
          eager_problem.coordinates[i], eager_problem.coordinates[j]
        )

  # 2. DuckDB ingestion (including .opt.tour 3D cost backfill) -> DatabaseLoader round-trip
  tour_file = tsp_dir / "cube5.opt.tour"
  tour_file.write_text(
    "NAME: cube5.opt.tour\nTYPE: TOUR\nDIMENSION: 5\nTOUR_SECTION\n1\n2\n4\n5\n3\n-1\nEOF\n",
    encoding="utf-8",
  )
  db_path = tmp_path / "euc3d.duckdb"
  assert ingest_tsplib_directory_to_duckdb(tsp_dir, db_path) == 1

  import duckdb

  with duckdb.connect(str(db_path), read_only=True) as conn:
    solution_rows = conn.execute("SELECT cost, routes FROM solutions").fetchall()
  assert solution_rows == [(36.0, [[0, 1, 3, 4, 2]])]

  with DatabaseLoader(db_path, fallback_tsp_dir=None) as db_loader:
    db_problem = db_loader.load("cube5")
    listed = db_loader.list_problems("TSP")

  assert len(listed) == 1
  assert listed[0]["edge_weight_type"] == "EUC_3D"
  assert db_problem.edge_type == "EUC_3D"
  assert db_problem.coordinates is not None
  assert db_problem.coordinates.shape == (5, 3)
  assert db_problem.distances is not None
  np.testing.assert_array_equal(db_problem.distances, expected_matrix)

  with DatabaseLoader(db_path, fallback_tsp_dir=None, max_eager_matrix_dimension=0) as lazy_db_loader:
    lazy_db_problem = lazy_db_loader.load("cube5")
  assert lazy_db_problem.distances is None

  # 3. ProblemContext (lazy CPU matrix, candidate list, VRAM estimate with d=3) and CPU GA
  context = ProblemContext(lazy_db_problem, xp=np, seed=42)
  cpu_coords = context.get_cpu_coordinates()
  assert cpu_coords is not None
  assert cpu_coords.shape == (5, 3)
  np.testing.assert_array_equal(context.get_cpu_distances(), expected_matrix)
  assert context.get_cpu_candidate_list(k=3).shape == (5, 3)
  # 4n^2 + 12Pn + 24P + 8dn with n=5, P=4, d=3 -> 100 + 240 + 96 + 120 = 556 bytes
  assert context.estimate_vram_bytes(4) == 4 * 5 * 5 + 12 * 4 * 5 + 24 * 4 + 8 * 3 * 5

  ga = GeneticAlgorithm(
    population_size=8,
    mutation_rate=0.2,
    seed=42,
    initialization=InitializationRandom(seed=42),
    selection=SelectionTournament(k=2, seed=42),
    crossover=CrossoverOrder(seed=42),
    mutation=MutationSwap(seed=42),
    evaluation=EvaluationCPUCost(),
    local_search=LocalSearchCPUTwoOpt(max_iterations=5),
    local_search_stage="post_mutation",
  )
  customers = list(range(5))
  best_tour, stats = ga.evolve(context, customers, max_generations=5)

  assert best_tour.shape == (5,)
  assert sorted(best_tour.tolist()) == customers
  tour_cost = sum(int(expected_matrix[int(best_tour[idx]), int(best_tour[(idx + 1) % 5])]) for idx in range(5))
  assert stats["best_fitness"] == pytest.approx(float(tour_cost))


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU not available")
@pytest.mark.parametrize("max_matrix_dimension", [10, 1], ids=["dense-matrix", "on-the-fly-3d"])
def test_euc3d_gpu_end_to_end_dense_and_on_the_fly(tmp_path: Path, max_matrix_dimension: int) -> None:
  """Verify EUC_3D executes on GPU in both dense-matrix and on-the-fly GPU_EDGE_EUC_3D modes."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")

  tsp_dir = tmp_path / "euc3d_gpu_tsp"
  tsp_dir.mkdir()
  tsp_file = _write_euc3d_fixture(tsp_dir)
  problem = bundle_to_problem(parse_tsplib_file(tsp_file), max_eager_matrix_dimension=0)

  cpu_context = ProblemContext(problem, xp=np, seed=7)
  gpu_context = ProblemContext(problem, xp=cp, seed=7, max_matrix_dimension=max_matrix_dimension)
  expected_matrix = cpu_context.get_cpu_distances()

  assert gpu_context.uses_on_the_fly_distances(problem.dimension) is (max_matrix_dimension < problem.dimension)
  gpu_coords = gpu_context.get_gpu_coordinates()
  assert gpu_coords is not None
  assert gpu_coords.shape == (5, 3)

  probe_population = np.array([[0, 1, 2, 3, 4], [4, 2, 0, 3, 1]], dtype=np.int32)
  cpu_costs = EvaluationCPUCost().evaluate_batch(probe_population, cpu_context)
  gpu_eval = EvaluationGPUCost(max_matrix_dimension=max_matrix_dimension)
  gpu_costs = cp.asnumpy(gpu_eval.evaluate_batch(probe_population, gpu_context))
  np.testing.assert_array_equal(gpu_costs, cpu_costs)

  ga = GeneticAlgorithm(
    population_size=8,
    mutation_rate=0.2,
    seed=7,
    initialization=InitializationRandom(seed=7),
    selection=SelectionTournament(k=2, seed=7),
    crossover=CrossoverOrder(seed=7),
    mutation=MutationSwap(seed=7),
    evaluation=EvaluationGPUCost(max_matrix_dimension=max_matrix_dimension),
    local_search=LocalSearchGPUTwoOpt(mode="batch", max_iterations=5, max_matrix_dimension=max_matrix_dimension),
    local_search_stage="post_mutation",
  )
  customers = list(range(5))
  best_tour, stats = ga.evolve(gpu_context, customers, max_generations=4)

  assert best_tour.shape == (5,)
  assert sorted(best_tour.tolist()) == customers
  tour_cost = sum(int(expected_matrix[int(best_tour[idx]), int(best_tour[(idx + 1) % 5])]) for idx in range(5))
  assert stats["best_fitness"] == pytest.approx(float(tour_cost))


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU not available")
def test_euc3d_gpu_on_the_fly_all_modes_and_non_contiguous_float32_inputs(tmp_path: Path) -> None:
  """Verify EUC_3D on-the-fly GPU modes (`multi_move`, `naive` + `candidate_k`) and F-order/`float32` coords."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CuPy is not importable")

  tsp_dir = tmp_path / "euc3d_gpu_variants"
  tsp_dir.mkdir()
  tsp_file = _write_euc3d_fixture(tsp_dir)
  canonical = bundle_to_problem(parse_tsplib_file(tsp_file), max_eager_matrix_dimension=0)
  assert canonical.coordinates is not None

  probe_population = np.array([[0, 2, 1, 4, 3], [4, 1, 3, 0, 2]], dtype=np.int32)
  cpu_ctx = ProblemContext(canonical, xp=np, seed=11)
  cpu_costs = EvaluationCPUCost().evaluate_batch(probe_population, cpu_ctx)

  # F-order, sliced non-contiguous, and float32 inputs must normalize cleanly and preserve GPU on-the-fly parity
  f_order = np.asfortranarray(canonical.coordinates)
  sliced = np.column_stack([canonical.coordinates, np.ones(5)])[:, :3]
  f32 = canonical.coordinates.astype(np.float32)
  for variant in (f_order, sliced, f32):
    var_problem = Problem("var_3d", 5, "TSP", "EUC_3D", coordinates=variant, distances=None)
    var_gpu_ctx = ProblemContext(var_problem, xp=cp, seed=11, max_matrix_dimension=0)
    gpu_costs = cp.asnumpy(EvaluationGPUCost(max_matrix_dimension=0).evaluate_batch(probe_population, var_gpu_ctx))
    np.testing.assert_array_equal(gpu_costs, cpu_costs)

  gpu_ctx = ProblemContext(canonical, xp=cp, seed=11, max_matrix_dimension=0)
  for mode in ("multi_move", "naive"):
    ls = LocalSearchGPUTwoOpt(mode=mode, max_iterations=5, candidate_k=3, max_matrix_dimension=0)
    improved = cp.asnumpy(ls.improve_batch(probe_population, gpu_ctx))
    after_costs = EvaluationCPUCost().evaluate_batch(improved, cpu_ctx)
    assert (after_costs <= cpu_costs).all()
