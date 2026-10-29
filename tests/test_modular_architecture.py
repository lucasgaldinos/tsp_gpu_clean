"""
Tests for the modular Genetic Algorithm architecture, strategy contracts,
flexible 2-opt placement stages, and TSPLIB95 loader / DuckDB ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from algorithms import (
  GeneticAlgorithm,
  GeneticAlgorithmCPU,
  GeneticAlgorithmFullGPUEarlyStop,
  GeneticAlgorithmFullGPUModular,
  GeneticAlgorithmFullGPUMonolithic,
  GeneticAlgorithmHybridNaive,
  GeneticAlgorithmHybridOptimized,
)
from algorithms.strategies import (
  CPUCostEvaluation,
  CPUTwoOpt,
  InversionMutation,
  NoOpLocalSearch,
  OrderCrossover,
  PartiallyMappedCrossover,
  RandomPermutationInitialization,
  RouletteWheelSelection,
  SwapMutation,
  TournamentSelection,
)
from benchmarking.runner.config_loader import load_algorithms_config
from loaders import (
  DatabaseLoader,
  TsplibLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  parse_tsp_file_with_tsplib95,
)
from protocols import (
  CrossoverStrategy,
  EvaluationStrategy,
  Fitness,
  InitializationStrategy,
  LocalSearchStrategy,
  MutationStrategy,
  ProblemContext,
  Route,
  RouteOrRoutes,
  Routes,
  SelectionStrategy,
  ensure_routes_batch,
  restore_routes_shape,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray

  from data_models.problem import Problem
  from loaders.tsplib_loader import TsplibIngestionBundle

try:
  import cupy as cp

  try:
    _ = cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
  except Exception:
    GPU_AVAILABLE = False
except ImportError:
  GPU_AVAILABLE = False
  cp = None


@pytest.fixture
def berlin52_problem() -> Problem:
  """Load berlin52 problem via TsplibLoader."""
  with TsplibLoader("datasets/tsp") as loader:
    return loader.load("berlin52")


@pytest.fixture
def burma14_problem() -> Problem:
  """Load burma14 problem (small instance, N=14) via TsplibLoader."""
  with TsplibLoader("datasets/tsp") as loader:
    return loader.load("burma14")


# =============================================================================
# 1. TSPLIB95 Loader & DuckDB Ingestion Tests
# =============================================================================


def test_tsplib95_parse_and_bundle_to_problem() -> None:
  """Verify parsing EUC_2D and EXPLICIT .tsp instances with tsplib95."""
  bundle_euc: TsplibIngestionBundle = parse_tsp_file_with_tsplib95("datasets/tsp/berlin52.tsp", problem_id=1)
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

  bundle_exp: TsplibIngestionBundle = parse_tsp_file_with_tsplib95("datasets/tsp/brg180.tsp", problem_id=2)
  assert bundle_exp.problem.name == "brg180"
  assert bundle_exp.problem.edge_weight_type == "EXPLICIT"
  assert bundle_exp.edge_matrix is not None

  problem_exp: Problem = bundle_to_problem(bundle_exp)
  assert problem_exp.dimension == 180
  assert problem_exp.distances is not None
  assert problem_exp.distances.shape == (180, 180)


def test_database_loader_fallback_and_duckdb_ingestion(tmp_path: Path) -> None:
  """Verify DatabaseLoader fallback to TsplibLoader and full DuckDB ingestion."""
  # 1. Test automatic fallback when routing.duckdb does not exist yet
  missing_default_db = tmp_path / "routing.duckdb"
  with DatabaseLoader(str(missing_default_db), fallback_tsp_dir="datasets/tsp") as loader:
    prob = loader.load("berlin52")
    assert prob.name == "berlin52"
    assert prob.dimension == 52

  # 2. Test ingesting a subset of .tsp files into a temporary DuckDB file
  mini_tsp_dir: Path = tmp_path / "mini_tsp"
  mini_tsp_dir.mkdir()
  for name in ("berlin52.tsp", "brg180.tsp"):
    src_file: Path = Path("datasets/tsp") / name
    (mini_tsp_dir / name).write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

  ingested_db: Path = tmp_path / "ingested.duckdb"
  count = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, ingested_db)
  assert count == 2
  assert ingested_db.exists()

  with DatabaseLoader(str(ingested_db), fallback_tsp_dir=None) as db_loader:
    p1 = db_loader.load("berlin52")
    p2 = db_loader.load("brg180")
    assert p1.dimension == 52
    assert p2.dimension == 180
    assert p2.distances is not None


# =============================================================================
# 2. Route / Batch Contract & Strategy Protocol Tests
# =============================================================================


def test_ensure_and_restore_routes_batch() -> None:
  """Verify 1D route and 2D route batch shape normalization."""
  single_route: NDArray[np.int32] = np.array([0, 2, 1, 3], dtype=np.int32)
  batch_from_1d, was_1d = ensure_routes_batch(single_route)
  assert was_1d is True
  assert batch_from_1d.shape == (1, 4)
  restored_1d: RouteOrRoutes = restore_routes_shape(batch_from_1d, was_1d)
  assert restored_1d.shape == (4,)

  batch_routes: NDArray[np.int32] = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
  batch_from_2d, was_1d_2 = ensure_routes_batch(batch_routes)
  assert was_1d_2 is False
  assert batch_from_2d.shape == (2, 4)
  restored_2d: RouteOrRoutes = restore_routes_shape(batch_from_2d, was_1d_2)
  assert restored_2d.shape == (2, 4)


def test_all_strategies_satisfy_protocols_and_route_contracts(burma14_problem: Problem) -> None:
  """Verify all strategies satisfy their Protocol and accept both 1D Route and 2D Routes."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))
  n = len(customers)

  init_strat = RandomPermutationInitialization(seed=42)
  assert isinstance(init_strat, InitializationStrategy)
  routes: Routes = init_strat.initialize(customers, population_size=10, context=context)
  assert routes.shape == (10, n)

  eval_strat = CPUCostEvaluation()
  assert isinstance(eval_strat, EvaluationStrategy)
  fitness: Fitness = eval_strat.evaluate_batch(routes, context)
  assert fitness.shape == (10,)
  single_cost: float = eval_strat.evaluate(routes[0], context)
  assert np.isclose(single_cost, fitness[0])

  # Selection strategies
  for sel in (TournamentSelection(k=3, seed=42), RouletteWheelSelection(seed=42)):
    assert isinstance(sel, SelectionStrategy)
    one_parent: Route = sel.select(routes, fitness, context=context)
    assert one_parent.shape == (n,)
    batch_parents: Routes = sel.select_batch(routes, fitness, num_parents=10, context=context)
    assert batch_parents.shape == (10, n)

  # Crossover strategies (1D pair and 2D batch)
  for cx in (OrderCrossover(seed=42), PartiallyMappedCrossover(seed=42)):
    assert isinstance(cx, CrossoverStrategy)
    child_1d: Route = cx.crossover(routes[0], routes[1], context=context)
    assert child_1d.shape == (n,)
    assert sorted(child_1d.tolist()) == customers

    children_2d: RouteOrRoutes = cx.crossover_batch(routes, context=context)
    assert children_2d.shape == (10, n)
    for row in children_2d:
      assert sorted(row.tolist()) == customers

  # Mutation strategies (1D route and 2D batch)
  for mut in (SwapMutation(seed=42), InversionMutation(seed=42)):
    assert isinstance(mut, MutationStrategy)
    mut_1d: RouteOrRoutes = mut.mutate_batch(routes[0], mutation_rate=1.0, context=context)
    assert mut_1d.shape == (n,)
    assert sorted(mut_1d.tolist()) == customers

    mut_2d: RouteOrRoutes = mut.mutate_batch(routes, mutation_rate=0.5, context=context)
    assert mut_2d.shape == (10, n)
    for row in mut_2d:
      assert sorted(row.tolist()) == customers

  # Local search strategies (1D route and 2D batch)
  for ls in (CPUTwoOpt(max_iterations=5), NoOpLocalSearch()):
    assert isinstance(ls, LocalSearchStrategy)
    imp_1d: RouteOrRoutes = ls.improve_batch(routes[0], context)
    assert imp_1d.shape == (n,)
    assert sorted(imp_1d.tolist()) == customers
    assert eval_strat.evaluate(imp_1d, context) <= eval_strat.evaluate(routes[0], context) + 1e-6

    imp_2d: RouteOrRoutes = ls.improve_batch(routes, context)
    assert imp_2d.shape == (10, n)


# =============================================================================
# 3. Flexible 2-opt Placement Stages Tests
# =============================================================================


@pytest.mark.parametrize(
  "stage",
  [
    "post_crossover",
    "post_mutation",
    "post_evaluation",
    "post_evolution",
    "none",
  ],
)
def test_ga_composable_local_search_stages(burma14_problem: Problem, stage: str) -> None:
  """Verify 2-opt can be placed after OX, after mutation, after fitness, or at the end of GA."""
  context = ProblemContext(burma14_problem, xp=np, seed=42)
  customers: list[int] = list(range(burma14_problem.dimension))

  ga: GeneticAlgorithm = GeneticAlgorithm.cpu(
    population_size=16,
    mutation_rate=0.05,
    tournament_size=3,
    two_opt_iterations=4,
    seed=42,
    local_search_stage=stage,  # type: ignore[arg-type]
  )
  best_tour, stats = ga.evolve(context, customers, max_generations=6, patience=10)
  assert best_tour.shape == (burma14_problem.dimension,)
  assert sorted(best_tour.tolist()) == customers
  assert stats["best_fitness"] <= stats["initial_fitness"] + 1e-6
  assert stats["local_search_stage"] == stage


# =============================================================================
# 4. Benchmark Config & GPU Variant Preset Tests
# =============================================================================


def test_benchmark_algorithms_json_loads_all_classes() -> None:
  """Verify configs/algorithms.json resolves all modular and monolithic classes."""
  config_dir: Path = Path(__file__).resolve().parent.parent / "configs"
  ga_params, alg_configs = load_algorithms_config(config_dir)
  assert set(alg_configs.keys()) == {
    "CPU",
    "HybridNaive",
    "HybridOptimized",
    "FullGPUModular",
    "FullGPU",
    "FullGPUMonolithic",
  }

  for name, cfg in alg_configs.items():
    cls: type = cfg.get_class()
    instance = cls(**ga_params.to_dict())
    assert hasattr(instance, "evolve"), f"{name} missing evolve() method"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA GPU not available")
def test_gpu_variants_and_modular_vs_monolithic(burma14_problem: Problem) -> None:
  """Verify HybridNaive, HybridOptimized, FullGPUModular, and FullGPUMonolithic on GPU."""
  customers: list[int] = list(range(burma14_problem.dimension))

  variants = [
    GeneticAlgorithmCPU(population_size=32, two_opt_iterations=5, seed=42),
    GeneticAlgorithmHybridNaive(population_size=32, two_opt_iterations=5, seed=42),
    GeneticAlgorithmHybridOptimized(population_size=32, two_opt_iterations=5, seed=42),
    GeneticAlgorithmFullGPUModular(population_size=32, two_opt_iterations=5, seed=42),
    GeneticAlgorithmFullGPUMonolithic(population_size=32, two_opt_iterations=5, seed=42),
    GeneticAlgorithmFullGPUEarlyStop(population_size=32, two_opt_iterations=5, seed=42),
  ]

  for algo in variants:
    xp: Any = cp if not isinstance(algo, GeneticAlgorithmCPU) else np
    context = ProblemContext(burma14_problem, xp=xp, seed=42)
    best_tour, stats = algo.evolve(context, customers, max_generations=10, patience=5)
    assert len(best_tour) == burma14_problem.dimension
    assert sorted(best_tour.tolist()) == customers
    assert stats["best_fitness"] > 0
    assert "generations_completed" in stats
    assert "stop_reason" in stats
