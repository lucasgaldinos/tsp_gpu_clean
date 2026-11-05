"""Hermetic integration tests for the benchmarking pipeline and analysis components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking import (
  ProblemConfig,
  StatisticalAnalyzer,
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
  compare_algorithms,
  compare_algorithms_by_group,
  generate_result_tables,
  load_all_configs,
  load_problems_config,
  perform_cross_problem_analysis_stratified,
  run_comprehensive_benchmark,
  validate_configs,
)
from loaders.tsplib_loader import TsplibLoader
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from pathlib import Path

  from benchmarking.analysis.statistics import StatisticalSummary
  from data_models.records import NemenyiResult

FIXTURE_NAMES: tuple[str, ...] = ("burma14", "gr17", "bayg29", "att48", "berlin52")


def test_problem_configs_from_tsp_files(fixture_tsp_dir: Path) -> None:
  """Verify fixture .tsp files build valid ProblemConfigs and load properly via TsplibLoader."""
  tsp_file_paths: list[Path] = [fixture_tsp_dir / f"{name}.tsp" for name in FIXTURE_NAMES]
  expected: dict[str, tuple[int, float, str]] = {
    "burma14": (14, 3323.0, "GEO"),
    "gr17": (17, 2085.0, "EXPLICIT"),
    "bayg29": (29, 1610.0, "EXPLICIT"),
    "att48": (48, 10628.0, "ATT"),
    "berlin52": (52, 7542.0, "EUC_2D"),
  }
  configs, path_map = build_problem_configs_from_tsp_files(
    tsp_file_paths,
    known_optimals={name: optimal for name, (_size, optimal, _edge) in expected.items()},
  )
  assert len(configs) == 5

  assert {c.name for c in configs} == set(expected.keys())

  with TsplibLoader(fixture_tsp_dir) as loader:
    for cfg in configs:
      exp_size, exp_opt, exp_edge_type = expected[cfg.name]
      assert cfg.size == exp_size
      assert cfg.optimal == exp_opt
      problem = loader.load(path_map[cfg.name])
      assert problem.dimension == exp_size
      assert problem.edge_type == exp_edge_type
      ctx = ProblemContext(problem, xp=np)
      mat = ctx.get_cpu_distances()
      assert mat.shape == (exp_size, exp_size)
      assert np.all(np.diag(mat) == 0)


def test_problem_configs_from_duckdb(fixture_duckdb_path: Path, fixture_config_dir: Path) -> None:
  """Verify building ProblemConfigs from DuckDB fixture database.

  The fixture database carries no `solutions` rows, so the optimal costs must come from
  the `configs/problems.json` catalogue that production orchestration passes as
  `existing_configs` (`DP11-4`), not from a hardcoded table.
  """
  catalogue: list[ProblemConfig] = load_problems_config(fixture_config_dir)
  db_configs = build_problem_configs_from_duckdb(
    db_path=fixture_duckdb_path,
    problem_names=FIXTURE_NAMES,
    existing_configs=catalogue,
  )
  assert len(db_configs) == 5
  assert [c.name for c in db_configs] == list(FIXTURE_NAMES)
  catalogue_optimals = {cfg.name: cfg.optimal for cfg in catalogue}
  assert [cfg.optimal for cfg in db_configs] == [catalogue_optimals[name] for name in FIXTURE_NAMES]


def test_validate_configs_allows_none_patience(fixture_config_dir: Path) -> None:
  """Verify load_all_configs and validate_configs handle configurations cleanly."""
  configs: dict[str, Any] = load_all_configs(fixture_config_dir)
  warnings: list[str] = validate_configs(configs)
  assert warnings == []


def test_statistical_analyzer_small_and_degenerate_samples() -> None:
  """Verify StatisticalAnalyzer handles n=1, zero variance, and Nemenyi post-hoc cleanly."""
  analyzer = StatisticalAnalyzer()

  # n=1 sample size should not divide by zero
  summary_n1: StatisticalSummary = analyzer.paired_comparison(
    np.array([2085.0]),
    np.array([2090.0]),
    "CPU vs GPU",
    "cost",
  )
  assert summary_n1.sample_size == 1
  assert summary_n1.test_used == "no_test_needed"

  # Tied differences in Wilcoxon should not raise ValueError
  summary_ties: StatisticalSummary = analyzer.paired_comparison(
    np.array([100.0, 100.0, 105.0, 110.0]),
    np.array([100.0, 100.0, 100.0, 100.0]),
    "A vs B",
    "cost",
  )
  assert 0.0 <= summary_ties.p_value <= 1.0

  # Nemenyi post-hoc
  nemenyi: NemenyiResult = analyzer.nemenyi_posthoc(
    [
      np.array([10.0, 12.0, 11.0, 13.0]),
      np.array([5.0, 6.0, 5.5, 6.5]),
      np.array([1.0, 1.2, 1.1, 1.3]),
    ],
    algorithm_names=["CPU", "HybridNaive", "HybridOptimized"],
  )
  assert nemenyi["p_values_matrix"].shape == (3, 3)
  assert nemenyi["critical_distance"] > 0


def test_report_generator_table2_speedup(tmp_path: Path) -> None:
  """Verify Table 2 computes Avg Speedup for algorithms against baseline."""
  all_results: dict[str, dict[str, dict[str, float]]] = {
    "gr17": {
      "CPU": {
        "mean_cost": 2085.0,
        "std_cost": 0.0,
        "mean_gap": 0.0,
        "std_gap": 0.0,
        "mean_time": 0.80,
        "std_time": 0.01,
      },
      "FullGPUMonolithic": {
        "mean_cost": 2085.0,
        "std_cost": 0.0,
        "mean_gap": 0.0,
        "std_gap": 0.0,
        "mean_time": 0.02,
        "std_time": 0.001,
      },
      "HybridNaive": {
        "mean_cost": 2085.0,
        "std_cost": 0.0,
        "mean_gap": 0.0,
        "std_gap": 0.0,
        "mean_time": 0.40,
        "std_time": 0.01,
      },
      "HybridOptimized": {
        "mean_cost": 2085.0,
        "std_cost": 0.0,
        "mean_gap": 0.0,
        "std_gap": 0.0,
        "mean_time": 0.04,
        "std_time": 0.002,
      },
    }
  }
  problem_configs: list[ProblemConfig] = [ProblemConfig(name="gr17", optimal=2085.0, size=17)]
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="test_table",
  )
  md_content: str = (tmp_path / "test_table.md").read_text(encoding="utf-8")
  assert "20.00x" in md_content
  assert "0.50x" in md_content


def test_run_comprehensive_benchmark_fixture(
  fixture_config_dir: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """Verify run_comprehensive_benchmark executes end-to-end with fixture configs and data."""
  gr17_path: Path = fixture_tsp_dir / "gr17.tsp"
  results: dict[str, dict[str, dict[str, Any]]] = run_comprehensive_benchmark(
    config_dir=fixture_config_dir,
    output_dir=tmp_path,
    tsp_files=[gr17_path],
    algorithms=["CPU"],
    repetitions=1,
    skip_cpu=False,
    benchmark_name="gr17_smoke",
  )
  assert "gr17" in results
  assert "CPU" in results["gr17"]
  assert results["gr17"]["CPU"]["mean_cost"] >= 2085.0
  assert (tmp_path / "tables" / "gr17_smoke.md").exists()
  assert (tmp_path / "tables" / "gr17_smoke.tex").exists()

  # `BUG-18`: a checkpoint must describe the strategy parameters it actually ran.
  algorithm_config = results["gr17"]["CPU"]["algorithm_config"]
  strategy_params = algorithm_config["strategy_params"]
  assert strategy_params["initialization"]["class"] == "InitializationRandom"
  assert strategy_params["selection"]["k"] == 3, "configured tournament size must survive into the checkpoint"
  assert strategy_params["local_search"]["max_iterations"] >= 1
  assert results["gr17"]["CPU"]["construction_heuristic"] == "InitializationRandom"


def test_compare_algorithms_and_grouping() -> None:
  """Verify compare_algorithms, grouping, and stratified cross-problem analysis."""
  all_results = {
    "gr17": {
      "CPU": {"mean_gap": 0.5, "mean_time": 1.0, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.8, "mean_time": 0.5, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.1, "mean_time": 0.05, "edge_type": "EXPLICIT"},
      "FullGPUMonolithic": {"mean_gap": 0.2, "mean_time": 0.02, "edge_type": "EXPLICIT"},
    },
    "gr21": {
      "CPU": {"mean_gap": 0.6, "mean_time": 1.2, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.7, "mean_time": 0.6, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.0, "mean_time": 0.06, "edge_type": "EXPLICIT"},
      "FullGPUMonolithic": {"mean_gap": 0.3, "mean_time": 0.03, "edge_type": "EXPLICIT"},
    },
    "gr24": {
      "CPU": {"mean_gap": 0.4, "mean_time": 1.1, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.9, "mean_time": 0.55, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.05, "mean_time": 0.07, "edge_type": "EXPLICIT"},
      "FullGPUMonolithic": {"mean_gap": 0.25, "mean_time": 0.025, "edge_type": "EXPLICIT"},
    },
    "berlin52": {
      "HybridNaive": {"mean_gap": 1.2, "mean_time": 2.0, "edge_type": "EUC_2D"},
      "HybridOptimized": {"mean_gap": 0.3, "mean_time": 0.2, "edge_type": "EUC_2D"},
      "FullGPUMonolithic": {"mean_gap": 0.4, "mean_time": 0.08, "edge_type": "EUC_2D"},
    },
  }
  problems: list[str] = ["gr17", "gr21", "gr24"]
  algorithms: list[str] = ["CPU", "HybridNaive", "HybridOptimized", "FullGPUMonolithic"]

  analyzer = StatisticalAnalyzer()
  res: dict[str, Any] | None = compare_algorithms(
    all_results=all_results,
    problems=problems,
    algorithms=algorithms,
    analyzer=analyzer,
    baseline_algorithm="HybridNaive",
    label="Test Comparison",
  )
  assert res is not None
  assert res["problems"] == problems
  assert res["algorithms"] == algorithms
  assert res["n_problems"] == 3
  assert res["n_algorithms"] == 4
  assert res["friedman_test"]["statistic"] >= 0.0
  assert 0.0 <= res["friedman_test"]["p_value"] <= 1.0
  # Geometric mean of the literal baseline/algo mean_time ratios for each algorithm.
  assert res["speedups"]["HybridNaive"] == pytest.approx(1.0)
  assert res["speedups"]["FullGPUMonolithic"] == pytest.approx(22.2398, rel=1e-4)
  assert res["speedups"]["HybridOptimized"] == pytest.approx(9.2276, rel=1e-4)

  # Degenerate: less than 3 problems or algorithms returns None
  assert compare_algorithms(all_results, ["gr17"], algorithms, analyzer) is None
  assert compare_algorithms(all_results, problems, ["CPU", "FullGPUMonolithic"], analyzer) is None

  # Grouping test
  p_configs: list[ProblemConfig] = [
    ProblemConfig(name="gr17", optimal=2085.0, size=17),
    ProblemConfig(name="gr21", optimal=2707.0, size=21),
    ProblemConfig(name="gr24", optimal=1272.0, size=24),
    ProblemConfig(name="berlin52", optimal=7542.0, size=52),
  ]
  group_res: dict[str, Any] = compare_algorithms_by_group(
    all_results=all_results,
    problem_configs=p_configs,
    group_key="edge_type",
    analyzer=analyzer,
  )
  assert set(group_res) == {"EXPLICIT", "EUC_2D"}
  explicit_group: dict[str, Any] | None = group_res["EXPLICIT"]
  assert explicit_group is not None
  assert explicit_group["n_problems"] == 3
  assert explicit_group["n_algorithms"] == 4
  assert set(explicit_group["problems"]) == {"gr17", "gr21", "gr24"}
  # berlin52 is the only EUC_2D problem, so that group cannot reach the 3-problem minimum.
  assert group_res["EUC_2D"] is None

  # Stratified cross-problem analysis
  strat_res: dict[str, Any] = perform_cross_problem_analysis_stratified(
    all_results=all_results,
    problem_configs=p_configs,
    analyzer=analyzer,
    cpu_threshold=30,
  )
  assert strat_res["cpu_threshold"] == 30
  assert strat_res["total_problems"] == 4
  assert strat_res["total_algorithms"] == 4

  stratum1: dict[str, Any] | None = strat_res["stratum1_small_problems"]
  assert stratum1 is not None
  assert stratum1["n_problems"] == 3
  assert stratum1["n_algorithms"] == 4
  assert set(stratum1["problems"]) == {"gr17", "gr21", "gr24"}

  stratum2: dict[str, Any] | None = strat_res["stratum2_all_gpu_problems"]
  assert stratum2 is not None
  assert stratum2["n_problems"] == 4
  assert stratum2["n_algorithms"] == 3
  assert stratum2["algorithms"] == ["FullGPUMonolithic", "HybridNaive", "HybridOptimized"]
  assert stratum2["speedups"]["HybridNaive"] == pytest.approx(1.0)
  assert stratum2["speedups"]["FullGPUMonolithic"] == pytest.approx(22.8999, rel=1e-4)
  assert stratum2["speedups"]["HybridOptimized"] == pytest.approx(9.4149, rel=1e-4)

  # CPU covers only the three small problems; FullGPU covers all four.
  assert strat_res["summary_statistics"]["CPU"]["n_problems"] == 3
  assert strat_res["summary_statistics"]["FullGPUMonolithic"]["n_problems"] == 4


def test_seed_override_drives_repetition_seeds(fixture_config_dir: Path, fixture_tsp_dir: Path, tmp_path: Path) -> None:
  """Verify that CLI/orchestration seed parameter drives algorithm repetition seeds."""
  gr17_path = fixture_tsp_dir / "gr17.tsp"
  benchmark_results = run_comprehensive_benchmark(
    config_dir=fixture_config_dir,
    output_dir=tmp_path,
    tsp_files=[gr17_path],
    algorithms=["CPU"],
    repetitions=2,
    seed=7,
  )
  assert "gr17" in benchmark_results
  assert "CPU" in benchmark_results["gr17"]
  seeds = benchmark_results["gr17"]["CPU"]["raw_seeds"]
  assert seeds == [7, 8]


def test_strategy_override_reaches_instance(fixture_config_dir: Path) -> None:
  """Verify strategy parameter overrides modify instantiated strategy properties and reject invalid keys."""
  configs = load_all_configs(fixture_config_dir)
  ga_params = configs["ga_params"]
  algo_configs = configs["algorithms"]

  target_cfg = algo_configs.get("HybridOptimized") or algo_configs.get("FullGPUModular") or algo_configs["CPU"]

  inst = target_cfg.instantiate(
    ga_params,
    strategy_overrides={"local_search": {"max_iterations": 20}},
  )
  assert inst.local_search.max_iterations == 20

  with pytest.raises(ValueError, match="does not accept parameter 'non_existent_param'"):
    target_cfg.instantiate(
      ga_params,
      strategy_overrides={"local_search": {"non_existent_param": 999}},
    )

  with pytest.raises(ValueError, match="Unknown strategy slot 'unknown_slot'"):
    target_cfg.instantiate(
      ga_params,
      strategy_overrides={"unknown_slot": {"param": 1}},
    )

  # Monolithic strategy-override mapping is covered by the opt-in archive config
  # (`configs/algorithms.archive.json`), which the suite no longer loads: no test
  # executes archive code.
