"""
Integration tests for the adjusted Benchmark V2 API and the 10 selected TSPLIB problems.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking import (
  DEFAULT_SELECTED_TSP_FILES,
  KNOWN_TSPLIB_OPTIMALS,
  ProblemConfig,
  StatisticalAnalyzer,
  build_problem_configs_from_tsp_files,
  compare_algorithms,
  compare_algorithms_by_group,
  generate_result_tables,
  load_all_configs,
  perform_cross_problem_analysis_stratified,
  run_comprehensive_benchmark,
  validate_configs,
)
from benchmarking.analysis.statistics import StatisticalSummary
from loaders.tsplib_loader import TsplibLoader
from protocols.problem_context import ProblemContext

if TYPE_CHECKING:
  from benchmarking.analysis.statistics import StatisticalSummary
  from data_models.records import NemenyiResult


def test_selected_10_tsp_files_parse_and_match_optimals() -> None:
  """Verify all 10 target .tsp files parse via tsplib95 and build valid ProblemConfigs."""
  assert len(DEFAULT_SELECTED_TSP_FILES) == 10
  configs, path_map = build_problem_configs_from_tsp_files(DEFAULT_SELECTED_TSP_FILES)

  expected: dict[str, tuple[int, float, str]] = {
    "att48": (48, 10628.0, "ATT"),
    "bayg29": (29, 1610.0, "EXPLICIT"),
    "bays29": (29, 2020.0, "EXPLICIT"),
    "berlin52": (52, 7542.0, "EUC_2D"),
    "brg180": (180, 1950.0, "EXPLICIT"),
    "gr17": (17, 2085.0, "EXPLICIT"),
    "gr21": (21, 2707.0, "EXPLICIT"),
    "gr24": (24, 1272.0, "EXPLICIT"),
    "gr48": (48, 5046.0, "EXPLICIT"),
    "gr96": (96, 55209.0, "GEO"),
  }

  assert {c.name for c in configs} == set(expected.keys())

  with TsplibLoader() as loader:
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


def test_validate_configs_allows_none_patience() -> None:
  """Verify load_all_configs and validate_configs handle patience=null in benchmark.json."""
  config_dir: Path = Path(__file__).resolve().parent.parent / "configs"
  configs: dict[str, Any] = load_all_configs(config_dir)
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

  # Nemenyi post-hoc should work even without external posthocs library
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


def test_report_generator_table2_speedup_all_algorithms(tmp_path: Path) -> None:
  """Verify Table 2 computes Avg Speedup for algorithms alphabetically before baseline."""
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
      "FullGPU": {
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
  # HybridNaive is baseline (0.40s), so FullGPU (0.02s) should have 20.00x speedup in Table 2
  assert "20.00x" in md_content
  assert "0.50x" in md_content


def test_run_comprehensive_benchmark_custom_tsp_file(tmp_path: Path) -> None:
  """Verify run_comprehensive_benchmark executes end-to-end on a custom .tsp path."""
  gr17_path: Path = DEFAULT_SELECTED_TSP_FILES[5]
  args = Namespace(
    repetitions=1,
    skip_cpu=True,
    test_mode=False,
    incremental_runs=False,
  )
  results: dict[str, dict[str, dict[str, Any]]] = run_comprehensive_benchmark(
    args,
    tsp_files=[gr17_path],
    algorithms=["HybridOptimized", "FullGPU"],
    output_dir=tmp_path,
    benchmark_name="gr17_smoke",
  )
  assert "gr17" in results
  assert results["gr17"]["HybridOptimized"]["mean_cost"] >= KNOWN_TSPLIB_OPTIMALS["gr17"]
  assert results["gr17"]["FullGPU"]["mean_cost"] >= KNOWN_TSPLIB_OPTIMALS["gr17"]
  assert (tmp_path / "tables" / "gr17_smoke.md").exists()
  assert (tmp_path / "tables" / "gr17_smoke.tex").exists()


def test_compare_algorithms_and_grouping() -> None:
  """Verify compare_algorithms, grouping, and stratified cross-problem analysis."""
  all_results = {
    "gr17": {
      "CPU": {"mean_gap": 0.5, "mean_time": 1.0, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.8, "mean_time": 0.5, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.1, "mean_time": 0.05, "edge_type": "EXPLICIT"},
      "FullGPU": {"mean_gap": 0.2, "mean_time": 0.02, "edge_type": "EXPLICIT"},
    },
    "gr21": {
      "CPU": {"mean_gap": 0.6, "mean_time": 1.2, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.7, "mean_time": 0.6, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.0, "mean_time": 0.06, "edge_type": "EXPLICIT"},
      "FullGPU": {"mean_gap": 0.3, "mean_time": 0.03, "edge_type": "EXPLICIT"},
    },
    "gr24": {
      "CPU": {"mean_gap": 0.4, "mean_time": 1.1, "edge_type": "EXPLICIT"},
      "HybridNaive": {"mean_gap": 0.9, "mean_time": 0.55, "edge_type": "EXPLICIT"},
      "HybridOptimized": {"mean_gap": 0.05, "mean_time": 0.07, "edge_type": "EXPLICIT"},
      "FullGPU": {"mean_gap": 0.25, "mean_time": 0.025, "edge_type": "EXPLICIT"},
    },
    "berlin52": {
      "HybridNaive": {"mean_gap": 1.2, "mean_time": 2.0, "edge_type": "EUC_2D"},
      "HybridOptimized": {"mean_gap": 0.3, "mean_time": 0.2, "edge_type": "EUC_2D"},
      "FullGPU": {"mean_gap": 0.4, "mean_time": 0.08, "edge_type": "EUC_2D"},
    },
  }
  problems: list[str] = ["gr17", "gr21", "gr24"]
  algorithms: list[str] = ["CPU", "HybridNaive", "HybridOptimized", "FullGPU"]

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
  assert res["n_problems"] == 3
  assert res["n_algorithms"] == 4
  assert "friedman_test" in res
  assert "speedups" in res
  assert res["speedups"]["FullGPU"] > 1.0

  # Degenerate: less than 3 problems or algorithms returns None
  assert compare_algorithms(all_results, ["gr17"], algorithms, analyzer) is None
  assert compare_algorithms(all_results, problems, ["CPU", "FullGPU"], analyzer) is None

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
  assert "EXPLICIT" in group_res
  assert group_res["EXPLICIT"] is not None
  assert group_res["EXPLICIT"]["n_problems"] == 3

  # Stratified cross-problem analysis
  strat_res: dict[str, Any] = perform_cross_problem_analysis_stratified(
    all_results=all_results,
    problem_configs=p_configs,
    analyzer=analyzer,
    cpu_threshold=30,
  )
  assert strat_res["stratum1_small_problems"] is not None
  assert strat_res["stratum2_all_gpu_problems"] is not None
