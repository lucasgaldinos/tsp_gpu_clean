"""Hermetic integration tests for the benchmarking pipeline and analysis components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking import (
  ProblemAnalysis,
  ProblemConfig,
  StatisticalAnalyzer,
  analyze_cohort,
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
  compare_algorithms,
  generate_result_tables,
  load_all_configs,
  load_problems_config,
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
    baseline_algorithm="HybridNaive",
  )
  md_path = tmp_path / "test_table.md"
  tex_path = tmp_path / "test_table.tex"
  assert md_path.exists()
  assert tex_path.exists()

  md_content: str = md_path.read_text(encoding="utf-8")
  tex_content: str = tex_path.read_text(encoding="utf-8")

  # Verify relative speedup ratios against HybridNaive (0.40s):
  # FullGPUMonolithic (0.02s) -> 20.00x (n=1), CPU (0.80s) -> 0.50x (n=1), HybridNaive -> 1.00x
  assert "20.00x (n=1)" in md_content
  assert "0.50x (n=1)" in md_content
  assert "1.00x" in md_content
  assert "20.00$\\times$ ($n=1$)" in tex_content
  assert "0.50$\\times$ ($n=1$)" in tex_content


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
    reports=("markdown", "latex"),
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

  # Phase timings persistence (Task 11.7, DP11-7, DP11-13)
  cpu_res = results["gr17"]["CPU"]
  assert "raw_phase_timings" in cpu_res
  assert "mean_phase_timings" in cpu_res
  assert "phase_share_pct" in cpu_res
  assert "phase_share_cv_pct" in cpu_res
  assert len(cpu_res["raw_phase_timings"]) == 1
  assert "initialization" in cpu_res["mean_phase_timings"]
  assert "host_sync" in cpu_res["mean_phase_timings"]
  assert "evaluation" in cpu_res["mean_phase_timings"]


def test_run_comprehensive_benchmark_forwards_baseline_algorithm(
  fixture_config_dir: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """Verify run_comprehensive_benchmark forwards baseline_algorithm to statistical and cross-problem analysis.

  WHY: Task 12.39 (BUG-55) & Task 14.2 (BUG-92) — Prevent silent baseline divergence across layers.
  """
  from unittest.mock import patch

  gr17_path: Path = fixture_tsp_dir / "gr17.tsp"
  bayg29_path: Path = fixture_tsp_dir / "bayg29.tsp"
  with (
    patch("benchmarking.runner.orchestration.analyze_problem") as mock_stat,
    patch("benchmarking.runner.orchestration.analyze_cohort") as mock_cross,
    patch("benchmarking.runner.orchestration.log_problem_analysis"),
    patch("benchmarking.runner.orchestration.log_cohort_analysis"),
  ):
    mock_stat.return_value = ProblemAnalysis(
      problem_name="gr17",
      optimal_cost=2085.0,
      baseline_algorithm="CustomBaseline",
      algorithms=("CPU",),
      alpha=0.05,
      insufficient_data=True,
    )
    mock_cross.return_value = None
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      tsp_files=[gr17_path, bayg29_path],
      algorithms=["CPU"],
      repetitions=1,
      skip_cpu=False,
      benchmark_name="test_baseline_forward",
      baseline_algorithm="CustomBaseline",
    )
    assert mock_stat.called
    _, stat_kwargs = mock_stat.call_args
    assert stat_kwargs.get("baseline_algorithm") == "CustomBaseline"
    assert mock_cross.called
    _, cross_kwargs = mock_cross.call_args
    assert cross_kwargs.get("baseline_algorithm") == "CustomBaseline"


def test_compare_algorithms_and_stratified_cohort_analysis() -> None:
  """Verify compare_algorithms and stratified CohortAnalysis."""
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
  res = compare_algorithms(
    all_results=all_results,
    problems=problems,
    algorithms=algorithms,
    analyzer=analyzer,
    baseline_algorithm="HybridNaive",
    label="Test Comparison",
  )
  assert res is not None
  assert list(res.problems) == problems
  assert list(res.algorithms) == algorithms
  assert res.omnibus is not None
  assert res.omnibus.n_blocks == 3
  assert res.omnibus.k_algorithms == 4
  assert res.omnibus.chi2_stat >= 0.0
  assert 0.0 <= res.omnibus.chi2_pvalue <= 1.0
  assert res.speedups["HybridNaive"].geomean == pytest.approx(1.0)
  assert res.speedups["FullGPUMonolithic"].geomean == pytest.approx(22.2398, rel=1e-4)
  assert res.speedups["HybridOptimized"].geomean == pytest.approx(9.2276, rel=1e-4)

  # Degenerate: < 2 problems or < 3 algorithms returns None
  assert compare_algorithms(all_results, ["gr17"], algorithms, analyzer=analyzer) is None
  assert compare_algorithms(all_results, problems, ["CPU", "FullGPUMonolithic"], analyzer=analyzer) is None

  p_configs: list[ProblemConfig] = [
    ProblemConfig(name="gr17", optimal=2085.0, size=17),
    ProblemConfig(name="gr21", optimal=2707.0, size=21),
    ProblemConfig(name="gr24", optimal=1272.0, size=24),
    ProblemConfig(name="berlin52", optimal=7542.0, size=52),
  ]

  # Stratified cross-problem analysis
  strat_res = analyze_cohort(
    all_results,
    p_configs,
    alpha=analyzer.alpha,
    cpu_threshold=30,
  )
  assert strat_res.cpu_threshold == 30
  assert len(strat_res.cohort.problems) == 4
  assert len(strat_res.cohort.algorithms) == 4
  assert len(strat_res.strata) == 2

  stratum1 = strat_res.strata[0]
  assert stratum1.omnibus is not None
  assert stratum1.omnibus.n_blocks == 3
  assert stratum1.omnibus.k_algorithms == 4
  assert set(stratum1.problems) == {"gr17", "gr21", "gr24"}

  stratum2 = strat_res.strata[1]
  assert stratum2.omnibus is not None
  assert stratum2.omnibus.n_blocks == 4
  assert stratum2.omnibus.k_algorithms == 3
  assert list(stratum2.algorithms) == ["HybridNaive", "HybridOptimized", "FullGPUMonolithic"]
  assert stratum2.speedups["HybridNaive"].geomean == pytest.approx(1.0)
  assert stratum2.speedups["FullGPUMonolithic"].geomean == pytest.approx(22.8999, rel=1e-4)
  assert stratum2.speedups["HybridOptimized"].geomean == pytest.approx(9.4149, rel=1e-4)

  # Unified cross-problem analysis with cpu_threshold=None
  strat_res_none = analyze_cohort(
    all_results,
    p_configs,
    alpha=analyzer.alpha,
    cpu_threshold=None,
    baseline_algorithm="CPU",
  )
  assert strat_res_none.strata == ()
  assert strat_res_none.cpu_threshold is None
  assert strat_res_none.baseline_algorithm == "CPU"
  assert strat_res_none.quality_omnibus is not None
  assert strat_res_none.quality_omnibus.n_blocks == 3
  assert strat_res_none.time_omnibus is not None
  assert strat_res_none.time_omnibus.n_blocks == 3
  assert strat_res_none.baseline_speedups["CPU"].geomean == pytest.approx(1.0)

  # CPU covers only the three small problems; FullGPU covers all four.
  assert strat_res.per_algorithm["CPU"].n_problems == 3
  assert strat_res.per_algorithm["FullGPUMonolithic"].n_problems == 4


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


def test_generate_result_tables_emits_phase_timings_table(tmp_path: Path) -> None:
  """Verify that generate_result_tables emits Table 4 when mean_phase_timings are present (Task 11.7)."""
  problem_configs = [ProblemConfig(name="berlin52", optimal=7542.0, size=52)]
  all_results = {
    "berlin52": {
      "CPU": {
        "mean_time": 0.050,  # < 100 ms to also trigger floor warning
        "std_time": 0.002,
        "mean_cost": 7600.0,
        "std_cost": 10.0,
        "best_cost": 7590.0,
        "mean_gap": 0.77,
        "std_gap": 0.13,
        "best_gap": 0.64,
        "successful_runs": 1,
        "raw_times": [0.050],
        "raw_costs": [7600.0],
        "raw_gaps": [0.77],
        "raw_seeds": [42],
        "mean_phase_timings": {
          "initialization": {"total_ms": 2.0, "calls": 1, "mean_ms": 2.0},
          "evaluation": {"total_ms": 15.0, "calls": 20, "mean_ms": 0.75},
          "selection": {"total_ms": 5.0, "calls": 20, "mean_ms": 0.25},
          "crossover": {"total_ms": 10.0, "calls": 20, "mean_ms": 0.5},
          "mutation": {"total_ms": 5.0, "calls": 20, "mean_ms": 0.25},
          "local_search": {"total_ms": 10.0, "calls": 5, "mean_ms": 2.0},
          "survival": {"total_ms": 4.0, "calls": 20, "mean_ms": 0.2},
          "host_sync": {"total_ms": 3.0, "calls": 21, "mean_ms": 0.14},
        },
        "phase_share_pct": {
          "initialization": 4.0,
          "evaluation": 27.0,
          "selection": 10.0,
          "crossover": 18.0,
          "mutation": 10.0,
          "local_search": 17.0,
          "survival": 8.0,
          "host_sync": 6.0,
        },
        "phase_share_cv_pct": {
          "initialization": 2.5,
          "evaluation": 1.2,
          "selection": 3.0,
          "crossover": 1.8,
          "mutation": 2.1,
          "local_search": 1.5,
          "survival": 3.3,
          "host_sync": 4.0,
        },
      }
    }
  }

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "tables",
    benchmark_name="phase_timings_test",
  )

  md_content = (tmp_path / "tables" / "phase_timings_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "tables" / "phase_timings_test.tex").read_text(encoding="utf-8")

  assert "## Table 4: Average Phase Timings and Relative Share Breakdown" in md_content
  assert "Host Sync" in md_content
  assert "Surv (%)" in md_content, "Table 4 must show the survival bucket (BUG-30)"
  assert "8.0% (CV 3.3%)" in md_content, "the survival share and its CV must reach the row"
  assert "Runtime Floor Warning" in md_content

  assert "\\caption{Average Phase Timings, Relative Share, and Dispersion" in tex_content
  assert "Host Sync" in tex_content
  assert "Surv &" in tex_content, "the LaTeX header must carry the survival column"
  assert "8.0\\% (3\\%)" in tex_content, "the LaTeX row must carry the survival share and CV"
  assert "Warning:" in tex_content
  assert "\\end{document}" in tex_content


def test_run_comprehensive_benchmark_rejects_incompatible_checkpoint(
  fixture_config_dir: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """Verify run_comprehensive_benchmark fails fast in preflight on old-schema or mismatched checkpoints."""
  import json

  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir(parents=True)
  ckpt_path = ckpt_dir / "gr17_CPU.json"

  # Case 1: old-schema checkpoint missing modular keys ('strategies', 'algorithm_config', etc.)
  old_payload = {
    "algorithm": "CPU",
    "repetitions": 1,
    "successful_runs": 1,
    "mean_time": 0.1,
    "mean_cost": 2085.0,
    "raw_times": [0.1],
    "raw_costs": [2085.0],
    "raw_gaps": [0.0],
    "raw_seeds": [42],
  }
  ckpt_path.write_text(json.dumps(old_payload), encoding="utf-8")

  with pytest.raises(ValueError, match="missing required keys"):
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      tsp_files=[fixture_tsp_dir / "gr17.tsp"],
      algorithms=["CPU"],
      repetitions=1,
    )

  # Stale checkpoint must not be overwritten
  assert json.loads(ckpt_path.read_text(encoding="utf-8")) == old_payload

  # Case 2: mismatched strategy class in modular checkpoint
  mismatched_strategy_payload = {
    **old_payload,
    "problem_name": "gr17",
    "strategies": {
      "initialization": "InitializationRandom",
      "selection": "SelectionTournament",
      "crossover": "CrossoverOrder",
      "mutation": "MutationSwap",
      "local_search": "LocalSearchGPUTwoOpt",
      "evaluation": "EvaluationCPUCost",
    },
    "algorithm_config": {
      "population_size": 32,
      "mutation_rate": 0.05,
      "seed": 42,
    },
  }
  ckpt_path.write_text(json.dumps(mismatched_strategy_payload), encoding="utf-8")

  with pytest.raises(ValueError, match=r"strategies\.local_search"):
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      tsp_files=[fixture_tsp_dir / "gr17.tsp"],
      algorithms=["CPU"],
      repetitions=1,
    )


def test_run_comprehensive_benchmark_rejects_dimension_mismatch(
  fixture_config_dir: Path,
  fixture_duckdb_path: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """Verify run_comprehensive_benchmark raises ValueError when ProblemConfig.size != Problem.dimension."""
  with pytest.raises(ValueError, match=r"configured size \(999\) does not match loaded dimension \(17\)"):
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      db_path=fixture_duckdb_path,
      tsp_dir=fixture_tsp_dir,
      problem_configs=[ProblemConfig(name="gr17", optimal=2085.0, size=999)],
      algorithms=["CPU"],
      repetitions=1,
    )


def test_run_comprehensive_benchmark_preflight_rejects_invalid_config_and_missing_gpu(
  fixture_config_dir: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify preflight rejects invalid overrides and missing GPU backend when GPU algorithm is selected."""
  from benchmarking.runner import orchestration

  gr17_path = fixture_tsp_dir / "gr17.tsp"

  with pytest.raises(ValueError, match="Invalid repetitions: 0"):
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      tsp_files=[gr17_path],
      algorithms=["CPU"],
      repetitions=0,
    )

  monkeypatch.setattr(orchestration, "gpu_available", lambda: False)
  monkeypatch.setattr(orchestration, "cupy_backend", lambda: None)

  with pytest.raises(ValueError, match="no usable CUDA device"):
    run_comprehensive_benchmark(
      config_dir=fixture_config_dir,
      output_dir=tmp_path,
      tsp_files=[gr17_path],
      algorithms=["HybridNaive"],
      repetitions=1,
    )
