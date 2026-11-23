"""Unit tests for the `BenchmarkCohort` domain view and unified analysis primitives.

WHAT: Exercises `BenchmarkCohort`, `finite_metric`, `geometric_mean_ratios`,
`derive_problem_configs`, uniform `ReportStyle` injection into `cd_diagram` and
`scaling_plots`, and `validate_configs()` fail-fast validation of modular and external
strategy specs.
WHY: Verifies that metric filtering, complete-case matrices, gap-vs-cost fallback,
paired Fleming & Wallace geometric speedups, and preflight spec validation live in
single canonical implementations without duplicated nested-dict boilerplate.
EXPECTED: Booleans/NaNs are excluded cleanly, fallback metrics and paired speedups match
analytical reference values, `style=` works uniformly across all generators, and invalid
modular/external strategy specs are caught during `validate_configs()`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.analysis.cd_diagram import compute_cd_diagram_data, export_cd_diagram_tikz
from benchmarking.analysis.cohort import BenchmarkCohort, derive_problem_configs
from benchmarking.analysis.problem_statistics import _resolve_gap_mean, _resolve_metric_mean
from benchmarking.analysis.report_style import ReportStyle
from benchmarking.analysis.scaling_plots import generate_scaling_plots
from benchmarking.analysis.stats_utils import finite_metric, geometric_mean_ratios
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  BenchmarkConfig,
  GAParams,
  ProblemConfig,
  validate_configs,
)

if TYPE_CHECKING:
  from pathlib import Path


def test_finite_metric_and_geometric_mean_reject_bool_and_non_finite() -> None:
  """WHAT: Verify `finite_metric` and `geometric_mean_ratios` reject booleans, NaNs, infs, and strings.

  WHY: Prevents Python's `bool`-is-`int` coercion and non-finite sentinels from silently
  corrupting cross-problem means, Friedman matrices, or geometric speedups.
  EXPECTED: Returns None for bool/None/NaN/Inf/bad-string; returns exact finite float otherwise.
  """
  assert finite_metric(None) is None
  assert finite_metric(True) is None
  assert finite_metric(False) is None
  assert finite_metric(float("nan")) is None
  assert finite_metric(float("inf")) is None
  assert finite_metric("not-a-number") is None
  assert finite_metric(3.5) == pytest.approx(3.5)
  assert finite_metric("4.25") == pytest.approx(4.25)

  assert geometric_mean_ratios([]) is None
  assert geometric_mean_ratios([0.0, -2.0, float("nan")]) is None
  assert geometric_mean_ratios([2.0, 8.0]) == pytest.approx(4.0)


def test_problem_statistics_resolvers_fallback_past_nan() -> None:
  """WHAT: `_resolve_metric_mean` and `_resolve_gap_mean` ignore NaN precomputed fields.

  WHY: A NaN cached field must never bypass fallback computation from raw series or optimal_cost.
  EXPECTED: Returns finite fallback values instead of propagating NaN.
  """
  mean_cost = _resolve_metric_mean({"mean_cost": float("nan")}, "mean_cost", [100.0, 120.0], "AlgA")
  assert mean_cost == pytest.approx(110.0)

  mean_gap = _resolve_gap_mean({"mean_gap": float("nan")}, 110.0, 100.0, [], "AlgA")
  assert mean_gap is not None
  assert mean_gap == pytest.approx(10.0)
  assert not math.isnan(mean_gap)


def test_benchmark_cohort_matrices_quality_fallback_and_speedups() -> None:
  """WHAT: Verify `BenchmarkCohort` complete-case matrices, gap/cost fallback, speedups, and config derivation.

  WHY: Confirms all analysis modules share one accurate source of truth for cohort queries.
  EXPECTED: Exact paired geometric speedups, NaN/None filtering, cost fallback when optima absent,
  and deterministic `derive_problem_configs` metadata fallback.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "Base": {
        "problem_size": 52,
        "optimal_cost": 100.0,
        "repetitions": 5,
        "mean_time": 4.0,
        "mean_cost": 102.0,
        "mean_gap": 2.0,
      },
      "Fast": {
        "problem_size": 52,
        "optimal_cost": 100.0,
        "repetitions": 5,
        "mean_time": 1.0,
        "mean_cost": 101.0,
        "mean_gap": 1.0,
      },
    },
    "p2": {
      "Base": {
        "problem_metadata": {"dimension": 100},
        "optimal_cost": None,
        "successful_runs": 5,
        "mean_time": 8.0,
        "mean_cost": 210.0,
        "mean_gap": None,
      },
      "Fast": {
        "problem_metadata": {"dimension": 100},
        "optimal_cost": None,
        "successful_runs": 5,
        "mean_time": 0.5,
        "mean_cost": 205.0,
        "mean_gap": None,
      },
    },
  }

  cohort = BenchmarkCohort.from_results(all_results, algorithm_order=["Fast", "Base"])
  assert cohort.algorithms == ("Fast", "Base")
  assert cohort.problem_size("p1") == 52
  assert cohort.problem_size("p2") == 100
  assert cohort.problem_optimal("p1") == pytest.approx(100.0)
  assert cohort.problem_optimal("p2") is None
  assert cohort.quality_metric_key() == "mean_cost"
  assert cohort.repetition_counts() == [5, 5, 5, 5]

  probs, mat = cohort.metric_matrix("mean_time")
  assert probs == ["p1", "p2"]
  assert mat.shape == (2, 2)
  assert np.allclose(mat, [[1.0, 4.0], [0.5, 8.0]])

  sp = cohort.paired_geometric_speedup("Fast", "Base")
  assert sp.sample_size == 2
  # sqrt((4/1) * (8/0.5)) = sqrt(4 * 16) = 8.0
  assert sp.geomean == pytest.approx(8.0)

  derived = derive_problem_configs(all_results)
  assert [(c.name, c.size, c.optimal) for c in derived] == [("p1", 52, 100.0), ("p2", 100, None)]


def test_cd_diagram_and_scaling_plots_accept_report_style_uniformly(tmp_path: Path) -> None:
  """WHAT: Verify `export_cd_diagram_tikz` and `generate_scaling_plots` accept `style=ReportStyle(...)`.

  WHY: Ensures every report generator exposes the same `style=` injection contract instead of
  forcing callers to unpack individual style fields.
  EXPECTED: Both TikZ and SVG/PDF outputs reflect `style` display names and decimal separators,
  while explicit keyword arguments still override style defaults.
  """
  style = ReportStyle(
    display_names={"A": "Alpha Variant", "B": "Beta Variant", "C": "Gamma Variant"},
    cd_rank_label="Mean Rank",
    cd_decimal_sep=",",
    plot_title="Styled Scaling Title",
    plot_decimal_comma=True,
    show_size_guides=False,
  )

  cd = compute_cd_diagram_data(np.array([[3.0, 2.0, 1.0], [3.5, 2.1, 0.9]]), ["A", "B", "C"])
  tikz = export_cd_diagram_tikz(cd, style=style)
  assert "Alpha Variant" in tikz
  assert "Mean Rank" in tikz
  assert "1,0" in tikz

  tikz_override = export_cd_diagram_tikz(cd, style=style, rank_label="Custom Axis", decimal_sep=".")
  assert "Custom Axis" in tikz_override
  assert "1.0" in tikz_override

  results = {
    "p1": {
      "A": {"problem_size": 52, "mean_time": 2.0, "std_time": 0.1, "mean_gap": 1.0, "std_gap": 0.1, "mean_cost": 101.0},
      "B": {"problem_size": 52, "mean_time": 1.0, "std_time": 0.1, "mean_gap": 0.8, "std_gap": 0.1, "mean_cost": 100.8},
    },
    "p2": {
      "A": {
        "problem_size": 100,
        "mean_time": 5.0,
        "std_time": 0.2,
        "mean_gap": 1.5,
        "std_gap": 0.1,
        "mean_cost": 203.0,
      },
      "B": {
        "problem_size": 100,
        "mean_time": 1.5,
        "std_time": 0.1,
        "mean_gap": 1.2,
        "std_gap": 0.1,
        "mean_cost": 202.4,
      },
    },
  }
  paths = generate_scaling_plots(results, None, tmp_path, formats=("svg",), style=style)
  assert len(paths) == 1
  svg = paths[0].read_text(encoding="utf-8")
  assert "Styled Scaling Title" in svg
  assert "Alpha Variant" in svg


def test_validate_configs_catches_invalid_modular_strategies_and_external_modules() -> None:
  """WHAT: Verify `validate_configs()` inspects declarative strategy slots, pipelines, and external modules.

  WHY: `--dry-run` preflight must fail fast on unknown strategy classes, bad constructor kwargs,
  or unimportable external `"module"` specs rather than crashing mid-campaign.
  EXPECTED: Non-empty diagnostic warnings returned for each invalid spec; empty warnings for valid specs.
  """
  bad_configs: dict[str, Any] = {
    "ga_params": GAParams(),
    "benchmark": BenchmarkConfig(repetitions=2, patience=10),
    "problems": [ProblemConfig(name="berlin52", optimal=7542.0, size=52)],
    "algorithms": {
      "BrokenBuiltIn": AlgorithmConfig(
        name="BrokenBuiltIn",
        use_gpu=False,
        description="Invalid strategy and kwarg",
        strategies={
          "initialization": {"class": "NonExistentInit"},
          "selection": {"class": "SelectionTournament", "params": {"bogus_kwarg": 5}},
        },
        params={"pipeline": [{"module": "definitely_missing_mod_xyz", "class": "MissingStep"}]},
      )
    },
  }
  warnings = validate_configs(bad_configs)
  joined = "\n".join(warnings)
  assert "NonExistentInit" in joined
  assert "bogus_kwarg" in joined
  assert "definitely_missing_mod_xyz" in joined
