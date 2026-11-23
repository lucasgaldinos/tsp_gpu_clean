"""Unit tests for report generator speedup aggregation and phase timings display.

WHAT: Covers:
1. BUG-34: Table 2 average speedup aggregated by Fleming & Wallace (1986) paired
   geometric mean across problems instead of ratio of arithmetic means.
2. BUG-35: Table 4 phase timings rendered even when some algorithms (e.g. CPU)
   do not have phase telemetry, showing '-' for missing algorithms.
3. BUG-38: Standard deviation in Table 2 computed with Bessel's correction (ddof=1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.analysis.stats_utils import resolve_baseline_algorithm
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.reporting.latex_macros import generate_latex_macros
from benchmarking.reporting.style import (
  DEFAULT_SIZE_THRESHOLDS,
  ReportStyle,
  categorize_by_size,
  load_report_style,
  parse_report_artifacts,
  size_category_labels,
)
from benchmarking.reporting.table_reports import (
  _PHASE_NAMES,
  _compute_table2_stats,
  _compute_table3_size_category_winners,
  _generate_table3_latex,
  _generate_table3_markdown,
  _has_phase_timings,
  generate_latex_tables,
  generate_markdown_tables,
)
from benchmarking.runner.config_loader import ProblemConfig

SMALL_SIZE_LABEL, MEDIUM_SIZE_LABEL, LARGE_SIZE_LABEL = size_category_labels(DEFAULT_SIZE_THRESHOLDS)

if TYPE_CHECKING:
  from pathlib import Path


def test_table2_speedup_uses_paired_geometric_mean(tmp_path: Path) -> None:
  """WHAT: Two problems with disparate runtimes (10s and 100s baseline).

  WHY: BUG-34 — Arithmetic mean ratio (110 / 7 ≈ 15.71x) distorts benchmarks;
  Fleming & Wallace (1986) paired geometric mean sqrt(5 * 20) = 10.00x is correct,
  annotated with sample size (n=2) per DP12-9.
  EXPECTED: '10.00x (n=2)' in Markdown Table 2 and '10.00$\\times$ ($n=2$)' in LaTeX Table 2.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "small_prob": {
      "CPU": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 1.0},
      "GPU": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.5},
    },
    "large_prob": {
      "CPU": {"mean_time": 100.0, "mean_cost": 500.0, "std_cost": 0.0, "mean_gap": 2.0},
      "GPU": {"mean_time": 5.0, "mean_cost": 500.0, "std_cost": 0.0, "mean_gap": 0.8},
    },
  }

  problem_configs = [
    ProblemConfig(name="small_prob", optimal=100.0, size=50),
    ProblemConfig(name="large_prob", optimal=500.0, size=200),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="geom_speedup_test",
    baseline_algorithm="CPU",
  )

  md_content = (tmp_path / "geom_speedup_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "geom_speedup_test.tex").read_text(encoding="utf-8")

  # Geometric mean is sqrt((10/2) * (100/5)) = sqrt(5 * 20) = 10.00x across 2 paired instances
  assert "10.00x (n=2)" in md_content
  assert "10.00$\\times$ ($n=2$)" in tex_content
  # Naive arithmetic ratio would have been 15.71x
  assert "15.71x" not in md_content


def test_table4_emitted_for_mixed_runs_when_one_algorithm_has_phase_timings(
  tmp_path: Path,
) -> None:
  """WHAT: Mixed run where GPU has phase timings but CPU does not.

  WHY: BUG-35 — `_has_phase_timings` previously required all algorithms to have
  phase telemetry, suppressing Table 4 entirely when CPU was included.
  EXPECTED: `_has_phase_timings` returns True; Table 4 is emitted; CPU row shows
  '-' for phases and Sum/Wall, while GPU row shows formatted percentages.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "berlin52": {
      "CPU": {
        "mean_time": 0.500,
        "mean_cost": 7600.0,
        "std_cost": 0.0,
        "mean_gap": 0.8,
        # CPU has NO mean_phase_timings
      },
      "GPU": {
        "mean_time": 0.050,
        "mean_cost": 7550.0,
        "std_cost": 0.0,
        "mean_gap": 0.1,
        "mean_phase_timings": {
          "initialization": {"total_ms": 1.0, "calls": 1},
          "evaluation": {"total_ms": 20.0, "calls": 10},
        },
        "phase_share_pct": {
          "initialization": 2.0,
          "evaluation": 40.0,
        },
        "phase_share_cv_pct": {
          "initialization": 0.5,
          "evaluation": 1.0,
        },
      },
    }
  }

  assert _has_phase_timings(all_results) is True

  problem_configs = [ProblemConfig(name="berlin52", optimal=7542.0, size=52)]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="mixed_phases_test",
    baseline_algorithm="CPU",
  )

  md_content = (tmp_path / "mixed_phases_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "mixed_phases_test.tex").read_text(encoding="utf-8")

  assert "## Table 4: Average Phase Timings and Relative Share Breakdown" in md_content
  assert "\\caption{Average Phase Timings, Relative Share, and Dispersion" in tex_content

  # CPU row has '-' in phases and sum
  assert "| berlin52 | CPU | 500.0 | - | - | - | - | - | - | - | - | - |" in md_content
  assert "berlin52 & CPU & 500.0 & - & - & - & - & - & - & - & - & - \\\\" in tex_content

  # GPU row has formatted percentage
  assert "40.0% (CV 1.0%)" in md_content


def test_table2_std_uses_bessels_correction(tmp_path: Path) -> None:
  """WHAT: Table 2 std_gap and std_time computed across 3 problems.

  WHY: BUG-38 — Sample standard deviation must use ddof=1 (Bessel's correction).
  EXPECTED: For values [1.0, 2.0, 3.0], sample std is sqrt(2/2) = 1.00,
  not population std sqrt(2/3) ≈ 0.82.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob1": {"CPU": {"mean_time": 1.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 1.0}},
    "prob2": {"CPU": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 2.0}},
    "prob3": {"CPU": {"mean_time": 3.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 3.0}},
  }

  problem_configs = [
    ProblemConfig(name="prob1", optimal=100.0, size=10),
    ProblemConfig(name="prob2", optimal=100.0, size=20),
    ProblemConfig(name="prob3", optimal=100.0, size=30),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="bessel_test",
  )

  md_content = (tmp_path / "bessel_test.md").read_text(encoding="utf-8")
  # Table 2: | CPU | 2.00 | 1.00 | 2.00 | 1.00 | 1.00x |
  assert "| CPU | 2.00 | 1.00 | 2.00 | 1.00 |" in md_content
  assert "0.82" not in md_content


def test_report_generator_arbitrary_algorithm_names_and_dynamic_baseline(
  tmp_path: Path,
) -> None:
  """WHAT: Report generation with arbitrary algorithm names (DP12-9).

  WHY: Task 12.16 — Algorithms should not be tied to hardcoded strings like "CPU"
  or "HybridNaive". Baseline should default deterministically and report 1.00x.
  EXPECTED: Table 1 and Table 2 generate cleanly in Markdown and LaTeX with valid
  geometric speedups and 1.00x for the baseline algorithm.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "instance_1": {
      "CustomGA_Alpha": {"mean_time": 4.0, "mean_cost": 200.0, "std_cost": 5.0, "mean_gap": 1.5},
      "CustomGA_Beta": {"mean_time": 2.0, "mean_cost": 200.0, "std_cost": 4.0, "mean_gap": 1.2},
      "CustomGA_Gamma": {"mean_time": 1.0, "mean_cost": 200.0, "std_cost": 3.0, "mean_gap": 1.0},
    },
    "instance_2": {
      "CustomGA_Alpha": {"mean_time": 8.0, "mean_cost": 400.0, "std_cost": 10.0, "mean_gap": 2.5},
      "CustomGA_Beta": {"mean_time": 2.0, "mean_cost": 400.0, "std_cost": 8.0, "mean_gap": 2.0},
      "CustomGA_Gamma": {"mean_time": 1.0, "mean_cost": 400.0, "std_cost": 6.0, "mean_gap": 1.8},
    },
  }

  problem_configs = [
    ProblemConfig(name="instance_1", optimal=197.0, size=50),
    ProblemConfig(name="instance_2", optimal=390.0, size=100),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="arbitrary_names_test",
  )

  md_content = (tmp_path / "arbitrary_names_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "arbitrary_names_test.tex").read_text(encoding="utf-8")

  # Table 1: Speedups relative to CustomGA_Alpha
  assert "| instance_1 | CustomGA_Alpha | 200.00 | 5.00 | 1.50 | 4.00 | 1.00x |" in md_content
  assert "| instance_1 | CustomGA_Beta | 200.00 | 4.00 | 1.20 | 2.00 | 2.00x |" in md_content
  assert "| instance_1 | CustomGA_Gamma | 200.00 | 3.00 | 1.00 | 1.00 | 4.00x |" in md_content

  # Table 2: Baseline algorithm shows 1.00x in MD and 1.00$\times$ in TeX
  assert "| CustomGA_Alpha |" in md_content
  assert "| 1.00x |" in md_content
  assert "1.00$\\times$" in tex_content

  # Fleming & Wallace geometric mean speedup for Beta vs Alpha: sqrt((4/2) * (8/2)) = sqrt(2 * 4) ≈ 2.83x (n=2)
  assert "2.83x (n=2)" in md_content
  assert "2.83$\\times$ ($n=2$)" in tex_content


def test_report_generator_table1_fallback_dagger_when_baseline_missing(
  tmp_path: Path,
) -> None:
  """WHAT: Table 1 handles missing baseline on large problem with dagger annotation (†).

  WHY: Task 12.16 (DP12-9) — When baseline algorithm is missing for a problem tier,
  dynamic fallback to first available algorithm must be annotated with dagger (†).
  EXPECTED:
  - Small problem with baseline shows normal speedup (1.00x, 2.00x).
  - Large problem missing baseline falls back to first alg and annotates with † (1.00x†, 2.00x†).
  - Note at bottom explains † fallback.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "small_prob": {
      "BaseAlg": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "FastAlg": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "large_prob": {
      # BaseAlg was skipped for large problem
      "FastAlg": {"mean_time": 10.0, "mean_cost": 500.0, "std_cost": 0.0, "mean_gap": 0.0},
      "SuperFastAlg": {"mean_time": 5.0, "mean_cost": 500.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }

  problem_configs = [
    ProblemConfig(name="small_prob", optimal=100.0, size=50),
    ProblemConfig(name="large_prob", optimal=500.0, size=500),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="fallback_dagger_test",
    baseline_algorithm="BaseAlg",
  )

  md_content = (tmp_path / "fallback_dagger_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "fallback_dagger_test.tex").read_text(encoding="utf-8")

  # Small problem: standard speedups
  assert "| small_prob | BaseAlg | 100.00 | 0.00 | 0.00 | 10.00 | 1.00x |" in md_content
  assert "| small_prob | FastAlg | 100.00 | 0.00 | 0.00 | 2.00 | 5.00x |" in md_content

  # Large problem: fallback with †
  assert "1.00x†" in md_content
  assert "2.00x†" in md_content
  assert "1.00$\\times^{\\dagger}$" in tex_content
  assert "2.00$\\times^{\\dagger}$" in tex_content

  # Footnote explanation present
  assert "† Indicates speedup calculated relative to local fallback baseline" in md_content
  assert "$\\dagger$ Indicates speedup calculated relative to local fallback baseline" in tex_content


def test_get_baseline_algorithm_dynamic() -> None:
  """WHAT: resolve_baseline_algorithm resolves baseline dynamically from first entry.

  WHY: Task 12.16 (DP12-9 / BUG-55) — No hardcoded algorithm names ('CPU', 'HybridNaive').
  EXPECTED: First element returned regardless of presence of 'CPU' or 'HybridNaive'.
  """
  assert resolve_baseline_algorithm(["Alpha", "CPU", "HybridNaive"]) == "Alpha"
  assert resolve_baseline_algorithm(["Custom_1", "Custom_2"]) == "Custom_1"
  assert resolve_baseline_algorithm(["HybridNaive", "CPU"]) == "HybridNaive"
  assert resolve_baseline_algorithm(["CPU", "Custom_B"]) == "CPU"
  assert resolve_baseline_algorithm([]) == "Unknown"


def test_report_generator_explicit_baseline_override(tmp_path: Path) -> None:
  """WHAT: Passing explicit baseline_algorithm overrides automatic first-element selection.

  WHY: Task 12.16 (DP12-9) — User or runner can specify any baseline algorithm.
  EXPECTED: Specified baseline displays 1.00x and all speedups are computed against it.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob1": {
      "AlgA": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "AlgB": {"mean_time": 5.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob2": {
      "AlgA": {"mean_time": 20.0, "mean_cost": 200.0, "std_cost": 0.0, "mean_gap": 0.0},
      "AlgB": {"mean_time": 10.0, "mean_cost": 200.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="prob1", optimal=100.0, size=50),
    ProblemConfig(name="prob2", optimal=200.0, size=100),
  ]

  # Override baseline to AlgB (second algorithm)
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="override_test",
    baseline_algorithm="AlgB",
  )

  md_content = (tmp_path / "override_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "override_test.tex").read_text(encoding="utf-8")

  # AlgB is baseline -> 1.00x
  assert "| AlgB |" in md_content
  assert "| 1.00x |" in md_content
  assert "1.00$\\times$" in tex_content

  # AlgA takes twice as long -> speedup 5/10 = 0.50x (n=2)
  assert "0.50x (n=2)" in md_content
  assert "0.50$\\times$ ($n=2$)" in tex_content


def test_table2_speedup_annotates_sample_size_with_partial_overlap(tmp_path: Path) -> None:
  """WHAT: Table 2 annotates correct paired sample count |P_cap| when instances only partially overlap.

  WHY: Task 12.16 (DP12-9) — When an algorithm was only evaluated on a subset of the baseline's problems,
  n must reflect |P_cap|, not total problem count.
  EXPECTED: (n=1) when only 1 problem has both baseline and candidate data.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "prob1": {
      "Base": {"mean_time": 10.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
      "Target": {"mean_time": 2.0, "mean_cost": 100.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob2": {
      # Target did not run on prob2
      "Base": {"mean_time": 20.0, "mean_cost": 200.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
    "prob3": {
      # Base did not run on prob3
      "Target": {"mean_time": 4.0, "mean_cost": 300.0, "std_cost": 0.0, "mean_gap": 0.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="prob1", optimal=100.0, size=50),
    ProblemConfig(name="prob2", optimal=200.0, size=100),
    ProblemConfig(name="prob3", optimal=300.0, size=150),
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path,
    benchmark_name="partial_overlap_test",
    baseline_algorithm="Base",
  )

  md_content = (tmp_path / "partial_overlap_test.md").read_text(encoding="utf-8")
  tex_content = (tmp_path / "partial_overlap_test.tex").read_text(encoding="utf-8")

  # Overlap |P_cap| is only prob1 (1 instance), speedup is 10/2 = 5.00x (n=1)
  assert "5.00x (n=1)" in md_content
  assert "5.00$\\times$ ($n=1$)" in tex_content


def test_table2_and_table3_helpers_dry_refactoring() -> None:
  """WHAT: Test _compute_table2_stats, _compute_table3_size_category_winners, and _PHASE_NAMES.

  WHY: Task 12.23 (DP12-16) — Ensure DRY extraction computes identical stats and category winners.
  EXPECTED: Valid statistics dictionaries and category-grouped winners matching expectation.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "small1": {
      "AlgoA": {"mean_time": 1.0, "mean_gap": 2.0},
      "AlgoB": {"mean_time": 0.5, "mean_gap": 1.0},
    },
    "large1": {
      "AlgoA": {"mean_time": 10.0, "mean_gap": 4.0},
      "AlgoB": {"mean_time": 2.0, "mean_gap": 2.0},
    },
  }
  problem_configs = [
    ProblemConfig(name="small1", optimal=100.0, size=50),
    ProblemConfig(name="large1", optimal=500.0, size=500),
  ]

  # 1. _compute_table2_stats
  t2_stats = _compute_table2_stats(all_results, ["small1", "large1"], ["AlgoA", "AlgoB"], baseline_algorithm="AlgoA")
  assert len(t2_stats) == 2
  assert t2_stats[0]["algorithm"] == "AlgoA"
  assert t2_stats[0]["is_baseline"] is True
  assert t2_stats[0]["avg_speedup"] == 1.0
  assert t2_stats[1]["algorithm"] == "AlgoB"
  assert t2_stats[1]["is_baseline"] is False
  assert t2_stats[1]["sample_size"] == 2
  # Speedups are 1.0/0.5 = 2.0 and 10.0/2.0 = 5.0; geomean = sqrt(10.0) ≈ 3.162277
  assert t2_stats[1]["avg_speedup"] == pytest.approx(np.sqrt(10.0))

  # 2. _compute_table3_size_category_winners
  t3_winners = _compute_table3_size_category_winners(all_results, problem_configs, ["AlgoA", "AlgoB"])
  assert len(t3_winners) == 2
  # Small (n≤100): AlgoB has gap 1.0 vs AlgoA 2.0 -> AlgoB wins
  assert t3_winners[0]["category"] == SMALL_SIZE_LABEL
  assert t3_winners[0]["best_alg"] == "AlgoB"
  assert t3_winners[0]["best_gap"] == 1.0
  # Large (n>400): AlgoB has gap 2.0 vs AlgoA 4.0 -> AlgoB wins
  assert t3_winners[1]["category"] == LARGE_SIZE_LABEL
  assert t3_winners[1]["best_alg"] == "AlgoB"
  assert t3_winners[1]["best_gap"] == 2.0

  # 3. _PHASE_NAMES constant
  assert isinstance(_PHASE_NAMES, tuple)
  assert len(_PHASE_NAMES) == 8
  assert "initialization" in _PHASE_NAMES
  assert "host_sync" in _PHASE_NAMES


def test_table3_partial_coverage_cannot_beat_full_coverage() -> None:
  """WHAT: An algorithm with partial coverage cannot win a size category.

  WHY: BUG-50 / DP12-26 — If CPU only solved 1 of 3 large problems with a tiny gap,
  it previously beat GPU which solved all 3 problems. A category winner must evaluate
  all problems in that size category. If no algorithm evaluates all problems, best_alg
  must be None and Table 3 renders '—'.
  """
  problem_configs = [
    ProblemConfig(name="large1", optimal=500.0, size=500),
    ProblemConfig(name="large2", optimal=1000.0, size=1000),
  ]
  # CPU evaluated only large1 with gap 0.05
  # FullGPU evaluated large1 (gap 0.5) and large2 (gap 1.0), mean gap 0.75
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "large1": {
      "CPU": {"mean_time": 100.0, "mean_gap": 0.05},
      "FullGPU": {"mean_time": 2.0, "mean_gap": 0.5},
    },
    "large2": {
      "FullGPU": {"mean_time": 5.0, "mean_gap": 1.0},
    },
  }

  winners = _compute_table3_size_category_winners(all_results, problem_configs, ["CPU", "FullGPU"])
  assert len(winners) == 1
  assert winners[0]["category"] == LARGE_SIZE_LABEL
  # FullGPU has full coverage (2/2) and wins over partial CPU (1/2) despite CPU's lower gap on large1:
  assert winners[0]["best_alg"] == "FullGPU"
  assert winners[0]["best_gap"] == pytest.approx(0.75)
  assert winners[0]["best_time"] == pytest.approx(3.5)

  # Case 2: Neither algorithm has full coverage
  all_results_partial: dict[str, dict[str, dict[str, Any]]] = {
    "large1": {
      "CPU": {"mean_time": 100.0, "mean_gap": 0.05},
    },
    "large2": {
      "FullGPU": {"mean_time": 5.0, "mean_gap": 1.0},
    },
  }
  winners_none = _compute_table3_size_category_winners(all_results_partial, problem_configs, ["CPU", "FullGPU"])
  assert len(winners_none) == 1
  assert winners_none[0]["best_alg"] is None
  assert winners_none[0]["best_gap"] is None
  assert winners_none[0]["best_time"] is None

  # Rendering checks: both Markdown and LaTeX must render '—'
  default_style = ReportStyle()
  md_lines = _generate_table3_markdown(all_results_partial, problem_configs, ["CPU", "FullGPU"], default_style)
  md_table = "\n".join(md_lines)
  assert "—" in md_table
  assert "None" not in md_table
  assert "inf" not in md_table

  tex_lines = _generate_table3_latex(all_results_partial, problem_configs, ["CPU", "FullGPU"], default_style)
  tex_table = "\n".join(tex_lines)
  assert "—" in tex_table
  assert "None" not in tex_table
  assert "inf" not in tex_table
  assert "\\label{tab:size_categories}" in tex_table


def test_size_category_boundaries_default_and_configurable() -> None:
  """WHAT: Default (100, 400) and custom size_thresholds map instances accurately into strata."""
  assert DEFAULT_SIZE_THRESHOLDS == (100, 400)
  assert SMALL_SIZE_LABEL == "Small (n≤100)"
  assert MEDIUM_SIZE_LABEL == "Medium (100<n≤400)"
  assert LARGE_SIZE_LABEL == "Large (n>400)"

  boundary_sizes = [99, 100, 101, 399, 400, 401]
  problem_configs = [ProblemConfig(name=f"p_{n}", optimal=100.0, size=n) for n in boundary_sizes]
  problem_names = [cfg.name for cfg in problem_configs]

  categories = categorize_by_size(problem_names, problem_configs)
  assert categories[SMALL_SIZE_LABEL] == ["p_99", "p_100"]
  assert categories[MEDIUM_SIZE_LABEL] == ["p_101", "p_399", "p_400"]
  assert categories[LARGE_SIZE_LABEL] == ["p_401"]

  custom_categories = categorize_by_size(problem_names, problem_configs, size_thresholds=(50, 150))
  assert custom_categories["Small (n≤50)"] == []
  assert custom_categories["Medium (50<n≤150)"] == ["p_99", "p_100", "p_101"]
  assert custom_categories["Large (n>150)"] == ["p_399", "p_400", "p_401"]

  all_results: dict[str, dict[str, dict[str, Any]]] = {
    name: {
      "AlgoA": {"mean_time": 2.0, "mean_gap": 1.0},
      "AlgoB": {"mean_time": 1.0, "mean_gap": 0.5},
    }
    for name in problem_names
  }
  winners = _compute_table3_size_category_winners(all_results, problem_configs, ["AlgoA", "AlgoB"])
  assert [(w["category"], w["problems"], w["best_alg"]) for w in winners] == [
    (SMALL_SIZE_LABEL, ["p_99", "p_100"], "AlgoB"),
    (MEDIUM_SIZE_LABEL, ["p_101", "p_399", "p_400"], "AlgoB"),
    (LARGE_SIZE_LABEL, ["p_401"], "AlgoB"),
  ]


def test_generate_result_tables_emits_both_gap_and_time_cd_diagrams(tmp_path: Path) -> None:
  """WHAT: generate_result_tables emits separate TikZ CD diagrams for Optimality Gap and Execution Time.

  WHY: Task 14.2 (DP14-2 / BUG-92) — Visualizing both quality equivalence and runtime acceleration
  requires distinct CD diagrams for gap and time.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    p: {
      "SlowAccurate": {"mean_cost": 100.0, "std_cost": 0.5, "mean_gap": 0.1, "mean_time": 10.0},
      "MidVariant": {"mean_cost": 101.0, "std_cost": 0.5, "mean_gap": 0.5, "mean_time": 4.0},
      "FastVariant": {"mean_cost": 102.0, "std_cost": 0.5, "mean_gap": 1.0, "mean_time": 0.5},
    }
    for p in ("p1", "p2", "p3", "p4")
  }
  problem_configs = [
    ProblemConfig(name=p, optimal=100.0, size=50 + i * 20) for i, p in enumerate(("p1", "p2", "p3", "p4"))
  ]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "tables",
    benchmark_name="dual_cd",
    baseline_algorithm="SlowAccurate",
  )

  gap_tikz = tmp_path / "tables" / "dual_cd_cd_diagram_gap.tikz"
  time_tikz = tmp_path / "tables" / "dual_cd_cd_diagram_time.tikz"
  assert gap_tikz.is_file()
  assert time_tikz.is_file()

  gap_content = gap_tikz.read_text(encoding="utf-8")
  time_content = time_tikz.read_text(encoding="utf-8")
  assert "% Metric: Optimality Gap" in gap_content
  assert "% Metric: Execution Time" in time_content
  # In Gap: SlowAccurate is rank 1.00; in Time: FastVariant is rank 1.00
  assert "SlowAccurate\\\\(1.00)" in gap_content
  assert "FastVariant\\\\(1.00)" in time_content


def test_generate_result_tables_emits_gap_cd_with_incomplete_problem_and_no_mean_cost(tmp_path: Path) -> None:
  """WHAT: Partial problem coverage with mean_gap (no mean_cost) still emits *_cd_diagram_gap.tikz."""
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "A": {"mean_cost": 100.0, "std_cost": 0.5, "mean_gap": 0.1, "mean_time": 10.0},
      "B": {"mean_cost": 101.0, "std_cost": 0.5, "mean_gap": 0.5, "mean_time": 4.0},
      "C": {"mean_cost": 102.0, "std_cost": 0.5, "mean_gap": 1.0, "mean_time": 0.5},
    },
    "p2": {
      "A": {"mean_cost": 100.0, "std_cost": 0.5, "mean_gap": 0.1, "mean_time": 10.0},
      "B": {"mean_cost": 101.0, "std_cost": 0.5, "mean_gap": 0.5, "mean_time": 4.0},
      "C": {"mean_cost": 102.0, "std_cost": 0.5, "mean_gap": 1.0, "mean_time": 0.5},
    },
    # Incomplete problem (missing A), and B/C omit mean_cost on p1/p2? Actually test partial coverage explicitly:
    "p3": {
      "B": {"mean_cost": 101.0, "std_cost": 0.5, "mean_gap": 0.5, "mean_time": 4.0},
      "C": {"mean_cost": 102.0, "std_cost": 0.5, "mean_gap": 1.0, "mean_time": 0.5},
    },
  }
  # Ensure complete problems p1, p2 do NOT rely on fallback to mean_cost if mean_gap is present
  # (remove mean_cost from CD candidate by checking metric label in output)
  problem_configs = [ProblemConfig(name=p, optimal=100.0, size=50 + i * 20) for i, p in enumerate(("p1", "p2", "p3"))]

  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "tables_partial",
    benchmark_name="partial_cd",
    baseline_algorithm="A",
  )
  gap_tikz = tmp_path / "tables_partial" / "partial_cd_cd_diagram_gap.tikz"
  time_tikz = tmp_path / "tables_partial" / "partial_cd_cd_diagram_time.tikz"
  assert gap_tikz.is_file()
  assert time_tikz.is_file()
  assert "N = 2 instances" in gap_tikz.read_text(encoding="utf-8")


def test_generate_latex_macros_neutral_default_and_custom_separator(tmp_path: Path) -> None:
  """WHAT: generate_result_tables and generate_latex_macros default to '.' and honor ReportStyle separators.

  WHY: Neutral English '.' decimal default keeps src/ thesis-free while ReportStyle(decimal_separator="{,}")
  supports localized LaTeX documents without hardcoding presets inside src/.
  """
  # 1. Multi-problem, 3-algorithm cohort with consistent speedup ordering (Gamma > Beta > Alpha)
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    f"p{i}": {
      "Alpha_1": {
        "mean_cost": 100.0,
        "std_cost": 1.0,
        "mean_gap": 1.0 + 0.1 * i,
        "mean_time": 10.0 * i,
        "repetitions": 30,
      },
      "Beta-2": {
        "mean_cost": 100.0,
        "std_cost": 1.0,
        "mean_gap": 1.0 + 0.1 * i,
        "mean_time": 5.0 * i,
        "repetitions": 30,
      },
      "Gamma": {
        "mean_cost": 101.0,
        "std_cost": 1.0,
        "mean_gap": 1.5 + 0.1 * i,
        "mean_time": 1.0 * i,
        "repetitions": 30,
      },
    }
    for i in range(1, 5)
  }
  problem_configs = [ProblemConfig(name=f"p{i}", optimal=100.0, size=50 * i) for i in range(1, 5)]

  tables_dir = tmp_path / "tables"
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tables_dir,
    benchmark_name="macros_test",
    baseline_algorithm="Alpha_1",
  )

  macros_path = tables_dir / "benchmark_macros.tex"
  assert macros_path.is_file()
  content = macros_path.read_text(encoding="utf-8")

  # Strictly quantitative: every non-comment, non-empty line must be a \newcommand
  for raw_line in content.splitlines():
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("%"):
      continue
    assert stripped.startswith("\\newcommand{\\"), f"Unexpected non-macro line: {stripped}"

  # Cohort & sanitized tokens (Alpha_1 -> AlphaOne, Beta-2 -> BetaTwo)
  assert "\\newcommand{\\benchmarkNumProblems}{4}" in content
  assert "\\newcommand{\\benchmarkNumAlgorithms}{3}" in content
  assert "\\newcommand{\\benchmarkBaselineAlgorithm}{Alpha\\_1}" in content
  assert "\\newcommand{\\benchmarkTotalRuns}{360}" in content
  assert "\\newcommand{\\benchmarkRepetitions}{30}" in content

  # Aggregated means & geometric speedups in neutral dot format ('.')
  assert "\\newcommand{\\meanGapAlphaOne}{1.25}" in content
  assert "\\newcommand{\\speedupGeomBetaTwo}{2.00}" in content
  assert "\\newcommand{\\speedupGeomGamma}{10.00}" in content
  assert "\\newcommand{\\speedupGammaVsBetaTwo}{5.00}" in content

  # Friedman / Iman-Davenport & Nemenyi CD + mean ranks
  assert "\\newcommand{\\friedmanTimeF}{\\ensuremath{\\infty}}" in content
  assert "\\newcommand{\\friedmanTimeP}{\\ensuremath{< 10^{-6}}}" in content
  assert "\\newcommand{\\meanRankTimeGamma}{1.00}" in content
  assert "\\newcommand{\\meanRankTimeAlphaOne}{3.00}" in content
  assert "\\newcommand{\\nemenyiTimeCD}{" in content
  assert "\\newcommand{\\friedmanGapF}{" in content

  # Wilcoxon + step-down Holm vs baseline
  assert "\\newcommand{\\wilcoxonGapWGamma}{0.0}" in content
  assert "\\newcommand{\\wilcoxonTimePHolmGamma}{" in content

  # 2. Custom comma decimal separator via ReportStyle & edge cases (empty, N=1, k=2)
  comma_style = ReportStyle(decimal_separator="{,}")
  comma_path = generate_latex_macros(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "comma_macros",
    baseline_algorithm="Alpha_1",
    style=comma_style,
  )
  comma_content = comma_path.read_text(encoding="utf-8")
  assert "\\newcommand{\\speedupGeomGamma}{10{,}00}" in comma_content

  single_prob = {"only": {"A": {"mean_gap": 0.5, "mean_time": 4.0}, "B": {"mean_gap": 0.5, "mean_time": 2.0}}}
  single_path = generate_latex_macros(single_prob, [], tmp_path / "single_macros", baseline_algorithm="A")
  single_text = single_path.read_text(encoding="utf-8")
  assert "\\newcommand{\\speedupBVsA}{2.00}" in single_text
  assert "\\friedman" not in single_text

  with pytest.raises(ValueError, match="empty"):
    generate_latex_macros({}, [], tmp_path / "empty_macros")

  # 3. mean_cost-only fallback (no mean_gap) + successful_runs + tied/zero-diff pairs
  cost_only: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "AlgA": {"mean_cost": 100.0, "mean_time": 10.0, "successful_runs": 10},
      "AlgB": {"mean_cost": 100.0, "mean_time": 5.0, "successful_runs": 10},
      "AlgC": {"mean_cost": 102.0, "mean_time": 2.0, "successful_runs": 10},
    },
    "p2": {
      "AlgA": {"mean_cost": 200.0, "mean_time": 20.0, "successful_runs": 10},
      "AlgB": {"mean_cost": 200.0, "mean_time": 10.0, "successful_runs": 10},
      "AlgC": {"mean_cost": 204.0, "mean_time": 4.0, "successful_runs": 10},
    },
    "p3": {
      "AlgA": {"mean_cost": 300.0, "mean_time": 30.0, "successful_runs": 10},
      "AlgB": {"mean_cost": 300.0, "mean_time": 15.0, "successful_runs": 10},
    },
  }
  cost_path = generate_latex_macros(cost_only, [], tmp_path / "cost_only_macros", baseline_algorithm="AlgA")
  cost_txt = cost_path.read_text(encoding="utf-8")
  assert "\\newcommand{\\benchmarkTotalRuns}{80}" in cost_txt
  assert "\\newcommand{\\meanCostAlgA}{200.00}" in cost_txt
  assert "\\newcommand{\\speedupGeomAlgB}{2.00}" in cost_txt
  assert "\\newcommand{\\friedmanGapN}{2}" in cost_txt
  assert "\\newcommand{\\wilcoxonGapPAlgB}{1.00}" in cost_txt


def test_composable_report_artifacts_and_external_style(tmp_path: Path) -> None:
  """WHAT: Independent report generators, selective facade artifacts, and ReportStyle injection."""
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "VarA": {"mean_cost": 100.0, "std_cost": 1.0, "mean_gap": 1.0, "mean_time": 2.0},
      "VarB": {"mean_cost": 99.0, "std_cost": 0.5, "mean_gap": 0.0, "mean_time": 1.0},
    },
  }
  problem_configs = [ProblemConfig(name="p1", optimal=99.0, size=50)]
  custom_style = ReportStyle(
    display_names={"VarA": "Baseline Alpha", "VarB": "Fast Beta"},
    latex_label_prefix="custom_",
    size_thresholds=(40, 120),
  )

  # 1. Standalone markdown-only and latex-only generators
  md_only = generate_markdown_tables(
    all_results,
    problem_configs,
    tmp_path / "standalone_md",
    benchmark_name="solo",
    style=custom_style,
  )
  assert md_only.is_file()
  assert not (tmp_path / "standalone_md" / "solo.tex").exists()
  md_text = md_only.read_text(encoding="utf-8")
  assert "Baseline Alpha" in md_text
  assert "Fast Beta" in md_text

  tex_only = generate_latex_tables(
    all_results,
    problem_configs,
    tmp_path / "standalone_tex",
    benchmark_name="solo",
    style=custom_style,
  )
  assert tex_only.is_file()
  tex_text = tex_only.read_text(encoding="utf-8")
  assert "\\label{tab:custom_summary}" in tex_text
  assert "\\label{tab:custom_algorithm_comparison}" in tex_text
  assert "Fast Beta" in tex_text

  # 2. Selective artifacts on facade
  partial_out = generate_result_tables(
    all_results,
    problem_configs,
    tmp_path / "selective" / "tables",
    benchmark_name="sel",
    style=custom_style,
    artifacts=("markdown",),
  )
  assert set(partial_out.keys()) == {"markdown"}
  assert (tmp_path / "selective" / "tables" / "sel.md").is_file()
  assert not (tmp_path / "selective" / "tables" / "sel.tex").exists()
  assert not (tmp_path / "selective" / "tables" / "benchmark_macros.tex").exists()
  assert not (tmp_path / "selective" / "figures").exists()

  # 3. Empty artifacts emits nothing
  assert generate_result_tables(all_results, problem_configs, tmp_path / "empty_art", artifacts=()) == {}

  # 4. parse_report_artifacts & load_report_style validation
  assert parse_report_artifacts(None, default=()) == ()
  assert parse_report_artifacts(["none"], default=("markdown",)) == ()
  assert parse_report_artifacts(["markdown", "plots", "markdown"], default=()) == ("markdown", "plots")
  with pytest.raises(ValueError, match="Cannot combine 'none'"):
    parse_report_artifacts(["none", "markdown"], default=())
  with pytest.raises(ValueError, match="Unsupported report artifact"):
    parse_report_artifacts(["bogus"], default=())

  bad_style = tmp_path / "bad_style.json"
  bad_style.write_text('{"unknown_key": true}', encoding="utf-8")
  with pytest.raises(ValueError, match="Unknown key"):
    load_report_style(bad_style)

  bad_bool_style = tmp_path / "bad_bool_style.json"
  bad_bool_style.write_text('{"plot_decimal_comma": "false"}', encoding="utf-8")
  with pytest.raises(ValueError, match="must be a boolean"):
    load_report_style(bad_bool_style)


def test_generate_result_tables_computes_omnibus_once_per_metric(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Emitting all 5 report artifacts computes Friedman/Nemenyi at most once per metric (2 total).

  WHY: Regression test eliminating the prior 3x-4x recomputation across aggregate_statistics,
  latex_macros, and cd_diagram, and proving precomputed `analysis=` incurs zero extra calls.
  """
  import benchmarking.analysis.cohort_analysis as cohort_analysis_mod
  from benchmarking.analysis.statistics import run_friedman_nemenyi_analysis as real_fn

  call_count = 0

  def spy_omnibus(*args: Any, **kwargs: Any) -> Any:
    nonlocal call_count
    call_count += 1
    return real_fn(*args, **kwargs)

  monkeypatch.setattr(cohort_analysis_mod, "run_friedman_nemenyi_analysis", spy_omnibus)

  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "p1": {
      "AlgA": {
        "mean_gap": 1.0,
        "std_gap": 0.1,
        "mean_cost": 101.0,
        "std_cost": 1.0,
        "mean_time": 10.0,
        "std_time": 0.2,
      },
      "AlgB": {"mean_gap": 1.5, "std_gap": 0.1, "mean_cost": 101.5, "std_cost": 1.0, "mean_time": 4.0, "std_time": 0.1},
      "AlgC": {
        "mean_gap": 2.0,
        "std_gap": 0.1,
        "mean_cost": 102.0,
        "std_cost": 1.0,
        "mean_time": 1.0,
        "std_time": 0.05,
      },
    },
    "p2": {
      "AlgA": {
        "mean_gap": 1.2,
        "std_gap": 0.1,
        "mean_cost": 202.4,
        "std_cost": 1.0,
        "mean_time": 20.0,
        "std_time": 0.4,
      },
      "AlgB": {"mean_gap": 1.4, "std_gap": 0.1, "mean_cost": 202.8, "std_cost": 1.0, "mean_time": 6.0, "std_time": 0.2},
      "AlgC": {"mean_gap": 2.1, "std_gap": 0.1, "mean_cost": 204.2, "std_cost": 1.0, "mean_time": 2.0, "std_time": 0.1},
    },
    "p3": {
      "AlgA": {
        "mean_gap": 0.9,
        "std_gap": 0.1,
        "mean_cost": 302.7,
        "std_cost": 1.0,
        "mean_time": 30.0,
        "std_time": 0.5,
      },
      "AlgB": {"mean_gap": 1.6, "std_gap": 0.1, "mean_cost": 304.8, "std_cost": 1.0, "mean_time": 9.0, "std_time": 0.3},
      "AlgC": {"mean_gap": 2.2, "std_gap": 0.1, "mean_cost": 306.6, "std_cost": 1.0, "mean_time": 3.0, "std_time": 0.1},
    },
  }
  problem_configs = [
    ProblemConfig(name="p1", optimal=100.0, size=50),
    ProblemConfig(name="p2", optimal=200.0, size=100),
    ProblemConfig(name="p3", optimal=300.0, size=300),
  ]

  generated = generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "all" / "tables",
    benchmark_name="single_pass",
    baseline_algorithm="AlgA",
  )
  assert set(generated.keys()) == {"markdown", "latex", "macros", "cd", "plots"}
  assert call_count == 2  # Exactly once for quality + once for time

  # Passing a precomputed CohortAnalysis incurs zero additional omnibus runs across all renderers
  precomputed = cohort_analysis_mod.analyze_cohort(all_results, problem_configs, baseline_algorithm="AlgA")
  assert call_count == 4
  generate_result_tables(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=tmp_path / "precomputed" / "tables",
    benchmark_name="precomputed",
    baseline_algorithm="AlgA",
    analysis=precomputed,
  )
  assert call_count == 4
