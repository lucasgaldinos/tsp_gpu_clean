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

from benchmarking.analysis.report_generator import (
  _PHASE_NAMES,
  _compute_table2_stats,
  _compute_table3_size_category_winners,
  _generate_table3_latex,
  _generate_table3_markdown,
  _has_phase_timings,
  generate_result_tables,
)
from benchmarking.analysis.stats_utils import resolve_baseline_algorithm
from benchmarking.runner.config_loader import ProblemConfig

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
  # Small (n<100): AlgoB has gap 1.0 vs AlgoA 2.0 -> AlgoB wins
  assert t3_winners[0]["category"] == "Small (n<100)"
  assert t3_winners[0]["best_alg"] == "AlgoB"
  assert t3_winners[0]["best_gap"] == 1.0
  # Large (n≥300): AlgoB has gap 2.0 vs AlgoA 4.0 -> AlgoB wins
  assert t3_winners[1]["category"] == "Large (n≥300)"
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
  assert winners[0]["category"] == "Large (n≥300)"
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
  md_lines = _generate_table3_markdown(all_results_partial, problem_configs, ["CPU", "FullGPU"])
  md_table = "\n".join(md_lines)
  assert "—" in md_table
  assert "None" not in md_table
  assert "inf" not in md_table

  tex_lines = _generate_table3_latex(all_results_partial, problem_configs, ["CPU", "FullGPU"])
  tex_table = "\n".join(tex_lines)
  assert "—" in tex_table
  assert "None" not in tex_table
  assert "inf" not in tex_table



