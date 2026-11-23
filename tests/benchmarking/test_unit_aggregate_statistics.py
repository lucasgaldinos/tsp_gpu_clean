"""Unit tests for single-pass cross-problem cohort analysis (`benchmarking.analysis.cohort_analysis`).

WHAT: Covers `compare_algorithms`, `analyze_cohort`, unified `k >= 3, N >= 2` omnibus
contracts with `low_power` (< 5 blocks) and `omnibus_significant` flags, paired geometric
speedups, Wilcoxon + Holm-Bonferroni baseline comparisons, optional `cpu_threshold`
stratification, and `log_cohort_analysis` console rendering.
WHY: Ensures all cross-problem statistical contracts share one single source of truth
with zero logging side effects inside computation.
EXPECTED: Immutable typed `CohortAnalysis` / `StratumAnalysis` snapshots with exact
counts, ranks, speedups, and consistent omnibus/post-hoc metadata.
"""

from __future__ import annotations

import logging

import pytest

from benchmarking.analysis.cohort_analysis import (
  CohortAnalysis,
  StratumAnalysis,
  analyze_cohort,
  compare_algorithms,
)
from benchmarking.reporting.console import log_cohort_analysis
from benchmarking.runner.config_loader import ProblemConfig


def _constant_results(
  problems: list[str],
  algorithms: list[str],
  mean_times: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
  """Build `problem -> algorithm -> metrics` records with constant gaps."""
  return {
    problem: {algo: {"mean_gap": 5.0, "mean_time": mean_times[algo]} for algo in algorithms} for problem in problems
  }


@pytest.mark.parametrize(
  ("problems", "algorithms"),
  [
    pytest.param(["a"], ["CPU", "G", "H"], id="single_problem"),
    pytest.param(["a", "b", "c"], ["CPU", "G"], id="too_few_algorithms"),
  ],
)
def test_compare_algorithms_skips_insufficient_designs(problems: list[str], algorithms: list[str]) -> None:
  """WHAT: Unified domain requires k >= 3 algorithms and N >= 2 complete problems."""
  results = _constant_results(["a", "b", "c"], ["CPU", "G", "H"], {"CPU": 10.0, "G": 5.0, "H": 2.0})
  assert compare_algorithms(results, problems, algorithms) is None


def test_compare_algorithms_accepts_two_problems_and_marks_low_power() -> None:
  """WHAT: N = 2, k = 3 is inside the exact Iman-Davenport domain (df2 = 2) with low_power=True."""
  results = {
    "p1": {
      "A": {"mean_gap": 1.0, "mean_time": 10.0},
      "B": {"mean_gap": 2.0, "mean_time": 5.0},
      "C": {"mean_gap": 3.0, "mean_time": 1.0},
    },
    "p2": {
      "A": {"mean_gap": 1.2, "mean_time": 12.0},
      "B": {"mean_gap": 2.1, "mean_time": 6.0},
      "C": {"mean_gap": 3.3, "mean_time": 1.2},
    },
  }
  outcome = compare_algorithms(results, ["p1", "p2"], ["A", "B", "C"])
  assert isinstance(outcome, StratumAnalysis)
  assert outcome.omnibus is not None
  assert outcome.omnibus.n_blocks == 2
  assert outcome.omnibus.df1 == 2
  assert outcome.omnibus.df2 == 2
  assert outcome.omnibus.low_power is True


def test_compare_algorithms_reports_counts_and_input_order() -> None:
  """WHAT: Successful 3-algorithm x 3-problem comparison preserves caller problem and algorithm order."""
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["b", "a", "c"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})

  outcome = compare_algorithms(results, ["c", "a", "b"], algorithms)

  assert outcome is not None
  assert outcome.problems == ("c", "a", "b")
  assert outcome.algorithms == ("CPU", "G", "H")
  assert outcome.omnibus is not None
  assert outcome.omnibus.n_blocks == 3
  assert outcome.omnibus.k_algorithms == 3
  assert outcome.omnibus.omnibus_significant is False
  assert outcome.omnibus.low_power is True


def test_compare_algorithms_filters_incomplete_problems() -> None:
  """WHAT: Incomplete problems are excluded while preserving relative order of complete problems."""
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["a", "b", "c", "d"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})
  del results["d"]["H"]

  outcome = compare_algorithms(results, ["d", "a", "b", "c"], algorithms)

  assert outcome is not None
  assert outcome.problems == ("a", "b", "c")


def test_compare_algorithms_computes_geometric_mean_speedups() -> None:
  """WHAT: Baseline speedups compute Fleming & Wallace geometric means when baseline is supplied."""
  algorithms = ["CPU", "G", "H"]
  results = _constant_results(["a", "b", "c"], algorithms, {"CPU": 10.0, "G": 5.0, "H": 2.0})

  outcome = compare_algorithms(results, ["a", "b", "c"], algorithms, baseline_algorithm="CPU")
  assert outcome is not None
  assert outcome.baseline_algorithm == "CPU"
  assert outcome.speedups["CPU"].geomean == 1.0
  assert outcome.speedups["G"].geomean == pytest.approx(2.0, rel=1e-12)
  assert outcome.speedups["H"].geomean == pytest.approx(5.0, rel=1e-12)

  outcome_no_base = compare_algorithms(results, ["a", "b", "c"], algorithms)
  assert outcome_no_base is not None
  assert outcome_no_base.speedups == {}


def test_compare_algorithms_supports_time_metric_and_higher_is_better() -> None:
  """WHAT: compare_algorithms parameterizes metric_key and higher_is_better rank direction."""
  algorithms = ["Slow", "Medium", "Fast"]
  results = {
    p: {
      "Slow": {"mean_gap": 1.0, "mean_time": 10.0, "score": 10.0},
      "Medium": {"mean_gap": 2.0, "mean_time": 5.0, "score": 50.0},
      "Fast": {"mean_gap": 3.0, "mean_time": 1.0, "score": 100.0},
    }
    for p in ["p1", "p2", "p3", "p4", "p5"]
  }

  time_outcome = compare_algorithms(
    results,
    ["p1", "p2", "p3", "p4", "p5"],
    algorithms,
    baseline_algorithm="Slow",
    metric_key="mean_time",
    higher_is_better=False,
  )
  assert time_outcome is not None
  assert time_outcome.omnibus is not None
  assert time_outcome.omnibus.ranks == {
    "Fast": pytest.approx(1.0),
    "Medium": pytest.approx(2.0),
    "Slow": pytest.approx(3.0),
  }
  assert sum(time_outcome.omnibus.ranks.values()) == pytest.approx(6.0)
  assert time_outcome.omnibus.omnibus_significant is True
  assert time_outcome.omnibus.low_power is False
  assert time_outcome.speedups["Fast"].geomean == pytest.approx(10.0)

  score_outcome = compare_algorithms(
    results,
    ["p1", "p2", "p3", "p4", "p5"],
    algorithms,
    metric_key="score",
    higher_is_better=True,
  )
  assert score_outcome is not None
  assert score_outcome.omnibus is not None
  assert score_outcome.omnibus.ranks == {
    "Fast": pytest.approx(1.0),
    "Medium": pytest.approx(2.0),
    "Slow": pytest.approx(3.0),
  }


def test_analyze_cohort_unified_and_stratified_with_console_logging(caplog: pytest.LogCaptureFixture) -> None:
  """WHAT: `analyze_cohort` computes all cohort stats in one pure pass and `log_cohort_analysis` logs cleanly."""
  problem_configs = [
    ProblemConfig(name="p1", size=50, optimal=100.0),
    ProblemConfig(name="p2", size=80, optimal=200.0),
    ProblemConfig(name="p3", size=500, optimal=500.0),
  ]
  results = {
    p.name: {
      "BaseAlg": {"mean_gap": 1.0, "mean_time": 10.0},
      "GPU_A": {"mean_gap": 1.5, "mean_time": 5.0},
      "GPU_B": {"mean_gap": 2.0, "mean_time": 1.0},
    }
    for p in problem_configs
  }

  # 1. Pure computation emits zero INFO logs
  with caplog.at_level(logging.INFO):
    unified = analyze_cohort(results, problem_configs, baseline_algorithm="BaseAlg", cpu_threshold=None)
  assert not [r for r in caplog.records if r.levelno == logging.INFO]
  assert isinstance(unified, CohortAnalysis)
  assert unified.strata == ()
  assert unified.baseline_algorithm == "BaseAlg"
  assert unified.quality_omnibus is not None
  assert unified.time_omnibus is not None
  assert unified.baseline_speedups["GPU_B"].geomean == pytest.approx(10.0)
  assert len(unified.time_wilcoxon_vs_baseline) == 2

  # 2. Console renderer logs unified report cleanly without STRATUM 1/2
  caplog.clear()
  with caplog.at_level(logging.INFO):
    log_cohort_analysis(unified)
  log_text = "\n".join(r.getMessage() for r in caplog.records)
  assert "CROSS-PROBLEM ANALYSIS" in log_text
  assert "STRATUM 1" not in log_text

  # 3. Stratified mode with cpu_threshold=100 populates both strata (p1, p2 -> N=2 inside A3 domain!)
  stratified = analyze_cohort(results, problem_configs, baseline_algorithm="BaseAlg", cpu_threshold=100)
  assert stratified.cpu_threshold == 100
  assert len(stratified.strata) == 2
  assert stratified.strata[0].problems == ("p1", "p2")
  assert stratified.strata[0].omnibus is not None
  assert stratified.strata[0].omnibus.low_power is True
  assert stratified.strata[1].problems == ("p1", "p2", "p3")
