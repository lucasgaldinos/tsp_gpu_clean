"""Pure statistical analysis package for benchmark evaluation (`src.benchmarking.analysis`).

Contains strictly statistical computation and immutable analytical snapshots with zero
rendering or console-formatting side effects. Presentation, console formatting, tables,
LaTeX macros, TikZ CD diagrams, and Matplotlib plots live in ``benchmarking.reporting``.
"""

from benchmarking.analysis.cohort import (
  AlgorithmAggregate,
  BenchmarkCohort,
  SpeedupSummary,
  derive_problem_configs,
  resolve_algorithm_names,
)
from benchmarking.analysis.cohort_analysis import (
  BaselineWilcoxonComparison,
  CohortAnalysis,
  StratumAnalysis,
  analyze_cohort,
  compare_algorithms,
)
from benchmarking.analysis.problem_statistics import (
  AlgorithmProblemRow,
  PairwiseComparisonRecord,
  ProblemAnalysis,
  analyze_problem,
)
from benchmarking.analysis.statistics import (
  OmnibusTestResult,
  StatisticalAnalyzer,
  StatisticalSummary,
  compute_iman_davenport_f,
  compute_mean_ranks,
  interpret_rank_biserial,
  run_friedman_nemenyi_analysis,
)
from benchmarking.analysis.stats_utils import (
  compute_nemenyi_critical_distance,
  finite_metric,
  geometric_mean_ratios,
  holm_bonferroni_correction,
  resolve_baseline_algorithm,
)

__all__ = [
  "AlgorithmAggregate",
  "AlgorithmProblemRow",
  "BaselineWilcoxonComparison",
  "BenchmarkCohort",
  "CohortAnalysis",
  "OmnibusTestResult",
  "PairwiseComparisonRecord",
  "ProblemAnalysis",
  "SpeedupSummary",
  "StatisticalAnalyzer",
  "StatisticalSummary",
  "StratumAnalysis",
  "analyze_cohort",
  "analyze_problem",
  "compare_algorithms",
  "compute_iman_davenport_f",
  "compute_mean_ranks",
  "compute_nemenyi_critical_distance",
  "derive_problem_configs",
  "finite_metric",
  "geometric_mean_ratios",
  "holm_bonferroni_correction",
  "interpret_rank_biserial",
  "resolve_algorithm_names",
  "resolve_baseline_algorithm",
  "run_friedman_nemenyi_analysis",
]
