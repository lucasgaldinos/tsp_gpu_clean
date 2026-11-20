"""Statistical analysis, aggregation, and report generation (`src.benchmarking.analysis`)."""

from benchmarking.analysis.aggregate_statistics import (
  compare_algorithms,
  compare_algorithms_by_group,
  perform_cross_problem_analysis_stratified,
)
from benchmarking.analysis.cd_diagram import (
  CDResults,
  compute_cd_diagram_data,
  export_cd_diagram_tikz,
)
from benchmarking.analysis.problem_statistics import (
  holm_bonferroni_correction,
  perform_statistical_analysis,
)
from benchmarking.analysis.report_generator import generate_latex_macros, generate_result_tables
from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary
from benchmarking.analysis.stats_utils import (
  adjusted_r2,
  calculate_aic_bic,
  calculate_r2,
  loocv_regression,
  mean_absolute_error,
  mean_absolute_percentage_error,
  root_mean_squared_error,
)

__all__ = [
  "CDResults",
  "StatisticalAnalyzer",
  "StatisticalSummary",
  "adjusted_r2",
  "calculate_aic_bic",
  "calculate_r2",
  "compare_algorithms",
  "compare_algorithms_by_group",
  "compute_cd_diagram_data",
  "export_cd_diagram_tikz",
  "generate_latex_macros",
  "generate_result_tables",
  "holm_bonferroni_correction",
  "loocv_regression",
  "mean_absolute_error",
  "mean_absolute_percentage_error",
  "perform_cross_problem_analysis_stratified",
  "perform_statistical_analysis",
  "root_mean_squared_error",
]
