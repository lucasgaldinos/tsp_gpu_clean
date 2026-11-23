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
from benchmarking.analysis.cohort import (
  AlgorithmAggregate,
  BenchmarkCohort,
  SpeedupSummary,
  derive_problem_configs,
  resolve_algorithm_names,
)
from benchmarking.analysis.latex_macros import generate_latex_macros
from benchmarking.analysis.problem_statistics import (
  perform_statistical_analysis,
)
from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.analysis.report_style import (
  ALL_REPORT_ARTIFACTS,
  DEFAULT_SIZE_THRESHOLDS,
  ReportArtifact,
  ReportStyle,
  categorize_by_size,
  escape_latex_identifier,
  load_report_style,
  parse_report_artifacts,
  size_category_labels,
)
from benchmarking.analysis.statistics import StatisticalAnalyzer, StatisticalSummary
from benchmarking.analysis.stats_utils import (
  adjusted_r2,
  calculate_aic_bic,
  calculate_r2,
  finite_metric,
  geometric_mean_ratios,
  holm_bonferroni_correction,
  loocv_regression,
  mean_absolute_error,
  mean_absolute_percentage_error,
  root_mean_squared_error,
)
from benchmarking.analysis.table_reports import (
  generate_latex_tables,
  generate_markdown_tables,
)

__all__ = [
  "ALL_REPORT_ARTIFACTS",
  "DEFAULT_SIZE_THRESHOLDS",
  "AlgorithmAggregate",
  "BenchmarkCohort",
  "CDResults",
  "ReportArtifact",
  "ReportStyle",
  "SpeedupSummary",
  "StatisticalAnalyzer",
  "StatisticalSummary",
  "adjusted_r2",
  "calculate_aic_bic",
  "calculate_r2",
  "categorize_by_size",
  "compare_algorithms",
  "compare_algorithms_by_group",
  "compute_cd_diagram_data",
  "derive_problem_configs",
  "escape_latex_identifier",
  "export_cd_diagram_tikz",
  "finite_metric",
  "generate_latex_macros",
  "generate_latex_tables",
  "generate_markdown_tables",
  "generate_result_tables",
  "geometric_mean_ratios",
  "holm_bonferroni_correction",
  "load_report_style",
  "loocv_regression",
  "mean_absolute_error",
  "mean_absolute_percentage_error",
  "parse_report_artifacts",
  "perform_cross_problem_analysis_stratified",
  "perform_statistical_analysis",
  "resolve_algorithm_names",
  "root_mean_squared_error",
  "size_category_labels",
]
