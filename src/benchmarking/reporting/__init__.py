"""Benchmark report renderers and console formatting (`src.benchmarking.reporting`).

Consumes immutable ``ProblemAnalysis`` and ``CohortAnalysis`` snapshots produced by
``benchmarking.analysis`` and renders terminal summaries, Markdown/LaTeX tables,
quantitative LaTeX macros, Demšar TikZ CD diagrams, and Matplotlib scaling plots.
"""

from benchmarking.reporting.cd_diagram import export_cd_diagram_tikz
from benchmarking.reporting.console import log_cohort_analysis, log_problem_analysis
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.reporting.latex_macros import generate_latex_macros
from benchmarking.reporting.style import (
  ALL_REPORT_ARTIFACTS,
  DEFAULT_SIZE_THRESHOLDS,
  ReportArtifact,
  ReportStyle,
  categorize_by_size,
  escape_latex_identifier,
  load_report_style,
  parse_report_artifacts,
  size_category_labels,
  validate_size_thresholds,
)
from benchmarking.reporting.table_reports import (
  generate_latex_tables,
  generate_markdown_tables,
)

__all__ = [
  "ALL_REPORT_ARTIFACTS",
  "DEFAULT_SIZE_THRESHOLDS",
  "ReportArtifact",
  "ReportStyle",
  "categorize_by_size",
  "escape_latex_identifier",
  "export_cd_diagram_tikz",
  "generate_latex_macros",
  "generate_latex_tables",
  "generate_markdown_tables",
  "generate_result_tables",
  "load_report_style",
  "log_cohort_analysis",
  "log_problem_analysis",
  "parse_report_artifacts",
  "size_category_labels",
  "validate_size_thresholds",
]
