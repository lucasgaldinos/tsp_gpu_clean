"""Composable report orchestration facade (`DP14-2`, `DP14-5`).

Wires independent report generators — Markdown tables, LaTeX tables, quantitative
LaTeX macros, Demšar Critical Difference TikZ diagrams, and Matplotlib scaling
figures — behind ``generate_result_tables`` with a single ``BenchmarkCohort`` view,
explicit artifact selection, and uniform external ``ReportStyle`` injection.

Each generator can also be imported and called directly without this facade:
- ``benchmarking.analysis.table_reports.generate_markdown_tables``
- ``benchmarking.analysis.table_reports.generate_latex_tables``
- ``benchmarking.analysis.latex_macros.generate_latex_macros``
- ``benchmarking.analysis.cd_diagram.{compute_cd_diagram_data, export_cd_diagram_tikz}``
- ``benchmarking.analysis.scaling_plots.generate_scaling_plots``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarking.analysis.cohort import BenchmarkCohort
from benchmarking.analysis.latex_macros import generate_latex_macros
from benchmarking.analysis.report_style import (
  ALL_REPORT_ARTIFACTS,
  ReportArtifact,
  ReportStyle,
)
from benchmarking.analysis.table_reports import (
  generate_latex_tables,
  generate_markdown_tables,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.runner.config_loader import ProblemConfig


def _generate_cd_diagrams(
  cohort: BenchmarkCohort,
  output_dir: Path,
  benchmark_name: str,
  style: ReportStyle,
) -> tuple[Path, ...]:
  """Generate Demšar Critical Difference TikZ diagrams for quality and execution time."""
  algorithm_names = list(cohort.algorithms)
  if len(algorithm_names) < 3 or len(cohort.problems) < 2:
    return ()

  from benchmarking.analysis.cd_diagram import compute_cd_diagram_data, export_cd_diagram_tikz

  gap_metric = cohort.quality_metric_key()
  quality_label = style.cd_quality_label if gap_metric == "mean_gap" else style.cd_cost_label

  cd_specs: tuple[tuple[str, str, str], ...] = (
    (gap_metric, f"{benchmark_name}_cd_diagram_gap.tikz", quality_label),
    ("mean_time", f"{benchmark_name}_cd_diagram_time.tikz", style.cd_time_label),
  )
  written: list[Path] = []
  for metric_key, filename, metric_label in cd_specs:
    valid_problems, matrix = cohort.metric_matrix(metric_key)
    if len(valid_problems) < 2:
      continue
    try:
      cd_data = compute_cd_diagram_data(matrix, algorithm_names)
      tikz_path = output_dir / filename
      export_cd_diagram_tikz(
        cd_data,
        tikz_path,
        metric_label=metric_label,
        style=style,
      )
      written.append(tikz_path)
      logging.info(f"  CD Diagram TikZ ({metric_label}): {tikz_path}")
    except Exception as e:
      logging.warning("Could not generate Demšar CD diagram for %s: %s", metric_label, e)

  return tuple(written)


def generate_result_tables(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: Sequence[ProblemConfig],
  output_dir: Path | str,
  benchmark_name: str = "benchmark",
  baseline_algorithm: str | None = None,
  *,
  include_size_category_table: bool = False,
  style: ReportStyle | None = None,
  artifacts: Sequence[ReportArtifact] | None = None,
) -> dict[ReportArtifact, tuple[Path, ...]]:
  """Generate selected benchmark report artifacts with injectable ``ReportStyle``.

  Args:
      all_results: Nested mapping ``problem_name -> algorithm_name -> summary_dict``.
      problem_configs: Problem metadata configurations for sizes and strata.
      output_dir: Directory where table/macro/CD artifacts are written. Scaling plots
          write into the sibling ``figures/`` directory when ``output_dir.name == "tables"``,
          otherwise into ``output_dir / "figures"``.
      benchmark_name: Output file stem for tables and CD diagrams (default: ``"benchmark"``).
      baseline_algorithm: Optional baseline algorithm for speedup ratios.
      include_size_category_table: Whether to include Table 3 (size strata) in tables.
      style: Optional ``ReportStyle`` customizing display names, labels, decimal format,
          LaTeX label prefix, and size thresholds. Defaults to neutral English ``ReportStyle()``.
      artifacts: Optional subset of ``("markdown", "latex", "macros", "cd", "plots")``.
          ``None`` emits all five artifacts; an empty sequence emits nothing.

  Returns:
      Mapping from each requested ``ReportArtifact`` to the tuple of paths written.
  """
  if not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_style = style or ReportStyle()
  selected: tuple[ReportArtifact, ...] = ALL_REPORT_ARTIFACTS if artifacts is None else tuple(artifacts)
  invalid = [item for item in selected if item not in ALL_REPORT_ARTIFACTS]
  if invalid:
    raise ValueError(f"Unsupported report artifact(s): {invalid}. Expected subset of {list(ALL_REPORT_ARTIFACTS)}.")

  if not selected:
    return {}

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  cohort = BenchmarkCohort.from_results(
    all_results,
    problem_configs,
    algorithm_order=effective_style.algorithm_order,
  )
  generated: dict[ReportArtifact, tuple[Path, ...]] = {}

  logging.info(
    "Generating report artifacts (%s) for %d problem(s), %d algorithm(s)",
    ", ".join(selected),
    len(cohort.problems),
    len(cohort.algorithms),
  )

  if "markdown" in selected:
    md_path = generate_markdown_tables(
      all_results=all_results,
      problem_configs=problem_configs,
      output_dir=out_dir,
      benchmark_name=benchmark_name,
      baseline_algorithm=baseline_algorithm,
      include_size_category_table=include_size_category_table,
      style=effective_style,
      cohort=cohort,
    )
    generated["markdown"] = (md_path,)
    logging.info(f"  Markdown: {md_path}")

  if "latex" in selected:
    tex_path = generate_latex_tables(
      all_results=all_results,
      problem_configs=problem_configs,
      output_dir=out_dir,
      benchmark_name=benchmark_name,
      baseline_algorithm=baseline_algorithm,
      include_size_category_table=include_size_category_table,
      style=effective_style,
      cohort=cohort,
    )
    generated["latex"] = (tex_path,)
    logging.info(f"  LaTeX: {tex_path}")

  if "macros" in selected:
    try:
      macros_path = generate_latex_macros(
        all_results=all_results,
        problem_configs=problem_configs,
        output_dir=out_dir,
        baseline_algorithm=baseline_algorithm,
        style=effective_style,
        cohort=cohort,
      )
      generated["macros"] = (macros_path,)
      logging.info(f"  LaTeX Macros: {macros_path}")
    except Exception as e:
      logging.warning("Could not generate LaTeX macros: %s", e)
      generated["macros"] = ()

  if "cd" in selected:
    generated["cd"] = _generate_cd_diagrams(
      cohort=cohort,
      output_dir=out_dir,
      benchmark_name=benchmark_name,
      style=effective_style,
    )

  if "plots" in selected:
    figures_dir = (out_dir.parent / "figures") if out_dir.name == "tables" else (out_dir / "figures")
    try:
      from benchmarking.analysis.scaling_plots import generate_scaling_plots

      generated_plots = generate_scaling_plots(
        all_results=all_results,
        problem_configs=problem_configs,
        output_dir=figures_dir,
        style=effective_style,
      )
      generated["plots"] = generated_plots
      logging.info("  Scaling Plots: %s", ", ".join(str(p) for p in generated_plots))
    except Exception as e:
      logging.warning("Could not generate scaling plots: %s", e)
      generated["plots"] = ()

  return generated


__all__ = ["generate_result_tables"]
