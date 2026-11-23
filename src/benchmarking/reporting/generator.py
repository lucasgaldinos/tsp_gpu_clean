"""Single-pass report orchestration facade (`DP14-2`, `DP14-5`).

Computes a ``CohortAnalysis`` snapshot once (or accepts a precomputed one from the
caller) and dispatches it to independent report renderers — Markdown tables, LaTeX
tables, quantitative LaTeX macros, Demšar Critical Difference TikZ diagrams, and
Matplotlib scaling figures — with uniform external ``ReportStyle`` injection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarking.analysis.cohort_analysis import CohortAnalysis, analyze_cohort
from benchmarking.reporting.latex_macros import generate_latex_macros
from benchmarking.reporting.style import (
  ALL_REPORT_ARTIFACTS,
  ReportArtifact,
  ReportStyle,
)
from benchmarking.reporting.table_reports import (
  generate_latex_tables,
  generate_markdown_tables,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.runner.config_loader import ProblemConfig


def _generate_cd_diagrams(
  analysis: CohortAnalysis,
  output_dir: Path,
  benchmark_name: str,
  style: ReportStyle,
) -> tuple[Path, ...]:
  """Render Demšar Critical Difference TikZ diagrams directly from ``CohortAnalysis``."""
  from benchmarking.reporting.cd_diagram import export_cd_diagram_tikz

  quality_label = style.cd_quality_label if analysis.quality_metric_key == "mean_gap" else style.cd_cost_label
  cd_specs = (
    (analysis.quality_omnibus, f"{benchmark_name}_cd_diagram_gap.tikz", quality_label),
    (analysis.time_omnibus, f"{benchmark_name}_cd_diagram_time.tikz", style.cd_time_label),
  )

  written: list[Path] = []
  for omnibus, filename, metric_label in cd_specs:
    if omnibus is None:
      continue
    try:
      tikz_path = output_dir / filename
      export_cd_diagram_tikz(
        omnibus,
        tikz_path,
        metric_label=metric_label,
        style=style,
      )
      written.append(tikz_path)
      logging.info("  CD Diagram TikZ (%s): %s", metric_label, tikz_path)
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
  analysis: CohortAnalysis | None = None,
  alpha: float = 0.05,
) -> dict[ReportArtifact, tuple[Path, ...]]:
  """Generate selected benchmark report artifacts from a single ``CohortAnalysis`` pass."""
  if analysis is None and not all_results:
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

  active_analysis = analysis or analyze_cohort(
    all_results,
    problem_configs,
    baseline_algorithm=baseline_algorithm,
    alpha=alpha,
    algorithm_order=effective_style.algorithm_order,
  )
  cohort = active_analysis.cohort
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
      analysis=active_analysis,
    )
    generated["markdown"] = (md_path,)
    logging.info("  Markdown: %s", md_path)

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
      analysis=active_analysis,
    )
    generated["latex"] = (tex_path,)
    logging.info("  LaTeX: %s", tex_path)

  if "macros" in selected:
    try:
      macros_path = generate_latex_macros(
        all_results=all_results,
        problem_configs=problem_configs,
        output_dir=out_dir,
        baseline_algorithm=baseline_algorithm,
        style=effective_style,
        cohort=cohort,
        analysis=active_analysis,
      )
      generated["macros"] = (macros_path,)
      logging.info("  LaTeX Macros: %s", macros_path)
    except Exception as e:
      logging.warning("Could not generate LaTeX macros: %s", e)
      generated["macros"] = ()

  if "cd" in selected:
    generated["cd"] = _generate_cd_diagrams(
      analysis=active_analysis,
      output_dir=out_dir,
      benchmark_name=benchmark_name,
      style=effective_style,
    )

  if "plots" in selected:
    figures_dir = (out_dir.parent / "figures") if out_dir.name == "tables" else (out_dir / "figures")
    try:
      from benchmarking.reporting.scaling_plots import generate_scaling_plots

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
