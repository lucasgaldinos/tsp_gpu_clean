"""Pure quantitative LaTeX ``\\newcommand`` macro exporter (`DP14-5`).

Reads a precomputed ``CohortAnalysis`` snapshot directly without running any
statistical tests itself, emitting strictly numeric LaTeX macros (cohort counts,
per-algorithm means/SDs, paired geometric speedups, Friedman/Iman-Davenport/Nemenyi
statistics, and paired Wilcoxon + Holm-Bonferroni statistics).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarking.analysis.cohort_analysis import CohortAnalysis, analyze_cohort
from benchmarking.reporting.style import ReportStyle, escape_latex_identifier

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.analysis.cohort import BenchmarkCohort
  from benchmarking.analysis.cohort_analysis import BaselineWilcoxonComparison
  from benchmarking.analysis.statistics import OmnibusTestResult
  from benchmarking.runner.config_loader import ProblemConfig

_MACRO_DIGIT_WORDS: dict[str, str] = {
  "0": "Zero",
  "1": "One",
  "2": "Two",
  "3": "Three",
  "4": "Four",
  "5": "Five",
  "6": "Six",
  "7": "Seven",
  "8": "Eight",
  "9": "Nine",
}


def _latex_macro_token(name: str) -> str:
  """Sanitize a canonical algorithm key into a valid ASCII-letter-only LaTeX macro suffix."""
  parts: list[str] = []
  for ch in name:
    if ch.isascii() and ch.isalpha():
      parts.append(ch)
    elif ch in _MACRO_DIGIT_WORDS:
      parts.append(_MACRO_DIGIT_WORDS[ch])
  return "".join(parts) or "Algorithm"


def _build_algorithm_macro_tokens(algorithm_names: Sequence[str]) -> dict[str, str]:
  """Map each canonical algorithm name to a unique valid LaTeX command suffix."""
  tokens: dict[str, str] = {}
  used: dict[str, int] = {}
  for alg in algorithm_names:
    base = _latex_macro_token(alg)
    if base not in used:
      used[base] = 1
      tokens[alg] = base
    else:
      used[base] += 1
      suffix_idx = used[base]
      candidate = f"{base}{_MACRO_DIGIT_WORDS.get(str(suffix_idx), 'Extra')}"
      while candidate in used:
        suffix_idx += 1
        candidate = f"{base}{suffix_idx * 'X'}"
      used[candidate] = 1
      tokens[alg] = candidate
  return tokens


def _format_decimal_latex(
  value: float,
  decimals: int = 2,
  decimal_separator: str = ".",
) -> str:
  """Format a finite or infinite float for LaTeX with configurable decimal separator."""
  if math.isnan(value):
    return r"\mathrm{NaN}"
  if math.isinf(value):
    return r"\ensuremath{\infty}" if value > 0 else r"\ensuremath{-\infty}"
  if abs(value) < 0.5 * (10.0**-decimals):
    value = 0.0
  return f"{value:.{decimals}f}".replace(".", decimal_separator)


def _format_pvalue_latex(p_val: float, decimal_separator: str = ".") -> str:
  """Format a statistical p-value in fixed decimal or LaTeX scientific notation."""
  if math.isnan(p_val):
    return r"\mathrm{NaN}"
  if p_val <= 0.0:
    return r"\ensuremath{< 10^{-6}}"
  if p_val < 1e-3:
    exp = math.floor(math.log10(p_val))
    mant = p_val / (10.0**exp)
    if round(mant, 2) >= 10.0:
      mant /= 10.0
      exp += 1
    mant_str = f"{mant:.2f}".replace(".", decimal_separator)
    return rf"\ensuremath{{{mant_str} \times 10^{{{exp}}}}}"
  if p_val >= 0.99995:
    return f"1{decimal_separator}00"
  return f"{p_val:.4f}".replace(".", decimal_separator)


def _emit_per_algorithm_macros(
  analysis: CohortAnalysis,
  tokens: dict[str, str],
  decimal_separator: str,
) -> list[str]:
  """Emit aggregated per-algorithm mean/SD and baseline geometric-speedup macros."""
  lines: list[str] = ["", "% Aggregated per-algorithm statistics across evaluated instances"]
  for alg in analysis.cohort.algorithms:
    tok = tokens[alg]
    agg = analysis.per_algorithm.get(alg)
    if agg is not None:
      if agg.mean_gap is not None and agg.std_gap is not None:
        lines.append(f"\\newcommand{{\\meanGap{tok}}}{{{_format_decimal_latex(agg.mean_gap, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\stdGap{tok}}}{{{_format_decimal_latex(agg.std_gap, 2, decimal_separator)}}}")
      elif agg.mean_cost is not None and agg.std_cost is not None:
        lines.append(f"\\newcommand{{\\meanCost{tok}}}{{{_format_decimal_latex(agg.mean_cost, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\stdCost{tok}}}{{{_format_decimal_latex(agg.std_cost, 2, decimal_separator)}}}")
      if agg.mean_time is not None and agg.std_time is not None:
        lines.append(f"\\newcommand{{\\meanTime{tok}}}{{{_format_decimal_latex(agg.mean_time, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\stdTime{tok}}}{{{_format_decimal_latex(agg.std_time, 2, decimal_separator)}}}")

    if alg == analysis.baseline_algorithm:
      lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(1.0, 2, decimal_separator)}}}")
    else:
      sp = analysis.baseline_speedups.get(alg)
      if sp is not None and sp.geomean is not None:
        lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(sp.geomean, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\speedupSampleSize{tok}}}{{{sp.sample_size}}}")
  return lines


def _emit_pairwise_speedup_macros(
  analysis: CohortAnalysis,
  tokens: dict[str, str],
  decimal_separator: str,
) -> list[str]:
  """Emit ordered pairwise geometric mean speedup macros (Fleming & Wallace, 1986)."""
  pairwise_lines: list[str] = []
  for alg_a in analysis.cohort.algorithms:
    for alg_b in analysis.cohort.algorithms:
      if alg_a == alg_b:
        continue
      sp = analysis.pairwise_speedups.get((alg_a, alg_b))
      if sp is not None and sp.geomean is not None:
        tok_a = tokens[alg_a]
        tok_b = tokens[alg_b]
        pairwise_lines.append(
          f"\\newcommand{{\\speedup{tok_a}Vs{tok_b}}}{{{_format_decimal_latex(sp.geomean, 2, decimal_separator)}}}"
        )
  if not pairwise_lines:
    return []
  return ["", "% Pairwise geometric mean speedups (Fleming & Wallace, 1986)", *pairwise_lines]


def _emit_omnibus_block(
  metric_tag: str,
  omnibus: OmnibusTestResult | None,
  tokens: dict[str, str],
  decimal_separator: str,
) -> list[str]:
  """Emit Friedman / Iman-Davenport omnibus and Nemenyi post-hoc macros for one metric."""
  if omnibus is None:
    return []

  chi2_2 = _format_decimal_latex(omnibus.chi2_stat, 2, decimal_separator)
  chi2_4 = _format_decimal_latex(omnibus.chi2_stat, 4, decimal_separator)
  chi2_p = _format_pvalue_latex(omnibus.chi2_pvalue, decimal_separator)
  f_tex = _format_decimal_latex(omnibus.friedman_f, 4, decimal_separator)
  f_p_tex = _format_pvalue_latex(omnibus.friedman_pvalue, decimal_separator)
  cd_tex = _format_decimal_latex(omnibus.critical_difference, 3, decimal_separator)

  lines: list[str] = [
    "",
    f"% Omnibus Friedman / Iman-Davenport & Nemenyi statistics ({metric_tag})",
    f"\\newcommand{{\\friedman{metric_tag}N}}{{{omnibus.n_blocks}}}",
    f"\\newcommand{{\\friedman{metric_tag}K}}{{{omnibus.k_algorithms}}}",
    f"\\newcommand{{\\friedman{metric_tag}ChiSq}}{{{chi2_2}}}",
    f"\\newcommand{{\\friedman{metric_tag}ChiSqExact}}{{{chi2_4}}}",
    f"\\newcommand{{\\friedman{metric_tag}ChiSqP}}{{{chi2_p}}}",
    f"\\newcommand{{\\friedman{metric_tag}F}}{{{f_tex}}}",
    f"\\newcommand{{\\friedman{metric_tag}DfOne}}{{{omnibus.df1}}}",
    f"\\newcommand{{\\friedman{metric_tag}DfTwo}}{{{omnibus.df2}}}",
    f"\\newcommand{{\\friedman{metric_tag}P}}{{{f_p_tex}}}",
    f"\\newcommand{{\\nemenyi{metric_tag}CD}}{{{cd_tex}}}",
  ]
  for alg in omnibus.algorithm_names:
    tok = tokens[alg]
    rank_tex = _format_decimal_latex(omnibus.ranks[alg], 2, decimal_separator)
    lines.append(f"\\newcommand{{\\meanRank{metric_tag}{tok}}}{{{rank_tex}}}")

  if all(math.isfinite(p) for row in omnibus.nemenyi_pvalues for p in row):
    k = len(omnibus.algorithm_names)
    for i in range(k):
      for j in range(i + 1, k):
        tok_i = tokens[omnibus.algorithm_names[i]]
        tok_j = tokens[omnibus.algorithm_names[j]]
        p_str = _format_pvalue_latex(omnibus.nemenyi_pvalues[i][j], decimal_separator)
        lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_i}Vs{tok_j}}}{{{p_str}}}")
        lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_j}Vs{tok_i}}}{{{p_str}}}")
  return lines


def _emit_wilcoxon_block(
  metric_tag: str,
  comparisons: Sequence[BaselineWilcoxonComparison],
  tokens: dict[str, str],
  decimal_separator: str,
) -> list[str]:
  """Emit paired Wilcoxon signed-rank + Holm-Bonferroni macros against baseline."""
  if not comparisons:
    return []
  lines: list[str] = ["", f"% Paired Wilcoxon + Holm-Bonferroni vs baseline ({metric_tag})"]
  for comp in comparisons:
    tok = tokens[comp.algorithm]
    w_tex = _format_decimal_latex(comp.w_statistic, 1, decimal_separator)
    p_tex = _format_pvalue_latex(comp.p_value, decimal_separator)
    p_holm_tex = _format_pvalue_latex(comp.p_value_holm, decimal_separator)
    eff_tex = _format_decimal_latex(comp.rank_biserial_r, 2, decimal_separator)
    lines.extend(
      [
        f"\\newcommand{{\\wilcoxon{metric_tag}W{tok}}}{{{w_tex}}}",
        f"\\newcommand{{\\wilcoxon{metric_tag}P{tok}}}{{{p_tex}}}",
        f"\\newcommand{{\\wilcoxon{metric_tag}PHolm{tok}}}{{{p_holm_tex}}}",
        f"\\newcommand{{\\wilcoxon{metric_tag}Effect{tok}}}{{{eff_tex}}}",
      ]
    )
  return lines


def generate_latex_macros(
  all_results: dict[str, dict[str, dict[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None,
  output_dir: Path | str,
  baseline_algorithm: str | None = None,
  *,
  filename: str = "benchmark_macros.tex",
  decimal_separator: str | None = None,
  alpha: float = 0.05,
  style: ReportStyle | None = None,
  cohort: BenchmarkCohort | None = None,
  analysis: CohortAnalysis | None = None,
) -> Path:
  """Export strictly quantitative LaTeX ``\\newcommand`` macros to ``benchmark_macros.tex``."""
  if analysis is None and not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_sep = (
    decimal_separator if decimal_separator is not None else (style.decimal_separator if style is not None else ".")
  )
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  macro_path = out_dir / filename

  active_analysis = analysis or analyze_cohort(
    cohort if cohort is not None else all_results,
    problem_configs,
    baseline_algorithm=baseline_algorithm,
    alpha=alpha,
    algorithm_order=style.algorithm_order if style is not None else None,
  )
  active_cohort = active_analysis.cohort
  tokens = _build_algorithm_macro_tokens(active_cohort.algorithms)
  rep_counts = active_cohort.repetition_counts()

  lines: list[str] = [
    "% Auto-generated quantitative benchmark macros (`DP14-5`)",
    "% Strictly numeric values formatted with configurable decimal separator.",
    f"\\newcommand{{\\benchmarkNumProblems}}{{{len(active_cohort.problems)}}}",
    f"\\newcommand{{\\benchmarkNumAlgorithms}}{{{len(active_cohort.algorithms)}}}",
    f"\\newcommand{{\\benchmarkBaselineAlgorithm}}{{{escape_latex_identifier(active_analysis.baseline_algorithm)}}}",
    f"\\newcommand{{\\benchmarkTotalRuns}}{{{sum(rep_counts)}}}",
  ]
  if rep_counts:
    min_rep, max_rep = min(rep_counts), max(rep_counts)
    rep_macro = str(min_rep) if min_rep == max_rep else f"{min_rep}--{max_rep}"
    lines.append(f"\\newcommand{{\\benchmarkRepetitions}}{{{rep_macro}}}")

  lines.extend(_emit_per_algorithm_macros(active_analysis, tokens, effective_sep))
  lines.extend(_emit_pairwise_speedup_macros(active_analysis, tokens, effective_sep))
  lines.extend(_emit_omnibus_block("Gap", active_analysis.quality_omnibus, tokens, effective_sep))
  lines.extend(_emit_omnibus_block("Time", active_analysis.time_omnibus, tokens, effective_sep))
  lines.extend(_emit_wilcoxon_block("Gap", active_analysis.quality_wilcoxon_vs_baseline, tokens, effective_sep))
  lines.extend(_emit_wilcoxon_block("Time", active_analysis.time_wilcoxon_vs_baseline, tokens, effective_sep))

  macro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return macro_path


__all__ = ["generate_latex_macros"]
