"""Quantitative LaTeX ``\\newcommand`` macro export for benchmark summaries (`DP14-5`).

Emits strictly numeric macros (cohort counts, per-algorithm means/SDs, paired geometric
speedups, Friedman/Iman-Davenport/Nemenyi tests, and paired Wilcoxon + Holm-Bonferroni
statistics) backed by ``BenchmarkCohort`` and ``StatisticalAnalyzer``, with a neutral
``"."`` decimal separator by default, overridable via ``decimal_separator`` or ``ReportStyle``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.cohort import BenchmarkCohort
from benchmarking.analysis.report_style import ReportStyle, escape_latex_identifier
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.analysis.stats_utils import holm_bonferroni_correction

if TYPE_CHECKING:
  from collections.abc import Sequence

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
  cohort: BenchmarkCohort,
  tokens: dict[str, str],
  effective_baseline: str,
  decimal_separator: str,
) -> list[str]:
  """Emit aggregated per-algorithm mean/SD and baseline geometric-speedup macros."""
  lines: list[str] = ["", "% Aggregated per-algorithm statistics across evaluated instances"]
  for alg in cohort.algorithms:
    tok = tokens[alg]
    agg = cohort.summarize_algorithm(alg)
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

    if alg == effective_baseline:
      lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(1.0, 2, decimal_separator)}}}")
    else:
      sp = cohort.paired_geometric_speedup(alg, effective_baseline)
      if sp.geomean is not None:
        lines.append(f"\\newcommand{{\\speedupGeom{tok}}}{{{_format_decimal_latex(sp.geomean, 2, decimal_separator)}}}")
        lines.append(f"\\newcommand{{\\speedupSampleSize{tok}}}{{{sp.sample_size}}}")
  return lines


def _emit_pairwise_speedup_macros(
  cohort: BenchmarkCohort,
  tokens: dict[str, str],
  decimal_separator: str,
) -> list[str]:
  """Emit ordered pairwise geometric mean speedup macros (Fleming & Wallace, 1986)."""
  pairwise_lines: list[str] = []
  for alg_a in cohort.algorithms:
    for alg_b in cohort.algorithms:
      if alg_a == alg_b:
        continue
      sp = cohort.paired_geometric_speedup(alg_a, alg_b)
      if sp.geomean is not None:
        tok_a = tokens[alg_a]
        tok_b = tokens[alg_b]
        pairwise_lines.append(
          f"\\newcommand{{\\speedup{tok_a}Vs{tok_b}}}{{{_format_decimal_latex(sp.geomean, 2, decimal_separator)}}}"
        )
  if not pairwise_lines:
    return []
  return ["", "% Pairwise geometric mean speedups (Fleming & Wallace, 1986)", *pairwise_lines]


def _emit_omnibus_macros(
  cohort: BenchmarkCohort,
  tokens: dict[str, str],
  analyzer: StatisticalAnalyzer,
  gap_metric_key: str,
  alpha: float,
  decimal_separator: str,
) -> list[str]:
  """Emit Friedman / Iman-Davenport omnibus and Nemenyi post-hoc macros."""
  algorithm_names = list(cohort.algorithms)
  if len(algorithm_names) < 3 or len(cohort.complete_problems()) < 2:
    return []

  from benchmarking.analysis.cd_diagram import compute_cd_diagram_data

  lines: list[str] = []
  for metric_tag, metric_key in (("Gap", gap_metric_key), ("Time", "mean_time")):
    valid_problems, matrix = cohort.metric_matrix(metric_key)
    if len(valid_problems) < 2:
      continue

    cd_data = compute_cd_diagram_data(matrix, algorithm_names, alpha=alpha)
    cols = [matrix[:, j] for j in range(len(algorithm_names))]
    friedman_res = analyzer.friedman_test(cols)

    chi2_2 = _format_decimal_latex(friedman_res["statistic"], 2, decimal_separator)
    chi2_4 = _format_decimal_latex(friedman_res["statistic"], 4, decimal_separator)
    chi2_p = _format_pvalue_latex(friedman_res["p_value"], decimal_separator)
    f_tex = _format_decimal_latex(cd_data.friedman_f, 4, decimal_separator)
    f_p_tex = _format_pvalue_latex(cd_data.friedman_pvalue, decimal_separator)
    cd_tex = _format_decimal_latex(cd_data.critical_difference, 3, decimal_separator)

    lines.extend(
      [
        "",
        f"% Omnibus Friedman / Iman-Davenport & Nemenyi statistics ({metric_tag})",
        f"\\newcommand{{\\friedman{metric_tag}N}}{{{len(valid_problems)}}}",
        f"\\newcommand{{\\friedman{metric_tag}K}}{{{len(algorithm_names)}}}",
        f"\\newcommand{{\\friedman{metric_tag}ChiSq}}{{{chi2_2}}}",
        f"\\newcommand{{\\friedman{metric_tag}ChiSqExact}}{{{chi2_4}}}",
        f"\\newcommand{{\\friedman{metric_tag}ChiSqP}}{{{chi2_p}}}",
        f"\\newcommand{{\\friedman{metric_tag}F}}{{{f_tex}}}",
        f"\\newcommand{{\\friedman{metric_tag}DfOne}}{{{friedman_res['df1']}}}",
        f"\\newcommand{{\\friedman{metric_tag}DfTwo}}{{{friedman_res['df2']}}}",
        f"\\newcommand{{\\friedman{metric_tag}P}}{{{f_p_tex}}}",
        f"\\newcommand{{\\nemenyi{metric_tag}CD}}{{{cd_tex}}}",
      ]
    )
    for alg in algorithm_names:
      tok = tokens[alg]
      rank_tex = _format_decimal_latex(cd_data.ranks[alg], 2, decimal_separator)
      lines.append(f"\\newcommand{{\\meanRank{metric_tag}{tok}}}{{{rank_tex}}}")

    if not bool(np.all(np.ptp(matrix, axis=1) == 0.0)):
      nemenyi_res = analyzer.nemenyi_posthoc(cols, algorithm_names)
      p_mat = np.asarray(nemenyi_res["p_values_matrix"], dtype=float)
      if bool(np.all(np.isfinite(p_mat))):
        for i in range(len(algorithm_names)):
          for j in range(i + 1, len(algorithm_names)):
            tok_i = tokens[algorithm_names[i]]
            tok_j = tokens[algorithm_names[j]]
            p_str = _format_pvalue_latex(float(p_mat[i, j]), decimal_separator)
            lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_i}Vs{tok_j}}}{{{p_str}}}")
            lines.append(f"\\newcommand{{\\nemenyi{metric_tag}P{tok_j}Vs{tok_i}}}{{{p_str}}}")
  return lines


def _emit_wilcoxon_macros(
  cohort: BenchmarkCohort,
  tokens: dict[str, str],
  analyzer: StatisticalAnalyzer,
  effective_baseline: str,
  gap_metric_key: str,
  alpha: float,
  decimal_separator: str,
) -> list[str]:
  """Emit paired Wilcoxon signed-rank + Holm-Bonferroni macros against baseline."""
  non_baseline = [alg for alg in cohort.algorithms if alg != effective_baseline]
  if effective_baseline not in cohort.algorithms or not non_baseline or len(cohort.problems) < 2:
    return []

  lines: list[str] = []
  for metric_tag, metric_key in (("Gap", gap_metric_key), ("Time", "mean_time")):
    paired_records: list[tuple[str, float, float, float]] = []
    for cand in non_baseline:
      paired_probs, base_vals, cand_vals = cohort.paired_metric_vectors(effective_baseline, cand, metric_key)
      if len(paired_probs) < 2:
        continue
      w_stat, p_val, r_eff = analyzer.wilcoxon_paired_summary(base_vals, cand_vals)
      paired_records.append((cand, w_stat, p_val, r_eff))

    if not paired_records:
      continue

    _, holm_p_vals = holm_bonferroni_correction([rec[2] for rec in paired_records], alpha=alpha)
    lines.extend(["", f"% Paired Wilcoxon + Holm-Bonferroni vs baseline ({metric_tag})"])
    for (cand, w_stat, p_val, r_eff), p_holm in zip(paired_records, holm_p_vals, strict=True):
      tok = tokens[cand]
      lines.extend(
        [
          f"\\newcommand{{\\wilcoxon{metric_tag}W{tok}}}{{{_format_decimal_latex(w_stat, 1, decimal_separator)}}}",
          f"\\newcommand{{\\wilcoxon{metric_tag}P{tok}}}{{{_format_pvalue_latex(p_val, decimal_separator)}}}",
          f"\\newcommand{{\\wilcoxon{metric_tag}PHolm{tok}}}{{{_format_pvalue_latex(p_holm, decimal_separator)}}}",
          f"\\newcommand{{\\wilcoxon{metric_tag}Effect{tok}}}{{{_format_decimal_latex(r_eff, 2, decimal_separator)}}}",
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
) -> Path:
  """Export strictly quantitative LaTeX ``\\newcommand`` macros to ``benchmark_macros.tex``."""
  if not all_results:
    raise ValueError("all_results dictionary is empty")

  effective_sep = (
    decimal_separator if decimal_separator is not None else (style.decimal_separator if style is not None else ".")
  )
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  macro_path = out_dir / filename

  active_cohort = cohort or BenchmarkCohort.from_results(
    all_results,
    problem_configs,
    algorithm_order=style.algorithm_order if style is not None else None,
  )
  effective_baseline = active_cohort.baseline(baseline_algorithm)
  tokens = _build_algorithm_macro_tokens(active_cohort.algorithms)
  rep_counts = active_cohort.repetition_counts()

  lines: list[str] = [
    "% Auto-generated quantitative benchmark macros (`DP14-5`)",
    "% Strictly numeric values formatted with configurable decimal separator.",
    f"\\newcommand{{\\benchmarkNumProblems}}{{{len(active_cohort.problems)}}}",
    f"\\newcommand{{\\benchmarkNumAlgorithms}}{{{len(active_cohort.algorithms)}}}",
    f"\\newcommand{{\\benchmarkBaselineAlgorithm}}{{{escape_latex_identifier(effective_baseline)}}}",
    f"\\newcommand{{\\benchmarkTotalRuns}}{{{sum(rep_counts)}}}",
  ]
  if rep_counts:
    min_rep, max_rep = min(rep_counts), max(rep_counts)
    rep_macro = str(min_rep) if min_rep == max_rep else f"{min_rep}--{max_rep}"
    lines.append(f"\\newcommand{{\\benchmarkRepetitions}}{{{rep_macro}}}")

  analyzer = StatisticalAnalyzer(alpha=alpha)
  gap_metric_key = active_cohort.quality_metric_key()

  lines.extend(_emit_per_algorithm_macros(active_cohort, tokens, effective_baseline, effective_sep))
  lines.extend(_emit_pairwise_speedup_macros(active_cohort, tokens, effective_sep))
  lines.extend(_emit_omnibus_macros(active_cohort, tokens, analyzer, gap_metric_key, alpha, effective_sep))
  lines.extend(
    _emit_wilcoxon_macros(active_cohort, tokens, analyzer, effective_baseline, gap_metric_key, alpha, effective_sep)
  )

  macro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return macro_path


__all__ = ["generate_latex_macros"]
