#!/usr/bin/env python3
"""
Chapter 4 Validation Benchmark Runner
=====================================

Executes the iso-algorithmic four-variant comparison of Chapter 4 by delegating
argument parsing and execution orchestration to ``src/cli/benchmark.py`` while
keeping dissertation-specific protocol constants, comparison-contract checks,
Brazilian Portuguese figure labels, and joint statistical summaries isolated
outside ``src/`` (`DP13-8`, `DP14-2`, `DP14-3`).

Variants compared under identical GA parameters (`population_size=256`,
`mutation_rate=0.02`, `seed=42`, `selection.k=5`, `local_search.max_iterations=10`,
`local_search.candidate_k=20`, `local_search_stage="post_mutation"`):

    slot          config variant      execution strategy
    ------------  ------------------  -------------------------------------------------
    cpu           CPU                 NumPy everywhere; `LocalSearchCPUTwoOpt`
    hybridnaive   HybridNaive         per-route GPU 2-opt; CPU evaluation
    hybrid        HybridOptimized     batched GPU 2-opt; GPU evaluation
    fullgpu       FullGPUModular      every operator on GPU (per-phase kernels)

Checkpoint repetition contract (`--incremental` vs `--no-incremental`)
---------------------------------------------------------------------
Because ``INCREMENTAL_RUNS = False`` by default in this thesis script, invoking
without either flag executes the ``--no-incremental`` branch:

    flag               incremental_runs   behavior when checkpoint has ``existing_runs``
    -----------------  ------------------ ------------------------------------------------
    --incremental      True               Appends a full new block of ``repetitions``
                                          runs (target = ``existing_runs + repetitions``;
                                          e.g. running again with ``r=30`` yields 60).
    (default) /        False              Completes only missing repetitions up to
    --no-incremental                      ``repetitions`` (``repetitions - existing_runs``),
                                          or skips/reuses if ``existing_runs >= repetitions``.

Usage:
    python -m benchmark_scripts.run_chapter4_benchmark
    python -m benchmark_scripts.run_chapter4_benchmark --no-incremental -r 30
    python -m benchmark_scripts.run_chapter4_benchmark --dry-run
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarking.analysis.aggregate_statistics import compare_algorithms
from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.analysis.report_style import load_report_style
from benchmarking.analysis.scaling_plots import (
  generate_scaling_plots,
  generate_scaling_plots_from_checkpoints,
)
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.runner.algorithm_runner import adaptive_generations
from benchmarking.runner.config_loader import load_problems_config
from cli import benchmark as benchmark_cli

if TYPE_CHECKING:
  import argparse
  from collections.abc import Mapping, Sequence

  from benchmarking.runner.config_loader import AlgorithmConfig, GAParams, ProblemConfig
  from protocols.types import LocalSearchStage

# =============================================================================
# Experiment declaration (thesis-specific presets isolated from `src/`)
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "benchmark_scripts" / "configs" / "chapter4"
REPORT_STYLE_PATH: Path = CONFIG_DIR / "report_style.json"
DB_PATH: Path = PROJECT_ROOT / "datasets" / "tsp.duckdb"
FALLBACK_TSP_DIR: Path = PROJECT_ROOT / "datasets" / "tsp"
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v4"
BENCHMARK_NAME: str = "chapter4_validation"
BASELINE_ALGORITHM: str = "CPU"

#: Thesis table slots -> active config variant names.
COMPARISON_SLOTS: dict[str, str] = {
  "cpu": "CPU",
  "hybridnaive": "HybridNaive",
  "hybrid": "HybridOptimized",
  "fullgpu": "FullGPUModular",
}
ALGORITHMS: tuple[str, ...] = tuple(COMPARISON_SLOTS.values())

#: Chapter 4 evaluation cohort spanning all 15 TSPLIB instances.
PROBLEMS: tuple[str, ...] = (
  "berlin52",
  "eil51",
  "att48",
  "gr96",
  "kroA100",
  "ch150",
  "d198",
  "pr264",
  "lin318",
  "pcb442",
  "att532",
  "si535",
  "gr666",
  "rat783",
  "pr1002",
)

CANDIDATE_K: int = 20
LOCAL_SEARCH_MAX_ITERATIONS: int = 10
SELECTION_K: int = 5
SEED: int = 42
POPULATION_SIZE: int = 256
MUTATION_RATE: float = 0.02
REPETITIONS: int = 30
INCREMENTAL_RUNS: bool = False
LOCAL_SEARCH_STAGE: LocalSearchStage = "post_mutation"

# =============================================================================
# Comparison contract (thesis iso-algorithmic guarantees)
# =============================================================================

STRATEGY_SLOTS: tuple[str, ...] = (
  "initialization",
  "selection",
  "crossover",
  "mutation",
  "local_search",
  "evaluation",
)

#: Iso-algorithmic parameter values documented in the thesis.
EXPECTED_GA_PARAMS: dict[str, float] = {"population_size": 256.0, "mutation_rate": 0.02, "seed": 42.0}

#: Operator composition that defines each variant's execution backend (`DP11-1` naming).
EXPECTED_STRATEGIES: dict[str, dict[str, str]] = {
  "CPU": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchCPUTwoOpt",
    "evaluation": "EvaluationCPUCost",
  },
  "HybridNaive": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationCPUCost",
  },
  "HybridOptimized": {
    "initialization": "InitializationRandom",
    "selection": "SelectionTournament",
    "crossover": "CrossoverOrder",
    "mutation": "MutationSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationGPUCost",
  },
  "FullGPUModular": {
    "initialization": "InitializationRandom",
    "selection": "SelectionGPUTournament",
    "crossover": "CrossoverGPUOrder",
    "mutation": "MutationGPUSwap",
    "local_search": "LocalSearchGPUTwoOpt",
    "evaluation": "EvaluationGPUCost",
  },
}

#: Strategy parameters that must be identical across the four variants.
EXPECTED_STRATEGY_PARAMS: tuple[tuple[str, str, float], ...] = (
  ("selection", "k", 5.0),
  ("local_search", "max_iterations", 10.0),
  ("local_search", "candidate_k", 20.0),
)

#: `LocalSearchGPUTwoOpt.mode` selects the transfer strategy; `CPU` runs the vectorised best-improvement loop.
EXPECTED_LOCAL_SEARCH_MODE: dict[str, str] = {
  "CPU": "best",
  "HybridNaive": "naive",
  "HybridOptimized": "batch",
  "FullGPUModular": "batch",
}

#: `return_gpu` for (initialization, evaluation); only the full-GPU variant keeps routes in VRAM.
EXPECTED_RETURN_GPU: dict[str, tuple[bool, bool]] = {
  "CPU": (False, False),
  "HybridNaive": (False, False),
  "HybridOptimized": (False, False),
  "FullGPUModular": (True, True),
}

_CHAPTER4_HELP_EPILOG: str = (
  "Chapter 4 script defaults (overridable via the flags above):\n"
  f"  --config-dir          {CONFIG_DIR}\n"
  f"  --output-dir          {OUTPUT_DIR}\n"
  f"  --benchmark-name      {BENCHMARK_NAME}\n"
  f"  --baseline-algorithm  {BASELINE_ALGORITHM}\n"
  f"  --repetitions         {REPETITIONS}\n"
  f"  --incremental         {INCREMENTAL_RUNS} (use --no-incremental to complete only missing runs)\n"
  f"  --problems            {len(PROBLEMS)} instances ({PROBLEMS[0]} .. {PROBLEMS[-1]})\n"
  f"  --algorithms          {', '.join(ALGORITHMS)}\n"
)


def _strategy_names(config: AlgorithmConfig) -> dict[str, str]:
  """Return `{slot: class name}` for a declarative variant configuration."""
  return {slot: str(config.strategies.get(slot, {}).get("class", "None")) for slot in STRATEGY_SLOTS}


def validate_comparison_contract(
  algorithms: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
) -> list[str]:
  """Check that `benchmark_scripts/configs/chapter4/` still encodes the iso-algorithmic comparison."""
  errors: list[str] = []

  for slot, variant in COMPARISON_SLOTS.items():
    if variant not in algorithms:
      errors.append(f"slot '{slot}': variant '{variant}' missing from chapter4/algorithms.json")
      continue

    config = algorithms[variant]
    actual = _strategy_names(config)
    for strategy_slot, expected_class in EXPECTED_STRATEGIES[variant].items():
      if actual[strategy_slot] != expected_class:
        errors.append(
          f"slot '{slot}' ({variant}).{strategy_slot}: expected {expected_class}, found {actual[strategy_slot]}"
        )

    expected_use_gpu = variant != "CPU"
    if config.use_gpu != expected_use_gpu:
      errors.append(f"slot '{slot}' ({variant}).use_gpu: expected {expected_use_gpu}, found {config.use_gpu}")

    stage = str(config.params.get("local_search_stage", LOCAL_SEARCH_STAGE))
    if stage != LOCAL_SEARCH_STAGE:
      errors.append(f"slot '{slot}' ({variant}).local_search_stage: expected {LOCAL_SEARCH_STAGE}, found {stage}")

    for strategy_slot, key, expected_value in EXPECTED_STRATEGY_PARAMS:
      params = config.strategies.get(strategy_slot, {}).get("params", {})
      if key not in params:
        errors.append(f"slot '{slot}' ({variant}).{strategy_slot}.{key}: missing (expected {expected_value:g})")
      elif float(params[key]) != expected_value:
        errors.append(
          f"slot '{slot}' ({variant}).{strategy_slot}.{key}: expected {expected_value:g}, found {params[key]}"
        )

  overrides: dict[str, float | int | None] = {
    "population_size": POPULATION_SIZE,
    "mutation_rate": MUTATION_RATE,
    "seed": SEED,
  }
  actual_ga = {
    "population_size": float(ga_params.population_size),
    "mutation_rate": float(ga_params.mutation_rate),
    "seed": float(ga_params.seed),
  }
  for key, expected_value in EXPECTED_GA_PARAMS.items():
    if overrides[key] is not None:
      continue
    if actual_ga[key] != expected_value:
      errors.append(f"GA {key}: expected {expected_value:g}, found {actual_ga[key]:g} in chapter4/algorithms.json")

  return errors


def validate_effective_instances(
  algorithms: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
) -> list[str]:
  """Instantiate every variant and compare the values the run will actually use."""
  errors: list[str] = []
  expected_seed = SEED if SEED is not None else 42
  expected_population = POPULATION_SIZE if POPULATION_SIZE is not None else 256
  expected_mutation = MUTATION_RATE if MUTATION_RATE is not None else 0.02

  for slot, variant in COMPARISON_SLOTS.items():
    if variant not in algorithms:
      continue

    instance = algorithms[variant].instantiate(
      ga_params,
      seed=expected_seed,
      local_search_stage=LOCAL_SEARCH_STAGE,
    )

    effective: dict[str, Any] = {
      "population_size": getattr(instance, "population_size", None),
      "mutation_rate": getattr(instance, "mutation_rate", None),
      "seed": getattr(instance, "seed", None),
      "local_search_stage": getattr(instance, "local_search_stage", None),
    }
    expected: dict[str, Any] = {
      "population_size": expected_population,
      "mutation_rate": expected_mutation,
      "seed": expected_seed,
      "local_search_stage": LOCAL_SEARCH_STAGE,
    }
    for key, expected_value in expected.items():
      if effective[key] != expected_value:
        errors.append(f"slot '{slot}' ({variant}): effective {key}={effective[key]!r}, declared {expected_value!r}")

    for strategy_slot, expected_class in EXPECTED_STRATEGIES[variant].items():
      strategy = getattr(instance, strategy_slot, None)
      actual_class = type(strategy).__name__ if strategy is not None else "None"
      if actual_class != expected_class:
        errors.append(f"slot '{slot}' ({variant}): effective {strategy_slot}={actual_class}, declared {expected_class}")

    for strategy_slot, key, expected_value in EXPECTED_STRATEGY_PARAMS:
      strategy = getattr(instance, strategy_slot, None)
      actual_value = getattr(strategy, key, None)
      if actual_value != expected_value:
        errors.append(
          f"slot '{slot}' ({variant}): effective {strategy_slot}.{key}={actual_value!r}, declared {expected_value!r}"
        )

    expected_init_gpu, expected_eval_gpu = EXPECTED_RETURN_GPU[variant]
    for strategy_slot, expected_gpu in (("initialization", expected_init_gpu), ("evaluation", expected_eval_gpu)):
      strategy = getattr(instance, strategy_slot, None)
      if strategy is None or not hasattr(strategy, "return_gpu"):
        continue
      actual_gpu = strategy.return_gpu
      if actual_gpu != expected_gpu:
        errors.append(
          f"slot '{slot}' ({variant}): effective {strategy_slot}.return_gpu={actual_gpu!r}, declared {expected_gpu!r}"
        )

    local_search = getattr(instance, "local_search", None)
    expected_mode = EXPECTED_LOCAL_SEARCH_MODE[variant]
    actual_mode = getattr(local_search, "mode", None)
    if actual_mode != expected_mode:
      errors.append(
        f"slot '{slot}' ({variant}): effective local_search.mode={actual_mode!r}, declared {expected_mode!r}"
      )

  return errors


def validate_generation_budget(
  problems: Sequence[ProblemConfig],
  target_problem_names: Sequence[str] = PROBLEMS,
) -> list[str]:
  """Confirm the pipeline's generation budget matches the documented `2 * n * sqrt(n)` formula."""
  errors: list[str] = []
  target_set = set(target_problem_names)
  for problem in problems:
    if problem.name not in target_set:
      continue
    expected = int(2 * problem.size * math.sqrt(problem.size))
    actual = adaptive_generations(problem.size)
    if actual != expected:
      errors.append(f"problem '{problem.name}': adaptive_generations({problem.size})={actual}, expected {expected}")
  return errors


def _chapter4_preflight_validator(
  algorithms: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
  problems: Sequence[ProblemConfig],
) -> list[str]:
  """Run thesis-specific iso-algorithmic contract and budget validations."""
  target_names = [p.name for p in problems]
  return [
    *validate_comparison_contract(algorithms, ga_params),
    *validate_effective_instances(algorithms, ga_params),
    *validate_generation_budget(problems, target_names),
  ]


def generate_thesis_scaling_plots(
  *,
  output_dir: Path | str,
  results: dict[str, dict[str, dict[str, Any]]] | None = None,
  checkpoint_dir: Path | str | None = None,
  problem_configs: Sequence[ProblemConfig] | None = None,
  filename_stem: str = "scaling_time_gap",
) -> tuple[Path, ...]:
  """Emit ABNT-sized vector `.pdf` and `.svg` scaling figures styled from `report_style.json`."""
  if problem_configs is None and CONFIG_DIR.exists():
    problem_configs = load_problems_config(CONFIG_DIR)

  thesis_style = load_report_style(REPORT_STYLE_PATH)
  plot_kwargs: dict[str, Any] = {
    "style": thesis_style,
    "formats": ("pdf", "svg"),
    "dispersion_metric": "iqr",
    "x_axis_mode": "continuous",
    "x_tick_mode": "instances",
  }
  if results is not None:
    return generate_scaling_plots(
      all_results=results,
      problem_configs=problem_configs,
      output_dir=output_dir,
      filename_stem=filename_stem,
      **plot_kwargs,
    )
  resolved_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else (OUTPUT_DIR / "checkpoints")
  return generate_scaling_plots_from_checkpoints(
    checkpoint_dir=resolved_ckpt_dir,
    output_dir=output_dir,
    problem_configs=problem_configs,
    filename_stem=filename_stem,
    **plot_kwargs,
  )


def log_comparison(
  results: Mapping[str, Mapping[str, dict[str, Any]]],
  *,
  baseline_algorithm: str = BASELINE_ALGORITHM,
) -> None:
  """Log per-problem summary table and joint multi-metric thesis synthesis."""
  if not results:
    return

  logging.info("")
  logging.info("=" * 80)
  logging.info("CHAPTER 4 COMPARISON (mean gap and mean time per problem)")
  logging.info("=" * 80)
  header = f"{'problem':10s} " + " ".join(f"{slot:>22s}" for slot in COMPARISON_SLOTS)
  logging.info(header)
  logging.info("-" * len(header))

  for problem, algorithms in results.items():
    cells: list[str] = []
    for variant in ALGORITHMS:
      record = algorithms.get(variant)
      if record is None:
        cells.append(f"{'(skipped)':>22s}")
      else:
        gap = float(record.get("mean_gap", float("nan")))
        elapsed = float(record.get("mean_time", float("nan")))
        cells.append(f"{gap:8.2f}% {elapsed:9.1f}s".rjust(22))
    logging.info(f"{problem:10s} " + " ".join(cells))

  logging.info("-" * len(header))
  logging.info("'(skipped)' marks an algorithm that did not run for that instance.")

  # Multi-metric cross-problem summary & joint interpretive synthesis (Task 14.2)
  present_variants = [v for v in ALGORITHMS if any(v in prob_res for prob_res in results.values())]
  all_results_dict: dict[str, dict[str, dict[str, Any]]] = {prob: dict(alg_map) for prob, alg_map in results.items()}
  problem_list = list(all_results_dict.keys())
  analyzer = StatisticalAnalyzer()

  gap_comp = compare_algorithms(
    all_results=all_results_dict,
    problems=problem_list,
    algorithms=present_variants,
    analyzer=analyzer,
    baseline_algorithm=None,
    label="CHAPTER 4 SYNTHESIS — QUALITY (mean_gap)",
    metric_key="mean_gap",
    higher_is_better=False,
  )
  time_comp = compare_algorithms(
    all_results=all_results_dict,
    problems=problem_list,
    algorithms=present_variants,
    analyzer=analyzer,
    baseline_algorithm=baseline_algorithm if baseline_algorithm in present_variants else None,
    label="CHAPTER 4 SYNTHESIS — RUNTIME & SPEEDUP (mean_time)",
    metric_key="mean_time",
    higher_is_better=False,
  )

  logging.info("=" * 80)
  logging.info("CHAPTER 4 JOINT INTERPRETIVE SUMMARY (RUNTIME vs. SOLUTION QUALITY)")
  logging.info("=" * 80)

  if baseline_algorithm in present_variants:
    for variant in present_variants:
      paired_ratios: list[float] = []
      for prob_res in all_results_dict.values():
        if baseline_algorithm in prob_res and variant in prob_res:
          base_t = float(prob_res[baseline_algorithm].get("mean_time", 0.0))
          cand_t = float(prob_res[variant].get("mean_time", 0.0))
          if base_t > 0.0 and cand_t > 0.0:
            paired_ratios.append(base_t / cand_t)
      if paired_ratios:
        geomean = float(np.exp(np.mean(np.log(paired_ratios))))
        logging.info(
          "  %-18s geomean speedup vs %-4s: %6.2fx (n=%d)",
          variant,
          baseline_algorithm,
          geomean,
          len(paired_ratios),
        )

  if gap_comp is not None and time_comp is not None:
    gap_f = gap_comp["friedman_test"]
    time_f = time_comp["friedman_test"]
    gap_p = float(gap_f["p_value_f"])
    time_p = float(time_f["p_value_f"])
    speedups = time_comp.get("speedups", {})
    time_ranks = time_comp.get("mean_ranks", {})
    if speedups:
      fastest_variant = max(speedups.items(), key=lambda kv: kv[1])[0]
      fastest_gain: float | None = float(speedups[fastest_variant])
    elif time_ranks:
      fastest_variant = min(time_ranks.items(), key=lambda kv: kv[1])[0]
      fastest_gain = None
    else:
      fastest_variant = present_variants[0]
      fastest_gain = None

    logging.info(
      "  Quality (mean_gap):   Friedman F_F(%d, %d) = %.4f, p = %.6f (%s)",
      gap_f["df1"],
      gap_f["df2"],
      gap_f["statistic_f"],
      gap_p,
      "statistically equivalent across variants" if not gap_f["significant"] else "significant quality divergence",
    )
    logging.info(
      "  Runtime (mean_time):  Friedman F_F(%d, %d) = %.4f, p = %.6f (%s)",
      time_f["df1"],
      time_f["df2"],
      time_f["statistic_f"],
      time_p,
      "significant acceleration across variants" if time_f["significant"] else "no significant runtime divergence",
    )
    if time_f["significant"] and not gap_f["significant"]:
      if fastest_gain is not None:
        logging.info(
          "  Joint verdict: %s achieves %.2fx geometric speedup over %s with statistically "
          "indistinguishable solution quality (p_gap=%.4f >= 0.05, p_time=%.4g < 0.05).",
          fastest_variant,
          fastest_gain,
          baseline_algorithm,
          gap_p,
          time_p,
        )
      else:
        logging.info(
          "  Joint verdict: %s ranks fastest (mean_rank=%.2f; baseline %s omitted) with statistically "
          "indistinguishable solution quality (p_gap=%.4f >= 0.05, p_time=%.4g < 0.05).",
          fastest_variant,
          time_ranks.get(fastest_variant, 1.0),
          baseline_algorithm,
          gap_p,
          time_p,
        )
    elif fastest_gain is not None:
      logging.info(
        "  Joint verdict: Fastest variant %s (%.2fx vs %s); p_gap=%.4f, p_time=%.4g.",
        fastest_variant,
        fastest_gain,
        baseline_algorithm,
        gap_p,
        time_p,
      )
    else:
      logging.info(
        "  Joint verdict: Fastest variant %s (mean_rank=%.2f; baseline %s omitted); p_gap=%.4f, p_time=%.4g.",
        fastest_variant,
        time_ranks.get(fastest_variant, 1.0),
        baseline_algorithm,
        gap_p,
        time_p,
      )
  logging.info("=" * 80)


def build_chapter4_argv(user_argv: Sequence[str] | None = None) -> list[str]:
  """Construct canonical `cli.benchmark` arguments with thesis defaults plus user overrides."""
  extra = list(user_argv) if user_argv is not None else list(sys.argv[1:])
  has_explicit_tsp_files = any(arg == "--tsp-files" or arg.startswith("--tsp-files=") for arg in extra)
  has_explicit_problems = any(arg == "--problems" or arg.startswith("--problems=") for arg in extra)

  defaults: list[str] = [
    "--config-dir",
    str(CONFIG_DIR),
    "--output-dir",
    str(OUTPUT_DIR),
    "--tsp-dir",
    str(FALLBACK_TSP_DIR),
    "--algorithms",
    *ALGORITHMS,
    "--repetitions",
    str(REPETITIONS),
    "--benchmark-name",
    BENCHMARK_NAME,
    "--reports",
    "all",
    "--report-style",
    str(REPORT_STYLE_PATH),
    "--local-search-stage",
    LOCAL_SEARCH_STAGE,
    "--baseline-algorithm",
    BASELINE_ALGORITHM,
    "--seed",
    str(SEED),
    "--population-size",
    str(POPULATION_SIZE),
    "--mutation-rate",
    str(MUTATION_RATE),
    "--incremental" if INCREMENTAL_RUNS else "--no-incremental",
  ]
  if DB_PATH.exists():
    defaults.extend(["--db-path", str(DB_PATH)])
  if not has_explicit_tsp_files and not has_explicit_problems:
    if DB_PATH.exists():
      defaults.extend(["--problems", *PROBLEMS])
    else:
      defaults.extend(["--tsp-files", *(str(FALLBACK_TSP_DIR / f"{name}.tsp") for name in PROBLEMS)])

  return [*defaults, *extra]


def generate_thesis_cd_diagrams(
  *,
  output_dir: Path | str,
  results: Mapping[str, Mapping[str, dict[str, Any]]],
  benchmark_name: str = BENCHMARK_NAME,
) -> tuple[Path, ...]:
  """Emit pt-BR Demšar CD TikZ diagrams styled from `report_style.json` via `generate_result_tables`."""
  if not results:
    return ()
  problem_configs = load_problems_config(CONFIG_DIR) if CONFIG_DIR.exists() else []
  thesis_style = load_report_style(REPORT_STYLE_PATH)
  normalized = {p: {a: dict(rec) for a, rec in algs.items()} for p, algs in results.items()}
  emitted = generate_result_tables(
    all_results=normalized,
    problem_configs=problem_configs,
    output_dir=output_dir,
    benchmark_name=benchmark_name,
    baseline_algorithm=BASELINE_ALGORITHM,
    style=thesis_style,
    artifacts=("cd",),
  )
  return emitted.get("cd", ())


def _on_chapter4_complete(
  args: argparse.Namespace,
  results: dict[str, dict[str, dict[str, Any]]],
) -> None:
  """Log thesis joint comparison summary (pt-BR CD diagrams and scaling plots are rendered in one pass)."""
  log_comparison(results, baseline_algorithm=args.baseline_algorithm or BASELINE_ALGORITHM)


def run(
  *,
  repetitions: int = REPETITIONS,
  output_dir: Path = OUTPUT_DIR,
  benchmark_name: str = BENCHMARK_NAME,
  baseline_algorithm: str = BASELINE_ALGORITHM,
  skip_cpu: bool = False,
  incremental_runs: bool = INCREMENTAL_RUNS,
  problem_names: Sequence[str] | None = None,
  dry_run: bool = False,
) -> int:
  """Execute the Chapter 4 comparison by delegating to `cli.benchmark.main`."""
  argv: list[str] = [
    "--repetitions",
    str(repetitions),
    "--output-dir",
    str(output_dir),
    "--benchmark-name",
    benchmark_name,
    "--baseline-algorithm",
    baseline_algorithm,
    "--incremental" if incremental_runs else "--no-incremental",
  ]
  if skip_cpu:
    argv.append("--skip-cpu")
  if problem_names:
    argv.extend(["--problems", *problem_names])
  if dry_run:
    argv.append("--dry-run")
  return main(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Execute the Chapter 4 validation benchmark via the canonical `cli.benchmark` entrypoint."""
  combined_argv = build_chapter4_argv(argv)
  return benchmark_cli.main(
    combined_argv,
    preflight_validator=_chapter4_preflight_validator,
    on_complete=_on_chapter4_complete,
    parser_epilog=_CHAPTER4_HELP_EPILOG,
  )


if __name__ == "__main__":
  sys.exit(main())
