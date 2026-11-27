"""
Orchestration Module for Benchmark V2
======================================

Main benchmark loop orchestrating configuration resolution, preflight validation,
incremental checkpoint resume, per-problem statistical analysis, cohort analysis,
and report generation.
"""

import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

from benchmarking.analysis.cohort_analysis import analyze_cohort
from benchmarking.analysis.problem_statistics import analyze_problem
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.reporting.console import log_cohort_analysis, log_problem_analysis
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.reporting.style import ReportArtifact, ReportStyle
from benchmarking.runner.checkpoint_io import (
  CheckpointManager,
  validate_existing_checkpoints,
)
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  BenchmarkConfig,
  GAParams,
  ProblemConfig,
  resolve_benchmark_configs,
  validate_configs,
)
from benchmarking.runner.orchestration_helpers import (
  ResumePlan,
  load_problem_and_optimal,
  plan_checkpoint_resume,
  resolve_problem_selection,
)
from benchmarking.runner.orchestration_helpers import serialize_custom_pipeline as _serialize_custom_pipeline
from benchmarking.runner.orchestration_runtime import (
  build_checkpoint_static_metadata,
  finalize_cohort_and_reports,
  resolve_effective_runtime_configs,
  run_single_problem_phase,
)
from protocols.backend import cupy_backend, gpu_available


def _release_gpu_memory() -> None:
  """Return cached CuPy memory blocks to the driver; a no-op without a CUDA device."""
  cp = cupy_backend()
  if cp is not None:
    cp.get_default_memory_pool().free_all_blocks()


def _log_benchmark_configuration(
  repetitions: int,
  selected_algorithms: list[str],
  problem_configs: list[ProblemConfig],
  ga_params: GAParams,
  benchmark_config: BenchmarkConfig,
  benchmark_name: str = "benchmark",
) -> None:
  """Log benchmark configuration at startup."""
  banner_title = benchmark_name.replace("_", " ").upper()
  logging.info("=" * 80)
  logging.info(f"{banner_title}: GPU-ACCELERATED GENETIC ALGORITHM BENCHMARK")
  logging.info("=" * 80)
  logging.info("")

  logging.info("Configuration:")
  logging.info(f"  Algorithms: {', '.join(selected_algorithms)}")
  logging.info(f"  Problems: {len(problem_configs)} routing instances")
  logging.info(f"  Repetitions: {repetitions}")
  if benchmark_config.cpu_size_threshold is None:
    logging.info("  CPU threshold: all instances (no cutoff)")
  else:
    logging.info(f"  CPU threshold: n ≤ {benchmark_config.cpu_size_threshold}")
  logging.info("")

  logging.info("GA Parameters:")
  logging.info(f"  Population size: {ga_params.population_size}")
  logging.info(f"  Mutation rate: {ga_params.mutation_rate}")
  logging.info(f"  Seed: {ga_params.seed}")
  if benchmark_config.patience is None:
    logging.info("  Early stop patience: Adaptive (2x√n generations)")
  else:
    logging.info(f"  Early stop patience: {benchmark_config.patience} generations")
  if benchmark_config.max_generations is None:
    logging.info("  Max generations: Adaptive (2*n*√n generations)")
  else:
    logging.info(f"  Max generations: {benchmark_config.max_generations} (fixed)")
  logging.info("")


def _determine_problem_algorithms(
  problem_size: int,
  selected_algorithms: list[str],
  algorithm_configs: Mapping[str, AlgorithmConfig],
  cpu_threshold: int | None,
) -> list[str]:
  """Determine which algorithms to run for a problem based on size."""
  if cpu_threshold is None or problem_size <= cpu_threshold:
    return selected_algorithms
  return [alg for alg in selected_algorithms if algorithm_configs[alg].use_gpu]


def validate_gpu_backend_for_algorithms(
  selected_algorithms: Sequence[str],
  algorithm_configs: Mapping[str, AlgorithmConfig],
) -> list[str]:
  """Confirm a functional CUDA/CuPy backend when any selected algorithm requires GPU."""
  requires_gpu = any(algorithm_configs[alg].use_gpu for alg in selected_algorithms if alg in algorithm_configs)
  if not requires_gpu:
    return []

  errors: list[str] = []
  if not gpu_available():
    errors.append("no usable CUDA device (`protocols.backend.gpu_available()` is False)")
  if cupy_backend() is None:
    errors.append("`protocols.backend.cupy_backend()` returned None; GPU variants have no backend")
  return errors


def _run_preflight(
  *,
  effective_ga_params: GAParams,
  algorithm_configs: dict[str, AlgorithmConfig],
  effective_benchmark_config: BenchmarkConfig,
  problem_configs: Sequence[ProblemConfig],
  selected_algorithms: Sequence[str],
  checkpoint_dir: Path,
  ga_params: GAParams,
  param_overrides: Mapping[str, Any] | None,
  effective_seed: int,
  local_search_stage: str | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  allow_corrupt_checkpoints: bool = False,
  preflight_validator: Callable[
    [Mapping[str, AlgorithmConfig], GAParams, Sequence[ProblemConfig]],
    Sequence[str],
  ]
  | None = None,
) -> None:
  """Validate effective configs, GPU backend availability, and existing checkpoints."""
  errors: list[str] = validate_configs(
    {
      "ga_params": effective_ga_params,
      "algorithms": algorithm_configs,
      "benchmark": effective_benchmark_config,
      "problems": list(problem_configs),
    }
  )
  if not selected_algorithms:
    errors.append("No algorithms selected for benchmark execution")
  if preflight_validator is not None:
    errors.extend(preflight_validator(algorithm_configs, ga_params, problem_configs))
  errors.extend(validate_gpu_backend_for_algorithms(selected_algorithms, algorithm_configs))
  errors.extend(
    validate_existing_checkpoints(
      checkpoint_dir,
      problem_configs,
      selected_algorithms,
      algorithm_configs,
      ga_params,
      param_overrides=param_overrides,
      seed_override=effective_seed,
      cpu_size_threshold=effective_benchmark_config.cpu_size_threshold,
      max_generations=effective_benchmark_config.max_generations,
      local_search_stage=local_search_stage,
      strategy_overrides=strategy_overrides,
      allow_corrupt=allow_corrupt_checkpoints,
    )
  )
  if errors:
    for err in errors:
      logging.error(f"Preflight error: {err}")
    raise ValueError("Benchmark preflight validation failed:\n" + "\n".join(f"  - {err}" for err in errors))


def _resolve_selected_algorithms(
  algorithms: Sequence[str] | None,
  algorithm_configs: Mapping[str, AlgorithmConfig],
  *,
  skip_cpu: bool,
) -> list[str]:
  """Validate requested algorithm names and filter CPU entries when `skip_cpu=True`."""
  candidate_algorithms: list[str] = list(algorithms) if algorithms else list(algorithm_configs.keys())
  selected: list[str] = []
  for alg_name in candidate_algorithms:
    if alg_name not in algorithm_configs:
      raise ValueError(f"Unknown algorithm '{alg_name}'. Available: {list(algorithm_configs.keys())}")
    if skip_cpu and not algorithm_configs[alg_name].use_gpu:
      continue
    selected.append(alg_name)
  return selected


@dataclass(frozen=True, slots=True)
class _BenchmarkSession:
  """Resolved configurations, problem selection, and checkpoint state for one benchmark run."""

  ga_params: GAParams
  algorithm_configs: dict[str, AlgorithmConfig]
  selected_algorithms: list[str]
  effective_benchmark_config: BenchmarkConfig
  effective_seed: int
  param_overrides: dict[str, Any]
  resolved_problems: list[ProblemConfig]
  tsp_path_map: dict[str, Path]
  has_db: bool
  resolved_db_path: Path | None
  has_tsp_dir: bool
  resolved_tsp_dir: Path | None
  resolved_output_dir: Path
  checkpoint_manager: CheckpointManager


def _prepare_benchmark_session(
  *,
  output_dir: str | Path,
  config_dir: str | Path | None,
  algorithms_config: str | Path | None,
  benchmark_config: str | Path | None,
  problems_config: str | Path | None,
  tsp_files: Sequence[str | Path] | None,
  problem_names: Sequence[str] | None,
  db_path: str | Path | None,
  tsp_dir: str | Path | None,
  problem_configs: list[ProblemConfig] | None,
  algorithms: Sequence[str] | None,
  algorithm_configs_override: dict[str, AlgorithmConfig] | None,
  repetitions: int | None,
  max_generations: int | None,
  benchmark_name: str,
  local_search_stage: str | None,
  skip_cpu: bool,
  incremental_runs: bool | None,
  cpu_size_threshold: int | None,
  test_mode: bool,
  patience: int | None,
  seed: int | None,
  population_size: int | None,
  mutation_rate: float | None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
  dry_run: bool,
  preflight_validator: Callable[
    [Mapping[str, AlgorithmConfig], GAParams, Sequence[ProblemConfig]],
    Sequence[str],
  ]
  | None,
) -> _BenchmarkSession | None:
  """Resolve configs, validate preflight checks, and initialize checkpoints (returns `None` on dry run)."""
  setup_benchmark_logging(output_dir, benchmark_name, log_level=None)
  logging.info("Loading configurations...")

  configs = resolve_benchmark_configs(
    config_dir=config_dir,
    algorithms_config=algorithms_config,
    benchmark_config=benchmark_config,
    problems_config=problems_config,
    require_algorithms=algorithm_configs_override is None,
  )
  (
    ga_params,
    algorithm_configs,
    default_problems,
    selected_algorithms,
    effective_ga_params,
    effective_benchmark_config,
    effective_seed,
    param_overrides,
  ) = resolve_effective_runtime_configs(
    configs=configs,
    algorithm_configs_override=algorithm_configs_override,
    algorithms=algorithms,
    skip_cpu=skip_cpu,
    repetitions=repetitions,
    patience=patience,
    max_generations=max_generations,
    incremental_runs=incremental_runs,
    cpu_size_threshold=cpu_size_threshold,
    seed=seed,
    population_size=population_size,
    mutation_rate=mutation_rate,
  )

  resolved_db_path = Path(db_path) if db_path is not None else None
  resolved_tsp_dir = Path(tsp_dir) if tsp_dir is not None else None
  has_db = resolved_db_path is not None and resolved_db_path.exists()
  has_tsp_dir = resolved_tsp_dir is not None and resolved_tsp_dir.exists()

  resolved_problems, tsp_path_map = resolve_problem_selection(
    problem_configs=problem_configs,
    tsp_files=tsp_files,
    problem_names=problem_names,
    default_problems=default_problems,
    resolved_tsp_dir=resolved_tsp_dir,
    resolved_db_path=resolved_db_path,
    has_tsp_dir=has_tsp_dir,
    has_db=has_db,
  )
  if test_mode and not tsp_files and resolved_problems:
    resolved_problems = resolved_problems[:2]
    logging.info(f"TEST MODE ENABLED: running first {len(resolved_problems)} problems only")

  resolved_output_dir = Path(output_dir)
  _run_preflight(
    effective_ga_params=effective_ga_params,
    algorithm_configs=algorithm_configs,
    effective_benchmark_config=effective_benchmark_config,
    problem_configs=resolved_problems,
    selected_algorithms=selected_algorithms,
    checkpoint_dir=resolved_output_dir / "checkpoints",
    ga_params=ga_params,
    param_overrides=param_overrides or None,
    effective_seed=effective_seed,
    local_search_stage=local_search_stage,
    strategy_overrides=strategy_overrides,
    allow_corrupt_checkpoints=not dry_run,
    preflight_validator=preflight_validator,
  )
  _log_benchmark_configuration(
    effective_benchmark_config.repetitions,
    selected_algorithms,
    resolved_problems,
    effective_ga_params,
    effective_benchmark_config,
    benchmark_name=str(benchmark_name),
  )
  if dry_run:
    logging.info("Dry-run requested: preflight validation passed. Exiting without benchmark execution.")
    return None

  checkpoint_manager = CheckpointManager(
    checkpoint_dir=resolved_output_dir / "checkpoints",
    problem_stats_dir=resolved_output_dir / "problem_statistics",
  )
  completed, total = checkpoint_manager.count_completed_checkpoints(
    [p.__dict__ for p in resolved_problems],
    selected_algorithms,
    effective_benchmark_config.repetitions,
    effective_benchmark_config.cpu_size_threshold,
    algorithm_configs=algorithm_configs,
  )
  if completed > 0:
    logging.info(f"\n✓ Found {completed}/{total} completed checkpoints; resuming...")
  if not tsp_path_map and not has_db and not has_tsp_dir:
    raise FileNotFoundError(f"Neither TSPLIB directory ({resolved_tsp_dir}) nor database ({resolved_db_path}) found")

  return _BenchmarkSession(
    ga_params=ga_params,
    algorithm_configs=algorithm_configs,
    selected_algorithms=selected_algorithms,
    effective_benchmark_config=effective_benchmark_config,
    effective_seed=effective_seed,
    param_overrides=param_overrides,
    resolved_problems=resolved_problems,
    tsp_path_map=tsp_path_map,
    has_db=has_db,
    resolved_db_path=resolved_db_path,
    has_tsp_dir=has_tsp_dir,
    resolved_tsp_dir=resolved_tsp_dir,
    resolved_output_dir=resolved_output_dir,
    checkpoint_manager=checkpoint_manager,
  )


def run_comprehensive_benchmark(
  *,
  output_dir: str | Path,
  config_dir: str | Path | None = None,
  algorithms_config: str | Path | None = None,
  benchmark_config: str | Path | None = None,
  problems_config: str | Path | None = None,
  tsp_files: Sequence[str | Path] | None = None,
  problem_names: Sequence[str] | None = None,
  db_path: str | Path | None = None,
  tsp_dir: str | Path | None = None,
  problem_configs: list[ProblemConfig] | None = None,
  algorithms: Sequence[str] | None = None,
  algorithm_configs_override: dict[str, AlgorithmConfig] | None = None,
  repetitions: int | None = None,
  max_generations: int | None = None,
  benchmark_name: str = "benchmark",
  local_search_stage: str | None = None,
  skip_cpu: bool = False,
  incremental_runs: bool | None = None,
  cpu_size_threshold: int | None = None,
  test_mode: bool = False,
  patience: int | None = None,
  seed: int | None = None,
  population_size: int | None = None,
  mutation_rate: float | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  baseline_algorithm: str | None = None,
  reports: Sequence[ReportArtifact] = (),
  report_style: ReportStyle | None = None,
  include_size_category_table: bool = False,
  dry_run: bool = False,
  preflight_validator: Callable[
    [Mapping[str, AlgorithmConfig], GAParams, Sequence[ProblemConfig]],
    Sequence[str],
  ]
  | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
  """Run comprehensive benchmark on configured problems and algorithms."""
  session = _prepare_benchmark_session(
    output_dir=output_dir,
    config_dir=config_dir,
    algorithms_config=algorithms_config,
    benchmark_config=benchmark_config,
    problems_config=problems_config,
    tsp_files=tsp_files,
    problem_names=problem_names,
    db_path=db_path,
    tsp_dir=tsp_dir,
    problem_configs=problem_configs,
    algorithms=algorithms,
    algorithm_configs_override=algorithm_configs_override,
    repetitions=repetitions,
    max_generations=max_generations,
    benchmark_name=benchmark_name,
    local_search_stage=local_search_stage,
    skip_cpu=skip_cpu,
    incremental_runs=incremental_runs,
    cpu_size_threshold=cpu_size_threshold,
    test_mode=test_mode,
    patience=patience,
    seed=seed,
    population_size=population_size,
    mutation_rate=mutation_rate,
    strategy_overrides=strategy_overrides,
    dry_run=dry_run,
    preflight_validator=preflight_validator,
  )
  if session is None:
    return {}

  analyzer = StatisticalAnalyzer()
  problem_results: dict[str, dict[str, dict[str, Any]]] = {}
  for prob_idx, problem_config in enumerate(session.resolved_problems, 1):
    per_algo_results = run_single_problem_phase(
      prob_idx=prob_idx,
      total_problems=len(session.resolved_problems),
      problem_config=problem_config,
      selected_algorithms=session.selected_algorithms,
      algorithm_configs=session.algorithm_configs,
      effective_benchmark_config=session.effective_benchmark_config,
      effective_seed=session.effective_seed,
      ga_params=session.ga_params,
      local_search_stage=local_search_stage,
      param_overrides=session.param_overrides,
      strategy_overrides=strategy_overrides,
      tsp_path_map=session.tsp_path_map,
      has_db=session.has_db,
      resolved_db_path=session.resolved_db_path,
      has_tsp_dir=session.has_tsp_dir,
      resolved_tsp_dir=session.resolved_tsp_dir,
      checkpoint_manager=session.checkpoint_manager,
      analyzer=analyzer,
      baseline_algorithm=baseline_algorithm,
      load_problem_fn=load_problem_and_optimal,
      determine_problem_algorithms_fn=_determine_problem_algorithms,
      analyze_problem_fn=analyze_problem,
      log_problem_analysis_fn=log_problem_analysis,
      release_gpu_fn=_release_gpu_memory,
    )
    if per_algo_results is not None:
      problem_results[problem_config.name] = per_algo_results

  finalize_cohort_and_reports(
    problem_results=problem_results,
    resolved_problems=session.resolved_problems,
    resolved_output_dir=session.resolved_output_dir,
    benchmark_name=str(benchmark_name),
    baseline_algorithm=baseline_algorithm,
    analyzer=analyzer,
    cpu_threshold=session.effective_benchmark_config.cpu_size_threshold,
    reports=reports,
    report_style=report_style,
    include_size_category_table=include_size_category_table,
    analyze_cohort_fn=analyze_cohort,
    log_cohort_analysis_fn=log_cohort_analysis,
    generate_result_tables_fn=generate_result_tables,
  )
  logging.info("BENCHMARK COMPLETE")
  return problem_results


__all__: list[str] = [
  "ResumePlan",
  "_serialize_custom_pipeline",
  "build_checkpoint_static_metadata",
  "load_problem_and_optimal",
  "plan_checkpoint_resume",
  "resolve_problem_selection",
  "run_comprehensive_benchmark",
  "validate_gpu_backend_for_algorithms",
]
