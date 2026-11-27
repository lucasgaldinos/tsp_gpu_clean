"""Runtime override resolution, algorithm cell execution, and checkpoint metadata builders (`BUG-116`)."""

from __future__ import annotations

import gc
import logging
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from benchmarking.runner.algorithm_runner import adaptive_generations, run_single_algorithm
from benchmarking.runner.checkpoint_io import save_checkpoint_atomic, save_problem_statistics
from benchmarking.runner.config_loader import AlgorithmConfig, BenchmarkConfig, GAParams, ProblemConfig
from benchmarking.runner.orchestration_helpers import (
  ResumePlan,
  get_strategy_name,
  get_strategy_params,
  plan_checkpoint_resume,
  serialize_custom_pipeline,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Mapping, Sequence
  from pathlib import Path

  from benchmarking.analysis.statistics import StatisticalAnalyzer
  from benchmarking.reporting.style import ReportArtifact, ReportStyle
  from benchmarking.runner.checkpoint_io import CheckpointManager
  from data_models.problem import Problem


def instantiate_algorithm_for_seed(
  alg_config: AlgorithmConfig,
  ga_params: GAParams,
  seed: int,
  *,
  local_search_stage: str | None = None,
  param_overrides: Mapping[str, Any] | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
  """Top-level algorithm factory replacing the per-loop `_factory` closure."""
  return alg_config.instantiate(
    ga_params,
    seed=seed,
    local_search_stage=local_search_stage,
    param_overrides=param_overrides or None,
    strategy_overrides=strategy_overrides,
  )


def build_checkpoint_static_metadata(
  *,
  problem: Problem,
  problem_name: str,
  problem_size: int,
  optimal_cost: float | None,
  max_generations: int,
  alg_config: AlgorithmConfig,
  algorithm_instance: Any,
  ga_params: GAParams,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
  """Assemble static problem and algorithm metadata dictionary persisted on every checkpoint."""
  algo_cfg_data: dict[str, Any] = {
    "population_size": getattr(algorithm_instance, "population_size", ga_params.population_size),
    "mutation_rate": getattr(algorithm_instance, "mutation_rate", ga_params.mutation_rate),
    "seed": getattr(algorithm_instance, "seed", ga_params.seed),
  }
  if hasattr(algorithm_instance, "local_search_stage"):
    algo_cfg_data["local_search_stage"] = str(algorithm_instance.local_search_stage)

  slots = ("initialization", "selection", "crossover", "mutation", "local_search", "evaluation")
  algo_cfg_data["strategy_params"] = {
    slot: params
    for slot in slots
    if (
      params := get_strategy_params(
        slot,
        instance=algorithm_instance,
        config=alg_config,
        strategy_overrides=strategy_overrides,
      )
    )
  }

  if hasattr(algorithm_instance, "pipeline") and algorithm_instance.pipeline is not None:
    custom_steps: list[Any] = list(getattr(algorithm_instance, "custom_pipeline_steps", None) or [])
    if custom_steps:
      algo_cfg_data["pipeline"] = serialize_custom_pipeline(custom_steps)

  return {
    "problem_name": problem_name,
    "problem_size": problem_size,
    "problem_type": getattr(problem, "problem_type", "TSP"),
    "edge_type": getattr(problem, "edge_type", "EUC_2D"),
    "optimal_cost": optimal_cost,
    "max_generations_configured": max_generations,
    "backend": "GPU" if alg_config.use_gpu else "CPU",
    "problem_metadata": problem.to_metadata_dict(optimal_cost),
    "comment": problem.comment,
    "edge_weight_format": problem.edge_weight_format,
    "tsplib_name": problem.tsplib_name,
    "source_file": problem.source_file,
    "file_checksum": problem.file_checksum,
    "algorithm_config": algo_cfg_data,
    "strategies": {slot: get_strategy_name(getattr(algorithm_instance, slot, None)) for slot in slots},
    "construction_heuristic": get_strategy_name(getattr(algorithm_instance, "initialization", None)),
  }


def save_repetition_checkpoint(
  _rep_num: int,
  current_summary: dict[str, Any],
  *,
  checkpoint_path: Path,
  static_metadata: Mapping[str, Any],
) -> None:
  """Top-level repetition callback writing incremental checkpoint snapshots."""
  payload = dict(current_summary)
  payload.update(static_metadata)
  payload["timestamp"] = datetime.now().isoformat()
  save_checkpoint_atomic(checkpoint_path, payload)


def resolve_effective_runtime_configs(
  *,
  configs: Mapping[str, Any],
  algorithm_configs_override: dict[str, AlgorithmConfig] | None,
  algorithms: Sequence[str] | None,
  skip_cpu: bool,
  repetitions: int | None,
  patience: int | None,
  max_generations: int | None,
  incremental_runs: bool | None,
  cpu_size_threshold: int | None,
  seed: int | None,
  population_size: int | None,
  mutation_rate: float | None,
) -> tuple[
  GAParams,
  dict[str, AlgorithmConfig],
  list[ProblemConfig],
  list[str],
  GAParams,
  BenchmarkConfig,
  int,
  dict[str, Any],
]:
  """Merge CLI overrides onto loaded configs and resolve selected algorithms."""
  ga_params: GAParams = configs["ga_params"]
  algorithm_configs: dict[str, AlgorithmConfig] = algorithm_configs_override or configs["algorithms"]
  loaded_bench: BenchmarkConfig = configs["benchmark"]
  default_problems: list[ProblemConfig] = configs["problems"]

  if patience is not None:
    loaded_bench.patience = patience
  if max_generations is not None:
    if isinstance(max_generations, bool) or max_generations <= 0:
      raise ValueError(f"max_generations must be a positive integer, got {max_generations!r}")
    loaded_bench.max_generations = int(max_generations)
  elif loaded_bench.max_generations is not None and loaded_bench.max_generations <= 0:
    raise ValueError(f"benchmark_config.max_generations must be > 0, got {loaded_bench.max_generations}")
  if incremental_runs is not None:
    loaded_bench.incremental_runs = incremental_runs
  if cpu_size_threshold is not None:
    loaded_bench.cpu_size_threshold = int(cpu_size_threshold)

  effective_seed: int = seed if seed is not None else ga_params.seed
  param_overrides: dict[str, Any] = {}
  if population_size is not None:
    param_overrides["population_size"] = population_size
  if mutation_rate is not None:
    param_overrides["mutation_rate"] = mutation_rate

  candidate_algorithms: list[str] = list(algorithms) if algorithms else list(algorithm_configs.keys())
  selected_algorithms: list[str] = []
  for alg_name in candidate_algorithms:
    if alg_name not in algorithm_configs:
      raise ValueError(f"Unknown algorithm '{alg_name}'. Available: {list(algorithm_configs.keys())}")
    if skip_cpu and not algorithm_configs[alg_name].use_gpu:
      continue
    selected_algorithms.append(alg_name)

  effective_ga_params = GAParams(
    population_size=population_size if population_size is not None else ga_params.population_size,
    mutation_rate=mutation_rate if mutation_rate is not None else ga_params.mutation_rate,
    seed=effective_seed,
  )
  effective_benchmark_config = BenchmarkConfig(
    repetitions=repetitions if repetitions is not None else loaded_bench.repetitions,
    patience=loaded_bench.patience,
    max_generations=loaded_bench.max_generations,
    cpu_size_threshold=loaded_bench.cpu_size_threshold,
    incremental_runs=loaded_bench.incremental_runs,
    line_width=loaded_bench.line_width,
  )
  return (
    ga_params,
    algorithm_configs,
    default_problems,
    selected_algorithms,
    effective_ga_params,
    effective_benchmark_config,
    effective_seed,
    param_overrides,
  )


def run_problem_algorithm_cell(
  *,
  alg_name: str,
  algo_idx: int,
  total_algos: int,
  alg_config: AlgorithmConfig,
  problem: Problem,
  problem_name: str,
  problem_size: int,
  customers: list[int],
  optimal_cost: float | None,
  problem_max_generations: int,
  effective_repetitions: int,
  incremental_runs: bool,
  patience: int | None,
  effective_seed: int,
  ga_params: GAParams,
  local_search_stage: str | None,
  param_overrides: Mapping[str, Any] | None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
  checkpoint_path: Path,
  release_gpu_fn: Callable[[], None],
) -> dict[str, Any] | None:
  """Execute or resume one `(problem, algorithm)` cell and return its checkpoint summary dict."""
  backend_type = "GPU" if alg_config.use_gpu else "CPU"
  resume_plan: ResumePlan = plan_checkpoint_resume(
    checkpoint_path,
    effective_repetitions=effective_repetitions,
    incremental_runs=incremental_runs,
    alg_name=alg_name,
    algo_idx=algo_idx,
    total_algos=total_algos,
    backend_type=backend_type,
  )
  if resume_plan.skip:
    if alg_config.use_gpu:
      release_gpu_fn()
    return resume_plan.existing_checkpoint

  algo_factory = partial(
    instantiate_algorithm_for_seed,
    alg_config,
    ga_params,
    local_search_stage=local_search_stage,
    param_overrides=param_overrides or None,
    strategy_overrides=strategy_overrides,
  )
  algorithm_instance = algo_factory(effective_seed)
  static_metadata = build_checkpoint_static_metadata(
    problem=problem,
    problem_name=problem_name,
    problem_size=problem_size,
    optimal_cost=optimal_cost,
    max_generations=problem_max_generations,
    alg_config=alg_config,
    algorithm_instance=algorithm_instance,
    ga_params=ga_params,
    strategy_overrides=strategy_overrides,
  )
  rep_callback = partial(
    save_repetition_checkpoint,
    checkpoint_path=checkpoint_path,
    static_metadata=static_metadata,
  )
  results = run_single_algorithm(
    algorithm_name=alg_name,
    algorithm_instance=algorithm_instance,
    problem=problem,
    problem_name=problem_name,
    problem_size=problem_size,
    customers=customers,
    max_generations=problem_max_generations,
    optimal_cost=optimal_cost,
    repetitions=resume_plan.actual_repetitions,
    use_gpu=alg_config.use_gpu,
    patience=patience,
    seed_base=effective_seed,
    existing_checkpoint=resume_plan.existing_checkpoint,
    incremental_mode=resume_plan.incremental_mode and resume_plan.existing_checkpoint is not None,
    algorithm_factory=algo_factory,
    on_repetition_completed=rep_callback,
  )
  results.update(static_metadata)
  results["timestamp"] = datetime.now().isoformat()
  save_checkpoint_atomic(checkpoint_path, results)
  del algorithm_instance
  if alg_config.use_gpu:
    release_gpu_fn()
  return results


def run_single_problem_phase(
  *,
  prob_idx: int,
  total_problems: int,
  problem_config: ProblemConfig,
  selected_algorithms: list[str],
  algorithm_configs: Mapping[str, AlgorithmConfig],
  effective_benchmark_config: BenchmarkConfig,
  effective_seed: int,
  ga_params: GAParams,
  local_search_stage: str | None,
  param_overrides: Mapping[str, Any] | None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
  tsp_path_map: Mapping[str, Path],
  has_db: bool,
  resolved_db_path: Path | None,
  has_tsp_dir: bool,
  resolved_tsp_dir: Path | None,
  checkpoint_manager: CheckpointManager,
  analyzer: StatisticalAnalyzer,
  baseline_algorithm: str | None,
  load_problem_fn: Callable[..., tuple[Problem, float | None]],
  determine_problem_algorithms_fn: Callable[[int, list[str], Mapping[str, AlgorithmConfig], int | None], list[str]],
  analyze_problem_fn: Callable[..., Any],
  log_problem_analysis_fn: Callable[[Any], None],
  release_gpu_fn: Callable[[], None],
) -> dict[str, dict[str, Any]] | None:
  """Load one problem instance, run all selected algorithms, analyze, and persist problem statistics."""
  problem_name, problem_size = problem_config.name, problem_config.size
  problem_algorithms = determine_problem_algorithms_fn(
    problem_size,
    selected_algorithms,
    algorithm_configs,
    effective_benchmark_config.cpu_size_threshold,
  )
  logging.info(f"\nPROBLEM {prob_idx}/{total_problems}: {problem_name} (n={problem_size})")
  try:
    problem, optimal_cost = load_problem_fn(
      problem_config,
      tsp_path_map=tsp_path_map,
      has_db=has_db,
      resolved_db_path=resolved_db_path,
      has_tsp_dir=has_tsp_dir,
      resolved_tsp_dir=resolved_tsp_dir,
    )
  except Exception as exc:
    logging.error(f"Failed to load {problem_name}: {exc}")
    return None
  if problem_config.size != problem.dimension:
    raise ValueError(
      f"Problem '{problem_name}' configured size ({problem_config.size}) "
      f"does not match loaded dimension ({problem.dimension})"
    )

  per_algo_results: dict[str, dict[str, Any]] = {}
  problem_max_generations = (
    effective_benchmark_config.max_generations
    if effective_benchmark_config.max_generations is not None
    else adaptive_generations(problem_size)
  )
  customers = list(range(problem_size))

  for algo_idx, alg_name in enumerate(problem_algorithms, 1):
    cell_result = run_problem_algorithm_cell(
      alg_name=alg_name,
      algo_idx=algo_idx,
      total_algos=len(problem_algorithms),
      alg_config=algorithm_configs[alg_name],
      problem=problem,
      problem_name=problem_name,
      problem_size=problem_size,
      customers=customers,
      optimal_cost=optimal_cost,
      problem_max_generations=problem_max_generations,
      effective_repetitions=effective_benchmark_config.repetitions,
      incremental_runs=effective_benchmark_config.incremental_runs,
      patience=effective_benchmark_config.patience,
      effective_seed=effective_seed,
      ga_params=ga_params,
      local_search_stage=local_search_stage,
      param_overrides=param_overrides,
      strategy_overrides=strategy_overrides,
      checkpoint_path=checkpoint_manager.get_checkpoint_path(problem_name, alg_name),
      release_gpu_fn=release_gpu_fn,
    )
    if cell_result is not None:
      per_algo_results[alg_name] = cell_result
    gc.collect()

  problem_analysis = analyze_problem_fn(
    problem_name=problem_name,
    optimal_cost=optimal_cost,
    results=per_algo_results,
    analyzer=analyzer,
    baseline_algorithm=baseline_algorithm,
  )
  log_problem_analysis_fn(problem_analysis)
  save_problem_statistics(
    problem_name,
    problem_size,
    optimal_cost,
    per_algo_results,
    problem_analysis.to_dict(),
    checkpoint_manager,
    problem_metadata=problem.to_metadata_dict(optimal_cost),
  )
  del problem
  gc.collect()
  return per_algo_results


def finalize_cohort_and_reports(
  *,
  problem_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  resolved_problems: Sequence[ProblemConfig],
  resolved_output_dir: Path,
  benchmark_name: str,
  baseline_algorithm: str | None,
  analyzer: StatisticalAnalyzer,
  cpu_threshold: int | None,
  reports: Sequence[ReportArtifact],
  report_style: ReportStyle | None,
  include_size_category_table: bool,
  analyze_cohort_fn: Callable[..., Any],
  log_cohort_analysis_fn: Callable[[Any], None],
  generate_result_tables_fn: Callable[..., Any],
) -> None:
  """Run cohort-level statistical synthesis and generate requested report tables."""
  cohort_analysis = None
  if len(problem_results) > 1 or (problem_results and reports):
    cohort_analysis = analyze_cohort_fn(
      problem_results,
      resolved_problems,
      baseline_algorithm=baseline_algorithm,
      alpha=analyzer.alpha,
      cpu_threshold=cpu_threshold,
      algorithm_order=report_style.algorithm_order if report_style is not None else None,
    )
    if len(problem_results) > 1:
      log_cohort_analysis_fn(cohort_analysis)

  if problem_results and reports:
    tables_dir = resolved_output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_result_tables_fn(
      all_results=problem_results,
      problem_configs=resolved_problems,
      output_dir=tables_dir,
      benchmark_name=benchmark_name,
      baseline_algorithm=baseline_algorithm,
      include_size_category_table=include_size_category_table,
      style=report_style,
      artifacts=reports,
      analysis=cohort_analysis,
    )


__all__: list[str] = [
  "build_checkpoint_static_metadata",
  "finalize_cohort_and_reports",
  "instantiate_algorithm_for_seed",
  "resolve_effective_runtime_configs",
  "run_problem_algorithm_cell",
  "run_single_problem_phase",
  "save_repetition_checkpoint",
]
