"""
Orchestration Module for Benchmark V2
======================================

Main benchmark loop orchestrating all v2 modules.

This module provides the high-level benchmark workflow:
1. Load configurations
2. Check for existing checkpoints
3. Execute benchmark loop (Problems x Algorithms)
4. Perform per-problem statistical analysis
5. Perform cross-problem analysis (stratified)
6. Generate result tables

Usage:
    from benchmarking.runner.orchestration import run_comprehensive_benchmark

    results = run_comprehensive_benchmark(
        config_dir="configs",
        output_dir="results",
        repetitions=30,
        skip_cpu=False,
        test_mode=False,
    )

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import gc
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
  from data_models.problem import Problem

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

from benchmarking.analysis.cohort_analysis import analyze_cohort
from benchmarking.analysis.problem_statistics import analyze_problem
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.logging_config import setup_benchmark_logging
from benchmarking.reporting.console import log_cohort_analysis, log_problem_analysis
from benchmarking.reporting.generator import generate_result_tables
from benchmarking.reporting.style import ReportArtifact, ReportStyle
from benchmarking.runner.algorithm_runner import (
  adaptive_generations,
  run_single_algorithm,
)
from benchmarking.runner.checkpoint_io import (
  CheckpointManager,
  load_checkpoint,
  quarantine_checkpoint,
  save_checkpoint_atomic,
  save_problem_statistics,
  validate_existing_checkpoints,
)
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  BenchmarkConfig,
  GAParams,
  ProblemConfig,
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
  resolve_benchmark_configs,
  validate_configs,
)
from loaders.database_loader import DatabaseLoader
from loaders.tsplib_loader import TsplibLoader, load_tsplib_file
from protocols.backend import cupy_backend, gpu_available


def _release_gpu_memory() -> None:
  """Return cached CuPy memory blocks to the driver; a no-op without a CUDA device."""
  cp = cupy_backend()
  if cp is not None:
    cp.get_default_memory_pool().free_all_blocks()


#: Public strategy attributes recorded in every checkpoint (`BUG-18`). They are
#: the knobs that change search behaviour, so a checkpoint without them cannot
#: be reproduced from its own metadata.
_STRATEGY_PARAM_ATTRS: tuple[str, ...] = (
  "k",
  "max_iterations",
  "candidate_k",
  "return_gpu",
  "max_matrix_dimension",
  "max_shared_tour_cities",
  "threads_per_block",
  "mode",
)


def _serialize_custom_pipeline(steps: Sequence[Any]) -> list[dict[str, Any]]:
  """Serialize user-declared pipeline steps with the parameters that define them.

  A checkpoint that records only `class` for a pipeline step cannot be reproduced:
  `elite_count`, `interval` and `target` change when and how often local search
  runs, so dropping them silently rewrites the experiment (`BUG-30`). Only the
  caller-declared steps arrive here, so the core steps composed by
  `GeneticAlgorithm` are not listed as if they were configuration.
  """
  specs: list[dict[str, Any]] = []
  for step in steps:
    public_params = getattr(step, "public_params", None)
    params: dict[str, Any] = public_params() if callable(public_params) else {}
    spec: dict[str, Any] = {"class": type(step).__name__}
    if params:
      spec["params"] = params
    specs.append(spec)
  return specs


# =============================================================================
# Helper Functions
# =============================================================================


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
  logging.info(f"  Problems: {len(problem_configs)} TSP instances")
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
  algorithm_configs: dict[str, AlgorithmConfig],
  cpu_threshold: int | None,
) -> list[str]:
  """Determine which algorithms to run for a problem based on size."""
  if cpu_threshold is None or problem_size <= cpu_threshold:
    return selected_algorithms
  else:
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


# =============================================================================
# Main Benchmark Function
# =============================================================================


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
  """Run comprehensive benchmark on configured problems and algorithms.

  By default (``reports=()``), executes collection only (JSON checkpoints and
  per-problem statistical summaries). Pass ``reports=("markdown", "latex", ...)``
  or run ``tsp-report`` afterwards to render summary tables, macros, CD diagrams,
  and scaling plots.
  """
  setup_benchmark_logging(output_dir, benchmark_name, log_level=None)
  logging.info("Loading configurations...")

  configs: dict[str, Any] = resolve_benchmark_configs(
    config_dir=config_dir,
    algorithms_config=algorithms_config,
    benchmark_config=benchmark_config,
    problems_config=problems_config,
    require_algorithms=algorithm_configs_override is None,
  )

  ga_params: GAParams = configs["ga_params"]
  algorithm_configs: dict[str, AlgorithmConfig] = algorithm_configs_override or configs["algorithms"]
  loaded_benchmark_config: BenchmarkConfig = configs["benchmark"]
  default_problems: list[ProblemConfig] = configs["problems"]

  if patience is not None:
    logging.info(f"Applying override: patience={patience}")
    loaded_benchmark_config.patience = patience

  if max_generations is not None:
    if isinstance(max_generations, bool) or max_generations <= 0:
      raise ValueError(f"max_generations must be a positive integer, got {max_generations!r}")
    logging.info(f"Applying override: max_generations={max_generations}")
    loaded_benchmark_config.max_generations = int(max_generations)
  elif loaded_benchmark_config.max_generations is not None and loaded_benchmark_config.max_generations <= 0:
    raise ValueError(f"benchmark_config.max_generations must be > 0, got {loaded_benchmark_config.max_generations}")

  effective_seed: int = seed if seed is not None else ga_params.seed

  param_overrides: dict[str, Any] = {}
  if population_size is not None:
    param_overrides["population_size"] = population_size
  if mutation_rate is not None:
    param_overrides["mutation_rate"] = mutation_rate

  if incremental_runs is not None:
    logging.info(f"Applying override: incremental_runs={incremental_runs}")
    loaded_benchmark_config.incremental_runs = incremental_runs

  if cpu_size_threshold is not None:
    loaded_benchmark_config.cpu_size_threshold = int(cpu_size_threshold)

  effective_repetitions = repetitions if repetitions is not None else loaded_benchmark_config.repetitions
  effective_skip_cpu = skip_cpu
  effective_test_mode = test_mode

  resolved_db_path: Path | None = Path(db_path) if db_path is not None else None
  resolved_tsp_dir: Path | None = Path(tsp_dir) if tsp_dir is not None else None

  has_db: bool = resolved_db_path is not None and resolved_db_path.exists()
  has_tsp_dir: bool = resolved_tsp_dir is not None and resolved_tsp_dir.exists()

  tsp_path_map: dict[str, Path] = {}
  if problem_configs is not None:
    resolved_problems: list[ProblemConfig] = list(problem_configs)
  elif tsp_files:
    resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
      tsp_files,
      existing_configs=default_problems,
      tsp_dir=resolved_tsp_dir,
    )
  elif problem_names:
    if has_tsp_dir and not has_db:
      resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
        problem_names,
        existing_configs=default_problems,
        tsp_dir=resolved_tsp_dir,
      )
    elif has_db and resolved_db_path is not None:
      resolved_problems = build_problem_configs_from_duckdb(
        db_path=resolved_db_path,
        problem_names=problem_names,
        existing_configs=default_problems,
      )
    else:
      raise ValueError(
        "Either an existing --tsp-dir or an existing --db-path must be provided when --problems is specified"
      )
  elif default_problems:
    resolved_problems = list(default_problems)
  elif has_tsp_dir and resolved_tsp_dir is not None:
    discovered_files = sorted(resolved_tsp_dir.glob("*.tsp"))
    if not discovered_files:
      raise FileNotFoundError(f"No .tsp files found in --tsp-dir: {resolved_tsp_dir}")
    resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
      discovered_files,
      existing_configs=default_problems,
      tsp_dir=resolved_tsp_dir,
    )
    resolved_problems.sort(key=lambda cfg: (cfg.size, cfg.name))
  elif has_db and resolved_db_path is not None:
    resolved_problems = build_problem_configs_from_duckdb(
      db_path=resolved_db_path,
      existing_configs=default_problems,
    )
  else:
    raise ValueError(
      "No problems specified: provide --problems-config / problems.json, --tsp-files, --tsp-dir, or --db-path."
    )

  problem_configs = resolved_problems

  if effective_test_mode and not tsp_files and problem_configs:
    problem_configs = problem_configs[:2]
    logging.info("=" * 80)
    logging.info("⚠️  TEST MODE ENABLED")
    logging.info("=" * 80)
    problem_list_str = ", ".join(p.name for p in problem_configs)
    logging.info(f"Running on first {len(problem_configs)} problems only ({problem_list_str})")
    logging.info("Use without --test-mode for full benchmark")
    logging.info("=" * 80)
    logging.info("")

  candidate_algorithms: list[str] = list(algorithms) if algorithms else list(algorithm_configs.keys())

  selected_algorithms = []
  for alg_name in candidate_algorithms:
    if alg_name not in algorithm_configs:
      raise ValueError(f"Unknown algorithm '{alg_name}'. Available: {list(algorithm_configs.keys())}")
    if effective_skip_cpu and not algorithm_configs[alg_name].use_gpu:
      continue
    selected_algorithms.append(alg_name)

  effective_ga_params = GAParams(
    population_size=population_size if population_size is not None else ga_params.population_size,
    mutation_rate=mutation_rate if mutation_rate is not None else ga_params.mutation_rate,
    seed=effective_seed,
  )
  effective_benchmark_config = BenchmarkConfig(
    repetitions=effective_repetitions,
    patience=loaded_benchmark_config.patience,
    max_generations=loaded_benchmark_config.max_generations,
    cpu_size_threshold=loaded_benchmark_config.cpu_size_threshold,
    incremental_runs=loaded_benchmark_config.incremental_runs,
    line_width=loaded_benchmark_config.line_width,
  )

  resolved_output_dir: Path = Path(output_dir)
  resolved_benchmark_name = str(benchmark_name)

  _run_preflight(
    effective_ga_params=effective_ga_params,
    algorithm_configs=algorithm_configs,
    effective_benchmark_config=effective_benchmark_config,
    problem_configs=problem_configs,
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
    effective_repetitions,
    selected_algorithms,
    problem_configs,
    effective_ga_params,
    effective_benchmark_config,
    benchmark_name=resolved_benchmark_name,
  )

  if dry_run:
    logging.info("Dry-run requested: preflight validation passed. Exiting without benchmark execution.")
    return {}

  checkpoint_manager = CheckpointManager(
    checkpoint_dir=resolved_output_dir / "checkpoints",
    problem_stats_dir=resolved_output_dir / "problem_statistics",
  )

  completed, total = checkpoint_manager.count_completed_checkpoints(
    [p.__dict__ for p in problem_configs],
    selected_algorithms,
    effective_repetitions,
    effective_benchmark_config.cpu_size_threshold,
    algorithm_configs=algorithm_configs,
  )
  if completed > 0:
    logging.info(f"\n✓ Found {completed}/{total} completed checkpoints")
    logging.info("  Resuming from previous run...\n")

  if not tsp_path_map:
    if not has_db and not has_tsp_dir:
      raise FileNotFoundError(f"Neither TSPLIB directory ({resolved_tsp_dir}) nor database ({resolved_db_path}) found")

  analyzer = StatisticalAnalyzer()
  problem_results: dict[str, dict[str, dict[str, Any]]] = {}

  for prob_idx, problem_config in enumerate(problem_configs, 1):
    problem_name: str = problem_config.name
    optimal_cost: float | None = problem_config.optimal
    problem_size: int = problem_config.size

    logging.info("\n" + "=" * 80)
    logging.info(f"PROBLEM {prob_idx}/{len(problem_configs)}: {problem_name}")
    opt_display = f"{optimal_cost}" if optimal_cost is not None else "unknown"
    logging.info(f"Size: {problem_size} cities | Optimal: {opt_display}")
    logging.info("=" * 80)

    problem_algorithms: list[str] = _determine_problem_algorithms(
      problem_size,
      selected_algorithms,
      algorithm_configs,
      effective_benchmark_config.cpu_size_threshold,
    )

    cpu_thresh = effective_benchmark_config.cpu_size_threshold
    if cpu_thresh is None or problem_size <= cpu_thresh:
      thresh_label = "all instances" if cpu_thresh is None else f"n≤{cpu_thresh}"
      logging.info(f"Running {len(problem_algorithms)} algorithm(s) ({thresh_label})\n")
    else:
      logging.info(f"Running {len(problem_algorithms)} GPU algorithms only (n>{cpu_thresh})\n")

    try:
      if problem_name in tsp_path_map:
        problem: Problem = load_tsplib_file(tsp_path_map[problem_name])
      elif has_db and resolved_db_path is not None:
        with DatabaseLoader(str(resolved_db_path), fallback_tsp_dir=resolved_tsp_dir) as db_loader:
          problem = db_loader.load(problem_name)
      elif has_tsp_dir and resolved_tsp_dir is not None:
        with TsplibLoader(resolved_tsp_dir) as tsp_loader:
          problem = tsp_loader.load(problem_name)
      else:
        raise FileNotFoundError(
          f"Cannot load problem '{problem_name}': neither --tsp-dir ({resolved_tsp_dir}) "
          f"nor --db-path ({resolved_db_path}) exists"
        )
    except Exception as e:
      logging.error(f"Failed to load {problem_name}: {e}")
      continue

    if problem_config.size != problem.dimension:
      raise ValueError(
        f"Problem '{problem_name}' configured size ({problem_config.size}) "
        f"does not match loaded dimension ({problem.dimension})"
      )

    problem_results[problem_name] = {}
    problem_max_generations: int = (
      effective_benchmark_config.max_generations
      if effective_benchmark_config.max_generations is not None
      else adaptive_generations(problem_size)
    )
    customers: list[int] = list(range(problem_size))

    for algo_idx, alg_name in enumerate(problem_algorithms, 1):
      alg_config = algorithm_configs[alg_name]
      backend_type: Literal["GPU"] | Literal["CPU"] = "GPU" if alg_config.use_gpu else "CPU"

      checkpoint_path: Path = checkpoint_manager.get_checkpoint_path(problem_name, alg_name)
      existing_checkpoint: dict[str, Any] | None = None
      incremental_mode = effective_benchmark_config.incremental_runs

      checkpoint_exists: bool = checkpoint_path.exists()
      existing_runs = 0

      if checkpoint_exists:
        try:
          existing_checkpoint = load_checkpoint(checkpoint_path)
          existing_runs = existing_checkpoint.get("successful_runs", 0) if existing_checkpoint else 0
        except (json.JSONDecodeError, KeyError, OSError) as e:
          quarantine_path = quarantine_checkpoint(checkpoint_path)
          logging.error(
            f"Corrupted checkpoint detected at {checkpoint_path}: {e}. "
            f"Quarantined to {quarantine_path.name}; starting clean run."
          )
          existing_checkpoint = None
          existing_runs = 0
          checkpoint_exists = False

      if checkpoint_exists:
        if incremental_mode:
          new_reps_to_add = effective_repetitions
          total_target_reps = existing_runs + new_reps_to_add

          logging.info(
            f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
            f"✓ Resuming from checkpoint ({existing_runs} runs) - "
            f"adding {new_reps_to_add} more runs → {total_target_reps} total"
          )
          actual_repetitions = new_reps_to_add
        else:
          if existing_runs >= effective_repetitions:
            logging.info(
              f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
              f"✓ Loading from checkpoint (complete)"
            )
            if existing_checkpoint is not None:
              problem_results[problem_name][alg_name] = existing_checkpoint

            if alg_config.use_gpu:
              _release_gpu_memory()

            continue
          else:
            actual_repetitions = effective_repetitions - existing_runs
            logging.info(
              f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
              f"Resuming: {existing_runs}/{effective_repetitions} runs exist, "
              f"completing remaining {actual_repetitions}"
            )
            incremental_mode = True
      else:
        actual_repetitions = effective_repetitions
        logging.info(
          f"  [{algo_idx}/{len(problem_algorithms)}] {alg_name} ({backend_type}): "
          f"Running {actual_repetitions} repetitions..."
        )

      effective_stage = local_search_stage

      def _factory(s: int, ac: AlgorithmConfig = alg_config, stage: str | None = effective_stage) -> Any:
        return ac.instantiate(
          ga_params,
          seed=s,
          local_search_stage=stage,
          param_overrides=param_overrides or None,
          strategy_overrides=strategy_overrides,
        )

      algorithm_instance = _factory(effective_seed)

      def _get_strat_name(strat: Any) -> str:
        if strat is None:
          return "None"
        return getattr(strat, "__name__", type(strat).__name__)

      def _get_strategy_params(
        slot: str,
        *,
        instance: Any = algorithm_instance,
        config: AlgorithmConfig = alg_config,
      ) -> dict[str, Any]:
        """Configured + live parameters of one strategy slot.

        The JSON config is the declared intent, but the instance is what actually
        runs: the CLI can override `candidate_k`, `max_iterations` or `mode`, and
        without recording them a checkpoint cannot be reproduced or compared
        (`BUG-18`). Configured values seed the dict; public attributes read back
        from the instance win. The instance and its config are bound as default
        arguments because the enclosing loop deletes `algorithm_instance` and
        reuses `alg_config` on the next iteration.
        """
        strat: Any = getattr(instance, slot, None)
        if strat is None:
          return {}

        spec: dict[str, Any] = config.strategies.get(slot) or {}
        configured: dict[str, Any] = dict(spec.get("params") or {})
        params: dict[str, Any] = {"class": _get_strat_name(strat)}
        if isinstance(spec.get("module"), str) and spec["module"].strip():
          params["module"] = spec["module"]
        params.update(configured)
        if strategy_overrides and slot in strategy_overrides:
          params.update(strategy_overrides[slot])
        for attr in _STRATEGY_PARAM_ATTRS:
          value: Any = getattr(strat, attr, None)
          if value is not None:
            params[attr] = value
        return params

      algo_cfg_data: dict[str, Any] = {
        "population_size": getattr(algorithm_instance, "population_size", ga_params.population_size),
        "mutation_rate": getattr(algorithm_instance, "mutation_rate", ga_params.mutation_rate),
        "seed": getattr(algorithm_instance, "seed", ga_params.seed),
      }
      if hasattr(algorithm_instance, "local_search_stage"):
        algo_cfg_data["local_search_stage"] = str(algorithm_instance.local_search_stage)

      algo_cfg_data["strategy_params"] = {
        slot: params
        for slot in ("initialization", "selection", "crossover", "mutation", "local_search", "evaluation")
        if (params := _get_strategy_params(slot))
      }

      if hasattr(algorithm_instance, "pipeline") and algorithm_instance.pipeline is not None:
        custom_steps: list[Any] = list(getattr(algorithm_instance, "custom_pipeline_steps", None) or [])
        if custom_steps:
          algo_cfg_data["pipeline"] = _serialize_custom_pipeline(custom_steps)

      static_metadata: dict[str, Any] = {
        "problem_name": problem_name,
        "problem_size": problem_size,
        "problem_type": getattr(problem, "problem_type", "TSP"),
        "edge_type": getattr(problem, "edge_type", "EUC_2D"),
        "optimal_cost": optimal_cost,
        "max_generations_configured": problem_max_generations,
        "backend": "GPU" if alg_config.use_gpu else "CPU",
        "problem_metadata": problem.to_metadata_dict(optimal_cost),
        "comment": problem.comment,
        "edge_weight_format": problem.edge_weight_format,
        "tsplib_name": problem.tsplib_name,
        "source_file": problem.source_file,
        "file_checksum": problem.file_checksum,
        "algorithm_config": algo_cfg_data,
        "strategies": {
          "initialization": _get_strat_name(getattr(algorithm_instance, "initialization", None)),
          "selection": _get_strat_name(getattr(algorithm_instance, "selection", None)),
          "crossover": _get_strat_name(getattr(algorithm_instance, "crossover", None)),
          "mutation": _get_strat_name(getattr(algorithm_instance, "mutation", None)),
          "local_search": _get_strat_name(getattr(algorithm_instance, "local_search", None)),
          "evaluation": _get_strat_name(getattr(algorithm_instance, "evaluation", None)),
        },
        "construction_heuristic": _get_strat_name(getattr(algorithm_instance, "initialization", None)),
      }

      def _on_repetition_completed(
        rep_num: int,
        current_summary: dict[str, Any],
        *,
        _meta: dict[str, Any] = static_metadata,
        _path: Path = checkpoint_path,
      ) -> None:
        payload = dict(current_summary)
        payload.update(_meta)
        payload["timestamp"] = datetime.now().isoformat()
        save_checkpoint_atomic(_path, payload)

      results: dict[str, Any] = run_single_algorithm(
        algorithm_name=alg_name,
        algorithm_instance=algorithm_instance,
        problem=problem,
        problem_name=problem_name,
        problem_size=problem_size,
        customers=customers,
        max_generations=problem_max_generations,
        optimal_cost=optimal_cost,
        repetitions=actual_repetitions,
        use_gpu=alg_config.use_gpu,
        patience=effective_benchmark_config.patience,
        seed_base=effective_seed,
        existing_checkpoint=existing_checkpoint,
        incremental_mode=incremental_mode and existing_checkpoint is not None,
        algorithm_factory=_factory,
        on_repetition_completed=_on_repetition_completed,
      )

      results.update(static_metadata)
      results["timestamp"] = datetime.now().isoformat()

      checkpoint_path = checkpoint_manager.get_checkpoint_path(problem_name, alg_name)
      save_checkpoint_atomic(checkpoint_path, results)
      problem_results[problem_name][alg_name] = results

      del algorithm_instance
      gc.collect()
      if alg_config.use_gpu:
        _release_gpu_memory()

    logging.info(f"\n{'─' * 80}")
    logging.info(f"STATISTICS FOR {problem_name}")
    logging.info(f"{'─' * 80}")

    problem_analysis = analyze_problem(
      problem_name=problem_name,
      optimal_cost=optimal_cost,
      results=problem_results[problem_name],
      analyzer=analyzer,
      baseline_algorithm=baseline_algorithm,
    )
    log_problem_analysis(problem_analysis)

    save_problem_statistics(
      problem_name,
      problem_size,
      optimal_cost,
      problem_results[problem_name],
      problem_analysis.to_dict(),
      checkpoint_manager,
      problem_metadata=problem.to_metadata_dict(optimal_cost),
    )

    del problem
    gc.collect()

  logging.info("\n" + "=" * 80)
  logging.info("CROSS-PROBLEM ANALYSIS")
  logging.info("=" * 80)

  cohort_analysis = None
  if len(problem_results) > 1 or (problem_results and reports):
    cohort_analysis = analyze_cohort(
      problem_results,
      problem_configs,
      baseline_algorithm=baseline_algorithm,
      alpha=analyzer.alpha,
      cpu_threshold=effective_benchmark_config.cpu_size_threshold,
      algorithm_order=report_style.algorithm_order if report_style is not None else None,
    )
    if len(problem_results) > 1:
      log_cohort_analysis(cohort_analysis)

  if problem_results and reports:
    tables_dir = resolved_output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generate_result_tables(
      all_results=problem_results,
      problem_configs=problem_configs,
      output_dir=tables_dir,
      benchmark_name=resolved_benchmark_name,
      baseline_algorithm=baseline_algorithm,
      include_size_category_table=include_size_category_table,
      style=report_style,
      artifacts=reports,
      analysis=cohort_analysis,
    )
  elif problem_results:
    logging.info(
      "Collection complete (no reports requested). Run 'tsp-report -i %s' or pass '--reports all' to generate reports.",
      resolved_output_dir / "checkpoints",
    )

  logging.info("=" * 80)
  logging.info("BENCHMARK COMPLETE")
  logging.info("=" * 80)

  return problem_results
