"""
Benchmarking Framework for GPU-Accelerated Routing Optimization (`src.benchmarking`).

Modular package organized into:
- `benchmarking.runner`: Configuration loading, checkpoint management, and execution orchestration.
- `benchmarking.analysis`: Statistical tests (Friedman, Nemenyi, Holm-Bonferroni) and reporting.
- `benchmarking.database`: DuckDB schema and operations for benchmark results.
"""

from benchmarking.analysis import (
  StatisticalAnalyzer,
  StatisticalSummary,
  compare_algorithms,
  compare_algorithms_by_group,
  generate_result_tables,
  holm_bonferroni_correction,
  perform_cross_problem_analysis_stratified,
  perform_statistical_analysis,
)
from benchmarking.database import (
  ALL_RUNS_BY_ID,
  COUNT_RUNS,
  TABLE_RUNS,
  create_database,
  insert_benchmark_run,
  validate_benchmark_data,
)
from benchmarking.runner import (
  DEFAULT_SELECTED_TSP_FILES,
  KNOWN_TSPLIB_OPTIMALS,
  AlgorithmConfig,
  BenchmarkConfig,
  CheckpointManager,
  GAParams,
  ProblemConfig,
  adaptive_generations,
  adaptive_patience,
  build_problem_configs_from_tsp_files,
  default_config_dir,
  load_algorithms_config,
  load_all_configs,
  load_benchmark_config,
  load_checkpoint,
  load_problems_config,
  run_comprehensive_benchmark,
  run_single_algorithm,
  save_checkpoint,
  save_checkpoint_atomic,
  validate_configs,
)

__all__ = [
  "ALL_RUNS_BY_ID",
  "COUNT_RUNS",
  "DEFAULT_SELECTED_TSP_FILES",
  "KNOWN_TSPLIB_OPTIMALS",
  "TABLE_RUNS",
  "AlgorithmConfig",
  "BenchmarkConfig",
  "CheckpointManager",
  "GAParams",
  "ProblemConfig",
  "StatisticalAnalyzer",
  "StatisticalSummary",
  "adaptive_generations",
  "adaptive_patience",
  "build_problem_configs_from_tsp_files",
  "compare_algorithms",
  "compare_algorithms_by_group",
  "create_database",
  "default_config_dir",
  "generate_result_tables",
  "holm_bonferroni_correction",
  "insert_benchmark_run",
  "load_algorithms_config",
  "load_all_configs",
  "load_benchmark_config",
  "load_checkpoint",
  "load_problems_config",
  "perform_cross_problem_analysis_stratified",
  "perform_statistical_analysis",
  "run_comprehensive_benchmark",
  "run_single_algorithm",
  "save_checkpoint",
  "save_checkpoint_atomic",
  "validate_benchmark_data",
  "validate_configs",
]
