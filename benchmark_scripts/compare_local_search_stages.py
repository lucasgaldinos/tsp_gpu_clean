#!/usr/bin/env python3
"""
Empirical Comparison of Local Search Placement Stages
=====================================================

Compares 2-opt refinement across three pipeline stages:
1. `post_mutation`: canonical memetic GA (refines offspring after mutation)
2. `post_evaluation`: refines the evaluated & surviving population each generation
3. `post_evolution`: refines the full final population once after evolution completes

Evaluates `CPU`, `HybridNaive`, `HybridOptimized`, and `FullGPUModular` across
target TSPLIB benchmark instances (`att48`, `bayg29`, `bays29`).

Usage:
    python -m benchmark_scripts.compare_local_search_stages
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  load_algorithms_config,
)
from benchmarking.runner.orchestration import run_comprehensive_benchmark

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

# Experiment parameters
TARGET_PROBLEMS: list[Path] = [
  PROJECT_ROOT / "datasets" / "tsp" / "att48.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "bayg29.tsp",
  PROJECT_ROOT / "datasets" / "tsp" / "bays29.tsp",
]
STAGES: tuple[str, ...] = ("post_mutation", "post_evaluation", "post_evolution")
BASE_ALGORITHMS: tuple[str, ...] = ("CPU", "HybridNaive", "HybridOptimized", "FullGPUModular")
REPETITIONS: int = 3
OUTPUT_DIR: Path = PROJECT_ROOT / "results_v3" / "local_search_comparison"
BENCHMARK_NAME: str = "local_search_stage_comparison"


def setup_logging() -> None:
  """Configure standard INFO logging for benchmark execution."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
  )


def build_stage_variant_configs() -> dict[str, AlgorithmConfig]:
  """Construct declarative AlgorithmConfig entries for all (algorithm, stage) pairs."""
  _, base_configs = load_algorithms_config(CONFIG_DIR)
  variant_configs: dict[str, AlgorithmConfig] = {}

  for stage in STAGES:
    for base_name in BASE_ALGORITHMS:
      if base_name not in base_configs:
        continue
      base_cfg: AlgorithmConfig = base_configs[base_name]
      variant_name: str = f"{base_name}_{stage}"
      variant_configs[variant_name] = AlgorithmConfig(
        name=variant_name,
        use_gpu=base_cfg.use_gpu,
        description=f"{base_cfg.description} (stage: {stage})",
        metaheuristic=base_cfg.metaheuristic,
        class_module=base_cfg.class_module,
        class_name=base_cfg.class_name,
        params={**base_cfg.params, "local_search_stage": stage},
        strategies=dict(base_cfg.strategies),
      )

  return variant_configs


def main() -> int:
  """Execute the local search stage comparison benchmark via the orchestrator."""
  setup_logging()
  logging.info("Building stage variant configurations (4 algorithms x 3 stages = 12 variants)...")
  stage_configs: dict[str, AlgorithmConfig] = build_stage_variant_configs()

  logging.info(f"Running stage comparison on {len(TARGET_PROBLEMS)} instances: {[p.stem for p in TARGET_PROBLEMS]}")

  run_comprehensive_benchmark(
    config_dir=CONFIG_DIR,
    output_dir=OUTPUT_DIR,
    tsp_files=TARGET_PROBLEMS,
    algorithm_configs_override=stage_configs,
    algorithms=list(stage_configs.keys()),
    repetitions=REPETITIONS,
    benchmark_name=BENCHMARK_NAME,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
