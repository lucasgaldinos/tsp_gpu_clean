"""
Algorithm Runner Module for Benchmark V2
=========================================

Executes algorithm instances with proper resource management.

This module provides the core execution logic for running genetic algorithms
on TSP problems, including:
- Adaptive generation calculation (2xnx√n formula)
- Single algorithm execution with error handling
- GPU memory cleanup
- Comprehensive metrics collection

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import gc
import logging
import math
import os
import tempfile
import time
from typing import Any

import numpy as np

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

try:
  import cupy as cp

  CUPY_AVAILABLE = True
except ImportError:
  CUPY_AVAILABLE = False
  cp = None

from data_models.problem import Problem
from protocols.problem_context import ProblemContext

# =============================================================================
# Generation Calculation
# =============================================================================


def adaptive_generations(n: int) -> int:
  """
  Calculate adaptive generations using formula: 2 x n x sqrt(n).

  Args:
      n: Problem size (number of cities)

  Returns:
      Number of generations to run
  """
  return int(2 * n * math.sqrt(n))


def adaptive_patience(n: int) -> int:
  """
  Calculate adaptive patience (stagnation window) using formula: 2 x sqrt(n).

  Args:
      n: Problem size (number of cities)

  Returns:
      Patience value
  """
  return int(2 * math.sqrt(n))


# =============================================================================
# Algorithm Execution
# =============================================================================


def run_single_algorithm(
  algorithm_name: str,
  algorithm_instance: Any,
  problem: Problem,
  problem_name: str,
  problem_size: int,
  customers: list[int],
  max_generations: int,
  optimal_cost: float,
  repetitions: int,
  use_gpu: bool,
  patience: int | None = None,
  seed_base: int = 42,
  existing_checkpoint: dict[str, Any] | None = None,
  incremental_mode: bool = False,
) -> dict[str, Any]:
  """
  Run single algorithm for specified repetitions.
  """
  backend = "GPU" if use_gpu else "CPU"

  if patience is None:
    patience = adaptive_patience(problem_size)

  if incremental_mode and existing_checkpoint:
    existing_runs = existing_checkpoint["successful_runs"]
    total_runs_after = existing_runs + repetitions
    logging.info(
      f"\n  [{algorithm_name}] INCREMENTAL MODE: Extending {problem_name} "
      f"from {existing_runs} to {total_runs_after} runs"
    )
    logging.info(
      f"    Backend: {backend} | Early stopping: patience={patience} (adaptive: 2x√{problem_size}) | "
      f"New repetitions: {repetitions}"
    )

    results = {
      "times": existing_checkpoint.get("raw_times", []).copy(),
      "initial_costs": existing_checkpoint.get("raw_initial_costs", []).copy(),
      "final_costs": existing_checkpoint.get("raw_costs", []).copy(),
      "gaps": existing_checkpoint.get("raw_gaps", []).copy(),
      "improvements": [],
      "generations": existing_checkpoint.get("raw_generations", []).copy(),
      "stop_reasons": existing_checkpoint.get("raw_stop_reasons", []).copy(),
      "h2d_bytes": [],
      "d2h_bytes": [],
      "kernel_launches": [],
      "best_tours": existing_checkpoint.get("raw_best_tours", []).copy(),
      "seeds": existing_checkpoint.get("raw_seeds", []).copy(),
    }
  else:
    logging.info(f"\n  [{algorithm_name}] Processing {problem_name} (n={problem_size}, optimal={optimal_cost:.0f})")
    logging.info(f"    Backend: {backend} | Early stopping: {patience} gen patience | Repetitions: {repetitions}")

    results = {
      "times": [],
      "initial_costs": [],
      "final_costs": [],
      "gaps": [],
      "improvements": [],
      "generations": [],
      "stop_reasons": [],
      "h2d_bytes": [],
      "d2h_bytes": [],
      "kernel_launches": [],
      "best_tours": [],
      "seeds": [],
    }

  for rep in range(repetitions):
    run_seed = np.random.randint(0, 2**31 - 1)
    results["seeds"].append(run_seed)

    xp = cp if use_gpu and CUPY_AVAILABLE else np
    context = ProblemContext(problem, xp=xp, seed=run_seed)

    if incremental_mode and existing_checkpoint:
      current_rep = existing_checkpoint["successful_runs"] + rep + 1
      total_target = existing_checkpoint["successful_runs"] + repetitions
      logging.info(f"    Run {current_rep}/{total_target} (new #{rep + 1}/{repetitions}, seed={run_seed})...")
    else:
      logging.info(f"    Repetition {rep + 1}/{repetitions} (seed={run_seed})...")

    start_time = time.perf_counter()
    try:
      tour, stats = algorithm_instance.evolve(
        context,
        customers,
        max_generations,
        optimal_cost=optimal_cost,
        patience=patience,
      )
      elapsed = time.perf_counter() - start_time

      initial_cost = stats["initial_fitness"]
      final_cost = stats["best_fitness"]
      gap = (final_cost - optimal_cost) / optimal_cost * 100
      improvement = (initial_cost - final_cost) / initial_cost * 100
      generations = stats.get("generations_completed", 0)
      stop_reason = stats.get("stop_reason", "completed")

      results["times"].append(elapsed)
      results["initial_costs"].append(initial_cost)
      results["final_costs"].append(final_cost)
      results["gaps"].append(gap)
      results["improvements"].append(improvement)
      results["generations"].append(generations)
      results["stop_reasons"].append(stop_reason)
      results["h2d_bytes"].append(stats.get("h2d_bytes", 0))
      results["d2h_bytes"].append(stats.get("d2h_bytes", 0))
      results["kernel_launches"].append(stats.get("kernel_launches", 0))
      tour_list = tour.tolist() if hasattr(tour, "tolist") else list(tour)
      results["best_tours"].append(tour_list)

      logging.info(
        f"      → Completed in {elapsed:.2f}s | "
        f"Cost: {final_cost:.0f} (gap: {gap:.2f}%) | "
        f"Generations: {generations}/{max_generations}"
      )

    except Exception as e:
      logging.error(f"    Rep {rep + 1} FAILED: {e}")
      import traceback

      traceback.print_exc()

      results["times"].append(np.nan)
      results["initial_costs"].append(np.nan)
      results["final_costs"].append(np.nan)
      results["gaps"].append(np.nan)
      results["improvements"].append(np.nan)
      results["generations"].append(0)
      results["stop_reasons"].append("error")
      results["h2d_bytes"].append(0)
      results["d2h_bytes"].append(0)
      results["kernel_launches"].append(0)
      results["best_tours"].append(None)

  times = np.array(results["times"])
  initial_costs_arr = np.array(results["initial_costs"])
  final_costs = np.array(results["final_costs"])
  gaps = np.array(results["gaps"])
  generations_arr = np.array(results["generations"])
  seeds_arr = np.array(results["seeds"])

  improvements = (initial_costs_arr - final_costs) / initial_costs_arr * 100

  h2d = np.array(results["h2d_bytes"]) if results["h2d_bytes"] else np.zeros(len(times))
  d2h = np.array(results["d2h_bytes"]) if results["d2h_bytes"] else np.zeros(len(times))

  valid_mask = ~np.isnan(times)
  total_successful = int(np.sum(valid_mask))

  valid_tours = [
    results["best_tours"][i]
    for i in range(len(valid_mask))
    if valid_mask[i] and results["best_tours"][i] is not None
  ]
  best_run_idx = int(np.nanargmin(final_costs[valid_mask])) if total_successful > 0 else 0
  best_tour = valid_tours[best_run_idx] if valid_tours and best_run_idx < len(valid_tours) else None

  summary = {
    "algorithm": algorithm_name,
    "repetitions": len(times),
    "successful_runs": total_successful,
    "mean_time": float(np.nanmean(times)),
    "std_time": float(np.nanstd(times)),
    "min_time": float(np.nanmin(times)),
    "max_time": float(np.nanmax(times)),
    "mean_cost": float(np.nanmean(final_costs)),
    "std_cost": float(np.nanstd(final_costs)),
    "best_cost": float(np.nanmin(final_costs)),
    "worst_cost": float(np.nanmax(final_costs)),
    "mean_initial_cost": float(np.nanmean(initial_costs_arr)),
    "std_initial_cost": float(np.nanstd(initial_costs_arr)),
    "mean_gap": float(np.nanmean(gaps)),
    "std_gap": float(np.nanstd(gaps)),
    "best_gap": float(np.nanmin(gaps)),
    "mean_improvement": float(np.nanmean(improvements)),
    "mean_generations": float(np.nanmean(generations_arr)),
    "std_generations": float(np.nanstd(generations_arr)),
    "min_generations": float(np.nanmin(generations_arr)),
    "max_generations": float(np.nanmax(generations_arr)),
    "mean_h2d_mb": float(np.nanmean(h2d)) / 1e6 if len(h2d) > 0 else 0.0,
    "mean_d2h_mb": float(np.nanmean(d2h)) / 1e6 if len(d2h) > 0 else 0.0,
    "total_transfer_mb": (float(np.nanmean(h2d)) + float(np.nanmean(d2h))) / 1e6 if len(h2d) > 0 else 0.0,
    "mean_kernels": float(np.nanmean(results["kernel_launches"])) if results["kernel_launches"] else 0.0,
    "best_tour": best_tour,
    "raw_best_tours": valid_tours,
    "problem_type": getattr(problem, "problem_type", "TSP"),
    "edge_type": getattr(problem, "edge_type", "EUC_2D"),
    "raw_times": times[valid_mask].tolist(),
    "raw_initial_costs": initial_costs_arr[valid_mask].tolist(),
    "raw_costs": final_costs[valid_mask].tolist(),
    "raw_gaps": gaps[valid_mask].tolist(),
    "raw_generations": generations_arr[valid_mask].tolist(),
    "raw_stop_reasons": [results["stop_reasons"][i] for i in range(len(valid_mask)) if valid_mask[i]]
    if results["stop_reasons"]
    else [],
    "raw_seeds": seeds_arr[valid_mask].tolist(),
  }

  if incremental_mode and existing_checkpoint:
    logging.info(
      f"\n    ✓ {algorithm_name} INCREMENTAL summary for {problem_name} "
      f"({total_successful} total runs): "
      f"Cost={summary['mean_cost']:.0f}±{summary['std_cost']:.1f} | "
      f"Gap={summary['mean_gap']:.2f}%±{summary['std_gap']:.2f}% | "
      f"Time={summary['mean_time']:.2f}s±{summary['std_time']:.2f}s"
    )
  else:
    logging.info(
      f"\n    ✓ {algorithm_name} summary for {problem_name}: "
      f"Cost={summary['mean_cost']:.0f}±{summary['std_cost']:.1f} | "
      f"Gap={summary['mean_gap']:.2f}%±{summary['std_gap']:.2f}% | "
      f"Time={summary['mean_time']:.2f}s±{summary['std_time']:.2f}s"
    )

  return summary


# =============================================================================
# GPU Memory Management
# =============================================================================


def cleanup_gpu_memory() -> None:
  """Free all GPU memory blocks."""
  if CUPY_AVAILABLE and cp is not None:
    cp.get_default_memory_pool().free_all_blocks()
    logging.debug("GPU memory pool freed")


def cleanup_resources(use_gpu: bool = False) -> None:
  """Clean up algorithm resources and memory."""
  gc.collect()
  if use_gpu:
    cleanup_gpu_memory()
