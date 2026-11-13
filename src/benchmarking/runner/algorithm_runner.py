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

import copy
import gc
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

from data_models.problem import Problem
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext
from utils.gpu_helpers import _compile_kernel_cached

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
# Algorithm Execution Helpers
# =============================================================================


def compute_series_stats(arr: np.ndarray, total_successful: int) -> tuple[float, float, float, float]:
  """Compute (mean, std, min, max) for non-NaN entries across successful runs.

  Applies Bessel's correction (`ddof=1`) when at least 2 finite values are present
  (`total_successful > 0`), and returns `(0.0, 0.0, 0.0, 0.0)` on empty/all-NaN series.
  """
  if total_successful <= 0 or arr.size == 0:
    return 0.0, 0.0, 0.0, 0.0
  valid = arr[~np.isnan(arr)]
  n_non_nan = int(valid.size)
  if n_non_nan <= 0:
    return 0.0, 0.0, 0.0, 0.0
  mean_val = float(np.mean(valid))
  std_val = float(np.std(valid, ddof=1)) if n_non_nan > 1 else 0.0
  min_val = float(np.min(valid))
  max_val = float(np.max(valid))
  return mean_val, std_val, min_val, max_val


def _initialize_results_dict(existing_checkpoint: dict[str, Any] | None = None) -> dict[str, list[Any]]:
  """Initialize an empty results dictionary or copy existing checkpoint lists."""
  if existing_checkpoint:
    raw_costs = existing_checkpoint.get("raw_costs", [])
    raw_init = existing_checkpoint.get("raw_initial_costs")
    if raw_init is not None:
      initial_costs = list(raw_init)
      if len(initial_costs) < len(raw_costs):
        initial_costs.extend([np.nan] * (len(raw_costs) - len(initial_costs)))
    else:
      initial_costs = [np.nan] * len(raw_costs)

    return {
      "times": existing_checkpoint.get("raw_times", []).copy(),
      "initial_costs": initial_costs,
      "final_costs": raw_costs.copy(),
      "gaps": existing_checkpoint.get("raw_gaps", []).copy(),
      "improvements": [],
      "generations": existing_checkpoint.get("raw_generations", []).copy(),
      "stop_reasons": existing_checkpoint.get("raw_stop_reasons", []).copy(),
      "phase_timings": existing_checkpoint.get("raw_phase_timings", []).copy(),
      "h2d_bytes": [],
      "d2h_bytes": [],
      "kernel_launches": [],
      "best_tours": existing_checkpoint.get("raw_best_tours", []).copy(),
      "seeds": existing_checkpoint.get("raw_seeds", []).copy(),
    }
  return {
    "times": [],
    "initial_costs": [],
    "final_costs": [],
    "gaps": [],
    "improvements": [],
    "generations": [],
    "stop_reasons": [],
    "phase_timings": [],
    "h2d_bytes": [],
    "d2h_bytes": [],
    "kernel_launches": [],
    "best_tours": [],
    "seeds": [],
  }


def _record_failure(results: dict[str, list[Any]]) -> None:
  """Record sentinel NaN and zero values for a failed repetition."""
  results["times"].append(np.nan)
  results["initial_costs"].append(np.nan)
  results["final_costs"].append(np.nan)
  results["gaps"].append(np.nan)
  results["improvements"].append(np.nan)
  results["generations"].append(0)
  results["stop_reasons"].append("error")
  results["phase_timings"].append({})
  results["h2d_bytes"].append(0)
  results["d2h_bytes"].append(0)
  results["kernel_launches"].append(0)
  results["best_tours"].append(None)


# =============================================================================
# Algorithm Execution
# =============================================================================


def _build_summary_payload(
  results: dict[str, list[Any]],
  problem: Problem,
  algorithm_name: str,
  carried_runs: int = 0,
  carried_h2d_bytes: float = 0.0,
  carried_d2h_bytes: float = 0.0,
  carried_kernel_launches: float = 0.0,
) -> dict[str, Any]:
  """Construct normalized benchmark summary dictionary from intermediate or final results.

  Calculates online running aggregates including sample standard deviation with Bessel's
  correction (ddof=1 for n > 1) and phase timing breakdowns.
  """
  times = np.array(results["times"])
  initial_costs_arr = np.array(results["initial_costs"], dtype=np.float64)
  final_costs = np.array(results["final_costs"], dtype=np.float64)
  gaps = np.array(results["gaps"])
  generations_arr = np.array(results["generations"])
  seeds_arr = np.array(results["seeds"])

  valid_init = ~np.isnan(initial_costs_arr) & (initial_costs_arr > 0)
  improvements = np.full_like(initial_costs_arr, np.nan, dtype=np.float64)
  np.divide(initial_costs_arr - final_costs, initial_costs_arr, out=improvements, where=valid_init)
  improvements *= 100.0

  h2d = np.array(results["h2d_bytes"]) if results["h2d_bytes"] else np.zeros(len(times))
  d2h = np.array(results["d2h_bytes"]) if results["d2h_bytes"] else np.zeros(len(times))

  valid_mask = ~np.isnan(times)
  total_successful = int(np.sum(valid_mask))

  valid_tours = [
    results["best_tours"][i] for i in range(len(valid_mask)) if valid_mask[i] and results["best_tours"][i] is not None
  ]
  best_run_idx: int = int(np.nanargmin(final_costs[valid_mask])) if total_successful > 0 else 0
  best_tour = valid_tours[best_run_idx] if valid_tours and best_run_idx < len(valid_tours) else None

  # Aggregate phase statistics across successful runs (Task 11.7, DP11-7, DP11-13)
  raw_phase_timings: list[dict[str, dict[str, float | int]]] = [
    results["phase_timings"][i] for i in range(len(valid_mask)) if valid_mask[i] and i < len(results["phase_timings"])
  ]

  mean_phase_timings: dict[str, dict[str, float]] = {}
  phase_share_pct: dict[str, float] = {}
  phase_share_cv_pct: dict[str, float] = {}

  if raw_phase_timings:
    all_phases: set[str] = set()
    for run_pt in raw_phase_timings:
      all_phases.update(run_pt.keys())

    valid_times_ms = times[valid_mask] * 1000.0

    for phase in sorted(all_phases):
      phase_totals: list[float] = []
      phase_calls: list[int] = []
      phase_shares: list[float] = []

      for idx, run_pt in enumerate(raw_phase_timings):
        p_info = run_pt.get(phase, {})
        tot_ms = float(p_info.get("total_ms", 0.0))
        calls = int(p_info.get("calls", 0))
        phase_totals.append(tot_ms)
        phase_calls.append(calls)
        wall_ms = float(valid_times_ms[idx]) if idx < len(valid_times_ms) else 0.0
        share = (tot_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
        phase_shares.append(share)

      mean_tot = float(np.mean(phase_totals)) if phase_totals else 0.0
      mean_c = float(np.mean(phase_calls)) if phase_calls else 0.0
      mean_phase_timings[phase] = {
        "total_ms": mean_tot,
        "calls": mean_c,
        "mean_ms": (mean_tot / mean_c) if mean_c > 0 else 0.0,
      }
      m_share = float(np.mean(phase_shares)) if phase_shares else 0.0
      phase_share_pct[phase] = m_share
      std_share = float(np.std(phase_shares, ddof=1)) if len(phase_shares) > 1 else 0.0
      phase_share_cv_pct[phase] = (std_share / m_share * 100.0) if m_share > 1e-9 else 0.0

  new_runs = len(h2d)
  transfer_runs: int = carried_runs + new_runs
  mean_h2d_bytes: float = (float(np.sum(h2d)) + carried_h2d_bytes) / transfer_runs if transfer_runs else 0.0
  mean_d2h_bytes: float = (float(np.sum(d2h)) + carried_d2h_bytes) / transfer_runs if transfer_runs else 0.0
  mean_kernel_launches: float = (
    (float(np.sum(results["kernel_launches"])) + carried_kernel_launches) / transfer_runs if transfer_runs else 0.0
  )

  mean_time, std_time, min_time, max_time = compute_series_stats(times, total_successful)
  mean_cost, std_cost, best_cost, worst_cost = compute_series_stats(final_costs, total_successful)
  mean_initial_cost, std_initial_cost, _, _ = compute_series_stats(initial_costs_arr, total_successful)
  mean_gap, std_gap, best_gap, _ = compute_series_stats(gaps, total_successful)
  mean_generations, std_generations, min_generations, max_generations = compute_series_stats(
    generations_arr, total_successful
  )
  mean_improvement, _, _, _ = compute_series_stats(improvements, total_successful)

  summary: dict[str, Any] = {
    "algorithm": algorithm_name,
    "repetitions": len(times),
    "successful_runs": total_successful,
    "mean_time": mean_time,
    "std_time": std_time,
    "min_time": min_time,
    "max_time": max_time,
    "mean_cost": mean_cost,
    "std_cost": std_cost,
    "best_cost": best_cost,
    "worst_cost": worst_cost,
    "mean_initial_cost": mean_initial_cost,
    "std_initial_cost": std_initial_cost,
    "mean_gap": mean_gap,
    "std_gap": std_gap,
    "best_gap": best_gap,
    "mean_improvement": mean_improvement,
    "mean_generations": mean_generations,
    "std_generations": std_generations,
    "min_generations": min_generations,
    "max_generations": max_generations,
    "mean_h2d_mb": mean_h2d_bytes / 1e6,
    "mean_d2h_mb": mean_d2h_bytes / 1e6,
    "total_transfer_mb": (mean_h2d_bytes + mean_d2h_bytes) / 1e6,
    "mean_kernels": mean_kernel_launches,
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
    "raw_phase_timings": raw_phase_timings,
    "mean_phase_timings": mean_phase_timings,
    "phase_share_pct": phase_share_pct,
    "phase_share_cv_pct": phase_share_cv_pct,
  }
  return summary


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
  algorithm_factory: Any | None = None,
  on_repetition_completed: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  """
  Run single algorithm for specified repetitions.
  """
  backend: Literal["GPU"] | Literal["CPU"] = "GPU" if use_gpu else "CPU"

  if patience is None:
    patience = adaptive_patience(problem_size)

  # Telemetry carried from the runs already on disk (incremental mode). `results`
  # below starts its per-run lists empty and only the *new* repetitions land there,
  # so without these totals an incremental resume would report the transfer and
  # launch means of the new slice as if they described the whole cohort (`BUG-30`).
  carried_runs: int = 0
  carried_h2d_bytes: float = 0.0
  carried_d2h_bytes: float = 0.0
  carried_kernel_launches: float = 0.0

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

    carried_runs = int(existing_runs)
    carried_h2d_bytes = float(existing_checkpoint.get("mean_h2d_mb", 0.0) or 0.0) * 1e6 * carried_runs
    carried_d2h_bytes = float(existing_checkpoint.get("mean_d2h_mb", 0.0) or 0.0) * 1e6 * carried_runs
    carried_kernel_launches = float(existing_checkpoint.get("mean_kernels", 0.0) or 0.0) * carried_runs

    results = _initialize_results_dict(existing_checkpoint)
  else:
    logging.info(f"\n  [{algorithm_name}] Processing {problem_name} (n={problem_size}, optimal={optimal_cost:.0f})")
    logging.info(f"    Backend: {backend} | Early stopping: {patience} gen patience | Repetitions: {repetitions}")

    results = _initialize_results_dict()

  # Explicit ahead-of-time pre-compilation for GPU variants (Task 11.11, DP11-12, BUG-25).
  # A discarded one-generation run is not a warm-up: `PeriodicLocalSearchStep(interval=10)`
  # never fires at generation 1, so `two_opt_batch.cu` was JIT-compiled inside the first
  # *measured* repetition (26.27x spike). `warmup(context)` visits every strategy and every
  # pipeline step, so coverage does not depend on any generation cadence.
  jit_cache_before: int | None = None
  if use_gpu and repetitions > 0:
    try:
      warmup_instance: Any
      if callable(algorithm_factory):
        warmup_instance = algorithm_factory(seed_base)
      else:
        # Isolated clone: warming the measured instance would advance its RNG and
        # internal state, so repetition 1 would no longer run the seeded experiment.
        warmup_instance = copy.deepcopy(algorithm_instance)
      warmup_xp: Any = cupy_backend() or np
      warmup_context = ProblemContext(problem, xp=warmup_xp, seed=seed_base)
      logging.info("    [Warm-up] Pre-compiling every pipeline kernel (AOT protocol)...")
      warmup_instance.warmup(warmup_context)
      jit_cache_before = _compile_kernel_cached.cache_info().currsize
    except Exception as e:
      logging.warning(f"    [Warm-up] GPU pre-compilation skipped or failed: {e}")

  offset: int = 0
  existing_seeds: set[int] = set()
  if incremental_mode and existing_checkpoint:
    offset = int(existing_checkpoint.get("repetitions", existing_checkpoint.get("successful_runs", 0)))
    raw_seeds = existing_checkpoint.get("raw_seeds", [])
    existing_seeds = {int(s) for s in raw_seeds}
    if raw_seeds:
      max_offset = max((int(s) - seed_base for s in raw_seeds), default=-1)
      if max_offset >= offset:
        offset = max_offset + 1

  current_run_idx: int = offset
  for rep in range(repetitions):
    run_seed: int = seed_base + current_run_idx
    while run_seed in existing_seeds:
      current_run_idx += 1
      run_seed = seed_base + current_run_idx

    results["seeds"].append(run_seed)
    existing_seeds.add(run_seed)
    current_run_idx += 1

    xp: Any = cupy_backend() if use_gpu else np
    context = ProblemContext(problem, xp=xp, seed=run_seed)

    current_algo = algorithm_factory(run_seed) if callable(algorithm_factory) else algorithm_instance

    if incremental_mode and existing_checkpoint:
      current_rep = existing_checkpoint["successful_runs"] + rep + 1
      total_target = existing_checkpoint["successful_runs"] + repetitions
      logging.info(f"    Run {current_rep}/{total_target} (new #{rep + 1}/{repetitions}, seed={run_seed})...")
    else:
      logging.info(f"    Repetition {rep + 1}/{repetitions} (seed={run_seed})...")

    start_time: float = time.perf_counter()
    try:
      tour, stats = current_algo.evolve(
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
      if math.isfinite(initial_cost) and initial_cost > 0:
        improvement = (initial_cost - final_cost) / initial_cost * 100.0
      else:
        improvement = np.nan
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
      results["phase_timings"].append(stats.get("phase_timings", {}))
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

      _record_failure(results)

    if on_repetition_completed is not None:
      current_summary = _build_summary_payload(
        results=results,
        problem=problem,
        algorithm_name=algorithm_name,
        carried_runs=carried_runs,
        carried_h2d_bytes=carried_h2d_bytes,
        carried_d2h_bytes=carried_d2h_bytes,
        carried_kernel_launches=carried_kernel_launches,
      )
      on_repetition_completed(rep + 1, current_summary)

  if jit_cache_before is not None:
    jit_cache_after: int = _compile_kernel_cached.cache_info().currsize
    if jit_cache_after != jit_cache_before:
      raise RuntimeError(
        "JIT compilation landed inside the measured loop: "
        f"_compile_kernel_cached.currsize {jit_cache_before} -> {jit_cache_after}. "
        "A strategy or pipeline step is missing its warmup() coverage (BUG-25)."
      )

  summary = _build_summary_payload(
    results=results,
    problem=problem,
    algorithm_name=algorithm_name,
    carried_runs=carried_runs,
    carried_h2d_bytes=carried_h2d_bytes,
    carried_d2h_bytes=carried_d2h_bytes,
    carried_kernel_launches=carried_kernel_launches,
  )

  if incremental_mode and existing_checkpoint:
    logging.info(
      f"\n    ✓ {algorithm_name} INCREMENTAL summary for {problem_name} "
      f"({summary['successful_runs']} total runs): "
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
  """Free all GPU memory blocks (no-op when no CUDA device is reachable)."""
  cp = cupy_backend()
  if cp is not None:
    cp.get_default_memory_pool().free_all_blocks()
    logging.debug("GPU memory pool freed")


def cleanup_resources(use_gpu: bool = False) -> None:
  """Clean up algorithm resources and memory."""
  gc.collect()
  if use_gpu:
    cleanup_gpu_memory()
