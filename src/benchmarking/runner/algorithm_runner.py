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
from protocols.vrp_strategies import assess_with_evaluator
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


def _is_vrp_problem(problem: Problem) -> bool:
  """Return True when ``problem`` is a VRP variant or carries active VRP constraints."""
  return bool(getattr(problem, "is_vrp", False))


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

    raw_gaps = existing_checkpoint.get("raw_gaps")
    gaps_list: list[Any] = (
      [np.nan if g is None else g for g in raw_gaps] if raw_gaps is not None else [np.nan] * len(raw_costs)
    )

    raw_is_feasible = existing_checkpoint.get("raw_is_feasible")
    raw_unpenalized_costs = existing_checkpoint.get("raw_unpenalized_costs")
    raw_violations = existing_checkpoint.get("raw_violations")
    raw_normalized_violations = existing_checkpoint.get("raw_normalized_violations")
    raw_num_vehicles = existing_checkpoint.get("raw_num_vehicles")

    return {
      "times": existing_checkpoint.get("raw_times", []).copy(),
      "initial_costs": initial_costs,
      "final_costs": raw_costs.copy(),
      "gaps": gaps_list,
      "improvements": [],
      "generations": existing_checkpoint.get("raw_generations", []).copy(),
      "stop_reasons": existing_checkpoint.get("raw_stop_reasons", []).copy(),
      "phase_timings": existing_checkpoint.get("raw_phase_timings", []).copy(),
      "h2d_bytes": [],
      "d2h_bytes": [],
      "kernel_launches": [],
      "best_tours": existing_checkpoint.get("raw_best_tours", []).copy(),
      "seeds": existing_checkpoint.get("raw_seeds", []).copy(),
      "is_feasible": list(raw_is_feasible) if raw_is_feasible is not None else [],
      "unpenalized_costs": list(raw_unpenalized_costs) if raw_unpenalized_costs is not None else [],
      "violations": list(raw_violations) if raw_violations is not None else [],
      "normalized_violations": (
        list(raw_normalized_violations) if raw_normalized_violations is not None else []
      ),
      "num_vehicles": list(raw_num_vehicles) if raw_num_vehicles is not None else [],
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
    "is_feasible": [],
    "unpenalized_costs": [],
    "violations": [],
    "normalized_violations": [],
    "num_vehicles": [],
  }


def _record_failure(results: dict[str, list[Any]], *, is_vrp: bool = False) -> None:
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
  if is_vrp:
    results["is_feasible"].append(False)
    results["unpenalized_costs"].append(np.nan)
    results["violations"].append(np.nan)
    results["normalized_violations"].append(np.nan)
    results["num_vehicles"].append(0)


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
  has_known_optimal: bool = True,
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

  valid_init = ~np.isnan(initial_costs_arr) & (initial_costs_arr > 0) & ~np.isnan(final_costs)
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
  valid_final = final_costs[valid_mask]
  best_run_idx: int = (
    int(np.nanargmin(valid_final))
    if (total_successful > 0 and bool(np.any(~np.isnan(valid_final))))
    else 0
  )
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

  is_vrp = _is_vrp_problem(problem)

  mean_time, std_time, min_time, max_time = compute_series_stats(times, total_successful)
  mean_initial_cost, std_initial_cost, _, _ = compute_series_stats(initial_costs_arr, total_successful)
  mean_cost: float | None
  std_cost: float | None
  best_cost: float | None
  worst_cost: float | None
  mean_gap: float | None
  std_gap: float | None
  best_gap: float | None
  raw_gaps_out: list[float | None] | None
  vrp_fields: dict[str, Any] = {}

  if is_vrp:
    valid_indices = [i for i in range(len(valid_mask)) if valid_mask[i]]
    raw_is_feasible: list[bool] = [
      bool(results["is_feasible"][i]) for i in valid_indices if i < len(results.get("is_feasible", []))
    ]
    raw_unpenalized_costs: list[float] = [
      float(results["unpenalized_costs"][i]) for i in valid_indices if i < len(results.get("unpenalized_costs", []))
    ]
    raw_violations: list[float] = [
      float(results["violations"][i]) for i in valid_indices if i < len(results.get("violations", []))
    ]
    raw_normalized_violations: list[float] = [
      float(results["normalized_violations"][i])
      for i in valid_indices
      if i < len(results.get("normalized_violations", []))
    ]
    raw_num_vehicles: list[int] = [
      int(results["num_vehicles"][i]) for i in valid_indices if i < len(results.get("num_vehicles", []))
    ]

    feasible_indices = [idx for idx, is_feas in enumerate(raw_is_feasible) if is_feas]
    feasible_runs = len(feasible_indices)
    feasibility_rate = (100.0 * feasible_runs / total_successful) if total_successful > 0 else 0.0

    mean_vehicles: float | None = None
    best_vehicles: int | None = None

    if feasible_runs > 0:
      feas_costs_arr = np.array([raw_unpenalized_costs[i] for i in feasible_indices], dtype=np.float64)
      mean_cost, std_cost, best_cost, worst_cost = compute_series_stats(feas_costs_arr, feasible_runs)
      feas_vehicles_arr = np.array([raw_num_vehicles[i] for i in feasible_indices], dtype=np.float64)
      mean_vehicles = float(np.mean(feas_vehicles_arr))
      best_feas_pos = int(np.argmin(feas_costs_arr))
      best_valid_idx = feasible_indices[best_feas_pos]
      best_vehicles = int(raw_num_vehicles[best_valid_idx])
      if valid_tours and best_valid_idx < len(valid_tours):
        best_tour = valid_tours[best_valid_idx]

      if has_known_optimal:
        valid_gaps_arr = np.array(gaps[valid_mask], dtype=np.float64)
        feas_gaps_arr = np.array([valid_gaps_arr[i] for i in feasible_indices], dtype=np.float64)
        mean_gap, std_gap, best_gap, _ = compute_series_stats(feas_gaps_arr, feasible_runs)
        raw_gaps_out = [None if math.isnan(float(g)) else float(g) for g in valid_gaps_arr]
      else:
        mean_gap = None
        std_gap = None
        best_gap = None
        raw_gaps_out = None
    else:
      mean_cost = None
      std_cost = None
      best_cost = None
      worst_cost = None
      mean_gap = None
      std_gap = None
      best_gap = None
      raw_gaps_out = [None] * total_successful if has_known_optimal else None
      best_viol_src = (
        raw_normalized_violations
        if len(raw_normalized_violations) == len(valid_tours)
        else raw_violations
      )
      if best_viol_src and valid_tours:
        best_viol_idx = int(np.argmin(np.array(best_viol_src, dtype=np.float64)))
        if best_viol_idx < len(valid_tours):
          best_tour = valid_tours[best_viol_idx]

    vrp_fields = {
      "raw_is_feasible": raw_is_feasible,
      "raw_unpenalized_costs": raw_unpenalized_costs,
      "raw_violations": raw_violations,
      "raw_normalized_violations": raw_normalized_violations,
      "raw_num_vehicles": raw_num_vehicles,
      "feasible_runs": feasible_runs,
      "feasibility_rate": feasibility_rate,
      "mean_vehicles": mean_vehicles,
      "best_vehicles": best_vehicles,
    }
  else:
    mean_cost, std_cost, best_cost, worst_cost = compute_series_stats(final_costs, total_successful)
    if has_known_optimal:
      mean_gap, std_gap, best_gap, _ = compute_series_stats(gaps, total_successful)
      raw_gaps_out = gaps[valid_mask].tolist()
    else:
      mean_gap = None
      std_gap = None
      best_gap = None
      raw_gaps_out = None

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
    "raw_gaps": raw_gaps_out,
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
  if vrp_fields:
    summary.update(vrp_fields)
  return summary


def run_single_algorithm(
  algorithm_name: str,
  algorithm_instance: Any,
  problem: Problem,
  problem_name: str,
  problem_size: int,
  customers: list[int],
  max_generations: int,
  optimal_cost: float | None,
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

  is_vrp = _is_vrp_problem(problem)
  effective_customers: list[int] = (
    problem.default_evolutionary_nodes if len(customers) == problem.dimension else customers
  )

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
    opt_label = f"{optimal_cost:.0f}" if optimal_cost is not None else "unknown"
    logging.info(f"\n  [{algorithm_name}] Processing {problem_name} (n={problem_size}, optimal={opt_label})")
    logging.info(f"    Backend: {backend} | Early stopping: {patience} gen patience | Repetitions: {repetitions}")

    results = _initialize_results_dict()

  # Explicit ahead-of-time pre-compilation for GPU variants (Task 11.11, DP11-12, BUG-25).
  # A discarded one-generation run is not a warm-up: `PeriodicLocalSearchStep(interval=10)`
  # never fires at generation 1, so `two_opt_batch.cu` was JIT-compiled inside the first
  # *measured* repetition (26.27x spike). `warmup(context)` visits every strategy and every
  # pipeline step, so coverage does not depend on any generation cadence.
  jit_cache_before: int | None = None
  # One candidate-list cache for the whole run: contexts are rebuilt per repetition, so a
  # per-context cache would rebuild `N_k` inside every measured window.
  candidate_cache: dict[int, np.ndarray] = {}

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
      warmup_context = ProblemContext(problem, xp=warmup_xp, seed=seed_base, candidate_cache=candidate_cache)
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
    context = ProblemContext(problem, xp=xp, seed=run_seed, candidate_cache=candidate_cache)

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
        effective_customers,
        max_generations,
        optimal_cost=optimal_cost,
        patience=patience,
      )
      elapsed = time.perf_counter() - start_time

      initial_cost = stats["initial_fitness"]
      final_cost = stats["best_fitness"]
      has_known_optimal = optimal_cost is not None and optimal_cost > 0
      tour_list = tour.tolist() if hasattr(tour, "tolist") else list(tour)

      if is_vrp:
        evaluator = getattr(current_algo, "evaluation", None)
        if evaluator is None:
          raise AttributeError(
            f"Algorithm '{algorithm_name}' must expose an 'evaluation' strategy to assess constrained/VRP routes."
          )
        assessment = assess_with_evaluator(
          evaluator,
          np.asarray(tour_list, dtype=np.int32),
          context,
          return_gpu=False,
        )
        unpenalized_cost = float(assessment.travel_costs[0])
        violation = float(assessment.violations[0])
        norm_violation = float(assessment.normalized_violations[0])
        is_feasible = bool(assessment.is_feasible[0])
        num_vehicles = int(assessment.num_vehicles[0]) if assessment.num_vehicles is not None else 1
        results["unpenalized_costs"].append(unpenalized_cost)
        results["violations"].append(violation)
        results["normalized_violations"].append(norm_violation)
        results["is_feasible"].append(is_feasible)
        results["num_vehicles"].append(num_vehicles)
        recorded_final_cost = unpenalized_cost if is_feasible else float("nan")
        gap = (
          (unpenalized_cost - optimal_cost) / optimal_cost * 100.0
          if (has_known_optimal and optimal_cost is not None and is_feasible)
          else np.nan
        )
        initial_feas_raw = stats.get("initial_feasible_cost")
        initial_feas_cost = float(initial_feas_raw) if initial_feas_raw is not None else float(initial_cost)
        recorded_initial_cost = initial_feas_cost
        if is_feasible and math.isfinite(initial_feas_cost) and initial_feas_cost > 0:
          improvement = (initial_feas_cost - unpenalized_cost) / initial_feas_cost * 100.0
        else:
          improvement = float("nan")
      else:
        recorded_initial_cost = initial_cost
        recorded_final_cost = final_cost
        gap = (
          (final_cost - optimal_cost) / optimal_cost * 100.0
          if (optimal_cost is not None and optimal_cost > 0)
          else np.nan
        )
        if math.isfinite(initial_cost) and initial_cost > 0:
          improvement = (initial_cost - final_cost) / initial_cost * 100.0
        else:
          improvement = np.nan

      generations = stats.get("generations_completed", 0)
      stop_reason = stats.get("stop_reason", "completed")

      results["times"].append(elapsed)
      results["initial_costs"].append(recorded_initial_cost)
      results["final_costs"].append(recorded_final_cost)
      results["gaps"].append(gap)
      results["improvements"].append(improvement)
      results["generations"].append(generations)
      results["stop_reasons"].append(stop_reason)
      results["h2d_bytes"].append(stats.get("h2d_bytes", 0))
      results["d2h_bytes"].append(stats.get("d2h_bytes", 0))
      results["kernel_launches"].append(stats.get("kernel_launches", 0))
      results["phase_timings"].append(stats.get("phase_timings", {}))
      results["best_tours"].append(tour_list)

      cost_log = f"{recorded_final_cost:.0f}" if math.isfinite(recorded_final_cost) else "N/A (infeasible)"
      gap_log = f"gap: {gap:.2f}%" if has_known_optimal and math.isfinite(gap) else "gap: N/A"
      logging.info(
        f"      → Completed in {elapsed:.2f}s | "
        f"Cost: {cost_log} ({gap_log}) | "
        f"Generations: {generations}/{max_generations}"
      )

    except Exception as e:
      logging.error(f"    Rep {rep + 1} FAILED: {e}")
      import traceback

      traceback.print_exc()

      _record_failure(results, is_vrp=is_vrp)

    if on_repetition_completed is not None:
      current_summary = _build_summary_payload(
        results=results,
        problem=problem,
        algorithm_name=algorithm_name,
        carried_runs=carried_runs,
        carried_h2d_bytes=carried_h2d_bytes,
        carried_d2h_bytes=carried_d2h_bytes,
        carried_kernel_launches=carried_kernel_launches,
        has_known_optimal=optimal_cost is not None and optimal_cost > 0,
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
    has_known_optimal=optimal_cost is not None and optimal_cost > 0,
  )

  cost_summary = (
    f"Cost={summary['mean_cost']:.0f}±{summary['std_cost']:.1f}"
    if summary["mean_cost"] is not None and summary["std_cost"] is not None
    else "Cost=N/A"
  )
  gap_summary = (
    f"Gap={summary['mean_gap']:.2f}%±{summary['std_gap']:.2f}%"
    if summary["mean_gap"] is not None and summary["std_gap"] is not None
    else "Gap=N/A"
  )
  if incremental_mode and existing_checkpoint:
    logging.info(
      f"\n    ✓ {algorithm_name} INCREMENTAL summary for {problem_name} "
      f"({summary['successful_runs']} total runs): "
      f"{cost_summary} | "
      f"{gap_summary} | "
      f"Time={summary['mean_time']:.2f}s±{summary['std_time']:.2f}s"
    )
  else:
    logging.info(
      f"\n    ✓ {algorithm_name} summary for {problem_name}: "
      f"{cost_summary} | "
      f"{gap_summary} | "
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
