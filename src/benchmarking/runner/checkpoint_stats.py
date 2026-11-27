"""Per-problem checkpoint statistics persistence and JSON serialization helpers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from pathlib import Path

  from benchmarking.runner.checkpoint_io import CheckpointManager


def numpy_to_json_converter(obj: Any) -> Any:
  """Convert numpy types to JSON-serializable Python types."""
  if hasattr(obj, "tolist"):
    return obj.tolist()
  if hasattr(obj, "item"):
    val = obj.item()
    if isinstance(val, float) and (val != val):
      return None
    return val
  raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_problem_statistics(
  problem_name: str,
  problem_size: int,
  optimal_cost: float | None,
  problem_results: dict[str, dict[str, Any]],
  statistical_test_results: dict[str, Any],
  checkpoint_manager: CheckpointManager,
  problem_metadata: dict[str, Any] | None = None,
) -> None:
  """Persist per-problem summary metrics and statistical test outputs to JSON."""
  from benchmarking.runner.algorithm_runner import compute_series_stats

  stats_path: Path = checkpoint_manager.get_problem_stats_path(problem_name)

  try:
    problem_info: dict[str, Any] = {
      "name": problem_name,
      "size": problem_size,
      "optimal_cost": optimal_cost,
    }
    if problem_metadata:
      problem_info.update(problem_metadata)

    problem_stats: dict[str, Any] = {
      "problem_info": problem_info,
      "algorithms": {},
      "statistical_tests": statistical_test_results,
    }

    for alg_name, results in problem_results.items():
      raw_times = results.get("raw_times", results.get("times"))
      raw_initial_costs = results.get("raw_initial_costs", results.get("initial_costs"))
      raw_costs = results.get("raw_costs", results.get("final_costs"))
      raw_gaps = results.get("raw_gaps", results.get("gaps"))
      raw_generations = results.get("raw_generations", results.get("generations"))

      times = np.array(raw_times if raw_times is not None else [], dtype=np.float64)
      initial_costs = np.array(raw_initial_costs if raw_initial_costs is not None else [], dtype=np.float64)
      costs = np.array(raw_costs if raw_costs is not None else [], dtype=np.float64)
      generations = np.array(raw_generations if raw_generations is not None else [], dtype=np.float64)

      n_valid = int(np.sum(~np.isnan(times)))
      if times.size > 0:
        mean_time, std_time, _, _ = compute_series_stats(times, n_valid)
      else:
        mean_time = float(results["mean_time"]) if results.get("mean_time") is not None else 0.0
        std_time = float(results["std_time"]) if results.get("std_time") is not None else 0.0

      n_valid_init = int(np.sum(~np.isnan(initial_costs)))
      mean_initial_cost, std_initial_cost, _, _ = compute_series_stats(initial_costs, n_valid_init)

      raw_is_feasible = results.get("raw_is_feasible")
      raw_unpenalized_costs = results.get("raw_unpenalized_costs")
      has_vrp_metrics = raw_is_feasible is not None or "feasible_runs" in results

      mean_cost: float | None
      std_cost: float | None
      mean_gap: float | None
      std_gap: float | None
      vrp_extra: dict[str, Any] = {}

      if has_vrp_metrics:
        feas_flags = [bool(v) for v in raw_is_feasible] if isinstance(raw_is_feasible, list) else []
        unpen_source = raw_unpenalized_costs if isinstance(raw_unpenalized_costs, list) else raw_costs
        unpen_list = list(unpen_source) if isinstance(unpen_source, list) else []
        feas_costs = np.array(
          [float(unpen_list[i]) for i in range(min(len(feas_flags), len(unpen_list))) if feas_flags[i]],
          dtype=np.float64,
        )
        feasible_runs = int(results["feasible_runs"]) if "feasible_runs" in results else int(feas_costs.size)
        if feas_costs.size > 0:
          mean_cost, std_cost, _, _ = compute_series_stats(feas_costs, int(feas_costs.size))
          if optimal_cost is not None and optimal_cost > 0:
            feas_gaps = (feas_costs - float(optimal_cost)) / float(optimal_cost) * 100.0
            mean_gap, std_gap, _, _ = compute_series_stats(feas_gaps, int(feas_gaps.size))
          else:
            mean_gap = None
            std_gap = None
        elif feasible_runs == 0:
          mean_cost = None
          std_cost = None
          mean_gap = None
          std_gap = None
        else:
          mean_cost = float(results["mean_cost"]) if results.get("mean_cost") is not None else None
          std_cost = float(results["std_cost"]) if results.get("std_cost") is not None else None
          mean_gap = float(results["mean_gap"]) if results.get("mean_gap") is not None else None
          std_gap = float(results["std_gap"]) if results.get("std_gap") is not None else None

        denom = len(feas_flags) if feas_flags else max(n_valid, 0)
        feasibility_rate = (
          float(results["feasibility_rate"])
          if results.get("feasibility_rate") is not None
          else ((100.0 * feasible_runs / denom) if denom > 0 else 0.0)
        )
        vrp_extra = {
          "feasible_runs": feasible_runs,
          "feasibility_rate": feasibility_rate,
          "mean_vehicles": results.get("mean_vehicles"),
          "best_vehicles": results.get("best_vehicles"),
        }
      else:
        n_valid_costs = int(np.sum(~np.isnan(costs)))
        if costs.size > 0:
          mean_cost, std_cost, _, _ = compute_series_stats(costs, n_valid_costs)
        else:
          mean_cost = float(results["mean_cost"]) if results.get("mean_cost") is not None else 0.0
          std_cost = float(results["std_cost"]) if results.get("std_cost") is not None else 0.0

        if raw_gaps is not None and optimal_cost is not None:
          gaps = np.array(raw_gaps, dtype=np.float64)
          mean_gap, std_gap, _, _ = compute_series_stats(gaps, int(np.sum(~np.isnan(gaps))))
        elif optimal_cost is not None and optimal_cost > 0:
          if costs.size > 0:
            gaps = (costs - float(optimal_cost)) / float(optimal_cost) * 100.0
            mean_gap, std_gap, _, _ = compute_series_stats(gaps, n_valid_costs)
          elif results.get("mean_gap") is not None:
            mean_gap = float(results["mean_gap"])
            std_gap = float(results["std_gap"]) if results.get("std_gap") is not None else 0.0
          elif results.get("mean_cost") is not None:
            mean_gap = float((float(results["mean_cost"]) - float(optimal_cost)) / float(optimal_cost) * 100.0)
            std_gap = (
              float(float(results["std_cost"]) / float(optimal_cost) * 100.0)
              if results.get("std_cost") is not None
              else 0.0
            )
          else:
            mean_gap = None
            std_gap = None
        else:
          mean_gap = None
          std_gap = None

      n_valid_gens = int(np.sum(~np.isnan(generations)))
      mean_generations, std_generations, _, _ = compute_series_stats(generations, n_valid_gens)

      if len(times) > 0:
        success_rate = float(n_valid / len(times))
      elif results.get("success_rate") is not None:
        success_rate = float(results["success_rate"])
      else:
        success_rate = 0.0

      alg_entry: dict[str, Any] = {
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_initial_cost": mean_initial_cost,
        "std_initial_cost": std_initial_cost,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "mean_generations": mean_generations,
        "std_generations": std_generations,
        "success_rate": success_rate,
        "backend": results.get("backend", "unknown"),
        "mean_phase_timings": results.get("mean_phase_timings", {}),
        "phase_share_pct": results.get("phase_share_pct", {}),
        "phase_share_cv_pct": results.get("phase_share_cv_pct", {}),
        "algorithm_config": results.get("algorithm_config", {}),
        "strategies": results.get("strategies", {}),
      }
      if vrp_extra:
        alg_entry.update(vrp_extra)
      problem_stats["algorithms"][alg_name] = alg_entry

    with open(stats_path, "w", encoding="utf-8") as f:
      json.dump(problem_stats, f, indent=2, default=numpy_to_json_converter)
    logging.info(f"✓ Problem statistics saved: {stats_path.name}\n")
  except Exception as e:
    logging.warning(f"Failed to save problem statistics: {e}\n")


__all__ = [
  "numpy_to_json_converter",
  "save_problem_statistics",
]
