"""Checkpoint validation and compatibility checks for the modular benchmark runner."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Sequence
  from pathlib import Path

  from benchmarking.runner.config_loader import AlgorithmConfig, GAParams, ProblemConfig

REQUIRED_MODULAR_CHECKPOINT_KEYS: tuple[str, ...] = (
  "problem_name",
  "algorithm",
  "successful_runs",
  "raw_costs",
  "raw_times",
  "strategies",
  "algorithm_config",
)


def _resolve_expected_reps(data: dict[str, Any], expected_reps: int | None) -> int | None:
  """Return the repetition target a checkpoint must reach, or ``None`` if unknown."""
  if expected_reps is not None:
    return expected_reps

  reps: Any = data.get("repetitions")
  if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
    return None
  return reps


def _validate_checkpoint_arrays(filepath: Path, data: Mapping[str, Any]) -> bool:
  """Validate per-repetition raw array lengths against ``successful_runs``."""
  successful_runs = data["successful_runs"]
  if isinstance(successful_runs, bool) or not isinstance(successful_runs, int) or successful_runs < 0:
    logging.warning(f"Checkpoint {filepath.name} has invalid 'successful_runs': {successful_runs!r}")
    return False

  raw_times = data["raw_times"]
  raw_costs = data["raw_costs"]
  raw_seeds = data["raw_seeds"]
  raw_gaps = data.get("raw_gaps")

  for field_name, values in (("raw_times", raw_times), ("raw_costs", raw_costs), ("raw_seeds", raw_seeds)):
    if not isinstance(values, list):
      logging.warning(f"Checkpoint {filepath.name} field '{field_name}' must be a list")
      return False

  if not (len(raw_times) == len(raw_costs) == len(raw_seeds)):
    logging.warning(f"Checkpoint {filepath.name} has inconsistent array lengths")
    return False

  if len(raw_times) != successful_runs:
    logging.warning(f"Checkpoint {filepath.name} array length mismatch")
    return False

  if raw_gaps is not None and (not isinstance(raw_gaps, list) or len(raw_gaps) != successful_runs):
    logging.warning(f"Checkpoint {filepath.name} has inconsistent raw_gaps length")
    return False

  raw_phase_timings = data.get("raw_phase_timings")
  if raw_phase_timings is not None and (
    not isinstance(raw_phase_timings, list) or len(raw_phase_timings) != successful_runs
  ):
    actual_desc = len(raw_phase_timings) if isinstance(raw_phase_timings, list) else type(raw_phase_timings).__name__
    logging.warning(
      f"Checkpoint {filepath.name} raw_phase_timings length mismatch: expected {successful_runs}, got {actual_desc}"
    )
    return False

  raw_is_feasible = data.get("raw_is_feasible")
  if raw_is_feasible is not None and (
    not isinstance(raw_is_feasible, list)
    or len(raw_is_feasible) != successful_runs
    or not all(isinstance(v, bool) for v in raw_is_feasible)
  ):
    logging.warning(f"Checkpoint {filepath.name} has invalid raw_is_feasible array")
    return False

  raw_unpenalized_costs = data.get("raw_unpenalized_costs")
  if raw_unpenalized_costs is not None and (
    not isinstance(raw_unpenalized_costs, list)
    or len(raw_unpenalized_costs) != successful_runs
    or not all(
      not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))
      for v in raw_unpenalized_costs
    )
  ):
    logging.warning(f"Checkpoint {filepath.name} has invalid raw_unpenalized_costs array")
    return False

  for viol_key in ("raw_violations", "raw_normalized_violations"):
    viol_arr = data.get(viol_key)
    if viol_arr is not None and (
      not isinstance(viol_arr, list)
      or len(viol_arr) != successful_runs
      or not all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0.0
        for v in viol_arr
      )
    ):
      logging.warning(f"Checkpoint {filepath.name} has invalid {viol_key} array")
      return False

  raw_num_vehicles = data.get("raw_num_vehicles")
  if raw_num_vehicles is not None and (
    not isinstance(raw_num_vehicles, list)
    or len(raw_num_vehicles) != successful_runs
    or not all(not isinstance(v, bool) and isinstance(v, int) and v >= 0 for v in raw_num_vehicles)
  ):
    logging.warning(f"Checkpoint {filepath.name} has invalid raw_num_vehicles array")
    return False

  return True


def is_valid_checkpoint(filepath: Path, expected_reps: int | None = None) -> bool:
  """Validate checkpoint file integrity and completeness."""
  if not filepath.exists():
    return False

  try:
    with open(filepath) as f:
      data = json.load(f)

    if not isinstance(data, dict):
      logging.warning(f"Checkpoint {filepath.name} payload must be a JSON object")
      return False

    required_fields = [
      "algorithm",
      "repetitions",
      "successful_runs",
      "mean_time",
      "mean_cost",
      "raw_times",
      "raw_costs",
      "raw_seeds",
    ]
    if not all(field in data for field in required_fields):
      logging.warning(f"Checkpoint {filepath.name} missing required fields")
      return False

    target_reps: int | None = _resolve_expected_reps(data, expected_reps)
    if target_reps is None:
      logging.warning(f"Checkpoint {filepath.name} has no usable 'repetitions' value")
      return False

    successful_runs = data["successful_runs"]
    if isinstance(successful_runs, bool) or not isinstance(successful_runs, int):
      logging.warning(f"Checkpoint {filepath.name} has invalid 'successful_runs': {successful_runs!r}")
      return False

    if successful_runs < target_reps:
      logging.debug(f"Checkpoint {filepath.name} incomplete: {successful_runs}/{target_reps} successful runs")
      return False

    if data["mean_cost"] is None:
      feasible_runs = data.get("feasible_runs")
      raw_is_feasible = data.get("raw_is_feasible")
      zero_feasible = feasible_runs == 0 or (isinstance(raw_is_feasible, list) and not any(raw_is_feasible))
      if not zero_feasible:
        logging.warning(f"Checkpoint {filepath.name} has null 'mean_cost' without feasible_runs == 0")
        return False

    return _validate_checkpoint_arrays(filepath, data)

  except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
    logging.warning(f"Checkpoint {filepath.name} corrupted: {e}")
    return False


def can_resume_checkpoint(filepath: Path, expected_reps: int | None = None) -> bool:
  """Check if checkpoint can be resumed and extended."""
  if not filepath.exists():
    return False

  try:
    with open(filepath) as f:
      data = json.load(f)

    if not isinstance(data, dict):
      return False

    required_fields = ["algorithm", "repetitions", "successful_runs", "raw_times", "raw_costs", "raw_seeds"]
    if not all(field in data for field in required_fields):
      return False

    target_reps: int | None = _resolve_expected_reps(data, expected_reps)
    if target_reps is None:
      logging.warning(f"Checkpoint {filepath.name} has no usable 'repetitions' value")
      return False

    successful_runs = data["successful_runs"]
    if isinstance(successful_runs, bool) or not isinstance(successful_runs, int):
      return False

    if successful_runs >= target_reps:
      return False

    return _validate_checkpoint_arrays(filepath, data)

  except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
    logging.debug(f"Cannot resume checkpoint {filepath.name}: {e}")
    return False


def get_remaining_repetitions(filepath: Path, expected_reps: int = 30) -> int:
  """Calculate how many additional repetitions are needed."""
  if not can_resume_checkpoint(filepath, expected_reps):
    if is_valid_checkpoint(filepath, expected_reps):
      return 0
    return expected_reps

  try:
    with open(filepath) as f:
      data = json.load(f)
    completed = data["successful_runs"]
    return max(0, expected_reps - int(completed))
  except Exception:
    return expected_reps


def _config_values_match(recorded: Any, expected: Any) -> bool:
  """Compare recorded vs expected scalar/nested checkpoint config values safely."""
  if isinstance(recorded, bool) or isinstance(expected, bool):
    return isinstance(recorded, bool) and isinstance(expected, bool) and recorded == expected
  if isinstance(recorded, (int, float)) and isinstance(expected, (int, float)):
    return math.isclose(float(recorded), float(expected), rel_tol=1e-9, abs_tol=1e-12)
  return bool(recorded == expected)


def _validate_single_checkpoint_compatibility(
  path: Path,
  expected_problem: str,
  expected_algorithm: str,
  alg_cfg: AlgorithmConfig,
  ga_params: GAParams,
  param_overrides: Mapping[str, Any] | None,
  seed_override: int | None,
  max_generations: int | None = None,
  local_search_stage: str | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  *,
  allow_corrupt: bool = False,
) -> list[str]:
  """Verify one checkpoint file matches the modular schema and active algorithm configuration."""
  try:
    with open(path) as f:
      data: Any = json.load(f)
  except json.JSONDecodeError as e:
    if allow_corrupt:
      logging.warning(
        "Corrupt checkpoint JSON detected during preflight at %s (%s); deferring to execution-loop quarantine.",
        path.name,
        e,
      )
      return []
    return [f"{path.name}: failed to read checkpoint ({e})"]
  except OSError as e:
    return [f"{path.name}: failed to read checkpoint ({e})"]

  if not isinstance(data, dict):
    return [f"{path.name}: checkpoint payload must be a JSON object"]

  missing: list[str] = [key for key in REQUIRED_MODULAR_CHECKPOINT_KEYS if key not in data]
  if missing:
    return [f"{path.name}: missing required keys {missing} (incompatible or pre-modular checkpoint schema)"]

  errors: list[str] = []
  if data.get("problem_name") != expected_problem:
    errors.append(f"{path.name}: problem_name is {data.get('problem_name')!r}, expected {expected_problem!r}")
  if data.get("algorithm") != expected_algorithm:
    errors.append(f"{path.name}: algorithm is {data.get('algorithm')!r}, expected {expected_algorithm!r}")

  successful_runs = data.get("successful_runs")
  if isinstance(successful_runs, bool) or not isinstance(successful_runs, int) or successful_runs < 0:
    errors.append(f"{path.name}: successful_runs is {successful_runs!r}, expected non-negative int")
  else:
    for arr_key in ("raw_costs", "raw_times"):
      arr_val = data.get(arr_key)
      if not isinstance(arr_val, list) or len(arr_val) != successful_runs:
        errors.append(f"{path.name}: {arr_key} length does not match successful_runs ({successful_runs})")
    for opt_arr_key in (
      "raw_seeds",
      "raw_gaps",
      "raw_phase_timings",
      "raw_is_feasible",
      "raw_unpenalized_costs",
      "raw_violations",
      "raw_normalized_violations",
      "raw_num_vehicles",
    ):
      opt_val = data.get(opt_arr_key)
      if opt_val is not None and (not isinstance(opt_val, list) or len(opt_val) != successful_runs):
        errors.append(f"{path.name}: {opt_arr_key} length does not match successful_runs ({successful_runs})")

  if max_generations is not None:
    rec_max_gen = data.get("max_generations_configured")
    if isinstance(rec_max_gen, bool) or not isinstance(rec_max_gen, int) or rec_max_gen != max_generations:
      errors.append(f"{path.name}: max_generations_configured is {rec_max_gen!r}, expected {max_generations!r}")

  recorded_strategies = data.get("strategies")
  if not isinstance(recorded_strategies, dict):
    errors.append(f"{path.name}: 'strategies' must be a mapping")
  else:
    for slot, spec in alg_cfg.strategies.items():
      expected_class = spec.get("class") if isinstance(spec, dict) else None
      if expected_class is not None and recorded_strategies.get(slot) != expected_class:
        errors.append(
          f"{path.name}: strategies.{slot} is {recorded_strategies.get(slot)!r}, expected {expected_class!r}"
        )

  recorded_cfg = data.get("algorithm_config")
  if not isinstance(recorded_cfg, dict):
    errors.append(f"{path.name}: 'algorithm_config' must be a mapping")
  else:
    expected_pop = (
      int(param_overrides["population_size"])
      if (param_overrides and "population_size" in param_overrides)
      else int(alg_cfg.params.get("population_size", ga_params.population_size))
    )
    expected_mut = (
      float(param_overrides["mutation_rate"])
      if (param_overrides and "mutation_rate" in param_overrides)
      else float(alg_cfg.params.get("mutation_rate", ga_params.mutation_rate))
    )
    expected_seed = int(seed_override) if seed_override is not None else int(alg_cfg.params.get("seed", ga_params.seed))

    rec_pop = recorded_cfg.get("population_size")
    if isinstance(rec_pop, bool) or not isinstance(rec_pop, int) or rec_pop != expected_pop:
      errors.append(f"{path.name}: algorithm_config.population_size is {rec_pop!r}, expected {expected_pop!r}")

    rec_mut = recorded_cfg.get("mutation_rate")
    if (
      isinstance(rec_mut, bool)
      or not isinstance(rec_mut, (int, float))
      or not math.isclose(float(rec_mut), expected_mut, rel_tol=1e-9, abs_tol=1e-12)
    ):
      errors.append(f"{path.name}: algorithm_config.mutation_rate is {rec_mut!r}, expected {expected_mut!r}")

    rec_seed = recorded_cfg.get("seed")
    if isinstance(rec_seed, bool) or not isinstance(rec_seed, int) or rec_seed != expected_seed:
      errors.append(f"{path.name}: algorithm_config.seed is {rec_seed!r}, expected {expected_seed!r}")

    explicit_stage = (
      local_search_stage
      or (
        str(param_overrides["local_search_stage"])
        if (param_overrides and param_overrides.get("local_search_stage"))
        else None
      )
      or (str(alg_cfg.params["local_search_stage"]) if alg_cfg.params.get("local_search_stage") else None)
    )
    if explicit_stage is not None:
      rec_stage = recorded_cfg.get("local_search_stage")
      if rec_stage != explicit_stage:
        errors.append(f"{path.name}: algorithm_config.local_search_stage is {rec_stage!r}, expected {explicit_stage!r}")
    elif not (alg_cfg.class_module and alg_cfg.class_name) and "local_search_stage" in recorded_cfg:
      rec_stage = recorded_cfg.get("local_search_stage")
      if rec_stage != "post_mutation":
        errors.append(f"{path.name}: algorithm_config.local_search_stage is {rec_stage!r}, expected 'post_mutation'")

    recorded_strategy_params = recorded_cfg.get("strategy_params")
    for slot, spec in alg_cfg.strategies.items():
      if not isinstance(spec, dict):
        continue
      expected_module = spec.get("module")
      expected_params: dict[str, Any] = dict(spec.get("params") or {})
      if strategy_overrides and slot in strategy_overrides:
        expected_params.update(strategy_overrides[slot])

      if not expected_params and expected_module is None:
        continue

      if not isinstance(recorded_strategy_params, dict):
        errors.append(
          f"{path.name}: algorithm_config.strategy_params is missing, cannot verify configured parameters for '{slot}'"
        )
        continue

      slot_params = recorded_strategy_params.get(slot)
      if not isinstance(slot_params, dict):
        errors.append(
          f"{path.name}: algorithm_config.strategy_params.{slot} is missing, expected configured parameters"
        )
        continue

      if expected_module is not None and "module" in slot_params and slot_params.get("module") != expected_module:
        errors.append(
          f"{path.name}: algorithm_config.strategy_params.{slot}.module is "
          f"{slot_params.get('module')!r}, expected {expected_module!r}"
        )

      for param_key, expected_val in expected_params.items():
        if param_key not in slot_params:
          errors.append(
            f"{path.name}: algorithm_config.strategy_params.{slot}.{param_key} is missing, expected {expected_val!r}"
          )
        elif not _config_values_match(slot_params[param_key], expected_val):
          errors.append(
            f"{path.name}: algorithm_config.strategy_params.{slot}.{param_key} is "
            f"{slot_params[param_key]!r}, expected {expected_val!r}"
          )

    expected_pipeline = (
      param_overrides.get("pipeline")
      if (param_overrides and "pipeline" in param_overrides)
      else alg_cfg.params.get("pipeline")
    )
    recorded_pipeline = recorded_cfg.get("pipeline")
    if expected_pipeline:
      if not isinstance(recorded_pipeline, list) or len(recorded_pipeline) != len(expected_pipeline):
        errors.append(
          f"{path.name}: algorithm_config.pipeline length mismatch "
          f"(recorded {len(recorded_pipeline) if isinstance(recorded_pipeline, list) else None}, "
          f"expected {len(expected_pipeline)})"
        )
      else:
        for idx, (rec_step, exp_step) in enumerate(zip(recorded_pipeline, expected_pipeline, strict=True)):
          if not isinstance(rec_step, dict) or not isinstance(exp_step, Mapping):
            errors.append(f"{path.name}: algorithm_config.pipeline[{idx}] has malformed step metadata")
            continue
          exp_cls = exp_step.get("class")
          if exp_cls is not None and rec_step.get("class") != exp_cls:
            errors.append(
              f"{path.name}: algorithm_config.pipeline[{idx}].class is {rec_step.get('class')!r}, expected {exp_cls!r}"
            )
          exp_step_params = exp_step.get("params") or {}
          rec_step_params = rec_step.get("params") or {}
          if isinstance(exp_step_params, Mapping) and isinstance(rec_step_params, Mapping):
            for p_key, p_val in exp_step_params.items():
              if isinstance(p_val, Mapping) and "class" in p_val:
                rec_nested = rec_step_params.get(p_key)
                rec_nested_cls = rec_nested.get("class") if isinstance(rec_nested, Mapping) else rec_nested
                if rec_nested_cls is not None and rec_nested_cls != p_val["class"]:
                  errors.append(
                    f"{path.name}: algorithm_config.pipeline[{idx}].params.{p_key} is "
                    f"{rec_nested_cls!r}, expected {p_val['class']!r}"
                  )
              elif p_key in rec_step_params and not _config_values_match(rec_step_params[p_key], p_val):
                errors.append(
                  f"{path.name}: algorithm_config.pipeline[{idx}].params.{p_key} is "
                  f"{rec_step_params[p_key]!r}, expected {p_val!r}"
                )
    elif recorded_pipeline:
      errors.append(f"{path.name}: algorithm_config.pipeline recorded custom steps, expected standard pipeline")

  return errors


def validate_existing_checkpoints(
  checkpoint_dir: Path,
  problem_configs: Sequence[ProblemConfig],
  selected_algorithms: Sequence[str],
  algorithm_configs: Mapping[str, AlgorithmConfig],
  ga_params: GAParams,
  *,
  param_overrides: Mapping[str, Any] | None = None,
  seed_override: int | None = None,
  cpu_size_threshold: int | None = None,
  max_generations: int | None = None,
  local_search_stage: str | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  allow_corrupt: bool = False,
) -> list[str]:
  """Validate existing checkpoints in ``checkpoint_dir`` before running or resuming."""
  if not checkpoint_dir.exists():
    return []

  errors: list[str] = []
  for problem_config in problem_configs:
    if cpu_size_threshold is None or problem_config.size <= cpu_size_threshold:
      algs_for_problem = list(selected_algorithms)
    else:
      algs_for_problem = [
        alg for alg in selected_algorithms if alg in algorithm_configs and algorithm_configs[alg].use_gpu
      ]

    for alg_name in algs_for_problem:
      if alg_name not in algorithm_configs:
        continue
      checkpoint_path = checkpoint_dir / f"{problem_config.name}_{alg_name}.json"
      if not checkpoint_path.exists():
        continue
      errors.extend(
        _validate_single_checkpoint_compatibility(
          checkpoint_path,
          problem_config.name,
          alg_name,
          algorithm_configs[alg_name],
          ga_params,
          param_overrides,
          seed_override,
          max_generations=max_generations,
          local_search_stage=local_search_stage,
          strategy_overrides=strategy_overrides,
          allow_corrupt=allow_corrupt,
        )
      )

  return errors


__all__ = [
  "REQUIRED_MODULAR_CHECKPOINT_KEYS",
  "_config_values_match",
  "_resolve_expected_reps",
  "_validate_checkpoint_arrays",
  "_validate_single_checkpoint_compatibility",
  "can_resume_checkpoint",
  "get_remaining_repetitions",
  "is_valid_checkpoint",
  "validate_existing_checkpoints",
]
