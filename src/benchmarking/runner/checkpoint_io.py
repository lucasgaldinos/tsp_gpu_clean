"""
Checkpoint I/O Module for Benchmark V2
=======================================

Manages checkpoint file operations for the modular benchmark system.

This module provides fault-tolerant checkpoint management, enabling:
- Atomic checkpoint writes (prevents corruption)
- Checkpoint validation (integrity checks)
- Resume capability (skip completed runs)
- Incremental checkpoint support (extend existing runs with additional repetitions)
- Progress tracking across benchmark sessions

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Sequence

  from benchmarking.runner.config_loader import AlgorithmConfig, GAParams, ProblemConfig

# =============================================================================
# Checkpoint Manager Class
# =============================================================================


class CheckpointManager:
  """
  Manages checkpoint directory structure and file paths.

  Attributes:
      checkpoint_dir: Root directory for checkpoint files
      problem_stats_dir: Directory for per-problem statistics
  """

  def __init__(
    self,
    checkpoint_dir: str | Path,
    problem_stats_dir: str | Path,
  ) -> None:
    """
    Initialize checkpoint manager with directory paths.

    Args:
        checkpoint_dir: Directory for checkpoint JSON files
        problem_stats_dir: Directory for problem statistics
    """
    self.checkpoint_dir = Path(checkpoint_dir)
    self.problem_stats_dir = Path(problem_stats_dir)
    self.ensure_directories()

  def ensure_directories(self) -> None:
    """Create checkpoint directories if they don't exist."""
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    self.problem_stats_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Checkpoint directories ensured: {self.checkpoint_dir}, {self.problem_stats_dir}")

  def get_checkpoint_path(self, problem_name: str, algorithm_name: str) -> Path:
    """
    Get checkpoint file path for a (problem, algorithm) pair.

    Args:
        problem_name: Name of TSP problem (e.g., "berlin52")
        algorithm_name: Name of algorithm (e.g., "MemeticGA")

    Returns:
        Path to checkpoint JSON file
    """
    return self.checkpoint_dir / f"{problem_name}_{algorithm_name}.json"

  def get_problem_stats_path(self, problem_name: str) -> Path:
    """
    Get statistics file path for a problem.

    Args:
        problem_name: Name of TSP problem

    Returns:
        Path to problem statistics JSON file
    """
    return self.problem_stats_dir / f"{problem_name}_stats.json"

  def is_valid_checkpoint(self, problem_name: str, algorithm_name: str, expected_reps: int | None = None) -> bool:
    """
    Validate checkpoint file integrity and completeness.

    Args:
        problem_name: Name of TSP problem
        algorithm_name: Name of algorithm
        expected_reps: Expected number of repetitions; ``None`` reads the
            checkpoint's own ``repetitions`` field

    Returns:
        True if checkpoint is valid and complete
    """
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return is_valid_checkpoint(filepath, expected_reps)

  def can_resume_checkpoint(self, problem_name: str, algorithm_name: str, expected_reps: int | None = None) -> bool:
    """
    Check if checkpoint exists and can be extended with more repetitions.

    Args:
        problem_name: Name of TSP problem
        algorithm_name: Name of algorithm
        expected_reps: Target number of repetitions

    Returns:
        True if checkpoint exists with fewer than expected_reps and can be extended
    """
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return can_resume_checkpoint(filepath, expected_reps)

  def get_remaining_repetitions(self, problem_name: str, algorithm_name: str, expected_reps: int = 30) -> int:
    """
    Get number of additional repetitions needed to reach target.

    Args:
        problem_name: Name of TSP problem
        algorithm_name: Name of algorithm
        expected_reps: Target number of repetitions

    Returns:
        Number of additional runs needed
    """
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return get_remaining_repetitions(filepath, expected_reps)

  def count_completed_checkpoints(
    self,
    problems: list[dict[str, Any]],
    algorithms: list[str],
    expected_reps: int = 30,
    cpu_size_threshold: int | None = None,
    algorithm_configs: Mapping[str, AlgorithmConfig] | None = None,
  ) -> tuple[int, int]:
    """
    Count how many checkpoints already exist.

    Args:
        problems: List of problem configurations (dicts with 'name' and 'size')
        algorithms: List of algorithm names
        expected_reps: Expected repetitions for validation
        cpu_size_threshold: Skip non-GPU algorithms for problems larger than this (None = all instances)
        algorithm_configs: Optional mapping of algorithm names to AlgorithmConfig (uses ``use_gpu`` flag)

    Returns:
        Tuple of (completed_count, total_count)
    """
    completed = 0
    total = 0

    for problem in problems:
      problem_name = problem["name"]
      problem_size = problem["size"]

      if cpu_size_threshold is None or problem_size <= cpu_size_threshold:
        problem_algorithms = algorithms
      elif algorithm_configs is not None:
        problem_algorithms = [a for a in algorithms if a in algorithm_configs and algorithm_configs[a].use_gpu]
      else:
        problem_algorithms = [a for a in algorithms if not a.upper().startswith("CPU")]

      for algo in problem_algorithms:
        total += 1
        if self.is_valid_checkpoint(problem_name, algo, expected_reps):
          completed += 1

    return completed, total


# =============================================================================
# Checkpoint I/O Functions
# =============================================================================


def numpy_to_json_converter(obj: Any) -> Any:
  """
  Convert numpy types to JSON-serializable Python types.
  """
  if hasattr(obj, "tolist"):
    return obj.tolist()
  elif hasattr(obj, "item"):
    val = obj.item()
    if isinstance(val, float) and (val != val):
      return None
    return val
  raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(filepath: Path, results: dict[str, Any], atomic: bool = True) -> None:
  """
  Save checkpoint to JSON file.
  """
  if atomic:
    save_checkpoint_atomic(filepath, results)
  else:
    with open(filepath, "w") as f:
      json.dump(results, f, indent=2, default=numpy_to_json_converter)
    logging.info(f"    ✓ Checkpoint saved: {filepath.name}")


def save_checkpoint_atomic(filepath: Path, results: dict[str, Any]) -> None:
  """
  Save checkpoint using atomic write (temp file + rename).
  """
  temp_fd, temp_path = tempfile.mkstemp(dir=filepath.parent, prefix=f".tmp_{filepath.stem}_", suffix=".json")

  try:
    with os.fdopen(temp_fd, "w") as f:
      json.dump(results, f, indent=2, default=numpy_to_json_converter)
      f.flush()
      os.fsync(f.fileno())

    os.replace(temp_path, filepath)
    logging.info(f"    ✓ Checkpoint saved: {filepath.name}")

  except Exception as e:
    try:
      os.unlink(temp_path)
    except Exception as cleanup_error:
      logging.warning(f"Failed to remove temp checkpoint file {temp_path}: {cleanup_error}")
    raise RuntimeError(f"Failed to save checkpoint: {e}") from e


def load_checkpoint(filepath: Path) -> dict[str, Any]:
  """
  Load results from checkpoint file.
  """
  with open(filepath) as f:
    return cast("dict[str, Any]", json.load(f))


def load_checkpoints_dir(checkpoint_dir: Path | str) -> dict[str, dict[str, dict[str, Any]]]:
  """Load consolidated checkpoint JSON files from ``checkpoint_dir`` (read-only).

  Accepts either a directory containing ``*.json`` checkpoint files or a parent
  benchmark output directory that contains a ``checkpoints/`` subdirectory.

  Args:
      checkpoint_dir: Path to the checkpoints directory (or parent results directory).

  Returns:
      Nested mapping ``problem_name -> algorithm_name -> checkpoint_payload``.

  Raises:
      FileNotFoundError: If ``checkpoint_dir`` does not exist.
  """
  ckpt_path = Path(checkpoint_dir)
  if not ckpt_path.is_dir():
    raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")

  if not any(ckpt_path.glob("*.json")) and (ckpt_path / "checkpoints").is_dir():
    ckpt_path = ckpt_path / "checkpoints"

  all_results: dict[str, dict[str, dict[str, Any]]] = {}
  for json_file in sorted(ckpt_path.glob("*.json")):
    try:
      with json_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
      logging.warning("Skipping unreadable checkpoint %s: %s", json_file.name, exc)
      continue

    if not isinstance(payload, dict):
      continue
    problem_name = payload.get("problem_name")
    algorithm = payload.get("algorithm")
    if not isinstance(problem_name, str) or not isinstance(algorithm, str):
      continue

    all_results.setdefault(problem_name, {})[algorithm] = payload

  return all_results


def quarantine_checkpoint(filepath: Path, timestamp: int | None = None) -> Path:
  """Quarantine a corrupted checkpoint by renaming it to <path>.corrupt.<timestamp>.bak.

  Args:
      filepath: Path to the corrupted checkpoint file.
      timestamp: Optional epoch timestamp for deterministic testing.

  Returns:
      The Path to the quarantined file.
  """
  ts: int = int(time.time()) if timestamp is None else timestamp
  quarantine_path = filepath.with_name(f"{filepath.name}.corrupt.{ts}.bak")
  try:
    filepath.rename(quarantine_path)
  except OSError as e:
    logging.error(f"Failed to quarantine corrupt checkpoint {filepath} to {quarantine_path}: {e}")
  return quarantine_path


def _resolve_expected_reps(data: dict[str, Any], expected_reps: int | None) -> int | None:
  """Return the repetition target a checkpoint must reach, or ``None`` if unknown.

  ``benchmark.json`` configures a run-specific count (15 in the chapter benchmark,
  3 in ``selected_10``), so a hard-coded default of 30 marked every real
  checkpoint as incomplete. When the caller names no target, the checkpoint's own
  ``repetitions`` field is the only honest source. A missing, non-integer or
  non-positive value means the file cannot be judged and must not be accepted.

  Args:
      data: Decoded checkpoint payload.
      expected_reps: Caller-supplied target, or ``None`` to read it from ``data``.

  Returns:
      The repetition target, or ``None`` when it cannot be determined.
  """
  if expected_reps is not None:
    return expected_reps

  reps: Any = data.get("repetitions")
  if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
    return None
  return reps


def _validate_checkpoint_arrays(filepath: Path, data: Mapping[str, Any]) -> bool:
  """Validate per-repetition raw array lengths against ``successful_runs``.

  Accepts ``raw_gaps=None`` (or omitted) for instances without a known optimal tour
  cost (``BUG-96``), while enforcing ``len(raw_gaps) == successful_runs`` whenever
  ``raw_gaps`` is present.
  """
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

  if raw_gaps is not None:
    if not isinstance(raw_gaps, list) or len(raw_gaps) != successful_runs:
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

  return True


def is_valid_checkpoint(filepath: Path, expected_reps: int | None = None) -> bool:
  """
  Validate checkpoint file integrity and completeness.

  Args:
      filepath: Path to the checkpoint JSON.
      expected_reps: Target repetitions; when ``None``, the value recorded in the
          checkpoint's own ``repetitions`` field is used.
  """
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

    return _validate_checkpoint_arrays(filepath, data)

  except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
    logging.warning(f"Checkpoint {filepath.name} corrupted: {e}")
    return False


# =============================================================================
# Convenience Functions
# =============================================================================


def ensure_checkpoint_dirs(
  checkpoint_dir: str | Path,
  problem_stats_dir: str | Path,
) -> tuple[Path, Path]:
  """
  Create checkpoint directories if they don't exist.
  """
  checkpoint_path = Path(checkpoint_dir)
  stats_path = Path(problem_stats_dir)

  checkpoint_path.mkdir(parents=True, exist_ok=True)
  stats_path.mkdir(parents=True, exist_ok=True)

  return checkpoint_path, stats_path


def can_resume_checkpoint(filepath: Path, expected_reps: int | None = None) -> bool:
  """
  Check if checkpoint can be resumed and extended.

  Args:
      filepath: Path to the checkpoint JSON.
      expected_reps: Target repetitions; when ``None``, the value recorded in the
          checkpoint's own ``repetitions`` field is used.
  """
  if not filepath.exists():
    return False

  try:
    with open(filepath) as f:
      data = json.load(f)

    if not isinstance(data, dict):
      return False

    required_fields = [
      "algorithm",
      "repetitions",
      "successful_runs",
      "raw_times",
      "raw_costs",
      "raw_seeds",
    ]
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
  """
  Calculate how many additional repetitions are needed.
  """
  if not can_resume_checkpoint(filepath, expected_reps):
    if is_valid_checkpoint(filepath, expected_reps):
      return 0
    return expected_reps

  try:
    data = load_checkpoint(filepath)
    completed = data["successful_runs"]
    return max(0, expected_reps - int(completed))
  except Exception:
    return expected_reps


REQUIRED_MODULAR_CHECKPOINT_KEYS: tuple[str, ...] = (
  "problem_name",
  "algorithm",
  "successful_runs",
  "raw_costs",
  "raw_times",
  "strategies",
  "algorithm_config",
)


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
    data: Any = load_checkpoint(path)
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
    for opt_arr_key in ("raw_seeds", "raw_gaps", "raw_phase_timings"):
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
  """Validate existing checkpoints in ``checkpoint_dir`` before running or resuming.

  Rejects pre-modular checkpoints, strategy/parameter/stage/pipeline mismatches, explicit
  ``max_generations`` mismatches, or GA parameter drift for any selected
  ``(problem, algorithm)`` pair so stale cohorts cannot be silently reused. When
  ``allow_corrupt=True``, malformed JSON files are logged and deferred to the execution
  loop's ``quarantine_checkpoint`` recovery path.
  """
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

      n_valid_costs = int(np.sum(~np.isnan(costs)))
      if costs.size > 0:
        mean_cost, std_cost, _, _ = compute_series_stats(costs, n_valid_costs)
      else:
        mean_cost = float(results["mean_cost"]) if results.get("mean_cost") is not None else 0.0
        std_cost = float(results["std_cost"]) if results.get("std_cost") is not None else 0.0

      mean_gap: float | None
      std_gap: float | None
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

      problem_stats["algorithms"][alg_name] = {
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

    with open(stats_path, "w", encoding="utf-8") as f:
      json.dump(problem_stats, f, indent=2, default=numpy_to_json_converter)
    logging.info(f"✓ Problem statistics saved: {stats_path.name}\n")
  except Exception as e:
    logging.warning(f"Failed to save problem statistics: {e}\n")
