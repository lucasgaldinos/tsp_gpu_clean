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
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

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
        algorithm_name: Name of algorithm (e.g., "HybridOptimized")

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

    required_fields = [
      "algorithm",
      "repetitions",
      "successful_runs",
      "mean_time",
      "mean_cost",
      "raw_times",
      "raw_costs",
      "raw_gaps",
      "raw_seeds",
    ]
    if not all(field in data for field in required_fields):
      logging.warning(f"Checkpoint {filepath.name} missing required fields")
      return False

    target_reps: int | None = _resolve_expected_reps(data, expected_reps)
    if target_reps is None:
      logging.warning(f"Checkpoint {filepath.name} has no usable 'repetitions' value")
      return False

    if data["successful_runs"] < target_reps:
      logging.debug(f"Checkpoint {filepath.name} incomplete: {data['successful_runs']}/{target_reps} successful runs")
      return False

    raw_times = data["raw_times"]
    raw_costs = data["raw_costs"]
    raw_gaps = data["raw_gaps"]
    raw_seeds = data["raw_seeds"]

    if not (len(raw_times) == len(raw_costs) == len(raw_gaps) == len(raw_seeds)):
      logging.warning(f"Checkpoint {filepath.name} has inconsistent array lengths")
      return False

    if len(raw_times) != data["successful_runs"]:
      logging.warning(f"Checkpoint {filepath.name} array length mismatch")
      return False

    if "raw_phase_timings" in data and len(data["raw_phase_timings"]) != data["successful_runs"]:
      logging.warning(
        f"Checkpoint {filepath.name} raw_phase_timings length mismatch: "
        f"expected {data['successful_runs']}, got {len(data['raw_phase_timings'])}"
      )
      return False

    return True

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

    required_fields = [
      "algorithm",
      "repetitions",
      "successful_runs",
      "raw_times",
      "raw_costs",
      "raw_gaps",
      "raw_seeds",
    ]
    if not all(field in data for field in required_fields):
      return False

    target_reps: int | None = _resolve_expected_reps(data, expected_reps)
    if target_reps is None:
      logging.warning(f"Checkpoint {filepath.name} has no usable 'repetitions' value")
      return False

    if data["successful_runs"] >= target_reps:
      return False

    raw_times = data["raw_times"]
    raw_costs = data["raw_costs"]
    raw_gaps = data["raw_gaps"]
    raw_seeds = data["raw_seeds"]

    if not (len(raw_times) == len(raw_costs) == len(raw_gaps) == len(raw_seeds)):
      logging.warning(f"Checkpoint {filepath.name} has inconsistent array lengths")
      return False

    if len(raw_times) != data["successful_runs"]:
      logging.warning(f"Checkpoint {filepath.name} array length mismatch")
      return False

    if "raw_phase_timings" in data and len(data["raw_phase_timings"]) != data["successful_runs"]:
      logging.warning(
        f"Checkpoint {filepath.name} raw_phase_timings length mismatch: "
        f"expected {data['successful_runs']}, got {len(data['raw_phase_timings'])}"
      )
      return False

    return True

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


def _validate_single_checkpoint_compatibility(
  path: Path,
  expected_problem: str,
  expected_algorithm: str,
  alg_cfg: AlgorithmConfig,
  ga_params: GAParams,
  param_overrides: Mapping[str, Any] | None,
  seed_override: int | None,
) -> list[str]:
  """Verify one checkpoint file matches the modular schema and active algorithm configuration."""
  try:
    data: Any = load_checkpoint(path)
  except (json.JSONDecodeError, OSError) as e:
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
    expected_seed = (
      int(seed_override)
      if seed_override is not None
      else int(alg_cfg.params.get("seed", ga_params.seed))
    )

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
) -> list[str]:
  """Validate existing checkpoints in ``checkpoint_dir`` before running or resuming.

  Rejects pre-modular checkpoints, strategy mismatches, or GA parameter drift
  for any selected ``(problem, algorithm)`` pair so stale cohorts cannot be
  silently reused.
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
        )
      )

  return errors
