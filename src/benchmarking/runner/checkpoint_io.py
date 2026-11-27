"""Checkpoint I/O facade for the modular benchmark runner.

Provides atomic checkpoint persistence, directory loading, corruption quarantine,
and ``CheckpointManager``, while re-exporting validation helpers from
``checkpoint_validation`` and statistics persistence from ``checkpoint_stats``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from benchmarking.runner.checkpoint_stats import (
  numpy_to_json_converter,
  save_problem_statistics,
)
from benchmarking.runner.checkpoint_validation import (
  REQUIRED_MODULAR_CHECKPOINT_KEYS,
  _config_values_match,
  _resolve_expected_reps,
  _validate_checkpoint_arrays,
  _validate_single_checkpoint_compatibility,
  can_resume_checkpoint,
  get_remaining_repetitions,
  is_valid_checkpoint,
  validate_existing_checkpoints,
)

if TYPE_CHECKING:
  from collections.abc import Mapping

  from benchmarking.runner.config_loader import AlgorithmConfig


class CheckpointManager:
  """Manages checkpoint directory structure and file paths."""

  def __init__(
    self,
    checkpoint_dir: str | Path,
    problem_stats_dir: str | Path,
  ) -> None:
    """Initialize checkpoint manager with directory paths."""
    self.checkpoint_dir = Path(checkpoint_dir)
    self.problem_stats_dir = Path(problem_stats_dir)
    self.ensure_directories()

  def ensure_directories(self) -> None:
    """Create checkpoint directories if they don't exist."""
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    self.problem_stats_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Checkpoint directories ensured: {self.checkpoint_dir}, {self.problem_stats_dir}")

  def get_checkpoint_path(self, problem_name: str, algorithm_name: str) -> Path:
    """Get checkpoint file path for a ``(problem, algorithm)`` pair."""
    return self.checkpoint_dir / f"{problem_name}_{algorithm_name}.json"

  def get_problem_stats_path(self, problem_name: str) -> Path:
    """Get statistics file path for a problem."""
    return self.problem_stats_dir / f"{problem_name}_stats.json"

  def is_valid_checkpoint(self, problem_name: str, algorithm_name: str, expected_reps: int | None = None) -> bool:
    """Validate checkpoint file integrity and completeness."""
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return is_valid_checkpoint(filepath, expected_reps)

  def can_resume_checkpoint(self, problem_name: str, algorithm_name: str, expected_reps: int | None = None) -> bool:
    """Check if checkpoint exists and can be extended with more repetitions."""
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return can_resume_checkpoint(filepath, expected_reps)

  def get_remaining_repetitions(self, problem_name: str, algorithm_name: str, expected_reps: int = 30) -> int:
    """Get number of additional repetitions needed to reach target."""
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
    """Count how many checkpoints already exist."""
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


def save_checkpoint(filepath: Path, results: dict[str, Any], atomic: bool = True) -> None:
  """Save checkpoint to JSON file."""
  if atomic:
    save_checkpoint_atomic(filepath, results)
  else:
    with open(filepath, "w") as f:
      json.dump(results, f, indent=2, default=numpy_to_json_converter)
    logging.info(f"    ✓ Checkpoint saved: {filepath.name}")


def save_checkpoint_atomic(filepath: Path, results: dict[str, Any]) -> None:
  """Save checkpoint using atomic write (temp file + rename)."""
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
  """Load results from checkpoint file."""
  with open(filepath) as f:
    return cast("dict[str, Any]", json.load(f))


def load_checkpoints_dir(checkpoint_dir: Path | str) -> dict[str, dict[str, dict[str, Any]]]:
  """Load consolidated checkpoint JSON files from ``checkpoint_dir`` (read-only)."""
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
  """Quarantine a corrupted checkpoint by renaming it to ``<path>.corrupt.<timestamp>.bak``."""
  ts: int = int(time.time()) if timestamp is None else timestamp
  quarantine_path = filepath.with_name(f"{filepath.name}.corrupt.{ts}.bak")
  try:
    filepath.rename(quarantine_path)
  except OSError as e:
    logging.error(f"Failed to quarantine corrupt checkpoint {filepath} to {quarantine_path}: {e}")
  return quarantine_path


def ensure_checkpoint_dirs(
  checkpoint_dir: str | Path,
  problem_stats_dir: str | Path,
) -> tuple[Path, Path]:
  """Create checkpoint directories if they don't exist."""
  checkpoint_path = Path(checkpoint_dir)
  stats_path = Path(problem_stats_dir)
  checkpoint_path.mkdir(parents=True, exist_ok=True)
  stats_path.mkdir(parents=True, exist_ok=True)
  return checkpoint_path, stats_path


__all__ = [
  "REQUIRED_MODULAR_CHECKPOINT_KEYS",
  "CheckpointManager",
  "_config_values_match",
  "_resolve_expected_reps",
  "_validate_checkpoint_arrays",
  "_validate_single_checkpoint_compatibility",
  "can_resume_checkpoint",
  "ensure_checkpoint_dirs",
  "get_remaining_repetitions",
  "is_valid_checkpoint",
  "load_checkpoint",
  "load_checkpoints_dir",
  "numpy_to_json_converter",
  "quarantine_checkpoint",
  "save_checkpoint",
  "save_checkpoint_atomic",
  "save_problem_statistics",
  "validate_existing_checkpoints",
]
