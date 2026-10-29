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

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, cast

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
    checkpoint_dir: str | Path = "results_v3/checkpoints",
    problem_stats_dir: str | Path = "results_v3/problem_statistics",
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

  def is_valid_checkpoint(
    self, problem_name: str, algorithm_name: str, expected_reps: int = 30
  ) -> bool:
    """
    Validate checkpoint file integrity and completeness.

    Args:
        problem_name: Name of TSP problem
        algorithm_name: Name of algorithm
        expected_reps: Expected number of repetitions (default: 30)

    Returns:
        True if checkpoint is valid and complete
    """
    filepath = self.get_checkpoint_path(problem_name, algorithm_name)
    return is_valid_checkpoint(filepath, expected_reps)

  def can_resume_checkpoint(
    self, problem_name: str, algorithm_name: str, expected_reps: int = 30
  ) -> bool:
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

  def get_remaining_repetitions(
    self, problem_name: str, algorithm_name: str, expected_reps: int = 30
  ) -> int:
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
    cpu_size_threshold: int = 100,
  ) -> tuple[int, int]:
    """
    Count how many checkpoints already exist.

    Args:
        problems: List of problem configurations (dicts with 'name' and 'size')
        algorithms: List of algorithm names
        expected_reps: Expected repetitions for validation
        cpu_size_threshold: Skip CPU for problems larger than this

    Returns:
        Tuple of (completed_count, total_count)
    """
    completed = 0
    total = 0

    for problem in problems:
      problem_name = problem["name"]
      problem_size = problem["size"]

      if problem_size <= cpu_size_threshold:
        problem_algorithms = algorithms
      else:
        problem_algorithms = [a for a in algorithms if a != "CPU"]

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
  temp_fd, temp_path = tempfile.mkstemp(
    dir=filepath.parent, prefix=f".tmp_{filepath.stem}_", suffix=".json"
  )

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


def is_valid_checkpoint(filepath: Path, expected_reps: int = 30) -> bool:
  """
  Validate checkpoint file integrity and completeness.
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

    if data["successful_runs"] < expected_reps:
      logging.debug(
        f"Checkpoint {filepath.name} incomplete: {data['successful_runs']}/{expected_reps} successful runs"
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
  checkpoint_dir: str | Path = "results_v3/checkpoints",
  problem_stats_dir: str | Path = "results_v3/problem_statistics",
) -> tuple[Path, Path]:
  """
  Create checkpoint directories if they don't exist.
  """
  checkpoint_path = Path(checkpoint_dir)
  stats_path = Path(problem_stats_dir)

  checkpoint_path.mkdir(parents=True, exist_ok=True)
  stats_path.mkdir(parents=True, exist_ok=True)

  return checkpoint_path, stats_path


def can_resume_checkpoint(filepath: Path, expected_reps: int = 30) -> bool:
  """
  Check if checkpoint can be resumed and extended.
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

    if data["successful_runs"] >= expected_reps:
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
