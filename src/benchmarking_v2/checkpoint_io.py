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

Extracted from chapter4_validation.py lines 167-300.

Usage:
    from src.benchmarking_v2.checkpoint_io import (
        CheckpointManager, save_checkpoint, load_checkpoint
    )

    # Initialize checkpoint system
    manager = CheckpointManager(checkpoint_dir="results/checkpoints")

    # Save results atomically
    save_checkpoint(
        manager.get_checkpoint_path("berlin52", "HybridOptimized"),
        results_dict
    )

    # Load existing checkpoint
    if manager.is_valid_checkpoint("berlin52", "HybridOptimized", expected_reps=30):
        data = load_checkpoint(manager.get_checkpoint_path("berlin52", "HybridOptimized"))

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional


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
    ):
        """
        Initialize checkpoint manager with directory paths.

        Args:
            checkpoint_dir: Directory for checkpoint JSON files
            problem_stats_dir: Directory for problem statistics
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.problem_stats_dir = Path(problem_stats_dir)

        # Create directories on initialization
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create checkpoint directories if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.problem_stats_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(
            f"Checkpoint directories ensured: {self.checkpoint_dir}, {self.problem_stats_dir}"
        )

    def get_checkpoint_path(self, problem_name: str, algorithm_name: str) -> Path:
        """
        Get checkpoint file path for a (problem, algorithm) pair.

        Args:
            problem_name: Name of TSP problem (e.g., "berlin52")
            algorithm_name: Name of algorithm (e.g., "HybridOptimized")

        Returns:
            Path to checkpoint JSON file

        Example:
            >>> manager = CheckpointManager()
            >>> path = manager.get_checkpoint_path("berlin52", "HybridOptimized")
            >>> print(path)
            results/checkpoints/berlin52_HybridOptimized.json
        """
        return self.checkpoint_dir / f"{problem_name}_{algorithm_name}.json"

    def get_problem_stats_path(self, problem_name: str) -> Path:
        """
        Get statistics file path for a problem.

        Args:
            problem_name: Name of TSP problem

        Returns:
            Path to problem statistics JSON file

        Example:
            >>> manager = CheckpointManager()
            >>> path = manager.get_problem_stats_path("berlin52")
            >>> print(path)
            results/problem_statistics/berlin52_stats.json
        """
        return self.problem_stats_dir / f"{problem_name}_stats.json"

    def is_valid_checkpoint(
        self, problem_name: str, algorithm_name: str, expected_reps: int = 30
    ) -> bool:
        """
        Validate checkpoint file integrity and completeness.

        Checks:
        - File exists
        - Valid JSON format
        - Required fields present
        - Expected number of repetitions completed

        Args:
            problem_name: Name of TSP problem
            algorithm_name: Name of algorithm
            expected_reps: Expected number of repetitions (default: 30)

        Returns:
            True if checkpoint is valid and complete

        Example:
            >>> manager = CheckpointManager()
            >>> if manager.is_valid_checkpoint("berlin52", "CPU", expected_reps=30):
            ...     print("Checkpoint valid, skipping run")
            ... else:
            ...     print("Need to run benchmark")
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
            
        Example:
            >>> manager = CheckpointManager()
            >>> if manager.can_resume_checkpoint("berlin52", "CPU", expected_reps=30):
            ...     remaining = manager.get_remaining_repetitions("berlin52", "CPU", 30)
            ...     print(f"Need to run {remaining} more repetitions")
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
            Number of additional runs needed (0 if checkpoint is complete or invalid)
            
        Example:
            >>> manager = CheckpointManager()
            >>> remaining = manager.get_remaining_repetitions("berlin52", "CPU", 30)
            >>> if remaining > 0:
            ...     print(f"Running {remaining} additional repetitions")
        """
        filepath = self.get_checkpoint_path(problem_name, algorithm_name)
        return get_remaining_repetitions(filepath, expected_reps)

    def count_completed_checkpoints(
        self,
        problems: List[Dict[str, Any]],
        algorithms: List[str],
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

        Example:
            >>> manager = CheckpointManager()
            >>> completed, total = manager.count_completed_checkpoints(
            ...     problems, ["CPU", "HybridOptimized"], expected_reps=30
            ... )
            >>> print(f"Progress: {completed}/{total} checkpoints exist")
        """
        completed = 0
        total = 0

        for problem in problems:
            problem_name = problem["name"]
            problem_size = problem["size"]

            # Determine which algorithms to check for this problem
            if problem_size <= cpu_size_threshold:
                problem_algorithms = algorithms
            else:
                # Skip CPU for large problems
                problem_algorithms = [a for a in algorithms if a != "CPU"]

            for algo in problem_algorithms:
                total += 1
                if self.is_valid_checkpoint(problem_name, algo, expected_reps):
                    completed += 1

        return completed, total


# =============================================================================
# Checkpoint I/O Functions
# =============================================================================


def numpy_to_json_converter(obj):
    """
    Convert numpy types to JSON-serializable Python types.

    Handles numpy arrays (via tolist()) and numpy scalars (via item()).
    This covers all numpy dtypes: int8-64, float16-64, bool, etc.

    Args:
        obj: Object to convert (numpy array, scalar, or other)

    Returns:
        JSON-serializable Python type

    Raises:
        TypeError: If object is not JSON serializable

    Example:
        >>> import numpy as np
        >>> data = {"array": np.array([1, 2, 3]), "scalar": np.float64(3.14)}
        >>> json.dumps(data, default=numpy_to_json_converter)
        '{"array": [1, 2, 3], "scalar": 3.14}'
    """
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    elif hasattr(obj, "item"):  # numpy scalar
        val = obj.item()
        # Handle NaN: convert to null for valid JSON (RFC 8259 compliance)
        if isinstance(val, float) and (val != val):  # NaN check
            return None
        return val
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(
    filepath: Path, results: Dict[str, Any], atomic: bool = True
) -> None:
    """
    Save checkpoint to JSON file.

    Uses atomic write by default (temp file + rename) to prevent corruption
    if process crashes during write.

    Args:
        filepath: Destination checkpoint file path
        results: Results dictionary to save
        atomic: Use atomic write (default: True)

    Raises:
        RuntimeError: If save operation fails

    Example:
        >>> results = {
        ...     "algorithm": "HybridOptimized",
        ...     "mean_time": 1.23,
        ...     "raw_times": [1.2, 1.3, 1.1],
        ...     # ... other fields
        ... }
        >>> save_checkpoint(Path("results/berlin52_HybridOptimized.json"), results)
    """
    if atomic:
        save_checkpoint_atomic(filepath, results)
    else:
        # Direct write (not recommended for production)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=numpy_to_json_converter)
        logging.info(f"    ✓ Checkpoint saved: {filepath.name}")


def save_checkpoint_atomic(filepath: Path, results: Dict[str, Any]) -> None:
    """
    Save checkpoint using atomic write (temp file + rename).

    Prevents corruption if process crashes during write.

    Implementation:
    1. Write to temporary file in same directory
    2. Flush Python buffers
    3. Sync OS buffers to disk (fsync)
    4. Atomic rename (POSIX guarantees atomicity)

    Args:
        filepath: Destination checkpoint file
        results: Results dictionary to save

    Raises:
        RuntimeError: If save operation fails

    Example:
        >>> results = {"algorithm": "CPU", "mean_time": 12.5, ...}
        >>> save_checkpoint_atomic(Path("results/berlin52_CPU.json"), results)
    """
    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent, prefix=f".tmp_{filepath.stem}_", suffix=".json"
    )

    try:
        with os.fdopen(temp_fd, "w") as f:
            # Use default converter to handle all numpy types
            json.dump(results, f, indent=2, default=numpy_to_json_converter)
            f.flush()  # Flush Python buffers
            os.fsync(f.fileno())  # Ensure OS writes to disk

        # Atomic rename (POSIX guarantees atomicity)
        os.replace(temp_path, filepath)
        logging.info(f"    ✓ Checkpoint saved: {filepath.name}")

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise RuntimeError(f"Failed to save checkpoint: {e}") from e


def load_checkpoint(filepath: Path) -> Dict[str, Any]:
    """
    Load results from checkpoint file.

    Args:
        filepath: Checkpoint file to load

    Returns:
        Results dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> data = load_checkpoint(Path("results/berlin52_CPU.json"))
        >>> print(f"Algorithm: {data['algorithm']}")
        >>> print(f"Mean time: {data['mean_time']:.2f}s")
    """
    with open(filepath, "r") as f:
        return json.load(f)


def is_valid_checkpoint(filepath: Path, expected_reps: int = 30) -> bool:
    """
    Validate checkpoint file integrity and completeness.

    Checks:
    - File exists
    - Valid JSON format
    - All required fields present
    - Expected number of repetitions completed

    Args:
        filepath: Path to checkpoint file
        expected_reps: Expected number of repetitions (default: 30)

    Returns:
        True if checkpoint exists, is valid JSON, has all required fields,
        and has expected number of successful runs

    Example:
        >>> if is_valid_checkpoint(Path("results/berlin52_CPU.json"), expected_reps=30):
        ...     print("Checkpoint valid")
        ... else:
        ...     print("Need to rerun")
    """
    if not filepath.exists():
        return False

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Check structure - checkpoint contains summary statistics, not raw repetitions
        required_fields = [
            "algorithm",
            "repetitions",
            "successful_runs",
            "mean_time",
            "mean_cost",
            "raw_times",
            "raw_costs",
            "raw_gaps",
            "raw_seeds",  # NEW: seed tracking for reproducibility
        ]
        if not all(field in data for field in required_fields):
            logging.warning(f"Checkpoint {filepath.name} missing required fields")
            return False

        # Check repetitions - must have at least expected_reps
        if data["successful_runs"] < expected_reps:
            logging.debug(
                f"Checkpoint {filepath.name} incomplete: "
                f"{data['successful_runs']}/{expected_reps} successful runs"
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

    Args:
        checkpoint_dir: Directory for checkpoint files
        problem_stats_dir: Directory for problem statistics

    Returns:
        Tuple of (checkpoint_dir_path, problem_stats_dir_path)

    Example:
        >>> checkpoint_dir, stats_dir = ensure_checkpoint_dirs()
        >>> print(f"Checkpoints: {checkpoint_dir}")
        >>> print(f"Statistics: {stats_dir}")
    """
    checkpoint_path = Path(checkpoint_dir)
    stats_path = Path(problem_stats_dir)

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    stats_path.mkdir(parents=True, exist_ok=True)

    return checkpoint_path, stats_path


def can_resume_checkpoint(filepath: Path, expected_reps: int = 30) -> bool:
    """
    Check if checkpoint can be resumed and extended.
    
    Returns True if checkpoint:
    - Exists and is valid JSON
    - Has all required fields
    - Has fewer successful runs than expected_reps
    - Has valid data that can be extended
    
    Args:
        filepath: Path to checkpoint file
        expected_reps: Target number of repetitions
        
    Returns:
        True if checkpoint can be extended with additional runs
        
    Example:
        >>> if can_resume_checkpoint(Path("berlin52_CPU.json"), expected_reps=30):
        ...     existing_data = load_checkpoint(Path("berlin52_CPU.json"))
        ...     remaining = expected_reps - existing_data["successful_runs"]
        ...     print(f"Need {remaining} more runs")
    """
    if not filepath.exists():
        return False
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = [
            "algorithm",
            "repetitions",
            "successful_runs",
            "raw_times",
            "raw_costs",
            "raw_gaps",
            "raw_seeds",  # NEW: seed tracking
        ]
        if not all(field in data for field in required_fields):
            return False
        
        # Check if extension is needed
        if data["successful_runs"] >= expected_reps:
            return False  # Already complete or over-complete
        
        # Validate that raw data arrays are consistent
        raw_times = data["raw_times"]
        raw_costs = data["raw_costs"]
        raw_gaps = data["raw_gaps"]
        raw_seeds = data["raw_seeds"]
        
        if not (len(raw_times) == len(raw_costs) == len(raw_gaps) == len(raw_seeds)):
            logging.warning(
                f"Checkpoint {filepath.name} has inconsistent array lengths"
            )
            return False
        
        # Check that successful_runs matches array length
        if len(raw_times) != data["successful_runs"]:
            logging.warning(
                f"Checkpoint {filepath.name} array length mismatch"
            )
            return False
        
        return True
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logging.debug(f"Cannot resume checkpoint {filepath.name}: {e}")
        return False


def get_remaining_repetitions(filepath: Path, expected_reps: int = 30) -> int:
    """
    Calculate how many additional repetitions are needed.
    
    Args:
        filepath: Path to checkpoint file
        expected_reps: Target number of repetitions
        
    Returns:
        Number of additional runs needed (0 if checkpoint doesn't exist or is invalid)
        
    Example:
        >>> remaining = get_remaining_repetitions(Path("berlin52_CPU.json"), 30)
        >>> print(f"Need to run {remaining} more times")
    """
    if not can_resume_checkpoint(filepath, expected_reps):
        # If checkpoint is complete, return 0
        if is_valid_checkpoint(filepath, expected_reps):
            return 0
        # Otherwise need to run all reps
        return expected_reps
    
    try:
        data = load_checkpoint(filepath)
        completed = data["successful_runs"]
        return max(0, expected_reps - completed)
    except Exception:
        return expected_reps
