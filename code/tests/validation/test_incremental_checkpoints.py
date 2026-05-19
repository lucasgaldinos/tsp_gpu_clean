"""
Test Suite for Incremental Checkpoint Feature
==============================================

Tests the incremental repetition appending functionality for benchmarks.

Tests cover:
- Seed tracking and storage
- Incremental run appending
- Statistics recalculation
- Validation edge cases
- Config flag handling

Author: GPU Accelerated Routing Optimization Team
Date: 2025-01-28
"""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Add parent directories to path for imports (go up to code/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.benchmarking_v2.checkpoint_io import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    is_valid_checkpoint,
    can_resume_checkpoint,
    get_remaining_repetitions,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for test checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_checkpoint_data():
    """Create sample checkpoint data with seeds."""
    return {
        "algorithm": "TestAlgo",
        "problem_name": "test52",
        "problem_size": 52,
        "optimal_cost": 7542.0,
        "repetitions": 5,
        "successful_runs": 5,
        "mean_time": 1.23,
        "std_time": 0.15,
        "mean_cost": 7600.0,
        "std_cost": 20.0,
        "mean_gap": 0.77,
        "std_gap": 0.15,
        "raw_times": [1.1, 1.2, 1.3, 1.2, 1.25],
        "raw_costs": [7580, 7590, 7610, 7600, 7620],
        "raw_gaps": [0.50, 0.64, 0.90, 0.77, 1.03],
        "raw_seeds": [42, 123, 456, 789, 1011],  # NEW: seed tracking
    }


# =============================================================================
# Test Checkpoint Schema with Seeds
# =============================================================================


def test_checkpoint_schema_includes_seeds(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that checkpoint schema includes seed tracking."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    save_checkpoint(filepath, sample_checkpoint_data)
    loaded = load_checkpoint(filepath)
    
    assert "raw_seeds" in loaded
    assert loaded["raw_seeds"] == [42, 123, 456, 789, 1011]
    assert len(loaded["raw_seeds"]) == loaded["successful_runs"]


def test_checkpoint_seed_array_consistency(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that seed array length matches other raw arrays."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    save_checkpoint(filepath, sample_checkpoint_data)
    loaded = load_checkpoint(filepath)
    
    assert len(loaded["raw_seeds"]) == len(loaded["raw_times"])
    assert len(loaded["raw_seeds"]) == len(loaded["raw_costs"])
    assert len(loaded["raw_seeds"]) == len(loaded["raw_gaps"])


# =============================================================================
# Test Incremental Append Logic
# =============================================================================


def test_incremental_append_extends_arrays(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that incremental append correctly extends all arrays."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Save initial checkpoint (5 runs)
    save_checkpoint(filepath, sample_checkpoint_data)
    
    # Simulate appending 3 more runs
    loaded = load_checkpoint(filepath)
    new_runs = {
        "raw_times": [1.15, 1.28, 1.19],
        "raw_costs": [7595, 7605, 7585],
        "raw_gaps": [0.70, 0.83, 0.57],
        "raw_seeds": [2022, 3033, 4044],
    }
    
    # Append new data
    loaded["raw_times"].extend(new_runs["raw_times"])
    loaded["raw_costs"].extend(new_runs["raw_costs"])
    loaded["raw_gaps"].extend(new_runs["raw_gaps"])
    loaded["raw_seeds"].extend(new_runs["raw_seeds"])
    loaded["successful_runs"] = len(loaded["raw_times"])
    loaded["repetitions"] = loaded["successful_runs"]
    
    # Recalculate statistics
    loaded["mean_time"] = np.mean(loaded["raw_times"])
    loaded["std_time"] = np.std(loaded["raw_times"], ddof=1)
    loaded["mean_cost"] = np.mean(loaded["raw_costs"])
    loaded["std_cost"] = np.std(loaded["raw_costs"], ddof=1)
    loaded["mean_gap"] = np.mean(loaded["raw_gaps"])
    loaded["std_gap"] = np.std(loaded["raw_gaps"], ddof=1)
    
    save_checkpoint(filepath, loaded)
    
    # Verify appended checkpoint
    final = load_checkpoint(filepath)
    assert final["successful_runs"] == 8
    assert len(final["raw_times"]) == 8
    assert len(final["raw_seeds"]) == 8
    assert final["raw_seeds"] == [42, 123, 456, 789, 1011, 2022, 3033, 4044]


def test_incremental_statistics_recalculation(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that statistics are correctly recalculated after append."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Initial checkpoint
    save_checkpoint(filepath, sample_checkpoint_data)
    original_mean = sample_checkpoint_data["mean_time"]
    
    # Append runs
    loaded = load_checkpoint(filepath)
    loaded["raw_times"].extend([2.0, 2.5, 3.0])  # Higher values to shift mean
    loaded["raw_costs"].extend([8000, 8100, 8200])
    loaded["raw_gaps"].extend([6.07, 7.40, 8.73])
    loaded["raw_seeds"].extend([5001, 5002, 5003])
    loaded["successful_runs"] = len(loaded["raw_times"])
    
    # Recalculate
    new_mean_time = np.mean(loaded["raw_times"])
    loaded["mean_time"] = new_mean_time
    loaded["std_time"] = np.std(loaded["raw_times"], ddof=1)
    loaded["mean_cost"] = np.mean(loaded["raw_costs"])
    loaded["std_cost"] = np.std(loaded["raw_costs"], ddof=1)
    loaded["mean_gap"] = np.mean(loaded["raw_gaps"])
    loaded["std_gap"] = np.std(loaded["raw_gaps"], ddof=1)
    
    save_checkpoint(filepath, loaded)
    
    # Verify statistics changed
    final = load_checkpoint(filepath)
    assert final["mean_time"] != original_mean
    assert final["mean_time"] > original_mean  # Higher values should increase mean
    assert final["successful_runs"] == 8


# =============================================================================
# Test Validation with Incremental Mode
# =============================================================================


def test_can_resume_checkpoint_detects_incomplete(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that can_resume_checkpoint identifies incomplete checkpoints."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Create incomplete checkpoint (5 runs, expect 10)
    sample_checkpoint_data["successful_runs"] = 5
    sample_checkpoint_data["repetitions"] = 5
    save_checkpoint(filepath, sample_checkpoint_data)
    
    assert can_resume_checkpoint(filepath, expected_reps=10) is True
    assert is_valid_checkpoint(filepath, expected_reps=10) is False


def test_get_remaining_repetitions_calculates_correctly(temp_checkpoint_dir, sample_checkpoint_data):
    """Test calculation of remaining repetitions."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Checkpoint with 5 successful runs
    save_checkpoint(filepath, sample_checkpoint_data)
    
    remaining = get_remaining_repetitions(filepath, expected_reps=15)
    assert remaining == 10  # 15 - 5 = 10


def test_incremental_mode_rejects_complete_checkpoint(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that complete checkpoints are not resumable."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Complete checkpoint (30 runs)
    sample_checkpoint_data["successful_runs"] = 30
    sample_checkpoint_data["repetitions"] = 30
    save_checkpoint(filepath, sample_checkpoint_data)
    
    assert can_resume_checkpoint(filepath, expected_reps=30) is False
    assert is_valid_checkpoint(filepath, expected_reps=30) is True
    assert get_remaining_repetitions(filepath, expected_reps=30) == 0


# =============================================================================
# Test Edge Cases
# =============================================================================


def test_incremental_append_empty_checkpoint(temp_checkpoint_dir):
    """Test appending to non-existent checkpoint (should create new)."""
    filepath = temp_checkpoint_dir / "new_problem_TestAlgo.json"
    
    # Checkpoint doesn't exist
    assert not filepath.exists()
    assert can_resume_checkpoint(filepath, expected_reps=30) is False
    assert get_remaining_repetitions(filepath, expected_reps=30) == 30


def test_incremental_append_corrupted_checkpoint(temp_checkpoint_dir):
    """Test handling of corrupted checkpoint in incremental mode."""
    filepath = temp_checkpoint_dir / "corrupted_TestAlgo.json"
    
    # Create corrupted checkpoint (invalid JSON)
    with open(filepath, "w") as f:
        f.write("{ invalid json }")
    
    assert can_resume_checkpoint(filepath, expected_reps=30) is False
    assert get_remaining_repetitions(filepath, expected_reps=30) == 30


def test_incremental_append_mismatched_arrays(temp_checkpoint_dir, sample_checkpoint_data):
    """Test validation rejects checkpoints with mismatched array lengths."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Create checkpoint with mismatched arrays
    sample_checkpoint_data["raw_times"] = [1.0, 2.0, 3.0]  # 3 elements
    sample_checkpoint_data["raw_seeds"] = [1, 2]  # 2 elements - MISMATCH
    sample_checkpoint_data["successful_runs"] = 3
    
    save_checkpoint(filepath, sample_checkpoint_data)
    
    # Should fail validation
    assert can_resume_checkpoint(filepath, expected_reps=30) is False


def test_seed_uniqueness_across_increments(temp_checkpoint_dir, sample_checkpoint_data):
    """Test that seeds remain unique across incremental appends."""
    filepath = temp_checkpoint_dir / "test52_TestAlgo.json"
    
    # Save initial checkpoint
    save_checkpoint(filepath, sample_checkpoint_data)
    
    # Append with new unique seeds
    loaded = load_checkpoint(filepath)
    new_seeds = [9999, 8888, 7777]
    loaded["raw_seeds"].extend(new_seeds)
    loaded["raw_times"].extend([1.5, 1.6, 1.7])
    loaded["raw_costs"].extend([7650, 7660, 7670])
    loaded["raw_gaps"].extend([1.43, 1.56, 1.70])
    loaded["successful_runs"] = len(loaded["raw_seeds"])
    
    save_checkpoint(filepath, loaded)
    
    # Verify all seeds are present and unique
    final = load_checkpoint(filepath)
    all_seeds = final["raw_seeds"]
    assert len(all_seeds) == len(set(all_seeds))  # No duplicates
    assert 9999 in all_seeds
    assert 42 in all_seeds  # Original seed still present


# =============================================================================
# Test CheckpointManager Integration
# =============================================================================


def test_checkpoint_manager_incremental_validation(temp_checkpoint_dir, sample_checkpoint_data):
    """Test CheckpointManager methods for incremental checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    # Save checkpoint via manager path
    filepath = manager.get_checkpoint_path("test52", "TestAlgo")
    save_checkpoint(filepath, sample_checkpoint_data)
    
    # Test manager methods
    assert manager.can_resume_checkpoint("test52", "TestAlgo", expected_reps=10) is True
    assert manager.get_remaining_repetitions("test52", "TestAlgo", expected_reps=10) == 5
    assert manager.is_valid_checkpoint("test52", "TestAlgo", expected_reps=5) is True


def test_checkpoint_manager_count_with_incomplete(temp_checkpoint_dir, sample_checkpoint_data):
    """Test checkpoint counting with incomplete checkpoints."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    # Create mix of complete and incomplete checkpoints
    problems = [
        {"name": "problem1", "size": 50},
        {"name": "problem2", "size": 100},
    ]
    algorithms = ["AlgoA", "AlgoB"]
    
    # problem1-AlgoA: complete (30 runs)
    data1 = sample_checkpoint_data.copy()
    data1["successful_runs"] = 30
    save_checkpoint(manager.get_checkpoint_path("problem1", "AlgoA"), data1)
    
    # problem1-AlgoB: incomplete (10 runs)
    data2 = sample_checkpoint_data.copy()
    data2["successful_runs"] = 10
    save_checkpoint(manager.get_checkpoint_path("problem1", "AlgoB"), data2)
    
    # problem2-AlgoA: missing
    # problem2-AlgoB: complete (30 runs)
    data3 = sample_checkpoint_data.copy()
    data3["successful_runs"] = 30
    save_checkpoint(manager.get_checkpoint_path("problem2", "AlgoB"), data3)
    
    # Count completed (should only count those with >=30 runs)
    completed, total = manager.count_completed_checkpoints(
        problems, algorithms, expected_reps=30
    )
    
    assert total == 4  # 2 problems * 2 algorithms
    assert completed == 2  # problem1-AlgoA and problem2-AlgoB


# =============================================================================
# Test Random Seed Generation Pattern
# =============================================================================


def test_random_seed_generation_produces_unique_seeds():
    """Test that random seed generation produces unique seeds."""
    np.random.seed(42)  # For reproducibility in test
    
    # Generate 100 random seeds
    seeds = [np.random.randint(0, 2**31 - 1) for _ in range(100)]
    
    # Should have high uniqueness (allow for rare collisions)
    unique_seeds = len(set(seeds))
    assert unique_seeds >= 95  # At least 95% unique


def test_seed_reproducibility_within_checkpoint():
    """Test that seeds stored in checkpoint enable run reproduction."""
    # This is a design test - seeds should be stored for reproducibility
    # Actual algorithm re-execution would be integration test
    
    checkpoint = {
        "raw_seeds": [123, 456, 789],
        "raw_costs": [7600, 7550, 7650],
    }
    
    # Each run should be reproducible using its seed
    for i, seed in enumerate(checkpoint["raw_seeds"]):
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31  # Valid NumPy seed range
        # In real code: np.random.seed(seed) would reproduce run i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
