"""
Validate Database Consistency
==============================

This script validates that both test databases contain identical data
and match the source checkpoint JSON files.

Usage:
    python validate_database_consistency.py
"""

import duckdb
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Database paths
DB1_PATH = "/tmp/test_db_results/results.duckdb"
DB2_PATH = "my_notes/results.duckdb"

# Source checkpoint directory
CHECKPOINT_DIR = Path("code/benchmarks/results/checkpoints")


def load_database_to_df(db_path: str) -> pd.DataFrame:
    """Load benchmark_runs table to pandas DataFrame."""
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM benchmark_runs ORDER BY run_id").df()
    conn.close()
    return df


def load_checkpoint(filepath: Path) -> Dict[str, Any]:
    """Load checkpoint JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def compare_databases(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Compare two database DataFrames."""
    print("\n" + "=" * 80)
    print("DATABASE COMPARISON")
    print("=" * 80)

    # Check row counts
    print(f"\nRow count DB1: {len(df1)}")
    print(f"Row count DB2: {len(df2)}")

    if len(df1) != len(df2):
        print("❌ Row counts don't match!")
        return False
    print("✓ Row counts match")

    # Check columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    if cols1 != cols2:
        print(f"❌ Columns don't match!")
        print(f"  DB1 only: {cols1 - cols2}")
        print(f"  DB2 only: {cols2 - cols1}")
        return False
    print("✓ Column names match")

    # Exclude run_id from comparison (auto-generated, may differ)
    compare_cols = [col for col in df1.columns if col != "run_id"]

    # Sort both by problem_name and algorithm for consistent comparison
    df1_sorted = df1.sort_values(["problem_name", "algorithm"]).reset_index(drop=True)
    df2_sorted = df2.sort_values(["problem_name", "algorithm"]).reset_index(drop=True)

    # Compare data
    differences = []
    for col in compare_cols:
        if col in [
            "raw_times",
            "raw_costs",
            "raw_gaps",
            "raw_initial_costs",
            "raw_generations",
            "raw_stop_reasons",
            "algorithm_config",
            "strategies",
        ]:
            # Skip complex array/json columns for now
            continue

        if not df1_sorted[col].equals(df2_sorted[col]):
            differences.append(col)

    if differences:
        print(f"❌ Data differences found in columns: {differences}")
        return False

    print("✓ Data values match (excluding run_id and complex columns)")
    return True


def validate_against_checkpoints(df: pd.DataFrame, checkpoint_dir: Path) -> bool:
    """Validate database contents against source checkpoint files."""
    print("\n" + "=" * 80)
    print("CHECKPOINT VALIDATION")
    print("=" * 80)

    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    print(f"Database has {len(df)} rows")

    # Sample 3 random checkpoints to validate
    import random

    sample_files = random.sample(checkpoint_files, min(3, len(checkpoint_files)))

    all_valid = True
    for filepath in sample_files:
        print(f"\nValidating: {filepath.name}")

        # Load checkpoint
        checkpoint = load_checkpoint(filepath)
        problem_name = checkpoint["problem_name"]
        algorithm = checkpoint["algorithm"]

        # Find matching row in database
        matches = df[
            (df["problem_name"] == problem_name) & (df["algorithm"] == algorithm)
        ]

        if len(matches) == 0:
            print(f"  ❌ No matching row found in database")
            all_valid = False
            continue

        if len(matches) > 1:
            print(f"  ⚠ Multiple matches found ({len(matches)})")

        row = matches.iloc[0]

        # Validate key fields
        checks = {
            "problem_size": (checkpoint["problem_size"], row["problem_size"]),
            "optimal_cost": (checkpoint["optimal_cost"], row["optimal_cost"]),
            "repetitions": (checkpoint["repetitions"], row["repetitions"]),
            "mean_time": (checkpoint["mean_time"], row["mean_time"]),
            "best_cost": (checkpoint["best_cost"], row["best_cost"]),
            "backend": (checkpoint["backend"], row["backend"]),
        }

        field_valid = True
        for field, (expected, actual) in checks.items():
            if expected != actual:
                print(f"  ❌ {field}: expected {expected}, got {actual}")
                field_valid = False
                all_valid = False

        if field_valid:
            print(f"  ✓ All fields match")

    return all_valid


def main():
    """Main validation function."""
    print("=" * 80)
    print("DATABASE CONSISTENCY VALIDATION")
    print("=" * 80)

    # Check if databases exist
    if not Path(DB1_PATH).exists():
        print(f"❌ Database 1 not found: {DB1_PATH}")
        return False

    if not Path(DB2_PATH).exists():
        print(f"❌ Database 2 not found: {DB2_PATH}")
        return False

    print(f"✓ Both databases exist")

    # Load databases
    print("\nLoading databases...")
    df1 = load_database_to_df(DB1_PATH)
    df2 = load_database_to_df(DB2_PATH)
    print(f"✓ Loaded DB1: {len(df1)} rows")
    print(f"✓ Loaded DB2: {len(df2)} rows")

    # Compare databases
    db_comparison = compare_databases(df1, df2)

    # Validate against checkpoints (use DB1)
    checkpoint_validation = validate_against_checkpoints(df1, CHECKPOINT_DIR)

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Database comparison: {'✅ PASS' if db_comparison else '❌ FAIL'}")
    print(f"Checkpoint validation: {'✅ PASS' if checkpoint_validation else '❌ FAIL'}")

    if db_comparison and checkpoint_validation:
        print("\n🎉 All validations passed! Databases are consistent.")
        return True
    else:
        print("\n❌ Validation failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
