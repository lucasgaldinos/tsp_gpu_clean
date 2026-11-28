"""Unit tests for the `cli.checkpoints` command group entrypoints."""

import pytest

from cli.checkpoints import import_main, validate_main


def test_checkpoints_import_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify checkpoint import CLI prints help text with --help."""
  with pytest.raises(SystemExit) as exc:
    import_main(["--help"])
  assert exc.value.code == 0
  captured = capsys.readouterr()
  assert "tsp-import-checkpoints" in captured.out or "usage:" in captured.out.lower()


def test_checkpoints_validate_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify database validate CLI prints help text with --help."""
  with pytest.raises(SystemExit) as exc:
    validate_main(["--help"])
  assert exc.value.code == 0
  captured = capsys.readouterr()
  assert "tsp-validate-db" in captured.out or "usage:" in captured.out.lower()


def test_checkpoints_validate_cli_without_db2(tmp_path: pytest.TempPathFactory | object) -> None:
  """Verify `tsp-validate-db` validates a single DuckDB file against checkpoints without `--db2`."""
  import json
  from pathlib import Path

  root = Path(str(tmp_path))
  ckpt_dir = root / "checkpoints"
  ckpt_dir.mkdir()
  payload = {
    "problem_name": "custom14",
    "problem_size": 14,
    "optimal_cost": None,
    "algorithm": "CPU",
    "backend": "CPU",
    "repetitions": 2,
    "successful_runs": 2,
    "timestamp": "2026-11-23T10:00:00",
    "max_generations_configured": 100,
    "mean_time": 0.5,
    "std_time": 0.1,
    "min_time": 0.4,
    "max_time": 0.6,
    "mean_cost": 3500.0,
    "std_cost": 20.0,
    "best_cost": 3480.0,
    "worst_cost": 3520.0,
    "mean_gap": None,
    "std_gap": None,
    "best_gap": None,
    "mean_generations": 40.0,
    "std_generations": 2.0,
    "raw_times": [0.4, 0.6],
    "raw_costs": [3480.0, 3520.0],
    "raw_gaps": None,
    "algorithm_config": {"population_size": 32},
    "strategies": {"local_search": "LocalSearchCPUTourTwoOpt"},
  }
  (ckpt_dir / "custom14_CPU.json").write_text(json.dumps(payload), encoding="utf-8")

  assert import_main(["--input", str(ckpt_dir), "--output-dir", str(root)]) == 0
  db_path = root / "results.duckdb"
  assert db_path.is_file()
  assert validate_main(["--db1", str(db_path), "--checkpoints", str(ckpt_dir)]) == 0
