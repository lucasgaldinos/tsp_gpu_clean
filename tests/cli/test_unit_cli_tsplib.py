"""Unit tests for the `cli.tsplib` command group entrypoint."""

from pathlib import Path

import pytest

from cli.tsplib import main as tsplib_main


def test_tsplib_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify tsplib CLI prints help text with --help."""
  with pytest.raises(SystemExit) as exc:
    tsplib_main(["--help"])
  assert exc.value.code == 0
  captured = capsys.readouterr()
  assert "tsp-ingest" in captured.out or "usage:" in captured.out.lower()


def test_tsplib_cli_inspect_instance(fixture_tsp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verify tsplib CLI can inspect a single instance."""
  exit_code = tsplib_main(["--tsp-dir", str(fixture_tsp_dir), "--instance", "burma14"])
  assert exit_code == 0
  captured = capsys.readouterr()
  assert "Loaded burma14" in captured.out
  assert "dimension=14" in captured.out


def test_tsplib_cli_to_duckdb(fixture_tsp_dir: Path, tmp_path: Path) -> None:
  """Verify tsplib CLI can ingest a directory of TSP files into DuckDB."""
  db_path = tmp_path / "cli_test.duckdb"
  exit_code = tsplib_main(["--tsp-dir", str(fixture_tsp_dir), "--to-duckdb", str(db_path)])
  assert exit_code == 0
  assert db_path.exists()


def test_tsplib_cli_to_duckdb_overwrite_protection(
  fixture_tsp_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verify tsplib CLI returns exit code 1 when attempting to overwrite without --overwrite, and succeeds with it."""
  db_path = tmp_path / "cli_overwrite_test.duckdb"
  exit_code1 = tsplib_main(["--tsp-dir", str(fixture_tsp_dir), "--to-duckdb", str(db_path)])
  assert exit_code1 == 0

  exit_code2 = tsplib_main(["--tsp-dir", str(fixture_tsp_dir), "--to-duckdb", str(db_path)])
  assert exit_code2 == 1
  captured = capsys.readouterr()
  assert "Pass overwrite=True or use CLI flag --overwrite" in captured.err

  exit_code3 = tsplib_main(["--tsp-dir", str(fixture_tsp_dir), "--to-duckdb", str(db_path), "--overwrite"])
  assert exit_code3 == 0


def test_tsplib_cli_handles_euc3d_instance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verify tsplib CLI inspects, verifies, and ingests EUC_3D `.tsp` files."""
  tsp_dir = tmp_path / "euc3d_cli"
  tsp_dir.mkdir()
  (tsp_dir / "cube3.tsp").write_text(
    "NAME: cube3\n"
    "TYPE: TSP\n"
    "DIMENSION: 3\n"
    "EDGE_WEIGHT_TYPE: EUC_3D\n"
    "NODE_COORD_SECTION\n"
    "1 0.0 0.0 0.0\n"
    "2 1.0 2.0 3.0\n"
    "3 4.0 6.0 8.0\n"
    "EOF\n",
    encoding="utf-8",
  )

  assert tsplib_main(["--tsp-dir", str(tsp_dir), "--instance", "cube3"]) == 0
  captured = capsys.readouterr()
  assert "Loaded cube3: type=TSP, dimension=3, edge_type=EUC_3D" in captured.out

  assert tsplib_main(["--tsp-dir", str(tsp_dir)]) == 0
  captured_verify = capsys.readouterr()
  assert "Verified 1/1 .tsp files" in captured_verify.out

  db_path = tmp_path / "cube3.duckdb"
  assert tsplib_main(["--tsp-dir", str(tsp_dir), "--to-duckdb", str(db_path)]) == 0
  assert db_path.exists()

