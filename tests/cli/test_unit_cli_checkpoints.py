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
