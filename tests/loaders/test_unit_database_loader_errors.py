"""Unit tests for `DatabaseLoader.__enter__` failure paths.

A missing database file is a configuration error the caller must be able to branch on,
so it surfaces as `DatabaseConnectionError` carrying the original `FileNotFoundError`
(``BUG-16`` / ``DP10-11``). The existence check runs before the connection ``try``, so the
error is never re-captured and re-wrapped by the generic handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_models.exceptions import DatabaseConnectionError
from loaders.database_loader import DatabaseLoader

if TYPE_CHECKING:
  from pathlib import Path


def test_missing_db_without_fallback_raises_database_connection_error(tmp_path: Path) -> None:
  """WHAT: entering a loader whose `db_path` does not exist and has no `.tsp` fallback.

  WHY: the pre-fix code raised `FileNotFoundError` inside the connection `try` and
  caught it with `except Exception`, producing a generic error with a self-referential
  cause chain instead of an actionable loader error.

  EXPECTED: `DatabaseConnectionError` naming the path with "does not exist", its
  `original_error` a `FileNotFoundError`, no `__cause__`, and no open connection.
  """
  # Arrange
  missing_db = tmp_path / "absent.duckdb"
  loader = DatabaseLoader(str(missing_db), fallback_tsp_dir=None)

  # Act / Assert
  with pytest.raises(DatabaseConnectionError) as excinfo:
    loader.__enter__()

  error = excinfo.value
  assert error.db_path == str(missing_db)
  assert isinstance(error.original_error, FileNotFoundError)
  assert "does not exist" in str(error)
  assert error.__cause__ is None, "the FileNotFoundError must not be captured and re-wrapped"
  assert loader.conn is None
  assert loader._tsplib_fallback is None


def test_missing_db_with_existing_fallback_directory_uses_tsplib_loader(
  tmp_path: Path,
  fixture_tsp_dir: Path,
) -> None:
  """WHAT: the same missing database, but with a populated `.tsp` fallback directory.

  WHY: the fallback must be chosen before the existence error, so the loader still works
  while the DuckDB file has not been built.

  EXPECTED: no exception, loads berlin52 through `TsplibLoader`, and the fallback is
  released on context exit.
  """
  # Arrange
  missing_db = tmp_path / "absent_with_fallback.duckdb"

  # Act
  with DatabaseLoader(str(missing_db), fallback_tsp_dir=fixture_tsp_dir) as loader:
    problem = loader.load("berlin52")
    fallback_active = loader._tsplib_fallback is not None

  # Assert
  assert problem.name == "berlin52"
  assert problem.dimension == 52
  assert fallback_active
  assert loader._tsplib_fallback is None
