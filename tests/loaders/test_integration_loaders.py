"""Integration tests for the TSPLIB95 / DuckDB loading layer.

Covers the `tsplib95` parse into a `Problem` bundle, the `DatabaseLoader` fallback to
`TsplibLoader` when the database file is missing, the DuckDB ingestion round-trip, and
the normalized-schema metadata exposed by `Problem.to_metadata_dict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from loaders import (
  DatabaseLoader,
  bundle_to_problem,
  ingest_tsplib_directory_to_duckdb,
  parse_tsp_file_with_tsplib95,
)

if TYPE_CHECKING:
  from pathlib import Path

  from data_models.problem import Problem
  from loaders.tsplib_loader import TsplibIngestionBundle


def test_tsplib95_parse_and_bundle_to_problem(fixture_tsp_dir: Path) -> None:
  """Verify parsing EUC_2D and EXPLICIT .tsp instances with tsplib95."""
  bundle_euc: TsplibIngestionBundle = parse_tsp_file_with_tsplib95(fixture_tsp_dir / "berlin52.tsp", problem_id=1)
  assert bundle_euc.problem.name == "berlin52"
  assert bundle_euc.problem.dimension == 52
  assert bundle_euc.problem.edge_weight_type == "EUC_2D"
  assert len(bundle_euc.nodes) == 52
  assert bundle_euc.edge_matrix is None

  problem_euc: Problem = bundle_to_problem(bundle_euc)
  assert problem_euc.dimension == 52
  assert problem_euc.coordinates is not None
  assert problem_euc.coordinates.shape == (52, 2)
  assert problem_euc.distances is not None
  assert problem_euc.distances.shape == (52, 52)

  bundle_exp: TsplibIngestionBundle = parse_tsp_file_with_tsplib95(fixture_tsp_dir / "gr17.tsp", problem_id=2)
  assert bundle_exp.problem.name == "gr17"
  assert bundle_exp.problem.edge_weight_type == "EXPLICIT"
  assert bundle_exp.edge_matrix is not None

  problem_exp: Problem = bundle_to_problem(bundle_exp)
  assert problem_exp.dimension == 17
  assert problem_exp.distances is not None
  assert problem_exp.distances.shape == (17, 17)


def test_database_loader_fallback_and_duckdb_ingestion(tmp_path: Path, fixture_tsp_dir: Path) -> None:
  """Verify DatabaseLoader fallback to TsplibLoader and full DuckDB ingestion."""
  # 1. Test automatic fallback when routing.duckdb does not exist yet
  missing_default_db = tmp_path / "routing.duckdb"
  with DatabaseLoader(str(missing_default_db), fallback_tsp_dir=fixture_tsp_dir) as loader:
    prob = loader.load("berlin52")
    assert prob.name == "berlin52"
    assert prob.dimension == 52

  # 2. Test ingesting a subset of .tsp files into a temporary DuckDB file
  mini_tsp_dir: Path = tmp_path / "mini_tsp"
  mini_tsp_dir.mkdir()
  for name in ("berlin52.tsp", "gr17.tsp"):
    src_file: Path = fixture_tsp_dir / name
    (mini_tsp_dir / name).write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

  ingested_db: Path = tmp_path / "ingested.duckdb"
  count = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, ingested_db)
  assert count == 2
  assert ingested_db.exists()

  with DatabaseLoader(str(ingested_db), fallback_tsp_dir=None) as db_loader:
    p1 = db_loader.load("berlin52")
    p2 = db_loader.load("gr17")
    assert p1.dimension == 52
    assert p2.dimension == 17
    assert p2.distances is not None


def test_database_loader_normalized_schema_and_problem_metadata(fixture_duckdb_path: Path) -> None:
  """Verify DatabaseLoader queries normalized DuckDB and exposes metadata."""
  with DatabaseLoader(fixture_duckdb_path, fallback_tsp_dir=None) as loader:
    prob_b52 = loader.load("berlin52")
    assert prob_b52.name == "berlin52"
    assert prob_b52.dimension == 52
    assert prob_b52.edge_type == "EUC_2D"
    assert prob_b52.problem_id is not None
    meta_b52: dict[str, Any] = prob_b52.to_metadata_dict(optimal_cost=7542.0)
    assert meta_b52["name"] == "berlin52"
    assert meta_b52["dimension"] == 52
    assert meta_b52["optimal_cost"] == 7542.0
    assert "source_file" in meta_b52
    assert "file_checksum" in meta_b52

    prob_bayg29 = loader.load("bayg29")
    assert prob_bayg29.name == "bayg29"
    assert prob_bayg29.dimension == 29
    assert prob_bayg29.edge_type == "EXPLICIT"
    assert prob_bayg29.distances is not None
    assert prob_bayg29.distances.shape == (29, 29)
    meta_bayg29: dict[str, Any] = prob_bayg29.to_metadata_dict(optimal_cost=1610.0)
    assert meta_bayg29["dimension"] == 29
    assert meta_bayg29["edge_type"] == "EXPLICIT"


def test_ingest_refuses_non_empty_db(tmp_path: Path, fixture_tsp_dir: Path) -> None:
  """Verify ingest_tsplib_directory_to_duckdb raises DatabaseNotEmptyError unless overwrite=True."""
  import pytest

  from data_models.exceptions import DatabaseNotEmptyError

  mini_tsp_dir: Path = tmp_path / "mini_tsp_overwrite"
  mini_tsp_dir.mkdir()
  src_file: Path = fixture_tsp_dir / "gr17.tsp"
  (mini_tsp_dir / "gr17.tsp").write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

  test_db: Path = tmp_path / "test_overwrite.duckdb"

  # First ingestion succeeds
  count1 = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=False)
  assert count1 == 1

  # Second ingestion without overwrite raises DatabaseNotEmptyError
  with pytest.raises(DatabaseNotEmptyError, match="already contains records in 'problems'"):
    ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=False)

  # Second ingestion with overwrite=True succeeds
  count2 = ingest_tsplib_directory_to_duckdb(mini_tsp_dir, test_db, overwrite=True)
  assert count2 == 1


def test_ingest_missing_directory_raises(tmp_path: Path) -> None:
  """Verify ingest_tsplib_directory_to_duckdb raises FileNotFoundError if source directory does not exist."""
  with pytest.raises(FileNotFoundError, match="TSPLIB source directory not found"):
    ingest_tsplib_directory_to_duckdb(tmp_path / "non_existent_tsp", tmp_path / "out.duckdb")

