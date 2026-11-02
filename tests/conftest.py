"""Pytest fixtures for hermetic, self-contained testing without external dataset dependencies.

Also owns the collection hook that turns `@pytest.mark.gpu` into an automatic skip
when `protocols.backend.gpu_available()` reports no CUDA device.
"""

import os
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pytest

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

from data_models.problem import Problem
from loaders.database_loader import DatabaseLoader
from loaders.tsplib_loader import TsplibLoader, ingest_tsplib_directory_to_duckdb
from protocols.backend import gpu_available

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TSP_DIR = FIXTURES_DIR / "tsp"
CONFIGS_DIR = FIXTURES_DIR / "configs"

# ---------------------------------------------------------------------------
# GPU availability: delegates to `protocols.backend.gpu_available()`.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
  """Skip every `gpu`-marked test when the probe reports no device.

  EXPECTED: without CUDA, the device layer is reported as skipped instead of failing
  inside CuPy; with a device, the marks are inert and the tests run unchanged.
  """
  if gpu_available():
    return
  skip_no_gpu = pytest.mark.skip(reason="CUDA GPU not available")
  for item in items:
    if item.get_closest_marker("gpu") is not None:
      item.add_marker(skip_no_gpu)


@pytest.fixture
def make_population() -> Callable[..., np.ndarray]:
  """Fixture returning a factory for deterministic permutation populations."""

  def _factory(num_routes: int, n_genes: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n_genes, dtype=np.int32)
    return np.stack([rng.permutation(base) for _ in range(num_routes)]).astype(np.int32, copy=False)

  return _factory


@pytest.fixture
def population(make_population: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
  """Fixture alias for `make_population` supporting legacy and helper calls."""
  return make_population


@pytest.fixture(scope="session")
def fixture_tsp_dir() -> Path:
  """Path to the hermetic TSPLIB test files directory."""
  return TSP_DIR


@pytest.fixture(scope="session")
def fixture_config_dir() -> Path:
  """Path to the test configuration directory."""
  return CONFIGS_DIR


@pytest.fixture(scope="session")
def fixture_duckdb_path(tmp_path_factory: pytest.TempPathFactory, fixture_tsp_dir: Path) -> Path:
  """Session-scoped temporary DuckDB database populated from test fixture .tsp files."""
  db_dir = tmp_path_factory.mktemp("db")
  db_path = db_dir / "test_tsp.duckdb"
  ingest_tsplib_directory_to_duckdb(fixture_tsp_dir, db_path)
  return db_path


@pytest.fixture
def burma14_problem(fixture_tsp_dir: Path) -> Generator[Problem, None, None]:
  """Loaded Problem instance for the 14-city GEO instance (burma14)."""
  with TsplibLoader(fixture_tsp_dir) as loader:
    yield loader.load("burma14")


@pytest.fixture
def berlin52_problem(fixture_tsp_dir: Path) -> Generator[Problem, None, None]:
  """Loaded Problem instance for the 52-city EUC_2D instance (berlin52)."""
  with TsplibLoader(fixture_tsp_dir) as loader:
    yield loader.load("berlin52")


@pytest.fixture
def db_loader(fixture_duckdb_path: Path, fixture_tsp_dir: Path) -> Generator[DatabaseLoader, None, None]:
  """DatabaseLoader wired to the session-scoped fixture DuckDB database."""
  with DatabaseLoader(fixture_duckdb_path, fallback_tsp_dir=fixture_tsp_dir) as loader:
    yield loader
