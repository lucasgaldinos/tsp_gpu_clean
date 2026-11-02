"""Unit tests for config loading in `benchmarking.runner.config_loader`.

WHAT: Covers config loading from the hermetic `fixture_config_dir` (returned
structure and parsed values) and every loader error branch that guards a missing
directory, missing file, missing section or wrong document shape.
WHY: The runner builds every benchmark from these configs; a silent fallback or a
misread field turns into a misconfigured run instead of a loud failure.
EXPECTED: Exact parsed values and structures, and the documented exception with
its message for each malformed input.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from benchmarking.runner.config_loader import (
  BenchmarkConfig,
  ProblemConfig,
  load_algorithms_config,
  load_all_configs,
  load_benchmark_config,
  load_problems_config,
)

if TYPE_CHECKING:
  from pathlib import Path


# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def fixture_configs(fixture_config_dir: Path) -> dict[str, Any]:
  """Load the hermetic fixture configuration directory once per test."""
  return load_all_configs(fixture_config_dir)


def _write_json(path: Path, payload: Any) -> None:
  """Write `payload` as JSON to `path` (tests always target `tmp_path`)."""
  path.write_text(json.dumps(payload))


# ============================================================================
# load_all_configs structure
# ============================================================================


def test_load_all_configs_returns_expected_structure(fixture_configs: dict[str, Any]) -> None:
  """WHAT: `load_all_configs` over the hermetic fixture directory.

  WHY: The runner indexes this dict by fixed keys and relies on the parsed types;
  a shape change breaks every benchmark entry point.
  EXPECTED: keys ga_params/algorithms/benchmark/problems with the fixture values.
  """
  # Arrange / Act
  configs = fixture_configs

  # Assert
  assert set(configs) == {"ga_params", "algorithms", "benchmark", "problems"}
  assert configs["ga_params"].population_size == 32
  assert configs["ga_params"].mutation_rate == pytest.approx(0.05)
  assert configs["ga_params"].seed == 42
  assert list(configs["algorithms"]) == [
    "CPU",
    "HybridNaive",
    "HybridOptimized",
    "FullGPUModular",
    "FullGPU",
    "FullGPUMonolithic",
  ]
  assert configs["algorithms"]["CPU"].use_gpu is False
  assert configs["algorithms"]["CPU"].metaheuristic == "GeneticAlgorithm"
  assert configs["algorithms"]["CPU"].strategies["selection"]["class"] == "TournamentSelection"


def test_load_all_configs_parses_benchmark_and_problem_values(fixture_configs: dict[str, Any]) -> None:
  """WHAT: `BenchmarkConfig` fields and `ProblemConfig` list from the fixtures.

  WHY: Repetition counts and optimal costs feed every gap and completeness check.
  EXPECTED: repetitions 2, patience 5, threshold 100, incremental_runs False,
  line_width 80; five problems in JSON order with their known optimals.
  """
  # Arrange
  benchmark: BenchmarkConfig = fixture_configs["benchmark"]
  problems: list[ProblemConfig] = fixture_configs["problems"]

  # Act / Assert
  assert benchmark.repetitions == 2
  assert benchmark.patience == 5
  assert benchmark.cpu_size_threshold == 100
  assert benchmark.skip_cpu_default is False
  assert benchmark.incremental_runs is False
  assert benchmark.line_width == 80
  assert [problem.name for problem in problems] == ["burma14", "gr17", "bayg29", "att48", "berlin52"]
  assert [problem.optimal for problem in problems] == [3323.0, 2085.0, 1610.0, 10628.0, 7542.0]
  assert [problem.size for problem in problems] == [14, 17, 29, 48, 52]


def test_benchmark_config_separators_match_line_width() -> None:
  """WHAT: `BenchmarkConfig.separator` and `short_sep` properties.

  WHY: Log headers rely on the widths to stay aligned with 80-column output.
  EXPECTED: `=`/`-` repeated `line_width` times.
  """
  # Arrange
  benchmark = BenchmarkConfig(repetitions=3, patience=None, cpu_size_threshold=50, skip_cpu_default=True, line_width=40)

  # Act / Assert
  assert benchmark.separator == "=" * 40
  assert benchmark.short_sep == "-" * 40


# ============================================================================
# Loader error branches
# ============================================================================


def test_load_all_configs_missing_directory_raises(tmp_path: Path) -> None:
  """WHAT: Loading a directory that does not exist.

  WHY: A typo in the config path must fail loudly, not fall back to defaults.
  EXPECTED: `FileNotFoundError` naming the directory.
  """
  # Arrange
  missing = tmp_path / "no_such_config_dir"

  # Act / Assert
  with pytest.raises(FileNotFoundError, match="Configuration directory not found"):
    load_all_configs(missing)


def test_load_all_configs_rejects_file_path(tmp_path: Path) -> None:
  """WHAT: Loading a path that exists but is a file.

  WHY: Silently treating a file as a config directory would raise a confusing
  secondary error deeper in a loader.
  EXPECTED: `ValueError` mentioning the directory requirement.
  """
  # Arrange
  config_file = tmp_path / "algorithms.json"
  _write_json(config_file, {})

  # Act / Assert
  with pytest.raises(ValueError, match="must be a directory"):
    load_all_configs(config_file)


@pytest.mark.parametrize("missing_name", ["algorithms.json", "benchmark.json", "problems.json"])
def test_load_all_configs_requires_each_file(tmp_path: Path, missing_name: str) -> None:
  """WHAT: One of the three required JSON files is absent.

  WHY: Each loader must name the missing file instead of reusing stale configs.
  EXPECTED: `FileNotFoundError` mentioning the file.
  """
  # Arrange
  config_dir = tmp_path / "configs"
  config_dir.mkdir()
  complete_configs = {
    "algorithms.json": {"algorithm_configs": {}},
    "benchmark.json": {"repetitions": 2, "patience": 5, "cpu_size_threshold": 100, "skip_cpu_default": False},
    "problems.json": [],
  }
  for name, payload in complete_configs.items():
    if name != missing_name:
      _write_json(config_dir / name, payload)

  # Act / Assert
  with pytest.raises(FileNotFoundError, match=missing_name):
    load_all_configs(config_dir)


def test_load_algorithms_config_requires_algorithm_configs(tmp_path: Path) -> None:
  """WHAT: algorithms.json without the `algorithm_configs` section.

  WHY: An empty algorithm dict would let a benchmark run "successfully" with zero
  algorithms.
  EXPECTED: `ValueError` naming the missing section.
  """
  # Arrange
  _write_json(tmp_path / "algorithms.json", {"default_params": {"population_size": 8}})

  # Act / Assert
  with pytest.raises(ValueError, match="algorithm_configs"):
    load_algorithms_config(tmp_path)


def test_load_benchmark_config_requires_each_field(tmp_path: Path) -> None:
  """WHAT: benchmark.json missing `repetitions`.

  WHY: Required execution parameters must not be defaulted silently.
  EXPECTED: `ValueError` naming the missing field.
  """
  # Arrange
  _write_json(
    tmp_path / "benchmark.json",
    {"patience": 5, "cpu_size_threshold": 100, "skip_cpu_default": False},
  )

  # Act / Assert
  with pytest.raises(ValueError, match="missing required field: repetitions"):
    load_benchmark_config(tmp_path)


def test_load_problems_config_rejects_non_array(tmp_path: Path) -> None:
  """WHAT: problems.json containing an object instead of an array.

  WHY: The loader iterates the document; an object would iterate keys and produce
  nonsense errors.
  EXPECTED: `ValueError` mentioning the JSON array requirement.
  """
  # Arrange
  _write_json(tmp_path / "problems.json", {"name": "burma14"})

  # Act / Assert
  with pytest.raises(ValueError, match="JSON array"):
    load_problems_config(tmp_path)


def test_load_problems_config_requires_each_field(tmp_path: Path) -> None:
  """WHAT: A problem entry missing `optimal`.

  WHY: A missing optimal would turn every gap computation into a silent skip.
  EXPECTED: `ValueError` with the entry index and field name.
  """
  # Arrange
  _write_json(tmp_path / "problems.json", [{"name": "burma14", "size": 14}])

  # Act / Assert
  with pytest.raises(ValueError, match=r"index 0 missing required field: optimal"):
    load_problems_config(tmp_path)
