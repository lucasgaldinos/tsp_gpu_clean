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
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
  load_algorithms_config,
  load_all_configs,
  load_benchmark_config,
  load_problems_config,
)
from data_models.exceptions import AmbiguousProblemError

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
  ]
  assert configs["algorithms"]["CPU"].use_gpu is False
  assert configs["algorithms"]["CPU"].metaheuristic == "GeneticAlgorithm"
  assert configs["algorithms"]["CPU"].strategies["selection"]["class"] == "SelectionTournament"


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
  assert not hasattr(benchmark, "skip_cpu_default")
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
  benchmark = BenchmarkConfig(repetitions=3, patience=None, cpu_size_threshold=50, line_width=40)

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
    "benchmark.json": {"repetitions": 2, "patience": 5, "cpu_size_threshold": 100},
    "problems.json": {"problems": []},
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
    {"patience": 5, "cpu_size_threshold": 100},
  )

  # Act / Assert
  with pytest.raises(ValueError, match="missing required field: repetitions"):
    load_benchmark_config(tmp_path)


def test_load_benchmark_config_allows_optional_or_null_cpu_size_threshold(tmp_path: Path) -> None:
  """WHAT: benchmark.json with cpu_size_threshold null or omitted parses as None (BUG-81 / Task 13.26)."""
  # Case 1: explicitly null
  _write_json(
    tmp_path / "benchmark.json",
    {"repetitions": 10, "patience": 5, "cpu_size_threshold": None},
  )
  cfg = load_benchmark_config(tmp_path)
  assert cfg.cpu_size_threshold is None

  # Case 2: omitted
  _write_json(
    tmp_path / "benchmark.json",
    {"repetitions": 10, "patience": 5},
  )
  cfg_omitted = load_benchmark_config(tmp_path)
  assert cfg_omitted.cpu_size_threshold is None


def test_determine_problem_algorithms_threshold_none_vs_integer() -> None:
  """WHAT: _determine_problem_algorithms includes all algorithms when cpu_threshold is None,
  and skips CPU algorithms when problem_size > cpu_threshold.

  WHY: Task 13.26 (BUG-81 / DP13-1) — DoD requirement for None vs integer threshold behavior.
  """
  from benchmarking.runner.config_loader import AlgorithmConfig
  from benchmarking.runner.orchestration import _determine_problem_algorithms

  algo_configs = {
    "CPU": AlgorithmConfig(name="CPU", use_gpu=False, description="CPU"),
    "HybridNaive": AlgorithmConfig(name="HybridNaive", use_gpu=True, description="GPU 1"),
    "HybridOptimized": AlgorithmConfig(name="HybridOptimized", use_gpu=True, description="GPU 2"),
    "FullGPUModular": AlgorithmConfig(name="FullGPUModular", use_gpu=True, description="GPU 3"),
  }
  selected = list(algo_configs.keys())

  # Case 1: cpu_threshold=None (no cutoff) -> all 4 algorithms on size 50 and size 1002
  assert _determine_problem_algorithms(50, selected, algo_configs, None) == selected
  assert _determine_problem_algorithms(1002, selected, algo_configs, None) == selected

  # Case 2: cpu_threshold=400 -> all 4 algorithms on size 50, but size 1002 skips CPU
  assert _determine_problem_algorithms(50, selected, algo_configs, 400) == selected
  gpu_only = ["HybridNaive", "HybridOptimized", "FullGPUModular"]
  assert _determine_problem_algorithms(1002, selected, algo_configs, 400) == gpu_only


@pytest.mark.parametrize(
  "payload",
  [
    pytest.param([{"name": "burma14", "optimal": 3323, "size": 14}], id="bare-array"),
    pytest.param({"name": "burma14"}, id="object-without-problems"),
  ],
)
def test_load_problems_config_requires_problems_array(tmp_path: Path, payload: object) -> None:
  """WHAT: problems.json that is not an object with a `problems` array.

  WHY: The document carries `$schema` next to the array (`DP11-5`); any other shape
  must fail loudly instead of iterating keys and producing nonsense errors.
  EXPECTED: `ValueError` naming the expected object/array shape.
  """
  # Arrange
  _write_json(tmp_path / "problems.json", payload)

  # Act / Assert
  with pytest.raises(ValueError, match="'problems' array"):
    load_problems_config(tmp_path)


@pytest.mark.parametrize(
  ("entry", "missing_field"),
  [
    pytest.param({"size": 14}, "name", id="missing-name"),
    pytest.param({"name": "burma14"}, "size", id="missing-size"),
  ],
)
def test_load_problems_config_requires_each_field(
  tmp_path: Path,
  entry: dict[str, Any],
  missing_field: str,
) -> None:
  """WHAT: A problem entry missing required `name` or `size`.

  WHY: Problem identity and dimension are mandatory; `optimal` is optional (`BUG-106`).
  EXPECTED: `ValueError` with the entry index and missing field name.
  """
  _write_json(tmp_path / "problems.json", {"problems": [entry]})

  with pytest.raises(ValueError, match=rf"index 0 missing required field: {missing_field}"):
    load_problems_config(tmp_path)


def test_load_benchmark_and_problems_optional_fields_and_bool_guards(tmp_path: Path) -> None:
  """WHAT: Omitted `patience` / `optimal` default to `None` and boolean values are rejected (`BUG-106`)."""
  _write_json(tmp_path / "benchmark.json", {"repetitions": 5})
  bench = load_benchmark_config(tmp_path)
  assert bench.repetitions == 5
  assert bench.patience is None
  assert bench.max_generations is None
  assert bench.cpu_size_threshold is None

  for bad_bench in (
    {"repetitions": True},
    {"repetitions": 5, "patience": False},
    {"repetitions": 5, "cpu_size_threshold": True},
    {"repetitions": 5, "max_generations": True},
    {"repetitions": 5, "display": {"line_width": True}},
    {"repetitions": 5, "incremental_runs": "yes"},
  ):
    _write_json(tmp_path / "benchmark.json", bad_bench)
    with pytest.raises(ValueError):
      load_benchmark_config(tmp_path)

  _write_json(tmp_path / "problems.json", {"problems": [{"name": "custom50", "size": 50}]})
  probs = load_problems_config(tmp_path)
  assert probs == [ProblemConfig(name="custom50", optimal=None, size=50)]

  for bad_prob in (
    {"problems": [{"name": "custom50", "size": True}]},
    {"problems": [{"name": "custom50", "size": 50, "optimal": False}]},
    {"problems": [{"name": "custom50", "size": 50, "problem_type": "MDVRP"}]},
  ):
    _write_json(tmp_path / "problems.json", bad_prob)
    with pytest.raises(ValueError):
      load_problems_config(tmp_path)


def test_problems_config_supports_problem_type_and_vrp_builders(tmp_path: Path) -> None:
  """WHAT: `ProblemConfig` and DuckDB/TSPLIB builders support `problem_type` and `.vrp`/`.sol` (`BUG-109`)."""
  _write_json(
    tmp_path / "problems.json",
    {"problems": [{"name": "eil51", "size": 51, "optimal": 521.0, "problem_type": "cvrp"}]},
  )
  parsed = load_problems_config(tmp_path)
  assert parsed == [ProblemConfig(name="eil51", optimal=521.0, size=51, problem_type="CVRP")]

  vrp_dir = tmp_path / "vrp_files"
  vrp_dir.mkdir()
  vrp_path = vrp_dir / "mini4.vrp"
  vrp_path.write_text(
    "NAME : mini4\n"
    "TYPE : CVRP\n"
    "DIMENSION : 4\n"
    "EDGE_WEIGHT_TYPE : EUC_2D\n"
    "CAPACITY : 10\n"
    "NODE_COORD_SECTION\n"
    "1 0 0\n2 1 1\n3 4 1\n4 4 5\n"
    "DEMAND_SECTION\n"
    "1 0\n2 4\n3 5\n4 3\n"
    "DEPOT_SECTION\n"
    "1\n-1\nEOF\n",
    encoding="utf-8",
  )
  (vrp_dir / "mini4.sol").write_text("Route #1: 1 2\nRoute #2: 3\nCost 19.75\n", encoding="utf-8")

  configs, path_map = build_problem_configs_from_tsp_files(["mini4"], tsp_dir=vrp_dir)
  assert configs == [ProblemConfig(name="mini4", optimal=20.0, size=4, problem_type="CVRP")]
  assert path_map["mini4"] == vrp_path.resolve()

  from loaders.tsplib_loader import ingest_tsplib_directory_to_duckdb

  (vrp_dir / "mini4.tsp").write_text(
    "NAME : mini4\nTYPE : TSP\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : EUC_2D\n"
    "NODE_COORD_SECTION\n1 0 0\n2 3 0\n3 3 4\n4 0 4\nEOF\n",
    encoding="utf-8",
  )
  (vrp_dir / "mini4.opt.tour").write_text(
    "NAME : mini4.opt.tour\nCOMMENT : Length 14\nTYPE : TOUR\nDIMENSION : 4\n"
    "TOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n",
    encoding="utf-8",
  )
  db_path = tmp_path / "mini.duckdb"
  assert ingest_tsplib_directory_to_duckdb(vrp_dir, db_path) == 2

  with pytest.raises(AmbiguousProblemError):
    build_problem_configs_from_duckdb(db_path, problem_names=["mini4"])

  disambiguated = build_problem_configs_from_duckdb(
    db_path,
    problem_names=["mini4"],
    existing_configs=[ProblemConfig(name="mini4", optimal=None, size=4, problem_type="CVRP")],
  )
  assert disambiguated == [ProblemConfig(name="mini4", optimal=20.0, size=4, problem_type="CVRP")]

  all_db_configs = build_problem_configs_from_duckdb(db_path)
  assert sorted((c.name, c.problem_type, c.optimal) for c in all_db_configs) == [
    ("mini4", "CVRP", 20.0),
    ("mini4", "TSP", 14.0),
  ]


def test_problem_config_declarative_overrides_and_validation(tmp_path: Path) -> None:
  """WHAT: `load_problems_config` and `ProblemConfig` validate declarative constraint overrides (`DP17-10`)."""
  _write_json(
    tmp_path / "problems.json",
    {
      "problems": [
        {
          "name": "eil51",
          "size": 51,
          "problem_type": "OVRP",
          "return_to_depot": False,
          "max_vehicles": 5,
          "max_route_distance": 250,
          "capacity": 120,
        }
      ]
    },
  )
  parsed = load_problems_config(tmp_path)
  assert len(parsed) == 1
  cfg = parsed[0]
  assert cfg.problem_type == "OVRP"
  assert cfg.return_to_depot is False
  assert cfg.max_vehicles == 5
  assert cfg.max_route_distance == pytest.approx(250.0)
  assert cfg.capacity == 120
  assert cfg.has_constraint_overrides() is True

  for bad_kwargs in (
    {"max_vehicles": True},
    {"max_vehicles": 0},
    {"capacity": False},
    {"capacity": -1},
    {"max_route_distance": True},
    {"max_route_distance": 0.0},
    {"max_route_distance": float("nan")},
    {"max_route_distance": float("inf")},
    {"return_to_depot": "false"},
  ):
    with pytest.raises(ValueError):
      ProblemConfig(name="eil51", optimal=None, size=51, **bad_kwargs)
