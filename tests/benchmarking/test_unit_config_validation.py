"""Unit tests for config validation in `benchmarking.runner.config_loader`.

WHAT: Covers every `validate_configs` warning branch, the `known_optimals`
hit/miss paths of `build_problem_configs_from_tsp_files` and the adaptive
generation/patience formulas.
WHY: The runner trusts these checks before executing a benchmark; a missed
warning or a wrong lookup turns into a silently misconfigured or mislabelled run.
EXPECTED: One warning per violated rule with its documented text, `KeyError` for
unknown optimals, and exact integer formulas.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest

from benchmarking.runner.algorithm_runner import adaptive_generations, adaptive_patience
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  ProblemConfig,
  build_problem_configs_from_tsp_files,
  load_all_configs,
  validate_configs,
)

# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def fixture_configs(fixture_config_dir: Path) -> dict[str, Any]:
  """Load the hermetic fixture configuration directory once per test."""
  return load_all_configs(fixture_config_dir)


def _copy_configs(configs: dict[str, Any]) -> dict[str, Any]:
  """Shallow-copy the config bundle so a mutation cannot leak between cases."""
  return {
    "ga_params": replace(configs["ga_params"]),
    "algorithms": {name: replace(cfg) for name, cfg in configs["algorithms"].items()},
    "benchmark": replace(configs["benchmark"]),
    "problems": [replace(problem) for problem in configs["problems"]],
  }


def _mutated_configs(configs: dict[str, Any], case: str) -> dict[str, Any]:
  """Return a copy of `configs` with a single named violation applied."""
  mutated = _copy_configs(configs)
  cpu_config: AlgorithmConfig = mutated["algorithms"]["CPU"]

  match case:
    case "population_zero":
      mutated["ga_params"] = replace(mutated["ga_params"], population_size=0)
    case "population_negative":
      mutated["ga_params"] = replace(mutated["ga_params"], population_size=-4)
    case "mutation_too_high":
      mutated["ga_params"] = replace(mutated["ga_params"], mutation_rate=1.5)
    case "mutation_negative":
      mutated["ga_params"] = replace(mutated["ga_params"], mutation_rate=-0.1)
    case "mutation_zero":
      mutated["ga_params"] = replace(mutated["ga_params"], mutation_rate=0.0)
    case "mutation_one":
      mutated["ga_params"] = replace(mutated["ga_params"], mutation_rate=1.0)
    case "tournament_zero":
      mutated["algorithms"]["CPU"] = replace(
        cpu_config,
        strategies={**cpu_config.strategies, "selection": {"params": {"k": 0}}},
      )
    case "two_opt_negative":
      mutated["algorithms"]["CPU"] = replace(
        cpu_config,
        strategies={**cpu_config.strategies, "local_search": {"params": {"max_iterations": -3}}},
      )
    case "repetitions_zero":
      mutated["benchmark"] = replace(mutated["benchmark"], repetitions=0)
    case "patience_zero":
      mutated["benchmark"] = replace(mutated["benchmark"], patience=0)
    case "patience_none":
      mutated["benchmark"] = replace(mutated["benchmark"], patience=None)
    case "threshold_zero":
      mutated["benchmark"] = replace(mutated["benchmark"], cpu_size_threshold=0)
    case "no_problems":
      mutated["problems"] = []
    case "problem_size_zero":
      mutated["problems"][0] = replace(mutated["problems"][0], size=0)
    case "problem_optimal_zero":
      mutated["problems"][0] = replace(mutated["problems"][0], optimal=0.0)
    case "duplicate_problems":
      mutated["problems"] = [*mutated["problems"], replace(mutated["problems"][0])]
    case "unknown_algorithm_class":
      mutated["algorithms"]["CPU"] = replace(
        cpu_config,
        class_module="definitely.not.a.real.module",
        class_name="Ghost",
      )
    case _:
      raise AssertionError(f"unknown validation case: {case}")

  return mutated


# ============================================================================
# validate_configs
# ============================================================================


def test_validate_configs_accepts_fixture_configs(fixture_configs: dict[str, Any]) -> None:
  """WHAT: Validation of the hermetic fixture configuration.

  WHY: It is the reference "valid" config; warnings here would mean every test
  run reports noise.
  EXPECTED: empty warning list.
  """
  # Act
  warnings = validate_configs(_copy_configs(fixture_configs))

  # Assert
  assert warnings == []


@pytest.mark.parametrize(
  ("case", "expected"),
  [
    pytest.param("population_zero", "Invalid population_size: 0 (must be > 0)", id="population_zero"),
    pytest.param("population_negative", "Invalid population_size: -4 (must be > 0)", id="population_negative"),
    pytest.param("mutation_too_high", "Invalid mutation_rate: 1.5 (must be in [0, 1])", id="mutation_too_high"),
    pytest.param("mutation_negative", "Invalid mutation_rate: -0.1 (must be in [0, 1])", id="mutation_negative"),
    pytest.param(
      "tournament_zero",
      "Invalid k in CPU.selection: 0 (must be > 0)",
      id="tournament_zero",
    ),
    pytest.param(
      "two_opt_negative",
      "Invalid max_iterations in CPU.local_search: -3 (must be > 0)",
      id="two_opt_negative",
    ),
    pytest.param("repetitions_zero", "Invalid repetitions: 0 (must be > 0)", id="repetitions_zero"),
    pytest.param("patience_zero", "Invalid patience: 0 (must be > 0 or None)", id="patience_zero"),
    pytest.param("threshold_zero", "Invalid cpu_size_threshold: 0 (must be > 0)", id="threshold_zero"),
    pytest.param("no_problems", "No problems defined in problems.json", id="no_problems"),
    pytest.param(
      "problem_size_zero",
      "Problem 'burma14' has invalid size: 0 (must be > 0)",
      id="problem_size_zero",
    ),
    pytest.param(
      "problem_optimal_zero",
      "Problem 'burma14' has invalid optimal: 0.0 (must be > 0)",
      id="problem_optimal_zero",
    ),
    pytest.param("duplicate_problems", "Duplicate problem names found", id="duplicate_problems"),
    pytest.param(
      "unknown_algorithm_class",
      "Algorithm 'CPU' class not found",
      id="unknown_algorithm_class",
    ),
  ],
)
def test_validate_configs_reports_each_violation(
  fixture_configs: dict[str, Any],
  case: str,
  expected: str,
) -> None:
  """WHAT: One isolated rule violation per case.

  WHY: Each branch is a separate guard; a shared test would hide which rule broke.
  EXPECTED: exactly one warning containing the documented message.
  """
  # Arrange
  mutated = _mutated_configs(fixture_configs, case)

  # Act
  warnings = validate_configs(mutated)

  # Assert
  assert len(warnings) == 1, f"expected exactly one warning for {case}, got {warnings}"
  assert expected in warnings[0]


@pytest.mark.parametrize("case", ["mutation_zero", "mutation_one", "patience_none"])
def test_validate_configs_accepts_boundary_values(fixture_configs: dict[str, Any], case: str) -> None:
  """WHAT: Legal boundary values: mutation_rate 0.0/1.0 and patience None.

  WHY: The guards are inclusive at the boundaries; rejecting them would block
  valid exploratory configs.
  EXPECTED: no warnings.
  """
  # Arrange
  mutated = _mutated_configs(fixture_configs, case)

  # Act / Assert
  assert validate_configs(mutated) == []


def test_validate_configs_accumulates_warnings_in_rule_order(fixture_configs: dict[str, Any]) -> None:
  """WHAT: Two simultaneous violations.

  WHY: The warning list is rendered in order; population rules are checked before
  benchmark rules.
  EXPECTED: population warning first, repetitions warning second.
  """
  # Arrange
  mutated = _copy_configs(fixture_configs)
  mutated["ga_params"] = replace(mutated["ga_params"], population_size=0)
  mutated["benchmark"] = replace(mutated["benchmark"], repetitions=-2)

  # Act
  warnings = validate_configs(mutated)

  # Assert
  assert len(warnings) == 2
  assert warnings[0].startswith("Invalid population_size")
  assert warnings[1].startswith("Invalid repetitions")


# ============================================================================
# build_problem_configs_from_tsp_files
# ============================================================================


def test_build_problem_configs_uses_known_optimal(fixture_tsp_dir: Path) -> None:
  """WHAT: Building configs from a fixture `.tsp` with a supplied optimal cost.

  WHY: This is the "lookup hit" path used by the real benchmark entry point, which
  derives the map from `configs/problems.json` (`DP11-4`).
  EXPECTED: burma14 with optimal 3323.0 and size 14.
  """
  # Act
  configs, path_map = build_problem_configs_from_tsp_files(
    ["burma14"],
    tsp_dir=fixture_tsp_dir,
    known_optimals={"burma14": 3323.0},
  )

  # Assert
  assert configs == [ProblemConfig(name="burma14", optimal=3323.0, size=14)]
  assert path_map["burma14"].name == "burma14.tsp"
  assert path_map["burma14"].is_absolute()


def test_build_problem_configs_prefers_existing_config(fixture_tsp_dir: Path) -> None:
  """WHAT: `existing_configs` takes precedence over `known_optimals`.

  WHY: Experiments may pin a custom reference cost; the override must win.
  EXPECTED: optimal 999.0 from the supplied config.
  """
  # Arrange
  existing = [ProblemConfig(name="burma14", optimal=999.0, size=14)]

  # Act
  configs, _ = build_problem_configs_from_tsp_files(
    ["burma14"],
    existing_configs=existing,
    tsp_dir=fixture_tsp_dir,
    known_optimals={"burma14": 3323.0},
  )

  # Assert
  assert configs[0].optimal == 999.0


def test_build_problem_configs_unknown_optimal_returns_none_and_uses_opt_tour(
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """WHAT: Building configs for a `.tsp` absent from `known_optimals`, with and without sibling `.opt.tour`.

  WHY: Arbitrary `.tsp` files must work out-of-the-box without editing `configs/problems.json`:
  if a sibling `<stem>.opt.tour` exists, its canonical cost is resolved; otherwise `optimal` is `None`.
  EXPECTED: `optimal is None` without `.opt.tour`; positive canonical cost when `.opt.tour` is present.
  """
  source_text = (fixture_tsp_dir / "burma14.tsp").read_text(encoding="utf-8")
  renamed = tmp_path / "zzunknown.tsp"
  renamed.write_text(source_text.replace("burma14", "zzunknown"), encoding="utf-8")

  configs, _ = build_problem_configs_from_tsp_files([renamed], known_optimals={"burma14": 3323.0})
  assert configs == [ProblemConfig(name="zzunknown", optimal=None, size=14)]

  # Add a sibling .opt.tour (valid 1..14 permutation) and verify cost is resolved automatically.
  tour_lines = [
    "NAME : zzunknown.opt.tour",
    "TYPE : TOUR",
    "DIMENSION : 14",
    "TOUR_SECTION",
    *(str(i) for i in range(1, 15)),
    "-1",
    "EOF",
  ]
  (tmp_path / "zzunknown.opt.tour").write_text("\n".join(tour_lines) + "\n", encoding="utf-8")
  configs_with_tour, _ = build_problem_configs_from_tsp_files([renamed], known_optimals={"burma14": 3323.0})
  assert configs_with_tour[0].optimal is not None
  assert configs_with_tour[0].optimal > 0


# ============================================================================
# Adaptive formulas
# ============================================================================


@pytest.mark.parametrize(
  ("n", "expected"),
  [
    pytest.param(0, 0, id="zero"),
    pytest.param(1, 2, id="one"),
    pytest.param(16, 128, id="perfect_square"),
    pytest.param(49, 686, id="perfect_square_7"),
    pytest.param(52, 749, id="truncates_750"),
    pytest.param(100, 2000, id="hundred"),
    pytest.param(318, int(2 * 318 * math.sqrt(318)), id="lin318"),
    pytest.param(532, int(2 * 532 * math.sqrt(532)), id="att532"),
    pytest.param(783, int(2 * 783 * math.sqrt(783)), id="rat783"),
    pytest.param(1002, int(2 * 1002 * math.sqrt(1002)), id="pr1002"),
  ],
)
def test_adaptive_generations_formula(n: int, expected: int) -> None:
  """WHAT: `adaptive_generations(n) = int(2 * n * sqrt(n))`.

  WHY: The generation budget defines experiment cost and comparability across
  problem sizes.
  EXPECTED: exact integers; n=52 truncates 749.95... to 749 (not 750).
  """
  # Act / Assert
  assert adaptive_generations(n) == expected
  assert adaptive_generations(n) == int(2 * n * math.sqrt(n))


@pytest.mark.parametrize(
  ("n", "expected"),
  [
    pytest.param(0, 0, id="zero"),
    pytest.param(1, 2, id="one"),
    pytest.param(16, 8, id="perfect_square"),
    pytest.param(49, 14, id="perfect_square_7"),
    pytest.param(52, 14, id="truncates_14_42"),
    pytest.param(100, 20, id="hundred"),
  ],
)
def test_adaptive_patience_formula(n: int, expected: int) -> None:
  """WHAT: `adaptive_patience(n) = int(2 * sqrt(n))`.

  WHY: Patience controls early stopping; an off-by-one changes when runs stop.
  EXPECTED: exact integers; n=52 truncates 14.42... to 14.
  """
  # Act / Assert
  assert adaptive_patience(n) == expected
  assert adaptive_patience(n) == int(2 * math.sqrt(n))


# ============================================================================
# Formal JSON Schema Validation (Task 12.17, DP12-10)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHAPTER4_CONFIGS_DIR = PROJECT_ROOT / "benchmark_scripts" / "configs" / "chapter4"
SCHEMAS_DIR = CONFIGS_DIR / "schemas"


@pytest.mark.parametrize(
  ("config_path", "schema_filename"),
  [
    (CHAPTER4_CONFIGS_DIR / "benchmark.json", "benchmark.schema.json"),
    (CHAPTER4_CONFIGS_DIR / "problems.json", "problems.schema.json"),
    (CHAPTER4_CONFIGS_DIR / "algorithms.json", "algorithms.schema.json"),
    (CHAPTER4_CONFIGS_DIR / "algorithms.archive.json", "algorithms.schema.json"),
    (CONFIGS_DIR / "algorithms.full.example.json", "algorithms.schema.json"),
    (CONFIGS_DIR / "cpu-operators.example.json", "algorithms.schema.json"),
    (CONFIGS_DIR / "local-search-pipelines.example.json", "algorithms.schema.json"),
    (CONFIGS_DIR / "gpu-backends.example.json", "algorithms.schema.json"),
    (CONFIGS_DIR / "report-style.example.json", "report_style.schema.json"),
    (CHAPTER4_CONFIGS_DIR / "report_style.json", "report_style.schema.json"),
  ],
)
def test_all_json_configs_validate_against_their_schemas(
  config_path: Path,
  schema_filename: str,
) -> None:
  """WHAT: Every example and Chapter-4 config file validates against its JSON Schema.

  WHY: Task 12.17 (DP12-10) — Ensures strict structural conformity across all config
  files and prevents schema drift between configuration files and dataclasses.
  EXPECTED: jsonschema.validate completes without raising ValidationError.
  """
  schema_path = SCHEMAS_DIR / schema_filename

  config_data = json.loads(config_path.read_text(encoding="utf-8"))
  schema_data = json.loads(schema_path.read_text(encoding="utf-8"))

  jsonschema.validate(instance=config_data, schema=schema_data)


@pytest.mark.parametrize(
  ("mutation_fn", "error_match"),
  [
    (lambda d: d.update({"repetitions": 0}), "0 is less than the minimum of 1"),
    (lambda d: d.update({"patience": 0}), "0 is less than the minimum of 1"),
    (lambda d: d.update({"max_generations": 0}), "0 is less than the minimum of 1"),
    (lambda d: d.update({"cpu_size_threshold": 0}), "0 is less than the minimum of 1"),
    (lambda d: d.update({"spurious_field": 123}), "Additional properties are not allowed"),
    (lambda d: d.update({"skip_cpu_default": False}), "Additional properties are not allowed"),
    (lambda d: d.pop("repetitions"), "'repetitions' is a required property"),
  ],
)
def test_benchmark_schema_rejects_invalid_payloads(
  mutation_fn: Any,
  error_match: str,
) -> None:
  """WHAT: benchmark.schema.json rejects malformed or invalid benchmark configs.

  WHY: Task 12.17 — Strict schema validation catches erroneous bounds and spurious fields.
  EXPECTED: jsonschema.ValidationError raised with matching diagnostic error message.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "benchmark.json").read_text(encoding="utf-8"))
  mutation_fn(base)
  schema = json.loads((SCHEMAS_DIR / "benchmark.schema.json").read_text(encoding="utf-8"))

  with pytest.raises(jsonschema.ValidationError, match=error_match):
    jsonschema.validate(instance=base, schema=schema)


def test_benchmark_schema_accepts_null_and_omitted_cpu_size_threshold() -> None:
  """WHAT: benchmark.schema.json accepts null or omitted cpu_size_threshold and max_generations."""
  base = json.loads((CHAPTER4_CONFIGS_DIR / "benchmark.json").read_text(encoding="utf-8"))
  schema = json.loads((SCHEMAS_DIR / "benchmark.schema.json").read_text(encoding="utf-8"))

  # Case 1: null
  base["cpu_size_threshold"] = None
  base["max_generations"] = None
  jsonschema.validate(instance=base, schema=schema)

  # Case 2: omitted
  base.pop("cpu_size_threshold", None)
  base.pop("max_generations", None)
  jsonschema.validate(instance=base, schema=schema)

  # Case 3: positive int
  base["cpu_size_threshold"] = 250
  base["max_generations"] = 50
  jsonschema.validate(instance=base, schema=schema)


@pytest.mark.parametrize(
  ("mutation_fn", "error_match"),
  [
    (lambda d: d.update({"problems": []}), "should be non-empty|is too short"),
    (lambda d: d["problems"][0].update({"size": 0}), "0 is less than the minimum of 1"),
    (lambda d: d["problems"][0].update({"optimal": 0}), "0 is less than or equal to the minimum of 0"),
    (lambda d: d["problems"][0].update({"optimal": -5}), "less than or equal to the minimum"),
    (lambda d: d["problems"][0].update({"spurious": True}), "Additional properties are not allowed"),
    (lambda d: d["problems"][0].pop("name"), "'name' is a required property"),
  ],
)
def test_problems_schema_rejects_invalid_payloads(
  mutation_fn: Any,
  error_match: str,
) -> None:
  """WHAT: problems.schema.json rejects invalid instance entries.

  WHY: Task 12.17 — Problems catalogue is the single source of truth for instance dimensions and optimals.
  EXPECTED: jsonschema.ValidationError raised.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "problems.json").read_text(encoding="utf-8"))
  mutation_fn(base)
  schema = json.loads((SCHEMAS_DIR / "problems.schema.json").read_text(encoding="utf-8"))

  with pytest.raises(jsonschema.ValidationError, match=error_match):
    jsonschema.validate(instance=base, schema=schema)


@pytest.mark.parametrize(
  ("mutation_fn", "error_match"),
  [
    (lambda d: d["default_params"].update({"population_size": 0}), "0 is less than the minimum of 1"),
    (lambda d: d["default_params"].update({"mutation_rate": 1.5}), "1.5 is greater than the maximum of 1"),
    (
      lambda d: d["algorithm_configs"]["CPU"]["strategies"]["selection"].update({"class": "UnknownSelection"}),
      "is not one of",
    ),
    (
      lambda d: d["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"].update({"spurious_arg": 1}),
      "Additional properties are not allowed",
    ),
    (
      lambda d: d["algorithm_configs"]["CPU"]["params"].update({"local_search_stage": "invalid_stage"}),
      "is not one of",
    ),
  ],
)
def test_algorithms_schema_rejects_invalid_payloads(
  mutation_fn: Any,
  error_match: str,
) -> None:
  """WHAT: algorithms.schema.json rejects invalid algorithm declarations.

  WHY: Task 12.17 — Strategy classes and parameters must be recognized and validated.
  EXPECTED: jsonschema.ValidationError raised.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "algorithms.json").read_text(encoding="utf-8"))
  mutation_fn(base)
  schema = json.loads((SCHEMAS_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))

  with pytest.raises(jsonschema.ValidationError, match=error_match):
    jsonschema.validate(instance=base, schema=schema)


def test_periodic_local_search_step_schema_validation_with_re_evaluate() -> None:
  """WHAT: algorithms.schema.json accepts PeriodicLocalSearchStep with re_evaluate.

  WHY: Task 12.17 (DP12-10) — When target='population', PeriodicLocalSearchStep
  requires re_evaluate. The schema must permit re_evaluate as a valid strategy_slot.
  EXPECTED: jsonschema.validate completes without raising ValidationError.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "algorithms.json").read_text(encoding="utf-8"))
  schema = json.loads((SCHEMAS_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))

  # Inject PeriodicLocalSearchStep with re_evaluate slot
  base["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {
      "class": "PeriodicLocalSearchStep",
      "params": {
        "interval": 5,
        "target": "population",
        "re_evaluate": {
          "class": "EvaluationCPUCost",
        },
      },
    },
    {
      "class": "PeriodicLocalSearchStep",
      "params": {
        "interval": 10,
        "target": "population",
        "re_evaluate": {
          "class": "EvaluationGPUCost",
          "params": {
            "threads_per_block": 128,
          },
        },
      },
    },
    {
      "class": "PeriodicLocalSearchStep",
      "params": {
        "interval": 15,
        "target": "offspring",
      },
    },
  ]

  jsonschema.validate(instance=base, schema=schema)


@pytest.mark.parametrize(
  ("step_params", "error_match"),
  [
    (
      {"interval": 5, "target": "population", "spurious_key": True},
      "Additional properties are not allowed",
    ),
    (
      {"interval": 5, "target": "population", "re_evaluate": "NotAStrategyDict"},
      "is not of type 'object'",
    ),
    (
      {"interval": 5, "target": "population", "re_evaluate": {"class": "InvalidEvaluator"}},
      "is not one of",
    ),
    (
      {"interval": 5, "target": "invalid_target"},
      "is not one of",
    ),
  ],
)
def test_periodic_local_search_step_schema_rejects_malformed_params(
  step_params: dict[str, Any],
  error_match: str,
) -> None:
  """WHAT: algorithms.schema.json strictly rejects malformed params on pipeline steps.

  WHY: Task 12.17 (DP12-10) — additionalProperties: false must enforce exact property names.
  EXPECTED: jsonschema.ValidationError raised with matching diagnostic error message.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "algorithms.json").read_text(encoding="utf-8"))
  schema = json.loads((SCHEMAS_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))

  base["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {
      "class": "PeriodicLocalSearchStep",
      "params": step_params,
    }
  ]

  with pytest.raises(jsonschema.ValidationError, match=error_match):
    jsonschema.validate(instance=base, schema=schema)


@pytest.mark.parametrize(
  "metadata_update",
  [
    pytest.param({"edge_weight_type": "EUC_2D"}, id="edge_weight_type"),
    pytest.param({"comment": "17-city problem from Groetschel"}, id="comment"),
    pytest.param({"source": "TSPLIB95"}, id="source"),
    pytest.param({"problem_type": "TSP"}, id="problem_type"),
  ],
)
def test_problems_schema_rejects_tsplib_metadata(metadata_update: dict[str, Any]) -> None:
  """WHAT: problems.schema.json rejects duplicate TSPLIB metadata fields (BUG-83).

  WHY: Task 13.27(ii) — `routing_data` (`Problem`) is the single source of truth for
  `edge_type`, `problem_type`, `comment`, and `source`; `ProblemConfig` stores strictly
  `(name, optimal, size)` and `additionalProperties: false` must reject extra keys.
  EXPECTED: jsonschema.ValidationError raised for Additional properties.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "problems.json").read_text(encoding="utf-8"))
  schema = json.loads((SCHEMAS_DIR / "problems.schema.json").read_text(encoding="utf-8"))

  base["problems"][0].update(metadata_update)

  with pytest.raises(jsonschema.ValidationError, match="Additional properties are not allowed"):
    jsonschema.validate(instance=base, schema=schema)


@pytest.mark.parametrize(
  ("metadata_update", "error_match"),
  [
    ({"name": 123}, "123 is not of type 'string'"),
    ({"optimal": "not_a_number"}, "is not of type 'number'"),
    ({"size": "17"}, "is not of type 'integer'"),
    ({"unknown_custom_field": "disallowed"}, "Additional properties are not allowed"),
  ],
)
def test_problems_schema_rejects_invalid_metadata(
  metadata_update: dict[str, Any],
  error_match: str,
) -> None:
  """WHAT: problems.schema.json rejects invalid field types and unknown fields.

  WHY: Task 12.17 / Task 13.27(ii) — Strict types on `(name, optimal, size)` and `additionalProperties: false`.
  EXPECTED: jsonschema.ValidationError raised with matching diagnostic error message.
  """
  base = json.loads((CHAPTER4_CONFIGS_DIR / "problems.json").read_text(encoding="utf-8"))
  schema = json.loads((SCHEMAS_DIR / "problems.schema.json").read_text(encoding="utf-8"))

  base["problems"][0].update(metadata_update)

  with pytest.raises(jsonschema.ValidationError, match=error_match):
    jsonschema.validate(instance=base, schema=schema)


def test_validate_gpu_backend_for_algorithms_only_checks_when_gpu_selected(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: GPU backend preflight runs only when at least one selected algorithm has `use_gpu=True`.

  WHY: Task 13.25 (`BUG-79`) — CPU-only runs must pass on GPU-less hosts, while GPU runs
  must fail fast in preflight before writing any checkpoints.
  """
  from benchmarking.runner import orchestration

  algo_configs = {
    "CPU": AlgorithmConfig(name="CPU", use_gpu=False, description="CPU"),
    "HybridNaive": AlgorithmConfig(name="HybridNaive", use_gpu=True, description="GPU"),
  }

  monkeypatch.setattr(orchestration, "gpu_available", lambda: False)
  monkeypatch.setattr(orchestration, "cupy_backend", lambda: None)

  assert orchestration.validate_gpu_backend_for_algorithms(["CPU"], algo_configs) == []

  gpu_errors = orchestration.validate_gpu_backend_for_algorithms(["CPU", "HybridNaive"], algo_configs)
  assert len(gpu_errors) == 2
  assert any("gpu_available" in err for err in gpu_errors)
  assert any("cupy_backend" in err for err in gpu_errors)

  monkeypatch.setattr(orchestration, "gpu_available", lambda: True)
  monkeypatch.setattr(orchestration, "cupy_backend", object)
  assert orchestration.validate_gpu_backend_for_algorithms(["HybridNaive"], algo_configs) == []


def test_report_style_schema_matches_generator_and_rejects_invalid_payloads() -> None:
  """WHAT: Verify `report_style.schema.json` stays locked to `ReportStyle` and rejects bad payloads.

  WHY: Reflective schema generation keeps `ReportStyle`, `load_report_style()`, and
  `configs/schemas/report_style.schema.json` on a single source of truth without drift.
  EXPECTED: On-disk schema matches `render_report_style_schema()`, and invalid keys/types fail validation.
  """
  from benchmarking.runner.schema_generator import render_report_style_schema

  schema_path = SCHEMAS_DIR / "report_style.schema.json"
  assert schema_path.read_text(encoding="utf-8") == render_report_style_schema()

  schema = json.loads(schema_path.read_text(encoding="utf-8"))
  valid_base = json.loads((CONFIGS_DIR / "report-style.example.json").read_text(encoding="utf-8"))
  jsonschema.validate(instance=valid_base, schema=schema)

  invalid_mutations: list[tuple[dict[str, Any], str]] = [
    ({"decimal_separator": ""}, "should be non-empty|too short"),
    ({"size_thresholds": [100]}, "too short"),
    ({"plot_decimal_comma": "true"}, "is not of type 'boolean'"),
    ({"spurious_style_key": 123}, "Additional properties are not allowed"),
  ]
  for update, err_match in invalid_mutations:
    candidate = dict(valid_base)
    candidate.update(update)
    with pytest.raises(jsonschema.ValidationError, match=err_match):
      jsonschema.validate(instance=candidate, schema=schema)


def test_validate_configs_and_instantiate_catch_bad_slots_protocols_and_algo_params() -> None:
  """WHAT: `validate_configs` and `instantiate` reject unknown slots, mismatched protocols, and bad params.

  WHY: BUG-101 — Typoed slot keys (`cross_over`), wrong-slot strategies (`MutationSwap` in
  `crossover`), and out-of-range per-algorithm params previously slipped through with zero warnings.
  """
  from benchmarking.runner.config_loader import BenchmarkConfig, GAParams

  cfg = AlgorithmConfig(
    name="BrokenAlg",
    use_gpu=False,
    description="Broken",
    params={"population_size": -5, "mutation_rate": 9.9, "local_search_stage": "not_a_stage"},
    strategies={
      "cross_over": {"class": "CrossoverOrder"},
      "crossover": {"class": "MutationSwap"},
    },
  )
  warnings = validate_configs(
    {
      "ga_params": GAParams(),
      "algorithms": {"BrokenAlg": cfg},
      "benchmark": BenchmarkConfig(),
      "problems": [ProblemConfig("p1", 100.0, 10)],
    }
  )
  assert any("population_size" in w for w in warnings)
  assert any("mutation_rate" in w for w in warnings)
  assert any("local_search_stage" in w for w in warnings)
  assert any("unknown strategy slot 'cross_over'" in w for w in warnings)
  assert any("CrossoverStrategy" in w for w in warnings)

  with pytest.raises(ValueError, match="Unknown strategy slot 'cross_over'"):
    cfg.instantiate(GAParams())
