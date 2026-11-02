"""Unit tests for config validation in `benchmarking.runner.config_loader`.

WHAT: Covers every `validate_configs` warning branch, the `KNOWN_TSPLIB_OPTIMALS`
hit/miss paths, `build_problem_configs_from_tsp_files` and the adaptive
generation/patience formulas.
WHY: The runner trusts these checks before executing a benchmark; a missed
warning or a wrong lookup turns into a silently misconfigured or mislabelled run.
EXPECTED: One warning per violated rule with its documented text, `KeyError` for
unknown optimals, and exact integer formulas.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from benchmarking.runner.algorithm_runner import adaptive_generations, adaptive_patience
from benchmarking.runner.config_loader import (
  KNOWN_TSPLIB_OPTIMALS,
  AlgorithmConfig,
  ProblemConfig,
  build_problem_configs_from_tsp_files,
  load_all_configs,
  validate_configs,
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
        strategies={**cpu_config.strategies, "selection": {"params": {"tournament_size": 0}}},
      )
    case "two_opt_negative":
      mutated["algorithms"]["CPU"] = replace(
        cpu_config,
        strategies={**cpu_config.strategies, "local_search": {"params": {"two_opt_iterations": -3}}},
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
      "Invalid tournament_size in CPU: 0 (must be > 0)",
      id="tournament_zero",
    ),
    pytest.param(
      "two_opt_negative",
      "Invalid two_opt_iterations in CPU: -3 (must be > 0)",
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
# KNOWN_TSPLIB_OPTIMALS
# ============================================================================


@pytest.mark.parametrize(
  ("name", "expected"),
  [
    pytest.param("burma14", 3323.0, id="burma14"),
    pytest.param("att48", 10628.0, id="att48"),
    pytest.param("berlin52", 7542.0, id="berlin52"),
    pytest.param("kroA100", 21282.0, id="kroA100"),
  ],
)
def test_known_tsplib_optimals_hit(name: str, expected: float) -> None:
  """WHAT: Known-optimal lookup for benchmark instances.

  WHY: Gap percentages are computed against these constants; a wrong value
  silently shifts every reported gap.
  EXPECTED: the canonical TSPLIB95 optimal tour cost.
  """
  # Act / Assert
  assert KNOWN_TSPLIB_OPTIMALS[name] == expected


def test_known_tsplib_optimals_miss_returns_none() -> None:
  """WHAT: Lookup of an unknown problem name.

  WHY: Callers use `in` / `.get` to decide whether to raise; the miss path must
  stay distinguishable from a stored zero.
  EXPECTED: `not in` and `.get(...) is None`.
  """
  # Act / Assert
  assert "definitely_not_a_tsplib_instance" not in KNOWN_TSPLIB_OPTIMALS
  assert KNOWN_TSPLIB_OPTIMALS.get("definitely_not_a_tsplib_instance") is None


def test_build_problem_configs_uses_known_optimal(fixture_tsp_dir: Path) -> None:
  """WHAT: Building configs from a fixture `.tsp` with a known optimal.

  WHY: This is the "lookup hit" path used by the real benchmark entry point.
  EXPECTED: burma14 with optimal 3323.0 and size 14.
  """
  # Act
  configs, path_map = build_problem_configs_from_tsp_files(["burma14"], tsp_dir=fixture_tsp_dir)

  # Assert
  assert configs == [ProblemConfig(name="burma14", optimal=3323.0, size=14)]
  assert path_map["burma14"].name == "burma14.tsp"
  assert path_map["burma14"].is_absolute()


def test_build_problem_configs_prefers_existing_config(fixture_tsp_dir: Path) -> None:
  """WHAT: `existing_configs` takes precedence over `KNOWN_TSPLIB_OPTIMALS`.

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
  )

  # Assert
  assert configs[0].optimal == 999.0


def test_build_problem_configs_unknown_optimal_raises(fixture_tsp_dir: Path, tmp_path: Path) -> None:
  """WHAT: Building configs for a problem absent from `KNOWN_TSPLIB_OPTIMALS`.

  WHY: The "lookup miss" must raise instead of producing a config with a fake
  optimal cost.
  EXPECTED: `KeyError` naming the known-optimal registry.
  """
  # Arrange
  source_text = (fixture_tsp_dir / "burma14.tsp").read_text()
  renamed = tmp_path / "zzunknown.tsp"
  renamed.write_text(source_text.replace("burma14", "zzunknown"))

  # Act / Assert
  with pytest.raises(KeyError, match="Known optimal tour cost not found"):
    build_problem_configs_from_tsp_files([renamed])


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
