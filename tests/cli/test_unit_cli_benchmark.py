"""Unit tests for the `cli.benchmark` command group entrypoint."""

from pathlib import Path

import pytest

from cli.benchmark import main as benchmark_main


def test_benchmark_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify benchmark CLI prints help text with --help."""
  with pytest.raises(SystemExit) as exc:
    benchmark_main(["--help"])
  assert exc.value.code == 0
  captured = capsys.readouterr()
  assert "tsp-benchmark" in captured.out or "usage:" in captured.out.lower()


def test_benchmark_cli_run(fixture_config_dir: Path, fixture_tsp_dir: Path, tmp_path: Path) -> None:
  """Verify benchmark CLI executes with required config and output arguments."""
  gr17_path = fixture_tsp_dir / "gr17.tsp"
  exit_code = benchmark_main(
    [
      "--config-dir",
      str(fixture_config_dir),
      "--output-dir",
      str(tmp_path),
      "--tsp-files",
      str(gr17_path),
      "--algorithms",
      "CPU",
      "--repetitions",
      "1",
    ]
  )
  assert exit_code == 0
  assert (tmp_path / "checkpoints").exists()


def test_benchmark_parser_accepts_overrides() -> None:
  """Verify benchmark CLI parser parses all new override flags correctly."""
  from cli.benchmark import build_parser

  parser = build_parser()
  args = parser.parse_args(
    [
      "--config-dir",
      "configs",
      "--output-dir",
      "results",
      "--patience",
      "15",
      "--seed",
      "777",
      "--population-size",
      "250",
      "--mutation-rate",
      "0.08",
      "--strategy-param",
      "local_search.max_iterations=20",
      "--strategy-param",
      "mutation.swap_prob=0.5",
      "--no-incremental-runs",
      "--log-level",
      "DEBUG",
      "--local-search-stage",
      "post_evolution",
    ]
  )
  assert args.patience == 15
  assert args.seed == 777
  assert args.population_size == 250
  assert args.mutation_rate == 0.08
  assert args.strategy_param == ["local_search.max_iterations=20", "mutation.swap_prob=0.5"]
  assert args.incremental_runs is False
  assert args.log_level == "DEBUG"
  assert args.local_search_stage == "post_evolution"


def test_benchmark_parser_rejects_invalid_local_search_stage() -> None:
  """Verify benchmark CLI parser rejects invalid local-search stages (e.g. 'never')."""
  from cli.benchmark import build_parser

  parser = build_parser()
  with pytest.raises(SystemExit) as exc:
    parser.parse_args(["--config-dir", "configs", "--output-dir", "results", "--local-search-stage", "never"])
  assert exc.value.code == 2


def test_parse_strategy_param_types() -> None:
  """Verify parse_strategy_param correctly parses numbers, bools, and strings."""
  from cli.benchmark import parse_strategy_param

  slot, key, val = parse_strategy_param("local_search.max_iterations=25")
  assert (slot, key, val) == ("local_search", "max_iterations", 25)

  slot, key, val = parse_strategy_param("crossover.rate=0.85")
  assert (slot, key, val) == ("crossover", "rate", 0.85)

  slot, key, val = parse_strategy_param("local_search.enabled=true")
  assert (slot, key, val) == ("local_search", "enabled", True)

  slot, key, val = parse_strategy_param("local_search.mode=steepest")
  assert (slot, key, val) == ("local_search", "mode", "steepest")

  with pytest.raises(ValueError, match="Expected format"):
    parse_strategy_param("invalid_format")

  with pytest.raises(ValueError, match=r"Target must be SLOT\.KEY"):
    parse_strategy_param("no_slot=10")

  with pytest.raises(ValueError, match="non-empty SLOT and KEY"):
    parse_strategy_param(".max_iterations=10")

  with pytest.raises(ValueError, match="non-empty SLOT and KEY"):
    parse_strategy_param("local_search.=10")


