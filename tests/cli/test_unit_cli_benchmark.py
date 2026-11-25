"""Unit tests for the `cli.benchmark` command group entrypoint."""

from pathlib import Path
from typing import Any

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


def test_benchmark_cli_tsp_dir_without_db_and_custom_unknown_optimal(
  fixture_config_dir: Path,
  fixture_tsp_dir: Path,
  tmp_path: Path,
) -> None:
  """Verify `--tsp-dir` works without `--db-path`, defaults to collection-only, and honors `--reports`."""
  tsp_dir = tmp_path / "custom_tsps"
  tsp_dir.mkdir()
  (tsp_dir / "gr17.tsp").write_text((fixture_tsp_dir / "gr17.tsp").read_text(encoding="utf-8"), encoding="utf-8")
  custom_text = (fixture_tsp_dir / "burma14.tsp").read_text(encoding="utf-8").replace("burma14", "custom14")
  (tsp_dir / "custom14.tsp").write_text(custom_text, encoding="utf-8")

  # 1. Collection-only default (no tables/ or figures/ emitted)
  collect_dir = tmp_path / "collect_only"
  exit_code = benchmark_main(
    [
      "--config-dir",
      str(fixture_config_dir),
      "--output-dir",
      str(collect_dir),
      "--tsp-dir",
      str(tsp_dir),
      "--problems",
      "gr17",
      "custom14",
      "--algorithms",
      "CPU",
      "-r",
      "1",
      "--max-generations",
      "5",
    ]
  )
  assert exit_code == 0
  assert (collect_dir / "checkpoints" / "gr17_CPU.json").is_file()
  assert (collect_dir / "checkpoints" / "custom14_CPU.json").is_file()
  assert not (collect_dir / "tables").exists()
  assert not (collect_dir / "figures").exists()

  # 2. Zero-boilerplate single-file --algorithms-config + auto-discovered --tsp-dir + opt-in --reports
  out_dir = tmp_path / "single_file_out"
  example_alg = Path(__file__).resolve().parents[2] / "configs" / "cpu-operators.example.json"
  example_style = Path(__file__).resolve().parents[2] / "configs" / "report-style.example.json"
  exit_code2 = benchmark_main(
    [
      "--algorithms-config",
      str(example_alg),
      "--output-dir",
      str(out_dir),
      "--tsp-dir",
      str(tsp_dir),
      "--algorithms",
      "PureGA_OX_Swap",
      "-r",
      "1",
      "--max-generations",
      "5",
      "--reports",
      "markdown",
      "plots",
      "--report-style",
      str(example_style),
    ]
  )
  assert exit_code2 == 0
  assert (out_dir / "checkpoints" / "gr17_PureGA_OX_Swap.json").is_file()
  assert (out_dir / "checkpoints" / "custom14_PureGA_OX_Swap.json").is_file()
  md_text = (out_dir / "tables" / "benchmark.md").read_text(encoding="utf-8")
  assert "custom14" in md_text
  assert "Pure GA (OX + Swap)" in md_text
  assert "| — |" in md_text
  assert not (out_dir / "tables" / "benchmark.tex").exists()
  assert (out_dir / "figures" / "scaling_time_gap.pdf").is_file()


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
      "--baseline-algorithm",
      "HybridNaive",
      "--cpu-size-threshold",
      "200",
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
  assert args.baseline_algorithm == "HybridNaive"
  assert args.cpu_size_threshold == 200
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


def test_benchmark_parser_supports_incremental_aliases_and_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
  """WHAT: Verify canonical CLI parser exposes --incremental / --no-incremental, -r, --baseline, and --dry-run.

  WHY: Task 13.17 (BUG-66) & Task 13.24 (DP13-8 / BUG-80) — Single canonical parser supports all flags
  and documents both incremental (append block) and no-incremental (complete missing runs) branches.
  """
  from cli.benchmark import build_parser

  parser = build_parser()
  args_inc = parser.parse_args(["--config-dir", "c", "--output-dir", "o", "--incremental", "-r", "30", "--dry-run"])
  assert args_inc.incremental_runs is True
  assert args_inc.repetitions == 30
  assert args_inc.dry_run is True

  args_no_inc = parser.parse_args(["--config-dir", "c", "--output-dir", "o", "--no-incremental", "--baseline", "CPU"])
  assert args_no_inc.incremental_runs is False
  assert args_no_inc.baseline_algorithm == "CPU"

  with pytest.raises(SystemExit) as exc:
    parser.parse_args(["--help"])
  assert exc.value.code == 0
  help_text = capsys.readouterr().out
  assert "--incremental" in help_text
  assert "--no-incremental" in help_text
  assert "Append a full block" in help_text
  assert "missing repetitions" in help_text


def _load_chapter4_script() -> Any:
  """Load `benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py` without mutating sys.path."""
  import importlib.util

  script_path = (
    Path(__file__).resolve().parents[2]
    / "benchmark_scripts"
    / "experiments-and-examples"
    / "chapter4"
    / "run_chapter4_benchmark.py"
  )
  spec = importlib.util.spec_from_file_location("run_chapter4_benchmark", script_path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_chapter4_script_help_and_dry_run_logging_isolation(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: run_chapter4_benchmark delegates --help and --dry-run to cli.benchmark without touching chapter4/results/.

  WHY: Task 13.17, Task 13.22 (BUG-74), and Task 13.24 (DP13-8 / BUG-80).
  EXPECTED: --help lists full CLI flags including --incremental/--no-incremental and --patience/--seed;
  --output-dir redirects logging to a single log file with preflight lines and zero files created in chapter4/results/.
  """
  import json
  import subprocess
  import sys

  run_chapter4_benchmark = _load_chapter4_script()
  repo_root = Path(__file__).resolve().parents[2]

  # Verify repetition parity across configs/benchmark.json and thesis script (Task 13.17 / BUG-67)
  benchmark_cfg = json.loads((run_chapter4_benchmark.CONFIG_DIR / "benchmark.json").read_text(encoding="utf-8"))
  assert benchmark_cfg["repetitions"] == run_chapter4_benchmark.REPETITIONS == 30

  # 1. --help delegates to cli.benchmark and exposes all flags without touching filesystem
  with pytest.raises(SystemExit) as exc:
    run_chapter4_benchmark.main(["--help"])
  assert exc.value.code == 0
  help_out = capsys.readouterr().out
  assert "--incremental" in help_out
  assert "--no-incremental" in help_out
  assert "--patience" in help_out
  assert "--seed" in help_out
  assert "--strategy-param" in help_out
  assert "--dry-run" in help_out

  script_path = repo_root / "benchmark_scripts" / "experiments-and-examples" / "chapter4" / "run_chapter4_benchmark.py"
  proc = subprocess.run(  # noqa: S603
    [sys.executable, str(script_path), "--help"],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0
  assert "--incremental" in proc.stdout
  assert "--no-incremental" in proc.stdout

  # 2. --output-dir + --dry-run redirects log strictly to tmp_path and leaves chapter4/results/ untouched
  results_dir = run_chapter4_benchmark.OUTPUT_DIR
  before_snapshot = (
    {p: (p.stat().st_mtime, p.stat().st_size) for p in results_dir.rglob("*")} if results_dir.exists() else {}
  )

  # Force CPU-only fallback to run hermetic dry-run under CUDA_VISIBLE_DEVICES=""
  monkeypatch.setattr("benchmarking.runner.orchestration.gpu_available", lambda: True)
  monkeypatch.setattr("benchmarking.runner.orchestration.cupy_backend", lambda: object())

  out_dir = tmp_path / "isolated_run"
  exit_code = run_chapter4_benchmark.main(["--output-dir", str(out_dir), "--dry-run"])
  assert exit_code == 0

  log_files = list(out_dir.glob("*.log"))
  assert len(log_files) == 1
  log_text = log_files[0].read_text(encoding="utf-8")
  assert "Loading configurations..." in log_text
  assert "Dry-run requested" in log_text
  assert log_text.count("Loading configurations...") == 1

  # Switching output_dir to a second path must not leak back to out_dir or duplicate lines
  out_dir2 = tmp_path / "isolated_run_2"
  exit_code2 = run_chapter4_benchmark.main(
    ["--output-dir", str(out_dir2), "--dry-run", "--seed", "99", "--patience", "15"]
  )
  assert exit_code2 == 0
  log_files2 = list(out_dir2.glob("*.log"))
  assert len(log_files2) == 1
  log_text2 = log_files2[0].read_text(encoding="utf-8")
  assert log_text2.count("Loading configurations...") == 1
  assert "Seed: 99" in log_text2
  assert "Early stop patience: 15 generations" in log_text2
  assert log_files[0].read_text(encoding="utf-8") == log_text

  if results_dir.exists():
    after_snapshot = {p: (p.stat().st_mtime, p.stat().st_size) for p in results_dir.rglob("*")}
    assert after_snapshot == before_snapshot


def test_chapter4_script_end_to_end_delegation_and_joint_comparison(
  fixture_tsp_dir: Path,
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """WHAT: run_chapter4_benchmark executes CPU smoke run via cli.benchmark and logs joint synthesis."""
  import logging

  run_chapter4_benchmark = _load_chapter4_script()

  gr17_path = fixture_tsp_dir / "gr17.tsp"
  out_dir = tmp_path / "e2e_ch4"
  with caplog.at_level(logging.INFO):
    exit_code = run_chapter4_benchmark.main(
      [
        "--output-dir",
        str(out_dir),
        "--tsp-files",
        str(gr17_path),
        "--algorithms",
        "CPU",
        "-r",
        "1",
        "--no-incremental",
        "--seed",
        "77",
      ]
    )
  assert exit_code == 0
  assert (out_dir / "checkpoints" / "gr17_CPU.json").is_file()
  assert (out_dir / "tables" / "chapter4_validation.md").is_file()
  assert (out_dir / "figures" / "scaling_time_gap.pdf").is_file()
  assert (out_dir / "figures" / "scaling_time_gap.svg").is_file()
  assert not (out_dir / "figures" / "scaling_time_gap.png").exists()
  assert (out_dir / "chapter4_validation.log").is_file()

  # Verify log_comparison joint synthesis with 3+ algorithms and 3+ problems
  synthetic_results = {
    p: {
      "CPU": {"mean_gap": 1.0, "mean_time": 10.0},
      "HybridNaive": {"mean_gap": 1.0, "mean_time": 4.0},
      "HybridOptimized": {"mean_gap": 1.0, "mean_time": 1.0},
      "FullGPUModular": {"mean_gap": 1.0, "mean_time": 0.2},
    }
    for p in ("p1", "p2", "p3", "p4")
  }
  caplog.clear()
  with caplog.at_level(logging.INFO):
    run_chapter4_benchmark.log_comparison(synthetic_results, baseline_algorithm="CPU")
  joint_log = "\n".join(r.getMessage() for r in caplog.records)
  assert "CHAPTER 4 JOINT INTERPRETIVE SUMMARY" in joint_log
  assert "50.00x" in joint_log
  assert "Joint verdict:" in joint_log

  # Verify log_comparison does not fabricate a 1.00x speedup when baseline is omitted (--skip-cpu)
  gpu_only_results = {
    p: {
      "HybridNaive": {"mean_gap": 1.0, "mean_time": 4.0},
      "HybridOptimized": {"mean_gap": 1.0, "mean_time": 1.0},
      "FullGPUModular": {"mean_gap": 1.0, "mean_time": 0.2},
    }
    for p in ("p1", "p2", "p3", "p4")
  }
  caplog.clear()
  with caplog.at_level(logging.INFO):
    run_chapter4_benchmark.log_comparison(gpu_only_results, baseline_algorithm="CPU")
  gpu_only_log = "\n".join(r.getMessage() for r in caplog.records)
  assert "FullGPUModular ranks fastest" in gpu_only_log
  assert "baseline CPU omitted" in gpu_only_log
  assert "1.00x geometric speedup" not in gpu_only_log
