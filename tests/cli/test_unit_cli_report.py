"""Unit tests for the `cli.report` standalone checkpoint-to-report entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cli.report import main as report_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPTER4_STYLE_PATH = PROJECT_ROOT / "benchmark_scripts" / "configs" / "chapter4" / "report_style.json"


def _write_synthetic_checkpoints(ckpt_dir: Path, *, with_optimal: bool) -> None:
  """Write synthetic 2-problem x 3-algorithm checkpoints into ``ckpt_dir``."""
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  problems = [("berlin52", 52, 7542.0), ("kroA100", 100, 21282.0)]
  algos = [
    ("CPU", 4.0, 1.0),
    ("HybridOptimized", 0.4, 0.9),
    ("FullGPUModular", 0.08, 0.8),
  ]
  for prob_name, size, opt in problems:
    for alg_name, base_time, gap_val in algos:
      cost_val = opt * (1.0 + gap_val / 100.0)
      payload: dict[str, Any] = {
        "problem_name": prob_name,
        "problem_size": size,
        "optimal_cost": opt if with_optimal else None,
        "algorithm": alg_name,
        "backend": "CPU" if alg_name == "CPU" else "GPU",
        "repetitions": 3,
        "successful_runs": 3,
        "timestamp": "2026-11-23T10:00:00",
        "max_generations_configured": 100,
        "mean_time": base_time,
        "std_time": base_time * 0.05,
        "min_time": base_time * 0.95,
        "max_time": base_time * 1.05,
        "mean_cost": cost_val,
        "std_cost": 10.0,
        "best_cost": cost_val - 10.0,
        "worst_cost": cost_val + 10.0,
        "mean_gap": gap_val if with_optimal else None,
        "std_gap": 0.1 if with_optimal else None,
        "best_gap": (gap_val - 0.1) if with_optimal else None,
        "mean_generations": 50.0,
        "std_generations": 2.0,
        "raw_times": [base_time * 0.95, base_time, base_time * 1.05],
        "raw_costs": [cost_val - 10.0, cost_val, cost_val + 10.0],
        "raw_gaps": [gap_val - 0.1, gap_val, gap_val + 0.1] if with_optimal else None,
        "algorithm_config": {"population_size": 256},
        "strategies": {"local_search": "TwoOpt"},
      }
      (ckpt_dir / f"{prob_name}_{alg_name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_report_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
  """WHAT: Verify `tsp-report --help` exposes neutral report flags cleanly.

  WHY: Ensures the standalone checkpoint-to-report CLI advertises composable artifact
  selection and external styling without hardcoded thesis presets.
  EXPECTED: Exits with status 0 and prints --checkpoints, --report-style, and --reports.
  """
  with pytest.raises(SystemExit) as exc:
    report_main(["--help"])
  assert exc.value.code == 0
  out = capsys.readouterr().out
  assert "tsp-report" in out
  assert "--checkpoints" in out
  assert "--report-style" in out
  assert "--reports" in out
  assert "--locale" not in out


def test_report_cli_applies_external_report_style_without_src_hardcoding(tmp_path: Path) -> None:
  """WHAT: Verify `tsp-report --report-style` renders pt-BR thesis labels purely from external JSON.

  WHY: Confirms dissertation formatting is 100% injected from benchmark_scripts/configs/chapter4/
  with zero Portuguese strings hardcoded inside src/.
  EXPECTED: Tables, macros, CD TikZ diagrams, and PDF/SVG scaling plots carry injected pt-BR labels.
  """
  ckpt_dir = tmp_path / "run" / "checkpoints"
  _write_synthetic_checkpoints(ckpt_dir, with_optimal=True)

  exit_code = report_main(
    [
      "--checkpoints",
      str(ckpt_dir),
      "--benchmark-name",
      "demo_styled",
      "--baseline",
      "CPU",
      "--report-style",
      str(CHAPTER4_STYLE_PATH),
    ]
  )
  assert exit_code == 0

  out_root = tmp_path / "run"
  assert (out_root / "problem_statistics" / "berlin52_stats.json").is_file()
  assert (out_root / "problem_statistics" / "kroA100_stats.json").is_file()
  assert (out_root / "tables" / "demo_styled.md").is_file()
  tex_text = (out_root / "tables" / "demo_styled.tex").read_text(encoding="utf-8")
  assert "\\label{tab:chapter4_summary}" in tex_text

  macros_text = (out_root / "tables" / "benchmark_macros.tex").read_text(encoding="utf-8")
  assert "{,}" in macros_text

  cd_time = (out_root / "tables" / "demo_styled_cd_diagram_time.tikz").read_text(encoding="utf-8")
  assert "Posto médio" in cd_time
  assert "Totalmente em GPU" in cd_time
  assert (out_root / "figures" / "scaling_time_gap.pdf").read_bytes()[:4] == b"%PDF"
  assert b"<svg" in (out_root / "figures" / "scaling_time_gap.svg").read_bytes()[:512]


def test_report_cli_defaults_to_neutral_english_and_supports_selective_reports(tmp_path: Path) -> None:
  """WHAT: Verify neutral English defaults, unknown-optimal checkpoints, and selective `--reports`.

  WHY: Guarantees arbitrary custom .tsp cohorts without known optima render cleanly and
  only requested report artifacts are written.
  EXPECTED: Neutral English headings/labels, em-dash gap cells, cost-based CD diagram, and no unrequested macros.
  """
  ckpt_dir = tmp_path / "checkpoints"
  out_dir = tmp_path / "report_out"
  _write_synthetic_checkpoints(ckpt_dir, with_optimal=False)

  exit_code = report_main(
    [
      "--checkpoints",
      str(ckpt_dir),
      "--output-dir",
      str(out_dir),
      "--benchmark-name",
      "unknown_opt",
      "--reports",
      "markdown",
      "latex",
      "cd",
      "plots",
    ]
  )
  assert exit_code == 0
  md_text = (out_dir / "tables" / "unknown_opt.md").read_text(encoding="utf-8")
  assert "| — |" in md_text
  assert "FullGPUModular" in md_text
  tex_text = (out_dir / "tables" / "unknown_opt.tex").read_text(encoding="utf-8")
  assert "\\label{tab:summary}" in tex_text
  assert not (out_dir / "tables" / "benchmark_macros.tex").exists()
  cd_gap = (out_dir / "tables" / "unknown_opt_cd_diagram_gap.tikz").read_text(encoding="utf-8")
  assert "Average rank" in cd_gap
  assert "Mean Cost" in cd_gap
  assert (out_dir / "figures" / "scaling_time_gap.pdf").is_file()


def test_report_cli_validates_config_dir_avoids_spurious_checkpoints_and_sets_log_level(tmp_path: Path) -> None:
  """WHAT: `tsp-report` rejects missing `--config-dir`, avoids empty `checkpoints/`, sets log level.

  WHY: BUG-100 — Nonexistent `--config-dir` previously succeeded silently, `CheckpointManager`
  created an empty `output_root/checkpoints/` folder, and `basicConfig` was a no-op with active handlers.
  """
  import logging

  in_dir = tmp_path / "raw_jsons"
  out_dir = tmp_path / "out_reports"
  _write_synthetic_checkpoints(in_dir, with_optimal=True)

  # 1. Missing --config-dir fails fast with exit code 2 and creates nothing
  rc_bad = report_main(
    [
      "-i",
      str(in_dir),
      "-o",
      str(out_dir),
      "--config-dir",
      str(tmp_path / "DOES_NOT_EXIST"),
      "--reports",
      "none",
    ]
  )
  assert rc_bad == 2
  assert not (out_dir / "checkpoints").exists()

  # 2. Valid run creates problem_statistics/ without a spurious checkpoints/ dir and updates root log level
  root_logger = logging.getLogger()
  prev_level = root_logger.level
  try:
    root_logger.setLevel(logging.WARNING)
    rc_ok = report_main(
      [
        "-i",
        str(in_dir),
        "-o",
        str(out_dir),
        "--reports",
        "none",
        "--log-level",
        "DEBUG",
      ]
    )
    assert rc_ok == 0
    assert (out_dir / "problem_statistics" / "berlin52_stats.json").is_file()
    assert not (out_dir / "checkpoints").exists()
    assert root_logger.level == logging.DEBUG
  finally:
    root_logger.setLevel(prev_level)
