"""Unit tests for unified dual-target logging infrastructure (`src.benchmarking.logging_config`).

WHAT: Tests `setup_benchmark_logging` dual-target emission to stdout and persistent log file on disk.
WHY: DP12-8 / Task 12.15 — Logging must be durable across unexpected terminations without losing
terminal visibility or duplicating log handlers on repeated re-entry.
EXPECTED: Both sys.stdout and output_dir / f"{benchmark_name}.log" receive identical formatted records,
and repeated calls update configuration without creating duplicate handlers.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

from benchmarking.logging_config import setup_benchmark_logging

if TYPE_CHECKING:
  from pathlib import Path


def test_setup_benchmark_logging_emits_to_stdout_and_disk(tmp_path: Path) -> None:
  """WHAT: Logging emits simultaneously to sys.stdout and file on disk.

  WHY: Guarantees post-mortem durability for unattended/detached runs while preserving
  immediate console feedback.
  EXPECTED: Message appears in stdout buffer and in log file on disk.
  """
  output_dir = tmp_path / "logs"
  benchmark_name = "test_run"

  stdout_capture = io.StringIO()
  with patch("sys.stdout", stdout_capture):
    log_path = setup_benchmark_logging(output_dir, benchmark_name=benchmark_name, log_level=logging.INFO)
    logging.info("Benchmarking dual target test message 12345")

  assert log_path.exists()
  file_content = log_path.read_text(encoding="utf-8")
  assert "Benchmarking dual target test message 12345" in file_content
  assert "INFO" in file_content

  stdout_content = stdout_capture.getvalue()
  assert "Benchmarking dual target test message 12345" in stdout_content


def test_setup_benchmark_logging_no_duplicate_handlers(tmp_path: Path) -> None:
  """WHAT: Calling setup_benchmark_logging multiple times does not duplicate handlers.

  WHY: Repeated entries (e.g. CLI followed by orchestrator) must not print duplicate lines.
  EXPECTED: Exactly one line per log call in stdout and file.
  """
  output_dir = tmp_path / "logs_repeat"
  benchmark_name = "repeat_run"

  stdout_capture = io.StringIO()
  with patch("sys.stdout", stdout_capture):
    # Call twice
    log_path1 = setup_benchmark_logging(output_dir, benchmark_name=benchmark_name)
    log_path2 = setup_benchmark_logging(output_dir, benchmark_name=benchmark_name)
    assert log_path1 == log_path2

    logging.info("Unique single message 98765")

  file_content = log_path1.read_text(encoding="utf-8")
  assert file_content.count("Unique single message 98765") == 1

  stdout_content = stdout_capture.getvalue()
  assert stdout_content.count("Unique single message 98765") == 1


def test_setup_benchmark_logging_switches_directory_and_preserves_level(tmp_path: Path) -> None:
  """WHAT: Switching output_dir closes the old managed FileHandler and preserves log_level when None is passed.

  WHY: Task 13.22 (BUG-74) — Re-running in a new output directory or nesting CLI -> orchestration
  must never split logs across directories, duplicate lines, or clobber --log-level DEBUG.
  """
  dir_a = tmp_path / "run_a"
  dir_b = tmp_path / "run_b"

  log_a = setup_benchmark_logging(dir_a, benchmark_name="bench", log_level=logging.DEBUG)
  logging.debug("debug message A")

  # Nested call with log_level=None must preserve DEBUG level and not duplicate handlers
  log_a_again = setup_benchmark_logging(dir_a, benchmark_name="bench", log_level=None)
  assert log_a_again == log_a
  logging.debug("debug message A2")

  # Switching to dir_b must detach dir_a handler so dir_a receives nothing for dir_b
  log_b = setup_benchmark_logging(dir_b, benchmark_name="bench", log_level=logging.INFO)
  logging.info("info message B")

  text_a = log_a.read_text(encoding="utf-8")
  text_b = log_b.read_text(encoding="utf-8")
  assert text_a.count("debug message A2") == 1
  assert "info message B" not in text_a
  assert text_b.count("info message B") == 1
