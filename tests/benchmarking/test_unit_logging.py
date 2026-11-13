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
