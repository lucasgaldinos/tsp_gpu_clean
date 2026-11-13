"""Centralized logging configuration for benchmarking executions (`src.benchmarking.logging_config`)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_benchmark_logging(
  output_dir: Path | str,
  benchmark_name: str = "benchmark",
  log_level: int = logging.INFO,
) -> Path:
  """Configure dual-target console (stdout) and persistent file logging for a benchmark session.

  Attaches a StreamHandler pointing to sys.stdout and a FileHandler appending
  to output_dir / f"{benchmark_name}.log". Avoids duplicate handlers on repeated calls.

  Args:
      output_dir: Directory where the log file will reside.
      benchmark_name: Base name for the log file ({benchmark_name}.log).
      log_level: Logging threshold level (default logging.INFO).

  Returns:
      Resolved Path to the configured log file.
  """
  resolved_dir = Path(output_dir).resolve()
  resolved_dir.mkdir(parents=True, exist_ok=True)
  log_file = resolved_dir / f"{benchmark_name}.log"

  root_logger = logging.getLogger()
  root_logger.setLevel(log_level)

  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

  # Detect and update or add stdout StreamHandler
  has_stream_handler = False
  for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
      stream = getattr(h, "stream", None)
      if stream is sys.stdout or stream is sys.stderr:
        has_stream_handler = True
        h.setLevel(log_level)
        h.setFormatter(formatter)

  if not has_stream_handler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

  # Detect and update or add FileHandler for this specific log file
  has_file_handler = False
  for h in root_logger.handlers:
    if isinstance(h, logging.FileHandler):
      try:
        if Path(h.baseFilename).resolve() == log_file:
          has_file_handler = True
          h.setLevel(log_level)
          h.setFormatter(formatter)
      except (OSError, ValueError) as err:
        logging.debug("Could not resolve log file path for handler %s: %s", h, err)

  if not has_file_handler:
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

  return log_file
