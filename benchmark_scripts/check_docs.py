#!/usr/bin/env python3
"""
Documentation Drift Gate
========================

Detects three classes of documentation rot that the test suite cannot see:

1. **Stale docstrings** — module/class/function docstrings in `src/` that name
   identifiers which no longer exist in the tree.
2. **Dead links** — relative Markdown links in the plan and walkthroughs whose
   target file is absent.
3. **Undefined symbols** — `DP-*`, `BUG-*` or `DP<phase>-*` identifiers used in the
   plan or walkthroughs but never registered in `table-of-symbols.plan.md`.

Exit codes: 0 when clean, 1 when any finding is reported, 2 on missing inputs.

Usage:
    python -m benchmark_scripts.check_docs
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
PLAN_DIR: Path = PROJECT_ROOT / "documents_tcc" / "plan"
WALKTHROUGH_DIR: Path = PROJECT_ROOT / "documentation" / "walkthroughs"
SYMBOL_TABLE: Path = PLAN_DIR / "table-of-symbols.plan.md"

# Identifiers that were renamed or deleted and must not reappear in a docstring.
STALE_DOCSTRING_TOKENS: tuple[str, ...] = (
  "benchmarking_v2",
  "benchmarks_v2",
  "default_config_dir",
  "gpu_probe",
  "strategy_fixtures",
  "chapter4_validation.py",
  "DEFAULT_SELECTED_TSP_FILES",
  "selection_strategies.py",
  "evaluation_strategies.py",
  "monolithic/full_gpu_monolithic",
)

MARKDOWN_LINK = re.compile(r"\]\((?P<path>[^)#]+?)(?:#L\d+(?:-L\d+)?)?\)")
SYMBOL_ID = re.compile(r"\b(?:DP-\d+|DP\d+-\d+|BUG-\d+)\b")


def collect_docstrings(path: Path) -> list[tuple[int, str, str]]:
  """Return `(line, kind, text)` for every docstring in `path`."""
  try:
    tree = ast.parse(path.read_text(encoding="utf-8"))
  except SyntaxError:
    return []

  found: list[tuple[int, str, str]] = []
  module_doc = ast.get_docstring(tree)
  if module_doc:
    found.append((1, "module", module_doc))
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
      doc = ast.get_docstring(node)
      if doc:
        found.append((node.lineno, type(node).__name__, doc))
  return found


def check_stale_docstrings() -> list[str]:
  findings: list[str] = []
  for path in sorted(SRC_DIR.rglob("*.py")):
    if "__pycache__" in path.parts:
      continue
    for line, kind, text in collect_docstrings(path):
      lowered = text.lower()
      for token in STALE_DOCSTRING_TOKENS:
        if token.lower() in lowered:
          rel = path.relative_to(PROJECT_ROOT)
          findings.append(f"{rel}:{line} [{kind}] docstring names stale `{token}`")
          break
  return findings


def check_dead_links() -> list[str]:
  findings: list[str] = []
  sources = sorted(PLAN_DIR.glob("*.md")) + sorted(WALKTHROUGH_DIR.glob("*.md"))
  for path in sources:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      for match in MARKDOWN_LINK.finditer(line):
        raw = match.group("path").strip()
        if raw.startswith(("http", "file:", "mailto", "#")):
          continue
        if not (path.parent / raw).resolve().exists():
          findings.append(f"{path.relative_to(PROJECT_ROOT)}:{number} dead link `{raw}`")
  return findings


def registered_symbols() -> set[str]:
  if not SYMBOL_TABLE.exists():
    return set()
  text = SYMBOL_TABLE.read_text(encoding="utf-8")
  return set(SYMBOL_ID.findall(text))


def check_undefined_symbols() -> list[str]:
  known = registered_symbols()
  findings: list[str] = []
  sources = sorted(PLAN_DIR.glob("*.md")) + sorted(WALKTHROUGH_DIR.glob("*.md"))
  for path in sources:
    if path.name == SYMBOL_TABLE.name:
      continue
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      for symbol in SYMBOL_ID.findall(line):
        if symbol not in known:
          findings.append(
            f"{path.relative_to(PROJECT_ROOT)}:{number} undefined symbol `{symbol}`"
          )
  return findings


def main() -> int:
  """Run every documentation check and report the aggregated result."""
  if not SYMBOL_TABLE.exists():
    print(f"missing symbol table: {SYMBOL_TABLE}", file=sys.stderr)
    return 2

  sections = (
    ("stale docstrings", check_stale_docstrings()),
    ("dead links", check_dead_links()),
    ("undefined symbols", check_undefined_symbols()),
  )

  total = 0
  for label, findings in sections:
    print(f"{label}: {len(findings)}")
    for finding in findings:
      print(f"  {finding}")
    total += len(findings)

  if total:
    print(f"\nFAIL: {total} documentation drift finding(s)")
    return 1

  print("\nOK: no documentation drift detected")
  return 0


if __name__ == "__main__":
  sys.exit(main())
