#!/usr/bin/env python3
"""
Documentation Drift Gate
========================

Detects six classes of documentation rot that the test suite cannot see:

1. **Stale docstrings** — module/class/function docstrings in `src/` that name
   identifiers which no longer exist in the tree.
2. **Dead links** — relative Markdown links in the plan and walkthroughs whose
   target file is absent.
3. **Undefined symbols** — `DP-*`, `BUG-*` or `DP<phase>-*` identifiers used in the
   plan or walkthroughs but never registered in `table-of-symbols.plan.md`.
4. **Broken symbol anchors** — `[file.py::Symbol](path)` links whose label names a
   different file than the link target, or whose symbols do not exist in the target.
   Remaining `#L` line anchors are reported as a non-fatal migration summary.
5. **Broken heading anchors** — `[file.md#fragment](path)` links whose fragment is not
   a heading in the target document, resolved with the GitHub slug algorithm.
6. **Markdown defects** — table rows whose cell count disagrees with the header
   (MD056) and `[[Name]]` wiki links, which VS Code reports as unresolved reference
   links because no link definition exists.

Exit codes: 0 when clean, 1 when any finding is reported, 2 on missing inputs.

Usage:
    python -m benchmark_scripts.check_docs
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Iterator

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
PLAN_DIR: Path = PROJECT_ROOT / "documents_tcc" / "plan"
WALKTHROUGH_DIR: Path = PROJECT_ROOT / "documentation" / "walkthroughs"
SYMBOL_TABLE: Path = PLAN_DIR / "table-of-symbols.plan.md"
MARKDOWN_ROOTS: tuple[Path, ...] = (
  PROJECT_ROOT / "documents_tcc",
  PROJECT_ROOT / "documentation",
)
MARKDOWN_EXCLUDED_PARTS: frozenset[str] = frozenset(
  {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)

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
LABELLED_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<target>[^)\s]+)\)")
SYMBOL_LABEL = re.compile(
  r"^(?P<base>[A-Za-z0-9_.-]+\.(?P<ext>py|cu|cuh))::(?P<symbols>.+)$"
)
LINE_ANCHOR = re.compile(r"\]\((?P<path>[^)#]+)#L\d+(?:-L\d+)?\)")
CUDA_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
CUDA_CALLABLE = re.compile(
  r"(?:__global__|__device__|__host__)\b[^;{}]*?\b(?P<name>[A-Za-z_]\w*)\s*\("
)
CUDA_TYPE = re.compile(r"\b(?:struct|class)\s+(?P<name>[A-Za-z_]\w*)\s*(?::|\{)")

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
# GitHub strips this punctuation class from a heading before slugging it; Unicode
# letters, marks, digits, `_`, and `-` survive, which is why pt-BR headings keep
# their accents in the anchor.
GFM_PUNCTUATION = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\\'!\"#$%&()*+,./:;<=>?@\[\]^`{|}~]")
HEADING_FRAGMENT = re.compile(r"\]\((?P<target>[^)\s]*\.md)#(?P<fragment>[^)\s]+)\)")
SAME_FILE_FRAGMENT = re.compile(r"\]\(#(?P<fragment>[^)\s]+)\)")
LINE_FRAGMENT = re.compile(r"L\d+(?:-L\d+)?")
# Pandoc/Quarto headings carry an explicit anchor (`## Título {#sec-x}`); that id,
# not the slugged title, is what a cross-reference must use.
PANDOC_ANCHOR = re.compile(r"\{#(?P<anchor>[A-Za-z0-9_.:-]+)[^}]*\}\s*$")
# Hand-authored HTML anchors (`<a id="fig-x"></a>`) are real fragment targets too.
HTML_ANCHOR = re.compile(r"<a\s+[^>]*?(?:id|name)=[\"'](?P<anchor>[^\"']+)[\"']", re.IGNORECASE)
# A heading renders to its link labels, not to the link targets, so `[x](y)` must be
# reduced to `x` before slugging; emphasis markers and backticks never render either.
INLINE_LINK = re.compile(r"\[(?P<label>[^\]]*)\]\([^)]*\)")
EMPHASIS = re.compile(r"[`*~]")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
CELL_SPLIT = re.compile(r"(?<!\\)\|")
WIKI_LINK = re.compile(r"(?<![\w\[])\[\[(?P<name>[^\]\n]{1,80})\]\](?!\])")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")


def markdown_sources() -> list[Path]:
  """Return every documentation `.md` file the Markdown rules govern."""
  discovered: list[Path] = list(PROJECT_ROOT.glob("*.md"))
  for root in MARKDOWN_ROOTS:
    discovered.extend(root.rglob("*.md"))
  return sorted(
    path for path in set(discovered) if not MARKDOWN_EXCLUDED_PARTS & set(path.parts)
  )


def iter_markdown_lines(path: Path) -> Iterator[tuple[int, str]]:
  """Yield `(line_number, line)` outside YAML frontmatter and fenced code blocks."""
  lines: list[str] = path.read_text(encoding="utf-8").splitlines()
  start: int = 0
  if lines and lines[0].strip() == "---":
    for index in range(1, len(lines)):
      if lines[index].strip() == "---":
        start = index + 1
        break
  in_fence: bool = False
  for index in range(start, len(lines)):
    line: str = lines[index]
    if line.lstrip().startswith("```"):
      in_fence = not in_fence
      continue
    if in_fence:
      continue
    yield index + 1, line


def heading_slugs(text: str) -> set[str]:
  """Return the anchors generated for the headings in `text`.

  GitHub slugs the visible title; Pandoc/Quarto documents declare the anchor
  explicitly with `{#id}`, and that id wins over the slug.
  """
  slugs: set[str] = set()
  seen: dict[str, int] = {}
  slugs.update(match.group("anchor") for match in HTML_ANCHOR.finditer(text))
  for match in MARKDOWN_HEADING.finditer(text):
    title: str = match.group("title")
    explicit = PANDOC_ANCHOR.search(title)
    if explicit:
      slugs.add(explicit.group("anchor"))
      continue
    rendered: str = EMPHASIS.sub("", INLINE_LINK.sub(lambda link: link.group("label"), title))
    base: str = GFM_PUNCTUATION.sub("", rendered.lower().strip()).replace(" ", "-")
    if base in seen:
      seen[base] += 1
      slugs.add(f"{base}-{seen[base]}")
    else:
      seen[base] = 0
      slugs.add(base)
  return slugs


def check_gfm_heading_anchors() -> list[str]:
  """Verify that every heading anchor resolves, for cross-file and same-file links."""
  findings: list[str] = []
  cache: dict[Path, set[str]] = {}
  for path in markdown_sources():
    rel = path.relative_to(PROJECT_ROOT)
    for number, line in iter_markdown_lines(path):
      candidates: list[tuple[Path, str]] = [
        (
          (path.parent / match.group("target")).resolve() if match.group("target") else path,
          match.group("fragment"),
        )
        for match in HEADING_FRAGMENT.finditer(line)
      ]
      candidates.extend((path, match.group("fragment")) for match in SAME_FILE_FRAGMENT.finditer(line))
      for target, fragment in candidates:
        if LINE_FRAGMENT.fullmatch(fragment):
          continue
        if not target.exists():
          continue  # reported by check_dead_links
        if target not in cache:
          cache[target] = heading_slugs(target.read_text(encoding="utf-8"))
        if fragment not in cache[target]:
          findings.append(
            f"{rel}:{number} unknown heading anchor `#{fragment}` in"
            f" `{target.relative_to(PROJECT_ROOT)}`"
          )
  return findings


def table_cell_count(line: str) -> int:
  """Return the number of GFM table cells in `line`, ignoring escaped pipes."""
  stripped: str = line.strip()
  if stripped.startswith("|"):
    stripped = stripped[1:]
  if stripped.endswith("|"):
    stripped = stripped[:-1]
  return len(CELL_SPLIT.split(stripped))


def check_markdown_tables() -> list[str]:
  """Report table rows whose cell count disagrees with their header (MD056)."""
  findings: list[str] = []
  for path in markdown_sources():
    rel = path.relative_to(PROJECT_ROOT)
    expected: int | None = None
    for number, line in iter_markdown_lines(path):
      if TABLE_SEPARATOR.match(line):
        expected = table_cell_count(line)
        continue
      if TABLE_ROW.match(line):
        if expected is not None and table_cell_count(line) != expected:
          findings.append(
            f"{rel}:{number} table row has {table_cell_count(line)} cells,"
            f" expected {expected} (unescaped `|`?)"
          )
      else:
        expected = None
  return findings


def check_wiki_links() -> list[str]:
  """Report `[[Name]]` wiki links, which render as unresolved reference links.

  Inline code is stripped first: `Callable[[int, Any], None]` is a type annotation,
  not a wiki link.
  """
  findings: list[str] = []
  for path in markdown_sources():
    rel = path.relative_to(PROJECT_ROOT)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      for match in WIKI_LINK.finditer(INLINE_CODE.sub("", line)):
        findings.append(f"{rel}:{number} wiki link `[[{match.group('name')}]]`")
  return findings


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


def python_symbols(path: Path) -> set[str]:
  """Return the module, class, function, and constant names defined in `path`."""
  try:
    tree = ast.parse(path.read_text(encoding="utf-8"))
  except SyntaxError:
    return set()

  names: set[str] = set()

  def visit(body: list[ast.stmt], prefix: str) -> None:
    for node in body:
      if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        names.add(prefix + node.name)
        if isinstance(node, ast.ClassDef):
          visit(node.body, f"{prefix}{node.name}.")
      elif isinstance(node, ast.Assign):
        names.update(
          prefix + target.id
          for target in node.targets
          if isinstance(target, ast.Name)
        )
      elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.add(prefix + node.target.id)
      elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        names.add(prefix + node.name.id)

  visit(tree.body, "")
  return names


def cuda_symbols(path: Path) -> set[str]:
  """Return the kernel, device-function, and named CUDA type symbols in `path`."""
  text = CUDA_COMMENT.sub("", path.read_text(encoding="utf-8"))
  return {match.group("name") for match in CUDA_CALLABLE.finditer(text)} | {
    match.group("name") for match in CUDA_TYPE.finditer(text)
  }


def check_symbol_anchors() -> list[str]:
  """Verify that `[file.py::Symbol](path)` labels resolve inside the linked file."""
  findings: list[str] = []
  sources = sorted(PLAN_DIR.glob("*.md")) + sorted(WALKTHROUGH_DIR.glob("*.md"))
  for path in sources:
    rel = path.relative_to(PROJECT_ROOT)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      for match in LABELLED_LINK.finditer(line):
        label = match.group("label").strip().strip("`")
        parsed = SYMBOL_LABEL.match(label)
        target = match.group("target")
        if parsed is None or target.startswith(("http", "https", "file:", "mailto:", "#")):
          continue
        relative_target = target.split("#", 1)[0]
        target_name = Path(relative_target).name
        if target_name != parsed.group("base"):
          findings.append(
            f"{rel}:{number} label `{parsed.group('base')}` does not match"
            f" link target `{target_name}`"
          )
          continue
        file_path = (path.parent / relative_target).resolve()
        if not file_path.exists():
          continue  # reported by check_dead_links
        available = (
          cuda_symbols(file_path)
          if parsed.group("ext") in {"cu", "cuh"}
          else python_symbols(file_path)
        )
        for raw_symbol in parsed.group("symbols").split(","):
          symbol = raw_symbol.strip()
          if symbol and symbol not in available:
            findings.append(
              f"{rel}:{number} unknown symbol `{symbol}` in `{target_name}`"
            )
  return findings


def line_anchor_counts() -> dict[str, int]:
  """Count remaining `#L` links by target suffix; a migration aid, never fatal."""
  counts: dict[str, int] = {}
  sources = sorted(PLAN_DIR.glob("*.md")) + sorted(WALKTHROUGH_DIR.glob("*.md"))
  for path in sources:
    for line in path.read_text(encoding="utf-8").splitlines():
      for match in LINE_ANCHOR.finditer(line):
        suffix = Path(match.group("path")).suffix or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
  return counts


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
    ("symbol anchors", check_symbol_anchors()),
    ("heading anchors", check_gfm_heading_anchors()),
    ("markdown tables", check_markdown_tables()),
    ("wiki links", check_wiki_links()),
  )

  total = 0
  for label, findings in sections:
    print(f"{label}: {len(findings)}")
    for finding in findings:
      print(f"  {finding}")
    total += len(findings)

  pending_anchors = line_anchor_counts()
  if pending_anchors:
    pending_total = sum(pending_anchors.values())
    breakdown = ", ".join(
      f"{suffix} {count}"
      for suffix, count in sorted(
        pending_anchors.items(), key=lambda item: (-item[1], item[0])
      )
    )
    print(
      "\nline anchors (non-fatal, migrate to `path::symbol` / `path#heading`):"
      f" {pending_total} ({breakdown})"
    )

  if total:
    print(f"\nFAIL: {total} documentation drift finding(s)")
    return 1

  print("\nOK: no documentation drift detected")
  return 0


if __name__ == "__main__":
  sys.exit(main())
