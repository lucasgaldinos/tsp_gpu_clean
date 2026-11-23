#!/usr/bin/env python3
"""
Thesis LaTeX & BibTeX Integrity Gate
====================================

Validates structural and bibliographic invariants for the UFSC thesis under
`documents_tcc/tex/thesis/` without invoking `latexmk`:

1. **Balanced braces** — checks `{` / `}` balance (ignoring comments and escaped
   braces) across `second_draft.tex`, `parts/*.tex`, and `refs.bib`.
2. **Labels and cross-references** — verifies every `\\ref`, `\\eqref`, `\\pageref`,
   `\\hyperref[label]`, and custom thesis macro (`\\capref`, `\\secref`, `\\apref`,
   `\\figref`, `\\tabref`, `\\algref`) resolves to a defined `\\label{...}`, and flags
   duplicate `\\label` definitions.
3. **Included files and assets** — verifies every `\\input{...}` and
   `\\includegraphics[...]{...}` target exists on disk (resolving `assets/` via
   `\\graphicspath`).
4. **BibTeX citations & metadata** — checks that every cited key exists in
   `refs.bib`, that cited entries carry required fields for their BibTeX type
   (`@article` -> `journal`, `@inproceedings` -> `booktitle`, `@book` -> `publisher`,
   `@incollection` -> `booktitle`, `@phdthesis` -> `school`, `@techreport` ->
   `institution`), and that personal `note` fields cannot leak into the compiled
   bibliography (guarded by `\\DeclareSourcemap` / `\\clearfield{note}` in
   `second_draft.tex`).

Exit codes: 0 when clean, 1 when any finding is reported, 2 on missing inputs.

Usage:
    python -m benchmark_scripts.check_thesis
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
THESIS_DIR: Path = PROJECT_ROOT / "documents_tcc" / "tex" / "thesis"
MAIN_TEX: Path = THESIS_DIR / "second_draft.tex"
PARTS_DIR: Path = THESIS_DIR / "parts"
ASSETS_DIR: Path = THESIS_DIR / "assets"
BIB_FILE: Path = THESIS_DIR / "refs.bib"

CITE_RE: re.Pattern[str] = re.compile(
  r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}"
)
CITES_RE: re.Pattern[str] = re.compile(
  r"\\(?:auto|text|paren)?cites\*?(?:\[[^\]]*\])*((?:\{[^}]+\})+)"
)
LABEL_RE: re.Pattern[str] = re.compile(r"\\label\{([^}]+)\}")
REF_RE: re.Pattern[str] = re.compile(
  r"\\(?:eq|page|cap|sec|ap|fig|tab|alg)?ref\*?\{([^}]+)\}|\\hyperref\[([^\]]+)\]"
)
INPUT_RE: re.Pattern[str] = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE: re.Pattern[str] = re.compile(
  r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
)
ENTRY_HEAD_RE: re.Pattern[str] = re.compile(r"^@(\w+)\{([^,]+),", re.MULTILINE)

REQUIRED_BIB_FIELDS: dict[str, tuple[str, ...]] = {
  "article": ("journal",),
  "inproceedings": ("booktitle",),
  "incollection": ("booktitle",),
  "book": ("publisher",),
  "phdthesis": ("school",),
  "techreport": ("institution",),
}


def strip_tex_comments(text: str) -> str:
  """Remove unescaped `%` comments line by line."""
  cleaned: list[str] = []
  for line in text.splitlines():
    out_chars: list[str] = []
    escaped = False
    for ch in line:
      if ch == "%" and not escaped:
        break
      out_chars.append(ch)
      escaped = (ch == "\\") and not escaped
    cleaned.append("".join(out_chars))
  return "\n".join(cleaned)


def thesis_tex_files() -> list[Path]:
  """Return `second_draft.tex` followed by sorted `parts/*.tex`."""
  files: list[Path] = []
  if MAIN_TEX.exists():
    files.append(MAIN_TEX)
  if PARTS_DIR.exists():
    files.extend(sorted(PARTS_DIR.glob("*.tex")))
  return files


def check_balanced_braces(paths: list[Path]) -> list[str]:
  """Verify balanced `{` and `}` in TeX and BibTeX files."""
  findings: list[str] = []
  for path in paths:
    raw = path.read_text(encoding="utf-8")
    text = strip_tex_comments(raw) if path.suffix == ".tex" else raw
    depth = 0
    min_depth = 0
    escaped = False
    for ch in text:
      if not escaped:
        if ch == "{":
          depth += 1
        elif ch == "}":
          depth -= 1
          min_depth = min(min_depth, depth)
      escaped = (ch == "\\") and not escaped
    if depth != 0 or min_depth < 0:
      rel = path.relative_to(PROJECT_ROOT)
      findings.append(f"{rel}: unbalanced braces (final={depth}, min={min_depth})")
  return findings


def check_labels_and_refs(tex_files: list[Path]) -> tuple[list[str], int, int]:
  """Verify all cross-references resolve and no duplicate labels exist."""
  findings: list[str] = []
  label_locations: dict[str, list[Path]] = {}
  ref_uses: list[tuple[Path, str]] = []

  for path in tex_files:
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    for match in LABEL_RE.finditer(text):
      label = match.group(1).strip()
      label_locations.setdefault(label, []).append(path)
    for match in REF_RE.finditer(text):
      target = (match.group(1) or match.group(2) or "").strip()
      if target and not target.startswith("#"):
        ref_uses.append((path, target))

  for label, locations in sorted(label_locations.items()):
    if len(locations) > 1:
      rels = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in locations)
      findings.append(f"duplicate \\label{{{label}}} in {rels}")

  for path, target in ref_uses:
    if target not in label_locations:
      rel = path.relative_to(PROJECT_ROOT)
      findings.append(f"{rel}: unresolved reference `{target}`")

  return findings, len(label_locations), len(ref_uses)


def _resolve_tex_asset(raw_target: str, is_graphics: bool) -> Path | None:
  """Resolve a relative `\\input` or `\\includegraphics` path against `THESIS_DIR`."""
  target = raw_target.strip()
  candidates: list[Path] = [THESIS_DIR / target]
  if not Path(target).suffix:
    if is_graphics:
      for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps"):
        candidates.append(THESIS_DIR / f"{target}{ext}")
        candidates.append(ASSETS_DIR / f"{target}{ext}")
    else:
      candidates.append(THESIS_DIR / f"{target}.tex")
  if is_graphics:
    candidates.append(ASSETS_DIR / target)

  for cand in candidates:
    if cand.exists():
      return cand
  return None


def check_includes_and_assets(tex_files: list[Path]) -> list[str]:
  """Verify every `\\input` and `\\includegraphics` resolves to an existing file."""
  findings: list[str] = []
  for path in tex_files:
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    rel = path.relative_to(PROJECT_ROOT)
    for match in INPUT_RE.finditer(text):
      target = match.group(1).strip()
      if _resolve_tex_asset(target, is_graphics=False) is None:
        findings.append(f"{rel}: missing \\input target `{target}`")
    for match in GRAPHICS_RE.finditer(text):
      target = match.group(1).strip()
      if _resolve_tex_asset(target, is_graphics=True) is None:
        findings.append(f"{rel}: missing \\includegraphics target `{target}`")
  return findings


def _collect_cited_keys(tex_files: list[Path]) -> set[str]:
  """Extract all BibTeX keys cited across the thesis TeX files."""
  cited: set[str] = set()
  for path in tex_files:
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    for match in CITE_RE.finditer(text):
      for key in match.group(1).split(","):
        if key.strip():
          cited.add(key.strip())
    for match in CITES_RE.finditer(text):
      for group in re.findall(r"\{([^}]+)\}", match.group(1)):
        for key in group.split(","):
          if key.strip():
            cited.add(key.strip())
  return cited


def _note_suppression_enabled(main_tex_text: str) -> tuple[bool, bool]:
  """Return `(note_suppressed, annote_suppressed)` for `second_draft.tex`."""
  cleaned = strip_tex_comments(main_tex_text)
  note_suppressed = bool(
    re.search(r"fieldset\s*=\s*note\s*,\s*null", cleaned)
    or re.search(r"\\clearfield\{note\}", cleaned)
  )
  annote_suppressed = bool(
    re.search(r"fieldset\s*=\s*annote\s*,\s*null", cleaned)
    or re.search(r"\\clearfield\{annote\}", cleaned)
  )
  return note_suppressed, annote_suppressed


def check_bib_integrity(
  tex_files: list[Path],
  bib_path: Path,
  main_tex: Path,
) -> tuple[list[str], int, bool]:
  """Validate cited keys, metadata, legacy aliases, DOI/URL hygiene, and note suppression."""
  findings: list[str] = []
  cited = _collect_cited_keys(tex_files)
  bib_text = bib_path.read_text(encoding="utf-8")
  entries: dict[str, tuple[str, str]] = {}

  for block in re.split(r"\n(?=@)", bib_text):
    head = ENTRY_HEAD_RE.match(block.strip())
    if not head:
      continue
    etype = head.group(1).lower()
    key = head.group(2).strip()
    entries[key] = (etype, block)

  note_suppressed, annote_suppressed = _note_suppression_enabled(
    main_tex.read_text(encoding="utf-8")
  )
  both_suppressed = note_suppressed and annote_suppressed
  if not note_suppressed:
    findings.append("second_draft.tex: missing Biber/BibLaTeX suppression for `note` field")
  if not annote_suppressed:
    findings.append("second_draft.tex: missing Biber/BibLaTeX suppression for `annote` field")

  for key in sorted(cited):
    if key not in entries:
      findings.append(f"refs.bib: cited key `{key}` is missing")
      continue
    etype, block = entries[key]
    for req_field in REQUIRED_BIB_FIELDS.get(etype, ()):
      if not re.search(rf"^\s*{req_field}\s*=", block, re.MULTILINE | re.IGNORECASE):
        findings.append(f"refs.bib: @{etype}{{{key}}} missing required field `{req_field}`")
    for legacy_field, native_field in (
      ("archiveprefix", "eprinttype"),
      ("primaryclass", "eprintclass"),
    ):
      if re.search(rf"^\s*{legacy_field}\s*=", block, re.MULTILINE | re.IGNORECASE):
        findings.append(
          f"refs.bib: cited key `{key}` uses legacy `{legacy_field}`; replace with `{native_field}`"
        )
    doi_match = re.search(r"^\s*doi\s*=\s*\{([^}]+)\}", block, re.MULTILINE | re.IGNORECASE)
    url_match = re.search(r"^\s*url\s*=\s*\{([^}]+)\}", block, re.MULTILINE | re.IGNORECASE)
    if doi_match:
      doi_val = doi_match.group(1).strip()
      if doi_val.lower().startswith(("http://", "https://")):
        findings.append(f"refs.bib: cited key `{key}` has URL prefix inside `doi` (`{doi_val}`)")
      if url_match and re.search(r"https?://(?:dx\.)?doi\.org/", url_match.group(1), re.IGNORECASE):
        findings.append(
          f"refs.bib: cited key `{key}` duplicates `doi` inside `url` (`{url_match.group(1).strip()}`)"
        )

  return findings, len(cited), both_suppressed


def main() -> int:
  """Run all thesis LaTeX and BibTeX checks and report findings."""
  if not MAIN_TEX.exists() or not BIB_FILE.exists():
    print(f"ERROR: thesis sources not found under {THESIS_DIR}", file=sys.stderr)
    return 2

  tex_files = thesis_tex_files()
  brace_findings = check_balanced_braces([*tex_files, BIB_FILE])
  ref_findings, label_count, ref_count = check_labels_and_refs(tex_files)
  asset_findings = check_includes_and_assets(tex_files)
  bib_findings, cited_count, note_suppressed = check_bib_integrity(tex_files, BIB_FILE, MAIN_TEX)

  sections: tuple[tuple[str, list[str]], ...] = (
    ("balanced braces", brace_findings),
    ("labels & refs", ref_findings),
    ("includes & assets", asset_findings),
    ("bibtex & citations", bib_findings),
  )

  total = 0
  for name, items in sections:
    print(f"{name}: {len(items)}")
    for item in items:
      print(f"  - {item}")
    total += len(items)

  print(
    f"summary: tex_files={len(tex_files)}, labels={label_count}, refs={ref_count}, "
    f"cited_keys={cited_count}, note_suppression={'on' if note_suppressed else 'off'}"
  )
  return 1 if total else 0


if __name__ == "__main__":
  raise SystemExit(main())
