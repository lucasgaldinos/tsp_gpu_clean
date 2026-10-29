---
description: Whenever an agent must explore the workspace
applyTo: 'Whenever an agent must explore the workspace', 'explore', 'gathering_information'
trigger: model_decision
model: deepseek-flash
---

# Workspace exploration

Hard rules for exploring the workspace. The full workflow lives in the `explore` skill; this file holds the non-negotiable constraints. Prefix all terminal commands with `rtk` for token savings.

Default `eza` exclude set (non-source noise): `.git|.venv|.github|.agents|node_modules|__pycache__|.mypy_cache|.pytest_cache|.ruff_cache|build|dist`.

## Tree listing (`eza`)

- **First pass** — broad tree at depth 3, excluding noise:

  ```bash
  rtk eza . -I ".git|.venv|.github|.agents|node_modules|__pycache__|.mypy_cache|.pytest_cache|.ruff_cache|build|dist" -T -L 3
  ```

  Increase `-L` depth on demand only if the first pass is insufficient.
- **Second pass** — when a specific folder/path is given, list it flat (no `-T`, no `-L`):

  ```bash
  rtk eza <folder/path> -I ".git|.venv|.github|.agents|node_modules|__pycache__|.mypy_cache|.pytest_cache|.ruff_cache|build|dist"
  ```

## Targeted search

- Search before reading: use `grep`/`file_search` to locate symbols and files first; read a file only when its path is known or full context is needed.
- Prefer targeted reads with explicit line ranges over whole-file dumps.

## Parallelize and stop early

- Batch independent tool calls (multiple searches/reads at once).
- Stop once sufficient context is gathered — targeted, not exhaustive.

## Output

- Report findings as absolute file links plus reusable symbols.
- Add a short synthesized summary of what was found.
- Direct answers only; no tour.
