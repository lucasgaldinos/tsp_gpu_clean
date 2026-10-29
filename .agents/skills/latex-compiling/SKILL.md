---
name: latex-compiling
description: >
  Repo-scoped LaTeX compilation workflow for this project. Use when diagnosing
  or fixing LaTeX/TeX builds, LuaLaTeX, latexmk, biblatex/biber, abnTeX2,
  ltmermaid, Mermaid rendering, fonts, or thesis PDF generation.
argument-hint: "[LaTeX target or error] [what changed] [what validation is needed]"
---

# LaTeX Compiling — Repo-Scoped Workflow

Use this skill for LaTeX build/debug work in this repository. Keep generic task management in `task-managing`; keep LaTeX-specific compile discipline here.

## Project Defaults

- Canonical thesis source: `documents_tcc/tex/second_draft.tex`.
- Compiler orchestration: `latexmk` with LuaLaTeX.
- Default build command from `documents_tcc/tex/`:

  ```bash
  latexmk -lualatex -shell-escape second_draft.tex
  ```

- Do not guess `pdflatex`, `xelatex`, or a Pandoc-first flow when the task is about the current thesis build.
- Do not use `-interaction=nonstopmode` for diagnosis. It can continue past the first meaningful failure and create cascaded errors that waste the debugging pass.
- Full official builds of `second_draft.tex` belong in explicit validation tasks or require user approval. Subsystem diagnosis happens in a sandbox first.

## Compile Discipline

Before each compile, state the hypothesis the run is testing. A compile without a new hypothesis is noise.

1. For breaking changes, rendering subsystems, new packages, bibliography changes, fonts, or Mermaid/diagram work, create a temp copy or dummy file first.
2. Compile once with the project default command or the smallest equivalent dummy command.
3. If it fails, read evidence before retrying: `.log` for the first real LaTeX error and package stack; `.fls` for file lookup/path problems; `.bcf`, `.blg`, and `biber` output for bibliography problems; generated `.mmd`, Mermaid CLI output, and browser/Puppeteer errors for `ltmermaid`; fontspec/luaotfload output and `tlmgr info` for missing fonts/packages.
4. Fix one root cause, then compile at most one more time for that hypothesis.
5. After two failed compiles for the same hypothesis, stop compiling. Use documentation, web search, GitHub/code search, or package docs for the concrete error. If the next action is still unclear, use `ask-clarification-questions` and `vscode_askQuestions`.

## Sandbox Rules

- Use `tmp/` or another disposable directory for temporary copies and dummy files.
- Keep generated auxiliary files out of the official thesis directory until the subsystem is understood.
- A dummy file should isolate the package or feature being tested: preamble, one minimal body case, and the smallest required assets.
- When moving a sandbox-proven fix to the official file, port only the proven delta.
- Delete temporary sandboxes after the walkthrough has captured the relevant evidence, but ask the user for confirmation before removal.

## Mermaid and `ltmermaid`

For Mermaid diagrams:

1. Validate the Mermaid source independently when possible with `mmdc` or the generated `.mmd` file.
2. Use a dummy LuaLaTeX document with `ltmermaid` before compiling the thesis.
3. Read both the LaTeX log and Mermaid CLI output; the real error may be a Mermaid parse error, Chromium/Puppeteer problem, shell-escape issue, or oversized float.
4. Do not adopt a static image fallback until logs have been read, external sources consulted if needed, and the user has confirmed the fallback.

## Environment Changes

- Treat TeX packages, fonts, Node packages, browser dependencies, and `tlmgr` installs as environment changes.
- Install only after confirming the missing dependency from logs or package tools.
- Ask the user before broad installs, fallback tools, or changes that affect the repository-wide build environment unless the task explicitly authorizes them.

## Documentation Requirements

Every LaTeX debugging task must record in the walkthrough:

- compile hypotheses and command attempts;
- logs/intermediate files inspected before retrying;
- sandbox versus official-build boundary;
- sandbox cleanup status and the user's cleanup confirmation;
- external sources consulted for unresolved errors;
- user decisions, especially fallback approvals or environment changes;
- what remains unvalidated and which later task owns it.

Reusable package behavior, build strategy, or toolchain findings belong in a knowledge artifact written as a guide, not in the plan solution block.
