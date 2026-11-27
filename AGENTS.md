---
title: "AGENTS.md — agent and contributor guidelines"
description: >
  Mandatory working agreement for AI coding agents and human contributors in
  `gpu_accelerated_clean`: plan gate, read budget, skill announcement, executable
  explanations, verification standard, session handoff, and reviewable file edits.
created: 2026-10-27
modifications:
  - date_modified: 2026-10-27
    modifications:
      - description: >
          Created the repository-root working agreement with seven rules derived from
          observed friction in chat-session history.
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Added anti-stubbing/pacing, coordinator non-duplication and 4.5 KB subagent output
          cap, anti-caveman domain explanations with KaTeX/citations, compaction preservation,
          in-place editing, task unmarking on defect discovery, repo plan isolation, and zero legacy aliases.
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Expanded working-agreement rule 6 from a one-line pointer into a concrete
          `## Handoff` block template with write triggers and the evidence requirement
          for a `done` state.
  - date_modified: 2026-11-14
    modifications:
      - description: >
          Added rule 11 (stable `path::symbol` references instead of drifting `#L` line
          anchors) and rule 12 (English for agent-facing files, pt-BR for `documents_tcc/`
          and walkthroughs), pointing at the new prose-language reference and the
          `check_docs` symbol-anchor gate.
  - date_modified: 2026-11-17
    modifications:
      - description: >
          Added rule 13 (decoupling heavy empirical benchmarks from reflexive subagent
          review loops), preventing multi-layer agent review from wrapping long physical
          compute runs.
related_files:
  - [markdown.instructions.md](./.agents/rules/markdown.instructions.md)
  - [exploration.instructions.md](./.agents/rules/exploration.instructions.md)
  - [prose-language.instructions.md](./.agents/rules/prose-language.instructions.md)
tags:
  - guide/agents
  - guide/instructions
  - notes/workflow
  - notes/session
  - notes/context
  - analysis/planning
  - review/quality
  - guide/testing
  - notes/handoff
---

# AGENTS.md — agent and contributor guidelines

This repository hosts the GPU-accelerated routing solver work: `src/`, `tests/`,
`documentation/`, `results*/`, and the `.agents/` skill and rule set.

## Working agreement

Applies to every task in this repository.

1. **Plan before writes.** A change touching more than one file starts with a numbered
   plan: what, why in business terms, the observation that fails today, and the blast
   radius. No file is written until the plan is approved. Approval of the plan is not
   approval to batch every item: never pre-create empty placeholder/stub files, keep
   interactive edits reviewable (~300 lines or ≤3 new documentation files per turn unless
   executing an approved subagent batch), and never run `git commit` or `git push` without
   explicit user approval.
2. **Bound the reads.** `grep_search` before `read_file`; read line ranges, not whole
   files; no unprompted whole-file read of files over ~400 lines, binaries, PDFs, or
   spilled `chat-session-resources/*.txt` tool output. Every read serves the current item.
   When delegating exploration or review to a subagent, do **not** read, list, or grep the
   delegated paths yourself first, and cap subagent final reports at `< 4,500` characters
   so VS Code returns them inline instead of spilling to `chat-session-resources/*.txt`.
3. **Announce skills.** The first line of a task names the skills loaded (or
   `skills: none`) and the step each one covers. If no skill matches, say so instead of
   proceeding silently.
4. **Executable anchors.** An explanation is incomplete without a runnable snippet, the
   exact `path::symbol` reference, and the expected output. A concept explained a third
   time becomes a knowledge-base note, not another chat answer.
   Never use terse "caveman" phrasing when explaining python internals, GPU hardware models,
   mathematical trade-offs, or code reviews: pair an intuitive mental model ("why" in domain
   terms, not restating syntax) with KaTeX equations and primary academic/spec citations
   (e.g., Reinelt TSPLIB95, Rocki & Suda, O'Neil & Burtscher) where applicable.
5. **Verification that can fail.** A fix ships with the observation that fails without it.
   "Tests pass" is not evidence when the test passed before the change.
6. **Session handoff.** At the end of every work item, write the `## Handoff` block below
   into the active plan file. Bridge sessions with the handoff, never with the transcript.

   ```markdown
   ## Handoff

   - **State**: done | in progress | blocked
   - **Last verified observation**: `<command>` -> `<output line>` (YYYY-MM-DD)
   - **Next item**: task id + one-line action
   - **Open decisions**: question | options | owner
   - **Read first**: `path::symbol`, `path#heading` (max 5 entries)
   ```

   Write it at every boundary where the next session would otherwise start by guessing:
   end of a task, end of a batch, before a compaction, before delegating to a subagent.
   `State: done` is only legal when the last verified observation is an observation that
   fails without the change.
   At the end of each plan phase, propose atomic git commits so dirty paths do not
   accumulate across phases. When compacting (`/compact`), explicitly preserve active
   **negative constraints** (e.g., `C1`–`C3`, zero legacy aliases, repo boundaries), locked
   `DP-*`/`BUG-*` contracts, why any reopened task failed, and links to `knowledge-base/`
   files instead of re-inlining long derivations.
7. **File edits use built-in tools.** Create and edit files with `create_file`,
   `replace_string_in_file`, or `multi_replace_string_in_file` so every change is
   reviewable as a diff. The terminal is for exploration and commands, never for writing
   file contents. Edit plans, specs, and `TODO.md` files **in place** — never append
   errata or corrections at EOF while leaving contradicted text intact above.
8. **Unmark incomplete tasks immediately.** Whenever an audit, review gate, or test run
   reveals that a `[x]` task has residual bugs, partial scope, or missed callers, flip
   its checkbox back to `[ ]` in the plan immediately with a note on what remains — never
   report defects solely in chat while leaving the plan marked done.
9. **Strict repository plan isolation.** Keep tasks in their owning repository's plan:
   `gpu_accelerated_clean` tasks belong exclusively in `documents_tcc/plan/*.plan.md` (with
   every `DP-*`/`BUG-*` registered in `table-of-symbols.plan.md`); `routing_data` tasks
   belong exclusively in `routing_data/TODO.md` (and `src/converter/TODO.md`).
10. **Zero legacy aliases.** In these single-developer repositories, never leave
    backward-compatibility aliases or "one-release-cycle" deprecation shims when renaming
    classes or functions (e.g., `DataTransformer = ProblemNormalizer`). Delete the old
    symbol and update all callers in `src/`, `tests/`, and `docs/` in the same change.
11. **Stable references.** Anchor to names, not lines: `[two_opt.py::TwoOpt._compute_gain](path)`
    for code, `[file.md §Heading](path#heading)` for Markdown, `[file.tex §label](path#label)`
    for LaTeX. The `::` separator is the pytest node-ID idiom and works for `.py`, `.cu`,
    and `.cuh`; it is validated by `benchmark_scripts.check_docs` (use `check_docs` for
    Markdown, plans, walkthroughs, and python/CUDA docstrings; use `benchmark_scripts.check_thesis`
    for LaTeX `.tex` and BibTeX `refs.bib` integrity under `documents_tcc/tex/thesis/`).
    `#L<line>` is allowed only for targets with no nameable unit (logs, generated tables,
    frozen audit snapshots). Line numbers drift with every refactor; symbols do not.
12. **Prose language.** English is the default for agent-facing files (`AGENTS.md`,
    `.agents/**`, `.github/**`) and for `documentation/knowledge-base/**`. Brazilian
    Portuguese is mandatory for `documents_tcc/**` and `documentation/walkthroughs/**`.
    Translation follows [prose-language.instructions.md](./.agents/rules/prose-language.instructions.md):
    pt-BR register, no calqued headings, technical terms kept verbatim.
13. **Decouple physical hardware benchmarks from reflexive subagent review loops.**
    Distinguish software engineering and analytical tasks (refactoring, typing, statistical tests,
    TeX writing) from long-running empirical hardware benchmarks (>5 min of GPU/CPU compute):
    - Code, stats, and TeX tasks use `/boost` or subagents for deep reflection, verification, and regression prevention.
    - Tasks that perform large-scale empirical collection (re-running checkpoint suites, parameter sweeps)
      must be executed directly or in an isolated stage with explicit progress logging, rather than wrapping
      physical hardware time inside multi-pass recursive subagent review loops (avoiding redundant post-run
      audits and 40+ minute opaque wait times).
14. **Wave-boundary quality review & phased human checkpoints.** Automated checks
    (`pytest`, `mypy`, `ruff`, `check_docs`) are necessary but never sufficient to mark
    a task `[x]` or merge a subagent wave. Before flipping any task checkbox to `[x]`,
    audit the wave diff against `code-review-and-quality` (including
    `references/gpu-and-scientific-computing-review.md` for CUDA, RNG, `warmup()`,
    baseline fairness, and statistical code). On large multi-wave changes ($>2$ waves, $>1,500$ LOC cumulative, or cross-repo work), pause between major phases for user
    review instead of executing every wave end-to-end unreviewed.
15. **Escalate unasked design decisions & protect terminology.** Never silently invent
    magic defaults (e.g. arbitrary penalty weights), undocumented fallback semantics,
    divergent CPU/GPU class naming, or jargon that collides with existing repository/thesis
    terms (e.g. reusing `"Tier-1"` when `Tier 1 / Tier 2` already denotes statistical
    testing tiers). When an unasked design choice surfaces during planning *or* inside a
    subagent wave, pause and resolve it with the user via `ask-clarification-questions`.

## Repository rules that still apply

- Markdown files follow [markdown.instructions.md](./.agents/rules/markdown.instructions.md)
  (frontmatter, language tags, filename casing).
- Workspace exploration follows
  [exploration.instructions.md](./.agents/rules/exploration.instructions.md).
- Portuguese prose follows
  [prose-language.instructions.md](./.agents/rules/prose-language.instructions.md).
- Skills live in `.agents/skills/`; agent definitions in `.agents/agents/`.
