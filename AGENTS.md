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
related_files:
  - [copilot-instructions.md](./.github/copilot-instructions.md)
  - [markdown.instructions.md](./.agents/rules/markdown.instructions.md)
  - [exploration.instructions.md](./.agents/rules/exploration.instructions.md)
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
   approval to batch every item.
2. **Bound the reads.** `grep_search` before `read_file`; read line ranges, not whole
   files; no unprompted whole-file read of files over ~400 lines, binaries, PDFs, or
   spilled `chat-session-resources/*.txt` tool output. Every read serves the current item.
3. **Announce skills.** The first line of a task names the skills loaded (or
   `skills: none`) and the step each one covers. If no skill matches, say so instead of
   proceeding silently.
4. **Executable anchors.** An explanation is incomplete without a runnable snippet, the
   exact `path:line`, and the expected output. A concept explained a third time becomes a
   knowledge-base note, not another chat answer.
5. **Verification that can fail.** A fix ships with the observation that fails without it.
   "Tests pass" is not evidence when the test passed before the change.
6. **Session handoff.** At the end of a work item, write the `## Handoff` block into the
   active plan file: state, last verified observation, next item, open decisions, files to
   read first. Bridge sessions with the handoff, never with the transcript.
7. **File edits use built-in tools.** Create and edit files with `create_file`,
   `replace_string_in_file`, or `multi_replace_string_in_file` so every change is
   reviewable as a diff. The terminal is for exploration and commands, never for writing
   file contents.

## Repository rules that still apply

- Markdown files follow [markdown.instructions.md](./.agents/rules/markdown.instructions.md)
  (frontmatter, language tags, filename casing).
- Workspace exploration follows
  [exploration.instructions.md](./.agents/rules/exploration.instructions.md).
- Skills live in `.agents/skills/`; agent definitions in `.agents/agents/`.
