---
# --- Antigravity rule keys ---
trigger: glob
globs: "**/*.md, !**/.agents/**, !**/.github/**"
description: "Use when creating or editing any Markdown file: required frontmatter fields, code-block language tags, and filename casing rules." # shared with VS Code
# --- VS Code instruction keys ---
name: "Markdown Standards"
applyTo: "**/*.md, !**/.agents/**, !**/.github/**"
---

<check>

# repository markdown standards

When **updating** an existing file inside this repository that does not conform with the instructions below, confirm with `vscode_askQuestions` (or `ask_question`) before changing its structure.

</check>

<frontmatter_rules>

## Frontmatter

Every new `.md` file must open with a YAML frontmatter block.

### Mandatory fields

```yaml
---
title: <concise title>
description: >
  <one-paragraph description of the file's purpose>
created: YYYY-MM-DD
modifications:
  - date_modified: YYYY-MM-DD
    modifications:
      - description: >
          <summary of changes made in this session>
related_files:
  - "[filename.md](./relative/path/to/filename.md)"
  - "[filename2.md](./relative/path/to/filename2.md)" # It's important that the links are inside double quotes and in a list.
---
```

- **title** — plain text; use backticks only when referencing a filename (e.g. `` `second_draft.md` ``).
- **description** — YAML block scalar (`>`); purpose of the file, not an outline.
- **created** — ISO 8601: `YYYY-MM-DD`.
- **modifications** — append one entry per editing session; never pre-fill empty placeholders.
  + `date_modified`: ISO 8601.
  + `modifications`: list of `description` entries using `>` block scalar.
- **related_files** — Markdown links to workspace files; add an empty list `[]` if none.
- **tags** — 10–15 tags. Reuse existing vault tags when possible. Hierarchical format:
  `<namespace>/<subtag>[/<subtag>]`. Approved namespaces: `benchmark/`, `algorithm/`, `analysis/`, `notes/`, `guide/`, `review/`.

### On-demand fields

Add only when relevant. Insert after `created` and before `modifications`.

| Field | Values | When to add |
| ----- | ------ | ----------- |
| `status` | `draft` \| `review` \| `final` | Document has a publication lifecycle |
| `author` | YAML list | Authorship is meaningful |
| `type` | `analysis` \| `guide` \| `notes` \| `draft` \| `report` | Document type aids navigation |
| `scope` | `local` \| `global` | File targets a specific module (`local`) or whole project (`global`) |

Default `author` block:

```yaml
author:
  - "[[Lucas Galdino]]"
  - "[[<LLM-model-name>]]"   # e.g. [[Claude Sonnet 4.6]]
```

</frontmatter_rules>

<code_block_rules>

## Code Blocks

Every fenced code block must have a language identifier. Never leave the fence tag blank.

| Content | Tag |
| ------- | --- |
| python | `python` |
| YAML / frontmatter | `yaml` |
| JSON | `json` |
| shell / terminal | `bash` |
| CUDA / C++ | `cpp` |
| Markdown source | `markdown` |
| plain output / logs | `text` |
| SQL | `sql` |

- Use `text` for plain output or prose that needs monospace but has no language.
- Use `markdown` (not `md`) for Markdown source shown inside a code block.
- Do not invent tags; use the identifiers above or common aliases (`py`, `sh`, `js`, `ts`).

</code_block_rules>

<file_naming_rules>

## File Naming

Rules apply to the **filename only**. The frontmatter `title` may be human-readable prose.

### Casing by type prefix

| Type prefix | Case | Example |
| ----------- | ---- | ------- |
| `analysis-` | kebab-case | `analysis-benchmark-speedup-results.md` |
| `guide-` | kebab-case | `guide-algorithm-composition.md` |
| `review-` | kebab-case | `review-second-draft-chapter4.md` |
| `report-` | kebab-case | `report-statistical-summary.md` |
| `notes_` | snake_case | `notes_gpu_timing_session.md` |

- Format: `<type-prefix><title-slug>.md`.
  + Doc types (`analysis`, `guide`, `review`, `report`): hyphens → kebab-case.
  + Notes: underscores → snake_case.
- Max 50 characters excluding `.md`.
- No spaces or special characters; only `-` or `_` as separators.
- No abbreviations — use full words.
- If document type is ambiguous, default to `notes_` (snake_case).
- The frontmatter `title` is independent (e.g. `title: "Review for the second draft document"`).

</file_naming_rules>

<referencing_rules>

## Referencing

Use repo-relative Markdown links (never workstation-absolute `file:///home/...` URIs inside committed docs, and never bare backticked filenames without a link):

- Cross-file section links: `[file.md §Heading](./relative/path/to/file.md#gfm-header)` — contextualize *which section/paragraph* explains the concept, and keep bidirectional `Links out` / `Backlinks` graphs symmetric when splitting multi-part guides.
- GFM heading anchor slugs: lowercase, spaces → `-`, strip `.` and backticks, preserve `_`; validate `MD051`/`MD052` after heading edits.
- Line-drift resilience: in narrative/conceptual prose, link symbols, classes, functions, or section headers rather than hardcoding fragile line numbers in prose text; reserve `#L<line>` anchors for plan `Where`/`Solution` entries and audit citations verified via `grep_search`.

</referencing_rules>

<authoring_hygiene>

## Authoring Hygiene, Math & Diagrams

- **Edit in place, never append errata:** Update specifications, plans, and `TODO.md` where the relevant section lives. Never append corrections at EOF while leaving contradicted text intact above, and rewrite decision reports to record the chosen architecture + rationale rather than preserving obsolete rejected options as open noise.
- **No placeholder stubs:** Never pre-create empty or `"Placeholder"` `.md` files across turns; author each document completely and atomically (≤3 new docs per interactive turn unless batched in an approved plan).
- **Self-contained prose:** Committed documentation must never leak chat-session artifacts (e.g., *"as discussed in the transcript..."* or *"in the previous turn..."*), must verify every cited symbol exists in code, and should use prose + GitHub callouts (`> [!NOTE]`, `> [!IMPORTANT]`, with a space after `>`) rather than turning every paragraph into a table.
- **KaTeX & citations:** Write multi-term memory/complexity/cost formulas in KaTeX (`$...$` / `$$...$$`) and cite primary sources (e.g., Reinelt 1991; Rocki & Suda 2012; O'Neil & Burtscher 2015).
- **Mermaid diagrams:** Use palettes legible in both light and dark themes, give nodes descriptive human-readable labels (never bare 2-letter abbreviations), ensure YAML frontmatter closes with `---` before an unindented diagram keyword (`flowchart TD`), and validate modified diagrams via `rtk npx -y @mermaid-js/mermaid-cli` on `/tmp/` extracts when Node/CLI is available.

</authoring_hygiene>
