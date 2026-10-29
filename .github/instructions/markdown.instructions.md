---
name: "Markdown Standards"
description: >
  Frontmatter template, code block language tags, and filename conventions
  for all .md files in this repository. Apply when creating or editing any
  markdown file, or when asked about naming, metadata, or structure of .md files.
applyTo: "**/*.md"
---

<check>

# repository markdown standards

When **updating** a file **without** that does not conform with below instructions inside this repository, confirm with `vscode_askQuestions` tool before doing changes.

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
  - [filename.md](./relative/path/to/filename.md)
tags: [] # 10–15 tags; hierarchical Obsidian-style, e.g. benchmark/results
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

Make usage of markdown internal links and external links, e.g. for referencing other files, use the \[file-name/file-info\]\(/file/path#gfm-header\) or \[file-name/file-info\]\(/file/path#l\<number\>\) and for internal links \[paragraph/section\]\(#gfm-header\). Repo current uses backticks to reference files, which makes no sense without the link to the file itself.

</referencing_rules>
