---
name: explore
description: >
  Codebase exploration and targeted research. Use when the reviewer (or any agent) must understand
  repo structure, locate symbols and reusable patterns, read docs/PDFs, or gather external sources
  (papers, articles, API references) to ground a review or plan. Trigger on "explore the codebase",
  "find where X is defined", "how do these modules interact", "research the literature on Y",
  "gather sources". Loads on demand; reports findings, not narration.
argument-hint: "[what to find] [thoroughness: quick | medium | thorough]"
---

# Explore

<context>

Exploration has two modes, both feeding the review/planning pipeline: (1) **codebase exploration** — locate symbols, patterns, structure, dependencies; (2) **targeted research** — gather external sources to ground a decision. Both bias toward speed and concision: exploration is a means, never the deliverable.

</context>

## Codebase Exploration

<instructions>

Orient first with the `eza` rules from `exploration.instructions.md`: a broad tree at `-L 3`, then a flat listing of any targeted folder. Prefix terminal commands with `rtk`.

Go broad to narrow:

1. Discover candidate areas with glob patterns or semantic code search.
2. Narrow with text search (regex) or symbol usages (LSP).
3. Read files only when you know the path or need full context.

Respect the repo's own instructions, rules, and skills as you traverse — they explain architecture and conventions. Use the GitHub repo tool to trace external dependencies.

</instructions>

<rules>

- Parallelize independent tool calls (multiple greps, multiple reads at once).
- Stop once you have sufficient context — targeted searches, not exhaustive sweeps.
- Adapt depth to the requested thoroughness (quick / medium / thorough).
- Prefix shell commands with `rtk` for token savings.

</rules>

<output_format>

Report findings directly, including:

- Files as absolute links.
- Specific functions, types, or patterns that can be reused.
- Analogous existing features that serve as implementation templates.
- A short synthesized summary of what was found.
- Direct answers to what was asked — not a comprehensive tour.

</output_format>

## Repository Analysis

When the goal is understanding the repo as a whole (structure, functionality, dependencies, code quality):

- Read documentation, PDFs, and data files in addition to source.
- Map components and their interactions; identify the main dependencies.
- Evaluate code quality and surface reusable patterns.

## Targeted Research

When external grounding is needed (papers, articles, API references):

- Infer the underlying intent of the query — go beyond keyword matching.
- Prefer scientific sources (arXiv, Google Scholar) for technical/scientific themes; regular web for general topics.
- Return at least 10 sources; reference where each finding came from.
- Provide a concise summary highlighting the most relevant insights and their sources.

## Report Discipline

Return findings and their sources with a concise synthesized summary. No verbose narration. If thoroughness isn't specified, default to quick and say so.
