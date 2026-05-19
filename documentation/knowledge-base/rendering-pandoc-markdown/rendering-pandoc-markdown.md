# Pandoc & Vector Diagrams Compilation Guide

This document tracks our implementation of `pandoc-ext/diagram` for rendering Mermaid (and other) diagrams natively to PDF via XeLaTeX in our Pandoc pipeline.

## Why `pandoc-ext/diagram`? (The AST Context)

In Pandoc's Abstract Syntax Tree (AST), `Para` represents a Paragraph. When external tools generate a diagram, they often emit an older formatting syntax: `Para[Image(...)]`. LaTeX interprets this as an *inline* image. Inline images do not trigger block-level figure environments (`\begin{figure}`), thus preventing the diagram from appearing in `\listoffigures` or supporting valid `\ref` cross-references.

Pandoc 3 introduced a native `Figure` block-level element precisely to solve this. The `pandoc-ext/diagram` extension is natively aware of Pandoc 3 and directly emits `Figure` nodes, guaranteeing standard LaTeX integration out of the box.

## Status: Migration & Learning

We are actively migrating to this pure Lua filter. Because we are exploring `pandoc-ext` in real-time, the bug tracking sections below will be updated dynamically as we encounter compilation quirks.

**Current Setup Actions (Task 1 Cleanup Complete):**

- Wiped all previous legacy Node.js-based filter proxies (e.g., `mermaid-filter`) to avoid conflicts.
- Removed local and global Lua scripts that were previously used to hack the AST.
- Verified that `@mermaid-js/mermaid-cli` (`mmdc`) is installed globally, which `pandoc-ext/diagram` requires to perform the actual SVG/PDF vector drawing under the hood.

## Installation

The `pandoc-ext/diagram` tool is implemented as a single Lua filter. We downloaded it directly from their GitHub repository into our `scripts` folder:

```bash
curl -L https://raw.githubusercontent.com/pandoc-ext/diagram/main/_extensions/diagram/diagram.lua -o documents_tcc/scripts/diagram.lua
```

## Compilation Pipeline

The target baseline compile command using this new setup is:

```bash
pandoc second_draft.md \
    --pdf-engine=xelatex \
    --lua-filter scripts/diagram.lua \
    --citeproc \
    --csl=associacao-brasileira-de-normas-tecnicas-numerico.csl \
    -o second_draft.pdf
```

## Known Bugs & Fixes

*(To be populated as we test `pandoc-ext/diagram`)*

## Tested Fixes and Observations

**1. Code Block Attribute Syntax mapping:**
When switching to `pandoc-ext/diagram`, the Mermaid block syntax had to change significantly.
- Legacy `mermaid-filter` recognized ` ```{.mermaid caption="..." #fig:id} `.
- `pandoc-ext/diagram` expects native Pandoc attributes: ` ```{.mermaid caption="..." #fig:id width="80%"} `.
- *CRITICAL FIX*: A previous bug caused `pandoc` to parse the ` ``` ` blocks as raw text instead of code blocks because of escaped quotation marks (`\"`). Removing the escapes resolved the AST parsing so the script emitted proper block-level `Figure` nodes.

**2. Figure Captions & Double Printing:**
Because `pandoc-ext/diagram` outputs native `Figure` nodes (which XeLaTeX natively wraps in `\begin{figure}` and applies the caption), the manual `Figura~\ref{...}: My caption` text placed below the original code blocks was causing visually duplicated labels (e.g., `Figura 1: ...` and `Figura 1: ...`). These duplicate hardcoded captions were permanently removed.

**3. Math Alignments:**
We refactored specific mathematical equations (like the 2-opt $\Delta C$ calculation) to use canonical LaTeX `\begin{align} ... \end{align}` blocks for professional typesetting, rather than relying on standard Markdown `$$ ... $$` math gaps.

## Tested Fixes and Observations

**1. Code Block Attribute Syntax mapping:**
When switching to `pandoc-ext/diagram`, the Mermaid block syntax mapped correctly.
- `pandoc-ext/diagram` expects native Pandoc attributes: ` ```{.mermaid caption="..." #fig:id width="80%"} `.
- *CRITICAL FIX*: A previous regex bug injected escaped quotes (`\\"`). Pandoc treats escaped quotes in code block headers as plaintext strings rather than metadata, so the block was interpreted as a `Str` AST node. Removing the invalid backslashes instantly brought our native `Figure` nodes to life.

**2. Figure Captions & Double Printing:**
Because `pandoc-ext/diagram` outputs native `Figure` nodes (which XeLaTeX natively wraps in `\\begin{figure}\\caption{}`), the manually duplicated captions placed below the diagrams (`Figura~\\ref{...}: My caption`) caused double labels (e.g. `Figura 1: ...` immediately followed by another `Figura 1: ...`). We deleted all hardcoded trailing captions from the source.

**3. Math Alignments:**
Specific equations (such as the 2-opt `\\Delta C` formula) were restructured using canonical LaTeX `\\begin{align} ... \\end{align}` blocks for robust justification, replacing unaligned raw Markdown blocks.
