---
title: "Guide: Pandoc and XeLaTeX Academic Compilation Pipeline"
description: >
  Comprehensive guide and agent instructions for compiling academic Pandoc Markdown
  into publication-ready PDFs via XeLaTeX, including pre-rendering workflows for Mermaid
  vector diagrams, ABNT CSL citation mapping, LaTeX preamble dissection, and AST integration.
created: 2026-11-05
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Consolidated rendering-pandoc-markdown/ (AGENTS.md, convertion-to-tex.md,
          mermaid-filter.md, rendering-pandoc-markdown.md) into a unified compilation
          guide and updated repository paths to gpu_accelerated_clean.
      - description: >
          Updated stale paths to real locations in documents_tcc/tex/ (second_draft.md,
          second_draft.tex, refs.bib, associacao-brasileira-de-normas-tecnicas-numerico.csl)
          and documents_tcc/scripts/ (diagram.lua, fix_captions.py). Clarified that the
          active thesis compiles natively with LuaLaTeX while Pandoc serves as the Markdown draft pipeline.
related_files:
  - "[guide-latex-math-notation.md](./guide-latex-math-notation.md)"
  - "[second_draft.md](../../documents_tcc/tex/second_draft.md)"
  - "[second_draft.tex](../../documents_tcc/tex/second_draft.tex)"
  - "[diagram.lua](../../documents_tcc/scripts/diagram.lua)"
  - "[fix_captions.py](../../documents_tcc/scripts/fix_captions.py)"
tags:
  - guide/pandoc
  - guide/latex
  - guide/compilation
  - guide/mermaid
  - notes/xelatex
  - notes/csl
  - notes/pdf
  - notes/ast
  - guide/academic_writing
  - notes/abnt
---

# Guide: Pandoc and XeLaTeX Academic Compilation Pipeline

This guide defines the compilation process, toolchain configuration, LaTeX preamble architecture, and vector diagram rendering pipeline for authoring academic manuscripts (such as [second_draft.md](../../documents_tcc/tex/second_draft.md)) using Pandoc and XeLaTeX.

> [!NOTE]
> The active thesis manuscript is maintained and compiled directly in native LuaLaTeX from [second_draft.tex](../../documents_tcc/tex/second_draft.tex) (`lualatex --shell-escape second_draft.tex`). Pandoc remains the compilation pipeline for the Markdown draft ([second_draft.md](../../documents_tcc/tex/second_draft.md)). Both manuscript source files, along with [refs.bib](../../documents_tcc/tex/refs.bib) and [associacao-brasileira-de-normas-tecnicas-numerico.csl](../../documents_tcc/tex/associacao-brasileira-de-normas-tecnicas-numerico.csl), reside in `documents_tcc/tex/` (NOT `documents_tcc/`).

---

## 1. Automated Compilation Pipeline

When compiling the academic Markdown draft ([second_draft.md](../../documents_tcc/tex/second_draft.md)) into a formatted PDF, automated agents and contributors follow this workflow:

1. **Pre-render diagrams or use the native Lua filter** (`[diagram.lua](../../documents_tcc/scripts/diagram.lua)`).
2. **Compile with Pandoc**, ensuring Brazilian ABNT CSL citations and Unicode typography map correctly.

> [!IMPORTANT]
> The legacy `mermaid-filter` is known to cause severe layout instability when generating LaTeX documents (ignoring `width=...` bounding and causing whole-page vertical overflow when captions are attached). The repository maintains `[diagram.lua](../../documents_tcc/scripts/diagram.lua)` in `documents_tcc/scripts/` to handle diagrams natively, while caption post-processing is supported via `[fix_captions.py](../../documents_tcc/scripts/fix_captions.py)`. Note that `scripts/extract_mermaid.py` does not exist on disk.

### Pipeline Execution Commands

```bash
cd /home/lucas_galdino/chimera/gpu_accelerated_clean/documents_tcc/tex

# Step 1: Compile Markdown draft to PDF using Pandoc and XeLaTeX
pandoc second_draft.md \
    --pdf-engine=xelatex \
    --lua-filter ../scripts/diagram.lua \
    --citeproc \
    --csl=associacao-brasileira-de-normas-tecnicas-numerico.csl \
    -o second_draft.pdf

# Optional Step 2: Fix figure caption positioning if post-processing LaTeX AST
python3 ../scripts/fix_captions.py

# Native LuaLaTeX Active Thesis Compilation (alternative / production target):
# cd /home/lucas_galdino/chimera/gpu_accelerated_clean/documents_tcc/tex
# lualatex --shell-escape second_draft.tex
```

### Critical Compilation Rules

1. **Explicit CSL Specification**: Always supply `--csl` explicitly pointing to [associacao-brasileira-de-normas-tecnicas-numerico.csl](../../documents_tcc/tex/associacao-brasileira-de-normas-tecnicas-numerico.csl) for Brazilian numerical ABNT compliance.
2. **XeLaTeX Engine**: `--pdf-engine=xelatex` is required for the Markdown draft to ensure font specification (`fontspec`), Unicode mathematical characters, and TrueType/OpenType font stability.
3. **No `mermaid-filter` in Production**: Do not pass `-F mermaid-filter` in the LaTeX output pipeline; use the pure Lua filter (`[diagram.lua](../../documents_tcc/scripts/diagram.lua)`).
4. **Verification Protocol**: Automated agents should not attempt to visually verify the compiled PDF render. Use interactive evaluation tools to request human review for final page breaks, citation numbering, and diagram scaling.

---

## 2. Dissecting the Pandoc-Generated LaTeX Preamble

When converting Markdown to standalone LaTeX (`pandoc -s file.md -o file.tex`), Pandoc injects an extensive generic preamble designed to ensure multi-engine support. Below is an architectural breakdown of each preamble component:

### 2.1 Initial Options and Section Numbering

```text
% Options for packages loaded elsewhere
\PassOptionsToPackage{unicode}{hyperref}
\PassOptionsToPackage{hyphens}{url}
\documentclass[]{article}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\setcounter{secnumdepth}{-\maxdimen} % remove section numbering
```

- **`amsmath,amssymb`**: Automatically loaded to support LaTeX math environments (`$$...$$`).
- **`\setcounter{secnumdepth}{-\maxdimen}`**: By default, Pandoc suppresses heading numbering unless `--number-sections` is explicitly passed on the command line.

### 2.2 Engine Conditionals and Unicode Typography

```text
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{textcomp} % provide euro and other symbols
\else % if luatex or xetex
  \usepackage{unicode-math} % this also loads fontspec
  \defaultfontfeatures{Scale=MatchLowercase}
  \defaultfontfeatures[\rmfamily]{Ligatures=TeX,Scale=1}
\fi
```

- **PDFTeX**: Legacy 8-bit engine requiring explicit `inputenc` and `fontenc` to map UTF-8 glyphs into font slots.
- **XeTeX / LuaTeX**: Modern native Unicode engines. They use `unicode-math` and `fontspec` to bind system TTF/OTF fonts directly.

### 2.3 Microtypography and Verbatim Quotes

```text
% Use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
\IfFileExists{microtype.sty}{%
  \usepackage[]{microtype}
  \UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
```

- **`upquote`**: Enforces straight single and double quotation marks inside verbatim/code blocks (preventing copy-paste syntax errors caused by curved smart quotes).
- **`microtype`**: Fine-tunes kerning, margin protrusion, and font expansion, eliminating ragged white gaps in justified paragraphs.

### 2.4 Syntax Highlighting Macros

```text
\usepackage{fancyvrb}
\newcommand{\VerbBar}{|}
\newcommand{\VERB}{\Verb[commandchars=\\\{\}]}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}}
\newcommand{\AlertTok}[1]{\textcolor[rgb]{1.00,0.00,0.00}{\textbf{#1}}}
\newcommand{\AnnotationTok}[1]{\textcolor[rgb]{0.38,0.63,0.69}{\textbf{\textit{#1}}}}
\newcommand{\KeywordTok}[1]{\textcolor[rgb]{0.00,0.44,0.13}{\textbf{#1}}}
\newcommand{\StringTok}[1]{\textcolor[rgb]{0.25,0.44,0.63}{#1}}
```

- Pandoc’s built-in Skylighting engine converts fenced code blocks into tokenized macros (`\KeywordTok`, `\StringTok`, `\CommentTok`) mapped to precise RGB color definitions.

### 2.5 Tables and Bounded Images

- **`longtable` and `booktabs`**: Allow multipage tables with publication-grade horizontal rules and page breaks.
- **`footnotehyper`**: Enables footnotes within table header cells.
- **`\pandocbounded{...}`**: Dynamically evaluates image geometry; if an image exceeds `\linewidth`, it scales the graphic proportionally to prevent margins from clipping.

### 2.6 Citation Engine Comparison: `--citeproc` vs. `--biblatex`

1. **`--citeproc` (Internal Filter)**:
   - Pandoc parses [refs.bib](../../documents_tcc/tex/refs.bib) internally and replaces citation keys (`[@cook2012]`) with static formatted text strings directly in the generated LaTeX source.
   - The underlying LaTeX compiler does not need `biblatex` or Biber.
2. **`--biblatex` + `--bibliography` (LaTeX Engine Delegate)**:
   - Pandoc emits `\usepackage{biblatex}` and `\addbibresource{refs.bib}`, replacing citation keys with `\autocite{key}` and printing the bibliography via `\printbibliography`.
   - Requires running `biber` as an auxiliary compilation stage.

### 2.7 Frontmatter Metadata Mapping

Metadata declared in Markdown YAML frontmatter is injected into the LaTeX `\hypersetup{...}` block (populating PDF metadata fields `pdftitle`, `pdfauthor`) and into standard document title macros (`\title{...}`, `\author{...}`).

---

## 3. Mermaid & Vector Diagram Rendering: AST Context

### Why AST Architecture Matters: `Para` vs. `Figure`

In Pandoc's Abstract Syntax Tree (AST), a `Para` represents a standard text paragraph. Legacy tools often converted diagrams into inline images: `Para[Image(...)]`. In LaTeX, inline images do not generate floating figure environments (`\begin{figure}`), which breaks:
- Inclusion in the List of Figures (`\listoffigures`).
- Proper caption placement.
- Cross-referencing via `\ref{fig:...}`.

Pandoc 3 introduced native `Figure` block-level AST elements. Modern Lua filters (such as `pandoc-ext/diagram`) emit native `Figure` AST nodes directly, guaranteeing canonical LaTeX `\begin{figure}` wrappers.

### Lua Filter Workflow (`diagram.lua`)

```bash
# In documents_tcc/tex:
cd /home/lucas_galdino/chimera/gpu_accelerated_clean/documents_tcc/tex

# Compilation with Lua filter and ABNT CSL
pandoc second_draft.md \
    --pdf-engine=xelatex \
    --lua-filter ../scripts/diagram.lua \
    --citeproc \
    --csl=associacao-brasileira-de-normas-tecnicas-numerico.csl \
    -o second_draft.pdf
```

### Legacy `mermaid-filter` Configuration Attributes

When inspecting or maintaining legacy documents, fenced code block attributes were configured using curly-brace syntax:

```text
[mermaid attribute syntax: loc=img caption="Architecture Diagram" #fig:arch width=800]
sequenceDiagram
    Alice->>Bob: Request Tour
    Bob-->>Alice: Return 2-opt Tour
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `caption` | Empty | Sets caption and `alt` text attribute |
| `format` | `png` | Image format (`png`, `svg`) |
| `width` | `800` | Target render width |
| `theme` | `default` | Mermaid theme (`default`, `dark`, `neutral`) |
| `loc` | `inline` | Image location (`inline`, directory path like `img`) |
| `scale` | `1` | Scaling factor |

---

## 4. Tested Troubleshooting & Fixes

### 1. Escaped Quotes in Code Block Attributes
- **Symptom**: Code blocks rendered as raw Markdown text rather than generating figures.
- **Root Cause**: Escaped quotes (e.g., `\"80%\"`) cause Pandoc's reader to classify the header as a plaintext string (`Str`) instead of attribute metadata.
- **Fix**: Use clean unescaped quotes in the attribute header: `caption="Diagram Title"` and `width="80%"`.

### 2. Duplicate Figure Captions
- **Symptom**: Double caption labels appeared in the PDF (e.g., `Figura 1: ...` immediately followed by another `Figura 1: ...`).
- **Root Cause**: The filter automatically wraps the output in `\begin{figure}\caption{...}`, while the Markdown source also contained a manual trailing paragraph `Figura~\ref{fig:myfig}: Description`.
- **Fix**: Remove all hardcoded trailing figure labels from the Markdown source; rely exclusively on the code block `caption="..."` attribute.

### 3. LaTeX Math Alignment
- **Symptom**: Formula misalignment in complex expressions like 2-opt $\Delta C$ calculations.
- **Fix**: Replace raw Markdown `$$` block math with canonical LaTeX alignment environments:
  ```text
  \begin{align}
      \Delta C &= d(t_i, t_{j+1}) + d(t_{i+1}, t_j) - d(t_i, t_{i+1}) - d(t_j, t_{j+1})
  \end{align}
  ```
