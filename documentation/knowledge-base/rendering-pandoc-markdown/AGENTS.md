# Agent Instructions for documents_tcc

## Pandoc Compilation Command

When compiling the thesis draft (`second_draft.md`) to PDF, agents **MUST** follow this two-step process:

1. **Pre-render Mermaid blocks** into `.png` images.
2. **Compile with Pandoc**, ensuring ABNT CSL citations map correctly.

Mermaid-filter is inherently unstable with LaTeX image scaling when captions are involved. We instead use a python script containing regex to extract and render `.mmd` blocks down to transparent `.png` using `mmdc` (Mermaid CLI), and substitute standard markdown image tags in their place.

### The Pipeline

```bash
cd /home/lucas_galdino/chimera/gpu_accelerated/documents_tcc

# Step 1: Pre-process mermaid blocks into PNG files and image tags
python3 scripts/extract_mermaid.py

# Step 2: Compile to PDF using XeLaTeX
pandoc second_draft.md \
    --pdf-engine=xelatex \
    --citeproc \
    --csl=associacao-brasileira-de-normas-tecnicas-numerico.csl \
    -o second_draft.pdf
```

## Critical Rules

1. **CSL Application:** Always provide the `--csl` flag explicitly pointing to the local `associacao-brasileira-de-normas-tecnicas-numerico.csl` file.

2. **Engine:** `--pdf-engine=xelatex` is required to ensure math and unicode compilation stability.

3. **No `mermaid-filter`:** Do not use `mermaid-filter`, as it ruins the max height of pages when converting to XeLaTeX and ignores `width=...` formatting. Run the python `extract_mermaid.py` script instead.

4. **Verification:** Agents should not attempt to visually verify the PDF render. Instead, use the `askQuestions` tool to request that the human user evaluates the resulting PDF for layout errors, citation formatting, and diagram correctness.
