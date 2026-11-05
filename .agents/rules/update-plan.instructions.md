---
# --- Antigravity rule keys ---
trigger: always_on
description: "Use whenever updating plan files, TODOs, symbol registries, walkthroughs, or Solution blocks." # shared with VS Code
# --- VS Code instruction keys ---
name: "Plan & Task Update Standards"
applyTo: "**"
---

- Always update the plan APPROPRIATELY after doing changes. The tasks description and problems shouldn't be changed, but what was/is being/needs to be done is extremely important.

- It's really important that tasks are updated and other important artifacts, like walkthroughs, be generated.

- When adding these, do not change the problem descriptions and details, only add what was done. I want to know what the task asked and what was the solution.

- For the "Solution / O que foi feito" section, you MUST use the following standard structured list format:
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** Use relative markdown links with line anchors (relative to the plan file), e.g., `[filename](../../relative/path/to/file.ext#L<line_number>)`.
    - **Resumo:** A brief explanation of the changes made.
    - **Atendimento ao DoD:** Concrete, falsifiable evidence showing how the Definition of Done was satisfied. Any `grep`/`rg` DoD check MUST exclude cache directories (`--exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}`) so `.pyc` or cache blobs never produce false matches, and the DoD claim must never contradict exceptions noted in `Resumo`.

- **Never leave incomplete or defective tasks marked `[x]`:** No task may be marked `[x]` without its `Solution` block, and whenever an audit or test run finds missed files, partial fixes, or regressions on a `[x]` task, immediately unmark it back to `[ ]` in place and document what remains.

- **Separate plans by repository & enforce the symbol gate:** Keep `gpu_accelerated_clean` tasks exclusively in [plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md) and register every `DP-*`, `DP<phase>-*`, and `BUG-*` identifier in [table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md). Keep `routing_data` tasks exclusively in `routing_data/TODO.md` (and `routing_data/src/converter/TODO.md`). After editing any plan or walkthrough in `gpu_accelerated_clean`, run `rtk .venv/bin/python -m benchmark_scripts.check_docs` and require exit code `0`.

- ALWAYS correlate target files (using bounded reads per [AGENTS.md](../../AGENTS.md)) with [src](../../src/), [documents_tcc/plan](../../documents_tcc/plan/), [documentation/knowledge-base](../../documentation/knowledge-base/), and [second_draft.tex](../../documents_tcc/tex/second_draft.tex) before proposing plans or changes. Always perform a thorough technical analysis first.

- ALWAYS use the `ask-clarification-questions` skill whenever you are confused, an instruction is ambiguous, or a required format/standard is missing. Do not guess; stop and ask the user to clarify before taking irreversible actions.

- Remember to properly add paths as markdown links (repo-relative in committed `.md` files, never `file:///home/...`).

- Use GitHub callouts (`> [!WARNING]`, `> [!NOTE]`, `> [!TIP]`, `> [!CAUTION]`, `> [!IMPORTANT]`, with a space after `>`) when needed.
