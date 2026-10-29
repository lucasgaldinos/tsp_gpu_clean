---
trigger: always_on
---

- Always update the plan APPROPRIATELY after doing changes. The tasks description and problems shouldn't be changed, but what was/is being/needs to be done is extremely important.

- It's really important that tasks are updated and other important artifacts, like walkthroughs, be generated.

- When adding these, do not change the problem descriptions and details, only add what was done. I want to know what the task asked and what was the solution.

- For the "Solution / O que foi feito" section, you MUST use the following standard structured list format:
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** Use relative markdown links with line anchors (relative to the plan file), e.g., `[filename](../../relative/path/to/file.ext#L<line_number>)`.
    - **Resumo:** A brief explanation of the changes made.
    - **Atendimento ao DoD:** How the Definition of Done was satisfied.

- ALWAYS read the relevant files entirely and correlate them with data in [tmp](../../../gpu_accelerated_clean/tmp/), [code/src](../../src/), [second_draft.tex](../../documents_tcc/tex/second_draft.tex), and other relevant directories before proposing any results, plans, or changes. Always perform a thorough technical analysis first.

- ALWAYS use the `ask-clarification-questions` skill whenever you are confused, an instruction is ambiguous, or a required format/standard is missing. Do not guess; stop and ask the user to clarify before taking irreversible actions.

- Remember to properly add paths as markdown links.

- use github callouts like `>[!{warning},{note},{tip},{caution},{important}]` when needed.
