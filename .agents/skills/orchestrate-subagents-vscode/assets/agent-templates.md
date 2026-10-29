# Copy-Paste Agent Templates

Load this file when a new orchestrator or worker `.agent.md` file is needed. Fill every `[FILL: …]` placeholder. This repo's fleet is **RUG** (orchestrator) + **reviewer** + **implementer**; the templates mirror that model. For field semantics, see [agent-frontmatter.md](../references/agent-frontmatter.md).

> [!WARNING]
> A subagent without an explicit `tools` field **inherits the parent agent's tools** — and RUG's tools are read-only. A worker that omits `tools` cannot edit files or run commands. Every worker MUST declare its own `tools`.

## 1. Orchestrator (RUG)

```markdown
---
name: RUG
description: >
  Pure orchestration agent. Decomposes a request, delegates all planning, review, and execution to
  the reviewer and implementer workers, tracks progress, and synthesizes results. Loads the
  orchestrate-subagents-vscode skill. Reads files to orient but never writes code or edits files.
tools:
  [
    "read",
    "search",
    "agent",
    "todo",
  ]
agents:
  [
    "reviewer",
    "implementer",
  ]
---
[FILL: coordination instructions — load the orchestrate-subagents-vscode skill and follow its
pipeline. Do not write code or edit files, including the plan/TODO.]
```

Rules:

- The orchestrator's `tools` never include `edit` or `execute` — it does not write code or files. Task registration is the reviewer's job.
- `agents` names the real workers (`reviewer`, `implementer`). Never `SWE`/`QA` placeholders.

## 2. Reviewer (review + plan + register)

```markdown
---
name: reviewer
description: >
  Review + planning + task-registration worker. Loads code-review-and-quality and task-managing;
  reviews a reading list (read-only on source), plans the work, and registers tasks in the plan
  (edit scoped to the plan file only). Runs at the start and end gates.
user-invocable: false
tools: [vscode/memory, vscode/resolveMemoryFileUri, execute/getTerminalOutput, execute/killTerminal, execute/sendToTerminal, execute/runInTerminal, read, edit/createDirectory, edit/createFile, edit/editFiles, edit/rename, search, 'sequentialthinking/*', 'actor-critic-thinking/*', vscodeGeneral/rename, todo]
---
[FILL: load code-review-and-quality + task-managing + explore. Review, then plan, then register
each task in the 5W2H format (What / Why / Who / Where / When / How / How much / DoD) with a
measurable DoD and a file target. `edit` is scoped to the plan file only — source is read-only.]
```

Why `edit` is present but scoped: VS Code has no per-file `edit` scope, so the reviewer's frontmatter grants `edit` but its body restricts it to the plan file. Source stays read-only.

## 3. Implementer (consume + update)

```markdown
---
name: implementer
description: >
  Execution worker. Consumes well-specified, batched tasks (<=5 per invocation, one shared context)
  and loads the solving-tasks workflow plus task-managing to execute and update task state and
  artifacts. Does not plan or write tasks.
user-invocable: false
tools: [vscode/memory, vscode/resolveMemoryFileUri, execute/runNotebookCell, execute/executionSubagent, execute/getTerminalOutput, execute/killTerminal, execute/sendToTerminal, execute/runInTerminal, read, edit, search, vscodeNotebooks/runNotebookCell]
model:
  [
    "DeepSeek V4 Flash (deepseek)",
    "DeepSeek V4 Pro (deepseek)",
  ]
---
[FILL: load solving-tasks + task-managing. Execute the batched tasks; update task state and
artifacts. Do not plan or write new tasks.]
```

## Sanity Checklist for Every Agent File

- [ ] Orchestrator has `agent` + `todo` (+ `read`/`search` to orient); never `edit`/`execute`.
- [ ] Reviewer has `read` + `search` + `edit` (edit scoped to the plan by instruction).
- [ ] Implementer may `edit`/`execute` (it writes source and runs commands) but never registers tasks.
- [ ] Workers are `user-invocable: false`.
- [ ] `agents` allowlist names real workers, not placeholders.
- [ ] Model names match the picker exactly (`(deepseek)` lowercase).
- [ ] No `infer` field — deprecated.
