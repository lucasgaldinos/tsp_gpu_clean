---
name: task-managing
description: >
  The 5W2H task format, plan structure, task states, and artifact decisions for the review→plan→
  execute pipeline. Use when decomposing deficiencies into tasks, registering findings in a plan,
  or marking task states. Loaded by the reviewer (plan + register) and the implementer (execute +
  update state).
---

# Task Managing

## Overview

Task-managing is the shared vocabulary that turns review findings into tracked work. It defines the plan directory layout, the 5W2H task format (every task carries What / Why / Who / Where / When / How / How much + a measurable DoD), the Solution block the implementer fills in, and the task states the reviewer flips. The reviewer loads it to plan and register; the implementer loads it to execute and update state.

A task without a DoD and a file target is underspecified — never write prose bullets in place of a task.

## 1. Directory Structure

Every plan lives in a `plan/` directory:

```text
plan/
├── plan-<name>.prompt.md     # the registered plan (index + tasks)
├── task-resolutions/          # walkthrough artifacts (one per task)
│   └── walkthrough_<task-id>.md
└── knowledge-base/            # reusable knowledge artifacts
    └── knowledge_<topic-slug>.md
```

- **plan file** — the single index: decisions pinned, batches, and every task in 5W2H.
- **task-resolutions/** — walkthroughs preserving the execution history of each task.
- **knowledge-base/** — topic-scoped, reusable references extracted from walkthroughs.

If `plan/` doesn't exist, create it with these three entries before registering the first task.

## 2. Task Format (5W2H)

Every task is a 5W2H entry. The fields are fixed; each carries its own meaning and the DoD is the acceptance criteria.

### 2.1 Task Entry

```markdown
- [ ] **Task X.Y: Title**

  - **What:** <concrete deliverable — one logical change>
  - **Why:** <root cause / motivation — not a restatement of What>
  - **Who:** Agent (<implementer|reviewer>).
  - **Where:** [file.ext](../path/to/file.ext#Lnn)
  - **When:** <dependencies; "No dependencies; first in batch" if none>
  - **How:** <the concrete approach — specific files/functions/transforms, not "fix it">
  - **How much:** <Trivial | Low | Medium | High | Very High>.
  - **Stakeholder Implications:** <None, or who/what is affected>
  - **DoD:** <measurable acceptance criteria, scoped to THIS task, not a later one>
  - **Options:** <required when there is a genuine decision — Recommended vs Alternative(s) + the data reality; otherwise omit>
```

Every field is mandatory and must be substantive. A missing, empty, or placeholder field (`<...>`, `TBD`, "fix it", "implement it") fails the quality gate ([§8](#8-task-quality-gate)) and the task does not get registered.

**Clarity over brevity.** An explanation's length is irrelevant — its clarity is not. A task is defined only when a domain-aware human can restate, in their own words, what the task is doing and why. If `What`/`Why` read like gibberish to a person, they will confuse the implementer too, and it will guess instead of asking.

If the implementer is confused about what to do, it should refuse to do the task saying it cannot do, given wrong information.

The reviewer should be critical about the task being done.

Worked bad→good examples of every rule above: [references/task-examples.md](./references/task-examples.md).

- `What` — the deliverable **and the data/context reality that constrains it**, in plain terms. State what is being done and the concrete situation that shapes the approach (e.g. "the COMMENT field is unreliable — 23 files scrape a number, 18 compute it"). No line numbers, no implementation detail — those live in `How`/`Where`. One logical change, not a bundle.
- `Why` — the root cause in plain terms: what is broken or missing today, and what goes wrong if left alone. Not a restatement of `What`.
- `How` — the concrete approach: the specific files/functions/transforms to touch. Never a tautology ("fix the bug", "implement the feature").
- `Who` — the worker that owns it (`implementer` for edits, `reviewer` for review). Mark `decision owner: user` when the task hinges on a choice the user must make.
- `Where` — the file (and line range) the task depends on. Every task has a file target.
- `When` — dependency ordering. States which tasks must complete first.
- `How much` — effort tier; drives the artifact decision ([§7](#7-effort-based-guide)) and the sizing rule ([§2.3](#23-task-sizing)).
- `Stakeholder Implications` — who/what is affected beyond the change itself.
- `DoD` — measurable and scoped to this task. Everything executed must satisfy it.
- `Options` — **required** when a task hides a genuine decision (multiple valid approaches with different tradeoffs). Record `Recommended` + `Alternative(s)` and the data reality that makes the choice non-obvious, so the implementer asks the user rather than assuming a policy. Omit only when the approach is genuinely unambiguous.

### 2.2 Solution Block

After execution, the implementer appends the Solution block under the task (see the `solving-tasks` workflow, step 4):

```markdown
  + **Solution:**
    - **Changed Files:**
      + [file.py L42](../path/to/file.py#L42) — brief what changed
    - **Summary:** 2-3 sentences.
    - **Technical Justification:** 2-4 sentences. WHY, not HOW.
    - **DoD Compliance:** 1-2 sentences.
    - **Artifacts:**
      + 📝 [walkthrough_<task-id>.md](./task-resolutions/walkthrough_<task-id>.md) — Walkthrough
      + 📚 [knowledge_<topic-slug>.md](./knowledge-base/knowledge_<topic-slug>.md) — Knowledge
```

> [!WARNING]
> **Brevity check:** keep the whole Solution block under **~1,500 characters (≈ 350 tokens)** — measured on the raw text, not rendered lines. "Lines" are ambiguous (one markdown line can hold a 2,000-character essay); characters/tokens are objective. Excess goes to the walkthrough.

The reviewer later appends a **Review Verdict** under the Solution block — per task, never a global notes block. Cap it at **~300 characters (≈ 75 tokens)** — the verdict, the DoD pass/fail, and one piece of evidence or one finding, not an essay (a single markdown line can hold a 2,000-character essay, so the cap is in characters, not lines):

```markdown
    - **Review Verdict (YYYY-MM-DD, reviewer):** <verdict> DoD: pass/fail with evidence.
```

A finding that isn't attached to the task it concerns is noise.

### 2.3 Task Sizing

A task is one logical change. Split it when either holds:

- **Multiple concerns** — the task bundles more than one distinct operation ("rebuild + copy + delete", "triage 18 tests"). One verb, one task.
- **High effort** — `How much` is `High` or `Very High`. Decompose into smaller tasks.

Anti-patterns from real plans: a single "rebuild the DB + copy it + delete stale husks" task is three tasks; a "triage N failing tests" task with N failures inlined is N classification decisions, not one. Split before registering — walkthrough granularity follows task granularity ([§5](#5-artifacts)). **Full bad→good sizing cases: [references/task-examples.md](./references/task-examples.md).**

## 3. Plan Structure

A plan file has, in order:

1. **Frontmatter** — title, description, status, author, related_files, tags.
2. **Scope & rules** — what may be touched, and what must not (commit/push/docs/tests).
3. **Source of truth** — the locked decisions file (e.g. `TODO.md`) and the plan itself; decisions pinned in the plan win over any draft text.
4. **Decisions pinned** — numbered, canonical answers to the open design decisions; the implementer applies them, the reviewer gates against them.
5. **Batches** — tasks grouped into batches that share context ([§4](#4-batching)).
6. **Deferred** — out-of-scope items explicitly parked, never silently dropped.

## 4. Batching

Group tasks into batches before execution:

- **≤5 tasks per batch** — one shared context, one subagent invocation.
- **Batch by dependency, not by file count** — tasks that feed one another are sequential (ordered invocations); only independent tasks on disjoint files may run in parallel.
- **State the ordering** — parallel vs sequential — explicitly in the plan, and confirm it with the user before fan-out.
- Each batch names the worker that runs it (`implementer` for edits, `reviewer` for review).

## 5. Artifacts

Two artifact kinds live under `plan/`:

- **Walkthrough** (`task-resolutions/walkthrough_<task-id>.md`) — the execution history of one task: objective, context correlation, design decisions, failure/correction log, affected files, validation evidence, and deep analysis. Template: `templates/walkthrough-template.md`.
- **Knowledge** (`knowledge-base/knowledge_<topic-slug>.md`) — a topic-scoped, reusable reference, not a task log. Template: `templates/knowledge-template.md`.

Which artifacts to create is decided by the Effort-Based Guide ([§7](#7-effort-based-guide)). The walkthrough links to the knowledge; the knowledge links back to the walkthrough and the plan (bidirectional linking).

## 6. Task States

A task's checkbox and title tag encode its state:

| State | Markup | Meaning |
| --- | --- | --- |
| Not started | `- [ ] **Task X.Y: Title**` | planned, not executed |
| In review | `- [ ] **[IN REVIEW]** **Task X.Y: Title**` | executed, awaiting review |
| Done | `- [x] **[REVIEWED]** **Task X.Y: Title**` | DoD verified, review passed |
| Rework | `- [ ] **Task X.Y: Title**` + `must redo: <finding>` | a finding requires rework |

The implementer marks `**[IN REVIEW]**` after execution. The reviewer flips the state at the gate: `[x]` when the DoD is verified against the code; back to `[ ]` with a `must redo: <finding>` note when a finding requires rework. A review that leaves the plan's checkboxes untouched has not registered.

## 7. Effort-Based Guide

The `How much` field drives which artifacts are required:

| Effort | Walkthrough | Knowledge |
| --- | --- | --- |
| Very High / High | ✅ Required (full depth) | ✅ If reusable content exists |
| Medium | ✅ Required | 📋 If reusable content exists |
| Low | ✅ Required (minimal) | ❌ Usually not needed |
| Trivial | 📋 Optional | ❌ Not needed |

If any reusable content exists, create the knowledge artifact before the walkthrough so the walkthrough can link to it.

## 8. Task Quality Gate

Before the reviewer registers a task in the plan, every one of these must hold — otherwise the task is sent back to be redefined, not registered:

- [ ] Every 5W2H field is present and substantive — no placeholders, no `TBD`, no "fix it".
- [ ] `What` states the deliverable **and** the data/context reality; a domain-aware human can restate the task's intention (what it does and why) in their own words. No line numbers or implementation detail in `What`/`Why`.
- [ ] `How` describes the concrete approach; `Why` states the root cause in plain terms (not a restatement of `What`).
- [ ] `DoD` is measurable and scoped to this task alone.
- [ ] `Where` points at a real file (and line range).
- [ ] The task is one logical change — no bundled operations, and `How much` is not `High`/`Very High` ([§2.3](#23-task-sizing)).
- [ ] `Options` records `Recommended` + `Alternative(s)` **and the data reality** wherever the approach hides a genuine decision. A task that presents one approach as obvious when the data makes it ambiguous is a failure.

The same gate applies on the implementer side before execution (see the `solving-tasks` workflow, step 1.1) — a task that fails there is fed back to the reviewer, not guessed at.
