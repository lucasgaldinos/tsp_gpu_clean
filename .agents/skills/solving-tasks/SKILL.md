---
name: solving-tasks
description: "Step-by-step process for executing plan tasks: understand, execute, create artifacts (walkthrough + knowledge), update plan solution, handle review feedback, and manage state transitions."
disable-model-invocation: true
---
# Solving Tasks — Workflow

> [!NOTE]
> This workflow describes the **process** (the "how-to").
> For templates, naming conventions, and formatting standards, read the
> **skill** first: [task-managing/SKILL.md](../skills/task-managing/SKILL.md).
> If the task belongs to a specialized domain with a repo skill, load that skill before executing. Example: load `latex-compiling` before LaTeX build/debug work.
> If the task creates or improves an agent skill, load `skills-best-practices` when available. If the skill registry cannot load it, search for and read the local skill file before editing rather than guessing the structure.

---

## Step 0: Locate the Plan

<instructions step=0>

Before starting any task, identify the plan directory and plan file:

1. Look for a directory typically named `plan/` within the project structure
2. Inside it, find the main plan file (e.g., `plan.md`, `plan-roteiro.md`, `roadmap.md`)
3. Note the paths to `task-resolutions/` and `knowledge-base/` subdirectories

If the plan directory doesn't exist yet, create the structure as defined in
the skill's [§1 (Directory Structure)](../skills/task-managing/SKILL.md#1-directory-structure).

</instructions>

---

## Step 1: Understand the Task

<instructions step=1>

Read the task's 5W2H definition in the plan file carefully. Before writing any code:

1. **Read the DoD** — This is your acceptance criteria. Everything you do must satisfy it.
2. **Read the "Where"** — Open and read ALL referenced files entirely. Correlate with
   relevant data files, source code, and documentation.
3. **Read the "Why"** — Understand the root cause / motivation. Don't just fix symptoms.
4. **Read the "How"** — Understand the concrete approach. If it's vague or missing, that's a quality-gate failure (see [1.1](#11-task-quality-gate)).
5. **Check the "Options"** — If the task records `Options` (Recommended vs Alternative), confirm which one the user chose. Do not silently pick.
6. **Check for dependencies** — Look at the "When" field. Are prerequisite tasks done?
7. **Check existing artifacts** — Are there walkthroughs or knowledge artifacts from
   related tasks that provide context?

### 1.1 Task Quality Gate

Before implementing, verify the task is well-defined and correctly sized (the reviewer's [§8 gate](../skills/task-managing/SKILL.md#8-task-quality-gate), re-checked on the execute side):

1. Every 5W2H field is present and substantive — no placeholders, no `TBD`, no "fix it". The decision must be done by using #skill [ask-clarification-question](../skills/ask-clarification-questions/SKILL.md)
2. The `How` describes a concrete approach; the `Why` states the root cause (not a restatement of `What`).
3. Every `Where` path exists and points to the file the task actually depends on.
4. The DoD is measurable and scoped to this task, not to a later task.
5. The `When` dependencies make sense and are already satisfied or intentionally deferred.
6. The task is one logical change — no bundled operations, and `How much` is not `High`/`Very High` (skill §2.3).
7. If the task records `Options`, one option is marked as chosen by the user — never silently pick one.
8. Important choices that change scope, environment, dependencies, or user-facing behavior are explicit.

If any item fails, fix the plan entry first if the correction is factual and low-risk. If it requires judgment — or the task is oversized/ambiguous and needs redefinition — feed it back to the reviewer (or ask the user) rather than guessing.

> [!IMPORTANT]
> Do NOT start executing until you fully understand the task. If the task description
> is ambiguous or missing key details, ask the user for clarification BEFORE proceeding.
> Use the clarification skill (`ask-clarification-questions` / `ask-questions-if-underspecified`) and `vscode_askQuestions` when available.

</instructions>

---

## Step 2: Execute the Work

<instructions step=2>

Perform the actual implementation. During execution:

1. **Track changes** — Keep a mental (or scratch) note of every file modified, with
   line numbers and what changed.
2. **Note design decisions** — Record WHY you chose approach A over approach B.
   These will go into the walkthrough.
3. **Identify reusable knowledge** — If you research a topic, tool, or pattern that
   could be useful beyond this specific task, flag it for a knowledge artifact.
4. **Watch for incidental findings** — Bugs, inconsistencies, or issues discovered
   during execution should be documented, even if not directly related to the task.

### 2.1 Diagnostic Discipline

When a command, build, test, render, or compile fails:

1. Record the hypothesis for the next attempt before rerunning anything.
2. Retry only if the next run tests a new, concrete hypothesis.
3. After one failed retry, stop running the same class of command and inspect primary evidence: logs, generated intermediate files, stack traces, exit codes, and changed outputs.
4. For high-blast-radius targets, reproduce the smallest failing case in a sandbox, temp copy, fixture, or dummy file before touching the canonical target again.
5. Do not use official/canonical artifacts as scratchpads unless the task explicitly requires it.
6. Sandbox and temp files should be deleted after their evidence is captured, but ask the user for confirmation before removing them.
7. Ask before installing dependencies, changing environment assumptions, adopting fallbacks, or broadening the task scope.
8. If the root cause remains unclear after reading evidence and trying one focused fix, search the repository first with `rtk rg` when the error, symbol, or path may exist locally. Then use available documentation, web, GitHub, or code-search tools. If it is still unclear, ask the user how to proceed.

</instructions>

---

## Step 3: Create / Update Artifacts

<instructions step=3>

After the work is done (or at natural checkpoints for large tasks), create the artifacts.
The order matters — create artifacts first, then update the plan.

### 3.1: Decide What Artifacts Are Needed

Use the **Effort-Based Guide** from the skill's §7:

| Effort | Walkthrough | Knowledge |
| --- | --- | --- |
| Very High / High | ✅ Required (full depth) | ✅ If reusable content exists |
| Medium | ✅ Required | 📋 If reusable content exists |
| Low | ✅ Required (minimal) | ❌ Usually not needed |
| Trivial | 📋 Optional | ❌ Not needed |

### 3.2: Create Knowledge Artifact(s) — If Applicable

> [!IMPORTANT]
> **BEFORE skipping this step**, consider these questions about the content
> produced during execution. Answer the ones most relevant to this task:
>
> 1. "Would someone working on a DIFFERENT task benefit from reading any of this?"
> 2. "Did I research a standard, spec, or tool that isn't documented elsewhere?"
> 3. "Is any concept here reusable outside the context of this specific change?"
> 4. "Did I find architectural or library behavior that surprised me?"
> 5. "Would I want this reference available if I worked on a similar problem later?"
>
> If **any** answer is "yes" → create a knowledge artifact for that content.
>
> Create knowledge BEFORE the walkthrough, so the walkthrough can link to it.

1. Copy the template from `templates/knowledge-template.md`
2. Fill in the YAML frontmatter:
   - `title`: "Knowledge: Topic Title"
   - `created`: current date/time
   - `author`: project author
   - `tags`: relevant technology/domain/scope tags
   - `links`: link back to the plan file and the task's walkthrough (add after creating)
3. Write the content — topic-scoped, reusable, NOT task-specific
4. Save to `<plan_dir>/knowledge-base/knowledge_<topic-slug>.md`

### 3.3: Create / Update Walkthrough

1. Copy the template from `templates/walkthrough-template.md`
2. Fill in the YAML frontmatter:
   - `title`: "Walkthrough: Task X.Y — Title"
   - `task_id`: the task identifier
   - `status`: `"in-review"`
   - `tags`: relevant tags
   - `links`: link to plan file + any knowledge artifacts created
3. Write the content:
   - **Objective**: mirrors the task's "What"
   - **Context table**: correlate code ↔ docs ↔ data (the "before" state)
   - **Design Decisions**: table or prose with key trade-offs
   - **Failure / Correction Log**: failed attempts, evidence read, user corrections, and final outcome
   - **Affected Files**: detailed, grouped by component, with line anchors
   - **Checklist**: all changes with `[x]` marks
   - **Validation Evidence**: commands, logs, sandbox results, and known validation boundaries
   - **Deep Analysis**: deep pros/cons, calculations, justifications —
     everything that would be TOO LONG for the plan's solution section
4. If knowledge artifacts were created, link to them inline:
   `For full analysis, see [knowledge_<topic-slug>.md](../knowledge-base/knowledge_<topic-slug>.md)`
5. Save to `<plan_dir>/task-resolutions/walkthrough_<task-id>.md`

### 3.4: Update Knowledge Artifact Links

Go back to any knowledge artifact(s) created in 3.2 and add the walkthrough
to their `links` frontmatter field (bidirectional linking).

</instructions>

---

## Step 4: Update the Plan File

<instructions step=4>

After artifacts are created, update the plan file's task entry:

### 4.1: Append the Solution Section

Add the **brief** solution block (see skill §2.2 for format):

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
> **Brevity check:** If the Solution section exceeds ~1,500 characters (≈ 350 tokens), it's too long.
>
> Move the excess to the walkthrough.

### 4.2: Mark as In Review

Change the task title to include the `**[IN REVIEW]**` tag:

```markdown
- [ ] **[IN REVIEW]** **Task X.Y: Task Title**
```

### 4.3: Notify the User

Inform the user that the task is ready for review. Summarize:

- What was done (1-2 sentences)
- Which artifacts were created/updated (with links)
- Any incidental findings or concerns

</instructions>

---

## Step 5: Handle Review Feedback

<instructions step=5>

The user reviews the work and either approves or requests changes.

### If Approved (User marks as done)

User removes `**[IN REVIEW]**` and changes `[ ]` → `[x]`.
Update the walkthrough's `status` frontmatter from `"in-review"` to `"done"`.

### If Changes Are Needed

1. **Execute the requested changes**
2. **Update ALL artifacts** — not just the plan:
   - Update the **walkthrough** to reflect the new state
   - Update the **knowledge artifact** if its content was affected
   - Update the **plan's solution section** if files/summary changed
3. **Re-mark as `**[IN REVIEW]**`**
4. Notify the user again

> [!IMPORTANT]
> Every review cycle updates ALL artifacts. The walkthrough and knowledge must
> always reflect the **final state** of the work, not just the initial version.

</instructions>

---

## Step 6: Completion

<instructions step=6>

When the user marks the task as `[x]`:

1. Verify the walkthrough `status` is `"done"`
2. Verify all cross-links are correct and bidirectional
3. Verify tags are consistent across plan, walkthrough, and knowledge

The task is now closed. The plan file serves as the index, the walkthrough
preserves the execution history, and the knowledge is available for future tasks.

</instructions>

---

## Quick Reference: Artifact Checklist

Use this checklist after completing any task:

- [ ] **Walkthrough** created/updated in `task-resolutions/`
- [ ] Walkthrough has YAML frontmatter with `status: "in-review"`
- [ ] Walkthrough links to plan file in `links` field
- [ ] Explicitly decided: **Knowledge** artifact ✅ created / ❌ not needed because `_________`
- [ ] If created, knowledge has YAML frontmatter with tags and links
- [ ] If created, knowledge links back to walkthrough and plan
- [ ] **Plan Solution** section is brief (~20 lines max)
- [ ] Plan Solution has `Artifacts` field linking to walkthrough + knowledge
- [ ] Task title has `**[IN REVIEW]**` tag
- [ ] Task quality gate passed, or plan corrections/user decisions were recorded
- [ ] Diagnostic retries were bounded and evidence was read before further attempts
- [ ] User has been notified
