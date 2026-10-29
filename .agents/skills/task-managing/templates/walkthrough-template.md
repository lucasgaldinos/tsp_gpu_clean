---
title: "Walkthrough: Task {{task_id}} — {{task_title}}"
task_id: "{{task_id}}"
description: "{{brief_description}}"
created: "{{date:YYYY-MM-DD HH:mm Z}}"
author: "[[{{author}}]]"
status: "in-review"
tags: []
links:
  - "[{{plan_file}}]({{plan_file_path}})"   # {{plan_file_path}} is relative to this file, e.g. ../plan-<name>.prompt.md
---

# Walkthrough: Action {{task_id}} — {{task_title}}

## Objective

<!-- 1-2 sentences. Mirrors the task's What, Why, and How — the deliverable, the root
     cause, and the concrete approach. If the task recorded Options, state which option
     was chosen. -->

## Context: Code ↔ Documentation ↔ Data Correlation

<!-- Table correlating the relevant artifacts BEFORE changes were made.
     This establishes the "before" snapshot. -->

| Artifact | Location | Relevant Observation |
| --- | --- | --- |
| <!-- component --> | [file](../path/to/file.ext#Lnn) | <!-- what's relevant --> |

---

## Impact Analysis

### Root Cause

<!-- What was wrong / what motivated this task. Link to evidence. -->

### Design Decisions

<!-- Table or prose explaining the key decisions made. If the task recorded `Options`,
     state which option was chosen and why — link the decision back to the plan.
     For examples of well-defined tasks (bad vs good), see the task-managing skill's
     references/task-examples.md. -->

| Component | Before | After | Rationale |
| --- | --- | --- | --- |
| <!-- what --> | <!-- old --> | <!-- new --> | <!-- why --> |

### Failure / Correction Log

<!-- Record failed attempts, evidence read before retrying, user corrections,
     and what changed because of that evidence. Keep this task-specific. -->

| Attempt / Event | Evidence Read | Correction / Decision | Outcome |
| --- | --- | --- | --- |
| <!-- compile/test/edit attempt --> | <!-- log/file/user note --> | <!-- changed approach --> | <!-- result --> |

---

## Affected Files

<!-- Grouped by language/component. More detail than the plan's "Changed Files". -->

### {{Language/Component Group}} (N files)

1. [file.ext](../path/to/file.ext#Lnn) — Lnn: detailed description of change
2. [file2.ext](../path/to/file2.ext#Lnn) — Lnn: detailed description of change

---

## Change Checklist

- [ ] 1. `file.ext` — brief description
- [ ] 2. `file2.ext` — brief description

---

## Validation Evidence

<!-- Commands, logs, sandbox/dummy results, manual checks, and boundaries.
     State explicitly what was NOT validated and why, especially when validation
     belongs to a later task. -->

| Check | Evidence | Result | Boundary / Follow-up |
| --- | --- | --- | --- |
| <!-- test/build/manual check --> | <!-- command/log/file --> | <!-- pass/fail/blocked --> | <!-- what remains --> |

---

## Deep Analysis

<!-- This is where deep technical content lives — the content that is TOO LONG
     for the plan's Solution section. Examples:
     - Pros vs. Cons analysis
     - Overflow/precision calculations
     - Performance impact analysis
     - Mathematical proofs / derivations
     - Architecture trade-offs
     
     If this analysis is REUSABLE (not specific to this task), extract it into
     a Knowledge artifact and link to it here instead. -->

---

## Additional Findings

<!-- Optional. Bugs found during execution, unexpected findings, etc.
     Use callouts for visibility. -->

> [!WARNING]
> <!-- Description of unexpected finding -->

---

## Review Notes

<!-- Optional. Things the reviewer should pay special attention to. -->
