---
# --- Antigravity rule keys ---
trigger: model_decision
description: "Use whenever the sequentialthinking MCP tool is called: when to invoke it, what each schema field means, and the revision, branch, and extension rules." # shared with VS Code
# --- VS Code instruction keys ---
name: "Sequential Thinking Tool Usage"
---

# sequentialthinking — call contract and field usage

> [!IMPORTANT]
> The tool is a reasoning ledger, not a transcript. The optional fields — `isRevision`, `revisesThought`, `branchFromThought`, `branchId`, `needsMoreThoughts` — are what separate real analysis from a numbered monologue. Leaving them at defaults is the failure this rule prevents.

## When to call the tool

Use `sequentialthinking` (VS Code exposes it as `mcp_sequential-th_sequentialthinking`) when at least one of these holds:

- The full scope is unknown and must be discovered step by step.
- Two or more designs, hypotheses, or root causes compete.
- Later evidence may invalidate an earlier conclusion.
- A long chain of dependent steps must stay internally consistent.
- Irrelevant context must be filtered out without losing the thread.

Do not call it when:

- One search, one file read, or one command answers the question.
- The next thought would only restate the previous one.
- The call would replace an actual tool call: running tests, reading `path:line`, searching the repo.

## Field contract

| Field | Type | Required | What it means | How to use it |
| :-- | :-- | :-- | :-- | :-- |
| `thought` | string | yes | The reasoning step itself. | One claim or decision per call. Carry evidence (`path:line`, command output) and the next step. Never paste whole files. |
| `thoughtNumber` | integer ≥ 1 | yes | Position in the chain. | Start at `1`, increment by exactly `1`. Never reset or reuse a number inside one chain. |
| `totalThoughts` | integer ≥ 1 | yes | Current estimate of chain length. | Estimate at thought 1 (3–8 is typical). Revise it in later calls when reality disagrees. It is an estimate, not a cap. |
| `nextThoughtNeeded` | boolean | yes | Whether another thought follows. | `true` on every intermediate thought; `false` only on the final thought that states the answer. |
| `isRevision` | boolean | no | Marks this thought as replacing an earlier one. | Set `true` when new evidence contradicts an earlier thought, and name that thought in `revisesThought`. |
| `revisesThought` | integer ≥ 1 | with `isRevision: true` | Number of the thought being replaced. | Must be lower than `thoughtNumber`. Send only together with `isRevision: true`. |
| `branchFromThought` | integer ≥ 1 | no | Thought where an alternative line of reasoning starts. | Set when exploring a competing option without discarding the main line. Must reference an existing thought. |
| `branchId` | string | with `branchFromThought` | Stable label for that branch. | Reuse the same id for every thought in the branch (`alt-validation-normalization`). Never reuse an id for a different hypothesis. |
| `needsMoreThoughts` | boolean | no | Signals the chain must extend past `totalThoughts`. | Set `true` on the last planned thought when the problem is still open, and raise `totalThoughts` in the same call. |

> [!WARNING]
> The schema accepts `boolean | string` for `isRevision`, `needsMoreThoughts`, and `nextThoughtNeeded`. Always send JSON booleans (`true`), never `"true"`. Integers are JSON numbers (`3`), never `"3"`; string integers are rejected with `Invalid totalThoughts: must be a number`.

## Rules

1. **Use the field that matches reality.** A chain that never sets `isRevision`, `branchFromThought`/`branchId`, or `needsMoreThoughts` is a monologue wearing a plan's clothes. If a thought contradicts, revises, branches, or extends, the field must say so.
2. **Revise, do not overwrite.** When evidence invalidates thought *n*, submit a new thought with `isRevision: true` and `revisesThought: n`, then continue from the corrected position. Silently contradicting an earlier thought destroys the audit trail.
3. **Branch, do not abandon.** To compare two designs, open a branch with `branchFromThought` + `branchId`, keep that id stable for the whole branch, and close the branch by stating which line wins and why.
4. **Adjust `totalThoughts` out loud.** Raise it when the problem is bigger than estimated, lower it when smaller. On the last planned thought, send `nextThoughtNeeded: true` with `needsMoreThoughts: true` if unresolved; send `false` only when the answer is ready.
5. **Close with a falsifiable hypothesis.** The final thought (`nextThoughtNeeded: false`) states the decision, the evidence, and the check that can fail: a command, a test, or a `path:line` comparison. Without a falsifiable check the chain is not finished.
6. **Thinking is not doing.** A thought chain never replaces the command, test, or file read that verifies it. Run the check after the chain closes; if it fails, reopen the chain with `isRevision: true` against the closing thought.
7. **Stop on evidence, not on a counter.** If three consecutive thoughts add no new evidence — no new `path:line`, command output, dataset, or external source consulted after checking other files and possibilities — stop the chain and answer with what is known, or ask the user. The loop signal is repetition, not the number of calls.
8. **One chain per task.** `thoughtNumber` increases monotonically within a task. A new, unrelated task starts again at `1`.

## Worked calls

Start a chain with a real estimate:

```json
{
  "thought": "GPU 2-opt stage is slower than CPU for n<500. Hypothesis: per-launch overhead dominates fixed cost in src/algorithms/presets.py. Next: read benchmark_scripts/compare_local_search_stages.py to confirm launch counts.",
  "nextThoughtNeeded": true,
  "thoughtNumber": 1,
  "totalThoughts": 5
}
```

Expected: server records thought 1 of 5 and returns no error.

Revise an earlier thought after evidence arrives:

```json
{
  "thought": "Revision of thought 1: launch count is not the cause; the distance matrix is rebuilt per chromosome. Evidence: src/distances/matrix.py rebuilds on every call. Replacing the hypothesis.",
  "nextThoughtNeeded": true,
  "thoughtNumber": 3,
  "totalThoughts": 5,
  "isRevision": true,
  "revisesThought": 1
}
```

Expected: thought 3 is attached to thought 1 as a revision; chain length unchanged.

Branch into a competing option:

```json
{
  "thought": "Branch from thought 2: alternative is caching the matrix at population level instead of per chromosome. Compare blast radius against a kernel-level fix.",
  "nextThoughtNeeded": true,
  "thoughtNumber": 4,
  "totalThoughts": 6,
  "branchFromThought": 2,
  "branchId": "alt-matrix-cache"
}
```

Expected: thought 4 is labeled under `alt-matrix-cache`, branching from thought 2.

Extend past the estimate:

```json
{
  "thought": "Reached the planned fifth thought; branch comparison is unresolved. Extending the chain by two thoughts to close the comparison.",
  "nextThoughtNeeded": true,
  "thoughtNumber": 5,
  "totalThoughts": 7,
  "needsMoreThoughts": true
}
```

Expected: chain continues past the original estimate; no truncation.

Close with a falsifiable hypothesis:

```json
{
  "thought": "Hypothesis: population-level matrix caching removes the per-chromosome rebuild and closes the n<500 gap. Verify: run benchmark_scripts/compare_local_search_stages.py on a280 and require the GPU stage to beat CPU; the pre-change run must lose.",
  "nextThoughtNeeded": false,
  "thoughtNumber": 6,
  "totalThoughts": 6
}
```

Expected: server records the final thought; no further calls in this chain.

## Failure modes

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| `Invalid totalThoughts: must be a number` | String integer (`"5"`). | Send JSON numbers. |
| Chain reads as a list of statements | Optional fields never used. | Set revision, branch, extension when the reasoning actually does it. |
| Same insight repeated for many calls | No stop condition. | Apply rule 7: three thoughts without new evidence means stop. |
| `revisesThought` has no effect | Sent without `isRevision: true`. | Send both together. |
| Branch ids reused across hypotheses | `branchId` chosen per call, not per hypothesis. | One stable id per branch. |
