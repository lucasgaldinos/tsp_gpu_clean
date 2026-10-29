# Orchestration Patterns — Worked Examples

Load this file when a concrete orchestration pattern is chosen (step 2 of the main skill) and worked `.agent.md` examples are needed. Compiled 2026-08-26 from the official VS Code subagents and custom-agents docs.

## 1. Coordinator and Worker

One coordinator owns the workflow; narrow specialists do the work. The coordinator stays lightweight because each worker carries its own instructions, tools, and model.

**Coordinator** — enables the `agent` tool and restricts delegation to an allowlist:

```markdown
---
name: feature-builder
description: Coordinates feature development across planning, implementation, and review subagents.
tools: ['agent', 'edit', 'search', 'read']
agents: ['planner', 'plan-architect', 'implementer', 'reviewer']
---
You are a feature development coordinator. For each feature request:

1. Use the planner agent to break down the feature into tasks.
2. Use the plan-architect agent to validate the plan against codebase patterns.
3. If the architect identifies reusable patterns or libraries, send feedback to the planner to update the plan.
4. Use the implementer agent to write the code for each task.
5. Use the reviewer agent to check the implementation.
6. If the reviewer identifies issues, use the implementer agent again to apply fixes.

Iterate between planning and architecture, and between review and implementation, until each phase converges.
```

**Workers** — hidden from the picker (`user-invocable: false`), least-privilege tools, cheap model where the job is narrow:

```markdown
---
name: planner
user-invocable: false
tools: ['read', 'search']
---
Break down feature requests into implementation tasks. Incorporate feedback from the plan-architect agent.
```

```markdown
---
name: plan-architect
user-invocable: false
tools: ['read', 'search']
---
Validate plans against the codebase. Identify existing patterns, utilities, and libraries that should be reused. Flag any plan steps that duplicate existing functionality.
```

```markdown
---
name: implementer
user-invocable: false
model: ['DeepSeek V4 Flash (deepseek)', 'DeepSeek V4 Pro (deepseek)']
---
Write code to complete assigned tasks.
```

Model names must match the model picker exactly (format `Model Name (vendor)`, vendor suffix case-sensitive — `(deepseek)`, not `(DeepSeek)`). The examples use BYOK DeepSeek names; substitute whatever cheap, capable model your environment exposes — the point is a prioritized list: first available wins, falls through in order.

Why this works: planning and review agents get read-only tools (they can't break anything), the implementer gets a cheap fast model (its job is narrow), and the coordinator's context holds only the workflow, not the workers' intermediate noise.

## 2. Multi-Perspective Code Review

One artifact, several independent lenses run in parallel, then synthesized. The cheap version needs **no worker agent files** — the orchestrator shapes each lens through its prompt:

```markdown
---
name: thorough-reviewer
description: Reviews code through multiple independent perspectives in parallel.
tools: ['agent', 'read', 'search']
---
You review code through multiple perspectives simultaneously. Run each perspective as a parallel subagent so findings are independent and unbiased.

When asked to review code, run these subagents in parallel:
- Correctness reviewer: logic errors, edge cases, type issues.
- Code quality reviewer: readability, naming, duplication.
- Security reviewer: input validation, injection risks, data exposure.
- Architecture reviewer: codebase patterns, design consistency, structural alignment.

After all subagents complete, synthesize findings into a prioritized summary.
Note which issues are critical versus nice-to-have. Acknowledge what the code does well.
```

Why this works: each lens starts fresh and is not anchored by what another lens already found. The prompt shapes the focus — no extra files to maintain.

Upgrade path when lenses need different tooling: make each perspective its own custom agent. A security reviewer could use a security-focused MCP server, a quality reviewer could access linting CLI tools. Each lens then gets the best tools for its job instead of the orchestrator's generic set.

## 3. Research, Then Act

Facts first, edits second. One subagent gathers; the main agent (or a second subagent) acts on the gathered facts. Keeps exploratory noise out of the main thread.

```text
Run a subagent to research the new feature implementation details and return only
information relevant for user documentation. Then update the docs/ folder with the
new documentation.
```

The research subagent returns a distilled summary; the main thread only ever sees that summary. This is the default pattern for "explain X in this codebase, then change Y based on it" requests.

## 4. Recursive Divide-and-Conquer

Uniform large work (N items) split into halves recursively. Requires two conditions: the agent lists itself in its own `agents` property, and `chat.subagents.allowInvocationsFromSubagents` is enabled (off by default; max nesting depth 5 when on).

```markdown
---
name: recursive-processor
tools: ['agent', 'read', 'search']
agents: [recursive-processor]
argument-hint: A list of items to process
---
You process a list of items by dividing and conquering:

- If the list has more than 4 items, split it in half and delegate each half to a recursive-processor subagent.
- If the list has 4 or fewer items, process the items directly.
- Merge the results from each subagent into a final result.
```

Why nesting is off by default: it prevents infinite recursion when agents accidentally call themselves in a loop. Only enable the setting when the recursion is genuinely bounded, and keep a hard base case (here: ≤ 4 items) so the chain always terminates.

## Sequencing and File Partitioning Rules

- **Parallel** iff tracks are independent (no output feeds another) **and** target disjoint files. Same filesystem, no merge — overlapping writes clobber silently.
- **Sequential** when track B consumes track A's output. Pass the distilled output into B's prompt; never try to message a running subagent.
- **Partition explicitly**: write the file-target split into each prompt (e.g., "You may only edit `src/benchmarks_v2/**`"). Do not rely on the workers to coordinate among themselves — they can't.

## Synthesis Template

After all tracks report, produce:

```text
## <Artifact> — synthesis

### Summary
<2-3 sentences; overall verdict>

### Findings (critical)
| Finding | Track | Location | Action |
| --- | --- | --- | --- |

### Findings (nice-to-have)
| Finding | Track | Location | Action |
| --- | --- | --- | --- |

### Unverified
<what could not be confirmed and why>
```
