---
name: orchestrate-subagents-vscode
description: >
  Use this skill whenever work should be delegated to subagents in VS Code with GitHub Copilot:
  review-then-plan pipelines, coordinator/worker orchestration, multi-perspective review,
  research-then-implement, or scaffolding `.agent.md` orchestrator and worker files. Trigger on
  "use a subagent", "delegate this", "run these in parallel", "orchestrate agents", "spawn a
  reviewer", "set up an orchestrator agent", or whenever a task benefits from context isolation,
  parallel tracks, or a second un-anchored opinion — even if the user never says "subagent".
argument-hint: "task to orchestrate; pipeline: assess -> plan/register -> launch -> execute -> review -> synthesize"
---

# Orchestrating Subagents in VS Code (GitHub Copilot)

<context>

A subagent is a one-shot, context-isolated worker: the main agent launches it through the `agent` / `runSubagent` tool and it returns exactly one report. The orchestrator is the `RUG` agent, which loads this skill as its operating manual. RUG decomposes, batches, dispatches, tracks, and synthesizes — the reviewer plans and registers; the implementer executes. Review runs at the start and end gates only.

</context>

<prerequisites>

- [ ] The orchestrator (RUG) has the `agent` tool in `tools` and an `agents` allowlist of `['reviewer', 'implementer']`.
- [ ] The repo's canonical skills are present: `code-review-and-quality`, `task-managing`, `explore` (`.agents/skills/`), and the `solving-tasks` workflow (`.agents/workflows/`). The reviewer loads the first three; the implementer loads the last two.
- [ ] Worker names are exact and case-sensitive (`reviewer`, `implementer`).
- [ ] Nesting is off by default — no worker spawns subagents; only RUG delegates.

</prerequisites>

<steps>

<step n="1">

<instructions>

Review, plan & register — one reviewer dispatch. Build a reading list from the user's request (the files and line ranges relevant to the goal), then dispatch `reviewer` ONCE to do all three in a single pass: review the reading list (loads `code-review-and-quality`, plus `explore` when it must locate code or gather external sources), decompose the deficiencies into 5W2H tasks (each with a measurable DoD and a file target) batched by shared context (≤5 per batch), and write those tasks to the plan via `task-managing` — never prose bullets. The reviewer that reads the code is the same one that writes the tasks — no separate review-then-write pass. No git-diff workflows.

</instructions>

<validation>

Every deficiency maps to a registered task with a DoD and a file target; findings and tasks trace back to the reading list.

</validation>

<recovery>

If the request doesn't yield a reading list, ask the user for the files before dispatching review.

</recovery>

</step>

<step n="2">

<instructions>

Confirm — you (RUG) confirm the batching and the worker assignment with the user before any fan-out — hybrid authority. Use `vscode_askQuestions` for the go/no-go and scope decisions (never a bare "reply go" prompt). State which batches run in parallel vs sequentially. This step is yours alone; no subagent runs.

</instructions>

<validation>

The user has seen and approved the plan (tasks, batches, worker assignment).

</validation>

<recovery>

If the user rejects the plan, send it back to the reviewer to revise, then re-propose.

</recovery>

</step>

<step n="3">

<instructions>

Launch — thin dispatch. Independent batches in parallel; dependent batches sequential (feed prior output into the next prompt). Name workers exactly. Model resolution: explicit `runSubagent` model → worker frontmatter `model` → parent model; never exceed the parent's cost tier. Local only — for CLI delegation set DeepSeek via `COPILOT_*` env vars. Every prompt = goal + reading list + deltas — thin, per `<output_format>`; never restate the worker's procedure, skills, or output shape.

</instructions>

<validation>

Track count matches the plan; each batch got its full prompt; no two parallel batches write the same files.

</validation>

<recovery>

If `runSubagent` is unavailable, check RUG's `tools` for `agent` — or do the work inline rather than faking parallelism.

</recovery>

</step>

<step n="4">

<instructions>

Execute & track. Execution is delegated to `implementer`, which loads `solving-tasks` + `task-managing`, executes, and updates task state and artifacts. Track batches with a todo list (one entry per batch). If a batch fails mid-run, register why it failed — do not auto-retry.

</instructions>

<validation>

Every batch's outcome is recorded; failures are captured with a reason.

</validation>

<recovery>

An underspecified task is a planning gap — feed it back to the reviewer (step 1), not into more execution.

</recovery>

</step>

<step n="5">

<instructions>

Final review — integration guard AND quality critique. Dispatch `reviewer` again over the SAME reading list (plus files changed during execution). The reviewer's own agent file already defines the critique depth (full `code-review-and-quality` five-axis), the per-task `Solution` annotations, and the checklist state marking — send a thin prompt (goal + reading list + deltas) and never restate or override those duties. Findings flow back to `implementer` for fixes. Cap 2 review→fix iterations; after the cap, register the remaining findings as unresolved tasks and ask the user. No per-task validation subagent — review happens at gates only.

</instructions>

<validation>

The final review's reading list matches the initial review's context; unresolved findings are registered, not abandoned.

</validation>

<recovery>

If context was lost, rebuild the reading list from the plan before reviewing.

</recovery>

</step>

<step n="6">

<instructions>

Synthesize & register. Merge findings into one prioritized summary (deduplicate, sort by severity, keep attribution). Register final task states in `task-managing` and mark the orchestration done.

</instructions>

<validation>

The summary is shorter than the raw reports; every claim traces to a batch; task states reflect reality.

</validation>

<recovery>

State explicitly what you could not verify rather than implying success.

</recovery>

</step>

</steps>

<checkpoint>

Pause and confirm with the user when any of these hold:

- The fan-out exceeds ~3 parallel batches and cost matters.
- Batches would write overlapping files and you cannot partition targets.
- The work is destructive or irreversible (deletions, migrations, force pushes).

</checkpoint>

<rules>

1. **Thin dispatcher.** RUG decomposes, batches, dispatches, tracks, and synthesizes. It never plans, reviews, or executes — those are the reviewer's and implementer's jobs. RUG may read files to orient but never writes code or edits files — including the plan/TODO. Task registration is the reviewer's job.
2. **Reviewer owns review + plan + register.** One worker reviews, plans, and registers tasks (loads `code-review-and-quality` + `task-managing`). Its `edit` is scoped to the plan file; source stays read-only.
3. **Implementer consumes, never plans.** The implementer executes batched tasks and updates state/artifacts (loads `solving-tasks` + `task-managing`). It does not write new tasks.
4. **Stateless, one-shot.** A subagent gets one prompt and returns one report — no follow-ups, no mid-run iteration.
5. **No interactive tools in subagents.** Subagents can't use clarifying-question tools or todo management. Resolve user decisions in the main thread first.
6. **Context = skill reference + reading list.** A subagent's background is the canonical skill file(s) it must load plus a curated reading list. Never inline a skill's content; never build a giant shared "context pack."
7. **Review at start and end, same context.** Initial review surfaces deficiencies; final review (same skill, same reading list) is the integration guard. Both run `code-review-and-quality`.
8. **Hybrid authority.** RUG proposes the plan; the user confirms before fan-out.
9. **Batching ceiling.** ≤5 tasks per subagent invocation, grouped by shared context.
10. **Failure discipline.** Cap 2 review→fix iterations; then register the failure in `task-managing` and ask the user.
11. **Same filesystem, no merge.** Parallel batches share the workspace; partition file targets explicitly.
12. **Exact names & models.** Agent names are case-sensitive; model names must match the picker exactly (e.g. `DeepSeek V4 Flash (deepseek)`). Requesting an unavailable or pricier model makes the subagent refuse.
13. **`infer` is dead.** Use `user-invocable` and `disable-model-invocation`.
14. **Batch by dependency, not by file count.** Before proposing batches, analyze the dependency graph: tasks that feed one another are sequential (separate invocations, ordered); only independent tasks on disjoint files can run in parallel. State the ordering (parallel vs sequential) explicitly in the plan and confirm it with the user.
15. **Thin prompts; the agent file is the spec.** Never restate a worker's procedure, skills, or output shape in a dispatch prompt — the worker's `.agent.md` defines them, and restating creates a second source of truth that drifts. The prompt carries only goal + reading list + deltas. Changing a worker's duties (e.g. "skip plan registration this pass") requires the user's confirmation first — never imposed silently.

</rules>

<gotchas>

- **Cloud agent ignores some frontmatter** (`handoffs`, `argument-hint`). This workflow is local-only anyway.
- **Handoffs ≠ subagents.** Handoffs are human-in-the-loop UI transitions; subagents are automatic workers.
- **A skill can fork into a subagent** (`context: fork`). Don't confuse forked skills with `runSubagent` workers.
- **Worker frontmatter overrides inheritance.** A worker is read-only only if its frontmatter makes it so.
- **Claude Code tool names differ** (camelCase in VS Code, `Write`/`Edit` in Claude). Keep harness-specific names out of shared scripts.

</gotchas>

<output_format>

Delegation prompt template — keep it thin. The worker's `.agent.md` already defines its procedure, the skills it loads, and its output shape; restating any of it clutters the worker and creates a second source of truth that drifts apart. The prompt carries only run-specific deltas:

```text
<goal>
What this dispatch must accomplish — the specific tasks with their DoD (the implementer's task list, or the reviewer's gate and focus).

<reading_list>
The exact files (and line ranges) to read. Review/operate only on these.

<deltas>
Only what differs from the worker's defaults: extra constraints (e.g. "do not edit <paths>"), scope changes, or duty changes. A duty change (e.g. "skip plan registration this pass") must be confirmed with the user before dispatch — never imposed silently.
```

Append this report-back discipline line to every prompt:

```text
Return only the result. No narration, no "here's what I did".
```

</output_format>

## Adjacent Surface: Copilot CLI

The VS Code delegation model above has a CLI sibling, relevant when the user works from the terminal:

- **`/fleet`** — the CLI orchestrator: decomposes an objective, runs independent work items as parallel background subagents, respects dependencies, synthesizes the result. Non-interactive: `copilot -p "/fleet <task>" --no-ask-user`.
- **`/delegate`** — hands a task to a Copilot cloud agent: new branch prefixed `copilot/`, file changes, tests, then a draft PR for human review.
- **Rubber-duck agent** (`/experimental`, v1.0.42+) — multi-model second perspective routed through a complementary model family.
- **Concurrency and depth limits** (`/settings`, v1.0.66+, usage-based billing) — cap parallel subagents and delegation depth.

Load [references/cli-fleet.md](./references/cli-fleet.md) when the user actually runs Copilot CLI orchestration, needs `/fleet` prompt-mode caveats, or asks about CLI/cloud-agent behavior differences.

## References

| Load when | Source |
| --- | --- |
| The reviewer's review procedure (five axes, spec-adherence) | `code-review-and-quality` skill (`.agents/skills/`) |
| Task format (5W2H), plan structure, task states | `task-managing` skill (`.agents/skills/`) |
| Codebase exploration and targeted research | `explore` skill (`.agents/skills/`) |
| Execution discipline and artifact creation | `solving-tasks` workflow (`.agents/workflows/`) |
| Concrete orchestration patterns (coordinator/worker, review, research-then-act, recursion) | [references/patterns.md](./references/patterns.md) |
| `.agent.md` field semantics (tools, agents, model, handoffs, hooks) | [references/agent-frontmatter.md](./references/agent-frontmatter.md) |
| Copilot CLI `/fleet`, `/delegate`, rubber-duck details | [references/cli-fleet.md](./references/cli-fleet.md) |
| Copy-paste worker skeletons | [assets/agent-templates.md](./assets/agent-templates.md) |
