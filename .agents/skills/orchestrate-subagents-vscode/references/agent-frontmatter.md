# `.agent.md` Frontmatter Reference — VS Code / GitHub Copilot

Load this file when creating or editing custom agent files, or when exact field semantics are needed (allowlists, invocation control, model lists, handoffs, hooks). Compiled 2026-08-26 from the official VS Code custom-agents, subagents, hooks, and MCP docs, plus field usage observed in `github/awesome-copilot`.

## File Locations

| Scope | Location | Notes |
| --- | --- | --- |
| Workspace | `.github/agents/` | `.agent.md` files; VS Code detects any `.md` there |
| Workspace (Claude format) | `.claude/agents/` | Plain `.md`, Claude frontmatter (`name`, `description`, `tools` comma-separated, `disallowedTools`) |
| User profile | `~/.copilot/agents` | Reusable across workspaces |
| Extra locations | `chat.agentFilesLocations` setting | Add custom folders |
| Monorepo parent | — | Enable `chat.useCustomizationsInParentRepositories` |

## Core Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | file name | Agent identifier. Case-sensitive when referenced from prompts or allowlists. |
| `description` | — | Shown as placeholder text in the chat input; used by the model to decide which agent to pick. Write it so a similar-named agent can't be confused with it. |
| `argument-hint` | — | Hint text shown in the chat input for user-invocable agents. Ignored by the Copilot cloud agent on GitHub.com. |
| `tools` | inherited | Tool or tool-set names. Built-in tools (`agent`, `read`, `search`, `edit`, …), MCP tools, or `<server>/*` to include all tools of an MCP server. Unavailable tools are silently ignored. Prompt-file tools take precedence over custom-agent tools when both apply. |
| `agents` | `*` | Subagent allowlist for this agent. List of names / `*` (all) / `[]` (none). Requires the `agent` tool in `tools`. Explicitly listing an agent **overrides** that agent's `disable-model-invocation: true`. Listing yourself enables recursion (see nesting setting). |
| `model` | parent's model | Single model name or prioritized array; tried in order until available. |
| `user-invocable` | `true` | `false` hides the agent from the picker — still invocable as a subagent. |
| `disable-model-invocation` | `false` | `true` prevents other agents from invoking this agent as a subagent — still selectable by users. |
| `infer` | — | **Deprecated.** Replaced by `user-invocable` + `disable-model-invocation`. |
| `target` | — | `vscode` or `github-copilot`. |
| `mcp-servers` | — | MCP server config JSON for use with `target: github-copilot`. |
| `handoffs` | — | Sequential human-in-the-loop transitions (below). Ignored by the cloud agent. |
| `hooks` | — | Agent-scoped hooks (preview; below). |

## Invocation Control Matrix

| `user-invocable` | `disable-model-invocation` | Effect |
| --- | --- | --- |
| `true` (default) | `false` (default) | In picker **and** delegatable — general-purpose agent |
| `false` | `false` | Pure worker: subagent-only, hidden from picker |
| `true` | `true` | User-only: visible in picker, protected from subagent use |
| `false` | `true` | Effectively inert |

Two real-world workers (`github/awesome-copilot`, `gem-*` family) also carry `mode: subagent` and `hidden: true`. These are **not** documented VS Code fields — treat them as community conventions, not guarantees.

## This Repo's Agent Fleet

The orchestration system ships three agents with fixed tool conventions:

| Agent | `tools` | `agents` | Role |
| --- | --- | --- | --- |
| `RUG` | `read`, `search`, `agent`, `todo` | `["reviewer", "implementer"]` | Orchestrator — never `edit`/`execute`; never writes the plan |
| `reviewer` | `read`, `search`, `edit` | — | Review + plan + register; `edit` scoped to the plan file only |
| `implementer` | `read`, `edit`, `execute`, `search` | — | Execute batched tasks; update state via `solving-tasks` |

Two non-obvious conventions worth preserving:

- **The orchestrator has no `edit`/`execute`.** Its `agents` allowlist is how it reaches workers; granting write tools invites it to do work — and, as seen in practice, to edit the plan itself instead of delegating registration to the reviewer.
- **The reviewer's `edit` is scoped by instruction, not by tool.** VS Code has no per-file `edit` scope, so the reviewer's frontmatter grants `edit` but its body restricts it to the plan file; source stays read-only.
- **A worker must declare its own `tools`.** A subagent without an explicit `tools` field inherits the parent's tools — and RUG's tools are read-only. The implementer that omits `tools` cannot edit files or run commands; that is exactly the bug that surfaced in the first real run.

## Handoffs

Guided sequential workflows: after a response completes, buttons let the user move to the next agent with context and a pre-filled prompt. Human-in-the-loop by design.

```markdown
---
description: Generate an implementation plan
tools: ['search', 'web']
handoffs:
  - label: Start Implementation
    agent: implementation
    prompt: Now implement the plan outlined above.
    send: false
    model: DeepSeek V4 Flash (deepseek)
---
```

| Field | Meaning |
| --- | --- |
| `handoffs.label` | Button text |
| `handoffs.agent` | Target agent identifier |
| `handoffs.prompt` | Prompt text to send |
| `handoffs.send` | Auto-submit the prompt (`false` default) |
| `handoffs.model` | Model for the handoff execution, qualified format `Model Name (vendor)` |

Classic chains: Review+plan → Implementation; Implementation → Final review; Write failing tests → Write passing tests.

## Model Selection for Subagents

Priority order when a subagent runs:

1. Explicit model requested in the `runSubagent` invocation.
2. The worker's frontmatter `model` (string or prioritized list).
3. The parent conversation's model.

Hard limit: the requested model cannot exceed the main model's cost tier. If it does, the subagent does not run and reports which models are available.

## Agent-Scoped Hooks (Preview)

`hooks` in `.agent.md` frontmatter runs only while that agent is active — user-selected **or invoked as a subagent**. Runs in addition to workspace/user hooks. Requires `chat.useCustomAgentHooks: true`.

```markdown
---
name: "Strict Formatter"
description: "Agent that auto-formats code after every edit"
hooks:
  PostToolUse:
    - type: command
      command: "./scripts/format-changed-files.sh"
---
```

Lifecycle events include `SubagentStart` / `SubagentStop` — use them to track or gate delegation (e.g., aggregate results, initialize subagent resources). Hook config files live in `.github/hooks/*.json` (workspace) or `~/.copilot/hooks` (user). Hooks receive JSON on stdin and control the agent via stdout (`continue`, `stopReason`, `systemMessage`; exit code 2 blocks).

## MCP Tools in Agents

- Workspace servers: `.vscode/mcp.json`; user servers via `MCP: Open User Configuration`.
- In agent frontmatter, `tools: ['github/*']` includes every tool of the `github` MCP server.
- Each perspective in a multi-perspective review can own different MCP tooling — that's the upgrade path from prompt-shaped lenses to dedicated agents.

## Real-World Examples (github/awesome-copilot)

Delegation-related agent files worth inspecting before writing your own:

- `agents/context7.agent.md` — VS Code-style `handoffs`: a handoff button that passes work to another agent after research completes.
- `agents/rug-orchestrator.agent.md` — coordinator example: enables the `agent` tool and restricts delegation with `agents: ['SWE', 'QA']`.
- `agents/gem-orchestrator.agent.md` — invocation control with `user-invocable` and `disable-model-invocation`: decides whether an orchestrator is directly selectable, delegatable, or both.
- `agents/custom-agent-foundry.agent.md` — documents the VS Code `handoffs` shape in its guidance section; a template before creating your own coordinator workflow.

## Claude-Format Agents (`.claude/agents/*.md`)

| Field | Meaning |
| --- | --- |
| `name` | Required |
| `description` | What the agent does |
| `tools` | Comma-separated Claude tool names (e.g., `Read`, `Grep`, `Glob`) — VS Code maps them to its equivalents |
| `disallowedTools` | Comma-separated blocklist |

VS Code supports both formats, so the same agent definitions can be shared with Claude Code.
