# Copilot CLI — Fleet, Delegation, and Multi-Model Tooling

Load this file when the user runs Copilot CLI orchestration (`/fleet`, `/delegate`), needs prompt-mode caveats, or asks how CLI/cloud-agent behavior differs from VS Code subagents. VS Code subagent mechanics stay in the main skill and `patterns.md`.

## `/fleet` — the CLI orchestrator

Fleet decomposes a larger objective into work items, launches independent ones as parallel background subagents, respects dependencies, then synthesizes the final result.

```bash
# In an interactive session
/fleet Update the auth docs, refactor the auth service, and add related tests.

# Non-interactive
copilot -p "/fleet Update the auth docs, refactor the auth service, and add related tests." --no-ask-user
```

Behavior differences from a single chat turn:

- the orchestrator plans work items first;
- independent tasks run in parallel, each subagent in its own context window;
- subagents share one filesystem — overlapping writes must be avoided (same partitioning rule as VS Code);
- only the synthesized result reaches the user.

**Prompt-mode repo hooks caveat (v1.0.40+):** with `copilot -p "..."`, repository hooks are disabled by default for security. If the `/fleet` workflow relies on hooks (auto-format, lint after edits), opt in:

```bash
GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS=true copilot -p "/fleet ..." --no-ask-user
```

## `/delegate` — hand off to the Copilot cloud agent

For complex asynchronous tasks, `/delegate` sends work to a Copilot cloud agent running on GitHub's servers: it creates a branch (prefixed `copilot/`), makes file changes, runs tests, and opens a draft pull request for human review.

Use it when the task should leave the local machine entirely and come back as a reviewable PR. Community-documented; verify current behavior against the Copilot CLI docs before relying on it in a scripted pipeline.

## Rubber-duck agent (experimental)

Available under `/experimental` (v1.0.42+). Multi-model second perspective: in a GPT-powered session, the rubber-duck agent internally routes certain requests through Claude (and vice versa) — same idea as talking a problem through with a different listener, surfacing blind spots the primary model missed.

- Configure it, including its complementary model strategy, via `/subagents` (v1.0.64+).
- Complementary strategy: automatically pick a model from a different family than the primary model.
- Runs as a sub-agent layer — the primary session's model and context are kept; the analysis happens in the background.
- Experimental; may change. Feedback via `/feedback`.

## Concurrency and depth limits

Usage-based billing users (v1.0.66+) can configure **subagent concurrency and depth limits** from `/settings`:

- **Concurrency limit** — how many subagents run in parallel.
- **Depth limit** — how many levels deep delegation can chain, preventing runaway recursive subagent trees.

Set these when orchestrating on a budget — they are the CLI-side equivalent of VS Code's `chat.subagents.allowInvocationsFromSubagents` guard.

## Platform nuance: what ports where

VS Code documents subagents and `handoffs`; GitHub's custom agent configuration reference notes that `handoffs` and `argument-hint` are currently **ignored** for the Copilot cloud agent on GitHub.com. Product-specific expectations:

| Surface | Delegation features |
| --- | --- |
| VS Code | Subagents, `agents` allowlists, handoffs, invocation control (`user-invocable`, `disable-model-invocation`) |
| Copilot CLI | `/fleet` orchestration, `/delegate`, rubber-duck |
| GitHub.com cloud agent | Custom agents run, but some VS Code-specific frontmatter is intentionally ignored |

When sharing agent files across surfaces, document which behaviors are editor-specific. Do not promise handoffs to a user whose workflow runs in the cloud agent.

## BYOK models (no paid Copilot plan)

Custom agent `model` fields accept any model available in the model picker, including BYOK providers (e.g., DeepSeek via API key). Rules:

- Names are qualified as `Model Name (vendor)` and must match the picker **exactly** — vendor suffix casing matters (e.g., `DeepSeek V4 Flash (deepseek)`, `DeepSeek V4 Pro (deepseek)`).
- A prioritized list falls through in order until an available model is found.
- The subagent cost-tier rule still applies on VS Code: a requested model cannot exceed the main model's cost tier, or the subagent refuses to run and reports alternatives.
