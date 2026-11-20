---
title: "Question packs — deriving decision-auditor questions from a review framework"
description: >
  How to turn an existing review framework (for example the five-axis
  code-review-and-quality skill) into atomic decision-model questions, which checks belong
  to the decision model and which belong to the agent, and the concrete mappings already
  validated.
tags:
  - guide/api
  - analysis/planning
  - review/quality
---

# Question packs

The harness is generic: it assembles a shared input, expands questions per symbol, batches
them, and reports which symbols failed. A **question pack** is the part that changes per
review framework. This file shows how to derive one.

## The division of labour

| Layer | Who does it | Why |
| :-- | :-- | :-- |
| Reading the review framework | The agent | Needs judgement about which checks matter for this change |
| Authoring atomic questions | The agent | Requires translating prose criteria into decidable predicates |
| Screening every symbol | `gemini-decision-1` | 1,000 independent checks in one pass, zero output tokens |
| Deep review of the shortlist | The agent | Requires reasoning, generation, and reading code in full |

The decision model never replaces the review. It answers "which of these 200 symbols are
worth reading for this specific criterion", and the agent reviews only that shortlist.

## Deriving a pack: one bullet becomes one question

Most review frameworks already contain checkable predicates. Read each bullet and ask: *is
this decidable from the symbol's text alone?*

- **Yes** → one `noul` question with `per_symbol: true`, and criteria that draw the yes/no
  boundary in the framework's own words.
- **No, it needs tracing state** → the model cannot do it in one pass. Keep it for the
  agent, or ask it and expect the answer to land mid-band and escalate.
- **No, it is a fact the AST already proves** → compute it in code; do not spend a question.

## Mapping from `code-review-and-quality`

The five axes in [code-review-and-quality](../../code-review-and-quality/SKILL.md) map as
follows. Each row is a real bullet from that skill.

| Axis / bullet | Atomic question | `per_symbol` | Expectation |
| :-- | :-- | :-- | :-- |
| Correctness — *No silent data evaporation* | Does this symbol drop, filter, or skip records without logging or raising? | yes | Validated: flagged 4 real findings across 6 modules |
| Correctness — *Spec & mathematical parity* | Does this symbol round or index differently from the rule its docstring states? | yes | Decision-shaped, but the criterion must quote the rule |
| Correctness — *Whole-module sibling sweep* | Does this symbol have a sibling in the module that applies the same fix? | yes | Works: the module is inside the shared input |
| Readability — *comments explain WHY* | Do this symbol's comments narrate syntax instead of stating a domain invariant? | yes | Decision-shaped |
| Readability — *dead code artifacts* | Does this symbol contain a compatibility alias, a no-op variable, or a `// removed` comment? | yes | Decision-shaped |
| Readability — *repeated conditionals* | Does this symbol repeat a conditional on the same shape that belongs in a dispatcher? | yes | Decision-shaped |
| Architecture — dependency direction | Does this symbol import from a layer above its own? | yes | Deterministic in code; prefer an import check |
| Security — injection surface | Does this symbol interpolate untrusted input into a query, path, or shell command? | yes | Decision-shaped |
| Performance — allocation in a loop | Does this symbol allocate per iteration over a collection? | yes | Decision-shaped |
| GPU reference §1 — sync placement | Is a `cuda.syncthreads()` missing after this write to shared memory? | yes | Simple case decided; divergent-index case escalates |

Two rules follow from this table:

1. **A bullet that names a mechanism is decidable** ("does it drop records without logging",
   "does it import from above"). A bullet that names an outcome is not ("does it match the
   spec") — split it until it names a mechanism.
2. **Deterministic checks stay in code.** If `ruff` or an import graph can answer it, a
   question only adds latency and cost.

## The DRY pack, as a worked example

Duplication is one pack among many. It differs from the table above because the *candidate*
can be generated deterministically before any model call:

```bash
<skill-dir>/scripts/verify_claims.py src/pipeline.py --candidates
<skill-dir>/scripts/verify_claims.py src/pipeline.py --emit-questions duplication.json
```

`--candidates` clusters symbols on an identifier-erased AST shape (exact) and a structural
5-gram Jaccard similarity (`--similarity`, default 0.60). Over-generation is intentional:
each candidate costs one question, and the model filters.

The emitted pack asks two independent calibrated questions per candidate, because a forced
choice cannot say "neither":

| Question | Type | Markers |
| :-- | :-- | :-- |
| Is this shared shape accidental duplication? | `noul` | `flag_at: 0.30`, `basis` required |
| Is that shape required by a protocol or the language? | `noul` | `informational: true` |

Measured on two known duplicates and two known non-duplicates, the yes/no pair separated
them with a 0.32 margin, against 0.23 for a three-way `choice` and 0.18 for an ordered
`score`. The choice format also tripled the reading on the known non-duplicate, because its
probabilities must sum to one even when no option fits.

## Validation before trusting a pack

1. **Lint it**: `--strict-questions` rejects pairwise phrasing, negation, missing criteria,
   and duplicate instructions.
2. **Canary it**: `--canary` with a file of deliberate violations. A question that does not
   fire on its canary is proven blind; a question with no canary answers is *untested*, not
   passing.
3. **Check the signal**: a `flat-clean` verdict in `question_summary` means every answer
   landed at or below 0.20 — indistinguishable from a blind question without a known
   positive.
4. **Calibrate once labels exist**: `--calibrate labels.json` sweeps the threshold and
   reports precision/recall.
