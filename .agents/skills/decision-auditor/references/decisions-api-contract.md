---
title: "Decisions API contract — verified facts for decision-auditor"
description: >
  Ground-truth request/response schema, hard limits, answer semantics, and error shapes for
  Gemini Decision 1 (`gemini-decision-1`), captured from the official AI Studio
  documentation so the skill never re-derives them from memory.
source: https://aistudio.google.com/docs/decisions-api
model_page: https://aistudio.google.com/docs/models/gemini-decision-1
captured: 2026-11-14
tags:
  - guide/api
  - notes/reference
  - analysis/verification
---

# Decisions API contract

Everything below was read from the official pages cited in the front matter. Anything not
stated there is marked **unverified** and must not be quoted as fact.

## Endpoint and client

```python
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})
result = client.decisions.create(model="gemini-decision-1", input=..., questions=...)
```

- SDK: `google-genai>=3.2.0` (`pip install -U google-genai`).
- REST: `POST https://generativelanguage.googleapis.com/v1beta/decisions`, auth header
  `x-goog-api-key`.
- Available on the Gemini Developer API (`v1beta`) with `gemini-decision-1` only.

## Request

| Field | Type | Required | Notes |
| :-- | :-- | :-- | :-- |
| `model` | string | yes | `gemini-decision-1` |
| `input` | string, `Content`, or `Content[]` | yes | Shared context. Text, image, audio, video, PDF. |
| `questions` | map<string, Question> | yes | 1 to 1,000 entries. |

Each `Question`:

| Field | Type | Required | Notes |
| :-- | :-- | :-- | :-- |
| `type` | `noul` \| `choice` \| `score` | yes | Answer type mirrors this. |
| `instructions` | string or text content | yes | **Text only.** The full question goes here. |
| `criteria` | object or array | `choice`/`score` yes, `noul` no | See below. |

- `noul` criteria (optional): `{"true_description": ..., "false_description": ...}`.
- `choice` criteria: map `option_id -> description`; `""` allowed when the id is
  self-explanatory. 2 to 255 options.
- `score` criteria: array of 2 to 255 non-empty strings ordered **lowest to highest**
  (2 to 10 recommended), indexed from 0.

## Response

```json
{
  "answers": {
    "is_damaged": { "type": "noul", "noul": 0.99, "confidence": 0.99 },
    "route": { "type": "choice", "choice": "billing",
               "probabilities": { "billing": 0.99, "other": 0.01 }, "confidence": 0.99 },
    "severity": { "type": "score", "score": 1.12,
                  "legend": { "0": "Cosmetic", "1": "Workaround", "2": "Blocked" },
                  "probabilities": { "0": 0.03, "1": 0.83, "2": 0.15 }, "confidence": 0.83 }
  },
  "usage": {
    "total_input_tokens": 324,
    "input_tokens_by_modality": [{ "modality": "text", "tokens": 324 }],
    "total_output_tokens": 0,
    "total_tokens": 324
  }
}
```

Answer semantics:

- `noul` is the calibrated probability of **yes**, always in `0.01..0.99`. The probability
  of no is `1 - noul`.
- `choice.probabilities` sums to 1 across the supplied options even when none fits.
- `score.score` is the probability-weighted mean of level indices, so it can sit between
  levels; `legend` maps index to description.
- `confidence` is the estimated probability that the top answer is correct: the top
  outcome's probability for `choice`/`score`, and `max(noul, 1 - noul)` for `noul`.

## Hard limits

| Limit | Value |
| :-- | :-- |
| Questions per request | 1 to 1,000 |
| Context window | 262,144 tokens across shared input and all questions |
| Question key | 1 to 64 characters matching `[A-Za-z0-9_-]+` |
| Options per `choice` | 2 to 255 |
| Levels per `score` | 2 to 255 non-empty strings |
| Option key | 1 to 128 chars (≤512 UTF-8 bytes), internal ASCII spaces allowed |
| JSON nesting depth | 100 |
| Inline media | 100 MB per request, 50 MB for PDFs |
| External URL media | 15 MB |

Exceeding a limit or failing key validation returns HTTP 400 naming the rule.

## Behaviour that changes how questions are written

- Questions are evaluated **independently**; they cannot see each other, and adding or
  removing one does not change the others. Dependent decisions need a second request or an
  ignored branch.
- **The model never sees question keys**, so the question must be fully written in
  `instructions`.
- Choice option keys are sorted alphabetically in UTF-8 byte order before evaluation, and
  exact probability ties break to the alphabetically first key. Avoid positional option
  names such as `none_of_the_above`; use `other`.
- Positive phrasing is recommended: ask "Is the tier Gold?" and invert in code rather than
  asking negated questions.
- Score scales are index-averaged, so keep them linear; exponential scales skew the mean.

## Errors

Safety blocks return HTTP 400 with an `error` object carrying `message` and a `code` of
`safety`, `prohibited_content`, `blocklist`, `image_safety`, or `content_blocked`.

## Unsupported

Context caching, Batch API, Flex/Priority inference, conversation-history/thought/tool-call
input blocks, YouTube URLs, and `gs://` URIs are all unavailable. Repeated input across
requests is counted in full — which is why batching beyond 1,000 questions rebills the
shared input once per batch.

## Unverified

- A specific per-request latency figure. The docs only state that latency "stays nearly
  flat" as questions are added. Measure before quoting milliseconds.
- Pricing numbers. Only "input tokens are billed; output is zero" is safe to state without
  re-reading the pricing page.
