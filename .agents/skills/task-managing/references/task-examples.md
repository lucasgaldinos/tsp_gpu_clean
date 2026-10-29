# Task Examples — Bad vs Good

Load this file when writing or reviewing tasks in a plan, or when a task's `What`/`Why` read like gibberish and you need a concrete counter-example. Every example is a self-contained case: a bad task, a good rewrite, and the specific rules each one exercises (task-managing SKILL.md [§2.1](../SKILL.md#21-task-entry), [§2.3](../SKILL.md#23-task-sizing), [§8](../SKILL.md#8-task-quality-gate)). The domain is fictional (an order/billing service and a database migration) so the principles transfer to any codebase — a cold reader with no repo access must be able to act on each good task.

## Domain glossary

Terms used across the examples. A task is only well-defined when these need no further lookup.

| Term | Meaning |
| --- | --- |
| **v1 vs v2 schema** | v1 stored every record type in one wide table with nullable columns (nothing enforceable); v2 splits into dedicated per-type tables that can assert their own invariants. A **v1-behavior assertion** is a test pinned to the old shape. |
| **Fixture DB** | a small test database built from sample data into a temp location; CI runs against it so no test depends on production. |
| **Gate suite** | the committed test set that asserts the DB's invariants (per-type counts, NOT NULL, foreign keys, known values). |
| **Known-values table** | a curated reference of expected outputs (e.g. published order totals) used as a verification oracle. |
| **Real bug vs v1-behavior** | a failing test either reveals a defect in current code (fix `src/`) or asserts the pre-migration shape (rewrite the test). Never "make it pass" blindly. |
| **Given–When–Then** | acceptance-criteria syntax: Given (precondition), When (action), Then (expected outcome). One behavior per scenario. |
| **Red–green–refactor** | the TDD loop: write a failing test (red), implement the minimum to pass (green), refactor under the test's safety net. |
| **Subtasks** | the concrete step checklist a well-defined task decomposes into — the `How` made explicit. |

## Rules exercised

| Rule | Skill section | The failure it prevents |
| --- | --- | --- |
| Clarity over brevity | [§2.1](../SKILL.md#21-task-entry) | "verbose but not clear" — jargon and line numbers that no human or LLM can act on |
| Sizing (one verb, one task) | [§2.3](../SKILL.md#23-task-sizing) | one task bundling many operations, or High/Very High effort not decomposed |
| `Options` required for decisions | [§2.1](../SKILL.md#21-task-entry), [§8](../SKILL.md#8-task-quality-gate) | a hidden decision gets silently resolved by the implementer's guess, not the user |
| Self-containment | [§2.1](../SKILL.md#21-task-entry) | implied context a cold reader cannot act on |
| TDD acceptance criteria | [§8](../SKILL.md#8-task-quality-gate) | prose DoDs that cannot be checked |

---

## Example 1 — Clarity: the "scrape the notes field" case

### Context

The order service persists every order in an `orders` table. The `orders.total` column should hold the order's monetary total. Today 19 of 43 orders have `total = NULL` because the only source was a best-effort scrape of the free-text `notes` field — which is unreliable: 23 orders happen to carry a parenthesized number in their notes, 18 do not (e.g. `notes: "paid by card"` has no number at all). The total is deterministically derivable from the `line_items` table (unit price × quantity), so gating on whether a number happened to be typed into the notes is backwards.

### The bad task

```markdown
- [ ] **Task: orders.total backfill policy — decide and implement**
  - **What:** Resolve the 19/43 NULL-total orders (scraped best-effort from notes; e.g. order 7542 has no number).
  - **Why:** Finance reports need totals; "scrape a note" was never a data source.
  - **Who:** Agent (implementer); decision owner: user.
  - **Where:** order_service/orders.py L342 or a new order_service/utils/total.py.
  - **When:** After 1.2.
  - **How:** Per the chosen option; the gate suite gains the corresponding assertion.
  - **How much:** Low–Medium.
  - **DoD:** Decision recorded in this plan; if implemented, zero NULL totals for orders whose line items are complete.
```

> [!WARNING] What is wrong
>
> - **The `What` bakes in the bad policy.** "scraped best-effort from notes" is an implementation choice presented as fact — and it is the wrong one. No data reality is stated: notes is a human annotation; 23 orders carry a parenthesized number while 18 don't; `notes: "paid by card"` has no number at all.
> - **The `How` is empty.** "Per the chosen option" names no option — and there is no `Options` field, so "the chosen option" does not exist.
> - **Jargon and line numbers in `What`/`Why`** ("L342") — implementation detail that belongs in `How`/`Where`.
> - **Result:** a human cannot restate the task's intention; the LLM assumed the scrape policy instead of asking.

### The good task

```markdown
- [ ] **Task: orders.total derivation policy — decide and implement**
  - **What:** Give `orders.total` a real value for every order. Today 19/43 are NULL because the only source was a best-effort scrape of the free-text `notes` field — which is unreliable: 23 orders carry a parenthesized number, 18 do not (e.g. `notes: "paid by card"` has no number at all).
  - **Why:** Finance reports need totals; `notes` is a human annotation, not a data source. The total is deterministically derivable from `line_items` (unit price × quantity), so gating on whether a number happened to be typed into the notes is backwards.
  - **Who:** Agent (implementer); decision owner: user.
  - **Where:** `order_service/orders.py` (`compute_total`, `parse_notes`), `order_service/queries.py`.
  - **When:** After the `line_items` schema change.
  - **How:** Implement the chosen option; delete the notes-scrape path entirely; the gate suite gains the corresponding assertions.
  - **How much:** Low–Medium.
  - **Stakeholder Implications:** Finance reports; the known-totals table becomes a test fixture, not a runtime source.
  - **DoD:** `orders.total` always derived from `line_items` (never `notes`); zero NULL where line items are complete; verified by gate test.
  - **Options:** A — always compute from `line_items` (**recommended**: one runtime path, no divergence); B — curated known-totals table as a runtime source (a second path that can silently disagree); C — both.
  - **Subtasks:**
    - [ ] 1. Write the failing gate assertions (Given an order with line items, When `compute_total` runs, Then total == sum(price × qty)).
    - [ ] 2. Implement `compute_total` from `line_items`; drop `parse_notes`.
    - [ ] 3. Backfill the 19 NULL rows; re-run the gate suite.
```

> [!NOTE] Why this is good
>
> - The `What` states the deliverable **and the data reality** (23 vs 18, the "paid by card" counterexample) in plain terms.
> - The `Why` states the root cause plainly — "gating on whether a number happened to be typed into the notes is backwards" — no jargon.
> - The decision is **surfaced, not assumed**: `Options` A/B/C with a recommendation and the reason (one runtime path). Option A keeps a single source of truth; the known-totals table is demoted to a test oracle, so it can never silently diverge from compute.
> - No line numbers in `What`/`Why`; the line numbers live in `Where`.
> - **Subtasks** make the execution concrete and reviewable before work starts.

### Tests (TDD acceptance criteria)

```markdown
Given an order with line items 2×10.0 and 1×5.0
When  compute_total(order) runs
Then  order.total == 25.0        # no notes field is consulted

Given order 7542 with a published total of 7542.0
When  the gate suite runs
Then  order_7542.total == 7542.0 # the known-totals table is the test oracle
```

---

## Example 2 — Sizing: the "migrate + backup + delete" case

### Context

The orders database `orders.db` (v1 schema) must be migrated to the v2 schema. A stale export copy also exists in a second location and should be removed. These are three separate operations with different failure modes and different rollback stories — they must not share one task and one DoD.

### The bad task

```markdown
- [ ] **Task: Migrate the database + back it up + delete the stale export**
  - **What:** Rebuild `orders.db` on the v2 schema, copy it to the archive path, and delete the stale export.
  - **Why:** The DB is stale and the export is dead weight.
  - **How much:** High.
  - **DoD:** Migrated DB exists and is > 12 KB; stale export deleted.
```

> [!WARNING] What is wrong
>
> - **Three operations, one task** — migrate, copy, delete are independent concerns with different failure modes and rollback stories. One DoD cannot certify all three.
> - **A file-size DoD** (`> 12 KB`) certifies nothing about correctness — the gate that later ran proved the migrated DB was wrong.
> - `How much: High` is the [§2.3](../SKILL.md#23-task-sizing) split trigger, and it was not split.

### The good task (batch header + four tasks, with subtasks)

```markdown
## Batch 7 — DB migration — `implementer` — sequential (7.1 → 7.2 → 7.3 → 7.4)

- [ ] **Task 7.1: Run the v2 migration to scratch output**
  - **What:** Full migration into a throwaway output file (`/tmp/orders_v2.db`) — never directly over the canonical `orders.db`.
  - **Why:** Migrating in place couples a failed run to a destroyed canonical artifact; scratch-then-promote keeps the failure atomic.
  - **How much:** Low.
  - **DoD:** CLI reports 0 failures; the scratch DB exists. (Byte-counts are not a DoD — the gate in 7.2 is.)

- [ ] **Task 7.2: Validate the scratch DB with the committed gate suite**
  - **What:** Run the gate suite against `/tmp/orders_v2.db` (the canonical DB is untouched).
  - **Why:** The gate is the DoD — validate BEFORE promote, so the canonical file is only ever replaced by a verified artifact.
  - **How much:** Low.
  - **DoD:** All gate assertions green against the scratch DB.

- [ ] **Task 7.3: Atomically promote the scratch DB to canonical `orders.db`**
  - **What:** Rename `/tmp/orders_v2.db` → `orders.db` (same filesystem, atomic); keep `orders.db.v1-backup` until the smoke check.
  - **Why:** Rename has no partial-write window; the backup gives rollback until the smoke check.
  - **How much:** Trivial.
  - **Subtasks:**
    - [ ] 1. `mv orders.db orders.db.v1-backup`
    - [ ] 2. `mv /tmp/orders_v2.db orders.db`
  - **DoD:** Canonical path opens read-only; row count > 0.

- [ ] **Task 7.4: Smoke check + delete the backup**
  - **What:** Run a consumer smoke check against the canonical DB, then delete the v1 backup.
  - **Why:** The destructive step is gated by the smoke check — never a silent side effect.
  - **How much:** Trivial.
  - **DoD:** Smoke check passes; `orders.db.v1-backup` removed.
```

> [!NOTE] Why this is good
>
> - **Four verbs → four tasks**, each with its own `DoD` and failure mode; the batch header shows the task hierarchy and ordering.
> - **Validation precedes promotion** — the canonical file is only ever replaced by a *verified* artifact; a failed gate never touches it.
> - **Atomic rename** replaces copy, so there is no partial-write window on the canonical path.
> - The destructive step (backup deletion) is a **separate, gated** task, not a side effect.
> - `Subtasks` make the two-step promote explicit and reviewable.

### Tests (TDD acceptance criteria)

```markdown
Given the gate suite and a scratch DB at /tmp/orders_v2.db
When  Task 7.2 runs
Then  all per-type counts and invariants pass, and orders.db is byte-identical

Given a post-promote smoke request
When  a consumer loads order 7542
Then  it resolves without error
```

---

## Example 3 — Sizing: the "triage 18 tests" case

### Context

After the v1→v2 schema migration, 18 committed tests fail across five test files. Each failure must be classified as a **real bug** (fix in `src/`) or a **v1-behavior assertion** (rewrite the test against v2). Classifying and fixing inside the same task lets the implementer rubber-stamp its own verdicts — so classification is its own, read-only, reviewable deliverable first.

### The bad task

```markdown
- [ ] **Task: Triage the 18 red tests**
  - **What:** For each of the 18 failing tests (list inlined below), classify as real bug or v1-behavior assertion and resolve. Suspected real bugs: the scanner-file failures. Suspected v1-behavior: the key-set, migration, API-structure assertions. [18 test names inlined]
  - **How much:** Medium–High.
  - **DoD:** Every one of the 18 has a recorded classification and resolution; suite green.
```

> [!WARNING] What is wrong
>
> - **18 decisions inlined as one "task"** — the `What` is a wall of test names, the DoD is unverifiable, and one dispatch holds contexts from five different files.
> - **"Triage" hides the real work** — each failure is a distinct classification with its own fix target; one task title cannot name that.
> - **The same hand classifies and fixes** — nothing stops the implementer from declaring its own fixes "v1-behavior" and rewriting the test instead of fixing the code.

### The good task (classify first, then resolve)

```markdown
## Batch 3 — Test triage — `implementer`

Ordering: 3.1 first (produces the matrix), then 3.2–3.6 per owning file. Classification discipline: real bug → fix `src/`; v1-behavior → rewrite the test. Never a blind make-it-pass.

- [ ] **Task 3.1: Produce the failure-classification matrix (read-only)**
  - **What:** For the 18 failures across the 5 files, record test → verdict (real-bug / v1-behavior) → owning file → fix target. Output: the matrix in the Solution block. No code changes.
  - **Why:** Verdicts must be reviewable and approved before any fix, so the same implementer cannot rubber-stamp its own classification.
  - **Who:** Agent (implementer).
  - **Where:** `tests/checkout/` (the 5 failing files).
  - **How:** Run each failing test; read the assertion against the v2 implementation; classify; do not fix.
  - **How much:** Low.
  - **DoD:** Matrix covers all 18 with verdicts and owners; no `src/` or `tests/` edits made.

- [ ] **Task 3.2: Resolve the scanner-file failures (6) per the approved matrix**
  - **What:** Apply the 3.1 verdicts for `tests/test_scanner.py` — fix `src/scanner.py` for real bugs, rewrite the test for v1-behavior assertions.
  - **Why:** The scanner feeds all ingestion — a real bug here poisons every pipeline run.
  - **Who:** Agent (implementer).
  - **Where:** `tests/test_scanner.py`; fix target `src/scanner.py`.
  - **When:** After 3.1 is approved.
  - **How much:** Medium.
  - **DoD:** Each fix matches its matrix verdict; `test_scanner.py` is green; any `src/` fix is covered by a rewritten or new test.
```

> [!NOTE] Why this is good
>
> - **Classification is a first-class, reviewable artifact** (Task 3.1, read-only) — the verdicts can be checked before any code moves.
> - **Resolution tasks consume the approved matrix** — the implementer can no longer declare its own bug a "v1-behavior assertion" and rewrite the test instead of fixing the code.
> - One file, one task: each resolution task names its owning file, fix target, and verifiable DoD.

### Tests (TDD acceptance criteria)

```markdown
Given the approved classification matrix (Task 3.1's Solution block)
When  Task 3.2 fixes the 6 scanner-file failures
Then  each fix matches its matrix verdict, and test_scanner.py is green

Given a fix classified as "real bug" in the matrix
When  the resolution task runs
Then  the fix lives in src/, and the test is either kept or strengthened — not weakened
```

---

## Example 4 — Hidden decision: `Options` or nothing

### Context

The order service's CLI writes exports to a path that the config file's `export_path` setting does not actually control — the CLI derives the path from its `-o` flag instead. Someone must decide how the export path is resolved. This is a genuine decision with tradeoffs, so it must be **presented** to the user, not guessed.

### The bad task

```markdown
- [ ] **Task: Choose the export path strategy**
  - **Who:** Agent (implementer); decision owner: user.
  - **How:** Per the chosen option.
```

> [!WARNING] What is wrong
>
> - "decision owner: user" with **no options to choose from** is a contradiction — the user cannot decide what was never presented, and the implementer has to invent the option set.

### The good task

```markdown
- [ ] **Task: Choose the export path strategy**
  - **What:** Decide how the CLI resolves its export path, given that `export_path` in config is currently ignored by the `-o` derivation.
  - **Why:** Flag-less runs currently write to a path nobody configured; the choice affects every user of the export feature.
  - **Who:** Agent (implementer); decision owner: user.
  - **Where:** `order_service/cli.py` (export command), `order_service/config.py`.
  - **When:** Before the export feature ships.
  - **How:** Implement the option the user chooses; record the decision in the Solution block.
  - **How much:** Low.
  - **Stakeholder Implications:** Anyone running flag-less exports.
  - **DoD:** The chosen resolution is implemented, and the decision is recorded in the Solution block.
  - **Options:** A — make `export` honor `export_path`; B — keep `-o` derivation and document it (**recommended**: smallest surface, matches existing behavior); C — both, with `export_path` as the override.
```

> [!NOTE] Why this is good
>
> - The decision is real and **presented** — concrete alternatives with a recommendation and the reason, instead of the implementer guessing.
> - The 5W2H is **complete** (Where / When / How / How much / Stakeholder Implications / DoD), so the task passes the [§8](../SKILL.md#8-task-quality-gate) gate.
> - The DoD requires the decision to be **recorded** in the Solution block — the choice becomes traceable history.

### Tests (TDD acceptance criteria)

```markdown
Given the chosen option is B (keep -o, document it)
When  `export --help` runs
Then  help text states that export_path is not consulted and shows how to target the canonical path

Given the chosen option is A (honor export_path)
When  a flag-less export runs
Then  output lands under the configured export_path
```

---

## Example 5 — TDD: red → green → refactor

### Context

A new rule: an order must be rejected when stock is insufficient. This example shows a task written to **demand TDD** — acceptance criteria first as failing tests, then minimal implementation, then refactor under the safety net.

### The task (TDD-shaped)

```markdown
- [ ] **Task: Reject orders when stock is insufficient**
  - **What:** Make order placement fail with `StockInsufficientError` when any line item's quantity exceeds available stock; no order row is written.
  - **Why:** Overselling ships orders that cannot be fulfilled; today stock is checked only at fulfillment time.
  - **Who:** Agent (implementer).
  - **Where:** `order_service/stock.py` (`reserve`), `order_service/orders.py` (`place_order`).
  - **How much:** Low.
  - **DoD:** Red → green → refactor: the acceptance tests below fail first, pass after the minimal implementation, and stay green after refactor.
  - **Subtasks:**
    - [ ] 1. Write the two acceptance tests below (red — `StockInsufficientError` does not exist yet).
    - [ ] 2. Add a minimal stock check in `place_order` (green).
    - [ ] 3. Extract the check into `stock.reserve` and keep the suite green (refactor).
  - **Tests (acceptance criteria):**

    ```text
    Given stock for SKU-A is 3 and the order asks for 4 of SKU-A
    When  place_order runs
    Then  StockInsufficientError is raised and no order row is written

    Given stock for SKU-A is 3 and the order asks for 3 of SKU-A
    When  place_order runs
    Then  the order is accepted and stock is reserved
    ```
```

> [!NOTE] Why this is good
>
> - **Acceptance criteria are executable** — Given-When-Then, one behavior per scenario, and the DoD explicitly demands the **red→green→refactor** loop (tests fail first), not "implement it".
> - The **Subtasks** encode the TDD phases, so the implementer cannot skip writing the failing test.
> - Small enough to finish, leaves room for the implementer's design (where the check lives) while pinning the behavior.

### The red → green → refactor execution

```markdown
Red:      write the two acceptance tests; run → both fail (StockInsufficientError doesn't exist yet).
Green:    add a minimal stock check in place_order that raises StockInsufficientError; run → both pass.
Refactor: extract the check into stock.reserve; run → still green.
```

> [!TIP]
> If the implementer's first move is to write production code, the task is not being executed as specified — the DoD's red-first clause is the contract.
