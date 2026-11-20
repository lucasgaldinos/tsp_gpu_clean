---
title: "Adding a strategy without desynchronizing the config schema"
description: >
  Operational note on the fail-loud loop for registering a new strategy or pipeline
  step (for example a real 3-opt): implement the class, register it, let the schema
  drift test fail, regenerate `configs/schemas/algorithms.schema.json` with
  `schema_generator`, and commit class plus schema in the same change.
created: 2026-11-16
status: review
author:
  - "Lucas Galdino"
type: notes
scope: global
modifications:
  - date_modified: 2026-11-17
    modifications:
      - description: >
          Added the missing YAML frontmatter and expanded the three-line copy-paste
          fragment into a complete note: the silent failure mode that motivates the
          loop, the ordered workflow, the exact commands, and the tests that enforce it.
related_files:
  - "[schema_generator.py](../../../src/benchmarking/runner/schema_generator.py)"
  - "[algorithms/__init__.py](../../../src/algorithms/strategies/__init__.py)"
  - "[algorithms.schema.json](../../../configs/schemas/algorithms.schema.json)"
  - "[test_unit_algorithms_config_contract.py](../../../tests/benchmarking/test_unit_algorithms_config_contract.py)"
tags:
  - guide/configuration
  - guide/refactoring
  - analysis/architecture
  - guide/testing
  - notes/planning
  - review/quality
  - algorithm/local-search
  - notes/session
  - guide/benchmarking
  - analysis/statistics
---

# Adding a strategy without desynchronizing the config schema

`configs/schemas/algorithms.schema.json` is **generated**, not hand-written: `schema_generator` reflects over `STRATEGY_REGISTRY` and `PIPELINE_STEP_REGISTRY`, deriving parameter keys from `inspect.signature` and `mode`-style enums from `typing.get_args` of each constructor's `Literal`.
Because of that, the only correct way to extend the configuration surface is to change the code and let the generator follow — never to edit the JSON by hand.

The failure mode this protects against is silent: a strategy missing from the schema gets its `params` left **unrestricted** instead of rejected, so the schema keeps validating whatever the constructor would refuse.
This is the class of defect tracked as `BUG-85` and closed by the reflection + drift tests of `TASK 13.27`.

## Workflow

1. Implement the class (inherit `StrategyBase` — it gives `seed`, `rng`, and the `h2d_bytes`/`d2h_bytes`/`kernel_launches` counters; GPU variants also implement `warmup(context)`).
2. Register it in `STRATEGY_REGISTRY` ([`algorithms/__init__.py`](../../../src/algorithms/strategies/__init__.py)), or in `PIPELINE_STEP_REGISTRY` if it is a `GenerationStep`.
3. Run the contract tests. `test_algorithms_schema_matches_generator_and_registries_drift_contract` now fails: the committed schema has no `allOf` branch for the new class.
4. Regenerate the schema from the registries.
5. Re-run the suite and commit the class **and** the regenerated schema together.

```bash
# Step 3 — the drift gate fails while the committed schema is stale
.venv/bin/python -m benchmarking.runner.schema_generator --check
# -> "Schema drift detected in configs/schemas/algorithms.schema.json" (exit 1)

# Step 4 — write the regenerated schema
.venv/bin/python -m benchmarking.runner.schema_generator
# -> configs/schemas/algorithms.schema.json rewritten with one new allOf branch

# Step 5 — verify the whole loop
.venv/bin/python -m pytest tests/benchmarking/test_unit_algorithms_config_contract.py -q
# -> all green, including the reflexivity test that injects a synthetic strategy
```

The reflexivity test (`test_schema_generator_reflects_synthetic_strategy_without_manual_table_edits`) registers a throwaway class at runtime and proves the generator picks it up with no table edits — which is the formal statement of "the constructor signature is the single source of truth".
`test_algorithms_schema_enforces_backend_specific_two_opt_modes_and_zero_candidate_k` pins the other direction: the schema must reject `mode: "batch"` for `LocalSearchCPUTwoOpt` and `mode: "best"` for `LocalSearchGPUTwoOpt`, because those combinations fail in the constructor.

Numeric bounds (`minimum: 32` on `threads_per_block`, `minimum: 2` on `max_shared_tour_cities`) are **consultative**: they are not derivable from `inspect.signature` and are not runtime invariants.
Do not promote them to requirements of the generator; `threads_per_block` in particular is a ceiling that `_improve_grid` legitimately lowers with `prev_power_of_2`.
