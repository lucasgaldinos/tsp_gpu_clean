"""Unit tests verifying structural parity and import contracts for algorithm configurations.

Compares production `configs/algorithms.json` against the test fixture
`tests/fixtures/configs/algorithms.json` to prevent schema/contract drift, and
validates the `$schema` documents plus the complete example configuration (`DP11-5`).
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Literal, get_args

import jsonschema  # type: ignore[import-untyped]
import pytest

from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
from algorithms.strategies import STRATEGY_REGISTRY
from algorithms.strategies.local_search import TwoOptCPUMode, TwoOptGPUMode
from benchmarking.runner.config_loader import load_algorithms_config
from benchmarking.runner.schema_generator import (
  generate_algorithms_schema,
  render_algorithms_schema,
  resolve_init_hints,
  unwrap_type_annotation,
)
from protocols.types import LocalSearchStage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
PROD_ALGORITHMS_JSON = PROJECT_ROOT / "configs" / "algorithms.json"
FIXTURE_ALGORITHMS_JSON = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "algorithms.json"
SCHEMA_DIR = CONFIG_DIR / "schemas"
SCHEMA_NAMES: tuple[str, ...] = ("algorithms.schema.json", "benchmark.schema.json", "problems.schema.json")
CONFIG_NAMES: tuple[str, ...] = ("algorithms.json", "algorithms.archive.json", "benchmark.json", "problems.json")
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _load_raw_configs() -> tuple[dict[str, object], dict[str, object]]:
  with PROD_ALGORITHMS_JSON.open("r", encoding="utf-8") as f:
    prod_data = json.load(f)
  with FIXTURE_ALGORITHMS_JSON.open("r", encoding="utf-8") as f:
    fixture_data = json.load(f)
  return prod_data, fixture_data


def test_algorithms_json_structural_parity() -> None:
  """WHAT: Compare production algorithms.json with fixture algorithms.json.

  WHY: Production config must stay structurally aligned with test fixtures to
  prevent silent divergence in variant names, class modules, or strategy slots.

  EXPECTED: Identical variant names, metaheuristics, class paths, descriptions,
  and strategy/parameter slot names.
  """
  prod, fixture = _load_raw_configs()

  assert set(prod.keys()) == set(fixture.keys())
  prod_default = prod["default_params"]
  fixture_default = fixture["default_params"]
  assert isinstance(prod_default, dict) and isinstance(fixture_default, dict)
  assert set(prod_default.keys()) == set(fixture_default.keys())

  prod_algs = prod["algorithm_configs"]
  fixture_algs = fixture["algorithm_configs"]
  assert isinstance(prod_algs, dict) and isinstance(fixture_algs, dict)
  assert set(prod_algs.keys()) == set(fixture_algs.keys()), (
    f"Variant sets differ: prod={set(prod_algs.keys())}, fixture={set(fixture_algs.keys())}"
  )

  for variant_name, prod_cfg in prod_algs.items():
    assert isinstance(prod_cfg, dict)
    fix_cfg = fixture_algs[variant_name]
    assert isinstance(fix_cfg, dict)

    assert prod_cfg.get("metaheuristic") == fix_cfg.get("metaheuristic"), f"Mismatch in {variant_name}.metaheuristic"
    assert prod_cfg.get("use_gpu") == fix_cfg.get("use_gpu"), f"Mismatch in {variant_name}.use_gpu"
    assert prod_cfg.get("class_module") == fix_cfg.get("class_module"), f"Mismatch in {variant_name}.class_module"
    assert prod_cfg.get("class_name") == fix_cfg.get("class_name"), f"Mismatch in {variant_name}.class_name"
    assert prod_cfg.get("description") == fix_cfg.get("description"), f"Mismatch in {variant_name}.description"

    prod_params = prod_cfg.get("params") or {}
    fix_params = fix_cfg.get("params") or {}
    assert set(prod_params.keys()) == set(fix_params.keys()), f"Param keys differ in {variant_name}"

    prod_strats = prod_cfg.get("strategies") or {}
    fix_strats = fix_cfg.get("strategies") or {}
    assert set(prod_strats.keys()) == set(fix_strats.keys()), f"Strategy slots differ in {variant_name}"

    for slot, p_slot_cfg in prod_strats.items():
      f_slot_cfg = fix_strats[slot]
      assert p_slot_cfg.get("class") == f_slot_cfg.get("class"), (
        f"Strategy class mismatch in {variant_name}.{slot}"
      )
      p_slot_params = p_slot_cfg.get("params") or {}
      f_slot_params = f_slot_cfg.get("params") or {}
      assert set(p_slot_params.keys()) == set(f_slot_params.keys()), (
        f"Strategy param keys differ in {variant_name}.{slot}"
      )


def test_production_variants_importable() -> None:
  """WHAT: Verify every variant and strategy in production config resolves to an importable class.

  WHY: Guarantees that production config has no broken import paths or typos.

  EXPECTED: All modules and class symbols can be successfully imported.
  """
  prod, _ = _load_raw_configs()
  prod_algs = prod["algorithm_configs"]
  assert isinstance(prod_algs, dict)

  for variant_name, cfg in prod_algs.items():
    assert isinstance(cfg, dict)
    class_module = cfg.get("class_module")
    class_name = cfg.get("class_name")

    if class_module and class_name:
      mod = importlib.import_module(str(class_module))
      cls = getattr(mod, str(class_name))
      assert callable(cls), f"Class {class_name} in {class_module} is not callable"

    strategies = cfg.get("strategies") or {}
    for slot, s_cfg in strategies.items():
      s_cls_name = s_cfg.get("class")
      assert s_cls_name, f"Missing strategy class in {variant_name}.{slot}"
      assert s_cls_name in STRATEGY_REGISTRY, (
        f"Strategy '{s_cls_name}' for slot '{slot}' not found in STRATEGY_REGISTRY"
      )


def test_schema_documents_exist_as_draft_2020_12() -> None:
  """WHAT: The three `configs/schemas/*.schema.json` documents.

  WHY: The `$schema` keys added to every config resolve against these files; a
  missing or malformed document turns editor validation into noise (`DP11-5`).
  EXPECTED: Each file exists, parses as JSON, and declares Draft 2020-12.
  """
  for schema_name in SCHEMA_NAMES:
    schema_path = SCHEMA_DIR / schema_name
    assert schema_path.is_file(), f"missing schema: {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"] == DRAFT_2020_12, f"{schema_name} must declare Draft 2020-12"
    assert schema["type"] == "object", f"{schema_name} must describe an object document"


def test_every_config_declares_a_schema_path() -> None:
  """WHAT: The `$schema` key of each production configuration file.

  WHY: Without it the editor cannot resolve completions or flag unknown keys.
  EXPECTED: A relative `./schemas/<name>.schema.json` reference pointing at an existing file.
  """
  for config_name in CONFIG_NAMES:
    config = json.loads((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    schema_ref = config["$schema"]
    assert isinstance(schema_ref, str) and schema_ref.startswith("./schemas/"), (
      f"{config_name} must reference ./schemas/, got {schema_ref!r}"
    )
    assert (CONFIG_DIR / schema_ref).is_file(), f"{config_name} references a missing schema"


def test_example_config_instantiates_every_variant() -> None:
  """WHAT: `configs/algorithms.full.example.json` loaded through the real loader.

  WHY: The example is the human-facing catalogue of every `Design B` class, mode and
  knob; it is only useful while it still instantiates (`DP11-5`).
  EXPECTED: At least six variants, all strategy classes in `STRATEGY_REGISTRY`, every
  variant builds a `GeneticAlgorithm` without raising, and all five modes appear.
  """
  ga_params, algorithms = load_algorithms_config(CONFIG_DIR, config_name="algorithms.full.example.json")

  assert len(algorithms) >= 6
  modes: set[str] = set()
  for name, cfg in algorithms.items():
    for slot, spec in cfg.strategies.items():
      assert spec["class"] in STRATEGY_REGISTRY, f"unknown class in {name}.{slot}"
      mode = (spec.get("params") or {}).get("mode")
      if mode is not None:
        modes.add(str(mode))
    assert cfg.instantiate(ga_params) is not None, f"variant {name} failed to instantiate"

  assert modes == {"best", "first", "multi_move", "batch", "naive"}


def _extract_allof_branch_map(slot_schema: dict[str, Any]) -> dict[str, str]:
  """Map each `allOf` branch's `class.const` value to its referenced `$defs` key."""
  branch_map: dict[str, str] = {}
  for branch in slot_schema["allOf"]:
    assert branch["if"]["required"] == ["class"]
    class_name = branch["if"]["properties"]["class"]["const"]
    ref = branch["then"]["properties"]["params"]["$ref"]
    assert ref.startswith("#/$defs/")
    branch_map[class_name] = ref.removeprefix("#/$defs/")
  return branch_map


def test_algorithms_schema_matches_generator_and_registries_drift_contract() -> None:
  """WHAT: Reflective drift contract between registries and `algorithms.schema.json` (BUG-85).

  WHY: Hand-maintained schemas drift silently when new strategies or `Literal` modes are added,
  leaving unregistered or invalid params unconstrained (`Task 13.27(iv)`).
  EXPECTED:
    (a) `len(allOf)` in `strategy_slot` == `len(STRATEGY_REGISTRY)` (17) and in
        `pipeline_step` == `len(PIPELINE_STEP_REGISTRY)` (3);
    (b) every `$defs/params_*` key set equals the constructor signature parameters;
    (c) every `Literal`-derived `enum` matches `typing.get_args`;
    (d) committed `configs/schemas/algorithms.schema.json` equals generator output.
  """
  committed_text = (SCHEMA_DIR / "algorithms.schema.json").read_text(encoding="utf-8")
  committed_schema = json.loads(committed_text)
  generated_schema = generate_algorithms_schema()

  # (d) Committed schema is byte-for-byte identical to generator output.
  assert committed_schema == generated_schema
  assert committed_text == render_algorithms_schema()

  defs = committed_schema["$defs"]

  # (a) Branch coverage across both registries.
  strategy_slot = defs["strategy_slot"]
  pipeline_step = defs["pipeline_step"]
  assert len(strategy_slot["allOf"]) == len(STRATEGY_REGISTRY) == 17
  assert len(pipeline_step["allOf"]) == len(PIPELINE_STEP_REGISTRY) == 3

  strategy_branches = _extract_allof_branch_map(strategy_slot)
  pipeline_branches = _extract_allof_branch_map(pipeline_step)
  assert set(strategy_branches.keys()) == set(STRATEGY_REGISTRY.keys())
  assert set(pipeline_branches.keys()) == set(PIPELINE_STEP_REGISTRY.keys())
  assert set(strategy_slot["properties"]["class"]["enum"]) == set(STRATEGY_REGISTRY.keys())
  assert set(pipeline_step["properties"]["class"]["enum"]) == set(PIPELINE_STEP_REGISTRY.keys())

  # (b) & (c) Constructor parameter fidelity and Literal enum parity across all 20 classes.
  for registry, branch_map in (
    (STRATEGY_REGISTRY, strategy_branches),
    (PIPELINE_STEP_REGISTRY, pipeline_branches),
  ):
    for class_name, cls in registry.items():
      def_key = branch_map[class_name]
      param_def = defs[def_key]
      assert param_def["additionalProperties"] is False

      sig = inspect.signature(cls)
      expected_params = {
        name
        for name, param in sig.parameters.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
      }
      assert set(param_def["properties"].keys()) == expected_params, (
        f"Schema keys mismatch for {class_name}: {set(param_def['properties'].keys())} != {expected_params}"
      )

      hints = resolve_init_hints(cls)
      for param_name in expected_params:
        ann = unwrap_type_annotation(hints.get(param_name, sig.parameters[param_name].annotation))
        if getattr(ann, "__origin__", None) is Literal:
          assert param_def["properties"][param_name]["enum"] == list(get_args(ann))

  assert defs["params_local_search_cpu_two_opt"]["properties"]["mode"]["enum"] == list(
    get_args(unwrap_type_annotation(TwoOptCPUMode))
  )
  assert defs["params_local_search_gpu_two_opt"]["properties"]["mode"]["enum"] == list(
    get_args(unwrap_type_annotation(TwoOptGPUMode))
  )
  assert defs["local_search_stage"]["enum"] == list(get_args(unwrap_type_annotation(LocalSearchStage)))


def test_schema_generator_reflects_synthetic_strategy_without_manual_table_edits(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Reflexivity check with a temporary synthetic strategy in `STRATEGY_REGISTRY`.

  WHY: Proves `schema_generator` inspects registries dynamically rather than reading
  a hardcoded class-to-params lookup table (`Task 13.27(iv)`).
  EXPECTED: Registering a synthetic class emits a new `allOf` branch and `$defs`
  schema matching its `__init__` parameters and `Literal` enum.
  """

  class SyntheticThreeOptPreview:
    def __init__(
      self,
      acceptance: Literal["greedy", "tabu"] = "greedy",
      max_passes: int = 3,
      tolerance: float | None = None,
      dry_run: bool = False,
    ) -> None:
      self.acceptance = acceptance
      self.max_passes = max_passes
      self.tolerance = tolerance
      self.dry_run = dry_run

  monkeypatch.setitem(STRATEGY_REGISTRY, "SyntheticThreeOptPreview", SyntheticThreeOptPreview)

  schema = generate_algorithms_schema()
  strategy_slot = schema["$defs"]["strategy_slot"]
  assert len(strategy_slot["allOf"]) == len(STRATEGY_REGISTRY)
  branch_map = _extract_allof_branch_map(strategy_slot)
  assert "SyntheticThreeOptPreview" in branch_map

  def_key = branch_map["SyntheticThreeOptPreview"]
  param_def = schema["$defs"][def_key]
  assert param_def["additionalProperties"] is False
  assert set(param_def["properties"].keys()) == {"acceptance", "max_passes", "tolerance", "dry_run"}
  assert param_def["properties"]["acceptance"] == {"type": "string", "enum": ["greedy", "tabu"]}
  assert param_def["properties"]["max_passes"] == {"type": "integer"}
  assert param_def["properties"]["tolerance"] == {"type": ["number", "null"]}
  assert param_def["properties"]["dry_run"] == {"type": "boolean"}

  base_cfg = json.loads((CONFIG_DIR / "algorithms.json").read_text(encoding="utf-8"))
  base_cfg["algorithm_configs"]["CPU"]["strategies"]["local_search"] = {
    "class": "SyntheticThreeOptPreview",
    "params": {"acceptance": "greedy", "max_passes": 4, "tolerance": 0.01, "dry_run": True},
  }
  jsonschema.validate(instance=base_cfg, schema=schema)

  base_cfg["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"]["acceptance"] = "invalid"
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=base_cfg, schema=schema)


def test_algorithms_schema_enforces_backend_specific_two_opt_modes_and_zero_candidate_k() -> None:
  """WHAT: Per-class `mode` enum separation and `candidate_k: 0` acceptance (`BUG-85`).

  WHY: Before `Task 13.27(iv)`, `algorithms.schema.json` shared one `mode` enum across
  CPU and GPU 2-opt and forbade `candidate_k: 0`, inverting runtime acceptance.
  EXPECTED:
    - `LocalSearchCPUTwoOpt` with `mode: "batch"` fails `jsonschema.validate`;
    - `LocalSearchGPUTwoOpt` with `mode: "best"` fails `jsonschema.validate`;
    - `LocalSearchCPUTwoOpt` with `mode: "best"` and `candidate_k: 0` passes `jsonschema.validate`;
    - `PostInitLocalSearchStep` with `interval: 5` fails `jsonschema.validate`.
  """
  schema = json.loads((SCHEMA_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))
  base = json.loads((CONFIG_DIR / "algorithms.json").read_text(encoding="utf-8"))

  bad_cpu = json.loads(json.dumps(base))
  bad_cpu["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"] = {
    "max_iterations": 10,
    "mode": "batch",
  }
  with pytest.raises(jsonschema.ValidationError, match="is not one of"):
    jsonschema.validate(instance=bad_cpu, schema=schema)

  bad_gpu = json.loads(json.dumps(base))
  bad_gpu["algorithm_configs"]["HybridOptimized"]["strategies"]["local_search"]["params"] = {
    "max_iterations": 10,
    "mode": "best",
  }
  with pytest.raises(jsonschema.ValidationError, match="is not one of"):
    jsonschema.validate(instance=bad_gpu, schema=schema)

  valid_cpu = json.loads(json.dumps(base))
  valid_cpu["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"] = {
    "max_iterations": 10,
    "mode": "best",
    "candidate_k": 0,
  }
  jsonschema.validate(instance=valid_cpu, schema=schema)

  bad_step = json.loads(json.dumps(base))
  bad_step["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {"class": "PostInitLocalSearchStep", "params": {"interval": 5}},
  ]
  with pytest.raises(jsonschema.ValidationError, match="Additional properties are not allowed"):
    jsonschema.validate(instance=bad_step, schema=schema)
