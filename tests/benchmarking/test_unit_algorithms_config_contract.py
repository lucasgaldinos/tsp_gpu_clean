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
  DEFAULT_SCHEMA_PATH,
  STRATEGY_SLOT_PROTOCOLS,
  generate_algorithms_schema,
  render_algorithms_schema,
  resolve_init_hints,
  unwrap_type_annotation,
)
from benchmarking.runner.schema_generator import (
  main as schema_generator_main,
)
from protocols.types import LocalSearchStage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
CHAPTER4_CONFIG_DIR = PROJECT_ROOT / "benchmark_scripts" / "experiments-and-examples" / "chapter4" / "configs"
PROD_ALGORITHMS_JSON = CHAPTER4_CONFIG_DIR / "algorithms.json"
FIXTURE_ALGORITHMS_JSON = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "algorithms.json"
SCHEMA_DIR = CONFIG_DIR / "schemas"
SCHEMA_NAMES: tuple[str, ...] = ("algorithms.schema.json", "benchmark.schema.json", "problems.schema.json")
CHAPTER4_CONFIG_NAMES: tuple[str, ...] = (
  "algorithms.json",
  "algorithms.archive.json",
  "benchmark.json",
  "problems.json",
)
EXAMPLE_ALGORITHM_CONFIGS: tuple[str, ...] = (
  "algorithms.full.example.json",
  "cpu-operators.example.json",
  "local-search-pipelines.example.json",
  "gpu-backends.example.json",
)
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
      assert p_slot_cfg.get("class") == f_slot_cfg.get("class"), f"Strategy class mismatch in {variant_name}.{slot}"
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
  """WHAT: The `$schema` key of every example and Chapter-4 configuration file.

  WHY: Without it the editor cannot resolve completions or flag unknown keys.
  EXPECTED: Every relative `$schema` reference points to an existing schema file under `configs/schemas/`.
  """
  for config_name in EXAMPLE_ALGORITHM_CONFIGS:
    config_path = CONFIG_DIR / config_name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema_ref = config["$schema"]
    assert isinstance(schema_ref, str) and schema_ref.startswith("./schemas/"), (
      f"{config_name} must reference ./schemas/, got {schema_ref!r}"
    )
    assert (config_path.parent / schema_ref).resolve().is_file(), f"{config_name} references a missing schema"

  for config_name in CHAPTER4_CONFIG_NAMES:
    config_path = CHAPTER4_CONFIG_DIR / config_name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema_ref = config["$schema"]
    assert isinstance(schema_ref, str) and "configs/schemas/" in schema_ref, (
      f"{config_name} must reference configs/schemas/, got {schema_ref!r}"
    )
    assert (config_path.parent / schema_ref).resolve().is_file(), f"{config_name} references a missing schema"


def test_example_configs_instantiate_every_variant() -> None:
  """WHAT: All generic `configs/*.example.json` algorithm configs loaded through the real loader.

  WHY: Example configs are the user-facing showcase of Design B strategies, modes, and
  composable pipelines; every shipped example must stay schema-valid and instantiable.
  """
  schema = json.loads((SCHEMA_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))

  for example_name in EXAMPLE_ALGORITHM_CONFIGS:
    example_path = CONFIG_DIR / example_name
    jsonschema.validate(instance=json.loads(example_path.read_text(encoding="utf-8")), schema=schema)
    ga_params, algorithms = load_algorithms_config(example_path)
    assert len(algorithms) >= 3, f"{example_name} should showcase at least 3 variants"
    for name, cfg in algorithms.items():
      for slot, spec in cfg.strategies.items():
        assert spec["class"] in STRATEGY_REGISTRY, f"unknown class in {example_name}:{name}.{slot}"
      assert cfg.instantiate(ga_params) is not None, f"{example_name}:{name} failed to instantiate"

  _, full_algorithms = load_algorithms_config(CONFIG_DIR, config_name="algorithms.full.example.json")
  assert len(full_algorithms) >= 6
  modes = {
    str((spec.get("params") or {}).get("mode"))
    for cfg in full_algorithms.values()
    for spec in cfg.strategies.values()
    if (spec.get("params") or {}).get("mode") is not None
  }
  assert modes == {"best", "first", "multi_move", "batch", "naive"}


def _extract_oneof_builtin_branch_map(slot_schema: dict[str, Any]) -> dict[str, str]:
  """Map each built-in `oneOf` branch's `class.const` value to its referenced `$defs` key."""
  branch_map: dict[str, str] = {}
  generic_external_count = 0
  for branch in slot_schema["oneOf"]:
    props = branch["properties"]
    if "module" in props:
      assert branch["required"] == ["module", "class"]
      assert branch["additionalProperties"] is False
      if "const" not in props["module"]:
        generic_external_count += 1
      continue
    assert branch["required"] == ["class"]
    assert branch["additionalProperties"] is False
    class_name = props["class"]["const"]
    ref = props["params"]["$ref"]
    assert ref.startswith("#/$defs/")
    branch_map[class_name] = ref.removeprefix("#/$defs/")
  assert generic_external_count == 1
  return branch_map


def _all_error_messages(exc: jsonschema.ValidationError) -> str:
  """Collect top-level and nested `oneOf`/`anyOf` sub-error messages for diagnostic assertions."""
  messages = [exc.message]
  for sub in exc.context:
    messages.append(_all_error_messages(sub))
  return "\n".join(messages)


def test_algorithms_schema_matches_generator_and_registries_drift_contract() -> None:
  """WHAT: Reflective drift contract between registries and `algorithms.schema.json` (BUG-85, BUG-103).

  WHY: Hand-maintained schemas drift silently when new strategies or `Literal` modes are added,
  and a single flat `strategy_slot` fails to partition strategies by PEP 544 protocol slots.
  EXPECTED:
    (a) 6 per-slot `strategy_slot_*` `$defs` partition `STRATEGY_REGISTRY` (18) via `oneOf`,
        and `pipeline_step` covers `PIPELINE_STEP_REGISTRY` (3) via `oneOf` (no legacy `strategy_slot`);
    (b) every `$defs/params_*` key set equals the constructor signature parameters;
    (c) every `Literal`-derived `enum` matches `typing.get_args`;
    (d) nested pipeline strategy refs point to protocol-specific slots;
    (e) committed `configs/schemas/algorithms.schema.json` equals generator output.
  """
  committed_text = (SCHEMA_DIR / "algorithms.schema.json").read_text(encoding="utf-8")
  committed_schema = json.loads(committed_text)
  generated_schema = generate_algorithms_schema()

  # (e) Committed schema is byte-for-byte identical to generator output.
  assert committed_schema == generated_schema
  assert committed_text == render_algorithms_schema()

  defs = committed_schema["$defs"]
  assert "strategy_slot" not in defs

  # (a) Per-slot protocol-partitioned `oneOf` coverage across all 6 strategy slots + pipeline steps.
  strategy_branches: dict[str, str] = {}
  for slot_name, proto, def_name in STRATEGY_SLOT_PROTOCOLS:
    slot_def = defs[def_name]
    slot_branch_map = _extract_oneof_builtin_branch_map(slot_def)
    expected_slot_classes = {name for name, cls in STRATEGY_REGISTRY.items() if issubclass(cls, proto)}
    assert set(slot_branch_map.keys()) == expected_slot_classes, f"Slot mismatch for {slot_name}"
    assert len(slot_def["oneOf"]) == len(expected_slot_classes) + 1
    strategy_branches.update(slot_branch_map)

  assert set(strategy_branches.keys()) == set(STRATEGY_REGISTRY.keys())
  assert len(strategy_branches) == len(STRATEGY_REGISTRY) == 18

  pipeline_step = defs["pipeline_step"]
  pipeline_branches = _extract_oneof_builtin_branch_map(pipeline_step)
  assert set(pipeline_branches.keys()) == set(PIPELINE_STEP_REGISTRY.keys())
  assert len(pipeline_step["oneOf"]) == len(PIPELINE_STEP_REGISTRY) + 1

  # (d) Nested pipeline step strategy refs target protocol-specific slots.
  assert defs["params_elite_only_local_search_step"]["properties"]["local_search"]["$ref"] == (
    "#/$defs/strategy_slot_local_search"
  )
  assert defs["params_elite_only_local_search_step"]["properties"]["evaluation"]["$ref"] == (
    "#/$defs/strategy_slot_evaluation"
  )
  assert defs["params_periodic_local_search_step"]["properties"]["local_search"]["$ref"] == (
    "#/$defs/strategy_slot_local_search"
  )
  assert defs["params_periodic_local_search_step"]["properties"]["re_evaluate"]["$ref"] == (
    "#/$defs/strategy_slot_evaluation"
  )
  assert defs["params_post_init_local_search_step"]["properties"]["local_search"]["$ref"] == (
    "#/$defs/strategy_slot_local_search"
  )
  assert defs["params_post_init_local_search_step"]["properties"]["evaluation"]["$ref"] == (
    "#/$defs/strategy_slot_evaluation"
  )

  # Modular vs. monolithic discriminated `oneOf` on `algorithm_config`.
  assert len(defs["algorithm_config"]["oneOf"]) == 2

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
  """WHAT: Reflexivity check with a temporary synthetic `LocalSearchStrategy` in `STRATEGY_REGISTRY`.

  WHY: Proves `schema_generator` inspects registries and PEP 544 protocols dynamically rather than
  reading a hardcoded lookup table (`Task 13.27(iv)`, `BUG-103`).
  EXPECTED: Registering a synthetic `LocalSearchStrategy` emits a `oneOf` branch strictly under
  `strategy_slot_local_search` (not other slots) with `$defs` matching its `__init__` params.
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

    def improve(self, route: Any, context: Any) -> Any:
      del context
      return route

    def improve_batch(self, routes: Any, context: Any) -> Any:
      del context
      return routes

  monkeypatch.setitem(STRATEGY_REGISTRY, "SyntheticThreeOptPreview", SyntheticThreeOptPreview)

  schema = generate_algorithms_schema()
  ls_slot = schema["$defs"]["strategy_slot_local_search"]
  branch_map = _extract_oneof_builtin_branch_map(ls_slot)
  assert "SyntheticThreeOptPreview" in branch_map
  assert "SyntheticThreeOptPreview" not in _extract_oneof_builtin_branch_map(schema["$defs"]["strategy_slot_mutation"])

  def_key = branch_map["SyntheticThreeOptPreview"]
  param_def = schema["$defs"][def_key]
  assert param_def["additionalProperties"] is False
  assert set(param_def["properties"].keys()) == {"acceptance", "max_passes", "tolerance", "dry_run"}
  assert param_def["properties"]["acceptance"] == {"type": "string", "enum": ["greedy", "tabu"]}
  assert param_def["properties"]["max_passes"] == {"type": "integer"}
  assert param_def["properties"]["tolerance"] == {"type": ["number", "null"]}
  assert param_def["properties"]["dry_run"] == {"type": "boolean"}

  base_cfg = json.loads(PROD_ALGORITHMS_JSON.read_text(encoding="utf-8"))
  base_cfg["algorithm_configs"]["CPU"]["strategies"]["local_search"] = {
    "class": "SyntheticThreeOptPreview",
    "params": {"acceptance": "greedy", "max_passes": 4, "tolerance": 0.01, "dry_run": True},
  }
  jsonschema.validate(instance=base_cfg, schema=schema)

  base_cfg["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"]["acceptance"] = "invalid"
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=base_cfg, schema=schema)


def test_algorithms_schema_enforces_backend_specific_two_opt_modes_and_zero_candidate_k() -> None:
  """WHAT: Per-class `mode` enum separation, per-slot protocol isolation, and `candidate_k: 0` (`BUG-85`, `BUG-103`).

  WHY: Ensures invalid backend modes, wrong-slot classes, monolithic keys on modular variants,
  and unknown built-in classes all fail JSON Schema validation while external `module` specs pass.
  """
  schema = json.loads((SCHEMA_DIR / "algorithms.schema.json").read_text(encoding="utf-8"))
  base = json.loads(PROD_ALGORITHMS_JSON.read_text(encoding="utf-8"))

  bad_cpu = json.loads(json.dumps(base))
  bad_cpu["algorithm_configs"]["CPU"]["strategies"]["local_search"]["params"] = {
    "max_iterations": 10,
    "mode": "batch",
  }
  with pytest.raises(jsonschema.ValidationError) as cpu_exc:
    jsonschema.validate(instance=bad_cpu, schema=schema)
  assert "is not one of" in _all_error_messages(cpu_exc.value)

  bad_gpu = json.loads(json.dumps(base))
  bad_gpu["algorithm_configs"]["HybridOptimized"]["strategies"]["local_search"]["params"] = {
    "max_iterations": 10,
    "mode": "best",
  }
  with pytest.raises(jsonschema.ValidationError) as gpu_exc:
    jsonschema.validate(instance=bad_gpu, schema=schema)
  assert "is not one of" in _all_error_messages(gpu_exc.value)

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
  with pytest.raises(jsonschema.ValidationError) as step_exc:
    jsonschema.validate(instance=bad_step, schema=schema)
  assert "Additional properties are not allowed" in _all_error_messages(step_exc.value)

  # Wrong-slot strategy in top-level slot AND inside nested pipeline step param (BUG-103).
  wrong_slot = json.loads(json.dumps(base))
  wrong_slot["algorithm_configs"]["CPU"]["strategies"]["crossover"] = {"class": "MutationSwap"}
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=wrong_slot, schema=schema)

  wrong_nested_slot = json.loads(json.dumps(base))
  wrong_nested_slot["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {
      "class": "PeriodicLocalSearchStep",
      "params": {"interval": 5, "target": "population", "re_evaluate": {"class": "MutationSwap"}},
    }
  ]
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=wrong_nested_slot, schema=schema)

  # Monolithic-only keys rejected on modular GeneticAlgorithm (BUG-105).
  monolithic_leak = json.loads(json.dumps(base))
  monolithic_leak["algorithm_configs"]["CPU"]["params"]["tournament_size"] = 5
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=monolithic_leak, schema=schema)

  external_cfg = json.loads(json.dumps(base))
  external_cfg["algorithm_configs"]["CPU"]["strategies"]["local_search"] = {
    "module": "my_custom_ops.local_search",
    "class": "CustomOrOptSearch",
    "params": {"window_size": 3, "custom_flag": True},
  }
  external_cfg["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {
      "module": "my_custom_ops.steps",
      "class": "CustomDiversityStep",
      "params": {"every_n_generations": 5},
    }
  ]
  jsonschema.validate(instance=external_cfg, schema=schema)

  unknown_builtin = json.loads(json.dumps(base))
  unknown_builtin["algorithm_configs"]["CPU"]["strategies"]["local_search"] = {
    "class": "CustomOrOptSearch",
  }
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=unknown_builtin, schema=schema)


def test_schema_generator_extra_file_emits_typed_external_oneof_and_guards_core_schema(
  tmp_path: Path,
) -> None:
  """WHAT: Experiment-local schema generation via `--extra-file` + `--output` (`BUG-103`, `DP16-3`).

  WHY: Custom experiment operators in hyphenated folders need typed slot/param IntelliSense
  without polluting `configs/schemas/algorithms.schema.json`.
  EXPECTED:
    - Refuses `--extra-file` without `--output` or targeting `DEFAULT_SCHEMA_PATH` (exit code 2);
    - Emits config-relative `"module": "../my-ops.py"` typed `oneOf` branches in the matching slot
      only, rejects unknown params on discovered external classes, and keeps generic fallback working.
  """
  exp_dir = tmp_path / "experiment-comparing-OX-and-PMX"
  configs_dir = exp_dir / "configs"
  configs_dir.mkdir(parents=True)
  ops_file = exp_dir / "my-ops.py"
  ops_file.write_text(
    (
      "from __future__ import annotations\n"
      "from typing import Any, Literal\n"
      "from algorithms.strategies import StrategyBase\n\n"
      "class CustomShiftMutation(StrategyBase):\n"
      "  def __init__(\n"
      "    self,\n"
      "    shift_span: int = 2,\n"
      "    direction: Literal['left', 'right'] = 'left',\n"
      "    seed: int | None = None,\n"
      "  ) -> None:\n"
      "    super().__init__(seed=seed)\n"
      "    self.shift_span = shift_span\n"
      "    self.direction = direction\n"
      "  def mutate(self, individual: Any, mutation_rate: float, context: Any = None) -> Any:\n"
      "    return individual\n"
      "  def mutate_batch(self, routes: Any, mutation_rate: float, context: Any = None) -> Any:\n"
      "    return routes\n\n"
      "class CustomAuditStep:\n"
      "  stage_anchor = 'post_survival'\n"
      "  def __init__(self, every: int = 4) -> None:\n"
      "    self.every = every\n"
      "  def run(self, state: Any, timer: Any) -> None:\n"
      "    del state, timer\n"
    ),
    encoding="utf-8",
  )

  # Guard: cannot target core schema with --extra-file.
  assert schema_generator_main(["--extra-file", str(ops_file)]) == 2
  assert schema_generator_main(["--extra-file", str(ops_file), "--output", str(DEFAULT_SCHEMA_PATH)]) == 2

  local_schema_path = configs_dir / "algorithms.schema.json"
  assert schema_generator_main(["--extra-file", str(ops_file), "--output", str(local_schema_path)]) == 0
  assert schema_generator_main(["--extra-file", str(ops_file), "--output", str(local_schema_path), "--check"]) == 0

  local_schema = json.loads(local_schema_path.read_text(encoding="utf-8"))
  mut_branches = local_schema["$defs"]["strategy_slot_mutation"]["oneOf"]
  custom_mut = [b for b in mut_branches if b["properties"].get("class", {}).get("const") == "CustomShiftMutation"]
  assert len(custom_mut) == 1
  assert custom_mut[0]["properties"]["module"] == {
    "const": "../my-ops.py",
    "description": "External Python module path or config-relative .py file path.",
  }
  # Must NOT leak into unrelated slots.
  cx_branches = local_schema["$defs"]["strategy_slot_crossover"]["oneOf"]
  assert not any(b["properties"].get("class", {}).get("const") == "CustomShiftMutation" for b in cx_branches)

  base = json.loads(PROD_ALGORITHMS_JSON.read_text(encoding="utf-8"))
  valid_ext = json.loads(json.dumps(base))
  valid_ext["algorithm_configs"]["CPU"]["strategies"]["mutation"] = {
    "module": "../my-ops.py",
    "class": "CustomShiftMutation",
    "params": {"shift_span": 4, "direction": "right"},
  }
  valid_ext["algorithm_configs"]["CPU"]["params"]["pipeline"] = [
    {"module": "../my-ops.py", "class": "CustomAuditStep", "params": {"every": 2}}
  ]
  jsonschema.validate(instance=valid_ext, schema=local_schema)

  # Typo on discovered external class MUST fail (not silently fall back to generic branch).
  typo_ext = json.loads(json.dumps(valid_ext))
  typo_ext["algorithm_configs"]["CPU"]["strategies"]["mutation"]["params"] = {"shfit_span": 4}
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=typo_ext, schema=local_schema)

  # Unrelated ad-hoc external class still passes via the generic fallback branch.
  adhoc_ext = json.loads(json.dumps(base))
  adhoc_ext["algorithm_configs"]["CPU"]["strategies"]["mutation"] = {
    "module": "other_pkg.ops",
    "class": "AdHocMutation",
    "params": {"anything": True},
  }
  jsonschema.validate(instance=adhoc_ext, schema=local_schema)
