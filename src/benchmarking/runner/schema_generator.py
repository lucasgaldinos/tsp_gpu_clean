"""Reflective JSON Schema generator for `algorithms.schema.json` and `report_style.schema.json`.

Derives per-slot protocol-partitioned `oneOf` branches and per-class parameter schemas
directly from `STRATEGY_REGISTRY`, `PIPELINE_STEP_REGISTRY`, and optional external
user modules/files via `inspect.signature`, `typing.get_type_hints`, and
`typing.get_args` (`BUG-85`, `BUG-103`).

Why discriminated `oneOf` per protocol slot instead of one flat `allOf` if/then:
  1. Partitioning `STRATEGY_REGISTRY` by `@runtime_checkable` PEP 544 slot protocols
     (`InitializationStrategy`, ..., `EvaluationStrategy`) aligns JSON Schema validation
     and editor completions with `config_loader.py::_validate_class_spec` (`BUG-101`).
  2. Mutually exclusive `oneOf` branches (`class: {const: ...}` + `params: {$ref: ...}`,
     plus a `module + class` external branch) let both Draft 2020-12 validators and
     `vscode-json-languageservice` narrow slot-specific class names and class-specific
     constructor parameters directly without negated `if`/`not` guards.
  3. Passing `--extra-file` / `--extra-module` with `--output` generates an
     experiment-local schema containing typed `oneOf` branches for custom operators
     without coupling `configs/schemas/algorithms.schema.json` or `src/` to experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import protocols.algorithm_strategies as _strategy_protocols
import protocols.types as _protocol_types
import protocols.vrp_strategies as _vrp_strategies
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
from algorithms.strategies import STRATEGY_REGISTRY, SUB_STRATEGY_REGISTRY
from algorithms.strategies.base import StrategyBase
from benchmarking.reporting.style import ReportStyle
from benchmarking.runner.spec_resolver import (
  is_file_module_ref,
  load_external_module,
)
from benchmarking.runner.spec_resolver import (
  unwrap_type_annotation as unwrap_type_annotation,
)

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://github.com/lucasgaldinos/gpu_accelerated_clean/configs/schemas/algorithms.schema.json"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "configs" / "schemas" / "algorithms.schema.json"
REPORT_STYLE_SCHEMA_ID = (
  "https://github.com/lucasgaldinos/gpu_accelerated_clean/configs/schemas/report_style.schema.json"
)
DEFAULT_REPORT_STYLE_SCHEMA_PATH = (
  Path(__file__).resolve().parents[3] / "configs" / "schemas" / "report_style.schema.json"
)

STRATEGY_SLOT_PROTOCOLS: tuple[tuple[str, type, str], ...] = (
  ("initialization", _strategy_protocols.InitializationStrategy, "strategy_slot_initialization"),
  ("selection", _strategy_protocols.SelectionStrategy, "strategy_slot_selection"),
  ("crossover", _strategy_protocols.CrossoverStrategy, "strategy_slot_crossover"),
  ("mutation", _strategy_protocols.MutationStrategy, "strategy_slot_mutation"),
  ("local_search", _strategy_protocols.LocalSearchStrategy, "strategy_slot_local_search"),
  ("evaluation", _strategy_protocols.EvaluationStrategy, "strategy_slot_evaluation"),
)

SUB_STRATEGY_PROTOCOLS: tuple[tuple[str, type, str], ...] = (
  ("penalty_policy", _vrp_strategies.PenaltyPolicy, "sub_strategy_penalty_policy"),
  ("decoder", _vrp_strategies.RouteDecoder, "sub_strategy_route_decoder"),
  ("bin_packing", _vrp_strategies.BinPackingStrategy, "sub_strategy_bin_packing"),
)


@dataclasses.dataclass(frozen=True)
class ExternalClassSpec:
  """External custom strategy or pipeline step discovered for an experiment-local schema."""

  module: str
  class_name: str
  cls: type


# Optional advisory metadata (numeric bounds, $comment, descriptions) overlaid onto
# reflectively discovered constructor parameters. Never introduces parameter keys
# or overrides derived `type` / `enum` / `$ref` constraints.
ADVISORY_PARAM_METADATA: dict[str, dict[str, Any]] = {
  "seed": {
    "description": "Explicit seed override; the loader injects the run seed when omitted.",
  },
  "return_gpu": {
    "description": "Keep produced routes/costs in VRAM instead of returning host arrays.",
  },
  "candidate_k": {
    "minimum": 0,
    "$comment": "Advisory bound: <= 0 or null disables candidate-list pruning.",
    "description": "Nearest-neighbour list width; null or <= 0 keeps the exhaustive scan (DP10-20).",
  },
  "k": {
    "minimum": 1,
    "$comment": "Advisory bound: tournament size or candidate-list width >= 1.",
    "description": "Tournament size (selection) or candidate-list width (initialization).",
  },
  "max_iterations": {
    "minimum": 0,
    "$comment": "Advisory bound: 0 disables 2-opt refinement passes.",
    "description": "2-opt pass cap for the local-search strategies.",
  },
  "mode": {
    "description": "Acceptance rule supported by this strategy's backend.",
  },
  "threads_per_block": {
    "minimum": 32,
    "$comment": "Advisory bound: GPU strategies clamp or derive block dimensions at runtime.",
    "description": "CUDA threads per block for the GPU strategies (advisory lower bound).",
  },
  "max_matrix_dimension": {
    "minimum": 0,
    "$comment": "Advisory bound: n threshold for on-the-fly distances; null defers to DEFAULT_MAX_MATRIX_DIMENSION.",
    "description": (
      "`n` above which coordinate-based problems use on-the-fly distances; "
      "`null` defers to `DEFAULT_MAX_MATRIX_DIMENSION` (10 000)."
    ),
  },
  "max_shared_tour_cities": {
    "minimum": 2,
    "$comment": "Advisory bound: minimum tour size for shared-memory caching.",
    "description": (
      "`n` above which the GPU 2-opt scans the tour tile by tile instead of caching it "
      "in shared memory (default 11 520)."
    ),
  },
  "elite_count": {
    "minimum": 1,
    "$comment": "Advisory bound: constructor enforces elite_count > 0.",
    "description": "Number of top-fitness routes to refine (EliteOnlyLocalSearchStep).",
  },
  "interval": {
    "minimum": 1,
    "$comment": "Advisory bound: constructor enforces interval > 0.",
    "description": "Generation interval between local search passes (PeriodicLocalSearchStep).",
  },
  "target": {
    "description": "Target array to refine ('offspring' or 'population').",
  },
  "local_search": {
    "description": "Local search strategy slot override; omitted inherits the variant's local_search strategy.",
  },
  "evaluation": {
    "description": "Evaluation strategy slot override; omitted inherits the variant's evaluation strategy.",
  },
  "re_evaluate": {
    "description": (
      "Evaluation strategy slot required when target='population' to recompute fitness after local search."
    ),
  },
}


def camel_to_snake(name: str) -> str:
  """Convert PascalCase/CamelCase class names (preserving acronyms like CPU/GPU) to snake_case."""
  step1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
  return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1).lower()


def param_def_name(class_name: str) -> str:
  """Return the `$defs` key name for `class_name`'s constructor parameters."""
  return f"params_{camel_to_snake(class_name)}"


def ext_param_def_name(module_ref: str, class_name: str) -> str:
  """Return a collision-free `$defs` key name for an external class's constructor parameters."""
  mod_slug = re.sub(r"\W+", "_", module_ref).strip("_").lower() or "ext"
  return f"params_ext_{mod_slug}_{camel_to_snake(class_name)}"


def _strategy_slot_ref_for_type(tp: Any) -> dict[str, Any] | None:
  """Return the slot-specific `$ref` schema for core strategies, sub-strategies, or `StrategyBase`."""
  unwrapped = unwrap_type_annotation(tp)
  if not isinstance(unwrapped, type):
    return None
  for protocols_group in (STRATEGY_SLOT_PROTOCOLS, SUB_STRATEGY_PROTOCOLS):
    matched_defs = [
      def_name for _, proto, def_name in protocols_group if unwrapped is proto or issubclass(unwrapped, proto)
    ]
    if len(matched_defs) == 1:
      return {"$ref": f"#/$defs/{matched_defs[0]}"}
    if len(matched_defs) > 1:
      return {"anyOf": [{"$ref": f"#/$defs/{def_name}"} for def_name in matched_defs]}
  if issubclass(unwrapped, StrategyBase):
    return {"anyOf": [{"$ref": f"#/$defs/{def_name}"} for _, _, def_name in STRATEGY_SLOT_PROTOCOLS]}
  return None


def _is_strategy_type(tp: Any) -> bool:
  """Return True when `tp` is a strategy protocol or `StrategyBase` subclass."""
  return _strategy_slot_ref_for_type(tp) is not None


def _find_init_method(cls: type) -> Any:
  """Locate the defining `__init__` function along `cls.__mro__`."""
  for base in cls.__mro__:
    if base is object:
      continue
    if "__init__" in base.__dict__:
      return base.__dict__["__init__"]
  return None


def resolve_init_hints(cls: type) -> dict[str, Any]:
  """Resolve `__init__` parameter annotations for `cls`, including `TYPE_CHECKING` imports."""
  init_fn = _find_init_method(cls)
  if init_fn is None:
    return {}
  globalns: dict[str, Any] = {
    "Any": Any,
    "Literal": Literal,
    "Sequence": Sequence,
    **vars(_protocol_types),
    **vars(_strategy_protocols),
    **vars(_vrp_strategies),
  }
  module = sys.modules.get(cls.__module__)
  if module is not None:
    globalns.update(vars(module))
  try:
    return get_type_hints(init_fn, globalns=globalns, include_extras=True)
  except Exception:
    sig = inspect.signature(cls)
    return {
      name: param.annotation
      for name, param in sig.parameters.items()
      if param.annotation is not inspect.Parameter.empty
    }


def _literal_schema(annotation: Any) -> dict[str, Any]:
  """Build a JSON Schema fragment from a `Literal[...]` type annotation."""
  values = list(get_args(annotation))
  if values and all(isinstance(v, str) for v in values):
    return {"type": "string", "enum": values}
  if values and all(isinstance(v, bool) for v in values):
    return {"type": "boolean", "enum": values}
  if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
    return {"type": "integer", "enum": values}
  if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
    return {"type": "number", "enum": values}
  return {"enum": values}


def _single_member_to_schema(annotation: Any) -> dict[str, Any] | None:
  """Map one non-union Python type annotation to a JSON Schema fragment, or `None` for runtime-only types."""
  unwrapped = unwrap_type_annotation(annotation)
  slot_ref = _strategy_slot_ref_for_type(unwrapped)
  if slot_ref is not None:
    return slot_ref

  origin = get_origin(unwrapped)
  if origin is Literal:
    return _literal_schema(unwrapped)

  if unwrapped is bool:
    return {"type": "boolean"}
  if unwrapped is int:
    return {"type": "integer"}
  if unwrapped is float:
    return {"type": "number"}
  if unwrapped is str:
    return {"type": "string"}
  if unwrapped is type(None):
    return {"type": "null"}

  if unwrapped in (list, tuple) or origin in (list, tuple, Sequence):
    return {"type": "array"}
  if unwrapped is dict or origin in (dict, Mapping):
    return {"type": "object"}

  # Non-JSON runtime objects (e.g. `np.random.Generator`, `Callable`) inside unions like `SeedLike`.
  if origin is Callable or unwrapped is Callable:
    return None
  return None


def annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
  """Reflectively convert a Python type annotation into a JSON Schema property dict."""
  unwrapped = unwrap_type_annotation(annotation)
  if unwrapped is inspect.Parameter.empty or unwrapped is Any:
    return {}

  origin = get_origin(unwrapped)
  if origin in (Union, types.UnionType):
    raw_members = [unwrap_type_annotation(member) for member in get_args(unwrapped)]
    non_none = [member for member in raw_members if member is not type(None)]
    if non_none and all(_is_strategy_type(member) for member in non_none):
      slot_refs: list[dict[str, Any]] = []
      for member in non_none:
        ref_schema = _strategy_slot_ref_for_type(member)
        if ref_schema is not None and ref_schema not in slot_refs:
          slot_refs.append(ref_schema)
      if len(slot_refs) == 1:
        return slot_refs[0]
      if slot_refs:
        return {"anyOf": slot_refs}

    member_schemas: list[dict[str, Any]] = []
    for member in raw_members:
      sub_origin = get_origin(member)
      if sub_origin in (Union, types.UnionType):
        nested = annotation_to_json_schema(member)
        if nested:
          member_schemas.append(nested)
        continue
      mapped = _single_member_to_schema(member)
      if mapped is not None and mapped not in member_schemas:
        member_schemas.append(mapped)

    if not member_schemas:
      return {}
    if len(member_schemas) == 1:
      return member_schemas[0]
    if all(set(schema.keys()) == {"type"} and isinstance(schema["type"], str) for schema in member_schemas):
      return {"type": [schema["type"] for schema in member_schemas]}
    return {"anyOf": member_schemas}

  mapped = _single_member_to_schema(unwrapped)
  return mapped if mapped is not None else {}


def build_class_params_schema(cls: type) -> dict[str, Any]:
  """Inspect `cls` constructor signature and build its strict `params_*` JSON Schema object."""
  sig = inspect.signature(cls)
  hints = resolve_init_hints(cls)
  properties: dict[str, Any] = {}

  for param_name, param in sig.parameters.items():
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
      continue

    ann = hints.get(param_name, param.annotation)
    prop_schema = annotation_to_json_schema(ann)
    advisory = ADVISORY_PARAM_METADATA.get(param_name)
    if advisory:
      for key, value in advisory.items():
        if key not in prop_schema:
          prop_schema[key] = value
    properties[param_name] = prop_schema

  return {
    "type": "object",
    "additionalProperties": False,
    "properties": properties,
  }


def _partition_by_protocols(
  registry: Mapping[str, type],
  protocols: Sequence[tuple[str, type, str]],
  protocol_module_desc: str,
) -> dict[str, dict[str, type]]:
  """Partition `registry` across ``@runtime_checkable`` PEP 544 protocols."""
  partitioned: dict[str, dict[str, type]] = {slot_name: {} for slot_name, _, _ in protocols}
  for class_name, cls in sorted(registry.items(), key=lambda item: item[0]):
    matched_slots = [slot_name for slot_name, proto, _ in protocols if issubclass(cls, proto)]
    if not matched_slots:
      raise ValueError(
        f"Component '{class_name}' does not implement any PEP 544 protocol in {protocol_module_desc}."
      )
    for slot_name in matched_slots:
      partitioned[slot_name][class_name] = cls
  return partitioned


def _partition_strategy_registry(strategies: Mapping[str, type]) -> dict[str, dict[str, type]]:
  """Partition `strategies` into the 6 canonical slots via `@runtime_checkable` PEP 544 protocols."""
  return _partition_by_protocols(strategies, STRATEGY_SLOT_PROTOCOLS, "protocols.algorithm_strategies")


def _normalize_external_module_ref(
  raw_ref: str | Path,
  *,
  is_file: bool,
  schema_output_dir: Path | None,
) -> tuple[str, Path | None]:
  """Return `(emitted_module_ref, base_dir_for_loading)` for an external module or file argument."""
  ref_str = str(raw_ref)
  if is_file or is_file_module_ref(ref_str):
    resolved = Path(ref_str).resolve()
    if schema_output_dir is not None:
      rel = os.path.relpath(resolved, schema_output_dir.resolve()).replace("\\", "/")
      return rel, schema_output_dir.resolve()
    return resolved.as_posix(), None
  return ref_str, None


def discover_external_extensions(
  *,
  extra_modules: Sequence[str] = (),
  extra_files: Sequence[str | Path] = (),
  schema_output_dir: Path | None = None,
) -> tuple[dict[str, list[ExternalClassSpec]], list[ExternalClassSpec]]:
  """Import external modules/files and classify their public strategies and pipeline steps.

  Each class defined directly in an imported module (excluding private names, protocols,
  abstract classes, and `StrategyBase`) is classified via `issubclass` against the 6
  PEP 544 strategy protocols or by defining a callable `run(state, timer)` method.
  """
  external_strategies: dict[str, list[ExternalClassSpec]] = {
    slot_name: [] for slot_name, _, _ in STRATEGY_SLOT_PROTOCOLS
  }
  external_pipeline_steps: list[ExternalClassSpec] = []
  seen_strategy_keys: set[tuple[str, str, str]] = set()
  seen_step_keys: set[tuple[str, str]] = set()

  targets: list[tuple[str | Path, bool]] = [
    *((mod_ref, False) for mod_ref in extra_modules),
    *((file_path, True) for file_path in extra_files),
  ]

  for raw_ref, explicit_file in targets:
    emitted_module, load_base_dir = _normalize_external_module_ref(
      raw_ref,
      is_file=explicit_file,
      schema_output_dir=schema_output_dir,
    )
    mod = load_external_module(emitted_module, base_dir=load_base_dir)
    discovered_count = 0

    for class_name, cls in sorted(inspect.getmembers(mod, inspect.isclass), key=lambda item: item[0]):
      if class_name.startswith("_") or cls.__module__ != mod.__name__:
        continue
      if cls is StrategyBase or inspect.isabstract(cls) or bool(getattr(cls, "_is_protocol", False)):
        continue

      matched_slots = [slot_name for slot_name, proto, _ in STRATEGY_SLOT_PROTOCOLS if issubclass(cls, proto)]
      if matched_slots:
        for slot_name in matched_slots:
          key = (slot_name, emitted_module, class_name)
          if key not in seen_strategy_keys:
            seen_strategy_keys.add(key)
            external_strategies[slot_name].append(ExternalClassSpec(emitted_module, class_name, cls))
        discovered_count += 1
        continue

      if callable(getattr(cls, "run", None)):
        step_key = (emitted_module, class_name)
        if step_key not in seen_step_keys:
          seen_step_keys.add(step_key)
          external_pipeline_steps.append(ExternalClassSpec(emitted_module, class_name, cls))
        discovered_count += 1

    if discovered_count == 0:
      print(
        f"Warning: no public strategy or GenerationStep classes discovered in '{raw_ref}'.",
        file=sys.stderr,
      )

  return external_strategies, external_pipeline_steps


def _build_slot_definition(
  registry: Mapping[str, type],
  *,
  slot_label: str,
  class_description: str,
  params_description: str,
  external_specs: Sequence[ExternalClassSpec] = (),
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Build a mutually exclusive discriminated `oneOf` slot schema and its `$defs/params_*` definitions."""
  sorted_items = sorted(registry.items(), key=lambda item: item[0])
  one_of_branches: list[dict[str, Any]] = []
  param_defs: dict[str, dict[str, Any]] = {}

  for class_name, cls in sorted_items:
    def_key = param_def_name(class_name)
    param_defs[def_key] = build_class_params_schema(cls)
    one_of_branches.append(
      {
        "title": class_name,
        "type": "object",
        "required": ["class"],
        "additionalProperties": False,
        "properties": {
          "class": {
            "const": class_name,
            "description": class_description,
          },
          "params": {
            "$ref": f"#/$defs/{def_key}",
            "description": params_description,
          },
        },
      }
    )

  sorted_external = sorted(external_specs, key=lambda spec: (spec.module, spec.class_name))
  for ext_spec in sorted_external:
    def_key = ext_param_def_name(ext_spec.module, ext_spec.class_name)
    param_defs[def_key] = build_class_params_schema(ext_spec.cls)
    one_of_branches.append(
      {
        "title": f"{ext_spec.class_name} (external: {ext_spec.module})",
        "type": "object",
        "required": ["module", "class"],
        "additionalProperties": False,
        "properties": {
          "module": {
            "const": ext_spec.module,
            "description": "External Python module path or config-relative .py file path.",
          },
          "class": {
            "const": ext_spec.class_name,
            "description": f"External custom {slot_label} class exported by `{ext_spec.module}`.",
          },
          "params": {
            "$ref": f"#/$defs/{def_key}",
            "description": params_description,
          },
        },
      }
    )

  generic_external_branch: dict[str, Any] = {
    "title": f"External custom {slot_label} (module + class)",
    "type": "object",
    "required": ["module", "class"],
    "additionalProperties": False,
    "properties": {
      "module": {
        "type": "string",
        "minLength": 1,
        "description": (
          "Dotted Python module path or config-relative .py file path for loading an external "
          "custom class outside the built-in registry without modifying src/."
        ),
      },
      "class": {
        "type": "string",
        "minLength": 1,
        "description": f"External custom {slot_label} class name exported by `module`.",
      },
      "params": {
        "type": "object",
        "description": "Constructor keyword arguments passed to the external custom class.",
      },
    },
  }
  if sorted_external:
    generic_external_branch["not"] = {
      "anyOf": [
        {
          "required": ["module", "class"],
          "properties": {
            "module": {"const": ext_spec.module},
            "class": {"const": ext_spec.class_name},
          },
        }
        for ext_spec in sorted_external
      ]
    }
  one_of_branches.append(generic_external_branch)

  slot_schema: dict[str, Any] = {
    "type": "object",
    "oneOf": one_of_branches,
  }
  return slot_schema, param_defs


def generate_algorithms_schema(
  strategy_registry: Mapping[str, type] | None = None,
  pipeline_step_registry: Mapping[str, type] | None = None,
  *,
  sub_strategy_registry: Mapping[str, type] | None = None,
  external_strategies: Mapping[str, Sequence[ExternalClassSpec]] | None = None,
  external_sub_strategies: Mapping[str, Sequence[ExternalClassSpec]] | None = None,
  external_pipeline_steps: Sequence[ExternalClassSpec] = (),
) -> dict[str, Any]:
  """Generate the complete `algorithms.schema.json` document reflectively from registries."""
  strategies = STRATEGY_REGISTRY if strategy_registry is None else strategy_registry
  pipeline_steps = PIPELINE_STEP_REGISTRY if pipeline_step_registry is None else pipeline_step_registry
  sub_strategies = SUB_STRATEGY_REGISTRY if sub_strategy_registry is None else sub_strategy_registry
  ext_by_slot = external_strategies or {}
  ext_sub_by_slot = external_sub_strategies or {}

  partitioned_strategies = _partition_strategy_registry(strategies)
  strategy_slot_defs: dict[str, dict[str, Any]] = {}
  strategy_param_defs: dict[str, dict[str, Any]] = {}
  for slot_name, proto, def_name in STRATEGY_SLOT_PROTOCOLS:
    slot_schema, slot_params = _build_slot_definition(
      partitioned_strategies[slot_name],
      slot_label=f"{slot_name} strategy",
      class_description=f"Built-in `{proto.__name__}` implementation registered in `STRATEGY_REGISTRY`.",
      params_description=f"Constructor keyword arguments for this `{slot_name}` strategy class.",
      external_specs=ext_by_slot.get(slot_name, ()),
    )
    strategy_slot_defs[def_name] = slot_schema
    strategy_param_defs.update(slot_params)

  partitioned_sub_strategies = _partition_by_protocols(
    sub_strategies, SUB_STRATEGY_PROTOCOLS, "protocols.vrp_strategies"
  )
  sub_strategy_slot_defs: dict[str, dict[str, Any]] = {}
  sub_strategy_param_defs: dict[str, dict[str, Any]] = {}
  for slot_name, proto, def_name in SUB_STRATEGY_PROTOCOLS:
    slot_schema, slot_params = _build_slot_definition(
      partitioned_sub_strategies[slot_name],
      slot_label=f"{slot_name} sub-strategy",
      class_description=f"Built-in `{proto.__name__}` implementation registered in `SUB_STRATEGY_REGISTRY`.",
      params_description=f"Constructor keyword arguments for this `{slot_name}` sub-strategy class.",
      external_specs=ext_sub_by_slot.get(slot_name, ()),
    )
    sub_strategy_slot_defs[def_name] = slot_schema
    sub_strategy_param_defs.update(slot_params)

  pipeline_step_schema, pipeline_param_defs = _build_slot_definition(
    pipeline_steps,
    slot_label="pipeline step",
    class_description="Built-in `GenerationStep` implementation registered in `PIPELINE_STEP_REGISTRY`.",
    params_description="Constructor keyword arguments for this pipeline step class.",
    external_specs=external_pipeline_steps,
  )

  local_search_stage_schema = annotation_to_json_schema(_protocol_types.LocalSearchStage)
  local_search_stage_schema["description"] = (
    "Where the local-search strategy runs inside the generation pipeline (`LocalSearchStage`)."
  )

  shared_variant_properties: dict[str, Any] = {
    "use_gpu": {
      "type": "boolean",
      "description": (
        "Whether the variant needs CuPy/CUDA at run time (the benchmark skips GPU variants when no device is present)."
      ),
    },
    "description": {
      "type": "string",
      "description": "Human-readable label copied into reports.",
    },
  }
  shared_ga_param_properties: dict[str, Any] = {
    "local_search_stage": {
      "$ref": "#/$defs/local_search_stage",
    },
    "population_size": {
      "type": "integer",
      "minimum": 1,
    },
    "mutation_rate": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
    },
    "seed": {
      "type": "integer",
    },
  }

  modular_algorithm_branch: dict[str, Any] = {
    "title": "Modular GeneticAlgorithm",
    "type": "object",
    "required": ["metaheuristic", "use_gpu", "description"],
    "additionalProperties": False,
    "properties": {
      "metaheuristic": {
        "const": "GeneticAlgorithm",
        "description": "Selects the modular strategy-driven `GeneticAlgorithm` pipeline.",
      },
      **shared_variant_properties,
      "params": {
        "type": "object",
        "description": (
          "Modular `GeneticAlgorithm` parameters: `pipeline`, `local_search_stage`, "
          "`population_size`, `mutation_rate`, and `seed`."
        ),
        "additionalProperties": False,
        "properties": {
          "pipeline": {
            "type": "array",
            "description": "Optional list of generational pipeline step specifications (DP11-6, DP11-10).",
            "items": {
              "$ref": "#/$defs/pipeline_step",
            },
          },
          **{
            slot_name: {"$ref": f"#/$defs/{def_name}"}
            for slot_name, _, def_name in SUB_STRATEGY_PROTOCOLS
          },
          **shared_ga_param_properties,
        },
      },
      "strategies": {
        "type": "object",
        "description": "Strategy slot -> class + parameters (Design B). Slots omitted from the pipeline stay `None`.",
        "additionalProperties": False,
        "properties": {
          slot_name: {"$ref": f"#/$defs/{def_name}"} for slot_name, _, def_name in STRATEGY_SLOT_PROTOCOLS
        },
      },
    },
  }

  monolithic_algorithm_branch: dict[str, Any] = {
    "title": "Monolithic Algorithm Class",
    "type": "object",
    "required": ["metaheuristic", "use_gpu", "description", "class_module", "class_name"],
    "additionalProperties": False,
    "properties": {
      "metaheuristic": {
        "type": "string",
        "minLength": 1,
        "not": {"const": "GeneticAlgorithm"},
        "description": "Non-modular metaheuristic identifier loaded dynamically via `class_module`/`class_name`.",
        "examples": ["GeneticAlgorithmFullGPUMonolithic"],
      },
      **shared_variant_properties,
      "class_module": {
        "type": "string",
        "minLength": 1,
        "description": (
          "Dotted module path or config-relative .py file path for monolithic variants "
          "resolved by `AlgorithmConfig.get_class()`."
        ),
        "examples": ["algorithms.archive.full_gpu_monolithic"],
      },
      "class_name": {
        "type": "string",
        "minLength": 1,
        "description": "Class name inside `class_module`.",
        "examples": ["GeneticAlgorithmFullGPUMonolithic"],
      },
      "params": {
        "type": "object",
        "description": "Constructor keyword arguments forwarded to the monolithic algorithm class.",
        "additionalProperties": True,
        "properties": {
          **shared_ga_param_properties,
          "tournament_size": {
            "type": "integer",
            "minimum": 2,
            "description": "Monolithic-only alias of the selection tournament size.",
          },
          "two_opt_iterations": {
            "type": "integer",
            "minimum": 0,
            "description": "Monolithic-only alias of the 2-opt pass cap.",
          },
        },
      },
    },
  }

  defs: dict[str, Any] = {
    "algorithm_config": {
      "type": "object",
      "oneOf": [modular_algorithm_branch, monolithic_algorithm_branch],
    },
    "local_search_stage": local_search_stage_schema,
    **strategy_slot_defs,
    **sub_strategy_slot_defs,
    "pipeline_step": pipeline_step_schema,
    **strategy_param_defs,
    **sub_strategy_param_defs,
    **pipeline_param_defs,
  }

  return {
    "$schema": DRAFT_2020_12,
    "$id": SCHEMA_ID,
    "title": "Algorithm variants configuration",
    "description": (
      "Declarative GA composition (Design B, DP11-1) or monolithic class loading. "
      "Built-in strategies and pipeline steps are partitioned by PEP 544 protocol slots and validated "
      "against constructor signatures via discriminated `oneOf` branches; external custom classes are "
      "supported via `module` + `class` without modifying src/ (DP11-5, BUG-85, BUG-103)."
    ),
    "type": "object",
    "required": ["default_params", "algorithm_configs"],
    "additionalProperties": False,
    "properties": {
      "$schema": {
        "type": "string",
        "description": "Optional JSON Schema reference used by editors for completion and validation.",
      },
      "default_params": {
        "type": "object",
        "description": "Genetic Algorithm defaults applied to every variant unless overridden.",
        "required": ["population_size", "mutation_rate", "seed"],
        "additionalProperties": False,
        "properties": {
          "population_size": {
            "type": "integer",
            "minimum": 1,
            "description": "Number of individuals kept per generation.",
          },
          "mutation_rate": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Per-individual probability of applying the mutation operator.",
          },
          "seed": {
            "type": "integer",
            "description": "Base random seed; repetitions offset it.",
          },
        },
      },
      "algorithm_configs": {
        "type": "object",
        "description": (
          "Variant name -> configuration. Variant names are the labels reported in checkpoints and statistics."
        ),
        "minProperties": 1,
        "additionalProperties": {
          "$ref": "#/$defs/algorithm_config",
        },
      },
    },
    "$defs": defs,
  }


def render_algorithms_schema(
  strategy_registry: Mapping[str, type] | None = None,
  pipeline_step_registry: Mapping[str, type] | None = None,
  *,
  sub_strategy_registry: Mapping[str, type] | None = None,
  external_strategies: Mapping[str, Sequence[ExternalClassSpec]] | None = None,
  external_sub_strategies: Mapping[str, Sequence[ExternalClassSpec]] | None = None,
  external_pipeline_steps: Sequence[ExternalClassSpec] = (),
) -> str:
  """Render `generate_algorithms_schema(...)` as deterministic formatted JSON."""
  schema = generate_algorithms_schema(
    strategy_registry=strategy_registry,
    pipeline_step_registry=pipeline_step_registry,
    sub_strategy_registry=sub_strategy_registry,
    external_strategies=external_strategies,
    external_sub_strategies=external_sub_strategies,
    external_pipeline_steps=external_pipeline_steps,
  )
  return f"{json.dumps(schema, indent=2)}\n"


_REPORT_STYLE_FIELD_METADATA: dict[str, dict[str, Any]] = {
  "display_names": {
    "additionalProperties": {"type": "string"},
    "description": "Optional mapping from canonical JSON algorithm keys to rendered table/figure labels.",
  },
  "decimal_separator": {
    "minLength": 1,
    "description": "Decimal separator used in quantitative LaTeX macros (e.g., '.' or '{,}').",
  },
  "cd_decimal_sep": {
    "minLength": 1,
    "description": "Decimal separator used in Demšar Critical Difference TikZ diagrams (e.g., '.' or ',').",
  },
  "plot_decimal_comma": {
    "description": "Format Matplotlib y-axis gap/cost tick labels with decimal commas when true.",
  },
  "latex_label_prefix": {
    "description": "Optional prefix inserted into LaTeX \\label{tab:<prefix>...} identifiers.",
  },
  "size_thresholds": {
    "prefixItems": [
      {"type": "integer", "minimum": 1},
      {"type": "integer", "minimum": 2},
    ],
    "items": False,
    "minItems": 2,
    "maxItems": 2,
    "description": "Two-element [small_max, medium_max] stratum cutoffs satisfying 0 < small_max < medium_max.",
  },
  "show_size_guides": {
    "description": "Draw vertical problem-size stratum guides on scaling plots when true.",
  },
  "cd_quality_label": {
    "description": "Metric label for optimality-gap Demšar Critical Difference diagrams.",
  },
  "cd_cost_label": {
    "description": "Fallback metric label for cost Demšar CD diagrams when reference optima are absent.",
  },
  "cd_time_label": {
    "description": "Metric label for execution-time Demšar Critical Difference diagrams.",
  },
  "cd_rank_label": {
    "description": "Axis label placed at the right end of Demšar CD rank axes.",
  },
  "plot_title": {
    "description": "Main suptitle rendered above the dual-panel scaling figure.",
  },
  "plot_time_ylabel": {
    "description": "Y-axis label for the top (execution time) scaling panel.",
  },
  "plot_gap_ylabel": {
    "description": "Y-axis label for the bottom (optimality gap) scaling panel.",
  },
  "plot_cost_ylabel": {
    "description": "Fallback y-axis label for the bottom panel when reference optima are absent.",
  },
  "plot_xlabel": {
    "description": "Shared x-axis label for problem dimension n.",
  },
  "plot_band_label": {
    "description": "Optional legend label for the empirical dispersion band (null uses metric default).",
  },
  "algorithm_order": {
    "description": "Optional explicit algorithm ordering applied across tables, macros, CD diagrams, and plots.",
  },
}


def _report_style_field_schema(field_name: str, annotation: Any) -> dict[str, Any]:
  """Reflectively convert one ``ReportStyle`` dataclass field into a strict JSON Schema property."""
  prop_schema = annotation_to_json_schema(annotation)
  if field_name == "algorithm_order":
    prop_schema = {
      "anyOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"},
      ]
    }
  advisory = _REPORT_STYLE_FIELD_METADATA.get(field_name)
  if advisory:
    for key, value in advisory.items():
      if key not in prop_schema:
        prop_schema[key] = value
  return prop_schema


def generate_report_style_schema() -> dict[str, Any]:
  """Reflectively generate Draft 2020-12 JSON Schema for ``ReportStyle`` JSON files."""
  type_hints = get_type_hints(ReportStyle, globalns={"Mapping": Mapping, "Sequence": Sequence})
  properties: dict[str, Any] = {
    "$schema": {
      "type": "string",
      "description": "Optional JSON Schema reference used by editors for completion and validation.",
    }
  }
  for dc_field in dataclasses.fields(ReportStyle):
    ann = type_hints.get(dc_field.name, dc_field.type)
    properties[dc_field.name] = _report_style_field_schema(dc_field.name, ann)

  return {
    "$schema": DRAFT_2020_12,
    "$id": REPORT_STYLE_SCHEMA_ID,
    "title": "Benchmark report presentation style configuration",
    "description": (
      "External presentation settings injected via --report-style or load_report_style(). "
      "Generated reflectively from benchmarking.reporting.style.ReportStyle so schema "
      "and runtime validation stay locked to a single source of truth."
    ),
    "type": "object",
    "additionalProperties": False,
    "properties": properties,
  }


def render_report_style_schema() -> str:
  """Render `generate_report_style_schema()` as deterministic formatted JSON."""
  return f"{json.dumps(generate_report_style_schema(), indent=2)}\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate or verify algorithms and report_style JSON schemas (with optional external experiment ops)"
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Target algorithms schema JSON file path (default: configs/schemas/algorithms.schema.json).",
  )
  parser.add_argument(
    "--extra-module",
    action="append",
    default=[],
    metavar="MODULE",
    help=(
      "Repeatable dotted Python module (or .py path) exporting custom strategies/pipeline steps "
      "to include as typed oneOf branches in the generated schema (requires --output)."
    ),
  )
  parser.add_argument(
    "--extra-file",
    action="append",
    type=Path,
    default=[],
    metavar="PATH",
    help=(
      "Repeatable Python .py file exporting custom strategies/pipeline steps; emits config-relative "
      "module paths relative to --output's directory (requires --output)."
    ),
  )
  parser.add_argument(
    "--report-style-output",
    type=Path,
    default=None,
    help="Target report_style schema JSON file path (default: configs/schemas/report_style.schema.json).",
  )
  parser.add_argument(
    "--check",
    action="store_true",
    help="Verify that target schema files match generated schemas without modifying them.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point to write or verify reflective JSON schemas."""
  args = _parse_args(argv)
  has_extras = bool(args.extra_module or args.extra_file)
  if has_extras and args.output is None:
    print(
      "Error: --extra-file and --extra-module require an explicit --output path so the core "
      "configs/schemas/algorithms.schema.json stays decoupled from user experiments.",
      file=sys.stderr,
    )
    return 2
  if has_extras and args.output is not None and args.output.resolve() == DEFAULT_SCHEMA_PATH.resolve():
    print(
      f"Error: refusing to write --extra-file/--extra-module classes into core schema {DEFAULT_SCHEMA_PATH}. "
      "Pass an experiment-local --output path instead.",
      file=sys.stderr,
    )
    return 2

  external_strategies: dict[str, list[ExternalClassSpec]] = {}
  external_pipeline_steps: list[ExternalClassSpec] = []
  if has_extras:
    external_strategies, external_pipeline_steps = discover_external_extensions(
      extra_modules=args.extra_module,
      extra_files=args.extra_file,
      schema_output_dir=args.output.parent if args.output is not None else None,
    )

  targets: list[tuple[Path, str]] = []
  if args.output is not None:
    targets.append(
      (
        args.output,
        render_algorithms_schema(
          external_strategies=external_strategies or None,
          external_pipeline_steps=external_pipeline_steps,
        ),
      )
    )
    if args.report_style_output is not None:
      targets.append((args.report_style_output, render_report_style_schema()))
  elif args.report_style_output is not None:
    targets.append((args.report_style_output, render_report_style_schema()))
  else:
    targets.append((DEFAULT_SCHEMA_PATH, render_algorithms_schema()))
    targets.append((DEFAULT_REPORT_STYLE_SCHEMA_PATH, render_report_style_schema()))

  if args.check:
    drifted = False
    for path, rendered in targets:
      if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
        print(f"Schema drift detected in {path}", file=sys.stderr)
        drifted = True
    return 1 if drifted else 0

  for path, rendered in targets:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
