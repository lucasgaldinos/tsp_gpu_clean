"""Reflective JSON Schema generator for `configs/schemas/algorithms.schema.json`.

Derives per-class parameter schemas and conditional `allOf` branches directly from
`STRATEGY_REGISTRY` and `PIPELINE_STEP_REGISTRY` via `inspect.signature`,
`typing.get_type_hints`, and `typing.get_args` (`BUG-85` / Task 13.27(iv)).

Why reflection instead of a hand-maintained lookup table:
  A single shared `params` object across all 17 strategies and 3 pipeline steps
  accepted invalid combinations in the schema (e.g. `LocalSearchCPUTwoOpt` with
  `mode="batch"`) while constructor guards rejected them at runtime. Generating
  `$defs/params_<snake_name>` and `allOf` `if`/`then` branches from constructor
  signatures and `Literal` / `TypeAliasType` annotations ensures the schema and
  runtime share one source of truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

import protocols.algorithm_strategies as _strategy_protocols
import protocols.types as _protocol_types
from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY
from algorithms.strategies import STRATEGY_REGISTRY
from algorithms.strategies.base import StrategyBase
from benchmarking.analysis.report_style import ReportStyle

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://github.com/lucasgaldinos/gpu_accelerated_clean/configs/schemas/algorithms.schema.json"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "configs" / "schemas" / "algorithms.schema.json"
REPORT_STYLE_SCHEMA_ID = (
  "https://github.com/lucasgaldinos/gpu_accelerated_clean/configs/schemas/report_style.schema.json"
)
DEFAULT_REPORT_STYLE_SCHEMA_PATH = (
  Path(__file__).resolve().parents[3] / "configs" / "schemas" / "report_style.schema.json"
)

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


def unwrap_type_annotation(annotation: Any) -> Any:
  """Recursively unwrap PEP 695 `TypeAliasType` and `Annotated` type wrappers."""
  while True:
    if isinstance(annotation, TypeAliasType):
      annotation = annotation.__value__
      continue
    if get_origin(annotation) is Annotated:
      annotation = get_args(annotation)[0]
      continue
    return annotation


def _is_strategy_type(tp: Any) -> bool:
  """Return True when `tp` is a strategy protocol or `StrategyBase` subclass."""
  unwrapped = unwrap_type_annotation(tp)
  if not isinstance(unwrapped, type):
    return False
  if issubclass(unwrapped, StrategyBase):
    return True
  return unwrapped.__module__ == _strategy_protocols.__name__ and bool(getattr(unwrapped, "_is_protocol", False))


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
  if _is_strategy_type(unwrapped):
    return {"$ref": "#/$defs/strategy_slot"}

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
      return {"$ref": "#/$defs/strategy_slot"}

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


def _build_slot_definition(
  registry: Mapping[str, type],
  *,
  class_description: str,
  params_description: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Build a conditional slot schema (`allOf` if/then) and its `$defs/params_*` definitions."""
  sorted_items = sorted(registry.items(), key=lambda item: item[0])
  all_of_branches: list[dict[str, Any]] = []
  param_defs: dict[str, dict[str, Any]] = {}

  for class_name, cls in sorted_items:
    def_key = param_def_name(class_name)
    param_defs[def_key] = build_class_params_schema(cls)
    all_of_branches.append(
      {
        "if": {
          "not": {"required": ["module"]},
          "required": ["class"],
          "properties": {
            "class": {
              "const": class_name,
            },
          },
        },
        "then": {
          "properties": {
            "params": {
              "$ref": f"#/$defs/{def_key}",
            },
          },
        },
      }
    )

  slot_schema: dict[str, Any] = {
    "type": "object",
    "required": ["class"],
    "additionalProperties": False,
    "properties": {
      "module": {
        "type": "string",
        "minLength": 1,
        "description": (
          "Optional dotted Python module path for loading an external custom class "
          "outside the built-in registry without modifying src/."
        ),
      },
      "class": {
        "type": "string",
        "minLength": 1,
        "description": class_description,
      },
      "params": {
        "type": "object",
        "description": params_description,
      },
    },
    "if": {
      "not": {"required": ["module"]},
    },
    "then": {
      "properties": {
        "class": {
          "enum": [class_name for class_name, _ in sorted_items],
        },
      },
    },
    "allOf": all_of_branches,
  }
  return slot_schema, param_defs


def generate_algorithms_schema(
  strategy_registry: Mapping[str, type] | None = None,
  pipeline_step_registry: Mapping[str, type] | None = None,
) -> dict[str, Any]:
  """Generate the complete `algorithms.schema.json` document reflectively from registries."""
  strategies = STRATEGY_REGISTRY if strategy_registry is None else strategy_registry
  pipeline_steps = PIPELINE_STEP_REGISTRY if pipeline_step_registry is None else pipeline_step_registry

  strategy_slot_schema, strategy_param_defs = _build_slot_definition(
    strategies,
    class_description="Key of `STRATEGY_REGISTRY`; the slot prefix identifies the stage.",
    params_description="Constructor keyword arguments validated per selected strategy class via `allOf`.",
  )
  pipeline_step_schema, pipeline_param_defs = _build_slot_definition(
    pipeline_steps,
    class_description="Key of `PIPELINE_STEP_REGISTRY`.",
    params_description="Constructor keyword arguments validated per selected pipeline step class via `allOf`.",
  )

  local_search_stage_schema = annotation_to_json_schema(_protocol_types.LocalSearchStage)
  local_search_stage_schema["description"] = (
    "Where the local-search strategy runs inside the generation pipeline (`LocalSearchStage`)."
  )

  defs: dict[str, Any] = {
    "algorithm_config": {
      "type": "object",
      "required": ["metaheuristic", "use_gpu", "description"],
      "additionalProperties": False,
      "properties": {
        "metaheuristic": {
          "type": "string",
          "description": (
            "Metaheuristic identifier; `GeneticAlgorithm` selects the modular pipeline, "
            "any other value requires `class_module`/`class_name`."
          ),
          "examples": ["GeneticAlgorithm", "GeneticAlgorithmFullGPUMonolithic"],
        },
        "use_gpu": {
          "type": "boolean",
          "description": (
            "Whether the variant needs CuPy/CUDA at run time "
            "(the benchmark skips GPU variants when no device is present)."
          ),
        },
        "description": {
          "type": "string",
          "description": "Human-readable label copied into reports.",
        },
        "class_module": {
          "type": "string",
          "description": "Dotted module path for monolithic variants resolved by `AlgorithmConfig.get_class()`.",
          "examples": ["algorithms.archive.full_gpu_monolithic"],
        },
        "class_name": {
          "type": "string",
          "description": "Class name inside `class_module`.",
          "examples": ["GeneticAlgorithmFullGPUMonolithic"],
        },
        "params": {
          "type": "object",
          "description": (
            "Algorithm-level parameters. Modular pipelines accept "
            "`pipeline`, `local_search_stage`, `population_size`, `mutation_rate` and `seed`."
          ),
          "properties": {
            "pipeline": {
              "type": "array",
              "description": "Optional list of generational pipeline step specifications (DP11-6, DP11-10).",
              "items": {
                "$ref": "#/$defs/pipeline_step",
              },
            },
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
          "additionalProperties": False,
        },
        "strategies": {
          "type": "object",
          "description": "Strategy slot -> class + parameters (Design B). Slots omitted from the pipeline stay `None`.",
          "additionalProperties": False,
          "properties": {
            "initialization": {"$ref": "#/$defs/strategy_slot"},
            "selection": {"$ref": "#/$defs/strategy_slot"},
            "crossover": {"$ref": "#/$defs/strategy_slot"},
            "mutation": {"$ref": "#/$defs/strategy_slot"},
            "local_search": {"$ref": "#/$defs/strategy_slot"},
            "evaluation": {"$ref": "#/$defs/strategy_slot"},
          },
        },
      },
      "allOf": [
        {
          "if": {
            "properties": {
              "metaheuristic": {
                "not": {
                  "const": "GeneticAlgorithm",
                },
              },
            },
          },
          "then": {
            "required": ["class_module", "class_name"],
          },
        },
      ],
    },
    "local_search_stage": local_search_stage_schema,
    "strategy_slot": strategy_slot_schema,
    "pipeline_step": pipeline_step_schema,
    **strategy_param_defs,
    **pipeline_param_defs,
  }

  return {
    "$schema": DRAFT_2020_12,
    "$id": SCHEMA_ID,
    "title": "Algorithm variants configuration",
    "description": (
      "Declarative GA composition (Design B, DP11-1) or monolithic class loading. "
      "Every strategy class named here must exist in algorithms.strategies.STRATEGY_REGISTRY, "
      "and every parameter key must be accepted by the class constructor (DP11-5, BUG-85)."
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
) -> str:
  """Render `generate_algorithms_schema(...)` as deterministic formatted JSON."""
  schema = generate_algorithms_schema(
    strategy_registry=strategy_registry,
    pipeline_step_registry=pipeline_step_registry,
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
      "Generated reflectively from benchmarking.analysis.report_style.ReportStyle so schema "
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
    description="Generate or verify configs/schemas/{algorithms,report_style}.schema.json"
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Target algorithms schema JSON file path (default: configs/schemas/algorithms.schema.json).",
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
  targets: list[tuple[Path, str]] = []
  if args.output is not None:
    targets.append((args.output, render_algorithms_schema()))
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
