"""Reflective class/module resolution, strategy-slot validation, and type-hint helpers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, TypeAliasType, Union, get_args, get_origin, get_type_hints

from algorithms.pipeline_steps import StageAnchor

VALID_STRATEGY_SLOTS: frozenset[str] = frozenset(
  {"initialization", "selection", "crossover", "mutation", "evaluation", "local_search"}
)
SUB_STRATEGY_SLOTS: frozenset[str] = frozenset({"penalty_policy", "decoder", "bin_packing"})
MODULAR_ALGO_PARAMS: frozenset[str] = frozenset(
  {"pipeline", "local_search_stage", "population_size", "mutation_rate", "seed"} | SUB_STRATEGY_SLOTS
)
VALID_STAGE_ANCHORS: frozenset[str] = frozenset(get_args(getattr(StageAnchor, "__value__", StageAnchor)))
_FILE_MODULE_CACHE: dict[tuple[Path, int, int], types.ModuleType] = {}


def is_file_module_ref(module_ref: str) -> bool:
  """Return True when ``module_ref`` specifies a ``.py`` file path instead of a dotted import path."""
  return module_ref.endswith(".py") or "/" in module_ref or "\\" in module_ref


def load_external_module(module_ref: str, *, base_dir: Path | None = None) -> types.ModuleType:
  """Load a Python module from either a dotted module name or a config-relative ``.py`` file path."""
  if not is_file_module_ref(module_ref):
    return importlib.import_module(module_ref)

  candidate = Path(module_ref)
  if candidate.is_absolute():
    resolved = candidate.resolve()
  elif base_dir is not None:
    resolved = (base_dir / candidate).resolve()
  else:
    resolved = candidate.resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f"External module file not found: '{module_ref}' (resolved to '{resolved}')")

  stat = resolved.stat()
  cache_key = (resolved, stat.st_mtime_ns, stat.st_size)
  cached = _FILE_MODULE_CACHE.get(cache_key)
  if cached is not None:
    return cached

  stem_slug = re.sub(r"\W+", "_", resolved.stem).strip("_") or "module"
  digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
  mod_name = f"_tsp_external_{stem_slug}_{digest}"

  spec = importlib.util.spec_from_file_location(mod_name, resolved)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load Python module specification from '{resolved}'.")

  module = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    sys.modules.pop(mod_name, None)
    raise

  _FILE_MODULE_CACHE[cache_key] = module
  return module


def unwrap_type_annotation(annotation: Any) -> Any:
  """Recursively unwrap PEP 695 ``TypeAliasType`` and ``Annotated`` type wrappers."""
  if isinstance(annotation, TypeAliasType):
    return unwrap_type_annotation(annotation.__value__)
  if get_origin(annotation) is Annotated:
    return unwrap_type_annotation(get_args(annotation)[0])
  return annotation


def _slot_protocol_map() -> dict[str, type]:
  """Return the ``@runtime_checkable`` protocol expected for each core GA strategy slot."""
  from protocols.algorithm_strategies import (
    CrossoverStrategy,
    EvaluationStrategy,
    InitializationStrategy,
    LocalSearchStrategy,
    MutationStrategy,
    SelectionStrategy,
  )

  return {
    "initialization": InitializationStrategy, "selection": SelectionStrategy,
    "crossover": CrossoverStrategy, "mutation": MutationStrategy,
    "evaluation": EvaluationStrategy, "local_search": LocalSearchStrategy,
  }


def _sub_strategy_protocol_map() -> dict[str, type]:
  """Return the ``@runtime_checkable`` protocol expected for each composable sub-strategy slot."""
  from protocols.vrp_strategies import BinPackingStrategy, PenaltyPolicy, RouteDecoder

  return {"penalty_policy": PenaltyPolicy, "decoder": RouteDecoder, "bin_packing": BinPackingStrategy}


def _resolve_init_type_hints(cls: type) -> dict[str, Any]:
  """Resolve ``__init__`` parameter annotations including ``TYPE_CHECKING`` protocol imports."""
  init_fn: Any = next(
    (b.__dict__["__init__"] for b in cls.__mro__ if b is not object and "__init__" in b.__dict__),
    None,
  )
  if init_fn is None:
    return {}

  import protocols.algorithm_strategies as strategy_protocols
  import protocols.types as protocol_types
  import protocols.vrp_strategies as vrp_strategies

  globalns: dict[str, Any] = {
    "Any": Any, "Sequence": Sequence, "Mapping": Mapping,
    **vars(protocol_types), **vars(strategy_protocols), **vars(vrp_strategies),
  }
  mod = sys.modules.get(cls.__module__)
  if mod is not None:
    globalns.update(vars(mod))
  try:
    return get_type_hints(init_fn, globalns=globalns, include_extras=True)
  except Exception:
    return {}


def _infer_nested_strategy_slot(param_name: str, annotation: Any) -> str | None:
  """Infer the expected core or sub-strategy slot for a nested constructor parameter."""
  all_protocols = {**_slot_protocol_map(), **_sub_strategy_protocol_map()}
  if annotation is not None and annotation is not inspect.Parameter.empty:
    unwrapped = unwrap_type_annotation(annotation)
    origin = get_origin(unwrapped)
    members = (
      [unwrap_type_annotation(m) for m in get_args(unwrapped) if unwrap_type_annotation(m) is not type(None)]
      if origin in (Union, types.UnionType)
      else [unwrapped]
    )
    matched: set[str] = {
      slot_name
      for member in members
      if isinstance(member, type)
      for slot_name, proto in all_protocols.items()
      if member is proto or issubclass(member, proto)
    }
    if len(matched) == 1:
      return next(iter(matched))

  if param_name in VALID_STRATEGY_SLOTS or param_name in SUB_STRATEGY_SLOTS:
    return param_name
  return "evaluation" if param_name == "re_evaluate" else None


def _resolve_class_spec(
  spec: Mapping[str, Any],
  registry: Mapping[str, type],
  registry_name: str,
  slot: str | None = None,
  base_dir: Path | None = None,
) -> type:
  """Resolve a built-in registry or external ``{"module", "class"}`` specification to a class."""
  class_name = spec.get("class")
  if not isinstance(class_name, str) or not class_name.strip():
    slot_ctx = f" in slot '{slot}'" if slot else ""
    raise ValueError(f"Missing or invalid 'class' name for {registry_name}{slot_ctx}.")

  module_name = spec.get("module")
  if module_name is not None:
    if not isinstance(module_name, str) or not module_name.strip():
      raise ValueError(f"Invalid 'module' value {module_name!r} for {registry_name} '{class_name}'.")
    mod = load_external_module(module_name, base_dir=base_dir)
    loaded_cls = getattr(mod, class_name, None)
    if not isinstance(loaded_cls, type):
      raise ValueError(f"External {registry_name} '{class_name}' not found as a class in module '{module_name}'.")
    return loaded_cls

  if class_name not in registry:
    slot_ctx = f" in slot '{slot}'" if slot else ""
    raise ValueError(
      f"Unknown {registry_name} class '{class_name}'{slot_ctx}. "
      f"Available: {sorted(registry.keys())} (or provide 'module' to import an external class)"
    )
  return registry[class_name]


def _check_component_contract(target_cls: type, registry_name: str, slot: str | None) -> str | None:
  """Verify strategy/sub-strategy protocol conformance or pipeline step `run` contract."""
  if slot is not None and registry_name in ("strategy", "sub-strategy"):
    expected_protocol = {**_slot_protocol_map(), **_sub_strategy_protocol_map()}.get(slot)
    if expected_protocol is not None and not issubclass(target_cls, expected_protocol):
      return f"class '{target_cls.__name__}' does not implement {expected_protocol.__name__} required for slot '{slot}'"
  if registry_name == "pipeline step" and not callable(getattr(target_cls, "run", None)):
    return f"pipeline step class '{target_cls.__name__}' must define a callable 'run(state, timer)' method"
  return None


def _resolve_nested_registry(
  nested_slot: str | None,
  spec: Mapping[str, Any],
  strategy_registry: Mapping[str, type] | None,
  sub_strategy_registry: Mapping[str, type] | None,
) -> tuple[Mapping[str, type], str] | None:
  """Select the matching registry and label for a nested component spec dict."""
  if nested_slot in SUB_STRATEGY_SLOTS:
    if sub_strategy_registry is None:
      from algorithms.strategies import SUB_STRATEGY_REGISTRY

      return SUB_STRATEGY_REGISTRY, "sub-strategy"
    return sub_strategy_registry, "sub-strategy"
  if sub_strategy_registry is not None and spec.get("class") in sub_strategy_registry:
    return sub_strategy_registry, "sub-strategy"
  if strategy_registry is not None and (
    nested_slot is not None or "module" in spec or spec.get("class") in strategy_registry
  ):
    return strategy_registry, "strategy"
  return None


def _validate_class_spec(
  spec: Mapping[str, Any] | None, registry: Mapping[str, type], registry_name: str, *,
  context_label: str, slot: str | None = None, strategy_registry: Mapping[str, type] | None = None,
  sub_strategy_registry: Mapping[str, type] | None = None, base_dir: Path | None = None,
) -> list[str]:
  """Validate class resolution, slot-protocol conformance, constructor params, and nested specs."""
  if not spec or ("class" not in spec and "module" not in spec):
    return []
  if slot is not None and registry_name == "strategy" and slot not in VALID_STRATEGY_SLOTS:
    return [f"{context_label}: unknown strategy slot '{slot}'. Valid slots: {sorted(VALID_STRATEGY_SLOTS)}"]
  if slot is not None and registry_name == "sub-strategy" and slot not in SUB_STRATEGY_SLOTS:
    return [f"{context_label}: unknown sub-strategy slot '{slot}'. Valid slots: {sorted(SUB_STRATEGY_SLOTS)}"]

  errors: list[str] = []
  try:
    target_cls = _resolve_class_spec(spec, registry, registry_name, slot=slot, base_dir=base_dir)
  except (ImportError, ValueError, AttributeError, FileNotFoundError, OSError) as exc:
    return [f"{context_label}: {exc}"]

  contract_err = _check_component_contract(target_cls, registry_name, slot)
  if contract_err is not None:
    errors.append(f"{context_label}: {contract_err}")

  if registry_name == "pipeline step":
    anchor_attr = getattr(target_cls, "stage_anchor", None)
    if anchor_attr is not None and not isinstance(anchor_attr, property):
      if not isinstance(anchor_attr, str) or anchor_attr not in VALID_STAGE_ANCHORS:
        errors.append(
          f"{context_label}: pipeline step class '{target_cls.__name__}' has invalid stage_anchor "
          f"{anchor_attr!r} (expected one of {sorted(VALID_STAGE_ANCHORS)})"
        )

  raw_params = spec.get("params", {})
  if not isinstance(raw_params, Mapping):
    return [f"{context_label}: 'params' must be a mapping, got {type(raw_params).__name__}"]

  sig, hints = inspect.signature(target_cls), _resolve_init_type_hints(target_cls)
  has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
  for k, v in raw_params.items():
    if not has_var_kw and k not in sig.parameters:
      errors.append(
        f"{context_label}: {registry_name} class '{target_cls.__name__}' does not accept parameter '{k}'. "
        f"Accepted parameters: {list(sig.parameters.keys())}"
      )
    nested_slot = _infer_nested_strategy_slot(k, hints.get(k))
    if isinstance(v, Mapping) and ("class" in v or "module" in v):
      target = _resolve_nested_registry(nested_slot, v, strategy_registry, sub_strategy_registry)
      if target is not None:
        nested_reg, nested_kind = target
        errors.extend(
          _validate_class_spec(
            v, nested_reg, nested_kind, context_label=f"{context_label}.params.{k}",
            slot=nested_slot, strategy_registry=strategy_registry,
            sub_strategy_registry=sub_strategy_registry, base_dir=base_dir,
          )
        )
  return errors


def build_component_spec(
  spec: Mapping[str, Any] | None, registry: Mapping[str, type], registry_name: str, *,
  resolved_seed: Any, slot: str | None = None, base_dir: Path | None = None,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
  default_kwargs: Mapping[str, Any] | None = None,
  strategy_registry: Mapping[str, type] | None = None,
  sub_strategy_registry: Mapping[str, type] | None = None,
) -> Any:
  """Instantiate one strategy, sub-strategy, or pipeline step including nested component params."""
  if not spec or ("class" not in spec and "module" not in spec):
    return None
  class_name = spec["class"]
  target_cls = _resolve_class_spec(spec, registry, registry_name, slot=slot, base_dir=base_dir)
  contract_err = _check_component_contract(target_cls, registry_name, slot)
  if contract_err is not None:
    raise ValueError(contract_err)

  params = dict(spec.get("params", {}))
  sig, hints = inspect.signature(target_cls), _resolve_init_type_hints(target_cls)
  has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

  if slot and strategy_overrides and slot in strategy_overrides:
    for k, v in strategy_overrides[slot].items():
      if not has_var_kw and k not in sig.parameters:
        raise ValueError(
          f"Strategy class '{class_name}' in slot '{slot}' does not accept parameter '{k}'. "
          f"Accepted parameters: {list(sig.parameters.keys())}"
        )
      params[k] = v

  if default_kwargs:
    for k, v in default_kwargs.items():
      if k in sig.parameters and k not in params and v is not None:
        params[k] = v

  for k, v in list(params.items()):
    nested_slot = _infer_nested_strategy_slot(k, hints.get(k))
    if isinstance(v, Mapping) and ("class" in v or "module" in v):
      target = _resolve_nested_registry(nested_slot, v, strategy_registry, sub_strategy_registry)
      if target is not None:
        nested_reg, nested_kind = target
        params[k] = build_component_spec(
          v, nested_reg, nested_kind, resolved_seed=resolved_seed, slot=nested_slot,
          base_dir=base_dir, strategy_registry=strategy_registry, sub_strategy_registry=sub_strategy_registry,
        )

  if "seed" in sig.parameters and "seed" not in params:
    params["seed"] = resolved_seed
  if not has_var_kw:
    for k in params:
      if k not in sig.parameters:
        raise ValueError(
          f"{registry_name} class '{class_name}' does not accept parameter '{k}'. "
          f"Accepted parameters: {list(sig.parameters.keys())}"
        )
  return target_cls(**params)


__all__: list[str] = [
  "MODULAR_ALGO_PARAMS", "SUB_STRATEGY_SLOTS", "VALID_STAGE_ANCHORS", "VALID_STRATEGY_SLOTS",
  "_infer_nested_strategy_slot", "_resolve_class_spec", "_resolve_init_type_hints",
  "_slot_protocol_map", "_sub_strategy_protocol_map", "_validate_class_spec",
  "build_component_spec", "is_file_module_ref", "load_external_module", "unwrap_type_annotation",
]
