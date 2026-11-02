"""Unit tests verifying structural parity and import contracts for algorithm configurations.

Compares production `configs/algorithms.json` against the test fixture
`tests/fixtures/configs/algorithms.json` to prevent schema/contract drift.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from algorithms.strategies import STRATEGY_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_ALGORITHMS_JSON = PROJECT_ROOT / "configs" / "algorithms.json"
FIXTURE_ALGORITHMS_JSON = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "algorithms.json"


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
