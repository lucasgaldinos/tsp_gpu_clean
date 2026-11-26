"""Unit tests for `CrossoverGPUPartiallyMapped` (`pmx_crossover.cu`).

Verifies 100% bit-identity between `CrossoverGPUPartiallyMapped` and
`CrossoverPartiallyMapped` across open permutations, closed depot-bounded routes,
boundary dimensions, zero-filled warm-up resilience (`BUG-24`), telemetry counters,
and `AlgorithmConfig.instantiate` + JSON Schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import numpy as np
import pytest

from algorithms.strategies import STRATEGY_REGISTRY
from algorithms.strategies.crossover import CrossoverGPUPartiallyMapped, CrossoverPartiallyMapped
from benchmarking.runner.config_loader import load_algorithms_config
from data_models.problem import Problem
from protocols.algorithm_strategies import CrossoverStrategy
from protocols.backend import cupy_backend
from protocols.problem_context import ProblemContext

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = PROJECT_ROOT / "configs" / "schemas" / "algorithms.schema.json"


def _cupy() -> Any:
  """Return CuPy module or skip if unavailable."""
  cp = cupy_backend()
  if cp is None:
    pytest.skip("CUDA device or CuPy unavailable")
  return cp


def _make_dummy_problem(n: int) -> Problem:
  """Create a dummy problem for context initialization."""
  return Problem(
    name=f"dummy_pmx_{n}",
    dimension=n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.zeros((n, 2), dtype=np.float64),
    distances=None,
    capacity=None,
    demands=None,
  )


def test_gpu_pmx_registry_protocol_schema_and_instantiate(tmp_path: Path) -> None:
  """WHAT: Registry export, protocol conformance, JSON schema slot, and instantiation.

  WHY: Promoting `CrossoverGPUPartiallyMapped` into `src/` requires it to live in
  `STRATEGY_REGISTRY`, land strictly in `strategy_slot_crossover` of the generated
  schema, and instantiate cleanly via `AlgorithmConfig.instantiate`.

  EXPECTED: Registered in `STRATEGY_REGISTRY`, satisfies `CrossoverStrategy`, passes
  schema validation only in `crossover`, and instantiates `GeneticAlgorithm`.
  """
  assert STRATEGY_REGISTRY["CrossoverGPUPartiallyMapped"] is CrossoverGPUPartiallyMapped
  strategy = CrossoverGPUPartiallyMapped(seed=42)
  assert isinstance(strategy, CrossoverStrategy)

  schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
  cx_slot = schema["$defs"]["strategy_slot_crossover"]
  cx_classes = {
    branch["properties"]["class"]["const"] for branch in cx_slot["oneOf"] if "const" in branch["properties"]["class"]
  }
  assert "CrossoverGPUPartiallyMapped" in cx_classes

  mut_slot = schema["$defs"]["strategy_slot_mutation"]
  mut_classes = {
    branch["properties"]["class"]["const"] for branch in mut_slot["oneOf"] if "const" in branch["properties"]["class"]
  }
  assert "CrossoverGPUPartiallyMapped" not in mut_classes

  raw_cfg: dict[str, Any] = {
    "$schema": "./schemas/algorithms.schema.json",
    "default_params": {"population_size": 16, "mutation_rate": 0.03, "seed": 42},
    "algorithm_configs": {
      "PMX_GPU_Test": {
        "metaheuristic": "GeneticAlgorithm",
        "use_gpu": True,
        "description": "Test built-in CrossoverGPUPartiallyMapped config",
        "params": {"local_search_stage": "none"},
        "strategies": {
          "initialization": {"class": "InitializationRandom", "params": {"return_gpu": True}},
          "selection": {"class": "SelectionGPUTournament", "params": {"k": 3}},
          "crossover": {"class": "CrossoverGPUPartiallyMapped", "params": {"seed": 42}},
          "mutation": {"class": "MutationGPUSwap"},
          "local_search": {"class": "LocalSearchNoOp"},
          "evaluation": {"class": "EvaluationGPUCost", "params": {"return_gpu": True}},
        },
      }
    },
  }
  jsonschema.validate(instance=raw_cfg, schema=schema)

  bad_cfg = json.loads(json.dumps(raw_cfg))
  bad_cfg["algorithm_configs"]["PMX_GPU_Test"]["strategies"]["mutation"] = {"class": "CrossoverGPUPartiallyMapped"}
  with pytest.raises(jsonschema.ValidationError):
    jsonschema.validate(instance=bad_cfg, schema=schema)

  cfg_file = tmp_path / "algorithms.json"
  cfg_file.write_text(json.dumps(raw_cfg), encoding="utf-8")
  ga_params, alg_configs = load_algorithms_config(cfg_file)
  ga = alg_configs["PMX_GPU_Test"].instantiate(ga_params)
  assert type(ga.crossover) is CrossoverGPUPartiallyMapped


@pytest.mark.gpu
@pytest.mark.parametrize("n", [52, 100, 200, 500, 1000, 2000])
def test_gpu_pmx_crossover_matches_cpu_bit_identical(n: int) -> None:
  """WHAT: Compare GPU and CPU PMX crossover across problem sizes.

  WHY: Ensures 100% bit-identity between `CrossoverGPUPartiallyMapped` and
  `CrossoverPartiallyMapped` on disjoint pairs with identical host-sampled cuts.

  EXPECTED: Identical 2D offspring permutation arrays for every `n`.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  pop_size = 64

  rng = np.random.default_rng(300 + n)
  base_tour = np.arange(n, dtype=np.int32)
  population = np.stack([rng.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=77)
  cpu_offspring = CrossoverPartiallyMapped().crossover_batch(population.copy(), context=cpu_ctx)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=77)
  gpu_offspring = CrossoverGPUPartiallyMapped().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(
    np.asarray(gpu_offspring),
    np.asarray(cpu_offspring),
    err_msg=f"Bit divergence in CrossoverGPUPartiallyMapped at n={n}",
  )
  for row in np.asarray(gpu_offspring):
    assert np.array_equal(np.sort(row), base_tour)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [14, 52, 100])
def test_gpu_pmx_crossover_preserves_closed_route_depot_endpoints(n: int) -> None:
  """WHAT: Crossover on closed tours `[0, ..., 0]` with fixed depot endpoints.

  WHY: Depot endpoints must stay at index 0 and index `n-1`, and offspring must
  match the CPU implementation bit-for-bit.

  EXPECTED: Endpoints untouched, interior valid permutation, bit-identical to CPU.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  pop_size = 16

  rng = np.random.default_rng(400 + n)
  inner_cities = np.arange(1, n - 1, dtype=np.int32)
  routes = [
    np.concatenate([[0], rng.permutation(inner_cities).astype(np.int32), [0]]).astype(np.int32) for _ in range(pop_size)
  ]
  population = np.stack(routes).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=88)
  cpu_offspring = CrossoverPartiallyMapped().crossover_batch(population.copy(), context=cpu_ctx)

  gpu_ctx = ProblemContext(problem, xp=cp, seed=88)
  gpu_offspring = CrossoverGPUPartiallyMapped().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_offspring), np.asarray(cpu_offspring))
  for child in np.asarray(gpu_offspring):
    assert child[0] == 0
    assert child[-1] == 0
    assert sorted(child[1:-1].tolist()) == list(range(1, n - 1))


@pytest.mark.gpu
def test_gpu_pmx_crossover_preserves_gpu_residence_and_telemetry() -> None:
  """WHAT: GPU-resident parent arrays stay on device and record accurate telemetry.

  EXPECTED: Device input yields CuPy array with `d2h_bytes == 0`, `h2d_bytes == 8 * N`,
  `kernel_launches == 1`; host input records parent H2D and offspring D2H bytes.
  """
  cp = _cupy()
  n = 30
  pop_size = 8
  problem = _make_dummy_problem(n)
  rng = np.random.default_rng(99)
  base = np.arange(n, dtype=np.int32)
  routes_np = np.stack([rng.permutation(base) for _ in range(pop_size)]).astype(np.int32)
  routes_gpu = cp.asarray(routes_np)

  ctx = ProblemContext(problem, xp=cp, seed=99)
  strategy = CrossoverGPUPartiallyMapped()
  offspring = strategy.crossover_batch(routes_gpu, context=ctx)

  assert isinstance(offspring, cp.ndarray)
  assert offspring.shape == (pop_size, n)
  assert strategy.kernel_launches == 1
  assert strategy.d2h_bytes == 0
  assert strategy.h2d_bytes == 2 * pop_size * 4  # cut1 + cut2 int32 arrays

  strategy.reset_metrics()
  assert (strategy.h2d_bytes, strategy.d2h_bytes, strategy.kernel_launches) == (0, 0, 0)

  host_offspring = strategy.crossover_batch(routes_np.copy(), context=ctx)
  assert isinstance(host_offspring, np.ndarray)
  assert strategy.kernel_launches == 1
  assert strategy.h2d_bytes == routes_np.nbytes + 2 * pop_size * 4
  assert strategy.d2h_bytes == routes_np.nbytes


@pytest.mark.gpu
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("pop_size", [1, 3, 5])
def test_gpu_pmx_crossover_small_n_open_routes(n: int, pop_size: int) -> None:
  """WHAT: Crossover on boundary dimensions `n in {2, 3, 4}` with odd and single populations.

  EXPECTED: Bit-identical offspring to CPU reference.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  rng = np.random.default_rng(500 + n + pop_size)
  base_tour = np.arange(n, dtype=np.int32)
  if pop_size == 1:
    population = rng.permutation(base_tour).astype(np.int32)
  else:
    population = np.stack([rng.permutation(base_tour) for _ in range(pop_size)]).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=123)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=123)

  cpu_off = CrossoverPartiallyMapped().crossover_batch(population.copy(), context=cpu_ctx)
  gpu_off = CrossoverGPUPartiallyMapped().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_off), np.asarray(cpu_off))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("pop_size", [2, 3])
def test_gpu_pmx_crossover_small_n_closed_routes(n: int, pop_size: int) -> None:
  """WHAT: Crossover on small closed tours `[0, ..., 0]` including degenerate `n=3`.

  EXPECTED: Bit-identical offspring to CPU reference with depot ends preserved.
  """
  cp = _cupy()
  problem = _make_dummy_problem(n)
  rng = np.random.default_rng(600 + n)
  inner_cities = np.arange(1, n - 1, dtype=np.int32)
  routes = [
    np.concatenate([[0], rng.permutation(inner_cities).astype(np.int32), [0]]).astype(np.int32) for _ in range(pop_size)
  ]
  population = np.stack(routes).astype(np.int32)

  cpu_ctx = ProblemContext(problem, xp=np, seed=124)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=124)

  cpu_off = CrossoverPartiallyMapped().crossover_batch(population.copy(), context=cpu_ctx)
  gpu_off = CrossoverGPUPartiallyMapped().crossover_batch(population.copy(), context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(gpu_off), np.asarray(cpu_off))


@pytest.mark.gpu
def test_gpu_pmx_crossover_recovers_from_warmup_and_zero_filled_batch() -> None:
  """WHAT: AOT `warmup(context)` and zero-filled batch followed by open permutations (`BUG-24`).

  EXPECTED: `warmup(context)` compiles & launches safely without poisoning subsequent
  open-tour runs, and a zero-filled batch does not corrupt later open-tour parity.
  """
  cp = _cupy()
  n = 52
  pop_size = 32
  problem = _make_dummy_problem(n)

  gpu_strategy = CrossoverGPUPartiallyMapped(seed=5)
  poison_ctx = ProblemContext(problem, xp=cp, seed=1)
  gpu_strategy.warmup(poison_ctx)
  assert gpu_strategy.kernel_launches == 1

  gpu_strategy.crossover_batch(cp.zeros((pop_size, n), dtype=cp.int32), context=poison_ctx)

  base = np.arange(n, dtype=np.int32)
  population = np.stack([np.random.default_rng(700 + row).permutation(base) for row in range(pop_size)]).astype(
    np.int32
  )
  cpu_offspring = CrossoverPartiallyMapped(seed=5).crossover_batch(
    population.copy(), context=ProblemContext(problem, xp=np, seed=77)
  )
  gpu_offspring = gpu_strategy.crossover_batch(population.copy(), context=ProblemContext(problem, xp=cp, seed=77))

  np.testing.assert_array_equal(np.asarray(gpu_offspring), np.asarray(cpu_offspring))


@pytest.mark.gpu
def test_gpu_pmx_crossover_1d_pair() -> None:
  """WHAT: Crossover between two 1D parent vectors.

  EXPECTED: Bit-identical 1D child to CPU reference.
  """
  cp = _cupy()
  n = 20
  problem = _make_dummy_problem(n)
  p1 = np.arange(n, dtype=np.int32)
  p2 = np.random.default_rng(1).permutation(p1)

  cpu_ctx = ProblemContext(problem, xp=np, seed=42)
  gpu_ctx = ProblemContext(problem, xp=cp, seed=42)

  c_cpu = CrossoverPartiallyMapped().crossover(p1, p2, context=cpu_ctx)
  c_gpu = CrossoverGPUPartiallyMapped().crossover(p1, p2, context=gpu_ctx)

  np.testing.assert_array_equal(np.asarray(c_gpu), np.asarray(c_cpu))
  assert c_gpu.ndim == 1
