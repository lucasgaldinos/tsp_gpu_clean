"""Smoke test script for the archived monolithic Fujimoto GA baseline.

Pure-Python script with no command-line arguments to verify that the
archived monolithic Full-GPU kernel instantiates, executes, and honours
the standardized convergence statistics contract when CUDA is available.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

from algorithms.archive import GeneticAlgorithmFullGPUMonolithic
from loaders.tsplib_loader import TsplibLoader
from protocols.backend import cupy_backend, gpu_available
from protocols.problem_context import ProblemContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEM_PATH = PROJECT_ROOT / "tests/fixtures/tsp/burma14.tsp"
MAX_GENERATIONS = 10
PATIENCE = 5
POPULATION_SIZE = 64
SEED = 42


def main() -> int:
  """Run a short smoke test of the archived baseline model."""
  if not gpu_available():
    print("[SKIP] CUDA GPU or CuPy is unavailable; skipping archive baseline smoke check.")
    return 2

  cp = cupy_backend()
  if cp is None:
    print("[SKIP] CuPy backend returned None; skipping archive baseline smoke check.")
    return 2

  if not PROBLEM_PATH.exists():
    print(f"[ERROR] Problem file not found at {PROBLEM_PATH}", file=sys.stderr)
    return 1

  print(f"[INFO] Loading problem from {PROBLEM_PATH}...")
  with TsplibLoader(PROBLEM_PATH.parent) as loader:
    problem = loader.load(PROBLEM_PATH.stem)

  print(f"[INFO] Initializing ProblemContext for {problem.name} on GPU backend...")
  context = ProblemContext(problem, xp=cp, seed=SEED)
  customers = list(range(problem.dimension))

  print(f"[INFO] Instantiating GeneticAlgorithmFullGPUMonolithic (pop_size={POPULATION_SIZE})...")
  ga = GeneticAlgorithmFullGPUMonolithic(
    population_size=POPULATION_SIZE,
    seed=SEED,
  )

  print(f"[INFO] Running evolve for up to {MAX_GENERATIONS} generations (patience={PATIENCE})...")
  best_tour, stats = ga.evolve(
    context,
    customers,
    max_generations=MAX_GENERATIONS,
    patience=PATIENCE,
  )

  # Contract validation
  required_keys = (
    "best_fitness",
    "initial_fitness",
    "improvement_pct",
    "convergence_history",
    "h2d_bytes",
    "d2h_bytes",
    "kernel_launches",
    "generations_completed",
    "stop_reason",
  )
  for key in required_keys:
    if key not in stats:
      print(f"[ERROR] Required stats key '{key}' missing from return dict!", file=sys.stderr)
      return 1

  if "best_cost_history" in stats:
    print("[ERROR] Deprecated key 'best_cost_history' still present in stats dict!", file=sys.stderr)
    return 1

  conv_history = stats["convergence_history"]
  generations_completed = stats["generations_completed"]
  expected_points = generations_completed + 1
  if len(conv_history) != expected_points:
    print(
      f"[ERROR] convergence_history length mismatch: expected {expected_points}, got {len(conv_history)}",
      file=sys.stderr,
    )
    return 1

  if len(best_tour) != problem.dimension:
    print(
      f"[ERROR] best_tour dimension mismatch: expected {problem.dimension}, got {len(best_tour)}",
      file=sys.stderr,
    )
    return 1

  print("[SUCCESS] Archive baseline smoke test passed!")
  print(f"          Best fitness: {stats['best_fitness']}")
  print(f"          Generations: {generations_completed}")
  print(f"          Stop reason: {stats['stop_reason']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
