"""Order Crossover (OX) GPU vs CPU Scaling Sweep Harness (`DP11-11`, `BUG-22`, `OX-A`).

Evaluates runtime scaling of CrossoverGPUOrder vs CrossoverOrder for N=256 across
n in {52, 100, 200, 500, 1000, 2000}, fitting power-law exponent beta.
"""

from __future__ import annotations

import time

import numpy as np

from algorithms.strategies.crossover.order_crossover import CrossoverGPUOrder, CrossoverOrder
from data_models.problem import Problem
from protocols.backend import require_gpu
from protocols.problem_context import ProblemContext


def run_crossover_scaling_sweep(
  pop_size: int = 256,
  sizes: list[int] | None = None,
  repetitions: int = 15,
) -> tuple[float, list[dict[str, float]]]:
  """Run scaling sweep for CrossoverGPUOrder and CrossoverOrder."""
  if sizes is None:
    sizes = [52, 100, 200, 500, 1000, 2000]

  cp = require_gpu()
  cpu_strategy = CrossoverOrder(seed=42)
  gpu_strategy = CrossoverGPUOrder(seed=42)

  # Warm-up run on GPU
  warm_n = 52
  warm_prob = Problem(
    name="warmup",
    dimension=warm_n,
    problem_type="TSP",
    edge_type="EUC_2D",
    coordinates=np.zeros((warm_n, 2)),
    distances=None,
    capacity=None,
    demands=None,
  )
  warm_ctx = ProblemContext(warm_prob, xp=cp, seed=1)
  # Valid permutation rows: a zero-filled batch has `0 == 0` endpoints and used to
  # poison the route-topology cache, silently marking later open tours as closed (`BUG-24`).
  warm_pop = cp.tile(cp.arange(warm_n, dtype=cp.int32), (pop_size, 1))
  gpu_strategy.crossover_batch(warm_pop, context=warm_ctx)
  cp.cuda.Device().synchronize()

  records: list[dict[str, float]] = []

  for n in sizes:
    rng = np.random.default_rng(100 + n)
    base = np.arange(n, dtype=np.int32)
    pop_cpu = np.stack([rng.permutation(base) for _ in range(pop_size)]).astype(np.int32)
    pop_gpu = cp.asarray(pop_cpu)

    problem = Problem(
      name=f"scale_{n}",
      dimension=n,
      problem_type="TSP",
      edge_type="EUC_2D",
      coordinates=np.zeros((n, 2)),
      distances=None,
      capacity=None,
      demands=None,
    )
    cpu_ctx = ProblemContext(problem, xp=np, seed=200 + n)
    gpu_ctx = ProblemContext(problem, xp=cp, seed=200 + n)

    # Un-timed warm-up for this size to warm memory pools, JIT launch state and caches
    cpu_strategy.crossover_batch(pop_cpu, context=cpu_ctx)
    gpu_strategy.crossover_batch(pop_gpu, context=gpu_ctx)
    cp.cuda.Device().synchronize()

    # CPU Timing
    cpu_times = []
    for _ in range(max(3, repetitions // 3)):
      t0 = time.perf_counter()
      cpu_strategy.crossover_batch(pop_cpu, context=cpu_ctx)
      t1 = time.perf_counter()
      cpu_times.append((t1 - t0) * 1000.0)
    cpu_med = float(np.median(cpu_times))

    # GPU Timing (Device-to-device kernel execution)
    gpu_times = []
    for _ in range(repetitions):
      cp.cuda.Device().synchronize()
      t0 = time.perf_counter()
      gpu_strategy.crossover_batch(pop_gpu, context=gpu_ctx)
      cp.cuda.Device().synchronize()
      t1 = time.perf_counter()
      gpu_times.append((t1 - t0) * 1000.0)
    gpu_med = float(np.median(gpu_times))

    speedup = cpu_med / gpu_med if gpu_med > 0 else 0.0
    records.append(
      {
        "n": float(n),
        "cpu_ms": cpu_med,
        "gpu_ms": gpu_med,
        "speedup": speedup,
      }
    )

  # Fit power-law: log(T) = beta * log(n) + alpha
  log_n = np.log([r["n"] for r in records])
  log_t_gpu = np.log([r["gpu_ms"] for r in records])
  beta_gpu, _ = np.polyfit(log_n, log_t_gpu, 1)

  return float(beta_gpu), records


#: Baseline legacy GPU timings (2026-11-06 before Task 11.10: O(n^2) linear scan, beta=1.88)
LEGACY_GPU_BASELINE_MS: dict[int, float] = {
  52: 0.78,
  100: 2.21,
  200: 8.54,
  500: 51.30,
  1000: 198.40,
  2000: 742.05,
}


def main() -> None:
  """Execute and print scaling benchmark results with before/after comparison."""
  beta_gpu, records = run_crossover_scaling_sweep()
  print("=" * 82)
  print(f"Order Crossover Scaling Sweep (N=256) — Fitted GPU Exponent: beta = {beta_gpu:.4f}")
  print("=" * 82)
  print(f"{'n':>6} | {'CPU (ms)':>10} | {'Legacy GPU (ms)':>15} | {'Optimized GPU (ms)':>18} | {'Speedup vs CPU':>15}")
  print("-" * 82)
  for r in records:
    n_val = int(r["n"])
    legacy_ms = LEGACY_GPU_BASELINE_MS.get(n_val, float("nan"))
    legacy_str = f"{legacy_ms:15.3f}" if not np.isnan(legacy_ms) else f"{'N/A':>15}"
    print(f"{n_val:6d} | {r['cpu_ms']:10.3f} | {legacy_str} | {r['gpu_ms']:18.3f} | {r['speedup']:14.2f}x")
  print("-" * 82)
  print(f"DoD Condition beta <= 1.2: {'SATISFIED' if beta_gpu <= 1.2 else 'FAILED'}")
  print(f"Legacy GPU Exponent: beta_legacy ≈ 1.88 (quadratic) -> Optimized: beta_new = {beta_gpu:.4f}")


if __name__ == "__main__":
  main()
