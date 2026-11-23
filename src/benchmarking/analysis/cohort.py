"""Typed, immutable benchmark cohort view for statistical analysis and reporting.

Parses and validates nested benchmark result dictionaries (``problem -> algorithm -> metrics``)
once at the module boundary, exposing flat, single-source query methods for complete-case
matrices, finite metric extraction, paired Fleming & Wallace geometric speedups, quality
metric fallback (``mean_gap`` vs. ``mean_cost``), and per-algorithm cohort summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from benchmarking.analysis.stats_utils import (
  finite_metric,
  geometric_mean_ratios,
  resolve_baseline_algorithm,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from numpy.typing import NDArray

  from benchmarking.runner.config_loader import ProblemConfig


def _positive_int(value: Any) -> int | None:
  """Return positive integer value when ``value`` is a finite non-bool numeric > 0."""
  num = finite_metric(value)
  if num is None or num <= 0.0:
    return None
  return int(num)


def _positive_float(value: Any) -> float | None:
  """Return positive float value when ``value`` is a finite non-bool numeric > 0."""
  num = finite_metric(value)
  if num is None or num <= 0.0:
    return None
  return num


def resolve_algorithm_names(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  algorithm_order: Sequence[str] | None = None,
) -> list[str]:
  """Collect unique algorithm names in requested display order followed by discovery order."""
  ordered: dict[str, None] = {}
  if algorithm_order:
    for algo in algorithm_order:
      if any(algo in prob_res for prob_res in all_results.values()):
        ordered.setdefault(algo, None)
  for prob_res in all_results.values():
    for algo in prob_res:
      ordered.setdefault(algo, None)
  return list(ordered.keys())


def _extract_entry_size(entry: Mapping[str, Any]) -> int | None:
  """Extract problem dimension from a checkpoint/result entry when present."""
  size = _positive_int(entry.get("problem_size"))
  if size is not None:
    return size
  meta = entry.get("problem_metadata")
  if isinstance(meta, Mapping):
    return _positive_int(meta.get("dimension") or meta.get("size"))
  return None


def derive_problem_configs(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  explicit_configs: Sequence[ProblemConfig] | None = None,
) -> list[ProblemConfig]:
  """Build ``ProblemConfig`` entries from explicit metadata plus checkpoint fallback fields.

  Explicit entries take precedence; any missing problem derives its ``size`` and optional
  ``optimal`` cost from the first valid checkpoint payload for that instance. Returned list
  is deterministically ordered by ``(size, name)``.
  """
  from benchmarking.runner.config_loader import ProblemConfig

  config_map: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (explicit_configs or ())}
  resolved: list[ProblemConfig] = []

  for problem_name in sorted(all_results.keys()):
    if problem_name in config_map:
      resolved.append(config_map[problem_name])
      continue

    size: int = 0
    optimal: float | None = None
    for entry in all_results[problem_name].values():
      if size == 0:
        found_size = _extract_entry_size(entry)
        if found_size is not None:
          size = found_size
      if optimal is None:
        optimal = _positive_float(entry.get("optimal_cost"))
      if size > 0 and optimal is not None:
        break

    resolved.append(ProblemConfig(name=problem_name, optimal=optimal, size=size))

  resolved.sort(key=lambda cfg: (cfg.size, cfg.name))
  return resolved


@dataclass(frozen=True)
class SpeedupSummary:
  """Paired geometric mean speedup summary (Fleming & Wallace, 1986)."""

  geomean: float | None
  sample_size: int


@dataclass(frozen=True)
class AlgorithmAggregate:
  """Aggregated cross-problem descriptive statistics for one algorithm."""

  algorithm: str
  n_problems: int
  mean_gap: float | None
  std_gap: float | None
  mean_cost: float | None
  std_cost: float | None
  mean_time: float | None
  std_time: float | None


def _mean_and_sample_std(values: Sequence[float]) -> tuple[float | None, float | None]:
  """Compute ``(mean, sample_std)`` with Bessel's correction (``ddof=1``; ``0.0`` for ``N=1``)."""
  if not values:
    return None, None
  arr = np.asarray(values, dtype=float)
  mean_val = float(np.mean(arr))
  std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
  return mean_val, std_val


@dataclass(frozen=True)
class BenchmarkCohort:
  """Immutable analytical view over a multi-problem, multi-algorithm benchmark cohort."""

  results: Mapping[str, Mapping[str, Mapping[str, Any]]]
  problems: tuple[str, ...]
  algorithms: tuple[str, ...]
  problem_configs: tuple[ProblemConfig, ...]
  _config_by_name: Mapping[str, ProblemConfig]

  @classmethod
  def from_results(
    cls,
    all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    problem_configs: Sequence[ProblemConfig] | None = None,
    *,
    algorithm_order: Sequence[str] | None = None,
    sort_problems: bool = True,
  ) -> BenchmarkCohort:
    """Construct a validated ``BenchmarkCohort`` from nested result dictionaries."""
    problems = tuple(sorted(all_results.keys()) if sort_problems else list(all_results.keys()))
    algorithms = tuple(resolve_algorithm_names(all_results, algorithm_order))
    resolved_configs = tuple(derive_problem_configs(all_results, problem_configs))
    config_by_name = {cfg.name: cfg for cfg in resolved_configs}
    return cls(
      results=all_results,
      problems=problems,
      algorithms=algorithms,
      problem_configs=resolved_configs,
      _config_by_name=config_by_name,
    )

  def baseline(self, requested: str | None = None, algorithms: Sequence[str] | None = None) -> str:
    """Resolve the baseline algorithm name preserving canonical display/insertion order."""
    target_algs = self.algorithms if algorithms is None else algorithms
    return resolve_baseline_algorithm(target_algs, requested)

  def problem_size(self, problem: str) -> int | None:
    """Return problem city count (> 0) if known from config or checkpoint metadata."""
    cfg = self._config_by_name.get(problem)
    if cfg is not None and cfg.size > 0:
      return int(cfg.size)
    for entry in self.results.get(problem, {}).values():
      found = _extract_entry_size(entry)
      if found is not None:
        return found
    return None

  def problem_optimal(self, problem: str) -> float | None:
    """Return reference optimal tour cost when known (> 0), else ``None``."""
    cfg = self._config_by_name.get(problem)
    if cfg is not None and cfg.optimal is not None and cfg.optimal > 0:
      return float(cfg.optimal)
    for entry in self.results.get(problem, {}).values():
      found = _positive_float(entry.get("optimal_cost"))
      if found is not None:
        return found
    return None

  def entry(self, problem: str, algorithm: str) -> Mapping[str, Any] | None:
    """Return the raw result mapping for ``(problem, algorithm)`` if present."""
    prob_res = self.results.get(problem)
    if prob_res is None:
      return None
    return prob_res.get(algorithm)

  def metric(self, problem: str, algorithm: str, metric_key: str) -> float | None:
    """Extract one finite numeric metric from ``(problem, algorithm)``, or ``None``."""
    rec = self.entry(problem, algorithm)
    if rec is None:
      return None
    return finite_metric(rec.get(metric_key))

  def algorithms_for_problem(
    self,
    problem: str,
    algorithms: Sequence[str] | None = None,
  ) -> list[str]:
    """Return ordered algorithms present for ``problem``."""
    prob_res = self.results.get(problem, {})
    target_algs = self.algorithms if algorithms is None else algorithms
    return [alg for alg in target_algs if alg in prob_res]

  def algorithm_metric_values(
    self,
    algorithm: str,
    metric_key: str,
    problems: Sequence[str] | None = None,
  ) -> list[float]:
    """Collect finite metric values for ``algorithm`` across ``problems``."""
    target_probs = self.problems if problems is None else problems
    return [val for p in target_probs if (val := self.metric(p, algorithm, metric_key)) is not None]

  def complete_problems(
    self,
    algorithms: Sequence[str] | None = None,
    problems: Sequence[str] | None = None,
    metric_key: str | None = None,
  ) -> list[str]:
    """Return problems having entries (and optional finite ``metric_key``) for every target algorithm."""
    target_algs = self.algorithms if algorithms is None else algorithms
    target_probs = self.problems if problems is None else problems
    if not target_algs:
      return []

    if metric_key is None:
      return [p for p in target_probs if p in self.results and all(a in self.results[p] for a in target_algs)]
    return [
      p
      for p in target_probs
      if p in self.results and all(self.metric(p, a, metric_key) is not None for a in target_algs)
    ]

  def metric_matrix(
    self,
    metric_key: str,
    algorithms: Sequence[str] | None = None,
    problems: Sequence[str] | None = None,
  ) -> tuple[list[str], NDArray[np.float64]]:
    """Build complete-case ``(valid_problems, matrix[N, k])`` for ``metric_key``."""
    target_algs = list(self.algorithms if algorithms is None else algorithms)
    valid_probs = self.complete_problems(algorithms=target_algs, problems=problems, metric_key=metric_key)
    if not valid_probs or not target_algs:
      return valid_probs, np.empty((0, len(target_algs)), dtype=np.float64)

    rows = [[float(self.results[p][a][metric_key]) for a in target_algs] for p in valid_probs]
    return valid_probs, np.asarray(rows, dtype=np.float64)

  def quality_metric_key(
    self,
    algorithms: Sequence[str] | None = None,
    problems: Sequence[str] | None = None,
    *,
    require_complete: bool = True,
  ) -> Literal["mean_gap", "mean_cost"]:
    """Choose ``"mean_gap"`` when finite gaps cover the cohort, else fall back to ``"mean_cost"``."""
    target_algs = list(self.algorithms if algorithms is None else algorithms)
    target_probs = list(self.problems if problems is None else problems)
    if not target_algs or not target_probs:
      return "mean_cost"

    if require_complete:
      complete = self.complete_problems(algorithms=target_algs, problems=target_probs)
      has_all_gaps = bool(
        complete and all(self.metric(p, a, "mean_gap") is not None for p in complete for a in target_algs)
      )
      return "mean_gap" if has_all_gaps else "mean_cost"

    present_cells = [(p, a) for p in target_probs if p in self.results for a in target_algs if a in self.results[p]]
    has_all_gaps = bool(present_cells and all(self.metric(p, a, "mean_gap") is not None for p, a in present_cells))
    return "mean_gap" if has_all_gaps else "mean_cost"

  def paired_metric_vectors(
    self,
    alg_a: str,
    alg_b: str,
    metric_key: str,
    problems: Sequence[str] | None = None,
  ) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64]]:
    """Extract aligned finite metric vectors ``(paired_problems, vec_a, vec_b)``."""
    target_probs = self.problems if problems is None else problems
    paired_probs: list[str] = []
    vals_a: list[float] = []
    vals_b: list[float] = []
    for p in target_probs:
      va = self.metric(p, alg_a, metric_key)
      vb = self.metric(p, alg_b, metric_key)
      if va is not None and vb is not None:
        paired_probs.append(p)
        vals_a.append(va)
        vals_b.append(vb)
    return paired_probs, np.asarray(vals_a, dtype=np.float64), np.asarray(vals_b, dtype=np.float64)

  def paired_geometric_speedup(
    self,
    candidate: str,
    baseline: str,
    problems: Sequence[str] | None = None,
  ) -> SpeedupSummary:
    """Compute paired geometric mean speedup ``baseline_time / candidate_time`` across shared instances."""
    _, base_times, cand_times = self.paired_metric_vectors(baseline, candidate, "mean_time", problems=problems)
    positive_mask = (base_times > 0.0) & (cand_times > 0.0)
    if not bool(np.any(positive_mask)):
      return SpeedupSummary(geomean=None, sample_size=0)
    ratios = base_times[positive_mask] / cand_times[positive_mask]
    return SpeedupSummary(
      geomean=geometric_mean_ratios(ratios.tolist()),
      sample_size=int(np.sum(positive_mask)),
    )

  def summarize_algorithm(
    self,
    algorithm: str,
    problems: Sequence[str] | None = None,
  ) -> AlgorithmAggregate | None:
    """Compute cross-problem mean and sample SD (``ddof=1``) for gap, cost, and time."""
    gaps = self.algorithm_metric_values(algorithm, "mean_gap", problems=problems)
    costs = self.algorithm_metric_values(algorithm, "mean_cost", problems=problems)
    times = self.algorithm_metric_values(algorithm, "mean_time", problems=problems)
    n_problems = max(len(gaps), len(costs), len(times))
    if n_problems == 0:
      return None

    mean_gap, std_gap = _mean_and_sample_std(gaps)
    mean_cost, std_cost = _mean_and_sample_std(costs)
    mean_time, std_time = _mean_and_sample_std(times)
    return AlgorithmAggregate(
      algorithm=algorithm,
      n_problems=n_problems,
      mean_gap=mean_gap,
      std_gap=std_gap,
      mean_cost=mean_cost,
      std_cost=std_cost,
      mean_time=mean_time,
      std_time=std_time,
    )

  def repetition_counts(self) -> list[int]:
    """Extract positive repetition counts across every ``(problem, algorithm)`` cell."""
    counts: list[int] = []
    for prob_res in self.results.values():
      for rec in prob_res.values():
        reps = _positive_int(rec.get("repetitions")) or _positive_int(rec.get("successful_runs"))
        if reps is None:
          raw_times = rec.get("raw_times")
          if isinstance(raw_times, (list, tuple)) and len(raw_times) > 0:
            reps = len(raw_times)
        if reps is not None:
          counts.append(reps)
    return counts


__all__ = [
  "AlgorithmAggregate",
  "BenchmarkCohort",
  "SpeedupSummary",
  "derive_problem_configs",
  "resolve_algorithm_names",
]
