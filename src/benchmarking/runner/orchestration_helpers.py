"""Step-level problem resolution and checkpoint resume planning helpers (`BUG-116`)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmarking.runner.checkpoint_io import load_checkpoint, quarantine_checkpoint
from benchmarking.runner.config_loader import (
  AlgorithmConfig,
  ProblemConfig,
  build_problem_configs_from_duckdb,
  build_problem_configs_from_tsp_files,
)
from loaders.database_loader import DatabaseLoader
from loaders.tsplib_loader import TsplibLoader, load_tsplib_file, resolve_opt_tour_cost

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence
  from pathlib import Path

  from data_models.problem import Problem

_STRATEGY_PARAM_ATTRS: tuple[str, ...] = (
  "k",
  "max_iterations",
  "candidate_k",
  "return_gpu",
  "max_matrix_dimension",
  "max_shared_tour_cities",
  "threads_per_block",
  "mode",
)


@dataclass(frozen=True, slots=True)
class ResumePlan:
  """Decision object produced by :func:`plan_checkpoint_resume` for one `(problem, algorithm)` cell."""

  existing_checkpoint: dict[str, Any] | None
  actual_repetitions: int
  incremental_mode: bool
  skip: bool


def serialize_custom_pipeline(steps: Sequence[Any]) -> list[dict[str, Any]]:
  """Serialize user-declared pipeline steps with the parameters that define them (`BUG-30`)."""
  specs: list[dict[str, Any]] = []
  for step in steps:
    public_params = getattr(step, "public_params", None)
    params: dict[str, Any] = public_params() if callable(public_params) else {}
    spec: dict[str, Any] = {"class": type(step).__name__}
    if params:
      spec["params"] = params
    specs.append(spec)
  return specs


def resolve_problem_selection(
  *,
  problem_configs: list[ProblemConfig] | None,
  tsp_files: Sequence[str | Path] | None,
  problem_names: Sequence[str] | None,
  default_problems: list[ProblemConfig],
  resolved_tsp_dir: Path | None,
  resolved_db_path: Path | None,
  has_tsp_dir: bool,
  has_db: bool,
) -> tuple[list[ProblemConfig], dict[str, Path]]:
  """Resolve problem configs and optional TSPLIB path map across the 6 input modes."""
  tsp_path_map: dict[str, Path] = {}
  if problem_configs is not None:
    return list(problem_configs), tsp_path_map
  if tsp_files:
    return build_problem_configs_from_tsp_files(
      tsp_files,
      existing_configs=default_problems,
      tsp_dir=resolved_tsp_dir,
    )
  if problem_names:
    if has_tsp_dir and not has_db:
      return build_problem_configs_from_tsp_files(
        problem_names,
        existing_configs=default_problems,
        tsp_dir=resolved_tsp_dir,
      )
    if has_db and resolved_db_path is not None:
      return (
        build_problem_configs_from_duckdb(
          db_path=resolved_db_path,
          problem_names=problem_names,
          existing_configs=default_problems,
        ),
        tsp_path_map,
      )
    raise ValueError(
      "Either an existing --tsp-dir or an existing --db-path must be provided when --problems is specified"
    )
  if default_problems:
    return list(default_problems), tsp_path_map
  if has_tsp_dir and resolved_tsp_dir is not None:
    discovered_files = sorted([*resolved_tsp_dir.glob("*.tsp"), *resolved_tsp_dir.glob("*.vrp")])
    if not discovered_files:
      raise FileNotFoundError(f"No .tsp or .vrp files found in --tsp-dir: {resolved_tsp_dir}")
    resolved_problems, tsp_path_map = build_problem_configs_from_tsp_files(
      discovered_files,
      existing_configs=default_problems,
      tsp_dir=resolved_tsp_dir,
    )
    resolved_problems.sort(key=lambda cfg: (cfg.size, cfg.name))
    return resolved_problems, tsp_path_map
  if has_db and resolved_db_path is not None:
    return (
      build_problem_configs_from_duckdb(
        db_path=resolved_db_path,
        existing_configs=default_problems,
      ),
      tsp_path_map,
    )
  raise ValueError(
    "No problems specified: provide --problems-config / problems.json, --tsp-files, --tsp-dir, or --db-path."
  )


def load_problem_and_optimal(
  problem_config: ProblemConfig,
  *,
  tsp_path_map: Mapping[str, Path],
  has_db: bool,
  resolved_db_path: Path | None,
  has_tsp_dir: bool,
  resolved_tsp_dir: Path | None,
) -> tuple[Problem, float | None]:
  """Load one `Problem` and resolve its BKS respecting declarative overrides (`DP17-10`)."""
  problem_name: str = problem_config.name
  optimal_cost: float | None = problem_config.optimal
  has_overrides: bool = problem_config.has_constraint_overrides()
  override_kwargs: dict[str, Any] = {
    "problem_type": problem_config.problem_type,
    "capacity": problem_config.capacity,
    "max_vehicles": problem_config.max_vehicles,
    "max_route_distance": problem_config.max_route_distance,
    "return_to_depot": problem_config.return_to_depot,
  }

  if problem_name in tsp_path_map and problem_config.problem_type is None and not has_overrides:
    problem: Problem = load_tsplib_file(tsp_path_map[problem_name])
    if optimal_cost is None:
      optimal_cost = resolve_opt_tour_cost(tsp_path_map[problem_name])
    return problem, optimal_cost

  if has_db and resolved_db_path is not None:
    with DatabaseLoader(str(resolved_db_path), fallback_tsp_dir=resolved_tsp_dir) as db_loader:
      problem = db_loader.load(problem_name, **override_kwargs)
      if optimal_cost is None and not has_overrides:
        optimal_cost = db_loader.load_optimal_cost(problem_name, problem_type=problem_config.problem_type)
    return problem, optimal_cost

  if has_tsp_dir and resolved_tsp_dir is not None:
    with TsplibLoader(resolved_tsp_dir) as tsp_loader:
      problem = tsp_loader.load(problem_name, **override_kwargs)
      if optimal_cost is None and not has_overrides:
        optimal_cost = tsp_loader.resolve_optimal_cost(problem_name, problem_type=problem_config.problem_type)
    return problem, optimal_cost

  if problem_name in tsp_path_map:
    base_problem = load_tsplib_file(tsp_path_map[problem_name])
    problem = base_problem.with_constraints(**override_kwargs)
    if optimal_cost is None and not has_overrides and problem.problem_type == base_problem.problem_type:
      optimal_cost = resolve_opt_tour_cost(tsp_path_map[problem_name])
    return problem, optimal_cost

  raise FileNotFoundError(
    f"Cannot load problem '{problem_name}': neither --tsp-dir ({resolved_tsp_dir}) "
    f"nor --db-path ({resolved_db_path}) exists"
  )


def plan_checkpoint_resume(
  checkpoint_path: Path,
  *,
  effective_repetitions: int,
  incremental_runs: bool,
  alg_name: str,
  algo_idx: int,
  total_algos: int,
  backend_type: str,
) -> ResumePlan:
  """Inspect an existing checkpoint file and produce a :class:`ResumePlan`."""
  existing_checkpoint: dict[str, Any] | None = None
  incremental_mode = incremental_runs
  checkpoint_exists = checkpoint_path.exists()
  existing_runs = 0

  if checkpoint_exists:
    try:
      existing_checkpoint = load_checkpoint(checkpoint_path)
      existing_runs = existing_checkpoint.get("successful_runs", 0) if existing_checkpoint else 0
    except (json.JSONDecodeError, KeyError, OSError) as exc:
      quarantine_path = quarantine_checkpoint(checkpoint_path)
      logging.error(
        f"Corrupted checkpoint detected at {checkpoint_path}: {exc}. "
        f"Quarantined to {quarantine_path.name}; starting clean run."
      )
      existing_checkpoint = None
      existing_runs = 0
      checkpoint_exists = False

  if not checkpoint_exists:
    logging.info(
      f"  [{algo_idx}/{total_algos}] {alg_name} ({backend_type}): Running {effective_repetitions} repetitions..."
    )
    return ResumePlan(
      existing_checkpoint=None,
      actual_repetitions=effective_repetitions,
      incremental_mode=incremental_mode,
      skip=False,
    )

  if incremental_mode:
    new_reps_to_add = effective_repetitions
    total_target_reps = existing_runs + new_reps_to_add
    logging.info(
      f"  [{algo_idx}/{total_algos}] {alg_name} ({backend_type}): "
      f"✓ Resuming from checkpoint ({existing_runs} runs) - "
      f"adding {new_reps_to_add} more runs → {total_target_reps} total"
    )
    return ResumePlan(
      existing_checkpoint=existing_checkpoint,
      actual_repetitions=new_reps_to_add,
      incremental_mode=True,
      skip=False,
    )

  if existing_runs >= effective_repetitions:
    logging.info(f"  [{algo_idx}/{total_algos}] {alg_name} ({backend_type}): ✓ Loading from checkpoint (complete)")
    return ResumePlan(
      existing_checkpoint=existing_checkpoint,
      actual_repetitions=0,
      incremental_mode=False,
      skip=True,
    )

  actual_repetitions = effective_repetitions - existing_runs
  logging.info(
    f"  [{algo_idx}/{total_algos}] {alg_name} ({backend_type}): "
    f"Resuming: {existing_runs}/{effective_repetitions} runs exist, completing remaining {actual_repetitions}"
  )
  return ResumePlan(
    existing_checkpoint=existing_checkpoint,
    actual_repetitions=actual_repetitions,
    incremental_mode=True,
    skip=False,
  )


def get_strategy_name(strat: Any) -> str:
  """Return human-readable class name for a strategy slot instance."""
  if strat is None:
    return "None"
  return getattr(strat, "__name__", type(strat).__name__)


def get_strategy_params(
  slot: str,
  *,
  instance: Any,
  config: AlgorithmConfig,
  strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
  """Capture configured + live attributes of one strategy slot for checkpoint provenance (`BUG-18`)."""
  strat: Any = getattr(instance, slot, None)
  if strat is None:
    return {}

  spec: dict[str, Any] = config.strategies.get(slot) or {}
  configured: dict[str, Any] = dict(spec.get("params") or {})
  params: dict[str, Any] = {"class": get_strategy_name(strat)}
  if isinstance(spec.get("module"), str) and spec["module"].strip():
    params["module"] = spec["module"]
  params.update(configured)
  if strategy_overrides and slot in strategy_overrides:
    params.update(strategy_overrides[slot])
  for attr in _STRATEGY_PARAM_ATTRS:
    value: Any = getattr(strat, attr, None)
    if value is not None:
      params[attr] = value
  return params


__all__: list[str] = [
  "ResumePlan",
  "get_strategy_name",
  "get_strategy_params",
  "load_problem_and_optimal",
  "plan_checkpoint_resume",
  "resolve_problem_selection",
  "serialize_custom_pipeline",
]
