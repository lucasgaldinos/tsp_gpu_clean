"""Problem catalogue builders from `.tsp`/`.vrp` files and DuckDB databases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

from data_models.exceptions import AmbiguousProblemError
from loaders.tsplib_loader import TsplibIngestionBundle, parse_tsplib_file, resolve_opt_tour_cost

from .config_models import ProblemConfig


def _resolve_config_problem_type(
  parsed_type: str,
  *,
  disambiguate_twins: bool,
  matched_existing: ProblemConfig | None,
) -> str | None:
  """Return explicit `problem_type` for VRP/ATSP/twin configs, keeping single TSP as `None`."""
  has_explicit_existing = matched_existing is not None and matched_existing.problem_type is not None
  if parsed_type != "TSP" or disambiguate_twins or has_explicit_existing:
    return parsed_type
  return None


def _finalize_problem_config(
  *,
  name: str,
  parsed_type: str,
  dimension: int,
  resolved_optimal: float | None,
  disambiguate_twins: bool,
  existing_by_type: Mapping[tuple[str, str], ProblemConfig],
  existing_by_name: Mapping[str, ProblemConfig],
) -> ProblemConfig:
  """Build one normalized `ProblemConfig` preserving catalogue overrides and optimal fallback."""
  matched_existing = existing_by_type.get((name, parsed_type)) or existing_by_name.get(name)
  optimal: float | None
  if resolved_optimal is not None:
    optimal = float(resolved_optimal)
  elif matched_existing is not None:
    optimal = matched_existing.optimal
  else:
    optimal = None

  cfg_ptype = _resolve_config_problem_type(
    parsed_type,
    disambiguate_twins=disambiguate_twins,
    matched_existing=matched_existing,
  )
  return ProblemConfig(name=name, optimal=optimal, size=int(dimension), problem_type=cfg_ptype)


def build_problem_configs_from_tsp_files(
  tsp_files: Sequence[str | Path],
  existing_configs: Sequence[ProblemConfig] | None = None,
  tsp_dir: str | Path | None = None,
  known_optimals: Mapping[str, float] | None = None,
) -> tuple[list[ProblemConfig], dict[str, Path]]:
  """Build `ProblemConfig` instances directly from `.tsp` and `.vrp` paths using `routing_data`."""
  existing_by_type: dict[tuple[str, str], ProblemConfig] = {
    (cfg.name, cfg.problem_type): cfg for cfg in (existing_configs or []) if cfg.problem_type is not None
  }
  existing_by_name: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}
  optimal_map: Mapping[str, float] = known_optimals or {}
  base_dir: Path | None = Path(tsp_dir) if tsp_dir is not None else None

  resolved_configs: list[ProblemConfig] = []
  tsp_path_map: dict[str, Path] = {}

  for item in tsp_files:
    candidate = Path(item)
    if not candidate.exists():
      if base_dir is not None:
        if candidate.suffix.lower() in (".tsp", ".vrp") and (base_dir / candidate.name).is_file():
          candidate = base_dir / candidate.name
        else:
          stem: str = candidate.stem if candidate.suffix.lower() in (".tsp", ".vrp") else str(item)
          matches: list[Path] = [
            cand for cand in (base_dir / f"{stem}.tsp", base_dir / f"{stem}.vrp") if cand.is_file()
          ]
          if not matches:
            raise FileNotFoundError(f"TSPLIB file not found: {item}")
          existing_cfg = existing_by_name.get(stem)
          if len(matches) > 1 and existing_cfg is not None and existing_cfg.problem_type is not None:
            wanted_type = existing_cfg.problem_type.upper().strip()
            matched = [p for p in matches if parse_tsplib_file(p).problem.problem_type == wanted_type]
            if not matched:
              raise FileNotFoundError(f"TSPLIB file not found for {item} ({wanted_type})")
            candidate = matched[0]
          else:
            candidate = matches[0]
      else:
        raise FileNotFoundError(f"TSPLIB file not found: {item}")

    bundle: TsplibIngestionBundle = parse_tsplib_file(candidate)
    name = str(bundle.problem.name)
    dimension = int(bundle.problem.dimension)
    parsed_type = str(bundle.problem.problem_type)
    tsp_path_map[name] = candidate.resolve()

    matched_existing = existing_by_type.get((name, parsed_type)) or existing_by_name.get(name)
    optimal_override: float | None = None
    if matched_existing is not None:
      optimal_override = matched_existing.optimal
    elif name in optimal_map:
      optimal_override = float(optimal_map[name])
    elif candidate.stem in optimal_map:
      optimal_override = float(optimal_map[candidate.stem])
    else:
      optimal_override = resolve_opt_tour_cost(candidate, bundle=bundle)

    twin_exists = (candidate.parent / f"{candidate.stem}.tsp").is_file() and (
      candidate.parent / f"{candidate.stem}.vrp"
    ).is_file()
    resolved_configs.append(
      _finalize_problem_config(
        name=name,
        parsed_type=parsed_type,
        dimension=dimension,
        resolved_optimal=optimal_override,
        disambiguate_twins=twin_exists,
        existing_by_type=existing_by_type,
        existing_by_name=existing_by_name,
      )
    )

  return resolved_configs, tsp_path_map


def build_problem_configs_from_duckdb(
  db_path: str | Path,
  problem_names: Sequence[str] | None = None,
  existing_configs: Sequence[ProblemConfig] | None = None,
) -> list[ProblemConfig]:
  """Build `ProblemConfig` instances by querying DuckDB for dimension and canonical solution cost."""
  import duckdb

  db_file = Path(db_path)
  if not db_file.exists():
    raise FileNotFoundError(f"Database file not found: {db_file}")

  existing_by_type: dict[tuple[str, str], ProblemConfig] = {
    (cfg.name, cfg.problem_type): cfg for cfg in (existing_configs or []) if cfg.problem_type is not None
  }
  existing_by_name: dict[str, ProblemConfig] = {cfg.name: cfg for cfg in (existing_configs or [])}

  conn = duckdb.connect(str(db_file), read_only=True)
  try:
    tables = {str(r[0]) for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    dim_parts: list[str] = ["t.dimension"]
    joins: list[str] = ["LEFT JOIN tsp_problems t ON p.id = t.problem_id"]
    if "atsp_problems" in tables:
      dim_parts.append("a.dimension")
      joins.append("LEFT JOIN atsp_problems a ON p.id = a.problem_id")
    if "cvrp_problems" in tables:
      dim_parts.append("c.dimension")
      joins.append("LEFT JOIN cvrp_problems c ON p.id = c.problem_id")

    dim_expr = dim_parts[0] if len(dim_parts) == 1 else f"COALESCE({', '.join(dim_parts)})"
    join_sql = "\n        ".join(joins)
    has_solutions = "solutions" in tables
    solution_subquery = (
      """(SELECT s.cost FROM solutions s
                WHERE s.problem_id = p.id AND s.cost IS NOT NULL
                ORDER BY CASE s.quality
                  WHEN 'proven_optimal' THEN 0
                  WHEN 'best_known' THEN 1
                  ELSE 2
                END ASC, s.id ASC
                LIMIT 1)"""
      if has_solutions
      else "NULL"
    )

    if problem_names is not None:
      names_list = list(problem_names)
      if not names_list:
        return []
      placeholders = ", ".join("?" for _ in names_list)
      query = f"""
        SELECT p.id, p.name, p.type, {dim_expr} AS dimension,
               {solution_subquery} AS solution_cost
        FROM problems p
        {join_sql}
        WHERE p.name IN ({placeholders}) AND {dim_expr} IS NOT NULL
        ORDER BY dimension ASC, p.name ASC, p.type ASC
      """  # noqa: S608
      rows = conn.execute(query, names_list).fetchall()
    else:
      query = f"""
        SELECT p.id, p.name, p.type, {dim_expr} AS dimension,
               {solution_subquery} AS solution_cost
        FROM problems p
        {join_sql}
        WHERE {dim_expr} IS NOT NULL
        ORDER BY dimension ASC, p.name ASC, p.type ASC
      """  # noqa: S608
      rows = conn.execute(query).fetchall()

    name_counts: dict[str, int] = {}
    for _prob_id, name, _ptype, _dim, _sol_cost in rows:
      name_counts[name] = name_counts.get(name, 0) + 1

    if problem_names is not None:
      rows_by_name: dict[str, list[tuple[Any, ...]]] = {}
      for r in rows:
        rows_by_name.setdefault(str(r[1]), []).append(r)

      missing: list[str] = [n for n in problem_names if n not in rows_by_name]
      if missing:
        raise KeyError(f"Problem(s) not found in DuckDB database: {missing}")

      ordered_configs: list[ProblemConfig] = []
      for n in problem_names:
        candidates = rows_by_name[n]
        existing_cfg = existing_by_name.get(n)
        if len(candidates) > 1:
          if existing_cfg is not None and existing_cfg.problem_type is not None:
            wanted = existing_cfg.problem_type.upper().strip()
            candidates = [c for c in candidates if str(c[2]) == wanted]
            if not candidates:
              raise KeyError(f"Problem '{n}' with problem_type='{wanted}' not found in DuckDB database")
          else:
            raise AmbiguousProblemError(n, [str(c[2]) for c in candidates])

        _prob_id, name, ptype, dimension, solution_cost = candidates[0]
        ordered_configs.append(
          _finalize_problem_config(
            name=name,
            parsed_type=str(ptype),
            dimension=int(dimension),
            resolved_optimal=float(solution_cost) if solution_cost is not None else None,
            disambiguate_twins=name_counts.get(name, 0) > 1,
            existing_by_type=existing_by_type,
            existing_by_name=existing_by_name,
          )
        )
      return ordered_configs

    return [
      _finalize_problem_config(
        name=name,
        parsed_type=str(ptype),
        dimension=int(dimension),
        resolved_optimal=float(solution_cost) if solution_cost is not None else None,
        disambiguate_twins=name_counts.get(name, 0) > 1,
        existing_by_type=existing_by_type,
        existing_by_name=existing_by_name,
      )
      for _prob_id, name, ptype, dimension, solution_cost in rows
    ]
  finally:
    conn.close()


__all__: list[str] = [
  "_finalize_problem_config",
  "_resolve_config_problem_type",
  "build_problem_configs_from_duckdb",
  "build_problem_configs_from_tsp_files",
]
