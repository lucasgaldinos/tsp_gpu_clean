"""Typed records exchanged between the benchmarking layers.

Every value here crosses a module boundary. Today the producers build plain
``dict`` objects and the consumers index them by string literal, so a renamed
or missing key fails at runtime. These TypedDicts name the payload once and let
the checker verify both sides.

Note the boundary caveat: a payload parsed from JSON is ``Any`` until it is
validated, so a function returning one of these types must either construct the
dict literally (checked) or validate and ``cast`` at the parsing site.
"""

from collections.abc import Mapping
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray


class NemenyiResult(TypedDict):
  """Payload of `StatisticalAnalyzer.nemenyi_posthoc`.

  Attributes:
      p_values_matrix: `(k, k)` matrix of pairwise p-values; symmetric.
      significant_pairs: `(i, j, p_value)` triples whose p-value is below alpha.
      algorithm_names: Names aligned with the matrix indices.
      critical_distance: Nemenyi critical distance at the configured alpha.
  """

  p_values_matrix: NDArray[np.float64]
  significant_pairs: list[tuple[int, int, float]]
  algorithm_names: list[str]
  critical_distance: float


#: A payload bound for a `JSON` column: either already serialized, or still a
#: mapping that the writer serializes with `json.dumps`. `Mapping` rather than
#: `dict` because `Mapping` is covariant in its value type, so `dict[str, int]`
#: is accepted where `dict[str, object]` would be rejected.
type JsonPayload = str | Mapping[str, object]


class BenchmarkRunInput(TypedDict):
  """One checkpoint payload, as accepted by `insert_benchmark_run`.

  Mirrors `src/benchmarking/database/schema.sql`, which stays the
  authority for names and nullability. Two field groups:

  * **required** — the 18 columns declared `NOT NULL` without a default. The
    runtime validator derives its required-field list from
    `BenchmarkRunInput.__required_keys__`, so this class and
    `validate_benchmark_data` cannot drift apart.
  * **`NotRequired`** — the 34 nullable or default-backed columns (including
    `problem_type`, nullable cost/gap fields for `feasible_runs == 0` or
    unknown-optimum runs, and VRP feasibility/fleet metrics). The writer
    reads these with `dict.get`, so a legacy TSP checkpoint without them
    imports cleanly with `problem_type = 'TSP'` and SQL `NULL` for VRP fields.

  `run_id` is absent on purpose: the database assigns it from
  `benchmark_runs_seq` and returns it from the insert.
  """

  # --- required: problem identification -----------------------------------
  problem_name: str
  problem_size: int

  # --- required: algorithm identification ---------------------------------
  algorithm: str
  backend: str

  # --- required: run metadata --------------------------------------------
  repetitions: int
  successful_runs: int
  timestamp: str
  max_generations_configured: int

  # --- required: execution time ------------------------------------------
  mean_time: float
  std_time: float
  min_time: float
  max_time: float

  # --- required: generations ---------------------------------------------
  mean_generations: float
  std_generations: float

  # --- required: raw arrays (exact reproduction of the runs) --------------
  raw_times: list[float]
  raw_costs: list[float]

  # --- required: configuration -------------------------------------------
  algorithm_config: JsonPayload
  strategies: JsonPayload

  # --- default-backed / nullable: problem type & solution quality ---------
  problem_type: NotRequired[str | None]
  mean_cost: NotRequired[float | None]
  std_cost: NotRequired[float | None]
  best_cost: NotRequired[float | None]
  worst_cost: NotRequired[float | None]

  # --- nullable: optimal cost & gap (absent when optimum is unknown) ------
  optimal_cost: NotRequired[float | None]
  mean_gap: NotRequired[float | None]
  std_gap: NotRequired[float | None]
  best_gap: NotRequired[float | None]
  raw_gaps: NotRequired[list[float | None] | None]

  # --- nullable: VRP feasibility & fleet metrics (DP17-4) -----------------
  feasible_runs: NotRequired[int | None]
  feasibility_rate: NotRequired[float | None]
  mean_vehicles: NotRequired[float | None]
  best_vehicles: NotRequired[int | None]
  raw_is_feasible: NotRequired[list[bool] | None]
  raw_unpenalized_costs: NotRequired[list[float] | None]
  raw_violations: NotRequired[list[float] | None]
  raw_num_vehicles: NotRequired[list[int] | None]

  # --- nullable: absent from older checkpoints ---------------------------
  mean_initial_cost: NotRequired[float | None]
  std_initial_cost: NotRequired[float | None]
  mean_improvement: NotRequired[float | None]
  min_generations: NotRequired[int | None]
  max_generations: NotRequired[int | None]
  mean_h2d_mb: NotRequired[float | None]
  mean_d2h_mb: NotRequired[float | None]
  total_transfer_mb: NotRequired[float | None]
  mean_kernels: NotRequired[float | None]
  mean_phase_timings: NotRequired[JsonPayload | None]
  phase_share_pct: NotRequired[JsonPayload | None]
  phase_share_cv_pct: NotRequired[JsonPayload | None]
  raw_initial_costs: NotRequired[list[float] | None]
  raw_generations: NotRequired[list[int] | None]
  raw_stop_reasons: NotRequired[list[str] | None]
  construction_heuristic: NotRequired[str | None]
