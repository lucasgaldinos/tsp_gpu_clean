"""Unit tests for the DuckDB writer/reader in `benchmarking.database.db_operations`.

WHAT: Exercises `create_database` (file creation, schema, sequence, idempotent
reopen), `validate_benchmark_data` (every required key, every NaN-checked field,
every raw array, with the exact exception message), `insert_benchmark_run`
(sequential ids, verbatim persistence, JSON/LIST columns, generated `is_test`
flag, timestamp formats) and the query helpers `get_algorithm_summary`,
`get_problems_with_cpu` and `compute_speedup_matrix`.
WHY: This module is the only path from a checkpoint payload into the results
database, and every downstream table, figure and speedup claim reads the values
it writes. A silently shifted column, a dropped nullable field or a validator
that lets a malformed payload through corrupts the chapter results without any
test noticing.
EXPECTED: Each write is readable back with the value it was given, every
malformed payload raises the documented `ValueError` with the exact message,
and no file is created outside `tmp_path`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from benchmarking.database.db_operations import (
  compute_run_aggregates,
  compute_speedup_matrix,
  create_database,
  get_algorithm_summary,
  get_problems_with_cpu,
  insert_benchmark_run,
  validate_benchmark_data,
)
from data_models.records import BenchmarkRunInput

if TYPE_CHECKING:
  from collections.abc import Iterator
  from pathlib import Path

  import duckdb

TIMESTAMP_ISO = "2025-11-23T12:00:00"
EXPECTED_TIMESTAMP = datetime(2025, 11, 23, 12, 0)

#: Fields the validator rejects when they carry NaN (`db_operations.validate_benchmark_data`).
NAN_CHECKED_REQUIRED_FIELDS: list[str] = [
  "mean_time",
  "std_time",
  "min_time",
  "max_time",
  "mean_generations",
  "std_generations",
]

#: Nullable numeric fields that are NaN-checked only when present.
NAN_CHECKED_OPTIONAL_FIELDS: list[str] = [
  "mean_cost",
  "std_cost",
  "best_cost",
  "worst_cost",
  "optimal_cost",
  "mean_gap",
  "std_gap",
  "best_gap",
  "mean_initial_cost",
  "std_initial_cost",
  "mean_improvement",
  "mean_h2d_mb",
  "mean_d2h_mb",
  "total_transfer_mb",
  "mean_kernels",
  "feasibility_rate",
  "mean_vehicles",
]

#: Required raw arrays whose length must equal `repetitions`.
REQUIRED_RAW_ARRAYS: list[str] = ["raw_times", "raw_costs"]

#: Nullable raw arrays whose length must equal `repetitions` when present.
OPTIONAL_RAW_ARRAYS: list[str] = [
  "raw_gaps",
  "raw_initial_costs",
  "raw_generations",
  "raw_stop_reasons",
  "raw_is_feasible",
  "raw_unpenalized_costs",
  "raw_violations",
  "raw_num_vehicles",
]

#: Either a mapping or a pre-serialised JSON string is accepted for these columns.
JSON_PAYLOAD: dict[str, int | float] = {"population_size": 256, "mutation_rate": 0.02}


def run_payload(**overrides: Any) -> dict[str, Any]:
  """Build a complete, valid checkpoint payload.

  Args:
      **overrides: Field values replacing the defaults; passing `field=None`
          deliberately exercises the nullable columns.

  Returns:
      A payload carrying every required `BenchmarkRunInput` key.
  """
  payload: dict[str, Any] = {
    "problem_name": "berlin52",
    "problem_size": 52,
    "optimal_cost": 7542.0,
    "algorithm": "CPU",
    "backend": "CPU",
    "repetitions": 3,
    "successful_runs": 3,
    "timestamp": TIMESTAMP_ISO,
    "max_generations_configured": 500,
    "mean_time": 1.5,
    "std_time": 0.1,
    "min_time": 1.4,
    "max_time": 1.6,
    "mean_cost": 7600.0,
    "std_cost": 5.0,
    "best_cost": 7590.0,
    "worst_cost": 7610.0,
    "mean_gap": 0.77,
    "std_gap": 0.1,
    "best_gap": 0.63,
    "mean_generations": 120.0,
    "std_generations": 3.0,
    "raw_times": [1.4, 1.5, 1.6],
    "raw_costs": [7590.0, 7600.0, 7610.0],
    "raw_gaps": [0.63, 0.77, 0.9],
    "algorithm_config": JSON_PAYLOAD,
    "strategies": {"selection": "Tournament", "crossover": "Order"},
  }
  payload.update(overrides)
  return payload


def production_payload(mean_time: float, timestamp: str, **overrides: Any) -> dict[str, Any]:
  """Build a payload with `repetitions=15`, so the generated `is_test` flag is false."""
  reps = 15
  defaults: dict[str, Any] = {
    "repetitions": reps,
    "successful_runs": reps,
    "mean_time": mean_time,
    "timestamp": timestamp,
    "raw_times": [mean_time] * reps,
    "raw_costs": [7600.0] * reps,
    "raw_gaps": [0.5] * reps,
  }
  defaults.update(overrides)
  return run_payload(**defaults)


def insert_payload(conn: duckdb.DuckDBPyConnection, payload: dict[str, Any]) -> int:
  """Validate `payload`, insert it and return the new `run_id`."""
  return insert_benchmark_run(conn, validate_benchmark_data(payload))


def insert_run(conn: duckdb.DuckDBPyConnection, **overrides: Any) -> int:
  """Validate `run_payload(**overrides)`, insert it and return the new `run_id`."""
  return insert_payload(conn, run_payload(**overrides))


def validation_message(payload: dict[str, Any]) -> str:
  """Return the message of the `ValueError` raised by `validate_benchmark_data`."""
  with pytest.raises(ValueError) as excinfo:
    validate_benchmark_data(payload)
  return str(excinfo.value)


def describe_columns(conn: duckdb.DuckDBPyConnection) -> dict[str, tuple[str, str, str]]:
  """Map each `benchmark_runs` column to its `(type, nullable, key)` DESCRIBE triple."""
  rows = conn.execute("DESCRIBE benchmark_runs").fetchall()
  return {row[0]: (row[1], row[2], row[3]) for row in rows}


def fetch_run(conn: duckdb.DuckDBPyConnection, run_id: int) -> dict[str, Any]:
  """Fetch one inserted run keyed by column name."""
  cursor = conn.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", [run_id])
  row: tuple[Any, ...] | None = cursor.fetchone()
  assert row is not None, f"run {run_id} was not persisted"
  columns: list[str] = [column[0] for column in cursor.description or []]
  return dict(zip(columns, row, strict=True))


@pytest.fixture
def db_conn(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
  """Connection to a schema-initialised database file inside `tmp_path`."""
  conn = create_database(str(tmp_path / "results.duckdb"))
  yield conn
  conn.close()


# ============================================================================
# Schema creation
# ============================================================================


def test_create_database_creates_runs_table_and_sequence(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: `create_database` on a fresh path inside `tmp_path`.

  WHY: the sequence supplies `run_id` through the column default; without it
  every insert fails, and the table name is the contract the rest of the module
  interpolates into its SQL.
  EXPECTED: `benchmark_runs` is described with 45 columns and
  `nextval('benchmark_runs_seq')` yields the first id, 1.
  """
  columns: dict[str, tuple[str, str, str]] = describe_columns(db_conn)
  first_id: tuple[Any, ...] | None = db_conn.execute("SELECT nextval('benchmark_runs_seq')").fetchone()

  assert len(columns) == 54, sorted(columns)
  assert "run_id" in columns
  assert "problem_type" in columns
  assert first_id == (1,)


@pytest.mark.parametrize(
  "field",
  [pytest.param(field, id=field) for field in sorted(BenchmarkRunInput.__required_keys__)],
)
def test_schema_declares_every_required_typed_dict_field_not_null(
  db_conn: duckdb.DuckDBPyConnection, field: str
) -> None:
  """WHAT: one required `BenchmarkRunInput` key per case against `DESCRIBE benchmark_runs`.

  WHY: `records.BenchmarkRunInput` and `schema.sql` claim to mirror each other;
  a key that is required in the TypedDict but nullable (or absent) in SQL turns
  a missing value into a silent NULL instead of a validation error.
  EXPECTED: the column exists and DESCRIBE reports it `NOT NULL` (`NO`).
  """
  columns: dict[str, tuple[str, str, str]] = describe_columns(db_conn)

  assert field in columns, f"{field} is required by BenchmarkRunInput but missing from the schema"
  _, nullable, _ = columns[field]
  assert nullable == "NO", f"{field} is required by BenchmarkRunInput but nullable in the schema"


@pytest.mark.parametrize(
  ("column", "expected_type"),
  [
    pytest.param("run_id", "INTEGER", id="run_id"),
    pytest.param("problem_name", "VARCHAR", id="problem_name"),
    pytest.param("optimal_cost", "DOUBLE", id="optimal_cost"),
    pytest.param("timestamp", "TIMESTAMP", id="timestamp"),
    pytest.param("repetitions", "INTEGER", id="repetitions"),
    pytest.param("mean_generations", "DOUBLE", id="mean_generations"),
    pytest.param("raw_times", "DOUBLE[]", id="raw_times"),
    pytest.param("raw_costs", "DOUBLE[]", id="raw_costs"),
    pytest.param("raw_gaps", "DOUBLE[]", id="raw_gaps"),
    pytest.param("raw_initial_costs", "DOUBLE[]", id="raw_initial_costs"),
    pytest.param("raw_generations", "INTEGER[]", id="raw_generations"),
    pytest.param("raw_stop_reasons", "VARCHAR[]", id="raw_stop_reasons"),
    pytest.param("algorithm_config", "JSON", id="algorithm_config"),
    pytest.param("strategies", "JSON", id="strategies"),
    pytest.param("construction_heuristic", "VARCHAR", id="construction_heuristic"),
    pytest.param("is_test", "BOOLEAN", id="is_test_generated_column"),
  ],
)
def test_schema_uses_native_duckdb_types(db_conn: duckdb.DuckDBPyConnection, column: str, expected_type: str) -> None:
  """WHAT: the declared SQL type of every column whose type carries meaning.

  WHY: JSON and LIST types are what let queries slice `raw_times` and aggregate
  `algorithm_config` without re-parsing text; a VARCHAR regression would still
  insert successfully, so only the schema assertion catches it.
  EXPECTED: DESCRIBE reports the documented native type for each column.
  """
  columns: dict[str, tuple[str, str, str]] = describe_columns(db_conn)

  assert columns[column][0] == expected_type


def test_create_database_reopens_existing_file_without_losing_data(tmp_path: Path) -> None:
  """WHAT: inserting a run, closing, then calling `create_database` again on the same path.

  WHY: `import_checkpoints` opens the database once per import; `IF NOT EXISTS`
  must be idempotent, and the sequence must continue where it stopped instead of
  reusing `run_id` 1 for a different run.
  EXPECTED: the row survives the reopen and the next insert receives `run_id` 2.
  """
  db_path: Path | str = tmp_path / "results.duckdb"

  first_conn = create_database(str(db_path))
  first_id = insert_run(first_conn)
  first_conn.close()

  second_conn = create_database(str(db_path))
  try:
    row_count = second_conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
    second_id = insert_run(second_conn)
  finally:
    second_conn.close()

  assert (first_id, second_id) == (1, 2)
  assert row_count == (1,)


# ============================================================================
# Validation
# ============================================================================


def test_validate_benchmark_data_returns_the_supplied_mapping() -> None:
  """WHAT: `validate_benchmark_data` on a complete payload.

  WHY: the writer indexes the returned value by key; an implementation that
  copied a subset of keys would break callers that rely on optional fields.
  EXPECTED: the same object comes back with every key intact.
  """
  payload = run_payload()

  record = validate_benchmark_data(payload)

  assert record is payload
  assert set(record) == set(payload)


@pytest.mark.parametrize(
  "field",
  [pytest.param(field, id=field) for field in sorted(BenchmarkRunInput.__required_keys__)],
)
def test_validate_benchmark_data_names_each_missing_required_field(field: str) -> None:
  """WHAT: one removed required key per case.

  WHY: the message is what an operator reads when an import fails; it must name
  the exact key, and the validator derives its list from
  `BenchmarkRunInput.__required_keys__`, so the two cannot drift.
  EXPECTED: `ValueError: Missing required fields: <field>`.
  """
  payload = run_payload()
  del payload[field]

  assert validation_message(payload) == f"Missing required fields: {field}"


@pytest.mark.parametrize("field", NAN_CHECKED_REQUIRED_FIELDS)
def test_validate_benchmark_data_rejects_nan_in_required_numeric_field(field: str) -> None:
  """WHAT: each NaN-checked required aggregate field set to `float('nan')`.

  WHY: NaN compares false against every bound, so a NaN aggregate passes every
  downstream filter and rank while corrupting the reported mean.
  EXPECTED: `ValueError: Field '<field>' cannot be NaN`.
  """
  payload = run_payload(**{field: float("nan")})

  assert validation_message(payload) == f"Field '{field}' cannot be NaN"


@pytest.mark.parametrize("field", NAN_CHECKED_OPTIONAL_FIELDS)
def test_validate_benchmark_data_rejects_nan_in_optional_numeric_field(field: str) -> None:
  """WHAT: each NaN-checked nullable numeric field set to `float('nan')`.

  WHY: these columns exist only for newer checkpoints; a NaN slipping through
  here reaches the speedup and memory-transfer tables just like a required one.
  EXPECTED: `ValueError: Optional field '<field>' cannot be NaN`.
  """
  payload = run_payload(**{field: float("nan")})

  assert validation_message(payload) == f"Optional field '{field}' cannot be NaN"


#: A valid three-element value per raw-array field, used to build the truncated payload.
RAW_ARRAY_VALUES: dict[str, list[Any]] = {
  "raw_times": [1.4, 1.5, 1.6],
  "raw_costs": [7590.0, 7600.0, 7610.0],
  "raw_gaps": [0.63, 0.77, 0.9],
  "raw_initial_costs": [8190.0, 8200.0, 8210.0],
  "raw_generations": [90, 120, 150],
  "raw_stop_reasons": ["completed", "completed", "optimal reached"],
}


@pytest.mark.parametrize(
  ("field", "optional"),
  [
    pytest.param("raw_times", False, id="raw_times"),
    pytest.param("raw_costs", False, id="raw_costs"),
    pytest.param("raw_gaps", True, id="optional-raw_gaps"),
    pytest.param("raw_initial_costs", True, id="optional-raw_initial_costs"),
    pytest.param("raw_generations", True, id="optional-raw_generations"),
    pytest.param("raw_stop_reasons", True, id="optional-raw_stop_reasons"),
  ],
)
def test_validate_benchmark_data_rejects_raw_array_shorter_than_repetitions(field: str, optional: bool) -> None:
  """WHAT: every required and optional raw array truncated to two elements with `repetitions=3`.

  WHY: the raw arrays are the exact-reproduction ledger; a short array makes the
  per-run table index past its end, and the mismatch is only detectable here.
  EXPECTED: `ValueError` naming the field, the expected length and the actual one.
  """
  payload = run_payload(**{field: RAW_ARRAY_VALUES[field][:2]})

  prefix = "Array length mismatch for optional" if optional else "Array length mismatch for"
  assert validation_message(payload) == f"{prefix} '{field}': expected 3, got 2"


def test_insert_benchmark_run_raises_key_error_for_unvalidated_payload(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: `insert_benchmark_run` called directly with a payload missing `mean_time`.

  WHY: the writer reads required keys with `data[...]` and nullable ones with
  `dict.get`; callers that skip `validate_benchmark_data` must fail fast on the
  first missing key rather than insert a NULL into a NOT NULL column.
  EXPECTED: `KeyError('mean_time')`, and the table stays empty.
  """
  payload = run_payload()
  del payload["mean_time"]

  with pytest.raises(KeyError) as excinfo:
    # The cast is the point of the test: bypass the validator on purpose so the
    # writer's own indexing behaviour is observable.
    insert_benchmark_run(db_conn, cast("BenchmarkRunInput", payload))

  assert excinfo.value.args == ("mean_time",)
  assert db_conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone() == (0,)


# ============================================================================
# Insert
# ============================================================================


def test_insert_benchmark_run_returns_sequential_run_ids(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: two consecutive inserts into the same database.

  WHY: `run_id` is the join key every consumer uses to find a run; a duplicate
  id silently overwrites the earlier row in downstream joins.
  EXPECTED: ids 1 and 2, in that order.
  """
  first_id = insert_run(db_conn)
  second_id = insert_run(db_conn, algorithm="HybridOptimized")

  assert (first_id, second_id) == (1, 2)


def test_insert_benchmark_run_persists_every_required_value(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: one valid run inserted and read back column by column.

  WHY: the insert binds 40 positional parameters; one transposition would store
  a plausible-looking wrong value (for example `min_time` in `max_time`) that
  only shows up in a report nobody re-derives.
  EXPECTED: every scalar, aggregate, list and JSON column equals the payload
  value, and every nullable column the payload omitted is NULL.
  """
  payload = run_payload()

  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)

  assert stored["run_id"] == run_id
  assert stored["is_test"] is True, "repetitions=3 must mark the run as a test run"

  for field in ("problem_name", "algorithm", "backend"):
    assert stored[field] == payload[field], f"{field} was not persisted verbatim"
  for field in ("problem_size", "repetitions", "successful_runs", "max_generations_configured"):
    assert stored[field] == payload[field], f"{field} was not persisted verbatim"
  for field in ("optimal_cost", *NAN_CHECKED_REQUIRED_FIELDS):
    assert stored[field] == pytest.approx(payload[field]), f"{field} was not persisted"
  for field in REQUIRED_RAW_ARRAYS:
    assert stored[field] == payload[field], f"{field} was not persisted verbatim"
  assert stored["timestamp"] == EXPECTED_TIMESTAMP

  assert json.loads(stored["algorithm_config"]) == payload["algorithm_config"]
  assert json.loads(stored["strategies"]) == payload["strategies"]

  omitted_nullables = (
    "mean_initial_cost",
    "std_initial_cost",
    "mean_improvement",
    "min_generations",
    "max_generations",
    "mean_h2d_mb",
    "mean_d2h_mb",
    "total_transfer_mb",
    "mean_kernels",
    "raw_initial_costs",
    "raw_generations",
    "raw_stop_reasons",
    "construction_heuristic",
  )
  for field in omitted_nullables:
    assert stored[field] is None, f"{field} must be NULL when the payload omits it"


def test_insert_benchmark_run_allows_null_optimal_and_gap_fields(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: inserting a run for an arbitrary `.tsp` instance whose optimal cost is unknown.

  WHY: Custom `.tsp` files without a known optimal cost emit `None` for `optimal_cost`
  and gap fields; DuckDB must persist them as SQL NULL without failing validation.
  EXPECTED: insert succeeds and all five optimal/gap columns read back as `None`.
  """
  payload = run_payload(
    optimal_cost=None,
    mean_gap=None,
    std_gap=None,
    best_gap=None,
    raw_gaps=None,
  )
  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)
  for field in ("optimal_cost", "mean_gap", "std_gap", "best_gap", "raw_gaps"):
    assert stored[field] is None, f"{field} must round-trip as NULL"


def test_insert_benchmark_run_persists_optional_fields(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: one run carrying every nullable column.

  WHY: the writer reads these with `dict.get`, so a typo in the key would
  silently store NULL and the GPU memory-transfer figures would disappear from
  every table while the import still reports success.
  EXPECTED: each supplied nullable value is read back unchanged.
  """
  payload = run_payload(
    mean_initial_cost=8200.0,
    std_initial_cost=50.0,
    mean_improvement=7.3,
    min_generations=90,
    max_generations=150,
    mean_h2d_mb=1.5,
    mean_d2h_mb=0.25,
    total_transfer_mb=1.75,
    mean_kernels=230.0,
    raw_initial_costs=[8190.0, 8200.0, 8210.0],
    raw_generations=[90, 120, 150],
    raw_stop_reasons=["completed", "completed", "optimal reached"],
    construction_heuristic="nearest_neighbor",
  )

  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)

  for field in (
    "mean_initial_cost",
    "std_initial_cost",
    "mean_improvement",
    "min_generations",
    "max_generations",
    "mean_h2d_mb",
    "mean_d2h_mb",
    "total_transfer_mb",
    "mean_kernels",
    "raw_initial_costs",
    "raw_generations",
    "raw_stop_reasons",
    "construction_heuristic",
  ):
    assert stored[field] == payload[field], f"{field} was not persisted verbatim"


@pytest.mark.parametrize(
  "config_payload",
  [
    pytest.param(JSON_PAYLOAD, id="mapping"),
    pytest.param(json.dumps(JSON_PAYLOAD), id="pre_serialised_json"),
  ],
)
def test_insert_benchmark_run_serialises_json_columns(
  db_conn: duckdb.DuckDBPyConnection, config_payload: object
) -> None:
  """WHAT: `algorithm_config` and `strategies` supplied both as a dict and as JSON text.

  WHY: checkpoints store these payloads as parsed objects while re-imported rows
  may already be strings; both must land as parsed JSON, never as a JSON string
  literal nested in a VARCHAR.
  EXPECTED: reading the column back and parsing it yields the original mapping.
  """
  payload = run_payload(algorithm_config=config_payload, strategies=config_payload)

  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)

  assert json.loads(stored["algorithm_config"]) == JSON_PAYLOAD
  assert json.loads(stored["strategies"]) == JSON_PAYLOAD


def test_insert_benchmark_run_persists_phase_telemetry(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: `mean_phase_timings`, `phase_share_pct` and `phase_share_cv_pct` round trip.

  WHY: `DP11-7`/`BUG-30` — the runner computes the phase breakdown but the schema had
  no column for it, so every import silently dropped the only evidence Chapter 4 has
  for where the wall time goes. Absent payloads must still import as SQL `NULL`.
  EXPECTED: the three JSON columns read back parsed and identical to the payload;
  a payload without them stores `NULL` instead of failing the insert.
  """
  phase_timings: dict[str, dict[str, float]] = {
    "initialization": {"total_ms": 2.0, "calls": 1, "mean_ms": 2.0},
    "evaluation": {"total_ms": 15.0, "calls": 20, "mean_ms": 0.75},
    "survival": {"total_ms": 4.0, "calls": 20, "mean_ms": 0.2},
    "host_sync": {"total_ms": 3.0, "calls": 21, "mean_ms": 0.14},
  }
  share_pct: dict[str, float] = {"initialization": 4.0, "evaluation": 30.0, "survival": 8.0, "host_sync": 6.0}
  share_cv_pct: dict[str, float] = {"initialization": 2.5, "evaluation": 1.2, "survival": 3.3, "host_sync": 4.0}
  payload = run_payload(
    mean_phase_timings=phase_timings,
    phase_share_pct=share_pct,
    phase_share_cv_pct=share_cv_pct,
  )

  stored = fetch_run(db_conn, insert_benchmark_run(db_conn, validate_benchmark_data(payload)))
  assert json.loads(stored["mean_phase_timings"]) == phase_timings
  assert json.loads(stored["phase_share_pct"]) == share_pct
  assert json.loads(stored["phase_share_cv_pct"]) == share_cv_pct

  legacy = fetch_run(db_conn, insert_benchmark_run(db_conn, validate_benchmark_data(run_payload())))
  assert legacy["mean_phase_timings"] is None
  assert legacy["phase_share_pct"] is None
  assert legacy["phase_share_cv_pct"] is None


@pytest.mark.parametrize(
  ("repetitions", "expected_is_test"),
  [
    pytest.param(1, True, id="single-run"),
    pytest.param(5, True, id="last-test-repetition"),
    pytest.param(6, False, id="first-production-repetition"),
    pytest.param(15, False, id="production-run"),
  ],
)
def test_insert_benchmark_run_derives_is_test_from_repetitions(
  db_conn: duckdb.DuckDBPyConnection, repetitions: int, expected_is_test: bool
) -> None:
  """WHAT: the generated `is_test` column around its `repetitions <= 5` boundary.

  WHY: `get_problems_with_cpu` and `compute_speedup_matrix` filter on this
  column; a boundary off-by-one would either ban real runs from the speedup
  table or admit 2-repetition smoke tests into the thesis figures.
  EXPECTED: `is_test` is true at 1 and 5 repetitions, false at 6 and 15.
  """
  payload = run_payload(
    repetitions=repetitions,
    successful_runs=repetitions,
    raw_times=[1.5] * repetitions,
    raw_costs=[7600.0] * repetitions,
    raw_gaps=[0.5] * repetitions,
  )

  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)

  assert stored["is_test"] is expected_is_test


@pytest.mark.parametrize(
  ("timestamp", "expected"),
  [
    pytest.param(TIMESTAMP_ISO, EXPECTED_TIMESTAMP, id="iso-t-separator"),
    pytest.param("2025-11-23 12:00:00", EXPECTED_TIMESTAMP, id="space-separated"),
    pytest.param(datetime(2025, 11, 23, 12, 0), EXPECTED_TIMESTAMP, id="datetime-object"),
  ],
)
def test_insert_benchmark_run_parses_every_timestamp_form(
  db_conn: duckdb.DuckDBPyConnection, timestamp: object, expected: datetime
) -> None:
  """WHAT: the three timestamp forms `insert_benchmark_run` accepts.

  WHY: the writer branches on `isinstance(ts_raw, str)` and then on two parse
  formats; an unhandled form raises mid-import, after earlier checkpoints have
  already been written.
  EXPECTED: every form is stored as the same naive `datetime`.
  """
  payload = run_payload(timestamp=timestamp)

  run_id = insert_benchmark_run(db_conn, validate_benchmark_data(payload))
  stored = fetch_run(db_conn, run_id)

  assert stored["timestamp"] == expected


# ============================================================================
# Queries
# ============================================================================


def test_get_algorithm_summary_returns_the_inserted_values(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: `get_algorithm_summary` for a problem/algorithm pair that was just inserted.

  WHY: the summary is what the comparison tables read; returning a different
  row's statistics would be indistinguishable from a real regression.
  EXPECTED: the nine documented keys carry the inserted run's values.
  """
  payload = run_payload()
  insert_benchmark_run(db_conn, validate_benchmark_data(payload))

  summary = get_algorithm_summary(db_conn, "berlin52", "CPU")

  assert summary is not None
  assert set(summary) == {
    "mean_time",
    "std_time",
    "mean_cost",
    "std_cost",
    "best_cost",
    "best_gap",
    "repetitions",
    "is_test",
    "timestamp",
  }
  assert summary["mean_time"] == pytest.approx(payload["mean_time"])
  assert summary["std_time"] == pytest.approx(payload["std_time"])
  assert summary["mean_cost"] == pytest.approx(payload["mean_cost"])
  assert summary["std_cost"] == pytest.approx(payload["std_cost"])
  assert summary["best_cost"] == pytest.approx(payload["best_cost"])
  assert summary["best_gap"] == pytest.approx(payload["best_gap"])
  assert summary["repetitions"] == payload["repetitions"]
  assert summary["is_test"] is True
  assert summary["timestamp"] == EXPECTED_TIMESTAMP


def test_get_algorithm_summary_returns_none_when_no_run_matches(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: `get_algorithm_summary` against an empty table and against a missing algorithm.

  WHY: callers branch on `None` to print "not run yet"; an empty dict or an
  exception would both turn a missing benchmark into a crash.
  EXPECTED: `None` in both cases, while the inserted run stays queryable.
  """
  empty_summary = get_algorithm_summary(db_conn, "berlin52", "CPU")

  insert_run(db_conn)
  missing_algorithm = get_algorithm_summary(db_conn, "berlin52", "FullGPUMonolithic")

  assert empty_summary is None
  assert missing_algorithm is None


def test_get_algorithm_summary_returns_the_most_recent_run(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: two runs of the same pair, the newer one inserted first.

  WHY: `ORDER BY timestamp DESC` is the only tie-break; without it the result
  follows insertion order and re-importing an older checkpoint would silently
  become the reported result.
  EXPECTED: the 12:00 run is returned, not the 10:00 run.
  """
  insert_run(db_conn, mean_time=1.0, timestamp="2025-11-23T12:00:00")
  insert_run(db_conn, mean_time=2.0, timestamp="2025-11-23T10:00:00")

  summary = get_algorithm_summary(db_conn, "berlin52", "CPU")

  assert summary is not None
  assert summary["mean_time"] == pytest.approx(1.0)
  assert summary["timestamp"] == datetime(2025, 11, 23, 12, 0)


def test_get_problems_with_cpu_lists_distinct_cpu_problems_by_size(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: four runs across two problems, one of them GPU-only and one problem repeated.

  WHY: the helper feeds the "problems available for comparison" list; a missing
  `DISTINCT` duplicates entries and a missing filter hides GPU-only problems as
  if their CPU baseline existed.
  EXPECTED: `['berlin52', 'st70']`, ordered by problem size and deduplicated.
  """
  insert_run(db_conn)  # CPU, berlin52, size 52
  insert_run(db_conn, timestamp="2025-11-23T13:00:00")  # CPU, berlin52 again
  insert_run(db_conn, problem_name="st70", problem_size=70)  # CPU, size 70
  insert_run(db_conn, problem_name="eil76", problem_size=76, algorithm="FullGPUMonolithic", backend="GPU")

  assert get_problems_with_cpu(db_conn) == ["berlin52", "st70"]


def test_compute_speedup_matrix_reports_ratio_to_baseline(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: a 2.0 s CPU baseline, a 0.5 s production GPU run and a 0.1 s GPU test run.

  WHY: the speedup table is a headline result; it must divide by the baseline
  of the same problem and ignore smoke runs, otherwise a 3-repetition test
  records a fictitious 20x speedup.
  EXPECTED: one `(berlin52, FullGPU, 4.0, 2.0)` entry; the unrun algorithm is
  skipped rather than reported as zero.
  """
  insert_payload(db_conn, production_payload(mean_time=2.0, timestamp="2025-11-23T10:00:00"))
  insert_payload(
    db_conn,
    production_payload(mean_time=0.5, timestamp="2025-11-23T10:00:00", algorithm="FullGPUMonolithic", backend="GPU"),
  )
  insert_run(db_conn, algorithm="FullGPUMonolithic", backend="GPU", mean_time=0.1, timestamp="2025-11-23T12:00:00")

  speedups = compute_speedup_matrix(db_conn, "CPU", ["FullGPUMonolithic", "HybridOptimized"])
  default_speedups = compute_speedup_matrix(db_conn)

  assert [entry[:2] for entry in speedups] == [("berlin52", "FullGPUMonolithic")]
  assert speedups[0][2] == pytest.approx(4.0)
  assert speedups[0][3] == pytest.approx(2.0)
  assert [entry[:2] for entry in default_speedups] == [("berlin52", "FullGPUMonolithic")]
  assert default_speedups[0][2] == pytest.approx(4.0)
  assert default_speedups[0][3] == pytest.approx(2.0)


# ============================================================================
# BUG-36: Synchronization of successful_runs and averages in partial executions
# ============================================================================


def test_validate_benchmark_data_accepts_partial_runs_matching_successful_runs() -> None:
  """WHAT: Checkpoint payload where repetitions=5 but successful_runs=3 and arrays have length 3.

  WHY: BUG-36 — Validating len(arr) against repetitions broke on partial/cancelled runs.
  EXPECTED: Validates cleanly and returns validated BenchmarkRunInput.
  """
  payload = run_payload(
    repetitions=5,
    successful_runs=3,
    raw_times=[1.0, 1.2, 1.1],
    raw_costs=[7550.0, 7560.0, 7570.0],
    raw_gaps=[0.1, 0.2, 0.3],
    raw_initial_costs=[8000.0, 8050.0, 8100.0],
    raw_generations=[100, 110, 120],
    raw_stop_reasons=["completed", "completed", "completed"],
  )
  validated = validate_benchmark_data(payload)
  assert validated["repetitions"] == 5
  assert validated["successful_runs"] == 3
  assert len(validated["raw_times"]) == 3


def test_validate_benchmark_data_rejects_raw_array_mismatched_to_successful_runs() -> None:
  """WHAT: Checkpoint payload with successful_runs=3 but raw_times has 2 items.

  WHY: BUG-36 — Raw arrays must exactly equal successful_runs.
  EXPECTED: ValueError indicating expected 3, got 2.
  """
  payload = run_payload(
    repetitions=5,
    successful_runs=3,
    raw_times=[1.0, 1.2],
    raw_costs=[7550.0, 7560.0, 7570.0],
    raw_gaps=[0.1, 0.2, 0.3],
  )
  assert validation_message(payload) == "Array length mismatch for 'raw_times': expected 3, got 2"


def test_validate_benchmark_data_rejects_successful_runs_exceeding_repetitions() -> None:
  """WHAT: Checkpoint payload with successful_runs=6 but repetitions=5.

  WHY: BUG-36 — Successful runs cannot exceed total attempted repetitions.
  EXPECTED: ValueError naming the invalid count.
  """
  payload = run_payload(
    repetitions=5,
    successful_runs=6,
  )
  assert validation_message(payload) == "successful_runs (6) cannot be negative or exceed repetitions (5)"


def test_compute_run_aggregates_uses_successful_runs_denominator(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: 3 successful runs out of 5 repetitions.

  WHY: BUG-36 — Averages must divide strictly by successful_runs, never repetitions.
  EXPECTED: Mean time is exactly sum(times)/3, mean cost is sum(costs)/3;
  inserting into DuckDB persists consistent values.
  """
  raw_times = [1.0, 2.0, 3.0]
  raw_costs = [100.0, 110.0, 120.0]
  raw_gaps = [1.0, 2.0, 3.0]
  raw_initial = [150.0, 150.0, 150.0]
  raw_gens = [10, 20, 30]

  aggs = compute_run_aggregates(
    raw_times=raw_times,
    raw_costs=raw_costs,
    raw_gaps=raw_gaps,
    optimal_cost=100.0,
    repetitions=5,
    raw_initial_costs=raw_initial,
    raw_generations=raw_gens,
  )

  assert aggs["repetitions"] == 5
  assert aggs["successful_runs"] == 3
  # sum([1, 2, 3]) / 3 = 2.0 (if divided by 5 it would be 1.2)
  assert aggs["mean_time"] == pytest.approx(2.0)
  # sum([100, 110, 120]) / 3 = 110.0
  assert aggs["mean_cost"] == pytest.approx(110.0)
  # sum([1, 2, 3]) / 3 = 2.0
  assert aggs["mean_gap"] == pytest.approx(2.0)
  assert aggs["mean_generations"] == pytest.approx(20.0)

  # Check that inserting into database succeeds
  full_payload = run_payload(**aggs)
  validated = validate_benchmark_data(full_payload)
  run_id = insert_benchmark_run(db_conn, validated)
  assert run_id > 0

  summary = get_algorithm_summary(db_conn, "berlin52", "CPU")
  assert summary is not None
  assert summary["mean_time"] == pytest.approx(2.0)
  assert summary["mean_cost"] == pytest.approx(110.0)
  assert summary["repetitions"] == 5


def test_compute_run_aggregates_zero_successful_runs_returns_graceful_defaults(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: 0 successful runs out of 5 repetitions (all failed/cancelled).

  WHY: BUG-36 — Division by zero must be prevented and column nullability respected.
  EXPECTED: Returns 0.0 for NOT NULL columns, None for nullable columns,
  validates without NaN error, and can be inserted into database.
  """
  aggs = compute_run_aggregates(
    raw_times=[],
    raw_costs=[],
    raw_gaps=[],
    optimal_cost=100.0,
    repetitions=5,
  )

  assert aggs["repetitions"] == 5
  assert aggs["successful_runs"] == 0
  assert aggs["mean_time"] == 0.0
  assert aggs["mean_cost"] == 0.0
  assert aggs["mean_gap"] == 0.0
  assert aggs["mean_initial_cost"] is None
  assert aggs["mean_improvement"] is None

  full_payload = run_payload(**aggs)
  validated = validate_benchmark_data(full_payload)
  run_id = insert_benchmark_run(db_conn, validated)
  assert run_id > 0

  summary = get_algorithm_summary(db_conn, "berlin52", "CPU")
  assert summary is not None
  assert summary["mean_time"] == 0.0
  assert summary["mean_cost"] == 0.0


# ============================================================================
# BUG-39: Single Query Speedup Matrix and FullGPUModular Inclusion
# ============================================================================


def test_compute_speedup_matrix_executes_single_query(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: compute_speedup_matrix executes a single analytical GROUP BY query.

  WHY: BUG-39 — The previous N+1 query loop incurred redundant latency across problems.
  EXPECTED: Exact execute call count is 1.
  """
  insert_payload(
    db_conn,
    production_payload(mean_time=2.0, problem_name="berlin52", algorithm="CPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(
      mean_time=0.5,
      problem_name="berlin52",
      algorithm="FullGPUModular",
      backend="GPU",
      timestamp="2025-11-23T10:00:00",
    ),
  )
  insert_payload(
    db_conn,
    production_payload(mean_time=4.0, problem_name="st70", algorithm="CPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(
      mean_time=1.0,
      problem_name="st70",
      algorithm="FullGPUModular",
      backend="GPU",
      timestamp="2025-11-23T10:00:00",
    ),
  )

  class ConnSpy:
    def __init__(self, target: duckdb.DuckDBPyConnection) -> None:
      self._target = target
      self.execute_count = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
      self.execute_count += 1
      return self._target.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
      return getattr(self._target, name)

  spy = ConnSpy(db_conn)
  speedups = compute_speedup_matrix(cast("duckdb.DuckDBPyConnection", spy), "CPU", ["FullGPUModular"])
  assert spy.execute_count == 1
  assert len(speedups) == 2
  assert speedups[0][:2] == ("berlin52", "FullGPUModular")
  assert speedups[0][2] == pytest.approx(4.0)
  assert speedups[0][3] == pytest.approx(2.0)
  assert speedups[1][:2] == ("st70", "FullGPUModular")
  assert speedups[1][2] == pytest.approx(4.0)
  assert speedups[1][3] == pytest.approx(4.0)


def test_compute_speedup_matrix_includes_full_gpu_modular_by_default(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: Default comparison list includes FullGPUModular alongside FullGPUMonolithic.

  WHY: BUG-39 — FullGPUModular was previously omitted from the default speedup matrix.
  EXPECTED: All GPU variants including FullGPUModular appear in default_speedups.
  """
  insert_payload(db_conn, production_payload(mean_time=2.0, algorithm="CPU", timestamp="2025-11-23T10:00:00"))
  insert_payload(
    db_conn,
    production_payload(mean_time=1.0, algorithm="HybridNaive", backend="GPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(mean_time=0.8, algorithm="HybridOptimized", backend="GPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(mean_time=0.5, algorithm="FullGPUMonolithic", backend="GPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(mean_time=0.4, algorithm="FullGPUModular", backend="GPU", timestamp="2025-11-23T10:00:00"),
  )

  speedups = compute_speedup_matrix(db_conn)
  algs = [s[1] for s in speedups]

  assert "FullGPUModular" in algs
  assert "FullGPUMonolithic" in algs
  assert "HybridOptimized" in algs
  assert "HybridNaive" in algs

  modular_entry = next(s for s in speedups if s[1] == "FullGPUModular")
  assert modular_entry[2] == pytest.approx(2.0 / 0.4)


def test_compute_speedup_matrix_handles_missing_and_zero_times_gracefully(
  db_conn: duckdb.DuckDBPyConnection,
) -> None:
  """WHAT: Data with missing candidate algorithms, 0.0 runtime, or NaN.

  WHY: BUG-39 — Missing candidates or division by zero must be gracefully handled.
  EXPECTED: include_missing=False skips invalid/missing; include_missing=True returns fill_missing.
  """
  insert_payload(
    db_conn,
    production_payload(mean_time=2.0, problem_name="eil51", algorithm="CPU", timestamp="2025-11-23T10:00:00"),
  )
  insert_payload(
    db_conn,
    production_payload(
      mean_time=0.0,
      problem_name="eil51",
      algorithm="HybridNaive",
      backend="GPU",
      timestamp="2025-11-23T10:00:00",
    ),
  )

  # include_missing=False (default): HybridNaive skipped due to mean_time=0.0
  speedups_default = compute_speedup_matrix(db_conn, "CPU", ["HybridNaive", "FullGPUModular"])
  assert len(speedups_default) == 0

  # include_missing=True: both filled with None (or provided fill value)
  speedups_filled = compute_speedup_matrix(
    db_conn,
    "CPU",
    ["HybridNaive", "FullGPUModular"],
    include_missing=True,
    fill_missing=None,
  )
  assert len(speedups_filled) == 2
  assert speedups_filled[0] == ("eil51", "HybridNaive", None, 2.0)
  assert speedups_filled[1] == ("eil51", "FullGPUModular", None, 2.0)


def test_import_checkpoints_natural_key_replacement_prevents_duplicate_runs(tmp_path: Path) -> None:
  """WHAT: Importing an updated checkpoint file replaces obsolete runs.

  WHY: BUG-49 / DP12-26 — Incremental resume produces checkpoints with updated repetitions
  (e.g. N=5 -> N=10) and newer timestamps. The DB import previously checked (problem, algo, timestamp),
  leaving duplicate rows in benchmark_runs that corrupted compute_speedup_matrix.
  EXPECTED: Exactly 1 row in benchmark_runs for (berlin52, CPU) with reps=10 and accurate speedup.
  """
  from benchmarking.database.import_checkpoints import check_duplicate, import_checkpoints
  from benchmarking.database.schema import TABLE_RUNS

  checkpoints_dir = tmp_path / "checkpoints"
  checkpoints_dir.mkdir()
  results_dir = tmp_path / "results"
  results_dir.mkdir()

  # Step 1: Write initial checkpoint for CPU with repetitions=5 (test run), mean_time=4.0
  reps5 = 5
  cp_initial = run_payload(
    problem_name="berlin52",
    algorithm="CPU",
    backend="CPU",
    repetitions=reps5,
    successful_runs=reps5,
    timestamp="2025-11-23T10:00:00",
    mean_time=4.0,
    raw_times=[4.0] * reps5,
    raw_costs=[7600.0] * reps5,
    raw_gaps=[0.5] * reps5,
  )
  cp_path = checkpoints_dir / "berlin52_CPU.json"
  with open(cp_path, "w", encoding="utf-8") as f:
    json.dump(cp_initial, f)

  # Write FullGPU production run: mean_time=1.0, repetitions=15
  reps15 = 15
  gpu_cp = run_payload(
    problem_name="berlin52",
    algorithm="FullGPUModular",
    backend="GPU",
    repetitions=reps15,
    successful_runs=reps15,
    timestamp="2025-11-23T10:00:00",
    mean_time=1.0,
    raw_times=[1.0] * reps15,
    raw_costs=[7600.0] * reps15,
    raw_gaps=[0.5] * reps15,
  )
  with open(checkpoints_dir / "berlin52_FullGPUModular.json", "w", encoding="utf-8") as f:
    json.dump(gpu_cp, f)

  # First import
  import_checkpoints(checkpoints_dir, results_dir)

  db_path = results_dir / "results.duckdb"
  conn = create_database(str(db_path))
  rows = conn.execute(
    f"SELECT repetitions, mean_time FROM {TABLE_RUNS} WHERE problem_name = ? AND algorithm = ?",  # noqa: S608
    ["berlin52", "CPU"],
  ).fetchall()
  assert len(rows) == 1
  assert rows[0][0] == 5
  assert rows[0][1] == pytest.approx(4.0)

  # Step 2: Now simulate incremental resume finishing with repetitions=10 (production run, is_test=False), mean_time=2.0
  reps10 = 10
  cp_updated = run_payload(
    problem_name="berlin52",
    algorithm="CPU",
    backend="CPU",
    repetitions=reps10,
    successful_runs=reps10,
    timestamp="2025-11-23T12:00:00",
    mean_time=2.0,
    raw_times=[2.0] * reps10,
    raw_costs=[7600.0] * reps10,
    raw_gaps=[0.5] * reps10,
  )
  with open(cp_path, "w", encoding="utf-8") as f:
    json.dump(cp_updated, f)

  # Verify check_duplicate returns False for the updated checkpoint against current DB
  assert check_duplicate(conn, cp_updated) is False

  # Verify check_duplicate returns True if trying to re-import the older checkpoint (reps=5)
  assert check_duplicate(conn, cp_initial) is True

  conn.close()

  # Second import
  import_checkpoints(checkpoints_dir, results_dir)

  conn = create_database(str(db_path))
  rows_after = conn.execute(
    f"SELECT repetitions, mean_time, timestamp FROM {TABLE_RUNS} WHERE problem_name = ? AND algorithm = ?",  # noqa: S608
    ["berlin52", "CPU"],
  ).fetchall()
  # Must have EXACTLY 1 row, not 2!
  assert len(rows_after) == 1
  assert rows_after[0][0] == 10
  assert rows_after[0][1] == pytest.approx(2.0)

  # compute_speedup_matrix with CPU as baseline: CPU mean_time=2.0, FullGPUModular mean_time=1.0 -> speedup = 2.0
  speedups = compute_speedup_matrix(conn, baseline="CPU", comparisons=["FullGPUModular"])
  assert len(speedups) == 1
  assert speedups[0][0] == "berlin52"
  assert speedups[0][1] == "FullGPUModular"
  assert speedups[0][2] == pytest.approx(2.0)
  assert speedups[0][3] == pytest.approx(2.0)
  conn.close()


def test_import_checkpoints_failed_insert_rolls_back_and_preserves_row(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """WHAT: A failed insert during replacement rolls back without losing the prior row.

  WHY: Review Gate 22c181dd / Task 12.33 — In autocommit mode, deleting before insert
  leaves the DB without any record if insert fails. Wrapping DELETE + INSERT in an explicit
  transaction ensures rollback preserves the original row.
  """
  import sys

  from benchmarking.database.import_checkpoints import import_checkpoints
  from benchmarking.database.schema import TABLE_RUNS

  import_mod = sys.modules["benchmarking.database.import_checkpoints"]

  checkpoints_dir = tmp_path / "checkpoints"
  checkpoints_dir.mkdir()
  results_dir = tmp_path / "results"
  results_dir.mkdir()

  # Write initial checkpoint and import
  cp_initial = run_payload(
    problem_name="berlin52",
    algorithm="CPU",
    backend="CPU",
    repetitions=5,
    successful_runs=5,
    timestamp="2025-11-23T10:00:00",
    mean_time=4.0,
    raw_times=[4.0] * 5,
    raw_costs=[7600.0] * 5,
    raw_gaps=[0.5] * 5,
  )
  cp_path = checkpoints_dir / "berlin52_CPU.json"
  with open(cp_path, "w", encoding="utf-8") as f:
    json.dump(cp_initial, f)

  import_checkpoints(checkpoints_dir, results_dir)

  db_path = results_dir / "results.duckdb"
  conn = create_database(str(db_path))
  row_cnt = conn.execute(f"SELECT COUNT(*) FROM {TABLE_RUNS}").fetchone()  # noqa: S608
  assert row_cnt is not None
  count_before = int(row_cnt[0])
  assert count_before == 1
  conn.close()

  # Update checkpoint to higher reps (N=10)
  cp_updated = run_payload(
    problem_name="berlin52",
    algorithm="CPU",
    backend="CPU",
    repetitions=10,
    successful_runs=10,
    timestamp="2025-11-23T12:00:00",
    mean_time=2.0,
    raw_times=[2.0] * 10,
    raw_costs=[7600.0] * 10,
    raw_gaps=[0.5] * 10,
  )
  with open(cp_path, "w", encoding="utf-8") as f:
    json.dump(cp_updated, f)

  # Monkeypatch insert_benchmark_run to simulate mid-transaction failure
  def _failing_insert(*args: Any, **kwargs: Any) -> int:
    raise RuntimeError("Simulated mid-transaction failure")

  monkeypatch.setattr(import_mod, "insert_benchmark_run", _failing_insert)

  # Run import: the exception is caught by the per-file try/except block in import_checkpoints
  import_checkpoints(checkpoints_dir, results_dir)

  # Verify row was preserved via rollback!
  conn = create_database(str(db_path))
  rows = conn.execute(
    f"SELECT repetitions, mean_time FROM {TABLE_RUNS} WHERE problem_name = ? AND algorithm = ?",  # noqa: S608
    ["berlin52", "CPU"],
  ).fetchall()
  assert len(rows) == 1
  assert rows[0][0] == 5
  assert rows[0][1] == pytest.approx(4.0)
  conn.close()


def test_check_duplicate_equal_repetitions_timestamp_rules(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: Test check_duplicate resolution when repetitions are equal.

  WHY: Review Gate 22c181dd / Task 12.33 — Equal repetitions must replace when the incoming
  timestamp is strictly newer, and skip when equal or older. Timezone-aware datetimes must
  compare correctly against UTC naive timestamps.
  """
  from benchmarking.database.import_checkpoints import check_duplicate

  # Insert baseline run with timestamp 2025-11-23T12:00:00
  insert_payload(
    db_conn,
    production_payload(mean_time=2.0, algorithm="CPU", timestamp="2025-11-23T12:00:00"),
  )

  # 1. Strictly newer timestamp (14:00:00 > 12:00:00): replace -> False (not duplicate)
  newer_payload = production_payload(mean_time=1.8, algorithm="CPU", timestamp="2025-11-23T14:00:00")
  assert check_duplicate(db_conn, newer_payload) is False

  # 2. Equal timestamp (12:00:00 == 12:00:00): skip -> True (duplicate)
  equal_payload = production_payload(mean_time=2.0, algorithm="CPU", timestamp="2025-11-23T12:00:00")
  assert check_duplicate(db_conn, equal_payload) is True

  # 3. Strictly older timestamp (10:00:00 < 12:00:00): skip -> True (duplicate)
  older_payload = production_payload(mean_time=2.2, algorithm="CPU", timestamp="2025-11-23T10:00:00")
  assert check_duplicate(db_conn, older_payload) is True

  # 4. Timezone-aware UTC timestamp (13:00:00Z > 12:00:00 naive): replace -> False
  tz_newer = production_payload(mean_time=1.9, algorithm="CPU", timestamp="2025-11-23T13:00:00+00:00")
  assert check_duplicate(db_conn, tz_newer) is False


def test_compare_databases_handles_empty_dbs_reordered_rows_and_tail_differences(tmp_path: Path) -> None:
  """WHAT: `compare_databases` handles empty DBs, ignores insertion order, and checks every row.

  WHY: BUG-102 — Previous comparator crashed with `IndexError` on empty DBs, leaked DuckDB
  connections on early returns, and only sampled 2 rows by `run_id`.
  """
  from benchmarking.database.validate_consistency import compare_databases

  db1 = str(tmp_path / "a.duckdb")
  db2 = str(tmp_path / "b.duckdb")
  create_database(db1).close()
  create_database(db2).close()

  # 1. Empty databases compare cleanly without IndexError
  identical_empty, diffs_empty = compare_databases(db1, db2)
  assert identical_empty is True
  assert diffs_empty == []

  # 2. Four identical rows inserted in opposite order compare equal
  rows = [
    production_payload(
      problem_name="p1", algorithm="CPU", mean_cost=100.0, mean_time=1.0, timestamp="2025-11-23T12:00:00"
    ),
    production_payload(
      problem_name="p2", algorithm="CPU", mean_cost=200.0, mean_time=2.0, timestamp="2025-11-23T12:00:00"
    ),
    production_payload(
      problem_name="p3", algorithm="CPU", mean_cost=300.0, mean_time=3.0, timestamp="2025-11-23T12:00:00"
    ),
    production_payload(
      problem_name="p4", algorithm="CPU", mean_cost=400.0, mean_time=4.0, timestamp="2025-11-23T12:00:00"
    ),
  ]
  c1 = create_database(db1)
  c2 = create_database(db2)
  for payload in rows:
    insert_payload(c1, payload)
  for payload in reversed(rows):
    insert_payload(c2, payload)
  c1.close()
  c2.close()

  identical_reordered, diffs_reordered = compare_databases(db1, db2)
  assert identical_reordered is True
  assert diffs_reordered == []

  # 3. Difference strictly in the last row (missed by old [0, len//2] sampling) is detected
  db3 = str(tmp_path / "c.duckdb")
  c3 = create_database(db3)
  for payload in rows[:-1]:
    insert_payload(c3, payload)
  insert_payload(
    c3,
    production_payload(
      problem_name="p4", algorithm="CPU", mean_cost=499.0, mean_time=4.0, timestamp="2025-11-23T12:00:00"
    ),
  )
  c3.close()

  identical_diff, diffs = compare_databases(db1, db3)
  assert identical_diff is False
  assert any("p4 / CPU" in d and "mean_cost" in d for d in diffs)


def test_vrp_zero_feasible_and_feasible_runs_import_cleanly(db_conn: duckdb.DuckDBPyConnection) -> None:
  """WHAT: VRP checkpoints with 0% and partial feasibility persist cleanly into DuckDB (DP17-4).

  WHY: `feasible_runs == 0` produces NULL aggregate cost/gap/vehicle scalars while keeping
  `raw_is_feasible`, `raw_unpenalized_costs`, `raw_violations`, and `raw_num_vehicles` intact.
  EXPECTED: Both rows insert without error, NULLs round-trip as None, and native LIST/scalar
  columns match the source payload.
  """
  zero_feas_id = insert_run(
    db_conn,
    problem_name="A-n32-k5",
    problem_type="CVRP",
    algorithm="CVRP_Penalized",
    mean_cost=None,
    std_cost=None,
    best_cost=None,
    worst_cost=None,
    mean_gap=None,
    std_gap=None,
    best_gap=None,
    raw_gaps=[None, None, None],
    feasible_runs=0,
    feasibility_rate=0.0,
    mean_vehicles=None,
    best_vehicles=None,
    raw_is_feasible=[False, False, False],
    raw_unpenalized_costs=[700.0, 710.0, 720.0],
    raw_violations=[15.0, 10.0, 20.0],
    raw_num_vehicles=[4, 4, 4],
  )
  zero_row = fetch_run(db_conn, zero_feas_id)
  assert zero_row["problem_type"] == "CVRP"
  assert zero_row["mean_cost"] is None
  assert zero_row["std_cost"] is None
  assert zero_row["best_cost"] is None
  assert zero_row["worst_cost"] is None
  assert zero_row["feasible_runs"] == 0
  assert zero_row["feasibility_rate"] == pytest.approx(0.0)
  assert zero_row["mean_vehicles"] is None
  assert zero_row["best_vehicles"] is None
  assert zero_row["raw_is_feasible"] == [False, False, False]
  assert zero_row["raw_violations"] == [15.0, 10.0, 20.0]

  feas_id = insert_run(
    db_conn,
    problem_name="A-n32-k5",
    problem_type="CVRP",
    algorithm="CVRP_Split",
    mean_cost=790.0,
    std_cost=10.0,
    best_cost=784.0,
    worst_cost=800.0,
    feasible_runs=2,
    feasibility_rate=66.6666666667,
    mean_vehicles=5.0,
    best_vehicles=5,
    raw_is_feasible=[True, False, True],
    raw_unpenalized_costs=[784.0, 650.0, 796.0],
    raw_violations=[0.0, 12.0, 0.0],
    raw_num_vehicles=[5, 4, 5],
  )
  feas_row = fetch_run(db_conn, feas_id)
  assert feas_row["problem_type"] == "CVRP"
  assert feas_row["feasible_runs"] == 2
  assert feas_row["mean_vehicles"] == pytest.approx(5.0)
  assert feas_row["best_vehicles"] == 5
  assert feas_row["raw_num_vehicles"] == [5, 4, 5]


def test_import_checkpoints_disambiguates_twin_problem_name_across_tsp_and_cvrp(tmp_path: Path) -> None:
  """WHAT: Importing `(eil51, TSP, GA)` and `(eil51, CVRP, GA)` preserves both rows (DP17-4).

  WHY: TSPLIB and CVRPLIB share instance names such as `eil51`; deduplicating only on
  `(problem_name, algorithm)` would silently overwrite TSP data with CVRP data or vice versa.
  EXPECTED: Both rows coexist in `results.duckdb` with distinct `problem_type` values, and
  `validate_against_checkpoints` validates both cleanly.
  """
  from benchmarking.database.import_checkpoints import import_checkpoints
  from benchmarking.database.validate_consistency import validate_against_checkpoints

  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  out_dir = tmp_path / "db_out"

  tsp_payload = run_payload(problem_name="eil51", algorithm="GA", mean_cost=430.0)
  # Legacy TSP checkpoint omits problem_type -> defaults to 'TSP'
  assert "problem_type" not in tsp_payload
  (ckpt_dir / "eil51_GA.json").write_text(json.dumps(tsp_payload), encoding="utf-8")

  cvrp_payload = run_payload(
    problem_name="eil51",
    problem_type="CVRP",
    algorithm="GA",
    mean_cost=525.0,
    feasible_runs=3,
    feasibility_rate=100.0,
    mean_vehicles=5.0,
    best_vehicles=5,
    raw_is_feasible=[True, True, True],
    raw_unpenalized_costs=[520.0, 525.0, 530.0],
    raw_violations=[0.0, 0.0, 0.0],
    raw_num_vehicles=[5, 5, 5],
  )
  (ckpt_dir / "eil51_CVRP_GA.json").write_text(json.dumps(cvrp_payload), encoding="utf-8")

  import_checkpoints(ckpt_dir, out_dir)

  db_path = out_dir / "results.duckdb"
  conn = create_database(str(db_path))
  try:
    rows = conn.execute(
      "SELECT problem_name, problem_type, algorithm, mean_cost FROM benchmark_runs ORDER BY problem_type"
    ).fetchall()
  finally:
    conn.close()

  assert rows == [
    ("eil51", "CVRP", "GA", 525.0),
    ("eil51", "TSP", "GA", 430.0),
  ]

  ok, mismatches = validate_against_checkpoints(str(db_path), str(ckpt_dir))
  assert ok is True
  assert mismatches.empty
