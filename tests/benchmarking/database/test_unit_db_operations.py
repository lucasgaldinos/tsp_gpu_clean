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
  "mean_cost",
  "std_cost",
  "best_cost",
  "worst_cost",
  "mean_gap",
  "std_gap",
  "best_gap",
  "mean_generations",
  "std_generations",
]

#: Nullable numeric fields that are NaN-checked only when present.
NAN_CHECKED_OPTIONAL_FIELDS: list[str] = [
  "mean_initial_cost",
  "std_initial_cost",
  "mean_improvement",
  "mean_h2d_mb",
  "mean_d2h_mb",
  "total_transfer_mb",
  "mean_kernels",
]

#: Required raw arrays whose length must equal `repetitions`.
REQUIRED_RAW_ARRAYS: list[str] = ["raw_times", "raw_costs", "raw_gaps"]

#: Nullable raw arrays whose length must equal `repetitions` when present.
OPTIONAL_RAW_ARRAYS: list[str] = ["raw_initial_costs", "raw_generations", "raw_stop_reasons"]

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
  EXPECTED: `benchmark_runs` is described with 42 columns and
  `nextval('benchmark_runs_seq')` yields the first id, 1.
  """
  columns: dict[str, tuple[str, str, str]] = describe_columns(db_conn)
  first_id: tuple[Any, ...] | None = db_conn.execute("SELECT nextval('benchmark_runs_seq')").fetchone()

  assert len(columns) == 42, sorted(columns)
  assert "run_id" in columns
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
    pytest.param("raw_gaps", False, id="raw_gaps"),
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
  missing_algorithm = get_algorithm_summary(db_conn, "berlin52", "FullGPU")

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
  insert_run(db_conn, problem_name="eil76", problem_size=76, algorithm="FullGPU", backend="GPU")

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
    production_payload(mean_time=0.5, timestamp="2025-11-23T10:00:00", algorithm="FullGPU", backend="GPU"),
  )
  insert_run(db_conn, algorithm="FullGPU", backend="GPU", mean_time=0.1, timestamp="2025-11-23T12:00:00")

  speedups = compute_speedup_matrix(db_conn, "CPU", ["FullGPU", "HybridOptimized"])
  default_speedups = compute_speedup_matrix(db_conn)

  assert [entry[:2] for entry in speedups] == [("berlin52", "FullGPU")]
  assert speedups[0][2] == pytest.approx(4.0)
  assert speedups[0][3] == pytest.approx(2.0)
  assert [entry[:2] for entry in default_speedups] == [("berlin52", "FullGPU")]
  assert default_speedups[0][2] == pytest.approx(4.0)
  assert default_speedups[0][3] == pytest.approx(2.0)
