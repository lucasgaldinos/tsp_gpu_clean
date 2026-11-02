"""Contract tests for `data_models.exceptions`.

Pins the hierarchy guarantees (`ProblemLoaderError` as the loader catch-all,
`VRAMInsufficientError` outside that family), the constructor attributes of every
concrete loader error, and the exact message formatting, including the VRAM
byte-to-gigabyte rendering. No GPU, no randomness, no dataset access.
"""

from __future__ import annotations

import pytest

from data_models.exceptions import (
  DatabaseConnectionError,
  InstanceNotFoundError,
  InvalidEdgeTypeError,
  InvalidProblemDataError,
  MissingCoordinatesError,
  ProblemLoaderError,
  VRAMInsufficientError,
)

DB_PATH = "fixtures/db/problem.duckdb"
ORIGINAL_ERROR = RuntimeError("connection refused")

#: One constructed error per concrete loader exception, paired with its exact message.
LOADER_ERROR_MESSAGE_CASES = [
  pytest.param(
    InstanceNotFoundError("burma14"),
    "Instance 'burma14' not found in database. Check instance name for typos or verify it's been imported.",
    id="instance-not-found",
  ),
  pytest.param(
    InvalidEdgeTypeError("CEIL_2D"),
    "Unsupported edge_type: 'CEIL_2D'. Valid types: EUC_2D, GEO, ATT, EXPLICIT",
    id="invalid-edge-type",
  ),
  pytest.param(
    MissingCoordinatesError("inst", "EUC_2D"),
    "Instance 'inst' has edge_type 'EUC_2D' but coordinates are NULL in database. "
    "This indicates a data integrity issue - problem should either have coordinates or use EXPLICIT edge_type.",
    id="missing-coordinates",
  ),
  pytest.param(
    DatabaseConnectionError(DB_PATH, ORIGINAL_ERROR),
    f"Failed to connect to database at '{DB_PATH}': {ORIGINAL_ERROR}",
    id="database-connection",
  ),
  pytest.param(
    InvalidProblemDataError("inst", "dimension=52 but only 50 nodes found"),
    "Invalid data for instance 'inst': dimension=52 but only 50 nodes found",
    id="invalid-problem-data",
  ),
]

#: One constructed error per concrete loader exception, paired with its full attribute surface.
LOADER_ERROR_ATTRIBUTE_CASES = [
  pytest.param(InstanceNotFoundError("burma14"), {"instance_name": "burma14"}, id="instance-not-found"),
  pytest.param(InvalidEdgeTypeError("CEIL_2D"), {"edge_type": "CEIL_2D"}, id="invalid-edge-type"),
  pytest.param(
    MissingCoordinatesError("inst", "EUC_2D"),
    {"instance_name": "inst", "edge_type": "EUC_2D"},
    id="missing-coordinates",
  ),
  pytest.param(
    DatabaseConnectionError(DB_PATH, ORIGINAL_ERROR),
    {"db_path": DB_PATH, "original_error": ORIGINAL_ERROR},
    id="database-connection",
  ),
  pytest.param(
    InvalidProblemDataError("inst", "bad shape"),
    {"instance_name": "inst", "reason": "bad shape"},
    id="invalid-problem-data",
  ),
]

LOADER_ERROR_TYPES = [
  pytest.param(InstanceNotFoundError, id="instance-not-found"),
  pytest.param(InvalidEdgeTypeError, id="invalid-edge-type"),
  pytest.param(MissingCoordinatesError, id="missing-coordinates"),
  pytest.param(DatabaseConnectionError, id="database-connection"),
  pytest.param(InvalidProblemDataError, id="invalid-problem-data"),
]


# ═══════════════════════════════════════════════════════════════════
# exceptions.py — hierarchy
# ═══════════════════════════════════════════════════════════════════


def test_problem_loader_error_derives_directly_from_exception() -> None:
  """
  WHAT: the base of the loader error family.
  WHY: a catch-all `except ProblemLoaderError` must not swallow unrelated errors from a wider base class.
  EXPECTED: `Exception` as the only base.
  """
  assert ProblemLoaderError.__bases__ == (Exception,)


@pytest.mark.parametrize("error_type", LOADER_ERROR_TYPES)
def test_loader_errors_inherit_from_problem_loader_error(error_type: type[ProblemLoaderError]) -> None:
  """
  WHAT: inheritance of each concrete loader error.
  WHY: the documented catch-all guarantee requires every loader failure to be reachable as the base type.
  EXPECTED: a strict subclass of `ProblemLoaderError`.
  """
  assert issubclass(error_type, ProblemLoaderError)
  assert error_type is not ProblemLoaderError


def test_vram_error_is_a_runtime_error_outside_the_loader_family() -> None:
  """
  WHAT: the base of `VRAMInsufficientError`.
  WHY: a device-capacity failure is a runtime condition, so `except ProblemLoaderError` must not catch it.
  EXPECTED: `RuntimeError` as the only base, not a `ProblemLoaderError`.
  """
  assert VRAMInsufficientError.__bases__ == (RuntimeError,)
  assert not issubclass(VRAMInsufficientError, ProblemLoaderError)


# ═══════════════════════════════════════════════════════════════════
# exceptions.py — constructor attributes and messages
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("error", "expected_attributes"), LOADER_ERROR_ATTRIBUTE_CASES)
def test_loader_error_stores_every_constructor_argument(
  error: ProblemLoaderError,
  expected_attributes: dict[str, object],
) -> None:
  """
  WHAT: instance attributes created by each loader error constructor.
  WHY: callers branch on these attributes (`instance_name`, `edge_type`), so a renamed attribute breaks handlers.
  EXPECTED: the instance dictionary equals the constructor arguments exactly.
  """
  assert vars(error) == expected_attributes


@pytest.mark.parametrize(("error", "expected_message"), LOADER_ERROR_MESSAGE_CASES)
def test_loader_error_message_is_formatted_and_catchable_as_the_base(
  error: ProblemLoaderError,
  expected_message: str,
) -> None:
  """
  WHAT: the message produced by each loader error, and catchability through the base class.
  WHY: the messages are user-facing guidance and are asserted verbatim in logs; the catch-all path must work.
  EXPECTED: the exact message, and the same instance caught by `except ProblemLoaderError`.
  """
  with pytest.raises(ProblemLoaderError) as exc_info:
    raise error

  assert exc_info.value is error
  assert str(exc_info.value) == expected_message


def test_database_connection_error_formats_an_empty_underlying_message() -> None:
  """
  WHAT: `DatabaseConnectionError` wrapping an exception whose message is empty.
  WHY: this is the formatting boundary where the template's colon has nothing after it.
  EXPECTED: the prefix followed by the empty string.
  """
  error = DatabaseConnectionError("fixtures/db/empty.duckdb", Exception())

  assert str(error) == "Failed to connect to database at 'fixtures/db/empty.duckdb': "
  assert error.original_error.args == ()


# ═══════════════════════════════════════════════════════════════════
# exceptions.py — VRAMInsufficientError formatting
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
  ("required_bytes", "available_bytes"),
  [
    pytest.param(None, None, id="no-details"),
    pytest.param(518_000_000, None, id="required-only"),
    pytest.param(None, 4_000_000_000, id="available-only"),
  ],
)
def test_vram_error_str_without_both_bounds_is_the_bare_message(
  required_bytes: int | None,
  available_bytes: int | None,
) -> None:
  """
  WHAT: `str()` when fewer than two capacity bounds are known.
  WHY: a partial bound would print a misleading comparison, so the formatter must fall back to the message.
  EXPECTED: the bare message, with both attributes stored as given.
  """
  message = "Insufficient VRAM for n=10000 problem"
  error = VRAMInsufficientError(message, required_bytes, available_bytes)

  assert str(error) == message
  assert error.message == message
  assert error.required_bytes == required_bytes
  assert error.available_bytes == available_bytes


def test_vram_error_str_renders_gigabytes_when_both_bounds_are_known() -> None:
  """
  WHAT: `str()` with both capacity bounds supplied.
  WHY: the fallback logic compares gigabytes, so the rendered units and precision are the contract.
  EXPECTED: message followed by both bounds at two decimal places.
  """
  error = VRAMInsufficientError(
    "Insufficient VRAM for n=10000 problem",
    required_bytes=518_000_000,
    available_bytes=4_000_000_000,
  )

  assert str(error) == "Insufficient VRAM for n=10000 problem (required: 0.52GB, available: 4.00GB)"


def test_vram_error_str_renders_zero_bounds() -> None:
  """
  WHAT: `str()` with both bounds equal to zero.
  WHY: zero is falsy but not None; the formatter must test for None and still render the comparison.
  EXPECTED: both bounds shown as 0.00GB.
  """
  error = VRAMInsufficientError("zero capacity", required_bytes=0, available_bytes=0)

  assert str(error) == "zero capacity (required: 0.00GB, available: 0.00GB)"


@pytest.mark.parametrize(
  ("required_bytes", "available_bytes"),
  [
    pytest.param(None, None, id="no-details"),
    pytest.param(518_000_000, 4_000_000_000, id="with-details"),
    pytest.param(0, 0, id="zero-details"),
  ],
)
def test_vram_error_repr_names_every_attribute(required_bytes: int | None, available_bytes: int | None) -> None:
  """
  WHAT: `repr()` of a VRAM error.
  WHY: the repr is the debugging record of the failing batch, so it must carry both bounds even when None.
  EXPECTED: the constructor-style representation with raw byte values.
  """
  error = VRAMInsufficientError("msg", required_bytes, available_bytes)
  expected = f"VRAMInsufficientError('msg', required_bytes={required_bytes}, available_bytes={available_bytes})"

  assert repr(error) == expected
