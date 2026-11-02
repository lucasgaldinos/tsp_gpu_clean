"""CPU-only unit tests for `PhaseTimer` bookkeeping in `utils.phase_timer`.

Only counts, sums and summaries are asserted; no wall-clock bound is ever checked, so the
tests stay deterministic on any machine.
"""

from __future__ import annotations

import pytest

from utils.phase_timer import PhaseTimer


def _raise_boom() -> None:
  """Auxiliary callable that always fails, proving `measure` records nothing on error."""
  raise RuntimeError("boom")


def test_record_accumulates_counts_and_totals() -> None:
  """WHAT: two `record` calls for the same phase.

  WHY: counts and totals are the only state `summary` reports to callers.
  EXPECTED: `calls == 2`, `total_ms` is the sum, `mean_ms` is the average.
  """
  timer = PhaseTimer(use_cuda_events=False)

  timer.record("selection", 1.5)
  timer.record("selection", 2.5)

  entry = timer.summary()["selection"]
  assert entry["calls"] == 2
  assert entry["total_ms"] == pytest.approx(4.0)
  assert entry["mean_ms"] == pytest.approx(2.0)


def test_summary_preserves_first_record_order() -> None:
  """WHAT: phases recorded as mutation, crossover, mutation.

  WHY: reports read `summary()` in insertion order to mirror execution order.
  EXPECTED: keys ordered by first occurrence, not by last or alphabetically.
  """
  timer = PhaseTimer(use_cuda_events=False)

  timer.record("mutation", 1.0)
  timer.record("crossover", 2.0)
  timer.record("mutation", 1.0)

  assert list(timer.summary()) == ["mutation", "crossover"]


def test_record_on_disabled_timer_is_noop() -> None:
  """WHAT: `record` on a timer constructed with `enabled=False`.

  WHY: disabling instrumentation must leave zero bookkeeping overhead and state.
  EXPECTED: `summary()` stays empty.
  """
  timer = PhaseTimer(enabled=False)

  timer.record("phase", 12.5)

  assert timer.summary() == {}


def test_measure_on_disabled_timer_still_runs_function() -> None:
  """WHAT: `measure` on a timer constructed with `enabled=False`.

  WHY: disabling timing must not change program behavior, only skip recording.
  EXPECTED: the wrapped call still executes and returns its value; nothing is recorded.
  """
  timer = PhaseTimer(enabled=False)

  result = timer.measure("phase", lambda: "done")

  assert result == "done"
  assert timer.summary() == {}


def test_measure_returns_result_and_records_one_call() -> None:
  """WHAT: `measure` wrapping a callable with positional and keyword arguments.

  WHY: `measure` must forward arguments and return the wrapped result unchanged.
  EXPECTED: the result is forwarded and exactly one non-negative duration is recorded.
  """
  timer = PhaseTimer(use_cuda_events=False)

  def _multiply(a: int, b: int = 0) -> int:
    return a * b

  result = timer.measure("fitness", _multiply, 3, b=4)

  entry = timer.summary()["fitness"]
  assert result == 12
  assert entry["calls"] == 1
  assert entry["total_ms"] >= 0.0
  assert entry["mean_ms"] == pytest.approx(entry["total_ms"])


def test_measure_propagates_exception_without_recording() -> None:
  """WHAT: `measure` wrapping a callable that raises.

  WHY: a failed phase must not appear as a completed call in the report.
  EXPECTED: the exception propagates and no entry is recorded.
  """
  timer = PhaseTimer(use_cuda_events=False)

  with pytest.raises(RuntimeError, match="boom"):
    timer.measure("phase", _raise_boom)

  assert timer.summary() == {}


@pytest.mark.parametrize(
  "durations",
  [
    pytest.param((), id="no-records"),
    pytest.param((0.0,), id="zero-duration"),
    pytest.param((0.5,), id="single-record"),
    pytest.param((0.5, 0.5, 1.0), id="three-records"),
    pytest.param((12.75, 0.25), id="fractional"),
  ],
)
def test_summary_invariants_hold_for_each_record_set(durations: tuple[float, ...]) -> None:
  """WHAT: summary invariants across empty, zero, single, and multi-record phases.

  WHY: totals, counts and means must stay non-negative and mutually consistent for any input.
  EXPECTED: `calls == len(durations)`, `total_ms == sum`, `mean_ms == total / calls`.
  """
  timer = PhaseTimer(use_cuda_events=False)

  for duration in durations:
    timer.record("phase", duration)

  summary = timer.summary()
  if not durations:
    assert summary == {}
    return

  entry = summary["phase"]
  assert entry["calls"] == len(durations)
  assert isinstance(entry["calls"], int)
  assert entry["total_ms"] >= 0.0
  assert entry["total_ms"] == pytest.approx(sum(durations))
  assert entry["mean_ms"] >= 0.0
  assert entry["mean_ms"] == pytest.approx(entry["total_ms"] / entry["calls"])


def test_reset_clears_timings_and_keeps_configuration() -> None:
  """WHAT: `reset` after recording, followed by a fresh record.

  WHY: reset must wipe accumulated bookkeeping without touching timer configuration.
  EXPECTED: `summary()` is empty after reset, flags are unchanged, and recording restarts.
  """
  timer = PhaseTimer(use_cuda_events=False)
  timer.record("selection", 1.0)
  timer.record("crossover", 2.0)

  timer.reset()

  assert timer.summary() == {}
  assert timer.enabled is True
  assert timer.use_cuda_events is False

  timer.record("selection", 3.0)
  entry = timer.summary()["selection"]
  assert entry["calls"] == 1
  assert entry["total_ms"] == pytest.approx(3.0)


def test_toggling_enabled_controls_accumulation() -> None:
  """WHAT: `enabled` flipped off and back on between records.

  WHY: runtime toggling is the supported way to instrument only part of a run.
  EXPECTED: records while disabled are skipped; recording resumes after re-enabling.
  """
  timer = PhaseTimer(use_cuda_events=False)
  timer.record("phase", 1.0)

  timer.enabled = False
  timer.record("phase", 5.0)

  assert timer.summary()["phase"]["calls"] == 1
  assert timer.summary()["phase"]["total_ms"] == pytest.approx(1.0)

  timer.enabled = True
  timer.record("phase", 0.5)

  entry = timer.summary()["phase"]
  assert entry["calls"] == 2
  assert entry["total_ms"] == pytest.approx(1.5)


def test_measure_and_record_share_one_accumulator() -> None:
  """WHAT: one manual `record` and one `measure` for the same phase.

  WHY: manual and measured timings must land in the same per-phase counters.
  EXPECTED: two calls, total at least the manual amount, mean equals total over two.
  """
  timer = PhaseTimer(use_cuda_events=False)
  timer.record("phase", 2.0)

  timer.measure("phase", lambda: None)

  entry = timer.summary()["phase"]
  assert entry["calls"] == 2
  assert entry["total_ms"] >= 2.0
  assert entry["mean_ms"] == pytest.approx(entry["total_ms"] / 2)


def test_summary_returns_defensive_copy() -> None:
  """WHAT: mutating the dict returned by `summary()`.

  WHY: reports are handed to callers; they must not corrupt the timer's own state.
  EXPECTED: internal counts are unaffected by mutations of a previously returned summary.
  """
  timer = PhaseTimer(use_cuda_events=False)
  timer.record("phase", 1.0)

  first = timer.summary()
  first["phase"]["calls"] = 999

  assert timer.summary()["phase"]["calls"] == 1


def test_summary_entry_exposes_documented_fields() -> None:
  """WHAT: the shape of one recorded phase entry.

  WHY: downstream table writers index these exact keys and rely on their types.
  EXPECTED: exactly `total_ms`, `calls`, `mean_ms`, with float/int/float types.
  """
  timer = PhaseTimer(use_cuda_events=False)
  timer.record("phase", 2.0)

  entry = timer.summary()["phase"]

  assert set(entry) == {"total_ms", "calls", "mean_ms"}
  assert isinstance(entry["total_ms"], float)
  assert isinstance(entry["mean_ms"], float)
  assert isinstance(entry["calls"], int)
