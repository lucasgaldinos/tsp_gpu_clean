"""CPU-only unit tests for `make_rng` and `resolve_rng` seed contracts in `utils.rng`.

Pins seed resolution, generator passthrough and fallback precedence. Every test is
hermetic: explicit seeds and no CUDA device access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from utils.rng import make_rng, resolve_rng

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext


class _RngContextStub:
  """Duck-typed stand-in for `ProblemContext` exposing only the `rng` attribute `resolve_rng` reads."""

  def __init__(self, rng: object) -> None:
    self.rng = rng


def test_make_rng_same_seed_reproduces_sequence() -> None:
  """WHAT: two independent `make_rng` calls with the same explicit seed.

  WHY: every strategy in the suite depends on seed-for-seed reproducibility.
  EXPECTED: both generators emit the same draw sequence.
  """
  first = make_rng(20261030).random(16)
  second = make_rng(20261030).random(16)

  assert np.array_equal(first, second)


def test_make_rng_different_seeds_diverge() -> None:
  """WHAT: `make_rng` with two different explicit seeds.

  WHY: a resolver that ignores the seed would make every experiment share one stream.
  EXPECTED: the draw sequences differ.
  """
  first = make_rng(1).random(16)
  second = make_rng(2).random(16)

  assert not np.array_equal(first, second)


def test_make_rng_returns_existing_generator_as_is() -> None:
  """WHAT: `make_rng` handed an existing `np.random.Generator`.

  WHY: callers pass a live generator to keep an advanced stream, not to reseed it.
  EXPECTED: the same object is returned and its stream matches a reference generator
  seeded identically.
  """
  rng = np.random.default_rng(2024)
  reference = np.random.default_rng(2024)

  returned = make_rng(rng)

  assert returned is rng
  assert np.array_equal(returned.random(8), reference.random(8))


def test_make_rng_accepts_zero_argument_factory() -> None:
  """WHAT: `make_rng` handed a zero-argument factory returning a generator.

  WHY: deferred construction is a supported `SeedLike` form.
  EXPECTED: the factory result is returned unchanged.
  """
  rng = np.random.default_rng(5)

  assert make_rng(lambda: rng) is rng


@pytest.mark.parametrize(
  "seed",
  [
    pytest.param(1.5, id="float"),
    pytest.param("not-a-seed", id="string"),
    pytest.param(object(), id="object"),
  ],
)
def test_make_rng_rejects_invalid_seed_types(seed: object) -> None:
  """WHAT: `make_rng` with a seed that is neither int, generator, callable, nor None.

  WHY: silently coercing junk seeds would hide caller bugs.
  EXPECTED: `TypeError` from the underlying seed sequence.
  """
  with pytest.raises(TypeError):
    make_rng(cast("int", seed))


def test_make_rng_rejects_factory_returning_non_generator() -> None:
  """WHAT: `make_rng` with a callable whose result is not a generator.

  WHY: the helper promises a `np.random.Generator`; a wrong factory must fail loudly.
  EXPECTED: `TypeError` naming the required return type.
  """
  with pytest.raises(TypeError, match=r"must return np\.random\.Generator"):
    make_rng(lambda: 42)  # type: ignore[arg-type,return-value]


def test_make_rng_without_seed_returns_generator() -> None:
  """WHAT: `make_rng` with the default and explicit `None` seed.

  WHY: entropy-backed generators are the documented boundary case.
  EXPECTED: a `np.random.Generator` instance in both cases.
  """
  assert isinstance(make_rng(), np.random.Generator)
  assert isinstance(make_rng(None), np.random.Generator)


def test_resolve_rng_prefers_context_generator() -> None:
  """WHAT: `resolve_rng` with a context already carrying a generator.

  WHY: per-problem RNG isolation is the reason the helper exists.
  EXPECTED: the context generator wins, and the fallback is ignored.
  """
  rng = np.random.default_rng(99)
  context = cast("ProblemContext", _RngContextStub(rng))

  assert resolve_rng(context) is rng
  assert resolve_rng(context, fallback=123) is rng


@pytest.mark.parametrize(
  "context",
  [
    pytest.param(None, id="none-context"),
    pytest.param(_RngContextStub(42), id="non-generator-rng"),
    pytest.param(object(), id="no-rng-attribute"),
  ],
)
def test_resolve_rng_uses_fallback_generator(context: object) -> None:
  """WHAT: `resolve_rng` with no usable context generator.

  WHY: fallback resolution must keep working for missing or foreign contexts.
  EXPECTED: the exact fallback generator is returned.
  """
  rng = np.random.default_rng(8)

  assert resolve_rng(cast("ProblemContext", context), rng) is rng


def test_resolve_rng_int_fallback_is_deterministic() -> None:
  """WHAT: `resolve_rng` falling back to an integer seed.

  WHY: fallback seeding must produce the same stream as a direct `make_rng` call.
  EXPECTED: identical draws from both generators.
  """
  context = cast("ProblemContext", _RngContextStub(np.random.RandomState(3)))

  resolved = resolve_rng(context, 7)

  assert isinstance(resolved, np.random.Generator)
  assert np.array_equal(resolved.random(4), make_rng(7).random(4))


def test_resolve_rng_uses_seeded_problem_context_rng() -> None:
  """WHAT: `resolve_rng` with the RNG shape a real `ProblemContext` exposes.

  WHY: `ProblemContext.__init__` assigns a `np.random.Generator`, so the context RNG
  must be honoured for per-problem determinism.
  EXPECTED: `resolve_rng` returns `context.rng` instead of falling back.
  """
  context_rng = make_rng(42)
  context = cast("ProblemContext", _RngContextStub(context_rng))
  assert cast("object", resolve_rng(context, 7)) is context_rng
