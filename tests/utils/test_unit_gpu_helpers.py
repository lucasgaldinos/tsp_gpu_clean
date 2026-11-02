"""CPU-side tests for kernel lookup in `utils.gpu_helpers`, plus a device-only cache test.

WHAT: Exercises `load_kernel`'s three-directory search (`<caller>/kernels`, then
`<caller>/../kernels`, then `<caller>/../../kernels`) against a fake caller
layout built under `tmp_path`, its `FileNotFoundError` report when the kernel is
absent everywhere, and - when a device is present - the `lru_cache` that the
module keeps around `_compile_kernel_cached`.
WHY: `load_kernel` is called by every GPU strategy with `__file__`; if the
search order or the root cause of a missing kernel is wrong, the failure appears
as an opaque NVRTC error at the first kernel launch instead of a path listing
pointing at the file the caller must add.
EXPECTED: The nearest existing candidate wins, a missing kernel names all three
searched paths in order, and compilation is cached per (path, function name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import utils.gpu_helpers as gpu_helpers
from protocols.backend import cupy_backend
from utils.gpu_helpers import load_kernel

cp = cupy_backend()

if TYPE_CHECKING:
  from pathlib import Path

  from _pytest.mark.structures import ParameterSet

KERNEL_FILENAME = "noop.cu"
KERNEL_FUNCTION = "noop_kernel"
KERNEL_SOURCE = 'extern "C" __global__ void noop_kernel() {}\n'

#: `(candidate indices that hold the kernel, index the loader must pick)`.
#: Candidate 0 is `<caller>/kernels`, 1 is `<caller>/../kernels`, 2 is `<caller>/../../kernels`.
SEARCH_ORDER_CASES: list[ParameterSet] = [
  pytest.param((0,), 0, id="local-only"),
  pytest.param((1,), 1, id="parent-only"),
  pytest.param((2,), 2, id="grandparent-only"),
  pytest.param((0, 1), 0, id="local-shadows-parent"),
  pytest.param((0, 2), 0, id="local-shadows-grandparent"),
  pytest.param((1, 2), 1, id="parent-shadows-grandparent"),
  pytest.param((0, 1, 2), 0, id="all-three-prefer-local"),
]


class RecordingCompiler:
  """Stand-in for `_compile_kernel_cached` that records the path it is given.

  Compiling CUDA is a device operation, so the CPU-side tests replace the
  compiler with this recorder: what they verify is which file `load_kernel`
  selects, not that NVRTC accepts it.
  """

  def __init__(self) -> None:
    self.calls: list[tuple[str, str]] = []
    self.compiled: object = object()

  def __call__(self, kernel_path_str: str, kernel_function_name: str) -> object:
    self.calls.append((kernel_path_str, kernel_function_name))
    return self.compiled


def caller_path(tmp_path: Path) -> Path:
  """Create `<tmp>/pkg/sub/module.py` and return it, standing in for a caller's `__file__`."""
  caller = tmp_path / "pkg" / "sub" / "module.py"
  caller.parent.mkdir(parents=True, exist_ok=True)
  caller.write_text("", encoding="utf-8")
  return caller


def candidate_dirs(caller: Path) -> list[Path]:
  """The three directories `load_kernel` searches, nearest first.

  Mirrors the resolver (the caller directory is resolved before searching) so
  the expectations stay correct when `tmp_path` itself contains symlinks.
  """
  caller_dir = caller.resolve().parent
  return [
    caller_dir / "kernels",
    caller_dir.parent / "kernels",
    caller_dir.parent.parent / "kernels",
  ]


@pytest.mark.parametrize(("existing", "expected_index"), SEARCH_ORDER_CASES)
def test_load_kernel_picks_the_nearest_existing_candidate(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  existing: tuple[int, ...],
  expected_index: int,
) -> None:
  """WHAT: `load_kernel` with the kernel file placed in every combination of the three candidates.

  WHY: strategies at different package depths rely on the nearest-directory-first
  order; resolving to the wrong file would silently run a same-named kernel from
  another package, which is far worse than a missing-file error.
  EXPECTED: the compiler receives exactly `<chosen candidate>/noop.cu` and the
  function name, and that compiled object is what `load_kernel` returns.
  """
  compiler = RecordingCompiler()
  monkeypatch.setattr(gpu_helpers, "_compile_kernel_cached", compiler)
  caller = caller_path(tmp_path)
  candidates = candidate_dirs(caller)
  for index in existing:
    candidates[index].mkdir(parents=True, exist_ok=True)
    (candidates[index] / KERNEL_FILENAME).write_text(KERNEL_SOURCE, encoding="utf-8")

  result = load_kernel(KERNEL_FILENAME, KERNEL_FUNCTION, str(caller))

  expected_path = candidates[expected_index] / KERNEL_FILENAME
  assert compiler.calls == [(str(expected_path), KERNEL_FUNCTION)]
  assert result is compiler.compiled


def test_load_kernel_reports_every_searched_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """WHAT: `load_kernel` when no candidate directory holds the kernel file.

  WHY: the exception message is the only diagnostic the caller gets; it must
  list the three absolute paths actually searched, in search order, so the
  developer knows which `.cu` file to add and where.
  EXPECTED: `FileNotFoundError` whose message is the documented sentence with
  all three candidate paths, nearest first, and no compilation attempted.
  """
  compiler = RecordingCompiler()
  monkeypatch.setattr(gpu_helpers, "_compile_kernel_cached", compiler)
  caller = caller_path(tmp_path)
  searched_paths = [str(directory / KERNEL_FILENAME) for directory in candidate_dirs(caller)]
  expected_message = f"Kernel file '{KERNEL_FILENAME}' not found. Searched locations: {', '.join(searched_paths)}"

  with pytest.raises(FileNotFoundError) as excinfo:
    load_kernel(KERNEL_FILENAME, KERNEL_FUNCTION, str(caller))

  message = str(excinfo.value)
  assert message == expected_message
  assert message.index(searched_paths[0]) < message.index(searched_paths[1]) < message.index(searched_paths[2])
  assert compiler.calls == []


@pytest.mark.gpu
def test_compile_kernel_cache_reuses_the_compiled_kernel(tmp_path: Path) -> None:
  """WHAT: two `_compile_kernel_cached` calls for the same kernel path and function name.

  WHY: the loader sits behind every strategy stage; without the `lru_cache`,
  each lookup recompiles the same `.cu` source and dominates GPU runtime, which
  no CPU-side test can observe.
  EXPECTED: the second call returns the identical `cp.RawKernel` object and the
  cache reports one hit and one miss.
  """
  kernel_path = tmp_path / KERNEL_FILENAME
  kernel_path.write_text(KERNEL_SOURCE, encoding="utf-8")

  gpu_helpers._compile_kernel_cached.cache_clear()
  try:
    first = gpu_helpers._compile_kernel_cached(str(kernel_path), KERNEL_FUNCTION)
    second = gpu_helpers._compile_kernel_cached(str(kernel_path), KERNEL_FUNCTION)
    cache_info = gpu_helpers._compile_kernel_cached.cache_info()
  finally:
    gpu_helpers._compile_kernel_cached.cache_clear()

  assert cp is not None
  assert isinstance(first, cp.RawKernel)
  assert first is second
  assert (cache_info.hits, cache_info.misses) == (1, 1)
