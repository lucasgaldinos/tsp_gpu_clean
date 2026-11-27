from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from protocols.backend import require_gpu

if TYPE_CHECKING:
  import cupy as cp


#: Directory holding generic CUDA reduction headers (`block_reductions.cuh`).
UTILS_KERNELS_DIR: Path = Path(__file__).resolve().parent / "kernels"

#: Directory holding device-side distance headers (`pairwise_device.cuh`).
DISTANCE_KERNELS_DIR: Path = Path(__file__).resolve().parents[1] / "distances" / "kernels"

#: Last source digest seen per kernel path, used to invalidate a stale compiled binary.
#: `lru_cache` keys on the path, so without this an edited `.cu` keeps running the
#: binary built from the previous revision inside a long-lived process (`BUG-58`).
_KERNEL_SOURCE_DIGESTS: dict[str, str] = {}


def _kernel_compile_options(kernel_dir: Path | None = None) -> tuple[str, ...]:
  """Return the nvcc options every JIT-compiled kernel is built with.

  Includes `-I<src/utils/kernels>` (`block_reductions.cuh`) and
  `-I<src/distances/kernels>` (`pairwise_device.cuh`, `DP10-19`) plus
  `-I<kernel_dir>` when distinct so co-located `.cuh` headers resolve cleanly.
  """
  options: list[str] = [f"-I{UTILS_KERNELS_DIR}", f"-I{DISTANCE_KERNELS_DIR}"]
  if kernel_dir is not None and kernel_dir not in (UTILS_KERNELS_DIR, DISTANCE_KERNELS_DIR):
    options.append(f"-I{kernel_dir}")
  return tuple(options)


@functools.lru_cache(maxsize=32)
def _compile_kernel_cached(kernel_path_str: str, kernel_function_name: str) -> cp.RawKernel:
  """Compile and cache a CUDA kernel from an absolute file path."""
  cp = require_gpu()
  kernel_path = Path(kernel_path_str)
  kernel_source = kernel_path.read_text(encoding="utf-8")
  return cp.RawKernel(
    kernel_source,
    kernel_function_name,
    options=_kernel_compile_options(kernel_path.parent),
  )


def load_kernel(kernel_filename: str, kernel_function_name: str, caller_file_path: str) -> cp.RawKernel:
  """
  Loads and compiles a CUDA kernel from a .cu file (with compilation caching).

  Searches in the following order relative to `caller_file_path`:
  1. `<caller_dir>/kernels/<kernel_filename>` (modular strategy/monolithic local kernels)
  2. `<caller_dir>/../kernels/<kernel_filename>` (parent kernels directory)
  3. `<caller_dir>/../../kernels/<kernel_filename>` (package-level `algorithms/kernels`)

  Args:
      kernel_filename (str): The name of the .cu file (e.g., "my_kernel.cu").
      kernel_function_name (str): The name of the kernel function to load.
      caller_file_path (str): The __file__ attribute of the calling script.

  Returns:
      A CuPy RawKernel object.
  """
  caller_dir = Path(caller_file_path).resolve().parent
  candidate_dirs = [
    caller_dir / "kernels",
    caller_dir.parent / "kernels",
    caller_dir.parent.parent / "kernels",
  ]

  for kernel_dir in candidate_dirs:
    kernel_path = kernel_dir / kernel_filename
    if kernel_path.exists():
      _invalidate_if_source_changed(kernel_path)
      return _compile_kernel_cached(str(kernel_path), kernel_function_name)

  searched = ", ".join(str(d / kernel_filename) for d in candidate_dirs)
  raise FileNotFoundError(f"Kernel file '{kernel_filename}' not found. Searched locations: {searched}")


def _invalidate_if_source_changed(kernel_path: Path) -> None:
  """Drop every cached binary when a kernel source differs from the last one compiled.

  The compilation cache is keyed by path, which is what keeps it cheap, but that key
  says nothing about the file contents. Editing a `.cu` inside a long-lived process ---
  a running benchmark, an interactive session --- would otherwise keep executing the
  binary built from the previous revision, with no error and no log, so the collected
  numbers could come from code that is no longer in the repository (`BUG-58`).
  Hashing a few kilobytes per lookup is far cheaper than the compilation it guards.
  """
  digest: str = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
  previous: str | None = _KERNEL_SOURCE_DIGESTS.get(str(kernel_path))
  if previous is not None and previous != digest:
    _compile_kernel_cached.cache_clear()
  _KERNEL_SOURCE_DIGESTS[str(kernel_path)] = digest


def prev_power_of_2(value: int, min_val: int = 1, max_val: int = 256) -> int:
  """Round `value` down to the greatest power of 2 in `[min_val, max_val]` for CUDA block/warp sizing."""
  power = 1 << ((max(1, value)).bit_length() - 1)
  return max(min_val, min(power, max_val))

