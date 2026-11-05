from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

from protocols.backend import require_gpu

if TYPE_CHECKING:
  import cupy as cp


#: Directory holding device-side distance headers (`pairwise_device.cuh`).
DISTANCE_KERNELS_DIR: Path = Path(__file__).resolve().parents[1] / "distances" / "kernels"


def _kernel_compile_options() -> tuple[str, ...]:
  """Return the nvcc options every JIT-compiled kernel is built with.

  `-I<src/distances/kernels>` lets any kernel `#include "pairwise_device.cuh"`
  (`DP10-19`) without copying the header next to the `.cu` file.
  """
  return (f"-I{DISTANCE_KERNELS_DIR}",)


@functools.lru_cache(maxsize=32)
def _compile_kernel_cached(kernel_path_str: str, kernel_function_name: str) -> cp.RawKernel:
  """Compile and cache a CUDA kernel from an absolute file path."""
  cp = require_gpu()
  kernel_source = Path(kernel_path_str).read_text(encoding="utf-8")
  return cp.RawKernel(kernel_source, kernel_function_name, options=_kernel_compile_options())


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
      return _compile_kernel_cached(str(kernel_path), kernel_function_name)

  searched = ", ".join(str(d / kernel_filename) for d in candidate_dirs)
  raise FileNotFoundError(f"Kernel file '{kernel_filename}' not found. Searched locations: {searched}")
