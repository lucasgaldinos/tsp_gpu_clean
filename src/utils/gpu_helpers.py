from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path

if "CUPY_CACHE_DIR" not in os.environ and not os.access(os.path.expanduser("~"), os.W_OK):
  os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

import cupy as cp


@functools.lru_cache(maxsize=32)
def _compile_kernel_cached(kernel_path_str: str, kernel_function_name: str) -> cp.RawKernel:
  """Compile and cache a CUDA kernel from an absolute file path."""
  kernel_source = Path(kernel_path_str).read_text(encoding="utf-8")
  return cp.RawKernel(kernel_source, kernel_function_name)


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
