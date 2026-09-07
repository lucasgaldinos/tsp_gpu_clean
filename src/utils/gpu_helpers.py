from pathlib import Path
import cupy as cp


def load_kernel(kernel_filename: str, kernel_function_name: str, caller_file_path: str):
    """
    Loads and compiles a CUDA kernel from a .cu file.

    Args:
        kernel_filename (str): The name of the .cu file (e.g., "my_kernel.cu").
        kernel_function_name (str): The name of the kernel function to load.
        caller_file_path (str): The __file__ attribute of the calling script.
                               This is used to locate the consolidated 'kernels' directory.

    Returns:
        A CuPy RawKernel object.
    """
    # Navigate to algorithms/kernels/ from any algorithm subdirectory
    caller_dir = Path(caller_file_path).parent
    # Go up to algorithms/ directory, then into kernels/
    kernel_dir = caller_dir.parent / "kernels"
    kernel_path = kernel_dir / kernel_filename

    if not kernel_path.exists():
        raise FileNotFoundError(
            f"Kernel file not found: {kernel_path}\n"
            f"Expected a 'kernels' directory alongside the calling script."
        )

    kernel_source = kernel_path.read_text(encoding="utf-8")
    return cp.RawKernel(kernel_source, kernel_function_name)
