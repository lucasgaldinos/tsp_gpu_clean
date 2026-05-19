# Hardware & Software Specifications

**Benchmark System Configuration**  
**Date**: 2025-11-20

## Hardware Specifications

### GPU (Primary Compute Device)

- **Model**: NVIDIA GeForce GTX 1050 Mobile
- **VRAM**: 4GB GDDR5
- **CUDA Cores**: 640
- **Base Clock**: 1354 MHz
- **Boost Clock**: 1493 MHz
- **Memory Bandwidth**: 112 GB/s
- **Compute Capability**: 6.1
- **TDP**: 75W (mobile variant)

**Classification**: Entry-level mobile GPU (2016 architecture, Pascal generation)

### CPU (Baseline Comparison)

- **Architecture**: x86_64
- **Usage**: CPU baseline benchmarks (small problems only, n < 100)
- **Note**: Specific model TBD - document with `lscpu` output

### Memory

- **System RAM**: TBD - document with `free -h`
- **VRAM Constraint**: 65% utilization limit (2.6GB usable)
  - Reason: Stability on mobile GPU with shared system resources
  - Impact: Maximum problem size ~18,000 cities (double precision distance matrix)

### Storage

- **Type**: SSD (assumed for reasonable I/O)
- **Usage**: TSP problem loading from `routing.duckdb`

## Software Environment

### Operating System

- **OS**: Linux
- **Distribution**: TBD (check with `lsb_release -a`)
- **Kernel**: TBD (check with `uname -r`)

### Python Environment

- **Python Version**: 3.10.16
- **Environment Manager**: pyenv
- **Virtual Environment**: .venv (project-local)

### GPU Computing Stack

- **CUDA Toolkit**: TBD (check with `nvcc --version`)
  - Recommended: CUDA 11.x or 12.x
- **NVIDIA Driver**: TBD (check with `nvidia-smi`)
  - Must support CUDA Toolkit version
- **CuPy Version**: TBD (check with `python -c "import cupy; print(cupy.__version__)"`)
  - Must match CUDA version (cupy-cuda11x or cupy-cuda12x)

### Core Dependencies

```python
# From pyproject.toml
numpy >= 1.24.0        # CPU arrays and operations
cupy                   # GPU arrays (CUDA-compatible)
scipy >= 1.10.0        # Statistical tests
duckdb >= 0.9.0        # TSP problem database
```

### Development Tools

- **Version Control**: Git (branch: dev)
- **Type Checking**: mypy (configured in mypy.ini)
- **Code Quality**: Ruff (linting and formatting)

## Benchmark Configuration

### Problem Set

- **Source**: `datasets/routing.duckdb`
- **Instances**: 38 TSP problems from TSPLIB
- **Size Range**: 51 to 1002 cities
- **Format**: Distance matrices (double precision, symmetric)

### Algorithm Parameters (GA_PARAMS)

```python
{
    "population_size": 256,
    "mutation_rate": 0.02,
    "tournament_size": 5,
    "two_opt_iterations": 10,
    "seed": 42                    # For reproducibility
}
```

### Benchmark Parameters (BENCHMARK_PARAMS)

```python
{
    "repetitions": 30,             # Statistical power
    "patience": 50,                # Early stopping generations
    "skip_cpu_default": False,     # Include CPU baseline
    "cpu_size_threshold": 100      # CPU only for n < 100
}
```

### Timing Methodology

- **Method**: `time.perf_counter()` (wall-clock time)
- **Granularity**: Microsecond precision
- **Scope**: Complete algorithm execution (initialization through final solution)
- **Exclusions**: Problem loading time, result logging

### Memory Tracking

- **Host-to-Device (H2D)**: Transfer size in MB
- **Device-to-Host (D2H)**: Transfer size in MB
- **Kernel Launches**: Count of GPU kernel invocations
- **Purpose**: Overhead analysis for hybrid vs full-GPU implementations

## Performance Context

### Expected Speedups (Literature Baseline)

**Fujimoto & Tsutsui (2011)** on Tesla C2050 (2011 research GPU):

- Speedup range: 203x - 9,573x over CPU
- Problem sizes: 100 - 1,000 cities
- Hardware advantage: High-end research GPU

**This Work** on GTX 1050 Mobile (2025 entry-level GPU):

- Expected speedup: 30x - 100x over CPU (small problems)
- Hardware constraint: Entry-level mobile GPU (4GB VRAM)
- Advantage: Validates GPU benefits on consumer hardware

### Hardware Evolution Context

- **2011**: Tesla C2050 (research GPU, ~$2000)
- **2025**: GTX 1050 Mobile (consumer GPU, ~$200 equivalent)
- **Implication**: GPU acceleration viable on budget hardware

## Reproducibility Checklist

To reproduce these results, ensure:

1. ✅ **GPU**: NVIDIA GPU with CUDA support (Compute Capability ≥ 3.5)
2. ✅ **VRAM**: ≥4GB for full problem set (≥8GB recommended for n > 1000)
3. ✅ **CUDA**: Toolkit matching CuPy version
4. ✅ **Python**: 3.10+ with dependencies from pyproject.toml
5. ✅ **Dataset**: routing.duckdb with 38 TSPLIB instances
6. ✅ **Seed**: seed=42 (reproducible random initialization)

## System Information Commands

Run these to complete hardware documentation:

```bash
# GPU info
nvidia-smi
nvcc --version

# CPU info
lscpu
cat /proc/cpuinfo | grep "model name" | head -1

# Memory info
free -h

# OS info
lsb_release -a
uname -r

# Python environment
python --version
python -c "import cupy; print(f'CuPy: {cupy.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"

# CUDA runtime
python -c "import cupy; print(f'CUDA: {cupy.cuda.runtime.runtimeGetVersion()}')"
```

## Citation Information

When citing this work, include:

- GPU model: NVIDIA GeForce GTX 1050 Mobile (4GB VRAM)
- Software: Python 3.10, CuPy, NumPy
- Dataset: 38 TSPLIB instances (51-1002 cities)
- Methodology: 30 repetitions, early stopping (patience=50)

**Example Citation Format**:
> Experiments conducted on NVIDIA GeForce GTX 1050 Mobile (4GB VRAM) using Python 3.10 with CuPy for GPU acceleration. Benchmarks performed on 38 TSPLIB instances (n=51 to n=1002) with 30 repetitions per algorithm and early stopping after 50 generations without improvement.

## Performance Disclaimers

1. **Hardware-Dependent**: Absolute runtimes vary by GPU model
2. **Relative Performance**: Speedup ratios more portable than absolute times
3. **Problem-Specific**: Performance patterns depend on problem characteristics
4. **Early Stopping**: Actual generations vary (adaptive to convergence)
5. **VRAM Constraint**: 65% limit for stability on mobile GPU

## Future Hardware Considerations

For extended experiments (beyond thesis scope):

- **Larger VRAM** (8GB+): Enable n > 1000 without memory constraints
- **Desktop GPU**: Higher bandwidth and compute capability
- **Multi-GPU**: Parallel problem solving (38 problems × 30 reps ideal for GPU farm)
- **CPU Upgrade**: Meaningful CPU baseline on larger problems

---

**Last Updated**: 2025-11-20  
**Status**: Hardware specs partially documented - complete after benchmark setup
