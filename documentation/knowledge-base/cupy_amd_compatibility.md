# CuPy AMD GPU Compatibility

**Document Purpose:** Technical analysis of CuPy compatibility with AMD GPUs, ROCm support status, and architectural differences between NVIDIA CUDA and AMD compute platforms.

**Status as of 2025:** CuPy provides **experimental** support for AMD GPUs via ROCm backend.

---

## Executive Summary

**Key Findings:**

1. **CuPy AMD Support Status:** Experimental (not production-ready)
2. **Dependency:** Requires AMD ROCm $\geq 3.5$ and HIP (Heterogeneous-Compute Interface for Portability)
3. **Installation Method:** Must build from source (no precompiled wheels for ROCm as of v8.6.0)
4. **Architectural Difference:** NVIDIA CUDA cores ≠ AMD Stream Processors (different granularity and organization)
5. **Ecosystem Maturity:** NVIDIA CUDA has significantly more mature tooling and library support compared to AMD ROCm

**Recommendation for This Work:**

- Target NVIDIA GPUs exclusively (GTX 1050, RTX 3090, A100) due to mature CUDA ecosystem
- CuPy's ROCm support is experimental and would introduce deployment complexity
- Focus on stable CUDA backend for reproducible academic results

---

## 1. Does CuPy Run on AMD GPUs?

**Answer:** Yes, with significant limitations.

CuPy provides **experimental** support for AMD GPUs through the ROCm platform. However, this support requires:

1. **Building from source:** No pip-installable wheels available for ROCm (as of CuPy v8.6.0)
2. **ROCm installation:** AMD's Radeon Open Compute platform $\geq$ v3.5
3. **HIP compatibility layer:** Translates CUDA-like code to AMD GPU instructions
4. **Manual configuration:** Must set environment variables (`HCC_AMDGPU_TARGET`, `CUPY_INSTALL_USE_HIP=1`, `ROCM_HOME`)

**Installation Example (from CuPy documentation):**

```bash
# Install ROCm libraries (Ubuntu/Debian)
sudo apt install hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim

# Configure build for specific GPU architecture
export HCC_AMDGPU_TARGET=gfx900  # Example: RX Vega 56/64
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm

# Build CuPy from source
pip install cupy
```

**Compatibility Caveats:**

- **Experimental status:** No production-ready guarantees for stability or performance
- **GPU architecture detection:** Must manually specify ISA name (e.g., `gfx900`, `gfx1030`) via `rocminfo`
- **Limited testing:** ROCm backend receives less testing than CUDA backend in CuPy
- **API coverage:** Not all CuPy features guaranteed to work on ROCm

**References:**

- CuPy Documentation: [Using CuPy on AMD GPU (experimental)](https://docs.cupy.dev/en/v8.6.0/install_rocm.html)
- AMD ROCm: [Official Documentation](https://rocmdocs.amd.com/)

---

## 2. CUDA Cores vs AMD GPU Architecture

**Fundamental Difference:** NVIDIA CUDA cores and AMD Stream Processors/Compute Units are **not equivalent** in granularity or organization.

### NVIDIA CUDA Architecture

- **CUDA Core:** Smallest processing element (ALU - Arithmetic Logic Unit)
- **Streaming Multiprocessor (SM):** Cluster of $64$-$128$ CUDA cores sharing control logic and memory
- **Grid/Block/Thread Hierarchy:** Programmer-exposed parallelism model
- **SIMT Execution:** Single Instruction Multiple Thread with warp size of $32$ threads

**Example:** GTX 1050 Mobile

- $640$ CUDA cores
- $5$ SMs ($128$ cores per SM)
- Each thread executes independently within warp constraints

### AMD GPU Architecture

AMD uses two different naming conventions:

**Consumer GPUs (RDNA - Radeon DNA):**

- **Stream Processor:** Equivalent to CUDA core (ALU)
- **Compute Unit (CU):** Cluster of $64$ Stream Processors
- **Workgroup/Wavefront Model:** Similar to CUDA blocks/warps but with wavefront size of $64$ threads (RDNA 1/2) or $32$ threads (RDNA 3)

**Data Center GPUs (CDNA - Compute DNA):**

- **Compute Unit (CU):** Optimized for HPC/AI workloads
- **Matrix Cores:** Tensor-like acceleration (similar to NVIDIA Tensor Cores)
- **Higher double-precision performance:** Unlike gaming GPUs

**Example:** AMD Radeon RX 7900 XTX (RDNA 3)

- $6144$ Stream Processors
- $96$ Compute Units ($64$ Stream Processors per CU)
- Wavefront size: $32$ threads (aligned with NVIDIA's warp size in RDNA 3)

**Key Architectural Differences:**

| Feature | NVIDIA CUDA | AMD ROCm |
|---------|-------------|----------|
| **Basic Unit** | CUDA Core (ALU) | Stream Processor (ALU) |
| **Core Cluster** | SM ($64$-$128$ cores) | CU ($64$ Stream Processors) |
| **Warp/Wavefront Size** | $32$ threads | $64$ (RDNA 1/2), $32$ (RDNA 3) |
| **Memory Hierarchy** | Registers → Shared → L1/L2 → Global | Registers → LDS → L1/L2 → Global |
| **Programming Model** | CUDA C/C++ | HIP (CUDA-like), OpenCL |

**Performance Comparison Challenges:**

- **Cannot compare core counts directly:** $1000$ CUDA cores ≠ $1000$ Stream Processors
- **Clock frequency differences:** AMD cores typically run at higher frequencies
- **Memory bandwidth:** Varies by model (GTX 1050: $\sim 112$ GB/s, RX 7900 XTX: $\sim 960$ GB/s)
- **Architecture optimizations:** NVIDIA focuses on specialized cores (Tensor/RT cores), AMD emphasizes cache hierarchy

**References:**

- Medium: [Decoding CPU vs. GPU: NVIDIA and AMD GPU Architectures](https://medium.com/@rohithreddy66666/decoding-cpu-vs-gpu-nvidia-and-amd-architectures-e09ebfb594ed)
- CGDirector: [CUDA Cores vs Stream Processors](https://www.cgdirector.com/cuda-cores-vs-stream-processors/)

---

## 3. ROCm as CUDA Alternative

**ROCm (Radeon Open Compute)** is AMD's open-source platform for GPU computing, designed as a CUDA alternative.

### ROCm Components

1. **HIP (Heterogeneous-Compute Interface for Portability):**
   - CUDA-like API for GPU programming
   - Syntax nearly identical to CUDA C/C++
   - Can compile to both AMD and NVIDIA GPUs with minimal code changes
   - Example: `hipMalloc()` vs `cudaMalloc()`, `hipMemcpy()` vs `cudaMemcpy()`

2. **ROCm Libraries:**
   - `hipBLAS`: BLAS operations (equivalent to cuBLAS)
   - `rocSPARSE`: Sparse linear algebra (equivalent to cuSPARSE)
   - `rocFFT`: Fast Fourier Transforms (equivalent to cuFFT)
   - `rocRAND`: Random number generation (equivalent to cuRAND)
   - `rocThrust`: Parallel algorithms (equivalent to Thrust)

3. **ROCm Runtime:**
   - Device management and kernel execution
   - Memory management (host ↔ device transfers)
   - Stream/event synchronization

### HIP Code Example (CUDA-Compatible)

```cpp
// HIP code (compiles for AMD or NVIDIA)
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code
hipMalloc(&d_a, size);
hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0, d_a, d_b, d_c, n);
hipDeviceSynchronize();
```

**HIP Compilation:**

- For AMD: `hipcc --amdgpu-target=gfx900 kernel.cpp -o kernel`
- For NVIDIA: `hipcc --x=cuda kernel.cpp -o kernel` (compiles to CUDA)

### ROCm vs CUDA Ecosystem Comparison

| Aspect | NVIDIA CUDA | AMD ROCm |
|--------|-------------|----------|
| **Maturity** | $15+$ years (since 2006) | $\sim 8$ years (since 2016) |
| **Hardware Support** | All NVIDIA GPUs since 2006 | AMD GCN 3.0+ (2015+), RDNA, CDNA |
| **Library Ecosystem** | Extensive (cuDNN, TensorRT, NCCL, etc.) | Growing (MIOpen, RCCL, hipBLAS, etc.) |
| **Framework Support** | PyTorch, TensorFlow, JAX (native) | PyTorch, TensorFlow (via ROCm builds) |
| **Tooling** | Nsight Systems/Compute, CUDA-GDB | ROCProfiler, rocgdb |
| **Windows Support** | Full support | Limited (Linux-first platform) |
| **Python Libraries** | CuPy (stable), Numba (stable) | CuPy (experimental), limited Numba support |

**ROCm Advantages:**

- Open-source (Apache 2.0 license)
- Cross-vendor potential (AMD + NVIDIA via HIP)
- Strong support for HPC/scientific computing on AMD MI-series GPUs

**ROCm Limitations:**

- Smaller user community and fewer tutorials
- Fewer pre-trained models and libraries compared to CUDA
- Consumer GPU support varies (gaming-focused RDNA less prioritized than data center CDNA)

**References:**

- AMD ROCm Blogs: [CuPy and hipDF on AMD](https://rocm.blogs.amd.com/artificial-intelligence/cupy_hipdf_portfolio_opt/README.html)
- System76 Support: [Install ROCm Guide](https://support.system76.com/articles/rocm/)

---

## 4. Current State of Python Scientific Computing on AMD GPUs (2025)

### CuPy ROCm Support Status

**PyPI Packages Available:**

- `cupy-rocm-4-3`, `cupy-rocm-5-0`: Precompiled wheels for specific ROCm versions
- **Important:** These wheels are outdated (ROCm 4.3 released 2021, ROCm 5.0 released 2022)
- **Current ROCm version:** 6.1+ (as of 2024-2025)
- **Implication:** May need to build from source for latest ROCm compatibility

**Installation Example (Prebuilt Wheel):**

```bash
# For ROCm 5.0 (if compatible with your ROCm installation)
pip install cupy-rocm-5-0

# Verify installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

**Known Limitations:**

1. **Custom CUDA Kernels:** `cp.RawKernel()` compatibility with HIP not fully tested
2. **Performance Parity:** May not match CUDA performance due to less optimized code paths
3. **Documentation Gaps:** Fewer examples and troubleshooting resources for ROCm backend

### Alternative AMD GPU Libraries for Python

Since CuPy's ROCm support is experimental, consider these alternatives:

**1. hipDF (ROCm-Native):**

- DataFrames library (similar to Pandas/cuDF)
- Directly targets ROCm without CUDA translation layer
- Better optimized for AMD GPUs than CuPy-ROCm
- Example usage:
  ```python
  import hipdf
  df = hipdf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
  result = df['a'] * df['b']  # Executes on AMD GPU
  ```

**2. PyTorch with ROCm:**

- Mature ROCm backend (officially supported by AMD and PyTorch team)
- Installation: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`
- Tensor operations run on AMD GPU via ROCm
- More reliable than CuPy-ROCm for production workloads

**3. TensorFlow with ROCm:**

- Official ROCm support (AMD maintains fork)
- Installation: Use AMD's TensorFlow-ROCm Docker images
- Complex setup compared to NVIDIA CUDA + TensorFlow

**4. JAX (Experimental ROCm Support):**

- Google's numerical computing library
- Community-driven ROCm backend (not officially supported)
- Less stable than PyTorch/TensorFlow ROCm

### Ecosystem Maturity Assessment (2025)

| Library | NVIDIA CUDA | AMD ROCm | Verdict |
|---------|-------------|----------|---------|
| **CuPy** | Stable | Experimental | CUDA recommended |
| **PyTorch** | Stable | Stable (AMD-supported) | Both viable |
| **TensorFlow** | Stable | Stable (AMD fork) | CUDA preferred (simpler) |
| **Numba** | Stable | Experimental | CUDA only |
| **JAX** | Stable | Experimental (community) | CUDA only |

**References:**

- PyPI: [cupy-rocm-5-0 Package](https://pypi.org/project/cupy-rocm-5-0/)
- Reddit Discussion: [Using AMD GPU for ML tasks with NumPy alternatives](https://www.reddit.com/r/AyyMD/comments/uvaco5/any_way_i_use_and_amd_gpu_for_machine_learning/)

---

## 5. Recommendations for This Work

### Why This Work Targets NVIDIA CUDA Exclusively

**Academic Reproducibility:**

- CUDA has $15+$ years of stable APIs and mature tooling
- CuPy-CUDA is production-ready with extensive testing and documentation
- Results are reproducible across NVIDIA GPU generations (Pascal → Ampere → Hopper)

**Implementation Complexity:**

- CuPy-CUDA: `pip install cupy-cuda12x` (one command)
- CuPy-ROCm: Build from source, configure environment variables, debug experimental features
- Focus on algorithmic contributions, not platform compatibility

**Performance Validation:**

- CUDA Best Practices Guide provides clear optimization guidelines
- CuPy-CUDA performance well-documented in literature
- ROCm backend performance characteristics less studied for metaheuristics

**Hardware Availability:**

- GTX 1050 Mobile (development), RTX 3090/A100 (benchmarking) readily available
- AMD MI-series GPUs (CDNA for HPC) expensive and less accessible for academic work
- Consumer AMD GPUs (RDNA) prioritize gaming over compute workloads

### Future Work: Multi-Backend Support

If AMD GPU support becomes a future research direction, consider:

1. **Abstract Backend Interface:**
   ```python
   from protocols import BackendProtocol
   
   def get_backend(backend: str = 'numpy') -> BackendProtocol:
       if backend == 'numpy':
           return numpy
       elif backend == 'cupy':
           return cupy  # CUDA or ROCm detected automatically
       # CuPy handles CUDA vs ROCm internally based on available hardware
   ```

2. **HIP Kernel Compatibility:**
   - Evaluate `cp.RawKernel()` compatibility with HIP-compiled kernels
   - Test 2-opt GPU implementation on AMD hardware
   - Document performance differences (CUDA vs ROCm)

3. **Conditional Dependencies:**
   ```toml
   [project.optional-dependencies]
   cuda = ["cupy-cuda12x>=13.0.0"]
   rocm = ["cupy-rocm-5-0>=13.0.0"]  # Or build from source
   ```

4. **Benchmark Parity Testing:**
   - Run SA/GA/ACO benchmarks on NVIDIA (CUDA) and AMD (ROCm)
   - Compare speedup, memory usage, kernel launch overhead
   - Publish multi-backend performance study

---

## 6. Conclusion

**CuPy AMD GPU Compatibility Summary:**

✅ **Technically Possible:** CuPy supports AMD GPUs via ROCm (experimental)  
⚠️ **Practical Limitations:** Requires source build, manual configuration, experimental status  
❌ **Not Recommended for This Work:** CUDA ecosystem maturity and reproducibility prioritized  

**CUDA vs ROCm Decision:**

For this TCC (undergraduate thesis) work targeting GPU-accelerated metaheuristics for routing problems:

- **Target Platform:** NVIDIA CUDA (GTX 1050, RTX 3090, A100)
- **Rationale:** Stable ecosystem, mature CuPy support, reproducible results, extensive literature
- **Future Work:** Multi-backend support (CUDA + ROCm) could be explored in graduate research

**Key Takeaway:**

AMD GPUs with ROCm represent a viable alternative to NVIDIA CUDA for GPU computing, but the ecosystem maturity gap (especially for Python scientific libraries like CuPy) makes NVIDIA the pragmatic choice for academic work requiring reproducibility and reliability.

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Author:** [Your Name]  
**References:** See inline citations throughout document
