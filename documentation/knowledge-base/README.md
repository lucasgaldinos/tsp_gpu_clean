# Technical Decisions Documentation

This folder contains comprehensive analyses of architectural and implementation decisions for the GPU-accelerated routing optimization project.

## Key Documents

### GPU Reduction Patterns and Lessons

**File**: [`GPU_REDUCTION_PATTERNS_AND_LESSONS.md`](./GPU_REDUCTION_PATTERNS_AND_LESSONS.md)  
**Status**: ✅ Complete (2025-01-28)  
**Length**: ~2000 lines (comprehensive reference)

**What it covers**:

1. **Bug Report**: Parallel reduction bug in 2-opt CUDA kernels (now fixed)
2. **CUB Library Integration**: How to use NVIDIA CUB with CuPy RawKernel (saves 5-7 hours per kernel!)
3. **Warp Shuffle Mechanics**: Detailed explanation of `__shfl_down_sync()`, masks (0xffffffff), and offset progression
4. **Block vs Warp Level**: Clear boundary between warp-level (32 threads) and block-level (98+ threads) operations
5. **VRP Applicability**: Current 2-opt works for TSP + intra-route VRP, missing inter-route operators
6. **Lessons Learned**: Time cost analysis, testing checklist, recommendations for future work

**Quick Navigation** (by question):

| Your Question | Section to Read |
|---------------|----------------|
| "Why did manual reduction fail?" | **Bug Evolution** (Stage 1-3) |
| "How do I use CUB with CuPy?" | **NVIDIA CUB Library: Production Alternative** (lines 86-160) |
| "What is 0xffffffff?" | **Understanding Warp Shuffle Primitives** (lines 950-995) |
| "Why offset starts at 16?" | **Warp Shuffle: Step-by-Step Breakdown** (lines 996-1075) |
| "What's the difference between warp and block?" | **WARP vs BLOCK Level Boundary** (lines 1085-1140) |
| "Does 2-opt work for VRP?" | **Applicability to Vehicle Routing Problems** (lines 165-235) |
| "Should I use CUB or manual implementation?" | **Recommendations for Future Kernel Development** (lines 1485-1540) |
| "How much time did debugging cost?" | **Time Cost Analysis** (lines 1445-1455) |

**Key Takeaways**:

- ✅ Bug fixed: 0.00% error vs CPU (was 1.79%)
- ✅ CUB library available via CuPy: `options=('-I{cuda_include}',)` in RawKernel
- ✅ Warp shuffles use 0xffffffff mask for all 32 threads participation
- ✅ Block-level needs `__syncthreads()`, warp-level doesn't (hardware synchronous)
- ✅ Current 2-opt works for TSP + VRP intra-route, missing relocate/exchange for full CVRP
- ⚠️ Manual kernel debugging cost: 9.5 hours (CUB would take 30 minutes)

**For Quick Reference**:

- CUB integration code: Lines 106-140
- Warp shuffle visual diagram: Lines 1000-1045
- Block vs warp comparison table: Lines 1115-1120
- VRP operator table: Lines 190-200
- Testing checklist: Lines 1555-1565

---

## Other Documents in This Folder

- **`2opt_parallelization_strategies_comparison.md`**: Analysis of HybridNaive vs HybridOptimized vs FullGPU
- **`compiling-tsp-heuristics-analysis.md`**: Performance analysis of compilation-based TSP heuristics
- **`cuda_synchronization_levels.md`**: Thread, warp, and block synchronization primitives
- **`cupy_amd_compatibility.md`**: Status of CuPy on AMD GPUs (ROCm)
- **`cupy_kernels_for_parallel_optimization.md`**: Guide to writing custom CUDA kernels in CuPy
- **`GPU_2OPT_INTEGRATION_ANALYSIS.md`**: Design decisions for GPU 2-opt integration
- **`non_power_of_2_block_sizes_proof.md`**: Mathematical proof of orphaned threads issue
- **`SA_COMPREHENSIVE_ANALYSIS.md`**: Simulated Annealing GPU acceleration analysis

---

## How to Use This Documentation

**When implementing new kernels**:

1. Read **CUB Library Integration** section first
2. Use CUB for primitives (reduce, scan, sort)
3. Only manual implementation if algorithm is domain-specific

**When debugging GPU kernels**:

1. Check **Testing Checklist** (lines 1555-1565)
2. Test non-power-of-2 sizes (n=98, 100, 198)
3. Create minimal reproducible test case

**When writing thesis**:

1. Cite **Academic Context & Literature Review** (lines 18-160)
2. Reference **Lessons Learned** for methodology discussion
3. Use **VRP Applicability** to clarify scope limitations

---

## Contributing

When adding new technical decisions:

1. Create markdown file with descriptive name
2. Add entry to this README with summary
3. Include code examples and diagrams
4. Link from relevant source code comments

**Document template**:

```markdown
# [Decision Title]

## Status
[Active/Deprecated/Superseded]

## Context
What problem are we solving?

## Decision
What did we choose and why?

## Consequences
What are the trade-offs?

## Alternatives Considered
What else did we evaluate?

## References
Academic papers, benchmarks, discussions
```
