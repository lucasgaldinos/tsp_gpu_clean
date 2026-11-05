# Selected 10 Validation

Generated: 2026-11-04 11:30:03

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| att48 | CPU | 10647.33 | 14.06 | 0.18 | 9.67 | 0.05x |
| att48 | FullGPUModular | 10688.00 | 28.71 | 0.56 | 0.04 | 11.64x |
| att48 | HybridNaive | 10639.00 | 15.56 | 0.10 | 0.49 | — |
| att48 | HybridOptimized | 10639.00 | 15.56 | 0.10 | 0.06 | 7.88x |
| bayg29 | CPU | 1611.67 | 2.36 | 0.10 | 1.80 | 0.09x |
| bayg29 | FullGPUModular | 1616.00 | 4.32 | 0.37 | 0.01 | 20.38x |
| bayg29 | HybridNaive | 1611.67 | 2.36 | 0.10 | 0.16 | — |
| bayg29 | HybridOptimized | 1611.67 | 2.36 | 0.10 | 0.02 | 9.64x |
| bays29 | CPU | 2022.00 | 2.83 | 0.10 | 1.80 | 0.07x |
| bays29 | FullGPUModular | 2020.67 | 0.94 | 0.03 | 0.01 | 19.24x |
| bays29 | HybridNaive | 2022.00 | 2.83 | 0.10 | 0.13 | — |
| bays29 | HybridOptimized | 2022.00 | 2.83 | 0.10 | 0.01 | 9.91x |
| berlin52 | CPU | 7543.67 | 2.36 | 0.02 | 14.21 | 0.03x |
| berlin52 | FullGPUModular | 7542.00 | 0.00 | 0.00 | 0.02 | 21.14x |
| berlin52 | HybridNaive | 7570.67 | 33.07 | 0.38 | 0.48 | — |
| berlin52 | HybridOptimized | 7570.67 | 33.07 | 0.38 | 0.05 | 8.85x |
| brg180 | FullGPUModular | 1956.67 | 4.71 | 0.34 | 0.79 | 10.39x |
| brg180 | HybridNaive | 1956.67 | 4.71 | 0.34 | 8.18 | — |
| brg180 | HybridOptimized | 1956.67 | 4.71 | 0.34 | 1.48 | 5.53x |
| gr17 | CPU | 2085.00 | 0.00 | 0.00 | 0.44 | 0.28x |
| gr17 | FullGPUModular | 2085.00 | 0.00 | 0.00 | 0.01 | 17.46x |
| gr17 | HybridNaive | 2085.00 | 0.00 | 0.00 | 0.13 | — |
| gr17 | HybridOptimized | 2085.00 | 0.00 | 0.00 | 0.01 | 9.66x |
| gr21 | CPU | 2707.00 | 0.00 | 0.00 | 0.89 | 0.14x |
| gr21 | FullGPUModular | 2707.00 | 0.00 | 0.00 | 0.01 | 21.88x |
| gr21 | HybridNaive | 2707.00 | 0.00 | 0.00 | 0.12 | — |
| gr21 | HybridOptimized | 2707.00 | 0.00 | 0.00 | 0.01 | 9.17x |
| gr24 | CPU | 1272.00 | 0.00 | 0.00 | 1.25 | 0.11x |
| gr24 | FullGPUModular | 1272.00 | 0.00 | 0.00 | 0.01 | 19.94x |
| gr24 | HybridNaive | 1272.00 | 0.00 | 0.00 | 0.14 | — |
| gr24 | HybridOptimized | 1272.00 | 0.00 | 0.00 | 0.01 | 10.61x |
| gr48 | CPU | 5057.67 | 3.86 | 0.23 | 13.01 | 0.03x |
| gr48 | FullGPUModular | 5075.67 | 9.10 | 0.59 | 0.02 | 20.87x |
| gr48 | HybridNaive | 5072.33 | 6.80 | 0.52 | 0.41 | — |
| gr48 | HybridOptimized | 5072.33 | 6.80 | 0.52 | 0.05 | 8.49x |
| gr96 | CPU | 55273.67 | 70.05 | 0.12 | 96.34 | 0.02x |
| gr96 | FullGPUModular | 55336.67 | 96.09 | 0.23 | 0.10 | 16.32x |
| gr96 | HybridNaive | 55521.33 | 114.27 | 0.57 | 1.58 | — |
| gr96 | HybridOptimized | 55521.33 | 114.27 | 0.57 | 0.18 | 8.71x |

*Note: Speedup calculated relative to HybridNaive (— indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.08 | 0.08 | 15.49 | 29.05 | 0.08x |
| FullGPUModular | 0.21 | 0.23 | 0.10 | 0.23 | 11.78x |
| HybridNaive | 0.21 | 0.21 | 1.18 | 2.37 | — |
| HybridOptimized | 0.21 | 0.21 | 0.19 | 0.43 | 6.24x |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|:--------------|:---------|:---------------|-------------:|--------------:|
| Small (n<100) | att48, bayg29, bays29, berlin52, gr17, gr21, gr24, gr48, gr96 | CPU | 0.08 | 15.49 |
| Medium (100≤n<300) | brg180 | FullGPUModular | 0.34 | 0.79 |

*Note: Best algorithm determined by lowest mean gap to known optimum.*
