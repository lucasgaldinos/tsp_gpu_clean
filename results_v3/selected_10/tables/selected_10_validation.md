# Selected 10 Validation

Generated: 2026-11-05 14:06:06

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| att48 | CPU | 10683.00 | 29.71 | 0.52 | 4.36 | 0.12x |
| att48 | FullGPUModular | 10637.33 | 13.20 | 0.09 | 0.04 | 11.80x |
| att48 | HybridNaive | 10639.00 | 15.56 | 0.10 | 0.52 | — |
| att48 | HybridOptimized | 10663.33 | 7.59 | 0.33 | 0.06 | 8.41x |
| bayg29 | CPU | 1613.33 | 4.71 | 0.21 | 1.42 | 0.11x |
| bayg29 | FullGPUModular | 1619.00 | 6.48 | 0.56 | 0.01 | 19.91x |
| bayg29 | HybridNaive | 1611.67 | 2.36 | 0.10 | 0.16 | — |
| bayg29 | HybridOptimized | 1611.67 | 2.36 | 0.10 | 0.02 | 7.69x |
| bays29 | CPU | 2024.00 | 5.66 | 0.20 | 1.33 | 0.11x |
| bays29 | FullGPUModular | 2020.67 | 0.94 | 0.03 | 0.01 | 15.35x |
| bays29 | HybridNaive | 2022.00 | 2.83 | 0.10 | 0.15 | — |
| bays29 | HybridOptimized | 2022.67 | 2.49 | 0.13 | 0.02 | 9.55x |
| berlin52 | CPU | 7578.00 | 25.46 | 0.48 | 4.00 | 0.11x |
| berlin52 | FullGPUModular | 7542.00 | 0.00 | 0.00 | 0.02 | 21.04x |
| berlin52 | HybridNaive | 7570.67 | 33.07 | 0.38 | 0.46 | — |
| berlin52 | HybridOptimized | 7566.33 | 34.41 | 0.32 | 0.06 | 8.25x |
| brg180 | FullGPUModular | 1960.00 | 0.00 | 0.51 | 0.38 | 21.51x |
| brg180 | HybridNaive | 1956.67 | 4.71 | 0.34 | 8.17 | — |
| brg180 | HybridOptimized | 1960.00 | 0.00 | 0.51 | 0.91 | 9.03x |
| gr17 | CPU | 2085.00 | 0.00 | 0.00 | 0.45 | 0.24x |
| gr17 | FullGPUModular | 2085.00 | 0.00 | 0.00 | 0.01 | 14.38x |
| gr17 | HybridNaive | 2085.00 | 0.00 | 0.00 | 0.11 | — |
| gr17 | HybridOptimized | 2085.00 | 0.00 | 0.00 | 0.01 | 7.25x |
| gr21 | CPU | 2707.00 | 0.00 | 0.00 | 0.53 | 0.22x |
| gr21 | FullGPUModular | 2707.00 | 0.00 | 0.00 | 0.01 | 13.07x |
| gr21 | HybridNaive | 2707.00 | 0.00 | 0.00 | 0.12 | — |
| gr21 | HybridOptimized | 2707.00 | 0.00 | 0.00 | 0.01 | 8.23x |
| gr24 | CPU | 1274.33 | 3.30 | 0.18 | 0.58 | 0.22x |
| gr24 | FullGPUModular | 1272.00 | 0.00 | 0.00 | 0.01 | 14.99x |
| gr24 | HybridNaive | 1272.00 | 0.00 | 0.00 | 0.13 | — |
| gr24 | HybridOptimized | 1272.00 | 0.00 | 0.00 | 0.01 | 8.79x |
| gr48 | CPU | 5073.33 | 18.26 | 0.54 | 3.48 | 0.11x |
| gr48 | FullGPUModular | 5060.00 | 13.93 | 0.28 | 0.02 | 20.02x |
| gr48 | HybridNaive | 5072.33 | 6.80 | 0.52 | 0.38 | — |
| gr48 | HybridOptimized | 5056.33 | 8.22 | 0.20 | 0.05 | 8.12x |
| gr96 | CPU | 55590.00 | 50.62 | 0.69 | 18.78 | 0.09x |
| gr96 | FullGPUModular | 55615.67 | 107.13 | 0.74 | 0.07 | 24.44x |
| gr96 | HybridNaive | 55521.33 | 114.27 | 0.57 | 1.73 | — |
| gr96 | HybridOptimized | 55594.67 | 126.91 | 0.70 | 0.26 | 6.54x |

*Note: Speedup calculated relative to HybridNaive (— indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.31 | 0.24 | 3.88 | 5.47 | 0.31x |
| FullGPUModular | 0.22 | 0.27 | 0.06 | 0.11 | 20.63x |
| HybridNaive | 0.21 | 0.21 | 1.19 | 2.37 | — |
| HybridOptimized | 0.23 | 0.22 | 0.14 | 0.26 | 8.43x |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|:--------------|:---------|:---------------|-------------:|--------------:|
| Small (n<100) | att48, bayg29, bays29, berlin52, gr17, gr21, gr24, gr48, gr96 | FullGPUModular | 0.19 | 0.02 |
| Medium (100≤n<300) | brg180 | HybridNaive | 0.34 | 8.17 |

*Note: Best algorithm determined by lowest mean gap to known optimum.*
