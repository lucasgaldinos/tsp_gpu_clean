# Chapter 4 Validation Results

Generated: 2026-10-26 19:21:10

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| att48 | CPU | 10631.33 | 4.71 | 0.03 | 21.39 | 0.02x |
| att48 | FullGPU | 10659.67 | 4.71 | 0.30 | 0.01 | 32.10x |
| att48 | HybridNaive | 10628.00 | 0.00 | 0.00 | 0.45 | — |
| att48 | HybridOptimized | 10628.00 | 0.00 | 0.00 | 0.06 | 7.71x |
| bayg29 | CPU | 1611.67 | 2.36 | 0.10 | 3.22 | 0.04x |
| bayg29 | FullGPU | 1610.00 | 0.00 | 0.00 | 0.00 | 37.51x |
| bayg29 | HybridNaive | 1611.67 | 2.36 | 0.10 | 0.13 | — |
| bayg29 | HybridOptimized | 1611.67 | 2.36 | 0.10 | 0.01 | 9.61x |
| bays29 | CPU | 2024.67 | 1.89 | 0.23 | 3.23 | 0.04x |
| bays29 | FullGPU | 2020.00 | 0.00 | 0.00 | 0.00 | 43.55x |
| bays29 | HybridNaive | 2024.67 | 1.89 | 0.23 | 0.12 | — |
| bays29 | HybridOptimized | 2024.67 | 1.89 | 0.23 | 0.01 | 8.42x |
| berlin52 | CPU | 7542.00 | 0.00 | 0.00 | 23.43 | 0.02x |
| berlin52 | FullGPU | 7562.67 | 29.23 | 0.27 | 0.01 | 34.54x |
| berlin52 | HybridNaive | 7542.00 | 0.00 | 0.00 | 0.40 | — |
| berlin52 | HybridOptimized | 7542.00 | 0.00 | 0.00 | 0.05 | 8.01x |
| brg180 | FullGPU | 1960.00 | 0.00 | 0.51 | 1.20 | 4.87x |
| brg180 | HybridNaive | 1956.67 | 4.71 | 0.34 | 5.85 | — |
| brg180 | HybridOptimized | 1956.67 | 4.71 | 0.34 | 1.18 | 4.96x |
| gr17 | CPU | 2085.00 | 0.00 | 0.00 | 0.73 | 0.14x |
| gr17 | FullGPU | 2085.00 | 0.00 | 0.00 | 0.00 | 89.25x |
| gr17 | HybridNaive | 2085.00 | 0.00 | 0.00 | 0.10 | — |
| gr17 | HybridOptimized | 2085.00 | 0.00 | 0.00 | 0.01 | 8.33x |
| gr21 | CPU | 2707.00 | 0.00 | 0.00 | 1.25 | 0.09x |
| gr21 | FullGPU | 2707.00 | 0.00 | 0.00 | 0.00 | 63.63x |
| gr21 | HybridNaive | 2707.00 | 0.00 | 0.00 | 0.11 | — |
| gr21 | HybridOptimized | 2707.00 | 0.00 | 0.00 | 0.02 | 7.36x |
| gr24 | CPU | 1272.00 | 0.00 | 0.00 | 1.87 | 0.06x |
| gr24 | FullGPU | 1272.00 | 0.00 | 0.00 | 0.00 | 54.14x |
| gr24 | HybridNaive | 1272.00 | 0.00 | 0.00 | 0.11 | — |
| gr24 | HybridOptimized | 1272.00 | 0.00 | 0.00 | 0.01 | 9.59x |
| gr48 | CPU | 5054.67 | 6.94 | 0.17 | 17.96 | 0.02x |
| gr48 | FullGPU | 5063.00 | 19.80 | 0.34 | 0.01 | 40.78x |
| gr48 | HybridNaive | 5054.67 | 6.94 | 0.17 | 0.32 | — |
| gr48 | HybridOptimized | 5054.67 | 6.94 | 0.17 | 0.04 | 7.14x |
| gr96 | CPU | 55597.67 | 85.80 | 0.70 | 163.45 | 0.01x |
| gr96 | FullGPU | 55559.00 | 38.27 | 0.63 | 0.08 | 13.91x |
| gr96 | HybridNaive | 55644.33 | 86.45 | 0.79 | 1.16 | — |
| gr96 | HybridOptimized | 55644.33 | 86.45 | 0.79 | 0.22 | 5.27x |

*Note: Speedup calculated relative to HybridNaive (— indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.14 | 0.22 | 26.28 | 49.28 | 0.03x |
| FullGPU | 0.21 | 0.23 | 0.13 | 0.36 | 6.58x |
| HybridNaive | 0.16 | 0.24 | 0.88 | 1.69 | — |
| HybridOptimized | 0.16 | 0.24 | 0.16 | 0.34 | 5.41x |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|:--------------|:---------|:---------------|-------------:|--------------:|
| Small (n<100) | att48, bayg29, bays29, berlin52, gr17, gr21, gr24, gr48, gr96 | CPU | 0.14 | 26.28 |
| Medium (100≤n<300) | brg180 | HybridNaive | 0.34 | 5.85 |

*Note: Best algorithm determined by lowest mean gap to known optimum.*
