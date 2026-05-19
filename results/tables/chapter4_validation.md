# Chapter 4 Validation Results

Generated: 2025-11-20 22:37:22

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| berlin52 | CPU | 7542.00 | 0.00 | 0.00 | 20.64 | — |
| berlin52 | FullGPU | 7542.00 | 0.00 | 0.00 | 0.30 | 68.98× |
| berlin52 | HybridNaive | 7542.00 | 0.00 | 0.00 | 0.58 | 35.85× |
| berlin52 | HybridOptimized | 7542.00 | 0.00 | 0.00 | 0.11 | 184.50× |
| eil51 | CPU | 426.50 | 0.50 | 0.12 | 61.32 | — |
| eil51 | FullGPU | 429.00 | 0.00 | 0.70 | 0.36 | 170.35× |
| eil51 | HybridNaive | 427.00 | 0.00 | 0.23 | 5.74 | 10.68× |
| eil51 | HybridOptimized | 426.00 | 0.00 | 0.00 | 0.12 | 529.47× |

*Note: Speedup calculated relative to CPU (—\ indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.06 | 0.06 | 40.98 | 20.34 | — |
| FullGPU | 0.35 | 0.35 | 0.33 | 0.03 | 124.34× |
| HybridNaive | 0.12 | 0.12 | 3.16 | 2.58 | 12.98× |
| HybridOptimized | 0.00 | 0.00 | 0.11 | 0.00 | 359.99× |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|---------------|----------|----------------|--------------|---------------|
| Small (n<100) | berlin52, eil51 | HybridOptimized | 0.00 | 0.11 |