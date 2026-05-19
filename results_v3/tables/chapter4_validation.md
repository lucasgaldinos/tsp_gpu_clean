# Chapter 4 Validation Results

Generated: 2025-11-28 00:21:36

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| berlin52 | CPU | 7542.00 | 0.00 | 0.00 | 33.11 | 0.01× |
| berlin52 | FullGPU | 7542.00 | 0.00 | 0.00 | 0.10 | 5.04× |
| berlin52 | HybridNaive | 7550.00 | 8.00 | 0.11 | 0.50 | — |
| berlin52 | HybridOptimized | 7542.00 | 0.00 | 0.00 | 0.13 | 3.95× |
| eil51 | CPU | 429.00 | 1.00 | 0.70 | 32.21 | 0.02× |
| eil51 | FullGPU | 427.00 | 0.00 | 0.23 | 0.14 | 5.46× |
| eil51 | HybridNaive | 428.00 | 1.00 | 0.47 | 0.79 | — |
| eil51 | HybridOptimized | 429.00 | 0.00 | 0.70 | 0.27 | 2.97× |

*Note: Speedup calculated relative to HybridNaive (— indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.35 | 0.35 | 32.66 | 0.45 | — |
| FullGPU | 0.12 | 0.12 | 0.12 | 0.02 | — |
| HybridNaive | 0.29 | 0.18 | 0.64 | 0.15 | — |
| HybridOptimized | 0.35 | 0.35 | 0.20 | 0.07 | 3.28× |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|:--------------|:---------|:---------------|-------------:|--------------:|
| Small (n<100) | eil51, berlin52 | FullGPU | 0.12 | 0.12 |

*Note: Best algorithm determined by lowest mean gap to known optimum.*
