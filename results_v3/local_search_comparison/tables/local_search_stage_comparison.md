# Local Search Stage Comparison Benchmark

Generated: 2026-10-28 19:10:00

Compares solution quality, gap, and runtime across 2-opt placement stages:
- `post_mutation`: Traditional memetic GA (refinement after mutation each generation)
- `post_evaluation`: Refinement applied after fitness evaluation & survival
- `post_evolution`: Pure GA evolution, with 2-opt refinement applied to the full final population

| Problem | Algorithm | Stage | Mean Cost | Std Cost | Gap (%) | Mean Time (s) | Stop Reason |
|:---|:---|:---|---:|---:|---:|---:|:---|
| att48 | CPU | `post_mutation` | 10631.33 | 4.71 | 0.03% | 20.914 | max_generations |
| att48 | HybridNaive | `post_mutation` | 10628.00 | 0.00 | 0.00% | 0.443 | max_generations |
| att48 | HybridOptimized | `post_mutation` | 10628.00 | 0.00 | 0.00% | 0.061 | max_generations |
| att48 | FullGPUModular | `post_mutation` | 10695.67 | 9.03 | 0.64% | 0.034 | max_generations |
| att48 | CPU | `post_evaluation` | 10663.33 | 10.96 | 0.33% | 16.402 | max_generations |
| att48 | HybridNaive | `post_evaluation` | 10665.00 | 8.64 | 0.35% | 0.311 | max_generations |
| att48 | HybridOptimized | `post_evaluation` | 10665.00 | 8.64 | 0.35% | 0.037 | max_generations |
| att48 | FullGPUModular | `post_evaluation` | 10683.00 | 14.72 | 0.52% | 0.014 | max_generations |
| att48 | CPU | `post_evolution` | 11073.67 | 136.48 | 4.19% | 4.770 | max_generations |
| att48 | HybridNaive | `post_evolution` | 11073.67 | 136.48 | 4.19% | 1.059 | max_generations |
| att48 | HybridOptimized | `post_evolution` | 11073.67 | 136.48 | 4.19% | 0.852 | max_generations |
| att48 | FullGPUModular | `post_evolution` | 11125.67 | 49.26 | 4.68% | 0.150 | max_generations |
| bayg29 | CPU | `post_mutation` | 1611.67 | 2.36 | 0.10% | 3.026 | max_generations |
| bayg29 | HybridNaive | `post_mutation` | 1611.67 | 2.36 | 0.10% | 0.119 | max_generations |
| bayg29 | HybridOptimized | `post_mutation` | 1611.67 | 2.36 | 0.10% | 0.012 | max_generations |
| bayg29 | FullGPUModular | `post_mutation` | 1613.33 | 4.71 | 0.21% | 0.004 | max_generations |
| bayg29 | CPU | `post_evaluation` | 1610.00 | 0.00 | 0.00% | 2.794 | max_generations |
| bayg29 | HybridNaive | `post_evaluation` | 1610.00 | 0.00 | 0.00% | 0.112 | max_generations |
| bayg29 | HybridOptimized | `post_evaluation` | 1610.00 | 0.00 | 0.00% | 0.012 | max_generations |
| bayg29 | FullGPUModular | `post_evaluation` | 1610.00 | 0.00 | 0.00% | 0.006 | max_generations |
| bayg29 | CPU | `post_evolution` | 1637.33 | 17.31 | 1.70% | 1.028 | max_generations |
| bayg29 | HybridNaive | `post_evolution` | 1637.33 | 17.31 | 1.70% | 0.549 | max_generations |
| bayg29 | HybridOptimized | `post_evolution` | 1637.33 | 17.31 | 1.70% | 0.540 | max_generations |
| bayg29 | FullGPUModular | `post_evolution` | 1672.00 | 21.18 | 3.85% | 0.070 | max_generations |
| bays29 | CPU | `post_mutation` | 2024.67 | 1.89 | 0.23% | 2.651 | max_generations |
| bays29 | HybridNaive | `post_mutation` | 2024.67 | 1.89 | 0.23% | 0.111 | max_generations |
| bays29 | HybridOptimized | `post_mutation` | 2024.67 | 1.89 | 0.23% | 0.012 | max_generations |
| bays29 | FullGPUModular | `post_mutation` | 2022.00 | 2.83 | 0.10% | 0.005 | max_generations |
| bays29 | CPU | `post_evaluation` | 2022.67 | 2.49 | 0.13% | 2.673 | max_generations |
| bays29 | HybridNaive | `post_evaluation` | 2022.67 | 2.49 | 0.13% | 0.107 | max_generations |
| bays29 | HybridOptimized | `post_evaluation` | 2022.67 | 2.49 | 0.13% | 0.011 | max_generations |
| bays29 | FullGPUModular | `post_evaluation` | 2022.00 | 2.83 | 0.10% | 0.005 | max_generations |
| bays29 | CPU | `post_evolution` | 2045.33 | 24.24 | 1.25% | 0.913 | max_generations |
| bays29 | HybridNaive | `post_evolution` | 2045.33 | 24.24 | 1.25% | 0.437 | max_generations |
| bays29 | HybridOptimized | `post_evolution` | 2045.33 | 24.24 | 1.25% | 0.392 | max_generations |
| bays29 | FullGPUModular | `post_evolution` | 2057.33 | 20.24 | 1.85% | 0.082 | max_generations |
