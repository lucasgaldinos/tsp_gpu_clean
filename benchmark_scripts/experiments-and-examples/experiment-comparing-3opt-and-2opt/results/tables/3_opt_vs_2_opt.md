# 3 Opt Vs 2 Opt

Generated: 2026-11-24 16:11:20

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| att532 | 2-opt Batch (GPU) | 28083.40 | 112.44 | 1.44 | 2.30 | 1.00x |
| att532 | 2-opt Multi-Move (GPU) | 27954.90 | 22.74 | 0.97 | 0.51 | 4.51x |
| att532 | 3-opt Batch (GPU) | 28264.60 | 177.71 | 2.09 | 27.39 | 0.08x |
| att532 | 3-opt Multi-Move (GPU) | 27920.60 | 32.24 | 0.85 | 3.61 | 0.64x |
| att532 | ThreeOpt_Batch_GPU_less_candidates | 28115.30 | 138.95 | 1.55 | 5.56 | 0.41x |
| att532 | ThreeOpt_MultiMove_GPU_less_candidates | 27924.70 | 26.92 | 0.86 | 0.62 | 3.69x |
| d198 | 2-opt Batch (GPU) | 15915.80 | 12.99 | 0.86 | 0.13 | 1.00x |
| d198 | 2-opt Multi-Move (GPU) | 15906.20 | 18.94 | 0.80 | 0.04 | 3.34x |
| d198 | 3-opt Batch (GPU) | 15858.10 | 33.06 | 0.49 | 1.30 | 0.10x |
| d198 | 3-opt Multi-Move (GPU) | 15846.00 | 20.83 | 0.42 | 0.51 | 0.25x |
| d198 | ThreeOpt_Batch_GPU_less_candidates | 15883.50 | 39.18 | 0.66 | 0.21 | 0.61x |
| d198 | ThreeOpt_MultiMove_GPU_less_candidates | 15872.00 | 30.34 | 0.58 | 0.08 | 1.63x |
| gr666 | 2-opt Batch (GPU) | 304893.90 | 7450.14 | 3.58 | 2.45 | 1.00x |
| gr666 | 2-opt Multi-Move (GPU) | 297585.50 | 502.40 | 1.10 | 0.91 | 2.69x |
| gr666 | 3-opt Batch (GPU) | 303426.50 | 2142.32 | 3.08 | 41.92 | 0.06x |
| gr666 | 3-opt Multi-Move (GPU) | 297093.30 | 225.61 | 0.93 | 5.69 | 0.43x |
| gr666 | ThreeOpt_Batch_GPU_less_candidates | 301473.20 | 2982.57 | 2.42 | 9.48 | 0.26x |
| gr666 | ThreeOpt_MultiMove_GPU_less_candidates | 297145.20 | 132.20 | 0.95 | 1.42 | 1.72x |
| lin318 | 2-opt Batch (GPU) | 42527.30 | 121.76 | 1.19 | 0.59 | 1.00x |
| lin318 | 2-opt Multi-Move (GPU) | 42398.00 | 55.61 | 0.88 | 0.12 | 4.76x |
| lin318 | 3-opt Batch (GPU) | 42542.30 | 132.80 | 1.22 | 8.01 | 0.07x |
| lin318 | 3-opt Multi-Move (GPU) | 42393.40 | 39.61 | 0.87 | 1.45 | 0.40x |
| lin318 | ThreeOpt_Batch_GPU_less_candidates | 42550.10 | 120.45 | 1.24 | 1.69 | 0.35x |
| lin318 | ThreeOpt_MultiMove_GPU_less_candidates | 42400.70 | 33.11 | 0.88 | 0.28 | 2.08x |
| pcb442 | 2-opt Batch (GPU) | 51696.30 | 231.57 | 1.81 | 1.42 | 1.00x |
| pcb442 | 2-opt Multi-Move (GPU) | 51259.00 | 21.10 | 0.95 | 0.27 | 5.23x |
| pcb442 | 3-opt Batch (GPU) | 51499.90 | 197.49 | 1.42 | 13.90 | 0.10x |
| pcb442 | 3-opt Multi-Move (GPU) | 51219.90 | 43.80 | 0.87 | 2.29 | 0.62x |
| pcb442 | ThreeOpt_Batch_GPU_less_candidates | 51609.80 | 313.80 | 1.64 | 2.94 | 0.48x |
| pcb442 | ThreeOpt_MultiMove_GPU_less_candidates | 51238.70 | 54.02 | 0.91 | 0.41 | 3.43x |
| pr1002 | 2-opt Batch (GPU) | 270554.90 | 6212.52 | 4.44 | 5.32 | 1.00x |
| pr1002 | 2-opt Multi-Move (GPU) | 263363.10 | 1160.17 | 1.67 | 2.65 | 2.01x |
| pr1002 | 3-opt Batch (GPU) | 267395.40 | 1365.01 | 3.22 | 79.14 | 0.07x |
| pr1002 | 3-opt Multi-Move (GPU) | 261805.20 | 375.96 | 1.07 | 38.30 | 0.14x |
| pr1002 | ThreeOpt_Batch_GPU_less_candidates | 266513.30 | 1464.79 | 2.88 | 19.25 | 0.28x |
| pr1002 | ThreeOpt_MultiMove_GPU_less_candidates | 262051.80 | 602.05 | 1.16 | 9.11 | 0.58x |
| pr264 | 2-opt Batch (GPU) | 49520.60 | 92.65 | 0.78 | 0.25 | 1.00x |
| pr264 | 2-opt Multi-Move (GPU) | 49462.80 | 93.37 | 0.67 | 0.07 | 3.78x |
| pr264 | 3-opt Batch (GPU) | 49520.10 | 512.13 | 0.78 | 3.17 | 0.08x |
| pr264 | 3-opt Multi-Move (GPU) | 49345.90 | 132.16 | 0.43 | 0.82 | 0.31x |
| pr264 | ThreeOpt_Batch_GPU_less_candidates | 49382.50 | 90.16 | 0.50 | 0.38 | 0.66x |
| pr264 | ThreeOpt_MultiMove_GPU_less_candidates | 49450.80 | 92.10 | 0.64 | 0.12 | 2.05x |
| rat783 | 2-opt Batch (GPU) | 9197.30 | 355.75 | 4.44 | 4.14 | 1.00x |
| rat783 | 2-opt Multi-Move (GPU) | 8921.30 | 33.15 | 1.31 | 1.94 | 2.14x |
| rat783 | 3-opt Batch (GPU) | 9089.30 | 53.62 | 3.22 | 47.84 | 0.09x |
| rat783 | 3-opt Multi-Move (GPU) | 8889.30 | 8.77 | 0.95 | 10.07 | 0.41x |
| rat783 | ThreeOpt_Batch_GPU_less_candidates | 9060.50 | 48.56 | 2.89 | 11.73 | 0.35x |
| rat783 | ThreeOpt_MultiMove_GPU_less_candidates | 8892.40 | 2.12 | 0.98 | 2.09 | 1.98x |
| si535 | 2-opt Batch (GPU) | 48904.00 | 30.33 | 0.94 | 0.65 | 1.00x |
| si535 | 2-opt Multi-Move (GPU) | 48900.30 | 29.99 | 0.93 | 0.12 | 5.29x |
| si535 | 3-opt Batch (GPU) | 48857.00 | 36.31 | 0.84 | 12.60 | 0.05x |
| si535 | 3-opt Multi-Move (GPU) | 48737.10 | 105.51 | 0.59 | 3.47 | 0.19x |
| si535 | ThreeOpt_Batch_GPU_less_candidates | 48884.50 | 30.10 | 0.90 | 1.60 | 0.41x |
| si535 | ThreeOpt_MultiMove_GPU_less_candidates | 48912.00 | 12.07 | 0.95 | 0.56 | 1.15x |

*Note: Speedup calculated relative to 2-opt Batch (GPU).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| 2-opt Batch (GPU) | 2.16 | 1.55 | 1.91 | 1.82 | 1.00x |
| 2-opt Multi-Move (GPU) | 1.03 | 0.30 | 0.74 | 0.94 | 3.54x (n=9) |
| 3-opt Batch (GPU) | 1.82 | 1.11 | 26.14 | 25.83 | 0.08x (n=9) |
| 3-opt Multi-Move (GPU) | 0.77 | 0.23 | 7.35 | 11.98 | 0.34x (n=9) |
| ThreeOpt_Batch_GPU_less_candidates | 1.63 | 0.91 | 5.87 | 6.45 | 0.40x (n=9) |
| ThreeOpt_MultiMove_GPU_less_candidates | 0.88 | 0.17 | 1.63 | 2.88 | 1.80x (n=9) |

## Table 4: Average Phase Timings and Relative Share Breakdown

Phase execution times (ms), relative share of wall time (%), and CV (%) across repetitions.

| Problem | Algorithm | Wall (ms) | Init (%) | Sel (%) | CX (%) | Mut (%) | Eval (%) | LS (%) | Surv (%) | Host Sync (%) | Sum/Wall (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| att532 | 2-opt Batch (GPU) | 2296.1 | 0.2% (CV 49.5%) | 3.0% (CV 9.8%) | 2.9% (CV 5.3%) | 3.4% (CV 4.7%) | 1.0% (CV 4.9%) | 2.4% (CV 5.6%) | 3.9% (CV 4.8%) | 82.7% (CV 1.2%) | 99.5% |
| att532 | 2-opt Multi-Move (GPU) | 508.6 | 0.8% (CV 41.2%) | 3.1% (CV 8.5%) | 3.1% (CV 7.2%) | 3.7% (CV 15.1%) | 1.3% (CV 12.8%) | 2.6% (CV 14.2%) | 4.2% (CV 17.0%) | 80.5% (CV 2.8%) | 99.4% |
| att532 | 3-opt Batch (GPU) | 27389.3 | 0.0% (CV 50.6%) | 0.2% (CV 9.7%) | 0.2% (CV 7.7%) | 0.2% (CV 7.8%) | 0.1% (CV 7.4%) | 0.2% (CV 7.5%) | 0.2% (CV 8.1%) | 99.0% (CV 0.1%) | 100.0% |
| att532 | 3-opt Multi-Move (GPU) | 3606.4 | 0.1% (CV 39.4%) | 0.1% (CV 6.0%) | 0.1% (CV 5.8%) | 0.1% (CV 6.1%) | 0.1% (CV 8.3%) | 0.1% (CV 8.5%) | 0.1% (CV 5.6%) | 99.2% (CV 0.1%) | 100.0% |
| att532 | ThreeOpt_Batch_GPU_less_candidates | 5562.3 | 0.1% (CV 42.3%) | 0.8% (CV 12.5%) | 0.7% (CV 8.8%) | 0.8% (CV 8.6%) | 0.3% (CV 6.9%) | 0.7% (CV 8.4%) | 0.9% (CV 8.8%) | 95.5% (CV 0.4%) | 99.9% |
| att532 | ThreeOpt_MultiMove_GPU_less_candidates | 621.8 | 0.6% (CV 28.3%) | 0.7% (CV 8.8%) | 0.7% (CV 11.2%) | 0.8% (CV 12.4%) | 0.4% (CV 9.5%) | 0.7% (CV 15.9%) | 0.8% (CV 12.3%) | 95.2% (CV 0.5%) | 99.8% |
| d198 | 2-opt Batch (GPU) | 127.1 | 1.6% (CV 41.7%) | 5.5% (CV 14.0%) | 6.0% (CV 13.8%) | 6.6% (CV 12.4%) | 2.3% (CV 13.1%) | 4.8% (CV 13.9%) | 8.8% (CV 44.1%) | 63.4% (CV 2.3%) | 98.9% |
| d198 | 2-opt Multi-Move (GPU) | 38.1 | 4.2% (CV 30.5%) | 6.9% (CV 10.8%) | 7.4% (CV 12.1%) | 8.4% (CV 12.6%) | 3.7% (CV 10.5%) | 6.6% (CV 17.6%) | 9.3% (CV 11.6%) | 51.8% (CV 10.6%) | 98.4% |
| d198 | 3-opt Batch (GPU) | 1304.0 | 0.1% (CV 43.1%) | 0.3% (CV 11.3%) | 0.3% (CV 13.0%) | 0.3% (CV 13.8%) | 0.1% (CV 12.2%) | 0.3% (CV 12.2%) | 0.3% (CV 12.3%) | 98.3% (CV 0.2%) | 99.9% |
| d198 | 3-opt Multi-Move (GPU) | 506.6 | 0.4% (CV 34.1%) | 0.4% (CV 15.5%) | 0.4% (CV 16.7%) | 0.5% (CV 19.2%) | 0.3% (CV 17.0%) | 0.5% (CV 62.6%) | 0.4% (CV 12.4%) | 97.0% (CV 0.6%) | 99.9% |
| d198 | ThreeOpt_Batch_GPU_less_candidates | 209.5 | 1.0% (CV 44.5%) | 1.7% (CV 13.3%) | 1.6% (CV 11.2%) | 1.8% (CV 11.9%) | 0.8% (CV 18.5%) | 1.6% (CV 15.0%) | 3.4% (CV 125.8%) | 87.8% (CV 3.9%) | 99.6% |
| d198 | ThreeOpt_MultiMove_GPU_less_candidates | 77.8 | 1.9% (CV 29.7%) | 1.9% (CV 12.8%) | 1.9% (CV 9.6%) | 2.0% (CV 9.8%) | 1.2% (CV 8.3%) | 1.9% (CV 16.1%) | 2.1% (CV 12.8%) | 86.6% (CV 1.4%) | 99.5% |
| gr666 | 2-opt Batch (GPU) | 2447.8 | 0.2% (CV 28.9%) | 2.7% (CV 5.4%) | 2.6% (CV 4.0%) | 3.0% (CV 3.8%) | 0.9% (CV 3.0%) | 2.1% (CV 3.6%) | 3.4% (CV 3.9%) | 84.6% (CV 0.7%) | 99.5% |
| gr666 | 2-opt Multi-Move (GPU) | 909.2 | 0.5% (CV 58.4%) | 3.1% (CV 13.9%) | 3.0% (CV 9.7%) | 3.5% (CV 9.8%) | 1.1% (CV 6.3%) | 2.5% (CV 10.1%) | 3.9% (CV 10.0%) | 81.9% (CV 1.9%) | 99.4% |
| gr666 | 3-opt Batch (GPU) | 41918.0 | 0.0% (CV 56.9%) | 0.2% (CV 4.3%) | 0.1% (CV 3.9%) | 0.2% (CV 3.9%) | 0.0% (CV 3.0%) | 0.1% (CV 4.9%) | 0.2% (CV 4.1%) | 99.2% (CV 0.0%) | 100.0% |
| gr666 | 3-opt Multi-Move (GPU) | 5688.6 | 0.1% (CV 60.3%) | 0.1% (CV 5.0%) | 0.1% (CV 5.5%) | 0.1% (CV 6.5%) | 0.0% (CV 7.1%) | 0.1% (CV 7.4%) | 0.1% (CV 5.5%) | 99.3% (CV 0.0%) | 100.0% |
| gr666 | ThreeOpt_Batch_GPU_less_candidates | 9481.3 | 0.0% (CV 49.4%) | 0.7% (CV 6.1%) | 0.7% (CV 3.1%) | 0.8% (CV 2.6%) | 0.2% (CV 2.5%) | 0.7% (CV 2.8%) | 0.9% (CV 2.6%) | 95.9% (CV 0.1%) | 99.9% |
| gr666 | ThreeOpt_MultiMove_GPU_less_candidates | 1420.0 | 0.3% (CV 42.5%) | 0.6% (CV 7.5%) | 0.6% (CV 5.4%) | 0.7% (CV 5.3%) | 0.3% (CV 12.3%) | 0.7% (CV 6.8%) | 0.8% (CV 6.2%) | 95.8% (CV 0.2%) | 99.9% |
| lin318 | 2-opt Batch (GPU) | 585.6 | 0.4% (CV 60.1%) | 3.8% (CV 7.0%) | 4.2% (CV 7.1%) | 4.9% (CV 6.5%) | 1.5% (CV 9.0%) | 3.4% (CV 7.1%) | 5.4% (CV 6.8%) | 75.6% (CV 2.3%) | 99.3% |
| lin318 | 2-opt Multi-Move (GPU) | 122.9 | 1.4% (CV 34.8%) | 3.8% (CV 7.8%) | 3.9% (CV 6.3%) | 4.6% (CV 6.6%) | 1.8% (CV 6.4%) | 3.3% (CV 4.0%) | 5.0% (CV 6.3%) | 75.6% (CV 1.7%) | 99.2% |
| lin318 | 3-opt Batch (GPU) | 8008.7 | 0.0% (CV 38.7%) | 0.2% (CV 13.0%) | 0.2% (CV 8.9%) | 0.2% (CV 8.0%) | 0.1% (CV 6.0%) | 0.2% (CV 8.3%) | 0.2% (CV 8.9%) | 98.8% (CV 0.1%) | 100.0% |
| lin318 | 3-opt Multi-Move (GPU) | 1447.7 | 0.1% (CV 26.7%) | 0.2% (CV 4.7%) | 0.2% (CV 4.8%) | 0.2% (CV 3.6%) | 0.1% (CV 6.5%) | 0.2% (CV 4.5%) | 0.2% (CV 3.0%) | 98.8% (CV 0.0%) | 100.0% |
| lin318 | ThreeOpt_Batch_GPU_less_candidates | 1685.3 | 0.1% (CV 59.8%) | 1.0% (CV 9.5%) | 0.9% (CV 9.3%) | 1.0% (CV 9.8%) | 0.3% (CV 9.0%) | 0.9% (CV 9.5%) | 1.2% (CV 8.5%) | 94.3% (CV 0.5%) | 99.8% |
| lin318 | ThreeOpt_MultiMove_GPU_less_candidates | 281.2 | 0.9% (CV 47.6%) | 1.1% (CV 10.3%) | 1.1% (CV 12.7%) | 1.3% (CV 13.0%) | 0.6% (CV 16.3%) | 1.1% (CV 13.7%) | 1.3% (CV 13.1%) | 92.2% (CV 1.3%) | 99.7% |
| pcb442 | 2-opt Batch (GPU) | 1421.8 | 0.2% (CV 54.4%) | 3.8% (CV 7.1%) | 3.8% (CV 4.0%) | 4.5% (CV 3.7%) | 1.3% (CV 4.2%) | 3.1% (CV 7.0%) | 5.0% (CV 4.5%) | 77.6% (CV 1.4%) | 99.3% |
| pcb442 | 2-opt Multi-Move (GPU) | 272.1 | 1.0% (CV 24.1%) | 3.8% (CV 5.6%) | 4.0% (CV 4.3%) | 4.6% (CV 5.2%) | 1.5% (CV 4.5%) | 3.3% (CV 7.5%) | 5.0% (CV 4.4%) | 76.1% (CV 1.4%) | 99.2% |
| pcb442 | 3-opt Batch (GPU) | 13902.4 | 0.0% (CV 35.1%) | 0.3% (CV 8.7%) | 0.2% (CV 6.7%) | 0.3% (CV 6.6%) | 0.1% (CV 5.3%) | 0.2% (CV 6.4%) | 0.3% (CV 6.8%) | 98.5% (CV 0.1%) | 100.0% |
| pcb442 | 3-opt Multi-Move (GPU) | 2290.5 | 0.1% (CV 39.0%) | 0.2% (CV 3.8%) | 0.2% (CV 5.5%) | 0.2% (CV 4.9%) | 0.1% (CV 4.8%) | 0.2% (CV 4.9%) | 0.2% (CV 4.0%) | 98.8% (CV 0.1%) | 100.0% |
| pcb442 | ThreeOpt_Batch_GPU_less_candidates | 2936.1 | 0.1% (CV 68.2%) | 1.2% (CV 11.6%) | 1.1% (CV 10.8%) | 1.2% (CV 11.4%) | 0.4% (CV 10.6%) | 1.0% (CV 11.6%) | 1.3% (CV 11.3%) | 93.5% (CV 0.7%) | 99.8% |
| pcb442 | ThreeOpt_MultiMove_GPU_less_candidates | 414.0 | 0.7% (CV 31.6%) | 1.0% (CV 6.7%) | 1.0% (CV 8.4%) | 1.2% (CV 9.5%) | 0.5% (CV 10.2%) | 1.1% (CV 12.1%) | 1.3% (CV 10.0%) | 93.0% (CV 0.5%) | 99.8% |
| pr1002 | 2-opt Batch (GPU) | 5316.1 | 0.1% (CV 40.4%) | 1.5% (CV 6.9%) | 1.6% (CV 6.9%) | 1.9% (CV 7.2%) | 0.6% (CV 5.7%) | 1.3% (CV 7.7%) | 2.1% (CV 6.9%) | 90.6% (CV 0.7%) | 99.7% |
| pr1002 | 2-opt Multi-Move (GPU) | 2645.0 | 0.2% (CV 35.3%) | 2.0% (CV 7.1%) | 2.1% (CV 4.9%) | 2.5% (CV 4.9%) | 0.7% (CV 3.1%) | 1.7% (CV 4.7%) | 2.8% (CV 5.1%) | 87.7% (CV 0.7%) | 99.6% |
| pr1002 | 3-opt Batch (GPU) | 79139.8 | 0.0% (CV 40.8%) | 0.1% (CV 8.9%) | 0.1% (CV 7.3%) | 0.1% (CV 7.1%) | 0.0% (CV 5.7%) | 0.1% (CV 7.2%) | 0.1% (CV 7.1%) | 99.4% (CV 0.0%) | 100.0% |
| pr1002 | 3-opt Multi-Move (GPU) | 38301.6 | 0.0% (CV 68.4%) | 0.1% (CV 21.7%) | 0.1% (CV 15.9%) | 0.1% (CV 16.1%) | 0.0% (CV 6.2%) | 0.1% (CV 15.6%) | 0.1% (CV 16.2%) | 99.3% (CV 0.1%) | 100.0% |
| pr1002 | ThreeOpt_Batch_GPU_less_candidates | 19249.2 | 0.0% (CV 45.1%) | 0.6% (CV 5.1%) | 0.5% (CV 3.2%) | 0.6% (CV 3.1%) | 0.2% (CV 2.8%) | 0.5% (CV 3.3%) | 0.6% (CV 3.3%) | 96.9% (CV 0.1%) | 99.9% |
| pr1002 | ThreeOpt_MultiMove_GPU_less_candidates | 9106.2 | 0.1% (CV 67.2%) | 0.6% (CV 9.1%) | 0.5% (CV 3.8%) | 0.6% (CV 3.7%) | 0.2% (CV 1.9%) | 0.5% (CV 3.9%) | 0.7% (CV 3.7%) | 96.8% (CV 0.1%) | 99.9% |
| pr264 | 2-opt Batch (GPU) | 248.8 | 1.0% (CV 55.1%) | 4.9% (CV 10.6%) | 5.0% (CV 13.6%) | 5.6% (CV 10.3%) | 1.8% (CV 10.1%) | 4.0% (CV 10.0%) | 6.3% (CV 9.6%) | 70.4% (CV 4.0%) | 99.0% |
| pr264 | 2-opt Multi-Move (GPU) | 65.8 | 2.6% (CV 33.0%) | 4.9% (CV 11.2%) | 4.9% (CV 9.4%) | 5.3% (CV 9.3%) | 2.4% (CV 10.4%) | 3.8% (CV 10.7%) | 5.8% (CV 9.9%) | 69.2% (CV 3.7%) | 98.9% |
| pr264 | 3-opt Batch (GPU) | 3172.8 | 0.1% (CV 60.1%) | 0.2% (CV 34.7%) | 0.2% (CV 28.2%) | 0.3% (CV 28.0%) | 0.1% (CV 20.3%) | 0.2% (CV 28.9%) | 0.3% (CV 28.2%) | 98.6% (CV 0.4%) | 100.0% |
| pr264 | 3-opt Multi-Move (GPU) | 815.4 | 0.2% (CV 59.6%) | 0.2% (CV 8.0%) | 0.2% (CV 10.9%) | 0.2% (CV 8.9%) | 0.1% (CV 16.4%) | 0.2% (CV 9.7%) | 0.2% (CV 11.6%) | 98.6% (CV 0.2%) | 100.0% |
| pr264 | ThreeOpt_Batch_GPU_less_candidates | 377.4 | 0.5% (CV 49.9%) | 1.2% (CV 9.1%) | 1.2% (CV 7.5%) | 1.4% (CV 7.0%) | 0.6% (CV 12.7%) | 1.2% (CV 7.6%) | 1.5% (CV 6.1%) | 92.1% (CV 0.6%) | 99.7% |
| pr264 | ThreeOpt_MultiMove_GPU_less_candidates | 121.3 | 1.6% (CV 42.1%) | 1.3% (CV 8.8%) | 1.3% (CV 8.6%) | 1.4% (CV 13.6%) | 0.9% (CV 11.6%) | 1.3% (CV 11.8%) | 1.5% (CV 9.8%) | 90.4% (CV 1.0%) | 99.7% |
| rat783 | 2-opt Batch (GPU) | 4141.7 | 0.1% (CV 51.1%) | 1.7% (CV 5.7%) | 1.8% (CV 5.1%) | 2.2% (CV 5.2%) | 0.6% (CV 2.9%) | 1.5% (CV 5.5%) | 2.4% (CV 5.1%) | 89.3% (CV 0.6%) | 99.7% |
| rat783 | 2-opt Multi-Move (GPU) | 1939.3 | 0.2% (CV 24.7%) | 2.0% (CV 8.5%) | 2.1% (CV 4.4%) | 2.5% (CV 4.2%) | 0.8% (CV 3.8%) | 1.8% (CV 4.8%) | 2.8% (CV 4.7%) | 87.4% (CV 0.7%) | 99.6% |
| rat783 | 3-opt Batch (GPU) | 47836.1 | 0.0% (CV 33.3%) | 0.2% (CV 8.1%) | 0.1% (CV 7.7%) | 0.2% (CV 6.9%) | 0.0% (CV 5.5%) | 0.1% (CV 7.0%) | 0.2% (CV 7.1%) | 99.2% (CV 0.1%) | 100.0% |
| rat783 | 3-opt Multi-Move (GPU) | 10072.6 | 0.0% (CV 32.5%) | 0.1% (CV 30.8%) | 0.1% (CV 23.6%) | 0.1% (CV 23.1%) | 0.0% (CV 13.3%) | 0.1% (CV 22.2%) | 0.1% (CV 23.7%) | 99.3% (CV 0.1%) | 100.0% |
| rat783 | ThreeOpt_Batch_GPU_less_candidates | 11726.9 | 0.0% (CV 43.0%) | 0.7% (CV 7.7%) | 0.6% (CV 5.4%) | 0.7% (CV 4.9%) | 0.2% (CV 4.1%) | 0.6% (CV 5.5%) | 0.8% (CV 4.7%) | 96.2% (CV 0.2%) | 99.9% |
| rat783 | ThreeOpt_MultiMove_GPU_less_candidates | 2089.4 | 0.2% (CV 53.3%) | 0.6% (CV 15.1%) | 0.6% (CV 9.0%) | 0.7% (CV 9.4%) | 0.2% (CV 5.8%) | 0.6% (CV 9.9%) | 0.7% (CV 10.4%) | 96.3% (CV 0.3%) | 99.9% |
| si535 | 2-opt Batch (GPU) | 647.3 | 0.4% (CV 35.9%) | 2.3% (CV 6.6%) | 2.5% (CV 2.8%) | 3.0% (CV 1.7%) | 1.0% (CV 3.7%) | 2.1% (CV 3.0%) | 3.4% (CV 2.9%) | 84.8% (CV 0.6%) | 99.5% |
| si535 | 2-opt Multi-Move (GPU) | 122.5 | 2.8% (CV 53.8%) | 2.4% (CV 10.2%) | 2.5% (CV 8.2%) | 2.9% (CV 8.3%) | 1.4% (CV 12.0%) | 2.2% (CV 14.6%) | 3.1% (CV 7.7%) | 82.1% (CV 3.3%) | 99.5% |
| si535 | 3-opt Batch (GPU) | 12601.3 | 0.0% (CV 37.8%) | 0.1% (CV 3.2%) | 0.1% (CV 4.5%) | 0.1% (CV 5.0%) | 0.0% (CV 6.4%) | 0.1% (CV 8.0%) | 0.1% (CV 5.0%) | 99.5% (CV 0.0%) | 100.0% |
| si535 | 3-opt Multi-Move (GPU) | 3465.5 | 0.1% (CV 25.0%) | 0.1% (CV 6.5%) | 0.1% (CV 8.5%) | 0.1% (CV 7.3%) | 0.1% (CV 8.8%) | 0.1% (CV 10.6%) | 0.1% (CV 8.1%) | 99.4% (CV 0.1%) | 100.0% |
| si535 | ThreeOpt_Batch_GPU_less_candidates | 1595.3 | 0.2% (CV 41.8%) | 0.6% (CV 4.4%) | 0.6% (CV 4.4%) | 0.7% (CV 4.0%) | 0.3% (CV 7.9%) | 0.6% (CV 6.2%) | 0.8% (CV 5.0%) | 96.1% (CV 0.2%) | 99.9% |
| si535 | ThreeOpt_MultiMove_GPU_less_candidates | 563.3 | 0.7% (CV 32.8%) | 0.7% (CV 7.1%) | 0.7% (CV 7.5%) | 0.8% (CV 6.2%) | 0.4% (CV 8.7%) | 0.7% (CV 11.0%) | 0.9% (CV 7.1%) | 95.0% (CV 0.4%) | 99.8% |

> [!WARNING]
> **Runtime Floor Warning:** Executions with mean runtime < 100 ms exhibit elevated
> relative host synchronization and dispatch overhead (~0.5-1.0 ms fixed latency).
> Phase shares for sub-100ms runs should be interpreted with caution.