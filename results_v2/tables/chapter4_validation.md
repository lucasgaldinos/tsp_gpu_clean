# Chapter 4 Validation Results

Generated: 2025-11-26 21:15:30

## Table 1: Summary Statistics by Problem

| Problem | Algorithm | Mean Cost | Std Dev | Gap (%) | Mean Time (s) | Speedup |
|:--------|:----------|----------:|--------:|--------:|--------------:|--------:|
| berlin52 | CPU | 7554.65 | 22.93 | 0.17 | 25.82 | 0.02× |
| berlin52 | FullGPU | 7542.00 | 0.00 | 0.00 | 0.25 | 2.27× |
| berlin52 | HybridNaive | 7547.31 | 14.45 | 0.07 | 0.57 | — |
| berlin52 | HybridOptimized | 7557.85 | 23.91 | 0.21 | 0.11 | 5.15× |
| bier127 | FullGPU | 118789.88 | 164.01 | 0.43 | 3.30 | 0.92× |
| bier127 | HybridNaive | 119075.19 | 239.82 | 0.67 | 3.03 | — |
| bier127 | HybridOptimized | 119148.58 | 232.98 | 0.73 | 0.56 | 5.43× |
| ch130 | FullGPU | 6162.92 | 19.18 | 0.87 | 4.39 | 0.76× |
| ch130 | HybridNaive | 6154.23 | 12.07 | 0.72 | 3.35 | — |
| ch130 | HybridOptimized | 6148.58 | 14.57 | 0.63 | 0.59 | 5.67× |
| ch150 | FullGPU | 6556.54 | 23.62 | 0.44 | 5.04 | 0.94× |
| ch150 | HybridNaive | 6580.15 | 13.91 | 0.80 | 4.75 | — |
| ch150 | HybridOptimized | 6576.19 | 10.44 | 0.74 | 0.79 | 6.00× |
| d198 | FullGPU | 15916.50 | 26.37 | 0.87 | 12.38 | 0.65× |
| d198 | HybridNaive | 15913.46 | 17.17 | 0.85 | 8.07 | — |
| d198 | HybridOptimized | 15914.19 | 23.91 | 0.85 | 1.52 | 5.31× |
| eil101 | FullGPU | 633.77 | 2.34 | 0.76 | 1.74 | 1.37× |
| eil101 | HybridNaive | 633.50 | 1.65 | 0.72 | 2.38 | — |
| eil101 | HybridOptimized | 633.81 | 1.30 | 0.76 | 0.39 | 6.05× |
| eil51 | CPU | 428.65 | 1.11 | 0.62 | 22.53 | 0.03× |
| eil51 | FullGPU | 428.04 | 0.71 | 0.48 | 0.41 | 1.50× |
| eil51 | HybridNaive | 428.69 | 1.03 | 0.63 | 0.61 | — |
| eil51 | HybridOptimized | 428.77 | 1.01 | 0.65 | 0.11 | 5.42× |
| eil76 | CPU | 541.46 | 1.74 | 0.64 | 84.61 | 0.02× |
| eil76 | FullGPU | 541.81 | 1.96 | 0.71 | 0.89 | 1.52× |
| eil76 | HybridNaive | 541.35 | 1.82 | 0.62 | 1.36 | — |
| eil76 | HybridOptimized | 541.31 | 1.61 | 0.61 | 0.21 | 6.59× |
| fl417 | FullGPU | 11986.38 | 13.01 | 1.06 | 87.88 | 1.05× |
| fl417 | HybridNaive | 12048.73 | 70.08 | 1.58 | 92.55 | — |
| fl417 | HybridOptimized | 12089.50 | 126.28 | 1.93 | 18.92 | 4.89× |
| kroA100 | CPU | 21382.88 | 62.04 | 0.47 | 185.08 | 0.01× |
| kroA100 | FullGPU | 21285.23 | 4.90 | 0.02 | 1.37 | 1.32× |
| kroA100 | HybridNaive | 21382.38 | 63.30 | 0.47 | 1.81 | — |
| kroA100 | HybridOptimized | 21364.23 | 65.23 | 0.39 | 0.30 | 6.07× |
| kroA150 | FullGPU | 26687.96 | 83.81 | 0.62 | 5.56 | 0.86× |
| kroA150 | HybridNaive | 26712.92 | 54.62 | 0.71 | 4.79 | — |
| kroA150 | HybridOptimized | 26711.04 | 55.94 | 0.71 | 0.81 | 5.88× |
| kroA200 | FullGPU | 29678.08 | 88.06 | 1.06 | 11.60 | 1.09× |
| kroA200 | HybridNaive | 29642.38 | 91.19 | 0.93 | 12.58 | — |
| kroA200 | HybridOptimized | 29630.19 | 91.54 | 0.89 | 2.02 | 6.23× |
| kroB100 | CPU | 22270.42 | 55.22 | 0.58 | 185.46 | 0.01× |
| kroB100 | FullGPU | 22186.96 | 44.63 | 0.21 | 1.77 | 1.08× |
| kroB100 | HybridNaive | 22272.27 | 59.98 | 0.59 | 1.91 | — |
| kroB100 | HybridOptimized | 22262.00 | 65.37 | 0.55 | 0.34 | 5.69× |
| kroB150 | FullGPU | 26283.65 | 72.01 | 0.59 | 6.97 | 0.63× |
| kroB150 | HybridNaive | 26325.77 | 46.80 | 0.75 | 4.42 | — |
| kroB150 | HybridOptimized | 26326.81 | 47.01 | 0.75 | 0.73 | 6.07× |
| kroC100 | CPU | 20845.58 | 59.12 | 0.47 | 175.80 | 0.01× |
| kroC100 | FullGPU | 20760.69 | 16.97 | 0.06 | 1.35 | 1.34× |
| kroC100 | HybridNaive | 20851.04 | 62.37 | 0.49 | 1.81 | — |
| kroC100 | HybridOptimized | 20832.08 | 55.32 | 0.40 | 0.29 | 6.14× |
| kroD100 | CPU | 21414.77 | 48.30 | 0.57 | 185.51 | 0.01× |
| kroD100 | FullGPU | 21381.08 | 45.97 | 0.41 | 1.97 | 0.93× |
| kroD100 | HybridNaive | 21431.46 | 51.86 | 0.65 | 1.84 | — |
| kroD100 | HybridOptimized | 21416.42 | 40.40 | 0.57 | 0.31 | 5.83× |
| kroE100 | CPU | 22202.31 | 56.02 | 0.61 | 189.61 | 0.01× |
| kroE100 | FullGPU | 22169.62 | 37.47 | 0.46 | 1.81 | 1.02× |
| kroE100 | HybridNaive | 22211.23 | 42.17 | 0.65 | 1.85 | — |
| kroE100 | HybridOptimized | 22189.00 | 61.97 | 0.55 | 0.30 | 6.20× |
| lin105 | FullGPU | 14381.27 | 8.13 | 0.02 | 1.50 | 1.28× |
| lin105 | HybridNaive | 14448.77 | 39.15 | 0.49 | 1.93 | — |
| lin105 | HybridOptimized | 14445.62 | 42.91 | 0.46 | 0.32 | 6.00× |
| lin318 | FullGPU | 42970.62 | 136.48 | 2.24 | 36.59 | 1.46× |
| lin318 | HybridNaive | 42847.58 | 389.56 | 1.95 | 53.48 | — |
| lin318 | HybridOptimized | 42941.15 | 434.66 | 2.17 | 10.02 | 5.34× |
| pcb442 | FullGPU | 52418.15 | 190.62 | 3.23 | 79.60 | 1.63× |
| pcb442 | HybridNaive | 52526.88 | 1051.61 | 3.44 | 129.46 | — |
| pcb442 | HybridOptimized | 52090.73 | 641.93 | 2.59 | 31.24 | 4.14× |
| pr1002 | FullGPU | 269776.77 | 743.93 | 4.14 | 758.40 | 0.92× |
| pr1002 | HybridNaive | 279875.50 | 3210.15 | 8.04 | 696.05 | — |
| pr1002 | HybridOptimized | 279184.62 | 2789.63 | 7.77 | 374.50 | 1.86× |
| pr107 | FullGPU | 44474.96 | 92.11 | 0.39 | 2.41 | 0.85× |
| pr107 | HybridNaive | 44559.50 | 116.68 | 0.58 | 2.05 | — |
| pr107 | HybridOptimized | 44580.19 | 103.38 | 0.63 | 0.36 | 5.77× |
| pr124 | FullGPU | 59031.77 | 8.85 | 0.00 | 1.85 | 1.70× |
| pr124 | HybridNaive | 59272.69 | 235.68 | 0.41 | 3.14 | — |
| pr124 | HybridOptimized | 59274.31 | 193.02 | 0.41 | 0.47 | 6.66× |
| pr136 | FullGPU | 97253.12 | 160.17 | 0.50 | 4.18 | 0.92× |
| pr136 | HybridNaive | 97503.31 | 180.84 | 0.76 | 3.86 | — |
| pr136 | HybridOptimized | 97509.38 | 183.14 | 0.76 | 0.67 | 5.75× |
| pr144 | FullGPU | 58537.00 | 0.00 | 0.00 | 2.87 | 1.32× |
| pr144 | HybridNaive | 58801.08 | 156.84 | 0.45 | 3.79 | — |
| pr144 | HybridOptimized | 58778.38 | 140.01 | 0.41 | 0.66 | 5.79× |
| pr152 | FullGPU | 73936.85 | 142.85 | 0.35 | 4.45 | 1.00× |
| pr152 | HybridNaive | 74144.27 | 136.40 | 0.63 | 4.46 | — |
| pr152 | HybridOptimized | 74143.15 | 181.90 | 0.63 | 0.75 | 5.95× |
| pr264 | FullGPU | 49307.08 | 99.94 | 0.35 | 21.48 | 0.94× |
| pr264 | HybridNaive | 49604.46 | 388.32 | 0.96 | 20.09 | — |
| pr264 | HybridOptimized | 49692.77 | 315.70 | 1.14 | 3.90 | 5.15× |
| pr299 | FullGPU | 49112.73 | 120.67 | 1.91 | 30.14 | 1.22× |
| pr299 | HybridNaive | 48865.96 | 390.18 | 1.40 | 36.75 | — |
| pr299 | HybridOptimized | 48891.54 | 343.61 | 1.45 | 6.70 | 5.48× |
| pr439 | FullGPU | 108506.50 | 200.93 | 1.20 | 85.94 | 1.29× |
| pr439 | HybridNaive | 110157.54 | 1528.18 | 2.74 | 111.28 | — |
| pr439 | HybridOptimized | 110099.23 | 1485.16 | 2.69 | 23.47 | 4.74× |
| pr76 | CPU | 108749.54 | 296.53 | 0.55 | 73.84 | 0.01× |
| pr76 | FullGPU | 108206.15 | 63.08 | 0.04 | 0.72 | 1.49× |
| pr76 | HybridNaive | 108610.50 | 325.63 | 0.42 | 1.08 | — |
| pr76 | HybridOptimized | 108676.46 | 334.86 | 0.48 | 0.19 | 5.78× |
| rat195 | FullGPU | 2373.50 | 9.03 | 2.17 | 11.59 | 1.26× |
| rat195 | HybridNaive | 2347.31 | 7.10 | 1.05 | 14.56 | — |
| rat195 | HybridOptimized | 2348.88 | 8.95 | 1.11 | 2.89 | 5.05× |
| rat783 | FullGPU | 9201.73 | 24.61 | 4.49 | 370.21 | 1.18× |
| rat783 | HybridNaive | 9452.96 | 215.29 | 7.35 | 436.79 | — |
| rat783 | HybridOptimized | 9579.92 | 165.35 | 8.79 | 207.28 | 2.11× |
| rat99 | CPU | 1219.58 | 3.03 | 0.71 | 188.04 | 0.01× |
| rat99 | FullGPU | 1215.27 | 3.14 | 0.35 | 1.65 | 1.19× |
| rat99 | HybridNaive | 1218.62 | 3.21 | 0.63 | 1.96 | — |
| rat99 | HybridOptimized | 1217.38 | 3.95 | 0.53 | 0.35 | 5.67× |
| rd100 | CPU | 7960.62 | 22.90 | 0.64 | 186.60 | 0.01× |
| rd100 | FullGPU | 7915.12 | 8.05 | 0.06 | 1.57 | 1.17× |
| rd100 | HybridNaive | 7949.88 | 22.47 | 0.50 | 1.84 | — |
| rd100 | HybridOptimized | 7959.12 | 19.36 | 0.62 | 0.29 | 6.24× |
| rd400 | FullGPU | 15727.65 | 50.48 | 2.92 | 73.33 | 1.40× |
| rd400 | HybridNaive | 15771.62 | 178.42 | 3.21 | 102.60 | — |
| rd400 | HybridOptimized | 15759.31 | 237.53 | 3.13 | 24.57 | 4.18× |
| st70 | CPU | 677.62 | 1.94 | 0.39 | 56.53 | 0.02× |
| st70 | FullGPU | 675.00 | 0.00 | 0.00 | 0.55 | 1.70× |
| st70 | HybridNaive | 677.23 | 2.17 | 0.33 | 0.94 | — |
| st70 | HybridOptimized | 677.38 | 1.78 | 0.35 | 0.17 | 5.49× |
| ts225 | FullGPU | 126643.00 | 0.00 | 0.00 | 11.80 | 0.86× |
| ts225 | HybridNaive | 127625.96 | 160.45 | 0.78 | 10.14 | — |
| ts225 | HybridOptimized | 127542.81 | 174.99 | 0.71 | 1.87 | 5.41× |
| u159 | FullGPU | 42094.15 | 44.05 | 0.03 | 3.99 | 1.32× |
| u159 | HybridNaive | 42399.77 | 117.13 | 0.76 | 5.26 | — |
| u159 | HybridOptimized | 42358.77 | 132.74 | 0.66 | 0.88 | 6.00× |

*Note: Speedup calculated relative to HybridNaive (— indicates baseline algorithm).*

## Table 2: Algorithm Performance Comparison (All Problems)

| Algorithm | Mean Gap (%) | Std Gap | Mean Time (s) | Std Time | Avg Speedup |
|:----------|-------------:|--------:|--------------:|---------:|------------:|
| CPU | 0.53 | 0.14 | 129.95 | 67.35 | — |
| FullGPU | 0.88 | 1.14 | 43.51 | 132.84 | — |
| HybridNaive | 1.28 | 1.68 | 47.08 | 130.25 | — |
| HybridOptimized | 1.30 | 1.78 | 18.94 | 67.33 | 2.49× |

## Table 3: Best Algorithm by Problem Size

| Size Category | Problems | Best Algorithm | Mean Gap (%) | Mean Time (s) |
|:--------------|:---------|:---------------|-------------:|--------------:|
| Small (n<100) | eil51, berlin52, st70, eil76, pr76, rat99 | FullGPU | 0.26 | 0.75 |
| Medium (100≤n<300) | kroA100, kroB100, kroC100, kroD100, kroE100, rd100, eil101, lin105, pr107, pr124, bier127, ch130, pr136, pr144, ch150, kroA150, kroB150, pr152, u159, rat195, d198, kroA200, ts225, pr264, pr299 | FullGPU | 0.50 | 6.28 |
| Large (n≥300) | lin318, rd400, fl417, pr439, pcb442, rat783, pr1002 | FullGPU | 2.76 | 213.13 |

*Note: Best algorithm determined by lowest mean gap to known optimum.*
