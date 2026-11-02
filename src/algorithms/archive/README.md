---
title: "Archive Algorithm Implementations"
description: >
  Historical monolithic and baseline genetic algorithm implementations
  preserved for empirical benchmarking and comparative analysis in Chapter 4 of the thesis.
created: 2026-10-30
modifications:
  - date_modified: 2026-10-30
    modifications:
      - description: >
          Reorganized archive directory: moved full_gpu_monolithic.py and kernels,
          removed five legacy unimportable modules, and added documentation.
related_files:
  - "[full_gpu_monolithic.py](./full_gpu_monolithic.py)"
  - "[ga_fujimoto_early_stop.cu](./kernels/ga_fujimoto_early_stop.cu)"
  - "[configs/algorithms.json](../../../configs/algorithms.json)"
tags:
  - algorithm/archive
  - algorithm/baseline
  - algorithm/genetic
  - algorithm/monolithic
  - algorithm/gpu
  - analysis/thesis
  - benchmark/baseline
  - guide/archive
  - notes/history
  - review/archive
---

# Archive Algorithm Implementations

This directory preserves historical monolithic and pre-modularization baseline implementations. These implementations are maintained exclusively for empirical comparison and benchmarking in Chapter 4 of the thesis and are intentionally isolated from the production modular strategy pipeline.

## Overview

- [full_gpu_monolithic.py](./full_gpu_monolithic.py): Monolithic Full-GPU Genetic Algorithm (`GeneticAlgorithmFullGPUMonolithic`) implementing the fused Fujimoto CUDA kernel pipeline with early stopping.
- [kernels/ga_fujimoto_early_stop.cu](./kernels/ga_fujimoto_early_stop.cu): CUDA C++ kernel containing fused initialization, selection, crossover, mutation, 2-opt local search, and evaluation.
- [kernels/ga_fujimoto.cu](./kernels/ga_fujimoto.cu): Original baseline kernel without early stopping.

## Usage

These archived models are not exported in the root `algorithms` package. They can be imported explicitly:

```python
from algorithms.archive import GeneticAlgorithmFullGPUMonolithic
```

Or accessed via benchmark configurations referencing `algorithms.archive.full_gpu_monolithic`.
