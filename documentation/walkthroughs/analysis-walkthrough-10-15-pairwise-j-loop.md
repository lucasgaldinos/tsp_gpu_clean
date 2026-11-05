---
title: "Walkthrough: Tasks 10.15 e 10.16 — `pairwise_device.cuh` e Otimizações Exatas do Laço j"
description: >
  Histórico de execução das Tasks 10.15 (`pairwise_device.cuh`, include path no JIT do CuPy,
  teste de paridade CPU/GPU nas 9 métricas TSPLIB95) e 10.16 (triângulo superior, carry-over e
  hoisting nos três laços CPU e nos quatro kernels CUDA de 2-opt/custo).
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Registro da execução: artefatos criados, falha do `math.h` sob NVRTC, evidência de
          paridade por métrica, e desvio de trajetória observado em `two_opt_multi_move.cu`.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[knowledge_device_pairwise_and_two_opt_j_loop.md](../../documents_tcc/plan/knowledge-base/knowledge_device_pairwise_and_two_opt_j_loop.md)"
  - "[pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh)"
  - "[test_unit_gpu_pairwise_device.py](../../tests/distances/test_unit_gpu_pairwise_device.py)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - notes/walkthrough
  - analysis/gpu
  - analysis/numerics
  - algorithm/tsp
  - algorithm/local-search
  - guide/cuda
  - guide/testing
  - notes/performance
  - analysis/cuda-architecture
  - benchmark/execution-time
  - notes/implementation
  - review/verification
---

# Walkthrough: Tasks 10.15 e 10.16

## 1. Objetivo

- **Task 10.15:** expor a aritmética TSPLIB95 em device code (`pairwise_device.cuh`) com um
  despachante único `gpu_pairwise_distance`, e habilitar `#include "pairwise_device.cuh"` em
  qualquer `RawKernel` via `-I` no JIT.
- **Task 10.16:** aplicar as três otimizações exatas do laço `j` do 2-opt em CPU e nos kernels
  CUDA, sem alterar a aritmética.

## 2. Correlação de contexto

A leitura de `src/distances/pairwise.py` mostrou que `GEO` **trunca** o grau (`int(x)`) e que
`CEIL_2D` não passa por `nint`; `src/distances/matrix.py` confirma as mesmas escolhas via
`xp.trunc`/`xp.ceil`. O header replica exatamente essas decisões. `src/utils/gpu_helpers.py`
compilava `cp.RawKernel` sem `options`, portanto nenhum kernel conseguia incluir um header fora do
seu diretório.

## 3. Decisões de projeto

1. `gpu_pairwise_distance` retorna `0` para `u == v` e lê `dist[(size_t)u * n + v]` quando
   `dist != nullptr`; caso contrário, calcula de `coords` (`double`, 2 ou 3 por nó).
2. Os códigos vêm de `EDGE_TYPE_TO_CODE` (`src/distances/pairwise.py`), espelhados por
   `enum GpuEdgeTypeCode`; um teste de regex reprova qualquer divergência.
3. `cost_calculator.cu`: como a varredura era `i += block_size`, arestas de um mesmo thread não são
   adjacentes; o carry-over exigiu chunks contíguos. A soma inteira é associativa, então o custo
   final é idêntico.
4. `two_opt_multi_move.cu`: a decodificação `O(n)` por par (`O(n^3)` total) foi substituída pelo
   laço row-major com `j` interno, o que habilita carry-over e hoisting.
5. O kernel de prova dos testes vive em `tests/distances/kernels/probe_pairwise.cu` para que
   `load_kernel` exercite o include path recém-adicionado, em vez de compilar com `options`
   montado à mão pelo teste.

## 4. Log de falhas e correções

| Observação que falhou | Correção |
| --- | --- |
| `nvrtc: catastrophic error: cannot open source file "math.h"` em 9 de 10 testes | Remover `#include <math.h>` do header; o NVRTC pré-declara as funções de device. |
| `RUF022 __all__ is not sorted` em `pairwise.py` | Reordenar `EDGE_TYPE_TO_CODE` antes de `PI`. |
| `berlin52:batchmm` com custo `8301` contra `8071` do baseline | Investigado: ambas as rotas são ótimos locais 2-opt (`best_delta == 0`); a diferença vem do novo mapeamento par→thread do heurístico guloso, não da aritmética. Experimento com 6 populações aleatórias (4 rotas cada) mostrou resultados mistos, sem regressão sistemática. Registrado como mudança de trajetória em [§5 do knowledge](../../documents_tcc/plan/knowledge-base/knowledge_device_pairwise_and_two_opt_j_loop.md). |

## 5. Arquivos afetados

| Arquivo | Mudança |
| --- | --- |
| `src/distances/kernels/pairwise_device.cuh` *(novo)* | 9 métricas `__device__ __forceinline__` + `enum GpuEdgeTypeCode` + `gpu_pairwise_distance` |
| `src/utils/gpu_helpers.py` | `DISTANCE_KERNELS_DIR` e `-I` em `RawKernel(..., options=...)` |
| `src/distances/pairwise.py` | `EDGE_TYPE_TO_CODE` e `get_edge_type_code` |
| `src/algorithms/strategies/local_search/two_opt.py` | hoisting + carry-over nos três laços CPU |
| `src/algorithms/strategies/evaluation/kernels/cost_calculator.cu` | chunks contíguos com carry-over `from_node = to_node` |
| `src/algorithms/strategies/local_search/kernels/two_opt_single.cu` | row pointers + `d_ab` hoistados, `node_j = node_j1` |
| `src/algorithms/strategies/local_search/kernels/two_opt_batch.cu` | idem |
| `src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu` | laço row-major com `j` interno |
| `tests/distances/test_unit_gpu_pairwise_device.py` *(novo)* | paridade matriz/coordenadas e escalar nas 9 métricas |
| `tests/distances/kernels/probe_pairwise.cu` *(novo)* | kernel de prova dos dois modos |

## 6. Evidência de validação

```text
.venv/bin/python -m pytest tests/distances/test_unit_gpu_pairwise_device.py -q
10 passed

.venv/bin/python -m pytest tests/distances tests/algorithms/strategies -q
265 passed

.venv/bin/python -m mypy src tests
Success: no issues found in 116 source files

.venv/bin/python -m ruff check <arquivos tocados>
All checks passed!

.venv/bin/python -m benchmark_scripts.check_docs
exit=0
```

Paridade medida: `from_matrix == from_coords == compute_distance_matrix(...)` com
`np.array_equal` (bit-a-bit) para `EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`,
`CEIL_2D`, `ATT` e `GEO`, e cada entrada fora da diagonal igual à função escalar de `pairwise.py`.

## 7. Análise profunda

O ganho de `two_opt_multi_move.cu` não vem das três otimizações em si, mas da eliminação da
decodificação de `pair_id`: o laço `for (row = 0; row < n - 2; row++)` rodava uma vez por par,
tornando a fase de avaliação `O(n^3)`. O laço row-major avalia o mesmo conjunto `n(n-3)/2` de pares
em `O(n^2)`. As três otimizações reduzem constantes de memória; a mudança de estrutura remove uma
ordem de complexidade.

O preço é a trajetória do heurístico guloso: a seleção de movimentos disjuntos usa o melhor delta
*por thread*, e o novo mapeamento troca quais deltas cada thread vê. Por isso a rota final pode
divergir mesmo com aritmética idêntica.
