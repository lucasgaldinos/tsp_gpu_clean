---
title: "Walkthrough: Tasks 10.17 e 10.18 — Distâncias On-the-Fly e Tiling de Shared Memory"
description: >
  Histórico de execução das Tasks 10.17 (comutação automática para distâncias calculadas de
  coordenadas em `n > 10.000`) e 10.18 (janelamento da rota em shared memory para `n > 11.520`)
  nos três kernels CUDA e nas três estratégias python, com os testes de paridade que provam
  equivalência bit-a-bit entre os modos.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Registro da execução: assinaturas de kernel alteradas, troca de matriz densa por
          `gpu_pairwise_distance`, tiling cooperativo de 2048 posições, falha de escopo do
          macro `tile_start`, e evidência de paridade densa versus on-the-fly e tiled versus
          untiled.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[knowledge_on_the_fly_distances_and_shared_memory_tiling.md](../../documents_tcc/plan/knowledge-base/knowledge_on_the_fly_distances_and_shared_memory_tiling.md)"
  - "[pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh)"
  - "[cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py)"
tags:
  - notes/walkthrough
  - analysis/gpu
  - algorithm/tsp
  - algorithm/local-search
  - guide/cuda
  - guide/testing
  - notes/performance
  - analysis/memory
  - analysis/scalability
  - review/quality
  - analysis/tiling
  - notes/implementation
---

# Walkthrough: Tasks 10.17 e 10.18

## 1. Objetivo

- **Task 10.17:** quando `n > max_matrix_dimension` (10.000) e o problema é baseado em
  coordenadas, as estratégias passam `distances = nullptr`, `coords` (`float64` contíguo) e
  `edge_type_code`; os kernels calculam cada aresta com `gpu_pairwise_distance` e nunca alocam
  a matriz $(n, n)$.
- **Task 10.18:** quando `n > max_shared_tour_cities` (11.520), o buffer `s_tour[n]` é trocado
  por um tile de `2048 + 1` posições recarregado cooperativamente a cada janela do laço externo.

## 2. Correlação de contexto

`pairwise_device.cuh` já expunha o despachante com a aritmética TSPLIB95 bit-a-bit; faltava
apenas chamá-lo dos kernels de custo e de 2-opt. A conta de shared memory é
$4n + 12 \cdot \text{block size}$; com 256 threads o limite de 48 KB é atingido em
$n = 11{,}520$ ($46{,}080 + 3{,}072 = 49{,}152$ bytes).

## 3. Decisões de projeto

1. `_distance_operands(context, n)` é a única porta de decisão nas estratégias: retorna
   `(d_distances, d_coords, edge_code)` e, no modo coordenadas, devolve `0` para o ponteiro
   da matriz.
2. `ProblemContext.uses_on_the_fly_distances(n, threshold)` concentra a regra
   (`n > threshold`, `edge_type != EXPLICIT`, coordenadas presentes), com limiar
   configurável no contexto e na estratégia.
3. Os kernels recebem `int tour_tile_size` (`0` = rota inteira, `2048` = tile). O layout de
   shared memory coloca a rota primeiro para que os dois modos compartilhem uma alocação.
4. `TOUR_AT`/`TOUR_SET` isolam a leitura/escrita da rota: dentro da janela usa o tile, fora
   dela cai para a memória global; no modo tiled a escrita já é write-through.
5. A largura do tile é múltipla de qualquer `block_size` potência de 2 suportado, portanto o
   mapeamento linha→thread (`i % block_size`) é idêntico nos dois modos — condição para
   resultados bit-a-bit iguais.

## 4. Log de falhas e correções

| Observação que falhou | Correção |
| --- | --- |
| `nvrtc: error: identifier "tile_start" is undefined` nos 6 testes de 2-opt | `tile_start` estava declarado no inicializador do `for`; foi içado para `int tile_start = 0;` antes do laço, pois os macros da fase de aplicação o referenciam. |
| `assert dense_context._gpu_distances is not None` falhou | `BatchGPUTwoOpt`/`BatchGPUMultiMoveTwoOpt` cacheiam a matriz na própria estratégia, não no contexto; o teste passou a verificar `strategy._cached_distances_gpu` e `_cached_coords_gpu`. |

## 5. Arquivos afetados

| Arquivo | Mudança |
| --- | --- |
| `src/algorithms/strategies/evaluation/kernels/cost_calculator.cu` | `#include "pairwise_device.cuh"`, parâmetros `coords`/`edge_type_code`, soma via `gpu_pairwise_distance` |
| `src/algorithms/strategies/local_search/kernels/two_opt_batch.cu` | modo coordenadas + tiling (`TWO_OPT_TILE_SIZE`, `TOUR_AT`/`TOUR_SET`) |
| `src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu` | idem, mantendo a seleção gulosa de movimentos disjuntos |
| `src/protocols/problem_context.py` | `max_matrix_dimension` e `uses_on_the_fly_distances` |
| `src/algorithms/strategies/evaluation/cost_evaluation.py` | `max_matrix_dimension`, cache de coordenadas, `_distance_operands` |
| `src/algorithms/strategies/local_search/two_opt.py` | `max_matrix_dimension`/`max_shared_tour_cities`, caches e dimensão de shared memory |
| `tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py` | paridade densa versus coordenadas em EUC_2D e GEO |
| `tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py` | paridade densa/coordenadas, tiled/untiled e tour de 2.100 cidades |

## 6. Evidência de validação

```text
.venv/bin/python -m pytest tests/distances tests/algorithms/strategies -q
273 passed

.venv/bin/python -m mypy src tests
Success: no issues found in 116 source files

.venv/bin/python -m ruff check src tests
All checks passed!
```

## 7. Análise profunda

O ganho real não é a constante de acesso à memória: é a ordem de grandeza da alocação. A
matriz densa cresce como $O(n^2)$ e domina o orçamento de VRAM antes de qualquer outra
estrutura; remover esse termo é o que permite `pla33810` caber em uma GTX 1050. O tiling
resolve o problema simétrico no outro espaço de memória: shared memory é fixa em 48 KB, então
a rota precisa ser tratada como fluxo, não como buffer. Em ambos os casos a exatidão é
preservada porque a aritmética não muda — muda apenas de onde o inteiro vem.
