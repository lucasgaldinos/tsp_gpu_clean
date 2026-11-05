---
title: "Walkthrough: Task 10.20 — Listas de candidatos $k$-NN e parâmetro `candidate_k`"
description: >
  Histórico de execução da Task 10.20 (`DP10-20`): `compute_candidate_list` em blocos,
  caches `get_cpu_candidate_list` / `get_gpu_candidate_list` no `ProblemContext`, o mapa
  inverso de posições no 2-opt de CPU e no kernel `two_opt_batch.cu`, e a orientação por
  proximidade nas mutações e na inicialização.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Registro da execução: construção em blocos de $\mathcal{N}_k$, cache por `k`,
          caminho de candidatos nos três 2-opt de CPU e no kernel de GPU, mutação e
          inicialização guiadas por proximidade, e as evidências de validação.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[knowledge_candidate_lists_and_inverse_positions.md](../../documents_tcc/plan/knowledge-base/knowledge_candidate_lists_and_inverse_positions.md)"
  - "[matrix.py](../../src/distances/matrix.py)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
  - "[swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py)"
  - "[inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py)"
  - "[random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py)"
tags:
  - notes/walkthrough
  - analysis/algorithm
  - algorithm/tsp
  - guide/cuda
  - notes/performance
  - guide/testing
  - algorithm/candidate-lists
  - algorithm/k-nearest-neighbors
  - analysis/computational-complexity
  - benchmark/speedup
  - notes/implementation
  - review/optimization
---

# Walkthrough: Task 10.20

## 1. Objetivo

Centralizar a construção de $\mathcal{N}_k$ (os $k$ vizinhos mais próximos de cada nó) em
`ProblemContext` e expor um parâmetro opcional `candidate_k` que reduz a busca local, a
mutação e a inicialização de $\Theta(n^2)$ para $\Theta(n \cdot k)$, sem alterar nenhum
contrato de `Protocol`.

## 2. Correlação de contexto

| Artefato | Local | Observação relevante |
| --- | --- | --- |
| Cálculo de distâncias | [matrix.py](../../src/distances/matrix.py) | `_distance_block` já avaliava blocos de linhas; serviu de base para a construção em blocos. |
| Contexto | [problem_context.py](../../src/protocols/problem_context.py) | Já era o dono de todos os dados derivados (matrizes, coordenadas, demandas) com cache lazy. |
| 2-opt CPU | [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py) | Laço `j` exaustivo com hoisting e carry-over da Task 10.16. |
| 2-opt GPU | [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu) | Um bloco por tour, `s_tour` compartilhada, varredura `j` serial por thread. |
| Mutação | [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py) | Sorteio uniforme de dois índices interiores. |
| Inicialização | [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py) | `rng.permutation` uniforme. |

## 3. Análise de impacto

### Causa raiz

A varredura exaustiva avalia $\frac{n(n-3)}{2}$ pares por passada ($114{,}171{,}164$ em
`d15112`) e nenhuma estratégia tinha acesso a uma vizinhança restrita. Mutação uniforme
injeta arestas longas que a seleção descarta. A construção de $\mathcal{N}_k$ não existia.

### Decisões de projeto

| Componente | Antes | Depois | Justificativa |
| --- | --- | --- | --- |
| Construção de $\mathcal{N}_k$ | inexistente | `compute_candidate_list` em blocos de 512 linhas | Pico `O(chunk_size * n)`; nunca aloca $(n, n)$ a partir de coordenadas. |
| Acesso | inexistente | `get_cpu_candidate_list` / `get_gpu_candidate_list` com cache por `k` | Serviço compartilhado, sem 7º estágio de estratégia; `k` diferente ⇒ tabela diferente. |
| 2-opt CPU | laço `j` completo | `pos[tour[idx]] = idx` + `j = pos[N_k[a, r]]` | Reduz a passada para $\Theta(n \cdot k)$ mantendo a aritmética exata do delta. |
| 2-opt GPU | laço `j` completo | kernel com `candidates`, `candidate_k` e buffer `positions` | O tile de `s_tour` não cobre posições arbitrárias; o modo candidato lê `tour` global. |
| Mutação | dois índices uniformes | segundo índice = posição de um vizinho candidato | Recombina cidades próximas em vez de saltos longos. |
| Inicialização | `rng.permutation` | tour guloso aleatorizado (GRASP/RCL) sobre $\mathcal{N}_k$ | População inicial melhor com custo $O(n \cdot k)$. |

### Log de falhas e correções

| Tentativa / evento | Evidência lida | Correção / decisão | Resultado |
| --- | --- | --- | --- |
| Teste de igualdade exata $\mathcal{N}_k$ vs brute force em coordenadas inteiras | `Mismatched elements: 2 / 60`, índices `[4,4]` e `[5,4]` | Empate na $k$-ésima distância: `argpartition` escolhe um subconjunto arbitrário | Contrato reescrito como multiconjunto + "todo nó estritamente mais próximo presente" |
| `nvcc -c` reprovou os dois `.cu` | `cospi`/`sinpi` com `noexcept` incompatível em `mathcalls.h` | Incompatibilidade glibc × CUDA do ambiente, presente também em `two_opt_single.cu` | Verificado com `clang++ -x cuda --cuda-gpu-arch=sm_75 -fsyntax-only` (sem erros) e com os testes `-m gpu` reais |

## 4. Arquivos afetados

### python (7 arquivos)

1. [matrix.py](../../src/distances/matrix.py) — `compute_candidate_list` com máscara de auto-aresta, `argpartition` e `lexsort`.
2. [backend.py](../../src/protocols/backend.py) — `argpartition` e `lexsort` declarados no `BackendModule`.
3. [problem_context.py](../../src/protocols/problem_context.py) — `_cpu_candidate_lists`, `_gpu_candidate_lists`, os dois acessores.
4. [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py) — `candidate_k` nos três 2-opt de CPU e no `BatchGPUTwoOpt`; `_inverse_positions`, `_resolve_candidates`.
5. [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py) — `candidate_k` + `_proximity_index`.
6. [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py) — `candidate_k` + `_proximity_index`.
7. [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py) — `candidate_k` + `_greedy_candidate_tour`.

### CUDA (1 arquivo)

1. [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu) — parâmetros `candidates`, `candidate_k`, `positions`, ramo de candidatos e escrita direta na memória global.

### Testes (5 arquivos)

1. [test_unit_distance_matrix.py](../../tests/distances/test_unit_distance_matrix.py) — forma, dtype, brute force, blocos, ausência de matriz completa, ordenação.
2. [test_unit_problem_context.py](../../tests/protocols/test_unit_problem_context.py) — contrato do acessor, cache por `k`, guarda de backend.
3. [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py) — CPU com `candidate_k`, `k=0` igual ao exaustivo, GPU com candidatos.
4. [test_unit_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py) — permutação, depósitos e `_proximity_index`.
5. [test_unit_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py), [test_unit_random_initialization.py](../../tests/algorithms/strategies/initialization/test_unit_random_initialization.py) — mutação e inicialização guiadas.

## 5. Evidência de validação

```text
.venv/bin/python -m pytest tests -m "not slow" -q
805 passed, 2 deselected in 13.10s

.venv/bin/python -m pytest tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py -m gpu -v
7 passed, 15 deselected in 0.38s
  (inclui test_gpu_two_opt_candidate_list_path e o tiled de 2100 cidades)

.venv/bin/python -m mypy src tests
Success: no issues found in 116 source files

.venv/bin/python -m ruff check src tests
All checks passed!

.venv/bin/python -m benchmark_scripts.check_docs
stale docstrings: 0 / dead links: 0 / undefined symbols: 0
```

## 6. Handoff

- **Estado:** Task 10.20 concluída; `candidate_k` disponível em 2-opt CPU/GPU, mutação e inicialização.
- **Próximo item:** Task 10.21 (limpeza de documentação e scripts auxiliares).
- **Decisões abertas:** nenhuma; o modo candidato da GPU cobre apenas `BatchGPUTwoOpt` (o multi-move segue exaustivo).
