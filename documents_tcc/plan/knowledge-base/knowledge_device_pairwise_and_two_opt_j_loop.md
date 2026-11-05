---
title: "Knowledge: Distâncias TSPLIB95 em Device e Otimizações Exatas do Laço j do 2-Opt"
description: >
  Como replicar em CUDA device code a aritmética inteira TSPLIB95 de `src/distances/pairwise.py`
  sob NVRTC, por que `math.h` não existe no JIT do CuPy, e quais das três otimizações exatas do
  laço j do 2-opt (triângulo superior, carry-over, hoisting) se aplicam a cada kernel.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Criado a partir da execução das Tasks 10.15 e 10.16: enum de códigos de métrica,
          include path do NVRTC, e mapa de aplicabilidade das otimizações por kernel.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
  - "[walkthrough-10.15-10.16-pairwise-device-and-j-loop.md](../../../documentation/walkthroughs/walkthrough-10.15-10.16-pairwise-device-and-j-loop.md)"
  - "[pairwise_device.cuh](../../../src/distances/kernels/pairwise_device.cuh)"
  - "[pairwise.py](../../../src/distances/pairwise.py)"
  - "[gpu_helpers.py](../../../src/utils/gpu_helpers.py)"
  - "[two_opt.py](../../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - analysis/gpu
  - analysis/numerics
  - algorithm/tsp
  - algorithm/local-search
  - guide/cuda
  - guide/testing
  - notes/performance
  - review/exactness
---

# Knowledge: Distâncias TSPLIB95 em Device e Otimizações Exatas do Laço j do 2-Opt

## 1. NVRTC não é nvcc: não há `math.h`

O `cp.RawKernel` compila com o **NVRTC**, que não monta o include path do sistema. Um `#include <math.h>` falha com `catastrophic error: cannot open source file "math.h"`. As funções de device usadas (`floor`, `sqrt`, `fabs`, `ceil`, `cos`, `acos`) já vêm pré-declaradas pelo NVRTC, então o header simplesmente não inclui nada:

```cpp
// src/distances/kernels/pairwise_device.cuh:27-30
// No <math.h>: NVRTC (CuPy's JIT) ships no system headers but predeclares the
// device math functions (floor, sqrt, fabs, ceil, acos) this header uses.
```

O que **precisa** ser adicionado é o include path do header: `src/utils/gpu_helpers.py` (`_kernel_compile_options`) injeta `-I<src/distances/kernels>` em todo `RawKernel`, então qualquer kernel pode `#include "pairwise_device.cuh"`.

## 2. Contrato de códigos: enum C e tabela `dict` em python

`gpu_pairwise_distance` recebe um `int edge_type_code`, não uma string. As duas tabelas precisam concordar exatamente, ou o kernel calcula outra métrica silenciosamente:

| Métrica | Código |
| --- | --- |
| `EUC_2D` | 0 |
| `EUC_3D` | 1 |
| `MAN_2D` | 2 |
| `MAN_3D` | 3 |
| `MAX_2D` | 4 |
| `MAX_3D` | 5 |
| `CEIL_2D` | 6 |
| `ATT` | 7 |
| `GEO` | 8 |

A paridade é testada por regex contra o header (`tests/distances/test_unit_gpu_pairwise_device.py::test_edge_type_codes_match_device_header`), então adicionar uma métrica exige atualizar os dois lados e o teste falha se só um mudar.

## 3. Aritmética que não pode "melhorar"

Três detalhes decidem a igualdade bit-a-bit com `pairwise.py`:

1. `nint(x) = (int)floor(x + 0.5)`, nunca `round()` (que arredonda `.5` para o par mais próximo).
2. `CEIL_2D` usa `ceil(sqrt(...))` cru; passar por `nint` muda o resultado em raízes não inteiras.
3. `GEO` **trunca** a parte inteira do grau (`(int)x`), não arredonda: arredondar muda `burma14 d(1,3)` de `510` para `560`. A implementação local em `pairwise.compute_geographical` existe justamente porque a de `routing_data` arredonda.

## 4. Aplicabilidade das três otimizações exatas por kernel

As otimizações só são exatas quando a ordem de avaliação não muda o resultado:

| Kernel | Triângulo superior | Carry-over | Hoisting `(node_i, node_i1, d_ab)` |
| --- | --- | --- | --- |
| `two_opt_single.cu` | já existia (`j = i + 2`) | `node_j = node_j1` | `row_i`, `row_i1`, `d_ab` |
| `two_opt_batch.cu` | já existia | `node_j = node_j1` | idem |
| `two_opt.py` (3 variantes CPU) | já existia | `c = d` | `a`, `b`, `d_ab` |
| `two_opt_multi_move.cu` | `total_pairs = n(n-3)/2` | necessário reescrever para laços aninhados | idem |
| `cost_calculator.cu` | não se aplica (varredura 1-D) | `from_node = to_node` (chunk contíguo) | não se aplica |

`cost_calculator.cu` varria `for (i = tid; i < n; i += block_size)`, então as arestas de um mesmo thread não são adjacentes e o carry-over é impossível. A troca por **chunks contíguos** preserva o conjunto de arestas somadas e a redução `int64` (soma inteira é associativa), habilitando o carry-over sem alterar o custo.

## 5. O que muda quando o mapeamento par→thread muda

`two_opt_multi_move.cu` decodificava cada `pair_id` com um laço `O(n)` por par (`O(n^3)` no total). A versão row-major (`for i = tid; i < n - 2; i += block_size`, `j` interno) elimina a decodificação e habilita carry-over/hoisting, mantendo o mesmo conjunto de pares `n(n-3)/2`.

> [!WARNING]
> O mapeamento muda **qual thread** propõe **qual** movimento, e a seleção gulosa de movimentos
> disjuntos desempata por thread. Em berlin52, a rota de `BatchGPUMultiMoveTwoOpt` com
> `tour = arange(52)` muda de custo `8071` para `8301` — ambas são ótimos locais 2-opt
> (`best_delta == 0`). Um experimento com 6 populações aleatórias (4 rotas cada) mostrou resultados
> mistos, sem regressão sistemática. Ou seja: a reescrita é *exata na aritmética*, mas não é
> *idêntica na trajetória* do heurístico guloso.

## 6. Como provar a paridade

O teste monta uma matriz densa em CPU (`compute_distance_matrix`), lança um kernel de prova nos dois modos (`dist != nullptr` e `dist == nullptr`) e compara **com `np.array_equal`** — não com `allclose`: qualquer diferença de 1 unidade é uma falha.

```bash
.venv/bin/python -m pytest tests/distances/test_unit_gpu_pairwise_device.py -q
```
