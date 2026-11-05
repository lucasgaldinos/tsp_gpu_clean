---
title: "Listas de candidatos $k$-NN ($\\mathcal{N}_k$) e mapa inverso de posições"
description: >
  Referência reutilizável sobre a construção em blocos de $\\mathcal{N}_k$ em
  `distances/matrix.py`, o contrato `get_cpu_candidate_list` / `get_gpu_candidate_list`
  do `ProblemContext`, e o uso do mapa inverso `pos[$\\pi_i$] = i` para reduzir a busca
  local de $\\Theta(n^2)$ para $\\Theta(n \\cdot k)$ na CPU e no kernel CUDA.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Extração das decisões e dos invariantes da Task 10.20: construção em blocos,
          desempate na fronteira do `argpartition`, guarda de dimensão do tour e o caminho
          de candidatos do kernel `two_opt_batch.cu`.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
  - "[walkthrough-10.20-candidate-lists.md](../../documentation/walkthroughs/walkthrough-10.20-candidate-lists.md)"
  - "[matrix.py](../../src/distances/matrix.py)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
tags:
  - analysis/algorithm
  - algorithm/tsp
  - guide/cuda
  - notes/performance
  - analysis/gpu
  - guide/testing
---

# Listas de candidatos $k$-NN ($\mathcal{N}_k$) e mapa inverso de posições

## 1. Definição e contrato

$\mathcal{N}_k \in \{0, \dots, n-1\}^{n \times k_{\text{eff}}}$, com
$k_{\text{eff}} = \min(k, n - 1)$, ordenado por distância crescente e com desempate pelo
índice do nó. A auto-aresta nunca aparece: a linha $u$ exclui $u$.

- `distances/matrix.py::compute_candidate_list(...)` é a única implementação.
- `ProblemContext.get_cpu_candidate_list(k)` cacheia em `_cpu_candidate_lists: dict[int, np.ndarray]`.
- `ProblemContext.get_gpu_candidate_list(k)` cacheia em `_gpu_candidate_lists: dict[int, CupyArray]`
  e exige backend CuPy (`ValueError` no backend NumPy).

Cache por `k` é obrigatório: reconstruir $\mathcal{N}_k$ a cada mutação devolveria o custo
$\Theta(n^2)$ que a estrutura existe para eliminar.

## 2. Construção em blocos

```python
from distances.matrix import compute_candidate_list

candidates = compute_candidate_list(coordinates, "EUC_2D", k=20, chunk_size=512)
candidates.shape   # (n, min(k, n - 1))
candidates.dtype   # int32
# n = 15_112: pico de memória O(chunk_size * n) = ~32 MB, nunca (n, n)
```

Passo por bloco de `chunk_size` linhas:

1. `block = _distance_block(coordinates[start:end], coordinates, edge, xp)` (ou a fatia
   correspondente de `distances`), convertido para `int64`;
2. `block[rows, arange(start, end)] = np.iinfo(np.int64).max` mascara a auto-aresta;
3. `xp.argpartition(block, k_eff, axis=1)[:, :k_eff]` seleciona os $k_{\text{eff}}$ menores
   em tempo linear por linha;
4. `xp.lexsort((candidate_columns, candidate_distances), axis=1)` ordena
   (chave primária = última) por distância e desempata por índice de nó.

## 3. Fronteira de empate

`argpartition` é um particionamento linear, não uma ordenação total: quando vários nós
empatam exatamente na $k$-ésima distância, o subconjunto escolhido é arbitrário. O contrato
correto é, portanto, *multiconjunto* de distâncias, não igualdade elemento a elemento:

- toda distância selecionada é igual à da referência brute force;
- todo nó estritamente mais próximo que a $k$-ésima distância está presente.

Um teste que compare tabelas exatas em coordenadas inteiras falha por esse motivo
(ver `tests/distances/test_unit_distance_matrix.py`).

## 4. Mapa inverso de posições

$$
\text{pos}[\pi_{\text{idx}}] = \text{idx} \quad\Longleftrightarrow\quad \pi_{\text{pos}[u]} = u
$$

Construção em uma passada $O(n)$:

```python
positions = np.empty(len(tour), dtype=np.int32)
positions[tour] = np.arange(len(tour), dtype=np.int32)
```

Aplicação no 2-opt: para cada $i$, percorre $r \in [0, k_{\text{eff}})$, resolve
$j = \text{pos}[\mathcal{N}_k[\pi_i, r]]$ e avalia o delta apenas quando
$j > i + 1$ e $(i, j) \neq (0, n-1)$. Total por passada: $\Theta(n \cdot k)$ em vez de
$\Theta(n^2)$.

## 5. Guardas que evitam `IndexError`

- `_resolve_candidates(context, tour, candidate_k)` devolve `None` (varredura exaustiva)
  quando `candidate_k` é `None`/`<= 0` **ou** quando o tour contém ids de nó fora da tabela
  do problema (rotas de subconjunto, CVRP).
- Mutação: `_proximity_index` devolve `None` quando `route.max() >= route.size` e o chamador
  cai no sorteio uniforme do interior — a permutação nunca é quebrada.
- Kernel CUDA: `positions` é um buffer `num_tours * n` por bloco; o modo candidato ignora
  `s_tour` e lê `tour` global (o tile não cobre posições arbitrárias).

## 6. Pontos de verificação

```bash
.venv/bin/python -m pytest tests/distances/test_unit_distance_matrix.py -q
.venv/bin/python -m pytest tests/protocols/test_unit_problem_context.py -q
.venv/bin/python -m pytest tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py -m gpu -q
```
