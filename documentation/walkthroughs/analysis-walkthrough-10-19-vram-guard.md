---
title: "Walkthrough: Task 10.19 — Modelo de Capacidade de VRAM e `VRAMInsufficientError`"
description: >
  Histórico de execução da Task 10.19: estimativa de VRAM por população (`DP10-19`) em
  `ProblemContext`, validação ativa contra `cp.cuda.Device().mem_info`, guarda antes da
  alocação da matriz densa em `get_gpu_distances()` e os testes unitários com backend falso.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Registro da execução: fórmulas densa e on-the-fly, injeção de orçamento de VRAM,
          backend CUDA falso para o caminho de `mem_info`, e evidência de validação.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
  - "[exceptions.py](../../src/data_models/exceptions.py)"
  - "[test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py)"
tags:
  - notes/walkthrough
  - analysis/gpu
  - guide/cuda
  - guide/testing
  - notes/performance
  - algorithm/tsp
  - analysis/vram-management
  - analysis/memory-capacity
  - notes/safety-guards
  - benchmark/gpu-memory
  - notes/implementation
  - review/robustness
---

# Walkthrough: Task 10.19

## 1. Objetivo

- Substituir o status de código morto de `VRAMInsufficientError` por uma validação executável:
  estimar os bytes exigidos por uma população de `P` tours e recusar a operação antes que o
  driver CUDA aborte o processo.

## 2. Correlação de contexto

`ProblemContext.uses_on_the_fly_distances(n)` (`src/protocols/problem_context.py`) já existia
desde a Task 10.17 e decide entre matriz densa e avaliação a partir de coordenadas. Ela é a
fonte única da decisão de memória, portanto `estimate_vram_bytes` consulta a mesma função com
`n = self.dimension` em vez de repetir a condição (`n > max_matrix_dimension`,
`edge_type != "EXPLICIT"`, `coordinates is not None`).

`VRAMInsufficientError.__init__` (`src/data_models/exceptions.py`) já aceitava
`required_bytes` e `available_bytes` em bytes e os renderiza em GB no `__str__`; a Task não
alterou a exceção, apenas passou a usá-la.

## 3. Decisões de projeto

1. **`estimate_vram_bytes(pop_size)`** soma `12Pn` (três buffers `int32` de população/filhos),
   `24P` (três buffers `int64`/`float64` de fitness) e `8dn` (coordenadas/auxiliares). No modo
   denso acrescenta `4n²` da matriz `int32`; no modo on-the-fly omite o termo.
2. **`d` vem de `coordinates.shape[1]`** quando a matriz é 2-D; sem coordenadas o fator cai
   para `1`, produzindo `8n` em vez de zero — o buffer auxiliar existe de qualquer forma.
3. **`check_vram_capacity(pop_size, available_bytes=None)`** injeta o orçamento quando o
   chamador informa (testes, dry runs); caso contrário lê `xp.cuda.Device().mem_info`. No
   backend NumPy não existe VRAM a consultar, então a checagem degrada para retornar a
   estimativa.
4. **Comparação `required > available`**: um ajuste exato é aceito; rejeitá-lo superestimaria o
   requisito em um byte.
5. **`get_gpu_distances()` valida `4n²` antes de alocar.** O caminho denso é o único que
   materializa `(n, n)`; falhar ali evita o `CUDA_ERROR_OUT_OF_MEMORY` que mata o processo.

## 4. Log de falhas e correções

- `ruff` `TC006` em dois `cast()`: expressões de tipo passaram a ser literais de string
  (`cast("Any", self.xp)`, `cast("BackendModule", _FakeBackend(...))`).
- Primeira versão do teste usava `type("FakeBackend", (), {...})` inline; substituído por
  `_FakeDevice`, `_FakeCuda` e `_FakeBackend` tipados, mantendo `mypy` limpo.

## 5. Arquivos afetados

- [problem_context.py](../../src/protocols/problem_context.py) — `estimate_vram_bytes`,
  `check_vram_capacity`, `_coordinate_bytes`, `_query_free_vram_bytes`,
  `_raise_if_insufficient`, guarda em `get_gpu_distances()`.
- [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py)
  — 12 novos casos (fórmula densa/on-the-fly, problemas sem coordenadas, `pop_size` negativo,
  disparo e aceitação de orçamento, backend NumPy, leitura de `mem_info`).

## 6. Evidência de validação

```text
$ .venv/bin/python -m pytest tests/protocols -q
36 passed in 0.25s

$ .venv/bin/python -m pytest tests/protocols/test_unit_problem_context_gpu_guards.py -q
22 passed in 0.15s

$ .venv/bin/python -m mypy src tests
Success: no issues found in 116 source files

$ .venv/bin/python -m ruff check src tests
All checks passed!

$ .venv/bin/python -m benchmark_scripts.check_docs
stale docstrings: 0 / dead links: 0 / undefined symbols: 0
```

## 7. Análise

A estimativa é um limite superior barato: não modela fragmentação do alocador nem picos
transitórios de kernels, mas cobre os buffers persistentes que dominam `n` grande. Como o modo
on-the-fly remove `4n²`, a guarda passa a aceitar populações em instâncias gigantes exatamente
onde a matriz densa estouraria — coerente com `DP10-19`.
