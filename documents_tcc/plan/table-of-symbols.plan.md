---
title: "Tabela de Símbolos — decisões, defeitos e tarefas"
description: >
  Registro único dos identificadores usados nos planos e walkthroughs do projeto:
  decisões de arquitetura (`DP-*`), defeitos (`BUG-*`) e tarefas (`TASK n.m`).
  Toda referência a um identificador em qualquer documento do repositório deve
  resolver para uma linha desta tabela.
created: 2026-11-02
status: review
author:
  - "[[Lucas Galdino]]"
  - "[[GitHub Copilot]]"
type: notes
scope: global
modifications:
  - date_modified: 2026-11-02
    modifications:
      - description: >
          Criação da tabela após a unificação dos três esquemas de identificadores
          conflitantes (`D1`–`D5` como decisões no plano, `D1`–`D8` como defeitos no
          walkthrough da suíte, e uma terceira ordem no checkpoint da Fase 9).
related_files:
  - "[plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md)"
  - "[test-suite-walkthrough.md](../../documentation/walkthroughs/test-suite-walkthrough.md)"
  - "[analysis-phase-9-gpu-capability-walkthrough.md](../../documentation/walkthroughs/analysis-phase-9-gpu-capability-walkthrough.md)"
  - "[analysis-phase-8-hermeticity-walkthrough.md](../../documentation/walkthroughs/analysis-phase-8-hermeticity-walkthrough.md)"
  - "[analysis-phase-7-residency-rng-walkthrough.md](../../documentation/walkthroughs/analysis-phase-7-residency-rng-walkthrough.md)"
  - "[analysis-phases-1-6-reorganization-walkthrough.md](../../documentation/walkthroughs/analysis-phases-1-6-reorganization-walkthrough.md)"
tags:
  - notes/planning
  - notes/reference
  - review/quality
  - analysis/architecture
  - guide/reorganization
  - notes/session
---

# Tabela de Símbolos

## Regras de namespace

| Prefixo | Significado | Escopo | Onde é definido |
| ------- | ----------- | ------ | --------------- |
| `DP-<n>` | Decisão de arquitetura do plano | Global (rodada de alinhamento inicial) | Esta tabela, seção [Decisões](#decisões-dp) |
| `DP<fase>-<n>` | Decisão de arquitetura tomada durante uma fase | Escopado à fase | Esta tabela, seção [Decisões](#decisões-dp) |
| `BUG-<n>` | Defeito em código de produção, com evidência `path:line` | Global, numeração estável | Esta tabela, seção [Defeitos](#defeitos-bug) |
| `TASK <fase>.<n>` | Tarefa do plano 5W2H | Escopado à fase | [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md) |

> [!IMPORTANT]
> Um identificador só pode ser usado em outro documento depois de registrado aqui.
> Se um identificador aparecer em um walkthrough e não existir nesta tabela, o
> walkthrough está incorreto.

## Decisões (`DP`)

| ID | Decisão | Evidência |
| -- | ------- | --------- |
| `DP-1` | Tipagem sem generics em cascata: `BackendModule` é protocolo não-genérico retornando `Array` (`NDArray[Any]`). | [plan#L65](./plan-reorganization-and-typing.plan.md#L65) |
| `DP-2` | Biblioteca em `src/benchmarking/{runner,analysis,database}/`; entrypoints em `benchmark_scripts/`. | [plan#L66](./plan-reorganization-and-typing.plan.md#L66) |
| `DP-3` | `pythonpath = ["src"]` estrito; constantes compartilhadas vivem na biblioteca. | [plan#L67](./plan-reorganization-and-typing.plan.md#L67) |
| `DP-4` | Checkpoints exportam `best_tour`, `raw_best_tours`, `problem_type`, `edge_type` e as 6 estratégias. | [plan#L68](./plan-reorganization-and-typing.plan.md#L68) |
| `DP-5` | `"hit_optimal"` exige `best_cost <= optimal_cost + 1e-6`; gap < 1% grava `"within_1pct_optimal"`. | [plan#L69](./plan-reorganization-and-typing.plan.md#L69) |
| `DP9-1` | API de GPU em `protocols.backend`; nenhum `__init__` faz probe na importação. | [plan#L630](./plan-reorganization-and-typing.plan.md#L630) |
| `DP9-2` | `GPUUnavailableError(RuntimeError)` cobre "CuPy ausente" e "sem dispositivo". | [plan#L631](./plan-reorganization-and-typing.plan.md#L631) |
| `DP9-3` | `cupy-cuda12x` obrigatório; imports de CuPy sempre tardios. | [plan#L632](./plan-reorganization-and-typing.plan.md#L632) |
| `DP9-4` | Monolítico e kernels em `algorithms/archive/`; exportações removidas de `algorithms.__all__`. | [plan#L633](./plan-reorganization-and-typing.plan.md#L633) |
| `DP9-5` | Variantes de archive permanecem no config; testes comportamentais não executam archive. | [plan#L634](./plan-reorganization-and-typing.plan.md#L634) |
| `DP9-6` | `evolve` rejeita `customers` vazio com `ValueError`. | [plan#L635](./plan-reorganization-and-typing.plan.md#L635) |
| `DP9-7` | Contrato compara estrutura de configs; valores numéricos podem diferir. | [plan#L636](./plan-reorganization-and-typing.plan.md#L636) |
| `DP10-1` | Identificadores padronizados: `DP-<n>` decisões, `DP<fase>-<n>` decisões de fase, `BUG-<n>` defeitos, `TASK <fase>.<n>` tarefas. | [plan#L828](./plan-reorganization-and-typing.plan.md#L828) |
| `DP10-2` | Variantes monolíticas saem do config ativo e vivem em `configs/algorithms.archive.json`, carregado sob demanda. | [plan#L829](./plan-reorganization-and-typing.plan.md#L829) |
| `DP10-3` | `FullGPUMonolithic` é o nome único da variante monolítica; `FullGPU` foi eliminado. | [plan#L830](./plan-reorganization-and-typing.plan.md#L830) |
| `DP10-4` | `benchmark_scripts/check_docs.py` é o gate standalone de docstrings, links e símbolos. | [plan#L831](./plan-reorganization-and-typing.plan.md#L831) |
| `DP10-5` | Checkpoints de schemas diferentes não são combinados na mesma pasta de resultados. | [plan#L832](./plan-reorganization-and-typing.plan.md#L832) |

## Defeitos (`BUG`)

Numeração herdada de `D1`–`D8` do [walkthrough da suíte](../walkthroughs/test-suite-walkthrough.md#6-defect-found-strict-xfails-not-fixed-in-this-change)
(`D<n>` → `BUG-<n>`, mesma ordem, sem reordenação).

### Abertos

| ID | Severidade | Afirmação | Evidência | Teste que fixa | Status |
| -- | ---------- | --------- | --------- | -------------- | ------ |
| `BUG-1` | Baixa | `Problem` é dataclass `frozen` sobre `ndarray`; `==` levanta `ValueError` e `hash()` levanta `TypeError`. | [problem.py#L30](../../src/data_models/problem.py#L30) | [test_unit_problem_validation.py#L124](../../tests/data_models/test_unit_problem_validation.py#L124) | xfail `strict` |
| `BUG-2` | Alta | `resolve_rng` aceita apenas `np.random.Generator`, mas `ProblemContext.rng` é `RandomState`; o RNG semeado do contexto é ignorado. | [rng.py#L56](../../src/utils/rng.py#L56), [problem_context.py#L166](../../src/protocols/problem_context.py#L166) | [test_unit_rng.py#L163](../../tests/utils/test_unit_rng.py#L163) | xfail `strict` |
| `BUG-3` | Alta | Ramo de variância zero fixa `mean_difference=0.0` e `ci_95_difference=(0.0, 0.0)` mesmo com diferença constante não nula. | [statistics.py#L220](../../src/benchmarking/analysis/statistics.py#L220) | [test_unit_statistics_comparison.py#L173](../../tests/benchmarking/test_unit_statistics_comparison.py#L173) | xfail `strict` |
| `BUG-4` | Alta | `is_valid_checkpoint` ignora o comprimento dos arrays `raw_*` que `can_resume_checkpoint` valida. | [checkpoint_io.py#L266](../../src/benchmarking/runner/checkpoint_io.py#L266) | [test_unit_checkpoint_io.py#L240](../../tests/benchmarking/test_unit_checkpoint_io.py#L240) | xfail `strict` |
| `BUG-5` | Média | `_mutation_bounds` protege depots apenas quando `n > 3`; rota `[depot, nó, depot]` pode trocar um endpoint. | [swap_mutation.py#L48](../../src/algorithms/strategies/mutation/swap_mutation.py#L48) | [test_unit_swap_mutation.py#L76](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L76) | xfail `strict` |
| `BUG-6` | Média | Mesma falha de `_mutation_bounds`, duplicada na mutação por inversão. | [inversion_mutation.py#L48](../../src/algorithms/strategies/mutation/inversion_mutation.py#L48) | [test_unit_inversion_mutation.py#L77](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py#L77) | xfail `strict` |
| `BUG-9` | Média | `validate_configs` lê `tournament_size` e `two_opt_iterations`, renomeados para `k` e `max_iterations` na TASK 6.1; o ramo nunca dispara. | [config_loader.py#L551](../../src/benchmarking/runner/config_loader.py#L551) | [test_unit_config_validation.py#L73](../../tests/benchmarking/test_unit_config_validation.py#L73) | resolvido na TASK 10.4 |
| `BUG-10` | Média | Todo relatório gerado recebe o título `# Chapter 4 Validation Results`, inclusive a comparação de estágios de busca local. | [report_generator.py#L104](../../src/benchmarking/analysis/report_generator.py#L104) | — | resolvido na TASK 10.4 |

### Resolvidos

| ID | Afirmação | Correção |
| -- | --------- | -------- |
| `BUG-7` | `GeneticAlgorithmFullGPUMonolithic.evolve` retornava `best_cost_history` em vez da chave de protocolo `convergence_history`. | Corrigido na TASK 9.6 ([full_gpu_monolithic.py#L248](../../src/algorithms/archive/full_gpu_monolithic.py#L248)); dois `xfail(strict=True)` removidos. |
| `BUG-8` | `evolve` aceitava `customers` vazio e retornava um tour `(0,)` com `generations_completed=1`. | Corrigido na TASK 9.5 ([genetic_algorithm.py#L452](../../src/algorithms/metaheuristics/genetic_algorithm.py#L452)); `xfail` removido. |
| `BUG-11` | `configs/algorithms.json` trazia `FullGPU` e `FullGPUMonolithic` idênticos (mesma classe, mesmos parâmetros, só a descrição diferia), gerando experimentos duplicados. | Consolidado em `FullGPUMonolithic`; variantes de archive movidas para `configs/algorithms.archive.json`. |
| `BUG-12` | `results_v3/selected_10/checkpoints/` misturava 39 checkpoints do esquema antigo (41 chaves) com 20 do novo (45 chaves), sob dois esquemas de semente distintos. | Os 20 checkpoints de esquema incompatível foram removidos; nova coleta fica pendente. |

## Migração de identificadores

| Antes | Depois | Ocorrências |
| ----- | ------ | ----------- |
| `D1`–`D5` no plano | `DP-1`–`DP-5` | 5 definições + referências |
| `DP9-1`–`DP9-7` | mantidos (namespace escopado por fase) | 23 |
| `D1`–`D6` no walkthrough da suíte | `BUG-1`–`BUG-6` | 6 linhas de tabela + 2 referências |
| `D7`, `D8` | `BUG-7`, `BUG-8` (resolvidos) | 8 |
