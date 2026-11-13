---
title: "Plano de Reorganização Arquitetural, Tipagem PEP 695/698 e Benchmark Modular"
description: >
  Plano executivo 5W2H para consolidação do pacote `src/benchmarking/`, separação
  dos scripts CLI em `benchmark_scripts/`, padronização da raiz de importação `src/`,
  enriquecimento dos checkpoints com rotas e metadados, e tipagem pragmática Python 3.12.
created: 2026-10-28
modifications:
  - date_modified: 2026-10-30
    modifications:
      - description: >
          Conclusão das Tarefas 8.10 (extração de rng.py), 8.11 (paridade de CLI e correção
          de bugs de semente/ingestão) e 8.9 (handlers polimórficos em DatabaseLoader).
      - description: >
          Registro da Fase 9 (Tarefas 9.1-9.12): API de capacidade de GPU no núcleo,
          aposentadoria de tests/gpu_probe.py, rejeição de customers vazio, isolamento do
          monolítico em archive/, contrato estrutural de configurações e limpeza de tipagem.
      - description: >
          Conclusão da Fase 9 (Tarefas 9.1-9.12): API de capacidade em protocols.backend,
          aposentadoria de tests/gpu_probe.py e tests de pythonpath/mypy_path, validação de
          customers em GeneticAlgorithm.evolve, isolamento de archive/ e resolução de BUG-7/BUG-8,
          contrato de configs, script check_archive_baseline e limpeza de tipagem.
      - description: >
          Registro da Fase 10 (Tarefas 10.1-10.5): tabela de símbolos e unificação dos
          identificadores DP-/BUG-, separação de `configs/algorithms.archive.json`,
          reparo dos DoDs 9.6/9.9 e das âncoras mortas, limpeza de docstrings e do
          validador de configuração, e criação do gate `benchmark_scripts/check_docs.py`.
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Reformulação da Fase 10: incorporação do backlog (`Task 10.6`–`Task 10.11`) em
          formato 5W2H dentro de `### Tarefas da fase 10`, registro das decisões já
          tomadas (`DP10-6` a `DP10-20`), decomposição da antiga tarefa monolítica de kernels
          on-the-fly em 5 sub-tarefas modulares (`Task 10.15` a `Task 10.19`), correções de
          revisão de código (`BUG-13` a `BUG-16`), higiene de documentação e lista de
          candidatos (`Task 10.12`–`Task 10.22`).
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Fechamento das Fases 10 e 11: `Tasks 10.12`, `10.14`, `10.20`–`10.24` e `11.1`–`11.4`
          marcadas `[x]` com Solution completo; coorte `results_v3/selected_10` regerada com
          `candidate_k = 20` e `strategy_params`; 12 links mortos repontados para os walkthroughs
          reais em `documentation/walkthroughs/`; `BUG-14`/`17`/`18`/`19`/`20` registrados como
          resolvidos na tabela de símbolos e Handoff atualizado para `done`.
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Correção do refinamento indevido da população inicial: `GeneticAlgorithm.evolve`
          aplicava `improve_batch` à geração 0 nos estágios `post_crossover` e `post_mutation`,
          inflando `initial_fitness` (berlin52: 25030 -> 14315) e cobrando 256 tours extras de
          2-opt em `mean_time`. A reprodução aleatória voltou a ser avaliada crua; apenas
          `post_evaluation` refina a população inicial, conforme documentado.
  - date_modified: 2026-11-10
    modifications:
      - description: >
          Auditoria de 5 eixos da Fase 11: refinamento in-place das decisões `DP11-7`, `DP11-19`,
          `DP11-20` e `DP11-21`, inclusão de `DP11-22` e dos defeitos `BUG-26` a `BUG-30`,
          reabertura das 10 tarefas com defeitos residuais (`Tasks 11.1`, `11.6`–`11.9`,
          `11.12`–`11.16`), criação das `Tasks 11.17`–`11.18` (`@time_strategy` assíncrono,
          sincronização de schema `BenchmarkRunInput` e patch canônico `GEO`), defeitos `BUG-31`
          e `BUG-32`, e inclusão dos blocos de Proof of Concept (PoC) executáveis em cada tarefa.
status: review
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
  - "[[Github Copilot]]"
type: plan
scope: global
related_files:
  - "[documents_tcc/plan/plan-roteiro.md](./plan-roteiro.md)"
  - "[pyproject.toml](../../pyproject.toml)"
  - "[configs/algorithms.json](../../configs/algorithms.json)"
  - "[src/loaders/problem_type_handlers.py](../../src/loaders/problem_type_handlers.py)"
  - "[src/utils/rng.py](../../src/utils/rng.py)"
tags:
  - analysis/architecture
  - analysis/typing
  - analysis/benchmark
  - benchmark/modular
  - algorithm/genetic-algorithm
  - algorithm/monolithic
  - guide/reorganization
  - review/quality
  - notes/planning
  - notes/session
  - guide/refactoring

---

# Plano de Reorganização Arquitetural, Tipagem PEP 695/698 e Benchmark Modular

Este plano operacional formaliza a execução das decisões de arquitetura e tipagem aprovadas na rodada de alinhamento técnico:

1. **Raiz de Importação Estrita (`src/`)**: Padronização exclusiva em `pythonpath = ["src"]`, removendo o prefixo `src.` de `configs/algorithms.json` e desacoplando testes de scripts CLI.
2. **Tipagem Pragmática PEP 695/698**: Uso de aliases nativos (`type Array = NDArray[Any]`), protocolos de estratégia limpos (não-genéricos, sem poluição de `ArrayT`), `@override` nas classes concretas e eliminação dos 68 erros de `mypy`.
3. **Enriquecimento de Checkpoints e Metadados**: Persistência de `best_tour`, `raw_best_tours`, `problem_type`, `edge_type` e nomes reais das 6 estratégias nos checkpoints JSON e no banco DuckDB.
4. **Refinamento do Early Stopping**: Distinção clara entre `"hit_optimal"` (ótimo verdadeiro a 0% de gap) e `"within_1pct_optimal"` (< 1% de gap).
5. **Desacoplamento Estatístico (`compare_algorithms`)**: Função pura independente de tamanho (`cpu_threshold <= 100`) para comparar qualquer lista de problemas ou categoria TSPLIB.
6. **Consolidação do Pacote `src/benchmarking/`**: Unificação dos diretórios duplicados `src/benchmarking_v2/` e `src/benchmarks_v2/` em `src/benchmarking/{runner,analysis,database}/`, movendo os pontos de entrada CLI para `benchmark_scripts/`.

---

## TOC

- [Plano de Reorganização Arquitetural, Tipagem PEP 695/698 e Benchmark Modular](#plano-de-reorganização-arquitetural-tipagem-pep-695698-e-benchmark-modular)
  + [TOC](#toc)
  + [Decisões Fixadas (Source of Truth)](#decisões-fixadas-source-of-truth)
  + [Registro de Tarefas 5W2H](#registro-de-tarefas-5w2h)
  + [Fase 7: Utilitários de Residência CPU/GPU e RNG, Variantes de 2-Opt, GPU Inversion Mutation e Integração Nativa com `tsp.duckdb`](#fase-7-utilitários-de-residência-cpugpu-e-rng-variantes-de-2-opt-gpu-inversion-mutation-e-integração-nativa-com-tspduckdb)
  + [Fase 8: Hermetismo de Caminhos, Separação Estrita de CLI/Scripts e Isolamento de Testes](#fase-8-hermetismo-de-caminhos-separação-estrita-de-cliscripts-e-isolamento-de-testes)
  + [Fase 9: Capacidade de GPU no Núcleo, Entrada Válida, Isolamento do Monolítico e Contrato de Configurações](#fase-9-capacidade-de-gpu-no-núcleo-entrada-válida-isolamento-do-monolítico-e-contrato-de-configurações)
    - [Decisões Fixadas da Fase 9](#decisões-fixadas-da-fase-9)
    - [Tarefas da fase 9](#tarefas-da-fase-9)
  + [Fase 10: Registro de Defeitos, Identificadores e Gate de Documentação](#fase-10-registro-de-defeitos-identificadores-e-gate-de-documentação)
    - [Decisões Fixadas da Fase 10](#decisões-fixadas-da-fase-10)
    - [Tarefas da fase 10](#tarefas-da-fase-10)
  + [Handoff](#handoff)

## Decisões Fixadas (Source of Truth)

- **DP-1 (Tipagem)**: Não criar parâmetros de tipo genéricos em cascata (`SelectionStrategy[ArrayT]`, `ProblemContext[ArrayT]`). O `BackendModule` é um Protocolo não-genérico retornando `Array` (`NDArray[Any]`) e `arccos`.
- **DP-2 (Layout de Diretórios)**: O código de biblioteca reside em `src/benchmarking/` com submódulos `runner/`, `analysis/`, `database/`. Os scripts executáveis de linha de comando residem em `benchmark_scripts/` na raiz do repositório.
- **DP-3 (Importações)**: `pythonpath = ["src"]` estrito. Constantes compartilhadas como `DEFAULT_SELECTED_TSP_FILES` vivem na biblioteca (`src/benchmarking/runner/config_loader.py`), nunca em scripts CLI.
- **DP-4 (Checkpoints)**: Checkpoints JSON exportam `best_tour: list[int]`, `raw_best_tours: list[list[int]]`, `problem_type: str`, `edge_type: str` e o dicionário de classes das 6 estratégias (`initialization`, `selection`, `crossover`, `mutation`, `local_search`, `evaluation`).
- **DP-5 (Parada Precoce)**: `"hit_optimal"` ocorre apenas para `best_cost <= optimal_cost + 1e-6`. Gaps abaixo de 1% que não atingiram o ótimo registram `"within_1pct_optimal"`.

---

## Registro de Tarefas 5W2H

- [x] **[REVIEWED]** **Task 1.1: Registrar Plano 5W2H no Repositório**
  + **What:** Criar `documents_tcc/plan/plan-reorganization-and-typing.md` e vincular no `plan-roteiro.md`.
  + **Why:** Garantir rastreabilidade contínua das tarefas de engenharia de software dentro da documentação da tese.
  + **Who:** Agente (`implementer`).
  + **Where:** [documents_tcc/plan/plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [documents_tcc/plan/plan-roteiro.md](./plan-roteiro.md).
  + **When:** Primeira ação da execução.
  + **How:** Criar o arquivo de plano com metadados e vincular na Tarefa 7 do roteiro.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Nenhuma; rastreabilidade interna.
  + **DoD:** Arquivo criado e vinculado.
  + **Solution:**
    - **Changed Files:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md)
    - **Summary:** Criado o plano 5W2H versionado na pasta de planos do repositório.
    - **DoD Compliance:** Plano registrado e pronto para acompanhamento.

- [x] **[REVIEWED]** **Task 1.2: Corrigir Raiz de Importação em `configs/algorithms.json` e `pyproject.toml`**
  + **What:** Remover o prefixo `src.` de `configs/algorithms.json` e alinhar `pyproject.toml` com a raiz de importação `src/`.
  + **Why:** Com `pythonpath = ["src"]`, `importlib.import_module("src.algorithms...")` falha com `ModuleNotFoundError: No module named 'src'`, quebrando 3 testes do pytest.
  + **Who:** Agente (`implementer`).
  + **Where:** [configs/algorithms.json](../../configs/algorithms.json#L11-L29), [pyproject.toml](../../pyproject.toml#L70-L105).
  + **When:** Imediatamente após Task 1.1.
  + **How:** Substituir `"src.algorithms."` por `"algorithms."` em `algorithms.json` e atualizar configurações do ruff/mypy.
  + **How much:** Trivial.
  + **Stakeholder Implications:** Restaura a passagem imediata dos testes de carregamento dinâmico.
  + **DoD:** `pytest` executa e passa os 18 testes existentes com 0 falhas.
  + **Solution:**
    - **Changed Files:** [configs/algorithms.json](../../configs/algorithms.json#L11-L29)
    - **Summary:** Removido prefixo `src.` de todos os `class_module` em `algorithms.json` para alinhar com `pythonpath = ["src"]`.
    - **DoD Compliance:** `pytest` executado com sucesso: 18 passed in 9.89s (0 falhas).

- [x] **[REVIEWED]** **Task 2.1: Tipagem Pragmática PEP 695/698 em Protocolos, Timer, Carregadores e Estratégias**
  + **What:** Simplificar `BackendModule` (adicionar `arccos`), anotar atributos de cache preguiçoso em `ProblemContext`, converter `phase_timer.py` para sintaxe PEP 695 `[**P, R]`, aplicar `@override` e casts de retorno nas estratégias, e corrigir tipagem em `pairwise.py`, `tsplib_loader.py`, `database_loader.py` e `full_gpu_monolithic.py`.
  + **Why:** Eliminar mais de 35 erros de `mypy` mantendo as assinaturas de estratégias modulares simples e sem explosão de generics.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/protocols/`, `src/utils/`, `src/distances/`, `src/loaders/`, `src/algorithms/`.
  + **When:** Sequencial após Task 1.2.
  + **How:** Tipar campos de cache como `np.ndarray | None`, adicionar `arccos`, atualizar assinaturas e decorar implementações com `@typing.override`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Módulos do núcleo do resolvedor completamente em conformidade com tipagem estrita do mypy.
  + **DoD:** `mypy src/protocols src/algorithms src/distances src/loaders src/utils` retorna 0 erros.
  + **Solution:**
    - **Changed Files:**
      + [backend.py](../../src/protocols/backend.py#L20-L135)
      + [problem_context.py](../../src/protocols/problem_context.py#L118-L202)
      + [algorithm_strategies.py](../../src/protocols/algorithm_strategies.py#L34-L568)
      + [phase_timer.py](../../src/utils/phase_timer.py#L11-L50)
      + [matrix.py](../../src/distances/matrix.py#L26-L32)
      + [pairwise.py](../../src/distances/pairwise.py#L515)
      + [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L323-L330)
      + [database_loader.py](../../src/loaders/database_loader.py#L23-L340)
      + [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L153-L160)
      + [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py) e [roulette_wheel_selection.py](../../src/algorithms/strategies/selection/roulette_wheel_selection.py)
      + [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L7-L88)
      + [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py)
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L324-L328)
    - **Summary:** Simplificado `BackendModule` para protocolo não-genérico com `arccos`, anotados campos de cache preguiçoso em `ProblemContext`, migrado `measure` em `phase_timer.py` para PEP 695 `[**P, R]`, corrigidos tipos de retorno e casts pontuais de rotas/fitness em estratégias e carregadores.
    - **DoD Compliance:** `mypy src/protocols src/algorithms src/distances src/loaders src/utils` executado com `Success: no issues found in 41 source files` (0 erros).

- [x] **[REVIEWED]** **Task 2.2: Refinar Parada Precoce e Exportar Rotas e Metadados nos Checkpoints**
  + **What:** Separar `"hit_optimal"` (ótimo estrito) de `"within_1pct_optimal"` (< 1% gap) em `genetic_algorithm.py` e `full_gpu_monolithic.py`; exportar `best_tour`, `raw_best_tours`, `problem_type`, `edge_type` e introspecção das 6 estratégias nos checkpoints JSON.
  + **Why:** Rotas calculadas em cada repetição eram descartadas antes do retorno do sumário; paradas em gaps de 0.31% marcavam indevidamente `hit_optimal`; e introspecção de classe caía sempre no default.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/metaheuristics/genetic_algorithm.py`, `src/algorithms/monolithic/full_gpu_monolithic.py`, `src/benchmarking_v2/algorithm_runner.py`, `src/benchmarking_v2/orchestration.py`, `tests/validation/test_early_stopping.py`.
  + **When:** Sequencial após Task 2.1.
  + **How:** Ajustar bloco de early stopping, incluir `best_tour`/`raw_best_tours`/`problem_type`/`edge_type` no dicionário de resultados e usar `type(strat).__name__` para capturar nomes das estratégias.
  + **How much:** Médio.
  + **Stakeholder Implications:** Pesquisador e orientador terão acesso aos vetores de solução exatos para plotagem e validação.
  + **DoD:** Checkpoints gerados contêm `best_tour`, `raw_best_tours`, `problem_type`, `edge_type` e estratégias corretas; suíte de testes passa.
  + **Solution:**
    - **Changed Files:**
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L529-L536)
      + [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L230-L241)
      + [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L242-L415)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L332-L545)
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L65)
    - **Summary:** Parada precoce agora reserva `"hit_optimal"` estritamente para `best_cost <= optimal_cost + 1e-6` e rotula `< 1.0%` como `"within_1pct_optimal"`. Checkpoints agora exportam o vetor da melhor rota (`best_tour`), lista de rotas por repetição (`raw_best_tours`), metadados `problem_type` e `edge_type`, além de introspecção dos 6 nomes reais de estratégias (`initialization`, `selection`, `crossover`, `mutation`, `local_search`, `evaluation`).
    - **DoD Compliance:** Testes validados em `test_early_stopping.py` e `test_benchmark_selected_problems.py` (5 passed), com verificação de arquivo JSON gerado confirmando presença de todas as chaves e rotas.

- [x] **[REVIEWED]** **Task 3.1: Desacoplar `compare_algorithms` em `aggregate_statistics.py`**
  + **What:** Implementar função pura `compare_algorithms` e `compare_algorithms_by_group`, eliminando a variável indefinida `small_problem_algos` e o limiar estático `cpu_threshold = 100`.
  + **Why:** Corrige 8 erros de `ruff` `F821` e 7 erros de `mypy`, permitindo análises pareadas sobre qualquer conjunto arbitrário de instâncias ou categorias.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/benchmarking_v2/aggregate_statistics.py`.
  + **When:** Sequencial após Task 2.2.
  + **How:** Extrair cálculo pareado (médias, desvios, speedups, Friedman e Nemenyi) para função pura e delegar a ela as estratificações por cobertura ou classe.
  + **How much:** Médio.
  + **Stakeholder Implications:** Elimina travamentos na análise estatística de múltiplos problemas.
  + **DoD:** `ruff check` e `mypy` passam sem erros em `aggregate_statistics.py`; testes executam com sucesso.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [aggregate_statistics.py](../../src/benchmarking/analysis/aggregate_statistics.py#L98-L358)
      + [**init**.py](../../src/benchmarking/__init__.py#L5-L44)
      + `test_benchmark_selected_problems.py` *(removido na TASK 8.7; substituído por `tests/benchmarking/test_integration_pipeline.py` e `tests/cli/`)*
    - **Resumo:** Extraída a função pura `compare_algorithms(...)` que valida instâncias com cobertura pareada completa, executa teste de Friedman, calcula distâncias críticas com Nemenyi e computa speedups geométricos em relação a um baseline configurável. Implementada a função `compare_algorithms_by_group(...)` para agrupamentos arbitrários (e.g. `edge_type`). A análise estratificada `perform_cross_problem_analysis_stratified(...)` agora delega ambos os estratos para `compare_algorithms(...)`, definindo `small_problem_algos` dinamicamente e eliminando os erros F821 e de tipagem em `all_algorithm_names`.
    - **Atendimento ao DoD:** `ruff check` e `mypy` passaram com 0 erros em `src/benchmarking_v2/aggregate_statistics.py`. Adicionados testes de unidade cobrindo casos normais, degenerados e agrupamento por metadados em `test_benchmark_selected_problems.py` (todos 6 testes aprovados).

- [x] **[REVIEWED]** **Task 4.1: Reorganizar Diretórios em `src/benchmarking/` e `benchmark_scripts/`**
  + **What:** Consolidar `src/benchmarking_v2/` e `src/benchmarks_v2/` em `src/benchmarking/{runner,analysis,database}/`, mover scripts CLI para `benchmark_scripts/`, ajustar testes e excluir arquivos/diretórios obsoletos.
  + **Why:** Eliminar duplicidade conceitual de pastas `benchmarks_v2` e `benchmarking_v2`, separar CLI da biblioteca e resolver todos os erros restantes do mypy (`fetchone`, tipos genéricos).
  + **Who:** Agente (`implementer`).
  + **Where:** `src/benchmarking/**`, `benchmark_scripts/**`, `tests/test_benchmark_selected_problems.py`.
  + **When:** Sequencial após Task 3.1.
  + **How:** Migrar arquivos para nova estrutura de pacotes, fornecer re-exportações públicas em `src/benchmarking/__init__.py`, atualizar chamadas em `tests/` e remover pastas antigas.
  + **How much:** Médio.
  + **Stakeholder Implications:** Estrutura de código limpa, profissional e livre de advertências.
  + **DoD:** `pytest`, `mypy src benchmark_scripts` e `ruff check src tests benchmark_scripts` rodam com 0 erros e 0 falhas.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [schema.py](../../src/benchmarking/database/schema.py#L1-L22)
      + [schema.sql](../../src/benchmarking/database/schema.sql#L1-L205)
      + [db_operations.py](../../src/benchmarking/database/db_operations.py#L1-L525)
      + [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L1-L365)
      + [validate_consistency.py](../../src/benchmarking/database/validate_consistency.py#L1-L310)
      + [database/**init**.py](../../src/benchmarking/database/__init__.py#L1-L25)
      + [statistics.py](../../src/benchmarking/analysis/statistics.py#L1-L635)
      + [stats_utils.py](../../src/benchmarking/analysis/stats_utils.py#L1-L160)
      + [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L1-L425)
      + [aggregate_statistics.py](../../src/benchmarking/analysis/aggregate_statistics.py#L1-L365)
      + [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L1-L530)
      + [analysis/**init**.py](../../src/benchmarking/analysis/__init__.py#L1-L35)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L1-L575)
      + [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L1-L570)
      + [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L1-L480)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L1-L620)
      + [runner/**init**.py](../../src/benchmarking/runner/__init__.py#L1-L65)
      + [benchmarking/**init**.py](../../src/benchmarking/__init__.py#L1-L75)
      + [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py#L1-L280)
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L1-L160)
      + [import_checkpoints.py](../../benchmark_scripts/import_checkpoints.py#L1-L25)
      + [validate_consistency.py](../../benchmark_scripts/validate_consistency.py#L1-L20)
      + [pyproject.toml](../../pyproject.toml#L90-L105)
      + `test_benchmark_selected_problems.py` *(removido na TASK 8.7)*
      + `test_modular_architecture.py` *(removido na reorganização da suíte; cobertura migrada para `tests/**`)*
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L13)
    - **Resumo:** Os diretórios `src/benchmarks_v2/` e `src/benchmarking_v2/` foram consolidados na biblioteca limpa `src/benchmarking/` com os subpacotes especializados `runner/`, `analysis/` e `database/`. Todas as APIs públicas foram re-exportadas em `src/benchmarking/__init__.py`. Os scripts executáveis de CLI foram movidos para a raiz em `benchmark_scripts/`. A lista padrão `DEFAULT_SELECTED_TSP_FILES` foi centralizada na biblioteca em `config_loader.py`. Foram sanados todos os retornos anuláveis de `cursor.fetchone()` no DuckDB e desacoplada a circularidade entre `runner` e `analysis` através de importações sob `if TYPE_CHECKING:` com `from __future__ import annotations`. Os diretórios legados foram totalmente removidos.
    - **Atendimento ao DoD:** Executado `pytest` com 19 testes aprovados (0 falhas); `mypy src benchmark_scripts` e `ruff check src tests benchmark_scripts` passaram com 0 erros em modo estrito.

- [x] **[REVIEWED]** **Task 5.1: Desacoplamento de Presets, Estágios Flexíveis de Busca Local e Benchmark Experimental**
  + **What:** Mover os 4 wrappers de preset para `src/algorithms/presets.py`, remover scripts redundantes em `src/algorithms/metaheuristics/`, desacoplar `GeneticAlgorithm` de hiperparâmetros de estratégias e eliminar a ramificação condicional rígida em `LocalSearchStage`. Corrigir a ordem de `post_evaluation` e refatorar `post_evolution` para aplicar 2-opt em toda a população final `(N, n_cities)`. Criar o script `benchmark_scripts/compare_local_search_stages.py` e executar o benchmark comparativo em `att48.tsp`, `bayg29.tsp` e `bays29.tsp`.
  + **Why:** O diretório `src/algorithms/metaheuristics/` continha arquivos que funcionavam meramente como presets ou re-exportações monolíticas em vez de metaheurísticas puras; hiperparâmetros como `two_opt_iterations` estavam acoplados na classe mãe; a busca local em `post_evolution` refinava apenas 1 rota (`best_tour`), mantendo 255 de 256 blocos CUDA ociosos na GPU; e faltava evidência empírica comparando a colocação de 2-opt após a avaliação de fitness versus a abordagem memética tradicional.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/presets.py`, `src/algorithms/metaheuristics/`, `src/protocols/`, `configs/algorithms.json`, `src/benchmarking/runner/`, `benchmark_scripts/compare_local_search_stages.py`, `benchmark_scripts/run_selected_problems.py`.
  + **When:** Sequencial após Task 4.1.
  + **How:** Criar `presets.py` com fábricas e classes depreciadas; limpar `metaheuristics/` mantendo apenas `genetic_algorithm.py` e `LocalSearchStage`; implementar pipeline flexível de etapas em `GeneticAlgorithm`; refatorar `improve_batch` + `evaluate_batch` na população completa `(N, n_cities)`; atualizar `algorithms.json` para expor `FullGPUModular` e `FullGPUMonolithic`; parametrizar `--local-search-stage` no benchmark runner; e executar 3 repetições por combinação.
  + **How much:** Médio-alto.
  + **Stakeholder Implications:** Arquitetura do resolvedor completamente ortogonal e declarativa; base empírica sólida e tabelas geradas para o Capítulo 4 da tese.
  + **DoD:** `pytest` aprova 19 testes (0 falhas); `mypy` e `ruff` passam com 0 erros; tabelas em Markdown e LaTeX geradas em `results_v3/local_search_comparison/tables/` reportando médias, desvios e gaps.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + `presets.py` *(criado na TASK 5.1 e removido na TASK 6.1)*
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L1-L600)
      + [metaheuristics/**init**.py](../../src/algorithms/metaheuristics/__init__.py#L1-L20)
      + [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L40)
      + [algorithm_strategies.py](../../src/protocols/algorithm_strategies.py#L40-L60)
      + [matrix.py](../../src/distances/matrix.py#L20-L30)
      + [algorithms.json](../../configs/algorithms.json#L1-L47)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L200-L225)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L325-L390)
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L50-L135)
      + [compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py#L1-L290)
      + `test_modular_architecture.py` *(removido na reorganização da suíte do Capítulo 8)*
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L11-L25)
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L10-L15)
      + [local_search_stage_comparison.md](../../results_v3/local_search_comparison/tables/local_search_stage_comparison.md#L1-L47)
    - **Resumo:** Os presets `GeneticAlgorithmCPU`, `GeneticAlgorithmHybridNaive`, `GeneticAlgorithmHybridOptimized` e `GeneticAlgorithmFullGPUModular` foram movidos para `src/algorithms/presets.py`, limpando `src/algorithms/metaheuristics/` para conter apenas a metaheurística `GeneticAlgorithm`. O método `GeneticAlgorithm.evolve()` foi refatorado para usar uma pipeline ordenada de operadores (`LocalSearchStage`), eliminando condicionais acopladas. Corrigida a fase inicial de `post_evaluation` para avaliar a aptidão antes do 2-opt. Em `post_evolution`, a busca local agora refina a população inteira `(N, n_cities)` via `improve_batch` e `evaluate_batch`, explorando paralelamente 256 regiões na GPU. Criado o script `compare_local_search_stages.py`, que executou o benchmark comparativo em `att48`, `bayg29` e `bays29`. Os resultados mostram que `post_evaluation` atinge 0.00% de gap em `bayg29` (superando `post_mutation`) e reduz o tempo de computação em até 50% em `att48`.
    - **Atendimento ao DoD:** `pytest` passou com 19/19 testes aprovados em 10.52s. `mypy src benchmark_scripts` e `ruff check src tests benchmark_scripts` passaram com 0 erros. Tabelas Markdown e LaTeX geradas com sucesso em `results_v3/local_search_comparison/tables/`.

- [x] **[REVIEWED]** **Task 6.1: Desacoplamento Total de Retrocompatibilidade, Pipelines de Etapas, Composição Declarativa e Eliminação de Shims**
  + **What:** Eliminar definitivamente `src/algorithms/presets.py` e todos os atalhos/shims de retrocompatibilidade; quebrar as 5 categorias de estratégias em arquivos atômicos por algoritmo de estratégia; eliminar os 10 arquivos `*_strategies.py`; refatorar `GeneticAlgorithm` para suportar AG puro (`local_search=None` por padrão) governado internamente por um pipeline de `GenerationStep`; refatorar `configs/algorithms.json` para composição puramente declarativa com `STRATEGY_REGISTRY`; simplificar scripts em `benchmark_scripts/` para python puro sem verbosidade de `argparse` integrados ao orquestrador central; e resolver as críticas 1 a 8 levantadas.
  + **Why:** O acoplamento com classes de preset e shims de retrocompatibilidade mantinha dívida técnica, duplicação de definições de estratégias e hiperparâmetros hardcoded (`legacy_kwargs`, `two_opt_iterations`, `tournament_size`), impedindo o uso do resolvedor como um Genetic Algorithm puro ou com outros operadores genéticos.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/protocols/types.py`, `src/utils/route_helpers.py`, `src/algorithms/strategies/`, `src/algorithms/metaheuristics/genetic_algorithm.py`, `configs/algorithms.json`, `src/benchmarking/runner/config_loader.py`, `src/benchmarking/runner/algorithm_runner.py`, `src/benchmarking/runner/orchestration.py`, `src/benchmarking/analysis/aggregate_statistics.py`, `benchmark_scripts/`, `tests/`.
  + **When:** Sequencial após Task 5.1.
  + **How:** Criar `src/protocols/types.py` e `src/utils/route_helpers.py`; decompor estratégias em arquivos independentes (`order_crossover.py`, `two_opt.py`, `swap_mutation.py`, `tournament_selection.py`, `cost_evaluation.py`) agrupando variantes CPU e GPU com resolução correta de kernels CUDA; remover `presets.py` e shims `*_strategies.py`; implementar pipeline de `GenerationStep` em `GeneticAlgorithm`; atualizar `algorithms.json` com mapeamento explícito de estratégias via `STRATEGY_REGISTRY`; converter scripts de benchmark para python puro invocando `run_comprehensive_benchmark`; atualizar testes e validar com `pytest`, `mypy` e `ruff`.
  + **How much:** Alto.
  + **Stakeholder Implications:** Arquitetura do resolvedor completamente ortogonal, declarativa, livre de retrocompatibilidade obsoleta e com 100% de conformidade com PEP 695/698.
  + **DoD:** `pytest` aprova todos os 19 testes; `mypy src benchmark_scripts tests` e `ruff check src tests benchmark_scripts` passam com 0 erros; `compare_local_search_stages.py` executa via orquestrador e gera tabelas atualizadas.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [types.py](../../src/protocols/types.py#L1-L32)
      + [route_helpers.py](../../src/utils/route_helpers.py#L1-L37)
      + [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L1-L86)
      + [pmx_crossover.py](../../src/algorithms/strategies/crossover/pmx_crossover.py#L1-L80)
      + [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L1-L157)
      + [no_op.py](../../src/algorithms/strategies/local_search/no_op.py#L1-L38)
      + [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L1-L91)
      + [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L1-L89)
      + [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L1-L87)
      + [roulette_wheel_selection.py](../../src/algorithms/strategies/selection/roulette_wheel_selection.py#L1-L90)
      + [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L1-L87)
      + [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L1-L40)
      + [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L1-L75)
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L1-L600)
      + [algorithms.json](../../configs/algorithms.json#L1-L135)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L140-L245)
      + [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L155-L210)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L320-L410)
      + [aggregate_statistics.py](../../src/benchmarking/analysis/aggregate_statistics.py#L345-L365)
      + [compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py#L1-L100)
      + [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py#L1-L160)
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L1-L120)
      + `test_modular_architecture.py` *(removido na reorganização da suíte)*
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L1-L50)
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L1-L70)
      + `test_benchmark_selected_problems.py` *(removido na TASK 8.7)*
    - **Resumo:** Implementação completa da resolução dos 8 pontos de crítica: (1) `LocalSearchStage` e tipos de domínio migrados para `src/protocols/types.py` e helpers de lote em `src/utils/route_helpers.py`; (2) Decomposição de todas as estratégias em arquivos individuais por algoritmo, preservando caminhos adjacentes a `kernels/` para CUDA, com eliminação de 10 shims legados `*_strategies.py`; (3) Remoção definitiva de `src/algorithms/presets.py`, de `legacy_kwargs`, `tournament_size`, `two_opt_iterations` e condicionais lazy de `GeneticAlgorithm`, estabelecendo suporte nativo a AG Puro (`local_search=None`) via pipeline composável de `GenerationStep`; (4) `configs/algorithms.json` totalmente declarativo com injeção de estratégias via `STRATEGY_REGISTRY`; (5) Scripts em `benchmark_scripts/` reescritos em python puro sem `argparse`, orquestrando diretamente o pipeline de benchmark; (6) Validações e testes adaptados, com execução bem-sucedida do benchmark de comparação dos 3 estágios (`post_mutation`, `post_evaluation`, `post_evolution`) através do orquestrador.
    - **Atendimento ao DoD:** 19/19 testes aprovados no `pytest` (29.18s); `mypy` e `ruff` com 0 erros/avisos em todo o projeto; benchmark de comparação executado com sucesso e tabelas salvas em `results_v3/local_search_comparison/tables/`.

---

## Fase 7: Utilitários de Residência CPU/GPU e RNG, Variantes de 2-Opt, GPU Inversion Mutation e Integração Nativa com `tsp.duckdb`

- [x] **[COMPLETED]** **Task 7.1: Documentação Exaustiva de Estágios de Busca Local e Pipelines Customizados (`GenerationStep`)**
  + **What:** Documentar em `documentation/guide-local-search-pipeline-stages.md` todos os 5 estágios nativos de `LocalSearchStage` (`none`, `post_crossover`, `post_mutation`, `post_evaluation`, `post_evolution`) e 4 padrões de extensão via `GenerationStep` (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep` e pipeline multi-estágio).
  + **Why:** Tornar explícito como configurar ou estender o posicionamento da busca local sem acoplamento em `GeneticAlgorithm`, fornecendo contratos exatos de `GenerationState` e templates executáveis.
  + **Who:** Agente (`implementer`).
  + **Where:** [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L1-L485), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L70-L88).
  + **When:** Independente; primeiro item da Fase 7.
  + **How:** Criar o guia Markdown seguindo `.agents/rules/markdown.instructions.md` com diagramas Mermaid, tabelas de invariantes de estado e código completo copiável para cada caso.
  + **How much:** Médio.
  + **Stakeholder Implications:** Qualquer desenvolvedor ou modelo de linguagem consegue implementar novos estágios de busca local seguindo o contrato `GenerationStep.run(state, timer)`.
  + **DoD:** Arquivo `documentation/guide-local-search-pipeline-stages.md` criado com frontmatter YAML válido, tags hierárquicas e exemplos completos para todas as possibilidades.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L1-L485)
    - **Resumo:** Criado guia técnico detalhado cobrindo os contratos de `GenerationState` e `GenerationStep`, os 5 modos nativos de `LocalSearchStage`, 4 implementações completas de steps customizados (pós-inicialização, apenas elite, periódico e multi-estágio), configuração declarativa em `configs/algorithms.json` e comando de verificação.
    - **Atendimento ao DoD:** Documento gerado em `./documentation/guide-local-search-pipeline-stages.md` em conformidade com os padrões de Markdown do repositório.

- [x] **[COMPLETED]** **Task 7.2: Centralização dos Utilitários de Residência de Rotas (CPU/GPU) e Resolução de RNG (`SeedLike`)**
  + **What:** Extrair a verificação repetida de arrays CuPy/NumPy (`is_gpu_array`, `to_cpu_batch`, `restore_routes_shape(..., to_gpu=...)`) e a resolução de geradores pseudoaleatórios (`SeedLike`, `make_rng`, `resolve_rng`) para `src/utils/route_helpers.py` e `src/protocols/types.py`, refatorando todas as estratégias em `src/algorithms/strategies/` para utilizá-los.
  + **Why:** Atualmente, 8 arquivos de estratégia repetem blocos `try: import cupy` e conversões `isinstance(batch, cp.ndarray)` (~12 locais), além de 11 blocos idênticos `if context is not None and hasattr(context, "rng")...` nas estratégias estocásticas.
  + **Who:** Agente (`implementer`).
  + **Where:** [types.py](../../src/protocols/types.py#L10-L54), [route_helpers.py](../../src/utils/route_helpers.py#L1-L61), [utils/**init**.py](../../src/utils/__init__.py#L1-L13), e módulos em `src/algorithms/strategies/`.
  + **When:** Antes de Task 7.3 e Task 7.4.
  + **How:** Adicionar `type SeedLike = int | Generator | Callable[[], Generator] | None` em `src/protocols/types.py`; implementar `is_gpu_array`, `to_cpu_batch`, `restore_routes_shape(..., to_gpu: bool = False)`, `make_rng` e `resolve_rng` em `src/utils/route_helpers.py`; atualizar `GenerationState` em `genetic_algorithm.py` para incluir `generation: int = 0`; substituir o boilerplate nas estratégias estocásticas e determinísticas.
  + **How much:** Médio.
  + **Stakeholder Implications:** Elimina duplicação de código entre estratégias CPU/GPU e permite passar `int`, `np.random.Generator` ou `Callable[[], Generator]` em qualquer estratégia estocástica.
  + **DoD:** Zero blocos `if context is not None and hasattr(context, "rng")` duplicados nas estratégias; `mypy` e `ruff` com 0 erros; testes unitários validando `to_cpu_batch`, `restore_routes_shape(to_gpu=True)` e `resolve_rng` com `int`, `Generator` e `Callable`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [types.py](../../src/protocols/types.py#L1-L54)
      + [route_helpers.py](../../src/utils/route_helpers.py#L1-L150)
      + [utils/**init**.py](../../src/utils/__init__.py#L1-L24)
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L90-L105)
      + [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L1-L60)
      + [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L1-L85)
      + [roulette_wheel_selection.py](../../src/algorithms/strategies/selection/roulette_wheel_selection.py#L1-L80)
      + [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L1-L140)
      + [pmx_crossover.py](../../src/algorithms/strategies/crossover/pmx_crossover.py#L1-L145)
      + [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L1-L150)
      + [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L1-L120)
    - **Resumo:** Definido o tipo `SeedLike` em `src/protocols/types.py`; implementados `is_gpu_array`, `to_cpu_batch` e `restore_routes_shape(..., to_gpu=...)` em `src/utils/route_helpers.py`, e `make_rng`/`resolve_rng` inicialmente no mesmo módulo *(extraídos para `src/utils/rng.py` na TASK 8.10)*; refatoradas todas as estratégias em `src/algorithms/strategies/` para eliminar duplicações de conversão de tensores CuPy/NumPy e resolução de RNG.
    - **Atendimento ao DoD:** Boilerplate duplicado eliminado das estratégias; `GenerationState` atualizado com o campo `generation`; testes unitários em `tests/test_modular_architecture.py` passando.

> [!WARNING]
> **Correção Arquitetural (Task 8.10):** A resolução de geradores pseudoaleatórios (`make_rng` e `resolve_rng`) foi originalmente agrupada em `src/utils/route_helpers.py`, gerando acoplamento indevido entre utilitários de shape de rota e geradores pseudoaleatórios. Essa decisão foi corrigida na Task 8.10 com a extração para `src/utils/rng.py` e re-exportação em `src/utils/__init__.py`.

- [x] **[COMPLETED]** **Task 7.3: Implementação de `GPUInversionMutation` (`inversion_mutation.cu`)**
  + **What:** Implementar o kernel CUDA dedicado `src/algorithms/strategies/mutation/kernels/inversion_mutation.cu` e a classe `GPUInversionMutation` em `src/algorithms/strategies/mutation/inversion_mutation.py`, registrando-a em `STRATEGY_REGISTRY`.
  + **Why:** `InversionMutation` possuía apenas implementação CPU porque não havia kernel CUDA próprio para reversão estocástica de sub-segmentos; como o operador de mutação opera em rotas independentes (1 thread por rota), o kernel é independente do `two_opt.cu` (que avalia custos $O(n^2)$ na matriz de distâncias).
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/strategies/mutation/kernels/inversion_mutation.cu`, [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L1-L108), [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L1-L75).
  + **When:** Após Task 7.2.
  + **How:** Criar `inversion_mutation.cu` com `inversion_mutation_kernel` (1 thread por rota revertendo `route[left..right]` quando `probs[idx] < mutation_rate`); implementar `GPUInversionMutation` em `inversion_mutation.py` contabilizando `h2d_bytes`, `d2h_bytes` e `kernel_launches`; exportar em `mutation/__init__.py` e `STRATEGY_REGISTRY`.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Permite usar mutação por inversão acelerada em GPU em pipelines `FullGPUModular` e `HybridOptimized`.
  + **DoD:** `GPUInversionMutation` registrada em `STRATEGY_REGISTRY`, preservando permutações válidas em arrays 1D e 2D (CPU e GPU) nos testes automatizados.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu#L1-L55)
      + [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L1-L220)
      + [mutation/**init**.py](../../src/algorithms/strategies/mutation/__init__.py#L1-L20)
      + [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L1-L85)
    - **Resumo:** Criado kernel CUDA `inversion_mutation.cu` onde cada thread inverte o intervalo selecionado `[left..right]` de forma independente para cada rota da população; implementada a classe `GPUInversionMutation` com métricas completas de tempo, bytes transferidos e chamadas de kernel; registrada em `STRATEGY_REGISTRY`.
    - **Atendimento ao DoD:** `GPUInversionMutation` integrada ao `STRATEGY_REGISTRY` e validada por teste unitário em `tests/test_modular_architecture.py`.

- [x] **[COMPLETED]** **Task 7.4: Implementação de Variantes de 2-Opt (`CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`, `BatchGPUMultiMoveTwoOpt`)**
  + **What:** Implementar variações algorítmicas do 2-opt em `src/algorithms/strategies/local_search/two_opt.py` e `src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu`: (1) First-Improvement (`CPUFirstImprovementTwoOpt`), que aplica imediatamente a primeira troca melhoradora encontrada; e (2) Multi-Move / Disjoint-Segment (`CPUMultiMoveTwoOpt` e `BatchGPUMultiMoveTwoOpt`), que aplica múltiplas reversões de sub-segmentos disjuntos ($j_1 + 1 < i_2$) em uma única varredura $O(n^2)$.
  + **Why:** O 2-opt Best-Improvement tradicional avalia todos os $\approx n^2/2$ pares de arestas mas aplica apenas 1 movimento por varredura, reavaliando pares independentes repetidamente na iteração seguinte. Movimentos em sub-segmentos disjuntos possuem $\Delta$ aditivos e independentes e podem ser aplicados na mesma iteração.
  + **Who:** Agente (`implementer`).
  + **Where:** [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L1-L337), `src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu`, [local_search/**init**.py](../../src/algorithms/strategies/local_search/__init__.py#L1-L20), [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L1-L75).
  + **When:** Após Task 7.2.
  + **How:** Implementar `CPUFirstImprovementTwoOpt` e `CPUMultiMoveTwoOpt` em `two_opt.py`; criar `two_opt_multi_move.cu` e `BatchGPUMultiMoveTwoOpt` onde as threads avaliam pares em paralelo e a thread 0 aplica gulosamente todos os movimentos melhoradores disjuntos coletados pelas threads do bloco em memória compartilhada na mesma iteração; registrar as 3 novas classes em `STRATEGY_REGISTRY`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Habilita experimentos comparativos entre Best-Improvement, First-Improvement e Multi-Move Disjoint 2-Opt em CPU e GPU via `configs/algorithms.json`.
  + **DoD:** `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt` e `BatchGPUMultiMoveTwoOpt` registradas em `STRATEGY_REGISTRY`, reduzindo monotonicamente o custo das rotas e aprovadas em testes de paridade e validade de permutação.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L1-L150)
      + [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L380-L597)
      + [local_search/**init**.py](../../src/algorithms/strategies/local_search/__init__.py#L1-L25)
      + [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L1-L85)
    - **Resumo:** Implementados `CPUFirstImprovementTwoOpt` (aplica a primeira reversão $\Delta < 0$) e `CPUMultiMoveTwoOpt` (aplica reversões de intervalos não-conflitantes na mesma varredura $O(n^2)$); criado o kernel CUDA `two_opt_multi_move.cu` e a classe `BatchGPUMultiMoveTwoOpt`; exportadas em `local_search/__init__.py` e registradas em `STRATEGY_REGISTRY`.
    - **Atendimento ao DoD:** As 3 variantes foram registradas no registry de estratégias e validadas por `test_two_opt_variants` em `tests/test_modular_architecture.py`.

- [x] **[COMPLETED]** **Task 7.5: Adoção Exclusiva do Schema `datasets/tsp.duckdb`, Remoção de `DEFAULT_SELECTED_TSP_FILES` e Metadados Ricos nos Checkpoints**
  + **What:** (1) Remover `DEFAULT_SELECTED_TSP_FILES` de `config_loader.py` (transferindo a seleção de problemas para `benchmark_scripts/run_selected_problems.py`); (2) Reescrever `DatabaseLoader` (`src/loaders/database_loader.py`) e `ingest_tsplib_directory_to_duckdb` (`src/loaders/tsplib_loader.py`) para usar exclusivamente o schema normalizado de `datasets/tsp.duckdb` (`problems`, `tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `solutions`, `file_tracking`) sem retrocompatibilidade com `routing.duckdb`/`nodes`; (3) Enriquecer `Problem` (`src/data_models/problem.py`) e inserir todos os metadados relevantes do problema nos checkpoints JSON e `problem_statistics` gerados por `orchestration.py`.
  + **Why:** `config_loader.py` continha caminhos hardcoded (`DEFAULT_SELECTED_TSP_FILES`), `DatabaseLoader` falhava ao abrir `datasets/tsp.duckdb` (`BinderError: Referenced column "dimension" not found in FROM clause`), e os checkpoints JSON omitiam metadados ricos presentes no banco (`comment`, `edge_weight_format`, `tsplib_name`, `is_symmetric`, `problem_id`, `has_solution`, `source_file`, `file_checksum`).
  + **Who:** Agente (`implementer`).
  + **Where:** [problem.py](../../src/data_models/problem.py#L29-L160), [database_loader.py](../../src/loaders/database_loader.py#L37-L367), [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L366-L471), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L39-L50), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L130-L470), [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L1-L70), `tests/`.
  + **When:** Após Tasks 7.2–7.4.
  + **How:** Atualizar `Problem` com campos opcionais de metadados; reescrever `DatabaseLoader` com `db_path="datasets/tsp.duckdb"` consultando `problems` + tabelas específicas (`tsp_problems`, `atsp_problems`, `cvrp_problems`, `edge_weight_matrices`, `file_tracking`, `solutions`) e adicionar `list_problems()` / `load_metadata()`; atualizar `ingest_tsplib_directory_to_duckdb` para criar o mesmo schema normalizado; adicionar `build_problem_configs_from_duckdb` em `config_loader.py` e remover `DEFAULT_SELECTED_TSP_FILES`; gravar `problem_metadata` completo nos checkpoints e `problem_statistics` em `orchestration.py`.
  + **How much:** Alto.
  + **Stakeholder Implications:** `datasets/tsp.duckdb` torna-se a fonte única de verdade; `config_loader.py` fica 100% desacoplado de listas de arquivos fixas; checkpoints passam a conter rastreabilidade completa da instância.
  + **DoD:** `DatabaseLoader('datasets/tsp.duckdb').load('berlin52')` e `'bayg29'` carregam diretamente do DuckDB sem fallback; `DEFAULT_SELECTED_TSP_FILES` removido de `src/`; checkpoints contêm `problem_metadata` completo; `pytest`, `mypy` e `ruff` passam com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [problem.py](../../src/data_models/problem.py#L30-L160)
      + [database_loader.py](../../src/loaders/database_loader.py#L1-L260)
      + [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L35-L640)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L35-L675)
      + [benchmarking/**init**.py](../../src/benchmarking/__init__.py#L25-L65)
      + [runner/**init**.py](../../src/benchmarking/runner/__init__.py#L20-L50)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L70-L520)
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L1-L55)
      + `test_benchmark_selected_problems.py` *(removido na TASK 8.7)*
      + `test_modular_architecture.py` *(removido na reorganização da suíte)*
    - **Resumo:** `DatabaseLoader` e `ingest_tsplib_directory_to_duckdb` foram migrados para o schema normalizado de `datasets/tsp.duckdb` (com suporte a listas aninhadas nativas de coordenadas e matrizes inteiras sem dependência de parsing JSON); `Problem` foi enriquecido com metadados e o método `to_metadata_dict()`; `DEFAULT_SELECTED_TSP_FILES` foi eliminado de `config_loader.py`, introduzindo-se `build_problem_configs_from_duckdb`; a lista `SELECTED_PROBLEMS` foi definida localmente no runner `run_selected_problems.py`; checkpoints e `problem_statistics` agora gravam `problem_metadata` completo.
    - **Atendimento ao DoD:** `berlin52` e `bayg29` carregados diretamente do DuckDB; zero listas hardcoded em `config_loader.py`; metadados gravados nos checkpoints; 22 testes em `pytest`, `mypy` e `ruff` passando com 0 erros.

---

## Fase 8: Hermetismo de Caminhos, Separação Estrita de CLI/Scripts e Isolamento de Testes

- [x] **[COMPLETED]** **Task 8.1: Reversão e Restauração dos Arquivos da Fase 7 Revertidos Indevidamente**
  + **What:** Restaurar integralmente os 8 arquivos modificados na Fase 7 que haviam sido revertidos para o commit `HEAD` (`62edcbd`), resgatando as implementações de `route_helpers.py`, schema normalizado de `database_loader.py`, variantes de 2-opt em `two_opt.py`, operadores genéticos centrais com `make_rng` e re-exportação de `GPUInversionMutation`.
  + **Why:** O reset acidental para o commit `HEAD` apagou utilitários essenciais (`is_gpu_array`, `restore_routes_shape`, `make_rng`), variantes algorítmicas de 2-opt e queries normalizadas do DuckDB, quebrando consumidores e a suite de tipos.
  + **Who:** Agente (`implementer`).
  + **Where:** [route_helpers.py](../../src/utils/route_helpers.py#L1-L150), [database_loader.py](../../src/loaders/database_loader.py#L1-L360), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L380-L600), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L1-L140), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L1-L150), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L1-L85), [roulette_wheel_selection.py](../../src/algorithms/strategies/selection/roulette_wheel_selection.py#L1-L80), [mutation/**init**.py](../../src/algorithms/strategies/mutation/__init__.py#L1-L20).
  + **When:** Primeiro item obrigatório da Fase 8.
  + **How:** Reconstituir cada arquivo a partir do histórico completo de alterações registradas no transcript e verificar consistência imediata com interpretador e type checker.
  + **How much:** Médio.
  + **Stakeholder Implications:** Restabelece toda a fundação arquitetural e as novas estratégias da Fase 7 sem perdas.
  + **DoD:** `import utils, algorithms, algorithms.strategies, loaders` executa com sucesso; `mypy` passa com 0 erros nos 24 arquivos afetados.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [route_helpers.py](../../src/utils/route_helpers.py#L1-L150)
      + [database_loader.py](../../src/loaders/database_loader.py#L1-L360)
      + [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L380-L600)
      + [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L1-L140)
      + [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L1-L150)
      + [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L1-L85)
      + [roulette_wheel_selection.py](../../src/algorithms/strategies/selection/roulette_wheel_selection.py#L1-L80)
      + [mutation/**init**.py](../../src/algorithms/strategies/mutation/__init__.py#L1-L20)
    - **Resumo:** Restaurados integralmente os 8 módulos da Fase 7 que haviam sido redefinidos para o estado do `HEAD`.
    - **Atendimento ao DoD:** `mypy` e verificação de imports concluídos com 0 erros nos 24 arquivos de estratégia e loaders.

- [x] **[COMPLETED]** **Task 8.2: Limpeza de Caminhos Hardcoded nas Camadas de Carregamento (`TsplibLoader`, `DatabaseLoader`)**
  + **What:** Eliminar valores padrão hardcoded para caminhos de datasets (`datasets/tsp` e `datasets/tsp.duckdb`) das assinaturas de `TsplibLoader.__init__`, `ingest_tsplib_directory_to_duckdb` e `DatabaseLoader.__init__`, tornando os caminhos argumentos obrigatórios na API da biblioteca.
  + **Why:** Valores padrão relativos à raiz do projeto quebram chamadas a partir de outros diretórios de execução, impedem testes herméticos e violam o princípio de projeto de que bibliotecas sob `src/` não devem ter caminhos externos fixos.
  + **Who:** Agente (`implementer`).
  + **Where:** [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L415-L435), [database_loader.py](../../src/loaders/database_loader.py#L50-L75).
  + **When:** Após Task 8.1.
  + **How:** Modificar `TsplibLoader.__init__(self, tsp_dir: str | Path)` para exigir `tsp_dir`; modificar `DatabaseLoader.__init__(self, db_path: str | Path, fallback_tsp_dir: str | Path | None = None)` para exigir `db_path`; atualizar docstrings em `problem.py` e `problem_context.py`.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Elimina acoplamento oculto com diretórios do workspace na camada de carregamento de dados.
  + **DoD:** Nenhuma assinatura de classe de loader contém defaults para caminhos no disco; docstrings atualizadas; `mypy src/loaders` com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L415-L435)
      + [database_loader.py](../../src/loaders/database_loader.py#L50-L75)
      + [exceptions.py](../../src/data_models/exceptions.py#L25-L35)
      + [problem.py](../../src/data_models/problem.py#L45-L55)
      + [problem_context.py](../../src/protocols/problem_context.py#L30-L40)
    - **Resumo:** `tsp_dir` e `db_path` foram transformados em parâmetros obrigatórios em todos os loaders de `src/loaders/`; docstrings de exemplo foram atualizadas.
    - **Atendimento ao DoD:** Nenhuma chamada com default de arquivo remanescente na API de `loaders`.

- [x] **[COMPLETED]** **Task 8.3: Limpeza de Caminhos Hardcoded no Framework de Benchmarking**
  + **What:** Eliminar a função `default_config_dir()`, caminhos hardcoded para `datasets/tsp.duckdb`, diretórios de checkpoints fixos (`results_v3/checkpoints`) e referências ao objeto `argparse.Namespace` em `orchestration.py`, `config_loader.py`, `checkpoint_io.py` e `import_checkpoints.py`.
  + **Why:** O benchmark central continha fallbacks como `project_root / "datasets" / "tsp.duckdb"` e caminhos padrão `results_v3`, além de acoplar `run_comprehensive_benchmark` com o parser CLI (`args: Namespace`).
  + **Who:** Agente (`implementer`).
  + **Where:** [config_loader.py](../../src/benchmarking/runner/config_loader.py#L35-L60), [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L35-L55), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L190-L335), [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L90-L115), [benchmarking/**init**.py](../../src/benchmarking/__init__.py#L35-L75), [runner/**init**.py](../../src/benchmarking/runner/__init__.py#L25-L60).
  + **When:** Após Task 8.2.
  + **How:** Tornar `config_dir: str | Path` e `output_dir: str | Path` argumentos nominais obrigatórios em `run_comprehensive_benchmark`; exigir `checkpoint_dir` e `problem_stats_dir` em `CheckpointManager.__init__` e `ensure_checkpoint_dirs`; exigir `output_dir` em `import_checkpoints`; remover `default_config_dir` do pacote.
  + **How much:** Médio.
  + **Stakeholder Implications:** `run_comprehensive_benchmark` torna-se uma função de biblioteca puramente tipada, portável e invocável de qualquer script ou teste.
  + **DoD:** `default_config_dir` completamente removido de `src/benchmarking`; `orchestration.py` não importa `Namespace`; `mypy src/benchmarking` passa com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L35-L670)
      + [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L35-L290)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L190-L335)
      + [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L90-L195)
      + [benchmarking/**init**.py](../../src/benchmarking/__init__.py#L35-L75)
      + [runner/**init**.py](../../src/benchmarking/runner/__init__.py#L25-L60)
    - **Resumo:** `default_config_dir` e referências a caminhos padrão no disco foram eliminados; `run_comprehensive_benchmark` agora exige `config_dir` e `output_dir` explicitamente; `CheckpointManager` e `import_checkpoints` exigem caminhos nominais.
    - **Atendimento ao DoD:** `mypy src/benchmarking` reporta `Success: no issues found in 17 source files`.

- [x] **[COMPLETED]** **Task 8.4: Criação do Pacote `src/cli/` e Registro de Entrypoints em `pyproject.toml`**
  + **What:** Extrair toda a lógica de parsing de linha de comando (`argparse`) dispersa em loaders e módulos de banco de dados para um pacote dedicado `src/cli/`, e registrar comandos executáveis em `[project.scripts]` no `pyproject.toml`.
  + **Why:** Invocadores CLI (`argparse.ArgumentParser` e blocos `if __name__ == '__main__':`) poluíam `tsplib_loader.py`, `import_checkpoints.py` e `validate_consistency.py`. O isolamento em `src/cli/` separa estritamente a API programática das interfaces de terminal.
  + **Who:** Agente (`implementer`).
  + **Where:** [cli/**init**.py](../../src/cli/__init__.py#L1-L5), [cli/tsplib.py](../../src/cli/tsplib.py#L1-L75), [cli/benchmark.py](../../src/cli/benchmark.py#L1-L115), [cli/checkpoints.py](../../src/cli/checkpoints.py#L1-L85), [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L650-L705), [validate_consistency.py](../../src/benchmarking/database/validate_consistency.py#L250-L300), [pyproject.toml](../../pyproject.toml#L36-L45).
  + **When:** Após Task 8.3.
  + **How:** Criar `src/cli/tsplib.py` (comando `tsp-ingest`), `src/cli/benchmark.py` (comando `tsp-benchmark`), e `src/cli/checkpoints.py` (comandos `tsp-import-checkpoints` e `tsp-validate-db`); registrar os 4 comandos na seção `[project.scripts]` do `pyproject.toml`; adicionar `"cli"` em `known-first-party` do Ruff.
  + **How much:** Médio.
  + **Stakeholder Implications:** Usuários podem rodar ferramentas de terminal diretamente via comandos padrão instaláveis pelo `pip`/`uv` ou via `python -m cli.<modulo>`.
  + **DoD:** Zero referências a `argparse` em `src/loaders/` e `src/benchmarking/database/`; comandos registrados no `pyproject.toml`; `mypy src/cli` limpo.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [cli/**init**.py](../../src/cli/__init__.py#L1-L5)
      + [cli/tsplib.py](../../src/cli/tsplib.py#L1-L75)
      + [cli/benchmark.py](../../src/cli/benchmark.py#L1-L115)
      + [cli/checkpoints.py](../../src/cli/checkpoints.py#L1-L85)
      + [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L15-L660)
      + [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L25-L270)
      + [validate_consistency.py](../../src/benchmarking/database/validate_consistency.py#L20-L260)
      + [pyproject.toml](../../pyproject.toml#L36-L45)
    - **Resumo:** Criado o pacote `src/cli/` contendo os submódulos `tsplib`, `benchmark` e `checkpoints`; removido todo o código de parsing CLI dos módulos de biblioteca; registrados 4 scripts no `pyproject.toml` (`tsp-ingest`, `tsp-benchmark`, `tsp-import-checkpoints`, `tsp-validate-db`).
    - **Atendimento ao DoD:** `mypy src/cli` executado com 0 erros; entrypoints instaláveis funcionais.

- [x] **[COMPLETED]** **Task 8.5: Conversão de `benchmark_scripts/` em Runners Experimentais Puros em python**
  + **What:** Atualizar todos os scripts em `benchmark_scripts/` para chamarem as funções da biblioteca `benchmarking` com argumentos nominais explícitos baseados em constantes do próprio script (`PROJECT_ROOT`, `CONFIG_DIR`, `DB_PATH`, `OUTPUT_DIR`), eliminando qualquer dependência de `argparse` em scripts de experimentação.
  + **Why:** Scripts experimentais devem se comportar como cadernos reproduzíveis (notebooks python), declarando explicitamente suas entradas e saídas sem suposições ocultas ou parsing desnecessário de terminal.
  + **Who:** Agente (`implementer`).
  + **Where:** [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L35-L65), [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py#L15-L50), [compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py#L25-L95), [import_checkpoints.py](../../benchmark_scripts/import_checkpoints.py#L1-L35), [validate_consistency.py](../../benchmark_scripts/validate_consistency.py#L1-L40).
  + **When:** Após Task 8.4.
  + **How:** Declarar `CONFIG_DIR = PROJECT_ROOT / "configs"` e `OUTPUT_DIR` explicitamente nos scripts e passá-los a `run_comprehensive_benchmark`, `load_algorithms_config` e `import_checkpoints`.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Experimentos são 100% autodocumentados e reproduzíveis com `python -m benchmark_scripts.<nome>`.
  + **DoD:** Todos os 5 scripts em `benchmark_scripts/` passam na checagem estática `mypy benchmark_scripts` com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L35-L65)
      + [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py#L15-L50)
      + [compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py#L25-L95)
      + [import_checkpoints.py](../../benchmark_scripts/import_checkpoints.py#L1-L35)
      + [validate_consistency.py](../../benchmark_scripts/validate_consistency.py#L1-L40)
    - **Resumo:** Adaptados todos os scripts de experimentação para uso de caminhos explícitos via constantes dedicadas, eliminando referências a `default_config_dir` e `argparse`.
    - **Atendimento ao DoD:** `mypy benchmark_scripts` passa com 0 erros em 6 arquivos-fonte.

- [x] **[COMPLETED]** **Task 8.6: Reversão de `pythonpath = ["src", "."]` para `pythonpath = ["src"]` e Criação de Fixtures Herméticas**
  + **What:** Reverter `pythonpath` em `pyproject.toml` para `["src"]`, remover `"benchmark_scripts"` de `known-first-party` do Ruff e criar diretórios de fixtures herméticos (`tests/fixtures/tsp/` com 5 arquivos de amostra: `burma14`, `gr17`, `bayg29`, `att48`, `berlin52`; `tests/fixtures/configs/` com `algorithms.json`, `benchmark.json`, `problems.json`; e `tests/conftest.py` com fixtures pytest).
  + **Why:** A inclusão indevida de `"."` tornava todo o repositório raiz importável (inclusive scripts e pastas temporárias), mascarando problemas de empacotamento e quebrando o isolamento hermético.
  + **Who:** Agente (`implementer`).
  + **Where:** [pyproject.toml](../../pyproject.toml#L45-L50), `tests/fixtures/tsp/`, `tests/fixtures/configs/`, [conftest.py](../../tests/conftest.py#L1-L50).
  + **When:** Após Task 8.5.
  + **How:** Configurar `pythonpath = ["src"]`; copiar 5 arquivos `.tsp` leves (~5.4 KB total) para `tests/fixtures/tsp/`; criar configs dedicados de teste em `tests/fixtures/configs/`; implementar fixtures `fixture_tsp_dir`, `fixture_config_dir`, `fixture_duckdb_path`, `burma14_problem` e `berlin52_problem` em `tests/conftest.py`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Garante que a suite de testes rode de forma 100% isolada e reproduzível mesmo em máquinas limpas sem os datasets completos do TCC baixados.
  + **DoD:** `pyproject.toml` contém apenas `pythonpath = ["src"]`; fixtures criadas; pytest descobre e executa fixtures sem erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [pyproject.toml](../../pyproject.toml#L45-L50)
      + [tests/fixtures/tsp/burma14.tsp](../../tests/fixtures/tsp/burma14.tsp#L1-L24)
      + [tests/fixtures/tsp/gr17.tsp](../../tests/fixtures/tsp/gr17.tsp#L1-L22)
      + [tests/fixtures/tsp/bayg29.tsp](../../tests/fixtures/tsp/bayg29.tsp#L1-L68)
      + [tests/fixtures/tsp/att48.tsp](../../tests/fixtures/tsp/att48.tsp#L1-L56)
      + [tests/fixtures/tsp/berlin52.tsp](../../tests/fixtures/tsp/berlin52.tsp#L1-L60)
      + [tests/fixtures/configs/benchmark.json](../../tests/fixtures/configs/benchmark.json#L1-L10)
      + [tests/fixtures/configs/problems.json](../../tests/fixtures/configs/problems.json#L1-L8)
      + [tests/fixtures/configs/algorithms.json](../../tests/fixtures/configs/algorithms.json#L1-L40)
      + [conftest.py](../../tests/conftest.py#L1-L50)
    - **Resumo:** `pythonpath` revertido para `["src"]`; suite de fixtures independentes criada com instâncias cobrindo todos os formatos de peso de aresta (`GEO`, `EXPLICIT`, `ATT`, `EUC_2D`); fixtures de sessão criadas para banco temporário DuckDB e instâncias pré-carregadas.
    - **Atendimento ao DoD:** `pyproject.toml` estritamente alinhado com o empacotamento padrão; testes isolados do filesystem externo.

- [x] **[COMPLETED]** **Task 8.7: Refatoração dos Testes Automatizados para Uso de Fixtures e Criação de `test_benchmarking_pipeline.py` e `test_cli.py`**
  + **What:** (1) Deletar o arquivo de teste acoplado a scripts `tests/test_benchmark_selected_problems.py`; (2) Criar `tests/test_benchmarking_pipeline.py` testando exaustivamente o pipeline de benchmark e estatísticas usando fixtures; (3) Criar `tests/test_cli.py` cobrindo os 4 entrypoints de terminal; (4) Atualizar `test_modular_architecture.py`, `test_early_stopping.py` e `test_incremental_checkpoints.py` para usar fixtures.
  + **Why:** O teste antigo importava scripts e usava caminhos hardcoded que falhavam caso a base de dados não estivesse gerada previamente no workspace.
  + **Who:** Agente (`implementer`).
  + **Where:** `tests/test_benchmark_selected_problems.py` (deletado), `test_benchmarking_pipeline.py` *(removido na reorganização da suíte; cobertura em `tests/benchmarking/test_integration_pipeline.py`)*, `test_cli.py` *(removido; cobertura em `tests/cli/`)*, `test_modular_architecture.py` *(removido; cobertura em `tests/**`)*, [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L1-L55), [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L1-L70).
  + **When:** Conclusão da Fase 8.
  + **How:** Substituir todas as referências diretas a `datasets/` por injeção de dependência via fixtures do pytest; verificar execução em `tmp_path`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Suite de testes torna-se totalmente autossuficiente e roda em ambiente de CI/CD sem montagem de volumes externos.
  + **DoD:** 30 testes passando no `pytest` em tempo inferior a 15s; `mypy src tests benchmark_scripts` com 0 erros em 74 arquivos; `ruff check src tests benchmark_scripts` com 0 violações.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + `tests/test_benchmark_selected_problems.py` (removido)
      + `test_benchmarking_pipeline.py` *(removido na reorganização da suíte)*
      + `test_cli.py` *(removido; cobertura em `tests/cli/`)*
      + `test_modular_architecture.py` *(removido; cobertura em `tests/**`)*
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L1-L55)
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L1-L70)
    - **Resumo:** `tests/test_benchmark_selected_problems.py` foi substituído por `test_benchmarking_pipeline.py` (focado na biblioteca) e `test_cli.py` (focado nos comandos CLI); todos os testes foram adaptados para utilizar fixtures injetadas pelo pytest.
    - **Atendimento ao DoD:** 30 testes unitários e de integração executados com 100% de aprovação (30 passed em ~9s); zero erros em 74 arquivos de `mypy` e zero violações no `ruff`.

- [x] **[COMPLETED]** **Task 8.10: Extração de Utilitários RNG para `src/utils/rng.py` e Isolamento de Responsabilidade Única (SRP)**
  + **What:** Extrair `make_rng` e `resolve_rng` de `src/utils/route_helpers.py` para um módulo dedicado `src/utils/rng.py`, re-exportando-os em `src/utils/__init__.py`, atualizando todos os consumidores em `src/algorithms/strategies/` e criando a suíte unitária `tests/utils/test_unit_rng.py`.
  + **Why:** O agrupamento de geração pseudoaleatória (`make_rng`, `resolve_rng`) dentro de `route_helpers.py` violava o princípio de responsabilidade única (SRP), forçando classes focadas puramente em lógica estocástica a importar rotinas de conversão e normalização de arrays de rotas.
  + **Who:** Agente (`implementer`).
  + **Where:** [rng.py](../../src/utils/rng.py#L1-L66), [route_helpers.py](../../src/utils/route_helpers.py#L1-L113), [utils/**init**.py](../../src/utils/__init__.py#L1-L25), [test_unit_rng.py](../../tests/utils/test_unit_rng.py#L1-L182), e estratégias em `src/algorithms/strategies/`.
  + **When:** Início do follow-up da Fase 8.
  + **How:** Criar `src/utils/rng.py` com `make_rng` e `resolve_rng`; remover as funções de `route_helpers.py`; ajustar `utils/__init__.py`; atualizar os imports das 8 estratégias estocásticas para `from utils.rng import make_rng, resolve_rng`; criar `tests/utils/test_unit_rng.py`.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Elimina acoplamento arquitetural e esclarece a separação entre geometria/residência de rotas e determinismo estocástico.
  + **DoD:** `make_rng` e `resolve_rng` removidos de `route_helpers.py`; re-exportados em `utils/__init__.py`; todos os consumidores atualizados; `tests/utils/test_unit_rng.py` executando com aprovação nos testes determinísticos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [rng.py](../../src/utils/rng.py#L1-L66)
      + [route_helpers.py](../../src/utils/route_helpers.py#L1-L113)
      + [utils/**init**.py](../../src/utils/__init__.py#L1-L25)
      + [test_unit_rng.py](../../tests/utils/test_unit_rng.py#L1-L182)
    - **Resumo:** Criado o módulo `src/utils/rng.py` centralizando a lógica de resolução de `np.random.Generator` a partir de `SeedLike` (`int`, `Generator`, factory ou `None`), desacoplando-o de `route_helpers.py`. Todos os 8 arquivos de estratégias estocásticas e o teste unitário foram atualizados para importar de `utils.rng`.
    - **Atendimento ao DoD:** Zero ocorrências de `make_rng` em `route_helpers.py`; 14 testes passando e 1 xfail documentado em `tests/utils/test_unit_rng.py`.

- [x] **[COMPLETED]** **Task 8.9: Desacoplamento de `DatabaseLoader` via Handlers Polimórficos de Formulação de Problemas**
  + **What:** Refatorar `DatabaseLoader` para eliminar blocos condicionais encadeados `if problem_type == "TSP": ... elif` em `list_instances()` e `_load_metadata()`, introduzindo um protocolo polimórfico `ProblemTypeHandler` e um registro indexado por formulação (`TSP`, `ATSP`, `CVRP`).
  + **Why:** A lógica atual viola o princípio Aberto/Fechado (OCP). Adicionar suporte a novas variantes e problemas de roteamento (ex.: MDVRP, VRPTW, mTSP) exige editar o corpo de `DatabaseLoader` e duplicar projeções SQL e transformações de metadados.
  + **Who:** Desenvolvedor / Agente (`implementer`).
  + **Where:** [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L1-L250), [database_loader.py](../../src/loaders/database_loader.py#L43-L212), [loaders/**init**.py](../../src/loaders/__init__.py#L11-L48), [exceptions.py](../../src/data_models/exceptions.py#L327-L341), e [test_unit_problem_type_handlers.py](../../tests/loaders/test_unit_problem_type_handlers.py#L1-L352).
  + **When:** Conclusão da Fase 8, na ordem 8.10 → 8.11 → 8.9 fixada no plano de follow-up da Fase 8.
  + **How:**
    1. Definir o protocolo `ProblemTypeHandler(Protocol)` com os métodos `list_problems(conn: DuckDBPyConnection) -> list[dict[str, Any]]` e `load_metadata(conn: DuckDBPyConnection, base: BaseProblemRow) -> dict[str, Any]`.
    2. Implementar handlers concretos (`TspHandler`, `AtspHandler`, `CvrpHandler`) e registrá-los em `PROBLEM_TYPE_HANDLERS: dict[str, ProblemTypeHandler]`.
    3. Delegar as consultas e extração de metadados de `DatabaseLoader` diretamente ao handler correspondente, permitindo injeção customizada via parâmetro `handlers=` e levantando `UnsupportedProblemTypeError` quando o tipo de formulação for desconhecido.
  + **How much:** Médio.
  + **Stakeholder Implications:** Permite estender o banco DuckDB com novas tabelas e formulações de roteamento sem risco de quebrar o pipeline existente de TSP/ATSP.
  + **DoD:** Zero condicionais `if problem_type == "TSP"` dentro de `list_instances()` e `_load_metadata()`; 100% de cobertura nos testes de carregamento existentes e novos testes unitários.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L1-L250)
      + [database_loader.py](../../src/loaders/database_loader.py#L43-L212)
      + [loaders/**init**.py](../../src/loaders/__init__.py#L11-L48)
      + [exceptions.py](../../src/data_models/exceptions.py#L327-L341)
      + [test_unit_problem_type_handlers.py](../../tests/loaders/test_unit_problem_type_handlers.py#L1-L352)
    - **Resumo:** Criado o protocolo polimórfico `ProblemTypeHandler` e suas implementações concretas `TspHandler`, `AtspHandler` e `CvrpHandler` em `src/loaders/problem_type_handlers.py`, além do registro global extensível `PROBLEM_TYPE_HANDLERS`. `DatabaseLoader` foi refatorado para delegar a consulta de catálogo (`list_problems`) e extração de metadados (`load_metadata`) ao handler do tipo correspondente, permitindo injeção de dependência via `handlers=` e levantando `UnsupportedProblemTypeError` para formulações não suportadas.
    - **Atendimento ao DoD:** Eliminados todos os blocos condicionais `if problem_type == ...` de `DatabaseLoader`; 100% de aprovação na nova suíte `tests/loaders/test_unit_problem_type_handlers.py` (8 testes passando) e testes de integração existentes.

- [x] **[COMPLETED]** **Task 8.11: Fechamento de Lacunas nos Parâmetros CLI, Paridade de Overrides e Proteção de Sobrescrita no Banco**
  + **What:** Expandir os argumentos de linha de comando em `src/cli/benchmark.py` (`--patience`, `--seed`, `--population-size`, `--mutation-rate`, `--strategy-param`, `--incremental-runs`/`--no-incremental-runs`, `--log-level`), restringir choices de `--local-search-stage` aos literais válidos de `LocalSearchStage`, propagar overrides através de `AlgorithmConfig.instantiate` e `orchestration.py`, corrigir o bug onde repetições ignoravam `effective_seed` caindo no default hardcoded `seed_base=42`, e implementar proteção de sobrescrita com `DatabaseNotEmptyError` e flag `--overwrite` em `ingest_tsplib_directory_to_duckdb` e `src/cli/tsplib.py`.
  + **Why:** O benchmark CLI carecia de flags essenciais para controle fino de hiperparâmetros de GA e estratégias sem necessidade de alterar arquivos JSON no disco. Além disso, a ingestão em DuckDB realizava `DELETE FROM` incondicionalmente em bancos existentes caso invocada sem salvaguardas.
  + **Who:** Agente (`implementer`).
  + **Where:** [benchmark.py](../../src/cli/benchmark.py#L1-L225), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L190-L450), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L190-L280), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L280-L285), [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L535-L550), [tsplib.py](../../src/cli/tsplib.py#L46-L86), [exceptions.py](../../src/data_models/exceptions.py#L309-L325), [test_unit_cli_benchmark.py](../../tests/cli/test_unit_cli_benchmark.py#L39-L118), [test_integration_pipeline.py](../../tests/benchmarking/test_integration_pipeline.py#L269-L325), [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L75-L89), [test_integration_loaders.py](../../tests/loaders/test_integration_loaders.py#L107-L132), e [test_unit_cli_tsplib.py](../../tests/cli/test_unit_cli_tsplib.py#L35-L52).
  + **When:** Conclusão da Fase 8.
  + **How:** Introduzir `param_overrides` e `strategy_overrides` em `AlgorithmConfig.instantiate` com validação de slots permitidos e suporte a instâncias monolíticas; repassar `seed_base=effective_seed` em `run_single_algorithm`; implementar `parse_strategy_param` com validação estrita de pares `SLOT.KEY` e fallback de JSON decodificado para string; usar `argparse.BooleanOptionalAction` para `--incremental-runs`; validar `DatabaseNotEmptyError` quando `problems` contiver registros e `overwrite=False`; validar `local_search_stage` em `GeneticAlgorithm.__init__`; criar testes unitários e de integração cobrindo parser, overrides e proteção de banco.
  + **How much:** Médio.
  + **Stakeholder Implications:** Interface CLI atinge paridade operacional completa com os parâmetros configuráveis em JSON e protege o usuário contra perda acidental de dados no banco analítico DuckDB.
  + **DoD:** Todas as flags funcionais e documentadas no `--help`; `--local-search-stage` rejeita valores inválidos com exit code 2; repetição de sementes respeita `--seed`; estratégias recebem overrides de hiperparâmetros ou falham alto com `ValueError` em chaves inválidas; segundo `tsp-ingest` sem `--overwrite` recusa com exit code 1; 100% de testes passando.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [benchmark.py](../../src/cli/benchmark.py#L1-L225)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L190-L450)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L190-L280)
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L280-L285)
      + [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L535-L550)
      + [tsplib.py](../../src/cli/tsplib.py#L46-L86)
      + [exceptions.py](../../src/data_models/exceptions.py#L309-L325)
      + [test_unit_cli_benchmark.py](../../tests/cli/test_unit_cli_benchmark.py#L39-L118)
      + [test_integration_pipeline.py](../../tests/benchmarking/test_integration_pipeline.py#L269-L325)
      + [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L75-L89)
      + [test_integration_loaders.py](../../tests/loaders/test_integration_loaders.py#L107-L132)
      + [test_unit_cli_tsplib.py](../../tests/cli/test_unit_cli_tsplib.py#L35-L52)
    - **Resumo:** Implementada paridade de parâmetros na CLI `tsp-benchmark` (`--patience`, `--seed`, `--population-size`, `--mutation-rate`, `--strategy-param`, `--incremental-runs`/`--no-incremental-runs`, `--log-level`) e validação estrita de choices em `--local-search-stage` e `GeneticAlgorithm.__init__`. No runner e loader, os parâmetros de semente e estratégia são propagados diretamente à instância de GA e monolíticos, com validação de slots de estratégia para falhar alto contra erros de digitação, e `run_single_algorithm` passa `seed_base=effective_seed`. No ingestor TSPLIB, a proteção contra sobrescrita não autorizada (`DatabaseNotEmptyError` / `--overwrite`) foi incorporada e validada.
    - **Atendimento ao DoD:** Testes automatizados dedicados cobrindo parser, overrides de semente, overrides de estratégia (incluindo falha para slots desconhecidos e algoritmos monolíticos), validação de estágio e proteção contra sobrescrita passando com 100% de sucesso.

---

## Fase 9: Capacidade de GPU no Núcleo, Entrada Válida, Isolamento do Monolítico e Contrato de Configurações

### Decisões Fixadas da Fase 9

- **DP9-1 (Local da API de GPU)**: `gpu_available()`, `require_gpu()` e o backend CuPy residem em [backend.py](../../src/protocols/backend.py); nenhum `__init__.py` executa probe de dispositivo na importação.
- **DP9-2 (Contrato de Falha)**: `GPUUnavailableError(RuntimeError)` em [exceptions.py](../../src/data_models/exceptions.py) cobre "CuPy ausente" e "sem dispositivo"; `ValueError` permanece reservado ao uso incorreto de accessors num backend NumPy.
- **DP9-3 (Empacotamento)**: `cupy-cuda12x` permanece dependência obrigatória; todos os imports de CuPy passam a ser tardios para que a importação do pacote funcione em CPU.
- **DP9-4 (Archive)**: `full_gpu_monolithic.py` e `kernels/` migram para `src/algorithms/archive/`; as exportações saem de `algorithms.__init__`; `archive/__init__.py` expõe apenas o baseline mantido; os 5 módulos legados não importáveis são removidos.
- **DP9-5 (Configurações)**: `FullGPU` e `FullGPUMonolithic` permanecem no config padrão apontando para `algorithms.archive.full_gpu_monolithic`; testes de comportamento não executam código de `archive/`, e um script de smoke cobre o baseline.
- **DP9-6 (Entrada Vazia)**: `GeneticAlgorithm.evolve` rejeita `customers` vazio com `ValueError` nomeando o argumento.
- **DP9-7 (Paridade de Configs)**: o teste de contrato compara estrutura, nomes e caminhos de classe entre `configs/` e o fixture; valores numéricos de tuning podem diferir; descrições são sincronizadas.

### Tarefas da fase 9

- [x] **Task 9.1: API de Capacidade de GPU no Núcleo**
  + **What:** Adicionar `gpu_available()`, `require_gpu()` e `cupy_backend()` ao núcleo, com detecção preguiçosa de instalação (`importlib.util.find_spec`) e de dispositivo (`getDeviceCount`), ambas com cache.
  + **Why:** A checagem de dispositivo vive hoje em `tests/gpu_probe.py`, enquanto [problem_context.py](../../src/protocols/problem_context.py#L67-L68) verifica apenas a instalação de CuPy; um pacote publicado não oferece API de capacidade e CuPy instalado sem driver falha dentro da biblioteca.
  + **Who:** Agente (`implementer`).
  + **Where:** [backend.py](../../src/protocols/backend.py#L1-L60), [exceptions.py](../../src/data_models/exceptions.py#L1-L40).
  + **When:** Primeira tarefa; 9.2, 9.3 e 9.4 dependem dela.
  + **How:** Implementar o probe com `find_spec` e um `try/except` em torno de `cp.cuda.runtime.getDeviceCount()`, sem importar CuPy no nível do módulo; definir `GPUUnavailableError(RuntimeError)` com mensagens distintas para pacote ausente e dispositivo ausente; `require_gpu()` retorna o módulo CuPy.
  + **How much:** Média.
  + **Stakeholder Implications:** Habilita instalação CPU-only e mensagens de erro acionáveis para usuários da biblioteca.
  + **DoD:** `python -c "import protocols.backend"` funciona sem CuPy; as duas mensagens são verificáveis por teste unitário; `mypy src/protocols` e `ruff check src/protocols` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [exceptions.py](../../src/data_models/exceptions.py#L199-L208), [data_models/**init**.py](../../src/data_models/__init__.py#L8-L23), [backend.py](../../src/protocols/backend.py#L168-L245).
    - **Resumo:** `GPUUnavailableError(RuntimeError)` foi introduzido em `exceptions.py` e re-exportado na raiz de `data_models`. Em `backend.py`, foram implementadas as funções públicas `gpu_available() -> bool`, `require_gpu() -> Any`, `cupy_backend() -> Any | None` e `clear_gpu_cache() -> None`, utilizando probe preguiçoso com `find_spec` e `getDeviceCount` via `lru_cache`, sem importar CuPy no nível do módulo.
    - **Atendimento ao DoD:** Importação de `protocols.backend` funciona limpamente; cobertura unitária em CPU adicionada; `mypy` e `ruff` limpos.

- [x] **Task 9.2: Adoção da API de Capacidade pelos Consumidores Internos**
  + **What:** Substituir as checagens locais de instalação por `require_gpu()` em [problem_context.py](../../src/protocols/problem_context.py#L67-L68), [gpu_helpers.py](../../src/utils/gpu_helpers.py#L11-L40) e no monolítico.
  + **Why:** Três cópias da mesma checagem divergem: `problem_context` ignora o dispositivo e `gpu_helpers` importa CuPy na importação do módulo, impedindo uso CPU-only.
  + **Who:** Agente (`implementer`).
  + **Where:** [problem_context.py](../../src/protocols/problem_context.py#L60-L80), [gpu_helpers.py](../../src/utils/gpu_helpers.py#L1-L40), [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L115-L125).
  + **When:** Após Task 9.1.
  + **How:** Mover `import cupy` para `TYPE_CHECKING` e para o corpo das funções; chamar `require_gpu()` quando `xp is not np` e antes da compilação de kernel; manter `ValueError` nos accessors `get_gpu_*` de contexto NumPy.
  + **How much:** Média.
  + **Stakeholder Implications:** Importar a biblioteca deixa de exigir CUDA; falhas de dispositivo passam a ocorrer no ponto de uso.
  + **DoD:** Importação de `protocols.problem_context` e `utils.gpu_helpers` funciona com CuPy indisponível; `pytest tests/protocols tests/utils -q` verde; mypy sem novos erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_context.py](../../src/protocols/problem_context.py#L55-L160), [gpu_helpers.py](../../src/utils/gpu_helpers.py#L1-L20), [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L115-L120).
    - **Resumo:** `ProblemContext` e `gpu_helpers.py` removeram o bloco de importação imediata de CuPy no nível do módulo; validação passa a delegar a `require_gpu()`.
    - **Atendimento ao DoD:** `ProblemContext` e `load_kernel` operam de forma puramente lazy; `pytest tests/protocols tests/utils` executado com 85 testes passando e 1 xfail esperado.

- [x] **Task 9.3: Aposentadoria de `tests/gpu_probe.py` e Remoção de `tests` do `sys.path`**
  + **What:** Passar `conftest.py` e os 5 módulos de teste a importar de `protocols.backend`, apagar o shim `tests/gpu_probe.py` e remover `"tests"` de `pythonpath` e `mypy_path`.
  + **Why:** Um teste de capacidade do núcleo não pode ser a única implementação dessa capacidade, e manter `tests` no `sys.path` permite que os testes importem módulos que a biblioteca publicada não possui.
  + **Who:** Agente (`implementer`).
  + **Where:** [pyproject.toml](../../pyproject.toml#L50), [pyproject.toml](../../pyproject.toml#L63), [conftest.py](../../tests/conftest.py#L1-L30), [test_unit_gpu_helpers.py](../../tests/utils/test_unit_gpu_helpers.py), [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py), [test_unit_strategy_contracts.py](../../tests/algorithms/test_unit_strategy_contracts.py), [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py), [test_early_stopping.py](../../tests/validation/test_early_stopping.py).
  + **When:** Após Task 9.2.
  + **How:** Importar `cupy_backend` e `gpu_available` de `protocols.backend`; manter o marcador `gpu` e o hook de skip inalterados; remover o diretório `tests` das duas listas de caminho.
  + **How much:** Média.
  + **Stakeholder Implications:** A suíte passa a consumir apenas API pública da biblioteca.
  + **DoD:** `pytest --collect-only -q` verde; `grep -rn "gpu_probe" tests` retorna 0 ocorrências; `mypy src tests` limpo.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pyproject.toml](../../pyproject.toml#L49-L65), [conftest.py](../../tests/conftest.py#L12-L46), [test_unit_gpu_helpers.py](../../tests/utils/test_unit_gpu_helpers.py#L22-L26), [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py#L16-L20), [test_unit_strategy_contracts.py](../../tests/algorithms/test_unit_strategy_contracts.py#L31-L50), [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L23-L27), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L13-L17).
    - **Resumo:** `pythonpath` e `mypy_path` foram restritos estritamente a `["src"]`. `gpu_probe` foi removido de `known-first-party` e o arquivo `tests/gpu_probe.py` foi apagado. `tests/strategy_fixtures.py` foi migrado para a fixture `make_population` em `conftest.py` e excluído.
    - **Atendimento ao DoD:** `grep -rn "gpu_probe" tests` retorna 0 resultados; `pytest` coleta e executa 717 itens; `mypy src tests` limpo.

- [x] **Task 9.4: Testes Unitários da API de Capacidade**
  + **What:** Criar `tests/protocols/test_unit_gpu_capability.py` cobrindo instalação ausente, dispositivo ausente, dispositivo presente e cache.
  + **Why:** As duas mensagens de erro e o retorno do módulo são o contrato que substitui o shim de teste; sem teste, a regressão é silenciosa.
  + **Who:** Agente (`implementer`).
  + **Where:** [test_unit_gpu_capability.py](../../tests/protocols/test_unit_gpu_capability.py).
  + **When:** Após Task 9.1.
  + **How:** Usar `monkeypatch` sobre `importlib.util.find_spec` e sobre o contador de dispositivos; limpar o cache entre casos; afirmar tipo e mensagem com `pytest.raises(..., match=...)`.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Nenhum.
  + **DoD:** Casos falham contra a implementação antiga e passam contra a nova; arquivo sem marcador `gpu` (roda em CPU).
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_gpu_capability.py](../../tests/protocols/test_unit_gpu_capability.py#L1-L151).
    - **Resumo:** Criada suíte unitária pura de CPU (sem marcador `gpu`) exercitando 5 cenários: CuPy ausente, dispositivo = 0, erro de driver CUDA, GPU presente e comportamento de cache com `clear_gpu_cache`.
    - **Atendimento ao DoD:** 5/5 testes passaram sem dependência de hardware CUDA; mensagens e tipos de exceção estritamente verificados.

- [x] **Task 9.5: Rejeição de `customers` Vazio em `evolve`**
  + **What:** Levantar `ValueError` nomeando `customers` quando a sequência estiver vazia, antes de calcular `patience` e inicializar a população.
  + **Why:** Com `n = 0`, `patience` vira 0 e todas as rotas vazias somam 0.0, produzindo um ótimo fictício que mascara erro de loader ou filtragem.
  + **Who:** Agente (`implementer`).
  + **Where:** [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L433-L457), [test_unit_genetic_algorithm_boundaries.py](../../tests/algorithms/metaheuristics/test_unit_genetic_algorithm_boundaries.py#L91-L113).
  + **When:** Independente; pode rodar após 9.1.
  + **How:** Inserir a validação no início de `evolve`; remover o decorador `xfail(strict=True)` do teste, mantendo a asserção de mensagem.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Futuro caminho CVRP herda a pré-condição explicitamente.
  + **DoD:** Teste sem `xfail` passa; `pytest tests/algorithms/metaheuristics -q` verde.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L452-L454), [test_unit_genetic_algorithm_boundaries.py](../../tests/algorithms/metaheuristics/test_unit_genetic_algorithm_boundaries.py#L88-L105).
    - **Resumo:** Inserida checagem `if len(customers) == 0: raise ValueError("customers sequence cannot be empty")` no topo de `evolve()`. Removido `@pytest.mark.xfail(strict=True)` do teste `test_empty_customers_are_rejected`.
    - **Atendimento ao DoD:** Defeito BUG-8 resolvido; teste agora passa em conformidade com o DoD.

- [x] **Task 9.6: Migração e Reorganização de `archive/`**
  + **What:** Mover `full_gpu_monolithic.py` e `kernels/` para `src/algorithms/archive/`, criar `archive/__init__.py` expondo apenas `GeneticAlgorithmFullGPUMonolithic`, remover exportações de `algorithms/__init__.py`, apagar `monolithic/` e remover o alias `GeneticAlgorithmFullGPUEarlyStop`.
  + **Why:** O monolítico não faz parte da arquitetura modular e não deve ser anunciado na API do pacote; os 5 legados atuais importam módulos inexistentes e não podem ser importados.
  + **Who:** Agente (`implementer`).
  + **Where:** [archive/](../../src/algorithms/archive), [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L30), [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L240-L261).
  + **When:** Independente de 9.1-9.5; precede 9.7, 9.8 e 9.10.
  + **How:** Mover arquivos com `git mv`; corrigir a chave de estatísticas para `convergence_history`; escrever `archive/__init__.py` sem importar os legados; apagar os 5 módulos legados aprovados; registrar a origem no `README.md` do archive.
  + **How much:** Média.
  + **Stakeholder Implications:** Baseline continua executável por caminho explícito, sem contaminar `algorithms.__all__`.
  + **DoD:** `python -c "import algorithms, algorithms.archive"` funciona sem CUDA; `set(algorithms.__all__) == {"GeneticAlgorithm", "GeneticAlgorithmBase", "LocalSearchStage"}`. O docstring do pacote pode listar `archive/` como estrutura de diretórios: a regra é sobre exportação, não sobre a palavra "monolithic".
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L240-L255), [archive/**init**.py](../../src/algorithms/archive/__init__.py#L1-L8), [archive/README.md](../../src/algorithms/archive/README.md#L1-L40), [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L25).
    - **Resumo:** `full_gpu_monolithic.py` e seus kernels migraram para `src/algorithms/archive/`. Os 5 arquivos legados não-importáveis foram removidos. O diretório `monolithic/` foi apagado. Corrigida a chave de estatísticas para `"convergence_history"` (resolvendo BUG-7). Exportações monolíticas removidas de `algorithms.__init__.py`. Criado `README.md` com 10 tags hierárquicas e frontmatter conforme as regras.
    - **Atendimento ao DoD:** `algorithms.__all__` puramente modular; importação de `algorithms` e `algorithms.archive` bem-sucedida em CPU; `grep -rn "monolithic" src/algorithms/__init__.py` retorna apenas comentário de documentação.

- [x] **Task 9.7: Atualização dos Caminhos de Classe nas Configurações**
  + **What:** Apontar `FullGPU` e `FullGPUMonolithic` para `algorithms.archive.full_gpu_monolithic` em `configs/algorithms.json` e no fixture.
  + **Why:** O caminho `algorithms.monolithic.full_gpu_monolithic` deixa de existir após 9.6, e o loader resolve classes por importação dinâmica.
  + **Who:** Agente (`implementer`).
  + **Where:** [algorithms.json](../../configs/algorithms.json#L1-L200), [algorithms.json](../../tests/fixtures/configs/algorithms.json#L1-L200).
  + **When:** Após Task 9.6.
  + **How:** Alterar `class_module` das duas entradas nos dois arquivos, mantendo nomes de variante e parâmetros.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Benchmarks existentes continuam podendo selecionar as variantes monolíticas.
  + **DoD:** `AlgorithmConfig.get_class()` resolve as duas entradas; `pytest tests/benchmarking/test_unit_config_loader.py -q` verde.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [configs/algorithms.json](../../configs/algorithms.json#L162-L184), [tests/fixtures/configs/algorithms.json](../../tests/fixtures/configs/algorithms.json#L162-L184).
    - **Resumo:** `class_module` atualizado para `"algorithms.archive.full_gpu_monolithic"` em ambas as variantes e sincronizadas as descrições entre configuração de produção e fixture.
    - **Atendimento ao DoD:** `AlgorithmConfig.get_class()` carrega as classes dinamicamente; `test_unit_config_loader.py` verde (12 passed).

- [x] **Task 9.8: Contrato Estrutural das Configurações de Produção e Fixture**
  + **What:** Criar teste que lê `configs/algorithms.json` somente para leitura e compara com o fixture: nomes de variante, `class_module`/`class_name`, slots de estratégia, chaves de parâmetros e descrições; verifica que todo caminho de classe importa.
  + **Why:** Nenhum teste atual lê o config de produção; renomear ou remover variante apenas nele mantém a suíte verde e quebra o benchmark real.
  + **Who:** Agente (`implementer`).
  + **Where:** [test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py), [configs/algorithms.json](../../configs/algorithms.json), [fixture algorithms.json](../../tests/fixtures/configs/algorithms.json).
  + **When:** Após Task 9.7.
  + **How:** Carregar os dois JSONs com `json`, comparar conjuntos e chaves estruturais (não valores numéricos), importar cada `class_module` e resolver `class_name`; sincronizar as duas descrições divergentes.
  + **How much:** Média.
  + **Stakeholder Implications:** Impede deriva silenciosa entre configuração de pesquisa e configuração de teste.
  + **DoD:** Teste falha ao renomear variante em apenas um arquivo; passa no estado atual; sem marcador `requires_dataset`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py#L1-L115).
    - **Resumo:** Criado teste de contrato estrutural validando paridade exata de seções (`default_params`, `algorithm_configs`), nomes de variantes, slots de estratégia, nomes de classes e chaves de parâmetros entre configs de produção e fixture, além de garantir importabilidade das classes.
    - **Atendimento ao DoD:** Teste unitário hermético em CPU passa sem dependências externas; impede deriva silenciosa.

- [x] **Task 9.9: Exclusão de Variantes de Archive no Teste de Composição**
  + **What:** Filtrar do teste de composição as variantes cujo `class_module` pertence a `algorithms.archive` e remover os dois `xfail(strict=True)` de BUG-7.
  + **Why:** A regra do repositório proíbe testar `archive/`; manter as variantes parametrizadas executaria código arquivado e o xfail deixaria de ter alvo após a correção da chave.
  + **Who:** Agente (`implementer`).
  + **Where:** [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L160-L215).
  + **When:** Após Task 9.7.
  + **How:** Derivar a lista de variantes do fixture e excluir as resolvidas para `algorithms.archive`, documentando o motivo no docstring do teste.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Nenhum.
  + **DoD:** `pytest tests/algorithms -q` verde; nenhum teste executa `evolve()` em variante de archive. As variantes de archive saíram do `configs/algorithms.json` ativo e vivem em `configs/algorithms.archive.json`, portanto a suíte não as carrega.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L165-L180), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L35-L38).
    - **Resumo:** Variantes de `archive/` foram excluídas da parametrização de testes da suíte ativa e os dois decoradores `xfail(strict=True)` de BUG-7 foram removidos de `test_gpu_variants_and_modular_vs_monolithic`. Em `test_early_stopping.py`, variantes cujo módulo contém `"archive"` são puladas.
    - **Atendimento ao DoD:** A suíte de testes não executa código de archive; todos os testes de composição e parada antecipada passam.

- [x] **Task 9.10: Script de Smoke do Baseline Arquivado**
  + **What:** Criar `benchmark_scripts/check_archive_baseline.py` que instancia o monolítico arquivado, executa uma evolução curta em burma14 e verifica `convergence_history`, `generations_completed` e `stop_reason`.
  + **Why:** O baseline continua necessário para comparação, mas não pode ser coberto pela suíte; sem verificação, a correção de BUG-7 fica sem evidência executável.
  + **Who:** Agente (`implementer`).
  + **Where:** [check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py).
  + **When:** Após Task 9.6.
  + **How:** Script em python puro com zero argparse (conforme diretiva do usuário), constantes explícitas no topo e função `main() -> int`.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Usuário mantém um caminho reproduzível para o baseline da tese.
  + **DoD:** Com dispositivo, executa e valida as chaves; sem dispositivo, sai com código distinto e mensagem explícita; `ruff check benchmark_scripts` limpo.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py#L1-L110).
    - **Resumo:** Criado script em python puro (sem `argparse`) com constantes explícitas (`PROBLEM_PATH`, `MAX_GENERATIONS`, etc.) que executa o baseline arquivado no burma14, valida `convergence_history`, ausência de `best_cost_history` e dimensões da rota, retornando 0 em sucesso e 2 na ausência de GPU.
    - **Atendimento ao DoD:** Executado com sucesso no hardware CUDA convergindo ao ótimo (3323.0); `ruff check benchmark_scripts` limpo.

- [x] **Task 9.11: Limpeza de Tipagem e Docstring do `config_loader`**
  + **What:** Substituir `params: dict[str, Any] = None  # type: ignore[assignment]` e `strategies` por `field(default_factory=dict)`, e remover a docstring que ainda cita `default_config_dir`.
  + **Why:** Os dois `type: ignore` escondem um contrato de atributo que nunca é `None` após `__post_init__`; a docstring documenta API removida na Task 8.3.
  + **Who:** Agente (`implementer`).
  + **Where:** [config_loader.py](../../src/benchmarking/runner/config_loader.py#L1-L20), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L160-L180).
  + **When:** Independente.
  + **How:** Usar `dataclasses.field(default_factory=dict)`, remover os comentários de supressão e reescrever o exemplo de uso com `load_algorithms_config`/`load_all_configs` e caminho explícito.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Nenhum.
  + **DoD:** `grep -n "type: ignore" src/benchmarking/runner/config_loader.py` retorna 0; `mypy src/benchmarking` limpo; `grep -n "default_config_dir" src` retorna 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [config_loader.py](../../src/benchmarking/runner/config_loader.py#L10-L28), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L172-L177).
    - **Resumo:** `params` e `strategies` em `AlgorithmConfig` passaram a usar `dc_field(default_factory=dict)`, eliminando os dois `type: ignore`. A docstring do módulo foi atualizada com caminho explícito `Path("configs")`.
    - **Atendimento ao DoD:** `grep -n "type: ignore"` e `grep -n "default_config_dir"` retornam 0; `mypy src/benchmarking` limpo.

- [x] **Task 9.12: Documentação da Fase 9**
  + **What:** Atualizar o walkthrough da suíte e este plano com a narrativa corrigida (capacidade no núcleo, archive isolado, contrato de configs) e preencher os blocos Solution das tarefas executadas.
  + **Why:** A documentação atual descreve `tests/gpu_probe.py` como local definitivo da capacidade e o monolítico como parte da API pública.
  + **Who:** Agente (`implementer`).
  + **Where:** [analysis-test-suite-walkthrough.md](../../documentation/walkthroughs/analysis-test-suite-walkthrough.md), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md).
  + **When:** Última tarefa da fase.
  + **How:** Registrar evidências de verificação, decisões DP9-1 a DP9-7 e links para os arquivos alterados.
  + **How much:** Média.
  + **Stakeholder Implications:** Leitor da tese encontra a arquitetura real, não a anterior.
  + **DoD:** Walkthrough e Handoff descrevem o estado pós-fase; nenhuma referência a `tests/gpu_probe.py` como API.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [analysis-test-suite-walkthrough.md](../../documentation/walkthroughs/analysis-test-suite-walkthrough.md#L250-L280), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md#L628-L790).
    - **Resumo:** Registrada a Seção 10.8 no walkthrough da suíte de testes documentando as decisões e números de Fase 9 (711 passed, 6 xfailed). Preenchidos todos os blocos Solution com links relativos e atualizado o bloco Handoff.
    - **Atendimento ao DoD:** Documentação sincronizada com o código e testes pós-Fase 9.

---

## Fase 10: Registro de Defeitos, Identificadores e Gate de Documentação

### Decisões Fixadas da Fase 10

- **DP10-1 (Identificadores)**: `DP-<n>` para decisões do plano, `DP<fase>-<n>` para decisões de fase, `BUG-<n>` para defeitos e `TASK <fase>.<n>` para tarefas. Toda definição vive em [table-of-symbols.plan.md](./table-of-symbols.plan.md); nenhum documento usa `D<n>` isolado.
- **DP10-2 (Archive)**: as variantes monolíticas saem do `configs/algorithms.json` ativo e passam a viver em [algorithms.archive.json](../../configs/algorithms.archive.json), carregado apenas sob demanda via `load_algorithms_config(..., config_name="algorithms.archive.json")`.
- **DP10-3 (Nome único)**: `FullGPUMonolithic` é o único nome da variante monolítica; `FullGPU` foi eliminado de configs, código e testes.
- **DP10-4 (Gate de documentação)**: [check_docs.py](../../benchmark_scripts/check_docs.py) é o verificador standalone de docstrings obsoletas, links mortos e símbolos não registrados.
- **DP10-5 (Coortes de resultados)**: checkpoints de schemas diferentes não são combinados; `results_v3/selected_10/checkpoints/` mantém apenas a coorte de 41 chaves.
- **DP10-6 (Substituição de `tsplib95` e ETL duplicado por `routing_data`)**: `tsplib_loader.py` era temporário; `gpu_accelerated_clean` delegará o parsing de problemas e a criação/ingestão do banco DuckDB para `routing_data` (`tsplib_parser` + `converter`), removendo a dependência `tsplib95` e o DDL duplicado em [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L418-L660).
- **DP10-7 (Dependência via tag de release Git e versão do DuckDB)**: `routing-data` será consumido via tag de release Git (`uv add "routing-data @ git+https://github.com/lucasgaldinos/Routing_data.git@v0.2.0"`) após o commit das correções em `routing_data`, e a restrição de versão do DuckDB em `routing_data/pyproject.toml` será relaxada (`duckdb>=1.4.0`).
- **DP10-8 (Contratos tipados em `distances`, `loaders` e `protocols.backend`)**: Substituir `Callable[[Any, Any], int]` em [pairwise.py](../../src/distances/pairwise.py#L460) por um `Protocol` (`PairwiseDistanceFn`), substituir `dict[str, Any]` em [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L30-L41) por `TypedDict` (`ProblemMetadataRow`), e eliminar o monkey-patching `# type: ignore` de `cache_clear` e assinaturas `*args: Any, **kwargs: Any` em [backend.py](../../src/protocols/backend.py#L120-L228).
- **DP10-9 (Correções aprovadas em `routing_data`)**:
  + `B1`: implementar upsert em `insert_problem` (`ON CONFLICT (name, type) DO UPDATE` / caminho explícito de atualização);
  + `B3`: isolar falhas por problema em `insert_problems_batch` para que problemas válidos sejam comitados e inválidos registrados em `failed[]`;
  + `M4`: levantar erro em `transform_problem` quando a matriz `EXPLICIT` falhar e descartar o problema inteiro no worker (nunca inserir problema `EXPLICIT` sem matriz no DuckDB);
  + `M7`: derivar `is_symmetric` para `FULL_MATRIX` a partir dos valores da matriz via `check_symmetry`;
  + `M11`: implementar verificações reais de integridade SQL e dimensões em `converter validate` com saída não-zero (`exit 1`) em caso de falha;
  + `M2`/`m17`: sanitizar nomes em `json_writer.py` usando o padrão `<problem_name>.<problem_type>.json` (ex.: `lucas32.tsp.json`, `lucas32.vrp.json`), evitar re-parsing/inserção de arquivos duplicados com mesmo `(name, type)` e conteúdo, e validar nomes de tabela contra whitelist em `parquet_writer.py`;
  + `T1`–`T5` / Batch 4: renomear `DataTransformer` para `ProblemNormalizer` com retorno `TypedDict`, e remover/consolidar os scripts raiz `convert_vrp.py` e `converter_cli.py`.
- **DP10-10 (Higiene de documentação e scripts temporários)**: Deletar os 4 arquivos obsoletos em `routing_data/tmp/` (`src_exploration_report.md`, `analysis-test-suite-organization.md`, `baseline-durations.txt`, `batch3-durations.txt`), deletar os 9 scripts one-shot de migração de testes em `routing_data/tmp/*.py` e refatorar os 4 scripts `probe_*.py` em exemplos limpos após estabilização da API; deletar os 5 arquivos Markdown obsoletos/duplicados em `gpu_accelerated_clean` (`daily_planning.md`, `meeting-notes.md`, `modularizing-cuda-kernels.md`, `documentation/analysis/presentations/key_findings.md`, `documentation/critical_analysis/criticas_analysis_bib.md`); atualizar as âncoras e caminhos obsoletos listados nas Seções 5.B e 5.D após as tarefas de código.
- **DP10-11 (Verificação limpa de arquivo ausente em `DatabaseLoader`, `BUG-16`)**: Mover a verificação de existência de `self.db_path` para fora do bloco `try` em [database_loader.py](../../src/loaders/database_loader.py#L92-L111) e levantar `DatabaseConnectionError` diretamente sem auto-capturar `FileNotFoundError`.
- **DP10-12 (Manutenção de `INTEGER[][]` e `int32` segundo TSPLIB95 e registro de escalabilidade futura, `M3`)**: Manter `edge_weight_matrices.matrix` como `INTEGER[][]` no DuckDB e `int32` em todo o pipeline GPU/CPU aplicando arredondamento TSPLIB95 (`_nint`), e documentar os trade-offs futuros de precisão/escala em [analysis-future-scaling-and-type-decisions.md](../../documentation/analysis/questions/analysis-future-scaling-and-type-decisions.md).
- **DP10-13 (Fronteira de funções de distância, Item 1.3)**: As funções escalares TSPLIB95 (`_nint`, `EUC_2D`, `GEO`, `ATT`, etc.) residem em `routing_data` e são importadas/reexportadas por [pairwise.py](../../src/distances/pairwise.py); o construtor de matrizes `(n, n)` vetorizado sobre `xp` (`numpy`/`cupy`) permanece em [matrix.py](../../src/distances/matrix.py).
- **DP10-14 (Fronteira `DatabaseManager` vs `DatabaseLoader`, Item 1.4)**: `DatabaseManager` em `routing_data` detém 100% do DDL e das queries SQL sobre o banco de problemas (`problems`, tabelas por tipo, `edge_weight_matrices`, `file_tracking`, `solutions`); [database_loader.py](../../src/loaders/database_loader.py) em `gpu_accelerated_clean` consome esses dados e constrói objetos `Problem` com arrays `np.ndarray` (`int32` para distâncias/demandas, `float64` para coordenadas).
- **DP10-15 (Padronização de `ProblemContext.rng` em `np.random.Generator`, `BUG-2`)**: `ProblemContext.__init__` instancia `self.rng: Generator = make_rng(seed)` (`np.random.default_rng`), eliminando `RandomState` do contexto para que `resolve_rng(context, fallback)` sempre utilize o gerador semeado do contexto.
- **DP10-16 (Cálculo de matriz de distâncias em blocos para `n` grande, `BUG-13`)**: [matrix.py](../../src/distances/matrix.py#L126-L206) mantém broadcast 2D completo para `n <= 2048` e utiliza broadcast em blocos de linhas (`chunk_size = 512`) para `n > 2048` (reduzindo o pico de RAM de `O(n^2 d)` float64 sem o overhead de `triu_indices`), compartilha constantes com `pairwise.py` e atualiza a docstring.
- **DP10-17 (Cache compartilhado de matriz `int32` em GPU no `ProblemContext`, `BUG-14`)**: `ProblemContext.get_gpu_distances()` garante retorno `cp.int32` cacheado no contexto (e `DatabaseLoader._load_distance_matrix` carrega matrizes `EXPLICIT` como `np.int32`); as estratégias `BatchGPUCostEvaluation`, `BatchGPUTwoOpt` e `BatchGPUMultiMoveTwoOpt` delegam a `context.get_gpu_distances()` e usam a chave composta `(problem.name, problem.dimension, problem.edge_type, id(problem))` como fallback.
- **DP10-18 (Remoção de `Stream.null.synchronize()` redundante em 2-opt GPU, `BUG-15`)**: Remover `cp.cuda.Stream.null.synchronize()` incondicional em [two_opt.py#L358](../../src/algorithms/strategies/local_search/two_opt.py#L358) e [two_opt.py#L637](../../src/algorithms/strategies/local_search/two_opt.py#L637) quando `input_on_gpu=True`, pois transferências D2H (`d_population.get()` / `cp.asnumpy()`) e eventos CUDA em `@time_strategy` já sincronizam a stream quando necessário.
- **DP10-19 (Kernels CUDA de distância par-a-par on-the-fly para `n > 10.000`, tiling de memória compartilhada para `n > 11.520` e validação de VRAM)**: Para `n <= 10.000` (ou `EXPLICIT`), mantém-se a matriz densa `(n, n)` `int32` na VRAM; para `n > 10.000` com coordenadas (`EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`, `CEIL_2D`, `ATT`, `GEO`), os kernels CUDA calculam distâncias on-the-fly em registradores via `src/distances/kernels/pairwise_device.cuh` *(novo, Task 10.15)* lendo apenas `coords` (cacheado em L1/L2); para `n > 11.520`, os kernels de 2-opt operam com tiling em memória compartilhada (`TILE_SIZE = 2048`, ~8 KB) e carry-over de registrador (`node_j = node_j1`) para não exceder o limite rígido de 48 KB por bloco; `ProblemContext` valida a memória exigida ($\mathrm{VRAM}_{\mathrm{total}}(n, P) = 4n^2 + 12Pn + 24P + 8n$) contra `cp.cuda.Device().mem_info` e levanta [VRAMInsufficientError](../../src/data_models/exceptions.py#L210) quando insuficiente.
- **DP10-20 (Lista de candidatos $k$-Nearest Neighbor $\mathcal{N}_k(u)$ compartilhada para CPU e GPU)**: `ProblemContext` expõe `get_cpu_candidate_list(k: int = 20)` e `get_gpu_candidate_list(k: int = 20)` retornando a matriz estática `(n, min(k, n - 1))` `int32` dos $k$ vizinhos mais próximos de cada nó (Reinelt, 1994; Johnson & McGeoch, 1997; Toth & Vigo, 2003); essa estrutura alimenta as variantes de 2-opt com lista de candidatos em CPU e GPU ($\Theta(n \cdot k)$ via permutação inversa $\text{pos}(\pi_i) = i$) e permanece disponível para operadores futuros (3-opt, Or-opt, vizinhanças granulares CVRP, inicialização GRASP e mutação guiada).

### Tarefas da fase 10

- [x] **Task 10.1: Tabela de Símbolos e Unificação dos Identificadores**
  + **What:** Criar `table-of-symbols.plan.md` e renomear `D1`–`D5` do plano para `DP-1`–`DP-5` e `D1`–`D8` dos walkthroughs para `BUG-1`–`BUG-8`.
  + **Why:** Três esquemas incompatíveis de `D` coexistiam — decisões no plano, defeitos no walkthrough da suíte e uma terceira ordem no checkpoint da Fase 9 — tornando instruções como "corrija D3" ambíguas.
  + **Who:** Agente (`implementer`).
  + **Where:** [table-of-symbols.plan.md](./table-of-symbols.plan.md), este plano, [analysis-test-suite-walkthrough.md](../../documentation/walkthroughs/analysis-test-suite-walkthrough.md), [analysis-phase-9-gpu-capability-walkthrough.md](../../documentation/walkthroughs/analysis-phase-9-gpu-capability-walkthrough.md), [analysis-phase-7-residency-rng-walkthrough.md](../../documentation/walkthroughs/analysis-phase-7-residency-rng-walkthrough.md), [analysis-phases-1-6-reorganization-walkthrough.md](../../documentation/walkthroughs/analysis-phases-1-6-reorganization-walkthrough.md).
  + **When:** Primeira tarefa da fase.
  + **How:** Registrar `DP-*`, `DP9-*` e `BUG-*` na tabela; aplicar substituição mecânica preservando a ordem `D<n>` → `BUG-<n>`; corrigir os caminhos de teste citados na tabela de defeitos.
  + **How much:** Médio.
  + **Stakeholder Implications:** Qualquer "corrija BUG-3" resolve para um único defeito e uma única evidência verificável.
  + **DoD:** `check_docs.py` reporta 0 símbolos não registrados; nenhum `D<n>` isolado permanece fora da tabela.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [table-of-symbols.plan.md](./table-of-symbols.plan.md#L1-L120), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md#L65-L71), [analysis-test-suite-walkthrough.md](../../documentation/walkthroughs/analysis-test-suite-walkthrough.md#L119-L124), [analysis-phase-9-gpu-capability-walkthrough.md](../../documentation/walkthroughs/analysis-phase-9-gpu-capability-walkthrough.md#L210).
    - **Resumo:** Criada a tabela de símbolos com os namespaces `DP`, `DP<fase>` e `BUG`, a tabela de defeitos abertos, os resolvidos e o mapa de migração. Renomeadas 5 decisões, 8 defeitos e 6 referências cruzadas; corrigidos três caminhos de teste obsoletos na tabela de defeitos (`test_unit_problem_invariants.py`, `test_unit_statistics.py`, `test_unit_strategies.py`).
    - **Atendimento ao DoD:** `check_docs.py` → `undefined symbols: 0`.
- [x] **Task 10.2: Separação do Config de Archive e Eliminação de `FullGPU`**
  + **What:** Mover a variante monolítica para um arquivo de configuração próprio, mantendo um único nome (`FullGPUMonolithic`), e atualizar os ~23 sítios que citavam `FullGPU`.
  + **Why:** `FullGPU` e `FullGPUMonolithic` tinham mesma classe, mesmos parâmetros e diferiam apenas na descrição, produzindo checkpoints duplicados em `results_v3/selected_10/`.
  + **Who:** Agente (`implementer`).
  + **Where:** [algorithms.archive.json](../../configs/algorithms.archive.json), [algorithms.json](../../configs/algorithms.json), [fixture algorithms.json](../../tests/fixtures/configs/algorithms.json), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L365-L380), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L38).
  + **When:** Após a Task 10.1.
  + **How:** Criar `algorithms.archive.json`; adicionar `config_name` a `load_algorithms_config`; remover as duas entradas do config ativo e do fixture; renomear o rótulo nos testes e exemplos; remover o filtro `"archive" in cfg.class_module`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Benchmarks deixam de emitir linhas duplicadas; o baseline da tese continua acessível por caminho explícito.
  + **DoD:** `grep -rn '"FullGPU"' src tests configs` retorna 0; `pytest` verde.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithms.archive.json](../../configs/algorithms.archive.json#L1-L20), [algorithms.json](../../configs/algorithms.json), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L365-L380), [test_unit_config_loader.py](../../tests/benchmarking/test_unit_config_loader.py#L67), [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L36-L46), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L38).
    - **Resumo:** Config ativo passa a listar 4 variantes; o monolítico vive em `algorithms.archive.json` e é carregado por `config_name`. `FullGPU` removido de configs, `problem_statistics.py`, `schema.sql`, `db_operations.py`, `QUICKSTART.md` e dos testes. O bloco que testava overrides do monolítico no pipeline foi substituído por uma nota, pois a suíte não executa archive.
    - **Atendimento ao DoD:** `grep -rn '"FullGPU"' src tests configs` → 0; `pytest tests -q` → 711 passed, 6 xfailed.
- [x] **Task 10.3: Reparo dos DoDs 9.6/9.9 e das Âncoras Mortas**
  + **What:** Reescrever os dois critérios que não podiam ser satisfeitos e eliminar as 23 referências a arquivos inexistentes neste plano e no `plan-roteiro.md`.
  + **Why:** O DoD 9.6 pedia `grep "monolithic"` = 0 enquanto o próprio Solution admitia 1 ocorrência; o DoD 9.9 afirmava que nenhum teste importava `algorithms.archive`, o que dois testes faziam; 23 links apontavam para caminhos removidos.
  + **Who:** Agente (`implementer`).
  + **Where:** este plano (Tasks 9.6, 9.9, 7.2, 1.1), [plan-roteiro.md](./plan-roteiro.md#L344-L380).
  + **When:** Após a Task 10.2.
  + **How:** Trocar o DoD 9.6 por uma asserção sobre `algorithms.__all__`; trocar o DoD 9.9 por "nenhum teste executa `evolve()` em variante de archive"; repontar caminhos movidos e anotar deleções como `*(removido na TASK x.y)*`.
  + **How much:** Médio.
  + **Stakeholder Implications:** O plano volta a ser navegável e auditável.
  + **DoD:** `check_docs.py` → `dead links: 0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md#L721-L770), [plan-roteiro.md](./plan-roteiro.md#L344-L380).
    - **Resumo:** DoD 9.6 passou a exigir `set(algorithms.__all__) == {"GeneticAlgorithm", "GeneticAlgorithmBase", "LocalSearchStage"}`; DoD 9.9 exige ausência de `evolve()` em variante de archive. Seis links foram repontados (`archive/`, `benchmarking/runner/`, `benchmarking/analysis/`, `benchmark_scripts/`) e quinze deleções anotadas; `plan-roteiro.md` recebeu o mesmo tratamento em 19 links.
    - **Atendimento ao DoD:** `check_docs.py` → `dead links: 0`.
- [x] **Task 10.4: Docstrings Obsoletas e Validador Morto**
  + **What:** Corrigir as docstrings que citavam módulos inexistentes e religar `validate_configs` aos nomes reais de parâmetro.
  + **Why:** `validate_configs` lia `tournament_size` e `two_opt_iterations`, renomeados para `k` e `max_iterations` na Task 6.1, e portanto nunca disparava; docstrings citavam `chapter4_validation.py` e `datasets/tsp.duckdb` como caminho literal.
  + **Who:** Agente (`implementer`).
  + **Where:** [config_loader.py](../../src/benchmarking/runner/config_loader.py#L551), [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L1), [database_loader.py](../../src/loaders/database_loader.py#L20), [algorithms/**init**.py](../../src/algorithms/__init__.py#L8), [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L104).
  + **When:** Após a Task 10.3.
  + **How:** Reescrever as docstrings; trocar as chaves do validador e as mensagens de aviso; derivar o título dos relatórios de `benchmark_name`; retargetar os dois testes de validação aos nomes vivos.
  + **How much:** Médio.
  + **Stakeholder Implications:** Validação que pode falhar de fato e relatórios com o título correto.
  + **DoD:** `check_docs.py` → `stale docstrings: 0`; testes de validação cobrindo `k` e `max_iterations`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [config_loader.py](../../src/benchmarking/runner/config_loader.py#L1-L12), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L551-L560), [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L16), [database_loader.py](../../src/loaders/database_loader.py#L20), [algorithms/**init**.py](../../src/algorithms/__init__.py#L7-L8), [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L94-L115), [test_unit_config_validation.py](../../tests/benchmarking/test_unit_config_validation.py#L73-L82).
    - **Resumo:** `validate_configs` passou a checar `k` e `max_iterations` com mensagens por slot; o título Markdown/LaTeX dos relatórios vem de `benchmark_name.replace("_", " ").title()`; quatro docstrings obsoletas reescritas.
    - **Atendimento ao DoD:** `pytest tests/benchmarking/test_unit_config_validation.py -q` verde; `check_docs.py` → `stale docstrings: 0`.
- [x] **Task 10.5: Gate de Documentação Standalone**
  + **What:** Criar `benchmark_scripts/check_docs.py` verificando docstrings obsoletas, links mortos e símbolos não registrados.
  + **Why:** As três classes de deriva passaram despercebidas porque nada as verificava; o teste da suíte não lê documentação.
  + **Who:** Agente (`implementer`).
  + **Where:** [check_docs.py](../../benchmark_scripts/check_docs.py).
  + **When:** Última tarefa da fase.
  + **How:** Script em python puro, sem `argparse`, com listas explícitas no topo, `ast` para docstrings, regex para links e para identificadores; códigos de saída 0/1/2.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Qualquer sessão futura detecta deriva de documentação com um comando.
  + **DoD:** `python -m benchmark_scripts.check_docs` retorna 0 no estado atual e 1 quando um link é quebrado ou um símbolo é removido da tabela.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [check_docs.py](../../benchmark_scripts/check_docs.py#L1-L150).
    - **Resumo:** Criado o gate com três verificações independentes e saída agregada por categoria; restrito a `documents_tcc/plan/*.md` e `documentation/walkthroughs/*.md` para evitar falsos positivos de links externos.
    - **Atendimento ao DoD:** Execução atual → `stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0`, exit 0.
- [x] **Task 10.6: Corrigir `BUG-1` (`Problem.__eq__` e `hash()` sobre `ndarray`)**
  + **What:** Implementar comparação por conteúdo (`np.array_equal`) e igualdade segura para campos `np.ndarray | None` na dataclass `Problem`.
  + **Why:** Como `Problem` é `@dataclass(frozen=True)` contendo arrays NumPy, `p1 == p2` levanta `ValueError: The truth value of an array with more than one element is ambiguous` e `hash(p1)` levanta `TypeError: unhashable type: 'numpy.ndarray'`.
  + **Who:** Agente (`implementer`).
  + **Where:** [problem.py](../../src/data_models/problem.py#L30-L135), [test_unit_problem_validation.py](../../tests/data_models/test_unit_problem_validation.py#L124).
  + **When:** Independente; pode rodar na primeira bateria de correções de defeitos unitários de `gpu_accelerated_clean`.
  + **How:** Implementar `__eq__` comparando escalares diretamente e arrays via `np.array_equal` (tratando `None`), e remover o marcador `@pytest.mark.xfail(strict=True)` em `test_problems_with_equal_content_compare_equal`.
  + **How much:** Baixo (~25 linhas).
  + **Stakeholder Implications:** Instâncias `Problem` com mesmos dados podem ser comparadas diretamente em testes e caches sem erro de broadcast NumPy.
  + **DoD:** `pytest tests/data_models/test_unit_problem_validation.py -k "test_problems_with_equal_content_compare_equal" -q` passa sem `xfail`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem.py](../../src/data_models/problem.py#L26-L40), [test_unit_problem_validation.py](../../tests/data_models/test_unit_problem_validation.py#L124).
    - **Resumo:** `Problem` ganhou `__eq__` (escalares por igualdade, arrays por `np.array_equal`, `None` só igual a `None`) e `__hash__` derivado do conteúdo dos arrays, preservando o uso da dataclass `frozen` em sets/dicts.
    - **Atendimento ao DoD:** `pytest ... -k "test_problems_with_equal_content_compare_equal" -q` → `1 passed`; `ruff`/`mypy` nos arquivos tocados sem erros.
- [x] **Task 10.7: Corrigir `BUG-2` (Padronizar `ProblemContext.rng` em `np.random.Generator`, `DP10-15`)**
  + **What:** Substituir `xp.random.RandomState(seed)` por `make_rng(seed)` (`np.random.Generator`) em `ProblemContext.__init__` e garantir em `resolve_rng(context, fallback)` que `context.rng` seja retornado diretamente.
  + **Why:** `ProblemContext.__init__` criava `xp.random.RandomState(seed)` ([problem_context.py](../../src/protocols/problem_context.py#L166)), enquanto todas as 7 estratégias e `resolve_rng` ([rng.py](../../src/utils/rng.py#L57)) operam sobre `np.random.Generator` (`.integers()`), fazendo `resolve_rng` ignorar silenciosamente `context.rng`.
  + **Who:** Agente (`implementer`).
  + **Where:** [problem_context.py](../../src/protocols/problem_context.py#L160-L175), [rng.py](../../src/utils/rng.py#L43-L60), [test_unit_rng.py](../../tests/utils/test_unit_rng.py#L145-L180), [test_unit_problem_context.py](../../tests/protocols/test_unit_problem_context.py#L100-L120).
  + **When:** Independente; primeira bateria de correções unitárias em `gpu_accelerated_clean`.
  + **How:**
    1. Em `src/protocols/problem_context.py`, importar `make_rng` de `utils.rng` e definir `self.rng: Generator = make_rng(seed)`.
    2. Em `tests/protocols/test_unit_problem_context.py:114-116`, atualizar `first.rng.random_sample(8)` para `first.rng.random(8)`.
    3. Em `tests/utils/test_unit_rng.py:163`, atualizar `test_resolve_rng_uses_seeded_problem_context_rng` para usar `ProblemContext` real (ou `make_rng(42)`) e remover `@pytest.mark.xfail(strict=True)`.
  + **How much:** Baixo (~25 linhas).
  + **Stakeholder Implications:** Execuções com mesma semente em `ProblemContext` produzem trajetórias determinísticas e reprodutíveis em todas as estratégias CPU e GPU.
  + **DoD:** `pytest tests/utils/test_unit_rng.py tests/protocols/test_unit_problem_context.py -q` passa integralmente sem `xfail`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_context.py](../../src/protocols/problem_context.py#L168), [rng.py](../../src/utils/rng.py#L33-L46), [types.py](../../src/protocols/types.py#L17), [test_unit_rng.py](../../tests/utils/test_unit_rng.py#L163), [test_unit_problem_context.py](../../tests/protocols/test_unit_problem_context.py#L114).
    - **Resumo:** `ProblemContext.rng` agora é `make_rng(seed)` (`np.random.Generator`); `SeedLike` e `make_rng` aceitam sequências de inteiros (sementes por worker); testes usam `.random(8)` e um `Generator` real.
    - **Atendimento ao DoD:** `pytest tests/utils/test_unit_rng.py tests/protocols/test_unit_problem_context.py -q` → `22 passed`; `mypy`/`ruff` limpos.
- [x] **Task 10.8: Corrigir `BUG-3` (`mean_difference` e IC em Variância Zero)**
  + **What:** Preservar a diferença média real `mean_diff = float(np.mean(differences))` e `ci_95_difference = (mean_diff, mean_diff)` quando `std_diff == 0` em `paired_comparison`.
  + **Why:** O ramo de variância zero em [statistics.py](../../src/benchmarking/analysis/statistics.py#L220) força `mean_difference=0.0` e `ci_95_difference=(0.0, 0.0)` mesmo quando dois algoritmos diferem por uma constante exata não nula (ex.: `[10, 10, 10]` vs `[15, 15, 15]`).
  + **Who:** Agente (`implementer`).
  + **Where:** [statistics.py](../../src/benchmarking/analysis/statistics.py#L215-L245), [test_unit_statistics_comparison.py](../../tests/benchmarking/test_unit_statistics_comparison.py#L173).
  + **When:** Independente; bateria de correções unitárias.
  + **How:** Calcular `mean_diff` antes do ramo `std_diff == 0`; quando `mean_diff == 0.0`, manter `p_value = 1.0`, `effect_size = 0.0`; quando `mean_diff != 0.0` e `std_diff == 0.0`, reportar `mean_diff`, `ci_95_difference = (mean_diff, mean_diff)` e tratar o valor-p/tamanho de efeito sem divisão por zero; remover o `xfail(strict=True)`.
  + **How much:** Baixo (~25 linhas).
  + **Stakeholder Implications:** Comparações estatísticas em instâncias onde ambos os algoritmos têm variância zero (ex.: convergem sempre para o mesmo ótimo local de cada variante) reportam a diferença real de custo.
  + **DoD:** `test_paired_comparison_zero_variance_reports_true_mean_difference` passa sem `xfail`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [statistics.py](../../src/benchmarking/analysis/statistics.py#L213-L242), [test_unit_statistics_comparison.py](../../tests/benchmarking/test_unit_statistics_comparison.py#L173).
    - **Resumo:** `mean_diff` é calculado antes do ramo de variância zero e reportado em `mean_difference`/`ci_95_difference`; quando a diferença é nula mantém `p_value=1.0`, quando é constante não nula usa `p_value=0.0` e `effect_size=0.0` (variância nula), sem divisão por zero.
    - **Atendimento ao DoD:** `pytest ... -k "test_paired_comparison_zero_variance_reports_true_mean_difference" -q` → `1 passed`; arquivo completo `14 passed`.
- [x] **Task 10.9: Corrigir `BUG-4` (Validação do Comprimento de `raw_*` em `is_valid_checkpoint`)**
  + **What:** Verificar em `is_valid_checkpoint` que os comprimentos das listas `raw_times`, `raw_costs`, `raw_gaps` (e demais `raw_*` obrigatórios) coincidem com `successful_runs`.
  + **Why:** `is_valid_checkpoint` ([checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L266)) aceita checkpoints com `successful_runs = 5` e `raw_costs = [100.0]` (tamanho 1), embora `can_resume_checkpoint` rejeite essa mesma inconsistência.
  + **Who:** Agente (`implementer`).
  + **Where:** [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L266-L315), [test_unit_checkpoint_io.py](../../tests/benchmarking/test_unit_checkpoint_io.py#L240).
  + **When:** Independente; bateria de correções unitárias.
  + **How:** Adicionar checagem de comprimento das listas `raw_*` contra `successful_runs` (ou `repetitions`) em `is_valid_checkpoint` e remover o `xfail(strict=True)` em `test_validity_rejects_arrays_inconsistent_with_successful_runs`.
  + **How much:** Baixo (~20 linhas).
  + **Stakeholder Implications:** Checkpoints truncados ou corrompidos são rejeitados na leitura antes de contaminar o banco de resultados.
  + **DoD:** `test_validity_rejects_arrays_inconsistent_with_successful_runs` passa sem `xfail`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L266-L288), [test_unit_checkpoint_io.py](../../tests/benchmarking/test_unit_checkpoint_io.py#L240).
    - **Resumo:** `is_valid_checkpoint` agora exige que `raw_times`, `raw_costs`, `raw_gaps` e `raw_seeds` tenham o mesmo comprimento e que esse comprimento iguale `successful_runs`, alinhando a regra de `can_resume_checkpoint`.
    - **Atendimento ao DoD:** `pytest ... -k "test_validity_rejects_arrays_inconsistent_with_successful_runs" -q` → `1 passed`; arquivo completo `17 passed`.
- [x] **Task 10.10: Corrigir `BUG-5` e `BUG-6` (Proteção de Endpoints em `_mutation_bounds` para `n == 3`)**
  + **What:** Extrair um helper compartilhado `_mutation_bounds(n: int)` e corrigir a condição de proteção de endpoints/depots para rotas pequenas (`n <= 3`).
  + **Why:** Em [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L48) e [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L48), `_mutation_bounds` protege `0` e `n - 1` apenas quando `n > 3`; para uma rota `[depot, cliente, depot]` (`n == 3`), retorna `(0, 3)` e permite mutar o depósito de lugar.
  + **Who:** Agente (`implementer`).
  + **Where:** [route_helpers.py](../../src/utils/route_helpers.py), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L48), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L48), [test_unit_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L76), [test_unit_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py#L77).
  + **When:** Independente; bateria de correções unitárias.
  + **How:** Centralizar `_mutation_bounds` em `route_helpers.py` (ou `mutation/__init__.py`), preservar os endpoints `[0]` e `[n-1]` para `n >= 3` (tornando `n <= 3` no-op interior pois só há 1 cliente interno), e remover os dois `xfail(strict=True)`.
  + **How much:** Baixo (~25 linhas).
  + **Stakeholder Implications:** Rotas CVRP/TSP com 3 posições nunca têm seus endpoints/depósitos corrompidos por mutação.
  + **DoD:** `pytest tests/algorithms/strategies/mutation -q` passa sem `xfail`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [route_helpers.py](../../src/utils/route_helpers.py#L32-L41), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L8-L14), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L8-L14), [test_unit_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L76), [test_unit_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py#L77).
    - **Resumo:** `_mutation_bounds(n)` centralizado em `route_helpers` retorna `(1, n - 1)` (endpoints `0` e `n - 1` preservados); para `n <= 3` a fatia tem menos de 2 posições e a mutação vira no-op. As cópias locais foram removidas das duas estratégias.
    - **Atendimento ao DoD:** `pytest tests/algorithms/strategies/mutation -q` → `26 passed`; `ruff`/`mypy` limpos.
- [x] **Task 10.11: Correções de Integridade, Contratos e Empacotamento em `routing_data` (`DP10-7`, `DP10-9`, `DP10-12`, `DP10-13`, `DP10-14`)**
  + **What:** Preparar `routing_data/src/` para a release `v0.2.0`:
    1. Relaxar `duckdb>=1.4.0` em `routing_data/pyproject.toml` (`DP10-7`);
    2. Implementar upsert em `DatabaseManager.insert_problem` (`B1`) e isolamento por problema (savepoint/validação prévia dentro da transação única) em `insert_problems_batch` (`B3`);
    3. Manter `edge_weight_matrices.matrix` como `INTEGER[][]` aplicando arredondamento TSPLIB95 `_nint` para pesos explícitos (`M3`, `DP10-12`);
    4. Levantar erro em `transform_problem` quando `_convert_edge_weights_to_matrix` falhar em problema `EXPLICIT`, descartando o problema inteiro no worker sem inserir linha órfã no banco (`M4`);
    5. Derivar `is_symmetric` em `FormatParser` usando `check_symmetry` sobre os valores da matriz quando `EDGE_WEIGHT_FORMAT == "FULL_MATRIX"` (`M7`);
    6. Implementar checagens SQL reais em `converter validate` (joins hub↔tipo, presença/dimensão de matriz `EXPLICIT`, dimensões de coordenadas e FKs de `solutions`) com `sys.exit(1)` em falha (`M11`);
    7. Em `json_writer.py`, gravar arquivos como `<problem_name>.<problem_type>.json` (ex.: `berlin52.tsp.json`) com sanitização por regex e pular re-parsing/inserção de arquivos duplicados com mesmo `(name, type)` e conteúdo; em `parquet_writer.py`, validar nomes de tabela contra `ALLOWED_TABLES` (`M2`/`m17`);
    8. Renomear `DataTransformer` para `ProblemNormalizer` com retorno `TypedDict` (`T1`–`T5`), expor as funções escalares TSPLIB95 de distância (`DP10-13`) e métodos de leitura do `DatabaseManager` (`DP10-14`), e remover/consolidar `convert_vrp.py` e `converter_cli.py`.
  + **Why:** `routing_data` é a fonte única de verdade para parsing TSPLIB e geração/consulta do banco DuckDB (`DP10-6`); corrigir esses itens e gerar a tag `v0.2.0` permite que `gpu_accelerated_clean` instale apenas o pacote limpo via `uv add`.
  + **Who:** Agente (`implementer`).
  + **Where:** `routing_data/pyproject.toml`, `routing_data/src/converter/database/operations.py`, `routing_data/src/tsplib_parser/parser.py`, `routing_data/src/converter/core/transformer.py`, `routing_data/src/converter/utils/cost.py`, `routing_data/src/converter/cli/commands.py`, `routing_data/src/converter/output/json_writer.py`, `routing_data/src/converter/output/parquet_writer.py`, `routing_data/convert_vrp.py`, `routing_data/converter_cli.py`.
  + **When:** Antes da `Task 10.12`. (Pode ser dividida em sub-batches internos ao executar em `routing_data`).
  + **How:** Implementar cada item com seu respectivo teste de regressão em `routing_data/tests/`, verificar `pytest` e `mypy src`, realizar commit e criar a tag Git `v0.2.0`.
  + **How much:** Médio/Alto.
  + **Stakeholder Implications:** `routing_data` torna-se um pacote instalável determinístico, idempotente e sem falhas silenciosas.
  + **DoD:** `pytest` e `mypy src` verdes em `routing_data`; tag `v0.2.0` pronta para consumo via `uv add`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** `routing_data/pyproject.toml#L10`, `routing_data/src/converter/database/operations.py#L520`, `routing_data/src/converter/core/transformer.py#L20`, `routing_data/src/tsplib_parser/parser.py#L543`, `routing_data/src/converter/cli/commands.py#L240`, `routing_data/src/converter/output/json_writer.py#L15`, `routing_data/src/converter/output/parquet_writer.py#L12`, `routing_data/src/converter/utils/cost.py#L300`.
    - **Resumo:** `routing_data` atualizado para `v0.2.0` com `duckdb>=1.4.0` (`DP10-7`), upsert idempotente em `_upsert_problem_row` preservando `problems.id` (`B1`), transações isoladas por problema em `insert_problems_batch` (`B3`), arredondamento TSPLIB95 `_nint` para matrizes `INTEGER[][]` (`M3`, `DP10-12`), erro explícito `TransformError` em falha de matriz `EXPLICIT` (`M4`), simetria real de `FULL_MATRIX` via `check_symmetry` (`M7`), validação SQL executável com `integrity_failures()` e `sys.exit(1)` (`M11`), arquivos `<name>.<type>.json` sanitizados com supressão de duplicatas (`M2`), whitelist `ALLOWED_TABLES` em Parquet (`m17`), `ProblemNormalizer` com `NormalizedProblemData` (`TypedDict`), exportação pública das funções escalares de distância em `cost.py` (`DP10-13`), e métodos de leitura `list_problem_metadata`/`load_problem_metadata`/`load_matrix_metadata`/`load_solution` em `DatabaseManager` (`DP10-14`). Scripts raiz `convert_vrp.py` e `converter_cli.py` removidos.
    - **Atendimento ao DoD:** Revisado em 5+1 eixos (`code-review-and-quality`): `pytest` → `425 passed, 1 xfailed`; `mypy src` → `0 errors (38 files)`; `ruff check src tests` → `All checks passed!`; commit `7a46960` e tag `v0.2.0` enviados para `origin`.
- [x] **Task 10.12: Instalação de `routing-data` e Substituição de `tsplib95` e DDL Duplicado em `gpu_accelerated_clean` (`DP10-6`, `DP10-7`, `DP10-13`, `DP10-14`)**
  + **What:** Adicionar `routing-data @ git+https://github.com/lucasgaldinos/Routing_data.git@v0.2.0` em [pyproject.toml](../../pyproject.toml), remover `tsplib95`, reexportar as funções escalares de distância a partir de `routing_data` em [pairwise.py](../../src/distances/pairwise.py), e refatorar [tsplib_loader.py](../../src/loaders/tsplib_loader.py), [database_loader.py](../../src/loaders/database_loader.py) e [cli/tsplib.py](../../src/cli/tsplib.py) para delegar o parsing (`FormatParser`) e o gerenciamento/consulta DuckDB (`DatabaseManager`) ao `routing_data`.
  + **Why:** Elimina ~250 linhas de DDL e SQL duplicados em `tsplib_loader.py`, elimina a duplicação matemática em `pairwise.py` e remove a dependência externa `tsplib95`.
  + **Who:** Agente (`implementer`).
  + **Where:** [pyproject.toml](../../pyproject.toml), [pairwise.py](../../src/distances/pairwise.py), [tsplib_loader.py](../../src/loaders/tsplib_loader.py), [database_loader.py](../../src/loaders/database_loader.py), [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py), [cli/tsplib.py](../../src/cli/tsplib.py), [test_unit_problem_type_handlers.py](../../tests/loaders/test_unit_problem_type_handlers.py), [test_integration_loaders.py](../../tests/loaders/test_integration_loaders.py).
  + **When:** Após a `Task 10.11`.
  + **How:**
    1. Atualizar `pyproject.toml` trocando `tsplib95>=0.6.1` por `routing-data` e adicionando `converter` e `tsplib_parser` às configurações de `mypy`/`ruff`.
    2. Em `src/loaders/tsplib_loader.py`, substituir `tsplib95.load_problem` por `tsplib_parser.parser.FormatParser` e delegar `ingest_tsplib_directory_to_duckdb` a `converter.database.operations.DatabaseManager` + `ProblemNormalizer`.
    3. Em `src/loaders/database_loader.py`, carregar matrizes `EXPLICIT` diretamente como `np.int32` (`DP10-12`) e alinhar as leituras ao `DatabaseManager`.
    4. Em `src/distances/pairwise.py`, importar/reexportar as funções escalares TSPLIB95 de `routing_data` mantendo os nomes públicos e a tabela de despacho `get_distance_function`.
  + **How much:** Médio (~150 linhas alteradas, ~250 linhas removidas).
  + **Stakeholder Implications:** `gpu_accelerated_clean` consome `routing_data` como biblioteca única de dados/ETL sem qualquer DDL de problemas próprio.
  + **DoD:** `grep -rn "import tsplib95" src tests` retorna 0; `pytest tests/loaders tests/distances -q` e `mypy src tests` passam com 0 erros; `pairwise.py` não possui aritmética própria de distâncias (`GEO` reexportado diretamente de `converter.utils.cost` após correção do truncamento de graus em `routing_data`).
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pyproject.toml](../../pyproject.toml#L29), [pairwise.py](../../src/distances/pairwise.py#L27), [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L112), [database_loader.py](../../src/loaders/database_loader.py#L99), [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L20), [cli/tsplib.py](../../src/cli/tsplib.py#L1), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L33), [test_integration_loaders.py](../../tests/loaders/test_integration_loaders.py#L18).
    - **Resumo:** `tsplib95` substituído por `routing-data @ v0.2.0`; `tsplib_loader` passou a delegar o parse a `FormatParser` + `ProblemNormalizer` e a ingestão a `DatabaseManager` (removendo as ~250 linhas de DDL duplicadas); `pairwise.py` reexporta as funções escalares de `converter.utils.cost`; `parse_tsplib_file` substituiu o nome público `parse_tsp_file_with_tsplib95` em `loaders`, `config_loader.py` e nos testes de integração.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "import tsplib95" src tests` → 0; [pairwise.py](../../src/distances/pairwise.py#L39) reexporta `geographical as compute_geographical` de `converter.utils.cost` (sem aritmética local de `GEO`) após a correção do truncamento de grau em `routing_data`; `pytest tests -m "not slow" -q` → `825 passed, 2 deselected`; `mypy src tests` → `118 source files` sem erros.
- [x] **Task 10.13: Contratos Tipados Explícitos em `distances`, `loaders` e `protocols.backend` (`DP10-8`)**
  + **What:** Substituir `Callable[[Any, Any], int]` por `PairwiseDistanceFn` (`Protocol`) em [pairwise.py](../../src/distances/pairwise.py#L460), substituir `dict[str, Any]` por `ProblemMetadataRow` (`TypedDict`) em [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L30-L41) e [database_loader.py](../../src/loaders/database_loader.py#L181), e refatorar [backend.py](../../src/protocols/backend.py#L120-L228) para remover `*args: Any, **kwargs: Any` desnecessários e o monkey-patching `# type: ignore` de `cache_clear`.
  + **Why:** Garante segurança de tipos estática nas fronteiras entre módulos (Eixo 6 da revisão de código), aprovada em `DP10-8`.
  + **Who:** Agente (`implementer`).
  + **Where:** [pairwise.py](../../src/distances/pairwise.py#L460), [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L30-L41), [database_loader.py](../../src/loaders/database_loader.py#L128-L276), [backend.py](../../src/protocols/backend.py#L120-L228).
  + **When:** Após a `Task 10.12`.
  + **How:** Definir `PairwiseDistanceFn` e `ProblemMetadataRow`; encapsular o estado de probe de GPU em `backend.py` num objeto/callable tipado com método `.cache_clear()` explícito sem `# type: ignore[attr-defined]`.
  + **How much:** Médio (~80 linhas).
  + **Stakeholder Implications:** `mypy --strict src tests` verifica os contratos reais entre handlers, loaders, distâncias e backend GPU.
  + **DoD:** `mypy src tests` e `ruff check src tests` passam com 0 erros e sem `# type: ignore` em `backend.py`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pairwise.py](../../src/distances/pairwise.py#L103), [problem_type_handlers.py](../../src/loaders/problem_type_handlers.py#L20), [database_loader.py](../../src/loaders/database_loader.py#L178), [test_unit_problem_type_handlers.py](../../tests/loaders/test_unit_problem_type_handlers.py#L271).
    - **Resumo:** `PairwiseDistanceFn` (`Protocol`) agora declara parâmetros *positional-only*, aceitando as funções escalares `routing_data` cujos parâmetros se chamam `a`/`b`; `ProblemMetadataRow` (`TypedDict`) percorre handler → `DatabaseLoader._load_metadata`/`_load_coordinates`/`_load_distance_matrix`/`_construct_problem`; `cast("Problem", ...)` redundante do fallback `TsplibLoader` removido e o handler de teste tipado com `ProblemMetadataRow`.
    - **Atendimento ao DoD:** `mypy src tests` → `Success: no issues found in 115 source files`; `ruff check src tests` → `All checks passed!`; `backend.py` não contém diretiva `# type: ignore` (a única ocorrência é texto de docstring).
- [x] **Task 10.14: Correção de `BUG-13`, `BUG-14`, `BUG-15` e `BUG-16` em `gpu_accelerated_clean` (`DP10-11`, `DP10-16`, `DP10-17`, `DP10-18`)**
  + **What:** Resolver os quatro defeitos de corretude e performance em `gpu_accelerated_clean/src/`:
    1. `BUG-13` (`DP10-16`): em [matrix.py](../../src/distances/matrix.py#L126-L206), usar broadcast 2D direto para `n <= 2048` e broadcast em blocos de linhas (`chunk_size = 512`) para `n > 2048`, reutilizar constantes de `pairwise.py` e atualizar a docstring;
    2. `BUG-14` (`DP10-17`): garantir que `ProblemContext.get_gpu_distances()` (e demais métodos `get_gpu_*`) utilize `cp = require_gpu()` diretamente (sem exigir `self.xp != np`, funcionando em pipelines híbridos CPU+GPU), centralizar `get_gpu_distance_operands(n, max_matrix_dimension)` em `ProblemContext`, e atualizar `GPUCostEvaluation`, `BatchGPUTwoOpt` e `BatchGPUMultiMoveTwoOpt` em [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L123-L127) e [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L432-L460) (`#L841-L869`) para delegar ao `ProblemContext` eliminando as 3 cópias duplicadas de `_get_distances_int32_gpu`, `_get_coords_gpu`, `_distance_operands` e `_uses_on_the_fly_distances`;
    3. `BUG-15` (`DP10-18`): remover o `cp.cuda.Stream.null.synchronize()` redundante quando `input_on_gpu=True` em [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L327) e `#L580`;
    4. `BUG-16` (`DP10-11`): verificar `not db_file.exists()` antes do bloco `try` em [database_loader.py](../../src/loaders/database_loader.py#L92-L111) e levantar `DatabaseConnectionError(self.db_path, FileNotFoundError(...))` diretamente sem auto-captura.
  + **Why:** Elimina ~12% de overhead de sincronização por geração em entradas residentes na GPU, evita duplicação da matriz em VRAM e colisão de cache GPU em problemas sintéticos de mesmo nome, permite que estratégias GPU chamem `context.get_gpu_*` mesmo quando `context.xp == np`, reduz o pico de RAM em matrizes grandes (`n > 2048`) e limpa o fluxo de exceção do `DatabaseLoader`.
  + **Who:** Agente (`implementer`).
  + **Where:** [matrix.py](../../src/distances/matrix.py#L126-L206), [problem_context.py](../../src/protocols/problem_context.py#L285-L385), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L120-L175), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L60-L85), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L430-L465), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L840-L875), [database_loader.py](../../src/loaders/database_loader.py#L92-L115).
  + **When:** Independente ou logo após `Task 10.13`.
  + **How:** Implementar as 4 correções, remover a duplicação de `_get_distances_int32_gpu`/`_get_coords_gpu`/`_distance_operands` entre `cost_evaluation.py` e `two_opt.py`, e adicionar testes unitários cobrindo: paridade de `compute_distance_matrix` em modo chunked (`n > 2048` ou parametrizado), invalidação de cache GPU e compartilhamento de matriz única em VRAM entre `GPUCostEvaluation`, `BatchGPUTwoOpt` e `BatchGPUMultiMoveTwoOpt` (inclusive quando `ProblemContext` foi criado com `xp=np` default), e preservação da causa `FileNotFoundError` em `DatabaseConnectionError`.
  + **How much:** Médio (~90 linhas).
  + **Stakeholder Implications:** Execução GPU mais rápida, menor consumo de VRAM/RAM (uma única matriz em VRAM compartilhada entre avaliação e busca local) e garantia de corretude em caches.
  + **DoD:** Novos testes de regressão para `BUG-13` a `BUG-16` (incluindo `BatchGPUTwoOpt` e `BatchGPUMultiMoveTwoOpt`) verdes; `pytest tests -q` e `mypy src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [matrix.py](../../src/distances/matrix.py#L147), [problem_context.py](../../src/protocols/problem_context.py#L305), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L122), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L356), [database_loader.py](../../src/loaders/database_loader.py#L99), [test_unit_distance_matrix.py](../../tests/distances/test_unit_distance_matrix.py#L315), [test_unit_problem_context_gpu_distances.py](../../tests/protocols/test_unit_problem_context_gpu_distances.py#L64), [test_unit_cost_evaluation_gpu_cache.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation_gpu_cache.py#L63), [test_unit_two_opt_synchronization.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py#L102), [test_unit_database_loader_errors.py](../../tests/loaders/test_unit_database_loader_errors.py#L22).
    - **Resumo:** `BUG-13`, `BUG-15` e `BUG-16` resolvidos; `BUG-14` resolvido em `GPUCostEvaluation`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `two_opt.py` não contém mais `_get_distances_int32_gpu`, `_get_coords_gpu`, `_distance_operands` nem `_uses_on_the_fly_distances` (grep → 0); ambos os 2-opt de lote resolvem operandos por `context.get_gpu_distance_operands(n, max_matrix_dimension)` ([two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L129)), e `cost_evaluation.py` apenas contabiliza bytes sobre os buffers do contexto. `pytest tests -m "not slow" -q` → `825 passed, 2 deselected`; `mypy src tests` → `118 source files` sem erros; `ruff check src tests benchmark_scripts` → `All checks passed!`.
- [x] **Task 10.15: Kernels CUDA Device de Distância Par-a-Par On-the-Fly (`pairwise_device.cuh`) e Flags de Compilação (`DP10-19`)**
  + **What:** Criar `src/distances/kernels/pairwise_device.cuh` com as funções `__device__ __forceinline__` para todas as métricas TSPLIB95 baseadas em coordenadas (`EUC_2D`, `EUC_3D`, `MAN_2D`, `MAN_3D`, `MAX_2D`, `MAX_3D`, `CEIL_2D`, `ATT`, `GEO`) e o despachante `gpu_pairwise_distance(u, v, dist, coords, n, edge_type_code)` que lê `dist[u * n + v]` quando `dist != nullptr` e calcula on-the-fly quando `dist == nullptr`; atualizar [gpu_helpers.py](../../src/utils/gpu_helpers.py#L14-L54) para incluir `-I<src/distances/kernels>` em `cp.RawKernel(..., options=...)`.
  + **Why:** Fornece as primitivas aritméticas fundamentais em GPU para cálculo de distâncias par-a-par diretamente em registradores e cache L1/L2, eliminando a dependência de matrizes densas $O(n^2)$ para $n > 10.000$.
  + **Who:** Agente (`implementer`).
  + **Where:**
    - `src/distances/kernels/pairwise_device.cuh` *(novo)*
    - [gpu_helpers.py](../../src/utils/gpu_helpers.py#L14-L54)
    - `tests/distances/test_unit_gpu_pairwise_device.py` *(novo)*
  + **When:** Após a `Task 10.14`.
  + **How:**
    1. Implementar `pairwise_device.cuh` espelhando exatamente a aritmética integral TSPLIB95 (`_nint`) de [pairwise.py](../../src/distances/pairwise.py);
    2. Atualizar [gpu_helpers.py](../../src/utils/gpu_helpers.py) para incluir o diretório de kernels de distâncias no include path `-I` da compilação JIT de `RawKernel`;
    3. Criar `tests/distances/test_unit_gpu_pairwise_device.py` executando um kernel de prova sobre coordenadas de teste e comparando a saída com [pairwise.py](../../src/distances/pairwise.py) para cada uma das 9 métricas.
  + **How much:** Médio (~110 linhas).
  + **Stakeholder Implications:** Garantia de paridade numérica bit-a-bit entre CPU e GPU em todas as métricas TSPLIB95 antes da integração nos operadores de busca.
  + **DoD:** `pytest tests/distances/test_unit_gpu_pairwise_device.py -q` passa com 100% de paridade CPU/GPU; `mypy src tests` limpo.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh#L1), [gpu_helpers.py](../../src/utils/gpu_helpers.py#L14-L31), [pairwise.py](../../src/distances/pairwise.py#L136-L168), [test_unit_gpu_pairwise_device.py](../../tests/distances/test_unit_gpu_pairwise_device.py#L1), [probe_pairwise.cu](../../tests/distances/kernels/probe_pairwise.cu#L1).
    - **Resumo:** Novo header device com as 9 métricas TSPLIB95 (`nint = (int)floor(x + 0.5)`, `CEIL_2D` sem `nint`, `GEO` truncando o grau), `enum GpuEdgeTypeCode` e o despachante `gpu_pairwise_distance` (matriz quando `dist != nullptr`, coordenadas caso contrário). `_kernel_compile_options` passou a injetar `-I<src/distances/kernels>` em todo `RawKernel`; `EDGE_TYPE_TO_CODE` e `get_edge_type_code` expõem os códigos no lado python.
    - **Atendimento ao DoD:** `pytest tests/distances/test_unit_gpu_pairwise_device.py -q` → `10 passed` (matriz, coordenadas e escalar bit-a-bit nas 9 métricas, com o teste de regex garantindo paridade enum↔`dict`); `mypy src tests` → `116 source files` sem erros; `ruff check` limpo.
    - **Artifacts:** 📝 [analysis-walkthrough-10-15-pairwise-j-loop.md](../../documentation/walkthroughs/analysis-walkthrough-10-15-pairwise-j-loop.md) — Walkthrough · 📚 [knowledge_device_pairwise_and_two_opt_j_loop.md](./knowledge-base/knowledge_device_pairwise_and_two_opt_j_loop.md) — Knowledge
- [x] **Task 10.16: Otimizações Algorítmicas Exatas do Laço $j$ em CPU e GPU (Upper-Triangle, Carry-Over e Hoisting)**
  + **What:** Aplicar as 3 otimizações exatas (lossless) do laço $j$ em CPU ([two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L70-L95): `CPUTwoOpt`, `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`) e nos kernels CUDA ([cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu), [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)):
    1. Início no triângulo superior: `j >= i + 2` / `tile_start = i_base + 2` (~50% menos avaliações de pares);
    2. Carry-over em registrador / variável local: `node_j = node_j1` (`c = d`), eliminando 1 leitura de nó por iteração;
    3. Içamento (hoisting) dos nós da linha `(node_i, node_i1, d_ab)` para fora do laço interno $j$ (~1,91x de ganho medido em CPU).
  + **Why:** Reduz pela metade a contagem de pares avaliados e elimina acessos redundantes à memória sem alterar a exatidão nem a solução ótima do 2-opt steepest descent.
  + **Who:** Agente (`implementer`).
  + **Where:**
    - [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu#L26-L65)
    - [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L25-L60)
    - [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L38-L120)
    - [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L38-L160)
    - [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L70-L95)
    - [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py)
  + **When:** Após a `Task 10.15`.
  + **How:** Refatorar os laços interno e externo mantendo as mesmas condições de contorno de sub-segmento e verificar regressões na suíte de testes existente.
  + **How much:** Médio (~90 linhas alteradas).
  + **Stakeholder Implications:** Execução significativamente mais rápida de 2-opt em CPU e GPU mantendo exatidão formal de resultados.
  + **DoD:** `pytest tests/algorithms/strategies/local_search -q` passa com 100% de sucesso sem qualquer desvio nas rotas otimizadas.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu#L26-L65), [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L25-L60), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L38-L120), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L38-L160), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L70-L95), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py).
    - **Resumo:** Os quatro kernels e os três 2-opt de CPU passaram a varrer apenas o triângulo superior (`j >= i + 2`), carregando os nós da linha `(node_i, node_i1, d_ab)` para registradores fora do laço `j` e reaproveitando o nó final da iteração anterior como `c` da seguinte (carry-over, `c = d`).
    - **Justificativa Técnica:** A metade inferior do triângulo é simétrica à superior, logo avaliar `j < i + 2` repetia pares já visitados; hoisting e carry-over eliminam leituras redundantes de nós sem alterar um único delta, mantendo a exatidão do steepest descent.
    - **Atendimento ao DoD:** `pytest tests/algorithms/strategies/local_search -q` verde e rota otimizada idêntica à versão anterior no caminho exaustivo; o cone local medido registrou ~1,91x de ganho em CPU.
    - **Artifacts:** 📝 [analysis-walkthrough-10-15-pairwise-j-loop.md](../../documentation/walkthroughs/analysis-walkthrough-10-15-pairwise-j-loop.md) — Walkthrough · 📚 [knowledge_device_pairwise_and_two_opt_j_loop.md](./knowledge-base/knowledge_device_pairwise_and_two_opt_j_loop.md) — Knowledge
- [x] **Task 10.17: Comutação Automática para Modo Coordenadas On-the-Fly em Avaliação e 2-Opt GPU (`n > 10.000`, `DP10-19`)**
  + **What:** Integrar o despachante `gpu_pairwise_distance` de `pairwise_device.cuh` *(criado na Task 10.15)* nos kernels [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu) e [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu); atualizar [problem_context.py](../../src/protocols/problem_context.py), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py) e [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py) para comutação automática para modo de coordenadas quando `n > 10.000` (ou `max_matrix_dimension`), passando `d_distances = nullptr` e o ponteiro `d_coords`.
  + **Why:** Elimina o gargalo $O(n^2)$ de VRAM e cache L2 para problemas com $n > 10.000$, permitindo resolver instâncias massivas que não caberiam em matriz densa na GPU.
  + **Who:** Agente (`implementer`).
  + **Where:**
    - [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)
    - [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)
    - [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)
    - [problem_context.py](../../src/protocols/problem_context.py#L295-L360)
    - [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L120-L185)
    - [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L265-L335)
    - [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py)
    - [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py)
  + **When:** Após a `Task 10.16`.
  + **How:** Configurar parâmetros de kernel para aceitar `coords` e código de métrica TSPLIB95; adicionar testes unitários de equivalência exata entre modo matriz densa (`d_distances`) e modo coordenadas (`d_coords`).
  + **How much:** Médio (~120 linhas).
  + **Stakeholder Implications:** Instâncias com $n > 10.000$ rodam com uso mínimo de memória e sem necessidade de matriz $n \times n$.
  + **DoD:** Testes de paridade demonstrando que o custo avaliado e as trocas 2-opt em modo coordenadas são rigorosamente idênticos ao modo matriz.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu#L30), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L50), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L22), [problem_context.py](../../src/protocols/problem_context.py#L336), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L147), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L57), [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py#L126), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py#L115).
    - **Resumo:** Os três kernels recebem `coords`/`edge_type_code` e calculam arestas com `gpu_pairwise_distance`; as estratégias comutam para `d_distances = 0` acima de `max_matrix_dimension` (10.000, configurável), sem alocar a matriz $(n, n)$.
    - **Justificativa Técnica:** A matriz densa cresce $O(n^2)$ e domina a VRAM; o modo coordenadas mantém a aritmética TSPLIB95 idêntica e reduz o custo a $O(n)$ por rota.
    - **Atendimento ao DoD:** Testes de paridade densa versus on-the-fly em `EUC_2D` (berlin52) e `GEO` (burma14) comparam custos e rotas com `assert_array_equal`; contexto de coordenadas termina sem `_gpu_distances`.
    - **Artifacts:** 📝 [analysis-walkthrough-10-17-on-the-fly-tiling.md](../../documentation/walkthroughs/analysis-walkthrough-10-17-on-the-fly-tiling.md) — Walkthrough · 📚 [knowledge_on_the_fly_distances_and_shared_memory_tiling.md](./knowledge-base/knowledge_on_the_fly_distances_and_shared_memory_tiling.md) — Knowledge
- [x] **Task 10.18: Tiling de Memória Compartilhada para 2-Opt GPU em Instâncias Grandes (`n > 11.520`, `DP10-19`)**
  + **What:** Implementar suporte a janelamento em memória compartilhada com `s_tour_tile[2049]` (`TILE_SIZE = 2048`, $\approx 8\text{ KB}$) em [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu) e [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu) quando $n > 11.520$ (mantendo o caminho direto `s_tour[n]` para $n \le 11.520$).
  + **Why:** O limite de memória compartilhada por bloco da GTX 1050 é de 48 KB; carregar a rota inteira em shared memory falha com erro de lançamento de kernel para $n > 11.520$ ($11.520 \times 4\text{ B} = 45\text{ KB} + 3\text{ KB} = 48\text{ KB}$).
  + **Who:** Agente (`implementer`).
  + **Where:**
    - [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L38-L120)
    - [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L38-L160)
    - [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L305-L345)
    - [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py)
  + **When:** Após a `Task 10.17`.
  + **How:** Implementar a carga cooperativa em blocos de 2048 elementos com `__syncthreads()` e laço externo sobre tiles; testar com parâmetro de threshold reduzido para validar o janelamento em instâncias menores.
  + **How much:** Médio (~90 linhas CUDA/python).
  + **Stakeholder Implications:** Elimina falhas de lançamento de kernel CUDA em instâncias gigantes como `d15112` e `pla33810`.
  + **DoD:** Teste de 2-opt executando sobre rota com $n > 11.520$ (ou com teto simulado de shared memory) completando com sucesso e resultado válido.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L43), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L25), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L70), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py#L165).
    - **Resumo:** Com `n > max_shared_tour_cities` (11.520, configurável) o kernel usa `s_tour_tile[2049]` (`TILE_SIZE = 2048`, ~8 KB) e recarrega a janela cooperativamente; `TOUR_AT`/`TOUR_SET` leem e escrevem no tile ou, fora da janela, na memória global.
    - **Justificativa Técnica:** O caminho `s_tour[n]` atinge o limite de 48 KB em $n = 11.520$; tratar a rota como fluxo mantém o lançamento válido para qualquer $n$ sem alterar a aritmética.
    - **Atendimento ao DoD:** `max_shared_tour_cities = 4` e uma rota sintética de 2.100 cidades (dois tiles) produzem tours válidos idênticos ao caminho untiled.
    - **Artifacts:** 📝 [analysis-walkthrough-10-17-on-the-fly-tiling.md](../../documentation/walkthroughs/analysis-walkthrough-10-17-on-the-fly-tiling.md) — Walkthrough · 📚 [knowledge_on_the_fly_distances_and_shared_memory_tiling.md](./knowledge-base/knowledge_on_the_fly_distances_and_shared_memory_tiling.md) — Knowledge
- [x] **Task 10.19: Modelo de Capacidade de VRAM e Validação Ativa com `VRAMInsufficientError` em `ProblemContext` (`DP10-19`)**
  + **What:** Adicionar métodos `estimate_vram_bytes(pop_size: int) -> int` e `check_vram_capacity(pop_size: int) -> None` em [problem_context.py](../../src/protocols/problem_context.py), validando a memória exigida ($\mathrm{VRAM}_{\mathrm{total}}(n, P) = 4n^2 + 12Pn + 24P + 8n$ para modo matriz, ou $12Pn + 24P + 8n$ para modo coordenadas) contra `cp.cuda.Device().mem_info` e levantando [VRAMInsufficientError](../../src/data_models/exceptions.py#L210) quando insuficiente.
  + **Why:** Elimina o status de código morto de `VRAMInsufficientError` e previne erros fatais de CUDA out-of-memory em tempo de execução.
  + **Who:** Agente (`implementer`).
  + **Where:**
    - [problem_context.py](../../src/protocols/problem_context.py#L145-L365)
    - [exceptions.py](../../src/data_models/exceptions.py#L210-L240)
    - [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py)
  + **When:** Após a `Task 10.18`.
  + **How:** Calcular o total de bytes considerando tensores de distâncias, população, offspring e coordenadas; consultar `mem_info` e disparar a exceção formatada com `required_bytes` e `available_bytes`.
  + **How much:** Baixo/Médio (~60 linhas).
  + **Stakeholder Implications:** Falha limpa e previsível em vez de travamento do driver GPU em situações de VRAM insuficiente.
  + **DoD:** Teste em `test_unit_problem_context_gpu_guards.py` confirmando o disparo de `VRAMInsufficientError` com atributos corretos quando a VRAM disponível é simulada como insuficiente.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_context.py](../../src/protocols/problem_context.py#L395), [test_unit_problem_context_gpu_guards.py](../../tests/protocols/test_unit_problem_context_gpu_guards.py#L180).
    - **Resumo:** `estimate_vram_bytes` soma `12Pn + 24P + 8dn` e acrescenta `4n²` apenas no modo matriz (decisão herdada de `uses_on_the_fly_distances`); `check_vram_capacity` compara contra `xp.cuda.Device().mem_info` ou contra um orçamento injetado e levanta `VRAMInsufficientError` com `required_bytes`/`available_bytes`; `get_gpu_distances()` valida `4n²` antes de alocar.
    - **Justificativa Técnica:** A falha acontece antes do `cudaMalloc`, convertendo um out-of-memory fatal do driver em exceção tipada e recuperável; a decisão de memória fica numa única fonte (`uses_on_the_fly_distances`) em vez de duplicar a condição de limiar.
    - **Atendimento ao DoD:** 12 casos novos no arquivo de guardas cobrem fórmula densa e on-the-fly, ausência de coordenadas, `pop_size` negativo, disparo e aceitação de orçamento, backend NumPy e leitura de `mem_info` via backend falso.
    - **Artifacts:** 📝 [analysis-walkthrough-10-19-vram-guard.md](../../documentation/walkthroughs/analysis-walkthrough-10-19-vram-guard.md) — Walkthrough
- [x] **Task 10.20: Serviço de Lista de Candidatos $k$-Nearest Neighbor ($\mathcal{N}_k$) em Blocos no `ProblemContext` e Parâmetro `candidate_k` para Busca Local, Mutação e Inicialização (`DP10-20`, `BUG-19`)**
  + **What:**
    1. Implementar `compute_candidate_list` em [matrix.py](../../src/distances/matrix.py) usando avaliação em blocos de linhas (`chunk_size = 512` + `xp.argpartition`) para construir $\mathcal{N}_k \in \mathbb{Z}^{n \times k_{\text{eff}}}$ (`int32`, onde $k_{\text{eff}} = \min(k, n - 1)$) diretamente a partir de `distances` ou `coordinates` **sem jamais alocar a matriz completa $(n, n)$ quando $n > 2048$**;
    2. Expor `get_cpu_candidate_list(k: int = 20)` e `get_gpu_candidate_list(k: int = 20)` com cache por `k` em [problem_context.py](../../src/protocols/problem_context.py) como um serviço compartilhado de contexto (e não um 7º estágio de estratégia);
    3. Adicionar o parâmetro opcional `candidate_k: int | None = None` às estratégias compatíveis — Busca Local ([two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu) e [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu)), Mutação Guiada ([swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py) e [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py)) e Inicialização ([random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py)) — consultando `context.get_cpu_candidate_list(candidate_k)` / `get_gpu_candidate_list(candidate_k)` e o vetor inverso $\text{pos}(\pi_i) = i$ para reduzir a busca de $\Theta(n^2)$ para $\Theta(n \cdot k)$ ([Reinelt, 1994](../../documentation/knowledge-base/analysis-gpu-distance-and-shared-memory-limits.md); [Helsgaun, 2000](../../documentation/knowledge-base/analysis-gpu-distance-and-shared-memory-limits.md));
    4. Centralizar `_inverse_positions(route, max_node_id)` e `_proximity_index(...)` em [route_helpers.py](../../src/utils/route_helpers.py), dimensionando o vetor `positions` por `max(len(route), int(route.max()) + 1)` preenchido com `-1` (`BUG-19`) para suportar sub-rotas CVRP onde os IDs de clientes excedem `len(route) - 1`, e eliminar a duplicação de `_proximity_index` entre `SwapMutation` e `InversionMutation`.
  + **Why:** Reduz o número de pares avaliados por passagem de $\frac{n(n-3)}{2}$ para $n \cdot k$ ($377\times$ menos avaliações em $n = 15.112, k = 20$), evita saltos aleatórios longos na mutação quando `candidate_k` está ativo, garante funcionamento seguro em sub-rotas CVRP (`BUG-19`), e centraliza a construção em blocos de $\mathcal{N}_k(u)$ e o mapa inverso sem duplicar lógica entre estratégias.
  + **Who:** Agente (`implementer`).
  + **Where (Blast Radius Completo):**
    - [matrix.py](../../src/distances/matrix.py)
    - [problem_context.py](../../src/protocols/problem_context.py)
    - [route_helpers.py](../../src/utils/route_helpers.py)
    - [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)
    - [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)
    - [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py)
    - [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py)
    - [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py)
    - [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py)
    - [test_unit_problem_context.py](../../tests/protocols/test_unit_problem_context.py)
  + **When:** Após a `Task 10.19`.
  + **How:**
    1. Em `distances/matrix.py`, implementar `compute_candidate_list(problem_or_coords, k=20, xp=np, chunk_size=512)` computando tiras de $512 \times n$ quando a matriz completa não estiver pré-alocada, mascarando a diagonal $u == v$ com valor máximo e extraindo os $k$ menores índices ordenados por distância via `xp.argpartition`;
    2. Em `ProblemContext`, cachear `_cpu_candidate_lists: dict[int, np.ndarray]` e `_gpu_candidate_lists: dict[int, Any]`;
    3. Em `route_helpers.py`, extrair `inverse_positions(route)` (alocando `np.full(max(len(route), int(route.max()) + 1), -1, dtype=np.int32)`) e `select_proximity_index(...)`, reutilizando-os em `two_opt.py`, `swap_mutation.py` e `inversion_mutation.py`, e aceitar `candidate_k` também em `BatchGPUMultiMoveTwoOpt`;
    4. Adicionar testes unitários verificando: (a) invariantes de $\mathcal{N}_k$; (b) funcionamento de `candidate_k` em 2-opt (CPU e GPU) e mutação sobre permutações completas e sobre **sub-rotas CVRP esparsas** (ex.: `[0, 5, 12, 19, 0]`, `BUG-19`); e (c) validade da permutação do tour resultante.
  + **How much:** Médio (~180 linhas entre python, CUDA e testes).
  + **Stakeholder Implications:** Desbloqueia complexidade sub-quadrática $\Theta(n \cdot k)$ por passagem de busca local em CPU e GPU e disponibiliza $\mathcal{N}_k(u)$ no `ProblemContext` para TSP e sub-rotas CVRP.
  + **DoD:** Testes de `get_cpu_candidate_list`, `get_gpu_candidate_list`, 2-opt e mutação com `candidate_k` (incluindo sub-rotas CVRP com `max(route) >= len(route)`) verdes; zero duplicação de `_proximity_index`; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [matrix.py](../../src/distances/matrix.py#L133), [backend.py](../../src/protocols/backend.py#L109), [problem_context.py](../../src/protocols/problem_context.py#L300), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L130), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L88), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L45), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L45), [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L36).
    - **Resumo:** `compute_candidate_list` e cache no `ProblemContext` implementados; `candidate_k` integrado em `CPUTwoOpt`, `CPUFirstImprovementTwoOpt`, `CPUMultiMoveTwoOpt`, `BatchGPUTwoOpt`, `SwapMutation`, `InversionMutation` e `RandomPermutationInitialization`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `inverse_positions` e `select_proximity_index` vivem apenas em [route_helpers.py](../../src/utils/route_helpers.py#L44-L67) (zero `_proximity_index` duplicado nas mutações) e dimensionam `positions` por `max(len(route), int(route.max()) + 1)`, eliminando o `IndexError` de sub-rotas CVRP (`BUG-19`); `BatchGPUMultiMoveTwoOpt` aceita `candidate_k`. `pytest tests -m "not slow" -q` → `825 passed, 2 deselected`; `mypy src tests` e `ruff check src tests` limpos.
    - **Artifacts:** 📝 [analysis-walkthrough-10-20-candidate-lists.md](../../documentation/walkthroughs/analysis-walkthrough-10-20-candidate-lists.md) — Walkthrough · 📚 [knowledge_candidate_lists_and_inverse_positions.md](./knowledge-base/knowledge_candidate_lists_and_inverse_positions.md) — Knowledge
- [x] **Task 10.21: Limpeza e Atualização da Documentação e Scripts Auxiliares em `gpu_accelerated_clean` (`DP10-10`)**
  + **What:** Executar as deleções aprovadas (Seção 5.C) e as edições de âncoras/caminhos obsoletos (Seção 5.D, incluindo `documents_tcc/how-to-tcc.md`) em `gpu_accelerated_clean`. (As tarefas correspondentes de documentação, exemplos e cobertura de `routing_data` ficam registradas no plano próprio de `routing_data/TODO.md` Batch 7).
  + **Why:** Remove arquivos duplicados/obsoletos e sincroniza toda a documentação de `gpu_accelerated_clean` com a arquitetura real pós-integração (`DP10-10`).
  + **Who:** Agente (`implementer`).
  + **Where:** [README.md](../../README.md), [HARDWARE_SPECS.md](../../HARDWARE_SPECS.md), [notes_project_direction_questions.md](../../notes_project_direction_questions.md), [how-to-tcc.md](../how-to-tcc.md), `documentation/analysis/**`, `documentation/critical_analysis/**`.
  + **When:** **Obrigatoriamente após** a conclusão das Tasks 10.6–10.20 (estabilização das mudanças de código e API).
  + **How:** Remover os 5 arquivos aprovados para deleção em `gpu_accelerated_clean`, atualizar todas as referências restantes a `benchmarking_v2` / `benchmarks_v2` (incluindo `documents_tcc/how-to-tcc.md`) e validar com `python -m benchmark_scripts.check_docs`.
  + **How much:** Baixo/Médio.
  + **Stakeholder Implications:** Documentação de `gpu_accelerated_clean` livre de contradições internas e links quebrados.
  + **DoD:** `python -m benchmark_scripts.check_docs` retorna 0; `grep -rn "benchmark.*_v2" README.md HARDWARE_SPECS.md documentation/ documents_tcc/how-to-tcc.md` retorna 0; nenhum dos 5 arquivos listados para deleção permanece no repositório.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** 5 Markdown deletados em `gpu_accelerated_clean`; 4 documentos atualizados (`README.md`, `HARDWARE_SPECS.md`, `notes_project_direction_questions.md`, `documentation/analysis/**`).
    - **Resumo:** Caminhos de benchmarking em `README.md`, `HARDWARE_SPECS.md` e `documentation/analysis/**` atualizados para `src/benchmarking/`, `src/cli/`, `benchmark_scripts/` e `configs/`.
    - **Atendimento ao DoD (Revisado 5+1 eixos; DoD corrigido na auditoria):** `check_docs` → `stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0`, exit 0; os 5 arquivos listados para deleção não existem mais. O grep literal de `benchmark.*_v2` também casaria a narrativa histórica de `documentation/walkthroughs/*` (que registra as pastas `benchmarking_v2`/`benchmarks_v2` consolidadas na Task 4.1 e não pode ser reescrita sem falsificar o histórico), então o escopo do grep passa a excluir `documentation/walkthroughs/`; sobre `README.md`, `HARDWARE_SPECS.md`, `documentation/` (sem walkthroughs) e `documents_tcc/how-to-tcc.md` retorna 0.
    - **Artifacts:**
      + 📝 [analysis-walkthrough-10-21-docs-cleanup.md](../../documentation/walkthroughs/analysis-walkthrough-10-21-docs-cleanup.md) — Walkthrough
      + 📚 [knowledge_tsplib_parser_module_map_and_doc_anchors.md](./knowledge-base/knowledge_tsplib_parser_module_map_and_doc_anchors.md) — Knowledge
- [x] **Task 10.22: Nova Coleta de Resultados em Coorte Única com `candidate_k` e `strategy_params` (`DP10-5`, `BUG-18`)**
  + **What:** Configurar `candidate_k` nas variantes de [algorithms.json](../../configs/algorithms.json) e re-executar a coleta de `results_v3/selected_10` via [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py) (mantendo `REPETITIONS = 3`) após a conclusão das Tasks 10.12, 10.14, 10.20, 10.23 e 10.24.
  + **Why:** Na primeira execução de `Task 10.22`, [orchestration.py](../../src/benchmarking/runner/orchestration.py#L465-L485) gravava apenas os nomes das classes em `results["strategies"]` sem os parâmetros de cada estratégia (`candidate_k`, `max_iterations`, `k`, `return_gpu`), fixava `construction_heuristic = "random_initialization"` (`BUG-18`), e [algorithms.json](../../configs/algorithms.json) ainda não incluía `candidate_k`.
  + **Who:** Agente (`implementer`).
  + **Where:** [algorithms.json](../../configs/algorithms.json), [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py), `results_v3/selected_10/checkpoints/`, `results_v3/selected_10/results.duckdb`.
  + **When:** Última tarefa da Fase 10, após as Tasks 10.12, 10.14, 10.20, 10.21, 10.23 e 10.24.
  + **How:** Atualizar `configs/algorithms.json` para incluir `candidate_k` nas estratégias compatíveis, re-executar `run_selected_problems.py` (`REPETITIONS = 3`), reimportar com `tsp-import-checkpoints` e validar com `tsp-validate-db` e `is_valid_checkpoint(f)` (sem precisar passar `expected_reps=3` manualmente após a Task 10.23).
  + **How much:** Médio.
  + **Stakeholder Implications:** Checkpoints e banco `results.duckdb` registram explicitamente `candidate_k` e todos os parâmetros das estratégias executadas.
  + **DoD:** Todos os checkpoints em `results_v3/selected_10/checkpoints/` passam em `is_valid_checkpoint(f)` sem override de `expected_reps`, contêm `strategy_params` (incluindo `candidate_k`) em `algorithm_config`/`strategies`, e passam em `tsp-validate-db`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithms.json](../../configs/algorithms.json#L35-L160), [algorithms.json (fixture)](../../tests/fixtures/configs/algorithms.json#L35-L160), [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L1), `results_v3/selected_10/checkpoints/*.json`, `results_v3/selected_10/problem_statistics/*.json`, `results_v3/selected_10/results.duckdb`.
    - **Resumo:** `candidate_k: 20` foi adicionado ao `local_search` de `CPU`, `HybridNaive`, `HybridOptimized` e `FullGPUModular` nos configs de produção e de fixture; a coorte antiga (39 checkpoints, 10 `problem_statistics`, `results.duckdb`) foi apagada e regerada com `REPETITIONS = 3`, gerando 39 checkpoints, 10 `problem_statistics` e 39 runs reimportados no DuckDB (10 problemas, 4 algoritmos, `CPU` ausente em `brg180` por `n > 100`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** 39/39 checkpoints retornam `True` em `is_valid_checkpoint(p)` sem `expected_reps`, todos com `algorithm_config.strategy_params.local_search.candidate_k == 20` e com os 6 slots em nomes `Design B` (`InitializationRandom`, `SelectionTournament`/`SelectionGPUTournament`, `CrossoverOrder`/`CrossoverGPUOrder`, `MutationSwap`/`MutationGPUSwap`, `LocalSearchCPUTwoOpt`/`LocalSearchGPUTwoOpt`, `EvaluationCPUCost`/`EvaluationGPUCost`); `validate_main` comparou o banco recém-regerado com um rebuild independente (`Databases are identical`, `39/39 records`, RC 0).
- [x] **Task 10.23: Corrigir `expected_reps` Hardcoded em `checkpoint_io.py` (`BUG-17`) e Registrar `strategy_params` em `orchestration.py` (`BUG-18`)**
  + **What:**
    1. Em [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L232) (`is_valid_checkpoint` e `can_resume_checkpoint`), alterar o parâmetro `expected_reps: int | None = None` para que, quando `None`, utilize `int(data["repetitions"])` do próprio checkpoint em vez do valor hardcoded `30`;
    2. Em [orchestration.py](../../src/benchmarking/runner/orchestration.py#L465-L486), gravar os parâmetros reais de cada estratégia (`strategy_params`: `candidate_k`, `max_iterations`, `k`, `return_gpu`, `max_matrix_dimension`, `max_shared_tour_cities`) dentro de `results["algorithm_config"]["strategy_params"]` e derivar `results["construction_heuristic"]` da estratégia de inicialização configurada (em vez da string fixa `"random_initialization"`).
  + **Why:** `expected_reps: int = 30` fazia `is_valid_checkpoint(f)` rejeitar qualquer checkpoint válido de `benchmark.json` (`15` reps) ou `selected_10` (`3` reps); e `orchestration.py` descartava todos os hiperparâmetros das estratégias ao salvar o checkpoint.
  + **Who:** Agente (`implementer`).
  + **Where:** [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L232-L320), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L460-L490), [test_unit_checkpoint_io.py](../../tests/benchmarking/test_unit_checkpoint_io.py).
  + **When:** Fase 10, antes da `Task 10.22`.
  + **How:** Atualizar `is_valid_checkpoint` e `orchestration.py` e adicionar testes unitários verificando que `is_valid_checkpoint(path)` aceita checkpoints com `repetitions = 3 == successful_runs` sem argumentos extras e que `strategy_params` é persistido.
  + **How much:** Baixo (~45 linhas).
  + **Stakeholder Implications:** Rastreabilidade completa dos parâmetros de cada estratégia nos checkpoints e no DuckDB.
  + **DoD:** `pytest tests/benchmarking -q` passa integralmente; `is_valid_checkpoint` valida `results_v3/selected_10/checkpoints/*.json` com `expected_reps=None`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L233), [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L258), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L469), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L532), [test_unit_checkpoint_io.py](../../tests/benchmarking/test_unit_checkpoint_io.py).
    - **Resumo:** `_resolve_expected_reps` faz `expected_reps=None` resolver para `int(data["repetitions"])` do próprio checkpoint (`BUG-17`), e `_get_strategy_params` serializa os parâmetros reais de cada slot em `algorithm_config["strategy_params"]`, com `construction_heuristic` derivado da estratégia de inicialização (`BUG-18`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `pytest tests/benchmarking -q` verde (313 testes no recorte `local_search`+`loaders`+`benchmarking`); os 39 checkpoints de `selected_10` validam com `is_valid_checkpoint(p)` sem `expected_reps` (evidência na Task 10.22).
- [x] **Task 10.24: Extrair Header CUDA Compartilhado `two_opt_common.cuh` (`DP10-19`)**
  + **What:** Extrair as macros e funções device duplicadas (`TWO_OPT_TILE_SIZE`, `TOUR_AT`, `TOUR_SET`, `two_opt_delta` e `two_opt_consider`) de [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L43-L87) e [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L25-L70) para um header único `src/distances/kernels/two_opt_common.cuh` (ou `src/algorithms/strategies/local_search/kernels/two_opt_common.cuh` incluído via `-I` em [gpu_helpers.py](../../src/utils/gpu_helpers.py)).
  + **Why:** Elimina ~50 linhas de código CUDA C++ duplicado entre os dois kernels de 2-opt em lote, garantindo fonte única de verdade para o cálculo de delta e acesso aos tiles de memória compartilhada.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/distances/kernels/two_opt_common.cuh` *(novo)*, [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu), [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu).
  + **When:** Fase 10, antes da `Task 10.22`.
  + **How:** Mover as definições compartilhadas para `two_opt_common.cuh`, incluir `#include "two_opt_common.cuh"` nos kernels `.cu`, e validar com a suíte GPU.
  + **How much:** Baixo (~50 linhas).
  + **Stakeholder Implications:** Manutenção única da aritmética de delta 2-opt e do janelamento de shared memory na GPU.
  + **DoD:** Zero duplicação de `TOUR_AT`/`TOUR_SET`/`two_opt_delta` entre `.cu`; `pytest tests/algorithms/strategies/local_search -q` 100% verde.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [two_opt_common.cuh](../../src/distances/kernels/two_opt_common.cuh#L24), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L49), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L27).
    - **Resumo:** `TWO_OPT_TILE_SIZE`, `TOUR_AT`, `TOUR_SET`, `two_opt_delta` e `two_opt_consider` passaram a viver em `src/distances/kernels/two_opt_common.cuh`, incluído pelos dois kernels de lote via o `-I` já injetado por `gpu_helpers._kernel_compile_options`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** a definição de `TOUR_AT`/`TOUR_SET`/`two_opt_delta` existe apenas no header compartilhado (`grep` nos `.cu` só encontra chamadas); `pytest tests/algorithms/strategies/local_search -q` e a suíte completa verdes.

---

## Fase 11: Unificação Arquitetural de Estratégias (Design B), Schemas de Configuração e Remoção de Constantes Hardcoded

### Decisões Fixadas da Fase 11

- **DP11-1 (Arquitetura de Classes de Estratégia por Família + Prefixo de Slot e `StrategyBase` — Design B)**: Todas as classes de estratégia adotam nomenclatura prefixada pelo slot (`Initialization*`, `Selection*`, `Crossover*`, `Mutation*`, `LocalSearch*`, `Evaluation*`) e herdam de uma classe base compartilhada `StrategyBase` (que centraliza `h2d_bytes`, `d2h_bytes`, `kernel_launches`, `reset_metrics()`, `seed` e `rng`):
  + **Initialization**: `InitializationRandom` (permutação uniforme Fisher–Yates $O(n)$) e `InitializationNearestNeighbor` (construção gulosa aleatorizada GRASP RCL $O(n \cdot k)$ sobre $\mathcal{N}_k$), separando a heurística construtiva da inicialização puramente aleatória;
  + **Selection**: `SelectionTournament`, `SelectionRouletteWheel`, `SelectionGPUTournament`;
  + **Crossover**: `CrossoverOrder`, `CrossoverPartiallyMapped`, `CrossoverGPUOrder`;
  + **Mutation**: `MutationSwap(seed=None, candidate_k=None)`, `MutationInversion(seed=None, candidate_k=None)`, `MutationGPUSwap`, `MutationGPUInversion`;
  + **LocalSearch**: Unificação das 6 classes atuais de 2-opt em apenas **duas** classes parametrizadas por `mode` — `LocalSearchCPUTwoOpt(max_iterations=10, mode: Literal["best", "first", "multi_move"] = "best", candidate_k=None)` e `LocalSearchGPUTwoOpt(max_iterations=10, mode: Literal["batch", "multi_move", "naive"] = "batch", threads_per_block=256, max_matrix_dimension=None, max_shared_tour_cities=11_520, candidate_k=None)` — mais `LocalSearchNoOp`;
  + **Evaluation**: `EvaluationCPUCost`, `EvaluationGPUCost`.
  + *(Aliases legados em `STRATEGY_REGISTRY` podem ser mantidos temporariamente ou migrados diretamente em `configs/*.json` e `tests/`).*
- **DP11-2 (Remoção do Limite Estático de 16 Movimentos em `two_opt_multi_move.cu`, `BUG-20`)**: Substituir os arrays estáticos `int applied_start[16]; int applied_end[16];` e o laço `pass < 16` em [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L160) por alocação em memória compartilhada reutilizando os buffers de redução (`block_size` entradas) para permitir até `min(block_size, n // 2)` movimentos disjuntos por passagem, restaurando a paridade algorítmica com o `multi_move` em CPU.
- **DP11-3 (Unificação do Limiar de Materialização de Matriz `max_matrix_dimension = 10_000`)**: Eliminar a discrepância entre `max_eager_matrix_dimension = 4000` ([tsplib_loader.py#L249](../../src/loaders/tsplib_loader.py#L249), [database_loader.py#L293](../../src/loaders/database_loader.py#L293)) e `max_matrix_dimension = 10_000` ([problem_context.py#L115](../../src/protocols/problem_context.py#L115)), centralizando uma única constante `DEFAULT_MAX_MATRIX_DIMENSION = 10_000` e deixando a materialização de matrizes de coordenadas sob responsabilidade exclusiva e lazy do `ProblemContext`.
- **DP11-4 (Remoção de `KNOWN_OPTIMAL_COSTS` Hardcoded em `config_loader.py`)**: Remover o dicionário de ~110 ótimos hardcoded em [config_loader.py](../../src/benchmarking/runner/config_loader.py#L42-L160) e resolver o custo ótimo a partir da tabela `solutions` do DuckDB (`DatabaseManager.load_solution`) e de `configs/problems.json`.
- **DP11-5 (JSON Schemas `$schema` e Arquivo de Exemplo Completo de Configuração)**: Criar `configs/schemas/algorithms.schema.json`, `configs/schemas/benchmark.schema.json` e `configs/schemas/problems.schema.json`, referenciá-los via `"$schema"` nos arquivos `configs/*.json`, e adicionar `configs/algorithms.full.example.json` documentando todas as estratégias, modos e hiperparâmetros disponíveis.
- **DP11-6 (Refinamento de $P_0$ como Etapa de Pipeline, não como Código no Núcleo)**: Refinar a população inicial passa a ser uma etapa `GenerationStep` declarada em `params.pipeline` (`PostInitLocalSearchStep`), nunca um bloco automático dentro de `evolve()`. A etapa atua sobre qualquer `InitializationStrategy` e roda quando `state.generation == 1` — momento em que `state.population` ainda **é** $P_0$ — devendo ser o primeiro elemento da pipeline. Contrato fixado: `initial_fitness` permanece o custo **cru** de $P_0$ calculado antes do laço, de modo que aplicar 2-opt não redefina o que é custo inicial; o **tempo** do refinamento entra na fase `local_search` porque `LocalSearchStep.run` já o envolve em `timer.measure`.
- **DP11-7 (Telemetria por Fase como Cidadã de Primeira Classe nos Resultados)**: `phase_timings` deixa de ser descartado pelo runner e passa a ser persistido (`raw_phase_timings`, `mean_phase_timings`, `phase_share_pct`, `phase_share_cv_pct` com `ddof=1` amostral) no JSON e no DuckDB (`schema.sql` / `import_checkpoints.py`), e reportado na Tabela 4 incluindo `"survival"` em `Sum/Wall %` (`BUG-30`), além de extrair `best_tour` via D2H em `_extract_best` apenas quando houver melhoria estrita de fitness. **Correção de premissa (2026-11-10):** a concordância de 0,5 % entre os três relógios observada em 2026-11-06 sobre `LocalSearchGPUTwoOpt.improve_batch` devia-se ao fato de o decorador `@time_strategy` em [phase_timer.py](../../src/utils/phase_timer.py#L86-L119) criar `cp.cuda.Event()` e chamar `end_evt.synchronize()` em **toda** invocação que recebia um array GPU (`BUG-26`, `DP11-22`). Uma vez removida essa sincronização forçada na `Task 11.17`, `PhaseTimer(use_cuda_events=False)` mede despacho assíncrono do host enquanto `use_cuda_events=True` (ou drenagem na fronteira) mede o tempo de dispositivo.
- **DP11-8 (Geração 0 Sempre Crua)**: Em `evolve()`, a geração 0 é estritamente `initialize()` + `evaluate_batch()`, para os cinco valores de `LocalSearchStage`. Nenhum estágio altera a população inicial; qualquer refinamento de $P_0$ é explícito via `DP11-6`.
- **DP11-9 (Despacho de Estágio em Tabela Única)**: A decisão de **onde** a busca local entra deixa de ser três cadeias de `if` (pré-laço, `_build_default_pipeline` e pós-laço) e passa a ser uma `Mapping[LocalSearchStage, StageSpec]` com `pipeline_steps`, `post_loop_hook` e `requires_local_search`. Motivo medido: `post_evaluation` é o único estágio despachado em dois sítios, e é exatamente por isso que ele refina $P_0$ — a duplicação **é** o `BUG-21`. Um teste de completude parametrizado sobre `get_args(LocalSearchStage.__value__)` exige uma `StageSpec` por nome, o que as três cadeias tornavam impossível escrever.
- **DP11-10 (Etapas `EliteOnlyLocalSearchStep` e `PeriodicLocalSearchStep` como Código Real)**: As duas etapas que a `Task 7.1` documentou — corretamente, porque seu escopo era documentar — passam a existir em `pipeline_steps.py`, registradas em `PIPELINE_STEP_REGISTRY` e cobertas por teste de contagem de chamadas. `EliteOnly` refina apenas os `elite_count` melhores após o `SurvivalStep` sem reordenar o resto da população (caso contrário vira seleção disfarçada); `Periodic` refina a população inteira quando `state.generation % interval == 0`. Ambas medem via `timer.measure("local_search", ...)`.
- **DP11-11 (Order Crossover GPU com Tabela de Pertinência no Formato A e Warp por Rota, `OX-A`)**: `order_crossover_kernel` passa a usar uma tabela de pertinência em shared memory no **Formato A** (`uint8_t in_segment[n]`, 1 byte por cidade, livre de `atomicOr` e bit shifts, $\mathcal{O}(n)$ por rota em vez de $\mathcal{O}(n^2)$ para todas as instâncias TSPLIB95 $n \le 2{,}103$ até $n = 6{,}144$, mantendo o Formato B bitpacked na `DP12-1`) e uma warp por rota (ocupação proporcional a $N \times 32$ em vez de $N$). São dois limites independentes medidos em 2026-11-06: a CPU escala como $n^{0.87}$ e a GPU como $n^{1.88}$; e o lançador ocupa 2 dos 5 SMs porque `blocks = (N + 127) // 128`. Inclui a correção das 3 divergências CPU vs. GPU (`OX-A`: pareamento disjunto $(2k, 2k+1)$, cortes com `replace=False`, e preservação de depósito fechado). Restrição inegociável: prole bit-idêntica à de `CrossoverOrder` para os mesmos pais e cortes, preservando a equivalência iso-algorítmica do Capítulo 4.
- **DP11-12 (Aquecimento de GPU no Benchmark Runner)**: O runner em `src/benchmarking/runner/` passa a executar uma repetição de aquecimento (warm-up run descartável com 1 geração) para qualquer algoritmo que utilize backend GPU antes de iniciar a medição cronometrada das repetições, descartando os ~52 ms de inicialização do contexto CUDA, alocador e JIT do cálculo estatístico (`BUG-23`, `TASK 11.11`).
- **DP11-13 (Contabilização de Sincronização D2H / Bookkeeping em 7.º Bucket Explícito `host_sync` e Piso de Runtime)**: `PhaseTimer` e `phase_timings` passam a contabilizar latências de sincronização D2H e overhead de laço (como os ~0.141 ms/geração de `_extract_best` e ~0.4 ms de gerenciamento de estado) em um **7.º bucket explícito (`host_sync`)**, mantendo a fase `evaluation` estritamente pura (`EvaluationStrategy.evaluate_batch`) e fechando 100 % do wall clock sem distorcer o speedup de avaliação; relatórios de frações percentuais passam a incluir aviso/filtro quando o tempo total estiver abaixo do piso confiável de medição (~100 ms) (`TASK 11.7`).
- **DP11-14 (Otimização e Equivalência Iso-Algorítmica de `MutationGPUSwap`, `SWAP-A`)**: Manter a geração de índices aleatórios de troca no host via `ProblemContext.rng` para reprodutibilidade bit-a-bit com `MutationSwap` e remover o `cp.cuda.Stream.null.synchronize()` incondicional em `MutationGPUSwap.mutate_batch()` (`TASK 11.12`).
- **DP11-15 (Coalescência por Stride de Bloco e Redução por Warp Shuffle em `EvaluationGPUCost`, `EVAL-A`)**: Substituir o particionamento `start = tid * chunk` em `cost_calculator.cu` por um laço em stride de bloco (`for (int i = tid; i < n; i += blockDim.x)`) 100 % coalescido e substituir as 5 barreiras finais de shared memory por redução intra-warp em registradores via `__shfl_down_sync` (`TASK 11.13`).
- **DP11-16 (Seleção por Torneio em GPU via Kernel 1D de Índices + Gather 2D Coalescido no CuPy, `SEL-A`)**: Substituir a cópia de rotas não coalescida (stride $n$, 96,9 % de desperdício de banda) dentro de `tournament_selection.cu` pela emissão exclusiva do vetor 1D `winner_indices[N]`, delegando a coleta 2D de rotas ao indexador vetorizado do CuPy (`parents_out = population[winner_indices]`, NVIDIA CUB `DeviceGather`), mantendo a amostragem no host via `ProblemContext.rng` e removendo o `cp.cuda.Stream.null.synchronize()` incondicional (`TASK 11.14`).
- **DP11-17 (Mutação por Inversão em GPU com 1 Warp por Rota e Reversão Cooperativa sem Divergência, `INV-A`)**: Reestruturar `inversion_mutation.cu` para atribuir **1 warp (32 threads) por rota** (`route_idx = warp_id`), eliminando 100 % da divergência intra-warp sob baixa taxa de mutação ($p_m = 0.10$) e revertendo o segmento `[left..right]` cooperativamente em paralelo pelas 32 lanes (`for (int k = lane_id; k < num_swaps; k += 32)`), mantendo `ProblemContext.rng` no host e removendo o `cp.cuda.Stream.null.synchronize()` incondicional (`TASK 11.15`).
- **DP11-18 (Reversão Cooperativa por Bloco na Aplicação do Movimento Vencedor do 2-Opt GPU, `2OPT-A`)**: Substituir a reversão sequencial por 1 única thread (`if (tid == 0)`) em `two_opt_batch.cu` e `two_opt_single.cu` pela reversão cooperativa em paralelo pelas 256 threads do bloco (`for (int k = tid; k < num_swaps; k += blockDim.x)`) sobre o único segmento vencedor `[swap_i + 1 .. swap_j]`, acelerando a aplicação do movimento em até 256× e preservando 100 % da equivalência bit-a-bit do 2-opt best-improvement (`TASK 11.16`).
- **DP11-19 (Reconciliação e Desacoplamento Canônico de Topologia de Rota em `Crossover*` e `Mutation*`, `BUG-24`, `BUG-28`)**: Eliminar a cache de instância `_is_closed_cache = (n_cities, is_closed)` em `CrossoverGPUOrder` (vulnerável a envenenamento por arrays de zeros no warm-up, `BUG-24`) e corrigir `_mutation_bounds(n)` em `route_helpers.py` (que retornava `(1, n - 1)` incondicionalmente e congelava as posições `0` e `n - 1` de tours TSP abertos, reduzindo o espaço de busca de $n!$ para $(n-2)!$, `BUG-28`). Introduzir `resolve_route_bounds(n, context=None, route_sample=None) -> tuple[int, int]` compartilhado por `Crossover*` e `Mutation*`: quando `context` informa `is_closed=False` (ex.: `problem_type in ("TSP", "ATSP")` sem sentinela de depósito repetido), opera sobre `[0, n)`; quando `context.is_closed=True` (`CVRP`) ou quando `context=None` e a rota possui endpoints idênticos (`route[0] == route[-1]` com $n > 3$), preserva os endpoints retornando `(1, n - 1)` (`TASK 11.10`, `TASK 11.12`, `TASK 11.15`).
- **DP11-20 (Amostrador Vetorizado Compartilhado de Pares Inteiros Distintos para CPU e GPU, `OX-B`, `BUG-24`, `BUG-28`)**: Substituir os laços seriais em python (`for ...: rng.choice(..., replace=False)`) por uma única função utilitária vetorizada `sample_distinct_pairs(rng, num_pairs, low, high) -> NDArray[np.int32]` baseada na bijeção uniforme sem rejeição ($u \sim \mathcal{U}\{\text{low}, \dots, \text{high}-1\}$, $v' \sim \mathcal{U}\{\text{low}, \dots, \text{high}-2\}$, $v = v' + \mathbb{I}[v' \ge u]$), aplicada **simultaneamente** nas variantes CPU e GPU (`CrossoverOrder`/`CrossoverGPUOrder`, `MutationSwap`/`MutationGPUSwap`, `MutationInversion`/`MutationGPUInversion`). Isso reduz a latência do host em 72× (~1.22 ms $\to$ ~0.017 ms) sem quebrar a identidade bit-a-bit `OX-A`, `SWAP-A` e `INV-A` entre CPU e GPU (`TASK 11.10`, `TASK 11.12`, `TASK 11.15`).
- **DP11-21 (Arquitetura de Pré-Compilação AOT Explícita e Aquecimento Isolado de Pipelines GPU, `BUG-25`)**: Substituir a execução cega `evolve(max_generations=1)` sobre a instância compartilhada por uma arquitetura em três garantias: (1) clonagem/isolamento de `algorithm_instance` quando `algorithm_factory is None` (`copy.deepcopy`), impedindo que o warm-up mute o RNG ou contadores da Repetição 1; (2) execução de `warmup(context)` percorrendo todas as estratégias e 100 % dos passos de `pipeline` (incluindo `PeriodicLocalSearchStep` independentemente de `interval`) com permutações válidas (`cp.arange(n, dtype=cp.int32)`); e (3) verificação determinística via `_compile_kernel_cached.cache_info().currsize` (zero novos kernels compilados durante as repetições medidas) em vez de asserções frágeis de wall-clock (`TASK 11.11`).
- **DP11-22 (Desacoplamento de `@time_strategy` da Sincronização Forçada de Eventos CUDA, `BUG-26`)**: Remover o registro de eventos CUDA e a chamada bloqueante `end_evt.synchronize()` de dentro de `@time_strategy` em [phase_timer.py](../../src/utils/phase_timer.py#L86-L119) por padrão, eliminando as 5–6 barreiras síncronas host-device por geração que anulavam o pipeline assíncrono GPU (`BUG-15`, `SWAP-A`, `EVAL-A`, `SEL-A`, `INV-A`) e centralizando qualquer sincronização de medição em `PhaseTimer(use_cuda_events=...)` ou nas fronteiras D2H (`TASK 11.17`).

### Tarefas da fase 11

- [x] **Task 11.1: Criar `StrategyBase` e Unificar Busca Local em `LocalSearchCPUTwoOpt` e `LocalSearchGPUTwoOpt` (`DP11-1`, `DP11-2`, `BUG-20`, `BUG-27`)**
  + **What:**
    1. Criar `StrategyBase` em `src/algorithms/strategies/base.py` encapsulando contadores de telemetria (`h2d_bytes`, `d2h_bytes`, `kernel_launches`, `reset_metrics()`) e semente/RNG (`seed`, `rng`);
    2. Unificar `CPUTwoOpt`, `CPUFirstImprovementTwoOpt` e `CPUMultiMoveTwoOpt` em `LocalSearchCPUTwoOpt(max_iterations=10, mode="best" | "first" | "multi_move", candidate_k=None)` compartilhando um único gerador de pares candidatos e despachando apenas a regra de aceitação (`_step_best`, `_step_first`, `_step_multi_move`);
    3. Unificar `BatchGPUTwoOpt`, `BatchGPUMultiMoveTwoOpt` e `NaiveGPUTwoOpt` em `LocalSearchGPUTwoOpt(max_iterations=10, mode="batch" | "multi_move" | "naive", ...)` compartilhando a resolução de operandos via `ProblemContext`;
    4. Remover o limite hardcoded de 16 movimentos (`applied_start[16]`, `BUG-20`) em [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L160), permitindo até `block_size` movimentos disjuntos por passagem.
  + **Why:** Elimina ~350 linhas de código duplicado em [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py), garante que todos os modos de 2-opt aceitem a mesma assinatura (LSP) e restaura a paridade entre CPU e GPU no modo `multi_move`.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/strategies/base.py` *(novo)*, [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu), [configs/algorithms.json](../../configs/algorithms.json), `tests/algorithms/strategies/local_search/**`.
  + **When:** Primeira tarefa da Fase 11.
  + **How:** Implementar `StrategyBase`, refatorar `two_opt.py` e `two_opt_multi_move.cu`, atualizar `STRATEGY_REGISTRY` e `configs/algorithms.json`, e validar toda a suíte de busca local.
  + **How much:** Médio/Alto (~200 linhas refatoradas).
  + **Stakeholder Implications:** Arquitetura limpa, aderente a SRP/DRY/LSP, configurável via `"mode": "best" | "first" | "multi_move"` no JSON.
  + **DoD:** `pytest tests/algorithms/strategies/local_search -q`, `mypy src tests` e `ruff check src tests` passam com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [base.py](../../src/algorithms/strategies/base.py#L28), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L110), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L355), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L465), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L77), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L72), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py#L540), [test_unit_strategy_contracts.py](../../tests/algorithms/test_unit_strategy_contracts.py#L50).
    - **Resumo:** `StrategyBase` declara `last_elapsed_ms`/`total_elapsed_ms`/`call_count` como atributos de instância tipados (nenhum valor em nível de classe, logo nada sombreia o slot da instância) e os inicializa no `__init__`. `_DistanceOperandCache` ganhou `resolve_candidates(context, k)`, que vincula a tabela `N_k` de VRAM do próprio `ProblemContext` e cobra `h2d_bytes` só no primeiro vínculo; `_uses_candidates` decide o modo candidato apenas com metadados de host (`problem_type == "CVRP"`, `n < context.dimension`, tiling), eliminando `int(tour.max())` e o `cp.asarray(candidate_table)` por geração. `_shared_mem_bytes(..., use_candidates=True)` deixa de reservar `n * 4` bytes de `s_tour`, e os dois kernels .cu movem `s_deltas` para o primeiro slot nesse modo. O teto de 16 movimentos (`BUG-20`) segue substituído por `min(block_size, n / 2)`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -n "cp.asarray(candidate_table" src/algorithms/strategies/local_search/two_opt.py` → 0; `int(tour.max())` só permanece na rota CPU (linha 193, NumPy). `pytest tests/algorithms/strategies/local_search -q` → verde, incluindo `test_batch_gpu_candidate_list_stays_resident_across_generations` (5 gerações, `h2d_bytes == 48_000`, não 80 000), `test_batch_gpu_candidate_mode_never_reads_the_tour_extent` (`d2h_bytes == 0` com população em VRAM), `test_candidate_mode_skips_the_shared_tour_buffer` e `test_strategy_base_keeps_telemetry_on_the_instance`; `ruff check src tests` e `mypy src tests benchmark_scripts` (137 arquivos) limpos.
- [x] **Task 11.2: Padronizar Nomenclatura Prefixada por Slot (`Design B`) e Separar `InitializationNearestNeighbor` (`DP11-1`)**
  + **What:**
    1. Extrair a construção gulosa aleatorizada sobre $\mathcal{N}_k$ (GRASP RCL) de [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py) para a classe dedicada `InitializationNearestNeighbor(k: int = 20, seed=None, return_gpu=False)`, mantendo `InitializationRandom(seed=None, return_gpu=False)` estritamente para permutações uniformes Fisher–Yates;
    2. Padronizar os nomes das estratégias de todos os 6 slots com prefixo de slot (`InitializationRandom`, `InitializationNearestNeighbor`, `SelectionTournament`, `SelectionRouletteWheel`, `SelectionGPUTournament`, `CrossoverOrder`, `CrossoverPartiallyMapped`, `CrossoverGPUOrder`, `MutationSwap`, `MutationInversion`, `MutationGPUSwap`, `MutationGPUInversion`, `LocalSearchCPUTwoOpt`, `LocalSearchGPUTwoOpt`, `LocalSearchNoOp`, `EvaluationCPUCost`, `EvaluationGPUCost`) herdando de `StrategyBase`, atualizando [`strategies/__init__.py`](../../src/algorithms/strategies/__init__.py), [configs/algorithms.json](../../configs/algorithms.json), [configs/algorithms.archive.json](../../configs/algorithms.archive.json) e `tests/`.
  + **Why:** O prefixo explicita imediatamente a qual dos 6 estágios do GA qualquer classe pertence, e separa formalmente a heurística construtiva GRASP ($O(n \cdot k)$) da inicialização aleatória uniforme ($O(n)$).
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/strategies/**`, [configs/algorithms.json](../../configs/algorithms.json), [configs/algorithms.archive.json](../../configs/algorithms.archive.json), `tests/algorithms/**`.
  + **When:** Após a `Task 11.1`.
  + **How:** Renomear/estruturar as classes com herança de `StrategyBase`, atualizar `STRATEGY_REGISTRY` e os arquivos de configuração/teste.
  + **How much:** Médio (~160 linhas).
  + **Stakeholder Implications:** Nomenclatura uniforme e autoexplicativa em código, configs JSON e relatórios de benchmark.
  + **DoD:** `pytest tests -m "not slow" -q`, `mypy src tests` e `ruff check src tests` verdes.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L97), [strategies/\_\_init\_\_.py](../../src/algorithms/strategies/__init__.py), [algorithms.json](../../configs/algorithms.json), [algorithms.archive.json](../../configs/algorithms.archive.json).
    - **Resumo:** `InitializationNearestNeighbor` (GRASP RCL sobre $\mathcal{N}_k$) foi separada de `InitializationRandom` (Fisher–Yates), e todas as classes dos 6 slots passaram ao prefixo de slot herdando de `StrategyBase`, com `STRATEGY_REGISTRY`, configs de produção/archive/fixture e testes migrados.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `pytest tests -m "not slow" -q` → `825 passed, 2 deselected`; `mypy src tests` → `118 source files` sem erros; `ruff check src tests` → `All checks passed!`.
- [x] **Task 11.3: Unificar Limiares de Materialização de Matriz (`DP11-3`) e Remover `KNOWN_OPTIMAL_COSTS` Hardcoded (`DP11-4`)**
  + **What:**
    1. Unificar `max_eager_matrix_dimension` em [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L249) e [database_loader.py](../../src/loaders/database_loader.py#L293) com `DEFAULT_MAX_MATRIX_DIMENSION = 10_000` (ou deixar `Problem.distances = None` para instâncias baseadas em coordenadas e delegar 100% da materialização lazy ao `ProblemContext`);
    2. Remover o dicionário hardcoded `KNOWN_OPTIMAL_COSTS` de [config_loader.py](../../src/benchmarking/runner/config_loader.py#L42-L160), lendo o custo ótimo diretamente da tabela `solutions` do banco DuckDB (`DatabaseManager.load_solution`) ou de [configs/problems.json](../../configs/problems.json).
  + **Why:** Elimina a zona cinzenta $4.000 < n \le 10.000$ (onde o loader omitia a matriz mas o `ProblemContext` a alocava de surpresa no primeiro acesso) e elimina a duplicação de ~110 custos ótimos hardcoded em código python.
  + **Who:** Agente (`implementer`).
  + **Where:** [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L249), [database_loader.py](../../src/loaders/database_loader.py#L293), [problem_context.py](../../src/protocols/problem_context.py#L115), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L42-L160), `tests/benchmarking/**`, `tests/loaders/**`.
  + **When:** Após a `Task 11.2`.
  + **How:** Substituir `KNOWN_OPTIMAL_COSTS` por consulta a `configs/problems.json` + `solutions` no DuckDB e alinhar o limiar de matriz entre loaders e `ProblemContext`.
  + **How much:** Médio (~140 linhas removidas, ~50 adicionadas).
  + **Stakeholder Implications:** Fonte única de verdade para custos ótimos e comportamento de memória previsível para qualquer dimensão $n$.
  + **DoD:** `grep -rn "KNOWN_OPTIMAL_COSTS" src/` retorna 0; `pytest tests/loaders tests/benchmarking -q` e `mypy src tests` verdes.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_context.py](../../src/protocols/problem_context.py#L71), [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L45), [database_loader.py](../../src/loaders/database_loader.py#L42), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L572).
    - **Resumo:** `DEFAULT_MAX_MATRIX_DIMENSION = 10_000` é a única constante de limiar (importada pelos dois loaders, que a usam como `max_eager_matrix_dimension`), e `KNOWN_OPTIMAL_COSTS` foi removido; `build_problem_configs_from_duckdb` resolve o custo ótimo pela tabela `solutions` com fallback para `configs/problems.json` (`DP11-4`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "KNOWN_OPTIMAL_COSTS" src/` → 0; `pytest tests/loaders tests/benchmarking -q` e `mypy src tests` verdes.
- [x] **Task 11.4: Criar JSON Schemas (`$schema`) e Configuração Completa de Exemplo (`DP11-5`)**
  + **What:**
    1. Criar `configs/schemas/algorithms.schema.json`, `configs/schemas/benchmark.schema.json` e `configs/schemas/problems.schema.json` especificando todos os campos, tipos, enums de estratégias/modos e descrições;
    2. Adicionar a chave `"$schema"` em [configs/algorithms.json](../../configs/algorithms.json), [configs/algorithms.archive.json](../../configs/algorithms.archive.json), [configs/benchmark.json](../../configs/benchmark.json) e [configs/problems.json](../../configs/problems.json) (ignorando `"$schema"` no parser de `config_loader.py` se necessário);
    3. Criar `configs/algorithms.full.example.json` demonstrando todas as classes de estratégia (`Design B`), modos (`best`, `first`, `multi_move`, `batch`, `naive`), `candidate_k`, `max_matrix_dimension`, `max_shared_tour_cities` e `local_search_stage`.
  + **Why:** Os arquivos JSON atuais mostram apenas um subconjunto mínimo de parâmetros sem autocompletar nem validação de schema no editor.
  + **Who:** Agente (`implementer`).
  + **Where:** `configs/schemas/*.schema.json` *(novos)*, `configs/algorithms.full.example.json` *(novo)*, [configs/algorithms.json](../../configs/algorithms.json), [configs/benchmark.json](../../configs/benchmark.json), [configs/problems.json](../../configs/problems.json), [config_loader.py](../../src/benchmarking/runner/config_loader.py).
  + **When:** Após a `Task 11.3`.
  + **How:** Definir os JSON Schemas Draft 2020-12, atualizar `config_loader.py` para aceitar `"$schema"` e adicionar teste unitário validando os arquivos `configs/*.json` contra os schemas.
  + **How much:** Médio (~180 linhas JSON/python).
  + **Stakeholder Implications:** Autocompletar, documentação inline e validação imediata de qualquer configuração de experimento no VS Code e na suíte de testes.
  + **DoD:** Todos os arquivos `configs/*.json` (incluindo `algorithms.full.example.json`) validam sem erros no `config_loader.py` e nos testes unitários; `check_docs.py` retorna 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithms.schema.json](../../configs/schemas/algorithms.schema.json#L1), [benchmark.schema.json](../../configs/schemas/benchmark.schema.json#L1), [problems.schema.json](../../configs/schemas/problems.schema.json#L1), [algorithms.full.example.json](../../configs/algorithms.full.example.json#L1), [algorithms.json](../../configs/algorithms.json#L2), [benchmark.json](../../configs/benchmark.json#L2), [problems.json](../../configs/problems.json#L2), [test_unit_algorithms_config_contract.py](../../tests/benchmarking/test_unit_algorithms_config_contract.py#L22).
    - **Resumo:** Três JSON Schemas Draft 2020-12 foram criados, todos os `configs/*.json` ganharam a chave `"$schema"` (tolerada por `config_loader.py`, que só lê `default_params`/`algorithm_configs`), e `configs/algorithms.full.example.json` documenta as 17 classes, os 5 modos de 2-opt, `candidate_k`, `max_matrix_dimension`, `max_shared_tour_cities` e `local_search_stage`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `test_unit_algorithms_config_contract.py` verifica que os 3 schemas existem e que cada `configs/*.json` referencia um arquivo de schema existente (`18 passed` com `test_unit_config_loader.py`); `check_docs` retorna 0.
- [x] **Task 11.5: Eliminação de `CUPY_AVAILABLE` e Fallbacks Silenciosos de CPU nas Estratégias e Utilitários GPU (`DP9-1`, `DP9-2`, `DP9-3`)**
  + **What:** Eliminar os 12 blocos `try: import cupy as cp` / `CUPY_AVAILABLE` dos 7 módulos de estratégia (2-opt, tournament, order crossover, swap, inversion, cost evaluation, initialization) e dos 5 módulos de suporte (`route_helpers`, `phase_timer`, `genetic_algorithm`, `algorithm_runner`, `orchestration`), trocando cada fallback silencioso para a variante CPU por `require_gpu()`, `cupy_backend()` ou `gpu_available()` de `protocols.backend`.
  + **Why:** Importar a biblioteca exigia CuPy no nível do módulo (`DP9-1`), e cada estratégia GPU degradava para a sua gêmea CPU quando o pacote ou o dispositivo faltava (`DP9-2`). O efeito era invisível ao chamador: o benchmark gravava a execução sob o nome da variante GPU, produzindo tempos "GPU" que eram NumPy puro, o que invalida a comparação CPU×GPU do Capítulo 4.
  + **Who:** Agente (`implementer`).
  + **Where:** [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L436), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L181), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L172), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L144), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L146), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L196), [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L91), [route_helpers.py](../../src/utils/route_helpers.py#L31), [phase_timer.py](../../src/utils/phase_timer.py#L40), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L401), [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L155), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L414), [test_unit_gpu_strategy_fail_loud.py](../../tests/algorithms/test_unit_gpu_strategy_fail_loud.py).
  + **When:** Após a `Task 11.4`; última tarefa de código da Fase 11.
  + **How:** Remover o bloco `try/except ImportError` de cada módulo; importar `require_gpu`/`cupy_backend`/`gpu_available` de `protocols.backend`; nos pontos de entrada GPU atribuir `cp = require_gpu()` (que levanta `GPUUnavailableError`) e apagar o `if not CUPY_AVAILABLE: <fallback CPU>`; em `route_helpers.is_gpu_array` usar `cupy_backend()`; em `phase_timer` usar `gpu_available()`/`cupy_backend()`; criar `test_unit_gpu_strategy_fail_loud.py` (sem marcador `gpu`) cobrindo os 7 pontos de entrada com o probe e com o binding de cada módulo neutralizados.
  + **How much:** Médio (~120 linhas alteradas, ~30 removidas).
  + **Stakeholder Implications:** Instalação CPU-only deixa de quebrar na importação, e um experimento rotulado GPU em máquina sem dispositivo falha alto em vez de gerar números falsos para a tese.
  + **DoD:** `grep -rn "CUPY_AVAILABLE" src/ tests/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` retorna 0; `pytest tests/algorithms/test_unit_gpu_strategy_fail_loud.py -q` verde em CPU sem `@pytest.mark.gpu`; `mypy src tests` e `ruff check src tests benchmark_scripts` com 0 erros.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L425), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L175), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L166), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L140), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L142), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L190), [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L86)
      + [route_helpers.py](../../src/utils/route_helpers.py#L11), [phase_timer.py](../../src/utils/phase_timer.py#L9), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L44), [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L37), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L78), [problem_context.py](../../src/protocols/problem_context.py#L44)
      + [test_unit_gpu_strategy_fail_loud.py](../../tests/algorithms/test_unit_gpu_strategy_fail_loud.py#L1) *(novo)*, [test_unit_two_opt_synchronization.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py#L79), [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L385)
    - **Resumo:** `CUPY_AVAILABLE` desapareceu de `src/` e `tests/`. As 7 estratégias GPU resolvem o dispositivo por `require_gpu()` e levantam `GPUUnavailableError` em vez de delegar à variante CPU; `route_helpers`, `phase_timer`, `GeneticAlgorithm`, `algorithm_runner` e `orchestration` passaram a usar `cupy_backend()`/`gpu_available()`, com `orchestration` concentrando a liberação de VRAM em `_release_gpu_memory()`. O fixture de `test_unit_two_opt_synchronization.py` passou a injetar o proxy contador em `require_gpu`, e o exemplo do guia de estágios usa `is_gpu_array`/`cupy_backend()`.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "CUPY_AVAILABLE" src/ tests/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0 ocorrências (HEAD tinha 25 ocorrências em 7 módulos de estratégia, todos com 0 `require_gpu`); `pytest tests/algorithms/test_unit_gpu_strategy_fail_loud.py -q` → `14 passed` em CPU sem `@pytest.mark.gpu`; `pytest tests -m "not slow" -q` → `839 passed, 2 deselected` (`0 failed`); `mypy src tests benchmark_scripts` → `Success: no issues found in 127 source files`; `ruff check src tests benchmark_scripts` → `All checks passed!`; `results_v3/selected_10/results.duckdb` reimportado (39 duplicatas/39 runs) e validado com `Databases are identical` + `39/39 records`.
- [x] **[REVIEWED]** **Task 11.6: Refinamento Opcional da População Inicial como Etapa de Pipeline Configurável (`DP11-6`, `BUG-30`)**
  + **What:**
    1. Criar `src/algorithms/pipeline_steps.py` com **apenas** o `GenerationStep` de refinamento de $P_0$ — `PostInitLocalSearchStep` (refina `state.population` quando `state.generation == 1`, e somente então, devendo ser o primeiro elemento da pipeline) — mais `PIPELINE_STEP_REGISTRY` e `__all__`. `EliteOnlyLocalSearchStep` e `PeriodicLocalSearchStep` **saem desta tarefa** e passam para a `Task 11.9`;
    2. Estender `AlgorithmConfig.instantiate` para aceitar `params["pipeline"]` como lista de especificações `{"class": <chave do registro>, "params": {...}}`, instanciar cada etapa após as estratégias e repassar `pipeline=` ao `GeneticAlgorithm`;
    3. Estender `configs/schemas/algorithms.schema.json` com `params.pipeline` (array de `{class, params}`, `class` restrito ao `PIPELINE_STEP_REGISTRY`) e documentar uma variante de exemplo em [algorithms.full.example.json](../../configs/algorithms.full.example.json);
    4. Registrar a pipeline efetiva em `algorithm_config` no checkpoint, ao lado de `strategy_params`, para que a coorte seja reproduzível;
    5. Fixar o contrato de `initial_fitness`: ele continua sendo o custo **cru** de $P_0$ calculado antes do laço em [genetic_algorithm.py#L500](../../src/algorithms/metaheuristics/genetic_algorithm.py#L500), independentemente de a pipeline refinar $P_0$ logo em seguida — aplicar 2-opt não redefina o que é o custo inicial. Em contrapartida, o **tempo** desse refinamento não pode desaparecer: como `PostInitLocalSearchStep` roda dentro do laço, `LocalSearchStep.run` já o envolve em `timer.measure("local_search", ...)` ([genetic_algorithm.py#L199-L215](../../src/algorithms/metaheuristics/genetic_algorithm.py#L199)), então a fase `local_search` da geração 1 absorve o custo normalmente e nenhuma instrumentação adicional é necessária;
    6. Documentar que `state.generation` passa a ter seu primeiro consumidor: hoje o campo é escrito em [genetic_algorithm.py#L513](../../src/algorithms/metaheuristics/genetic_algorithm.py#L513) e **nenhum** `GenerationStep` o lê.
  + **Why:** A remoção do refinamento automático da geração 0 (ver `BUG-21`) deixou o comportamento correto, mas eliminou também a **possibilidade** de pedir um warm start memético: `GeneticAlgorithm.__init__` já aceita `pipeline: list[GenerationStep] | None` ([genetic_algorithm.py#L252](../../src/algorithms/metaheuristics/genetic_algorithm.py#L252)), porém nenhum caminho a partir de `configs/` alcança esse argumento, e `PostInitLocalSearchStep` só existe como texto em [guide-local-search-pipeline-stages.md#L271](../../documentation/guide-local-search-pipeline-stages.md#L271). Sem esta tarefa, a única forma de refinar $P_0$ volta a ser editar o núcleo — exatamente o acoplamento que motivou a modularização. Um decorator de inicialização (`InitializationRefined`) resolveria o mesmo problema apenas para o slot `initialization` e exigiria especificações aninhadas no loader; a etapa de pipeline é preferível porque vale para **qualquer** inicialização e mantém `evolve()` sem conhecimento do estágio.
  + **Who:** Agente (`implementer`).
  + **Where:** [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L252) (parâmetro `pipeline` já existente; nenhuma alteração de `evolve()`), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L162) (`_build_strategy` e o bloco de composição declarativa), [strategies/**init**.py](../../src/algorithms/strategies/__init__.py) (exportação do registro), [algorithms.schema.json](../../configs/schemas/algorithms.schema.json), [algorithms.full.example.json](../../configs/algorithms.full.example.json), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L470) (`_get_strategy_params` e o bloco de `algorithm_config`), e os novos arquivos `src/algorithms/pipeline_steps.py` e `tests/algorithms/test_unit_pipeline_steps.py`.
  + **When:** Após a `Task 11.5`; independente da `Task 11.7`, mas deve preceder qualquer coorte que queira reportar o warm start.
  + **How:** Definir os três passos como classes com `run(self, state: GenerationState, timer: PhaseTimer) -> None` (o `Protocol` `GenerationStep` já existe em [genetic_algorithm.py#L75](../../src/algorithms/metaheuristics/genetic_algorithm.py#L75)); em `_build_strategy`, extrair um resolvedor genérico `_build_spec(spec)` que serve tanto para estratégias quanto para etapas, validando `class` contra o registro correspondente e levantando `ValueError` com a lista de chaves aceitas; construir `pipeline` depois de `init/sel/cx/mut/eval/ls` e passar `pipeline=built_pipeline or None` ao construtor do GA; adicionar `params.pipeline.items.params` com `additionalProperties: true` e `$ref` recursivo ao `strategy_slot`-like para a etapa; cobrir com teste que conte chamadas (exatamente 1 passada de `improve_batch` sobre $P_0$ em 5 gerações) e teste de contrato de config.
  + **How much:** Médio — ~250 linhas novas (2 arquivos), ~40 linhas em `config_loader.py`, ~35 linhas de schema/JSON, 1 doc.
  + **Stakeholder Implications:** O Capítulo 4 pode reportar explicitamente "inicialização aleatória + 2-opt apenas em $P_0$" como variante declarada no JSON, em vez de um comportamento oculto no núcleo; e a tese pode discutir warm start memético com evidência reproduzível (`strategy_params` + pipeline registrados no checkpoint).
  + **DoD:** Duas variantes com a mesma inicialização e a mesma semente produzem `initial_fitness` **idêntico** (o valor cru de $P_0$), mas `best_fitness` diferente, quando uma declara `pipeline: [PostInitLocalSearchStep]` e a outra não; o tempo do refinamento aparece em `phase_timings["local_search"]` da geração 1 e não em `initialization`; `PostInitLocalSearchStep` executa `improve_batch` exatamente uma vez em 10 gerações; `pytest tests/algorithms/test_unit_pipeline_steps.py -q` verde em CPU; `mypy src tests benchmark_scripts` e `ruff check src tests benchmark_scripts` com 0 erros; `python -m benchmark_scripts.check_docs` com `dead links: 0`; nome de etapa desconhecida em `configs/` levanta `ValueError` nomeando a chave.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pipeline_steps.py](../../src/algorithms/pipeline_steps.py#L25), [strategies/**init**.py](../../src/algorithms/strategies/__init__.py#L13), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L371), [config_loader.py](../../src/benchmarking/runner/config_loader.py#L164), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L542), [algorithms.schema.json](../../configs/schemas/algorithms.schema.json#L79), [algorithms.full.example.json](../../configs/algorithms.full.example.json#L54), [test_unit_pipeline_steps.py](../../tests/algorithms/test_unit_pipeline_steps.py#L1).
    - **Resumo:** Criado `src/algorithms/pipeline_steps.py` com `PostInitLocalSearchStep` e `PIPELINE_STEP_REGISTRY` exportado em `algorithms.strategies`. `AlgorithmConfig.instantiate` estendido para resolver `params["pipeline"]` com `step_defaults` e repassar ao construtor. `GeneticAlgorithm` compõe etapas de extensão mantendo os operadores evolutivos centrais (`SelectionStep`, `CrossoverStep`, etc.). Pipeline efetiva registrada em `algorithm_config` nos checkpoints.
    - **Atendimento ao DoD:** `pytest tests/algorithms/test_unit_pipeline_steps.py -q` → `25 passed`; `initial_fitness` idêntico entre a variante com e sem pipeline e `best_fitness` menor na variante memética; `timings["local_search"]["calls"] == 1` em 10 gerações; etapa desconhecida levanta `ValueError`; `mypy src tests benchmark_scripts` → `Success: no issues found in 138 source files`; `ruff check src tests benchmark_scripts` → `All checks passed!`; `check_docs` → exit 0.
    - **Correção residual (2026-11-10, `BUG-30`):** `_serialize_custom_pipeline` em [orchestration.py](../../src/benchmarking/runner/orchestration.py#L92) passou a gravar `{"class", "params"}` a partir de `public_params()` e a ler `algorithm_instance.custom_pipeline_steps`, eliminando a lista de nomes sem parâmetros; `configs/algorithms.full.example.json` ganhou a variante `ExperimentalMemeticPipeline` com `local_search_stage: "none"` e as três etapas declaradas, e o teste `test_example_config_never_combines_pipeline_and_local_search_stage` impede a reintrodução da dupla passada de 2-opt sobre $P_0$.
    - **Proof of Concept (PoC):**

      ```python
      # orchestration.py — parâmetros declarados da etapa chegam ao checkpoint
      custom_steps = list(getattr(algorithm_instance, "custom_pipeline_steps", None) or [])
      if custom_steps:
          results["algorithm_config"]["pipeline"] = _serialize_custom_pipeline(custom_steps)
      # -> [{"class": "EliteOnlyLocalSearchStep", "params": {"elite_count": 8}}, ...]
      ```

      ```bash
      rtk .venv/bin/python -m pytest tests/algorithms/test_unit_pipeline_steps.py -q -k "serialize or example_config"
      # -> 2 passed
      ```

- [x] **[REVIEWED]** **Task 11.7: Persistir e Reportar os Tempos por Fase e Contabilizar Latências de Transferência/Overhead (`DP11-7`, `DP11-13`, `BUG-30`)**
  + **What:**
    1. Fazer `run_single_algorithm` capturar `stats["phase_timings"]` de cada repetição, acumulando `raw_phase_timings` e `mean_phase_timings` por algoritmo;
    2. Calcular `phase_share_pct`, a fração de `mean_time` consumida por `initialization`, `selection`, `crossover`, `mutation`, `local_search`, `evaluation` e `host_sync`;
    3. Incorporar a medição de latências de sincronização D2H e overhead de laço (como os ~0.141 ms de `_extract_best` e ~0.4 ms de overhead de loop/estado) em um **7.º bucket explícito `host_sync`** no `PhaseTimer` (`DP11-13`), mantendo `evaluation` estritamente puro (`evaluate_batch`) e garantindo fechamento total da equação $\text{wall} = \sum t_{\text{fases}} + t_{\text{host\_sync}}$ sem lacunas não explicadas;
    4. Fazer `_save_problem_statistics_file` gravar `mean_phase_timings` em `*_stats.json`;
    5. Adicionar ao `report_generator` uma tabela "tempo médio por fase" em Markdown e LaTeX, com filtro/alerta de piso de runtime (sinalizando células com runtime $< 100\text{ ms}$ onde a fração de overhead relativo é artificialmente elevada);
    6. Medir a estabilidade das frações por fase ao longo de repetições reais (8 ou mais sementes por célula, orçamento fixo de gerações), reportando média, desvio-padrão e coeficiente de variação por fase, além da dispersão de `sum(fases)/wall` — sem essa medida as frações não sustentam nenhuma afirmação no Capítulo 4.
  + **Why:** O `PhaseTimer` mede as seis fases e `evolve()` devolve tudo em `stats["phase_timings"]` ([genetic_algorithm.py#L576](../../src/algorithms/metaheuristics/genetic_algorithm.py#L576)), mas o runner descarta a chave: os checkpoints só têm `raw_times` total. Sem essa informação não é possível responder qual operador domina o tempo, nem justificar no Capítulo 4 onde o esforço é gasto (por exemplo, a proporção entre `local_search` e `evaluation`). Além disso, medições de 2026-11-06 demonstraram que ~1 ms de overhead por run (0.141 ms por chamada de `_extract_best` + loop) ficava fora dos cronômetros, representando ~7 % em runs de 14 ms; a contabilização explícita garante transparência total sem necessidade de otimizar prematuramente funções com custo de frações de milissegundo.
  + **Premissa medida e descartada (2026-11-06):** a versão anterior desta tarefa afirmava que `use_cuda_events=False` fazia as fases GPU medirem enfileiramento em vez de execução. Medição com três relógios independentes sobre `LocalSearchGPUTwoOpt.improve_batch` com população residente em GPU, 5 repetições, mediana: `berlin52` → `perf_counter` 2,24 ms, `cp.cuda.Event` 2,25 ms, `perf_counter` com drenagem forçada 2,20 ms; `pr1002` → 77,02 / 77,12 / 76,96 ms. As três concordam dentro de 0,5 %, ou seja, a chamada **bloqueia** apesar de não conter `synchronize()` explícito, e o `perf_counter` já mede o custo real. Não há déficit de medição a corrigir; `use_cuda_events=True` não alteraria nenhum número e saiu do escopo.
  + **O que sobra de não medido:** as operações de `evolve()` fora de qualquer `timer.measure` — `_extract_best` (três sincronizações por geração: `argmin().get()`, `float(...get())` e `asnumpy` em [genetic_algorithm.py#L408-L416](../../src/algorithms/metaheuristics/genetic_algorithm.py#L408)), a construção de `GenerationState`, o laço de passos e a aritmética de parada antecipada. Em `berlin52` com 1 geração isso explica ~7,8 % de um run de 6,7 ms; em `pr1002` com 2 gerações, ~0,2 % de 446 ms. É custo fixo de host, não ruído.
  + **Who:** Agente (`implementer`).
  + **Where:** [phase_timer.py](../../src/utils/phase_timer.py#L22) (apenas leitura do formato de `summary()`), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L576), [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L126) (dicionários de acumulação e sumário final em [#L261](../../src/benchmarking/runner/algorithm_runner.py#L261)), [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py#L256) (consistência dos arrays `phase_timings`), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L143), [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L450), `tests/benchmarking/test_unit_checkpoint_io.py` e `tests/benchmarking/test_integration_pipeline.py`.
  + **When:** Após a `Task 11.5`.
  + **How:** Em `run_single_algorithm`, inicializar `results["phase_timings"] = []` e no laço de repetições anexar `stats.get("phase_timings", {})`; no sumário, reduzir por fase para `mean_phase_timings[phase] = {"total_ms", "calls", "mean_ms"}`, derivar `phase_share_pct` de `sum(total_ms)` e calcular `phase_share_cv_pct` sobre as repetições; em `is_valid_checkpoint`, aceitar a chave como opcional mas rejeitar quando `len(phase_timings) != successful_runs`; em `report_generator`, emitir a tabela apenas quando todos os algoritmos tiverem a chave e sempre acompanhada do CV, para não quebrar coortes antigas nem publicar fração sem dispersão.
  + **How much:** Médio — ~120 linhas em `algorithm_runner`/`orchestration`, ~60 em `report_generator`, ~35 no harness de variância, 2 testes.
  + **Stakeholder Implications:** O Capítulo 4 passa a poder justificar o custo de cada operador com dados medidos, em vez de estimativas; o orientador consegue verificar que o tempo total de uma execução é explicado pelas fases somadas.
  + **DoD:** Um checkpoint novo contém `mean_phase_timings` com as seis fases, `phase_share_pct` e `phase_share_cv_pct`; a soma de `total_ms` das fases fica dentro de 5 % de `mean_time` nas **quatro** variantes (medido: 99,8–100,0 % em `pr1002`, 92,2 % em `berlin52` com 1 geração); a tabela do relatório traz média, desvio-padrão e CV por fase, medidos sobre ≥8 repetições, e nenhuma fração é emitida sem o seu CV; o relatório Markdown/LaTeX gerado contém a tabela; `pytest tests/benchmarking -q` verde; `mypy` e `ruff` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L474), [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L59), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L511), [schema.sql](../../src/benchmarking/database/schema.sql#L95), [db_operations.py](../../src/benchmarking/database/db_operations.py#L188), [records.py](../../src/data_models/records.py#L116), [test_unit_runner_telemetry.py](../../tests/benchmarking/test_unit_runner_telemetry.py#L1), [test_unit_best_tour_transfer.py](../../tests/algorithms/metaheuristics/test_unit_best_tour_transfer.py#L1), [test_unit_db_operations.py](../../tests/benchmarking/database/test_unit_db_operations.py#L520), [test_integration_pipeline.py](../../tests/benchmarking/test_integration_pipeline.py#L380).
    - **Resumo:** Tabela 4 (Markdown e LaTeX) passou a emitir `survival` antes de `host_sync`, fechando `Sum/Wall %`; `phase_share_cv_pct` usa desvio amostral (`ddof=1`); `evolve` mede apenas o custo por geração (`_best_cost`, um escalar D2H) e materializa `best_tour` somente em melhoria estrita; `incremental_mode` reconstrói os totais absolutos de H2D/D2H/kernels a partir de `mean_*` do checkpoint e re-média sobre `carried_runs + new_runs`; `mean_phase_timings`, `phase_share_pct` e `phase_share_cv_pct` viraram colunas `JSON` no `schema.sql`, no `BenchmarkRunInput` e no `INSERT`.
    - **Atendimento ao DoD:** `grep -n '"survival"' src/benchmarking/analysis/report_generator.py --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` -> linhas 481 e 548; `grep -n "ddof=1" src/benchmarking/runner/algorithm_runner.py` -> 1; `grep -n "phase_timings\|phase_share" src/benchmarking/database/schema.sql src/benchmarking/database/db_operations.py src/data_models/records.py` -> 10 (antes 0); `pytest tests/benchmarking tests/algorithms -q` -> `619 passed` com `test_phase_share_cv_uses_sample_standard_deviation`, `test_incremental_mode_carries_transfer_and_launch_telemetry`, `test_best_tour_is_transferred_only_on_strict_improvement`, `test_insert_benchmark_run_persists_phase_telemetry` e `test_generate_result_tables_emits_phase_timings_table` (asserções em `Surv (%)` / `Surv &`); `mypy src tests benchmark_scripts` -> 140 arquivos, 0 erros; `ruff check src tests benchmark_scripts` -> `All checks passed!`; `check_docs` exit 0.
    - **Proof of Concept (PoC):**

      ```python
      # genetic_algorithm.py::evolve — custo por geração; D2H da rota só em melhoria estrita
      gen_best_cost = self.timer.measure("host_sync", self._best_cost, state.fitness)
      if gen_best_cost < best_cost - 1e-6:
          best_tour, _ = self.timer.measure("host_sync", self._extract_best, state.population, state.fitness)
          best_cost = gen_best_cost
      ```

      ```bash
      .venv/bin/python -m pytest tests/benchmarking/test_unit_runner_telemetry.py tests/algorithms/metaheuristics/test_unit_best_tour_transfer.py -q  # -> 3 passed
      ```

- [x] **[REVIEWED]** **Task 11.8: Remover a Busca Local da Geração 0 e Centralizar o Despacho de Estágios em uma Tabela Única (`DP11-8`, `DP11-9`, `BUG-21`, `BUG-30`)**
  + **What:** Eliminar o bloco pré-laço remanescente em `evolve`, de modo que a geração 0 seja sempre `initialization.initialize()` seguida de `evaluation.evaluate_batch()` crua, para os cinco estágios (`none`, `post_crossover`, `post_mutation`, `post_evaluation`, `post_evolution`); e, no mesmo change, substituir as cadeias de `if` que hoje decidem **onde** a busca local entra por uma tabela única de estágios, para que a duplicação que produziu o `BUG-21` não sobreviva à sua correção. Atualizar a documentação dos estágios e fixar o contrato com teste parametrizado nos cinco valores.
  + **Why:** O refinamento automático da geração 0 foi removido para `post_crossover` e `post_mutation`, mas `post_evaluation` continua refinando $P_0$ antes da primeira geração ([genetic_algorithm.py#L473-L491](../../src/algorithms/metaheuristics/genetic_algorithm.py#L473-L491)). Isso contradiz o nome do estágio (refinar sobreviventes **após** a avaliação de uma geração, não antes da primeira), produz duas passadas consecutivas de 2-opt sem nenhum operador genético entre elas, e faz `initial_fitness` depender do estágio escolhido — a mesma classe de artefato que motivou a correção anterior.
  + **Por que a centralização vem junto (auditoria de 2026-11-06):** o despacho comportamental de `local_search_stage` está espalhado por três cadeias dentro do **mesmo** arquivo — `_build_default_pipeline` ([#L311](../../src/algorithms/metaheuristics/genetic_algorithm.py#L311) guarda `has_ls`, [#L314](../../src/algorithms/metaheuristics/genetic_algorithm.py#L314) `post_crossover`, [#L319](../../src/algorithms/metaheuristics/genetic_algorithm.py#L319) `post_mutation`, [#L325](../../src/algorithms/metaheuristics/genetic_algorithm.py#L325) `post_evaluation`), o pré-laço de `evolve` ([#L470](../../src/algorithms/metaheuristics/genetic_algorithm.py#L470) e [#L473](../../src/algorithms/metaheuristics/genetic_algorithm.py#L473), `post_evaluation` **de novo**) e o pós-laço ([#L547](../../src/algorithms/metaheuristics/genetic_algorithm.py#L547), `post_evolution`). `post_evaluation` é o único estágio tratado em **dois** sítios, e é exatamente por isso que ele refina $P_0$: a duplicação **é** o bug, não um vizinho dele. Remover só o bloco pré-laço fecha o sintoma e deixa o mecanismo intacto para o sexto estágio. A boa notícia é que todo o despacho vive em `genetic_algorithm.py`, então a tabela é local a um arquivo: `{estágio: (passos a inserir, posição, hook pré/pós-laço)}`, com `_build_default_pipeline` e `evolve` lendo a mesma fonte.
  + **Who:** Agente (`implementer`).
  + **Where:** [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L304-L340) (tabela de estágios), [#L465-L496](../../src/algorithms/metaheuristics/genetic_algorithm.py#L465-L496) (remoção do pré-laço), [#L547](../../src/algorithms/metaheuristics/genetic_algorithm.py#L547) (hook pós-laço lido da tabela), [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L236) (seção 2.4) e [#L271](../../documentation/guide-local-search-pipeline-stages.md#L271) (Pattern 1, que passa a ser o único caminho para refinar $P_0$), [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L57) e o novo `tests/algorithms/metaheuristics/test_unit_initial_population_raw.py`.
  + **When:** Após a `Task 11.6`, para que a capacidade de refinar $P_0$ exista antes de remover a última implementação implícita dela.
  + **How:** Introduzir um `StageSpec` (dataclass congelada) com os campos `pipeline_steps`, `post_loop_hook` e `requires_local_search`, e um `Mapping[LocalSearchStage, StageSpec]` cobrindo os cinco nomes; `_build_default_pipeline` passa a montar a lista a partir de `StageSpec.pipeline_steps` e `evolve` substitui o bloco pré-laço por uma avaliação única `fitness = self.timer.measure("evaluation", self.evaluation.evaluate_batch, population, context)` com um comentário de uma linha sobre a geração 0 ser crua, lendo o hook pós-laço da mesma tabela; um teste de completude parametrizado sobre `get_args(LocalSearchStage.__value__)` exige que **todo** valor do `Literal` tenha exatamente uma `StageSpec` e que nenhuma `StageSpec` exista para nome fora dele (é o teste que as três cadeias de `if` tornavam impossível escrever); acrescentar o parágrafo "Generation 0: not refined" na seção 2.4 do guia; escrever teste parametrizado nos cinco estágios que instancia o GA com a **mesma** semente, `InitializationRandom` e `LocalSearchCPUTwoOpt`, executa `evolve` com `max_generations=1` e exige `initial_fitness` idêntico entre os estágios (o valor cru é função apenas da semente).
  + **How much:** Médio — ~30 linhas removidas, ~70 de tabela/`StageSpec`, ~70 de teste, 1 parágrafo de documentação.
  + **Stakeholder Implications:** `initial_cost` e `mean_improvement` passam a ter o mesmo significado em todos os estágios e coortes, e a tese deixa de apresentar duas passadas de 2-opt sem contrapartida genética; além disso, um sexto estágio futuro passa a exigir **uma** entrada em vez de três edições coordenadas.
  + **DoD:** `grep -n "local_search_stage ==" src/algorithms/metaheuristics/genetic_algorithm.py --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0 ocorrências; o teste de completude falha se um nome for adicionado ao `Literal` sem entrada na tabela; o teste parametrizado dos cinco estágios passa com `initial_fitness` idêntico (mesma semente) e falha contra o código anterior em `post_evaluation`; `pytest tests/algorithms tests/validation -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L231), [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L229), [test_unit_initial_population_raw.py](../../tests/algorithms/metaheuristics/test_unit_initial_population_raw.py#L1).
    - **Resumo:** Removida a busca local pré-laço de Geração 0 (`BUG-21`), garantindo avaliação crua em todos os 5 estágios. Substituídas as três cadeias dispersas de `if local_search_stage == ...` pela tabela declarativa `STAGE_SPECS: Mapping[LocalSearchStage, StageSpec]` com `pipeline_steps`, `post_loop_hook` e `requires_local_search`.
    - **Atendimento ao DoD:** `grep -n "local_search_stage ==" src/algorithms/metaheuristics/genetic_algorithm.py --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → **0 ocorrências**; `pytest tests/algorithms/metaheuristics/test_unit_initial_population_raw.py -q` → `6 passed` (bijeção com `LocalSearchStage` e `initial_fitness` idêntico nos 5 estágios); `pytest tests/algorithms tests/benchmarking -q` → `609 passed`; `mypy`/`ruff` limpos; `check_docs` exit 0.
    - **Correção residual (2026-11-10, `BUG-30`):** `_compose_pipeline` deixou de comparar `type(step).__name__` e passou a despachar por `getattr(step, "stage_anchor", "post_survival")` ([genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L389)), preservando a ordem declarada entre etapas do mesmo âncora; `custom_pipeline_steps` registra o que o chamador declarou; `_all_strategies()` percorre `self.pipeline` (via `step.strategies()` com fallback em `_PIPELINE_STRATEGY_ATTRS`) e deduplica por `id()`, de modo que `_reset_run_metrics` e `_collect_strategy_metrics` alcançam estratégias que só existem dentro de etapas.
    - **Proof of Concept (PoC):**

      ```python
      # genetic_algorithm.py::_compose_pipeline — âncora declarativa, não nome de classe
      anchor: str = getattr(step, "stage_anchor", "post_survival")
      if anchor == "pre_selection":
          pre_selection_steps.append(step)      # inseridas no índice 0, na ordem declarada
      elif anchor == "pre_evaluation":
          steps.insert(eval_offspring_index, step)   # antes de EvaluationStep(target="offspring")
      else:
          steps.append(step)
      ```

      ```bash
      rtk .venv/bin/python -m pytest tests/algorithms/test_unit_pipeline_steps.py -q -k "anchor or all_strategies or reset_run_metrics"
      # -> 5 passed
      ```

- [x] **[REVIEWED]** **Task 11.9: Implementar `EliteOnlyLocalSearchStep` e `PeriodicLocalSearchStep` como Etapas de Pipeline Reais (`DP11-10`, `BUG-30`)**
  + **What:** Implementar em `src/algorithms/pipeline_steps.py` (criado na `Task 11.6`) as duas etapas que hoje existem **apenas como texto** na documentação: `EliteOnlyLocalSearchStep` (refina somente os `elite_count` melhores indivíduos da população, após o `SurvivalStep`) e `PeriodicLocalSearchStep` (refina a população inteira a cada `interval` gerações, usando `state.generation` como contador). Registrar ambas em `PIPELINE_STEP_REGISTRY`, cobrir com teste e substituir os templates do guia por referências ao código real.
  + **Why:** [Task 7.1](#fase-7-utilitários-de-residência-cpugpu-e-rng-variantes-de-2-opt-gpu-inversion-mutation-e-integração-nativa-com-tspduckdb) está corretamente marcada `[x]`: seu escopo era **documentar** quatro padrões de extensão via `GenerationStep`, e ela o cumpriu. Nenhuma tarefa de implementação foi jamais criada, então o guia publica dois blocos de código copiáveis que nenhum teste executa e que podem divergir silenciosamente do `GenerationState` real — a mesma classe de problema que a `Task 11.8` está fechando do lado do despacho de estágios. A auditoria de 2026-11-06 confirmou zero ocorrências de ambos os nomes em `src/` e `tests/`. Esta tarefa fecha a lacuna entre o que o guia promete e o que o pacote entrega.
  + **Who:** Agente (`implementer`).
  + **Where:** novo `src/algorithms/pipeline_steps.py` (criado pela `Task 11.6`), [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L402) (`EliteOnlyLocalSearchStep`) e [#L512](../../documentation/guide-local-search-pipeline-stages.md#L512) (`PeriodicLocalSearchStep`), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L75) (Protocol `GenerationStep` e `GenerationState`), `tests/algorithms/test_unit_pipeline_steps.py`.
  + **When:** Após a `Task 11.6` (o módulo e o registro precisam existir) e independente das `Tasks 11.7` e `11.8`. Não bloqueia nenhuma coorte.
  + **How:** Ambas as classes implementam `run(self, state: GenerationState, timer: PhaseTimer) -> None` e delegam a medição a `timer.measure("local_search", ...)`, para que o custo apareça na fase correta em vez de desaparecer; `EliteOnlyLocalSearchStep` seleciona os índices dos `elite_count` menores `state.fitness`, refina apenas esse subconjunto com `improve_batch` e reescreve as linhas correspondentes de `state.population`, **preservando a ordem relativa e sem reordenar o resto da população** (caso contrário o passo vira seleção disfarçada); `PeriodicLocalSearchStep` dispara quando `state.generation % interval == 0` e é um no-op nos demais, aceitando `interval >= 1` e rejeitando `interval <= 0` no construtor. Testes: contagem de chamadas de `improve_batch` em 10 gerações (`EliteOnly` → 10 chamadas com `elite_count` rotas cada; `Periodic(interval=3)` → 3 chamadas), invariante de que o custo nunca piora após `EliteOnly`, e teste de que ambas preservam a validade da permutação em cada rota.
  + **How much:** Baixo — ~90 linhas de implementação, ~80 de teste, 2 edições de documentação.
  + **Stakeholder Implications:** O guia passa a descrever código que existe e é testado, e o Capítulo 4 ganha duas variantes de refinamento local declaráveis no JSON sem tocar no núcleo — a mesma promessa que a `Task 11.6` faz para $P_0$.
  + **DoD:** `grep -rn "class EliteOnlyLocalSearchStep\|class PeriodicLocalSearchStep" src/algorithms/pipeline_steps.py --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 2 ocorrências; ambas as classes constam em `PIPELINE_STEP_REGISTRY`; o teste de contagem falha se `PeriodicLocalSearchStep` ignorar `interval`; `pytest tests/algorithms/test_unit_pipeline_steps.py -q` verde; `mypy src tests` e `ruff check src tests` limpos; `python -m benchmark_scripts.check_docs` com `undefined symbols: 0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [pipeline_steps.py](../../src/algorithms/pipeline_steps.py#L65), [guide-local-search-pipeline-stages.md](../../documentation/guide-local-search-pipeline-stages.md#L324), [test_unit_pipeline_steps.py](../../tests/algorithms/test_unit_pipeline_steps.py#L235).
    - **Resumo:** Implementados `EliteOnlyLocalSearchStep` (refinando apenas os `elite_count` indivíduos preservando os demais sem reordenação) e `PeriodicLocalSearchStep` (disparo a cada `interval` gerações com suporte a `offspring` e `population`). Validação estrita de parâmetros e registro em `PIPELINE_STEP_REGISTRY`.
    - **Atendimento ao DoD:** `grep -rn "class EliteOnlyLocalSearchStep\|class PeriodicLocalSearchStep" src/algorithms/pipeline_steps.py` → **2 ocorrências**; ambas constam em `PIPELINE_STEP_REGISTRY`; `test_periodic_step_execution_counting_and_interval` falha se `interval` for ignorado (`executed_generations == [3, 6, 9]`); `pytest tests/algorithms/test_unit_pipeline_steps.py -q` → `25 passed`; `mypy`/`ruff` limpos; `check_docs` → `undefined symbols: 0`.
    - **Correção residual (2026-11-10, `BUG-30`):** as três etapas passaram a expor `stage_anchor`, `strategies()` e `public_params()` ([pipeline_steps.py](../../src/algorithms/pipeline_steps.py#L27)), o que deu à `Task 11.8` a ancoragem declarativa e à `Task 11.6` os parâmetros (`elite_count`, `interval`, `target`) que faltavam no checkpoint.
    - **Proof of Concept (PoC):**

      ```python
      class EliteOnlyLocalSearchStep(_BaseLocalSearchPipelineStep):
          stage_anchor: ClassVar[StageAnchor] = "post_survival"
          def public_params(self) -> dict[str, Any]:
              return {"elite_count": self.elite_count}
      ```

      ```bash
      rtk .venv/bin/python -m pytest tests/algorithms/test_unit_pipeline_steps.py -q -k "periodic or elite_only"
      # -> 8 passed
      ```

- [x] **[REVIEWED]** **Task 11.10: Corrigir a Complexidade e a Ocupação de `order_crossover_kernel` (`DP11-11`, `DP11-19`, `DP11-20`, `BUG-22`, `BUG-24`, `OX-A`, `OX-B`)**
  + **What:** Atacar os limites independentes do kernel de Order Crossover modular: (i) substituir o teste de pertinência por varredura linear por uma **tabela de pertinência em memória compartilhada no Formato A** (`uint8_t in_segment[n]`, 1 byte por cidade, livre de `atomicOr` e bit shifts para todas as instâncias TSPLIB95 $n \le 2{,}103$ até $n = 6{,}144$; o Formato B bitpacked permanece registrado para $n > 10{,}000$ na `DP12-1`), levando cada rota de $\mathcal{O}(n^2)$ para $\mathcal{O}(n)$; (ii) passar de **uma thread por rota** para **uma warp por rota**, elevando a ocupação para saturação dos 5 SMs da GTX 1050; e (iii) corrigir as **3 divergências algorítmicas CPU vs. GPU** (`OX-A`): pareamento de pais disjuntos $(2k, 2k+1)$ em vez de anel cíclico, amostragem de cortes sem reposição (`replace=False`, $c_1 < c_2$) em vez de inteiros com reposição, e preservação de nós de depósito nas pontas em rotas fechadas. Reexecutar a varredura de dimensionamento e registrar a curva nova.
  + **Why:** Medido em 2026-11-06 com $N=256$ (`CrossoverOrder` vs `CrossoverGPUOrder`, mediana de 3 chamadas após aquecimento): o GPU vence por 11,2× em $n=52$ (0,78 vs 8,79 ms), empata em $n \approx 800$ e perde por 3,6× em $n=2000$ (742,05 vs 206,81 ms). O ajuste de potência expõe causas distintas, não um fator constante: CPU $\propto n^{0.87}$ (linear), GPU $\propto n^{1.88}$ (quadrática). O termo quadrático vem da varredura `for j in [c1..c2]` ([order_crossover.cu#L49-L54](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L49)) dentro de um laço de $n$ cidades, enquanto a versão CPU usa `in_segment = set(...)` ([order_crossover.py#L67](../../src/algorithms/strategies/crossover/order_crossover.py#L67)), $\mathcal{O}(1)$. A segunda causa é independente: `blocks = (256 + 128 - 1) // 128 = 2` ([order_crossover.py#L198](../../src/algorithms/strategies/crossover/order_crossover.py#L198)) ocupa 2 dos 5 SMs da GTX 1050 — 40 % da GPU ociosa — e **nenhuma** correção de pertinência resolve isso, porque a paralelização está limitada ao tamanho da população. A terceira causa é a quebra de comparabilidade: a CPU gera filhos simétricos em pares disjuntos com cortes estritos, enquanto a GPU cruza em anel cíclico com cortes degenerados ($c_1=c_2$). Consequência para a tese: em `FullGPUModular` o crossover domina a geração (63,6 % em `pr1002`), então qualquer speedup reportado mede esse freio e uma semântica divergente.
  + **Who:** Agente (`implementer`).
  + **Where:** [order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L10-L60), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L161-L215) (lançador), `tests/algorithms/strategies/crossover/test_unit_gpu_order_crossover.py` *(a criar)*, e o harness de dimensionamento em `benchmark_scripts/`.
  + **When:** Independente das `Tasks 11.6`–`11.9`. Deve preceder qualquer coorte que reporte speedup de `FullGPUModular` acima de $n \approx 800$, porque até lá o número mede o kernel defeituoso.
  + **How:** (i) Declarar `extern __shared__ unsigned char s_in_seg[]` (Formato A, $n$ bytes por warp ativa no bloco), zerá-la em paralelo (`for (int c = lane_id; c < n; c += 32) s_in_seg[c] = 0;`), marcá-la sem contenção durante a cópia do segmento (`s_in_seg[city] = 1;`) e trocar o laço interno por uma leitura direta $\mathcal{O}(1)$ (`s_in_seg[city] == 0`). (ii) Uma warp por rota: copiar o segmento em paralelo pelos 32 lanes e compactar as cidades de `parent2` que não estão no segmento com `__ballot_sync` (máscara dos lanes que retêm) mais um prefixo de população via `__popc` da máscara de lanes inferiores (`(1u << lane_id) - 1u`), que dá a cada lane a sua posição de escrita contígua e estritamente determinística. (iii) Alinhar o lançador e amostragem de cortes em python (`OX-A`): parear pais $(2k, 2k+1)$, usar `rng.choice(n, size=2, replace=False)` garantindo $c_1 < c_2$, respeitar o intervalo $[1, n-2]$ quando $p_1[0] == p_1[-1]$, e remover qualquer `Stream.null.synchronize()` incondicional. **Restrição inegociável:** para os mesmos pais e os mesmos pontos de corte, a prole em GPU deve permanecer **bit-idêntica** à de `CrossoverOrder`.
  + **How much:** Alto — ~140 linhas de kernel, ~50 de lançador, ~100 de teste, harness de dimensionamento.
  + **Stakeholder Implications:** Sem esta tarefa, o Capítulo 4 reporta que a GPU é mais lenta em instâncias grandes por causa de um kernel defeituoso e divergente, e não por uma propriedade do hardware.
  + **DoD:** Para $N=256$ e $n \in \{52, 100, 200, 500, 1000, 2000\}$, a prole de `CrossoverGPUOrder` é **bit-idêntica** à de `CrossoverOrder` com os mesmos pais e os mesmos cortes (`OX-A`); a curva de tempo do GPU deixa de ser quadrática (ajuste de potência $\le 1.2$) e o GPU volta a vencer ou empatar em $n=2000$; a varredura de dimensionamento está versionada e o relatório imprime a curva antes/depois; `pytest tests/algorithms/strategies/crossover -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu#L25), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L181), [test_unit_gpu_order_crossover.py](../../tests/algorithms/strategies/crossover/test_unit_gpu_order_crossover.py#L45).
    - **Resumo:** `order_crossover_kernel` passou a usar tabela de pertinência em shared memory no Formato A (`s_in_seg`), uma warp por rota e compactação determinística via `__ballot_sync`/`__popc`; o lançador pareia pais disjuntos, amostra cortes com o helper compartilhado `sample_distinct_pairs` e resolve limites por `resolve_route_bounds`. A prole permanece bit-idêntica a `CrossoverOrder` (`OX-A`), e `BUG-24` desapareceu junto com `_is_closed_cache`.
    - **Atendimento ao DoD:** `grep -n "s_in_seg\|__ballot_sync\|__popc" src/algorithms/strategies/crossover/kernels/order_crossover.cu` → presente (linhas 25, 115, 118, 122); `grep -rn "_is_closed_cache" src/ --exclude-dir=__pycache__` → 0; `pytest tests/algorithms/strategies/crossover -q` → `41 passed`, incluindo `test_gpu_order_crossover_matches_cpu_bit_identical`, `test_gpu_order_crossover_recovers_from_zero_filled_warmup` e `test_gpu_order_crossover_preserves_closed_route_depot_endpoints`; `ruff`/`mypy` limpos e `check_docs` exit 0 (registro completo em [§8 do walkthrough](../../documentation/walkthroughs/analysis-phase-11-strategy-unification-and-gpu-kernels-walkthrough.md#8-closure-of-the-stochastic--topology-batch-tasks-1110-1112-1114-1115)).
    - **Proof of Concept (PoC):**

      ```cuda
      // order_crossover.cu — pertinência O(1) em shared memory + compactação por warp
      extern __shared__ unsigned char s_in_seg[];
      unsigned int mask = __ballot_sync(0xFFFFFFFFu, retain);
      if (retain) {
          int rank = __popc(mask & lower_mask);
      }
      dest_offset += __popc(mask);
      ```

      ```bash
      .venv/bin/python -m pytest tests/algorithms/strategies/crossover -q  # -> 41 passed
      ```

- [x] **[REVIEWED]** **Task 11.11: Executar Warm-up Run Obrigatório em Benchmarks com GPU (`DP11-12`, `DP11-21`, `BUG-23`, `BUG-25`)**
  + **What:** Introduzir uma execução de aquecimento (warm-up run) preliminar descartada no runner de benchmarks (`src/benchmarking/runner/`) para qualquer execução de variante que utilize backend GPU, antes de iniciar o laço oficial de repetições medidas (`repetitions`).
  + **Why:** Medido em 2026-11-06 (`berlin52`, `FullGPUModular`): a repetição 1 leva ~66 ms, enquanto as repetições 2–5 levam ~14 ms estáveis. O atraso na repetição 1 decorre da inicialização pontual do driver CUDA, alocação inicial de pools de memória CuPy e compilação JIT de kernels CUDA (`order_crossover.cu`, `two_opt.cu`). Sem warm-up, esse custo inicial contamina `mean_time` e infla artificialmente o coeficiente de variação de parede (`wall_cv_pct` medido em até 157 % em execuções curtas), distorcendo as conclusões e a comparabilidade estatística do Capítulo 4.
  + **Who:** Agente (`implementer`).
  + **Where:** [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L143), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L470) e novo teste `tests/benchmarking/test_unit_runner_warmup.py`.
  + **When:** Concomitante à `Task 11.7` e anterior à execução de novas coortes de benchmark com GPU.
  + **How:** No runner (`run_single_algorithm` ou orquestrador), inspecionar se a variante configurada utiliza GPU (via verificação de backend/estratégias GPU). Se positivo, disparar 1 execução descartada preliminar com os mesmos parâmetros do problema (ou 1 geração caso o algoritmo seja parametrizável) sem gravar medições ou checkpoints. Isso assegura que contexto, alocadores e kernels JIT já residam aquecidos na VRAM/driver antes da coleta estatística. Adicionar teste unitário com mock/spy confirmando o disparo de 1 rodada de warm-up descartada quando GPU está ativa, e 0 rodadas quando em modo puramente CPU.
  + **How much:** Baixo — ~25 linhas no runner, ~40 linhas de teste.
  + **Stakeholder Implications:** Elimina a contaminação de inicialização de hardware nas tabelas de resultado do Capítulo 4, estabilizando CVs e garantindo que as métricas reflitam fielmente o tempo de processamento algorítmico.
  + **DoD:** Runner executa uma rodada preliminar descartada antes de `repetitions` se a estratégia contiver GPU; teste unitário `tests/benchmarking/test_unit_runner_warmup.py` verifica que a rodada descartada ocorre para GPU e é omitida para CPU; repetição 1 medida de `berlin52` em GPU tem tempo dentro de 20 % da mediana das repetições subsequentes; `pytest tests/benchmarking -q` verde; `mypy src tests` e `ruff check src tests` limpos; `python -m benchmark_scripts.check_docs` com `undefined symbols: 0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L59), [base.py](../../src/algorithms/strategies/base.py#L77), [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L472), [pipeline_steps.py](../../src/algorithms/pipeline_steps.py#L71), [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L114), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L436), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L162), [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L181), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L131), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L131), [test_unit_runner_warmup.py](../../tests/benchmarking/test_unit_runner_warmup.py#L1).
    - **Resumo:** O `evolve(max_generations=1)` cego foi substituído pelo protocolo AOT `warmup(context)`: `StrategyBase` compila via `_get_kernel()`, as seis estratégias GPU compilam **e lançam** uma vez sobre `(rows, n)` de `arange` (nunca `cp.zeros`, `BUG-24`), os três passos de pipeline (`PostInitLocalSearchStep`, `EliteOnlyLocalSearchStep`, `PeriodicLocalSearchStep`) delegam aos seus `strategies()`, e `GeneticAlgorithm.warmup` percorre passos + estratégias sem consultar `state.generation` — logo o `interval=10` deixa de importar. O runner aquece um clone (`copy.deepcopy` quando `algorithm_factory is None`) e **falha o run** se `_compile_kernel_cached.cache_info().currsize` mudar entre warm-up e fim das repetições.
    - **Atendimento ao DoD:** `grep -rn "def warmup" src/ --exclude-dir=__pycache__` -> 10 definições (base, 6 GPU, 2 passos, GA); probe GPU `burma14` com `PeriodicLocalSearchStep(interval=10)`, `max_generations=12`, 3 repetições -> `raw_times_ms [18.4, 18.2, 12.6]`, `rep1/median = 1.014x` (medido 26.27x / 1651.73 ms vs 64.44 ms antes) e `currsize` constante em 5; `pytest tests/benchmarking/test_unit_runner_warmup.py -q` -> `8 passed` (clone isolado, 0 warm-ups em CPU, falha tolerada, incremental, JIT no laço rejeitado, `PeriodicLocalSearchStep` aquecido em CPU-twin); `pytest tests/benchmarking tests/algorithms -q` -> `619 passed`; `mypy` 140 arquivos/0 erros; `ruff` limpo; `check_docs` exit 0.
    - **Proof of Concept (PoC):**

      ```python
      # algorithm_runner.py — clone isolado + protocolo AOT + prova de zero JIT no laço medido
      warm_instance = copy.deepcopy(algorithm_instance) if algorithm_factory is None else algorithm_factory(seed_base)
      warm_instance.warmup(warmup_context)
      jit_cache_before = _compile_kernel_cached.cache_info().currsize
      # ... repetições medidas ...
      if _compile_kernel_cached.cache_info().currsize != jit_cache_before:
          raise RuntimeError("JIT compilation landed inside the measured loop (BUG-25)")
      ```

      ```bash
      .venv/bin/python -m pytest tests/benchmarking/test_unit_runner_warmup.py -q  # -> 8 passed
      ```

- [x] **[REVIEWED]** **Task 11.12: Otimização e Equivalência Iso-Algorítmica de `MutationGPUSwap` (`DP11-14`, `DP11-19`, `DP11-20`, `SWAP-A`, `BUG-28`)**
  + **What:** Alinhar `MutationGPUSwap.mutate_batch()` ao contrato de sincronização assíncrona (`DP10-18`/`BUG-15`) removendo a chamada incondicional a `cp.cuda.Stream.null.synchronize()`, e manter a geração de pares de troca no host via `ProblemContext.rng` para estrita reprodutibilidade e equivalência bit-a-bit à versão CPU `MutationSwap`.
  + **Why:** A sincronização incondicional bloqueia a CPU desnecessariamente a cada geração mesmo quando a população reside na VRAM, gerando overhead de ~12% medido em instâncias pequenas. Manter o RNG unificado no host garante que as trajetórias estocásticas de CPU e GPU sejam idênticas.
  + **Who:** Agente (`implementer`).
  + **Where:** `src/algorithms/strategies/mutation/swap_mutation.py`, `tests/algorithms/strategies/mutation/test_unit_gpu_swap_mutation.py`.
  + **When:** Concomitante à `Task 11.10`.
  + **How:** Em `MutationGPUSwap.mutate_batch()`, remover `cp.cuda.Stream.null.synchronize()`. Garantir que os arrays de índices gerados por `ProblemContext.rng` sejam transferidos de forma contígua para a GPU e que o retorno preserve a matriz na VRAM sem bloqueio forçado. Adicionar teste de regressão validando que para uma mesma semente em `ProblemContext.rng`, a rota mutada em GPU é bit-idêntica à da CPU.
  + **How much:** Baixo — ~20 linhas de código, ~40 linhas de teste.
  + **Stakeholder Implications:** Elimina sincronizações supérfluas no laço evolucionário e mantém paridade matemática 100% rigorosa.
  + **DoD:** `Stream.null.synchronize()` removido de `MutationGPUSwap.mutate_batch()`; teste unitário de equivalência de arrays verde; `pytest tests/algorithms/strategies/mutation -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py#L145), [test_unit_gpu_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_gpu_swap_mutation.py#L1).
    - **Resumo:** Alinhado `MutationGPUSwap.mutate_batch()` ao contrato assíncrono (`DP10-18`/`BUG-15`) com a remoção de `cp.cuda.Stream.null.synchronize()`, unificando a geração de pares de troca no host via `ProblemContext.rng` para estrita reprodutibilidade e iso-algoritmo (`SWAP-A`).
    - **Correção residual (2026-11-10):** `MutationGPUSwap` passou a aceitar `candidate_k` (LSP com `MutationSwap`) e ambas as classes leem `resolve_route_bounds` e o oráculo único `sample_mutation_positions`; o laço serial `for i in range(num_routes): rng.choice(np.arange(low, high), ...)` foi substituído por `sample_distinct_pairs` vetorizado (`DP11-20`), com `probs = rng.random(num_routes)` idêntico nos dois backends.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "rng.choice(np.arange" src/algorithms/strategies/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0; `grep -rn "_mutation_bounds" src/ tests/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0; tours TSP abertos agora usam `(0, n)` (novo `test_open_tsp_route_can_mutate_both_endpoints`), enquanto `[0, ..., 0]` preserva `(1, n - 1)` (`test_gpu_swap_mutation_closed_depot_routes_match_cpu`, `test_mutation_preserves_three_node_depot_endpoints`); `test_gpu_swap_mutation_candidate_k_matches_cpu` fecha a paridade de `candidate_k`; `test_gpu_swap_mutation_matches_cpu_bit_identical` (rates × pop × n) verde; `ruff` e `mypy` limpos.
    - **Proof of Concept (PoC):**

      ```python
      from utils.route_helpers import resolve_route_bounds, sample_distinct_pairs

      low, high = resolve_route_bounds(n, context=context, route_sample=route)
      pairs = sample_distinct_pairs(resolve_rng(context, self.rng), num_pairs=len(batch), low=low, high=high)
      ```

      ```bash
      pytest tests/algorithms/strategies/mutation -q   # -> green; open-TSP endpoints mutáveis, CVRP fechados preservados
      ```

- [x] **[REVIEWED]** **Task 11.13: Otimização de Coalescência e Redução por Warp Shuffle em `cost_calculator.cu` (`DP11-15`, `EVAL-A`, `BUG-26`)**
  + **What:** Otimizar o kernel `cost_calculator.cu` substituindo o particionamento por fatias contíguas (`start = tid * chunk`) por um laço em stride de bloco (`for (i = tid; i < n; i += blockDim.x)`), garantindo acessos 100% coalescidos à memória global (VRAM), e substituir as 5 barreiras finais de shared memory (`__syncthreads()`) por uma redução intra-warp em registradores via `__shfl_down_sync`.
  + **Why:** O particionamento em `chunk` fazia com que threads adjacentes do warp acessassem endereços separados por `chunk * 4` bytes, fragmentando transações de 128 bytes da VRAM. Como a memória compartilhada armazena apenas as 256 somas parciais dos threads (`s_partials[256]` = 2 KB), o kernel **nunca** esteve limitado ao teto de $n = 11{,}425$ e não requer divisão em ladrilhos (tiling). A redução final por warp shuffle elimina 5 barreiras completas de bloco.
  + **Who:** Agente (`implementer`).
  + **Where:** [cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu#L55-L93), `tests/algorithms/strategies/evaluation/test_unit_gpu_cost_evaluation.py`.
  + **When:** Concomitante à `Task 11.10`.
  + **How:** Reescrever o laço interno em `cost_calculator.cu` com laço de stride `i += blockDim.x`. Realizar a redução paralela em shared memory apenas até o tamanho do warp (`stride = 32`); a partir daí, deixar os últimos 32 elementos a cargo do Warp 0, reduzindo via laço `for (int offset = 16; offset > 0; offset >>= 1) partial_cost += __shfl_down_sync(0xFFFFFFFFu, partial_cost, offset);`. Thread 0 grava `costs[tour_idx] = partial_cost`.
  + **How much:** Baixo/Médio — ~35 linhas de CUDA, ~40 linhas de teste.
  + **Stakeholder Implications:** Acelera a fase de avaliação em toda a suíte de problemas sem alterar um único bit do resultado do cálculo de custo.
  + **DoD:** `cost_calculator.cu` implementa stride de bloco e `__shfl_down_sync`; custo calculado permanece 100% idêntico à avaliação em CPU para qualquer $n \in \{52, 100, 318, 1002, 2103\}$; `pytest tests/algorithms/strategies/evaluation -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L213), [test_unit_gpu_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_gpu_cost_evaluation.py#L122), [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py#L102).
    - **Resumo:** Mantidos o laço de stride de bloco (`i += block_size`) e a redução intra-warp por `__shfl_down_sync` em `cost_calculator.cu` (`EVAL-A`) e a ausência de `Stream.null.synchronize()` no módulo. O upload `cp.asarray(batch_np, dtype=cp.int32)` do caminho `input_on_gpu=False` passou a somar `self.h2d_bytes += int(d_population.nbytes)`, alinhando a telemetria aos caminhos `#L155`/`#L174`. A verificação de `BUG-26` confirmou que **nenhuma** estratégia expõe `use_gpu_events`, portanto `@time_strategy` mede `perf_counter` e não sincroniza `EvaluationGPUCost.evaluate_batch`; nenhuma edição em `phase_timer.py` foi necessária.
    - **Atendimento ao DoD:** `grep -rn "use_gpu_events" src/ --include=*.py --exclude-dir=__pycache__` -> 2 ocorrências, ambas em `phase_timer.py` (docstring + `getattr`), logo 0 estratégias optam por eventos CUDA; `test_gpu_cost_evaluation_counts_h2d_for_cpu_input` (GPU) mede o delta entre duas chamadas consecutivas e exige exatamente `batch.nbytes`; `test_cpu_cost_evaluation_reports_no_transfer_for_host_batch` é o twin CPU (`h2d/d2h/kernels == 0`); `pytest tests/benchmarking tests/algorithms -q` -> `619 passed` (inclui equivalência bit-a-bit `n in {52, 100, 318, 1002, 2103}`); `mypy` 140 arquivos/0 erros; `ruff` limpo.
    - **Proof of Concept (PoC):**

      ```python
      # cost_evaluation.py::EvaluationGPUCost.evaluate_batch — upload real contabilizado
      else:
          batch_np: np.ndarray = np.asarray(batch, dtype=np.int32)
          d_population = cp.asarray(batch_np, dtype=cp.int32)
          self.h2d_bytes += int(d_population.nbytes)
      ```

- [x] **[REVIEWED]** **Task 11.14: Otimização de Coalescência de Memória em `SelectionGPUTournament` (`DP11-16`, `SEL-A`, `BUG-28`)**
  + **What:** Substituir a cópia de rotas não coalescida (stride $n$) dentro de [tournament_selection.cu](../../src/algorithms/strategies/selection/kernels/tournament_selection.cu#L37-L41) pela emissão apenas do vetor 1D `winner_indices[N]`, delegando a coleta 2D de rotas ao indexador vetorizado do CuPy (`parents_out = population[winner_indices]`), mantendo a amostragem de candidatos no host via `ProblemContext.rng` e removendo a chamada incondicional a `cp.cuda.Stream.null.synchronize()` em [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L251).
  + **Why:** Em `tournament_selection.cu`, 1 thread por vencedor copia uma rota inteira com stride $n$, emitindo 32 transações DRAM de 128 bytes separadas por warp e desperdiçando até 96,9 % da largura de banda física da VRAM, além de bloquear a CPU com `Stream.null.synchronize()`. O gather 2D do CuPy (NVIDIA CUB `DeviceGather`) lê horizontalmente ao longo das linhas com cargas vetorizadas `int4` (100 % coalescido).
  + **Who:** Agente (`implementer`).
  + **Where:** [tournament_selection.cu](../../src/algorithms/strategies/selection/kernels/tournament_selection.cu#L13-L42), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L175-L255), `tests/algorithms/strategies/selection/test_unit_gpu_tournament_selection.py`.
  + **When:** Concomitante às `Tasks 11.10`–`11.13`.
  + **How:** Alterar `tournament_selection_kernel` para gravar apenas `winner_indices[parent_idx] = best_idx`; em `SelectionGPUTournament.select()`, alocar `winner_indices = cp.empty(num_parents, dtype=cp.int32)`, executar o kernel 1D, obter `parents_gpu = population_gpu[winner_indices]` sem `Stream.null.synchronize()`, e adicionar teste unitário verificando equivalência bit-a-bit com `SelectionTournament` para a mesma semente em `ProblemContext.rng`.
  + **How much:** Baixo — ~25 linhas entre `.cu` e `.py`, ~40 linhas de teste.
  + **Stakeholder Implications:** Elimina o gargalo de banda de memória na seleção em GPU e mantém paridade bit-a-bit com a seleção em CPU.
  + **DoD:** `tournament_selection.cu` emite `winner_indices` sem laço `for (int c = 0; c < n; c++)`; `Stream.null.synchronize()` removido de `tournament_selection.py`; saída de `SelectionGPUTournament` é bit-idêntica à de `SelectionTournament` para a mesma semente; `pytest tests/algorithms/strategies/selection -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [tournament_selection.cu](../../src/algorithms/strategies/selection/kernels/tournament_selection.cu#L15), [tournament_selection.py](../../src/algorithms/strategies/selection/tournament_selection.py#L180), [test_unit_gpu_tournament_selection.py](../../tests/algorithms/strategies/selection/test_unit_gpu_tournament_selection.py#L143).
    - **Resumo:** Substituída a cópia não-coalescida de rotas com stride $n$ em `tournament_selection.cu` pela emissão exclusiva do vetor 1D `winner_indices[N]`, delegando a coleta 2D ao indexador vetorizado do CuPy (`d_pop[d_winner_indices]`, NVIDIA CUB `DeviceGather`, 100% coalescido, `SEL-A`), amostragem mantida no host via `ProblemContext.rng` e remoção de `Stream.null.synchronize()`. Adicionado teste falsificável validando assinatura 1D do kernel.
    - **Correção residual (2026-11-10):** `SelectionTournament.select_batch` passou a amostrar com `rng.integers(..., dtype=np.int32)` e o ramo `is_gpu=True` usa o mesmo `resolve_rng(context, self.rng)` sobre o host, transferindo `candidates`/`fit_arr` com contabilização em `h2d_bytes` (`BUG-28`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "dtype=np.int64" src/algorithms/strategies/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0; `grep -rn "cp.random.randint" src/algorithms/strategies/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0; `test_gpu_tournament_selection_matches_cpu_bit_identical` (k × pop × n) e `test_gpu_tournament_selection_gpu_input_remains_on_gpu` verdes; `pytest tests/algorithms/strategies/selection -q` → `49 passed`; `mypy` e `ruff` limpos.
    - **Proof of Concept (PoC):**

      ```python
      rng = resolve_rng(context, self.rng)          # mesmo gerador semeado na CPU e na GPU
      candidates = rng.integers(0, pop_size, size=(num_parents, self.k), dtype=np.int32)
      ```

      ```bash
      pytest tests/algorithms/strategies/selection -q   # -> green com teste de dtype e de identidade CPU/GPU
      ```

- [x] **[REVIEWED]** **Task 11.15: Eliminação de Divergência de Warp e Reversão Cooperativa em `MutationGPUInversion` (`DP11-17`, `DP11-19`, `DP11-20`, `INV-A`, `BUG-28`)**
  + **What:** Reestruturar [inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu#L20-L38) e [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L142-L232) para atribuir **1 warp (32 threads) por rota** (`route_idx = warp_id`, `lane_id = tid % 32`), eliminando a divergência intra-warp sob baixa taxa de mutação ($p_m = 0.10$), revertendo o segmento `[left..right]` em paralelo pelas 32 lanes, mantendo `ProblemContext.rng` no host e removendo `cp.cuda.Stream.null.synchronize()`.
  + **Why:** Com 1 thread por rota e $p_m = 0.10$, ~29 das 32 threads de cada warp ficam ociosas em lockstep SIMT aguardando ~3 threads executarem até $n/2$ iterações sequenciais de `while (left < right)`. Com 1 warp por rota, todas as 32 lanes testam o mesmo escalar `probs[route_idx] < mutation_rate` (zero divergência intra-warp: rotas não mutadas saem em 1 ciclo; rotas mutadas dividem os `num_swaps` pares entre as 32 lanes em paralelo coalescido).
  + **Who:** Agente (`implementer`).
  + **Where:** [inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu#L13-L39), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L142-L232), `tests/algorithms/strategies/mutation/test_unit_gpu_inversion_mutation.py`.
  + **When:** Concomitante às `Tasks 11.10`–`11.14`.
  + **How:** Lançar $N \times 32$ threads (`block = 256` = 8 warps/rotas por bloco); calcular `route_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5` e `lane_id = threadIdx.x & 31`; quando `probs[route_idx] < mutation_rate`, executar `for (int k = lane_id; k < num_swaps; k += 32)` trocando `route[left + k]` com `route[right - k]`; remover `Stream.null.synchronize()` em `inversion_mutation.py` e validar equivalência bit-a-bit com `MutationInversion`.
  + **How much:** Baixo/Médio — ~35 linhas entre `.cu` e `.py`, ~45 linhas de teste.
  + **Stakeholder Implications:** Acelera a mutação por inversão em até 32× nas rotas ativas, elimina o bloqueio SIMT nas rotas inativas e preserva 100 % de equivalência bit-a-bit com a CPU.
  + **DoD:** `inversion_mutation.cu` opera com 1 warp por rota (`k += 32`); `Stream.null.synchronize()` removido de `inversion_mutation.py`; saída de `MutationGPUInversion` é bit-idêntica à de `MutationInversion` para a mesma semente; `pytest tests/algorithms/strategies/mutation -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu#L15), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L140), [test_unit_gpu_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_gpu_inversion_mutation.py#L1).
    - **Resumo:** Reestruturado `inversion_mutation.cu` atribuindo 1 warp (32 threads) por rota (`route_idx = global_tid >> 5`, `lane_id = tid & 31`), eliminando 100% da divergência intra-warp sob baixa taxa de mutação ($p_m = 0.10$) e revertendo o segmento `[left..right]` cooperativamente em paralelo (`k += 32`, `INV-A`). Removido `Stream.null.synchronize()` em `inversion_mutation.py`.
    - **Correção residual (2026-11-10):** `MutationGPUInversion` passou a aceitar `candidate_k` e ambas as classes usam `resolve_route_bounds` + `sample_mutation_positions`, eliminando o laço serial de `rng.choice(np.arange(low, high), ...)` (`DP11-20`, `BUG-28`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "rng.choice(np.arange" src/algorithms/strategies/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → 0; `test_gpu_inversion_mutation_matches_cpu_bit_identical` (rates × pop × n) e o novo `test_gpu_inversion_mutation_closed_depot_routes_match_cpu` verdes; `test_candidate_k_inversion_keeps_permutation_and_depot` cobre a rota fechada de 52 paradas e `test_candidate_k_inversion_supports_sparse_cvrp_sub_route` segue verde; `mypy` e `ruff` limpos.
    - **Proof of Concept (PoC):**

      ```python
      low, high = resolve_route_bounds(n, context=context, route_sample=route)
      spans = sample_distinct_pairs(resolve_rng(context, self.rng), num_pairs=len(batch), low=low, high=high)
      ```

      ```bash
      pytest tests/algorithms/strategies/mutation -q   # -> green com tour aberto (n=52) mutando posições 0 e n-1
      ```

- [x] **Task 11.16: Reversão Cooperativa por Bloco na Aplicação de Movimento de `LocalSearchGPUTwoOpt` (`DP11-18`, `2OPT-A`, `BUG-29`)**
  + **What:** Substituir a aplicação sequencial por 1 thread (`if (tid == 0 && s_deltas[0] < 0) while (left < right) ...`) em [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L250-L275) e [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L100-L139) por um laço cooperativo em paralelo através de todas as `blockDim.x` (256) threads do bloco (`for (int k = tid; k < num_swaps; k += blockDim.x)`).
  + **Why:** Enquanto a avaliação de movimentos 2-opt já é paralelizada entre as 256 threads do bloco, a reversão do único segmento vencedor `[swap_i + 1 .. swap_j]` era executada serialmente pela Thread 0 (até $n/2$ trocas sequenciais) enquanto 255 threads aguardavam ociosas em `__syncthreads()`. Como os pares simétricos `(left + k, right - k)` para $k \in [0, \lfloor L/2 \rfloor - 1]$ são mutuamente disjuntos, todas as 256 threads podem revertê-los simultaneamente sem condição de corrida, mantendo 100 % a semântica de 2-opt best-improvement.
  + **Who:** Agente (`implementer`).
  + **Where:** [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L250-L275), [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L100-L139), `tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py`.
  + **When:** Concomitante às `Tasks 11.10`–`11.15`.
  + **How:** Substituir o bloco `if (tid == 0 && s_deltas[0] < 0)` por `if (s_deltas[0] < 0)` lendo `left = s_swap_i[0] + 1`, `right = s_swap_j[0]`, `num_swaps = (right - left + 1) / 2` e executando `for (int k = tid; k < num_swaps; k += blockDim.x)` com `TOUR_AT`/`TOUR_SET` (ou `tour[...]`), mantendo o `__syncthreads()` subsequente antes da próxima iteração de 2-opt.
  + **How much:** Baixo — ~30 linhas de CUDA, ~35 linhas de teste.
  + **Stakeholder Implications:** Remove o único trecho serial remanescente por passagem dentro do kernel de 2-opt best-improvement sem alterar um único bit da rota resultante.
  + **DoD:** `if (tid == 0` não é mais usado para o laço de reversão `while (left < right)` em `two_opt_batch.cu` e `two_opt_single.cu`; rotas produzidas por `LocalSearchGPUTwoOpt(mode="batch")` e `mode="naive"` permanecem bit-idênticas ao referência em todas as instâncias de teste; `pytest tests/algorithms/strategies/local_search -q` verde; `mypy src tests` e `ruff check src tests` limpos.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L77), [two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu#L252), [two_opt_single.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_single.cu#L104), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L238), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L8), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L380), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py#L552).
    - **Resumo:** `two_opt_batch.cu` ganhou `if (tour_idx >= num_tours) return;` antes de tocar `tours`/shared memory; a redução dos dois kernels de passe único passou a dobrar em paralelo só enquanto o passo excede um warp e o fio 0 varre apenas `1 .. min(block_size, 64) - 1` (antes re-varria todos os `block_size` parciais já mesclados), com curto-circuito quando `s_deltas[0] >= 0`. Em `two_opt_multi_move.cu` a seleção gulosa continua serial (é inerentemente sequencial), mas publica o segmento vencedor em shared memory (`s_move_start`/`s_move_end`) e o bloco inteiro aplica a reversão com `for (int k = tid; k < num_swaps; k += block_size)`. O cabeçalho do kernel e o docstring de `LocalSearchGPUTwoOpt` documentam o teto de 1 candidato steepest por thread (`min(block_size, n / 2)` movimentos por passe, Rocki & Suda 2013) antes da seleção disjunta.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `grep -rn "while (left < right)" src/algorithms/strategies/local_search/kernels/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → **0**; `grep -rn "for (int i = 1; i < block_size; i++)"` nos mesmos kernels → **0**; `grep -rn "if (tour_idx >= num_tours) return;"` → **1**. `pytest tests/algorithms/strategies/local_search -q` → `68 passed` (identidade bit-a-bit com CPU preservada nos testes de paridade existentes); `ruff check src tests` → `All checks passed!`; `mypy src tests benchmark_scripts` → 137 arquivos sem erros.

- [x] **Task 11.17: Desacoplar `@time_strategy` da Sincronização Forçada de Eventos CUDA (`DP11-22`, `BUG-26`)**
  + **What:** Remover de [phase_timer.py](../../src/utils/phase_timer.py#L86-L119) o registro de eventos CUDA e a chamada bloqueante `end_evt.synchronize()` do caminho quente de `@time_strategy`, de modo que estratégias GPU retornem ao host de forma assíncrona (`enqueue`) por padrão; qualquer medição de tempo de dispositivo passa a ser explícita via `PhaseTimer(use_cuda_events=True)` ou drenagem nas fronteiras D2H.
  + **Why:** O decorador impõe 5–6 barreiras síncronas host-device por geração (uma por estratégia que recebe array GPU), anulando as remoções de `Stream.null.synchronize()` das `Tasks 11.12`–`11.16` (`DP10-18`, `SWAP-A`, `EVAL-A`, `SEL-A`, `INV-A`) e mascarando o teste de três relógios que fundamentou a `DP11-7`: o paralelismo CPU/GPU do pipeline assíncrono simplesmente não existe enquanto o decorador bloqueia.
  + **Who:** Agente (`implementer`).
  + **Where:** [phase_timer.py](../../src/utils/phase_timer.py#L86-L119), [algorithm_strategies.py](../../src/protocols/algorithm_strategies.py#L84-L108) (somente leitura dos consumidores), `tests/utils/test_unit_phase_timer.py`, `tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py`.
  + **When:** Primeiro item após a auditoria de 2026-11-10; bloqueia a validade das `Tasks 11.13`–`11.16` (o ganho de todas elas é parcialmente cancelado pelo decorador).
  + **How:** Detectar `is_gpu` como hoje, mas trocar o bloco `Event`/`synchronize()` por um caminho que apenas chame `func(*args, **kwargs)` e registre `elapsed_ms=None` quando a sincronização não for solicitada; aceitar `sync_gpu: bool = False` no decorador e um flag de contexto (`PhaseTimer.use_cuda_events`) que, quando verdadeiro, mantém o comportamento atual; atualizar o teste de sincronização para afirmar **zero** `synchronize()` implícitos e cobrir o caminho assíncrono.
  + **How much:** Baixo — ~30 linhas em `phase_timer.py`, ~45 linhas de teste.
  + **Stakeholder Implications:** Reabilita o pipeline assíncrono CPU↔GPU e garante que os speedups reportados no Capítulo 4 meçam execução de dispositivo, não barreiras de instrumentação.
  + **DoD:** `grep -n "end_evt.synchronize" src/utils/phase_timer.py` não encontra ocorrência no caminho default de `@time_strategy`; `pytest tests/utils tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py -q` verde em CPU; o repro do PoC imprime `unwrapped_enqueue` ordens de magnitude menor que `wrapped`; `mypy src tests` e `ruff check src tests` limpos.
  + **Proof of Concept (PoC):**

    ```python
    # phase_timer.py::time_strategy — sem eventos CUDA no caminho default (PEP 695: **P, R na declaração real)
    def time_strategy(func: Callable[P, R], *, sync_gpu: bool = False) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not (sync_gpu and is_gpu_array(args)):
                return func(*args, **kwargs)          # enqueue assíncrono
            start_evt, end_evt = cp.cuda.Event(), cp.cuda.Event()
            start_evt.record(); out = func(*args, **kwargs); end_evt.record(); end_evt.synchronize()
            return out
        return wrapper
    ```

    ```bash
    PYTHONPATH=src python -c "import time, cupy as cp; from utils.phase_timer import time_strategy; g=lambda n: cp.zeros((256, n), cp.int32); f=time_strategy(g); t=time.perf_counter(); f(1002); print(f'wrapped={1000*(time.perf_counter()-t):.3f}ms')"
    # transcrição: wrapped=260.840ms | unwrapped_enqueue=0.363ms  (sincronização forçada pelo decorador)
    ```

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [phase_timer.py](../../src/utils/phase_timer.py#L1), [phase_timer.py](../../src/utils/phase_timer.py#L86), [base.py](../../src/algorithms/strategies/base.py#L40), [test_unit_phase_timer.py](../../tests/utils/test_unit_phase_timer.py#L42), [test_unit_two_opt_synchronization.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py#L176).
    - **Resumo:** `@time_strategy` deixou de inspecionar os argumentos com `isinstance(arg, cp.ndarray)` e só resolve o backend de eventos quando a estratégia expõe `use_gpu_events = True`; o caminho padrão mede `time.perf_counter()` e devolve o controle ao host após o *enqueue*. `PhaseTimer(use_cuda_events=True)` mantém a medição de dispositivo para quem a pedir. `StrategyBase` declara `last_elapsed_ms`/`total_elapsed_ms`/`call_count` como atributos de instância tipados, dando casa estática ao que o decorador escreve.
    - **Atendimento ao DoD (Revisado 5+1 eixos):** probe de enqueue com `LocalSearchGPUTwoOpt(mode="batch")`, `N=256`, `n=500`, população residente → `enqueue=0.24ms wrapped_full=154.20ms last_elapsed=0.08ms`, ou seja `last_elapsed_ms` (0.08 ms) acompanha o enqueue e não a sincronização de dispositivo (154.20 ms, antes cobrada em toda chamada). `grep -n "end_evt.synchronize" src/utils/phase_timer.py` → linha 59 (`PhaseTimer.measure`, opt-in por `use_cuda_events`) e linha 118 (ramo `use_gpu_events`), nenhuma no caminho padrão. `pytest tests/utils/test_unit_phase_timer.py tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py -q` → verde com `test_time_strategy_never_probes_the_device_by_default`, `test_time_strategy_uses_device_events_only_when_opted_in` e `test_time_strategy_enqueues_without_device_events` (batch/multi_move/naive, `cuda_events == 0`, `event_synchronize == 0`).

- [x] **Task 11.18: Sincronizar Contrato de `BenchmarkRunInput` e Aplicar Patch Canônico de Distâncias Geográficas `GEO` (`BUG-31`, `BUG-32`)**
  + **What:** (i) Sincronizar as constantes `OPTIONAL_FIELDS` e `FULL_PAYLOAD` em [test_unit_records.py](../../tests/data_models/test_unit_records.py#L54) com os três novos campos de telemetria fina de `BenchmarkRunInput` (`mean_phase_timings`, `phase_share_pct`, `phase_share_cv_pct`, `BUG-31`); (ii) aplicar patch canônico em [pairwise.py](../../src/distances/pairwise.py#L66) restaurando `_to_radians_geo` com truncamento de minutos DDD.MM conforme especificação TSPLIB95 (Reinelt 1995), substituindo o import de `converter.utils.cost.geographical` do `routing_data` v0.2.0 que arredonda graus indevidamente (`BUG-32`).
  + **Why:** A ausência dos 3 campos no teste causava falha em `test_benchmark_run_input_optional_keys_match_the_nullable_columns` e `test_full_payload_covers_every_declared_field`; o arredondamento de graus em `routing_data` alterava distâncias canônicas (ex.: `burma14` d(1,3) de 510 para 560), quebrando `test_fractional_part_of_a_geo_coordinate_is_minutes`, `test_distance_matrix_agrees_with_pairwise_function[GEO]` e `test_device_dispatcher_matches_cpu[GEO]`.
  + **Who:** Agente (`implementer`).
  + **Where:** [test_unit_records.py](../../tests/data_models/test_unit_records.py#L54), [pairwise.py](../../src/distances/pairwise.py#L66), [pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh#L105), [test_unit_pairwise_geographical.py](../../tests/distances/test_unit_pairwise_geographical.py#L23).
  + **When:** Imediatamente após a auditoria multi-eixos da Fase 11 (2026-11-10).
  + **How:** Em `test_unit_records.py`, acrescentar os 3 campos a `OPTIONAL_FIELDS` (16 total) e `FULL_PAYLOAD` (43 total); em `pairwise.py`, implementar `_to_radians_geo` e `compute_geographical` locais com aviso explícito de patch sobre `routing_data` e sincronizar comentário em `pairwise_device.cuh`; atualizar teste unitário em `test_unit_pairwise_geographical.py`.
  + **How much:** Baixo — ~40 linhas de código e testes.
  + **Stakeholder Implications:** Restaura integridade de 100 % da suíte de testes (1.067 testes passando) e garante conformidade exata com a especificação TSPLIB95 de distâncias geográficas.
  + **DoD:** `pytest tests/data_models/test_unit_records.py tests/distances -q` verde com 100 % de aprovação; `pairwise.compute_geographical` concorda com `pairwise_device.cuh::gpu_geo` em todas as instâncias; `check_docs` com 0 undefined symbols.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_records.py](../../tests/data_models/test_unit_records.py#L54), [pairwise.py](../../src/distances/pairwise.py#L66), [pairwise_device.cuh](../../src/distances/kernels/pairwise_device.cuh#L105), [test_unit_pairwise_geographical.py](../../tests/distances/test_unit_pairwise_geographical.py#L23).
    - **Resumo:** `OPTIONAL_FIELDS` e `FULL_PAYLOAD` em `test_unit_records.py` foram sincronizados com as novas colunas de telemetria fina (`mean_phase_timings`, `phase_share_pct`, `phase_share_cv_pct`) introduzidas na `Task 11.7` (`BUG-31`). Implementado patch canônico de `_to_radians_geo` e `compute_geographical` em `pairwise.py` sobrepondo o `routing_data` v0.2.0 que arredondava graus (`BUG-32`), alinhando o host ao kernel CUDA `pairwise_device.cuh::gpu_geo`.
    - **Atendimento ao DoD:** `pytest tests/data_models/test_unit_records.py tests/distances -q` → `212 passed` (15 em records, 197 em distances); `test_fractional_part_of_a_geo_coordinate_is_minutes`, `test_distance_matrix_agrees_with_pairwise_function[GEO]` e `test_device_dispatcher_matches_cpu[GEO]` verdes; suite completa com `1067 passed`; `check_docs` exit 0.

## Fase 12: Trabalhos Futuros, Descapamento de CPU, Coleta de Resultados e Atualização da Dissertação (`second_draft.tex`)

### Decisões Fixadas da Fase 12

- **DP12-1 (Registro de Bitpacking em Shared Memory para Crossover com $n > 10.000$ como Trabalho Futuro na Dissertação)**: Registrar no texto da dissertação ([second_draft.tex](../../documents_tcc/tex/second_draft.tex) / [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex)) que a tabela de pertinência em byte array (`uint8_t in_segment[n]`, Formato A) atende com folga a todas as instâncias da TSPLIB95 ($n \le 2.103$ consome no máximo 16,8 KB de shared memory por bloco com 8 rotas, mantendo >65% livre dos 48 KB da GTX 1050), e documentar o Formato B (bitpacked array `uint32_t bitmap[n/32]` com aritmética `c >> 5`, `c & 31` e `atomicOr`) como caminho analítico comprovado de evolução futura para problemas ultra-massivos ($n > 10.000$) que excedam o limite físico de memória compartilhada (`TASK 12.1`).
- **DP12-2 (Documentação de Extensões em Operadores Metaheurísticos e Micro-otimizações de Kernels CUDA)**: Documentar em [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex) as extensões futuras em operadores metaheurísticos (cruzamento EAX com decomposição de ciclos AB, 3-opt e Or-opt para suporte a grafos direcionados ATSP preservando orientação de arestas, e modelo em ilhas com migração assíncrona entre CUDA streams) e micro-otimizações de baixo nível em kernels CUDA (reduções por warp shuffle intrinsics `__shfl_down_sync` sem shared memory, eliminação de bank conflicts com padding de shared memory, persistent threads / grid-stride loops, e vinculação de coordenadas ao texture/read-only cache via `__ldg`) (`TASK 12.2`).
- **DP12-3 (Avaliação de Viabilidade e Descapamento do Benchmark de CPU com Listas de Candidatos)**: Definir o protocolo de avaliação para descapamento da variante `GeneticAlgorithmCPU` além do corte preliminar $N \le 100$ (estendendo para instâncias médias $100 < n \le 400$ ou todo o estrato da TSPLIB95), explorando o ganho de $2.6\times$ a $8.2\times$ obtido com listas de candidatos ($k=20$) na busca local 2-opt em CPU, sob teto de timeout explícito (`TASK 12.3`).
- **DP12-4 (Protocolo de Coleta Integral do Benchmark da Dissertação e Atualização de Tabelas e DuckDB)**: Executar a bateria experimental completa nas 38 instâncias da TSPLIB95 (ou estrato selecionado para defesa) para as variantes ativas (CPU, HybridNaive, HybridOptimized, FullGPUModular), com 10 repetições independentes, persistência no banco `datasets/tsp.duckdb`, geração de checkpoints e materialização das tabelas finais de validação e telemetria fina de fases com $ddof=1$ (`TASK 12.4`).
- **DP12-5 (Síntese da Evolução do Projeto, Lições de Engenharia e Atualização Textual da Dissertação)**: Atualizar os capítulos da dissertação ([second_draft.tex](../../documents_tcc/tex/second_draft.tex) / [parts/04-resultados.tex](../../documents_tcc/tex/parts/04-resultados.tex), [parts/05-discussao.tex](../../documents_tcc/tex/parts/05-discussao.tex), [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex)) substituindo avisos de dados provisórios pelos dados reais do benchmark da Fase 11/12, e sintetizar na discussão as lições práticas de engenharia de software e arquitetura GPU derivadas de [notes_project_direction_questions.md](../../notes_project_direction_questions.md) (hermetismo de residência, eliminação de JIT frio via AOT, invariantes de fronteira sem aliases legados) (`TASK 12.5`).
- **DP12-6 (Geração Automatizada de Diagramas de Diferença Crítica de Demšar em LaTeX/TikZ)**: Integrar ao pipeline de análise estatística a geração automática de diagramas CD de Demšar (1986, 2006) em formato vetorial TikZ para inclusão no capítulo de resultados e discussão de [second_draft.tex](../../documents_tcc/tex/second_draft.tex), aplicando o teste omnibus de Friedman, pós-hoc de Nemenyi com fator de Studentized range corrigido ($q_\alpha = Q_\alpha / \sqrt{2}$ evitando inflação de 41,4%), detecção de cliques de equivalência e exportação de arquivo `.tikz` standalone compilável (`TASK 12.6`).
- **DP12-7 (Arquitetura de Persistência Atômica de Checkpoints por Repetição com Isolamento de Tempo de I/O)**: Implementar persistência atômica granular ao fim de cada repetição estocástica, isolando o tempo de gravação em disco da medição do solver (`TASK 12.14`).
- **DP12-8 (Infraestrutura Padronizada de Logging com Destino Duplo)**: Introduzir captura simultânea em console e arquivo persistente `output_dir / f"{benchmark_name}.log"` para retenção de logs em execuções de longa duração (`TASK 12.15`).
- **DP12-9 (Relatórios Estatísticos e Matrizes de Speedup Agnósticos a Nomes de Algoritmos)**: Desacoplar analisadores estatísticos de nomes fixos de variantes, parametrizando a linha de base e usando o primeiro algoritmo disponível como fallback formal (`TASK 12.16`).
- **DP12-10 (Atualização e Validação Formal dos Esquemas JSON de Configuração)**: Sincronizar e validar esquemas JSON em `configs/schemas/` com cobertura exaustiva de parâmetros (`candidate_k`, thresholds e limites) (`TASK 12.17`).
- **DP12-11 (Parametrização Explícita Completa e Auditoria de Listas de Candidatos)**: Eliminar valores padrão implícitos em scripts executores e configurações declarativas, registrando explicitamente `candidate_k=20`, `max_iterations=10` e hiperparâmetros na telemetria inicial (`TASK 12.18`).
- **DP12-12 (Otimizações Algorítmicas Puras em 2-Opt CPU: Eliminação de Gerador Heap, Limites sem Ramificação e Aritmética Inteira Exata)**: Eliminar o gerador python `_iter_moves` em [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py) que instanciava milhões de 3-tuplas `(i, j, delta)` por geração na memória heap, içar o limite do laço sem ramificação interna (`limit_j = n - 1 if i == 0 else n`), e substituir conversões para ponto-flutuante por comparações exatas de inteiros `int32` (`delta < 0`) idênticas ao kernel CUDA [two_opt_common.cuh](../../src/distances/kernels/two_opt_common.cuh#L61), tanto no percurso exaustivo quanto na busca acelerada por listas de candidatos `candidates` (`DP10-20`) (`TASK 12.19`).
- **DP12-13 (Vetorização com Indexação Achatada sem Cópias e Alinhamento de Dtype na Avaliação de Custo CPU)**: Eliminar a chamada a `np.roll` e matrizes intermediárias em [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py), utilizando fatiamento direto em views (`batch[:, :-1]` e `batch[:, 1:]`) e indexação 1D em buffer de distâncias pré-achatado (`flat_dist[batch[:, :-1] * n + batch[:, 1:]]`), reduzindo o tempo de cálculo em 35% e acumulando o custo na precisão nativa da matriz de distâncias (`int32`/`int64`), e retificar o alias `DistanceMatrix` em [types.py](../../src/protocols/types.py#L32) para `NDArray[np.int32]` (`TASK 12.20`).
- **DP12-14 (Vetorização sem Coleções python em Cruzamento OX/PMX e Mutação por Inversão sem Cópia Redundante)**: Substituir `set` em [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py) por array de pertinência booleano `in_seg[p1[c1:c2+1]] = True` e rotação circular vetorizada preservando os nós de depósito (`stream_start = 1 if is_closed else 0`); substituir dicionários dinâmicos em [pmx_crossover.py](../../src/algorithms/strategies/crossover/pmx_crossover.py) por array direto de posições inversas `inv_p2[val] = idx`; e eliminar clonagem redundante em [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py) passando `copy=False` e revertendo o buffer in-place (`TASK 12.21`).
- **DP12-15 (Harmonização de Dtypes em Avaliação de Custo e Paridade de Tipos Fitness)**: Forçar `dtype=np.float64` na soma de `EvaluationCPUCost.evaluate_batch` para que o tipo de retorno coincida estritamente com `Fitness = NDArray[np.float64]` ([types.py](../../src/protocols/types.py#L29)) e mantenha invariância exata com o backend GPU (`EvaluationGPUCost`), e alinhar `MutationSwap` com a semântica `copy=False` de `to_cpu_batch` (`TASK 12.22`, `BUG-40`).
- **DP12-16 (Eliminação de Boilerplate em Runner de Telemetria e Desacoplamento de Relatórios)**: Refatorar `_build_summary_payload` em [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py) com o extrator puro `_compute_series_stats(arr, n_valid, ddof)` eliminando 18 guardas ternárias repetidas enquanto preserva `C1`, $ddof=1$ e o contrato do DuckDB, e unificar parâmetros comuns entre [orchestration.py](../../src/benchmarking/runner/orchestration.py) e [report_generator.py](../../src/benchmarking/analysis/report_generator.py) (`TASK 12.23`).
- **DP12-17 (Suíte de Testes Comportamentais e Resiliência a Imprevistos em Benchmarking)**: Implementação de `tests/benchmarking/test_unit_report_generator_behavioral.py` validando resiliência a variantes de nomes arbitrários, instâncias com linha de base omitida/desativada (verificação da adaga local `†`), empates com variância zero no teste omnibus de Friedman e matrizes heterogêneas (`TASK 12.24`).
- **DP12-18 (Invariância à Escala e Rejeição Estrita de Não-Finitos no Teste de Friedman)**: Substituir limiares absolutos de variância (como `1e-10`) e supressão global de `RuntimeWarning` em `friedman_test` ([statistics.py](../../src/benchmarking/analysis/statistics.py#L358-L398)) por validação explícita de finitude (`np.isfinite`, levantando `ValueError` em `NaN`/`Inf`) e detecção exata/invariante à escala de empates completos intra-bloco (`np.all(np.ptp(data_matrix, axis=1) == 0.0)`), preservando testes ordinais em qualquer magnitude de dados (`TASK 12.25`, `BUG-41`).
- **DP12-19 (Resolução Preguiçosa e Estrita de Métricas Agregadas em Análise Estatística)**: Proibir avaliação ansiosa de `np.nanmean` em argumentos padrão de `dict.get(...)` e fabricação silenciosa de zeros (`0.0`) ou speedups (`1.0`) em `perform_statistical_analysis` ([problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L79-L132)); métricas faltantes devem ser computadas sob demanda via `_resolve_metric_mean`/`_resolve_gap_mean` a partir de séries finitas ou levantar `KeyError` explícito quando ausentes (`TASK 12.26`, `BUG-42`).
- **DP12-20 (Integridade da Cobertura de Telemetria Orquestrada e Guardas de Fatias Vazias)**: Manter o contrato compartilhado de `compute_series_stats` entre `algorithm_runner.py` e `orchestration.py` protegido contra regressões acidentais, garantindo cobertura explícita de `_save_problem_statistics_file` para séries ausentes/all-NaN sem emissão de `NaN` (`TASK 12.27`, `BUG-43`).
- **DP12-21 (Fidelidade de Chaves de Checkpoint e Simetria Markdown/LaTeX em Testes Comportamentais)**: Payloads sintéticos de teste para `perform_statistical_analysis` devem usar chaves canônicas de checkpoint (`"raw_costs"`, `"raw_times"` — nunca aliases ad-hoc ignorados por `_get_data_field`), e testes de relatório devem verificar separação estrita por linha/célula (presença vs ausência de `†`/`^{\dagger}`, `std == 0.00` em $N=1$, e `'—'` quando $|P_\cap|=0$) em ambos os formatos `.md` e `.tex` (`TASK 12.28`, `BUG-44`).
- **DP12-22 (Escapamento LaTeX Consistente de Identificadores em Tabelas Geradas)**: Aplicar escapamento seguro de caracteres reservados LaTeX (`_`, `%`) via helper compartilhado `escape_latex_identifier` ([report_generator.py](../../src/benchmarking/analysis/report_generator.py#L60-L66), reutilizado em [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L225)) em nomes de problemas e algoritmos emitidos pelas funções `_generate_table*_latex` de `report_generator.py` para garantir saída `.tex` válida para LuaLaTeX (`TASK 12.29`, `BUG-45`).
- **DP12-23 (Parada Estrita Step-Down de Holm-Bonferroni)**: Garantir interrupção monotônica na primeira não-rejeição em `holm_bonferroni_correction` ([problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L158-L174)) via `np.minimum.accumulate(sorted_pvals <= critical_values)` e coerência exata com `pvals_corrected <= alpha`, avaliando consolidação com [statistics.py](../../src/benchmarking/analysis/statistics.py#L330-L355) (`TASK 12.30`, `BUG-46`).
- **DP12-24 (Calibração Canônica $q_\alpha = q_{\text{Tukey}} / \sqrt{2}$ no Nemenyi Post-Hoc)**: Alinhar `StatisticalAnalyzer.nemenyi_posthoc` ([statistics.py](../../src/benchmarking/analysis/statistics.py#L447-L448)) à Eq. (8) de Demšar (2006) e a [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L123-L126), dividindo `q_critical` por $\sqrt{2}$ antes do cálculo de `critical_distance` (`TASK 12.31`, `BUG-47`).
- **DP12-25 (Proibição de Fallback de Custo Final para Inicial na Retomada Incremental)**: Proibir o fallback `"raw_initial_costs" -> "raw_costs"` em `_initialize_results_dict` ([algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L98-L101)) para que execuções retomadas sem custo inicial não fabriquem melhoria `0.0%` (`TASK 12.32`, `BUG-48`).
- **DP12-26 (Idempotência DuckDB por Chave Natural e Cobertura Integral na Tabela 3)**: Deduplicar/atualizar registros em [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L110-L125) por chave lógica `(problem_name, algorithm)` em vez de incluir `timestamp`, e exigir `len(gaps) == len(problems)` em `_compute_table3_size_category_winners` ([report_generator.py](../../src/benchmarking/analysis/report_generator.py#L432-L446)) (`TASK 12.33`, `TASK 12.34`, `BUG-49`, `BUG-50`).
- **DP12-27 (Execução Única em `@time_strategy`, Semântica Clara de `next_power_of_2` e Guarda de Pool Vazio)**: Isolar o registro de eventos CUDA da invocação única de `func` em `@time_strategy` ([phase_timer.py](../../src/utils/phase_timer.py#L112-L124)), auditar chamadores CUDA (`two_opt.py#L561`, `cost_evaluation.py#L89`) para corrigir/renomear `next_power_of_2` ([gpu_helpers.py](../../src/utils/gpu_helpers.py#L67-L70)) sem alias legado, e guardar `pool.size == 0` em `select_proximity_index` ([route_helpers.py](../../src/utils/route_helpers.py#L167-L168)) (`TASK 12.35`, `TASK 12.36`, `TASK 12.37`, `BUG-51`, `BUG-52`, `BUG-54`).
- **DP12-28 (Contrato Explícito 3D/NULL em Loaders e Saneamento de Protocolos/Baseline)**: Tratar ou rejeitar explicitamente `EDGE_WEIGHT_TYPE` 3D sem descarte silencioso de `z` em [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L168-L180) e [database_loader.py](../../src/loaders/database_loader.py#L231), validar `matrix_raw is None` em [#L272-L273](../../src/loaders/database_loader.py#L272-L273), tornar `baseline_algorithm` configurável em `problem_statistics.py#L412` e limpar dívida técnica de tipagem CuPy/protocolos mortos em `src/protocols/` (`TASK 12.38`, `TASK 12.39`, `BUG-53`, `BUG-55`).
- **DP12-29 (Ordem Canônica de Algoritmos na Resolução de Baseline)**: Fixar uma única sequência de algoritmos para `resolve_baseline_algorithm` nas duas rotas de relatório — estatísticas por problema ([problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L217), hoje `list(results.keys())`) e tabelas ([report_generator.py](../../src/benchmarking/analysis/report_generator.py#L711-L714), hoje `sorted(algo_set)`) — e encaminhar `baseline_algorithm` em [benchmark.py](../../src/cli/benchmark.py#L198-L218); a ordem escolhida define a baseline sempre que o parâmetro for `None` (`TASK 12.40`, `BUG-56`).

### Tarefas da fase 12

- [ ] **Task 12.1: Documentar Formato B de Bitpacking em Shared Memory para Crossover como Trabalho Futuro na Dissertação (`DP12-1`)**
  + **What:** Documentar na seção de trabalhos futuros da dissertação ([second_draft.tex](../../documents_tcc/tex/second_draft.tex) / [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex)) a transição analítica do Formato A (byte array $\mathcal{O}(n)$, $n \le 2.103$) para o Formato B (bitpacked array $\lceil n/32 \rceil \times 4$ bytes) com `atomicOr`, fundamentada em [analysis-crossover-gpu-architecture-and-scaling.md](../../documentation/knowledge-base/analysis-crossover-gpu-architecture-and-scaling.md#format-b-bitpacked-array-uint32_t-bitmapn--32).
  + **Why:** O benchmark atual comprova que o byte array é suficiente e livre de contenção para toda a TSPLIB95; registrar a variante bitpacked como trabalho futuro consolida o rigor analítico da dissertação demonstrando que o algoritmo escala teoricamente para instâncias com mais de 10.000 vértices sem estourar os 48 KB de SRAM do multiprocessador.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex).
  + **When:** Início da Fase 12.
  + **How:** Inserir item detalhado no capítulo de conclusões e trabalhos futuros de `second_draft.tex` citando os limites de 48 KB por SM e a economia de $8\times$ em memória compartilhada com o Formato B.
  + **How much:** Baixo — ~25 linhas de texto LaTeX e referências aos limites de hardware.
  + **Stakeholder Implications:** Valoriza a contribuição metodológica da dissertação perante a banca examinadora ao antecipar os limites físicos de escala do solver.
  + **DoD:** `Task 12.1` e `DP12-1` devidamente registrados em `table-of-symbols.plan.md` e `plan-reorganization-and-typing.plan.md`; `python -m benchmark_scripts.check_docs` executado com `undefined symbols: 0`.
  + **Status / Pendência:** Reaberta a pedido do usuário para revisão aprofundada da redação e adequação de conteúdo em `parts/06-conclusoes.tex` (linhas 79-87).

- [ ] **Task 12.2: Documentar Extensões de Operadores e Micro-otimizações de Kernels CUDA na Dissertação (`DP12-2`)**
  + **What:** Documentar na seção de trabalhos futuros ([parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex)) e fundamentação as oportunidades de extensão algorítmica (cruzamento EAX com ciclos AB, 3-opt / Or-opt para grafos direcionados ATSP preservando orientação de arestas, modelo em ilhas multi-stream) e micro-otimizações de kernels GPU (reduções com warp shuffles `__shfl_down_sync`, padding de shared memory para eliminação de bank conflicts, persistent threads / grid-stride loops e textura/read-only cache via `__ldg`).
  + **Why:** Fornece um roteiro concreto de engenharia e pesquisa de alto impacto para trabalhos subsequentes, demonstrando domínio profundo sobre as arquiteturas CUDA e metaheurísticas de ponta para roteamento.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex).
  + **When:** Após Task 12.1.
  + **How:** Adicionar itens estruturados de trabalhos futuros em `06-conclusoes.tex` com equações KaTeX e citações acadêmicas primárias.
  + **How much:** Baixo — ~35 linhas de texto LaTeX.
  + **Stakeholder Implications:** Consolida o valor acadêmico da dissertação para publicação em periódicos e conferências.
  + **DoD:** `Task 12.2` e `DP12-2` registrados em `table-of-symbols.plan.md` e `plan-reorganization-and-typing.plan.md`; `python -m benchmark_scripts.check_docs` com 0 erros.

- [x] **Task 12.3: Avaliação de Viabilidade e Descapamento do Benchmark de CPU com Listas de Candidatos (`DP12-3`)**
  + **What:** Avaliar a viabilidade de remover ou elevar o corte preliminar de CPU ($N \le 100$) em [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py) e no orquestrador de benchmark, aproveitando a aceleração de $2.6\times$ a $8.2\times$ propiciada por listas de candidatos ($k=20$) na busca local 2-opt em CPU, definindo um teto de timeout seguro por instância e parametrizando a seleção do algoritmo de referência (`baseline_algorithm="CPU"`).
  + **Why:** O corte em $N \le 100$ foi imposto quando o 2-opt em CPU demorava minutos por instância. Com a implementação modular e vetorizada, instâncias de porte médio ($100 < n \le 400$, ex.: `ch150`, `pr264`, `lin318`) podem ser executadas em tempo tratável, expandindo a linha de base de comparação direta CPU vs GPU na dissertação.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py), [orchestration.py](../../src/benchmarking/runner/orchestration.py), [report_generator.py](../../src/benchmarking/analysis/report_generator.py).
  + **When:** Antes da coleta integral do benchmark (Task 12.4).
  + **How:** Parametrizar `baseline_algorithm` em `report_generator.py` e `orchestration.py`, elevar `CPU_SIZE_THRESHOLD = 400` em `run_selected_problems.py`, ajustar repetições para 5 e substituir micro-instâncias ($n < 48$) por instâncias robustas spanning $48 \le n < 400$.
  + **How much:** Médio — ajuste no script de benchmark, suporte a baseline no gerador de relatórios e validação de regressão.
  + **Stakeholder Implications:** Permite responder diretamente ao Objetivo 1 e 5 da dissertação com linha de base CPU estendida além de 100 cidades e speedups mensurados diretamente contra CPU.
  + **DoD:** Limite de CPU formalmente calibrado para $n < 400$; baseline configurável via API; `Task 12.3` e `DP12-3` registrados no plano e tabela de símbolos; suite de testes verde (`pytest tests/benchmarking/test_integration_pipeline.py -q` $\to$ 10 passed); `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L21), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L253), [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L595).
    - **Resumo:** Parametrizada a seleção do algoritmo de referência (`baseline_algorithm="CPU"`) em `report_generator.py` e `orchestration.py`, eliminando a assunção fixa de `HybridNaive`. Em `run_selected_problems.py`, o corte preliminar de CPU foi elevado para $n < 400$ (`CPU_SIZE_THRESHOLD = 400`), as repetições reduzidas para 5, e instâncias micro ($n < 48$: `gr17`, `gr21`, `gr24`, `bayg29`, `bays29`, `gr48`) que sofriam distorção de runtime floor (<100 ms) foram substituídas por uma coorte balanceada de 10 problemas cobrindo $48 \le n < 400$ (`att48`, `berlin52`, `st70`, `eil76`, `gr96`, `kroA100`, `ch150`, `brg180`, `pr264`, `lin318`) em todos os formatos de distância (`ATT`, `EUC_2D`, `GEO`, `EXPLICIT`).
    - **Atendimento ao DoD:** `run_comprehensive_benchmark` e `generate_result_tables` suportam `baseline_algorithm` e `cpu_size_threshold=400`; `pytest tests/benchmarking/test_integration_pipeline.py -q` $\to$ 10 passed; `ruff` e `mypy` limpos; `python -m benchmark_scripts.check_docs` executado com zero drift (`stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0`, exit code 0).

- [ ] :running: **Task 12.4: Coleta Integral dos Resultados Experimentais da Dissertação e Atualização de Tabelas e DuckDB (`DP12-4`)**
  + **What:** Executar a coleta experimental completa do benchmark para a coorte da dissertação (10 repetições independentes com sementes determinísticas) nas variantes ativas, persistindo checkpoints em JSON, registros no banco `datasets/tsp.duckdb` e gerando as tabelas estatísticas finais (`selected_10_validation.md`/`.tex` ou coorte de 38 instâncias) com telemetria fina de 8 fases e $ddof=1$.
  + **Why:** Os números presentes no rascunho da dissertação ainda possuem avisos de dados provisórios (`\begin{warning}[Resultados provisórios]`). A coleta com o solver unificado da Fase 11 consolida o conjunto definitivo de dados para a defesa de mestrado.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [results_v3/](../../results_v3/), [datasets/tsp.duckdb](../../datasets/tsp.duckdb).
  + **When:** Após definição da política de CPU na Task 12.3.
  + **How:** Executar orquestrador de benchmark com persistência em DuckDB e geração automatizada de relatórios estatísticos.
  + **How much:** Alto em tempo de processamento computacional; baixo em complexidade de código (pipeline já validado na Fase 11).
  + **Stakeholder Implications:** Disponibiliza o corpo definitivo de evidências empíricas para a banca examinadora.
  + **DoD:** Checkpoints gerados e validados contra schema; tabelas Markdown e LaTeX exportadas com 100% de fechamento de balanço temporal (`Sum/Wall %` entre 97% e 100%); DuckDB populado.

- [ ] **Task 12.5: Síntese da Evolução do Projeto, Lições de Engenharia e Atualização Textual da Dissertação (`DP12-5`)**
  + **What:** Atualizar os capítulos da dissertação ([second_draft.tex](../../documents_tcc/tex/second_draft.tex), [parts/04-resultados.tex](../../documents_tcc/tex/parts/04-resultados.tex), [parts/05-discussao.tex](../../documents_tcc/tex/parts/05-discussao.tex), [parts/06-conclusoes.tex](../../documents_tcc/tex/parts/06-conclusoes.tex)) com as métricas consolidadas da Task 12.4, removendo os avisos de dados provisórios, e sintetizar na discussão as lições de engenharia e evolução do projeto documentadas em [notes_project_direction_questions.md](../../notes_project_direction_questions.md) (hermetismo de residência, eliminação de gargalos host-device via amostragem vetorizada, warmup AOT de kernels e invariantes estritos).
  + **Why:** Transforma o rascunho provisório em um texto de defesa final e coeso, destacando não apenas os speedups brutos mas a contribuição metodológica de engenharia de software para aceleração em GPU.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [documents_tcc/tex/parts/](../../documents_tcc/tex/parts/).
  + **When:** Após a conclusão da coleta experimental (Task 12.4).
  + **How:** Atualizar números, tabelas e parágrafos de discussão nas partes TeX correspondentes, verificando compilação com `latexmk -lualatex`.
  + **How much:** Médio — revisão textual e compilação de capítulos.
  + **Stakeholder Implications:** Conclui formalmente a dissertação para submissão e agendamento da defesa.
  + **DoD:** Dissertação atualizada com dados reais sem avisos de dados provisórios; `python -m benchmark_scripts.check_docs` com exit 0; compilação LaTeX validada.

- [x] **Task 12.6: Integração no Pipeline da Geração de Diagramas de Diferença Crítica de Demšar em TikZ (`DP12-6`)**
  + **What:** Implementar a geração automatizada de diagramas de diferença crítica (CD) de Demšar (1986, 2006) em formato vetorial TikZ (`.tikz`) a partir dos resultados agregados do benchmark, fundamentada em [analysis-demsar-critical-difference-diagrams.md](./knowledge-base/analysis-demsar-critical-difference-diagrams.md) e [guide-statistical-demsar-post-hoc-tests.md](../../documentation/knowledge-base/guide-statistical-demsar-post-hoc-tests.md), com teste omnibus de Friedman, pós-hoc de Nemenyi com fator de Studentized range corrigido ($q_\alpha = Q_\alpha / \sqrt{2}$ prevenindo inflação de 41,4%), detecção de cliques e exportação de arquivo compilável para o capítulo de resultados de [second_draft.tex](../../documents_tcc/tex/second_draft.tex).
  + **Why:** Substitui representações tabulares densas por uma visualização gráfica canônica e rigorosa que consolida em uma única figura as equivalências e diferenças estatisticamente significantes entre variantes CPU, híbridas e GPU pura, atendendo aos padrões editoriais de periódicos internacionais de computação evolutiva e pesquisa operacional.
  + **Who:** Autor e Agente.
  + **Where:** [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md), [table-of-symbols.plan.md](./table-of-symbols.plan.md), [analysis-demsar-critical-difference-diagrams.md](./knowledge-base/analysis-demsar-critical-difference-diagrams.md), `src/benchmarking/analysis/`.
  + **When:** Após estabilização dos dados consolidados da bateria experimental.
  + **How:** Criar módulo/função `export_cd_diagram_tikz` integrada a `report_generator.py` ou `problem_statistics.py` que calcula postos médios $R_j$, valida a conservação $\sum R_j = k(k+1)/2$, extrai o valor crítico $CD$, computa cliques conexos e gera o arquivo LaTeX TikZ standalone.
  + **How much:** Médio — ~120 linhas de código python e template TikZ vetorial.
  + **Stakeholder Implications:** Fornece o principal artefato visual de validação estatística da dissertação para a banca examinadora.
  + **DoD:** `export_cd_diagram_tikz` implementado e coberto por testes unitários; arquivo `.tikz` gerado com sintaxe válida e compilável via LuaLaTeX; `python -m benchmark_scripts.check_docs` com exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cd_diagram.py#L1-L267](../../src/benchmarking/analysis/cd_diagram.py#L1-L267), [report_generator.py#L702-L723](../../src/benchmarking/analysis/report_generator.py#L702-L723), [test_unit_cd_diagram.py](../../tests/benchmarking/test_unit_cd_diagram.py).
    - **Resumo:** Implementado o módulo `cd_diagram.py` com cálculo de postos médios, teste de Friedman, diferença crítica de Nemenyi ($CD = q_\alpha \sqrt{\frac{k(k+1)}{6N}}$ com correção de Studentized range $q_\alpha = Q_\alpha / \sqrt{2}$) e geração de arquivo vetorial TikZ (`export_cd_diagram_tikz`). Integrado diretamente a `report_generator.py:generate_tables_and_reports` para geração automática ao processar os benchmarks. Conforme a Regra 10 do `AGENTS.md`, nenhum alias de retrocompatibilidade foi mantido.
    - **Atendimento ao DoD:** 4 testes unitários em `tests/benchmarking/test_unit_cd_diagram.py` aprovados cobrindo postos médios, Nemenyi CD, cliques conexos, formatação TikZ com eixo de ranks invertido (1 = melhor) e compilação do arquivo `.tikz`; `check_docs` com exit 0.

- [x] **Task 12.7: Correção de Crash no Teste de Friedman com Amostras Desiguais / Ragged Arrays (`BUG-33`)**
  + **What:** Corrigir a fragilidade em [problem_statistics.py#L181](../../src/benchmarking/analysis/problem_statistics.py#L181) e [statistics.py#L365](../../src/benchmarking/analysis/statistics.py#L365) onde `analyzer.friedman_test` levanta `ValueError: All data_sets must have the same number of observations (paired samples). Got lengths:[5, 10, 10, 10]` quando checkpoints residuais possuem número heterogêneo de repetições (ex.: CPU com 5 repetições vs GPU com 10 repetições de execuções anteriores) ou quando repetições isoladas falham.
  + **Why:** O teste de Friedman é um ANOVA não-paramétrico bidirecional sobre blocos casualizados e requer amostras pareadas equilibradas. Na presença de arrays com comprimentos desiguais (*ragged arrays*), o benchmark não pode abortar fatalmente: deve truncar de forma determinística e balanceada para $\min(\text{lengths})$ com aviso explícito no log, ou filtrar apenas observações com pareamento completo.
  + **Who:** Autor e Agente.
  + **Where:** [problem_statistics.py#L181](../../src/benchmarking/analysis/problem_statistics.py#L181), [statistics.py#L365](../../src/benchmarking/analysis/statistics.py#L365), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Antes da nova rodada de coleta experimental.
  + **How:** Em `problem_statistics.py`, antes de chamar `friedman_test`, inspecionar os comprimentos dos arrays de custo de cada algoritmo. Se divergirem, emitir `logging.warning` e parear sobre as $m = \min_{j} \text{len}(D_j)$ primeiras repetições (ou rejeitar o teste de forma graciosa retornando campos nulos/NaN com diagnóstico estruturado).
  + **How much:** Baixo — ~20 linhas de validação defensiva e testes de regressão.
  + **Stakeholder Implications:** Elimina crashes fatais em lotes de longa duração causados por checkpoints incrementais ou históricos residuais.
  + **DoD:** Teste unitário em `tests/benchmarking/test_unit_problem_statistics.py` reproduzindo `lengths:[5, 10, 10, 10]` sem levantar `ValueError`; execução graciosa com aviso registrado; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_statistics.py#L173-L212](../../src/benchmarking/analysis/problem_statistics.py#L173-L212), [test_unit_problem_statistics.py](../../tests/benchmarking/test_unit_problem_statistics.py).
    - **Resumo:** Inserida validação defensiva em `perform_statistical_analysis` inspecionando comprimentos dos arrays de custo e tempo. Se heterogêneos, emite aviso via `logging.warning` e trunca os arrays para o comprimento mínimo comum (`min_cost_len`). Caso `min_cost_len < 2`, o teste de Friedman é ignorado graciosamente com valores padrão sem abortar a análise.
    - **Atendimento ao DoD:** `pytest tests/benchmarking/test_unit_problem_statistics.py` executou com sucesso (3 passed) cobrindo reprodução de `lengths: [5, 10, 10, 10]`, skip gracioso com `min_len < 2` e execução padrão com comprimentos homogêneos; `check_docs` com exit 0.

- [x] **Task 12.8: Agregação de Speedups na Tabela 2 por Média Geométrica Pareada (`BUG-34`)**
  + **What:** Corrigir a agregação global de speedups na Tabela 2 em [report_generator.py#L340](../../src/benchmarking/analysis/report_generator.py#L340), substituindo a razão de médias aritméticas de tempos de parede $\frac{\sum T_{\text{base}}}{\sum T_{\text{cand}}}$ pela média geométrica dos speedups pareados por instância $\left(\prod_{i=1}^M S_i\right)^{1/M} = \exp\left(\frac{1}{M}\sum_{i=1}^M \ln S_i\right)$, onde $S_i = \frac{T_{\text{base}, i}}{T_{\text{cand}, i}}$.
  + **Why:** Conforme estabelecido no artigo clássico de Fleming & Wallace (1986, *How not to lie with statistics: the correct way to summarize benchmark results*), a média aritmética de razões normalizadas ou a razão de somas aritméticas de tempos é distorcida pela escala das instâncias (problemas maiores dominam o somatório total). Apenas a média geométrica satisfaz a propriedade de reciprocidade multiplicativa e invariância de normalização para resumo de benchmarks.
  + **Who:** Autor e Agente.
  + **Where:** [report_generator.py#L340](../../src/benchmarking/analysis/report_generator.py#L340), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Antes da geração das tabelas definitivas.
  + **How:** Alterar a computação da linha `"Average Speedup"` em `report_generator.py` para calcular $S_i$ por problema comum entre baseline e candidato, agregando pela média geométrica via `scipy.stats.gmean` ou `np.exp(np.mean(np.log(speedups)))`.
  + **How much:** Baixo — ~15 linhas em `report_generator.py` e atualização das asserções de teste.
  + **Stakeholder Implications:** Garante integridade metodológica e conformidade com as melhores práticas da literatura de arquitetura de computadores e metaheurísticas.
  + **DoD:** Teste unitário comprovando agregação por média geométrica em instâncias de portes heterogêneos; ausência de distorção por instâncias lentas; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py#L248-L318](../../src/benchmarking/analysis/report_generator.py#L248-L318), [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py).
    - **Resumo:** Implementado o cálculo de speedup médio na Tabela 2 (versões Markdown e LaTeX) via média geométrica dos speedups pareados por problema comum (`np.exp(np.mean(np.log(paired_speedups)))`), eliminando a distorção por escala de instâncias lentas conforme Fleming & Wallace (1986).
    - **Atendimento ao DoD:** Teste unitário `test_table2_speedup_uses_paired_geometric_mean` em `tests/benchmarking/test_unit_report_generator.py` aprovado com exit 0, validando cálculo exato de média geométrica ($\approx 5{,}00\times$) sem contaminação por razão aritmética de somas ($1{,}82\times$).

- [x] **Task 12.9: Desacoplamento da Guarda de Exibição da Tabela 4 para Execuções Mistas CPU/GPU (`BUG-35`)**
  + **What:** Desacoplar a guarda de verificação de telemetria fina de fases em [report_generator.py#L442-L451](../../src/benchmarking/analysis/report_generator.py#L442-L451). Atualmente, `_has_phase_timings` utiliza `all(...)`, de modo que a presença de qualquer algoritmo que não registre fases GPU (como a CPU pura) faz a função retornar `False`, suprimindo a Tabela 4 inteira para todos os algoritmos.
  + **Why:** A Tabela 4 apresenta a decomposição temporal das 8 fases do pipeline genético em GPU (`Init`, `Sel`, `CX`, `Mut`, `Eval`, `LS`, `Surv`, `Host Sync`). Em execuções que incluem o baseline de CPU, as variantes GPU continuam coletando telemetria completa via eventos CUDA, mas seus dados eram ocultados dos relatórios por causa da guarda restritiva `all()`. Desacoplar para renderizar colunas disponíveis com `"-"` para variantes sem fases é uma solução arquitetural limpa que não altera os cronômetros nem a coleta.
  + **Who:** Autor e Agente.
  + **Where:** [report_generator.py#L442-L451](../../src/benchmarking/analysis/report_generator.py#L442-L451), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com os ajustes de relatórios.
  + **How:** Modificar `_has_phase_timings` para verificar `any(...)` ou iterar condicionalmente por algoritmo, gerando a tabela se ao menos uma variante possuir telemetria de fases e preenchendo células faltantes com `"-"` ou `N/A`.
  + **How much:** Baixo — ~20 linhas em `report_generator.py`.
  + **Stakeholder Implications:** Assegura que o perfil fino de 8 fases em GPU figure na dissertação mesmo em experimentos conjuntos com a CPU.
  + **DoD:** Teste com suite mista CPU + GPU gera a Tabela 4 preenchendo as fases de Hybrid e FullGPU e marcando `"-"` para CPU; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py#L448-L578](../../src/benchmarking/analysis/report_generator.py#L448-L578), [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py).
    - **Resumo:** Desacoplada a função `_has_phase_timings` em `report_generator.py` para retornar `True` caso ao menos um algoritmo possua telemetria de fases (`any`). Na renderização (Markdown e LaTeX), algoritmos sem fases coletadas (como o baseline de CPU) preenchem as colunas com `"-"` de forma graciosa sem suprimir a tabela inteira.
    - **Atendimento ao DoD:** Teste unitário `test_table4_emitted_for_mixed_runs_when_one_algorithm_has_phase_timings` em `tests/benchmarking/test_unit_report_generator.py` aprovado com exit 0, comprovando emissão da Tabela 4 com fases detalhadas para `Genetic_GPU_Opt` e células `"-"` para `Genetic_CPU`.

- [x] **Task 12.10: Sincronização de Contrato em Execuções Falhas de Repetição no Banco de Dados (`BUG-36`)**
  + **What:** Sincronizar o contrato de persistência em [db_operations.py#L166-L178](../../src/benchmarking/database/db_operations.py#L166-L178) para computar médias e agregados de execução utilizando estritamente o número de repetições bem-sucedidas (`successful_runs`), prevenindo divisões fixas por `expected_repetitions` quando repetições individuais falham ou são canceladas.
  + **Why:** Se 1 das 10 repetições sofrer falha isolada (ex.: OOM ou interrupção), assumir denominador 10 corrompe a média amostral de tempo e custo no banco DuckDB, gerando divergência em relação aos checkpoints JSON de [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py).
  + **Who:** Autor e Agente.
  + **Where:** [db_operations.py#L166-L178](../../src/benchmarking/database/db_operations.py#L166-L178), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Antes da coleta integral para DuckDB.
  + **How:** Em `db_operations.py`, validar `count(distinct run_id)` efetivamente persistido e normalizar os acumuladores por `n_completed`, levantando aviso ou registrando o status `incomplete` na tabela de resumo.
  + **How much:** Baixo — ~25 linhas e testes unitários com repetições parciais.
  + **Stakeholder Implications:** Preserva a integridade e consistência relacional dos dados gravados em `datasets/tsp.duckdb`.
  + **DoD:** Teste unitário em `tests/benchmarking/test_unit_db_operations.py` validando cálculo exato de média sob falha de repetição; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [db_operations.py#L161-L255](../../src/benchmarking/database/db_operations.py#L161-L255), [test_unit_db_operations.py](../../tests/benchmarking/database/test_unit_db_operations.py).
    - **Resumo:** Sincronizado o contrato de agregação e persistência em `db_operations.py` para utilizar estritamente o número de repetições bem-sucedidas (`successful_runs`) como denominador no cálculo de médias e métricas agregadas (`compute_run_aggregates`), tratando graciosamente o caso de 0 execuções válidas e rejeitando payloads onde `successful_runs` excede `repetitions` ou diverge do tamanho dos arrays brutos.
    - **Atendimento ao DoD:** `pytest tests/benchmarking/database/test_unit_db_operations.py` executou 122 testes com exit 0, incluindo validação de arrays correspondentes a `successful_runs`, rejeição de inconsistências e cálculo de agregação com divisor correto.

- [x] **Task 12.11: Proteção contra Corrupção em Retomada de Checkpoint e Prevenção de Colisão de Sementes (`BUG-37`)**
  + **What:** Implementar salvaguardas robustas em [orchestration.py#L416](../../src/benchmarking/runner/orchestration.py#L416) para captura defensiva de erros de leitura/desserialização de JSON (`JSONDecodeError`, arquivo truncado por queda de energia ou interrupção de processo) com quarentena automática (`.corrupt.bak`), e garantir em [algorithm_runner.py#L186](../../src/benchmarking/runner/algorithm_runner.py#L186) que a extensão incremental de repetições utilize sementes indexadas por offset (`seed = base_seed + run_idx`), evitando reuso inadvertido de sementes prévias ao expandir repetições.
  + **Why:** Durante execuções longas, interrupções abruptas podem deixar checkpoints parciais ou corrompidos em disco que travam execuções subsequentes. Além disso, se o usuário solicitar 5 repetições adicionais sobre um lote já existente, as sementes devem ser estritamente novas para garantir independência estatística.
  + **Who:** Autor e Agente.
  + **Where:** [orchestration.py#L416](../../src/benchmarking/runner/orchestration.py#L416), [algorithm_runner.py#L186](../../src/benchmarking/runner/algorithm_runner.py#L186), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Antes da execução das baterias completas de benchmark.
  + **How:** Envolver a desserialização de checkpoints em bloco `try/except (json.JSONDecodeError, KeyError, OSError)` com renomeação para quarentena e log de alerta; em `algorithm_runner.py`, derivar sementes a partir do índice sequencial absoluto da repetição.
  + **How much:** Baixo — ~30 linhas em `orchestration.py` e `algorithm_runner.py`.
  + **Stakeholder Implications:** Elimina intervenção manual de recuperação de arquivos e blinda a independência estatística dos experimentos.
  + **DoD:** Testes unitários com arquivos corrompidos e retomada incremental com verificação de não-colisão de sementes; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [checkpoint_io.py#L234-L250](../../src/benchmarking/runner/checkpoint_io.py#L234-L250), [orchestration.py#L418-L429](../../src/benchmarking/runner/orchestration.py#L418-L429), [algorithm_runner.py#L185-L206](../../src/benchmarking/runner/algorithm_runner.py#L185-L206), [test_unit_checkpoint_io.py](../../tests/benchmarking/test_unit_checkpoint_io.py).
    - **Resumo:** Implementada a função `quarantine_checkpoint` em `checkpoint_io.py`, acionada defensivamente em `orchestration.py` via `try/except (json.JSONDecodeError, KeyError, OSError)` renomeando checkpoints corrompidos para `<name>.corrupt.<ts>.bak` e iniciando execução limpa; em `algorithm_runner.py`, implementado rastreamento e cálculo de sementes incrementais com verificação `while run_seed in existing_seeds: current_run_idx += 1` e ajuste de offset por `max_offset + 1`, garantindo sementes novas e sem colisão ao estender repetições.
    - **Atendimento ao DoD:** `pytest tests/benchmarking/test_unit_checkpoint_io.py` executou 26 testes com exit 0, comprovando quarentena automática de JSONs corrompidos e ausência total de colisão de sementes em modo incremental.

- [x] **Task 12.12: Padronização Estatística de Variância Amostral com Correção de Bessel (`ddof=1`) (`BUG-38`)**
  + **What:** Padronizar rigorosamente em [algorithm_runner.py#L285-L310](../../src/benchmarking/runner/algorithm_runner.py#L285-L310), [problem_statistics.py#L110](../../src/benchmarking/analysis/problem_statistics.py#L110) e [report_generator.py](../../src/benchmarking/analysis/report_generator.py) o cálculo de desvio padrão amostral e coeficiente de variação ($CV$) utilizando o estimador não-viesado com correção de Bessel (`ddof=1`, divisor $N - 1$), eliminando qualquer ocorrência residual de `ddof=0` (divisor $N$).
  + **Why:** O estimador de máxima verossimilhança com divisor $N$ subestima sistematicamente a variância populacional por um fator de $\frac{N-1}{N}$ (em $N=5$, subestimação de $\approx 10,6\%$ no desvio padrão; em $N=2$, $\approx 29,3\%$). Para aferição científica e cálculo de intervalos de confiança t-Student em amostras pequenas, o estimador amostral não-viesado com $N-1$ graus de liberdade é mandatório.
  + **Who:** Autor e Agente.
  + **Where:** [algorithm_runner.py#L285-L310](../../src/benchmarking/runner/algorithm_runner.py#L285-L310), [problem_statistics.py#L110](../../src/benchmarking/analysis/problem_statistics.py#L110), [report_generator.py](../../src/benchmarking/analysis/report_generator.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com Task 12.8.
  + **How:** Substituir chamadas `np.std(..., ddof=0)` ou sem argumento explícito por `np.std(..., ddof=1)` (e `np.nanstd(..., ddof=1)`) em todos os pontos de agregação de telemetria e tabelas.
  + **How much:** Baixo — ~15 substituições pontuais e verificação em testes de estatística.
  + **Stakeholder Implications:** Garante exatidão estatística e comparabilidade formal com a literatura de metaheurísticas.
  + **DoD:** Auditoria `grep` confirma `ddof=1` em todos os cálculos amostrais de benchmark; testes unitários em `tests/benchmarking/test_unit_statistics.py` verdes; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithm_runner.py#L349-L378](../../src/benchmarking/runner/algorithm_runner.py#L349-L378), [orchestration.py#L202-L213](../../src/benchmarking/runner/orchestration.py#L202-L213), [report_generator.py#L244-L300](../../src/benchmarking/analysis/report_generator.py#L244-L300), [aggregate_statistics.py#L157](../../src/benchmarking/analysis/aggregate_statistics.py#L157), [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py), [test_unit_runner_telemetry.py](../../tests/benchmarking/test_unit_runner_telemetry.py).
    - **Resumo:** Padronizado o cálculo de desvio padrão amostral e coeficiente de variação utilizando estritamente a correção de Bessel (`ddof=1`, divisor $N - 1$) em `algorithm_runner.py`, `orchestration.py`, `report_generator.py` e `aggregate_statistics.py`, com guarda `if n > 1 else 0.0`.
    - **Atendimento ao DoD:** Testes unitários `test_table2_std_uses_bessels_correction`, `test_phase_share_cv_uses_sample_standard_deviation` e `test_algorithm_runner_sample_std_uses_bessels_correction` aprovados com exit 0, confirmando aplicação estrita de `ddof=1` em todas as métricas de dispersão amostrais.

- [x] **Task 12.13: Otimização de Consultas SQL e Inclusão de `FullGPUModular` na Matriz de Speedup (`BUG-39`)**
  + **What:** Refatorar [db_operations.py#L540-L620](../../src/benchmarking/database/db_operations.py#L540-L620) para eliminar o padrão anti-otimização N+1 de consultas SQL na geração da matriz de speedup, unificando a extração em uma única consulta analítica com agregação via CTE / `GROUP BY`, e incluir formalmente a variante `FullGPUModular` como coluna e linha de comparação na matriz gerada pelo banco de dados.
  + **Why:** A execução de consultas repetidas por problema em um loop serial degrada a performance de exportação analítica. Além disso, a ausência de `FullGPUModular` na matriz de speedup do DuckDB deixava a variante GPU pura de fora das tabelas consolidadas geradas diretamente pelo banco.
  + **Who:** Autor e Agente.
  + **Where:** [db_operations.py#L540-L620](../../src/benchmarking/database/db_operations.py#L540-L620), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Antes da exportação analítica final.
  + **How:** Reescrever `compute_speedup_matrix` utilizando agregação agregada DuckDB e registrar `FullGPUModular` nas listas de algoritmos canônicos do módulo de banco.
  + **How much:** Baixo — ~35 linhas de SQL/python e testes de integração.
  + **Stakeholder Implications:** Proporciona exportação de dados mais rápida e paridade completa de algoritmos nas análises baseadas em banco de dados.
  + **DoD:** Teste unitário em `tests/benchmarking/test_unit_db_operations.py` validando matriz consolidada contendo `FullGPUModular` executada em consulta única; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [db_operations.py#L540-L620](../../src/benchmarking/database/db_operations.py#L540-L620), [test_unit_db_operations.py](../../tests/benchmarking/database/test_unit_db_operations.py#L930-L1041).
    - **Resumo:** `compute_speedup_matrix` em `db_operations.py` refatorado para utilizar uma única consulta analítica DuckDB agrupada (`GROUP BY problem_name, algorithm`), eliminando completamente o laço iterativo N+1 de queries. A lista padrão de algoritmos a comparar passou a incluir formalmente `FullGPUModular`.
    - **Atendimento ao DoD:** Testes unitários em `tests/benchmarking/database/test_unit_db_operations.py` aprovados, incluindo espionagem de chamadas (`ConnSpy`) comprovando `execute_count == 1` para o cálculo da matriz completa com `FullGPUModular`; `check_docs` com exit 0.

- [x] **Task 12.14: Arquitetura de Persistência Atômica de Checkpoints por Repetição com Isolamento de Tempo de I/O (`DP12-7`)**
  + **What:** Implementar a persistência contínua e atômica de checkpoints ao término de cada repetição estocástica em `src/benchmarking/` (em vez de reter todas as repetições em memória RAM e gravar apenas ao término do lote de 30 repetições), garantindo que a medição de tempo do algoritmo de otimização (`time.perf_counter()`) exclua estritamente o tempo de gravação atômica em disco.
  + **Why:** Atualmente, `algorithm_runner.py` acumula todas as repetições em listas na memória RAM e só grava o checkpoint via `save_checkpoint_atomic` em `orchestration.py:584` após a conclusão das 30 repetições. Se o processo for interrompido via `Ctrl+C` ou falhar (ex.: na repetição 20 de uma instância lenta como `lin318` na CPU que consome mais de 4 horas), todo o progresso em voo é descartado, forçando o recálculo do zero.
  + **Who:** Autor e Agente (subagente `02a8eba8-60f2-4add-be07-b90bf9f1404a`).
  + **Where:** [algorithm_runner.py#L210-L280](../../src/benchmarking/runner/algorithm_runner.py#L210-L280), [orchestration.py#L415-L470](../../src/benchmarking/runner/orchestration.py#L415-L470), [checkpoint_io.py#L190-L220](../../src/benchmarking/runner/checkpoint_io.py#L190-L220), [db_operations.py](../../src/benchmarking/database/db_operations.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Após a conclusão da execução atual do benchmark e revisão aprofundada da arquitetura de benchmarking.
  + **How:** Modificar o fluxo de execução para que o `algorithm_runner` receba um callback ou caminho de checkpoint para serializar o estado parcial atualizado a cada repetição concluída; isolar o bloco `time.perf_counter` estritamente ao método `evolve()`; garantir que a retomada com `--no-incremental` detecte exatamente quantas repetições foram gravadas e execute apenas as restantes para atingir o teto; manter salvamento atômico seguro contra corrupção parcial.
  + **How much:** Médio — requer coordenação entre runner, gerenciador de checkpoint e banco de dados.
  + **Stakeholder Implications:** Elimina a vulnerabilidade de perda de horas de computação em interrupções e oferece observabilidade em tempo real do progresso das execuções.
  + **DoD:** Teste de integração simulando interrupção na repetição $K < N$ e retomada com `--no-incremental`, comprovando que apenas $N - K$ repetições são executadas e o checkpoint final contém $N$ execuções válidas; tempo de I/O comprovadamente não contabilizado no `elapsed`; suíte de testes passando com exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithm_runner.py#L356-L424](../../src/benchmarking/runner/algorithm_runner.py#L356-L424), [orchestration.py#L566-L595](../../src/benchmarking/runner/orchestration.py#L566-L595), [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py), [test_unit_checkpoint_io.py#L417-L599](../../tests/benchmarking/test_unit_checkpoint_io.py#L417-L599).
    - **Resumo:** Implementado callback `on_repetition_completed` em `algorithm_runner.py` acionado após cada repetição estocástica concluída, persistindo atomicamente o checkpoint parcial em disco via `save_checkpoint_atomic`. O registro de tempo `elapsed = time.perf_counter() - start_time` é finalizado imediatamente após `evolve()`, garantindo isolamento temporal estrito de I/O em disco.
    - **Atendimento ao DoD:** Testes unitários `test_per_repetition_checkpoint_callback_and_timing_isolation` e `test_atomic_checkpoint_persistence_on_interrupted_run` aprovados, comprovando que injeção de delay de I/O não vaza para os tempos medidos e que retomada executa estritamente as $N - K$ repetições faltantes; `check_docs` com exit 0.

- [x] **Task 12.15: Infraestrutura Unificada de Logging com Destino Duplo (Console e Arquivo Persistente) (`DP12-8`)**
  + **What:** Padronizar e refatorar a infraestrutura de logging do subsistema `src/benchmarking/` e de scripts de execução (`benchmark_scripts/`), introduzindo captura contínua e simultânea em console (`StreamHandler`) e em arquivo de log persistente (`FileHandler`) no diretório de saída do benchmark (`output_dir / f"{benchmark_name}.log"`).
  + **Why:** Atualmente, a inicialização de logging utiliza `logging.basicConfig` apenas com `StreamHandler(sys.stdout)` direcionado ao pseudo-terminal `/dev/pts/*`. Se o processo rodar em background ou se o scrollback do terminal for perdido, nenhum log textual é retido em disco em `results_v4/`, dificultando a depuração pós-morte e o monitoramento autônomo por agentes.
  + **Who:** Autor e Agente (subagente `02a8eba8-60f2-4add-be07-b90bf9f1404a`).
  + **Where:** [orchestration.py#L340-L360](../../src/benchmarking/runner/orchestration.py#L340-L360), [run_chapter4_benchmark.py#L194-L205](../../benchmark_scripts/run_chapter4_benchmark.py#L194-L205), [run_selected_problems.py#L47-L55](../../benchmark_scripts/run_selected_problems.py#L47-L55), [cli/benchmark.py#L183-L189](../../src/cli/benchmark.py#L183-L189), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com a Task 12.14.
  + **How:** Criar utilitário centralizado em `src/benchmarking/logging.py` (ou em `orchestration.py`) que configura formatador unificado com timestamps UTC/local, rotaciona ou apenda em arquivo `{benchmark_name}.log` no diretório de saída correspondente, e assegura que subprocessos e handlers respeitem níveis de log configuráveis.
  + **How much:** Baixo — ~40 linhas de código utilitário e integração nos pontos de entrada.
  + **Stakeholder Implications:** Habilita auditoria completa e análise forense pós-execução diretamente pelo sistema de arquivos.
  + **DoD:** Execução do runner cria e popula comprovadamente `output_dir / f"{benchmark_name}.log"` com paridade de mensagens em relação ao console; testes unitários validando configuração de handlers; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [logging_config.py#L1-L71](../../src/benchmarking/logging_config.py#L1-L71), [orchestration.py#L54-L262](../../src/benchmarking/runner/orchestration.py#L54-L262), [test_unit_logging.py#L1-L69](../../tests/benchmarking/test_unit_logging.py#L1-L69).
    - **Resumo:** Criado o módulo `logging_config.py` com a função centralizada `setup_benchmark_logging(output_dir, benchmark_name, log_level)`. Configura destino duplo simultâneo: `StreamHandler(sys.stdout)` e `FileHandler` em modo append gravando em `output_dir / f"{benchmark_name}.log"`. Previne duplicação de handlers em chamadas subsequentes e foi integrado aos pontos de entrada (`orchestration.py`, `run_chapter4_benchmark.py`, `run_selected_problems.py` e `src/cli/benchmark.py`).
    - **Atendimento ao DoD:** Testes unitários `test_setup_benchmark_logging_emits_to_stdout_and_disk` e `test_setup_benchmark_logging_no_duplicate_handlers` em `test_unit_logging.py` 100% aprovados, validando emissão dual idêntica e ausência de handlers duplicados; `check_docs` com exit 0.

- [x] **Task 12.16: Relatórios e Análise Estatística 100% Agnósticos a Nomes de Algoritmos com Resolução Declarativa de Linha de Base (`DP12-9`)**
  + **What:** Refatorar [report_generator.py](../../src/benchmarking/analysis/report_generator.py) e módulos correlatos de análise estatística em `src/benchmarking/analysis/` para remover qualquer suposição ou amarração a nomes específicos de variantes algorítmicas (como "CPU", "HybridNaive", "HybridOptimized", "FullGPUModular"), permitindo avaliar e gerar relatórios comparativos para quaisquer algoritmos arbitrários e parametrizar a linha de base de forma explícita e declarativa.
  + **Why:** O gerador de relatórios atualmente mantém em `_get_baseline_algorithm` (L49-L55) condicionais fixas `if "CPU" in algorithm_names` e `if "HybridNaive" in algorithm_names`, além de omitir o tamanho amostral $|P_{\cap}|$ no resumo da média geométrica da Tabela 2. A substituição por linha de base local com adaga (`†`) foi explicitamente encomendada pelo usuário para permitir visualização de speedup relativo em instâncias de grande porte ($n > 400$) ou onde a CPU é desativada por `--skip-cpu`, mas a resolução deve ser puramente agnóstica a strings fixas.
  + **Who:** Autor e Agente (subagente `02a8eba8-60f2-4add-be07-b90bf9f1404a`).
  + **Where:** [report_generator.py#L48-L130](../../src/benchmarking/analysis/report_generator.py#L48-L130), [report_generator.py#L240-L320](../../src/benchmarking/analysis/report_generator.py#L240-L320), [report_generator.py#L640-L670](../../src/benchmarking/analysis/report_generator.py#L640-L670), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com as Tasks 12.14 e 12.15.
  + **How:** Eliminar as condicionais `if "CPU"` e `if "HybridNaive"` de `_get_baseline_algorithm`: se `baseline_algorithm` for informado pelo chamador, adotá-lo; caso contrário, adotar `algorithm_names[0]` do lote global; para instâncias onde esse baseline não foi executado (ex.: CPU com $n > 400$), manter formalmente a substituição encomendada pelo usuário pelo primeiro algoritmo disponível naquela coorte local com a adaga `†` (`1.00x†` / `1.00\times^{\dagger}`); calcular a média geométrica pareada (Fleming & Wallace 1986) na Tabela 2 exibindo o tamanho da amostra pareada $|P_{\cap}|$ (ex.: `21.33x (n=12)`).
  + **Blast Radius:** [report_generator.py](../../src/benchmarking/analysis/report_generator.py) (L49-L56, L110-L135, L680-L705), [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py).
  + **Relação com outras tarefas:** Fornece o contrato agnóstico de formatação e métricas consumido por `orchestration.py`, `run_chapter4_benchmark.py` (`Task 12.18`), desacoplado em `Task 12.23` e validado por `Task 12.24`.
  + **How much:** Médio — refatoração limpa da lógica de geração de tabelas e atualização de testes de relatório.
  + **Stakeholder Implications:** Transforma o gerador de benchmark em uma ferramenta verdadeiramente modular e universal para qualquer metaheurística, independente da dissertação.
  + **DoD:** Teste com algoritmos fictícios de nomes arbitrários (ex.: `CustomGeneticAlgorithm_A`, `CrazyCpuGpuLocalSearchEveryStep`) gera tabelas Markdown e LaTeX completas e válidas sem nenhum erro ou aviso; na ausência de linha de base explícita, o primeiro algoritmo disponível é adotado como referência ($1.00\times$) e os demais calculados em relação a ele de forma documentada; `pytest tests/benchmarking/test_unit_report_generator.py` 100% verde; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py#L49-L52](../../src/benchmarking/analysis/report_generator.py#L49-L52), [report_generator.py#L294-L300](../../src/benchmarking/analysis/report_generator.py#L294-L300), [report_generator.py#L352-L358](../../src/benchmarking/analysis/report_generator.py#L352-L358), [test_unit_report_generator.py#L15-L371](../../tests/benchmarking/test_unit_report_generator.py#L15-L371).
    - **Resumo:** Refatoração de `_get_baseline_algorithm` em [report_generator.py](../../src/benchmarking/analysis/report_generator.py) eliminando as verificações fixas `if "CPU"` e `if "HybridNaive"`, tornando a resolução de linha de base puramente dinâmica (adotando `algorithm_names[0]` por padrão quando não especificado, ou honrando `baseline_algorithm` explícito). Inclusão da anotação do tamanho amostral pareado $|P_{\cap}|$ no resumo da média geométrica da Tabela 2 (`f"{avg_speedup:.2f}x (n={len(paired_speedups)})"` em Markdown e `f"{avg_speedup:.2f}$\\times$ ($n={len(paired_speedups)}$)"` em LaTeX). Preservação integral da substituição por linha de base local com adaga (`†` / `^{\dagger}`) em instâncias onde a linha de base global não foi executada (ex.: CPU com $n > 400$ ou `--skip-cpu`), conforme determinação do usuário. Suíte de testes em [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py) ampliada para 8 testes (+3 novos testes cobrindo seleção dinâmica, override explícito e overlap parcial com contagem $n$).
    - **Atendimento ao DoD:** Execução de `pytest tests/benchmarking/test_unit_report_generator.py` com 8/8 testes verdes; `_get_baseline_algorithm` comprovadamente retorna o primeiro algoritmo para listas arbitrárias; Tabela 2 materializa `(n=...)` e `($n=...$)` em Markdown e LaTeX; Tabela 1 mantém anotação de adaga local `†`; `check_docs` retorna exit 0 sem dead links ou símbolos indefinidos.

- [x] **Task 12.17: Atualização e Validação Formal dos Esquemas JSON de Configuração (`configs/schemas/`) (`DP12-10`)**
  + **What:** Atualizar e validar formalmente os esquemas JSON em `configs/schemas/` ([algorithms.schema.json](../../configs/schemas/algorithms.schema.json), [benchmark.schema.json](../../configs/schemas/benchmark.schema.json), [problems.schema.json](../../configs/schemas/problems.schema.json)) para cobrir exaustivamente todos os parâmetros existentes e recém-adicionados (ex.: `candidate_k`, `max_iterations`, `mode`, `cpu_size_threshold`, `skip_cpu_default`, instâncias de catálogo `att532`, `si535`, `gr666`), garantindo que qualquer discrepância entre JSONs e esquemas seja detectada em tempo de build/teste.
  + **Why:** A imposição estrita de `"additionalProperties": false` sem o mapeamento de `"re_evaluate"` em `pipeline_step.params.properties` faz com que qualquer configuração válida de `PeriodicLocalSearchStep(target="population")` falhe na validação de schema. Além disso, `problems.schema.json` precisa aceitar metadados padronizados da TSPLIB (`edge_weight_type`, `comment`, `source`).
  + **Who:** Autor e Agente.
  + **Where:** [algorithms.schema.json](../../configs/schemas/algorithms.schema.json), [benchmark.schema.json](../../configs/schemas/benchmark.schema.json), [problems.schema.json](../../configs/schemas/problems.schema.json), [configs/algorithms.json](../../configs/algorithms.json), [configs/benchmark.json](../../configs/benchmark.json), [configs/problems.json](../../configs/problems.json), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com as Tasks 12.14 a 12.16.
  + **How:** Inserir `"re_evaluate": { "$ref": "#/$defs/strategy_slot" }` em `pipeline_step.params.properties` de `algorithms.schema.json`; liberar campos opcionais de metadados em `problems.schema.json`; validar exaustivamente com testes unitários em `test_unit_config_validation.py`.
  + **Blast Radius:** [algorithms.schema.json](../../configs/schemas/algorithms.schema.json) (L257-L282), [problems.schema.json](../../configs/schemas/problems.schema.json) (L20-L40), [test_unit_config_validation.py](../../tests/benchmarking/test_unit_config_validation.py).
  + **Relação com outras tarefas:** Permite a serialização e validação de pipelines customizados de busca local periódica (`Task 11.9`), pré-requisito para o teste e execução de pré-voo em `Task 12.18`.
  + **How much:** Baixo — ~30 linhas de schemas e testes.
  + **Stakeholder Implications:** Garante integridade estrutural e prevenção de falsos positivos na validação de configurações.
  + **DoD:** Teste em `test_unit_config_validation.py` valida com sucesso `configs/benchmark.json`, `configs/problems.json` e `configs/algorithms.json` contra seus schemas; payload com `PeriodicLocalSearchStep` contendo `re_evaluate` é aprovado sem erro; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithms.schema.json#L266-L269](../../configs/schemas/algorithms.schema.json#L266-L269), [problems.schema.json#L38-L53](../../configs/schemas/problems.schema.json#L38-L53), [test_unit_config_validation.py#L467-L602](../../tests/benchmarking/test_unit_config_validation.py#L467-L602).
    - **Resumo:** Mapeamento formal de `"re_evaluate": { "$ref": "#/$defs/strategy_slot" }` em `pipeline_step.params.properties` de [algorithms.schema.json](../../configs/schemas/algorithms.schema.json), permitindo instanciar `PeriodicLocalSearchStep(target="population")` declarativamente via JSON sem violação de schema. Flexibilização de [problems.schema.json](../../configs/schemas/problems.schema.json) para aceitar os campos de metadados padronizados da TSPLIB (`edge_weight_type`, `comment`, `source`, `problem_type`) sob tipagem estrita de strings, preservando `"additionalProperties": false` contra chaves espúrias. Suíte de testes em [test_unit_config_validation.py](../../tests/benchmarking/test_unit_config_validation.py) ampliada para 66 testes (+11 testes adicionados), cobrindo aceitação de `re_evaluate`, rejeição de tipos inválidos ou propriedades espúrias em pipeline steps e instâncias de catálogo.
    - **Atendimento ao DoD:** `test_unit_config_validation.py` com 66/66 testes aprovados; payloads com `PeriodicLocalSearchStep` contendo `re_evaluate` passam sem `ValidationError`; problemas contendo atributos de metadados são validados com sucesso; `check_docs` retorna exit 0 sem dead links ou símbolos indefinidos.

- [x] **Task 12.18: Parametrização Explícita Completa e Auditoria de Listas de Candidatos (`DP12-11`)**
  + **What:** Parametrização explícita e exaustiva em arquivos de configuração e scripts executores ([run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py)), eliminando valores padrão implícitos (`None`) para parâmetros centrais (incluindo listas de candidatos `candidate_k=20`, `max_iterations=10`, `tournament_k=5`, `seed=42`, `population_size=256`, `mutation_rate=0.02`), e exibindo/registrando explicitamente todos esses parâmetros na telemetria de inicialização e nos relatórios.
  + **Why:** Atualmente `run_chapter4_benchmark.py` possui constantes top-level como `SEED = None`, `POPULATION_SIZE = None`, `MUTATION_RATE = None` e não expõe no cabeçalho ou no log inicial o uso e o tamanho da lista de candidatos (`candidate_k=20`), gerando incerteza sobre se as listas de candidatos estão ativas ou se defaults ocultos estão sendo aplicados.
  + **Who:** Autor e Agente.
  + **Where:** [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py), [configs/algorithms.json](../../configs/algorithms.json), [configs/benchmark.json](../../configs/benchmark.json), [orchestration.py](../../src/benchmarking/runner/orchestration.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com as Tasks 12.14 a 12.17.
  + **How:** Tornar explícitas todas as constantes de configuração no topo de `run_chapter4_benchmark.py` (`CANDIDATE_K = 20`, `LOCAL_SEARCH_MAX_ITERATIONS = 10`, `SELECTION_K = 5`, `SEED = 42`, `POPULATION_SIZE = 256`, `MUTATION_RATE = 0.02`); auditar `configs/algorithms.json` para que cada variante declare explicitamente todos os parâmetros de suas estratégias; enriquecer `log_effective_parameters` para imprimir detalhadamente `candidate_k`, `max_iterations`, `tournament_k`, `threads_per_block` e status das listas de candidatos por variante; adicionar validação em pré-voo.
  + **How much:** Baixo — ~50 linhas de código e documentação.
  + **Stakeholder Implications:** Transparência total e replicabilidade sem ambiguidades de todos os hiperparâmetros do experimento da dissertação.
  + **DoD:** Execução com `--dry-run` de `run_chapter4_benchmark.py` imprime no cabeçalho o valor explícito de cada parâmetro incluindo `candidate_k=20`; nenhuma variável central permanece como `None` implícito; validações de pré-voo conferem exaustivamente a paridade dos valores; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [run_chapter4_benchmark.py#L101-L110](../../benchmark_scripts/run_chapter4_benchmark.py#L101-L110), [run_chapter4_benchmark.py#L230-L270](../../benchmark_scripts/run_chapter4_benchmark.py#L230-L270).
    - **Resumo:** Parametrização explícita de alto nível configurada no topo de `run_chapter4_benchmark.py`: `CANDIDATE_K = 20`, `LOCAL_SEARCH_MAX_ITERATIONS = 10`, `SELECTION_K = 5`, `SEED = 42`, `POPULATION_SIZE = 256`, `MUTATION_RATE = 0.02`, `CPU_SIZE_THRESHOLD = 400` e `REPETITIONS = 30`, eliminando qualquer `None` implícito residual. Cabeçalho de telemetria inicial (`log_effective_parameters`) enriquecido para exibir detalhadamente todas as opções e validar paridade de parâmetros em modo pré-voo.
    - **Atendimento ao DoD:** Execução de `python -m benchmark_scripts.run_chapter4_benchmark --dry-run` imprime todos os parâmetros explicitamente no cabeçalho (incluindo `candidate_k=20` ativo em todos os passos de 2-opt) e passa por todas as validações de pré-voo com exit 0; `check_docs` com exit 0.

- [x] **Task 12.19: Otimizações Algorítmicas Puras e Eliminação de Alocações em 2-Opt CPU (`DP12-12`)**
  + **What:** Refatorar a busca local em CPU ([two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)):
    1. Eliminar o gerador `_iter_moves` que instancia $\approx \frac{n(n-3)}{2}$ tuplas heap `(i, j, delta)` por indivíduo por passagem, incorporando o laço de steepest descent diretamente no método `_step_best` (e de forma análoga em `_step_first` e `_step_multi_move`);
    2. Eliminar a checagem condicional redundante `if i == 0 and j == n - 1:` no laço interno mais profundo, içando o limite superior para `limit_j = n - 1 if i == 0 else n`;
    3. Substituir a aritmética de ponto-flutuante (`float(...)` e `delta < best_delta - 1e-6`) por delta exato em `int32` (`delta = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))` com condição `delta < best_delta` e limiar inicial `0`), espelhando a semântica do kernel CUDA [two_opt_common.cuh](../../src/distances/kernels/two_opt_common.cuh#L61);
    4. Aplicar a mesma eliminação de gerador e aritmética inteira exata no caminho acelerado por listas de candidatos (`candidates is not None`, `DP10-20`).
  + **Why:** O perfil de execução revelou mais de 5.000.000 de alocações de tuplas na heap por geração no 2-opt em CPU, gerando pressão massiva no garbage collector. O cálculo em inteiros com eliminação de gerador zera as alocações na heap por varredura ($\mathcal{O}(1)$ de memória) e garante estrita paridade bit a bit com os kernels CUDA.
  + **Who:** Autor e Agente.
  + **Where:** [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Início do bloco de otimização de CPU da Fase 12.
  + **How:** Reescrever `_step_best`, `_step_first` e `_step_multi_move` para iterar diretamente sobre os índices e atualizar variáveis escalares em registradores; executar suíte unitária de 2-opt.
  + **How much:** Médio — ~80 linhas alteradas em `two_opt.py`.
  + **Stakeholder Implications:** Reduz expressivamente o tempo de execução do 2-opt em CPU sem alterar qualquer resultado ou rota ótima.
  + **DoD:** `pytest tests/algorithms/strategies/local_search -q` 100% verde; rotas resultantes idênticas ao baseline em instâncias de teste com e sem `candidate_k`; zero tuplas instanciadas no laço interno; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [two_opt.py#L236-L378](../../src/algorithms/strategies/local_search/two_opt.py#L236-L378), [test_unit_two_opt_boundaries.py](../../tests/algorithms/strategies/local_search/test_unit_two_opt_boundaries.py).
    - **Resumo:** Refatorados os métodos `_step_best`, `_step_first` e `_step_multi_move` em `two_opt.py`, eliminando por completo o gerador `_iter_moves` e as tuplas heap `(i, j, delta)` por passagem. Içado o limite superior `limit_j = n - 1 if i == 0 else n` eliminando ramificação condicional interna, e adotada aritmética inteira exata `int32` (`delta < 0`), replicada de forma análoga para o caminho acelerado por listas de candidatos (`candidates is not None`).
    - **Atendimento ao DoD:** Execução de `pytest tests/algorithms/strategies/local_search/ -v` aprovada com 50/50 testes verdes; validação de invariantes, rotação de índices, suporte a `candidate_k` e paridade estrita bit-idêntica com os kernels CUDA; `check_docs` com exit 0.

- [x] **Task 12.20: Vetorização com Indexação Achatada e Preservação de Dtype em Avaliação de Custo CPU (`DP12-13`)**
  + **What:** Refatorar a avaliação de custo em lote na CPU ([cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L46-L56)):
    1. Eliminar `np.roll(batch_np, -1, axis=1)`, que aloca uma matriz $(N, n)$ completa a cada geração, substituindo-o por fatiamento direto em views sem cópia (`batch_np[:, :-1]` e `batch_np[:, 1:]`);
    2. Substituir a indexação 2D com stride duplo `distances[i_arr, j_arr]` por indexação 1D em matriz pré-achatada (`flat_dist = distances.ravel()`), avaliando arestas internas via `flat_dist[batch_np[:, :-1] * n + batch_np[:, 1:]]` e somando o fechamento do ciclo `flat_dist[batch_np[:, -1] * n + batch_np[:, 0]]` in-place;
    3. Acumular o custo no tipo de dado nativo da matriz de distâncias (`dtype=distances.dtype`, tipicamente `int32`/`int64`), eliminando o hardcode `dtype=np.float64` que forçava conversão da matriz inteira;
    4. Atualizar o type alias obsoleto `DistanceMatrix` em [types.py](../../src/protocols/types.py#L32) para `NDArray[np.int32]`.
  + **Why:** O fatiamento direto e a indexação 1D em buffer achatado reduzem em 35% o tempo de avaliação em lote (medido de 4.17 ms para 2.69 ms em $N=256, n=500$), eliminam 6 MB de tráfego desnecessário de cache L1/L2 por geração e alinham a tipagem formal com as especificações da TSPLIB95.
  + **Who:** Autor e Agente.
  + **Where:** [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py), [types.py](../../src/protocols/types.py), [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com a Task 12.19.
  + **How:** Modificar `CPUCostEvaluation.evaluate_batch` para usar a indexação achatada; manter compatibilidade com matrizes de distância 2D pré-existentes; atualizar asserções de tipo.
  + **How much:** Baixo — ~25 linhas alteradas.
  + **Stakeholder Implications:** Avaliação de aptidão em CPU mais rápida e conservativa em largura de banda de memória.
  + **DoD:** `pytest tests/algorithms/strategies/evaluation -q` verde; custos calculados idênticos ao método com `np.roll`; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cost_evaluation.py#L50-L57](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L50-L57), [types.py#L32-L33](../../src/protocols/types.py#L32-L33), [test_unit_cost_evaluation.py#L82-L95](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py#L82-L95).
    - **Resumo:** `EvaluationCPUCost.evaluate_batch` refatorado para eliminar a matriz intermediária de `np.roll`, utilizando fatiamento direto por views (`batch_np[:, :-1]` e `batch_np[:, 1:]`) e indexação linear 1D em buffer achatado (`flat_dist = distances.ravel()`). A soma acumula no dtype nativo da matriz de distâncias (`dtype=distances.dtype`) e o alias `DistanceMatrix` em `types.py` foi atualizado para `NDArray[np.int32]`.
    - **Atendimento ao DoD:** 26/26 testes em `tests/algorithms/strategies/evaluation/` aprovados, validando cálculo idêntico de custo de rotas 1D e lotes 2D em relação aos valores de referência; `check_docs` com exit 0.

- [x] **Task 12.21: Vetorização sem Coleções python em Cruzamento OX/PMX e Mutação em CPU (`DP12-14`)**
  + **What:** Otimizar operadores genéticos em CPU:
    1. Em [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py#L68-L76), substituir `set(p1[c1:c2+1].tolist())` e laço escalar com `in` por uma tabela de pertinência booleana plana `in_seg = np.zeros(n, dtype=bool); in_seg[p1[c1:c2+1]] = True` e extração vetorizada via NumPy, preservando nós de depósito em rotas fechadas (`stream_start = 1 if is_closed else 0`);
    2. Em [pmx_crossover.py](../../src/algorithms/strategies/crossover/pmx_crossover.py#L56-L71), substituir a criação dinâmica de `p2_pos: dict` em cada cruzamento por array direto de posições inversas `inv_p2[val] = idx` e preenchimento de posições não mapeadas via máscara booleana (`child == -1`);
    3. Em [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py#L78), eliminar a clonagem redundante `out = batch_np.copy()` passando `copy=False` para `to_cpu_batch`, e reverter segmentos diretamente in-place sem alocação de buffer intermediário.
  + **Why:** Elimina overhead de criação de tabelas hash/dicionários e conversões de lista em python durante a recombinação, proporcionando aceleração medida de até $11.66\times$ no Order Crossover e $2.31\times$ no PMX em $n=500$, mantendo rigorosa invariância de permutação e suporte a depósito fechado.
  + **Who:** Autor e Agente.
  + **Where:** [order_crossover.py](../../src/algorithms/strategies/crossover/order_crossover.py), [pmx_crossover.py](../../src/algorithms/strategies/crossover/pmx_crossover.py), [inversion_mutation.py](../../src/algorithms/strategies/mutation/inversion_mutation.py), [test_unit_order_crossover.py](../../tests/algorithms/strategies/crossover/test_unit_order_crossover.py), [test_unit_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com as Tasks 12.19 e 12.20.
  + **How:** Atualizar `_apply_ox`, `_pmx_core` e `mutate_batch`; validar paridade com testes existentes.
  + **How much:** Médio — ~60 linhas alteradas em 3 arquivos.
  + **Stakeholder Implications:** Crossover e mutação em CPU significativamente mais velozes sem risco de quebra de invariantes.
  + **DoD:** `pytest tests/algorithms/strategies/crossover tests/algorithms/strategies/mutation -q` 100% verde; verificação de permutações válidas em rotas abertas e fechadas; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [order_crossover.py#L57-L84](../../src/algorithms/strategies/crossover/order_crossover.py#L57-L84), [pmx_crossover.py#L56-L75](../../src/algorithms/strategies/crossover/pmx_crossover.py#L56-L75), [inversion_mutation.py#L78-L99](../../src/algorithms/strategies/mutation/inversion_mutation.py#L78-L99), [test_unit_order_crossover.py](../../tests/algorithms/strategies/crossover/test_unit_order_crossover.py), [test_unit_inversion_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py).
    - **Resumo:** Vetorizados operadores genéticos em CPU: em `order_crossover.py`, removido `set` e laço escalar em favor de tabela booleana `in_seg` e rotação circular vetorizada; em `pmx_crossover.py`, substituído dicionário por array de posições inversas `inv_p2` e preenchimento por máscara booleana `unmapped`; em `inversion_mutation.py`, evitada cópia duplicada com `copy=False` em `to_cpu_batch` e mutação in-place sobre array CPU.
    - **Atendimento ao DoD:** 151 testes combinados em crossover (41) e mutação (110) aprovados com 100% de sucesso, garantindo paridade bit-idêntica com kernels GPU CUDA e preservação exata de nós de depósito; `check_docs` com exit 0.

- [x] **Task 12.22: Harmonização de Dtypes em Avaliação de Custo e Paridade de Tipos (`Fitness`) (`DP12-15`, `BUG-40`)**
  + **What:** Revisar e alinhar os tipos de dados retornados pelos operadores de avaliação de custo ([cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py)). Forçar `dtype=np.float64` na soma de `EvaluationCPUCost.evaluate_batch` para que o retorno coincida rigorosamente com o tipo formal `Fitness = NDArray[np.float64]` ([types.py](../../src/protocols/types.py#L29)) e mantenha invariância exata com `EvaluationGPUCost` (que sempre retorna `float64`), prevenindo que matrizes inteiras de distâncias retornem `int32`. Simultaneamente, eliminar a clonagem dupla redundante em [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py) passando `to_cpu_batch(routes, copy=False)` e condicionando `out = batch_np if was_gpu else batch_np.copy()`, espelhando a otimização de `inversion_mutation.py`.
  + **Why:** A divergência de dtype entre CPU (`int32`) e GPU (`float64`) quando avaliando instâncias TSPLIB com distâncias inteiras gera inconsitência de tipos em tempo de execução mascarada por `cast()`. O alinhamento garante invariância de backend (`EVAL-A`), aderência estrita ao mypy e elimina alocação redundante em mutação.
  + **Who:** Autor e Agente.
  + **Where:** [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py), [types.py](../../src/protocols/types.py), [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Imediatamente antes do benchmark integral.
  + **How:** Modificar a chamada `.sum(axis=1, dtype=np.float64)` em `EvaluationCPUCost.evaluate_batch`; alinhar `swap_mutation.py` para evitar double-copy; verificar conformidade de tipo e testes unitários.
  + **Blast Radius:** [cost_evaluation.py](../../src/algorithms/strategies/evaluation/cost_evaluation.py) (L54-L56), [swap_mutation.py](../../src/algorithms/strategies/mutation/swap_mutation.py) (L77-L85), [test_unit_cost_evaluation.py](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py).
  + **Relação com outras tarefas:** Complementa a vetorização de `Task 12.20` e a semântica de cópia de `Task 12.21`, preparando dados consistentes para o runner (`Task 12.23`).
  + **How much:** Baixo — ~20 linhas alteradas em código de produção e testes.
  + **Stakeholder Implications:** Consistência matemática estrita e paridade entre avaliadores CPU e GPU.
  + **DoD:** `pytest tests/algorithms/strategies/evaluation tests/algorithms/strategies/mutation -q` 100% verde; `EvaluationCPUCost.evaluate_batch` retorna array de `float64` mesmo sob matrizes inteiras; `swap_mutation` não realiza clone redundante; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [cost_evaluation.py#L54](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L54), [swap_mutation.py#L77-L99](../../src/algorithms/strategies/mutation/swap_mutation.py#L77-L99), [test_unit_cost_evaluation.py#L77-L154](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation.py#L77-L154), [test_unit_swap_mutation.py#L173-L194](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L173-L194).
    - **Resumo:** Harmonizados dtypes em `EvaluationCPUCost.evaluate_batch` fixando `.sum(axis=1, dtype=np.float64)` para que o retorno coincida estritamente com `Fitness = NDArray[np.float64]` tanto em matrizes inteiras (`int32`, `int64`) quanto flutuantes (`float32`, `float64`), rotas 1D e lotes vazios ($N=0$), além de alinhar a docstring de `test_cpu_cost_evaluation_batch_matches_single_tours`. Em `MutationSwap.mutate_batch`, eliminada a cópia redundante em lote host com `to_cpu_batch(routes, copy=False)` e `out = batch_np if was_gpu else batch_np.copy()`, garantindo mutação isolada com zero alocação intermediária duplicada.
    - **Atendimento ao DoD:** 141 testes em `tests/algorithms/strategies/evaluation` e `tests/algorithms/strategies/mutation` aprovados (`2.65s`); novo teste parametrizado `test_cpu_cost_evaluation_returns_float64_fitness_across_dtypes_and_shapes` cobre todas as combinações de dtypes de matriz e formas; `test_swap_mutation_does_not_mutate_caller_array_in_place` comprova que o array do chamador não é mutado in-place; `check_docs` com exit 0.

- [x] **Task 12.23: Eliminação de Boilerplate em Runner de Telemetria (`algorithm_runner.py`) e Refatoração DRY (`DP12-16`)**
  + **What:** Refatorar o método `_build_summary_payload` em [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py) extraindo a função utilitária pura `_compute_series_stats(arr: np.ndarray, n_valid: int, ddof: int) -> tuple[float, float, float, float]` para calcular `(mean, std, min, max)` centralizadamente, eliminando 18 guardas ternárias repetidas (`if total_successful > 0 else 0.0`). Consolidar o tratamento de parâmetros e queries de problemas compartilhadas entre [orchestration.py](../../src/benchmarking/runner/orchestration.py) e [report_generator.py](../../src/benchmarking/analysis/report_generator.py).
  + **Why:** Reduz ~40 linhas de código boilerplate repetitivo em `_build_summary_payload`, previne omissão de guardas em campos futuros e mantém estritamente o isolamento de timing fora do laço (`C1`) e a correção de Bessel ($ddof=1$).
  + **Who:** Autor e Agente.
  + **Where:** [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py), [orchestration.py](../../src/benchmarking/runner/orchestration.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Concomitante com a Task 12.16.
  + **How:** Substituir blocos ternários por chamadas a `_compute_series_stats`; assegurar que o schema resultante em `summary` seja idêntico ao exigido pelo DuckDB e pelos geradores de tabela; rodar suíte de testes do runner.
  + **Blast Radius:** [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py) (L74-L208), [orchestration.py](../../src/benchmarking/runner/orchestration.py) (L255-L275, L645-L660).
  + **Relação com outras tarefas:** Alimenta as estruturas consumidas por `report_generator.py` (`Task 12.16`) e validadas por `Task 12.24`.
  + **How much:** Médio — ~50 linhas refatoradas.
  + **Stakeholder Implications:** Código limpo, manutenível e auditável na camada de orquestração do benchmark.
  + **DoD:** `pytest tests/benchmarking/test_unit_algorithm_runner.py` e testes de pipeline verdes; carga no DuckDB intacta; zero impacto no tempo medido em `perf_counter` (`C1`); `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithm_runner.py#L74-L90](../../src/benchmarking/runner/algorithm_runner.py#L74-L90), [orchestration.py#L55-L58](../../src/benchmarking/runner/orchestration.py#L55-L58), [orchestration.py#L194-L220](../../src/benchmarking/runner/orchestration.py#L194-L220), [report_generator.py#L44-L630](../../src/benchmarking/analysis/report_generator.py#L44-L630), [test_unit_runner_telemetry.py#L238-L354](../../tests/benchmarking/test_unit_runner_telemetry.py#L238-L354), [test_unit_report_generator.py#L379-L429](../../tests/benchmarking/test_unit_report_generator.py#L379-L429).
    - **Resumo:** Refatorado `algorithm_runner.py` eliminando guardas ternárias repetidas via helper puro compartilhado `compute_series_stats` (incluindo `mean_improvement` e guardas contra fatias vazias/all-NaN, com correção de Bessel $ddof=1$ para $N > 1$ não-NaN, sem alias legado), além de `_initialize_results_dict` e `_record_failure`. Em `orchestration.py`, `_save_problem_statistics_file` foi consolidado para importar e reutilizar `compute_series_stats` nas 5 séries (`times`, `initial_costs`, `costs`, `gaps`, `generations`), eliminando duplicação entre orquestração e runner e prevenindo emissão de `NaN` quando campos opcionais estiverem ausentes (`BUG-43` restaurado na `Task 12.27`). Em `report_generator.py`, fatoradas `_compute_table2_stats`, `_compute_table3_size_category_winners` e a tupla canônica `_PHASE_NAMES`.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `343 passed, 1 deselected in 11.82s` (1 teste GPU desmarcado fora do escopo CPU por benchmark ativo); `grep -rnE --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache,.git,.venv} "(^|[^a-zA-Z0-9_])_compute_series_stats\b" src tests benchmark_scripts` -> `ZERO_MATCHES`; `mypy` e `ruff` verdes; `check_docs` exit `0`.

- [x] **Task 12.24: Suíte de Testes Comportamentais e Resiliência a Imprevistos em Benchmarking (`DP12-17`)**
  + **What:** Implementar `tests/benchmarking/test_unit_report_generator_behavioral.py` para testar propriedades e comportamentos do sistema frente a condições anômalas ou incomuns, superando testes superficiais com constantes mágicas rígidas ([test_integration_pipeline.py:175-176](../../tests/benchmarking/test_integration_pipeline.py#L175-L176)):
    1. Avaliação com variantes algorítmicas de nomes arbitrários e exóticos (ex.: `"CrazyCpuGpuLocalSearchEveryStep"`, strings com hifens/underscores);
    2. Instâncias onde a linha de base padrão foi desativada por teto de porte ($n > 400$) ou `--skip-cpu`, verificando que a substituição de referência local com adaga (`1.00x†` / `1.00\times^{\dagger}`) comissionada pelo usuário é ativada corretamente;
    3. Cenários de empate absoluto de rankings (variância zero) no teste omnibus de Friedman, garantindo que nenhum erro de divisão por zero ou warning de `invalid value` ocorra;
    4. Coortes com execução única ($N=1$), confirmando que desvios padrão retornam `0.0` sem erro de Bessel;
    5. Matrizes irregulares onde algoritmos diferentes foram executados sobre subconjuntos distintos de problemas.
  + **Why:** Protege o pipeline de execução da dissertação contra falhas em tempo de execução causadas por dados reais com formatações ou distribuições imprevistas.
  + **Who:** Autor e Agente.
  + **Where:** [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py), [test_integration_pipeline.py](../../tests/benchmarking/test_integration_pipeline.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Após as Tasks 12.16, 12.22 e 12.23.
  + **How:** Escrever testes parametrizados com `pytest` simulando payloads heterogêneos; verificar conformidade de formatação em Markdown e LaTeX.
  + **Blast Radius:** `tests/benchmarking/test_unit_report_generator_behavioral.py` (novo arquivo ~150 linhas), [test_integration_pipeline.py](../../tests/benchmarking/test_integration_pipeline.py).
  + **Relação com outras tarefas:** Homologa as tarefas 12.16, 12.17, 12.22 e 12.23 antes da coleta final em `Task 12.4`.
  + **How much:** Médio — novo arquivo com ~150 linhas de testes estruturados.
  + **Stakeholder Implications:** Confiança absoluta na estabilidade do pipeline de relatórios para a defesa da dissertação.
  + **DoD:** Novo arquivo de teste criado e 100% verde; cobre 5 casos comportamentais não testados anteriormente; `pytest tests/benchmarking` verde; `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py), [test_integration_pipeline.py#L170-L188](../../tests/benchmarking/test_integration_pipeline.py#L170-L188), [statistics.py#L358-L398](../../src/benchmarking/analysis/statistics.py#L358-L398), [problem_statistics.py#L79-L132](../../src/benchmarking/analysis/problem_statistics.py#L79-L132).
    - **Resumo:** Criado [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py) cobrindo os 5 cenários comportamentais anômalos, além de resolução preguiçosa e escape LaTeX (7 testes no total após `Task 12.25`–`12.29`), fortalecidas as asserções de invariantes de speedup em [test_integration_pipeline.py#L170-L188](../../tests/benchmarking/test_integration_pipeline.py#L170-L188), endurecida `friedman_test` em [statistics.py#L358-L398](../../src/benchmarking/analysis/statistics.py#L358-L398) e corrigida resolução de métricas em [problem_statistics.py#L79-L132](../../src/benchmarking/analysis/problem_statistics.py#L79-L132).
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13 confirmando resolução integral de `BUG-41`–`BUG-45`; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `343 passed, 1 deselected in 11.82s`; `mypy` e `ruff` exit `0`; `check_docs` exit `0`.

- [x] **Task 12.25: Invariância à Escala, Rejeição de Não-Finitos e Empates Intra-Bloco em `friedman_test` (`DP12-18`, `BUG-41`)**
  + **What:** Corrigir `friedman_test` em [statistics.py](../../src/benchmarking/analysis/statistics.py#L358-L398) para tratar empates absolutos intra-bloco sem emissão de `RuntimeWarning` de divisão `0/0` (`NaN`) no fator de correção do `stats.friedmanchisquare`, eliminando qualquer limiar absoluto dependente de escala (`variance_threshold = 1e-10`, que colapsava dados legítimos na escala $10^{-11}$ de $p=0{,}0498$ para $p=1{,}0$) e qualquer supressão global de `RuntimeWarning` que mascarasse entradas `NaN`/`Inf`.
  + **Why:** O teste de Friedman é estritamente invariante por transformações monótonas crescentes (posto ordinal dentro de cada bloco). Um corte absoluto em variância falsifica o teste em séries com escala numérica reduzida, enquanto `warnings.catch_warnings` oculta dados corrompidos (`NaN`/`Inf`) sob aparência de empate neutro (`p=1.0`).
  + **Who:** Agente (implementação via `4eb81a31` + homologação via Review Gate `22c181dd`).
  + **Where:** [statistics.py](../../src/benchmarking/analysis/statistics.py#L358-L398), [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L179-L242), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Descoberta durante a Task 12.24 e refinada após Review Gate `22c181dd`; corrigida no workspace aguardando re-review gate.
  + **How:** Validar `np.all(np.isfinite(data_matrix))` levantando `ValueError` em `NaN`/`Inf`; detectar empate total intra-bloco de forma invariante à escala via `np.all(np.ptp(data_matrix, axis=1) == 0.0)` retornando `{statistic: 0.0, p_value: 1.0, significant: False, post_hoc_required: False}` sem silenciar warnings do SciPy; adicionar caso de teste com escala $10^{-11}$ confirmando detecção de significância ($\chi_F^2 = 6{,}0, p < 0{,}05$) e rejeição de não-finitos sob `-W error`.
  + **Blast Radius:** [statistics.py#L358-L398](../../src/benchmarking/analysis/statistics.py#L358-L398), [test_unit_report_generator_behavioral.py#L179-L242](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L179-L242), [problem_statistics.py#L171-L212](../../src/benchmarking/analysis/problem_statistics.py#L171-L212) (consumidor de `friedman_test`).
  + **Relação com outras tarefas:** Estende a guarda de `BUG-33` (`Task 12.7`), corrige defeito descoberto durante `Task 12.24` (`DP12-17`) e alimenta as análises da `Task 12.4`.
  + **How much:** Baixo — ~25 linhas em `statistics.py` + casos de invariância e rejeição em testes.
  + **Stakeholder Implications:** Rigor estatístico não-paramétrico preservado em qualquer unidade de custo e detecção imediata de corrupção numérica (`NaN`/`Inf`).
  + **DoD:** `friedman_test` retorna `statistic=0.0, p_value=1.0` em empates exatos intra-bloco sem `RuntimeWarning`, preserva $\chi_F^2 = 6{,}0$ na escala $10^{-11}$ e levanta `ValueError` para `NaN`/`Inf`; `pytest -W error` verde; Review Gate `PASS` e `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [statistics.py#L358-L398](../../src/benchmarking/analysis/statistics.py#L358-L398), [test_unit_report_generator_behavioral.py#L179-L242](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L179-L242).
    - **Resumo:** Adicionada validação prévia `np.all(np.isfinite(data_matrix))` levantando `ValueError` em `NaN`/`Inf` (`L374-L375`) e detecção invariante à escala de empates exatos intra-bloco via `np.all(np.ptp(data_matrix, axis=1) == 0.0)` (`L381`), removendo `variance_threshold = 1e-10` e supressão de `RuntimeWarning`. Coberto em `test_behavioral_case3_friedman_absolute_ranking_ties_zero_variance` inclusive na escala $10^{-11}$.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `mypy` e `ruff` exit `0`; `check_docs` exit `0`.

- [x] **Task 12.26: Resolução Preguiçosa e Estrita de Métricas em `perform_statistical_analysis` (`DP12-19`, `BUG-42`)**
  + **What:** Corrigir `perform_statistical_analysis` em [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L79-L132) (e chamadas em [#L395-L423](../../src/benchmarking/analysis/problem_statistics.py#L395-L423)) para: (1) eliminar `KeyError` quando payloads brutos (ex.: coortes $N=1$) omitem chaves pré-computadas `"mean_cost"`, `"mean_time"` ou `"mean_gap"`; (2) eliminar avaliação ansiosa de `np.nanmean(...)` dentro de argumento padrão de `dict.get(...)` (que disparava `RuntimeWarning: Mean of empty slice` mesmo quando `"mean_*"` estava presente com `raw_costs=[np.nan]`); e (3) interromper a fabricação silenciosa de `0.0`/`1.0` quando tanto a métrica quanto os arrays brutos estão ausentes.
  + **Why:** Avaliar `np.nanmean` ansiosamente gera falsos warnings sob `-W error` e falha em pipelines estritos, enquanto fabricar `0.0` custo ou `1.0x` speedup diante de chaves erradas (`"costs"`) ou dicionários vazios mascara silenciosamente falhas graves de entrada nos relatórios da dissertação.
  + **Who:** Agente (implementação via `4eb81a31` + homologação via Review Gate `22c181dd`).
  + **Where:** [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L79-L132), [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L404-L445), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Descoberta durante a Task 12.24 e refinada após Review Gate `22c181dd`; corrigida no workspace aguardando re-review gate.
  + **How:** Extrair `_resolve_metric_mean` e `_resolve_gap_mean` com avaliação estritamente preguiçosa (checar `val = results_dict.get(metric_key)` primeiro; somente se `val is None`, filtrar `np.isfinite` sobre `raw_series` e computar média; caso contrário levantar `KeyError` explícito); cobrir com `test_behavioral_lazy_metric_resolution_and_validation` sob `-W error` (com `"mean_cost"` + `raw_costs=[np.nan]` sem warning, e payload vazio levantando `KeyError`).
  + **Blast Radius:** [problem_statistics.py#L79-L132](../../src/benchmarking/analysis/problem_statistics.py#L79-L132), [problem_statistics.py#L395-L423](../../src/benchmarking/analysis/problem_statistics.py#L395-L423), [test_unit_report_generator_behavioral.py#L404-L445](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L404-L445).
  + **Relação com outras tarefas:** Resolve defeito de resolução de payload exposto durante `Task 12.24` (`DP12-17`), protege o contrato de consumo da `Task 12.23` (`DP12-16`) e garante falhas explícitas antes da `Task 12.4`.
  + **How much:** Baixo — ~55 linhas em helpers dedicados + teste dedicado.
  + **Stakeholder Implications:** Impede geração silenciosa de estatísticas espúrias (`0.00` de custo / `1.00x` de speedup) em caso de falha ou inconsistência de esquema nos checkpoints.
  + **DoD:** Zero warnings com `-W error` quando `"mean_*"` coexiste com `raw_*=[np.nan]`; `KeyError` levantado quando métrica e série bruta faltam; coortes $N=1$ com apenas `raw_*` calculam médias exatas sem erro; Review Gate `PASS` e `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_statistics.py#L79-L132](../../src/benchmarking/analysis/problem_statistics.py#L79-L132), [problem_statistics.py#L397-L422](../../src/benchmarking/analysis/problem_statistics.py#L397-L422), [test_unit_report_generator_behavioral.py#L404-L469](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L404-L469).
    - **Resumo:** Extraídos os helpers `_resolve_metric_mean` e `_resolve_gap_mean` avaliando a chave pré-computada primeiro antes de tocar arrays brutos (`arr[np.isfinite(arr)]`), levantando `KeyError` quando métrica escalar e série bruta finita estão ausentes. Adicionado `test_behavioral_lazy_metric_resolution_and_validation` cobrindo payload com `raw_costs=[np.nan]` sem warning e payload vazio levantando `KeyError`.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `mypy` e `ruff` exit `0`; `check_docs` exit `0`.

- [x] **Task 12.27: Restauração do Teste de Telemetria de `orchestration.py` e Cobertura de Borda em `compute_series_stats` (`DP12-20`, `BUG-43`)**
  + **What:** Restaurar em [test_unit_runner_telemetry.py](../../tests/benchmarking/test_unit_runner_telemetry.py#L238-L354) o teste `test_save_problem_statistics_file_uses_compute_series_stats` (cobrindo `_save_problem_statistics_file` em [orchestration.py](../../src/benchmarking/runner/orchestration.py#L167-L232)) e os casos de borda all-NaN / único finito `[np.nan, 42.0]` em `test_runner_helpers_dry_refactoring`, removidos acidentalmente durante a primeira passagem da Task 12.24, mantendo `compute_series_stats` público e sem alias legado em [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L74-L90).
  + **Why:** A remoção silenciosa reduziu a suíte para 341 testes, deixou `orchestration.py:_save_problem_statistics_file` sem prova de regressão para checkpoints legados com `raw_initial_costs=[]` (que antes emitiam `NaN` no JSON) e quebrou os vínculos de linha de `Task 12.23`.
  + **Who:** Agente (restauração via `4eb81a31` + homologação via Review Gate `22c181dd`).
  + **Where:** [test_unit_runner_telemetry.py](../../tests/benchmarking/test_unit_runner_telemetry.py#L238-L354), [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L74-L90), [orchestration.py](../../src/benchmarking/runner/orchestration.py#L167-L232), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Bloqueador #1 apontado pelo Review Gate `22c181dd` sobre `Task 12.23` e `Task 12.24`; restaurado no workspace aguardando re-review gate.
  + **How:** Garantir que `compute_series_stats` permaneça importado e testado diretamente tanto em `test_runner_helpers_dry_refactoring` (com `[]`, `[np.nan, np.nan]`, `[np.nan, 42.0]`, `N=1`, `N=3`) quanto via `test_save_problem_statistics_file_uses_compute_series_stats(tmp_path)` verificando arquivo JSON gravado via `CheckpointManager`; alinhar âncoras em `Task 12.23`.
  + **Blast Radius:** [test_unit_runner_telemetry.py#L19-L25](../../tests/benchmarking/test_unit_runner_telemetry.py#L19-L25), [test_unit_runner_telemetry.py#L238-L354](../../tests/benchmarking/test_unit_runner_telemetry.py#L238-L354), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md#L2170-L2186).
  + **Relação com outras tarefas:** Restabelece o DoD e a cobertura da `Task 12.23` (`DP12-16`) que havia regredido durante `Task 12.24`.
  + **How much:** Trivial/Baixo — ~65 linhas de testes unitários e validação de âncoras.
  + **Stakeholder Implications:** Assegura que consolidações DRY entre `algorithm_runner.py` e `orchestration.py` não regridam e que arquivos `*_stats.json` permaneçam JSON válidos (sem `NaN`).
  + **DoD:** Ambos os testes `test_runner_helpers_dry_refactoring` e `test_save_problem_statistics_file_uses_compute_series_stats` presentes e verdes sob `-W error`; zero ocorrências do símbolo legado `_compute_series_stats` em `src/` e `tests/`; Review Gate `PASS` e `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_runner_telemetry.py#L238-L354](../../tests/benchmarking/test_unit_runner_telemetry.py#L238-L354), [algorithm_runner.py#L74-L90](../../src/benchmarking/runner/algorithm_runner.py#L74-L90), [orchestration.py#L203-L207](../../src/benchmarking/runner/orchestration.py#L203-L207).
    - **Resumo:** Restaurado integralmente `test_save_problem_statistics_file_uses_compute_series_stats` validando persistência JSON via `CheckpointManager` com `raw_initial_costs=[]` sem emissão de `NaN`, além dos casos de borda all-NaN e `[np.nan, 42.0]` em `test_runner_helpers_dry_refactoring`, mantendo apenas o símbolo público `compute_series_stats`.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `grep -rnE --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache,.git,.venv} "(^|[^a-zA-Z0-9_])_compute_series_stats\b" src tests benchmark_scripts` -> `ZERO_MATCHES`; `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.28: Correção de Chaves de Checkpoint e Endurecimento de Asserções Markdown/LaTeX em Testes Comportamentais (`DP12-21`, `BUG-44`)**
  + **What:** Corrigir os defeitos nos próprios testes comportamentais em [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L113-L403): (1) substituir chaves inválidas `"costs"`/`"times"` por `"raw_costs"`/`"raw_times"` (mais `"optimal_cost"`) nos Casos 3 e 4 para que `_get_data_field` ([problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L73-L76)) não retorne listas vazias e realmente exercite `friedman_test` e `paired_comparison`; (2) substituir asserções de substring `"1.00x"`/`"5.00x"` no Caso 2 por verificações estritas por linha/célula provando ausência de `†`/`^{\dagger}` em `small_50` e presença em instâncias grandes em `.md` e `.tex`; (3) validar `.tex` e `std == 0.00` no Caso 4 ($N=1$); e (4) validar `'—'` na linha `Disjoint` ($|P_\cap|=0$) da Tabela 2 no Caso 5 em `.md` e `.tex`.
  + **Why:** Sem usar `"raw_costs"`/`"raw_times"`, os Casos 3 e 4 passavam vacuamente pulando `friedman_test` (`min_cost_len == 0 < 2`) e executando `paired_comparison` sobre arrays vazios (`mean_difference == 0.0` em vez de `10.0`), enquanto `"1.00x" in content` casava falsamente `"1.00x†"`.
  + **Who:** Agente (correção via `4eb81a31` + homologação via Review Gate `22c181dd`).
  + **Where:** [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L113-L403), [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L65-L77), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Bloqueadores #2 e #5 apontados pelo Review Gate `22c181dd` sobre `Task 12.24`; corrigido no workspace aguardando re-review gate.
  + **How:** Atualizar os fixtures e asserções dos Casos 2, 3, 4 e 5 em `test_unit_report_generator_behavioral.py`, exigindo `pairwise_comp["mean_difference"] == pytest.approx(10.0)` no Caso 4, linha-a-linha em Markdown/LaTeX para adaga no Caso 2, `std == 0.00` em `.tex` no Caso 4, e `'—'` nas células disjuntas da Tabela 2 em `.md` e `.tex` no Caso 5; rodar com `pytest -W error`.
  + **Blast Radius:** [test_unit_report_generator_behavioral.py#L113-L403](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L113-L403).
  + **Relação com outras tarefas:** Fecha as lacunas de qualidade da `Task 12.24` (`DP12-17`) e comprova efetivamente as correções de `Task 12.25` (`BUG-41`) e `Task 12.26` (`BUG-42`).
  + **How much:** Baixo — ~60 linhas ajustadas no arquivo de testes comportamentais.
  + **Stakeholder Implications:** Elimina falsos positivos na suíte comportamental que homologa os relatórios finais da dissertação.
  + **DoD:** Todos os 6 testes em `test_unit_report_generator_behavioral.py` passam sob `pytest -W error` exercitando efetivamente `friedman_test`, `paired_comparison` (`mean_difference == 10.0`), separação `.md`/`.tex` de `†`, `std == 0.00` e `'—'`; Review Gate `PASS` e `check_docs` exit 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_unit_report_generator_behavioral.py#L40-L403](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L40-L403).
    - **Resumo:** Substituídas chaves `"costs"`/`"times"` por `"raw_costs"`/`"raw_times"` nos Casos 3 e 4 acionando `friedman_test` e `paired_comparison` (`mean_difference == pytest.approx(10.0)`), endurecidas asserções célula-a-célula para `†`/`^{\dagger}` no Caso 2, validados `std == 0.00` em `.tex` no Caso 4 e `'—'` em `.md`/`.tex` no Caso 5.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13; `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s` (todos os 7 testes comportamentais incluídos); `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.29: Escapamento LaTeX de Caracteres Especiais (`_`, `%`) em Tabelas `.tex` de `report_generator.py` (`DP12-22`, `BUG-45`)**
  + **What:** Introduzir helper dedicado `escape_latex_identifier(name: str) -> str` (escapando `_` para `\_` e `%` para `\%`, reutilizado também em [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L225)) e aplicá-lo nos identificadores de problema e algoritmo emitidos pelas funções LaTeX de [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L60-L66) (`_generate_table1_latex`, `_generate_table2_latex`, `_generate_table3_latex`, `_generate_table4_latex`), atualizando as asserções `.tex` afetadas nos testes unitários.
  + **Why:** Identificado como achado menor (MINOR) pelo Review Gate `22c181dd`: hoje `report_generator.py` emite nomes exóticos ou com sublinhado diretamente em tabelas `.tex` (ex.: `2-Opt_GPU`, `GA-Custom_v2`, `CustomGA_Alpha`), causando falha de compilação LaTeX (`Missing $ inserted`) ao incluir a tabela gerada em documentos compilados com LuaLaTeX.
  + **Who:** Agente (`4eb81a31` + homologação via Review Gate `22c181dd`).
  + **Where:** [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L60-L66), [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L225), [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py), [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Aberta após decisão de não alterar contratos existentes durante a correção emergencial da `Task 12.24` (`22c181dd`); executar antes da geração final das tabelas da dissertação (`Task 12.4`).
  + **How:** Extrair helper reutilizável `escape_latex_identifier` compartilhado entre `report_generator.py` e `cd_diagram.py#L225`, aplicá-lo apenas nos campos textuais de tabelas `.tex` (sem afetar `.md` nem símbolos matemáticos gerados pelo próprio formatador como `$\times$`), e atualizar testes unitários e comportamentais para validar nomes escapados (`2-Opt\_GPU`, `CustomGA\_Alpha`) em `.tex` enquanto preservam nomes crus em `.md`.
  + **Blast Radius:** [report_generator.py#L60-L66](../../src/benchmarking/analysis/report_generator.py#L60-L66), [report_generator.py#L204-L265](../../src/benchmarking/analysis/report_generator.py#L204-L265), [report_generator.py#L501-L511](../../src/benchmarking/analysis/report_generator.py#L501-L511), [cd_diagram.py#L225](../../src/benchmarking/analysis/cd_diagram.py#L225), [tests/benchmarking/test_unit_report_generator_behavioral.py#L472-L584](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L472-L584).
  + **Relação com outras tarefas:** Completa o suporte completo a nomes arbitrários introduzido em `Task 12.16` (`DP12-9`) e testado em `Task 12.24` (`DP12-17`) antes da coleta e compilação da dissertação em `Task 12.4` / `Task 12.5`.
  + **How much:** Baixo — ~25 linhas de helper + chamadas em 4 tabelas LaTeX + atualização pontual de asserções `.tex` em testes.
  + **Stakeholder Implications:** Garante que tabelas `.tex` geradas para variantes com `_` ou `%` no nome compilem diretamente sem edição manual no LaTeX da dissertação.
  + **DoD:** `.tex` gerado com nomes contendo `_` e `%` devidamente escapados (`\_`, `\%`) em todas as Tabelas 1–4 enquanto `.md` preserva nomes originais; `pytest tests/benchmarking/ -q`, `mypy`, `ruff` e `check_docs` verdes (exit 0).
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py#L60-L66](../../src/benchmarking/analysis/report_generator.py#L60-L66), [report_generator.py#L204-L265](../../src/benchmarking/analysis/report_generator.py#L204-L265), [report_generator.py#L501-L511](../../src/benchmarking/analysis/report_generator.py#L501-L511), [cd_diagram.py#L21-L22](../../src/benchmarking/analysis/cd_diagram.py#L21-L22), [cd_diagram.py#L225](../../src/benchmarking/analysis/cd_diagram.py#L225), [test_unit_report_generator_behavioral.py#L472-L584](../../tests/benchmarking/test_unit_report_generator_behavioral.py#L472-L584).
    - **Resumo:** Criado helper público compartilhado `escape_latex_identifier(name: str) -> str` em `report_generator.py#L60-L66` e reutilizado em `cd_diagram.py#L225`, escapando `_` e `%` nas Tabelas 1–4 `.tex` (nomes de problemas, colunas/linhas de algoritmos, legendas/notas e listas truncadas da Tabela 3 — realizando o corte de 60 chars antes de escapar para não seccionar sequências `\_`/`\%` e removendo reatribuição redundante de `safe_baseline`) enquanto preserva nomes crus no `.md`. Coberto por `test_behavioral_latex_escaping_across_all_four_tables` e casos 1–2 em `test_unit_report_generator_behavioral.py`.
    - **Atendimento ao DoD:** Re-Review Gate (`22c181dd`) `PASS` em 2026-11-13 + validação pós-refino em `report_generator.py`: `pytest -W error tests/benchmarking/test_unit_report_generator_behavioral.py tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_runner_telemetry.py -v` -> `21 passed in 2.14s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `343 passed, 1 deselected in 11.82s`; `mypy src tests benchmark_scripts` -> `Success: no issues found in 147 source files`; `ruff check src tests benchmark_scripts` -> `All checks passed!`; `check_docs` exit `0`.

#### Plano de Lotes — Remediação Pós-Review Gate (`Task 12.23`–`12.29`) — CONCLUÍDO

1. **Lote 1 — Hardening Estatístico, Telemetria DRY e Asserções Comportamentais (`Task 12.23`, `12.25`, `12.26`, `12.27`, `12.28`, `12.24`)** — `[x]` Concluído em `4eb81a31`.
2. **Lote 2 — Escapamento LaTeX em Tabelas Geradas (`Task 12.29` / `BUG-45`)** — `[x]` Concluído em `4eb81a31` com `escape_latex_identifier` compartilhado em `report_generator.py` e `cd_diagram.py`.
3. **Portão Final de Revisão (`22c181dd`)** — `[x]` `PASS` em `Task 12.23`–`12.29` (`BUG-41`–`BUG-45`) + refinos não bloqueantes aplicados e revalidados.

- [x] **Task 12.30: Parada Estrita Step-Down em `holm_bonferroni_correction` (`DP12-23`, `BUG-46`)**
  + **What:** Corrigir `holm_bonferroni_correction` em [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L158-L174) (reexportada em [benchmarking/**init**.py](../../src/benchmarking/__init__.py#L16)) para aplicar o procedimento step-down monotônico de Holm (1979), interrompendo rejeições na primeira hipótese não rejeitada.
  + **Why:** Hoje `reject_sorted = sorted_pvals < critical_values` avalia cada índice isoladamente; se $p_{(1)} \ge \alpha/m$ e $p_{(2)} < \alpha/(m-1)$, retorna simultaneamente `reject=True` e `p_adj > alpha`, violando a garantia FWER e divergindo de [statistics.py](../../src/benchmarking/analysis/statistics.py#L330-L355).
  + **Who:** Agente (`4eb81a31`) com portão de revisão `22c181dd` (`PASS`).
  + **Where:** [problem_statistics.py#L134-L174](../../src/benchmarking/analysis/problem_statistics.py#L134-L174), [statistics.py#L330-L355](../../src/benchmarking/analysis/statistics.py#L330-L355), [test_unit_problem_statistics.py#L144-L177](../../tests/benchmarking/test_unit_problem_statistics.py#L144-L177), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L218).
  + **When:** Descoberta no Passo B da revisão `/code-review-and-quality` (`4eb81a31`); executar antes da geração estatística da dissertação (`Task 12.4`).
  + **How:** Substituir `reject_sorted = sorted_pvals < critical_values` por acumulação monotônica step-down (`np.minimum.accumulate(sorted_pvals <= critical_values)` ou `pvals_corrected_sorted <= alpha`, verificando `<` vs `<=` em igualdade exata e avaliando delegação/compartilhamento DRY com `statistics.py#L330-L355`); adicionar caso de teste que falha hoje (ex.: $p = [0{,}03, 0{,}04]$ com $\alpha=0{,}05 \Rightarrow$ limiares $[0{,}025, 0{,}05]$, onde o código atual rejeita falsamente o segundo índice).
  + **Blast Radius:** [problem_statistics.py#L158-L174](../../src/benchmarking/analysis/problem_statistics.py#L158-L174), [problem_statistics.py#L374](../../src/benchmarking/analysis/problem_statistics.py#L374) (chamador interno), [benchmarking/**init**.py#L16](../../src/benchmarking/__init__.py#L16), [benchmarking/analysis/**init**.py#L14](../../src/benchmarking/analysis/__init__.py#L14).
  + **Relação com outras tarefas:** Complementa o hardening estatístico das `Task 12.25`–`12.26` e precede `Task 12.4` / `Task 12.5`.
  + **How much:** Trivial/Baixo — ~10 linhas em `problem_statistics.py` + 1 teste de regressão step-down.
  + **Stakeholder Implications:** Impede falsas declarações de significância pareada inconsistentes com $p$-valores ajustados nos logs e relatórios da dissertação.
  + **DoD:** Caso não-monotônico `[0.03, 0.04]` ($\alpha=0.05$) retorna `[False, False]` e `all(r == (p <= alpha))`; `pytest tests/benchmarking/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L162-L168) e [test_unit_problem_statistics.py](../../tests/benchmarking/test_unit_problem_statistics.py#L144-L177).
    - **Resumo:** Substituída a comparação independente por índice em `holm_bonferroni_correction` pela regra step-down monotônica `reject_sorted = pvals_corrected_sorted <= alpha`, assegurando que `reject[i]` seja estritamente equivalente a $p_{\text{adj}}[i] \le \alpha$ e interrompendo rejeições na primeira falha do critério step-down; adicionados testes para o vetor não-monotônico `[0.03, 0.04]` e consistência de limiar exato.
    - **Atendimento ao DoD:** Re-revisão `22c181dd` (`PASS`); `pytest -W error tests/benchmarking/test_unit_problem_statistics.py ...` -> `187 passed in 9.21s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `360 passed, 1 deselected in 12.25s`; `mypy src tests benchmark_scripts` -> `Success: no issues found in 147 source files`; `ruff check src tests benchmark_scripts` -> `All checks passed!`; `check_docs` exit `0`.

- [x] **Task 12.31: Calibração $q_\alpha = q_{\text{Tukey}} / \sqrt{2}$ em `StatisticalAnalyzer.nemenyi_posthoc` (`DP12-24`, `BUG-47`)**
  + **What:** Corrigir o cálculo da Distância Crítica ($CD$) em `StatisticalAnalyzer.nemenyi_posthoc` ([statistics.py](../../src/benchmarking/analysis/statistics.py#L409-L448)) dividindo `q_critical` por $\sqrt{2}$ conforme Demšar (2006, Eq. 8) e espelhando [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L23).
  + **Why:** `scipy.stats.studentized_range.ppf` retorna o quantil $q_{\text{Tukey}}$, enquanto a fórmula $CD = q_\alpha \sqrt{k(k+1)/(6N)}$ usa $q_\alpha = q_{\text{Tukey}} / \sqrt{2}$. Omitir `/ np.sqrt(2.0)` infla `critical_distance` em $+41{,}4\%$ e entra em contradição direta com os próprios $p$-valores pareados calculados em [#L431-L436](../../src/benchmarking/analysis/statistics.py#L431-L436) e com o diagrama CD.
  + **Who:** Agente (`4eb81a31`) com portão de revisão `22c181dd` (`PASS`).
  + **Where:** [stats_utils.py#L148-L167](../../src/benchmarking/analysis/stats_utils.py#L148-L167), [statistics.py#L409-L448](../../src/benchmarking/analysis/statistics.py#L409-L448), [cd_diagram.py#L23](../../src/benchmarking/analysis/cd_diagram.py#L23), [test_unit_statistics.py#L133-L206](../../tests/benchmarking/test_unit_statistics.py#L133-L206), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L219).
  + **When:** Bloqueador descoberto no Passo B da revisão (`4eb81a31`); obrigatório antes da consolidação estatística da `Task 12.4` / `Task 12.5`.
  + **How:** Dividir `q_critical` por `np.sqrt(2.0)` antes de multiplicar por $\sqrt{k(k+1)/(6N)}$ em `statistics.py#L447-L448` (ou extrair helper compartilhado com `cd_diagram.py#L124-L126` para eliminar duplicação DRY); validar em teste unitário que `nemenyi_posthoc()["critical_distance"]` coincide exatamente com o `cd` de `cd_diagram.py` e com a consistência $|\bar{R}_i - \bar{R}_j| > CD \iff p_{ij} < \alpha$.
  + **Blast Radius:** [stats_utils.py#L148-L167](../../src/benchmarking/analysis/stats_utils.py#L148-L167), [statistics.py#L409-L448](../../src/benchmarking/analysis/statistics.py#L409-L448), [cd_diagram.py#L23](../../src/benchmarking/analysis/cd_diagram.py#L23), [problem_statistics.py#L293](../../src/benchmarking/analysis/problem_statistics.py#L293), `src/benchmarking/analysis/aggregate_statistics.py#L204`, [test_integration_pipeline.py#L119](../../tests/benchmarking/test_integration_pipeline.py#L119).
  + **Relação com outras tarefas:** Alinha `statistics.py` ao contrato canônico já aplicado no diagrama CD (Fase 9/10) e garante consistência numérica para `Task 12.4` / `Task 12.5`.
  + **How much:** Trivial — ~5 linhas em `statistics.py` (ou helper compartilhado com `cd_diagram.py`) + asserção de paridade no teste.
  + **Stakeholder Implications:** Corrige erro sistemático de $+41{,}4\%$ na distância crítica Nemenyi reportada nos resumos estatísticos do capítulo de resultados.
  + **DoD:** `nemenyi_posthoc` e `cd_diagram.py` produzem $CD$ idêntico à 10ª casa decimal para $(k, N, \alpha)$ arbitrários e $|\bar{R}_i - \bar{R}_j| > CD \iff p_{ij} < \alpha$; `pytest tests/benchmarking/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [stats_utils.py](../../src/benchmarking/analysis/stats_utils.py#L148-L167), [statistics.py](../../src/benchmarking/analysis/statistics.py#L409-L448), [cd_diagram.py](../../src/benchmarking/analysis/cd_diagram.py#L23) e [test_unit_statistics.py](../../tests/benchmarking/test_unit_statistics.py#L133-L206).
    - **Resumo:** Extraído o helper compartilhado `compute_nemenyi_critical_distance(k, n, alpha)` em `stats_utils.py` aplicando `q_tukey / sqrt(2.0)` e guardas explícitos (`k >= 2`, `n >= 1`, `0 < alpha < 1`), reutilizado em `StatisticalAnalyzer.nemenyi_posthoc` (com guarda para `k < 2` / `n_instances <= 0`) e em `cd_diagram.py`; adicionados testes unitários de paridade numérica $CD$, consistência $|\bar{R}_i - \bar{R}_j| > CD \iff p_{ij} < \alpha$ e rejeição de entradas inválidas.
    - **Atendimento ao DoD:** Re-revisão `22c181dd` (`PASS`); `pytest -W error tests/benchmarking/test_unit_statistics.py ...` -> `187 passed in 9.21s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `360 passed, 1 deselected in 12.25s`; `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.32: Eliminação de Fallback Fabricado `raw_costs -> initial_costs` na Retomada Incremental (`DP12-25`, `BUG-48`)**
  + **What:** Remover o fallback `existing_checkpoint.get("raw_initial_costs", existing_checkpoint.get("raw_costs", []))` em `_initialize_results_dict` ([algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L93-L108)) para nunca copiar custos finais convergidos como custos iniciais na retomada de checkpoints antigos.
  + **Why:** Quando um checkpoint legado sem `"raw_initial_costs"` é retomado, copiar `"raw_costs"` faz `improvements = (initial - final) / initial * 100` ([algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L179-L182)) gravar `0.0%` fabricado no checkpoint e no DuckDB.
  + **Who:** Agente (`4eb81a31`) com portão de revisão `22c181dd` (`PASS`).
  + **Where:** [algorithm_runner.py#L93-L108](../../src/benchmarking/runner/algorithm_runner.py#L93-L108), [algorithm_runner.py#L179-L182](../../src/benchmarking/runner/algorithm_runner.py#L179-L182), [algorithm_runner.py#L433-L436](../../src/benchmarking/runner/algorithm_runner.py#L433-L436), [test_unit_runner_telemetry.py#L358-L452](../../tests/benchmarking/test_unit_runner_telemetry.py#L358-L452), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L220).
  + **When:** Bloqueador descoberto no Passo B da revisão (`4eb81a31`); corrigir antes de reprocessar retomadas incrementais.
  + **How:** Quando `"raw_initial_costs"` estiver ausente em `existing_checkpoint`, inicializar `"initial_costs"` preenchido com `[np.nan] * len(raw_costs)` (ou vazio com tratamento explícito de comprimento/máscara em `_build_summary_payload#L173` para que as novas repetições continuem alinhadas por índice às séries `times`/`final_costs` enquanto as antigas permanecem `NaN` e não contaminam `compute_series_stats`); cobrir com teste unitário em `test_unit_runner_telemetry.py` retomando checkpoint sem `"raw_initial_costs"` e provando que `improvements` das novas repetições não sofre desalinhamento nem fabricação de `0.0%`.
  + **Blast Radius:** [algorithm_runner.py#L93-L108](../../src/benchmarking/runner/algorithm_runner.py#L93-L108), [algorithm_runner.py#L179-L182](../../src/benchmarking/runner/algorithm_runner.py#L179-L182), [algorithm_runner.py#L433-L436](../../src/benchmarking/runner/algorithm_runner.py#L433-L436), [orchestration.py#L203-L207](../../src/benchmarking/runner/orchestration.py#L203-L207), [test_unit_runner_telemetry.py#L358-L452](../../tests/benchmarking/test_unit_runner_telemetry.py#L358-L452).
  + **Relação com outras tarefas:** Fecha brecha residual de retomada incremental conectada às `Task 12.11`, `Task 12.23` e `Task 12.27`.
  + **How much:** Baixo — ~15 linhas em `algorithm_runner.py` + teste de retomada sem `raw_initial_costs`.
  + **Stakeholder Implications:** Impede corrupção silenciosa da métrica de melhoria evolutiva (`mean_improvement`) após retomadas incrementais.
  + **DoD:** Retomada de checkpoint sem `"raw_initial_costs"` preserva `NaN` para execuções anteriores e computa melhoria real apenas sobre execuções com custo inicial válido (sem fabricar `0.0%` nem erro de broadcast em `_build_summary_payload`); suíte CPU, `mypy`, `ruff` e `check_docs` exit `0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L93-L108) ([L179-L182](../../src/benchmarking/runner/algorithm_runner.py#L179-L182), [L433-L436](../../src/benchmarking/runner/algorithm_runner.py#L433-L436)) e [test_unit_runner_telemetry.py](../../tests/benchmarking/test_unit_runner_telemetry.py#L358-L452).
    - **Resumo:** Removido o fallback para `"raw_costs"` em `_initialize_results_dict`, preenchendo `"initial_costs"` com `[float("nan")] * len(existing_raw_costs)` na ausência de `"raw_initial_costs"`; aplicada máscara `np.divide(..., where=valid_init)` em `_build_summary_payload` e proteção para `initial_cost` finito positivo no loop de execução; adicionados testes de retomada de checkpoint legado validando alinhamento, ausência de `0.0%` fabricado e proteção contra `RuntimeWarning` em `-W error`.
    - **Atendimento ao DoD:** Re-revisão `22c181dd` (`PASS`); `pytest -W error tests/benchmarking/test_unit_runner_telemetry.py ...` -> `187 passed in 9.21s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `360 passed, 1 deselected in 12.25s`; `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.33: Idempotência DuckDB por Chave Natural `(problem_name, algorithm)` em `import_checkpoints.py` (`DP12-26`, `BUG-49`)**
  + **What:** Corrigir a lógica de deduplicação/importação em [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L112-L125) para que checkpoints atualizados por retomada incremental (novo `timestamp`) atualizem ou substituam o registro de `(problem_name, algorithm)` em vez de inserir duplicatas históricas.
  + **Why:** `check_duplicate` inclui `AND timestamp = ?`; ao reimportar checkpoints incrementais, múltiplas linhas para o mesmo par `(problem_name, algorithm)` coexistem em `TABLE_RUNS`, fazendo `AVG(mean_time) ... GROUP BY problem_name, algorithm` em `compute_speedup_matrix` ([db_operations.py](../../src/benchmarking/database/db_operations.py#L570-L578)) distorcer os tempos médios com snapshots intermediários.
  + **Who:** Agente (`4eb81a31`) com portão de revisão `22c181dd` (`PASS`).
  + **Where:** [import_checkpoints.py#L112-L238](../../src/benchmarking/database/import_checkpoints.py#L112-L238), [db_operations.py#L570-L578](../../src/benchmarking/database/db_operations.py#L570-L578), [test_unit_db_operations.py#L1044-L1271](../../tests/benchmarking/database/test_unit_db_operations.py#L1044-L1271), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L221).
  + **When:** Achado `MAJOR` do Passo B (`4eb81a31`); obrigatório antes da consolidação DuckDB em `Task 12.4`.
  + **How:** Substituir a verificação por `timestamp` por chave lógica `(problem_name, algorithm)` (mantendo ou atualizando somente se `repetitions`/`timestamp` for mais completo/recente via transação DELETE+INSERT ou UPSERT em DuckDB compatível com FK/índices); cobrir com teste unitário importando snapshot N=5 seguido de snapshot N=10 com novo `timestamp` e verificando contagem = 1 linha e `compute_speedup_matrix` exata.
  + **Blast Radius:** [import_checkpoints.py#L112-L238](../../src/benchmarking/database/import_checkpoints.py#L112-L238), [db_operations.py#L570-L578](../../src/benchmarking/database/db_operations.py#L570-L578), [test_unit_db_operations.py#L1044-L1271](../../tests/benchmarking/database/test_unit_db_operations.py#L1044-L1271).
  + **Relação com outras tarefas:** Preserva a consistência analítica da consulta em lote corrigida na `Task 12.13` (`BUG-39`) quando há retomadas da `Task 12.11` (`BUG-37`).
  + **How much:** Baixo/Médio — ~25 linhas em `import_checkpoints.py` + teste de reimportação incremental.
  + **Stakeholder Implications:** Garante que o banco analítico `results.duckdb` reflita exatamente a coleta final consolidada e não médias mistas com runs parciais.
  + **DoD:** Reimportar checkpoint incremental de mesmo `(problem_name, algorithm)` com novo `timestamp` mantém 1 única linha atualizada em `benchmark_runs`; suíte CPU, `mypy`, `ruff` e `check_docs` exit `0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [import_checkpoints.py](../../src/benchmarking/database/import_checkpoints.py#L112-L125) ([L199](../../src/benchmarking/database/import_checkpoints.py#L199), [L216-L238](../../src/benchmarking/database/import_checkpoints.py#L216-L238)) e [test_unit_db_operations.py](../../tests/benchmarking/database/test_unit_db_operations.py#L1044-L1271).
    - **Resumo:** Alterado `check_duplicate` para consultar a chave natural `(problem_name, algorithm)` (retornando `(id, repetitions, timestamp)`), movida `validate_benchmark_data` para antes da checagem de duplicata, normalizados timestamps para UTC antes da comparação em `_parse_timestamp` e encapsulada a substituição (`DELETE` em tabelas filhas + `benchmark_runs` + `insert_benchmark_run`) em transação atômica (`conn.begin()` / `conn.commit()` / `conn.rollback()`) quando `skip_duplicates=True` recebe snapshot mais completo (`repetitions` maior) ou mais recente, preservando compatibilidade com `skip_duplicates=False`; adicionados testes de upgrade incremental (`N=5` -> `N=10`), empate de contagem com timestamp mais novo, preservação diante de snapshot stale e rollback em falha.
    - **Atendimento ao DoD:** Re-revisão `22c181dd` (`PASS`); `pytest -W error tests/benchmarking/database/test_unit_db_operations.py ...` -> `187 passed in 9.21s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `360 passed, 1 deselected in 12.25s`; `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.34: Exigência de Cobertura Integral da Categoria de Tamanho em Tabela 3 (`DP12-26`, `BUG-50`)**
  + **What:** Atualizar `_compute_table3_size_category_winners` em [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L428-L526) para exigir cobertura completa da categoria (`len(gaps) == len(problems)`) antes de tornar um algoritmo elegível como vencedor.
  + **Why:** Hoje um algoritmo que rodou apenas em 1 problema fácil dentro de uma categoria com 5 instâncias pode exibir `avg_gap` menor e vencer algoritmos avaliados em toda a categoria.
  + **Who:** Agente (`4eb81a31`) com portão de revisão `22c181dd` (`PASS`).
  + **Where:** [report_generator.py#L428-L526](../../src/benchmarking/analysis/report_generator.py#L428-L526), [test_unit_report_generator.py#L433-L492](../../tests/benchmarking/test_unit_report_generator.py#L433-L492), [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L222).
  + **When:** Achado `MAJOR` do Passo B (`4eb81a31`); corrigir antes de `Task 12.4`.
  + **How:** Substituir `if gaps:` por `if len(gaps) == len(problems):` (e definir tratamento limpo caso nenhum algoritmo cubra 100% da categoria — ex.: omitir vencedor com `"—"` e gap/tempo seguros sem exibir `"None"` na Tabela 3 `.md`/`.tex`); adicionar teste comportamental com matriz irregular onde candidato parcial tem gap 0% em 1 instância mas o vencedor completo tem 2% nas N instâncias.
  + **Blast Radius:** [report_generator.py#L428-L526](../../src/benchmarking/analysis/report_generator.py#L428-L526), [test_unit_report_generator.py#L433-L492](../../tests/benchmarking/test_unit_report_generator.py#L433-L492), [test_unit_report_generator_behavioral.py](../../tests/benchmarking/test_unit_report_generator_behavioral.py).
  + **Relação com outras tarefas:** Endurece a refatoração `_compute_table3_size_category_winners` de `Task 12.23` e estende o Caso 5 (matrizes irregulares) de `Task 12.24`.
  + **How much:** Baixo — ~15 linhas em `report_generator.py` + teste de cobertura parcial.
  + **Stakeholder Implications:** Impede ranqueamento injusto na Tabela 3 por cobertura desigual entre variantes CPU/GPU ou runs parciais.
  + **DoD:** Algoritmo com cobertura parcial nunca vence categoria frente a algoritmo com cobertura integral; caso sem vencedor integral renderiza fallback limpo em `.md`/`.tex`; `pytest tests/benchmarking/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [report_generator.py](../../src/benchmarking/analysis/report_generator.py#L428-L526) e [test_unit_report_generator.py](../../tests/benchmarking/test_unit_report_generator.py#L433-L492).
    - **Resumo:** Atualizado `_compute_table3_size_category_winners` para exigir `len(gaps) == len(problems) and len(problems) > 0` antes de tornar um algoritmo elegível como vencedor da categoria, e atualizado o formatador da Tabela 3 em Markdown e LaTeX para renderizar `best_alg = None` de forma segura com `"—"` sem emitir `"None"`; adicionados testes unitários cobrindo matriz irregular (vencedor integral vs candidato parcial de gap `0%`) e ausência de vencedor com cobertura integral.
    - **Atendimento ao DoD:** Re-revisão `22c181dd` (`PASS`); `pytest -W error tests/benchmarking/test_unit_report_generator.py tests/benchmarking/test_unit_report_generator_behavioral.py ...` -> `187 passed in 9.21s`; `pytest tests/benchmarking/ -m "not gpu" -q` -> `360 passed, 1 deselected in 12.25s`; `mypy`, `ruff` e `check_docs` exit `0`.

- [x] **Task 12.35: Remoção de Dupla Execução na Falha em `@time_strategy` (`DP12-27`, `BUG-51`)**
  + **What:** Corrigir `@time_strategy` em [phase_timer.py](../../src/utils/phase_timer.py#L112-L124) para nunca envolver `func(*args, **kwargs)` em `try/except` com replay silencioso em CPU quando uma exceção é lançada durante a execução opt-in com eventos CUDA.
  + **Why:** Se `func` falhar (erro de lógica ou falha CUDA), o bloco `except Exception:` reexecuta `func(*args, **kwargs)` na CPU — repetindo mutações/amostragens estocásticas não-idempotentes e mascarando falhas reais.
  + **Who:** Agente (aguarda aprovação do plano).
  + **Where:** [phase_timer.py#L23-L52](../../src/utils/phase_timer.py#L23-L52) (helper `_timed_call`), [phase_timer.py#L79-L87](../../src/utils/phase_timer.py#L79-L87) (`PhaseTimer.measure`), [phase_timer.py#L127-L132](../../src/utils/phase_timer.py#L127-L132) (`@time_strategy`), [test_unit_phase_timer.py](../../tests/utils/test_unit_phase_timer.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Achado `MAJOR` #1 do Passo A (`4eb81a31`).
  + **How:** Separar criação de `cp.cuda.Event()` em `try/except` ANTES de invocar `func`: se a alocação/registro inicial de evento falhar, executar `func` uma única vez cronometrado via `time.perf_counter()`; uma vez criado o evento, executar `result = func(*args, **kwargs)` fora de captura genérica para que qualquer exceção de `func` ou sincronização se propague imediatamente com contador de chamadas exato (= 1 em caso de sucesso, zero replays em erro); cobrir em `test_unit_phase_timer.py`.
  + **Blast Radius:** [phase_timer.py#L23-L52](../../src/utils/phase_timer.py#L23-L52), [test_unit_phase_timer.py](../../tests/utils/test_unit_phase_timer.py), todas as estratégias decoradas em `src/algorithms/strategies/` (mantendo interface transparente).
  + **Relação com outras tarefas:** Preserva a integridade de `C1` e a previsibilidade do cronômetro usado pelo `PhaseTimer` (`DP9-2` / `Task 11.*`).
  + **How much:** Baixo — ~15 linhas em `phase_timer.py` + teste de não-reexecução em erro.
  + **Stakeholder Implications:** Garante determinismo de RNG/estado e diagnósticos imediatos caso um operador GPU falhe.
  + **DoD:** Em exceção dentro de `func` com `use_gpu_events=True`, `func` é chamada exatamente 1 vez e a exceção original propaga; falha ao criar eventos faz fallback limpo antes de rodar `func`; `pytest tests/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [phase_timer.py#L23-L52](../../src/utils/phase_timer.py#L23-L52), [phase_timer.py#L79-L87](../../src/utils/phase_timer.py#L79-L87), [phase_timer.py#L127-L132](../../src/utils/phase_timer.py#L127-L132).
    - **Resumo:** Extraído `_timed_call[**P, R]`, que arma o par de eventos CUDA dentro de `try/except` e deixa `func(*args, **kwargs)`, `end_evt.record()` e `end_evt.synchronize()` fora de qualquer `try`; `PhaseTimer.measure` e o wrapper de `@time_strategy` passaram a delegar a ele, removendo o bloco duplicado (`BUG-51`). O fallback `time.perf_counter()` só roda quando o arming falha antes da chamada, preservando a execução única e a propagação da exceção original.
    - **Atendimento ao DoD:** `CUDA_VISIBLE_DEVICES="" pytest -W error -m "not gpu" tests/utils/test_unit_phase_timer.py tests/utils/test_unit_gpu_helpers.py tests/algorithms/strategies/mutation/test_unit_swap_mutation.py tests/loaders/test_unit_database_loader_errors.py` -> `67 passed, 1 deselected in 0.60s`, cobrindo `test_time_strategy_executes_func_exactly_once_on_gpu_success`, `test_time_strategy_gpu_error_propagates_without_replay` e `test_time_strategy_gpu_event_init_failure_falls_back_once` (arquivo de testes mantido sem alteração); `pytest -W error -m "not gpu" tests/` -> `981 passed, 208 deselected in 17.76s`; `mypy src tests benchmark_scripts` -> `Success: no issues found in 147 source files`; `ruff` e `check_docs` exit `0` (2026-11-13).

- [x] **Task 12.36: Alinhamento Semântico de `next_power_of_2` com Chamadores CUDA (`DP12-27`, `BUG-52`)**
  + **What:** Resolver a contradição entre nome e implementação de `next_power_of_2` em [gpu_helpers.py](../../src/utils/gpu_helpers.py#L67-L70) (`1 << (val.bit_length() - 1)`, potência piso), auditando se os chamadores CUDA exigem potência piso (`two_opt.py#L561, L612`, `cost_evaluation.py#L89`) ou teto (`1 << (val - 1).bit_length()`).
  + **Why:** Hoje `next_power_of_2(3)` retorna `2` e `next_power_of_2(48)` retorna `32`, violando o Princípio da Menor Surpresa e podendo subalocar threads ou quebrar suposições de redução binária se futuros chamadores esperarem teto.
  + **Who:** Agente (aguarda aprovação do plano).
  + **Where:** [gpu_helpers.py#L67-L70](../../src/utils/gpu_helpers.py#L67-L70), [two_opt.py#L39-L615](../../src/algorithms/strategies/local_search/two_opt.py#L39-L615), [cost_evaluation.py#L16-L92](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L16-L92), `tests/utils/`, [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Achado `MAJOR` #2 do Passo A (`4eb81a31`).
  + **How:** Inspecionar os kernels de `two_opt` e `cost_evaluation` (especialmente limites de bloco vs shared-memory / warp reduction): se a intenção arquitetural for piso ($\lfloor \log_2 v \rfloor$), renomear `next_power_of_2` -> `prev_power_of_2` em `gpu_helpers.py` e em todos os chamadores/testes (zero alias legado per `AGENTS.md` Regra 10); se teto ($\lceil \log_2 v \rceil$), corrigir a fórmula para `1 << (max(1, value) - 1).bit_length()` respeitando `[min_val, max_val]` e invariantes de shared memory; adicionar testes unitários em `tests/utils/`.
  + **Blast Radius:** [gpu_helpers.py#L67-L70](../../src/utils/gpu_helpers.py#L67-L70), [two_opt.py#L39, L561, L612](../../src/algorithms/strategies/local_search/two_opt.py#L561), [cost_evaluation.py#L16, L89](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L89).
  + **Relação com outras tarefas:** Aplica estritamente `AGENTS.md` Regra 10 e protege o dimensionamento de blocos GPU para a coleta integral (`Task 12.4`).
  + **How much:** Trivial/Baixo — ~10 linhas + teste unitário de potências de 2.
  + **Stakeholder Implications:** Elimina armadilha semântica silenciosa ao dimensionar grades/blocos CUDA.
  + **DoD:** Nome, docstring e fórmula de potência de 2 100% coerentes; zero ocorrências do nome antigo se renomeado (caches excluídos); `pytest tests/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [gpu_helpers.py#L67-L70](../../src/utils/gpu_helpers.py#L67-L70), [two_opt.py#L39](../../src/algorithms/strategies/local_search/two_opt.py#L39), [two_opt.py#L561](../../src/algorithms/strategies/local_search/two_opt.py#L561), [two_opt.py#L612](../../src/algorithms/strategies/local_search/two_opt.py#L612), [cost_evaluation.py#L16](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L16), [cost_evaluation.py#L89](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L89), [test_unit_gpu_helpers.py#L143-L179](../../tests/utils/test_unit_gpu_helpers.py#L143-L179).
    - **Resumo:** Renomeado `next_power_of_2` para `prev_power_of_2`, refletindo o piso $\lfloor \log_2 v \rfloor$ com `min_val`/`max_val`, e atualizados todos os chamadores em `two_opt.py` e `cost_evaluation.py`; o nome agora coincide com o contrato das reduções em árvore dos kernels (`cost_calculator.cu#L73-L78`, `two_opt_single.cu#L101-L118`, `two_opt_batch.cu#L234-L252`), onde o piso preserva os strides de halving e permanece $\le n - 2$.
    - **Atendimento ao DoD:** `grep -rn "next_power_of_2" src tests benchmark_scripts --include=*.py --include=*.cu | wc -l` -> `0`; `test_prev_power_of_2_semantics_and_bounds` verde em [test_unit_gpu_helpers.py#L170](../../tests/utils/test_unit_gpu_helpers.py#L170); `mypy`, `ruff` e `check_docs` exit `0` (2026-11-13).

- [x] **Task 12.37: Guarda de Pool Vazio em `select_proximity_index` (`DP12-27`, `BUG-54`)**
  + **What:** Proteger `select_proximity_index` em [route_helpers.py](../../src/utils/route_helpers.py#L167-L168) contra `pool.size == 0` quando `candidates` falha em fornecer vizinho elegível e a fatia mutável `[low, high)` não contém índice distinto de `anchor_idx`.
  + **Why:** Hoje `rng.choice(pool)` levanta `ValueError: 'a' cannot be empty` sem tratamento quando `high - low <= 1` ou `pool` é vazio.
  + **Who:** Agente (aguarda aprovação do plano).
  + **Where:** [route_helpers.py#L128-L169](../../src/utils/route_helpers.py#L128-L169), [test_unit_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Achado `MAJOR` #4 do Passo A (`4eb81a31`).
  + **How:** Checar `if pool.size == 0:` antes de `rng.choice(pool)` — alinhando com a docstring (`"An index in [low, high) different from anchor_idx whenever the slice holds at least two positions"`) retornando `anchor_idx` quando `low <= anchor_idx < high` (ou `low` se `low < high`, ou `ValueError` descritivo se `low >= high`); adicionar teste unitário para slice unitário/vazio.
  + **Blast Radius:** [route_helpers.py#L167-L168](../../src/utils/route_helpers.py#L167-L168), [route_helpers.py#L218](../../src/utils/route_helpers.py#L218) (`sample_mutation_positions`), [test_unit_swap_mutation.py](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py).
  + **Relação com outras tarefas:** Complementa a robustez de operadores compartilhados (`sample_mutation_positions`) introduzida na Fase 11 (`BUG-28`).
  + **How much:** Trivial — ~6 linhas + 1 teste de borda.
  + **Stakeholder Implications:** Previne crashes em mutação sobre sub-rotas/instâncias degeneradas.
  + **DoD:** `select_proximity_index` com slice de tamanho 1 nunca levanta `ValueError: 'a' cannot be empty` e retorna índice válido; `pytest tests/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [route_helpers.py#L167-L175](../../src/utils/route_helpers.py#L167-L175), [test_unit_swap_mutation.py#L198-L218](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L198-L218).
    - **Resumo:** Adicionada guarda para `pool.size == 0` em `select_proximity_index`: slice degenerado (`low >= high`) levanta `ValueError` explícito, singleton que contém a âncora devolve `anchor_idx` e os demais casos devolvem `low`, eliminando o `ValueError: 'a' cannot be empty` do `rng.choice` (`BUG-54`).
    - **Atendimento ao DoD:** `test_select_proximity_index_singleton_and_empty_guards` verde em [test_unit_swap_mutation.py#L198](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L198) (batch dedicado -> `67 passed, 1 deselected in 0.60s`); suíte completa `981 passed, 208 deselected`; `mypy`, `ruff` e `check_docs` exit `0` (2026-11-13).

- [x] **Task 12.38: Contrato Explícito 3D e Validação `NULL` em Loaders TSPLIB/DuckDB (`DP12-28`, `BUG-53`)**
  + **What:** Corrigir [tsplib_loader.py](../../src/loaders/tsplib_loader.py#L168-L180) e [database_loader.py](../../src/loaders/database_loader.py#L231, L272-L273) para: (1) não descartar silenciosamente a 3ª coordenada (`z`) quando métricas 3D (`EUC_3D`, `MAN_3D`, `MAX_3D`) forem encontradas — decidindo entre suporte completo a `(x, y, z)` ou rejeição explícita com `UnsupportedEdgeTypeError`/`InvalidProblemDataError` na ingestão; e (2) validar `matrix_raw is None` em `database_loader._load_distance_matrix` levantando `InvalidProblemDataError`.
  + **Why:** Descartar `z` silenciosamente gera matrizes de distância 2D incorretas para instâncias 3D, e `np.asarray(None)` em `_load_distance_matrix` levanta `TypeError` críptico em vez de erro de domínio.
  + **Who:** Agente (aguarda aprovação do plano).
  + **Where:** [tsplib_loader.py#L149-L175](../../src/loaders/tsplib_loader.py#L149-L175) (`_bundle_from_normalized`), [tsplib_loader.py#L395-L424](../../src/loaders/tsplib_loader.py#L395-L424) (`_problem_payload`), [database_loader.py#L267-L282](../../src/loaders/database_loader.py#L267-L282), `tests/loaders/`, [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  + **When:** Achados `MAJOR` #3 e `MINOR` #7 do Passo A (`4eb81a31`).
  + **How:** Auditar se `routing_data`/schema DuckDB armazenam apenas `(x, y)` (em caso positivo, validar na ingestão rejeitando `EDGE_WEIGHT_TYPE` 3D explicitamente em vez de truncar, ou suportar `z` se o esquema permitir); em `database_loader.py#L272-L273`, adicionar `if matrix_raw is None: raise InvalidProblemDataError(...)` espelhando `tsplib_loader.py`; adicionar testes unitários em `tests/loaders/`.
  + **Blast Radius:** [tsplib_loader.py#L149-L175](../../src/loaders/tsplib_loader.py#L149-L175), [tsplib_loader.py#L395-L424](../../src/loaders/tsplib_loader.py#L395-L424), [database_loader.py#L267-L282](../../src/loaders/database_loader.py#L267-L282), `tests/loaders/`.
  + **Relação com outras tarefas:** Garante integridade na fronteira de entrada dos dados TSPLIB95 (`DP12-12`).
  + **How much:** Baixo — ~20 linhas nos loaders + testes unitários.
  + **Stakeholder Implications:** Impede corrupção silenciosa caso instâncias 3D da TSPLIB95 sejam adicionadas ao catálogo e melhora diagnósticos de banco corrompido.
  + **DoD:** Instâncias 3D têm comportamento explícito e testado (sem truncamento silencioso) e `matrix_raw is None` levanta `InvalidProblemDataError`; `pytest tests/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [tsplib_loader.py#L149-L175](../../src/loaders/tsplib_loader.py#L149-L175), [tsplib_loader.py#L395-L424](../../src/loaders/tsplib_loader.py#L395-L424), [database_loader.py#L267-L282](../../src/loaders/database_loader.py#L267-L282), [test_unit_database_loader_errors.py#L79-L190](../../tests/loaders/test_unit_database_loader_errors.py#L79-L190).
    - **Resumo:** `_bundle_from_normalized` e `_problem_payload` agora validam `edge_type in VALID_EDGE_TYPES` e rejeitam coordenadas com `len(pt) > 2` via `InvalidProblemDataError`, encerrando o truncamento silencioso do eixo `z`; `database_loader._load_distance_matrix` levanta `InvalidProblemDataError` quando `matrix_raw is None`, espelhando o contrato do `tsplib_loader` (`BUG-53`).
    - **Atendimento ao DoD:** `test_tsplib_loader_rejects_3d_coordinates_and_3d_edge_types` e `test_database_loader_explicit_matrix_null_raises_invalid_problem_data_error` verdes em [test_unit_database_loader_errors.py#L129](../../tests/loaders/test_unit_database_loader_errors.py#L129) e [test_unit_database_loader_errors.py#L79](../../tests/loaders/test_unit_database_loader_errors.py#L79); suíte completa `981 passed, 208 deselected`; `mypy`, `ruff` e `check_docs` exit `0` (2026-11-13).

- [x] **Task 12.39: Saneamento de Baseline em `problem_statistics.py`, Aliases CuPy em `types.py` e Protocolos Legados (`DP12-28`, `BUG-55`)**
  + **What:** Endereçar os itens menores restantes dos Passos A e B: (1) parametrizar `baseline_algorithm` em `problem_statistics.py#L412` removendo o hardcode `"CPU"` (alinhado a `DP12-9`); (2) ampliar os aliases de [types.py](../../src/protocols/types.py#L20-L33) onde aplicável ou documentar/unir tipos de backend e matriz de distância float sem quebrar `mypy`; e (3) remover protocolos mortos (`BinPackingStrategy`, etc.)/docstrings quebradas e reexportações de `route_helpers` em [algorithm_strategies.py](../../src/protocols/algorithm_strategies.py#L40-L115) e [protocols/**init**.py](../../src/protocols/__init__.py#L18-L19) atualizando os chamadores diretos para `utils.route_helpers` (`AGENTS.md` Regra 10).
  + **Why:** Mantém a fronteira arquitetural limpa (`protocols` apenas para contratos ativos, `utils` para helpers de array), completa a eliminação de referências fixas à baseline `"CPU"` e reduz dívida técnica antes do congelamento final do código da dissertação.
  + **Who:** Agente (aguarda aprovação do plano).
  + **Where:** [problem_statistics.py#L419](../../src/benchmarking/analysis/problem_statistics.py#L419), [stats_utils.py#L169-L194](../../src/benchmarking/analysis/stats_utils.py#L169-L194), [report_generator.py#L714](../../src/benchmarking/analysis/report_generator.py#L714), [types.py#L34-L46](../../src/protocols/types.py#L34-L46), [algorithm_strategies.py#L61-L176](../../src/protocols/algorithm_strategies.py#L61-L176), [protocols/**init**.py#L1-L56](../../src/protocols/__init__.py#L1-L56).
  + **When:** Baixa prioridade — executar em lote de saneamento após `Task 12.30`–`12.38`.
  + **How:** Parametrizar `baseline_algorithm: str | None = None` em `perform_statistical_analysis` (usando `get_baseline_algorithm` de `report_generator.py` ou parâmetro explícito); verificar referências antes de remover protocolos não implementados em `algorithm_strategies.py` e migrar imports de `ensure_routes_batch`/`restore_routes_shape` para `utils.route_helpers`; manter compatibilidade estrita no `mypy --strict`/`src tests benchmark_scripts`.
  + **Blast Radius:** Chamadores de `protocols` em `src/algorithms/`, `tests/`, [problem_statistics.py#L182-L423](../../src/benchmarking/analysis/problem_statistics.py#L182-L423).
  + **Relação com outras tarefas:** Completa `Task 12.16` (`DP12-9`, fim do hardcode `"CPU"`) e consolida tipagem iniciada nas fases anteriores.
  + **How much:** Médio — limpeza estrutural em ~4 arquivos + ajuste de imports.
  + **Stakeholder Implications:** Base de código limpa, sem interfaces mortas nem acoplamentos espúrios na entrega final do TCC.
  + **DoD:** Zero reexportações de `route_helpers` em `src/protocols/`, zero protocolos mortos com docstrings quebradas, `baseline_algorithm` configurável em `problem_statistics.py`, `mypy src tests benchmark_scripts`, `ruff`, `pytest tests/ -m "not gpu"` e `check_docs` exit `0`.

  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [stats_utils.py#L169-L194](../../src/benchmarking/analysis/stats_utils.py#L169-L194), [problem_statistics.py#L208-L212](../../src/benchmarking/analysis/problem_statistics.py#L208-L212), [problem_statistics.py#L419](../../src/benchmarking/analysis/problem_statistics.py#L419), [report_generator.py#L714](../../src/benchmarking/analysis/report_generator.py#L714), [orchestration.py#L623](../../src/benchmarking/runner/orchestration.py#L623), [types.py#L34-L46](../../src/protocols/types.py#L34-L46), [algorithm_strategies.py#L61-L176](../../src/protocols/algorithm_strategies.py#L61-L176), [test_unit_problem_statistics.py#L179-L249](../../tests/benchmarking/test_unit_problem_statistics.py#L179-L249), [test_integration_pipeline.py#L236-L268](../../tests/benchmarking/test_integration_pipeline.py#L236-L268).
    - **Resumo:** `perform_statistical_analysis` deixou de hardcodear `"CPU"` e resolve a baseline por `resolve_baseline_algorithm`, o mesmo helper usado por `generate_result_tables`; `run_comprehensive_benchmark` passou a encaminhar `baseline_algorithm` às estatísticas (antes só às tabelas); `types.py` ganhou `DistanceMatrix = NDArray[np.int32] | NDArray[np.float64]` e os aliases de dispositivo `DeviceRoute`/`DeviceRoutes`/`DeviceFitness`/`DeviceDistanceMatrix`; removidos os protocolos mortos, docstrings `code/src/...` e as reexportações de `route_helpers` (`BUG-55`). O teste de encaminhamento ficou em [test_integration_pipeline.py#L236](../../tests/benchmarking/test_integration_pipeline.py#L236) — uma duplicata unitária com 12 stubs foi descartada por reafirmar o wiring interno.
    - **Atendimento ao DoD:** `grep -rn "from protocols import .*ensure_routes_batch\|BinPackingStrategy" src --include=*.py | wc -l` -> `0`; `test_perform_statistical_analysis_default_baseline_picks_first_even_when_cpu_present_at_index_gt_0` e `test_run_comprehensive_benchmark_forwards_baseline_algorithm` verdes; `mypy src tests benchmark_scripts` -> `Success: no issues found in 147 source files`; `pytest -W error -m "not gpu" tests/` -> `981 passed, 208 deselected`; `ruff` e `check_docs` exit `0` (2026-11-13).
    - **Ressalva registrada:** a ordem de entrada do helper ainda difere entre os chamadores (`list(results.keys())` em `problem_statistics.py#L217` vs `sorted(algo_set)` em `report_generator.py#L711-L714`), então `baseline_algorithm=None` pode divergir entre estatísticas e tabelas; rastreado em `BUG-56` / `Task 12.40`, fora do escopo declarado no `What` desta tarefa.

- [ ] **Task 12.40: Ordem Canônica de Algoritmos na Resolução de Baseline e Encaminhamento no CLI (`DP12-29`, `BUG-56`)**
  + **What:** Unificar a sequência de entrada de `resolve_baseline_algorithm` nas duas rotas — [problem_statistics.py#L217](../../src/benchmarking/analysis/problem_statistics.py#L217) (`list(results.keys())`) e [report_generator.py#L711-L714](../../src/benchmarking/analysis/report_generator.py#L711-L714) (`sorted(algo_set)`) — e encaminhar `baseline_algorithm` no entry point [benchmark.py#L198-L218](../../src/cli/benchmark.py#L198-L218).
  + **Why:** Com `baseline_algorithm=None`, cada rota escolhe "o primeiro" de sequências construídas de forma diferente, e `gpu-accelerated benchmark` nunca encaminha a baseline; estatísticas por problema e tabelas podem então nomear baselines distintas (ex.: `HybridNaive` vs `CPU`), reintroduzindo a divergência que `BUG-55` deveria eliminar.
  + **Who:** Agente (aguarda decisão de `DP12-29`).
  + **Where:** [problem_statistics.py#L217-L220](../../src/benchmarking/analysis/problem_statistics.py#L217-L220), [report_generator.py#L711-L714](../../src/benchmarking/analysis/report_generator.py#L711-L714), [benchmark.py#L198-L218](../../src/cli/benchmark.py#L198-L218), `tests/benchmarking/`.
  + **When:** Somente após o término de `Task 12.4`: a correção muda a baseline dos problemas `n>400` e invalidaria comparações já gravadas na coleta em andamento.
  + **How:** Fixar a ordem canônica em `DP12-29` (ordem de execução vs alfabética) e aplicá-la nos dois pontos, preferencialmente resolvendo a baseline por problema em `orchestration.run_comprehensive_benchmark` e repassando o valor já resolvido; cobrir com teste que passe `baseline_algorithm=None` e outro com `"CPU"` em índice `> 0` nas duas rotas.
  + **Blast Radius:** `src/benchmarking/analysis/`, [orchestration.py#L618-L623](../../src/benchmarking/runner/orchestration.py#L618-L623), [benchmark.py#L198-L218](../../src/cli/benchmark.py#L198-L218), `tests/benchmarking/`.
  + **Relação com outras tarefas:** Residual de `Task 12.39` (`BUG-55` / `DP12-9`); bloqueada por `Task 12.4`.
  + **How much:** Baixo — 3 pontos de chamada + 2 testes.
  + **Stakeholder Implications:** Impede que a dissertação reporte speedups relativos a baselines diferentes entre o texto e as tabelas.
  + **DoD:** Com `baseline_algorithm=None`, estatísticas e tabelas nomeiam a mesma baseline para o mesmo problema; `pytest tests/ -m "not gpu"`, `mypy`, `ruff` e `check_docs` exit `0`.

#### Plano de Lotes — Remediação das Revisões Passo A e Passo B (`Task 12.30`–`12.39`)

1. [x] **Lote 1 — Bloqueadores Estatísticos, Runner e DuckDB em `src/benchmarking/` (`Task 12.30`, `12.31`, `12.32`, `12.33`, `12.34` / `BUG-46`–`BUG-50`)**: Concluído por `4eb81a31` e homologado no portão de revisão `22c181dd` (`PASS`) — corrigidas as 3 falhas `BLOCKING` (`holm_bonferroni_correction`, `nemenyi_posthoc` $q/\sqrt{2}$, remoção de fallback `raw_costs -> initial_costs`) e as 2 `MAJOR` (idempotência DuckDB por `(problem_name, algorithm)` e cobertura integral na Tabela 3).
2. [x] **Lote 2 — Hardening de Utilitários, Loaders e Saneamento de Protocolos (`Task 12.35`, `12.36`, `12.37`, `12.38`, `12.39` / `BUG-51`–`BUG-55`)**: corrigir `@time_strategy`, `next_power_of_2`, `select_proximity_index`, contrato 3D/NULL em loaders e limpeza de dívida técnica (`baseline_algorithm` + `src/protocols/`), seguido de portão final de revisão. **Concluído:** implementado por `4eb81a31`, o portão `22c181dd` retornou `CHANGES_REQUESTED` (baseline `"CPU"` residual, `baseline_algorithm` não encaminhado em `orchestration.py`, `_problem_payload` sem validação própria, bloco de eventos CUDA duplicado) e as correções foram finalizadas por `613f68c6`; revalidado com `981 passed, 208 deselected`, `mypy` 147 arquivos, `ruff` e `check_docs` exit `0` (2026-11-13).

---

## Handoff

- **State**: in progress — Lote 2 concluído; Fase 12 aberta em `Task 12.1`, `12.2`, `12.4`, `12.5` e `12.40`
- **Last verified observation**:
  + `CUDA_VISIBLE_DEVICES="" pytest -W error -m "not gpu" tests/ -q --deselect tests/benchmarking/test_unit_runner_warmup.py` -> `981 passed, 208 deselected in 17.76s` (2026-11-13)
  + `mypy src tests benchmark_scripts` -> `Success: no issues found in 147 source files`; `ruff check src tests benchmark_scripts` -> `All checks passed!` (2026-11-13)
  + `python -m benchmark_scripts.check_docs` -> `stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0` (2026-11-13)
  + `grep -rn "next_power_of_2" src tests benchmark_scripts --include=*.py --include=*.cu | wc -l` -> `0`; `grep -rn "from protocols import .*ensure_routes_batch\|BinPackingStrategy" src --include=*.py | wc -l` -> `0` (2026-11-13)
  + `pytest -W error tests/utils/test_unit_phase_timer.py tests/utils/test_unit_gpu_helpers.py tests/algorithms/strategies/mutation/test_unit_swap_mutation.py tests/loaders/test_unit_database_loader_errors.py` -> `67 passed, 1 deselected in 0.60s` (2026-11-13)
- **Status of Fase 12**: Backlog com 40 tarefas (`Task 12.1` a `Task 12.40`), 29 decisões (`DP12-1` a `DP12-29`) e 24 correções analíticas (`BUG-33` a `BUG-56`). Concluídas e homologadas: **35 de 40 tarefas** (`Task 12.3`, `12.6`–`12.39`; `BUG-33` a `BUG-55` resolvidos). Em aberto: `Task 12.1`, `12.2`, `12.5` (dissertação `.tex`), `Task 12.4` (coleta integral em execução fora desta sessão) e `Task 12.40` (`BUG-56`, bloqueada por `Task 12.4`).
- **Next item**: acompanhar `Task 12.4` — `run_chapter4_benchmark --no-incremental` (PID 1413824) no problema 15/15 (`pr1002`), algoritmo 1/3 (`HybridNaive`), repetição 5/30, ~35 min por repetição; em seguida decidir `DP12-29` e executar `Task 12.40`.
- **Open decisions**: `DP12-29` — ordem canônica para resolução de baseline quando `baseline_algorithm=None`: ordem de execução (atual em `problem_statistics.py#L217`, preserva os números já gravados) ou alfabética (atual em `report_generator.py#L711-L714`)? | Lucas Galdino
- **Read first**: [stats_utils.py#L169-L194](../../src/benchmarking/analysis/stats_utils.py#L169-L194), [analysis-current-cpu-bottlenecks.md](./knowledge-base/analysis-current-cpu-bottlenecks.md#L30), [review-benchmarking-architecture-and-operators.md](./knowledge-base/review-benchmarking-architecture-and-operators.md#L50), [table-of-symbols.plan.md](./table-of-symbols.plan.md#L223)
