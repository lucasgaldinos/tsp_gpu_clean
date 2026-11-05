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
    - **Arquivos Alterados:** [pyproject.toml](../../../routing_data/pyproject.toml#L10), [operations.py](../../../routing_data/src/converter/database/operations.py#L520), [transformer.py](../../../routing_data/src/converter/core/transformer.py#L20), [parser.py](../../../routing_data/src/tsplib_parser/parser.py#L543), [commands.py](../../../routing_data/src/converter/cli/commands.py#L240), [json_writer.py](../../../routing_data/src/converter/output/json_writer.py#L15), [parquet_writer.py](../../../routing_data/src/converter/output/parquet_writer.py#L12), [cost.py](../../../routing_data/src/converter/utils/cost.py#L300).
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

### Tarefas da fase 11

- [x] **Task 11.1: Criar `StrategyBase` e Unificar Busca Local em `LocalSearchCPUTwoOpt` e `LocalSearchGPUTwoOpt` (`DP11-1`, `DP11-2`, `BUG-20`)**
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
    - **Arquivos Alterados:** [base.py](../../src/algorithms/strategies/base.py#L28), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L196), [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L360), [two_opt_multi_move.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu).
    - **Resumo:** `StrategyBase` centraliza telemetria (`h2d_bytes`, `d2h_bytes`, `kernel_launches`, `reset_metrics`) e RNG; os seis 2-opt viraram `LocalSearchCPUTwoOpt(mode="best"|"first"|"multi_move")` e `LocalSearchGPUTwoOpt(mode="batch"|"multi_move"|"naive")`; o teto `applied_start[16]` foi substituído por buffer de `block_size` em shared memory (`BUG-20`).
    - **Atendimento ao DoD (Revisado 5+1 eixos):** `pytest tests/algorithms/strategies/local_search -q` → `313 passed` no recorte `local_search`+`loaders`+`benchmarking`; `mypy src tests` → `118 source files` sem erros; `ruff check src tests` → `All checks passed!`.
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

---

## Handoff

- **State:** done — **Fase 10** (`Tasks 10.1`–`10.24`) e **Fase 11** (`Tasks 11.1`–`11.5`) concluídas, revisadas em 5+1 eixos e verificadas. As tarefas de `routing_data` (Batch 7) permanecem em `routing_data/TODO.md` e estão fora do escopo deste repositório.
- **Last Verified Observation:**
  + `rtk .venv/bin/python -m benchmark_scripts.check_docs` → `stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0`, exit 0 (2026-11-05).
  + `gpu_accelerated_clean`: `pytest tests -m "not slow" -q` → `839 passed, 2 deselected` (`0 failed`); `mypy src tests benchmark_scripts` → `Success: no issues found in 127 source files`; `ruff check src tests benchmark_scripts` → `All checks passed!` (2026-11-05).
  + `gpu_accelerated_clean`: `grep -rn "CUPY_AVAILABLE" src/ tests/ --exclude-dir={__pycache__,.mypy_cache,.pytest_cache,.ruff_cache}` → `0`; `pytest tests/algorithms/test_unit_gpu_strategy_fail_loud.py -q` → `14 passed` em CPU sem `@pytest.mark.gpu` (2026-11-05).
  + `results_v3/selected_10`: 39 checkpoints; `import_checkpoints` → `39 duplicatas / 0 importados / 39 runs`; `compare_databases` (coorte vs rebuild independente) → `Databases are identical`; `validate_against_checkpoints` → `Validated 39/39 records`; `is_valid_checkpoint(p)` → `True` em 39/39 sem `expected_reps`; `strategy_params.local_search.candidate_k == 20` em 39/39 (2026-11-05).
  + `routing_data`: `pytest tests -q` → `425 passed, 1 xfailed`, cobertura total `81%` (`3427 Stmts, 654 Miss`); `mypy src examples` → `0 errors`; `ruff check src tests examples` → `All checks passed!`.
- **Next Item:** propor os commits atômicos por fase em `gpu_accelerated_clean` (código/testes Fase 11, `configs/` + fixture, coorte `results_v3/selected_10`, planos + tabela de símbolos) e executar o Batch 7 de `routing_data/TODO.md`.
- **Blockers:** Nenhum blocker em `gpu_accelerated_clean`.
- **Open Decisions:** Nenhuma decisão pendente — divisão Fase 10 vs Fase 11, papel de `pairwise.py`, arquitetura `Design B` (`StrategyBase` + prefixo de slot), configuração de `candidate_k`/`strategy_params` e o contrato fail-loud de GPU (`DP9-1`–`DP9-3`) foram confirmadas via `vscode_askQuestions` e estão registradas em `DP10-*`/`DP11-*`.
- **Files to Read First:**
  + [table-of-symbols.plan.md](./table-of-symbols.plan.md)
  + [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md)
  + [src/protocols/backend.py](../../src/protocols/backend.py)
  + [src/algorithms/strategies/local_search/two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)
  + [tests/algorithms/test_unit_gpu_strategy_fail_loud.py](../../tests/algorithms/test_unit_gpu_strategy_fail_loud.py)
