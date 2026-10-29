---
title: "Plano de Reorganização Arquitetural, Tipagem PEP 695/698 e Benchmark Modular"
description: >
  Plano executivo 5W2H para consolidação do pacote `src/benchmarking/`, separação
  dos scripts CLI em `benchmark_scripts/`, padronização da raiz de importação `src/`,
  enriquecimento dos checkpoints com rotas e metadados, e tipagem pragmática Python 3.12.
created: 2026-10-28
status: completed
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: plan
scope: global
related_files:
  - [documents_tcc/plan/plan-roteiro.md](./plan-roteiro.md)
  - [pyproject.toml](../../pyproject.toml)
  - [configs/algorithms.json](../../configs/algorithms.json)
tags:
  - plan/architecture
  - plan/typing
  - plan/benchmark
  - benchmark/modular
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

## Decisões Fixadas (Source of Truth)

- **D1 (Tipagem)**: Não criar parâmetros de tipo genéricos em cascata (`SelectionStrategy[ArrayT]`, `ProblemContext[ArrayT]`). O `BackendModule` é um Protocolo não-genérico retornando `Array` (`NDArray[Any]`) e `arccos`.
- **D2 (Layout de Diretórios)**: O código de biblioteca reside em `src/benchmarking/` com submódulos `runner/`, `analysis/`, `database/`. Os scripts executáveis de linha de comando residem em `benchmark_scripts/` na raiz do repositório.
- **D3 (Importações)**: `pythonpath = ["src"]` estrito. Constantes compartilhadas como `DEFAULT_SELECTED_TSP_FILES` vivem na biblioteca (`src/benchmarking/runner/config_loader.py`), nunca em scripts CLI.
- **D4 (Checkpoints)**: Checkpoints JSON exportam `best_tour: list[int]`, `raw_best_tours: list[list[int]]`, `problem_type: str`, `edge_type: str` e o dicionário de classes das 6 estratégias (`initialization`, `selection`, `crossover`, `mutation`, `local_search`, `evaluation`).
- **D5 (Parada Precoce)**: `"hit_optimal"` ocorre apenas para `best_cost <= optimal_cost + 1e-6`. Gaps abaixo de 1% que não atingiram o ótimo registram `"within_1pct_optimal"`.

---

## Registro de Tarefas 5W2H

- [x] **[REVIEWED]** **Task 1.1: Registrar Plano 5W2H no Repositório**
  + **What:** Criar `documents_tcc/plan/plan-reorganization-and-typing.md` e vincular no `plan-roteiro.md`.
  + **Why:** Garantir rastreabilidade contínua das tarefas de engenharia de software dentro da documentação da tese.
  + **Who:** Agente (`implementer`).
  + **Where:** [documents_tcc/plan/plan-reorganization-and-typing.md](./plan-reorganization-and-typing.md), [documents_tcc/plan/plan-roteiro.md](./plan-roteiro.md).
  + **When:** Primeira ação da execução.
  + **How:** Criar o arquivo de plano com metadados e vincular na Tarefa 7 do roteiro.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Nenhuma; rastreabilidade interna.
  + **DoD:** Arquivo criado e vinculado.
  + **Solution:**
    - **Changed Files:** [plan-reorganization-and-typing.md](./plan-reorganization-and-typing.md)
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
      + [full_gpu_monolithic.py](../../src/algorithms/monolithic/full_gpu_monolithic.py#L153-L160)
      + [selection_strategies.py](../../src/algorithms/strategies/selection/selection_strategies.py#L11-L317)
      + [random_initialization.py](../../src/algorithms/strategies/initialization/random_initialization.py#L7-L88)
      + [evaluation_strategies.py](../../src/algorithms/strategies/evaluation/evaluation_strategies.py#L11-L189)
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
      + [full_gpu_monolithic.py](../../src/algorithms/monolithic/full_gpu_monolithic.py#L230-L241)
      + [algorithm_runner.py](../../src/benchmarking_v2/algorithm_runner.py#L242-L415)
      + [orchestration.py](../../src/benchmarking_v2/orchestration.py#L332-L545)
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
      + [aggregate_statistics.py](../../src/benchmarking_v2/aggregate_statistics.py#L98-L358)
      + [**init**.py](../../src/benchmarking_v2/__init__.py#L5-L44)
      + [test_benchmark_selected_problems.py](../../tests/test_benchmark_selected_problems.py#L12-L245)
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
      + [test_benchmark_selected_problems.py](../../tests/test_benchmark_selected_problems.py#L12-L27)
      + [test_modular_architecture.py](../../tests/test_modular_architecture.py#L35)
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
      + [presets.py](../../src/algorithms/presets.py#L1-L150)
      + [genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py#L1-L600)
      + [metaheuristics/__init__.py](../../src/algorithms/metaheuristics/__init__.py#L1-L20)
      + [algorithms/__init__.py](../../src/algorithms/__init__.py#L1-L40)
      + [algorithm_strategies.py](../../src/protocols/algorithm_strategies.py#L40-L60)
      + [matrix.py](../../src/distances/matrix.py#L20-L30)
      + [algorithms.json](../../configs/algorithms.json#L1-L47)
      + [config_loader.py](../../src/benchmarking/runner/config_loader.py#L200-L225)
      + [orchestration.py](../../src/benchmarking/runner/orchestration.py#L325-L390)
      + [run_selected_problems.py](../../benchmark_scripts/run_selected_problems.py#L50-L135)
      + [compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py#L1-L290)
      + [test_modular_architecture.py](../../tests/test_modular_architecture.py#L275-L288)
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L11-L25)
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L10-L15)
      + [local_search_stage_comparison.md](../../results_v3/local_search_comparison/tables/local_search_stage_comparison.md#L1-L47)
    - **Resumo:** Os presets `GeneticAlgorithmCPU`, `GeneticAlgorithmHybridNaive`, `GeneticAlgorithmHybridOptimized` e `GeneticAlgorithmFullGPUModular` foram movidos para `src/algorithms/presets.py`, limpando `src/algorithms/metaheuristics/` para conter apenas a metaheurística `GeneticAlgorithm`. O método `GeneticAlgorithm.evolve()` foi refatorado para usar uma pipeline ordenada de operadores (`LocalSearchStage`), eliminando condicionais acopladas. Corrigida a fase inicial de `post_evaluation` para avaliar a aptidão antes do 2-opt. Em `post_evolution`, a busca local agora refina a população inteira `(N, n_cities)` via `improve_batch` e `evaluate_batch`, explorando paralelamente 256 regiões na GPU. Criado o script `compare_local_search_stages.py`, que executou o benchmark comparativo em `att48`, `bayg29` e `bays29`. Os resultados mostram que `post_evaluation` atinge 0.00% de gap em `bayg29` (superando `post_mutation`) e reduz o tempo de computação em até 50% em `att48`.
    - **Atendimento ao DoD:** `pytest` passou com 19/19 testes aprovados em 10.52s. `mypy src benchmark_scripts` e `ruff check src tests benchmark_scripts` passaram com 0 erros. Tabelas Markdown e LaTeX geradas com sucesso em `results_v3/local_search_comparison/tables/`.

---

## Handoff

- **State:** Todas as tarefas do plano de reorganização, tipagem PEP 695/698, benchmarking modular e desacoplamento de estágios de busca local foram concluídas com êxito (Tasks 1.1, 1.2, 2.1, 2.2, 3.1, 4.1 e 5.1).
- **Last Verified Observation:**
  + `pytest`: 19 passed em 10.52s (0 erros, 0 falhas).
  + `mypy src benchmark_scripts`: 0 erros em modo estrito em todos os módulos.
  + `ruff check src tests benchmark_scripts`: 0 violações de lint.
  + `benchmark_scripts.compare_local_search_stages`: Concluído com sucesso gerando tabelas de comparação em Markdown e LaTeX em `results_v3/local_search_comparison/tables/`.
- **Next Item:** Execução de testes adicionais em instâncias de maior escala com as diferentes configurações de busca local, ou incorporação das tabelas comparativas ao Capítulo 4 do TCC.
- **Open Decisions:** Nenhuma decisão pendente; todas as definições arquiteturais e experimentais foram consolidadas.
- **Files to Read First:**
  + [src/algorithms/presets.py](../../src/algorithms/presets.py)
  + [src/algorithms/metaheuristics/genetic_algorithm.py](../../src/algorithms/metaheuristics/genetic_algorithm.py)
  + [benchmark_scripts/compare_local_search_stages.py](../../benchmark_scripts/compare_local_search_stages.py)
  + [results_v3/local_search_comparison/tables/local_search_stage_comparison.md](../../results_v3/local_search_comparison/tables/local_search_stage_comparison.md)
