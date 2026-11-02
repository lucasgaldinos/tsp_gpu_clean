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
          customers em GeneticAlgorithm.evolve, isolamento de archive/ e resolução de D7/D8,
          contrato de configs, script check_archive_baseline e limpeza de tipagem.
status: complete
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
  + **Where:** [documents_tcc/plan/plan-reorganization-and-typing.md](./plan-reorganization-and-typing.plan.md), [documents_tcc/plan/plan-roteiro.md](./plan-roteiro.md).
  + **When:** Primeira ação da execução.
  + **How:** Criar o arquivo de plano com metadados e vincular na Tarefa 7 do roteiro.
  + **How much:** Baixo.
  + **Stakeholder Implications:** Nenhuma; rastreabilidade interna.
  + **DoD:** Arquivo criado e vinculado.
  + **Solution:**
    - **Changed Files:** [plan-reorganization-and-typing.md](./plan-reorganization-and-typing.plan.md)
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
      + [metaheuristics/**init**.py](../../src/algorithms/metaheuristics/__init__.py#L1-L20)
      + [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L40)
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
      + [test_modular_architecture.py](../../tests/test_modular_architecture.py#L1-L310)
      + [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L1-L50)
      + [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L1-L70)
      + [test_benchmark_selected_problems.py](../../tests/test_benchmark_selected_problems.py#L50-L100)
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
    - **Resumo:** Definido o tipo `SeedLike` em `src/protocols/types.py`; implementados `is_gpu_array`, `to_cpu_batch`, `restore_routes_shape(..., to_gpu=...)`, `make_rng` e `resolve_rng` em `src/utils/route_helpers.py`; refatoradas todas as estratégias em `src/algorithms/strategies/` para eliminar duplicações de conversão de tensores CuPy/NumPy e resolução de RNG.
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
      + [test_benchmark_selected_problems.py](../../tests/test_benchmark_selected_problems.py#L1-L180)
      + [test_modular_architecture.py](../../tests/test_modular_architecture.py#L315-L420)
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
  + **Where:** `tests/test_benchmark_selected_problems.py` (deletado), [test_benchmarking_pipeline.py](../../tests/test_benchmarking_pipeline.py#L1-L265), [test_cli.py](../../tests/test_cli.py#L1-L75), [test_modular_architecture.py](../../tests/test_modular_architecture.py#L75-L390), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L1-L55), [test_incremental_checkpoints.py](../../tests/validation/test_incremental_checkpoints.py#L1-L70).
  + **When:** Conclusão da Fase 8.
  + **How:** Substituir todas as referências diretas a `datasets/` por injeção de dependência via fixtures do pytest; verificar execução em `tmp_path`.
  + **How much:** Médio.
  + **Stakeholder Implications:** Suite de testes torna-se totalmente autossuficiente e roda em ambiente de CI/CD sem montagem de volumes externos.
  + **DoD:** 30 testes passando no `pytest` em tempo inferior a 15s; `mypy src tests benchmark_scripts` com 0 erros em 74 arquivos; `ruff check src tests benchmark_scripts` com 0 violações.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:**
      + `tests/test_benchmark_selected_problems.py` (removido)
      + [test_benchmarking_pipeline.py](../../tests/test_benchmarking_pipeline.py#L1-L265)
      + [test_cli.py](../../tests/test_cli.py#L1-L75)
      + [test_modular_architecture.py](../../tests/test_modular_architecture.py#L75-L390)
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
  + **When:** Conclusão da Fase 8.
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
    - **Atendimento ao DoD:** Defeito D8 resolvido; teste agora passa em conformidade com o DoD.

- [x] **Task 9.6: Migração e Reorganização de `archive/`**
  + **What:** Mover `full_gpu_monolithic.py` e `kernels/` para `src/algorithms/archive/`, criar `archive/__init__.py` expondo apenas `GeneticAlgorithmFullGPUMonolithic`, remover exportações de `algorithms/__init__.py`, apagar `monolithic/` e remover o alias `GeneticAlgorithmFullGPUEarlyStop`.
  + **Why:** O monolítico não faz parte da arquitetura modular e não deve ser anunciado na API do pacote; os 5 legados atuais importam módulos inexistentes e não podem ser importados.
  + **Who:** Agente (`implementer`).
  + **Where:** [archive/](../../src/algorithms/archive), [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L30), [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L240-L261).
  + **When:** Independente de 9.1-9.5; precede 9.7, 9.8 e 9.10.
  + **How:** Mover arquivos com `git mv`; corrigir a chave de estatísticas para `convergence_history`; escrever `archive/__init__.py` sem importar os legados; apagar os 5 módulos legados aprovados; registrar a origem no `README.md` do archive.
  + **How much:** Média.
  + **Stakeholder Implications:** Baseline continua executável por caminho explícito, sem contaminar `algorithms.__all__`.
  + **DoD:** `python -c "import algorithms, algorithms.archive"` funciona sem CUDA; `algorithms.__all__` não contém símbolos de archive; `grep -rn "monolithic" src/algorithms/__init__.py` retorna 0.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py#L240-L255), [archive/**init**.py](../../src/algorithms/archive/__init__.py#L1-L8), [archive/README.md](../../src/algorithms/archive/README.md#L1-L40), [algorithms/**init**.py](../../src/algorithms/__init__.py#L1-L25).
    - **Resumo:** `full_gpu_monolithic.py` e seus kernels migraram para `src/algorithms/archive/`. Os 5 arquivos legados não-importáveis foram removidos. O diretório `monolithic/` foi apagado. Corrigida a chave de estatísticas para `"convergence_history"` (resolvendo D7). Exportações monolíticas removidas de `algorithms.__init__.py`. Criado `README.md` com 10 tags hierárquicas e frontmatter conforme as regras.
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
  + **What:** Filtrar do teste de composição as variantes cujo `class_module` pertence a `algorithms.archive` e remover os dois `xfail(strict=True)` de D7.
  + **Why:** A regra do repositório proíbe testar `archive/`; manter as variantes parametrizadas executaria código arquivado e o xfail deixaria de ter alvo após a correção da chave.
  + **Who:** Agente (`implementer`).
  + **Where:** [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L160-L215).
  + **When:** Após Task 9.7.
  + **How:** Derivar a lista de variantes do fixture e excluir as resolvidas para `algorithms.archive`, documentando o motivo no docstring do teste.
  + **How much:** Baixa.
  + **Stakeholder Implications:** Nenhum.
  + **DoD:** `pytest tests/algorithms -q` verde; nenhum teste importa `algorithms.archive`.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test_integration_ga_composition.py](../../tests/algorithms/test_integration_ga_composition.py#L165-L180), [test_early_stopping.py](../../tests/validation/test_early_stopping.py#L35-L38).
    - **Resumo:** Variantes de `archive/` foram excluídas da parametrização de testes da suíte ativa e os dois decoradores `xfail(strict=True)` de D7 foram removidos de `test_gpu_variants_and_modular_vs_monolithic`. Em `test_early_stopping.py`, variantes cujo módulo contém `"archive"` são puladas.
    - **Atendimento ao DoD:** A suíte de testes não executa código de archive; todos os testes de composição e parada antecipada passam.

- [x] **Task 9.10: Script de Smoke do Baseline Arquivado**
  + **What:** Criar `benchmark_scripts/check_archive_baseline.py` que instancia o monolítico arquivado, executa uma evolução curta em burma14 e verifica `convergence_history`, `generations_completed` e `stop_reason`.
  + **Why:** O baseline continua necessário para comparação, mas não pode ser coberto pela suíte; sem verificação, a correção de D7 fica sem evidência executável.
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
  + **Where:** [test-suite-walkthrough.md](../../documentation/walkthroughs/test-suite-walkthrough.md), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md).
  + **When:** Última tarefa da fase.
  + **How:** Registrar evidências de verificação, decisões DP9-1 a DP9-7 e links para os arquivos alterados.
  + **How much:** Média.
  + **Stakeholder Implications:** Leitor da tese encontra a arquitetura real, não a anterior.
  + **DoD:** Walkthrough e Handoff descrevem o estado pós-fase; nenhuma referência a `tests/gpu_probe.py` como API.
  + **Solution / O que foi feito:**
    - **Arquivos Alterados:** [test-suite-walkthrough.md](../../documentation/walkthroughs/test-suite-walkthrough.md#L250-L280), [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md#L628-L790).
    - **Resumo:** Registrada a Seção 10.8 no walkthrough da suíte de testes documentando as decisões e números de Fase 9 (711 passed, 6 xfailed). Preenchidos todos os blocos Solution com links relativos e atualizado o bloco Handoff.
    - **Atendimento ao DoD:** Documentação sincronizada com o código e testes pós-Fase 9.

---

## Handoff

- **State:** Fases 1 a 9 concluídas e verificadas; 35 tarefas marcadas `[x]`, 0 tarefas abertas.
- **Last Verified Observation:**
  + `pytest tests -q`: `711 passed, 6 xfailed in 16.12s`.
  + `mypy src tests`: `Success: no issues found in 111 source files`.
  + `ruff check src tests benchmark_scripts`: `All checks passed!`.
  + `check_archive_baseline.py`: executado com sucesso no dispositivo CUDA (burma14: 3323.0 em 6 gerações, stop_reason='no_improvements').
- **Next Item:** Consolidação de experimentos e benchmarking comparativo (Capítulo 4 da tese).
- **Open Decisions:** `CUPY_CACHE_DIR` fallback no núcleo mantido em espera (hold) conforme instrução do usuário; cache dir configurado em `tests/conftest.py` para isolamento de ambiente.
- **Files to Read First:**
  + [src/protocols/backend.py](../../src/protocols/backend.py)
  + [src/algorithms/archive/README.md](../../src/algorithms/archive/README.md)
  + [benchmark_scripts/check_archive_baseline.py](../../benchmark_scripts/check_archive_baseline.py)
  + [src/protocols/problem_context.py](../../src/protocols/problem_context.py)
  + [src/utils/gpu_helpers.py](../../src/utils/gpu_helpers.py)
  + [src/algorithms/archive/full_gpu_monolithic.py](../../src/algorithms/archive/full_gpu_monolithic.py)
  + [src/benchmarking/runner/config_loader.py](../../src/benchmarking/runner/config_loader.py)
  + [configs/algorithms.json](../../configs/algorithms.json)
  + [tests/fixtures/configs/algorithms.json](../../tests/fixtures/configs/algorithms.json)
