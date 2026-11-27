---
title: "Plano de Documentação Arquitetural do Repositório"
description: >
  Plano estruturado em 5W2H para produzir quatro guias arquiteturais de subsistema,
  um blueprint mestre do projeto em nível Implementation-Ready e atualizar o catálogo
  de documentação e gates de integridade.
created: 2026-11-21
status: final
author:
  - "Lucas Galdino"
  - "Gemini 4 Hydrogen (High)"
type: plan
scope: global
modifications:
  - date_modified: 2026-11-21
    modifications:
      - description: >
          Criação do plano da Fase 15 com as decisões fixadas `DP15-1` a `DP15-3`,
          Lote 1 paralelo (Tasks 1.1 a 1.4 para guias de subsistema) e Lote 2 sequencial
          (Tasks 1.5 e 1.6 para blueprint mestre, catálogo e validação documental), além
          do registro da execução da Task 1.4.
      - description: >
          Atualização em Lote 2 com os blocos `Solution / O que foi feito` das Tasks 1.1,
          1.2, 1.3, 1.5 e 1.6 em estado `**[IN REVIEW]**`, alinhamento do caminho do
          blueprint mestre (`documentation/guide-project-architecture-blueprint.md`),
          atualização do catálogo `documentation/README.md` e registro de handoff para
          a revisão final do `reviewer-architect` (Wave 3).
      - description: >
          Fechamento final pós-aprovação da Wave 3b (`reviewer-architect`): atualização dos
          blocos `Solution / O que foi feito` com as correções da Wave 3, registro dos
          vereditos `Review Verdict`, marcação de todas as 6 tarefas como `- [x] **[REVIEWED]**`,
          transição para `status: final` e `## Handoff` com `State: done`.
related_files:
  - "[table-of-symbols.plan.md](./table-of-symbols.plan.md)"
  - "[AGENTS.md](../../AGENTS.md)"
  - "[README.md](../../documentation/README.md)"
  - "[design-patterns-links.md](../../design-patterns-links.md)"
  - "[guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md)"
  - "[guide-architecture-protocols-and-data-models.md](../../documentation/analysis/guide-architecture-protocols-and-data-models.md)"
  - "[guide-architecture-loaders-and-distances.md](../../documentation/analysis/guide-architecture-loaders-and-distances.md)"
  - "[guide-architecture-algorithms-and-strategies.md](../../documentation/analysis/guide-architecture-algorithms-and-strategies.md)"
  - "[guide-architecture-benchmarking-cli-and-utils.md](../../documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md)"
tags:
  - analysis/planning
  - analysis/architecture
  - guide/documentation
  - guide/architecture
  - benchmark/runner
  - algorithm/genetic-algorithm
  - review/quality
  - notes/planning
  - notes/session
  - notes/handoff
---

# Plano de Documentação Arquitetural do Repositório

## Escopo e regras

Este plano governa a criação da suíte de documentação arquitetural (`Fase 15`) para os pacotes Python e CUDA em `src/` e seus contratos de configuração em `configs/`.

- **Escopo permitido:** criação dos quatro guias de subsistema em `documentation/analysis/`, do blueprint mestre em [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md), atualização do catálogo em [documentation/README.md](../../documentation/README.md) e registro de identificadores em [table-of-symbols.plan.md](./table-of-symbols.plan.md).
- **Escopo proibido:** nenhuma alteração em código de produção (`src/`), testes (`tests/`), capítulos da dissertação (`documents_tcc/tex/`) nem execução de `git commit`/`git push` sem autorização explícita.
- **Regras de qualidade e idioma:** este plano segue português brasileiro natural ([prose-language.instructions.md](../../.agents/rules/prose-language.instructions.md)) e formato 5W2H ([task-managing](../../.agents/skills/task-managing/SKILL.md)); os guias em `documentation/` e `documentation/analysis/` são redigidos em inglês com nível `Implementation-Ready` (`architecture-blueprint-generator`), fórmulas KaTeX, mapeamento GoF para [design-patterns-links.md](../../design-patterns-links.md), diagramas Mermaid com frontmatter YAML interno e âncoras por símbolo (`arquivo.py::Simbolo`) verificadas por `benchmark_scripts.check_docs`.

## Fonte da verdade

As decisões arquiteturais e defeitos citados nos guias arquiteturais devem corresponder exatamente ao registro único em [table-of-symbols.plan.md](./table-of-symbols.plan.md) e ao código-fonte verificado em `src/`. Qualquer divergência entre documentos anteriores e a implementação atual em `src/` resolve-se a favor do código verificado.

## Decisões fixadas

- `DP15-1`: Estruturação em **4 guias de subsistema** (`protocols + data_models`, `loaders + distances`, `algorithms`, e `benchmarking + cli + utils` em `documentation/analysis/`) mais **1 blueprint mestre** ([guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md) na raiz de `documentation/`), todos em profundidade `Implementation-Ready`.
- `DP15-2`: Separação idiomática estrita e indexação simétrica — este plano e seus blocos `Solution / O que foi feito` ficam em `pt-BR` natural; todos os guias em `documentation/` e `documentation/analysis/` ficam em inglês; e [documentation/README.md](../../documentation/README.md) mantém contagem, índice e `related_files` atualizados.
- `DP15-3`: Rastreabilidade mecânica obrigatória via âncoras `arquivo::Simbolo` (zero `#L` em código), equações KaTeX para modelos matemáticos/estatísticos, mapeamento GoF explícito e aprovação integral em `rtk .venv/bin/python -m benchmark_scripts.check_docs`.

## Lote 1 — Guias de subsistema (Paralelo)

Todas as tarefas do Lote 1 operam em arquivos disjuntos após o contrato de padronização do revisor e podem ser executadas em paralelo (`worker-solver-core` para Tasks 1.1–1.3; `worker-bench-ops` para Task 1.4).

- [x] **[REVIEWED]** **Task 1.1: Documentar arquitetura de `protocols` e `data_models`**

  - **O quê:** Criar o guia arquitetural `documentation/analysis/guide-architecture-protocols-and-data-models.md` cobrindo `src/protocols/` (`backend.py`, `problem_context.py`, `algorithm_strategies.py`, `types.py`) e `src/data_models/` (`problem.py`, `records.py`, `exceptions.py`).
  - **Por quê:** Os contratos estruturais (`Protocol`, `TypeAliasType`, `TypedDict`), a detecção tardia de GPU (`GPUUnavailableError`), o cache de matrizes/listas $k$-NN em `ProblemContext` e a validação `EUC_3D` em `Problem` formam a camada fundacional de todo o repositório, mas carecem de um blueprint consolidado.
  - **Quem:** Agent (`worker-solver-core`).
  - **Onde:** [protocols/__init__.py](../../src/protocols/__init__.py), [data_models/__init__.py](../../src/data_models/__init__.py).
  - **Quando:** Sem dependências; paralelo no Lote 1.
  - **Como:** Analisar todos os símbolos públicos/internos, mapear dependências, escrever as 8 seções `Implementation-Ready` em inglês com Mermaid (`flowchart`, `classDiagram`/`sequenceDiagram`), fórmulas KaTeX de VRAM/memória, mapeamento GoF e ADRs (`DP-1`, `DP9-1`, `DP10-15`, `DP10-17`, `DP10-19`, `DP10-20`, `DP14-6`).
  - **Esforço:** Médio.
  - **Impacto:** Estabelece referência única de contratos de tipos e ciclo de vida de contexto para todas as camadas superiores.
  - **DoD:** Arquivo criado em conformidade com `markdown.instructions.md`, com 10–15 tags, âncoras `::` válidas em `src/`, diagramas Mermaid válidos e digesto enviado para a Task 1.5.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [guide-architecture-protocols-and-data-models.md](../../documentation/analysis/guide-architecture-protocols-and-data-models.md) — criado o guia completo (464 linhas, 10 tags, 3 diagramas Mermaid) referenciando [backend.py::BackendModule, GPUCapabilityProbe, gpu_available, require_gpu](../../src/protocols/backend.py), [problem_context.py::ProblemContext, DEFAULT_MAX_MATRIX_DIMENSION](../../src/protocols/problem_context.py), [problem.py::Problem, expected_coordinate_dimension](../../src/data_models/problem.py), [exceptions.py::GPUUnavailableError, VRAMInsufficientError](../../src/data_models/exceptions.py) e [records.py::BenchmarkRunInput, NemenyiResult](../../src/data_models/records.py).
    - **Resumo:** Documentados os protocolos PEP 544 e aliases PEP 695 de `src/protocols/`, o ciclo de vida tardio do backend CUDA, o cache de distâncias/candidatos $k$-NN com guardrails de VRAM em `ProblemContext`, a propriedade `Problem.is_symmetric`, `ProblemContext.is_closed`/`h2d_bytes` no `classDiagram` de §5.1, `EUC_3D` (`DP14-6`), título YAML sem crases e `related_files` simétricos para os 4 guias.
    - **DoD atendido:** Guia validado com 0 achados em `benchmark_scripts.check_docs`, 3/3 diagramas revalidados via `mmdc` e âncoras `::` verificadas contra a AST de `src/`.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: 8 seções Implementation-Ready, YAML válido (10 tags, 44 chars), GoF, KaTeX de VRAM, 3 Mermaid válidos e âncoras verificadas em src/protocols e src/data_models.

- [x] **[REVIEWED]** **Task 1.2: Documentar arquitetura de `loaders` e `distances`**

  - **O quê:** Criar o guia arquitetural `documentation/analysis/guide-architecture-loaders-and-distances.md` cobrindo `src/loaders/` (`tsplib_loader.py`, `database_loader.py`, `problem_type_handlers.py`) e `src/distances/` (`pairwise.py`, `matrix.py`, `kernels/pairwise_device.cuh`, `kernels/two_opt_common.cuh`).
  - **Por quê:** A fronteira ETL com o pacote externo `routing_data`, o cálculo vetorizado em blocos (*row-chunked*), a paridade escalar/CUDA (`GEO` truncado vs `EUC_2D`/`EUC_3D`/`ATT`) e a decisão de materialização ($n \le 10\,000$ vs *on-the-fly*) contêm invariantes críticos espalhados em múltiplos módulos.
  - **Quem:** Agent (`worker-solver-core`).
  - **Onde:** [loaders/__init__.py](../../src/loaders/__init__.py), [distances/__init__.py](../../src/distances/__init__.py).
  - **Quando:** Sem dependências; paralelo no Lote 1.
  - **Como:** Mapear contratos de ingestão TSPLIB/DuckDB, fluxo de construção de `Problem`, kernels em `pairwise_device.cuh`, diagramas Mermaid, equações KaTeX de distância e uso de memória, padrões GoF e ADRs (`DP10-6`, `DP10-12`, `DP10-13`, `DP10-14`, `DP10-16`, `DP11-3`, `DP14-6`, `BUG-32`).
  - **Esforço:** Médio.
  - **Impacto:** Documenta o contrato único de distância e ingestão usado tanto na CPU quanto nos kernels CUDA.
  - **DoD:** Arquivo criado com todas as 8 seções em inglês, âncoras verificadas, fórmulas TSPLIB95 em KaTeX e digesto entregue para a Task 1.5.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [guide-architecture-loaders-and-distances.md](../../documentation/analysis/guide-architecture-loaders-and-distances.md) — criado o guia completo (374 linhas, 10 tags, 3 diagramas Mermaid) ancorado em [database_loader.py::DatabaseLoader](../../src/loaders/database_loader.py), [problem_type_handlers.py::PROBLEM_TYPE_HANDLERS, ProblemTypeHandler](../../src/loaders/problem_type_handlers.py), [tsplib_reader.py::TsplibLoader, bundle_to_problem](../../src/loaders/tsplib_reader.py), [tsplib_models.py::parse_tsplib_file](../../src/loaders/tsplib_models.py), [pairwise.py::_to_radians_geo, compute_geographical, get_distance_function](../../src/distances/pairwise.py), [matrix.py::compute_distance_matrix, compute_candidate_list](../../src/distances/matrix.py), [pairwise_device.cuh::gpu_pairwise_distance, gpu_geo_radians](../../src/distances/kernels/pairwise_device.cuh) e [two_opt_common.cuh::two_opt_delta](../../src/distances/kernels/two_opt_common.cuh).
    - **Resumo:** Documentada a ingestão polimórfica via `PROBLEM_TYPE_HANDLERS`, a fronteira com `routing_data`, o cálculo matricial em blocos (`chunk_size=512`), o patch canônico de `GEO` (`BUG-32`), a arquitetura em três níveis de distância até os registradores CUDA, título YAML sem crases e `related_files` simétricos.
    - **DoD atendido:** Guia com 8 seções e equações TSPLIB95 em KaTeX, 3/3 diagramas aprovados via `mmdc` e 0 erros em `benchmark_scripts.check_docs`.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: 8 seções em inglês (40 chars, 10 tags), 3 tiers de distância e patch BUG-32 de GEO verificados contra src/loaders e src/distances, GoF, KaTeX e 3 Mermaid.

- [x] **[REVIEWED]** **Task 1.3: Documentar arquitetura de `algorithms` e kernels CUDA**

  - **O quê:** Criar o guia arquitetural `documentation/analysis/guide-architecture-algorithms-and-strategies.md` cobrindo `src/algorithms/` (`metaheuristics/genetic_algorithm.py`, `pipeline_steps.py`, `strategies/`, `kernels/` e `archive/`).
  - **Por quê:** A composição modular do `GeneticAlgorithm` (`Design B`), os 6 slots de estratégia, o despacho de `LocalSearchStage`, a geração 0 crua (`DP11-8`), os kernels cooperativos (`OX-A`, `SWAP-A`, `EVAL-A`, `SEL-A`, `INV-A`, `2OPT-A`) e o isolamento do legado em `archive/` precisam de especificação técnica unificada.
  - **Quem:** Agent (`worker-solver-core`).
  - **Onde:** [algorithms/__init__.py](../../src/algorithms/__init__.py), [genetic_algorithm.py::GeneticAlgorithm](../../src/algorithms/metaheuristics/genetic_algorithm.py).
  - **Quando:** Sem dependências; paralelo no Lote 1.
  - **Como:** Documentar registros (`STRATEGY_REGISTRY`, `PIPELINE_STEP_REGISTRY`), contratos de residência GPU/CPU, etapas de pipeline, tiling de memória compartilhada (`TILE_SIZE`), mapeamento GoF, diagramas Mermaid de fluxo geracional e ADRs (`DP9-4`, `DP11-1`, `DP11-8`, `DP11-9`, `DP11-10`, `DP11-11`, `DP11-14` a `DP11-20`, `DP12-12` a `DP12-14`, `DP13-5`, `DP13-6`).
  - **Esforço:** Médio.
  - **Impacto:** Consolida o núcleo algorítmico e de aceleração CUDA em um guia único atualizado pós-Fase 14.
  - **DoD:** Guia criado com 8 seções em inglês, âncoras `.py::` e `.cu::` verificadas, equações KaTeX de memória/complexidade e digesto entregue para a Task 1.5.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [guide-architecture-algorithms-and-strategies.md](../../documentation/analysis/guide-architecture-algorithms-and-strategies.md) — criado o guia completo (416 linhas, 10 tags, 3 diagramas Mermaid) referenciando [genetic_algorithm.py::GeneticAlgorithm, GeneticAlgorithm._collect_strategy_metrics, STAGE_SPECS, GenerationStep](../../src/algorithms/metaheuristics/genetic_algorithm.py), [pipeline_steps.py::PIPELINE_STEP_REGISTRY, PostInitLocalSearchStep, EliteOnlyLocalSearchStep, PeriodicLocalSearchStep](../../src/algorithms/pipeline_steps.py), [base.py::StrategyBase](../../src/algorithms/strategies/base.py), [__init__.py::STRATEGY_REGISTRY](../../src/algorithms/strategies/__init__.py), [gpu_base.py::DistanceOperandCache, DistanceOperandCache.resolve, DistanceOperandCache.resolve_candidates](../../src/algorithms/strategies/local_search/gpu_base.py), [base.py::EvaluationGPUBase._distance_operands](../../src/algorithms/strategies/evaluation/base.py) e [full_gpu_monolithic.py::GeneticAlgorithmFullGPUMonolithic](../../src/algorithms/archive/full_gpu_monolithic.py).
    - **Resumo:** Detalhados o despacho de `STAGE_SPECS`, geração 0 crua (`DP11-8`), etapas de pipeline (`DP11-6`, `DP11-10`), tabela em SRAM do `order_crossover_kernel` (`DP11-11`), tiling de 2-opt e §6 reescrita para atribuição única do delta de `context.h2d_bytes` (`_ = context` em `_collect_strategy_metrics`), além de título YAML sem crases e `related_files` simétricos.
    - **DoD atendido:** Guia validado com 0 achados em `benchmark_scripts.check_docs`, 3/3 diagramas aprovados via `mmdc` e âncoras Python/CUDA verificadas.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: §6 corrigida para refletir _ = context e atribuição única do delta de context.h2d_bytes em two_opt.py/cost_evaluation.py; 8 seções, 3 Mermaid, KaTeX, GoF e simetria de related_files verificados.

- [x] **[REVIEWED]** **Task 1.4: Documentar arquitetura de `benchmarking`, `cli`, `utils` e `configs`**

  - **O quê:** Criar o guia arquitetural [guide-architecture-benchmarking-cli-and-utils.md](../../documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md) cobrindo `src/benchmarking/`, `src/cli/`, `src/utils/` e `configs/schemas/`.
  - **Por quê:** A infraestrutura de execução, telemetria sem barreiras (`DP11-22`), warmup AOT isolado (`DP11-21`), checkpoints atômicos (`DP12-7`), schema reflexivo (`BUG-85`), persistência em DuckDB e análise estatística Demšar (`DP13-3`, `DP13-4`) concentram acoplamentos transversais (incluindo ciclos preguiçosos em `analysis/`) que exigem registro preciso.
  - **Quem:** Agent (`worker-bench-ops`).
  - **Onde:** [guide-architecture-benchmarking-cli-and-utils.md](../../documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md), [benchmarking/__init__.py](../../src/benchmarking/__init__.py), [cli/benchmark.py::main](../../src/cli/benchmark.py), [utils/__init__.py](../../src/utils/__init__.py).
  - **Quando:** Sem dependências; paralelo no Lote 1.
  - **Como:** Mapear todas as dependências e acoplamentos, documentar contratos verificados de `runner/`, `database/`, `analysis/`, `cli/` e `utils/`, mapear padrões GoF, diagramar sequência de execução e ER do DuckDB em Mermaid, formalizar equações estatísticas em KaTeX e consolidar receitas de extensão e ADRs.
  - **Esforço:** Médio.
  - **Impacto:** Torna explícitos os contratos de telemetria, inferência não-paramétrica, schema JSON e CLI para manutenção futura.
  - **DoD:** Guia criado com as 8 seções em inglês, âncoras `::` válidas, diagramas Mermaid com YAML interno e digesto enviado a `worker-solver-core`.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [guide-architecture-benchmarking-cli-and-utils.md](../../documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md) — criado o guia completo (486 linhas, 8 seções `Implementation-Ready`, 12 tags, 3 diagramas Mermaid) com `related_files` simétricos para [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md) e os 3 guias irmãos em `analysis/`.
      + [table-of-symbols.plan.md](./table-of-symbols.plan.md) — registradas as decisões `DP15-1` a `DP15-3`.
    - **Resumo:** Documentados `src/benchmarking/`, `src/cli/`, `src/utils/` e `configs/schemas/` em inglês com grafo de dependências (expondo o ciclo preguiçoso em `analysis/`, o import privado de `_compile_kernel_cached` e a cobertura parcial do barrel `utils/__init__.py`), mapeamento GoF, diagramas Mermaid, modelos KaTeX e 13 ADRs.
    - **DoD atendido:** 182 links/âncoras `::` verificados com 0 erros via AST, 3/3 diagramas aprovados via `mmdc` (`73,6 KB`, `40,6 KB`, `133,8 KB`), `check_docs` com código `0` e digesto enviado a `worker-solver-core`.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: 8 seções (45 chars, 12 tags), ciclo em analysis/, _compile_kernel_cached, GoF, KaTeX Demšar/ddof=1, 3 Mermaid e âncoras verificados em src/{benchmarking,cli,utils}.

## Lote 2 — Blueprint mestre e catálogo (Sequencial)

As tarefas do Lote 2 dependem da conclusão e dos digestos do Lote 1 e executam em sequência (`Task 1.5` $\to$ `Task 1.6`).

- [x] **[REVIEWED]** **Task 1.5: Sintetizar o Blueprint Mestre da Arquitetura (`guide-project-architecture-blueprint.md`)**

  - **O quê:** Criar [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md) consolidando a arquitetura global do projeto a partir dos quatro guias do Lote 1, em nível `Implementation-Ready`.
  - **Por quê:** Os quatro guias detalham cada subsistema isoladamente; falta uma visão sistêmica em camadas e fluxo ponta a ponta que conecte CLI, runner, loaders, protocolos, algoritmos/CUDA, persistência DuckDB e relatórios estatísticos, registrando violações de camada e guia de evolução.
  - **Quem:** Agent (`worker-solver-core`).
  - **Onde:** [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md), [documentation/README.md](../../documentation/README.md).
  - **Quando:** Depende da conclusão e dos digestos das Tasks 1.1, 1.2, 1.3 e 1.4.
  - **Como:** Integrar os quatro guias com links cruzados e `related_files` simétricos, construir diagramas Mermaid sistêmicos (arquitetura em camadas + fluxo de dados ponta a ponta), matriz consolidada de padrões GoF, preocupações transversais globais, roadmap de desacoplamento e tabela consolidada de ADRs.
  - **Esforço:** Médio.
  - **Impacto:** Entrega o documento central de referência arquitetural para novos desenvolvedores e para a documentação do TCC.
  - **DoD:** Blueprint mestre criado em inglês com links cruzados válidos para os 4 guias de subsistema, zero referências inventadas e conformidade com `markdown.instructions.md`.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md) — criado o blueprint mestre (568 linhas, 13 tags, 4 diagramas Mermaid, 17 seções `architecture-blueprint-generator`) consolidando os 8 pacotes de `src/` e integrando o digesto da Task 1.4.
    - **Resumo:** Sintetizada a arquitetura global com matriz GoF unificada, regras de dependência entre camadas, anomalias transversais (ciclo em `analysis/`, acoplamento `records.py` e `_compile_kernel_cached`), fluxos CUDA/DuckDB/Demšar e links bidirecionais para os 4 guias sob `analysis/`.
    - **DoD atendido:** 4/4 diagramas Mermaid aprovados via `mmdc`, 0 achados em `benchmark_scripts.check_docs` e vínculos `related_files` verificados.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: 17 seções Implementation-Ready (36 chars, 13 tags), 4 Mermaid, GoF, KaTeX, links mútuos com os 4 guias e DP10–DP15 correlacionados com table-of-symbols.plan.md.

- [x] **[REVIEWED]** **Task 1.6: Atualizar catálogo, tabela de símbolos e validar gates documentais**

  - **O quê:** Atualizar [documentation/README.md](../../documentation/README.md) com os cinco novos guias arquiteturais (`related_files`, contagens e links), atualizar este plano com os blocos `Solution / O que foi feito` e executar a suíte de validação documental e de diagramas Mermaid.
  - **Por quê:** Novos arquivos em `documentation/` e `documentation/analysis/` não indexados no catálogo mestre geram deriva de navegação, e qualquer erro de âncora GFM, tabela MD056 ou símbolo `DP15-*` quebra o gate automatizado do repositório.
  - **Quem:** Agent (`worker-bench-ops`).
  - **Onde:** [documentation/README.md](../../documentation/README.md), [check_docs.py::main](../../benchmark_scripts/check_docs.py), [table-of-symbols.plan.md](./table-of-symbols.plan.md).
  - **Quando:** Depende da conclusão da Task 1.5; antecede a revisão crítica final (Wave 3).
  - **Como:** Atualizar o índice e `related_files` em `documentation/README.md` (contando 2 guias na raiz e 16 arquivos em `analysis/`), validar blocos Mermaid de Task 1.4 com `mmdc`, sincronizar `DP15-1`–`DP15-3` em `table-of-symbols.plan.md`, preencher `Solution / O que foi feito` neste plano e rodar `rtk .venv/bin/python -m benchmark_scripts.check_docs` exigindo código de saída `0`.
  - **Esforço:** Baixo.
  - **Impacto:** Garante integridade bidirecional do grafo de documentação e conformidade mecânica com todos os gates antes da revisão final.
  - **DoD:** `rtk .venv/bin/python -m benchmark_scripts.check_docs` retorna código `0` sem achados; `documentation/README.md` lista o blueprint mestre e os 4 guias de subsistema com contagens atualizadas; e `## Handoff` aponta para a revisão crítica final (Wave 3).

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [README.md](../../documentation/README.md) — atualizados `Directory overview` (`2` guias na raiz, `16` arquivos em `analysis/`), seção `1.`, subseção `Architecture subsystem guides` em `3.`, links `useful_notes/` corrigidos para `useful-notes/` (`89` links verificados, `0` mortos), `modifications` e `related_files` (`167` linhas).
      + [guide-architecture-benchmarking-cli-and-utils.md](../../documentation/analysis/guide-architecture-benchmarking-cli-and-utils.md) — adicionados vínculos `related_files` simétricos para [guide-project-architecture-blueprint.md](../../documentation/guide-project-architecture-blueprint.md) e os 3 guias irmãos.
      + [table-of-symbols.plan.md](./table-of-symbols.plan.md) e [plan-architecture-documentation.plan.md](./plan-architecture-documentation.plan.md) — sincronizados `DP15-1`–`DP15-3`, blocos `Solution`, vereditos e estado `[x] **[REVIEWED]**`.
    - **Resumo:** Catálogo mestre, links de `useful-notes/`, `related_files` simétricos, `table-of-symbols.plan.md` e plano da Fase 15 consolidados com vereditos aprovados da Wave 3b.
    - **DoD atendido:** `README_LINKS_CHECKED=89 DEAD=0`; `mmdc` validou 3/3 diagramas da Task 1.4 (`rc=0`); e `rtk .venv/bin/python -m benchmark_scripts.check_docs` retornou saída `0` após a última edição.
    - **Review Verdict (2026-11-21, reviewer):** Aprovada. DoD atendido: links useful_notes/ corrigidos para useful-notes/ em documentation/README.md (89 links, 0 mortos reportados), DP15-1..DP15-3 e plano pt-BR sincronizados e check_docs com saída 0 reportada.

## Handoff

- **State**: done
- **Last verified observation**: `rtk .venv/bin/python -m benchmark_scripts.check_docs` -> `OK: no documentation drift detected` (exit 0, 2026-11-21)
- **Next item**: Nenhum (arquitetura documentada e validada; revisão do diff e commit mediante aprovação do usuário)
- **Open decisions**: Nenhuma | N/A | N/A
- **Read first**: `plan-architecture-documentation.plan.md#lote-2--blueprint-mestre-e-catálogo-sequencial`, `documentation/guide-project-architecture-blueprint.md#1-architecture-detection--analysis`, `documentation/README.md#1-root-architecture--pipeline-guides`, `table-of-symbols.plan.md#fase-15-dp15`
