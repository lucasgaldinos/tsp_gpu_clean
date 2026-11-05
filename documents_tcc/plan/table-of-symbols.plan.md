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
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Adição das decisões `DP10-6` a `DP10-20` e dos defeitos `BUG-13` a `BUG-16`
          decorrentes da revisão comparativa entre `gpu_accelerated_clean` e `routing_data`,
          e reconciliação de âncoras de linha da Fase 10 com o plano operacional.
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
| `DP10-1` | Identificadores padronizados: `DP-<n>` decisões, `DP<fase>-<n>` decisões de fase, `BUG-<n>` defeitos, `TASK <fase>.<n>` tarefas. | [plan#L853](./plan-reorganization-and-typing.plan.md#L853) |
| `DP10-2` | Variantes monolíticas saem do config ativo e vivem em `configs/algorithms.archive.json`, carregado sob demanda. | [plan#L854](./plan-reorganization-and-typing.plan.md#L854) |
| `DP10-3` | `FullGPUMonolithic` é o nome único da variante monolítica; `FullGPU` foi eliminado. | [plan#L855](./plan-reorganization-and-typing.plan.md#L855) |
| `DP10-4` | `benchmark_scripts/check_docs.py` é o gate standalone de docstrings, links e símbolos. | [plan#L856](./plan-reorganization-and-typing.plan.md#L856) |
| `DP10-5` | Checkpoints de schemas diferentes não são combinados na mesma pasta de resultados. | [plan#L857](./plan-reorganization-and-typing.plan.md#L857) |
| `DP10-6` | Substituição de `tsplib95` e do ETL duplicado em `tsplib_loader.py` pelo pacote `routing-data` (`tsplib_parser` + `converter`). | [plan#L858](./plan-reorganization-and-typing.plan.md#L858) |
| `DP10-7` | Consumo de `routing-data` via tag de release Git (`uv add "routing-data @ git+https://github.com/lucasgaldinos/Routing_data.git@v0.2.0"`) e relaxamento de `duckdb>=1.4.0`. | [plan#L859](./plan-reorganization-and-typing.plan.md#L859) |
| `DP10-8` | Contratos tipados explícitos em `distances/pairwise.py` (`PairwiseDistanceFn`), `loaders/problem_type_handlers.py` (`ProblemMetadataRow`) e `protocols/backend.py` (sem monkey-patching `# type: ignore`). | [plan#L860](./plan-reorganization-and-typing.plan.md#L860) |
| `DP10-9` | Correções aprovadas em `routing_data`: upsert em `insert_problem` (`B1`), isolamento por problema em `insert_problems_batch` (`B3`), rejeição completa de problema `EXPLICIT` sem matriz (`M4`), simetria por valores via `check_symmetry` (`M7`), validação SQL real em `converter validate` (`M11`), arquivos JSON `<name>.<type>.json` sem duplicatas (`M2`/`m17`), e `DataTransformer` → `ProblemNormalizer` (`T1`–`T5`). | [plan#L861](./plan-reorganization-and-typing.plan.md#L861) |
| `DP10-10` | Higiene de documentação e scripts: deleção dos arquivos temporários/duplicados, deleção dos 9 scripts de migração em `routing_data/tmp/` e refatoração dos 4 `probe_*.py` em exemplos após estabilização da API. | [plan#L869](./plan-reorganization-and-typing.plan.md#L869) |
| `DP10-11` | `DatabaseLoader.__enter__` verifica a existência do arquivo antes do bloco `try` e levanta `DatabaseConnectionError` diretamente sem auto-captura de `FileNotFoundError` (`BUG-16`). | [plan#L870](./plan-reorganization-and-typing.plan.md#L870) |
| `DP10-12` | Manutenção de `edge_weight_matrices.matrix` como `INTEGER[][]` e `int32` em todo o pipeline segundo a regra integral da TSPLIB95 (`_nint`), registrando os trade-offs futuros de escala em [analysis-future-scaling-and-type-decisions.md](../../documentation/analysis/questions/analysis-future-scaling-and-type-decisions.md) (`M3`). | [plan#L871](./plan-reorganization-and-typing.plan.md#L871) |
| `DP10-13` | Funções escalares TSPLIB95 (`_nint`, `EUC_2D`, `GEO`, `ATT`, etc.) residem em `routing_data` e são reexportadas por `distances/pairwise.py`; o construtor matricial vetorizado `xp` (`numpy`/`cupy`) permanece em `distances/matrix.py`. | [plan#L872](./plan-reorganization-and-typing.plan.md#L872) |
| `DP10-14` | `DatabaseManager` (`routing_data`) detém 100% do DDL e queries SQL do corpus; `DatabaseLoader` (`gpu_accelerated_clean`) converte os registros retornados em objetos `Problem` / `np.ndarray` (`int32`). | [plan#L873](./plan-reorganization-and-typing.plan.md#L873) |
| `DP10-15` | Padronização de `ProblemContext.rng` em `np.random.Generator` (`make_rng(seed)`), eliminando `RandomState` do contexto e resolvendo `BUG-2`. | [plan#L874](./plan-reorganization-and-typing.plan.md#L874) |
| `DP10-16` | Cálculo matricial em `distances/matrix.py` usa broadcast 2D direto para `n <= 2048` e broadcast em blocos de linhas (row-chunked, `chunk_size=512`) para `n > 2048`, compartilhando constantes com `pairwise.py` (`BUG-13`). | [plan#L875](./plan-reorganization-and-typing.plan.md#L875) |
| `DP10-17` | Cache único da matriz de distâncias `int32` em GPU centralizado em `ProblemContext` (`get_gpu_distances`), com chave composta `(name, dimension, edge_type, id(problem))` como fallback nas estratégias (`BUG-14`). | [plan#L876](./plan-reorganization-and-typing.plan.md#L876) |
| `DP10-18` | Remoção do `cp.cuda.Stream.null.synchronize()` redundante em `BatchGPUTwoOpt` e `BatchGPUMultiMoveTwoOpt` quando `input_on_gpu=True`, preservando sincronização automática em D2H (`.get()`) e em `@time_strategy` (`BUG-15`). | [plan#L877](./plan-reorganization-and-typing.plan.md#L877) |
| `DP10-19` | Acima de `n > 10.000` (métricas baseadas em coordenadas), as estratégias GPU calculam distâncias on-the-fly em registradores via `pairwise_device.cuh` sem alocar a matriz `(n, n)` na VRAM; acima de `n > 11.520`, os kernels de 2-opt usam tiling em memória compartilhada (`TILE_SIZE = 2048`) para respeitar o teto de 48 KB por bloco; `ProblemContext` valida a VRAM disponível e levanta `VRAMInsufficientError` quando necessário. | [plan#L878](./plan-reorganization-and-typing.plan.md#L878) |
| `DP10-20` | Lista de candidatos $k$-Nearest Neighbor ($\mathcal{N}_k \in \mathbb{Z}^{n \times k}$, Reinelt, 1994; Helsgaun, 2000; Toth & Vigo, 2003) pré-computada e cacheada em `ProblemContext` (`get_cpu_candidate_list` / `get_gpu_candidate_list`), reutilizável por 2-opt em CPU e GPU ($\Theta(n \cdot k)$ com mapa inverso $\text{pos}(u)$) e por operadores futuros (3-opt/Or-opt, vizinhanças granulares CVRP, inicialização GRASP e mutação guiada). | [plan#L879](./plan-reorganization-and-typing.plan.md#L879) |

### Fase 11 (`DP11`)

| ID | Decisão | Evidência |
| -- | ------- | ---- |
| `DP11-1` | Arquitetura de classes de estratégia por Família + Prefixo de Slot (`Design B`) herdando de `StrategyBase` (`InitializationRandom`, `InitializationNearestNeighbor`, `SelectionTournament`, `SelectionRouletteWheel`, `SelectionGPUTournament`, `CrossoverOrder`, `CrossoverPartiallyMapped`, `CrossoverGPUOrder`, `MutationSwap`, `MutationInversion`, `MutationGPUSwap`, `MutationGPUInversion`, `LocalSearchCPUTwoOpt(mode=...)`, `LocalSearchGPUTwoOpt(mode=...)`, `LocalSearchNoOp`, `EvaluationCPUCost`, `EvaluationGPUCost`). | [plan#L1369](./plan-reorganization-and-typing.plan.md#L1369) |
| `DP11-2` | Remoção do limite estático de 16 movimentos (`applied_start[16]`) em `two_opt_multi_move.cu` para paridade com o modo `multi_move` em CPU (`BUG-20`). | [plan#L1378](./plan-reorganization-and-typing.plan.md#L1378) |
| `DP11-3` | Unificação do limiar de materialização de matriz (`DEFAULT_MAX_MATRIX_DIMENSION = 10_000`) entre `tsplib_loader.py`, `database_loader.py` e `ProblemContext`. | [plan#L1379](./plan-reorganization-and-typing.plan.md#L1379) |
| `DP11-4` | Remoção de `KNOWN_OPTIMAL_COSTS` hardcoded em `config_loader.py`, consultando `solutions.cost` do DuckDB e `configs/problems.json`. | [plan#L1380](./plan-reorganization-and-typing.plan.md#L1380) |
| `DP11-5` | Adição de JSON Schemas (`configs/schemas/*.schema.json`) com `"$schema"` nos arquivos `configs/*.json` e criação de `configs/algorithms.full.example.json`. | [plan#L1381](./plan-reorganization-and-typing.plan.md#L1381) |

## Defeitos (`BUG`)

Numeração herdada de `D1`–`D8` do [walkthrough da suíte](../../documentation/walkthroughs/test-suite-walkthrough.md#6-defect-found-strict-xfails-not-fixed-in-this-change)
(`D<n>` → `BUG-<n>`, mesma ordem, sem reordenação).

### Abertos

| ID | Severidade | Afirmação | Evidência | Teste que fixa | Status |
| -- | ---------- | --------- | --------- | -------------- | ------ |
| `BUG-1` | Baixa | `Problem` é dataclass `frozen` sobre `ndarray`; `==` levanta `ValueError` e `hash()` levanta `TypeError`. | [problem.py#L30](../../src/data_models/problem.py#L30) | [test_unit_problem_validation.py#L124](../../tests/data_models/test_unit_problem_validation.py#L124) | resolvido na TASK 10.6 |
| `BUG-2` | Alta | `resolve_rng` aceita apenas `np.random.Generator`, mas `ProblemContext.rng` é `RandomState`; o RNG semeado do contexto é ignorado. | [rng.py#L56](../../src/utils/rng.py#L56), [problem_context.py#L166](../../src/protocols/problem_context.py#L166) | [test_unit_rng.py#L163](../../tests/utils/test_unit_rng.py#L163) | resolvido na TASK 10.7 |
| `BUG-3` | Alta | Ramo de variância zero fixa `mean_difference=0.0` e `ci_95_difference=(0.0, 0.0)` mesmo com diferença constante não nula. | [statistics.py#L220](../../src/benchmarking/analysis/statistics.py#L220) | [test_unit_statistics_comparison.py#L173](../../tests/benchmarking/test_unit_statistics_comparison.py#L173) | resolvido na TASK 10.8 |
| `BUG-4` | Alta | `is_valid_checkpoint` ignora o comprimento dos arrays `raw_*` que `can_resume_checkpoint` valida. | [checkpoint_io.py#L266](../../src/benchmarking/runner/checkpoint_io.py#L266) | [test_unit_checkpoint_io.py#L240](../../tests/benchmarking/test_unit_checkpoint_io.py#L240) | resolvido na TASK 10.9 |
| `BUG-5` | Média | `_mutation_bounds` protege depots apenas quando `n > 3`; rota `[depot, nó, depot]` pode trocar um endpoint. | [swap_mutation.py#L48](../../src/algorithms/strategies/mutation/swap_mutation.py#L48) | [test_unit_swap_mutation.py#L76](../../tests/algorithms/strategies/mutation/test_unit_swap_mutation.py#L76) | resolvido na TASK 10.10 |
| `BUG-6` | Média | Mesma falha de `_mutation_bounds`, duplicada na mutação por inversão. | [inversion_mutation.py#L48](../../src/algorithms/strategies/mutation/inversion_mutation.py#L48) | [test_unit_inversion_mutation.py#L77](../../tests/algorithms/strategies/mutation/test_unit_inversion_mutation.py#L77) | resolvido na TASK 10.10 |
| `BUG-9` | Média | `validate_configs` lê `tournament_size` e `two_opt_iterations`, renomeados para `k` e `max_iterations` na TASK 6.1; o ramo nunca dispara. | [config_loader.py#L551](../../src/benchmarking/runner/config_loader.py#L551) | [test_unit_config_validation.py#L73](../../tests/benchmarking/test_unit_config_validation.py#L73) | resolvido na TASK 10.4 |
| `BUG-10` | Média | Todo relatório gerado recebe o título `# Chapter 4 Validation Results`, inclusive a comparação de estágios de busca local. | [report_generator.py#L104](../../src/benchmarking/analysis/report_generator.py#L104) | — | resolvido na TASK 10.4 |
| `BUG-13` | Alta | `compute_distance_matrix` aloca tensor completo `(n, n, d)` (contrariando docstring de triângulo superior) e reimplementa fórmulas de arredondamento de `pairwise.py`. | [matrix.py#L126-L206](../../src/distances/matrix.py#L126-L206) | [test_unit_distance_matrix.py#L315](../../tests/distances/test_unit_distance_matrix.py#L315) | resolvido na TASK 10.14 |
| `BUG-14` | Alta | Cache de matriz de distâncias em GPU nas estratégias `BatchGPUCostEvaluation` e `BatchGPUTwoOpt`/`BatchGPUMultiMoveTwoOpt` indexado apenas por `problem.name`, e `ProblemContext.get_gpu_*` exige `self.xp != np` em vez de usar `cp = require_gpu()`. | [cost_evaluation.py#L123-L127](../../src/algorithms/strategies/evaluation/cost_evaluation.py#L123-L127), [two_opt.py#L432-L439](../../src/algorithms/strategies/local_search/two_opt.py#L432-L439), [problem_context.py#L305](../../src/protocols/problem_context.py#L305) | [test_unit_problem_context_gpu_distances.py#L64](../../tests/protocols/test_unit_problem_context_gpu_distances.py#L64), [test_unit_cost_evaluation_gpu_cache.py#L63](../../tests/algorithms/strategies/evaluation/test_unit_cost_evaluation_gpu_cache.py#L63) | aberto (parcial — falta `two_opt.py` e `ProblemContext.get_gpu_*` na `TASK 10.14`) |
| `BUG-15` | Alta | `cp.cuda.Stream.null.synchronize()` executado a cada chamada de `improve_batch` em `two_opt.py`, impondo barreira host-device mesmo com população residente em GPU (~12% de overhead medido em `n=52`). | [two_opt.py#L327](../../src/algorithms/strategies/local_search/two_opt.py#L327), [two_opt.py#L580](../../src/algorithms/strategies/local_search/two_opt.py#L580) | [test_unit_two_opt_synchronization.py#L102](../../tests/algorithms/strategies/local_search/test_unit_two_opt_synchronization.py#L102) | resolvido na TASK 10.14 |
| `BUG-16` | Média | `DatabaseLoader.__enter__` levanta `FileNotFoundError` dentro do `try` e o captura em `except Exception`, convertendo ausência de arquivo em `DatabaseConnectionError` genérico. | [database_loader.py#L105-L111](../../src/loaders/database_loader.py#L105-L111) | [test_unit_database_loader_errors.py#L22](../../tests/loaders/test_unit_database_loader_errors.py#L22) | resolvido na TASK 10.14 |
| `BUG-17` | Média | `is_valid_checkpoint` e `can_resume_checkpoint` usam `expected_reps: int = 30` hardcoded, rejeitando checkpoints válidos com `repetitions < 30` quando chamados com argumento default. | [checkpoint_io.py#L232](../../src/benchmarking/runner/checkpoint_io.py#L232) | — | aberto (`TASK 10.23`) |
| `BUG-18` | Alta | `orchestration.py` grava apenas o nome da classe em `results["strategies"]` (descartando `candidate_k`, `max_iterations`, `mode`, `k`) e fixa `construction_heuristic = "random_initialization"`. | [orchestration.py#L465-L485](../../src/benchmarking/runner/orchestration.py#L465-L485) | — | aberto (`TASK 10.23`, `TASK 10.22`) |
| `BUG-19` | Alta | `_inverse_positions` em `two_opt.py` aloca `np.empty(len(tour))` (causando `IndexError` em sub-rotas CVRP onde `max(tour) >= len(tour)`), e `_proximity_index` em `swap_mutation.py`/`inversion_mutation.py` desativa silenciosamente `candidate_k` quando `int(route.max()) >= route.size`. | [two_opt.py#L88](../../src/algorithms/strategies/local_search/two_opt.py#L88), [swap_mutation.py#L82](../../src/algorithms/strategies/mutation/swap_mutation.py#L82) | — | aberto (`TASK 10.20`) |
| `BUG-20` | Média | `two_opt_multi_move.cu` limita os movimentos disjuntos por passagem a `16` (`int applied_start[16]`), divergindo de `CPUMultiMoveTwoOpt` em instâncias grandes. | [two_opt_multi_move.cu#L160](../../src/algorithms/strategies/local_search/kernels/two_opt_multi_move.cu#L160) | — | aberto (`DP11-2`, `TASK 11.1`) |

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
