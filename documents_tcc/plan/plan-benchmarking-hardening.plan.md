---
title: "Plano de Endurecimento de Benchmarking, Análise Estatística e Relatórios (Fase 16)"
description: >
  Plano estruturado em 5W2H para sanar os sete defeitos (`BUG-96` a `BUG-102`)
  levantados na revisão pós-desacoplamento de `src/benchmarking/` e `src/cli/`,
  cobrindo checkpoints com `optimal_cost` nulo, invariância à escala em Wilcoxon,
  validação de deriva e quarentena no preflight, alinhamento de coortes/estilos
  em relatórios, validação de slots/protocolos e comparação integral de bancos DuckDB.
created: 2026-11-23
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: plan
scope: local
modifications:
  - date_modified: 2026-11-23
    modifications:
      - description: >
          Criação do plano da Fase 16 em arquivo dedicado (separado de
          `plan-reorganization-and-typing.plan.md` por tamanho) com as decisões
          `DP16-1` e `DP16-2`, registro das Tasks `16.1` a `16.5` para os defeitos
          `BUG-96` a `BUG-102` e sincronização com `table-of-symbols.plan.md`.
related_files:
  - "[table-of-symbols.plan.md](./table-of-symbols.plan.md)"
  - "[plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md)"
  - "[AGENTS.md](../../AGENTS.md)"
  - "[checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py)"
  - "[orchestration.py](../../src/benchmarking/runner/orchestration.py)"
  - "[config_loader.py](../../src/benchmarking/runner/config_loader.py)"
  - "[statistics.py](../../src/benchmarking/analysis/statistics.py)"
  - "[scaling_plots.py](../../src/benchmarking/reporting/scaling_plots.py)"
  - "[table_reports.py](../../src/benchmarking/reporting/table_reports.py)"
  - "[generator.py](../../src/benchmarking/reporting/generator.py)"
  - "[report.py](../../src/cli/report.py)"
  - "[validate_consistency.py](../../src/benchmarking/database/validate_consistency.py)"
tags:
  - analysis/planning
  - analysis/statistics
  - benchmark/runner
  - benchmark/reporting
  - benchmark/checkpoints
  - benchmark/database
  - review/quality
  - notes/planning
  - notes/session
  - notes/handoff
---

# Plano de Endurecimento de Benchmarking, Análise Estatística e Relatórios (Fase 16)

## Escopo e regras

Este plano consolida as correções estruturais dos sete defeitos (`BUG-96` a `BUG-102`) identificados na revisão de cinco eixos dos commits `0b202d7..b7fae85`, além de dois defeitos correlatos descobertos durante a inspeção direta do código (indentação pós-quarentena em `orchestration.py` e propagação de `all_results` bruto em `generator.py`).

- **Escopo permitido:** correções em `src/benchmarking/`, `src/cli/report.py`, testes unitários de regressão falsificáveis em `tests/benchmarking/` e `tests/cli/`, e rastreabilidade neste plano e em [table-of-symbols.plan.md](./table-of-symbols.plan.md).
- **Escopo proibido:** nenhuma alteração em `results_v4/`, nenhum acoplamento de nomes de variantes da dissertação ou prosa em português dentro de `src/`, nenhum alias legado de compatibilidade (`AGENTS.md` Regra 10) e nenhum `git commit`/`git push` sem aprovação explícita.
- **Regras de verificação:** toda correção acompanha teste unitário falsificável que falha sem ela; ao final, exigir código `0` em `pytest -m "not gpu and not slow"`, `mypy src tests benchmark_scripts`, `ruff check`, `schema_generator --check`, `benchmark_scripts.check_docs` e paridade byte a byte do `tsp-report` contra `results_v4/checkpoints` sem alterar `results_v4/`.

## Fonte da verdade

Os identificadores `DP16-*` e `BUG-96` a `BUG-102` têm registro canônico em [table-of-symbols.plan.md](./table-of-symbols.plan.md). As decisões estatísticas (`DP13-3`, `DP13-4`, `DP12-18`) e a separação acíclica entre `src/benchmarking/analysis/` e `src/benchmarking/reporting/` permanecem inalteradas.

## Decisões fixadas

- `DP16-1`: Contratos estritos de checkpoints e estatística pareada — predicados de checkpoint aceitam `raw_gaps: None` quando o ótimo é desconhecido (`BUG-96`), `save_problem_statistics` deriva *gaps* a partir de `optimal_cost` retroativo, `wilcoxon_paired_summary` compartilha tolerância absoluta invariante à escala (`1e-10`) com `paired_comparison` (`BUG-97`), e o *preflight* distingue deriva de configuração (erro rígido) de JSON truncado/corrompido (quarentena automática fora de `--dry-run`, `BUG-98`).
- `DP16-2`: Fonte única de coorte, estilo, validação de slots e integridade DuckDB — renderizadores (`table_reports.py`, `scaling_plots.py`, `generator.py`) consomem exclusivamente `cohort.results`, `cohort.problem_configs`, `cohort.algorithms_for_problem` e `meta["dimension"]` com regra unificada `all(...)` para *gap* vs. custo (`BUG-99`); `tsp-report` valida `--config-dir` sem criar `checkpoints/` espúrio (`BUG-100`); `config_loader.py` valida nomes de slots, protocolos `@runtime_checkable` e limites por algoritmo (`BUG-101`); e `compare_databases` verifica todas as linhas ordenadas com fechamento garantido de conexões (`BUG-102`).

## Lote 1 — Checkpoints, estatística e orquestração (Tasks 16.1–16.3)

- [x] **Task 16.1: Suportar `raw_gaps: None` nos predicados de checkpoint e `optimal_cost` retroativo em estatísticas por problema (`BUG-96`)**

  - **O quê:** Unificar a validação de comprimentos de arrays brutos em `is_valid_checkpoint` e `can_resume_checkpoint` aceitando `raw_gaps: None` (ou ausente) para instâncias sem custo ótimo conhecido, e derivar `mean_gap`/`std_gap` em `save_problem_statistics` quando `optimal_cost` é fornecido retroativamente ou a entrada traz apenas resumos.
  - **Por quê:** O carregamento direto de arquivos `.tsp` sem ótimo emite `raw_gaps: None`; chamar `len(raw_gaps)` lançava `TypeError`, marcando checkpoints íntegros como corrompidos e impedindo retomada, além de gravar `mean_gap: null` mesmo quando `problems.json` informava o ótimo posteriormente.
  - **Quem:** Agent (`implementer`).
  - **Onde:** [checkpoint_io.py::is_valid_checkpoint](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::can_resume_checkpoint](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::save_problem_statistics](../../src/benchmarking/runner/checkpoint_io.py).
  - **Quando:** Sem dependências; primeiro item do Lote 1.
  - **Como:** Extrair `_validate_checkpoint_arrays` exigindo `len(raw_times) == len(raw_costs) == len(raw_seeds) == successful_runs` e verificando `raw_gaps` apenas quando não nulo; em `save_problem_statistics`, calcular `(costs - optimal_cost) / optimal_cost * 100.0` quando `raw_gaps is None` e preservar campos `mean_*`/`std_*` quando arrays brutos estão ausentes.
  - **Esforço:** Baixo.
  - **Impacto:** Restaura validação, contagem de concluídos, retomada e persistência de estatísticas para instâncias `.tsp` arbitrárias sem ótimo conhecido.
  - **DoD:** Testes `test_checkpoint_predicates_accept_null_raw_gaps_for_unknown_optimal` e `test_save_problem_statistics_derives_retroactive_gaps_and_handles_summary_only` aprovados em `pytest` sem avisos de corrupção.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [checkpoint_io.py::_validate_checkpoint_arrays](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::is_valid_checkpoint](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::can_resume_checkpoint](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::save_problem_statistics](../../src/benchmarking/runner/checkpoint_io.py)
      + [test_unit_checkpoint_io.py::test_checkpoint_predicates_accept_null_raw_gaps_for_unknown_optimal](../../tests/benchmarking/test_unit_checkpoint_io.py), [test_unit_checkpoint_io.py::test_save_problem_statistics_derives_retroactive_gaps_and_handles_summary_only](../../tests/benchmarking/test_unit_checkpoint_io.py)
    - **Resumo:** Validação de arrays centralizada em `_validate_checkpoint_arrays` tratando `raw_gaps: None` como válido quando ausente e validando comprimento quando presente; `save_problem_statistics` passa a derivar *gaps* retroativos a partir de `raw_costs` e preservar métricas de entradas resumidas.
    - **DoD atendido:** `rtk .venv/bin/pytest -q tests/benchmarking/test_unit_checkpoint_io.py` aprovado sem avisos de corrupção, cobrindo `raw_gaps: None`, rejeição de `raw_gaps` não nulo desalinhado, *gaps* retroativos (`mean_gap == 10.0`) e entradas resumidas.

- [x] **Task 16.2: Garantir invariância à escala e verificação de comprimento em `wilcoxon_paired_summary` (`BUG-97`)**

  - **O quê:** Substituir `np.allclose` em `StatisticalAnalyzer.wilcoxon_paired_summary` pela mesma tolerância absoluta invariante à escala (`1e-10`) e validação de igualdade de comprimento usadas por `paired_comparison`.
  - **Por quê:** O padrão `rtol=1e-5` de `np.allclose` colapsava diferenças pareadas reais (ex.: $\Delta = 5{,}0$ em custos de ordem $10^6$) para `(0.0, 1.0, 0.0)`, contradizendo `paired_comparison` e contaminando as macros de `benchmark_macros.tex`.
  - **Quem:** Agent (`implementer`).
  - **Onde:** [statistics.py::StatisticalAnalyzer.wilcoxon_paired_summary](../../src/benchmarking/analysis/statistics.py), [statistics.py::StatisticalAnalyzer.paired_comparison](../../src/benchmarking/analysis/statistics.py).
  - **Quando:** Sem dependências; paralelo à Task 16.1.
  - **Como:** Extrair o helper estático `StatisticalAnalyzer._count_zero_differences`, validar `arr_a.size != arr_b.size` levantando `ValueError` e curto-circuitar apenas quando todas as diferenças absolutas forem menores que `1e-10`.
  - **Esforço:** Trivial.
  - **Impacto:** Alinha integralmente o teste de Wilcoxon de coorte com o teste pareado por problema e preserva invariância ordinal (`DP12-18`, `DP13-4`).
  - **DoD:** `test_wilcoxon_paired_summary_is_scale_invariant_and_checks_length` confirma concordância exata com `paired_comparison` ($p = 2^{-9}, r = -1{,}0$) em vetores de ordem $10^6$ e `ValueError` em comprimentos distintos.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [statistics.py::StatisticalAnalyzer._count_zero_differences](../../src/benchmarking/analysis/statistics.py), [statistics.py::StatisticalAnalyzer.paired_comparison](../../src/benchmarking/analysis/statistics.py), [statistics.py::StatisticalAnalyzer.wilcoxon_paired_summary](../../src/benchmarking/analysis/statistics.py)
      + [test_unit_statistics.py::test_wilcoxon_paired_summary_is_scale_invariant_and_checks_length](../../tests/benchmarking/test_unit_statistics.py)
    - **Resumo:** Removido `np.allclose` de `wilcoxon_paired_summary`; ambos os caminhos pareados compartilham `_count_zero_differences` (`tolerance=1e-10`) e rejeitam vetores de comprimentos diferentes.
    - **DoD atendido:** `test_wilcoxon_paired_summary_is_scale_invariant_and_checks_length` aprovado em `pytest`, confirmando $p = 2^{-9}$ e $r = -1{,}0$ idênticos a `paired_comparison` em vetores $10^6$ e `ValueError` em comprimentos distintos.

- [x] **Task 16.3: Endurecer validação de deriva de checkpoints, persistência de `strategy_overrides` e fluxo de quarentena (`BUG-98`)**

  - **O quê:** Validar deriva de `local_search_stage`, `strategy_params`, `pipeline` e comprimentos de arrays brutos no preflight, mesclar `strategy_overrides` customizados em `_get_strategy_params`, deferir JSON truncado à quarentena fora de `--dry-run` e corrigir a indentação pós-quarentena em `orchestration.py`.
  - **Por quê:** Alterações em `--local-search-stage`, `--strategy-param` ou `params` passavam silenciosamente no preflight reutilizando checkpoints incompatíveis; overrides fora de `_STRATEGY_PARAM_ATTRS` eram descartados nos metadados; e arquivos JSON corrompidos abortavam o preflight antes de atingir `quarantine_checkpoint` (que, por indentação incorreta, ainda caía no ramo de retomada).
  - **Quem:** Agent (`implementer`).
  - **Onde:** [checkpoint_io.py::_validate_single_checkpoint_compatibility](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::validate_existing_checkpoints](../../src/benchmarking/runner/checkpoint_io.py), [orchestration.py::_run_preflight](../../src/benchmarking/runner/orchestration.py), [orchestration.py::run_comprehensive_benchmark](../../src/benchmarking/runner/orchestration.py).
  - **Quando:** Depende da Task 16.1 (mesmo módulo `checkpoint_io.py`).
  - **Como:** Adicionar `_config_values_match` e checagens de `local_search_stage`, `strategy_params`, `pipeline` e arrays em `_validate_single_checkpoint_compatibility`; expor `allow_corrupt=not dry_run` para diferenciar `--dry-run` (erro sem mutação) de execução real (quarentena via `quarantine_checkpoint`); desindentar o ramo `if checkpoint_exists:` após o `try/except` de quarentena; e mesclar `strategy_overrides[slot]` em `_get_strategy_params`.
  - **Esforço:** Médio.
  - **Impacto:** Impede mistura silenciosa de experimentos com hiperparâmetros distintos e restaura recuperação automática de falhas de escrita (`BUG-37`).
  - **DoD:** Testes `test_validate_existing_checkpoints_detects_stage_strategy_params_pipeline_and_corrupt_mode` e `test_run_comprehensive_benchmark_quarantines_corrupt_checkpoint_and_runs_cleanly` aprovados, preservando compatibilidade integral com `results_v4/checkpoints/`.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [checkpoint_io.py::_config_values_match](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::_validate_single_checkpoint_compatibility](../../src/benchmarking/runner/checkpoint_io.py), [checkpoint_io.py::validate_existing_checkpoints](../../src/benchmarking/runner/checkpoint_io.py)
      + [orchestration.py::_run_preflight](../../src/benchmarking/runner/orchestration.py), [orchestration.py::run_comprehensive_benchmark](../../src/benchmarking/runner/orchestration.py)
      + [test_unit_checkpoint_io.py::test_validate_existing_checkpoints_detects_stage_strategy_params_pipeline_and_corrupt_mode](../../tests/benchmarking/test_unit_checkpoint_io.py), [test_unit_checkpoint_io.py::test_run_comprehensive_benchmark_quarantines_corrupt_checkpoint_and_runs_cleanly](../../tests/benchmarking/test_unit_checkpoint_io.py)
    - **Resumo:** Preflight agora detecta deriva em estágio de busca local, parâmetros de estratégias (inclusive CLI overrides), pipeline customizado e arrays brutos; arquivos JSON truncados falham em `--dry-run` e são quarentenados em execuções reais com início limpo; `_get_strategy_params` persiste overrides customizados.
    - **DoD atendido:** `test_validate_existing_checkpoints_detects_stage_strategy_params_pipeline_and_corrupt_mode` e `test_run_comprehensive_benchmark_quarantines_corrupt_checkpoint_and_runs_cleanly` aprovados em `pytest` (incluindo `custom_span == 7` persistido e `.corrupt.*.bak` gerado), com paridade integral preservada sobre `results_v4/checkpoints/`.

## Lote 2 — Relatórios, CLI `tsp-report`, validação de configuração e DuckDB (Tasks 16.4–16.5)

- [x] **Task 16.4: Alinhar resolução de metadados, linha de base de fallback, métrica de qualidade e propagação de coorte nos relatórios e em `tsp-report` (`BUG-99`, `BUG-100`)**

  - **O quê:** Corrigir resolução de `problem_metadata["dimension"]` e política `all(...)` em `scaling_plots.py`, alinhar o fallback da Tabela 1 a `cohort.algorithms_for_problem(problem)[0]`, propagar `cohort.results`/`cohort.problem_configs` pela Tabela 3 e `generate_result_tables`, e eliminar efeitos colaterais de diretório/configuração/log em `src/cli/report.py`.
  - **Por quê:** `Problem.to_metadata_dict` serializa `"dimension"` (não `"size"`); coortes mistas com e sem ótimo descartavam instâncias no painel inferior; o fallback de Tabela 1 ignorava `ReportStyle.algorithm_order`; passar apenas `analysis=` quebrava Tabela 3 e `plots`; e `tsp-report` criava `output_root/checkpoints/` vazio e silenciava `--config-dir` inexistente.
  - **Quem:** Agent (`implementer`).
  - **Onde:** [scaling_plots.py::_resolve_problem_size](../../src/benchmarking/reporting/scaling_plots.py), [scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py), [table_reports.py::_compute_baseline_times](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::_compute_table1_rows](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::_compute_table3_size_category_winners](../../src/benchmarking/reporting/table_reports.py), [generator.py::generate_result_tables](../../src/benchmarking/reporting/generator.py), [report.py::_load_problem_configs](../../src/cli/report.py), [report.py::main](../../src/cli/report.py).
  - **Quando:** Sem dependências sobre o Lote 1; paralelo à Task 16.5.
  - **Como:** Ler `meta.get("dimension")` antes de `meta.get("size")` e exigir `size > 0` em `_resolve_problem_size`; trocar `has_any_gap` por `has_all_gaps` em `generate_scaling_plots`; usar `cohort.algorithms_for_problem(problem)[0]` em `_compute_baseline_times`/`_compute_table1_rows`; rotear `cohort.results`/`cohort.problem_configs` em `generator.py` e `table_reports.py`; e validar `config_dir.is_dir()`, apontar `CheckpointManager.checkpoint_dir` para o diretório de entrada existente e atualizar `logging.getLogger().setLevel(...)` em `src/cli/report.py`.
  - **Esforço:** Médio.
  - **Impacto:** Garante paridade de métricas, ordenação e comportamento previsível em fluxos autônomos de `tsp-report` e chamadas com `CohortAnalysis` pré-computada.
  - **DoD:** Testes `test_resolve_problem_size_metadata_dimension_and_mixed_optimal_cost_fallback`, `test_table1_fallback_respects_algorithm_order_and_precomputed_analysis_drives_table3_and_plots` e `test_report_cli_validates_config_dir_avoids_spurious_checkpoints_and_sets_log_level` aprovados, com paridade byte a byte de `tsp-report` sobre `results_v4/checkpoints`.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [scaling_plots.py::_resolve_problem_size](../../src/benchmarking/reporting/scaling_plots.py), [scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py)
      + [table_reports.py::_compute_baseline_times](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::_compute_table1_rows](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::_compute_table3_size_category_winners](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::generate_markdown_tables](../../src/benchmarking/reporting/table_reports.py), [table_reports.py::generate_latex_tables](../../src/benchmarking/reporting/table_reports.py)
      + [generator.py::generate_result_tables](../../src/benchmarking/reporting/generator.py)
      + [report.py::_load_problem_configs](../../src/cli/report.py), [report.py::main](../../src/cli/report.py)
      + [test_unit_scaling_plots.py::test_resolve_problem_size_metadata_dimension_and_mixed_optimal_cost_fallback](../../tests/benchmarking/test_unit_scaling_plots.py), [test_unit_report_generator.py::test_table1_fallback_respects_algorithm_order_and_precomputed_analysis_drives_table3_and_plots](../../tests/benchmarking/test_unit_report_generator.py), [test_unit_cli_report.py::test_report_cli_validates_config_dir_avoids_spurious_checkpoints_and_sets_log_level](../../tests/cli/test_unit_cli_report.py)
    - **Resumo:** Todos os renderizadores passaram a consumir consistentemente a coorte derivada, respeitar `algorithm_order` no fallback da Tabela 1, ler `dimension` e usar `all(...)` para escolha entre *gap* e custo; `tsp-report` falha rapidamente em `--config-dir` inexistente, evita criar `checkpoints/` espúrio e aplica `--log-level` no logger raiz.
    - **DoD atendido:** Testes de `scaling_plots`, `report_generator` e `cli.report` aprovados (`309 passed` nos 8 módulos alvo), `mypy` sem erros (`162` arquivos) e `tsp-report` sobre `results_v4/checkpoints` produziu `diff -u` vazio para `benchmark_macros.tex` e ambos os `.tikz`, sem criar `checkpoints/` espúrio.

- [x] **Task 16.5: Validar slots, protocolos de estratégia e parâmetros por algoritmo em `config_loader.py` e corrigir comparação integral em `validate_consistency.py` (`BUG-101`, `BUG-102`)**

  - **O quê:** Rejeitar chaves de slot desconhecidas em `strategies`, verificar conformidade slot↔protocolo (`@runtime_checkable`) e limites de `algo_cfg.params` em `config_loader.py`, e corrigir `compare_databases` para fechar conexões com `try/finally`, suportar bancos vazios e comparar todas as linhas ordenadas por `(problem_name, algorithm)`.
  - **Por quê:** Erros de digitação em slots (ex.: `cross_over`), estratégias trocadas de slot (ex.: `MutationSwap` em `crossover`) e overrides inválidos em `algo_cfg.params` passavam com zero avisos; e `compare_databases` lançava `IndexError` em bancos vazios, vazava conexões e amostrava apenas 2 linhas por `run_id`.
  - **Quem:** Agent (`implementer`).
  - **Onde:** [config_loader.py::VALID_STRATEGY_SLOTS](../../src/benchmarking/runner/config_loader.py), [config_loader.py::AlgorithmConfig.instantiate](../../src/benchmarking/runner/config_loader.py), [config_loader.py::_validate_class_spec](../../src/benchmarking/runner/config_loader.py), [config_loader.py::validate_configs](../../src/benchmarking/runner/config_loader.py), [validate_consistency.py::compare_databases](../../src/benchmarking/database/validate_consistency.py).
  - **Quando:** Sem dependências; paralelo à Task 16.4.
  - **Como:** Definir `VALID_STRATEGY_SLOTS` e `_slot_protocol_map()` validando `issubclass(target_cls, protocol)` em `_validate_class_spec`, checar `self.strategies` em `instantiate`, validar `population_size`, `mutation_rate` e `local_search_stage` (`LocalSearchStage.__value__`) em `validate_configs`; em `compare_databases`, usar `try/finally`, tratar `count1 == 0`, ordenar deterministicamente por `(problem_name, algorithm, timestamp, run_id)` e comparar todas as linhas/colunas de dados via `_values_equal`.
  - **Esforço:** Baixo.
  - **Impacto:** Detecta configurações inválidas em pré-voo antes de iniciar o AG e torna a validação de consistência DuckDB confiável sobre toda a tabela.
  - **DoD:** Testes `test_validate_configs_and_instantiate_catch_bad_slots_protocols_and_algo_params` e `test_compare_databases_handles_empty_dbs_reordered_rows_and_tail_differences` aprovados.

  + **Solution / O que foi feito:**
    - **Arquivos alterados:**
      + [config_loader.py::VALID_STRATEGY_SLOTS](../../src/benchmarking/runner/config_loader.py), [config_loader.py::AlgorithmConfig.instantiate](../../src/benchmarking/runner/config_loader.py), [config_loader.py::_validate_class_spec](../../src/benchmarking/runner/config_loader.py), [config_loader.py::validate_configs](../../src/benchmarking/runner/config_loader.py)
      + [validate_consistency.py::_values_equal](../../src/benchmarking/database/validate_consistency.py), [validate_consistency.py::compare_databases](../../src/benchmarking/database/validate_consistency.py)
      + [test_unit_config_validation.py::test_validate_configs_and_instantiate_catch_bad_slots_protocols_and_algo_params](../../tests/benchmarking/test_unit_config_validation.py), [test_unit_db_operations.py::test_compare_databases_handles_empty_dbs_reordered_rows_and_tail_differences](../../tests/benchmarking/database/test_unit_db_operations.py)
    - **Resumo:** Validação de configuração ampliada para rejeitar slots desconhecidos, violação de protocolos PEP 544 e overrides fora de faixa em `algo_cfg.params`; `compare_databases` reescrito para cobertura total de linhas/colunas sem vazamento de conexões nem falha em tabelas vazias.
    - **DoD atendido:** `test_validate_configs_and_instantiate_catch_bad_slots_protocols_and_algo_params` e `test_compare_databases_handles_empty_dbs_reordered_rows_and_tail_differences` aprovados em `pytest`, com `mypy`, `ruff check`, `ruff format --check` e `schema_generator --check` todos em código `0`.

## Itens adiados

- `TASK 14.7` (substituição de literais nos `.tex` da dissertação por macros de `benchmark_macros.tex`) permanece adiada em [plan-reorganization-and-typing.plan.md](./plan-reorganization-and-typing.plan.md) por decisão explícita do usuário.
- Simplificação interna de `table_reports.py` e `scaling_plots.py` permanece fora do escopo desta fase.

## Handoff

- **State**: done
- **Last verified observation**: `rtk .venv/bin/mypy src tests benchmark_scripts && rtk .venv/bin/ruff check src tests benchmark_scripts && rtk .venv/bin/pytest -q tests/benchmarking/test_unit_checkpoint_io.py ... tests/cli/test_unit_cli_report.py && ... && check_docs && check_thesis && tsp-report parity` -> `Success: no issues found in 162 source files`, `All checks passed!`, `309 passed` (`1189` no conjunto não-GPU/não-lento), `OK: no documentation drift detected`, `ALL_GATES_PASSED` (2026-11-23)
- **Next item**: Nenhum (Fase 16 concluída; revisão do diff e commit mediante aprovação explícita do usuário)
- **Open decisions**: Nenhuma | N/A | N/A
- **Read first**: `plan-benchmarking-hardening.plan.md#decisões-fixadas`, `table-of-symbols.plan.md#fase-16-dp16`, `src/benchmarking/runner/checkpoint_io.py::_validate_single_checkpoint_compatibility`, `src/benchmarking/analysis/statistics.py::StatisticalAnalyzer.wilcoxon_paired_summary`
