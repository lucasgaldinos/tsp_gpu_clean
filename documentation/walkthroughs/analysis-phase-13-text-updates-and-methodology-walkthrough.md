---
title: "Fase 13 — Redação e Atualização Textual da Dissertação (Walkthrough)"
description: >
  Registro detalhado da conclusão da Fase 13: reescrita completa dos capítulos
  1, 2, 3, 5, 6 e Apêndice A da dissertação, alinhamento metodológico a Demšar (2006),
  vetorização da busca local em CPU (DP13-5), equivalência bit a bit das variantes
  híbridas (DP13-6) e saneamento de 33 tarefas com zero alertas provisórios.
created: 2026-11-17
type: analysis
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[second_draft.tex](../../documents_tcc/tex/thesis/second_draft.tex)"
  - "[00-frontmatter.tex](../../documents_tcc/tex/thesis/parts/00-frontmatter.tex)"
  - "[01-introducao.tex](../../documents_tcc/tex/thesis/parts/01-introducao.tex)"
  - "[02-fundamentacao.tex](../../documents_tcc/tex/thesis/parts/02-fundamentacao.tex)"
  - "[03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex)"
  - "[04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex)"
  - "[05-discussao.tex](../../documents_tcc/tex/thesis/parts/05-discussao.tex)"
  - "[06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex)"
  - "[07-apendices.tex](../../documents_tcc/tex/thesis/parts/07-apendices.tex)"
  - "[two_opt.py::LocalSearchCPUTwoOpt](../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[statistics.py::StatisticalAnalyzer](../../src/benchmarking/analysis/statistics.py)"
tags:
  - analysis/thesis
  - docs/latex
  - methodology/statistics
  - review/quality
  - benchmark/modular
---

# Fase 13 — Redação e Atualização Textual da Dissertação (Walkthrough)

## 1. Visão Geral e Objetivo da Fase

A **Fase 13** consolidou o alinhamento integral entre o código-fonte de produção do framework (`src/`), a infraestrutura de telemetria e benchmarking (`benchmark_scripts/`), e o texto acadêmico da dissertação de mestrado ([second_draft.tex](../../documents_tcc/tex/thesis/second_draft.tex)).

Historicamente, o texto da dissertação continha afirmações defasadas de versões anteriores do projeto (menções a 38 instâncias da TSPLIB95, 26 repetições, cortes fixos de CPU em $n \le 100$ ou $n \le 400$, suposta lentidão da variante totalmente em GPU e alertas de resultados provisórios via blocos `\begin{warning}`). A Fase 13 eliminou sistematicamente todas as divergências documentais e teóricas, fechando **33 tarefas concluídas** com verificação automatizada e conformidade estrita aos padrões do comitê examinador e da literatura de referência (`demsar2006statistical`).

---

## 2. Decisões Fixadas e Implementadas (`DP13`)

| Identificador | Decisão | Implementação e Evidência |
| :--- | :--- | :--- |
| `DP13-1` | Variantes análogas (não equivalentes), coorte canônica de 15 instâncias e 30 repetições. | Remoção definitiva do corte de CPU (`CPU_SIZE_THRESHOLD = None`). Cobertura integral das 15 instâncias pela CPU. Nomenclatura unificada das quatro variantes (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`). |
| `DP13-2` | Estatística bayesiana como complemento futuro. | Inclusão de seções dedicadas a testes bayesianos como trabalho futuro em [06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex) (§6.4) e menções pontuais em [02-fundamentacao.tex](../../documents_tcc/tex/thesis/parts/02-fundamentacao.tex) e [03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex). |
| `DP13-3` | Conformidade com Demšar (2006) na comparação omnibus e pós-hoc. | Adição da variante $F_F$ de Iman–Davenport ao teste de Friedman em [statistics.py::StatisticalAnalyzer.friedman_test](../../src/benchmarking/analysis/statistics.py), inclusão de `mean_ranks`, estatística e graus de liberdade no retorno, e ajuste da amplitude studentizada no diagrama CD ($q_\alpha = Q_\alpha / \sqrt{2}$). |
| `DP13-4` | Protocolo pareado robusto: Wilcoxon incondicional sem portão de Shapiro–Wilk. | Remoção da dependência espúria de pré-teste de normalidade para amostras pequenas em [statistics.py::StatisticalAnalyzer.paired_comparison](../../src/benchmarking/analysis/statistics.py); estimador de Hodges–Lehmann para intervalos de confiança e correlação bisserial por postos como tamanho de efeito; correção de Holm–Bonferroni preservada. |
| `DP13-5` | Vetorização da busca local 2-opt em CPU via NumPy. | Reescrita de [two_opt.py::LocalSearchCPUTwoOpt._step_best](../../src/algorithms/strategies/local_search/two_opt.py) eliminando laço serial com conversão `int()`; aceleração medida de `14,3×` a `28,2×` mantendo identidade bit a bit de trajetórias em 3.510 verificações; viabilização da execução da CPU em instâncias até $n = 1002$. |
| `DP13-6` | Equivalência bit a bit das híbridas documentada como resultado de integridade. | Prova documental em [01-introducao.tex](../../documents_tcc/tex/thesis/parts/01-introducao.tex) e [05-discussao.tex](../../documents_tcc/tex/thesis/parts/05-discussao.tex) de que `HybridNaive` e `HybridOptimized` geram custos finais idênticos nas 15 instâncias, confirmando equivalência semântica do cálculo de custo. |
| `DP13-7` | Constantes de relato com fonte única e macros LaTeX. | Unificação de estratos de tamanho em [report_generator.py](../../src/benchmarking/reporting/generator.py) e criação das macros `\totalrepeticoes{30}` e `\totalinstancias` em [second_draft.tex](../../documents_tcc/tex/thesis/second_draft.tex), eliminando números dispersos e divergentes no texto. |
| `DP13-8` | Scripts da tese fora de `src/` e delegação sobre CLI principal. | O script [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py) atua como delegador sem replicar o `ArgumentParser` da CLI de produção em [benchmark.py](../../src/cli/benchmark.py). |
| `DP13-9` | Painel gráfico de escalonamento tempo × dimensão e gap × dimensão. | Criação de [scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py) com eixos alinhados, intervalos de dispersão de 95% e exportação vetorial integrada ao pipeline. |
| `DP13-10` | Saneamento de avisos provisórios no texto. | Remoção de todos os 6 blocos `\begin{warning}` e alertas de resultados preliminares em [00-frontmatter.tex](../../documents_tcc/tex/thesis/parts/00-frontmatter.tex), [04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex), [05-discussao.tex](../../documents_tcc/tex/thesis/parts/05-discussao.tex) e [06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex). |
| `DP13-12` | Fidelidade arquitetural de diagramas metodológicos. | Alinhamento do diagrama Mermaid de classes (`fig:class-diagram`) em [03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex) separando a classe `CheckpointManager` das funções utilitárias de módulo de [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py). |

---

## 3. Resumo das Tarefas Executadas na Fase 13

A Fase 13 foi estruturada em grupos de trabalho coordenados:

1. **Reescrita dos Capítulos da Dissertação**:
   - `Task 13.2` ([01-introducao.tex](../../documents_tcc/tex/thesis/parts/01-introducao.tex)): Apresentação do problema com formulação inteira da TSPLIB95, paridade paramétrica e descrição rigorosa das quatro variantes.
   - `Task 13.3` ([02-fundamentacao.tex](../../documents_tcc/tex/thesis/parts/02-fundamentacao.tex)): Revisão bibliográfica com categorização de paralelismo (Crainic e Simchi-Levi), formalização do operador 2-opt restrito com exclusão do par `(0, n-1)` e equações de amplitude de postos de Demšar.
   - `Task 13.4` ([03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex)): Protocolo experimental com 15 instâncias, envelope paramétrico comum ($N = 256$, $\kappa = 10$ varreduras, $N_\kappa = 20$ candidatos, $p_m = 0,02$) e fluxograma metodológico.
   - `Task 13.5` e `Task 13.6` ([05-discussao.tex](../../documents_tcc/tex/thesis/parts/05-discussao.tex) e [06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex)): Revisão terminológica e conceitual, remoção de hipóteses de assimetria de gerações e correção de registro pt-BR.
   - `Task 13.7` ([07-apendices.tex](../../documents_tcc/tex/thesis/parts/07-apendices.tex)): Documentação de reprodutibilidade, esquema de hardware da bancada de testes e dicionário de dados DuckDB.
   - `Task 12.1` e `Task 12.2` ([06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex)): Registro formal de trabalhos futuros (formato bitpacked em memória compartilhada para além de 2.048 cidades, cruzamento EAX com ciclos AB e modelo em ilhas multi-stream com CUDA streams concorrentes).

2. **Saneamento Estatístico e Metodológico (Swarm A & Tier 3)**:
   - `Task 13.8`, `Task 13.9` e `Task 13.10`: Implementação da estatística $F_F$ de Iman–Davenport, renomeação de campos em [cd_diagram.py](../../src/benchmarking/reporting/cd_diagram.py) e incorporação de `mean_ranks` no retorno agregado.
   - `Task 13.11`, `Task 13.12` e `Task 13.13`: Protocolo pareado puro de Wilcoxon com correção de Holm–Bonferroni, eliminando pré-testes de normalidade espúrios em amostras pequenas e alinhando com as recomendações de Demšar (2006).
   - `Task 13.19`, `Task 13.20` e `Task 13.21`: Desambiguação de símbolos ($\kappa$ para iterações vs. $N_\kappa$ para candidatos), substituição do termo incorreto "mediano" por "média geométrica" em speedups pareados e retificação de descrições de operadores.
   - `Task 13.29` e `Task 13.30`: Atualização do fluxograma de análise de resultados (`fig:results-flowchart`) e eliminação de caixas de advertência sobre dados preliminares.

3. **Validações Pré-Voo e Esquemas de Configuração (Swarm B)**:
   - `Task 13.16`: Unificação das constantes de estratos de tamanho em [report_generator.py](../../src/benchmarking/reporting/generator.py) (`SMALL_SIZE_MAX = 100`, `MEDIUM_SIZE_MAX = 400`).
   - `Task 13.25` e `Task 13.27`: Generalização das validações de pré-voo em [orchestration.py::_run_preflight](../../src/benchmarking/runner/orchestration.py) e sincronização reflexiva de [algorithms.schema.json](../../configs/schemas/algorithms.schema.json) via gerador reflexivo [schema_generator.py](../../src/benchmarking/runner/schema_generator.py).
   - `Task 13.26`: Parametrização de `cpu_size_threshold: int | None` permitindo formalmente `null` para ausência de corte.
   - `Task 13.28`: Implementação do módulo [scaling_plots.py](../../src/benchmarking/reporting/scaling_plots.py) para traçado vetorial tempo × dimensão e gap × dimensão.
   - `Task 13.33` e `Task 13.34`: Consolidação da macro `\totalrepeticoes{30}` e auditoria do diagrama de classes com o módulo [checkpoint_io.py](../../src/benchmarking/runner/checkpoint_io.py).

---

## 4. Transição para a Fase 14 e Fechamento

Ao término da Fase 13, sete tarefas de cunho empírico e consolidação final permaneceram abertas:

- `Task 12.4` (coleta integral e importação para DuckDB)
- `Task 12.5` (síntese final da evolução do projeto)
- `Task 13.17` (alinhamento do texto de `--incremental`)
- `Task 13.22` (isolamento de logging sob `--output-dir`)
- `Task 13.24` (delegação do script da tese sobre a CLI principal)
- `Task 13.31` (atualização das tabelas com dados consolidados)
- `Task 13.32` (atualização da discussão com base nos resultados reais)

Em consonância com as regras de planejamento do repositório (`AGENTS.md` Regra 9 e `update-plan.instructions.md`), a **Fase 13 foi formalmente congelada e fechada** como histórico 100% concluído (33 tarefas `[x]`). As sete tarefas abertas foram migradas intactas (preservando identificadores originais) para a **Fase 14**, onde foram combinadas com as novas tarefas `Task 14.1` a `Task 14.5` derivadas da auditoria pós-execução de `results_v4`.
