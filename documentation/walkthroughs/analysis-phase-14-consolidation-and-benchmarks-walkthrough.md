---
title: "Fase 14 — Consolidação Homogênea, Extensão Estatística e Entrega da Dissertação (Walkthrough)"
description: >
  Registro técnico do progresso da Fase 14: auditoria e saneamento de Batches 1 e 2,
  recoleta limpa de HybridOptimized (DP14-1, BUG-91), padronização estrita de n = 30
  em todas as 60 células, generalização multi-métrica em src/ (DP14-2, BUG-92), figuras
  vetoriais nativas ABNT (DP14-3, BUG-93) e consolidação do banco DuckDB (DP12-4).
created: 2026-11-17
type: analysis
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../../documents_tcc/plan/table-of-symbols.plan.md)"
  - "[orchestration.py::run_comprehensive_benchmark](../../src/benchmarking/runner/orchestration.py)"
  - "[cohort_analysis.py::compare_algorithms](../../src/benchmarking/analysis/cohort_analysis.py)"
  - "[problem_statistics.py::analyze_problem](../../src/benchmarking/analysis/problem_statistics.py)"
  - "[scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py)"
  - "[run_chapter4_benchmark.py](../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py)"
  - "[checkpoint_io.py::CheckpointManager](../../src/benchmarking/runner/checkpoint_io.py)"
  - "[04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex)"
tags:
  - analysis/benchmark
  - benchmark/hardware
  - methodology/statistics
  - docs/latex
  - review/quality
---

# Fase 14 — Consolidação Homogênea, Extensão Estatística e Entrega da Dissertação (Walkthrough)

## 1. Contexto e Motivação da Fase 14

Após a conclusão da Fase 13, que alinhou os capítulos teóricos da dissertação ao protocolo experimental, a execução completa do benchmark oficial em `results_v4` foi submetida a uma auditoria aprofundada de dados brutos e contratos de código. Essa análise revelou três ordens de desafios que motivaram a abertura da **Fase 14**:

1. **Heterogeneidade de Amostras e Não-Estacionariedade (`BUG-91`, `DP14-1`)**: Os primeiros 30 runs (`[0:30]`) da variante `HybridOptimized` nas instâncias `att48` até `rat783` haviam sido executados antes do commit `dcb9027` (que vetorizou o cruzamento `CrossoverCPUPermOrder._apply_ox` em NumPy). Isso gerou uma lentidão espúria de 4,23× no cruzamento de `rat783`, invertendo artificialmente a curva de escalonamento (`rat783` com 32,75s contra `pr1002` com 25,55s). Além disso, conviviam checkpoints de 60, 37 e 30 repetições.
2. **Estatística Agregada Restrita e Acoplamento em `src/` (`BUG-92`, `DP14-2`)**: O pipeline agregado comparava estatisticamente apenas o gap de qualidade (`mean_gap`), sem testar formalmente se as diferenças de tempo de execução eram significativas via testes de Friedman e pós-testes de Nemenyi/Holm; o encaminhamento da baseline `CPU` estava ausente em [orchestration.py](../../src/benchmarking/runner/orchestration.py), gerando logs com ausência de speedups; e o pacote `src/benchmarking/analysis/` continha ramos obsoletos de estratificação (`STRATUM 1 / STRATUM 2`) e menções literais a nomes de algoritmos da tese.
3. **Tipografia e Dimensão Gráfica (`BUG-93`, `DP14-3`)**: O módulo de gráficos gerava PDFs com fontes Type 3 rasterizadas (sem `pdf.fonttype = 42`), em largura de 8,5 polegadas (incompatível com a mancha de 6,30 polegadas da ABNT) e emitia arquivos `.png` desnecessários.

---

## 2. Decisões Fixadas da Fase 14 (`DP14`)

- **`DP14-1` (Dataset Homogêneo Canônico $n = 30$, Sementes `42..71` e Código Pós-`dcb9027`)**: Arquivar os 14 checkpoints afetados de `HybridOptimized`, re-executar na GPU física exclusivamente essas células sob o código atual e padronizar todas as `15 × 4 = 60` células em exatamente 30 repetições.
- **`DP14-2` (Remoção da Estratificação de Resultados e Extensão Multi-Métrica Desacoplada de `src/`)**: Suprimir a divisão por faixas de tamanho (`Small`/`Medium`/`Large`) nos resultados do Capítulo 4, manter `src/` 100% genérico e em inglês, adicionar comparação omnibus e pós-hoc para tempo de execução e estender a inferência pareada com tamanhos de efeito.
- **`DP14-3` (Figuras Vetoriais Nativas ABNT em `.pdf` e `.svg`, Sem `.png`)**: Configurar `pdf.fonttype = 42`, família `Libertinus Serif` com largura de 6,30 polegadas, parametrização de rótulos em pt-BR via scripts da tese e eliminação de artefatos rasterizados.
- **`DP14-4` (Completude Empírica, Decomposição em 8 Fases e Prosa Acessível para PO)**: Integrar a tabela de fases (efeito da Lei de Amdahl), diagramas CD duplos (Tempo e Gap) e documentar as notas de `notes_project_direction_questions.md` em registro acessível para examinadores de Pesquisa Operacional.
- **`DP14-5` (Refinamento Visual de Escalonamento, Macros `benchmark_macros.tex` e Contrato de Figuras)**: Manter eixo contínuo numérico em $n$ com ticks exatos nas 15 dimensões da coorte (rotação 45° + dois níveis verticais sem colisão), adotar faixa interquartil (IQR: $Q_{25}\text{–}Q_{75}$ com mediana) no gráfico de escalonamento, documentar os degraus empíricos (`berlin52` e `si535`) em [04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex), automatizar a geração de `results_v4/tables/benchmark_macros.tex` (mantendo os literais nos `.tex` da dissertação até revisão posterior) e padronizar o contrato TikZ vs. Matplotlib Type 42.

---

## 3. O que já foi Executado e Verificado

A Fase 14 foi dividida em lotes operacionais. Os **Batches 1 e 2 já foram integralmente concluídos e verificados**:

### Batch 1: Infraestrutura de Código, Estatística e Gráficos (`Task 13.17`, `13.22`, `13.24`, `14.2`, `14.3`)

- **Contrato de CLI e Repetições (`Task 13.17`, `BUG-66`, `BUG-67`)**:
  + Retificada a string de ajuda e docstring de [run_chapter4_benchmark.py](../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py) distinguindo formalmente `--incremental` (adiciona bloco inteiro) de `--no-incremental` (completa até o alvo de repetições).
  + Alinhadas as menções de repetições à macro unificada `\totalrepeticoes{30}` e à configuração canônica de [benchmark.json](../../benchmark_scripts/experiments-and-examples/chapter4/configs/benchmark.json).
- **Isolamento de Logging e Idempotência (`Task 13.22`, `BUG-74`)**:
  + Movida a chamada de `setup_logging()` para **após** `parse_args()` em [run_chapter4_benchmark.py::main](../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py).
  + Tornada a configuração de handlers idempotente em [logging_config.py](../../src/benchmarking/logging_config.py), garantindo que passagens com `--output-dir` em diretórios temporários criem exatamente um arquivo de log e deixem `results_v4/` 100% intocado.
- **Delegação da CLI da Tese (`Task 13.24`, `DP13-8`, `BUG-80`)**:
  + Refatorado [run_chapter4_benchmark.py](../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py) para delegar diretamente à CLI canônica [benchmark.py](../../src/cli/benchmark.py).
  + Comprovado via `rg -n "ArgumentParser" benchmark_scripts/ src/` que existe **exatamente uma ocorrência** de `ArgumentParser` em todo o código (`src/cli/benchmark.py`).
- **Generalização Multi-Métrica e Remoção de Estratificação (`Task 14.2`, `DP14-2`, `BUG-92`)**:
  + Encaminhado `baseline_algorithm` em [orchestration.py::run_comprehensive_benchmark](../../src/benchmarking/runner/orchestration.py).
  + Eliminado o ramo de estratos obsoletos `STRATUM 1 / STRATUM 2` e removidas todas as ocorrências literais de `"CPU"` e `"HybridNaive"` em `src/benchmarking/analysis/` (`rg` retornou zero ocorrências).
  + Generalizada a função [compare_algorithms](../../src/benchmarking/analysis/cohort_analysis.py) para suportar comparações sobre `mean_gap` e `mean_time`/speedup, gerando os dois arquivos TikZ de Diferença Crítica (`chapter4_validation_cd_diagram_gap.tikz` e `chapter4_validation_cd_diagram_time.tikz`).
  + Adicionados testes pareados de Wilcoxon com intervalos de Hodges–Lehmann e postos bisseriais para `time_data` em [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py), corrigindo cabeçalhos desatualizados que ainda citavam `"paired t-test"`.
- **Figuras Vetoriais Nativas ABNT (`Task 14.3`, `DP14-3`, `BUG-93`)**:
  + Atualizado [scaling_plots.py](../../src/benchmarking/reporting/scaling_plots.py) com `pdf.fonttype = 42`, largura de 6,30 polegadas (`TEXT_WIDTH_INCHES = 6.3`) e família tipográfica nativa.
  + Implementada a injeção de títulos e legendas em português brasileiro pelo script da tese, mantendo padrões em inglês em `src/`.
  + Eliminada a geração padrão de `.png`, verificando que `pdffonts results_v4/figures/scaling_time_gap.pdf` contém zero fontes Type 3.

### Batch 2: Coleta Homogênea em GPU e Consolidação DuckDB (`Task 14.1`, `Task 12.4`)

- **Arquivamento e Truncamento de Checkpoints (`Task 14.1`, `DP14-1`, `BUG-91`)**:
  + Movidos os 14 checkpoints de `HybridOptimized` (`att48` até `rat783`) coletados antes do commit `dcb9027` para `results_v4/archive/pre_dcb9027_HybridOptimized/`.
  + Truncados os 46 checkpoints restantes das outras variantes para exatamente 30 repetições (`seeds 42..71`), recalculando todos os sumários estatísticos ($ddof=1$) e métricas de transferência com fechamento exato, salvando os arquivos originais de 60 repetições em `results_v4/archive/pre_n30_trim/`.
- **Re-execução de `HybridOptimized` na GPU física**:
  + Executada a re-coleta de `HybridOptimized` nas 14 instâncias (`att48` até `rat783`) por 30 repetições cada (420 execuções do algoritmo na GTX 1050, duração de ~27 minutos).
  + Comprovada a resolução da inversão de escalonamento: `rat783_HybridOptimized` demorou **19,95s** (abaixo dos 25,55s de `pr1002_HybridOptimized`), comparado aos 32,75s do arquivo antigo pré-`dcb9027`.
  + Comprovado que todos os 60 checkpoints em `results_v4/checkpoints/` possuem rigorosamente `repetitions == 30`, `len(raw_costs) == 30` e sementes pareadas `42..71`.
- **População do Banco DuckDB e Relatórios Finais (`Task 12.4`, `DP12-4`)**:
  + Importados com sucesso os 60 checkpoints padronizados para `results_v4/results.duckdb` e `datasets/tsp.duckdb` (1.800 execuções registradas sem duplicatas).
  + Regeneradas as tabelas em Markdown e LaTeX (`results_v4/tables/chapter4_validation.{md,tex}`), comprovando fechamento de balanço de fases (`Sum/Wall %`) entre **98,3% e 100,0%** em todas as 60 células.

---

## 4. Métricas Consolidadas do Dataset Canônico ($n = 30$)

Abaixo estão os dados consolidados obtidos ao término do Batch 2, que servem de fonte da verdade para a redação do Capítulo 4:

### Tabela Agregada (Média Geométrica de Speedup frente à CPU)

| Algoritmo | Gap Médio (%) | Desvio Gap (%) | Tempo Médio (s) | Desvio Tempo (s) | Speedup Geométrico | Posto Médio (Gap) | Posto Médio (Tempo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU** | 0,90% | 0,41% | 129,13s | 206,27s | **1,00×** (base) | 2,93 | 4,00 |
| **HybridNaive** | 0,90% | 0,44% | 23,09s | 35,83s | **4,57×** | 2,30 | 3,00 |
| **HybridOptimized** | 0,90% | 0,44% | 5,16s | 8,00s | **23,86×** | 2,30 | 2,00 |
| **FullGPUModular** | 0,91% | 0,46% | 2,46s | 4,13s | **71,03×** | 2,47 | 1,00 |

### Teste de Friedman e Pós-Hoc de Nemenyi ($k = 4$, $N = 15$, $\alpha = 0,05$, $CD = 1,2111$)

1. **Qualidade da Solução (`mean_gap`)**:
   + $\chi_F^2 = 2,6889$ ($p = 0,4421$)
   + $F_F(3, 42) = 0,8897$ ($p = 0,4543$)
   + **Veredito**: Hipótese nula **não rejeitada**. Não há diferença estatisticamente significativa de qualidade entre as quatro variantes. Todos os 4 algoritmos formam um único clique no diagrama CD ($|R_i - R_j| \le 1,2111$).
2. **Tempo de Execução (`mean_time`)**:
   + $\chi_F^2 = 45,0000$ ($p < 10^{-6}$)
   + $F_F(3, 42) = \infty$ ($p < 10^{-6}$)
   + **Veredito**: Hipótese nula **rejeitada com significância estrita**. A ordenação de postos de velocidade é perfeita em todas as 15 instâncias: $\text{FullGPUModular} (1,00) > \text{HybridOptimized} (2,00) > \text{HybridNaive} (3,00) > \text{CPU} (4,00)$.
   + Diferenças significativas no pós-hoc de Nemenyi: `CPU vs. FullGPUModular` ($p < 10^{-4}$), `CPU vs. HybridOptimized` ($p = 0,0001$) e `HybridNaive vs. FullGPUModular` ($p = 0,0001$).

---

## 5. Batch 3 e Batch 4: Atualização da Dissertação, Refinamento Visual e Contrato de Figuras

### Batch 3: Consolidação Textual da Dissertação (`Task 14.4`, `14.5`, `13.31`, `13.32`, `12.5`)

- Expandida a Tabela 4.1 (`tbl:results-detailed`) em [04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex) com `d198`, `pr264`, `att532` e `si535`, adicionada a tabela integral das 15 instâncias no Apêndice A.6 ([07-apendices.tex](../../documents_tcc/tex/thesis/parts/07-apendices.tex)) e removida a estratificação por tamanho (`tbl:best-algorithm`).
- Incorporados os dois diagramas TikZ de Diferença Crítica (`chapter4_validation_cd_diagram_time.tikz` e `chapter4_validation_cd_diagram_gap.tikz`), a tabela de decomposição em 8 fases (`tbl:phase-timings`, Lei de Amdahl) e as lições de `notes_project_direction_questions.md` em registro acessível à Pesquisa Operacional ([02-fundamentacao.tex](../../documents_tcc/tex/thesis/parts/02-fundamentacao.tex), [03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex), [05-discussao.tex](../../documents_tcc/tex/thesis/parts/05-discussao.tex), [06-conclusoes.tex](../../documents_tcc/tex/thesis/parts/06-conclusoes.tex)).

### Batch 4: Escalonamento Contínuo/IQR, Degraus e Automação de Macros (`Task 14.6`, `Task 14.7`, `DP14-5`)

1. **Eixo Contínuo Numérico e Faixa IQR (`Task 14.6`)**:
   + Parametrizados `dispersion_metric` (`"iqr" | "std" | "ci"`, padrão `"iqr"`), `x_axis_mode` (`"continuous" | "ordinal"`, padrão `"continuous"`) e `x_tick_mode` (`"instances" | "auto"`, padrão `"instances"`) em [scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py) e [scaling_plots.py::_summarize_metric](../../src/benchmarking/reporting/scaling_plots.py).
   + Implementada projeção isotônica (PAVA) com restrição bilateral de bordas (`[left_bound_pts, right_bound_pts]`), medição dinâmica da geometria tipográfica ativa (`Libertinus Serif` ou fallback `DejaVu Serif`) e escalonamento vertical em dois níveis a 45° em [scaling_plots.py::_apply_continuous_instance_ticks](../../src/benchmarking/reporting/scaling_plots.py), além do mapeamento ordinal indexado por nome da instância para preservar instâncias distintas com mesma dimensão $n$ (verificado via SAT em [test_unit_scaling_plots.py::test_continuous_instance_ticks_no_overlap_across_15_cohort_and_ordinal_mode](../../tests/benchmarking/test_unit_scaling_plots.py) e [test_unit_scaling_plots.py::test_continuous_ticks_narrow_right_cluster_5digit_fallback_and_ordinal_ties](../../tests/benchmarking/test_unit_scaling_plots.py)).
   + Configurado [run_chapter4_benchmark.py::generate_thesis_scaling_plots](../../benchmark_scripts/experiments-and-examples/chapter4/run_chapter4_benchmark.py) para emitir `results_v4/figures/scaling_time_gap.{pdf,svg}` com eixo contínuo em $n$, ticks exatos das instâncias e faixa interquartil (`Faixa IQR ($Q_{25}$–$Q_{75}$, mediana)`).
   + Documentados em [04-resultados.tex](../../documents_tcc/tex/thesis/parts/04-resultados.tex) ambos os degraus empíricos com os dados canônicos `n = 30`: (i) queda de *gap* em `berlin52` (`0,04%`–`0,15%`, mediana `0,00%`, `95/120` paradas por `hit_optimal` em `4,6`–`4,8` gerações vs. `4/120` em `eil51`) e piso CUDA (`14,0`–`42,7 ms`, `0,5`–`1,0 ms/geração` em `n < 100`, remetendo às Tabelas `tbl:phase-timings` e `tbl:phase-timings-all`); (ii) redução de tempo total até a parada entre `att532` (`158,24 s` CPU) e `si535` (`84,19 s` CPU, `EXPLICIT` com `30/30` paradas `within_1pct_optimal` em `62,6`–`63,0` gerações).

2. **Automação de Macros LaTeX (`Task 14.7`)**:
   + Implementada [latex_macros.py::generate_latex_macros](../../src/benchmarking/reporting/latex_macros.py) integrada a [generator.py::generate_result_tables](../../src/benchmarking/reporting/generator.py), exportando `results_v4/tables/benchmark_macros.tex` com comandos `\newcommand{\macro}{valor}` estritamente quantitativos em padrão pt-BR (`{,}`), cobrindo médias/desvios agregados (com fallback para `mean_cost` e `successful_runs` quando `mean_gap` ou `repetitions` são omitidos), speedups geométricos em todos os pares, Friedman/Iman–Davenport ($\chi_F^2$, $F_F$, $p$), Nemenyi ($CD$ e postos médios) e Wilcoxon + Holm–Bonferroni pareado.
   + Conforme decisão direta do usuário, os arquivos `.tex` dos capítulos mantêm seus literais numéricos nesta etapa (substituição por `\macro` reservada para revisão posterior), e a função foi validada por testes unitários dedicados em [test_unit_report_generator.py::test_generate_latex_macros_neutral_default_and_custom_separator](../../tests/benchmarking/test_unit_report_generator.py).

3. **Contrato Padronizado de Geração de Figuras (`DP14-5`, ver [implementation-plotting-tools.md](../../documents_tcc/plan/knowledge-base/implementation-plotting-tools.md#35-unified-figure-generation-contract-dp14-5))**:

| Família de Figura | Ferramenta / Módulo | Formato e Tipografia | Diretório de Saída e Inclusão LaTeX | Escopo |
| :--- | :--- | :--- | :--- | :--- |
| Diagramas de Diferença Crítica (Demšar) | TikZ nativo via [cd_diagram.py::export_cd_diagram_tikz](../../src/benchmarking/reporting/cd_diagram.py) | Vetorial `.tikz` (`scale=2.2`, cotovelos escalonados, fonte herdada de `second_draft.tex`: `Libertinus Serif`) | `results_v4/tables/<benchmark>_cd_diagram_{gap,time}.tikz`, incluído via `\input{...}` dentro de `adjustbox` | Diagramas analíticos de postos médios, cliques não significativos e limiar $CD$ |
| Curvas Empíricas de Escalonamento | Matplotlib (`Agg`) via [scaling_plots.py::generate_scaling_plots](../../src/benchmarking/reporting/scaling_plots.py) | Vetorial `.pdf` e `.svg` (zero `.png` padrão), `pdf.fonttype = 42`, `ps.fonttype = 42`, `Libertinus Serif`, largura útil ABNT `6,30 in` (`TEXT_WIDTH_INCHES`) | `results_v4/figures/scaling_time_gap.{pdf,svg}`, incluído exclusivamente via `\includegraphics[width=\linewidth]{...pdf}` | Séries empíricas com escala logarítmica, bandas de dispersão (`IQR`/`SD`/`CI`) e métricas contínuas ou ordinais |
| Macros Quantitativas | Gerador LaTeX via [latex_macros.py::generate_latex_macros](../../src/benchmarking/reporting/latex_macros.py) | `\newcommand` estritamente numérico em padrão decimal configurável (`{,}`) | `results_v4/tables/benchmark_macros.tex` (opcional via `\input{...}` no preâmbulo) | Contadores de coorte, médias, desvios, speedups e estatísticas globais |
