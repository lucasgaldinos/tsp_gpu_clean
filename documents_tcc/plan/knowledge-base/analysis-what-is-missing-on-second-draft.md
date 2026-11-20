---
title: "Divergências entre `second_draft.tex` e a coorte consolidada de `results_v4`"
description: >
  Diagnóstico registrado em 2026-11-17, logo após o término da coleta (30 repetições de
  `pr1002_CPU`), que confronta os números do texto da dissertação com os checkpoints
  consolidados de `results_v4` e identifica os artefatos defasados (gráfico de
  escalonamento com rótulos em inglês e curva da CPU truncada, tabelas com estratos
  antigos, diagrama CD não embutido, importação DuckDB pendente) e as passagens do texto
  que ficaram obsoletas.
created: 2026-11-17
status: review
author:
  - "Lucas Galdino"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-17
    modifications:
      - description: >
          Inclusão do frontmatter YAML ausente e reestruturação do arquivo, que era uma
          resposta de chat colada e começava em `###`. Links `file:///home/...`
          convertidos para caminhos relativos ao repositório, âncoras de linha trocadas
          por referências de símbolo e pendências reconciliadas com o estado atual do
          plano (`\totalrepeticoes{30}` da `Task 13.33`, remoção dos avisos da
          `Task 13.30`, módulo de escalonamento da `Task 13.28` e correção já aplicada da
          remissão em `03-materiais-metodos.tex`).
related_files:
  - "[chapter4_validation.md](../../../results_v4/tables/chapter4_validation.md)"
  - "[scaling_plots.py](../../../src/benchmarking/analysis/scaling_plots.py)"
  - "[04-resultados.tex](../../tex/thesis/parts/04-resultados.tex)"
  - "[report_generator.py](../../../src/benchmarking/analysis/report_generator.py)"
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
tags:
  - analysis/benchmark
  - analysis/planning
  - analysis/statistics
  - guide/benchmarking
  - notes/planning
  - notes/session
  - notes/performance
  - review/quality
  - analysis/architecture
  - guide/statistics
---

# Divergências entre `second_draft.tex` e a coorte consolidada de `results_v4`

Nota de diagnóstico registrada em 2026-11-17, imediatamente após o término da coleta: o
processo em segundo plano concluiu as 30 repetições de `pr1002_CPU` às 03:16:35. O
objetivo é confrontar o texto da dissertação com os checkpoints finais e listar, em um só
lugar, o que precisa ser regenerado e quais passagens ficaram obsoletas. O estado de cada
pendência está reconciliado na Seção 5.

## 1. Resultados consolidados vs. prévia parcial de `pr1002_CPU`

### 1.1. Diferenças entre a prévia parcial (6 repetições) e os dados finais (30 repetições)

Quando o Capítulo 4 foi atualizado (2026-11-16), o checkpoint
[pr1002_CPU.json](../../../results_v4/checkpoints/pr1002_CPU.json) continha apenas 6
repetições ($740{,}06\,\text{s}$, *gap* de $1{,}69\%$). Com as 30 repetições consolidadas,
os valores definitivos mudaram nos seguintes pontos:

| Escopo / Métrica | Valor no `.tex` Atual (Prévia 6 reps) | Valor Final Consolidado (30 reps em `pr1002_CPU`) | Valor Estrito nas Primeiras 30 Reps (`seeds 42..71` em todas) |
| :--- | :---: | :---: | :---: |
| **`pr1002` — CPU Custo Médio ($\pm \sigma$)** | $263\,420{,}33 \pm 1\,118{,}24$ | **$263\,558{,}83 \pm 771{,}47$** | $263\,558{,}83 \pm 771{,}47$ |
| **`pr1002` — CPU *Gap* Médio (%)** | $1{,}69\%$ | **$1{,}74\%$** ($\pm 0{,}30\%$) | $1{,}74\%$ ($\pm 0{,}30\%$) |
| **`pr1002` — CPU Tempo Médio (s)** | $740{,}06\,\text{s}$ | **$682{,}58\,\text{s}$** | $682{,}58\,\text{s}$ |
| **`pr1002` — *Speedup* Híbrida Ingênua** | $6{,}25\times$ | **$5{,}76\times$** | $5{,}77\times$ |
| **`pr1002` — *Speedup* Híbrida Otimizada** | $27{,}69\times$ | **$25{,}54\times$** | $17{,}65\times$ |
| **`pr1002` — *Speedup* Totalmente em GPU** | $50{,}87\times$ | **$46{,}91\times$** | $48{,}72\times$ |
| **Agregado (15 inst.) — Tempo Médio CPU** | $134{,}39\,\text{s}$ | **$130{,}56\,\text{s}$** | $129{,}13\,\text{s}$ |
| **Agregado — *Speedup* Geom. Híbrida Ingênua (vs. CPU)** | $4{,}85\times$ | **$4{,}82\times$** | $4{,}57\times$ |
| **Agregado — *Speedup* Geom. Híbrida Otimizada (vs. CPU)** | $23{,}78\times$ | **$23{,}65\times$** | $20{,}94\times$ |
| **Agregado — *Speedup* Geom. Totalmente em GPU (vs. CPU)** | $73{,}63\times$ | **$73{,}23\times$** | $71{,}03\times$ |
| **Estrato Grande ($n > 400$) — Tempo / *Gap* CPU** | $316{,}77\,\text{s}$ / $1{,}26\%$ | **$307{,}19\,\text{s}$ / $1{,}27\%$** | $303{,}61\,\text{s}$ / $1{,}28\%$ |
| **Teste de Friedman em *Gap* ($F_F$ / $p$)** | $F_F = 1{,}144$, $p = 0{,}342$ | **$F_F = 1{,}144$, $p = 0{,}342$** | $F_F = 0{,}890$, $p = 0{,}454$ |
| **Teste de Friedman em Tempo ($\chi_F^2$ / $p$)** | $\chi_F^2 = 45{,}0$, $p = 9{,}25 \times 10^{-10}$ | **$\chi_F^2 = 45{,}0$, $p = 9{,}25 \times 10^{-10}$** | $\chi_F^2 = 45{,}0$, $p = 9{,}25 \times 10^{-10}$ |

> [!IMPORTANT]
> **Contagem de repetições nos checkpoints (`60` vs. `37` vs. `30`).**
> Dos 60 checkpoints em `results_v4/checkpoints/`, **57 células possuem 60 repetições**, `gr666_CPU.json` possui **37**, e `rat783_CPU.json` e `pr1002_CPU.json` possuem **30**.
> O texto fixou `\totalrepeticoes{30}` como contagem canônica (`Task 13.33`), enquanto as tabelas geradas por [report_generator.py](../../../src/benchmarking/analysis/report_generator.py) usam a média de **todas** as repetições presentes em cada checkpoint (coluna do meio da tabela).
> Decidir entre truncar as células às primeiras 30 repetições (`seeds 42..71`) ou documentar explicitamente o critério permanece aberto na `Task 13.31` (ver Seção 5).

---

## 2. Artefatos de `results_v4` defasados pelo módulo em memória (PID 360242)

O processo iniciado em 2026-11-16 às 17:32 — **antes** da conclusão das `Task 13.16` e `Task 13.28` — terminou às 03:16:35 de 2026-11-17 executando a versão antiga de [report_generator.py](../../../src/benchmarking/analysis/report_generator.py) que estava carregada em memória. Consequências:

1. **Gráfico de escalonamento (`fig:scaling-time-gap`) desatualizado e incompleto.**
   Os arquivos [scaling_time_gap.pdf](../../../results_v4/tables/scaling_time_gap.pdf) e [scaling_time_gap.png](../../../results_v4/tables/scaling_time_gap.png) têm timestamp de `2026-11-16 18:33:00` — antes do término de `rat783_CPU` (21:35) e de `pr1002_CPU` (03:16).
   Três problemas:
   + **Curva da CPU truncada:** a curva azul termina em $n = 783$ (`rat783`) e não possui o ponto $n = 1002$ (`pr1002`), além de usar dados parciais em `rat783`.
   + **Rótulos em inglês:** os títulos, eixos e legendas em [scaling_plots.py](../../../src/benchmarking/analysis/scaling_plots.py) estão em inglês (`Execution Time (s)`, `Optimality Gap (%)`, `Execution Time and Solution Quality Scaling vs. Problem Dimension`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`), enquanto a dissertação exige português brasileiro (`Tempo de Execução (s)`, `Gap de Otimalidade (%)`, `Dimensão do Problema (n)`, `Híbrida Ingênua`, `Híbrida Otimizada`, `Totalmente em GPU`).
   + **Degrau em $n = 535$:** entre `att532` ($158{,}06\,\text{s}$, *gap* $1{,}10\%$) e `si535` ($84{,}39\,\text{s}$, *gap* $0{,}92\%$) há uma queda vertical acentuada em todas as quatro curvas, porque `si535` converge cedo com variância mínima ($\sigma = 0{,}05\%$). O comportamento merece explicação na Seção 4.3.

2. **Tabelas de `results_v4/tables/` com estratos antigos.**
   A `Table 3` de [chapter4_validation.md](../../../results_v4/tables/chapter4_validation.md) ainda usa os estratos `Small (n<100)`, `Medium (100≤n<300)` e `Large (n≥300)`. Reexecutar `generate_result_tables` com o código atual regenera `chapter4_validation.{md,tex}`, `chapter4_validation_cd_diagram.tikz` e `scaling_time_gap.{pdf,png}` com as fronteiras corretas ($n \le 100$, $100 < n \le 400$, $n > 400$) e os 15 pontos de CPU.

3. **Consolidação no DuckDB (`Task 12.4`).** Falta rodar o importador [import_checkpoints.py](../../../src/benchmarking/database/import_checkpoints.py) sobre `results_v4/checkpoints/` para materializar `results_v4/results.duckdb`, que ainda não existe.

---

## 3. Auditoria dos arquivos de `documents_tcc/tex/`

### 3.1. Correções pontuais de consistência

1. **[01-introducao.tex](../../documents_tcc/tex/parts/01-introducao.tex).** Duas passagens ainda limitam a CPU a instâncias pequenas e médias: a frase *"é utilizada como base para instâncias de porte pequeno e médio (até $n_{coords} \leq 400$); para instâncias maiores, a comparação de tempo passa a considerar como referência a versão híbrida"* e a menção a *"viável executar a referência em CPU nas instâncias médias"*.

   **Ação:** atualizar ambas para refletir a cobertura integral da CPU nas 15 instâncias
   (até $n = 1002$), viabilizada pela combinação da lista de candidatos ($N_\kappa = 20$)
   com a busca local vetorizada em NumPy (`Task 13.14`).
2. **[03-materiais-metodos.tex](../../documents_tcc/tex/parts/03-materiais-metodos.tex).** A remissão *"nas Observações do Capítulo 4 e na Seção 5.3.1.1"* **já foi corrigida**: a verificação de 2026-11-17 não encontra mais a menção às Observações do Capítulo 4, o que é consistente com a remoção dos blocos provisórios na `Task 13.30`.
3. **[04-resultados.tex](../../documents_tcc/tex/parts/04-resultados.tex).** Remover a ressalva de que *"instâncias cuja coleta em CPU ainda está em curso são omitidas apenas da respectiva curva"* — as 15 instâncias estão presentes. Atualizar também os números de `pr1002_CPU` e as médias agregadas globais conforme a Seção 1.1, propagando para `00-frontmatter.tex`, `05-discussao.tex` e `06-conclusoes.tex` (`Task 13.31`).

### 3.2. Tabelas e análises de alto valor ainda não exploradas

1. **Tabela completa das 15 instâncias.** A Tabela 4.1 (`tbl:results-detailed`) de [04-resultados.tex](../../documents_tcc/tex/parts/04-resultados.tex) exibe apenas `berlin52`, `kroA100` e `pr1002`, enquanto a introdução promete *"anexos com tabelas completas de resultados"*. As 15 instâncias cabem na própria Tabela 4.1 (que já usa `longtable`, ocupando ~1,5 página) ou em tabela no Apêndice.
2. **Decomposição temporal em 8 fases (achado de Amdahl no `CX` vs. `LS`).** O fluxograma metodológico (`fig:results-flowchart`) e o Apêndice A destacam a telemetria de 8 fases (`Init`, `Sel`, `CX`, `Mut`, `Eval`, `LS`, `Surv`, `Host Sync`), mas os Capítulos 4 e 5 ainda não apresentam os números medidos. A `Table 4` de [chapter4_validation.md](../../../results_v4/tables/chapter4_validation.md) explica mecanicamente cada ganho:
   + **CPU:** `LS` consome de **96,3% a 98,7%** do tempo de parede; `CX` apenas 1,2% a 3,5%.
   + **Híbrida Ingênua:** `LS` (incluindo as 256 transferências PCIe individuais por geração) consome **85,9% a 89,8%**, e `CX` sobe para 9,2% a 13,1%.
   + **Híbrida Otimizada (achado de Amdahl clássico):** com o 2-opt em lote na GPU, a fração de `LS` despenca para **17,6% a 55,1%** e o **cruzamento OX sequencial na CPU vira o novo gargalo dominante**, consumindo de 42,1% (`pr1002`) a 74,0% (`eil51`) do tempo total.
   + **Totalmente em GPU (`FullGPUModular`):** migrar `CX`, `Sel` e `Mut` para kernels CUDA residentes na VRAM elimina o gargalo do OX na CPU (daí o ganho adicional de ~3,1× sobre a Híbrida Otimizada).
     Como os kernels são enfileirados de forma assíncrona em microssegundos, entre **41,2% e 94,5%** do tempo de relógio medido no host recai em `Host Sync`, a barreira onde a CPU aguarda a fila assíncrona da GPU terminar a geração para avaliar o critério de parada.
3. **Inclusão do diagrama de diferença crítica de Demšar.** O TikZ preliminar foi gerado em [chapter4_validation_cd_diagram.tikz](../../../results_v4/archive/pre_n30_trim/chapter4_validation_cd_diagram.tikz) ($CD = 1{,}211$, barra horizontal única conectando as quatro variantes na métrica de *gap*, postos médios de 2,30 a 3,00; posteriormente substituído pelos diagramas duplos [chapter4_validation_cd_diagram_gap.tikz](../../../results_v4/tables/chapter4_validation_cd_diagram_gap.tikz) e [chapter4_validation_cd_diagram_time.tikz](../../../results_v4/tables/chapter4_validation_cd_diagram_time.tikz) após `Task 14.1`/`Task 14.2`).
    Os números já são citados em prosa na Seção 4.3, mas a figura não está embutida em `04-resultados.tex`. Um segundo painel para o ranqueamento de **tempo** (postos médios 1,00 Totalmente em GPU, 2,00 Híbrida Otimizada, 3,00 Híbrida Ingênua, 4,00 CPU) é opcional.

---

## 4. Cruzamento com `notes_project_direction_questions.md`

As notas de direção do projeto ([notes_project_direction_questions.md](../../../notes_project_direction_questions.md#findings)) contêm três pontos técnicos que podem ser consolidados no texto (`Task 12.5`):

1. **Micro-benchmark `candidate_k = 20` vs. `None`.** Restringir a busca aos $N_\kappa = 20$ vizinhos mais próximos acelera o passo de 2-opt em até **8,22× na CPU** e **4,05× na GPU** (`lin318`), enquanto em instâncias muito pequenas (`berlin52`, $n = 52$) o custo de indireção da matriz de candidatos empata com a varredura direta (0,99× na CPU e 0,79× na GPU).
   Cabe uma síntese concisa ou mini-tabela na Seção 3.2.1 ou na Seção 5.1.
2. **GLS, FLS e lista de candidatos $k$-NN.** Em \textcite{voudouris1999tsp}, *Guided Local Search* (GLS) adiciona penalidades dinâmicas à função de custo das arestas para escapar de ótimos locais, enquanto a redução da vizinhança examinada é feita por *Fast Local Search* (FLS) combinada a listas estáticas dos $k$ vizinhos mais próximos (*candidate lists*, de \textcite{lin1973efficient}).
   Na Seção 2.2.1 e na Seção 3.2.1, vale explicitar em poucas frases que este trabalho adota a **poda estática por lista de $N_\kappa = 20$ vizinhos mais próximos** (sem penalização de custos do GLS), esclarecendo a citação a \textcite{voudouris1999tsp}.
3. **Janelamento de memória compartilhada (`MAX_TOUR_SIZE = 2048`).** Explicitar na Seção 3.2.1 e na Seção 5.1 que a janela de memória compartilhada dos kernels CUDA em lote é de **2048 vértices** ($2 \times 2048 \times 4\text{ B} = 16\text{ KB}$ de SRAM por bloco para `s_tour` e `s_pos`), razão pela qual todas as 15 instâncias da coorte ($n \le 1002$) operam integralmente no caminho rápido em SRAM, sem *tiling* em múltiplas janelas.

---

## 5. Estado das pendências (reconciliado em 2026-11-17)

| Pendência | Estado | Onde fecha |
| :-- | :-- | :-- |
| Contagem de repetições do texto | **Resolvida** — `\totalrepeticoes{30}` unificado; checkpoints mantêm 60/37/30 | `Task 13.33` (feita) |
| Truncar tabelas às primeiras 30 repetições ou documentar o critério | **Aberta** | `Task 13.31` |
| Rótulos do gráfico de escalonamento em inglês | **Aberta** — `scaling_plots.py` ainda emite `Execution Time (s)` e `Optimality Gap (%)` | `Task 13.31` |
| Curva da CPU completa até $n = 1002$ no gráfico | **Aberta** — figura gerada em 2026-11-16 18:33 | `Task 13.28` / `Task 13.31` |
| Diagrama CD embutido em `04-resultados.tex` | **Parcial** — $CD = 1{,}211$ citado em prosa; o `.tikz` não está incluído | `Task 13.31` |
| Importação para `results_v4/results.duckdb` | **Aberta** — o arquivo ainda não existe | `Task 12.4` |
| Corte de CPU a 400 e "instâncias médias" em `01-introducao.tex` | **Aberta** — as duas passagens permanecem no texto | `Task 13.31` |
| Ressalva de coleta em curso em `04-resultados.tex` | **Aberta** — o trecho permanece no texto | `Task 13.31` |
| Remissão às Observações do Capítulo 4 em `03-materiais-metodos.tex` | **Resolvida** — a menção não existe mais no arquivo | `Task 13.30` (feita) |
| Tabela 4.1 com as 15 instâncias e tabela de 8 fases | **Aberta** | `Task 13.31` / `Task 13.32` |
| Regeneração de tabelas e figuras com o código atual | **Aberta** — o processo antigo usava o módulo em memória | `Task 12.4` |
| Ajustes finais de CLI e logging | **Abertas** — desbloqueadas com o fim do processo em segundo plano | `Task 13.17`, `Task 13.22`, `Task 13.24` |
