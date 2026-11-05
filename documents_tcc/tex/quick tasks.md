# Deep Analysis — `second_draft.tex` (snapshot `second_draft_build_5_5_20260525_fixed2`)

## Compile steps (you asked)

```bash
# Canonical thesis (documents_tcc/tex/second_draft.tex):
cd documents_tcc/tex && latexmk -lualatex -shell-escape second_draft.tex

# This snapshot:
cd tmp/second_draft_build_5_5_20260525_fixed2 && latexmk -lualatex -shell-escape second_draft.tex
```

- `latexmk` auto-runs `biber` (biblatex/abnt) — no separate biber step needed.
- Engine guard inside file: refuses anything but LuaLaTeX. Never use `pdflatex`/`xelatex`.
- Don't add `-interaction=nonstopmode` for diagnosis (cascades errors).
- Build health of this snapshot: **good** — 1 warning, 6 overfull, 8 underfull, 0 undefined refs, PDF (807KB) builds.

## Why it feels unorganized — root causes (ranked)

| # | Problem | Evidence | Impact |
| --- | --- | --- | --- |
| 1 | **Section hierarchy inverted** | Ch 1–4 = `\section`, Discussão/Conclusões = `\chapter` | TOC numbers: ch 1–4 show "0.1–0.4", Discussão = "1", Conclusões = "2". Double numbering everywhere: "0.1.1 **1.1 Contexto**". |
| 2 | **Ch 2 off-by-one level** | "2. Fundamentação" = `\subsection`, its content = `\subsubsection`; ch 1/3/4 use `\subsection` for real sections | Ch 2 sits one level deeper in TOC; also has redundant wrapper ("Capítulo 2 - Revisão bibliográfica" → "2. Fundamentação teórica e revisão bibliográfica" = duplicate intro). |
| 3 | **Pandoc garbage visible in PDF** | `{[}\^{}isoalgorithm-disambiguation{]}`, `[^ingenuity-disambiguation]`, `[^small-variations]` inline in Seção 1.4/1.5; `\tightlist`, `\def\labelenumi`, `\csname tableofcontents\endcsname`, pandoc-style longtables | Literal `[^...]` footnote syntax printed in the body. Worst visible artifact. |
| 4 | **`\section*{TODOs Pendentes}` inside thesis body** | Line 2552, sits inside Discussão chapter | A TODO list has no place in a thesis; it also **contradicts the chapter** (see below). Move to external task tracking. |
| 5 | **Monolith** | 2582 lines, 145KB, single file; `parts/` dir exists but empty | Hard to navigate, diff, and delegate edits. |
| 6 | **Manual numbering in titles** | "Capítulo 1 -- Introdução", "1.1 Contexto..." hardcoded | Fights LaTeX numbering; breaks cross-refs and ABNT compliance. |
| 7 | **Mixed label conventions** | `sec-41-...` vs `sec:discussao` vs `cap6-conclusoes` | Cosmetic, but adds confusion. |
| 8 | **Front matter leftover** | `\maketitle` temp + ABNT TODO (`\local`, `\instituicao`, `\preambulo` missing); Glossário/Símbolos as embedded pandoc longtables; `\date{2025-11-22}` vs today 2026-09-21 | Not ABNT-final yet. |

## Discussão (lines 2534–2562) — why confusing

**Current shape:** chapter intro = results recap → "Limitações da Análise" (1 limitation only) → "Comparação com Estado da Arte" → "TODOs Pendentes". No arc.

Problems:

1. **Intro is a recap, not discussion.** Repeats FullGPU 0.88% / HybridOptimized 5.69× — same numbers as Conclusões. Duplication across both chapters.
2. **Self-contradiction.** Intro asserts FullGPU dominance as fact; next page the TODO section + inline comment (line 2529) says that conclusion is **invalidated** (50 vs 10 2-opt iterations). Reader sees thesis arguing against itself. This is the #1 cause of "confusing".
3. **One limitation only (profiling).** Missing the real ones: parametric asymmetry, single GPU/hardware (GTX 1050), CPU omitted n>100, metric limits, stopping criteria.
4. **No answer to research question.** `notes_project_direction_questions.md` defines primary question: "which variant offers best trade-off speed vs quality" — Discussão should answer it explicitly. Never does.
5. **Memory axis missing.** Notes framing: runtime, **memory**, population, quality. Ch 3 covers VRAM limits; Discussão ignores them entirely.
6. **TODO list breaks tone** and duplicates future-work items.

**Proposed restructure:**

- 5.1 Síntese dos achados (answer research question: trade-off table, memory/communication axis)
- 5.2 Comparação com o estado da arte (keep, good content)
- 5.3 Ameaças à validade e limitações (profiling + parametric asymmetry + single hardware + CPU omission + metric limits)
- 5.4 Implicações práticas (when to use which variant — matches advisor framing: application-driven)
- Delete "TODOs Pendentes" from body.

## Conclusões (lines 2563–2582) — why confusing

Problems:

1. **No "Conclusões" section.** Only `\section{Trabalhos Futuros}`. Conclusion paragraphs float under chapter heading. Ch 1.6 *promises* "responde explicitamente aos objetivos" — currently doesn't.
2. **Wrong tone:** "O presente **relatório** documenta o **progresso**" — progress-report wording in a thesis.
3. **Duplication:** two paragraphs re-state Discussão numbers; future-work items duplicate Discussão TODOs (instrumentation, `__shfl_sync`, int64). Same content in 3 places.
4. **Vague headline:** "milhares de vezes mais rápida" — no number.
5. **Future work too code-level** (`__shfl_sync`, int64 routines). Advisor notes: code detail → appendix; frame as research questions.
6. **Missing directed-graph path.** Meeting notes: 2-opt works only on undirected graphs → 3-opt/ATSP as future work. Currently 3-opt listed only as "buscas mais agressivas".

**Proposed restructure:**

- 6.1 Conclusões (answer each objective from Seção 1.3/1.4 explicitly, with numbers)
- 6.2 Contribuições
- 6.3 Limitações (one short paragraph, cross-ref Ch 5)
- 6.4 Trabalhos Futuros (4 items, reframed as research questions; add ATSP/directed)

## Cross-check vs your notes (inconsistencies found)

| Note | Says | Thesis says | Action |
| --- | --- | --- | --- |
| `HARDWARE_SPECS.md` | 30 repetitions | "26 repetições independentes" (Ch 4.1) | **Decide: 26 or 30.** Affects all statistical claims. |
| `HARDWARE_SPECS.md` | `two_opt_iterations: 10` | TODO: FullGPU ran with 50 | Already tracked; but Discussão intro still claims FullGPU superiority — must soften until re-run. |
| `HARDWARE_SPECS.md` | CPU for n < 100 | CPU ≤ 100 | Minor, unify. |
| `meeting-notes.md` (advisor) | no code detail in body, code in appendix | class names (`.cu`, `GeneticAlgorithmCPU`...), kernel-level future work in body | Trim kernel-specific detail; keep pseudocode (already good via `algorithm2e`). |
| `meeting-notes.md` | 2-opt undirected-only → 3-opt future work | 3-opt mentioned but directed limitation not stated | Add ATSP/3-opt future-work item. |
| `notes_project_direction_questions.md` | memory behavior = core analysis axis | absent from Discussão/Conclusões | Add. |
| `daily_planning.md` | "demystify code" with architecture diagrams | thesis has 7 Mermaid diagrams (good) | Verify flowchart/class diagram map to `documentation/diagrams/*.md`. |

Build itself is healthy — problem is **structure, not compile**.

## Priority fix order

- [x] ~~1. Fix pandoc `[^...]` literals in 1.4/1.5 (visible garbage, 5 min).~~ **DONE 2026-09-21** — found 6 (`isoalgorithm`, `ingenuity`, `small-variations` ×2, `trivial`, `{[}!warning{]}` admonition) → inline prose.
- [x] ~~2. Convert ch 1–4 to real `\chapter`, strip manual numbers from titles; fix ch 2 depth + remove redundant wrapper (mechanical, biggest payoff).~~ **DONE 2026-09-21** — TOC now `1 Introdução … 6 Conclusões`; ch2 = single `\chapter` (2.1–2.7 `\section`, 2.6.1/2.7.1 `\subsection`).
- [x] ~~3. Remove "TODOs Pendentes" section; keep TODO comments only.~~ **DONE 2026-09-21** — section + 6 items deleted; 2 in-text refs repointed to `sec-61-trabalhos-futuros`.
- [x] ~~4. Restructure Discussão + Conclusões per proposals above (rewrite ~40 lines total).~~ **DONE 2026-09-21** — Discussão: 5.1 Síntese (answers research question + memory axis, `tbl:memory-axis`), 5.2 Estado da arte, 5.3 Ameaças (5 subsecções), 5.4 Implicações práticas. Conclusões: 6.1–6.4. Self-contradiction resolved (FullGPU agora "estatisticamente detectável, porém causalmente ambígua"); tom "relatório/progresso" removido; "milhares de vezes" → \(616{,}9\times\) (máximo real nos dados).
- [x] ~~5. Resolve reps 26 vs 30.~~ **DONE 2026-09-21** — resolvido empiricamente: **26 correto**. 126/134 checkpoints gravam `repetitions=26` (+ `successful_runs=26`, `raw_*` len 26); agregados do Cap. 4 batem com esses dados. `HARDWARE_SPECS.md` corrigido 30→26 (4 locais). Nota: `benchmark.json` diz 15 (config distinto).
- [x] ~~6. Split monolith into `parts/` (after 1–4 done).~~ **DONE 2026-09-21** — `parts/00-frontmatter` … `parts/06-conclusoes`; main = 81 linhas (preâmbulo + `\input`s). Split provado sem perdas (`diff` idêntico, 2837 linhas). Backup: `second_draft.tex.monolith_backup`.

**All items 1–6 complete.** Build: 69 páginas, 0 erros LaTeX, 0 refs indefinidas.

---

## Session 2 — Correções a partir do `review-report-2026-09-21.md` (2026-09-21)

- [x] **Aviso de placeholder no Cap. 4** — caixa de aviso no início do capítulo: valores numéricos não são finais; dois defeitos (assimetria 50 vs 10; aritmética anterior ao `int32`); tabelas mantidas como *placeholders*.
- [x] **Observação (1) em §4.2** — resultados da FullGPU **parcialmente inválidos**: laço 2-opt com 50 iterações em vez de 10 → melhora o *gap* **e** aumenta o tempo. Argumento central: os dois efeitos têm a mesma causa, logo não são evidências independentes.
- [x] **Observação (2) em §4.3** — valores agregados de *gap*/σ **aritmeticamente inválidos**; significância de Friedman/Nemenyi válida só para os dados que a produziram; afirmação de ganho "estrutural" é **prematura**.
- [x] **§3.2 reescrita** — removida a justificativa fabricada ("mitigação de *timeouts*", "variantes CUDA" no plural). Nova redação: o parâmetro de convergência não era propagado ao kernel, que usava limite fixo de 50 iterações como critério de parada **por engano**; defeito de implementação, **exclusivo da FullGPU**, já corrigido no código. Contradição afirmar/retratar resolvida.
- [x] **Fatos CRÍTICOS corrigidos** — `float64`/8 bytes → **`int32`/4 bytes** (conforme `matrix.py`); 1,83 GB → **0,91 GB**; removida a frase sobre `float32` (sem objeto); parágrafo de "precisão dupla mandatória" → **aritmética inteira** (`int32` distâncias, `int64` custo); `n_pop` "adaptativo, 2×n_coords" → **fixo em 256** (conforme `algorithms.json`, verificado).
- [x] **`review-report-2026-09-21.md` §4 corrigido** — retratada a conclusão de que o `50` não influenciou os resultados salvos; registrada a confirmação do autor de que influenciou em ambas as direções. Entrada adicionada ao `modifications`.
- [x] **Incidente: `pandoc_support.tex` ausente** — o arquivo havia sido removido (`rm -f`) enquanto `\input{pandoc_support.tex}` permanecia no preâmbulo, **quebrando o build por completo** (erro fatal, sem PDF). Restaurado a partir de `documents_tcc/tex/pandoc_support.tex` (fonte canônica, 625 B).

### Ainda em aberto (não solicitado nesta sessão)

- Nomenclatura "isoalgorítmica" — o orientador rejeitou o termo (`review §7`). Não alterado nesta sessão por decisão explícita.
- Erros CRÍTICOS restantes do review não cobertos: `Shapiro–Wilk`/`Cohen d` atribuídos a `demsar2006statistical` (Cap. 3 contradiz Cap. 2); limites de categorias de tamanho (3 versões diferentes no documento); Holm–Bonferroni implementado mas ausente do texto.
- `results_v2` continua inválido (pré-`int32`) — re-execução necessária para substituir os *placeholders*.
- `benchmark.json` diz `repetitions: 15` (config distinto dos 26 usados).

---

## Session 3 — Coerência de placeholders e verificações (2026-09-21)

Mecanismo único adotado: `\begin{aviso}[título]` (tcolorbox, `most`), definido no preâmbulo de `second_draft.tex`. Substitui as apresentações mistas (texto simples, `\begin{quote}`, comentário invisível).

- [x] **Cap. 3, memória (linha 20)** — reescrito. Antes eu apenas hesitava sobre um número não verificado. Verificado agora: a análise interna `max_memory_calculation.md` assumia matriz `float64` **e** buffers materializados de O(n²/2), resultando em `n≈13.000`. Os kernels reais **não materializam** esses buffers (`two_opt_batch.cu` mantém um melhor movimento por thread em shared memory O(blockDim); `two_opt_device` usa registradores). Estado residente real: `4n² + 2048n + 2048` bytes → teto ditado pela matriz, `n ≲ 26.000` sob a mesma margem de 65%. O texto agora traz a fórmula, explica por que a estimativa anterior superestimava, e não afirma um limite medido.
- [x] **Cap. 3, aritmética inteira (linha 60)** — verificado nas 4 afirmações. (i) `int64` para custo: **verdadeiro** (`fitness_gpu` int64). (ii) `int32` distâncias: **verdadeiro** (`matrix.py`, e assinatura `const int *dist`). (iii) "sem ponto flutuante no caminho crítico": **verdadeiro para avaliação/busca**, mas float existe na *construção* — texto corrigido para dizer isso. (iv) "nint vale para coordenadas **e** para a avaliação de melhorias": **falso** — o arredondamento ocorre **uma única vez**, na construção; a busca compara inteiros exatos (comentário no kernel: "Exact integer comparison, no epsilon needed").
- [x] **tsplib95 (pergunta explícita)** — `tsplib95` **não está instalado**; o projeto usa `tsplib_parser` próprio. `matrix.py` implementa a convenção da TSPLIB (`nint(x)=int(x+0.5)`), com regras próprias para `EUC_2D/3D`, `MAN`, `MAX`, `CEIL_2D`, `ATT` (regra `tij+1`) e `GEO` (RRR=6378.388). Texto ajustado: "especificação da TSPLIB" (Reinelt), não "biblioteca".
- [x] **Cap. 3, premissa isoalgorítmica (linha 141)** — TODO explícito adicionado no fonte, registrando: o orientador rejeitou o termo; a premissa voltará a ser falsa após a re-execução; e os dois motivos (operadores 2-opt não equivalentes; contagem de gerações divergente). Não foi feito o rename global, conforme decisão.
- [x] **Cap. 3, população (linha 154)** — fundamentado por pesquisa (subagente): modelo *gambler's ruin* indica crescimento sublinear com n; faixa prática 50–500 para TSP; Fujimoto & Tsutsui usaram 60 em GPU. Sobre "explodir memória": **não é explosão assintótica, mas dobra do termo dominante** — a razão `M_pop/M_dist = P/n` empata em `P = n`, logo `P = 2n` torna a população um segundo termo quadrático de coeficiente 2× o da matriz. Em n=25.000, `P=2n` sozinho exigiria 4,657 GB. Isso justifica o valor fixo.
- [x] **Cap. 5, soluções da literatura (linha 20)** — respondido: armazenamento triangular compactado (metade exata), recálculo sob demanda, *neighbourhood splitting* (Schulz 2013, com limiares concretos), e transferência em blocos. Citado.
- [x] **Cap. 4, coerência** — 4 admonições `aviso` numeradas (1–4), substituindo as apresentações mistas. A Observação (4) converteu o `% TODO` invisível em aviso visível no PDF.
- [x] **Abstract + Cap. 5/6** — `aviso` de provisoriedade adicionado ao Resumo, a §5.1, à abertura das Conclusões (6.1) e à discussão sobre implicações práticas.
- [x] **§5.3.1** — acrescentada a assimetria de **semântica dos operadores** 2-opt (FullGPU = varredura até convergência, first-improvement; CPU = best-improvement; híbridas = uma passada), mais a divergência de contagem de gerações. Registrado que essa caracterização é mais defensável que a assimetria dos 50, por ser observável no código atual.

**Verificação:** 70 páginas, 0 erros LaTeX, 0 refs/citações indefinidas, 4 admonições renderizadas. Números falsos (`1,83 GB`, `precisão dupla`) ausentes do PDF. 2 novas referências adicionadas a `refs.bib` (`harik1999gambler`, `goldberg1992massive`).

**Relatório de pesquisa gerado:** `documentation/knowledge-base/research/literature-review-gpu-memetic-tsp.md` (651 linhas, 20 referências, 10 ressalvas — inclui uma correção de fato sobre limites de shared memory da GTX 1050: `sharedMemPerBlockOptin` = 49152, ou seja, o opt-in **não** excede 48 KB/bloco neste modelo, ao contrário do que a documentação interna sugeria).
