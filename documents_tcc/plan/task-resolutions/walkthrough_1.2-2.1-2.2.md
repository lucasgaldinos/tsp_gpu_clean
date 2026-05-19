---
title: "Walkthrough: Task 1.2-2.1-2.2 — Simplificação de Hipóteses e Reescrita dos Capítulos 5 e 6"
task_id: "1.2-2.1-2.2"
description: "Remoção de hipóteses especulativas sem suporte empírico dos Capítulos 4 e 5, conversão para TODOs investigativos, e adequação da Conclusão ao status de relatório parcial."
created: "2026-05-13 20:00 -0300"
author: "[[Lucas Galdino]]"
status: "done"
tags: [latex, documentation, report, bugfix, metaheuristics, genetic-algorithm, 2-opt]
links:
  - "[plan-roteiro.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/plan-roteiro.md)"
---

# Walkthrough: Task 1.2-2.1-2.2 — Simplificação de Hipóteses e Reescrita dos Capítulos 5 e 6

> [!IMPORTANT]
> Este Walkthrough registra um conjunto de edições já aprovadas e aplicadas ao [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex).

## Objective

Remover das seções de Resultados (Cap. 4) e Discussão (Cap. 5) as afirmações especulativas sobre causas arquiteturais (pressão de registradores, divergência de warps, OX sequencial) sem suporte de profiling. Substituí-las por afirmações conservadoras e TODOs investigativos formais. Adequar a Conclusão (Cap. 6) ao status de relatório parcial.

---

## Context: Code ↔ Documentation ↔ Data Correlation

| Artifact | Location | Relevant Observation |
|---|---|---|
| Kernel CUDA (FullGPU) | [ga_fujimoto.cu L19](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu#L19) | Usa `float*` para distâncias (corrigido em Ação 3.1). O epsilon `1e-5f` (L56) era consequência direta disso. |
| Matriz de distâncias | [matrix.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py) | Cast `astype(xp.float32)` na L229 confirmava que dados entravam na GPU como float. |
| Estatísticas agregadas | [cross_problem_stats.json](file:///home/lucas_galdino/chimera/gpu_accelerated/tmp/stats_output/cross_problem_stats.json) | FullGPU: gap médio 0.88% ± 1.14. HybridNaive: 1.28% ± 1.68. Friedman p ≈ 6.9e-5. |
| Texto LaTeX (Cap. 4) | [second_draft.tex L2476-2491](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2476) | Hipótese de registradores/warps sem suporte empírico. |
| Texto LaTeX (Cap. 5) | [second_draft.tex L2612-2628](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2612) | Jargão "Efeito Heisenberg" e comparação LKH-3/EAX. |
| Texto LaTeX (Cap. 6) | [second_draft.tex L2637-2651](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2637) | Linguagem de conclusão final ("cumpriu a proposta") inadequada para relatório parcial. |
| TODOs existentes | [second_draft.tex L2630-2635](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2630) | Já registravam injustiça paramétrica e análise de variância. |

---

## Impact Analysis

### Root Cause

Sem profiling granular (Nsight Compute), inferir causa arquitetural específica (registradores, divergência de warps) é especulação pura. Muitas anomalias de performance podem ser *mistas* — efeitos algorítmicos e arquiteturais simultâneos. O texto da versão anterior tratava essas hipóteses como fatos estabelecidos, o que é academicamente incorreto para um relatório parcial.

Adicionalmente, o "Efeito Heisenberg" não é um termo técnico consagrado na literatura CUDA/GPU. O fato real (serialização de kernels e replay passes pelo Nsight Compute) é documentado, mas o jargão não tem referência bibliográfica própria.

### Design Decisions

| Component | Before | After | Rationale |
|---|---|---|---|
| Hipóteses Cap. 4 | Afirmações diretas (registradores, warps, OX) | Reconhecimento de causas mistas + referência a Trabalhos Futuros | Sem profiler, hipóteses específicas são especulação |
| Explicação da superioridade FullGPU | "Permanência na GPU" (vaga) | Bug das 50 vs. 10 iterações hardcoded (concreto e documentado) | A causa mais verificável deve ser apresentada primeiro |
| Seção 5.1 | Jargão "Efeito Heisenberg" | Linguagem técnica direta com citação CUDA | Rigor terminológico exige termos bibliograficamente suportados |
| Seção 5.2 | Tom de autoridade sobre LKH-3/EAX | Abertura para investigação futura | O GA+2-opt atual não compete com solucionadores exatos; é honesto admiti-lo |
| Cap. 6 abertura | "cumpriu a proposta" | "documenta o progresso" | O trabalho não está concluído; é um relatório parcial |
| Cap. 6 Warp-Shuffle | "reduzirá" (promessa) | "tem o potencial de reduzir" (possibilidade) | Sem teste, não se pode afirmar redução |

---

## Affected Files

### LaTeX (1 file, múltiplas seções)

1. [second_draft.tex L2482-2491](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2482) — Ação 1.2: hipóteses de registradores/warps → causas mistas
2. [second_draft.tex L2556-2560](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2556) — Ação 1.2: causa de qualidade FullGPU → bug 50 iterações
3. [second_draft.tex L2632-2635](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2632) — Ação 1.2: 3 novos TODOs investigativos inseridos
4. [second_draft.tex L2618-2622](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2618) — Ação 2.1: Seção 5.1 reescrita sem "Efeito Heisenberg"
5. [second_draft.tex L2626-2628](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2626) — Ação 2.1: Seção 5.2 com frase de investigação futura
6. [second_draft.tex L2639](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2639) — Ação 2.2: abertura Cap. 6 "cumpriu" → "documenta o progresso"
7. [second_draft.tex L2649](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2649) — Ação 2.2: Warp-Shuffle "reduzirá" → "tem o potencial de reduzir"

---

## Change Checklist

- [x] 1. Cap. 4 (L2482-2491) — hipóteses especulativas → causas mistas + ref. Trabalhos Futuros
- [x] 2. Cap. 4 (L2556-2560) — hipótese de qualidade FullGPU → bug iso-algorítmico (concreto)
- [x] 3. Cap. 4 TODOs (L2632-2635) — 3 novos itens de investigação inseridos
- [x] 4. Cap. 5 Seção 5.1 (L2618-2622) — "Efeito Heisenberg" removido; linguagem técnica direta
- [x] 5. Cap. 5 Seção 5.2 (L2626-2628) — frase de investigação futura LK-move adicionada
- [x] 6. Cap. 6 abertura (L2639) — "cumpriu" → "documenta o progresso"
- [x] 7. Cap. 6 Warp-Shuffle (L2649) — "reduzirá" → "tem o potencial de reduzir"

---

## Deep Analysis

### Ação 1.2: Passagem 1 — Hipótese de Registradores e Warps (L2482-2491)

**Texto original:**
```latex
Na ausência de dados empíricos extraídos por um \emph{profiler} de
\emph{hardware}, levanta-se a hipótese de que essa degradação de desempenho
bruto ocorra devido à alta pressão de registradores e à potencial divergência
de \emph{warps}. Como todos os operadores genéticos --- incluindo o cruzamento
OX, que possui características inerentemente sequenciais --- são executados como
\emph{kernels} CUDA interconectados, a concorrência pelos recursos escassos dos
Multiprocessadores de Fluxo (SMs) pode reduzir drasticamente a ocupação global
da GPU. Ainda assim, essa permanência estrita dos dados no dispositivo evita as
barreiras de latência do barramento PCIe, favorecendo uma exploração ininterrupta
que converge estatisticamente para melhores ótimos locais.
```

**Substituído por:**
```latex
Na ausência de dados empíricos extraídos por ferramentas de \emph{profiling}
de \emph{hardware}, as causas exatas dessa degradação de velocidade em instâncias
grandes permanecem em aberto. Os fatores podem estar relacionados tanto à
natureza do algoritmo genético escolhido quanto a limitações arquiteturais da
GPU, e a investigação detalhada dessas hipóteses constitui trabalho futuro
(ver Seção~\ref{sec-61-trabalhos-futuros}). Ainda assim, a permanência dos
dados no dispositivo evita as barreiras de latência do barramento PCIe,
favorecendo uma exploração ininterrupta que converge estatisticamente para
melhores ótimos locais.
```

### Ação 1.2: Passagem 2 — Superioridade de Qualidade FullGPU (L2556-2560)

**Texto original:** A hipótese era que a permanência dos dados na GPU permite exploração mais fluida. **Vaga e não verificável.**

**Substituído por:** A causa direta é o bug de 50 iterações hardcoded vs. 10 nas demais variantes. **Concreto, documentado, e corrigido pela Ação 3.1.**

### Ação 2.1: Seção 5.1 — "Efeito Heisenberg"

O termo "Efeito Heisenberg" no contexto de profiling GPU é usado informalmente na comunidade, mas **não aparece na documentação oficial NVIDIA** (`nvidia2024cuda`) nem em papers peer-reviewed. O fato subjacente (serialização e replay passes) está correto e é documentado — apenas o rótulo jornalístico foi removido. A nova redação cita o mesmo `\autocite{nvidia2024cuda}` com linguagem técnica direta.

### Ação 2.2: "Cumpriu a proposta"

Relatórios parciais de TCC não devem usar linguagem de conclusão final. "Cumpriu a proposta" implica que o trabalho está encerrado e os objetivos foram plenamente atingidos. Como a amostragem experimental ainda não foi refeita com aritmética int32 (Ação 3.1 concluída, mas benchmarks pendentes de re-execução), afirmar conclusão seria impreciso.

---

## Review Notes

- A mudança da explicação da qualidade FullGPU (vaga → bug concreto) é a mais significativa para o orientador: transforma uma hipótese especulativa em evidência direta de problema metodológico já corrigido.
- Os 3 TODOs investigativos inseridos (registradores, divergência de warps, complexidade algorítmica vs. arquitetural) mapeiam diretamente para as Fases 1 e 2 do plano de instrumentação em [knowledge_ferramentas_perfilacao_cuda.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/knowledge-base/knowledge_ferramentas_perfilacao_cuda.md).
