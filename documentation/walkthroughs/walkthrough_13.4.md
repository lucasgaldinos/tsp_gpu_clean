---
title: "Walkthrough: Task 13.4 — Reescrita do Capítulo 3 (Materiais e Métodos)"
task_id: "13.4"
description: "Capítulo 3 alinhado ao protocolo executado: variantes como configurações de um único GeneticAlgorithm, candidate_k = 20 e post_mutation, coorte de 15 instâncias, 30 repetições, CPU até n_coords <= 400, Nemenyi como procedimento de diferença crítica, Holm-Bonferroni e diagramas CD."
created: "2026-11-13 17:30 -03:00"
modifications:
  - date_modified: "2026-11-14"
    modifications:
      - description: >
          Saneamento do documento: pipes dentro de células de tabela escapados,
          delimitadores LaTeX inline convertidos para `$...$` e autoria sem wiki link.
author: "Lucas Galdino"
status: "done"
tags:
  - docs/thesis
  - docs/latex
  - methodology/protocol
  - methodology/statistics
  - review/quality
links:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[analysis-demsar-critical-difference-diagrams.md](../../documents_tcc/plan/knowledge-base/analysis-demsar-critical-difference-diagrams.md)"
  - "[knowledge_candidate_lists_and_inverse_positions.md](../../documents_tcc/plan/knowledge-base/knowledge_candidate_lists_and_inverse_positions.md)"
---

# Walkthrough: Task 13.4 — Reescrita do Capítulo 3 (Materiais e Métodos)

## Objective

Substituir no Capítulo 3 os números e a nomenclatura que não descrevem mais o protocolo executado (38 instâncias, 26 repetições, CPU até $n_{\text{coords}} \le 100$, classes `GeneticAlgorithm*` separadas, "10 varreduras completas") pelos fatos canônicos de `DP13-1`: quatro configurações da mesma classe `GeneticAlgorithm`, `candidate_k = 20` com estágio `post_mutation`, coorte de 15 instâncias (48–1002 cidades), 30 repetições, CPU até $n_{\text{coords}} \le 400$ e protocolo estatístico com Nemenyi como procedimento de diferença crítica, Holm–Bonferroni e diagramas CD.

## Context: Code ↔ Documentation ↔ Data Correlation

| Artifact | Location | Relevant Observation |
| --- | --- | --- |
| Configuração das variantes | [configs/algorithms.json](../../configs/algorithms.json) | Quatro chaves (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`), todas com `metaheuristic: GeneticAlgorithm`, `local_search_stage: post_mutation`, `max_iterations: 10` e `candidate_k: 20`. |
| Script da coorte | [run_chapter4_benchmark.py](../../benchmark_scripts/run_chapter4_benchmark.py#L87-L104) | `PROBLEMS` com 15 instâncias, `REPETITIONS = 30`, `CPU_SIZE_THRESHOLD = 400`, `SEED = 42`. |
| Derivação de sementes | [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L398) | `run_seed = seed_base + current_run_idx` — sementes fixas e distintas por repetição. |
| Compilação de kernels | [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L364-L381) | `warmup(context)` compila os kernels antes do laço medido; `cp.RawKernel` sobre fontes `.cu`. |
| Fórmula da CD | [stats_utils.py](../../src/benchmarking/analysis/stats_utils.py#L151-L166) | `CD = (q_tukey / sqrt(2)) * sqrt(k(k+1)/(6n))`. |
| Correção pareada | [problem_statistics.py](../../src/benchmarking/analysis/problem_statistics.py#L136) | `holm_bonferroni_correction` aplicada aos valores-p pareados. |
| Texto antes da edição | [03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex) | §3.2.1 "10 varreduras completas" e "ao final de cada geração"; §3.2.2 quatro classes `GeneticAlgorithm*`; §3.3 38 instâncias, 26 repetições, $N \le 100$; §3.4 Nemenyi descrito como teste par a par. |

---

## Impact Analysis

### Root Cause

Três números do capítulo (38 instâncias, 26 repetições, CPU $\le 100$) e a nomenclatura de classes deixaram de descrever o experimento: a coorte executada tem 15 instâncias, as repetições são 30, o limiar de CPU é 400 e as variantes são configurações de uma única classe. Além disso, a omissão de `candidate_k` escondia a assimetria semântica entre os operadores $2$-opt que a §5.3.1.1 denuncia, e o protocolo estatístico não registrava Holm–Bonferroni nem os diagramas CD, descrevendo Nemenyi como se ele devolvesse um valor-p por par.

### Design Decisions

| Component | Before | After | Rationale |
| --- | --- | --- | --- |
| Nomes das variantes | `GeneticAlgorithmCPU`, `GeneticAlgorithmHybridNaive`, `GeneticAlgorithmHybridOptimized`, `GeneticAlgorithmFullGPU` | `CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular` | São chaves de configuração, não subclasses; os nomes agora batem com `configs/algorithms.json` e com o Capítulo 4. |
| Item de busca local | "10 varreduras completas da vizinhança por indivíduo", "ao final de cada geração" | `max_iterations = 10` com `candidate_k = 20` (listas de $\kappa$-vizinhos) na CPU, estágio `post_mutation` | O orçamento efetivo em CPU é truncado por candidatos, e a busca roda nos descendentes mutados, não na população inteira ao fim da geração. |
| Assimetria dos operadores | Ausente | Frase remetendo à §5.3.1.1 | O mesmo limite de iterações não representa o mesmo trabalho entre CPU, híbridas e GPU residente. |
| Kernels CUDA | "kernels CUDA (através do CuPy)" | `RawKernel` do CuPy compilado de `.cu` antes do laço medido (*warm-up*) | Descreve o mecanismo real e explica por que o custo de compilação não entra nos tempos. |
| Coorte | 38 instâncias, 51–1002 | 15 instâncias, 48–1002, três faixas preservadas | Mesma estrutura de `longtable`, `\label{tbl:selected-instances}` e referência no texto. |
| Repetições e CPU | 26 repetições; CPU só $N \le 100$ | 30 repetições com sementes derivadas de 42; CPU até $n_{\text{coords}} \le 400$, referência `HybridNaive` acima disso | Alinha o texto ao script que gera os dados e explicita a consequência do corte. |
| Nemenyi | "identificar quais pares diferem" | Procedimento de diferença crítica com $CD = q_\alpha\sqrt{k(k+1)/(6N)}$, $q_\alpha$ já dividido por $\sqrt{2}$ | Nemenyi não produz valor-p por par; ele define um limiar sobre *ranks* médios. |
| Correções e visualização | Só Nemenyi | Holm–Bonferroni *step-down* para os pares + diagramas CD | Cobre a correção efetivamente aplicada e a forma de apresentação. |

### Failure / Correction Log

| Attempt / Event | Evidence Read | Correction / Decision | Outcome |
| --- | --- | --- | --- |
| Interpretar "10 varreduras completas" como orçamento comum às quatro variantes | [two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py#L171-L190) e §5.3.1.1 | Truncamento descrito por `candidate_k = 20` na CPU e por semântica própria nas GPU | Texto não contradiz mais a §5.3.1.1. |
| Atribuir `RawKernel` + *warm-up* a todas as variantes GPU | [algorithm_runner.py](../../src/benchmarking/runner/algorithm_runner.py#L364-L381) | A frase foi colocada apenas no item `FullGPUModular`, onde o mecanismo é integral | Evita generalização indevida para as híbridas. |
| Verificação `rg -n "GeneticAlgorithmCPU\|GeneticAlgorithmFullGPU"` → 0 | Diagrama `mermaid` de §3.1.3 (linhas L122, L137, L144, L147) | **Bloqueado:** o bloco protegido pelo escopo ainda nomeia as classes antigas | 4 ocorrências remanescentes; ver *Additional Findings*. |
| `rg "candidate_k"` não casa com o texto | `\texttt{candidate\_k}` (underscore escapado, obrigatório em LaTeX) | Mantida a convenção `\texttt{}` do arquivo; o termo aparece como `candidate\_k`/`post\_mutation` na fonte | A checagem combinada atinge 9 linhas (limiar $\ge 6$). |

---

## Affected Files

### LaTeX thesis text (1 file)

1. [03-materiais-metodos.tex](../../documents_tcc/tex/thesis/parts/03-materiais-metodos.tex)
   - L193–L196: §3.2.1 — busca local `post_mutation`, `max_iterations = 10` com `candidate_k = 20`, assimetria das GPU.
   - L203: §3.2.2 — quatro variantes como configurações da mesma `GeneticAlgorithm` via `configs/algorithms.json`.
   - L207–L222: §3.2.2 — itens renomeados (`CPU`, `HybridNaive`, `HybridOptimized`, `FullGPUModular`), padrão de transferência preservado, `RawKernel` + *warm-up*.
   - L227–L229: §3.3 — 15 instâncias, 48–1002 cidades.
   - L231–L265: §3.3 — `longtable` preservado; linhas L259–L264 com 5 + 4 + 6 instâncias.
   - L271–L277: §3.3 — 30 repetições, sementes derivadas de 42, CPU até 400 e `HybridNaive` como referência acima disso.
   - L293–L296: §3.4 — Nemenyi como procedimento de diferença crítica (fórmula da CD).
   - L297–L299: §3.4 — Holm–Bonferroni *step-down*.
   - L300–L301: §3.4 — diagramas de diferença crítica.
   - L306: sentença sobre a análise bayesiana de Benavoli — **inalterada**.

---

## Change Checklist

- [x] 1. §3.2.1 — `post_mutation`, `max_iterations = 10` com `candidate_k = 20`, assimetria remetida à §5.3.1.1.
- [x] 2. §3.2.2 — variantes como configurações da classe única; `RawKernel` com *warm-up*.
- [x] 3. §3.3 — 15 instâncias, 48–1002, tabela e rótulo preservados.
- [x] 4. §3.3 — 30 repetições, sementes distintas, CPU até 400.
- [x] 5. §3.4 — Nemenyi como diferença crítica, Holm–Bonferroni, diagramas CD.
- [x] 6. §3.4 — sentença bayesiana intacta; `\label`/`\ref` e blocos `mermaid` preservados.

---

## Validation Evidence

| Check | Evidence | Result | Boundary / Follow-up |
| --- | --- | --- | --- |
| Strings obsoletas | `rg -n "38 instâncias\|26 vezes\|varreduras completas"` → sem saída | pass | — |
| Nomes de classe antigos | `rg -n "GeneticAlgorithmCPU\|GeneticAlgorithmFullGPU"` → 4 linhas (L122, L137, L144, L147) | **blocked** | Todas dentro do diagrama `mermaid` de §3.1.3, que o escopo manda preservar. |
| Termos canônicos | `rg -n "candidate_k\|post_mutation\|30 repetições\|400\|FullGPUModular\|Holm\|diferença crítica"` → 9 linhas | pass ($\ge 6$) | `candidate_k`/`post_mutation` aparecem escapados (`candidate\_k`, `post\_mutation`). |
| Coorte | `rg -o` das 15 instâncias → 1 ocorrência de cada, 15 no total | pass | — |
| Rótulo da tabela | `rg -n "label\{tbl:selected-instances\}"` → L236 | pass | — |
| Gate de documentação | `.venv/bin/python -m benchmark_scripts.check_docs` → `stale docstrings: 0`, `dead links: 0`, `undefined symbols: 0`, exit 0 | pass | O gate não inspeciona `.tex`. |
| Estrutura LaTeX | contagem de `\begin`/`\end` (enumerate 5/5, figure 3/3, itemize 4/4, longtable 1/1, mermaid 3/3, minipage 6/6) e balanço de chaves (profundidade final 0, mínimo 0) | pass | Sem toolchain TeX: nenhuma compilação foi executada. |

---

## Deep Analysis

A fórmula adotada, $CD = q_\alpha\sqrt{k(k+1)/(6N)}$, com $q_\alpha$ já dividido por $\sqrt{2}$, é a mesma de `stats_utils.py` (L151–L166): o `scipy.stats.studentized_range.ppf` devolve a amplitude studentizada na convenção de Tukey, e a divisão por $\sqrt{2}$ converte para a escala de diferenças entre dois *ranks* médios. Com $k = 4$ e $N = 15$, o fator $\sqrt{k(k+1)/(6N)} = \sqrt{20/90} \approx 0{,}471$; para $\alpha = 0{,}05$, a amplitude studentizada de Tukey é $q \approx 3{,}633$, logo $q_\alpha \approx 2{,}569$ e $CD \approx 1{,}21$. Ou seja, diferenças de *rank* médio abaixo de $\approx 1{,}21$ não são distinguíveis nessa coorte — valor que o texto não precisa citar (a coleta ainda está em `Task 12.5`), mas que mostra por que a CD depende tanto de $N$.

Holm–Bonferroni foi incluído como item próprio porque ele opera sobre outra família de hipóteses: a CD de Nemenyi controla o erro ao longo dos *ranks* de Friedman, enquanto as afirmações par a par vêm dos testes do item 2 e precisam da correção *step-down*. Descrever Nemenyi como teste par a par (como o texto anterior fazia) sugeria um valor-p por par que o procedimento não fornece.

A assimetria registrada em §3.2.1 é o que a §5.3.1.1 qualifica: a CPU avalia a lista de candidatos e aplica a melhor melhoria; a GPU residente varre até não haver melhoria; as híbridas executam uma passada por chamada de *kernel*. Por isso o texto passou a dizer que `max_iterations` não representa o mesmo trabalho entre variantes, em vez de sugerir um orçamento idêntico.

Nenhum artefato de conhecimento novo foi criado: a derivação da CD já está em [analysis-demsar-critical-difference-diagrams.md](../../documents_tcc/plan/knowledge-base/analysis-demsar-critical-difference-diagrams.md) e as listas de candidatos em [knowledge_candidate_lists_and_inverse_positions.md](../../documents_tcc/plan/knowledge-base/knowledge_candidate_lists_and_inverse_positions.md).

---

## Additional Findings

> [!NOTE]
> Resolvido na revisão: o diagrama `mermaid` de §3.1.3 (`fig:class-diagram`) foi reescrito como composição de `GeneticAlgorithm` com seis protocolos de estratégia, e a legenda passou a declarar que a variante é uma configuração de `configs/algorithms.json`, não uma subclasse. O DoD de `Task 13.4` fecha em `0` ocorrências dos nomes de classe antigos.

> [!WARNING]
> A frase sobre os kernels de GPU ("Totalmente em GPU varre até não haver melhoria; híbridas executam uma passada por chamada de kernel") não correspondia ao código: `_improve_naive` e `_improve_grid` iteram `max_iterations` no hospedeiro e apenas a variante ingênua varre a vizinhança completa. §3.2.1 e §5.3.1.1 foram corrigidos com os fatos verificados. O `candidate_k = 20` declarado para `HybridNaive` é ignorado pelo kernel por rota — registrado como `BUG-57` / `Task 12.41`.

---

## Review Notes

- Confirmar a decisão sobre o diagrama de classes: renomear as caixas para refletir a classe única + configurações (e então a checagem do DoD fecha em 0) ou aceitar as 4 ocorrências remanescentes na figura.
- A tabela manteve as três faixas e o formato original; a contagem de instâncias é 5 + 4 + 6 = 15.
