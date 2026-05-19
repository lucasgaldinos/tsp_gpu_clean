Completed: *Design full TCC chapter skeleton* (5/7)

Here’s a concrete, UFSC-compliant skeleton for your thesis, aligned with the *real* scope (TSP + GA+2-opt, 4 iso-algorithmic variants, chapter4_validation benchmark).

I’ll also point out, for each chapter, what should be *moved / reused / rewritten* from first_draft.md and the technical_decisions docs.

---

## 1. Front matter (pré-textuais)

You already have most of this; main change is updating the **Resumo/Abstract** to the GA+2-opt/TSP scope.

- Capa, folha de rosto, ficha catalográfica, folha de aprovação, dedicatória, agradecimentos (as required by UFSC).
- **Resumo** (PT-BR, máx. 250–300 palavras)
  - Foco: TSP, GA+2-opt, comparação CPU vs 3 variantes GPU (HybridNaive, HybridOptimized, FullGPU), 38 instâncias TSPLIB, metodologia estatística (30 repetições, Friedman + Nemenyi, etc.).
  - Remove VRP and SA do escopo principal (podem aparecer só em 1 frase de contexto ou trabalhos futuros).
- **Abstract** (EN, espelho do resumo).
- **Lista de Figuras, Tabelas, Abreviaturas, Símbolos**  
  - Reaproveitar listas do first_draft.md, mas:
    - Podar símbolos/abreviaturas só de VRP e algoritmos não implementados (Tabu, ACO, etc.) ou movê-los para *Apêndice* se forem citados apenas em trabalhos futuros.
- **Sumário**  
  - Atualizar ao final, quando a estrutura abaixo estiver fixa.

---

## 2. Capítulo 1 – Introdução

**Objetivo**: responder às perguntas de problema/justificativa/objetivos do `roteiro_analise_critica.md`.

### 1.1 Contexto e motivação

- TSP como benchmark clássico de otimização combinatória.
- Crescimento do tamanho das instâncias e custo de metaheurísticas.
- GPUs como plataforma massivamente paralela para acelerar heurísticas de roteamento.
- Reaproveitar a abertura do Cap. 1 do first_draft.md, mas:
  - Cortar ênfase em VRP como escopo principal.
  - Manter VRP apenas como motivação e *trabalhos futuros*.

### 1.2 Problema de pesquisa

- Formulação clara, em português simples, algo como:
  - “Como comparar, de forma isoalgorítmica e estatisticamente rigorosa, o impacto de diferentes estratégias de paralelização em GPU para um Algoritmo Genético híbrido com 2-opt aplicado ao TSP?”
- Aqui você já “vende” que o código força algoritmos isoalgorítmicos: mesma lógica, só mudam os pontos de execução (CPU vs GPU).

### 1.3 Objetivo geral

- Ex.: “Investigar e quantificar o ganho de desempenho e o impacto na qualidade das soluções ao acelerar um Algoritmo Genético híbrido com 2-opt para TSP em GPU, usando variantes isoalgorítmicas e metodologia estatística padronizada.”

### 1.4 Objetivos específicos

Sugeridos (ajuste a gosto):

1. Implementar quatro **variações isoalgorítmicas** de GA+2-opt (CPU, HybridNaive, HybridOptimized, FullGPU).
2. Definir um conjunto representativo de instâncias TSPLIB (pequenas, médias, grandes).
3. Elaborar um protocolo experimental com $\approx 30$ repetições por instância/algoritmo.
4. Aplicar testes estatísticos (Shapiro–Wilk, t pareado/Wilcoxon, Friedman + Nemenyi, Holm–Bonferroni, Cohen’s *d*) para comparar algoritmos.
5. Discutir limites práticos (VRAM, overhead de kernel, thresholds de tamanho) e implicações para projetos reais.

### 1.5 Justificativa

- Conectar com:
  - Lacuna de estudos **reprodutíveis** comparando variantes isoalgorítmicas em GPU.
  - Falta de benchmarks que isolam ganho de paralelização vs ganho de “engenharia de algoritmo”.
  - Relevância prática (rotações em logística, mas mantendo escopo de implementação em TSP).

### 1.6 Organização do trabalho

- Breve parágrafo descrevendo o conteúdo dos Capítulos 2–6.

> **De onde vem o texto?**  
>
> - Cap. 1 do first_draft.md é a base, mas você simplifica o escopo para TSP+GA+2-opt e remove promessas de VRP/SA como resultados finais.

---

## 3. Capítulo 2 – Fundamentação Teórica / Revisão Bibliográfica

**Objetivo**: cumprir a *revisão bibliográfica* do roteiro, sem misturar demais com metodologia do seu framework.

### 2.1 Otimização combinatória e o Problema do Caixeiro Viajante

- Definição formal do TSP.
- NP-dificuldade.
- TSPLIB como conjunto padrão de instâncias.
- Referências: Cook, Reinelt etc.

### 2.2 Heurísticas para o TSP

- **Heurísticas de construção**: vizinho mais próximo, inserção, etc. (curto, pois não são o foco).
- **Heurísticas de melhoria**:
  - k-opt, com destaque para 2-opt (Croes), 3-opt, Lin–Kernighan.
  - Papel de 2-opt em metaheurísticas modernas e em memetic algorithms (GLS+FLS, etc.).

### 2.3 Algoritmos genéticos e heurísticas híbridas

- Visão geral de GA:
  - Representação, população, fitness, seleção, cruzamento, mutação.
- GA para TSP (Larrañaga, Goldberg, etc.).
- **Algoritmos meméticos / GA+LS**:
  - GA+2-opt, GA+3-opt, GA+Lin–Kernighan.
  - Citar Fujimoto & Tsutsui (GPU GA para TSP) e outros trabalhos relevantes.

### 2.4 Computação em GPU e paralelização de metaheurísticas

- Conceitos básicos de CUDA / GPU:
  - Hierarquia grid–block–thread, SIMT, memória global/compartilhada.
- Taxonomias:
  - Flynn (SISD/SIMD/MIMD).
  - Crainic & Toulouse (P-Data, S-Task, P-Task) para metaheurísticas paralelas.
- Exemplos de TSP em GPU:
  - GA, SA, ACO, 2-opt paralela, GLS, etc.

> **Importante**:  
>
> - Aqui você fala **dos trabalhos dos outros**, não do seu código.  
> - Parte do que hoje está em 2.4 do first_draft.md pode ficar aqui (definição de CUDA, taxonomias), mas:
>   - Os detalhes do *seu* kernel 2-opt e do *seu* framework vão para o Cap. 3.

### 2.5 Comparação estatística de algoritmos de otimização

- Por que precisamos de:
  - Múltiplas instâncias.
  - Múltiplas repetições.
- Visão geral (sem fórmulas pesadas):
  - Demšar (2006): Friedman + Nemenyi para múltiplos algoritmos.
  - Testes pareados (t, Wilcoxon) para comparações por instância.
  - Conceito de tamanho de efeito (Cohen’s *d*).

> **De onde vem o texto?**  
>
> - Adaptar e condensar trechos da `statistical_tests_comprehensive_guide.md` (parte conceitual).  
> - Detalhes matemáticos (PDFs, funções gama) podem ir para apêndice.

---

## 4. Capítulo 3 – Materiais e Métodos

Este é o coração metodológico, e deve usar:

- CHAPTER4_BENCHMARK_GUIDE.md
- `SA_COMPREHENSIVE_ANALYSIS.md` (como referência de parâmetros, mas agora centrado em GA)
- `GPU_2OPT_INTEGRATION_ANALYSIS.md`
- `statistical_tests_comprehensive_guide.md`

### 3.1 Ambiente computacional

- Hardware (CPU, RAM, GPU GTX 1050 Mobile, 4 GB VRAM, limite de 65%).
- Software: Linux, Python, NumPy, CuPy, versões, etc.

### 3.2 Arquitetura do framework

#### 3.2.1 Organização do código

- Breve descrição de:
  - `code/src/algorithms/{construction, improvement, metaheuristics}`
  - `protocols` (interfaces de estratégia, padrão `xp` para NumPy/CuPy).
  - `benchmarking` (scripts do capítulo 4, especialmente `chapter4_validation.py`).

#### 3.2.2 Representação do problema e estrutura de dados

- Como as instâncias TSPLIB são carregadas (routing.duckdb, loaders).
- Matriz de distâncias em `float64`, cálculo de distância euclidiana.
- **VRAM**:
  - Fórmula de capacidade máxima derivada do first_draft.md (65% de 4 GB, matriz $n \times n$ em `float64`).
  - Resultado $\approx 18{.}000$ cidades como limite teórico.
- Padrão `ProblemContext` e arrays `xp` para CPU/GPU.

#### 3.2.3 Heurística de melhoria 2-opt

- Versão CPU (NumPy) – ideia geral de vetorizar avaliação de vizinhos.
- Versão GPU (CuPy + RawKernel):
  - Conceito de avaliar todos os movimentos $O(n^2)$ em paralelo.
  - Redução para encontrar o melhor movimento (CUB / warp-level).
- Citar Fujimoto & Tsutsui como base conceitual; deixar detalhes de kernel (índices de thread, warps) para apêndice técnico.

#### 3.2.4 Algoritmo Genético base

- Representação do indivíduo (permutação de cidades).
- Inicialização da população (aleatória ou heurística).
- Operadores:
  - Seleção (torneio).
  - Cruzamento (por ex. OX).
  - Mutação (swap).
- Ciclo geracional, elitismo (se houver), e **early stopping**:
  - Parâmetro `patience = 50`.
  - Geração máxima adaptativa: $max\_gens = 14{.}3\,n - 46$ (com explicação curta).

#### 3.2.5 Variações isoalgorítmicas do GA

Aqui entra a terminologia:

- Sugestão em PT-BR:
  - “**Variações isoalgorítmicas** de um mesmo Algoritmo Genético”  
    (explicitar que “isoalgorítmico” = mesma estrutura algorítmica, mudando apenas onde cada parte roda).

- Descrever as quatro variantes:

1. **CPU**  
   - Tudo em NumPy; 2-opt sequencial na CPU.
2. **HybridNaive**  
   - GA na CPU, 2-opt no GPU, mas com **lançamento de kernel por indivíduo** (overhead alto).
3. **HybridOptimized**  
   - GA na CPU, 2-opt no GPU em **lotes (batch)** – reduz overhead de kernel.
4. **FullGPU**  
   - GA + 2-opt totalmente em GPU (Fujimoto-like), com early stopping.

- Classificar cada uma na taxonomia de Crainic & Toulouse (P-Data / S-Task etc.) de forma breve.

### 3.3 Conjunto de instâncias de teste

- 38 instâncias TSPLIB, agrupadas por tamanho (Small / Medium / Large) conforme CHAPTER4_BENCHMARK_GUIDE.md.
- Tabela com:
  - Nome, número de cidades, valor ótimo.
- Justificar seleção (diversidade de tamanhos, uso de casos clássicos como eil51, pr1002).

### 3.4 Parâmetros experimentais

- **Parâmetros do GA** (conforme `GA_PARAMS` do guia do Cap. 4):
  - `population_size = 256`
  - `mutation_rate = 0.02`
  - `tournament_size = 5`
  - `two_opt_iterations = 10` (discutir aqui se você manter ou se já corrigiu para maior, conforme `GPU_2OPT_INTEGRATION_ANALYSIS.md`).
  - `seed = 42` (reprodutibilidade).
- Número de repetições:
  - `repetitions = 30` (raciocínio estatístico do first_draft.md 3.5 + stats guide).
- Discussão curta sobre escolhas baseadas em:
  - Experimentos piloto.
  - Literatura de GA para TSP.

### 3.5 Metodologia estatística

- Pipeline **por instância**:
  1. Shapiro–Wilk para cada algoritmo (normalidade).
  2. Escolha de teste pareado:
     - t-test se normalidade razoável.
     - Wilcoxon se não.
  3. Cálculo de:
     - diferença média de custo.
     - intervalo de confiança de 95%.
     - tamanho de efeito (Cohen’s *d*).
  4. Correção de múltiplas comparações via Holm–Bonferroni.
- Pipeline **global** (várias instâncias):
  - Friedman para ranking de algoritmos.
  - Nemenyi para comparações pós-hoc.
- Refletir níveis de significância ($\alpha = 0{.}05$) e interpretação.

### 3.6 Procedimento experimental

- Descrever **passo a passo** o que `chapter4_validation.py` faz:
  1. Seleciona problemas (38 instâncias).
  2. Para cada problema e algoritmo:
     - executa `repetitions` vezes, respeitando `cpu_size_threshold` (somente CPU quando $n<100$).
  3. Coleta tempos, custos, gaps.
  4. Roda análises estatísticas e gera:
     - `results/chapter4_validation.log`
     - `results/tables/chapter4_validation.{md,tex}`.
- Explicar por que CPU é omitido em instâncias grandes (viabilidade prática, aceitável academicamente).

### 3.7 Ameaças à validade e correções históricas

- Inspirar-se em `GPU_2OPT_INTEGRATION_ANALYSIS.md`:
  - Erro de não usar 2-opt GPU de fato nas primeiras versões.
  - `two_opt_iterations` muito baixo em instâncias grandes.
  - VRAM contention em experimentos multi-processo.
- Explicar que:
  - Resultados antigos foram descartados/repetidos.
  - O protocolo atual corrige esses problemas.

---

## 5. Capítulo 4 – Resultados

Este capítulo “consome” diretamente o que `chapter4_validation.py` produz, conforme CHAPTER4_BENCHMARK_GUIDE.md.

### 4.1 Validação do protocolo

- Relatar um “test run” com poucas repetições (2) apenas para mostrar que:
  - Pipeline estatístico funciona.
  - Tabelas são geradas corretamente.
- Depois, focar nos resultados com 30 repetições.

### 4.2 Estatísticas por instância

- Basear-se na **Tabela 1** (summary por problema):
  - Gap médio, desvio padrão, tempo médio, speedup.
- Apresentar:
  - Alguns casos ilustrativos (por ex. uma instância pequena, uma média, uma grande).
  - Mostrar gráficos simples: boxplots de custo, tempos, speedup por instância ou por faixa de tamanho.

### 4.3 Comparação agregada entre algoritmos

- Basear-se na **Tabela 2** (estatísticas agregadas) e **Tabela 3** (melhor algoritmo por faixa).
- Discutir:
  - Qual algoritmo domina em gap e em tempo.
  - Situações em que HybridOptimized vs FullGPU se destacam.
  - Comparação com CPU onde é aplicável.

### 4.4 Escalabilidade e speedup

- Gráficos:
  - Speedup vs número de cidades.
  - Tempo absoluto vs tamanho, separando por algoritmo.
- Relacionar:
  - Overhead de kernel → porque HybridNaive sofre.
  - Benefícios de batch 2-opt e FullGPU à medida que n cresce.

### 4.5 Qualidade das soluções e significância estatística

- Mostrar exemplos de:
  - Comparações pareadas com significância (p-valor baixo, *d* grande).
  - Casos em que diferenças são pequenas / não significativas.
- Discutir:
  - Quando GPU traz apenas ganho de tempo, sem perda de qualidade.
  - Se há trade-offs entre gap e tempo.

---

## 6. Capítulo 5 – Discussão

- Conectar os resultados com:
  - Questão de pesquisa (Cap. 1).
  - Literatura (Cap. 2).
- Tópicos sugeridos:
  - Isoalgorítmico como ferramenta para isolar impacto da plataforma.
  - Interpretação dos ganhos de speedup vs custo de implementação.
  - Comparação qualitativa com trabalhos de Fujimoto & Tsutsui e outros (sem prometer melhor/igual, mas contextualizando).
  - Limitações:
    - Só TSP, nada de VRP.
    - Uma GPU específica (GTX 1050).
    - Um conjunto limitado de parâmetros de GA.

---

## 7. Capítulo 6 – Conclusões e Trabalhos Futuros

### 6.1 Síntese das contribuições

- Reforçar:
  - Implementação das quatro variantes isoalgorítmicas.
  - Protocolo estatístico completo e reprodutível.
  - Resultados de speedup e qualidade.

### 6.2 Respostas às questões de pesquisa e objetivos

- Ponto a ponto:
  - Como cada objetivo específico foi cumprido.
  - Quais hipóteses foram apoiadas/contrariadas.

### 6.3 Trabalhos futuros

- Extensões óbvias:
  - VRP (usando mesma arquitetura).
  - Outras metaheurísticas (SA, Tabu, GLS, ILS).
  - GPUs mais modernas e multi-GPU.
  - Persistent kernels, mais otimizações de kernel.

---

## 8. Pós-textuais

- **Referências** (a partir de refs.bib).
- **Glossário** (termos como “isoalgorítmico”, “2-opt”, “warp”, etc.).
- **Apêndices**:
  - Pseudocódigos dos algoritmos (GA+2-opt, variantes).
  - Detalhes matemáticos dos testes estatísticos (PDFs, derivação de t e $\chi^2$ se você quiser).
  - Trechos de configuração de benchmark (`chapter4_validation.py`).
- **Anexos**:
  - Tabelas completas de resultados (se forem muito grandes para o corpo principal).

---

## Terminologia para “iso-algoritmos”

Sugestão consistente para o texto:

- Use “**variações isoalgorítmicas** de Algoritmos Genéticos” ou “**algoritmos isoalgorítmicos**” e defina explicitamente uma vez:
  - “Chamamos de algoritmos **isoalgorítmicos** aqueles que mantêm a mesma estrutura algorítmica (mesmos operadores, parâmetros e fluxo de controle), diferindo apenas na forma de implementação e na plataforma de execução (CPU vs GPU).”
- Depois, referir-se às quatro variantes como:
  - “GA-CPU”, “GA-HybridNaive”, “GA-HybridOptimized”, “GA-FullGPU”.

---

## Checklist atual

Com base no que já foi feito e neste esqueleto:

1. Review core thesis and analysis docs – **concluído**  
2. Inspect core algorithm and benchmark code – não iniciado  
3. Survey benchmark results and data structure – não iniciado  
4. Analyze bibliography and select key references – **concluído (base inicial)**  
5. Design full TCC chapter skeleton – **concluído (neste texto)**  
6. Plan diagrams and methodological details – ainda por fazer  
7. Produce consolidated analysis and task list document – ainda por fazer  

Se você quiser, o próximo passo é escolher um capítulo (normalmente Cap. 1 ou 3) e começar a reescrita guiada com base neste esqueleto.
