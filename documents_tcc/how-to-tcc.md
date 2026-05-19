# TCC composition

The tcc composition is giver by the following h2 headers, until the next h1 header.

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

>[!caution]
> Ainda preciso justificar a escolha de algoritmos genéticos em detrimento de outros algoritmos:
>
> - Grande variedade de pesquisa.
> - Algoritmo propenso à paralelismo
> - Facilidade em demonstrar diferentes implementações.

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

> [!important]Importante:  
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

- `GA_HYPERPARAMETER_TUNING_AND_PERFORMANCE_BENCHMARKS.md`
- `GPU_2OPT_INTEGRATION_ANALYSIS.md`
- `statistical_tests_comprehensive_guide.md`
- `insights.md`
- `missing_decisions.md`

>[!caution]
>Usar apenas as referências acadêmicas contidas nestes arquivos.

### 3.1 Ambiente computacional

- **Hardware**:
  - CPU: Intel Core i7-7700HQ
    >[!caution]
    >add proper specs.
  - RAM: 16 GB DDR4.
  - GPU: NVIDIA GeForce GTX 1050 Mobile (4 GB GDDR5 VRAM).
- **Software**:
  - Sistema Operacional: Linux (debian 13).
  - Linguagem: Python 3.10.16.
  - Bibliotecas Principais:
    - NumPy 2.26 (CPU)
    - cupy-cuda12x 13.6.0 (GPU),
    - CUDA Toolkit
      ```out
      nvcc: NVIDIA (R) Cuda compiler driver
      Built on Fri_Jun_14_16:34:21_PDT_2024
      Cuda compilation tools, release 12.6, V12.6.20
      Build cuda_12.6.r12.6/compiler.34431801_0
      ```
- **Restrições de Hardware**:
  - Limite de VRAM: nas execuções deste trabalho, adotou-se um limite prático de 65% da capacidade total da GTX 1050 (≈2,7 GB de 4 GB), reservado explicitamente para a matriz de distâncias e os buffers $O(n^2)$ do kernel 2-opt. Esse valor segue a análise de `max_memory_calculation.md` e do utilitário `VRAMCalculator`, que consideram overhead do driver, fragmentação e margem de segurança.
    >[!caution]
    >No texto final, confirmar o valor medido com `nvidia-smi` e referenciar explicitamente AS REFERÊNCIAS CONTIDAS EM `max_memory_calculation.md`, explicando por que 65% foi escolhido em vez de 80% ou 100%.

### 3.2 Arquitetura do framework

#### 3.2.1 Organização do código

- Descreve a arquitetura modular final (`benchmarks_v2` e `src/benchmarking_v2`):
  - `code/src/algorithms/{improvement, metaheuristics}`: Implementações dos algoritmos de busca local e metaheurísticas.
  - `code/src/protocols`: Definição das interfaces e do padrão `xp` para abstração de backend (NumPy/CuPy), permitindo a execução agnóstica de hardware.
  - `code/benchmarks_v2/configs`: Arquivos JSON (`algorithms.json`, `benchmark.json`, `problems.json`) que definem a matriz experimental, desacoplando a configuração da lógica.
  - `code/src/benchmarking_v2`: Lógica desacoplada para orquestração (`orchestration.py`), execução (`algorithm_runner.py`) e coleta de dados (`checkpoint_io.py`).

>[!caution]
> deve ser descritivo. Adicionar códigos, a não ser que pseudocódigos em casos pontuai, é expressamente proibido. ccaso seja necessário adicionar códigos, devem estar no apêndice.

#### 3.2.2 Representação do problema e estrutura de dados

- **Carregamento**: Instâncias TSPLIB são carregadas de um banco de dados DuckDB (`datasets/routing.duckdb`) através dos `loaders`.
- **Estrutura**: A matriz de distâncias é pré-calculada em `float64` (precisão dupla) para garantir precisão numérica.
- **Limite de VRAM**: A capacidade máxima teórica de cidades ($n$), considerando apenas a matriz de distâncias, é dada aproximadamente por $n \approx \sqrt{VRAM_{disponivel\_bytes}/8}$. Contudo, ao incluir os buffers $O(n^2)$ da heurística 2-opt (melhorias e índices), `max_memory_calculation.md` mostra que, para cerca de 2,7 GB reservados (65% de 4 GB), o limite prático cai para aproximadamente $n \approx 13\,000$ nós. No texto, deixar claro que as 38 instâncias da TSPLIB usadas (até $n = 1002$) estão bem abaixo desse limite.
  >[!caution]
  >importante ressaltar, em alguma parte do [capítulo 3.1](#31-ambiente-computacional), sobre as limitaçõesda GPU, como os 5 SMs (definisr o que SM - Streaming Multiprocessors significam, um overall geral sobre arquitetura de gpus). [gpu_1](../documentation/technical_decisions/GPU_REDUCTION_PATTERNS_AND_LESSONS.md) [gpu_2](../documentation/technical_decisions/GPU_PERFORMANCE_BOTTLENECK_ANALYSIS.md)  
- **Padrão `ProblemContext`**: Objeto que encapsula a matriz de distâncias e a transfere para a GPU (`cp.asarray`) uma única vez por problema, evitando transferências repetidas da matriz de distância.
  >[!caution]
  >versões nào otimizadas do algoritmos devem transferir a matriz a cada geração (HybridNaive). O total de transferências de memória também será mostrado.
  >
  > é extremamente necessário que você explore os resultados em [results.duckdb](./../results/benchmark_results/results.duckdb/results.duckdb).

#### 3.2.3 Heurística de melhoria 2-opt

- **Versão CPU (NumPy)**: Avaliação vetorizada de todos os movimentos de uma vizinhança para acelerar a busca sequencial, explorando as instruções SIMD da CPU.
- **Versão GPU (CuPy + RawKernel)**:
  - Avaliação de todos os $O(n^2)$ movimentos 2-opt em paralelo.
  - **Redução**: Devido a limitações de integração do CuPy com a biblioteca CUB (CUDA Unbound), foi implementada uma redução manual eficiente em nível de warps e blocos utilizando memória compartilhada (`__shared__`) para encontrar o melhor movimento (Best Improvement), seguindo os padrões recomendados pela documentação da NVIDIA e discutidos em `GPU_REDUCTION_PATTERNS_AND_LESSONS.md` e `GPU_2OPT_INTEGRATION_ANALYSIS.md`.
  - *Diagrama*: Incluir diagrama de sequência ilustrando o paralelismo massivo na avaliação dos movimentos e a etapa de redução.

>[!caution]
>tá ruim demais. Isso foi propriamente documentado, já. Tem que colocar também a referência de onde veio a implementaçào disso (documentação NVIDIA/CUDA, padrão de redução tipo CUB), explicitar quais kernels usam esse padrão (2-opt e redução de custo) e apontar para `GPU_REDUCTION_PATTERNS_AND_LESSONS.md` / `GPU_2OPT_INTEGRATION_ANALYSIS.md`.

#### 3.2.4 Algoritmo Genético base

- **Estrutura**: Representação de permutação de cidades.
  Por que eu escolhi este algoritmo OX? porque fujimoto já havia implementado e mostrado. o objetivo era a implementação e comparação entre CPU, GPU e variações dos mesmo algoritmos, focando em variações nas taxonomias propostas por Crainic & Toulouse. Outras variações podem ser implementadas. Isso pode ser citado no capítulo 6.
- **Operadores**:
  - Seleção: Torneio (Tournament Selection). Explicar certinho como funciona.
  - Cruzamento: Order Crossover (OX), preservando a ordem relativa das cidades. Explicar certinho como funciona.
  - Mutação: Swap Mutation (troca de duas cidades). Explicar certinho como funciona.

#### 3.2.5 Algoritmo Memético

- **Integração**: O algoritmo combina a exploração global do Algoritmo Genético com a exploração local intensiva do 2-opt.
- **Funcionamento**: Após a geração da nova população (cruzamento e mutação), a heurística 2-opt é aplicada a cada indivíduo (ou a uma subpopulação) para refinar a solução, levando-a a um ótimo local antes da próxima geração.
- *Diagrama*: Incluir diagrama mostrando o ciclo do Algoritmo Memético, destacando onde o 2-opt se insere no fluxo do AG.
- **Ciclo e Elitismo**: Ciclo geracional padrão com elitismo, preservando os 2 melhores indivíduos para garantir monotonicidade na convergência.
- **Critérios de Parada**:
  - **Ótimo Atingido**: A execução para se a solução atinge um valor dentro de 1% do ótimo conhecido (`cost <= best_known * 1.01`). Este relaxamento (vs. $10^{-6}$) foi adotado para acelerar a bateria de testes em problemas difíceis sem comprometer a análise de convergência.
    - No texto, vincular essa decisão às discussões em `ISO_ALGORITHMIC_GA_VALIDATION_GUIDE.md` e `gpu_cpu_comparison_methodology.md`, destacando que a análise de qualidade foca em gaps percentuais e não em ótimos exatos.
  - **Estagnação (`patience`)**: Parâmetro adaptativo definido como $patience(n) = 2\sqrt{n}$ gerações consecutivas sem melhora.
    - *Justificativa*: Análises empíricas com o framework V1 (`results_and_stats_v3.ipynb`, `insights.md`) mostraram que um valor fixo (e.g., 50 gerações) inflava artificialmente os tempos em problemas pequenos ("cauda" de 50 gerações sem ganho após atingir o ótimo, reusltando em estatísticas bimodais) e ainda podia ser insuficiente em instâncias grandes. A regra $2\sqrt{n}$ reduz a bimodalidade dos tempos de execução e foi incorporada diretamente em `benchmarking_v2.algorithm_runner.adaptive_patience`.
  - **Geração Máxima**: Limite superior adaptativo: $max\_gens(n) = 2\,n\sqrt{n}$, calculado pela função `adaptive_generations(n)` em `algorithm_runner.py`.
  >[!caution]
  >- Referências devem ser acadêmicas, não os meus rascunhos em markdown. Porém, neles há as referências corretas a serem utilizadas.
  >- A seguinte afirmação ainda precisa ser validada.
  >   > Essa fórmula substitui a aproximação linear $14.3,n - 46$ usada em análises exploratórias de V1 e garante um orçamento de gerações crescente com o tamanho do problema, sem depender de correções a posteriori.
- *Diagrama*: Incluir fluxograma detalhado do AG Base, com notas explicativas para cada etapa (Seleção, Crossover, Mutação, Avaliação).

>[!caution]Muito simples. Este arquivo deve ser um guia.
>
> - Deve-se justificar os porquês dessas escolhas. Deve-se referenciar corretamente baseando-se nas [referências](./refs.bib). Pode-se dizer que estarão sendo melhores explicados a frente, no [subcapítulo 3.4](#34-parâmetros-experimentais)
> - Problemas encontrados e tudo o mais. Linkando com o subcapítulo onde os problemas encontrados são explicados.
> - O funcionamento de um algoritmo genético deve ser explicitada através de diagramas. O algoritmos base pode ter um flowchart com notas explicativas adicionais de cada um dos processos.
> - Umas subsessão para cada etapa do algoritmo deve ser feita.

>[!caution]
>
> - Explicacão clara sobre como o AG e o 2-opt agem dentro do algoritmo memético.
> - Como o 2-opt age na melhoria, para além do algoritmo genético.
> - diagramas do funcionamento base do algoritmo memético devem estar presentes.

#### 3.2.6 Variações isoalgorítmicas do GA

- **Conceito**: O termo "isoalgorítmico" é usado para descrever quatro variações que compartilham a mesma estrutura lógica de alto nível (definida por `GeneticAlgorithmBase`), diferindo apenas na plataforma de execução (CPU/GPU) de seus componentes (AG e 2-opt). O termo "isoalgorítmico será substituído, usando nomes como algoritmos funcionalmente equivalente, como expressado no capítulo 1.
- **Variantes**:
  1. **CPU**: AG e 2-opt executados puramente em CPU com NumPy.
  2. **HybridNaive**: AG em CPU, 2-opt em GPU. Lança um kernel CUDA separado para cada indivíduo da população. Serve para demonstrar o alto overhead de lançamento de kernels e transferência de dados.
  3. **HybridOptimized**: AG em CPU, 2-opt em GPU. Utiliza processamento em lote (Batch Processing), lançando um único kernel para processar toda a população simultaneamente, maximizando a ocupação da GPU.
  4. **FullGPU**: AG e 2-opt totalmente residentes na GPU. Minimiza drasticamente as transferências de dados entre Host e Device, mantendo a população na VRAM durante todo o processo.
- *Diagramas*: Incluir fluxogramas simples no texto principal para cada variante. Diagramas técnicos detalhados devem constar no Apêndice A.
- As variantes devem ser propriamente descritas. [Diagramas](../documentation/diagrams) simples como flowcharts devem estar no arquivo principal. Diagramas mais complexos devem estar no Apêndice A e propriamente linkados utilizando o formato pandoc.

### 3.3 Conjunto de instâncias de teste

- **Seleção**: 38 instâncias da TSPLIB, agrupadas por tamanho (Pequeno, Médio, Grande) conforme `CHAPTER4_BENCHMARK_GUIDE.md`.
- **Justificativa**: A seleção abrange uma ampla gama de tamanhos (de 51 a 4461 cidades) e inclui instâncias clássicas da literatura para garantir a comparabilidade com outros trabalhos.
- **Tabela**: Apresentar tabela com nome da instância, dimensão ($n$) e valor da melhor solução conhecida (BSF).
- Informações adicionais sobre a escolha devem ser justificadas e referenciadas.

### 3.4 Parâmetros experimentais

A escolha dos parâmetros foi guiada por recomendações clássicas (e.g., @goldberg1989genetic) e ajustada via análises empíricas (`GA_HYPERPARAMETER_TUNING...`) para o contexto específico de GPU.

- **Parâmetros Estáticos do GA**:
  - `population_size = 256`: Escolhido para garantir diversidade genética suficiente e alinhar com a arquitetura da GPU (múltiplo de 32/warp).
  - `mutation_rate = 0.02`: Taxa conservadora para manter a diversidade sem destruir esquemas promissores.
  - `tournament_size = 5`: Pressão seletiva moderada.
  >[!caution]
  >As escolhas devem estar propriamente referenciadas, acredito ser através de Goldberg nas referências.
- **Parâmetros da Busca Local**:
  - `two_opt_iterations = 10`: Número de iterações de 2-opt por indivíduo configurado em `algorithms.json`.
    - *Justificativa*: Estudos em `GA_HYPERPARAMETER_TUNING_AND_PERFORMANCE_BENCHMARKS.md` sugerem que valores mais altos (50–100) tendem a produzir soluções um pouco melhores, porém com custo computacional significativamente maior por geração. Dado que cada geração já incorpora uma busca local relativamente cara, o trabalho adotou um compromisso conservador (10 iterações) para manter o tempo de execução viável em 38 instâncias × 4 variantes, registrando essa escolha como ameaça à validade em §3.7.
    > [!caution]
    > Esse arquivo não é uma referência acadêmica e não pode ser utilizada como referência. As referências acadêmicas contidas nele, no entanto, devem ser utilizadas quando necessárias.
  > [!caution]
  > Explicar o que isso significa para o algoritmo e para os resultados. Justificar na literatura (e.g., @fujimoto2011highly; @lima2018hybrid; @vidal2013hybrid) por que não são utilizadas mais iterações e relacionar explicitamente com as limitações de tempo/VRAM do experimento, baseando-se nas observaǫes de `GA_HYPERPARAMETER_TUNING...`.
- **Parâmetros de Execução**:
  - `seed_base = 42`: Valor registrado em `algorithms.json` apenas como convenção; na implementação atual de `benchmarking_v2.algorithm_runner.run_single_algorithm`, cada repetição recebe uma semente pseudoaleatória gerada por `np.random.randint`, e a lista de sementes efetivamente usadas é armazenada no campo `raw_seeds` de cada checkpoint.
    - No texto do TCC, explicar que a reprodutibilidade é obtida reusando esses `raw_seeds` (e não via um único `seed` global fixo) e discutir essa diferença como parte da evolução metodológica em §3.7.
    >[!caution]
    >- Confirmar no código se o campo `ga_params.seed` é de fato utilizado; atualmente `run_single_algorithm` ignora esse valor. Deixar isso claro em §3.7 como limitação de reprodutibilidade "one-click".
    >- Se as raw_seeds são transferidas também para a GPU
  - `repetitions = to_adjust`: Número de execuções independentes por par (algoritmo, instância) configurado em `benchmark.json` para o framework V2.
    - *Justificativa*: 30 repetições seriam ideais do ponto de vista estatístico (conforme @demvsar2006statistical), mas, devido às limitações de tempo e hardware, optou-se por 15 repetições por problema/algoritmo na fase principal. O framework suporta modo incremental (`incremental_runs=true`), permitindo estender um conjunto de checkpoints até 30 repetições caso seja necessário refinamento adicional.
    >[!caution]
    >Use uma variável de fácil localização como $n_{reps}$. Eu continuarei rodando novas atualizações, já que a análise estatística é rápida de rodas.

### 3.5 Metodologia estatística

- **Pipeline por Instância**:
  1. **Normalidade**: Teste de Shapiro-Wilk para verificar a distribuição dos dados (custos e tempos).
  2. **Comparação Pareada**:
      - Se normal: Teste t de Student pareado.
      - Se não-normal: Teste de Wilcoxon Signed-Rank (não-paramétrico).
  3. **Tamanho de Efeito**: Cálculo do Cohen's *d* para quantificar a magnitude da diferença, não apenas sua significância.
  4. **Correção**: Aplicação de Holm-Bonferroni para controlar a taxa de erro familiar (FWER) em múltiplas comparações.
- **Pipeline Global**:
  - Teste de Friedman para detectar diferenças entre algoritmos através de múltiplos datasets.
  - Teste Nemenyi post-hoc para identificar quais pares de algoritmos diferem significativamente no ranking global.
- **Significância**: $\alpha = 0.05$ adotado como padrão.
- *Ameaças*: Discussão sobre violações de premissas estatísticas e como foram tratadas (e.g., uso de testes robustos/não-paramétricos).

>[!caution]
>
>- linkar com [subcapitulo 3.7](#37-ameaças-à-validade-e-evolução-metodológica) onde ameaças à validade são discutidas.
>- adicionar diagrama de decisão para testes a partir de [statistical_tests_comprehensive_guide.md](../documentation/technical_decisions/statistical_tests_comprehensive_guide.md#L6420)

### 3.6 Procedimento experimental

Descreve o fluxo de trabalho do framework `benchmarks_v2`, orquestrado pelo script `code/benchmarks_v2/run_chapter4_benchmark.py`:

1. **Configuração**: Carrega as configurações dos arquivos `benchmark.json`, `algorithms.json` e `problems.json`, definindo a matriz de experimentos.
2. **Execução**: Para cada problema e cada algoritmo, o `AlgorithmRunner` executa `repetitions` execuções independentes, conforme definido em `benchmark.json` ($n_{reps}$ repetições nas execuções principais desta monografia). A variante CPU é executada apenas para instâncias com $n < 100$ por questões de viabilidade temporal.

>[!warning]
> Will run validation to see if cpu can be extrapolated. For now, You may write other parts that do not specifically requires those validations, i.e. fixed parameter.

3. **Coleta de Dados**: O `CheckpointIO` salva os resultados de cada execução em arquivos JSON. Os dados incluem tempo de execução, custo final, gap para o ótimo, **número de gerações completadas e o motivo da parada** (e.g., `hit_optimal`, `no_improvements`).
4. **Análise e Geração de Relatórios**: Scripts auxiliares (`aggregate_statistics.py`, `report_generator.py`) processam os checkpoints para realizar a análise estatística e gerar as tabelas e gráficos para o Capítulo 4.

> [!caution]
> Existem dois tipos de benchmarks. O mais completo e metodologicamente consistente está na [`v2`](../code/benchmarks_v2) e é a base de **todos os resultados de tempo e speedup** do Capítulo 4. Os primeiros resultados foram analisados na [`v1`](../code/legacy/benchmarks) e são usados apenas como material histórico/de auditoria, com uma exceção controlada: nos problemas pequenos em que o GA de V1 atingiu exatamente o ótimo conhecido, essas execuções são reaproveitadas **apenas** para estatísticas de qualidade de solução (probabilidade de atingir o ótimo, distribuição de gaps) e nunca para métricas de tempo ou de gerações. Em todas as tabelas em que esse reaproveitamento aparece, ele é indicado explicitamente e os dados de V2 permanecem como referência principal.

### 3.7 Ameaças à validade e evolução metodológica

- **Evolução do Framework (V1 → V2)**: O projeto iniciou com um framework monolítico (`benchmarks/chapter4_validation.py`). Resultados preliminares (V1) revelaram falhas críticas na captura de metadados e rigidez nos parâmetros experimentais: uso de `patience = 50` fixo (independente do tamanho do problema), ausência de `raw_generations` confiáveis (especialmente nas execuções FullGPU) e motivos de parada genéricos (`"completed"`) que não diferenciavam entre convergência ao ótimo e estagnação. Uma auditoria metodológica sistemática conduzida em `results_and_stats_v3.1.ipynb` sobre um subconjunto de runs V1 (GA+2-opt, 12 instâncias com n ∈ [51,100]) quantificou o impacto dessas limitações: a patience fixa introduzia um viés médio de aproximadamente **7% nos tempos de CPU** (média legacy ≈196.8 s vs. média corrigida ≈183.2 s) e, em casos individuais de instâncias pequenas/médias onde a patience representava fração significativa do orçamento de gerações, o viés podia ser substancialmente maior. O framework atual (V2, implementado em `benchmarks_v2/` e `src/benchmarking_v2/`) foi desenvolvido para eliminar essas fontes de viés, introduzindo: (1) **modularidade** via configuração JSON desacoplada (`algorithms.json`, `benchmark.json`, `problems.json`); (2) **parâmetros adaptativos** (`adaptive_generations(n) = 2n√n`, `adaptive_patience(n) = 2√n`) calibrados para escalar com a dificuldade do problema; e (3) **metadados completos** nos checkpoints (`raw_generations`, `raw_stop_reasons`, `raw_seeds`, métricas de transferência host–device e contadores de `kernel_launches`). **Todos os resultados quantitativos de tempo e speedup do Capítulo 4 baseiam-se exclusivamente em dados V2**. Para estatísticas de **qualidade de solução** em problemas pequenos (por exemplo, probabilidade de atingir o ótimo conhecido), um subconjunto de execuções de V1 em que o GA atingiu exatamente o ótimo é reaproveitado como amostras "somente qualidade", sem reutilizar tempos ou gerações. Fora desse caso específico, os dados V1 servem apenas como estudo de caso metodológico (Apêndice B) para ilustrar as armadilhas de protocolos de medição inadequados e a importância de critérios de parada adaptativos.

- **Parâmetros Empíricos (patience, gerações, 2-opt)**: O ajuste de `patience(n) = 2√n` e `max_gens(n) = 2n√n` foi motivado por análises empíricas do comportamento do algoritmo em V1 (documentadas em `GA_HYPERPARAMETER_TUNING_AND_PERFORMANCE_BENCHMARKS.md` e `insights.md`), que mostraram que valores fixos criavam distribuições bimodais de tempo (runs que atingiam o ótimo rapidamente vs. runs que esgotavam a patience após estagnação). A fórmula adaptativa reduz essa bimodalidade e melhora a robustez média, embora possa não ser ótima para outliers (instâncias particularmente fáceis ou difíceis). O número de iterações de 2-opt (`two_opt_iterations = 10`, configurado em `algorithms.json`) representa um compromisso entre qualidade de solução e viabilidade computacional: valores mais altos (50–100) tendem a produzir soluções ligeiramente melhores, mas com custo computacional significativamente maior por geração. **Todos esses hiperparâmetros são mantidos idênticos para as quatro variantes** (CPU, HybridNaive, HybridOptimized, FullGPU), preservando a justiça da comparação isoalgorítmica. Configurações mais agressivas (populações maiores, mais iterações de 2-opt, tuning específico por instância) ficam como trabalho futuro e são discutidas explicitamente como limitações em §6.3.

- **Dureza da Instância e Variabilidade**: Observou-se, a partir de consultas ao banco de dados V2 (`results_v2/checkpoints/`) e análises exploratórias em `insights.md`, que o tamanho $n$ não é o único preditor de tempo de execução ou qualidade de solução; a **estrutura topológica da instância** (distribuição espacial dos nós, presença de clusters, simetrias) influencia fortemente a velocidade de convergência do GA+2-opt. Instâncias com dimensão similar (por exemplo, diferentes problemas da família `kroA100`, `kroB100`) exibem tempos médios e gaps substancialmente distintos. Essa variabilidade intrínseca reduz o poder preditivo de modelos de regressão simples baseados apenas em $n$, resultando em MAPEs não-desprezíveis mesmo após correções de viés. A estratégia adotada para mitigar (mas não eliminar) essa limitação foi: (1) uso de um **conjunto diversificado de 38 instâncias TSPLIB** cobrindo múltiplas famílias e faixas de tamanho; (2) **múltiplas repetições independentes** ($n_{reps}$ runs por par instância–algoritmo) com sementes pseudoaleatórias registradas (`raw_seeds`) para quantificar a variabilidade estocástica do GA; e (3) **análise estratificada** por faixa de tamanho (pequeno/médio/grande) nas estatísticas agregadas, em vez de pooling global ingênuo. No texto do Capítulo 4, essas estratégias devem ser explicitadas ao interpretar intervalos de confiança e significância estatística.

- **Limitações de Reprodutibilidade e Extensibilidade**: Embora o framework V2 registre `raw_seeds` em cada checkpoint, a reprodutibilidade "one-click" completa (reexecutar todo o benchmark e obter exatamente os mesmos resultados numéricos) requer: (1) reusar as sementes armazenadas; (2) fixar versões exatas de bibliotecas (`numpy`, `cupy`, `cuda-toolkit`); e (3) usar hardware idêntico (especialmente GPU). Variações em drivers CUDA, versões de CuPy ou diferenças de microarquitetura podem introduzir flutuações numéricas de ponto flutuante que, embora geralmente pequenas, podem acumular-se ao longo de muitas gerações. Essa limitação é comum a qualquer trabalho de GPU e não invalida as conclusões qualitativas, mas deve ser reconhecida ao reportar resultados. Adicionalmente, o framework atual suporta apenas **TSP euclidiano simétrico** e **um único tipo de metaheurística** (GA+2-opt); extensões para ATSP, VRP, ou outras metaheurísticas (SA, ACO, ILS) requerem modificações nos módulos `algorithms/` e possivelmente nos kernels CUDA, ficando explicitamente como trabalho futuro (§6.3).

>[!caution]
>
>- A auditoria detalhada do viés de timing V1 (incluindo a derivação da correção proporcional baseada em gerações efetivas, validação de premissas de linearidade tempo–geração, diagnósticos de regressão, e comparação qualitativa com padrões V2) deve ser movida para o **Apêndice B: Auditoria Metodológica e Evolução de Protocolos de Medição**, para não sobrecarregar o corpo do Capítulo 3. No texto principal, limitar-se a uma menção de 2–3 parágrafos resumindo o problema (viés de ~7% em média, bimodalidade, metadados incompletos) e a solução (V2 com parâmetros adaptativos), remetendo ao apêndice para detalhes técnicos.
>- Ainda não deve-se escrever este capítulo ou o apêndice; aguardar conclusão da análise estatística V2 (todo 8) e geração dos resultados principais (Capítulo 4) para garantir consistência terminológica e numérica.

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

# Procedures

## Links that may help

- [pandoc-markdown-manual](../documentation/pdfs/pandoc-markdown-manual.pdf)
- [pandoc-citation](../documentation/pdfs/pandoc-citation.pdf)
- [pandoc-full](../documentation/pdfs/pandoc-manual.pdf)
- [pandoc-images](https://pandoc.org/MANUAL.html#images)
- [website-manual-markdown](https://pandoc.org/MANUAL.html#pandocs-markdown) -> fonte da verdade para como escrever o arquivo em pandoc.
- [checklist do que o documento deve conter](../documentation/critical_analysis/critical_analysis_checklist.md)
- [mermaid in pandoc](https://github.com/raghur/mermaid-filter) -> add this package to uv.

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
