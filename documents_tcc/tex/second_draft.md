---
title: "GPU-Accelerated Isoalgorítmico GA+$2$-opt para TSP"
author: "Lucas Galdino"
date: 2025-11-22
lang: pt-BR
bibliography:
   - ./refs.bib
reference-section-title: "Referências"
link-citations: true
header-includes:
    - \newcommand{\standalonefiglabel}[1]{\refstepcounter{figure}\label{#1}}
    - \setcounter{tocdepth}{4}
---

```{=latex}
\clearpage
```

# Resumo {#resumo .unnumbered}

Este trabalho investiga o impacto de diferentes estratégias de paralelização em Unidades de Processamento Gráfico (GPUs) no desempenho de um algoritmo memético (Algoritmo Genético híbrido com busca local 2-opt) para o Problema do Caixeiro Viajante (TSP). Foram implementadas e comparadas quatro variantes isoalgorítmicas: uma versão sequencial em CPU, uma versão híbrida ingênua (transferência indivíduo a indivíduo), uma versão híbrida otimizada (processamento em lote) e uma versão totalmente residente em GPU. Os experimentos, conduzidos em um conjunto de 38 instâncias da TSPLIB com até 1002 cidades, demonstram que a estratégia de paralelização influencia drasticamente o trade-off entre tempo de execução e qualidade da solução. A variante híbrida otimizada obteve os menores tempos de execução (ganho de performance mediano por instância de 5,69× sobre a híbrida ingênua), enquanto a variante totalmente em GPU alcançou a melhor qualidade de solução (gap médio de 0,88%). Os resultados evidenciam que a minimização da transferência de dados entre CPU e GPU é crítica para o desempenho, e que arquiteturas totalmente residentes favorecem o refinamento das soluções em problemas de otimização combinatória.

**Palavras-chave:** Problema do Caixeiro Viajante, Algoritmos Genéticos, Computação em GPU, CUDA, Otimização Combinatória.

```{=latex}
\clearpage
```

# Abstract {#abstract .unnumbered}

This work investigates the impact of different parallelization strategies on Graphics Processing Units (GPUs) on the performance of a memetic algorithm (Genetic Algorithm hybridized with 2-opt local search) for the Traveling Salesman Problem (TSP). Four iso-algorithmic variants were implemented and compared: a sequential CPU version, a naive hybrid version (individual-by-individual transfer), an optimized hybrid version (batch processing), and a fully GPU-resident version. Experiments conducted on a set of 38 TSPLIB instances with up to 1002 cities demonstrate that the parallelization strategy drastically influences the trade-off between execution time and solution quality. The optimized hybrid variant achieved the lowest execution times (median per-instance Ganho de Performance of 5.69× over the naive hybrid), while the fully GPU-resident variant achieved the best solution quality (average gap of 0.88%). The results highlight that minimizing data transfer between CPU and GPU is critical for performance, and that fully resident architectures favor solution refinement in combinatorial optimization problems.

**Keywords:** Traveling Salesman Problem, Genetic Algorithms, GPU Computing, CUDA, Combinatorial Optimization.

```{=latex}
\clearpage
```

```{=latex}
\clearpage
\csname tableofcontents\endcsname
\clearpage
```

# Glossário de termos {#glossario .unnumbered}

| Termo em inglês | Forma adotada no texto | Observação |
| :--- | :--- | :--- |
| *batch* | lote | Usado para processamento de vários indivíduos em uma única chamada. |
| *benchmark* | benchmark / avaliação experimental | Mantém-se “benchmark” quando o foco é o protocolo comparativo. |
| *crossover* | cruzamento | Preferência por “cruzamento”, mantendo a sigla OX quando necessário. |
| *fitness* | aptidão | Quando útil, escreve-se “aptidão (*fitness*)” na primeira ocorrência. |
| *full GPU* / *GPU-resident* | totalmente em GPU / residente na GPU | Expressões usadas para a variante com dados mantidos em VRAM. |
| *gap* | gap / desvio percentual em relação ao ótimo | O termo “gap” é preservado por ser consagrado na literatura. |
| *host* | hospedeiro (CPU) | Contraposto a *device* (GPU). |
| *kernel launch* | lançamento de *kernel* | O termo *kernel* é mantido por ser técnico e específico de CUDA. |
| *offspring* | descendentes | Refere-se aos indivíduos gerados após cruzamento e mutação. |
| *overhead* | sobrecarga | Usado para custos adicionais de comunicação, sincronização ou inicialização. |
| *speedup* | ganho de performance | Preferência por esta expressão no texto analítico. |
| *swap mutation* | mutação por troca | Operador que troca a posição de duas cidades na permutação. |
| *throughput* | vazão | Preferência por “vazão”, mantendo o inglês quando a literatura exigir. |
| *warp* | *warp* (grupo de 32 *threads*) | O termo inglês é mantido, com explicação curta na primeira ocorrência. |

```{=latex}
\clearpage
```

# Lista de Símbolos {#lista-de-simbolos .unnumbered}

| Símbolo | Descrição | Ver seção |
| :--- | :--- | :--- |
| $n_{coords}$ | Número de cidades (coordenadas) da instância | [3.3](#sec-33-selecao-instancias) |
| $n_{pop}$ | Tamanho da população do Algoritmo Genético | [3.2.1](#sec-321-ga-base) |
| $n_{reps}$ | Número de repetições independentes por experimento | [3.3](#sec-33-selecao-instancias) |
| $p_{\text{mut}}$ | Taxa de mutação | [2.3](#sec-23-algoritmos-geneticos) |
| $p_c$ | Taxa de cruzamento | [2.3](#sec-23-algoritmos-geneticos) |
| $k_{\text{trnmt}}$ | Tamanho do torneio na seleção | [2.5.1](#sec-251-ga-2opt) |
| $n_{2\text{-opt}}$ | Número máximo de iterações da busca local 2-opt | [2.5.1](#sec-251-ga-2opt) |
| $B_{\mathrm{H2D}}$ | Volume de dados transferidos Host-to-Device | [Cap. 3](#cap3-materiais-metodos) |
| $B_{\mathrm{D2H}}$ | Volume de dados transferidos Device-to-Host | [Cap. 3](#cap3-materiais-metodos) |
| $T$ | Tempo de execução | [4.1](#sec-41-resultados-experimentais) |
| $\Delta C$ | Variação de custo em um movimento local | [2.2.1](#sec-221-2opt) |
| $G = (V, E)$ | Grafo composto por vértices $V$ e arestas $E$ | [2.1](#sec-21-otimizacao-tsp) |
| $c_{ij}$ | Custo (distância) entre as cidades $i$ e $j$ | [2.1](#sec-21-otimizacao-tsp) |
| $f(x)$ | Função de aptidão (*fitness*) de uma solução $x$ | [2.3](#sec-23-algoritmos-geneticos) |
| $P(t)$ | População de soluções na geração $t$ | [2.3](#sec-23-algoritmos-geneticos) |
| $\kappa$ | Limite de iterações para busca local | [2.2.1](#sec-221-2opt) e [2.5.1](#sec-251-ga-2opt) |

```{=latex}
\clearpage
```

\listoffigures

```{=latex}
\clearpage
```

\listoftables

# Capítulo 1 – Introdução {#cap1-introducao}

## 1.1 Contexto e motivação {#sec-11-contexto-e-motivacao}

O Problema do Caixeiro Viajante (Traveling Salesman Problem – TSP) é um dos problemas mais estudados em otimização combinatória [@cook2012pursuit]. Tem esse nome devido à sua história clássica: um vendedor precisa visitar um conjunto de cidades, passando por cada uma exatamente uma vez, e deseja minimizar a distância total percorrida.

O objetivo é encontrar um ciclo hamiltoniano (ciclo no qual se deve passar por todos os vértices uma vez e retornar ao vértice original). Apesar de sua formulação simples, o TSP é NP-difícil (classe de problemas para os quais não se conhece algoritmo polinomial eficiente) e, na sua formulação de decisão: "existe um tour com custo menor ou igual a $B$?", é NP-completo: não se conhece algoritmo em tempo polinomial que o resolva em geral, embora seja fácil verificar o custo de uma solução candidata. @cook2012pursuit argumenta que essa combinação de simplicidade, dificuldade teórica e rica estrutura geométrica faz do TSP um estudo de caso central tanto para a teoria da complexidade quanto para desenvolvimento de métodos exatos e heurísticos.

Na prática, variantes do TSP aparecem em domínios como roteirização de veículos, planejamento de inspeções, manufatura e testes de circuitos, genômica e astronomia, entre outros [@cook2012pursuit]. Exemplos clássicos incluem o posicionamento e ordenamento de furos em placas de circuito impresso, logística e problemas de mapeamento genético. Mesmo quando modelos reais são mais complexos (com janelas de tempo, múltiplos veículos ou restrições de capacidade), é comum validar ideias de projeto e análise de algoritmos primeiro em instâncias clássicas do TSP, justamente pela ampla disponibilidade de referências, testes padronizados e de soluções ótimas conhecidas.

Paralelamente, a evolução do hardware trouxe processadores gráficos (GPUs) como plataforma acessível para computação de alto desempenho (HPC) [@nvidia2024cuda] e com a popularização e acessibilidade, pesquisas nestas áreas cresceram rapidamente. Modelos de Inteligência Artificial Generativa, por exemplo, dependem fortemente desse tipo de hardware.

GPUs oferecem milhares de núcleos relativamente simples, organizados em um modelo de execução massivamente paralelo normalmente descrito, em CUDA, como SIMT (*Single Instruction, Multiple Threads*) [@flynn1972taxonomy; @nvidia2024cuda]. Isso não significa que CPUs sejam arquiteturas “SIMD por natureza”: CPUs modernas continuam sendo predominantemente MIMD (*Multiple Instruction, Multiple Data*) no nível de núcleos e *threads*, embora cada núcleo possa explorar extensões vetoriais SIMD (como SSE e AVX) para acelerar operações sobre vários dados por instrução. Em troca de um controle mais restrito nos fluxos de processo e memória, as GPUs entregam grande largura de banda de memória e alta taxa de operações aritméticas por segundo (FLOPs), desde que o problema ofereça operações semelhantes que possam ser executadas em paralelo. Isso torna GPUs particularmente atrativas para tarefas como o cálculo de matrizes, a avaliação de grandes populações de soluções e a aplicação de movimentos de vizinhança independentes em heurísticas de melhoria.

<!-- essa informação eu achei na internet. não sei ainda explicar como funcionam SSE e AVX -->

Do ponto de vista das meta-heurísticas, @crainic2003parallel propõem uma taxonomia de paralelização que distingue estratégias baseadas em decomposição de dados, decomposição de domínio e múltiplas trajetórias cooperativas. A discussão detalhada dessas categorias e sua adequação a arquiteturas de GPU é apresentada na [Seção 2.6.1](#sec-261-taxonomia-paralelizacao).

O projeto desta monografia foca principalmente em algoritmos do tipo 1, explorando paralelismo de dados em vizinhanças de $2$-opt e na avaliação de populações em algoritmos genéticos, em linha com estudos recentes sobre heurísticas para TSP em GPU [@schulz2013gpu; @tsp_gpu; @fujimoto2011highly].

Ainda assim, nem todo algoritmo se beneficia automaticamente de uma migração para GPU. Inicializações de kernels, movimentação de dados entre CPU e GPU podem anular ganhos de paralelismo, especialmente em problemas de porte pequeno ou em implementações que realizam pouco trabalho por inicializações de kernel [@van2013gpu; @luong2013gpu]. Limitações de memória (VRAM) devem ser cuidadosamente calculadas para que a memória total ocupada não passe do limite do hardware. Nesses cenários, uma comparação direta entre versões em CPU e GPU exige, ainda, cuidados extras: deve-se isolar o impacto da plataforma de execução sem confundir o efeito com mudanças na lógica do algoritmo. Ao longo deste trabalho, esses desafios são documentados explicitamente em estudos de caso, onde kernels são chamados de forma não cautelosa e versões otimizadas (Capítulos [3](#cap3-materiais-metodos) e [4](#cap4-resultados)), bem como em notas técnicas específicas sobre *overhead* de lançamentos de kernel e padrões de redução em GPU.

Este trabalho insere-se nesse contexto, investigando como acelerar, utilizando GPU, um Algoritmo Genético (Genetic Algorithm – GA) híbrido com $2$-opt para o TSP, de forma controlada e estatisticamente fundamentada, usando instâncias clássicas da biblioteca TSPLIB [@reinelt1991tsplib] e um protocolo experimental com um número adequado de repetições $n_{reps}$ e testes estatísticos apropriados (detalhado no [Capítulo 3](#cap3-materiais-metodos)). A combinação entre GA e busca local é um exemplo de algoritmo memético amplamente estudado na literatura [@larranaga1999genetic; @goldberg1989genetic], em que operadores evolutivos globais são complementados por heurísticas de melhoria como $2$-opt [@croes1958method] ou Lin–Kernighan [@lin1973efficient]. Neste Trabalho de Conclusão de Curso, será utilizada a nomenclatura em inglês para alguns termos técnicos como "$2$-opt", "TSPLIB", "GPU", "CPU" e "GA".

## 1.2 Problema de pesquisa {#sec-12-problema-de-pesquisa}

Muitos estudos em computação de alto desempenho comparam algoritmos em CPU e GPU utilizando implementações que diferem não apenas na plataforma de execução, mas também em detalhes relevantes da lógica do algoritmo (por exemplo, operadores distintos, parâmetros diferentes ou vizinhanças não equivalentes) [@schulz2013gpu; @van2013gpu; @benaini2018genetic]. Isso torna difícil atribuir ganhos de desempenho exclusivamente ao uso da GPU, uma vez que alterações no desenho algorítmico ou na parametrização podem, por si só, explicar diferenças observadas. Neste trabalho, os parâmetros e operadores são mantidos fixos entre as variantes, e as diferenças de desempenho observadas são discutidas à luz do protocolo experimental apresentado no [Capítulo 3](#cap3-materiais-metodos).

Neste trabalho, busquei implementar as variantes do algoritmo sob as mesmas condições. A estrutura dos algoritmos é projetada para ser idêntica e otimizada tanto para CPU quanto para GPU, de forma justa, para todos os Algoritmos Genéticos e para o algoritmo de busca local $2$-opt, variando apenas a forma como cada "módulo" estrutural do algoritmo é executado (em CPU ou em GPU) e o grau de paralelismo utilizado. A pergunta central pode ser formulada da seguinte forma:

> **Como comparar, de forma fiel, o impacto de diferentes estratégias de paralelização utilizando GPU no desempenho de um algoritmo memético (GA+$2$-opt) para o Problema do Caixeiro Viajante?**

Responder a essa pergunta exige, ao mesmo tempo, um desenho experimental cuidadoso de hiperparâmetros e validade estatística (número significativo de instâncias de problema, número adequado de repetições, métricas de qualidade e tempo). Os parâmetros básicos do GA — como tamanho de população $n_{pop}$, taxa de mutação $p_{\text{mut}}$, tamanho do torneio $k_{\text{trnmt}}$, número de iterações de $2$-opt por indivíduo $n_{2\text{-opt}}$ e número de repetições por combinação algoritmo–instância $n_{reps}$ — são escolhidos com base em recomendações da literatura de algoritmos evolutivos [@eiben2015introduction; @goldberg1989genetic] e em análises específicas documentadas nos relatórios técnicos do projeto. Esses valores e sua motivação são apresentados em detalhe no [Capítulo 3](#cap3-materiais-metodos).

## 1.3 Objetivo geral {#sec-13-objetivo-geral}

O objetivo geral deste trabalho é investigar e quantificar o impacto de diferentes estratégias de paralelização em processadores gráficos, com foco em:

- **Tempo de execução**: medir e comparar o tempo requerido para cada variante do algoritmo memético GA+$2$-opt;
- **Qualidade das soluções**: avaliar o custo final das rotas obtidas e o *gap* relativo ao ótimo conhecido, validando a viabilidade da solução;
- **Manutenção de equivalência funcional**: garantir que as quatro variantes (GA-CPU, GA-Híbrido-Ingênuo, GA-Híbrido-Otimizado, GA-FullGPU) implementem exatamente a mesma lógica algorítmica, diferindo apenas na plataforma de execução e no grau de paralelismo;
- **Metodologia estatística padronizada**: aplicar testes de hipótese apropriados (testes de normalidade, pareados e rankings) conforme recomendado por @demsar2006statistical;
- **Alinhamento com taxonomia de paralelização**: situar as estratégias adotadas no contexto da taxonomia proposta por @crainic2003parallel e aprofundada em @crainic2010parallel.

## 1.4 Objetivos específicos {#sec-14-objetivos-especificos}

Para viabilizar o objetivo geral, definem-se os seguintes objetivos específicos:

1. **Implementar quatro variações "isoalgorítmicas"**[^isoalgorithm-disambiguation] de um Algoritmo Genético híbrido com $2$-opt para o TSP, que diferem apenas na forma e no local de execução (CPU ou GPU):
   [^isoalgorithm-disambiguation]: O termo isoalgorítmico será utilizado neste trabalho para refletir variantes estruturalmente similares em sua composição.
   + uma versão executada puramente na CPU (**GA-CPU**), em que todas as operações (seleção, cruzamento, mutação, $2$-opt e cálculo de custo) são executadas com NumPy, servindo como base em CPU para as comparações. Na prática, essa versão é utilizada como base principalmente para instâncias de menor porte (por exemplo, com $n_{coords} \leq 100$); para instâncias maiores, a comparação de tempo passa a considerar como referência a versão híbrida, conforme discutido no Capítulo 3.
   + uma versão híbrida com $2$-opt executada na GPU e avaliação em CPU (**GA-Híbrido-Ingênuo[^ingenuity-disambiguation]**), que inicia kernels **individualmente** para cada rota e transfere as soluções (ou circuitos) completos entre CPU e GPU a cada iteração;
   + uma versão híbrida otimizada (**GA-Híbrido-Otimizado**), em que $2$-opt e o cálculo de custos são executados em GPU em lote (*batch*), reduzindo o número de lançamentos de kernel e o volume de dados transferidos;
   + uma versão totalmente em GPU (**GA-FullGPU**), na qual população, operadores genéticos, busca local e avaliação permanecem residentes na GPU durante toda a evolução.

   Os nomes utilizados aqui seguem as implementações `GeneticAlgorithmCPU`, `GeneticAlgorithmHybridNaive`, `GeneticAlgorithmHybridOptimized` e `GeneticAlgorithmFullGPU` do framework experimental, cujos detalhes arquiteturais são apresentados no [Capítulo 3](#cap3-materiais-metodos).
   [^ingenuity-disambiguation]: Na literatura, o termo "naive" (ingênuo) é frequentemente utilizado para descrever implementações simples que não exploram otimizações avançadas.

2. **Selecionar um conjunto de instâncias da TSPLIB** com diferentes faixas de tamanho (pequenas $n_{coords} \leq 100$, médias $100 < n_{coords} \leq 400$ e grandes $n_{coords} > 400$). Essa seleção visa demonstrar os ganhos de desempenho que podem ser atingidos e garantir diversidade suficiente para analisar ganhos em escala. A divisão de faixas adotada é inspirada em estudos prévios com a TSPLIB [@reinelt1991tsplib], mas levemente ajustada para refletir o foco deste trabalho nas instâncias pequenas, médias e grandes selecionadas. O ajuste consiste em adotar explicitamente o número de coordenadas $n_{coords} = 100$ e $n_{coords} = 400$ como limites entre as faixas, em linha com o limiar prático utilizado para a comparação CPU/GPU e com a distribuição de tamanhos das instâncias efetivamente utilizadas nos experimentos.

3. **Definir um protocolo experimental reprodutível**, incluindo número de repetições por instância e algoritmo, critérios de parada, parâmetros do GA e limites práticos impostos pela capacidade de memória da GPU. O desenho desse protocolo segue recomendações da literatura de comparação de algoritmos e de testes estatísticos [@demsar2006statistical] e é detalhado no [Capítulo 3](#cap3-materiais-metodos).

4. **Aplicar um conjunto de testes estatísticos apropriados** para comparação de algoritmos, incluindo testes de normalidade (Shapiro–Wilk), testes pareados paramétricos (teste *t-pareado*) e não paramétricos (Wilcoxon), análise de rankings em múltiplas instâncias (teste de Friedman com pós-teste de Nemenyi) e medidas de tamanho de efeito (Cohen $d$), conforme recomendado por @demsar2006statistical e literatura correlata de teste de hipóteses. A metodologia estatística completa é apresentada no [Capítulo 3](#cap3-materiais-metodos).

5. **Analisar os resultados obtidos**, discutindo as possíveis condições em que cada variante algorítmica é vantajosa — como por exemplo, tamanho da instância $n_{coords}$, relação entre custos de comunicação e custo computacional, acesso à memória e volume de dados transferidos entre CPU e GPU nas direções host-to-device (H2D ou CPU $\to$ GPU) e device-to-host (D2H GPU $\to$ CPU) [@nvidia2024cuda] —, limites práticos do uso de GPUs para o contexto de recursos limitados (paralelismo em uma placa gráfica já defasada e limitada) e as implicações para o projeto de algoritmos de roteamento em cenários reais, seguindo as taxonomias de paralelização de meta-heurísticas propostas por @crainic2003parallel e complementadas por @alba2005parallel, além de estudos de caso em problemas de roteamento utilizando GPUs [@schulz2013gpu; @tsp_gpu; @fujimoto2011highly].

## 1.5 Justificativa {#sec-15-justificativa}

Do ponto de vista científico, o TSP continua sendo um problema de referência para avaliar novas ideias em heurísticas e meta-heurísticas [@cook2012pursuit]. A vasta disponibilidade de estudos, instâncias utilizadas em larga escala e muitas soluções ótimas conhecidas, em particular as fornecidas pela TSPLIB [@reinelt1991tsplib], na qual o projeto se baseia, permite medir o desempenho de diferentes algoritmos tanto em termos de qualidade quanto em tempo de execução. Ao longo deste trabalho, será utilizada a notação $n_{coords}$ para o número de coordenadas (cidades) de cada instância, $n_{pop}$ para o tamanho da população do GA, $n_{reps}$ para o número de repetições por combinação algoritmo–instância, $T$ para tempos de execução médios e $B_{\mathrm{H2D}}$, $B_{\mathrm{D2H}}$ para os volumes totais de dados transferidos entre CPU e GPU nas direções host-to-device e device-to-host, respectivamente. Os detalhes de como esses elementos se relacionam com os limites de memória de cada variante são discutidos no [Capítulo 3](#cap3-materiais-metodos).

No campo da computação de alto desempenho, diferentes trabalhos relatam ganhos relevantes ao portar heurísticas de melhoria e algoritmos evolutivos para GPU, inclusive em variantes do TSP [@schulz2013gpu; @tsp_gpu; @fujimoto2011highly]. Esses resultados indicam que GPUs são um hardware promissor para esse tipo de algoritmo, mas também evidenciam um problema recorrente: em muitas comparações, as versões em CPU e GPU diferem não apenas na plataforma de execução, mas também em detalhes de implementação e parametrização, o que dificulta isolar o efeito específico da paralelização.

Além disso, para que conclusões sobre desempenho sejam confiáveis, não basta observar poucos experimentos isolados. É necessário adotar um desenho experimental com múltiplas repetições, tanto para instâncias quanto para, medidas de dispersão e testes de hipótese adequados, como discutido na literatura de comparação de algoritmos [@demsar2006statistical]. Neste trabalho, esses cuidados são incorporados ao experimento, de modo que as diferenças observadas entre variantes em CPU e GPU possam ser atribuídas, com maior segurança, às estratégias de paralelização avaliadas.

Este trabalho justifica-se por combinar três elementos que, em conjunto, fortalecem a contribuição científica:

1. **Implementação de heurísticas clássicas em GPU** (particularmente $2$-opt e operadores do AG), explorando o paralelismo de forma explícita.
2. **Comparação entre variantes funcionalmente equivalentes do algoritmo em CPU e GPU**, mantendo fixos operadores, parâmetros e critérios de parada, de modo a isolar o efeito da plataforma de execução. [^small-variations]
   [^small-variations]: Algumas variações dentro deste contexto são esperadas, mas de modo geral, o algoritmo segue a mesma estrutura. Sempre há a possibilidade de haver pequenas variações, mas o autor busca manter, ao máximo, a consistência entre as variantes.
3. **Aplicação de metodologia estatística rigorosa**, com múltiplas repetições por instância, testes de hipótese adequados e medidas de tamanho de efeito.

Os capítulos seguintes discutem em que condições cada variante é vantajosa, quais os limites práticos do uso de GPUs e recursos limitados e como os resultados dialogam com estudos prévios de heurísticas em GPU.

## 1.6 Organização do trabalho {#sec-16-organizacao}

Este texto está organizado da seguinte forma:

- **Capítulo 2 – Fundamentação Teórica / Revisão Bibliográfica**: apresenta os conceitos básicos de otimização combinatória e do TSP, revisa heurísticas de construção e melhoria (com ênfase em $2$-opt), discute algoritmos genéticos e meta-heurísticas híbridas, introduz princípios de computação em GPU e paralelização de meta-heurísticas, e resume abordagens estatísticas para comparação de algoritmos.
- **Capítulo 3 – Materiais e Métodos**: descreve o ambiente computacional, a arquitetura do framework desenvolvido (incluindo abstrações de backend e estratégias isoalgorítmicas), os detalhes do Algoritmo Genético híbrido com $2$-opt nas quatro variantes consideradas, o conjunto de instâncias TSPLIB utilizado, os parâmetros experimentais e o protocolo estatístico adotado.
- **Capítulo 4 – Resultados**: apresenta os resultados numéricos dos experimentos, incluindo estatísticas por instância, análises agregadas por faixa de tamanho, medidas de ganho de performance e qualidade das soluções, além da interpretação dos testes estatísticos aplicados.
- **Capítulo 5 – Discussão**: interpreta criticamente os resultados à luz da literatura, discute as implicações dos achados para o projeto de algoritmos de roteamento em GPU e analisa limitações do estudo.
- **Capítulo 6 – Conclusões e Trabalhos Futuros**: sintetiza as principais contribuições do trabalho, responde explicitamente aos objetivos propostos e indica possíveis extensões, como a aplicação do framework a problemas de roteirização mais complexos.
- Elementos pós-textuais, como referências, glossário, apêndices técnicos e anexos com tabelas completas de resultados, são apresentados ao final do documento, conforme normas da instituição.

# Capítulo 2 - Revisão bibliográfica {#cap2-revisao-bibliografica}

## 2. Fundamentação teórica e revisão bibliográfica {#sec-2-fundamentacao-e-revisao}

Este capítulo apresenta conceitos teóricos e a bibliografia utilizada para contextualizar o problema estudado e as escolhas metodológicas adotadas. São discutidos o Problema do Caixeiro Viajante (TSP) no contexto da otimização combinatória [@cook2012pursuit; @rosenkrantz1977analysis], sua importância geral e exemplos de utilização. Também são apresentadas generalizações do problema e sua relação com problemas de roteamento de veículos e variantes correlatas [@raff1983routing; @bodin1981state; @tan2021vehicle], heurísticas e meta-heurísticas clássicas para o TSP [@croes1958method; @lin1973efficient; @voudouris1999tsp], aprofundando-se então no tópico de algoritmos genéticos e algoritmos meméticos [@larranaga1999genetic; @goldberg1989genetic], princípios de computação em GPU [@nvidia2024cuda; @schulz2013gpu; @tsp_gpu], taxonomias de paralelização de meta-heurísticas [@crainic2003parallel; @crainic2010parallel; @alba2005parallel] e de comparação estatística de algoritmos de otimização [@demsar2006statistical].

### 2.1 Otimização combinatória e o Problema do Caixeiro Viajante {#sec-21-otimizacao-tsp}

O TSP pode ser formulado, na versão simétrica[^ATSP], como um problema de encontrar um ciclo hamiltoniano de custo mínimo — podendo este ser a distância percorrida, combustível gasto, ou mesmo uma combinação de fatores. Para fins deste trabalho, o custo será também chamado de distância — em um grafo completo não direcionado $G = (V, E)$, no qual cada vértice em $V$ representa uma cidade e cada aresta $(i, j) \in E$ possui um custo $c_{ij} \geq 0$ entre as cidades $i$ e $j$ [@cook2012pursuit]. O objetivo é determinar uma combinação das cidades que minimize a soma total dos custos, retornando à cidade de origem. Geralmente em aplicações práticas assume-se que a matriz de custos é métrica e euclidiana, isto é, os custos derivam de distâncias euclidianas entre coordenadas no plano.

[^ATSP]: Na versão assimétrica do TSP, os custos de viagem entre pares de cidades podem diferir dependendo da direção (ou seja, $c_{ij} \neq c_{ji}$). Em TSPs simétricos, o custo do cálculo de distâncias pode ser representado por uma matriz triangular que armazena apenas uma das metades (superior ou inferior) sem a diagonal, o que reduz o número de entradas únicas de $n^2$ para $n(n-1)/2$.

Do ponto de vista da complexidade, o TSP é um problema NP-difícil em sua forma de otimização e NP-completo em sua forma de decisão [@cook2012pursuit]. Isso significa que, em geral, não se conhece um algoritmo em tempo polinomial que resolva instâncias arbitrárias em grande escala. Mesmo que seja relativamente fácil verificar o custo de uma ou várias soluções candidatas, não podemos afirmar que esta é a solução exata em tempo polinomial. Algoritmos modernos como `Held-Karp` [@held1962dynamic] com uma complexidade temporal $O(n^2 2^n)$ e o algoritmo `Concorde` [@cook2012pursuit] com uma complexidade temporal indefinida, conseguem resolver instâncias com milhares de cidades, mas seu tempo de execução cresce exponencialmente ($O(n^2 2^n)$), o que, embora assintoticamente muito superior à força bruta ($O(n!)$), ainda o torna impraticável para instâncias muito grandes [^held-karp].

[^held-karp]: O algoritmo Held-Karp "troca" a complexidade temporal da força bruta $O(n!)$ por uma complexidade espacial $O(n 2^n)$, tornando sua viabilidade limitada a instâncias com $n_{coords} \approx 40$ cidades. Já o algoritmo Concorde, baseado em técnicas de ramificação e corte, é capaz de resolver instâncias com até 85.900 cidades, mas seu desempenho depende fortemente da estrutura específica da instância [@cook2012pursuit].

<!-- Interessante notar que isto também acontece para diversos outros algoritmos, inclusive algoritmos genéticos, a depender do método aplicado, sequência, tipo de busca -->

Ao longo das últimas décadas, o TSP consolidou-se também como um padrão de avaliação empírica de algoritmos, graças à disponibilidade de coleções de instâncias padronizadas, como a TSPLIB [@reinelt1991tsplib]. Essas coleções incluem instâncias com diferentes tamanhos, estruturas e origens (geográficas, sintéticas, industriais), a maioria delas com soluções ótimas conhecidas e obtidas por métodos exatos. No presente trabalho, são utilizadas algumas dessas instâncias como base para avaliar e comparar heurísticas e variantes paralelas de um algoritmo memético (combinações de algoritmos evolutivos com busca local) [@larranaga1999genetic; @fujimoto2011highly].

### 2.2 Heurísticas para o TSP {#sec-22-heuristicas-tsp}

Devido à dificuldade de resolver instâncias grandes do TSP exatamente, uma vasta literatura de heurísticas tem sido desenvolvida para produzir boas soluções em tempos de computação aceitáveis [@cook2012pursuit]. De forma geral, heurísticas podem ser agrupadas em dois grandes tipos: heurísticas de construção e heurísticas de melhoria.

Heurísticas de construção produzem uma solução viável “do zero”, frequentemente seguindo regras simples, também conhecidas como algoritmos gulosos, como escolher iterativamente o vizinho mais próximo ou inserir cidades em posições que causem o menor aumento de custo. Exemplos incluem a heurística do vizinho mais próximo (nearest neighbor — NN), heurísticas de inserção (insertion heuristics) e variantes baseadas em árvores de extensão mínimas (Minimum Spanning Trees — MSTs), como o Algoritmo de Christofides. Embora rápidas e fáceis de implementar, essas estratégias tendem a gerar soluções de qualidade moderada, servindo sobretudo como ponto de partida para métodos mais sofisticados. Resultados clássicos da literatura mostram que, em instâncias métricas, heurísticas baseadas em `MST` e em `Christofides` podem oferecer garantias de aproximação sobre o custo ótimo de $\frac{3}{2}B^*$, aspecto discutido em textos de referência em logística e roteamento [@simchi2005logic].

Heurísticas de melhoria partem de uma solução inicial e aplicam sucessivos movimentos locais que procuram reduzir seu custo. Entre essas, as chamadas $k$-opt são particularmente influentes: um movimento $2$-opt consiste em remover duas arestas de um circuito e reconectar os segmentos resultantes, escolhendo a reconexão que produz um circuito ainda viável e de menor comprimento [@croes1958method]. Em instâncias euclidianas é comum interpretar o $2$-opt como um mecanismo para eliminar cruzamentos, mas, genericamente, ele também pode melhorar circuitos que não apresentam interseções evidentes, ao substituir pares de arestas por combinações de menor custo [@croes1958method]. Movimentos $3$-opt e extensões mais complexas, como o algoritmo de Lin–Kernighan, generalizam essa ideia [@lin1973efficient]. Estas e outras heurísticas de melhoria desempenham papel central em muitos algoritmos modernos para TSP, tanto como procedimentos isolados quanto como componentes de meta-heurísticas, sendo utilizadas como procedimentos de busca local.

Neste trabalho, a heurística $2$-opt é utilizada como busca local básica acoplada ao Algoritmo Genético, em linha com estudos que combinam heurísticas de construção simples com procedimentos de melhoria mais complexos para obter soluções de alta qualidade em tempo razoável [@voudouris1999tsp; @larranaga1999genetic; @fujimoto2011highly].

#### 2.2.1 Entendendo o algoritmo $2$-opt {#sec-221-2opt}

A formulação básica de um movimento $2$-opt pode ser descrita da seguinte forma [@croes1958method]: dado um circuito $C = (v_0, v_1, \dots, v_{n-1}, v_0)$ e dois índices $i$ e $j$ tais que $0 \leq i < n-1$ e $i+2 \leq j < n$, considera-se a remoção das arestas $(v_i, v_{i+1})$ e $(v_j, v_{j+1})$, adotando-se a convenção circular de que, quando $j = n-1$, o sucessor de $v_j$ é novamente $v_0$. Em seguida, inserem-se as arestas $(v_i, v_j)$ e $(v_{i+1}, v_{j+1})$. Essa reconexão implica reverter a ordem do segmento intermediário $(v_{i+1}, \dots, v_j)$ do circuito: em outras palavras, o $2$-opt não apenas troca duas conexões, mas inverte o subcaminho entre elas. Essa operação é natural no TSP simétrico clássico, pois a rota continua viável mesmo após a reversão. Em variantes assimétricas ou em problemas com restrições adicionais, uma reversão ingênua pode alterar o significado das arestas ou a viabilidade da solução, exigindo vizinhanças adaptadas. A variação de custo associada ao movimento é dada por

$$
\Delta C = \bigl(c_{i,j} + c_{i+1,j+1}\bigr)
         - \bigl(c_{i,i+1} + c_{j,j+1}\bigr)
$$

onde $c_{i,j}$ denota o custo (ou distância) entre as cidades $i$ e $j$. Um movimento $2$-opt é aceitável quando $\Delta C < 0$, isto é, quando a substituição das arestas reduz o comprimento total do circuito [@croes1958method].

De forma geral, a vizinhança $2$-opt de um circuito é formada por todos os pares admissíveis $(i,j)$ que definem duas arestas a serem removidas e um segmento intermediário a ser invertido. Uma iteração do método corresponde a uma varredura dessa vizinhança. Dependendo da variante adotada, o algoritmo pode aplicar o primeiro movimento melhorador encontrado (*first-improvement*) ou examinar toda a passagem e aplicar apenas o melhor (*best-improvement*).

Assim, quando se determina "$\kappa = 10$ iterações em $2$-opt", significa permitir até 10 varreduras sucessivas da vizinhança da rota corrente. Isso não significa “varrer 10 vezes os 10 primeiros nós”: os índices percorrem a rota inteira em cada passagem, e a utilidade de múltiplas iterações está no fato de que, após cada reversão, surgem novas arestas e novas oportunidades de melhoria. Sem esse teto, o algoritmo continuaria até atingir um ótimo local — o que pode ser custoso computacionalmente em circuitos grandes [@lin1973efficient]. A forma específica como esta monografia usa essas 10 varreduras dentro do algoritmo memético é detalhada na [Seção 2.5](#sec-25-algoritmos-memeticos).

<!-- Como discutiomos, possível fonte de ineficiência algorítmica -->

De forma mais detalhada, o algoritmo $2$-opt clássico pode ser descrito como um procedimento iterativo de busca local aplicada a um único circuito. Em linhas gerais, o algoritmo segue os passos:

1. Começar com um circuito viável inicial $C$ (obtido por uma heurística de construção qualquer).
2. Definir uma ordem de varredura para pares de índices $(i,j)$ com $0 \leq i < n-1$ e $i+2 \leq j < n$. No presente trabalho, utiliza-se a ordem lexicográfica padrão, em que para cada $i$, varia-se $j = i+2$ até $n-1$ (a restrição $j \geq i+2$ evita movimentos triviais[^trivial] e garante que o segmento a ser revertido tenha comprimento mínimo [@croes1958method]). Quando $j = n-1$, considera-se novamente o vértice inicial $v_0$ como sucessor da última cidade. Cada passagem completa por essa ordem conta como **uma iteração** da busca local.
   [^trivial]: Movimentos triviais são aqueles que não alteram efetivamente o circuito, como tentar reverter um segmento de $i$ a $i+1$, pois $c_{i, i+1}=c_{i+1, i}$ é uma aresta única e sua reversão não muda o circuito, no caso do TSP clássico explorado neste documento. O mesmo não se aplica para o ATSP.
3. Para cada par possível $(i,j)$, calcular $\Delta C$ conforme a expressão acima.
4. Selecionar um critério de atualização: aplicar o primeiro melhor movimento encontrado (*first-improvement*) ou, alternativamente, concluir a passagem e aplicar apenas o melhor movimento da iteração (*best-improvement*).
5. Repetir a varredura sobre o circuito atualizado até que nenhum movimento com $\Delta C < 0$ exista ou até alcançar o limite máximo de iterações $\kappa$.

Em uma implementação ingênua, a vizinhança $2$-opt de um circuito com $n$ cidades possui ordem $O(n^2)$ movimentos possíveis, o que implica um custo potencialmente elevado quando todos os pares $(i,j)$ são examinados de forma exaustiva. Diversas otimizações são discutidas na literatura, como o uso de listas de vizinhança, estruturas de dados para poda de movimentos claramente não promissores e estratégias de parada antecipada [@lin1973efficient; @larranaga1999genetic]. No contexto desta monografia, o foco recai principalmente na forma como esse procedimento é paralelizado e acoplado ao Algoritmo Genético, mais do que em otimizações finas da vizinhança em CPU.

A Figura~\ref{fig:2opt-flowchart} ilustra uma variante clássica do algoritmo $2$-opt baseada em *best-improvement*: o laço externo continua enquanto houver melhorias e o número de varreduras completas não exceder o limite $\kappa$. Em cada iteração, todos os pares admissíveis $(i,j)$ são examinados, o melhor movimento da passagem é armazenado e apenas esse melhor movimento é aplicado ao final da varredura.

```{.mermaid #fig:2opt-flowchart}
flowchart TD
    Start(["Início: Circuito C"]) --> Init["varredura ← 0"]
    Init --> CheckIter{"varredura < κ?"}
    CheckIter -->|Não| End(["Fim: Retorna C"])
    CheckIter -->|Sim| ResetBest["melhor_Δ ← 0<br/>melhor_i ← -1<br/>melhor_j ← -1"]
    ResetBest --> LoopI["Para i = 0 até n-2"]
    LoopI --> LoopJ["Para j = i+2 até n-1"]
    LoopJ --> CalcDelta["Calcular ΔC = (c_ij + c_i+1,j+1)<br/>- (c_i,i+1 + c_j,j+1)"]
    CalcDelta --> CheckDelta{"ΔC < melhor Δ?"}
    CheckDelta -->|Sim| StoreBest["Guardar melhor par<br/>(i,j) e melhor_Δ"]
    CheckDelta -->|Não| NextJ
    StoreBest --> NextJ["Próximo j"]
    NextJ -->|Mais j| LoopJ
    NextJ -->|Fim j| NextI["Próximo i"]
    NextI -->|Mais i| LoopI
    NextI -->|Fim i| ApplyBest{"melhor_Δ < 0?"}
    ApplyBest -->|Sim| Apply["Reverter segmento<br/>(melhor_i+1..melhor_j)<br/>varredura ← varredura + 1"]
    ApplyBest -->|Não| End
    Apply --> CheckIter

```

\standalonefiglabel{fig:2opt-flowchart}

Figura~\ref{fig:2opt-flowchart}: Fluxograma do algoritmo $2$-opt.

```algorithm
Entrada: circuito inicial C, limite κ
Saída: circuito melhorado C

varredura = 0

enquanto varredura < κ faça
    melhor_delta = 0
    melhor_i = -1
    melhor_j = -1

    para i = 0 até n-2 faça
        para j = i+2 até n-1 faça
            delta = dist(i, j) + dist(i+1, j+1) - (dist(i, i+1) + dist(j, j+1))

            se delta < melhor_delta então
                melhor_delta = delta
                melhor_i = i
                melhor_j = j
            fim se
        fim para
    fim para

    se melhor_delta < 0 então
        Reverter segmento C[melhor_i+1...melhor_j]
        varredura = varredura + 1
    senão
        interromper
    fim se
fim enquanto

retornar C
```

### 2.3 Algoritmos genéticos {#sec-23-algoritmos-geneticos}

Algoritmos Genéticos (Genetic Algorithms – GAs) são meta-heurísticas inspiradas em princípios de evolução biológica, nas quais uma população de soluções candidatas é iterativamente modificada por operadores análogos à seleção natural, recombinação e mutação [@goldberg1989genetic; @eiben2015introduction]. Na sua forma mais simples, um AG mantém uma população de indivíduos representando soluções para o problema; em cada geração, indivíduos são selecionados com base em uma função de aptidão (fitness), recombinados por operadores de cruzamento e perturbados por operadores de mutação. A nova população resultante substitui total ou parcialmente a anterior, e o processo se repete até que um critério de parada seja satisfeito.

No contexto do TSP, é comum representar cada solução como uma permutação das cidades, utilizar operadores de cruzamento especializados para rotas — como Order Crossover (OX), Partially Mapped Crossover (PMX), entre outros — e empregar mutações que preservem a viabilidade da permutação, como trocas de posição entre duas cidades [@larranaga1999genetic]. Estudos clássicos mostram que GAs podem produzir soluções competitivas para o TSP quando combinados com operadores e parâmetros adequados [@goldberg1989genetic].

Do ponto de vista formal, um GA opera sobre uma população $P(t) = \{x_1^{(t)}, x_2^{(t)}, \dots, x_{n_{pop}}^{(t)}\}$ de indivíduos (soluções candidatas) na geração $t$. Cada indivíduo $x_i$ possui um valor de aptidão (*fitness*) $f(x_i)$ que mede a qualidade da solução; no caso do TSP, $f(x_i)$ corresponde ao custo do circuito representado por $x_i$, e busca-se minimizar esse valor. A cada geração, o GA aplica três operadores principais [@goldberg1989genetic; @eiben2015introduction]:

1. **Seleção**: escolhe indivíduos de $P(t)$ com probabilidade proporcional (ou baseada em ranking/torneio) à sua aptidão, favorecendo soluções de melhor qualidade. Formalmente, a probabilidade de selecionar $x_i$ pode ser dada por $p_i = f(x_i) / \sum_{j=1}^{n_{pop}} f(x_j)$ (seleção proporcional) ou por esquemas de torneio, nos quais um subconjunto aleatório de indivíduos compete e o melhor é escolhido.

2. **Cruzamento (recombinação)**: combina pares de indivíduos selecionados ("pais") para produzir descendentes. No TSP, operadores como OX e PMX preservam a viabilidade das permutações, herdando subsequências de um dos pais e preenchendo as posições restantes com a ordem do outro [@larranaga1999genetic]. A taxa de cruzamento $p_c$ controla a fração de pares submetidos a esse operador.

3. **Mutação**: introduz pequenas perturbações aleatórias em indivíduos, mantendo diversidade na população. No TSP, mutações típicas incluem troca de duas cidades ou reversão de segmentos. A taxa de mutação $p_{\text{mut}}$ controla a probabilidade de cada indivíduo sofrer mutação.

Após aplicar esses operadores, forma-se a nova população $P(t+1)$ por meio de um esquema de substituição (geracional, com elitismo, steady-state, etc.). A Figura~\ref{fig:ga-flowchart} ilustra o fluxo geral de um GA [@goldberg1989genetic; @eiben2015introduction]: a população evolui iterativamente por meio de seleção (escolha de pais com base em aptidão), cruzamento (combinação de pares de pais gerando descendentes), e mutação (perturbações para diversidade), até que um critério de parada seja atingido.

```pseudocode
INÍCIO AG
    Gerar população inicial aleatoriamente.
    ENQUANTO critério_de_parada não satisfeito FAÇA
        INÍCIO
            Selecionar pais da população.
            Produzir descendentes a partir dos pais selecionados.
            Mutar os indivíduos.
            Expandir a população adicionando os descendentes.
            Reduzir a população expandida.
        FIM
    Retornar o melhor indivíduo encontrado.
FIM AG
```

```{.mermaid #fig:ga-flowchart}
flowchart TD
    Start(["Início"]) --> InitPop["Gerar população inicial P(0)<br/>aleatória ou heurística"]
    InitPop --> Eval0["Avaliar aptidão f(x_i)<br/>para cada indivíduo"]
    Eval0 --> SetT["t ← 0"]
    SetT --> CheckStop{"Critério de<br/>parada<br/>satisfeito?"}
    CheckStop -->|Sim| Output(["Fim: Retorna<br/>melhor solução"])
    CheckStop -->|Não| Selection["Seleção: escolher pais<br/>de P(t) com base em f(x_i)"]
    Selection --> Crossover["Cruzamento: combinar pares<br/>de pais (taxa p_c) gerando<br/>descendentes"]
    Crossover --> Mutation["Mutação: perturbar indivíduos<br/>(taxa p_mut) para diversidade"]
    Mutation --> EvalNew["Avaliar aptidão dos<br/>novos indivíduos"]
    EvalNew --> Replace["Substituição: formar P(t+1)<br/>combinando P(t) e descendentes<br/>(com ou sem elitismo)"]
    Replace --> IncT["t ← t + 1"]
    IncT --> CheckStop
```

\standalonefiglabel{fig:ga-flowchart}

Figura~\ref{fig:ga-flowchart}: Fluxograma de Algoritmo Genético.

```algorithm
Input: Population size N, mutation rate p_mut, crossover rate p_cross
Output: Best solution found

Initialize population P(0) randomly
Evaluate fitness of all individuals in P(0)
t = 0

while not stop_condition do
    P_new = {}

    // Elitism: keep best individuals
    Add best of P(t) to P_new

    while size(P_new) < N do
        // Selection
        parent1 = TournamentSelection(P(t))
        parent2 = TournamentSelection(P(t))

        // Crossover
        if random() < p_cross then
            child = OrderCrossover(parent1, parent2)
        else
            child = parent1
        end if

        // Mutation
        if random() < p_mut then
            SwapMutation(child)
        end if

        Add child to P_new
    end while

    P(t+1) = P_new
    Evaluate fitness of P(t+1)
    t = t + 1
end while

return Best of P(t)
```

### 2.4 Heurísticas híbridas {#sec-24-heuristicas-hibridas}

De forma ampla, heurísticas híbridas combinam duas ou mais estratégias de busca, procurando explorar sinergias entre métodos com forças complementares. No contexto de problemas de roteamento e, em particular, do TSP, é comum combinar heurísticas de construção (como vizinho mais próximo, inserções ou heurísticas baseadas em MST) com heurísticas de melhoria (como movimentos $k$-opt) ou com meta-heurísticas populacionais (como GAs), de modo que uma componente forneça boas soluções iniciais e outra detalhe a exploração de vizinhanças [@voudouris1999tsp].

Diversos autores classificam heurísticas híbridas em categorias como: (i) **hibridização em nível de solução**, na qual uma meta-heurística utiliza sistematicamente uma busca local (ou outro procedimento) para melhorar indivíduos candidatos; (ii) **hibridização em nível de algoritmo**, em que diferentes técnicas (por exemplo, um método exato e um heurístico) são combinadas em fases distintas ou em esquemas de cooperação; e (iii) **hibridizações multiestágio**, nas quais diferentes heurísticas atuam em momentos distintos do processo de resolução (inicialização, intensificação, diversificação, etc.) [@larranaga1999genetic; @voudouris1999tsp]. Essas categorias ajudam a organizar o espectro de abordagens híbridas reportadas na literatura, embora nem sempre haja consenso absoluto sobre as fronteiras entre elas.

Uma extensão importante dessa ideia é o conceito de algoritmos meméticos, nos quais operadores evolutivos (seleção, cruzamento, mutação) são combinados com heurísticas de busca local aplicadas a indivíduos da população, de forma sistemática [@larranaga1999genetic]. Em problemas de roteamento, isso resulta em esquemas GA+LS, nos quais um GA guia a exploração global do espaço de soluções e um procedimento de melhoria, como $2$-opt ou Lin–Kernighan, refina soluções promissoras. Esses algoritmos tendem a oferecer melhor equilíbrio entre exploração e intensificação do que GAs “puros”, ao custo de maior tempo computacional por iteração.

Trabalhos como o de @fujimoto2011highly exploram justamente essa combinação GA+busca local para o TSP em ambientes de computação paralela, motivando a adoção de uma abordagem semelhante neste estudo. A [próxima seção](#sec-25-algoritmos-memeticos) aprofunda o conceito de algoritmos meméticos e a [subseção 2.5.1](#sec-251-ga-2opt) descreve, em linhas gerais, o esquema GA+$2$-opt adotado nesta monografia.

### 2.5 Algoritmos meméticos {#sec-25-algoritmos-memeticos}

Algoritmos meméticos podem ser vistos como uma família de meta-heurísticas evolutivas que incorporam, explicitamente, operadores de busca local ao ciclo de evolução populacional [@larranaga1999genetic]. A metáfora usual associa o termo “meme” a unidades de informação que se propagam e se transformam, de modo análogo a genes, mas em um nível mais alto de organização: além da recombinação e mutação de indivíduos, há um processo de “aprendizado” local que modifica soluções de forma dirigida.

Em um algoritmo memético típico, a cada geração, parte dos indivíduos (por exemplo, os mais aptos ou uma amostra da população) é submetida a uma busca local, que pode ser uma heurística de melhoria como $2$-opt, $3$-opt ou Lin–Kernighan no caso do TSP. Esse procedimento permite refinar soluções já boas, acelerando a convergência em direção a ótimos locais de alta qualidade. O GA fornece a diversidade global por meio de operadores de cruzamento e mutação, enquanto a busca local intensifica a exploração em torno de regiões promissoras do espaço de soluções.

O desenho de um algoritmo memético envolve decisões importantes, como: (i) **quais indivíduos** serão submetidos à busca local (apenas a elite, um subconjunto aleatório, toda a população); (ii) **com que frequência** a busca local será aplicada (toda geração, gerações alternadas, fases específicas do processo); e (iii) **com que intensidade** cada chamada de busca local será executada (por exemplo, número máximo de iterações de $2$-opt). Essas escolhas impactam o balanço entre qualidade das soluções e custo computacional, bem como a diversidade mantida na população.

Estudos fundamentais, como o de Larrañaga et al., indicam que os algoritmos meméticos são extremamente eficientes para o Problema do Caixeiro Viajante (TSP). Estes métodos funcionam melhor quando combinam o cruzamento de rotas com técnicas de melhoria intensiva. Neste trabalho, utilizamos um **Algoritmo Genético adaptado para rotas**, que possui as seguintes características:

1. **Representação Numérica:** Cada solução é um circuito $C$ denotando uma sequência de números inteiros representando a ordem das cidades.
2. **Seleção por Torneio:** Os "pais" são escolhidos através de competições diretas entre pequenos grupos de indivíduos.
3. **Cruzamento Ordenado (OX):** Um operador que preserva a ordem relativa das cidades herdadas dos pais.
4. **Mutação por Troca:** Pequenas alterações feitas trocando duas cidades de posição na rota.
5. **Substituição com Elitismo:** A nova geração substitui a antiga, mas preserva obrigatoriamente as melhores soluções encontradas até o momento.
6. **Hibridização:** Após essas etapas, aplica-se uma rotina de busca local ($2$-opt) para refinar os resultados.

A viabilidade de acelerar esse processo usando placas de vídeo (GPU) é reforçada por @fujimoto2011highly. Nesse estudo, os autores implementaram esse algoritmo em uma GPU (modelo GTX285) utilizando técnicas eficientes de soma paralela para gerenciar os dados. Ao comparar o desempenho contra um processador tradicional operando com **apenas um núcleo** (*single-core*), observaram que a GPU foi até **24,2 vezes mais rápida** na instância `ts225`. Embora a comparação tenha sido feita contra um processador mais antigo, o resultado mostra que problemas de tamanho médio (200-300 cidades) conseguem tirar proveito do paralelismo massivo da GPU.

<!-- levanta algumas dúvidas sobre se minha implementação é realmente parecida, já que os resultados obtidos por eles foi muito melhor que os meus -->

O processo formal de refinamento segue a lógica abaixo:

1. Considere uma população inicial $P(t)$.
2. Aplique os operadores genéticos (cruzamento e mutação) para criar descendentes.
3. Execute a função de **Busca Local Truncada**, denotada por $\mathcal{L}(x, \kappa)$, sobre cada descendente selecionado para refinamento.
4. Nesta monografia, $\mathcal{L}(x, \kappa)$ significa aplicar sucessivas **varreduras completas** da vizinhança $2$-opt da rota $x$: em cada varredura, os índices $(i,j)$ percorrem toda a rota admissível, calcula-se $\Delta C$ para cada par e registra-se apenas o melhor movimento encontrado naquela passagem.
5. Ao final de cada varredura, aplica-se no máximo **uma** reversão de segmento — a melhor da passagem —, desde que o melhor $\Delta C$ seja negativo. Em outras palavras, a implementação da monografia usa a variante *best-improvement* do $2$-opt, e é aqui, no contexto memético, que a escolha específica da varredura passa a importar metodologicamente.
6. O processo continua até que:
    + a rota não possa mais ser melhorada (ótimo local);
    **OU**
    + o número máximo de varreduras ($\kappa$) seja atingido.
7. Nos experimentos deste trabalho, usa-se $\kappa = n_{2\text{-opt}} = 10$. Portanto, cada descendente recebe no máximo **10 varreduras completas de $2$-opt** sobre toda a sua vizinhança admissível, e não “10 nós” ou “10 trocas locais” arbitrárias.
8. O parâmetro $\kappa$ funciona como um "freio": ele limita o tempo gasto refinando cada solução para evitar que o algoritmo fique lento demais, garantindo um equilíbrio entre qualidade e velocidade.

> [!warning] not what I wanted
> **Exemplo ilustrativo.** Considere uma rota com 6 cidades. Em uma única varredura *best-improvement*, o algoritmo examina os 10 pares admissíveis $(0,2)$, $(0,3)$, $(0,4)$, $(0,5)$, $(1,3)$, $(1,4)$, $(1,5)$, $(2,4)$, $(2,5)$ e $(3,5)$. Suponha que o melhor movimento da primeira passagem seja $(1,4)$; então o segmento interno dessa rota é invertido uma vez. Na segunda varredura, os mesmos pares de índices voltam a ser analisados, mas agora sobre **outra rota**, pois as arestas mudaram após a reversão. É justamente por isso que faz sentido permitir até 10 varreduras: cada nova passagem reavalia uma configuração diferente do circuito. Se, por exemplo, na terceira passagem nenhum $\Delta C < 0$ for encontrado, a busca local para esse indivíduo termina antes de chegar ao limite de 10.
>

#### 2.5.1 Algoritmo Genético + $2$-opt {#sec-251-ga-2opt}

O esquema GA+$2$-opt adotado neste trabalho segue a linha de algoritmos meméticos padrões para o TSP [@larranaga1999genetic], nos quais um Algoritmo Genético opera sobre uma população de permutações de cidades e, em momentos específicos do ciclo evolutivo, aplica-se uma rotina de busca local $2$-opt a alguns indivíduos. Em alto nível, cada iteração (geração) do algoritmo pode ser descrita pelos seguintes passos:

1. **Inicialização ($t=0$)**: gerar uma população inicial $P(0)$ de rotas, por exemplo, a partir de permutações aleatórias ou de heurísticas de construção simples (vizinho mais próximo, inserções).
2. **Avaliação (*fitness*)**: calcular o custo (comprimento) $f(x_i)$ de cada rota $x_i \in P(t)$, que servirá como medida de aptidão.
3. **Seleção**: escolher indivíduos para reprodução com base em seus valores de aptidão, utilizando seleção por torneio de tamanho $k_{\text{trnmt}}$, favorecendo soluções de menor custo.
4. **Cruzamento**: combinar pares de indivíduos selecionados por meio de Order Crossover (OX), produzindo descendentes que preservam a estrutura de permutação e herdam subsequências de ambos os pais.
5. **Mutação**: aplicar perturbações leves às rotas (trocas de posição de cidades, com taxa $p_{\text{mut}}$) para manter diversidade genética.
6. **Busca local $2$-opt**: para cada descendente (ou um subconjunto, conforme a estratégia), aplicar $\mathcal{L}(x, n_{2\text{-opt}})$ — isto é, executar até $n_{2\text{-opt}}$ varreduras completas de $2$-opt, cada uma examinando toda a vizinhança admissível e aplicando apenas o melhor movimento encontrado — de modo a refinar localmente as rotas resultantes. Esta etapa é o que transforma o GA em um algoritmo memético, introduzindo intensificação explícita da busca.
7. **Substituição**: formar a nova população $P(t+1)$ combinando os melhores indivíduos de $P(t)$ (elitismo) com os descendentes refinados, garantindo que as melhores soluções sejam preservadas ao longo das gerações.

O processo repete-se até que um critério de parada seja satisfeito (número máximo de gerações, estagnação da aptidão ou atingimento do custo ótimo conhecido). A Figura~\ref{fig:ga2opt-flowchart} ilustra o fluxo completo do GA+$2$-opt memético [@larranaga1999genetic]: a busca local de $2$-opt (etapa destacada) é aplicada após os operadores genéticos (seleção por torneio, cruzamento OX, mutação swap), refinando cada descendente com até $n_{2\text{-opt}}$ varreduras completas de $2$-opt ($\mathcal{L}(x, n_{2\text{-opt}})$), cada uma delas escolhendo o melhor movimento da passagem, antes da formação da nova população. Esse acoplamento caracteriza o algoritmo como memético, combinando exploração global (GA) e intensificação local ($2$-opt).

```{.mermaid #fig:ga2opt-flowchart}
flowchart TD
    Start(["Início"]) --> Init["Gerar população inicial P(0)<br/>(qualquer tipo de heurística construtica)"]
    Init --> Eval0["Avaliar f(x_i) para cada<br/>indivíduo em P(0)"]
    Eval0 --> SetT["gen ← 0"]
    SetT --> CheckStop{"Critério de<br/>parada?"}
    CheckStop -->|Sim| Output(["Fim: Retorna melhor solução"])
    CheckStop -->|Não| Selection["Seleção por torneio<br/>(k_trnmt) de pais"]
    Selection --> Crossover["Cruzamento (OX)<br/>gerar descendentes"]
    Crossover --> Mutation["Mutação (swap)<br/>taxa p_mut"]
    Mutation --> LocalSearch["Busca Local: aplicar<br/>n_2opt varreduras best-improvement<br/>de 2-opt a cada descendente<br/>(ℒ(x, n_2opt))"]
    LocalSearch --> EvalNew["Avaliar f(x_i) dos<br/>descendentes refinados"]
    EvalNew --> Replace["Substituição: formar P(gen+1)<br/>com elitismo (melhores de P(gen)<br/>+ descendentes)"]
    Replace --> IncT["gen ← gen + 1"]
    IncT --> CheckStop
```

\standalonefiglabel{fig:ga2opt-flowchart}

Figura~\ref{fig:ga2opt-flowchart}: Fluxograma do algoritmo GA+$2$-opt.

No contexto deste trabalho, essa etapa de busca local será posteriormente mapeada para diferentes implementações em CPU e GPU (GA-CPU, GA-Híbrido-Ingênuo, GA-Híbrido-Otimizado, GA-FullGPU), mantendo a mesma lógica de aplicação de $2$-opt (mesma vizinhança, mesmos limites de iterações $n_{2\text{-opt}}$), de forma a preservar o caráter estruturalmente equivalente das variantes. Os detalhes operacionais, incluindo a forma precisa de parametrizar $n_{2\text{-opt}}$, $n_{pop}$, $p_{\text{mut}}$ e os critérios de parada do GA, são apresentados no [Capítulo 3](#cap3-materiais-metodos).

### 2.6 Computação em GPU e paralelização de meta-heurísticas {#sec-26-gpu-metaheuristicas}

Processadores gráficos (GPUs) evoluíram, nas últimas décadas, de dispositivos voltados principalmente para renderização gráfica para plataformas de computação de uso geral (GPGPU - General-purpose computing on GPUs), amplamente utilizadas em aplicações científicas e de inteligência artificial [@nvidia2024cuda]. O modelo de programação CUDA, por exemplo, organiza o trabalho em grades (*grids*) de blocos de threads, seguindo um paradigma de execução massivamente paralelo próximo ao SIMT (Single Instruction, Multiple Threads), relacionado às classificações de arquiteturas de Flynn [@flynn1972taxonomy]. Nessa configuração, milhares de threads executam o mesmo kernel sobre dados distintos, o que é adequado a tarefas com alto grau de paralelismo em dados.

Internamente, as threads em uma GPU são agrupadas em *warps* (tipicamente 32 threads em arquiteturas NVIDIA), que executam uma mesma instrução de forma coordenada segundo o modelo SIMT (*Single Instruction, Multiple Threads*) [@nvidia2024cuda]. Antes de comparar isso com CPUs, convém fixar a nomenclatura: CPUs modernas **não** são classificadas como SIMD; arquiteturalmente, elas continuam sendo MIMD (*Multiple Instruction, Multiple Data*), pois diferentes núcleos e *threads* podem seguir fluxos independentes. Quando se fala em “SIMD de CPU”, fala-se apenas de extensões vetoriais dentro de um núcleo. Já em SIMT, cada *thread* possui seu próprio estado lógico, mas o hardware agenda grupos de *threads* em passo coordenado enquanto não há divergência. Segundo a taxonomia de Flynn [@flynn1972taxonomy], conforme introduzido na [Seção 1.1](#sec-11-contexto-e-motivacao), CPUs modernas continuam sendo predominantemente MIMD com extensões SIMD [^cpu-simd-simt], enquanto GPUs maximizam o *throughput* com execução massivamente paralela orientada a dados, sendo ideais para cargas em que a mesma computação é aplicada a grandes volumes de dados independentes.

[^cpu-simd-simt]: A classificação arquitetural mais comum para CPUs multicore é MIMD, pois diferentes núcleos ou *threads* podem executar fluxos de instruções independentes. Instruções vetoriais como SSE e AVX adicionam paralelismo SIMD dentro de cada núcleo, acelerando operações sobre vetores de dados sem alterar a classificação geral da CPU.

#### 2.6.1 Taxonomia de paralelização segundo Crainic e Toulouse {#sec-261-taxonomia-paralelizacao}

A taxonomia proposta por @crainic2003parallel e aprofundada em @crainic2010parallel classifica as estratégias de paralelização de meta-heurísticas em três categorias principais, cada uma com adequação distinta a arquiteturas de hardware:

1. **Tipo 1: Paralelismo de Baixo Nível (Decomposição de Dados)**
   + **Descrição:** O foco é na paralelização de tarefas computacionalmente intensivas dentro de uma iteração da meta-heurística, como a avaliação da função objetivo ou a exploração de vizinhança. O fluxo de controle do algoritmo permanece centralizado.
   + **Adequação:** Ideal para arquiteturas **SIMD/SIMT (GPUs)**. Como a mesma operação (ex: cálculo de distância) é aplicada a muitos dados (ex: rotas, cálculos de matrizes), o *overhead* de inicializações de kernels e transferência é mínimo e o *throughput* é maximizado.
    + **Exemplo Real:** @tsp_gpu utilizam paralelismo Tipo 1 para avaliar movimentos $2$-opt em paralelo na GPU. Cada thread calcula o ganho de uma troca de arestas, permitindo explorar vizinhanças massivas em tempo reduzido.

2. **Tipo 2: Decomposição do Domínio**
   + **Descrição:** O espaço de busca ou as variáveis de decisão são particionados em subconjuntos (técnica comum em VRPs no geral), e a meta-heurística opera em paralelo sobre essas partições.
   + **Adequação:** Mais apropriado para arquiteturas **MIMD (Clusters ou CPUs Multicore)**. Subproblemas distintos podem ter tempos de convergência variados, o que causa divergência de execução em uma GPU (onde as 32 threads de cada warp devem ser sincronizadas).
   + **Exemplo Real:** Em problemas de Roteamento de Veículos com Múltiplos Depósitos (MDVRP), diversos autores [@crainic2010parallel, @simchi2005logic] descrevem estratégias onde cada "depósito" é otimizado por um processo independente em uma CPU, com trocas ocasionais de clientes entre depósitos.

3. **Tipo 3: Múltiplas Trajetórias (Buscas Independentes ou Cooperativas)**
    + **Descrição:** Múltiplas instâncias da meta-heurística (ou meta-heurísticas diferentes) exploram o espaço de soluções simultaneamente. Podem ser independentes (*multi-start*) ou cooperativas (troca de informações).
    + **Adequação:** Versátil. Funciona bem em:
      - **Clusters/CPUs** para algoritmos complexos e heterogêneos, através de técnicas de `multithreading` e `multiprocessing`, por exemplo.
      - **GPUs** para muitas execuções de algoritmos leves e idênticos, como *Island Models*.
            - **Combinados**: Processamento massivamente paralelo, conforme revisado por @schulz2013gpu.
        + **Exemplo Real:** @luong2013gpu implementam um modelo de ilhas em GPU, onde milhares de pequenas populações evoluem em paralelo (cada uma em um bloco de threads), trocando periodicamente seus melhores indivíduos via memória compartilhada.

De forma resumida, a Figura~\ref{fig:crainic-taxonomy} coloca lado a lado esses três tipos: a partir de uma meta-heurística sequencial, pode-se paralelizar apenas a avaliação de vizinhança/população (Tipo 1), decompor o conjunto de variáveis em subproblemas (Tipo 2) ou rodar várias buscas completas em paralelo, independentes ou cooperativas (Tipo 3).

```{.mermaid #fig:crainic-taxonomy}
flowchart TB
    MH["Meta-heurística sequencial"] --> T1["Tipo 1<br/>(paralelismo em dados)<br/>Avaliar vizinhança/população em paralelo"]
    MH --> T2["Tipo 2<br/>(decomposição do domínio)<br/>Subproblemas com subconjuntos de variáveis"]
    MH --> T3["Tipo 3<br/>(múltiplas trajetórias)<br/>Buscas completas em paralelo<br/>independentes ou cooperativas"]
```

\standalonefiglabel{fig:crainic-taxonomy}

Figura~\ref{fig:crainic-taxonomy}: Resumo dos três tipos de paralelização de Crainic e Toulouse.

No contexto deste trabalho, as variantes desenvolvidas enquadram-se predominantemente no **Tipo 1**, focando na aceleração da avaliação de vizinhança $2$-opt e operadores genéticos via paralelismo de dados massivo na GPU.

#### 2.6.2 Aplicações de GPU a meta-heurísticas para o TSP {#sec-262-gpu-aplicacoes-tsp}

Aplicações de GPU a meta-heurísticas para o TSP exploram principalmente paralelismo em dados, seja na avaliação em massa de rotas em algoritmos genéticos (por exemplo, via cálculos de redução em GPU), seja aplicando paralelismo a movimentos de vizinhança em heurísticas de melhoria [@schulz2013gpu; @tsp_gpu; @fujimoto2011highly]. Outros trabalhos investigam paralelização de Recozimento Simulado, Colônias de Formigas (ACO) e outras meta-heurísticas em GPU, geralmente aproveitando o grande número de threads para explorar múltiplas soluções ou vizinhanças em cada passo da busca [@binjubier2024gpu; @rey2018cpu; @abdelatti2020improvedgpuheuristic].
Esses estudos motivam o uso de GPUs como plataforma para acelerar algoritmos meméticos, mas também evidenciam desafios relacionados à movimentação de dados entre CPU e GPU, à escolha de granularidade adequada de kernels, à ocupação dos multiprocessadores e às limitações de memória (VRAM). Esses aspectos são retomados no [Capítulo 3](#cap3-materiais-metodos), ao descrever o desenho dos kernels de $2$-opt e as estratégias de controle de memória adotadas neste trabalho.

#### 2.6.3 Contraste arquitetural: CPU vs GPU {#sec-263-cpu-vs-gpu}

As figuras \ref{fig:cpu-gpu-contrast}, \ref{fig:cpu-gpu-detail} e \ref{fig:gpu-internal} ajudam a visualizar, de forma simples, como CPUs e GPUs foram pensadas para resolver problemas diferentes. De maneira geral, CPUs têm poucos núcleos mais complexos e flexíveis. Uma parte grande da área do chip é usada para controle de fluxo, previsão de desvios e vários níveis de cache, o que favorece programas sequenciais, com muitas decisões e acesso irregular à memória. Nas GPUs acontece o contrário: a maior parte da área é ocupada por muitas unidades aritméticas simples (ALUs), organizadas em centenas ou milhares de núcleos em paralelo. Elas abrem mão de um controle sofisticado para ganhar vazão (*throughput*) quando muitas threads executam a mesma sequência de instruções sobre dados diferentes. Essa diferença é o motivo pelo qual GPUs funcionam bem para tarefas com muito paralelismo em dados, como avaliar muitas rotas de uma vez ou aplicar $2$-opt em vários indivíduos em paralelo.

![Comparação simplificada entre arquitetura de CPU e GPU. Fonte: @layerstack2023cpugpu.](../../assets/image-1.png){#fig:cpu-gpu-contrast width=80%}

A Figura~\ref{fig:cpu-gpu-detail} mostra esse contraste na divisão da área do chip. Em uma CPU típica, poucos núcleos complexos dividem espaço com grandes caches e lógica de controle. Em uma GPU, o desenho é o inverso: a maior parte da área é dedicada a conjuntos de núcleos de processamento paralelos (por exemplo, *CUDA cores* em GPUs NVIDIA ou *stream processors* em GPUs AMD) e à memória de alta largura de banda. O controle é mais simples, mas o número de operações por segundo é muito maior quando o problema é bem mapeado para esse tipo de arquitetura [@paz2011gpucpuimage].

![Alocação de silício em CPU e GPU. Adaptado de @paz2011gpucpuimage.](../../assets/image-2.png){#fig:cpu-gpu-detail width=80%}

A Figura~\ref{fig:gpu-internal} resume como esses recursos aparecem na organização interna de uma GPU moderna. A memória global (DRAM) oferece grande largura de banda, mas latência alta; cada *streaming multiprocessor* (SM) possui memórias compartilhadas menores e registradores, usados pelas threads de um mesmo bloco. As threads são agrupadas em *warps* (tipicamente 32 threads nas GPUs NVIDIA) que executam a mesma instrução ao mesmo tempo, seguindo o modelo SIMT [@shah2023gpuarchitectureimage]. Quando as threads de um warp seguem caminhos de controle diferentes ou acessam a memória de forma muito irregular, parte dessa paralelização é perdida; quando o acesso é organizado e o fluxo de controle é parecido, o ganho de desempenho é grande.

![Organização interna de GPU moderna e modelo de programação CUDA. Adaptado de @shah2023gpuarchitectureimage.](../../assets/image-3.png){#fig:gpu-internal width=80%}

Essas três figuras servem como pano de fundo para as decisões de projeto dos kernels CUDA apresentados no Capítulo 3: explorar paralelismo em dados (tipo 1) mapeando threads para indivíduos ou movimentos $2$-opt, evitar ao máximo divergência dentro de um mesmo warp e organizar as leituras e gravações de memória global para aproveitar melhor a largura de banda disponível.

É fundamental distinguir, portanto, três níveis de paralelismo que aparecem neste trabalho. No nível da máquina, a CPU é tratada como uma arquitetura MIMD: diferentes núcleos podem executar tarefas distintas ou *threads* independentes. Dentro de cada núcleo, porém, a CPU pode explorar SIMD por meio de instruções vetoriais, o que acelera trechos específicos do cálculo de custos e de manipulação de vetores. Já a GPU, no contexto de CUDA, é programada no modelo SIMT: o programador escreve milhares de *threads* lógicas, mas o hardware as executa em grupos (*warps*) que avançam de forma coordenada enquanto compartilham o mesmo fluxo de instruções. Quando há divergência de controle dentro de um *warp*, os caminhos precisam ser serializados temporariamente, reduzindo a eficiência. Essa distinção explica por que a CPU pode ser descrita, sem contradição, como MIMD com extensões SIMD, enquanto a GPU é mais adequadamente descrita como SIMT.

### 2.7 Comparação estatística de algoritmos de otimização {#sec-27-comparacao-estatistica}

Meta-heurísticas estocásticas, como GAs e algoritmos meméticos, produzem resultados que variam de execução para execução devido ao uso de aleatoriedade em diversos pontos (inicialização, seleção, mutação, entre outros). Por esse motivo, a comparação de algoritmos de otimização não pode se basear em uma única execução por instância, tampouco apenas em médias simples sem análise de variabilidade. A literatura de comparação de algoritmos enfatiza a importância de utilizar múltiplas instâncias de teste, múltiplas repetições por combinação algoritmo–instância e métricas que considerem tanto qualidade da solução quanto tempo de execução [@demsar2006statistical].
<!-- Convenção de citação no Pandoc: use @chave quando a citação precisa funcionar como sujeito/autoria na frase; use [@chave] quando a referência deve aparecer apenas de forma parentética. -->

@demsar2006statistical discute procedimentos estatísticos apropriados para comparar algoritmos em vários problemas, recomendando o uso de testes de hipótese pareados (como o teste *t-pareado* ou o teste de Wilcoxon) quando se deseja comparar dois algoritmos em uma coleção de instâncias, e testes baseados em rankings, como o teste de Friedman seguido de pós-testes de Nemenyi, quando mais de dois algoritmos são avaliados simultaneamente. Medidas de tamanho de efeito, como o *d* de Cohen, complementam a análise ao quantificar a magnitude prática das diferenças observadas.

Neste trabalho, esses princípios gerais orientam o desenho experimental e a análise dos resultados apresentados nos capítulos seguintes, garantindo que as comparações entre variantes em CPU e GPU sejam realizadas de forma estatisticamente fundamentada. De maneira mais concreta, para cada combinação algoritmo–instância (4 algoritmos vs 38 instâncias), são executadas múltiplas repetições independentes, registrando-se, no mínimo, medidas de qualidade da solução (custo final ou *gap* em relação ao melhor valor conhecido) e de tempo de execução.

#### 2.7.1 Comparações pareadas {#sec-271-comparacoes-pareadas}

Para comparações **par a par** entre duas variantes (por exemplo, GA-CPU versus uma versão em GPU) ao longo de várias instâncias, adota-se uma rotina em duas etapas: (i) verificação de normalidade das distribuições de interesse por meio do teste de Shapiro–Wilk [@shapiro1965analysis] e (ii) aplicação de um teste pareado apropriado. Quando a hipótese de normalidade é considerada aceitável, utiliza-se o teste *t-pareado* de Student [@student1908probable]; caso contrário, recorre-se ao teste não paramétrico de Wilcoxon para amostras pareadas [@wilcoxon1945individual]. Em ambos os casos, medidas de tamanho de efeito, como o *d* de Cohen [@cohen1988statistical], são utilizadas para qualificar a relevância prática das diferenças detectadas. Valores típicos de *d* de Cohen são interpretados como: $|d| < 0.2$ (efeito negligenciável), $0.2 \le |d| < 0.5$ (efeito pequeno), $0.5 \le |d| < 0.8$ (efeito médio), e $|d| \ge 0.8$ (efeito grande). Estas comparações seguem as diretrizes de testes de "Statistical Comparisons of Classifiers
over Multiple Data Sets" [@demsar2006statistical], e as fórmulas são de fácil implementação utilizando bibliotecas em `python`.

#### 2.7.2 Comparações múltiplas {#sec-272-comparacoes-multiplas}

Quando três ou mais algoritmos são comparados simultaneamente em um conjunto de instâncias (como ocorre com as quatro variantes isoalgorítmicas GA-CPU, GA-Híbrido-Ingênuo, GA-Híbrido-Otimizado e GA-FullGPU), emprega-se o teste de Friedman [@friedman1937use] sobre os rankings médios dos algoritmos em cada problema, seguido de pós-testes de Nemenyi [@nemenyi1963distribution] quando apropriado, conforme recomendado por @demsar2006statistical. Esse procedimento permite identificar, com controle de erro tipo I, quais pares de algoritmos apresentam diferenças estatisticamente significativas em termos de desempenho médio. O teste de Friedman verifica a hipótese nula de que todos os algoritmos têm desempenho equivalente, enquanto o pós-teste de Nemenyi ajusta os valores-p para múltiplas comparações, reduzindo o risco de falsos positivos.

Os detalhes específicos de parametrização desses testes, bem como o conjunto exato de métricas analisadas (tempo, qualidade, ganho de performance, entre outras), são apresentados no [Capítulo 3](#cap3-materiais-metodos) ao descrever o protocolo experimental, e retomados no [Capítulo 4](#cap4-resultados) ao discutir os resultados obtidos.

# Capítulo 3 – Materiais e Métodos {#cap3-materiais-metodos}

Este capítulo descreve o ambiente computacional, o *framework* experimental e o protocolo de execução utilizados para comparar as quatro variantes do algoritmo memético GA+$2$-opt apresentadas na [Seção 1.4](#sec-14-objetivos-especificos). O objetivo é permitir que outros pesquisadores reproduzam, com o máximo de precisão possível, os experimentos discutidos no [Capítulo 4](#cap4-resultados), respeitando as limitações de hardware disponíveis.

## 3.1 Ambiente computacional e framework experimental {#sec-31-ambiente-framework}

Esta seção detalha as características do hardware e do software utilizados, bem como a arquitetura do framework experimental desenvolvido para garantir a integridade e a reprodutibilidade das comparações.

### 3.1.1 Hardware e sistema operacional {#sec-311-hardware-software}

Os experimentos foram conduzidos em um *notebook* equipado com processador **Intel Core i7-7700HQ** e uma GPU **NVIDIA GeForce GTX 1050 Mobile**. Embora o uso deste hardware tenha sido motivado inicialmente pela indisponibilidade de recursos mais robustos, ele se mostrou vantajoso metodologicamente: por se tratar de uma arquitetura de entrada (Pascal) com limitações estritas de memória e potência, as diferenças de eficiência entre as variantes algorítmicas tornam-se mais evidentes do que em aceleradores de alto desempenho, onde a força bruta computacional poderia mascarar ineficiências de implementação. Além disso, o contraste entre uma CPU relativamente capaz (i7 quad-core) e uma GPU de entrada acentua os desafios de *offloading*, exigindo otimizações reais para obter ganhos de performance. Ainda, há uma maior necessidade no controle de execução e limitações. Devido ao alto custo computacional para cálculos de algoritmos genéticos, instâncias maiores não foram testadas. Mas módulos e *safeguards* tiveram que ser implementados no teste de outras variações algorítmicas mais rápidas como no recozimento simulado [@wu2013performance] não discutidas neste trabalho, devido ao baixo grau de paralelismo do algoritmo para **variações do tipo 1** [@crainic2003parallel], obtidas experimentalmente.

<!-- código, no entanto, não está nada performático -->

As especificações relevantes da GPU para este estudo são:

O desempenho computacional é estritamente limitado pelas fronteiras físicas do dispositivo (*device*). Detalha-se a seguir o impacto matemático destas restrições no processamento:

1. **Memória de vídeo (VRAM) e o tamanho da instância:**
    A unidade de processamento gráfico dispõe de $4\text{GB}$ de memória global. Para maximizar a capacidade de leitura, exige-se que os dados sejam pré-alocados, o que obriga a utilização da matriz de distâncias completa $(n_{coords}^2)$, em detrimento de estruturas triangulares otimizadas [@harris2007optimizing].
    Ao adotar precisão dupla (`double` ou `float64`, 8 bytes) para mitigar erros numéricos, o consumo de memória da matriz de distâncias ($M_{dist}$) cresce quadraticamente:
    $$Mem(M_{dist}) = n_{coords}^2 \times 8 \text{ bytes}$$
    Para uma instância de $n_{coords}=15.112$ cidades, a matriz ocupa isoladamente $\approx 1,83 \text{GB}$. Somando-se as estruturas auxiliares, atinge-se o teto operacional com aproximadamente $n_{coords} \approx 13.000$ cidades. A utilização de precisão simples (`float32`) reduziria esta ocupação pela metade, permitindo a resolução de instâncias de maior porte.

2. **Gargalo de Transferência (*PCIe Transfer Bottlenecks*):**
    A comunicação entre o *host* (CPU) e o *device* (GPU) ocorre através de um barramento com largura de banda de $\approx 16 \text{GB/s}$, significativamente inferior à largura de banda interna da memória do dispositivo ($\approx 112 \text{GB/s}$). Transferências excessivas de dados penalizam o tempo total de execução ($T_{total}$), podendo anular o ganho de aceleração computacional ($T_{ganho}$), conforme a relação:
    $$
    T_{total} = T_{cpu} + T_{gpu} + T_{transf}
    $$
    Na variante **Híbrida-Otimizada**, este efeito é mitigado através do encadeamento de *kernels*, realizando as etapas intensivas (busca local e cálculo de aptidão) inteiramente no dispositivo e retornando ao hospedeiro apenas o vetor de custos, minimizando o termo $T_{transf}$.

3. **Ocupação e Mascaramento de Latência:**
    A arquitetura é composta por 5 Multiprocessadores de Streaming (*Streaming Multiprocessors - SMs*), cada um com 128 cores, totalizando 640 núcleos de execução. Para evitar a ociosidade destes núcleos durante os ciclos de latência de memória (centenas de ciclos de *clock*), emprega-se a técnica de *Latency Hiding*, mantendo múltiplas *threads* residentes prontas para execução [@kirk2010programming].

    + **Restrição Matemática:** Cada SM suporta um máximo de 2048 *threads*. Algoritmos com alta complexidade de registradores reduzem a capacidade de manter *threads* em espera, resultando em subutilização dos núcleos e perda de eficiência computacional, desperdiçando o potencial de processamento massivo da placa.

### 3.1.2 Ambiente de software {#sec-312-ambiente-software}

O framework foi desenvolvido em **python 3.10**, utilizando um conjunto de bibliotecas selecionadas para garantir a equivalência funcional entre as implementações em CPU e GPU:

- **NumPy**: Utilizado para a implementação de referência em CPU (GA-CPU).
- **CuPy**: Utilizado para as implementações aceleradas (Híbridas e FullGPU). A compatibilidade de API entre CuPy e NumPy foi fundamental para assegurar que a lógica dos algoritmos permanecesse idêntica ("isoalgorítmica"), alterando apenas o *backend* de execução.
- **SciPy e scikit-posthocs**: Empregados para a análise estatística dos resultados, incluindo testes de normalidade (Shapiro-Wilk) e testes não-paramétricos de comparação múltipla (Friedman e Nemenyi).

Apesar da compatibilidade da API oferecida pelo CuPy (frequentemente referenciada como `xp` em códigos agnósticos de backend), a implementação de operações críticas exigiu o desenvolvimento de **kernels CUDA customizados** (`RawKernel`). Abstrações de alto nível do CuPy, embora conveniente, introduzem *overheads* de lançamento de kernel que se tornam proibitivos quando realizados repetidamente para cada indivíduo da população (como na busca local 2-opt). A implementação manual de kernels permite:

1. **Fusão de operações**: Realizar múltiplas etapas do 2-opt em uma única chamada de kernel, reduzindo a latência de comunicação CPU-GPU.
2. **Gerenciamento fino de memória**: Otimizar o uso da memória compartilhada (*shared memory*) e dos registradores, maximizando a ocupação dos *SMs*.
3. **Paralelismo em lote**: Processar populações inteiras simultaneamente, explorando o paralelismo massivo da GPU. A API padrão do CuPy é excelente para operações "primitivas" de álgebra linear — como multiplicar grandes matrizes de números reais —, onde todos os elementos sofrem a mesma operação. Porém, o $2$-opt trabalha com uma matriz de permutações (índices inteiros) e exige que cada linha (indivíduo) siga um fluxo de execução próprio, em sua thread, com laços `while` e condições `if` independentes. Fazer isso via API padrão seria inviável; por isso, foi necessário desenvolver kernels CUDA customizados que possibilitam que cada thread da GPU gerencie sua própria lógica de busca local.

Além disso, todas as operações de ponto flutuante foram padronizadas em **precisão dupla (`float64`)**. Embora GPUs de consumo (como a GTX 1050 utilizada) tenham desempenho superior em precisão simples (`float32`), a precisão dupla foi mandatória para garantir a estabilidade numérica no cálculo de distâncias acumuladas em rotas longas (instâncias com milhares de cidades), onde erros de arredondamento poderiam levar a avaliações incorretas de melhoria (`delta < 0`) e divergência entre as implementações de CPU e GPU.

<!--TODO: Eu não tenho certeza sobre esta afirmação, acredito ser devido, primeiramente, à baixa tolerância da versào 1, onde o ótimo era considera $<1e^{-6}$. Mas já não se aplica pois mudei para `float32`. Os resultados obtidos, no entanto, estão como `float64` -->

### 3.1.3 Arquitetura do framework experimental {#sec-313-arquitetura-framework}

Para garantir comparações justas, o código foi estruturado em uma arquitetura modular que separa a lógica dos algoritmos da orquestração dos experimentos. O framework opera em três camadas distintas:

1. **Camada de Algoritmos:** Contém as implementações das quatro variantes do algoritmo memético. Todas compartilham a mesma estrutura de classes e os mesmos operadores genéticos, diferindo apenas na estratégia de gerenciamento de memória e paralelismo.
2. **Camada de Orquestração (Benchmark):** Responsável por carregar as configurações experimentais (parâmetros, instâncias), gerenciar a execução das repetições independentes e garantir o isolamento entre testes. Esta camada implementa mecanismos de *checkpoint* para salvar o estado de cada execução, permitindo a recuperação em caso de falhas e a auditoria posterior dos dados.
3. **Camada de Análise:** Scripts dedicados ao processamento dos arquivos de *checkpoint*, consolidação dos dados em banco de dados (DuckDB) e geração automática de tabelas e relatórios estatísticos.

```{.mermaid #fig:class-diagram}
classDiagram
    direction TB

    class Orchestration {
        +run_benchmark()
        +manage_checkpoints()
    }

    class GeneticAlgorithmBase {
        <<Abstract>>
        #population_size: int
        #mutation_rate: float
        #tournament_size: int
        #two_opt_iterations: int
        +evolve()
        #_initialize_population()
        #_select_parents()
        #_create_offspring()
        #_survival_selection()
        #_improve_population()*
        #_evaluate_population()*
    }

    class GeneticAlgorithmCPU {
        #_improve_population()
        #_evaluate_population()
    }

    class GeneticAlgorithmHybridNaive {
        #_improve_population()
        #_evaluate_population()
    }

    class GeneticAlgorithmHybridOptimized {
        #_improve_population()
        #_evaluate_population()
    }

    class GeneticAlgorithmFullGPU {
        #_improve_population()
        #_evaluate_population()
    }

    Orchestration --> GeneticAlgorithmBase : Instantiates & Runs

    GeneticAlgorithmBase <|-- GeneticAlgorithmCPU
    GeneticAlgorithmBase <|-- GeneticAlgorithmHybridNaive
    GeneticAlgorithmBase <|-- GeneticAlgorithmHybridOptimized
    GeneticAlgorithmBase <|-- GeneticAlgorithmFullGPU

    note for GeneticAlgorithmBase "Padrão Template Method:<br>Define fluxo fixo (evolve),<br>subclasses implementam backends<br>de execução específicos (*)"
```

\standalonefiglabel{fig:class-diagram}

Figura~\ref{fig:class-diagram}: Diagrama de classes do framework experimental.

<!-- Preciso consertar esta tabela -->

Esta separação assegura que as métricas de tempo e qualidade sejam coletadas de forma consistente para todas as variantes, eliminando vieses que poderiam surgir de implementações *ad hoc* para cada plataforma.

## 3.2 Arquitetura do framework e variantes algorítmicas {#sec-32-arquitetura-variantes}

A premissa central deste trabalho é a comparação **isoalgorítmica**: todas as variantes implementam exatamente a mesma meta-heurística, com os mesmos operadores e hiperparâmetros. As diferenças residem exclusivamente em *onde* (CPU ou GPU) e *como* (sequencial, paralelo por indivíduo ou paralelo em lote) as operações computacionalmente intensivas são executadas.

### 3.2.1 O Algoritmo Genético Base (GA + 2-opt) {#sec-321-ga-base}

O algoritmo base é um Algoritmo Genético (GA) geracional com elitismo, hibridizado com uma busca local $2$-opt truncada. Esta combinação, frequentemente denominada Algoritmo Memético, equilibra a exploração global do espaço de busca (via operadores genéticos) com a exploração local intensiva (via $2$-opt).

Os componentes e parâmetros fundamentais, mantidos constantes em todas as variantes, são:

- **Representação:** Permutação de inteiros representando a sequência de cidades visitadas.
- **População Inicial:** Gerada aleatoriamente. O tamanho da população ($n_{pop}$) é adaptativo, definido como $2 \times n_{coords}$, onde $n_{coords}$ é o número de cidades da instância.
- **Seleção:** Torneio (*Tournament Selection*), favorecendo indivíduos com menor custo (distância total).
- **Cruzamento (Crossover):** *Order Crossover* (OX), escolhido por preservar a ordem relativa das cidades e gerar permutações válidas.
- **Mutação:** *Swap Mutation*, que troca a posição de duas cidades aleatórias no cromossomo.
- **Busca Local (2-opt):** Aplicada a cada indivíduo da população ao final de cada geração. Para controlar o custo computacional, a busca local é **truncada**: limita-se a um número fixo de iterações de melhoria, entendido aqui como **10 varreduras completas da vizinhança por indivíduo**, em vez de buscar o ótimo local completo (2-opt).
- **Critérios de Parada:** O algoritmo encerra sua execução se atingir o ótimo conhecido (com tolerância de 1%), se não houver melhoria na melhor solução por um número de gerações definido pela "paciência" ($2 \times \sqrt{n_{coords}}$), ou se atingir o limite máximo de gerações ($2 \times n_{coords} \times \sqrt{n_{coords}}$).

### 3.2.2 Estratégias de Paralelização (As 4 Variantes) {#sec-322-variantes}

Para investigar o impacto da GPU, o algoritmo base foi instanciado em quatro variantes distintas, representando diferentes níveis de utilização do hardware:

1. **GeneticAlgorithmCPU (Baseline):**
    Execução inteiramente na CPU utilizando NumPy. No nível arquitetural, a CPU permanece sendo a plataforma MIMD de referência; o que ocorre aqui é o aproveitamento pontual de vetorização SIMD dentro dos núcleos para acelerar partes da avaliação dos movimentos 2-opt. Já o processamento dos indivíduos, no nível do algoritmo, ocorre essencialmente de forma sequencial. Esta variante serve como linha de base para medir o ganho de performance absoluto.

2. **GeneticAlgorithmHybridNaive (Híbrido Ingênuo):**
    Mantém a população e os operadores genéticos na CPU, mas transfere cada indivíduo para a GPU para executar a busca local 2-opt. Esta abordagem é considerada "ingênua" pois realiza transferências de memória (Host-to-Device e Device-to-Host) para *cada indivíduo* em *cada geração*, expondo o gargalo da largura de banda do barramento PCIe.

3. **GeneticAlgorithmHybridOptimized (Híbrido Otimizado):**
    Também mantém a lógica genética na CPU, mas otimiza a comunicação enviando a população inteira (ou grandes lotes) para a GPU de uma única vez. O kernel 2-opt processa todos os indivíduos em paralelo na GPU, e os resultados são retornados em lote. Esta estratégia visa amortizar o custo de latência das transferências de memória.

4. **GeneticAlgorithmFullGPU (Totalmente em GPU):**
    Implementa a estratégia de "GPU residente". A população é inicializada na VRAM e permanece lá durante todo o processo evolutivo. Todos os operadores (seleção, cruzamento, mutação e 2-opt) são executados via kernels CUDA (através do CuPy), eliminando quase completamente a necessidade de comunicação com a CPU, exceto para coleta de estatísticas e verificação de critérios de parada.

## 3.3 Seleção de instâncias e protocolo experimental {#sec-33-selecao-instancias}

Para garantir a relevância estatística dos resultados, foram selecionadas 38 instâncias da biblioteca **TSPLIB** [@reinelt1991tsplib], variando de 51 a 1002 cidades. O conjunto inclui problemas com diferentes características geométricas, permitindo avaliar a robustez dos algoritmos em diversos cenários. As instâncias selecionadas estão listadas na Tabela~\ref{tbl:selected-instances}, categorizadas por faixa de tamanho.

| Categoria | Faixa de Tamanho ($n_{coords}$) | Instâncias Selecionadas |
| :--- | :---: | :--- |
| **Pequenas** | $n_{coords} \le 100$ | eil51, berlin52, st70, eil76, pr76, rat99, kroA100, kroB100, kroC100, kroD100, kroE100, rd100 |
| **Médias** | $100 < n_{coords} \le 400$ | eil101, lin105, pr107, pr124, bier127, ch130, pr136, pr144, ch150, kroA150, kroB150, pr152, u159, rat195, d198, kroA200, ts225, pr264, pr299, lin318, rd400 |
| **Grandes** | $n_{coords} > 400$ | fl417, pr439, pcb442, rat783, pr1002 |

: Instâncias da TSPLIB selecionadas para o estudo \label{tbl:selected-instances}

O protocolo de execução seguiu as diretrizes para comparação de algoritmos estocásticos:

- **Repetições:** Cada par (algoritmo, instância) foi executado **26 vezes** independentemente.
- **Sementes Aleatórias:** As sementes foram fixadas e registradas para garantir a reprodutibilidade, mas variaram entre as 26 repetições para amostrar adequadamente o comportamento estocástico.
- **Limitação da CPU:** Devido ao tempo proibitivo de execução, a variante **GeneticAlgorithmCPU** foi executada apenas para instâncias com $N \le 100$. Para instâncias maiores, as comparações de ganho de performance tomam como base a variante híbrida mais simples ou comparam as variantes GPU entre si.

## 3.4 Protocolo de análise estatística {#sec-34-protocolo-estatistico}

A análise dos resultados adota a metodologia recomendada por @demsar2006statistical para comparação de múltiplos classificadores/algoritmos em múltiplos datasets:

1. **Verificação de Normalidade:** Aplicação do teste de **Shapiro-Wilk** nas distribuições de tempo e qualidade das soluções.
2. **Testes Pareados:** Para comparações diretas entre duas variantes (ex: HybridNaive vs. FullGPU), utiliza-se o **Teste t pareado** (se a distribuição for normal) ou o teste de **Wilcoxon Signed-Rank** (se não normal).
3. **Comparação Múltipla:** Para comparar as quatro variantes simultaneamente, aplica-se o teste de **Friedman** (não-paramétrico) para detectar se há diferenças significativas nos rankings dos algoritmos.
4. **Análise Post-Hoc:** Em caso de rejeição da hipótese nula no teste de Friedman, utiliza-se o teste de **Nemenyi** para identificar quais pares de algoritmos diferem significativamente entre si.
5. **Tamanho de Efeito:** O **d de Cohen** é calculado para quantificar a magnitude da diferença de desempenho, permitindo distinguir entre melhorias estatisticamente significativas mas irrelevantes na prática, e melhorias com impacto real no tempo de execução.

```{.mermaid #fig:stat-flowchart}
flowchart TD
    Start[("Início: Escolher Teste Estatístico")] --> DataType{Tipo de Dado?}

    DataType -->|Contínuo| SampleSize{Tamanho da Amostra?}
    DataType -->|Ordinal/Rankeado| NonParam[Testes Não-Paramétricos]:::nonparametric
    DataType -->|Categórico| ChiSquare[Testes Qui-Quadrado]:::other

    SampleSize -->|n < 20<br/>Amostra Pequena| SmallN[Obrigatório<br/>Teste de Normalidade]:::warning
    SampleSize -->|20 <= n < 30<br/>Zona Cinzenta| MediumN[Recomendado<br/>Teste de Normalidade]:::decision
    SampleSize -->|n >= 30<br/>Amostra Grande| LargeN[TCL Aplica-se<br/>Teste Recomendado]:::decision

    SmallN --> Normality{Shapiro-Wilk<br/>p >= 0.05?}:::decision
    MediumN --> Normality
    LargeN --> Normality

    Normality -->|Normal| Parametric[Testes Paramétricos]:::parametric
    Normality -->|Não-Normal| NonParam

    Parametric --> Groups{Número de Grupos?}:::decision
    Groups -->|1 grupo| OneSample[Teste t de uma amostra]:::parametric
    Groups -->|2 grupos| TwoSample[Pareado ou Independente?]:::decision
    Groups -->|>=3 grupos| ANOVA[ANOVA / Kruskal-Wallis]:::parametric

    NonParam --> GroupsNP{Número de Grupos?}:::decision
    GroupsNP -->|2 pareados| Wilcoxon[Wilcoxon Signed-Rank]:::nonparametric
    GroupsNP -->|2 independentes| MannWhitney[Mann-Whitney U]:::nonparametric
    GroupsNP -->|>=3 grupos| Friedman[Friedman / Kruskal-Wallis]:::nonparametric
```

\standalonefiglabel{fig:stat-flowchart}

Figura~\ref{fig:stat-flowchart}: Fluxograma de decisão para testes estatísticos.

```{.mermaid #fig:results-flowchart}
flowchart TD
    %% Nodes
    Start([Início: Análise de Resultados])

    subgraph S1 ["Estrato 1: Pequenos (n_coords < 100)"]
        S1_Test[Friedman Test]
        S1_Result{p = 0.086}
        S1_Conc[Inconclusivo<br/>Poder Estatístico Baixo]
    end

    subgraph S2 ["Estrato 2: Todos GPU-Only"]
        S2_Test[Friedman Test]
        S2_Result{p < 0.001}
        S2_Nemenyi[Nemenyi Post-Hoc]
        S2_Conc["FullGPU > Híbridos<br/>(Diferença Significativa)"]
    end

    subgraph Perf [Performance & Speedup]
        Speedup_HO[HybridOptimized: ~5.69×]
        Speedup_FG[FullGPU: ~1.2x]
        Tradeoff[Trade-off: HybridOptimized = Velocidade<br/>FullGPU = Qualidade]
    end

    %% Flow
    Start --> S1
    S1_Test --> S1_Result -->|Não Sig.| S1_Conc

    Start --> S2
    S2_Test --> S2_Result -->|Sig.| S2_Nemenyi --> S2_Conc

    S2_Conc -.-> Tradeoff
    Speedup_HO --> Tradeoff
    Speedup_FG --> Tradeoff
```

\standalonefiglabel{fig:results-flowchart}

Figura~\ref{fig:results-flowchart}: Fluxo de análise dos resultados.

# Capítulo 4 – Resultados {#cap4-resultados}

Este capítulo apresenta os resultados experimentais obtidos a partir da execução das quatro variantes do algoritmo memético GA+$2$-opt (CPU, Híbrido-Ingênuo, Híbrido-Otimizado e FullGPU) sobre o conjunto de 38 instâncias da TSPLIB. A análise é dividida em três partes principais:

1. uma avaliação detalhada por instância, destacando o comportamento para diferentes tamanhos de problema;
2. uma análise agregada, focada nas métricas de ganho de performance e qualidade média das soluções; e
3. a aplicação dos testes estatísticos descritos no Capítulo 3 para validar as diferenças de desempenho observadas.

## 4.1 Resultados Experimentais {#sec-41-resultados-experimentais}

Os experimentos foram conduzidos conforme o protocolo definido na Seção 3.3, com 26 repetições independentes para cada par algoritmo-instância. Para as instâncias pequenas ($n_{coords} \le 100$), todas as quatro variantes foram executadas. Para instâncias médias e grandes ($n_{coords} > 100$), a variante puramente em CPU foi omitida devido ao tempo de execução proibitivo, mantendo-se a comparação entre as três variantes aceleradas por GPU.

As métricas primárias coletadas foram:

- **Tempo Médio de Execução (s):** Tempo total para completar a execução por repetição.
- **Gap Médio (%):** Desvio percentual do custo da melhor solução encontrada em relação ao ótimo conhecido da instância.
- **Ganho de Performance:** Razão entre o tempo de execução da variante de referência (CPU ou Híbrido-Ingênuo) e a variante avaliada.
<!--TODO: adicionar as fórmulas estatísticas. -->

## 4.2 Análise por Instância {#sec-42-analise-por-instancia}

A Tabela~\ref{tbl:results-detailed} apresenta os resultados detalhados para uma seleção representativa de instâncias, cobrindo as três faixas de tamanho definidas.

| Instância | Tamanho ($n_{coords}$) | Algoritmo | Custo Médio (± σ) | Gap Médio (%) | Tempo Médio (s) | Ganho de Perf. |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **berlin52** | 52 | CPU | 7554,65 ± 22,93 | 0,17 | 25,82 | 1,00x |
| | | HybridNaive | 7547,31 ± 14,45 | 0,07 | 0,57 | 45,3x |
| | | HybridOptimized | 7557,85 ± 23,91 | 0,21 | 0,11 | 234,7x |
| | | FullGPU | 7542,00 ± 0,00 | 0,00 | 0,25 | 103,3x |
| **kroA100** | 100 | CPU | 21382,88 ± 62,04 | 0,47 | 185,08 | 1,00x |
| | | HybridNaive | 21382,38 ± 63,30 | 0,47 | 1,81 | 102,3x |
| | | HybridOptimized | 21364,23 ± 65,23 | 0,39 | 0,30 | 616,9x |
| | | FullGPU | 21285,23 ± 4,90 | 0,02 | 1,37 | 135,1x |
| **pr1002** | 1002 | HybridNaive | 279875,50 ± 3210,15 | 8,04 | 696,05 | 1,00x |
| | | HybridOptimized | 279184,62 ± 2789,63 | 7,77 | 374,50 | 1,86x |
| | | FullGPU | 269776,77 ± 743,93 | **4,14** | 758,40 | 0,92x |

: Resultados detalhados por instância (Seleção) \label{tbl:results-detailed}

Para instâncias pequenas como `berlin52` e `kroA100`, observa-se um ganho de performance massivo da variante **HybridOptimized** em relação à CPU (chegando a mais de 600x no caso de `kroA100`). Isso demonstra a eficiência da paralelização em lote do $2$-opt, que amortiza o custo de lançamento de kernels. A variante **FullGPU**, embora muito mais rápida que a CPU, apresenta um *overhead* inicial maior que a HybridOptimized para problemas muito pequenos, devido à gestão de memória residente.

Para instâncias grandes como `pr1002`, o cenário muda. A variante **HybridOptimized** mantém a liderança em tempo de execução (1.86x mais rápida que a Naive), mas a variante **FullGPU** destaca-se na **qualidade da solução**, obtendo um gap médio de 4,14% contra 7,77% da otimizada. Curiosamente, o tempo da FullGPU em instâncias muito grandes aproxima-se ou até excede o da variante Naive em alguns casos, sugerindo que a complexidade de manter todo o estado na GPU (e a possível saturação de recursos do dispositivo) começa a impactar o desempenho bruto, embora favoreça a convergência para melhores ótimos locais.

## 4.3 Análise Agregada e Trade-offs {#sec-43-analise-agregada}

Ao agregar os resultados de todas as 38 instâncias, emergem padrões claros de desempenho que caracterizam o *trade-off* entre velocidade e qualidade.

| Algoritmo | Tempo Médio ± σ (s) | Gap Médio ± σ (%) | Ganho de Perf. Mediano |
| :--- | :---: | :---: | :---: |
| **HybridNaive** | 47,08 ± 130,25 | 1,28 ± 1,68 | 1,00x |
| **HybridOptimized** | 18,94 ± 67,33 | 1,30 ± 1,78 | **5,69x** |
| **FullGPU** | 43,51 ± 132,84 | **0,88 ± 1,14** | 1,21x |

: Médias Agregadas (Todas as Instâncias) \label{tbl:results-aggregated}

>[!warning] inconsistência tenebrosa
> Nos descios padrão

A análise da Tabela~\ref{tbl:results-aggregated} revela duas conclusões fundamentais:

1. **HybridOptimized é a campeã de velocidade:** Com um ganho de performance mediano por instância de **5,69x** sobre a versão ingênua, esta variante comprova que a estratégia de processamento em lote (*batch processing*) é a mais eficaz para reduzir o tempo total de execução. Ao enviar a população inteira para a GPU de uma só vez, minimiza-se a latência de comunicação PCIe, gargalo crítico em arquiteturas híbridas.
2. **FullGPU é a campeã de qualidade:** Com um gap médio de **0,88% (± 1,14)**, a versão totalmente residente em GPU supera as variantes híbridas (1,28% e 1,30%, respectivamente). A hipótese é que a permanência dos dados na GPU permite uma execução mais fluida e, possivelmente, uma exploração mais eficiente da vizinhança $2$-opt sem as interrupções de transferência, favorecendo a descoberta de melhores ótimos locais.

Para validar estatisticamente as diferenças de qualidade (Gap) entre as variantes implementadas em GPU (para todas as 38 instâncias), aplicou-se o teste não-paramétrico de Friedman. O teste rejeitou a hipótese nula com alta confiança ($p \approx 6,9 \times 10^{-5}$), confirmando que há diferença global significativa no ranqueamento de desempenho. Em seguida, o pós-teste de Nemenyi determinou pareadamente quais diferenças são reais: a variante FullGPU mostrou-se estatisticamente superior à HybridNaive ($p \approx 0,00028$) e à HybridOptimized ($p \approx 0,00071$). Contudo, não houve diferença estatística significativa entre as duas versões híbridas ($p \approx 0,97$). Isso atesta que o incremento de qualidade da abordagem puramente residente não é estocástico, mas sim estrutural.

## 4.4 Melhor Algoritmo por Categoria de Tamanho {#sec-44-melhor-algoritmo}

A Tabela~\ref{tbl:best-algorithm} segmenta o desempenho "vencedor" por faixa de tamanho da instância, considerando o critério de menor *gap* (qualidade).

| Categoria | Melhor Algoritmo | Gap Médio |
| :--- | :--- | :---: |
| **Pequeno ($n_{coords} \le 100$)** | **FullGPU** | 0,23% |
| **Médio ($100 < n_{coords} \le 400$)** | **FullGPU** | 0,79% |
| **Grande ($n_{coords} > 400$)** | **FullGPU** | 2,83% |

: Melhor Algoritmo (Menor Gap) por Tamanho \label{tbl:best-algorithm}

A dominância da variante **FullGPU** em todas as categorias de tamanho reforça a observação de que a arquitetura residente é superior para a qualidade da solução final. Mesmo que a variante **HybridOptimized** seja mais rápida, ela tende a ficar presa em ótimos locais ligeiramente piores com mais frequência. Isso sugere que, para aplicações onde a qualidade da rota é crítica e o tempo de resposta é secundário (mas ainda deve ser razoável), a abordagem FullGPU é a recomendada. Para cenários de tempo real ou onde o *throughput* é prioritário, a HybridOptimized é a escolha ideal.

## 4.5 Discussão Geral {#sec-45-discussao-geral}

Os resultados validam a hipótese de que o uso de GPUs pode acelerar significativamente meta-heurísticas para o TSP, mas ressaltam que "usar GPU" não é uma solução única. A forma como a paralelização é implementada define drasticamente o perfil de desempenho:

- A abordagem **Ingênua (Naive)**, embora funcional, sofre severamente com o gargalo de comunicação, servindo mais como um ponto de partida pedagógico do que como solução de produção.
- A abordagem **Otimizada (Batch)** resolve o problema de comunicação e entrega a maior velocidade bruta, sendo ideal para iterar rapidamente.
- A abordagem **Residente (FullGPU)** oferece o melhor refinamento das soluções. O fato de obter menores gaps sugere que a ausência de overhead de comunicação permite que o algoritmo "gaste" seu tempo de forma mais produtiva na exploração efetiva do espaço de busca.

A limitação da CPU para instâncias $N > 100$ (onde os tempos excederiam horas ou dias para as 26 repetições) confirma a necessidade imperativa de aceleração de hardware para problemas de otimização combinatória de médio e grande porte.
