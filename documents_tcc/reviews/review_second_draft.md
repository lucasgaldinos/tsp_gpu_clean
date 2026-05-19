---
title: refactor discussion
description: >
  `second_draft.md` discussion
created: 2026-05-04
modifications:
  - date_modified: 2026-05-05
    modifications:
      - description: >
          Added `# part 1` and `# part 2` which are reviews from gemini
      - description: >
          Added review from Stanford agentic reviewer
related_files:
  - [second_draft.md](../second_draft-draft.md)
  - [review Stanford agentic reviewer](../review-stanford-reviewer-second_draft.md)
tags: [] #list of relevant tags
---

# Revisão Acadêmica — `second_draft.md`

> **Data:** 2026-05-04
> **Documento analisado:** `documents_tcc/second_draft.md`
> **Tipo de trabalho:** TCC de Ciência da Computação (ABNT)
> **Escopo do benchmark:** 4 variantes isoalgorítmicas GA+2-opt para TSP · 38 instâncias TSPLIB · 26 repetições

## Other analysis

### Part 1

````md
### 1 Análise Metodológica e Contradições Lógicas

- **A falácia da equivalência isoalgorítmica**: O texto define as variantes do algoritmo como isoalgorítmicas, garantindo que elas mantêm exatamente a mesma lógica e parâmetros operacionais[cite: 1]. Além disso, estabelece explicitamente que as sementes aleatórias foram fixadas para garantir a reprodutibilidade dos ensaios[cite: 1]. Contudo, a Tabela 3 evidencia _gaps_ médios distintos entre a versão residente na placa de vídeo (0.88%) e as versões híbridas (1.30%)[cite: 1]. Se os algoritmos são de fato equivalentes e inicializados sob as mesmas condições determinísticas, a trajetória de busca evolutiva e as métricas de qualidade final devem ser estritamente idênticas. Esta assimetria indica falhas arquiteturais severas na implementação paralela, como potenciais condições de corrida (_race conditions_) nos _kernels_ CUDA, dessincronização de estado do gerador de números pseudoaleatórios (PRNG) entre _host_ e _device_, ou divergências na associatividade de reduções de ponto flutuante.
- **Incompatibilidade com o padrão de distâncias TSPLIB**: A justificativa para o uso de precisão dupla (`float64`) baseia-se na prevenção de instabilidades numéricas no acúmulo de distâncias de rotas longas[cite: 1]. No entanto, instâncias baseadas em coordenadas no plano (especificação `EUC_2D`) na biblioteca TSPLIB exigem contratualmente que os cálculos de distância euclidiana sejam arredondados para o número inteiro mais próximo, prescindindo do uso de precisão flutuante. A aplicação de cálculos em ponto flutuante distorce o cômputo da função objetivo frente ao referencial da biblioteca, invalidando comparações diretas de distância e qualidade com a literatura previamente estabelecida.

### 2 Inexatidões Conceituais e Teóricas

- **Complexidade do algoritmo Concorde**: A descrição teórica aponta a complexidade temporal do algoritmo exato Concorde como "indefinida"[cite: 1]. O Concorde emprega o método de _Branch-and-Cut_ para a resolução de modelos de Programação Linear Inteira[cite: 1]. Sua complexidade de pior caso é inerentemente exponencial, de acordo com o limite de tratabilidade de problemas NP-difíceis, não existindo suporte na teoria da complexidade para a classificação "indefinida".
- **Dicotomia simplista na taxonomia computacional**: O embasamento teórico classifica CPUs contemporâneas como arquiteturas MIMD operando com extensões SIMD e processadores gráficos exclusivamente sob o paradigma SIMT[cite: 1]. Embora conceitualmente aceitável para alto nível, o texto falha ao apontar em seções subsequentes que as _threads_ na GPU "podem, teoricamente, seguir fluxos distintos (divergência)" e que esta é uma vantagem flexível em algoritmos complexos[cite: 1]. A divergência de execução em um _warp_ exige a serialização obrigatória das instruções na arquitetura SIMT, causando degradação linear de desempenho para fluxos condicionais independentes, caracterizando-se como uma limitação severa, e não uma flexibilidade arquitetural.
- **Mascaramento de latência e limitação de núcleos**: A análise de restrições matemáticas conclui que a alta exigência de registradores diminui a ocupação dos multiprocessadores, o que inevitavelmente resulta em subutilização e perda de eficiência[cite: 1]. Esta correlação omite um princípio fundamental de arquitetura de _hardware_: o Paralelismo em Nível de Instrução (ILP). A execução paralela de instruções independentes dentro da mesma _thread_ pode mascarar as latências de acesso à memória global e maximizar a vazão (_throughput_), mesmo em estados de ocupação de _warps_ deliberadamente baixos.

### 3 Omissões e Ambiguidades nos Resultados

- **Ausência de validação quantitativa**: A metodologia define um rigoroso protocolo de hipóteses contendo testes de normalidade de Shapiro-Wilk, verificações de _t-pareado_, análises não-paramétricas de Wilcoxon e Friedman, pós-testes de Nemenyi, e medição de relevância pelo _d_ de Cohen[cite: 1]. Todavia, as Tabelas 1, 2 e 3 limitam-se a expor médias aritméticas de velocidade e custo[cite: 1]. O capítulo de resultados suprime integralmente todos os valores-p, métricas de correlação, tamanhos de efeito e intervalos de confiança. Sem a publicização das estatísticas de teste, as alegações inferenciais sobre ganhos de _performance_ não possuem lastro científico comprovável a partir dos dados apresentados.
- **Justificativa temporal infundada para a variante residente**: É postulado que a versão _FullGPU_ consome mais tempo de processamento devido ao "_overhead_ inicial" e à "complexidade de manter todo o estado na GPU" para problemas de grande escala[cite: 1]. A alocação estática de memória na inicialização não afeta as avaliações no laço principal de gerações. O gargalo reportado é mais provável de ser um sintoma de saturação dos registradores locais e do limite da taxa de acerto no acesso concorrente à memória do dispositivo, fenômenos negligenciados na discussão.
````

### Part 2

````md
### Avaliação do Manuscrito via Pipeline Paper Reviewer

Abaixo está a execução do fluxo de trabalho do orquestrador de revisão para o manuscrito referenciado, segmentada pelas oito fases operacionais[cite: 1, 2].

#### Fase 1: Triagem
- **Classificação do Estudo**: Estudo observacional focado em _benchmark_ de otimização combinatória e paralelização computacional[cite: 1, 2].
- **Requisitos de Periódico**: O manuscrito não especifica um periódico alvo, mas a estrutura e a taxonomia referenciam literaturas padrão de Pesquisa Operacional e Computação de Alto Desempenho[cite: 1].
- **Arquivos Disponíveis**: Apenas `second_draft.md` está disponível[cite: 1]. O arquivo de bibliografia `refs.bib` está ausente, limitando as validações da Fase 7[cite: 1].

#### Fase 2: Detecção de IA
- **Análise Heurística**: A execução da habilidade `ai-check` depende de ferramentas externas de inferência[cite: 2]. O texto exibe alta densidade de formatação estruturada e transições sintáticas padronizadas no Capítulo 1 e Capítulo 2, requerendo análise dedicada para quantificar se a presença de padrões gerados por LLM ultrapassa o limite padrão de 30% estabelecido pela configuração[cite: 1, 2].

#### Fase 3: Metodologia
- **Desenho do Estudo**: A proposta metodológica baseia-se na implementação de quatro variantes definidas como "isoalgorítmicas" (GA-CPU, GA-Híbrido-Ingênuo, GA-Híbrido-Otimizado, GA-FullGPU)[cite: 1].
- **Aderência ao Protocolo**: Há uma contradição fatal na execução do protocolo. O manuscrito garante que o algoritmo compartilha a mesma estrutura, parâmetros e sementes aleatórias fixadas por repetição[cite: 1]. Consequentemente, as variantes deveriam produzir exatamente a mesma trajetória evolutiva e resultados de custo idênticos. No entanto, as tabelas de resultados relatam que a variante _FullGPU_ obteve um _gap_ médio de 0.88%, enquanto as versões híbridas obtiveram 1.30%[cite: 1]. Isso indica uma falha na aderência metodológica, evidenciando que as variantes não são algoritmicamente idênticas ou sofrem de dessincronização de estado estocástico/numérico[cite: 1].

#### Fase 4: Risco de Viés
- **Viés de Seleção e Hardware**: A avaliação de viés utilizando _frameworks_ similares ao ROBINS-I adaptados para estudos computacionais revela limitações severas[cite: 2]. O uso de um equipamento restrito (GTX 1050 com 4GB de VRAM e processador i7-7700HQ de 2017) introduz viés de _hardware_, mascarando ganhos reais que arquiteturas modernas SIMT poderiam oferecer[cite: 1].
- **Viés de Linha de Base**: A variante principal de controle (GA-CPU) foi deliberadamente omitida dos testes para instâncias com $n_{coords} > 100$ cidades devido a limitações de tempo[cite: 1]. Isso impede a mensuração correta do "Ganho de Performance" absoluto em instâncias médias e grandes, distorcendo a análise comparativa agregada[cite: 1].

#### Fase 5: Validação Estatística
- **Teste de Suposições e Efeitos**: O protocolo experimental detalha explicitamente o uso de testes rigorosos, incluindo normalidade (Shapiro-Wilk), comparações de hipótese (teste t-pareado, Wilcoxon, Friedman, Nemenyi) e tamanho de efeito (_d_ de Cohen)[cite: 1].
- **Invalidação**: Nenhuma dessas métricas foi reportada no Capítulo 4[cite: 1]. As Tabelas 1, 2 e 3 limitam-se a relatar o "Tempo Médio", "Gap Médio" e "Ganho de Performance" (médias aritméticas puras)[cite: 1]. Sem a apresentação dos valores-p ou intervalos de confiança, as alegações de "diferenças significativas" presentes nos diagramas de fluxo e no texto carecem de fundamentação quantitativa, falhando no estágio de validação[cite: 1, 2].

#### Fase 6: Conformidade de Relato
- **Diretrizes**: A avaliação de conformidade (utilizando adaptações de STROBE para modelagem computacional) indica baixa aderência[cite: 2]. A ausência de apresentação dos dados estatísticos descritos na metodologia derruba o índice de conformidade para níveis inaceitáveis de publicação (abaixo do limiar de 85%)[cite: 1, 2].
- **Itens Faltantes**: Requer a inserção imediata de matrizes de valores-p, listagem de desvios-padrão nas tabelas de desempenho e os resultados formais dos pós-testes de Nemenyi referenciados[cite: 1, 2].

#### Fase 7: Auditoria de Citações
- **Verificação de Referências**: O manuscrito baseia-se fortemente em literaturas clássicas estabelecidas (e.g., Crainic para taxonomia paralela, Demšar para testes de hipóteses estatísticas, Cook para TSP)[cite: 1].
- **Verificação de DOIs**: Inconclusivo. A ausência do anexo `./refs.bib` bloqueia a verificação automatizada de DOIs e escaneamento contra bancos de dados de retrações[cite: 1, 2].

#### Fase 8: Reprodutibilidade
- **Acessibilidade de Código e Dados**: O texto menciona uma arquitetura modular em python e banco de dados DuckDB, mas falha em fornecer enlaces ou declarações de disponibilidade de dados (_Data Availability Statements_) apontando para repositórios públicos (e.g., GitHub, Zenodo)[cite: 1, 2]. Sem o fornecimento do código-fonte, a reprodutibilidade da simulação e a verificação dos _kernels_ CUDA são impossíveis[cite: 1, 2].
````

### Revisor agêntico de Stanford

>[!important] Documento que contém falhas, questões e análises
>Usando o revisor agentivo de Stanford, a revisão abaixo foi gerada e deve ser lida:
>
>- [Revisor agentivo de Stanford](../review-stanford-reviewer-second_draft.md)

---

## Sumário Executivo

| Severidade | Qtd | Descrição rápida |
| :--- | :---: | :--- |
| 🔴 **Crítico** | 4 | Impedem a validade científica do trabalho |
| 🟠 **Alto** | 5 | Devem ser corrigidos antes da entrega |
| 🟡 **Moderado** | 6 | Melhoram rigor e clareza |
| 🟢 **Menor** | 4 | Melhorias opcionais de precisão |

**Avaliação geral:** O trabalho apresenta base metodológica sólida (taxonomia Crainic correta, uso de Demšar 2006, hardware bem caracterizado, 38 instâncias, 26 repetições). Contudo, há quatro problemas críticos que comprometem a validade das conclusões: ausência dos Capítulos 5 e 6, supressão completa dos resultados dos testes estatísticos no Capítulo 4, inconsistência entre a premissa isoalgorítmica e gaps de qualidade divergentes, e uso de distâncias em ponto flutuante quando a especificação TSPLIB exige arredondamento inteiro.

---

## 1. Triagem do Manuscrito

### 1.1 Tipo e status

- **Tipo:** TCC experimental de ciências da computação — estudo comparativo de benchmark algorítmico
- **Capítulos presentes:** 1 (Introdução), 2 (Revisão Bibliográfica), 3 (Materiais e Métodos), 4 (Resultados)
- **Capítulos prometidos e ausentes:** **5 (Discussão)** e **6 (Conclusões e Trabalhos Futuros)** — anunciados em §1.6, mas não escritos

### 1.2 Afirmações centrais verificadas aritmeticamente

| Afirmação | Fonte | Verificação |
| :--- | :--- | :--- |
| berlin52: HybridOptimized = 234.7× vs CPU | Tabela 1 | 25.82 / 0.11 = **234.7×** ✓ |
| kroA100: HybridOptimized = 616.9× vs CPU | Tabela 1 | 185.08 / 0.30 = **616.9×** ✓ |
| pr1002: FullGPU speedup = 0.92× vs Naive | Tabela 1 | 696.05 / 758.40 = **0.92×** ✓ |
| HybridOptimized: ganho mediano = 5.69× | Tabela 2 | Plausível (mediana de razões por instância) ✓ |
| FullGPU gap médio = 0.88% | Tabela 2 | Compatível com dados de `tmp/stats_output/` ✓ |

---

## 2. Problemas Críticos 🔴

### C1 — Capítulos 5 e 6 Ausentes

**Localização:** §1.6 (promete capítulos), documento termina no Capítulo 4.

**Descrição:** A seção de organização anuncia explicitamente:

- *"Capítulo 5 – Discussão: interpreta criticamente os resultados à luz da literatura..."*
- *"Capítulo 6 – Conclusões e Trabalhos Futuros: sintetiza as principais contribuições..."*

Ambos estão ausentes. Para um TCC no padrão ABNT, esses capítulos são obrigatórios. Sem eles, o trabalho:

- Não responde explicitamente aos objetivos propostos em §1.3 e §1.4
- Não situa os resultados em relação à literatura citada (Fujimoto, Schulz, Rocki)
- Não apresenta a seção de limitações do estudo
- Não propõe trabalhos futuros

**Ação:** Escrever os Capítulos 5 e 6 completos antes da submissão.

---

### C2 — Ausência Total de Resultados dos Testes Estatísticos no Capítulo 4

**Localização:** §3.4 (metodologia prometida) vs. Capítulo 4 (resultados entregues).

**Descrição:** O protocolo estatístico em §3.4 compromete:

> *"Aplicação do teste de Shapiro-Wilk... Testes pareados... Friedman... Nemenyi... d de Cohen..."*

O Capítulo 4 apresenta **exclusivamente médias aritméticas** nas Tabelas 1, 2 e 3. Estão inteiramente ausentes:

- Valores-p de qualquer teste (Shapiro-Wilk, Wilcoxon, Friedman)
- Estatísticas de efeito (d de Cohen)
- Intervalos de confiança
- Desvios-padrão ou IQR
- Marcadores de significância nas tabelas

**Detalhe adicional:** O fluxograma da Figura \ref{fig:results-flowchart} (Mermaid) exibe internamente os valores `p = 0.086` (Estrato 1) e `p < 0.001` (Estrato 2), mas esses valores **não aparecem em nenhum parágrafo do texto**. Os dados existem em `tmp/stats_output/cross_problem_stats.json` mas não foram incorporados ao documento.

**Valores a reportar** (de `tmp/stats_output/`):

| Teste | Resultado |
| :--- | :--- |
| Friedman – Estrato 1 (n_coords ≤ 100, 4 algos) | p = 0,086 (não significativo a α=0,05) |
| Friedman – Estrato 2 (todos 38 inst., GPU-only) | p = 6,9×10⁻⁵ (significativo) |
| Nemenyi: FullGPU vs HybridNaive | p = 0,000283 (sig.) |
| Nemenyi: FullGPU vs HybridOptimized | p = 0,000708 (sig.) |
| Nemenyi: HybridNaive vs HybridOptimized | p = 0,97 (NÃO sig.) |

Sem esses dados no texto, todas as afirmações de superioridade estatística da FullGPU carecem de respaldo científico verificável.

**Ação:** Incorporar tabela de resultados estatísticos com p-valores, Friedman statistic, diferença crítica de Nemenyi e d de Cohen. Discutir a implicação do p=0,086 no Estrato 1 (baixo poder estatístico por n=6 instâncias pequenas).

---

### C3 — Falácia da Equivalência Isoalgorítmica

**Localização:** §1.2, §1.3, §3.2, Tabelas 1 e 2.

**Descrição:** O texto garante que as variantes implementam *"exatamente a mesma lógica algorítmica"* com sementes fixadas para reprodutibilidade (§3.3: *"As sementes foram fixadas e registradas para garantir a reprodutibilidade"*). Contudo, os gaps médios diferem substancialmente:

| Variante | Gap Médio |
| :--- | :---: |
| HybridNaive | 1,28% |
| HybridOptimized | 1,30% |
| **FullGPU** | **0,88%** |

Se algoritmos verdadeiramente isoalgorítmicos com sementes idênticas fossem executados, as trajetórias de busca seriam determinísticas e idênticas — os gaps deveriam ser iguais. A diferença de 0,42 pontos percentuais (FullGPU vs híbridas) indica uma divergência de implementação real, cujas causas prováveis incluem:

1. **PRNG não sincronizado entre host e device:** A geração de números aleatórios na GPU (cuRAND/CuPy interno) pode usar algoritmos ou seeds de forma diferente da CPU (NumPy). Isso altera as trajetórias de evolução.
2. **Não-associatividade de ponto flutuante em reduções:** Reduções paralelas em GPU (e.g., cálculo de fitness) podem produzir resultados ligeiramente diferentes por reordenação de operações de ponto flutuante, mesmo com os mesmos dados de entrada.
3. **Divergência no operador de cruzamento OX em GPU:** A implementação GPU do OX pode selecionar segmentos ou preencher posições em ordem diferente da versão CPU.

A nota de rodapé §1.5 (*"Algumas variações dentro deste contexto são esperadas"*) é insuficiente — não explica o mecanismo e não quantifica o impacto.

**Ação:** Adicionar uma subseção dedicada (em §3.2 ou §5) explicando: (a) por que as sementes fixas não garantem trajetórias idênticas entre CPU e GPU; (b) quais mecanismos concretos causam divergência (PRNG, associatividade de FP, ordenação de reduções); (c) qualificar que "isoalgorítmico" refere-se à equivalência *estrutural e paramétrica*, não determinística.

---

### C4 — Incompatibilidade com o Padrão de Distâncias TSPLIB (EUC_2D)

**Localização:** §3.1.1 (justificativa do float64) e §3.1.2 (TODO interno).

**Descrição:** A especificação oficial da TSPLIB para instâncias `EUC_2D` (Reinelt 1991) define:

```
d(i,j) = nint(sqrt((xi - xj)^2 + (yi - yj)^2))
```

onde `nint()` arredonda ao inteiro mais próximo. Os custos ótimos publicados na TSPLIB são calculados com essa métrica inteira.

O trabalho usa `float64` em todo o cálculo de distâncias (§3.1.1: *"precisão dupla (float64) foi mandatória"*). Consequências:

1. **Os gaps reportados são calculados contra ótimos inteiros usando custos float**: a distância euclidiana contínua (`sqrt(...)`) é geralmente *menor* que `nint(sqrt(...))`, podendo deflacionar artificialmente os gaps reportados ou torná-los não comparáveis com a literatura.
2. **Comparações diretas com resultados de outros papers TSP são inválidas**: a maioria dos estudos com TSPLIB usa distâncias inteiras conforme a especificação.
3. **A justificativa está incorreta**: aritmética inteira é exata por definição (não há erro de arredondamento acumulado). O problema de "instabilidade numérica no acúmulo de distâncias longas" mencionado não se aplica a inteiros.

O comentário interno `<!--TODO: Eu não tenho certeza sobre esta afirmação...-->` (§3.1.2) confirma a incerteza do próprio autor sobre essa escolha.

**Ação:**

- **Opção preferencial:** Implementar `nint(sqrt(...))` para cálculo de distâncias, realinhando com TSPLIB. Reexecutar benchmarks (ou justificar por que os resultados existentes são suficientemente representativos como aproximação).
- **Alternativa:** Explicitar que o trabalho usa distâncias euclidianas contínuas (não-TSPLIB) e que os ótimos de referência foram recalculados com a mesma métrica — neste caso, os gaps são internamente consistentes mas não comparáveis com literatura.
- Remover o comentário `<!--TODO-->` da versão final.

---

## 3. Problemas de Alta Prioridade 🟠

### H1 — Complexidade do Concorde Descrita como "Indefinida"

**Localização:** §2.1, primeiro parágrafo + nota de rodapé `[^held-karp]`.

**Texto atual:**
> *"o algoritmo Concorde [@cook2012pursuit] com uma complexidade temporal indefinida, conseguem resolver instâncias com milhares de cidades, mas seu tempo de execução cresce exponencialmente ($O(n^2 2^n)$)"*

**Problemas:**

1. A complexidade do Concorde é **exponencial no pior caso** (TSP é NP-difícil; nenhum algoritmo exato pode ter garantia polinomial salvo P=NP), não "indefinida". "Indefinida" não é um conceito da teoria da complexidade.
2. O bound $O(n^2 2^n)$ pertence ao **Held-Karp**, não ao Concorde. O Concorde usa Branch-and-Cut sobre relaxações LP e não tem esse bound — seu desempenho empírico é muito melhor que Held-Karp para grandes instâncias.
3. A frase mistura os dois algoritmos de forma enganosa.

**Correção sugerida:**
> *"O algoritmo Held-Karp [@held1962dynamic] tem complexidade $O(n^2 2^n)$, tornando-o impraticável para $n > 40$. O Concorde [@cook2012pursuit], baseado em Branch-and-Cut sobre relaxações de programação linear inteira, apresenta complexidade exponencial no pior caso (limite intrínseco para problemas NP-difíceis), mas resolve instâncias com até 85.900 cidades na prática, graças a limitantes inferiores muito ajustados."*

---

### H2 — Desvios-Padrão Ausentes das Tabelas de Resultados

**Localização:** Tabelas 1, 2 e 3 (Capítulo 4).

**Descrição:** Com 26 repetições independentes por par (algoritmo, instância), a variabilidade é informação essencial. Apenas médias não permitem:

- Avaliar consistência de desempenho entre runs
- Verificar se intervalos de confiança se sobrepõem
- Reproduzir ou auditar os testes estatísticos prometidos

**Ação:** Adicionar colunas de desvio padrão (σ) ou IQR para tempo e gap em cada tabela. No mínimo, reportar para as instâncias representativas da Tabela 1.

---

### H3 — Justificativa Técnica do Overhead do FullGPU Infundada

**Localização:** §4.2 (último parágrafo sobre pr1002).

**Texto atual:**
> *"sugerindo que a complexidade de manter todo o estado na GPU (e a possível saturação de recursos do dispositivo) começa a impactar o desempenho bruto, embora favoreça a convergência para melhores ótimos locais"*

**Problema:** A alocação estática de memória na inicialização não afeta o desempenho do laço principal de gerações. "Complexidade de manter todo o estado" não é um conceito técnico preciso.

**Causas prováveis mais corretas:**

1. **Pressão de registradores:** O FullGPU executa todos os operadores genéticos via kernel CUDA. Kernels complexos (OX crossover + mutação + 2-opt na mesma sequência) competem por registradores, reduzindo a ocupação dos SMs.
2. **Saturação de largura de banda de memória:** Com todos os operadores residindo na GPU, a memória global é acessada por mais kernels concorrentes, aumentando a contenção.
3. **Sincronização entre kernels:** O modelo geracional exige barreiras de sincronização entre geração e avaliação. Em variantes híbridas, essa sincronização ocorre via transferência CPU-GPU; no FullGPU, ocorre via `cudaDeviceSynchronize()` ou streaming, que pode ser menos eficiente para o perfil específico de acesso deste kernel.
4. **Crossover OX em GPU:** O OX é inerentemente sequencial (preenchimento posição a posição), o que pode levar a maior divergência de warps comparado ao 2-opt, que é mais regular.

**Ação:** Substituir a explicação vaga por uma análise técnica de pelo menos um dos mecanismos acima.

---

### H4 — Divergência de Fluxo SIMT Apresentada como "Flexibilidade"

**Localização:** §2.6 e §2.6.3.

**Descrição:** O texto descreve o SIMT como permitindo que threads "possam, teoricamente, seguir fluxos distintos (divergência)" e usa isso para afirmar "flexibilidade" do modelo. A mesma seção (§2.6.3) depois reconhece a penalidade: *"embora isso acarrete penalidade de desempenho pela 'serialização' dos caminhos divergentes"*.

A contradição cria uma impressão incorreta: divergência de warp **não é uma característica positiva** do SIMT para desempenho. A "flexibilidade" existe apenas no nível de modelo de programação (o programador pode escrever `if` por thread), mas execução divergente causa serialização linear.

**Correção sugerida:** Distinguir explicitamente: *"O SIMT oferece um modelo de programação mais expressivo que o SIMD puro (cada thread tem seu próprio estado de registradores e pode executar caminhos distintos), o que facilita a implementação de algoritmos complexos. Contudo, quando threads de um mesmo warp divergem, a GPU serializa os caminhos, executando cada ramo sequencialmente — o que caracteriza uma penalidade de desempenho, não uma vantagem."*

---

### H5 — Comentário TODO Interno Presente no Documento Final

**Localização:** §3.1.2, após justificativa do float64.

**Texto:**
> `<!--TODO: Eu não tenho certeza sobre esta afirmação, acredito ser devido, primeiramente, à baixa tolerância da versão 1, onde o ótimo era considerado <1e^{-6}. Mas já não se aplica. -->`

Este comentário revela dúvida do autor sobre uma afirmação presente no documento e não deve aparecer em uma versão submetida. Além disso, corrobora o problema identificado em C4 (justificativa fraca para float64).

**Ação:** Remover o comentário; resolver a dúvida de forma definitiva (ver C4).

---

## 4. Problemas Moderados 🟡

### M1 — ILP e Mascaramento de Latência Omitidos na Discussão de Ocupação

**Localização:** §3.1.1, item 3 ("Ocupação e Mascaramento de Latência").

**Descrição:** O texto conclui que alta pressão de registradores → menor ocupação → perda de eficiência. Essa cadeia causal omite o **Paralelismo em Nível de Instrução (ILP)**: threads individuais com múltiplas instruções independentes em voo (*in-flight*) podem mascarar latências de memória mesmo com baixa ocupação de warps. NVIDIA documenta casos em que 25% de ocupação é suficiente para throughput máximo se o ILP for alto.

**Ação:** Adicionar nota: *"A relação entre ocupação e desempenho não é linear: kernels com alto ILP interno podem atingir throughput próximo ao máximo mesmo com baixa ocupação de warps [@nvidia2024cuda]."*

---

### M2 — SIMT Atribuído à Taxonomia de Flynn

**Localização:** §2.6 — "relacionado às classificações de arquiteturas de Flynn [@flynn1972taxonomy]".

**Descrição:** SIMT não consta na taxonomia original de Flynn (SISD, SIMD, MISD, MIMD). Foi introduzido pela NVIDIA a partir de 2007 como extensão do modelo SIMD para execução em GPU. Dizer que é "relacionado às classificações de Flynn" é impreciso — seria mais correto dizer "inspirado nas classificações de Flynn" ou "que estende o paradigma SIMD de Flynn".

---

### M3 — Razão de Aproximação 3/2 Atribuída a "Heurísticas MST e Christofides"

**Localização:** §2.2 — *"heurísticas baseadas em MST e em Christofides podem oferecer garantias de aproximação sobre o custo ótimo de 3/2 B*"*

**Descrição:** A razão de aproximação 3/2 é propriedade exclusiva do **algoritmo de Christofides-Serdyukov**. Heurísticas MST simples (como a heurística de "duplicação" de MST) fornecem garantia de 2× para TSP métrico. Confundir os dois pode levar o leitor a superestimar garantias de heurísticas mais simples.

**Correção:** *"O algoritmo de Christofides-Serdyukov garante uma solução com custo no máximo 3/2 do ótimo para TSP métrico; heurísticas de duplicação de MST garantem no máximo 2×."*

---

### M4 — Definição Inconsistente dos Estratos

**Localização:** §3.3 (tabela: `n_coords ≤ 100`) vs. Figura \ref{fig:results-flowchart} (texto interno Mermaid: `n_coords < 100`).

**Descrição:** A tabela de instâncias seleciona kroA100 como "Pequena" (n=100), compatível com `≤ 100`. O fluxograma usa `< 100`. O threshold real do código (`cpu_size_threshold = 100` em `benchmark.json`) usa `≤ 100`. Uniformizar para `n_coords ≤ 100` em todas as ocorrências.

---

### M5 — Baixo Poder Estatístico do Estrato 1 Não Reconhecido

**Localização:** Capítulo 4, análise do Estrato 1.

**Descrição:** O Friedman test no Estrato 1 retornou p=0,086 com apenas 6 (ou 12) instâncias pequenas e 4 algoritmos. O texto deve reconhecer explicitamente que a não-significância pode refletir **poder estatístico insuficiente** (amostra pequena), não necessariamente ausência de efeito real. O cálculo de poder post-hoc deveria ser apresentado.

---

### M6 — Tamanho de População Não Referenciado à Literatura

**Localização:** §3.2.1 — *"O tamanho da população ($n_{pop}$) é adaptativo, definido como $2 \times n_{coords}$"*

**Descrição:** A fórmula `n_pop = 2 * n_coords` não é citada a nenhuma referência. É uma escolha razoável, mas deve ser justificada brevemente (e.g., referência à convenção de Eiben & Smith 2015, ou a resultados experimentais do próprio autor).

---

## 5. Problemas Menores 🟢

### N1 — Complexidade Espacial do Held-Karp na Nota de Rodapé

**Localização:** Rodapé `[^held-karp]`.

**Texto:** *"O algoritmo Held-Karp 'troca' a complexidade temporal da força bruta O(n!) por uma complexidade espacial O(n 2^n)"*

**Problema:** O Held-Karp tem complexidade temporal $O(n^2 2^n)$ e espacial $O(n \cdot 2^n)$. A frase faz soar como se *apenas* a complexidade espacial fosse exponencial, quando o tempo também o é. Reescrever como: *"reduz a complexidade temporal de $O(n!)$ para $O(n^2 2^n)$ (ainda exponencial), ao custo de $O(n \cdot 2^n)$ de memória"*.

---

### N2 — Atribuição do Algoritmo de Christofides

**Localização:** §2.2.

**Descrição:** Nygaard Serdyukov publicou independentemente o mesmo algoritmo em 1978 (em russo). A comunidade científica usa crescentemente "Christofides-Serdyukov" para reconhecer ambos. Considerar atualizar a nomenclatura.

---

### N3 — Fórmula do Ganho de Performance Não Explicitada

**Localização:** §4.3, Tabela 2.

**Descrição:** O valor 5,69× para HybridOptimized é descrito como "ganho de performance mediano por instância". Não está claro se é:

- **Mediana de (T_Naive_i / T_Opt_i)** para cada instância i — metodologicamente correto
- **Mediana(T_Naive) / Mediana(T_Opt)** — estatisticamente incorreto (razão de medianas ≠ mediana de razões)

Explicitar a fórmula usada.

---

### N4 — Discussão Embutida nos Resultados (Estrutural)

**Localização:** §4.5 "Discussão Geral" dentro do Capítulo 4.

**Descrição:** Em estrutura clássica de TCC, resultados e discussão são capítulos separados. Com a ausência do Capítulo 5, parte da discussão está embutida no Capítulo 4. Quando o Capítulo 5 for escrito, revisar para que §4.5 contenha apenas síntese factual e o Capítulo 5 contenha a interpretação crítica.

---

## 6. Aspectos Positivos

- **Taxonomia de Crainic corretamente aplicada:** Todas as 4 variantes classificadas como Tipo 1 (paralelismo de dados), o que está correto e bem justificado.
- **Protocolo estatístico bem planejado:** Uso de Demšar 2006, Shapiro-Wilk, Wilcoxon/t-pareado, Friedman + Nemenyi está metodologicamente correto — falta apenas executá-lo no texto.
- **38 instâncias e 26 repetições:** Conjunto de instâncias robusto; 26 repetições é adequado para testes não-paramétricos.
- **Caracterização do hardware:** A escolha da GTX 1050 Mobile como hardware de entrada é bem justificada metodologicamente (limitações evidentes = diferenças mais marcantes entre variantes).
- **Motivação para kernels CUDA customizados:** A justificativa para `RawKernel` vs. API de alto nível do CuPy está tecnicamente correta e bem explicada.
- **Fluxogramas Mermaid:** A documentação visual dos algoritmos (2-opt, GA, GA+2-opt) é clara e adequada.
- **Aritmética das tabelas confere:** Os ganhos por instância (Tabela 1) foram verificados e estão corretos.

---

## 7. Lista de Revisão Priorizada

### Crítico — Bloqueante para Submissão

- [ ] **[C1]** Escrever Capítulo 5 (Discussão) e Capítulo 6 (Conclusões + Trabalhos Futuros)
- [ ] **[C2]** Incluir tabela de resultados estatísticos no Capítulo 4: p-valores (Friedman: p=0,086 e p=6,9×10⁻⁵), Nemenyi (FullGPU vs Naive: p=0,000283; FullGPU vs Opt: p=0,000708; Naive vs Opt: p=0,97)
- [ ] **[C3]** Explicar mecanismos de divergência entre variantes (PRNG GPU, associatividade FP) e qualificar o termo "isoalgorítmico" como equivalência estrutural/paramétrica, não determinística
- [ ] **[C4]** Resolver inconsistência float64 vs. distâncias inteiras TSPLIB: ou migrar para `nint(sqrt(...))` ou declarar explicitamente o uso de distâncias euclidianas contínuas e recalcular os ótimos de referência com a mesma métrica

### Alto — Deve Corrigir

- [ ] **[H1]** Corrigir descrição da complexidade do Concorde ("exponencial no pior caso") e separar de Held-Karp ($O(n^2 2^n)$)
- [ ] **[H2]** Adicionar desvio-padrão (ou IQR) nas Tabelas 1, 2 e 3
- [ ] **[H3]** Substituir explicação de overhead do FullGPU (§4.2) por análise técnica: pressão de registradores, saturação de largura de banda, ou overhead de sincronização entre kernels
- [ ] **[H4]** Clarificar que divergência de warp é sempre uma penalidade de desempenho; reescrever para não sugerir que é uma "flexibilidade" benéfica
- [ ] **[H5]** Remover comentário `<!--TODO-->` de §3.1.2

### Moderado — Deve Melhorar

- [ ] **[M1]** Adicionar nota sobre ILP e mascaramento de latência na discussão de ocupação (§3.1.1)
- [ ] **[M2]** Corrigir atribuição do SIMT à taxonomia de Flynn (SIMT não consta em Flynn 1972)
- [ ] **[M3]** Separar garantia de aproximação 3/2 (exclusiva do Christofides-Serdyukov) da aproximação 2× de heurísticas MST simples
- [ ] **[M4]** Uniformizar threshold do Estrato 1 para `n_coords ≤ 100` em todas as ocorrências
- [ ] **[M5]** Reconhecer explicitamente o baixo poder estatístico do Estrato 1 (n=6 ou 12 instâncias) e considerar análise de poder post-hoc
- [ ] **[M6]** Referenciar ou justificar a fórmula `n_pop = 2 × n_coords`

### Menor — Opcional

- [ ] **[N1]** Corrigir nota de rodapé do Held-Karp: tempo é $O(n^2 2^n)$, espaço é $O(n \cdot 2^n)$
- [ ] **[N2]** Considerar nomenclatura "Christofides-Serdyukov"
- [ ] **[N3]** Explicitar fórmula do ganho de performance mediano (mediana de razões por instância)
- [ ] **[N4]** Ao escrever Capítulo 5, mover conteúdo interpretativo de §4.5 para lá

---

## Apêndice — Referência Cruzada de Problemas

| Problema do Usuário | ID neste Documento | Status |
| :--- | :--- | :--- |
| Falácia da equivalência isoalgorítmica | C3 | Confirmado + expandido |
| Incompatibilidade com padrão TSPLIB EUC_2D | C4 | Confirmado + expandido |
| Complexidade do Concorde "indefinida" | H1 | Confirmado + correção proposta |
| Dicotomia SIMT: flexibilidade vs. penalidade | H4 | Confirmado + correção proposta |
| Mascaramento de latência (ILP) omitido | M1 | Confirmado |
| Ausência de validação quantitativa (p-valores) | C2 | Confirmado + valores fornecidos |
| Justificativa temporal infundada para FullGPU | H3 | Confirmado + causas técnicas propostas |
| Capítulos 5 e 6 ausentes | C1 | Identificado nesta revisão |
| Desvio-padrão ausente das tabelas | H2 | Identificado nesta revisão |
| Comentário TODO no texto | H5 | Identificado nesta revisão |
