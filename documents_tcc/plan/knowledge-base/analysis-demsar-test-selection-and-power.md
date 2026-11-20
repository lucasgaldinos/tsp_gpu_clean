---
title: "Seleção de teste e poder estatístico em Demšar (2006): achados da auditoria de conformidade"
description: >
  Registro permanente dos achados estatísticos de uma auditoria de conformidade contra
  Demšar, Statistical Comparisons of Classifiers over Multiple Data Sets (JMLR 7, 2006).
  Explica, com intuição, fórmula em KaTeX e exemplo numérico, as três fraquezas do teste t
  pareado, a armadilha do pré-teste de normalidade, a mecânica e o piso de poder do Wilcoxon
  signed-ranks, a destruição do poder por empates e arredondamento, a correção de
  Iman–Davenport sobre o Friedman, a escala q_alpha/sqrt(2) da diferença crítica, a escolha
  entre Nemenyi e procedimentos contra um controle, e por que a unidade amostral é a instância
  e não a repetição. Fecha com as duas medições sobre results_v4: entre instâncias, sobre as
  médias de célula, e intra-instância, sobre os vetores por repetição dos checkpoints.
created: 2026-11-14
status: final
author:
  - "Lucas Galdino"
  - "GitHub Copilot"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-14
    modifications:
      - description: >
          Criado o registro dos achados da auditoria Demšar 2006: fraquezas do teste t pareado,
          armadilha do pré-teste de normalidade com contraexemplo medido e simulado, mecânica do
          Wilcoxon com o exemplo de 14 data sets verificado, empates e arredondamento, Friedman
          com Iman–Davenport e o caso degenerado F -> infinito, diferença crítica, Nemenyi contra
          procedimentos de controle, unidade de análise instância versus repetição, e as tabelas de
          medição sobre results_v4.
  - date_modified: 2026-11-14
    modifications:
      - description: >
          Acrescentada a seção 9.5 com os testes pareados por instância a partir de
          results_v4/checkpoints/*.json (69 comparações por métrica, pareamento por raw_seeds):
          zero vereditos alterados em 210 comparações, 19 de 69 células de gap dominadas por
          empates, as duas variantes híbridas idênticas em 30/30 repetições nos 14 problemas, e a
          guarda de variância nula documentada como requisito de recomputação. A seção 9.4 passou a
          tratar qual arquivo basta para qual escopo, em vez de declarar o escopo intra-instância
          inalcançável, e a tabela da seção 10 ganhou a regra correspondente.
  - date_modified: 2026-11-15
    modifications:
      - description: >
          Correção substantiva da seção 9.5 e da tabela da seção 10. A contagem anterior (46 de 69
          valores-p armazenados mudando) estava errada por dupla contagem recursiva: o bloco
          statistical_tests.pairwise_comparisons tem 72 registros — 57 wilcoxon, 15 no_test_needed
          e zero paired_t_test —, de modo que adotar Wilcoxon sempre move zero valores e zero
          vereditos. A regra alternativa de trocar para o teste t em n >= 30 é a que teria custo
          (57 de 72 valores), e a tabela passou a registrar por que nenhum limiar de n é usado:
          o "cerca de 30" do artigo conta data sets, não repetições.
related_files:
  - "[analysis-demsar-critical-difference-diagrams.md](./analysis-demsar-critical-difference-diagrams.md)"
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
  - "[table-of-symbols.plan.md](../table-of-symbols.plan.md)"
  - "[statistics.py](../../../src/benchmarking/analysis/statistics.py)"
  - "[cd_diagram.py](../../../src/benchmarking/analysis/cd_diagram.py)"
  - "[aggregate_statistics.py](../../../src/benchmarking/analysis/aggregate_statistics.py)"
  - "[stats_utils.py](../../../src/benchmarking/analysis/stats_utils.py)"
tags:
  - analysis/benchmarking
  - analysis/statistics
  - analysis/methodology
  - benchmark/comparison
  - benchmark/metrics
  - guide/statistics
  - guide/methodology
  - notes/decisions
  - notes/reference
  - review/statistics
  - review/quality
---

# Seleção de teste e poder estatístico em Demšar (2006)

Este documento fixa, de uma vez, o que uma auditoria de conformidade contra Demšar (2006)
estabeleceu sobre a estatística do projeto: quais testes o artigo recomenda, por que a
recomendação não é a que o código aplicava, e o que a medição sobre `results_v4` mostrou.
A auditoria comparou o módulo [src/benchmarking/analysis/](../../../src/benchmarking/analysis/)
com o texto do artigo por meio de 118 perguntas atômicas binárias. O núcleo de ranking passou
— ranks por instância, empates por média, correção de empates herdada do SciPy, normalização de
direção da métrica e a diferença crítica $CD = q_\alpha\sqrt{k(k+1)/6N}$ com $q_\alpha$ derivado
da amplitude studentizada — e três lacunas ficaram registradas nas decisões `DP13-3` e `DP13-4`,
nos defeitos `BUG-60`, `BUG-61`, `BUG-62` e nas tarefas `Task 13.8` a `13.13` do
[plano de reorganização](../plan-reorganization-and-typing.plan.md).

O objetivo deste registro é evitar redescoberta: cada seção traz primeiro a intuição em
linguagem comum, depois a fórmula e, por fim, um exemplo numérico. Números marcados como
**medidos** vêm de `results_v4`: as médias por célula de `problem_statistics/` alimentam os
testes entre instâncias, e os vetores `raw_*` de `checkpoints/` alimentam os testes
intra-instância da seção 9.5. Números marcados como **do artigo** vêm das Tabelas 2, 5 e 6 de
Demšar (2006); números marcados como **simulados** vêm do script reproduzível da seção 2.4.

> [!IMPORTANT]
> A recomendação de §4.2 não é incondicional: "we recommend using the Wilcoxon test, **unless
> the t-test assumptions are met**, either because we have many data sets or because we have
> reasons to believe that the measure of performance across data sets is distributed normally".
> O teste $t$ é admissível — mas por justificativa **a priori**, nunca por pré-teste de
> normalidade, que o próprio artigo descarta em §3.1.2.

## 1. As três fraquezas do teste $t$ pareado

O teste $t$ pareado testa se a diferença média entre dois algoritmos, medida nas mesmas $N$
instâncias, é indistinguível de zero:

$$t = \frac{\bar{d}}{\sigma_d/\sqrt{N}}, \qquad \bar{d} = \frac{1}{N}\sum_{i=1}^{N}\left(c_{2i} - c_{1i}\right),$$

distribuído segundo a distribuição de Student com $N-1$ graus de liberdade. Demšar (§3.1.2)
aponta três defeitos que atingem esse desenho no contexto de benchmarks entre conjuntos de dados.

### 1.1 Comensurabilidade

**Intuição.** A média $\bar d$ trata "+2 segundos em `att48`" e "+2 segundos em `gr666`" como a
mesma quantidade. Mas as instâncias diferem de escala por ordens de magnitude: um segundo é
ruído em uma instância que leva mil segundos e é a execução inteira em outra que leva
14 milissegundos. Quando as unidades não são comensuráveis, $\bar d$ é uma média de maçãs com
laranjas, ponderada pelas instâncias maiores.

**O que o artigo diz.** "the t-test only makes sense when the differences over the data sets are
commensurate" (§3.1.2). A resposta de Demšar é a mesma que ele usa para rejeitar médias de
acurácia: trocar a estatística — ranks, no caso do Wilcoxon e do Friedman — ou normalizar a
diferença pela média dos dois escores antes de testar.

**Exemplo medido.** Faixa de `mean_time` nas 14 instâncias, por variante:

| Variante | Mínimo (s) | Máximo (s) | Razão máximo/mínimo |
| --- | --- | --- | --- |
| `FullGPUModular` | 0,0140 | 9,34 | 666× |
| `HybridNaive` | 0,467 | 1033 | 2210× |
| `HybridOptimized` | 0,0313 | 46,5 | 1484× |

`mean_gap`, a métrica de qualidade em percentual, varia apenas cerca de 13× a 42× na mesma
coorte. A consequência é direta: uma diferença média de tempo é essencialmente a diferença
da instância mais lenta da coorte, enquanto uma diferença média de gap é comparável entre
instâncias.

### 1.2 Normalidade **das diferenças**, não das marginais

**Intuição.** O denominador do teste é $\sigma_d$, o desvio-padrão das diferenças $d_i$. A
derivação de Student supõe que os $d_i$ sejam normais. Não é a marginal de $A$ nem a de $B$ que
precisa ser normal: é a diferença. As marginais e a diferença são objetos distintos, e a forma
de uma não determina a da outra — marginais assimétricas podem produzir diferenças quase
simétricas, e marginais de aparência normal podem produzir diferenças com dois átomos, como na
seção 2.4.

**O que o artigo diz.** "unless the sample size is large enough (∼ 30 data sets), the paired
t-test requires that the differences between the two random variables compared are distributed
normally" (§3.1.2).

**Exemplo medido.** Ver a tabela da seção 2.3: em `mean_gap`, as seis marginais passam por
Shapiro–Wilk com folga (menor $p = 0{,}952$), enquanto a diferença de um dos três pares é
rejeitada com $p = 2{,}19\times10^{-7}$. Testar as marginais é testar a suposição errada.

### 1.3 Sensibilidade a outliers

**Intuição.** Um único ponto discrepante infla $\sigma_d$, e $\sigma_d$ está no denominador:
quanto maior a dispersão, menor o $t$, menor o poder. O teste não é apenas afetado — é
desarmado — por uma instância em que um algoritmo teve um comportamento atípico.

**O que o artigo diz.** "the t-test is, just as averaging over data sets, affected by outliers
which skew the test statistics and decrease the test's power by increasing the estimated
standard error" (§3.1.2), ao passo que "the outliers […] have less effect on the Wilcoxon than
on the t-test" (§3.1.3).

**Exemplo medido** (`mean_time`, `FullGPUModular` versus `HybridNaive`, $N=14$):

| Quantidade | Valor |
| --- | --- |
| Diferença média $\bar d$ | $-82{,}2$ s |
| Desvio-padrão $\sigma_d$ | $271{,}3$ s |
| Maior diferença absoluta (`rat783`) | $-1023{,}7$ s |
| Participação de `rat783` em $\sum_i d_i^2$ | $99{,}7\%$ |
| $\sigma_d$ sem `rat783` | $14{,}2$ s |
| $p$ do teste $t$ com todas as instâncias | $0{,}2774$ |
| $p$ do teste $t$ sem `rat783` | $0{,}0285$ |
| $p$ do Wilcoxon com todas as instâncias | $1{,}22\times10^{-4}$ |

Todas as 14 diferenças têm o mesmo sinal: `FullGPUModular` foi mais rápido em todas as
instâncias. O teste $t$ não vê isso porque uma única instância, `rat783`, responde por 99,7%
da soma de quadrados; ao removê-la, $\sigma_d$ cai por um fator de 19 e o veredito do teste $t$
se inverte. O Wilcoxon, que descarta as magnitudes e mantém os sinais e a ordem, devolve
$p = 1{,}22\times10^{-4}$ com as 14 instâncias e $p = 2{,}44\times10^{-4}$ sem `rat783` — nos
dois casos o piso exato para o $N$ correspondente, e o veredito não muda.

### 1.4 Quadro-resumo

| Fraqueza | O que é violado | Sintoma no relatório | Remédio de Demšar |
| --- | --- | --- | --- |
| Comensurabilidade | $\bar d$ soma unidades incomparáveis | p-valor dominado pelas instâncias maiores | ranks (Wilcoxon, Friedman) |
| Normalidade | $d_i$ precisa ser normal; $N<30$ impede verificar | p-valor sem nível nominal garantido | Wilcoxon por padrão |
| Outliers | $\sigma_d$ inflado por poucos pontos | poder cai justamente quando o efeito é consistente | Wilcoxon, insensível à magnitude |

## 2. Por que o pré-teste de normalidade é uma armadilha

Esta é a seção central: o código auditado escolhia entre $t$ e Wilcoxon pelo resultado de
Shapiro–Wilk em **cada amostra** — [statistics.py::StatisticalAnalyzer.paired_comparison](../../../src/benchmarking/analysis/statistics.py)
chama [statistics.py::StatisticalAnalyzer.test_normality](../../../src/benchmarking/analysis/statistics.py)
para `data_a` e `data_b` e, se ambas passarem, aplica `scipy.stats.ttest_rel`. O artigo não
prescreve isso em lugar algum; prescreve o contrário.

### 2.1 O paradoxo do tamanho amostral

**Intuição.** Testes de normalidade precisam de dados para detectar desvios. Com poucos pontos,
qualquer distribuição parece normal aos seus olhos. Então a situação é circular: a amostra é
pequena, logo precisamos da suposição de normalidade; e a amostra é pequena, logo não podemos
verificá-la.

**O que o artigo diz** (§3.1.2), literalmente:

> "Ironically, the Kolmogorov-Smirnov and similar tests for testing the normality of
> distributions have little power on small samples, that is, they are unlikely to detect
> abnormalities and warn against using the t-test. Therefore, for using the t-test we need
> normal distributions because we have small samples, but the small samples also prohibit us
> from checking the distribution shape."

**Consequência.** Um pré-teste com poder baixo tem taxa de aprovação alta mesmo sob violação
real. Um portão que aprova quase sempre não protege quase nada. Na seção 2.4, o portão aprova
91,5% das amostras nas quais a diferença **nunca** é normal.

### 2.2 O problema dos dois estágios

**Intuição.** Quando se testa a normalidade e depois se escolhe o teste, o procedimento final
não é nem o teste $t$ nem o Wilcoxon: é uma regra composta cujo nível de significância não é
$\alpha$. O p-valor reportado passa a ser condicional ao fato de a amostra ter passado no
portão, de modo que ele não é mais uniforme sob $H_0$ e não é comparável a um $\alpha$ fixo.

**Formulação.** Seja $\varphi_{\text{pré}}$ o indicador de que o portão aprovou. A regra
reporta o p-valor de um teste cuja identidade depende dos dados:

$$
p_{\text{reportado}} =
\begin{cases}
p_t, & \text{se } \varphi_{\text{pré}} = 1,\\
p_W, & \text{caso contrário},
\end{cases}
\qquad
\alpha^{\star} = P\!\left(\text{rejeitar} \mid H_0,\ \varphi_{\text{pré}} = 1\right) \neq \alpha .
$$

Duas consequências práticas, ambas visíveis no relatório gerado:

1. **Nível não garantido.** $\alpha^\star$ pode ser maior ou menor que $\alpha$, e o sinal da
   distorção depende de como a violação interage com a estimativa de variância. Não há como
   saber qual dos dois aconteceu lendo a saída.
2. **Irreprodutibilidade da leitura.** O campo `test_used` muda de par para par e de métrica
   para métrica; quem lê a tabela não consegue reconstruir o p-valor sem reexecutar o portão.
   Pior: o portão não escolhe o teste $t$ por um critério válido, porque a suposição que ele
   tenta checar é sobre a diferença, e ele checa as marginais.

> [!WARNING]
> O pré-teste não é apenas inútil: ele **muda o teste que o trabalho aplica** em função dos
> dados observados. A decisão passa a ser parte do resultado, e o resultado passa a depender
> da decisão.

### 2.3 O contraexemplo medido

Em `mean_gap`, o portão aprova **todas as seis marginais** dos três pares, com o menor
$p = 0{,}9524$ — folga enorme para o limiar de 5%. No mesmo conjunto, a normalidade **da
diferença** é rejeitada em um dos três pares com $p = 2{,}19\times10^{-7}$.

| Métrica | Margens: menor $p$ (6 amostras) | Margens com $p<0{,}05$ | Diferenças: menor $p$ (3 pares) | Pares com $p<0{,}05$ |
| --- | --- | --- | --- | --- |
| `mean_time` | $3{,}98\times10^{-7}$ | 6/6 | $2{,}94\times10^{-7}$ | 3/3 |
| `mean_gap` | $0{,}9524$ | 0/6 | $2{,}19\times10^{-7}$ | 1/3 |

O par problemático é `HybridNaive` versus `HybridOptimized`: 13 das 14 diferenças são
exatamente zero e a décima quarta vale 0,3278 ponto percentual, em `rat783`. Shapiro–Wilk
detecta esse átomo na diferença — e só na diferença. As duas marginais desse mesmo par passam
com $p = 0{,}987$ e $p = 0{,}978$.

Note a ironia dupla. Em `mean_gap` o portão escolhe o teste $t$ nos três pares (3/3), inclusive
naquele em que a suposição do teste $t$ falha. Em `mean_time` o portão escolhe o Wilcoxon nos
três pares (0/3 de teste $t$) porque as seis marginais são rejeitadas — acerta por acidente, não
por desenho, já que a decisão continua sendo tomada sobre a grandeza errada.

### 2.4 O contraexemplo analítico, com números simulados

Construa $X_i \sim \mathcal{N}(0,1)$ e $Y_i = X_i + \varepsilon_i$ com
$\varepsilon_i \in \{-\delta, +\delta\}$ equiprovável. A diferença $D_i = Y_i - X_i =
\varepsilon_i$ tem dois átomos: é o caso mais afastado da normalidade que existe. As marginais,
porém, são misturas de gaussianas com variância $1+\delta^2$; para $\delta$ pequeno elas são
quase indistinguíveis de $\mathcal{N}(0, 1+\delta^2)$. O portão passa; a suposição está violada
por construção.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261114)
n, reps, delta = 14, 20000, 0.5
gate_pass = t_all = t_gated = w_gated = sdiff = 0
for _ in range(reps):
    x = rng.normal(0, 1, n)
    y = x + rng.choice([-delta, delta], size=n)
    gate = stats.shapiro(x).pvalue >= 0.05 and stats.shapiro(y).pvalue >= 0.05
    p_t = stats.ttest_rel(y, x).pvalue
    p_w = stats.wilcoxon(y, x, zero_method="zsplit").pvalue
    t_all += p_t < 0.05
    if gate:
        gate_pass += 1
        t_gated += p_t < 0.05
        w_gated += p_w < 0.05
    sdiff += stats.shapiro(y - x).pvalue < 0.05
print(f"gate pass rate        = {gate_pass/reps:.4f}")
print(f"Shapiro(diff) rejects = {sdiff/reps:.4f}")
print(f"t alpha ungated       = {t_all/reps:.4f}")
print(f"t alpha gated         = {t_gated/gate_pass:.4f}")
print(f"W alpha gated         = {w_gated/gate_pass:.4f}")
```

Saída observada:

```text
gate pass rate        = 0.9146
Shapiro(diff) rejects = 1.0000
t alpha ungated       = 0.0594
t alpha gated         = 0.0599
W alpha gated         = 0.0557
```

O SciPy emite um `RuntimeWarning` de perda de precisão em réplicas com variância praticamente
nula; ele não impede a reprodução dos números acima.

Leitura: em 20 000 réplicas, o portão aprovou 91,46% das amostras enquanto a diferença era
**nunca** normal; e o teste $t$ aplicado depois do portão rejeitou 5,99% das vezes sob $H_0$,
contra os 5% nominais — inflação relativa de cerca de 20%. A distorção também corre para o
outro lado: um segundo cenário, com ruído contaminado (10% de $\mathcal{N}(0,3)$ mais 90% de
$\mathcal{N}(0,0{,}3)$, 8 000 réplicas, semente 7), deu portão aprovando 76,86%, Shapiro da
diferença rejeitando 61,42%, $t$ condicionado rejeitando 2,80% e Wilcoxon condicionado
rejeitando 5,24%. Em ambos os cenários o nível real é desconhecido a priori e não aparece em
lugar algum da saída.

> [!IMPORTANT]
> A regra adotada em `DP13-4` e implementada por `Task 13.11` é **Wilcoxon signed-ranks em todos os casos**, sem limiar de $n$. Shapiro–Wilk sai do caminho de decisão e permanece, no máximo, como diagnóstico descritivo. A cláusula de exceção de §4.2 — usar o teste $t$ quando "many data sets" ou quando há razão a priori para crer na normalidade — não é acionada por este projeto: a comparação entre instâncias tem $N=14$, e a comparação por instância é de repetições de um mesmo problema. Fixar um limiar de $n \ge 30$ importaria para repetições um número que o artigo associa a *data sets*, e a seção 9.5 mostra que essa regra moveria 57 dos 72 valores publicados sem mudar um único veredito.

## 3. Mecânica do Wilcoxon signed-ranks

### 3.1 Rankeamento e as duas somas

**Intuição.** Em vez de perguntar "a diferença média é zero?", pergunte "as diferenças
positivas tendem a ser maiores, em magnitude, do que as negativas?". Para isso, ordene todas as
diferenças pelo valor absoluto, atribua ranks de 1 a $N$ e some os ranks dos dois lados. Se os
dois algoritmos são equivalentes, as duas somas devem ficar próximas de $N(N+1)/4$.

**Fórmula** (§3.1.3), com $d_i = c_{2i} - c_{1i}$ e ranks médios em caso de empate:

$$
R^{+} = \sum_{d_i>0}\operatorname{rank}(|d_i|) + \frac{1}{2}\sum_{d_i=0}\operatorname{rank}(|d_i|),
\qquad
R^{-} = \sum_{d_i<0}\operatorname{rank}(|d_i|) + \frac{1}{2}\sum_{d_i=0}\operatorname{rank}(|d_i|),
$$

$$
T = \min\left(R^{+}, R^{-}\right).
$$

**Exemplo do artigo** (Tabela 2, 14 data sets, C4.5 com $m=0$ versus C4.5 com $m$ ajustado):
duas diferenças nulas (lung-cancer e mushroom) e um empate em $\pm0{,}005$ (adult e iris)
recebem ranks médios 1,5 e 3,5. As somas são $R^{+} = 93$ e $R^{-} = 12$; logo $T = 12$. A
tabela de valores críticos exatos para $N = 14$ e $\alpha = 0{,}05$ rejeita quando
$T \le 21$; como $12 \le 21$, a hipótese nula é rejeitada.

### 3.2 Empates e diferenças nulas

**Intuição.** O teste pressupõe diferenças contínuas, portanto sem empates. Quando há empates,
o rank que caberia a cada valor é dividido igualmente entre os empatados. Quando a diferença é
exatamente zero, o artigo manda dividi-la entre os dois lados: metade do rank vai para $R^{+}$,
metade para $R^{-}$; se o número de zeros for ímpar, um deles é descartado (§3.1.3, "Ranks of
$d_i = 0$ are split evenly among the sums; if there is an odd number of them, one is ignored").
É a mesma convenção que `scipy.stats.wilcoxon` chama de `zero_method="zsplit"`, e é a que
[statistics.py::StatisticalAnalyzer.paired_comparison](../../../src/benchmarking/analysis/statistics.py)
usa no ramo não paramétrico.

**Exemplo medido** (`mean_gap`, `HybridNaive` versus `HybridOptimized`, $N=14$): 13 diferenças
nulas de rank médio 7 e uma diferença não nula de rank 14. Metade de $13\times7 = 91$ vai para
cada lado ($45{,}5$), e o rank 14 cai no lado positivo:

| Quantidade | Valor |
| --- | --- |
| $R^{+}$ | $45{,}5 + 14 = 59{,}5$ |
| $R^{-}$ | $45{,}5$ |
| $T$ | $45{,}5$ |
| $\sigma_T$ sem correção de empates, $\sqrt{N(N+1)(2N+1)/24}$ | $15{,}93$ |
| $\sigma_T$ com correção, $\sqrt{\left[N(N+1)(2N+1) - \sum_j(t_j^3-t_j)/2\right]/24}$ | $14{,}43$ |
| $z = (T - N(N+1)/4)/\sigma_T$ | $-0{,}485$ |
| $p$ bilateral | $0{,}6276$ |

O número é correto e é inútil: 13 das 14 observações não carregam informação. A seção 4 volta
a esse ponto.

### 3.3 Exato até $N \approx 25$, normal além

**Intuição.** Para amostras pequenas, a distribuição de $T$ sob $H_0$ pode ser enumerada
exatamente, porque há apenas $2^N$ atribuições de sinal. Para amostras grandes a enumeração
cresce demais e a aproximação normal assume o serviço.

**Fórmula** (§3.1.3):

$$z = \frac{T - \tfrac{1}{4}N(N+1)}{\sqrt{\tfrac{1}{24}N(N+1)(2N+1)}} .$$

O artigo afirma: "Most books on general statistics include a table of exact critical values for
$T$ for $N$ up to 25 (or sometimes more)", e com $\alpha = 0{,}05$ rejeita-se quando
$z < -1{,}96$.

**Exemplo do artigo, com a aproximação conferida.** Com $T = 12$ e $N = 14$:

$$
z = \frac{12 - 52{,}5}{\sqrt{14\cdot15\cdot29/24}}
  = \frac{-40{,}5}{15{,}93}
  = -2{,}54 < -1{,}96 \;\Rightarrow\; \text{rejeitar}.
$$

A enumeração exata das $2^{14} = 16\,384$ atribuições de sinal dá $p = 0{,}00854$ para $T=12$,
e $p = 0{,}0494$ para $T = 21$ — coerente com a tabela crítica citada pelo artigo. A
aproximação normal e o valor exato concordam na decisão.

### 3.4 O piso de poder em $N = 14$

**Intuição.** Com 14 pares e teste exato, o menor p-valor bilateral possível é aquele em que
todos os sinais coincidem: $2/2^{14}$. Esse é o teto de evidência que o desenho permite; nenhum
tamanho de efeito compra mais.

**Fórmula e número.**

$$p_{\min}(N) = \frac{2}{2^{N}} = 2^{-(N-1)} \quad\Rightarrow\quad p_{\min}(14) = 2^{-13} = 1{,}2207\times10^{-4}.$$

**Exemplo medido.** Os três pares de `mean_time` retornam exatamente
$p_W = 0{,}0001221$ — o piso. Todos têm zero empates e as 14 diferenças compartilham o sinal.
Isso é uma informação de projeto, não de execução: uma coorte de 14 instâncias nunca poderá
declarar $p$ menor que $1{,}2\times10^{-4}$ no Wilcoxon exato.

### 3.5 Poder relativo ao teste $t$

O artigo é explícito (§3.1.3): "When the assumptions of the paired t-test are met, the Wilcoxon
signed-ranks test is less powerful than the paired t-test. On the other hand, when the
assumptions are violated, the Wilcoxon test can be even more powerful than the t-test." Os dados
deste repositório caem no segundo caso — ver seção 9.

## 4. O problema dos empates e dos arredondamentos

### 4.1 O caso medido

Em `mean_gap`, o par `HybridNaive` versus `HybridOptimized` tem 13 das 14 diferenças exatamente
nulas e Wilcoxon devolve $p = 0{,}6276$. A resposta está certa e não há o que corrigir: com uma
única observação informativa, não existe evidência suficiente para rejeitar nada. O que está
errado é **apresentar** esse $0{,}628$ como se fosse uma medida de evidência equivalente à dos
outros pares; ele é o resultado de um desenho sem poder, não de uma ausência de efeito.

> [!WARNING]
> Um p-valor alto com 13/14 diferenças nulas não é evidência de equivalência. É ausência de
> informação. O relatório precisa dizer qual dos dois casos ocorreu, e para isso precisa
> publicar a contagem de empates ao lado do p-valor.

### 4.2 Por que arredondar destrói o teste

**Intuição.** O rank é a única informação que o teste usa. Arredondar a métrica funde valores
distintos em um mesmo valor, e o teste lê essa fusão como empate. Muitos empates significam
poucos ranks distintos, logo pouca informação: o poder cai sem que nenhum dado tenha sido
perdido de verdade.

**O que o artigo diz.** "The Wilcoxon test assumes continuous differences $d_i$, therefore they
should not be rounded to, say, one or two decimals since this would decrease the power of the
test due to a high number of ties" (§3.1.3).

**Exemplo medido** — mesma coorte, mesma métrica, apenas arredondando antes de testar:

| Métrica e par | Sem arredondar | 2 casas | 1 casa | Inteiro |
| --- | --- | --- | --- | --- |
| `mean_time`, `FullGPUModular` vs `HybridOptimized` | $p=1{,}22\times10^{-4}$, 0 empates | $p=9{,}79\times10^{-4}$, 0 | $p=2{,}80\times10^{-3}$, 4 | $p=7{,}81\times10^{-3}$, 6 |
| `mean_gap`, `FullGPUModular` vs `HybridNaive` | $p=0{,}7609$, 0 empates | $p=0{,}6376$, 1 | $p=0{,}5007$, 8 | $p=1{,}0$, 12 |
| `mean_gap`, `HybridNaive` vs `HybridOptimized` | $p=0{,}6276$, 13 empates | $p=0{,}6276$, 13 | $p=0{,}6276$, 13 | $p=0{,}6276$, 13 |

O primeiro caso mostra o custo: arredondar o tempo para inteiros multiplica o p-valor por 64.
O segundo mostra o perigo silencioso: o p-valor muda de 0,76 para 1,0 sem que o veredito mude
aqui, mas um par com $p$ verdadeiro perto de 0,03 atravessaria $\alpha$ apenas por causa da
convenção de arredondamento do relatório.

### 4.3 O que fazer

1. Testar sempre a métrica em precisão total; arredondar apenas na apresentação.
2. Publicar a contagem de empates ao lado do p-valor, e o número efetivo de diferenças não
   nulas quando ele for pequeno.
3. Tratar "todas as diferenças nulas" como caso próprio (o código já faz isso: variância nula
   produz `test_used="no_test_needed"`), e tratar "quase todas nulas" como diagnóstico de poder,
   não como resultado substantivo.

## 5. Friedman → Iman–Davenport

### 5.1 O omnibus de Friedman

**Intuição.** Para comparar $k>2$ algoritmos nas mesmas $N$ instâncias, ranqueie os algoritmos
dentro de cada instância (1 = melhor) e olhe para o rank médio de cada um. Se todos fossem
equivalentes, todos os ranks médios seriam iguais a $(k+1)/2$. O teste mede o quanto eles se
afastam disso.

**Fórmula** (§3.2.2), com $R_j = \frac{1}{N}\sum_i r_i^j$:

$$\chi_F^2 = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4}\right],$$

assintoticamente $\chi^2$ com $k-1$ graus de liberdade "when $N$ and $k$ are big enough (as a
rule of a thumb, $N > 10$ and $k > 5$)".

### 5.2 Por que $\chi_F^2$ é conservador, e o que Iman e Davenport fizeram

**Intuição.** A aproximação $\chi^2$ supõe $N$ grande. Com $N$ pequeno ela devolve um valor
sistematicamente baixo, logo um p-valor sistematicamente alto, logo um teste que rejeita menos
do que deveria. Iman e Davenport corrigem isso reescalando a mesma estatística por uma razão
que depende de $N$ e de $\chi_F^2$, e comparando com a distribuição $F$ em vez da $\chi^2$.

**Fórmula** (§3.2.2): "Iman and Davenport (1980) showed that Friedman's $\chi_F^2$ is undesirably
conservative and derived a better statistic"

$$F_F = \frac{(N-1)\,\chi_F^2}{N(k-1) - \chi_F^2},$$

distribuída segundo $F$ com $k-1$ e $(k-1)(N-1)$ graus de liberdade.

**Exemplo do artigo** (Tabela 6: quatro algoritmos, 14 data sets, ranks médios 3,143, 2,000,
2,893, 1,964):

$$\chi_F^2 = \frac{12\cdot14}{4\cdot5}\left[(3{,}143^2 + 2{,}000^2 + 2{,}893^2 + 1{,}964^2) - \frac{4\cdot5^2}{4}\right] = 9{,}28,$$

$$F_F = \frac{13\cdot9{,}28}{14\cdot3 - 9{,}28} = 3{,}69 .$$

Com $F$ em $(3, 39)$ e valor crítico 2,85 para $\alpha = 0{,}05$, rejeita-se a hipótese nula.
A variante $\chi^2$ com 3 graus de liberdade também rejeitaria, mas com menos folga — que é
exatamente o viés que a correção remove.

### 5.3 O caso degenerado: $F_F \to +\infty$

**Intuição.** O denominador de $F_F$ é $N(k-1) - \chi_F^2$. Mas $\chi_F^2$ tem máximo
$N(k-1)$, atingido quando **todas** as instâncias ordenam os algoritmos exatamente na mesma
sequência. Nesse caso o denominador é zero, $F_F$ diverge e o p-valor tende a zero. O limite
está correto — concordância perfeita é a evidência mais forte possível —, mas a divisão não
pode ser executada.

**Fórmula e verificação.**

$$\chi_{F,\max}^2 = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k}(k+1-j)^2 - \frac{k(k+1)^2}{4}\right]
= \frac{12N}{k(k+1)}\cdot\frac{k(k+1)(k-1)}{12} = N(k-1).$$

**Exemplo medido** (`mean_time`, $N=14$, $k=3$): ranks médios exatamente 1,000 / 3,000 / 2,000
— as 14 instâncias ordenam as três variantes de forma idêntica. Então
$\chi_F^2 = 14\cdot2 = 28 = N(k-1)$ e

$$F_F = \frac{13\cdot28}{14\cdot2 - 28} = \frac{364}{0} \to +\infty, \qquad p_F = 0 .$$

O caminho de código que implementar Iman–Davenport precisa, portanto, detectar
$N(k-1) - \chi_F^2 \le 0$ e devolver $F_F = +\infty$ com $p_F = 0$, em vez de dividir. O caso
espelhado também existe: quando todos os blocos são integralmente empatados, $\chi_F^2 = 0$ e o
denominador vale $N(k-1)$; aí $F_F = 0$ e $p_F = 1$, o resultado correto para ausência total de
informação. As duas guardas já existentes em
[statistics.py::StatisticalAnalyzer.friedman_test](../../../src/benchmarking/analysis/statistics.py)
— $N<2$ e empates integrais em todos os blocos — tratam do segundo caso. O primeiro, denominador
nulo com $\chi_F^2$ no máximo, só aparece quando a variante $F$ passar a ser calculada.

### 5.4 Números medidos nos dois sentidos

| Métrica | $\chi_F^2$ | $p_{\chi^2}$ (df 2) | $F_F$ | $p_F$ (df 2; 26) | Ranks médios |
| --- | --- | --- | --- | --- | --- |
| `mean_time` | $28{,}000$ | $8{,}32\times10^{-7}$ | $+\infty$ | $0$ | 1,000 / 3,000 / 2,000 |
| `mean_gap` | $0{,}3256$ | $0{,}850$ | $0{,}1529$ | $0{,}859$ | 1,929 / 2,107 / 1,964 |

Nos dois casos o veredito é o mesmo nas duas variantes, o que torna a migração segura para o
que já está escrito: a correção muda a margem, não a conclusão. Ordem dos ranks médios:
`FullGPUModular`, `HybridNaive`, `HybridOptimized`.

## 6. A diferença crítica e o fator $q_\alpha/\sqrt{2}$

Esta seção fica curta de propósito: a derivação completa, a convenção de escala e a construção
do diagrama estão em
[analysis-demsar-critical-difference-diagrams.md](./analysis-demsar-critical-difference-diagrams.md).
O necessário para ler as tabelas seguintes é:

$$CD = q_\alpha\sqrt{\frac{k(k+1)}{6N}}, \qquad q_\alpha = \frac{Q_\alpha}{\sqrt{2}},$$

onde $Q_\alpha$ é o valor crítico da amplitude studentizada para $k$ grupos e infinitos graus
de liberdade. A divisão por $\sqrt{2}$ existe porque o Nemenyi compara pares, e a amplitude
studentizada é definida para o maior de $k$ desvios, não para uma diferença isolada.

**Exemplo do artigo** ($k=4$, $N=14$, $\alpha=0{,}05$): $Q_{0{,}05} = 3{,}633$ implica
$q_\alpha = 2{,}569$, e

$$CD = 2{,}569\sqrt{\frac{4\cdot5}{6\cdot14}} = 2{,}569 \times 0{,}488 = 1{,}25 .$$

O fator $\sqrt{k(k+1)/6N} = 0{,}488$ aparece no artigo como o erro-padrão usado nos $z$ dos
pós-testes. Em [stats_utils.py::compute_nemenyi_critical_distance](../../../src/benchmarking/analysis/stats_utils.py)
a mesma expressão é calculada diretamente a partir de `scipy.stats.studentized_range`.

## 7. Nemenyi versus procedimentos contra um controle

### 7.1 A conta das comparações

**Intuição.** Todo procedimento de comparação múltipla paga um preço pelo número de hipóteses
que testa: quanto mais comparações, mais alto o limiar para cada uma, menos poder. Nemenyi é o
procedimento de **todos os pares** e paga $k(k-1)/2$ comparações. Se a pergunta é apenas "cada
variante supera a referência?", as hipóteses são $k-1$, e o limiar pode ser mais baixo.

**Fórmula** (contagem):

$$\text{Nemenyi}: \binom{k}{2} = \frac{k(k-1)}{2}
\qquad\text{versus}\qquad
\text{controle}: k-1 .$$

Para $k=4$: 6 contra 3. Para $k=5$: 10 contra 4.

**O que o artigo diz** (§3.2.2), literalmente: os métodos de controle de erro familiar "are in
this specific case more powerful than the Nemenyi test, since the latter adjusts the critical
value for making $k(k-1)/2$ comparisons while when comparing with a control we only make $k-1$
comparisons", e conclui "We thus should not make pairwise comparisons when we in fact only test
whether a newly proposed method is better than the existing ones."

### 7.2 O mesmo dado, dois limiares

| Procedimento | $k$ | $q_{0{,}05}$ (Tabela 5) | $CD$ com $N=14$ | Pares significativos no exemplo do artigo |
| --- | --- | --- | --- | --- |
| Nemenyi (todos os pares) | 4 | 2,569 | 1,25 | nenhum |
| Bonferroni–Dunn (controle) | 4 | 2,394 | 1,16 | `C4.5+m+cf` vs `C4.5` (1,179 > 1,16) |
| Holm (controle, step-down) | 4 | — | — | `C4.5+m+cf` e `C4.5+m` vs `C4.5` |

Três leituras do mesmo conjunto de ranks médios: o Nemenyi não detecta nada; o
Bonferroni–Dunn detecta um par; o Holm, que é sequencial, detecta dois. O artigo registra
exatamente isso — "the Bonferroni-Dunn test found C4.5 and C4.5+m too similar" enquanto
"all step-down and step-up procedure found C4.5+cf+m and C4.5+m significantly different from
C4.5". É o mesmo dado, três respostas, ordenadas por poder.

### 7.3 Consequência para a tese

O trabalho resolve uma baseline explicitamente
([stats_utils.py::resolve_baseline_algorithm](../../../src/benchmarking/analysis/stats_utils.py)),
e a pergunta do Capítulo 5 é "as variantes superam a CPU", não "as variantes diferem entre si".
Para essa pergunta, o pós-teste adequado é um procedimento contra controle — Holm, já
implementado em
[statistics.py::StatisticalAnalyzer.holm_bonferroni_correction](../../../src/benchmarking/analysis/statistics.py),
ou Bonferroni–Dunn, cujo intervalo em torno do rank do controle é o que o diagrama sabe marcar.
Apresentar Nemenyi e Holm lado a lado como equivalentes convida a crítica oposta à habitual:
declarar "sem diferença" onde o procedimento mais potente encontraria uma. A decisão `DP13-3`
registra que o relato precisa escolher um e nomear a família de hipóteses a que ele se aplica.

## 8. O tamanho amostral é o número de instâncias, não o de repetições

### 8.1 A unidade de análise

**Intuição.** Demšar não testa repetições: testa **células**. Cada célula é um par
(instância, algoritmo) e contém um único número — que pode ele mesmo ser uma média sobre
repetições. No exemplo do artigo, "each cell in Table 6 represents an average over five-fold
cross validation" (§3.2.3). O $N$ dos testes de Friedman e de Wilcoxon é o número de instâncias.

**Consequência formal.** Se cada célula é $\bar c_{ij} = \frac{1}{R}\sum_{r=1}^{R} c_{ijr}$,
então o teste entre algoritmos usa $\bar c_{\cdot j}$ como observação, não $c_{ijr}$.

### 8.2 Independência é a condição, não o número de linhas

**Intuição.** Seria tentador empilhar as 30 repetições e testar com $N = 14\times30 = 420$. O
artigo proíbe: repetições da mesma instância não são independentes, porque compartilham dados e
sementes. Empilhar aumenta o $N$ aparente sem aumentar a informação.

**O que o artigo diz** (§3.2.3), literalmente:

> "There are variations of the ANOVA and the Friedman test which can consider multiple
> observations per cell provided that the observations are independent (Zar, 1998). This is not
> the case here, since training data in multiple random samples overlaps. We are not aware of
> any statistical test that could take this into account."

**Exemplo numérico.** Empilhar 14 instâncias vezes 30 repetições daria $N = 420$. Se as
repetições fossem independentes, o desvio-padrão da média cairia por $\sqrt{30} \approx 5{,}5$;
como elas compartilham instância, sementes e trajetória, são quase perfeitamente correlacionadas
— para uma correlação $\rho$, a variância da média cai para
$\sigma^2\left[1+(R-1)\rho\right]/R$, que não decresce quando $\rho \to 1$. O teste reportaria um
$p$ compatível com 420 observações independentes, e cada repetição adicional compraria confiança
sem comprar nenhuma instância nova. O número ficaria impressionante e não significaria nada.

### 8.3 Duas perguntas, dois escopos, ambos legítimos

| Escopo | Unidade | Pergunta que responde | Onde vive no código |
| --- | --- | --- | --- |
| Intra-instância | 30 repetições de uma instância | "nesta instância, o tempo médio de A difere do de B?" | `statistical_tests.pairwise_comparisons` em `results_v4/problem_statistics/*_stats.json` |
| Entre instâncias | 1 valor por célula, 14 instâncias | "A domina B na família de instâncias?" | [aggregate_statistics.py::compare_algorithms](../../../src/benchmarking/analysis/aggregate_statistics.py) e [cd_diagram.py::compute_cd_diagram_data](../../../src/benchmarking/analysis/cd_diagram.py) |

As duas perguntas são diferentes e nenhuma substitui a outra. A intra-instância mede
estabilidade de medição — ruído de relógio, escalonamento do sistema, variação de JIT — e é o que
justifica $n=30$ repetições. A entre-instâncias mede generalização sobre o conjunto de problemas,
que é o que a dissertação afirma. O erro seria misturar: um teste entre instâncias com $N=420$
não mede generalização, e um teste intra-instância não mede generalização nenhuma.

## 9. O que a medição mostrou neste repositório

As seções 9.1 a 9.3 comparam algoritmos **entre** instâncias, sobre uma célula por
(instância, algoritmo) — `mean_time` e `mean_gap` lidos de
`results_v4/problem_statistics/*_stats.json`, com o portão de Shapiro–Wilk nas duas marginais
exatamente como
[statistics.py::StatisticalAnalyzer.paired_comparison](../../../src/benchmarking/analysis/statistics.py)
faz hoje, e comparação de $p_t$ (`ttest_rel`) com $p_W$ (`wilcoxon`, `zero_method="zsplit"`).
A seção 9.5 faz a pergunta complementar, **intra-instância**, sobre os 30 valores por repetição
de `results_v4/checkpoints/*.json`.

### 9.1 Tempo de execução

O portão seleciona o Wilcoxon nos três pares (0/3 de teste $t$), porque as seis marginais são
rejeitadas:

| Par | $p_t$ | $p_W$ | Veredito de $t$ | Veredito de $W$ | Muda? | Empates |
| --- | --- | --- | --- | --- | --- | --- |
| `FullGPUModular` vs `HybridNaive` | 0,2774 | $1{,}22\times10^{-4}$ | não significativo | significativo | sim | 0 |
| `FullGPUModular` vs `HybridOptimized` | 0,0485 | $1{,}22\times10^{-4}$ | significativo | significativo | não | 0 |
| `HybridNaive` vs `HybridOptimized` | 0,2993 | $1{,}22\times10^{-4}$ | não significativo | significativo | sim | 0 |

Dois de três pares mudam de veredito, sempre na direção de **mais** significância. A causa está
na seção 1.3: as diferenças de tempo têm sinal consistente e cauda pesada, e o teste $t$ gasta
seu poder estimando um $\sigma_d$ que uma única instância domina. Não é o Wilcoxon sendo
generoso; é o teste $t$ sendo conservador por violação de suposição.

### 9.2 Gap de solução

Aqui o portão seleciona o teste $t$ nos três pares (3/3), e nenhum veredito muda:

| Par | $p_t$ | $p_W$ | Veredito de $t$ | Veredito de $W$ | Muda? | Empates |
| --- | --- | --- | --- | --- | --- | --- |
| `FullGPUModular` vs `HybridNaive` | 0,5617 | 0,7609 | não significativo | não significativo | não | 0 |
| `FullGPUModular` vs `HybridOptimized` | 0,6941 | 0,8077 | não significativo | não significativo | não | 0 |
| `HybridNaive` vs `HybridOptimized` | 0,3356 | 0,6276 | não significativo | não significativo | não | 13 |

O resultado tranquiliza quanto aos números já escritos e não tranquiliza quanto ao método: é
exatamente neste conjunto que o portão escolhe o teste $t$ no par em que a normalidade da
diferença é rejeitada com $p = 2{,}19\times10^{-7}$ e no qual 13 das 14 diferenças são nulas.
O veredito coincide; o procedimento, não.

### 9.3 Omnibus

Nenhum veredito de Friedman muda entre $\chi_F^2$ e $F_F$ — ver tabela em 5.4. Em `mean_time`
a estatística atinge o máximo e $F_F$ diverge; em `mean_gap` as duas variantes concordam em não
rejeitar. A migração para Iman–Davenport é, portanto, segura para o texto atual, e é obrigatória
para que a citação "(Friedman 1937, Iman & Davenport 1980)" no docstring de
[cd_diagram.py](../../../src/benchmarking/analysis/cd_diagram.py) descreva o que o código faz:
hoje o campo se chama `friedman_chi2` e é alimentado por `scipy.stats.friedmanchisquare`.

### 9.4 Limite do que pode ser recalculado offline

> [!WARNING]
> Nenhuma das duas fontes basta sozinha. `results_v4/problem_statistics/*_stats.json` guarda
> apenas agregados por célula — `mean_*`, `std_*` e um bloco `statistical_tests` —, o que é
> suficiente para o escopo entre instâncias (seções 9.1 a 9.3) e insuficiente para o escopo
> intra-instância (seção 9.5), que precisa dos 30 valores por célula. Esses valores estão em
> `results_v4/checkpoints/*.json`; uma auditoria que leia só as estatísticas consolidadas não
> consegue refazer a seção 9.5, e uma que leia só os checkpoints precisa reconstruir os
> agregados. A pergunta não é "é reproduzível?", mas "com qual arquivo?".

Os valores por repetição existem em outro lugar: `results_v4/checkpoints/*.json` carrega
`raw_times`, `raw_gaps`, `raw_costs`, `raw_generations` e `raw_seeds`, com 30 entradas por
célula — 51 checkpoints ativos em 2026-11-14, dos quais um (`rat783` com `HybridNaive`) tem 27,
mais 15 checkpoints de `HybridNaive` arquivados em `pre_bug57/`. Portanto o teste
intra-instância de 30 repetições é recalculável a partir dos checkpoints, mas não a partir das
estatísticas consolidadas, e a distinção importa para qualquer auditoria futura: escolher a
fonte errada produz a conclusão errada sobre o que é reproduzível.

Uma ressalva adicional de escopo: [aggregate_statistics.py::compare_algorithms](../../../src/benchmarking/analysis/aggregate_statistics.py)
agrega **apenas** `mean_gap` entre problemas (via `_extract_algorithm_gaps`). A análise de
`mean_time` entre instâncias existe na prévia desta auditoria e não no caminho de produção; os
números de 9.1 descrevem o que o protocolo faria, não o que o relatório publica hoje.

### 9.5 Testes pareados por instância, 30 repetições, pareados por `raw_seeds`

Os números de 9.1 e 9.2 comparam algoritmos **entre** instâncias. A pergunta complementar — e é
ela que `problem_statistics.perform_statistical_analysis` responde por problema — é
intra-instância: com 30 repetições da mesma instância, o tempo (ou o custo) médio de $A$ difere
do de $B$? Fonte: `results_v4/checkpoints/*.json`, pareando por **interseção de `raw_seeds`**,
nunca pela posição no vetor.

Entram 14 problemas — 9 com as quatro variantes (6 pares cada) e 5 com três variantes (3 pares
cada) —, total de **69 comparações por métrica**. `pr1002` tem um único checkpoint com uma
semente e não participa de nenhum par.

| Métrica | Comparações | Portão escolheria o teste $t$ | Vereditos que mudam entre $t$ e $W$ | Empates dominantes (>15/30) | Normalidade da diferença rejeitada | Mediana $p_t$ | Mediana $p_W$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `raw_times` | 69 | 23/69 | **0/69** | 0/69 | 26/69 | $1{,}93\times10^{-28}$ | $1{,}86\times10^{-9}$ |
| `raw_gaps` | 69 | 11/69 | **0/69** | 19/69 | 22/55 | $0{,}276$ | $0{,}385$ |

As medianas de `raw_gaps` são calculadas sobre as 55 comparações não degeneradas: as outras 14
têm variância nula e param na guarda descrita abaixo. `raw_costs` reproduz `raw_gaps`
exatamente — custo e gap são transformações afins dentro de cada instância, e tanto o Wilcoxon
quanto o teste $t$ são invariantes a transformações afins positivas —, e é a métrica que a
produção publica em `statistical_tests.pairwise_comparisons`.

Leitura. Em tempo, o teste $t$ e o Wilcoxon concordam em **todas** as 69 comparações, e não por
empate de fronteira: as medianas são $10^{-28}$ e $10^{-9}$, ordens de grandeza abaixo de
qualquer $\alpha$ razoável, de modo que a escolha do teste é irrelevante onde o efeito é grande.
Em gap, os dois também concordam em todas; ali o veredito é "sem diferença" nos dois testes, e
19 das 69 comparações têm mais de metade das diferenças nulas — em 14 delas a variância é
exatamente zero.

> [!IMPORTANT]
> As duas variantes híbridas produzem custo e gap **idênticos em todas as 30 repetições**, em
> **todos os 14 problemas** — 30/30 diferenças nulas em cada um. Não é um empate ocasional de
> arredondamento: é o mesmo resultado numérico. Nenhum teste, paramétrico ou não, tem o que
> medir nessa célula; o que se reporta é a igualdade, não um p-valor.

Casos degenerados medidos:

| Situação | Ocorrências |
| --- | --- |
| `HybridNaive` vs `HybridOptimized` com 30/30 diferenças nulas | 14 (todos os problemas) |
| `berlin52`, `CPU` vs `FullGPUModular` com 16/30 diferenças nulas | 1 |
| `berlin52`, `CPU` vs as duas híbridas com 18/30 diferenças nulas | 2 |
| `berlin52`, `FullGPUModular` vs as duas híbridas com 22/30 diferenças nulas | 2 |
| Comparações com mais de metade das diferenças nulas | 19 de 69 |

**A guarda de variância nula não é um defeito; é um requisito.** Em
[statistics.py::StatisticalAnalyzer.paired_comparison](../../../src/benchmarking/analysis/statistics.py)
o cálculo sai cedo quando `np.std(differences) < 1e-10` e devolve `test_used="no_test_needed"`,
com `p_value = 1.0` para diferença média nula e `p_value = 0.0` para deslocamento constante não
nulo. O JSON comprova: em `att48`, `HybridNaive` vs `HybridOptimized` está gravado com
`test_used: "no_test_needed"` e `p_value: 1.0`. Uma recomputação independente que **não**
replique a guarda chama `scipy.stats.ttest_rel` sobre um vetor constante, divide $0/0$ e obtém
`nan` com um `RuntimeWarning` — foi exatamente isso que produziu os `nan` que aparecem no painel
de gaps de uma prévia ingênua. O `nan` é artefato da recomputação, não comportamento do código.

**O custo de adotar a regra de Demšar é zero em vereditos.** Somando as 207 comparações
intra-instância (69 por métrica, em tempo, gap e custo) e os 3 pares entre instâncias em
`mean_gap`, nenhuma das **210 verificações** muda de veredito; as 69 de custo foram conferidas
contra o campo `p_value` já gravado nos JSON, com zero divergências de significância. No texto
da dissertação, os únicos p-valores com valor numérico citado são de Friedman
($p \approx 6{,}9\times10^{-5}$) e de Nemenyi ($p \approx 0{,}00028$, $0{,}00071$ e $0{,}97$),
ambos fora do protocolo pareado. **Nenhum número citado no texto muda.**

A ressalva é sobre os arquivos, e ela inverte a expectativa: **os 72 registros de `statistical_tests.pairwise_comparisons` não mudam em nada** — em nenhum deles. O bloco é calculado sobre custo e, nos dados atuais, contém **57 `wilcoxon_signed_rank`, 15 `no_test_needed` e zero `paired_t_test`**. Ou seja, o resultado publicado já é exatamente o que o padrão de Demšar prescreve; a não conformidade está no procedimento que escolhe o teste, não no teste que acabou escolhido. Recomputando os 72 registros a partir de `raw_costs` e replicando `holm_bonferroni_correction`, a regra de Wilcoxon sempre reproduz **0 de 72** valores-p e **0 de 72** vereditos `significant_after_correction`.

A regra alternativa — trocar para o teste $t$ em $n \ge 30$ — é a que teria custo: moveria **57 dos 72** valores-p e 57 rótulos de teste, sem mudar veredito algum. Isso a torna estritamente pior que não ter regra nenhuma, e é a razão de a regra adotada não ter limiar.

> [!IMPORTANT]
> Trocar de teste por um limiar de $n$ é uma tentação específica deste projeto, porque cada célula tem exatamente 30 repetições.
> O "∼30" de §3.1.2 é número de *data sets*, não de repetições, e §3.2.3 declara que múltiplas observações por célula só são admissíveis se independentes — condição que o artigo não pode verificar e que, num solver estocástico com sementes distintas sobre a mesma instância, é uma suposição e não um fato. Sem ganho medido, o limiar só adiciona um lugar onde o protocolo pode divergir do texto da dissertação.

O único escopo em que um veredito muda é o **tempo entre instâncias** — 2 dos 3 pares, seção
9.1 —, e esse recorte não é produzido pelo relatório, já que `compare_algorithms` agrega apenas
`mean_gap`. A troca de protocolo é, portanto, uma correção sem custo no que está publicado e um
ganho nos dois recortes que ainda não estão: o tempo entre instâncias, que passa a declarar duas
diferenças que hoje o teste $t$ esconde, e o diagnóstico de poder nas células dominadas por
empates.

## 10. Regras que ficam

| Pergunta | Regra | Fundamento |
| --- | --- | --- |
| Dois algoritmos, qualquer $n$? | Wilcoxon signed-ranks, ranks médios, `zsplit` | §3.1.2, §3.1.3, §4.2 |
| Trocar de teste por um limiar de $n$? | não — o "∼30" de §3.1.2 é número de *data sets*; a regra alternativa moveria 57 dos 72 valores publicados sem mudar veredito | §3.1.2, §3.2.3 |
| Testar normalidade para escolher? | nunca | §3.1.2 |
| Diferenças majoritariamente nulas? | publicar a contagem de empates e o número efetivo de diferenças não nulas | §3.1.3 |
| Métrica arredondada? | testar em precisão total, arredondar só na apresentação | §3.1.3 |
| $k>2$ algoritmos? | Friedman com $F_F$ de Iman–Davenport; $\chi_F^2$ só para comparabilidade | §3.2.2 |
| Denominador de $F_F$ nulo? | devolver $+\infty$ e $p=0$; nunca dividir | decorrência de §3.2.2 |
| Todos os pares, ou só a baseline? | só a baseline: Holm ou Bonferroni–Dunn, não Nemenyi | §3.2.2 |
| $N$ é o quê? | o número de instâncias; repetições não são unidades independentes | §3.2.3 |
| Recomputar teste intra-instância? | usar `results_v4/checkpoints/*.json` e parear por `raw_seeds`; `*_stats.json` não basta | §3.2.3, §9.4 |
| Um $p$ alto com muitas células empatadas? | é falta de poder, não evidência de equivalência | §3.1.3 |

> [!NOTE]
> A recomendação de §4.2 foi escrita para classificadores comparados sobre coleções de data
> sets; a transposição para variantes de um solver sobre instâncias de TSP é legítima porque a
> estrutura é a mesma — $N$ problemas, $k$ algoritmos, uma célula por par — e não porque o
> artigo fale de TSP. Quando o texto da dissertação citar o artigo, deve citar o que ele
> recomenda, e apresentar as escolhas do trabalho como escolhas, com a justificativa própria.
