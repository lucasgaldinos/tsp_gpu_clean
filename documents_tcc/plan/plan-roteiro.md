# Plano de Adequação ao Roteiro de Análise Crítica e Revisão por Pares

Este plano detalha as adições e modificações necessárias diretamente no documento [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) e no código base para cumprir os requisitos de engenharia de software e rigor acadêmico. Como o documento atual é um **relatório parcial/draft**, o foco é apresentar dados brutos, simplificar hipóteses infundadas e estabelecer um bloco claro de investigações pendentes (TODOs) para o orientador.

## 0. Sobre o Preâmbulo do Arquivo LaTeX

- [x] **Ação 0.1: Documentar o preâmbulo gerado pelo Pandoc**
  + **What (O quê):** Documentar os pacotes essenciais, opções de citação (`--citeproc` vs `biblatex`) e metadados estruturais YAML que compõem o preâmbulo.
  + **Why (Por quê):** Garantir que a transformação de Markdown para LaTeX seja reproduzível e que decisões estruturais não se percam.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** Arquivo auxiliar da base de conhecimento [convertion-to-tex.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/convertion-to-tex.md).
  + **When (Quando):** Concluído nas iterações iniciais.
  + **How (Como):** Criação de arquivo markdown de documentação interna, listando os comandos LaTeX do preâmbulo e configurações bibliográficas escolhidas.
  + **How much (Quanto):** Baixo esforço.
  + **Implicações no Relatório:** Servirá apenas como documentação (Knowledge Artifact), sem alterar visualmente a tese em PDF.
  + **DoD (Definition of Done):** Documento markdown técnico entregue e formatado contendo os pormenores do preâmbulo.

## 1. Revisão e Reescrita do Capítulo 4 (Resultados)

- [ ] **Ação 1.1: Inserir Estatísticas, Desvios, Explicar Alta Variância e Revisar Seção 4.4**
  + **What (O quê):** Adicionar valores de desvio padrão às tabelas de resultados e incluir p-values dos testes Friedman/Nemenyi. Simplificar as causas da alta variância a fatores genéricos. **Adicionalmente**, revisar a Seção 4.4 ("Melhor Algoritmo por Categoria de Tamanho"), cujas afirmações de "dominância FullGPU em todas as categorias" estão comprometidas pela injustiça paramétrica (50 vs 10 iterações de 2-opt). Inserir `TODO` explícito nessa seção alertando que os resultados serão reavaliados após a Ação 3.1.
  + **Why (Por quê):** O relatório demanda validade estatística das métricas divulgadas. Variâncias altas (ex: $> \mu$) sem análise matemática comprometem o projeto. A Seção 4.4 ([second_draft.tex L2605-2613](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2605)) afirma que FullGPU domina em todas as categorias, mas essa conclusão pode ser artefato direto da assimetria paramétrica e não da arquitetura residente.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):**
    - [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (linhas referentes às tabelas das Seções 4.2 e 4.3)
    - [second_draft.tex L2605-2613](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2605) (Seção 4.4 — Melhor Algoritmo por Categoria)
  + **When (Quando):** A ser executado logo após as coletas da Ação 3.1.
  + **How (Como):** (1) Editar as tabelas LaTeX inserindo `$Média \pm \sigma$`. (2) Inserir um texto objetivo indicando que a alta variância dos gaps provém possivelmente de comportamentos mistos (outliers vs algoritmo), os quais precisam de mais investigação futura (ver Ação 1.2). (3) Na Seção 4.4, inserir `% TODO` e nota textual explicitando que a dominância FullGPU pode ser artefato da assimetria paramétrica e que os dados serão recoletados.
  + **How much (Quanto):** Esforço Médio.
  + **Implicações no Relatório:** Como esta ação só será feita *após* recolhermos os dados sem erro de ponto flutuante, um `TODO` será deixado avisando o orientador de que a variância estatística das tabelas e as conclusões da Seção 4.4 pendem da nova amostragem iso-algorítmica.
  + **DoD (Definition of Done):** O documento descreve análises de desvio padrão e testes estatísticos acompanhados de `TODO` claro sobre a defasagem temporal dos dados devido à correção Ação 3.1. A Seção 4.4 contém ressalva explícita sobre a invalidação dos resultados pela assimetria paramétrica.
  + **Solution (parcial):**
    - **Changed Files:**
      + [second_draft.tex L2538](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2538) — `% TODO` após tabela agregada: dados invalidados por float32 + assimetria 50/10
      + [second_draft.tex L2617](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2617) — `% TODO` na Seção 4.4: conclusão de dominância FullGPU pode ser artefato
    - **Summary:** TODOs formais inseridos no LaTeX alertando sobre invalidação dos benchmarks. A inserção efetiva de $\sigma$, p-values e texto explicativo sobre variância alta depende da re-execução dos benchmarks com o código int32 corrigido (Ação 3.1 concluída).
    - **DoD Compliance:** Parcial — TODOs de invalidação inseridos. Estatísticas reais pendentes de nova coleta.

- [x] **Ação 1.2: Simplificação de Hipóteses Técnicas e Inserção de TODOs**
  + **What (O quê):** Remover das seções de Resultados (4) e Discussão (5) as afirmações complexas sobre "pressão de registradores", "divergência de warps" e "limitações microarquiteturais" exclusivas. Substituí-las por afirmações conservadoras e transferir essas hipóteses não testadas para um bloco explícito de próximos passos/TODOs no fim da dissertação.
  + **Why (Por quê):** Sem profiling granular (*Nsight Compute*), inferir causa arquitetural no código é pura especulação. Muitas anomalias de performance podem ser *mistas* (efeitos algorítmicos e arquiteturais simultâneos). Transformar essas hipóteses em etapas da pesquisa vindoura é a atitude acadêmica mais honesta num relatório parcial.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (Seções do Capítulo 4.5 e Capítulo 5/Discussão).
  + **When (Quando):** Imediatamente, na próxima fase de reescrita LaTeX.
  + **How (Como):** Buscar por palavras-chave (warps, registradores, divergência). Apagar as hipóteses absolutistas. Simplificar o texto, declarando que o limitador de performance pode estar alocado numa junção de efeitos algorítmicos (ex: complexidade de vizinhança O(n^2)) com hardware. Adicionar um bloco ao final do texto: `% TODO (Pesquisa): Validar hipótese de pressão de registradores via Nsight Compute` para formalizar a investigação pendente.
  + **How much (Quanto):** Esforço Médio.
  + **Implicações no Relatório:** Demonstra ao orientador maturidade de engenharia: as hipóteses deixaram de ser afirmações e passaram a integrar a rotina formal de pendências investigativas do TCC.
  + **DoD (Definition of Done):** O texto não possui mais hipóteses complexas tratadas como dogma. Para cada hipótese levantada, existe um registro explícito (`TODO`) na documentação das próximas investigações da tese.
  + **Solution:**
    - **Changed Files:** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2482) (L2482-2491, L2555-2562, L2634-2641)
    - **Summary:** Removidas hipóteses específicas de pressão de registradores, divergência de warps e OX sequencial do Capítulo 4. Substituídas por reconhecimento de causas mistas. Bug das 50 iterações hardcoded explicitado como causa mais provável da qualidade superior do FullGPU. Inseridos 4 novos TODOs investigativos (registradores, warps, complexidade, CUDA Events).
    - **DoD Compliance:** Nenhuma hipótese arquitetural permanece como afirmação no texto. Cada hipótese foi convertida em um TODO investigativo formal.
    - **Artifacts:**
      + 📝 [walkthrough_1.2-2.1-2.2.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_1.2-2.1-2.2.md) — Walkthrough

- [x] **Ação 1.3: Resolução da Falácia da Equivalência Isoalgorítmica**
  + **What (O quê):** Corrigir BUG severo em que o kernel GPU estava *hardcoded* para realizar **50 iterações** na busca 2-opt, enquanto a implementação CPU possuía limite base de **10 iterações**, acarretando comparação injusta e não isonômica.
  + **Why (Por quê):** Comparações sérias de metaheurísticas exigem rigor iso-algorítmico (mesmo volume de busca para mensurar speed-up).
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):**
    - [ga_fujimoto.cu](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu) (linhas 40-58)
    - [genetic_algorithm_full_gpu_early_stop.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_full_gpu_early_stop.py) (linhas do run() ga_kernel).
  + **When (Quando):** Concluído nas sessões de código.
  + **How (Como):** Refatoração da assinatura C++ e dos wrappers python/Cupy para enviar a variável `two_opt_iterations` da interface até o kernel.
  + **How much (Quanto):** Esforço Médio.
  + **Implicações no Relatório:** Invalida temporariamente todos os tempos de execução GPU mostrados na dissertação atual. Gera a obrigatoriedade explícita de `TODO` no documento atestando ao orientador que as coletas mostradas devem ser re-executadas.
  + **DoD (Definition of Done):** Kernels parametrizados, compilando, executando dinamicamente, isentos de literais mágicos.

## 2. Capítulos Faltantes (5 e 6) e Conclusão Parcial

- [x] **Ação 2.1: Reestruturação do Capítulo 5 (Discussão)**
  + **What (O quê):** Reescrever a Discussão retirando tom conclusivo precoce. Limpar jargões sobre "Heisenberg" e "LKH-3" como certezas investigativas e convertê-los em caminhos de exploração sugeridos.
  + **Why (Por quê):** O trabalho atual não está concluído, é um relatório. A discussão deve retratar o estado da arte do projeto: temos duas estratégias (batch transfer rápida e fully-resident com qualidade), as quais suscitam hipóteses mistas para o desempenho imperfeito atual.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (Capítulo 5).
  + **When (Quando):** Próxima iteração de correção LaTeX.
  + **How (Como):** Alterar redação do Cap 5. Ao invés de discursar densamente sobre teses que ainda serão avaliadas, compilar as observações comportamentais dos Gráficos. Mover as investigações complexas para uma seção designada de `TODOs de Investigação` APÓS a discussão.
  + **How much (Quanto):** Esforço Alto.
  + **Implicações no Relatório:** Transforma a tese em um documento muito mais amigável a revisões e dá transparência aos "próximos passos" ao orientador.
  + **DoD (Definition of Done):** Capítulo 5 reestruturado como um diagnóstico puramente parcial dos dados apresentados, transferindo toda experimentação aprofundada para bloco de TODOs futuros.
  + **Solution:**
    - **Changed Files:** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2620) (L2620-2624, L2630)
    - **Summary:** Seção 5.1 reescrita: removido jargão "Efeito Heisenberg", mantida informação factual sobre Nsight serializar kernels (com citação CUDA), adicionada conexão com TODOs futuros. Seção 5.2 ajustada: frase de investigação futura (LK-move) adicionada.
    - **DoD Compliance:** Cap. 5 relata limitações sem jargões não-consagrados e aponta para investigações futuras em vez de conclusões prematuras.
    - **Artifacts:**
      + 📝 [walkthrough_1.2-2.1-2.2.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_1.2-2.1-2.2.md) — Walkthrough

- [x] **Ação 2.2: Redação do Capítulo 6 (Conclusão do Relatório)**
  + **What (O quê):** Readequar a Conclusão para um fechamento do estado atual da pesquisa, sem forjar resoluções que ainda não possuímos de fato.
  + **Why (Por quê):** O relatório demanda sinceridade: a modelagem e testes estruturais foram montados, mas respostas sobre limitantes exatos arquiteturais são Trabalhos Futuros.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (Capítulo 6).
  + **When (Quando):** Em conjunto com Ação 2.1.
  + **How (Como):** A conclusão atestará que as implementações Híbrida e Full GPU se provaram viáveis, evidenciando o trade-off de vazão e tempo, mas encerrará o texto informando que o aprimoramento da amostragem com profilers definirá o verdadeiro teto de performance do problema na etapa final da monografia.
  + **How much (Quanto):** Esforço Baixo.
  + **Implicações no Relatório:** Encerra o relatório mantendo-o como documentação incremental da Tese.
  + **DoD (Definition of Done):** Capítulo 6 reescrito de forma humilde e parcial, delineando o marco onde o desenvolvimento parou.
  + **Solution:**
    - **Changed Files:** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2641) (L2641, L2653)
    - **Summary:** Abertura do Cap. 6 alterada de "cumpriu a proposta" para "documenta o progresso". Warp-Shuffle: "reduzirá" → "tem o potencial de reduzir".
    - **DoD Compliance:** Cap. 6 reflete honestamente o estado parcial do relatório, sem conclusões que dependem de investigação ainda não realizada.
    - **Artifacts:**
      + 📝 [walkthrough_1.2-2.1-2.2.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_1.2-2.1-2.2.md) — Walkthrough

## 3. Correções Teóricas e Metodológicas Críticas

- [x] **Ação 3.1: Padronização para Inteiros Discretos e Aritmética TSPLIB**
  + **What (O quê):** Converter toda a base matemática de distâncias do projeto, atualmente em Ponto Flutuante (`float32`), para Inteiros de 32 bits (`int32`). Documentar a mudança.
  + **Why (Por quê):** A TSPLIB exige que a distância `EUC_2D` seja calculada por `nint(d) = floor(d + 0.5)`. Em [matrix.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py) (linha 229), o array sofre cast `astype(xp.float32)`, entrando corrompido na GPU e inviabilizando gaps estritos ($0,0\\%$).
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):**
    - [matrix.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py)
    - [genetic_algorithm_full_gpu_early_stop.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_full_gpu_early_stop.py)
    - [ga_fujimoto.cu](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu) (e outros utilitários CuPy).
  + **When (Quando):** Imediatamente, sendo a primeira correção de código da próxima iteração.
  + **How (Como):** (1) Editar conversões do `matrix.py` e tipos nos wrappers para `int32`. (2) Alterar kernels CUDA trocando `float*` por `int*` nas distâncias/fitness.
  + **How much (Quanto):** Esforço Alto.
  + **Implicações no Relatório / Orientador (TODO explícito):** Será deixado um macro `\\textbf{TODO:}` no documento declarando que as tabelas de Resultados são do build antigo em `float` e deverão ser recalculadas pela nova base aritmética.
  + **DoD (Definition of Done):** O pipeline computacional compila tipagem nativa de `int32`, aderindo 100% ao TSPLIB, e TODO de re-coleta evidenciado no LaTeX.
  + **Solution:**
    - **Changed Files:**
      + [matrix.py L229](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py#L229) — `float32` → `int32` na saída
      + [ga_fujimoto.cu L19](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu#L19) — `float*` → `int*` dist; `float` → `long long` fitness
      + [ga_fujimoto_early_stop.cu L23](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto_early_stop.cu#L23) — mesma conversão + fix `50` hardcoded → `two_opt_iterations` param
      + [two_opt_single.cu L34](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/two_opt_single.cu#L34) — `float` → `int` em dist/deltas; remoção de epsilon `1e-5f`
      + [two_opt_batch.cu L40](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/two_opt_batch.cu#L40) — mesma conversão
      + [cost_calculator.cu L30](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/cost_calculator.cu#L30) — `float*` → `int*` dist; `float` → `long long` custos
      + [genetic_algorithm_full_gpu_early_stop.py L115](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_full_gpu_early_stop.py#L115) — `cp.float32` → `cp.int32`/`cp.int64`; reordenação de params
      + [genetic_algorithm_hybrid_naive.py L89](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_hybrid_naive.py#L89) — `cp.float32` → `cp.int32`
      + [genetic_algorithm_hybrid_optimized.py L119](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_hybrid_optimized.py#L119) — `cp.float32` → `cp.int32`/`cp.int64`; shared mem `4` → `8` bytes
      + [genetic_algorithm_cpu.py L119](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_cpu.py#L119) — `np.float32` → `np.int64`
    - **Summary:** Conversão completa de 10 arquivos (5 kernels CUDA + 5 wrappers python). Distâncias `int32`, fitness/custos `long long`/`int64`, epsilon removido. Fix do bug iso-algorítmico: 50 iterações hardcoded → parâmetro dinâmico. Todos os benchmarks anteriores invalidados e sinalizados via `% TODO` no LaTeX (Ação 1.1).
    - **Technical Justification:** `float32` tem apenas 23 bits de mantissa; o custo total de um tour supera $2^{23}$ em instâncias médias, corrompendo gaps. Distâncias TSPLIB são inteiras por definição (`nint(d)`), então `int32` é o tipo correto e exato. `long long` para fitness cobre CVRP/MDVRP sem risco de overflow.
    - **DoD Compliance:** Pipeline `int32` nativo.
    - **Artifacts:**
      + 📝 [walkthrough_3.1.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_3.1.md) — Walkthrough (inclui análise completa de prós/contras)
      + 📚 [knowledge_tsplib_arithmetic.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/knowledge-base/knowledge_tsplib_arithmetic_and_integer_types_on_gpu.md) — Knowledge

- [ ] **Ação 3.2: Rigor na Complexidade Espaço/Tempo do Solucionador Concorde**
  + **What (O quê):** Reescrever a fundamentação teórica de limites sobre algoritmos exatos.
  + **Why (Por quê):** O texto confunde gargalo temporal com gargalo espacial. Solucionadores exatos morrem primariamente por falta de memória RAM para a árvore de busca, não apenas pelo tempo exponencial puro.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (linhas de fundamentação teórica) e respectivo `refs.bib`.
  + **When (Quando):** Após reescrita dos Capítulos 5 e 6.
  + **How (Como):** Citar `applegate2006traveling` (Concorde) e `held1962dynamic` (relaxação dinâmica). Diferenciar limite assintótico O(n^2 2^n) do estouro de heap.
  + **How much (Quanto):** Esforço Médio.
  + **Implicações no Relatório / Orientador (TODO explícito):** `% TODO (Orientador): Fundamentação revisada para separar complexidade temporal e espacial no Held-Karp, favor avaliar rigor.`
  + **DoD (Definition of Done):** Ausência de ambiguidades entre as duas complexidades na base teórica.

- [x] **Ação 3.3: Desvinculação do Conceito SIMT da Taxonomia de Flynn**
  + **What (O quê):** Desvincular categoricamente "SIMT" da classificação taxonômica de Michael Flynn.
  + **Why (Por quê):** O modelo SIMT é uma abstração NVIDIA moderna. Taxonomia de Flynn (anos 60) cobre SISD, SIMD, MISD, MIMD. Essa associação direta no TCC é anacrônica.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (linhas do Referencial Teórico).
  + **When (Quando):** Após Ação 3.2.
  + **How (Como):** Apagar menção direta conectando SIMT a Flynn. Usar `\autocite{nvidia2024cuda}` e deixar Flynn apenas no SIMD clássico.
  + **How much (Quanto):** Esforço Baixo.
  + **Implicações no Relatório / Orientador (TODO explícito):** `% TODO (Orientador): Corrigido anacronismo conceitual que associava SIMT diretamente à década de 60 na taxonomia de Flynn.`
  + **DoD (Definition of Done):** Renderização limpa de confusões históricas das tecnologias de hardware abordadas.
  + **Solution:**
    - **Changed Files:** [second_draft.tex L1449](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L1449)
    - **Summary:** Redação do referencial teórico reescrita. SIMT agora é apresentado como abstração NVIDIA moderna, não englobado pela taxonomia de Flynn (que cobre apenas SIMD).
    - **DoD Compliance:** TODO inserido e confusões anacrônicas removidas do LaTeX.

- [x] **Ação 3.4: Estudo e Compreensão das Ferramentas de Perfilação CUDA**
  + **What (O quê):** Estudar em profundidade as ferramentas de análise de desempenho GPU disponíveis: o *NVIDIA Nsight Compute* (perfilação por núcleo de execução), o *Nsight Systems* (visão temporal do sistema completo), e as primitivas de eventos CUDA para instrumentação não-intrusiva via CuPy/python.
  + **Why (Por quê):** A Seção 5.1 do relatório reconhece que a análise atual baseia-se apenas em medições macroscópicas de tempo. Para validar ou refutar as hipóteses arquiteturais levantadas nos TODOs (pressão de registradores, divergência de grupos de execução), é necessário dominar as ferramentas antes de utilizá-las. Sem esse domínio, qualquer tentativa de perfilação será superficial ou incorretamente interpretada.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [CUDA C++ Best Practices Guide](file:///home/lucas_galdino/chimera/gpu_accelerated/documentation/pdfs/kirk2010programming.pdf) (capítulos sobre ocupação e perfilação)
    - [harris2007optimizing.pdf](file:///home/lucas_galdino/chimera/gpu_accelerated/documentation/pdfs/harris2007optimizing.pdf) (otimização de reduções paralelas)
    - Documentação oficial NVIDIA: `nvidia2024cuda` no `refs.bib`
    - [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2620) (Seção 5.1, onde as limitações são declaradas)
  + **When (Quando):** Antes de executar os TODOs de investigação de registradores e divergência.
  + **How (Como):** (1) Ler a documentação das ferramentas. (2) Produzir um artefato de Knowledge consolidando as capacidades de cada ferramenta, as métricas relevantes, e as limitações (intrusividade, replay passes). (3) Definir um plano de instrumentação com eventos CUDA como primeira etapa prática.
  + **How much (Quanto):** Esforço Médio.
  + **Implicações no Relatório / Orientador:** O entendimento prévio das ferramentas permitirá que os TODOs investigativos da Seção de Pendentes sejam executados com rigor empírico, produzindo dados confiáveis para o relatório final.
  + **DoD (Definition of Done):** Artefato de Knowledge produzido com resumo das ferramentas, métricas coletáveis e limitações. Plano de instrumentação definido.
  + **Solution:**
    - **Changed Files:** — (apenas artefato de Knowledge criado; sem alteração em código ou LaTeX)
    - **Summary:** Artefato de Knowledge consolidando Nsight Systems, Nsight Compute e CUDA Events/CuPy. Tabela comparativa de escopo/overhead/métricas, mapeamento de métricas para cada hipótese pendente, e plano de instrumentação em 3 fases com comandos CLI prontos.
    - **DoD Compliance:** Artefato produzido com resumo, métricas, limitações (kernel replay, suporte Pascal CC 6.1) e plano de instrumentação.
    - **Artifacts:**
      + 📚 [knowledge_ferramentas_perfilacao_cuda.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/knowledge-base/knowledge_ferramentas_perfilacao_cuda.md) — Knowledge

## 4. Elementos Pós-textuais e Classificação Rigorosa de Imagens

- [ ] **Ação 4.1: Ambientes LaTeX para Elementos Flutuantes (Diferenciação Visual)**
  + **What (O quê):** Criar classes específicas para reclassificar os floats: Listas de Figuras (esquemas lógicos), Listas de Gráficos (plots), Quadros (grades textuais fechadas) e Tabelas (grades numéricas estatísticas abertas).
  + **Why (Por quê):** Padrão normativo (ABNT) exige que um "gráfico em barra ou linha" gerado em matplotlib não seja preguiçosamente indexado numa `\listoffigures` de diagramas lógicos estruturais.
  + **Who (Quem):** Agente (IA).
  + **Where (Onde):** [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (Preâmbulo e marcações internas).
  + **When (Quando):** Etapa de refatoração final de formato.
  + **How (Como):** Incluir `\usepackage{newfloat}`. Inserir os macros declarativos para \begin{grafico} e \begin{quadro}. Substituir os blocos `figure` afetados via regex ou busca manual no `.tex`.
  + **How much (Quanto):** Esforço Muito Alto.
  + **Implicações no Relatório / Orientador (TODO explícito):** Será deixado um TODO global para validação das escolhas visuais ABNT feitas nessa fase.
  + **DoD (Definition of Done):** O PDF renderizado possui Sumários divididos (Tabelas e Gráficos autônomos) compondo hiperlinks exatos para o leitor.

## 5. Exploração de Variantes Algorítmicas do Memético

> [!NOTE]
> O algoritmo atual é um **Algoritmo Memético** (GA + 2-opt), não um simples GA. Cada fase utiliza um operador específico que pode ser substituído por alternativas com diferentes perfis de desempenho e adequação à GPU. As ações abaixo investigam teoricamente cada fase, classificando as alternativas por sua **adequação ao paralelismo massivo** (GPU-friendly vs. inerentemente sequencial). O objetivo imediato é **documentar no relatório** as alternativas existentes, justificando as escolhas atuais e registrando variantes promissoras como trabalhos futuros.

> [!IMPORTANT]
> **Classificação de paralelismo:** Cada ação indica se a fase é **🟢 Paralelizável** (independência entre indivíduos, operações vetorizáveis), **🟡 Parcialmente Paralelizável** (dependências internas mas entre indivíduos é independente) ou **🔴 Inerentemente Sequencial** (dependências de dados intra-indivíduo que dificultam paralelização eficiente).

> [!TIP]
> **Código existente:** Já existem implementações alternativas no diretório [strategies/](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/) que **não são utilizadas** na configuração atual de benchmarks. São elas: `RouletteWheelSelection`, `CrowdingSelection`, `PartiallyMappedCrossover` (PMX), `InversionMutation`, `InsertionMutation`. Essas implementações podem servir de base para experimentos comparativos futuros.

---

- [ ] **Ação 5.1: Geração da População Inicial — 🟢 Paralelizável**
  + **What (O quê):** Investigar alternativas à geração puramente aleatória de permutações para a população inicial. Documentar no relatório as opções disponíveis, suas implicações na convergência e adequação à GPU.
  + **Why (Por quê):** A geração atual (permutações uniformes via `numpy.random.permutation`) não incorpora nenhuma heurística construtiva, o que pode exigir mais gerações para convergir. Heurísticas como o vizinho mais próximo (*Nearest Neighbor*) ou inserção gulosa (*Greedy Insertion*) poderiam fornecer soluções iniciais de melhor qualidade, reduzindo o número de gerações necessárias.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [genetic_algorithm_base.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_base.py#L35) (`GeneticAlgorithmBase.__init__`)
    - [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (Seção 6.1 — Trabalhos Futuros)
  + **When (Quando):** Sequencialmente, após as ações do Grupo 3.
  + **How (Como):** Levantar e documentar as seguintes alternativas:
    - **Vizinho mais próximo** (`rosenkrantz1977analysis`): heurística $O(n^2)$, 🟢 paralelizável por indivíduo na GPU (cada thread calcula uma rota a partir de um vértice diferente).
    - **Inserção gulosa**: heurística construtiva que insere cidades minimizando o aumento de custo; 🟡 parcialmente paralelizável.
    - **Híbrida (aleatória + heurística)**: fração da população gerada por heurística, restante aleatório (`eiben2015introduction`). Mantém diversidade genética.
  + **How much (Quanto):** Esforço Baixo (apenas documentação teórica).
  + **Classificação de Paralelismo:** 🟢 Paralelizável — cada indivíduo é gerado independentemente; tanto a geração aleatória quanto as heurísticas construtivas podem ser executadas em paralelo por diferentes threads.
  + **Referências disponíveis em refs.bib:**
    - `rosenkrantz1977analysis` — análise de heurísticas construtivas para o TSP
    - `eiben2015introduction` — população inicial e diversidade em algoritmos evolutivos
    - `goldberg1989genetic` — dimensionamento de população e *building blocks*
  + **DoD (Definition of Done):** Seção no relatório (ou TODO) documentando as alternativas, com justificativa da escolha atual e indicação de qual variante investigar primeiro em trabalho futuro.

- [ ] **Ação 5.2: Seleção de Pais — 🟢 Paralelizável**
  + **What (O quê):** Investigar alternativas ao torneio de tamanho $k=5$ atualmente utilizado para seleção de pais. Documentar no relatório as opções, suas pressões seletivas e adequação à GPU.
  + **Why (Por quê):** O torneio de tamanho fixo impõe uma pressão seletiva constante. Alternativas como roleta proporcional (*Roulette Wheel*), seleção por ranqueamento (*Rank Selection*) ou torneio adaptativo podem oferecer melhor equilíbrio entre exploração e explotação, potencialmente acelerando a convergência ou evitando estagnação prematura.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [selection_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/selection_strategies.py#L27) (`TournamentSelection`, L27)
    - [selection_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/selection_strategies.py#L131) (`RouletteWheelSelection`, L131 — já implementada, não utilizada)
    - [selection_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/selection_strategies.py#L239) (`CrowdingSelection`, L239 — já implementada, não utilizada)
  + **When (Quando):** Sequencialmente, após Ação 5.1.
  + **How (Como):** Levantar e documentar as seguintes alternativas:
    - **Torneio (atual, $k=5$)** (`eiben2015introduction`): 🟢 altamente paralelizável; cada torneio é independente.
    - **Roleta proporcional** (`goldberg1989genetic`): 🟡 requer soma cumulativa global (prefix-sum), paralelizável com primitivas de redução.
    - **Seleção por *crowding*** (`eiben2015introduction`): 🟡 mantém diversidade fenotípica mas requer cálculo de distâncias entre indivíduos.
    - **Torneio adaptativo** (`eiben2011parameters`): torneio com $k$ variável ao longo das gerações; 🟢 mesma estrutura, potencialmente melhor convergência.
  + **How much (Quanto):** Esforço Baixo (documentação + avaliação teórica).
  + **Classificação de Paralelismo:** 🟢 Paralelizável — cada seleção de par é independente. Roleta requer uma operação de redução (prefix-sum) mas permanece GPU-friendly.
  + **Referências disponíveis em refs.bib:**
    - `eiben2015introduction` — introdução a operadores de seleção em EC
    - `goldberg1989genetic` — roleta, seleção proporcional e análise de convergência
    - `eiben2011parameters` — ajuste adaptativo de parâmetros em AEs
    - `cantupaz1998survey` — seleção em GAs paralelos
  + **DoD (Definition of Done):** Documentação das alternativas com análise de adequação à GPU e indicação de qual investigar em trabalho futuro.

- [ ] **Ação 5.3: Cruzamento (Recombinação) — 🟡 Parcialmente Paralelizável**
  + **What (O quê):** Investigar alternativas ao Order Crossover (OX) atualmente utilizado. Documentar no relatório as opções, suas complexidades e grau de paralelismo na GPU.
  + **Why (Por quê):** O OX é inerentemente sequencial *intra-indivíduo* (preenchimento circular com verificação de presença), o que pode causar divergência de execução dentro dos grupos de threads. Operadores alternativos podem ser mais regulares (menos ramificações condicionais) e, portanto, mais adequados à execução paralela massiva.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [crossover_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/crossover_strategies.py#L22) (`OrderCrossover`, L22 — operador atual)
    - [crossover_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/crossover_strategies.py#L112) (`PartiallyMappedCrossover` PMX, L112 — já implementado, não utilizado)
    - [ga_fujimoto.cu](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu) (kernel CUDA do cruzamento OX)
  + **When (Quando):** Sequencialmente, após Ação 5.2.
  + **How (Como):** Levantar e documentar as seguintes alternativas:
    - **OX — Order Crossover (atual)** (`larranaga1999genetic`): 🟡 preenchimento circular sequencial dentro de cada indivíduo; entre indivíduos, cada par é independente e pode ser processado em paralelo.
    - **PMX — Partially Mapped Crossover** (`larranaga1999genetic`): 🟡 já implementado em python; utiliza mapeamento posicional com resolução de conflitos via cadeia de troca; padrão de acesso à memória mais regular que OX.
    - **CX — Cycle Crossover** (`larranaga1999genetic`): 🔴 detecção de ciclos é inerentemente sequencial e com profundidade variável; pouco adequado à GPU.
    - **EAX — Edge Assembly Crossover** (`nagata2013powerful`): 🔴 constrói subgrafos a partir de arestas dos pais; extremamente eficaz em qualidade (estado da arte para TSP) mas altamente complexo e sequencial; difícil de paralelizar.
    - **GPX — Generalized Partition Crossover** (`whitley2010hybrid`): 🟡 baseado em particionamento de grafo; estrutura mais regular; potencial para paralelização via decomposição de componentes.
  + **How much (Quanto):** Esforço Médio (análise teórica aprofundada + documentação).
  + **Classificação de Paralelismo:** 🟡 Parcialmente Paralelizável — cada cruzamento individual é processável em paralelo (nível de população), mas a lógica intra-indivíduo contém laços condicionais dependentes de dados que causam divergência nos grupos de execução.
  + **Referências disponíveis em refs.bib:**
    - `larranaga1999genetic` — survey abrangente de operadores de cruzamento para TSP (OX, PMX, CX, ERX)
    - `nagata2013powerful` — Edge Assembly Crossover (EAX), estado da arte em qualidade
    - `whitley2010hybrid` — Generalized Partition Crossover (GPX), resultados competitivos com pop. pequena
    - `fujimoto2011highly` — cruzamento OX em GPU (implementação base deste trabalho)
  + **DoD (Definition of Done):** Tabela comparativa dos operadores de cruzamento (complexidade, regularidade de acesso, adequação GPU) documentada no relatório ou como TODO de trabalho futuro.

- [ ] **Ação 5.4: Mutação — 🟢 Paralelizável**
  + **What (O quê):** Investigar alternativas ao *Swap Mutation* (taxa 0,02) atualmente utilizado. Documentar no relatório as opções, seus efeitos na diversidade e adequação à GPU.
  + **Why (Por quê):** O Swap Mutation é o operador mais simples (troca duas posições), com perturbação mínima na rota. Operadores como inversão (*Inversion Mutation*) ou inserção (*Insertion Mutation*) produzem perturbações de escala variável, podendo escapar de ótimos locais mais eficientemente. Além disso, a taxa fixa de 0,02 pode ser insuficiente em estágios avançados de convergência.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [mutation_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/mutation_strategies.py#L23) (`SwapMutation`, L23 — operador atual)
    - [mutation_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/mutation_strategies.py#L104) (`InversionMutation`, L104 — já implementado, não utilizado)
    - [mutation_strategies.py](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/strategies/mutation_strategies.py#L184) (`InsertionMutation`, L184 — já implementado, não utilizado)
  + **When (Quando):** Sequencialmente, após Ação 5.3.
  + **How (Como):** Levantar e documentar as seguintes alternativas:
    - **Swap Mutation (atual)** (`larranaga1999genetic`): 🟢 troca duas posições aleatórias; $O(1)$; totalmente paralelizável (cada indivíduo é mutado independentemente).
    - **Inversion Mutation** (`larranaga1999genetic`): 🟢 inverte um segmento aleatório; $O(n)$ no pior caso; já implementado; produz perturbações mais estruturadas que swap.
    - **Insertion Mutation** (`larranaga1999genetic`): 🟢 remove e reinsere uma cidade em posição aleatória; $O(n)$; já implementado; perturbação intermediária.
    - **Scramble Mutation** (`eiben2015introduction`): 🟢 embaralha um segmento aleatório; perturbação forte; bom para diversificação.
    - **Taxa adaptativa** (`eiben2011parameters`): taxa de mutação que aumenta com a estagnação da população; potencialmente mais eficaz que taxa fixa.
  + **How much (Quanto):** Esforço Baixo (documentação + avaliação teórica).
  + **Classificação de Paralelismo:** 🟢 Paralelizável — todos os operadores de mutação são independentes por indivíduo; operações são simples e regulares sem divergência de execução significativa.
  + **Referências disponíveis em refs.bib:**
    - `larranaga1999genetic` — survey com análise detalhada de operadores de mutação para TSP
    - `eiben2015introduction` — operadores de mutação e auto-adaptação de parâmetros
    - `eiben2011parameters` — métodos de ajuste adaptativo de taxa de mutação
    - `goldberg1989genetic` — papel da mutação na manutenção da diversidade
  + **DoD (Definition of Done):** Documentação das alternativas com classificação de paralelismo e recomendação de qual operador investigar primeiro em trabalho futuro.

- [ ] **Ação 5.5: Busca Local (Melhoria) — 🟡 Parcialmente Paralelizável**
  + **What (O quê):** Investigar alternativas ao 2-opt de iterações fixas atualmente utilizado como busca local no algoritmo memético. Documentar no relatório as opções, suas complexidades, qualidade de soluções e adequação à GPU.
  + **Why (Por quê):** O 2-opt com iterações fixas é a busca local mais simples ($O(n^2)$ por iteração); buscas mais agressivas como 3-opt, Or-opt ou Lin-Kernighan (LK) produzem soluções significativamente melhores. O estado da arte (LKH-3) atinge gaps de $0\%$ na maioria das instâncias TSPLIB. Avaliar essas alternativas é essencial para entender o teto de qualidade alcançável.
  + **Who (Quem):** Agente (IA) + Autor.
  + **Where (Onde):**
    - [ga_fujimoto.cu](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu) (kernel `two_opt_device`)
    - [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) (TODOs Pendentes + Seção 6.1)
  + **When (Quando):** Sequencialmente, após Ação 5.4.
  + **How (Como):** Levantar e documentar as seguintes alternativas:
    - **2-opt (atual, iterações fixas)** (`croes1958method`): 🟡 $O(n^2)$ por iteração; entre indivíduos é independente (🟢), mas cada avaliação de troca requer varredura sequencial do par $(i,j)$.
    - **Or-opt** (`larranaga1999genetic`): 🟡 move segmentos de 1, 2 ou 3 cidades; menor complexidade que 2-opt; adequado como complemento.
    - **3-opt** (`lin1973efficient`): 🔴 $O(n^3)$ por iteração; qualidade significativamente superior ao 2-opt, mas custo computacional proibitivo para instâncias grandes em GPU sem otimizações.
    - **Lin-Kernighan (LK)** (`helsgaun2000effective`): 🔴 busca local variável ($k$-opt dinâmico); estado da arte em qualidade; altamente sequencial e dependente de dados (decisões locais dependem de resultados anteriores); dificilmente paralelizável num único indivíduo.
    - **Guided Local Search (GLS)** (`voudouris1999tsp`): 🟡 penaliza arestas frequentes para escapar de ótimos locais; complementar ao 2-opt; potencialmente paralelizável se cada indivíduo mantiver seu vetor de penalidades.
    - **2-opt GPU otimizado** (`tsp_gpu`): 🟢 implementações existentes de 2-opt massivamente paralelo em GPU (Rocki & Suda, 2013); cada thread avalia um par $(i,j)$ diferente simultaneamente.
    - **K-opt GPU** (`mast2017traveling`): 🟡 análise de escalabilidade de K-opt em GPU; resultados promissores para $k \leq 3$.
  + **How much (Quanto):** Esforço Alto (análise teórica aprofundada, é a fase mais impactante na qualidade das soluções).
  + **Classificação de Paralelismo:** 🟡 Parcialmente Paralelizável — a busca local de cada indivíduo é independente (🟢 nível de população), mas a avaliação intra-indivíduo envolve dependências sequenciais (especialmente em buscas do tipo LK). O 2-opt pode ser paralelizado por pares $(i,j)$ dentro de cada indivíduo.
  + **Referências disponíveis em refs.bib:**
    - `croes1958method` — 2-opt original para o TSP
    - `lin1973efficient` — Lin-Kernighan: 3-opt e $k$-opt variável
    - `helsgaun2000effective` — LKH-3: implementação eficiente de Lin-Kernighan; estado da arte
    - `voudouris1999tsp` — Guided Local Search (GLS) aplicado ao TSP
    - `tsp_gpu` — 2-opt massivamente paralelo em GPU (Rocki & Suda)
    - `mast2017traveling` — análise de K-opt em GPU
    - `fujimoto2011highly` — implementação base do 2-opt em GPU usada neste trabalho
    - `tsutsui2011fast` — ACO + 2-opt em GPU (abordagem híbrida)
  + **DoD (Definition of Done):** Tabela comparativa das buscas locais (complexidade, qualidade típica, adequação GPU) documentada no relatório ou como TODO de trabalho futuro. Indicação clara de qual alternativa investigar prioritariamente.
