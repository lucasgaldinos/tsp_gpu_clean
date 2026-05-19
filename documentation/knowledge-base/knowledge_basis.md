# Knowledge Basis

## Table of Contents

1. [TCCs Lidos](#tccs-lidos)
2. [TCCs a Ler](#tccs-a-ler)
   - [Marcos Pereira - Simple Ant Colony Routing Algorithm](#marcos-pereira---simple-ant-colony-routing-algorithm)
     - [Resumo](#resumo)
     - [Swarm Intelligence](#swarm-intelligence)
     - [Heurísticas e Estigmergia](#heurísticas-e-estigmergia)
     - [Modelo Matemático](#modelo-matemático)
     - [Algoritmo AODVjr](#algoritmo-aodvjr)

## TCCs Lidos

- [TCC do Victor Hammann Pereira - Método computacional para definição de rotas baseadas no sistema de transporte coletivo.](./DESENVOLVIMENTO%20DE%20MÉTODO%20COMPUTACIONAL%20PARA%20DEFINIÇÃO%20DE%20ROTAS%20BASEADAS%20NO%20USO%20DO%20SISTEMA%20DE%20TRANSPORTE%20COLETIVO.pdf)

## TCCs a Ler

[Simple_Ant_Colony_Routing_Algorithm.pdf](./TCCs_consulta/ant_colony/monografia_marcos_pereira_simple_ant_colony.pdf)

### Marcos Pereira - Simple Ant Colony Routing Algorithm

#### Resumo

O diferencial do algoritmo proposto, frente a outros que fazem uso de ant colony optimization, é procurar a máxima simplificação para a aplicação da técnica sem que ela perca suas características. O aspecto a ser otimizado é o consumo de energia. Para isto é utilizada uma métrica intimamente ligada a energia restante dos nós ao longo de um caminho e ao número de rotas que cada nó já mantém ativas no momento da descoberta. De posse deste valor, as heurísticas aplicadas pela técnica buscam formar rotas de forma que a distribuição do consumo de energia entre os nós de uma rede seja a melhor possível. Além disso, o algoritmo determina proativamente, quando um nó consome uma quantidade considerável de energia servindo como intermediário de rotas de dados, forçando estas a serem movidas para caminhos alternativos e com melhores saldos de energia restante.

#### Swarm Intelligence

Swarm Intelligence é uma ferramenta de inteligência artificial baseada no comportamento de enxames de insetos e outros grupos de animais. Ela baseia-se na interação entre os agentes e o seu ambiente para a obtenção de um objetivo. Por não possuir uma estrutura de controle centralizada, e ter como foco a obtenção de um comportamento inteligente global do sistema[1], esta técnica adequa-se perfeitamente às necessidades de um algoritmo de roteamento para redes de sensores sem fio onde se busca um balanceamento do consumo de energia global da rede.

#### Heurísticas e Estigmergia

Inserção de heurísticas a serem utilizadas para determinar o que é ou não um bom caminho a seguir, considerando o nível de energia de cada nó. Segundo Grassé, estigmergia: Processo de comunicação indireto mediado pelo ambiente onde se está inserido.

#### Modelo Matemático

Uma meta-heurística é em síntese um método heurístico utilizado para resolver de forma genérica um problema de otimização.
$ P = (S, \Omega, f) $

- $ S $ é o espaço de busca (conjunto de soluções possíveis), definido com um conjunto de variáveis discretas de decisão $ X_i = 1,\dots,n $
- $ \Omega $ é o conjunto de restrições sobre cada variável
- $ f $ é a função objetivo, a ser minimizada

O modelo $ P $ é utilizado para definir o modelo de feromônios em ant colony optimization.

\[ f: S \to \mathbb{R}^+ \]

No texto, o autor utiliza a seguinte notação: $ D_i = \{v(i,j) \quad|\quad  j=v_i^1, \dots, v_i^{D_i} \} $ para representar o conjunto de valores para $ X $ possíveis.
$ i $ é o nó atual em $ S $. $ j $ é um nó vizinho de $ i $. $ v_i^k $ é o k-ésimo vizinho de $ i $. $ D_i $ é o número de vizinhos de $ i $. $ v(i,j) $ é o valor da função objetivo para a aresta $ (i,j) $.

#### Algoritmo AODVjr

Simplificação do algoritmo AODV (Ad hoc On-demand Distance Vector). Protocolo reativo: Rotas são formadas sob demanda. Isso quer dizer que o protocolo só busca por rotas quando lhe é solicitado enviar dados a um destino que o mesmo não conhece o caminho. AODVjr diminui simplifica a fim de diminuir quantidade de pacotes de controle. No AODVjr, são retirados:

1. Números de sequência
2. Pacotes de RREP desnecessários
3. Contagem de saltos
4. Mensagens "HELLO"
5. Mensagens de falha de link (RERR)
6. Lista de precursores

No AODVjr, para eliminar a necessidade de números de sequência para garantia da ausência de loops de roteamento, apensa os destinos das RREQs (Route Requests) podem responder com RREPs (Route Replies). Nós intermediários não respondem com RREPs, evitando estes loops. A descoberta gera rotas bidirecionais.

Também elimina a necessidade de RERRs pois a manutenção de rotas é feita por meio de um tempo de vida pré-determinado. O tempo de vida é só atualizado no recebimento de pacotes, nunca pelo envio. Se a rota destino não enviar pacotes periodicamente, a rota é descartada ao final do tempo de vida. Quando unidirecional, o nó destino envia pacotes de conexão através da rota de origem. Caso a comunicação seja bidirecional, as rotas são atualizadas sem a necessidade de um pacote extra, diminuindo o overhead.  
Assim, facilmente é possível obter uma quebra de rota eficientemente pois após o nó de origem deixar de receber informações do nó destino, a rota é descartada após determinado tempo, tornando mensagens RERR desnecessárias.

#### AOER - Ant-based On-demand Energy Route protocol

O algoritmo AOER é uma meta-heurística que utiliza o modelo de feromônios para resolver o problema de roteamento de energia em redes ad hoc.

1. Pertence à família dos algoritmos cientes de energia.
    1. Busca uso eficiente dos recursos energéticos dos nós $ \to\ \uparrow tempo\ de\ vida$
    2. Utiliza ACO para encontrar rotas de menor custo energético.
2. Pertence a classe dos algoritmos reativos.

##### Descoberta de rotas

$\text{Forward ants}\ \to\ \text{new route}$: O pacote formiga carrega informações a respeito da energia de cada nó, média de energia restante dos nós onde já passou, além de outras métricas. Este pacote faz depósitos de "ferõmonios" proporcionalmente a quanto aquela rota pode torná-la eficiente. Esses valores são armazenados inversamente, onde a rota aponta para o nó de origem. Quando certo nó destino recebe uma dessas $\text{forward ants}$, ele responde com uma $\text{backward ant}$, que decidirá quais rotas utilizará baseada nas tabelas de feromônios.

Nesse algoritmo, também há um mecanismo onde caso a $\text{forward ant}$ esteja com um nível energético muito abaixo da média de energia, este nó não envia uma $\text{backward ant}$ para o nó de origem. Pode causar impacto negativo na taxa de entrega de dados quando um nó destino se recusa a receber uma $\text{forward ant}$ com baixo nível de energia, mesmo esta sendo a única alternativa.

> :memo: Para resolver isso, pode ser implementado no AOER um mecanismo de $\text{ant rescue}$, onde o nó destino envia uma $\text{forward ant}$ com um nível de energia maior para um nó intermediário que possua um nível de energia maior que o nível de energia mínimo necessário para enviar uma $\text{backward ant}$.

As principais diferenças entre o AOER e o AODVjr é que o AOER utiliza o modelo de feromônios para encontrar rotas de menor custo energético, enquanto o AODVjr utiliza o modelo de estado de link para encontrar rotas de menor custo de saltos e na capacidade de determinar a quebra e redescoberta de nós baseado na energia consumida enquanto server uma rota, permitindo a comunicação entre dois nós utilizando diferentes nós intermediários.

_SACRA_ utiliza mecanismo semelhante.

#### ADHOP - Ant based Dynamics hops Optimization Protocol

1. Protocolo reativo
2. Rotas não são predeterminadas. Saltos são selecionados dinamicamente, até o nó destino. Faciita a adaptação de rotas em ambientes dinâmicos como a topologia de uma rede móvel. Ainda que a formiga necessite explorar uma área para encontrar o destino, assim que localizado, há a entrega de dados.
    1. $\text{formiga exploratória}$ avança através de nós buscando uma próxima rota buscando o destino ou religando e descobrindo novas rotas.
3. _Permite aplicação de heurísticas._ em certos aspectos da rede.

#### O SACRA - Simple Ant Colony Routing Algorithm

Algoritmo proposto: 2 tabelas de rotas, uma inversa e outra rotas efetivas.

- [Gustavo Guidoni, orientador Mayerle](./Um%20problema%20de%20VRP.pdf)
