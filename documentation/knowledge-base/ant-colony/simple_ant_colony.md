7
1 Introdução
O consumo eciente da bateria de um nó sensor em uma rede de sensores semo é de suma
importância, visto que este pode estar instalado em um local de difícil acesso. O alto consumo
do emissor de rádio comparado aos demais componentes de um sensor[2], torna o roteamento
um ponto importante ao se considerar este problema.
O foco doSACRA(Simple Ant Colony Routing Algorithm) é realizar o roteamento de
dados utilizando os recursos energéticos da rede de forma inteligente e com o mínimo possível
de overhead no tráfego de pacotes. Para isto ele utiliza a técnica "Ant Colony Optimization".
A principal motivação para o desenvolvimento deste trabalho é contribuir com o desenvol-
vimento de redes de sensores semo energeticamente ecientes, uma vez que esta tecnologia
é aplicada para o desenvolvimento de pesquisas em diversas áreas como , por exemplo,"Inter-
net Of Things"[3], que constitui o futuro da disponibilização e troca de dados sem intervenção
humana pela internet. Além disso, a característica de eciência energética doSACRApode
contribuir muito com o desenvolvimento desmart buildings, visto que estes também buscam tal
eciência. Portanto, dadas as possíveis aplicações para RSSF as contribuições de seus algorit-
mos de roteamento vão além de sua importância para as redes em si.
Em seguida são apresentados os objetivos geral e especícos do trabalho, bem como, uma
descrição de sua estrutura.
1.1 Objetivo Geral
Construir um algoritmo ciente de energia para redes de sensores semo, capaz de oferecer
uma boa taxa de entrega de pacotes, com pouco overhead de pacotes de controle e um consumo
de energia baixo e balanceado.+
8
1.2 Objetivos Especícos
O algoritmo deve ser capaz de fornecer uma taxa de entrega de pacotes que garanta, ao
menos, um minimo de qualidade da rede.
Os níveis médios de consumo de energia devem ser inferiores aqueles vericados nos de-
mais algoritmos utilizados para comparação em condições de igualdade de ambientes e con-
gurações.
1.3 Metodologia
Conhecer diferentes algoritmos de roteamento paraRSSF’s am de obter informações a
respeito de abordagens já utilizadas neste contexto para atacar o problema da minimização do
consumo de energia.
Prototipar o algoritmo, focando em simplicidade e idéias criativas e ecientes para auxiliar
no balanceamento do consumo de energia.
Comparar o algoritmo com outros já conhecidos por meio de simulações. Além disso,
realizar uma análise dos dados obtidos.
1.4 Estrutura do Trabalho
O texto está organizado da seguinte maneira: o capítulo 2 mostra um visão geral deswarm
intelligencee descreve a técnicaant colony optimization. O capítulo 3 traz uma descrição de al-
guns algoritmos conhecidos em redes de sensores semo que serão utilizados nas comparações
com oSACRA. O capítulo 4 descreve o algoritmo proposto, suas tabelas de rotas e mecanis-
mos utilizados para realizar o roteamento visando a otimização do consumo de energia dos nós
da rede. O capítulo 5, a respeito dos resultados obtidos em simulações, traz uma comparação
entre o algoritmo proposto e os demais algoritmos descritos no capítulo 3. Os capítulos 6 e 7
apresentam, respetivamente, as conclusões obtidas com o trabalho realizado e as propostas de
trabalhos futuros para a melhoria doSACRA.+
9
2 Ant Colony Optimization
Swarm Inteligenceé uma ferramenta de inteligência articial baseada no comportamento
de enxames de insetos e outros grupos de animais. Ela baseia-se na interação entre os agentes
e o seu ambiente para a obtenção de um objetivo. Por não possuir uma estrutura de controle
centralizada, e ter como foco a obtenção de um comportamento inteligente global do sistema[1],
esta técnica adequa-se perfeitamente as necessidades de um algoritmo de roteamento para redes
de sensores semo onde se busca um balanceamento do consumo de energia global da rede.
Este capítulo descreve a técnica conhecida comoant colony optimizationque, dentre as
técnicas deSwarm Intelligence, é a que mais se destaca na aplicação em redes de sensores sem
o. O principal aspecto de destaque desta é permitir a inserção de heurísticas a serem utilizadas
para determinar o que é ou não um bom caminho a ser escolhido. Isto permite a consideração
da energia na escolha de rotas o que serve perfeitamente aos objetivos deste trabalho.
Grassé usa o termo estigmergia para denir o processo de comunicação indireto mediado
pelo ambiente onde os indivíduos estão inseridos[4]. Algumas espécies de formigas se comu-
nicam modicando o ambiente ao seu redor com o intuito de chegarem a um objetivo comum.
Informações stigmérgicas podem ser acessadas apenas localmente, ou seja, elas só interferem
no comportamento do indivíduo que o está acessando ou de sua vizinhança mais próxima.
Ant Colony Optimization é inspirada no comportamento de algumas espécies de formiga
em busca de alimento. Essas formigas depositam feromônios no chão com o intuito de marcar
o caminho da toca até o alimento enquanto o procuram. Ao encontrar o alimento a formiga
volta pela trilha de feromônios que ela previamente deixou depositado no chão. Neste processo
de volta até a toca ela segue depositando seus feromônios para manter a trilha ativa. As novas
formigas que partirão da toca perceberão que já existe uma ou mais trilhas partindo dali e
seguirão por uma delas também depositando feromônios na trilha que escolheram percorrer. Os
feromônios depositados evaporam ao longo do tempo, assim as trilhas que não são utilizadas
tendem a desaparecer. Quanto maior o número de formigas caminhando por uma trilha, maior
será a quantidade de feromônios depositados na mesma. As formigas se sentirão mais atraídas+
10
pelas trilhas com maiores níveis de feromônios e consequentemente estas trilhas acumularão
cada vez mais feromônios e se tornarão mais atraentes a elas. Isso faz com que o menor caminho
até o alimento seja naturalmente escolhido e de forma inconsciente por parte dos agentes, no
caso as formigas, isto porque a formiga que retornar primeiro à toca após encontrar o alimento,
acaba tornando sua trilha mais rica em feromônios atraindo novas formigas para ela.
O comportamento das formigas é um exemplo claro de como se dá a comunicação através
da interação com o ambiente e como esta interação leva a resolução de um problema, no caso,
a busca por comida.Uma meta-heurística é em síntese um método heurístico utilizado para re-
solver de forma genérica um problema de otimização. Trata-se de um método computacional
que busca uma solução iterativamente, considerando uma solução candidata a cada iteração e
realizando continuas tentativas de aprimoramento desta solução, onde é utilizada uma medida
de qualidade previamente denida para determinar o quão boa é a solução encontrada em uma
dada iteração[5].Ant Colony Optimization, é uma meta-heurística, e como tal, dene um mé-
todo capaz de resolver uma série de problemas através da realização de relativamente poucas
modicações na base do método. Um modelo P para um problema de otimização combinatória
pode ser denido como:
P= (S,Ω,f)
onde:
•Sé o espaço de busca, denido com um conjuntonito de variáveis discretas de decisão
Xi,i=1,...,n;
•Ωé um conjunto de restrições sobre as variáveis;
•fé uma função objetivof:S→R +, a ser minimizada ( ou maximizada tomandog=−f
como função objetivo ). Uma variávelX i toma valores num conjunto
Di={v(i,j)|j=v 1
i ,...,v |Di|
i }
Uma possível soluçãos∈Sconsiste na atribuição de valores a todas as variáveisXsatis-
fazendo as restrições contidas emΩ.
Uma soluçãos ∗ ∈Sé uma solução ótima global se, e somente se:
f(s∗)≤f(s)∀s∈S+
11
O modeloPé utilizado para denir o modelo de feromônios emant colony optimization.
Um valor de feromônioτ i j é dado a cada variável envolvida em uma possível soluçãoc i j do
problema, o que consiste na atribuição:
Xi =v i j
O conjunto de todas as possíveis soluções é denotado porC.
Dado um grafoG c(V,A), ondeVé um conjunto de vértices eAé um conjunto de Arestas.
Este grafo é denido a partir dos componentes de soluçãocque podem ser representados tanto
pelos vértices quanto pelas arestas deG. Formigas articiais buscarão uma solução caminhando
vértice a vértice através das arestas do grafo e construindo incrementalmente uma solução par-
cial. Durante este processo elas depositarão feromônios nos componentesc, fazendo com que
a quantidade de feromônios∆ τ depositada em um componente seja dependente da qualidade
da solução encontrada. Com isso as demais formigas utilizarão as quantidades de feromônios
já determinadas como um ponto de partida para sua tomada de decisão em relação as melhores
áreas a percorrer dentro do espaço de buscaS.
A meta-heurísticaant colony optmizationconsiste em uma fase de inicialização seguida por
iterações que, por sua vez, são divididas em duas fases principais.
Na fase de inicialização são setados os parâmetros do algoritmo como, por exemplo o fator
de evaporação do feromônio, e inicializar o feromônio das trilhas.
Dentro de cada iteração existe a fase de construção de soluções e a fase de atualização dos
ferômonios.
Algoritmo 1:Metaheurística Ant Colony Optimization
whilecondição de parada não satisfeitado
ContruirSoluções;
AtualizarFeromônios;
end
Na fase de construção, um conjunto de m formigas articiais constrói suas soluções a partir
de um conjuntonito de variáveis em:
C={c i j|i=1,...,n∧j=1,...,|D i|}
O Ponto de partida para a construção de uma solução é uma solução parcial vazias p =/ 0.
Dene-seN(s p)∈Ccomo o conjunto dos componentes que podem ser adicionados a atual+
12
solução parcials p sem violar nenhuma das restrições deΩ. A soluçãos p inicialmente vazia
vai sendo incrementada a cada iteração através da adição de uma possível solução presente no
conjuntoN(s p).
Na fase de atualização dos feromônios são realizadas duas operações:
1. Subtração dos valores de feromônios segundo a equação que rege a evaporação dos mes-
mos utilizando um valor que indica o fator de evaporação.
2. Adição de feromônios aos valores associados a boas soluções, ou seja, trilhas percorridas
por formigas na iteração corrente[6].
A escolha de uma possível solução presente emN(s p)é guiada por um mecanismo esto-
cástico regulado pelos feromônios relacionados a cada elementoN(s p). O mecanismo pode
variar para cada algoritmo onde é aplicada a técnica, porém, deve reetir algum modelo do
comportamento real de formigas como o visto em [6].+
13
3 Algoritmos de Roteamento
Este capítulo tem por objetivo fornecer uma breve análise de alguns pontos inerentes a redes
de sensores semo e uma descrição básica dos principais aspectos de alguns algoritmos que
serão citados e utilizados para comparação ao longo do trabalho.
Uma das questões a ser tratada para cada rede de sensores semo, é a aplicação a qual a
mesma servirá. Assim, diferentes algoritmos de roteamento poderão se comportar de maneira
adequada, ou não, dependendo da aplicação a ser considerada. Porém, alguns aspectos são
comuns a maioria das aplicações em redes de sensores semo. Entre eles estão:
•Recursos limitados de memória, processamento e energia.
•Topologia dinâmica devido a mobilidade. e possibilidade de falhas dos nós.
•Possibilidade de difícil manutençao, fazendo com que o tempo de vida da rede seja um
ponto crítico a ser considerado para sua implantação.
Tendo em vista estas características, podemos concluir que alguns dos principais pontos a
serem buscados por um algoritmo de roteamento para redes de sensores semo são[rotsensores]:
•Minimizar o consumo de energia.
•Distribuir tarefas de roteamento entre os nós.
•Ser tolerante a falhas.
3.1 AODVjr
O Algoritmo de roteamentoAODVjr(AODV simplicado)[7], trata-se de uma simplica-
ção do muito estudado[8] e utilizadoAODV(Ad-hoc On Demand Distance Vector)[9]. Este
protocolo se enquadra na família dos protocolos reativos. Nesta categoria as rotas são formadas+
14
sob demanda, ou seja, o protocolo só faz uma busca por rotas quando lhe é solicitado enviar
dados a um destino para o qual ele não conhece um caminho. OAODVjr, modica alguns as-
pectos doAODVtradicional am de diminuir a quantidade de pacotes de controle e simplicar
os mecanismos de descoberta e manutenção de rotas.
Este algoritmo retira os sequintes itens da especicação doAODVoriginal[9]:
•Números de sequencia.
•RREP’s desnecessários.
•Contagem de saltos.
•Mensagens "HELLO".
•Mensagens de falha de link (RERR).
•Lista de precursores.
O mecanismo de descoberta de rotas realiza o tradicionaloodde pacotes RREQ solici-
tando uma rota para o destino desejado. Para se livrar da necessidade de manter números de
sequencia para garantia da ausencia de loops de roteamento, o algoritmo faz com que apenas
os destinos das RREQ’s possam responder com um RREP. Como nenhum nó intermediário po-
derá responder a um RREQ, os RREP’s "gratuitos" também são eliminados. Além disso, toda
descoberta de rotas, gera rotas bidirecionais[7].
A manutenção das rotas é feito por meio de um tempo de vida pré determinado para uma
rota. Este tempo de vida só poderá ser atualizado através do recebimento de pacotes e nunca
pelo envio. Desta forma, é necessário que o nó destino envie algum pacote para o nó origem de
uma rota periodicamente, caso contrário, a mesma se tornará inválida aom de seu tempo de
vida. Caso a comunicação seja unidirecional, o nó destino envia pacotes "CONNECT" para a
origem da rota, am de atualizá-la. No caso em que a comunicação é bidirecional, as rotas são
atualizadas sem a necessidade de um pacote extra o que implica em um overhead de pacotes de
controle ainda menor.
Caso o nó origem deixe de receber dados ou "CONNECT" do nó destino, após algum
tempo sem ser atualizada, a rota será retirada da tabela de rotas do nó origem. Este mecanismo
é eciente em determinar uma quebra de rota, visto que neste caso o nó origem deixa de receber
informações do nó destino. Desta forma, as mensagens "RERR" não são mais necessárias.
Após a eliminação de uma rota, caso o nó origem deseje enviar dados para o mesmo destino,
ele dará inicio a um novo processo de descoberta de rotas.+
15
Este algoritmo tem um foco muito grande na simplicidade, e diversos aspectos doSACRA
foram inspirados nas simplicações por ele apresentadas.
3.2 AOER
OAOER(Ant-based On-demand Energy Route Protocol)[8] também é caracterizado como
um algoritmo reativo. Além disso este algoritmo pertence a família dos algoritmos cientes
de energia. Este tipo de algoritmo é caracterizado por buscar um uso eciente dos recursos
energéticos dos nós de uma rede buscando aumentar seu tempo de vida. Este algoritmo utiliza
ant colony optimizationpara alcançar seus objetivos de otimização do consumo de energia dos
nós.
Na fase de descoberta de rotas, são enviados pacotes de requisição denominados"forward
ants"am de encontrar uma rota eciente até o destino. Para isto o pacote formiga carrega
informações a respeito da energia restante do último nó visitado, além de outras métricas, como
a média de energia restante dos nós por onde ela já passou. Enquanto percorre os nós a formiga
faz depósitos de feromônios proporcionais ao quanto a participação de um dado nó em uma
rota, pode torná-la energeticamente eciente. Estes valores de feromônios, por sua vez, são
armazenados em tabelas de feromônios inversas. O termo "inversas" se referem ao fato de estas
tabelas apontarem caminhos para o nó origem da rota a ser descoberta.
Ao receber uma"forward ant"o nó destino envia uma"backward ant". Esta percorre o ca-
minho de volta até o nó origem decidindo quais nós intermediários utilizará baseada nas tabelas
de feromônios previamente preenchidas. A escolha do próximo destino em cada ponto da rota,
é realizada de forma probabilistica, onde quanto maior a quantidade feromônios associada a um
dado destino, maior a probabilidade desta rota ser escolhida pela formiga.
Um mecanismo particularmente interessante deste algoritmo ocorre quando um nó recebe
uma"forward ant". Caso a energia restante deste nó esteja muito abaixo da média de energia
calculada pela formiga recebida, este nó não retransmitirá esta formiga, impedindo que ele seja
escolhido para estabelecimento de uma rota. Apesar de interessante para o aumento do tempo
de vida da rede e distribuição uniforme do consumo de energia entre os nós, é possível que este
mecanismo tenha um impacto negativo na taxa de entrega de dados. Isto pode ocorrer no caso
em que o nó que se recusa a aceitar uma rota, seja a única alternativa para o estabelecimento da
mesma. Indícios deste comportamento podem ser vericados nas análises realizadas em [10]
e nos resultados obtidos neste trabalho. OSACRAutiliza um mecanismo parecido com este,
porém, propõe uma importante modicação que permite minimizar o impacto desta decisão na+
16
taxa de entrega de pacotes.
A manutenção de rotas noAOERé semelhante a utilizada noAODVjr. A principal dife-
rença reside na capacidade do algoritmo de determinar momentos oportunos para forçar uma
redescoberta de rotas baseado na energia consumida por um nó enquanto serve uma rota. Isto
permite que, periodicamente, a comunicação entre dois nós, utilize diferentes nós intermediá-
rios, distribuindo melhor o consumo de energia.
3.3 ADHOP
ADHOP(Ant-based Dynamic hops Optimization Protocol) é um protocolo reativo que
busca eciencia na taxa de entrega de dados assim como a minimização do overhead no tráfego
de pacotes da rede[10]. Para atingir seus objetivos, também é utilizada a técnicaACO.
Neste algoritmo as rotas não são pre-determinadas. Os saltos de uma rota vão sendo se-
lecionados dinamicamente em cada nó até o destino, o que facilita a adaptação as mudanças
constantes de topoliga de uma rede móvel. Além disso, todo pacote de dados é uma formiga,
ou seja, ainda que uma formiga necessite explorar uma área para encontrar o destino, uma vez
localizado, o pacote de dados é entregue.
Para sua operação o algoritmo utiliza dois tipos de pacotes (formigas). São eles:
•Exploratory Transport:reponsável por buscar o destino de um pacote quando, em um
dado nó da rota, não é possível encontrar um próximo salto. As formigas exploratórias
avançam através dos nós buscando o destino ou um nó que já possua uma trilha para o
destino. Assim elas possuem a função de descobrir uma rota ou religar trilhas (rotas)
evaporadas.
•Forward Transport:Carrega os dados utilizando as rotas e reforçando as trilhas de
feromônios das rotas deixadas pelas formigas exploratórias.
Além disso o algoritmo permite a aplicação de heuristicas para otimizar um determinado
aspecto da rede. Em [11] podemos ver resultados obtidos com a aplicação de heurísticas relaci-
onadas a otimização da energia ou da latência na comunicação.+
17
4 O algoritmo SACRA: Simple Ant
Colony Routing Algorithm
Este capítulo descreve o algoritmoSACRA(Simple Ant Colony Routing Algorithm), suas
estruturas e mecanismos de prevenção de loops destacando os principais pontos relacionados
ao consumo eciente de energia da rede. Como o próprio nome sugere, o algoritmo tem foco
na simplicidade. O que se pretende é utilizar mecanisnos simples, porém, que apresentem
potencial na tentativa de otimizar o consumo de energia da rede.
4.1 Tabelas de rotas
O algoritmo proposto possui duas tabelas de rotas. Uma delas é utilizada para armazenar
as rotas inversas, utilizadas na primeira fase da descoberta de rotas, enquanto a outra armazena
as rotas efetivas por onde trafegarão os dados.
A tabela de rotas inversas é especialmente importante para o algoritmo, visto que é nela que
estarão armazenados os valores de feromônios a serem utilizados na aplicação deant colony
optimizationdurante a descoberta de rotas.
O uso de uma tabela de rotas inversas, bem como, a utilização de um campotime to livenas
rotas efetivas foi inspirado pelo algoritmoAODVjr[7], anteriormente descrito.
A seguir encontra-se uma descrição completa dos campos de cada tabela:
4.1.1 Tabela de rotas inversas
Denição 1Sejam A e B dois nós da rede e A deseja enviar dados para B. Uma rota inversa é
denida como o caminho de retorno de B para A a ser percorrido por uma backward ant am
de estabelecer uma rota bidirecional entre A e B.
Cada entrada na tabela de rotas inversas (tabela 4.1) possui os seguintes campos:+
18
•Destino: Armazena o identicador do nó que iniciou a descoberta de rotas.
•Próximo: Para cadaDestinohá uma lista de um ou mais vizinhos que podem ser es-
colhidos para constituir uma rota efetiva. Estes vizinhos são identicados pelo campo
Próximo. Cada par (destino,próximo), identica uma rota inversa.
•Feromônios: A cada vizinho identicado comoPróximoé associado um valor de fe-
romônios, armazenado neste campo.
•Saltos: Armazena o menor número de saltos registrado para o destino.
Destino Próximos Saltos
A
Próximo Feromônios
A 130023
B 10989
D 11985
1
H
Próximo Feromônios
F 21565
G 1054
X 36775
5
Tabela 4.1: Exemplo entradas na tabela de rotas inversas.
4.1.2 Tabela de rotas
Cada entrada na tabela de rotas efetivas (tabela 4.2) possui os seguintes campos:
•Destino: Armazena o identicador do pontonal da rota efetiva, ou seja, o nó que rece-
berá os dados enviados pelo nó que solicitou uma rota.
•Próximo: Armazena o identicador do próximo nó que um pacote deve visitar am de
alcançar o destino por meio desta rota.
•TTL: Signica"time to live"e trata-se de um timeout para determinar se uma rota conti-
nua ativa. O campo é decrementado continuamente e, ao ser zerado, a rota é removida da
tabela de rotas. Este campo é utilizado pelo mecanismo de manutenção de rotas, desem-
penhando um papel importante em diversos aspectos do algoritmo.+
19
Destino Próximo TTL
D B 5
X G 8
C F 2
Tabela 4.2: Exemplos de entradas na tabela de rotas.
4.2 Descoberta de rotas
Am de manter a simplicidade, decidiu-se utilizarant colony optimizationapenas na fase
de descoberta de rotas. Assim esta fase é a responsável pela otimização do consumo de energia.
Quando um determinado nó deseja enviar dados a um outro nó da rede para o qual ele não
possui uma rota ativa, é iniciado um processo de descoberta de rotas am de obter o melhor
caminho até o destino desejado. Nesta fase há um intensa troca de informações entre todos os
nós da rede, devido aooodingde pacotes de requisição de rota. A aplicação deant colony
optimizationpermite que isto seja realizado com pouco overhead de pacotes de controle.
Após o estabelecimento das rotas inversas a técnica utilizada permite que os nós possuam
consciência, ainda que de forma indireta, de qual vizinho possível para o estabelecimento de
uma rota é o que possui maior quantidade de energia. Detalhes a respeito da obtenção esta
"consciência" serão explicados ao longo da descrição do algoritmo nas próximas seções.
Os pacotes envolvidos nas descobertas de rotas são os seguintes:
•Forward ant(tabela 4.3): partem do remetente estabelecendo rotas rumo a este. São
elas ainda que carregam as informações relevantes para os mecanismos de prevenção de
loops. A estrutura de uma forward ant é a seguinte:
– Origem: Identicação do nó que inicia a descoberta de rotas, ou seja, aquele que
deseja enviar dados.
– ReqId: Número que junto ao campo origem, identica unicamente uma requisição
de rotas.
– Destino: Identicação do nó para o qual a origem deseja estabelecer uma rota.
– Rank: Valor que determina o quanto um nó está apto a receber uma rota de dados.
Este é o valor mais importante da heurística aplicada para o consumo inteligente dos
recursos energéticos da rede. Este campo deve estar fortemente ligado a informações
como a quantidade de energia restante do nó.
– Último: Identicação do último nó visitado pelaForwad Ant.+
20
– Saltos: O número de nós visitados pelafantem uma rota. Este campo é incremen-
tado sempre que a fant é recebida por um nó.
Origem ReqId Destino Rank Último Saltos
Tabela 4.3: Estrutura de uma forward ant
•Backward ant(tabela 4.4): Seguem do destino para o remetente utilizando as rotas es-
tabelecidas pelasforward ants. Elas selecionam probabilisticamente as melhores rotas
armazenadas nas tabelas de rotas inversas, estabelecendo assim as rotas efetivas entre os
nós. A estrutura de uma backward ant é a seguinte:
– Origem: Identicação do nó que lançou a backward ant, ou seja, o nó destino do
processo de descoberta de rotas.
– Destino: Identicação do nó destino da backward ant, ou seja, o nó que iniciou a
descoberta.
– Último: Identicação do último nó visitado pelaBackward Ant.
Origem Destino Último
Tabela 4.4: Estrutura de uma backward ant
A partir deste ponto trataremos as forward ants e backward ants pelas abreviaçõesfante
bantrespectivamente.
O desao desta etapa é gerar o maior número possível de rotas inversas criadas pelasfants,
para que asbantspossuam um bom número de alternativas de rotas. Além disso, estas rotas
devem ser livres de loops de roteamento.
O mecanismo esolhido para evitar os loops é baseado na contagem de saltos das rotas
inversas geradas pelas fants e será explicado em detalhes mais a frente.
4.2.1 Envio de uma Forward Ant
O processo de descoberta de rotas inicia no momento em que um nó deseja enviar dados a
um destino para o qual não possui uma rota ativa. O nó incrementa o seu contador de requisições
e cria uma fant preenchendo seus campos da seguinta forma:
•Origem: Seu identicador.+
21
•ReqId: o valor de seu contador de requisições interno.
•Destino: O identicador do nó para o qual deseja enviar dados.
•Rank: Seu valor de Rank dado pela equação 4.1.
•Último: Seu próprio identicador.
•Saltos: 0.
Em seguida o nó envia fant (gura 4.2.1) a todos os seus vizinhos 1.
Figura 4.1: O nó origem (em verde) envia uma fant am de estabelecer uma rota para o nó
destino (em laranja).
4.2.2 Recebimento de uma Forward ant
Ao receber umaforward ant, o nó encontra-se no ponto crucial para a determinação do
maior número possível de rotas inversas sem loops. Após incrementar o número de saltos da
fant o nó decide se deve ou não adicionar uma rota em direção ao nó origem e apontando para
o último nó visitado, a sua tabela de rotas inversas.
Os critérios para que se adicione uma possível rota inversa são os seguintes:
1. O nó ainda não possui nenhuma rota inversa para a origem da fant.
1por vizinhos, entenda-se todos os nós capazes de receber uma mensagem enviada por broadcast a partir de um
dado nó.+
22
2. Já existe uma rota inversa para a origem, a tabela de próximos saltos da rota existente
ainda não possui o nó identicado pelo campoÚltimoe o contador de saltos dafanté
menor ou igual ao campoSaltosda rota. Isto garante que as rotas inversas sejam livres
de loop.
3. O nó atual é o destino da fant. O nó destino pode adicionar todas as rotas visto que ele
é o pontonal da fant e como as rotas até o ponto imediatamente anterior são livres de
loops, não é possível que seja gerado um loop a partir do destino.
Após analizar a rota percorrida pela fant e determinar se ela deve ser inserida na tabela de
rotas inversas, o nó verica se já existe uma rota com a mesma assinatura, ou seja, partindo
da mesma origem e chegando até o nó atual através do mesmo nó vizinho. Neste caso, a rota
existente recebe um incremento em sua quantidade de feromônios, tornando-se mais atraente
a uma bant. Caso trate-se de uma rota nova, ela é inserida contendo a mesma quantidade de
feromônios que receberia como incremento no caso anterior. Este valor é dado pela seguinte
equação:
τi =ωr i −γs (4.1)
Sendoτ i a quantidade de feromônios a ser depositado na rota cujo próximo salto é o nói,
ωo peso relacionado ao valor deRankcarregado pelafantrecebida,r i é oRankcalculado no
nói,γo peso que o número de saltos deve ter no valor deτeso número de saltos carregado
pelafant, assim, caso seja desejado que as rotas com menor número de saltos sejam preferidas,
este valor deve ser incrementado.
Após esta etapa, é vericado se a fant já foi recebida e processada pelo nó atual. Isso é
possível armazenando uma lista dos pares (origem, reqId) atendidos recentemente e vericando
se o par da fant recebida já existe nesta lista. Qualquer fant recebida que já tenha sido processada
será descartada neste ponto. A cada nova requisição recebida, os pares de requisições anteriores
podem ser excluídos am de poupar memória.
Aqui age o primeiro mecanismo responsável pela distribuição do consumo de energia. Caso
o nó atual seja apenas um intermediário para o estabelecimento da rota, é vericado se este
encontra-se em um estado de "baixa energia". O estado citado é determinado com base em um
threshold congurável que indica o limite a partir do qual a quantidade de energia restante é
considerada pouca. Quando neste estado, o nó possui setada umaag global que o impede de
processar a primeira requisição recebida em um processo de descoberta. Após negar o processa-
mento de uma fant devido a estaag, a mesma é resetada de forma a permitir o processamento+
23
da próxima requisição, o que é necessário pois este nó pode ser a única opção para o estabele-
cimento da rota desejada.
Este mecanismo tem a intenção de dicultar a formação de rotas passando por um nó que
possui pouca energia restante. A eciência deste procedimento reside no fato de que a negação
do estabelecimento de algumas rotas inversas passando por este nó, aumenta as chances de que
uma rota mais longa, porém com mais energia, seja escolhida.
Aag ainda é setada novamente para negar o processamento de requisições em dois mo-
mentos:
1. Quando o nó detecta a evaporação completa de alguma trilha de feromônios, indicando
que um processo de descoberta de rotas já foi terminado a um certo tempo.
2. Quando o nó recebe uma bant, pois neste caso uma rota já foi estabelecida passando pelo
nó.
Em um nó intermediário, o processamento de uma fant desempenha um papel crucial na
aplicação da heurística de energia aplicada emant colony optimization. Isto porque, antes de
ser reenviada a todos os vizinhos por broadcast, o campoRankda fant é atualizado com os
dados do nó atual. O valor aplicado a este campo é dado pela seguinte equação:
Rank=E−λ|R| (4.2)
SendoEa energia restante,Ro conjunto das rotas efetivas eλo peso relativo a penalidade
doRankem relação ao número de rotas efetivas dado por|R|.
Como o valor de Rank da fant reete na quantidade de feromônios da rota inversa apon-
tando para o nó atual, a utilização da quantidade de energia na equação faz com que nós com
maior quantidade de energia se tornem mais atraentes para uma bant. Pode-se perceber ainda
que o valor recebe uma penalização relacionada a quantidade de rotas efetivas que o nó já está
atendendo. A intenção desta penalidade é fazer com que nós que já possuem um grande nú-
mero de rotas passando por eles, se tornem menos atraentes e consequentemente possuam uma
probabilidade menor de serem selecionados para uma rota efetiva.
Além da atualização do"rank"da fant, ela recebe o identicador do nó atual em seu campo
Último.
O processo de envio de fants e formação de rotas inversas, é ilustrado pelagura a seguir:+
24
Figura 4.2: Processo de envio de fants e formação de trilhas de feromônios.
Ao chegar ao nó destino a fant encontra seu pontonal. Ao receber a primeira destas, o
nó destino inicia um timer para o envio de uma bant. A intenção de atrasar o envio de uma
bant é permitir que outras fants cheguem até o destino e, desta forma, fornecer um número
maior de opções de rotas no momento em que a bant iniciar sua jornada até a origem para o
estabelecimento de uma rota efetiva.
Aom da contagem do timer, o nó gera então umabantcom os seguintes valores em seus
campos:
•Origem: O identicador no nó destino dafant, ou seja, o nó atual.
•Destino: Recebe o campo Origem dafant.
•Último: Recebe o identicador do nó atual.
As próximas seções mostram como é feito o envio daemphbant e como as rotas efetivas
são estabelecidas. No entanto é importante entender, anteriormente, como abantseleciona os
melhores nós para compor uma rota.
4.2.3 Escolha da Melhor Rota Inversa
Com a tabela de rotas inversas devidamente preenchida nos passos anteriores, abantpossui
agora diversas opções de rotas a seguir. A cada passo ela deve decidir qual o melhor caminho
escolhendo, estocasticamente, uma entre todas as rotas possíveis até o seu destino.+
25
Figura 4.3: Uma bant no nói, decidindo qual seu próximo destino. O nó que possui maior
quantidade de feromônios na tabela deve possuir maior probabilidade de ser escolhido.
Para aumentar as probabilidades de que a trilha com maior concentração de feromônios
seja selecionada, é utilizado uma estratégia de seleção chamada "seleção proporcional pseudo-
randômica". A escolha desta estratégia é baseada na escolhida pelo algoritmoAOERdescrito
em [8].
Abantanalisa a rota cujo campoDestinocoincide com seu campo homônimo. A lista de
próximosdesta rota, constitui o conjunto dos possíveis próximos saltos dabant.
SejaNo conjunto de todos os nós vizinhos ao nó atual que podem ser escolhidos como
próximo salto rumo ao nó destino eτ n,n∈N, o valor que representa a quantidade de feromônios
associada ao nó vizinhon. A probabilidadep j de uma rota ser escolhida, ondejé o próximo
salto da mesma, é dada pela seguinte equação:
pj = τ j
∑
i∈N
τi
Uma vez calculadas as probabilidades, é sorteado um número pseudo-aleatórioαno inter-
valo[0,1]e o próximo saltoké selecionado através dos seguintes critérios:
k=



n∈N∧n=max
i∈N
τi,α<p n
m,caso contrário
Ondemé selecionado pelo método da roleta, descrito na sequência.
Dene-seA, como o conjunto das probabilidades acumuladas para cada nón∈N. Onde a
probabilidade acumuladaade um nóiser escolhido é dada por:+
26
ai = ∑
k∈N
k<i
pk,i∈N
O critério de ordenação dos nós é baseado na ordem que os mesmos aparecem na tabela
de rotas inversas que, por sua vez, é determinado pela ordem de inserção na mesma. Uma vez
denido o conjuntoA, o nómpodenalmente ser selecionado como:
m=j∈N|a j ∈A∧a j >α
A utilização deste método para a seleção dekpermite que a rota com o maior valor de
feromônios depositados tenha prioridade de escolha entre as demais rotas. Esta escolha, aliada
a forma como o valor dos feromônios é composto (vide equações (4.1) e (4.2)) favorece a
escolha dos nós com maior quantidade de energia restante e menor número de rotas já ativas
passando por eles.
4.2.4 Envio de uma Backward Ant
Aom do timer no destino da rota desejada, abantinicia sua jornada rumo ao ponto que
originou a descoberta de rotas. O nó deve, baseado nas informações armazenadas nas tabelas de
rotas inversas, decidir probabilisticamente o próximo passo dabant. O mecanismo responsável
por decidir qual rota abantdeve seguir será descrito na seção 4.2.3.
Após selecionar qual a rota inversa desejada, o nó adiciona uma entrada em sua tabela de
rotas efetivas referente a direção doDestinoàOrigemdos dados. Esta rota é referente a uma
das direções da rota bidirecional que será formada.
•Destino: O identicador do nó que consta comoDestinodabant.
•Próximo: O identicador do próximo salto da rota inversa selecionada.
•TTL: Recebe um valor congurável referente ao tempo que uma rota deve ser mantida
ativa, ainda que não utilizada.
Uma vez que a rota efetiva tenha sido adicionada, o nó aramazena seu identicador no
campoúltimodabante, em seguida, a envia por unicast ao vizinho fornecido como próximo
salto na rota inversa selecionada anteriormente.+
27
O processo de envio de umabanté o mesmo para todos os nós intermediários da rota a ser
estabelecida.
4.2.5 Recebimento de uma Backward Ant
Ao receber umabanto nó adiciona em sua tabela de rotas efetivas uma entrada referente ao
caminho que será estabelecido partindo do nó que deu origem a descoberta de rotas em direção
ao nó destino desejado. A rota criada recebe os seguintes dados:
•Destino: O identicador do nó que consta comoOrigemdabant.
•Próximo: O identicador do nó que enviou abantao nó atual. Valor obtido através do
campoúltimoda formiga.
•TTL: Recebe o valor de ttl conforme explicado na descrição da rota formada na seção
4.2.4.
Em um nó intermediário abanté encaminhada ao próximo salto conforme descrito na
seção 4.2.4.
Agura a seguir ilustra o caminho de uma bant até o no origem:
Figura 4.4: Caminho de uma bant. Asechas azuis representam as rotas possíveis de serem
escolhidas em cada nó, as verdes representam aquelas que foram efetivamente escolhidas pela
bant. Mais a esquerda, temos a rota efetiva estabelecida aom do processo.
A chegada ao nó apontado como destino dabant, ou seja, o nó que iniciou o processo de
descoberta de rotas, marca om deste processo. Neste ponto o nóOrigempossui uma rota ativa
para o envio de dados ao destino desejado, assim como oDestinopossui uma rota passando
pelos mesmos nós intermediários até o nóOrigem. A necessidade desta rota será explica na
seçãoManutenção de Rotas.+
28
4.3 Manutenção de Rotas
4.3.1 Rotas Inversas
Rotas inversas são abstrações das trilhas de feromônios deixadas pelas formigas enquanto
buscam seu "alimento", ou seja, o nó destino de uma transmissão de dados. Dessa forma, elas
se mantém ativas enquanto houver feromônios depositados nela. No entanto, os feromônios
possuem um fator de evaporação, o que faz com que seu valor diminua em uma trilha com
o passar do tempo. Para simular este fenômeno, a cada determinado período, todas as rotas
inversas tem seus valores de feromônios atualizados pela seguinte atribuição:
τi ←(1−ρ)τ i
Sendoρo fator de evaporação dos feromônios que também pode ser um valor congurável
no intervalo[0,1].
4.3.2 Rotas Efetivas
Uma vez estabelecidas as rotas, os mecanismos responsáveis por buscar a distribuição e-
ciente do consumo de energia atuam de forma pró ativa utilizando informações locais, evitando
a necessidade de troca de informações entre os nós e, consequentemente, mantendo baixo o
overhead de pacotes de controle. O alicerce do processo de manutenção de rotas foi baseado no
mecanismo utilizado pelo algoritmo AODVjr [7].
O campoTTLde todas as rotas efetivas é decrementado periodicamente. Assim que este
campo é zerado, a rota referente a ele é considerada inválida e retirada da tabela de rotas.
As rotas em direção a origem são mantidas através dos pacotes de dados enviados por ela.
Cada vez que um nóarecebe um pacode de dados de um nób, o nóaatualiza o campoTTL
de todas as rotas cujó campoPróximocorresponde ao nób, atribuindo a este, seu valor inicial.
Estas rotas são utilizadas pelo destino para o envio de pacotes de vericação de conexão entre
origem e destino. Chamaremos estes pacotes deConnect.
No sentido do destino para origem, as rotas são mantidas através do recebimento deCon-
nects. Ao receber um destes pacotes o nó realiza o mesmo procedimento do caso dos pacotes
de dados, a diferença entre os dois surge no momento de enviar o pacote ao próximo destino da
rota. O encaminhamento de umConnectpode ser negado para que a rota para o destino seja
quebrada, forçando um novo processo de descoberta e, consequentemente, a possibilidade do+
29
estabelecimento de uma nova rota com maior quantidade de energia restante.
O funcionamento básico deste sistema é mostrado nagura a seguir:
Figura 4.5: Manutenção de rotas efetivas do SACRA.
Cada nó possui um segundothresholdcongurável que especica o quanto de energia este
nó está "autorizado" a consumir a partir do último momento em que for estabelecida uma rota
passando por ele. Para que esta vericação seja possível, o nó armazena o estado atual de sua
bateria sempre que umabanté recebida. Assim, quando um pacoteConnecté recebido, o
nó o processa devidamente e somente o encaminha ao próximo destino caso seja satisfeita a
condição:
Echecagem −E atual <threshold
Ou seja, caso o consumo de energia desde a última checagem até o momento atual seja
maior que o treshold denido, o nó deixa de encaminhar osconnectspois ele já consumiu uma
boa quantidade de energia servindo esta ou mais rotas. Visto que são estes pacotes que atualizam
ottldas rotas utilizadas pelos pacotes de dados e que o nóOrigemdeixará de recebê-los, logo
a rota para o nó destino será invalidada e retirada da tabela de rotas da origem provocando uma
redescoberta caso este ainda deseje enviar dados ao destino.
Este mecanismo trabalhando junto ao processo de descoberta de rotas, é o principal res-+
30
ponsável pela distribuição do consumo de energia pelos nós da rede uma vez que ele promove
a renovação periódica das rotas, permitindo assim, queant colony optimizationentre em ação
selecionando os melhores nós.
A modicação do valor atribuído a estethreshold, bem como ao aplicado na negação do
processamento de umafant(ver seção 4.2.2), podem modicar signicamente o comporta-
mento do algoritmo. Caso congurado com um valor baixo para othresholdda checagem de
energia, o algoritmo provocará mais descobertas de rotas, consumindo mais energia da rede
como um todo, porém distribuindo melhor este consumo entre os nós. Por outro lado, se este
valor for alto, serão provocadas menos descobertas, porém, um nó poderá ter grande parte de
sua energia consumida enquanto serve a uma rota, o que prejudica a distribuição do consumo de
energia. Um dos desaos da conguração deste algoritmo é buscar o melhor equilíbrio possível
entre estes parâmetros.
Vale ressaltar que esta negação de encaminhamento de mensagens é realizada apenas com
os pacotesconnect, mantendo a entrega de dados até a exaustão do ttl da rota que corre em
sentido oposto a dosconnects. Procedendo desta maneira, busca-se que o impacto deste proce-
dimento na taxa de entrega de dados seja o mínimo possível.
É possível perceber que este mecanismo entrará em ação apenas caso a comunicação seja
unidirecional, ou seja, apenas uma das pontas da rota envie dados a outra. Não pode-se negar
o encaminhamento de pacotes de dados pois isto geraria um impacto negativo na taxa de en-
trega de pacotes o que não é desejado. Sendo assim, ambas direções terão seus valores dettl
atualizados enquanto durar a troca bidirecional de dados.
Para que não sejam enviados pacotesconnectdesnecessários, cada nó guarda uma lista dos
identicadores para os quais eles enviam dados. Assim osconnectsão enviados apenas para os
destinos da tabela de rotas que não estão presentes nesta lista.+
31
5 Resultados de Simulações
Este capítulo mostra os resultados obtidos com oSACRAem simulações realizadas em
dois simuladores de redes, oSinalgoe oOmnet++. Cada simulador foi utilizado em uma fase
distinta do desenvolvimento do algoritmo. A primeira fase foi marcada pela necessidade de
construir um protótipo que fornecesse facilidade para obeservação do comportamento do algo-
ritmo. Neste contexto oSinalgoserviu perfeitamente com suas ferramentas grácas altamente
intuitivas. A segunda fase, por sua vez, trouxe o amadurecimento do algoritmo. Um trabalho
realizado já com bases mais sólidas devido ao conhecimento emRSSF’s obtido na primeira
fase. Assim, para a validação dos resultados pretendidos com oSACRA, fez-se necessária a
utilização de um ambiente mais robusto para testes, o qual foi fornecido peloOmnet++.
5.1 Primeira Fase - Sinalgo
OSinalgoé um poderoso framework de simulação para testes e validações de algoritmos
de rede [12]. Escrito em java, este simulador permite a prototipação de algoritmos nesta lingua-
gem, fornecendo uma interface gráca amigável para vericação visual do comportamento do
algoritmo.
O tempo noSinalgoé medido emroundsde simulação. Este simulador não fornece qual-
quer abstração de um sistema energético e também não leva em consideração aspectos como
as colisões de pacotes em uma rede, porém, a possibilidade de acompanhar visualmente e de
forma extremamente intuitiva a troca de dados entre os nós, faz deste um ótimo simulador para
iniciar o desenvolvimento de um algoritmo de roteamento.
Para as simulações foi congurado um ambiente em duas dimensões de 500x500 metros.
A aplicação considerada consiste em uma estação base recebendo dados de outros 9 nós da
rede, totalizando assim, 10 nós comunicantes distribuídos de forma que o nó base estivesse no
centro do ambiente e os 9 nósorigemformassem um "círculo" ao redor dele. Os demais nós
da redes foram uniformemente distribuídos formando uma grade. O algoritmo utilizado para a+
32
comparação de resultados foi oAODVjr.
O envio de dados por parte dos 9 nós é realizado a cada 15 rounds de simulação enquanto
o nó base envia umconnecta cada 90 rounds. O ttl foi congurado para 120 rounds.
Asguras a seguir mostram 3 cenários distintos de uma simulação do algoritmo proposto:
Agura 5.1 temos a disposição dos nós antes do início da simulação, agura 5.2 mostra as
rotas inversas deixadas porfantslogo após um processo de descoberta de rotas e agura 5.3 as
rotas efetivas formadas e a troca de dados ocorrendo 1.
Figura 5.1: Disposição dos nós para o inicio das simulações. Os nós desalinhados da grade são
aqueles que realizam comunicação.
1Foram realizadas apenas simulações de ambientes estáticos nesta fase.+
33
Figura 5.2: Os 9 nós comunicantes já receberam bants e acabam de estabelecer suas rotas para
o destino. A imagem ilustra o grande número de rotas inversas presentes, demostrando a grande
quantidade de opções para o estabelecimento de rotas efetivas.
Figura 5.3: Após a evaporação completa das rotas inversas (trilhas de feromônios), restam as
rotas efetivas entre os nós comunicantes, estas mantidas por um mecanismo de timeout (time to
live) e dinamicamente modicadas através dos mecanismos anteriormente descritos.
Para simular o consumo de energia, foi atribuído a cada nó um valor de energia inicial de-+
34
crementado a cada mensagem recebida ou enviada. Este mesmo valor foi utilizado nas equações
onde se fazia necessário informações a respeito da energia restante de um nó. Para os pacotes
de dados, foi sorteado um número aleatório que determina o seu tamanho, valor este, que foi
utilizado para determinar a quantidade de bateria a ser consumida para este tipo de pacote.
Para uma comparação justa foram utilizados os mesmos valores de consumo por mensagem na
implementação doAODVjre doSACRA.
Um dos aspectos mais importantes a se considerar em uma rede de sensores semo é o
"tempo de vida" da rede, denido em [8] como o tempo decorrido até que um primeiro nó tenha
sua bateria completamente consumida. Este aspecto foi medido em rounds do simulador para
diferentes números de nós e foram obtidos os resultados mostrados nagura 5.4.
Como esperado, quanto maior o número de nós da rede, maior a eciência doSACRA
em relação aoAODVjr. Isto ocorre pois o aumento do número de nós em um mesmo espaço
implica em um maior número de rotas inversas formadas na primeira fase da descoberta. Assim
umabantpossui mais opcões de rotas a seguir, melhorando a distribuição do consumo de
energia entre os nós.
Figura 5.4: Tempo de vida
Outros dois aspectos vericados nas simulações foram a média global de energia restante
na rede e o desvio padrão da quantidade de energia dos nós em relação a média. Este é par-
ticularmente importante por ter relação com a variância da quantidade de energia entre os nós+
35
da rede, ou seja, é uma medida que reete a distribuição do consumo em relação a média. As
guras 5.5 e 5.6 ilustram os resultados obtidos.
Figura 5.5: Média de energia da rede após 50000 rounds de simulação
Figura 5.6: Desvio padrão da energia dos nós após 50000 rounds de simulação
Quanto menor o desvio padrão apresentado, melhor a distribuição da energia entre os nós da
rede. Dado que oAODVjr, em um cenário onde um nó só falha caso sua bateria seja esgotada,+
36
vai manter este nó em uma rota até que sua bateria seja completamente consumida, a distribui-
ção da energia tende a ser bastante prejudicada. Esse fato foi comprovado na comparação entre
os algoritmos, onde oSACRAapresentou uma distribuição melhor para diferentes quantidades
de nós compondo a rede.
5.2 Segunda Fase - Omnet++
A segunda fase de simulações foi realizada no simulador omnet++. Escrito emC++, ele
permite a implementação de algoritmos nesta linguagem. Foi utilizado o framework “inet”
implementado para o simulador. Este framework fornece todo o necessário para realizar simu-
lações de redes de sensores semo.
Nesta fase do desenvolvimento, as simulações foram realizadas em um cenário muito mais
realista am de vericar o comportamento do algoritmo quando sujeito a todas as variáveis
de uma RSSF real como colisões de pacotes e mobilidade dos nós. Todo os parâmetros de
energia do framework foram congurados para que o consumo representasse de forma adequada
o consumo de um EPOSMote[13], um nó sensor real desenvolvido no LISHA/UFSC.
Neste simulador, fatores presentes em redes reais são simulados com grandeza de detalhes.
É possível simular ambientes críticos, com grande taxa de perda de pacotes. Este tipo de cenário
é interessante para vericar a eciência do algoritmo simulado em situações extremas.
Para comparações, foram utilizados os algoritmosAODV,AOEReADHOP. Nesta fase,
tembém foram analisados os resultados de média global de consumo de energia e desvio padrão
da energia dos nós. A novidadeca por conta da análise da taxa de entrega de pacotes presente
nesta fase.
5.2.1 Redes Móveis
Para as simulações de redes móveis foi congurada uma área de 1000m x 1000m e a posição
inicial dos nós foi determinada aleatoriamente. A comunicação é realizada por meio de 10 nós
enviando dados para outros 10 a cada 4 segundos. O tempo de simulação neste caso foi de 1500
segundos e o tempo de espera para o envio de uma bant no nó destino foi de 100ms. Para as
rotas efetivas, foi congurado um TTL de 10 segundos e os pacotes connect congurados para
serem enviados a cada 6 segundos.
O esquema de mobilidade utilizado foi o“MassMobility”. Neste modelo um nó se mo-
vimenta em linha reta durante um determinado tempo escolhido aleatoriamente segundo uma+
37
mobilityType "MassMobility"
changeInterval truncnormal(7s, 5s)
changeAngleBy normal(0deg, 15deg)
speed truncnormal(5mps, 1mps)
waitTime 7s
updateInterval 100ms
Tabela 5.1: Parâmetros de conguração do esquema de mobilidade
distribuição normal e, aom deste período, ele muda de direçao e segue seu movimento reti-
líneo. A alteração no sentido do movimento também é determinada randomicamente seguindo
uma distribuição normal. Os parâmetros utilizados para o modelo de mobilidade são apresenta-
dos na tabela 5.1:
Os dados obtidos para a taxa de entrega de dados neste cenário de grande mobilidade podem
ser visualizados nagura 5.7:
Figura 5.7: Taxa De Entrega de Pacotes - Mobilidade
O cenário criado para as simulações apresenta grande mobilidade e uma taxa de envio de
pacotes relativamente alta. Dessa forma as baixas taxas de entrega de pacotes apresentadas
peloSACRAe oAOERpodem ser justicadas, em parte, pela grande quantidade de rotas
quebradas devido a mudança constante de posição dos nós. As taxas superiores apresentadas
peloADHOPsão alcançadas por sua mudança no paradigma tradicional de manter rotas pré
denidas, adaptando-se com mais eciência ao deparar-se com uma falta de rotas conhecidas+
38
para um destino. Além disso, as formigas exploratórias doADHOPcarregam pacotes de da-
dos para serem entregues. Este comportamento garante maior probabilidade de um pacote ser
entregue, porém, tem um impacto negativo no consumo de energia, uma vez que estes paco-
tes tendem a ser retransmitidos e multiplicados diversas vezes enquanto procuram um caminho
para o destino. Este impacto pode ser conferido nagura 5.8.
Outro fator agravante para a taxa de entrega de dados em um cenário de grande mobilidade
é a utilização, na camadaMAC, do padrão802.15.4. Este padrão não foi necessariamente pro-
jetado para obter um comportamento ótimo em redes com nós móveis. A presença de enlaces
assimétricos aumentam com a mobilidade, o que traz sérios problemas para o sucesso do estebe-
lecimento de rotas. Além disso, o aumento da densidade de uma rede, aumenta a incidência de
nós escondidos, implicando em um maior número de colisões de pacotes e, consequentemente,
a diminuição nas taxas de entrega de pacotes[14][15].
Os dados obtidos para a média de consumo global de energia para este cenário podem ser
visualizados nagura 5.8.
Figura 5.8: Média Global de Consumo - Mobilidade
É possível perceber que oSACRAmanteve as melhores médias de consumo em relação
aos demais algoritmos. Além disso, a tendência ao aumento do consumo com o aumento da
densidade da rede, apesar de apresentado por todos os algoritmos testados, é menor noSACRA.
Visto que, quanto maior o número de nós da rede, maior o número de pacotes de requisição+
39
de rotas que serão retransmitidos, este resultado nos dá indícios de que o algoritmo proposto
consegue consumir menos energia que os demais durante a realização dosoodsde descoberta
de rotas. Os pacotesfant(requisição de rota) doSACRAforam concebidos para carregar o
mínimo de dados possível para, além do estabelecimento de rotas, aplicarACOsobre o consumo
de energia. Levando em consideração que a quantidade destes pacotes trafegando em uma rede
durante esta fase constitui uma enorme fatia de tudo o que trafega em uma rede, esta decisão
mostrou sua eciência com o resultado acima.
Finalmente, o desvio padrão do consumo de energia dos nós pode ser visualizado no gráco
a seguir:
Figura 5.9: Desvio Padrão do Consumo de Energia - Mobilidade
Os baixos valores para o desvio padrão obtidos peloAOERpodem ser justicadas pela
baixa entrega de pacotes. O baixo desempenho em obter sucesso na entrega de um pacote,
indica que este algoritmo gerou uma grande quantidade de requisições de rotas, que por serem
realizadas poroodingde pacotes, fazem com que todos os nós da rede consumam quantidades
parecidas de energia, diminuindo assim, a discrepância entre os valores de consumo de energia
dos nós. Sendo assim, os resultados obtidos peloSACRApara a distribuição do consumo de
energia, podem ser considerados positivos, uma vez que o algoritmo obteve valores bastante
próximos daqueles registrados peloADHOPe conseguiu taxas de entrega signicativamente
maiores que oAOER.+
40
5.2.2 Redes Estáticas
Para a simulação de redes estáticas, foi congurada uma área de 1000mx1000m. Os nós
foram distribuídos de forma aleatória por esta área. Para a comunicação, foram escolhidos 10
nós da rede incumbidos de enviar dados para outros 10. Os nós "origem" enviam um pacote de
dados a cada 3 segundos para seus "destinos". O tempo de simulação para todos os cenários
foi de 2000 segundos e o tempo de espera para o envio de uma bant no nó destino foi de
100ms. Para as rotas efetivas, foi congurado um TTL de 10 segundos e os pacotes connect
congurados para serem enviados a cada 8 segundos.
Para a taxa de entrega de pacotes, oSACRAapresentou resultados bastante positivos,-
cando a frente dos demais algoritmos na grande maioria dos cenários simulados. Estes resultado
podem ser observados nagura 5.10.
Figura 5.10: Taxa De Entrega de Pacotes - Nós Fixos
A queda de desempenho na taxa de entrega, apresentada no cenário com 120 nós na rede,
pode ser justicada pelo aumento do overhead de pacotes de controle. O aumento da densidade
da rede, contribui para o aumento de colisões de pacotes, provocando quebras de rotas e perda
dos próprios pacotes envolvidos nas descobertas. Uma análise dos resultados obtidos para o
overhead de pacotes de controle (vergura 5.11), pode reforçar estas armações.+
41
Figura 5.11: Overhead de Controle - Nós Fixos
Agura 5.12 mostra os resultados obtidos para a média de consumo global da rede:
Figura 5.12: Média de Consumo Global - Nós Fixos
OSACRAmostra um ótimo desempenho para diferentes números de nós na rede. Além+
42
disso, é possível vericar um consumo bastante controlado como o aumento da quantidade de
nós, ao passo que, os demais algoritmos apresentaram uma tendência mais acentuada de au-
mento do consumo de energia nesta mesma situação. Este fato comprova a eciência alcançada
com a diminuição da quantidade de dados trafegados durante osoodsde descoberta de rotas.
Os resultados obtidos para o desvio padrão são apresentados pelagura 5.13.
Figura 5.13: Desvio Padrão - Nós Fixos
Para esta medida de desempenho, oSACRAdemonstrou uma taxa pouco pior do que os
demais algoritmos analisados para alguns casos, o que indica uma menor eciência em distribuir
o consumo entre os nós da rede. No entanto, a pequena diferença mostra que o impacto não é
suciente para considerar esta distribuição ruim. Este fator pode ser compensado pelo excelente
desempenho na média de consumo global.
Ainda nagura 5.13, pode-se perceber uma excessão ao comportamento geral doSACRA.
Em simulações com 100 nós compondo a rede, o algoritmo se comportou melhor do que os
demais algoritmos testados na distribuição do consumo de energia. Com este número de nós,
o algoritmo também obteve uma ótima taxa de entrega de pacotes (vergura 5.10). Estes dois
fatos somados, nos permitem chegar a conclusão de que as rotas formadascaram bastante
distribuídas, balanceando o consumo de energia e consequentemente reduzindo o número de
colisões de pacotes, o que contribuiu positivamente para a taxa de entrega.+
43
As taxas de entrega de dados apresentadas, aliado ao excelente consumo de energia, fazem
doSACRA, uma ótima alternativa para aplicações de RSSF’s estáticas onde o tempo de vida
dos nós da rede constituem um fator crítico.+
44
6 Conclusões
Neste trabalho foi apresentado oSACRA, um algoritmo de roteamento energeticamente
eente para redes de sensores semo. Para alcançar tal eciência, foi aplicado a técnica deant
colony optimization, trabalhando junto a outros mecanismos intimamente ligados a quantidade
de energia e número de rotas servidas por cada nó da rede. Todos os detalhes do algoritmo
foram escolhidos de forma a apresentarem eciência aliada a simplicidade de implementação.
Foram estudados diversos algoritmos para RSSF, principalmente aqueles considerados ci-
entes de energia. Foram identicadas diversos aspectos interessantes em cada um. Alguns de-
talhes presentes noSACRAforam inspirados no funcionamento destes algoritmos conhecidos,
o que mostra a grande importância deste levantamento para a realização do trabalho.
A prototipação do algoritmo foi realizada com o auxilio do simulador de redesSinalgo,
que mostrou ser uma ótima ferramenta para este tipo de atividade. O suporte visual fornecido
pela ferramenta foi essencial para auxiliar na compreensão das particularidades envolvidas no
desenvolvimento de um algortimo de roteamento. Durante a criação do protótipo foram de-
nidos todos os mecanismos utilizados na composição doSACRA. O prótotipo ainda passou
por uma validação, através de uma comparação com o algoritmoAODVjr. Foi utilizada uma
abstração bastante simples para simular o consumo de energia com base nos pacotes de dados e
roteamento enviados e recebidos. Neste cenário oSACRAapresentou resultados positivos para
para a distribuição do consumo de energia, apesar de o consumo de energia em si ter se mos-
trado muito próximo ao obtido peloAODVjr. A melhor distribuição do consumo energético no
SACRAmostrou, ainda, seu impacto positivo no tempo de vida da rede. Estes resultados de-
monstraram que o protótipo construído atendeu as expectativas de maximizar o tempo de vida
dos nós de uma RSSF.
A técnicaant colony optimizationmostrou ser bastante eciente na aplicação da heurística
de energia presente no algoritmo, fornecendo um mecanismo poderoso capaz de ser implemen-
tado sem a necessidade de qualquer alteração no número de pacotes de controle planejado. Os
dados referentes a aplicação da técnica, carregados por uma formiga de requisição de rotas no+
45
SACRA, foram reduzidos a um único campo, minimizando o impacto da técnica na quantidade
de dados efetivamente transmitidos.
A simplicidade alcançada no desenvolvimento doSACRA, foi posteriormente testada em
um ambiente mais realista, através do simuladorOmnet++. Com este simulador foi possível
vericar o comportamento do algoritmo sujeito aos efeitos da mobilidade dos nós, colisões de
pacotes e sua consequente perda, e os efeitos da camadaMACem algoritmos de roteamento.
Para a validação do algortimo nesta fase, foram realizadas comparações com os algoritmos
AOER,ADHOPeAODV. Os resultados obtidos para o consumo médio de energia foram
extremamente positivos, alcançando a eciência energética desejada.
As taxas de entrega de pacotes em ambientes de alta mobilidade mostraram-se satisfató-
rias se comparadas aoAOER. A comparação com oADHOPneste cenário, mostram que o
paradigma tradicional de manter rotas pré-denidas para a comunicação, utilizado noSACRA,
mostrou não ser tão interessante quanto o roteamento baseado apenas no próximo salto, utili-
zado noADHOP. Em ambientes com nósxos, este quesito mostrou resultados excelentes.
A análise dos resultados obtidos com simulações e a comparação do algoritmo proposto
com outros já conhecidos em redes de sensores semo, permitem concluir que o objetivo
central de construir um algoritmo simples e capaz de obter eciência energética no roteamento
foi alcançado satisfatoriamente, transformando oSACRAem um algoritmo promissor para
aplicações onde esta variável seja de grande importância. Como exemplo de uma tecnologia que
normalmente faz uso de redes de sensores semo cuja eciência energética seria grandemente
valorizada, podemos citar ossmart buildings, que possuem entre suas principais características,
a busca pelo consumo eciente de energia.+
46
7 Trabalhos Futuros
Alguns aspectos analisados ao longo do desenvolvimento doSACRA, permitem identicar
possíveis melhorias que contribuiriam ainda mais para a melhoria da taxa de entrega de pacotes
e a redução do consumo de energia. Entre elas pode-se destacar a aplicação de algum meca-
nismo de controle para ooodde pacotes de requisição utilizado na fase de descoberta de rotas
e a mudança da reação a quebras de rotas, fazendo com que estas possam ser reparadas ao invés
totalmente redescobertas.
Permitir que um nó intermediário seja capaz de solicitar uma reparação de rotas, e que um
nó que já possua uma rota para o destino desejado responda a uma requisição de rotas, faria
com que o algoritmo pudesse reutilizar "partes" de rotas quebradas. Esta modicação pode ter
impacto positivo na taxa de entrega de pacotes uma vez que a reconstrução de uma rota pode
ser signicativamente menos custoso para a rede, com relação a pacotes de controle, quando
comparada a uma descoberta completa de rotas.
A respeito do controle deood, a principal contribuição seria na redução do consumo de
energia. Umoodmassivo e sem controle de pacotes de requisição de rotas, torna-se proibitivo
em redes de grande densidade uma vez que uma parcela importante da energia de todos os nós
da rede é consumida durante este processo. Durante as pesquisas para o desenvolvimento do
SACRAforam analisados alguns mecanismos para o controle dooodingna rede. Um deles
mostrou-se particularmente promissor para a utilização junto comACO. O nome da técnica é
percolation drivenoode ela utiliza um mecanismo estocástico para a denição de quando um
pacote de requisição de rotas deve, ou não, ser retransmitido. A métrica originalmente utilizada
pela técnica, é baseada no número de vizinhos do nó que deve decidir a respeito da retrans-
missão do pacote[16]. Considerando que é possível utilizar outras métricas para determinar
a propabilidade de um pacote ser retransmitido, seria possível adicionar questões referentes a
energia neste calculo. O ponto negativo da especicação original da técnica é a adição de um
pacote de controle para a determinação do número de vizinhos de um nó, porém, se a métrica for
substituida por algo capaz de ser calculado localmente, este pacote tornaria-se desnecessário.+
