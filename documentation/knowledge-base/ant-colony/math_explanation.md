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
