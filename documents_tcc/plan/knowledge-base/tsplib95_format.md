# TSPLIB 95

Gerhard Reinelt  
Universität Heidelberg  
Institut für Angewandte Mathematik  
Im Neuenheimer Feld 294  
D-69120 Heidelberg  
<Gerhard.Reinelt@IWR.Uni-Heidelberg.DE>

TSPLIB is a library of sample instances for the TSP (and related problems) from various sources and of various types. Instances of the following problem classes are available.

- Symmetric traveling salesman problem (TSP)
  - Given a set of n nodes and distances for each pair of nodes, find a roundtrip of minimal total length visiting each node exactly once. The distance from node i to node j is the same as from node j to node i.
- Hamiltonian cycle problem (HCP)
  - Given a graph, test if the graph contains a Hamiltonian cycle or not.
- Asymmetric traveling salesman problem (ATSP)
  - Given a set of n nodes and distances for each pair of nodes, find a roundtrip of minimal total length visiting each node exactly once. In this case, the distance from node i to node j and the distance from node j to node i may be different.
- Sequential ordering problem (SOP)
  - This problem is an asymmetric traveling salesman problem with additional constraints. Given a set of n nodes and distances for each pair of nodes, find a Hamiltonian path from node 1 to node n of minimal length which takes given precedence constraints into account. Each precedence constraint requires that some node i has to be visited before some other node j.
- Capacitated vehicle routing problem (CVRP)
  - We are given n−1 nodes, one depot and distances from the nodes to the depot, as well as between nodes. All nodes have demands which can be satisfied by the depot. For delivery to the nodes, trucks with identical capacities are available. The problem is to find tours for the trucks of minimal total length that satisfy the node demands without violating truck capacity constraint. The number of trucks is not specified. Each tour visits a subset of the nodes and starts and terminates at the depot. (Remark: In some data files a collection of alternate depots is given. A CVRP is then given by selecting one of these depots.)

Except for the Hamiltonian cycle problems, all problems are defined on a complete graph and, at present, all distances are integer numbers. There is a possibility to require that certain edges appear in the solution of a problem.

---

## 1. The file format

Each file consists of a specification part and of a data part. The specification part contains information on the file format and on its contents. The data part contains explicit data.

### 1.1 The specification part

All entries in this section are of the form `<keyword>: <value>`, where `<keyword>` denotes an alphanumerical keyword and `<value>` denotes alphanumerical or numerical data. The terms `<string>`, `<integer>` and `<real>` denote character string, integer or real data, respectively. The order of specification of the keywords in the data file is arbitrary (in principle), but must be consistent, i.e., whenever a keyword is specified, all necessary information for the correct interpretation of the keyword has to be known.

Below we give a list of all available keywords.

#### 1.1.1 `NAME: <string>`

Identifies the data file.

#### 1.1.2 `TYPE: <string>`

Specifies the type of the data. Possible types are:

- `TSP` — Data for a symmetric traveling salesman problem
- `ATSP` — Data for an asymmetric traveling salesman problem
- `SOP` — Data for a sequential ordering problem
- `HCP` — Hamiltonian cycle problem data
- `CVRP` — Capacitated vehicle routing problem data
- `TOUR` — A collection of tours

#### 1.1.3 `COMMENT: <string>`

Additional comments (usually the name of the contributor or creator of the problem instance is given here).

#### 1.1.4 `DIMENSION: <integer>`

For a TSP or ATSP, the dimension is the number of its nodes. For a CVRP, it is the total number of nodes and depots. For a TOUR file it is the dimension of the corresponding problem.

#### 1.1.5 `CAPACITY: <integer>`

Specifies the truck capacity in a CVRP.

#### 1.1.6 `EDGE_WEIGHT_TYPE: <string>`

Specifies how the edge weights (or distances) are given. The values are:

- `EXPLICIT` — Weights are listed explicitly in the corresponding section
- `EUC_2D` — Weights are Euclidean distances in 2-D
- `EUC_3D` — Weights are Euclidean distances in 3-D
- `MAX_2D` — Weights are maximum distances in 2-D
- `MAX_3D` — Weights are maximum distances in 3-D
- `MAN_2D` — Weights are Manhattan distances in 2-D
- `MAN_3D` — Weights are Manhattan distances in 3-D
- `CEIL_2D` — Weights are Euclidean distances in 2-D rounded up
- `GEO` — Weights are geographical distances
- `ATT` — Special distance function for problems `att48` and `att532`
- `XRAY1` — Special distance function for crystallography problems (Version 1)
- `XRAY2` — Special distance function for crystallography problems (Version 2)
- `SPECIAL` — There is a special distance function documented elsewhere

#### 1.1.7 `EDGE_WEIGHT_FORMAT: <string>`

Describes the format of the edge weights if they are given explicitly. The values are:

- `FUNCTION` — Weights are given by a function (see above)
- `FULL_MATRIX` — Weights are given by a full matrix
- `UPPER_ROW` — Upper triangular matrix (row-wise without diagonal entries)
- `LOWER_ROW` — Lower triangular matrix (row-wise without diagonal entries)
- `UPPER_DIAG_ROW` — Upper triangular matrix (row-wise including diagonal entries)
- `LOWER_DIAG_ROW` — Lower triangular matrix (row-wise including diagonal entries)
- `UPPER_COL` — Upper triangular matrix (column-wise without diagonal entries)
- `LOWER_COL` — Lower triangular matrix (column-wise without diagonal entries)
- `UPPER_DIAG_COL` — Upper triangular matrix (column-wise including diagonal entries)
- `LOWER_DIAG_COL` — Lower triangular matrix (column-wise including diagonal entries)

#### 1.1.8 `EDGE_DATA_FORMAT: <string>`

Describes the format in which the edges of a graph are given, if the graph is not complete. The values are:

- `EDGE_LIST` — The graph is given by an edge list
- `ADJ_LIST` — The graph is given as an adjacency list

#### 1.1.9 `NODE_COORD_TYPE: <string>`

Specifies whether coordinates are associated with each node (which, for example, may be used for either graphical display or distance computations). The values are:

- `TWOD_COORDS` — Nodes are specified by coordinates in 2-D
- `THREED_COORDS` — Nodes are specified by coordinates in 3-D
- `NO_COORDS` — The nodes do not have associated coordinates

The default value is `NO_COORDS`.

#### 1.1.10 `DISPLAY_DATA_TYPE: <string>`

Specifies how a graphical display of the nodes can be obtained. The values are:

- `COORD_DISPLAY` — Display is generated from the node coordinates
- `TWOD_DISPLAY` — Explicit coordinates in 2-D are given
- `NO_DISPLAY` — No graphical display is possible

The default value is `COORD_DISPLAY` if node coordinates are specified and `NO_DISPLAY` otherwise.

#### 1.1.11 `EOF:`

Terminates the input data. This entry is optional.

### 1.2 The data part

Depending on the choice of specifications some additional data may be required. These data are given in corresponding data sections following the specification part. Each data section begins with the corresponding keyword. The length of the section is either implicitly known from the format specification, or the section is terminated by an appropriate end-of-section identifier.

#### 1.2.1 `NODE_COORD_SECTION`

Node coordinates are given in this section.

- If `NODE_COORD_TYPE` is `TWOD_COORDS`:
  ```
  <integer> <real> <real>
  ```
- If `NODE_COORD_TYPE` is `THREED_COORDS`:
  ```
  <integer> <real> <real> <real>
  ```

The integers give the number of the respective nodes. The real numbers give the associated coordinates.

#### 1.2.2 `DEPOT_SECTION`

Contains a list of possible alternate depot nodes. This list is terminated by a `-1`.

#### 1.2.3 `DEMAND_SECTION`

The demands of all nodes of a CVRP are given in the form (per line):

```
<integer> <integer>
```

The first integer specifies a node number, the second its demand. The depot nodes must also occur in this section. Their demands are 0.

#### 1.2.4 `EDGE_DATA_SECTION`

Edges of a graph are specified in either of the two formats allowed in the `EDGE_DATA_FORMAT` entry.

- If the type is `EDGE_LIST`, then the edges are given as a sequence of lines of the form:
  ```
  <integer> <integer>
  ```
  Each entry gives the terminal nodes of some edge. The list is terminated by a `-1`.

- If the type is `ADJ_LIST`, the section consists of a list of adjacency lists for nodes. The adjacency list of a node x is specified as:
  ```
  <integer> <integer> ... <integer> -1
  ```
  where the first integer gives the number of node x and the following integers (terminated by `-1`) the numbers of nodes adjacent to x. The list of adjacency lists is terminated by an additional `-1`.

#### 1.2.5 `FIXED_EDGES_SECTION`

In this section, edges are listed that are required to appear in each solution to the problem. The edges to be fixed are given in the form (per line):

```
<integer> <integer>
```

meaning that the edge (arc) from the first node to the second node has to be contained in a solution. This section is terminated by a `-1`.

#### 1.2.6 `DISPLAY_DATA_SECTION`

If `DISPLAY_DATA_TYPE` is `TWOD_DISPLAY`, the 2-dimensional coordinates from which a display can be generated are given in the form (per line):

```
<integer> <real> <real>
```

The integers specify the respective nodes and the real numbers give the associated coordinates.

#### 1.2.7 `TOUR_SECTION`

A collection of tours is specified in this section. Each tour is given by a list of integers giving the sequence in which the nodes are visited in this tour. Every such tour is terminated by a `-1`. An additional `-1` terminates this section.

#### 1.2.8 `EDGE_WEIGHT_SECTION`

The edge weights are given in the format specified by the `EDGE_WEIGHT_FORMAT` entry. At present, all explicit data is integral and is given in one of the matrix formats with implicitly known lengths.

---

## 2. The distance functions

For the various choices of `EDGE_WEIGHT_TYPE`, we now describe the computations of the respective distances. In each case we give a (simplified) C implementation for computing the distances from the input coordinates. All computations involving floating-point numbers are carried out in double precision arithmetic. The integers are assumed to be represented in 32-bit words. Since distances are required to be integral, we round to the nearest integer (in most cases). Below we have used the rounding function `nint` (`nint(x)` can be replaced by `(int)(x + 0.5)`).

### 2.1 Euclidean distance (L2 metric)

For edge weight type `EUC_2D` and `EUC_3D`, floating point coordinates must be specified for each node. Let `x[i]`, `y[i]`, and `z[i]` be the coordinates of node i.

- 2D case:

```c
xd = x[i] - x[j];
yd = y[i] - y[j];
dij = nint( sqrt( xd*xd + yd*yd ) );
```

- 3D case:

```c
xd = x[i] - x[j];
yd = y[i] - y[j];
zd = z[i] - z[j];
dij = nint( sqrt( xd*xd + yd*yd + zd*zd ) );
```

where `sqrt` is the C square root function.

### 2.2 Manhattan distance (L1 metric)

Distances are given as Manhattan distances if the edge weight type is `MAN_2D` or `MAN_3D`.

- 2D case:

```c
xd = abs( x[i] - x[j] );
yd = abs( y[i] - y[j] );
dij = nint( xd + yd );
```

- 3D case:

```c
xd = abs( x[i] - x[j] );
yd = abs( y[i] - y[j] );
zd = abs( z[i] - z[j] );
dij = nint( xd + yd + zd );
```

### 2.3 Maximum distance (L∞ metric)

Maximum distances are computed if the edge weight type is `MAX_2D` or `MAX_3D`.

- 2D case:

```c
xd = abs( x[i] - x[j] );
yd = abs( y[i] - y[j] );
dij = max( nint( xd ), nint( yd ) );
```

- 3D case:

```c
xd = abs( x[i] - x[j] );
yd = abs( y[i] - y[j] );
zd = abs( z[i] - z[j] );
dij = max( nint( xd ), nint( yd ), nint( zd ) );
```

### 2.4 Geographical distance

If the traveling salesman problem is a geographical problem, then the nodes correspond to points on the earth and the distance between two points is their distance on the idealized sphere with radius 6378.388 kilometers. The node coordinates give the geographical latitude and longitude of the corresponding point on the earth. Latitude and longitude are given in the form `DDD.MM` where `DDD` are the degrees and `MM` the minutes. A positive latitude is assumed to be “North”, negative latitude means “South”. Positive longitude means “East”, negative longitude is assumed to be “West”.

For example, the input coordinates for Augsburg are `48.23` and `10.53`, meaning 48° 23′ North and 10° 53′ East.

First convert input to radians:

```c
PI = 3.141592;
deg = nint( x[i] );
min = x[i] - deg;
latitude[i]  = PI * (deg + 5.0 * min / 3.0) / 180.0;
deg = nint( y[i] );
min = y[i] - deg;
longitude[i] = PI * (deg + 5.0 * min / 3.0) / 180.0;
```

Then the distance between two different nodes i and j in kilometers is computed as follows:

```c
RRR = 6378.388;
q1 = cos( longitude[i] - longitude[j] );
q2 = cos( latitude[i] - latitude[j] );
q3 = cos( latitude[i] + latitude[j] );
dij = (int) ( RRR * acos( 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3) ) + 1.0 );
```

The function `acos` is the inverse of the cosine function.

### 2.5 Pseudo-Euclidean distance

The edge weight type `ATT` corresponds to a special “pseudo-Euclidean” distance function. Let `x[i]` and `y[i]` be the coordinates of node i. The distance between two points i and j is computed as follows:

```c
xd = x[i] - x[j];
yd = y[i] - y[j];
rij = sqrt( (xd*xd + yd*yd) / 10.0 );
tij = nint( rij );
if (tij < rij) dij = tij + 1;
else           dij = tij;
```

### 2.6 Ceiling of the Euclidean distance

The edge weight type `CEIL_2D` requires that the 2-dimensional Euclidean distance is rounded up to the next integer.

### 2.7 Distance for crystallography problems

We have included into TSPLIB the crystallography problems as described in [1]. These problems are not explicitly given but subroutines are provided to generate the 12 problems mentioned in this reference and subproblems thereof (see section 3.2). To compute distances for these problems the movement of three motors has to be taken into consideration. There are two types of distance functions: one that assumes equal speed of the motors (`XRAY1`) and one that uses different speeds (`XRAY2`). The corresponding distance functions are given as FORTRAN implementations (files `deq.f`, resp. `duneq.f`) in the distribution file.

For obtaining integer distances, we propose to multiply the distances computed by the original subroutines by 100.0 and round to the nearest integer. We list our modified distance function for the case of equal motor speeds in the FORTRAN version below.

```fortran
      INTEGER FUNCTION ICOST(V,W)
      INTEGER V,W
      DOUBLE PRECISION DMIN1,DMAX1,DABS
      DOUBLE PRECISION DISTP,DISTC,DISTT,COST
      DISTP=DMIN1(DABS(PHI(V)-PHI(W)),DABS(DABS(PHI(V)-PHI(W))-360.0E+0))
      DISTC=DABS(CHI(V)-CHI(W))
      DISTT=DABS(TWOTH(V)-TWOTH(W))
      COST=DMAX1(DISTP/1.00E+0,DISTC/1.0E+0,DISTT/1.00E+0)
C *** Make integral distances ***
      ICOST=AINT(100.0E+0*COST+0.5E+0)
      RETURN
      END
```

The numbers `PHI()`, `CHI()`, and `TWOTH()` are the respective x-, y-, and z-coordinates of the points in the generated traveling salesman problems. Note, that TSPLIB95 contains only the original distance computation without the above modification.

### 2.8 Verification

To verify correctness of the distance function implementations we give the length of some “canonical” tours 1, 2, 3, …, n.

- `pcb442` — 221 440
- `gr666` — 423 710
- `att532` — 309 636
- `xray14012` (problem 8 in [1]):
  - with `XRAY1` — 15 429 219
  - with `XRAY2` — 12 943 294

---

## 3. Description of the library files

In this section we give a list of all problem instances that are currently available together with information on the length of optimal tours or lower and upper bounds for this length (if available).

### 3.1 Symmetric traveling salesman problems

The TSP instances are contained in directory `tsp`. Table 1 gives the problem names along with number of cities, problem type, and known lower and upper bounds for the optimal tour length (a single number indicating that the optimal length is known). The entry `MATRIX` indicates that the data is given in one of the matrix formats of 1.1.7. The names of the corresponding data files are obtained by appending the suffix `.tsp` to the problem name. Some optimal tours are also provided. The corresponding files have names with suffix `.opt.tour`.

| Name |  #cities | Type  |  Bounds |
|---|-------|-------|---------|
| a280 |   280 |      EUC2D |   2579 |
| ali535 | 535 |      GEO |     202310 |
| att48 |  48 |       ATT |     10628 |
| att532 | 532 |      ATT |     27686 |
| bayg29 | 29 |       GEO |     1610 |
| bays29 | 29 |       GEO |     2020 |
| berlin52 | 52 |     EUC2D |   7542 |
| bier127 | 127 |     EUC2D |   118282 |
| brazil58 | 58 |     MATRIX |  25395 |
| brd14051 | 14051 |  EUC2D |   [468942,469445] |
| brg180 | 180 |      MATRIX |  1950 |
| burma14 | 14 |      GEO |     3323 |
| ch130 |  130 |      EUC2D |   6110 |
| ch150 |  150 |      EUC2D |   6528 |
| d198 |   198 |      EUC2D |   15780 |
| d493 |   493 |      EUC2D |   35002 |
| d657 |   657 |      EUC2D |   48912 |
| d1291 |  1291 |     EUC2D |   50801 |
| d1655 |  1655 |     EUC2D |   62128 |
| d2103 |  2103 |     EUC2D |   [79952,80450] |
| d15112 | 15112 |    EUC2D |   [1564590,1573152] |
| d18512 | 18512 |    EUC2D |   [644650,645488] |
| dantzig42 | 42 |    MATRIX |  699 |
| dsj1000 | 1000 |    CEIL2D |  18659688 |
| eil51 |  51 |       EUC2D |   426 |
| eil76 |  76 |       EUC2D |   538 |
| eil101 | 101 |      EUC2D |   629 |
| fl417 |  417 |      EUC2D |   11861 |
| fl1400 | 1400 |     EUC2D |   20127 |
| fl1577 | 1577 |     EUC2D |   [22204,22249] |
| fl3795 | 3795 |     EUC2D |   [28723,28772] |
| fnl4461 | 4461 |    EUC2D |   182566 |
| fri26 |  26 |       MATRIX |  937 |
| gil262 | 262 |      EUC2D |   2378 |
| gr17 |   17 |       MATRIX |  2085 |
| gr21 |   21 |       MATRIX |  2707 |
| gr24 |   24 |       MATRIX |  1272 |
| gr48 |   48 |       MATRIX |  5046 |
| gr96 |   96 |       GEO |     55209 |
| gr120 |  120 |      MATRIX |  6942 |
| gr137 |  137 |      GEO |     69853 |
| gr202 |  202 |      GEO |     40160 |
| gr229 |  229 |      GEO |     134602 |
| gr431 |  431 |      GEO |     171414 |
| gr666 |  666 |      GEO |     294358 |
| hk48 |   48 |       MATRIX |  11461 |
| kroA100 | 100 |     EUC2D |   21282 |
| kroB100 | 100 |     EUC2D |   22141 |
| kroC100 | 100 |     EUC2D |   20749 |
| kroD100 | 100 |     EUC2D |   21294 |
| kroE100 | 100 |     EUC2D |   22068 |
| kroA150 | 150 |     EUC2D |   26524 |
| kroB150 | 150 |     EUC2D |   26130 |
| kroA200 | 200 |     EUC2D |   29368 |
| kroB200 | 200 |     EUC2D |   29437 |
| lin105 |  105 |     EUC2D |   14379 |
| lin318 |  318 |     EUC2D |   42029 |
| linhp318 | 318 |    EUC2D |   41345 |
| nrw1379 | 1379 |    EUC2D |   56638 |
| p654 |   654 |      EUC2D |   34643 |
| pa561 |  561 |      MATRIX |  2763 |
| pcb442 | 442 |      EUC2D |   50778 |
| pcb1173 | 1173 |    EUC2D |   56892 |
| pcb3038 | 3038 |    EUC2D |   137694 |
| pla7397 | 7397 |    CEIL2D |  23260728 |
| pla33810 | 33810 |  CEIL2D |  [65913275,66116530] |
| pla85900 | 85900 |  CEIL2D |  [141904862,142487006] |
| pr76 |   76 |       EUC2D |   108159 |
| pr107 |  107 |      EUC2D |   44303 |
| pr124 |  124 |      EUC2D |   59030 |
| pr136 |  136 |      EUC2D |   96772 |
| pr144 |  144 |      EUC2D |   58537 |
| pr152 |  152 |      EUC2D |   73682 |
| pr226 |  226 |      EUC2D |   80369 |
| pr264 |  264 |      EUC2D |   49135 |
| pr299 |  299 |      EUC2D |   48191 |
| pr439 |  439 |      EUC2D |   107217 |
| pr1002 | 1002 |     EUC2D |   259045 |
| pr2392 | 2392 |     EUC2D |   378032 |
| rat99 |  99 |       EUC2D |   1211 |
| rat195 | 195 |      EUC2D |   2323 |
| rat575 | 575 |      EUC2D |   6773 |
| rat783 | 783 |      EUC2D |   8806 |
| rd100 |  100 |      EUC2D |   7910 |
| rd400 |  400 |      EUC2D |   15281 |
| rl1304 | 1304 |     EUC2D |   252948 |
| rl1323 | 1323 |     EUC2D |   270199 |
| rl1889 | 1889 |     EUC2D |   316536 |
| rl5915 | 5915 |     EUC2D |   [565040,565530] |
| rl5934 | 5934 |     EUC2D |   [554070,556045] |
| rl11849 | 11849 |   EUC2D |   [920847,923368] |
| si175 |  175 |      MATRIX |  21407 |
| si535 |  535 |      MATRIX |  48450 |
| si1032 | 1032 |     MATRIX |  92650 |
| st70 |   70 |       EUC2D |   675 |
| swiss42 | 42 |      MATRIX |  1273 |
| ts225 |  225 |      EUC2D |   126643 |
| tsp225 | 225 |      EUC2D |   3919 |
| u159 |   159 |      EUC2D |   42080 |
| u574 |   574 |      EUC2D |   36905 |
| u724 |   724 |      EUC2D |   41910 |
| u1060 |  1060 |     EUC2D |   224094 |
| u1432 |  1432 |     EUC2D |   152970 |
| u1817 |  1817 |     EUC2D |   57201 |
| u2152 |  2152 |     EUC2D |   64253 |
| u2319 |  2319 |     EUC2D |   234256 |
| ulysses16 | 16 |    GEO |     6859 |
| ulysses22 | 22 |    GEO |     7013 |
| usa13509 | 13509 |  EUC2D |   [19947008,19982889] |
| vm1084 | 1084 |     EUC2D |   239297 |
| vm1748 | 1748 |     EUC2D |   336556 |

#### Crystallography problems

In the file `xray.problems` in directory `tsp` we distribute the routines written by Bland and Shallcross and the necessary data to generate the crystallography problems discussed in [1]. The file `xray.problems` is one file into which the single files mentioned in the sequel have been merged. These single files have to be extracted from `xray.problems` using an editor.

The following original files are provided

```xray
read.me
deq.f
duneq.f
daux.f
gentsp.f
a.data
b.data
d.data
e.data
f.data
```

In addition we have included specially prepared data files to generate the 12 problems mentioned in [1]. The files have the names `xray1.data` through `xray12.data`. Using these data files 12 symmetric TSPs can be generated using the program `gentsp.f`. We propose to name the respective problem instances `xray4472`, `xray2950`, `xray7008`, `xray2762`, `xray6922`, `xray9070`, `xray5888`, `xray14012`, `xray5520`, `xray13804`, `xray14464`, and `xray13590`.

To verify the correct use of the generating routines we list part of the file `xray14012.tsp`.

```xray
NAME : xray14012
COMMENT : Crystallography problem 8 (Bland/Shallcross)
TYPE : TSP
DIMENSION : 14012
EDGE_WEIGHT_TYPE : XRAY2
NODE_COORD_SECTION
1     -91.802854544029  -6.4097888697337  176.39830490027
2     -87.715643397938  -6.4659384343446  165.56800324542
3     -83.587211962870  -6.4895404648110  163.53828545043
4     -79.460007412434  -6.4797580053949  165.86438271158
...
14009  100.539992581837  6.4797580053949  165.86438271158
14010   96.412788031401  6.4895404648110  163.53828545043
14011   92.284356596333  6.4659384343446  165.56800324542
14012   88.197145450242  6.4097888697337  176.39830490027
```

### 3.2 Hamiltonian cycle problems

Instances of the Hamiltonian cycle problem are contained in the directory `hcp`. At present, we have the data files

```txt
alb1000.hcp alb2000.hcp
alb3000a.hcp alb3000b.hcp
alb3000c.hcp alb3000d.hcp
alb3000e.hcp alb4000.hcp
alb5000.hcp
```

Every instance contains a Hamiltonian cycle which is given in the corresponding `.opt.tour` file. In problem instance `alb4000` two edges are fixed.

In addition to these files, the directory contains the C program `tspleap.c` by M. Jünger and G. Rinaldi. This program can be used to generate TSP instances (in TSPLIB format) originating from the problem of deciding whether an (r,s)-leaper on a m×n chess board can start at some square of the board, visit each square exactly once, and return to its starting square. A detailed documentation is given in the file `tspleap.c`.

### 3.3 Asymmetric traveling salesman problems

Table 2 lists the ATSP instances (in directory `atsp`) together with their optimal solution values. The names of the corresponding data files are obtained by appending the suffix `.atsp` to the problem name. The data files for problems `ftv90`, `ftv100`, `ftv110`, `ftv120`, `ftv130`, `ftv140`, `ftv150`, and `ftv160` are not present. These instances are obtained from `ftv170`. E.g., `ftv120` is the subproblem of `ftv170` defined by the first 121 nodes, `ftv130` is defined by the first 131 nodes, etc.

Table 2:

| Name |   #cities |  Type |    Optimum |
|---|-------|-------|---------|
| br17 |   17 |       MATRIX |  39 |
| ft53 |   53 |       MATRIX |  6905 |
| ft70 |   70 |       MATRIX |  38673 |
| ftv33 |  34 |       MATRIX |  1286 |
| ftv35 |  36 |       MATRIX |  1473 |
| ftv38 |  39 |       MATRIX |  1530 |
| ftv44 |  45 |       MATRIX |  1613 |
| ftv47 |  48 |       MATRIX |  1776 |
| ftv55 |  56 |       MATRIX |  1608 |
| ftv64 |  65 |       MATRIX |  1839 |
| ftv70 |  71 |       MATRIX |  1950 |
| ftv90 |  91 |       MATRIX |  1579 |
| ftv100 | 101 |      MATRIX |  1788 |
| ftv110 | 111 |      MATRIX |  1958 |
| ftv120 | 121 |      MATRIX |  2166 |
| ftv130 | 131 |      MATRIX |  2307 |
| ftv140 | 141 |      MATRIX |  2420 |
| ftv150 | 151 |      MATRIX |  2611 |
| ftv160 | 161 |      MATRIX |  2683 |
| ftv170 | 171 |      MATRIX |  2755 |
| kro124 | 100 |      MATRIX |  36230 |
| p43 |    43 |       MATRIX |  5620 |
| rbg323 | 323 |      MATRIX |  1326 |
| rbg358 | 358 |      MATRIX |  1163 |
| rbg403 | 403 |      MATRIX |  2465 |
| rbg443 | 443 |      MATRIX |  2720 |
| ry48p |  48 |       MATRIX |  14422 |

### 3.4 Sequential ordering problems

Every instance of a sequential ordering problem is given by a full matrix C of the following kind. If node i has to precede node j, then `Cji` is set to `-1`. C is assumed to be transitively closed with respect to precedences, i.e., if i has to precede j and j has to precede k, then it is implied that i has to precede k and, therefore, also `Cki` has to be set to `-1`.

Because we require that node 1 is the first node and node n is the last node in each feasible path, a SOP problem instance always has `Ci1 = -1`, for all `i = 2, ..., n`, and `Cnj = -1`, for all `j = 1, ..., n-1`. The entry `C1n` is set to infinity. All other entries of C are nonnegative integer values.

Table 3 lists the SOP instances (in directory `sop`) together with their known lower and upper bounds for the optimal path length. The names of the corresponding data files are obtained by appending the suffix `.sop` to the problem name.

Table 3:

| Name       | #nodes  #| prec | Type|    Bounds |
|---|-----------|------|-----|-----------|
| ESC07      | 96      | - |     MATRIX|  2125 |
| ESC11      | 133     | - |     MATRIX|  2075 |
| ESC12      | 147     | - |     MATRIX|  1675 |
| ESC25      | 279     | - |     MATRIX|  1681 |
| ESC47      | 491     | 10 |    MATRIX|  1288 |
| ESC63      | 659     | 95 |    MATRIX|  62 |
| ESC78      | 807     | 77 |    MATRIX|  18230 |
| br17.10    | 17      | 10 |    MATRIX|  55 |
| br17.12    | 17      | 12 |    MATRIX|  55 |
| ft53.1     | 54      | 12 |    MATRIX|  [7438,7570] |
| ft53.2     | 54      | 25 |    MATRIX|  [7630,8335] |
| ft53.3     | 54      | 48 |    MATRIX|  [9473,10935] |
| ft53.4     | 56      | 63 |    MATRIX|  14425 |
| ft70.1     | 71      | 17 |    MATRIX|  39313 |
| ft70.2     | 71      | 35 |    MATRIX|  [39739,41778] |
| ft70.3     | 71      | 68 |    MATRIX|  [41305,44732] |
| ft70.4     | 71      | 86 |    MATRIX|  [52269,53882] |
| kro124p.1  | 101     | 25 |    MATRIX|  [37722,42845] |
| kro124p.2  | 101     | 49 |    MATRIX|  [38534,45848] |
| kro124p.3  | 101     | 97 |    MATRIX|  [40967,55649] |
| kro124p.4  | 101     | 131 |   MATRIX|  [64858,80753] |
| p43.1      | 44      | 9 |     MATRIX|  27990 |
| p43.2      | 44      | 20 |    MATRIX|  [28175,28330] |
| p43.3      | 44      | 37 |    MATRIX|  [28366,28680] |
| p43.4      | 44      | 50 |    MATRIX|  [69569,82960] |
| prob.42    | 42      | 10 |    MATRIX|  243 |
| prob.100   | 100     | 41 |    MATRIX|  [1024,1385] |
| rbg048a    | 50      | 192 |   MATRIX|  351 |
| rbg050c    | 52      | 256 |   MATRIX|  467 |
| rbg109a    | 111     | 622 |   MATRIX|  1038 |
| rbg150a    | 152     | 952 |   MATRIX|  [1748,1750] |
| rbg174a    | 176     | 1113 |  MATRIX|  2053 |
| rbg253a    | 255     | 1721 |  MATRIX|  [2928,2987] |
| rbg323a    | 325     | 2412 |  MATRIX|  [3136,3221] |
| rbg341a    | 343     | 2542 |  MATRIX|  [2543,2854] |
| rbg358a    | 360     | 3239 |  MATRIX|  [2518,2758] |
| rbg378a    | 380     | 3069 |  MATRIX|  [2761,3142] |
| ry48p.1    | 49      | 11 |    MATRIX|  [15220,15935] |
| ry48p.2    | 49      | 23 |    MATRIX|  [15524,17071] |
| ry48p.3    | 49      | 42 |    MATRIX|  [18156,20051] |
| ry48p.4    | 49      | 58 |    MATRIX|  [29967,31446] |

### 3.5 Capacitated vehicle routing problems

Data for capacitated vehicle routing problems is contained in the directory `vrp`. Data files have suffix `.vrp`. At present, we have the data files

```txt
att48.vrp
eil7.vrp
eil13.vrp
eil22.vrp
eil23.vrp
eil30.vrp
eil31.vrp
eil33.vrp
eil51.vrp
eilA76.vrp
eilB76.vrp
eilC76.vrp
eilD76.vrp
eilA101.vrp
eilB101.vrp
gil262.vrp
```

Various problems can be defined on these data sets, e.g., depending on whether the number of vehicles is fixed, so we do not list optimal solutions here. Some values are given in the data files themselves.

### 3.6 Further special files

In addition to the data and solution files, the following special files are contained in the library.

- `TSPLIB VERSION`: Gives the current version of the library
- `README`: A short information on TSPLIB
- `DOC.PS`: Description of TSPLIB (PostScript)

### 3.7 vrps from cvrp

| Instance | \|𝐶\| | \|𝐾\| | 𝑄  | UB  | Opt | Features |
| --- | --- | --- | --- | --- | --- | --- |
| **Set A ([Augerat et al., 1995](https://www.osti.gov/etdeweb/biblio/289002))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A.7z "Set File") |
| [A-n32-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n32-k5.vrp "Instance File") | 31  | 5   | 100 | [784.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n32-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/4 "Plotted Instance") |
| [A-n33-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n33-k5.vrp "Instance File") | 32  | 5   | 100 | [661.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n33-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/5 "Plotted Instance") |
| [A-n33-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n33-k6.vrp "Instance File") | 32  | 6   | 100 | [742.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n33-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/7 "Plotted Instance") |
| [A-n34-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n34-k5.vrp "Instance File") | 33  | 5   | 100 | [778.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n34-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/8 "Plotted Instance") |
| [A-n36-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n36-k5.vrp "Instance File") | 35  | 5   | 100 | [799.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n36-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/9 "Plotted Instance") |
| [A-n37-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n37-k5.vrp "Instance File") | 36  | 5   | 100 | [669.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n37-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/10 "Plotted Instance") |
| [A-n37-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n37-k6.vrp "Instance File") | 36  | 6   | 100 | [949.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n37-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/11 "Plotted Instance") |
| [A-n38-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n38-k5.vrp "Instance File") | 37  | 5   | 100 | [730.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n38-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/12 "Plotted Instance") |
| [A-n39-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n39-k5.vrp "Instance File") | 38  | 5   | 100 | [822.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n39-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/13 "Plotted Instance") |
| [A-n39-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n39-k6.vrp "Instance File") | 38  | 6   | 100 | [831.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n39-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/14 "Plotted Instance") |
| [A-n44-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n44-k6.vrp "Instance File") | 43  | 6   | 100 | [937.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n44-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/15 "Plotted Instance") |
| [A-n45-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n45-k6.vrp "Instance File") | 44  | 6   | 100 | [944.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n45-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/16 "Plotted Instance") |
| [A-n45-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n45-k7.vrp "Instance File") | 44  | 7   | 100 | [1,146.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n45-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/17 "Plotted Instance") |
| [A-n46-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n46-k7.vrp "Instance File") | 45  | 7   | 100 | [914.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n46-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/18 "Plotted Instance") |
| [A-n48-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n48-k7.vrp "Instance File") | 47  | 7   | 100 | [1,073.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n48-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/19 "Plotted Instance") |
| [A-n53-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n53-k7.vrp "Instance File") | 52  | 7   | 100 | [1,010.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n53-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/20 "Plotted Instance") |
| [A-n54-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n54-k7.vrp "Instance File") | 53  | 7   | 100 | [1,167.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n54-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/21 "Plotted Instance") |
| [A-n55-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n55-k9.vrp "Instance File") | 54  | 9   | 100 | [1,073.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n55-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/22 "Plotted Instance") |
| [A-n60-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n60-k9.vrp "Instance File") | 59  | 9   | 100 | [1,354.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n60-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/23 "Plotted Instance") |
| [A-n61-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n61-k9.vrp "Instance File") | 60  | 9   | 100 | [1,034.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n61-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/24 "Plotted Instance") |
| [A-n62-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n62-k8.vrp "Instance File") | 61  | 8   | 100 | [1,288.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n62-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/25 "Plotted Instance") |
| [A-n63-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n63-k9.vrp "Instance File") | 62  | 9   | 100 | [1,616.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n63-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/26 "Plotted Instance") |
| [A-n63-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n63-k10.vrp "Instance File") | 62  | 10  | 100 | [1,314.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n63-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/27 "Plotted Instance") |
| [A-n64-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n64-k9.vrp "Instance File") | 63  | 9   | 100 | [1,401.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n64-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/28 "Plotted Instance") |
| [A-n65-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n65-k9.vrp "Instance File") | 64  | 9   | 100 | [1,174.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n65-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/29 "Plotted Instance") |
| [A-n69-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n69-k9.vrp "Instance File") | 68  | 9   | 100 | [1,159.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n69-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/30 "Plotted Instance") |
| [A-n80-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n80-k10.vrp "Instance File") | 79  | 10  | 100 | [1,763.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A/A-n80-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/31 "Plotted Instance") |
| **Set B ([Augerat et al., 1995](https://www.osti.gov/etdeweb/biblio/289002))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B.7z "Set File") |
| [B-n31-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n31-k5.vrp "Instance File") | 30  | 5   | 100 | [672.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n31-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/6 "Plotted Instance") |
| [B-n34-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n34-k5.vrp "Instance File") | 33  | 5   | 100 | [788.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n34-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/32 "Plotted Instance") |
| [B-n35-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n35-k5.vrp "Instance File") | 34  | 5   | 100 | [955.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n35-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/33 "Plotted Instance") |
| [B-n38-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n38-k6.vrp "Instance File") | 37  | 6   | 100 | [805.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n38-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/34 "Plotted Instance") |
| [B-n39-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n39-k5.vrp "Instance File") | 38  | 5   | 100 | [549.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n39-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/35 "Plotted Instance") |
| [B-n41-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n41-k6.vrp "Instance File") | 40  | 6   | 100 | [829.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n41-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/36 "Plotted Instance") |
| [B-n43-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n43-k6.vrp "Instance File") | 42  | 6   | 100 | [742.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n43-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/37 "Plotted Instance") |
| [B-n44-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n44-k7.vrp "Instance File") | 43  | 7   | 100 | [909.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n44-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/38 "Plotted Instance") |
| [B-n45-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n45-k5.vrp "Instance File") | 44  | 5   | 100 | [751.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n45-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/39 "Plotted Instance") |
| [B-n45-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n45-k6.vrp "Instance File") | 44  | 6   | 100 | [678.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n45-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/40 "Plotted Instance") |
| [B-n50-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n50-k7.vrp "Instance File") | 49  | 7   | 100 | [741.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n50-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/41 "Plotted Instance") |
| [B-n50-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n50-k8.vrp "Instance File") | 49  | 8   | 100 | [1,312.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n50-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/42 "Plotted Instance") |
| [B-n51-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n51-k7.vrp "Instance File") | 50  | 7   | 100 | [1,032.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n51-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/43 "Plotted Instance") |
| [B-n52-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n52-k7.vrp "Instance File") | 51  | 7   | 100 | [747.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n52-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/44 "Plotted Instance") |
| [B-n56-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n56-k7.vrp "Instance File") | 55  | 7   | 100 | [707.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n56-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/45 "Plotted Instance") |
| [B-n57-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n57-k7.vrp "Instance File") | 56  | 7   | 100 | [1,153.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n57-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/46 "Plotted Instance") |
| [B-n57-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n57-k9.vrp "Instance File") | 56  | 9   | 100 | [1,598.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n57-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/47 "Plotted Instance") |
| [B-n63-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n63-k10.vrp "Instance File") | 62  | 10  | 100 | [1,496.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n63-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/48 "Plotted Instance") |
| [B-n64-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n64-k9.vrp "Instance File") | 63  | 9   | 100 | [861.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n64-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/49 "Plotted Instance") |
| [B-n66-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n66-k9.vrp "Instance File") | 65  | 9   | 100 | [1,316.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n66-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/50 "Plotted Instance") |
| [B-n67-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n67-k10.vrp "Instance File") | 66  | 10  | 100 | [1,032.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n67-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/51 "Plotted Instance") |
| [B-n68-k9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n68-k9.vrp "Instance File") | 67  | 9   | 100 | [1,272.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n68-k9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/52 "Plotted Instance") |
| [B-n78-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n78-k10.vrp "Instance File") | 77  | 10  | 100 | [1,221.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B/B-n78-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/53 "Plotted Instance") |
| **Set E ([Christofides et al., 1969](https://doi.org/10.1057/jors.1969.75))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E.7z "Set File") |
| [E-n13-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n13-k4.vrp "Instance File") | 12  | 4   | 6000 | [247.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n13-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/54 "Plotted Instance") |
| [E-n22-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n22-k4.vrp "Instance File") | 21  | 4   | 6000 | [375.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n22-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/55 "Plotted Instance") |
| [E-n23-k3](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n23-k3.vrp "Instance File") | 22  | 3   | 4500 | [569.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n23-k3.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/56 "Plotted Instance") |
| [E-n30-k3](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n30-k3.vrp "Instance File") | 29  | 3   | 4500 | [534.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n30-k3.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/57 "Plotted Instance") |
| [E-n31-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n31-k7.vrp "Instance File") | 30  | 7   | 140 | [379.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n31-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/58 "Plotted Instance") |
| [E-n33-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n33-k4.vrp "Instance File") | 32  | 4   | 8000 | [835.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n33-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/59 "Plotted Instance") |
| [E-n51-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n51-k5.vrp "Instance File") | 50  | 5   | 160 | [521.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n51-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/60 "Plotted Instance") |
| [E-n76-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k7.vrp "Instance File") | 75  | 7   | 220 | [682.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/61 "Plotted Instance") |
| [E-n76-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k8.vrp "Instance File") | 75  | 8   | 180 | [735.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/62 "Plotted Instance") |
| [E-n76-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k10.vrp "Instance File") | 75  | 10  | 140 | [830.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/63 "Plotted Instance") |
| [E-n76-k14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k14.vrp "Instance File") | 75  | 14  | 100 | [1,021.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n76-k14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/64 "Plotted Instance") |
| [E-n101-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n101-k8.vrp "Instance File") | 100 | 8   | 200 | [815.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n101-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/65 "Plotted Instance") |
| [E-n101-k14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n101-k14.vrp "Instance File") | 100 | 14  | 112 | [1,067.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E/E-n101-k14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/66 "Plotted Instance") |
| **Set F ([Fisher, 1994](https://doi.org/10.1287/opre.42.4.626))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F.7z "Set File") |
| [F-n45-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n45-k4.vrp "Instance File") | 44  | 4   | 2010 | [724.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n45-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/67 "Plotted Instance") |
| [F-n72-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n72-k4.vrp "Instance File") | 71  | 4   | 30000 | [237.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n72-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/68 "Plotted Instance") |
| [F-n135-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n135-k7.vrp "Instance File") | 134 | 7   | 2210 | [1,162.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F/F-n135-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/69 "Plotted Instance") |
| **Set M (Christofides et al., 1979)** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M.7z "Set File") |
| [M-n101-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n101-k10.vrp "Instance File") | 100 | 10  | 200 | [820.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n101-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/70 "Plotted Instance") |
| [M-n121-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n121-k7.vrp "Instance File") | 120 | 7   | 200 | [1,034.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n121-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/71 "Plotted Instance") |
| [M-n151-k12](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n151-k12.vrp "Instance File") | 150 | 12  | 200 | [1,015.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n151-k12.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/72 "Plotted Instance") |
| [M-n200-k16](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n200-k16.vrp "Instance File") | 199 | 16  | 200 | [1,274.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n200-k16.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/73 "Plotted Instance") |
| [M-n200-k17](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n200-k17.vrp "Instance File") | 199 | 17  | 200 | [1,275.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M/M-n200-k17.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/74 "Plotted Instance") |
| **Set P ([Augerat et al., 1995](https://www.osti.gov/etdeweb/biblio/289002))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P.7z "Set File") |
| [P-n16-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n16-k8.vrp "Instance File") | 15  | 8   | 35  | [450.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n16-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/75 "Plotted Instance") |
| [P-n19-k2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n19-k2.vrp "Instance File") | 18  | 2   | 160 | [212.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n19-k2.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/76 "Plotted Instance") |
| [P-n20-k2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n20-k2.vrp "Instance File") | 19  | 2   | 160 | [216.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n20-k2.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/77 "Plotted Instance") |
| [P-n21-k2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n21-k2.vrp "Instance File") | 20  | 2   | 160 | [211.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n21-k2.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/78 "Plotted Instance") |
| [P-n22-k2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n22-k2.vrp "Instance File") | 21  | 2   | 160 | [216.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n22-k2.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/79 "Plotted Instance") |
| [P-n22-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n22-k8.vrp "Instance File") | 21  | 8   | 3000 | [603.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n22-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/80 "Plotted Instance") |
| [P-n23-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n23-k8.vrp "Instance File") | 22  | 8   | 40  | [529.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n23-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/81 "Plotted Instance") |
| [P-n40-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n40-k5.vrp "Instance File") | 39  | 5   | 140 | [458.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n40-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/82 "Plotted Instance") |
| [P-n45-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n45-k5.vrp "Instance File") | 44  | 5   | 150 | [510.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n45-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/83 "Plotted Instance") |
| [P-n50-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k7.vrp "Instance File") | 49  | 7   | 150 | [554.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/84 "Plotted Instance") |
| [P-n50-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k8.vrp "Instance File") | 49  | 8   | 120 | [631.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/85 "Plotted Instance") |
| [P-n50-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k10.vrp "Instance File") | 49  | 10  | 100 | [696.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n50-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/86 "Plotted Instance") |
| [P-n51-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n51-k10.vrp "Instance File") | 50  | 10  | 80  | [741.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n51-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/87 "Plotted Instance") |
| [P-n55-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k7.vrp "Instance File") | 54  | 7   | 170 | [568.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/88 "Plotted Instance") |
| [P-n55-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k8.vrp "Instance File") | 54  | 8   | 160 | [588.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/89 "Plotted Instance") |
| [P-n55-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k10.vrp "Instance File") | 54  | 10  | 115 | [694.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/90 "Plotted Instance") |
| [P-n55-k15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k15.vrp "Instance File") | 54  | 15  | 70  | [989.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n55-k15.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/91 "Plotted Instance") |
| [P-n60-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n60-k10.vrp "Instance File") | 59  | 10  | 120 | [744.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n60-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/92 "Plotted Instance") |
| [P-n60-k15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n60-k15.vrp "Instance File") | 59  | 15  | 80  | [968.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n60-k15.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/93 "Plotted Instance") |
| [P-n65-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n65-k10.vrp "Instance File") | 64  | 10  | 130 | [792.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n65-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/94 "Plotted Instance") |
| [P-n70-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n70-k10.vrp "Instance File") | 69  | 10  | 135 | [827.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n70-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/95 "Plotted Instance") |
| [P-n76-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n76-k4.vrp "Instance File") | 75  | 4   | 350 | [593.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n76-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/96 "Plotted Instance") |
| [P-n76-k5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n76-k5.vrp "Instance File") | 75  | 5   | 280 | [627.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n76-k5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/97 "Plotted Instance") |
| [P-n101-k4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n101-k4.vrp "Instance File") | 100 | 4   | 400 | [681.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P/P-n101-k4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/98 "Plotted Instance") |
| **Set CMT (Christofides et al., 1979)** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT.7z "Set File") |
| [CMT1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT1.vrp "Instance File") | 50  | 5   | 160 | [524.61](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT1.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/99 "Plotted Instance") |
| [CMT2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT2.vrp "Instance File") | 75  | 10  | 140 | [835.26](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT2.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/100 "Plotted Instance") |
| [CMT3](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT3.vrp "Instance File") | 100 | 8   | 200 | [826.14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT3.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/101 "Plotted Instance") |
| [CMT4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT4.vrp "Instance File") | 150 | 12  | 200 | [1,028.42](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT4.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/102 "Plotted Instance") |
| [CMT5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT5.vrp "Instance File") | 199 | 17  | 200 | [1,291.29](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT5.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/103 "Plotted Instance") |
| [CMT6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT6.vrp "Instance File") | 50  | 6   | 160 | [555.43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/104 "Plotted Instance") |
| [CMT7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT7.vrp "Instance File") | 75  | 11  | 140 | [909.68](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/105 "Plotted Instance") |
| [CMT8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT8.vrp "Instance File") | 100 | 9   | 200 | [865.94](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/106 "Plotted Instance") |
| [CMT9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT9.vrp "Instance File") | 150 | 14  | 200 | [1,162.55](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT9.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/107 "Plotted Instance") |
| [CMT10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT10.vrp "Instance File") | 199 | 18  | 200 | [1,395.85](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/108 "Plotted Instance") |
| [CMT11](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT11.vrp "Instance File") | 120 | 7   | 200 | [1,042.12](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT11.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/109 "Plotted Instance") |
| [CMT12](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT12.vrp "Instance File") | 100 | 10  | 200 | [819.56](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT12.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/110 "Plotted Instance") |
| [CMT13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT13.vrp "Instance File") | 120 | 11  | 200 | [1,541.14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/111 "Plotted Instance") |
| [CMT14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT14.vrp "Instance File") | 100 | 11  | 200 | [866.37](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT/CMT14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/112 "Plotted Instance") |
| **Set tai ([Rochat et al., 1995](https://doi.org/10.1007/BF02430370))** |     |     |     |     |     | [Set file](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai.7z "Set File") |
| [tai75a](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75a.vrp "Instance File") | 75  | 10  | 1445 | [1,618.36](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75a.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/113 "Plotted Instance") |
| [tai75b](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75b.vrp "Instance File") | 75  | 9   | 1679 | [1,344.62](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75b.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/114 "Plotted Instance") |
| [tai75c](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75c.vrp "Instance File") | 75  | 9   | 1122 | [1,291.01](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75c.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/115 "Plotted Instance") |
| [tai75d](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75d.vrp "Instance File") | 75  | 9   | 1699 | [1,365.42](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai75d.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/116 "Plotted Instance") |
| [tai100a](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100a.vrp "Instance File") | 100 | 11  | 1409 | [2,041.34](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100a.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/117 "Plotted Instance") |
| [tai100b](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100b.vrp "Instance File") | 100 | 11  | 1842 | [1,939.90](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100b.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/118 "Plotted Instance") |
| [tai100c](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100c.vrp "Instance File") | 100 | 11  | 2043 | [1,406.20](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100c.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/119 "Plotted Instance") |
| [tai100d](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100d.vrp "Instance File") | 100 | 11  | 1297 | [1,580.46](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai100d.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/120 "Plotted Instance") |
| [tai150a](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150a.vrp "Instance File") | 150 | 15  | 1544 | [3,055.23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150a.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/121 "Plotted Instance") |
| [tai150b](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150b.vrp "Instance File") | 150 | 14  | 1918 | [2,727.03](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150b.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/122 "Plotted Instance") |
| [tai150c](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150c.vrp "Instance File") | 150 | 14  | 2021 | [2,358.66](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150c.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/123 "Plotted Instance") |
| [tai150d](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150d.vrp "Instance File") | 150 | 14  | 1874 | [2,645.39](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai150d.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/124 "Plotted Instance") |
| [tai385](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai385.vrp "Instance File") | 385 | 46  | 65  | [24,366.41](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai/tai385.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/125 "Plotted Instance") |
| **Set Golden ([Golden et al., 1998](https://doi.org/10.1007/978-1-4615-5755-5_2))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden.7z "Set File") |
| [Golden\_1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_1.vrp "Instance File") | 240 | 9   | 550 | [5,623.47](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/126 "Plotted Instance") |
| [Golden\_2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_2.vrp "Instance File") | 320 | 10  | 700 | [8,404.61](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/127 "Plotted Instance") |
| [Golden\_3](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_3.vrp "Instance File") | 400 | 9   | 900 | [10,997.80](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_3.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/128 "Plotted Instance") |
| [Golden\_4](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_4.vrp "Instance File") | 480 | 10  | 1000 | [13,588.60](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_4.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/129 "Plotted Instance") |
| [Golden\_5](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_5.vrp "Instance File") | 200 | 5   | 900 | [6,460.98](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_5.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/130 "Plotted Instance") |
| [Golden\_6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_6.vrp "Instance File") | 280 | 7   | 900 | [8,400.33](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_6.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/131 "Plotted Instance") |
| [Golden\_7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_7.vrp "Instance File") | 360 | 8   | 900 | [10,102.70](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_7.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/132 "Plotted Instance") |
| [Golden\_8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_8.vrp "Instance File") | 440 | 10  | 900 | [11,635.30](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_8.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/133 "Plotted Instance") |
| [Golden\_9](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_9.vrp "Instance File") | 255 | 14  | 1000 | [579.70](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_9.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/134 "Plotted Instance") |
| [Golden\_10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_10.vrp "Instance File") | 323 | 16  | 1000 | [735.43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_10.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/135 "Plotted Instance") |
| [Golden\_11](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_11.vrp "Instance File") | 399 | 17  | 1000 | [911.98](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_11.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/136 "Plotted Instance") |
| [Golden\_12](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_12.vrp "Instance File") | 483 | 19  | 1000 | [1,100.67](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_12.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/137 "Plotted Instance") |
| [Golden\_13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_13.vrp "Instance File") | 252 | 26  | 1000 | [857.19](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/138 "Plotted Instance") |
| [Golden\_14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_14.vrp "Instance File") | 320 | 29  | 1000 | [1,080.55](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/139 "Plotted Instance") |
| [Golden\_15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_15.vrp "Instance File") | 396 | 33  | 1000 | [1,337.27](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_15.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/140 "Plotted Instance") |
| [Golden\_16](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_16.vrp "Instance File") | 480 | 36  | 1000 | [1,611.28](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_16.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/141 "Plotted Instance") |
| [Golden\_17](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_17.vrp "Instance File") | 240 | 22  | 200 | [707.76](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_17.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/142 "Plotted Instance") |
| [Golden\_18](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_18.vrp "Instance File") | 300 | 27  | 200 | [995.13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_18.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/143 "Plotted Instance") |
| [Golden\_19](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_19.vrp "Instance File") | 360 | 33  | 200 | [1,365.60](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_19.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/144 "Plotted Instance") |
| [Golden\_20](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_20.vrp "Instance File") | 420 | 38  | 200 | [1,817.59](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden/Golden_20.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/145 "Plotted Instance") |
| **Set Li ([Li et al., 2005](https://doi.org/10.1016/j.cor.2003.10.002))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li.7z "Set File") |
| [Li\_21](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_21.vrp "Instance File") | 560 | 10  | 1200 | [16,212.83](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_21.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/146 "Plotted Instance") |
| [Li\_22](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_22.vrp "Instance File") | 600 | 15  | 900 | [14,499.04](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_22.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/147 "Plotted Instance") |
| [Li\_23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_23.vrp "Instance File") | 640 | 10  | 1400 | [18,801.13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_23.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/148 "Plotted Instance") |
| [Li\_24](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_24.vrp "Instance File") | 720 | 10  | 1500 | [21,389.43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_24.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/149 "Plotted Instance") |
| [Li\_25](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_25.vrp "Instance File") | 760 | 19  | 900 | [16,665.70](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_25.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/150 "Plotted Instance") |
| [Li\_26](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_26.vrp "Instance File") | 800 | 10  | 1700 | [23,977.73](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_26.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/151 "Plotted Instance") |
| [Li\_27](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_27.vrp "Instance File") | 840 | 20  | 900 | [17,320.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_27.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/152 "Plotted Instance") |
| [Li\_28](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_28.vrp "Instance File") | 880 | 10  | 1800 | [26,566.03](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_28.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/153 "Plotted Instance") |
| [Li\_29](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_29.vrp "Instance File") | 960 | 10  | 2000 | [29,154.34](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_29.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/154 "Plotted Instance") |
| [Li\_30](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_30.vrp "Instance File") | 1040 | 10  | 2100 | [31,742.64](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_30.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/155 "Plotted Instance") |
| [Li\_31](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_31.vrp "Instance File") | 1120 | 10  | 2300 | [34,330.94](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_31.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/156 "Plotted Instance") |
| [Li\_32](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_32.vrp "Instance File") | 1200 | 11  | 2500 | [37,159.41](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li/Li_32.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/157 "Plotted Instance") |
| **Set X ([Uchoa et al., 2017](https://doi.org/10.1016/j.ejor.2016.08.012))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X.7z "Set File") |
| [X-n101-k25](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n101-k25.vrp "Instance File") | 100 | 25  | 206 | [27,591.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n101-k25.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/158 "Plotted Instance") |
| [X-n106-k14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n106-k14.vrp "Instance File") | 105 | 14  | 600 | [26,362.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n106-k14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/159 "Plotted Instance") |
| [X-n110-k13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n110-k13.vrp "Instance File") | 109 | 13  | 66  | [14,971.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n110-k13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/160 "Plotted Instance") |
| [X-n115-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n115-k10.vrp "Instance File") | 114 | 10  | 169 | [12,747.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n115-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/161 "Plotted Instance") |
| [X-n120-k6](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n120-k6.vrp "Instance File") | 119 | 6   | 21  | [13,332.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n120-k6.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/162 "Plotted Instance") |
| [X-n125-k30](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n125-k30.vrp "Instance File") | 124 | 30  | 188 | [55,539.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n125-k30.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/163 "Plotted Instance") |
| [X-n129-k18](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n129-k18.vrp "Instance File") | 128 | 18  | 39  | [28,940.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n129-k18.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/164 "Plotted Instance") |
| [X-n134-k13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n134-k13.vrp "Instance File") | 133 | 13  | 643 | [10,916.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n134-k13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/165 "Plotted Instance") |
| [X-n139-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n139-k10.vrp "Instance File") | 138 | 10  | 106 | [13,590.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n139-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/166 "Plotted Instance") |
| [X-n143-k7](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n143-k7.vrp "Instance File") | 142 | 7   | 1190 | [15,700.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n143-k7.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/167 "Plotted Instance") |
| [X-n148-k46](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n148-k46.vrp "Instance File") | 147 | 46  | 18  | [43,448.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n148-k46.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/168 "Plotted Instance") |
| [X-n153-k22](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n153-k22.vrp "Instance File") | 152 | 22  | 144 | [21,220.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n153-k22.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/169 "Plotted Instance") |
| [X-n157-k13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n157-k13.vrp "Instance File") | 156 | 13  | 12  | [16,876.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n157-k13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/170 "Plotted Instance") |
| [X-n162-k11](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n162-k11.vrp "Instance File") | 161 | 11  | 1174 | [14,138.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n162-k11.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/171 "Plotted Instance") |
| [X-n167-k10](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n167-k10.vrp "Instance File") | 166 | 10  | 133 | [20,557.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n167-k10.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/172 "Plotted Instance") |
| [X-n172-k51](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n172-k51.vrp "Instance File") | 171 | 51  | 161 | [45,607.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n172-k51.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/173 "Plotted Instance") |
| [X-n176-k26](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n176-k26.vrp "Instance File") | 175 | 26  | 142 | [47,812.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n176-k26.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/174 "Plotted Instance") |
| [X-n181-k23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n181-k23.vrp "Instance File") | 180 | 23  | 8   | [25,569.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n181-k23.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/175 "Plotted Instance") |
| [X-n186-k15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n186-k15.vrp "Instance File") | 185 | 15  | 974 | [24,145.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n186-k15.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/176 "Plotted Instance") |
| [X-n190-k8](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n190-k8.vrp "Instance File") | 189 | 8   | 138 | [16,980.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n190-k8.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/177 "Plotted Instance") |
| [X-n195-k51](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n195-k51.vrp "Instance File") | 194 | 51  | 181 | [44,225.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n195-k51.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/178 "Plotted Instance") |
| [X-n200-k36](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n200-k36.vrp "Instance File") | 199 | 36  | 402 | [58,578.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n200-k36.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/179 "Plotted Instance") |
| [X-n204-k19](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n204-k19.vrp "Instance File") | 203 | 19  | 836 | [19,565.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n204-k19.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/180 "Plotted Instance") |
| [X-n209-k16](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n209-k16.vrp "Instance File") | 208 | 16  | 101 | [30,656.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n209-k16.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/181 "Plotted Instance") |
| [X-n214-k11](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n214-k11.vrp "Instance File") | 213 | 11  | 944 | [10,856.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n214-k11.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/182 "Plotted Instance") |
| [X-n219-k73](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n219-k73.vrp "Instance File") | 218 | 73  | 3   | [117,595.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n219-k73.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/183 "Plotted Instance") |
| [X-n223-k34](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n223-k34.vrp "Instance File") | 222 | 34  | 37  | [40,437.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n223-k34.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/184 "Plotted Instance") |
| [X-n228-k23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n228-k23.vrp "Instance File") | 227 | 23  | 154 | [25,742.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n228-k23.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/185 "Plotted Instance") |
| [X-n233-k16](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n233-k16.vrp "Instance File") | 232 | 16  | 631 | [19,230.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n233-k16.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/186 "Plotted Instance") |
| [X-n237-k14](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n237-k14.vrp "Instance File") | 236 | 14  | 18  | [27,042.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n237-k14.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/187 "Plotted Instance") |
| [X-n242-k48](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n242-k48.vrp "Instance File") | 241 | 48  | 28  | [82,751.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n242-k48.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/188 "Plotted Instance") |
| [X-n247-k50](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n247-k50.vrp "Instance File") | 246 | 47  | 134 | [37,274.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n247-k50.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/189 "Plotted Instance") |
| [X-n251-k28](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n251-k28.vrp "Instance File") | 250 | 28  | 69  | [38,684.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n251-k28.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/190 "Plotted Instance") |
| [X-n256-k16](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n256-k16.vrp "Instance File") | 255 | 16  | 1225 | [18,839.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n256-k16.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/191 "Plotted Instance") |
| [X-n261-k13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n261-k13.vrp "Instance File") | 260 | 13  | 1081 | [26,558.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n261-k13.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/192 "Plotted Instance") |
| [X-n266-k58](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n266-k58.vrp "Instance File") | 265 | 58  | 35  | [75,478.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n266-k58.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/193 "Plotted Instance") |
| [X-n270-k35](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n270-k35.vrp "Instance File") | 269 | 35  | 585 | [35,291.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n270-k35.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/194 "Plotted Instance") |
| [X-n275-k28](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n275-k28.vrp "Instance File") | 274 | 28  | 10  | [21,245.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n275-k28.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/195 "Plotted Instance") |
| [X-n280-k17](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n280-k17.vrp "Instance File") | 279 | 17  | 192 | [33,503.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n280-k17.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/196 "Plotted Instance") |
| [X-n284-k15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n284-k15.vrp "Instance File") | 283 | 15  | 109 | [20,215.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n284-k15.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/197 "Plotted Instance") |
| [X-n289-k60](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n289-k60.vrp "Instance File") | 288 | 60  | 267 | [95,151.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n289-k60.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/198 "Plotted Instance") |
| [X-n294-k50](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n294-k50.vrp "Instance File") | 293 | 50  | 285 | [47,161.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n294-k50.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/199 "Plotted Instance") |
| [X-n298-k31](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n298-k31.vrp "Instance File") | 297 | 31  | 55  | [34,231.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n298-k31.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/200 "Plotted Instance") |
| [X-n303-k21](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n303-k21.vrp "Instance File") | 302 | 21  | 794 | [21,736.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n303-k21.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/201 "Plotted Instance") |
| [X-n308-k13](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n308-k13.vrp "Instance File") | 307 | 13  | 246 | [25,859.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n308-k13.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/202 "Plotted Instance") |
| [X-n313-k71](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n313-k71.vrp "Instance File") | 312 | 71  | 248 | [94,043.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n313-k71.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/203 "Plotted Instance") |
| [X-n317-k53](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n317-k53.vrp "Instance File") | 316 | 53  | 6   | [78,355.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n317-k53.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/204 "Plotted Instance") |
| [X-n322-k28](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n322-k28.vrp "Instance File") | 321 | 28  | 868 | [29,834.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n322-k28.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/205 "Plotted Instance") |
| [X-n327-k20](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n327-k20.vrp "Instance File") | 326 | 20  | 128 | [27,532.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n327-k20.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/206 "Plotted Instance") |
| [X-n331-k15](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n331-k15.vrp "Instance File") | 330 | 15  | 23  | [31,102.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n331-k15.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/207 "Plotted Instance") |
| [X-n336-k84](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n336-k84.vrp "Instance File") | 335 | 84  | 203 | [139,111.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n336-k84.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/208 "Plotted Instance") |
| [X-n344-k43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n344-k43.vrp "Instance File") | 343 | 43  | 61  | [42,050.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n344-k43.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/209 "Plotted Instance") |
| [X-n351-k40](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n351-k40.vrp "Instance File") | 350 | 40  | 436 | [25,896.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n351-k40.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/210 "Plotted Instance") |
| [X-n359-k29](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n359-k29.vrp "Instance File") | 358 | 29  | 68  | [51,505.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n359-k29.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/211 "Plotted Instance") |
| [X-n367-k17](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n367-k17.vrp "Instance File") | 366 | 17  | 218 | [22,814.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n367-k17.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/212 "Plotted Instance") |
| [X-n376-k94](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n376-k94.vrp "Instance File") | 375 | 94  | 4   | [147,713.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n376-k94.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/213 "Plotted Instance") |
| [X-n384-k52](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n384-k52.vrp "Instance File") | 383 | 52  | 564 | [65,928.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n384-k52.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/214 "Plotted Instance") |
| [X-n393-k38](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n393-k38.vrp "Instance File") | 392 | 38  | 78  | [38,260.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n393-k38.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/215 "Plotted Instance") |
| [X-n401-k29](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n401-k29.vrp "Instance File") | 400 | 29  | 745 | [66,154.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n401-k29.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/216 "Plotted Instance") |
| [X-n411-k19](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n411-k19.vrp "Instance File") | 410 | 19  | 216 | [19,712.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n411-k19.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/217 "Plotted Instance") |
| [X-n420-k130](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n420-k130.vrp "Instance File") | 419 | 130 | 18  | [107,798.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n420-k130.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/218 "Plotted Instance") |
| [X-n429-k61](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n429-k61.vrp "Instance File") | 428 | 61  | 536 | [65,449.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n429-k61.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/219 "Plotted Instance") |
| [X-n439-k37](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n439-k37.vrp "Instance File") | 438 | 37  | 12  | [36,391.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n439-k37.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/220 "Plotted Instance") |
| [X-n449-k29](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n449-k29.vrp "Instance File") | 448 | 29  | 777 | [55,233.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n449-k29.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/221 "Plotted Instance") |
| [X-n459-k26](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n459-k26.vrp "Instance File") | 458 | 26  | 1106 | [24,139.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n459-k26.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/222 "Plotted Instance") |
| [X-n469-k138](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n469-k138.vrp "Instance File") | 468 | 138 | 256 | [221,824.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n469-k138.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/223 "Plotted Instance") |
| [X-n480-k70](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n480-k70.vrp "Instance File") | 479 | 70  | 52  | [89,449.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n480-k70.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/224 "Plotted Instance") |
| [X-n491-k59](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n491-k59.vrp "Instance File") | 490 | 59  | 428 | [66,483.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n491-k59.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/225 "Plotted Instance") |
| [X-n502-k39](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n502-k39.vrp "Instance File") | 501 | 39  | 13  | [69,226.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n502-k39.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/226 "Plotted Instance") |
| [X-n513-k21](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n513-k21.vrp "Instance File") | 512 | 21  | 142 | [24,201.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n513-k21.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/227 "Plotted Instance") |
| [X-n524-k153](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n524-k153.vrp "Instance File") | 523 | 137 | 125 | [154,593.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n524-k153.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/228 "Plotted Instance") |
| [X-n536-k96](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n536-k96.vrp "Instance File") | 535 | 96  | 371 | [94,846.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n536-k96.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/229 "Plotted Instance") |
| [X-n548-k50](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n548-k50.vrp "Instance File") | 547 | 50  | 11  | [86,700.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n548-k50.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/230 "Plotted Instance") |
| [X-n561-k42](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n561-k42.vrp "Instance File") | 560 | 42  | 74  | [42,717.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n561-k42.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/231 "Plotted Instance") |
| [X-n573-k30](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n573-k30.vrp "Instance File") | 572 | 30  | 210 | [50,673.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n573-k30.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/232 "Plotted Instance") |
| [X-n586-k159](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n586-k159.vrp "Instance File") | 585 | 159 | 28  | [190,316.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n586-k159.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/233 "Plotted Instance") |
| [X-n599-k92](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n599-k92.vrp "Instance File") | 598 | 92  | 487 | [108,451.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n599-k92.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/234 "Plotted Instance") |
| [X-n613-k62](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n613-k62.vrp "Instance File") | 612 | 62  | 523 | [59,535.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n613-k62.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/235 "Plotted Instance") |
| [X-n627-k43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n627-k43.vrp "Instance File") | 626 | 43  | 110 | [62,164.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n627-k43.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/236 "Plotted Instance") |
| [X-n641-k35](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n641-k35.vrp "Instance File") | 640 | 35  | 1381 | [63,682.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n641-k35.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/237 "Plotted Instance") |
| [X-n655-k131](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n655-k131.vrp "Instance File") | 654 | 131 | 5   | [106,780.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n655-k131.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/238 "Plotted Instance") |
| [X-n670-k130](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n670-k130.vrp "Instance File") | 669 | 126 | 129 | [146,332.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n670-k130.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/239 "Plotted Instance") |
| [X-n685-k75](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n685-k75.vrp "Instance File") | 684 | 75  | 408 | [68,205.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n685-k75.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/240 "Plotted Instance") |
| [X-n701-k44](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n701-k44.vrp "Instance File") | 700 | 44  | 87  | [81,923.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n701-k44.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/241 "Plotted Instance") |
| [X-n716-k35](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n716-k35.vrp "Instance File") | 715 | 35  | 1007 | [43,373.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n716-k35.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/242 "Plotted Instance") |
| [X-n733-k159](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n733-k159.vrp "Instance File") | 732 | 159 | 25  | [136,187.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n733-k159.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/243 "Plotted Instance") |
| [X-n749-k98](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n749-k98.vrp "Instance File") | 748 | 98  | 396 | [77,269.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n749-k98.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/244 "Plotted Instance") |
| [X-n766-k71](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n766-k71.vrp "Instance File") | 765 | 71  | 166 | [114,417.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n766-k71.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/245 "Plotted Instance") |
| [X-n783-k48](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n783-k48.vrp "Instance File") | 782 | 48  | 832 | [72,386.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n783-k48.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/246 "Plotted Instance") |
| [X-n801-k40](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n801-k40.vrp "Instance File") | 800 | 40  | 20  | [73,305.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n801-k40.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/247 "Plotted Instance") |
| [X-n819-k171](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n819-k171.vrp "Instance File") | 818 | 171 | 358 | [158,121.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n819-k171.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/248 "Plotted Instance") |
| [X-n837-k142](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n837-k142.vrp "Instance File") | 836 | 142 | 44  | [193,737.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n837-k142.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/249 "Plotted Instance") |
| [X-n856-k95](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n856-k95.vrp "Instance File") | 855 | 95  | 9   | [88,965.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n856-k95.sol "Solution File") | yes | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/250 "Plotted Instance") |
| [X-n876-k59](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n876-k59.vrp "Instance File") | 875 | 59  | 764 | [99,299.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n876-k59.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/251 "Plotted Instance") |
| [X-n895-k37](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n895-k37.vrp "Instance File") | 894 | 37  | 1816 | [53,860.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n895-k37.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/252 "Plotted Instance") |
| [X-n916-k207](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n916-k207.vrp "Instance File") | 915 | 207 | 33  | [329,179.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n916-k207.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/253 "Plotted Instance") |
| [X-n936-k151](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n936-k151.vrp "Instance File") | 935 | 151 | 138 | [132,715.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n936-k151.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/254 "Plotted Instance") |
| [X-n957-k87](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n957-k87.vrp "Instance File") | 956 | 87  | 11  | [85,465.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n957-k87.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/255 "Plotted Instance") |
| [X-n979-k58](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n979-k58.vrp "Instance File") | 978 | 58  | 998 | [118,976.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n979-k58.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/256 "Plotted Instance") |
| [X-n1001-k43](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n1001-k43.vrp "Instance File") | 1000 | 43  | 131 | [72,355.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/X-n1001-k43.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/257 "Plotted Instance") |
| **Set AGS ([Arnold et al., 2019](https://doi.org/10.1016/j.cor.2019.03.006))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS.7z "Set File") |
| [Antwerp1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Antwerp1.vrp "Instance File") | 6000 | —   | 30  | [477,277.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Antwerp1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/258 "Plotted Instance") |
| [Antwerp2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Antwerp2.vrp "Instance File") | 7000 | —   | 100 | [291,350.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Antwerp2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/259 "Plotted Instance") |
| [Brussels1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Brussels1.vrp "Instance File") | 15000 | —   | 50  | [501,719.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Brussels1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/260 "Plotted Instance") |
| [Brussels2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Brussels2.vrp "Instance File") | 16000 | —   | 150 | [345,468.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Brussels2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/261 "Plotted Instance") |
| [Flanders1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Flanders1.vrp "Instance File") | 20000 | —   | 50  | [7,240,118.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Flanders1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/262 "Plotted Instance") |
| [Flanders2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Flanders2.vrp "Instance File") | 30000 | —   | 200 | [4,373,244.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Flanders2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/263 "Plotted Instance") |
| [Ghent1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Ghent1.vrp "Instance File") | 10000 | —   | 35  | [469,531.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Ghent1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/264 "Plotted Instance") |
| [Ghent2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Ghent2.vrp "Instance File") | 11000 | —   | 170 | [257,748.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Ghent2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/265 "Plotted Instance") |
| [Leuven1](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Leuven1.vrp "Instance File") | 3000 | —   | 25  | [192,848.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Leuven1.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/266 "Plotted Instance") |
| [Leuven2](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Leuven2.vrp "Instance File") | 4000 | —   | 150 | [111,391.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS/Leuven2.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/267 "Plotted Instance") |
| **Set DIMACS ([DIMACS, 2021](http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS.7z "Set File") |
| [Loggi-n401-k23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n401-k23.vrp "Instance File") | 400 | 23  | 100 | [336,903.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n401-k23.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/268 "Plotted Instance") |
| [Loggi-n501-k24](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n501-k24.vrp "Instance File") | 500 | 24  | 120 | [177,078.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n501-k24.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/269 "Plotted Instance") |
| [Loggi-n601-k19](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n601-k19.vrp "Instance File") | 600 | 19  | 180 | [113,155.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n601-k19.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/270 "Plotted Instance") |
| [Loggi-n601-k42](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n601-k42.vrp "Instance File") | 600 | 42  | 80  | [347,046.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n601-k42.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/271 "Plotted Instance") |
| [Loggi-n901-k42](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n901-k42.vrp "Instance File") | 900 | 42  | 120 | [246,301.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n901-k42.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/272 "Plotted Instance") |
| [Loggi-n1001-k31](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n1001-k31.vrp "Instance File") | 1000 | 31  | 180 | [284,356.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/Loggi-n1001-k31.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/273 "Plotted Instance") |
| [ORTEC-n242-k12](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n242-k12.vrp "Instance File") | 241 | 12  | 125 | [123,750.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n242-k12.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/274 "Plotted Instance") |
| [ORTEC-n323-k21](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n323-k21.vrp "Instance File") | 322 | 21  | 100 | [214,071.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n323-k21.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/275 "Plotted Instance") |
| [ORTEC-n405-k18](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n405-k18.vrp "Instance File") | 404 | 18  | 160 | [200,986.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n405-k18.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/276 "Plotted Instance") |
| [ORTEC-n455-k41](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n455-k41.vrp "Instance File") | 454 | 41  | 70  | [292,485.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n455-k41.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/277 "Plotted Instance") |
| [ORTEC-n510-k23](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n510-k23.vrp "Instance File") | 509 | 23  | 145 | [184,529.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n510-k23.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/278 "Plotted Instance") |
| [ORTEC-n701-k64](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n701-k64.vrp "Instance File") | 700 | 64  | 80  | [445,541.00](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS/ORTEC-n701-k64.sol "Solution File") | no  | [](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/plots/279 "Plotted Instance") |
| **Set XML ([Queiroga et al., 2022](https://openreview.net/forum?id=yHiMXKN6nTl))** |     |     |     |     |     | [Set File](https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/XML.7z "Set File") |
| No instances in this set. |     |     |     |     |     |     |     |

---

## 4. Remarks

1. The problem `lin318` is originally a Hamiltonian path problem. One obtains this problem by adding the additional requirement that the edge from 1 to 214 is contained in the tour. The data is given in `linhp318.tsp`.
2. Some data sets are referred to by different names in the literature. Below we give the corresponding names used in [3] and [2].

| TSPLIB |    [3] |     [2]
| --- | --- | ------- |
| att48 |     ATT048 |  - |
| dantzig42 | - |       42 |
| eil51 |     EIL08 |   - |
| gil262 |    GIL249 |  - |
| gr137 |     GH137 |   137 |
| gr229 |     GH229 |   229 |
| gr666 |     GH666 |   666 |
| hk48 |      - |       48H |
| kroB100 |   KRO125 |  100B |
| kroD100 |   KRO127 |  100D |
| kroA150 |   KRO30 |   - |
| kroA200 |   KRO32 |   - |
| lin105 |    LK105 |   - |
| linhp318 |  LK318P |  - |
| pr107 |     TK107 |   - |
| pr136 |     TK136 |   - |
| pr152 |     TK152 |   - |
| pr2392 |    TK2392 |  - |
| pr299 |     TK299 |   - |
| pr76 |      TK076 |   - |
| att532 |    ATT532 |  - |
| eil101 |    EIL10 |   - |
| eil76 |     EIL09 |   - |
| gr120 |     - |       120 |
| gr202 |     GH202 |   202 |
| gr431 |     GH431 |   431 |
| gr96 |      GH096 |   96 |
| kroA100 |   KRO124 |  100A |
| kroC100 |   KRO126 |  100C |
| kroE100 |   KRO128 |  100E |
| kroB150 |   KRO31 |   - |
| kroB200 |   KRO33 |   - |
| lin318 |    LK318 |   - |
| pr1002 |    TK1002 |  - |
| pr124 |     TK124 |   - |
| pr144 |     TK144 |   - |
| pr226 |     TK226 |   - |
| pr264 |     TK264 |   - |
| pr439 |     TK439 |   - |
| st70 |      KRO070 |  70 |

3. Some vehicle routing problems are also available in a TSP version. Here the depots are just treated as normal nodes. The problem `gil262` originally contained two identical nodes, of which one was eliminated.
4. Potential contributors to this library should provide their data files in appropriate format and contact

   Gerhard Reinelt  
   Institut für Angewandte Mathematik, Universität Heidelberg  
   Im Neuenheimer Feld 294, D-69120 Heidelberg, Germany  
   Tel (6221) 56 3171  
   Fax (6221) 56 5634  
   E-Mail: <Gerhard.Reinelt@IWR.Uni-Heidelberg.DE>

5. Informations on new bounds or optimal solutions for library problems as well as references to computational studies (to be included in the list of references) are also appreciated.

---

## 5. Access

TSPLIB is available at:  
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>

---

## References

1. R.E. Bland & D.F. Shallcross (1989). _Large Traveling Salesman Problems Arising from Experiments in X-ray Crystallography: A Preliminary Report on Computation_, Operations Research Letters 8, 125–128.
2. M. Grötschel & O. Holland (1991). _Solution of Large-Scale Symmetric Travelling Salesman Problems_, Mathematical Programming 51, 141–202.
3. M.W. Padberg & G. Rinaldi (1991). _A Branch & Cut Algorithm for the Resolution of Large-scale Symmetric Traveling Salesman Problems_, SIAM Review 33, 60–100.
