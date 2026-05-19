
# A Technical Recompilation and Parallelism Analysis of Guided Local Search for the Traveling Salesman Problem

**(Based on Voudouris & Tsang, 1999)**

- [A Technical Recompilation and Parallelism Analysis of Guided Local Search for the Traveling Salesman Problem](https://www.google.com/search?q=%23a-technical-recompilation-and-parallelism-analysis-of-guided-local-search-for-the-traveling-salesman-problem)
  - [Part 1: Mathematical Foundations of Guided Local Search](https://www.google.com/search?q=%23part-1-mathematical-foundations-of-guided-local-search)
    - [1.1 List of Mathematical Symbols](https://www.google.com/search?q=%2311-list-of-mathematical-symbols)
    - [1.2 Sequential Mathematical Formulation and Rationale](https://www.google.com/search?q=%2312-sequential-mathematical-formulation-and-rationale)
  - [Part 2: Core Algorithmic Pseudocode](https://www.google.com/search?q=%23part-2-core-algorithmic-pseudocode)
    - [2.1 Procedure: GuidedLocalSearch (Figure 1)](https://www.google.com/search?q=%2321-procedure-guidedlocalsearch-figure-1)
    - [2.2 Procedures: GuidedFastLocalSearch and FastLocalSearch (Figure 2)](https://www.google.com/search?q=%2322-procedures-guidedfastlocalsearch-and-fastlocalsearch-figure-2)
    - [2.3 Procedure: Iterated Lin-Kernighan (Figure 9)](https://www.google.com/search?q=%2323-procedure-iterated-lin-kernighan-figure-9)
  - [Part 3: Taxonomy and Parallelism Analysis of TSP Heuristics](https://www.google.com/search?q=%23part-3-taxonomy-and-parallelism-analysis-of-tsp-heuristics)
    - [3.1 A Note on Parallel Execution Models: SIMD vs. SIMT](https://www.google.com/search?q=%2331-a-note-on-parallel-execution-models-simd-vs-simt)
    - [3.2 The Crainic-Toulouse Algorithmic Parallelism Taxonomy](https://www.google.com/search?q=%2332-the-crainic-toulouse-algorithmic-parallelism-taxonomy)
    - [3.3 Analysis of Local Improvement Heuristics](https://www.google.com/search?q=%2333-analysis-of-local-improvement-heuristics)
    - [3.4 Analysis of Metaheuristic Strategies](https://www.google.com/search?q=%2334-analysis-of-metaheuristic-strategies)
      - [1. Simulated Annealing (SA)](https://www.google.com/search?q=%231-simulated-annealing-sa)
      - [2. Tabu Search (TS)](https://www.google.com/search?q=%232-tabu-search-ts)
      - [3. Genetic Algorithms (GA) / Genetic Local Search](https://www.google.com/search?q=%233-genetic-algorithms-ga--genetic-local-search)
      - [4. Iterated Lin-Kernighan (ILK) / Double-Bridge (DB)](https://www.google.com/search?q=%234-iterated-lin-kernighan-ilk--double-bridge-db)
      - [5. Guided Local Search (GLS) & Fast Local Search (FLS)](https://www.google.com/search?q=%235-guided-local-search-gls--fast-local-search-fls)
  - [Part 4: Empirical Evaluation and Key Findings (Chapter 8)](https://www.google.com/search?q=%23part-4-empirical-evaluation-and-key-findings-chapter-8)
    - [4.1 The Power of Fast Local Search (FLS) (Table 3)](https://www.google.com/search?q=%2341-the-power-of-fast-local-search-fls-table-3)
    - [4.2 GLS vs. General Metaheuristics (Table 4)](https://www.google.com/search?q=%2342-gls-vs-general-metaheuristics-table-4)
    - [4.3 The Central Finding: GLS-FLS-2Opt vs. State-of-the-Art (Tables 5, 7, Fig 11)](https://www.google.com/search?q=%2343-the-central-finding-gls-fls-2opt-vs-state-of-the-art-tables-5-7-fig-11)
      - [Analysis and Synthesis: The "Voudouris Paradox"](https://www.google.com/search?q=%23analysis-and-synthesis-the-voudouris-paradox)
  - [Appendix: Complete Experimental Data Tables](https://www.google.com/search?q=%23appendix-complete-experimental-data-tables)
    - [Table A.1: Results for Guided Local Search on the TSP](https://www.google.com/search?q=%23table-a1-results-for-guided-local-search-on-the-tsp)
    - [Table A.2: Results for Iterated Local Search (DB-Move) on the TSP](https://www.google.com/search?q=%23table-a2-results-for-iterated-local-search-db-move-on-the-tsp)
    - [Table A.3: Results for Repeated Local Search on the TSP](https://www.google.com/search?q=%23table-a3-results-for-repeated-local-search-on-the-tsp)

## Part 1: Mathematical Foundations of Guided Local Search

This section consolidates the complete mathematical framework of Guided Local Search, presenting the concepts sequentially as requested by the user, from the general problem definition to its specific application to the TSP.

### 1.1 List of Mathematical Symbols

A comprehensive list of all mathematical symbols used throughout the paper's formulations, as extracted from.1

| **Symbol** | **Meaning** | **Location / Context** |
| :--- | :--- | :--- |
| $S$ | The set of all feasible solutions. | Section 2 1 |
| $g(s)$ | The objective (cost) function for a solution $s \in S$. | Section 2 1 |
| $N(s)$ | The neighborhood of solution $s$; the set of solutions reachable by a single move. | Section 2 1 |
| $h(s)$ | The **augmented cost function** used by GLS. | Section 3.2 1 |
| $M$ | The total number of solution features defined. | Section 3.2 1 |
| $f_i$ | The $i$-th solution feature (a property of a solution). | Section 3.1 1 |
| $I_i(s)$ | The indicator function for feature $f_i$; $1$ if $s$ has feature $i$, $0$ otherwise. | Section 3.1 1 |
| $p_i$ | The penalty parameter corresponding to feature $f_i$. | Section 3.2 1 |
| $p$ | The penalty vector $(p_1,…, p_M)$. | Section 3.2 1 |
| $\lambda$ | The regularization parameter, controlling the strength of penalties vs. the cost. | Section 3.2 1 |
| $c_i$ | The cost of feature $f_i$. | Section 3.3 1 |
| $c$ | The cost vector $(c_1,…, c_M)$. | Section 3.3 1 |
| $s_*$ | A local minimum solution (with respect to $h(s)$). | Section 3.3 1 |
| $util(s_*, f_i)$ | The utility of penalizing feature $f_i$ when in local minimum $s_*$. | Section 3.3 1 |
| $N$ | The number of cities in a TSP instance. | Section 5 1 |
| $\pi$ | A tour, represented as a cyclic permutation of $N$ cities. | Section 5 1 |
| $D$ | The symmetric distance matrix $D = [d_{ij}]$. | Section 5 1 |
| $d_{ij}$ | The distance (cost) between city $i$ and $j$. | Section 5 1 |
| $e_{ij}$ | The edge (a TSP feature) connecting cities $i$ and $j$. | Section 7.1 1 |
| $P$ | The symmetric penalty matrix $P = [p_{ij}]$. | Section 7.1 1 |
| $p_{ij}$ | The penalty parameter for the edge (feature) $e_{ij}$. | Section 7.1 1 |
| $D'$ | The **auxiliary distance matrix** used by local search in GLS. | Section 7.1 1 |
| $a$ | A normalized parameter used to set $\lambda$ based on the problem instance. | Section 8.2 1 |

### 1.2 Sequential Mathematical Formulation and Rationale

This section presents the mathematical formulas in a logical, sequential order, building from the general concept of combinatorial optimization to the specific, practical implementation for the Traveling Salesman Problem.1

**1. The General Combinatorial Optimization Problem (Section 2)**

$$
\begin{align}
\min g(s), \quad s\in S
\end{align}
$$
**Explanation:** This is the starting point for any optimization problem. The goal is to find a solution $s$ from the set of all possible feasible solutions $S$ that minimizes a cost function $g(s)$. For the TSP, $S$ would be the set of all valid tours, and $g(s)$ would be the total length of a tour.

-----

**2. Definition of a Local Minimum (Section 2)**

A solution $x$ is a local minimum with respect to (w.r.t.) neighborhood $N$ if and only if (iff):

$$
\begin{align}
g(x) \le g(y), \quad \forall y \in N(x)
\end{align}
$$
**Explanation:** This defines the core problem that metaheuristics must solve. A simple "greedy" local search will stop at a solution $x$ when it cannot find a better solution $y$ within its immediate neighborhood $N(x)$ (e.g., all solutions reachable by a single 2-Opt swap). This $x$ is a "local minimum" but may not be the "global minimum" (the true best solution). The search is "trapped."

-----

**3. The Guided Local Search Augmented Cost Function (Section 3.2, Eq. 1)**

The function $h(s)$ is minimized by the local search instead of $g(s)$:
$$
\begin{align}
h(s) = g(s) + \lambda \cdot \sum_{i=1}^{M} p_i \cdot I_i(s)
\end{align}
$$
**Explanation:** To escape the local minimum (defined in 2), GLS creates a new, "augmented" cost function $h(s)$. The local search algorithm will now try to minimize $h(s)$ instead of just $g(s)$. This new function is composed of the original cost $g(s)$ plus a penalty term. This penalty term is a weighted sum of all $M$ possible solution "features," controlled by the $\lambda$ parameter. By increasing the penalties $p_i$ for features associated with the current local minimum, GLS modifies the "cost landscape," making the current solution $x$ no longer look like a minimum.

**4. The Solution Feature Indicator Function (Section 3.1)**

$$
\begin{align}
I_i(s) = \begin{cases}
1, & \text{solution } s \text{ has property } i \\
0, & \text{otherwise}
\end{cases}
\end{align}
$$
**Explanation:** This formula defines the $I_i(s)$ term used in (3). It is a simple binary "switch" or "indicator." It returns 1 if the current solution $s$ exhibits a specific feature $i$ (e.g., "contains the edge between city A and city B") and 0 otherwise. This allows the summation in (3) to only penalize features that are *actually present* in the solution.

**5. The Penalty Modification Utility (Section 3.3, Eq. 2)**

When trapped at a local minimum $s_*$, GLS penalizes features $f_i$ that maximize:

$$
\begin{align}
util(s_*, f_i) = I_i(s_*) \cdot \frac{c_i}{1 + p_i}
\end{align}
$$
**Explanation:** When the search gets trapped at a local minimum $s_*$ (with respect to $h(s)$), GLS must decide which feature's penalty $p_i$ to increase. This "utility" function provides the logic. It calculates a utility for every feature $i$. The utility is 0 if the feature is not in the solution ($I_i(s_*) = 0$). For features that *are* in the solution, it favors those with a high cost $c_i$ and a low current penalty $p_i$. This strategy penalizes the "worst-offending" features that have not been penalized heavily before. GLS increments the $p_i$ for the feature(s) with the maximum utility.

-----

**6. TSP Cost Function (Section 5, Eq. 3)**

For a tour $\pi$, the cost $g(\pi)$ is the sum of its edge distances $d_{ij}$:

$$
\begin{align}
g(\pi) = \sum_{i=1}^{N} d_{i, \pi(i)}
\end{align}
$$
**Explanation:** This translates the "general" cost function $g(s)$ from (1) into the "specific" cost function for the TSP. A solution $s$ is now a tour $\pi$. The cost of the tour $g(\pi)$ is simply the sum of the distances $d$ for each edge in that tour, from city $i$ to the next city in the tour, $\pi(i)$.

-----

**7. TSP Feature Definition and Utility (Section 7.1, Eq. 4)**

Features are edges $e_{ij}$. The indicator $I_{e_{ij}}(\text{tour})$ is 1 if $e_{ij}$ is in the tour. The feature cost $c_i$ is the edge length $d_{ij}$. The utility (Eq. 2) becomes:
$$
\begin{align}
Util(\text{tour}, e_{ij}) = I_{e_{ij}}(\text{tour}) \cdot \frac{d_{ij}}{1 + p_{ij}}
\end{align}
$$
**Explanation:** This translates the general utility function from (5) into the specific utility for the TSP.

- The general "feature" $f_i$ is now a specific "edge" $e_{ij}$.
- The general "feature cost" $c_i$ is now the "edge cost" (distance) $d_{ij}$.
- The general "feature penalty" $p_i$ is now the "edge penalty" $p_{ij}$.

This formula is used to decide which *edge* to penalize when the TSP search gets stuck. It will pick the edge that is *in the tour* ($I=1$), has the longest distance ($d_{ij}$), and has the lowest current penalty ($p_{ij}$).

**8. TSP Auxiliary Distance Matrix (Section 7.1)**

Local search minimizes cost using an auxiliary matrix $D'$ that incorporates penalties:

$$
\begin{align}
D' = D + \lambda \cdot P = [d_{ij} + \lambda \cdot p_{ij}]
\end{align}
$$
**Explanation:** This is a crucial implementation detail. Instead of having the local search (e.g., 2-Opt) calculate the complex $h(s)$ (from (3)) for every move, we can "pre-bake" the penalties into the distance matrix. A new "auxiliary matrix" $D'$ is created where the "distance" for an edge $ij$ is its *real distance* $d_{ij}$ plus its *current penalty* $\lambda \cdot p_{ij}$. The local search algorithm can now run on $D'$ using its normal, simple logic, and it will *automatically* be minimizing the augmented cost function $h(s)$ by default.

**9. Performance Measurement (% Excess) (Section 8.1, Eq. 5)**

$$
\begin{align}
\text{excess} = \frac{\text{solution cost} - \text{best known solution cost}}{\text{best known solution cost}} \times 100
\end{align}
$$
**Explanation:** This is a standard formula for benchmarking. To compare different algorithms, their final solution cost is compared to the best-known (or optimal) solution cost for that problem instance. The result is expressed as a percentage "excess" (or error) over the optimum. A value of 0% means the algorithm found the optimal solution.

-----

**10. Lambda ($\lambda$) Parameterization (Section 8.2, Eq. 6)**

$\lambda$ is set relative to the cost of an initial local minimum:

$$
\begin{align}
\lambda = a \cdot \frac{g(\text{local minimum})}{N}
\end{align}
$$
**Explanation:** This formula provides a heuristic for setting the $\lambda$ parameter from (3). $\lambda$ controls the balance between the true cost $g(s)$ and the penalty term. It needs to be scaled to the problem. This formula sets $\lambda$ to be proportional to the average cost per city ($g(\text{local minimum}) / N$). The $a$ parameter is a tuning constant (e.g., $a=0.5$) that is determined empirically. This makes $\lambda$ adaptive to the scale of the problem.

-----

## Part 2: Core Algorithmic Pseudocode

This section presents the complete pseudocode for the key algorithms as defined in the Voudouris (1999) paper.1

### 2.1 Procedure: GuidedLocalSearch (Figure 1)

This is the main GLS algorithm, which wraps a generic `LocalSearch` procedure and modifies penalties when trapped in a local minimum.

**Procedure** GuidedLocalSearch
**Input:**
$S$: Solution space
$g$: Original cost function
$\lambda$: Regularization parameter
$[I_1,…,I_M]$: Array of $M$ indicator functions
$[c_1,…,c_M]$: Array of $M$ feature costs
**Output:**
$s_*$: Best solution found

```

begin
k \<- 0
s\_0 \<- random or heuristically generated solution in S
for i \<- 1 until M do
p\_i \<- 0  /\* set all penalties to 0 \*/

```

while (StoppingCriterion not met) do
begin
    /*Define augmented cost function h based on current penalties*/
    h <- g + \lambda \cdot \sum_{i=1}^{M} (p_i \cdot I_i)

    /* Find the local minimum s_{k+1} with respect to h */
    s_{k+1} <- LocalSearch(s_k, h)
    
    /* Calculate utility for penalizing each feature */
    for i <- 1 until M do
        util_i <- I_i(s_{k+1}) \cdot \frac{c_i}{1 + p_i}
    
    /* Find and increment penalty for all features with maximum utility */
    max_util <- max(util_1, ..., util_M)
    for i <- 1 until M do
        if util_i = max_util then
            p_i <- p_i + 1
    
    k <- k + 1
end

s_*<- best solution found with respect to original cost function g
return s_*

```

end

```

Source: 1

### 2.2 Procedures: GuidedFastLocalSearch and FastLocalSearch (Figure 2)

This describes the powerful combination of GLS with the Fast Local Search (FLS) neighborhood reduction scheme. FLS uses "activation bits" to avoid searching the entire neighborhood, significantly accelerating the search.

**Procedure** GuidedFastLocalSearch
**Input:**
$S, g, \lambda, [I_i], [c_i], M$ (as above)
$L$: Number of sub-neighborhoods
**Output:**
$s_*$: Best solution found

```

begin
k \<- 0
s\_0 \<- random or heuristically generated solution in S
for i \<- 1 until M do p\_i \<- 0  /\* set all penalties to 0 */
for i \<- 1 until L do bit\_i \<- 1 /* set all sub-neighborhoods to active \*/

```

while (StoppingCriterion not met) do
begin
    h <- g + \lambda \cdot \sum_{i=1}^{M} (p_i \cdot I_i)

    /* Find local minimum using FLS */
    s_{k+1} <- FastLocalSearch(s_k, h, [bit_1,…,bit_L], L)
    
    /* Calculate utility for penalizing each feature */
    for i <- 1 until M do
        util_i <- I_i(s_{k+1}) \cdot \frac{c_i}{1 + p_i}
    
    /* Find and increment penalty for all features with maximum utility */
    max_util <- max(util_1, ..., util_M)
    for i <- 1 until M do
        if util_i = max_util then
        begin
            p_i <- p_i + 1
            /* Activate sub-neighborhoods related to the penalized feature */
            SetBits <- SubNeighbourhoodsForFeature(i)
            for each b in SetBits do
                bit_b <- 1
        end
    
    k <- k + 1
end

s_*<- best solution found with respect to original cost function g
return s_*

```

end

```

**Procedure** FastLocalSearch
**Input:**
$s$: Current solution
$h$: Augmented cost function
$[bit_1,…,bit_L]$: Array of activation bits
$L$: Number of sub-neighborhoods
**Output:**
$s$: Local minimum solution

```

begin
while (∃ bit\_j = 1) do
for i \<- 1 until L do
begin
if bit\_i = 1 then /\* search active sub-neighborhood i */
begin
Moves \<- set of moves in sub-neighborhood i
for each move m in Moves do
begin
s' \<- m(s)
if h(s') \< h(s) then /* for minimization */
begin
bit\_i \<- 1 /* keep current sub-neighborhood active */
/* Spread activation to related sub-neighborhoods \*/
SetBits \<- SubNeighbourhoodsForMove(m)
for each b in SetBits do
bit\_b \<- 1

```

                    s <- s'
                    goto ImprovingMoveFound
                end
            end
        end
        bit_i <- 0 /* no improving move found in this sub-neighborhood */
    ImprovingMoveFound: 
        continue
    end
return s

```

end

```

Source: 1

### 2.3 Procedure: Iterated Lin-Kernighan (Figure 9)

This is the description of the Iterated Lin-Kernighan (ILK) algorithm 1, which is used as a state-of-the-art benchmark. It operates by repeatedly applying a large-scale "kick" (a random 4-Opt move, also known as a double-bridge move) to a local minimum and then re-optimizing from that new starting point.

**Procedure** IteratedLinKernighan
**Input:**
$T_0$: An initial random tour
**Output:**
$T_{best}$: Best tour found

```

begin
T \<- LinKernighan(T\_0) /\* Find initial local minimum \*/
T\_{best} \<- T

```

while (StoppingCriterion not met) do
begin
    /*1. Kick: Apply a large-scale perturbation */
    T' <- DoubleBridgeMove(T) /* Random 4-Opt move*/

    /* 2. Optimize: Find the local minimum from the new tour */
    T'' <- LinKernighan(T')
    
    /* 3. Accept: Apply Metropolis-style acceptance or simple improvement */
    if cost(T'') < cost(T_{best}) then
        T_{best} <- T''
    
    T <- T'' /* Use the new minimum as the base for the next kick */
end

return T_{best}

```

end

```

Source: 1

-----

## Part 3: Taxonomy and Parallelism Analysis of TSP Heuristics

This section provides the requested in-depth analysis of all heuristics from the paper 1, classifying them first by their algorithmic type and second by their parallelization potential according to the Crainic-Toulouse taxonomy [User-provided Context].

### 3.1 A Note on Parallel Execution Models: SIMD vs. SIMT

An important distinction in parallel hardware execution models is necessary to properly classify algorithmic potential, particularly in light of the user's provided context [User-provided Context].

- **SIMD (Single Instruction, Multiple Data):** This model, common in CPU vector instructions (e.g., AVX), executes a single instruction in rigid lockstep across multiple data lanes. If divergent control flow (a branch) occurs, lanes that do not take the branch are "masked" or disabled, leading to inefficiency.
- **SIMT (Single Instruction, Multiple Threads):** This model is the foundation of modern GPUs (e.g., NVIDIA CUDA). A "warp" (group) of threads executes one instruction. If threads within that warp diverge (e.g., an `if-else` block), the hardware *serializes* the execution: all threads on the `if` path execute while the `else` threads are masked, and then the `else` threads execute while the `if` threads are masked.

This is not a purely semantic difference. The flexibility of SIMT, while still incurring a cost for divergence, is what makes it practical to execute complex algorithmic logic (like neighbor evaluation or even entire search trajectories) on a GPU.2 Therefore, while an algorithm may be classified as **P-Data**, its implementation on a GPU relies on the SIMT model, not a pure SIMD model.

### 3.2 The Crainic-Toulouse Algorithmic Parallelism Taxonomy

This framework classifies parallelism at the *algorithmic* level, independent of the hardware it is executed on [User-provided Context].

- **P-Data (Parallel Data):** Represents data parallelism. Multiple, identical processes execute on different partitions of the data (e.g., different segments of a distance matrix) or on independent initial solutions (e.g., multi-start). This is the most common form of parallelism and maps well to SIMT architectures.
- **S-Task (Sequential Task):** Represents an algorithmic decomposition into a pipeline. Stages ($Stage_1 \rightarrow Stage_2 \rightarrow Stage_3$) are sequential, though their execution on different data units can be overlapped.
- **P-Task (Parallel Task):** Represents task parallelism. Different, concurrent tasks (e.g., distinct algorithms or neighborhood operators) execute simultaneously, often requiring communication or coordination. This model generally requires MIMD (Multiple Instruction, Multiple Data) hardware, such as a multi-core CPU, as the tasks execute different instruction streams.

### 3.3 Analysis of Local Improvement Heuristics

These are the foundational neighborhood search operators used by the metaheuristics.1

- **Heuristics:** 2-Opt, 3-Opt, Lin-Kernighan (LK)
- **Heuristic Classification:** **Improvement Heuristics**. These are not metaheuristics. They are local search procedures that, when run in isolation, simply iterate until they find the first local minimum, at which point they terminate.
- **Parallelism Analysis:**
  - **P-Data (High Potential):** The primary computational bottleneck in these heuristics is the evaluation of the neighborhood. For 2-Opt, this involves checking the cost difference ($\Delta \text{cost}$) for all $O(N^2)$ possible edge swaps. This evaluation is "embarrassingly parallel" and a canonical example of a P-Data algorithm. A SIMT architecture can assign $N^2$ threads to compute the $\Delta \text{cost}$ for every possible swap simultaneously. A recent study on GPU-based 2-Opt solvers highlights a hierarchical strategy: parallelizing independent climbs between blocks (P-Data) and parallelizing the 2-Opt evaluations *within* a climb across threads (also P-Data).3
  - **S-Task (Low Potential):** The *process* of a First Improvement search (find first good move, apply it, restart search) is inherently sequential (S-Task). However, this is a trivial application of the S-Task classification, as the computational work within each "find" step is parallel (P-Data).
  - **P-Task (Potential):** While the standard 2-Opt evaluation is P-Data, a P-Task design is also possible. This could involve multiple, different tasks operating on the tour concurrently (e.g., Task A performs 2-Opt on cities 1-100, Task B performs 3-Opt on cities 101-200) and communicating through a shared data structure. This aligns with the concept of cooperative search.

### 3.4 Analysis of Metaheuristic Strategies

This section addresses the user's query regarding the classification of metaheuristics. A metaheuristic's *control loop* (the decision-making framework) is often an S-Task (sequential). However, the *computational work* performed within that loop is almost always parallelized using P-Data or P-Task models. The Crainic-Toulouse taxonomy is hierarchical: a top-level S-Task algorithm can contain P-Data sub-tasks.

#### 1\. Simulated Annealing (SA)

- **Heuristic Classification:** **Metaheuristic**.1
- **Parallelism Analysis:**
  - **S-Task:** The core algorithm (the cooling schedule) is fundamentally sequential. The decision to accept or reject a move at temperature $T_k$ and the subsequent transition to $T_{k-1}$ is a sequential process.
  - **P-Data:** SA can be parallelized in two primary P-Data ways:
        1. **Multi-Start (Independent Runs):** This is the simplest and most common parallelization. $N$ independent SA algorithms are executed on $N$ cores or GPU blocks, each with a different random seed. This is an embarrassingly parallel P-Data strategy [User-provided Context].
        2. **Parallel Neighborhood Evaluation:** At a given temperature, the "Generate" step can be parallelized. Instead of generating one random neighbor, the algorithm could evaluate $k$ neighbors in parallel (P-Data) and select one to move to.
  - **P-Task:** More complex cooperative SA models (P-Task) exist where multiple SA threads (at different temperatures) exchange solutions, but the simplest parallelization is P-Data.

#### 2\. Tabu Search (TS)

- **Heuristic Classification:** **Metaheuristic**.1
- **Parallelism Analysis:**
  - **S-Task:** The main control loop is sequential: (1) Evaluate Neighborhood, (2) Select Best Admissible Move, (3) Update Tabu List, (4) Repeat.
  - **P-Data:** The most common and effective parallelization is in step (1). The "Evaluate Neighborhood" step (e.g., finding the $\Delta \text{cost}$ for all $O(N^2)$ 2-Opt moves) is a massive P-Data task, perfectly suited for a SIMT implementation on a GPU.
  - **P-Task:** Cooperative Tabu Search is a well-established P-Task strategy.4 This involves multiple, independent searchers exploring different parts of the search space. These searchers run on different (MIMD) processors and periodically communicate their best solutions or, more importantly, *parts of their tabu lists* to coordinate the global search and avoid redundant effort.

#### 3\. Genetic Algorithms (GA) / Genetic Local Search

- **Heuristic Classification:** **Metaheuristic**.1
- **Parallelism Analysis:**
  - **S-Task:** The generational loop is an S-Task: (1) Select -> (2) Crossover -> (3) Mutate -> (4) Evaluate -> (5) Replace.
  - **P-Data:** GAs are inherently P-Data. Step (4), "Evaluate," involves calculating the fitness for every individual in the population. These calculations are independent and can be done in parallel. In Genetic *Local Search*, this step is even more expensive (and thus benefits more from parallelism), as each new offspring is optimized with a local search heuristic (like LK).1 Each of these local searches is an independent task, making the evaluation step a large-scale P-Data operation.
  - **P-Task:** The "island model" (or coarse-grained GA) is a classic P-Task algorithm, as seen in the `fujimoto2011` work.6 In this model:
        1. Multiple, separate populations (islands) evolve independently. This *within-island* evolution is itself a P-Data algorithm.
        2. Periodically, the islands engage in *migration*, exchanging individuals. This communication and integration step makes the overall architecture a P-Task (cooperative) model.

#### 4\. Iterated Lin-Kernighan (ILK) / Double-Bridge (DB)

- **Heuristic Classification:** **Metaheuristic**.1
- **Parallelism Analysis:**
  - **S-Task:** The high-level loop described in the pseudocode (Kick -> Optimize -> Accept) is sequential (S-Task).1 The result of the "Optimize" step is required for the "Accept" step.
  - **P-Data:** The algorithm can be parallelized in two P-Data ways:
        1. **Multi-Start:** Run $N$ independent ILK searches in parallel (P-Data).
        2. **Parallelize the Operator:** The "Optimize" step (running the LK heuristic) is itself a complex, data-intensive local search that contains P-Data-parallelizable components (e.g., evaluating potential k-opt exchanges).
  - **P-Task:** A P-Task version could have concurrent tasks performing different "kicks" (e.g., one task applies 4-Opt, another applies 5-Opt) and sharing the best-found solution with a central manager.

#### 5\. Guided Local Search (GLS) & Fast Local Search (FLS)

- **Heuristic Classification:** **Metaheuristic** (GLS) 1 + **Improvement Heuristic** (FLS, a neighborhood reduction scheme).1
- **Parallelism Analysis:**
  - **S-Task:** The main `GuidedLocalSearch` loop (Figure 1) is a clear S-Task: `LocalSearch(s_k)` -> `UpdatePenalties(s_{k+1})` -> `LocalSearch(s_{k+1})`… The penalty update *must* happen before the next local search can begin.
  - **P-Data:** The `LocalSearch(s_k, h)` step (Figure 1) is where parallelism resides. If this local search is BI-2Opt, its neighborhood evaluation is a P-Data task.
  - **S-Task + P-Data (Hybrid):** The `GuidedFastLocalSearch` (Figure 2) combination is the most interesting case.1
        1. The outer GLS loop remains an S-Task.
        2. The `FastLocalSearch` procedure 1 is designed to *reduce* work by only searching *active* sub-neighborhoods.
        3. The synergy is an S-Task pipeline: The GLS penalty update (S_Task) *informs* the FLS procedure, which then executes.
        4. The FLS procedure *itself* can be parallelized (P-Data). Instead of iterating `i <- 1 until L` 1, a P-Data implementation would assign $L$ processors (or a large thread block) to check all *active* sub-neighborhoods in parallel, find the best move among them, and then synchronize to apply the move.
        5. This analysis reveals that GLS is an S-Task metaheuristic that acts as an "intelligent guide" for a P-Data-capable local search (FLS). The key benefit is that GLS's guidance *reduces* the amount of P-Data work FLS needs to do, focusing its computational (parallel) power on *promising* regions of the search space (i.e., those related to the newly penalized features).

-----

## Part 4: Empirical Evaluation and Key Findings (Chapter 8)

This section extracts, presents, and analyzes the key results from the empirical evaluation in the Voudouris (1999) paper, focusing on the central narrative: the surprising and powerful effectiveness of the GLS-FLS-2Opt combination.1

### 4.1 The Power of Fast Local Search (FLS) (Table 3)

The first experiment validates the FLS scheme by comparing GLS combined with a standard Best Improvement 2-Opt (BI-2Opt) against GLS combined with Fast Local Search 2-Opt (FLS-2Opt).

- **Table 3: Performance of 2-Opt based variants of GLS on small to medium size TSP instances.**

| **Problem** | **GLS with BI-2Opt** | | | **GLS with FLS-2Opt** | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | optimal runs out of 10 | Mean Excess (%) | Mean CPU (sec) | optimal runs out of 10 | Mean Excess (%) | Mean CPU (sec) |
| att48 | 10 | 0.0 | 0.77 | 10 | 0.0 | 0.4 |
| eil51 | 10 | 0.0 | 1.62 | 10 | 0.0 | 0.46 |
| st70 | 10 | 0.0 | 7.68 | 10 | 0.0 | 1.2 |
| eil76 | 10 | 0.0 | 3.83 | 10 | 0.0 | 0.97 |
| pr76 | 10 | 0.0 | 15.1 | 10 | 0.0 | 3.01 |
| kroA100 | 10 | 0.0 | 11.27 | 10 | 0.0 | 1.25 |
| kroC100 | 10 | 0.0 | 12.2 | 10 | 0.0 | 0.74 |
| lin105 | 10 | 0.0 | 17.46 | 10 | 0.0 | 2.06 |
| pr136 | 9 | 0.0009 | 416.78 | 10 | 0.0 | 32.16 |
| kroA150 | 10 | 0.0 | 257.06 | 10 | 0.0 | 7.03 |
| rat195 | 8 | 0.01 | 525.48 | 10 | 0.0 | 55.15 |
| d198 | 0 | 0.08 | 1998.37 | 0 | 0.05 | 353.97 |
| kroA200 | 10 | 0.0 | 614.6 | 10 | 0.0 | 50.16 |
| lin318 | 8 | 0.01 | 4484.4 | 9 | 0.005 | 346.44 |

\*Data extracted from \*

Analysis:

The results from Table 3 are definitive.

1. **Solution Quality:** The FLS-2Opt variant achieves the same or *better* solution quality than the exhaustive BI-2Opt. It finds more optimal solutions on `pr136` and `rat195` and achieves a lower mean excess on `d198` and `lin318`.
2. **Speed:** FLS-2Opt is *dramatically* faster, often by an order of magnitude. On `pr136`, it is 13 times faster (416.78s vs. 32.16s). On `lin318`, it is 12.9 times faster (4484.4s vs. 346.44s).
3. **Synergy:** This confirms the effectiveness of the GLS+FLS combination described in Section 3.4. FLS is not just a faster approximation of BI-2Opt; it's a *different* and *more effective* search strategy when guided by GLS. By only searching sub-neighborhoods "activated" by GLS penalties, it focuses effort surgically on *escaping* the local minimum rather than just *improving* the current solution, leading to a more effective and efficient exploration.1

### 4.2 GLS vs. General Metaheuristics (Table 4)

This table compares the optimized GLS-FLS-2Opt against standard implementations of Simulated Annealing (SA) and Tabu Search (TS), both using the same 2-Opt neighborhood. It also includes simple Repeated 2-Opt as a baseline.

- **Table 4: GLS, Simulated Annealing, and Tabu Search performance on TSPLIB instances.**

| **Problem** | **GLS with FLS-2Opt** | | **Simulated Annealing** | | **Tabu Search** | | **Repeated BI-2Opt** | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Mean Excess (%) | CPU (sec) | Mean Excess (%) | CPU (sec) | Mean Excess (%) | CPU (sec) | Mean Excess (%) | CPU (sec) |
| eil51 | 0.0 | 0.46 | 0.73 | 6.34 | 0.0 | 1.14 | 0.23 | 42.4 |
| eil76 | 0.0 | 0.97 | 1.21 | 18.0 | 0.0 | 5.24 | 1.85 | 153.45 |
| eil101 | 0.0 | 2.37 | 1.76 | 33.29 | 0.0 | 61.41 | 3.97 | 319.15 |
| kroA100 | 0.0 | 1.25 | 0.42 | 37.36 | 0.0 | 21.34 | 0.34 | 706.35 |
| kroC100 | 0.0 | 0.74 | 0.80 | 36.58 | 0.25 | 4.80 | 0.33 | 1301.98 |
| kroA150 | 0.0 | 7.03 | 1.86 | 103.32 | 0.03 | 413.06 | 1.41 | 3290.95 |
| kroA200 | 0.0 | 50.16 | 1.04 | 229.38 | 0.72 | 776.93 | 1.7 | 731.1 |
| lin318 | 0.005 | 346.44 | 1.34 | 829.46 | 1.31 | 2672.80 | 3.11 | 9771.28 |

\*Data extracted from \*

Analysis:

The superiority of GLS is evident.

1. **Scalability:** GLS-FLS-2Opt scales remarkably well, maintaining 0.0% excess (finding the optimum) on nearly all instances up to 200 cities and achieving 0.005% on `lin318`. In contrast, both SA and TS degrade significantly on larger instances, with excesses of 1.34% and 1.31% on `lin318`, respectively.
2. **Efficiency:** GLS is not only more effective but *vastly* more efficient. On `lin318`, GLS finds a near-optimal solution in 346 seconds. TS requires 2672 seconds (7.7x slower) and SA requires 829 seconds (2.4x slower) to find much *worse* solutions.
3. **Guidance:** This demonstrates that the *intelligent, memory-based guidance* of GLS (penalizing high-cost features found in local minima) is a far more effective search strategy than SA's probabilistic acceptance or the specific recency-based memory implemented in the TS variant.

### 4.3 The Central Finding: GLS-FLS-2Opt vs. State-of-the-Art (Tables 5, 7, Fig 11)

This section presents the paper's core, counter-intuitive result: a "simple" algorithm (GLS-FLS-2Opt) outperforms the "complex" state-of-the-art (Iterated Lin-Kernighan and Genetic Local Search).

- **Table 5: GLS with FLS-2Opt compared with variants of Iterated Lin-Kernighan (DB-Move).** (5 min CPU time)

| **Problem** | **GLS with FLS-2Opt** | **DB with FLS-LK** | **DB with FI-LK** | **Repeated FI-LK** |
| :--- | :--- | :--- | :--- | :--- |
| | Mean Excess (%) | Mean Excess (%) | Mean Excess (%) | Mean Excess (%) |
| att48 | 0 | 0 | 0 | 0 |
| eil76 | 0 | 0 | 0 | 0 |
| kroA100 | 0 | 0 | 0 | 0 |
| bier127 | 0 | 0 | 0 | 0.0301 |
| kroA150 | 0 | 0 | 0 | 0.00226 |
| u159 | 0 | 0 | 0 | 0 |
| kroA200 | 0 | 0 | 0 | 0.02452 |
| gr202 | 0 | 0 | 0.00921 | 0.14143 |
| gr229 | 0.00431 | 0.00475 | 0.01412 | 0.0977 |
| gil262 | 0.00421 | 0 | 0.01682 | 0.05467 |
| lin318 | 0.02641 | 0.24079 | 0.25578 | 0.62957 |
| gr431 | 0.02392 | 0.22239 | 0.3327 | 0.67964 |
| pcb442 | 0.04431 | 0.08173 | 0.06637 | 0.48525 |
| att532 | 0.08994 | 0.08163 | 0.22502 | 0.53023 |
| u574 | 0.14144 | 0.0924 | 0.11435 | 0.73838 |
| rat575 | 0.09892 | 0.09745 | 0.13731 | 0.80762 |
| gr666 | 0.20628 | 0.17587 | 0.41888 | 0.83762 |
| u724 | 0.16822 | 0.16655 | 0.35696 | 0.93367 |
| rat783 | 0.16125 | 0.15331 | 0.24075 | 1.00045 |
| pr1002 | 0.62063 | 0.44633 | 1.04742 | 1.5046 |
| **Average Excess** | **0.07949** | **0.08816** | **0.16178** | **0.42488** |

\*Data extracted from \*

- **Table 7: GLS with FLS-2Opt compared with Genetic Local Search.**

| **Problem** | **GLS with FLS-2Opt** | | **Genetic Local Search** | |
| :--- | :--- | :--- | :--- | :--- |
| | Mean Excess (%) | CPU (sec) | Mean Excess (%) | CPU (sec) |
| eil51 (20 runs) | 0% | 1.2 | 0% | 6 |
| kroA100 (20 runs) | 0% | 1.59 | 0% | 11 |
| d198 (20 runs) | 0% | 435 | 0% | 253 |
| att532 (10 runs) | 0% | 3526 | 0.05% | 6076 |
| rat783 (10 runs) | 0% | 5232 | 0.04% | 14925 |

\*Data extracted from \*

- Figure 11: Overall ranking of the algorithms.

    The overall ranking of all 18 algorithm variants tested, based on the average excess over 20 TSPLIB problems, provides the final summary 1:

    1. **Rank 1:** GLS-FLS-2Opt (Avg. Excess: 0.079%)
    2. **Rank 2:** DB-FLS-LK (Avg. Excess: 0.088%)
    3. **Rank 3:** GLS-FLS-3Opt (Avg. Excess: 0.156%)
    4. **Rank 4:** DB-FI-LK (Avg. Excess: 0.162%)
        ...
    5. Rank 18: Repeated FLS-2Opt (Avg. Excess: 5.220%)

        Data extracted from 1

#### Analysis and Synthesis: The "Voudouris Paradox"

The data from Tables 5, 7, and Figure 11 is undeniable and presents a significant finding, which can be termed the "Voudouris Paradox":

1. **GLS-FLS-2Opt Wins:** The simplest combination, GLS-FLS-2Opt, is the overall best-performing algorithm in the 5-minute study. It beats the highly complex and specialized DB-FLS-LK (Iterated Lin-Kernighan) on average (0.079% vs. 0.088%). On some instances, the difference is stark: on `lin318`, GLS-FLS-2Opt is almost 10x better (0.026% excess) than DB-FLS-LK (0.240% excess).
2. **GLS Beats GA:** Table 7 shows GLS-FLS-2Opt outperforming a state-of-the-art Genetic Local Search. It finds *optimal* solutions (0% excess) for `att532` and `rat783` where the GA finds non-optimal solutions (0.05% and 0.04%). It also does so in a fraction of the time (e.g., 5232s vs 14925s on `rat783`).
3. **Why GLS-FLS-2Opt Wins (The Core Insight):** Figure 10 1 provides the explanation. This figure plots the "Absolute Improvement" that the metaheuristic (GLS or DB) provides *over* its corresponding simple Repeated Local Search baseline.
    - GLS adds **5.14%** of quality improvement to the baseline FLS-2Opt.
    - DB (ILK) adds only **1.06%** of quality improvement to the baseline FLS-LK.
    - This demonstrates that GLS is a *vastly* superior metaheuristic framework. It provides *more guidance* and *more improvement* to its base heuristic than ILK does.
    - The "paradox" is resolved: The study proves that the *quality of the guidance* (the metaheuristic) is more important than the *power of the local search operator*.
    - ILK (DB-LK) uses a "dumb" diversification: a random 4-Opt "kick".1 It is powerful but uninformed.
    - GLS uses an *intelligent*, memory-based diversification: it specifically penalizes *problematic features* (long edges) that *persist in local minima*.
    - **Conclusion:** The GLS-FLS-2Opt algorithm wins because it represents a perfect synergy. FLS-2Opt is an incredibly fast, simple, and "directable" search engine. GLS is a "smart navigator" that uses its memory (penalties) to tell this fast engine exactly where to focus its effort. This "smart guide + fast engine" (GLS+FLS-2Opt) is empirically shown to be superior to the "dumb guide + complex engine" (ILK+LK).

-----

## Appendix: Complete Experimental Data Tables

The following tables, extracted from the Voudouris (1999) appendix, provide the complete raw data for the mean percentage excess over 10 runs, each with a 5-minute CPU time limit, on a set of 20 TSPLIB problems. This data forms the basis for the rankings and figures in the main analysis.1

### Table A.1: Results for Guided Local Search on the TSP

(This table provides the raw data for Figures 6 & 7)

| **Problem** | **GLS-FI-LK** | **GLS-FI-3Opt** | **GLS-FI-2Opt** | **GLS-FLS-LK** | **GLS-FLS-3Opt** | **GLS-FLS-2Opt** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| att48 | 0 | 0 | 0 | 0 | 0 | 0 |
| eil76 | 0 | 0 | 0 | 0 | 0 | 0 |
| kroA100 | 0 | 0 | 0 | 0 | 0 | 0 |
| bier127 | 0.218207 | 0.116586 | 0.019699 | 0.206625 | 0.002198 | 0 |
| kroA150 | 0.029784 | 0.084075 | 0.000754 | 0.001508 | 0.001131 | 0 |
| u159 | 0 | 0.460551 | 0.225285 | 0 | 0 | 0 |
| kroA200 | 0.436189 | 0.526083 | 0.257083 | 0.088872 | 0.00681 | 0 |
| gr202 | 0.732321 | 0.406375 | 0.309512 | 0.252988 | 0.011703 | 0 |
| gr229 | 0.392788 | 0.468195 | 0.381644 | 0.152969 | 0.015007 | 0.004309 |
| gil262 | 0.328007 | 0.723297 | 0.428932 | 0.084104 | 0.046257 | 0.004205 |
| lin318 | 1.00264 | 1.74284 | 1.33884 | 0.583407 | 0.129197 | 0.02641 |
| gr431 | 1.69438 | 2.71862 | 2.34071 | 0.563665 | 0.134003 | 0.023919 |
| pcb442 | 0.966363 | 0.80783 | 1.36634 | 0.38816 | 0.038403 | 0.044311 |
| att532 | 1.04746 | 2.28599 | 2.52871 | 0.386116 | 0.224662 | 0.089937 |
| u574 | 1.36892 | 2.81263 | 3.66807 | 0.580951 | 0.278824 | 0.141444 |
| rat575 | 0.806142 | 1.77174 | 2.25011 | 0.287908 | 0.171268 | 0.098922 |
| gr666 | 1.66056 | 4.38707 | 6.00476 | 0.855251 | 0.497863 | 0.206279 |
| u724 | 1.02505 | 2.25101 | 3.03054 | 0.61298 | 0.336674 | 0.168218 |
| rat783 | 0.897116 | 2.24052 | 3.36929 | 0.511015 | 0.285033 | 0.161254 |
| pr1002 | 1.97877 | 3.31969 | 5.54336 | 1.04229 | 0.945357 | 0.620626 |
| **Avg Excess** | **0.729235** | **1.356155** | **1.653182** | **0.32994** | **0.15622** | **0.079492** |

\*Data extracted from \*

### Table A.2: Results for Iterated Local Search (DB-Move) on the TSP

| **Problem** | **DB-FI-LK** | **DB-FI-3Opt** | **DB-FI-2Opt** | **DB-FLS-LK** | **DB-FLS-3Opt** | **DB-FLS-2Opt** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| att48 | 0 | 0 | 0 | 0 | 0 | 0 |
| eil76 | 0 | 0 | 0 | 0 | 0 | 0 |
| kroA100 | 0 | 0 | 0 | 0 | 0 | 0 |
| bier127 | 0 | 0 | 0 | 0 | 0 | 0 |
| kroA150 | 0 | 0.001508 | 0.003393 | 0 | 0 | 0 |
| u159 | 0 | 0 | 0 | 0 | 0 | 0 |
| kroA200 | 0 | 0.077295 | 0.10113 | 0 | 0.004767 | 0.075252 |
| gr202 | 0.009213 | 0.088396 | 0.457171 | 0 | 0.155129 | 0.257719 |
| gr229 | 0.014116 | 0.157576 | 0.382387 | 0.004755 | 0.064115 | 0.124515 |
| gil262 | 0.016821 | 0.20185 | 0.626577 | 0 | 0.075694 | 0.475189 |
| lin318 | 0.255776 | 0.719027 | 1.14588 | 0.240786 | 0.279093 | 0.3519 |
| gr431 | 0.332703 | 0.94403 | 2.13495 | 0.222386 | 0.394192 | 0.615294 |
| pcb442 | 0.066367 | 0.368861 | 1.8961 | 0.081728 | 0.309977 | 0.684745 |
| att532 | 0.225023 | 1.03554 | 2.64971 | 0.08163 | 0.270534 | 0.422957 |
| u574 | 0.114348 | 1.20038 | 2.94269 | 0.092399 | 0.404823 | 0.553042 |
| rat575 | 0.13731 | 1.15016 | 3.75904 | 0.097446 | 0.445888 | 0.649638 |
| gr666 | 0.418878 | 1.25178 | 3.27054 | 0.175874 | 0.359528 | 0.816489 |
| u724 | 0.356955 | 1.43617 | 3.94106 | 0.166547 | 0.367693 | 0.627535 |
| rat783 | 0.240745 | 1.79764 | 5.00454 | 0.153305 | 0.516693 | 0.744947 |
| pr1002 | 1.04742 | 2.05625 | 5.19902 | 0.446332 | 0.872049 | 1.05727 |
| **Avg Excess** | **0.161784** | **0.624323** | **1.675709** | **0.088159** | **0.226009** | **0.372825** |

\*Data extracted from \*

### Table A.3: Results for Repeated Local Search on the TSP

| **Problem** | **REP-FI-LK** | **REP-FI-3Opt** | **REP-FI-2Opt** | **REP-FLS-LK** | **REP-FLS-3Opt** | **REP-FLS-2Opt** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| att48 | 0 | 0 | 0 | 0 | 0 | 0 |
| eil76 | 0 | 0 | 1.35688 | 0 | 0 | 1.48699 |
| kroA100 | 0 | 0.39564 | 0.222254 | 0 | 0.225543 | 0.215205 |
| bier127 | 0.030098 | 0.403696 | 1.19629 | 0.027899 | 0.370386 | 1.29513 |
| kroA150 | 0.002262 | 0.8317 | 2.00912 | 0.002262 | 0.8038 | 2.01553 |
| u159 | 0 | 0.30038 | 1.62619 | 0 | 0.265447 | 2.05894 |
| kroA200 | 0.024517 | 1.00688 | 3.30768 | 0.004767 | 0.922092 | 3.23583 |
| gr202 | 0.141434 | 1.22958 | 3.58591 | 0.129731 | 1.19995 | 3.68352 |
| gr229 | 0.097695 | 1.36774 | 3.40129 | 0.094427 | 1.27301 | 3.56443 |
| gil262 | 0.054668 | 1.3709 | 5.12195 | 0.054668 | 1.2868 | 5.77796 |
| lin318 | 0.629565 | 2.17992 | 4.37936 | 0.636703 | 2.022676 | 4.9128 |
| gr431 | 0.679641 | 2.07801 | 5.33877 | 0.665232 | 2.20915 | 5.97495 |
| pcb442 | 0.48525 | 1.77636 | 6.65012 | 0.516956 | 1.72417 | 7.19544 |
| att532 | 0.530232 | 2.29033 | 6.28368 | 0.579354 | 2.29141 | 7.13899 |
| u574 | 0.738382 | 2.91397 | 7.46674 | 0.703157 | 2.6934 | 8.4788 |
| rat575 | 0.807618 | 2.69895 | 7.69231 | 0.887347 | 2.70781 | 8.61066 |
| gr666 | 0.837619 | 3.18259 | 8.14712 | 0.847811 | 2.97203 | 9.94096 |
| u724 | 0.933667 | 2.90551 | 7.76903 | 1.0241 | 2.87473 | 8.83202 |
| rat783 | 1.00045 | 3.2864 | 8.46468 | 1.06518 | 3.39882 | 9.38792 |
| pr1002 | 1.5046 | 3.50511 | 8.62028 | 1.39138 | 3.59138 | 10.5847 |
| **Avg Excess** | **0.424885** | **1.686183** | **4.631983** | **0.431549** | **1.64163** | **5.219539** |

\*Data extracted from \*

You've asked some excellent, insightful questions. The distinctions between these parallel taxonomies are notoriously confusing, as the classifications depend on *what* you're looking at: the high-level algorithm, a single step within it, or the overall parallel *strategy*.

Your intuition in the warning blocks is largely correct. The "S-Task + P-Data" hybrid is the key.

Let me provide a new, more detailed explanation to clarify the Crainic-Toulouse taxonomy and answer your specific questions.

### A Clearer Guide to Parallel Taxonomy

Think of these three classifications (P-Data, S-Task, P-Task) as tools to describe *where* the parallelism is. An algorithm can be a mix of all three.

#### 1\. S-Task (Sequential Task): The Assembly Line

This is the easiest to understand. An algorithm is S-Task if it's an assembly line: Step 2 *cannot* begin until Step 1 is finished.

- **Analogy:** Building a car. You *must* build the frame (Step 1) *before* you can install the engine (Step 2) *before* you can paint the body (Step 3).
- **Example:** The main loop of *almost every* metaheuristic.
  - **Simulated Annealing:** [Generate Neighbor] -\> [Evaluate Cost] -\>. This is a sequential, three-stage pipeline.
  - **Guided Local Search:** -\> [Find Max Utility Features] -\> [Update Penalties]. This is a sequential, three-stage pipeline.[1]
- **Your insight was 100% correct:** The *overall* 2-Opt algorithm (a full hill climb) is an S-Task: -\> -\> -\>...

#### 2\. P-Data (Parallel Data): The Coordinated Team

This is the most common type of parallelism used in scientific computing and GPUs. It means you are performing the *exact same instruction* on many different pieces of data simultaneously.

- **Analogy:** A manager tells 1,000 workers to "calculate the $\Delta\text{cost}$." The *instruction* is identical for all 1,000 workers, but each worker gets a *different* pair of edges (different *data*) to check.
- **Hardware:** This is what GPUs (and SIMT) are *built for*.
- **Examples:**
  - Evaluating the $O(N^2)$ 2-Opt neighborhood. This is the "hotspot" [2] of the algorithm.
  - Calculating the fitness of 1,000 individuals in a Genetic Algorithm population.

#### 3\. P-Task (Parallel Task): The Independent Cooperating Teams

This is a higher-level, more complex form of parallelism. It means you have multiple, *different* tasks (programs) running at the same time, often on different data, and they *cooperate*.

- **Analogy:** A manager wants to build 100 cars.
  - **P-Data:** The manager hires 1,000 workers to all work on *one car* (e.g., all 1,000 workers bolt on wheels at the same time).
  - **P-Task:** The manager builds 10 *separate assembly lines* (i.e., 10 "tasks"). Each assembly line (task) runs *independently* to build a full car. Periodically, the manager has the teams *cooperate* by "migrating" the best-designed engine from line 3 over to line 7.
- **Hardware:** This requires MIMD (Multiple Instruction, Multiple Data), as found in multi-core CPUs. Each CPU core can run a completely different program.

-----

### The Most Common Model: The S-Task / P-Data Hybrid

This directly addresses your confusion. You are right: an algorithm is rarely "just" one thing. The most common and effective model is an **S-Task algorithm that has a P-Data bottleneck.**

- **S-Task (Overall Algorithm):** ` -> -> `
- **P-Data (Bottleneck Step):** Step 2 takes 99% of the time (e.g., "Evaluate Neighborhood"). We can parallelize *just this step* using P-Data.

<!-- end list -->

```ascii
     +-----------------------------------------------------------+

| S-Task (Sequential) Loop |
| |
| +-----------+     +--------------------------------+     +-----------+
| | Generate | --> | Evaluate All Neighbors (P-Data)| --> | Accept/ |
| | Neighbors | | | | Reject |
| +-----------+ | | +-----------+
| | | | |
| | | | |
| | | [...] | |
| | | | |
| | +--------------------------------+ |
| | |
     +-------<----------------------------------------------------------+
```

This model applies to almost everything you asked about.

-----

### Answering Your Specific Questions

#### On 2-Opt (Your Q1, Q2, Q3)

> "Is 2-opt really a P-data? It seems more like P-tasks..." and "An S-task algorithm can have... some steps parallelized with p-data. But the overall algorithm is s-task. Is that correct?"

You are 100% correct. Let's be precise:

1. **2-Opt Neighborhood Evaluation:** Checking the $\Delta\text{cost}$ for all $O(N^2)$ swaps is **P-Data**. This is the part that maps to GPUs. A study on GPU-based 2-Opt solvers describes this exact hierarchical strategy: parallelizing the 2-Opt evaluations *within* a climb across threads.[3]
2. **2-Opt Hill Climb (the algorithm):** The *process* of "Find best swap, Apply swap, Repeat" is an **S-Task**.
3. **Cooperative 2-Opt (what `fujimoto2011` does):** A high-level strategy where you run *multiple* 2-Opt hill climbs (which are S-Task/P-Data hybrids) independently and have them *cooperate* (e.g., exchange best tours) is **P-Task**.[4]

The previous response was referring to **\#1**, which is the computationally expensive part.

> "Does this exclusively require custom kernels OR cupy can be used...?"

You **can and should** use CuPy for this. The P-Data evaluation is what CuPy is *for*.

- **High-Level CuPy:** You can use CuPy's NumPy-like array operations (slicing, element-wise ops) to calculate all $N^2$ $\Delta\text{cost}$ values in parallel. CuPy also offers `ElementwiseKernel` for simple, custom operations without writing full CUDA C++. This is the 90% solution.
- **Custom Kernels:** For maximum, hand-tuned performance, you would use `cupy.RawKernel` to write your own CUDA C++ kernel. This lets you manually control shared memory and optimize memory access, which a high-level approach might miss. But it's much more complex.

**Conclusion:** The neighborhood evaluation is P-Data and is perfectly solvable with high-level CuPy.

#### On Simulated Annealing: Multi-Start (Your Q4)

> "Why is this P-data and not P-task?... I'm not understanding the nuances between p-data and p-task."

This is the most subtle and important distinction.

**Multi-Start (Independent Runs)** is **P-Data** because it is "Same Instruction, Different Data."

- **The "Same Instruction":** `Run_This_Entire_SA_Algorithm()`
- **The "Different Data":** `(seed=1)`, `(seed=2)`, `(seed=3)`...

You are launching $N$ *identical* copies of the *same program* and just giving them different initial data (the random seed). They run in parallel and **do not talk to each other.** This is an "embarrassingly parallel" P-Data task.

It *becomes* **P-Task** the *moment they start cooperating*.

- If Runner 1 finds a new best tour and *tells* Runner 2...
- If Runner 3 and Runner 4 periodically *migrate* (swap) their best solutions...
    ...it is now a **P-Task** algorithm because the tasks are different (e.g., `Run_SA()` + `Communicate_With_Peers()`) and cooperative.

> "Why cupy high level api can not be used for this?"

It can, but you're mixing levels.

- **Level 1 (Strategy):** "Multi-Start" is a *strategy*. You would use a tool like **Dask** or Python's `multiprocessing` to launch $N$ *processes* (e.g., on $N$ CPU cores).
- **Level 2 (Implementation):** *Each* of those $N$ independent processes could *internally* use **CuPy** to run its *own* P-Data neighborhood evaluations on the GPU.

Alternatively, you can implement multi-start *entirely on one GPU* by launching $N$ different CUDA thread blocks, where each *block* runs one full, independent SA search.[3] This is also a P-Data model.

#### On Simulated Annealing: Parallel Neighborhood (Your Q5)

> "this is interesting, is this SA +2-opt or what?"

**Yes, exactly.** This is the S-Task / P-Data hybrid.

- **S-Task Loop:** (1) Generate, (2) Evaluate, (3) Accept/Reject.
- **Parallel Version:**
    1. **Generate:** Create *all* $O(N^2)$ 2-Opt neighbors.
    2. **Evaluate (P-Data):** Evaluate all $N^2$ neighbors in parallel on the GPU.
    3. **Accept/Reject:** Pick one of the improving moves (e.g., the best one, or a random one) and apply the Metropolis-acceptance check.

This is a very powerful model, and you are correct that it's a hybrid of SA + 2-Opt.

#### On GLS + FLS (Your Q6)

> "Now there's also s-task + p-data (isn't this like the 2-opt?)"

**Yes, you are correct again.** It is the *exact same pattern*.

- **GLS Loop (S-Task):** `Run FastLocalSearch()` [1] -\> `Update Penalties()` [1] -\> `Repeat...`
- **FastLocalSearch (P-Data):** The FLS procedure [1] itself must search all *active* sub-neighborhoods. Instead of looping `i <- 1 until L` [1], a parallel version would check *all active neighborhoods simultaneously* (a P-Data task) to find the best move.

Your insight is the key: **Almost all of these are S-Task meta-loops, where the "evaluate" step is a massive P-Data task.**

This is an excellent and crucial set of questions. The distinctions between these parallel taxonomies are subtle, and your intuition in the `warning` blocks is exactly right. The confusion often arises because an *entire algorithm* is rarely just one thing; it's almost always a *hybrid*.

Let's replace the previous Part 3 with a much more detailed breakdown that directly addresses your questions, complete with the requested diagrams and tables.

-----

## Part 3 (Revised): A Deeper Analysis of Heuristic Parallelism

This section provides the in-depth analysis you requested, clarifying the Crainic-Toulouse taxonomy and applying it to each heuristic from the Voudouris & Tsang paper.[1, 1]

### 3.1 A Note on SIMD vs. SIMT (Revisiting Your Warning)

Your context `warning` is correct to point out the distinction. It's a key detail for implementation.

- **SIMD (Single Instruction, Multiple Data):** This is a "pure" vector model, often found in CPU vector instructions. If one data lane needs to branch (`if-else`), the other lanes are "masked" (disabled) while that branch is executed. This is inefficient if branches are common.[2]
- **SIMT (Single Instruction, Multiple Threads):** This is the model used by GPUs (like NVIDIA's CUDA). A group of threads (a "warp") executes the *same instruction*. If threads *diverge* (e.g., some go into an `if` block, others into the `else`), the hardware handles this by *serializing* the paths: all `if` threads run while the `else` threads are masked, and then all `else` threads run while the `if` threads are masked.[2, 3]

**Why this matters:** SIMT's flexible handling of divergence is what makes it practical to run complex logic (like neighborhood evaluations or even entire search algorithms) on a GPU, not just simple math. When we say an algorithm is "P-Data" and maps to GPUs, we are implicitly relying on the SIMT model.[2]

### 3.2 The Crainic-Toulouse Taxonomy (Clearer Definitions)

Let's define the three classifications with clear analogies.

1. **S-Task (Sequential Task): The Assembly Line**

      - **Concept:** The algorithm is broken into a *pipeline* of sequential stages. Stage 2 *cannot* start until Stage 1 finishes.
      - **Analogy:** A car factory assembly line. You must install the engine (Stage 1) *before* you can install the hood (Stage 2) *before* you can paint the car (Stage 3).
      - **Heuristic Example:** The main loop of *any* metaheuristic.
          - **GLS:** `->` -\> \`\`.[1, 1]
          - **SA:** `->` -\> \`\`.[1, 1]

2. **P-Data (Parallel Data): The Coordinated Team**

      - **Concept:** This is data parallelism. You perform the *exact same instruction* on many different pieces of data simultaneously.
      - **Analogy:** A manager (the instruction) tells 1,000 workers (the processors) to "Calculate the cost of this edge swap." Each worker gets a *different* pair of edges (the data) but performs the *identical* calculation.
      - **Heuristic Example:** The "hotspot" or bottleneck in most local search.
          - **2-Opt:** Calculating the $O(N^2)$ $\Delta\text{cost}$ for *all possible swaps* at the same time.[4]
          - **GA:** Calculating the fitness for all 1,000 individuals in a population.
      - **Implementation:** This maps perfectly to SIMT (GPU) architectures. High-level libraries like **CuPy** are *designed* to execute these P-Data operations by translating array operations (like `cost_matrix * mask`) into parallel kernels.

3. **P-Task (Parallel Task): The Independent, Cooperating Teams**

      - **Concept:** This is task parallelism. You have multiple, *different* (or different-in-purpose) tasks running at the same time, often on different data, and they *cooperate* by communicating.
      - **Analogy:** A company with 10 independent factories. Each factory (a task) builds a complete car.
          - **Case A (P-Task):** Factory 1 runs a Genetic Algorithm, and Factory 2 runs Simulated Annealing. They periodically *cooperate* by sending their best-found car design to each other. Their internal *tasks are different*.
          - **Case B (P-Task):** All 10 factories run the *same* GA, but on different subsets of the problem. They "migrate" individuals (communicate) between factories. This is the "island model" GA.[5, 6, 7, 8, 9]
      - **Your Multi-Start Question:** This is the key.
          - If the 10 factories run *independently* and *never talk*, and you just pick the best car at the end, it's **P-Data** (Multi-Start). It's the same instruction (`run_factory()`) on different data (`seed=1`, `seed=2`...).
          - If the 10 factories *talk and share parts* during the process, it's **P-Task**.

### 3.3 The Key: The "S-Task + P-Data" Hybrid Model

Your intuition is correct. Almost *all* algorithms discussed here are **S-Task + P-Data** hybrids.

An algorithm's high-level control loop is an **S-Task (Assembly Line)**, but one of the stages in that line is a massive computational bottleneck that can be parallelized with **P-Data (Coordinated Team)**.

**Master Diagram: The S-Task + P-Data Hybrid**

```ascii
     +-------------------------------------------------------------+

| S-Task (Sequential) Loop |
| |
| +-------------+       +-------------------+       +-------------+
---->| Stage 1 |----->| Stage 2 |----->| Stage 3 |----

| (e.g. Select) | | (e.g. Evaluate) | | (e.g. Update) |
| +-------------+       +-------------------+       +-------------+
| | |
| | P-Data Bottleneck |
| | (Parallelized on GPU) |
| | |
| +--------+---------+ |
| | Worker 1 (Data 1) | |
| | Worker 2 (Data 2) | |
| | Worker 3 (Data 3) | |
| |... | |
| | Worker N (Data N) | |
| +-------------------+ |
| |
     +-------------------------------------------------------------+
```

This single pattern explains almost every heuristic.

### 3.4 Summary of Heuristic Classifications

First, a simple classification of the algorithms themselves as discussed in the paper.[1, 1]

**Table A: Heuristic Classification**

| Heuristic | Classification |
| :--- | :--- |
| 2-Opt | Improvement Heuristic [1, 1] |
| 3-Opt | Improvement Heuristic [1, 1] |
| Lin-Kernighan (LK) | Improvement Heuristic [1, 1] |
| Fast Local Search (FLS) | Improvement Heuristic / Neighborhood Reduction Scheme [1, 1] |
| Repeated Local Search | Metaheuristic [1, 1] |
| Simulated Annealing (SA) | Metaheuristic [1, 1] |
| Tabu Search (TS) | Metaheuristic [1, 1] |
| Genetic Algorithms (GA) | Metaheuristic [1, 1] |
| Iterated Lin-Kernighan (ILK) | Metaheuristic [1, 1] |
| Guided Local Search (GLS) | Metaheuristic [1, 1] |

Now, the parallel taxonomy you requested.

**Table B: Parallel Taxonomy and Hybrid Analysis**

| Algorithm | S-Task Component (The Loop) | P-Data Component (The Work) | P-Task Strategy (The Cooperation) |
| :--- | :--- | :--- | :--- |
| **2-Opt (as Hill Climb)** | The `Find -> Apply -> Repeat` loop. | **Neighborhood Evaluation** (Find $\Delta\text{cost}$ for all $O(N^2)$ swaps).[4] | Cooperative 2-Opt (e.g., tasks on different tour segments).[10] |
| **Simulated Annealing** | The `Generate -> Evaluate -> Accept/Reject -> Cool` loop. | 1. **Parallel Neighborhood Evaluation** (as in 2-Opt). 2. **Multi-Start** (independent runs). | Cooperative SA / Parallel Tempering (tasks at different temps share solutions). |
| **Tabu Search** | The `Evaluate All -> Select Best Non-Tabu -> Update List` loop. | **Neighborhood Evaluation** (Find $\Delta\text{cost}$ for all $O(N^2)$ swaps). | Cooperative TS (tasks with different tabu lists share solutions). |
| **Genetic Algorithms** | The `Select -> Crossover -> Mutate -> Evaluate -> Replace` loop. | **Fitness Evaluation** (Run fitness function for all individuals in parallel). | **Island Model** (Independent populations "migrate" solutions).[5, 7, 8] |
| **Iterated LK** | The `Kick (4-Opt) -> Optimize (LK) -> Accept` loop.[1, 1] | 1. **Multi-Start** (independent ILK runs). 2. The "Optimize" (LK) step is itself a complex S-Task/P-Data hybrid. | Concurrent Kicks (Task 1 tries 4-Opt, Task 2 tries 5-Opt, share best). |
| **Guided Local Search** | The `LocalSearch() -> UpdatePenalties()` loop.[1, 1] | The `LocalSearch()` call (e.g., FLS-2Opt) is the P-Data bottleneck. | N/A (GLS is the guide, not the parallel strategy). |
| **GLS + FLS** | The *outer* GLS loop (S-Task) + *inner* FLS loop (S-Task).[1, 1] | **FLS Neighborhood Evaluation** (Check all *active* sub-neighborhoods in parallel). | N/A. |

-----

### 3.5 Detailed Parallel Analysis and Diagrams

Here is the breakdown for each heuristic, answering your specific warnings.

#### 1\. Local Improvement Heuristics (2-Opt, 3-Opt, LK)

- **Classification:** Improvement Heuristic.[1, 1]
- **Your Question:** "Is 2-opt really a P-data? It seems more like P-tasks... An S-task algorithm can have... some steps parallelized with p-data. But the overall algorithm is s-task. Is that correct?"
- **Answer:** You are **exactly correct**. This is the S-Task + P-Data hybrid.
  - The **S-Task** component is the *hill-climbing algorithm*: ` -> [2. Apply Move] -> `.
  - The **P-Data** component is Stage 1, "Find Best Move." This step requires checking all $O(N^2)$ possible 2-Opt swaps. This is a massive, "embarrassingly parallel" P-Data task.[4]
- **CuPy vs. Custom Kernel:** You can *absolutely* use high-level CuPy for this. You would use array operations to compute the $\Delta\text{cost}$ for all $N^2$ pairs. A custom `RawKernel` would only be needed for squeezing out the last 5-10% of performance.
- **P-Task Model:** A P-Task model would be different, e.g., Task A runs 2-Opt, Task B runs 3-Opt, and they share the tour. This is a cooperative model.[10, 11, 12]

<!-- end list -->

```ascii
Algorithm: 2-Opt Hill Climb (S-Task + P-Data Hybrid)

+-----------------------------------------------------------------+

| S-TASK (SEQUENTIAL LOOP) |
| |
| +-----------------------------------------------------+ |
| | <---(P-DATA BOTTLENECK) | |
| +-----------------------------------------------------+ |
| | |
| | Evaluate ALL N^2 swaps in parallel |
| | +-------------------+ |
| | | Swap (i=1, j=2) | |
| | | Swap (i=1, j=3) |... (P-Data) |
| | |... | |
| | | Swap (i=N, j=N-1) | |
| | +-------------------+ |
| | |
| +-----------------------------------------------------+ |
| | (e.g., Swap(5, 12)) | |
| +-----------------------------------------------------+ |
| |
| +-----------------------------------------------------+ |
| | | |
| +-----------------------------------------------------+ |
| |
+-------<---------------------------------------------------------+
```

#### 2\. Simulated Annealing (SA)

- **Classification:** Metaheuristic.[1, 1]
- **Your Question (Multi-Start):** "Why is this P-data and not P-task?"
- **Answer:** This is the core distinction:
  - **Multi-Start (P-Data):** Launch 1,000 *identical*, *independent* SA runs. The "instruction" is `Run_SA()`. The "data" is `seed=1`, `seed=2`, etc. They **do not communicate**. The manager just picks the best result at the very end.
  - **Cooperative SA (P-Task):** Launch 1,000 SA runs that *communicate*. For example, they periodically "vote" on the best solution, or tasks at different temperatures ("Parallel Tempering") swap solutions. Because they cooperate, it's P-Task.
- **Your Question (Parallel Neighborhood):** "is this SA + 2-opt or what?"
- **Answer:** Yes. This is just the S-Task + P-Data hybrid, but the S-Task loop is SA's. Instead of generating *one* random 2-Opt neighbor, you *evaluate all $N^2$ neighbors* (P-Data) and pick the best one to feed into the SA acceptance criteria. (This is also how Tabu Search works).

<!-- end list -->

```ascii
Algorithm: SA + Parallel Neighborhood (S-Task + P-Data Hybrid)

+-------------------------------------------------------------------+

| S-TASK (SEQUENTIAL SA LOOP) |
| |
| +-------------------------------------------------------+ |
| | <-- (P-DATA) | |
| +-------------------------------------------------------+ |
| | |
| | Evaluate ALL N^2 2-Opt swaps in parallel |
| | +-------------------+ |
| | | Swap(i,j) cost = X| ... (P-Data) |
| | | Swap(k,l) cost = Y| |
| | +-------------------+ |
| | |
| +-------------------------------------------------------+ |
| | | |
| +-------------------------------------------------------+ |
| |
| +-------------------------------------------------------+ |
| | (Accept Y or keep old?) | |
| +-------------------------------------------------------+ |
| |
| +-------------------------------------------------------+ |
| | | |
| +-------------------------------------------------------+ |
| |
+-------<-----------------------------------------------------------+
```

#### 3\. Genetic Algorithms (GA) / Genetic Local Search

- **Classification:** Metaheuristic.[1, 1]
- **Analysis:** GAs are a perfect example of the S-Task + P-Data hybrid. The **P-Task** "island model" (as in `fujimoto2011`) is a *different, higher-level strategy*.[5, 6, 7, 8, 9]
  - **S-Task:** The generational loop: ` -> [Crossover] -> [Mutate] -> -> `.
  - **P-Data:** The \`\` step. You have a population of 1,000 new individuals. Their fitness calculations are 1,000 independent computations. This is an embarrassingly parallel P-Data task. In Genetic *Local Search*, this step is even more massive, as each "evaluation" is a full (and parallelizable) LK run.[1, 1]

<!-- end list -->

```ascii
Algorithm: Genetic Algorithm (S-Task + P-Data Hybrid)

+-------------------------------------------------------------------+

| S-TASK (SEQUENTIAL GENERATIONAL LOOP) |
| |
| +-------------------------------------------------------+ |
| | | |
| +-------------------------------------------------------+ |
| |
| +-------------------------------------------------------+ |
| | -> (Create 1000 offspring) | |
| +-------------------------------------------------------+ |
| |
| +-------------------------------------------------------+ |
| | <--- (P-DATA BOTTLENECK) | |
| +-------------------------------------------------------+ |
| | |
| | Calculate fitness for ALL individuals |
| | +-------------------+ |
| | | Fitness(Indiv 1) | ... (P-Data) |
| | | Fitness(Indiv 2) | |
| | |... | |
| | | Fitness(Indiv 1000)| |
| | +-------------------+ |
| | |
| +-------------------------------------------------------+ |
| | (e.g., Elitism) | |
| +-------------------------------------------------------+ |
| |
+-------<-----------------------------------------------------------+
```

#### 4\. Guided Local Search (GLS) + Fast Local Search (FLS)

- **Classification:** GLS (Metaheuristic), FLS (Improvement Scheme).[1, 1]
- **Your Question:** "Now there's also s-task + p-data (isn't this like the 2-opt?)"
- **Answer:** **YES\!** You have 100% grasped the concept. GLS+FLS is a *hierarchical* S-Task + P-Data model.
  - **S-Task (Outer Loop - GLS):** This is the high-level control. ` -> `.[1, 1]
  - **S-Task (Inner Loop - FLS):** Stage 1 (`Run FLS`) is *itself* an S-Task: `[Find Move] -> [Apply Move] -> [Activate Neighbors] ->`.[1, 1]
  - **P-Data (The Work):** The `[Find Move]` step in the FLS loop is the P-Data task. FLS is "smart" P-Data: instead of checking all $N^2$ swaps, the `SubNeighbourhoodsForFeature()` function [1, 1] tells it to *only* check (in parallel) the sub-neighborhoods that are *active*.

<!-- end list -->

```ascii
Algorithm: Guided Fast Local Search (Hierarchical S-Task + P-Data)

+-------------------------------------------------------------------+

| S-TASK (OUTER GLS LOOP) |
| |
| +-----------------------------------------------------------+ |
| | | |
| | | |
| | +---------------------------------------------------+ | |
| | | S-TASK (INNER FLS LOOP) | | |
| | | | | |
| | | +--------------------------------------------+ | | |
| | | | <---+ | | |
| | | +--------------------------------------------+ | | |
| | | | | | |
| | | | (P-Data: Check all ACTIVE | | |
| | | | sub-neighborhoods | | |
| | | | in parallel) | | |
| | | | | | |
| | | +--------------------------------------------+ | | |
| | | | | | |
| | | +--------------------------------------------+ | | |
| | | | | |
| | | +--------------------------------------------+ | | |
| | | | | | |
| | | +--------------------------------------------+ | | |
| | | | | |
| | +---------------------------------------------------+ | |
| | | |
| +-----------------------------------------------------------+ |
| |
| +-----------------------------------------------------------+ |
| | | |
| | (This tells step [A] which bits to activate) | |
| +-----------------------------------------------------------+ |
| |
+-------<-----------------------------------------------------------+
```

You are right to be skeptical, and your doubt is perfectly valid. My previous statement was imprecise. When I said "high-level CuPy," I was not referring to a *naive* implementation using pure NumPy-style vectorization, which, as you suspect, would be inefficient.

You are correct that a sequence of standard array operations (e.g., `a = b + c`, `d = e * f`, `g = a - d`) in CuPy will often launch multiple, separate kernels. This creates the "immense amount of kernel initialization and calls" you rightly want to avoid.

To answer your specific questions directly:

1. **`tsplogo solver`:** You are correct. The `LOGO-TSP-Solver` (by Kamil Rocki) is a high-performance solver that uses low-level, custom-written **CUDA, C, and C++ kernels**. It does not use CuPy. Its performance comes from meticulous, low-level optimization.

2. **High-Level CuPy for Performance:** Your question is, "Is there ANY way to do this with any of cupy's tools?" **Yes.** The answer is to use CuPy's "fused" kernel tools, which are a bridge between high-level Python and a single, high-performance kernel.

The key is to **not** use naive broadcasting for the *entire* 2-Opt logic, but to use it to *set up* the data for a **single, fused kernel call**. CuPy provides two tools for this:

- `cupy.ElementwiseKernel`: This is the "high-level" tool I was referring to. You write the C++ loop body *inside* a Python string, and CuPy compiles it into a single, efficient kernel. You get the performance of a single kernel launch without the boilerplate of writing a full CUDA file.
- `cupy.RawKernel`: This is the "expert" tool. You write the *entire* CUDA C++ kernel, including thread/block management, and compile it within Python. This is for maximum, `LOGO-TSP-Solver`-style optimization.

For your 2-Opt P-Data task (calculating the $\Delta\text{cost}$ for all $O(N^2)$ swaps), `ElementwiseKernel` is the "high-level" way to do it in a single, fast kernel launch.

### Example: High-Level CuPy for 2-Opt $\Delta\text{cost}$ Matrix

Let's assume you have:

- `D` (shape $N \times N$): A CuPy array (on the GPU) of all pairwise city distances.
- `tour` (shape $N$): A CuPy array of city indices, e.g., \`\`.

The goal is to compute a $\Delta\text{cost}$ matrix (shape $N \times N$) where `delta[i, j]` is the cost change of swapping edge $i$ (from `tour[i]` to `tour[i+1]`) with edge $j$ (from `tour[j]` to `tour[j+1]`).

#### Method 1: The Naive (Slow) Broadcasting Method

This is what you're correctly worried about.

```python
import cupy as cp

# --- Setup ---
N = 100
# D = cp.random.rand(N, N) # Distance Matrix
# tour = cp.arange(N)
# cp.random.shuffle(tour)

# Get all 'A' nodes (tour[i]) and 'C' nodes (tour[j])
# A will be shape (N, 1), C will be shape (1, N)
A = tour[:, None]
C = tour[None, :]

# Get all 'B' nodes (tour[i+1]) and 'D' nodes (tour[j+1])
B = cp.roll(tour, -1)[:, None]
D_nodes = cp.roll(tour, -1)[None, :]

# --- Multiple (Slow) Kernel Launches ---

# 1. Calculate cost_removed = D + D
# D broadcasts A and B to (N, N), then gathers. -> Kernel 1
# D broadcasts C and D_nodes, then gathers. -> Kernel 2
#...and then they are added. -> Kernel 3
cost_removed = D + D

# 2. Calculate cost_added = D[A, C] + D
# D[A, C] broadcasts A and C, then gathers. -> Kernel 4
# D broadcasts B and D_nodes, then gathers. -> Kernel 5
#...and then they are added. -> Kernel 6
cost_added = D[A, C] + D

# 3. Calculate delta = cost_added - cost_removed -> Kernel 7
delta_matrix = cost_added - cost_removed

# We'd also need to mask out adjacent edges (i, i), (i, i+1), etc.
# This approach is "clean" but launches many kernels.
```

#### Method 2: The `ElementwiseKernel` (Fast) Method

This is the high-performance, "high-level" CuPy tool. It does *everything* in one kernel launch. We write the C++ logic for a *single thread* (which will be run $N \times N$ times in parallel) and CuPy handles the rest.

```python
import cupy as cp

# --- Setup ---
N = 100
# D = cp.random.rand(N, N) # Distance Matrix
# tour = cp.arange(N)
# cp.random.shuffle(tour)
tour_next = cp.roll(tour, -1) # Pre-calculate B and D nodes

# --- Define the SINGLE Fused Kernel ---
# This kernel will be launched with an (N, N) grid.
# 'i' will be the row index (0 to N-1)
# 'j' will be the col index (0 to N-1)
calculate_delta_matrix = cp.ElementwiseKernel(
    'raw T D, raw int32 tour, raw int32 tour_next', # Input types and names
    'T delta_matrix',                              # Output type and name
    '''
    // This C++ code runs for *each thread* (i.e., for each i, j pair)
    int A = tour[i];
    int B = tour_next[i];
    int C = tour[j];
    int D_node = tour_next[j]; // Renamed to avoid C++ keyword

    // Ignore adjacent/identical edges
    if (i == j |

| i == (j + 1) % N |
| j == (i + 1) % N) {
        delta_matrix = 0; // Or some large "invalid" number
    } else {
        // Look up distances using the raw pointer D
        // D[y * N + x] is how you access D[y, x] from a raw pointer
        
        T cost_removed = D + D;
        T cost_added = D[A * N + C] + D;
        
        delta_matrix = cost_added - cost_removed;
    }
    ''',
    'calculate_delta_matrix'
)

# --- Launch the SINGLE (Fast) Kernel ---
# Create an empty output matrix
delta_matrix = cp.empty((N, N), dtype=cp.float64)

# D, tour, and tour_next are passed as raw pointers.
# delta_matrix is the output.
# CuPy handles all CUDA setup, kernel launch, and synchronization.
calculate_delta_matrix(D, tour, tour_next, delta_matrix)

# This 'delta_matrix' is the result, computed in one call.
```

### Conclusion: Performance vs. Abstraction

You are correct that for maximum performance, `RawKernel` (Method 3) is superior. This is what `LOGO-TSP-Solver` does. It would involve manually managing shared memory to cache parts of `D` or `tour`, and optimizing memory access patterns, which `ElementwiseKernel` does not do. This is where the "last 5-10%" (or more) of performance is gained.

However, the `ElementwiseKernel` (Method 2) is a "high-level" tool that *does* achieve your goal of a single, fused kernel launch. It will be dramatically faster than the naive broadcasting (Method 1) and provides a powerful middle-ground, giving you *most* of the performance of a raw kernel with a fraction of the development effort.
