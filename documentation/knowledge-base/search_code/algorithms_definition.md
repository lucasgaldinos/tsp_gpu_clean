# Defining TCC steps

## Summary

- [Defining TCC steps](#defining-tcc-steps)
  - [Summary](#summary)
    - [1. Heuristics vs meta-heuristics](#1-heuristics-vs-meta-heuristics)
      - [1.1 Heuristics](#11-heuristics)
      - [1.2 Meta-heuristics](#12-meta-heuristics)
    - [2 What is a combinatorial optimization problem](#2-what-is-a-combinatorial-optimization-problem)
    - [3 Guided Local Search](#3-guided-local-search)
      - [3.1 Solution features](#31-solution-features)
      - [3.2 Augmented cost function (penalty terms)](#32-augmented-cost-function-penalty-terms)
  - [Make it later](#make-it-later)
    - [6](#6)
      - [](#)
      - [6.2 Local Search procedures for the TSP](#62-local-search-procedures-for-the-tsp)

### 1. Heuristics vs meta-heuristics

#### 1.1 Heuristics

- A technique designed for solving a problem more quickly when classic methods are too slow, using a trade-off between speed and quality of the solution.
- Not guaranteed to find the optimal solution.
- Can be used as a standalone algorithm or as a component of a meta-heuristic algorithm.
- It's problem specific.
  - Often rule-based approaches that use domain knowledge to guide the search.

**Examples:**

- Lin-Kernighan heuristic for the Traveling Salesman Problem (TSP)
- 3-opt heuristic for the TSP

#### 1.2 Meta-heuristics

- Higher-level procedures or strategies designed to guide underlying heuristics to explore the solution space more effectively.
- A heuristic that uses other heuristics to solve a problem.
- Generally problem independent and can be used for different optimization problems.

**Examples:**

- Genetic algorithms
- Simulated annealing
- Tabu search
- Ant colony optimization
- Particle swarm optimization
- Evolution strategies
- Scatter search
- Iterated local search
- Variable neighborhood search

### 2 What is a combinatorial optimization problem

Defined by a pair $(S,g)$ where $S$ is the set of all feasible solutions and $g:S\to \mathbb{R}$ is the objective function that maps each element $s \in S$ to a real number. The goal is to find a solution $s^* \in S$ that minimizes the objective function $g$.

$$
\begin{equation}
\min g(s), s \in S
\label{eq:min_g_s}
\end{equation}
$$

In case where constraints difficult to satisfy are present, penalty terms may incur in $g(s)$ to satisfy them. A neighborhood $N$ can be defined as a mapping from $S$ to its powerset:

$$
\begin{equation}
N: S \to 2^S
\label{eq:N_S}
\end{equation}
$$

$N(s)$ is called the neighborhood of $s$ and contains all the solutions that can be reached from $s$ by a single move.

A solution $x$ is called a local minimum if:

$$
\begin{equation}
g(x) \leq g(y), \quad y \in N(x)
\label{eq:local_minimum}
\end{equation}
$$

Local search is the procedure of minimizing the cost function $g$ in a number of successive steps in each of which the current solution $s$ is being replaced by a solution $y$ such that:

$$
\begin{equation}
g(y) \leq g(x), \quad y \in N(x)
\label{eq:local_search}
\end{equation}
$$

A basic local search algorithm begins with an arbitrary solution and ends up in a local minimum where no further improvement is possible. In between the stages there are different ways to conduct local search. For example:
>
> - Greedy algorithms (best improvement local search): replaces the current solution that improves most in cost after searching the whole neighborhood.
> - First improvement local search: replaces the current solution with the first solution found in the neighborhood that improves in cost.

**The computational complexity of a local search algorithm depends on the size of the neighborhood.**
The main problem with local search are the **Local Minima**. If a local search algorithm gets caught in local minimum, there is no obvious way to proceed any further towards a better solution. *Metaheuristics* are used to overcome this problem. They are designed to escape local minima and find better solutions. Repeated Local Search is one of the first methods in this class. The best local minimum found throughout multiple runs is returned as an **approximation** of the global minimum. Modern metaheuristics tend to be more sophisticated than Repeated Local Search. The way local search is used may vary and is not limited to applying it to a single solution, but to a population of solutions as in Hybrid Genetic Algorithms.

### 3 Guided Local Search

Has its root in a Neural Network architecture called **GENET** and it's applicable to a class known as Constraint Satisfaction Problems. GLS generalizes some elements present in the GENET architecture and applies them to the general class of combinatorial optimization. GLS augments the cost function of the problem to include a set of penalty terms and passes this — instead of the original one — for minimization by the local search procedure.

#### 3.1 Solution features

A solution feature can be any solution property that satisfies the simple constraint that is a non-trivial one. Some solutions have the property while others do not.

Constraints on features are introduced or strengthened on the basis of information about the problem and also the course of local search. Information pertaining to the problem is the cost of features. Feature costs may be constant or variable. A feature $f_i$ is represented by an indicator function:
$$
\begin{equation}
I_i(s) =
\begin{cases}
1, & \text{if } s \text{ has property } f_i \\
0, & \text{otherwise}
\end{cases}
\label{eq:indicator_function}
\end{equation}
$$

Solution features in notion are very similar to solution attributes in Tabu Search, but solution features are associated with a binary state given by their indicator function. The indicators replace the objective function of the problem during the search process and is dynamically manipulated by GLS to guide the local optimization algorithm used.

#### 3.2 Augmented cost function (penalty terms)

Constraints on features are made possible by augmenting the cost function $g$ of the problem to include a set of penalty terms. The new cost function formed is called augmented cost function and it's defined as follows:

$$
\begin{equation}
h(s) = g(s) +\lambda\sum_{i=1}^{M}p_i\cdot I_i(s)
\label{eq:augmented_cost_function}
\end{equation}
$$

Where $p_i$ is penalty parameter corresponding to feature $f_i$, $M$ is the number of features defined over solution and $\lambda$ a parameter for controlling the strength of constraints with respect to the actual solution cost.

---

## Make it later

---

### 6

---

####

---

#### 6.2 Local Search procedures for the TSP

"Fast local search", "best improvement search" and "first improvement search" are the local search procedures that can be applied to the TSP.  
First improvement search immediately performs improving moves while best improvement (greedy) local search selects the best improving move after testing all possible moves in a neighborhood.

1. Fast local search can easily be converted to first improvement local search by searching all sub-neighborhood irrespective of their state (active or inactive). To stop the search when a full rotation of the static order is completed without making a move is the *termination criterion*.
2. To perform best improvement local search by selecting the best move after testing all possible neighborhoods. It's very time-consuming for algorithms such as LK and 3-opt and should be performed under 2-opt algorithms.

**Seven variants of the problem were implemented**, combining different search schemes at neighborhood level with any of the 2-opt, 3-opt or LK algorithms as follows:

<center>

| Name | Local Search Type | Neighborhood Type |
|---|---|---|
| Bi-2-opt | Best improvement | 2-opt  |
| FI-2-opt | First Improvement | 2-opt  |
| FLS-2opt    | FLS | 2-opt |
| FI-3opt | First improvement | 3-opt |
| FLS-3opt| FLS | 3-opt |
| FI-LK | First Improvement | LK |
| FLS-LK | FLS | LK  |

</center>
