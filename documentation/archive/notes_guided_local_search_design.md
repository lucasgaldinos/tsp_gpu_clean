---
title: "Guided Local Search and Combinatorial Optimization: Algorithm Design and Theory"
description: >
  Foundational notes on combinatorial optimization, neighborhood mapping, local search
  procedures, Guided Local Search (GLS) penalty formulation, and Python reference implementations.
created: 2026-11-05
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Consolidated knowledge-base/search_code/algorithm_design_and_pseudocode.md
          and algorithms_definition.md into a unified reference note preserving combinatorial
          optimization definitions, GLS penalty equations, and Python implementations.
      - description: >
          Fixed guided_local_search_typed termination bug by adding max_iterations stopping
          criterion and tracking best solution with respect to unpenalized objective g.
related_files:
  - "[two_opt.py](../../src/algorithms/strategies/local_search/two_opt.py)"
tags:
  - notes/gls
  - notes/local_search
  - notes/archive
  - notes/tsp
  - algorithm/local_search
  - algorithm/optimization
  - analysis/local_search
  - guide/algorithms
  - notes/python
  - notes/pseudocode
  - notes/combinatorial_optimization
---

# Guided Local Search and Combinatorial Optimization: Algorithm Design and Theory

## 1. Combinatorial Optimization Foundations

A combinatorial optimization problem is defined by a pair $(S, g)$, where $S$ is the set of all feasible solutions and $g: S \to \mathbb{R}$ is the objective function that maps each feasible solution $s \in S$ to a real-valued scalar cost. The goal is to find a global optimum solution $s^* \in S$ that minimizes $g$:

$$\min_{s \in S} g(s)$$

In problems where constraints are difficult to satisfy strictly, penalty terms may be introduced into $g(s)$ to penalize infeasibility.

### 1.1 Neighborhood Mapping and Local Minima

A neighborhood function $N$ is a mapping from the solution space $S$ to its power set $2^S$:

$$N: S \to 2^S$$

For any solution $s \in S$, $N(s)$ defines the neighborhood of $s$, containing all solutions reachable from $s$ by applying a single predefined move operator (e.g., a 2-opt edge exchange or a vertex relocation).

A solution $x \in S$ is defined as a **local minimum** with respect to neighborhood $N$ if:

$$g(x) \le g(y), \quad \forall y \in N(x)$$

### 1.2 Local Search Mechanics

Local search minimizes the objective function $g$ through a sequence of successive transitions. In each step, the current solution $x$ is replaced by an improving solution $y \in N(x)$:

$$g(y) < g(x), \quad y \in N(x)$$

A standard local search procedure terminates when a local minimum is reached, where no further improving move exists in $N(x)$. Two primary move selection schemes are:
- **Best Improvement (Greedy Local Search)**: Evaluates the complete neighborhood $N(x)$ and selects the move yielding the maximal cost reduction.
- **First Improvement**: Scans the neighborhood $N(x)$ in a deterministic or random order and immediately executes the first move satisfying $g(y) < g(x)$.

The computational complexity of each step depends directly on the neighborhood cardinality $|N(x)|$.

---

## 2. Heuristics vs. Metaheuristics

### 2.1 Heuristics
- Problem-specific rules designed to produce high-quality solutions quickly when exact methods are computationally prohibitive.
- Offer no theoretical guarantee of global optimality.
- Often serve as building blocks within higher-level search strategies.
- **Examples**: Nearest Neighbor construction, Lin-Kernighan (LK) variable-depth search, and 2-opt / 3-opt edge exchanges for the TSP.

### 2.2 Metaheuristics
- Problem-independent, master-level algorithmic frameworks designed to guide subordinate heuristics to explore solution spaces effectively and escape local minima.
- **Examples**: Guided Local Search (GLS), Genetic Algorithms (GA), Simulated Annealing (SA), Tabu Search (TS), and Ant Colony Optimization (ACO).

---

## 3. Guided Local Search (GLS) Formulation

Guided Local Search originated from neural network architectures (specifically the GENET model for Constraint Satisfaction Problems) and generalizes penalty-driven exploration to arbitrary combinatorial optimization problems.

Instead of operating on the original cost function $g(s)$, GLS augments $g(s)$ with dynamic penalty terms whenever the underlying local search settles into a local minimum.

### 3.1 Solution Features and Indicator Functions

A solution feature $f_i$ ($i \in \{1, \dots, M\}$) represents a distinct structural attribute of a solution (for TSP, a feature typically indicates whether a specific directed edge $(u, v)$ belongs to tour $s$). Each feature is defined by a binary indicator function:

$$
I_i(s) =
\begin{cases}
1, & \text{if solution } s \text{ exhibits feature } f_i \\
0, & \text{otherwise}
\end{cases}
$$

Each feature $f_i$ is associated with a base cost $c_i$ (e.g., the Euclidean distance of edge $(u, v)$) and a non-negative penalty counter $p_i$, initialized to zero.

### 3.2 Augmented Cost Function

When local search reaches a local minimum, GLS modifies the objective landscape by augmenting $g(s)$ with penalty terms:

$$h(s) = g(s) + \lambda \sum_{i=1}^M p_i \cdot I_i(s)$$

where:
- $M$ is the number of defined features.
- $p_i$ is the accumulated penalty counter for feature $f_i$.
- $\lambda$ is a regularization parameter controlling penalty scaling relative to the original objective $g(s)$.

### 3.3 Penalty Update and Utility Maximization

Upon trapping at local minimum $s$, GLS computes the utility of penalizing each active feature $f_i$ present in $s$:

$$\operatorname{util}(s, f_i) = I_i(s) \cdot \frac{c_i}{1 + p_i}$$

Features exhibiting maximal utility are penalized:

$$p_i \leftarrow p_i + 1, \quad \forall i \in \arg\max_j \operatorname{util}(s, f_j)$$

This penalty penalizes unfavorable, high-cost features, raising the augmented cost of the current basin of attraction and forcing local search to escape the local minimum.

---

## 4. Local Search Procedures and Taxonomy for the TSP

For the Traveling Salesperson Problem (TSP), local search procedures combine neighborhood operators with search strategies:
1. **Best Improvement (Bi)**: Scans all moves in $N(s)$ and applies the best delta. Highly effective but computationally intensive for large neighborhoods.
2. **First Improvement (FI)**: Immediately executes the first improving move found.
3. **Fast Local Search (FLS)**: Sub-neighborhood activation scheme that restricts evaluation to active vertices, deactivating sub-neighborhoods when no improvement is found and reactivating them when an adjacent edge changes.

### Taxonomy of Local Search Schemes

| Variant | Local Search Type | Neighborhood Type | Evaluation Characteristics |
| :--- | :--- | :--- | :--- |
| **Bi-2-opt** | Best Improvement | 2-opt | Full $O(n^2)$ scan; best move selected |
| **FI-2-opt** | First Improvement | 2-opt | Sequential scan; immediate move execution |
| **FLS-2opt** | Fast Local Search | 2-opt | Sub-neighborhood flags; high execution speed |
| **FI-3opt** | First Improvement | 3-opt | $O(n^3)$ scan pruned by candidate lists |
| **FLS-3opt** | Fast Local Search | 3-opt | Restricted 3-opt evaluated on active sub-neighborhoods |
| **FI-LK** | First Improvement | Lin-Kernighan | Variable depth with gain criterion |
| **FLS-LK** | Fast Local Search | Lin-Kernighan | LK evaluation on active nodes |

---

## 5. Algorithmic Pseudocode

```text
procedure GuidedLocalSearch(S, g, lambda, [I_1, ..., I_M], [c_1, ..., c_M], M)
begin
  k = 0;
  s_0 = InitialSolution(S);
  
  for i = 1 to M do
    p_i = 0;
    
  s_k = s_0;
  
  while not StoppingCriterion(k, s_k) do
  begin
    h = g + lambda * sum(p_i * I_i);
    s_{k+1} = LocalSearch(s_k, h);
    
    for i = 1 to M do
      utility_i = I_i(s_{k+1}) * c_i / (1 + p_i);
      
    max_util = max_{i} utility_i;
    for each i where utility_i == max_util do
      p_i = p_i + 1;
      
    s_k = s_{k+1};
    k = k + 1;
  end
  
  s* = BestSolutionFoundWithRespectTo(g);
  return s*;
end
```

---

## 6. Python Reference Implementations

### 6.1 Basic Implementation

```python
import random

def guided_local_search(S, g, lambda_, I, c, M, stopping_criterion):
    k = 0
    s_0 = random.choice(S)
    p = [0] * M
    s_k = s_0

    while not stopping_criterion(k, s_k):
        # Compute augmented cost h = g + lambda * sum(p_i * I_i)
        h = g(s_k) + lambda_ * sum(p[i] * I[i](s_k) for i in range(M))

        # Perform local search minimizing augmented cost h
        s_k_plus_1 = local_search(s_k, h)

        # Calculate utility for each feature
        utilities = [I[i](s_k_plus_1) * c[i] / (1 + p[i]) for i in range(M)]
        max_utility = max(utilities)

        # Update penalties for features with maximum utility
        for i in range(M):
            if utilities[i] == max_utility:
                p[i] += 1

        s_k = s_k_plus_1
        k += 1

    return s_k

def local_search(s, h):
    # Domain-specific local search logic (e.g., 2-opt)
    return s
```

### 6.2 Typed Implementation with Explicit Local Minimum Predicate

```python
import random
from typing import Callable, List, Any

def guided_local_search_typed(
    S: List[Any],
    g: Callable[[Any], float],
    lambda_: float,
    I: List[Callable[[Any], float]],
    c: List[float],
    M: int,
    max_iterations: int = 1000,
    is_local_minimum: Callable[[Any], bool] = lambda s: True
) -> Any:
    """
    Performs Guided Local Search over a discrete solution space.

    :param S: Solution candidate pool or initialization generator.
    :param g: Objective function to minimize.
    :param lambda_: Penalty regularization parameter.
    :param I: Feature indicator functions.
    :param c: Feature baseline cost coefficients.
    :param M: Number of distinct features.
    :param max_iterations: Maximum search iterations before termination.
    :param is_local_minimum: Predicate verifying if solution is locally optimal under h.
    :return: Best solution evaluated under original objective g.
    """
    s_best = random.choice(S)
    best_cost = g(s_best)
    s_k = s_best
    p = [0] * M

    for k in range(max_iterations):
        h = g(s_k) + lambda_ * sum(p[i] * I[i](s_k) for i in range(M))
        s_k_plus_1 = local_search_typed(s_k, h)

        cost_unpenalized = g(s_k_plus_1)
        if cost_unpenalized < best_cost:
            s_best = s_k_plus_1
            best_cost = cost_unpenalized

        if not is_local_minimum(s_k_plus_1):
            break

        utilities = [I[i](s_k_plus_1) * c[i] / (1 + p[i]) for i in range(M)]
        max_utility = max(utilities)

        for i in range(M):
            if utilities[i] == max_utility:
                p[i] += 1

        s_k = s_k_plus_1

    return s_best

def local_search_typed(s: Any, h: float) -> Any:
    return s
```

---

## 7. Implementation Notes

- **Feature Choice for TSP**: For routing problems, features typically correspond to edges $e = (u, v)$ in the tour graph. $I_e(s) = 1$ if edge $e \in s$, and $c_e$ is the edge distance.
- **Penalty Scaling ($\lambda$)**: Setting $\lambda \approx \alpha \cdot \frac{g(s_{\text{initial}})}{n}$ (where $\alpha \in [0.1, 0.5]$) balances the search between exploring new basins and refining tour quality.
- **Fast Local Search (FLS)**: Coupling GLS with FLS achieves orders-of-magnitude speedups by only evaluating moves involving vertices adjacent to newly penalized edges.
