---
date: "2025-11-26"
title: "Explain, step by step, as proposed by Goldberg in..."
description: "This is a screenshot of the Gemini AI chatbot interface. It displays the user's recent chat history, a 'Gems' section for pinned or important conversations (like 'TSP variants master' or 'prompt master'), and the current active conversation with Gemini. The interface provides options for starting new chats, exploring saved 'Gems', and accessing account settings."
---
### Logical Analysis of TSP Operators

You require a rigorous mathematical formulation of the Genetic Algorithm operators defined in your `GeneticAlgorithmBase` class, specifically applied to the Traveling Salesperson Problem (TSP).

The following analysis breaks down the problem definition and the four specific operators: **Tournament Selection**, **Order Crossover (OX)**, **Swap Mutation**, and **($\mu+\lambda$) Survival Selection**.

---

### 1. Mathematical Definition of the TSP

The TSP is a combinatorial optimization problem formulated on a complete graph.

#### Symbol Definitions
  - **$G = (V, E)$**: The graph representing the problem.
  - **$V = \{1, 2, \dots, n\}$**: The set of vertices (cities), where $n$ is the number of cities.
  - **$E$**: The set of edges connecting every pair of vertices.
  - **$d_{ij}$**: The scalar weight (distance/cost) associated with the edge connecting city $i$ to city $j$.
  - **$\Pi$**: The solution space (set of all possible permutations of $V$).
  - **$\pi$**: A single solution vector (genotype), represented as a permutation $\pi = [\pi_1, \pi_2, \dots, \pi_n]$.
  - **$\pi_i$**: The city located at position $i$ in the tour.

#### Objective Function
The goal is to find a permutation $\pi^* \in \Pi$ that minimizes the Hamiltonian cycle cost $f(\pi)$:

$$f(\pi) = \left( \sum_{i=1}^{n-1} d_{\pi_i, \pi_{i+1}} \right) + d_{\pi_n, \pi_1}$$

---

### 2. Tournament Selection
Your code implements a $k$-way tournament selection. This is a stochastic operator that exerts selection pressure by comparing a small subset of the population.

#### Symbol Definitions
  - **$P_t$**: The population set at generation $t$.
  - **$N$**: The size of the population ($|P_t| = N$).
  - **$k$**: The tournament size (parameter `tournament_size=5`).
  - **$S$**: The tournament subset.
  - **$U(P_t)$**: A uniform random sampling function from $P_t$.

#### Operator Formulation
To select a single parent $\pi_{selected}$:

1.  **Sampling:** A subset $S$ is formed by sampling $k$ individuals from $P_t$ independently with replacement.
    $$S = \{s_1, s_2, \dots, s_k\} \quad \text{where } s_i \sim U(P_t)$$

2.  **Selection:** The operator returns the individual in $S$ with the minimum objective function value (minimization problem).
    $$\pi_{selected} = \operatorname*{argmin}_{s \in S} f(s)$$

---

### 3. Order Crossover (OX)
The OX operator constructs an offspring by preserving a sub-tour from the first parent and filling the remaining slots with the relative ordering of cities from the second parent. This avoids invalid tours (duplicates or missing cities).

#### Symbol Definitions
  - **$\pi^1$**: The first parent permutation.
  - **$\pi^2$**: The second parent permutation.
  - **$\chi$**: The resulting offspring permutation.
  - **$c_1, c_2$**: Two integer cut points such that $1 \le c_1 < c_2 \le n$.
  - **$\Phi$**: The set of cities already contained in the inherited segment.

#### Operator Formulation
The offspring $\chi$ is constructed in two phases:

**Phase 1: Segment Inheritance**
The offspring inherits the segment between cut points $c_1$ and $c_2$ directly from $\pi^1$.
$$\forall k \in [c_1, c_2], \quad \chi_k = \pi^1_k$$

Define the set of fixed cities $\Phi$:
$$\Phi = \{ \pi^1_k \mid c_1 \le k \le c_2 \}$$

**Phase 2: Sequence Completion**
We generate a reference sequence from $\pi^2$ starting after the second cut point, wrapping around:
$$Seq(\pi^2) = \langle \pi^2_{c_2+1}, \dots, \pi^2_n, \pi^2_1, \dots, \pi^2_{c_2} \rangle$$

We filter this sequence to exclude cities already in $\Phi$:
$$Seq' = \{ v \mid v \in Seq(\pi^2) \land v \notin \Phi \}$$

The elements of $Seq'$ are then placed into the empty positions of $\chi$ starting from $c_2+1$ (wrapping around to $1$):
$$\text{Indices to fill: } K = \langle c_2+1, \dots, n, 1, \dots, c_1-1 \rangle$$

For the $m$-th index in $K$ (denoted $K_m$), and the $m$-th value in $Seq'$ (denoted $Seq'_m$):
$$\chi_{K_m} = Seq'_m$$

---

### 4. Swap Mutation
This operator introduces diversity by performing a single 2-exchange (transposition) on the tour.

#### Symbol Definitions
  - **$\pi$**: The original individual.
  - **$\pi'$**: The mutated individual.
  - **$r_1, r_2$**: Two distinct random indices sampled uniformly from $\{1, \dots, n\}$.

#### Operator Formulation
The mutation defines a permutation $\pi'$ where the values at indices $r_1$ and $r_2$ are swapped, while all other positions remain the identity:

$$
\pi'_k = 
\begin{cases} 
\pi_{r_2} & \text{if } k = r_1 \\
\pi_{r_1} & \text{if } k = r_2 \\
\pi_k & \text{otherwise}
\end{cases}
$$

Mathematically, this applies the transposition $\tau_{r_1, r_2}$ to the permutation $\pi$.

---

### 5. Survival Selection ($\mu + \lambda$)
Your algorithm uses a strict elitist replacement strategy known as $(\mu + \lambda)$. This ensures the population size remains constant and the best solutions are preserved.

#### Symbol Definitions
  - **$\mu$**: The size of the parent population ($|P_t| = \mu$).
  - **$\lambda$**: The size of the offspring population ($|Q_t| = \lambda$).
  - **$P_t$**: Parent population at generation $t$.
  - **$Q_t$**: Offspring population created in generation $t$.
  - **$P_{t+1}$**: Population for the next generation.
  - **$\mathcal{R}(x, A)$**: The rank of individual $x$ within set $A$ based on fitness $f(x)$ (where rank 1 is best).

#### Operator Formulation
1.  **Union:** Combine parents and offspring into a temporary pool $U$.
    $$U = P_t \cup Q_t$$

2.  **Ranking:** Sort all individuals in $U$ based on fitness $f$ in ascending order (minimizing cost).
    $$U_{sorted} = \text{sort}(U, f)$$

3.  **Truncation:** The new population consists of the top $\mu$ individuals from $U$.
    $$P_{t+1} = \{ x \in U \mid \mathcal{R}(x, U) \le \mu \}$$

This formulation guarantees:
$$\min_{p \in P_{t+1}} f(p) \le \min_{p \in P_t} f(p)$$
(The best fitness in the population is monotonically non-increasing).