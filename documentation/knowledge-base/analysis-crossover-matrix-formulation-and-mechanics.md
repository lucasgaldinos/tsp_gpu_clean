---
title: "Mathematical matrix development of the GPU Genetic Algorithm: 2-route formulations, matrix multiplication analysis, and warp streaming mechanics"
description: >
  Comprehensive linear algebraic development of the entire Genetic Algorithm and
  Local Search pipeline using matrix algebra across 2 parent routes. Formulates
  tour representation, Frobenius inner-product cost evaluation, selection projectors,
  Davis (1985) Order Crossover, antidiagonal exchange mutation, and 2-opt move matrices.
  Rigorously proves why dense matrix multiplication (GEMM, O(n^3)) and permutation
  matrices (O(n^2) VRAM) are computationally prohibitive for discrete routing, and
  how GPU warp-level streaming kernels physically realize these linear transformations
  in O(n) time and O(1) auxiliary memory.
created: 2026-11-09
date_created: 2026-11-09
date_changed: 2026-11-09
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Authored comprehensive matrix development covering the entire GA pipeline
          (Evaluation, Selection, Crossover, Mutation, 2-Opt) using a 2-route formulation,
          contrasting dense matrix multiplication against warp streaming execution.
related_files:
  - "[analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[guide-cuda-hardware-bitwise-and-registers.md](./guide-cuda-hardware-bitwise-and-registers.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[analysis-gpu-2opt-parallel-architectures.md](./analysis-gpu-2opt-parallel-architectures.md)"
  - "[order_crossover.cu](../../src/algorithms/strategies/crossover/kernels/order_crossover.cu)"
  - "[cost_calculator.cu](../../src/algorithms/strategies/evaluation/kernels/cost_calculator.cu)"
  - "[tournament_selection.cu](../../src/algorithms/strategies/selection/kernels/tournament_selection.cu)"
  - "[inversion_mutation.cu](../../src/algorithms/strategies/mutation/kernels/inversion_mutation.cu)"
  - "[two_opt_batch.cu](../../src/algorithms/strategies/local_search/kernels/two_opt_batch.cu)"
tags:
  - analysis/architecture
  - analysis/performance
  - algorithm/crossover
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - algorithm/tsp
  - guide/cuda
  - notes/cuda
  - notes/hardware
  - benchmark/gpu
---

# Mathematical matrix development of the GPU Genetic Algorithm: 2-route formulations, matrix multiplication analysis, and warp streaming mechanics

- [Mathematical matrix development of the GPU Genetic Algorithm: 2-route formulations, matrix multiplication analysis, and warp streaming mechanics](#mathematical-matrix-development-of-the-gpu-genetic-algorithm-2-route-formulations-matrix-multiplication-analysis-and-warp-streaming-mechanics)
  + [1. Executive overview: why formalize the entire GA with matrices?](#1-executive-overview-why-formalize-the-entire-ga-with-matrices)
  + [2. Problem foundation and the 2-route canonical instance](#2-problem-foundation-and-the-2-route-canonical-instance)
    - [2.1 The 8-city benchmark instance](#21-the-8-city-benchmark-instance)
    - [2.2 1D route vectors vs. 2D population batch slices](#22-1d-route-vectors-vs-2d-population-batch-slices)
    - [2.3 Permutation matrices and adjacency matrices](#23-permutation-matrices-and-adjacency-matrices)
  + [3. Stage 1: Tour cost evaluation via matrix algebra](#3-stage-1-tour-cost-evaluation-via-matrix-algebra)
    - [3.1 The Distance Matrix $\mathbf{D}$](#31-the-distance-matrix-mathbfd)
    - [3.2 The Frobenius inner-product formulation: $\operatorname{Tr}(\mathbf{A} \mathbf{D}^{\mathsf{T}})$](#32-the-frobenius-inner-product-formulation-operatornametrmathbfa-mathbfdmathsft)
    - [3.3 Dense matrix multiply vs. GPU block-stride stream evaluation](#33-dense-matrix-multiply-vs-gpu-block-stride-stream-evaluation)
  + [4. Stage 2: Selection via binary projection matrices](#4-stage-2-selection-via-binary-projection-matrices)
    - [4.1 Tournament selection as a 1-hot transformation matrix $\mathbf{W}$](#41-tournament-selection-as-a-1-hot-transformation-matrix-mathbfw)
    - [4.2 Why GEMM $\mathbf{W} \mathbf{P}$ is replaced by 1D winner index gathering](#42-why-gemm-mathbfw-mathbfp-is-replaced-by-1d-winner-index-gathering)
  + [5. Stage 3: Order Crossover (OX) via projection, rotation, and compaction matrices](#5-stage-3-order-crossover-ox-via-projection-rotation-and-compaction-matrices)
    - [5.1 Segment extraction operator $\mathbf{S}$](#51-segment-extraction-operator-mathbfs)
    - [5.2 Incidence indicator vector $\mathbf{v}$](#52-incidence-indicator-vector-mathbfv)
    - [5.3 Circulant shift operator $\mathbf{R}_{(c_2+1)}$ and Candidate Stream](#53-circulant-shift-operator-mathbfr_c_21-and-candidate-stream)
    - [5.4 Lower-triangular compaction operator $\mathbf{L}$](#54-lower-triangular-compaction-operator-mathbfl)
    - [5.5 Offspring tour synthesis](#55-offspring-tour-synthesis)
  + [6. Stage 4: Inversion Mutation via antidiagonal exchange matrices](#6-stage-4-inversion-mutation-via-antidiagonal-exchange-matrices)
    - [6.1 The exchange matrix $\mathbf{J}_k$ and segment reversal operator $\mathbf{M}_{\text{inv}}$](#61-the-exchange-matrix-mathbfj_k-and-segment-reversal-operator-mathbfm_textinv)
    - [6.2 Matrix multiplication $\mathbf{p} \mathbf{M}_{\text{inv}}$ vs. cooperative warp reversal](#62-matrix-multiplication-mathbfp-mathbfm_textinv-vs-cooperative-warp-reversal)
  + [7. Stage 5: 2-Opt Local Search move evaluation and application](#7-stage-5-2-opt-local-search-move-evaluation-and-application)
    - [7.1 The 2-Opt gain matrix $\mathbf{\Delta} \in \mathbb{R}^{n \times n}$](#71-the-2-opt-gain-matrix-mathbfdelta-in-mathbbrn-times-n)
    - [7.2 Tour update as an orthogonal similarity transformation](#72-tour-update-as-an-orthogonal-similarity-transformation)
  + [8. Comprehensive architectural comparison: GEMM vs. GPU streaming kernels](#8-comprehensive-architectural-comparison-gemm-vs-gpu-streaming-kernels)
    - [8.1 Complexity and VRAM footprint across all stages](#81-complexity-and-vram-footprint-across-all-stages)
    - [8.2 Why discrete routing heuristics reject dense GEMM](#82-why-discrete-routing-heuristics-reject-dense-gemm)

---

## 1. Executive overview: why formalize the entire GA with matrices?

In GPU computing and artificial intelligence, students frequently ask:
> *"GPUs have thousands of cores specialized in Matrix Multiplication ($\mathbf{C} = \mathbf{A} \times \mathbf{B}$, or GEMM). Why don't we express the entire Genetic Algorithm — evaluation, selection, crossover, mutation, and 2-opt — using matrix multiplications? What does the mathematical matrix development of the whole process look like across two parent routes?"*

This article provides the complete linear algebraic formalization of the **entire GPU Genetic Algorithm pipeline**. We derive the exact matrix equations for each stage using a concrete 2-route formulation, and perform a rigorous asymptotic and physical hardware analysis proving why dense matrix multiplication is computationally disastrous for discrete routing, while GPU warp-synchronous streaming kernels physically realize the exact algebraic operators in $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ auxiliary memory.

---

## 2. Problem foundation and the 2-route canonical instance

### 2.1 The 8-city benchmark instance

To make every matrix, vector, and trace concrete and visually verifiable on a single page, we use our canonical 8-city TSP instance ($n = 8$):

- **Cities**: $A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7$.
- **Parent 1 Route**:
  $$\mathbf{p}_1 = \begin{bmatrix} C & D & H & B & G & A & F & E \end{bmatrix} = \begin{bmatrix} 2 & 3 & 7 & 1 & 6 & 0 & 5 & 4 \end{bmatrix}$$
- **Parent 2 Route**:
  $$\mathbf{p}_2 = \begin{bmatrix} A & B & C & D & E & F & G & H \end{bmatrix} = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix}$$
- **Crossover Cut Points**: $c_1 = 2, \quad c_2 = 5$.
- **Mutation Segment**: $i = 2, \quad j = 5$.

### 2.2 1D route vectors vs. 2D population batch slices

In a population of size $N$, routes are stored horizontally in row-major order as a 2D matrix $\mathbf{P} \in \{0, \dots, n-1\}^{N \times n}$.
For our formalization, we isolate a **2-route slice**:

$$\mathbf{P}_{\text{2-routes}} = \begin{bmatrix} \mathbf{p}_1 \\ \mathbf{p}_2 \end{bmatrix} = \begin{bmatrix}
2 & 3 & 7 & 1 & 6 & 0 & 5 & 4 \\
0 & 1 & 2 & 3 & 4 & 5 & 6 & 7
\end{bmatrix} \in \mathbb{N}^{2 \times 8}$$

### 2.3 Permutation matrices and adjacency matrices

In linear algebra, a discrete route $\mathbf{p} \in \{0, \dots, n-1\}^n$ can be represented in two canonical matrix forms:

#### 1. Permutation Matrix $\mathbf{\Pi} \in \{0, 1\}^{n \times n}$
$\Pi_{i, j} = 1$ if city $j$ appears at tour index $i$:
$$\mathbf{\Pi}_2 = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} = \mathbf{I}_8$$

Since Parent 2 is identity sorted ($0, 1, 2, 3, 4, 5, 6, 7$), its permutation matrix is the identity matrix $\mathbf{I}_8$.
For Parent 1 $\mathbf{p}_1 = [2, 3, 7, 1, 6, 0, 5, 4]$:
$$\mathbf{\Pi}_1 = \begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}$$

#### 2. Directed Adjacency Matrix $\mathbf{A} \in \{0, 1\}^{n \times n}$
$A_{u, v} = 1$ if the directed edge $(u, v)$ is traversed in the closed tour:
$$\mathbf{A} = \mathbf{\Pi}^{\mathsf{T}} \mathbf{C}_{\text{cyclic}} \mathbf{\Pi}$$
where $\mathbf{C}_{\text{cyclic}}$ is the canonical cyclic permutation matrix (with $1$s on the superdiagonal and at $(n-1, 0)$).

---

## 3. Stage 1: Tour cost evaluation via matrix algebra

### 3.1 The Distance Matrix $\mathbf{D}$

Let $\mathbf{D} \in \mathbb{R}^{n \times n}$ be the TSP distance matrix where $D_{u, v}$ is the Euclidean or geographic distance between city $u$ and city $v$.

### 3.2 The Frobenius inner-product formulation: $\operatorname{Tr}(\mathbf{A} \mathbf{D}^{\mathsf{T}})$

In linear algebra, the total tour cost $f(\mathbf{p})$ of a closed-loop route is the **Frobenius inner product** (matrix dot product) of the adjacency matrix $\mathbf{A}$ and distance matrix $\mathbf{D}$:

$$f(\mathbf{p}) = \langle \mathbf{A}, \mathbf{D} \rangle_F = \sum_{u=0}^{n-1} \sum_{v=0}^{n-1} A_{u, v} D_{u, v} = \operatorname{Tr}(\mathbf{A} \mathbf{D}^{\mathsf{T}})$$

Substituting the permutation matrix $\mathbf{\Pi}$:

$$f(\mathbf{p}) = \operatorname{Tr}\left( \mathbf{\Pi}^{\mathsf{T}} \mathbf{C}_{\text{cyclic}} \mathbf{\Pi} \mathbf{D}^{\mathsf{T}} \right)$$

For our 2-route slice $\mathbf{P}_{\text{2-routes}} = [\mathbf{p}_1, \mathbf{p}_2]^{\mathsf{T}}$, the cost vector $\mathbf{f} \in \mathbb{R}^2$ is:

$$\mathbf{f} = \begin{bmatrix}
\operatorname{Tr}(\mathbf{A}_1 \mathbf{D}^{\mathsf{T}}) \\
\operatorname{Tr}(\mathbf{A}_2 \mathbf{D}^{\mathsf{T}})
\end{bmatrix}$$

### 3.3 Dense matrix multiply vs. GPU block-stride stream evaluation

Can we compute tour cost on GPU using general matrix multiplication (GEMM)?

- **Dense GEMM Approach**:
  Computing $\mathbf{M} = \mathbf{\Pi}^{\mathsf{T}} \mathbf{C}_{\text{cyclic}} \mathbf{\Pi}$ requires two dense matrix multiplications ($\mathcal{O}(n^3)$ operations).
  Computing $\operatorname{Tr}(\mathbf{M} \mathbf{D}^{\mathsf{T}})$ requires a third matrix multiplication ($\mathcal{O}(n^3)$).
  For $n = 2{,}000$ cities: $2 \times (2{,}000)^3 = \mathbf{16\text{ billion operations}}$ per route!
- **GPU Streaming Kernel (`cost_calculator.cu`)**:
  Because $\mathbf{A}$ has only **$n$ non-zero entries** ($A_{p_i, p_{(i+1)\%n}} = 1$), the sum reduces to:
  $$f(\mathbf{p}) = \sum_{i=0}^{n-1} D_{p_i, \, p_{(i+1) \bmod n}}$$
  Our GPU kernel streams $p_i$ using a block-stride loop with **$n$ memory loads**, reducing the work from $\mathcal{O}(n^3)$ to **$\mathcal{O}(n)$** operations ($\approx 2{,}000$ operations)!
  Warp shuffle reduction (`__shfl_down_sync`) sums the partial edges in **5 clock cycles**.

---

## 4. Stage 2: Selection via binary projection matrices

### 4.1 Tournament selection as a 1-hot transformation matrix $\mathbf{W}$

In tournament selection, $M$ parents are selected from a population of size $N$.
In linear algebra, this is a linear mapping from $\mathbb{R}^{N \times n}$ to $\mathbb{R}^{M \times n}$ mediated by a **row selection matrix** $\mathbf{W} \in \{0, 1\}^{M \times N}$:

$$\mathbf{P}_{\text{parents}} = \mathbf{W} \mathbf{P}_{\text{population}}$$

Each row $m$ of $\mathbf{W}$ is a 1-hot vector where $W_{m, k} = 1$ if individual $k$ won the $m$-th tournament.

#### Concrete 2-Route Example:
Suppose our population has $N = 2$ routes ($P_1$ and $P_2$).
We hold $M = 2$ tournaments:
1. Tournament 1 picks $P_1$ as winner $\implies \mathbf{w}_1 = [1, 0]$.
2. Tournament 2 picks $P_2$ as winner $\implies \mathbf{w}_2 = [0, 1]$.

The selection matrix is:

$$\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \mathbf{I}_2$$

If $P_1$ had won both tournaments, $\mathbf{W}$ would be:

$$\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix} \implies \mathbf{P}_{\text{parents}} = \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} \mathbf{p}_1 \\ \mathbf{p}_2 \end{bmatrix} = \begin{bmatrix} \mathbf{p}_1 \\ \mathbf{p}_1 \end{bmatrix}$$

### 4.2 Why GEMM $\mathbf{W} \mathbf{P}$ is replaced by 1D winner index gathering

- **Dense Matrix Multiply ($\mathbf{W} \times \mathbf{P}$)**:
  Multiplying an $(M \times N)$ matrix by an $(N \times n)$ matrix requires $2 M N n$ arithmetic operations.
  For $M = N = 256$ and $n = 2{,}000$:
  $$2 \times 256 \times 256 \times 2{,}000 = \mathbf{262{,}144{,}000\text{ operations}}$$
  Over $262\text{ million operations}$ just to copy 256 routes!
- **Our `SEL-A` Gather Solution (`population[winner_indices]`)**:
  Because each row of $\mathbf{W}$ has only a single $1$, $\mathbf{W}$ is stored as a 1D index array $\mathbf{w} \in \mathbb{N}^M$:
  $$\mathbf{w} = \begin{bmatrix} 0 & 1 \end{bmatrix}$$
  The CUDA kernel computes the winner ID in $\mathcal{O}(k)$ comparisons, and CuPy's CUB gather copies routes in $M \times n$ memory writes ($\mathbf{512{,}000\text{ operations}}$).
  **Performance Gain**: **$512\times$ fewer operations**, 0 bytes of wasted GEMM compute!

---

## 5. Stage 3: Order Crossover (OX) via projection, rotation, and compaction matrices

### 5.1 Segment extraction operator $\mathbf{S}$

Order Crossover extracts the substring between cut points $c_1 = 2$ and $c_2 = 5$ from Parent 1:
$$\mathbf{S} = \operatorname{diag}(0, 0, 1, 1, 1, 1, 0, 0) \in \{0, 1\}^{8 \times 8}$$

Multiplying Parent 1 by $\mathbf{S}$ isolates the preserved slice:
$$\mathbf{s}_1 = \mathbf{p}_1 \mathbf{S} = \begin{bmatrix} \emptyset & \emptyset & H & B & G & A & \emptyset & \emptyset \end{bmatrix} = \begin{bmatrix} \emptyset & \emptyset & 7 & 1 & 6 & 0 & \emptyset & \emptyset \end{bmatrix}$$

> [!NOTE]
> In our integer encoding, City $A = 0$. Therefore, tour index $5$ holds the valid city $A$ ($0$), while tour indices $0, 1, 6, 7$ are vacant slots awaiting genes from Parent 2 (denoted $\emptyset$). In matrix terms, the non-zero rows of the sliced permutation matrix $\mathbf{\Pi}_{\text{slice}} = \mathbf{S} \mathbf{\Pi}_1$ contain the exact 1-hot encodings of $\{H, B, G, A\}$.

### 5.2 Incidence indicator vector $\mathbf{v}$

The city incidence vector $\mathbf{v} \in \{0, 1\}^8$ marks which cities are present in $\mathbf{s}_1$:
$$\mathbf{v} = \mathbf{1}^{\mathsf{T}} (\mathbf{S} \mathbf{\Pi}_1) = \begin{bmatrix} \mathbf{1} & \mathbf{1} & 0 & 0 & 0 & 0 & \mathbf{1} & \mathbf{1} \end{bmatrix} \quad (\text{Cities } A, B, G, H)$$

### 5.3 Circulant shift operator $\mathbf{R}_{(c_2+1)}$ and Candidate Stream

In Davis (1985) Order Crossover, candidate cities from Parent 2 are streamed cyclically starting immediately after the second cut point at index $(c_2 + 1) \bmod 8 = \mathbf{6}$.
In linear algebra, rotating a row vector cyclically by $k=6$ positions to the left is mediated by the circulant shift permutation matrix $\mathbf{R}_6 \in \{0, 1\}^{8 \times 8}$:

$$\mathbf{R}_6 = \begin{bmatrix}
\mathbf{0}_{6 \times 2} & \mathbf{I}_{6 \times 6} \\
\mathbf{I}_{2 \times 2} & \mathbf{0}_{2 \times 6}
\end{bmatrix} = \begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}$$

Multiplying $\mathbf{p}_2$ by $\mathbf{R}_6$ produces the cyclically rotated candidate stream:

$$\mathbf{p}_2^{(\text{rot})} = \mathbf{p}_2 \mathbf{R}_6 = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix} \mathbf{R}_6 = \begin{bmatrix} 6 & 7 & 0 & 1 & 2 & 3 & 4 & 5 \end{bmatrix} = \begin{bmatrix} G & H & A & B & C & D & E & F \end{bmatrix}$$

Validity predicate vector $\mathbf{u} = \mathbf{1} - \mathbf{v}(\mathbf{p}_2^{(\text{rot})})$ identifies which streamed cities are missing from the preserved slice:
$$\mathbf{u} = \begin{bmatrix} 0 & 0 & 0 & 0 & \mathbf{1} & \mathbf{1} & \mathbf{1} & \mathbf{1} \end{bmatrix}$$

### 5.4 Lower-triangular compaction operator $\mathbf{L}$

The exclusive prefix sum rank is computed by multiplying by the strictly lower-triangular matrix $\mathbf{L}$:

$$\mathbf{r} = \mathbf{L} \mathbf{u}^{\mathsf{T}} = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 0
\end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ \mathbf{0} \\ \mathbf{1} \\ \mathbf{2} \\ \mathbf{3} \end{bmatrix}$$

### 5.5 Offspring tour synthesis

Each valid lane $m \in \{4, 5, 6, 7\}$ writes to cyclic slot $(6 + r_m) \bmod 8$:
- City $C$ writes to Slot 6
- City $D$ writes to Slot 7
- City $E$ writes to Slot 0
- City $F$ writes to Slot 1

Child route vector:
$$\mathbf{p}_{\text{child}} = \begin{bmatrix} 4 & 5 & 7 & 1 & 6 & 0 & 2 & 3 \end{bmatrix} = \begin{bmatrix} E & F & H & B & G & A & C & D \end{bmatrix}$$

In hardware, $\mathbf{L} \mathbf{u}^{\mathsf{T}}$ is executed in **1 single clock cycle** using `__ballot_sync` + `__popc`!

---

## 6. Stage 4: Inversion Mutation via antidiagonal exchange matrices

### 6.1 The exchange matrix $\mathbf{J}_k$ and segment reversal operator $\mathbf{M}_{\text{inv}}$

Inversion mutation reverses a subsegment of cities between indices $i$ and $j$.
In linear algebra, reversing an array is mediated by the **exchange matrix $\mathbf{J}$** (the standard antidiagonal reflection matrix):

$$\mathbf{J}_k = \begin{bmatrix}
0 & 0 & \dots & 0 & 1 \\
0 & 0 & \dots & 1 & 0 \\
\vdots & \vdots & \iddots & \vdots & \vdots \\
1 & 0 & \dots & 0 & 0
\end{bmatrix} \in \{0, 1\}^{k \times k}$$

For an 8-city route reversing segment $i=2$ to $j=5$ ($k = 4$ cities: $[2 \dots 5]$):
The segment reversal matrix $\mathbf{M}_{\text{inv}} \in \{0, 1\}^{8 \times 8}$ is a block-diagonal matrix:

$$\mathbf{M}_{\text{inv}} = \begin{bmatrix}
\mathbf{I}_2 & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{J}_4 & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{I}_2
\end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{1} & 0 & 0 \\
0 & 0 & \mathbf{0} & \mathbf{0} & \mathbf{1} & \mathbf{0} & 0 & 0 \\
0 & 0 & \mathbf{0} & \mathbf{1} & \mathbf{0} & \mathbf{0} & 0 & 0 \\
0 & 0 & \mathbf{1} & \mathbf{0} & \mathbf{0} & \mathbf{0} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

Multiplying Parent 2 by $\mathbf{M}_{\text{inv}}$:
$$\mathbf{p}_{\text{mutated}} = \mathbf{p}_2 \mathbf{M}_{\text{inv}} = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix} \mathbf{M}_{\text{inv}} = \begin{bmatrix} 0 & 1 & \mathbf{5} & \mathbf{4} & \mathbf{3} & \mathbf{2} & 6 & 7 \end{bmatrix}$$
The segment $[2, 3, 4, 5]$ is inverted to $[5, 4, 3, 2]$!

### 6.2 Matrix multiplication $\mathbf{p} \mathbf{M}_{\text{inv}}$ vs. cooperative warp reversal

- **GEMM Approach**:
  Multiplying a route by $\mathbf{M}_{\text{inv}}$ takes $\mathcal{O}(n^2)$ operations. For $n = 2{,}000$, that is $4{,}000{,}000$ operations!
- **Cooperative Warp Reversal (`INV-A`)**:
  Because $\mathbf{M}_{\text{inv}}$ is an exchange matrix, reversing requires only swapping $k/2$ pairs of elements.
  Our 32-thread warp executes all swaps in parallel across registers:
  $$\text{Thread } t \text{ swaps: } \operatorname{route}[i + t] \longleftrightarrow \operatorname{route}[j - t]$$
  For a 64-city segment, 32 threads complete the reversal in **1 single loop iteration**!

---

## 7. Stage 5: 2-Opt Local Search move evaluation and application

### 7.1 The 2-Opt gain matrix $\mathbf{\Delta} \in \mathbb{R}^{n \times n}$

In 2-opt local search, every pair of edges $(u, v)$ and $(x, y)$ is evaluated for potential reconnection.
The cost difference $\Delta(i, j)$ defines the **2-Opt Gain Matrix** $\mathbf{\Delta} \in \mathbb{R}^{n \times n}$:

$$\Delta_{i, j} = D_{p_i, \, p_j} + D_{p_{i+1}, \, p_{j+1}} - D_{p_i, \, p_{i+1}} - D_{p_j, \, p_{j+1}}$$

Finding the best 2-opt move is formulated as finding the minimum entry in $\mathbf{\Delta}$:

$$(i^*, j^*) = \arg\min_{i < j} \Delta_{i, j}$$

### 7.2 Tour update as an orthogonal similarity transformation

When $\Delta_{i^*, j^*} < 0$, the move is applied.
Applying 2-opt corresponds to multiplying the tour permutation matrix by the exchange matrix $\mathbf{M}_{\text{inv}}(i^*, j^*)$ from Section 6.1.
Because $\mathbf{M}_{\text{inv}}$ operates on **tour position indices** (rows of $\mathbf{\Pi}$):

$$\mathbf{\Pi}_{\text{new}} = \mathbf{M}_{\text{inv}}(i^*, j^*) \, \mathbf{\Pi}_{\text{old}}$$

The adjacency matrix $\mathbf{A} \in \{0, 1\}^{n \times n}$ lives in **city space** ($A_{u, v} = 1$ if edge between city $u$ and city $v$ is traversed). Recalling from Section 2.3 that $\mathbf{A} = \mathbf{\Pi}^{\mathsf{T}} \mathbf{C}_{\text{cyclic}} \mathbf{\Pi}$, the updated adjacency matrix becomes:

$$\mathbf{A}_{\text{new}} = \mathbf{\Pi}_{\text{new}}^{\mathsf{T}} \, \mathbf{C}_{\text{cyclic}} \, \mathbf{\Pi}_{\text{new}} = \mathbf{\Pi}_{\text{old}}^{\mathsf{T}} \, \left( \mathbf{M}_{\text{inv}}^{\mathsf{T}} \, \mathbf{C}_{\text{cyclic}} \, \mathbf{M}_{\text{inv}} \right) \, \mathbf{\Pi}_{\text{old}}$$

Notice that in position space, the cyclic successor matrix undergoes an orthogonal similarity transformation: $\mathbf{C}_{\text{new}} = \mathbf{M}_{\text{inv}}^{\mathsf{T}} \, \mathbf{C}_{\text{cyclic}} \, \mathbf{M}_{\text{inv}}$.
Equivalently, defining the **city-space reflection operator** $\mathbf{T}_{\text{city}} \in \{0, 1\}^{n \times n}$:

$$\mathbf{T}_{\text{city}} = \mathbf{\Pi}_{\text{old}}^{\mathsf{T}} \, \mathbf{M}_{\text{inv}}(i^*, j^*) \, \mathbf{\Pi}_{\text{old}}$$

The adjacency matrix transforms in city space via the orthogonal similarity transformation:

$$\mathbf{A}_{\text{new}} = \mathbf{T}_{\text{city}}^{\mathsf{T}} \, \mathbf{A}_{\text{old}} \, \mathbf{T}_{\text{city}}$$

In our GPU kernels (`two_opt_batch.cu`), instead of materializing the dense $n \times n$ transformation matrix, the block of threads reverses the subsegment directly in shared memory (`2OPT-A`), completing the move in $\mathcal{O}(n / 256)$ parallel steps!

---

## 8. Comprehensive architectural comparison: GEMM vs. GPU streaming kernels

### 8.1 Complexity and VRAM footprint across all stages

| GA Pipeline Stage | Linear Algebra Formulation | Dense GEMM Time Complexity | Linear / Warp Streaming Complexity | Work Reduction Factor ($n = 2{,}000$) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Tour Evaluation** | $\operatorname{Tr}(\mathbf{A} \mathbf{D}^{\mathsf{T}})$ (Frobenius product) | $\mathcal{O}(n^3)$ | $\mathbf{\mathcal{O}(n)}$ (Block-stride loop) | **$4{,}000{,}000\times$ faster** |
| **2. Selection** | $\mathbf{P}_{\text{parents}} = \mathbf{W} \mathbf{P}_{\text{pop}}$ | $\mathcal{O}(M N n)$ | $\mathbf{\mathcal{O}(M n)}$ (CuPy index gather) | **$256\times$ faster** |
| **3. Crossover (OX)** | $\mathbf{s}_1 + \mathbf{L} \mathbf{u}^{\mathsf{T}}$ projection | $\mathcal{O}(n^3)$ | $\mathbf{\mathcal{O}(n)}$ (Warp stream compaction) | **$250{,}000{,}000\times$ faster** |
| **4. Mutation** | $\mathbf{p} \mathbf{M}_{\text{inv}}$ (Antidiagonal reflection) | $\mathcal{O}(n^2)$ | $\mathbf{\mathcal{O}(L / 32)}$ (Cooperative warp) | **$125{,}000\times$ faster** |
| **5. 2-Opt Local Search** | $\mathbf{\Pi}_{\text{new}} = \mathbf{M}_{\text{inv}} \mathbf{\Pi}$ | $\mathcal{O}(n^3)$ | $\mathbf{\mathcal{O}(n / \text{block})}$ (Shared memory reverse) | **$16{,}000{,}000\times$ faster** |

### 8.2 Why discrete routing heuristics reject dense GEMM

Matrix multiplication is the cornerstone of deep learning because neural networks compute dense linear combinations where every weight interacts with every activation ($\Theta(n^2)$ parameters, $\Theta(n^3)$ arithmetic).

In discrete routing:
1. **Permutations are Ultra-Sparse**: A tour of $n$ cities has only $n$ edges out of $n^2$ possible connections. Storing and multiplying dense $n \times n$ matrices wastes $99.95\%$ of memory on zeros!
2. **VRAM Exhaustion**: Storing 256 routes as permutation matrices for $n = 2{,}103$ requires **4.53 GB** of VRAM, exceeding the physical memory limit of our GTX 1050 (2 GB) and causing an immediate out-of-memory crash.
3. **The Power of Warp Collectives**: NVIDIA warp primitives (`__ballot_sync`, `__popc`, `__shfl_*_sync`) allow 32 threads to execute linear algebraic projections, prefix sums, and reductions directly inside registers in **1 single clock tick** without touching DRAM.

**Conclusion**: Linear algebra provides the rigorous mathematical language to specify the Genetic Algorithm, but **warp-synchronous streaming kernels** are the physically optimal execution engine!
