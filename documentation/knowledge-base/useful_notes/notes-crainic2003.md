Ran `read-pdf`
Completed with input: {
  "file": "/home/lucas_galdino/chimera/gpu_accelerated/documentation/academic/first_draft.pdf",
  "pages": "all",
  "include_metadata": true,
  "clean_text": true
}

Read [](file:///home/lucas_galdino/chimera/gpu_accelerated/documentation)

Read [](file:///home/lucas_galdino/chimera/gpu_accelerated/documentation/pdfs)

Ran `read-pdf`
Completed with input: {
  "file": "/home/lucas_galdino/chimera/gpu_accelerated/documentation/pdfs/Parallel_Strategies_for_Meta-Heuristics.pdf",
  "pages": "all",
  "include_metadata": true,
  "clean_text": true
}

Summarized conversation history

Starting: *Summarize Crainic-Toulouse PDF* (1/2)

Completed: *Summarize Crainic-Toulouse PDF* (1/2)

## Crainic & Toulouse (2003) — Parallel Strategies for Meta‑Heuristics

Below is a concise, thesis‑oriented markdown summary of the PDF you just had me read, with emphasis on concepts you are likely to reuse.

---

### 1. Scope and Motivation (Intro, pp. 476–478)

- Parallel metaheuristics are positioned as:
  - A way to **solve larger instances faster**.
  - A way to obtain **more robust** behavior across different instances (less sensitivity to parameters, better consistency).
- Metaheuristics are described as “master strategies” that guide local heuristics to avoid local optima and cycling.
- The paper explicitly focuses on **three major classes**:
  - Genetic methods (genetic algorithms / evolutionary methods).
  - Simulated annealing.
  - Tabu search.
- It provides:
  1. A **generic metaheuristic framework**.
  2. A **classification of parallelization strategies** (Types 1–3).
  3. Instantiations for GA, SA, TS.
  4. Emerging trends and research directions.

Useful for:  

- Chapter 2 (theoretical foundation of metaheuristics) to justify your definitions.  
- Chapter 3 (parallelization taxonomy) to position your work relative to known strategies.

---

### 2. Generic Metaheuristic Framework (Sec. 2, pp. 477–480)

They formalize metaheuristics as a **“walk through neighborhoods”** in the solution space of a combinatorial optimization problem.

Core components:

1. **Initialization**  
   - Build an initial solution or population (feasible or not).

2. **Neighborhoods**  
   - For a solution \(x\), define multiple neighborhoods \(N_i(x)\), \(i=1,\dots,q\).
   - Have a **neighborhood selection rule** (when to use which neighborhood).

3. **Candidate Selection**  
   - Neighborhoods can be very large → restrict to a **candidate list**.
   - Candidate list construction can be static (updated incrementally) or rebuilt per iteration.

4. **Acceptance Criterion**  
   - Evaluate a move using function \(g(x, y)\).
   - May combine objective value, constraint violations, penalties, history, randomness.
   - Choose move(s) according to this criterion.

5. **Stopping Criteria**  
   - Time, number of iterations, rate of improvement, or phase‑specific criteria.

They then show that GA, SA and TS all fit into this generic loop (Figures 17.2–17.6).

Useful for:  

- Chapter 2: a clean generic template for describing your GA+2‑opt as a metaheuristic instance.  
- Later, to contrast “generic” metaheuristic steps with what is pushed to GPU in your design.

---

### 3. Parallel Computing Concepts (Sec. 3.1, pp. 483–486)

They review classic parallelism notions:

- **Data parallelism**  
  - Same operation on many data items (e.g., SIMD).
  - Example: evaluate fitness of many individuals independently, or costs for many moves.

- **Functional parallelism**  
  - Different operations executed in parallel (e.g., multiple functions or phases).

- **Irregular algorithms** like metaheuristics
  - Irregular data structures (graphs, tours) and dependencies between iterations limit straightforward data/functional parallelism.
  - Nevertheless, **two main parallelism sources** in metaheuristics:
    - Within an iteration (e.g., evaluating a neighborhood).
    - Across independent or loosely coupled runs (multi‑start / multi‑thread).

This is the conceptual base for their taxonomy.

Useful for:  

- Your distinction between CPU vs GPU **data parallelism** (kernels) vs possible multi‑thread GA variants (which you *don’t* use, but can mention).

---

### 4. Three Types of Parallelization Strategies (Sec. 3.1, pp. 485–487)

This is the key taxonomy:

1. **Type 1: Low‑level / intra‑iteration parallelism**
   - Parallel work **within one iteration** of a single search trajectory.
   - Examples:
     - Evaluate neighborhood moves in parallel.
     - Parallel evaluation of population fitness in GA.
   - Aim: **speed up** a given method, without conceptually changing the search (same trajectory if the same neighbors evaluated).
   - Sometimes extended with **“probing”**: exploring a few steps ahead for each candidate move.
   - Classical speedup metrics apply when the parallel and sequential algorithms are “algorithmically equivalent”.

2. **Type 2: Decomposition by decision variables**
   - Partition decision variables into disjoint subsets; each subset explored by a slave, usually in a master–slave framework.
   - Often requires **repeated re‑partitioning** to cover the entire space.
   - The parallel method typically **does not follow** the same search path as the sequential one.
   - Used, e.g., as domain decomposition or adaptive memory ideas.

3. **Type 3: Multiple concurrent searches (multi‑thread / multi‑walk)**
   - Run **several search threads** (independent or cooperative) through solution space.
   - Threads can:
     - Use the same or different metaheuristics.
     - Start from same or different initial solutions.
     - Exchange information synchronously or asynchronously.
   - Difficult to compare with classical speedups (overlap of work, different trajectories).
   - Very general umbrella: includes **independent multi‑start**, **cooperative island models**, central memory, adaptive memory, etc.

Useful for your TCC:

- Your GA+2‑opt GPU work uses **Type 1** parallelism:  
  - Evaluate many tours / 2‑opt moves on GPU in parallel, preserving a **single GA trajectory**.
- You can explicitly say: *“We focus on Type 1 (low‑level data‑parallel) strategies on GPUs, following the classification of Crainic & Toulouse [@crainic2003parallel].”*

---

### 5. Other Existing Taxonomies (Sec. 3.2, pp. 487–488)

They connect their three‑type classification to other well‑known taxonomies:

- Greening (1990) for SA: synchronous vs asynchronous; serial‑like vs altered generation.
- Cantú‑Paz (1995, 1998) for GA: global, coarse‑grained, fine‑grained.
- Verhoeven & Aarts (1995): single‑walk vs multiple‑walk; synchronous vs asynchronous (their “multiple‑walk” ≈ Type 3).
- Crainic et al. (1997) for tabu search: dimensions for:
  - **Control cardinality** (single controller vs collegial).
  - **Search differentiation** (same vs different start solutions and strategies).
  - **Control type** (rigid synchronization, knowledge synchronization, collegial, knowledge collegial).

Useful for:

- A literature subsection where you **map your work into multiple taxonomies** (e.g., your GA is single‑walk, Type 1; you could mention that you do not explore p‑control / Type 3 in this thesis).

---

### 6. Genetic Algorithms (Sec. 4, pp. 488–493)

They survey parallel GA patterns:

- **Type 1**:
  - Parallelize the **fitness evaluation / average fitness** over population (classical).
  - Communication overhead often prevents ideal speedups, especially on slow networks.

- **Type 3**:
  - **Coarse‑grained (island) models**:
    - Several subpopulations; each runs a GA.
    - Migration operator with parameters: who migrates, migration rate, interval, topology (island vs stepping‑stone), synchronous vs asynchronous.
    - Focus on convergence behavior and diversity; speedups are usually secondary.
  - **Fine‑grained (cellular) models**:
    - Each processor corresponds to one or few individuals; neighborhoods define small “demes”.
    - Genetic operators act locally; diffusion of genetic material across the grid.
    - Topology (e.g. torus) is crucial for balancing isolation vs premature convergence.
  - **Self‑adaptation / co‑evolution**:
    - E.g. evolving GA parameters (mutation probability, etc.) across subpopulations.
  - **Memetic algorithms / hybrids with local search**:
    - GA at higher level, local improvement (hill climbing, 2‑opt, etc.) at lower level.
    - Particularly interesting when local search is expensive.

Crucially, they remark:

- Pure Type 1 parallelization of **standard GA** is still relevant and competitive, especially for large populations and expensive fitness or local search.
- Type 3 is more innovation‑rich (islands, co‑evolution, memetic).

Useful for your TCC:

- When introducing your **GA+2‑opt with GPU**, you can:
  - Refer to GA+local search as a **memetic algorithm**.
  - Clarify that you **fix the GA architecture** (single population) and only change **where** heavy local search and cost evaluation are executed (CPU vs GPU), in a Type 1 spirit.
  - Distinguish your work from island or fine‑grained GAs (you’re not exploring subpopulation/migration design).

---

### 7. Simulated Annealing (Sec. 5, pp. 492–496)

They describe:

- **Single‑trial SA parallelization (Type 1)**:
  - Parallelize cost evaluation for **one move** (functional or data parallelism).
  - Preserves SA convergence theory, but degree of parallelism is limited.

- **Multiple‑trial SA**:
  - Evaluate several candidate moves in parallel, but replacement step is sequential.
  - To preserve “error‑free” cost evaluation, they restrict to serializable subsets or choose exactly one accepted move.

- **Type 2 SA**:
  - Partition variables, each slave runs SA on subset; master synchronizes and merges partial configurations.
  - Issues: maintaining correctness, controlling accumulated error, locking when variables overlap, cost of global updates.

- **Type 3 SA**:
  - Division strategy: splitting the Markov chain iterations over processors.
  - MIR/MMC: multiple independent Markov chains.
  - Population‑based SA and SA+GA hybrids (SA threads with GA‑style selection/migration).

Not directly central to your implementation, but:

- This section is a **reference model** for how to discuss SA parallelization in a compact but rigorous way, if you decide to compare with SA or mention SA literature in related work.

---

### 8. Tabu Search (Sec. 6, pp. 496–501)

Tabu search is portrayed as a very fertile playground for parallelization; they survey:

- **Type 1 TS**:
  - Parallel neighborhood exploration (master–slave).
  - Very effective when:
    - Huge neighborhood.
    - Cheap individual move evaluations.
  - Examples: QAP, TSP, VRP.

- **Type 2 TS**:
  - Decompose decision variables or tours (e.g. partition a TSP tour, VRP routes).
  - Sometimes used as part of adaptive memory strategies.

- **Type 3 TS**:
  - Independent multiple tabu searches (multi‑start).
  - Synchronous cooperative TS:
    - Exchange best solutions periodically.
  - Asynchronous cooperative TS with **central / adaptive memory**:
    - Shared pool of high‑quality (partial or complete) solutions.
    - Multiple threads asynchronously take and contribute solutions.
    - Concept of **adaptive memory** and restart from recombined building blocks.
  - Hybrid TS with GA:
    - GA uses solutions from TS pool; TS refines GA offspring; mutual reinforcement.

Key conclusions:

- **Asynchronous cooperative multi‑thread TS** (Type 3 with central/adaptive memory) often beats:
  - Sequential TS.
  - Synchronous cooperation.
  - Pure independent multi‑start.

Useful for TCC:

- Conceptual template for **adaptive memory** and **central pool** if you later discuss possible extensions (multi‑start GA+2‑opt with shared elite tours).
- Good source for **terminology** and citations about cooperative multi‑thread metaheuristics (even if you don’t implement them now).

---

### 9. Other Metaheuristics and Perspectives (Sec. 7, pp. 501–504)

They briefly discuss:

- Other methods: scatter search, GRASP, VNS, ant colony optimization, swarm methods.
- The same three‑type classification and design principles apply.
- Emphasis on:
  - **Hybridization** (mixing metaheuristics, e.g., GA+TS, SA+GA).
  - **Multi‑thread cooperative search** as a **new metaheuristic class**, not just “parallel versions” of existing ones.
  - Open research questions:
    - How to design cooperation (what to exchange, when, how to integrate it)?
    - How to combine global (shared) and local (per‑thread) memory?
    - How to compare such algorithms rigorously?

Useful for TCC:

- In conclusion and related work, you can:
  - Place your GPU GA+2‑opt as **focusing on low‑level data parallelism** (Type 1) and not yet on these “new” cooperative schemes.
  - Argue that **future work** could explore Type 3 cooperative parallelism on top of your GPU kernels (e.g., multiple GA populations running in parallel, sharing elite tours).

---

### 10. How This Supports Your Thesis

In short, this paper gives you:

1. **Canonical definitions**  
   - What “metaheuristics” are.  
   - How to structure a generic metaheuristic algorithm.  
   - How to talk about local vs global phases, neighborhood, acceptance rules.

2. **A well‑cited taxonomy of parallel metaheuristics**  
   - Types 1, 2, 3 → direct hook for classifying your **CPU vs GPU GA variants**.
   - You can say, for example:
     - CPU GA and GA+GPU variants are all **Type 1** strategies (low‑level parallelism), differing only in *where* cost and local search are computed.
     - You are **not** altering the higher‑level multi‑thread or cooperative structure (Type 3).

3. **Language to discuss robustness and performance**  
   - They stress that parallel metaheuristics target both **speed** and **robustness**.
   - This aligns perfectly with your **iso‑algorithmic GA variants** and statistical comparison goals.

4. **Hooks for future work**  
   - Cooperative multi‑thread/metaheuristics with central memory and adaptive memory are natural **extensions** of your GA+2‑opt framework once you move beyond a single population/single run.

If you’d like, next we can:

- Draft a short subsection for Chapter 2 that:
  - Introduces the three parallelization types from this paper.
  - Explicitly classifies your four GA variants (CPU, HybridNaive, HybridOptimized, FullGPU) under this taxonomy.
  - Adds Pandoc citations using the correct BibTeX key for this paper.
