Ran `read-pdf`
Completed with input: {
  "file": "/home/lucas_galdino/chimera/gpu_accelerated/documentation/pdfs/William J. Cook - In Pursuit of the Traveling Salesman_Mathematics at the Limits of Computation (2011, Princeton University Press) - libgen.li.pdf",
  "pages": "all",
  "include_metadata": true,
  "clean_text": true
}

Summarized conversation historyI’ll distill Cook’s book with a TCC‑oriented lens and show exactly where it plugs into your chapters.

---

## 1. Reference and scope

**Full reference (already in your `refs.bib` as `@cook2012pursuit`):**

> Cook, W. J. *In Pursuit of the Traveling Salesman: Mathematics at the Limits of Computation.* Princeton University Press, 2012.

What it is:

- Semi‑popular but technically sound monograph on the TSP.
- Focus: history, applications, algorithms (from 2‑opt to cutting planes), and **complexity (P vs NP)**.
- Ideal as a **“umbrella” citation** whenever you need:
  - Historical context for TSP.
  - Statements like “TSP is NP‑difícil/NP‑complete and central in combinatorial optimization.”
  - Examples of massive exact computations (85 900 cities, World TSP, Mona Lisa TSP).
  - Motivation for using TSP as a *laboratório conceitual*.

---

## 2. High‑level structure of the book

You don’t need all details, but the chapter map is useful when you write “como discutido em Cook [@cook2012pursuit, cap. X]”:

1. **Challenges** – Procter & Gamble’s Car 54 contest, 48‑states problem, factorial explosion, Roadrunner supercomputer illustration, and first view of **P vs NP** and the Clay prize.
2. **Origins of the Problem** – Pre‑mathematical TSP (real salesmen, Commis‑Voyageur), Euler’s bridges, Hamilton’s Icosian game, Menger’s messenger problem, Whitney/Flood/Dantzig at RAND.
3. **The Salesman in Action** – Applications: logistics, genome mapping, telescopes, circuit‑board drilling, VLSI, microprocessor tests, clustering.
4. **Searching for a Tour** – Heuristics:
   - Nearest neighbor, insertion, **2‑opt, 3‑opt, Lin–Kernighan**, simulated annealing, genetic algorithms, ant colony, etc.
5. **Linear Programming** – Introduction to LP, simplex, degree relaxation, subtour constraints.
6. **Cutting Planes** – Cutting‑plane method, polyhedral view of TSP.
7. **Branching** – Branch‑and‑bound / branch‑and‑cut.
8. **Big Computing** – History of record‑size exact TSPs (from 49 to **85 900 cities**), Concorde, TSPLIB, VLSI and image instances.
9. **Complexity** – Formal P/NP, Cook’s theorem, Karp’s list, Held–Karp DP \(O(n^2 2^n)\), approximations (Christofides 1.5, Arora PTAS), NP‑completeness, Euclidean vs metric TSP.
10. **The Human Touch** – Humans/animals solving small TSPs; cognitive aspects.
11. **Aesthetics** – TSP tours in art (Lethbridge, Bosch, Kaplan, Galanter, Mo).
12. **Pushing the Limits** – Open problems and TSP as an “engine of discovery”.

---

## 3. Core themes that matter for your thesis

### 3.1 TSP as central NP‑hard reference problem

Cook frames TSP as:

- A **canonical NP‑hard problem**:
  - “Despite decades of research by top applied mathematicians…, in general it is not known how to significantly improve upon simple brute‑force checking.” (ch. 1)
  - TSP is **NP‑complete** via Karp’s list (ch. 9).

>[!important] citar examples de aplicaçãao no capítulo 1
>
>- A **testbed for optimization algorithms**:
> - “One of the enduring strengths of the traveling salesman problem has been its remarkable success as an **engine of discovery** in applied mathematics and its fields of operations research and mathematical programming.” (preface/ch. 1).
> - Exact record: 85 900‑city chip instance solved with branch‑and‑cut (ch. 8), after ~136 CPU‑years equivalent.

**How you can use this:**

- Cap. 1 – “Contexto e motivação”:
  - Justify TSP as *problema modelo* for new heuristics/meta‑heurísticas and for CPU vs GPU experiments:
    - Cite `[@cook2012pursuit]` when you say that TSP has been used “por décadas como laboratório conceitual para novos algoritmos de otimização combinatória e meta‑heurística”.
- Cap. 2 – “Fundamentação teórica”:
  - When formally stating NP‑dificuldade / NP‑completude do TSP and its role in complexity theory, you can refer to Cook’s accessible discussion (ch. 1 and 9) and then cite the classical sources separately (Cook 1971, Karp 1972 from `refs.bib`).

---

### 3.2 Historical narrative you can borrow

Cook gives you a clean, readable history that you can compress into 1–2 paragraphs in Cap. 2:

- **Pre‑TSP practice** (ch. 2):
  - 19th/early 20th century **traveling salesmen** planning efficient tours (Page Seed Company itineraries, 1925).
  - 1832 book *Der Handlungsreisende* (Commis‑Voyageur) already describing “visit as many locations as possible without touching them twice”.
- **Whitney → Flood → RAND**:
  - Hassler Whitney’s seminar at Princeton (ca. 1934) describing a “travel through all countries once” problem.
  - Merrill Flood’s recollections of Whitney’s 48‑states problem and the RAND prize list (late 1940s).
  - Julia Robinson’s early work formulating the “travel from Washington visiting all state capitals and returning” problem.

>[!important] adicionar ao capítulo 2.
>
> - **Early mathematical roots**:
>   - Euler’s Königsberg bridges → graph theory, Eulerian walks.
>   - Hamilton’s Icosian game → Hamiltonian circuits (tours through graph vertices).
> - **First TSP‑like formulations**:
>   - Menger’s 1930 “messenger problem” (find shortest path through a finite set of points with known pairwise distances).
>   - Mahalanobis (1940s) – statistical view: expected length of random TSP tours in sample surveys on jute farms (India).
> - **Dantzig–Fulkerson–Johnson 48‑state problem** (Newsweek 1954; ch. 1 & 2):
>   - Solved with **linear programming + cutting planes**, tour length 12 345 miles.
>   - This is the **foundational exact TSP computation**.

**How you can use this:**

- Cap. 2 – Section “Histórico do TSP”:
  - Use Cook as your primary narrative source; you don’t need to chase each historical original.
  - A short paragraph can go:
    - Pre‑history (salesmen, Commis‑Voyageur).
    - Graph‑theoretic roots (Euler, Hamilton).
    - Early “messenger/sample survey” formulations (Menger, Mahalanobis).
    - Modern TSP emergence with Whitney/Flood and the RAND era.
    - First large exact solution via Dantzig–Fulkerson–Johnson.
  - Cite `[@cook2012pursuit]` for the entire historical arc; optionally add individual original refs you already have in `refs.bib` if you want to impress the examiner.

---

### 3.3 Applications relevant for your motivation

In **ch. 3 – The Salesman in Action**, Cook lists many TSP applications; the most relevant to your Cap. 1:

- **Logística / roteirização**:
  - Classical route planning (road trips, 48‑states tour, Car 54 contest).
- **Genômica**:
  - Mapping genomes via ordering problems that reduce to TSP: radiation hybrid mapping, etc.
- **Astronomia**:
  - Aiming telescopes / X‑ray / laser equipment by minimizing slewing time between targets.
- **Manufatura & VLSI**:
  - PCB drilling paths.
  - Laser cutting in semiconductor manufacturing (the 85 900‑city chip instance).
- **Testes e organização de dados**:
  - Microprocessor test pattern ordering.
  - Clustering and data ordering.

**How you can use this:**

- Cap. 1 – “Contexto e motivação”:
  - When you say “variantes do TSP aparecem em roteirização, inspeção, manufatura, robótica …”, you can explicitly ground that with `[@cook2012pursuit, cap. 3]`.
- Cap. 5/6 – Discussion:
  - To argue external relevance of GPU speedups in TSP heuristics (e.g. for PCB drilling or VLSI), you can quote Cook’s remark that the 85 900‑city circuit instance directly impacted production cost (laser travel time).

---

### 3.4 Algorithms: heuristics, 2‑opt, metaheuristics, and exact methods

This is where Cook’s exposition lines up very directly with your **GA+2‑opt** framework.

**Heuristic and local‑search side (ch. 4):**

- **2‑opt / 3‑opt**:
  - Introduces edge‑exchange heuristics, 2‑opt and 3‑opt, as classic improvement operators.
  - Emphasizes that 2‑opt is simple but powerful and forms the basis for more advanced methods (Lin–Kernighan).
- **Lin–Kernighan**:
  - Shows how iterated variable‑k exchanges (starting from 2‑opt) lead to state‑of‑the‑art tours.
- **Metaheuristics**:
  - Simulated annealing: early applications to TSP.
  - Genetic algorithms: GA with crossover and mutation applied to permutations of cities.
  - Ant‑colony optimization.

**Exact / polyhedral side (ch. 5–7):**

- **Linear Programming**:
  - Introduces LP, simplex, degree constraints, subtour elimination (degree LP relaxation → subtour LP).
  - LP duality, polytope viewpoint.
- **Cutting Planes**:
  - General cutting‑plane method, TSP‑specific inequalities (subtours, combs, blossoms, etc.).
  - Separation problem, Edmonds’s polyhedral perspective.
- **Branch‑and‑Bound / Branch‑and‑Cut**:
  - Systematic enumeration + bounding; how LP relaxations + cuts drive pruning.

**Big computations (ch. 8):**

- History of record‑breaking exact TSPs culminating in:
  - **3038‑city**, **13 509‑city US tour**, **24 978‑city Sweden**, and **85 900‑city VLSI** instance.
  - All done by **Concorde**, built on branch‑and‑cut + sophisticated separation algorithms.

**Complexity and approximations (ch. 9):**

- Held–Karp DP $O(n^2 2^n)$ as best known general exact worst‑case bound.
- Christofides’s 1.5‑approximation (metric TSP).
- Arora/Mitchell PTAS for Euclidean TSP; practical limitations.
- Hardness of improving Christofides below ~1.0045 unless \(P = NP\).

**How you can use this:**

- Cap. 2 – “Algoritmos e meta‑heurísticas para TSP”:
  - When you introduce **2‑opt** as your local search inside GA, Cook gives you a **canonical, didactic reference** showing:
    - 2‑opt → building block of serious TSP heuristics.
    - Lin–Kernighan as the natural extension.
  - Cite `[@cook2012pursuit, cap. 4]` when you motivate 2‑opt and mention its role in the literature.
- Cap. 2 or 3 – “Metaheurísticas”:
  - Short paragraph summarizing GA, SA, ACO as standard metaheuristics for TSP can safely lean on Cook (plus more specialized GA papers you already have).
- Cap. 3 – “Métodos exatos e benchmarks”:
  - For the polyhedral / branch‑and‑cut background that justifies using TSP as a **gold‑standard benchmark**, cite `[@cook2012pursuit, caps. 5–8]`.

---

### 3.5 Complexity, P vs NP, and why TSP is “hard but solvable in practice”

Cook’s **ch. 1 and 9** are almost tailor‑made for a short complexity subsection in your thesis:

- **Edmonds’s notion of “good algorithms”**:
  - Polynomial‑time algorithms vs exponential like \(2^n\) and \(n!\).
- **Classes P and NP**:
  - P: problems solvable in polynomial time.
  - NP: problems whose *solutions* can be verified in polynomial time (TSP decision version included).
- **NP‑completeness**:
  - Cook’s theorem (satisfiability is NP‑complete).
  - Karp’s 21 NP‑complete problems, including TSP/Hamiltonian circuit.
- **Held–Karp algorithm**:
  - Exact DP with running time \(O(n^2 2^n)\); best known general bound today.
- **Approximations and hardness**:
  - Christofides 1.5 bound for metric TSP.
  - Lower bounds on possible approximation factors under \(P \neq NP\).
- **Euclidean subtleties**:
  - Sum‑of‑square‑roots problem and why Euclidean TSP is not trivially in NP (but has PTAS).

**How you can use this:**

- Cap. 2 – Subsection “Complexidade do TSP”:
  - A concise narrative can be anchored on Cook’s explanation:
    - TSP is **NP‑completo** (decision version).
    - Best known generic exact method is **Held–Karp DP** \(O(n^2 2^n)\).
    - Despite NP‑completude, large instances are routinely solved with **branch‑and‑cut**.
  - Cite `[@cook2012pursuit, caps. 1 e 9]` for an accessible, book‑level reference, and separately cite Cook 1971 / Karp 1972 for the original theorems if you wish.

---

### 3.6 Large‑scale benchmarks: TSPLIB, Mona Lisa, World TSP, Star TSP

In **ch. 8**, Cook describes the “big league” instances that your GA+GPU framework should conceptually hook into:

- **TSPLIB**:
  - Public collection of TSP/ATSP instances used by the community.
  - Many record computations (including the 13 509‑city US tour) are based on TSPLIB‑style instances.
- **Mona Lisa TSP (100 000 cities)**:
  - Source: Bosch’s TSP art.
  - Best known tour by Yuichi Nagata; bound established by Concorde, leaving a tiny but non‑zero gap.
- **World TSP (1 904 711 cities)**:
  - Every city/town/village in the world + Antarctica research bases.
  - Best tour from Keld Helsgaun; lower bound from Concorde; current gap ~0.0476%.
- **Star TSP (526 280 881 celestial objects)**:
  - TSP on celestial coordinates, modelling telescope movement.

**How you can use this:**

- Cap. 1 – Motivation:
  - When arguing that TSP is used as a benchmark for **very large‑scale** computations, you can point to:
    - TSPLIB.
    - Mona Lisa and World TSP.
  - Cite `[@cook2012pursuit, cap. 8]`.
- Cap. 4 – Resultados:
  - If you scale your instances (e.g. up to some thousands of cities), you can contextualize:
    - “Embora os experimentos deste trabalho utilizem instâncias de até \(n_{coords} \approx X\), a literatura de TSP inclui problemas com até milhões de cidades [@cook2012pursuit].”

---

## 4. Practical “how to cite Cook” cheatsheet

Here are ready‑made “hooks” you can use directly in your text:

1. **TSP as classical, NP‑difícil reference problem and engine of discovery**  
   - Use in Cap. 1/2:
     - “O TSP é um dos problemas de referência em otimização combinatória e está no núcleo da discussão sobre os limites da computação eficiente [@cook2012pursuit].”
2. **Historical sketch: from Commis‑Voyageur and salesmen to RAND and DFJ**  
   - Use in Cap. 2 (histórico):
     - “Uma síntese acessível desse percurso histórico, incluindo o problema dos 48 estados resolvido por Dantzig, Fulkerson e Johnson, é apresentada em [@cook2012pursuit].”
3. **Applications**  
   - Use in Cap. 1 (motivação):
     - “Cook [@cook2012pursuit] ilustra aplicações do TSP em logística, genômica, astronomia, manufatura e testes de microprocessadores, reforçando seu papel como modelo abstrato para problemas de roteamento.”
4. **Heuristics and 2‑opt / Lin–Kernighan**  
   - Use in Cap. 2/3 (descrição do GA+2‑opt):
     - “Heurísticas de melhoria baseadas em 2‑opt e seus desdobramentos, como Lin–Kernighan, formam a base dos métodos mais eficazes para TSP [@cook2012pursuit, cap. 4].”
5. **Exact methods & big computations (Concorde, 85 900‑city chip)**  
   - Use in Cap. 2/3 (fundamentação) and Cap. 5 (discussão):
     - “Cook [@cook2012pursuit] documenta como algoritmos de branch‑and‑cut conseguiram resolver instâncias com até 85 900 cidades, em particular um problema real de roteamento de feixes de laser em fabricação de chips.”
6. **Complexity, P vs NP, NP‑completeness of TSP**  
   - Use in Cap. 2:
     - “Para uma discussão acessível sobre a NP‑completude do TSP e sua relação com a questão \(P\) versus \(NP\), ver [@cook2012pursuit, cap. 9].”
7. **Large‑scale artistic and geographic instances (Mona Lisa, World TSP)**  
   - Use in Cap. 1 or Cap. 5:
     - “Instâncias artísticas (como a Mona Lisa com 100 000 pontos) e geográficas de escala planetária (World TSP, com 1,9 milhões de cidades) são descritas em [@cook2012pursuit], destacando o papel do TSP como benchmark extremo.”

---

If you’d like, next step I can do is: draft **one clean paragraph** for your Cap. 1 “Contexto e motivação” that replaces the current `>[!caution]` under `[@cook2012pursuit]` using exactly these Cook insights, already in Pandoc‑ready Portuguese.
