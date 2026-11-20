---
title: "Why the CPU 2-opt vectorization is ~20× faster: cost model and exactness arguments"
description: >
  Write-up of the CPU local-search vectorization (`TASK 13.14`, `DP13-5`): the
  interpreter-overhead cost model that explains an ~20× gain on a conventional x86
  core, the pair-set arithmetic behind $n(n-3)/2$, the edge-vector identity that
  halves the matrix gathers, why `argmin` and a stable `argsort` reproduce the
  sequential scan bit for bit, and the `int64` widening rationale. Also records the
  audit of explicit dtype casts (which are load-bearing) and the deliberate
  `xp: Any` at the selection site.
created: 2026-11-16
status: review
author:
  - "Lucas Galdino"
type: analysis
scope: local
modifications:
  - date_modified: 2026-11-17
    modifications:
      - description: >
          Added the missing YAML frontmatter and rewrote the file, which was a
          first-person chat reply (answers addressed to a reviewer, plus an
          "I cannot write files from ask mode" section). Converted to an impersonal
          technical note: dropped the meta-instructions, fixed the
          `torunament_selection.py` typo, replaced line anchors with symbol
          references and stated the consequence of a ~18.6× faster CPU baseline for
          the reported speedups.
related_files:
  - "[two_opt.py](../../../src/algorithms/strategies/local_search/two_opt.py)"
  - "[matrix.py](../../../src/distances/matrix.py)"
  - "[problem_context.py](../../../src/protocols/problem_context.py)"
  - "[backend.py](../../../src/protocols/backend.py)"
  - "[test_unit_two_opt_vectorized_scan.py](../../../tests/algorithms/strategies/local_search/test_unit_two_opt_vectorized_scan.py)"
tags:
  - analysis/algorithm
  - algorithm/local-search
  - algorithm/tsp
  - notes/performance
  - analysis/benchmark
  - guide/python
  - guide/testing
  - analysis/architecture
  - review/quality
  - notes/planning
---

# Why the CPU 2-opt vectorization is ~20× faster

This note records the reasoning behind the CPU local-search vectorization (`TASK 13.14`, `DP13-5`): the scalar double loop in `LocalSearchCPUTwoOpt` was replaced by a vectorized pair enumeration shared by the three acceptance rules — [`two_opt.py::_exhaustive_pairs`](../../../src/algorithms/strategies/local_search/two_opt.py), [`two_opt.py::_pair_blocks`](../../../src/algorithms/strategies/local_search/two_opt.py) and [`two_opt.py::_pair_deltas`](../../../src/algorithms/strategies/local_search/two_opt.py).
The change is exact: 3,510 parity checks against a literal reconstruction of the scalar scan report zero divergences. What follows explains the measured speedup, the arithmetic of the pair set and the two exactness arguments (tie-breaking and width).

## 1. Policy note: `xp: Any` at the selection site is deliberate

In `tournament_selection.py` the accepted type of `xp` is `Any`, and that is forced by repository policy rather than being an oversight: `require_gpu()` returns `Any`, and the typing layer declares `type CupyArray = Any` in [backend.py](../../../src/protocols/backend.py) because the CuPy stubs are incomplete.
An expression such as `cp if is_gpu else np` is therefore irreducibly `Any`.

Consequence for the backend protocol: `float64` is a genuine member of `BackendModule` — `np.float64` and `cp.float64` both exist, and it receives the same treatment already given to `int32`, `int64`, `abs`, `trunc` and `sqrt`.
The declaration is exercised under a *typed* `xp` in `matrix.py::_xray_block`, which is precisely what a protocol is for.

Tightening `xp` at that call site is not worth doing: it would require `cast("BackendModule", cp)`, and the moment `xp.asarray` stops returning `Any`, every downstream assignment in the function (`fit_arr`, `candidates`, …) turns into a typing fight over `Array | ndarray`.

## 2. Audit of explicit dtype casts

The convention "an explicit cast selects the active backend's dtype" is real and used.
The audit separates the casts that carry that meaning from those that do not:

| site | cast | load-bearing? |
| --- | --- | --- |
| `tournament_selection.py` (fitness promotion) | `xp.asarray(fitness, dtype=xp.float64)` | **yes** — fitness arrives as `int32` (tour costs are integers) and must be promoted for float arithmetic |
| `base.py` (warm-up permutations) | `xp.arange(..., dtype=xp.int32)` | **yes** — `arange` defaults to the platform int |
| `matrix.py` (matrix build, candidate list, XRAY bounds) | `…astype(xp.int32/int64)` | **yes** — the output must be integer |
| `matrix.py::_xray_block` | `rows.astype(xp.float64)` | **no** |

The last row is the only non-load-bearing one, and it can be proven rather than assumed.
Coordinates are `float64` by contract on **both** backends:

+ `tsplib_loader.py` — `np.asarray(coords_list, dtype=np.float64)`;
+ `problem_context.py` — `self._gpu_coordinates = cp.asarray(source)` with no dtype override, so it inherits that `float64`;
+ `problem_context.py::estimate_vram_bytes` — the memory budget documents the coordinate buffer as `8 * d * n`, **`float64`**.

So the cast is a no-op copy: 512 rows × 3 floats ≈ 12 KB per block, against a 512 × n broadcast — microseconds. Dropping the two cast lines is safe and keeps the dtype contract documented in the docstring, with `xp.float64` still justified by the selection path.
Keeping them is also defensible; the point of the audit is that they do not select a dtype.

## 3. Cost model and exactness arguments

**Setup.** Tour $\pi = (\pi_0, \dots, \pi_{n-1})$, integer distance matrix $D \in \mathbb{Z}^{n \times n}_{\ge 0}$, indices taken mod $n$.

### The move and its gain

A 2-opt move with break points $(i,j)$ reverses the segment $\pi_{i+1 \dots j}$. Writing $a=\pi_i,\ b=\pi_{i+1},\ c=\pi_j,\ d=\pi_{j+1}$, the two edges $(a,b),(c,d)$ are replaced by $(a,c),(b,d)$:

$$\Delta_{ij} \;=\; D\lbrack a,c\rbrack + D\lbrack b,d\rbrack - \underbrace{D\lbrack a,b\rbrack - D\lbrack c,d\rbrack}_{\text{edges being retired}}$$

$\Delta_{ij} < 0$ means the reversal shortens the tour. A sequential search evaluates $\Theta(n^2)$ of these per pass.

### Why the gain is ~20×, not 4×

This is the central question, because it looks like it should be 4× or 8×.

Let $P$ be the number of pairs evaluated, $\tau_{\text{py}}$ the cost of one interpreted element, $\tau_{\text{mem}}$ one dependent memory access in compiled code, and $\tau_{\text{call}}$ the fixed overhead of one NumPy call.

$$T_{\text{seq}} = P \cdot \tau_{\text{py}}$$

$$T_{\text{vec}} = \frac{P}{B}\tau_{\text{call}} + P\left(\tau_{\text{mem}} + \tau_{\text{alu}}\right)$$

With block size $B$ large, the call overhead vanishes:

$$S \;=\; \frac{T_{\text{seq}}}{T_{\text{vec}}} \;\overset{B \gg 1}{\longrightarrow}\; \frac{\tau_{\text{py}}}{\tau_{\text{mem}} + \tau_{\text{alu}}}$$

Substituting the measured numbers for $n=1002$, $k=20$ (10,053 pairs per pass, 10 passes):

$$\tau_{\text{py}} = \frac{281.5\ \text{ms}}{100{,}530} = 2.80\ \mu s, \qquad \tau_{\text{mem}}+\tau_{\text{alu}} = \frac{12.9\ \text{ms}}{100{,}530} = 128\ \text{ns}$$

$$S = \frac{2.80}{0.128} \approx 21.9$$

which is the measurement. **The speedup equals the ratio of interpreter overhead to memory latency, not the SIMD width.** $\tau_{\text{alu}}$ already contains whatever the lanes buy (a factor $\le 16$); it is simply not the term that dominates, because $\tau_{\text{py}}$ is inflated by ~25 bytecode operations per element — `int()` conversions, four subscriptions, tuple arithmetic, comparison, assignment.

That also predicts the ceiling: once $\tau_{\text{py}}$ is gone you are left paying for random gathers into a 4 MB matrix, and no amount of vectorization removes those.
That is why the curve flattens near 20× and not 100×.

### The pair set

$$\mathcal{P} = \{(i,j) \;:\; 0 \le i \le n-3,\; i+2 \le j \le n-1\} \setminus \{(0,\,n-1)\}$$

$$|\mathcal{P}| = \frac{(n-1)(n-2)}{2} - 1 = \frac{n(n-3)}{2}$$

For $n=1002$: $1002 \cdot 999/2 = 500{,}499$. The exclusion is structural, not cosmetic — for $i=0,\,j=n-1$ the reversal is $\pi_{1\dots n-1}$, which yields the same edge set read backwards. Keeping it would evaluate $\Delta = 0$ always and waste a full row.

`np.triu_indices(n, k=2)` produces exactly $\mathcal{P} \cup \{(0,n-1)\}$ in row-major order, so scanning the arrays left to right *is* the sequential visit order.

In candidate mode the set becomes $\mathcal{P}_k$, with $\nu \in N_k(\pi_i)$ mapped through $\mathrm{pos}(\cdot)$:

$$\mathcal{P}_k = \bigl\{(i,\, \mathrm{pos}(\nu)) : i \in \lbrack 0,n-3\rbrack,\ \nu \in N_k(\pi_i),\ i+1 < \mathrm{pos}(\nu) < \ell_i\bigr\}, \quad \ell_i = \begin{cases} n-1 & i = 0 \\ n & i > 0\end{cases}$$

of size $O(nk)$ rather than $O(n^2)$ — which is why the benchmark path costs 10k pairs per pass, not 500k.

### Collapsing four gathers into two

$\Delta$ needs $D\lbrack a,c\rbrack$ and $D\lbrack b,d\rbrack$ — genuinely random, unavoidable. But $D\lbrack a,b\rbrack$ and $D\lbrack c,d\rbrack$ are *consecutive* edges of the current tour.
Define the edge vector

$$e_k = D\bigl\lbrack\pi_k,\ \pi_{(k+1)\bmod n}\bigr\rbrack, \qquad k = 0,\dots,n-1$$

Then $D\lbrack a,b\rbrack = e_i$ and $D\lbrack c,d\rbrack = e_j$, giving

$$\boxed{\;\Delta_{ij} = D\lbrack\pi_i,\pi_j\rbrack + D\lbrack\pi_{i+1},\pi_{j+1}\rbrack - e_i - e_j\;}$$

$e$ is $n$ int32 = 4 KB for $n=1002$ — it sits in L1.
So the random-access count per pass drops from $4P$ to $2P$. In the cost model above, that halves $\tau_{\text{mem}}$, which directly raises $S$.

### Why `argmin` and `argsort(stable)` are bit-exact

The sequential rule is `if Δ < best` — **strictly** less.
Therefore among equal minima the *first in scan order* wins:

$$\text{winner} = \arg\min_{p \in \mathcal{P}}\ \Delta_p \quad\text{with ties} \to \text{first occurrence}$$

`np.argmin` returns the first occurrence of the minimum. Since `triu_indices` emits scan order and blocks are consumed in order with a strict `<` across them, the composition is exactly the sequential answer.

For `multi_move`, python's `list.sort` is stable (Timsort), so equal $\Delta$ retain insertion order.
`np.argsort(..., kind="stable")` is mergesort with the same guarantee.
Hence the greedy pass receives the identical sequence.

Disjointness is the overlap predicate, and the two formulations are equivalent:

$$\text{conflict}(\lbrack i,j\rbrack, \lbrack s,e\rbrack) \;\iff\; \neg\,(j < s \ \lor\ i > e) \;\iff\; \bigl\lvert\{i,\dots,j\} \cap \{s,\dots,e\}\bigr\rvert > 0$$

The right-hand form is what a boolean `occupied` vector tests, in $O(j-i)$ instead of $O(\#\text{applied})$.

### The `int64` widening

$$\Delta_{ij} \le 2 \max D, \qquad \text{overflow in int32} \iff 2\max D > 2^{31}-1 \iff \max D > 1.07\times10^{9}$$

None of the 15 instances in the cohort comes close — $\max D$ there is of order $10^5$, so int32 would not wrap. python's arbitrary-precision ints never had the problem, so the vectorized version would introduce a *new* failure mode that the scalar reference could not exhibit.
Widening before the addition costs 2× bytes on temporaries, which the profile shows is not the bottleneck: it is insurance, not a requirement.

### XRAY1, the other loop

$$\phi_s = \min\bigl(\phi,\ |\phi - 360|\bigr), \qquad \text{cost} = \max\!\left(\frac{\phi_s}{s_x},\ \frac{|\Delta\chi|}{s_y},\ \frac{|\Delta\theta|}{s_z}\right), \qquad d = \operatorname{trunc}(100\,\text{cost} + 0.5)$$

Every operation is elementwise, so it is one broadcast pair `(m,1) × (1,n)` instead of $m \cdot n$ python calls.
Verified on 3216 cells against the scalar `converter.utils.cost.xray1`, including negative coordinates and values above 360.

### Intuition

The old scan compared every pair of cities one at a time, paying a python statement per pair.
The vectorized version hands NumPy the whole list of pairs and asks "which gain is smallest?"; NumPy answers in compiled machine code.
The result is identical every time and roughly twenty times quicker, because the time was never spent on arithmetic — it was spent on interpretation.

### Consequences for the reported numbers

Local search accounts for 99.6% of CPU wall time and is now ~20× faster, so the CPU column of every table falls by roughly 18.6×.
Every CPU/GPU speedup ratio divides by about that factor: a previously reported 200× becomes ~11×. In other words, the "200–600×" figures measured CPython as much as they measured the serial-versus-parallel distinction (`BUG-63`); the recoleta of the CPU time columns remains pending under `TASK 12.4`.
