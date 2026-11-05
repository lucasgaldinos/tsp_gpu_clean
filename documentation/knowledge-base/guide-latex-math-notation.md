---
title: "Guide: Mathematical Notation and LaTeX Cheat Sheet for Markdown"
description: >
  Comprehensive reference guide and cheat sheet for authoring mathematical notation
  in GitHub Markdown and Pandoc LaTeX, including arithmetic, calculus, linear algebra,
  probability, Greek symbols, matrices, and typesetting best practices.
created: 2026-11-05
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Consolidated math_notation_github_style.md and latex_cheat_sheet.md into
          a unified, standardized guide with full frontmatter and table formatting.
      - description: >
          Replaced unapproved latex code fence with text per repository markdown guidelines.
related_files:
  - "[CheatSheet.tex](./tex-docs/CheatSheet.tex)"
  - "[Template.tex](./tex-docs/Template.tex)"
tags:
  - guide/latex
  - guide/math
  - guide/markdown
  - notes/latex
  - notes/math
  - notes/cheat_sheet
  - analysis/typesetting
  - guide/documentation
  - notes/github_markdown
  - notes/katex
---

# Guide: Mathematical Notation and LaTeX Cheat Sheet for Markdown

A reference guide and cheat sheet on writing mathematical notation and formulas in GitHub Markdown and LaTeX/Pandoc documents.

## 1. Fundamentals of Math in Markdown

There are two primary ways to include mathematical notation in Markdown:

1. **Inline Math**: Wrapped in single dollar signs (`$...$`). Notation flows within the sentence or paragraph (e.g., `$x^2$` renders as $x^2$).
2. **Display / Block Math**: Wrapped in double dollar signs (`$$...$$`). Formatted centered on its own paragraph.

### Core Syntax Rules

- **Escaping Dollar Signs**: Use a backslash `\$` when you need a dollar sign in prose without triggering math parsing.
- **Common Operators & Functions**: Standard arithmetic operators (`+`, `-`, `*`, `/`, `=`) and standard functions (`\sin`, `\cos`, `\tan`, `\log`, `\ln`, `\exp`, `\max`, `\min`).
- **Subscripts and Superscripts**:
  - Subscript: Underscore `_` (e.g., `x_i`, `x_{i+1}`).
  - Superscript: Caret `^` (e.g., `x^2`, `x^{2k}`).
- **Fractions**: `\frac{numerator}{denominator}` (e.g., `\frac{a}{b}`).
- **Roots**:
  - Square root: `\sqrt{x}`
  - $n$-th root: `\sqrt[n]{x}`
- **Brackets and Parentheses**:
  - Use `\left(` and `\right)` (or `\left[`, `\right]`, `\left\{`, `\right\}`) to dynamically scale delimiters to fit the enclosed expression:
    $$\left( \frac{a}{b} \right)$$
- **Matrices & Vectors**:
  - Use environments like `pmatrix` (parentheses), `bmatrix` (square brackets), or `vmatrix` (determinants) inside `\begin{...}` and `\end{...}`:
    $$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

```text
\sum_{k=3}^5 k^2 = 3^2 + 4^2 + 5^2 = 50 \\
\frac{a}{b}
```

---

## 2. LaTeX Math Reference Tables

> [!TIP]
> Wide tables can be scrolled horizontally on compact displays.

### 2.1 Arithmetic

| Notation | Description | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| Addition | $a + b$ | `$a+b$` | `$$a+b$$` |
| Subtraction | $a - b$ | `$a-b$` | `$$a-b$$` |
| Multiplication | $a \times b$, $a \ast b$, $a \cdot b$ | `$a \times b$`, `$a \cdot b$` | `$$a \cdot b$$` |
| Division | $a / b$, $a \div b$, $\frac{a}{b}$ | `$a / b$`, `$a \div b$`, `$\frac{a}{b}$` | `$$\frac{a}{b}$$` |
| Modulo / Remainder | $5 \bmod 2 = 1$ | `$5 \bmod 2 = 1$` | `$$5 \bmod 2 = 1$$` |
| Negative Value | $-a$ | `$-a$` | `$$-a$$` |
| Plus/Minus, Minus/Plus | $\pm a$, $\mp a$ | `$\pm a$`, `$\mp a$` | `$$\pm a$$` |
| Exponents | $a^2$, $a^3$, $a^n$ | `$a^2$`, `$a^3$`, `$a^n$` | `$$a^n$$` |
| Roots | $\sqrt{a}$, $\sqrt[3]{a}$, $\sqrt[n]{a}$ | `$\sqrt{a}$`, `$\sqrt[3]{a}$`, `$\sqrt[n]{a}$` | `$$\sqrt[n]{a}$$` |

### 2.2 Equality and Relations

| Notation | Meaning | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| $a = b$ | Equals | `$a = b$` | `$$a = b$$` |
| $a \neq b$ | Not equal to | `$a \neq b$` | `$$a \neq b$$` |
| $a \equiv b$ | Identical / equivalent to | `$a \equiv b$` | `$$a \equiv b$$` |
| $a \propto b$ | Proportional to | `$a \propto b$` | `$$a \propto b$$` |
| $a \approx b$ | Approximately equal | `$\sin(0.01) \approx 0.01$` | `$$\sin(0.01) \approx 0.01$$` |
| $a \sim b$ | Distributed as / asymptotic | `$a \sim b$` | `$$a \sim b$$` |

### 2.3 Comparison and Inequalities

| Notation | Meaning | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| $a < b$, $a > b$ | Less than, greater than | `$a < b$`, `$a > b$` | `$$a < b$$` |
| $a \leq b$, $a \geq b$ | Less / greater than or equal | `$a \leq b$`, `$a \geq b$` | `$$a \leq b$$` |
| $a \ll b$, $a \gg b$ | Much less / greater than | `$a \ll b$`, `$a \gg b$` | `$$a \ll b$$` |

### 2.4 Algebra, Summations, and Products

| Notation | Example | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| Factorial | $5! = 5 \times 4 \times 3 \times 2 \times 1$ | `$5!$` | `$$5!$$` |
| Absolute Value / Norm | $\|{-5}\| = 5$ | `$\|-5\| = 5$` | `$$\|-5\| = 5$$` |
| Function Evaluation | $f(x) = 2x^2$ | `$f(x) = 2x^2$` | `$$f(x) = 2x^2$$` |
| Difference / Delta | $\Delta x = x_1 - x_0$ | `$\Delta x = x_1 - x_0$` | `$$\Delta x = x_1 - x_0$$` |
| Constants ($\pi, e$) | $\pi \approx 3.14159$, $e \approx 2.71828$ | `$\pi$`, `$e$` | `$$\pi \approx 3.14159$$` |
| Summation | $\sum_{k=3}^5 k^2 = 50$ | `$\sum_{k=3}^5 k^2 = 50$` | `$$\displaystyle\sum_{k=3}^5 k^2 = 50$$` |
| Series Product | $\prod_{x=2}^4 x^2 = 576$ | `$\prod_{x=2}^4 x^2 = 576$` | `$$\displaystyle\prod_{x=2}^4 x^2 = 576$$` |

### 2.5 Probability and Statistics

| Notation | Concept | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| $P(A)$, $\Pr(A)$ | Probability of event $A$ | `$P(A)$`, `$\Pr(A)$` | `$$P(A)$$` |
| $P(A \cap B)$ | Intersection probability ($A$ and $B$) | `$P(A \cap B)$` | `$$P(A \cap B)$$` |
| $P(A \cup B)$ | Union probability ($A$ or $B$) | `$P(A \cup B)$` | `$$P(A \cup B)$$` |
| $P(A \mid B)$ | Conditional probability ($A$ given $B$) | `$P(A \mid B)$` | `$$P(A \mid B)$$` |
| $\tilde{x}$ | Median | `$\tilde{x}$` | `$$\tilde{x}$$` |
| $\mu, \bar{x}, \langle x \rangle$ | Population mean, sample mean, expectation | `$\mu, \bar{x}, \langle x \rangle$` | `$$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$` |
| $\sigma, s$ | Population / sample standard deviation | `$\sigma$`, `$s$` | `$$\sigma$$` |
| $\sigma^2, s^2$ | Variance | `$\sigma^2$`, `$s^2$` | `$$\sigma^2$$` |

### 2.6 Linear Algebra: Vectors and Matrices

| Category | Example | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| Vector Notations | $\mathbf{v}, \vec{v}, \bar{v}$ | `$\mathbf{v}$`, `$\vec{v}$` | `$$\mathbf{v} \in \mathbb{R}^n$$` |
| Row Vector | $v = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$ | `$v = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$` | `$$v = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$$` |
| Column Vector | $w = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ | `$w = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$` | `$$w = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$$` |
| Dot Product | $\mathbf{v} \cdot \mathbf{w}, \langle v, w \rangle$ | `$\mathbf{v} \cdot \mathbf{w}$` | `$$\mathbf{v} \cdot \mathbf{w}$$` |
| Cross Product | $v \times w$ | `$v \times w$` | `$$v \times w$$` |
| Vector Norm / Length | $\|v\|, \|v\|_2$ | `$\|v\|_2$` | `$$\|v\|_2 = \sqrt{\sum_i v_i^2}$$` |
| Matrix Definition | $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ | `$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$` | `$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$` |
| Matrix Product | $A \cdot B, AB$ | `$A \cdot B$` | `$$A \cdot B$$` |
| Hadamard (elementwise) | $A \circ B$ | `$A \circ B$` | `$$A \circ B$$` |
| Kronecker Product | $A \otimes B$ | `$A \otimes B$` | `$$A \otimes B$$` |
| Transpose | $A^T$ | `$A^T$` | `$$A^T$$` |
| Conjugate Transpose | $A^\dagger, A^*$ | `$A^\dagger$` | `$$A^\dagger$$` |
| Inverse Matrix | $A^{-1}$ | `$A^{-1}$` | `$$A^{-1}$$` |
| Determinant | $\det(A), \|A\|$ | `$\det(A)$`, `$\|A\|$` | `$$\det(A)$$` |

### 2.7 Calculus and Differential Equations

| Concept | Expression | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| Definite Integral | $\int_1^4 \frac{x^2}{x} dx$ | `$\int_1^4 \frac{x^2}{x} dx$` | `$$A = \int_1^4 x\,dx$$` |
| First Derivative | $\frac{df}{dx}, f'(x)$ | `$\frac{df}{dx}$`, `$f'(x)$` | `$$\frac{df}{dx}$$` |
| Partial Derivative | $\frac{\partial f}{\partial x}$ | `$\frac{\partial f}{\partial x}$` | `$$\frac{\partial f}{\partial x}$$` |
| Higher Derivatives | $\frac{d^2f}{dx^2}, f''(x)$ | `$\frac{d^2f}{dx^2}$`, `$f''(x)$` | `$$\frac{d^2f}{dx^2}$$` |
| Time Derivatives | $\dot{f}, \ddot{f}$ | `$\dot{f}$`, `$\ddot{f}$` | `$$\ddot{x}(t) + \omega^2 x(t) = 0$$` |

### 2.8 Complex Numbers

| Concept | Expression | Inline Syntax | Display Syntax |
| --- | --- | --- | --- |
| Complex Number | $z = a + bi$ | `$z = a + bi$` | `$$z = 3 + 2i$$` |
| Real Part | $\operatorname{Re}(z), \Re(z)$ | `$\operatorname{Re}(z)$` | `$$\operatorname{Re}(z) = a$$` |
| Imaginary Part | $\operatorname{Im}(z), \Im(z)$ | `$\operatorname{Im}(z)$` | `$$\operatorname{Im}(z) = b$$` |
| Complex Conjugate | $\bar{z}, z^*$ | `$\bar{z}$`, `$z^*$` | `$$\bar{z} = a - bi$$` |

### 2.9 Greek Alphabet

| Name | Lowercase | Uppercase | Name | Lowercase | Uppercase |
| --- | --- | --- | --- | --- | --- |
| Alpha | $\alpha$ (`\alpha`) | $A$ (`A`) | Nu | $\nu$ (`\nu`) | $N$ (`N`) |
| Beta | $\beta$ (`\beta`) | $B$ (`B`) | Xi | $\xi$ (`\xi`) | $\Xi$ (`\Xi`) |
| Gamma | $\gamma$ (`\gamma`) | $\Gamma$ (`\Gamma`) | Omicron | $o$ (`o`) | $O$ (`O`) |
| Delta | $\delta$ (`\delta`) | $\Delta$ (`\Delta`) | Pi | $\pi$ (`\pi`) | $\Pi$ (`\Pi`) |
| Epsilon | $\epsilon, \varepsilon$ (`\epsilon`, `\varepsilon`) | $E$ (`E`) | Rho | $\rho, \varrho$ (`\rho`, `\varrho`) | $P$ (`P`) |
| Zeta | $\zeta$ (`\zeta`) | $Z$ (`Z`) | Sigma | $\sigma, \varsigma$ (`\sigma`, `\varsigma`) | $\Sigma$ (`\Sigma`) |
| Eta | $\eta$ (`\eta`) | $H$ (`H`) | Tau | $\tau$ (`\tau`) | $T$ (`T`) |
| Theta | $\theta, \vartheta$ (`\theta`, `\vartheta`) | $\Theta$ (`\Theta`) | Upsilon | $\upsilon$ (`\upsilon`) | $\Upsilon$ (`\Upsilon`) |
| Iota | $\iota$ (`\iota`) | $I$ (`I`) | Phi | $\phi, \varphi$ (`\phi`, `\varphi`) | $\Phi$ (`\Phi`) |
| Kappa | $\kappa$ (`\kappa`) | $K$ (`K`) | Chi | $\chi$ (`\chi`) | $X$ (`X`) |
| Lambda | $\lambda$ (`\lambda`) | $\Lambda$ (`\Lambda`) | Psi | $\psi$ (`\psi`) | $\Psi$ (`\Psi`) |
| Mu | $\mu$ (`\mu`) | $M$ (`M`) | Omega | $\omega$ (`\omega`) | $\Omega$ (`\Omega`) |

---

## 3. Tooling and Rendering Support

- **KaTeX / MathJax**: GitHub renders LaTeX expressions in issues, pull requests, discussions, and markdown files using KaTeX.
- **VS Code**: Markdown live preview supports KaTeX math rendering natively (toggle with `Ctrl+Shift+V` or `Ctrl+K V`).
- **Pandoc / XeLaTeX**: When compiling markdown to PDF via Pandoc, math equations enclosed in `$...$` or `$$...$$` are automatically translated to native LaTeX mathematical environments.
