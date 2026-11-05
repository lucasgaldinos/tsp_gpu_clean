---
title: "Guide: Academic Critical Appraisal and TCC Evaluation Framework"
description: >
  Comprehensive guide and structured checklist for the critical appraisal of academic
  papers and Brazilian TCC monographs, integrating ABNT NBR standards (NBR 14724, NBR 6027),
  methodological validity checks, and statistical rigor criteria.
created: 2026-11-05
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Consolidated analise_critica.md, roteiro_analise_critica.md, and
          critical_analysis_checklist.md into a unified academic evaluation guide.
related_files:
  - "[analysis-literature-synthesis-prior-art.md](./analysis-literature-synthesis-prior-art.md)"
  - "[analysis-routing-literature-corpus.md](./analysis-routing-literature-corpus.md)"
tags:
  - guide/critical_analysis
  - guide/academic_writing
  - guide/tcc
  - review/methodology
  - review/quality
  - analysis/evaluation
  - notes/abnt
  - notes/checklist
  - guide/peer_review
  - notes/reproducibility
---

# Academic Critical Appraisal and TCC Evaluation Framework

This guide provides a comprehensive framework for the critical evaluation of scholarly literature and the structural auditing of academic monographs (Trabalho de Conclusão de Curso - TCC), integrating Brazilian Association of Technical Standards (ABNT) requirements with international scientific appraisal criteria.

---

## 1. Structural Standards for Academic Monographs (ABNT / NBR)

In accordance with Brazilian academic standards (notably **NBR 14724:2011** for academic work presentation and **NBR 6027:2012** for tables of contents/sumários), an academic monograph is divided into three structural divisions:

### 1.1 Elementos Pré-Textuais (Pre-Textual Elements)

Pre-textual elements provide identification, attribution, and organizational navigation. **Note:** In compliance with NBR 6027, pre-textual elements must **not** appear in the Table of Contents (*Sumário*).

- **Capa e Folha de Rosto**: Institutional affiliation, author, title, subtitle, academic degree, advisor, and date/city.
- **Folha de Aprovação**: Formal evaluation board records.
- **Dedicatória, Agradecimentos e Epígrafe** (Optional).
- **Resumo na Língua Vernácula**: Portuguese abstract synthesizing problem, methodology, results, and conclusion (150–500 words), followed by keywords (*Palavras-chave*).
- **Abstract / Resumo em Língua Estrangeira**: Faithful English translation of the abstract and keywords.
- **Listas Ilustrativas**:
  - Lista de Figuras / Gráficos
  - Lista de Tabelas / Quadros
  - Lista de Símbolos
  - Lista de Abreviaturas e Siglas
- **Sumário**: Enumeration of primary divisions, chapters, and sections in the exact order and spelling as they appear in the body of the text.

### 1.2 Elementos Textuais (Textual Elements)

- **Introdução e Justificativa**:
  - Clear statement of the central research problem and motivation.
  - Guiding questions: *Por que mudar? Como mudar? O quê mudar? Para o que mudar? Como causar a mudança?*
- **Objetivos**:
  - *Objetivo Geral*: Unambiguous macro objective of the research.
  - *Objetivos Específicos*: Operational milestones required to achieve the general objective.
- **Revisão Bibliográfica / Fundamentação Teórica**:
  - Systematic synthesis of prior art, seminal works, and contemporary developments.
  - Identification of the validated research gap.
- **Materiais e Métodos**:
  - Precise qualification and identification of datasets, benchmarks, hardware, and software libraries.
  - Detailed experimental protocols ensuring reproducibility.
- **Resultados e Discussão**:
  - Empirical presentation (tables, figures, convergence profiles).
  - In-depth statistical analysis (hypothesis testing, significance levels, effect size).
  - Critical contextualization against state-of-the-art baselines.
- **Conclusão**:
  - Direct correspondence to the initial objectives.
  - Synthesis of contributions, acknowledged limitations, and suggested future work.

### 1.3 Elementos Pós-Textuais (Post-Textual Elements)

- **Referências (NBR 6023)**: Complete list of cited works.
- **Glossário** (Optional): Alphabetical list of specialized terms and definitions.
- **Apêndices**: Supplementary materials authored by the candidate (e.g., proofs, raw scripts).
- **Anexos**: Documents or external data not authored by the candidate included for documentation.

---

## 2. Eight-Dimension Critical Appraisal Checklist

Use this structured rubric when reviewing research papers, theses, or draft chapters.

### Dimension 1: Initial Assessment & Research Relevance
- [ ] **1.1 Research Question**: Is the central research problem focused, well-defined, and unambiguous?
- [ ] **1.2 Scope & Relevance**: Does the paper directly address routing optimization, high-performance computing, or parallel heuristics?
- [ ] **1.3 Timeliness**: Does the work incorporate contemporary research (last 3–5 years) alongside seminal foundation literature?
- [ ] **1.4 Publication Venue**: Is the publication peer-reviewed? Does it adhere to recognized academic rigor?
- [ ] **1.5 Abstract Alignment**: Does the abstract accurately reflect the actual experimental results without exaggeration?

### Dimension 2: Introduction, Justification & Hypotheses
- [ ] **2.1 Problem Framing**: Is there an articulated necessity for this investigation?
- [ ] **2.2 Literature Gap**: Does the introduction demonstrate why existing solutions or classical baselines are insufficient?
- [ ] **2.3 Formulation of Objectives**: Are the general and specific objectives stated in testable, measurable terms?
- [ ] **2.4 Value Proposition**: What is the practical, theoretical, or computational impact of solving this problem?

### Dimension 3: Methodology & Experimental Design
- [ ] **3.1 Design Validity**: Is the experimental methodology appropriate for answering the stated research question?
- [ ] **3.2 Benchmark Selection**: Are standard, recognized problem instances used (e.g., TSPLIB95, CVRPLIB)?
- [ ] **3.3 Bias Mitigation & Fair Comparison**:
  - Are CPU and GPU implementations tested under identical problem instances and termination criteria?
  - Are warmup runs, multiple random seeds, and host-device transfer timings segregated appropriately?
- [ ] **3.4 Instrumentation & Tools**: Are hardware specifications (CPU, GPU, VRAM, memory bandwidth) and software versions (Python, CUDA, CuPy, Numba) explicitly documented?
- [ ] **3.5 Reproducibility**: Is the implementation described in sufficient detail—ideally backed by an open-source repository—to allow independent replication?

### Dimension 4: Results & Statistical Analysis
- [ ] **4.1 Data Presentation**: Are results logically organized using clear tables and figures with descriptive captions?
- [ ] **4.2 Statistical Rigor**:
  - Is there appropriate statistical treatment for stochastic algorithms (mean, median, standard deviation, interquartile range)?
  - Are non-parametric tests (Mann-Whitney $U$, Kruskal-Wallis) utilized when normality cannot be assumed?
  - Are effect sizes (e.g., Vargha-Delaney $\hat{A}_{12}$, Cohen's $d$) calculated to verify practical significance?
- [ ] **4.3 Significance Integrity**: Is there complete absence of "significance fishing" or selective reporting of favorable runs?
- [ ] **4.4 Completeness**: Are all metrics mentioned in the methodology (runtime, solution quality, memory consumption) systematically reported?

### Dimension 5: Discussion & Interpretation
- [ ] **5.1 Alignment with Objectives**: Do the results directly satisfy the questions posed in the introduction?
- [ ] **5.2 Plausibility**: Are speedup claims physically plausible given GPU memory bandwidth and compute bounds?
- [ ] **5.3 Contextualization**: How do the findings compare against established literature (e.g., LKH-3, state-of-the-art CUDA solvers)?
- [ ] **5.4 Limitation Acknowledgment**: Do the authors candidly address boundary conditions, hardware bottlenecks, and algorithmic constraints?

### Dimension 6: Conclusion & Implications
- [ ] **6.1 Goal Attainment**: Do the conclusions strictly follow from the presented evidence without unsubstantiated extrapolations?
- [ ] **6.2 Theoretical Contribution**: How does this work advance operations research or parallel algorithmic paradigms?
- [ ] **6.3 Practical Implications**: Can practitioners apply these findings to real-world logistics or route planning?
- [ ] **6.4 Future Work**: Are concrete, actionable future research avenues proposed?

### Dimension 7: Linguistic Quality & Technical Rigor
- [ ] **7.1 Coherence & Clarity**: Are paragraphs logically sequenced with smooth transitions?
- [ ] **7.2 Grammar & Terminology**: Is technical terminology used precisely? Are verb tenses consistent (e.g., past tense for completed experiments, present tense for established facts)?
- [ ] **7.3 Standards Compliance**: Does the citation style and document layout comply with institutional guidelines (ABNT NBR 14724 / 6023 / 6027)?

### Dimension 8: Critical Synthesis & Reflection
- [ ] **8.1 Core Takeaway**: What new technical understanding does this study provide regarding GPU acceleration?
- [ ] **8.2 Primary Strengths**: What constitutes the most robust, well-executed aspect of the study?
- [ ] **8.3 Key Vulnerability**: Where is the analysis most fragile or in need of further empirical verification?
