---
title: "Formal Problem Definitions: Traveling Salesperson and Capacitated Vehicle Routing"
description: >
  Formal mathematical specifications for the Traveling Salesperson Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP): graph models, decision variables, capacity constraints, and sub-tour elimination.
created: 2025-01-12
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: analysis
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Standardized frontmatter and filenames per markdown standards, aligned related files and validated tags.
related_files:
  - "[notes_average_worst_case_analysis.md](./notes_average_worst_case_analysis.md)"
  - "[guide-construction-heuristics.md](./guide-construction-heuristics.md)"
tags:
  - analysis/problem-definition
  - algorithm/tsp
  - algorithm/cvrp
  - notes/theory
  - notes/mathematics
  - review/quality
  - guide/algorithm-composition
  - analysis/complexity
  - notes/optimization
  - algorithm/combinatorial-optimization
---

# Machine specs

Make a step by step of how to develop the following problem: `# Analysis, development and comparison of multiple vrp optimization algorithms`, with focus in a solution that will solve both Vehicle Routing Problem with Pickup and Delivery (VRPPD) and Capacitated Vehicle Routing Problem (CVRP), with another variation analysis for the Open Vehicle Routing Problem (OVRP).

The solution may use any of the following software technologies as the project requirements: sqlite3, python, c++, javascript, docker and linux.

```json
"machine_specs": [
    {"cpu":"Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
    CPU family:           6
    Model:                158
    Thread(s) per core:   2
    Core(s) per socket:   4
    Socket(s):            1
    Stepping:             9
    CPU(s) scaling MHz:   91%
    CPU max MHz:          3800.0000
    CPU min MHz:          800.0000
    BogoMIPS:             5599.85",
    ""
    }
    ]
```

=
