---
title: Project direction and question synthesis
description: >
  Consolidates meeting-notes.md and daily_planning.md into a single working note
  that frames the project, groups the main open questions, and extracts immediate
  academic and technical actions.
created: 2026-05-12
status: draft
author:
  - "[[Lucas Galdino]]"
  - "[[GPT-5.4]]"
type: notes
scope: global
modifications:
  - date_modified: 2026-05-12
    modifications:
      - description: >
          Created a synthesized planning note that merges the meeting notes and
          daily planning backlog into a coherent project framing with categorized
          questions, working assumptions, and immediate next actions.
related_files:
  - [meeting-notes.md](./meeting-notes.md)
  - [daily_planning.md](./daily_planning.md)
  - [documentation/diagrams/flowchart_ga_loop_variants.md](./documentation/diagrams/flowchart_ga_loop_variants.md)
  - [documentation/diagrams/class_diagram_genetic_algorithms.md](./documentation/diagrams/class_diagram_genetic_algorithms.md)
tags:
  - notes/project-direction
  - notes/question-backlog
  - analysis/project-scope
  - analysis/research-framing
  - analysis/performance-evaluation
  - analysis/reproducibility
  - benchmark/memetic-algorithm
  - algorithm/genetic-algorithm
  - algorithm/two-opt
  - guide/presentation-planning
  - guide/thesis-structure
  - review/open-questions
---

# Project direction and question synthesis

## Interpreted project framing

- The project is best framed as an engineering-oriented study of a memetic algorithm for the TSP, with emphasis on how different CPU/GPU execution strategies affect runtime, memory behavior, population-scale behavior, and solution quality.
- The most coherent academic contribution is not to claim a brand-new metaheuristic, but to demonstrate a reproducible implementation, a clear benchmark methodology, and a useful explanation of the trade-offs.
- The meeting notes consistently point toward a TCC-level scope: apply existing knowledge well, justify the problem and the implementation choices, and communicate the work clearly to peers.

> [!important]
> A strong one-sentence scope is: evaluate and explain the computational trade-offs of GPU-accelerated memetic genetic algorithms for the TSP, focusing on runtime, memory, population effects, and solution quality.

## Proposed question hierarchy

Primary question:

- How do different GPU integration strategies in a memetic genetic algorithm affect execution time, memory behavior, and solution quality on TSP benchmark instances?

Secondary questions:

- Which variant offers the best trade-off between speed and solution quality?
- How do instance size and population-related choices affect the observed performance?
- Which implementation decisions explain the speed differences, such as data transfers, batch processing, and where local search is executed?
- Which assumptions and constraints define the current scope of the study?
- Which parts of the implementation must be documented so the work is understandable and reproducible?
- What future extension path is most credible: better analysis, stronger reproducibility, or broader problem coverage such as directed instances that may require 3-opt?

## Categorized question backlog

### Project positioning and scope

This group determines whether the project stays feasible and academically coherent as a TCC.

- Is this a good project? What could be changed?
- Should I proceed with this idea?
- What is missing for an OR-focused memetic algorithm?
- How should the project be simplified so it remains deliverable?
- Which boundaries should be explicit from the start: symmetric TSP only, or a future extension to directed problems?

### Methodology and evaluation

This group defines what must actually be measured and defended.

- Which analyses should be carried out?
- Which graphics best explain the inner working of the algorithm?
- How will the questions be answered experimentally?
- Which metrics matter most: execution time, memory use, population behavior, and solution quality?
- Which modeling techniques, constraints, and assumptions are part of the work?
- Which information should be extracted from the code and stored in DuckDB?
- How could new constraints be added to the algorithms?
- How could the algorithms be generalized without losing focus?

### Literature and theoretical grounding

This group supports the justification for the topic and connects the implementation to OR and GPU-computing literature.

- What should be read to implement and explain the work more practically?
- How should the literature review be organized so it covers both the current implementation and the state of the art?
- How should the text connect the parallelization choices to the theory that supports them?
- Which practical application examples should be used to justify research value, such as modeFRONTIER, scheduling, facility location, or aerospace optimization?

### Code understanding and reproducibility

This group addresses the problem of "demystifying" the code for both the author and the reader.

- How can the codebase be explained without turning the thesis into a code walkthrough?
- Should each folder have a README.md or AGENTS.md that explains its purpose and execution flow?
- Which diagrams should be used to explain the implementation at a high level?
- Which details belong in the main text, and which belong in appendices, links, or pseudo-code?
- Which low-level implementation decisions are important enough to mention, such as float32, kernels, or transfer strategy?

### Writing and presentation

This group covers how the work should be communicated.

- How should the text be structured?
- Should the presentation be in English?
- What does the advisor expect from the work?
- How should the material be written so classmates can understand and reproduce it?
- How should falsifiability and replicability appear in the written text?

### Academic process and planning

This group covers institutional constraints and execution planning.

- What are the TCC requirements for mechanical engineering?
- How is the committee assembled, and can the advisor be part of it?
- How many meetings should happen each week or month?
- What timeline is realistic?
- In how much time can the current goals be accomplished?

## Working assumptions extracted from the notes

- The work should be positioned as a solid undergraduate application of known ideas, not as a doctoral-level attempt to cross the research frontier.
- Focus matters more than breadth. The notes argue against expanding into a problem class that is too difficult to finish well.
- TSP is a relevant study vehicle because it is foundational and transferable to many applied optimization contexts.
- A useful explanatory emphasis is practical performance exploration: population, time, memory, and solution quality.
- The 2-opt limitation on directed problems suggests a credible future-work direction around 3-opt or ATSP-style extensions, but that should not necessarily be part of the core TCC scope.

## Immediate priorities

1. Prepare the presentation for Mayerle with a broad overview, collected results, and no detailed code walkthrough. The target format from the notes is 30 to 40 minutes.
2. Confirm the TCC rules for mechanical engineering, especially committee composition and formal requirements.
3. Define the thesis structure before expanding the writing effort. A practical baseline is motivation, related work, algorithmic approach, benchmark methodology, results, limitations, and future work.
4. Turn the literature review into a focused argument: what has been implemented, what the literature does, how the parallelization choices are justified, and why the problem matters in practice.
5. Improve project explainability and reproducibility with architecture notes, diagrams, and a clearer definition of what should be persisted in DuckDB.

## Existing supporting material

- Use [documentation/diagrams/flowchart_ga_loop_variants.md](./documentation/diagrams/flowchart_ga_loop_variants.md) as the high-level visual explanation of the shared GA loop and the CPU/GPU divergence points.
- Use [documentation/diagrams/class_diagram_genetic_algorithms.md](./documentation/diagrams/class_diagram_genetic_algorithms.md) as the architecture reference for the GA family and its shared abstractions.
- These two files can replace the placeholder note in the planning document that asks for a Mermaid diagram.

## Recommended next pass on the notes

- Keep [meeting-notes.md](./meeting-notes.md) as raw meeting capture.
- Keep [daily_planning.md](./daily_planning.md) as a lightweight operational checklist.
- Use this synthesis file as the stable project-level question map that turns scattered notes into a coherent planning structure.
