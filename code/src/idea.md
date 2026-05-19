---
title: ideas
---

1. more parents, better section partitioning for GA.
   1. In my [crossover function](./algorithms/strategies/crossover_strategies.py#l57-l109), I use 2 parents to create 2 children.
   2. Could use more parents (3, 4, etc) to create a better fitted individual? like mixing 3 parents and partitioning the population/solutions into 3 sections instead of 2 selecting e.g. a better fitted part from one parent etc.
   3. How would this addect parallelism?
2. mating pool size is variable, not fixed, but population size is fixed.
   Could change so the mating pool, in parallel, has variable population size.
