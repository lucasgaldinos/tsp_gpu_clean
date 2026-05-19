
# Algorithm design

- [Algorithm design](#algorithm-design)
  - [Algorithm](#algorithm)
  - [Asymmetrical traveling salesman problem](#asymmetrical-traveling-salesman-problem)
  - [Guided local search](#guided-local-search)
    - [pseudocode](#pseudocode)
    - [Python code](#python-code)
    - [Key Adjustments](#key-adjustments)

## Algorithm

Design different local search algorithms, based on size of neighborhood. Depending on the size of neighborhood, choose some algorithm for the local search.

## Asymmetrical traveling salesman problem

...

## Guided local search

### pseudocode

```pseudocode
procedure GuidedLocalSearch(S, g, lambda, [I_1, ...,I_M], [c_1,...,c_M], M)

begin
  
  k = 0;
  s_0 = random or heuristically generated solution in S;
  
  for i = 1; i=M; do /* set all penalties to 0 */
    p_i = 0;
  
  /*Until here are just initialization steps /*



  while local_minimum do
  begin
    h = g + lambda * sum(p_i *I_i); /* should do another for here? /*
    s_k+1 = LocalSearch(s_k, h); /* finds a solution with a cost
  
    for i=1; i=M; do
      utility = I_i(s_k+1) * c_i / (1+p_i);
    
    for each i such that utility is maximum do /* should do another for here? Is this the right indentation? */
      p_i = p_i + 1;
    
    k = k+1;
  
  end
  s* = best solution found with respect to cost function g;
  return s*;
end
```

### Python code

```python
import random

def guided_local_search(S, g, lambda_, I, c, M, stopping_criterion):
    # Initialize variables
    k = 0
    s_0 = random.choice(S)  # Assuming S is a list of possible solutions or a generator function

    # Initialize penalties
    p = [0] * M

    s_k = s_0

    while not stopping_criterion(k, s_k):
        # Calculate modified cost function h
        h = g(s_k) + lambda_ * sum(p[i] * I[i](s_k) for i in range(M))

        # Perform local search
        s_k_plus_1 = local_search(s_k, h)

        # Calculate utility for each feature
        utilities = [I[i](s_k_plus_1) * c[i] / (1 + p[i]) for i in range(M)]

        # Find the maximum utility value
        max_utility = max(utilities)

        # Update penalties for features with maximum utility
        for i in range(M):
            if utilities[i] == max_utility:
                p[i] += 1

        # Update the current solution and iteration counter
        s_k = s_k_plus_1
        k += 1

    # Return the best solution found with respect to cost function g
    return s_k

# Assuming you have a function for LocalSearch
def local_search(s, h):
    # Implement your local search logic here
    pass

# Example usage:
# Define your cost function g, indicator functions I, costs c, and stopping criterion
```

##########

```python
import random
from typing import Callable, List, Any

def guided_local_search(
    S: List[Any],
    g: Callable[[Any], float],
    lambda_: float,
    I: List[Callable[[Any], float]],
    c: List[float],
    M: int,
    is_local_minimum: Callable[[Any], bool]
) -> Any:
    """
    Performs the Guided Local Search algorithm.

    :param S: A list of possible solutions.
    :param g: A cost function that evaluates a solution.
    :param lambda_: A parameter to balance between original cost and penalties.
    :param I: A list of indicator functions for features.
    :param c: A list of costs associated with each feature.
    :param M: The number of features.
    :param is_local_minimum: A function to determine if s_k is at a local minimum.
    :return: The best solution found with respect to cost function g.
    """
    
    # Initialize variables
    k = 0
    s_k = random.choice(S)  # Assuming S is a list of possible solutions

    # Initialize penalties
    p = [0] * M

    while True:
        # Calculate modified cost function h
        h = g(s_k) + lambda_ * sum(p[i] * I[i](s_k) for i in range(M))

        # Perform local search
        s_k_plus_1 = local_search(s_k, h)

        # Check if s_k_plus_1 is at a local minimum
        if not is_local_minimum(s_k_plus_1):
            break

        # Calculate utility for each feature
        utilities = [I[i](s_k_plus_1) * c[i] / (1 + p[i]) for i in range(M)]

        # Find the maximum utility value
        max_utility = max(utilities)

        # Update penalties for features with maximum utility
        for i in range(M):
            if utilities[i] == max_utility:
                p[i] += 1

        # Update the current solution and iteration counter
        s_k = s_k_plus_1
        k += 1

    # Return the best solution found with respect to cost function g
    return s_k

# Assuming you have a function for LocalSearch
def local_search(s: Any, h: float) -> Any:
    """
    A placeholder function for local search logic.

    :param s: The current solution.
    :param h: The modified cost function value.
    :return: A new solution found by local search.
    """
    # Implement your local search logic here
    pass

# Example usage:
# Define your cost function g, indicator functions I, costs c, and a function to check if at local minimum
```

### Key Adjustments

- **Stopping Criterion Function**: Replace `stopping_criterion` with `is_local_minimum`, which should be defined to check if `s_k` is at a local minimum.

- **Loop Condition**: The loop now breaks when `s_k_plus_1` is not at a local minimum, aligning more closely with your intended logic from the pseudocode.

This setup assumes you have implemented logic within `is_local_minimum` to correctly identify when a solution is at a local minimum based on your problem's specifics.
