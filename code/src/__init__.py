"""
GPU-accelerated optimization algorithms for routing problems.

This package provides implementations of TSP, ATSP, and CVRP algorithms
with support for both CPU (NumPy) and GPU (CuPy) backends.

Main modules:
- protocols: Type safety interfaces for backend compatibility
- data_models: Problem representation and exceptions
- loaders: Database loading and data access
- distances: Distance matrix computation functions
- algorithms: Construction and improvement heuristics (future)
"""

__version__ = "0.1.0"
