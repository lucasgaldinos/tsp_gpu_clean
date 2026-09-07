"""
Distance computation for TSPLIB95 edge weight types.

This module provides complete TSPLIB95-compliant distance functions for all
13 edge weight types. Implementations are adapted from proven academic code
(thesis project: sem0ark/met_projects) and verified against TSPLIB95 specification.

Edge Weight Types Supported:
    - EUC_2D, EUC_3D: Euclidean distance (L2-metric)
    - MAN_2D, MAN_3D: Manhattan distance (L1-metric)
    - MAX_2D, MAX_3D: Maximum distance (L∞-metric)
    - CEIL_2D: Ceiling of Euclidean distance
    - GEO: Geographical distance on spherical Earth
    - ATT: Pseudo-Euclidean distance (att48, att532)
    - EXPLICIT: Pre-computed matrix (handled by matrix lookup)
    - XRAY1, XRAY2: Crystallography distances
    - SPECIAL: Problem-specific (requires external documentation)

Design Philosophy:
    - Simple pure functions (no classes, no state)
    - NumPy/CuPy agnostic (operate on coordinates, not arrays)
    - TSPLIB95-compliant rounding (nint = int(x + 0.5))
    - Proven formulas from academic implementations
    - Backend-agnostic (adaptable for GPU via matrix operations)

Usage:
    >>> from distances import compute_euclidean_2d
    >>> dist = compute_euclidean_2d((0.0, 0.0), (3.0, 4.0))
    >>> print(dist)  # 5 (rounded to nearest integer)
    5

    >>> from distances import get_distance_function
    >>> dist_func = get_distance_function('EUC_2D')
    >>> dist_func((0.0, 0.0), (3.0, 4.0))
    5

References:
    - TSPLIB95 specification: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
    - Academic implementation: https://github.com/sem0ark/met_projects/
"""

from .pairwise import (
    compute_euclidean_2d,
    compute_euclidean_3d,
    compute_manhattan_2d,
    compute_manhattan_3d,
    compute_maximum_2d,
    compute_maximum_3d,
    compute_ceiling_2d,
    compute_geographical,
    compute_att,
    compute_xray1,
    get_distance_function,
)

__all__ = [
    "compute_euclidean_2d",
    "compute_euclidean_3d",
    "compute_manhattan_2d",
    "compute_manhattan_3d",
    "compute_maximum_2d",
    "compute_maximum_3d",
    "compute_ceiling_2d",
    "compute_geographical",
    "compute_att",
    "compute_xray1",
    "get_distance_function",
]
