"""Unit tests for `ProblemContext` VRP constraints, topology, and demand accessors (`DP17-1`, `DP17-3`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from data_models.exceptions import GPUUnavailableError
from data_models.problem import Problem
from data_models.vrp_constraints import VRPConstraints
from protocols.backend import gpu_available
from protocols.problem_context import ProblemContext
from utils.route_helpers import resolve_route_bounds

DIMENSION = 5
COORDINATES: np.ndarray = np.array(
  [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
  dtype=np.float64,
)
DISTANCES: np.ndarray = np.zeros((DIMENSION, DIMENSION), dtype=np.int32)
DEMANDS: np.ndarray = np.array([0, 2, 3, 1, 4], dtype=np.int32)


def _make_variant(problem_type: str, *, constraints: VRPConstraints | None = None) -> Problem:
  """Construct a minimal valid `Problem` for `TSP`, `ATSP`, `CVRP`, `OVRP`, or `DCVRP`."""
  kwargs: dict[str, object] = {}
  if constraints is not None:
    kwargs["constraints"] = constraints

  if problem_type == "ATSP":
    return Problem(
      name="atsp5",
      dimension=DIMENSION,
      problem_type="ATSP",
      edge_type="EXPLICIT",
      coordinates=None,
      distances=DISTANCES,
      **kwargs,  # type: ignore[arg-type]
    )
  if problem_type == "TSP":
    return Problem(
      name="tsp5",
      dimension=DIMENSION,
      problem_type="TSP",
      edge_type="EUC_2D",
      coordinates=COORDINATES,
      distances=None,
      **kwargs,  # type: ignore[arg-type]
    )
  if problem_type == "DCVRP" and constraints is None:
    kwargs["constraints"] = VRPConstraints(capacity=10, max_route_distance=50.0)
  return Problem(
    name=f"{problem_type.lower()}5",
    dimension=DIMENSION,
    problem_type=problem_type,
    edge_type="EUC_2D",
    coordinates=COORDINATES,
    distances=None,
    capacity=10,
    demands=DEMANDS,
    **kwargs,  # type: ignore[arg-type]
  )


@pytest.mark.parametrize(
  ("problem_type", "expected_return", "expected_closed", "expected_bounds"),
  [
    pytest.param("TSP", True, False, (0, 5), id="tsp"),
    pytest.param("ATSP", True, False, (0, 5), id="atsp"),
    pytest.param("CVRP", True, True, (1, 4), id="cvrp"),
    pytest.param("DCVRP", True, True, (1, 4), id="dcvrp"),
    pytest.param("OVRP", False, False, (0, 5), id="ovrp"),
  ],
)
def test_problem_context_exposes_constraints_and_topology(
  problem_type: str,
  expected_return: bool,
  expected_closed: bool,
  expected_bounds: tuple[int, int],
) -> None:
  """WHAT: `constraints`, `return_to_depot`, `is_closed`, and `resolve_route_bounds` per variant.
  WHY: Operators read `context.constraints` and `context.is_closed` without device syncs.
  EXPECTED: closed VRP (`CVRP`/`DCVRP`) -> `is_closed=True`, bounds `(1, n-1)`;
  open variants (`TSP`/`ATSP`/`OVRP`) -> `is_closed=False`, bounds `(0, n)`.
  """
  problem = _make_variant(problem_type)
  context = ProblemContext(problem, xp=np, seed=42)

  assert context.constraints is problem.constraints
  assert context.return_to_depot is expected_return
  assert context.is_closed is expected_closed
  assert resolve_route_bounds(DIMENSION, context=context) == expected_bounds


def test_open_tsp_exposes_return_to_depot_false() -> None:
  """WHAT: `ProblemContext` wrapping an Open TSP (`TSP` with `return_to_depot=False`).
  WHY: `DP17-1` supports Open TSP via `VRPConstraints(return_to_depot=False)`.
  EXPECTED: `return_to_depot=False`, `is_closed=False`, `demands=None`.
  """
  problem = _make_variant("TSP", constraints=VRPConstraints(return_to_depot=False))
  context = ProblemContext(problem, xp=np, seed=7)

  assert context.return_to_depot is False
  assert context.is_closed is False
  assert context.get_cpu_demands() is None


@pytest.mark.parametrize("problem_type", ["CVRP", "OVRP", "DCVRP"])
def test_cpu_demands_are_int32_contiguous_and_cached(problem_type: str) -> None:
  """WHAT: `get_cpu_demands()` across all VRP variants.
  WHY: `DP17-3` requires C-contiguous `int32` demand vectors with lazy caching.
  EXPECTED: C-contiguous `int32` array matching `DEMANDS`, cached on repeat calls.
  """
  problem = _make_variant(problem_type)
  context = ProblemContext(problem, xp=np, seed=1)

  assert context._cpu_demands is None
  demands = context.get_cpu_demands()
  assert demands is not None
  assert demands.dtype == np.dtype(np.int32)
  assert demands.flags["C_CONTIGUOUS"]
  np.testing.assert_array_equal(demands, DEMANDS)
  assert context.get_cpu_demands() is demands


@pytest.mark.parametrize("problem_type", ["CVRP", "OVRP", "DCVRP"])
def test_gpu_demands_contract_and_h2d_accounting(problem_type: str) -> None:
  """WHAT: `get_gpu_demands()` across all VRP variants.
  WHY: `DP17-3` requires C-contiguous `int32` VRAM demands uploaded once (`BUG-95`).
  EXPECTED: C-contiguous `int32` CuPy array with single H2D byte accounting when GPU is available,
  or `GPUUnavailableError` otherwise.
  """
  problem = _make_variant(problem_type)
  context = ProblemContext(problem, xp=np, seed=2)

  if not gpu_available():
    with pytest.raises(GPUUnavailableError):
      context.get_gpu_demands()
    return

  gpu_demands: Any = context.get_gpu_demands()
  assert gpu_demands is not None
  assert gpu_demands.dtype == np.dtype(np.int32)
  assert gpu_demands.flags.c_contiguous
  expected_bytes = int(gpu_demands.nbytes)
  assert context.h2d_bytes == expected_bytes
  assert context.get_gpu_demands() is gpu_demands
  assert context.h2d_bytes == expected_bytes
