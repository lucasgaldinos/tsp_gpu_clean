"""Polar Sweep (Gillett & Miller, 1974) CPU and GPU initializers (`Task 17.31`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.gpu_base import DistanceOperandCache
from protocols.algorithm_strategies import InitializationStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.splitmix64 import draw_row_seeds, splitmix64_bounded_scalar

from .vrp_helpers import compute_sorted_sweep_customers, pack_vrp_routes_np, resolve_vrp_customers

if TYPE_CHECKING:
  from collections.abc import Sequence

  from protocols.problem_context import ProblemContext
  from protocols.types import Routes, SeedLike


def _validate_output_format(output_format: str) -> Literal["giant_tour", "delimited"]:
  if output_format not in ("giant_tour", "delimited"):
    raise ValueError(f"output_format must be 'giant_tour' or 'delimited', got {output_format!r}")
  return cast("Literal['giant_tour', 'delimited']", output_format)


def _pack_greedy_sweep_cpu(
  ordered: np.ndarray,
  distances: np.ndarray,
  demands: np.ndarray,
  context: ProblemContext,
) -> list[np.ndarray]:
  """Pack ``ordered`` customers greedily into routes respecting capacity and distance limits."""
  constraints = context.constraints
  capacity = constraints.capacity
  max_route_distance = constraints.max_route_distance
  return_to_depot = constraints.return_to_depot

  routes: list[list[int]] = []
  current_route: list[int] = []
  current_load = 0
  current_open_dist = 0

  for cust_raw in ordered:
    cust = int(cust_raw)
    q_cust = int(demands[cust])
    if not current_route:
      current_route = [cust]
      current_load = q_cust
      current_open_dist = int(distances[0, cust])
      continue

    new_load = current_load + q_cust
    new_open_dist = current_open_dist + int(distances[current_route[-1], cust])
    ret_edge = int(distances[cust, 0]) if return_to_depot else 0
    cap_ok = capacity is None or new_load <= capacity
    dist_ok = max_route_distance is None or float(new_open_dist + ret_edge) <= max_route_distance

    if cap_ok and dist_ok:
      current_route.append(cust)
      current_load = new_load
      current_open_dist = new_open_dist
    else:
      routes.append(current_route)
      current_route = [cust]
      current_load = q_cust
      current_open_dist = int(distances[0, cust])

  if current_route:
    routes.append(current_route)
  return [np.asarray(r, dtype=np.int32) for r in routes]


class InitializationCPUSweep(StrategyBase, InitializationStrategy):
  """CPU Polar Sweep initialization with deterministic ``SplitMix64`` cyclic offsets."""

  def __init__(
    self,
    output_format: Literal["giant_tour", "delimited"] = "giant_tour",
    seed: SeedLike = None,
  ) -> None:
    super().__init__(seed)
    self.output_format: Literal["giant_tour", "delimited"] = _validate_output_format(output_format)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` polar-sweep VRP routes on CPU."""
    if context is None:
      raise ValueError("InitializationCPUSweep requires a ProblemContext")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    customer_ids = resolve_vrp_customers(customers, context)
    n_cust = int(customer_ids.size)
    k_max = context.constraints.max_vehicles or max(1, n_cust)
    row_width = n_cust if self.output_format == "giant_tour" else context.dimension + k_max
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0:
      return np.empty((0, row_width), dtype=np.int32)

    sorted_customers = compute_sorted_sweep_customers(customer_ids, context)
    pop = np.empty((population_size, row_width), dtype=np.int32)
    if self.output_format == "giant_tour":
      for idx in range(population_size):
        if n_cust == 0:
          pop[idx] = sorted_customers
        else:
          offset = 0 if (idx == 0 or n_cust <= 1) else splitmix64_bounded_scalar(int(row_seeds[idx]), n_cust)[1]
          pop[idx] = np.roll(sorted_customers, -offset)
      return pop

    distances = context.get_cpu_distances()
    demands = context.get_cpu_demands()
    if demands is None:
      demands = np.zeros(context.dimension, dtype=np.int32)

    for idx in range(population_size):
      if n_cust == 0:
        ordered = sorted_customers
      else:
        offset = 0 if (idx == 0 or n_cust <= 1) else splitmix64_bounded_scalar(int(row_seeds[idx]), n_cust)[1]
        ordered = np.roll(sorted_customers, -offset)
      routes = _pack_greedy_sweep_cpu(ordered, distances, demands, context)
      pop[idx] = pack_vrp_routes_np(
        routes,
        dimension=context.dimension,
        k_max=k_max,
        output_format="delimited",
      )
    return pop


class InitializationGPUSweep(StrategyBase, InitializationStrategy):
  """VRAM-native CUDA Polar Sweep initialization bit-identical to ``InitializationCPUSweep``."""

  def __init__(
    self,
    output_format: Literal["giant_tour", "delimited"] = "giant_tour",
    seed: SeedLike = None,
    max_matrix_dimension: int | None = None,
    return_gpu: bool = True,
  ) -> None:
    if max_matrix_dimension is not None and (
      isinstance(max_matrix_dimension, bool) or max_matrix_dimension < 0
    ):
      raise ValueError(f"max_matrix_dimension must be >= 0 or None, got {max_matrix_dimension!r}")
    super().__init__(seed)
    self.output_format: Literal["giant_tour", "delimited"] = _validate_output_format(output_format)
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.return_gpu: bool = return_gpu
    self._kernel: Any = None
    self._operands: DistanceOperandCache = DistanceOperandCache()

  def reset_metrics(self) -> None:
    """Reset transfer/launch counters and mirrored operand bindings."""
    super().reset_metrics()
    self._operands.reset()

  def _get_kernel(self) -> Any:
    """Lazily compile and cache ``polar_sweep_kernel``."""
    if self._kernel is None:
      self._kernel = load_kernel("polar_sweep.cu", "polar_sweep_kernel", __file__)
    return self._kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch ``polar_sweep_kernel`` ahead of measured runs without mutating state."""
    require_gpu()
    super()._warmup_impl(context)
    self._get_kernel()
    if context.dimension <= 1:
      return
    self.initialize(list(range(1, context.dimension)), 2, context)

  @time_strategy
  def initialize(
    self,
    customers: Sequence[int],
    population_size: int,
    context: ProblemContext | None = None,
  ) -> Routes:
    """Generate ``population_size`` polar-sweep VRP routes directly in VRAM."""
    if context is None:
      raise ValueError("InitializationGPUSweep requires a ProblemContext")
    if population_size < 0:
      raise ValueError(f"population_size must be >= 0, got {population_size}")

    cp = require_gpu()
    customer_ids = resolve_vrp_customers(customers, context)
    n_cust = int(customer_ids.size)
    k_max = context.constraints.max_vehicles or max(1, n_cust)
    row_width = n_cust if self.output_format == "giant_tour" else context.dimension + k_max
    row_seeds = draw_row_seeds(resolve_rng(context, self.rng), population_size)

    if population_size == 0:
      empty_gpu = cp.empty((0, row_width), dtype=cp.int32)
      return cast("Routes", empty_gpu) if self.return_gpu else np.empty((0, row_width), dtype=np.int32)

    sorted_customers = compute_sorted_sweep_customers(customer_ids, context)
    d_sorted = cp.asarray(sorted_customers, dtype=cp.int32)
    d_seeds = cp.asarray(row_seeds, dtype=cp.uint64)
    self.h2d_bytes += int(sorted_customers.nbytes + row_seeds.nbytes)

    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    operands, uploaded_bytes = self._operands.resolve(
      context, context.dimension, self.max_matrix_dimension
    )
    self.h2d_bytes += uploaded_bytes
    d_distances, d_coords, edge_type_code = operands

    d_pop = cp.empty((population_size, row_width), dtype=cp.int32)
    constraints = context.constraints
    block = 128
    grid = (population_size + block - 1) // block
    kernel = self._get_kernel()
    kernel(
      (grid,),
      (block,),
      (
        d_sorted,
        d_demands,
        d_distances,
        d_coords,
        d_seeds,
        d_pop,
        np.int32(population_size),
        np.int32(n_cust),
        np.int32(row_width),
        np.int32(context.dimension),
        np.int32(edge_type_code),
        np.int32(k_max),
        np.int32(1 if self.output_format == "delimited" else 0),
        np.int32(1 if constraints.return_to_depot else 0),
        np.int32(1 if constraints.capacity is not None else 0),
        np.int64(constraints.capacity if constraints.capacity is not None else 0),
        np.int32(1 if constraints.max_route_distance is not None else 0),
        np.float64(
          constraints.max_route_distance if constraints.max_route_distance is not None else 0.0
        ),
      ),
    )
    self.kernel_launches += 1

    if self.return_gpu:
      return cast("Routes", d_pop)
    host_pop = np.asarray(d_pop.get(), dtype=np.int32)
    self.d2h_bytes += int(host_pop.nbytes)
    return host_pop


__all__: list[str] = ["InitializationCPUSweep", "InitializationGPUSweep"]
