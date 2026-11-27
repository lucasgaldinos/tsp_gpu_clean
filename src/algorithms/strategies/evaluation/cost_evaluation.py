"""Unified CPU and GPU routing cost evaluation strategies (`DP11-15`, `DP17-1`, `DP17-7`, `DP17-9`).

Provides `EvaluationCPUCost` and `EvaluationGPUCost` handling both:
- Unconstrained Hamiltonian tours (`not context.problem.is_vrp`, closed and open TSP/ATSP) via
  `tour_cost.py` / `tour_cost.cu`.
- Constrained VRP batches (`context.problem.is_vrp`) in either Giant-Tour `(N, n - 1)` format
  (delegated to `self._decoder`) or `0`-delimited `(N, W)` (`W >= n`) format via
  `delimited_cost.py` / `vrp_delimited_cost.cu`, dispatching with zero D2H sync on GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel, prev_power_of_2
from utils.phase_timer import time_strategy
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape, to_cpu_batch

from .base import EvaluationCPUBase, EvaluationGPUBase
from .delimited_cost import (
  UnifiedCostMixin,
  build_cpu_vrp_assessment,
  build_cpu_vrp_detailed,
  build_gpu_vrp_assessment,
  evaluate_delimited_cpu,
  launch_delimited_cost_gpu,
)
from .tour_cost import (
  build_tour_assessment,
  build_tour_detailed_result,
  evaluate_tour_cpu,
  launch_tour_cost_gpu,
)

if TYPE_CHECKING:
  from algorithms.strategies.decoding import SplitBatchResult
  from data_models.problem import Problem
  from protocols.problem_context import ProblemContext
  from protocols.types import Fitness, RouteOrRoutes
  from protocols.vrp_strategies import PenaltyPolicy, RouteAssessment, RouteDecoder


class EvaluationCPUCost(UnifiedCostMixin, EvaluationCPUBase):
  """Unified CPU cost evaluator for TSP/ATSP tours, VRP Giant-Tours, and `0`-delimited routes."""

  def __init__(
    self,
    decoder: RouteDecoder | None = None,
    split_mode: Literal["prins", "greedy"] = "prins",
    penalty_policy: PenaltyPolicy | None = None,
    max_matrix_dimension: int | None = None,
  ) -> None:
    """Initialize unified CPU cost evaluator."""
    super().__init__()
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self._init_vrp_components(
      decoder=decoder,
      split_mode=split_mode,
      penalty_policy=penalty_policy,
      max_matrix_dimension=max_matrix_dimension,
      use_gpu=False,
    )

  def reset_metrics(self) -> None:
    """Reset transfer and launch counters on evaluator, decoder, and penalty policy."""
    super().reset_metrics()
    self._reset_vrp_components()

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Warm up VRP decoder only when `context.problem.is_vrp`."""
    if context.problem.is_vrp:
      hook = getattr(self._decoder, "warmup", None)
      if callable(hook):
        hook(context)

  def evaluate_detailed_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool = False,
  ) -> SplitBatchResult:
    """Return full `SplitBatchResult` for Hamiltonian tours, Giant-Tours, or `0`-delimited routes."""
    batch_np, was_1d, _ = to_cpu_batch(routes)
    if not context.problem.is_vrp:
      costs = evaluate_tour_cpu(batch_np, context)
      routes_out: Any = restore_routes_shape(batch_np, was_1d, to_gpu=return_gpu)
      return build_tour_detailed_result(routes_out, costs, on_gpu=return_gpu)

    if self._check_vrp_width(batch_np.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      res = self._decoder.decode_batch(routes, context, return_gpu=return_gpu)
      self._absorb_decoder_metrics(before)
      return cast("SplitBatchResult", res)

    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)
    pen, unpen, viol, num_veh, _ = evaluate_delimited_cpu(
      batch_np, context, omega_q=omega_q, omega_l=omega_l, omega_k=omega_k
    )
    return build_cpu_vrp_detailed(batch_np, pen, unpen, viol, num_veh, was_1d=was_1d, return_gpu=return_gpu)

  def assess_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool = False,
  ) -> RouteAssessment:
    """Return polymorphic `RouteAssessment` on CPU (or uploaded to GPU when `return_gpu=True`)."""
    batch_np, _, _ = to_cpu_batch(routes)
    if not context.problem.is_vrp:
      costs = evaluate_tour_cpu(batch_np, context)
      return build_tour_assessment(costs, on_gpu=return_gpu)

    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)
    if self._check_vrp_width(batch_np.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      split_res = self._decoder.decode_batch(batch_np, context, return_gpu=False)
      self._absorb_decoder_metrics(before)
      delimited_np = np.asarray(split_res.routes, dtype=np.int32)
      _, _, _, _, comps = evaluate_delimited_cpu(
        delimited_np, context, omega_q=omega_q, omega_l=omega_l, omega_k=omega_k
      )
      pen = np.asarray(split_res.penalized_costs, dtype=np.float64)
      unpen = np.asarray(split_res.unpenalized_costs, dtype=np.float64)
      viol = np.asarray(split_res.violations, dtype=np.float64)
      num_veh = np.asarray(split_res.num_vehicles, dtype=np.int32)
    else:
      pen, unpen, viol, num_veh, comps = evaluate_delimited_cpu(
        batch_np, context, omega_q=omega_q, omega_l=omega_l, omega_k=omega_k
      )
    return build_cpu_vrp_assessment(pen, unpen, viol, num_veh, comps, context, return_gpu=return_gpu)

  @time_strategy
  def evaluate_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool = False,
  ) -> Fitness:
    """Compute `float64` fitness `(N,)` on CPU for TSP/ATSP or VRP routes."""
    batch_np, _, _ = to_cpu_batch(routes)
    if not context.problem.is_vrp:
      costs = evaluate_tour_cpu(batch_np, context)
      if return_gpu:
        cp = require_gpu()
        return cast("Fitness", cp.asarray(costs, dtype=cp.float64))
      return cast("Fitness", costs)

    if self._check_vrp_width(batch_np.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      res = self._decoder.decode_costs_only(routes, context, return_gpu=return_gpu)
      self._absorb_decoder_metrics(before)
      return cast("Fitness", res.penalized_costs)

    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)
    pen, _, _, _, _ = evaluate_delimited_cpu(batch_np, context, omega_q=omega_q, omega_l=omega_l, omega_k=omega_k)
    if return_gpu:
      cp = require_gpu()
      return cast("Fitness", cp.asarray(pen, dtype=cp.float64))
    return cast("Fitness", pen)


class EvaluationGPUCost(UnifiedCostMixin, EvaluationGPUBase):
  """Unified GPU CUDA cost evaluator for TSP/ATSP tours, VRP Giant-Tours, and `0`-delimited routes."""

  def __init__(
    self,
    return_gpu: bool = False,
    threads_per_block: int = 256,
    max_matrix_dimension: int | None = None,
    decoder: RouteDecoder | None = None,
    split_mode: Literal["prins", "greedy"] = "prins",
    penalty_policy: PenaltyPolicy | None = None,
  ) -> None:
    """Initialize unified GPU cost evaluator."""
    super().__init__(return_gpu=return_gpu, max_matrix_dimension=max_matrix_dimension)
    self.threads_per_block: int = prev_power_of_2(threads_per_block, min_val=32, max_val=256)
    self._delimited_kernel: Any = None
    self._init_vrp_components(
      decoder=decoder,
      split_mode=split_mode,
      penalty_policy=penalty_policy,
      max_matrix_dimension=max_matrix_dimension,
      use_gpu=True,
    )

  def reset_metrics(self) -> None:
    """Reset transfer and launch counters on evaluator, decoder, and penalty policy."""
    super().reset_metrics()
    self._reset_vrp_components()

  def _get_tour_kernel(self) -> Any:
    """Lazily compile and cache `tour_cost_kernel`."""
    if self._kernel is None:
      self._kernel = load_kernel("tour_cost.cu", "tour_cost_kernel", __file__)
    return self._kernel

  def _get_delimited_kernel(self) -> Any:
    """Lazily compile and cache `vrp_delimited_cost_kernel`."""
    if self._delimited_kernel is None:
      self._delimited_kernel = load_kernel("vrp_delimited_cost.cu", "vrp_delimited_cost_kernel", __file__)
    return self._delimited_kernel

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Warm up only the CUDA path matching `context.problem.is_vrp` (`BUG-24`, `BUG-25`)."""
    if not context.problem.is_vrp:
      self.evaluate_batch(self._warmup_batch(context, rows=1), context, return_gpu=True)
      return
    hook = getattr(self._decoder, "warmup", None)
    if callable(hook):
      hook(context)
    cp = require_gpu()
    delimited = np.zeros((1, context.dimension + 1), dtype=np.int32)
    delimited[0, 1 : context.dimension] = np.arange(1, context.dimension, dtype=np.int32)
    self.evaluate_batch(cp.asarray(delimited, dtype=cp.int32), context, return_gpu=True)

  def _get_distances_int32_gpu(self, context: ProblemContext | None = None, problem: Problem | None = None) -> Any:
    """Get the cached `int32` distance matrix on GPU for `tour_cost_kernel`."""
    if context is not None:
      return context.get_gpu_distances()
    if problem is None:
      if self._cached_distances_gpu is None:
        raise RuntimeError("A ProblemContext or Problem is required to obtain GPU distance matrices")
      return self._cached_distances_gpu

    problem_key = (problem.name, problem.dimension, problem.edge_type, id(problem))
    if self._cached_distances_gpu is None or self._cached_problem_key != problem_key:
      if problem.distances is not None:
        source: np.ndarray = problem.distances
      elif problem.coordinates is not None:
        from distances.matrix import compute_distance_matrix

        source = compute_distance_matrix(problem.coordinates, problem.edge_type)
      else:
        raise ValueError(f"Problem '{problem.name}' has no distances or coordinates")
      cp = require_gpu()
      self._cached_distances_gpu = cp.asarray(source, dtype=cp.int32)
      self._cached_problem_key = problem_key
    return self._cached_distances_gpu

  def _launch_tour_gpu(self, batch: Any, context: ProblemContext) -> tuple[Any, Any, bool]:
    d_pop, input_on_gpu = self._upload_routes_batch(batch)
    _, n = d_pop.shape
    d_distances, d_coords, edge_type_code = self._distance_operands(context, n)
    d_costs = launch_tour_cost_gpu(
      self._get_tour_kernel(),
      d_pop,
      d_distances,
      d_coords,
      edge_type_code=edge_type_code,
      threads_per_block=self.threads_per_block,
      return_to_depot=context.return_to_depot,
    )
    if d_pop.shape[0] > 0:
      self.kernel_launches += 1
    return d_pop, d_costs, input_on_gpu

  def _launch_delimited_gpu(
    self,
    d_routes: Any,
    context: ProblemContext,
    *,
    need_components: bool = False,
  ) -> tuple[Any, Any, Any, Any, Any]:
    cp = require_gpu()
    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    d_distances, d_coords, edge_type_code = self._distance_operands(context, context.dimension)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)
    d_pen, d_unpen, d_viol, d_veh, d_comps, launched = launch_delimited_cost_gpu(
      self._get_delimited_kernel(),
      d_routes,
      d_demands,
      d_distances,
      d_coords,
      context,
      edge_type_code=edge_type_code,
      omega_q=omega_q,
      omega_l=omega_l,
      omega_k=omega_k,
      need_components=need_components,
    )
    if launched:
      self.kernel_launches += 1
    return d_pen, d_unpen, d_viol, d_veh, d_comps

  def _eval_delimited_gpu(
    self,
    batch: Any,
    *,
    was_1d: bool,
    out_on_gpu: bool,
    context: ProblemContext,
    materialize_routes: bool,
  ) -> SplitBatchResult:
    from algorithms.strategies.decoding import SplitBatchResult

    d_routes, input_on_gpu = self._upload_routes_batch(batch)
    d_pen, d_unpen, d_viol, d_veh, _ = self._launch_delimited_gpu(d_routes, context, need_components=False)
    if out_on_gpu:
      routes_out = restore_routes_shape(d_routes, was_1d) if materialize_routes else None
      return SplitBatchResult(
        routes=routes_out,
        penalized_costs=d_pen,
        unpenalized_costs=d_unpen,
        violations=d_viol,
        num_vehicles=d_veh,
      )

    pen_host = d_pen.get()
    self.d2h_bytes += int(pen_host.nbytes)
    if not materialize_routes:
      return SplitBatchResult(
        routes=None, penalized_costs=pen_host, unpenalized_costs=None, violations=None, num_vehicles=None
      )

    unpen_host, viol_host, veh_host = d_unpen.get(), d_viol.get(), d_veh.get()
    self.d2h_bytes += int(unpen_host.nbytes + viol_host.nbytes + veh_host.nbytes)
    if input_on_gpu:
      routes_host = restore_routes_shape(d_routes.get(), was_1d)
      self.d2h_bytes += int(d_routes.nbytes)
    else:
      routes_host = restore_routes_shape(np.ascontiguousarray(batch, dtype=np.int32), was_1d)
    return SplitBatchResult(
      routes=routes_host,
      penalized_costs=pen_host,
      unpenalized_costs=unpen_host,
      violations=viol_host,
      num_vehicles=veh_host,
    )

  def evaluate_detailed_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool | None = None,
  ) -> SplitBatchResult:
    """Return full `SplitBatchResult` on GPU or CPU for TSP/ATSP or VRP routes."""
    require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    input_on_gpu = is_gpu_array(batch)
    out_on_gpu = (input_on_gpu and self.return_gpu) if return_gpu is None else return_gpu

    if not context.problem.is_vrp:
      d_pop, d_costs, _ = self._launch_tour_gpu(batch, context)
      costs = self._finalize_costs(d_costs, input_on_gpu=input_on_gpu, out_on_gpu=out_on_gpu)
      if out_on_gpu:
        routes_out: Any = restore_routes_shape(d_pop, was_1d)
      elif input_on_gpu:
        routes_out = restore_routes_shape(d_pop.get(), was_1d)
        self.d2h_bytes += int(d_pop.nbytes)
      else:
        routes_out = restore_routes_shape(np.ascontiguousarray(batch, dtype=np.int32), was_1d)
      return build_tour_detailed_result(routes_out, costs, on_gpu=out_on_gpu)

    if self._check_vrp_width(batch.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      res = self._decoder.decode_batch(routes, context, return_gpu=out_on_gpu)
      self._absorb_decoder_metrics(before)
      return cast("SplitBatchResult", res)

    return self._eval_delimited_gpu(
      batch, was_1d=was_1d, out_on_gpu=out_on_gpu, context=context, materialize_routes=True
    )

  def assess_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool = False,
  ) -> RouteAssessment:
    """Return polymorphic `RouteAssessment` on GPU or CPU (`d2h_bytes == 0` when `return_gpu=True`)."""
    require_gpu()
    batch, _ = ensure_routes_batch(routes)
    if not context.problem.is_vrp:
      _, d_costs, input_on_gpu = self._launch_tour_gpu(batch, context)
      costs = self._finalize_costs(d_costs, input_on_gpu=input_on_gpu, out_on_gpu=return_gpu)
      return build_tour_assessment(costs, on_gpu=return_gpu)

    if self._check_vrp_width(batch.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      split_res = self._decoder.decode_batch(routes, context, return_gpu=True)
      self._absorb_decoder_metrics(before)
      _, _, _, _, d_comps = self._launch_delimited_gpu(split_res.routes, context, need_components=True)
      d_pen, d_unpen = split_res.penalized_costs, split_res.unpenalized_costs
      d_viol, d_veh = split_res.violations, split_res.num_vehicles
    else:
      d_routes, _ = self._upload_routes_batch(batch)
      d_pen, d_unpen, d_viol, d_veh, d_comps = self._launch_delimited_gpu(d_routes, context, need_components=True)

    assessment, d2h = build_gpu_vrp_assessment(d_pen, d_unpen, d_viol, d_veh, d_comps, context, return_gpu=return_gpu)
    self.d2h_bytes += d2h
    return assessment

  @time_strategy
  def evaluate_batch(
    self,
    routes: RouteOrRoutes,
    context: ProblemContext,
    return_gpu: bool | None = None,
  ) -> Fitness:
    """Compute `float64` fitness `(N,)` on GPU with zero D2H when `return_gpu=True`."""
    require_gpu()
    batch, was_1d = ensure_routes_batch(routes)
    input_on_gpu = is_gpu_array(batch)
    out_on_gpu = (input_on_gpu and self.return_gpu) if return_gpu is None else return_gpu

    if not context.problem.is_vrp:
      _, d_costs, _ = self._launch_tour_gpu(batch, context)
      return self._finalize_costs(d_costs, input_on_gpu=input_on_gpu, out_on_gpu=out_on_gpu)

    if self._check_vrp_width(batch.shape[1], context.dimension):
      before = self._decoder_metrics_snapshot()
      res = self._decoder.decode_costs_only(routes, context, return_gpu=out_on_gpu)
      self._absorb_decoder_metrics(before)
      return cast("Fitness", res.penalized_costs)

    res = self._eval_delimited_gpu(
      batch, was_1d=was_1d, out_on_gpu=out_on_gpu, context=context, materialize_routes=False
    )
    return cast("Fitness", res.penalized_costs)


__all__: list[str] = ["EvaluationCPUCost", "EvaluationGPUCost"]
