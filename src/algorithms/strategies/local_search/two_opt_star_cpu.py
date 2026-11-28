"""CPU VRP inter-route and intra-route local search (`LocalSearchCPURouteTwoOptStar`, `DP17-7`, `DP17-9`).

Supports `Relocate(1, 0)`, `Swap(1, 1)`, `2-opt*` (Potvin & Rousseau, 1995), and intra-route
`2-opt` across `(N, dimension - 1)` Giant-Tours and `(N, W)` `0`-delimited VRP routes while
honoring `capacity`, `max_route_distance`, `max_vehicles`, and `return_to_depot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.decoding import strip_depot_delimiters
from protocols.algorithm_strategies import LocalSearchStrategy
from utils.phase_timer import time_strategy
from utils.route_helpers import restore_routes_shape, to_cpu_batch

from .two_opt_star_helpers import (
  _CPU_MODES,
  _DELTA_EPS,
  MOVE_RELOCATE,
  MOVE_SWAP,
  MOVE_TWO_OPT_STAR,
  InterRouteCPUMode,
  RelocateEvalKw,
  RoutePenaltyKw,
  RouteState,
  VRPSplitMode,
  apply_move_to_routes,
  eval_relocate_delta,
  eval_swap_delta,
  eval_two_opt_star_delta,
  extract_routes_from_delimited,
  intra_route_two_opt_cpu,
  pack_routes_to_delimited,
  resolve_inter_route_policies,
)

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import Route, RouteOrRoutes
  from protocols.vrp_strategies import PenaltyPolicy, RouteDecoder


class LocalSearchCPURouteTwoOptStar(StrategyBase, LocalSearchStrategy):
  """CPU VRP local search oracle (`Relocate`, `Swap`, `2-opt*` + intra-route `2-opt`)."""

  def __init__(
    self,
    mode: InterRouteCPUMode = "best",
    max_iterations: int = 50,
    candidate_k: int | None = 20,
    split_mode: VRPSplitMode = "prins",
    intra_two_opt: bool = True,
    max_matrix_dimension: int | None = None,
    penalty_policy: PenaltyPolicy | None = None,
    decoder: RouteDecoder | None = None,
  ) -> None:
    super().__init__()
    if mode not in _CPU_MODES:
      raise ValueError(f"Invalid VRP CPU local search mode {mode!r}; expected one of {sorted(_CPU_MODES)}")
    self.mode: InterRouteCPUMode = mode
    self.max_iterations: int = max_iterations
    self.candidate_k: int | None = candidate_k
    self.split_mode: VRPSplitMode = split_mode
    self.intra_two_opt: bool = intra_two_opt
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.penalty_policy, self._decoder = resolve_inter_route_policies(
      split_mode=split_mode,
      max_iterations=max_iterations,
      candidate_k=candidate_k,
      max_matrix_dimension=max_matrix_dimension,
      penalty_policy=penalty_policy,
      decoder=decoder,
      use_gpu=False,
    )

  def reset_metrics(self) -> None:
    """Reset transfer counters and underlying decoder / penalty policy state."""
    super().reset_metrics()
    for obj in (self._decoder, self.penalty_policy):
      hook = getattr(obj, "reset_metrics", None)
      if callable(hook):
        hook()

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Warm up attached decoder without mutating counters or RNG."""
    super()._warmup_impl(context)
    hook = getattr(self._decoder, "warmup", None)
    if callable(hook):
      hook(context)

  def improve(self, route: Route, context: ProblemContext) -> Route:
    """Improve a single 1D Giant-Tour `(dimension - 1,)` or `0`-delimited `(W,)` route."""
    return self.improve_batch(route, context)

  def _enumerate_candidates_for_customer(
    self,
    state: RouteState,
    r_a: int,
    pos_v: int,
    candidates: np.ndarray | None,
    *,
    capacity: int | None,
    max_route_distance: float | None,
    max_vehicles: int | None,
    omega_q: float,
    omega_l: float,
    omega_k: float,
  ) -> tuple[float, int, int, int, int]:
    """Find the steepest improving inter-route move anchored at `(r_a, pos_v)`."""
    v = state.routes[r_a][pos_v]
    m = len(state.routes)
    best_delta = _DELTA_EPS
    best_type = 0
    best_arg1 = -1
    best_rb = -1
    best_arg2 = -1

    def consider(move_type: int, rb: int, arg1: int, arg2: int, delta: float) -> None:
      nonlocal best_delta, best_type, best_arg1, best_rb, best_arg2
      if delta < best_delta:
        best_delta, best_type, best_arg1, best_rb, best_arg2 = delta, move_type, arg1, rb, arg2

    rel_kw: RelocateEvalKw = {
      "capacity": capacity,
      "max_route_distance": max_route_distance,
      "max_vehicles": max_vehicles,
      "omega_q": omega_q,
      "omega_l": omega_l,
      "omega_k": omega_k,
    }
    sw_kw: RoutePenaltyKw = {
      "capacity": capacity,
      "max_route_distance": max_route_distance,
      "omega_q": omega_q,
      "omega_l": omega_l,
    }

    if candidates is not None:
      row_c = candidates[v]
      for idx in range(int(row_c.shape[0])):
        u = int(row_c[idx])
        if u == 0:
          for r_b in range(m):
            if r_b == r_a:
              continue
            consider(MOVE_RELOCATE, r_b, pos_v, 0, eval_relocate_delta(state, r_a, pos_v, r_b, 0, **rel_kw))
            consider(
              MOVE_TWO_OPT_STAR,
              r_b,
              pos_v,
              -1,
              eval_two_opt_star_delta(state, r_a, pos_v, r_b, -1, **rel_kw),
            )
          continue
        if u < 0 or u >= state.dimension:
          continue
        r_b = state.cust_route[u]
        if r_b < 0 or r_b == r_a:
          continue
        pos_u = state.cust_pos[u]
        consider(
          MOVE_RELOCATE,
          r_b,
          pos_v,
          pos_u + 1,
          eval_relocate_delta(state, r_a, pos_v, r_b, pos_u + 1, **rel_kw),
        )
        if pos_u == 0:
          consider(MOVE_RELOCATE, r_b, pos_v, 0, eval_relocate_delta(state, r_a, pos_v, r_b, 0, **rel_kw))
          consider(
            MOVE_TWO_OPT_STAR,
            r_b,
            pos_v,
            -1,
            eval_two_opt_star_delta(state, r_a, pos_v, r_b, -1, **rel_kw),
          )
        consider(MOVE_SWAP, r_b, pos_v, pos_u, eval_swap_delta(state, r_a, pos_v, r_b, pos_u, **sw_kw))
        consider(
          MOVE_TWO_OPT_STAR,
          r_b,
          pos_v,
          pos_u,
          eval_two_opt_star_delta(state, r_a, pos_v, r_b, pos_u, **rel_kw),
        )
        if pos_v == 0:
          consider(
            MOVE_TWO_OPT_STAR,
            r_b,
            -1,
            pos_u,
            eval_two_opt_star_delta(state, r_a, -1, r_b, pos_u, **rel_kw),
          )
      return best_delta, best_type, best_arg1, best_rb, best_arg2

    for r_b in range(m):
      if r_b == r_a:
        continue
      s_b = len(state.routes[r_b])
      consider(MOVE_RELOCATE, r_b, pos_v, 0, eval_relocate_delta(state, r_a, pos_v, r_b, 0, **rel_kw))
      consider(
        MOVE_TWO_OPT_STAR,
        r_b,
        pos_v,
        -1,
        eval_two_opt_star_delta(state, r_a, pos_v, r_b, -1, **rel_kw),
      )
      for pos_u in range(s_b):
        consider(
          MOVE_RELOCATE,
          r_b,
          pos_v,
          pos_u + 1,
          eval_relocate_delta(state, r_a, pos_v, r_b, pos_u + 1, **rel_kw),
        )
        if r_a < r_b:
          consider(MOVE_SWAP, r_b, pos_v, pos_u, eval_swap_delta(state, r_a, pos_v, r_b, pos_u, **sw_kw))
          consider(
            MOVE_TWO_OPT_STAR,
            r_b,
            pos_v,
            pos_u,
            eval_two_opt_star_delta(state, r_a, pos_v, r_b, pos_u, **rel_kw),
          )
    return best_delta, best_type, best_arg1, best_rb, best_arg2

  def _refine_individual(
    self,
    row: np.ndarray,
    distances: np.ndarray,
    demands: np.ndarray,
    candidates: np.ndarray | None,
    context: ProblemContext,
    omega_q: float,
    omega_l: float,
    omega_k: float,
  ) -> np.ndarray:
    """Run inter-route (`Relocate`, `Swap`, `2-opt*`) + optional intra-route `2-opt` on 1 row."""
    width = int(row.size)
    routes = extract_routes_from_delimited(row)
    if not routes:
      return row.copy()

    constraints = context.constraints
    capacity = constraints.capacity
    max_dist = constraints.max_route_distance
    max_veh = constraints.max_vehicles
    closed = constraints.return_to_depot
    state = RouteState(routes, distances, demands, context.dimension, return_to_depot=closed)

    for _ in range(self.max_iterations):
      m = len(state.routes)
      if m < 2:
        break
      if self.mode == "best":
        best_delta = _DELTA_EPS
        best_move: tuple[int, int, int, int, int] | None = None
        for r_a in range(m):
          for pos_v in range(len(state.routes[r_a])):
            delta, mtype, arg1, r_b, arg2 = self._enumerate_candidates_for_customer(
              state,
              r_a,
              pos_v,
              candidates,
              capacity=capacity,
              max_route_distance=max_dist,
              max_vehicles=max_veh,
              omega_q=omega_q,
              omega_l=omega_l,
              omega_k=omega_k,
            )
            if delta < best_delta:
              best_delta = delta
              best_move = (mtype, r_a, arg1, r_b, arg2)
        if best_move is None:
          break
        apply_move_to_routes(state.routes, *best_move)
        state.rebuild()
      else:
        pool: list[tuple[float, int, int, int, int, int]] = []
        for r_a in range(m):
          for pos_v in range(len(state.routes[r_a])):
            delta, mtype, arg1, r_b, arg2 = self._enumerate_candidates_for_customer(
              state,
              r_a,
              pos_v,
              candidates,
              capacity=capacity,
              max_route_distance=max_dist,
              max_vehicles=max_veh,
              omega_q=omega_q,
              omega_l=omega_l,
              omega_k=omega_k,
            )
            if mtype != 0 and delta < _DELTA_EPS:
              pool.append((delta, mtype, r_a, arg1, r_b, arg2))
        if not pool:
          break
        pool.sort(key=lambda item: item[0])
        used_routes: set[int] = set()
        applied_any = False
        for _d, mtype, r_a, arg1, r_b, arg2 in pool:
          if r_a in used_routes or r_b in used_routes:
            continue
          used_routes.add(r_a)
          used_routes.add(r_b)
          apply_move_to_routes(state.routes, mtype, r_a, arg1, r_b, arg2)
          applied_any = True
        if not applied_any:
          break
        state.rebuild()

    if self.intra_two_opt and self.max_iterations > 0:
      state.routes = [
        intra_route_two_opt_cpu(
          r,
          distances,
          return_to_depot=closed,
          max_route_distance=max_dist,
          omega_l=omega_l,
          max_iterations=self.max_iterations,
        )
        for r in state.routes
        if r
      ]
    return pack_routes_to_delimited(state.routes, width)

  @time_strategy
  def improve_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Improve a 1D route or 2D batch on CPU, supporting Giant-Tour and `0`-delimited formats."""
    batch_np, was_1d, was_gpu = to_cpu_batch(routes, copy=True)
    _, width = batch_np.shape
    if context.dimension <= 2 or self.max_iterations <= 0:
      return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

    n_cust = context.dimension - 1
    is_giant_tour = width == n_cust
    if is_giant_tour:
      delimited = np.asarray(
        self._decoder.decode_batch(batch_np, context, return_gpu=False).routes,
        dtype=np.int32,
      )
    else:
      delimited = batch_np

    distances = context.get_cpu_distances()
    demands = context.get_cpu_demands()
    if demands is None:
      demands = np.zeros(context.dimension, dtype=np.int32)
    candidates = context.get_cpu_candidate_list(self.candidate_k) if self.candidate_k is not None else None
    omega_q, omega_l, omega_k = self.penalty_policy.weights(context)

    refined = np.empty_like(delimited, dtype=np.int32)
    for idx in range(len(delimited)):
      refined[idx] = self._refine_individual(
        delimited[idx], distances, demands, candidates, context, omega_q, omega_l, omega_k
      )

    out_np = strip_depot_delimiters(refined, n_cust) if is_giant_tour else refined
    return restore_routes_shape(out_np, was_1d, to_gpu=was_gpu)


__all__: list[str] = ["LocalSearchCPURouteTwoOptStar"]
