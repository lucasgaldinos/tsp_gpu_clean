"""Shared helpers, `PenaltyPolicy`/`RouteDecoder` resolution, and O(1) CPU move evaluators (`DP17-7`, `DP17-9`)."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from algorithms.strategies.decoding import GreedySplitDecoder, PrinsSplitDecoder
from algorithms.strategies.penalties import InstanceScaledPenaltyPolicy
from protocols.vrp_strategies import PenaltyPolicy, RouteDecoder

type InterRouteCPUMode = Literal["best", "multi_move"]
"""Acceptance rule for `LocalSearchCPURouteTwoOptStar`."""

type InterRouteGPUMode = Literal["batch", "multi_move"]
"""Kernel family for `LocalSearchGPURouteTwoOptStar`."""

type VRPSplitMode = Literal["prins", "greedy"]
"""Split decoder used when refining `(N, dimension - 1)` Giant-Tours."""

#: Maximum problem dimension cached in Pascal `sm_61` 48 KB shared memory (~32.7 KB at 384).
DEFAULT_VRP_LS_MAX_SHARED_DIMENSION: int = 384

_CPU_MODES: frozenset[str] = frozenset({"best", "multi_move"})
_GPU_MODES: frozenset[str] = frozenset({"batch", "multi_move"})
_SPLIT_MODES: frozenset[str] = frozenset({"prins", "greedy"})
_DELTA_EPS: float = -1e-9

MOVE_RELOCATE: int = 1
MOVE_SWAP: int = 2
MOVE_TWO_OPT_STAR: int = 3


class RoutePenaltyKw(TypedDict):
  """Typed keyword bundle for `route_penalty`."""

  capacity: int | None
  max_route_distance: float | None
  omega_q: float
  omega_l: float


class RelocateEvalKw(TypedDict):
  """Typed keyword bundle for `eval_relocate_delta` and `eval_two_opt_star_delta`."""

  capacity: int | None
  max_route_distance: float | None
  max_vehicles: int | None
  omega_q: float
  omega_l: float
  omega_k: float


def resolve_inter_route_policies(
  *,
  split_mode: str,
  max_iterations: int,
  candidate_k: int | None,
  max_matrix_dimension: int | None,
  penalty_policy: PenaltyPolicy | None,
  decoder: RouteDecoder | None,
  use_gpu: bool,
) -> tuple[PenaltyPolicy, RouteDecoder]:
  """Validate common inter-route parameters and resolve shared `PenaltyPolicy` and `RouteDecoder`."""
  if isinstance(max_iterations, bool) or max_iterations < 0:
    raise ValueError(f"max_iterations must be >= 0, got {max_iterations!r}")
  if candidate_k is not None and (isinstance(candidate_k, bool) or candidate_k <= 0):
    raise ValueError(f"candidate_k must be > 0 or None, got {candidate_k!r}")
  if split_mode not in _SPLIT_MODES:
    raise ValueError(f"Invalid split_mode {split_mode!r}; expected one of {sorted(_SPLIT_MODES)}")
  if penalty_policy is not None and not isinstance(penalty_policy, PenaltyPolicy):
    raise TypeError(f"penalty_policy must implement PenaltyPolicy, got {type(penalty_policy).__name__}")
  if decoder is not None and not isinstance(decoder, RouteDecoder):
    raise TypeError(f"decoder must implement RouteDecoder, got {type(decoder).__name__}")

  resolved_policy: PenaltyPolicy = penalty_policy if penalty_policy is not None else InstanceScaledPenaltyPolicy()

  if decoder is not None:
    resolved_decoder: RouteDecoder = decoder
  else:
    dec_cls = PrinsSplitDecoder if split_mode == "prins" else GreedySplitDecoder
    resolved_decoder = dec_cls(
      use_gpu=use_gpu,
      max_matrix_dimension=max_matrix_dimension,
      penalty_policy=resolved_policy,
    )
  return resolved_policy, resolved_decoder


def route_penalty(
  load: int,
  dist: int,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  omega_q: float,
  omega_l: float,
) -> float:
  """Return IEEE-754-safe penalized route cost (never multiplying `0.0 * inf`)."""
  if load == 0 and dist == 0:
    return 0.0
  cap_pen = (omega_q * float(load - capacity)) if (capacity is not None and load > capacity) else 0.0
  dist_pen = (
    (omega_l * (float(dist) - max_route_distance))
    if (max_route_distance is not None and float(dist) > max_route_distance)
    else 0.0
  )
  return float(dist) + cap_pen + dist_pen


def extract_routes_from_delimited(row: np.ndarray) -> list[list[int]]:
  """Parse non-empty routes from a `0`-delimited `int32` row using vectorized NumPy slicing (`BUG-116`)."""
  arr = np.asarray(row, dtype=np.int32).ravel()
  if arr.size == 0:
    return []
  nz = arr > 0
  if not bool(np.any(nz)):
    return []
  padded = np.empty(arr.size + 2, dtype=bool)
  padded[0] = False
  padded[1:-1] = nz
  padded[-1] = False
  starts = np.flatnonzero(~padded[:-1] & padded[1:])
  ends = np.flatnonzero(padded[:-1] & ~padded[1:])
  return [[int(x) for x in arr[int(s) : int(e)]] for s, e in zip(starts, ends, strict=True)]


def pack_routes_to_delimited(routes: list[list[int]], width: int) -> np.ndarray:
  """Pack non-empty routes into a fixed-width `int32` array `[0, r_1..., 0, ..., 0]`."""
  out = np.zeros(width, dtype=np.int32)
  non_empty = [r for r in routes if r]
  if not non_empty:
    return out
  pos = 1
  for route in non_empty:
    for node in route:
      if pos >= width - 1:
        break
      out[pos] = node
      pos += 1
    if pos < width:
      pos += 1
  return out


class RouteState:
  """Auxiliary O(1) prefix/suffix state over non-empty routes for one individual."""

  def __init__(
    self,
    routes: list[list[int]],
    distances: np.ndarray,
    demands: np.ndarray,
    dimension: int,
    *,
    return_to_depot: bool,
  ) -> None:
    self.routes: list[list[int]] = [list(r) for r in routes if r]
    self.distances: np.ndarray = distances
    self.demands: np.ndarray = demands
    self.dimension: int = dimension
    self.return_to_depot: bool = return_to_depot

    self.route_loads: list[int] = []
    self.route_dists: list[int] = []
    self.prefix_loads: list[list[int]] = []
    self.prefix_dists: list[list[int]] = []
    self.suffix_dists: list[list[int]] = []
    self.cust_route: list[int] = [-1] * dimension
    self.cust_pos: list[int] = [-1] * dimension
    self.rebuild()

  def rebuild(self) -> None:
    """Recompute route loads, distances, prefix/suffix tables, and customer lookup maps."""
    self.routes = [r for r in self.routes if r]
    self.route_loads.clear()
    self.route_dists.clear()
    self.prefix_loads.clear()
    self.prefix_dists.clear()
    self.suffix_dists.clear()
    for i in range(self.dimension):
      self.cust_route[i] = -1
      self.cust_pos[i] = -1

    dist = self.distances
    dem = self.demands
    closed = self.return_to_depot

    for r_idx, route in enumerate(self.routes):
      s = len(route)
      p_load = [0] * s
      p_dist = [0] * s
      s_dist = [0] * s
      acc_load = 0
      acc_dist = 0
      for pos, node in enumerate(route):
        self.cust_route[node] = r_idx
        self.cust_pos[node] = pos
        acc_load += int(dem[node])
        acc_dist += int(dist[0, node]) if pos == 0 else int(dist[route[pos - 1], node])
        p_load[pos] = acc_load
        p_dist[pos] = acc_dist

      tail_dist = int(dist[route[s - 1], 0]) if closed else 0
      s_dist[s - 1] = tail_dist
      for pos in range(s - 2, -1, -1):
        tail_dist += int(dist[route[pos], route[pos + 1]])
        s_dist[pos] = tail_dist

      self.route_loads.append(acc_load)
      self.route_dists.append(acc_dist + (int(dist[route[s - 1], 0]) if closed else 0))
      self.prefix_loads.append(p_load)
      self.prefix_dists.append(p_dist)
      self.suffix_dists.append(s_dist)


def eval_relocate_delta(
  state: RouteState,
  r_a: int,
  pos_v: int,
  r_b: int,
  ins_pos: int,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  max_vehicles: int | None,
  omega_q: float,
  omega_l: float,
  omega_k: float,
) -> float:
  """Return exact O(1) penalized delta for moving `r_a[pos_v]` into `r_b` before `ins_pos`."""
  dist, dem, closed = state.distances, state.demands, state.return_to_depot
  route_a, route_b = state.routes[r_a], state.routes[r_b]
  s_a, s_b = len(route_a), len(route_b)
  v = route_a[pos_v]
  q_v = int(dem[v])

  old_load_a, old_dist_a = state.route_loads[r_a], state.route_dists[r_a]
  old_load_b, old_dist_b = state.route_loads[r_b], state.route_dists[r_b]
  new_load_a = old_load_a - q_v
  delta_m = 0
  if s_a == 1:
    new_dist_a = 0
    delta_m = -1
  else:
    prev_v = 0 if pos_v == 0 else route_a[pos_v - 1]
    rem_a = int(dist[prev_v, v])
    if pos_v < s_a - 1:
      next_v = route_a[pos_v + 1]
      rem_a += int(dist[v, next_v])
      add_a = int(dist[prev_v, next_v])
    elif closed:
      rem_a += int(dist[v, 0])
      add_a = int(dist[prev_v, 0])
    else:
      add_a = 0
    new_dist_a = old_dist_a - rem_a + add_a

  new_load_b = old_load_b + q_v
  prev_b = 0 if ins_pos == 0 else route_b[ins_pos - 1]
  if ins_pos < s_b:
    next_b = route_b[ins_pos]
    new_dist_b = old_dist_b - int(dist[prev_b, next_b]) + int(dist[prev_b, v]) + int(dist[v, next_b])
  elif closed:
    new_dist_b = old_dist_b - int(dist[prev_b, 0]) + int(dist[prev_b, v]) + int(dist[v, 0])
  else:
    new_dist_b = old_dist_b + int(dist[prev_b, v])

  kw: RoutePenaltyKw = {
    "capacity": capacity,
    "max_route_distance": max_route_distance,
    "omega_q": omega_q,
    "omega_l": omega_l,
  }
  old_pen = route_penalty(old_load_a, old_dist_a, **kw) + route_penalty(old_load_b, old_dist_b, **kw)
  new_pen = route_penalty(new_load_a, new_dist_a, **kw) + route_penalty(new_load_b, new_dist_b, **kw)
  if max_vehicles is not None and delta_m != 0:
    m = len(state.routes)
    old_ex = max(0, m - max_vehicles)
    new_ex = max(0, m + delta_m - max_vehicles)
    if old_ex > 0:
      old_pen += omega_k * float(old_ex)
    if new_ex > 0:
      new_pen += omega_k * float(new_ex)
  return new_pen - old_pen


def eval_swap_delta(
  state: RouteState,
  r_a: int,
  pos_v: int,
  r_b: int,
  pos_u: int,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  omega_q: float,
  omega_l: float,
) -> float:
  """Return exact O(1) penalized delta for exchanging `r_a[pos_v]` with `r_b[pos_u]`."""
  dist, dem, closed = state.distances, state.demands, state.return_to_depot
  route_a, route_b = state.routes[r_a], state.routes[r_b]
  s_a, s_b = len(route_a), len(route_b)
  v, u = route_a[pos_v], route_b[pos_u]

  old_load_a, old_dist_a = state.route_loads[r_a], state.route_dists[r_a]
  old_load_b, old_dist_b = state.route_loads[r_b], state.route_dists[r_b]
  new_load_a = old_load_a - int(dem[v]) + int(dem[u])
  new_load_b = old_load_b - int(dem[u]) + int(dem[v])

  prev_v = 0 if pos_v == 0 else route_a[pos_v - 1]
  rem_a, add_a = int(dist[prev_v, v]), int(dist[prev_v, u])
  if pos_v < s_a - 1:
    next_v = route_a[pos_v + 1]
    rem_a += int(dist[v, next_v])
    add_a += int(dist[u, next_v])
  elif closed:
    rem_a += int(dist[v, 0])
    add_a += int(dist[u, 0])

  prev_u = 0 if pos_u == 0 else route_b[pos_u - 1]
  rem_b, add_b = int(dist[prev_u, u]), int(dist[prev_u, v])
  if pos_u < s_b - 1:
    next_u = route_b[pos_u + 1]
    rem_b += int(dist[u, next_u])
    add_b += int(dist[v, next_u])
  elif closed:
    rem_b += int(dist[u, 0])
    add_b += int(dist[v, 0])

  kw: RoutePenaltyKw = {
    "capacity": capacity,
    "max_route_distance": max_route_distance,
    "omega_q": omega_q,
    "omega_l": omega_l,
  }
  old_pen = route_penalty(old_load_a, old_dist_a, **kw) + route_penalty(old_load_b, old_dist_b, **kw)
  new_pen = route_penalty(new_load_a, old_dist_a - rem_a + add_a, **kw) + route_penalty(
    new_load_b, old_dist_b - rem_b + add_b, **kw
  )
  return new_pen - old_pen


def eval_two_opt_star_delta(
  state: RouteState,
  r_a: int,
  cut_a: int,
  r_b: int,
  cut_b: int,
  *,
  capacity: int | None,
  max_route_distance: float | None,
  max_vehicles: int | None,
  omega_q: float,
  omega_l: float,
  omega_k: float,
) -> float:
  """Return exact O(1) penalized delta for inter-route 2-opt* tail exchange (Potvin & Rousseau, 1995)."""
  route_a, route_b = state.routes[r_a], state.routes[r_b]
  s_a, s_b = len(route_a), len(route_b)
  if (cut_a == -1 and cut_b == -1) or (cut_a == s_a - 1 and cut_b == s_b - 1):
    return 0.0

  dist, closed = state.distances, state.return_to_depot
  old_load_a, old_dist_a = state.route_loads[r_a], state.route_dists[r_a]
  old_load_b, old_dist_b = state.route_loads[r_b], state.route_dists[r_b]

  head_load_a = 0 if cut_a < 0 else state.prefix_loads[r_a][cut_a]
  head_open_a = 0 if cut_a < 0 else state.prefix_dists[r_a][cut_a]
  tail_load_a = old_load_a - head_load_a
  tail_close_a = 0 if cut_a >= s_a - 1 else state.suffix_dists[r_a][cut_a + 1]

  head_load_b = 0 if cut_b < 0 else state.prefix_loads[r_b][cut_b]
  head_open_b = 0 if cut_b < 0 else state.prefix_dists[r_b][cut_b]
  tail_load_b = old_load_b - head_load_b
  tail_close_b = 0 if cut_b >= s_b - 1 else state.suffix_dists[r_b][cut_b + 1]

  delta_m = 0
  if cut_a < 0 and cut_b >= s_b - 1:
    new_dist_a = 0
    delta_m -= 1
  elif cut_a < 0:
    new_dist_a = int(dist[0, route_b[cut_b + 1]]) + tail_close_b
  elif cut_b >= s_b - 1:
    new_dist_a = head_open_a + (int(dist[route_a[cut_a], 0]) if closed else 0)
  else:
    new_dist_a = head_open_a + int(dist[route_a[cut_a], route_b[cut_b + 1]]) + tail_close_b

  if cut_b < 0 and cut_a >= s_a - 1:
    new_dist_b = 0
    delta_m -= 1
  elif cut_b < 0:
    new_dist_b = int(dist[0, route_a[cut_a + 1]]) + tail_close_a
  elif cut_a >= s_a - 1:
    new_dist_b = head_open_b + (int(dist[route_b[cut_b], 0]) if closed else 0)
  else:
    new_dist_b = head_open_b + int(dist[route_b[cut_b], route_a[cut_a + 1]]) + tail_close_a

  kw: RoutePenaltyKw = {
    "capacity": capacity,
    "max_route_distance": max_route_distance,
    "omega_q": omega_q,
    "omega_l": omega_l,
  }
  old_pen = route_penalty(old_load_a, old_dist_a, **kw) + route_penalty(old_load_b, old_dist_b, **kw)
  new_pen = route_penalty(head_load_a + tail_load_b, new_dist_a, **kw) + route_penalty(
    head_load_b + tail_load_a, new_dist_b, **kw
  )
  if max_vehicles is not None and delta_m != 0:
    m = len(state.routes)
    old_ex = max(0, m - max_vehicles)
    new_ex = max(0, m + delta_m - max_vehicles)
    if old_ex > 0:
      old_pen += omega_k * float(old_ex)
    if new_ex > 0:
      new_pen += omega_k * float(new_ex)
  return new_pen - old_pen


def apply_move_to_routes(routes: list[list[int]], move_type: int, r_a: int, arg1: int, r_b: int, arg2: int) -> None:
  """Apply a validated inter-route move `(move_type, r_a, arg1, r_b, arg2)` in-place."""
  route_a, route_b = routes[r_a], routes[r_b]
  if move_type == MOVE_RELOCATE:
    route_b.insert(arg2, route_a.pop(arg1))
  elif move_type == MOVE_SWAP:
    route_a[arg1], route_b[arg2] = route_b[arg2], route_a[arg1]
  elif move_type == MOVE_TWO_OPT_STAR:
    routes[r_a] = route_a[: arg1 + 1] + route_b[arg2 + 1 :]
    routes[r_b] = route_b[: arg2 + 1] + route_a[arg1 + 1 :]
  else:
    raise ValueError(f"Unknown inter-route move_type: {move_type}")


def intra_route_two_opt_cpu(
  route: list[int],
  distances: np.ndarray,
  *,
  return_to_depot: bool,
  max_route_distance: float | None,
  omega_l: float,
  max_iterations: int,
) -> list[int]:
  """Refine a single non-empty route via steepest-descent intra-route 2-opt."""
  s = len(route)
  if s < 2 or max_iterations <= 0:
    return route
  work = list(route)
  for _ in range(max_iterations):
    cur_dist = int(distances[0, work[0]]) + sum(int(distances[work[k], work[k + 1]]) for k in range(s - 1))
    if return_to_depot:
      cur_dist += int(distances[work[s - 1], 0])

    best_pen_delta = _DELTA_EPS
    best_i = -1
    best_j = -1
    full = [0, *work, 0] if return_to_depot else [0, *work]
    n_full = len(full)
    j_limit = (n_full - 1) if return_to_depot else n_full
    i_limit = (n_full - 3) if return_to_depot else (n_full - 2)

    for i in range(i_limit):
      a, b = full[i], full[i + 1]
      d_ab = int(distances[a, b])
      for j in range(i + 2, j_limit):
        c = full[j]
        if not return_to_depot and j == n_full - 1:
          delta_d = int(distances[a, c]) - d_ab
        else:
          d = full[j + 1]
          delta_d = (int(distances[a, c]) + int(distances[b, d])) - (d_ab + int(distances[c, d]))
        if delta_d < 0:
          pen_delta = float(delta_d)
          if max_route_distance is not None:
            old_v = max(0.0, float(cur_dist) - max_route_distance)
            new_v = max(0.0, float(cur_dist + delta_d) - max_route_distance)
            old_p = (omega_l * old_v) if old_v > 0.0 else 0.0
            new_p = (omega_l * new_v) if new_v > 0.0 else 0.0
            pen_delta += new_p - old_p
          if pen_delta < best_pen_delta:
            best_pen_delta, best_i, best_j = pen_delta, i, j
    if best_i < 0:
      break
    work[best_i:best_j] = work[best_i:best_j][::-1]
  return work
