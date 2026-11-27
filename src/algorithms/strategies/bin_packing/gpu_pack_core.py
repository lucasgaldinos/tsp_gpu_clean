"""VRAM-native CUDA launcher and workspace manager for FFD/BFD fleet repacking (`Task 17.30`, `BUG-117`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from algorithms.strategies.base import StrategyBase
from algorithms.strategies.local_search.gpu_base import DistanceOperandCache, ensure_gpu_workspace
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.route_helpers import ensure_routes_batch, is_gpu_array, restore_routes_shape

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import RouteOrRoutes, SeedLike

#: Default shared-memory node ceiling keeping per-block dynamic smem <= 44 KB on `sm_61`.
DEFAULT_BIN_PACK_MAX_SHARED_DIMENSION: int = 768

#: Maximum dynamic shared memory per block (44 KB leaves 4 KB headroom under `sm_61` 48 KB limit).
_MAX_SHARED_MEM_BYTES: int = 44 * 1024


def _bin_pack_shared_mem_bytes(cap_c: int) -> int:
  """Return dynamic shared-memory bytes required by ``bin_packing_pack_kernel``."""
  return 56 * cap_c + 16


class _BaseGPUBinPacking(StrategyBase):
  """Shared CUDA base class for ``BinPackingGPUFFD`` (``mode=0``) and ``BinPackingGPUBFD`` (``mode=1``)."""

  _mode: ClassVar[int] = 0

  def __init__(
    self,
    max_matrix_dimension: int | None = None,
    max_shared_dimension: int = DEFAULT_BIN_PACK_MAX_SHARED_DIMENSION,
    return_gpu: bool | None = None,
    seed: SeedLike = None,
  ) -> None:
    if max_matrix_dimension is not None and (isinstance(max_matrix_dimension, bool) or max_matrix_dimension < 0):
      raise ValueError(f"max_matrix_dimension must be >= 0 or None, got {max_matrix_dimension!r}")
    if isinstance(max_shared_dimension, bool) or max_shared_dimension < 0:
      raise ValueError(f"max_shared_dimension must be >= 0, got {max_shared_dimension!r}")
    super().__init__(seed=seed)
    self.max_matrix_dimension: int | None = max_matrix_dimension
    self.max_shared_dimension: int = max_shared_dimension
    self.return_gpu: bool | None = return_gpu
    self._kernel: Any = None
    self._operands: DistanceOperandCache = DistanceOperandCache()
    self._ws_f64: Any = None
    self._ws_i64: Any = None
    self._ws_i32: Any = None

  def reset_metrics(self) -> None:
    """Reset transfer/kernel counters, mirrored operand cache, and GPU workspaces."""
    super().reset_metrics()
    self._operands.reset()
    self._ws_f64 = None
    self._ws_i64 = None
    self._ws_i32 = None

  def _get_kernel(self) -> Any:
    """Lazily compile and cache ``bin_packing_pack_kernel`` via NVRTC."""
    if self._kernel is None:
      self._kernel = load_kernel("bin_packing.cu", "bin_packing_pack_kernel", __file__)
    return self._kernel

  def _ensure_workspace(
    self,
    batch_size: int,
    cap_c: int,
  ) -> tuple[Any, Any, Any]:
    """Allocate/reuse monotonic global-memory workspaces when shared memory is disabled."""
    cp = require_gpu()
    need_f64 = batch_size * cap_c
    need_i64 = batch_size * 2 * cap_c
    need_i32 = batch_size * (8 * cap_c + 1)
    self._ws_f64 = ensure_gpu_workspace(self._ws_f64, (need_f64,), cp.float64)
    self._ws_i64 = ensure_gpu_workspace(self._ws_i64, (need_i64,), cp.int64)
    self._ws_i32 = ensure_gpu_workspace(self._ws_i32, (need_i32,), cp.int32)
    return self._ws_f64[:need_f64], self._ws_i64[:need_i64], self._ws_i32[:need_i32]

  def _resolve_operands(self, context: ProblemContext) -> tuple[Any, Any, Any, int]:
    """Bind context-owned demands and distance/coordinate operands, charging H2D delta once."""
    cp = require_gpu()
    before_h2d = int(getattr(context, "h2d_bytes", 0))
    d_demands = context.get_gpu_demands()
    self.h2d_bytes += max(0, int(getattr(context, "h2d_bytes", 0)) - before_h2d)
    if d_demands is None:
      d_demands = cp.zeros(context.dimension, dtype=cp.int32)

    operands, uploaded_bytes = self._operands.resolve(context, context.dimension, self.max_matrix_dimension)
    self.h2d_bytes += uploaded_bytes
    d_distances, d_coords, edge_type_code = operands
    return d_demands, d_distances, d_coords, edge_type_code

  def _warmup_impl(self, context: ProblemContext) -> None:
    """Compile and launch ``bin_packing_pack_kernel`` ahead of measured runs without mutating state."""
    require_gpu()
    super()._warmup_impl(context)
    self._get_kernel()
    if context.dimension <= 1 or not context.problem.is_vrp:
      return
    cp = require_gpu()
    self._resolve_operands(context)
    n_cust = context.dimension - 1
    probe = cp.zeros((1, 2 * n_cust + 1), dtype=cp.int32)
    probe[0, 1::2] = cp.arange(1, context.dimension, dtype=cp.int32)
    self.pack_batch(probe, context)

  @time_strategy
  def pack_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Repack violating ``0``-delimited VRP rows directly in VRAM with zero D2H sync."""
    constraints = context.constraints
    if not context.problem.is_vrp or (constraints.max_vehicles is None and constraints.capacity is None):
      return routes

    batch, was_1d = ensure_routes_batch(routes)
    batch_size = int(batch.shape[0])
    width = int(batch.shape[1]) if batch_size > 0 else 0
    if batch_size == 0 or width < context.dimension:
      return routes

    cp = require_gpu()
    input_on_gpu = is_gpu_array(batch)
    out_on_gpu = self.return_gpu if self.return_gpu is not None else input_on_gpu

    if input_on_gpu:
      d_in = cp.ascontiguousarray(batch, dtype=cp.int32)
    else:
      host_in = np.ascontiguousarray(batch, dtype=np.int32)
      d_in = cp.asarray(host_in, dtype=cp.int32)
      self.h2d_bytes += int(host_in.nbytes)

    d_demands, d_distances, d_coords, edge_type_code = self._resolve_operands(context)

    max_vehicles = constraints.max_vehicles
    min_needed = context.dimension + (max_vehicles if max_vehicles is not None else max(1, context.dimension - 1))
    out_width = max(width, min_needed)

    if out_width == width:
      d_out = d_in.copy()
    else:
      d_out = cp.zeros((batch_size, out_width), dtype=cp.int32)
      d_out[:, :width] = d_in

    n_cust = max(1, context.dimension - 1)
    req_smem = _bin_pack_shared_mem_bytes(n_cust)
    use_shared = context.dimension <= self.max_shared_dimension and req_smem <= _MAX_SHARED_MEM_BYTES
    if use_shared:
      cap_c = n_cust
      shared_mem = req_smem
      ws_f64: Any = 0
      ws_i64: Any = 0
      ws_i32: Any = 0
      grid = (batch_size,)
      block = (32,)
    else:
      cap_c = max(n_cust, width)
      shared_mem = 0
      ws_f64, ws_i64, ws_i32 = self._ensure_workspace(batch_size, cap_c)
      grid = ((batch_size + 3) // 4,)
      block = (128,)

    kernel = self._get_kernel()
    kernel(
      grid,
      block,
      (
        d_in,
        d_demands,
        d_distances,
        d_coords,
        np.int32(batch_size),
        np.int32(width),
        np.int32(out_width),
        np.int32(context.dimension),
        np.int32(edge_type_code),
        np.int32(self._mode),
        np.int32(1 if constraints.return_to_depot else 0),
        np.int32(1 if constraints.capacity is not None else 0),
        np.int64(constraints.capacity if constraints.capacity is not None else 0),
        np.int32(1 if constraints.max_route_distance is not None else 0),
        np.float64(constraints.max_route_distance if constraints.max_route_distance is not None else 0.0),
        np.int32(1 if max_vehicles is not None else 0),
        np.int32(max_vehicles if max_vehicles is not None else 0),
        np.int32(cap_c),
        np.int32(1 if use_shared else 0),
        ws_f64,
        ws_i64,
        ws_i32,
        d_out,
      ),
      shared_mem=shared_mem,
    )
    self.kernel_launches += 1

    if out_on_gpu:
      return restore_routes_shape(d_out, was_1d)

    out_host = d_out.get()
    self.d2h_bytes += int(out_host.nbytes)
    return restore_routes_shape(out_host, was_1d)


__all__: list[str] = ["DEFAULT_BIN_PACK_MAX_SHARED_DIMENSION", "_BaseGPUBinPacking"]
