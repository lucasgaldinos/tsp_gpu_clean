"""Best-Fit Decreasing (BFD) CPU and GPU fleet bin-packing strategies (`Task 17.30`, `BUG-117`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from algorithms.strategies.base import StrategyBase
from utils.phase_timer import time_strategy

from .cpu_pack_core import pack_batch_cpu
from .gpu_pack_core import DEFAULT_BIN_PACK_MAX_SHARED_DIMENSION, _BaseGPUBinPacking

if TYPE_CHECKING:
  from protocols.problem_context import ProblemContext
  from protocols.types import RouteOrRoutes, SeedLike


class BinPackingCPUBFD(StrategyBase):
  """Vectorized NumPy Best-Fit Decreasing (BFD) fleet bin-packing strategy (`DP17-7`, `DP17-9`)."""

  def __init__(self, seed: SeedLike = None) -> None:
    super().__init__(seed=seed)

  @time_strategy
  def pack_batch(self, routes: RouteOrRoutes, context: ProblemContext) -> RouteOrRoutes:
    """Repack violating ``0``-delimited VRP rows on CPU using BFD + minimum-residual insertion."""
    result, d2h_bytes, h2d_bytes = pack_batch_cpu(routes, context, mode=1)
    self.d2h_bytes += d2h_bytes
    self.h2d_bytes += h2d_bytes
    return result


class BinPackingGPUBFD(_BaseGPUBinPacking):
  """VRAM-native CUDA Best-Fit Decreasing (BFD) fleet bin-packing strategy (`DP17-7`, `DP17-9`)."""

  _mode: ClassVar[int] = 1

  def __init__(
    self,
    max_matrix_dimension: int | None = None,
    max_shared_dimension: int = DEFAULT_BIN_PACK_MAX_SHARED_DIMENSION,
    return_gpu: bool | None = None,
    seed: SeedLike = None,
  ) -> None:
    super().__init__(
      max_matrix_dimension=max_matrix_dimension,
      max_shared_dimension=max_shared_dimension,
      return_gpu=return_gpu,
      seed=seed,
    )


__all__: list[str] = ["BinPackingCPUBFD", "BinPackingGPUBFD"]
