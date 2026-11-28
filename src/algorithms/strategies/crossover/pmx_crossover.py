"""
Partially Mapped Crossover (PMX) strategies for routing permutations.

Provides pure CPU NumPy and accelerated GPU CUDA implementations of Partially Mapped Crossover (PMX).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from protocols.algorithm_strategies import CrossoverStrategy
from protocols.backend import require_gpu
from utils.gpu_helpers import load_kernel
from utils.phase_timer import time_strategy
from utils.rng import resolve_rng
from utils.route_helpers import (
    ensure_routes_batch,
    is_gpu_array,
    resolve_route_bounds,
    restore_routes_shape,
    sample_distinct_pairs,
    to_cpu_batch,
)

from ..base import StrategyBase

if TYPE_CHECKING:
    from numpy.random import Generator

    from protocols.problem_context import ProblemContext
    from protocols.types import Route, RouteOrRoutes, SeedLike


class CrossoverCPUPermPartiallyMapped(StrategyBase, CrossoverStrategy):
    """
    Partially Mapped Crossover (PMX) strategy for TSP permutations and depot-bounded tours.

    Preserves a contiguous segment `[c1, c2]` from `parent1`, places unmapped
    entries from `parent2[c1..c2]` via inverse mapping chains, and fills remaining
    positions from `parent2`.
    """

    def __init__(self, seed: SeedLike = None) -> None:
        """Initialize PMX crossover with optional RNG seed."""
        super().__init__(seed)

    @staticmethod
    def _apply_pmx(
        p1: np.ndarray,
        p2: np.ndarray,
        c1: int,
        c2: int,
        is_closed: bool,
    ) -> np.ndarray:
        """Apply Partially Mapped Crossover to a single pair of parent routes with fixed cut points."""
        n = len(p1)
        stream_start = 1 if is_closed else 0
        stream_len = (n - 2) if is_closed else n
        if stream_len <= 1:
            return p1.copy()

        if c1 > c2:
            c1, c2 = c2, c1

        child = np.full(n, -1, dtype=np.int32)
        if is_closed:
            child[0] = p1[0]
            child[-1] = p1[-1]

        child[c1 : c2 + 1] = p1[c1 : c2 + 1]

        max_node = max(n, int(p1.max()) + 1, int(p2.max()) + 1)
        inv_p2 = np.full(max_node, -1, dtype=np.int32)
        inv_p2[p2[stream_start : stream_start + stream_len]] = np.arange(
            stream_start, stream_start + stream_len, dtype=np.int32
        )

        in_seg = np.zeros(max_node, dtype=bool)
        in_seg[p1[c1 : c2 + 1]] = True

        seg_len = c2 - c1 + 1
        for idx in range(c1, c2 + 1):
            val = int(p2[idx])
            if val >= 0 and not in_seg[val]:
                pos = idx
                steps = 0
                while c1 <= pos <= c2 and steps <= seg_len:
                    mapped_val = int(p1[pos])
                    if mapped_val < 0:
                        break
                    next_pos = int(inv_p2[mapped_val])
                    if next_pos < 0:
                        break
                    pos = next_pos
                    steps += 1
                if stream_start <= pos < stream_start + stream_len and (
                    pos < c1 or pos > c2
                ):
                    child[pos] = val

        unmapped = child == -1
        child[unmapped] = p2[unmapped]
        return child

    def crossover(
        self,
        parent1: Route,
        parent2: Route,
        context: ProblemContext | None = None,
    ) -> Route:
        """Recombine two 1D parent routes into a single 1D child route via PMX."""
        p1 = np.asarray(parent1, dtype=np.int32)
        p2 = np.asarray(parent2, dtype=np.int32)
        n = len(p1)
        if n <= 1:
            return p1.copy()

        rng: Generator = resolve_rng(context, self.rng)
        low, high = resolve_route_bounds(n, context=context, route_sample=p1)
        if high - low < 2:
            return p1.copy()
        cuts = sample_distinct_pairs(rng, 1, low, high)[0]
        c1, c2 = int(min(cuts[0], cuts[1])), int(max(cuts[0], cuts[1]))
        return self._apply_pmx(p1, p2, c1, c2, is_closed=(low, high) != (0, n))

    @time_strategy
    def crossover_batch(
        self,
        routes: RouteOrRoutes,
        context: ProblemContext | None = None,
    ) -> RouteOrRoutes:
        """
        Recombine a 2D batch of routes `(N, n_cities)` via PMX.

        Pairs parents in disjoint pairs `(2k, 2k+1)` sharing identical cut points.
        """
        batch_np, was_1d, was_gpu = to_cpu_batch(routes)
        if was_1d or len(batch_np) < 2:
            return restore_routes_shape(batch_np, was_1d, to_gpu=was_gpu)

        num_routes: int = len(batch_np)
        n_cities: int = batch_np.shape[1]
        low, high = resolve_route_bounds(
            n_cities, context=context, route_sample=batch_np[0]
        )
        if n_cities <= 1 or high - low < 2:
            return restore_routes_shape(batch_np.copy(), was_1d, to_gpu=was_gpu)
        is_closed: bool = (low, high) != (0, n_cities)
        offspring: np.ndarray = np.empty_like(batch_np, dtype=np.int32)

        rng: Generator = resolve_rng(context, self.rng)
        num_pairs: int = num_routes // 2
        pairs = sample_distinct_pairs(rng, num_pairs, low, high)

        for k in range(num_pairs):
            i = 2 * k
            c1 = int(min(pairs[k, 0], pairs[k, 1]))
            c2 = int(max(pairs[k, 0], pairs[k, 1]))
            offspring[i] = self._apply_pmx(
                batch_np[i], batch_np[i + 1], c1, c2, is_closed
            )
            offspring[i + 1] = self._apply_pmx(
                batch_np[i + 1], batch_np[i], c1, c2, is_closed
            )

        if num_routes % 2 == 1:
            offspring[-1] = batch_np[-1]

        return restore_routes_shape(offspring, was_1d, to_gpu=was_gpu)


class CrossoverGPUPermPartiallyMapped(StrategyBase, CrossoverStrategy):
    """
    Modular GPU CUDA kernel Partially Mapped Crossover (PMX) strategy (`pmx_crossover.cu`).

    Recombines a 2D batch of parent routes `(N, n_cities)` in parallel on GPU
    using 1 warp per route and warp-private shared memory (`inv_p2` + `in_seg`),
    preserving CuPy device residency for zero-copy full-GPU GA pipelines.
    """

    def __init__(self, seed: SeedLike = None) -> None:
        """Initialize GPU PMX crossover strategy."""
        super().__init__(seed)
        self._kernel: Any = None

    def _get_kernel(self) -> Any:
        """Lazily compile and return the `pmx_crossover_kernel` CUDA kernel."""
        if self._kernel is None:
            self._kernel = load_kernel(
                "pmx_crossover.cu",
                "pmx_crossover_kernel",
                __file__,
            )
        return self._kernel

    def _warmup_impl(self, context: ProblemContext) -> None:
        """Compile the kernel and launch it once on two valid parent permutations.

        `crossover_batch` needs at least two routes to launch, so the warm-up batch is
        `(2, n)`; both rows are `arange`, never `cp.zeros` (`BUG-24`).
        """
        super()._warmup_impl(context)
        self.crossover_batch(self._warmup_batch(context, rows=2), context=context)

    def crossover(
        self,
        parent1: Route,
        parent2: Route,
        context: ProblemContext | None = None,
    ) -> Route:
        """Recombine two 1D parent routes."""
        cpu_pmx = CrossoverCPUPermPartiallyMapped(seed=self.seed)
        return cpu_pmx.crossover(parent1, parent2, context=context)

    @time_strategy
    def crossover_batch(
        self,
        routes: RouteOrRoutes,
        context: ProblemContext | None = None,
    ) -> RouteOrRoutes:
        """
        Recombine a 2D batch of routes `(N, n_cities)` in parallel on GPU.

        Pairs parents in disjoint pairs `(2k, 2k+1)` sharing cut points `(c1, c2)`.
        When `routes` is already a CuPy device array, returns a device array without
        a device-to-host copy.

        Raises:
            GPUUnavailableError: When CuPy is missing or no CUDA device is reachable.
            ValueError: When `5 * n_cities` exceeds the device's `MaxSharedMemoryPerBlock`.
        """
        cp = require_gpu()

        batch, was_1d = ensure_routes_batch(routes)
        num_routes, n_cities = batch.shape
        if was_1d or num_routes < 2:
            return restore_routes_shape(batch.copy(), was_1d)

        input_on_gpu: bool = is_gpu_array(batch)
        low, high = resolve_route_bounds(
            n_cities, context=context, route_sample=batch[0]
        )
        is_closed = (low, high) != (0, n_cities)

        if n_cities <= 1 or high - low < 2:
            return restore_routes_shape(batch.copy(), was_1d)

        if input_on_gpu:
            d_parents = cp.ascontiguousarray(batch, dtype=cp.int32)
        else:
            d_parents = cp.asarray(batch, dtype=cp.int32)
            self.h2d_bytes += int(d_parents.nbytes)

        rng: Generator = resolve_rng(context, self.rng)

        num_pairs: int = num_routes // 2
        pairs = sample_distinct_pairs(rng, num_pairs, low, high)
        ordered_low = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int32, copy=False)
        ordered_high = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int32, copy=False)
        cut1_np = np.zeros(num_routes, dtype=np.int32)
        cut2_np = np.zeros(num_routes, dtype=np.int32)
        cut1_np[: 2 * num_pairs] = np.repeat(ordered_low, 2)
        cut2_np[: 2 * num_pairs] = np.repeat(ordered_high, 2)

        d_cut1 = cp.asarray(cut1_np, dtype=cp.int32)
        d_cut2 = cp.asarray(cut2_np, dtype=cp.int32)
        self.h2d_bytes += int(d_cut1.nbytes + d_cut2.nbytes)

        d_offspring = cp.empty((num_routes, n_cities), dtype=cp.int32)

        dev_shared_mem = int(
            getattr(cp.cuda.Device(), "attributes", {}).get(
                "MaxSharedMemoryPerBlock", 48 * 1024
            )
        )
        bytes_per_warp = (
            5 * n_cities
        )  # 4 bytes/city for inv_p2 + 1 byte/city for in_seg
        if bytes_per_warp > dev_shared_mem:
            raise ValueError(
                f"CrossoverGPUPermPartiallyMapped shared-memory requirement ({bytes_per_warp} B/warp for n={n_cities}) "
                f"exceeds device MaxSharedMemoryPerBlock ({dev_shared_mem} B)."
            )

        threads_per_block = 256
        while (
            threads_per_block > 32
            and (threads_per_block // 32) * bytes_per_warp > dev_shared_mem
        ):
            threads_per_block //= 2
        warps_per_block = threads_per_block // 32
        blocks = (num_routes + warps_per_block - 1) // warps_per_block
        shared_mem_bytes = warps_per_block * bytes_per_warp

        kernel = self._get_kernel()
        kernel(
            (blocks,),
            (threads_per_block,),
            (
                d_parents,
                d_cut1,
                d_cut2,
                d_offspring,
                np.int32(num_routes),
                np.int32(n_cities),
                np.int32(1 if is_closed else 0),
            ),
            shared_mem=shared_mem_bytes,
        )
        self.kernel_launches += 1

        if input_on_gpu:
            return restore_routes_shape(d_offspring, was_1d)
        self.d2h_bytes += int(d_offspring.nbytes)
        return restore_routes_shape(cp.asnumpy(d_offspring), was_1d)


__all__: list[str] = ["CrossoverCPUPermPartiallyMapped", "CrossoverGPUPermPartiallyMapped"]
