/**
 * GPU Batch Tour Cost Calculator Kernel
 * 
 * Computes tour costs for multiple TSP tours in parallel.
 * Designed to be chained with two_opt_batch_kernel for hybrid MA.
 * 
 * Algorithm:
 *   1. Each block computes cost for one tour
 *   2. Threads cooperatively sum edge costs using parallel reduction
 *   3. Final cost written to global memory
 * 
 * Performance Benefits (Hybrid MA):
 *   - Eliminates need to transfer full tours CPU->GPU (4bn bytes)
 *   - Only transfers small cost array GPU->CPU (b*8 bytes)
 *   - Example: n=1000, b=256 saves 1MB transfer, keeps 2KB
 * 
 * Chaining Pattern:
 *   two_opt_batch_kernel(tours, ...) modifies tours in-place
 *   cost_calculator_kernel(tours, costs, ...) computes costs on GPU
 *   costs.get() transfers only costs to CPU (minimal overhead)
 * 
 * Author: AI Assistant
 * Date: 2025-11-19
 * Purpose: Chapter 4 validation (functional parallelism in hybrid MA)
 *
 * On-the-fly coordinate mode (`DP10-19`): when `distances == nullptr` the kernel
 * computes each edge with `gpu_pairwise_distance` from `coords` + `edge_type_code`,
 * so an instance with `n > 10_000` never needs the dense `(n, n)` matrix in VRAM.
 */

#include "pairwise_device.cuh"

extern "C" __global__
void cost_calculator_kernel(
    const int *tours,        // Flattened: num_tours * n (WITHOUT duplicate depot)
    const int *distances,    // n * n distance matrix (int32), or nullptr for coordinate mode
    const double *coords,    // Row-major coordinates, 2n or 3n doubles (ignored if distances != nullptr)
    long long *costs,        // Output: num_tours costs (int64 for overflow safety)
    int n,                   // Number of nodes per tour (without duplicate depot)
    int num_tours,           // Number of tours in batch
    int edge_type_code       // `GpuEdgeTypeCode` for the coordinate mode
) {
    extern __shared__ long long s_partials[];
    
    int tour_idx = blockIdx.x;  // Which tour this block processes
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Guard: Ensure block is within batch bounds
    if (tour_idx >= num_tours) {
        return;
    }
    
    // Offset to this tour's data
    const int *tour = tours + (tour_idx * n);
    
    // Initialize partial sum for this thread
    long long partial_cost = 0;
    
    // Coalesced block-stride loop (DP11-15, EVAL-A): threads read adjacent tour elements
    for (int i = tid; i < n; i += block_size) {
        int from_node = tour[i];
        int to_node = tour[(i + 1) % n];
        partial_cost += (long long)gpu_pairwise_distance(
            from_node, to_node, distances, coords, n, edge_type_code);
    }
    
    // Store partial sum in shared memory
    s_partials[tid] = partial_cost;
    __syncthreads();
    
    // Parallel reduction in shared memory down to warp size (stride = 32)
    for (int stride = block_size / 2; stride >= 32; stride >>= 1) {
        if (tid < stride && tid + stride < block_size) {
            s_partials[tid] += s_partials[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp 0 finishes the last 32 partial sums via register shuffle reduction
    if (tid < 32) {
        long long warp_cost = (tid < block_size) ? s_partials[tid] : 0LL;
        unsigned int mask = (block_size < 32) ? ((1u << block_size) - 1u) : 0xFFFFFFFFu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_cost += __shfl_down_sync(mask, warp_cost, offset);
        }
        if (tid == 0) {
            costs[tour_idx] = warp_cost;
        }
    }
}
