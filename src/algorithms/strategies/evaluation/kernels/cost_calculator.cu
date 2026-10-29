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
 */

extern "C" __global__
void cost_calculator_kernel(
    const int *tours,       // Flattened: num_tours * n (WITHOUT duplicate depot)
    const int *distances,    // n * n distance matrix (int32, TSPLIB compliant)
    long long *costs,        // Output: num_tours costs (int64 for overflow safety)
    int n,                  // Number of nodes per tour (without duplicate depot)
    int num_tours           // Number of tours in batch
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
    
    // Each thread sums a subset of edges
    // Edge i connects tour[i] to tour[(i+1) % n]
    for (int i = tid; i < n; i += block_size) {
        int from_node = tour[i];
        int to_node = tour[(i + 1) % n];  // Wrap around for last edge
        
        // Accumulate edge cost
        partial_cost += distances[from_node * n + to_node];
    }
    
    // Store partial sum in shared memory
    s_partials[tid] = partial_cost;
    __syncthreads();
    
    // Parallel reduction to sum all partial costs
    // Standard log-step reduction pattern
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < block_size) {
            s_partials[tid] += s_partials[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes final cost to global memory
    if (tid == 0) {
        costs[tour_idx] = s_partials[0];
    }
}
